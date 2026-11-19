"""Pipeline Workspace run logic: no Tk.

Everything the workspace does that is not drawing a window: the tree's item
ids, trial names and day numbers, the drop payloads built from the gate
editor, the run settings and units, preparing and computing a clustering run,
the job child's entry point, reading a run's outputs back, and
``WorkspaceModel`` (the groups, items and their comp / FMO choices). The
window in :mod:`openflo.workspace` draws over these. Keeping them here lets
the library, the job child and the tests use them without loading Tk.
"""

from __future__ import annotations

import copy
import logging
import os
import re

from .errors import note as note_handled
from .fcs_containers import item_set_number
from .gate_lineage import root_first
from .group_palettes import DEFAULT_GROUP_PALETTE, n_group_colors
from .paths import is_fcs_file
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS

# The reload warnings go to the logger FlowSample's own messages use.
_PIPELINE_LOG = logging.getLogger('openflo.pipeline')

# Marker prefixing the workspace's own progress lines on a job's stdout, so the
# parent can show them and ignore the backend's chatter (phenograph, warnings).
PROGRESS_MARKER = '\x01WSPROG\x01'

# ── iid encoding for editor-originated nodes ───────────────────────────────
#   Sample: 'S:<name>'   Gate: 'G:<name>/<gid>'   Comp: 'C:<name>'


def sample_iid(name: str) -> str:
    """Tree item id for a sample row."""
    return f'S:{name}'


def gate_iid(name: str, gid: str) -> str:
    """Tree item id for gate `gid` of sample `name`."""
    return f'G:{name}/{gid}'


def comp_iid(name: str) -> str:
    """Tree item id for a sample's compensation row."""
    return f'C:{name}'


def parse_iid(iid: str):
    """Return ('sample', name) | ('gate', name, gid) | ('comp', name) | None."""
    if iid.startswith('S:'):
        return ('sample', iid[2:])
    if iid.startswith('C:'):
        return ('comp', iid[2:])
    if iid.startswith('G:'):
        rest = iid[2:]
        if '/' not in rest:
            return None
        name, gid = rest.rsplit('/', 1)
        return ('gate', name, gid)
    return None


# A "Day N" collection-day token in a folder name, e.g. "2024-01-15 Day 3 Check",
# "day 0 raw", "day12 batch-a", "set2 day 15". The number is captured;
# the leading \b stops false hits inside words ("Monday", "Tuesday").
_DAY_TOKEN_RE = re.compile(r'\bday\s*[-_]?\s*(\d+)\b', re.IGNORECASE)


# The folder acquisition software names an experiment by default ("Blank
# Experiment with Sample Tube"): too generic to name a group, so the folder
# above it does.
_GENERIC_HOST_RE = re.compile(r'^blank experiment\b', re.IGNORECASE)


def derive_trial_name(path) -> str:
    """Trial / group name for an FCS ``path``.

    Time-course flow experiments are organised by collection day, but the
    'Day N' token sits at an inconsistent depth on disk — sometimes the
    immediate experiment folder (``…/2024-01-15 Day 3 Check/x.fcs``), sometimes a
    grandparent (``…/experiment-a/day 0 raw/Blank Experiment…/x.fcs``).
    So we scan the file's ancestor folders from the **nearest upward** and group
    by the first 'Day N' we find, normalised to ``"Day N"``. This keeps every
    sample from one collection day together (specimens + compensation controls
    alike) no matter how the surrounding folders are named — and keeps a whole
    drop from collapsing under one shared grandparent
    (e.g. everything sharing one study-root folder name).

    With no day token anywhere above the file, the group is the folder the
    file is in, or the one above it when that is a generic 'Blank Experiment
    with…' host; a hidden folder ('.openflo') never names a group. It used
    to be the grandparent whatever the parent was, so ``synth/pbmc/`` and
    ``synth/compensation/`` both showed as "synth", and the example data
    (``~/.openflo/example_data/``) as ".openflo".
    """
    if not path:
        return 'Trial'
    p = os.path.abspath(str(path))

    # Walk ancestor directories from the file's own folder upward; the nearest
    # ancestor carrying a 'Day N' token wins (most specific to this sample).
    cur, prev = os.path.dirname(p), None
    while cur and cur != prev:
        m = _DAY_TOKEN_RE.search(os.path.basename(cur))
        if m:
            return f'Day {int(m.group(1))}'
        prev, cur = cur, os.path.dirname(cur)

    # Fallback: the file's own folder, or the one above a generic host.
    host = os.path.dirname(p)
    cands = [os.path.basename(host), os.path.basename(os.path.dirname(host))]
    if _GENERIC_HOST_RE.match(cands[0]):
        cands.reverse()
    for cand in cands:
        if cand and not cand.startswith('.'):
            return cand
    return 'Trial'


def natural_sort_key(name):
    """Sort key that orders names as people read them: 'ctrl_d2' before
    'ctrl_d10', case aside. Digit runs compare as numbers."""
    parts = re.split(r'(\d+)', str(name))
    return [(0, int(p), '') if p.isdigit() else (1, 0, p.casefold())
            for p in parts if p]


# Names of compensation controls. A "word" is a run of letters, so digits,
# '_', '-', ' ' and brackets all separate words: 'Comp1_FL4', 'Comp008' and
# 'D3Comp' hold the word 'comp', while 'Compound' and 'Compensated' do not.
_COMP_NAME_RE = re.compile(
    r'unstain'                         # anywhere, also 'lateunstained_012'
    r'|(?<![a-z])(?:'
    # 'Comp_008', 'Day3_Comps', 'CD3_FL4_comp', comp beads 'CompBeads_FL5',
    # and the bead brands written as one word: 'UltraComp_FL5', 'OneComp',
    # 'VersaComp_FL4', 'AbCComp', 'ArCComp'. An open prefix ('[a-z]*comp')
    # also took 'recomp', 'precomp', 'Decomp_1', 'Autocomp' and 'TComp_1'.
    # 'comp' after 'no' / 'non' ('Non-comp', 'No-comp') is a sample.
    r'(?<!no[\s_-])(?<!non[\s_-])'
    r'(?:ultra|one|versa|abc|arc)?comp(?:s|beads?)?'
    r'|compensation(?:[\s_-]*controls?)?'   # 'Compensation Controls_FL2'
    r'|stained[\s_-]*controls?'        # 'FL4 Stained Control'
    r'|single[\s_-]*stain(?:ed|s)?'
    r'|no[\s_-]*stain(?:ed|ing|s)?'    # 'No stain', 'No staining control'
    r')(?![a-z])',
    re.IGNORECASE)


def is_comp_sample(name) -> bool:
    """True if a sample name looks like a compensation control: FlowJo / BD
    forms ('Compensation Controls_…', '… Stained Control', 'Unstained
    Control'), 'unstained' anywhere, 'comp' / 'comps' / 'compensation' as a
    whole word (letters only: 'Comp008' counts), comp beads ('CompBeads',
    or a word ending in 'comp': 'UltraComp', 'VersaComp'), or a single-stain
    or no-stain tube.

    The one name rule for both the tree's Comps subgroup and the samples the
    statistics leave out, so the two cannot disagree; the user's Comps/Samples
    drag overrides it (editor ``_is_comp``). 'control' or 'stained' alone is
    not enough: 'Control_1', 'Vehicle control', 'Isotype control' and 'CFSE
    stained' are real samples, and so is 'Compound_A'. The earlier substring
    test ('comp', 'control', 'stain') took all of them."""
    if not name:
        return False
    return _COMP_NAME_RE.search(str(name)) is not None


def trial_day_number(trial):
    """Integer ``N`` if ``trial`` is a normalised ``"Day N"`` group (as minted
    by :func:`derive_trial_name`), else ``None``. Lets the tree order day
    groups chronologically (Day 0 < Day 3 < … < Day 15) instead of by the
    arbitrary folder-sort order they happened to load in."""
    if not trial:
        return None
    m = re.fullmatch(r'Day (\d+)', str(trial))
    return int(m.group(1)) if m else None


# ── Pure context / payload helpers (no Tk) ─────────────────────────────────


def extract_one_context(editor, name: str) -> dict | None:
    """Snapshot a single loaded sample from the gate ``editor`` (trial, path,
    color, comp matrix + channels, labels, deep-copied gate tree). ``None`` if
    not loaded. Event data is never copied."""
    samples = getattr(editor, '_samples', {}) or {}
    if name not in samples:
        return None
    s = samples[name]
    gates = (getattr(editor, '_sample_gates', {}) or {}).get(name, {}) or {}
    gate_order = (getattr(editor, '_sample_gate_order', {}) or {}).get(name, []) or []
    # A sample restored from a processed CSV keeps its raw FCS in source_fcs.
    path = getattr(s, 'source_fcs', None) or getattr(s, 'path', None)
    trial = (getattr(editor, '_sample_trial', {}) or {}).get(name) or derive_trial_name(path)
    return {
        'name':           name,
        'trial':          trial,
        'path':           path,
        'color':          (getattr(editor, '_sample_colors', {}) or {}).get(name, '#000000'),
        'comp_matrix':    getattr(s, 'comp_matrix', None),
        'comp_channels':  list(getattr(s, 'comp_channels', None) or []),
        'qc_applied':     bool(getattr(s, 'qc_applied', False)),
        # The data set of a file holding several: a reload reads that one.
        'set_number':     getattr(s, 'set_number', None),
        # A BD Accuri well's member name: a reload finds the same well even
        # after the file gained wells (fcs_containers.item_set_number).
        'container_member': _container_member(s),
        'channel_labels': dict(getattr(editor, '_channel_labels', {}) or {}),
        'gates':          copy.deepcopy(dict(gates)),
        'gate_order':     list(gate_order),
        # The stored scale of each channel, so a gate path names intensities
        # as the editor's gate list does (build_drop_payload).
        'data_transforms': _display_transforms(editor, name, s),
    }


def _display_transforms(editor, name, sample):
    """The ``{channel: transform_spec}`` the editor says sample `name`'s
    gate bounds with (its ``_display_transforms``), else the sample's own
    record; None when there is neither (bounds then stay as stored)."""
    fn = getattr(editor, '_display_transforms', None)
    if callable(fn):
        return copy.deepcopy(fn(name))
    rec = getattr(sample, 'data_transforms', None)
    return None if rec is None else copy.deepcopy(dict(rec))


def extract_editor_context(editor) -> list[dict]:
    """Snapshot every loaded sample in the gate ``editor`` (in display order)."""
    out: list[dict] = []
    for name in (getattr(editor, '_sample_order', []) or []):
        ctx = extract_one_context(editor, name)
        if ctx is not None:
            out.append(ctx)
    return out


def gate_is_leaf(gates: dict, gid: str) -> bool:
    """True when no gate in `gates` has `gid` as its parent."""
    return not any(g.get('parent_id') == gid for g in gates.values())


def gate_chain(gates: dict, gid: str) -> list[str]:
    """Root→leaf gate-id list (a leaf is constricted by all ancestor gates)."""
    return root_first(gates, gid)


def gate_path(gates: dict, gid: str, transforms: dict | None = None) -> str:
    """The gate chain to `gid` as the Workspace tree shows it, each gate
    named as ``gating.population_path`` names it: its label or name, and its
    bounds only when it has neither. `transforms` (the sample's
    ``data_transforms``) has those bounds said as intensities, as the
    editor's gate list says them (gate_text.describe_gate). Display only:
    output files are named by gate id (run_label)."""
    from .gating import population_names
    return ' / '.join(population_names(gates, gid, transforms=transforms))


def _container_member(s) -> str:
    """The archive name of the BD Accuri well sample `s` was read from,
    else '' (session_format.container_member)."""
    from .session_format import container_member
    number = getattr(s, 'set_number', None)
    return container_member(s, int(number)) if number else ''


def build_drop_payload(ctx: dict, gid: str | None) -> dict:
    """Describe a pipeline item dropped from the editor (whole sample if
    ``gid`` is None, else the population at ``gid``). Carries trial, comp matrix
    + channels, labels, the data set number and the cumulative gate chain."""
    gates = ctx.get('gates', {}) or {}
    chain = gate_chain(gates, gid) if gid else []
    return {
        'sample':         ctx['name'],
        'trial':          ctx.get('trial', 'Trial'),
        'path':           ctx.get('path'),
        'color':          ctx.get('color', '#000000'),
        'gate_id':        gid,
        'gate_path':      (gate_path(gates, gid,
                                     transforms=ctx.get('data_transforms'))
                           if gid else None),
        'is_leaf':        gate_is_leaf(gates, gid) if gid else None,
        'gate_chain':     chain,
        'gates':          {g: gates[g] for g in chain},
        'comp_matrix':    ctx.get('comp_matrix'),
        'comp_channels':  list(ctx.get('comp_channels') or []),
        'qc_applied':     bool(ctx.get('qc_applied', True)),
        'channel_labels': dict(ctx.get('channel_labels') or {}),
        # The data set of a file holding several (fcs_data_sets): a reload
        # of the sample once it is closed reads that one again.
        'set_number':     ctx.get('set_number'),
        'container_member': ctx.get('container_member') or '',
    }


def comp_summary(payload_or_ctx: dict) -> str:
    """Short label for a raw metadata comp matrix: '12×12' or 'none'."""
    m = payload_or_ctx.get('comp_matrix')
    if m is None:
        return 'none'
    shape = getattr(m, 'shape', None)
    if shape and len(shape) == 2:
        return f'{shape[0]}×{shape[1]}'
    ch = payload_or_ctx.get('comp_channels') or []
    return f'{len(ch)} ch' if ch else 'present'


# ── Run engine (Tk-free, unit-testable) ───────────────────────────────────
#
# M2: run Phenograph + UMAP + TriMap on each workspace item, independently.
# Reuses the editor's already-compensated+transformed FlowSample (a shallow
# copy with a deep-copied .data, so clustering/embedding columns and gate
# filtering never mutate the editor's sample); falls back to loading from the
# FCS path when the sample isn't live. Every primitive is the trustworthy
# backend (pipeline.FlowSample) — NOT the discontinued in-process GUI engine.
# Each item is fully isolated: any failure becomes a per-item error result,
# never a crash or a retry loop.

DEFAULT_RUN_CFG = {
    'method': 'phenograph',  # 'phenograph' | 'leiden' | 'flowsom'
    'resolution': 1.0,       # Leiden / PhenoGraph resolution (higher → more clusters)
    'n_metaclusters': FLOWSOM_DEFAULT_METACLUSTERS,  # FlowSOM metaclusters
    'k': 30,              # Phenograph / Leiden / UMAP nearest-neighbours
    'max_events': 5000,   # subsample cap PER RUN UNIT (small = fast); 0 = all
    'seed': 42,
    # Prune the shared-nearest-neighbour graph to kNN pairs (what PhenoGraph
    # and Seurat do) instead of linking every pair sharing a neighbour. ~6-14x
    # fewer edges and 6-9x faster Leiden, and it recovers the planted cluster
    # count where the dense graph merges populations. OFF by default because it
    # changes every existing clustering result, and on heavily overlapping
    # populations the two disagree substantially. See cluster_graph.snn_jaccard_graph.
    'fast_graph': False,
    # PhenoGraph's Louvain is non-deterministic per process, so the same data
    # re-run gave different labels (ARI 0.76 on overlapping populations). Its
    # binaries seed with srand(time(NULL)+getpid()) -- the PID half is assigned
    # by the OS, so no clock injection can pin it (20 runs inside one second
    # gave 20 distinct partitions). Default ON since
    # 2.4.9: cluster() uses the seeded Leiden backend, so a re-run is
    # bit-identical. Turning it OFF restores Louvain and the GPU path.
    'reproducible': True,
    'channels': [],       # explicit marker selection ([] = auto: proper markers)
    # Embeddings to compute per run unit (each writes <PREFIX>1/2 + a PNG).
    'umap': True,
    'tsne': False,
    'phate': False,
    'trimap': True,
    'pacmap': False,
    # Run unit: a GROUP's samples are concatenated into ONE co-embedded UMAP
    # (events tagged by source sample). 'concatenate' merges ALL groups into a
    # single UMAP so groups can be compared in one embedding (FlowJo-style);
    # otherwise each group (and each loose item) is its own UMAP.
    'concatenate': False,
    # The per-event table is written binary (<label>_events.npz, see
    # write_run_events); this also writes it as <label>_events.csv, for
    # opening in Excel. Off by default: the CSV costs ~10 s a sample where
    # the binary file takes ~0.2 s. A workspace or recipe saved without the
    # key merges onto these defaults, so it reads as off.
    'events_csv': False,
}


def default_run_cfg() -> dict:
    """Return a fresh copy of the default run settings."""
    return dict(DEFAULT_RUN_CFG)


# Embeddings that need the optional [embed] extra: config key -> (module,
# name shown to the user). UMAP and t-SNE ship with the core install.
EXTRA_EMBEDDINGS = {'phate': ('phate', 'PHATE'), 'trimap': ('trimap', 'TriMap'),
                    'pacmap': ('pacmap', 'PaCMAP')}


def embedding_unavailable(key: str, load: bool = False) -> str | None:
    """Why embedding ``key`` can't run here, or None when it can. An [embed]
    backend that is missing is skipped by FlowSample with only a log line, so
    the Pipeline asks first rather than reporting a run that made no output.
    ``load`` imports the backend (catching one installed without its own
    dependencies) instead of only looking for it, which keeps the panel quick
    to open."""
    entry = EXTRA_EMBEDDINGS.get(key)
    if entry is None:
        return None
    mod, title = entry
    # The shared check (also behind Help → Environment and Compare
    # embeddings). Without ``load`` it gives the import's answer once
    # anything has tried the import (the panel starts that in the
    # background, _check_embed_engines), and presence until then.
    from .capabilities import NOT_INSTALLED, engine_unavailable
    why = engine_unavailable(mod, load=load)
    if why is None:
        return None
    if why == NOT_INSTALLED:
        return f"{title} skipped: not installed (pip install openflo[embed])"
    return f"{title} skipped: installed but {why}"


def unchosen_embeddings(cfg: dict, chosen) -> list[str]:
    """The [embed] engines ``cfg`` runs only because they are on by default
    (``DEFAULT_RUN_CFG``): not in ``chosen``, the keys whose box the user
    ticked or unticked. A run skips one of these that cannot run with a
    progress note, not a warning on the run's final line (TriMap is on by
    default, and an install that does not load warned every run of a user
    who never chose it)."""
    return [k for k in EXTRA_EMBEDDINGS
            if cfg.get(k) and DEFAULT_RUN_CFG.get(k) and k not in chosen]


def embedding_available(key: str) -> bool:
    """Whether the panel offers embedding ``key`` (see embedding_unavailable)."""
    return embedding_unavailable(key) is None


def run_label(item: dict) -> str:
    """Filesystem-safe label for an item's output files."""
    base = f"{item.get('trial', 'Trial')}_{item.get('sample', 'sample')}"
    if item.get('gate_id'):
        base += f"_{item['gate_id']}"
    return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in base) or 'item'


def proper_run_channels(sample) -> list[str]:
    """The PROPER marker channels for clustering/embedding: one column per
    marker out of the sample's fluor channels (area, else height, never
    width), as ``channel_roles.default_cluster_channels`` picks them — so UMAP and
    the clustering do not count each marker's collinear ``-A``/``-H``/``-W``
    versions two or three times.

    Scatter (FSC/SSC), Time, and analysis columns are already excluded upstream
    by ``FlowSample.fluor_channels``."""
    from .channel_roles import default_cluster_channels
    return default_cluster_channels(getattr(sample, 'fluor_channels', None))


def live_sample(editor, item: dict):
    """The editor's loaded sample a run of ``item`` starts from, or None.

    Found by the row's sample name. A row made from a run template
    (``template_slot``) never uses one: it runs from its own file with the
    template's gates, scales and compensation, so one template gives one
    answer whatever the editor holds (an editor sample of that name may be
    another day's tube, or carry the editor's own compensation)."""
    if editor is None or item.get('template_slot'):
        return None
    return (getattr(editor, '_samples', None) or {}).get(item.get('sample'))


def _reload_comp(s, item: dict, name, warnings: list | None) -> str:
    """Compensate reloaded sample `s` as row `item` asks when no override
    applies: a run template's row with its file's own $SPILLOVER (each
    day's file its own, as the editor loads it: a template carries no
    day's matrix), any other with the matrix the row saved. Returns words
    for the run's note."""
    if item.get('template_slot'):
        s.auto_compensate()
        if s.is_compensated():
            return ', compensated with its own $SPILLOVER'
        if warnings is not None:
            warnings.append(f"'{name}': its file has no usable $SPILLOVER, "
                            "so it ran uncompensated")
        return ', uncompensated (no $SPILLOVER in its file)'
    m = item.get('comp_matrix')
    ch = list(item.get('comp_channels') or [])
    if m is not None and ch:
        try:
            s.manual_compensate(m, ch)
        except Exception as e:  # noqa: BLE001 - best-effort match to editor
            _PIPELINE_LOG.warning("  [workspace] comp on reload failed: %s", e)
    return ''


def member_channels(editor, item: dict) -> list[str]:
    """The channels a bead matrix for ``item`` covers: the sample's own
    compensated channels, else its fluorescence channels (live sample, else
    the item's saved list, else read from its FCS file)."""
    live = live_sample(editor, item)
    if live is not None:
        return list(getattr(live, 'comp_channels', None)
                    or getattr(live, 'fluor_channels', None) or [])
    if item.get('comp_channels'):
        return list(item['comp_channels'])
    path = item.get('path')
    if path and os.path.exists(path):
        from .flow_sample import FlowSample
        return list(FlowSample(path, set_number=item_set_number(item))
                    .fluor_channels or [])
    return []


def bead_comp(comp: dict, channels: list[str]) -> dict:
    """Build the matrix a row's or group's comp beads give over
    ``channels`` (off the Tk thread: it reads every bead file). Each file
    is paired with its channel by name, as the compensation editor's
    Auto-fill pairs them (``comp_controls.ControlMatcher``), and the matrix
    is ``optimize_compensation``'s. Returns ``comp`` as a matrix choice
    (``kind`` 'matrix', ``bead_files``, ``warnings``), or with ``error``
    set when the beads cannot give one: a missing file, a file that fits two
    channels equally, or none that names a channel. A channel with no bead
    file keeps an identity row, and the warning names it."""
    from .comp_controls import ControlMatcher
    from .compensation_optimizer import optimize_compensation
    files = list(comp.get('files') or [])
    where = _where(comp)
    warns: list[str] = []
    out: dict = {**comp, 'bead_files': files, 'warnings': warns}
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        out['error'] = (f"the {where} comp beads cannot be read: "
                        f"{', '.join(missing[:3])} not found")
        return out
    if not channels:
        out['error'] = f"no channels to build the {where} bead matrix over"
        return out
    names = [os.path.basename(f) for f in files]
    paired, ambiguous, missed = ControlMatcher._match_controls(channels, names)
    if ambiguous:
        out['error'] = (f"the {where} comp beads do not pair with the "
                        f"channels one to one: {'; '.join(ambiguous[:3])}")
        return out
    if not paired:
        out['error'] = (f"none of the {where} comp bead files names one of "
                        f"the channels ({', '.join(channels)})")
        return out
    paths = {ch: files[names.index(fn)] for ch, fn in paired.items()}
    chans, matrix = optimize_compensation(list(channels), paths)
    import numpy as np
    m = np.asarray(matrix, dtype=float)
    flat = [ch for ch in paths
            if np.allclose(m[chans.index(ch)], np.eye(len(chans))[chans.index(ch)])]
    if missed:
        warns.append(f"no {where} comp bead file for "
                               f"{', '.join(missed)} (left uncorrected)")
    if flat:
        warns.append(
            f"no spillover could be measured from the {where} comp beads for "
            f"{', '.join(flat)} (see the log)")
    out.update(kind='matrix', matrix=m.tolist(), channels=list(chans))
    return out


def build_bead_comps(editor, members: list[dict]) -> None:
    """Turn every member's bead choice into the matrix it gives, in place
    (``member['comp']``). One matrix per bead set and channel list, so
    members sharing a group's beads compute it once, and each acquisition
    run's own beads give its own matrix. Reads files: call it off the Tk
    thread."""
    cache: dict = {}
    for m in members:
        comp = m.get('comp')
        if not comp or comp.get('kind') != 'beads':
            continue
        try:
            chans = m.get('channels') or member_channels(editor, m['item'])
            key = (tuple(comp.get('files') or []), tuple(chans), comp.get('source'))
            if key not in cache:
                cache[key] = bead_comp(comp, chans)
            m['comp'] = dict(cache[key])
        except Exception as e:  # noqa: BLE001 - any failure stops only this member, with the reason
            m['comp'] = dict(comp, error=f"the {_where(comp)} comp beads "
                             f"failed: {type(e).__name__}: {e}")


FMO_PERCENTILE = 99.0   # the CLI's and the FMO dialog's default


def _fmo_source(m: dict) -> str:
    return (m.get('fmo') or {}).get('source') or 'group'


def load_fmo_control(path: str, comp: dict | None, qc: bool):
    """An FMO file read as its member's events are: QC when the member had
    it, compensated with the MEMBER's matrix (not the FMO's own $SPILL: the
    cutoff must sit in the same compensation as the events it splits), then
    transformed. Reads the file: call it off the Tk thread."""
    from .flow_sample import FlowSample
    fs = FlowSample(path)
    if qc:
        try:
            fs.run_qc()
        except Exception as e:  # noqa: BLE001 - best-effort, as the member's own reload
            _PIPELINE_LOG.warning("  [workspace] QC on FMO %s failed: %s", path, e)
    if (comp and comp.get('kind') == 'matrix' and comp.get('matrix') is not None
            and comp.get('channels')):
        fs.manual_compensate(comp['matrix'], list(comp['channels']))
    fs.apply_transform()
    return fs


def _comp_key(comp: dict | None) -> tuple | None:
    """What an FMO read's compensation depends on, exactly: the channels
    and every matrix value (``repr`` elides the middle of large arrays)."""
    import numpy as np
    mat = (comp or {}).get('matrix')
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=float)
    return (tuple((comp or {}).get('channels') or ()), arr.shape, arr.tobytes())


def build_fmo_controls(editor, members: list[dict]) -> None:
    """Pair each member's FMO files with its channels (by name, as comp
    beads are) and read them (``member['fmo_controls']``), once per file
    and compensation. An FMO file that names no channel, or two equally,
    is skipped with a note: clustering does not depend on it. Reads files:
    call it off the Tk thread, after :func:`build_bead_comps`."""
    from .comp_controls import ControlMatcher
    cache: dict = {}
    for m in members:
        fmo = m.get('fmo') or {}
        files = list(fmo.get('files') or [])
        comp = m.get('comp')
        if not files or (comp or {}).get('error'):
            continue
        chans = m.get('channels') or member_channels(editor, m['item'])
        names = [os.path.basename(f) for f in files]
        paired, ambiguous, _missed = ControlMatcher._match_controls(chans, names)
        skipped = list(dict.fromkeys(f"FMO {a}" for a in ambiguous))
        skipped += [f"FMO {n} names none of the channels" for n in names
                    if n not in paired.values()
                    and not any(n in a for a in ambiguous)]
        controls = {}
        qc = bool(m['item'].get('qc_applied', True))
        mkey = _comp_key(comp)
        for ch, fn in paired.items():
            path = files[names.index(fn)]
            if not os.path.exists(path):
                skipped.append(f"FMO {fn} not found")
                continue
            key = (path, qc, mkey)
            try:
                if key not in cache:
                    cache[key] = load_fmo_control(path, comp, qc)
            except Exception as e:  # noqa: BLE001 - an FMO that cannot be read is skipped with its reason
                skipped.append(f"FMO {fn} could not be read ({type(e).__name__}: {e})")
                continue
            controls[ch] = (path, cache[key])
        m['fmo_controls'] = {'controls': controls, 'skipped': skipped,
                             'source': _fmo_source(m)}


def build_run_controls(editor, members: list[dict]) -> None:
    """Everything a run reads from control files before its queue starts:
    bead matrices, then the FMO controls compensated with them."""
    build_bead_comps(editor, members)
    build_fmo_controls(editor, members)


def apply_fmo_controls(s, m: dict, pct: float, warnings: list) -> list[dict]:
    """Add ``<channel>_pos`` (1.0 / 0.0, NaN where the value is unknown) to
    the member's events ``s`` for each FMO control it has: positive is at or
    above the ``pct`` percentile of the FMO's linear values INSIDE the
    member's population (its gate chain read on the FMO in the member's
    coordinates; a cutoff over every FMO event lets events the population
    excludes set it). Annotates only: no event is removed. Returns the rows
    of the run's thresholds table."""
    import numpy as np

    from .channel_transforms import linear_values
    from .fmo_gater import in_stained_coordinates
    from .gate_eval import cumulative_gate_mask
    from .gating import gate_channels
    from .gating_helpers import fmo_threshold
    ctl = m.get('fmo_controls') or {}
    name = m.get('sample') or m['item'].get('sample')
    for w in ctl.get('skipped') or []:
        warnings.append(f"{name}: {w}, skipped")
    item = m['item']
    gid, gates = item.get('gate_id'), item.get('gates') or {}
    gate_chs = set()
    for g in gates.values():
        gate_chs |= gate_channels(g)
    rows = []
    for ch, (path, fs) in sorted((ctl.get('controls') or {}).items()):
        fname = os.path.basename(path)
        if ch not in s.data.columns:
            continue
        inside = np.ones(len(fs.data), dtype=bool)
        if gid and gates:
            frame, bad = in_stained_coordinates(fs, s)
            if set(bad) & gate_chs:
                warnings.append(f"{name}: FMO {fname} cannot be read through "
                                "the population's gates (unknown scale), skipped")
                continue
            inside = np.asarray(cumulative_gate_mask(gates, gid, frame), dtype=bool)
        if not inside.any():
            warnings.append(f"{name}: FMO {fname} has no events inside the "
                            "population, skipped")
            continue
        try:
            cut = fmo_threshold(linear_values(
                fs, ch, fs.data[ch].to_numpy(float)[inside]), percentile=pct)
            lin = np.asarray(linear_values(s, ch), dtype=float)
        except Exception as e:  # noqa: BLE001 - one channel's cutoff failing skips that channel, named
            warnings.append(f"{name}: FMO {fname} for {ch} failed "
                            f"({type(e).__name__}: {e}), skipped")
            continue
        flag = np.where(np.isfinite(lin), (lin >= cut).astype(float), np.nan)
        s.data[f'{ch}_pos'] = flag
        rows.append({'member': name, 'channel': ch, 'fmo_file': fname,
                     'percentile': pct, 'threshold_linear': cut,
                     'fmo_events_inside': int(inside.sum()),
                     'source': ctl.get('source', 'group'),
                     'pct_pos': (float(np.nanmean(flag) * 100)
                                 if np.isfinite(flag).any() else float('nan'))})
    return rows


GATE_DRIFT_WARN = 0.05   # a gate keeping >5% more/fewer events is named


def override_matrix(comp: dict | None) -> dict | None:
    """``comp`` (a :func:`resolve_comp` result) when it is a row or group
    MATRIX the run must apply instead of the editor's; else None (the
    editor's or the saved matrix, which a run already uses as it is)."""
    if (comp and comp.get('kind') == 'matrix'
            and comp.get('source') in ('row', 'group')
            and comp.get('matrix') is not None):
        return comp
    return None


def comp_words(comp: dict | None) -> str:
    """The compensation a member's events carry, in words, for its line in
    the run's members file: from the same ``comp`` the run applied."""
    if not comp or comp.get('source') == 'none':
        return 'no compensation'
    if comp.get('source') == 'editor':
        return f"the editor's {comp.get('label', 'matrix').split(' ')[0]} matrix"
    if comp.get('source') == 'file':
        return "the file's own $SPILLOVER"
    if comp.get('source') == 'saved':
        return f"the {comp.get('label', 'matrix')} matrix saved when it was added"
    return f"the {_where(comp)} {comp.get('kind', 'matrix')} ({comp.get('label', '')})"


def _override_words(comp: dict) -> str:
    """'the row matrix', or 'the group matrix built from 2 comp bead files'
    when the matrix came from that row's or group's beads."""
    beads = comp.get('bead_files') or []
    return (f"the {_where(comp)} matrix"
            + (f" built from {len(beads)} comp bead files" if beads else ''))


def _where(comp: dict | None) -> str:
    return 'row' if (comp or {}).get('source') == 'row' else 'group'


def _recompensate_live(s, comp: dict, name: str) -> str:
    """Put the row/group matrix ``comp`` on the run's copy ``s`` of a live
    sample, keeping its events, transforms and columns (the comp editor's
    Apply does the same). Raises RuntimeError naming why it cannot."""
    import numpy as np

    from .exceptions import CompensationError
    path = str(getattr(s, 'path', '') or '')
    if not s.comp_channels and not is_fcs_file(path):
        # Built from a data table with no compensation record: its values
        # may be compensated already, and un-mixing them again compounds it.
        raise RuntimeError(
            f"cannot apply the {_where(comp)} matrix to '{name}': it was "
            f"built from {os.path.basename(path) or 'a data table'}, which "
            "does not record the compensation already in its values")
    try:
        s.recompensate(comp['matrix'], list(comp.get('channels') or []),
                       source=f'workspace {_where(comp)} matrix')
    except (ValueError, CompensationError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"cannot apply the {_where(comp)} matrix to "
                           f"'{name}': {exc}") from exc
    return f', compensated with {_override_words(comp)}'


def _put_on_gate_scales(editor, s, scales: dict | None) -> None:
    """Put reloaded sample `s` on the scales its row's gates were drawn
    on: a run template's record (`scales`), else the editor's."""
    from .sample_prep import restore_sample_transforms
    if scales is not None:
        restore_sample_transforms(s, scales)
        return
    conform = getattr(editor, '_conform_sample_transforms', None)
    if conform is not None:
        conform(s)


def resolve_run_sample(editor, item: dict, comp: dict | None = None,
                       warnings: list | None = None,
                       stats_out: list | None = None,
                       stat_forms: dict | None = None):
    """Return ``(FlowSample, note)`` ready to cluster for a workspace ``item``.

    Prefers the editor's live, already comp+transformed sample (shallow copy +
    deep-copied ``.data`` so we never mutate editor state); else loads the FCS
    from ``path`` and best-effort applies the item's metadata comp matrix +
    logicle transform. If the item is a gated population, ``.data`` is filtered
    to that population's cumulative gate mask. Raises only on an unrecoverable
    load (no live sample AND no usable path).

    ``comp`` is the item's :func:`resolve_comp` choice, taken when Run was
    pressed. A row or group matrix replaces the editor's (live) or the
    saved one (reload); one that cannot be applied raises RuntimeError
    rather than run on another compensation. Gates were drawn on the
    editor's compensation, so a gated live member's event count under both
    is in the note, and a change of more than 5% goes into ``warnings``.

    ``stats_out``, when given, gets the item's population row
    (:func:`population_table_rows`), taken on every event of the sample
    before it is gated: the run's subsample cap never touches it."""
    import copy as _copy

    from .flow_sample import FlowSample
    from .gate_eval import cumulative_gate_mask

    name = item.get('sample')
    live = live_sample(editor, item)
    if comp and comp.get('error'):
        raise RuntimeError(f"'{name}': {comp['error']}")
    if comp and comp.get('kind') == 'beads':
        # Never run on another matrix because the beads were not turned
        # into one (build_bead_comps did not run, or stopped).
        raise RuntimeError(f"'{name}': the {_where(comp)} comp beads were not read")
    ov = override_matrix(comp)
    editor_data = None
    # A run template's row (never live) keeps the scales its gates were
    # drawn on; any other row's gates are on the editor's.
    scales = item.get('transforms')
    if live is not None:
        s = _copy.copy(live)
        s.data = live.data.copy(deep=True)
        note = 'live editor sample'
        if ov is not None:
            editor_data = s.data
            note += _recompensate_live(s, ov, str(name))
    else:
        path = item.get('path')
        if not path or not os.path.exists(path):
            raise RuntimeError('sample not loaded in editor and no readable FCS path')
        s = FlowSample(path, set_number=item_set_number(item))
        # Same event set as the editor's sample: QC exactly when it ran there.
        # Without it a reloaded item was gated and clustered on events the
        # editor had removed, so its counts disagreed with the editor's for
        # the same gate. An item saved before the flag existed came from a
        # load that always ran QC.
        if item.get('qc_applied', True):
            try:
                s.run_qc()
            except Exception as e:  # noqa: BLE001 - best-effort match to editor
                _PIPELINE_LOG.warning("  [workspace] QC on reload failed: %s", e)
        if ov is not None:
            try:
                s.manual_compensate(ov['matrix'], list(ov.get('channels') or []))
            except Exception as e:  # noqa: BLE001 - any failure refuses the member: it must not run on another compensation
                raise RuntimeError(f"cannot apply the {_where(ov)} matrix to "
                                   f"'{name}': {e}") from e
            if not s.is_compensated():
                # _apply_comp skips, with only a log line, a singular matrix
                # or one naming none of the file's channels.
                raise RuntimeError(
                    f"cannot apply the {_where(ov)} matrix to '{name}': it "
                    "is singular or names none of the sample's channels")
            own = ''
        else:
            own = _reload_comp(s, item, name, warnings)
        try:
            s.apply_transform()
        except Exception as e:  # noqa: BLE001 - best-effort match to editor; the failure is logged and the reload goes on
            _PIPELINE_LOG.warning("  [workspace] transform on reload failed: %s", e)
        # The loader's logicle is only the default; the item's gate was
        # drawn on the editor's transforms. Put the reload on them BEFORE
        # gating (measured: an asinh gate on the logicle reload kept 0 of
        # 800 events where 390 are right).
        _put_on_gate_scales(editor, s, scales)
        note = 'reloaded from file' + (
            f', compensated with {_override_words(ov)}' if ov is not None else own)

    # Never carry the editor's analysis columns/refs into our run.
    s.clusters = None
    s.umap_coords = None
    s.trimap_coords = None

    gid = item.get('gate_id')
    gates = item.get('gates') or {}
    if stats_out is not None:
        stats_out.extend(population_table_rows(s, item, str(name), stat_forms))
    if gid and gates:
        mask = cumulative_gate_mask(gates, gid, s.data)
        kept = int(mask.sum())
        s.data = s.data.loc[mask].reset_index(drop=True)
        note += f', gated {kept:,}/{len(mask):,}'
        if editor_data is not None:
            was = int(cumulative_gate_mask(gates, gid, editor_data).sum())
            note += f' ({was:,} with the editor\'s compensation)'
            if warnings is not None and abs(kept - was) > GATE_DRIFT_WARN * max(was, 1):
                warnings.append(
                    f"{name}: the gate keeps {kept:,} events with the "
                    f"{_where(ov)} matrix, {was:,} with the editor's "
                    "compensation it was drawn on")
    return s, note


# The population table every run writes (<label>_population_stats.csv):
# counts and frequencies, then each channel's Median and GeoMean on its
# linear values, as the Statistics window reports them.
POP_STATS = ('Count', '%Parent', '%Total')
POP_CHANNEL_STATS = ('Median', 'GeoMean')
POP_STATS_SUFFIX = '_population_stats.csv'


def stat_channels(sample) -> list[str]:
    """The channels a run's population table reports for `sample`: its
    scatter and fluorescence channels, in data order (never Time or an
    analysis column)."""
    from .channel_names import is_time_channel
    keep = set(getattr(sample, 'scatter_channels', None) or []) | set(
        getattr(sample, 'fluor_channels', None) or [])
    # A file's Time parameter can be listed with the fluorescence ones.
    return [c for c in sample.data.columns
            if c in keep and not is_time_channel(c)]


def population_table_rows(sample, item: dict, name: str,
                          stat_forms: dict | None = None) -> list[dict]:
    """``item``'s row of the run's population table, from `sample` before
    the item's gates are applied: the gated population (Count, %Parent,
    %Total) or, for a whole-sample item, all its events; plus Median and
    GeoMean of each :func:`stat_channels` channel on its linear values
    (gating.population_stats; `stat_forms` picks the GeoMean form, None the
    defaults). Hidden keys (``__gid__`` ...) are dropped."""
    from .gating import population_stats
    gid = item.get('gate_id')
    gates = item.get('gates') or {}
    chans = stat_channels(sample)
    whole = not (gid and gates)
    rows = population_stats(
        name, sample.data, {} if whole else gates,
        [] if whole else list(item.get('gate_chain') or gates),
        {c: c for c in chans}, chans, {*POP_STATS, *POP_CHANNEL_STATS},
        POP_CHANNEL_STATS, select=[] if whole else [str(gid)],
        transforms=getattr(sample, 'data_transforms', None),
        stat_forms=stat_forms, stat_axes=getattr(sample, 'stat_axes', None),
        all_events=whole)
    return [{k: v for k, v in r.items() if not str(k).startswith('__')}
            for r in rows]


def _write_population_stats(rows: list, out_dir: str, label: str,
                            res: dict) -> None:
    """``<label>_population_stats.csv``: one row per member, its group and
    trial first, in the standard CSV style and written whole
    (run_history.write_history_table: the run history reads it back)."""
    if not rows:
        return
    import pandas as pd

    from .run_history import write_history_table
    df = pd.DataFrame(rows)
    head = [c for c in ('Group', 'Trial', 'Sample', 'Population', *POP_STATS)
            if c in df.columns]
    df = df[head + [c for c in df.columns if c not in head]]
    path = os.path.join(out_dir, f'{label}{POP_STATS_SUFFIX}')
    write_history_table(path, df)            # read back by the run history
    if path not in res['files']:             # rewritten with reference rows
        res['files'].append(path)


def _plot_embedding(sample, prefix: str, out_dir: str, label: str,
                    color_by: str = 'cluster', suffix: str = '',
                    palette: str = DEFAULT_GROUP_PALETTE):
    """Render a 2-D embedding (``UMAP``/``TRIMAP``) to PNG via matplotlib's
    object API (thread-safe — no pyplot). ``color_by`` is a column: 'cluster'
    (numeric, tab20) or a categorical source column ('__group__'/'__sample__',
    distinct colours + legend — this is the group/sample COMPARISON view),
    coloured from the group palette ``palette`` (see
    group_palettes.n_group_colors). Returns the written path or None."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return None
    x, y = f'{prefix}1', f'{prefix}2'
    df = sample.data
    if x not in df.columns or y not in df.columns:
        return None
    sub = df.dropna(subset=[x, y])
    if sub.empty:
        return None
    try:
        fig = Figure(figsize=(6.4, 5.5), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        col = color_by if (color_by and color_by in sub.columns) else None
        if col == 'cluster':
            ax.scatter(sub[x], sub[y], c=sub['cluster'], cmap='tab20',
                       s=4, alpha=0.6, linewidths=0)
            title = f'{label} — {prefix} by cluster ({len(sub):,} events)'
        elif col is not None:
            cats = sorted(sub[col].astype(str).unique())
            colors = n_group_colors(len(cats), palette)
            for i, cat in enumerate(cats):
                m = sub[col].astype(str) == cat
                ax.scatter(sub.loc[m, x], sub.loc[m, y], s=4, alpha=0.5,
                           linewidths=0, color=colors[i], label=cat)
            ax.legend(markerscale=3, fontsize=7, loc='best', framealpha=0.85)
            title = f'{label} — {prefix} by {col.strip("_")} ({len(sub):,} events)'
        else:
            ax.scatter(sub[x], sub[y], s=4, alpha=0.6, linewidths=0)
            title = f'{label} — {prefix} ({len(sub):,} events)'
        ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        fig.tight_layout()
        out = os.path.join(out_dir, f'{label}_{prefix.lower()}{suffix}.png')
        from .export_stamp import background_kwargs, savefig_metadata
        fig.savefig(out, metadata=savefig_metadata(out),
                    **background_kwargs(fig, out))
        return out
    except Exception as exc:  # noqa: BLE001 - noted; a plot failure must not fail the run
        note_handled("draw the embedding plot image", exc)
        return None


def run_events_text(res: dict) -> str:
    """How many events a run clustered, for its result line: "clustered N
    of M events" when the 'max ev' cap subsampled the run unit (the cap
    applies to the unit's COMBINED events, so a rare population can thin out
    or vanish), else "N ev"."""
    n = int(res.get('n_events') or 0)
    total = int(res.get('n_total') or 0)
    if total > n:
        return f"clustered {n:,} of {total:,} events"
    return f"{n:,} ev"


def _blank_result(label: str, n_events: int = 0) -> dict:
    return {'label': label, 'ok': False, 'note': '', 'n_events': int(n_events),
            'n_clusters': 0, 'method': 'phenograph',
            'umap': False, 'tsne': False, 'phate': False, 'trimap': False,
            'pacmap': False, 'channels': [], 'events': None,
            'files': [], 'cancelled': False, 'error': None, 'warnings': []}


WORKSPACE_FORMAT = 'openflo-workspace'
WORKSPACE_SCHEMA = 1            # bump when WorkspaceModel.to_dict shape changes


def run_output_root(workspace_file: str | None = None,
                    session_file: str | None = None,
                    chosen: str | None = None) -> str:
    """The folder a Pipeline run's ``run_<time>`` folder goes in.

    In order: the folder ``chosen`` in Preferences; an ``outputs`` folder
    beside the saved ``workspace_file``; one beside the ``session_file``
    when the workspace has not been saved; else ``outputs/workspace``
    under the working folder, where every run went before. A file whose
    folder no longer exists is passed over."""
    if chosen and chosen.strip():
        return os.path.abspath(os.path.expanduser(chosen.strip()))
    for f in (workspace_file, session_file):
        if f:
            d = os.path.dirname(os.path.abspath(f))
            if os.path.isdir(d):
                return os.path.join(d, 'outputs')
    return os.path.abspath(os.path.join('outputs', 'workspace'))


def is_workspace_dict(d) -> bool:
    """Whether ``d`` (a loaded JSON value) is a saved workspace: a dict that
    carries the workspace ``format`` marker, or (a file from before the
    marker) a ``groups`` or ``loose`` list. Anything else would load as an
    empty workspace and replace the open one."""
    if not isinstance(d, dict):
        return False
    if d.get('format') == WORKSPACE_FORMAT:
        return True
    return (isinstance(d.get('groups'), list)
            or isinstance(d.get('loose'), list))


def newer_version_note(d: dict, *, schema_current: int = WORKSPACE_SCHEMA,
                       kind: str = 'workspace') -> str | None:
    """Alert string if a loaded workspace/recipe dict was produced by a NEWER
    OpenFlo than this build — either a newer on-disk ``schema`` (settings may
    not load fully) or a newer producing ``app_version`` (a feature this build
    predates). Returns None for same/older files (the common, fine case).

    Tk-free so it's unit-testable; the GUI loaders show the returned string.
    """
    from .update import current_version, is_newer
    try:
        schema = int(d.get('schema', 1) or 1)
    except (TypeError, ValueError):
        schema = 1
    cur = current_version()
    saved = str(d.get('app_version') or '').strip()
    if schema > schema_current:
        tail = f" (you're running OpenFlo {cur})." if cur and cur != '0' else "."
        return (f"This {kind} was saved in a newer OpenFlo format "
                f"(schema v{schema}; this build reads v{schema_current}). "
                f"Some settings may not load correctly — updating OpenFlo is "
                f"recommended{tail}")
    if saved and is_newer(saved, cur):
        return (f"This {kind} was created with OpenFlo {saved}; you're running "
                f"{cur}. It may use newer features that won't load correctly — "
                "consider updating OpenFlo (Help → Check for updates…).")
    return None


def merge_unit_cfg(base_cfg: dict, group_cfg: dict | None) -> dict:
    """Per-unit run config = the base (run-bar) cfg with a group's override
    merged on. Only keys already present in the base are overridable, so a
    stale override key can't smuggle in an unexpected setting. An empty/None
    group override inherits the base unchanged."""
    cfg = dict(base_cfg)
    ov = group_cfg or {}
    cfg.update({k: v for k, v in ov.items() if k in cfg})
    return cfg


def build_run_units(model, cfg: dict) -> list[dict]:
    """Group the workspace's items into RUN UNITS (each → one co-embedded
    UMAP/cluster). A unit is ``{label, members}`` where each member is
    ``{item, group, sample}``.

    - ``concatenate`` → ONE unit containing every item (all groups + loose),
      so groups are compared in a single embedding.
    - otherwise → one unit per non-empty group (its samples co-embedded) plus
      one unit per loose (ungrouped) item.

    Only items ticked in the tree's 👁 column are run: an unticked item, or
    every item of an unticked group, is left out (``displayed_items``).
    """
    def _safe(name: str) -> str:
        return ''.join(c if (c.isalnum() or c in '-_.') else '_'
                       for c in str(name)) or 'group'

    def _ticked(items: dict) -> list:
        return [it for it in items.values() if it.get('display', True)]

    groups = [g for g in model.groups.values() if g.get('display', True)]
    if cfg.get('concatenate'):
        members = []
        for g in groups:
            for it in _ticked(g['items']):
                members.append({'item': it, 'group': g['name'], 'sample': it.get('sample')})
        for it in _ticked(model.loose):
            members.append({'item': it, 'group': None, 'sample': it.get('sample')})
        return [{'label': 'all_groups', 'members': members}] if members else []

    units = []
    for g in groups:
        members = [{'item': it, 'group': g['name'], 'sample': it.get('sample')}
                   for it in _ticked(g['items'])]
        if members:
            units.append({'label': _safe(g['name']), 'members': members,
                          'gid': g['gid']})
    for it in _ticked(model.loose):
        units.append({'label': run_label(it),
                      'members': [{'item': it, 'group': None, 'sample': it.get('sample')}]})
    # Dedupe labels: two units sanitising to the same string (e.g. groups
    # 'Day 1' and 'Day/1' → 'Day_1', or a group name == a loose trial_sample)
    # would write to the same {label}_*.csv/.png in the shared out dir and
    # silently overwrite each other (both still counted ok). Suffix collisions.
    seen = {}
    for u in units:
        lbl = u['label']
        if lbl in seen:
            seen[lbl] += 1
            u['label'] = f"{lbl}__{seen[lbl]}"
        else:
            seen[lbl] = 0
    return units


def _source_tags(names: list, lengths: list):
    """A per-event source-label column, as a CATEGORICAL built from codes.

    These labels are one short string repeated across every event of a member,
    so the natural `np.repeat` + object-dtype assignment materialises millions
    of Python string references. Measured on a 12 x 250k unit (3M rows), that
    single step cost 415 ms of `prepare_unit`'s 531 ms and 300 MB of the
    frame — and the frame is then pickled to hand to the child process, so the
    cost is paid twice.

    Codes are int32 and the categories are a handful of strings, so the same
    information costs almost nothing: 531 ms -> 112 ms to build, 730 MB -> 414
    MB in memory, and the job.pkl write 2846 ms -> 780 ms. The values are
    identical.
    """
    import numpy as np
    import pandas as pd
    categories = list(dict.fromkeys(names))          # unique, order preserved
    if not categories:
        return pd.Categorical([], categories=[])
    index = {name: i for i, name in enumerate(categories)}
    codes = np.repeat(np.array([index[n] for n in names], dtype=np.int32),
                      lengths)
    return pd.Categorical.from_codes(codes, categories=categories)


def recipe_cfg(run_cfg: dict) -> dict:
    """The run settings a recipe carries: `run_cfg` without a frozen
    reference map (``reference``). The map belongs to its run template;
    carried in a recipe it would quietly replace the clustering of any
    workspace that loads it."""
    return {k: v for k, v in run_cfg.items() if k != 'reference'}


def cap_events(data, cfg: dict):
    """``data`` randomly subsampled to the run bar's 'max ev' (``max_events``;
    0 / blank = all events), seeded by the run seed. Every path that feeds
    compute_run applies it, because compute_run clusters and embeds whatever
    frame it gets: Batch folder… built its frame without it and clustered all
    1,500 events of each file at max ev 500."""
    from .seeds import resolve_seed
    max_ev = int(cfg.get('max_events') or 0) or None
    if max_ev and len(data) > max_ev:
        data = data.sample(n=max_ev, random_state=resolve_seed(
            cfg.get('seed', 42))).reset_index(drop=True)
    return data


_BATCH_FAILURES_SHOWN = 3     # the status line names this many; the log has all


def batch_failure_text(failures: list) -> str:
    """The status-line tail naming why Batch folder… files failed: each as
    ``  [!] file: reason``, the first few only, then how many more the log
    holds. Empty when nothing failed."""
    shown = failures[:_BATCH_FAILURES_SHOWN]
    text = ''.join(f"  [!] {name}: {why}" for name, why in shown)
    more = len(failures) - len(shown)
    if more:
        text += f"  [!] …and {more} more (see the log)"
    return text


def _pooled_column_transform(sample, c: str, pooled, theirs, mixed: list):
    """The transform column ``c`` of the pooled frame carries once a member
    whose own transform is ``theirs`` joins it on ``pooled``; ``sample``'s
    column is put on the pooled transform when the two differ."""
    from .channel_transforms import (
        UNKNOWN_SPEC,
        guessed_spec,
        is_guessed_spec,
        is_unknown_spec,
        transform_params,
    )
    if is_unknown_spec(pooled) or is_unknown_spec(theirs):
        # Nothing converts to or from an unknown scale: the
        # pooled column mixes scales, and its record says so.
        # Two unknown entries differ when their ORIGINS do
        # (channel_transforms.unknown_spec): only one old session's data
        # folder vouches for one transform per channel.
        if c not in mixed:
            mixed.append(c)
        return dict(UNKNOWN_SPEC)
    want = transform_params(pooled)
    if want != transform_params(theirs):
        sample.retransform([c], **(want or {'method': 'linear'}))
    if is_guessed_spec(theirs) and not is_guessed_spec(pooled):
        # One member's values are a guess: so are the
        # pooled column's.
        return guessed_spec(pooled or {'method': 'linear'})
    return pooled


def _pool_member_transforms(sample, pooled_tf: dict, mixed: list) -> None:
    """Put one member's columns on the pooled frame's transforms.

    Per column, the transform the pooled frame carries: the first member
    holding the column sets it, later members are put on it. A member
    reloaded from file comes back logicle even when the editor has the
    channel on asinh, and one column cannot hold two scales. Columns holding
    two scales are appended to ``mixed`` (see prepare_unit)."""
    own = dict(getattr(sample, 'data_transforms', None) or {})
    for c in sample.data.columns:
        if c not in pooled_tf:
            pooled_tf[c] = own.get(c)
        elif pooled_tf[c] != own.get(c):
            pooled_tf[c] = _pooled_column_transform(
                sample, c, pooled_tf[c], own.get(c), mixed)


def _member_tags(m: dict) -> tuple[str, str, str]:
    """``(group tag, sample tag, sample name)`` for a run-unit member."""
    gname = m.get('group') or ''
    sname = m.get('sample') or m['item'].get('sample', 'sample')
    return (gname or '(ungrouped)',
            f'{gname} / {sname}' if gname else sname, sname)


def _member_key(m: dict, sname: str) -> dict:
    """A run-unit member's Group, Trial and Sample, as its rows of the
    population table name them."""
    return {'Group': m.get('group') or '',
            'Trial': m['item'].get('trial') or '',
            'Sample': str(m['item'].get('sample') or sname)}


def _reference_rows(sample, m: dict, sname: str, mine: list[dict],
                    cfg: dict, errors: list[str]) -> list[dict]:
    """One member's share of each frozen reference cluster (run_cfg
    ``reference``), on all its events, named by its population (the row
    `mine` its population table gave); none without a reference. A member
    the reference cannot label (a channel it lacks, another scale) gives
    none and adds why to `errors`: compute_run fails the unit with it only
    after writing the population table, which needs no clustering."""
    ref = cfg.get('reference')
    if not ref:
        return []
    from .run_reference import label_events, member_rows
    try:
        labels, _far = label_events(sample.data, ref, sample.data_transforms)
    except ValueError as exc:
        errors.append(f"{sname}: {exc}")
        return []
    pop = str(mine[-1].get('Population')) if mine else 'All events'
    return member_rows(labels, ref, _member_key(m, sname), pop)


def _first_member_channels(sample, cfg: dict) -> list:
    """The unit's channels, from its first member: an explicit marker
    selection (cfg['channels']) intersected with what this sample actually
    has; else auto-picked proper markers."""
    sel = [c for c in (cfg.get('channels') or [])
           if c in sample.data.columns]
    return sel or proper_run_channels(sample)


def _pooled_frame(frames: list, gnames: list, snames: list, lengths: list,
                  cfg: dict):
    """``(data, n_all)``: the members' frames concatenated, each event tagged
    with its source, then subsampled to ``max_events``; ``n_all`` counts the
    events before subsampling."""
    import pandas as pd
    data = pd.concat(frames, ignore_index=True, sort=False)
    data['__group__'] = _source_tags(gnames, lengths)
    data['__sample__'] = _source_tags(snames, lengths)
    n_all = len(data)
    data = cap_events(data, cfg)
    if len(data) < n_all:
        # A categorical keeps every category after subsampling, and the
        # cluster-by-source crosstab downstream would then gain a zero column
        # for a member that lost all its events. Object dtype never produced
        # one, so drop the empties and keep that output identical.
        for col in ('__group__', '__sample__'):
            data[col] = data[col].cat.remove_unused_categories()
    return data, n_all


def _clusterable_channels(channels, common, mixed: list) -> tuple[list, list]:
    """``(channels, warnings)``: the unit's channels without those it cannot
    cluster on, and one warning per reason any were left out."""
    # A column whose members are on DIFFERENT scales, or on unknown scales
    # not known to match, cannot be put on one scale: an older OpenFlo
    # sidecar beside a sample on logicle or asinh, or two older _events tables
    # (measured: asinh median 1.197 beside linear 225.6, and FlowSOM split
    # them by scale). Clustering and embedding on it would compare
    # coordinates of two scales and still report a clean run, so it is left
    # out of both, and the run says so. (A column every member holds on an
    # unknown scale from ONE old session's data folder is kept, as 2.6.1 kept
    # it: that session's samples shared the editor's transform.)
    warnings = []
    # The channels came from the FIRST member. One a later member lacks is
    # NaN for all its events, and every one of them went unclustered under
    # a clean 'ok' (measured: 0 of 800, 2.6.1 too). Cluster on the channels
    # every member has, and say which were left out.
    absent = [c for c in (channels or []) if c not in (common or set())]
    if absent:
        channels = [c for c in channels or [] if c not in absent]
        warnings.append(f"left out of clustering, missing from some "
                        f"members: {', '.join(absent)}")
    left_out = [c for c in (channels or []) if c in mixed]
    if left_out:
        channels = [c for c in channels or [] if c not in mixed]
        warnings.append(
            f"left out of clustering, its members' scales differ or cannot "
            f"be shown to match (older OpenFlo files): {', '.join(left_out)}")
    return channels or [], warnings


def prepare_unit(editor, members: list, label: str, cfg: dict) -> dict:
    """(Main process) Resolve every member sample, tag each event with its
    source (``__group__`` / ``__sample__``), concatenate, and subsample the
    COMBINED frame to ``max_events`` up front. Returns
    ``{data, channels, label, note, n_events, color_by}`` where ``color_by`` is
    the categorical column to colour embeddings by (``__group__`` when the unit
    spans multiple groups, ``__sample__`` for multiple samples in one group,
    else None → colour by cluster only). Raises if no member resolves."""
    frames, channels, groups, samples = [], None, set(), set()
    gnames, snames, lengths = [], [], []
    pooled_tf: dict[str, dict | None] = {}   # see _pool_member_transforms
    mixed: list[str] = []          # columns holding two scales
    common: set | None = None      # columns EVERY member has
    comp_warnings: list[str] = []
    member_notes: list[str] = []
    fmo_rows: list[dict] = []
    flagged: list[set] = []        # channels each member's FMOs flagged
    sources: dict[str, list[str]] = {}
    pop_rows: list[dict] = []
    ref_rows: list[dict] = []      # each member's frozen-cluster shares
    ref_errors: list[str] = []     # members the frozen reference cannot label
    for m in members:
        comp = m.get('comp')
        mine: list[dict] = []
        sample, mnote = resolve_run_sample(editor, m['item'], comp=comp,
                                           warnings=comp_warnings,
                                           stats_out=mine,
                                           stat_forms=cfg.get('stat_forms'))
        for r in mine:
            pop_rows.append({'Group': m.get('group') or '',
                             'Trial': m['item'].get('trial') or '', **r})
        if override_matrix(comp) is None:      # an override names itself
            mnote += f", {comp_words(comp)}"
        for w in (comp or {}).get('warnings') or []:
            if w not in comp_warnings:
                comp_warnings.append(w)
        rows = apply_fmo_controls(
            sample, m, float(cfg.get('fmo_percentile', FMO_PERCENTILE)),
            comp_warnings)
        fmo_rows += rows
        flagged.append({r['channel'] for r in rows})
        member_notes.append(f"{m.get('sample') or m['item'].get('sample')}: {mnote}")
        if comp:
            # Each sample's own matrix (editor or saved) is one kind; a row
            # or group matrix or bead set is named.
            key = (comp.get('label') or comp.get('source', '?')
                   if comp.get('source') in ('row', 'group') else 'own matrix')
            sources.setdefault(key, []).append(
                m.get('sample') or m['item'].get('sample'))
        cols = set(sample.data.columns)
        common = cols if common is None else common & cols
        _pool_member_transforms(sample, pooled_tf, mixed)
        gtag, stag, sname = _member_tags(m)
        ref_rows += _reference_rows(sample, m, sname, mine, cfg, ref_errors)
        # Concatenate the ORIGINAL frames (no per-member full-frame .copy() just
        # to tag them — that would be N large copies before the concat's own
        # copy, a UI freeze on grouped runs). The source-tags are added once, post-concat.
        df = sample.data
        frames.append(df)
        gnames.append(gtag)
        snames.append(stag)
        lengths.append(len(df))
        groups.add(gtag)
        samples.add(stag if len(df) else sname)
        if channels is None:
            channels = _first_member_channels(sample, cfg)
    if not frames:
        raise RuntimeError('run unit has no resolvable members')
    _drop_stale_flags(frames, flagged, {r['channel'] for r in fmo_rows})
    data, n_all = _pooled_frame(frames, gnames, snames, lengths, cfg)
    color_by = ('__group__' if len(groups) > 1
                else ('__sample__' if len(samples) > 1 else None))
    channels, warnings = _clusterable_channels(channels, common, mixed)
    warnings = comp_warnings + warnings
    if len(sources) > 1:
        warnings.append('members compensated differently: ' + '; '.join(
            f"{lbl} ({', '.join(names)})" for lbl, names in sources.items()))
    note = f'{len(members)} sample(s)'
    if warnings:
        note += '; ' + '; '.join(warnings)
    return {'data': data, 'channels': channels, 'label': label,
            'note': note, 'n_events': int(len(data)), 'n_total': int(n_all),
            'color_by': color_by, 'warnings': warnings,
            'member_notes': member_notes, 'fmo_rows': fmo_rows,
            'pop_rows': pop_rows, 'ref_rows': ref_rows,
            'ref_errors': ref_errors,
            'data_transforms': {c: tf for c, tf in pooled_tf.items() if tf}}


def _drop_stale_flags(frames: list, flagged: list[set], channels: set) -> None:
    """A member's own ``<channel>_pos`` (a threshold gate's, kept by its
    session) is no FMO call: drop it for every channel this run's FMOs
    flag on another member, so the pooled flag holds this run's cutoffs
    only (NaN where the member had no FMO for it)."""
    for i, (df, own) in enumerate(zip(frames, flagged, strict=True)):
        stale = [f'{ch}_pos' for ch in sorted(channels - own)
                 if f'{ch}_pos' in df.columns]
        if stale:
            frames[i] = df.drop(columns=stale)


def prepare_run(editor, item: dict, cfg: dict) -> dict:
    """Single-item convenience wrapper around :func:`prepare_unit` (used by
    tests / non-grouped callers)."""
    return prepare_unit(editor, [{'item': item, 'group': None,
                                  'sample': item.get('sample')}],
                        run_label(item), cfg)


def _check_run_input(n: int, channels: list) -> None:
    """Raise when ``n`` events on ``channels`` are too few to cluster."""
    if n < 3:
        raise RuntimeError(f'too few events to cluster ({n})')
    if len(channels) < 2:
        raise RuntimeError(f'need ≥2 marker channels, got {len(channels)}: {channels}')


def _cluster_sample(sample, method: str, channels: list, cfg: dict,
                    k: int, n: int, seed) -> str:
    """Cluster ``sample`` in place with ``method``; returns the label column
    the method writes."""
    if method == 'leiden':
        # Data is already subsampled to max_events upstream, so pass n to
        # disable a second subsample (run_leiden types max_events as int).
        sample.run_leiden(channels=channels, n_neighbors=k,
                          resolution=float(cfg.get('resolution', 1.0)),
                          max_events=n, random_state=seed,
                          fast_graph=bool(cfg.get('fast_graph', False)))
        return 'leiden'
    if method == 'flowsom':
        nmeta = int(cfg.get('n_metaclusters', FLOWSOM_DEFAULT_METACLUSTERS))
        sample.run_flowsom(channels=channels, n_metaclusters=nmeta,
                           max_events=n, seed=seed)
        return 'flowsom_meta'
    sample.cluster(channels=channels, k=k, max_events=None,
                   random_state=seed,
                   reproducible=bool(cfg.get('reproducible', True)),
                   resolution=float(cfg.get('resolution', 1.0)))
    return 'cluster'


def _write_flowsom_map(sample, out_dir: str, label: str, res: dict) -> None:
    """After a FlowSOM clustering (not a frozen reference's labelling),
    its map (``<label>_flowsom_map.json``), the one Freeze reference
    clusters can make the template's reference."""
    som = getattr(sample, 'flowsom_result', None)
    if not som:
        return
    from .run_reference import build_map, write_map
    res['files'].append(write_map(out_dir, label, build_map(
        som, sample.data, sample.data_transforms,
        f"{os.path.basename(os.path.normpath(out_dir))} · {label}")))


def _label_by_reference(sample, prep: dict, ref: dict, out_dir: str,
                        res: dict) -> str:
    """Label `sample`'s events with the frozen reference map `ref` (no
    clustering), warn when they lie far from it, and add each member's
    share of each reference cluster (``prep['ref_rows']``, from
    prepare_unit on all its events) to the population table. Returns the
    label column."""
    from .run_reference import drift_warning, label_events
    if prep.get('ref_errors'):
        raise ValueError('; '.join(prep['ref_errors']))
    labels, far_pct = label_events(sample.data, ref, sample.data_transforms)
    sample.data['flowsom_meta'] = labels
    res['reference'] = {'source': ref.get('source', ''), 'far_pct': far_pct}
    res['note'] = '; '.join(filter(None, (
        res['note'], f'frozen reference clusters ({far_pct:.1f}% of events '
                     'far from them)')))
    warning = drift_warning(far_pct)
    if warning:
        res['warnings'].append(warning)
    rows = list(prep.get('ref_rows') or [])
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        _write_population_stats(list(prep.get('pop_rows') or []) + rows,
                                out_dir, res['label'], res)
    return 'flowsom_meta'


def _applied_resolution(method: str, cfg: dict, sample):
    """The resolution the run applied (None: FlowSOM, or PhenoGraph's
    Louvain, which has none)."""
    if method == 'leiden':
        return float(cfg.get('resolution', 1.0))
    if method == 'flowsom':
        return None
    return (getattr(sample, 'cluster_settings', None) or {}).get('resolution')


def _alias_cluster_column(sample, src_col: str, method: str) -> None:
    """Alias the chosen method's label column to 'cluster' so every
    downstream step (frequency CSV, embedding colour, import-as-
    populations) stays uniform regardless of method."""
    if src_col == 'cluster':
        return
    if src_col not in sample.data.columns:
        # run_leiden / run_flowsom silently return WITHOUT a label
        # column on too-few-finite-events / no-usable-channels (e.g. a
        # gated rare population under flowsom's default 100-event floor).
        # Surface that as a failure instead of a spurious ok=True run
        # with 0 clusters and no cluster CSV.
        raise RuntimeError(
            f'{method} clustering produced no labels '
            '(too few finite events or no usable channels)')
    sample.data['cluster'] = sample.data[src_col]


def _add_fmo_positivity(freq, sample, channels) -> None:
    """``pct_pos_<channel>`` per cluster for each of ``channels`` (those
    this run's FMO controls flagged, ``<channel>_pos``; a column of that
    name the sample already had is no FMO call and is not reported), and a
    ``phenotype`` naming each such channel + (50% or more positive) or -.
    A member without that channel's FMO holds NaN there and does not count."""
    flags = [f'{ch}_pos' for ch in channels
             if f'{ch}_pos' in sample.data.columns]
    if not flags or freq.empty:
        return
    cluster = sample.data['cluster']
    pheno = {c: [] for c in freq['cluster']}
    for flag in flags:
        ch = flag[:-4]
        pct = (sample.data[flag].groupby(cluster).mean() * 100
               ).reindex(freq['cluster'])
        freq[f'pct_pos_{ch}'] = pct.round(2).to_numpy()
        for c, v in zip(freq['cluster'], pct.to_numpy(), strict=True):
            if v == v:                      # not NaN
                pheno[c].append(f"{ch}{'+' if v >= 50 else '-'}")
    freq['phenotype'] = [' '.join(pheno[c]) for c in freq['cluster']]


def _write_fmo_thresholds(rows: list, out_dir: str, label: str, res: dict) -> None:
    """``<label>_fmo_thresholds.csv``: one row per member and FMO channel."""
    if not rows:
        return
    import pandas as pd
    path = os.path.join(out_dir, f'{label}_fmo_thresholds.csv')
    from .table_export import write_frame_csv
    write_frame_csv(path, pd.DataFrame(rows))
    res['files'].append(path)


def _write_cluster_tables(sample, labels, n: int, out_dir: str, label: str,
                          color_by, res: dict, fmo_channels=()) -> None:
    """Write the cluster frequency CSV and, with ``color_by``, the % of each
    source's events per cluster; add their paths to ``res['files']``."""
    freq = (labels.value_counts().sort_index()
            .rename_axis('cluster').reset_index(name='count'))
    freq['percent'] = (freq['count'] / max(1, n) * 100).round(3)
    _add_fmo_positivity(freq, sample, fmo_channels)
    csv_path = os.path.join(out_dir, f'{label}_clusters.csv')
    freq.to_csv(csv_path, index=False)
    res['files'].append(csv_path)
    # Cross-group/sample comparison: % of each source's events per cluster.
    if color_by and color_by in sample.data.columns:
        import pandas as pd
        ct = pd.crosstab(sample.data['cluster'], sample.data[color_by])
        ct_pct = (ct / ct.sum(axis=0).replace(0, 1) * 100).round(2)
        xpath = os.path.join(out_dir, f'{label}_cluster_by_{color_by.strip("_")}.csv')
        ct_pct.to_csv(xpath)
        res['files'].append(xpath)


def _standardise_markers(sample, channels: list) -> None:
    """Z-score each marker in place, for the EMBEDDINGS only.

    Clustering already used the transformed values. Equal per-marker weight
    stops a single high-variance marker from stretching the layout (UMAP
    hides this by normalising its output; TriMap does NOT, so it matters most
    there). This matches FlowJo's pre-scaling before dimensionality
    reduction. Note: TriMap is global-structure preserving, so it will still
    spread more / use a larger coordinate range than UMAP by design."""
    for c in channels:
        col = sample.data[c].astype(float)
        sd = float(col.std())
        if sd > 0:
            sample.data[c] = (col - col.mean()) / sd


def _embedding_plots(sample, prefix: str, out_dir: str, label: str, color_by,
                     palette: str, res: dict) -> bool:
    """Plot embedding ``prefix`` by cluster and, with ``color_by``, by
    source; add the PNGs to ``res['files']``. True when the cluster plot was
    written."""
    wrote = False
    p = _plot_embedding(sample, prefix, out_dir, label, color_by='cluster')
    if p:
        res['files'].append(p)
        wrote = True
    if color_by:                          # the group/sample comparison view
        p2 = _plot_embedding(sample, prefix, out_dir, label,
                             color_by=color_by, suffix='_by_source',
                             palette=palette)
        if p2:
            res['files'].append(p2)
    return wrote


def _embedding_runners(sample, channels: list, nn: int, seed) -> list:
    """``(cfg key, column prefix, run)`` for each embedding. sample_n=0
    embeds all rows (data was already subsampled to max_events up front)."""
    return [
        ('umap',   'UMAP',   lambda: sample.run_umap(
            channels=channels, n_neighbors=nn, sample_n=0,
            random_state=seed)),
        ('tsne',   'TSNE',   lambda: sample.run_tsne(
            channels=channels, sample_n=0, random_state=seed)),
        ('phate',  'PHATE',  lambda: sample.run_phate(
            channels=channels, sample_n=0, random_state=seed)),
        ('trimap', 'TRIMAP', lambda: sample.run_trimap(
            channels=channels, sample_n=0, random_state=seed)),
        ('pacmap', 'PACMAP', lambda: sample.run_pacmap(
            channels=channels, sample_n=0, random_state=seed)),
    ]


def _run_embeddings(sample, cfg: dict, channels: list, nn: int, seed,
                    out_dir: str, label: str, color_by, palette: str,
                    res: dict, say, cancelled) -> bool:
    """Run each embedding ``cfg`` enables, each writing <PREFIX>1/2 + a PNG,
    and record it in ``res``. False when cancelled (``res['cancelled']`` is
    then set)."""
    for key, prefix, run in _embedding_runners(sample, channels, nn, seed):
        if not cfg.get(key):
            continue
        if cancelled():
            res['cancelled'] = True
            return False
        # Said in the run's summary, not only the progress line: the next
        # step overwrites that, and "Done: 4 ok" read as if it had run.
        missing = embedding_unavailable(key, load=True)
        if missing:
            res[key] = False
            if key in (cfg.get('unchosen_embeddings') or ()):
                # On only by default: one quiet note, no warning.
                say(f'{label}: {missing} (on by default; not chosen)')
            else:
                res['warnings'].append(missing)
            continue
        say(f'{label}: {prefix} on {len(channels)} markers…')
        try:
            run()
            res[key] = _embedding_plots(sample, prefix, out_dir, label,
                                        color_by, palette, res)
            if not res[key]:
                res['warnings'].append(f'{prefix} made no plot')
        except Exception as exc:  # noqa: BLE001 - one embedding failing must not stop the run
            say(f'{label}: {prefix} failed — {type(exc).__name__}: {exc}')
            res['warnings'].append(
                f'{prefix} failed: {type(exc).__name__}: {exc}')
            res[key] = False
    return True


def _write_events_table(sample, events_df, out_dir: str, label: str,
                        cfg: dict, res: dict, say) -> None:
    """Write the per-event table (original markers + cluster + source +
    embedding coords) -- importable back into the editor as a processed
    sample so the run's clusters/embeddings become populations to gate and
    plot. A failure is said and kept in the result's warnings, not raised:
    the clusters and plots are already written, so the run still counts, but
    the run's result line and final line show that the table is missing."""
    try:
        for prefix in ('UMAP', 'TSNE', 'PHATE', 'TRIMAP', 'PACMAP'):
            for ax in ('1', '2'):
                col = f'{prefix}{ax}'
                if col in sample.data.columns:
                    events_df[col] = sample.data[col].to_numpy()
        written = write_run_events(
            events_df, out_dir, label, sample.data_transforms,
            also_csv=bool(cfg.get('events_csv', False)))
        res['files'].extend(written)
        res['events'] = written[0]
    except Exception as exc:  # noqa: BLE001 - any write failure becomes a warning on the result; the run's other outputs stand
        say(f'{label}: events file failed — {type(exc).__name__}: {exc}')
        res['warnings'].append(
            f'events file not written: {type(exc).__name__}: {exc}')


def _write_member_notes(notes: list, out_dir: str, label: str,
                        res: dict) -> None:
    """``<label>_members.txt``: one line per member saying how its events
    were read (live or reloaded, the compensation put on them, a gate's
    count under that compensation and under the editor's)."""
    if not notes:
        return
    path = os.path.join(out_dir, f'{label}_members.txt')
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(notes) + '\n')
    except OSError as e:      # informational: the run's results still stand
        res['warnings'].append(f'members file not written: {e}')
        return
    res['files'].append(path)


def compute_run(prep: dict, cfg: dict, out_dir: str,
                progress=None, should_cancel=None) -> dict:
    """(Process-agnostic) Cluster + optional UMAP/TriMap on a prepared frame and
    write outputs (cluster CSV + embedding PNGs). Runs either in-process or
    inside a child process. Exception-safe: returns a status dict, never raises.

    ``progress`` = ``callable(str)`` for live phase messages; ``should_cancel``
    = ``callable() -> bool`` checked at phase boundaries (the subprocess path
    cancels by terminating the process instead).

    ``cfg['group_palette']`` names the group palette of the by-source plots
    ('okabe_ito' or 'classic'); absent, it is
    group_palettes.DEFAULT_GROUP_PALETTE."""
    from .flow_sample import FlowSample
    from .seeds import resolve_seed
    label = prep.get('label', 'item')
    channels = list(prep.get('channels') or [])
    say = progress if callable(progress) else (lambda _m: None)
    cancelled = should_cancel if callable(should_cancel) else (lambda: False)
    res = _blank_result(label, prep.get('n_events', 0))
    res['n_total'] = int(prep.get('n_total') or prep.get('n_events', 0))
    res['note'] = prep.get('note', '')
    res['warnings'] = list(prep.get('warnings') or [])
    res['member_notes'] = list(prep.get('member_notes') or [])
    fmo_rows = list(prep.get('fmo_rows') or [])
    res['channels'] = list(channels)
    try:
        data = prep['data']
        n = int(len(data))
        res['n_events'] = n
        res['n_total'] = max(n, res['n_total'])
        # Written first: the population table needs no clustering, so a
        # unit too small to cluster still reports its populations.
        if prep.get('pop_rows'):
            os.makedirs(out_dir, exist_ok=True)
            _write_population_stats(list(prep['pop_rows']), out_dir, label, res)
        _check_run_input(n, channels)

        sample = FlowSample.from_dataframe(data, name=label)
        # from_dataframe takes values as linear; say what they carry instead.
        sample.data_transforms = dict(prep.get('data_transforms') or {})
        seed = resolve_seed(cfg.get('seed', 42))
        # Clamp neighbours below the event count so small sets can't make
        # Phenograph / UMAP raise (the classic small-n crash).
        k = max(2, min(int(cfg.get('k', 30)), n - 1))

        if cancelled():
            res['cancelled'] = True
            return res
        method = str(cfg.get('method', 'phenograph')).lower()
        if cfg.get('reference'):
            say(f'{label}: labeling {n:,} cells with the frozen reference…')
            method = 'flowsom'
            src_col = _label_by_reference(sample, prep, cfg['reference'],
                                          out_dir, res)
        else:
            say(f'{label}: {method} clustering {n:,} cells on '
                f'{len(channels)} markers…')
            src_col = _cluster_sample(sample, method, channels, cfg, k, n, seed)
        res['method'] = method
        res['resolution'] = _applied_resolution(method, cfg, sample)
        _alias_cluster_column(sample, src_col, method)
        labels = sample.data.get('cluster')
        if labels is not None:
            res['n_clusters'] = int(labels[labels >= 0].nunique())

        os.makedirs(out_dir, exist_ok=True)
        _write_member_notes(res['member_notes'], out_dir, label, res)
        _write_flowsom_map(sample, out_dir, label, res)
        color_by = prep.get('color_by')          # '__group__' / '__sample__' / None
        # The GUI puts the user's group palette in cfg; the core never reads
        # preferences, so a caller that leaves it out gets the default.
        palette = str(cfg.get('group_palette') or DEFAULT_GROUP_PALETTE)
        if labels is not None:
            _write_cluster_tables(sample, labels, n, out_dir, label, color_by,
                                  res, fmo_channels=sorted(
                                      {r['channel'] for r in fmo_rows}))
        _write_fmo_thresholds(fmo_rows, out_dir, label, res)

        # Capture the per-event frame (ORIGINAL marker scale + cluster + source
        # tags) BEFORE the in-place z-score below, so an exported events CSV is
        # importable as a sample with usable marker values. Embedding coords are
        # grafted on after they're computed.
        events_df = sample.data.copy()
        _standardise_markers(sample, channels)
        if not _run_embeddings(sample, cfg, channels, k, seed, out_dir, label,
                               color_by, palette, res, say, cancelled):
            return res
        _write_events_table(sample, events_df, out_dir, label, cfg, res, say)
        res['ok'] = True
    except Exception as e:  # noqa: BLE001 - per-item isolation is the point
        res['error'] = f'{type(e).__name__}: {e}'
    return res


EVENTS_STEM = '_events'
EVENTS_CSV_EXT = '.csv'


def _events_frame_for_binary(df):
    """`df` with each categorical column of text turned into plain text, the
    form a CSV of it reads back as. The run's ``__group__`` / ``__sample__``
    source tags are categoricals (see `_source_tags`), which the binary
    format does not carry; left as they are, every run would fall back to
    CSV. A categorical holding anything else (or a missing value) is left
    alone, so `write_frame` refuses it and the caller writes CSV."""
    import pandas as pd
    out = None
    for c in df.columns:
        col = df[c]
        if not isinstance(col.dtype, pd.CategoricalDtype) or col.isna().any():
            continue
        cats = col.cat.categories
        if len(cats) and pd.api.types.infer_dtype(cats, skipna=False) != 'string':
            continue
        if out is None:
            out = df.copy(deep=False)
        out[c] = col.astype(str).astype(object)
    return df if out is None else out


def write_run_events(df, out_dir: str, label: str, transforms,
                     also_csv: bool = False) -> list[str]:
    """Write a run unit's per-event table and return the paths written, the
    one 'Load in editor' should read first.

    The table goes in the binary sidecar format (``<label>_events.npz``,
    session_sidecar.write_frame): writing ~300k events as CSV took ~10 s a
    sample against ~0.2 s binary. `also_csv` (the run setting 'Also write
    events as CSV') also writes ``<label>_events.csv``. A table holding a
    column the binary format can't carry is written as that CSV alone, as
    runs before the binary file did.

    One transform record serves both files: it is named from the full stem
    (``<label>_events_transforms.json``) and lists the same columns and row
    count. It is removed before anything is written, so a failed write can't
    leave an earlier run's record beside new data. A file of the other format
    left by an earlier run into this folder is removed too, so the folder
    never holds a stale table beside the current one."""
    from .session_sidecar import SIDECAR_EXT, write_frame
    from .transform_sidecar import (
        discard_transforms_sidecar,
        write_transforms_sidecar,
    )
    stem = os.path.join(out_dir, f'{label}{EVENTS_STEM}')
    bin_path, csv_path = stem + SIDECAR_EXT, stem + EVENTS_CSV_EXT
    discard_transforms_sidecar(bin_path)
    written = []
    if write_frame(_events_frame_for_binary(df), bin_path):
        written.append(bin_path)
    else:
        _remove_quietly(bin_path)
    if also_csv or not written:
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
    else:
        _remove_quietly(csv_path)
    write_transforms_sidecar(written[0], transforms, df.columns, len(df))
    return written


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def run_workspace_item(editor, item: dict, cfg: dict, out_dir: str,
                       progress=None, should_cancel=None) -> dict:
    """In-process run of one item (resolve → prepare → compute). Used by tests
    and any non-subprocess caller. The GUI instead runs items in CHILD
    PROCESSES (``prepare_unit`` in the parent + :func:`subprocess_main`) so a
    native crash / OOM / cancel can't take down the app."""
    label = run_label(item)
    try:
        prep = prepare_run(editor, item, cfg)
    except Exception as e:  # noqa: BLE001 - any failure preparing the item comes back as the result's error, as a child run would report it
        res = _blank_result(label)
        res['error'] = f'{type(e).__name__}: {e}'
        return res
    return compute_run(prep, cfg, out_dir, progress=progress, should_cancel=should_cancel)


_CREATE_NO_WINDOW   = 0x08000000
_CREATE_NEW_CONSOLE = 0x00000010
_DETACHED_PROCESS   = 0x00000008


def _no_window_creationflags(creationflags: int) -> int:
    """Add CREATE_NO_WINDOW unless the caller explicitly asked for a new or
    detached console. Pure + idempotent, so it's unit-testable off Windows."""
    if creationflags & (_CREATE_NEW_CONSOLE | _DETACHED_PROCESS):
        return creationflags
    return creationflags | _CREATE_NO_WINDOW


def _suppress_child_windows() -> None:
    """Windows only: make EVERY subprocess this worker spawns window-less.

    PhenoGraph shells out to the native Louvain binaries (convert.exe /
    community.exe / hierarchy.exe) via ``subprocess.Popen`` — repeatedly, and
    WITHOUT CREATE_NO_WINDOW — so each call flashes a console window that
    steals focus from the GUI (the gate editor "spasm"). CREATE_NO_WINDOW on
    the worker process itself doesn't reliably reach these grandchild console
    apps, so patch ``subprocess.Popen`` to inject the flag directly. Idempotent
    and a no-op off Windows."""
    import sys
    if sys.platform != 'win32':
        return
    import subprocess as _sp
    if getattr(_sp.Popen, '_openflo_nowindow', False):
        return
    _Orig = _sp.Popen

    class _NoWindowPopen(_Orig):
        _openflo_nowindow = True

        def __init__(self, *args, **kwargs):
            kwargs['creationflags'] = _no_window_creationflags(
                kwargs.get('creationflags', 0))
            super().__init__(*args, **kwargs)

    _sp.Popen = _NoWindowPopen


def own_process_group(proc):
    """The id of the process group `proc` leads (POSIX, `proc` still
    running), else None -- on Windows, or for a child that was not started
    in a session of its own."""
    import sys
    if sys.platform == 'win32' or proc.poll() is not None:
        return None
    getpgid = getattr(os, 'getpgid', None)      # POSIX only
    try:
        return proc.pid if getpgid and getpgid(proc.pid) == proc.pid else None
    except OSError:
        return None


def signal_process_group(proc, pgid, signame):
    """Send `signame` to process group `pgid`; without one, to `proc` alone
    (if given and still running). A group with no members left is fine."""
    import signal
    sig = getattr(signal, signame)
    killpg = getattr(os, 'killpg', None)        # POSIX only
    if pgid is not None and killpg is not None:
        try:
            killpg(pgid, sig)
        except OSError:
            pass
        return
    if proc is not None and proc.poll() is None:
        proc.send_signal(sig)


def subprocess_main() -> None:
    """Entry point for a job CHILD process (``python -m openflo.workspace_run
    job.pkl result.pkl``, see the guard at the end of this module): read a
    pickled ``{prep, cfg, out_dir}`` job from ``sys.argv[1]``, run it, stream
    marker-prefixed progress on stdout, and write the pickled result to
    ``sys.argv[2]``. A real subprocess (NOT multiprocessing) so the parent can
    kill the whole process tree to cancel without corrupting a shared queue —
    the failure mode that crashed the GUI on Cancel."""
    import pickle
    import sys

    from ._console import hide_future_warnings
    hide_future_warnings()
    # PhenoGraph's Louvain binaries would otherwise flash a console window per
    # call — silence the whole spawned subtree before any clustering runs.
    _suppress_child_windows()
    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path, 'rb') as fh:
        job = pickle.load(fh)
    # UMAP's numba kernels take ~18s to compile and are cached in NO process,
    # so every child in a 20-unit run pays it again. Clustering runs first and
    # is compiled C that releases the GIL, so the compile can happen alongside
    # it instead of in front of the embedding. Not speculative — cfg already
    # says whether an embedding is coming. See openflo/warmup.py.
    from . import warmup
    warmup.start(job['cfg'])
    res = compute_run(job['prep'], job['cfg'], job['out_dir'],
                      progress=lambda m: print(PROGRESS_MARKER + m, flush=True))
    with open(result_path, 'wb') as fh:
        pickle.dump(res, fh)


def find_run_outputs(out_dir: str) -> list[dict]:
    """Group a run directory's files by item label (Tk-free, testable).

    Returns ``[{label, csv, events, umap, tsne, phate, trimap, pacmap,
    populations, clusters}]`` — paths absolute or None, ``clusters`` = data-row count of the
    cluster CSV (or None). ``events`` is the binary ``<label>_events.npz``,
    or the ``<label>_events.csv`` of a run that wrote no binary file (runs
    before it, or a table the binary format can't carry)."""
    from .session_sidecar import SIDECAR_EXT
    out: list[dict] = []
    if not out_dir or not os.path.isdir(out_dir):
        return out
    by_label: dict[str, dict] = {}
    bin_events = EVENTS_STEM + SIDECAR_EXT
    for f in sorted(os.listdir(out_dir)):
        for suffix, key in ((bin_events, 'events'),
                            (EVENTS_STEM + EVENTS_CSV_EXT, 'events'),
                            ('_clusters.csv', 'csv'),
                            (POP_STATS_SUFFIX, 'populations'),
                            ('_umap.png', 'umap'),
                            ('_tsne.png', 'tsne'), ('_phate.png', 'phate'),
                            ('_trimap.png', 'trimap'), ('_pacmap.png', 'pacmap')):
            if f.endswith(suffix):
                files = by_label.setdefault(f[:-len(suffix)], {})
                # The binary file wins over a CSV of the same run, whichever
                # is listed first.
                if key != 'events' or suffix == bin_events or key not in files:
                    files[key] = os.path.join(out_dir, f)
                break
    for lbl in sorted(by_label):
        files = by_label[lbl]
        rec = {'label': lbl, 'csv': files.get('csv'),
               'events': files.get('events'),
               'umap': files.get('umap'), 'tsne': files.get('tsne'),
               'phate': files.get('phate'), 'trimap': files.get('trimap'),
               'pacmap': files.get('pacmap'),
               'populations': files.get('populations'), 'clusters': None}
        if rec['csv']:
            try:
                with open(rec['csv'], encoding='utf-8') as fh:
                    rec['clusters'] = max(0, sum(1 for _ in fh) - 1)   # minus header
            except OSError:
                pass
        out.append(rec)
    return out


def item_to_json(it: dict) -> dict:
    """Copy an item dict with its ``comp_matrix`` ndarray turned into a tagged
    nested-list so the whole thing is JSON-serialisable."""
    d = dict(it)
    m = d.get('comp_matrix')
    if m is not None and hasattr(m, 'tolist'):
        d['comp_matrix'] = {'__ndarray__': m.tolist()}
    return d


def item_from_json(d: dict) -> dict:
    """Inverse of :func:`item_to_json` — restore a tagged ``comp_matrix`` to an
    ndarray and re-seed the per-item run flags."""
    it = dict(d)
    m = it.get('comp_matrix')
    if isinstance(m, dict) and '__ndarray__' in m:
        import numpy as np
        it['comp_matrix'] = np.array(m['__ndarray__'], dtype=float)
    it.setdefault('display', True)
    it.setdefault('comp_override', None)
    it.setdefault('fmo', None)
    return it


# ── Model (no Tk) ──────────────────────────────────────────────────────────


def _shape_label(m) -> str:
    shape = getattr(m, 'shape', None)
    return f'{shape[0]}×{shape[1]}' if (shape and len(shape) == 2) else 'matrix'


def override_label(ov: dict, where: str) -> str:
    """The Comp column's wording for a row (``where='row'``) or group
    override, the one :func:`resolve_comp` gives its members."""
    return _override_comp(ov, where)['label']


def _override_comp(ov: dict, where: str) -> dict:
    """The run's comp from a row (``where='row'``) or group override."""
    tag = 'row' if where == 'row' else 'grp'
    if ov.get('kind') == 'beads':
        files = list(ov.get('files') or [])
        return {'source': where, 'kind': 'beads', 'files': files,
                'matrix': None, 'channels': [],
                'label': f'beads:{len(files)} ({tag})', 'ready': bool(files)}
    return {'source': where, 'kind': 'matrix', 'files': [],
            'matrix': ov.get('matrix'), 'channels': list(ov.get('channels') or []),
            'label': f'matrix ({tag})', 'ready': True}


def _same_matrix(a, b) -> bool:
    import numpy as np
    try:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return False
    return a.shape == b.shape and bool(np.allclose(a, b))


def resolve_comp(model, mid: str, editor=None) -> dict:
    """The compensation a run applies to item ``mid``, in this order (Sky,
    scope audit L3): the row's own beads or matrix; else its group's; else
    the sample's matrix in the editor now (when ``editor`` has it loaded,
    so an adjustment made after the drop counts); else the matrix saved
    when the item was dropped; else none.

    A run template's row (no override) takes its file's own $SPILLOVER.

    Returns ``{'source', 'kind', 'label', 'ready', 'live', 'matrix',
    'channels', 'files'}``: ``source`` is 'row', 'group', 'editor', 'file',
    'saved' or 'none'; ``kind`` 'beads', 'matrix' or 'none'; ``ready`` False when
    there is neither a matrix nor beads; ``live`` True when the sample is
    loaded in ``editor`` (a run then starts from the editor's sample).
    Without an editor the editor level cannot be seen, and the saved
    matrix is reported."""
    it = model.item(mid)
    if it is None:
        return {'source': 'none', 'kind': 'none', 'label': '⚠',
                'ready': False, 'live': False, 'matrix': None,
                'channels': [], 'files': []}
    live = live_sample(editor, it)
    gid = model.item_group(mid)
    for ov, where in ((it.get('comp_override'), 'row'),
                      (model.groups[gid]['comp'] if gid else None, 'group')):
        if ov:
            return dict(_override_comp(ov, where), live=live is not None)
    if it.get('template_slot'):
        # A run template's row: its file's own $SPILLOVER (_reload_comp).
        return {'source': 'file', 'kind': 'matrix', 'files': [],
                'matrix': None, 'channels': [], 'label': 'own $SPILLOVER',
                'ready': True, 'live': False}
    saved = it.get('comp_matrix')
    if live is not None:
        m = getattr(live, 'comp_matrix', None)
        if m is not None:
            changed = saved is not None and not _same_matrix(m, saved)
            return {'source': 'editor', 'kind': 'matrix', 'files': [],
                    'matrix': m,
                    'channels': list(getattr(live, 'comp_channels', None) or []),
                    'label': (f'{_shape_label(m)} (editor'
                              + (', changed)' if changed else ')')),
                    'ready': True, 'live': True}
        return {'source': 'none', 'kind': 'none', 'label': '⚠ none (editor)',
                'ready': False, 'live': True, 'matrix': None,
                'channels': [], 'files': []}
    if saved is not None:
        return {'source': 'saved', 'kind': 'matrix', 'files': [],
                'matrix': saved, 'channels': list(it.get('comp_channels') or []),
                'label': _shape_label(saved), 'ready': True, 'live': False}
    return {'source': 'none', 'kind': 'none', 'label': '⚠', 'ready': False,
            'live': False, 'matrix': None, 'channels': [], 'files': []}


class WorkspaceModel:
    """Groups of pipeline items + their comp/FMO sources and display flags.

    ``loose`` holds top-level items (mid → payload); ``groups`` holds groups
    (gid → {name, display, comp, fmo, items}). A payload gains ``display``,
    ``comp_override`` (None | {'kind':'beads','files':[...]} | {'kind':'matrix',…}),
    and ``fmo`` (None | {'files':[...]}). Group ``comp``/``fmo`` cascade.
    """

    def __init__(self, title: str = 'Pipeline'):
        self.title = title
        self.groups: dict[str, dict] = {}
        self.loose: dict[str, dict] = {}
        self._gseq = 0
        self._mseq = 0
        self.run_cfg = default_run_cfg()   # k / max_events / seed / umap / trimap
        # {'path', 'name', 'version'} of the run template this workspace
        # was built from (its runs go in that template's history), or None.
        self.template: dict | None = None

    # -- creation --
    def new_group(self, name: str | None = None) -> str:
        """Create an empty group and return its id; `name` defaults to ``Group <n>``."""
        self._gseq += 1
        gid = f'G{self._gseq}'
        self.groups[gid] = {'gid': gid, 'name': name or f'Group {self._gseq}',
                            'display': True, 'comp': None, 'fmo': None,
                            'cfg': {}, 'items': {}}
        return gid

    def add_item(self, payload: dict, gid: str | None = None) -> str:
        """Add a copy of `payload` as an item and return its id.

        The item joins group `gid` when it exists, otherwise the loose list.
        """
        self._mseq += 1
        mid = f'M{self._mseq}'
        item = dict(payload)
        item.setdefault('display', True)
        item.setdefault('comp_override', None)
        item.setdefault('fmo', None)
        if gid and gid in self.groups:
            self.groups[gid]['items'][mid] = item
        else:
            self.loose[mid] = item
        return mid

    # -- lookup --
    def _store_of(self, mid):
        if mid in self.loose:
            return self.loose, None
        for g in self.groups.values():
            if mid in g['items']:
                return g['items'], g['gid']
        return None, None

    def item(self, mid: str) -> dict | None:
        """Return the item with id `mid`, or None when there is none."""
        store, _ = self._store_of(mid)
        return store[mid] if store else None

    def item_group(self, mid: str) -> str | None:
        """Return the id of the group holding `mid`, or None when it is loose or absent."""
        _store, gid = self._store_of(mid)
        return gid

    def all_items(self):
        """Yield (mid, item, gid|None) for every item, loose first."""
        for mid, it in self.loose.items():
            yield mid, it, None
        for g in self.groups.values():
            for mid, it in g['items'].items():
                yield mid, it, g['gid']

    # -- removal / move --
    def remove_item(self, mid: str) -> None:
        """Delete item `mid` wherever it is; unknown ids are ignored."""
        store, _ = self._store_of(mid)
        if store is not None:
            store.pop(mid, None)

    def remove_group(self, gid: str, *, keep_items=True) -> None:
        """Delete group `gid`; its items move to the loose list unless `keep_items` is False."""
        g = self.groups.pop(gid, None)
        if g and keep_items:
            self.loose.update(g['items'])

    def clear(self) -> None:
        """Remove every group and loose item."""
        self.groups.clear()
        self.loose.clear()

    def move_item(self, mid: str, gid: str | None) -> None:
        """Move item `mid` into group `gid`, or to the loose list when `gid` is not a group."""
        it = self.item(mid)
        if it is None:
            return
        self.remove_item(mid)
        if gid and gid in self.groups:
            self.groups[gid]['items'][mid] = it
        else:
            self.loose[mid] = it

    def group_selected(self, mids, name: str | None = None) -> str:
        """Put the items `mids` into a new group and return its id."""
        gid = self.new_group(name)
        for mid in mids:
            self.move_item(mid, gid)
        return gid

    # -- comp / fmo --
    def set_item_comp(self, mid, comp):
        """Set item `mid`'s own compensation, overriding its group's."""
        it = self.item(mid)
        if it is not None:
            it['comp_override'] = comp

    def set_item_fmo(self, mid, fmo):
        """Set item `mid`'s own FMO set, overriding its group's."""
        it = self.item(mid)
        if it is not None:
            it['fmo'] = fmo

    def set_group_comp(self, gid, comp):
        """Set the compensation shared by group `gid`'s items."""
        if gid in self.groups:
            self.groups[gid]['comp'] = comp

    def set_group_fmo(self, gid, fmo):
        """Set the FMO set shared by group `gid`'s items."""
        if gid in self.groups:
            self.groups[gid]['fmo'] = fmo

    def effective_comp(self, mid: str, editor=None) -> dict:
        """The compensation a run applies to item ``mid``: :func:`resolve_comp`
        (row override, else group, else the editor's matrix, else the one
        saved at drop). Tree labels and the run's block read this too."""
        return resolve_comp(self, mid, editor)

    def effective_fmo(self, mid: str):
        """Return the item's FMO set, else its group's, else None."""
        it = self.item(mid)
        if it is None:
            return None
        gid = self.item_group(mid)
        return it.get('fmo') or (self.groups[gid]['fmo'] if gid else None)

    def comp_ready(self, mid: str, editor=None) -> bool:
        """True when item `mid` has a compensation matrix or beads to build one."""
        return self.effective_comp(mid, editor)['ready']

    def unready_items(self, editor=None) -> list[str]:
        """mids that have neither a matrix nor beads."""
        return [mid for mid, _it, _gid in self.all_items()
                if not self.comp_ready(mid, editor)]

    def run_blockers(self, editor=None, mids=None) -> list[str]:
        """mids a run refuses: no matrix, no beads, and not loaded in the
        editor, so the run would reload the file and cluster it
        uncompensated. A loaded sample with no compensation runs as it is
        shown in the editor (see :func:`uncompensated_live`)."""
        mids = [m for m, _it, _g in self.all_items()] if mids is None else mids
        out = []
        for mid in mids:
            c = self.effective_comp(mid, editor)
            if not c['ready'] and not c['live']:
                out.append(mid)
        return out

    def uncompensated_live(self, editor=None, mids=None) -> list[str]:
        """mids that run on a loaded sample with no compensation at all."""
        mids = [m for m, _it, _g in self.all_items()] if mids is None else mids
        return [mid for mid in mids
                if (c := self.effective_comp(mid, editor))['live']
                and not c['ready']]

    # -- display --
    def toggle_item_display(self, mid):
        """Flip whether item `mid` is shown."""
        it = self.item(mid)
        if it is not None:
            it['display'] = not it.get('display', True)

    def toggle_group_display(self, gid):
        """Flip whether group `gid` is shown."""
        if gid in self.groups:
            self.groups[gid]['display'] = not self.groups[gid].get('display', True)

    def set_all_display(self, on: bool):
        """Show or hide every item and group at once."""
        for _mid, it, _gid in self.all_items():
            it['display'] = bool(on)
        for g in self.groups.values():
            g['display'] = bool(on)

    def displayed_items(self) -> list[dict]:
        """Items effectively visible: item ☑ AND (group ☑ if grouped)."""
        out = []
        for _mid, it, gid in self.all_items():
            if gid is not None and not self.groups[gid].get('display', True):
                continue
            if it.get('display', True):
                out.append(it)
        return out

    # -- persistence (M6) --
    def to_dict(self) -> dict:
        """JSON-serialisable snapshot (groups, loose items, run settings).
        Item ``comp_matrix`` ndarrays are stored as nested lists; everything
        else (gate dicts, beads/FMO file lists, flags) is already JSON-safe."""
        from .update import current_version
        return {
            'format': WORKSPACE_FORMAT,
            'schema': WORKSPACE_SCHEMA,
            'app_version': current_version(),   # for the "saved by a newer
            #                                     OpenFlo → update" alert on load
            'title': self.title,
            'run_cfg': dict(self.run_cfg),
            'template': dict(self.template) if self.template else None,
            '_gseq': self._gseq,
            '_mseq': self._mseq,
            'groups': [
                {'gid': g['gid'], 'name': g['name'], 'display': g.get('display', True),
                 'comp': g.get('comp'), 'fmo': g.get('fmo'),
                 'cfg': g.get('cfg', {}),
                 'items': [{'mid': mid, **item_to_json(it)}
                           for mid, it in g['items'].items()]}
                for g in self.groups.values()],
            'loose': [{'mid': mid, **item_to_json(it)}
                      for mid, it in self.loose.items()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkspaceModel:
        """Rebuild a model from a :meth:`to_dict` snapshot.

        The id counters resume past the highest loaded ids so new items never reuse one.
        """
        m = cls(d.get('title', 'Pipeline'))
        if isinstance(d.get('run_cfg'), dict):
            m.run_cfg.update(d['run_cfg'])
        if isinstance(d.get('template'), dict):
            m.template = dict(d['template'])
        for g in d.get('groups', []):
            m.groups[g['gid']] = {
                'gid': g['gid'], 'name': g.get('name', g['gid']),
                'display': g.get('display', True), 'comp': g.get('comp'),
                'fmo': g.get('fmo'), 'cfg': g.get('cfg', {}),
                'items': {i['mid']: item_from_json({k: v for k, v in i.items()
                                                     if k != 'mid'})
                          for i in g.get('items', [])}}
        for i in d.get('loose', []):
            m.loose[i['mid']] = item_from_json({k: v for k, v in i.items()
                                                 if k != 'mid'})
        # Restore id counters so future adds don't collide with loaded ids. The
        # fallback for a legacy file lacking these must be the MAX existing
        # numeric suffix, NOT the item/group COUNT: with gaps left by deletions
        # (e.g. items M1,M2,M5) the count 3 would re-mint M4 then M5 and silently
        # overwrite the existing M5.
        def _suffix(key):
            return int(key[1:]) if key[1:].isdigit() else 0
        all_mids = list(m.loose.keys())
        for _g in m.groups.values():
            all_mids.extend(_g.get('items', {}).keys())
        m._gseq = int(d.get('_gseq', max((_suffix(k) for k in m.groups),
                                         default=0)))
        m._mseq = int(d.get('_mseq', max((_suffix(k) for k in all_mids),
                                         default=0)))
        return m


# ── Subprocess job entry point ─────────────────────────────────────────────
# Run by the workspace panel as:
#   python -m openflo.workspace_run <job.pkl> <result.pkl>
# A guarded __main__ (NOT `python -c`) so Phenograph's own worker pool can
# re-import this module cleanly under Windows `spawn`; the module loads no Tk,
# so neither the child nor its pool workers do.
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    subprocess_main()
