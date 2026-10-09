"""
openflo command-line runner.
----------------------------
Panel-agnostic flow-cytometry analysis: point it at one or more FCS trials,
describe your groups / samples / FMO controls, and it compensates, gates,
clusters, and exports per-sample / per-condition stats, plots and a `.wsp`.

The constants below (example panel, FMO sets, groups) are just defaults used
when you don't pass --groups / --samples / --fmo-sets — replace them with your
own, or supply the flags per run.

Usage (single trial):
    openflo-run --trials /path/to/fcs/ --samples sample_1,sample_2 --out results/

Usage (multiple trials — independent):
    openflo-run --trials /trial1,/trial2,/trial3 --samples sample_1 --batch-mode independent

Usage (multiple trials — concatenated, cells retain trial origin):
    openflo-run --trials /trial1,/trial2 --samples sample_1 --batch-mode concatenate
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
from typing import Literal, overload

import numpy as np


# Cap each process's BLAS thread pool to a fair share of cores. Must run
# BEFORE any numpy / scipy / sklearn import — those libraries read these
# env vars at load time and never re-check. Each parallel sample worker
# inherits this via os.environ, so on a 24-core box with --workers 4 each
# worker gets 6 BLAS threads instead of all spawning 24 (= 96 total threads,
# which is what was triggering the 'OpenBLAS error after 10 retries' kills).
def _cap_blas_threads():
    try:
        workers = 1
        if '--workers' in sys.argv:
            i = sys.argv.index('--workers')
            if i + 1 < len(sys.argv):
                workers = max(1, int(sys.argv[i + 1]))
        cores = os.cpu_count() or 2
        n = max(1, cores // max(1, workers))
        for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(var, str(n))
    except Exception:
        pass
_cap_blas_threads()

# UTF-8 stdout/stderr (so ≥, →, … don't crash a cp1252 console) is applied at
# the top of main(), NOT here.
#
# It used to run at import, and that was a genuine defect: reconfiguring a
# global stream is a side effect no library should have on import. It broke
# pytest-xdist outright — workers talk to the controller over stdout via
# execnet, so merely importing this module inside a worker corrupted the
# channel and killed it with "EOFError: expected 1 bytes, got 0", taking an
# unrelated test in that worker down with it. The same hazard applies to any
# embedding host or notebook that imports openflo.cli as a library.
#
# main() is the right place: it is the only entry point that owns the process's
# streams, and calling it there (before the parser is built) also covers
# `--help`, which an import-time call did not.
# E402 noqa on matplotlib imports: backend selection MUST happen after
# `import matplotlib` and BEFORE `import matplotlib.pyplot`, so these
# three statements have to be interleaved with logic.
import matplotlib  # noqa: E402

from ._console import force_utf8_streams  # noqa: E402

# Headless backend for the main run (unless --show-plots) and for every
# multiprocessing child ('__mp_main__'). Left untouched when imported elsewhere
# (e.g. preview.py) so that interactive previews still open a window.
if __name__ in ('__main__', '__mp_main__') and '--show-plots' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from .pipeline import FlowExperiment, FlowSample, FMOGater, concatenate  # noqa: E402

# ── Gate JSON parsing ─────────────────────────────────────────────────────────
#
# `--gates` accepts a JSON list of gate dicts (see flow_pipeline.gate_to_mask).
# 'threshold' kinds feed gate_overrides (which hooks into the existing FMO
# threshold merge); everything else feeds region_gates and is applied as a
# compound mask via FlowSample.apply_region_gates.

def _parse_gates_arg(s):
    """Returns (gate_overrides_dict, region_gates_list). Empty input → ({}, [])."""
    if not s:
        return {}, []
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        print(f"[gates] JSON parse error: {e}", flush=True)
        return {}, []
    if not isinstance(data, list):
        print(f"[gates] expected a JSON list of gate dicts, got "
              f"{type(data).__name__}", flush=True)
        return {}, []

    overrides = {}
    region_gates = []
    for g in data:
        if not isinstance(g, dict):
            print(f"[gates] dropped malformed gate (not a dict): {g!r}",
                  flush=True)
            continue
        kind = g.get('kind')
        if kind == 'threshold':
            try:
                overrides[str(g['channel'])] = float(g['value'])
            except (KeyError, TypeError, ValueError):
                print(f"[gates] dropped threshold gate with missing/invalid "
                      f"channel or value: {g!r}", flush=True)
                continue
        elif kind in ('interval', 'rect', 'polygon', 'ellipsoid'):
            region_gates.append(g)
        elif kind == 'flowjo_ellipse':
            # A FlowJo ellipse OpenFlo could not convert: it admits no events,
            # and skipping it would admit every event it excluded. Kept, so
            # --export-wsp also writes it back as FlowJo drew it.
            print(f"[gates] FlowJo ellipse {g.get('label') or g.get('name') or ''!r} "
                  "was not converted: it admits no events", flush=True)
            region_gates.append(g)
        else:
            print(f"[gates] unknown gate kind {kind!r} — skipped", flush=True)
    return overrides, region_gates


# ── Example panel + controls (EDIT for your experiment) ──────────────────────
# A generic 3-colour example mapping standard CD markers to detector channels.
# The pipeline is panel-agnostic — nothing here is required; override per run
# with --channels / --groups / --samples / --fmo-sets (or via the GUI).
CD901b = 'FL1-A'
CD902 = 'FL2-A'
CD903 = 'FL3-A'

# Example FMO control sets: each maps a detector channel to the FCS file
# basename holding that channel's single-stain / FMO control. Replace the
# basenames with your own (or pass --fmo-sets). The 'A'/'B' sets illustrate
# two staining runs; groups reference a set by name.
FMOS_SET_A = {
    CD901b: 'fmo_fl1_a',
    CD902: 'fmo_fl2_a',
    CD903: 'fmo_fl3_a',
}
FMOS_SET_B = {
    CD901b: 'fmo_fl1_b',
    CD902: 'fmo_fl2_b',
    CD903: 'fmo_fl3_b',
}

# Named FMO sets. Each set maps a detector channel to the FCS file basename
# that contains the single-stained / no-stain control for that channel. Groups
# reference these by name; extend this dict (via the GUI or by editing) for
# more staining runs without touching any other code.
DEFAULT_FMO_SETS = {
    'Set A': dict(FMOS_SET_A),
    'Set B': dict(FMOS_SET_B),
}

# Example default groups — used ONLY when neither --groups nor --samples is
# supplied. Replace with your own group / sample names.
DEFAULT_GROUPS = [
    {'name': 'Group A', 'samples': ['sample_1', 'sample_2'], 'fmo_set': 'Set A'},
    {'name': 'Group B', 'samples': ['sample_3', 'sample_4'], 'fmo_set': 'Set B'},
]


def _safe_filename(name):
    """Strip path-hostile characters so a group name can be used as a
    folder / file component."""
    return (re.sub(r'[<>:"/\\|?*\s]+', '_', str(name)).strip('_')
            or 'unnamed')


def _group_spec_problems(groups):
    """Why a normalised group spec cannot be run: one line per problem, []
    when it can. A group's name is one side of every comparison and its
    output folder, so two groups with one name had their samples merged
    into one side, a missing sample of the second went unreported, and
    compare_Ctrl_vs_Ctrl was written. Names that differ only in case or in
    characters _safe_filename strips share a folder on Windows / macOS. A
    sample listed twice in a group was processed and counted twice."""
    out = []
    names = [g['name'] for g in groups]
    for nm in dict.fromkeys(n for n in names if names.count(n) > 1):
        out.append(f'{names.count(nm)} groups are named {nm!r}; group names '
                   'must be unique.')
    folder = {}
    for nm in dict.fromkeys(names):
        folder.setdefault(_safe_filename(nm).casefold(), []).append(nm)
    for same in folder.values():
        if len(same) > 1:
            out.append(' and '.join(repr(n) for n in same) + ' would share '
                       f'the output folder {_safe_filename(same[0])!r}; '
                       'rename one.')
    for g in groups:
        twice = list(dict.fromkeys(s for s in g['samples']
                                   if g['samples'].count(s) > 1))
        if twice:
            out.append(f"group {g['name']!r} lists {', '.join(twice)} more "
                       'than once.')
    # Two pairs whose comparison files share a name ('A' vs 'B vs C' and
    # 'A vs B' vs 'C' both make compare_A_vs_B_vs_C): one overwrote the
    # other.
    from itertools import combinations
    by_file = {}
    for a, b in combinations(groups, 2):
        by_file.setdefault(_compare_base(a, b).casefold(), []).append((a, b))
    for pairs in by_file.values():
        if len(pairs) > 1:
            out.append(' and '.join(f"{a['name']!r} vs {b['name']!r}"
                                    for a, b in pairs)
                       + f' would both write {_compare_base(*pairs[0])}.csv;'
                       ' rename a group.')
    return out


def _sample_file_problems(groups, trial_dirs):
    """One line per group that lists one FCS file under two names. fcs_path
    ignores case and also matches a name's tail ('c1' finds expt_c1.fcs),
    so 'c1' and 'C1', or 'c1' and 'expt_c1', passed the name check and the
    file was processed, and counted in the comparison, twice. Files are
    compared by resolved path; a name with no file is left to the run."""
    import contextlib
    import io
    out = []
    for g in groups:
        seen = {}                        # resolved path -> names
        for d in ([g['trial_dir']] if g.get('trial_dir') else trial_dirs):
            for nm in g['samples']:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        path = fcs_path(d, nm)
                except OSError:          # incl. FileNotFoundError
                    continue
                names = seen.setdefault(
                    os.path.normcase(os.path.realpath(path)), [])
                if nm not in names:
                    names.append(nm)
        for path, names in seen.items():
            if len(names) > 1:
                out.append(f"group {g['name']!r} lists "
                           f"{' and '.join(names)}, which are one file: "
                           f'{path}.')
    return out


def _build_export_gate_list(region_gates, gate_overrides):
    """Combine region gates (with their parent_id structure preserved)
    and threshold overrides (added as roots) into a flat list with
    fresh consecutive ids. Used by the --export-wsp path."""
    out = []
    seq = [0]
    def alloc():
        seq[0] += 1
        return f'g{seq[0]}'
    # Two passes so a child gate that appears BEFORE its parent in the list
    # still re-maps correctly: allocate every new id first, then resolve each
    # parent_id against the complete map (a single pass would leave a
    # forward-referencing child rooted at None → wrong hierarchy in FlowJo).
    old_to_new = {}
    prepared = []
    for g in (region_gates or []):
        d = dict(g)
        src_id = d.pop('_import_id', None) or d.pop('id', None)
        d['id'] = alloc()
        if src_id is not None:
            old_to_new[src_id] = d['id']
        prepared.append((d, g.get('parent_id')))
    for d, pid in prepared:
        d['parent_id'] = old_to_new.get(pid) if pid else None
        out.append(d)
    for ch, val in (gate_overrides or {}).items():
        out.append({'kind': 'threshold',
                    'channel': ch,
                    'value': float(val),
                    'id': alloc(),
                    'parent_id': None})
    return out


def _own_spillover(fcs):
    """`(channels, matrix)` of the SPILL the pipeline applied to `fcs`
    (auto_compensate reads the same keyword), or None. It goes inside that
    sample in the .wsp, so FlowJo applies each file's own matrix."""
    try:
        from .pipeline import read_compensation_matrix
        chans, mat = read_compensation_matrix(fcs)
    except Exception as exc:
        print(f"[--export-wsp] {os.path.basename(fcs)}: spillover read failed: {exc}",
              flush=True)
        return None
    return None if chans is None or mat is None else (chans, mat)


def _pipeline_gate_transforms(gates):
    """``{channel: 'logicle'}`` for the fluorescence channels `gates` use.
    The run applies the default logicle to every fluorescence channel
    (FlowSample._classify_channels + apply_transform) BEFORE region gates and
    FMO thresholds, so that is the space their coordinates are in; scatter and
    Time stay linear. WspWriter converts these to FlowJo's linear units."""
    from .pipeline import _is_excluded, _is_scatter
    chans = set()
    for g in gates or []:
        for k in ('channel', 'x_channel', 'y_channel'):
            v = g.get(k)
            if isinstance(v, str):
                chans.add(v)
    return {c: 'logicle' for c in chans
            if not _is_scatter(c) and not _is_excluded(c)}


def _export_pipeline_workspace(out_path, trial_dirs, groups,
                                gate_overrides, region_gates):
    """Emit a FlowJo-compatible .wsp containing the gates that were
    applied during the run, one SampleNode per FCS file processed.

    Resolves each sample name to its FCS path via fcs_path(); samples
    that can't be located are skipped (the rest still write). FMO-derived
    thresholds are NOT included — those are recomputed from the FMO
    control FCS files on every run, so they belong with the inputs the
    pipeline derives them from, not with the per-sample workspace
    export."""
    try:
        from .pipeline import WspWriter
    except ImportError as exc:
        print(f"[--export-wsp] openflo.pipeline import failed: {exc}", flush=True)
        return

    # Normalise first: the raw --groups spec may carry `samples` as a comma
    # string or per-sample dicts ({'name':…,'fmo_set':…}). Without this the loop
    # would iterate a string char-by-char or hit dict sample "names" and resolve
    # ZERO samples (the run itself succeeds because run() normalises its copy).
    groups = _normalise_groups(groups)
    w = WspWriter(cytometer='OpenFlo-pipeline')
    samples_added = 0
    seen = set()
    for grp in (groups or []):
        # Honour a per-group trial_dir (by-day grouping) so filename collisions
        # across day folders resolve to the RIGHT day's FCS; fall back to
        # searching the run's trial dirs for groups without one.
        grp_dir = grp.get('trial_dir')
        search_dirs = [grp_dir] if grp_dir else list(trial_dirs or ['.'])
        for sample_name in grp.get('samples', []):
            fp = None
            last_exc = 'not found'
            for d in search_dirs:
                try:
                    fp = fcs_path(d, sample_name)
                    break
                except FileNotFoundError as exc:
                    last_exc = exc
            if fp is None:
                print(f"[--export-wsp] {sample_name}: {last_exc}", flush=True)
                continue
            if (sample_name, fp) in seen:        # avoid duplicate SampleNodes
                continue
            seen.add((sample_name, fp))
            gates = _build_export_gate_list(region_gates, gate_overrides)
            w.add_sample(
                name=sample_name,
                fcs_path=fp,
                channels=[],     # WspWriter doesn't currently need these
                gates=gates,
                compensation=_own_spillover(fp),
                transforms=_pipeline_gate_transforms(gates))
            samples_added += 1
    if not samples_added:
        print("[--export-wsp] no samples resolved; nothing to write",
              flush=True)
        return

    # Compensation: each sample already carries its own FCS SPILL (the
    # matrix auto_compensate applied to it); WspWriter lists the distinct
    # ones in <Matrices>. A sample without SPILL carries none.
    n_comp = sum(1 for s in w.samples if s.get('compensation') is not None)
    comp_note = ' + spillover' if n_comp else ''
    if n_comp < samples_added:
        print(f"[--export-wsp] {samples_added - n_comp} of {samples_added} "
              "sample(s) have no SPILL in FCS metadata — exported without "
              "a spillover matrix", flush=True)

    try:
        w.write(out_path)
        print(
            f"[--export-wsp] wrote {samples_added} sample(s){comp_note} "
            f"-> {out_path}", flush=True)
        # Where FlowJo will count differently from this run (see WspWriter).
        for msg in w.warnings:
            print(f"[--export-wsp] FlowJo will differ: {msg}", flush=True)
    except Exception as exc:
        print(f"[--export-wsp] write failed: {exc}", flush=True)


def _normalise_groups(groups):
    """Coerce a raw groups spec into a list of canonical dicts.

    Each normalised dict has:
      name       (str)
      samples    (list[str])          — sample tokens, in order
      fmo_set    (str)                — the group's DEFAULT FMO set
      sample_fmo ({name: fmo_set})    — per-sample resolved FMO set

    Per-sample FMO assignment: a `samples` entry may be a plain string
    (uses the group's default `fmo_set`) OR a dict
    ``{'name': ..., 'fmo_set': ...}`` to override just that sample. This
    keeps the common case terse while letting any sample point at a
    different FMO control set than its group-mates. Compensation and
    antibody labels are resolved per sample automatically from each
    FCS ($SPILL / $PnS) in the pipeline, so they don't need a slot here.

    Drops any group whose name or sample list is empty.
    """
    norm = []
    for g in groups or []:
        if not isinstance(g, dict):
            continue
        name = str(g.get('name', '')).strip()
        raw_samples = g.get('samples', [])
        if isinstance(raw_samples, str):
            raw_samples = [s.strip() for s in raw_samples.split(',') if s.strip()]
        group_fmo = str(g.get('fmo_set', '')).strip()
        # Per-sample assignments already resolved by an earlier pass, if any.
        prior = g.get('sample_fmo')
        prior_fmo = dict(prior) if isinstance(prior, dict) else {}

        sample_names = []
        sample_fmo = {}
        for s in raw_samples:
            if isinstance(s, dict):
                sname = str(s.get('name', '')).strip()
                if not sname:
                    continue
                sfmo = str(s.get('fmo_set', '') or group_fmo).strip()
            else:
                sname = str(s).strip()
                if not sname:
                    continue
                # Idempotence: a per-sample override survives the first pass
                # only inside `sample_fmo`, because that pass flattens the
                # dict form to a plain name. Re-normalising an already-
                # normalised group would then reassign the GROUP default and
                # silently discard the override — which would gate that sample
                # against the wrong FMO control. Callers are told to
                # "normalise first"; this makes doing it twice harmless.
                sfmo = str(prior_fmo.get(sname, group_fmo) or '').strip()
            sample_names.append(sname)
            sample_fmo[sname] = sfmo

        if not name or not sample_names:
            continue
        entry = {'name': name, 'samples': sample_names,
                 'fmo_set': group_fmo, 'sample_fmo': sample_fmo}
        # By-day auto-groups tag each group with its source folder so the
        # run resolves that group's samples there (rather than the run's
        # single trial dir). Preserve it through normalisation.
        if g.get('trial_dir'):
            entry['trial_dir'] = str(g['trial_dir'])
        norm.append(entry)
    return norm


# ── By-day auto-grouping ───────────────────────────────────────────────────────

def _discover_day_folders(trial_dirs):
    """Every directory at/under the given paths that DIRECTLY contains at
    least one .fcs file. Lets the user point at a single PARENT folder
    and have each sub-folder auto-become its own day/group, sampled
    independently. De-duplicated, sorted."""
    found, seen = [], set()
    for d in trial_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            if any(f.lower().endswith('.fcs') for f in files):
                rp = os.path.normpath(root)
                if rp not in seen:
                    seen.add(rp)
                    found.append(rp)
    return sorted(found)


def _auto_groups_by_day(trial_dirs):
    """Build one group per discovered day-folder (a folder that directly
    holds FCS files). Each group's samples are that folder's FCS files
    (by filename stem); each group is tagged with its source folder via
    `trial_dir` so the run resolves its samples there. Group name is the
    folder basename, tidied to 'Day N' when a day token is present;
    duplicate names are disambiguated with the parent folder.

    Returns [] when no FCS are found anywhere (caller falls back to the
    legacy default groups)."""
    folders = _discover_day_folders(trial_dirs)
    groups = []
    for d in folders:
        try:
            fcs = sorted(f for f in os.listdir(d)
                         if f.lower().endswith('.fcs'))
        except OSError:
            continue
        if not fcs:
            continue
        samples = [os.path.splitext(f)[0] for f in fcs]
        base = os.path.basename(d.rstrip('/\\')) or d
        m = re.search(r'day\s*[-_ ]?([0-9]+)', base, re.IGNORECASE)
        name = f'Day {m.group(1)}' if m else base
        groups.append({'name': name, 'samples': samples,
                       'fmo_set': '', 'trial_dir': d})

    # Disambiguate clashing group names. Names clash as output folders do:
    # by _safe_filename, without case ('ctrl' and 'Ctrl' were left as they
    # were, and the run refused them). First by the folder's own name where
    # it says more than the group name (exp/day3 and exp/day_3 share every
    # parent, so walking up never told them apart); then by the parent
    # folder, and more of its path while names still clash (X/exp/day3 and
    # Y/exp/day3 both gave 'Day 3 (exp)').
    def key(name):
        return _safe_filename(name).casefold()

    base = [g['name'] for g in groups]
    own = [os.path.basename(g['trial_dir'].rstrip('/\\')) for g in groups]
    parents = [[p for p in os.path.dirname(g['trial_dir'].rstrip('/\\'))
                .replace('\\', '/').split('/') if p and not p.endswith(':')]
               for g in groups]
    keys = [key(b) for b in base]
    clash = {i for i, k in enumerate(keys) if keys.count(k) > 1}
    depth = 0
    while clash and depth <= max(map(len, parents), default=0):
        for i in clash:
            if depth == 0:
                if key(own[i]) != key(base[i]):
                    groups[i]['name'] = f'{base[i]} ({own[i]})'
            elif parents[i][-depth:]:
                groups[i]['name'] = f"{base[i]} ({'/'.join(parents[i][-depth:])})"
        keys = [key(g['name']) for g in groups]
        clash = {i for i in clash if keys.count(keys[i]) > 1}
        depth += 1
    return groups


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_labels(labels_str):
    """Parse 'DetA=LabelA;DetB=LabelB' into {DetA: LabelA, ...}."""
    if not labels_str:
        return {}
    result = {}
    for pair in labels_str.split(';'):
        if '=' in pair:
            det, lbl = pair.split('=', 1)
            result[det.strip()] = lbl.strip()
    return result


def _norm_token(s):
    """Lower-case and strip all non-alphanumerics — so 'Dye/X7',
    'Dye-X7-A' and 'dyex7' all compare equal at the token level."""
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def read_staining_panel(path, channels):
    """Read a staining-panel spreadsheet → ``{detector_channel: CD_label}``.

    The sheet pairs a CD/marker token (e.g. ``CD901b``) with a fluorophore
    token (e.g. ``FL1``, ``Dye/X7``, ``FL2``) on the same row, in two
    columns, in any order, header optional. Each fluorophore is matched to
    the detector channel whose name *contains* it (normalised, case- and
    punctuation-insensitive), so ``FL1`` → ``FL1-A`` and
    ``Dye/X7`` → ``Dye-X7-A``. Returns ``{}`` when nothing matched (the run
    then falls back to whatever ``--labels`` provided, or the raw detector
    names).

    `channels` is the list of real detector channel names to match against
    (typically a sample's ``channel_names`` / DataFrame columns).
    """
    try:
        import pandas as pd
        sheets = pd.read_excel(path, header=None, sheet_name=None)
    except Exception as exc:
        print(f"  [panel] Could not read {path}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {}

    norm_channels = {ch: _norm_token(ch) for ch in channels}

    def match_fluor(fluor):
        nf = _norm_token(fluor)
        if not nf:
            return None
        cands = [ch for ch, nc in norm_channels.items() if nf and nf in nc]
        if not cands:
            return None
        # Prefer area ('-A') detectors and the shortest channel name.
        cands.sort(key=lambda c: ('-a' not in c.lower(), len(c)))
        return cands[0]

    mapping = {}
    cd_re = re.compile(r'^cd\d', re.I)
    for df in sheets.values():
        for _, row in df.iterrows():
            cells = [str(v).strip() for v in row.tolist()
                     if str(v).strip() and str(v).strip().lower() != 'nan']
            cd = next((c for c in cells if cd_re.match(c)), None)
            if not cd:
                continue
            for fl in (c for c in cells if c != cd):
                det = match_fluor(fl)
                if det and det not in mapping:
                    mapping[det] = cd
                    break
    if mapping:
        print(f"  [panel] {os.path.basename(path)} → "
              f"{dict(mapping)}", flush=True)
    else:
        print(f"  [panel] {os.path.basename(path)}: no CD↔fluorophore "
              f"rows matched the data channels.", flush=True)
    return mapping


def find_panel_xlsx(dirs):
    """Locate a staining-panel spreadsheet for the given trial dir(s).

    Searches each dir's descendants first, then walks up a few ancestor
    levels (the panel commonly sits in a study-root folder above the day
    sub-folders). Files whose name mentions 'panel' or 'stain' win ties.
    Returns a path or None.
    """
    candidates = []
    seen = set()

    def add(p):
        ap = os.path.abspath(p)
        if ap not in seen and os.path.isfile(ap):
            base = os.path.basename(ap)
            if base.lower().endswith(('.xlsx', '.xls')) \
                    and not base.startswith('~$'):
                seen.add(ap)
                candidates.append(ap)

    for d in dirs:
        if not d:
            continue
        d = os.path.abspath(d)
        # Descendants of the trial dir.
        for root, _dn, files in os.walk(d):
            for f in files:
                add(os.path.join(root, f))
        # Direct contents of up to 4 ancestor levels.
        cur = d
        for _ in range(4):
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            try:
                for f in os.listdir(parent):
                    add(os.path.join(parent, f))
            except OSError:
                pass
            cur = parent

    candidates.sort(key=lambda p: (
        'panel' not in os.path.basename(p).lower()
        and 'stain' not in os.path.basename(p).lower(),
        len(p)))
    return candidates[0] if candidates else None


# Default marker pairs for the group pair-scatter outputs. CD labels are
# resolved per sample via channel_labels, so these only render when the
# panel / --labels actually assigned them.
DEFAULT_SCATTER_PAIRS = [('CD902', 'CD901b'), ('CD901b', 'CD903'), ('CD902', 'CD903')]


def parse_pairs(pairs_str):
    """Parse 'CD902/CD901b,CD901b/CD903' → [('CD902','CD901b'), ('CD901b','CD903')]."""
    if not pairs_str:
        return None
    out = []
    for chunk in pairs_str.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        for sep in ('/', 'vs', ':', '-'):
            if sep in chunk:
                x, y = chunk.split(sep, 1)
                out.append((x.strip(), y.strip()))
                break
    return out or None


def _resolve_pair(s, xlabel, ylabel):
    """Resolve a CD-label pair to (x_detector, y_detector) for one sample,
    or (None, None) if either channel isn't present."""
    try:
        return s._resolve(xlabel), s._resolve(ylabel)
    except KeyError:
        return None, None


def _intensity_axes(ax, resolved):
    """Tick the overlay's axes at round intensities. The samples' STORED
    values are drawn (logicle for fluorescence), which matplotlib ticked
    0.2 .. 1.0 where the intensities are 100 .. 10^5. One axis takes the
    transform its samples share; if they differ (no shared intensity for a
    position), it keeps the stored ticks and its label says so."""
    from .scales import intensity_ticker
    for i, axis in enumerate((ax.xaxis, ax.yaxis)):
        specs = [(getattr(s, 'data_transforms', None) or {}).get(chans[i])
                 for s, *chans in resolved]
        if any(sp != specs[0] for sp in specs):
            axis.set_label_text(f'{axis.get_label_text()} '
                                '(stored display units)')
            continue
        ticker = intensity_ticker(specs[0], 'stored') if specs[0] else None
        if ticker is not None:
            axis.set_major_locator(ticker[0])
            axis.set_major_formatter(ticker[1])


def save_group_pair_scatters(samples, label, out_dir, pairs=None,
                             max_points=20_000, random_state=42):
    """Per marker pair, write two figures for a *group* of samples:

      • ``<group>_<X>_<Y>_overlay.png`` — every sample on one axes, coloured
        by sample, with a legend (the 'stacked' overlay view).
      • ``<group>_<X>_<Y>_grid.png`` — one density panel per sample on shared
        axis limits (small-multiples comparison).

    Pairs that no sample can resolve (panel labels absent) are skipped.
    """
    if not samples:
        return
    pairs = pairs or DEFAULT_SCATTER_PAIRS
    os.makedirs(out_dir, exist_ok=True)
    token = _safe_filename(label)
    base_cmap = plt.get_cmap('tab10' if len(samples) <= 10 else 'tab20')

    for xlabel, ylabel in pairs:
        resolved = []
        for s in samples:
            xch, ych = _resolve_pair(s, xlabel, ylabel)
            if xch and ych:
                resolved.append((s, xch, ych))
        if not resolved:
            print(f"  [pairs] {label}: '{xlabel} vs {ylabel}' skipped "
                  f"(channels not labelled on any sample).", flush=True)
            continue
        pair_token = f'{_safe_filename(xlabel)}_{_safe_filename(ylabel)}'

        # Shared, outlier-robust axis limits across all samples in the group.
        xc = np.concatenate([s.data[xch].to_numpy(dtype=float)
                             for s, xch, _ in resolved])
        yc = np.concatenate([s.data[ych].to_numpy(dtype=float)
                             for s, _, ych in resolved])
        xc = xc[np.isfinite(xc)]
        yc = yc[np.isfinite(yc)]
        xlim = (float(np.percentile(xc, 0.5)),
                float(np.percentile(xc, 99.5))) if xc.size else None
        ylim = (float(np.percentile(yc, 0.5)),
                float(np.percentile(yc, 99.5))) if yc.size else None

        # ── Overlay (all samples, coloured by sample) ──────────────────────
        try:
            import matplotlib.patches as mpatches
            fig, ax = plt.subplots(figsize=(7, 6))
            handles = []
            for i, (s, xch, ych) in enumerate(resolved):
                d = s.data.dropna(subset=[xch, ych])
                if max_points and len(d) > max_points:
                    d = d.sample(max_points, random_state=random_state)
                color = base_cmap(i % base_cmap.N)
                ax.scatter(d[xch].to_numpy(dtype=float),
                           d[ych].to_numpy(dtype=float),
                           s=1.0, alpha=0.4, color=color, linewidths=0,
                           rasterized=True)
                handles.append(mpatches.Patch(color=color, label=s.name))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            _intensity_axes(ax, resolved)
            if xlim:
                ax.set_xlim(*xlim)
            if ylim:
                ax.set_ylim(*ylim)
            ax.legend(handles=handles, fontsize=8, framealpha=0.8, loc='best')
            ax.set_title(f'{label} — {xlabel} vs {ylabel} (overlay)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f'{token}_{pair_token}_overlay.png'), dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  [!] overlay {xlabel}/{ylabel} ({label}) failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)

        # ── Grid (one density panel per sample, shared limits) ─────────────
        try:
            n = len(resolved)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 4.5 * nrows),
                                     squeeze=False)
            flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
            for i, (s, xch, ych) in enumerate(resolved):
                s.plot(xch, ych, color_by='density', ax=flat[i], title=s.name)
                if xlim:
                    flat[i].set_xlim(*xlim)
                if ylim:
                    flat[i].set_ylim(*ylim)
            for j in range(n, len(flat)):
                flat[j].set_visible(False)
            fig.suptitle(f'{label} — {xlabel} vs {ylabel} (per sample)',
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir,
                        f'{token}_{pair_token}_grid.png'), dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  [!] grid {xlabel}/{ylabel} ({label}) failed: "
                  f"{type(exc).__name__}: {exc}", flush=True)


def _trial_label(trial_dir):
    return os.path.basename(trial_dir.rstrip('/\\')) or trial_dir


def _calibrate_bead_micron(trial_dir, fmo_sets, bead_um=8.0):
    """Try every FMO control file in `fmo_sets` for `trial_dir`; the
    first one we can read gives us the bead population's median FSC-A,
    which we use to derive the µm-per-FSC scale factor.

    Returns (factor: float | None, bead_path: str | None). `factor` is
    `bead_um / median_FSC_A` — multiply any raw FSC-A value by `factor`
    to get its physical size in µm. None when no readable bead file
    could be found.
    """
    for set_name, mapping in (fmo_sets or {}).items():
        for ch, fname in (mapping or {}).items():
            try:
                path = fcs_path(trial_dir, fname)
            except FileNotFoundError:
                continue
            try:
                bead = FlowSample(path)
            except Exception as exc:
                print(f"  [Calibration] Could not read {path}: {exc}")
                continue
            fsc = next((c for c in bead.data.columns
                        if c.upper().startswith('FSC')
                        and c.upper().endswith('-A')), None)
            if fsc is None:
                continue
            vals = bead.data[fsc].astype(float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 100:
                continue
            median = float(np.median(vals))
            if median <= 0:
                continue
            factor = bead_um / median
            print(f"[Calibration] Bead file: {os.path.basename(path)} "
                  f"({set_name} / {ch})", flush=True)
            print(f"[Calibration] Median {fsc} = {median:,.0f} → "
                  f"{factor*1e6:.3f} µm per 10^6 FSC units "
                  f"(assuming {bead_um} µm beads).", flush=True)
            return factor, path
    return None, None


def _gpu_present():
    """True if an NVIDIA GPU is visible via nvidia-smi (driver-level check).
    Independent of whether RAPIDS cuML is installed."""
    try:
        import subprocess
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=6,
                             creationflags=flags)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def _find_unstained(fcs_dir, group='early'):
    prefer = 'lateunstained' if group == 'late' else 'unstained'
    for f in sorted(os.listdir(fcs_dir)):
        base = f.lower()
        if not base.endswith('.fcs'):
            continue
        if f'_{prefer}_' in base or base.endswith(f'_{prefer}.fcs'):
            return os.path.join(fcs_dir, f)
    for f in sorted(os.listdir(fcs_dir)):
        base = f.lower()
        if 'unstained' in base and base.endswith('.fcs'):
            if group == 'early' and 'late' in base:
                continue
            return os.path.join(fcs_dir, f)
    return None


@overload
def _scan_for_token(search_dir: str, token: str, recurse: bool,
                    all_matches: Literal[False] = False) -> str | None: ...
@overload
def _scan_for_token(search_dir: str, token: str, recurse: bool,
                    all_matches: Literal[True]) -> list[str]: ...
def _scan_for_token(search_dir, token, recurse, all_matches=False):
    """Find FCS files whose name matches `token`. A match is any of:
      • exact filename stem  (`<token>.fcs`)        — by-day auto-groups
        hand the whole stem here, so this is the primary path for them;
      • `_<token>_` substring                        — short logical names
      • ends with `_<token>.fcs`                     — trailing short name
    Returns the first match (or every match when `all_matches=True`);
    None / [] when nothing matches."""
    token_l = token.lower()
    matches: list[str] = []
    try:
        if recurse:
            walker = os.walk(search_dir)
        else:
            walker = [(search_dir, [], os.listdir(search_dir))]
    except OSError:
        return matches if all_matches else None
    for root, _dirs, files in walker:
        for f in sorted(files):
            if not f.lower().endswith('.fcs'):
                continue
            b = f.lower()
            stem = b[:-4]   # strip '.fcs'
            if (stem == token_l
                    or f'_{token_l}_' in b
                    or b.endswith(f'_{token_l}.fcs')):
                path = os.path.join(root, f)
                if not all_matches:
                    return path
                matches.append(path)
    return matches if all_matches else None


def fcs_path(fcs_dir: str, name: str) -> str:
    """Locate the FCS file for a logical sample name inside `fcs_dir`.

    `name` accepts a forward-slash subfolder prefix:
      • 'm1'         → search fcs_dir directly
      • 'day1/m1'    → search fcs_dir/day1/ directly (no fallback)
      • 'study/day1/m1' → search fcs_dir/study/day1/ directly

    When `name` has no slash and isn't found at the root of fcs_dir, we
    fall back to a recursive search. If that recursive search hits more
    than one match the first one is used and a warning lists the rest —
    the user can disambiguate by adding a subfolder prefix to the sample
    name in their group definition.
    """
    name = str(name).strip().replace('\\', '/')

    # Explicit subfolder path → search ONLY there
    if '/' in name:
        subfolder, token = name.rsplit('/', 1)
        search_dir = os.path.join(fcs_dir, subfolder)
        path = _scan_for_token(search_dir, token, recurse=False)
        if path:
            return path
        raise FileNotFoundError(
            f"Could not find FCS file for '{name}' in {search_dir}")

    # Bare token → try root first
    path = _scan_for_token(fcs_dir, name, recurse=False)
    if path:
        return path

    # Recursive fallback (handles FMO files placed in a controls/
    # subfolder, etc.).
    matches = _scan_for_token(fcs_dir, name, recurse=True, all_matches=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find FCS file for '{name}' in {fcs_dir}")
    if len(matches) > 1:
        print(f"  [!] '{name}' matched {len(matches)} files; using "
              f"{os.path.relpath(matches[0], fcs_dir)}")
        for extra in matches[1:]:
            print(f"      also: {os.path.relpath(extra, fcs_dir)}")
        print(f"      Disambiguate by using a subfolder prefix in the "
              f"sample name (e.g. 'day1/{name}').")
    return matches[0]


def build_fmo_thresholds(fcs_dir, fmo_map, label, fmo_percentile=99.5):
    print(f"\n-- FMO thresholds ({label})")
    gater   = FMOGater()
    missing = []

    for ch, name in fmo_map.items():
        try:
            gater.add_fmo(ch, fcs_path(fcs_dir, name))
        except FileNotFoundError:
            missing.append(ch)

    if missing:
        group = 'late' if 'late' in label.lower() else 'early'
        print(f"\n  [!] No FMO found for: {missing}")
        unstained = _find_unstained(fcs_dir, group)
        if unstained:
            print(f"  [!] WARNING: Falling back to unstained control for {missing}")
            print(f"      File : {os.path.basename(unstained)}")
            print("      Valid only when those channels have no spectral spillover.")
            for ch in missing:
                gater.add_fmo(ch, unstained, is_fallback=True)
        else:
            print(f"  [!] No unstained control found either. Channels will not be gated: {missing}")

    if not gater.fmos:
        return {}
    gater.prepare()
    return gater.compute(percentile=fmo_percentile)


def _find_scatter_channel(s, prefix):
    """Pick the area-version scatter channel matching prefix (FSC/SSC).
    Prefers '-A' channels; falls back to the first match."""
    prefix_u = prefix.upper()
    area = [c for c in s.scatter_channels
            if c.upper().startswith(prefix_u) and '-A' in c.upper()]
    if area:
        return area[0]
    other = [c for c in s.scatter_channels if c.upper().startswith(prefix_u)]
    return other[0] if other else None


def save_plots(s, out_dir):
    from itertools import combinations
    os.makedirs(out_dir, exist_ok=True)
    # Sample name may include a subfolder prefix (e.g. 'day1/m1'); flatten
    # it for use in PNG filenames so we don't accidentally create a
    # 'day1/' directory inside out_dir.
    fname_token = _safe_filename(s.name)

    channels = s.fluor_channels
    pairs    = list(combinations(channels, 2))
    if pairs:
        try:
            ncols = min(3, len(pairs))
            nrows = (len(pairs) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 5.5 * nrows),
                                     squeeze=False)
            flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
            for i, (xcol, ycol) in enumerate(pairs):
                s.plot(xcol, ycol, color_by='cluster', ax=flat[i])
            for j in range(len(pairs), len(flat)):
                flat[j].set_visible(False)
            fig.suptitle(f'{s.name} — pairwise scatter (cluster)', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{fname_token}_scatter.png'),
                        dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  [!] scatter save failed for {s.name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # FSC vs SSC scatter, colored by cluster — separate file so it's easy to
    # find. Uses the raw (un-transformed) scatter channels, which is the
    # standard flow-cytometry view for population gating.
    fsc = _find_scatter_channel(s, 'FSC')
    ssc = _find_scatter_channel(s, 'SSC')
    if fsc and ssc and 'cluster' in s.data.columns:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            s.plot(fsc, ssc, color_by='cluster', ax=ax,
                   title=f'{s.name} — FSC vs SSC (cluster)')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'{fname_token}_fsc_ssc.png'),
                        dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"  [!] FSC/SSC save failed for {s.name}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # Cluster-median heatmap. Wrapped so a missing 'cluster' column or any
    # other matplotlib hiccup doesn't bring down the rest of save_plots
    # (the scatter / fsc_ssc above) — that asymmetric failure mode was the
    # cause of the "compare CSV exists but heatmap doesn't" report.
    try:
        ax = s.cluster_heatmap()
        if ax is not None:
            out_pth = os.path.join(out_dir, f'{fname_token}_heatmap.png')
            ax.figure.savefig(out_pth, dpi=150)
            plt.close(ax.figure)
            print(f"  [save] heatmap: {out_pth}", flush=True)
        else:
            print(f"  [!] heatmap skipped for {s.name} — "
                  f"cluster_heatmap() returned None (no 'cluster' column?)",
                  flush=True)
    except Exception as exc:
        print(f"  [!] heatmap save failed for {s.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)


def run_group_umap(samples, label, out_dir, random_state=42):
    print(f"\n-- UMAP: {label} ({len(samples)} samples)")
    combined_data = concatenate(samples)
    ref = copy.copy(samples[0])
    ref.data = combined_data
    ref.name = label
    ref.umap_coords = None
    ref.run_umap(sample_n=80_000, random_state=random_state)

    # Locate FSC/SSC channels — add as a third panel when present so we get
    # both the UMAP view *and* the classic flow-cytometry FSC vs SSC view of
    # the same cluster assignment.
    fsc = _find_scatter_channel(ref, 'FSC')
    ssc = _find_scatter_channel(ref, 'SSC')
    have_scatter = bool(fsc and ssc)

    n_panels = 3 if have_scatter else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    axes = list(axes) if n_panels > 1 else [axes]

    ref.plot_umap(color_by='cluster',       ax=axes[0],
                  title=f'{label} -- UMAP (cluster)')
    ref.plot_umap(color_by='sample_origin', ax=axes[1],
                  title=f'{label} -- UMAP (sample)')
    if have_scatter:
        ref.plot(fsc, ssc, color_by='cluster', ax=axes[2],
                 title=f'{label} -- FSC vs SSC (cluster)')

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{label}_umap.png'), dpi=150)
    if not _KEEP_FINAL_PLOTS:
        plt.close(fig)
    return ref


def _already_done(out_dir, group, name):
    return os.path.exists(os.path.join(out_dir,
                                       _safe_filename(group),
                                       f'{_safe_filename(name)}_stats.csv'))


# When --show-plots is on, the final group-level figures (UMAP + condition
# comparison) are *not* closed after saving so that the trailing plt.show()
# in main() actually has something to display. Per-sample figures are still
# always closed — 4 samples × 3 figures each is too many windows.
_KEEP_FINAL_PLOTS = False


def _show_saved_group_plots(out_dir):
    """Re-open every previously-saved figure as a matplotlib window so
    --show-plots is useful on a re-run that hit --skip-existing for
    everything (and therefore drew nothing fresh).

    Includes group-level outputs (UMAP, condition comparison) AND
    per-sample outputs (cluster scatter, FSC/SSC scatter, cluster-median
    heatmap). The per-sample set was historically excluded because a
    big run produces dozens of files; if you need the lean view, run
    without --show-plots and open files from the explorer instead.
    Returns the number of figures opened.
    """
    from pathlib import Path
    p = Path(out_dir)
    if not p.exists():
        return 0
    seen = []
    for pat in ('*_umap.png',
                '*_heatmap.png',
                '*_scatter.png',
                '*_fsc_ssc.png',
                'compare_*.png',
                'condition_comparison.png'):
        # Not from a comparison staging folder: its PNGs are not this
        # run's set (a locked run leaves its new set there).
        seen.extend(sorted(f for f in p.rglob(pat)
                           if not any(part.startswith(_STAGING_PREFIX)
                                      for part in f.relative_to(p).parts)))
    if not seen:
        return 0
    print(f"\n[Show plots] Re-opening {len(seen)} saved figure(s) "
          f"from {out_dir}", flush=True)
    for png in seen:
        try:
            img = plt.imread(str(png))
            h, w = img.shape[:2]
            # Render at a reasonable size while keeping aspect ratio.
            target_w = 13.0
            fig = plt.figure(figsize=(target_w, target_w * h / max(w, 1)))
            ax  = fig.add_subplot(1, 1, 1)
            ax.imshow(img)
            ax.set_axis_off()
            fig.suptitle(str(png.relative_to(p)), fontsize=9)
            fig.tight_layout()
        except Exception as e:
            print(f"  [!] Could not load {png}: {e}", flush=True)
    return len(seen)


# ── Parallel sample processing ────────────────────────────────────────────────

def _load_task_sample(task):
    """Load every input FCS for one output sample, QC, compensate, transform
    and gate it, and concatenate the files when there is more than one
    (tagging each cell with its trial-of-origin). Does NOT cluster. Returns
    the FlowSample, or None when the task has no input file.

    Separate from the clustering so --skip-existing can reload a finished
    sample's events for the group comparison without clustering it again."""
    name = task['name']
    pieces = []
    for origin, path in task['paths']:
        s = FlowSample(path)
        s.run_qc()
        # Debris + doublet filtering, applied to raw (un-transformed)
        # FSC channels. Order matters: drop debris first so the
        # doublet ratio's median is computed on cell-sized events.
        if task.get('debris_min_fsc') is not None:
            s.filter_debris(min_fsc=task['debris_min_fsc'])
        if task.get('doublet_tol'):
            s.filter_doublets(tol=float(task['doublet_tol']))
        s.auto_compensate()
        s.apply_transform()
        if task['labels']:
            s.set_labels(task['labels'])
        s.apply_threshold_gates(task['thresholds'])
        # Compound region gates (rect/polygon/interval) — filter events,
        # applied after transform so coordinates match post-logicle space.
        if task.get('region_gates'):
            s.apply_region_gates(task['region_gates'])
        s.name = origin                     # → sample_origin column
        pieces.append(s)
    if not pieces:
        return None
    combined = copy.copy(pieces[0])
    if len(pieces) > 1:
        combined.data = concatenate(pieces)
    combined.name = name      # display / filename stem
    # Batch correction (CytoNorm) — apply the pre-fitted model after
    # transform, before clustering, so clusters/UMAP/stats use the
    # batch-aligned values.
    if task.get('cytonorm'):
        from .pipeline import CytoNorm
        combined.data = CytoNorm.from_dict(task['cytonorm']).apply(
            combined.data, task.get('batch_id', ''))
    return combined


def _cluster_settings(k, n_jobs, max_events, vram_admission_gb, random_state,
                      cluster_cfg=None):
    """The clustering settings every task of a run shares; the comparison's
    shared clustering uses the same ones."""
    cc = cluster_cfg or {}
    return {'k': k, 'n_jobs': n_jobs,
            'method': cc.get('method', 'phenograph'),
            'resolution': cc.get('resolution', 1.0),
            'n_metaclusters': cc.get('n_metaclusters', 10),
            'reproducible': cc.get('reproducible', True),
            'max_events': max_events,
            'vram_admission_gb': vram_admission_gb,
            'random_state': random_state}


# The events a method clusters when --max-events is not given: Leiden
# partitions at most this many and assigns the rest to their nearest
# partitioned event; FlowSOM trains its SOM on at most this many.
# PhenoGraph has no cap of its own.
_METHOD_EVENT_CAP = {'leiden': 200_000, 'flowsom': 50_000}


def _cluster_by_method(s, task):
    """Cluster `s` in place with the task's --cluster-method settings and
    alias the method's label column to 'cluster'."""
    # The seed is held constant across every sample in a trial so the
    # subsampling cuts (when max_events truncates) and any subsequent
    # UMAP are reproducible. Workers can run in parallel safely —
    # each spawns its own RNG from the same seed value.
    method = task.get('method', 'phenograph')
    mev = task.get('max_events')
    seed = task.get('random_state', 42)
    if method == 'leiden':
        s.run_leiden(n_neighbors=task['k'],
                     resolution=task.get('resolution', 1.0),
                     max_events=mev or _METHOD_EVENT_CAP['leiden'],
                     random_state=seed)
        if 'leiden' in s.data.columns:
            s.data['cluster'] = s.data['leiden']
    elif method == 'flowsom':
        s.run_flowsom(n_metaclusters=task.get('n_metaclusters', 10),
                      max_events=mev or _METHOD_EVENT_CAP['flowsom'],
                      seed=seed)
        if 'flowsom_meta' in s.data.columns:
            s.data['cluster'] = s.data['flowsom_meta']
    else:
        s.cluster(k=task['k'], n_jobs=task['n_jobs'],
                  max_events=mev,
                  vram_admission_gb=task.get('vram_admission_gb', 1.0),
                  random_state=seed,
                  reproducible=bool(task.get('reproducible', True)))
    return s


def _process_sample_task(task):
    """Top-level worker (picklable — runs in a child process).

    Loads every input FCS for one output sample, concatenates them when there
    is more than one (tagging each cell with its trial-of-origin), gates and
    clusters. Returns (key, FlowSample | None, error_msg | None) where
    `key` uniquely identifies the task (group + name) so the dispatcher
    can bucket results correctly even when two groups (e.g. two days)
    contain identically-named FCS files.
    """
    name = task['name']
    key  = task.get('key', name)
    try:
        combined = _load_task_sample(task)
        if combined is None:
            return (key, None, 'no input files found')
        _cluster_by_method(combined, task)
        return (key, combined, None)
    except Exception as e:
        return (key, None, f'{type(e).__name__}: {e}')


def _sample_stats(s):
    """Per-cluster stats for one sample, plus a ``pct_pos_<marker>`` column
    for every threshold gate applied to it (FMO-derived or a --gates
    threshold): the % of that cluster's events above the cut.

    apply_threshold_gates only flags events (``<channel>_pos``) and filters
    nothing, and no writer read the flags, so a threshold changed no file the
    run wrote: every output was byte-identical with and without one."""
    freq = s.cluster_frequencies()
    thresholds = getattr(s, 'thresholds', None) or {}
    if freq.empty or not thresholds or 'cluster' not in s.data.columns:
        return freq
    for ch in thresholds:
        try:
            col = s._resolve(ch)
        except KeyError:
            continue
        flag = f'{col}_pos'
        if flag not in s.data.columns:
            continue
        pct = s.data.groupby('cluster')[flag].mean() * 100
        freq[f'pct_pos_{s.channel_labels.get(col, col)}'] = (
            freq['cluster'].map(pct).to_numpy(dtype=float))
    return freq


def _run_sample_tasks(tasks, workers, admission_gb, report, failures=None):
    """Process sample tasks — in parallel when workers > 1, with admission
    control: a new worker is only submitted when free RAM ≥ `admission_gb`.
    In-flight workers are NEVER paused (suspending would not free RAM) — they
    complete naturally; only the *launcher* waits. Each completed sample's
    plots + stats are written immediately so --skip-existing works after a
    watchdog restart. Returns {group_name: [FlowSample, ...]} keyed by the
    'group' field on each task; `failures`, when given, gets each failed
    task's key -> error."""
    results  = {}                                  # group_name -> [samples]
    # Dispatch key MUST be unique per task. Bare sample name collides
    # when two groups (e.g. two day-folders) hold identically-named FCS
    # — that silently bucketed both into one group. Key by group+name.
    for t in tasks:
        t['key'] = f"{t['group']}\x1f{t['name']}"
    by_key   = {t['key']: t for t in tasks}

    def _handle(result):
        key, s, err = result
        t = by_key[key]
        nm = t['name']                              # for display + filenames
        if s is None:
            report(f"Failed: {nm} ({err})")
            if failures is not None:
                failures[key] = err
            # Detect the unmistakable Windows commit-charge / pagefile signature
            # and tell the user how to fix it. RAM might look fine in psutil
            # while commit is exhausted — the watchdog won't catch this.
            e = (err or '').lower()
            if ('paging file' in e or 'pagefile' in e
                    or 'winerror 1455' in e or 'winerror 8' in e
                    or 'not enough memory resources' in e):
                print(
                    "\n  [!] WINDOWS PAGEFILE TOO SMALL.\n"
                    "      Your physical RAM is fine but the OS commit charge\n"
                    "      ran out. Increase the pagefile:\n"
                    "        Win → View advanced system settings\n"
                    "        Performance → Settings → Advanced → Virtual memory → Change\n"
                    "        Uncheck 'Automatically manage' → pick C: → Custom size\n"
                    "        Initial + Maximum ≥ 32768 MB (32 GB) → Set → OK → restart\n",
                    flush=True)
            return
        group_dir = _safe_filename(t['group'])
        file_nm   = _safe_filename(nm)
        save_plots(s, os.path.join(t['out'], group_dir))
        stats_path = os.path.join(t['out'], group_dir, f"{file_nm}_stats.csv")
        _sample_stats(s).to_csv(stats_path, index=False)
        logging.getLogger(__name__).info(f"  Stats → {stats_path}")
        results.setdefault(t['group'], []).append(s)
        report(f"Processed {nm}  ({len(s.data):,} events)")

    if not tasks:
        return results

    if workers <= 1 or len(tasks) == 1:
        for t in tasks:
            _handle(_process_sample_task(t))
        return results

    # Parallel path with admission control.
    import time
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]

    n = min(workers, len(tasks))
    print(f"\n-- Processing {len(tasks)} sample(s) on {n} parallel worker(s) "
          f"({tasks[0]['n_jobs']} clustering thread(s) each).", flush=True)
    if admission_gb > 0 and psutil is not None:
        print(f"   Admission: a new worker starts only when ≥ {admission_gb:.1f} GB RAM is free.",
              flush=True)
    else:
        print("   Admission: disabled (psutil unavailable or threshold ≤ 0).",
              flush=True)

    waiting     = False
    no_progress = 0       # consecutive 2 s polls with no completion

    # ProcessPoolExecutor workers are NON-daemonic, so Phenograph can still
    # spawn its own Jaccard-kernel pool inside each worker. (multiprocessing.Pool
    # workers are daemonic and would crash with "daemonic processes are not
    # allowed to have children".)
    with ProcessPoolExecutor(max_workers=n) as ex:
        pending   = list(tasks)
        in_flight = {}                            # future → sample name

        while pending or in_flight:
            # ── Top up workers, admission-gated ──────────────────────────
            while pending and len(in_flight) < n:
                if admission_gb > 0 and psutil is not None:
                    free_gb = psutil.virtual_memory().available / (1024 ** 3)
                else:
                    free_gb = float('inf')
                if free_gb < admission_gb:
                    if not in_flight:
                        if not waiting:
                            print(f"   [Admission] Free RAM {free_gb:.1f} GB "
                                  f"< {admission_gb:.1f} GB — waiting …",
                                  flush=True)
                            waiting = True
                        time.sleep(3)
                        continue
                    break
                if waiting:
                    print(f"   [Admission] Free RAM {free_gb:.1f} GB ≥ "
                          f"{admission_gb:.1f} GB — resuming submission.",
                          flush=True)
                    waiting = False
                t = pending.pop(0)
                fut = ex.submit(_process_sample_task, t)
                in_flight[fut] = t['name']   # display name for status only

            if not in_flight:
                continue

            # ── Drain completions (timeout lets us re-poll RAM regularly) ─
            try:
                done, _ = wait(in_flight.keys(),
                               timeout=2.0, return_when=FIRST_COMPLETED)
            except Exception as e:
                # Pool broken etc. — surface and stop, don't hang silently.
                for nm in in_flight.values():
                    report(f"Failed: {nm} (pool broken: {e})")
                return results

            if done:
                no_progress = 0
                for fut in done:
                    nm = in_flight.pop(fut)
                    try:
                        _handle(fut.result())
                    except Exception as e:
                        report(f"Failed: {nm} ({type(e).__name__}: {e})")
            else:
                no_progress += 1
                # 90 polls × 2 s = ~3 minutes of silence → log a warning.
                # Repeats every 3 minutes so a hang can't go undetected.
                if no_progress % 90 == 0:
                    free_gb = (psutil.virtual_memory().available / (1024**3)
                               if psutil is not None else 0)
                    stuck = ', '.join(in_flight.values())
                    print(f"   [Stuck?] {no_progress*2//60} min with no "
                          f"completions. In flight: {stuck}. "
                          f"Free RAM {free_gb:.1f} GB. "
                          f"If this persists, Cancel and retry with fewer "
                          f"workers or a smaller --max-events cap.",
                          flush=True)

    return results


# ── Independent mode ──────────────────────────────────────────────────────────

def _build_fmo_thresholds_for_groups(trial_dir, groups, fmo_sets,
                                     fmo_percentile, gate_overrides, report,
                                     trial_label_prefix=''):
    """Build the FMO threshold dict for each FMO set that's actually used by
    at least one group. Returns {fmo_set_name: thresholds_dict}.
    `report` is called once per unique set (for accurate step counting)."""
    # Collect every FMO set referenced by any group default OR any
    # per-sample override. Skip the empty set ('' = ungated): the task
    # lookup returns {} for it without needing an entry here.
    used_sets = []
    for g in groups:
        for s in [g.get('fmo_set', '')] + list(g.get('sample_fmo', {}).values()):
            if s and s not in used_sets:
                used_sets.append(s)

    fmo_thresh = {}
    prefix = f'[{trial_label_prefix}] ' if trial_label_prefix else ''
    for set_name in used_sets:
        if set_name not in fmo_sets:
            report(f'{prefix}FMO set "{set_name}" not defined — '
                   f'groups using it will run ungated')
            fmo_thresh[set_name] = {}
            continue
        report(f'{prefix}Building FMO thresholds ({set_name}) …')
        thresh = build_fmo_thresholds(trial_dir, fmo_sets[set_name],
                                      set_name, fmo_percentile)
        if gate_overrides:
            thresh.update(gate_overrides)
        fmo_thresh[set_name] = thresh
    # --gates threshold overrides must apply to EVERY sample, including ungated
    # ones (fmo_set='') — which are the norm in by-day auto-grouping. Without a
    # '' entry the task lookup returns {} and the thresholds are silently dropped
    # (yet --export-wsp would still record them). Seed the empty set here.
    if gate_overrides:
        fmo_thresh.setdefault('', {}).update(gate_overrides)
    return fmo_thresh


_SHARED_IDS_NOTE = ('shared: one clustering of every sample; phenotypes in '
                    'compare_shared_clusters.csv; NOT the <name>_stats.csv '
                    'cluster ids')


# The only names openflo-run's comparison step writes (or wrote: the
# manifest of an earlier build). Anything else called compare_* is not
# ours: matching compare_*.csv deleted openflo-compare's compare_report.csv.
_COMPARE_OWN = re.compile(r'compare_(shared_clusters\.csv|manifest\.json'
                          r'|.+_vs_.+\.(csv|png))', re.IGNORECASE)


def _compare_files(out):
    """The comparison files openflo-run made in `out`: compare_shared_
    clusters.csv and compare_<A>_vs_<B>.csv / .png (plus an old
    compare_manifest.json), matched without case, as Windows and macOS
    file names are."""
    return sorted(f for f in os.listdir(out)
                  if _COMPARE_OWN.fullmatch(f)
                  and os.path.isfile(os.path.join(out, f)))


def _compare_base(g_a, g_b):
    """File stem of one pair's comparison: compare_<A>_vs_<B>."""
    return _safe_filename(f'compare_{g_a["name"]}_vs_{g_b["name"]}')


def _compare_group_pair(g_a, g_b, results, trial_out, report,
                        trial_label_prefix=''):
    """Run condition-comparison between two groups. Called for every pair.
    `results` maps group name → samples labelled by ONE clustering shared
    by every sample (see _shared_cluster_labels). Returns None when it
    wrote compare_<A>_vs_<B>.csv / .png, else why it wrote nothing."""
    prefix = f'[{trial_label_prefix}] ' if trial_label_prefix else ''
    report(f'{prefix}Compare: {g_a["name"]} vs {g_b["name"]} …')
    sa = results.get(g_a['name'], [])
    sb = results.get(g_b['name'], [])
    if not sa or not sb:
        print(f'  [!] {g_a["name"]} or {g_b["name"]} has no samples to '
              'compare — comparison skipped')
        return 'had no samples'
    # Key each sample by side + position, never by its name: by-day groups
    # hold the SAME filename in every day folder, and a name-keyed dict kept
    # only the last one, so the CSV had one condition holding the other
    # day's data. compare_conditions matches samples by `.name`, so each
    # gets a shallow copy carrying its key.
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.samples = {}
    keys = {'A': [], 'B': []}
    for side, members in (('A', sa), ('B', sb)):
        for i, s in enumerate(members):
            v = copy.copy(s)
            v.name = f'{side}{i}:{s.name}'
            exp.samples[v.name] = v
            keys[side].append(v.name)
    # Their 'cluster' labels ARE one clustering of every sample; without
    # this record compare_conditions would cluster them together again.
    exp.joint_clustering = {'samples': tuple(exp.samples),
                            'label_col': 'cluster',
                            'method': 'openflo-run shared clustering'}
    try:
        summary = exp.compare_conditions(
            groupA=keys['A'],
            groupB=keys['B'],
            label_a=g_a['name'], label_b=g_b['name'],
        )
    except Exception as exc:
        print(f'  [!] Comparison failed: {exc}')
        return f'failed ({type(exc).__name__}: {exc})'
    if summary.empty:
        print(f"  [!] {g_a['name']} vs {g_b['name']}: comparison produced no "
              "rows (no shared populations) — nothing written", flush=True)
        return 'had no rows'
    # The ids are the shared clustering's, not those of the <name>_stats.csv
    # beside it; say so in the file, where a reader comparing the two looks.
    summary['cluster_ids'] = _SHARED_IDS_NOTE
    base = _compare_base(g_a, g_b)
    summary.to_csv(os.path.join(trial_out, f'{base}.csv'), index=False)
    plt.gca().set_title(f"{g_a['name']} vs {g_b['name']} — shared clusters "
                        "(compare_shared_clusters.csv)")
    plt.savefig(os.path.join(trial_out, f'{base}.png'), dpi=150)
    if not _KEEP_FINAL_PLOTS:
        plt.close()
    return None


def _equal_quota(sizes, cap):
    """How many events each sample gives the shared clustering's pool of at
    most `cap`: an equal share each. A sample smaller than its share gives
    every event it has, and what it leaves is shared among the others; an
    indivisible remainder goes to the first samples.

    The pool used to be one uniform draw over all events, so each sample's
    share was its share of the events. A population held only by a small
    group then barely reached the pool (Ctrl 2 x 400k against Treat 2 x 20k
    gave Treat 5 % of it), while an equal share per sample, the usual
    practice when pooling samples for clustering, gives every sample the
    same weight in defining the clusters."""
    quota = [0] * len(sizes)
    left = min(int(cap), sum(sizes))
    active = [i for i, n in enumerate(sizes) if n > 0]
    while left > 0 and active:
        share = left // len(active)
        if share == 0:
            for i in active[:left]:
                quota[i] += 1
            break
        for i in active:
            take = min(share, sizes[i] - quota[i])
            quota[i] += take
            left -= take
        active = [i for i in active if quota[i] < sizes[i]]
    return quota


_DETECTOR_SUFFIXES = ('-A', '-H', '-W', ' A', ' H', ' W')


def _align_shared_channels(samples, names):
    """Match the marker channels of `samples` by antibody label, so each
    column of the shared clustering holds ONE marker in every sample.
    Returns (channels, columns, notes, display): `channels` are the first
    sample's detectors for the matched markers (the pooled frame's column
    names), `columns[i]` sample i's detectors in the same order, `notes`
    one line per channel matched across different detectors, matched by
    detector although its label differs, or left out, naming the samples
    by `names`, and `display` one distinct name per column.

    Matching by detector name put two antibodies in one column when a day's
    panel moved a marker to another fluor (CD901b on FL1 one day, on FL2
    the next), and dropped a channel missing from one sample without a word.
    A label is the channel's $PnS / --labels / --panel name, compared with
    case, spacing and punctuation ignored (letters of any script kept:
    'CD8α' is not 'CD8β') and a fluor name equal to the
    detector's dropped ('CD901b', 'CD901B ', 'CD901b FL1' on FL1-A); a
    label that is only the fluor ('FL1') is no label. The detector's
    -A/-H/-W suffix is part of the match, so CD901b-A never pairs with
    CD901b-H. A label a sample puts on more than one detector ('Dump') is
    matched by label AND detector.

    The rules look at every sample at once, so the result does not depend
    on their order (it did: the first sample's channels drove the match):

    1. A labelled marker is one column when EVERY sample has it.
    2. Otherwise a detector every sample has, and no rule-1 column uses, is
       one column by detector name when it carries at most ONE label across
       the samples, and no sample that leaves it unlabelled has that label
       on another detector (there, the unlabelled channel is something
       else). Two different labels on one detector are never pooled.
    3. Every other channel is left out, and named."""
    def suffix(det):
        return next((x for x in _DETECTOR_SUFFIXES if det.endswith(x)), '')

    def fold(text):
        """Case, spacing and punctuation dropped, letters of any script
        kept: _norm_token keeps only [a-z0-9], so 'TCRβ' / 'TCRγδ' and
        'CD8α' / 'CD8β' each folded to one marker and swapped panels were
        pooled silently."""
        return re.sub(r'[\W_]+', '', str(text).casefold())

    def fluor(det):
        """The detector's fluor as a token: FL1-A -> 'fl1'."""
        return fold(det[:len(det) - len(suffix(det))])

    def label(s, det):
        """The channel's marker label, or None when it has none: no $PnS,
        or one that only repeats the detector ('FL1-A') or its fluor
        ('FL1'), which names no marker."""
        lbl = (getattr(s, 'channel_labels', None) or {}).get(det) or det
        tok = fold(lbl)
        return None if tok in ('', fold(det), fluor(det)) else lbl

    def show(lbl, det):
        return f'{lbl} ({det})' if lbl else det

    def where(values):
        """'v1 in s0, s2; v2 in s1' over the samples."""
        by = {}
        for i, v in enumerate(values):
            by.setdefault(v, []).append(names[i])
        return '; '.join(f"{v} in {', '.join(ns)}" for v, ns in by.items())

    def norm(lbl, det):
        """A label as compared: case, spacing and punctuation dropped
        ('Ly-6G' is 'Ly6G'), and so is a leading or trailing fluor name
        equal to the detector's when it stands apart ('CD901b FL1' on
        FL1-A, 'CD3 FL3' on FL3-A, 'FL1-CD901b' are CD901b / CD3).
        Folding only case and spacing dropped a marker over punctuation in
        207 of 4000 fuzzed panels. The fluor is dropped only as a separate
        word: as a bare suffix it took the 'b' off 'CD901b' on a detector
        named 'B'."""
        if not lbl:
            return None
        fl = fluor(det)
        for sep in re.finditer(r'[\W_]+', lbl):   # a Greek letter is no gap
            head, tail = lbl[:sep.start()], lbl[sep.end():]
            if fl and fold(tail) == fl and fold(head):
                return fold(head)
            if fl and fold(head) == fl and fold(tail):
                return fold(tail)
        return fold(lbl)

    n = len(samples)
    dets = [[d for d in s.fluor_channels if d in s.data.columns]
            for s in samples]
    labs = [{d: label(s, d) for d in ds}
            for s, ds in zip(samples, dets, strict=True)]
    # A marker's key: its label (as norm() folds it) + -A/-H/-W. A
    # label a sample puts on two detectors ('Dump' on FL1-A and FL6-A)
    # is told apart by detector, so the same panel pairs detector with
    # detector; one key per label kept the first detector and paired a's
    # FL1-A with b's FL6-A when b listed its channels in another order.
    held = []                           # per sample: {(label, suffix)}
    keyed = []                          # per sample: {marker key: detector}
    for i in range(n):
        seen = [(norm(labs[i][d], d), suffix(d)) for d in dets[i]
                if labs[i][d] is not None]
        held.append(set(seen))
        m = {}
        for d in dets[i]:
            if labs[i][d] is not None:
                ls = (norm(labs[i][d], d), suffix(d))
                m[ls + ((None,) if seen.count(ls) == 1 else (d,))] = d
        keyed.append(m)

    used = [set() for _ in samples]
    by_first = {}                       # first sample's detector -> row
    display = {}                        # first sample's detector -> name
    notes = []
    # 1. markers every sample labels (walked in the first sample's channel
    #    order only so the notes come out in a stable order)
    common = set(keyed[0]).intersection(*keyed[1:])
    for key in sorted(common, key=lambda k: dets[0].index(keyed[0][k])):
        row = [keyed[i][key] for i in range(n)]
        for i, d in enumerate(row):
            used[i].add(d)
        by_first[row[0]] = row
        name = labs[0][row[0]]
        display[row[0]] = show(name, row[0]) if key[2] else name
        if len(set(row)) > 1:
            notes.append(f'{name}: {where(row)}')
    # 2. detectors every sample has, by name, when that pools ONE marker
    reported = set()
    for d in dets[0]:
        if any(d not in labs[i] or d in used[i] for i in range(n)):
            continue
        found = {norm(labs[i][d], d) for i in range(n)} - {None}
        if len(found) > 1:
            notes.append(f'{d}: different markers, not pooled; ' + where(
                [labs[i][d] or 'no label' for i in range(n)]))
            reported.add(d)
            continue
        named = next((labs[i][d] for i in range(n) if labs[i][d]), None)
        if found:
            lbl, = found
            if any(labs[i][d] is None and (lbl, suffix(d)) in held[i]
                   for i in range(n)):
                continue
            if any(labs[i][d] is None for i in range(n)):
                notes.append(f'{d}: matched by detector; ' + where(
                    [labs[i][d] or 'no label' for i in range(n)]))
        row = [d] * n
        for i in range(n):
            used[i].add(d)
        by_first[d] = row
        display[d] = named or d
    # 3. the rest is left out, by name
    left = {}
    for i in range(n):
        for d in dets[i]:
            if d not in used[i] and d not in reported:
                left.setdefault(show(labs[i][d], d), set()).add(i)
    for key, have in left.items():
        notes.append(f'{key}: no matching channel in '
                     + ', '.join(names[j] for j in range(n) if j not in have))

    channels = [d for d in dets[0] if d in by_first]
    columns = [[by_first[d][i] for d in channels] for i in range(n)]
    # One name per column: two columns sharing one ('Dump' twice) would give
    # the table two median_Dump columns.
    taken = [display[d] for d in channels]
    display = {d: (show(display[d], d) if taken.count(display[d]) > 1
                   and display[d] != d else display[d]) for d in channels}
    return channels, columns, notes, display


def _shared_cluster_labels(samples, cluster_task, names=None):
    """Cluster `samples` TOGETHER, once. Returns (labels, channels,
    columns, display): one int array per sample (-1 = unclustered) from a
    single clustering that every sample shares, the marker channels it used
    (the first sample's detector names), per sample that sample's detectors
    for them, and a name per channel (see _align_shared_channels; `names`
    name the samples in its messages). Raises ValueError when no shared
    clustering can be made.

    Each sample is also clustered on its own (its <name>_stats.csv), but
    PhenoGraph, Leiden and FlowSOM number clusters arbitrarily (PhenoGraph by
    size), so one id names different populations in different samples.
    Comparing groups by that id compared unrelated populations: with one
    population falling from 70% to 30% between two groups, the largest
    difference in the table was 2.2 points, and 17 of its 23 ids held
    different populations in different samples. A comparison needs ids that
    mean the same population in every sample, so the samples are pooled and
    clustered as one (the usual concatenate-then-cluster design), and each
    sample's events are then counted per shared cluster.

    The pooled run clusters a subsample (the pool) of at most --max-events
    events IN TOTAL, an equal share drawn at random from each sample (see
    _equal_quota). With no cap, the pool holds as many events as the
    largest sample, or the method's own cap when that is smaller (Leiden
    200,000, FlowSOM 50,000; _METHOD_EVENT_CAP), so the clustering itself
    costs no more than the largest per-sample run. Every event left out of
    the pool then takes the label of its nearest pooled event (1-NN), as
    --max-events already does inside one sample. FlowSOM instead labels
    every event through the SOM the pool trained (best-matching node ->
    its metacluster), which is FlowSOM's own rule and costs one pass.

    That 1-NN step is NOT free, and it is extra work no per-sample run does:
    it labels every unpooled event of the whole trial, in the main process.
    It runs on a scipy KDTree queried on every core. Measured on 14
    channels: 3 x 300k events with a 60k pool, 840k queries, took 144 s on
    one core with the scikit-learn KD-tree it replaced and 10 s here, with
    identical labels. A query costs more as the pool grows."""
    import pandas as pd
    from scipy.spatial import KDTree

    first = samples[0]
    names = names or [s.name for s in samples]
    channels, columns, notes, display = _align_shared_channels(samples, names)
    if notes:
        print("  [!] Shared clustering: channels are matched across samples "
              "by antibody label (by detector when one has no label). Left "
              "out, or matched across different detectors:", flush=True)
        for note in notes:
            print(f'      {note}', flush=True)
    sizes = [len(s.data) for s in samples]
    total = sum(sizes)
    if not channels:
        raise ValueError('no marker channel common to every sample')
    if not total:
        raise ValueError('the samples hold no events')
    method = cluster_task.get('method', 'phenograph')
    cap = _shared_pool_cap(sizes, cluster_task)
    rng = np.random.default_rng(cluster_task.get('random_state', 42))
    picked = [np.sort(rng.choice(n, q, replace=False)) if q < n
              else np.arange(n)
              for n, q in zip(sizes, _equal_quota(sizes, cap), strict=True)]
    pooled = copy.copy(first)
    pooled.data = pd.concat(
        [s.data[cols].iloc[ix].set_axis(channels, axis=1)
         for s, cols, ix in zip(samples, columns, picked, strict=True)],
        ignore_index=True)
    pooled.fluor_channels = list(channels)
    pooled.name = 'shared clustering'
    pooled.flowsom_result = None        # not the first sample's own SOM
    _cluster_by_method(pooled, cluster_task)
    if 'cluster' not in pooled.data.columns:
        raise ValueError(f"{cluster_task.get('method', 'phenograph')} gave "
                         f"no cluster labels for {len(pooled.data):,} events")
    som = (getattr(pooled, 'flowsom_result', None) if method == 'flowsom'
           else None)               # set by run_flowsom, inside the call
    if som and 'meta_of_node' in som:
        # FlowSOM's own label for an event is the metacluster of its
        # best-matching SOM node, so every event, pooled or not, gets that
        # from the SOM the pool trained. Its nearest POOLED event's label
        # differed for 996 of 9,000 events on overlapping test data.
        from .pipeline import _som_assign
        som_cols = [channels.index(c) for c in som['channels']]
        labels = []
        for s, cols in zip(samples, columns, strict=True):
            X = s.data[[cols[i] for i in som_cols]].to_numpy(dtype=float)
            ok = np.all(np.isfinite(X), axis=1)
            lab = np.full(len(s.data), -1, dtype=int)
            lab[ok] = som['meta_of_node'][_som_assign(X[ok], som['weights'])]
            labels.append(lab)
        return labels, channels, columns, display
    plab = pooled.data['cluster'].to_numpy(dtype=int)
    ref = plab >= 0
    tree = None
    if total > cap and ref.any():
        tree = KDTree(pooled.data[channels].to_numpy(dtype=float)[ref])
    ref_lab = plab[ref]
    labels, at = [], 0
    for s, cols, ix in zip(samples, columns, picked, strict=True):
        lab = np.full(len(s.data), -1, dtype=int)
        lab[ix] = plab[at:at + len(ix)]
        at += len(ix)
        if tree is not None:
            X = s.data[cols].to_numpy(dtype=float)
            rest = np.all(np.isfinite(X), axis=1)
            rest[ix] = False
            if rest.any():
                _, nbr = tree.query(X[rest], k=1, workers=-1)
                lab[rest] = ref_lab[nbr]
        labels.append(lab)
    return labels, channels, columns, display


def _shared_pool_cap(sizes, cluster_task):
    """At most how many events the shared clustering pools: --max-events,
    or with no cap the largest sample's size, never more than the method's
    own cap (_METHOD_EVENT_CAP)."""
    biggest = max(sizes)
    own = _METHOD_EVENT_CAP.get(cluster_task.get('method', 'phenograph'))
    return int(cluster_task.get('max_events')
               or (min(biggest, own) if own else biggest))


def _shared_n_jobs(n_jobs, pool):
    """PhenoGraph n_jobs for the shared clustering of `pool` events.
    --workers > 1 gives each per-sample run n_jobs=1, since they run side by
    side; the shared run starts after they have all finished and runs alone,
    so it may use every core. Measured on 14 channels, seeded Leiden,
    identical labels both ways (peak = RSS of the process and its children):

        pool 200k   n_jobs=1  356.6 s, 2.0 GB    n_jobs=-1  87.5 s, 5.7 GB
        pool  60k   n_jobs=1   45.6 s, 1.1 GB    n_jobs=-1  24.5 s, 4.6 GB

    The extra memory is the one process per core that PhenoGraph's Jaccard
    step starts: about 163 MB plus 240 B per pooled event each (starmap
    sends every child the kNN index; review-cli's measurement). Every core
    is used only when free RAM covers 2 GB plus that per core; otherwise
    `n_jobs` stands. A flat 0.25 GB a core let a 1M-event pool start
    children needing about 12 GB with 10 GB free."""
    if n_jobs == -1:
        return n_jobs
    try:
        import psutil
    except ImportError:
        return n_jobs
    free_gb = psutil.virtual_memory().available / 1024 ** 3
    need_gb = 2.0 + (os.cpu_count() or 1) * (0.17 + pool * 2.4e-7)
    if free_gb < need_gb:
        print(f"  Shared clustering keeps PhenoGraph n_jobs={n_jobs}: "
              f"{free_gb:.1f} GB RAM free, every core would need about "
              f"{need_gb:.1f} GB for a {pool:,}-event pool.", flush=True)
        return n_jobs
    return -1


_STAGING_PREFIX = '.compare-staging-'
_STAGING_MARK = '.openflo-compare-staging'


def _is_staging(path):
    """True for a staging folder OpenFlo made: the exact name prefix AND the
    mark file written when it was made, so a group's output folder that
    happens to share the prefix is never taken for one."""
    return (os.path.basename(path).startswith(_STAGING_PREFIX)
            and os.path.isfile(os.path.join(path, _STAGING_MARK)))


def _clear_stale_staging(out, prefix=''):
    """Remove staging folders an earlier run left in `out`: one killed
    while publishing, or one whose new set could not be moved in (a file in
    use). They were never removed, and --show-plots re-opened their PNGs."""
    import shutil
    if not os.path.isdir(out):
        return
    for name in sorted(os.listdir(out)):
        path = os.path.join(out, name)
        if os.path.isdir(path) and _is_staging(path):
            shutil.rmtree(path, ignore_errors=True)
            print(f'  {prefix}Removed {path}, left by an earlier run.',
                  flush=True)


def _publish_compare_set(out, staging, written, no_file, prefix):
    """Replace the comparison set in `out` with the one in `staging`: every
    compare_* file already in `out` is moved aside, the new files are moved
    in, and the old ones are deleted. So no old file stays beside the new
    set: not a pair of a group no longer in the run, not a pair that failed
    or had no rows this time. File names compare without case, as Windows
    and macOS do: comparing them with case deleted a just-written
    compare_Ctrl_vs_Treat.csv whose old name was compare_ctrl_vs_Treat.csv.
    `no_file` maps a pair's file stem to why it has no file this time.

    Returns None, or why the set could not be moved in: a file that cannot
    be moved (open in Excel: PermissionError). Then everything is put back
    as it was, the old set stays in `out`, and the new one in `staging`."""
    import shutil
    old = _compare_files(out)
    aside = os.path.join(staging, 'previous')
    os.makedirs(aside)
    moved, placed = [], []
    try:
        for f in old:
            os.replace(os.path.join(out, f), os.path.join(aside, f))
            moved.append(f)
        for f in written:
            os.replace(os.path.join(staging, f), os.path.join(out, f))
            placed.append(f)
    except OSError as exc:
        stuck = []
        for src, dst in ([(os.path.join(out, f), os.path.join(staging, f))
                          for f in reversed(placed)]
                         + [(os.path.join(aside, f), os.path.join(out, f))
                            for f in reversed(moved)]):
            try:
                os.replace(src, dst)
            except OSError:
                stuck.append(src)
        why = f'{exc.filename} cannot be moved ({type(exc).__name__}: ' \
              f'{exc.strerror or exc}), e.g. it is open in Excel'
        if stuck:
            why += ('; these could not be put back and are still where they '
                    'were moved: ' + ', '.join(stuck))
        else:
            os.rmdir(aside)                         # empty again
        return why
    new = {f.casefold() for f in written}
    why_of = {stem.casefold(): why for stem, why in no_file.items()}
    gone = {}
    for f in old:
        if f.casefold() not in new:
            why = why_of.get(os.path.splitext(f)[0].casefold(),
                             "not part of this run's set")
            gone.setdefault(why, []).append(f)
    for why, files in gone.items():
        print(f"  [!] {prefix}{', '.join(files)} from an earlier run "
              f'removed: {why}.', flush=True)
    shutil.rmtree(staging, ignore_errors=True)   # ours: made by mkdtemp
    return None


def _not_in_this_run(groups, results, reused, tasks=(), failures=None):
    """One 'group/sample: why' line per listed sample that is missing from
    the comparison: neither processed in this run nor kept from an earlier
    one (--skip-existing) -- it failed (with its error, from `failures`),
    or no task was made for it because its FCS file was not found -- or
    whose task is missing an input file (`missing_inputs`: a kept sample's
    file is gone, or a trial folder of a concatenated sample lacks it).
    Checked before any kept sample is reloaded."""
    failures = failures or {}
    tried = {(t['group'], t['name']) for t in tasks}
    kept = {(g, t['name']) for g, ts in reused.items() for t in ts}
    missing = [f"{t['group']}/{t['name']}: "
               + ('kept from an earlier run, but its input is missing'
                  if (t['group'], t['name']) in kept
                  else 'its input is missing')
               + f" ({'; '.join(t['missing_inputs'])})"
               for t in tasks if t.get('missing_inputs')]
    for g in groups:
        have = ({s.name for s in results.get(g['name'], [])}
                | {t['name'] for t in reused.get(g['name'], [])})
        for nm in g['samples']:
            if nm in have:
                continue
            err = failures.get(f"{g['name']}\x1f{nm}")
            if err:
                why = f'failed in this run ({err})'
            elif (g['name'], nm) in tried:
                why = f"failed in this run (see 'Failed: {nm}' above)"
            else:
                why = 'its FCS file was not found'
            missing.append(f"{g['name']}/{nm}: {why}")
    return missing


def _samples_to_compare(groups, results, reused, tasks=(), failures=None):
    """Every sample of every group, for the group comparisons. Returns
    (by_group, missing): {group name: [FlowSample]} with the samples
    processed in this run plus, under --skip-existing, the ones kept from
    an earlier run, whose events are reloaded (not re-clustered — the
    shared clustering does not need their own clusters), ordered as the
    group lists them so the shared clustering's subsample does not depend
    on the order parallel workers finished in; and `missing`, one
    'group/sample (why)' line per listed sample that is in neither. The
    caller writes no comparison when `missing` is not empty, so a sample
    not in this run is named BEFORE any reload, which would be wasted."""
    missing = _not_in_this_run(groups, results, reused, tasks, failures)
    if missing:
        return {}, missing
    by_group = {}
    for g in groups:
        name = g['name']
        if name in by_group:
            continue
        members = list(results.get(name, []))
        kept = reused.get(name, [])
        for i, t in enumerate(kept, 1):
            print(f"  Reloading {name}/{t['name']} (kept from an earlier run) "
                  f"for the group comparisons, {i}/{len(kept)} …", flush=True)
            try:
                s = _load_task_sample(t)
                err = None if s is not None else 'no input files found'
            except Exception as exc:
                s, err = None, f'{type(exc).__name__}: {exc}'
            if s is None:
                missing.append(f"{name}/{t['name']}: kept from an earlier "
                               f"run, but cannot be reloaded ({err})")
                continue
            members.append(s)
        rank = {n: i for i, n in enumerate(g['samples'])}
        by_group[name] = sorted(members,
                                key=lambda s: rank.get(s.name, len(rank)))
    return by_group, missing


def _write_group_outputs(groups, results, reused, out, report, cluster_task,
                         random_state=42, pairs=None, tl='', tasks=(),
                         failures=None):
    """Group-level outputs of one analysis: a UMAP + marker-pair scatters per
    group, then every pairwise group comparison. `results` holds the samples
    processed in this run; `reused` maps a group to the tasks of its samples
    that --skip-existing kept from an earlier run; `tasks` is every
    sample's task, processed or kept, and `failures` maps a failed task's
    key to its error (to say why a sample is missing).

    No group output is computed from part of a group. A resumed run used to
    rewrite them from the re-run samples alone: a 2-sample comparison was
    overwritten by a 1-sample one, and the UMAP was redrawn from one sample.

    Returns False when the comparisons asked for were not written (a sample
    missing, the shared clustering failed, or a file in use kept the new
    set out of `out`), so the run can end with a non-zero exit; True when
    they were, or when none was due (one group)."""
    from itertools import combinations

    import pandas as pd
    prefix = f'[{tl}] ' if tl else ''
    suffix = f'_{tl}' if tl else ''
    _clear_stale_staging(out, prefix)

    # UMAP per group with >= 1 sample. Single-sample groups still get
    # the per-group UMAP rendered (the by-sample_origin panel is just
    # one colour in that case, but the by-cluster + FSC/SSC panels
    # are still meaningful and were previously being dropped).
    for g in groups:
        samples = results.get(g['name'], [])
        if not samples:
            continue
        if reused.get(g['name']):
            # The UMAP colours by each sample's own clusters, which a kept
            # sample no longer has without clustering it again.
            print(f"  [skip-existing] {g['name']}: "
                  f"{len(reused[g['name']])} sample(s) kept from an earlier "
                  "run, so its UMAP and pair scatters are left as they are. "
                  "Re-run without --skip-existing to redraw them.", flush=True)
            continue
        report(f'{prefix}Running UMAP ({g["name"]}, {len(samples)} sample(s)) …')
        run_group_umap(samples,
                       _safe_filename(f'{g["name"]}{suffix}'),
                       out,
                       random_state=random_state)
        # Marker-pair scatters: per-sample grid + cross-sample overlay.
        save_group_pair_scatters(samples, f'{g["name"]}{suffix}', out,
                                 pairs=pairs)

    group_pairs = list(combinations(groups, 2))
    if not group_pairs:
        return True
    # The comparisons are recomputed on EVERY run, resumes included:
    # --skip-existing skips only the per-sample work. Deciding when an
    # earlier set could be kept took a record of everything that made it
    # (groups, input files, settings, code) and kept going wrong; the shared
    # clustering costs 10 s to a few minutes, small next to the samples.
    by_group, missing = _samples_to_compare(groups, results, reused, tasks,
                                            failures)
    if missing:
        # All the comparison files come from one shared clustering, so they
        # are written together or not at all (the user's rule, fresh runs
        # included). A resume that could not reload one kept sample used to
        # leave its group out and rewrite the rest: compare_shared_clusters
        # .csv and the other pairs came from a new clustering while that
        # group's pairs kept the old one, and one cluster id read 30 % in
        # one file and 70 % in the next.
        print(f"  [!] {prefix}No group comparisons were written: "
              f"{len(missing)} sample(s) of the groups are missing.",
              flush=True)
        for line in missing:
            print(f'      {line}', flush=True)
        print("      A comparison from part of a group would not be the "
              "group's. Fix these samples or remove them from the groups, "
              "then run again (--skip-existing reuses the finished samples). "
              f"Any compare_* files in {out} are from an earlier run, not "
              "this one.", flush=True)
        return False
    import shutil
    import tempfile

    # Every file of the set is written to a staging folder and moved into
    # `out` only once the whole set exists (_publish_compare_set). Writing
    # in place left a run killed after its first pair with
    # compare_shared_clusters.csv and that pair from the new clustering and
    # the other pairs from the old one. The folder is a NEW one (mkdtemp),
    # in `out` so the moves stay on one drive: a fixed name could be a
    # group's output folder, and a group named '.compare_staging' had its
    # stats deleted when that folder was cleared.
    staging = tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=out)
    with open(os.path.join(staging, _STAGING_MARK), 'w', encoding='utf-8'):
        pass                    # marks it ours, for _clear_stale_staging
    shared = {}
    flat = [(g, s) for g, members in by_group.items() for s in members]
    report(f'{prefix}Clustering {len(flat)} sample(s) together for the '
           'group comparisons …')
    if cluster_task.get('method', 'phenograph') == 'phenograph':
        sizes = [len(s.data) for _, s in flat]
        pool = min(sum(sizes), _shared_pool_cap(sizes, cluster_task))
        cluster_task = dict(cluster_task, n_jobs=_shared_n_jobs(
            cluster_task.get('n_jobs', 1), pool))
    try:
        labels, channels, columns, display = _shared_cluster_labels(
            [s for _, s in flat], cluster_task,
            names=[f'{g}/{s.name}' for g, s in flat])
    except Exception as exc:
        # Falling back to each sample's own cluster ids would compare
        # unrelated populations; write no comparison rather than that.
        shutil.rmtree(staging, ignore_errors=True)
        print(f"  [!] {prefix}Shared clustering failed ({type(exc).__name__}"
              f": {exc}). No group comparisons were written. Any compare_* "
              f"files in {out} are from an earlier run, not this one.",
              flush=True)
        return False
    # Medians are named by the shared columns' names for every sample:
    # the columns are matched by label, so one name is one marker.
    phenotypes = []
    for (g, s), lab, cols in zip(flat, labels, columns, strict=True):
        v = copy.copy(s)
        v.data = pd.DataFrame({'cluster': lab})
        v.fluor_channels = []
        shared.setdefault(g, []).append(v)
        m = copy.copy(s)
        m.data = s.data[cols].set_axis(channels, axis=1).assign(cluster=lab)
        m.fluor_channels = list(channels)
        m.channel_labels = display
        # The record goes with each column to its shared name, so the
        # medians are inverted with this sample's own transform.
        rec = getattr(s, 'data_transforms', None) or {}
        m.data_transforms = {ch: rec[col] for ch, col
                             in zip(channels, cols, strict=True) if col in rec}
        freq = m.cluster_frequencies()
        freq.insert(0, 'group', g)
        phenotypes.append(freq)
    table = 'compare_shared_clusters.csv'
    pd.concat(phenotypes, ignore_index=True).to_csv(
        os.path.join(staging, table), index=False)
    written, no_file = [table], {}
    for g_a, g_b in group_pairs:
        base = _compare_base(g_a, g_b)
        why = _compare_group_pair(g_a, g_b, shared, staging, report,
                                  trial_label_prefix=tl)
        if why:
            no_file[base] = (f"this run's {g_a['name']} vs {g_b['name']} "
                             f'comparison {why}')
        else:
            written += [f'{base}.csv', f'{base}.png']
    locked = _publish_compare_set(out, staging, written, no_file, prefix)
    if locked:
        print(f"  [!] {prefix}Group comparisons NOT updated: {locked}. The "
              f"comparison files in {out} are left as they were, from an "
              f"earlier run. This run's comparisons are in {staging}. Close "
              "the file and run again, or copy them from there before the "
              "next run, which removes that folder.", flush=True)
        return False
    print(f"  Group comparisons use clusters found by clustering all "
          f"{len(flat)} samples together, so an id names the same "
          "population in every sample. They are NOT the ids in each "
          "<name>_stats.csv. Each shared cluster's size and marker "
          f"medians per sample: {os.path.join(out, table)}", flush=True)
    return True


def _run_independent(trial_dirs, out_dir, groups, fmo_sets,
                     k, fmo_percentile, labels, gate_overrides, region_gates,
                     skip_existing,
                     workers, n_jobs, max_events, admission_gb,
                     vram_admission_gb, filter_debris_um, bead_um,
                     doublet_tol, report, random_state=42, pairs=None,
                     cytonorm=None, cluster_cfg=None):
    """Process each trial folder as a separate analysis; results go into
    per-trial sub-dirs. Samples within a trial are processed in parallel
    when workers > 1. Every group with >= 1 sample gets a UMAP; every
    pair of groups gets a comparison. Returns False when a trial's
    comparisons were not written (the other trials still run)."""
    settings = _cluster_settings(k, n_jobs, max_events, vram_admission_gb,
                                 random_state, cluster_cfg)
    n_trials = len(trial_dirs)
    published = True

    for trial_dir in trial_dirs:
        tl = _trial_label(trial_dir)
        trial_out = os.path.join(out_dir, tl) if n_trials > 1 else out_dir

        # FMO thresholds for each set actually referenced by the groups
        fmo_thresh = _build_fmo_thresholds_for_groups(
            trial_dir, groups, fmo_sets, fmo_percentile, gate_overrides,
            report, trial_label_prefix=tl)

        # Bead-based µm/FSC calibration (used only if debris filter is on).
        debris_min_fsc = None
        if filter_debris_um and filter_debris_um > 0:
            factor, _ = _calibrate_bead_micron(trial_dir, fmo_sets, bead_um)
            if factor:
                debris_min_fsc = float(filter_debris_um) / factor
                print(f"[Calibration] Debris cut-off: {filter_debris_um:.1f} µm "
                      f"→ FSC-A >= {debris_min_fsc:,.0f}", flush=True)
            else:
                print(f"[Calibration] No readable bead file in {tl} — "
                      f"debris filter disabled for this trial.", flush=True)

        # Build per-sample tasks. A sample --skip-existing finds done is not
        # processed again, but its task is kept so the group comparison can
        # reload its events (see _write_group_outputs).
        tasks, reused = [], {}
        for g in groups:
            for name in g['samples']:
                # Per-sample FMO set (override → group default → ungated).
                sfmo = g.get('sample_fmo', {}).get(name, g['fmo_set'])
                thresh = fmo_thresh.get(sfmo, {})
                done = skip_existing and _already_done(trial_out, g['name'], name)
                if done:
                    report(f'Skipping {name} (output already exists)')
                # By-day groups carry their own source folder; fall back to
                # the run's trial_dir for the classic single-folder layout.
                src_dir = g.get('trial_dir') or trial_dir
                try:
                    paths, gone = [(name, fcs_path(src_dir, name))], []
                except FileNotFoundError as e:
                    if not done:
                        report(f'[{tl}] {name}: {e}')
                        continue
                    paths, gone = [], [str(e)]   # kept, but cannot reload
                task = dict(settings,
                            name=name, group=g['name'], out=trial_out,
                            paths=paths, missing_inputs=gone,
                            thresholds=thresh,
                            region_gates=region_gates, labels=labels,
                            debris_min_fsc=debris_min_fsc,
                            doublet_tol=doublet_tol, cytonorm=cytonorm,
                            batch_id=src_dir)
                if done:
                    reused.setdefault(g['name'], []).append(task)
                else:
                    tasks.append(task)

        failures = {}
        results = _run_sample_tasks(tasks, workers, admission_gb, report,
                                    failures)
        published &= _write_group_outputs(
            groups, results, reused, trial_out, report, settings,
            random_state=random_state, pairs=pairs, tl=tl,
            failures=failures,
            tasks=tasks + [t for ts in reused.values() for t in ts])
    return published


# ── Concatenate mode ──────────────────────────────────────────────────────────

def _run_concatenated(trial_dirs, out_dir, groups, fmo_sets,
                      k, fmo_percentile, labels, gate_overrides, region_gates,
                      skip_existing,
                      workers, n_jobs, max_events, admission_gb,
                      vram_admission_gb, filter_debris_um, bead_um,
                      doublet_tol, report, random_state=42, pairs=None,
                      cytonorm=None, cluster_cfg=None):
    """Load each sample from every trial, concatenate with trial-origin labels,
    then cluster / UMAP once on the merged data per sample. Different
    samples are processed in parallel when workers > 1. FMO thresholds
    come from the first (reference) trial only."""
    settings = _cluster_settings(k, n_jobs, max_events, vram_admission_gb,
                                 random_state, cluster_cfg)
    ref_dir = trial_dirs[0]

    # FMO thresholds for each set actually referenced by the groups
    fmo_thresh = _build_fmo_thresholds_for_groups(
        ref_dir, groups, fmo_sets, fmo_percentile, gate_overrides,
        report, trial_label_prefix='ref trial')

    # Bead-based µm/FSC calibration from the reference trial.
    debris_min_fsc = None
    if filter_debris_um and filter_debris_um > 0:
        factor, _ = _calibrate_bead_micron(ref_dir, fmo_sets, bead_um)
        if factor:
            debris_min_fsc = float(filter_debris_um) / factor
            print(f"[Calibration] Debris cut-off: {filter_debris_um:.1f} µm "
                  f"→ FSC-A >= {debris_min_fsc:,.0f}", flush=True)
        else:
            print("[Calibration] No readable bead file in reference trial — "
                  "debris filter disabled.", flush=True)

    # Build per-sample tasks — paths span every trial folder. A sample
    # --skip-existing finds done keeps its task for the comparison's reload.
    tasks, reused = [], {}
    for g in groups:
        for name in g['samples']:
            # Per-sample FMO set (override → group default → ungated).
            sfmo = g.get('sample_fmo', {}).get(name, g['fmo_set'])
            thresh = fmo_thresh.get(sfmo, {})
            done = skip_existing and _already_done(out_dir, g['name'], name)
            if done:
                report(f'Skipping {name} (output already exists)')
            # A by-day group names its own folder: resolve its samples there.
            # Searching the run's folders instead (collapsed to the parent
            # for by-day runs) found the same filename in every day folder
            # and took the first one — day 0's file — for every day.
            src_dirs = [g['trial_dir']] if g.get('trial_dir') else trial_dirs
            # A trial folder without the sample's file: the sample is still
            # processed from the others (with a warning), but it is missing
            # data, so no comparison is written (_not_in_this_run). A kept
            # sample was reloaded from the remaining folders without a word:
            # Treat fell from 60 % to 30 % beside a t1_stats.csv made from
            # both trials.
            paths, gone = [], []
            for trial_dir in src_dirs:
                try:
                    paths.append((_trial_label(trial_dir),
                                   fcs_path(trial_dir, name)))
                except FileNotFoundError as e:
                    gone.append(str(e))
                    if not done:
                        print(f"  [!] {e}")
            if not paths and not done:
                report(f'{name}: not found in any trial folder')
                continue
            task = dict(settings,
                        name=name, group=g['name'], out=out_dir,
                        paths=paths, missing_inputs=gone, thresholds=thresh,
                        region_gates=region_gates, labels=labels,
                        debris_min_fsc=debris_min_fsc,
                        doublet_tol=doublet_tol, cytonorm=cytonorm,
                        batch_id=g.get('trial_dir') or ref_dir)
            if done:
                reused.setdefault(g['name'], []).append(task)
            else:
                tasks.append(task)

    failures = {}
    results = _run_sample_tasks(tasks, workers, admission_gb, report,
                                failures)
    return _write_group_outputs(
        groups, results, reused, out_dir, report, settings,
        random_state=random_state, pairs=pairs, failures=failures,
        tasks=tasks + [t for ts in reused.values() for t in ts])


# ── Programmatic entry point ──────────────────────────────────────────────────
# `run(...)` takes keyword arguments and is suitable for importing from other
# Python code. `main()` (below) is the no-arg argparse-driven entry point
# that the `openflo-run` console script invokes.

def _proper_markers(sample):
    """Fluor marker channels with -H/-W detector versions dropped."""
    fl = list(getattr(sample, 'fluor_channels', None) or [])
    prim = [c for c in fl
            if not (c.endswith('-H') or c.endswith('-W')
                    or c.endswith(' H') or c.endswith(' W'))]
    return prim or fl


def _fit_cytonorm(groups, default_trial, mode='goal', n_metaclusters=10,
                  control_token='', max_events_per_batch=20_000, report=None):
    """Fit a CytoNorm model across batches (each group's ``trial_dir``).
    'goal' pools all samples per batch; 'controls' pools only samples whose
    name contains ``control_token``. Returns ``(model_dict, channels)`` or
    ``(None, None)`` when it can't (fewer than 2 usable batches/markers)."""
    import numpy as np
    import pandas as pd

    from .pipeline import CytoNorm
    say = report or (lambda m: print(m, flush=True))
    by_batch = {}
    for g in groups:
        tdir = g.get('trial_dir') or default_trial
        for raw in g['samples']:
            nm = raw.get('name') if isinstance(raw, dict) else raw
            if not nm:
                continue
            nm = str(nm)
            if (mode == 'controls' and control_token
                    and control_token.lower() not in nm.lower()):
                continue
            try:
                by_batch.setdefault(tdir, []).append(fcs_path(tdir, nm))
            except FileNotFoundError:
                continue
    if len(by_batch) < 2:
        say(f"[batch-correct] need ≥2 batches with samples (got "
            f"{len(by_batch)}) — skipped.")
        return None, None

    loaded, shared, ref = {}, None, None
    for batch, paths in by_batch.items():
        frames = []
        for path in paths:
            try:
                s = FlowSample(path)
                s.run_qc()
                s.auto_compensate()
                s.apply_transform()
            except Exception as exc:
                say(f"[batch-correct] skip {os.path.basename(path)}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            ch = set(_proper_markers(s))
            shared = ch if shared is None else (shared & ch)
            ref = ref or s
            d = s.data
            if len(d) > max_events_per_batch:
                d = d.sample(max_events_per_batch, random_state=42)
            frames.append(d)
        if frames:
            loaded[batch] = frames
    if ref is None or shared is None or len(shared) < 2 or len(loaded) < 2:
        say("[batch-correct] insufficient shared markers / batches — skipped.")
        return None, None

    channels = [c for c in _proper_markers(ref) if c in shared]
    events_by_batch = {
        batch: pd.concat([d[channels] for d in frames], ignore_index=True)
        for batch, frames in loaded.items()}
    cn = CytoNorm(channels, n_metaclusters=n_metaclusters, mode=mode).fit(
        events_by_batch)
    qc = cn.qc(events_by_batch)
    # Channels that could not be evaluated report NaN, so average across the
    # ones that were actually measured and say how many were not.
    b = float(np.nanmean([v['before'] for v in qc.values()]))
    a = float(np.nanmean([v['after'] for v in qc.values()]))
    skipped = sum(1 for v in qc.values() if not np.isfinite(v['before']))
    note = f", {skipped} marker(s) not evaluable" if skipped else ""
    scores = "not evaluable" if not np.isfinite(b) else f"{b:.3f} → {a:.3f}"
    say(f"[batch-correct] CytoNorm ({mode}): {len(channels)} markers, "
        f"{len(events_by_batch)} batches, mean batch→goal "
        f"{scores}{note}.")
    return cn.to_dict(), channels


def run(trial_dirs, out_dir, groups, fmo_sets=None, batch_mode='independent',
        k=30, fmo_percentile=99.5, labels=None, gate_overrides=None,
        region_gates=None,
        show_plots=False, skip_existing=False, workers=1, max_events=None,
        admission_gb=4.0, vram_admission_gb=1.0,
        filter_debris_um=0.0, bead_um=8.0, doublet_tol=0.0,
        random_state=42, palette_name='auto', pairs=None,
        batch_correct=False, cytonorm_mode='goal', cytonorm_metaclusters=10,
        cytonorm_control='',
        cluster_method='phenograph', resolution=1.0, n_metaclusters=10,
        reproducible=True):
    """Run the full pipeline for a list of user-defined sample groups.

    groups
        List of dicts: ``[{'name': 'Day 1', 'samples': ['d1m1','d1m2'],
        'fmo_set': 'standard'}, ...]``. Supports any number of named groups
        (group 1 vs group 2, day 1 .. day 8, treated vs control, etc).

    fmo_sets
        Dict mapping a set name to a ``{detector_channel: fcs_basename}``
        dict that the FMO threshold builder needs. Defaults to
        ``DEFAULT_FMO_SETS`` (the Late / Early pair).

    Returns 0, or 1 when group comparisons asked for were not written: a
    sample was missing, the shared clustering failed, or a file in use kept
    the new set out (the run still finishes every trial).
    """

    global _KEEP_FINAL_PLOTS
    _KEEP_FINAL_PLOTS = bool(show_plots)

    if isinstance(trial_dirs, str):
        trial_dirs = [trial_dirs]
    trial_dirs = [d for d in trial_dirs if d]

    if fmo_sets is None:
        fmo_sets = DEFAULT_FMO_SETS
    groups = _normalise_groups(groups)
    if not groups:
        print("[!] No groups defined — nothing to do.", flush=True)
        return 0
    problems = (_group_spec_problems(groups)
                + _sample_file_problems(groups, trial_dirs))
    if problems:
        raise ValueError('invalid groups: ' + ' '.join(problems))

    os.makedirs(out_dir, exist_ok=True)

    # Batch correction (CytoNorm) — fit once across batches up front; the
    # serialized model rides along on each task and is applied in the worker.
    cytonorm_dict = None
    if batch_correct:
        default_trial = trial_dirs[0] if trial_dirs else '.'
        cytonorm_dict, _ = _fit_cytonorm(
            groups, default_trial, mode=cytonorm_mode,
            n_metaclusters=cytonorm_metaclusters,
            control_token=cytonorm_control)

    from .pipeline import GPU_AVAILABLE, GPU_CLUSTERING_AVAILABLE, GPU_NAME
    if GPU_AVAILABLE:
        clu = "+ clustering (cuML + cuGraph)" if GPU_CLUSTERING_AVAILABLE else "(UMAP only)"
        print(f"[GPU] {GPU_NAME} detected — running UMAP {clu}", flush=True)
        if GPU_CLUSTERING_AVAILABLE:
            print(f"[GPU] Per-worker VRAM admission: ≥ {vram_admission_gb:.1f} GB free "
                  "to use GPU clustering (else CPU Phenograph).", flush=True)
    elif _gpu_present():
        print("[GPU] NVIDIA GPU detected, but RAPIDS cuML is not installed — "
              "UMAP & clustering run on CPU. Install cuML+cuGraph (Linux/WSL2) "
              "for GPU acceleration.", flush=True)
    else:
        print("[GPU] No CUDA GPU detected — UMAP will run on CPU (numba JIT)", flush=True)

    # Parallel sample processing.
    #
    # When running multiple outer workers in parallel we deliberately set
    # n_jobs=1 inside Phenograph instead of `cores // workers`. The reason:
    # at any n_jobs other than 1, phenograph's Jaccard step spawns a
    # multiprocessing.Pool() of cpu_count() processes, and its size sort a
    # Pool of n_jobs (FlowSample.cluster caps that one; phenograph alone
    # would always use cpu_count()). On Windows each inner child must
    # re-import the full numpy/scipy/sklearn stack (~250 MB) AND receive a
    # fresh copy of the events array, which multiplies the per-worker RAM
    # footprint. Forcing n_jobs=1 keeps every outer worker to a single
    # process; total parallelism is simply `workers` (each one runs Louvain
    # on its own core, which is the single-threaded bottleneck anyway).
    workers = max(1, int(workers))
    cores   = os.cpu_count() or 2
    n_jobs  = -1 if workers <= 1 else 1

    # RAM-aware throttling — cap workers to what free RAM can actually hold.
    try:
        import psutil
        free_gb  = psutil.virtual_memory().available / (1024 ** 3)
        # Per-worker peak estimate (calibrated against observed Windows runs).
        # Each outer worker holds, at peak:
        #   • imports (numpy/scipy/sklearn/phenograph/igraph)         ~0.5 GB
        #   • kNN graph (indices + distances)                          events × ~30 B
        #   • Jaccard sparse graph + intermediate copies              events × ~80 B
        #   • Louvain native binary-file scratch + igraph state       events × ~30 B
        #   • Phenograph's pickled return buffers + igraph leiden     events × ~15 B
        # → ≈ 0.5 GB + events × 4 µ-GB.  For 500 k events that's ~2.5 GB,
        # which matched the working-set we saw before the pagefile gave out.
        events_per_worker = max_events if max_events else 3_000_000
        per_worker_gb     = max(1.0, 0.5 + events_per_worker * 4.0e-6)
        # Reserve 2 GB for the parent process + OS headroom.
        budget_gb         = max(1.0, free_gb - 2.0)
        max_safe          = max(1, int(budget_gb / per_worker_gb))
        if workers > max_safe:
            print(f"[RAM-throttle] Free RAM {free_gb:.1f} GB, per-worker estimate "
                  f"~{per_worker_gb:.1f} GB → reducing workers {workers} → {max_safe}",
                  flush=True)
            workers = max_safe
    except ImportError:
        pass

    n_trials = len(trial_dirs)
    cap_msg  = f"  |  cluster cap: {max_events:,} events" if max_events else ""
    group_msg = " / ".join(f"{g['name']}({len(g['samples'])})" for g in groups)
    print(f"[Trials] {n_trials} trial folder(s)  |  mode: {batch_mode}  |  "
          f"parallel workers: {workers}  ({cores} cores){cap_msg}", flush=True)
    print(f"[Groups] {group_msg}", flush=True)

    # Step total — one [STEP N/M] line per unit of work the runners will do:
    #   • one per unique FMO set built  (per trial in independent mode,
    #     once in concatenate mode)
    #   • one per sample processed (or skipped)
    #   • one per group that gets a UMAP run (every group with >= 1
    #     processed sample; counting only groups with >= 2 printed
    #     [STEP 5/3] for two single-sample groups)
    #   • one for the shared clustering the group comparisons use
    #   • one per pair of groups (pairwise condition comparison)
    # A step that does not happen (failed sample, UMAP left to an earlier
    # run) only leaves the count short of the total, never past it.
    # Unique FMO sets across group defaults + per-sample overrides
    # (one build step each; '' = ungated builds nothing).
    _all_sets = set()
    for g in groups:
        _all_sets.add(g.get('fmo_set', ''))
        _all_sets.update(g.get('sample_fmo', {}).values())
    n_used_sets = len({s for s in _all_sets if s})
    n_samples   = sum(len(g['samples']) for g in groups)
    n_umaps     = len(groups)       # normalised groups all hold >= 1 sample
    n_compares  = (len(groups) * (len(groups) - 1)) // 2
    n_shared    = 1 if n_compares else 0
    per_trial_steps = (n_used_sets + n_samples + n_umaps + n_shared
                       + n_compares)
    total = (per_trial_steps if batch_mode == 'concatenate'
             else n_trials * per_trial_steps)
    total = max(total, 1)   # avoid 0/0 in the progress bar

    step = [0]

    def report(msg):
        step[0] += 1
        print(f'[STEP {step[0]}/{total}] {msg}', flush=True)

    # Propagate the palette choice to the worker side too — if main()
    # is being called from a script that didn't go through __main__,
    # the palette set earlier in __main__ won't apply here.
    if palette_name and palette_name != 'auto':
        from .pipeline import set_default_palette
        set_default_palette(palette_name)

    _cluster_cfg = {'method': cluster_method, 'resolution': resolution,
                    'n_metaclusters': n_metaclusters,
                    'reproducible': reproducible}
    if batch_mode == 'concatenate':
        published = _run_concatenated(
            trial_dirs, out_dir, groups, fmo_sets,
            k, fmo_percentile, labels, gate_overrides, region_gates,
            skip_existing, workers, n_jobs, max_events,
            admission_gb, vram_admission_gb,
            filter_debris_um, bead_um, doublet_tol, report,
            random_state=random_state, pairs=pairs,
            cytonorm=cytonorm_dict, cluster_cfg=_cluster_cfg)
    else:
        published = _run_independent(
            trial_dirs, out_dir, groups, fmo_sets,
            k, fmo_percentile, labels, gate_overrides, region_gates,
            skip_existing, workers, n_jobs, max_events,
            admission_gb, vram_admission_gb,
            filter_debris_um, bead_um, doublet_tol, report,
            random_state=random_state, pairs=pairs,
            cytonorm=cytonorm_dict, cluster_cfg=_cluster_cfg)

    print(f"\nDone. Results in: {out_dir}")
    if not published:
        print("[!] Some group comparisons were not written; see the messages "
              "above.", flush=True)
    if show_plots:
        # If nothing fresh was drawn (e.g. every sample was --skip-existing'd
        # and no group-UMAPs ran because there weren't enough live samples
        # to UMAP), reload the saved group-level PNGs from disk so the user
        # can still review them interactively.
        if not plt.get_fignums():
            opened = _show_saved_group_plots(out_dir)
            if opened == 0:
                print("[Show plots] No figures to display — nothing fresh "
                      "this run and no saved group-level PNGs found in "
                      f"{out_dir}.", flush=True)
        plt.show()
    return 0 if published else 1


def _load_unmix_controls(spec):
    """Resolve the ``--unmix-controls`` spec (a JSON file path or inline JSON)
    to a ``{fluor: fcs_path}`` dict (with an optional ``unstained`` key)."""
    spec = (spec or '').strip()
    if not spec:
        raise ValueError("--unmix needs --unmix-controls (fluor→FCS mapping).")
    if os.path.isfile(spec):
        with open(spec, encoding='utf-8') as f:
            mapping = json.load(f)
    else:
        mapping = json.loads(spec)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("--unmix-controls must be a non-empty JSON object "
                         "mapping fluor names to FCS paths.")
    return {str(k): str(v) for k, v in mapping.items()}


def run_batch_unmix(args) -> int:
    """Spectral batch-unmixing CLI mode. Builds reference spectra from the
    single-stain controls, unmixes each input FCS into per-fluor abundance
    columns (written as CSV), and writes a QC report (similarity + spillover
    spread, Markdown + JSON) plus a reference-spectra plot. Returns 0 on
    success, non-zero on a setup error."""
    import glob

    import pandas as pd

    from .pipeline import FlowSample
    from .spectral import (
        build_reference_spectra,
        measured_signal,
        unmix,
        unmixing_qc,
    )

    out_dir = args.out or 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    try:
        controls = _load_unmix_controls(args.unmix_controls)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        print(f"[unmix] bad --unmix-controls: {exc}", flush=True)
        return 2

    unstained_path = controls.pop('unstained', None)
    if not controls:
        print("[unmix] no single-stain controls given (only 'unstained').",
              flush=True)
        return 2

    # Load the single-stain controls and inputs: acquisition QC, then the
    # detectors as measured (linear, never compensated -- spectral data is
    # unmixed instead). This read `s.raw`, which run_qc did not filter, so
    # QC had no effect on anything unmixed here.
    def _load_raw(path):
        s = FlowSample(path)
        s.run_qc()
        return s

    single_samples = {}
    for fluor, path in controls.items():
        if not os.path.isfile(path):
            print(f"[unmix] control for {fluor!r} not found: {path}",
                  flush=True)
            return 2
        single_samples[fluor] = _load_raw(path)
    first = next(iter(single_samples.values()))

    # Detectors: 'auto' → the first control's fluorescence channels.
    if args.unmix_detectors and args.unmix_detectors != 'auto':
        detectors = [d.strip() for d in args.unmix_detectors.split(',')
                     if d.strip()]
    else:
        detectors = list(getattr(first, 'fluor_channels', []) or [])
    if len(detectors) < 2:
        print("[unmix] need ≥2 detector channels (got "
              f"{len(detectors)}). Use --unmix-detectors.", flush=True)
        return 2

    stains = {}
    for fluor, s in single_samples.items():
        cols = [d for d in detectors if d in s.data.columns]
        if len(cols) != len(detectors):
            print(f"[unmix] control {fluor!r} is missing some detectors — "
                  "skipped.", flush=True)
            continue
        stains[fluor] = measured_signal(s, detectors)
    if len(stains) < 1:
        print("[unmix] no usable single-stain controls.", flush=True)
        return 2

    un = None
    if unstained_path and os.path.isfile(unstained_path):
        us = _load_raw(unstained_path)
        cols = [d for d in detectors if d in us.data.columns]
        if len(cols) == len(detectors):
            un = measured_signal(us, detectors)

    spectra_diag = []
    spectra, fluors = build_reference_spectra(
        stains, unstained=un, detectors=detectors, diagnostics=spectra_diag,
        method=getattr(args, 'unmix_spectra', None) or 'auto',
        beads=bool(getattr(args, 'unmix_beads', False)))
    print(f"[unmix] built {len(fluors)} reference spectra over "
          f"{len(detectors)} detectors.", flush=True)
    for d in spectra_diag:
        if d['method'] == 'matched':
            print(f"[unmix] {d['fluor']}: spectrum estimated against matched "
                  f"autofluorescence ({d['reason']}).", flush=True)
        if d['warning']:
            print(f"[unmix] [!] {d['fluor']} is too dim to define a spectrum "
                  f"reliably: {d['warning']}. Use beads or a brighter fluor.",
                  flush=True)

    # Resolve the inputs to unmix.
    raw_in = args.unmix_input or args.fcs or args.trials or ''
    inputs = []
    for tok in raw_in.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if os.path.isdir(tok):
            inputs.extend(sorted(glob.glob(os.path.join(tok, '*.fcs'))))
        elif os.path.isfile(tok):
            inputs.append(tok)
    # Don't unmix the control files themselves.
    control_paths = {os.path.abspath(p) for p in controls.values()}
    if unstained_path:
        control_paths.add(os.path.abspath(unstained_path))
    inputs = [p for p in inputs if os.path.abspath(p) not in control_paths]

    qc_stains = dict(stains)
    if un is not None and 'Autofluorescence' in fluors:
        qc_stains['Autofluorescence'] = un
    qc = unmixing_qc(qc_stains, spectra, fluors, unstained=un,
                     reference_spectra=spectra_diag)

    # Write the QC report (Markdown + JSON) and the reference-spectra plot.
    _write_unmix_qc(out_dir, qc, spectra, fluors, detectors)

    n_done = 0
    for path in inputs:
        try:
            s = _load_raw(path)
            cols = [d for d in detectors if d in s.data.columns]
            if len(cols) != len(detectors):
                print(f"[unmix] {os.path.basename(path)}: missing detectors — "
                      "skipped.", flush=True)
                continue
            A = unmix(measured_signal(s, detectors), spectra,
                      nonneg=bool(args.unmix_nonneg))
            stem = os.path.splitext(os.path.basename(path))[0]
            out_csv = os.path.join(out_dir, f'{stem}_unmixed.csv')
            ucols = [f'U:{f}' for f in fluors]
            out = pd.DataFrame(A, columns=pd.Index(ucols))
            # Unmixed from the measured (linear) detectors, so every column is
            # linear; say so, so File > Load CSV does not have to guess (an
            # earlier record goes first, so a failed write can't leave it
            # beside new data).
            from .pipeline import (
                discard_transforms_sidecar,
                write_transforms_sidecar,
            )
            discard_transforms_sidecar(out_csv)
            out.to_csv(out_csv, index=False)
            write_transforms_sidecar(out_csv, {}, ucols, len(out))
            n_done += 1
        except Exception as exc:
            print(f"[unmix] {os.path.basename(path)}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    cond = qc['condition_number']
    cond_txt = "inf" if cond == float('inf') else f"{cond:.1f}"
    print(f"[unmix] done: {n_done} sample(s) unmixed → {out_dir} "
          f"(condition number {cond_txt}, "
          f"{len(qc['similar_pairs'])} similar pair(s)).", flush=True)
    if not inputs:
        print("[unmix] (no input FCS to unmix — built spectra + QC only; "
              "pass --unmix-input).", flush=True)
    return 0


def _write_unmix_qc(out_dir, qc, spectra, fluors, detectors):
    """Write the spectral-QC Markdown + JSON report and a reference-spectra
    PNG into ``out_dir``."""
    import numpy as np

    cond = qc['condition_number']
    cond_txt = "inf" if cond == float('inf') else f"{cond:.2f}"
    md = ["# Spectral unmixing QC", "",
          f"- **fluors**: {len(fluors)}",
          f"- **detectors**: {len(detectors)}",
          f"- **condition_number**: {cond_txt}", ""]
    # A fluor with no usable spectrum is never among the similar pairs, so
    # the report would otherwise read as a well-separated panel.
    deg = list(qc.get('degenerate_fluors') or [])
    if deg:
        md += ["## Unusable reference spectra",
               "No usable spectrum (all zero, e.g. a stain no brighter than "
               "the unstained control), so these cannot be unmixed: "
               + ', '.join(deg), ""]
    md.append("## Spectrally-similar pairs")
    if qc['similar_pairs']:
        md += ["| Fluor A | Fluor B | Cosine similarity |", "|---|---|---|"]
        md += [f"| {d['fluor_a']} | {d['fluor_b']} | {d['similarity']:.4f} |"
               for d in qc['similar_pairs']]
    else:
        md.append("None above threshold.")
    md += ["", "## Largest spillover spread"]
    if qc['worst_spread']:
        md += ["| Into | From | Spread |", "|---|---|---|"]
        md += [f"| {d['into']} | {d['from']} | {d['spread']:.4g} |"
               for d in qc['worst_spread']]
    else:
        md.append("No measured spread (single-stain controls missing).")
    # How each spectrum was estimated, and which controls are too dim to
    # trust -- a contaminated spectrum looks like any other in the plot.
    from .spectral import reference_spectra_markdown
    ref_report = list(qc.get('reference_spectra') or [])
    if ref_report:
        md += [""] + reference_spectra_markdown(ref_report)
    with open(os.path.join(out_dir, 'spectral_qc.md'), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(md) + "\n")

    payload = {
        'format': 'openflo-spectral-qc', 'version': 1,
        'fluors': list(fluors), 'detectors': list(detectors),
        'condition_number': (None if cond == float('inf') else cond),
        'similarity': np.asarray(qc['similarity']).tolist(),
        'ssm': [[None if not np.isfinite(v) else float(v) for v in row]
                for row in np.asarray(qc['ssm'])],
        # The usual SSM table has the single stain on the rows; this one is
        # the transpose, so say which index is which.
        'ssm_orientation': 'ssm[i][j] = spread into fluors[i] from the '
                           'single stain fluors[j]',
        'similar_pairs': qc['similar_pairs'],
        'worst_spread': qc['worst_spread'],
        'degenerate_fluors': deg,
        'reference_spectra': ref_report,
        'dim_controls': [d['fluor'] for d in ref_report if d['warning']],
    }
    with open(os.path.join(out_dir, 'spectral_qc.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, fl in enumerate(fluors):
            ax.plot(range(spectra.shape[1]), spectra[i], marker='o', ms=2,
                    lw=1.2, label=fl)
        ax.set_xlabel('detector')
        ax.set_ylabel('normalized signal')
        ax.set_title('Reference spectra')
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'reference_spectra.png'), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[unmix] spectra plot skipped: {exc}", flush=True)


def main() -> int:
    """argparse-driven entry point for the ``openflo-run`` console script.

    Mirrors the historical ``python run_analysis.py ...`` invocation
    exactly. Returns 0 on clean exit; non-zero from argparse or an
    unhandled exception bubbles via sys.exit / SystemExit.
    """
    force_utf8_streams()
    import multiprocessing
    multiprocessing.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument('--trials',      default='',
                    help='Comma-separated list of FCS trial directories')
    ap.add_argument('--fcs',         default='',
                    help='Single FCS directory (alias for --trials with one folder)')
    ap.add_argument('--out',         default='outputs', help='Output directory')
    ap.add_argument('--groups',      default='',
                    help='JSON list of groups: '
                         '\'[{"name":"Day1","samples":["d1m1","d1m2"],'
                         '"fmo_set":"standard"}, ...]\'. Replaces --samples '
                         'when given.')
    ap.add_argument('--fmo-sets',    default='', dest='fmo_sets',
                    help='JSON dict of FMO sets: '
                         '\'{"standard": {"FL1-A":"fmo_fl1", ...}}\'. '
                         'Defaults to the built-in example sets.')
    ap.add_argument('--samples',     default='',
                    help='LEGACY: comma-separated sample names; names starting '
                         'with "late" go to "Group B", others to "Group A". '
                         'Use --groups for full control.')
    ap.add_argument('--batch-mode',  default='independent',
                    choices=['independent', 'concatenate'], dest='batch_mode',
                    help='independent: one analysis per trial | '
                         'concatenate: merge trials, cells retain trial-of-origin label')
    ap.add_argument('--workers',     type=int, default=1,
                    help='Number of samples to process in parallel (default 1)')
    ap.add_argument('--cluster-method', default='phenograph',
                    choices=['phenograph', 'leiden', 'flowsom'],
                    dest='cluster_method',
                    help='Clustering algorithm (default phenograph).')
    # Reproducible is now the DEFAULT. `--reproducible` is kept so existing
    # scripts and pipelines that pass it keep working; it is simply a no-op.
    ap.add_argument('--reproducible', action='store_true',
                    help='(default since 2.4.9; kept for compatibility) Seed '
                         'PhenoGraph clustering via its Leiden backend so a '
                         're-run is bit-identical.')
    ap.add_argument('--no-reproducible', action='store_true',
                    dest='no_reproducible',
                    help="Use PhenoGraph's unseeded Louvain instead. Re-running "
                         'the same data can then give different labels, but it '
                         'reproduces prior Louvain output and re-enables the GPU '
                         'clustering path.')
    ap.add_argument('--resolution', type=float, default=1.0,
                    help='Leiden resolution — higher = more clusters '
                         '(only used with --cluster-method leiden).')
    ap.add_argument('--n-metaclusters', type=int, default=10,
                    dest='n_metaclusters',
                    help='FlowSOM metaclusters (only with '
                         '--cluster-method flowsom).')
    ap.add_argument('--max-events',  type=int, default=0, dest='max_events',
                    help='Cap clustering at this many events per sample '
                         '(0 = no cap). When exceeded, the sample is '
                         'clustered on a random sub-sample and the rest are '
                         'assigned via nearest-neighbour. Recommended: '
                         '~500000. The group comparisons cluster every '
                         'sample of an analysis together, once (once per '
                         'trial folder with --batch-mode independent), on '
                         'at most this many events IN TOTAL (an equal share '
                         'from each '
                         'sample; with 0, as many as the largest sample '
                         'holds, at most 200000 for Leiden and 50000 for '
                         'FlowSOM), and assign the rest the same way.')
    ap.add_argument('--admission-ram', type=float, default=4.0, dest='admission_gb',
                    help='Free RAM (GB) required before submitting a new '
                         'parallel worker. 0 disables admission control.')
    ap.add_argument('--admission-vram', type=float, default=1.0, dest='vram_admission_gb',
                    help='Free VRAM (GB) required for a worker to take the GPU '
                         'clustering branch (cuML+cuGraph). Below this it uses '
                         'CPU Phenograph. 0 = always try GPU when available.')
    ap.add_argument('--filter-debris-um', type=float, default=0.0,
                    dest='filter_debris_um',
                    help='Drop events whose FSC-A falls below this size in µm. '
                         '0 disables the debris filter. The conversion µm→FSC '
                         'is calibrated per trial from the comp-bead controls '
                         '(see --bead-um). Recommended: 4.0')
    ap.add_argument('--bead-um', type=float, default=8.0, dest='bead_um',
                    help='Diameter (µm) of the comp-bead control used for FSC '
                         'calibration. Default 8.0 (BD CompBeads / Invitrogen '
                         'UltraComp). Spherotech is 7.5.')
    ap.add_argument('--doublet-tol', type=float, default=0.0, dest='doublet_tol',
                    help='Doublet exclusion: keep events whose FSC-A/FSC-H is '
                         'within ±tol of the population median. 0 disables. '
                         'Recommended: 0.25 for polyploid samples, 0.15 '
                         'for typical diploid leukocytes.')
    ap.add_argument('--k',              type=int,   default=30,   help='Phenograph k')
    ap.add_argument('--fmo-percentile', type=float, default=99.5, dest='fmo_percentile')
    ap.add_argument('--labels', default='',
                    help='Channel labels: "DetA=LabelA;DetB=LabelB"')
    ap.add_argument('--panel', default='',
                    help='Path to a staining-panel .xlsx (CD↔fluorophore '
                         'rows). Resolved to {detector: CD} labels and '
                         'merged with --labels (--labels wins on conflict). '
                         'Pass "auto" to search the trial folder(s) and a '
                         'few ancestor levels for one.')
    ap.add_argument('--pairs', default='',
                    help='Marker pairs for the per-group overlay + grid '
                         'scatters, e.g. "CD902/CD901b,CD901b/CD903,CD902/CD903". '
                         'Default: those three. Labels must be assigned '
                         '(via --panel / --labels) for a pair to render.')
    ap.add_argument('--batch-correct', action='store_true', dest='batch_correct',
                    help='CytoNorm batch-normalize across batches (each '
                         'trial/day folder) before clustering.')
    ap.add_argument('--cytonorm-mode', default='goal',
                    choices=['goal', 'controls'], dest='cytonorm_mode',
                    help="'goal' (CytoNorm 2.0, control-free, default) fits on "
                         "all samples; 'controls' (classic) fits only on "
                         "per-batch control samples — see --cytonorm-control.")
    ap.add_argument('--cytonorm-control', default='', dest='cytonorm_control',
                    help="Filename token identifying the per-batch control "
                         "sample(s) for --cytonorm-mode controls (e.g. 'ref').")
    ap.add_argument('--cytonorm-metaclusters', type=int, default=10,
                    dest='cytonorm_metaclusters',
                    help='FlowSOM metaclusters for CytoNorm (default 10).')
    # ── Spectral batch-unmixing mode (full-spectrum cytometers) ──────────
    ap.add_argument('--unmix', action='store_true',
                    help='Batch spectral-unmixing mode: build reference '
                         'spectra from single-stain controls and unmix a set '
                         'of FCS into per-fluor abundances + a QC report. '
                         'Bypasses the standard analysis pipeline.')
    ap.add_argument('--unmix-controls', default='', dest='unmix_controls',
                    help='Single-stain controls for --unmix: a path to a JSON '
                         'file OR inline JSON mapping fluor→FCS path, e.g. '
                         '\'{"FL4":"ss_fl4.fcs","FL5":"ss_fl5.fcs",'
                         '"unstained":"unstained.fcs"}\'. The optional '
                         '"unstained" key adds an autofluorescence endmember.')
    ap.add_argument('--unmix-input', default='', dest='unmix_input',
                    help='FCS to unmix in --unmix mode: a directory or a '
                         'comma-separated list of files. Defaults to '
                         '--fcs / --trials.')
    ap.add_argument('--unmix-detectors', default='auto', dest='unmix_detectors',
                    help="Detector channels for --unmix: 'auto' (default — the "
                         "fluorescence channels of the first control) or a "
                         "comma-separated channel list.")
    ap.add_argument('--unmix-nonneg', action='store_true', dest='unmix_nonneg',
                    help='Clip unmixed abundances at 0 (non-negative).')
    ap.add_argument('--unmix-spectra', default='auto', dest='unmix_spectra',
                    choices=('auto', 'total', 'matched'),
                    help="How --unmix estimates each reference spectrum: "
                         "'total' (brightest events by total signal minus "
                         "the mean autofluorescence), 'matched' (brightest "
                         "by the dye's own signal, each minus the "
                         "autofluorescence of matched unstained events), or "
                         "'auto' (default: 'total' unless the two disagree "
                         "or the dye sits inside the autofluorescence band, "
                         "as for a dim dye on cells with variable "
                         "autofluorescence).")
    ap.add_argument('--unmix-beads', action='store_true', dest='unmix_beads',
                    help="The --unmix single-stain controls are beads: "
                         "estimate their spectra by total signal (no "
                         "cellular autofluorescence to match).")
    ap.add_argument('--show-plots',  action='store_true', dest='show_plots')
    ap.add_argument('--gates',       default='',
                    help='JSON list of gate dicts, e.g. '
                         '\'[{"kind":"threshold","channel":"FL1-A","value":0.5}]\'. '
                         'Fluorescence coordinates are on the logicle scale '
                         'the run applies (0.11 is zero, 1.0 is 262,144; not '
                         'intensities); '
                         'scatter and Time are linear.')
    ap.add_argument('--skip-existing', action='store_true', dest='skip_existing',
                    help='Resume an interrupted run: a sample whose '
                         '<group>/<name>_stats.csv exists is not processed '
                         'again. The group comparisons are still recomputed '
                         'on every run: a kept sample\'s events are reloaded '
                         '(not re-clustered), and the comparisons are '
                         'written only when every sample of every group is '
                         'present. A group with a kept sample keeps its '
                         'existing UMAP and pair scatters.')
    ap.add_argument('--export-wsp', default='', dest='export_wsp',
                    help='After the run, write a FlowJo-compatible .wsp '
                         'workspace at this path. Contains the gates '
                         'that were applied (FMO thresholds + --gates) '
                         'and one SampleNode per input FCS.')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed used for sample subsampling and '
                         'UMAP. Held constant across every sample in a '
                         'trial so results are reproducible across runs.')
    ap.add_argument('--palette', default='auto',
                    help='Categorical palette for per-sample / per-condition '
                         'plot colouring. Default: auto (tab10 for ≤10 groups, '
                         'tab20 for 11-20, gist_ncar above). Common '
                         'alternatives: tab10, Set1, Set2, Dark2, Paired.')
    ap.add_argument('-v', '--verbose', action='count', default=0,
                    help='Increase log verbosity (repeat for more: -v=INFO '
                         '[default], -vv=DEBUG). Mutually exclusive with -q.')
    ap.add_argument('-q', '--quiet', action='store_true',
                    help='Suppress INFO messages — only WARNING and above.')
    args = ap.parse_args()

    # Configure the root logger ONCE based on -v / -q. Done before any
    # `from .pipeline import ...` runs in this process so worker subprocesses
    # inherit a sensible default via env. The actual handlers / format are
    # left at defaults — users wanting JSON / file output should configure
    # `logging` themselves before importing openflo.
    if args.quiet:
        _log_level = logging.WARNING
    elif args.verbose >= 2:
        _log_level = logging.DEBUG
    else:
        _log_level = logging.INFO
    logging.basicConfig(
        level=_log_level,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True,    # override any prior configuration
    )

    # Spectral batch-unmixing is a self-contained mode — run it and exit
    # before the standard analysis-pipeline argument resolution.
    if args.unmix:
        return run_batch_unmix(args)

    # Resolve trial dirs: --trials takes precedence, else --fcs
    raw_trials = args.trials or args.fcs or '.'
    trial_dirs = [t.strip() for t in raw_trials.split(',') if t.strip()]

    # Resolve groups / fmo_sets — four paths in order of preference:
    #   1. --groups (explicit) → use as-is. Pair with --fmo-sets if given.
    #   2. --samples (legacy) → auto-split by "late" prefix into two groups.
    #   3. Neither, but folders contain FCS → by-DAY auto-grouping: each
    #      folder (incl. sub-folders of a parent you point at) becomes a
    #      day/group, sampled independently, compared across days in one
    #      analysis. This is the new default.
    #   4. Fallback → a single generic group over the named samples.
    if args.groups:
        groups = json.loads(args.groups)
        fmo_sets = (json.loads(args.fmo_sets)
                    if args.fmo_sets else DEFAULT_FMO_SETS)
    elif args.samples:
        names = [s.strip() for s in args.samples.split(',') if s.strip()]
        # Optional two-group split on a name prefix (override with --groups).
        group_b = [n for n in names if n.lower().startswith('late')]
        group_a = [n for n in names if not n.lower().startswith('late')]
        groups = []
        if group_a:
            groups.append({'name': 'Group A',
                           'samples': group_a, 'fmo_set': 'Set A'})
        if group_b:
            groups.append({'name': 'Group B',
                           'samples': group_b, 'fmo_set': 'Set B'})
        fmo_sets = (json.loads(args.fmo_sets)
                    if args.fmo_sets else DEFAULT_FMO_SETS)
    else:
        auto = _auto_groups_by_day(trial_dirs)
        if auto:
            groups = auto
            fmo_sets = (json.loads(args.fmo_sets)
                        if args.fmo_sets else DEFAULT_FMO_SETS)
            print(f"[grouping] By-day: {len(auto)} folder(s) → "
                  f"{', '.join(g['name'] for g in auto)}", flush=True)
            # Each group carries its own trial_dir, so collapse the run to
            # a SINGLE pass (one combined analysis comparing days). The
            # representative trial_dir below is only used for labels /
            # bead calibration fallbacks.
            trial_dirs = [trial_dirs[0] if trial_dirs else '.']
        else:
            groups   = DEFAULT_GROUPS
            fmo_sets = DEFAULT_FMO_SETS

    # Refuse an ambiguous group spec before anything reads an FCS (the
    # --panel probe below does), naming where the groups came from.
    norm = _normalise_groups(groups)
    problems = (_group_spec_problems(norm)
                + _sample_file_problems(norm, trial_dirs))
    if problems:
        source = ('--groups' if args.groups else
                  '--samples' if args.samples else 'groups from the folders')
        for p in problems:
            print(f'[!] {source}: {p}', flush=True)
        return 2

    gate_overrides, region_gates = _parse_gates_arg(args.gates)
    parsed_overrides = gate_overrides   # keep the dict form for the wsp export
    if not gate_overrides and not region_gates:
        gate_overrides = None   # preserve the original "no gates" semantics

    # Channel labels: staining-panel xlsx first, then --labels overrides.
    labels = {}
    panel_path = args.panel
    if panel_path == 'auto':
        panel_path = find_panel_xlsx(trial_dirs)
        if panel_path:
            print(f"[panel] Auto-detected: {panel_path}", flush=True)
    if panel_path:
        try:
            # Normalise first — the raw --groups spec may carry `samples` as a
            # comma-string or per-sample dicts, so groups[0]['samples'][0] would
            # be a single char / a dict and the probe would silently fail.
            _ng = _normalise_groups(groups)
            probe_dir = _ng[0].get('trial_dir') or trial_dirs[0]
            probe = fcs_path(probe_dir, _ng[0]['samples'][0])
            channels = list(FlowSample(probe).data.columns)
            labels.update(read_staining_panel(panel_path, channels))
        except Exception as exc:
            print(f"[panel] Could not apply panel: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    labels.update(parse_labels(args.labels))   # explicit --labels win
    pairs = parse_pairs(args.pairs)

    # Apply the user's palette choice process-wide before any plot is
    # rendered. Worker processes inherit via os.environ-free fork so the
    # function reset below is enough for the parent; per-worker callers
    # also re-set after their imports (see _process_sample_task).
    if args.palette and args.palette != 'auto':
        from .pipeline import set_default_palette
        set_default_palette(args.palette)

    rc = run(
        trial_dirs     = trial_dirs,
        out_dir        = args.out,
        groups         = groups,
        fmo_sets       = fmo_sets,
        batch_mode     = args.batch_mode,
        k              = args.k,
        cluster_method = args.cluster_method,
        resolution     = args.resolution,
        n_metaclusters = args.n_metaclusters,
        reproducible   = not args.no_reproducible,
        fmo_percentile = args.fmo_percentile,
        labels         = labels,
        gate_overrides = gate_overrides,
        region_gates   = region_gates,
        show_plots     = args.show_plots,
        skip_existing  = args.skip_existing,
        workers        = args.workers,
        max_events     = args.max_events or None,
        admission_gb   = args.admission_gb,
        vram_admission_gb = args.vram_admission_gb,
        filter_debris_um = args.filter_debris_um,
        bead_um          = args.bead_um,
        doublet_tol      = args.doublet_tol,
        random_state     = args.seed,
        palette_name     = args.palette,
        pairs            = pairs,
        batch_correct    = args.batch_correct,
        cytonorm_mode    = args.cytonorm_mode,
        cytonorm_control = args.cytonorm_control,
        cytonorm_metaclusters = args.cytonorm_metaclusters,
    )

    if args.export_wsp:
        _export_pipeline_workspace(
            out_path       = args.export_wsp,
            trial_dirs     = trial_dirs,
            groups         = groups,
            gate_overrides = parsed_overrides,
            region_gates   = region_gates)

    return rc


if __name__ == '__main__':
    raise SystemExit(main())
