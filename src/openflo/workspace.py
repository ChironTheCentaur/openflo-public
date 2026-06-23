"""Pipeline Workspace — single-window grouping / comparison.

The workspace lives docked in the gate editor behind a draggable sash. Items
(samples, gate *leaves*/populations, or whole trials) are dragged in from the
editor's Samples & Gates tree. The workspace is a single tree split into three
channels:

    c1 (#0 tree)      c2 (comp)              c3 (fmo)
    group → samples   metadata / comp-beads  FMOs

  • Groups bundle items (e.g. "all of trial 1"). A group's comp/FMO cascades to
    its members; an item can override its own.
  • Compensation precedence: item override → group override → sample's own
    metadata matrix → none. Dropped comp *beads* OVERRIDE the metadata matrix
    (they become the basis for a later-derived matrix). Each item/group needs a
    matrix OR beads; rows with neither are flagged "⚠ none".
  • FMOs are optional (they also feed FMO-based auto-gating later).
  • Display model: an item shows in the preview when its own ☑ AND its group's
    ☑ are on (``WorkspaceModel.displayed_items``).

Beads/FMOs are supplied two ways: drag bead/FMO FCS from the editor onto a
row's comp/fmo column, or the "Set comp beads…" / "Set FMOs…" picker buttons.
Only file references travel — no event data — so many trials stay cheap.

Architecture — model / view split:
  • ``WorkspaceModel``           — plain data + logic (no Tk; unit-tested).
  • ``PipelineWorkspaceView``    — disposable Tk renderer over a model.
  • ``WorkspacePanel``           — hosts the single view + Pop-out / Import-all.
"""

from __future__ import annotations

import collections
import copy
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import ttk

# Marker prefixing the workspace's own progress lines on a job's stdout, so the
# parent can show them and ignore the backend's chatter (phenograph, warnings).
_PROG = '\x01WSPROG\x01'

# ── iid encoding for editor-originated nodes ───────────────────────────────
#   Sample: 'S:<name>'   Gate: 'G:<name>/<gid>'   Comp: 'C:<name>'


def sample_iid(name: str) -> str:
    return f'S:{name}'


def gate_iid(name: str, gid: str) -> str:
    return f'G:{name}/{gid}'


def comp_iid(name: str) -> str:
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


def derive_trial_name(path) -> str:
    """Trial / group name for an FCS ``path``.

    Time-course flow experiments are organised by collection day, but the
    'Day N' token sits at an inconsistent depth on disk — sometimes the
    immediate experiment folder (``…/2024-01-15 Day 3 Check/x.fcs``), sometimes a
    grandparent (``…/experiment-a/day 0 raw/Blank Experiment…/x.fcs``).
    So we scan the file's ancestor folders from the **nearest upward** and group
    by the first 'Day N' we find, normalised to ``"Day N"``. This keeps every
    sample from one collection day together (specimens + compensation controls
    alike) no matter how the surrounding folders are named — and avoids the old
    failure where a whole drop collapsed under one shared grandparent
    (e.g. everything sharing one study-root folder name).

    When no day token is present anywhere above the file we fall back to the
    historical heuristic: the **grandparent** folder (the immediate host is
    usually a generic 'Blank Experiment with…'), then the parent, then 'Trial'.
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

    # Fallback: grandparent, then parent.
    host = os.path.dirname(p)
    grand = os.path.dirname(host)
    for cand in (os.path.basename(grand), os.path.basename(host)):
        if cand:
            return cand
    return 'Trial'


_COMP_TOKENS = ('comp', 'control', 'stain')   # 'stain' covers (un)stained


def is_comp_sample(name) -> bool:
    """True if a sample name looks like a compensation control — FlowJo writes
    these as 'Compensation Controls_…', '…Stained Control', 'Unstained Control',
    etc. Used to split an imported day group into Comps vs Samples subgroups."""
    if not name:
        return False
    low = str(name).lower()
    return any(tok in low for tok in _COMP_TOKENS)


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
    path = getattr(s, 'path', None)
    trial = (getattr(editor, '_sample_trial', {}) or {}).get(name) or derive_trial_name(path)
    return {
        'name':           name,
        'trial':          trial,
        'path':           path,
        'color':          (getattr(editor, '_sample_colors', {}) or {}).get(name, '#000000'),
        'comp_matrix':    getattr(s, 'comp_matrix', None),
        'comp_channels':  list(getattr(s, 'comp_channels', None) or []),
        'channel_labels': dict(getattr(editor, '_channel_labels', {}) or {}),
        'gates':          copy.deepcopy(dict(gates)),
        'gate_order':     list(gate_order),
    }


def extract_editor_context(editor) -> list[dict]:
    """Snapshot every loaded sample in the gate ``editor`` (in display order)."""
    out: list[dict] = []
    for name in (getattr(editor, '_sample_order', []) or []):
        ctx = extract_one_context(editor, name)
        if ctx is not None:
            out.append(ctx)
    return out


def gate_is_leaf(gates: dict, gid: str) -> bool:
    return not any(g.get('parent_id') == gid for g in gates.values())


def gate_chain(gates: dict, gid: str) -> list[str]:
    """Root→leaf gate-id list (a leaf is constricted by all ancestor gates)."""
    chain: list[str] = []
    seen: set[str] = set()
    cur = gid
    while cur and cur in gates and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        cur = gates[cur].get('parent_id')
    chain.reverse()
    return chain


def gate_path(gates: dict, gid: str) -> str:
    from .pipeline import describe_gate
    return ' / '.join(describe_gate(gates[g]) for g in gate_chain(gates, gid))


def build_drop_payload(ctx: dict, gid: str | None) -> dict:
    """Describe a pipeline item dropped from the editor (whole sample if
    ``gid`` is None, else the population at ``gid``). Carries trial, comp matrix
    + channels, labels, and the cumulative gate chain."""
    gates = ctx.get('gates', {}) or {}
    chain = gate_chain(gates, gid) if gid else []
    return {
        'sample':         ctx['name'],
        'trial':          ctx.get('trial', 'Trial'),
        'path':           ctx.get('path'),
        'color':          ctx.get('color', '#000000'),
        'gate_id':        gid,
        'gate_path':      gate_path(gates, gid) if gid else None,
        'is_leaf':        gate_is_leaf(gates, gid) if gid else None,
        'gate_chain':     chain,
        'gates':          {g: gates[g] for g in chain},
        'comp_matrix':    ctx.get('comp_matrix'),
        'comp_channels':  list(ctx.get('comp_channels') or []),
        'channel_labels': dict(ctx.get('channel_labels') or {}),
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
    'k': 30,              # Phenograph nearest-neighbours
    'max_events': 5000,   # subsample cap PER RUN UNIT (small = fast); 0 = all
    'seed': 42,
    'umap': True,
    'trimap': True,
    # Run unit: a GROUP's samples are concatenated into ONE co-embedded UMAP
    # (events tagged by source sample). 'concatenate' merges ALL groups into a
    # single UMAP so groups can be compared in one embedding (FlowJo-style);
    # otherwise each group (and each loose item) is its own UMAP.
    'concatenate': False,
}


def default_run_cfg() -> dict:
    return dict(DEFAULT_RUN_CFG)


def run_label(item: dict) -> str:
    """Filesystem-safe label for an item's output files."""
    base = f"{item.get('trial', 'Trial')}_{item.get('sample', 'sample')}"
    if item.get('gate_id'):
        base += f"_{item['gate_id']}"
    return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in base) or 'item'


def proper_run_channels(sample) -> list[str]:
    """The PROPER marker channels for clustering/embedding: the sample's fluor
    channels with height/width detector versions dropped — so UMAP/TriMap use
    ONE column per marker (the area ``-A`` / label-only measurement) instead of
    every ``-A``/``-H``/``-W`` version of each fluorophore (which are collinear
    and distort the embedding). Falls back to all fluor channels if the filter
    would leave nothing (e.g. a cytometer that doesn't use the -A/-H/-W scheme).

    Scatter (FSC/SSC), Time, and analysis columns are already excluded upstream
    by ``FlowSample.fluor_channels``."""
    fl = list(getattr(sample, 'fluor_channels', None) or [])

    def _secondary(c: str) -> bool:
        cl = c.strip()
        return (cl.endswith('-H') or cl.endswith('-W')
                or cl.endswith(' H') or cl.endswith(' W')
                or cl.endswith('-Height') or cl.endswith('-Width'))

    primary = [c for c in fl if not _secondary(c)]
    return primary or fl


def resolve_run_sample(editor, item: dict):
    """Return ``(FlowSample, note)`` ready to cluster for a workspace ``item``.

    Prefers the editor's live, already comp+transformed sample (shallow copy +
    deep-copied ``.data`` so we never mutate editor state); else loads the FCS
    from ``path`` and best-effort applies the item's metadata comp matrix +
    logicle transform. If the item is a gated population, ``.data`` is filtered
    to that population's cumulative gate mask. Raises only on an unrecoverable
    load (no live sample AND no usable path)."""
    import copy as _copy

    from .pipeline import FlowSample, cumulative_gate_mask

    name = item.get('sample')
    samples = getattr(editor, '_samples', {}) or {}
    live = samples.get(name)
    if live is not None:
        s = _copy.copy(live)
        s.data = live.data.copy(deep=True)
        note = 'live editor sample'
    else:
        path = item.get('path')
        if not path or not os.path.exists(path):
            raise RuntimeError('sample not loaded in editor and no readable FCS path')
        s = FlowSample(path)
        m = item.get('comp_matrix')
        ch = list(item.get('comp_channels') or [])
        if m is not None and ch:
            try:
                s.manual_compensate(m, ch)
            except Exception as e:  # noqa: BLE001 - best-effort match to editor
                from . import pipeline as _p
                _p.log.warning("  [workspace] comp on reload failed: %s", e)
        try:
            s.apply_transform()
        except Exception as e:  # noqa: BLE001
            from . import pipeline as _p
            _p.log.warning("  [workspace] transform on reload failed: %s", e)
        note = 'reloaded from file'

    # Never carry the editor's analysis columns/refs into our run.
    s.clusters = None
    s.umap_coords = None
    s.trimap_coords = None

    gid = item.get('gate_id')
    gates = item.get('gates') or {}
    if gid and gates:
        mask = cumulative_gate_mask(gates, gid, s.data)
        kept = int(mask.sum())
        s.data = s.data.loc[mask].reset_index(drop=True)
        note += f', gated {kept:,}/{len(mask):,}'
    return s, note


def _plot_embedding(sample, prefix: str, out_dir: str, label: str,
                    color_by: str = 'cluster', suffix: str = ''):
    """Render a 2-D embedding (``UMAP``/``TRIMAP``) to PNG via matplotlib's
    object API (thread-safe — no pyplot). ``color_by`` is a column: 'cluster'
    (numeric, tab20) or a categorical source column ('__group__'/'__sample__',
    distinct colours + legend — this is the group/sample COMPARISON view).
    Returns the written path or None."""
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception:
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
            cmap = matplotlib.colormaps.get_cmap('tab10' if len(cats) <= 10 else 'tab20')
            for i, cat in enumerate(cats):
                m = sub[col].astype(str) == cat
                ax.scatter(sub.loc[m, x], sub.loc[m, y], s=4, alpha=0.5,
                           linewidths=0, color=cmap(i % cmap.N), label=cat)
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
        fig.savefig(out)
        return out
    except Exception:
        return None


def _blank_result(label: str, n_events: int = 0) -> dict:
    return {'label': label, 'ok': False, 'note': '', 'n_events': int(n_events),
            'n_clusters': 0, 'umap': False, 'trimap': False, 'channels': [],
            'files': [], 'cancelled': False, 'error': None}


def build_run_units(model, cfg: dict) -> list[dict]:
    """Group the workspace's items into RUN UNITS (each → one co-embedded
    UMAP/cluster). A unit is ``{label, members}`` where each member is
    ``{item, group, sample}``.

    - ``concatenate`` → ONE unit containing every item (all groups + loose),
      so groups are compared in a single embedding.
    - otherwise → one unit per non-empty group (its samples co-embedded) plus
      one unit per loose (ungrouped) item.
    """
    def _safe(name: str) -> str:
        return ''.join(c if (c.isalnum() or c in '-_.') else '_'
                       for c in str(name)) or 'group'

    if cfg.get('concatenate'):
        members = []
        for g in model.groups.values():
            for it in g['items'].values():
                members.append({'item': it, 'group': g['name'], 'sample': it.get('sample')})
        for it in model.loose.values():
            members.append({'item': it, 'group': None, 'sample': it.get('sample')})
        return [{'label': 'all_groups', 'members': members}] if members else []

    units = []
    for g in model.groups.values():
        members = [{'item': it, 'group': g['name'], 'sample': it.get('sample')}
                   for it in g['items'].values()]
        if members:
            units.append({'label': _safe(g['name']), 'members': members})
    for it in model.loose.values():
        units.append({'label': run_label(it),
                      'members': [{'item': it, 'group': None, 'sample': it.get('sample')}]})
    return units


def prepare_unit(editor, members: list, label: str, cfg: dict) -> dict:
    """(Main process) Resolve every member sample, tag each event with its
    source (``__group__`` / ``__sample__``), concatenate, and subsample the
    COMBINED frame to ``max_events`` up front. Returns
    ``{data, channels, label, note, n_events, color_by}`` where ``color_by`` is
    the categorical column to colour embeddings by (``__group__`` when the unit
    spans multiple groups, ``__sample__`` for multiple samples in one group,
    else None → colour by cluster only). Raises if no member resolves."""
    import pandas as pd
    seed = int(cfg.get('seed', 42))
    frames, channels, groups, samples = [], None, set(), set()
    for m in members:
        sample, _note = resolve_run_sample(editor, m['item'])
        gname = m.get('group') or ''
        sname = m.get('sample') or m['item'].get('sample', 'sample')
        df = sample.data.copy()
        df['__group__'] = gname or '(ungrouped)'
        df['__sample__'] = (f'{gname} / {sname}' if gname else sname)
        frames.append(df)
        groups.add(gname or '(ungrouped)')
        samples.add(df['__sample__'].iat[0] if len(df) else sname)
        if channels is None:
            channels = proper_run_channels(sample)
    if not frames:
        raise RuntimeError('run unit has no resolvable members')
    data = pd.concat(frames, ignore_index=True, sort=False)
    max_ev = int(cfg.get('max_events') or 0) or None
    if max_ev and len(data) > max_ev:
        data = data.sample(n=max_ev, random_state=seed).reset_index(drop=True)
    color_by = ('__group__' if len(groups) > 1
                else ('__sample__' if len(samples) > 1 else None))
    return {'data': data, 'channels': channels or [], 'label': label,
            'note': f'{len(members)} sample(s)', 'n_events': int(len(data)),
            'color_by': color_by}


def prepare_run(editor, item: dict, cfg: dict) -> dict:
    """Single-item convenience wrapper around :func:`prepare_unit` (used by
    tests / non-grouped callers)."""
    return prepare_unit(editor, [{'item': item, 'group': None,
                                  'sample': item.get('sample')}],
                        run_label(item), cfg)


def compute_run(prep: dict, cfg: dict, out_dir: str,
                progress=None, should_cancel=None) -> dict:
    """(Process-agnostic) Cluster + optional UMAP/TriMap on a prepared frame and
    write outputs (cluster CSV + embedding PNGs). Runs either in-process or
    inside a child process. Exception-safe: returns a status dict, never raises.

    ``progress`` = ``callable(str)`` for live phase messages; ``should_cancel``
    = ``callable() -> bool`` checked at phase boundaries (the subprocess path
    cancels by terminating the process instead)."""
    from .pipeline import FlowSample
    label = prep.get('label', 'item')
    channels = list(prep.get('channels') or [])
    say = progress if callable(progress) else (lambda _m: None)
    cancelled = should_cancel if callable(should_cancel) else (lambda: False)
    res = _blank_result(label, prep.get('n_events', 0))
    res['note'] = prep.get('note', '')
    res['channels'] = list(channels)
    try:
        data = prep['data']
        n = int(len(data))
        res['n_events'] = n
        if n < 3:
            raise RuntimeError(f'too few events to cluster ({n})')
        if len(channels) < 2:
            raise RuntimeError(f'need ≥2 marker channels, got {len(channels)}: {channels}')

        sample = FlowSample.from_dataframe(data, name=label)
        seed = int(cfg.get('seed', 42))
        # Clamp neighbours below the event count so small sets can't make
        # Phenograph / UMAP raise (the classic small-n crash).
        k = max(2, min(int(cfg.get('k', 30)), n - 1))
        nn = max(2, min(int(cfg.get('k', 30)), n - 1))

        if cancelled():
            res['cancelled'] = True
            return res
        say(f'{label}: clustering {n:,} cells on {len(channels)} markers (k={k})…')
        sample.cluster(channels=channels, k=k, max_events=None, random_state=seed)
        labels = sample.data.get('cluster')
        if labels is not None:
            res['n_clusters'] = int(labels[labels >= 0].nunique())

        os.makedirs(out_dir, exist_ok=True)
        color_by = prep.get('color_by')          # '__group__' / '__sample__' / None
        if labels is not None:
            freq = (labels.value_counts().sort_index()
                    .rename_axis('cluster').reset_index(name='count'))
            freq['percent'] = (freq['count'] / max(1, n) * 100).round(3)
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

        # Standardise (z-score) each marker for the EMBEDDINGS only — clustering
        # already used the transformed values. Equal per-marker weight stops a
        # single high-variance marker from stretching the layout (UMAP hides
        # this by normalising its output; TriMap does NOT, so it matters most
        # there). This matches FlowJo's pre-scaling before dimensionality
        # reduction. Note: TriMap is global-structure preserving, so it will
        # still spread more / use a larger coordinate range than UMAP by design.
        for c in channels:
            col = sample.data[c].astype(float)
            sd = float(col.std())  # pyright: ignore[reportArgumentType]  # Series.std() is scalar
            if sd > 0:
                sample.data[c] = (col - col.mean()) / sd

        def _emb_plots(prefix):
            wrote = False
            p = _plot_embedding(sample, prefix, out_dir, label, color_by='cluster')
            if p:
                res['files'].append(p)
                wrote = True
            if color_by:                          # the group/sample comparison view
                p2 = _plot_embedding(sample, prefix, out_dir, label,
                                     color_by=color_by, suffix='_by_source')
                if p2:
                    res['files'].append(p2)
            return wrote

        if cfg.get('umap', True):
            if cancelled():
                res['cancelled'] = True
                return res
            say(f'{label}: UMAP on {len(channels)} markers… (first run compiles, ~30s)')
            # sample_n=0 → embed all rows (already subsampled up front).
            sample.run_umap(channels=channels, n_neighbors=nn, sample_n=0,
                            random_state=seed)
            res['umap'] = _emb_plots('UMAP')
        if cfg.get('trimap', True):
            if cancelled():
                res['cancelled'] = True
                return res
            say(f'{label}: TriMap on {len(channels)} markers… (first run compiles, ~20s)')
            sample.run_trimap(channels=channels, sample_n=0, random_state=seed)
            res['trimap'] = _emb_plots('TRIMAP')

        res['ok'] = True
    except Exception as e:  # noqa: BLE001 - per-item isolation is the point
        res['error'] = f'{type(e).__name__}: {e}'
    return res


def run_workspace_item(editor, item: dict, cfg: dict, out_dir: str,
                       progress=None, should_cancel=None) -> dict:
    """In-process run of one item (resolve → prepare → compute). Used by tests
    and any non-subprocess caller. The GUI instead runs items in CHILD
    PROCESSES (``prepare_run`` in the parent + ``_run_item_subprocess``) so a
    native crash / OOM / cancel can't take down the app."""
    label = run_label(item)
    try:
        prep = prepare_run(editor, item, cfg)
    except Exception as e:  # noqa: BLE001
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


def _subprocess_main() -> None:
    """Entry point for a job CHILD process (launched via ``python -m``): read a
    pickled ``{prep, cfg, out_dir}`` job from ``sys.argv[1]``, run it, stream
    marker-prefixed progress on stdout, and write the pickled result to
    ``sys.argv[2]``. A real subprocess (NOT multiprocessing) so the parent can
    kill the whole process tree to cancel without corrupting a shared queue —
    the failure mode that crashed the GUI on Cancel."""
    import pickle
    import sys
    # PhenoGraph's Louvain binaries would otherwise flash a console window per
    # call — silence the whole spawned subtree before any clustering runs.
    _suppress_child_windows()
    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path, 'rb') as fh:
        job = pickle.load(fh)
    res = compute_run(job['prep'], job['cfg'], job['out_dir'],
                      progress=lambda m: print(_PROG + m, flush=True))
    with open(result_path, 'wb') as fh:
        pickle.dump(res, fh)


def find_run_outputs(out_dir: str) -> list[dict]:
    """Group a run directory's files by item label (Tk-free, testable).

    Returns ``[{label, csv, umap, trimap, clusters}]`` — paths absolute or None,
    ``clusters`` = data-row count of the cluster CSV (or None)."""
    out: list[dict] = []
    if not out_dir or not os.path.isdir(out_dir):
        return out
    by_label: dict[str, dict] = {}
    for f in sorted(os.listdir(out_dir)):
        for suffix, key in (('_clusters.csv', 'csv'), ('_umap.png', 'umap'),
                            ('_trimap.png', 'trimap')):
            if f.endswith(suffix):
                by_label.setdefault(f[:-len(suffix)], {})[key] = os.path.join(out_dir, f)
                break
    for lbl in sorted(by_label):
        files = by_label[lbl]
        rec = {'label': lbl, 'csv': files.get('csv'), 'umap': files.get('umap'),
               'trimap': files.get('trimap'), 'clusters': None}
        if rec['csv']:
            try:
                with open(rec['csv'], encoding='utf-8') as fh:
                    rec['clusters'] = max(0, sum(1 for _ in fh) - 1)   # minus header
            except OSError:
                pass
        out.append(rec)
    return out


def _item_to_json(it: dict) -> dict:
    """Copy an item dict with its ``comp_matrix`` ndarray turned into a tagged
    nested-list so the whole thing is JSON-serialisable."""
    d = dict(it)
    m = d.get('comp_matrix')
    if m is not None and hasattr(m, 'tolist'):
        d['comp_matrix'] = {'__ndarray__': m.tolist()}
    return d


def _item_from_json(d: dict) -> dict:
    """Inverse of :func:`_item_to_json` — restore a tagged ``comp_matrix`` to an
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

    # -- creation --
    def new_group(self, name: str | None = None) -> str:
        self._gseq += 1
        gid = f'G{self._gseq}'
        self.groups[gid] = {'gid': gid, 'name': name or f'Group {self._gseq}',
                            'display': True, 'comp': None, 'fmo': None, 'items': {}}
        return gid

    def add_item(self, payload: dict, gid: str | None = None) -> str:
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
        store, _ = self._store_of(mid)
        return store[mid] if store else None

    def item_group(self, mid: str) -> str | None:
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
        store, _ = self._store_of(mid)
        if store is not None:
            store.pop(mid, None)

    def remove_group(self, gid: str, *, keep_items=True) -> None:
        g = self.groups.pop(gid, None)
        if g and keep_items:
            self.loose.update(g['items'])

    def clear(self) -> None:
        self.groups.clear()
        self.loose.clear()

    def move_item(self, mid: str, gid: str | None) -> None:
        it = self.item(mid)
        if it is None:
            return
        self.remove_item(mid)
        if gid and gid in self.groups:
            self.groups[gid]['items'][mid] = it
        else:
            self.loose[mid] = it

    def group_selected(self, mids, name: str | None = None) -> str:
        gid = self.new_group(name)
        for mid in mids:
            self.move_item(mid, gid)
        return gid

    # -- comp / fmo --
    def set_item_comp(self, mid, comp):
        it = self.item(mid)
        if it is not None:
            it['comp_override'] = comp

    def set_item_fmo(self, mid, fmo):
        it = self.item(mid)
        if it is not None:
            it['fmo'] = fmo

    def set_group_comp(self, gid, comp):
        if gid in self.groups:
            self.groups[gid]['comp'] = comp

    def set_group_fmo(self, gid, fmo):
        if gid in self.groups:
            self.groups[gid]['fmo'] = fmo

    def effective_comp(self, mid: str) -> dict:
        """Resolve comp with precedence: item override → group override →
        metadata matrix → none. Returns {'kind','label','ready'} where
        ``ready`` is False when neither a matrix nor beads are available."""
        it = self.item(mid)
        if it is None:
            return {'kind': 'none', 'label': '⚠ none', 'ready': False}
        gid = self.item_group(mid)
        ov = it.get('comp_override') or (self.groups[gid]['comp'] if gid else None)
        if ov:
            if ov.get('kind') == 'beads':
                n = len(ov.get('files', []))
                return {'kind': 'beads', 'label': f'beads:{n}', 'ready': n > 0}
            return {'kind': 'matrix', 'label': 'matrix*', 'ready': True}
        m = it.get('comp_matrix')
        if m is not None:
            shape = getattr(m, 'shape', None)
            lbl = f'{shape[0]}×{shape[1]}' if (shape and len(shape) == 2) else 'matrix'
            return {'kind': 'matrix', 'label': lbl, 'ready': True}
        return {'kind': 'none', 'label': '⚠ none', 'ready': False}

    def effective_fmo(self, mid: str):
        it = self.item(mid)
        if it is None:
            return None
        gid = self.item_group(mid)
        return it.get('fmo') or (self.groups[gid]['fmo'] if gid else None)

    def comp_ready(self, mid: str) -> bool:
        return self.effective_comp(mid)['ready']

    def unready_items(self) -> list[str]:
        """mids that have neither a matrix nor beads (block a run)."""
        return [mid for mid, _it, _gid in self.all_items() if not self.comp_ready(mid)]

    # -- display --
    def toggle_item_display(self, mid):
        it = self.item(mid)
        if it is not None:
            it['display'] = not it.get('display', True)

    def toggle_group_display(self, gid):
        if gid in self.groups:
            self.groups[gid]['display'] = not self.groups[gid].get('display', True)

    def set_all_display(self, on: bool):
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
        return {
            'title': self.title,
            'run_cfg': dict(self.run_cfg),
            '_gseq': self._gseq,
            '_mseq': self._mseq,
            'groups': [
                {'gid': g['gid'], 'name': g['name'], 'display': g.get('display', True),
                 'comp': g.get('comp'), 'fmo': g.get('fmo'),
                 'items': [{'mid': mid, **_item_to_json(it)}
                           for mid, it in g['items'].items()]}
                for g in self.groups.values()],
            'loose': [{'mid': mid, **_item_to_json(it)}
                      for mid, it in self.loose.items()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> WorkspaceModel:
        m = cls(d.get('title', 'Pipeline'))
        if isinstance(d.get('run_cfg'), dict):
            m.run_cfg.update(d['run_cfg'])
        for g in d.get('groups', []):
            m.groups[g['gid']] = {
                'gid': g['gid'], 'name': g.get('name', g['gid']),
                'display': g.get('display', True), 'comp': g.get('comp'),
                'fmo': g.get('fmo'),
                'items': {i['mid']: _item_from_json({k: v for k, v in i.items()
                                                     if k != 'mid'})
                          for i in g.get('items', [])}}
        for i in d.get('loose', []):
            m.loose[i['mid']] = _item_from_json({k: v for k, v in i.items()
                                                 if k != 'mid'})
        # Restore id counters so future adds don't collide with loaded ids.
        m._gseq = int(d.get('_gseq', len(m.groups)))
        m._mseq = int(d.get('_mseq', sum(1 for _ in m.all_items())))
        return m


# ── View (Tk renderer over a model) ────────────────────────────────────────

_COL = {'#0': 'tree', '#1': 'on', '#2': 'comp', '#3': 'fmo'}


class PipelineWorkspaceView(ttk.Frame):
    """Renders a ``WorkspaceModel`` as a 3-column tree (pops | comp | fmo) with
    a 👁 display column. Disposable: pop-out / re-dock rebuild over the model."""

    def __init__(self, parent, model: WorkspaceModel, *, editor=None,
                 on_status=None, on_change=None, on_run=None, on_cancel=None,
                 on_results=None, on_before_change=None):
        super().__init__(parent)
        self.model = model
        self._editor = editor
        self._on_status = on_status
        self._on_change = on_change
        self._on_run = on_run
        self._on_cancel = on_cancel
        self._on_results = on_results
        self._on_before_change = on_before_change
        self._build()
        self._render()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 3))
        ttk.Button(bar, text="＋ Group", width=8,
                   command=self._new_group).pack(side='left')
        ttk.Button(bar, text="Group sel", width=9,
                   command=self._group_selected).pack(side='left', padx=(3, 0))
        ttk.Button(bar, text="Set comp…", width=9,
                   command=self._set_comp_picker).pack(side='left', padx=(3, 0))
        ttk.Button(bar, text="Set FMOs…", width=9,
                   command=self._set_fmo_picker).pack(side='left', padx=(3, 0))
        ttk.Button(bar, text="Remove", width=7,
                   command=self._remove_selected).pack(side='right')
        ttk.Button(bar, text="Clear", width=6,
                   command=self._clear).pack(side='right', padx=(0, 3))

        # Run bar — Phenograph / UMAP / TriMap modifiers + Run/Cancel (M2).
        # Split into a parameters row and an actions row so Run / Cancel /
        # Results are never clipped when the panel is docked narrow.
        runbar = ttk.Frame(self)
        runbar.grid(row=1, column=0, sticky='ew', pady=(0, 3))
        cfg = self.model.run_cfg
        self._k_var = tk.StringVar(value=str(cfg.get('k', 30)))
        self._max_var = tk.StringVar(value=str(cfg.get('max_events', 5000)))
        self._seed_var = tk.StringVar(value=str(cfg.get('seed', 42)))
        self._umap_var = tk.BooleanVar(value=bool(cfg.get('umap', True)))
        self._trimap_var = tk.BooleanVar(value=bool(cfg.get('trimap', True)))
        self._concat_var = tk.BooleanVar(value=bool(cfg.get('concatenate', False)))

        params = ttk.Frame(runbar)
        params.pack(fill='x', anchor='w')
        ttk.Label(params, text='k').pack(side='left')
        ttk.Entry(params, textvariable=self._k_var, width=4).pack(side='left', padx=(1, 6))
        ttk.Label(params, text='max ev').pack(side='left')
        ttk.Entry(params, textvariable=self._max_var, width=7).pack(side='left', padx=(1, 6))
        ttk.Label(params, text='seed').pack(side='left')
        ttk.Entry(params, textvariable=self._seed_var, width=5).pack(side='left', padx=(1, 6))
        ttk.Checkbutton(params, text='UMAP', variable=self._umap_var).pack(side='left')
        ttk.Checkbutton(params, text='TriMap',
                        variable=self._trimap_var).pack(side='left', padx=(2, 2))
        ttk.Checkbutton(params, text='Concat',
                        variable=self._concat_var).pack(side='left', padx=(0, 6))

        actions = ttk.Frame(runbar)
        actions.pack(fill='x', anchor='w', pady=(3, 0))
        self.btn_run = ttk.Button(actions, text='▶ Run', width=8, command=self._do_run)
        self.btn_run.pack(side='left')
        self.btn_cancel = ttk.Button(actions, text='Cancel', width=8,
                                     command=self._do_cancel, state='disabled')
        self.btn_cancel.pack(side='left', padx=(3, 0))
        ttk.Button(actions, text='Results…', width=9,
                   command=self._do_results).pack(side='left', padx=(3, 0))

        treef = ttk.Frame(self)
        treef.grid(row=2, column=0, sticky='nsew')
        treef.columnconfigure(0, weight=1)
        treef.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(treef, columns=('on', 'comp', 'fmo'),
                                 show='tree headings', selectmode='extended')
        self.tree.heading('#0', text='Samples / populations', anchor='w')
        self.tree.heading('on', text='👁', anchor='center')
        self.tree.heading('comp', text='Comp / beads', anchor='w')
        self.tree.heading('fmo', text='FMOs', anchor='w')
        self.tree.column('#0', anchor='w', stretch=True, width=240)
        self.tree.column('on', width=30, anchor='center', stretch=False)
        self.tree.column('comp', width=90, anchor='w', stretch=False)
        self.tree.column('fmo', width=70, anchor='w', stretch=False)
        self.tree.grid(row=0, column=0, sticky='nsew')
        sb = ttk.Scrollbar(treef, orient='vertical', command=self.tree.yview)
        sb.grid(row=0, column=1, sticky='ns')
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.tag_configure('grp', font=('TkDefaultFont', 9, 'bold'))
        self.tree.tag_configure('off', foreground='grey')
        self.tree.tag_configure('warn', foreground='#c0392b')
        self.tree.tag_configure('drop_target', background='#fff2a8')
        # Press / motion / release: a bare click toggles the 👁 column (old
        # _on_click); a drag past the threshold MOVES the dragged item(s) to the
        # group under the pointer (or to loose/top-level if dropped on empty
        # space) — fixes items accidentally dropped outside a group. A drag that
        # ends over an open Statistics window hands the item's sample to it.
        self.tree.bind('<Button-1>', self._on_press)
        self.tree.bind('<B1-Motion>', self._on_motion)
        self.tree.bind('<ButtonRelease-1>', self._on_release)
        self.tree.bind('<Delete>', lambda e: self._remove_selected())
        self._press_row = None
        self._press_col = None
        self._press_x = 0
        self._press_y = 0
        self._drag_active = False
        self._drag_threshold = 5
        self._drop_hi = None

    # -- drop target geometry --
    def contains(self, widget) -> bool:
        w = widget
        while w is not None:
            if w is self.tree:
                return True
            w = getattr(w, 'master', None)
        return False

    @staticmethod
    def _row_target(row):
        if not row:
            return None
        if row.startswith('G'):
            return ('group', row)
        if row.startswith('M'):
            return ('item', row)
        return None

    def drop_at(self, editor, nodes, x_root, y_root) -> bool:
        """Route an editor drop by the column + row under the pointer:
        comp column → set beads on that row; fmo column → set FMOs; else add
        the nodes as items (into the group row if dropped on one)."""
        try:
            x = x_root - self.tree.winfo_rootx()
            y = y_root - self.tree.winfo_rooty()
            col = self.tree.identify_column(x)
            row = self.tree.identify_row(y)
        except Exception:
            col, row = '#0', ''
        return self._route_drop(editor, nodes, _COL.get(col, 'tree'),
                                self._row_target(row))

    def _route_drop(self, editor, nodes, colname, target) -> bool:
        if colname == 'comp':
            return self._assign_comp(editor, nodes, target)
        if colname == 'fmo':
            return self._assign_fmo(editor, nodes, target)
        gid = target[1] if (target and target[0] == 'group') else None
        self._pre_change()
        added = 0
        for p in nodes:
            ctx = extract_one_context(editor, p[1])
            if ctx is None:
                continue
            leaf = p[2] if p[0] == 'gate' else None
            self.model.add_item(build_drop_payload(ctx, leaf), gid=gid)
            added += 1
        if added:
            self._render()
            self._changed()
            self._status(f"Added {added} item(s)"
                         + (f" to '{self.model.groups[gid]['name']}'." if gid else "."))
        return added > 0

    def _node_paths(self, editor, nodes):
        paths = []
        for p in nodes:
            ctx = extract_one_context(editor, p[1])
            if ctx and ctx.get('path'):
                paths.append(ctx['path'])
        return paths

    def _assign_comp(self, editor, nodes, target) -> bool:
        paths = self._node_paths(editor, nodes)
        if not paths or target is None:
            return False
        self._pre_change()
        comp = {'kind': 'beads', 'files': paths}
        if target[0] == 'group':
            self.model.set_group_comp(target[1], comp)
        else:
            self.model.set_item_comp(target[1], comp)
        self._render()
        self._changed()
        self._status(f"Set comp beads ({len(paths)} file(s)) — overrides matrix.")
        return True

    def _assign_fmo(self, editor, nodes, target) -> bool:
        paths = self._node_paths(editor, nodes)
        if not paths or target is None:
            return False
        self._pre_change()
        fmo = {'files': paths}
        if target[0] == 'group':
            self.model.set_group_fmo(target[1], fmo)
        else:
            self.model.set_item_fmo(target[1], fmo)
        self._render()
        self._changed()
        self._status(f"Set {len(paths)} FMO(s).")
        return True

    # -- rendering --
    def _render(self):
        tv = self.tree
        for iid in tv.get_children(''):
            tv.delete(iid)
        for gid, g in self.model.groups.items():
            on = g.get('display', True)
            tv.insert('', 'end', iid=gid,
                      text=f'▦ {g["name"]}  ({len(g["items"])})',
                      values=('☑' if on else '☐',
                              self._group_comp_label(g), self._group_fmo_label(g)),
                      open=True, tags=('grp',) if on else ('grp', 'off'))
            for mid, it in g['items'].items():
                self._insert_item(gid, mid, it, group_on=on)
        for mid, it in self.model.loose.items():
            self._insert_item('', mid, it, group_on=True)

    def _insert_item(self, parent, mid, it, group_on=True):
        disp = it.get('display', True)
        comp = self.model.effective_comp(mid)
        fmo = self.model.effective_fmo(mid)
        fmo_lbl = f"{len(fmo.get('files', []))} FMO" if fmo else '—'
        who = f"{it.get('trial', 'Trial')} / {it['sample']}"
        label = f"{who}  ›  {it['gate_path']}" if it.get('gate_id') else who
        if not comp['ready']:
            tag = 'warn'
        elif group_on and disp:
            tag = self._color_tag(mid, it.get('color'))
        else:
            tag = 'off'
        self.tree.insert(parent, 'end', iid=mid, text=f'■ {label}',
                         values=('☑' if disp else '☐', comp['label'], fmo_lbl),
                         tags=(tag,))

    def _group_comp_label(self, g):
        c = g.get('comp')
        if not c:
            return '—'
        return (f"beads:{len(c.get('files', []))} (grp)"
                if c.get('kind') == 'beads' else 'matrix (grp)')

    def _group_fmo_label(self, g):
        f = g.get('fmo')
        return f"{len(f.get('files', []))} FMO (grp)" if f else '—'

    def _color_tag(self, key, color):
        tag = f'col_{key}'
        self.tree.tag_configure(tag, foreground=color or '#000000',
                                font=('TkDefaultFont', 9, 'bold'))
        return tag

    # -- interactions --
    def _on_press(self, event):
        """Record the press so release can tell a click from a drag. Doesn't
        consume the event, so Treeview's own selection still happens."""
        self._press_row = self.tree.identify_row(event.y)
        self._press_col = self.tree.identify_column(event.x)
        self._press_x, self._press_y = event.x, event.y
        self._drag_active = False

    def _on_motion(self, event):
        """Past the threshold, an item-row press becomes a move-drag: the
        cursor changes and the group row under the pointer is highlighted."""
        if self._press_row is None or self._press_col == '#1':
            return
        if not self._drag_active:
            if (abs(event.x - self._press_x) <= self._drag_threshold
                    and abs(event.y - self._press_y) <= self._drag_threshold):
                return
            if not (self._press_row and self._press_row.startswith('M')):
                return   # only items are draggable; groups stay put
            self._drag_active = True
            try:
                self.tree.config(cursor='fleur')
            except Exception:
                pass
        row = self.tree.identify_row(event.y)
        self._set_drop_hi(row if (row and row.startswith('G')) else None)

    def _on_release(self, event):
        """Click → toggle the 👁 column (old _on_click). Drag → move the
        dragged item(s) to the destination group (or loose), unless the drop
        landed on an open Statistics window (then add the sample there)."""
        drag = self._drag_active
        press_row = self._press_row
        press_col = self._press_col
        try:
            if not drag:
                if press_col == '#1':
                    self._toggle_display_at(press_row)
                return
            if self._drop_to_stats(event):
                return
            self._move_drag_to(press_row, self.tree.identify_row(event.y))
        finally:
            self._clear_drag()

    def _toggle_display_at(self, row):
        target = self._row_target(row)
        if target is None:
            return
        self._pre_change()
        if target[0] == 'group':
            self.model.toggle_group_display(target[1])
        else:
            self.model.toggle_item_display(target[1])
        self._render()
        self._changed()

    def _move_drag_to(self, mid, drop_row):
        """Move the dragged item(s) to the group implied by the row under the
        pointer: a group row → that group; an item row → that item's group;
        empty space → loose (top level)."""
        if not (mid and mid.startswith('M')):
            return
        # Carry the whole item selection if the dragged row is part of it.
        sel = [t[1] for t in self._selected_targets() if t[0] == 'item']
        mids = sel if (mid in sel and len(sel) > 1) else [mid]
        dest_gid = None
        if drop_row:
            if drop_row.startswith('G'):
                dest_gid = drop_row
            elif drop_row.startswith('M'):
                dest_gid = self.model.item_group(drop_row)
        # No-op if every dragged item already lives in the destination.
        if all(self.model.item_group(m) == dest_gid for m in mids):
            return
        self._pre_change()
        for m in mids:
            self.model.move_item(m, dest_gid)
        self._render()
        self._changed()
        where = (f"'{self.model.groups[dest_gid]['name']}'"
                 if dest_gid else 'loose items')
        self._status(f"Moved {len(mids)} item(s) to {where}.")

    def _drop_to_stats(self, event):
        """If a drag ends over an open Statistics window, add the dragged
        item(s)' gated population(s) to it. Statistics is population-based, so
        only items that carry a gate qualify — a bare-sample item is consumed
        with an explanatory status. Returns True iff over a stats window.
        Defensive — never crashes the drag."""
        ed = self._editor
        finder = getattr(ed, '_stats_window_under', None)
        if finder is None:
            return False
        try:
            win = finder(event.x_root, event.y_root)
        except Exception:
            win = None
        if win is None:
            return False
        sel = [t[1] for t in self._selected_targets() if t[0] == 'item']
        mids = (sel if (self._press_row in sel and len(sel) > 1)
                else [self._press_row])
        targets = []
        for m in mids:
            it = self.model.item(m) if m else None
            if not it:
                continue
            nm, gid = it.get('sample'), it.get('gate_id')
            if nm and gid and (nm, gid) not in targets:
                targets.append((nm, gid))
        if not targets:
            self._status("Statistics accepts gated populations only — "
                         "this item has no gate.")
            return True
        try:
            win.add_targets(targets, 'workspace')
            self._status(f"Added {len(targets)} population(s) to statistics.")
        except Exception:
            pass
        return True

    def _set_drop_hi(self, iid):
        if self._drop_hi == iid:
            return
        if self._drop_hi is not None:
            self._strip_drop_tag(self._drop_hi)
        self._drop_hi = iid
        if iid is not None:
            try:
                tags = [t for t in self.tree.item(iid, 'tags') if t != 'drop_target']
                self.tree.item(iid, tags=tags + ['drop_target'])
            except Exception:
                pass

    def _strip_drop_tag(self, iid):
        try:
            tags = [t for t in self.tree.item(iid, 'tags') if t != 'drop_target']
            self.tree.item(iid, tags=tags)
        except Exception:
            pass

    def _clear_drag(self):
        if self._drop_hi is not None:
            self._strip_drop_tag(self._drop_hi)
        self._drop_hi = None
        self._press_row = None
        self._press_col = None
        self._drag_active = False
        try:
            self.tree.config(cursor='')
        except Exception:
            pass

    def _selected_targets(self):
        return [t for t in (self._row_target(r) for r in self.tree.selection()) if t]

    def _new_group(self):
        self._pre_change()
        gid = self.model.new_group()
        self._render()
        self._changed()
        self._status(f"New group '{self.model.groups[gid]['name']}'.")

    def _group_selected(self):
        mids = [t[1] for t in self._selected_targets() if t[0] == 'item']
        if not mids:
            self._status("Select item rows to group.")
            return
        self._pre_change()
        gid = self.model.group_selected(mids)
        self._render()
        self._changed()
        self._status(f"Grouped {len(mids)} item(s) into '{self.model.groups[gid]['name']}'.")

    def _set_comp_picker(self):
        files = self._pick_files("Select compensation bead FCS files")
        if not files:
            return
        comp = {'kind': 'beads', 'files': files}
        self._apply_to_selection(lambda gid: self.model.set_group_comp(gid, comp),
                                 lambda mid: self.model.set_item_comp(mid, comp),
                                 f"comp beads ({len(files)} file(s))")

    def _set_fmo_picker(self):
        files = self._pick_files("Select FMO control FCS files")
        if not files:
            return
        fmo = {'files': files}
        self._apply_to_selection(lambda gid: self.model.set_group_fmo(gid, fmo),
                                 lambda mid: self.model.set_item_fmo(mid, fmo),
                                 f"{len(files)} FMO(s)")

    def _pick_files(self, title):
        from tkinter import filedialog
        return list(filedialog.askopenfilenames(
            parent=self, title=title,
            filetypes=[('FCS files', '*.fcs'), ('All files', '*.*')]))

    def _apply_to_selection(self, on_group, on_item, what):
        targets = self._selected_targets()
        if not targets:
            self._status(f"Select a group or item to set {what}.")
            return
        self._pre_change()
        for t in targets:
            (on_group if t[0] == 'group' else on_item)(t[1])
        self._render()
        self._changed()
        self._status(f"Set {what} on {len(targets)} row(s).")

    def _remove_selected(self):
        targets = self._selected_targets()
        if not targets:
            return
        self._pre_change()
        for t in targets:
            if t[0] == 'group':
                self.model.remove_group(t[1], keep_items=True)
            else:
                self.model.remove_item(t[1])
        self._render()
        self._changed()
        self._status("Removed.")

    def _clear(self):
        if not self.model.groups and not self.model.loose:
            return
        self._pre_change()
        self.model.clear()
        self._render()
        self._changed()
        self._status("Workspace cleared.")

    def _status(self, msg):
        if self._on_status:
            self._on_status(msg)

    def _changed(self):
        if self._on_change:
            self._on_change(self)

    def _pre_change(self):
        """Fire the editor's undo-checkpoint hook before a model mutation."""
        if self._on_before_change:
            try:
                self._on_before_change()
            except Exception:
                pass

    # -- run (M2) --
    def _sync_run_cfg(self):
        """Push run-bar widgets into the model so settings persist across
        pop-out / re-dock. Unparseable ints keep the current model value."""
        cfg = self.model.run_cfg

        def _int(var, key, lo):
            try:
                cfg[key] = max(lo, int(float(var.get())))
            except (ValueError, tk.TclError):
                pass

        _int(self._k_var, 'k', 2)
        try:                       # max events: 0 / blank = all events
            cfg['max_events'] = max(0, int(float(self._max_var.get())))
        except (ValueError, tk.TclError):
            pass
        _int(self._seed_var, 'seed', 0)
        cfg['umap'] = bool(self._umap_var.get())
        cfg['trimap'] = bool(self._trimap_var.get())
        cfg['concatenate'] = bool(self._concat_var.get())
        return cfg

    def _do_run(self):
        self._sync_run_cfg()
        if self._on_run:
            self._on_run()

    def _do_cancel(self):
        if self._on_cancel:
            self._on_cancel()

    def _do_results(self):
        if self._on_results:
            self._on_results()

    def set_running(self, running: bool):
        for btn, on in ((getattr(self, 'btn_run', None), not running),
                        (getattr(self, 'btn_cancel', None), running)):
            if btn is not None:
                try:
                    btn.configure(state='normal' if on else 'disabled')
                except tk.TclError:
                    pass


# ── Results viewer (read-only image window) ────────────────────────────────


class ResultsViewer(tk.Toplevel):
    """Read-only window showing a run's UMAP / TriMap images + cluster counts.
    Tk ≥8.6 renders PNG via PhotoImage natively (no Pillow needed)."""

    def __init__(self, parent, out_dir: str, *, subsample: int = 2):
        super().__init__(parent)
        self.title(f"Run results — {os.path.basename(out_dir or 'none')}")
        self.geometry('900x720')
        self._imgs: list = []                       # keep PhotoImage refs alive
        recs = find_run_outputs(out_dir)

        top = ttk.Frame(self, padding=4)
        top.pack(fill='x')
        ttk.Label(top, text=f"{len(recs)} item(s)  ·  {out_dir}").pack(side='left')
        ttk.Button(top, text="Open folder",
                   command=lambda: self._open_folder(out_dir)).pack(side='right')

        canvas = tk.Canvas(self, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind('<Configure>',
                  lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=body, anchor='nw')
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        if not recs:
            ttk.Label(body, text="No results in this folder yet.",
                      foreground='#777').pack(padx=12, pady=12)
        for rec in recs:
            self._add_record(body, rec, subsample)

    def _add_record(self, parent, rec, subsample):
        head = rec['label']
        if rec['clusters'] is not None:
            head += f"   ·   {rec['clusters']} clusters"
        lf = ttk.LabelFrame(parent, text=head)
        lf.pack(fill='x', padx=6, pady=4)
        row = ttk.Frame(lf)
        row.pack(fill='x')
        for key, title in (('umap', 'UMAP'), ('trimap', 'TriMap')):
            cell = ttk.Frame(row)
            cell.pack(side='left', padx=6, pady=4)
            ttk.Label(cell, text=title).pack()
            img = self._load_image(rec.get(key), subsample)
            if img is not None:
                ttk.Label(cell, image=img).pack()
            else:
                ttk.Label(cell, text='(not generated)', foreground='#999').pack()

    def _load_image(self, path, subsample):
        if not path or not os.path.exists(path):
            return None
        try:
            img = tk.PhotoImage(file=path)
            if subsample and subsample > 1:
                img = img.subsample(subsample, subsample)
            self._imgs.append(img)
            return img
        except Exception:
            return None

    @staticmethod
    def _open_folder(path):
        if not path or not os.path.isdir(path):
            return
        try:
            import subprocess
            import sys
            if sys.platform == 'win32':
                os.startfile(os.path.abspath(path))      # type: ignore[attr-defined]  # noqa: S606
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', path])
            else:
                subprocess.Popen(['xdg-open', path])
        except Exception:
            pass


# ── Panel (hosts the single workspace) ─────────────────────────────────────


class WorkspacePanel(ttk.Frame):
    """Hosts the single Pipeline Workspace (one model + view) with Pop-out and
    Import-all. Drag routing delegates to whichever view is live."""

    def __init__(self, parent, editor=None, on_before_change=None):
        super().__init__(parent)
        self._editor = editor
        # Called BEFORE any workspace mutation so the editor can push an undo
        # checkpoint — this is what lets the editor's existing Undo button also
        # revert workspace changes.
        self._on_before_change = on_before_change
        self.model = WorkspaceModel('Pipeline')
        self._view: PipelineWorkspaceView | None = None
        self._pop: tuple | None = None
        self._placeholder: ttk.Label | None = None
        # Process-based job runner: a queue of items, one child process at a
        # time; cancel terminates it, a crash/OOM requeues it once.
        self._cur: dict | None = None
        self._jobs: collections.deque = collections.deque()
        self._cancel_requested = False
        self._total = 0
        self._idx = 0
        self._ok = 0
        self._err = 0
        self._out_dir: str | None = None
        self.last_run_dir: str | None = None
        self.status_var = tk.StringVar(
            value="Drag samples / leaves / whole trials in; drop beads on Comp, FMOs on FMO.")
        self._build()
        self._embed_view()

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        bar = ttk.Frame(self, padding=(2, 2))
        bar.grid(row=0, column=0, sticky='ew')
        ttk.Button(bar, text="Pop out", width=8,
                   command=self._toggle_popout).pack(side='left')
        ttk.Button(bar, text="Import all", width=9,
                   command=self._import_all).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text="Load…", width=7,
                   command=self._load).pack(side='right')
        ttk.Button(bar, text="Save…", width=7,
                   command=self._save).pack(side='right', padx=(0, 4))
        self._host = ttk.Frame(self)
        self._host.grid(row=1, column=0, sticky='nsew')
        self._host.columnconfigure(0, weight=1)
        self._host.rowconfigure(0, weight=1)
        status = ttk.Frame(self, padding=(4, 2))
        status.grid(row=2, column=0, sticky='ew')
        ttk.Label(status, textvariable=self.status_var, foreground='#444').pack(side='left')
        self.bind('<Destroy>', self._on_destroy)

    def _on_destroy(self, event):
        # Terminate a running (non-daemon) job so it can't keep the app alive.
        if event.widget is self:
            self._cancel_requested = True
            self._kill_current()
            self._jobs.clear()

    def _embed_view(self):
        self._view = PipelineWorkspaceView(self._host, self.model, editor=self._editor,
                                           on_status=self.status_var.set,
                                           on_run=self._run, on_cancel=self._cancel,
                                           on_results=self._open_results,
                                           on_before_change=self._before_change)
        self._view.grid(row=0, column=0, sticky='nsew')

    def _before_change(self):
        """Push an editor undo checkpoint before a workspace mutation, so the
        editor's existing Undo button also reverts workspace changes."""
        if callable(self._on_before_change):
            try:
                self._on_before_change()
            except Exception:
                pass

    def _reembed(self):
        """Tear down any live / popped view and rebuild it over the current
        model (used by Load and by undo restore)."""
        if self._pop is not None:
            self._redock()
        if self._view is not None:
            self._view.destroy()
            self._view = None
        self._embed_view()

    def restore_model(self, snap: dict):
        """Replace the model from a snapshot dict (called by the editor's undo
        when reverting a step that included workspace changes)."""
        self.model = WorkspaceModel.from_dict(snap)
        self._reembed()

    # -- pop-out / re-dock --
    def _toggle_popout(self):
        if self._pop is not None:
            self._redock()
            return
        if self._view is not None:
            self._view.destroy()
            self._view = None
        self._placeholder = ttk.Label(
            self._host, foreground='#777',
            text="(popped out — close that window to re-dock)")
        self._placeholder.grid(row=0, column=0)
        top = tk.Toplevel(self)
        top.title(f"Workspace — {self.model.title}")
        top.geometry('560x620')
        pv = PipelineWorkspaceView(top, self.model, editor=self._editor,
                                   on_status=self.status_var.set,
                                   on_run=self._run, on_cancel=self._cancel,
                                   on_results=self._open_results,
                                   on_before_change=self._before_change)
        pv.pack(fill='both', expand=True)
        self._pop = (top, pv)
        top.protocol('WM_DELETE_WINDOW', self._redock)
        self.status_var.set("Popped out. Close the window to re-dock.")

    def _redock(self):
        if self._pop is None:
            return
        top, _pv = self._pop
        self._pop = None
        try:
            top.destroy()
        except Exception:
            pass
        if self._placeholder is not None:
            try:
                self._placeholder.destroy()
            except Exception:
                pass
            self._placeholder = None
        self._embed_view()
        self.status_var.set("Re-docked.")

    def popped_count(self) -> int:
        return 1 if self._pop is not None else 0

    def _import_all(self):
        view = self._active_view()
        if view is None or self._editor is None:
            return
        self._before_change()
        n = 0
        for ctx in extract_editor_context(self._editor):
            view.model.add_item(build_drop_payload(ctx, None))
            n += 1
        view._render()
        self.status_var.set(f"Imported {n} sample(s)." if n else "Editor has no loaded samples.")

    # -- drop routing (from the editor) --
    def _live_views(self):
        vs = []
        if self._view is not None:
            vs.append(self._view)
        if self._pop is not None:
            vs.append(self._pop[1])
        return vs

    def _active_view(self):
        vs = self._live_views()
        return vs[0] if vs else None

    def is_drop_target(self, widget) -> bool:
        return any(v.contains(widget) for v in self._live_views())

    def drop_at(self, editor, nodes, x_root, y_root) -> bool:
        """Find the workspace view under the point and route the drop there
        (column-aware). Falls back to the active view (e.g. headless)."""
        try:
            widget = self.winfo_containing(x_root, y_root)
        except Exception:
            widget = None
        for v in self._live_views():
            if widget is not None and v.contains(widget):
                return v.drop_at(editor, nodes, x_root, y_root)
        v = self._active_view()
        return bool(v and v.drop_at(editor, nodes, x_root, y_root))

    # -- run (M2): Phenograph + UMAP + TriMap, off the Tk thread --------------
    def _set_running(self, running: bool):
        for v in self._live_views():
            v.set_running(running)

    def _running(self) -> bool:
        return self._cur is not None or bool(self._jobs)

    def _run(self):
        if self._running():
            self.status_var.set("A run is already in progress.")
            return
        if self._editor is None:
            self.status_var.set("No editor attached — cannot resolve samples.")
            return
        base_cfg = dict(self.model.run_cfg)
        units = build_run_units(self.model, base_cfg)
        if not units:
            self.status_var.set("Nothing to run — add items to the workspace first.")
            return
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._out_dir = os.path.abspath(os.path.join('outputs', 'workspace', f'run_{ts}'))
        self.last_run_dir = self._out_dir
        # One job per RUN UNIT: a group's samples co-embed into one UMAP; loose
        # items are their own unit; 'concatenate' merges everything into one.
        # Each carries its own cfg copy so a retry can lower max_events alone.
        self._jobs = collections.deque(
            {'unit': u, 'label': u['label'], 'attempt': 1, 'cfg': dict(base_cfg)}
            for u in units)
        self._total = len(self._jobs)
        self._idx = self._ok = self._err = 0
        self._cancel_requested = False
        self._cur = None
        self._set_running(True)
        mode = 'concatenated' if base_cfg.get('concatenate') else 'per-group'
        self.status_var.set(f"Queued {self._total} run(s) [{mode}] → {self._out_dir} …")
        self.after(50, self._pump_jobs)

    def _pump_jobs(self):
        """Main-thread driver: one child process at a time, cancel terminates,
        crash/OOM requeues once with fewer events."""
        if self._cancel_requested:
            self._kill_current()
            self._jobs.clear()
            self._finish(f"Cancelled. → {self._out_dir}")
            return
        if self._cur is not None and self._service_current():
            self.after(150, self._pump_jobs)        # still running
            return
        if self._cur is None and self._jobs:
            self._launch_next()
        if self._cur is None and not self._jobs:
            tail = f"{self._ok} ok" + (f", {self._err} failed" if self._err else "")
            self._finish(f"Done: {tail} → {self._out_dir}")
            return
        self.after(150, self._pump_jobs)

    def _launch_next(self):
        import pickle
        import subprocess
        import sys
        import tempfile
        job = self._jobs.popleft()
        self._idx += 1
        n = self._idx
        try:
            prep = prepare_unit(self._editor, job['unit']['members'],
                                job['unit']['label'], job['cfg'])      # cheap (main)
        except Exception as e:  # noqa: BLE001
            self._err += 1
            self.status_var.set(f"[{n}/{self._total}] {job['label']}: ✗ {type(e).__name__}: {e}")
            return
        jobdir = tempfile.mkdtemp(prefix='wsjob_')
        job_path = os.path.join(jobdir, 'job.pkl')
        result_path = os.path.join(jobdir, 'result.pkl')
        try:
            with open(job_path, 'wb') as fh:
                pickle.dump({'prep': prep, 'cfg': job['cfg'], 'out_dir': self._out_dir}, fh)
            # Launch a REAL subprocess running the guarded module entry point.
            # `-m openflo.workspace` (not `-c`) so Phenograph's own pool can
            # re-import the module cleanly on Windows spawn. stderr→stdout; our
            # progress lines are marker-prefixed (_PROG), the result comes back
            # via result.pkl. Cancel = taskkill the whole tree (see _kill_current).
            # CREATE_NO_WINDOW: when the GUI runs under pythonw (no console), the
            # child python.exe would otherwise allocate and flash its own console
            # window on every run — suppress it (and Phenograph's spawned pool,
            # which inherits the no-console state). No-op off Windows.
            proc = subprocess.Popen(
                [sys.executable, '-m', 'openflo.workspace', job_path, result_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception as e:  # noqa: BLE001
            self._err += 1
            self.status_var.set(f"[{n}/{self._total}] {job['label']}: ✗ launch failed: {e}")
            return
        msgq: queue.Queue = queue.Queue()
        reader = threading.Thread(target=self._read_stdout, args=(proc, msgq), daemon=True)
        reader.start()
        self._cur = {'proc': proc, 'msgq': msgq, 'job': job, 'n': n,
                     'result_path': result_path, 'jobdir': jobdir, 'lasterr': ''}
        self.status_var.set(f"[{n}/{self._total}] {job['label']}: running… (separate process)")

    @staticmethod
    def _read_stdout(proc, msgq):
        """Reader thread: push the child's stdout lines onto a thread-safe queue.
        Touches NO Tk — the main thread drains the queue in _service_current."""
        try:
            for line in proc.stdout:
                msgq.put(line.rstrip('\r\n'))
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _absorb_stdout(self, cur: dict):
        try:
            while True:
                line = cur['msgq'].get_nowait()
                if line.startswith(_PROG):
                    self.status_var.set(f"[{cur['n']}/{self._total}] {line[len(_PROG):]}")
                elif line.strip():
                    cur['lasterr'] = line.strip()       # keep last line for diagnostics
        except queue.Empty:
            pass

    def _service_current(self) -> bool:
        """Drain the child's stdout; return True while it's still running."""
        cur = self._cur
        if cur is None:
            return False
        self._absorb_stdout(cur)
        if cur['proc'].poll() is None:
            return True
        self._finalize_current()
        return False

    def _finalize_current(self):
        import pickle
        import shutil
        cur = self._cur
        if cur is None:
            return
        self._cur = None
        self._absorb_stdout(cur)                        # final lines
        job, proc = cur['job'], cur['proc']
        res = None
        try:
            if os.path.exists(cur['result_path']):
                with open(cur['result_path'], 'rb') as fh:
                    res = pickle.load(fh)
        except Exception:
            res = None
        shutil.rmtree(cur['jobdir'], ignore_errors=True)
        if res is None:
            code = proc.poll()
            detail = f" — {cur['lasterr']}" if cur.get('lasterr') else ''
            # Retry ONLY a real crash/kill (nonzero exit), once, at fewer events.
            # A clean exit (0) with no result is a logic error — record it rather
            # than re-run, so a unit can never be launched a second time spuriously.
            if code not in (0, None) and job['attempt'] < 2:
                job['attempt'] += 1
                old = int(job['cfg'].get('max_events') or 0)
                job['cfg']['max_events'] = (max(100, old // 2) if old else 2000)
                self._jobs.append(job)
                self.status_var.set(
                    f"[{cur['n']}/{self._total}] {job['label']}: crashed (exit {code}){detail} "
                    f"— retrying once at max_events={job['cfg']['max_events']}")
            else:
                self._err += 1
                self.status_var.set(
                    f"[{cur['n']}/{self._total}] {job['label']}: ✗ no result (exit {code}){detail}")
            return
        if res.get('ok'):
            self._ok += 1
            parts = [f"{res['n_clusters']} clusters", f"{res['n_events']:,} ev"]
            if res.get('umap'):
                parts.append('UMAP')
            if res.get('trimap'):
                parts.append('TriMap')
            self.status_var.set(f"[{cur['n']}/{self._total}] {job['label']}: ✓ " + ", ".join(parts))
        else:
            self._err += 1
            self.status_var.set(f"[{cur['n']}/{self._total}] {job['label']}: ✗ {res.get('error')}")

    def _kill_current(self):
        import shutil
        import subprocess
        import sys
        cur, self._cur = self._cur, None
        if cur is None:
            return
        proc = cur['proc']
        try:
            if proc.poll() is None:
                if sys.platform == 'win32':
                    # Kill the whole tree: the job process AND Phenograph's pool.
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                                   capture_output=True, timeout=10,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        shutil.rmtree(cur['jobdir'], ignore_errors=True)

    def _finish(self, msg):
        self._cur = None
        self._jobs.clear()
        self._set_running(False)
        self.status_var.set(msg)

    def _cancel(self):
        if self._running():
            self._cancel_requested = True
            self.status_var.set("Cancelling… (terminating the running job).")

    def _open_results(self):
        d = self.last_run_dir
        if not d or not os.path.isdir(d):
            self.status_var.set("No results yet — run the workspace first.")
            return
        try:
            ResultsViewer(self, d)
        except Exception as e:  # noqa: BLE001 - viewer must never crash the app
            self.status_var.set(f"Could not open results: {type(e).__name__}: {e}")

    # -- persistence (M6) --
    def _save(self):
        import json
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            parent=self, title="Save workspace", defaultextension='.json',
            filetypes=[('Workspace JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(self.model.to_dict(), fh, indent=2)
            n = sum(1 for _ in self.model.all_items())
            self.status_var.set(f"Saved {n} item(s) to {os.path.basename(path)}.")
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"Save failed: {type(e).__name__}: {e}")

    def _load(self):
        import json
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            parent=self, title="Load workspace",
            filetypes=[('Workspace JSON', '*.json'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                loaded = WorkspaceModel.from_dict(json.load(fh))
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"Load failed: {type(e).__name__}: {e}")
            return
        self._before_change()              # loading is undoable
        self.model = loaded
        self._reembed()
        n = sum(1 for _ in self.model.all_items())
        self.status_var.set(f"Loaded {n} item(s) from {os.path.basename(path)}.")


# ── Subprocess job entry point ─────────────────────────────────────────────
# Run by the panel as:  python -m openflo.workspace <job.pkl> <result.pkl>
# A guarded __main__ (NOT `python -c`) so Phenograph's own worker pool can
# re-import this module cleanly under Windows `spawn`.
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    _subprocess_main()
