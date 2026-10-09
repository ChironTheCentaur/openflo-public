"""Gating computations lifted out of the GUI — Tk-free, headless-testable.

Step of decomposing ``gui.py``: the selected-gate event-count / %-of-parent
readout is correctness-critical (it's the headline number in cytometry), so it
belongs where it can be unit-tested directly against a DataFrame rather than
only through a constructed editor window.

The cumulative-mask maths stays in :mod:`openflo.pipeline`; this just composes
it into the counts and the human-readable summary.
"""
from __future__ import annotations

from collections.abc import Mapping


def gate_counts(sample_gates: Mapping[str, Mapping], gid: str, df,
                overrides=None) -> tuple[int, int, str]:
    """``(n_gate, n_parent, of_label)`` for a gate:

    - ``n_gate``   — events inside the gate's cumulative mask (gate + ancestors)
    - ``n_parent`` — its parent population's cumulative count, or the total
      event count for a root gate
    - ``of_label`` — ``'parent'`` for a child gate, ``'all'`` for a root gate
    """
    import numpy as np

    from .pipeline import cumulative_gate_mask

    def _n(target):
        return int(np.asarray(
            cumulative_gate_mask(sample_gates, target, df, overrides=overrides),
            dtype=bool).sum())

    n_gate = _n(gid)
    pid = sample_gates[gid].get('parent_id') if gid in sample_gates else None
    if pid and pid in sample_gates:
        return n_gate, _n(pid), 'parent'
    return n_gate, len(df), 'all'


def format_gate_count(name: str, n_gate: int, n_parent: int,
                      of_label: str) -> str:
    """Status-bar string: ``"CD3+:  n = 12,345   (45.20% of parent)"``."""
    pct = (100.0 * n_gate / n_parent) if n_parent else 0.0
    return f"{name}:  n = {n_gate:,}   ({pct:.2f}% of {of_label})"


def gate_channels(gate: Mapping) -> set:
    """The set of FCS channel names a gate dict references (``channel`` for a
    1-D gate, ``x_channel`` / ``y_channel`` for a 2-D one)."""
    chs = set()
    for k in ('channel', 'x_channel', 'y_channel'):
        v = gate.get(k)
        if v:
            chs.add(v)
    return chs


def population_path(gates: Mapping[str, Mapping], gid: str,
                    transforms: Mapping[str, Mapping] | None = None) -> str:
    """Human-readable population path, e.g. ``'Cells/Singlets/CD901b+'``, built
    by walking ``parent_id`` to the root. Cycle-safe.

    An unnamed gate is named by its description, which carries its bounds.
    ``transforms`` (the sample's ``data_transforms``) has those said as
    intensities, as the gate list says them ('T  CD3-A >= 987', not the
    stored logicle '>= 0.453'); give it wherever the path is SHOWN. Without
    it the bounds are stored coordinates, as before. Never an identity: a
    population is matched across samples by ``population_keys``."""
    from .pipeline import describe_gate
    names, seen, cur = [], set(), gid
    while cur and cur in gates and cur not in seen:
        seen.add(cur)
        g = gates[cur]
        names.append(g.get('label') or g.get('name')
                     or describe_gate(g, transforms=transforms) or cur)
        cur = g.get('parent_id')
    return '/'.join(reversed(names)) if names else str(gid)


# ── Population identity across samples ───────────────────────────────────────
#
# Frequencies, Compare-all and differential abundance line populations up
# across samples by their path. An unnamed gate's path component is its
# description, which carries its coordinates ('T  CD3-A >= 0.453'), so a gate
# nudged in one sample, or a marker read on another detector in a second
# panel, split one population into several, each "missing" from every sample
# but one -- and missing was then counted as 0 events. The key below names a
# gate by what it IS rather than where it was drawn: its name when it has
# one, else its kind and the markers (antibody labels, not detectors) it is
# drawn on, with an ordinal among same-identity siblings so two bands on one
# channel stay two populations.

def _identity_text(s) -> str:
    return ' '.join(str(s).split()).casefold()


def _gate_base_identity(gates: Mapping[str, Mapping], gid: str,
                        channel_labels: Mapping[str, str]) -> str:
    g = gates[gid]
    name = g.get('label') or g.get('name')
    if name:
        return 'n:' + _identity_text(name)
    kind = g.get('kind') or '?'

    def marker(ch):
        return _identity_text(channel_labels.get(ch, ch)) if ch else '?'
    if 'channel' in g:
        return f"{kind}:{marker(g.get('channel'))}"
    if kind in ('rect', 'polygon', 'ellipsoid', 'flowjo_ellipse'):
        return f"{kind}:{marker(g.get('x_channel'))}x{marker(g.get('y_channel'))}"
    if kind == 'cluster':
        return f"cluster:{g.get('label_col', '')}={g.get('cluster_id')}"
    if kind == 'category':
        return f"category:{g.get('column', '')}={g.get('value')}"
    if kind == 'boolean':
        ops = ','.join(sorted(
            _gate_base_identity(gates, o, channel_labels)
            for o in (g.get('operands') or []) if o in gates))
        return f"boolean:{(g.get('op') or 'and').lower()}({ops})"
    return kind


def _gate_position(g: Mapping) -> tuple:
    """Where a gate sits, to order siblings that share an identity (two
    unnamed bands on one channel): low to high on each axis."""
    def f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    if 'channel' in g:
        return (f(g.get('value', g.get('lo'))), f(g.get('hi')))
    if g.get('vertices'):
        v = g['vertices']
        return (sum(f(p[0]) for p in v) / len(v), sum(f(p[1]) for p in v) / len(v))
    if g.get('mean') is not None:
        return (f(g['mean'][0]), f(g['mean'][1]))
    return (f(g.get('x0')), f(g.get('y0')))


def population_keys(gates: Mapping[str, Mapping],
                    channel_labels: Mapping[str, str] | None = None) -> dict:
    """``{gid: key}`` identifying each population across samples (see the
    note above). Two samples' populations share a key when they sit at the
    same place in the tree with the same name -- or, unnamed, the same kind
    of gate on the same markers in the same sibling rank -- however their
    coordinates or detectors differ."""
    labels = channel_labels or {}
    base = {gid: _gate_base_identity(gates, gid, labels) for gid in gates}
    # Ordinal among siblings that share parent and identity, so they stay
    # distinct; a lone gate gets none, keeping keys stable when a sibling of
    # another identity is added.
    by_slot: dict = {}
    for gid, g in gates.items():
        by_slot.setdefault((g.get('parent_id'), base[gid]), []).append(gid)
    local = {}
    for (_pid, ident), gids in by_slot.items():
        if len(gids) == 1:
            local[gids[0]] = ident
            continue
        ranked = sorted(gids, key=lambda k: (_gate_position(gates[k]), str(k)))
        for i, k in enumerate(ranked, 1):
            local[k] = f"{ident}#{i}"
    keys = {}
    for gid in gates:
        parts, seen, cur = [], set(), gid
        while cur and cur in gates and cur not in seen:
            seen.add(cur)
            parts.append(local[cur])
            cur = gates[cur].get('parent_id')
        keys[gid] = '/'.join(reversed(parts))
    return keys


def remap_operands(gate: dict, id_map: Mapping[str, str], keep=()) -> list:
    """Point a boolean gate's ``operands`` at the ids its gates were given.

    Every path that gives gates fresh ids (copy to samples, template apply,
    move, paste, session restore) remaps ``parent_id`` through an old -> new
    map. Operands are gate ids too, and left alone they name whatever gate
    holds that id where the boolean lands. Measured: a copied AND(CD3+, CD4+)
    combined the target's own 'CD3 bright' and 'CD4 not-dim', 1,233 events
    where 5,021 is right.

    An operand in ``id_map`` takes its new id; one in ``keep`` (still valid
    where the gate lands) stays. Any other operand cannot be resolved. It is
    not dropped, which silently widens an AND or a NOT, and not kept, since
    its id may be another gate's there: the operands are emptied, and
    gate_to_mask fails closed. Returns the unresolved ids. A non-boolean gate
    is left untouched.
    """
    ops = gate.get('operands')
    if gate.get('kind') != 'boolean' or not ops:
        return []
    new, missing = [], []
    for op in ops:
        if op in id_map:
            new.append(id_map[op])
        elif op in keep:
            new.append(op)
        else:
            missing.append(op)
    gate['operands'] = [] if missing else new
    return missing


def booleans_using(gates: Mapping[str, Mapping], ids) -> list:
    """Ids of the boolean gates in ``gates``, other than ``ids`` themselves,
    that combine any gate in ``ids``."""
    ids = set(ids)
    return [gid for gid, g in gates.items()
            if gid not in ids and g.get('kind') == 'boolean'
            and ids.intersection(g.get('operands') or ())]


def population_stats(sample_name, df, gates, order, channel_labels, channels,
                     want, stat_chan, select=None, transforms=None):
    """Statistic rows for ONE sample's populations (pure — no Tk).

    ``want`` is the set of selected stat names; ``stat_chan`` the per-channel
    subset (Median/Mean/GeoMean/CV/rCV). Counts are computed over the FULL
    gate tree (so
    %Parent stays correct) even when ``select`` restricts which gates emit
    rows. Each row carries a hidden ``__gid__``. Empty populations yield NaN
    per-channel and 0 counts.

    ``transforms`` maps a channel to the transform_spec baked into ``df``
    (a FlowSample's ``data_transforms``). Gates are evaluated on ``df`` as
    stored, because gate coordinates live in that display space, but the
    per-channel statistics are computed on the compensated LINEAR values, as
    FlowJo reports MFI, GeoMean and CV. On logicle values a lognormal(7, 0.5)
    channel measured GeoMean 0.46 and CV 11 % where its linear values give
    1,094 and 53 %, and only the median can be recovered from such a number
    afterwards.
    """
    import numpy as np

    from .pipeline import cumulative_gate_mask, is_unknown_spec, to_linear
    total = len(df)
    order = order or list(gates)
    counts, masks = {}, {}
    # One cache for this whole collection pass: `gates`/`df` are fixed here, so
    # a child reuses its parent's cumulative mask instead of re-walking (and
    # re-evaluating) the entire ancestor chain. Masks are read-only below —
    # only boolean-indexed and summed.
    mask_cache = {}
    for gid in order:
        if gid not in gates:
            continue
        m = cumulative_gate_mask(gates, gid, df, cache=mask_cache)
        masks[gid] = m
        counts[gid] = int(np.asarray(m).sum())

    emit = select if select is not None else order
    keys = population_keys(gates, channel_labels)
    rows = []
    row_gids = []
    for gid in emit:
        if gid not in gates or gid not in counts:
            continue
        g = gates[gid]
        cnt = counts[gid]
        parent = g.get('parent_id')
        parent_cnt = counts.get(parent, total) if parent else total
        row = {'Sample': sample_name,
               # Displayed and exported: an unnamed gate's bounds as the
               # intensities the gate list shows, not stored coordinates.
               'Population': population_path(gates, gid,
                                             transforms=transforms),
               '__gid__': gid,
               # Identity across samples (population_keys); the path above is
               # for display and changes with a gate's coordinates.
               '__key__': keys.get(gid, str(gid))}
        if 'Count' in want:
            row['Count'] = cnt
        if '%Parent' in want:
            # An empty parent has no fraction to give: NaN, not a 0 % that
            # reads as "measured, none of them".
            row['%Parent'] = ((cnt / parent_cnt * 100.0) if parent_cnt
                              else float('nan'))
        if '%Total' in want:
            row['%Total'] = (cnt / total * 100.0) if total else float('nan')
        rows.append(row)
        row_gids.append(gid)

    need_chan = want & set(stat_chan)
    if not (need_chan and channels):
        return rows
    # Channel by channel, so only ONE linearised column is alive at a time
    # (holding them all would be 8 MB per channel per million events), and
    # each column is inverted once rather than once per population. Every
    # row still receives its channels in `channels` order, so the column
    # order the callers derive from first-seen keys is unchanged.
    transforms = transforms or {}
    for ch in channels:
        if ch not in df.columns:
            continue
        lbl = channel_labels.get(ch, ch)
        # A column whose display transform is unknown (an old OpenFlo CSV)
        # has no linear values: its statistics are left out (NaN, to_linear)
        # and every row names it, so the window can say why they are blank.
        unknown = is_unknown_spec(transforms.get(ch))
        lin = None
        for row, gid in zip(rows, row_gids, strict=True):
            if unknown:
                row.setdefault('__scale_unknown__', []).append(lbl)
            if not counts[gid]:
                med = mean = cv = gmean = rcv = float('nan')
            else:
                if lin is None:
                    lin = to_linear(df[ch].values, transforms.get(ch))
                vals = lin[np.asarray(masks[gid], dtype=bool)]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    med = mean = cv = gmean = rcv = float('nan')
                else:
                    med = float(np.median(vals))
                    mean = float(np.mean(vals))
                    sd = float(np.std(vals))
                    cv = (sd / mean * 100.0) if mean else float('nan')
                    # Geometric mean, as FlowJo defines it: the mean in the
                    # channel's display (transform) space, back-transformed.
                    # That is what makes it defined for the negative values
                    # compensation produces. The classic exp(mean(log x))
                    # had to drop them, which biased a dim population
                    # upward (measured: median 151, GeoMean 211.7 from 69%
                    # of its events, where FlowJo's reads 83.5). A channel
                    # stored linear has no display space here, so it keeps
                    # the classic form over its positive values.
                    spec = transforms.get(ch)
                    if spec and spec.get('method', 'linear') != 'linear':
                        stored = np.asarray(df[ch].values, dtype=float)[
                            np.asarray(masks[gid], dtype=bool)]
                        stored = stored[np.isfinite(stored)]
                        gmean = (float(to_linear(
                            np.array([np.mean(stored)]), spec)[0])
                            if stored.size else float('nan'))
                    else:
                        pos = vals[vals > 0]
                        gmean = (float(np.exp(np.mean(np.log(pos))))
                                 if pos.size else float('nan'))
                    # Robust CV, FlowJo's definition: 100 * 1/2 (P84.13 -
                    # P15.87) / median. The MAD form used before agrees only
                    # for a symmetric population (log-normal sigma 0.8: 74%
                    # where FlowJo reports 89%), under the same column name.
                    p_lo, p_hi = np.percentile(vals, [15.87, 84.13])
                    rcv = (50.0 * float(p_hi - p_lo) / med
                           if med else float('nan'))
            if 'Median' in want:
                row[f'Median {lbl}'] = med
            if 'Mean' in want:
                row[f'Mean {lbl}'] = mean
            if 'GeoMean' in want:
                row[f'GeoMean {lbl}'] = gmean
            if 'CV' in want:
                row[f'CV {lbl}'] = cv
            if 'rCV' in want:
                row[f'rCV {lbl}'] = rcv
    return rows
