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


def population_path(gates: Mapping[str, Mapping], gid: str) -> str:
    """Human-readable population path, e.g. ``'Cells/Singlets/CD11b+'``, built
    by walking ``parent_id`` to the root. Cycle-safe."""
    from .pipeline import describe_gate
    names, seen, cur = [], set(), gid
    while cur and cur in gates and cur not in seen:
        seen.add(cur)
        g = gates[cur]
        names.append(g.get('label') or g.get('name') or describe_gate(g) or cur)
        cur = g.get('parent_id')
    return '/'.join(reversed(names)) if names else str(gid)


def population_stats(sample_name, df, gates, order, channel_labels, channels,
                     want, stat_chan, select=None):
    """Statistic rows for ONE sample's populations (pure — no Tk).

    ``want`` is the set of selected stat names; ``stat_chan`` the per-channel
    subset (Median/Mean/CV). Counts are computed over the FULL gate tree (so
    %Parent stays correct) even when ``select`` restricts which gates emit
    rows. Each row carries a hidden ``__gid__``. Empty populations yield NaN
    per-channel and 0 counts.
    """
    import numpy as np

    from .pipeline import cumulative_gate_mask
    total = len(df)
    order = order or list(gates)
    counts, masks = {}, {}
    for gid in order:
        if gid not in gates:
            continue
        m = cumulative_gate_mask(gates, gid, df)
        masks[gid] = m
        counts[gid] = int(np.asarray(m).sum())

    emit = select if select is not None else order
    rows = []
    for gid in emit:
        if gid not in gates or gid not in counts:
            continue
        g = gates[gid]
        cnt = counts[gid]
        parent = g.get('parent_id')
        parent_cnt = counts.get(parent, total) if parent else total
        row = {'Sample': sample_name,
               'Population': population_path(gates, gid),
               '__gid__': gid}
        if 'Count' in want:
            row['Count'] = cnt
        if '%Parent' in want:
            row['%Parent'] = (cnt / parent_cnt * 100.0) if parent_cnt else 0.0
        if '%Total' in want:
            row['%Total'] = (cnt / total * 100.0) if total else 0.0

        need_chan = want & set(stat_chan)
        if need_chan and channels:
            sub = df[masks[gid]] if cnt else None
            for ch in channels:
                lbl = channel_labels.get(ch, ch)
                if ch not in df.columns:
                    continue
                if sub is None or len(sub) == 0:
                    med = mean = cv = float('nan')
                else:
                    vals = np.asarray(sub[ch].values, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        med = mean = cv = float('nan')
                    else:
                        med = float(np.median(vals))
                        mean = float(np.mean(vals))
                        sd = float(np.std(vals))
                        cv = (sd / mean * 100.0) if mean else float('nan')
                if 'Median' in want:
                    row[f'Median {lbl}'] = med
                if 'Mean' in want:
                    row[f'Mean {lbl}'] = mean
                if 'CV' in want:
                    row[f'CV {lbl}'] = cv
        rows.append(row)
    return rows
