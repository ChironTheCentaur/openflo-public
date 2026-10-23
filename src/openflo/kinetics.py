"""Kinetics: a channel over acquisition time, and each sample's response.

Calcium flux is the classic case: a baseline is acquired, a stimulus added,
and the response read as the signal (often a derived ratio, derived.py)
against Time. Dye uptake or efflux and membrane-potential assays read the
same way. Per sample this gives a binned trace (the median, or mean, of the
events in each time bin) and its summary: the baseline level, the peak
after the stimulus, the time to that peak, the area between the trace and
the baseline over the response, and the peak over the baseline.

Time is the FCS Time channel, in seconds when the file says how long a tick
is ($TIMESTEP), else in ticks (and said so). Values are the channel's
compensated linear values. A number that can't be computed (no events in
the baseline, a baseline at or below 0 for a fold) is NaN, never 0.

Tk-free, so it is tested without a display.
"""
from __future__ import annotations

import numpy as np

STATS = ('median', 'mean')


def time_column(sample):
    """The sample's Time channel name, or None."""
    from .pipeline import _is_time
    for c in sample.data.columns:
        if _is_time(str(c)):
            return c
    return None


def seconds_per_tick(sample):
    """Seconds per Time tick from the FCS keywords ($TIMESTEP), or None."""
    from .pipeline import _positive_float
    meta = getattr(sample, 'metadata', None) or {}
    for k in ('timestep', '$TIMESTEP', 'TIMESTEP'):
        if k in meta:
            return _positive_float(meta[k])
    return None


def sample_time(sample):
    """``(seconds or ticks array, unit)`` for the sample's events; unit is
    's' or 'ticks'. Raises ValueError when there is no Time channel."""
    col = time_column(sample)
    if col is None:
        raise ValueError(f"'{getattr(sample, 'name', '?')}' has no Time "
                         "channel.")
    t = np.asarray(sample.data[col].values, dtype=float)
    ts = seconds_per_tick(sample)
    return (t * ts, 's') if ts else (t, 'ticks')


def trace(time, values, bins=100, stat='median', t_range=None):
    """Binned trace: ``(bin centres, statistic per bin, events per bin)``.
    `bins` equal-width bins over `t_range` (default: the events' own time
    range). A bin with no events is NaN."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}")
    if t_range is None:
        if not t.size:
            return np.array([]), np.array([]), np.array([], dtype=int)
        t_range = (float(t.min()), float(t.max()))
    lo, hi = float(t_range[0]), float(t_range[1])
    if not hi > lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, int(bins) + 1)
    idx = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, int(bins) - 1)
    inside = (t >= lo) & (t <= hi)
    idx, vv = idx[inside], v[inside]
    counts = np.bincount(idx, minlength=int(bins))
    out = np.full(int(bins), np.nan)
    if stat == 'mean':
        sums = np.bincount(idx, weights=vv, minlength=int(bins))
        has = counts > 0
        out[has] = sums[has] / counts[has]
    else:
        order = np.argsort(idx, kind='stable')
        idx_s, v_s = idx[order], vv[order]
        starts = np.searchsorted(idx_s, np.arange(int(bins)))
        stops = np.searchsorted(idx_s, np.arange(int(bins)), side='right')
        for b in np.flatnonzero(counts):
            out[b] = float(np.median(v_s[starts[b]:stops[b]]))
    return 0.5 * (edges[:-1] + edges[1:]), out, counts


def summarize(time, values, baseline_end, stimulus=None, bins=100,
              stat='median', t_range=None):
    """One sample's response.

    `baseline_end`: events before this time are the baseline. `stimulus`:
    when it was added (default `baseline_end`); the response is everything
    from it on. Returns::

        {baseline, peak, time_to_peak, auc, fold, n_events, n_baseline}

    ``baseline`` is the `stat` of the baseline events themselves (not of
    the bins); ``peak`` the highest bin of the trace from the stimulus on,
    and ``time_to_peak`` its bin centre minus the stimulus time; ``auc``
    the area between the trace and the baseline over the response (bins
    with no events are skipped, not taken as 0), in value x time units;
    ``fold`` peak / baseline, NaN when the baseline is not above 0."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    stim = float(baseline_end if stimulus is None else stimulus)
    nan = float('nan')
    base_v = v[t < float(baseline_end)]
    agg = np.median if stat == 'median' else np.mean
    baseline = float(agg(base_v)) if base_v.size else nan
    centres, tr, _counts = trace(t, v, bins=bins, stat=stat, t_range=t_range)
    resp = (centres >= stim) & np.isfinite(tr)
    out = {'baseline': baseline, 'peak': nan, 'time_to_peak': nan,
           'auc': nan, 'fold': nan, 'n_events': int(t.size),
           'n_baseline': int(base_v.size)}
    if not resp.any():
        return out
    rc, rv = centres[resp], tr[resp]
    k = int(np.argmax(rv))
    out['peak'] = float(rv[k])
    out['time_to_peak'] = float(rc[k] - stim)
    if np.isfinite(baseline):
        if rv.size > 1:                 # one bin has no width to integrate
            d = rv - baseline
            out['auc'] = float(np.sum(0.5 * (d[1:] + d[:-1]) * np.diff(rc)))
        if baseline > 0:
            out['fold'] = out['peak'] / baseline
    return out
