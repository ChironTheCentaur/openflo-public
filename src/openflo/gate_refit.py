"""Gates that re-fit to each sample -- Tk-free.

A copied gate stays where it was drawn on every sample, so a population that
drifts a little between samples or days is cut differently in each. The
literature answer is data-driven placement per sample: find the same
landmark (a density valley, a population's centre) in each sample's own
data (flowDensity, Malek et al., Bioinformatics 2015; openCyto, Finak et
al., PLoS Comput Biol 2014). This module re-fits one gate on one sample's
events:

* a threshold moves to the density valley nearest the reference cut, and an
  interval's finite ends each do the same;
* a gate an auto-gate proposed carries its rule (``gate['rule']``) and is
  re-fitted by that rule: the singlet band is fitted again, and a Gaussian
  mixture ellipse becomes the component of the new fit nearest the
  reference centre.

Every re-fit is a proposal; the editor shows each sample's placement before
anything changes. When the data give no landmark (too few events, no valley
near the cut) the gate stays as it was and the reason is said.
"""
from __future__ import annotations

import copy

import numpy as np

OPEN = 1e12

# How far from the reference cut a valley may be and still be "the same"
# valley, as a fraction of the data's 1st-99th percentile span.
SEARCH_FRACTION = 0.25

# A peak must stand this far (a fraction of the tallest) above the valleys
# either side to be a population rather than noise.
PROMINENCE = 0.10


def refittable(gate) -> bool:
    """Whether `refit` can re-place this gate."""
    k = gate.get('kind')
    if gate.get('scale_unknown'):
        return False
    if k == 'threshold':
        return abs(float(gate.get('value', 0.0))) < OPEN
    if k == 'interval':
        return any(abs(float(gate.get(e, OPEN))) < OPEN for e in ('lo', 'hi'))
    rule = (gate.get('rule') or {}).get('method')
    return (k, rule) in (('polygon', 'singlet'), ('ellipsoid', 'gmm'))


def _smoothed_hist(v, bins=128):
    """The histogram of ``v`` smoothed with Silverman's rule-of-thumb
    bandwidth, so a few hundred events from one population do not show
    noise bumps as extra populations (a fixed few-bin smoothing found a
    'valley' in 4 of 5 samples of 1,000 normal events)."""
    from scipy.ndimage import gaussian_filter1d
    lo, hi = np.percentile(v, [0.5, 99.5])
    if not hi > lo:
        return None
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    q75, q25 = np.percentile(v, [75, 25])
    spread = min(float(np.std(v)), float(q75 - q25) / 1.34) or float(np.std(v))
    h = 0.9 * spread * v.size ** -0.2
    sigma = max(1.0, h / (edges[1] - edges[0]))
    return gaussian_filter1d(hist.astype(float), sigma), centers


def nearest_valley(values, ref, search=SEARCH_FRACTION):
    """``(cut, why)``: the density valley of ``values`` nearest ``ref``
    within ``search`` of the data's 1st-99th percentile span, or ``(None,
    why)`` when there is none. A valley is a local minimum of the smoothed
    histogram with a peak on each side at least PROMINENCE of the tallest
    peak above it, so noise in a flat tail is not one."""
    from scipy.signal import find_peaks
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None, 'too few events'
    sh = _smoothed_hist(v)
    if sh is None:
        return None, 'no spread in the data'
    sm, centers = sh
    top = sm.max()
    if top <= 0:
        return None, 'no events in range'
    peaks, _ = find_peaks(sm, prominence=top * PROMINENCE)
    if peaks.size < 2:
        return None, 'one population, no valley'
    cands = []
    for a, b in zip(peaks[:-1], peaks[1:], strict=True):
        # The middle of the low ground between the two peaks: two
        # well-separated populations leave a flat, empty gap, and its first
        # bin would put the cut against the lower population.
        seg = sm[a:b + 1]
        low = np.flatnonzero(seg <= seg.min() + top * 0.01)
        cands.append(float(0.5 * (centers[a + low[0]] + centers[a + low[-1]])))
    p1, p99 = np.percentile(v, [1, 99])
    reach = search * max(float(p99 - p1), 1e-12)
    cands = [c for c in cands if abs(c - float(ref)) <= reach]
    if not cands:
        return None, 'no valley near the reference cut'
    return min(cands, key=lambda c: abs(c - float(ref))), None


def _refit_gmm(gate, x, y):
    from .pipeline import gmm_ellipse_gates
    rule = gate.get('rule') or {}
    props = gmm_ellipse_gates(
        x, y, max_components=int(rule.get('max_components', 6)),
        coverage=float(rule.get('coverage', 0.90)),
        min_weight=float(rule.get('min_weight', 0.02)))
    if not props:
        return None, 'no populations found'
    ref = np.asarray(gate['mean'], dtype=float)
    inv = np.linalg.pinv(np.asarray(gate['cov'], dtype=float))

    def dist(p):
        d = np.asarray(p[0]['mean'], dtype=float) - ref
        return float(d @ inv @ d)
    best = min(props, key=dist)
    if dist(best) > float(gate.get('distance_sq', 4.0)):
        return None, 'no population near the reference centre'
    return best[0], None


def refit(gate, data) -> tuple[dict, str | None]:
    """``(new_gate, why)``: ``gate`` re-placed on one sample's events.

    ``data`` maps each channel the gate uses to that sample's values in the
    gate's parent population (stored coordinates). ``why`` is None when the
    gate moved to a landmark, else the reason it was left as it was (the
    returned gate is then an unchanged copy)."""
    g = copy.deepcopy(gate)
    k = g.get('kind')
    if not refittable(g):
        return g, 'this gate has no rule to re-fit'
    if k == 'threshold':
        cut, why = nearest_valley(data[g['channel']], g['value'])
        if cut is not None:
            g['value'] = cut
        return g, why
    if k == 'interval':
        whys = []
        for end in ('lo', 'hi'):
            if abs(float(g[end])) >= OPEN:
                continue
            cut, why = nearest_valley(data[g['channel']], g[end])
            if cut is None:
                whys.append(f"{'lower' if end == 'lo' else 'upper'} end: {why}")
            else:
                g[end] = cut
        if g['lo'] >= g['hi']:
            return copy.deepcopy(gate), 'the ends met'
        return g, '; '.join(whys) or None
    x = np.asarray(data[g['x_channel']], dtype=float)
    y = np.asarray(data[g['y_channel']], dtype=float)
    rule = g.get('rule') or {}
    if rule.get('method') == 'singlet':
        from .pipeline import auto_singlet_gate
        verts, _q = auto_singlet_gate(x, y, k=float(rule.get('k', 3.0)))
        if not verts:
            return g, 'singlet band undefined (too little spread or data)'
        g['vertices'] = [[float(a), float(b)] for a, b in verts]
        return g, None
    new, why = _refit_gmm(g, x, y)
    if new is None:
        return g, why
    for key in ('mean', 'cov', 'distance_sq'):
        g[key] = copy.deepcopy(new[key])
    return g, None


def fraction_in(gate, df) -> float:
    """The fraction of ``df``'s events inside ``gate`` alone (NaN when
    ``df`` is empty): the review grid's before / after numbers."""
    from .pipeline import gate_to_mask
    if len(df) == 0:
        return float('nan')
    return float(np.asarray(gate_to_mask(gate, df), dtype=bool).mean())


# The keys that say where a gate is (and how it re-fits), copied from the
# reference gate onto each sample's own gate before it is re-fitted.
GEOMETRY_KEYS = ('value', 'lo', 'hi', 'vertices', 'mean', 'cov',
                 'distance_sq', 'rule')


def with_geometry(target, ref) -> dict:
    """A copy of ``target`` (one sample's gate, keeping its name, colour and
    place in the tree) placed where ``ref`` is."""
    g = copy.deepcopy(target)
    for k in GEOMETRY_KEYS:
        if k in ref:
            g[k] = copy.deepcopy(ref[k])
    return g


def outline(gate):
    """``(xs, ys)`` of a 2-D gate's boundary for the review grid, or None."""
    k = gate.get('kind')
    if k == 'polygon':
        v = np.asarray(gate['vertices'], dtype=float)
    elif k == 'ellipsoid':
        from .pipeline import _ellipse_polygon
        verts = _ellipse_polygon(gate['mean'], gate['cov'],
                                 gate.get('distance_sq', 4.0))
        if verts is None:
            return None
        v = np.asarray(verts, dtype=float)
    else:
        return None
    v = np.vstack([v, v[:1]])
    return v[:, 0], v[:, 1]
