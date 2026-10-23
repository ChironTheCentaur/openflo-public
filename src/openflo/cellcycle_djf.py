"""Dean-Jett-Fox cell-cycle model (Complaints 21).

The DNA histogram is fitted as a G1 Gaussian, a G2/M Gaussian near twice
its mean, and an S phase that is a second-order polynomial in DNA content
between the two peaks, each S-phase cell measured with the same broadening
as the peaks (Fox 1980, Cytometry 1:71; after Dean and Jett 1974). The
phase fractions are the fitted areas, not counts between intensity
boundaries: the S-phase cells hidden under each peak are counted as S, and
the G1 and G2 cells in the tails of the peaks are not.

Pure numpy / scipy.
"""
from __future__ import annotations

import numpy as np

_SQRT2PI = np.sqrt(2.0 * np.pi)
_S_POINTS = 96                  # S-phase integration grid between the peaks


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * _SQRT2PI)


def _components(p, x, width):
    """The three fitted curves (counts per bin) at bin centres ``x``."""
    a1, m1, s1, a2, ratio, s2, c0, c1, c2 = p
    m2 = ratio * m1
    t = (np.arange(_S_POINTS) + 0.5) / _S_POINTS
    dens = np.clip(c0 + c1 * t + c2 * t * t, 0.0, None)    # S cells per unit t
    pos = m1 + t * (m2 - m1)
    sd = s1 + t * (s2 - s1)
    s_curve = (dens / _S_POINTS)[None, :] * _gauss(x[:, None], pos[None, :],
                                                   sd[None, :])
    g1 = a1 * _gauss(x, m1, s1) * width
    g2 = a2 * _gauss(x, m2, s2) * width
    return g1, s_curve.sum(axis=1) * width, g2, float(dens.mean())


def fit_dean_jett_fox(values, g1_mean, g2_mean=None, g1_sd=None, g2_sd=None,
                      bins=256):
    """Fit the Dean-Jett-Fox model to linear DNA-content ``values``.

    ``g1_mean`` (and optionally ``g2_mean``, ``g1_sd``, ``g2_sd``) start the
    fit, as pipeline.analyze_dna finds them. Returns a dict with
    ``pct_g1``, ``pct_s``, ``pct_g2m`` (of the cycling cells), the fitted
    ``g1_mean``, ``g2_mean``, ``g1_sd``, ``g2_sd``, ``g1_cv``, the reduced
    chi-square ``rcs``, ``ok``, and the histogram and curves (``x``,
    ``hist``, ``fit_g1``, ``fit_s``, ``fit_g2``) for a plot. ``ok`` is
    False, and the percents NaN, when the fit fails."""
    from scipy.optimize import least_squares
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    nan = float('nan')
    out = {'method': 'Dean-Jett-Fox', 'ok': False, 'pct_g1': nan,
           'pct_s': nan, 'pct_g2m': nan}
    if not (np.isfinite(g1_mean) and g1_mean > 0) or v.size < 200:
        return out
    g2_mean = g2_mean if g2_mean and np.isfinite(g2_mean) else 2.0 * g1_mean
    g1_sd = g1_sd if g1_sd and np.isfinite(g1_sd) and g1_sd > 0 \
        else 0.05 * g1_mean
    g2_sd = g2_sd if g2_sd and np.isfinite(g2_sd) and g2_sd > 0 \
        else g1_sd * 1.4
    lo = g1_mean - 4.0 * g1_sd
    hi = g2_mean + 4.0 * g2_sd
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 200:
        return out
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    x = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])
    hist = hist.astype(float)
    n = float(hist.sum())
    weight = 1.0 / np.sqrt(np.maximum(hist, 1.0))      # Poisson errors

    def resid(p):
        g1, s, g2, _d = _components(p, x, width)
        return (g1 + s + g2 - hist) * weight

    ratio0 = min(max(g2_mean / g1_mean, 1.8), 2.2)
    p0 = [0.5 * n, g1_mean, g1_sd, 0.15 * n, ratio0, g2_sd,
          0.3 * n, 0.0, 0.0]
    lower = [0, 0.8 * g1_mean, 0.2 * g1_sd, 0, 1.75, 0.2 * g1_sd,
             -np.inf, -np.inf, -np.inf]
    upper = [np.inf, 1.2 * g1_mean, 5.0 * g1_sd, np.inf, 2.25,
             8.0 * g1_sd, np.inf, np.inf, np.inf]
    try:
        fit = least_squares(resid, p0, bounds=(lower, upper), x_scale='jac',
                            max_nfev=4000)
    except Exception:                                   # noqa: BLE001
        return out
    p = fit.x
    g1c, sc, g2c, s_area = _components(p, x, width)
    a1, m1, s1, a2, ratio, s2 = p[:6]
    total = a1 + a2 + s_area
    if not fit.success or total <= 0:
        return out
    dof = max(1, int(np.count_nonzero(hist)) - len(p))
    out.update(
        ok=True, pct_g1=100.0 * a1 / total, pct_s=100.0 * s_area / total,
        pct_g2m=100.0 * a2 / total, g1_mean=float(m1),
        g2_mean=float(ratio * m1), g1_sd=float(s1), g2_sd=float(s2),
        g1_cv=100.0 * float(s1 / m1), g2_g1_ratio=float(ratio),
        rcs=float(np.sum(fit.fun ** 2) / dof), n=int(n),
        x=x, hist=hist, fit_g1=g1c, fit_s=sc, fit_g2=g2c, params=p)
    return out


def phase_by_model(values, fit, current):
    """Per-event phases from a Dean-Jett-Fox ``fit``: each event the
    boundaries method called G1, S or G2M goes to the phase whose fitted
    curve is highest at its DNA content; sub-G1, >G2M and NA stay."""
    v = np.asarray(values, dtype=float)
    out = np.asarray(current, dtype=object).copy()
    if not fit.get('ok'):
        return out
    p = fit['params']
    cyc = np.isin(out, ['G1', 'S', 'G2M']) & np.isfinite(v)
    if not cyc.any():
        return out
    g1, s, g2, _d = _components(p, v[cyc], 1.0)
    names = np.array(['G1', 'S', 'G2M'], dtype=object)
    out[cyc] = names[np.argmax(np.vstack([g1, s, g2]), axis=0)]
    return out
