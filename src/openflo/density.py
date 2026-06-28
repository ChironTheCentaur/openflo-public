"""Per-event 2-D density for pseudocolour scatter — Tk-free, testable.

The heavy numpy/scipy core of the density-scatter plot: given event x/y and
pre-computed (scale-aware) bin edges, return a smooth per-event density. The
GUI keeps the scale-dependent edge calc and the matplotlib draw; this is the
maths, so it can be checked against arrays directly.
"""
from __future__ import annotations

import numpy as np


def event_density(xs, ys, x_edges, y_edges):
    """Smooth per-event density aligned to ``xs`` / ``ys``.

    A 2-D histogram over ``(x_edges, y_edges)`` is zero-padded, Gaussian-
    smoothed (adaptive sigma, floored at ~1.8 bins so a dense histogram's bin
    lattice doesn't show), then sampled per event by **cubic** interpolation
    (C2-continuous → no facet edges at bin boundaries). Events past the padded
    border interpolate toward zero, so the density support has no hard edge.
    Returns a non-negative ``z`` array the same length as ``xs``.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    hist, x_edges, y_edges = np.histogram2d(xs, ys, bins=[x_edges, y_edges])

    n_bins = (len(x_edges) - 1) * (len(y_edges) - 1)
    per_bin = xs.size / float(max(n_bins, 1))
    sigma = float(np.clip(np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.2, 1.8, 6.0))

    pad = int(np.ceil(3.0 * sigma)) + 1
    hist = np.pad(hist, pad, mode='constant', constant_values=0.0)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    x_edges = np.concatenate([x_edges[0] + dx * np.arange(-pad, 0), x_edges,
                              x_edges[-1] + dx * np.arange(1, pad + 1)])
    y_edges = np.concatenate([y_edges[0] + dy * np.arange(-pad, 0), y_edges,
                              y_edges[-1] + dy * np.arange(1, pad + 1)])
    hist = gaussian_filter(hist, sigma=sigma)

    nbx, nby = len(x_edges) - 1, len(y_edges) - 1
    ix = np.clip(np.searchsorted(x_edges, xs, side='right') - 1, 0, nbx - 1)
    iy = np.clip(np.searchsorted(y_edges, ys, side='right') - 1, 0, nby - 1)
    wx = x_edges[ix + 1] - x_edges[ix]
    wy = y_edges[iy + 1] - y_edges[iy]
    fx = ix + np.where(wx > 0, (xs - x_edges[ix]) / wx, 0.5) - 0.5
    fy = iy + np.where(wy > 0, (ys - y_edges[iy]) / wy, 0.5) - 0.5
    z = map_coordinates(hist, np.vstack([fx, fy]), order=3, mode='constant',
                        cval=0.0)
    np.clip(z, 0.0, None, out=z)
    return z


def kde_density(xs, ys, max_src=15_000, max_display=40_000, seed=42):
    """True Gaussian-KDE density for the pseudocolour scatter.

    ``gaussian_kde`` is O(n_src · n_query), so both the KDE source and the
    displayed points are independently subsampled to stay tractable. Returns
    ``(xs_d, ys_d, z, n_src)`` where ``xs_d``/``ys_d`` are the points to draw,
    ``z`` their KDE density, and ``n_src`` is the number of source events the
    kernel was fit on (``< len(xs)`` ⇒ the source was capped). The GUI keeps
    the status message and the matplotlib draw.
    """
    from scipy.stats import gaussian_kde

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rng = np.random.default_rng(seed)
    if xs.size > max_src:
        src = rng.choice(xs.size, max_src, replace=False)
        xs_src, ys_src = xs[src], ys[src]
    else:
        xs_src, ys_src = xs, ys
    if xs.size > max_display:
        disp = rng.choice(xs.size, max_display, replace=False)
        xs_d, ys_d = xs[disp], ys[disp]
    else:
        xs_d, ys_d = xs, ys
    kernel = gaussian_kde(np.vstack([xs_src, ys_src]))
    z = kernel(np.vstack([xs_d, ys_d]))
    return xs_d, ys_d, z, int(xs_src.size)
