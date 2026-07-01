"""Per-event 2-D density for pseudocolour scatter — Tk-free, testable.

The heavy numpy/scipy core of the density-scatter plot: given event x/y and
pre-computed (scale-aware) bin edges, return a smooth per-event density. The
GUI keeps the scale-dependent edge calc and the matplotlib draw; this is the
maths, so it can be checked against arrays directly.
"""
from __future__ import annotations

import numpy as np


def _event_density_gpu(xs, ys, x_edges, y_edges):
    """GPU twin of :func:`event_density` (cupy + cupyx.scipy.ndimage), mirroring
    the CPU steps exactly. Assumes UNIFORM bin edges (the GUI's scale-aware
    edges always are). Display-only, so small fp drift is fine. Returns None on
    any failure / unavailability so the caller falls back to the SciPy path."""
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter, map_coordinates
    except Exception:
        return None
    try:
        xs = cp.asarray(xs, dtype=cp.float64)
        ys = cp.asarray(ys, dtype=cp.float64)
        xe = cp.asarray(x_edges, dtype=cp.float64)
        ye = cp.asarray(y_edges, dtype=cp.float64)
        nbx, nby = int(xe.size - 1), int(ye.size - 1)
        x0, x1 = float(xe[0]), float(xe[-1])
        y0, y1 = float(ye[0]), float(ye[-1])
        # cupy.histogram2d takes bin COUNTS + range (not array edges); identical
        # to np.histogram2d for the uniform edges we receive here.
        hist, _, _ = cp.histogram2d(xs, ys, bins=(nbx, nby),
                                    range=((x0, x1), (y0, y1)))
        # sigma + pad — same formula as the CPU path below (keep in sync).
        per_bin = xs.size / float(max(nbx * nby, 1))
        sigma = float(np.clip(np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.2, 1.8, 6.0))
        pad = int(np.ceil(3.0 * sigma)) + 1
        hist = cp.pad(hist, pad, mode='constant', constant_values=0.0)
        dx, dy = (x1 - x0) / nbx, (y1 - y0) / nby
        xe_p = cp.concatenate([xe[0] + dx * cp.arange(-pad, 0), xe,
                               xe[-1] + dx * cp.arange(1, pad + 1)])
        ye_p = cp.concatenate([ye[0] + dy * cp.arange(-pad, 0), ye,
                               ye[-1] + dy * cp.arange(1, pad + 1)])
        hist = gaussian_filter(hist, sigma=sigma)
        nbxp, nbyp = int(xe_p.size - 1), int(ye_p.size - 1)
        ix = cp.clip(cp.searchsorted(xe_p, xs, side='right') - 1, 0, nbxp - 1)
        iy = cp.clip(cp.searchsorted(ye_p, ys, side='right') - 1, 0, nbyp - 1)
        wx, wy = xe_p[ix + 1] - xe_p[ix], ye_p[iy + 1] - ye_p[iy]
        fx = ix + cp.where(wx > 0, (xs - xe_p[ix]) / wx, 0.5) - 0.5
        fy = iy + cp.where(wy > 0, (ys - ye_p[iy]) / wy, 0.5) - 0.5
        z = map_coordinates(hist, cp.vstack([fx, fy]), order=3,
                            mode='constant', cval=0.0)
        return cp.asnumpy(cp.clip(z, 0.0, None))
    except Exception:
        return None


def event_density(xs, ys, x_edges, y_edges):
    """Smooth per-event density aligned to ``xs`` / ``ys``.

    A 2-D histogram over ``(x_edges, y_edges)`` is zero-padded, Gaussian-
    smoothed (adaptive sigma, floored at ~1.8 bins so a dense histogram's bin
    lattice doesn't show), then sampled per event by **cubic** interpolation
    (C2-continuous → no facet edges at bin boundaries). Events past the padded
    border interpolate toward zero, so the density support has no hard edge.
    Returns a non-negative ``z`` array the same length as ``xs``.

    Runs on the GPU when acceleration is enabled (Preferences); otherwise (and
    on any GPU error) the SciPy path below runs — so the default is unchanged.
    """
    from . import gpu_accel
    if gpu_accel.enabled():
        z = _event_density_gpu(xs, ys, x_edges, y_edges)
        if z is not None:
            return z

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


def _kde_eval_gpu(xs_src, ys_src, xs_q, ys_q):
    """GPU Gaussian-KDE evaluation matching scipy.stats.gaussian_kde (Scott
    bandwidth, 2-D). The 2x2 covariance / inverse / norm are computed on the
    CPU (trivial, and avoids needing cusolver on the GPU); the heavy
    O(n_src·n_query) kernel sum runs on the GPU, chunked over queries to bound
    memory. Returns the density array, or None on any failure → caller uses the
    SciPy path."""
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        n = int(xs_src.size)
        # scipy: kernel_cov = cov(data, ddof=1) * scotts_factor**2;
        # density = (1/norm) Σ exp(-0.5 (x-q)^T inv (x-q));
        # norm = sqrt(det(2π·kernel_cov)) · n.   (d = 2)
        cov = np.cov(np.vstack([xs_src, ys_src]))          # ddof=1, like scipy
        kcov = cov * (n ** (-2.0 / 6.0))                   # n^(-1/(d+4)) squared
        a, b, dd = float(kcov[0, 0]), float(kcov[0, 1]), float(kcov[1, 1])
        det = a * dd - b * b
        if not np.isfinite(det) or det <= 0:
            return None
        i00, i01, i11 = dd / det, -b / det, a / det        # inv of the 2x2
        norm = float(np.sqrt((2.0 * np.pi) ** 2 * det) * n)
        xs_s, ys_s = cp.asarray(xs_src), cp.asarray(ys_src)
        out = np.empty(int(xs_q.size))
        CH = 512
        for s in range(0, xs_q.size, CH):
            xq = cp.asarray(xs_q[s:s + CH])
            yq = cp.asarray(ys_q[s:s + CH])
            dx = xs_s[:, None] - xq[None, :]                # (n_src, chunk)
            dy = ys_s[:, None] - yq[None, :]
            energy = 0.5 * (i00 * dx * dx + 2.0 * i01 * dx * dy + i11 * dy * dy)
            out[s:s + CH] = cp.asnumpy(cp.sum(cp.exp(-energy), axis=0) / norm)
        return out
    except Exception:
        return None


def kde_density(xs, ys, max_src=15_000, max_display=40_000, seed=42):
    """True Gaussian-KDE density for the pseudocolour scatter.

    ``gaussian_kde`` is O(n_src · n_query), so both the KDE source and the
    displayed points are independently subsampled to stay tractable. Returns
    ``(xs_d, ys_d, z, n_src)`` where ``xs_d``/``ys_d`` are the points to draw,
    ``z`` their KDE density, and ``n_src`` is the number of source events the
    kernel was fit on (``< len(xs)`` ⇒ the source was capped). The GUI keeps
    the status message and the matplotlib draw.

    Subsampling stays on the CPU (identical points regardless of backend); only
    the kernel evaluation moves to the GPU when acceleration is enabled, else
    the SciPy path runs — so the default is unchanged.
    """
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

    from . import gpu_accel
    if gpu_accel.enabled():
        z = _kde_eval_gpu(xs_src, ys_src, xs_d, ys_d)
        if z is not None:
            return xs_d, ys_d, z, int(xs_src.size)

    from scipy.stats import gaussian_kde
    kernel = gaussian_kde(np.vstack([xs_src, ys_src]))
    z = kernel(np.vstack([xs_d, ys_d]))
    return xs_d, ys_d, z, int(xs_src.size)
