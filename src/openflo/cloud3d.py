"""The numpy core of the 3D scatter: the seeded subset of events drawn, the
cube the points sit in, and how crowded each point's neighbourhood is.

The desktop's 3D plot (``view3d``) and the phone's (``Document.cloud3d``)
both use it, so the two pick the same events from the same seed and color
them by the same density. numpy only (no SciPy, no matplotlib), as the
phone's Pyodide runtime ships neither.

Core module: no Tk.
"""
from __future__ import annotations

import math

import numpy as np

DENSITY_BINS = 64
DENSITY_SIGMA = 1.0
# How far the smoothing kernel reaches, in sigmas: SciPy's gaussian_filter
# default, so this smoothing is that filter's.
DENSITY_TRUNCATE = 4.0
# The 2D plot's color spread: (z / max z) ** 0.4 (plotmath.density_norm).
DENSITY_GAMMA = 0.4


def seeded_subset(n: int, cap: int | None, seed: int) -> np.ndarray | None:
    """Positions (sorted) of at most ``cap`` of ``n`` events, picked at
    random with ``seed``: the same seed always picks the same events. None
    when every event fits."""
    if cap is None or n <= cap:
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, int(cap), replace=False))


def normalize(screen: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Screen values -> -1 .. 1 across ``bounds`` (outside it: beyond ±1)."""
    lo, hi = bounds
    return (np.asarray(screen, dtype=float) - lo) / (hi - lo) * 2.0 - 1.0


def gaussian_kernel(sigma: float = DENSITY_SIGMA,
                    truncate: float = DENSITY_TRUNCATE) -> np.ndarray:
    """The 1D weights SciPy's gaussian_filter uses for ``sigma``: out to
    ``int(truncate * sigma + 0.5)`` bins each side, summing to 1."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return w / w.sum()


def smooth3d(grid: np.ndarray, sigma: float = DENSITY_SIGMA) -> np.ndarray:
    """``grid`` blurred by a Gaussian of ``sigma`` bins along each axis in
    turn, with nothing outside the grid (SciPy's ``mode='constant'``)."""
    out = np.asarray(grid, dtype=float)
    w = gaussian_kernel(sigma)
    r = len(w) // 2
    for axis in range(out.ndim):
        n = out.shape[axis]
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        padded = np.pad(out, pad)
        acc = np.zeros_like(out)
        for k, wk in enumerate(w):
            acc += wk * np.take(padded, np.arange(k, k + n), axis=axis)
        out = acc
    return out


def density_values(points: np.ndarray, bins: int = DENSITY_BINS) -> np.ndarray:
    """How crowded each point's neighbourhood is in 3D: a ``bins``³
    histogram of the -1..1 cube, smoothed by one bin (``smooth3d``), read
    back at each point's bin. Points outside the cube count in its edge
    bins."""
    p = np.asarray(points, dtype=float)
    if len(p) == 0:
        return np.zeros(0)
    idx = np.clip(np.floor((p + 1.0) / 2.0 * bins), 0, bins - 1).astype(np.int64)
    flat = (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]
    hist = np.bincount(flat, minlength=bins ** 3).astype(float)
    return smooth3d(hist.reshape(bins, bins, bins)).ravel()[flat]


def density_shade(z: np.ndarray) -> np.ndarray:
    """Densities -> 0..1 color positions with the 2D plot's spread,
    ``(z / max z) ** 0.4``, so a dense core does not wash out the rest."""
    v = np.clip(np.asarray(z, dtype=float), 0.0, None)
    top = float(v.max()) if v.size else 0.0
    if not math.isfinite(top) or top <= 0:
        return np.zeros_like(v)
    return (v / top) ** DENSITY_GAMMA
