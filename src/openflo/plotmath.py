"""Pure plotting/maths helpers lifted out of the GUI.

Numeric helpers the editor uses to lay out plots — bin edges, symlog linear
threshold, density colour norm, ellipsoid-gate geometry, point-in-box, and a
tree-row drop-count suffix. No Tkinter and no editor state, so they're unit-
testable against arrays directly instead of through a window.
"""
from __future__ import annotations

import numpy as np


def drop_suffix(drop, total) -> str:
    """``'  —  drops N (X%)'`` for a tree row, or ``''`` when nothing is
    dropped / the total is unknown."""
    if not total or drop is None:
        return ''
    return f'  —  drops {drop:,} ({100.0 * drop / total:.1f}%)'


def symlog_linthresh(data_sample) -> float:
    """Linear-region half-width for a native symlog axis: the 5th percentile of
    ``|nonzero data|``, floored at 1e-6. The same value feeds the display axis
    and the density binning so the two stay aligned. Defaults to 1.0."""
    linthresh = 1.0
    if data_sample is not None:
        arr = np.asarray(data_sample, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 50:
            nz = np.abs(arr[arr != 0])
            if nz.size > 0:
                linthresh = max(float(np.percentile(nz, 5)), 1e-6)
    return linthresh


def density_norm(z):
    """A ``PowerNorm`` (gamma 0.4) spreading the colour map across the
    populated density range, so the dense core doesn't wash structure flat on
    large samples."""
    from matplotlib.colors import PowerNorm
    zmax = float(np.max(z)) if len(z) else 1.0
    return PowerNorm(gamma=0.4, vmin=0.0, vmax=max(zmax, 1e-9))


def in_box(fr, box) -> bool:
    """True if point ``fr=(x, y)`` lies within ``box=(x0, y0, x1, y1)``."""
    if fr is None or box is None:
        return False
    x0, y0, x1, y1 = box
    return x0 <= fr[0] <= x1 and y0 <= fr[1] <= y1


def hist_bin_edges(lo, hi, scale, n_bins=200) -> list:
    """``n_bins + 1`` edges between ``lo`` and ``hi``, linear or log-spaced by
    axis scale.

    - ``'linear'`` / ``'symlog'`` → linear spacing (symlog's transform is
      linear near zero and only compresses the tails).
    - ``'log'`` → log-spaced; ``lo`` is clamped to a small positive when
      non-positive, falling back to linear if the clamped range is degenerate.

    Returns a Python list (matplotlib's hist stubs type bins as
    ``Sequence[float]``).
    """
    lo = float(lo)
    hi = float(hi)
    n_bins = int(n_bins)
    if scale == 'log':
        lo_pos = max(lo, max(hi * 1e-6, 1e-12))
        if hi <= lo_pos:
            return np.linspace(lo, hi, n_bins + 1).tolist()
        return np.logspace(np.log10(lo_pos), np.log10(hi), n_bins + 1).tolist()
    return np.linspace(lo, hi, n_bins + 1).tolist()


def ellipse_params(gate):
    """``(cx, cy, width, height, angle_deg)`` for matplotlib's ``Ellipse`` from
    an ellipsoid gate's ``(mean, cov, distance_sq)``; the boundary is the level
    set ``(p-µ)ᵀ Σ⁻¹ (p-µ) = distance_sq``. Returns ``None`` if degenerate."""
    try:
        mean = np.asarray(gate['mean'], dtype=float)
        cov = np.asarray(gate['cov'], dtype=float)
        dist_sq = float(gate.get('distance_sq', 4.0))
        if mean.shape != (2,) or cov.shape != (2, 2):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)      # symmetric → real eigenpairs
        if np.any(eigvals <= 0) or dist_sq <= 0:
            return None
        semis = np.sqrt(eigvals * dist_sq)          # full axis length = 2·semi
        width = 2.0 * float(semis[0])
        height = 2.0 * float(semis[1])
        v = eigvecs[:, 0]                           # angle of width's axis
        angle = float(np.degrees(np.arctan2(v[1], v[0])))
        return float(mean[0]), float(mean[1]), width, height, angle
    except Exception:
        return None


def ellipse_geom(gate):
    """Geometry an ellipsoid gate needs for hit-testing / editing:
    ``((mean_x, mean_y), Σ⁻¹, r0, (handle_x, handle_y))`` where ``r0 =
    sqrt(distance_sq)`` is the Mahalanobis rim radius and the handle sits just
    beyond the rim along the +height axis (the rotation grip). ``None`` if
    degenerate."""
    try:
        mean = np.asarray(gate['mean'], dtype=float)
        cov = np.asarray(gate['cov'], dtype=float)
        dist_sq = float(gate.get('distance_sq', 4.0))
        if mean.shape != (2,) or cov.shape != (2, 2) or dist_sq <= 0:
            return None
        inv = np.linalg.inv(cov)
        r0 = float(np.sqrt(dist_sq))
        eigvals, eigvecs = np.linalg.eigh(cov)
        if np.any(eigvals <= 0):
            return None
        v = eigvecs[:, 1]                            # +height axis
        semi_h = float(np.sqrt(eigvals[1] * dist_sq))
        hx = float(mean[0] + v[0] * semi_h * 1.18)   # handle clears the rim
        hy = float(mean[1] + v[1] * semi_h * 1.18)
        return (float(mean[0]), float(mean[1])), inv, r0, (hx, hy)
    except Exception:
        return None


def point_segment_dist(px, py, ax, ay, bx, by, span_x, span_y) -> float:
    """Axis-fraction distance from point ``(px, py)`` to segment
    ``(ax, ay)-(bx, by)``. Both axes are normalised by their view span so the
    distance is dimensionless (comparable to the hit-test tolerance)."""
    sx, sy = max(span_x, 1e-9), max(span_y, 1e-9)
    pxn, pyn = px / sx, py / sy
    axn, ayn = ax / sx, ay / sy
    bxn, byn = bx / sx, by / sy
    dx, dy = bxn - axn, byn - ayn
    seg2 = dx * dx + dy * dy
    if seg2 < 1e-18:
        ex, ey = pxn - axn, pyn - ayn
        return (ex * ex + ey * ey) ** 0.5
    t = ((pxn - axn) * dx + (pyn - ayn) * dy) / seg2
    t = max(0.0, min(1.0, t))
    qx, qy = axn + t * dx, ayn + t * dy
    ex, ey = pxn - qx, pyn - qy
    return (ex * ex + ey * ey) ** 0.5


def gid_from_hit(hit):
    """Extract the gate id from a hit tuple, or ``None`` when not gate-bound.
    Threshold/interval lines pack the id as ``'gid'`` or ``'gid:lo' / 'gid:hi'``;
    other shapes use the bare id."""
    if not hit or len(hit) < 2:
        return None
    second = hit[1]
    if not isinstance(second, str):
        return None
    return second.split(':', 1)[0] if ':' in second else second
