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
    if not total or not drop:          # None OR 0 dropped → no suffix
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
    - ``'log'`` → log-spaced. A log axis cannot place a non-positive value, so
      when ``lo`` is non-positive the range starts at a synthetic floor six
      decades below ``hi``. A POSITIVE ``lo`` is used as it stands.

    Returns a Python list (matplotlib's hist stubs type bins as
    ``Sequence[float]``).
    """
    lo = float(lo)
    hi = float(hi)
    n_bins = int(n_bins)
    if scale == 'log':
        # The floor used to be `max(lo, max(hi * 1e-6, 1e-12))`, which raised a
        # perfectly good positive `lo` up to six decades below `hi`. On data
        # with a wide dynamic range that discarded events the axis could
        # display perfectly well: measured, an all-positive channel spanning
        # 1e-3 to 1e6 had its floor moved from 0.001 to 1.0, putting HALF the
        # events off the bottom of the histogram. The clamp is only needed
        # when there is no positive lower bound to use.
        lo_pos = lo if lo > 0 else max(hi * 1e-6, 1e-12)
        if hi <= lo_pos:
            return np.linspace(lo, hi, n_bins + 1).tolist()
        edges = np.logspace(np.log10(lo_pos), np.log10(hi), n_bins + 1)
        # The round trip through log10 can land the outer edges an ulp inside
        # the requested range, which drops the single lowest (or highest)
        # event for no reason a reader could ever guess. Pin them.
        edges[0] = min(float(edges[0]), lo_pos)
        edges[-1] = max(float(edges[-1]), hi)
        return edges.tolist()
    return np.linspace(lo, hi, n_bins + 1).tolist()


def unplottable_count(values, scale) -> int:
    """How many finite ``values`` the SCALE itself cannot represent.

    A log axis cannot place a non-positive value at all — on a compensated
    channel that is routinely a third of the sample, and the curve looks
    exactly the same whether it omitted none of them or all of them. Every
    other scale here can represent any finite value.

    Deliberately NOT the same as "outside the plotted range". The view range is
    a robust percentile by design, so a histogram always clips a fraction of a
    percent at each tail; reporting that would fire on every plot and mean
    nothing. This counts only what is categorically absent.

    Non-finite values are not counted — they are missing for a different reason
    and are reported elsewhere.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or scale != 'log':
        return 0
    return int((v <= 0).sum())


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


# ── Shapes on non-linear axes ───────────────────────────────────────────────
#
# gate_to_mask joins a polygon's vertices with straight lines in DATA space;
# matplotlib draws them with straight lines on SCREEN. On a linear axis the two
# coincide. On a log (or symlog / logicle-view) axis they do not: a 2-decade
# triangle drawn on screen selected 46,226 events where 24,898 lie inside it.
# The helpers below place enough vertices that the two readings agree to a
# fraction of a pixel, so no gate's evaluation has to change.

def trace_on_screen(curve, t, to_screen, tol_px=0.25, window=None,
                    max_rounds=12):
    """Data points along ``curve`` dense enough that, between consecutive
    points, the screen chord matplotlib draws stays within ``tol_px`` of both
    the curve and the data-space chord gate_to_mask tests.

    ``curve`` maps an array of parameters to ``(n, 2)`` data points;
    ``t`` is the initial (sorted) parameter grid, whose points are kept;
    ``to_screen`` maps ``(n, 2)`` data to display pixels (an axes' transData).
    Each round splits, at its mid-parameter, every interval where the curve's
    midpoint or the data chord's midpoint lies more than ``tol_px`` off the
    screen chord. Distances are taken across the chord, not along it: on a
    log axis a straight horizontal edge is still straight, only parameterised
    unevenly, and must not be split. An interval with an end outside
    ``window`` ``(x0, x1, y0, y1)`` in pixels, or that does not map to a
    finite pixel (a non-positive value on a log axis), is left alone: it is
    off screen, and refining toward a clipped coordinate never converges.
    """
    def off_chord(p, a, b):
        ab, ap = b - a, p - a
        length = np.hypot(ab[:, 0], ab[:, 1])
        cross = np.abs(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0])
        return np.where(length > 1e-12, cross / np.maximum(length, 1e-12),
                        np.hypot(ap[:, 0], ap[:, 1]))

    t = np.asarray(t, dtype=float)
    for _ in range(max_rounds):
        with np.errstate(invalid='ignore', over='ignore'):
            pts = curve(t)
            disp = to_screen(pts)
            tm = 0.5 * (t[:-1] + t[1:])
            a, b = disp[:-1], disp[1:]
            dev = np.maximum(
                off_chord(to_screen(curve(tm)), a, b),
                off_chord(to_screen(0.5 * (pts[:-1] + pts[1:])), a, b))
            ok = np.isfinite(disp).all(axis=1)
            if window is not None:
                x0, x1, y0, y1 = window
                ok &= ((disp[:, 0] >= x0) & (disp[:, 0] <= x1)
                       & (disp[:, 1] >= y0) & (disp[:, 1] <= y1))
            need = ok[:-1] & ok[1:] & np.isfinite(dev) & (dev > tol_px)
        if not need.any():
            break
        t = np.sort(np.concatenate([t, tm[need]]))
    return curve(t)


def polygon_on_screen(verts, to_screen, from_screen, tol_px=0.25):
    """``(n, 2)`` data vertices of the closed polygon whose edges run straight
    ON SCREEN through ``verts`` -- the shape a polygon / lasso selector
    previews -- with points added along each edge until the data-space
    polygon matches it (see trace_on_screen). The given vertices are kept
    exactly; on a linear axis nothing is added."""
    verts = np.asarray(verts, dtype=float)
    return polyline_on_screen(np.vstack([verts, verts[:1]]), to_screen,
                              from_screen, tol_px)[:-1]


def polyline_on_screen(verts, to_screen, from_screen, tol_px=0.25):
    """polygon_on_screen for an OPEN chain: ``(n, 2)`` data points from
    ``verts[0]`` to ``verts[-1]`` whose segments run straight on screen
    through ``verts``. The given vertices, both ends included, are kept
    exactly."""
    verts = np.asarray(verts, dtype=float)
    m = len(verts) - 1
    disp = to_screen(verts)

    def curve(t):
        k = np.clip(np.floor(t).astype(int), 0, m - 1)
        f = (t - k)[:, None]
        out = from_screen(disp[k] + f * (disp[k + 1] - disp[k]))
        at_vertex = f[:, 0] == 0
        out[at_vertex] = verts[k[at_vertex]]
        out[t >= m] = verts[m]
        return out

    return trace_on_screen(curve, np.arange(m + 1.0), to_screen, tol_px)


def screen_corners(verts, to_screen, tol_px=1e-6):
    """Boolean mask of the vertices of the closed polygon ``verts`` that are
    corners ON SCREEN: not between their two neighbours on the straight
    screen line through them, to ``tol_px``. The points polygon_on_screen
    adds along an edge lie on that line to ~1e-13 px (~1e-10 px zoomed in
    1000x, measured on log, symlog and logicle views); a drawn corner is
    pixels off it. A vertex with no finite pixel counts as a corner.

    polygon_on_screen adds points only along an edge that is bent in data.
    A run of points on the line between two corners, along an edge that is
    straight in data as well (axis-parallel on log axes, say), was placed by
    the user, and each of them is a corner too. A user's point exactly on
    the screen line of a bent edge cannot be told from an added one."""
    verts = np.asarray(verts, dtype=float)
    d = to_screen(verts)
    a, b = np.roll(d, 1, axis=0), np.roll(d, -1, axis=0)
    ab, ap = b - a, d - a
    with np.errstate(invalid='ignore', divide='ignore'):
        length2 = ab[:, 0] ** 2 + ab[:, 1] ** 2
        off = np.abs(ab[:, 0] * ap[:, 1] - ab[:, 1] * ap[:, 0]) / np.sqrt(length2)
        s = (ab[:, 0] * ap[:, 0] + ab[:, 1] * ap[:, 1]) / length2
        corner = ~((off <= tol_px) & (s > 0) & (s < 1))
    n = len(verts)
    at = np.flatnonzero(corner)
    if len(at) < 2:
        return corner
    users = [(p + k) % n
             for p, q in zip(at, np.roll(at, -1), strict=True)
             if not _bent_in_data(verts[p], verts[q], to_screen, tol_px)
             for k in range(1, (q - p) % n)]
    corner[users] = True
    return corner


def _bent_in_data(p, q, to_screen, tol_px):
    """Whether the data-space chord from ``p`` to ``q`` leaves their straight
    screen line by more than ``tol_px`` (sampled at seven points)."""
    u = np.linspace(0.0, 1.0, 9)[1:-1, None]
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        (a, b), m = to_screen(np.array([p, q])), to_screen(p + u * (q - p))
        ab, am = b - a, m - a
        off = (np.abs(ab[0] * am[:, 1] - ab[1] * am[:, 0])
               / np.hypot(ab[0], ab[1]))
    return not bool(np.all(off <= tol_px))


def reroute_on_screen(verts, i, point, to_screen, from_screen, tol_px=0.25,
                      on_edge=False):
    """The closed polygon ``verts`` with vertex ``i`` moved to ``point``, or
    removed when ``point`` is None, its edges kept straight ON SCREEN. With
    ``on_edge``, ``point`` is instead a NEW vertex on the edge from stored
    point ``i`` to ``i + 1``.

    A polygon drawn on a non-linear axis is stored as its corners plus points
    along each screen-straight edge (polygon_on_screen). Moving one stored
    point alone left a notch, and the edges next to it, straight in data,
    bent on screen. Here the run of points between the screen corners around
    ``i`` (screen_corners, found before anything moves) is replaced by points
    along the new screen edges -- what drawing the new shape would store.
    Every other point is kept exactly. ``i`` itself counts as a corner, so a
    point taken from along an edge becomes one. Returns None when there are
    too few corners: fewer than 3 with ``i``, or fewer than 3 left after
    removing it, or fewer than 2 to put a new vertex between."""
    verts = np.asarray(verts, dtype=float)
    n = len(verts)
    corner = screen_corners(verts, to_screen)
    if on_edge:
        if int(corner.sum()) < 2:
            return None
        p = next((i - k) % n for k in range(n) if corner[(i - k) % n])
        q = next((i + 1 + k) % n for k in range(n) if corner[(i + 1 + k) % n])
    else:
        corner[i] = False
        # Two other corners to re-route between; three to remain a polygon
        # once ``i`` is removed.
        others = int(corner.sum())
        if others < 2 or (point is None and others < 3):
            return None
        p = next((i - k) % n for k in range(1, n) if corner[(i - k) % n])
        q = next((i + k) % n for k in range(1, n) if corner[(i + k) % n])
    chain = [verts[p], verts[q]] if point is None else [verts[p], point, verts[q]]
    inner = polyline_on_screen(chain, to_screen, from_screen, tol_px)[1:-1]
    run = {(p + k) % n for k in range(1, (q - p) % n)}
    out = []
    for j in range(n):
        if j not in run:
            out.append(verts[j])
            if j == p:
                out.extend(inner)
    return np.asarray(out, dtype=float)


def ellipse_outline(params, to_screen, tol_px=0.25, window=None,
                    max_rounds=12):
    """``(n, 2)`` data points on an ellipsoid gate's boundary (``params`` from
    ellipse_params), dense enough that straight screen segments between them
    follow the boundary's true image on any axis scale. matplotlib's Ellipse
    patch log-transforms only its bezier control points, which on a log axis
    drew a shape 16.7% different from the region the gate selects.
    ``max_rounds=0`` gives the 64 starting points unrefined."""
    cx, cy, width, height, angle = params
    th = np.radians(angle)
    c, s = np.cos(th), np.sin(th)
    a, b = 0.5 * width, 0.5 * height

    def curve(t):
        u, v = a * np.cos(t), b * np.sin(t)
        return np.column_stack([cx + c * u - s * v, cy + s * u + c * v])

    return trace_on_screen(curve, np.linspace(0.0, 2 * np.pi, 65),
                           to_screen, tol_px, window=window,
                           max_rounds=max_rounds)[:-1]


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
