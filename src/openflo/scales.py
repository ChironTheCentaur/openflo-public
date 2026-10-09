"""Axis display-scale view transforms — Tk-free, testable.

Fluor data is stored baked into a nonlinear transform (e.g. logicle). The
underlying *linear intensity* is the canonical master; the chosen display scale
(linear / log / symlog) is a pure VIEW of that intensity, composed as::

    forward(d) = view_forward(inverse_baked(d))
    inverse(p) = forward_baked(view_inverse(p))

so every scale is an independent, equation-derived view of the same intensity
(no double-transform) and gates kept in stored-data coords auto-follow the axis.
This is the maths behind the matplotlib ``FuncScale`` the editor installs.

The axis of such a view is labelled in intensity too (``intensity_ticker``):
matplotlib's own ticks on a ``FuncScale`` are in the STORED coordinate, so a
logicle axis read 0.0 .. 1.2, where '0.6' was an intensity of ~4,250. The
same holds for an axis that IS the stored coordinate (a colour bar, a CLI
scatter PNG: ``scale='stored'``), and for text that names a value
(``format_intensity_text``: a gate bound, a slider).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

# The log view cannot place an intensity <= 0: `forward` clips intensity to
# this value, so every such event (a third of a compensated channel, often) is
# drawn at ONE screen position, log10 of it -- the view's floor.
LOG_VIEW_MIN = 1e-6

# An open gate side, as WspReader and the quadrant tool write it (JSON cannot
# carry +-inf).
OPEN = 1e12


def view_funcs(transform: str, scale: str, data_sample=None):
    """``(forward, inverse)`` callables mapping a channel's STORED data
    coordinate to screen position for ``scale`` — or ``None`` when the channel
    is stored linearly (``transform == 'linear'``), in which case the caller
    uses matplotlib's native scale (nicer tick locators).

    ``symlog`` uses an arcsinh view whose cofactor is anchored on the data's 5th
    percentile of ``|nonzero intensity|`` when ``data_sample`` is given.
    """
    from .pipeline import inverse_transform_values, transform_values
    if transform == 'linear':
        return None

    def inv_baked(a):
        return inverse_transform_values(np.asarray(a, dtype=float),
                                        method=transform)

    def fwd_baked(a):
        return transform_values(np.asarray(a, dtype=float), method=transform)

    if scale == 'log':
        def forward(d):  # pyright: ignore[reportRedeclaration]
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.log10(np.clip(inv_baked(d), LOG_VIEW_MIN, None))

        def inverse(p):
            return fwd_baked(np.power(10.0, np.asarray(p, dtype=float)))
    elif scale == 'symlog':
        cof = _symlog_cofactor(inv_baked, data_sample)

        def forward(d):  # pyright: ignore[reportRedeclaration]
            return np.arcsinh(inv_baked(d) / cof)

        def inverse(p):
            return fwd_baked(np.sinh(np.asarray(p, dtype=float)) * cof)
    else:  # 'linear' view of nonlinear-baked data → stretch to intensity
        def forward(d):
            return inv_baked(d)

        def inverse(p):
            return fwd_baked(p)

    def _finite(fn):
        # matplotlib's FuncScale needs shape-preserving callables, but FlowKit's
        # (inverse_)logicle flattens to 1-D — reshape back and scrub NaNs.
        #
        # NOTE the consequence, which matters more since transform_values began
        # returning NaN for non-finite input: a NaN reaching here is mapped to
        # screen position 0.0 — the axis ORIGIN — not dropped. Events never get
        # that far (`_get_df` drops non-finite rows on the plotted channels
        # before drawing), so this only ever sees axis limits, ticks and gate
        # bounds, which are finite. If that upstream drop is ever removed, this
        # scrub silently becomes "plot bad data at the origin" and must change
        # with it.
        def wrapped(a):
            arr = np.asarray(a, dtype=float)
            out = np.nan_to_num(np.asarray(fn(arr), dtype=float), nan=0.0)
            return out.reshape(arr.shape)
        return wrapped

    return _finite(forward), _finite(inverse)


def _symlog_cofactor(inv_baked, data_sample):
    cof = 150.0
    if data_sample is not None:
        lin = inv_baked(np.asarray(data_sample, dtype=float))
        lin = lin[np.isfinite(lin)]
        nz = np.abs(lin[lin != 0])
        if nz.size > 50:
            cof = max(float(np.percentile(nz, 5)), 1e-3)
    return cof


def view_floor(transform: str, scale: str):
    """Screen position (the view's own units) at which ``view_funcs`` draws
    every event it cannot place, or None when the view places every value:
    log10(LOG_VIEW_MIN) for the log view of a transformed channel, where
    every intensity <= LOG_VIEW_MIN lands. A shape drawn down to it holds
    those events on screen, whatever their stored value."""
    if transform == 'linear' or scale != 'log':
        return None
    return float(np.log10(LOG_VIEW_MIN))


# ── Tick labels in intensity ─────────────────────────────────────────────────

def format_intensity(v) -> str:
    """Short axis label for a linear intensity: '0', '10', '100', powers of
    ten from 1,000 up (and below 1) as 10^k, other values from 1,000 as 'K'
    ('2.5K', '50K'). Rounded to 6 significant digits first, so a tick that
    went through transform and inverse (999.9999996) reads as what it is."""
    v = float(v)
    if not np.isfinite(v):
        return ''
    r = float(f'{v:.6g}')
    if r != 0:
        k = math.log10(abs(r))
        if abs(k - round(k)) < 1e-9 and (abs(r) >= 1000 or abs(r) < 1):
            return f"${'-' if r < 0 else ''}10^{{{int(round(k))}}}$"
    return format_intensity_text(r, digits=6)


def format_intensity_text(v, digits: int = 3) -> str:
    """A linear intensity in plain ASCII, for text rather than an axis: a
    gate bound in the gate list, a population name, a slider. The tick
    labels' rules (``format_intensity``) without their mathtext, so a bound
    reads like the axis it was drawn on: '0', '-736', '45.2', '1.23K',
    '50K'. `digits` significant digits; 'nan' / 'inf' for a non-finite
    value. No mathtext: the text lands in Tk labels, CSV cells and
    matplotlib legends, where '$...$' would show or be parsed."""
    v = float(v)
    if not np.isfinite(v):
        return f'{v:g}'
    r = float(f'{v:.{int(digits)}g}')
    if r == 0:
        return '0'
    if abs(r) >= 1000:
        return f'{r / 1000:g}K'
    return f'{r:g}'


def _transform_kwargs(transform) -> dict[str, Any]:
    """``inverse_transform_values`` / ``transform_values`` keywords for
    `transform`: a method name (default parameters, as the editor's axes
    use) or a ``transform_spec`` dict (a sample's ``data_transforms``
    entry, parameters included)."""
    if isinstance(transform, Mapping):
        return dict(transform)
    return {'method': transform or 'linear'}


def _no_intensity(transform) -> bool:
    """True when `transform` gives no intensity to tick by: linear-stored
    (matplotlib ticks it) or a scale that is unknown (nothing linear can be
    said, see pipeline.UNKNOWN_SPEC)."""
    return _transform_kwargs(transform).get('method', 'linear') in (
        'linear', 'unknown')


def _spaced_from_zero(cands, scr, gap):
    """The candidates (round intensities) kept, from 0 outward, so that no
    two sit closer than `gap` in screen units `scr`: near zero a compressed
    view puts 10 and 100 on top of 0."""
    order = np.argsort(np.abs(cands), kind='stable')
    kept: list[int] = []
    for i in order:
        if all(abs(scr[i] - scr[j]) >= gap for j in kept):
            kept.append(int(i))
    return np.sort(cands[kept])


def intensity_ticks(transform, scale: str, lo, hi, data_sample=None,
                    max_ticks: int = 8):
    """``(positions, intensities)`` for the axis of a ``transform``-stored
    channel shown with ``scale`` over the stored view ``[lo, hi]``: positions
    in stored units at round linear intensities -- decades on a log view,
    0 and +-decades on symlog (arcsinh), 1-2-2.5-5 steps on a linear view.
    ``scale='stored'`` is an axis drawn in the stored coordinate itself (a
    colour bar, a CLI scatter): 0 and +-decades, spaced on that axis.
    `transform` is a method name or a ``transform_spec`` dict. Both empty
    for a linear-stored channel (matplotlib ticks it) or an unknown one."""
    empty = (np.array([]), np.array([]))
    if _no_intensity(transform):
        return empty
    from .pipeline import inverse_transform_values, transform_values
    kw = _transform_kwargs(transform)
    lo, hi = sorted((float(lo), float(hi)))
    a, b = (float(x) for x in inverse_transform_values(
        np.array([lo, hi]), **kw))
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return empty
    if scale == 'log':
        a = max(a, LOG_VIEW_MIN)
        if b <= a:
            return empty
        ks = np.arange(math.ceil(math.log10(a) - 1e-9),
                       math.floor(math.log10(b) + 1e-9) + 1)
        step = max(1, math.ceil(len(ks) / max_ticks))
        # Multiples of the step, so the ticks stay put while panning.
        vals = 10.0 ** ks[ks % step == 0]
    elif scale == 'symlog':
        cands = np.array([0.0] + [s * 10.0 ** k for k in range(1, 9)
                                  for s in (1.0, -1.0)])
        cands = np.sort(cands[(cands >= a) & (cands <= b)])
        if cands.size == 0:
            return empty
        # Near zero the arcsinh view is linear, so 10 and 100 can sit on top
        # of 0: keep ticks at least a fraction of the axis apart, from 0 out.
        def inv_baked(x):
            return inverse_transform_values(np.asarray(x, dtype=float), **kw)
        cof = _symlog_cofactor(inv_baked, data_sample)
        scr = np.arcsinh(cands / cof)
        gap = (np.arcsinh(b / cof) - np.arcsinh(a / cof)) / (max_ticks + 1)
        vals = _spaced_from_zero(cands, scr, gap)
    elif scale == 'stored':
        # The axis is the stored coordinate: logicle (or asinh) is itself a
        # log-like view, so decades read evenly above ~100 and crowd 0 below
        # it. Spaced in stored units, from 0 out, as on the symlog view.
        cands = np.array([0.0] + [s * 10.0 ** k for k in range(-3, 9)
                                  for s in (1.0, -1.0)])
        cands = cands[(cands >= a) & (cands <= b)]
        if kw.get('method') == 'log':
            cands = cands[cands > 0]      # no position: log10 of <= 0
        if cands.size == 0:
            return empty
        scr = np.asarray(transform_values(cands, **kw), dtype=float)
        ok = np.isfinite(scr)
        vals = _spaced_from_zero(cands[ok], scr[ok],
                                 (hi - lo) / (max_ticks + 1))
    else:
        from matplotlib.ticker import MaxNLocator
        vals = np.asarray(MaxNLocator(nbins=max(max_ticks - 2, 2),
                                      steps=[1, 2, 2.5, 5, 10]).tick_values(a, b),
                          dtype=float)
        vals = vals[(vals >= a) & (vals <= b)]
    pos = np.asarray(transform_values(np.asarray(vals, dtype=float), **kw),
                     dtype=float)
    tol = 1e-9 * max(1.0, abs(hi - lo))
    keep = np.isfinite(pos) & (pos >= lo - tol) & (pos <= hi + tol)
    return pos[keep], np.asarray(vals, dtype=float)[keep]


def intensity_ticker(transform, scale: str, data_sample=None):
    """``(locator, formatter)`` for a matplotlib axis showing a
    ``transform``-stored channel with ``scale`` (see ``view_funcs``; or
    'stored' for an axis in the stored coordinate itself, see
    ``intensity_ticks``), so its ticks sit at round intensities and say
    them; None for a linear-stored channel or one of unknown scale.
    `transform` is a method name or a ``transform_spec`` dict. The locator
    follows the current view, so a zoom or pan re-ticks without a replot."""
    if _no_intensity(transform):
        return None
    from matplotlib.ticker import Formatter, Locator

    from .pipeline import inverse_transform_values
    kw = _transform_kwargs(transform)

    def intensity(x):
        return float(inverse_transform_values(np.array([float(x)]), **kw)[0])

    class IntensityLocator(Locator):
        def __call__(self):
            lo, hi = self.axis.get_view_interval()   # type: ignore[union-attr]
            return self.tick_values(lo, hi)

        def tick_values(self, vmin, vmax):
            return intensity_ticks(transform, scale, vmin, vmax,
                                   data_sample)[0]

    class IntensityFormatter(Formatter):
        def __call__(self, x, pos=None):
            return format_intensity(intensity(x))

        def format_data_short(self, value):
            # The cursor readout: plain digits, not the tick's mathtext.
            v = intensity(value)
            return f'{v:.4g}' if np.isfinite(v) else ''

    return IntensityLocator(), IntensityFormatter()


# ── Shapes drawn to the end of an axis ───────────────────────────────────────

def axis_ends(fwd, inv, bounds, extent_px, pile=None, tol_px=2.0):
    """One axis of a drawing, for ``open_rect_edges`` /
    ``open_polygon_edges``: ``(fwd, inv, floor, ceiling, tol)`` in the axis'
    screen units. `fwd` / `inv` map data <-> screen units (the axis
    transform, before the affine part); `bounds` are the axes limits in data;
    `extent_px` the axis length in pixels; `pile` the view's floor
    (``view_floor``) when it has one. The floor is the higher of the axes
    bottom and the pile: nothing is drawn below it. (On the log view of a
    logicle channel the two coincide: no stored value maps below the pile,
    so the axes cannot reach below it, and the pile sits on the axis line.)
    `tol` is `tol_px` in screen units: 2 px, about a marker's width, so an
    edge drawn on the markers piled at the floor counts as at it."""
    s0, s1 = (float(v) for v in np.asarray(fwd(np.asarray(sorted(bounds),
                                                          dtype=float))))
    floor = s0 if pile is None else min(max(s0, float(pile)), s1)
    tol = tol_px * (s1 - s0) / max(float(extent_px), 1.0)
    return fwd, inv, floor, s1, tol


def open_rect_edges(lo, hi, ends):
    """``(lo, hi)`` of a rectangle drawn on one axis, the lower edge the open
    sentinel when it is at or below the axis floor, the upper one when it is
    at or above the top of the axes. ``ends`` from ``axis_ends``."""
    fwd, _inv, floor, ceil, tol = ends
    s_lo, s_hi = (float(v) for v in np.asarray(fwd(np.array([lo, hi],
                                                            dtype=float))))
    return (-OPEN if s_lo <= floor + tol else lo,
            OPEN if s_hi >= ceil - tol else hi)


def open_polygon_edges(verts, ends_xy):
    """Vertices (data units) of a polygon drawn on screen, opened where it
    reaches an end of an axis: the part beyond the floor (or the top) is cut
    off at it, as the screen shows it, and each run of vertices on the cut is
    extended straight out to the open sentinel, so the region beyond that
    end between the cut points is inside. Unchanged (the same list) when no
    vertex reaches an end. ``ends_xy`` = (x, y) from ``axis_ends``."""
    from .pipeline import _open_polygon_at
    data = np.asarray(verts, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 3:
        return verts
    fwds = [e[0] for e in ends_xy]
    invs = [e[1] for e in ends_xy]

    def to_screen(d):
        return np.column_stack([np.asarray(fwds[i](d[:, i]), dtype=float)
                                for i in (0, 1)])

    scr = to_screen(data)
    changed = False
    for axis, (_f, inv, floor, ceil, tol) in enumerate(ends_xy):
        s = scr[:, axis]
        if not ((s <= floor + tol).any() or (s >= ceil - tol).any()):
            continue
        changed = True
        scr, data = _clip_polygon_carry(scr, data, axis, floor, ceil, invs)
        if len(scr) < 3:
            return verts
        for bound, near in ((floor, scr[:, axis] <= floor + tol),
                            (ceil, scr[:, axis] >= ceil - tol)):
            scr[near, axis] = bound
            data[near, axis] = float(np.asarray(inv(np.array([bound])))[0])
        for bound, far in ((floor, -OPEN), (ceil, OPEN)):
            scr, data = _open_polygon_at(scr, data, axis, bound, far)
    if not changed:
        return verts
    return data.tolist()


def _clip_polygon_carry(scr, data, axis, lo, hi, invs):
    """Polygon cut to lo <= screen coordinate `axis` <= hi (Sutherland-
    Hodgman); `data` rides along, a cut point's data from `invs` (both axes:
    the edge is straight on screen)."""
    def cut(S, D, keep, bound):
        outS, outD = [], []
        for k in range(len(S)):
            ps, qs = S[k - 1], S[k]
            if keep(qs) != keep(ps):
                r = ps + (bound - ps[axis]) / (qs[axis] - ps[axis]) * (qs - ps)
                r[axis] = bound
                outS.append(r)
                outD.append(np.array([float(np.asarray(invs[i](
                    np.array([r[i]])))[0]) for i in (0, 1)]))
            if keep(qs):
                outS.append(qs)
                outD.append(D[k])
        return outS, outD

    S, D = list(np.asarray(scr, dtype=float)), list(np.asarray(data, dtype=float))
    S, D = cut(S, D, lambda p: p[axis] >= lo, lo)
    if S:
        S, D = cut(S, D, lambda p: p[axis] <= hi, hi)
    return (np.asarray(S, dtype=float).reshape(-1, 2),
            np.asarray(D, dtype=float).reshape(-1, 2))
