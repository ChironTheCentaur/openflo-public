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


# ── Display scales with parameters ──────────────────────────────────────────
#
# A channel's display scale is one string, so every place that already saves,
# copies or restores it (sessions, tabs, layout items) carries the parameters
# along with no extra plumbing: 'logicle' / 'asinh' at their defaults, or
# 'logicle:w=1.2' / 'asinh:c=40' when the user has tuned them. The scale is a
# VIEW of the channel's linear intensity, never a re-transform of the stored
# data, so gates (kept in stored coordinates) follow it exactly.

# Logicle view defaults: the parameters the loader stores fluorescence with
# (pipeline.transform_values), so the default view IS the stored coordinate.
LOGICLE_DEFAULTS: dict[str, float] = {'t': 262144, 'm': 4.5, 'w': 0.5,
                                     'a': 0}
# The width W (decades of linear-like region around zero) the user may set.
# Logicle needs 0 <= W <= M/2.
LOGICLE_W_RANGE = (0.0, 2.0)
# The arcsinh view's default cofactor (fluorescence; mass cytometry uses ~5).
ASINH_DEFAULT_COFACTOR = 150.0
ASINH_C_RANGE = (0.1, 1e5)

# The scales a user picks from, in picker order, with their display names.
PICKABLE_SCALES = (('logicle', 'Logicle (biexponential)'),
                   ('asinh', 'Arcsinh'),
                   ('log', 'Log'),
                   ('linear', 'Linear'))


def parse_scale(scale):
    """``(kind, params)`` for a display-scale string: 'logicle:w=1.2' ->
    ('logicle', {'w': 1.2}). Parameters left out are at their defaults (not
    in `params`); a malformed or unknown parameter is dropped, never raised,
    since the string may come from a hand-edited or newer session."""
    text = str(scale or 'linear')
    kind, _, rest = text.partition(':')
    kind = kind.strip() or 'linear'
    params: dict[str, float] = {}
    allowed = {'logicle': ('w',), 'asinh': ('c',)}.get(kind, ())
    for part in rest.split(','):
        key, _, val = part.partition('=')
        key = key.strip()
        if key not in allowed:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        lo, hi = LOGICLE_W_RANGE if key == 'w' else ASINH_C_RANGE
        params[key] = min(max(v, lo), hi)
    return kind, params


def make_scale(kind, **params):
    """The display-scale string for `kind` with `params` (see parse_scale);
    a parameter at its default is left out, so the default stays the plain
    name older code compares against ('logicle')."""
    defaults = {'w': LOGICLE_DEFAULTS['w'], 'c': ASINH_DEFAULT_COFACTOR}
    parts = []
    for key in sorted(params):
        v = params[key]
        if v is None or abs(float(v) - defaults.get(key, float('nan'))) < 1e-12:
            continue
        parts.append(f'{key}={float(v):g}')
    return kind if not parts else f"{kind}:{','.join(parts)}"


def scale_kind(scale):
    """Just the kind of a display-scale string ('logicle:w=1' -> 'logicle')."""
    return parse_scale(scale)[0]


def scale_label(scale):
    """How a display scale reads in the UI: 'Logicle (W 1.2)', 'Log'."""
    kind, params = parse_scale(scale)
    names = dict(PICKABLE_SCALES)
    name = names.get(kind, kind.capitalize()).split(' (')[0]
    if kind == 'logicle':
        return f"{name} (W {params.get('w', LOGICLE_DEFAULTS['w']):g})"
    if kind == 'asinh':
        return f"{name} (cofactor {params.get('c', ASINH_DEFAULT_COFACTOR):g})"
    return name


def intensity_view(scale):
    """``(forward, inverse)`` between linear intensity and the screen
    coordinate of a logicle or arcsinh display scale, or None for any other
    kind. Exact (flowutils' logicle), elementwise, shape-preserving."""
    from .pipeline import inverse_transform_values, transform_values
    kind, params = parse_scale(scale)
    if kind == 'logicle':
        # Logicle needs W <= M/2; LOGICLE_W_RANGE keeps it there.
        w = params.get('w')
        w = LOGICLE_DEFAULTS['w'] if w is None else float(w)
        t, m, a = (LOGICLE_DEFAULTS[k] for k in ('t', 'm', 'a'))

        def fwd(v):
            arr = np.asarray(v, dtype=float)
            return np.asarray(transform_values(
                arr.ravel(), 'logicle', t=int(t), m=m, w=w, a=int(a)),
                dtype=float).reshape(arr.shape)

        def inv(p):
            arr = np.asarray(p, dtype=float)
            return np.asarray(inverse_transform_values(
                arr.ravel(), 'logicle', t=int(t), m=m, w=w, a=int(a)),
                dtype=float).reshape(arr.shape)
        return fwd, inv
    if kind == 'asinh':
        c = float(params.get('c', ASINH_DEFAULT_COFACTOR))
        return (lambda v: np.arcsinh(np.asarray(v, dtype=float) / c),
                lambda p: np.sinh(np.asarray(p, dtype=float)) * c)
    return None


def _is_identity_view(transform, scale):
    """True when the logicle display scale at its defaults is the stored
    coordinate itself (fluorescence the loader stored as default logicle):
    the view then costs nothing and is exact."""
    kind, params = parse_scale(scale)
    return (transform == 'logicle' and kind == 'logicle'
            and 'w' not in params)


def view_funcs(transform: str, scale: str, data_sample=None):
    """``(forward, inverse)`` callables mapping a channel's STORED data
    coordinate to screen position for ``scale`` — or ``None`` when matplotlib's
    own scale draws it: a linear-stored channel on a linear, log or symlog
    view (nicer native tick locators).

    ``symlog`` uses an arcsinh view whose cofactor is anchored on the data's 5th
    percentile of ``|nonzero intensity|`` when ``data_sample`` is given.
    ``logicle`` and ``asinh`` (with their parameters, see parse_scale) are
    views of the linear intensity, for linear-stored channels too.
    """
    from .pipeline import inverse_transform_values, transform_values
    kind = scale_kind(scale)
    if transform == 'unknown':
        return None
    if transform == 'linear' and kind not in ('logicle', 'asinh'):
        return None

    def inv_baked(a):
        return inverse_transform_values(np.asarray(a, dtype=float),
                                        method=transform)

    def fwd_baked(a):
        return transform_values(np.asarray(a, dtype=float), method=transform)

    if _is_identity_view(transform, scale):
        def forward(d):  # pyright: ignore[reportRedeclaration]
            return np.asarray(d, dtype=float)

        def inverse(p):  # pyright: ignore[reportRedeclaration]
            return np.asarray(p, dtype=float)
    elif kind in ('logicle', 'asinh'):
        vfwd, vinv = intensity_view(scale)  # type: ignore[misc]

        def forward(d):  # pyright: ignore[reportRedeclaration]
            return vfwd(inv_baked(d))

        def inverse(p):  # pyright: ignore[reportRedeclaration]
            return fwd_baked(vinv(p))
    elif kind == 'log':
        def forward(d):  # pyright: ignore[reportRedeclaration]
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.log10(np.clip(inv_baked(d), LOG_VIEW_MIN, None))

        def inverse(p):  # pyright: ignore[reportRedeclaration]
            return fwd_baked(np.power(10.0, np.asarray(p, dtype=float)))
    elif kind == 'symlog':
        cof = _symlog_cofactor(inv_baked, data_sample)

        def forward(d):  # pyright: ignore[reportRedeclaration]
            return np.arcsinh(inv_baked(d) / cof)

        def inverse(p):  # pyright: ignore[reportRedeclaration]
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
    if transform == 'linear' or scale_kind(scale) != 'log':
        return None
    return float(np.log10(LOG_VIEW_MIN))


# ── Tick labels in intensity ─────────────────────────────────────────────────

def format_intensity(v) -> str:
    """Short axis label for a linear intensity: '0', every power of ten as
    10^k (1 is 10^0, 100 is 10^2), other values from 1,000 as 'K' ('2.5K',
    '50K'). One rule for every decade, so an axis never reads '10, 100,
    10^3' in two notations. Rounded to 6 significant digits first, so a tick
    that went through transform and inverse (999.9999996) reads as what it
    is."""
    v = float(v)
    if not np.isfinite(v):
        return ''
    r = float(f'{v:.6g}')
    if r != 0:
        k = math.log10(abs(r))
        if abs(k - round(k)) < 1e-9:
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
    entry, parameters included; a 'guessed' mark or an unknown entry's
    'origin' is not a parameter and is left out)."""
    if isinstance(transform, Mapping):
        from .pipeline import transform_params
        return transform_params(transform)
    return {'method': transform or 'linear'}


def _no_intensity(transform, scale=None) -> bool:
    """True when `transform` gives no intensity ticks for `scale`: a scale
    that is unknown (nothing linear can be said, see pipeline.UNKNOWN_SPEC),
    or a linear-stored channel on a view matplotlib ticks itself (anything
    but logicle / arcsinh)."""
    method = _transform_kwargs(transform).get('method', 'linear')
    if method == 'unknown':
        return True
    return method == 'linear' and scale_kind(scale) not in ('logicle', 'asinh')


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
    if _no_intensity(transform, scale):
        return empty
    kind = scale_kind(scale)
    from .pipeline import inverse_transform_values, transform_values
    kw = _transform_kwargs(transform)
    lo, hi = sorted((float(lo), float(hi)))
    a, b = (float(x) for x in inverse_transform_values(
        np.array([lo, hi]), **kw))
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return empty
    if kind in ('logicle', 'asinh'):
        # Logicle / arcsinh: 0 and +-decades, spaced on the view itself so
        # 10 and 100 don't sit on top of 0 where the view is linear-like.
        vfwd = intensity_view(scale)[0]  # type: ignore[index]
        cands = np.array([0.0] + [sg * 10.0 ** k for k in range(1, 9)
                                  for sg in (1.0, -1.0)])
        cands = np.sort(cands[(cands >= a) & (cands <= b)])
        if cands.size < 2:
            # Zoomed between decades (a layout's magnified inset, a scroll
            # zoom): round values, as a narrow log view gets, not a bare
            # axis or a lone '10^3'.
            from matplotlib.ticker import MaxNLocator
            vals = np.asarray(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
                              .tick_values(a, b), dtype=float)
            vals = vals[(vals >= a) & (vals <= b)]
        else:
            s_lo, s_hi = (float(x) for x in vfwd(np.array([a, b])))
            vals = _spaced_from_zero(
                cands, np.asarray(vfwd(cands), dtype=float),
                (s_hi - s_lo) / (max_ticks + 1))
    elif kind == 'log':
        a = max(a, LOG_VIEW_MIN)
        if b <= a:
            return empty
        ks = np.arange(math.ceil(math.log10(a) - 1e-9),
                       math.floor(math.log10(b) + 1e-9) + 1)
        step = max(1, math.ceil(len(ks) / max_ticks))
        # Multiples of the step, so the ticks stay put while panning.
        vals = 10.0 ** ks[ks % step == 0]
        if _is_narrow_log(a, b):
            # Zoomed inside about a decade: round values instead ('500',
            # '1K'), as the layout exports tick such a view.
            vals = _narrow_log_values(a, b)
    elif kind == 'symlog':
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
    elif kind == 'stored':
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


def _is_narrow_log(a, b) -> bool:
    """True when a log view from intensity `a` to `b` holds fewer than two
    decades, so it is ticked with round values (all written plainly, a
    1,000 among them as '1K', not one '10^3' among '500' and '1.5K')."""
    a, b = sorted((float(a), float(b)))
    a = max(a, LOG_VIEW_MIN)
    # An empty or non-finite view is not narrow: the decade path, which
    # finds no ticks in it, handles it.
    return bool(np.isfinite(a) and np.isfinite(b) and b > a
                and (math.floor(math.log10(b) + 1e-9)
                     - math.ceil(math.log10(a) - 1e-9)) < 1)


def _narrow_log_values(a, b, nbins=4):
    """Round, positive tick values in [a, b] for a log view too narrow to
    hold two decades."""
    from matplotlib.ticker import MaxNLocator
    vals = np.asarray(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
                      .tick_values(a, b), dtype=float)
    return vals[(vals > 0) & (vals >= a) & (vals <= b)]


def native_log_ticker():
    """``(locator, formatter)`` for a matplotlib-native log axis (a channel
    stored linear, shown on log): decades as 10^k while the view holds two
    or more, round values ('500', '1K') when it is zoomed inside about a
    decade -- where matplotlib would label every minor tick and overprint
    them. Follows zoom and pan without a replot."""
    from matplotlib.ticker import Formatter, Locator, LogLocator

    class NarrowAwareLogLocator(Locator):
        def __init__(self):
            self._log = LogLocator(base=10.0)

        def set_axis(self, axis):
            super().set_axis(axis)
            self._log.set_axis(axis)

        def __call__(self):
            lo, hi = self.axis.get_view_interval()   # type: ignore[union-attr]
            return self.tick_values(lo, hi)

        def tick_values(self, vmin, vmax):
            vmin, vmax = sorted((float(vmin), float(vmax)))
            vmin = max(vmin, LOG_VIEW_MIN)
            if vmax <= vmin:
                return np.array([])
            if not _is_narrow_log(vmin, vmax):
                return self._log.tick_values(vmin, vmax)
            return _narrow_log_values(vmin, vmax)

    class IntensityText(Formatter):
        def __call__(self, x, pos=None):
            lo, hi = self.axis.get_view_interval()   # type: ignore[union-attr]
            if _is_narrow_log(lo, hi):
                return format_intensity_text(x, digits=4)
            return format_intensity(x)

    return NarrowAwareLogLocator(), IntensityText()


def intensity_ticker(transform, scale: str, data_sample=None):
    """``(locator, formatter)`` for a matplotlib axis showing a
    ``transform``-stored channel with ``scale`` (see ``view_funcs``; or
    'stored' for an axis in the stored coordinate itself, see
    ``intensity_ticks``), so its ticks sit at round intensities and say
    them; None for a linear-stored channel or one of unknown scale.
    `transform` is a method name or a ``transform_spec`` dict. The locator
    follows the current view, so a zoom or pan re-ticks without a replot."""
    if _no_intensity(transform, scale):
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
            if scale_kind(scale) == 'log':
                lo, hi = self.axis.get_view_interval()  # type: ignore[union-attr]
                if _is_narrow_log(intensity(lo), intensity(hi)):
                    return format_intensity_text(intensity(x), digits=4)
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


def linear_tick_formatter():
    """A matplotlib formatter for a native LINEAR axis that reads like the
    intensity axes: '0', '500', '50K', '262K' -- not '50000' or an offset
    '1e5' in the corner."""
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _pos=None):
        v = float(v)
        if not np.isfinite(v):
            return ''
        if abs(v) >= 1000:
            return format_intensity_text(v, digits=4)
        return f'{float(f"{v:.6g}"):g}'
    return FuncFormatter(fmt)
