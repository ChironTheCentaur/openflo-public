"""Which plot view shows a gate: its channels, a plot mode that draws it, and
whether the current zoom leaves any of it out.

"Show gate on plot" (the gate list's double-click and right-click entry)
asks :func:`view_for_gate` for the axes and mode, and
:func:`gate_bounds` / :func:`outside_view` whether the kept zoom must go.
Plain selection of a gate row never calls these, so it leaves the plot as it
was.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .gate_record import gate_record_problem
from .scales import OPEN

# Kinds the plot draws (editor_gating._draw_gates): a line on one channel,
# or a shape on an x/y pair.
_ONE_D = ('threshold', 'interval')
_TWO_D = ('rect', 'polygon', 'ellipsoid')

# A 2-D gate needs a 2-D plot; a histogram switches to the editor's
# default mode, the one a new window opens in.
DEFAULT_2D_MODE = 'pseudocolor'


def gate_plot_channels(g: Any) -> tuple[str, str | None] | None:
    """``(x, y)`` channels gate ``g`` is drawn on (``y`` None for a 1-D
    gate), or None when the plot never draws it (a boolean, auto-clean,
    cluster or damaged gate)."""
    if gate_record_problem(g) is not None:
        return None
    kind = g.get('kind')
    if kind in _ONE_D:
        return g['channel'], None
    if kind in _TWO_D:
        return g['x_channel'], g['y_channel']
    return None


def view_for_gate(g: Any, x: str | None, y: str | None,
                  mode: str) -> tuple[str | None, str | None, str] | None:
    """``(x, y, mode)`` of a plot on which gate ``g`` is drawn, changing as
    little of the current ``x`` / ``y`` / ``mode`` as it can; None when the
    plot never draws the gate.

    A 2-D gate is drawn only with its own x on X and y on Y, in a 2-D mode.
    A 1-D gate is drawn on X in any mode, and on Y in a 2-D mode, so a plot
    already showing its channel there is kept; otherwise its channel goes on
    X (the old X moves to Y if the channel was on Y of a histogram, so the
    two never match)."""
    chans = gate_plot_channels(g)
    if chans is None:
        return None
    gx, gy = chans
    if gy is not None:
        return gx, gy, (mode if mode != 'histogram' else DEFAULT_2D_MODE)
    if gx == x or (gx == y and mode != 'histogram'):
        return x, y, mode
    return gx, (x if gx == y else y), mode


def _finite(values) -> list[float]:
    """The closed, finite numbers of ``values`` (an open side's +-1e12 and
    a NaN say nothing about where the gate is)."""
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and abs(f) < OPEN:
            out.append(f)
    return out


def gate_bounds(g: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """``{channel: (lo, hi)}``: the extent of gate ``g``'s closed sides in
    stored values, per channel it is drawn on. A channel whose sides are
    all open is left out."""
    chans = gate_plot_channels(g)
    if chans is None:
        return {}
    kind = g['kind']
    if kind == 'threshold':
        per = {g['channel']: [g['value']]}
    elif kind == 'interval':
        per = {g['channel']: [g['lo'], g['hi']]}
    elif kind == 'rect':
        per = {g['x_channel']: [g['x0'], g['x1']],
               g['y_channel']: [g['y0'], g['y1']]}
    elif kind == 'polygon':
        verts = list(g['vertices'])
        per = {g['x_channel']: [v[0] for v in verts],
               g['y_channel']: [v[1] for v in verts]}
    else:                                   # ellipsoid
        d2 = float(g.get('distance_sq', 1.0))
        (cx, cy), cov = g['mean'], g['cov']
        try:
            hx = math.sqrt(max(d2 * float(cov[0][0]), 0.0))
            hy = math.sqrt(max(d2 * float(cov[1][1]), 0.0))
        except (TypeError, ValueError):
            return {}
        per = {g['x_channel']: [float(cx) - hx, float(cx) + hx],
               g['y_channel']: [float(cy) - hy, float(cy) + hy]}
    out = {}
    for ch, vals in per.items():
        f = _finite(vals)
        if f:
            out[ch] = (min(f), max(f))
    return out


def outside_view(bounds: Mapping[str, tuple[float, float]],
                 limits: Mapping[str, tuple[float, float]]) -> bool:
    """True when some part of a gate with ``bounds`` lies beyond the axis
    ``limits`` (``{channel: (a, b)}``, either order) on a channel both
    name."""
    for ch, (lo, hi) in bounds.items():
        if ch not in limits:
            continue
        a, b = sorted(float(v) for v in limits[ch])
        if lo < a or hi > b:
            return True
    return False


def view_limits(x: str | None, y: str | None, xlim, ylim
                ) -> dict[str, tuple[float, float]]:
    """``{channel: (a, b)}`` for :func:`outside_view` from a view's ``x`` /
    ``y`` channels (``y`` None for a histogram) and its ``xlim`` /
    ``ylim`` pairs; a channel without a channel name or a pair is left
    out."""
    out: dict[str, tuple[float, float]] = {}
    for ch, lim in ((x, xlim), (y, ylim)):
        if ch and lim is not None:
            out[ch] = (float(lim[0]), float(lim[1]))
    return out
