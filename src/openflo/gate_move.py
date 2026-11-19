"""Moving a whole gate on the plot: in display space, as FlowJo does.

A gate moved on a log, logicle or asinh axis keeps the look it has on the
screen: every stored value is taken to the axis, moved there by the same
on-screen distance, and taken back. Its stored width changes (on a log axis
a box 10..100 moved up a decade is 100..1000), where adding one stored delta
to every value would stretch it on the screen near zero and squash it far
from zero. On a linear axis the two are the same, and the stored delta is
added as it is.

This is the phone's move (mobile/src/gateedit.ts: dragGeo on the plot's
units, then applyGeo back to stored values), and mobile/tests/gate_moves.json
holds both to the same answers: mobile/tests/test_gate_moves_table.py
checks this module against it, mobile/tests/gate_moves.test.ts the phone.

  * A rectangle: each side moves; an open side (|v| >= OPEN) stays open,
    and so does a side the axis cannot show (below a log view's floor).
  * A polygon: each stored vertex moves.
  * An ellipsoid: the box around it (mean +- sqrt(distance_sq * var) on each
    axis) moves; the ellipse keeps its tilt and fills the new box, centred
    in it, each axis stretched by as much as the box's side was. A box
    reaching below a log view's floor: its centre moves, and the half of
    the box still on the axis sets the stretch.
  * A range or threshold: its ends move along its axis.

An axis is a pair of callables (to_display, to_stored), each mapping an
array of values, or None for an axis that shows stored values as they are.
Tk-free; axis_move_fns reads a matplotlib Axis it is handed.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .gate_store import ONE_D_KINDS
from .scales import OPEN

AxisFns = tuple[Callable[[np.ndarray], np.ndarray],
                Callable[[np.ndarray], np.ndarray]]

# Ellipsoid gates select (p-mu)' S^-1 (p-mu) <= distance_sq, 4 when not
# given (gate_eval).
_DEFAULT_DSQ = 4.0


def _shift(values, d, fns):
    """``values`` moved by ``d`` display units, and which of them moved;
    None when a moved place has no stored value."""
    vals = [float(v) for v in values]
    moved = [False] * len(vals)
    if not d:
        return vals, moved
    shut = [i for i, v in enumerate(vals) if abs(v) < OPEN]
    if not shut:
        return vals, moved
    a = np.asarray([vals[i] for i in shut], dtype=float)
    if fns is None:
        new = a + d
        placed = np.ones(a.shape, dtype=bool)
    else:
        to_display, to_stored = fns
        with np.errstate(all='ignore'):
            shown = np.asarray(to_display(a), dtype=float).ravel()
            back = np.asarray(to_stored(shown), dtype=float).ravel()
            new = np.asarray(to_stored(shown + d), dtype=float).ravel()
        # A value the axis shows somewhere else (clipped to a log view's
        # floor, or scrubbed to 0) is off the axis: it stays, as an open
        # side does.
        placed = (np.isfinite(shown) & np.isfinite(back)
                  & np.isclose(back, a, rtol=1e-7, atol=1e-12))
        if not np.isfinite(new[placed]).all():
            return None
    out = list(vals)
    for k, i in enumerate(shut):
        if placed[k]:
            out[i] = float(new[k])
            moved[i] = True
    return out, moved


def shift_values(values: Sequence[float], d: float,
                 fns: AxisFns | None) -> list[float] | None:
    """Stored ``values`` moved by ``d`` along an axis, in its display units.
    An open value (|v| >= OPEN) stays as it is, and so does a value the
    axis does not show where it is (below a log view's floor); with ``d``
    0 every value does, exactly (no round trip). None when a moved place
    has no stored value."""
    got = _shift(values, d, fns)
    return None if got is None else got[0]


def _move_rect(g, dx, dy, fx, fy):
    xs = shift_values([g['x0'], g['x1']], dx, fx)
    ys = shift_values([g['y0'], g['y1']], dy, fy)
    if xs is None or ys is None:
        return None
    return {'x0': min(xs), 'x1': max(xs), 'y0': min(ys), 'y1': max(ys)}


def _move_polygon(g, dx, dy, fx, fy):
    verts = [[float(v[0]), float(v[1])] for v in g.get('vertices') or []]
    xs = shift_values([v[0] for v in verts], dx, fx)
    ys = shift_values([v[1] for v in verts], dy, fy)
    if xs is None or ys is None:
        return None
    return {'vertices': [[x, y] for x, y in zip(xs, ys, strict=True)]}


def _ellipse_axis(g, mean, axis, d, fns):
    """The ellipse's centre on one axis after a move of ``d`` there, and how
    much its size on that axis scales; None if it cannot move."""
    if not d or fns is None:
        # Not moved on this axis, or a linear one: the centre moves by the
        # stored delta and the size is kept exactly.
        return mean + d, 1.0
    cov = np.asarray(g['cov'], dtype=float)
    dsq = float(g.get('distance_sq', _DEFAULT_DSQ))
    h = float(np.sqrt(max(dsq * cov[axis][axis], 0.0)))
    lo, hi = mean - h, mean + h
    got = _shift([lo, hi], d, fns)
    if got is None:
        return None
    (nlo, nhi), moved = got
    if all(moved) and hi > lo:
        nlo, nhi = min(nlo, nhi), max(nlo, nhi)
        return (nlo + nhi) / 2, (nhi - nlo) / (hi - lo)
    # A side of the box off the axis (below a log view's floor): the
    # centre moves as a point, and the size scales as the half of the box
    # still on the axis does, so the move does not jump as that side
    # crosses the floor. The centre off the axis too: it stays.
    got = _shift([mean, lo, hi], d, fns)
    if got is None:
        return None
    (nmean, _nlo, nhi), (cmoved, _lmoved, hmoved) = got
    if not cmoved:
        return mean, 1.0
    # A log view's floor is below: the upper half is the one on the axis.
    k = (nhi - nmean) / (hi - mean) if hmoved and hi > mean else 1.0
    return nmean, (k if np.isfinite(k) and k > 0 else 1.0)


def _move_ellipsoid(g, dx, dy, fx, fy):
    mean = [float(m) for m in g['mean']]
    ax = _ellipse_axis(g, mean[0], 0, dx, fx)
    ay = _ellipse_axis(g, mean[1], 1, dy, fy)
    if ax is None or ay is None:
        return None
    (mx, kx), (my, ky) = ax, ay
    if kx == 1.0 and ky == 1.0:
        return {'mean': [mx, my]}
    c = np.asarray(g['cov'], dtype=float)
    new_cov = [[float(c[0][0] * kx * kx), float(c[0][1] * kx * ky)],
               [float(c[1][0] * kx * ky), float(c[1][1] * ky * ky)]]
    return {'mean': [mx, my], 'cov': new_cov}


def _move_one_d(g, d, fns):
    if g['kind'] == 'threshold':
        got = shift_values([g['value']], d, fns)
        return None if got is None else {'value': got[0]}
    got = shift_values([g['lo'], g['hi']], d, fns)
    return None if got is None else {'lo': min(got), 'hi': max(got)}


def axis_move_fns(axis) -> AxisFns | None:
    """A matplotlib Axis (``ax.xaxis``) as move_gate wants it: (to display,
    to stored) through its scale (a FuncScale view of a logicle, asinh or
    log channel; matplotlib's own log or symlog), or None for a linear
    axis, where the stored delta is the move."""
    tr = axis.get_transform()
    if tr.is_affine:
        return None
    inv = tr.inverted()

    def to_display(a):
        return np.asarray(tr.transform(np.reshape(a, (-1, 1))),
                          dtype=float).ravel()

    def to_stored(a):
        return np.asarray(inv.transform(np.reshape(a, (-1, 1))),
                          dtype=float).ravel()
    return to_display, to_stored


def display_delta(fns: AxisFns | None, frm: float, to: float) -> float:
    """From stored value ``frm`` to stored value ``to``, in the axis's
    display units."""
    if fns is None:
        return float(to) - float(frm)
    d = fns[0](np.asarray([float(frm), float(to)], dtype=float))
    return float(d[1] - d[0])


def move_gate(g: dict, dx: float, dy: float,
              x_fns: AxisFns | None = None,
              y_fns: AxisFns | None = None) -> dict | None:
    """The fields of gate ``g`` that change when it is moved by ``dx`` along
    the plot's x axis and ``dy`` along y, in display units (``x_fns`` and
    ``y_fns`` the axes, for the gate's own x and y channels). For a range
    or threshold, ``dx`` and ``x_fns`` are its own axis's. None when the
    gate cannot move: a kind with no region (a boolean, a quadrant set is
    moved by its crossing), or a value the axis cannot place. ``g`` itself
    is not changed."""
    kind = g.get('kind')
    if kind == 'rect':
        return _move_rect(g, dx, dy, x_fns, y_fns)
    if kind == 'polygon':
        return _move_polygon(g, dx, dy, x_fns, y_fns)
    if kind == 'ellipsoid':
        return _move_ellipsoid(g, dx, dy, x_fns, y_fns)
    if kind in ONE_D_KINDS:
        return _move_one_d(g, dx, x_fns)
    return None
