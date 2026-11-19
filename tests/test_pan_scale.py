"""A middle-button pan moves each axis in its own scale.

The pan moved both limits by the same step in data units, a fraction of the
data span for the fraction of the axes the pointer travelled. On a log axis
that drags the view wider at one end and narrower at the other, and a pan to
the right takes the lower limit to zero or below, which a log axis cannot
place (it reads it as 10^-1000). On a logicle or asinh view it slid the
stored values, not what is drawn. Now the limits move by that fraction of the
span as drawn: in decades on log, in screen units on a function scale, as
before on linear.

Generated data only; driven through the editor's _on_press / _on_motion.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_editor_gating_dispatch import (
    LIMITS,
    _editor_or_skip,
    _ev,
    _install,
    _set_scale,
    _setup,
)


def _scale_span(ed, axis):
    """``axis``'s limits in its own scale's units (decades on log)."""
    ax_obj = ed.ax.xaxis if axis == 'x' else ed.ax.yaxis
    lims = ed.ax.get_xlim() if axis == 'x' else ed.ax.get_ylim()
    return np.asarray(ax_obj.get_transform().transform(np.asarray(lims, float)),
                      float).ravel()


def _pan(ed, frm, to):
    """Middle-drag from axes fraction ``frm`` to ``to``; the pointer's
    travel as fractions of the axes box."""
    p0 = _ev(ed, *frm, button=2)
    ed._on_press(p0)
    assert ed._pan_start is not None
    p1 = _ev(ed, *to)
    ed._on_motion(p1)
    bw, bh = ed._pan_start[4:]
    return (p1.x - p0.x) / bw, (p1.y - p0.y) / bh


@pytest.fixture
def ed():
    root, editor = _editor_or_skip()
    try:
        _setup(editor)
        yield editor
    finally:
        root.destroy()


def test_a_log_axis_pans_by_decades(ed):
    _set_scale(ed, 'log')
    _install(ed, {}, 'log')
    sx0, sy0 = _scale_span(ed, 'x'), _scale_span(ed, 'y')
    fx, fy = _pan(ed, (0.5, 0.5), (0.8, 0.3))
    sx1, sy1 = _scale_span(ed, 'x'), _scale_span(ed, 'y')
    # The same number of decades shown, moved by the pointer's share of them.
    assert sx1 == pytest.approx(sx0 - fx * (sx0[1] - sx0[0]), abs=1e-9)
    assert sy1 == pytest.approx(sy0 - fy * (sy0[1] - sy0[0]), abs=1e-9)
    assert fx > 0 and fy < 0


def test_a_long_pan_right_keeps_a_log_axis_positive(ed):
    """A linear step of most of the span took the lower limit below zero."""
    _set_scale(ed, 'log')
    _install(ed, {}, 'log')
    (xl, xh), _ylim = LIMITS['log']
    _pan(ed, (0.05, 0.5), (0.95, 0.5))
    lo, hi = ed.ax.get_xlim()
    assert 0 < lo < xl and hi < xh
    assert np.log10(hi / lo) == pytest.approx(np.log10(xh / xl), rel=1e-9)


def test_a_function_scale_pans_in_screen_units(ed):
    """A logicle view of a linear-stored channel is a function scale: the
    pan moves what is drawn, by the pointer's share of the view."""
    from tests.test_editor_gating_dispatch import X
    ed._channel_scale[X] = 'logicle'
    ed._replot()
    assert ed.ax.get_xscale() == 'function'
    _install(ed, {}, 'linear')
    s0 = _scale_span(ed, 'x')
    fx, _fy = _pan(ed, (0.5, 0.5), (0.75, 0.5))
    s1 = _scale_span(ed, 'x')
    assert s1 == pytest.approx(s0 - fx * (s0[1] - s0[0]), abs=1e-9)
    assert fx > 0


def test_a_pan_the_scale_cannot_invert_keeps_the_view():
    """A scale whose inverse is undefined past the pan (NaN) leaves the
    limits as they were rather than setting NaN limits."""
    from matplotlib.figure import Figure

    from openflo.editor_plot import panned_lims
    ax = Figure().add_subplot()
    # Scale units are twice the data, so limits in scale units would show.
    ax.set_xscale('function', functions=(
        lambda v: 2.0 * np.asarray(v, float),
        lambda v: np.where(np.asarray(v, float) < 0, np.nan,
                           np.asarray(v, float) / 2.0)))
    ax.set_xlim(1.0, 2.0)
    assert panned_lims(ax.xaxis, (1.0, 2.0), 0.5) == pytest.approx((0.5, 1.5))
    assert panned_lims(ax.xaxis, (1.0, 2.0), 1.5) == (1.0, 2.0)


def test_a_log_pan_from_a_non_positive_limit_keeps_its_decades():
    """A non-positive limit set while the axis was linear survives the switch
    to log; the pan measures from the smallest positive data value, as the
    axis draws it, not from 10^-1000 (which would collapse the view)."""
    from matplotlib.figure import Figure

    from openflo.editor_plot import panned_lims
    ax = Figure().add_subplot()
    ax.plot([2.0, 10.0, 1e4], [1.0, 2.0, 3.0])
    ax.set_xlim(-5.0, 1e5)
    ax.set_xscale('log')
    assert ax.get_xlim()[0] < 0
    span = np.log10(1e5 / 2.0)
    lo, hi = panned_lims(ax.xaxis, ax.get_xlim(), 0.1)
    assert (lo, hi) == pytest.approx((2.0 * 10 ** (-0.1 * span),
                                      1e5 * 10 ** (-0.1 * span)), rel=1e-9)
