"""Moving a whole gate works in display space, as FlowJo and the phone do: on
a log or logicle axis the gate keeps its look on the screen and its stored
size changes.

mobile/tests/test_gate_moves_table.py holds openflo.gate_move to the table
of the phone's answers. Here: what that table cannot show (open sides, a
log view's floor, linear axes kept exact), and the editor's drag and arrow
keys on a channel stored linear and shown on a logicle view (SSC-A) and
one on matplotlib's own log scale.

All data are generated here (synthetic sample names, generic channels).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import tests._four_sample_editor as shared
from openflo.editor_keyboard import NUDGE_STEP_LARGE, nudge_gate
from openflo.gate_move import OPEN, move_gate, shift_values

ed = shared.ed

def test_a_side_below_a_log_floor_stays_and_an_open_one_stays_open():
    log = (lambda v: np.log10(np.clip(v, 1e-6, None)),
           lambda v: np.power(10.0, v))
    assert shift_values([-50.0, 10.0, OPEN], 1.0, log) == [-50.0, 100.0, OPEN]
    # Just below the floor stays; just above it moves.
    below, above = shift_values([9.5e-7, 2e-6], 1.0, log)
    assert below == 9.5e-7
    assert above == pytest.approx(2e-5, rel=1e-9)
    # An ellipse whose box reaches below the floor: its centre moves a
    # decade, and its size scales as the half of the box still on the axis
    # does (x10), as it would with the box just inside the floor.
    e = {'kind': 'ellipsoid', 'mean': [100.0, 5.0],
         'cov': [[10000.0, 0.0], [0.0, 1.0]]}
    moved = move_gate(e, 1.0, 0.0, log, None)
    np.testing.assert_allclose(moved['mean'], [1000.0, 5.0])
    np.testing.assert_allclose(moved['cov'], [[1e6, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize('sd', [499.0, 501.0])
def test_an_ellipse_does_not_jump_as_its_box_crosses_the_floor(sd):
    """Mean 1000, distance 4: an SD of 499 keeps the box above zero, 501
    takes it below. Moved up a decade, both scale x10 and land at 10000."""
    log = (lambda v: np.log10(np.clip(v, 1e-6, None)),
           lambda v: np.power(10.0, v))
    e = {'kind': 'ellipsoid', 'mean': [1000.0, 5.0],
         'cov': [[sd * sd, 0.0], [0.0, 1.0]]}
    moved = move_gate(e, 1.0, 0.0, log, None)
    np.testing.assert_allclose(moved['mean'], [10000.0, 5.0], rtol=1e-9)
    np.testing.assert_allclose(moved['cov'][0][0], 100 * sd * sd, rtol=1e-9)


def test_an_ellipse_with_its_centre_below_the_floor_stays():
    log = (lambda v: np.log10(np.clip(v, 1e-6, None)),
           lambda v: np.power(10.0, v))
    e = {'kind': 'ellipsoid', 'mean': [-20.0, 5.0],
         'cov': [[2500.0, 0.0], [0.0, 1.0]]}
    assert move_gate(e, 1.0, 0.0, log, None) == {'mean': [-20.0, 5.0]}


def test_an_arrow_key_on_a_gate_below_the_floor_is_no_move():
    log = (lambda v: np.log10(np.clip(v, 1e-6, None)),
           lambda v: np.power(10.0, v))
    r = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'x0': -90.0, 'x1': -10.0, 'y0': 1.0, 'y1': 2.0}
    assert not nudge_gate(r, 0.5, 0.0, 'FL1-A', 'FL2-A', log, None)
    assert (r['x0'], r['x1']) == (-90.0, -10.0)


def test_a_move_with_no_stored_value_is_refused():
    log = (np.log10, lambda v: np.power(10.0, v))
    # A move to where the axis has no stored value is refused.
    assert move_gate({'kind': 'threshold', 'value': 10.0}, 400.0, 0.0,
                     log) is None


def test_linear_axes_add_the_stored_delta_exactly():
    e = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [1.0, 1.0]}
    assert nudge_gate(e, -1.0, 0.25, 'FL1-A', 'FL2-A')
    assert e['mean'] == [0.0, 1.25]
    assert not nudge_gate(e, 0.0, 0.0, 'FL1-A', 'FL2-A')


def test_arrow_keys_move_a_range_on_its_own_axis_as_shown():
    log = (np.log10, lambda v: np.power(10.0, v))
    # A threshold on the y channel takes dy, on the y axis (log here).
    t = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 30.0}
    assert nudge_gate(t, 0.5, 1.0, 'FL1-A', 'FL2-A', None, log)
    assert t['value'] == pytest.approx(300.0)
    i = {'kind': 'interval', 'channel': 'FL1-A', 'lo': 10.0, 'hi': 20.0}
    assert nudge_gate(i, 2.0, 1.0, 'FL1-A', 'FL2-A', log, None)
    assert (i['lo'], i['hi']) == (pytest.approx(1000.0), pytest.approx(2000.0))


def _place(ed, x, y, gate, scales):
    ed._set_active_sample('synthetic_a1')
    saved = {c: ed._channel_scale.get(c) for c in scales}
    ed._channel_scale.update(scales)
    ed.x_combo.set(x)
    ed.y_combo.set(y)
    gid = ed._next_gate_id()
    ed._gates[gid] = dict(gate, parent_id=None, name='Box', enabled=True)
    ed._sample_gate_order['synthetic_a1'].append(gid)
    ed._replot()
    return gid, saved


def _restore(ed, gid, saved):
    ed._gates.pop(gid, None)
    ed._sample_gate_order['synthetic_a1'].remove(gid)
    for c, s in saved.items():
        if s is None:
            ed._channel_scale.pop(c, None)
        else:
            ed._channel_scale[c] = s
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')
    ed._replot()


def _all_iids(tv, parent=''):
    for i in tv.get_children(parent):
        yield i
        yield from _all_iids(tv, i)


def _screen_box(ed, g):
    p = ed.ax.transData.transform([[g['x0'], g['y0']], [g['x1'], g['y1']]])
    return p[1] - p[0], p[0]


RECT = {'kind': 'rect', 'x_channel': 'SSC-A', 'y_channel': 'FL2-A',
        'x0': 300.0, 'x1': 1200.0, 'y0': 900.0, 'y1': 1100.0}
SCALES = {'SSC-A': 'logicle', 'FL2-A': 'log'}


def test_a_dragged_gate_keeps_its_size_on_screen(ed):
    gid, saved = _place(ed, 'SSC-A', 'FL2-A', RECT, SCALES)
    try:
        assert not ed.ax.transData.is_affine
        g = ed._gates[gid]
        size0, corner0 = _screen_box(ed, g)
        # Press inside, drag 60 px right and 40 px up.
        press = ed.ax.transData.inverted().transform(corner0 + size0 / 2)
        to = ed.ax.transData.inverted().transform(
            corner0 + size0 / 2 + [60.0, 40.0])
        ed._drag_state = ('rect_translate', gid)
        assert ed._begin_drag_translate(
            SimpleNamespace(xdata=press[0], ydata=press[1]),
            ('rect_translate', gid))
        assert ed._motion_drag_translate('rect_translate', to[0], to[1])
        size1, corner1 = _screen_box(ed, g)
        np.testing.assert_allclose(size1, size0, atol=1e-6)
        np.testing.assert_allclose(corner1 - corner0, [60.0, 40.0],
                                   atol=1e-6)
        # ... so its stored width changed: logicle is not a shift of SSC-A.
        assert abs((g['x1'] - g['x0']) - 900.0) > 1.0
        # Each motion starts again from the gate at the press: back at the
        # press point, the gate is where it was.
        assert ed._motion_drag_translate('rect_translate', press[0],
                                         press[1])
        for k in ('x0', 'x1', 'y0', 'y1'):
            assert g[k] == pytest.approx(RECT[k], rel=1e-9), k
    finally:
        ed._drag_state = None
        ed._drag_translate_ctx = None
        _restore(ed, gid, saved)


def test_an_arrow_key_step_is_a_share_of_the_axis_as_shown(ed):
    gid, saved = _place(ed, 'SSC-A', 'FL2-A', RECT, SCALES)
    try:
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(next(
            i for i in _all_iids(ed.gate_tv)
            if ed._parse_iid(i) == ('gate', 'synthetic_a1', gid)))
        g = ed._gates[gid]
        size0, corner0 = _screen_box(ed, g)
        assert ed._nudge_selected_gate(NUDGE_STEP_LARGE, NUDGE_STEP_LARGE)
        size1, corner1 = _screen_box(ed, g)
        np.testing.assert_allclose(size1, size0, atol=1e-6)
        width_px = ed.ax.bbox.width
        height_px = ed.ax.bbox.height
        np.testing.assert_allclose(
            corner1 - corner0,
            [NUDGE_STEP_LARGE * width_px, NUDGE_STEP_LARGE * height_px],
            atol=1e-6)
    finally:
        _restore(ed, gid, saved)
