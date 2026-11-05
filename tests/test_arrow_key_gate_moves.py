"""Arrow keys move the selected gate (item 14), on its own axes only, with one
undo step for a run of presses; with no gate selected they do nothing.

All data are generated here (synthetic sample names, generic FL channels).
"""
from __future__ import annotations

import pytest

import tests._four_sample_editor as shared
from openflo.editor_keyboard import nudge_gate

ed = shared.ed


def test_nudge_moves_each_gate_kind_on_its_axes():
    r = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'x0': 0.0, 'x1': 1.0, 'y0': 0.0, 'y1': 2.0}
    assert nudge_gate(r, 0.5, -1.0, 'FL1-A', 'FL2-A')
    assert (r['x0'], r['x1'], r['y0'], r['y1']) == (0.5, 1.5, -1.0, 1.0)
    p = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[0, 0], [1, 0], [0, 1]]}
    assert nudge_gate(p, 1.0, 2.0, 'FL1-A', 'FL2-A')
    assert p['vertices'] == [[1, 2], [2, 2], [1, 3]]
    e = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [1.0, 1.0]}
    assert nudge_gate(e, -1.0, 0.0, 'FL1-A', 'FL2-A')
    assert e['mean'] == [0.0, 1.0]
    t = {'kind': 'threshold', 'channel': 'FL2-A', 'value': 3.0}
    assert nudge_gate(t, 5.0, 0.25, 'FL1-A', 'FL2-A')       # y channel: dy
    assert t['value'] == 3.25
    i = {'kind': 'interval', 'channel': 'FL1-A', 'lo': 1.0, 'hi': 2.0}
    assert nudge_gate(i, 0.5, 9.0, 'FL1-A', 'FL2-A')
    assert (i['lo'], i['hi']) == (1.5, 2.5)


def test_nudge_leaves_a_gate_on_other_axes_alone():
    r = {'kind': 'rect', 'x_channel': 'FL3-A', 'y_channel': 'FL2-A',
         'x0': 0.0, 'x1': 1.0, 'y0': 0.0, 'y1': 2.0}
    assert not nudge_gate(r, 1.0, 1.0, 'FL1-A', 'FL2-A')
    assert r['x0'] == 0.0
    t = {'kind': 'threshold', 'channel': 'FL3-A', 'value': 3.0}
    assert not nudge_gate(t, 1.0, 1.0, 'FL1-A', 'FL2-A')
    assert t['value'] == 3.0
    q = {'kind': 'quadrant', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A'}
    assert not nudge_gate(q, 1.0, 1.0, 'FL1-A', 'FL2-A')


def _select_gate(ed, gid):
    ed._refresh_gate_list()
    iid = next(i for i in _all_iids(ed.gate_tv)
               if ed._parse_iid(i) == ('gate', 'synthetic_a1', gid))
    ed.gate_tv.selection_set(iid)


def _all_iids(tv, parent=''):
    for i in tv.get_children(parent):
        yield i
        yield from _all_iids(tv, i)


def test_arrow_keys_move_the_selected_gate_with_one_undo(ed):
    ed._set_active_sample('synthetic_a1')
    gid = ed._next_gate_id()
    ed._gates[gid] = {'kind': 'rect', 'x_channel': 'FL1-A',
                      'y_channel': 'FL2-A', 'x0': 900.0, 'x1': 1100.0,
                      'y0': 900.0, 'y1': 1100.0, 'parent_id': None,
                      'name': 'Box', 'enabled': True}
    ed._sample_gate_order['synthetic_a1'].append(gid)
    try:
        ed._replot()
        _select_gate(ed, gid)
        xl = ed.ax.get_xlim()
        span = xl[1] - xl[0]
        undo_before = len(ed._undo_stack)
        w = ed.canvas.get_tk_widget()
        assert w.bind('<Right>') and w.bind('<Shift-Up>')
        ed._on_arrow_key('Right', False)
        ed._undo_pending = False          # as if the Tk event loop ran
        ed._on_arrow_key('Right', False)
        g = ed._gates[gid]
        assert g['x0'] == pytest.approx(900 + 2 * 0.005 * span)
        assert g['y0'] == pytest.approx(900.0)            # y untouched
        assert len(ed._undo_stack) == undo_before + 1     # one undo step
        ed._undo_pending = False
        yl = ed.ax.get_ylim()
        ed._on_arrow_key('Up', True)
        assert g['y0'] == pytest.approx(900 + 0.05 * (yl[1] - yl[0]))
        ed._undo()
        assert ed._gates[gid]['x0'] == pytest.approx(900.0)
    finally:
        ed._gates.pop(gid, None)
        ed._sample_gate_order['synthetic_a1'].remove(gid)
        ed._refresh_gate_list()


def test_arrow_keys_do_nothing_without_a_selected_gate(ed):
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(ed._sample_iid('synthetic_a1'))
    assert ed._nudge_selected_gate(0.01, 0) is False
    assert 'Select a gate' in ed.status_var.get()
