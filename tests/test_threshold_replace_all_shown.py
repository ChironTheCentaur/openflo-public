"""Re-placing a threshold with "Apply to all shown" moves it on every sample.

A threshold or interval gate is one per (channel, parent): a second
double-click in histogram mode moves the active sample's threshold instead
of adding another. `_add_gate_multi` copied the gate onto every other shown
sample but always appended it, so each re-place left one more threshold on
every other sample (scope audit G4). The replica now replaces the target's
twin under the same population, keeping the target's own name and colour,
and a replica rooted only because the target lacks the parent population
never takes over a root gate of the target's own.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.conftest import editor_or_skip

COLS = ('FSC-A', 'SSC-A', 'FL1-A', 'FL2-A')
NAMES = ('synthetic_a1', 'synthetic_a2', 'synthetic_a3')


def _editor(n=3):
    root, ed = editor_or_skip()
    ed._finish_plot_canvas()
    rng = np.random.default_rng(7)
    for k, name in enumerate(NAMES[:n]):
        df = pd.DataFrame(rng.normal(1000 + 100 * k, 50, (500, 4)),
                          columns=COLS)
        ed._samples[name] = SimpleNamespace(
            name=name, path=f'{name}.fcs', data=df,
            fluor_channels=['FL1-A', 'FL2-A'],
            channel_labels={c: c for c in COLS})
        ed._sample_order.append(name)
        ed._sample_trial[name] = 'Day1'
        if 'Day1' not in ed._trial_order:
            ed._trial_order.append('Day1')
        ed._sample_plot_enabled[name] = True
        ed._sample_gates.setdefault(name, {})
        ed._sample_gate_seq.setdefault(name, 0)
        ed._sample_gate_order.setdefault(name, [])
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed._set_active_sample(NAMES[0])
    ed._gate_all_var.set(True)
    return root, ed


def _thr(value, channel='FL1-A', parent=None):
    return {'kind': 'threshold', 'channel': channel, 'op': '>',
            'value': value, 'parent_id': parent}


def _one_d(ed, name, channel='FL1-A'):
    """(id, value) of each 1-D gate on `channel` in `name`, in order."""
    gates = ed._sample_gates[name]
    return [(gid, gates[gid].get('value', gates[gid].get('lo')))
            for gid in ed._sample_gate_order[name]
            if gid in gates and gates[gid].get('channel') == channel]


def _place(ed, value, **kw):
    ed._clear_undo_pending()
    return ed._add_gate_multi(_thr(value, **kw))


def _seed(ed, name, gid, gate):
    gate = dict(gate, id=gid, enabled=True, color='#123456')
    ed._sample_gates[name][gid] = gate
    ed._sample_gate_order[name].append(gid)
    ed._sample_gate_seq[name] = max(ed._sample_gate_seq[name],
                                    int(gid.lstrip('g')))
    if name == ed._active_sample:
        ed._gate_id_seq = ed._sample_gate_seq[name]


def test_re_placing_moves_the_threshold_on_every_shown_sample():
    root, ed = _editor()
    try:
        for v in (1000.0, 1100.0, 1200.0):
            _place(ed, v)
        for name in NAMES:
            assert [v for _g, v in _one_d(ed, name)] == [1200.0], name
    finally:
        root.destroy()


def test_the_double_click_path_moves_it_everywhere():
    """The user's path: double-clicks in histogram mode with the gate tool."""
    root, ed = _editor()
    try:
        ed.x_combo.set('FL1-A')
        ed.mode_var.set('histogram')
        ed.gate_tool_var.set('quadrant')
        for xv in (1000.0, 1100.0, 1200.0):
            ed._clear_undo_pending()
            ed._on_press_quadrant_tool(SimpleNamespace(
                dblclick=True, key=None, xdata=xv, ydata=1.0))
        for name in NAMES:
            assert [v for _g, v in _one_d(ed, name)] == [1200.0], name
    finally:
        root.destroy()


def test_the_target_keeps_its_own_id_name_and_colour():
    root, ed = _editor()
    try:
        _seed(ed, NAMES[1], 'g4', dict(_thr(900.0), name='my cut'))
        ed._sample_gates[NAMES[1]]['g4']['color'] = '#abcdef'
        _place(ed, 1300.0)
        assert _one_d(ed, NAMES[1]) == [('g4', 1300.0)]
        g = ed._sample_gates[NAMES[1]]['g4']
        assert g['name'] == 'my cut'
        assert g['color'] == '#abcdef'
        assert g['id'] == 'g4'
        # A replacement takes no new id: the next new gate is g5.
        assert ed._sample_gate_seq[NAMES[1]] == 4
    finally:
        root.destroy()


def test_a_new_threshold_is_still_added_on_a_target_without_one():
    root, ed = _editor()
    try:
        _seed(ed, NAMES[1], 'g2', _thr(500.0, channel='FL2-A'))
        _place(ed, 1000.0)
        assert [v for _g, v in _one_d(ed, NAMES[1])] == [1000.0]
        # The other channel's threshold is left alone.
        assert _one_d(ed, NAMES[1], 'FL2-A') == [('g2', 500.0)]
        gid = _one_d(ed, NAMES[1])[0][0]
        assert ed._sample_gates[NAMES[1]][gid]['id'] == gid
        assert gid != 'g2'
    finally:
        root.destroy()


def test_only_the_twin_under_the_same_population_is_moved():
    """A threshold nested under 'Cells' replaces the target's nested twin,
    not its root threshold on the same channel."""
    root, ed = _editor(2)
    try:
        cells = {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                 'x0': 0.0, 'x1': 5000.0, 'y0': 0.0, 'y1': 5000.0,
                 'parent_id': None, 'name': 'Cells'}
        for name in NAMES[:2]:
            _seed(ed, name, 'g1', cells)
        _seed(ed, NAMES[1], 'g2', _thr(700.0))               # root
        _seed(ed, NAMES[1], 'g3', _thr(800.0, parent='g1'))  # in Cells
        _place(ed, 1250.0, parent='g1')
        gates = ed._sample_gates[NAMES[1]]
        assert gates['g2']['value'] == 700.0
        assert gates['g3']['value'] == 1250.0
        assert gates['g3']['parent_id'] == 'g1'
        assert len(_one_d(ed, NAMES[1])) == 2
    finally:
        root.destroy()


def test_a_rooted_orphan_replica_never_takes_over_a_root_gate():
    """The target lacks the parent population: the replica is rooted (and
    reported), and the target's own root threshold stays."""
    root, ed = _editor(2)
    try:
        _seed(ed, NAMES[0], 'g1', {
            'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
            'x0': 0.0, 'x1': 5000.0, 'y0': 0.0, 'y1': 5000.0,
            'parent_id': None, 'name': 'Cells'})
        _seed(ed, NAMES[1], 'g1', _thr(700.0))
        _place(ed, 1250.0, parent='g1')
        assert sorted(v for _g, v in _one_d(ed, NAMES[1])) == [700.0, 1250.0]
        assert ed._sample_gates[NAMES[1]]['g1']['value'] == 700.0
        assert 'lack' in ed.status_var.get()
    finally:
        root.destroy()


def test_an_interval_is_moved_too():
    root, ed = _editor(2)
    try:
        for lo, hi in ((900.0, 1000.0), (950.0, 1100.0)):
            ed._clear_undo_pending()
            ed._add_gate_multi({'kind': 'interval', 'channel': 'FL1-A',
                                'lo': lo, 'hi': hi, 'parent_id': None})
        for name in NAMES[:2]:
            ivs = [g for g in ed._sample_gates[name].values()
                   if g['kind'] == 'interval']
            assert [(g['lo'], g['hi']) for g in ivs] == [(950.0, 1100.0)]
    finally:
        root.destroy()


def test_two_d_gates_still_stack():
    """Only 1-D gates are one per channel: a second rect is a new gate."""
    root, ed = _editor(2)
    try:
        for x0 in (100.0, 200.0):
            ed._clear_undo_pending()
            ed._add_gate_multi({'kind': 'rect', 'x_channel': 'FL1-A',
                                'y_channel': 'FL2-A', 'x0': x0, 'x1': 900.0,
                                'y0': 100.0, 'y1': 900.0, 'parent_id': None})
        rects = [g for g in ed._sample_gates[NAMES[1]].values()
                 if g['kind'] == 'rect']
        assert sorted(g['x0'] for g in rects) == [100.0, 200.0]
    finally:
        root.destroy()


def test_one_undo_puts_every_target_back():
    root, ed = _editor()
    try:
        _place(ed, 1000.0)
        _place(ed, 1200.0)
        ed._undo()
        for name in NAMES:
            assert [v for _g, v in _one_d(ed, name)] == [1000.0], name
    finally:
        root.destroy()
