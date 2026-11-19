"""The real gate editor, driven through a short gate-editing session, keeps
its gate sets the way gate_store says: ids in the id order, per-sample and
active-sample counters in step, and an audit entry for each new gate but not
for a 1-D gate moved in place.

These pin the editor's call sites (add, 1-D replace, boolean, multi-sample
add, copy to samples, move, copy/cut and paste, undo/redo, combination gate,
clear) rather than gate_store itself, which tests/test_gate_store.py covers.
GUI test: a hidden editor, skipped only where Tk cannot start
(conftest.editor_or_skip). Generated data and generic channel names only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import tree_ids
from tests.conftest import editor_or_skip

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


@pytest.fixture
def ed():
    root, editor = editor_or_skip()
    editor._ask_clear_all = lambda total, n, n_ac: False   # keep auto-clean
    for i, name in enumerate(('s1', 's2', 's3')):
        rng = np.random.default_rng(i + 1)
        df = pd.DataFrame({'FSC-A': rng.uniform(1e4, 1e5, 500),
                           'SSC-A': rng.uniform(1e4, 1e5, 500),
                           'FL1-A': rng.normal(0, 1, 500),
                           'FL2-A': rng.normal(0, 1, 500)})
        editor._samples[name] = SimpleNamespace(
            name=name, path=f'{name}.fcs', data=df,
            fluor_channels=['FL1-A', 'FL2-A'],
            channel_labels={c: c for c in CHANNELS})
        editor._sample_order.append(name)
        editor._sample_colors[name] = '#1f77b4'
        editor._sample_trial[name] = 'T'
        editor._sample_plot_enabled[name] = True
    editor._trial_order.append('T')
    editor._channels = list(CHANNELS)
    editor._channel_labels = {c: c for c in CHANNELS}
    editor._populate_channel_combos()
    editor._set_active_sample('s1')
    try:
        yield editor
    finally:
        editor.destroy()
        root.destroy()


def _num(gid):
    return int(gid[1:])


def _consistent(ed):
    """Every sample's id order lists exactly its gates, once each; its counter
    is past every id it holds; the active sample's counter and shortcuts are
    the per-sample ones. Lets Tk run its idle calls, as between two user
    gestures, so each gesture is its own undo step."""
    ed.update_idletasks()
    for name, gates in ed._sample_gates.items():
        order = ed._sample_gate_order.get(name, [])
        assert sorted(order) == sorted(gates), (name, order, list(gates))
        assert len(order) == len(set(order)), (name, order)
        if gates:
            assert ed._sample_gate_seq[name] >= max(map(_num, gates)), name
    a = ed._active_sample
    assert ed._gate_id_seq == ed._sample_gate_seq.get(a, 0)
    assert ed._gates is ed._sample_gates[a]
    assert ed._gate_id_order is ed._sample_gate_order[a]


def _added_ids(ed):
    """(sample, id) of every gate.add audit entry, in order."""
    return [(e['details']['sample'], e['details']['id'])
            for e in ed._audit_log.entries() if e['action'] == 'gate.add']


def _select(ed, *iids):
    ed.update_idletasks()
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(iids)


def _thr(ch, value, parent=None, **kw):
    return {'kind': 'threshold', 'channel': ch, 'value': value,
            'parent_id': parent, **kw}


def _rect(parent=None, **kw):
    return {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
            'x0': 2e4, 'x1': 8e4, 'y0': 2e4, 'y1': 8e4,
            'parent_id': parent, **kw}


def test_a_gate_editing_session_keeps_ids_order_counters_and_audit(ed):
    gi = tree_ids.gate_iid
    # Add: new ids from the counter, last in the order, audited.
    a = ed._add_gate(_thr('FL1-A', 0.0, name='FL1+'))
    b = ed._add_gate({'kind': 'interval', 'channel': 'FL2-A', 'lo': -1,
                      'hi': 1, 'name': 'FL2 mid', 'parent_id': a})
    assert (a, b) == ('g1', 'g2')
    assert ed._gate_id_order == ['g1', 'g2']
    assert _added_ids(ed) == [('s1', 'g1'), ('s1', 'g2')]
    _consistent(ed)

    # A threshold on the interval's channel and parent moves it in place:
    # same id, the user's name and off state kept, no id spent, NOT audited.
    ed._gates[b]['enabled'] = False
    assert ed._add_gate(_thr('FL2-A', 0.5, parent=a, color='#000000')) == b
    assert ed._gates[b]['kind'] == 'threshold'
    assert ed._gates[b]['value'] == 0.5
    assert (ed._gates[b]['name'], ed._gates[b]['enabled']) == ('FL2 mid',
                                                             False)
    assert ed._gate_id_order == ['g1', 'g2']
    assert ed._gate_id_seq == 2
    assert _added_ids(ed) == [('s1', 'g1'), ('s1', 'g2')]
    _consistent(ed)

    c = ed._add_gate(_rect(name='cells'))
    d = ed._add_gate({'kind': 'boolean', 'op': 'and', 'operands': [a, c],
                      'name': 'A and C', 'parent_id': None})
    assert (c, d) == ('g3', 'g4')
    assert ed._gate_id_order == ['g1', 'g2', 'g3', 'g4']
    assert _added_ids(ed)[-2:] == [('s1', 'g3'), ('s1', 'g4')]
    _consistent(ed)

    # One gate on several samples: each gets its own id and order entry.
    ed._gate_targets_for_new = lambda: ['s1', 's2', 's3']
    m = ed._add_gate_multi(_thr('FSC-A', 3e4, parent=c))
    del ed._gate_targets_for_new
    assert m == 'g5'
    assert ed._sample_gate_order['s2'] == ['g1']
    assert ed._sample_gate_order['s3'] == ['g1']
    assert ed._sample_gate_seq['s2'] == ed._sample_gate_seq['s3'] == 1
    _consistent(ed)

    # Copy every gate to s2 (appended after its own) and move one to s3.
    ed._copy_gates_to(['s2'], append=True)
    assert len(ed._sample_gates['s2']) == 6
    _consistent(ed)
    assert ed._move_gate_to_sample('s1', b, 's3', None) == 'g2'
    assert b not in ed._gates
    assert ed._sample_gate_order['s3'] == ['g1', 'g2']
    _consistent(ed)

    # Move a gate INTO the active sample: its id comes from the active
    # sample's counter, and the active shortcut counter follows.
    seq_s1 = ed._sample_gate_seq['s1']
    moved = ed._move_gate_to_sample('s2', 'g1', 's1', None)
    assert moved == f'g{seq_s1 + 1}'
    assert ed._gate_id_order[-1] == moved
    assert ed._gate_id_seq == seq_s1 + 1
    _consistent(ed)

    # Copy a subtree and paste it under the active sample's root: fresh ids,
    # appended to the order.
    _select(ed, gi('s1', c))
    ed._on_copy()
    _select(ed, tree_ids.sample_iid('s1'))
    before = list(ed._gate_id_order)
    ed._paste_gate_tree()
    new = [g for g in ed._gate_id_order if g not in before]
    assert ed._gate_id_order[:len(before)] == before
    assert len(new) == 2 and new == sorted(new, key=_num)
    assert _num(new[0]) > max(map(_num, before))
    _consistent(ed)

    # Cut a gate and paste it back: the same ids return, at the end.
    _select(ed, gi('s1', a))
    ed._on_cut()
    assert a not in ed._gates and a not in ed._gate_id_order
    _consistent(ed)
    _select(ed, tree_ids.sample_iid('s1'))
    ed._paste_gate_tree()
    assert ed._gate_id_order[-1] == a
    _consistent(ed)

    # Undo/redo restore whole states, shortcuts still bound.
    after_paste = (dict(ed._gates), list(ed._gate_id_order))
    ed._undo()
    assert a not in ed._gates
    _consistent(ed)
    ed._redo()
    assert (dict(ed._gates), list(ed._gate_id_order)) == after_paste
    _consistent(ed)

    # Combination gates, on another sample and on the active one.
    seq_s2 = ed._sample_gate_seq['s2']
    g_s2 = ed._add_gate_to_sample('s2', _thr('FL1-A', 0.1))
    assert g_s2 == f'g{seq_s2 + 1}'
    assert ed._sample_gate_order['s2'][-1] == g_s2
    _consistent(ed)
    g_s1 = ed._add_gate_to_sample('s1', _thr('FL1-A', 0.2))
    assert ed._gate_id_order[-1] == g_s1
    assert ed._gate_id_seq == _num(g_s1)
    _consistent(ed)
    # ... and the next plain add does not reuse the combination's id.
    nxt = ed._add_gate(_rect(name='after combination'))
    assert _num(nxt) == _num(g_s1) + 1
    _consistent(ed)

    # Clear one sample's row: gates and order emptied, counter kept.
    seq_s2 = ed._sample_gate_seq['s2']
    _select(ed, tree_ids.sample_iid('s2'))
    ed._clear_selected_gate()
    assert ed._sample_gates['s2'] == {}
    assert ed._sample_gate_order['s2'] == []
    assert ed._sample_gate_seq['s2'] == seq_s2
    _consistent(ed)
    assert ed._add_gate_to_sample('s2', _rect()) == f'g{seq_s2 + 1}'
    _consistent(ed)

    # Clear all: every sample empty, the active shortcuts still bound.
    ed._clear_all()
    for name in ('s1', 's2', 's3'):
        assert ed._sample_gates[name] == {}
        assert ed._sample_gate_order[name] == []
    _consistent(ed)
    n_added = len(_added_ids(ed))
    gid = ed._add_gate(_thr('FL1-A', 3.0))
    assert ed._gate_id_order == [gid]
    assert len(_added_ids(ed)) == n_added + 1
    _consistent(ed)


def test_a_new_gate_whose_counter_id_is_taken_is_still_audited(ed):
    """The counter, not the dict's size, says a gate is new: a gate stored
    under an id the counter hands out is audited even when that id was
    already a gate (a counter left behind its ids)."""
    first = ed._add_gate(_rect(name='first'))
    ed._gate_id_seq = 0
    ed._sample_gate_seq['s1'] = 0
    again = ed._add_gate(_rect(name='second'))
    assert again == first
    assert ed._gates[first]['name'] == 'second'
    assert _added_ids(ed) == [('s1', first), ('s1', again)]


def test_the_loader_delegate_passes_labels_and_reads_qc_only_unstaged(
        tmp_path, monkeypatch):
    """The editor's _prepared_fcs_sample hands sample_prep its labels and the
    Preferences QC setting, which it reads only when nothing was staged."""
    from openflo import prefs
    from tests._fcs_writer import write_float_fcs
    rng = np.random.default_rng(5)
    ev = np.column_stack([rng.uniform(1e4, 1e5, (400, 4)),
                          np.arange(400) / 10.0]).astype(np.float32)
    path = write_float_fcs(tmp_path / 'sample_1.fcs', ev,
                           CHANNELS + ['Time'])
    root, editor = editor_or_skip()
    try:
        editor.labels_str = 'FL2-A=M2'
        monkeypatch.setattr(prefs, 'load_qc_enabled', lambda: True)
        s = editor._prepared_fcs_sample(path)
        assert s.qc_applied is True
        assert s.channel_labels['FL2-A'] == 'M2'

        def refuse():
            raise AssertionError('read the QC preference for a staged choice')
        monkeypatch.setattr(prefs, 'load_qc_enabled', refuse)
        staged = editor._prepared_fcs_sample(path, {'qc': False})
        assert staged.qc_applied is False
    finally:
        editor.destroy()
        root.destroy()


def test_population_imports_and_a_template_overwrite_keep_the_order(ed):
    """Imported populations (a label column, cell-cycle phases) and a
    template applied over a sample's gates leave the id order listing
    exactly the gates."""
    ed._add_gate(_rect(name='kept until the template'))
    ed._import_population_column(
        group_for='lbl', disp='labels', items_for=lambda name: ['a', 'b'],
        is_member=lambda g: g.get('channel') == 'lbl',
        member_key=lambda g: g.get('value'),
        make_gate=lambda name, it, i, grp: {
            'kind': 'category', 'channel': 'lbl', 'value': it,
            'name': it, 'color': '#123456'},
        none_msg='none', noun='labels')
    for name in ('s1', 's2', 's3'):
        kinds = [ed._sample_gates[name][g]['kind']
                 for g in ed._sample_gate_order[name]]
        assert kinds[-3:] == ['group', 'category', 'category'], name
    _consistent(ed)

    phases = np.array(['G1', 'S', 'G2M', 'NA'] * 125)
    ed._samples['s2'].data['cell_cycle'] = phases
    n_before = len(ed._sample_gate_order['s2'])
    ed._import_cell_cycle('s2')
    order = ed._sample_gate_order['s2']
    assert len(order) == n_before + 5            # a group and four phases
    assert ed._sample_gates['s2'][order[n_before]]['kind'] == 'group'
    _consistent(ed)

    seq = ed._sample_gate_seq['s1']
    ed._apply_template_to_sample('s1', [_rect(name='from template')],
                                 overwrite=True)
    assert ed._gate_id_order == [f'g{seq + 1}']
    assert ed._gates[f'g{seq + 1}']['name'] == 'from template'
    _consistent(ed)


def test_the_slider_finds_the_interval_gate_on_its_channel(ed):
    """The histogram slider moves the 1-D gate already on its channel: an
    interval gate as well as a threshold, and none on another channel."""
    ed._add_gate(_rect(name='not 1-D'))
    gid = ed._add_gate({'kind': 'interval', 'channel': 'FL2-A', 'lo': -1.0,
                        'hi': 1.0, 'parent_id': None, 'name': 'FL2 mid'})
    assert ed._find_1d_gate_id_for('FL2-A') == gid
    assert ed._find_1d_gate_id_for('FL1-A') is None
    thr = ed._add_gate(_thr('FL1-A', 0.5, name='FL1 pos'))
    assert ed._find_1d_gate_id_for('FL1-A') == thr
