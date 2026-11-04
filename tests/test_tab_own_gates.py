"""Each tray tab keeps its own gates: a gate drawn, moved or deleted in one
tab leaves every other tab as it was; Copy to tab copies gates; Undo goes
back to the tab a change was made in; a session saves each tab's gates, and
one saved before tabs kept their own gives every tab a copy. Synthetic
samples and generic channels only."""
from __future__ import annotations

import json

from openflo import view_tabs as vt
from tests.test_view_tabs import _editor_or_skip, _load


def _thr(name, value, parent=None, channel='FL1-A'):
    return {'kind': 'threshold', 'name': name, 'channel': channel,
            'op': '>', 'value': float(value), 'parent_id': parent}


# ── the model ────────────────────────────────────────────────────────────

def test_gate_closure_brings_ancestors_subtree_and_operands():
    gates = {'a': {'parent_id': None}, 'b': {'parent_id': 'a'},
             'c': {'parent_id': 'b'}, 'd': {'parent_id': None},
             'e': {'parent_id': 'd'},
             'x': {'parent_id': None, 'kind': 'boolean',
                   'operands': ['e', 'c']},
             'z': {'parent_id': None}}
    assert vt.gate_closure(gates, 'b') == {'a', 'b', 'c'}
    # A boolean brings what it combines, with their ancestors.
    assert vt.gate_closure(gates, 'x') == {'x', 'e', 'd', 'c', 'b', 'a'}
    assert vt.gate_closure(gates, 'missing') == set()


def test_copy_gates_keeps_what_the_target_shows_and_replaces_the_rest():
    src = {'s1': {'g1': {'parent_id': None, 'value': 1.0},
                  'g2': {'parent_id': 'g1', 'value': 2.0},
                  'g3': {'parent_id': None, 'value': 3.0}}}
    order = {'s1': ['g1', 'g2', 'g3']}
    # The target shows g1 (its own, edited there) and holds a stale g3 it
    # does not show.
    dst = {'s1': {'g1': {'parent_id': None, 'value': 10.0},
                  'g3': {'parent_id': None, 'value': 30.0}}}
    dst_order = {'s1': ['g1', 'g3']}
    tab = vt.ViewTab('t', members={'s1': ['g1']})
    n = vt.copy_gates([('s1', 'g2'), ('s1', 'g3')], src, order, dst,
                      dst_order, shown=tab.shown_gates)
    assert n == 2
    assert dst['s1']['g1']['value'] == 10.0          # the target's own
    assert dst['s1']['g2']['value'] == 2.0           # copied in
    assert dst['s1']['g3']['value'] == 3.0            # hidden: replaced
    assert dst_order['s1'] == ['g1', 'g3', 'g2']
    # A copy, not the same gate.
    dst['s1']['g2']['value'] = 99.0
    assert src['s1']['g2']['value'] == 2.0
    # A whole sample into a tab that has none of it.
    other: dict = {}
    other_order: dict = {}
    assert vt.copy_gates([('s1', None)], src, order, other, other_order) == 3
    assert other_order == order and other == src
    assert other['s1'] is not src['s1']


def test_a_tab_saves_its_own_gates_and_an_old_one_has_none():
    ts = vt.TabSet()
    b = ts.new_tab('B', members={'s1': None})
    ts.tabs[b].gates = {'s1': {'g2': {'parent_id': None, 'name': 'Hi'},
                               'g1': {'parent_id': None, 'name': 'Pos'}}}
    ts.tabs[b].gate_order = {'s1': ['g1', 'g2']}
    ts.tabs[b].saved_gates = {'gone': [{'id': 'g5', 'name': 'Off'}]}
    d = ts.to_dict()
    assert d['tabs'][0]['sample_gates'] is None      # the active tab's: live
    saved = d['tabs'][b]['sample_gates']
    assert [g['name'] for g in saved['s1']] == ['Pos', 'Hi']   # tree order
    assert [g['id'] for g in saved['s1']] == ['g1', 'g2']
    assert saved['gone'] == [{'id': 'g5', 'name': 'Off'}]
    back = vt.TabSet.from_dict(json.loads(json.dumps(d)))
    assert back.tabs[b].saved_gates == saved
    assert back.active.gates is None and back.tabs[b].gates == {}
    # A tab written before tabs kept their own gates.
    old = dict(d['tabs'][b])
    del old['sample_gates']
    assert vt.ViewTab.from_dict(old).saved_gates == {}


def test_duplicate_copies_the_gates_and_forget_drops_them():
    ts = vt.TabSet()
    live = ({'s1': {'g1': {'parent_id': None, 'value': 1.0}}},
            {'s1': ['g1']})
    dup = ts.duplicate(0, samples=['s1'], live=live)
    assert ts.tabs[dup].gates == live[0]
    ts.tabs[dup].gates['s1']['g1']['value'] = 5.0
    assert live[0]['s1']['g1']['value'] == 1.0
    # Duplicating a tab before the active one keeps the active one active.
    ts.current = dup
    ts.active.gates = None
    ts.duplicate(0, samples=['s1'], live=live)
    assert ts.current == dup + 1 and ts.active.gates is None
    ts.tabs[1].saved_gates = {'s1': [{'id': 'g1'}]}
    ts.forget_sample('s1')
    assert ts.tabs[1].gates == {} and ts.tabs[1].saved_gates == {}


def test_rename_gate_ids_moves_parents_operands_and_order():
    gates = {'g4': {'parent_id': None}, 'g5': {'parent_id': 'g4'},
             'g6': {'parent_id': None, 'kind': 'boolean',
                    'operands': ['g5', 'g4']}}
    order = ['g4', 'g5', 'g6']
    out = vt.rename_gate_ids(gates, order, {'g4': 'g1'})
    assert set(out) == {'g1', 'g5', 'g6'}
    assert out['g5']['parent_id'] == 'g1'
    assert out['g6']['operands'] == ['g5', 'g1']
    assert order == ['g1', 'g5', 'g6']


def test_remap_gate_ids_moves_members_states_and_presort():
    t = vt.ViewTab('t', members={'s1': ['g3'], 's2': ['g3']},
                   gate_enabled={'s1': {'g3': False}},
                   presort=[('s1', 'g3'), ('s2', 'g3'), ('s1', None)])
    t.remap_gate_ids('s1', {'g3': 'g1'})
    assert t.members == {'s1': ['g1'], 's2': ['g3']}
    assert t.gate_enabled == {'s1': {'g1': False}}
    assert t.presort == [('s1', 'g1'), ('s2', 'g3'), ('s1', None)]


# ── in a real editor ─────────────────────────────────────────────────────

def _names(gates):
    return {g.get('name') for g in gates.values()}


def _tab_from_sample(ed, name):
    from openflo.tree_ids import sample_iid
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(sample_iid(name))
    ed.update()
    ed._new_tab_from_selection()
    ed._set_active_sample(name)
    return ed._tabs().current


def test_a_gate_drawn_moved_or_deleted_in_one_tab_stays_in_that_tab():
    root, ed = _editor_or_skip()
    try:
        pos = _load(ed, 's1', 1)                       # 'Pos' at 500, in All
        _load(ed, 's2', 2)
        b = _tab_from_sample(ed, 's1')
        assert b == 1
        hi = ed._add_gate(_thr('Hi', 700.0, channel='FL2-A'), audit=False)
        ed._gates[pos]['value'] = 900.0                # moved in tab B
        ed._switch_tab(0)
        assert _names(ed._sample_gates['s1']) == {'Pos'}
        assert ed._sample_gates['s1'][pos]['value'] == 500.0
        # The active-sample shortcut follows the tab's gates.
        ed._set_active_sample('s1')
        assert ed._gates is ed._sample_gates['s1']
        # A gate drawn in All never takes an id tab B holds.
        lo = ed._add_gate(_thr('Lo', 100.0, channel='SSC-A'), audit=False)
        assert lo != hi
        ed._remove_gate_cascade_in('s1', pos)          # deleted in All
        assert _names(ed._sample_gates['s1']) == {'Lo'}
        ed._switch_tab(b)
        assert _names(ed._sample_gates['s1']) == {'Pos', 'Hi'}
        assert ed._sample_gates['s1'][pos]['value'] == 900.0
        # The tree shows tab B's gates, not All's.
        from openflo.tree_ids import gate_iid
        ed._refresh_gate_list()
        assert ed.gate_tv.exists(gate_iid('s1', hi))
        assert not ed.gate_tv.exists(gate_iid('s1', lo))
        # Copy to tab copies the gate: All gets its own Hi.
        ed.gate_tv.selection_set(gate_iid('s1', hi))
        ed._copy_rows_to_tab(0)
        ed._gates[hi]['value'] = 750.0
        ed._switch_tab(0)
        assert ed._sample_gates['s1'][hi]['value'] == 700.0
        assert _names(ed._sample_gates['s1']) == {'Lo', 'Hi'}
    finally:
        root.destroy()


def test_undo_goes_back_to_the_tab_the_change_was_made_in():
    root, ed = _editor_or_skip()
    try:
        pos = _load(ed, 's1', 1)
        b = _tab_from_sample(ed, 's1')
        ed._undo_stack.clear()
        ed._undo_pending = False
        hi = ed._add_gate(_thr('Hi', 700.0, channel='FL2-A'), audit=False)
        ed._undo_pending = False
        ed._switch_tab(0)
        ed._undo()
        ts = ed._tabs()
        assert ts.current == b
        assert _names(ed._sample_gates['s1']) == {'Pos'}
        ed._switch_tab(0)
        assert _names(ed._sample_gates['s1']) == {'Pos'}
        ed._redo()
        assert ts.current == b and hi in ed._sample_gates['s1']
        # A change made in a tab since deleted is not undone into another.
        ed._undo_pending = False
        ed._set_active_sample('s1')
        ed._gates[pos]['value'] = 1.0
        ed._remove_gate_cascade_in('s1', hi)
        ed._delete_tab()
        assert ts.current == 0
        before = json.dumps(ed._sample_gates, sort_keys=True, default=str)
        ed._undo()
        assert json.dumps(ed._sample_gates, sort_keys=True,
                          default=str) == before
        assert 'deleted' in ed.status_var.get()
    finally:
        root.destroy()


def test_apply_drop_within_and_remove_reach_each_tabs_own_gates():
    root, ed = _editor_or_skip()
    try:
        from openflo.tree_ids import gate_iid
        _load(ed, 's1', 1)
        _load(ed, 's3', 3)
        b = _tab_from_sample(ed, 's1')
        hi = ed._add_gate(_thr('Hi', 700.0, channel='FL2-A'), audit=False)
        ts = ed._tabs()
        c = ts.new_tab('C', members={'s1': None, 's3': None})
        d = ts.new_tab('D', members={'s1': None})
        ed._rebuild_tab_strip()
        # Apply to other tabs copies the gate into that tab's own.
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(gate_iid('s1', hi))
        ed._apply_rows_to_tabs([c])
        assert hi in ts.tabs[c].gates['s1']
        assert ts.tabs[c].gates['s1'][hi] is not ed._sample_gates['s1'][hi]
        # So does dropping the row on a tab.
        assert ed._drop_rows_on_tab(d, [('s1', hi)])
        assert hi in ts.tabs[d].gates['s1']
        # Apply within a tab finds the population in that tab's own gates:
        # only tab C's s3 has a Hi.
        s3_hi = 'g90'
        ts.tabs[c].gates['s3'][s3_hi] = dict(
            ed._sample_gates['s1'][hi], parent_id=None)
        ts.tabs[c].gate_order['s3'].append(s3_hi)
        assert s3_hi not in ed._sample_gates.get('s3', {})
        ed._gates[hi]['enabled'] = False
        assert ed._apply_rows_within_tab(c, ['s3'], [('s1', hi)]) == 1
        assert ts.tabs[c].gate_enabled['s3'][s3_hi] is False
        # Removing a sample from the session takes it out of every tab.
        ed._remove_samples(['s3'])
        assert all('s3' not in (t.gates or {}) for t in ts.tabs)
        assert not ts.tabs[c].has_sample('s3')
        assert ts.current == b
    finally:
        root.destroy()


# ── sessions ─────────────────────────────────────────────────────────────

def _session_with_tabs(tmp_path, monkeypatch):
    """A saved session: All holds 'Mid' on sample a ('Pos' deleted, so the
    saved ids have a gap); Tab 1 holds Pos, Mid and a 'Hi' under Mid, and
    shows Mid with its subtree."""
    from tests.test_compensation_editor_apply import _load as _load_fcs
    from tests.test_compensation_editor_apply import _write_mixed
    from tests.test_session_restore_transforms import _new_editor
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001 - any Tk start failure means no GUI
        from tests.conftest import gui_unavailable
        gui_unavailable(f'Tk cannot initialise: {e}')
    root.withdraw()
    from openflo.tree_ids import gate_iid
    ed = _new_editor(root)
    _write_mixed(tmp_path / 'a.fcs')
    _load_fcs(root, ed, [('a', tmp_path / 'a.fcs')])
    ed._set_active_sample('a')
    pos = ed._add_gate(_thr('Pos', 1.0, channel='FLA-A'), audit=False)
    mid = ed._add_gate(_thr('Mid', 2.0, channel='FLB-A'), audit=False)
    ed._refresh_gate_list()
    ed.gate_tv.selection_set(gate_iid('a', mid))
    ed.update()
    ed._new_tab_from_selection()
    ed._set_active_sample('a')
    ed._add_gate(_thr('Hi', 3.0, parent=mid, channel='FLC-A'), audit=False)
    ed._switch_tab(0)
    ed._remove_gate_cascade_in('a', pos)
    path = tmp_path / 'tabs.flowsession'
    ed._write_session(str(path))
    return root, path, mid


def _by_name(gates):
    return {g['name']: gid for gid, g in gates.items()}


def test_a_session_keeps_each_tabs_own_gates(tmp_path, monkeypatch):
    from tests.test_session_restore_transforms import _restore
    root, path, _mid = _session_with_tabs(tmp_path, monkeypatch)
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert [g['name'] for g in saved['sample_gates']['a']] == ['Mid']
        tab1 = saved['view_tabs']['tabs'][1]
        assert [g['name'] for g in tab1['sample_gates']['a']] == [
            'Pos', 'Mid', 'Hi']
        ed2 = _restore(root, path, raw_first=True)
        ts = ed2._tabs()
        assert ts.names() == ['All', 'Tab 1'] and ts.current == 0
        assert _names(ed2._sample_gates['a']) == {'Mid'}
        own = ts.tabs[1].gates['a']
        ids = _by_name(own)
        assert set(ids) == {'Pos', 'Mid', 'Hi'}
        assert own[ids['Hi']]['parent_id'] == ids['Mid']
        # The tab's members follow its gates' new ids.
        assert ts.tabs[1].members == {'a': [ids['Mid']]}
        # The gate both tabs held keeps one id in both; no id is held by
        # two tabs for different gates.
        live = ed2._sample_gates['a']
        assert _by_name(live)['Mid'] == ids['Mid']
        assert set(own) & set(live) == {ids['Mid']}
        # So Copy to tab still knows Tab 1's Mid for the same gate.
        from openflo.tree_ids import gate_iid
        ed2._refresh_gate_list()
        ed2.gate_tv.selection_set(gate_iid('a', ids['Mid']))
        ed2._copy_rows_to_tab(1)
        assert sorted(g['name'] for g in own.values()) == ['Hi', 'Mid',
                                                           'Pos']
        ed2._switch_tab(1)
        assert ts.active.shown_gates('a', ed2._sample_gates['a']) == {
            ids['Mid'], ids['Hi']}
        ed2._sample_gates['a'][ids['Mid']]['value'] = 7.0
        ed2._switch_tab(0)
        assert next(iter(ed2._sample_gates['a'].values()))['value'] == 2.0
    finally:
        root.destroy()


def test_an_older_session_gives_every_tab_a_copy(tmp_path, monkeypatch):
    from tests.test_session_restore_transforms import _restore
    root, path, mid = _session_with_tabs(tmp_path, monkeypatch)
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
        for t in saved['view_tabs']['tabs']:
            t.pop('sample_gates', None)           # as tabs were saved before
        # Its tab named the shared gate by its saved id.
        saved['view_tabs']['tabs'][1]['members'] = {'a': [mid]}
        path.write_text(json.dumps(saved), encoding='utf-8')
        ed2 = _restore(root, path, raw_first=True)
        ts = ed2._tabs()
        live = ed2._sample_gates['a']
        own = ts.tabs[1].gates['a']
        assert _names(live) == _names(own) == {'Mid'}
        assert own is not live
        new_mid = _by_name(own)['Mid']
        assert ts.tabs[1].members == {'a': [new_mid]}
        ed2._switch_tab(1)
        ed2._sample_gates['a'][new_mid]['value'] = 7.0
        ed2._switch_tab(0)
        assert ed2._sample_gates['a'][new_mid]['value'] == 2.0
    finally:
        root.destroy()


def test_a_sample_loaded_in_another_tab_reaches_every_tab_apart(
        tmp_path, monkeypatch):
    from tests.test_compensation_editor_apply import _load as _load_fcs
    from tests.test_compensation_editor_apply import _write_mixed
    from tests.test_session_restore_transforms import _new_editor
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001 - any Tk start failure means no GUI
        from tests.conftest import gui_unavailable
        gui_unavailable(f'Tk cannot initialise: {e}')
    root.withdraw()
    try:
        ed = _new_editor(root)
        _write_mixed(tmp_path / 'a.fcs')
        _write_mixed(tmp_path / 'b.fcs', seed=12)
        _load_fcs(root, ed, [('a', tmp_path / 'a.fcs')])
        b = _tab_from_sample(ed, 'a')
        _load_fcs(root, ed, [('b', tmp_path / 'b.fcs')])   # in tab B
        ts = ed._tabs()
        assert ts.current == b
        assert 'b' in ts.tabs[0].gates
        assert ts.tabs[0].gates['b'] is not ed._sample_gates['b']
        ed._set_active_sample('b')
        ed._add_gate(_thr('Hi', 3.0, channel='FLA-A'), audit=False)
        assert ts.tabs[0].gates['b'] == {}
        ed._switch_tab(0)
        assert ed._sample_gates['b'] == {}
    finally:
        root.destroy()


def test_a_gate_two_other_tabs_share_keeps_one_id_after_a_reopen(
        tmp_path, monkeypatch):
    """Hi drawn in tab B and copied to tab C, then saved with All active:
    after the reopen B and C still hold it under one id, so copying it from
    B to C again leaves C with one Hi."""
    from openflo.tree_ids import gate_iid
    from tests.test_compensation_editor_apply import _load as _load_fcs
    from tests.test_compensation_editor_apply import _write_mixed
    from tests.test_session_restore_transforms import (
        _new_editor,
        _restore,
    )
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001 - any Tk start failure means no GUI
        from tests.conftest import gui_unavailable
        gui_unavailable(f'Tk cannot initialise: {e}')
    root.withdraw()

    def pick(ed, gid):
        ed._refresh_gate_list()
        ed.gate_tv.selection_set(gate_iid('a', gid))
        ed.update()

    try:
        ed = _new_editor(root)
        _write_mixed(tmp_path / 'a.fcs')
        _load_fcs(root, ed, [('a', tmp_path / 'a.fcs')])
        ed._set_active_sample('a')
        pos = ed._add_gate(_thr('Pos', 1.0, channel='FLA-A'), audit=False)
        pick(ed, pos)
        ed._new_tab_from_selection()                  # B
        ed._switch_tab(0)
        pick(ed, pos)
        ed._new_tab_from_selection()                  # C
        ed._switch_tab(1)
        ed._set_active_sample('a')
        hi = ed._add_gate(_thr('Hi', 3.0, parent=pos, channel='FLC-A'),
                          audit=False)
        pick(ed, hi)
        ed._copy_rows_to_tab(2)
        ed._switch_tab(0)
        path = tmp_path / 'three.flowsession'
        ed._write_session(str(path))
        ed2 = _restore(root, path, raw_first=True)
        ts = ed2._tabs()
        b_hi = _by_name(ts.tabs[1].gates['a'])['Hi']
        assert _by_name(ts.tabs[2].gates['a'])['Hi'] == b_hi
        ed2._switch_tab(1)
        pick(ed2, b_hi)
        ed2._copy_rows_to_tab(2)
        names = sorted(g['name'] for g in ts.tabs[2].gates['a'].values())
        assert names == ['Hi', 'Pos']
    finally:
        root.destroy()
