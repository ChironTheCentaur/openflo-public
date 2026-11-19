"""Deleting a tray tab (scope-audit T7): a tab that shows anything is
confirmed, naming the samples it shows and the gates that go with it; Undo
puts the tab back where it was, with its own gates and on/off states, and
Redo deletes it again. Synthetic samples and generic channels only."""
from __future__ import annotations

import json

from openflo import view_tabs as vt
from tests.test_tab_own_gates import _names, _tab_from_sample, _thr
from tests.test_view_tabs import _editor_or_skip, _load


def _asks(monkeypatch, answer):
    """Record every askyesno (title, message, default) and answer it."""
    from tkinter import messagebox
    asked = []

    def ask(title, message, **kw):
        asked.append((title, message, kw.get('default')))
        return answer
    monkeypatch.setattr(messagebox, 'askyesno', ask)
    return asked


def _gates_json(ed):
    return json.dumps(ed._sample_gates, sort_keys=True, default=str)


# ── the model ────────────────────────────────────────────────────────────

def test_insert_puts_a_deleted_tab_back_where_it_was():
    ts = vt.TabSet()
    b = ts.new_tab('B', members={'s1': None})
    c = ts.new_tab('C', members={'s2': None})
    ts.current = c
    ts.active.gates = ts.active.gate_order = None     # the live ones
    tab_b = ts.tabs[b]
    ts.remove(b)
    assert ts.names() == ['All', 'C'] and ts.active.name == 'C'
    # Back at its index; the active tab stays active, its gates untouched.
    tab_b.gates = None
    assert ts.insert(b, tab_b) == b
    assert ts.names() == ['All', 'B', 'C']
    assert ts.active.name == 'C' and ts.active.gates is None
    assert tab_b.gates == {} and tab_b.gate_order == {}
    # A tab made since took its name: the restored one is renamed, as
    # new_tab and duplicate keep names unique.
    ts.remove(1)
    ts.new_tab('B')
    assert ts.insert(1, tab_b) == 1
    assert ts.names() == ['All', 'B 2', 'C', 'B']
    # Never before All, never past the end.
    d = vt.ViewTab('D', members={})
    assert ts.insert(0, d) == 1 and ts.names()[1] == 'D'
    e = vt.ViewTab('E', members={})
    assert ts.insert(99, e) == 5 and ts.names()[-1] == 'E'


# ── the editor ───────────────────────────────────────────────────────────

def test_delete_tab_asks_naming_what_goes_and_no_keeps_it(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 1)                            # 'Pos' in All
        _load(ed, 's2', 2)
        b = _tab_from_sample(ed, 's1')
        ed._add_gate(_thr('Hi', 700.0, channel='FL2-A'), audit=False)
        ts = ed._tabs()
        asked = _asks(monkeypatch, False)
        ed._delete_tab()
        assert len(asked) == 1
        title, msg, default = asked[0]
        assert title == 'Delete tab' and default == 'no'
        assert "'Tab 1'" in msg and "sample 's1'" in msg
        assert '2 gate(s)' in msg and 'across' not in msg
        assert 's2' not in msg and 'Undo' in msg
        # No: nothing changed.
        assert ts.current == b and ts.names() == ['All', 'Tab 1']
        assert _names(ed._sample_gates['s1']) == {'Pos', 'Hi'}
        # Yes: the tab goes, the samples stay loaded with All's gates.
        asked = _asks(monkeypatch, True)
        ed._delete_tab()
        assert len(asked) == 1
        assert ts.names() == ['All'] and ts.current == 0
        assert set(ed._samples) == {'s1', 's2'}
        assert _names(ed._sample_gates['s1']) == {'Pos'}
        # Several samples: the gate count says how many samples it spans.
        both = ts.new_tab('Both', members={'s1': None, 's2': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(both)
        asked = _asks(monkeypatch, False)
        ed._delete_tab()
        assert '2 gate(s) across 2 samples' in asked[0][1]
    finally:
        root.destroy()


def test_an_empty_tab_goes_without_a_question(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 1)
        ts = ed._tabs()
        ts.new_tab('Empty')
        ed._rebuild_tab_strip()
        ed._switch_tab(1)
        asked = _asks(monkeypatch, False)
        ed._delete_tab()
        assert asked == [] and ts.names() == ['All']
        # A tab with nothing shown but rows waiting in its Presort zone is
        # not empty.
        p = ts.new_tab('Presort only')
        ts.tabs[p].presort = [('s1', None)]
        ed._rebuild_tab_strip()
        ed._switch_tab(p)
        ed._delete_tab()
        assert len(asked) == 1 and 'Presort' in asked[0][1]
        assert ts.names() == ['All', 'Presort only']
    finally:
        root.destroy()


def test_undo_puts_the_tab_back_with_its_gates_and_redo_deletes_it_again(
        monkeypatch):
    root, ed = _editor_or_skip()
    try:
        pos = _load(ed, 's1', 1)                      # 'Pos' at 500, in All
        _load(ed, 's2', 2)
        b = _tab_from_sample(ed, 's1')
        ed._tabs().new_tab('C', members={'s2': None})
        ed._rebuild_tab_strip()
        ts = ed._tabs()
        ed._switch_tab(b)
        hi = ed._add_gate(_thr('Hi', 700.0, channel='FL2-A'), audit=False)
        ed._gates[pos]['value'] = 900.0               # moved in tab B only
        ed._gates[hi]['enabled'] = False
        ed._sample_plot_enabled['s1'] = False
        ed._undo_stack.clear()
        ed._redo_stack.clear()
        ed._undo_pending = False
        _asks(monkeypatch, True)
        ed._delete_tab()
        # The next tab along takes over (as before).
        assert ts.names() == ['All', 'C'] and ts.active.name == 'C'
        assert 'Undo' in ed.status_var.get()
        ed._switch_tab(0)
        all_before = _gates_json(ed)
        assert ed._sample_gates['s1'][pos]['value'] == 500.0
        ed._sample_plot_enabled['s1'] = True
        # Undo: the tab is back at its index, active, with its own gates,
        # their on/off states and its plot states; All is as it was.
        ed._undo()
        assert ts.names() == ['All', 'Tab 1', 'C'] and ts.current == b
        assert ts.active.name == 'Tab 1' and ts.active.gates is None
        assert _names(ed._sample_gates['s1']) == {'Pos', 'Hi'}
        assert ed._sample_gates['s1'][pos]['value'] == 900.0
        assert ed._sample_gates['s1'][hi]['enabled'] is False
        assert ed._sample_plot_enabled['s1'] is False
        assert 'Tab 1' in ed.status_var.get()
        ed._switch_tab(0)
        assert _gates_json(ed) == all_before
        assert ed._sample_gates['s1'][pos]['value'] == 500.0
        # Redo deletes it again, from wherever the user is.
        ed._redo()
        assert ts.names() == ['All', 'C'] and ts.active.name == 'C'
        ed._switch_tab(0)
        assert _gates_json(ed) == all_before
        # And Undo brings it back once more, unchanged.
        ed._undo()
        assert ts.names() == ['All', 'Tab 1', 'C'] and ts.current == b
        assert _names(ed._sample_gates['s1']) == {'Pos', 'Hi'}
        assert ed._sample_gates['s1'][pos]['value'] == 900.0
        # Tab C kept its own gates throughout.
        ed._switch_tab(2)
        assert _names(ed._sample_gates['s2']) == {'Pos'}
        assert 's1' in ed._sample_gates
    finally:
        root.destroy()


def test_a_sample_removed_after_the_deletion_does_not_come_back_with_it(
        monkeypatch):
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 1)
        _load(ed, 's2', 2)
        ts = ed._tabs()
        b = ts.new_tab('B', members={'s1': None, 's2': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(b)
        ed._undo_stack.clear()
        ed._undo_pending = False
        _asks(monkeypatch, True)
        ed._delete_tab()
        ed._remove_samples(['s2'])
        ed._undo()
        assert ts.current == b and ts.active.name == 'B'
        assert ts.active.samples(ed._sample_order) == ['s1']
        assert 's2' not in ed._sample_gates and 's2' not in ts.active.members
    finally:
        root.destroy()


def test_undo_renames_the_tab_when_a_new_one_took_its_name(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 1)
        _tab_from_sample(ed, 's1')                    # 'Tab 1'
        ts = ed._tabs()
        ed._undo_stack.clear()
        ed._undo_pending = False
        _asks(monkeypatch, True)
        ed._delete_tab()
        ts.new_tab()                                  # named by count: 'Tab 1'
        ed._rebuild_tab_strip()
        ed._undo()
        assert ts.names() == ['All', 'Tab 1 2', 'Tab 1']
        assert ts.active.name == 'Tab 1 2'
        assert _names(ed._sample_gates['s1']) == {'Pos'}
    finally:
        root.destroy()


def test_redo_of_a_deletion_keeps_later_redo_steps_and_the_tabs_states(
        monkeypatch):
    """A gate edit made after the deletion: Undo twice, Redo twice, Undo
    twice again. Redo of the deletion must not drop the edit's redo step,
    and an on/off change made in the restored tab between Undo and Redo
    comes back with the next Undo."""
    root, ed = _editor_or_skip()
    try:
        pos = _load(ed, 's1', 1)
        b = _tab_from_sample(ed, 's1')
        ts = ed._tabs()
        ed._gates[pos]['value'] = 900.0               # tab B's own Pos
        ed._undo_stack.clear()
        ed._redo_stack.clear()
        ed._undo_pending = False
        _asks(monkeypatch, True)
        ed._delete_tab()
        assert ts.current == 0
        ed._undo_pending = False
        ed._set_active_sample('s1')
        ed._checkpoint()                              # an edit in All
        ed._gates[pos]['value'] = 600.0
        ed._undo_pending = False
        ed._undo()
        assert ed._sample_gates['s1'][pos]['value'] == 500.0
        ed._undo()
        assert ts.names() == ['All', 'Tab 1'] and ts.current == b
        assert ed._sample_gates['s1'][pos]['value'] == 900.0
        # Changed in the restored tab before redoing its deletion.
        ed._sample_plot_enabled['s1'] = False
        ed._redo()
        assert ts.names() == ['All'] and len(ed._redo_stack) == 1
        assert ed._sample_gates['s1'][pos]['value'] == 500.0
        ed._sample_plot_enabled['s1'] = True
        ed._redo()                                    # the edit, still there
        assert ed._sample_gates['s1'][pos]['value'] == 600.0
        ed._undo()
        ed._undo()
        assert ts.names() == ['All', 'Tab 1'] and ts.current == b
        assert ed._sample_gates['s1'][pos]['value'] == 900.0
        assert ed._sample_plot_enabled['s1'] is False
        ed._switch_tab(0)
        assert ed._sample_gates['s1'][pos]['value'] == 500.0
    finally:
        root.destroy()
