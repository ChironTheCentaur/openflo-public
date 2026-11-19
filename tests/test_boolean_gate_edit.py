"""Boolean gates of boolean gates, and changing a boolean gate, on the desktop.

The Boolean gate dialog offers boolean gates as operands (as FlowJo nests
them, and as gate_eval counts them), and a boolean gate's right-click
Edit ▸ Combined gates… opens the same dialog to change its choice, gates
and name. The rules are boolean_edit's (the phone's Combine dialog keeps
the same ones, mobile/src/boolean.ts): a change never offers the gate
itself or a gate resting on it; a deleted operand is said and dropped on
Save; an operand not offered (under an unconverted FlowJo ellipse) is kept;
one Ctrl+Z puts the gate back as it was. The refusal rule is the engine's
(gate_store.operand_problem, also Document.update_gate's).

Generated data only: two generic channels CD3, CD4 ~ N(0,1) independent
(the helper file's convention); A = CD3 > 0, B = CD4 > 0, C = FSC-A > 5e4.
"""
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

from openflo import boolean_edit as be
from openflo import gate_store
from openflo.document import Document, DocumentError
from tests.conftest import editor_or_skip
from tests.test_boolean_gate_tools import _load, _press, _toplevel, _walk

# ── the rules, Tk-free ──────────────────────────────────────────────────


def _set():
    """A, B, C (thresholds), AB = AND(A, B), OUT = OR(AB, C), K under A,
    an unconverted ellipse E with a gate U under it."""
    return {
        'a': {'kind': 'threshold', 'parent_id': None},
        'b': {'kind': 'threshold', 'parent_id': None},
        'c': {'kind': 'threshold', 'parent_id': None},
        'ab': {'kind': 'boolean', 'op': 'and', 'operands': ['a', 'b'],
               'parent_id': None, 'name': 'AND (2)'},
        'out': {'kind': 'boolean', 'op': 'or', 'operands': ['ab', 'c'],
                'parent_id': None, 'name': 'Out'},
        'k': {'kind': 'threshold', 'parent_id': 'a'},
        'e': {'kind': 'flowjo_ellipse', 'parent_id': None},
        'u': {'kind': 'threshold', 'parent_id': 'e'},
    }


ORDER = ['a', 'b', 'c', 'ab', 'out', 'k', 'e', 'u']


def test_combinable_offers_booleans_but_not_unconverted_ellipses():
    assert be.combinable(_set(), ORDER) == ['a', 'b', 'c', 'ab', 'out', 'k']


def test_a_changed_boolean_is_never_offered_what_rests_on_it():
    gates = _set()
    assert be.resting_on(gates, 'ab') == {'ab', 'out'}
    assert be.resting_on(gates, 'a') == {'a', 'k', 'ab', 'out'}
    assert be.combinable(gates, ORDER, be.resting_on(gates, 'ab')) == [
        'a', 'b', 'c', 'k']


def test_editing_shows_a_default_name_blank_so_it_follows_the_choice():
    gates = _set()
    assert be.editing(gates['ab']) == ('and', ['a', 'b'], '')
    assert be.editing(gates['out']) == ('or', ['ab', 'c'], 'Out')
    made = be.changed_boolean(gates['ab'], 'or', ['a', 'b', 'c'], '')
    assert made['name'] == 'OR (3)' and made['op'] == 'or'
    assert be.changed_boolean(gates['ab'], 'and', [], 'x') == (
        "Select at least one gate.")


def test_an_imported_label_follows_the_new_name():
    before = {'kind': 'boolean', 'op': 'and', 'operands': ['a'],
              'name': 'Old', 'label': 'Old', 'color': '#123456'}
    made = be.changed_boolean(before, 'not', ['a'], 'New')
    assert made['label'] == 'New' and made['color'] == '#123456'
    plain = be.changed_boolean({**before, 'label': None}, 'not', ['a'], 'N')
    assert plain['label'] == 'N'
    nolabel = {k: v for k, v in before.items() if k != 'label'}
    assert 'label' not in be.changed_boolean(nolabel, 'not', ['a'], 'N')


def test_unshown_operands_lost_dropped_kept_kept_in_order():
    lost, kept = be.unshown_operands(['gone', 'u', 'a'], {'a', 'u'}, {'a'})
    assert (lost, kept) == (['gone'], ['u'])
    assert be.kept_operands(['gone', 'u', 'a'], ['b', 'a'], ['u']) == [
        'u', 'a', 'b']
    assert be.unshown_note(0, 0) == ''
    assert be.unshown_note(1, 0) == (
        "A gate it combined was deleted; Save leaves it out.")
    assert be.unshown_note(2, 0).startswith("2 gates it combined were")
    assert be.unshown_note(0, 1).startswith("A gate it combines is not listed")
    assert be.unshown_note(0, 2).endswith("Save keeps them.")


def test_same_boolean():
    g = _set()['ab']
    assert be.same_boolean(g, dict(g))
    for k, v in (('op', 'or'), ('name', 'x'), ('operands', ['b', 'a'])):
        assert not be.same_boolean(g, {**g, k: v})


def test_operand_problem_is_the_documents_rule():
    gates = _set()
    cyc = {**gates['ab'], 'operands': ['a', 'out']}
    assert gate_store.operand_problem(gates, cyc, 'ab') == (
        "gate 'ab' cannot rest on itself or on a gate that rests on it")
    assert gate_store.operand_problem(gates, {**gates['ab'], 'operands': [
        'a', 'gone']}, 'ab') == "no gate 'gone' to combine"
    assert gate_store.operand_problem(gates, {**gates['ab'], 'operands': []},
                                      'ab') == (
        "a boolean gate needs the gates it combines ('operands')")
    assert gate_store.operand_problem(gates, {**gates['ab'], 'operands': [
        'a', 'c']}, 'ab') is None
    # The Document refuses with the same words, under the sample's name.
    doc = Document()
    rec = type('R', (), {'name': 's1', 'gates': gates})()
    with pytest.raises(DocumentError, match="s1: gate 'ab' cannot rest"):
        doc._operands_ok(rec, cyc, 'ab')


# ── the dialog ──────────────────────────────────────────────────────────

STALE = "The gates have changed since the dialog opened; nothing saved."


def _frame(seed, n=20000):
    import pandas as pd
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.uniform(1e4, 1e5, n),
                         'SSC-A': rng.uniform(1e4, 1e5, n),
                         'CD3': rng.normal(0, 1, n), 'CD4': rng.normal(0, 1, n)})


@pytest.fixture
def ed():
    root, ed = editor_or_skip()
    try:
        _load(ed, 's1', _frame(11))
        ed._set_active_sample('s1')
        yield ed
    finally:
        root.destroy()


def _thr(ed, ch, v, nm, parent=None):
    return ed._add_gate({'kind': 'threshold', 'channel': ch, 'value': v,
                         'name': nm, 'parent_id': parent})


def _bool(ed, op, operands, nm):
    return ed._add_gate({'kind': 'boolean', 'op': op, 'operands': operands,
                         'name': nm, 'parent_id': None})


def _mask(ed, gid):
    from openflo import gate_eval
    return np.asarray(gate_eval.cumulative_gate_mask(
        ed._sample_gates['s1'], gid, ed._samples['s1'].data))


def _truth(ed):
    df = ed._samples['s1'].data
    return (df.CD3.values > 0, df.CD4.values > 0,
            df['FSC-A'].values > 5e4)


def _boxes(dlg):
    return {w.cget('text'): w for w in _walk(dlg)
            if isinstance(w, ttk.Checkbutton)}


def _ticked(dlg):
    return [t for t, w in _boxes(dlg).items() if w.instate(['selected'])]


def _label(ed, gid):
    from openflo.gate_text import describe_gate
    return describe_gate(ed._gates[gid],
                         transforms=ed._display_transforms('s1'))


def _fill(dlg, op=None, toggle=(), name=None):
    for w in _walk(dlg):
        if op and isinstance(w, ttk.Radiobutton) and w.cget('text') == op:
            w.invoke()
    boxes = _boxes(dlg)
    for t in toggle:
        boxes[t].invoke()
    if name is not None:
        (e,) = [w for w in _walk(dlg) if isinstance(w, ttk.Entry)]
        e.delete(0, 'end')
        e.insert(0, name)


def _settle(ed):
    """End the Tk event: the gates added so far are one undo step, and
    the next change takes its own (editor_undo._checkpoint coalesces
    within one event)."""
    ed.update_idletasks()
    assert not ed._undo_pending


def _change(ed, gid, **kw):
    _settle(ed)
    ed._open_boolean_dialog('s1', gid)
    dlg = _toplevel(ed, 'Change boolean gate')
    _fill(dlg, **kw)
    _press(dlg, 'Save')


def test_a_new_boolean_combines_a_boolean(ed):
    a, b, c = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B'), \
        _thr(ed, 'FSC-A', 5e4, 'C')
    ab = _bool(ed, 'and', [a, b], 'AB')
    ed._open_boolean_dialog()
    dlg = _toplevel(ed, 'Boolean gate')
    _fill(dlg, op='NOT', toggle=[_label(ed, ab)], name='not AB')
    before = set(ed._gates)
    _press(dlg, 'Create')
    (nab,) = set(ed._gates) - before
    x, y, _z = _truth(ed)
    assert ed._gates[nab]['operands'] == [ab]
    assert (_mask(ed, nab) == ~(x & y)).all()
    assert c in ed._gates


def test_change_dialog_shows_the_gate_and_leaves_out_what_rests_on_it(ed):
    a, b, c = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B'), \
        _thr(ed, 'FSC-A', 5e4, 'C')
    ab = _bool(ed, 'and', [a, b], 'AND (2)')
    out = _bool(ed, 'or', [ab, c], 'Out')
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, "Change boolean gate 'AND (2)'")
    assert list(_boxes(dlg)) == [_label(ed, g) for g in (a, b, c)]
    assert _ticked(dlg) == [_label(ed, a), _label(ed, b)]
    radios = [w for w in _walk(dlg) if isinstance(w, ttk.Radiobutton)
              and w.instate(['selected'])]
    assert [w.cget('text') for w in radios] == ['AND']
    (e,) = [w for w in _walk(dlg) if isinstance(w, ttk.Entry)]
    assert e.get() == ''                     # default name, follows choice
    assert not any(isinstance(w, ttk.Button) and w.cget('text') == 'Create'
                   for w in _walk(dlg))
    dlg.destroy()
    # The outer one may combine the inner one, but not itself.
    ed._open_boolean_dialog('s1', out)
    dlg = _toplevel(ed, "Change boolean gate 'Out'")
    assert list(_boxes(dlg)) == [_label(ed, g) for g in (a, b, c, ab)]
    assert [w.cget('text') for w in _walk(dlg)
            if isinstance(w, ttk.Radiobutton) and w.instate(['selected'])] == [
        'OR']
    (e,) = [w for w in _walk(dlg) if isinstance(w, ttk.Entry)]
    assert e.get() == 'Out'
    dlg.destroy()


def test_change_saves_counts_and_one_undo_puts_it_back(ed):
    a, b, c = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B'), \
        _thr(ed, 'FSC-A', 5e4, 'C')
    ab = _bool(ed, 'and', [a, b], 'AND (2)')
    out = _bool(ed, 'not', [ab], 'Out')
    x, y, z = _truth(ed)
    was = dict(ed._gates[ab])
    _settle(ed)
    depth = len(ed._undo_stack)
    _change(ed, ab, op='OR', toggle=[_label(ed, b), _label(ed, c)])
    g = ed._gates[ab]
    assert (g['op'], g['operands'], g['name']) == ('or', [a, c], 'OR (2)')
    assert g['color'] == was['color']
    assert "Boolean gate 'OR (2)' changed." in ed.status_var.get()
    last = ed._audit_log.entries()[-1]
    assert (last['action'], last['details']['id']) == ('gate.boolean', ab)
    assert (last['details']['op'], last['details']['operands'],
            last['details']['name']) == ('or', [a, c], 'OR (2)')
    assert (_mask(ed, ab) == (x | z)).all()
    assert (_mask(ed, out) == ~(x | z)).all()   # the nested one follows
    assert len(ed._undo_stack) == depth + 1
    ed._undo()
    assert ed._sample_gates['s1'][ab] == was
    assert (_mask(ed, out) == ~(x & y)).all()
    ed._redo()
    assert ed._sample_gates['s1'][ab]['operands'] == [a, c]


def test_an_unchanged_save_takes_no_undo_step(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    _settle(ed)
    depth = len(ed._undo_stack)
    _change(ed, ab)
    assert len(ed._undo_stack) == depth
    assert ed.status_var.get() == "No change to 'Mine'."


def test_unticking_every_gate_is_refused_and_nothing_changes(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    _fill(dlg, toggle=[_label(ed, a), _label(ed, b)])
    _press(dlg, 'Save')
    assert ed.status_var.get() == "Select at least one gate."
    assert ed._gates[ab]['operands'] == [a, b]
    assert dlg.winfo_exists()
    dlg.destroy()


def test_a_deleted_operand_is_said_dropped_on_save_and_undo_keeps_it(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    ed._remove_gate_cascade_in('s1', b)
    assert not _mask(ed, ab).any()               # lost: selects nothing
    lost = dict(ed._gates[ab])
    _settle(ed)
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    notes = [w.cget('text') for w in _walk(dlg) if isinstance(w, ttk.Label)]
    assert "A gate it combined was deleted; Save leaves it out." in notes
    assert _ticked(dlg) == [_label(ed, a)]
    _press(dlg, 'Save')
    x, _y, _z = _truth(ed)
    assert ed._gates[ab]['operands'] == [a]
    assert (_mask(ed, ab) == x).all()            # counts again
    ed._undo()
    assert ed._sample_gates['s1'][ab] == lost    # still naming the lost one
    assert not _mask(ed, ab).any()


def test_an_operand_under_an_unconverted_ellipse_is_kept(ed):
    a = _thr(ed, 'CD3', 0, 'A')
    e = ed._add_gate({'kind': 'flowjo_ellipse', 'x_channel': 'CD3',
                      'y_channel': 'CD4', 'name': 'Blob', 'parent_id': None})
    u = _thr(ed, 'CD4', 0, 'U', parent=e)
    ab = _bool(ed, 'and', [u, a], 'Mine')
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    assert list(_boxes(dlg)) == [_label(ed, a)]
    notes = [w.cget('text') for w in _walk(dlg) if isinstance(w, ttk.Label)]
    assert any(t.startswith("A gate it combines is not listed") for t in notes)
    _fill(dlg, op='OR')
    _press(dlg, 'Save')
    assert ed._gates[ab]['operands'] == [u, a]
    assert ed._gates[ab]['op'] == 'or'


def test_a_change_the_engine_refuses_is_not_saved(ed, monkeypatch):
    """Should the offered list ever let a cycle through, Save asks the
    engine's rule and refuses."""
    import openflo.boolean_edit as mod
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    out = _bool(ed, 'or', [ab], 'Out')
    monkeypatch.setattr(mod, 'resting_on', lambda gates, target: {target})
    _settle(ed)
    depth = len(ed._undo_stack)
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    _fill(dlg, toggle=[_label(ed, out)])
    _press(dlg, 'Save')
    assert ed.status_var.get().startswith("Not saved: gate ")
    assert ed._gates[ab]['operands'] == [a, b]
    assert len(ed._undo_stack) == depth
    dlg.destroy()


def test_change_on_a_sample_that_is_not_active(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    _load(ed, 's2', _frame(12))
    ed._set_active_sample('s2')
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, "Change boolean gate 'Mine' — s1")
    _fill(dlg, op='OR')
    _press(dlg, 'Save')
    assert ed._sample_gates['s1'][ab]['op'] == 'or'
    assert ed._active_sample == 's2'


def test_change_refuses_a_gate_that_is_not_boolean(ed):
    a = _thr(ed, 'CD3', 0, 'A')
    ed._open_boolean_dialog('s1', a)
    assert ed.status_var.get() == "Select a boolean gate first."
    assert not [w for w in _walk(ed.master) if isinstance(w, tk.Toplevel)
                and w.title().startswith('Change boolean gate')]


def test_boolean_gate_menu_offers_combined_gates(monkeypatch, menu_popups):
    from openflo.tree_ids import gate_iid
    from tests.conftest import (
        invoke_menu_path,
        load_menu_sample,
        menu_labels,
        right_click_row,
        submenu,
    )
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        g1 = load_menu_sample(ed, 's1', 0)
        ed._set_active_sample('s1')
        bid = ed._add_gate({'kind': 'boolean', 'op': 'not', 'operands': [g1],
                            'name': 'NOT', 'parent_id': None})
        ed._refresh_gate_list()
        calls = []
        monkeypatch.setattr(ed, '_open_boolean_dialog',
                            lambda *a: calls.append(a))
        menu = right_click_row(ed, monkeypatch, menu_popups,
                               gate_iid('s1', bid))
        assert "Combined gates…" in menu_labels(submenu(menu, "Edit"))
        invoke_menu_path(menu, "Edit", "Combined gates…")
        assert calls == [('s1', bid)]
        menu = right_click_row(ed, monkeypatch, menu_popups,
                               gate_iid('s1', g1))
        assert "Combined gates…" not in menu_labels(submenu(menu, "Edit"))
    finally:
        root.destroy()


def test_a_gate_gone_while_the_dialog_was_open_is_not_saved(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    _fill(dlg, op='OR')
    ed._remove_gate_cascade_in('s1', ab)
    _press(dlg, 'Save')
    assert ab not in ed._sample_gates['s1']
    assert ed.status_var.get() == STALE


def _ctrl_z_on(ed, widget):
    """Ctrl+Z pressed on `widget` (a check box of the modal dialog),
    dispatched as Tk does along its bindtags: the first tag binding
    <Control-z> runs, and the editor's own is on 'all' (bind_all). A
    synthetic key event needs keyboard focus, which xvfb without a window
    manager does not give. (From the #413 review's probe.)"""
    from types import SimpleNamespace
    for tag in widget.bindtags():
        if tag == 'all':
            assert ed.bind_all('<Control-z>')
            ed._undo(SimpleNamespace(widget=widget))
            break
        if str(widget.tk.call('bind', tag, '<Control-z>')):
            break                # a binding before 'all' (it returns 'break')
    ed.update()


def test_ctrl_z_in_the_dialog_does_not_undo_behind_it(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    _settle(ed)
    c = _thr(ed, 'FSC-A', 5e4, 'C')
    _settle(ed)
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    _fill(dlg, op='OR')
    _ctrl_z_on(ed, _boxes(dlg)[_label(ed, a)])
    assert c in ed._sample_gates['s1']           # nothing undone
    _press(dlg, 'Save')
    assert ed._sample_gates['s1'][ab]['op'] == 'or'
    assert ed.status_var.get() == "Boolean gate 'Mine' changed."
    # The Create dialog too.
    ed._open_boolean_dialog()
    dlg = _toplevel(ed, 'Boolean gate')
    _ctrl_z_on(ed, _boxes(dlg)[_label(ed, a)])
    assert ed._sample_gates['s1'][ab]['op'] == 'or'
    dlg.destroy()


def test_an_undo_behind_the_change_dialog_is_not_saved_over(ed):
    """However an Undo gets behind the dialog, Save does not write into
    the gate set it replaced, nor say it changed anything."""
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Mine')
    _settle(ed)
    _thr(ed, 'FSC-A', 5e4, 'C')
    _settle(ed)
    ed._open_boolean_dialog('s1', ab)
    dlg = _toplevel(ed, 'Change boolean gate')
    _fill(dlg, op='OR')
    ed._undo()                                   # takes C away, new dicts
    depth = len(ed._undo_stack)
    _press(dlg, 'Save')
    assert ed._sample_gates['s1'][ab]['op'] == 'and'
    assert ed.status_var.get() == STALE
    assert len(ed._undo_stack) == depth
    assert not dlg.winfo_exists()


def test_an_undo_behind_the_create_dialog_is_not_created_over(ed):
    a = _thr(ed, 'CD3', 0, 'A')
    _settle(ed)
    c = _thr(ed, 'FSC-A', 5e4, 'C')
    _settle(ed)
    ed._open_boolean_dialog()
    dlg = _toplevel(ed, 'Boolean gate')
    _fill(dlg, toggle=[_label(ed, a), _label(ed, c)])
    ed._undo()                                   # takes C away
    before = set(ed._gates)
    _press(dlg, 'Create')
    assert set(ed._gates) == before
    assert ed.status_var.get() == STALE


def test_a_boolean_under_a_gate_keeps_its_place_when_changed(ed):
    """Combination gates make booleans with a parent: a change keeps it."""
    p = _thr(ed, 'SSC-A', 5e4, 'P')
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = ed._add_gate({'kind': 'boolean', 'op': 'and', 'operands': [a, b],
                       'name': 'Mine', 'parent_id': p})
    _change(ed, ab, op='OR')
    x, y, _z = _truth(ed)
    inside = ed._samples['s1'].data['SSC-A'].values > 5e4
    assert ed._gates[ab]['parent_id'] == p
    assert (_mask(ed, ab) == (inside & (x | y))).all()


def test_the_gate_list_shows_the_new_name_and_the_old_after_undo(ed):
    a, b = _thr(ed, 'CD3', 0, 'A'), _thr(ed, 'CD4', 0, 'B')
    ab = _bool(ed, 'and', [a, b], 'Old name')
    ed._refresh_gate_list()

    def row():
        return ed.gate_tv.item(ed._gate_iid('s1', ab), 'text')
    assert 'Old name' in row()
    _change(ed, ab, name='New name')
    assert 'New name' in row() and 'Old name' not in row()
    ed._undo()
    assert 'Old name' in row() and 'New name' not in row()
