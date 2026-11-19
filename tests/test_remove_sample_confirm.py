"""Removing samples from the session asks first, and the Delete key on a
trial row only clears gates.

* Remove ▸ From session (all tabs) and the tray's Remove button unload the
  sample and its whole gate tree, which Undo cannot bring back. They used to
  do it in one click; now they ask, naming the sample(s) and the gate count
  and pointing to Remove ▸ From this tab, as Remove trial already did. A No
  leaves everything as it was.
* The Delete key is the accelerator Edit > Delete gate shows. On a trial row
  the menu entry clears the trial's gates (samples kept, undoable) but the
  key removed the trial's samples. Both now clear the gates.

Synthetic samples and generic channels only; no dialog is shown (askyesno is
replaced by a recorder) and the editor stays withdrawn.
"""
from __future__ import annotations

import pytest

from tests.conftest import (
    editor_or_skip,
    invoke_menu_path,
    load_menu_sample,
    right_click_row,
)


@pytest.fixture
def editor():
    """s1, s2 (trial T) and s3 (trial U), each with one gate, plus a second
    gate on s1; tab A = {s1, s2} is open."""
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        gids = {}
        for i, (n, tr) in enumerate((('s1', 'T'), ('s2', 'T'),
                                     ('s3', 'U'))):
            gids[n] = load_menu_sample(ed, n, i, trial=tr)
        ed._set_active_sample('s1')
        ed._add_gate({'kind': 'threshold', 'name': 'Hi', 'channel': 'FL1-A',
                      'op': '>', 'value': 800.0, 'parent_id': gids['s1']},
                     audit=False)
        ed._refresh_gate_list()
        ts = ed._tabs()
        a = ts.new_tab('A', members={'s1': None, 's2': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(a)
        ed.update()            # end the setup's undo coalescing (after_idle)
        yield ed
    finally:
        root.destroy()


class _Asked(list):
    """(title, message) per askyesno; ``defaults`` the default button each
    was given."""

    def __init__(self):
        super().__init__()
        self.defaults = []


def _asks(monkeypatch, answer):
    """Record every askyesno (title, message) and answer ``answer``."""
    from tkinter import messagebox
    asked = _Asked()
    defaults = asked.defaults

    def ask(title=None, message=None, **k):
        asked.append((title, message))
        defaults.append(k.get('default'))
        return answer
    monkeypatch.setattr(messagebox, 'askyesno', ask)
    return asked


def _state(ed):
    return (sorted(ed._samples),
            {n: sorted(g) for n, g in ed._sample_gates.items()},
            dict(ed._sample_trial))


# ── Remove from session asks first ─────────────────────────────────────────

def test_remove_from_session_menu_asks_naming_sample_and_gates(
        editor, monkeypatch, menu_popups):
    from tkinter import messagebox

    from openflo.tree_ids import sample_iid
    ed = editor
    before = _state(ed)
    asked = _asks(monkeypatch, False)
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    invoke_menu_path(menu, "Remove", "From session (all tabs)…")
    assert len(asked) == 1
    title, msg = asked[0]
    assert title == "Remove sample"
    assert "'s1'" in msg
    # s1 holds two gates (Pos and its child Hi), counted independently.
    assert len(ed._sample_gates['s1']) == 2
    assert "2 gate(s)" in msg
    assert "can't be undone" in msg
    assert "From this tab" in msg
    # It can't be undone, so Enter means No.
    assert asked.defaults == [messagebox.NO]
    # No: nothing removed.
    assert _state(ed) == before


def test_remove_from_session_yes_removes_only_the_selected_sample(
        editor, monkeypatch, menu_popups):
    from openflo.tree_ids import sample_iid
    ed = editor
    asked = _asks(monkeypatch, True)
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s2'))
    invoke_menu_path(menu, "Remove", "From session (all tabs)…")
    assert len(asked) == 1 and "'s2'" in asked[0][1]
    assert "1 gate(s)" in asked[0][1]
    assert sorted(ed._samples) == ['s1', 's3']
    assert 's2' not in ed._sample_gates
    assert "from the session" in ed.status_var.get()


def test_remove_button_with_two_samples_names_both_and_sums_gates(
        editor, monkeypatch):
    from openflo.tree_ids import sample_iid
    ed = editor
    ed._switch_tab(0)                      # All: no "From this tab" there
    before = _state(ed)
    asked = _asks(monkeypatch, False)
    ed.gate_tv.selection_set((sample_iid('s1'), sample_iid('s3')))
    ed._remove_selected()                  # the tray's Remove button
    assert len(asked) == 1
    msg = asked[0][1]
    assert "2 samples" in msg and "'s1'" in msg and "'s3'" in msg
    want = len(ed._sample_gates['s1']) + len(ed._sample_gates['s3'])
    assert want == 3 and f"{want} gate(s)" in msg
    assert "From this tab" not in msg      # not offered on the All tab
    assert _state(ed) == before


def test_remove_of_rows_still_loading_says_so_and_asks_nothing(
        editor, monkeypatch):
    """Every selected sample row still loading: nothing to remove yet, no
    question, and the status line says why nothing happened."""
    from openflo.tree_ids import sample_iid
    ed = editor
    ed._switch_tab(0)
    before = _state(ed)
    asked = _asks(monkeypatch, True)
    ed.gate_tv.insert('', 'end', iid=sample_iid('s9'), text='s9')
    ed.gate_tv.selection_set(sample_iid('s9'))
    ed.status_var.set('')
    ed._remove_selected()
    assert asked == []
    assert _state(ed) == before
    assert ed.status_var.get() == ("The selected sample(s) are still loading; "
                                   "remove them once they have loaded.")


def test_trial_removal_still_asks(editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._switch_tab(0)
    asked = _asks(monkeypatch, False)
    ed.gate_tv.selection_set(trial_iid('T'))
    ed._remove_selected()
    assert [t for t, _m in asked] == ["Remove trial"]
    assert sorted(ed._samples) == ['s1', 's2', 's3']


# ── Delete on a trial row does what Edit > Delete gate does ────────────────

def test_delete_key_on_a_trial_row_clears_gates_and_keeps_samples(
        editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._switch_tab(0)
    asked = _asks(monkeypatch, True)       # a box would say yes: must not ask
    ed.gate_tv.selection_set(trial_iid('T'))
    ed._on_delete_key()
    assert asked == []
    assert sorted(ed._samples) == ['s1', 's2', 's3']   # samples kept
    assert not ed._sample_gates['s1'] and not ed._sample_gates['s2']
    assert len(ed._sample_gates['s3']) == 1            # other trial untouched
    # Undoable, as the menu entry is.
    ed.update()
    ed._undo()
    assert len(ed._sample_gates['s1']) == 2
    assert len(ed._sample_gates['s2']) == 1


def test_delete_key_and_menu_entry_do_the_same_on_a_trial_row(
        editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._switch_tab(0)
    _asks(monkeypatch, True)
    ed.gate_tv.selection_set(trial_iid('T'))
    ed._clear_selected_gate()              # Edit > Delete gate
    by_menu = _state(ed)
    ed.update()
    ed._undo()
    ed.gate_tv.selection_set(trial_iid('T'))
    ed._on_delete_key()
    assert _state(ed) == by_menu


def test_delete_key_on_a_sample_row_still_does_nothing(editor, monkeypatch):
    from openflo.tree_ids import sample_iid
    ed = editor
    before = _state(ed)
    asked = _asks(monkeypatch, True)
    ed.gate_tv.selection_set(sample_iid('s1'))
    ed._on_delete_key()
    assert asked == [] and _state(ed) == before
