"""Remove trial from session, on a tab, removes only what the row shows.

A trial row on a tab other than All lists just the trial's samples that tab
holds ("T (1)"), but Remove trial from session used to unload every sample of
the trial, hidden ones included, and its question gave only a count. Now:

* on a tab, the row's Remove trial from session unloads only the trial's
  samples that tab shows; the members the tab hides stay loaded (with their
  gates, in All and every other tab), and the trial header stays for them;
* the question names the samples it will unload and their gate count, says
  which hidden samples stay loaded, and points to Remove trial from this tab
  and to the All tab for the whole trial; Enter means No;
* on the All tab the whole trial still goes, and the question names every
  sample.

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

TRIAL_SESSION = "Remove trial from session (samples + gates)…"


@pytest.fixture
def editor():
    """s1, s2, s3 (trial T) and s4 (trial U), one gate each plus a second
    gate on s1; tab A = {s1} is open, so it hides s2 and s3 of T."""
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        gids = {}
        for i, (n, tr) in enumerate((('s1', 'T'), ('s2', 'T'), ('s3', 'T'),
                                     ('s4', 'U'))):
            gids[n] = load_menu_sample(ed, n, i, trial=tr)
        ed._set_active_sample('s1')
        ed._add_gate({'kind': 'threshold', 'name': 'Hi', 'channel': 'FL1-A',
                      'op': '>', 'value': 800.0, 'parent_id': gids['s1']},
                     audit=False)
        ed._refresh_gate_list()
        ts = ed._tabs()
        a = ts.new_tab('A', members={'s1': None})
        ts.new_tab('B', members={'s2': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(a)
        ed.update()
        yield ed
    finally:
        root.destroy()


def _asks(monkeypatch, answer):
    """Record every askyesno (title, message, default) and answer it."""
    from tkinter import messagebox
    asked = []

    def ask(title=None, message=None, **k):
        asked.append((title, message, k.get('default')))
        return answer
    monkeypatch.setattr(messagebox, 'askyesno', ask)
    return asked


def _state(ed):
    return (sorted(ed._samples),
            {n: sorted(g) for n, g in ed._sample_gates.items()},
            dict(ed._sample_trial), list(ed._trial_order))


def _trial_row_text(ed, trial):
    return ed.gate_tv.item(ed._trial_iid(trial), 'text')


def test_on_a_tab_only_the_shown_members_are_unloaded(
        editor, monkeypatch, menu_popups):
    from openflo.tree_ids import trial_iid
    ed = editor
    assert _trial_row_text(ed, 'T').endswith('(1)')      # the row shows s1
    asked = _asks(monkeypatch, True)
    menu = right_click_row(ed, monkeypatch, menu_popups, trial_iid('T'))
    invoke_menu_path(menu, TRIAL_SESSION)
    assert len(asked) == 1
    assert sorted(ed._samples) == ['s2', 's3', 's4']
    # The hidden members keep their gates and trial, and the header stays.
    assert len(ed._sample_gates['s2']) == 1
    assert len(ed._sample_gates['s3']) == 1
    assert ed._sample_trial['s2'] == 'T' and 'T' in ed._trial_order
    assert 's1' not in ed._sample_gates
    assert ed._tabs().tabs[2].has_sample('s2')          # tab B untouched
    status = ed.status_var.get()
    assert "Removed trial 'T' (1 sample(s)) from the session" in status
    assert "2 sample(s) this tab doesn't show stay loaded" in status


def test_the_question_names_what_goes_and_what_stays(
        editor, monkeypatch, menu_popups):
    from tkinter import messagebox

    from openflo.tree_ids import trial_iid
    ed = editor
    before = _state(ed)
    asked = _asks(monkeypatch, False)
    menu = right_click_row(ed, monkeypatch, menu_popups, trial_iid('T'))
    invoke_menu_path(menu, TRIAL_SESSION)
    assert len(asked) == 1
    title, msg, default = asked[0]
    assert title == "Remove trial"
    assert "trial 'T'" in msg
    assert "sample 's1'" in msg
    assert len(ed._sample_gates['s1']) == 2 and "2 gate(s)" in msg
    assert "2 samples ('s2', 's3')" in msg and "stay loaded" in msg
    assert "use the All tab" in msg
    assert "Remove trial from this tab" in msg
    assert "can't be undone" in msg
    assert "2 sample(s)" not in msg        # no bare count of the whole trial
    assert default == messagebox.NO
    assert _state(ed) == before            # No: nothing removed


def test_on_the_all_tab_the_whole_trial_goes_and_every_name_is_given(
        editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._switch_tab(0)
    want = sum(len(ed._sample_gates[n]) for n in ('s1', 's2', 's3'))
    asked = _asks(monkeypatch, True)
    ed.gate_tv.selection_set(trial_iid('T'))
    ed._remove_selected()                  # the tray's Remove button
    assert len(asked) == 1
    msg = asked[0][1]
    assert "3 samples ('s1', 's2', 's3')" in msg
    assert want == 4 and f"{want} gate(s)" in msg
    assert "stay loaded" not in msg and "from this tab" not in msg
    assert sorted(ed._samples) == ['s4']
    assert 'T' not in ed._trial_order
    assert ed.status_var.get() == ("Removed trial 'T' (3 sample(s)) from "
                                   "the session.")


def test_two_trial_rows_on_a_tab_count_only_their_shown_members(
        editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    ts = ed._tabs()
    ts.tabs[ts.current].add('s4')          # tab A now shows s1 (T) and s4 (U)
    ed._refresh_gate_list()
    asked = _asks(monkeypatch, True)
    ed.gate_tv.selection_set((trial_iid('T'), trial_iid('U')))
    ed._remove_selected()
    msg = asked[0][1]
    assert "2 trials" in msg
    assert "2 samples ('s1', 's4')" in msg
    assert "2 samples ('s2', 's3')" in msg and "stay loaded" in msg
    assert sorted(ed._samples) == ['s2', 's3']
    assert ed._trial_order == ['T']        # U is empty now, T still has s2, s3


def test_a_trial_the_tab_hides_entirely_is_left_alone(editor, monkeypatch):
    """A selection can still carry a trial row the tab doesn't draw (the
    tree is refreshed later): nothing is unloaded, no question, and the
    status line says where to go."""
    from openflo.tree_ids import trial_iid
    ed = editor
    before = _state(ed)
    asked = _asks(monkeypatch, True)
    ed.gate_tv.insert('', 'end', iid=trial_iid('U'), text='U')
    ed.gate_tv.selection_set(trial_iid('U'))
    ed._remove_selected()
    assert asked == [] and _state(ed) == before
    assert ed.status_var.get() == (
        "This tab shows no sample of trial 'U'; its 1 sample(s) stay "
        "loaded. To remove the whole trial, use the All tab.")


def test_an_empty_trial_header_is_still_just_forgotten(editor, monkeypatch):
    from openflo.tree_ids import trial_iid
    ed = editor
    ed._switch_tab(0)
    ed._trial_order.append('Empty')        # a header without samples
    ed.gate_tv.insert('', 'end', iid=trial_iid('Empty'), text='Empty')
    asked = _asks(monkeypatch, True)
    ed.gate_tv.selection_set(trial_iid('Empty'))
    ed._remove_selected()
    assert asked == [] and 'Empty' not in ed._trial_order
    assert sorted(ed._samples) == ['s1', 's2', 's3', 's4']
