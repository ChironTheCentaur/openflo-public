"""The template dialog ticks the selected samples at first (scope audit G6).

Every listed sample was ticked when "Apply template to…" opened, and the
mode was Overwrite, so pressing Apply as it stood replaced the gates of
every sample in the tab. Now only the samples selected in the sample list
are ticked (a trial row stands for the members it shows), else the active
sample; "Select all" still ticks every one. Synthetic samples only.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

from openflo.tree_ids import sample_iid, trial_iid
from tests.conftest import editor_or_skip, load_menu_sample


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


@pytest.fixture
def ed(monkeypatch):
    """s1, s2 in trial T; s3, s4 in trial U; s2 is active."""
    root, ed = editor_or_skip()
    try:
        for i, (n, tr) in enumerate((('s1', 'T'), ('s2', 'T'), ('s3', 'U'),
                                     ('s4', 'U'))):
            load_menu_sample(ed, n, i, trial=tr)
        ed._set_active_sample('s2')
        ed._refresh_gate_list()
        ed.update()
        monkeypatch.setattr(ed, 'wait_window', lambda *_a: None)
        yield ed
    finally:
        root.destroy()


def _open(ed):
    before = set(_walk(ed.master))
    assert ed._ask_template_apply() is None        # wait_window returns
    (dlg,) = [w for w in _walk(ed.master) if w not in before
              and isinstance(w, tk.Toplevel)]
    return dlg


def _ticked(dlg):
    return [w.cget('text').split()[0] for w in _walk(dlg)
            if isinstance(w, ttk.Checkbutton) and w.instate(['selected'])]


def _press(dlg, text):
    (b,) = [w for w in _walk(dlg)
            if isinstance(w, ttk.Button) and w.cget('text') == text]
    b.invoke()


def test_nothing_selected_ticks_only_the_active_sample(ed):
    ed.gate_tv.selection_set(())
    dlg = _open(ed)
    assert _ticked(dlg) == ['s2']
    dlg.destroy()


def test_the_selected_samples_are_ticked(ed):
    ed.gate_tv.selection_set((sample_iid('s1'), sample_iid('s4')))
    dlg = _open(ed)
    assert _ticked(dlg) == ['s1', 's4']
    dlg.destroy()


def test_a_selected_trial_row_ticks_its_members(ed):
    ed.gate_tv.selection_set(trial_iid('U'))
    dlg = _open(ed)
    assert _ticked(dlg) == ['s3', 's4']
    dlg.destroy()


def test_in_a_tab_a_trial_row_ticks_only_the_members_it_shows(ed):
    idx = ed._tabs().new_tab('A', members={'s1': None, 's3': None,
                                           's4': None})
    ed._rebuild_tab_strip()
    ed._switch_tab(idx)
    ed.update()
    ed.gate_tv.selection_set(trial_iid('T'))
    dlg = _open(ed)
    assert _ticked(dlg) == ['s1']
    dlg.destroy()


def test_select_all_still_ticks_every_listed_sample(ed):
    ed.gate_tv.selection_set(())
    dlg = _open(ed)
    _press(dlg, 'Select all')
    assert _ticked(dlg) == ['s1', 's2', 's3', 's4']
    dlg.destroy()


def test_the_dialog_says_what_is_ticked(ed):
    dlg = _open(ed)
    texts = [str(w.cget('text')) for w in _walk(dlg)
             if isinstance(w, ttk.Label)]
    assert any(t.startswith('Ticked: the samples selected')
               and 'or the active sample' in t and 'Select all' in t
               for t in texts), texts
    dlg.destroy()
