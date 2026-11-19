"""The active sample is always one the active tab shows (G6 review follow-up).

Switching to a tab that does not hold the active sample, or taking the
active sample off the tab, left it outside the tab: gates were drawn on a
sample the tab hides, and "Apply template to…" with nothing selected ticked
nothing though it says "or the active sample". Now the active sample moves
to the tab's selected row's sample, else the tab's first sample; an empty
tab leaves it as it is. Synthetic samples only.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

from openflo.tree_ids import sample_iid
from tests.conftest import editor_or_skip, load_menu_sample


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


@pytest.fixture
def ed(monkeypatch):
    """s1, s2 in trial T; s3, s4 in trial U; s1 is active on the All tab."""
    root, ed = editor_or_skip()
    try:
        for i, (n, tr) in enumerate((('s1', 'T'), ('s2', 'T'), ('s3', 'U'),
                                     ('s4', 'U'))):
            load_menu_sample(ed, n, i, trial=tr)
        ed._set_active_sample('s1')
        ed._refresh_gate_list()
        ed.update()
        monkeypatch.setattr(ed, 'wait_window', lambda *_a: None)
        yield ed
    finally:
        root.destroy()


def _open_tab(ed, members, selected=None):
    ts = ed._tabs()
    idx = ts.new_tab('A', members=members)
    ts.tabs[idx].selected = selected
    ed._rebuild_tab_strip()
    ed.gate_tv.selection_set(())
    ed._switch_tab(idx)
    ed.update()
    return idx


def test_a_tab_without_the_active_sample_takes_its_first(ed):
    _open_tab(ed, {'s3': None, 's4': None})
    assert ed._active_sample == 's3'
    assert ed._gates is ed._sample_gates['s3']


def test_the_tabs_selected_row_comes_first(ed):
    _open_tab(ed, {'s3': None, 's4': None}, selected=sample_iid('s4'))
    assert ed._active_sample == 's4'


def test_an_active_sample_the_tab_shows_is_kept(ed):
    ed._set_active_sample('s2')
    _open_tab(ed, {'s1': None, 's2': None})
    assert ed._active_sample == 's2'


def test_an_empty_tab_leaves_the_active_sample(ed):
    _open_tab(ed, {})
    assert ed._active_sample == 's1'


def test_taking_the_active_sample_off_the_tab_moves_it(ed):
    _open_tab(ed, {'s1': None, 's3': None}, selected=sample_iid('s1'))
    assert ed._active_sample == 's1'
    ed.gate_tv.selection_set(sample_iid('s1'))
    ed._remove_rows_from_tab()
    assert ed._active_sample == 's3'


def test_the_template_dialog_then_ticks_the_tabs_sample(ed):
    _open_tab(ed, {'s3': None, 's4': None})
    ed.gate_tv.selection_set(())
    before = set(_walk(ed.master))
    ed._ask_template_apply()
    (dlg,) = [w for w in _walk(ed.master) if w not in before
              and isinstance(w, tk.Toplevel)]
    ticked = [w.cget('text').split()[0] for w in _walk(dlg)
              if isinstance(w, ttk.Checkbutton) and w.instate(['selected'])]
    dlg.destroy()
    assert ticked == ['s3']


def test_the_helper_prefers_the_selected_rows_sample(ed):
    """Without the tree's own select event: the helper reads tab.selected."""
    ts = ed._tabs()
    idx = ts.new_tab('A', members={'s3': None, 's4': None})
    ts.current = idx
    ts.tabs[idx].selected = sample_iid('s4')
    ed._keep_active_in_tab()
    assert ed._active_sample == 's4'
