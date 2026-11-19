"""Frequencies is a tab of the Statistics window, not a window of its own.

The side panel had a Frequencies button beside Statistics, and the two
windows showed the same Count / %Parent / %Total numbers: one as a table,
one with the samples split into groups and compared. Now Statistics has a
Table tab and a Frequencies by group tab, the side panel has one button,
and Analyze > Frequencies… opens Statistics on that tab. Six synthetic
samples (tests/test_frequency_window.DESIGN) carry the gates; no window is
shown (each is withdrawn before the event loop can map it).
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_frequency_window import DESIGN, _add, _inline  # noqa: E402
from test_gui_smoke import _editor_or_skip  # noqa: E402


@pytest.fixture
def editor(monkeypatch):
    import openflo.async_task as _at
    import openflo.ui_statistics as us
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    # Keep every Statistics window hidden: withdrawn as it is built, and a
    # second open (which raises the open one) does not map it either.
    orig = us.StatisticsWindow.__init__

    def _hidden(self, *a, **k):
        orig(self, *a, **k)
        self.withdraw()
    monkeypatch.setattr(us.StatisticsWindow, '__init__', _hidden)
    monkeypatch.setattr(us.StatisticsWindow, 'deiconify', lambda self: None)
    monkeypatch.setattr(us.StatisticsWindow, 'focus_force', lambda self: None)
    try:
        for i, (nm, tr, a, b) in enumerate(DESIGN):
            _add(ed, nm, tr, a, b, seed=i)
        ed._channels = ['CD3-A', 'CD4-A']
        ed._channel_labels = {'CD3-A': 'CD3-A', 'CD4-A': 'CD4-A'}
        ed._active_sample = 'Stim_1'
        yield ed
    finally:
        root.destroy()


def _tabs(win):
    return [win._nb.tab(t, 'text') for t in win._nb.tabs()]


def test_statistics_has_a_table_tab_and_a_frequencies_tab(editor):
    from openflo.ui_frequency import FrequencyPanel
    win = editor._open_stats_window()
    assert _tabs(win) == ['Table', 'Frequencies by group']
    assert win._nb.index('current') == 0           # opens on the table
    # The comparison is not built (or computed) until its tab is shown,
    # not even once the event loop has run.
    win.update()
    assert win.compare is None
    assert editor.status_var.get() != "Computing frequencies…"
    win._nb.select(win._compare_tab)
    win.update()                                   # <<NotebookTabChanged>>
    panel = win.compare
    assert isinstance(panel, FrequencyPanel)
    assert panel.winfo_toplevel() is win           # inside Statistics
    for _ in range(200):                           # its first collect runs
        panel.update()                             # 30 ms after it is built
        if panel.pop_combo['values']:
            break
        time.sleep(0.01)
    assert sorted(panel.pop_combo['values']) == ['CD3pos', 'CD3pos/CD4pos']
    # Switching back and forth keeps the one panel.
    win.show_table()
    win.update()
    assert win.show_compare() is panel


def test_frequencies_menu_opens_statistics_on_the_frequencies_tab(editor):
    win = editor._open_frequency_window()
    reg = editor._tool_windows
    assert reg.get('stats') is win and 'frequency' not in reg
    assert win._nb.index('current') == 1
    assert win.compare is not None
    # Statistics… afterwards raises the same window, not a second one.
    assert editor._open_stats_window() is win
    # And Frequencies… again, with the table showing, brings the tab back.
    win.show_table()
    assert editor._open_frequency_window() is win
    assert win._nb.index('current') == 1


def test_frequencies_needs_a_loaded_sample(monkeypatch):
    root, ed, _gui = _editor_or_skip()
    try:
        assert ed._open_frequency_window() is None
        assert ed.status_var.get() == ("Load samples first to compare "
                                       "frequencies.")
        assert 'stats' not in getattr(ed, '_tool_windows', {})
    finally:
        root.destroy()


def test_side_panel_has_statistics_but_no_frequencies_button():
    root, ed, _gui = _editor_or_skip()
    try:
        labels = []

        def walk(w):
            for c in w.winfo_children():
                if c.winfo_class() == 'TButton':
                    labels.append(str(c.cget('text')))
                walk(c)
        walk(ed)
        assert 'Statistics…' in labels and 'Cluster…' in labels
        assert 'Frequencies…' not in labels
    finally:
        root.destroy()


def test_an_older_table_finishing_late_does_not_overwrite_a_newer_one(
        editor, monkeypatch):
    # Each toggle or Refresh computes off the Tk thread. A slow, older
    # request finishing last used to put back its columns and counts.
    import openflo.async_task as _at
    win = editor._open_stats_window()
    pending = []
    monkeypatch.setattr(_at, 'run_async',
                        lambda widget, work, on_done=None, on_error=None,
                        on_finally=None: pending.append((work, on_done)))
    win._stat_vars['Mean'].set(True)
    win._refresh()                                # older: with Mean
    win._stat_vars['Mean'].set(False)
    editor._sample_gates['Stim_1']['g1']['value'] = 1.5
    win._refresh()                                # newer: no Mean, new gate
    (old_work, old_done), (new_work, new_done) = pending
    new_done(new_work())
    old_done(old_work())                          # lands last
    assert not any(c.startswith('Mean') for c in win._cols)
    stim1 = [r for r in win._rows
             if r['Sample'] == 'Stim_1' and r['Population'] == 'CD3pos']
    assert [r['Count'] for r in stim1] == [0]


@pytest.fixture
def app_look(editor):
    """The app's own theme at Tk scaling 1.39, as at startup on a scaled
    display. Under the plain ttk theme the rows are about a fifth narrower,
    and a row that hid Close passed."""
    from openflo.ui_style import apply_theme
    root = editor._root()
    base = float(root.tk.call('tk', 'scaling'))
    root.tk.call('tk', 'scaling', 1.39)
    apply_theme(root, 'light')
    try:
        yield editor
    finally:
        # Tk keeps the scaling per display, not per interpreter.
        root.tk.call('tk', 'scaling', base)


def _rows(holder, skip):
    return [c for c in holder.winfo_children()
            if c.winfo_class() == 'TFrame' and c not in skip]


def _texts(rows):
    return {str(k.cget('text')) for r in rows for k in r.winfo_children()
            if k.winfo_class() in ('TLabel', 'TButton', 'TCheckbutton')}


def test_every_control_row_fits_the_smallest_statistics_window(app_look):
    # The Frequencies rows were laid out for a 1120 px window; in
    # Statistics (1000 px, 720 px at its smallest) the packer squeezed Fig
    # background to nothing, and with the app's theme Close fell off both
    # tabs. Every row of both tabs must fit the smallest window, and the
    # smallest window must fit a 1024x768 screen.
    win = app_look._open_stats_window()
    panel = win.show_compare()
    win.update_idletasks()
    min_w, min_h = (int(v) for v in win.wm_minsize())
    assert min_w <= 1000 and min_h <= 640      # the opening size is larger
    assert min_w + 16 <= 1024 and min_h + 40 <= 768  # frame and title bar
    table = win._nb.nametowidget(win._nb.tabs()[0])
    tabs = (table, panel)
    # What the notebook takes around a tab, across and down.
    pad_w = win._nb.winfo_reqwidth() - max(t.winfo_reqwidth() for t in tabs)
    pad_h = win._nb.winfo_reqheight() - max(t.winfo_reqheight() for t in tabs)
    table_holder = win.tv.master
    for holder, grow in ((table, table_holder), (panel, panel._figframe)):
        rows = _rows(holder, {grow})
        for row in rows:
            labels = [str(k.cget('text')) for k in row.winfo_children()
                      if k.winfo_class() in ('TLabel', 'TButton')]
            assert row.winfo_reqwidth() <= min_w - pad_w, (
                labels, row.winfo_reqwidth(), min_w - pad_w)
        # Down: everything but the table / figure, which give way.
        fixed = sum(c.winfo_reqheight() for c in holder.winfo_children()
                    if c is not grow and c.winfo_manager() == 'pack')
        assert fixed <= min_h - pad_h, (fixed, min_h - pad_h)
    panel_rows = _rows(panel, {panel._figframe, panel._summary.master})
    assert len(panel_rows) == 4      # three control rows and the export row
    for want in ('Update', 'Fig background:', 'Paired by donor',
                 'Diff. abundance…', 'Compare all…', 'Export figure…',
                 'Close'):
        assert want in _texts(panel_rows)
    for want in ('Refresh', 'Export CSV…', 'Save as Excel…', 'Copy table',
                 'Import S&G gates', 'Import workspace', 'Clear', 'Close'):
        assert want in _texts(_rows(table, {table_holder}))


def test_closing_mid_collect_clears_the_status_message(editor, monkeypatch):
    # The collect's result is dropped once the window is gone, so the
    # editor's "Computing frequencies…" would stay up for good.
    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', lambda *a, **k: None)   # never lands
    win = editor._open_frequency_window()
    panel = win.compare
    panel._update()
    assert editor.status_var.get() == "Computing frequencies…"
    win.destroy()
    assert editor.status_var.get() == ""
    # A newer, unrelated message is left alone.
    win = editor._open_frequency_window()
    editor.status_var.set("Something else")
    win.destroy()
    assert editor.status_var.get() == "Something else"


def test_dropping_a_gate_shows_the_table_tab(editor):
    win = editor._open_frequency_window()
    assert win._nb.index('current') == 1
    win.add_targets([('Stim_1', 'g1')], 'editor')
    assert win._nb.index('current') == 0
    assert [(r['Sample'], r['Population']) for r in win._rows] == [
        ('Stim_1', 'CD3pos')]


def test_update_with_every_gate_deleted_picks_no_population(editor):
    win = editor._open_frequency_window()
    panel = win.compare
    panel._collect(panel._after_collect)
    assert panel.pop_var.get() == 'CD3pos'
    for nm, *_ in DESIGN:
        editor._sample_gates[nm] = {}
        editor._sample_gate_order[nm] = []
    panel.update_button.invoke()
    assert panel.pop_var.get() == ''
    assert panel._summary.get('1.0', 'end').strip() == (
        'No populations to compare.')


def test_a_failed_snapshot_retires_older_table_requests(editor, monkeypatch):
    import openflo.async_task as _at
    win = editor._open_stats_window()
    pending = []
    monkeypatch.setattr(_at, 'run_async',
                        lambda widget, work, on_done=None, on_error=None,
                        on_finally=None: pending.append((work, on_done)))
    win._refresh()                                 # older, in flight

    def _boom(*a, **k):
        raise RuntimeError("snapshot broke")
    monkeypatch.setattr(editor, '_stats_snapshot', _boom)
    win._refresh()                                 # newer: fails at once
    assert win.status_var.get() == "Stats failed: snapshot broke"
    (old_work, old_done), = pending
    monkeypatch.undo()
    old_done(old_work())                           # lands last
    assert win.status_var.get() == "Stats failed: snapshot broke"


def test_a_drop_on_a_window_opened_from_the_tab_is_not_a_drop_on_statistics(
        editor, monkeypatch):
    # Compare all / Diff. abundance windows hang off the panel, so walking
    # their masters reaches Statistics; a drop there must not land in it.
    import tkinter as tk
    win = editor._open_frequency_window()
    child = tk.Toplevel(win.compare)
    child.withdraw()
    on_child = tk.Label(child)
    monkeypatch.setattr(editor, 'winfo_containing', lambda x, y: on_child)
    assert editor._stats_window_under(0, 0) is None
    inside = win.compare.update_button
    monkeypatch.setattr(editor, 'winfo_containing', lambda x, y: inside)
    assert editor._stats_window_under(0, 0) is win
