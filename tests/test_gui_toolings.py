"""Smoke tests for the v2 tooling GUI wiring (menus + dialog construction).

These don't exercise the heavy compute (the logic modules have their own
unit tests); they assert the menu items exist and the dialogs/openers run
headlessly without error — catching wiring regressions.
"""
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip, _load_fake  # noqa: E402


def _menu_labels(root, ed, name):
    btn = [b for b in ed._menubar_buttons if b.cget('text') == name][0]
    m = root.nametowidget(btn.cget('menu'))
    return [m.entrycget(i, 'label') for i in range(m.index('end') + 1)
            if m.type(i) == 'command']


def invoke_menu_entry(ed, menu, label):
    """Run the command of `label` on the menubar's `menu`, as a click on it
    does, without posting the menu (a posted menu waits for a person on
    Windows)."""
    btn = [b for b in ed._menubar_buttons if b.cget('text') == menu][0]
    m = ed.nametowidget(btn.cget('menu'))
    (i,) = [i for i in range(m.index('end') + 1)
            if m.type(i) == 'command' and m.entrycget(i, 'label') == label]
    m.invoke(i)


def test_spectral_qc_has_a_tools_entry_that_says_when_there_is_none():
    """Spectral QC had no menu entry: it opened only after an unmix, so a
    report restored with a session was reachable only through Panel check.
    Before any unmix the entry says what to run, and opens nothing."""
    from openflo.ui_spectral_qc import SpectralQCWindow
    root, ed, _gui = _editor_or_skip()
    try:
        labels = _menu_labels(root, ed, 'Tools')
        assert labels.index('Spectral QC…') == (
            labels.index('Spectral unmix…') + 1)
        assert ed._last_unmix_qc is None
        invoke_menu_entry(ed, 'Tools', 'Spectral QC…')
        assert ed.status_var.get() == (
            "No spectral QC yet: run Tools, Spectral unmix first.")
        assert not any(isinstance(w, SpectralQCWindow)
                       for w in ed.winfo_children())
    finally:
        root.destroy()


def test_new_menu_items_present():
    root, ed, _gui = _editor_or_skip()
    try:
        assert {'Add singlet gate', 'FMO gating…', 'Preferences…'} <= set(
            _menu_labels(root, ed, 'Edit'))
        assert {'Group comparison…', 'Compare embeddings…',
                'Methods & provenance…', 'Absolute counts…'} <= set(
            _menu_labels(root, ed, 'Analyze'))
        assert {'Compensation QC…', 'Gating tree diagram…'} <= set(
            _menu_labels(root, ed, 'Tools'))
        assert 'Export populations (FCS)…' in _menu_labels(root, ed, 'File')
        assert {'User guide', 'Online documentation',
                'Keyboard shortcuts'} <= set(
            _menu_labels(root, ed, 'Help'))
        assert 'Reset plot view' in _menu_labels(root, ed, 'View')
    finally:
        root.destroy()


def test_tooling_dialogs_construct():
    root, ed, gui = _editor_or_skip()
    try:
        _load_fake(ed, 'a1', 'T')
        ed._set_active_sample('a1')
        from openflo.ui_abscounts import AbsCountsDialog
        from openflo.ui_methods import MethodsWindow
        from openflo.ui_preferences import PreferencesDialog
        for factory in (MethodsWindow, AbsCountsDialog, PreferencesDialog):
            w = factory(ed)
            w.destroy()
    finally:
        root.destroy()


def test_pan_zoom_and_find_hooks():
    root, ed, _gui = _editor_or_skip()
    try:
        assert ed._pan_start is None
        assert callable(ed._on_scroll) and callable(ed._reset_plot_view)
        _load_fake(ed, 'CD4_cells', 'T')
        ed._set_active_sample('CD4_cells')
        ed._refresh_gate_list()
        ed._find_var.set('cd4')
        assert ed.gate_tv.selection()        # find jumped to a match
    finally:
        root.destroy()


def test_chrome_font_scales_with_width():
    """The ttk control font steps down as the window narrows (responsive chrome)."""
    from tkinter import ttk
    root, ed, _gui = _editor_or_skip()
    try:
        st = ttk.Style(ed)
        ed.winfo_width = lambda: 1500
        ed._apply_chrome_scale(force=True)
        big = ed._chrome_font_size
        ed.winfo_width = lambda: 980
        ed._apply_chrome_scale(force=True)
        small = ed._chrome_font_size
        assert big == 10 and small == 7 and small < big
        assert str(small) in str(st.lookup('.', 'font'))
    finally:
        root.destroy()
