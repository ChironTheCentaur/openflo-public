"""The main window's control layout (Sky 11-20, toolbar reorganization).

Above the plot: channels; Mode with the Display and Events dropdowns; the
gate tools as one segmented group. Undo and Redo sit right of the Help menu.
The Samples & Gates tray has an Add and a Remove dropdown, and the button
grid below the tree is 3 x 3 in workflow order with no Delete all (it is in
Edit and the tree's right-click Remove menu). Each control must still drive
the handler or variable it did before. Synthetic samples and generic channel
names only; the editor stays withdrawn and no menu is shown.
"""
from __future__ import annotations

import pytest

from tests.conftest import (
    editor_or_skip,
    invoke_menu_path,
    load_menu_sample,
    menu_labels,
    right_click_row,
)

# Handlers each control must reach, recorded instead of run.
SIDE_HANDLERS = {
    'Delete gate': '_clear_selected_gate',
    'Copy gates…': '_open_copy_gates_dialog',
    'Auto-clean': '_create_autoclean_gate',
    'Populations…': '_open_clusters_menu',
    'Cluster…': '_open_cluster_dialog',
    'Statistics…': '_open_stats_window',
    'Report…': '_export_report',
    'Layout…': '_open_layout_editor',
    'Pipeline': '_open_pipeline_workspace',
}
RECORDED = ('_undo', '_redo', '_add_samples', '_load_processed_data',
            '_load_example_data', '_remove_selected', '_clear_all',
            '_on_gate_display_changed', '_on_ds_mode_changed',
            '_on_max_points_changed', *SIDE_HANDLERS.values())


@pytest.fixture
def rec(monkeypatch):
    """``(ed, calls)``: an editor whose RECORDED handlers append their name
    to ``calls``. Patched on the class before the window is built, since the
    controls hold the bound methods they were built with."""
    import openflo.gui as gui
    calls = []
    for name in RECORDED:
        monkeypatch.setattr(gui.ViewGateEditorWindow, name,
                            lambda self, *a, _n=name, **k: calls.append(_n))
    root, ed = editor_or_skip()
    try:
        yield ed, calls
    finally:
        root.destroy()


def _index(menu, label):
    end = menu.index('end')
    for i in range(0 if end is None else end + 1):
        if menu.type(i) != 'separator' and menu.entrycget(i, 'label') == label:
            return i
    raise AssertionError(f'no {label!r} in {menu_labels(menu)}')


def test_undo_and_redo_sit_right_after_the_help_menu(rec):
    ed, calls = rec
    strip = ed._menubar_frame.master
    first, second = strip.pack_slaves()[:2]
    assert first is ed._menubar_frame
    assert ed._menubar_frame.pack_slaves()[-1].cget('text') == 'Help'
    assert second.pack_slaves() == [ed._undo_btn, ed._redo_btn]
    # The menus' own frame holds only the six menu buttons.
    assert len(ed._menubar_frame.winfo_children()) == 6
    ed._undo_btn.invoke()
    ed._redo_btn.invoke()
    assert calls == ['_undo', '_redo']
    assert str(ed._undo_btn) in ed._tip_texts
    assert str(ed._redo_btn) in ed._tip_texts


def test_the_add_dropdown_loads_each_way(rec):
    ed, calls = rec
    menu = ed._add_menu
    assert str(ed._add_mb.cget('menu')) == str(menu)
    assert menu_labels(menu) == ['Add FCS…', 'Load CSV…',
                                 'Load example dataset']
    for i in range(3):
        menu.invoke(i)
    assert calls == ['_add_samples', '_load_processed_data',
                     '_load_example_data']


def test_remove_from_session_is_the_dropdowns_second_entry(rec):
    ed, calls = rec
    ed._remove_menu.invoke(ed.REMOVE_FROM_SESSION)
    assert calls == ['_remove_selected']


def test_the_display_dropdown_sets_the_mode_and_shows_it(rec):
    ed, calls = rec
    menu = ed._display_dd_menu
    assert menu_labels(menu) == ['All events', 'Highlight gated',
                                 'Filter to gated']
    menu.entryconfigure(2, state='normal')        # as with a real gate
    menu.invoke(2)
    assert ed.gate_display_var.get() == 'filter'
    assert calls == ['_on_gate_display_changed']
    assert ed._display_mb.cget('text') == 'Filter to gated'
    # Set from elsewhere (Ctrl+1, View > Display): the label follows.
    ed.gate_display_var.set('all')
    assert ed._display_mb.cget('text') == 'All events'


def test_the_events_dropdown_drives_downsample_cap_and_overlay(rec):
    ed, calls = rec
    menu = ed._events_menu
    ed._update_ds_visibility()
    menu.invoke(_index(menu, 'Off'))
    assert ed._ds_mode_var.get() == 'Off'
    menu.invoke(_index(menu, 'Display + data'))
    assert ed._ds_mode_var.get() == 'Display + data'
    menu.invoke(_index(menu, '100,000'))
    assert ed.max_points_var.get() == '100000'
    assert calls == ['_on_ds_mode_changed', '_on_ds_mode_changed',
                     '_on_max_points_changed']
    replots = []
    ed._schedule_replot = lambda delay=0: replots.append(delay)
    menu.invoke(_index(menu, '⚠ Show cleaned-out events'))
    assert ed.show_removed_var.get() is True
    assert replots == [0]
    assert ed._events_mb.cget('text') == (
        'Events: Display+data · 100k ⚠')
    ed.max_points_var.set('All')
    ed.show_removed_var.set(False)
    assert ed._events_mb.cget('text') == 'Events: Display+data · all'


def test_the_side_grid_is_three_rows_in_workflow_order(rec):
    ed, _calls = rec
    want = [['Delete gate', 'Copy gates…', 'Auto-clean'],
            ['Populations…', 'Cluster…', 'Statistics…'],
            ['Report…', 'Layout…', 'Pipeline']]
    got = [[None] * 3 for _ in range(3)]
    grid = ed._side_buttons['Delete gate'].master
    for w in grid.grid_slaves():
        info = w.grid_info()
        got[int(info['row'])][int(info['column'])] = str(w.cget('text'))
    assert got == want
    texts = {str(w.cget('text')) for w in grid.winfo_children()}
    assert not texts & {'Delete all', 'Undo', 'Redo'}


def test_each_side_button_runs_its_own_handler(rec):
    ed, calls = rec
    for label, handler in SIDE_HANDLERS.items():
        del calls[:]
        ed._side_buttons[label].invoke()
        assert calls == [handler], label


def test_delete_all_gates_is_in_the_samples_remove_menu(rec, monkeypatch,
                                                        menu_popups):
    from openflo.tree_ids import sample_iid
    ed, calls = rec
    load_menu_sample(ed, 's1', 0)
    ed._refresh_gate_list()
    menu = right_click_row(ed, monkeypatch, menu_popups, sample_iid('s1'))
    invoke_menu_path(menu, 'Remove', 'Delete all gates (whole tab)…')
    assert calls == ['_clear_all']


def test_the_gate_tools_are_one_segmented_group(rec):
    ed, _calls = rec
    radios = ed._gate_tool_widgets[:6]
    assert [str(r.cget('text')) for r in radios] == [
        'Quadrant', 'Rectangle', 'Polygon', 'Ellipse', 'Lasso', 'Edit']
    assert {str(r.cget('style')) for r in radios} == {'Toolbutton'}
    assert len({str(r.master) for r in radios}) == 1
    radios[2].invoke()
    assert ed.gate_tool_var.get() == 'polygon'


def test_each_dropdown_shows_entry_help_like_the_menubar(rec):
    ed, _calls = rec
    for menu in (ed._add_menu, ed._remove_menu, ed._display_dd_menu,
                 ed._events_menu):
        assert '_on_menu_select' in menu.bind('<<MenuSelect>>'), menu
        assert '_on_menu_unmap' in menu.bind('<Unmap>'), menu
        for label in menu_labels(menu):
            if label[:1].isdigit() or label in ('All', 'Off', 'Display only',
                                                'Display + data',
                                                '⚠ Show cleaned-out events'):
                continue                # values: the menu's own help covers them
            assert ed._MENU_ITEM_HELP.get(label), label


def test_the_events_button_keeps_one_width_whatever_it_says(rec):
    """Its label changes with the settings; a button that grew with it
    pushed the mode options off the row at 1366 px (review of #488). The
    fixed width still fits the longest label."""
    ed, _calls = rec
    mb = ed._events_mb
    widths, natural = set(), []
    for mode, cap, overlay in (('Off', '60000', False),
                               ('Display only', '60000', False),
                               ('Display + data', '500000', True),
                               ('Display + data', '1.5m', True),
                               ('Display + data', '999499', True),
                               ('Display + data', '1234567', True),
                               ('Display + data', 'All', True)):
        ed._ds_mode_var.set(mode)
        ed.max_points_var.set(cap)
        ed.show_removed_var.set(overlay)
        ed.update_idletasks()
        widths.add(mb.winfo_reqwidth())
        mb.configure(width=0)                    # what the text alone needs
        ed.update_idletasks()
        natural.append(mb.winfo_reqwidth())
        mb.configure(width=ed.EVENTS_BUTTON_WIDTH)
    assert len(widths) == 1, widths
    assert max(natural) <= widths.pop(), natural


def test_short_cap_forms():
    from openflo.editor_downsample import short_cap
    assert [short_cap(c, -1) for c in (-1, 1000, 1200, 60000, 250000,
                                       999_499, 999_500, 999_999,
                                       1_000_000, 1_500_000,
                                       999_600_000, 2_500_000_000)] == [
        'all', '1k', '1.2k', '60k', '250k', '999k', '1m', '1m', '1m',
        '1.5m', '1000m', '2500m']


def test_both_dropdowns_resync_as_they_open(rec):
    ed, _calls = rec
    assert '_sync_display_mode_availability' in str(
        ed._display_dd_menu.cget('postcommand'))
    assert '_update_ds_visibility' in str(ed._events_menu.cget('postcommand'))
