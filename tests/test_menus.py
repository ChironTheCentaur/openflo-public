"""The menus: Tools is grouped, exports live in File and the analysis
record in Analyze; F1 opens the offline user guide; the main window's
File menu ends in "Quit OpenFlo"; every menu item has status bar help;
"Figure layout" is "Quick figure grid", one name in the menu and window;
the menus offer Proliferation, Spreading matrix and Bead tracking.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

from pathlib import Path

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from openflo.ui_logic import accel
from tests._editor_ui import SRC, _all_entries, _entries, _menus
from tests.conftest import all_menus, submenu

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_tools_is_grouped_and_exports_and_the_record_moved(editor):
    _root, ed = editor
    menus = _menus(ed)
    # Tools holds its windows in submenus of at most six entries each.
    tools = _entries(menus['Tools'])
    tm = menus['Tools']
    groups = [_entries(tm.nametowidget(tm.entrycget(i, 'menu')))
              for i, (t, label, _a) in enumerate(tools)
              if t == 'cascade' and label != 'Templates']
    assert len(groups) >= 6, groups
    assert all(len(g) <= 6 for g in groups), groups
    assert len([e for e in tools if e[0] != 'separator']) <= 9, tools
    every = [label for g in [tools, *groups] for _t, label, _a in g]
    assert not [label for label in every
                if label.startswith(('Export', 'Methods', 'History'))]
    analyze = [label for _t, label, _a in _entries(menus['Analyze'])]
    i = analyze.index("Methods & provenance…")
    assert analyze[i + 1] == "History / audit…"
    # Every export in File > Export.
    fm = menus['File']
    exports = [label for _t, label, _a in _entries(submenu(fm, 'Export'))]
    assert exports == [
        "FlowJo workspace…", "Gating-ML 2.0 gates…", "Gate list (CSV)…",
        "Plot image…", "Populations (FCS)…",
        "Populations, all samples (FCS)…", "Analysis report (HTML)…"]
    # (The project archive is a way to save the session, beside it.)
    assert not [label for _t, label, _a in _entries(fm)
                if label.startswith('Export ')
                and label != 'Export project archive…']


def test_f1_opens_the_user_guide_offline(editor, monkeypatch):
    import webbrowser

    from openflo.ui_guide import GuideWindow
    root, ed = editor
    opened, about = [], []
    monkeypatch.setattr(webbrowser, 'open', lambda url: opened.append(url))
    monkeypatch.setattr(type(ed), '_show_about',
                        lambda self: about.append(1))
    ed.focus_force()
    root.update()
    ed.event_generate('<F1>')
    root.update()
    guides = [w for w in sweep._walk(root) if isinstance(w, GuideWindow)]
    assert len(guides) == 1 and not opened and not about
    guides[0].destroy()


def test_main_window_file_menu_says_quit(editor):
    root, ed = editor
    last = [e for e in _entries(_menus(ed)['File']) if e[0] != 'separator']
    assert last[-1][1:] == ("Close window", accel("Ctrl+W"))
    ed._primary = True
    for b in ed._menubar_buttons:
        b.destroy()
    ed._build_menubar()
    root.update()
    last = [e for e in _entries(_menus(ed)['File']) if e[0] != 'separator']
    assert last[-1][1:] == ("Quit OpenFlo", accel("Ctrl+Q"))
    assert '<Control-Key-q>' in ed.bind()
    ed._primary = False


def test_every_menu_item_has_status_bar_help(editor):
    _root, ed = editor
    labels = {label for t, label, _a in _all_entries(ed)
              if t in ('command', 'cascade')}
    skip = {"(no recent sessions)"}
    missing = sorted(label for label in labels - skip
                     if label not in ed._MENU_ITEM_HELP
                     and not label.startswith(('Bundled', 'Saved')))
    # Recent files and templates are names, not commands.
    missing = [m for m in missing if '.' not in m]
    assert not missing, missing


def test_one_name_for_the_quick_figure_grid(editor):
    _root, ed = editor
    labels = [label for _t, label, _a in _all_entries(ed)]
    assert "Quick figure grid…" in labels
    assert not [label for label in labels if 'Figure layout' in label]
    from openflo.ui_figure_layout import FigureLayoutDialog
    src = Path(SRC / 'ui_figure_layout.py').read_text(encoding='utf-8')
    assert 'self.title("Quick figure grid")' in src
    # And the window itself, once open, carries that name.
    dlg = FigureLayoutDialog(ed, 1, 'dot', '', lambda opts: None)
    try:
        dlg.withdraw()
        assert dlg.title() == "Quick figure grid"
    finally:
        dlg.destroy()


def test_menus_offer_the_new_windows(editor):
    _root, ed = editor
    labels = {label for _m, _i, label in sweep._menu_items(ed)}
    assert {"Proliferation…", "Spreading matrix…", "Bead tracking…"} <= labels


def test_tools_and_edit_submenus_hold_every_window(editor):
    """Tools in six submenus, gate tools in Edit > Gates: every entry the
    flat menus had is still there, under one submenu."""
    _root, ed = editor
    menus = _menus(ed)

    def labels(menu):
        return [lbl for t, lbl, _a in _entries(menu) if t != 'separator']
    tools = menus['Tools']
    assert labels(tools) == [
        "Compensation & spectral", "Channels", "Normalize & clean",
        "Figures", "Panel & instrument QC", "Files & checks", "Templates",
        "Save template…"]
    want = {
        "Compensation & spectral": ["Compensation…", "Compensation QC…",
                                    "Spectral unmix…", "Spectral QC…"],
        "Channels": ["Transforms…", "Calibration…", "Derived channels…"],
        "Normalize & clean": ["Normalize batches (CytoNorm)…",
                              "Undo batch normalization", "Debarcode…",
                              "Mass cytometry"],
        "Figures": ["Layout editor (beta)…", "Quick figure grid…",
                    "Gating tree diagram…"],
        "Panel & instrument QC": ["Panel check…", "Spreading matrix…",
                                  "Bead tracking…", "Voltage optimization…"],
        "Files & checks": ["Compare FlowJo workspace…", "FCS inspector…",
                           "Watch folder…",
                           "Review fixes for this session…"],
    }
    for group, entries in want.items():
        assert labels(submenu(tools, group)) == entries, group
    assert labels(submenu(submenu(tools, "Normalize & clean"),
                           "Mass cytometry")) == [
        "Bead normalization…", "Undo bead normalization",
        "Gaussian parameter clean-up…"]
    assert labels(submenu(menus['Edit'], "Gates")) == [
        "Add singlet gate", "FMO gating…", "Auto-clean gate",
        "Create boolean gate…", "Combination gates…",
        "Edit gate coordinates…", "Re-place gate on each sample…"]


def test_every_menubar_submenu_shows_help_while_hovered(editor):
    """Each submenu of the menubar puts the hovered entry's help in the
    status bar, as the top menus do."""
    _root, ed = editor
    every = [m for top in _menus(ed).values() for m in all_menus(top)]
    for m in every:
        assert '<<MenuSelect>>' in m.bind(), m
    assert len(every) >= 20
