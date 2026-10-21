"""Batch I (UI 24, 31 to 33, 35 to 40): menus, fonts and colours.

- Tools is grouped; exports live in File and the analysis record in Analyze.
- Menus show their shortcuts (Ctrl, or Cmd on macOS, where the same keys are
  bound to Command); F1 opens the documentation; the main window's File menu
  ends in "Quit OpenFlo" (Ctrl+Q).
- An export says so in the status bar, with Open folder, instead of an
  OK-only pop-up.
- Windows use named fonts (Heading, Small, Note, Mono, ...) of 9 pt or more,
  never a ("TkDefaultFont", 8) tuple.
- Colours come from the theme: error, warning, highlight and log colours
  per theme, each readable (4.5:1) on the window and panel colours.
- Theme is Light or Dark windows plus a White or Black plot background;
  pop-up figures follow the plot background.
- The Cluster dialog shows only the picked method's settings and lists
  markers as "marker (channel)"; the layout editor names its real buttons,
  keeps its status line whole, and has undo, redo and Ctrl+S.
- "Figure layout" is "Quick figure grid", one name in the menu and window.

Each check runs on generated data (synthetic sample names, generic
channels) and is proved to fail on the old behaviour by
a mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import ast
import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pytest

import tests.test_windows_nothing_hidden as sweep
from tests.test_batch_h_consistent_dialogs import _sequence

os.environ.setdefault('MPLBACKEND', 'Agg')

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'


@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known):
    virtual_screen(1920, 1080)
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


def _menus(ed):
    """{top label: tk.Menu} of the menubar."""
    return {b.cget('text'): b.nametowidget(b.cget('menu'))
            for b in ed._menubar_buttons}


def _entries(menu):
    """[(type, label, accelerator)] of a menu, separators included."""
    end = menu.index('end')
    out = []
    for i in range(0 if end is None else end + 1):
        t = menu.type(i)
        if t == 'separator':
            out.append((t, '', ''))
        else:
            out.append((t, menu.entrycget(i, 'label'),
                        menu.entrycget(i, 'accelerator')))
    return out


def _all_entries(ed):
    for w in sweep._walk(ed):
        if w.winfo_class() == 'Menu':
            for e in _entries(w):
                if e[0] != 'separator':
                    yield e


# ── Menus (UI 31 to 33) ─────────────────────────────────────────────────────

def test_tools_is_grouped_and_exports_and_the_record_moved(editor):
    _root, ed = editor
    menus = _menus(ed)
    tools = _entries(menus['Tools'])
    groups, cur = [], []
    for t, label, _a in tools:
        if t == 'separator':
            groups.append(cur)
            cur = []
        else:
            cur.append(label)
    groups.append(cur)
    assert len(groups) >= 4, groups
    assert all(len(g) <= 6 for g in groups), groups
    assert not [label for _t, label, _a in tools
                if label.startswith(('Export', 'Methods', 'History'))]
    analyze = [label for _t, label, _a in _entries(menus['Analyze'])]
    i = analyze.index("Methods & provenance…")
    assert analyze[i + 1] == "History / audit…"
    file_labels = [label for _t, label, _a in _entries(menus['File'])]
    assert "Export populations (FCS)…" in file_labels


def test_menus_show_their_shortcuts_and_each_is_bound(editor):
    _root, ed = editor
    shown = {label: acc for _t, label, acc in _all_entries(ed) if acc}
    for label, acc in (("Undo", "Ctrl+Z"), ("Redo", "Ctrl+Y"),
                       ("Find sample or gate", "Ctrl+F"),
                       ("Zoom in", "Ctrl++"), ("Zoom out", "Ctrl+-"),
                       ("Redraw plot", "F5"), ("User guide", "F1"),
                       ("Delete gate", "Delete")):
        assert shown.get(label) == acc, (label, shown.get(label))
    bound = set(ed.bind()) | set(ed.bind_all()) | set(ed.gate_tv.bind())
    unbound = [(label, acc) for label, acc in shown.items()
               if acc != 'Delete' and _sequence(acc) not in bound]
    assert not unbound, unbound


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
    assert last[-1][1:] == ("Close window", "Ctrl+W")
    ed._primary = True
    for b in ed._menubar_buttons:
        b.destroy()
    ed._build_menubar()
    root.update()
    last = [e for e in _entries(_menus(ed)['File']) if e[0] != 'separator']
    assert last[-1][1:] == ("Quit OpenFlo", "Ctrl+Q")
    assert '<Control-Key-q>' in ed.bind()
    ed._primary = False


def test_macos_shows_cmd_and_binds_command(monkeypatch):
    import sys

    from openflo.editor_lifecycle import LifecycleMixin
    from openflo.ui_logic import accel
    assert accel("Ctrl+Shift+S", 'darwin') == "Cmd+Shift+S"
    assert accel("Ctrl+Z", 'win32') == "Ctrl+Z"
    assert accel("F5", 'darwin') == "F5"

    class Stub:
        def __init__(self):
            self.seqs = []

        def bind(self, seq, _fn):
            self.seqs.append(seq)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(sys, 'platform', 'darwin')
    stub = Stub()
    LifecycleMixin._bind_shortcuts(stub)
    assert '<Command-q>' in stub.seqs and '<Control-q>' in stub.seqs
    assert '<Command-o>' in stub.seqs and '<F5>' in stub.seqs
    monkeypatch.setattr(sys, 'platform', 'linux')
    stub = Stub()
    LifecycleMixin._bind_shortcuts(stub)
    assert not [s for s in stub.seqs if 'Command' in s]


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


# ── Saved, without a pop-up (UI 35) ─────────────────────────────────────────

_SAVED_WORDS = ('saved', 'exported', 'wrote', 'written', 'signed')


def _first_text(node):
    """The leading text of a call argument (a literal, or the literal start
    of an f-string or concatenation)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        for v in node.values:
            if isinstance(v, ast.Constant):
                return v.value
            return ''
    if isinstance(node, ast.BinOp):
        return _first_text(node.left)
    return ''


def test_no_info_box_says_a_file_was_saved():
    found = []
    for path in sorted(SRC.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'showinfo' and len(node.args) > 1):
                text = _first_text(node.args[1]).strip().lower()
                if text.startswith(_SAVED_WORDS):
                    found.append(f"{path.name}:{node.lineno}")
    assert not found, found


def test_the_check_finds_a_saved_box():
    code = 'messagebox.showinfo("Export", f"Saved:\\n{path}", parent=p)'
    call = ast.parse(code).body[0].value
    assert _first_text(call.args[1]).lower().startswith(_SAVED_WORDS)


def test_an_export_says_so_in_the_status_bar_with_open_folder(
        editor, tmp_path, monkeypatch):
    from tkinter import messagebox

    from matplotlib.figure import Figure

    from openflo import tool_window, ui_notice
    root, ed = editor
    infos = []
    monkeypatch.setattr(messagebox, 'showinfo',
                        lambda *a, **k: infos.append(a))
    out = tmp_path / 'synthetic_plot.png'
    monkeypatch.setattr(tool_window, 'ask_figure_path',
                        lambda *a, **k: str(out))
    win = tk.Toplevel(ed)
    win.status_var = tk.StringVar(master=win, value='')
    fig = Figure(figsize=(2, 2))
    fig.add_subplot().plot([0, 1], [0, 1])
    assert tool_window.export_figure(win, fig) == str(out)
    root.update()
    assert out.exists() and not infos
    assert ed.status_var.get() == "Saved synthetic_plot.png"
    assert win.status_var.get() == "Saved synthetic_plot.png"
    btn = ed._saved_folder_btn
    assert btn.winfo_manager() == 'pack'
    opened = []
    monkeypatch.setattr(ui_notice, 'open_folder',
                        lambda p: opened.append(p))
    btn.invoke()
    assert opened == [str(out)]
    ed.status_var.set("Something else happened.")
    root.update()
    assert btn.winfo_manager() == ''
    win.destroy()


def test_open_folder_command_per_system(tmp_path, monkeypatch):
    import sys

    from openflo.ui_notice import folder_command
    f = tmp_path / 'synthetic.csv'
    f.write_text('x\n')
    monkeypatch.setattr(sys, 'platform', 'win32')
    assert folder_command(f) == ['explorer', f'/select,{f}']
    monkeypatch.setattr(sys, 'platform', 'darwin')
    assert folder_command(f) == ['open', '-R', str(f)]
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert folder_command(f) == ['xdg-open', str(tmp_path)]


# ── Fonts (UI 36) ───────────────────────────────────────────────────────────

def test_named_fonts_exist_follow_the_chrome_and_stay_readable(editor):
    import tkinter.font as tkfont

    from openflo.theme import FONT_MIN, NAMED_FONTS, ensure_fonts
    root, ed = editor
    for size in (7, 10, 12):
        ensure_fonts(root, size)
        sizes = {n: abs(int(tkfont.nametofont(n, root=root).actual('size')))
                 for n in NAMED_FONTS}
        assert min(sizes.values()) >= FONT_MIN, (size, sizes)
        assert sizes['OpenFloHeading'] == max(FONT_MIN, size)
        assert tkfont.nametofont('OpenFloHeading',
                                 root=root).actual('weight') == 'bold'
    base = tkfont.nametofont('TkDefaultFont', root=root).actual('family')
    assert tkfont.nametofont('OpenFloSmall',
                             root=root).actual('family') == base
    # The window-width scaling resizes them with the rest of the chrome.
    ed._apply_chrome_scale(force=True)
    assert abs(int(tkfont.nametofont('OpenFloLarge', root=root)
                   .actual('size'))) == ed._chrome_font_size + 1


_FAMILIES = {'TkDefaultFont', 'TkFixedFont', 'TkHeadingFont', 'TkTextFont',
             'Consolas', 'Segoe UI', 'Arial', 'Courier'}


def _font_tuples(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Tuple) and len(node.elts) >= 2
                and isinstance(node.elts[0], ast.Constant)
                and node.elts[0].value in _FAMILIES
                and isinstance(node.elts[1], ast.Constant)
                and isinstance(node.elts[1].value, int)):
            yield node


def test_no_window_asks_for_a_font_tuple():
    found = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'theme.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found += [f"{path.name}:{n.lineno}" for n in _font_tuples(tree)]
    assert not found, found


def test_the_check_finds_a_font_tuple():
    tree = ast.parse("ttk.Label(f, font=('TkDefaultFont', 8, 'bold'))")
    assert len(list(_font_tuples(tree))) == 1


# ── Colours (UI 37) ─────────────────────────────────────────────────────────

def _luminance(hex_color):
    h = hex_color.lstrip('#')
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_contrast_is_computed_as_wcag_does():
    assert round(contrast('#000000', '#ffffff'), 1) == 21.0
    assert round(contrast('#777777', '#ffffff'), 2) == 4.48


def test_every_text_colour_reads_on_its_background():
    from openflo.theme import THEMES
    low = []
    for name, pal in THEMES.items():
        for text in ('fg', 'muted', 'error', 'warning', 'success'):
            for ground in ('bg', 'panel'):
                c = contrast(pal[text], pal[ground])
                if c < 4.5:
                    low.append((name, text, ground, round(c, 2)))
        for text, ground in (('highlight_fg', 'highlight'),
                             ('console_fg', 'console_bg'),
                             ('accfg', 'accent'),
                             ('plot_fg', 'plot_bg')):
            c = contrast(pal[text], pal[ground])
            if c < 4.5:
                low.append((name, text, ground, round(c, 2)))
    assert not low, low


_COLOUR_KW = {'foreground', 'background', 'fg', 'bg', 'insertbackground',
              'selectbackground', 'selectforeground', 'activebackground',
              'activeforeground', 'disabledforeground', 'highlightcolor',
              'highlightbackground', 'troughcolor'}
# The layout editor draws the printed page, which is white in every theme
# (its desk is set from the palette once drawn).
_PAGE_DRAWING = {'ui_layout_editor.py'}


def _hard_coded_colours(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            v = kw.value
            if (kw.arg in _COLOUR_KW and isinstance(v, ast.Constant)
                    and isinstance(v.value, str) and v.value
                    and v.value != 'White'):     # savefig_background's name
                yield node.lineno, kw.arg, v.value


def test_no_widget_colour_is_hard_coded():
    found = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'theme.py' or path.name in _PAGE_DRAWING:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found += [f"{path.name}:{ln} {k}={v}"
                  for ln, k, v in _hard_coded_colours(tree)]
    assert not found, found


def test_the_check_finds_a_hard_coded_colour():
    tree = ast.parse("ttk.Label(f, foreground='red')\n"
                     "tk.Text(f, background='#1e1e1e')")
    assert len(list(_hard_coded_colours(tree))) == 2


def test_log_highlight_and_messages_follow_the_theme(editor):
    from openflo.theme import THEMES
    root, ed = editor
    log = ed._log_text
    assert str(log.cget('background')) == THEMES['light']['console_bg']
    assert str(ed.gate_tv.tag_configure('drop_target', 'background')) == \
        THEMES['light']['highlight']
    style = ttk.Style(root)
    assert str(style.lookup('Error.TLabel', 'foreground')) == \
        THEMES['light']['error']
    ed._theme_var.set('midnight')
    ed._set_theme()
    root.update()
    assert str(log.cget('background')) == THEMES['midnight']['console_bg']
    assert str(log.cget('foreground')) == THEMES['midnight']['console_fg']
    assert str(ed.gate_tv.tag_configure('drop_target', 'background')) == \
        THEMES['midnight']['highlight']
    assert str(style.lookup('Error.TLabel', 'foreground')) == \
        THEMES['midnight']['error']
    ed._theme_var.set('light')
    ed._set_theme()


# ── Theme: windows and plot background (UI 38) ─────────────────────────────

def test_theme_is_window_colour_plus_plot_background(editor):
    from openflo.theme import (
        THEMES,
        _dialog_dark_on,
        current_palette,
        theme_key,
        theme_parts,
    )
    root, ed = editor
    for key in ('light', 'dark', 'midnight', 'light-dark'):
        assert theme_key(*theme_parts(key)) == key
    assert theme_parts('no-such-theme') == ('light', 'light')
    menus = _menus(ed)
    view = [label for _t, label, _a in _entries(menus['View'])]
    assert "Theme" in view and "Plot background" in view
    assert not [label for label in view if 'Dark figures' in label]
    ed._chrome_var.set('light')
    ed._plot_bg_var.set('dark')
    ed._set_theme_parts()
    root.update()
    assert ed._theme_var.get() == 'light-dark'
    pal = current_palette()
    assert pal['bg'] == THEMES['light']['bg']
    assert pal['plot_bg'] == THEMES['midnight']['plot_bg']
    assert _dialog_dark_on(tk.Toplevel(ed))       # pop-ups follow the plot
    assert ed.status_var.get() == \
        "Light windows with a black plot background."
    ed._theme_var.set('dark')                     # e.g. from Preferences
    ed._set_theme()
    root.update()
    assert (ed._chrome_var.get(), ed._plot_bg_var.get()) == ('dark', 'light')
    assert not _dialog_dark_on(tk.Toplevel(ed))
    ed._theme_var.set('light')
    ed._set_theme()


# ── Cluster dialog (UI 40) ──────────────────────────────────────────────────

def _cluster_dialog(root, ed):
    before = set(sweep._toplevels(root))
    ed._open_cluster_dialog()
    root.update()
    (dlg,) = [w for k, w in sweep._toplevels(root).items()
              if k not in before]
    return dlg


def _shown(dlg):
    return {key for key, widgets in dlg.master._cluster_method_rows.items()
            if all(w.winfo_manager() == 'grid' for w in widgets)}


def test_cluster_dialog_shows_only_the_picked_methods_settings(editor):
    root, ed = editor
    dlg = _cluster_dialog(root, ed)
    assert _shown(dlg) == {'k', 'repro'}            # PhenoGraph
    radios = {w.cget('text'): w for w in sweep._walk(dlg)
              if w.winfo_class() == 'TRadiobutton'}
    radios['FlowSOM'].invoke()
    root.update()
    assert _shown(dlg) == {'grid', 'meta'}
    radios['Leiden'].invoke()
    root.update()
    assert _shown(dlg) == {'k', 'resolution'}
    dlg.destroy()


def test_cluster_dialog_lists_markers_by_name_and_runs_on_channels(
        editor, monkeypatch):
    root, ed = editor
    ed._channel_labels = {c: c for c in ed._channels}
    ed._channel_labels['FL2-A'] = 'Marker2'
    ran = []
    monkeypatch.setattr(type(ed), '_run_clustering',
                        lambda self, **kw: ran.append(kw))
    dlg = _cluster_dialog(root, ed)
    shown = list(ed._cluster_marker_list.get(0, 'end'))
    assert shown == ['FL1-A', 'Marker2 (FL2-A)', 'FL3-A', 'FL4-A']
    ed._cluster_marker_list.selection_clear(0, 'end')
    ed._cluster_marker_list.selection_set(1)
    run = next(w for w in sweep._walk(dlg)
               if w.winfo_class() == 'TButton' and w.cget('text') == 'Run')
    run.invoke()
    assert ran and ran[0]['channels'] == ['FL2-A']


# ── Layout editor (UI 39) and the quick figure grid (UI 24) ─────────────────

def _layout_editor(root, ed):
    from openflo.ui_layout_editor import LayoutEditorWindow
    win = LayoutEditorWindow(ed)
    root.update()
    win._redraw()
    return win


def test_layout_editor_hint_names_its_real_buttons(editor):
    root, ed = editor
    win = _layout_editor(root, ed)
    texts = [win.canvas.itemcget(i, 'text')
             for i in win.canvas.find_all()
             if win.canvas.type(i) == 'text']
    hint = next(t for t in texts if t.startswith("Empty page."))
    buttons = {w.cget('text') for w in sweep._walk(win)
               if w.winfo_class() == 'TButton'}
    named = [b for b in ("＋ Plot", "＋ Stats", "＋ Chart") if b in hint]
    assert named == ["＋ Plot", "＋ Stats", "＋ Chart"]
    assert set(named) <= buttons
    assert "Add plot" not in hint and "…." not in hint
    win.destroy()


def test_layout_editor_status_line_wraps_inside_the_window(editor):
    root, ed = editor
    win = _layout_editor(root, ed)
    win.geometry('900x700')
    win.status.set("Status: " + "a long message " * 30)
    root.update()
    lbl = win.status_label
    assert lbl.winfo_ismapped()
    assert 0 < int(lbl.cget('wraplength')) <= win.winfo_width()
    assert lbl.winfo_y() + lbl.winfo_height() <= win.winfo_height()
    assert lbl.winfo_reqwidth() <= win.winfo_width()
    win.destroy()


def test_layout_editor_undo_redo_and_save(editor, monkeypatch):
    from openflo.layout_page import text_item
    root, ed = editor
    ed._note_saved()
    win = _layout_editor(root, ed)
    assert not ed.title().startswith('*')
    win.layout.add(text_item('synthetic note'))
    win._redraw()
    assert len(win.layout.items) == 1
    assert ed.title().startswith('*')                # session now unsaved
    win.focus_force()
    root.update()
    win.event_generate('<Control-Key-z>')
    root.update()
    assert len(win.layout.items) == 0
    assert ed._layout_doc is win.doc
    win.event_generate('<Control-Key-y>')
    root.update()
    assert len(win.layout.items) == 1
    saved = []
    monkeypatch.setattr(type(ed), '_save_session',
                        lambda self: saved.append(1))
    win.event_generate('<Control-Key-s>')
    root.update()
    assert saved == [1]
    win.destroy()


def test_one_name_for_the_quick_figure_grid(editor):
    root, ed = editor
    labels = [label for _t, label, _a in _all_entries(ed)]
    assert "Quick figure grid…" in labels
    assert not [label for label in labels if 'Figure layout' in label]
    from openflo.ui_figure_layout import FigureLayoutDialog
    src = Path(SRC / 'ui_figure_layout.py').read_text(encoding='utf-8')
    assert 'self.title("Quick figure grid")' in src
    assert FigureLayoutDialog is not None
