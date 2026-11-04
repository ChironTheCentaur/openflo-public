"""Layout editor, Quick figure grid, export background and Help fixes.

- A Resolution change and Rename page are each their own undo step.
- Edit..., Duplicate and Delete are greyed out with nothing selected.
- A Statistics item refuses channel statistics without a channel, saying so.
- Background "Dark" exports a dark figure from a light preview, which is
  left as it was, even when the save fails; the Background boxes explain
  each choice.
- Quick figure grid names the channel pairs it could not find, and its
  Channel pairs box has a hover tip.
- Help > Online documentation says when no browser opened; a second F1
  raises the one user guide.

Generated data, synthetic sample names (alpha_1 ...) and generic channels
(FSC-A, SSC-A, FL1-A ...) only. Each check is shown to fail on the old code
by the group's mutation check.
"""
from __future__ import annotations

import io

import numpy as np
import pytest

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def _layout_editor(root, ed):
    from openflo.ui_layout_editor import LayoutEditorWindow
    win = LayoutEditorWindow(ed)
    win.geometry('1000x700')
    root.update()
    win._redraw()
    return win


# ── Page-level settings record their own undo step ────────────────────────
def test_resolution_change_is_its_own_undo_step(editor):
    root, ed = editor
    win = _layout_editor(root, ed)
    win.dpi_var.set('450')
    win._on_dpi()
    win._add_text()
    assert len(win.layout.items) == 1
    ed._note_saved()
    win.dpi_var.set('600')
    win._on_dpi()
    assert win.doc.export_dpi == 600
    assert ed.title().startswith('*')        # marked unsaved at once
    win._undo()
    assert win.doc.export_dpi == 450         # only the dpi change undone
    assert win.dpi_var.get() == '450'
    assert len(win.layout.items) == 1        # the text box stays
    win._redo()
    assert win.doc.export_dpi == 600
    assert len(win.layout.items) == 1
    win.destroy()


def test_rename_and_page_size_are_their_own_undo_steps(editor, monkeypatch):
    from openflo import ui_layout_editor as mod
    root, ed = editor
    win = _layout_editor(root, ed)
    old_name = win.layout.name
    win._add_text()
    monkeypatch.setattr(mod.simpledialog, 'askstring',
                        lambda *a, **k: 'Summary page')
    win._rename_page()
    assert win.layout.name == 'Summary page'
    win._undo()
    assert win.layout.name == old_name
    assert len(win.layout.items) == 1        # the text box stays
    sizes = [p for p in mod.PAGE_SIZES if p != win.layout.page]
    win.page_var.set(sizes[0])
    win._on_page_size()
    assert win.layout.page == sizes[0]
    win._undo()
    assert win.layout.page != sizes[0]
    assert len(win.layout.items) == 1
    win.destroy()


# ── Item buttons follow the selection ─────────────────────────────────────
def test_item_buttons_greyed_out_with_nothing_selected(editor):
    root, ed = editor
    win = _layout_editor(root, ed)
    labels = [b.cget('text') for b in win._item_buttons]
    assert labels == ["Edit…", "Duplicate", "Delete"]
    assert all(b.instate(['disabled']) for b in win._item_buttons)
    win._add_text()                          # a new item is selected
    assert all(not b.instate(['disabled']) for b in win._item_buttons)
    win._delete_selected()
    assert not win.layout.items
    assert all(b.instate(['disabled']) for b in win._item_buttons)
    win.status.set('')
    win._duplicate_selected()                # Delete key / menu path
    assert win.status.get().startswith("Select an item first")
    win.destroy()


# ── Channel statistics need a channel ─────────────────────────────────────
def test_stats_item_says_channel_statistics_need_a_channel(editor,
                                                           monkeypatch):
    from openflo import ui_layout_dialogs as dlgs
    from openflo.layout_page import stats_item
    root, ed = editor
    win = _layout_editor(root, ed)
    warned, got = [], []
    monkeypatch.setattr(dlgs.messagebox, 'showwarning',
                        lambda title, msg, **kw: warned.append(msg))
    item = stats_item(stats=('Count', 'Median', 'CV'), channel=None)
    dlg = dlgs.ItemDialog(win, item, win._choices(), on_ok=got.append)
    root.update()
    dlg._ok()
    assert len(warned) == 1 and not got
    assert 'Median, CV' in warned[0] and 'channel' in warned[0]
    assert dlg.winfo_exists()                # stays open to fix it
    dlg.chan_var.set('FL1-A')
    dlg._ok()
    assert len(warned) == 1
    assert got[0]['channel'] == 'FL1-A'
    assert got[0]['stats'] == ['Count', 'Median', 'CV']
    win.destroy()


def test_one_channel_statistic_without_a_channel_is_named_alone(editor,
                                                                monkeypatch):
    from openflo import ui_layout_dialogs as dlgs
    from openflo.layout_page import stats_item
    root, ed = editor
    win = _layout_editor(root, ed)
    warned, got = [], []
    monkeypatch.setattr(dlgs.messagebox, 'showwarning',
                        lambda title, msg, **kw: warned.append(msg))
    item = stats_item(stats=('Count', 'Median'), channel=None)
    dlg = dlgs.ItemDialog(win, item, win._choices(), on_ok=got.append)
    root.update()
    dlg.chan_var.set('')
    dlg._ok()
    assert warned == ["Median needs a channel. Pick one in Channel, or "
                      "untick it."]
    assert not got and dlg.winfo_exists()
    dlg.destroy()
    win.destroy()


# ── Dark exports dark ─────────────────────────────────────────────────────
def _light_figure():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    fig = Figure(figsize=(3, 2), dpi=50)
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel('FL1-A')
    ax.text(0.5, 0.5, 'note', color='black')
    ax.legend(['FL1-A'])
    return fig, ax


def _corner(buf):
    import matplotlib.image as mpimg
    buf.seek(0)
    img = mpimg.imread(buf, format='png')
    return np.round(img[0, 0] * 255).astype(int)


def test_dark_background_exports_a_dark_figure_from_a_light_one():
    from matplotlib.colors import to_hex, to_rgb

    from openflo.theme import THEME_ALL_DARK, THEMES, savefig_background
    fig, ax = _light_figure()
    tick_before = ax.xaxis.get_major_ticks()[0].label1.get_color()
    leg = ax.get_legend()
    legend_before = (to_hex(leg.get_frame().get_facecolor()),
                     to_hex(leg.get_frame().get_edgecolor()),
                     to_hex(leg.get_texts()[0].get_color()))
    during = []
    real_save = fig.savefig

    def save(*a, **k):
        # Labels, free text and the legend are light while the dark file
        # is written.
        during.append((to_hex(ax.xaxis.label.get_color()),
                       to_hex(ax.texts[0].get_color()),
                       to_hex(leg.get_texts()[0].get_color())))
        return real_save(*a, **k)
    fig.savefig = save
    buf = io.BytesIO()
    savefig_background(fig, buf, background='Dark', dpi=50, tight=False,
                       fmt='png')
    del fig.savefig
    fg = to_hex(THEMES[THEME_ALL_DARK]['plot_fg'])
    assert during == [(fg, fg, fg)]
    want = np.round(np.array(to_rgb(THEMES[THEME_ALL_DARK]['plot_bg']))
                    * 255).astype(int)
    got = _corner(buf)
    assert list(got[:3]) == list(want)
    assert got[3] == 255
    # The preview is unchanged after the save.
    assert to_hex(fig.get_facecolor()) == '#ffffff'
    assert to_hex(ax.get_facecolor()) == '#ffffff'
    assert ax.xaxis.label.get_color() == 'black'
    assert ax.texts[0].get_color() == 'black'
    assert ax.xaxis.get_major_ticks()[0].label1.get_color() == tick_before
    leg = ax.get_legend()
    assert to_hex(leg.get_frame().get_facecolor()) == legend_before[0]
    assert to_hex(leg.get_frame().get_edgecolor()) == legend_before[1]
    assert to_hex(leg.get_texts()[0].get_color()) == legend_before[2]
    # White is still white.
    buf = io.BytesIO()
    savefig_background(fig, buf, background='White', dpi=50, tight=False,
                       fmt='png')
    assert list(_corner(buf)) == [255, 255, 255, 255]


def test_a_dark_save_that_fails_leaves_the_preview_light():
    from matplotlib.colors import to_hex

    from openflo.theme import savefig_background
    fig, ax = _light_figure()
    legend_ink = to_hex(ax.get_legend().get_texts()[0].get_color())

    def save(*a, **k):
        raise OSError('disk full')
    fig.savefig = save
    with pytest.raises(OSError):
        savefig_background(fig, io.BytesIO(), background='Dark', dpi=50,
                           tight=False, fmt='png')
    assert to_hex(fig.get_facecolor()) == '#ffffff'
    assert to_hex(ax.get_facecolor()) == '#ffffff'
    assert ax.xaxis.label.get_color() == 'black'
    assert ax.texts[0].get_color() == 'black'
    assert to_hex(ax.get_legend().get_texts()[0].get_color()) == legend_ink


def test_dark_background_keeps_an_already_dark_figure():
    from matplotlib.colors import to_rgb

    from openflo.theme import savefig_background
    fig, _ax = _light_figure()
    fig.set_facecolor('#202830')
    buf = io.BytesIO()
    savefig_background(fig, buf, background='Dark', dpi=50, tight=False,
                       fmt='png')
    want = np.round(np.array(to_rgb('#202830')) * 255).astype(int)
    assert list(_corner(buf)[:3]) == list(want)


def test_background_boxes_explain_dark(editor):
    import tkinter as tk

    from matplotlib.figure import Figure

    from openflo.tool_window import background_selector
    root, ed = editor
    top = tk.Toplevel(ed)
    _var, combo = background_selector(top)
    tip = ed._tip_texts[str(combo)]
    assert 'Dark:' in tip and 'whichever theme' in tip
    assert 'Transparent' in tip and 'Translucent' in tip
    top.destroy()
    # The Quick figure grid preview has its own Background box.
    fig = Figure(figsize=(3, 2))
    fig.add_subplot().plot([0, 1], [0, 1])
    before = set(sweep._toplevels(root))
    ed._show_figure_preview(fig, ['alpha_1'])
    root.update()
    win = next(t for k, t in sweep._toplevels(root).items()
               if k not in before)
    box = next(w for w in sweep._walk(win)
               if w.winfo_class() == 'TCombobox'
               and 'Dark' in w.cget('values'))
    assert 'Dark:' in ed._tip_texts[str(box)]
    win.destroy()


# ── Quick figure grid channel pairs ───────────────────────────────────────
def _grid(ed, monkeypatch, pairs):
    from openflo import editor_figure
    warned, shown = [], []
    monkeypatch.setattr(editor_figure.messagebox, 'showwarning',
                        lambda title, msg, **kw: warned.append(msg))
    monkeypatch.setattr(type(ed), '_show_figure_preview',
                        lambda self, fig, samples=None: shown.append(fig))
    ed._build_and_preview_figure({'layout': 'per_pair', 'pairs': pairs,
                                  'ncols': 3, 'gates': False})
    return warned, shown


def test_quick_figure_grid_names_pairs_it_cannot_find(editor, monkeypatch):
    _root, ed = editor
    warned, shown = _grid(ed, monkeypatch, 'FLX-A / FLY-A')
    assert not shown                         # no stand-in FSC-A/SSC-A panel
    assert len(warned) == 1
    assert "'FLX-A'" in warned[0] and "'FLY-A'" in warned[0]
    warned, shown = _grid(ed, monkeypatch, 'FL1-A/FL2-A, FL3-A/FLZ-A')
    assert len(warned) == 1
    assert "FL3-A/FLZ-A" in warned[0] and "'FLZ-A'" in warned[0]
    assert "'FL3-A'" not in warned[0].split("no channel", 1)[1]
    assert len(shown) == 1 and len(shown[0].axes) == 1
    title = shown[0].axes[0].get_title()
    assert 'FL1-A' in title and 'FL2-A' in title
    # A good list builds without a warning; 'x' between spaces still works.
    warned, shown = _grid(ed, monkeypatch, 'FL1-A/FL2-A\nFSC-A x SSC-A')
    assert not warned and len(shown[0].axes) == 2


def test_channel_pairs_box_has_a_tip(editor):
    from openflo.ui_figure_layout import FigureLayoutDialog
    root, ed = editor
    dlg = FigureLayoutDialog(ed, 1, 'dot', '', lambda opts: None)
    root.update()
    tip = ed._tip_texts[str(dlg.pairs_txt)]
    assert 'X/Y' in tip and 'FL1-A/FL2-A' in tip and 'comma' in tip
    dlg.destroy()


# ── Help ──────────────────────────────────────────────────────────────────
def test_online_documentation_says_when_no_browser_opened(editor,
                                                          monkeypatch):
    import webbrowser

    from openflo.update import DOCS_URL
    _root, ed = editor
    monkeypatch.setattr(webbrowser, 'open', lambda url: False)
    ed._open_documentation()
    msg = ed.status_var.get()
    assert msg.startswith("Could not open a browser") and DOCS_URL in msg
    monkeypatch.setattr(webbrowser, 'open', lambda url: True)
    ed._open_documentation()
    assert ed.status_var.get().startswith("Opened the OpenFlo documentation")


def test_second_f1_raises_the_one_user_guide(editor):
    from openflo.ui_guide import GuideWindow
    from openflo.user_guide import topics
    root, ed = editor
    ed._open_guide()
    root.update()
    slug = topics()[1][0]
    ed._open_guide(topic=slug)
    root.update()
    guides = [w for w in sweep._walk(root) if isinstance(w, GuideWindow)]
    assert len(guides) == 1
    assert guides[0].current == slug
    guides[0].destroy()
    root.update()
    ed._open_guide()                         # closed: a fresh one opens
    root.update()
    guides = [w for w in sweep._walk(root) if isinstance(w, GuideWindow)]
    assert len(guides) == 1
    guides[0].destroy()
