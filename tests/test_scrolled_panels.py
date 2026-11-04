"""Long lists in dialogs scroll; the buttons below them stay on screen.

Nine dialogs list one row per sample, gate, channel, cluster or result. With
a long list on a small screen the window used to grow past the screen, or the
list pushed the Apply / Cancel row out of the window, and the mouse wheel did
nothing over the list. They now build the list with
``tool_window.scrolled_body``, which asks for room up to what the screen
allows, gives the button row its share, and scrolls with the wheel.

Each dialog is opened through the editor (whose window-map hook fits every
new window to the screen) on a virtual screen no larger than 1024x768, with
generated samples (s1, s2, ...) and channels (FL1-A, FL2-A, ...). Nothing
waits for a person: grabs and modal waits return at once and message boxes
are stubbed.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests._tk_wait import settle
from tests.conftest import editor_or_skip, skip_unless_the_screen_holds

BASE_SCALING = 96 / 72             # Tk's points-to-pixels at 100 %
MARGIN_W, MARGIN_H = 16, 72        # what tool_window.fit_window leaves free


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def screen(monkeypatch):
    """Pin the screen size every window measures itself against."""
    def use(width, height):
        monkeypatch.setattr(tk.Misc, 'winfo_screenwidth', lambda self: width)
        monkeypatch.setattr(tk.Misc, 'winfo_screenheight',
                            lambda self: height)
        return width, height
    return use


@pytest.fixture
def quiet(monkeypatch):
    """Nothing in a test waits for a click: grabs and modal waits return at
    once, message boxes answer without showing, no background jobs start."""
    from tkinter import filedialog, messagebox
    monkeypatch.setattr(tk.Misc, 'wait_window', lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, 'grab_set', lambda *a, **k: None)
    monkeypatch.setattr(tk.Misc, 'grab_set_global', lambda *a, **k: None)
    for name in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: 'ok')
    for name in ('askyesno', 'askokcancel', 'askyesnocancel'):
        monkeypatch.setattr(messagebox, name, lambda *a, **k: False)
    for name in ('askopenfilename', 'askopenfilenames', 'asksaveasfilename',
                 'askdirectory'):
        monkeypatch.setattr(filedialog, name, lambda *a, **k: '')
    from openflo import async_task
    monkeypatch.setattr(async_task, 'run_async', lambda *a, **k: None)


def _make_editor(width, height):
    """A shown editor, smaller than the screen, on an unscaled display.
    Shown, because a dialog that is transient for a withdrawn window never
    maps, and the editor fits a window when it maps."""
    root, ed = editor_or_skip(accept_confirms=False)
    root.tk.call('tk', 'scaling', BASE_SCALING)
    skip_unless_the_screen_holds(root, width, height)
    ed.geometry(f'{min(800, width - 100)}x{min(500, height - 150)}+0+0')
    ed.deiconify()
    settle(root, 0.2, pause=0.005)
    return root, ed


@pytest.fixture
def editor_on(screen, quiet):
    """``editor_on(w, h)``: a virtual w x h screen and an editor on it. Every
    window is destroyed at the end."""
    made = []

    def make(width, height):
        screen(width, height)
        root, ed = _make_editor(width, height)
        made.append(root)
        return ed
    yield make
    for root in made:
        try:
            root.destroy()
        except tk.TclError:
            pass


@pytest.fixture
def root(screen):
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot start: {exc}")
    r.withdraw()
    r.tk.call('tk', 'scaling', BASE_SCALING)
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


# ── generated data ──────────────────────────────────────────────────────────

def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _add_sample(ed, name, cols=('FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'),
                extra=None, n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 1000, (n, len(cols))), columns=list(cols))
    for col, values in (extra or {}).items():
        df[col] = values
    fluor = [c for c in cols if c.startswith('FL')]
    ed._samples[name] = SimpleNamespace(
        name=name, path=f'{name}.fcs', data=df, fluor_channels=fluor,
        channel_labels={c: c for c in df.columns}, data_transforms={})
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True


def _activate(ed, name):
    df = ed._samples[name].data
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()
    ed._set_active_sample(name)


def _samples(ed, n):
    """s1 (active, with one gate) and n other samples."""
    for i in range(n + 1):
        _add_sample(ed, f's{i + 1}', seed=i)
    _activate(ed, 's1')
    ed._add_gate({'kind': 'threshold', 'name': 'Pos', 'channel': 'FL1-A',
                  'op': '>', 'value': 500.0, 'parent_id': None}, audit=False)


def _gates(ed, n):
    """s1 with n threshold gates."""
    _add_sample(ed, 's1')
    _activate(ed, 's1')
    for i in range(n):
        ed._add_gate({'kind': 'threshold', 'name': f'G{i + 1}',
                      'channel': ('FL1-A', 'FL2-A')[i % 2], 'op': '>',
                      'value': 10.0 + i, 'parent_id': None}, audit=False,
                     replace_1d=False)


def _labelled(ed, col, n):
    """s1 whose ``col`` column has n distinct values."""
    _add_sample(ed, 's1', extra={col: np.arange(200) % n})
    _activate(ed, 's1')


def _channels(ed, n):
    """s1 with n fluorescence channels, plus two more samples."""
    cols = ['FSC-A', 'SSC-A'] + [f'FL{i + 1}-A' for i in range(n)]
    for i in range(3):
        _add_sample(ed, f's{i + 1}', cols=cols, n=100, seed=i)
    _activate(ed, 's1')


def _toplevels(ed):
    return [w for w in _walk(ed.master) if isinstance(w, tk.Toplevel)]


def _opened(ed, open_it):
    """Run ``open_it`` and return the one window it opened, once the editor
    has fitted it (the fit runs after the window maps)."""
    before = {str(w) for w in _toplevels(ed)}
    open_it()
    new = [w for w in _toplevels(ed) if str(w) not in before]
    assert len(new) == 1, f'expected one new window, got {len(new)}'
    settle(ed, 0.5, pause=0.005)
    return new[0]


# ── the nine dialogs ────────────────────────────────────────────────────────
#
# Each entry: how to fill the editor with n rows, how to open the dialog, its
# main and closing buttons, and the fixed size it opened at before.

def _open_annotate_populations(ed, n, _tmp):
    _labelled(ed, 'leiden', n)
    return _opened(ed, lambda: ed._annotate_populations('leiden'))


def _open_annotate_clusters(ed, n, _tmp):
    _labelled(ed, 'cluster', n)
    return _opened(ed, ed._annotate_clusters)


def _open_combinations(ed, n, _tmp):
    _gates(ed, n)
    return _opened(ed, ed._open_combination_dialog)


def _open_copy_gates(ed, n, _tmp):
    _samples(ed, n)
    return _opened(ed, ed._open_copy_gates_dialog)


def _open_boolean(ed, n, _tmp):
    _gates(ed, n)
    return _opened(ed, ed._open_boolean_dialog)


def _open_template_apply(ed, n, _tmp):
    _samples(ed, n)
    return _opened(ed, ed._ask_template_apply)   # wait_window returns at once


def _open_fmo(ed, n, _tmp):
    _channels(ed, n)
    return _opened(ed, ed._open_fmo_gating)


def _open_markers(ed, n, _tmp):
    _channels(ed, n)
    ed._open_pipeline_workspace()
    ed.update_idletasks()
    view = ed._workspace_panel._active_view()
    return _opened(ed, view._pick_markers)


def _open_results(ed, n, tmp):
    from openflo.workspace import ResultsViewer
    for i in range(n):
        (tmp / f'r{i + 1:03d}_events.csv').write_text('FL1-A,cluster\n1,0\n')
        (tmp / f'r{i + 1:03d}_clusters.csv').write_text('cluster,n\n0,1\n')
    return _opened(ed, lambda: ResultsViewer(ed, str(tmp), editor=ed))


PANELS = {
    # name: (opener, main button, close button, old fixed size)
    'annotate populations': (_open_annotate_populations, 'Apply', 'Cancel',
                             (360, 440)),
    'annotate clusters': (_open_annotate_clusters, 'Apply', 'Cancel',
                          (360, 440)),
    'combination gates': (_open_combinations, 'Create', 'Cancel', (420, 480)),
    'copy gates': (_open_copy_gates, 'Copy', 'Cancel', (460, 440)),
    'boolean gate': (_open_boolean, 'Create', 'Cancel', (360, 420)),
    'template apply': (_open_template_apply, 'Apply', 'Cancel', (420, 460)),
    'fmo gating': (_open_fmo, 'Apply', 'Close', (540, 540)),
    'clustering markers': (_open_markers, 'Apply', 'Cancel', (320, 460)),
    'results viewer': (_open_results, 'Open folder', 'Load in editor',
                       (900, 720)),
}

# Rows that fill more than the old fixed window could show but fit a
# 1024x768 screen. The results viewer is left out: its old window was
# already as tall as such a screen allows.
FITS = {'annotate populations': 20, 'annotate clusters': 20,
        'combination gates': 26, 'copy gates': 22, 'boolean gate': 24,
        'template apply': 21, 'fmo gating': 21, 'clustering markers': 28}

FEW = {'combination gates': 2, 'boolean gate': 2, 'results viewer': 2}


# ── measuring ───────────────────────────────────────────────────────────────

def _list_canvas(dlg):
    """The canvas that scrolls the dialog's list."""
    found = [w for w in _walk(dlg) if isinstance(w, tk.Canvas)
             and str(w.cget('yscrollcommand'))]
    assert len(found) == 1, f'expected one scrolling list, found {len(found)}'
    return found[0]


def _button(dlg, text):
    found = [w for w in _walk(dlg) if isinstance(w, ttk.Button)
             and w.cget('text') == text]
    assert found, f'no {text!r} button in {dlg.title()!r}'
    return found[0]


def _box(w):
    return (w.winfo_rootx(), w.winfo_rooty(),
            w.winfo_rootx() + w.winfo_width(),
            w.winfo_rooty() + w.winfo_height())


def _inside(inner, outer):
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def _assert_on_screen(dlg, sw, sh):
    x, y, x1, y1 = _box(dlg)
    assert x >= 0 and y >= 0 and x1 <= sw and y1 <= sh, (
        f'{dlg.title()!r} is {x1 - x}x{y1 - y} at +{x}+{y}: it runs off a '
        f'{sw}x{sh} screen')


def _assert_buttons_shown(dlg, *texts):
    win = _box(dlg)
    for text in texts:
        b = _button(dlg, text)
        assert b.winfo_viewable(), (
            f'{dlg.title()!r}: the {text!r} button is not shown (the list '
            f'pushed it out of the window)')
        assert _inside(_box(b), win), (
            f'{dlg.title()!r}: the {text!r} button at {_box(b)} is cut off '
            f'by the window edge {win}')


def _wheel_down(dlg, cv):
    """Point at the list and turn the wheel one notch down."""
    holder = cv.master
    holder.event_generate('<Enter>', x=5, y=5)
    cv.event_generate('<MouseWheel>', delta=-120, x=5, y=5)
    dlg.update()
    holder.event_generate('<Leave>', x=5, y=5)
    dlg.update()


# ── many rows: the list scrolls, the window and its buttons stay on screen ──

@pytest.mark.parametrize('name', sorted(PANELS))
def test_a_long_list_scrolls_and_keeps_the_buttons_on_screen(
        editor_on, tmp_path, name):
    """With a long list on a small screen the window runs off the screen, or
    the list pushes Apply / Cancel out of it, or the wheel does nothing."""
    opener, main, close, _old = PANELS[name]
    sw, sh = 1024, 600
    ed = editor_on(sw, sh)
    dlg = opener(ed, 120, tmp_path)
    _assert_on_screen(dlg, sw, sh)
    _assert_buttons_shown(dlg, main, close)
    cv = _list_canvas(dlg)
    top, bottom = (float(v) for v in cv.yview())
    assert top == 0.0 and bottom < 1.0, (
        f'{name}: 120 rows on a {sw}x{sh} screen should scroll, yview '
        f'is {cv.yview()}')
    _wheel_down(dlg, cv)
    assert float(cv.yview()[0]) > 0.0, (
        f'{name}: the mouse wheel over the list did not scroll it')


# ── rows that fit the screen: all show, no scrolling ────────────────────────

@pytest.mark.parametrize('name', sorted(FITS))
def test_a_list_that_fits_the_screen_shows_every_row(editor_on, tmp_path,
                                                     name):
    """A list taller than the old fixed window but short enough for the
    screen should open with every row showing, not behind a scrollbar."""
    opener, main, close, _old = PANELS[name]
    sw, sh = 1024, 768
    ed = editor_on(sw, sh)
    dlg = opener(ed, FITS[name], tmp_path)
    _assert_on_screen(dlg, sw, sh)
    _assert_buttons_shown(dlg, main, close)
    cv = _list_canvas(dlg)
    assert tuple(float(v) for v in cv.yview()) == (0.0, 1.0), (
        f'{name}: {FITS[name]} rows fit a {sw}x{sh} screen but the list '
        f'scrolls (yview {cv.yview()})')


# ── a few rows: no scrolling, and the window keeps its old size ─────────────

@pytest.mark.parametrize('name', sorted(PANELS))
def test_a_short_list_keeps_the_old_window_size(editor_on, tmp_path, name):
    """A short list should show without scrolling in a window the size it
    always was, not shrunk around three rows."""
    opener, main, close, (old_w, old_h) = PANELS[name]
    sw, sh = 1024, 768
    ed = editor_on(sw, sh)
    dlg = opener(ed, FEW.get(name, 3), tmp_path)
    _assert_on_screen(dlg, sw, sh)
    _assert_buttons_shown(dlg, main, close)
    cv = _list_canvas(dlg)
    assert tuple(float(v) for v in cv.yview()) == (0.0, 1.0)
    want_w, want_h = min(old_w, sw - MARGIN_W), min(old_h, sh - MARGIN_H)
    got_w, got_h = dlg.winfo_width(), dlg.winfo_height()
    assert got_w >= want_w - 2 and got_h >= want_h - 2, (
        f'{name}: opened at {got_w}x{got_h}, smaller than its old '
        f'{want_w}x{want_h}')


# ── scrolled_body itself ────────────────────────────────────────────────────

def test_scrolled_body_passes_padding_to_its_pack(root):
    """The dialogs pad their list from the window edge through pack_kw; if it
    were dropped the list would sit flush against the window border."""
    from openflo.tool_window import scrolled_body
    top = tk.Toplevel(root)
    top.geometry('300x200')
    inner = scrolled_body(top, pack_kw={'padx': 10, 'pady': (0, 6)})
    holder = inner.master.master
    info = holder.pack_info()
    assert int(info['padx']) == 10
    pady = info['pady']
    if not isinstance(pady, tuple):
        pady = root.tk.splitlist(str(pady))
    assert tuple(int(str(v)) for v in pady) == (0, 6)
    # The defaults it adds to stay.
    assert info['side'] == 'top' and info['fill'] == 'both'
    assert int(info['expand']) == 1
    # And an override wins over a default.
    inner2 = scrolled_body(top, pack_kw={'fill': 'x', 'expand': False})
    info2 = inner2.master.master.pack_info()
    assert info2['fill'] == 'x' and int(info2['expand']) == 0
    top.destroy()


def test_scrolled_body_leaves_room_for_the_buttons_below(root, screen):
    """A tall list above a button row on a short screen: after the window is
    fitted, the button must still be inside it, not pushed off the bottom."""
    from openflo.tool_window import fit_window, scrolled_body
    _sw, sh = screen(1024, 600)
    top = tk.Toplevel(root)
    top.geometry('300x200+0+0')
    # A header taller than the 220 px scrolled_body keeps free by itself.
    tk.Frame(top, height=260, width=280).pack(side='top', fill='x')
    body = scrolled_body(top)
    for i in range(80):
        tk.Checkbutton(body, text=f'row {i + 1}').pack(anchor='w')
    bar = tk.Frame(top)
    bar.pack(side='bottom', fill='x')
    ok = tk.Button(bar, text='OK')
    ok.pack(side='right')
    top.update()
    fit_window(top)
    settle(top, 0.3, pause=0.005)
    assert top.winfo_height() <= sh - MARGIN_H
    assert ok.winfo_viewable(), 'the list pushed the OK button out'
    assert _inside(_box(ok), _box(top))
    cv = body.master
    assert float(cv.yview()[1]) < 1.0             # the list scrolls instead
    top.destroy()


def _long_body(root, rows=60):
    from openflo.tool_window import scrolled_body
    top = tk.Toplevel(root)
    top.geometry('300x200+0+0')
    body = scrolled_body(top)
    for i in range(rows):
        ttk.Checkbutton(body, text=f'row {i}').pack(anchor='w')
    top.update()
    holder = body.master.master
    return top, body, holder


def test_closing_a_list_under_the_pointer_leaves_the_wheel_working(root):
    """Escape or Enter closes a dialog with the pointer still over its
    list, so the list never sees the pointer leave. The wheel stayed bound
    to the closed list and the next turn anywhere raised an error, which
    the editor shows as its "Something went wrong" window."""
    errors = []
    root.report_callback_exception = lambda *exc: errors.append(exc[1])
    top, _body, holder = _long_body(root)
    holder.event_generate('<Enter>')
    top.destroy()
    root.update()
    root.event_generate('<Button-5>')
    root.event_generate('<MouseWheel>', delta=-120)
    root.update()
    assert errors == []
    assert root.bind_all('<Button-5>') == ''         # the wheel is let go


def test_closing_one_list_keeps_the_wheel_on_the_list_under_the_pointer(
        root):
    """Closing a list must not take the wheel from another list the
    pointer has moved onto since."""
    top_a, _a, holder_a = _long_body(root)
    top_b, body_b, holder_b = _long_body(root)
    holder_a.event_generate('<Enter>')
    holder_b.event_generate('<Enter>')                # the pointer moved on
    top_a.destroy()
    root.update()
    canvas_b = body_b.master
    before = canvas_b.yview()
    canvas_b.event_generate('<MouseWheel>', delta=-120)
    root.update()
    assert canvas_b.yview()[0] > before[0]
    top_b.destroy()
