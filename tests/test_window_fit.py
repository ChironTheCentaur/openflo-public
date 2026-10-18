"""Windows open big enough for their content, centred on the main window and
on the screen; the DPI declaration on Windows; tooltips that can't be left
behind as an untitled box.

Every test builds its own small window, so each case is generated here, and
each is checked both ways: the behaviour with the fix, and the failure the
same window shows without it.
"""
from __future__ import annotations

import os
import time
import tkinter as tk
from types import SimpleNamespace

import pytest

from tests.conftest import gui_unavailable, skip_unless_the_screen_holds

os.environ.setdefault('MPLBACKEND', 'Agg')


BASE_SCALING = 96 / 72             # Tk's points-to-pixels at 100 %


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    r.withdraw()
    # As on an unscaled display, whatever DPI this one reports.
    r.tk.call('tk', 'scaling', BASE_SCALING)
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def screen(monkeypatch):
    def use(width, height):
        monkeypatch.setattr(tk.Misc, 'winfo_screenwidth', lambda self: width)
        monkeypatch.setattr(tk.Misc, 'winfo_screenheight',
                            lambda self: height)
    use(1366, 768)
    return use


def _crowded(root, size='300x200'):
    """A window whose content needs far more than its fixed size."""
    top = tk.Toplevel(root)
    top.geometry(size)
    tk.Text(top, height=30, width=70).pack(fill='both', expand=True)
    btn = tk.Button(top, text='Close')
    btn.pack(side='bottom')
    top.update()
    return top, btn


# ── fit_window ────────────────────────────────────────────────────────────

def test_fit_grows_a_window_to_show_its_content(root, screen):
    from openflo.tool_window import fit_window
    top, btn = _crowded(root)
    assert not btn.winfo_viewable()                   # the failure, as built
    fit_window(top)
    top.update()
    assert btn.winfo_viewable()
    assert top.winfo_width() >= top.winfo_reqwidth()
    assert top.winfo_height() >= top.winfo_reqheight()


def test_fit_keeps_a_larger_chosen_size(root, screen):
    from openflo.tool_window import fit_window
    top = tk.Toplevel(root)
    top.geometry('900x500')
    tk.Button(top, text='OK').pack()
    top.update()
    w, h, _x, _y = fit_window(top)
    assert (w, h) == (900, 500)


def test_fit_scales_a_chosen_size_with_the_display(root, screen):
    from openflo.tool_window import fit_window, ui_scale
    top = tk.Toplevel(root)
    top.geometry('400x300')
    tk.Button(top, text='OK').pack()
    top.update()
    assert ui_scale(top) == 1.0
    root.tk.call('tk', 'scaling', BASE_SCALING * 1.04)   # a few DPI off
    assert ui_scale(top) == 1.0
    root.tk.call('tk', 'scaling', BASE_SCALING * 1.5)    # a 150 % display
    assert ui_scale(top) == 1.5
    w, h, _x, _y = fit_window(top)
    assert (w, h) == (600, 450)


def test_fit_caps_a_window_at_the_screen(root, screen):
    from openflo.tool_window import fit_window
    screen(640, 480)
    top = tk.Toplevel(root)
    top.geometry('2000x1500')
    top.update()
    w, h, x, y = fit_window(top)
    assert w <= 640 and h <= 480
    assert x >= 0 and y >= 0 and x + w <= 640 and y + h <= 480


def test_fit_centres_on_the_anchor_and_stays_on_screen(root, screen):
    from openflo.tool_window import fit_window
    main = tk.Toplevel(root)
    main.geometry('800x600+100+50')
    main.update()
    top = tk.Toplevel(root)
    top.geometry('400x300')
    top.update()
    w, h, x, y = fit_window(top, anchor=main)
    mx, my = main.winfo_rootx(), main.winfo_rooty()
    assert abs((x + w / 2) - (mx + 400)) <= 1
    assert abs((y + h / 2) - (my + 300)) <= 1
    # A main window hanging off the right edge: the dialog is pulled back.
    main.geometry('800x600+1200+50')
    main.update()
    w, h, x, y = fit_window(top, anchor=main)
    assert x + w <= 1366


def test_fit_centres_on_the_screen_without_a_shown_anchor(root, screen):
    from openflo.tool_window import fit_window
    hidden = tk.Toplevel(root)
    hidden.withdraw()
    top = tk.Toplevel(root)
    top.geometry('400x300')
    top.update()
    w, h, x, y = fit_window(top, anchor=hidden)
    assert (x, y) == ((1366 - 400) // 2, (768 - 300) // 2)


def test_grow_to_content_after_text_is_added(root, screen):
    from openflo.tool_window import grow_to_content
    top = tk.Toplevel(root)
    top.geometry('200x80')
    top.update()
    lbl = tk.Label(top, text='\n'.join(['a line of filled-in text'] * 12))
    lbl.pack()
    top.update()
    assert top.winfo_height() < top.winfo_reqheight()     # outgrown
    grow_to_content(top)
    top.update()
    assert top.winfo_height() >= top.winfo_reqheight()
    assert top.winfo_width() >= 200                        # never shrinks


def test_scrolled_body_asks_for_its_content_and_scrolls_past_the_screen(
        root, screen):
    from openflo.tool_window import fit_window, scrolled_body
    top = tk.Toplevel(root)
    top.geometry('200x150')
    body = scrolled_body(top)
    for i in range(8):
        tk.Checkbutton(body, text=f'option number {i}').pack(anchor='w')
    top.update()
    fit_window(top)
    top.update()
    # Room on the screen: all eight show, no scrolling needed.
    assert top.winfo_height() >= body.winfo_reqheight()
    screen(1366, 300)                     # a screen too short for them
    for i in range(30):
        tk.Checkbutton(body, text=f'more {i}').pack(anchor='w')
    top.update()
    canvas = body.master
    assert canvas.winfo_reqheight() <= 300           # asks for the screen
    assert float(canvas.yview()[1]) < 1.0            # and scrolls the rest


# ── The editor fits each window as it opens ──────────────────────────────

def _editor(root):
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.update()
    return ed


def _settle(root, ms=200):
    """Run the event loop for a while: the editor finishes opening a window
    on timers (on Windows a title-bar repaint nudge resizes it by a pixel
    and back 30 ms after each map), so act as the user only once those ran."""
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        root.update()
        time.sleep(0.005)


def test_the_editor_fits_and_centres_a_dialog_once(root, screen):
    ed = _editor(root)
    top, btn = _crowded(ed)
    _settle(root)
    assert btn.winfo_viewable()
    ex, ew = ed.winfo_rootx(), ed.winfo_width()
    cx = top.winfo_x() + top.winfo_width() / 2
    cy = top.winfo_y() + top.winfo_height() / 2
    expect_x = min(max(ex + ew / 2, top.winfo_width() / 2),
                   1366 - top.winfo_width() / 2)
    assert abs(cx - expect_x) <= 2
    assert top.winfo_y() >= 0 and cy <= 768
    # Restored from minimised: left where and as big as the user had it.
    top.geometry('500x400+10+10')
    root.update()
    top.withdraw()
    root.update()
    top.deiconify()
    _settle(root)
    assert (top.winfo_width(), top.winfo_x()) == (500, 10)


def test_the_title_bar_nudge_keeps_the_fitted_size(root, screen,
                                                   monkeypatch):
    """On Windows the editor nudges each new window's size by a pixel and
    back so its title bar repaints. It read the size from winfo_width/height,
    which on Windows still report the size from before the fit for a while,
    so the nudge could put the window back to its unfitted size and hide the
    controls again (seen on windows-latest: Close not shown). The nudge is
    now given the size the fit applied. Simulated here: a nudge that, like
    Windows, sees only the pre-fit size unless told the fitted one."""
    from openflo.editor_chrome import ChromeMixin
    stale = {}

    def windows_like(self, win, nudge=False, size=None):
        if nudge:
            w, h = size or stale.get(str(win), (win.winfo_width(),
                                                win.winfo_height()))
            win.geometry(f'{w + 1}x{h}')
            win.update_idletasks()
            win.geometry(f'{w}x{h}')
    monkeypatch.setattr(ChromeMixin, '_apply_titlebar_to', windows_like)
    ed = _editor(root)
    top = tk.Toplevel(ed)
    top.geometry('300x200')
    tk.Text(top, height=30, width=70).pack(fill='both', expand=True)
    btn = tk.Button(top, text='Close')
    btn.pack(side='bottom')
    stale[str(top)] = (300, 200)                 # what Windows still reports
    _settle(root)
    assert btn.winfo_viewable()
    assert top.winfo_width() >= top.winfo_reqwidth()
    assert top.winfo_height() >= top.winfo_reqheight()


def test_the_editor_scales_its_own_size(root, screen):
    screen(3000, 2000)
    skip_unless_the_screen_holds(root, 3000, 2000)
    root.tk.call('tk', 'scaling', BASE_SCALING * 1.5)
    ed = _editor(root)
    assert ed.wm_minsize() == (1770, 930)
    assert (ed.winfo_width(), ed.winfo_height()) == (2250, 1260)


def test_the_editors_size_is_kept_within_the_screen(root, screen):
    screen(1280, 720)
    ed = _editor(root)
    assert ed.winfo_width() <= 1280 - 16
    assert ed.winfo_height() <= 720 - 72


# ── DPI awareness on Windows ─────────────────────────────────────────────

def _windll(shcore=None, user32=None):
    return SimpleNamespace(
        shcore=SimpleNamespace(SetProcessDpiAwareness=shcore)
        if shcore else SimpleNamespace(),
        user32=SimpleNamespace(SetProcessDPIAware=user32)
        if user32 else SimpleNamespace())


def test_dpi_awareness_is_declared_system_aware():
    from openflo.gui import _declare_dpi_awareness
    calls = []
    dll = _windll(shcore=lambda v: calls.append(v) or 0)
    assert _declare_dpi_awareness(dll) == 'system'
    assert calls == [1]


def test_dpi_awareness_already_set_is_fine():
    from openflo.gui import _declare_dpi_awareness
    # E_ACCESSDENIED, as ctypes returns an HRESULT: a negative int.
    dll = _windll(shcore=lambda v: -2147024891,
                  user32=lambda: pytest.fail("no fallback needed"))
    assert _declare_dpi_awareness(dll) == 'already'


def test_dpi_awareness_falls_back_on_old_windows():
    from openflo.gui import _declare_dpi_awareness
    dll = _windll(user32=lambda: 1)          # no shcore before Windows 8.1
    assert _declare_dpi_awareness(dll) == 'legacy'


def test_dpi_awareness_reports_when_nothing_worked():
    from openflo.gui import _declare_dpi_awareness
    assert _declare_dpi_awareness(_windll()) is None
    assert _declare_dpi_awareness(_windll(shcore=lambda v: 1,
                                          user32=lambda: 0)) is None


def test_dpi_awareness_is_not_attempted_off_windows(monkeypatch):
    import ctypes

    from openflo import gui
    calls = []
    # A windll that would answer, so only the platform check can stop it.
    monkeypatch.setattr(ctypes, 'windll', _windll(
        shcore=lambda v: calls.append(v) or 0), raising=False)
    monkeypatch.setattr(gui.sys, 'platform', 'linux')
    assert gui._declare_dpi_awareness() is None
    assert calls == []
    # Negative: the same windll on Windows is called.
    monkeypatch.setattr(gui.sys, 'platform', 'win32')
    assert gui._declare_dpi_awareness() == 'system'
    assert calls == [1]


def test_main_declares_dpi_awareness_before_any_window(monkeypatch):
    """Declared after the first window exists, the declaration does nothing
    for that window, so the order is the requirement."""
    from openflo import gui
    order = []

    class Stop(Exception):
        pass

    def first_window():
        order.append('window')
        raise Stop
    monkeypatch.setattr(gui, '_declare_dpi_awareness',
                        lambda: order.append('dpi'))
    monkeypatch.setattr(gui, '_APP_BASE', first_window)
    with pytest.raises(Stop):
        gui.main()
    assert order == ['dpi', 'window']


# ── Tooltips ─────────────────────────────────────────────────────────────

def _tooltip(root):
    from openflo.gui import _ToolTip
    top = tk.Toplevel(root)
    top.geometry('200x100+50+50')
    btn = tk.Button(top, text='hover me')
    btn.pack(fill='both', expand=True)
    top.update()
    tip = _ToolTip(btn, 'help text', lambda: True)
    return top, btn, tip


def _point_at(widget):
    widget.event_generate('<Motion>', warp=True,
                          x=widget.winfo_width() // 2,
                          y=widget.winfo_height() // 2)
    widget.update()


def test_a_tooltip_shows_while_pointed_at(root):
    top, btn, tip = _tooltip(root)
    _point_at(btn)
    tip._show()
    assert tip._tip is not None and tip._tip.winfo_viewable()
    assert tip._tip.wm_overrideredirect()
    tip._hide()


def test_no_tooltip_when_the_pointer_has_left(root):
    top, btn, tip = _tooltip(root)
    _point_at(btn)
    root.event_generate('<Motion>', warp=True, x=0, y=0)   # away from it
    root.update()
    top.event_generate('<Motion>', warp=True, x=-40, y=-40)
    root.update()
    tip._show()
    assert tip._tip is None


def test_a_tooltip_goes_when_its_window_closes(root):
    top, btn, tip = _tooltip(root)
    _point_at(btn)
    tip._show()
    shown = tip._tip
    assert shown is not None
    top.destroy()                 # closed under the pointer: no <Leave>
    root.update()
    assert tip._tip is None
    assert not shown.winfo_exists()


def test_a_tooltip_goes_when_its_window_is_hidden(root):
    top, btn, tip = _tooltip(root)
    _point_at(btn)
    tip._show()
    shown = tip._tip
    top.withdraw()
    root.update()
    assert tip._tip is None and not shown.winfo_exists()


def test_a_tooltip_stays_on_screen(root, screen):
    screen(400, 300)
    from openflo.gui import _ToolTip
    top = tk.Toplevel(root)
    top.geometry('120x40+270+250')        # bottom-right corner
    btn = tk.Button(top, text='edge')
    btn.pack(fill='both', expand=True)
    top.update()
    tip = _ToolTip(btn, 'a fairly long tooltip text ' * 4, lambda: True)
    _point_at(btn)
    tip._show()
    t = tip._tip
    t.update()
    assert t.winfo_x() + t.winfo_reqwidth() <= 400
    assert t.winfo_y() + t.winfo_reqheight() <= 300
    tip._hide()


# ── Preferences: theme by name ───────────────────────────────────────────

def test_preferences_lists_themes_by_name_and_stores_the_key(root):
    # Window colors and plot background are two boxes (UI 38) whose names
    # make one stored theme key.
    from openflo.ui_preferences import PreferencesDialog
    ed = _editor(root)
    ed._theme_var.set('light')
    dlg = PreferencesDialog(ed)
    root.update()
    box, pbox = dlg.theme_combos
    assert set(box.cget('values')) == {'Light', 'Dark'}
    assert set(pbox.cget('values')) == {'White', 'Black'}
    assert (box.get(), pbox.get()) == ('Light', 'White')
    applied = []
    ed._set_theme = lambda: applied.append(ed._theme_var.get())
    box.set('Dark')
    box.event_generate('<<ComboboxSelected>>')
    root.update()
    assert applied == ['dark'] and ed._theme_var.get() == 'dark'
    pbox.set('Black')
    pbox.event_generate('<<ComboboxSelected>>')
    root.update()
    assert applied[-1] == 'midnight'
    ed._theme_var.set('light-dark')           # picked from the View menu
    assert (box.get(), pbox.get()) == ('Light', 'Black')
    dlg.destroy()
    ed._theme_var.set('light')                # its follower went with it


def _all(w):
    yield w
    for c in w.winfo_children():
        yield from _all(c)
