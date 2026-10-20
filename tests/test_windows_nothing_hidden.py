"""Every window opened from the menus shows all of its controls at the size it
opens at, on a small laptop screen and on a 150 % scaled display.

The audit found nine windows hiding buttons at their default size: fixed
pixel sizes (``geometry('1000x560')``) chosen on one machine, with the button
row packed after a table or text box, so the packer, short of room, simply
did not show the buttons. Windows display scaling makes text bigger and the
fixed sizes relatively smaller, so it gets worse there.

The check: walk the editor's menus, open each window, and for every button,
check box, entry and dropdown in it ask Tk whether it is shown
(``winfo_viewable``) and whether it got the room it asked for. A control the
code deliberately took out of the layout (``pack_forget``), one on a
notebook tab not showing, or one inside a scrolling canvas is not counted.

Each check is proved to fail on a window built the old way
(``test_the_check_finds_*``), and the sweep itself failed on master before
these fixes (hidden Close/Update/Detect peaks/... in Statistics, Methods,
Absolute counts, Preferences, FCS inspector, Expression, Frequencies,
Calibration, Layout editor and Voltage).

Screens are virtual: ``winfo_screenwidth/height`` are pinned, so the result
does not depend on the CI runner's display, and the display scaling is
simulated with Tk's own ``tk scaling``, which is what a DPI-aware process on
a scaled Windows display gets.
"""
from __future__ import annotations

import os
import time
import tkinter as tk
import tkinter.font as tkfont
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable, skip_unless_the_screen_holds

os.environ.setdefault('MPLBACKEND', 'Agg')

# Synthetic data only: generic channel names, made-up sample names.
COLS = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']
FLUOR = COLS[3:]

CONTROLS = {'TButton', 'Button', 'TCheckbutton', 'Checkbutton',
            'TRadiobutton', 'Radiobutton', 'TCombobox', 'TEntry', 'Entry',
            'TSpinbox', 'Spinbox', 'TMenubutton', 'Menubutton'}

# Menu items that do not open a window of their own, or open a system file
# dialog, or act on the session (load, save, quit), or toggle a setting.
NOT_A_WINDOW = (
    'Close', 'Open session', 'Add FCS', 'Load CSV', 'Save', 'Export',
    'Upgrade', 'Load example', 'Batch-norm', 'Documentation',
    'Online documentation',
    'Report a problem', 'Check for updates', 'Watch folder', 'Undo', 'Redo',
    'Clear', 'Dock', 'Reset', 'Analysis report', 'Run diagnostics',
    'Add singlet', 'Auto-clean', 'About', 'Keyboard shortcuts',
    'Pipeline workspace')

# (screen width, height, display scaling)
SCREENS = {
    'laptop-1366x768-100%': (1366, 768, 1.0),
    'fullhd-1920x1080-150%': (1920, 1080, 1.5),
}


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _inside_canvas(w, top):
    p = w.master
    while p is not None and p is not top:
        if p.winfo_class() == 'Canvas':
            return True
        p = p.master
    return False


def problems(top):
    """Controls of the open window ``top`` that are hidden, cut short, or
    reach past its edge; one line each."""
    top.update_idletasks()
    top.update()
    tw, th = top.winfo_width(), top.winfo_height()
    tx, ty = top.winfo_rootx(), top.winfo_rooty()
    font = tkfont.nametofont('TkTextFont')
    out = []
    for w in _walk(top):
        if w is top or w.winfo_toplevel() is not top:
            continue
        cls = w.winfo_class()
        if cls not in CONTROLS:
            continue
        try:
            text = w.cget('text')
        except tk.TclError:
            text = ''
        name = f"{cls} {str(text)!r}"
        chain, c = [], w
        while c is not top:
            chain.append(c)
            c = c.master
        managers = [c.winfo_manager() for c in chain]
        if any(m in ('', 'canvas') for m in managers):
            continue                     # taken out on purpose, or scrolled
        if any(m == 'notebook' and not c.winfo_ismapped()
               for m, c in zip(managers, chain, strict=True)):
            continue                     # on a tab that is not showing
        if not w.winfo_viewable():
            out.append(f"hidden: {name}")
            continue
        ww, wh = w.winfo_width(), w.winfo_height()
        if cls == 'TCombobox':
            values = w.cget('values') or ()
            if values:
                need = max(font.measure(str(v)) for v in values) + 24
                if ww < need:
                    out.append(f"too narrow for its choices: {name} "
                               f"({ww} < {need} px)")
        if _inside_canvas(w, top):
            continue
        x, y = w.winfo_rootx() - tx, w.winfo_rooty() - ty
        if x < -1 or y < -1 or x + ww > tw + 1 or y + wh > th + 1:
            out.append(f"past the window's edge: {name}")
        elif ww < w.winfo_reqwidth() - 2 or wh < w.winfo_reqheight() - 2:
            out.append(f"cut short: {name} ({ww}x{wh} < "
                       f"{w.winfo_reqwidth()}x{w.winfo_reqheight()})")
    return out


def _settle(root, ms=250, still_ms=150, cap_ms=3000):
    """Run the event loop before measuring a window that just opened, until
    every open window has kept the same size and place for ``still_ms``
    (at least ``ms``, at most ``cap_ms``). The editor fits each new window
    on an idle callback and, on Windows, nudges its title bar 30 ms later;
    Windows also applies a move or resize through its own messages, so the
    size and place read right after one ``update()`` can still be the ones
    from before the fit. Measured that early, a different window came out
    "too small" on each run at 150 % (Absolute counts one time, Calibration
    or Compare embeddings another), and under a 4-worker run Plate view
    still sat where Windows first put it, below the bottom of the screen."""
    def shape():
        return tuple(w.winfo_geometry() for w in _toplevels(root).values()
                     if w.winfo_exists())
    start = time.monotonic()
    last, still_since = shape(), start
    while True:
        root.update()
        time.sleep(0.005)
        now = time.monotonic()
        cur = shape()
        if cur != last:
            last, still_since = cur, now
        if now - start >= cap_ms / 1000:
            return
        if now - start >= ms / 1000 and now - still_since >= still_ms / 1000:
            return


@pytest.fixture
def virtual_screen(monkeypatch):
    """Pin the screen size every window is fitted to."""
    def use(width, height):
        monkeypatch.setattr(tk.Misc, 'winfo_screenwidth', lambda self: width)
        monkeypatch.setattr(tk.Misc, 'winfo_screenheight',
                            lambda self: height)
    return use


def _root(scale=1.0):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    root.withdraw()
    root.openflo_base_scaling = float(root.tk.call('tk', 'scaling'))
    root.tk.call('tk', 'scaling', root.openflo_base_scaling * scale)
    from openflo.gui import apply_theme
    apply_theme(root, 'light')
    return root


def _editor(root):
    """A shown editor holding three generated samples, a gate and a child."""
    import importlib
    gui = importlib.import_module('openflo.gui')
    for n in ('askyesno', 'showinfo', 'showwarning', 'showerror',
              'askokcancel'):
        setattr(gui.messagebox, n, lambda *a, **k: True)
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    rng = np.random.default_rng(7)
    for i, nm in enumerate(('alpha_1', 'alpha_2', 'beta_1')):
        df = pd.DataFrame({c: rng.random(400) * 50000 for c in COLS})
        df['cluster'] = rng.integers(-1, 4, 400)
        ed._samples[nm] = SimpleNamespace(
            name=nm, path=f'{nm}.fcs', data=df, fluor_channels=list(FLUOR),
            raw=df.copy(), channel_labels={c: c for c in COLS})
        ed._sample_order.append(nm)
        ed._sample_colors[nm] = ['#1f77b4', '#d62728', '#2ca02c'][i]
        ed._sample_trial[nm] = 'Day1'
        ed._sample_plot_enabled[nm] = True
        ed._sample_gates.setdefault(nm, {})
        ed._sample_gate_order.setdefault(nm, [])
        ed._sample_gate_seq.setdefault(nm, 0)
    ed._trial_order.append('Day1')
    ed._channels = list(COLS) + ['cluster']
    ed._channel_labels = {c: c for c in ed._channels}
    ed._populate_channel_combos()
    ed.x_combo.set('FSC-A')
    ed.y_combo.set('SSC-A')
    ed._active_sample = 'alpha_1'
    ed._gates = ed._sample_gates['alpha_1']
    ed._gate_id_order, ed._gate_id_seq = [], 0
    parent = ed._add_gate({'kind': 'threshold', 'channel': 'FL4-A', 'op': '>',
                           'value': 25000.0, 'name': 'FL4 pos'}, audit=False)
    ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'op': '>',
                  'value': 20000.0, 'name': 'FL2 pos'}, parent_id=parent,
                 audit=False)
    ed.update()
    return ed


def _menu_items(ed):
    for w in _walk(ed):
        if w.winfo_class() != 'Menu':
            continue
        end = w.index('end')
        for i in range(0 if end is None else end + 1):
            if w.type(i) == 'command':
                yield w, i, w.entrycget(i, 'label')


def _toplevels(root):
    return {str(t): t for t in _walk(root) if isinstance(t, tk.Toplevel)}


@pytest.fixture
def no_dialogs(monkeypatch):
    from tkinter import filedialog, messagebox, simpledialog
    for mod in (filedialog, messagebox, simpledialog):
        for n in dir(mod):
            if n.startswith(('ask', 'show')):
                monkeypatch.setattr(mod, n, lambda *a, **k: None)


@pytest.fixture
def engines_known(monkeypatch):
    """Every engine's import already answered, so Help → Environment does
    not start the (slow) import check in the background."""
    from openflo import capabilities as cap
    monkeypatch.setattr(cap, '_IMPORT_RESULT',
                        {e[0]: None for e in cap._ENGINES}, raising=False)


@pytest.mark.parametrize('screen', list(SCREENS))
def test_every_window_shows_all_its_controls(screen, virtual_screen,
                                             no_dialogs, engines_known):
    sw, sh, scale = SCREENS[screen]
    virtual_screen(sw, sh)
    root = _root(scale)
    try:
        skip_unless_the_screen_holds(root, sw, sh)
        ed = _editor(root)
        opened, failures = [], []
        for menu, i, label in list(_menu_items(ed)):
            if label.startswith(NOT_A_WINDOW):
                continue
            before = set(_toplevels(root))
            try:
                menu.invoke(i)
                _settle(root)
            except Exception as exc:                  # noqa: BLE001
                failures.append(f"{label}: raised {type(exc).__name__}: "
                                f"{exc}")
                continue
            for key, win in _toplevels(root).items():
                if key in before or not win.winfo_exists():
                    continue
                opened.append(label)
                found = problems(win)
                x, y = win.winfo_x(), win.winfo_y()
                w, h = win.winfo_width(), win.winfo_height()
                if x < 0 or y < 0 or x + w > sw or y + h > sh:
                    found.append(f"window off the screen: {w}x{h}+{x}+{y}")
                failures += [f"{label} [{win.title()}] {p}" for p in found]
                win.destroy()
                root.update()
        # Guard the guard: the walk must actually have opened the windows.
        assert len(opened) >= 25, opened
        for must in ('Statistics…', 'Methods & provenance…', 'Preferences…',
                     'Absolute counts…', 'FCS inspector…', 'Expression…',
                     'Frequencies…', 'Calibration…', 'Transforms…'):
            assert must in opened, f"{must} was not opened: {opened}"
        assert not failures, "\n".join(failures)
    finally:
        # Tk keeps the scaling per display, not per interpreter.
        root.tk.call('tk', 'scaling', root.openflo_base_scaling)
        root.destroy()


# ── The checks themselves, proved on windows built the old way ─────────────

def _old_style_window(root, button_first=False):
    """A fixed-size window with a tall text box and a button row: the
    pattern the audit found. Packed text-first, the buttons are hidden."""
    top = tk.Toplevel(root)
    top.geometry('300x200')
    bar = tk.Frame(top)
    text = tk.Text(top, height=40)
    if button_first:
        bar.pack(side='bottom', fill='x')
        text.pack(fill='both', expand=True)
    else:
        text.pack(fill='both', expand=True)
        bar.pack(side='bottom', fill='x')
    tk.Button(bar, text='Apply').pack(side='left')
    tk.Button(bar, text='Close').pack(side='right')
    return top


def test_the_check_finds_a_hidden_button_row():
    root = _root()
    try:
        bad = _old_style_window(root)
        found = problems(bad)
        assert "hidden: Button 'Apply'" in found
        assert "hidden: Button 'Close'" in found
        # The same window with the button row packed first: nothing hidden.
        assert problems(_old_style_window(root, button_first=True)) == []
    finally:
        root.destroy()


def test_the_check_finds_a_cut_short_button():
    root = _root()
    try:
        top = tk.Toplevel(root)
        row = tk.Frame(top)
        row.pack(fill='x')
        # Wide enough that the window is above Windows' smallest width.
        first = tk.Button(row, text='A', width=30)
        first.pack(side='left')
        update = tk.Button(row, text='Update')
        update.pack(side='left')
        top.update_idletasks()
        # Room for the first and half of Update: shown, but cut short.
        top.geometry(f"{first.winfo_reqwidth() + update.winfo_reqwidth() // 2}"
                     f"x{row.winfo_reqheight()}")
        found = problems(top)
        assert any(p.startswith("cut short: Button 'Update'") for p in found)
        top.geometry(f"600x{row.winfo_reqheight()}")
        assert problems(top) == []
    finally:
        root.destroy()


def test_the_check_finds_a_dropdown_narrower_than_its_choices():
    from tkinter import ttk
    root = _root()
    try:
        top = tk.Toplevel(root)
        cb = ttk.Combobox(top, width=3, values=['Automatic (recompute)'])
        cb.pack()
        assert any(p.startswith('too narrow') for p in problems(top))
        cb.configure(width=30)
        assert problems(top) == []
    finally:
        root.destroy()


def test_the_check_skips_what_was_taken_out_on_purpose():
    root = _root()
    try:
        top = tk.Toplevel(root)
        b = tk.Button(top, text='Advanced')
        b.pack()
        b.pack_forget()
        assert problems(top) == []
    finally:
        root.destroy()
