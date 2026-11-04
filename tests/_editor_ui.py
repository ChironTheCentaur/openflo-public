"""Shared pieces for the tests that read a shown editor's menus, windows
and labels: the menu entries and the windows they open, every text a window
shows, the Tk form of a listed shortcut, and a hidden Tk root.

Generated data, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path

import pytest

import tests.test_windows_nothing_hidden as sweep
from tests.conftest import gui_unavailable

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'


def _open_windows(root, ed):
    """Yield (menu label, window) for each window a menu item opens; the
    window is destroyed after the caller is done with it."""
    for menu, i, label in list(sweep._menu_items(ed)):
        if label.startswith(sweep.NOT_A_WINDOW):
            continue
        before = set(sweep._toplevels(root))
        try:
            menu.invoke(i)
            root.update()
        except Exception:                              # noqa: BLE001 - a menu item that raises is reported by test_windows_nothing_hidden; this helper yields only windows that opened
            continue
        for key, win in sweep._toplevels(root).items():
            if key in before or not win.winfo_exists():
                continue
            root.update()
            yield label, win
            if win.winfo_exists():
                win.destroy()
            root.update()


def _texts(win, out, where):
    """Every label, button, check box, radio button, menu entry, dropdown
    choice and column heading of ``win`` (not of windows opened from it)."""
    for w in sweep._walk(win):
        if w is not win and isinstance(w, tk.Toplevel):
            continue
        cls = w.winfo_class()
        if cls == 'Menu':
            end = w.index('end')
            for i in range(0 if end is None else end + 1):
                if w.type(i) in ('command', 'cascade', 'checkbutton',
                                 'radiobutton'):
                    out.append((where, 'menu', w.entrycget(i, 'label')))
            continue
        try:
            t = w.cget('text')
        except tk.TclError:
            t = ''
        if isinstance(t, str) and t:
            out.append((where, cls, t))
        if cls == 'TCombobox':
            out += [(where, 'choice', str(v)) for v in w.cget('values') or ()]
        if cls == 'Treeview':
            out += [(where, 'heading', w.heading(c, 'text'))
                    for c in w['columns']]


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    r.withdraw()
    yield r
    r.destroy()


def press_key(widget, sequence):
    """Send the key ``sequence`` to ``widget`` with the keyboard focus on it,
    and return the path of the widget that had the focus as it was sent.

    Tk hands a key to the widget with the focus, and drops it while the
    application has none. On the macOS runners the application is often not
    the active one, and the window system can take the focus away from Tk at
    any turn of the event loop, so a key sent after a ``focus_force`` and an
    ``update`` may go nowhere (a Ctrl+C arrived, the Delete after it did
    not). Setting the focus and sending the key in one Tcl command leaves
    the event loop no turn in between, on every system."""
    w = str(widget)
    return widget.tk.eval(
        f'focus -force {w}; set ::openflo_test_focus [focus]; '
        f'event generate {w} {sequence}; set ::openflo_test_focus')


_KEYSYMS = {'Ctrl': 'Control', ',': 'comma', '`': 'grave', '+': 'plus',
            '-': 'minus', 'Esc': 'Escape'}


def _sequence(keys):
    """'Ctrl+Shift+S' -> '<Control-Shift-Key-S>', the form Tk reports."""
    parts = keys.split('+') if keys != 'Ctrl++' else ['Ctrl', '+']
    mods, key = parts[:-1], _KEYSYMS.get(parts[-1], parts[-1])
    if len(key) == 1 and key.isalpha() and 'Shift' not in mods:
        key = key.lower()
    return '<' + '-'.join([_KEYSYMS.get(m, m) for m in mods]
                          + [f'Key-{key}']) + '>'


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
