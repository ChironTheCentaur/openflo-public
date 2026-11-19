"""The suite's Tk clipboard belongs to the test process (tests/conftest.py).

Parallel workers share one display, so a clipboard on it is shared too: one
worker's copied samples landed between another worker's gate Copy and Paste,
and that Paste took the samples ("KeyError 'g2'" in test_boolean_gate_tools).
Here another process writes the real clipboard while this one holds text;
this process must still read its own text back.

The other process writes through the app's own clipboard_put, as OpenFlo
does: a bare ``clipboard clear`` from a window not made yet (here the
withdrawn root, never shown) kills the process on Windows with Tk 9.0.4
(Python 3.14), Tk ticket bd352828bb.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.conftest import gui_unavailable

# Another process takes the real clipboard and keeps serving it until its
# stdin closes (on X11 the owner must stay alive for others to read it).
_OTHER_PROCESS = textwrap.dedent("""
    import sys, tkinter as tk
    from openflo.clipboard_export import clipboard_put
    r = tk.Tk(); r.withdraw()
    clipboard_put(r, 'another process')
    r.update()
    print('ready', flush=True)
    sys.stdin.read()
    r.destroy()
""")


def _root():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available — headless environment')
    try:
        root = tk.Tk()
    except Exception as e:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f'Tk cannot initialise without a display: {e}')
    root.withdraw()
    return root


def test_another_process_writing_the_clipboard_does_not_reach_this_one():
    root = _root()
    other = None
    try:
        root.clipboard_clear()
        root.clipboard_append('this process')
        root.update()
        other = subprocess.Popen(
            [sys.executable, '-c', _OTHER_PROCESS], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True)
        assert other.stdout.readline().strip() == 'ready'
        for _ in range(5):                 # let any ownership change land
            root.update()
        assert root.clipboard_get() == 'this process'
    finally:
        if other is not None:
            other.stdin.close()
            other.wait(timeout=30)
        root.destroy()


def test_the_clipboard_keeps_tk_semantics():
    import tkinter as tk
    root = _root()
    try:
        with pytest.raises(tk.TclError):   # each test starts empty
            root.clipboard_get()
        root.clipboard_clear()
        with pytest.raises(tk.TclError):   # cleared, nothing appended
            root.clipboard_get()
        root.clipboard_append('a')
        root.clipboard_append('b')
        assert root.clipboard_get() == 'ab'
        other = tk.Toplevel(root)          # windows of a process share it
        assert other.clipboard_get() == 'ab'
        root.clipboard_clear()
        root.clipboard_append('c')
        assert other.clipboard_get() == 'c'
    finally:
        root.destroy()


# Windows that are not made yet copy, as the app's hidden root and a window
# not yet shown are: Tk 9.0.4 on Windows crashed in clipboard clear for them.
_NOT_MADE_YET = textwrap.dedent("""
    import faulthandler, tkinter as tk
    faulthandler.enable()
    from openflo.clipboard_export import clipboard_put
    r = tk.Tk(); r.withdraw()
    clipboard_put(r, 'hidden root')
    print(r.clipboard_get(), flush=True)
    t = tk.Toplevel(r)
    clipboard_put(t, 'new window')
    print(t.clipboard_get(), flush=True)
    r.destroy()
""")


def test_a_window_not_made_yet_can_copy():
    _root().destroy()                      # skip where Tk cannot start
    out = subprocess.run([sys.executable, '-c', _NOT_MADE_YET],
                         capture_output=True, text=True, timeout=120)
    assert (out.returncode, out.stdout.splitlines()) == (
        0, ['hidden root', 'new window']), (
        f'exit {out.returncode & 0xffffffff:#x}: {out.stdout}{out.stderr}')


class _Widget:
    def __init__(self):
        self.calls = []

    def winfo_id(self):
        self.calls.append('winfo_id')
        return 1

    def clipboard_clear(self):
        self.calls.append('clear')

    def clipboard_append(self, text):
        self.calls.append(('append', text))


def test_clipboard_put_makes_the_window_before_clearing():
    from openflo.clipboard_export import clipboard_put, copy_table
    w = _Widget()
    clipboard_put(w, 'x')
    assert w.calls == ['winfo_id', 'clear', ('append', 'x')]
    w = _Widget()
    copy_table(w, [['a', 'b']])
    assert w.calls == ['winfo_id', 'clear', ('append', 'a\tb\n')]


def test_every_app_text_copy_goes_through_clipboard_put():
    from pathlib import Path

    import openflo
    src = Path(openflo.__file__).parent
    bare = [f'{p.name}:{n}' for p in sorted(src.rglob('*.py'))
            if p.name != 'clipboard_export.py'
            for n, line in enumerate(p.read_text(encoding='utf-8')
                                     .splitlines(), 1)
            if 'clipboard_clear(' in line or "'clipboard', 'clear'" in line]
    assert bare == [], 'copy through clipboard_export.clipboard_put'
