"""The suite's Tk clipboard belongs to the test process (tests/conftest.py).

Parallel workers share one display, so a clipboard on it is shared too: one
worker's copied samples landed between another worker's gate Copy and Paste,
and that Paste took the samples ("KeyError 'g2'" in test_boolean_gate_tools).
Here another process writes the real clipboard while this one holds text;
this process must still read its own text back.
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
    r = tk.Tk(); r.withdraw()
    r.clipboard_clear(); r.clipboard_append('another process')
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
