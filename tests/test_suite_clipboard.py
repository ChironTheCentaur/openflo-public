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

import ast
import re
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


_TCL_CLEAR = re.compile(r'\bclipboard\s+clear\b')


def _bare_clipboard_clears(sources):
    """Every place in ``sources`` ({name: text}) that clears the clipboard
    other than inside ``clipboard_put`` itself: a ``clipboard_clear``
    attribute (a call, an alias, ``Misc.clipboard_clear``), a string naming
    it or Tcl's ``clipboard clear`` (``getattr``, ``tk.eval``), or a call
    whose first two string arguments are ``'clipboard', 'clear'``
    (``tk.call``). A name built up at run time is out of its reach."""
    found = []
    for name, text in sorted(sources.items()):
        tree = ast.parse(text)
        helper = set()
        for f in ast.walk(tree):
            if isinstance(f, ast.FunctionDef) and f.name == 'clipboard_put':
                helper |= {id(n) for n in ast.walk(f)}
        for n in ast.walk(tree):
            if id(n) in helper:
                continue
            if isinstance(n, ast.Attribute):
                hit = n.attr == 'clipboard_clear'
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                hit = ('clipboard_clear' in n.value
                       or bool(_TCL_CLEAR.search(n.value)))
            elif isinstance(n, ast.Call):
                hit = [a.value for a in n.args[:2]
                       if isinstance(a, ast.Constant)] == ['clipboard', 'clear']
            else:
                hit = False
            if hit:
                found.append(f'{name}:{n.lineno}')
    return found


def test_every_app_text_copy_goes_through_clipboard_put():
    """conftest swaps Tk's clipboard for a per-process one, so no test in
    this process reaches real Tk: this scan is what keeps a bare clear (the
    Tk 9.0.4 crash) out of the app."""
    from pathlib import Path

    import openflo
    src = Path(openflo.__file__).parent
    sources = {str(p.relative_to(src)): p.read_text(encoding='utf-8')
               for p in src.rglob('*.py')}
    bare = _bare_clipboard_clears(sources)
    assert bare == [], 'copy through clipboard_export.clipboard_put'


@pytest.mark.parametrize('line', [
    "w.clipboard_clear()",
    "self.tk.call('clipboard', 'clear')",
    'self.tk.call("clipboard", "clear")',
    "self.tk.call('clipboard','clear', '-displayof', w)",
    "self.tk.eval('clipboard clear')",
    "clr = self.clipboard_clear\nclr()",
    "getattr(w, 'clipboard_clear')()",
    "tk.Misc.clipboard_clear(w)",
])
def test_the_scan_finds_each_way_to_clear(line):
    assert _bare_clipboard_clears({'m.py': line}) != []


def test_the_scan_exempts_only_clipboard_puts_own_body():
    helper = ("def clipboard_put(w, text):\n"
              "    w.winfo_id()\n"
              "    w.clipboard_clear()\n"
              "    w.clipboard_append(text)\n")
    assert _bare_clipboard_clears({'clipboard_export.py': helper}) == []
    other = helper + "\ndef other(w):\n    w.clipboard_clear()\n"
    assert _bare_clipboard_clears({'clipboard_export.py': other}) == [
        'clipboard_export.py:7']
    assert _bare_clipboard_clears(
        {'m.py': "w.clipboard_append('x')\ntk.call('clipboard', 'get')"}) == []
