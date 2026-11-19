"""TEMPORARY probe: which Tk clipboard calls crash a child on Windows.

Runs each variant in its own interpreter with faulthandler and fails with a
table of outcomes, so the table shows in the CI log. Removed before merge.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'win32',
                                reason='probe for the Windows clipboard')

_HEAD = textwrap.dedent("""
    import faulthandler, sys
    faulthandler.enable()
    import tkinter as tk
    def say(*a):
        print(*a, flush=True)
""")

_VARIANTS = {
    'A withdrawn, clear first (test child)': """
        r = tk.Tk(); r.withdraw()
        r.clipboard_clear(); r.clipboard_append('x'); r.update()
        say('ok')
    """,
    'B shown, update, then clear': """
        r = tk.Tk(); r.update()
        r.clipboard_clear(); r.clipboard_append('x'); r.update()
        say('ok')
    """,
    'C withdrawn, update, then clear': """
        r = tk.Tk(); r.withdraw(); r.update()
        r.clipboard_clear(); r.clipboard_append('x'); r.update()
        say('ok')
    """,
    'D withdrawn, winfo_id, then clear': """
        r = tk.Tk(); r.withdraw(); r.winfo_id()
        r.clipboard_clear(); r.clipboard_append('x'); r.update()
        say('ok')
    """,
    'E withdrawn, append without clear': """
        r = tk.Tk(); r.withdraw()
        r.clipboard_append('x'); r.update()
        say('ok', repr(r.clipboard_get()))
    """,
    'F withdrawn root, shown toplevel (app shape)': """
        r = tk.Tk(); r.withdraw()
        t = tk.Toplevel(r); t.update()
        t.clipboard_clear(); t.clipboard_append('x'); t.update()
        say('ok', repr(t.clipboard_get()))
    """,
    'G withdrawn root, new toplevel, no update': """
        r = tk.Tk(); r.withdraw()
        t = tk.Toplevel(r)
        t.clipboard_clear(); t.clipboard_append('x'); t.update()
        say('ok')
    """,
    'H shown root, clear before any update': """
        r = tk.Tk()
        r.clipboard_clear(); r.clipboard_append('x'); r.update()
        say('ok')
    """,
    'I shown root, unpacked button': """
        r = tk.Tk(); r.update()
        b = tk.Button(r, text='b')
        b.clipboard_clear(); b.clipboard_append('x'); r.update()
        say('ok')
    """,
    'J no ttk, bare _tkinter, clipboard clear': """
        import _tkinter
        t = _tkinter.create(None, 'probe', 'Tk', False, True, True, False,
                            None)
        say('tk', t.call('info', 'patchlevel'))
        t.call('wm', 'withdraw', '.')
        t.call('clipboard', 'clear')
        t.call('clipboard', 'append', 'x')
        say('ok')
    """,
    'K ctypes OpenClipboard in this session': """
        import ctypes
        u = ctypes.windll.user32
        k = ctypes.windll.kernel32
        sid = ctypes.c_ulong()
        k.ProcessIdToSessionId(k.GetCurrentProcessId(), ctypes.byref(sid))
        r = tk.Tk(); r.withdraw()
        say('tk', r.tk.call('info', 'patchlevel'), 'session', sid.value,
            'owner', u.GetClipboardOwner())
        opened = u.OpenClipboard(None)
        err = k.GetLastError()
        emptied = u.EmptyClipboard() if opened else None
        closed = u.CloseClipboard() if opened else None
        say('ok open', opened, 'err', err, 'empty', emptied, 'close', closed)
    """,
}


def _run(body):
    code = _HEAD + textwrap.dedent(body)
    try:
        p = subprocess.run([sys.executable, '-c', code], capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 'timeout', ''
    out = (p.stdout + p.stderr).strip().splitlines()
    tail = ' | '.join(line for line in out[:4] if line.strip())
    return f'{p.returncode & 0xffffffff:#x}', tail


def test_probe_table():
    rows = [(name, *_run(body)) for name, body in _VARIANTS.items()]
    table = '\n'.join(f'PROBE {name:<48} exit={rc:<12} {tail}'
                      for name, rc, tail in rows)
    print('\n' + table)
    pytest.fail('clipboard probe table:\n' + table)
