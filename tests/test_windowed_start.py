"""The GUI starts with no console, as the Windows installer's shortcuts run it.

The Start-menu and Desktop shortcuts run ``pythonw.exe -m openflo.gui``, and a
pip install's ``openflo-gui.exe`` is a windowed launcher too. There is no
console then, so ``sys.stdout`` and ``sys.stderr`` are None. An unconditional
``faulthandler.enable()`` at import raised ``RuntimeError: sys.stderr is
None`` before ``main()`` could point the streams at its log file: no window
and no error. These tests run the module as ``__main__`` with both streams
None, the way ``-m`` does under pythonw, and check that a crash in C code
still leaves a traceback in the log.
"""
from __future__ import annotations

import faulthandler
import io
import os
import subprocess
import sys
import textwrap

from tests.conftest import gui_unavailable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The child sets both streams to None, stubs mainloop so main() returns, runs
# openflo.gui as __main__, writes what happened to argv[1] (it has no stderr),
# then crashes in C code: faulthandler must write that to the GUI's log file.
CHILD = textwrap.dedent('''
    import faulthandler, runpy, sys, traceback, tkinter
    out = open(sys.argv[1], 'w', encoding='utf-8')
    sys.path.insert(0, {src!r})
    tkinter.Tk.mainloop = lambda self, n=0: None
    sys.stdout = sys.stderr = None
    try:
        runpy.run_module('openflo.gui', run_name='__main__', alter_sys=True)
        out.write('returned\\n')
    except SystemExit as exc:
        out.write('exit %r\\n' % (exc.code,))
    except BaseException:
        out.write(traceback.format_exc())
    out.write('faulthandler %s\\n' % faulthandler.is_enabled())
    out.close()
    if sys.argv[2] == 'crash':
        faulthandler._sigsegv()
''')


def _tk_or_skip():
    r = subprocess.run([sys.executable, '-c', 'import tkinter; tkinter.Tk().destroy()'],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        gui_unavailable(f"Tk could not start: {r.stderr.strip()[-200:]}")


def _run_windowed(tmp_path, mode):
    result = tmp_path / 'result.txt'
    env = dict(os.environ, TMPDIR=str(tmp_path), TEMP=str(tmp_path), TMP=str(tmp_path))
    child = CHILD.format(src=os.path.join(ROOT, 'src'))
    proc = subprocess.run([sys.executable, '-c', child, str(result), mode],
                          env=env, cwd=str(tmp_path), capture_output=True,
                          text=True, timeout=600)
    return proc, result.read_text(encoding='utf-8')


def test_gui_module_starts_with_no_console(tmp_path):
    _tk_or_skip()
    proc, got = _run_windowed(tmp_path, 'start')
    assert got.splitlines()[0] == 'exit 0', got
    assert proc.returncode == 0, proc.stderr[-2000:]
    # main() re-enabled the crash handler on its log file.
    assert 'faulthandler True' in got, got


def test_crash_with_no_console_is_written_to_the_log(tmp_path):
    _tk_or_skip()
    proc, got = _run_windowed(tmp_path, 'crash')
    assert got.splitlines()[0] == 'exit 0', got
    assert proc.returncode != 0, 'the planted crash did not happen'
    log = (tmp_path / 'openflo-gui.log').read_text(encoding='utf-8', errors='replace')
    assert 'Fatal Python error' in log, log[-2000:]


def test_faulthandler_skips_a_missing_stream(monkeypatch):
    from openflo import gui
    was = faulthandler.is_enabled()
    monkeypatch.setattr(sys, 'stderr', None)
    try:
        assert gui._enable_faulthandler() is False
        assert gui._enable_faulthandler(io.StringIO()) is False   # no OS descriptor
    finally:
        if was:
            faulthandler.enable(file=sys.__stderr__)


def test_faulthandler_uses_the_given_file(tmp_path, monkeypatch):
    from openflo import gui
    was = faulthandler.is_enabled()
    monkeypatch.setattr(sys, 'stderr', None)     # the given file, not stderr
    with open(tmp_path / 'crash.log', 'w', encoding='utf-8') as f:
        try:
            assert gui._enable_faulthandler(f) is True
            assert faulthandler.is_enabled()
        finally:
            faulthandler.disable()
            if was:
                faulthandler.enable(file=sys.__stderr__)
