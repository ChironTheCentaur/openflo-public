"""Errors are never swallowed without a trace, and the ones nothing handled
reach the error window.

The silent-handler rule: an ``except`` that catches everything (bare,
``Exception`` or ``BaseException``) and then does nothing (``pass``,
``continue``, ``return <constant>``) hides every bug that ever lands in its
``try``. Each one in src/openflo must instead

* catch only what it expects (``tk.TclError``, ``OSError``, ...), or
* record what happened (``openflo.errors.note``) or handle it, or
* say on its ``except`` line why nothing can matter there:
  ``# noqa: BLE001 - <reason>``.
"""
from __future__ import annotations

import ast
import glob
import os
import re
import threading
import time

import pytest

from openflo import errors, tk_threads

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'src', 'openflo')

_REASON = re.compile(r'#\s*noqa:\s*BLE001\s*-+\s*\S.{3,}')


def _broad(handler):
    t = handler.type
    if t is None:
        return True
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(n, ast.Name) and n.id in ('Exception', 'BaseException')
               for n in names)


def _silent(handler):
    if len(handler.body) != 1:
        return False
    s = handler.body[0]
    return (isinstance(s, (ast.Pass, ast.Continue))
            or (isinstance(s, ast.Return)
                and (s.value is None or isinstance(s.value, ast.Constant)))
            or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)))


def silent_handlers(source):
    """(line number, line) of each silent catch-all without a reason."""
    lines = source.splitlines()
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ExceptHandler) and _broad(node) and _silent(node):
            line = lines[node.lineno - 1]
            if not _REASON.search(line):
                out.append((node.lineno, line.strip()))
    return out


def test_no_catch_all_swallows_an_error_without_saying_why():
    bad = []
    for path in sorted(glob.glob(os.path.join(SRC, '*.py'))):
        with open(path, encoding='utf-8') as f:
            for n, line in silent_handlers(f.read()):
                bad.append(f"{os.path.basename(path)}:{n}: {line}")
    assert not bad, (
        f"{len(bad)} catch-all handlers swallow errors silently; narrow the "
        "exception, call openflo.errors.note, or give a reason with "
        "'# noqa: BLE001 - why':\n" + "\n".join(bad))


@pytest.mark.parametrize('code, flagged', [
    ("try:\n    f()\nexcept Exception:\n    pass\n", True),
    ("try:\n    f()\nexcept:\n    pass\n", True),
    ("try:\n    f()\nexcept BaseException:\n    return None\n", True),
    ("for x in y:\n    try:\n        f()\n    except (ValueError, Exception):\n        continue\n", True),
    ("try:\n    f()\nexcept Exception:  # noqa: BLE001\n    pass\n", True),
    ("try:\n    f()\nexcept Exception:  # noqa: BLE001 - window already gone\n    pass\n", False),
    ("try:\n    f()\nexcept ValueError:\n    pass\n", False),
    ("try:\n    f()\nexcept Exception as e:\n    note('x', e)\n", False),
    ("try:\n    f()\nexcept Exception:\n    raise\n", False),
])
def test_the_rule_flags_exactly_the_silent_catch_alls(code, flagged):
    if 'return' in code:
        code = "def g():\n" + "".join("    " + ln + "\n" for ln in code.splitlines())
    assert bool(silent_handlers(code)) is flagged


def test_note_leaves_a_line_in_the_log(capsys):
    try:
        raise ValueError("bad value")
    except ValueError:
        errors.note("reading a preference")
    err = capsys.readouterr().err
    assert "[handled] reading a preference: ValueError: bad value" in err
    assert errors.noted()[-1].endswith("ValueError: bad value")


class _Root:
    def __init__(self):
        self.seen = []

    def report_callback_exception(self, *exc_info):
        self.seen.append(exc_info[1])

    def _root(self):
        return self

    def after(self, ms, fn):
        fn()


def test_report_hands_the_error_to_the_window():
    root = _Root()
    exc = RuntimeError("unhandled")
    errors.report(root, exc)
    assert root.seen == [exc]


def test_run_async_without_on_error_reports_instead_of_dropping():
    from openflo.async_task import run_async
    root = _Root()

    def boom():
        raise KeyError("lost")
    t = run_async(root, boom)
    t.join(5)
    assert len(root.seen) == 1 and isinstance(root.seen[0], KeyError)


def test_an_error_escaping_a_thread_reaches_the_window_on_the_main_thread():
    root = _Root()
    previous = threading.excepthook
    tk_threads.drain()
    try:
        errors.install_thread_hook(root)

        def boom():
            raise ZeroDivisionError("in a thread")
        t = threading.Thread(target=boom)
        t.start()
        t.join(5)
        assert root.seen == []              # not on the thread that failed
        end = time.monotonic() + 5
        while not root.seen and time.monotonic() < end:
            tk_threads.drain()
        assert len(root.seen) == 1
        assert isinstance(root.seen[0], ZeroDivisionError)
    finally:
        threading.excepthook = previous


# ── Numbers typed into dialogs ──────────────────────────────────────────────

def silent_defaults(source):
    """(line, text) of each ``try`` that reads a dialog field and, when the
    text is not a number, quietly uses a constant instead."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Try) and len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)):
            continue
        body = ast.unparse(node.body[0])
        if '.get()' not in body or not ('float(' in body or 'int(' in body):
            continue
        for h in node.handlers:
            if (len(h.body) == 1 and isinstance(h.body[0], ast.Assign)
                    and isinstance(h.body[0].value, ast.Constant)):
                out.append((node.lineno, body))
    return out


def test_no_dialog_swaps_a_mistyped_number_for_a_default():
    bad = []
    for path in sorted(glob.glob(os.path.join(SRC, '*.py'))):
        with open(path, encoding='utf-8') as f:
            for n, text in silent_defaults(f.read()):
                bad.append(f"{os.path.basename(path)}:{n}: {text}")
    assert not bad, ("these settings run with a value the user never chose "
                     "when the typed text is not a number; use "
                     "openflo.dialog_inputs.read_number:\n" + "\n".join(bad))


def test_the_default_rule_flags_a_silent_substitute():
    code = ("try:\n    k = int(self.k_var.get())\n"
            "except ValueError:\n    k = 15\n")
    assert silent_defaults(code)
    assert not silent_defaults(code.replace("k = 15", "return"))


@pytest.mark.parametrize('text, kw, expected', [
    ('8', dict(integer=True), 8),
    ('8.0', dict(integer=True), 8),
    ('8.5', dict(integer=True), None),
    (' 2.5 ', {}, 2.5),
    ('', {}, None),
    ('abc', {}, None),
    ('nan', {}, None),
    ('inf', {}, None),
    ('1', dict(lo=2), None),
    ('2', dict(lo=2, hi=5), 2),
    ('5', dict(lo=2, hi=5), 5),
    ('5.01', dict(lo=2, hi=5), None),
])
def test_parse_number(text, kw, expected):
    from openflo.dialog_inputs import parse_number
    assert parse_number(text, **kw) == expected


def test_read_number_says_which_setting_is_wrong(monkeypatch):
    from openflo import dialog_inputs
    shown = []
    monkeypatch.setattr(dialog_inputs.messagebox, 'showerror',
                        lambda title, msg, **k: shown.append(msg))

    class Var:
        def __init__(self, text):
            self.text = text

        def get(self):
            return self.text

    assert dialog_inputs.read_number(Var('7'), "Peaks", integer=True,
                                 lo=2, hi=12) == 7
    assert shown == []
    assert dialog_inputs.read_number(Var('x'), "Peaks", integer=True,
                                 lo=2, hi=12) is None
    assert shown == ['Peaks must be a whole number from 2 to 12 '
                     '(it says “x”).']


def test_read_number_reads_the_text_of_a_numeric_tk_variable(monkeypatch):
    """A DoubleVar holding text raises TclError on get(); the message still
    quotes what was typed."""
    import tkinter as tk

    from openflo import dialog_inputs
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot start: {exc}")
    try:
        shown = []
        monkeypatch.setattr(dialog_inputs.messagebox, 'showerror',
                            lambda title, msg, **k: shown.append(msg))
        var = tk.DoubleVar(root, value=1.5)
        assert dialog_inputs.read_number(var, "k", lo=0.5, hi=5) == 1.5
        root.globalsetvar(str(var), 'oops')
        assert dialog_inputs.read_number(var, "k", lo=0.5, hi=5) is None
        assert shown == ['k must be a number from 0.5 to 5 '
                         '(it says “oops”).']
        assert dialog_inputs.number_or(var, 9.0) == 9.0
    finally:
        root.destroy()


def test_the_editors_run_async_reports_an_error_nobody_handles():
    """ComputeMixin.run_async without on_error used to put the error in the
    status bar only: no traceback, no problem report."""
    from openflo.editor_compute import ComputeMixin

    class Ed(_Root):
        run_async = ComputeMixin.run_async

        def _end_busy(self):
            pass

    ed = Ed()

    def boom():
        raise OSError("disk gone")
    t = ed.run_async(boom)
    t.join(5)
    assert len(ed.seen) == 1 and isinstance(ed.seen[0], OSError)
