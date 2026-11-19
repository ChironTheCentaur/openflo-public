"""Tests wait on the Tk event loop through tests/_tk_wait.py.

Seventeen ``while ...: root.update(); time.sleep(...)`` loops and six
counted ``for _ in range(n): root.update()`` loops were written out by hand,
each with its own clock, pause, and timeout answer (return False, assert,
raise, carry on, or never time out), and nineteen files imported ``settle``
and ``_pump`` from test_transform_records. They now call ``pump_until`` and
``settle`` from one module, and these tests stop new copies coming back.

Two files keep their own loop on purpose: test_dialog_actions.py waits for
the threads a button started, and test_windows_nothing_hidden.py waits until
every window has kept its size; neither is a plain wait.
"""
from __future__ import annotations

import ast
import pathlib
import time

from tests._tk_wait import pump_until, settle, wait_until

TESTS = pathlib.Path(__file__).resolve().parent

# Files that may run the Tk event loop in a while loop of their own. This
# list may only shrink.
_MAY_PUMP_ON_THEIR_OWN = {
    '_tk_wait.py',                     # the shared waits themselves
    'test_dialog_actions.py',          # waits for a button's threads
    'test_windows_nothing_hidden.py',  # waits until windows stop moving
}

_SHARED_WAITS = {'pump_until', 'settle', 'wait_until', '_pump'}


def _own_pump_loops(tree):
    """Line numbers of ``while`` loops whose body calls ``<x>.update()`` with
    no arguments: a hand-written run of the Tk event loop."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        for stmt in node.body:
            if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == 'update' and not c.args
                   and not c.keywords for c in ast.walk(stmt)):
                found.append(node.lineno)
                break
    return found


def _waits_imported_from_a_test_module(tree):
    """``(line, name)`` for each shared wait imported from anywhere but
    tests._tk_wait (another test file's copy)."""
    return [(node.lineno, a.name) for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module not in ('tests._tk_wait', '_tk_wait')
            for a in node.names if a.name in _SHARED_WAITS]


def _test_files():
    return sorted(TESTS.glob('*.py'))


def test_no_test_file_runs_its_own_event_loop_wait():
    bad = []
    for path in _test_files():
        if path.name in _MAY_PUMP_ON_THEIR_OWN:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        bad += [f'{path.name}:{ln}' for ln in _own_pump_loops(tree)]
    assert not bad, (
        f'{bad} pump the Tk event loop in a while loop of their own; use '
        'tests._tk_wait.pump_until (wait for a condition) or settle (run '
        'for a while)')


def test_no_test_file_imports_a_wait_from_another_test_file():
    bad = []
    for path in _test_files():
        tree = ast.parse(path.read_text(encoding='utf-8'))
        bad += [f'{path.name}:{ln} {name}'
                for ln, name in _waits_imported_from_a_test_module(tree)]
    assert not bad, f'{bad}: import waits from tests._tk_wait'


def test_the_allowed_list_only_names_files_that_still_need_it():
    stale = [name for name in sorted(_MAY_PUMP_ON_THEIR_OWN)
             if not _own_pump_loops(ast.parse(
                 (TESTS / name).read_text(encoding='utf-8')))]
    assert not stale, f'{stale} no longer pump on their own; drop them from the list'


def test_the_scans_see_planted_copies():
    planted = ast.parse(
        "import time\n"
        "from tests.test_transform_records import _pump, settle\n"
        "from tests._tk_wait import pump_until\n"
        "def f(root, ed, done):\n"
        "    end = time.monotonic() + 5\n"
        "    while time.monotonic() < end and not done():\n"
        "        root.update()\n"
        "        time.sleep(0.01)\n"
        "    while not done():\n"
        "        if True:\n"
        "            ed.update()\n"
        "def left_alone(root, d, done):\n"
        "    while not done():\n"
        "        d.update({'a': 1})\n"
        "        root.update_idletasks()\n"
        "    for _ in range(3):\n"
        "        root.update()\n"
        "    pump_until(root, done)\n")
    assert _own_pump_loops(planted) == [6, 9]
    assert _waits_imported_from_a_test_module(planted) == [(2, '_pump'), (2, 'settle')]


class _Loop:
    """A stand-in for a Tk widget: counts runs of its event loop, and can
    make ``done`` hold after a given run."""

    def __init__(self, holds_after=None):
        self.runs = 0
        self.holds_after = holds_after

    def update(self):
        self.runs += 1

    def done(self):
        return self.holds_after is not None and self.runs >= self.holds_after


def test_pump_until_runs_the_loop_until_done_holds():
    w = _Loop(holds_after=3)
    assert pump_until(w, w.done, timeout=5, pause=0) is True
    assert w.runs == 3


def test_pump_until_runs_nothing_when_done_already_holds():
    w = _Loop(holds_after=0)
    assert pump_until(w, w.done, timeout=5) is True
    assert w.runs == 0


def test_pump_until_answers_false_once_time_runs_out():
    w = _Loop()
    t0 = time.monotonic()
    assert pump_until(w, w.done, timeout=0.1, pause=0.01) is False
    took = time.monotonic() - t0
    assert 0.1 <= took < 2.0, took
    assert w.runs >= 2


def test_settle_runs_the_loop_for_the_time_asked():
    w = _Loop()
    t0 = time.monotonic()
    settle(w, 0.1, pause=0.01)
    took = time.monotonic() - t0
    assert 0.1 <= took < 2.0, took
    assert w.runs >= 2


def test_wait_until_polls_without_an_event_loop():
    calls = []
    assert wait_until(lambda: calls.append(1) or len(calls) >= 3,
                      timeout=5, pause=0) is True
    assert len(calls) == 3
    t0 = time.monotonic()
    assert wait_until(lambda: False, timeout=0.1, pause=0.01) is False
    assert 0.1 <= time.monotonic() - t0 < 2.0
