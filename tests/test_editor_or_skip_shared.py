"""The hidden-editor helper lives once, in conftest.editor_or_skip.

Forty test files used to carry their own copy of ``_editor_or_skip`` and the
copies had drifted: some skipped with ``pytest.skip`` where Tk failed, so a CI
run that promised a GUI (OPENFLO_REQUIRE_GUI=1) went green without testing it,
and one never hid its editor. These tests pin what the shared helper does and
stop the copies coming back.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from tests import conftest
from tests.conftest import editor_or_skip

TESTS = pathlib.Path(__file__).resolve().parent


def _builds_tk_and_editor(func, src):
    """True when this function's own body (not a nested def) calls Tk() and
    ViewGateEditorWindow(...)."""
    calls = set()
    stack = list(func.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', '')
            calls.add(name)
        stack.extend(ast.iter_child_nodes(node))
    return 'Tk' in calls and 'ViewGateEditorWindow' in calls


def _local_editor_builders():
    """(file, function) for every function in tests/ outside conftest that
    starts Tk and builds the editor itself. Helper modules count too, so a
    copy cannot hide in a tests/_something.py."""
    found = set()
    for path in sorted(TESTS.glob('*.py')):
        if path.name == 'conftest.py':
            continue
        src = path.read_text(encoding='utf-8')
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and _builds_tk_and_editor(node, src):
                found.add((path.name, node.name))
    return found


def test_no_test_file_builds_its_own_hidden_editor():
    """No exceptions any more: the last sixteen builders (fixtures with their
    own teardown and one-off dialog tests) moved onto editor_or_skip, so a new
    one fails here instead of joining an allowlist."""
    found = sorted(_local_editor_builders())
    assert not found, (
        f'{found} start Tk and build ViewGateEditorWindow themselves; call '
        'tests.conftest.editor_or_skip() and add the test\'s extras after it')


def test_the_scan_sees_a_planted_copy():
    planted = ast.parse(
        'def _editor_or_skip():\n'
        '    root = tk.Tk()\n'
        '    ed = gui.ViewGateEditorWindow(root, fcs_dir=None)\n'
        '    return root, ed\n'
        'def _wrapper():\n'
        '    return editor_or_skip()\n'
        'def _outer():\n'
        '    def build():\n'
        '        return gui.ViewGateEditorWindow(tk.Tk(), fcs_dir=None)\n'
        '    later = lambda: (tk.Tk(), gui.ViewGateEditorWindow(None))\n'
        '    return build, later\n')
    funcs = {n.name: n for n in planted.body}
    assert _builds_tk_and_editor(funcs['_editor_or_skip'], '')
    assert not _builds_tk_and_editor(funcs['_wrapper'], '')
    # A nested builder is that nested function's own finding, not its parent's.
    assert not _builds_tk_and_editor(funcs['_outer'], '')


def test_editor_is_hidden_and_confirms_are_answered_yes():
    root, ed = editor_or_skip()
    try:
        import openflo.gui as gui
        assert ed.master is root
        assert root.state() == 'withdrawn'
        assert ed.state() == 'withdrawn'
        assert gui.messagebox.askyesno('Q', 'Continue?') is True
    finally:
        ed.destroy()
        root.destroy()


def test_accept_confirms_false_leaves_the_dialog_guard_in_place():
    root, ed = editor_or_skip(accept_confirms=False)
    try:
        import tkinter.messagebox as messagebox
        stand_in = conftest._DIALOG_STAND_INS[(messagebox, 'askyesno')]
        assert messagebox.askyesno is stand_in
    finally:
        ed.destroy()
        root.destroy()


class _TkWouldNotStart(Exception):
    pass


def test_tk_failure_goes_through_gui_unavailable(monkeypatch):
    """Not pytest.skip: gui_unavailable is what fails a run that promised a
    GUI, instead of letting it skip silently."""
    tk = pytest.importorskip('tkinter')

    def no_display():
        raise tk.TclError('no display name and no $DISPLAY variable')

    seen = []

    def record(reason):
        seen.append(reason)
        raise _TkWouldNotStart(reason)

    monkeypatch.setattr(tk, 'Tk', no_display)
    monkeypatch.setattr(conftest, 'gui_unavailable', record)
    with pytest.raises(_TkWouldNotStart):
        editor_or_skip()
    assert len(seen) == 1
    assert seen[0].startswith('Tk cannot initialise without a display')
    assert '$DISPLAY' in seen[0]
