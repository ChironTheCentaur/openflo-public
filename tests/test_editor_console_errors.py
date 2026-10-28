"""The log pane's ``except tk.TclError`` handlers catch Tcl errors at runtime.

``tkinter`` was once imported only under ``TYPE_CHECKING`` in
editor_console.py, so evaluating ``tk.TclError`` in an ``except`` clause raised
``NameError`` instead of catching the error. These tests drive the mixin with
stub widgets (no Tk window) and also check, statically, that no module in
src/openflo names a type-checking-only import in an ``except`` clause.
"""
from __future__ import annotations

import ast
import glob
import os
import queue
import tkinter as tk

from openflo.editor_console import ConsoleLogMixin

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'src', 'openflo')


class _DeadText:
    """A Text widget whose window is already destroyed."""

    def config(self, **kw):
        raise tk.TclError('invalid command name ".log"')


class _Editor(ConsoleLogMixin):
    def __init__(self, exists_raises=False):
        self._log_queue = queue.Queue()
        self._log_text = _DeadText()
        self._exists_raises = exists_raises
        self.rescheduled = []

    def winfo_exists(self):
        if self._exists_raises:
            raise tk.TclError('application has been destroyed')
        return 1

    def after(self, ms, func=None, *args):
        self.rescheduled.append((ms, func))


def test_append_log_to_destroyed_pane_is_ignored():
    _Editor()._append_log('sample_01 loaded\n')     # must not raise


def test_drain_log_stops_quietly_when_window_is_gone():
    ed = _Editor(exists_raises=True)
    ed._log_queue.put('line\n')
    ed._drain_log()                                  # must not raise
    assert ed.rescheduled == []
    assert ed._log_queue.empty()


def test_drain_log_reschedules_while_window_exists():
    ed = _Editor()
    ed._drain_log()
    assert ed.rescheduled == [(300, ed._drain_log)]


def _type_checking_only_names(tree):
    names = set()
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == 'TYPE_CHECKING'):
            for sub in node.body:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split('.')[0])
    for node in tree.body:                          # also imported for real?
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.discard((a.asname or a.name).split('.')[0])
    return names


def _root_name(expr):
    while isinstance(expr, ast.Attribute):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def test_no_except_clause_names_a_type_checking_only_import():
    bad = []
    for path in sorted(glob.glob(os.path.join(SRC, '**', '*.py'), recursive=True)):
        with open(path, encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        hidden = _type_checking_only_names(tree)
        if not hidden:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            types = (node.type.elts if isinstance(node.type, ast.Tuple)
                     else [node.type])
            for t in types:
                if _root_name(t) in hidden:
                    bad.append(f'{os.path.relpath(path, SRC)}:{node.lineno} '
                               f'{ast.unparse(t)}')
    assert not bad, ('except clauses name imports that exist only under '
                     'TYPE_CHECKING (NameError at runtime): ' + ', '.join(bad))
