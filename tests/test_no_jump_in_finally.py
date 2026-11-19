"""No ``return``, ``break`` or ``continue`` leaves a ``finally`` block in
OpenFlo's code (PEP 765): it silently drops any exception in flight, and
Python 3.14 warns about it at import ("'break' in a 'finally' block"), with
an error to follow in a later Python. Checked on the syntax tree, so it
holds on every Python the suite runs on, not only 3.14."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'


def _jumps_out_of(finalbody):
    """The return / break / continue statements in a finally body that leave
    it: a break or continue inside a loop that starts in the finally stays
    there, and a nested function or class is its own scope."""
    found = []

    def walk(node, in_loop):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda, ast.ClassDef)):
            return
        if isinstance(node, ast.Return):
            found.append(node)
        elif isinstance(node, (ast.Break, ast.Continue)) and not in_loop:
            found.append(node)
        loop = isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        for child in ast.iter_child_nodes(node):
            # A loop's else runs after the loop: a break there is not its own.
            inside = in_loop or (loop and child not in node.orelse)
            walk(child, inside)
    for stmt in finalbody:
        walk(stmt, False)
    return found


def _violations(source, name):
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Try, getattr(ast, 'TryStar', ast.Try))):
            for jump in _jumps_out_of(node.finalbody):
                out.append(f'{name}:{jump.lineno} '
                           f'{type(jump).__name__.lower()}')
    return out


def test_the_check_finds_each_kind_of_jump():
    src = '''
def f():
    while True:
        try:
            pass
        finally:
            try:
                g()
            except Exception:
                break
    for x in y:
        try:
            pass
        finally:
            continue
    try:
        pass
    finally:
        return 1
'''
    assert sorted(_violations(src, 'x')) == ['x:10 break', 'x:15 continue',
                                             'x:19 return']


def test_the_check_allows_jumps_that_stay_inside_the_finally():
    src = '''
def f():
    try:
        pass
    finally:
        for x in y:
            if x:
                break
            continue
        def g():
            return 1
        h = lambda: 2
'''
    assert _violations(src, 'x') == []


def test_no_jump_leaves_a_finally_in_openflo():
    found = []
    for path in sorted(SRC.rglob('*.py')):
        found += _violations(path.read_text(encoding='utf-8'),
                             str(path.relative_to(SRC)))
    assert not found, found
