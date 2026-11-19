#!/usr/bin/env python
"""Stop new functions that are too complex or too long (survey item R6).

    python scripts/function_size.py                  # check against the list
    python scripts/function_size.py --list           # every function over a limit
    python scripts/function_size.py --write-baseline # drop fixed entries

A function in src/openflo fails this check when its cyclomatic complexity is
30 or more (CC_LIMIT) or it runs past 200 lines (LINE_LIMIT), unless it is in
scripts/function_size_baseline.json, the list of functions that were already
over a limit when this check was written. The list may only shrink:

  * a function over a limit that is not on the list fails: split it;
  * a listed function that is back under both limits (or gone) passes, with a
    note to run --write-baseline, which removes it from the list;
  * --write-baseline never adds a function unless --allow-new is given (a
    listed function that was renamed or moved shows up as new; say why in
    the PR).

A listed function may still change size: the numbers in the list are a
record of the last --write-baseline, not a cap, so fixes inside a long
function are not blocked while it waits for its own split.

Complexity is counted as radon's `cc` counts it (radon 6), with the standard
library only so CI's lint job needs nothing installed: 1, plus 1 per `if`,
conditional expression, loop (plus 1 for its `else`), comprehension `for` and
each of its `if`s, `except` handler, `try ... else`, `assert` (nothing inside
it), `match` case (not counting `case _` or a bare capture), and each extra
operand of `and`/`or`. A nested
function or class is scored on its own and adds nothing to its parent;
a lambda's body counts toward the function it is in. (One difference from
radon: methods of a class defined inside a function are scored here, while
radon leaves them out.) Length is the def line
to the last line of the body (decorators not counted).

Exit 0 = nothing over a limit beyond the list; 1 = a new function over a limit.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'src/openflo'
# Byte-identical copies of another repository's module (each one's hash is
# checked by a test: tests/test_stats_plt.py for the Pocket Lab Table). They
# are fixed where they are written, so the rules here do not measure them.
VENDORED = frozenset({'src/openflo/pocketlab_table.py'})
BASELINE_NAME = 'function_size_baseline.json'
CC_LIMIT = 30      # the survey's "30+"
LINE_LIMIT = 200   # the survey's "past 200 lines"


class Size(NamedTuple):
    """One function's place and measures."""
    path: str       # repo-relative posix path
    name: str       # Class.method.inner, as it nests in the file
    line: int
    cc: int
    lines: int

    @property
    def key(self) -> str:
        """The baseline key: path::name."""
        return f'{self.path}::{self.name}'

    @property
    def over(self) -> bool:
        """True when either measure is over its limit."""
        return self.cc >= CC_LIMIT or self.lines > LINE_LIMIT

    def __str__(self) -> str:
        return (f'{self.path}:{self.line}: {self.name}: CC {self.cc}, '
                f'{self.lines} lines')


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _decisions(node: ast.AST) -> int:
    """Decision points in `node` and below, stopping at nested defs/classes."""
    if isinstance(node, ast.Assert):
        return 1    # radon does not look inside an assert
    n = 0
    if isinstance(node, ast.Try):     # radon 6 does not count try/except* groups
        n += len(node.handlers) + bool(node.orelse)
    elif isinstance(node, ast.BoolOp):
        n += len(node.values) - 1
    elif isinstance(node, (ast.If, ast.IfExp)):
        n += 1
    elif isinstance(node, ast.Match):
        wildcard = any(isinstance(c.pattern, ast.MatchAs) and c.pattern.pattern is None
                       for c in node.cases)
        n += max(0, len(node.cases) - wildcard)
    elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        n += 1 + bool(node.orelse)
    elif isinstance(node, ast.comprehension):
        n += 1 + len(node.ifs)
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, _DEFS):
            n += _decisions(child)
    return n


def _complexity(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """radon's cyclomatic complexity of one function's own body."""
    return 1 + sum(_decisions(stmt) for stmt in fn.body
                   if not isinstance(stmt, _DEFS))


def measure_source(path: str, source: str) -> list[Size]:
    """Every function and method in `source`, nested ones included."""
    out: list[Size] = []

    def visit(body, prefix):
        for node in body:
            for sub in _nested(node):
                if isinstance(sub, ast.ClassDef):
                    visit(sub.body, f'{prefix}{sub.name}.')
                else:
                    name = f'{prefix}{sub.name}'
                    end = sub.end_lineno or sub.lineno
                    out.append(Size(path, name, sub.lineno, _complexity(sub),
                                    end - sub.lineno + 1))
                    visit(sub.body, f'{name}.')

    visit(ast.parse(source, filename=path).body, '')
    return out


def _nested(node: ast.AST):
    """`node` if it is a def or class, else the defs/classes directly inside
    it without crossing another def or class (a def under `if`/`try`)."""
    if isinstance(node, _DEFS):
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _nested(child)


def measure_tree(root: Path = ROOT) -> list[Size]:
    """Measures for every .py file under src/openflo."""
    out: list[Size] = []
    for p in sorted((root / PACKAGE).rglob('*.py')):
        if '__pycache__' in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        if rel in VENDORED:
            continue
        out.extend(measure_source(rel, p.read_text(encoding='utf-8')))
    return out


def compare(sizes: list[Size], baseline: dict) -> tuple[list[Size], list[str]]:
    """(functions over a limit that are not listed, notes on listed entries
    that are now under both limits or gone)."""
    over = {s.key: s for s in sizes if s.over}
    new = [s for k, s in over.items() if k not in baseline]
    notes = [f'{k}: now under both limits or gone, run --write-baseline to '
             'drop it from the list' for k in sorted(baseline) if k not in over]
    return new, notes


def dump_baseline(sizes: list[Size]) -> str:
    """The list as JSON: {path::name: {cc, lines}} for each function over a limit."""
    entries = {s.key: {'cc': s.cc, 'lines': s.lines} for s in sizes if s.over}
    return json.dumps(dict(sorted(entries.items())), indent=1) + '\n'


def _read_baseline(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def main(argv=None) -> int:
    """Command line entry point; returns the exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--write-baseline', action='store_true',
                    help='rewrite the list from the current tree (drops fixed entries)')
    ap.add_argument('--allow-new', action='store_true',
                    help='with --write-baseline: also add functions (say why in the PR)')
    ap.add_argument('--list', action='store_true',
                    help='print every function over a limit, listed ones too')
    ap.add_argument('--root', type=Path, default=ROOT, help=argparse.SUPPRESS)
    ap.add_argument('--baseline', type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    baseline_path = args.baseline or args.root / 'scripts' / BASELINE_NAME

    try:
        sizes = measure_tree(args.root)
    except SyntaxError as exc:
        print(f'cannot parse: {exc}')
        return 1
    if args.list:
        for s in sizes:
            if s.over:
                print(s)
        return 0
    baseline = _read_baseline(baseline_path)
    new, notes = compare(sizes, baseline)
    if args.write_baseline:
        if new and baseline_path.exists() and not args.allow_new:
            for s in new:
                print(s)
            print('These functions are over a limit and not on the list. Split '
                  'them, or rerun with --allow-new and say why in the PR (a '
                  'renamed or moved function shows up here too).')
            return 1
        baseline_path.write_text(dump_baseline(sizes), encoding='utf-8')
        print(f'wrote {baseline_path.name}: '
              f'{sum(1 for s in sizes if s.over)} listed functions')
        return 0
    for note in notes:
        print(note)
    if new:
        print(f'\n{len(new)} function(s) over a limit (CC {CC_LIMIT}+ or more '
              f'than {LINE_LIMIT} lines) and not on {baseline_path.name}. '
              'Split them into helpers:')
        for s in new:
            print(f'  {s}')
        return 1
    print(f'function size: OK ({len(baseline)} listed, none new)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
