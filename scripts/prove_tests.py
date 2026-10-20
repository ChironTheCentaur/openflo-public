#!/usr/bin/env python
"""Prove that every test can fail: break the code each test guards, one small
change at a time, and record a change that makes the test fail.

A test that passes is only half a proof. It shows the code agrees with the
test today; it does not show the test would notice if the code stopped being
right. The other half is to break the code on purpose and watch the test go
red. This script does that for the whole suite, mechanically:

  1. ``coverage`` runs the suite once with per-test contexts, so for every
     line of openflo we know which tests execute it.
  2. Every test runs once on an unmutated copy of the checkout and must pass
     there (the positive half).
  3. For each test, the code it guards is mutated one change at a time (a
     ``<`` becomes ``<=``, a ``+`` becomes ``-``, a constant moves by one, an
     ``and`` becomes ``or``, a ``return x`` returns ``None``, a condition is
     negated, a statement, raise or import is deleted, an ``except`` catches
     nothing ...). It starts from the lines the test executes, preferring
     lines that mention the words the test uses and lines few other tests
     reach, so the mutant is about what *this* test guards. It then tries
     what the test names but coverage cannot attribute to it: functions,
     constants and defaults it reads, scripts/ it loads, and files such as a
     workflow or pyproject.toml it checks (those lose one line at a time).
  4. Each mutant runs against the not yet proven tests that reach the
     mutated line. A test that passed in step 2 and fails on the mutant is
     proven, and the mutant is recorded as its evidence (the negative half).

A failure that comes from an ``assert`` (or ``pytest.raises`` / ``approx``
noticing the change) is the proof we want: the test *checked* something and
the check caught the break. A failure that is only an exception escaping the
code under test is recorded too, marked ``crash``, and the search keeps
looking for an assertion kill for it. (For a test whose whole claim is "this
imports" or "this does not raise", the error is the check.)

The working tree is never modified: each worker runs in its own copy of the
checkout, with that copy's src/ first on ``PYTHONPATH``, made from a snapshot
taken when the sweep starts.

Not part of the test suite; a full sweep takes hours. Typical use:

    # 1. per-test coverage (GUI tests need a display: run under xvfb-run on Linux)
    xvfb-run -a python -m pytest -n auto --cov=openflo --cov-context=test \\
        --cov-report= -p no:randomly
    # 2. the sweep; results land in docs/test-assurance/
    xvfb-run -a python scripts/prove_tests.py --coverage .coverage
    # only some tests:
    xvfb-run -a python scripts/prove_tests.py --coverage .coverage -k test_gating

Output (``--out``, default docs/test-assurance):
  proofs.json   per test: verdict, the mutant that failed it, mutants tried
  mutants.jsonl every mutant run: where, what changed, which tests it killed

    # tests the search cannot reach, with breaks chosen by hand
    xvfb-run -a python scripts/prove_tests.py aimed \
        docs/test-assurance/aimed.json aimed-out
    # one table from several sweeps (e.g. a second run with extras installed)
    python scripts/prove_tests.py report docs/test-assurance a.json b.json

writes ledger.csv: one row per test, with the data it runs on, whether it
passes on the real code, and the break that made it fail.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pristine checkout mutants are made from and restored to: a frozen copy
# taken when the sweep starts (see snapshot()), so editing the repository
# mid-sweep cannot leak into a run. File keys everywhere are paths relative to
# it, e.g. 'src/openflo/gating.py', 'scripts/typecheck.py'.
SNAP = ROOT
_COPY_IGNORE = shutil.ignore_patterns(
    '.venv', 'venv', '__pycache__', '.pytest_cache', '*.pyc',
    'test-assurance', '.coverage*', 'build', 'dist', '*.egg-info')
# What can be mutated: Python under these directories, and (line deletion)
# any other text file a test names, such as a workflow or pyproject.toml.
MUTABLE_PY = ('src/openflo/', 'scripts/')


def snapshot(base):
    """Copy the checkout (code, tests and the files tests read) to ``base``
    and make it the pristine source of every workspace."""
    global SNAP
    dst = os.path.join(base, 'checkout')
    shutil.copytree(ROOT, dst, ignore=_COPY_IGNORE)
    SNAP = dst
    return dst

# ── Mutants ─────────────────────────────────────────────────────────────────

_CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot,
    ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In,
}
_CMP_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}
_BIN_SWAP = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
    ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd,
}
_OP_TEXT = {
    ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==',
    ast.NotEq: '!=', ast.Is: 'is', ast.IsNot: 'is not', ast.In: 'in',
    ast.NotIn: 'not in', ast.Add: '+', ast.Sub: '-', ast.Mult: '*',
    ast.Div: '/', ast.FloorDiv: '//', ast.Mod: '%', ast.Pow: '**',
    ast.BitAnd: '&', ast.BitOr: '|', ast.And: 'and', ast.Or: 'or',
}


@dataclass
class Mutant:
    id: str
    file: str            # relative to the checkout, e.g. src/openflo/gating.py
    line: int            # first line of the mutated node
    lines: tuple         # every line the node spans
    span: tuple          # (lineno, col, end_lineno, end_col), ast offsets
    new: str             # replacement source for the span
    what: str            # human description, e.g. "'<' -> '<='"
    old: str = ''


def _function_bodies(tree):
    """Nodes worth mutating: function bodies and default values, and the
    module- and class-level statements other than imports and definitions
    (constants, tables, a script's top-level code). A mutant there that breaks
    the import only shows as a crash; one that changes a constant is often the
    only thing that can fail a test about it."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body:
                out.extend(ast.walk(stmt))
            # Default argument values: a default is behaviour too (e.g.
            # reproducible=True), and changing one breaks no import.
            for d in [*node.args.defaults, *node.args.kw_defaults]:
                if isinstance(d, ast.Constant):
                    out.append(d)
    top = list(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            top.extend(node.body)
    for stmt in top:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if stmt in tree.body and not (isinstance(stmt, ast.ImportFrom)
                                          and stmt.module == '__future__'):
                out.append(stmt)
        elif not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
            out.extend(ast.walk(stmt))
    seen, uniq = set(), []
    for n in out:
        if id(n) not in seen:
            seen.add(id(n))
            uniq.append(n)
    return uniq


def _is_docstring(node, parents):
    p = parents.get(id(node))
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and isinstance(p, ast.Expr))


def _variants(node):
    """(replacement node, description) pairs for one AST node."""
    out = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = type(node.ops[0])
        for table in (_CMP_SWAP, _CMP_FLIP):
            if op in table:
                m = copy.deepcopy(node)
                m.ops = [table[op]()]
                out.append((m, f"'{_OP_TEXT[op]}' -> '{_OP_TEXT[table[op]]}'"))
    elif isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
        op = type(node.op)
        m = copy.deepcopy(node)
        m.op = _BIN_SWAP[op]()
        out.append((m, f"'{_OP_TEXT[op]}' -> '{_OP_TEXT[_BIN_SWAP[op]]}'"))
    elif isinstance(node, ast.BoolOp):
        op = type(node.op)
        m = copy.deepcopy(node)
        m.op = ast.Or() if op is ast.And else ast.And()
        other = 'or' if op is ast.And else 'and'
        out.append((m, f"'{_OP_TEXT[op]}' -> '{other}'"))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        out.append((copy.deepcopy(node.operand), "drop 'not'"))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        out.append((copy.deepcopy(node.operand), "drop unary '-'"))
    elif isinstance(node, ast.Constant) and not isinstance(node.value, str):
        v = node.value
        if isinstance(v, bool):
            out.append((ast.Constant(not v), f'{v} -> {not v}'))
        elif isinstance(v, int):
            out.append((ast.Constant(v + 1), f'{v} -> {v + 1}'))
            if v != 0:
                out.append((ast.Constant(0), f'{v} -> 0'))
        elif isinstance(v, float):
            nv = v * 2 if v else 1.0
            out.append((ast.Constant(nv), f'{v!r} -> {nv!r}'))
        elif v is None:
            pass
    elif isinstance(node, ast.Return) and node.value is not None:
        if not (isinstance(node.value, ast.Constant) and node.value.value is None):
            out.append((ast.Return(ast.Constant(None)), 'return None instead'))
    elif isinstance(node, ast.Return):
        out.append((ast.Pass(), 'delete return'))
    elif isinstance(node, ast.AugAssign) and type(node.op) in _BIN_SWAP:
        op = type(node.op)
        m = copy.deepcopy(node)
        m.op = _BIN_SWAP[op]()
        out.append((m, f"'{_OP_TEXT[op]}=' -> '{_OP_TEXT[_BIN_SWAP[op]]}='"))
    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr)) and not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
        out.append((ast.Pass(), 'delete statement'))
    elif isinstance(node, ast.Raise):
        out.append((ast.Pass(), 'delete raise'))
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        out.append((ast.Pass(), 'delete import'))
    return out


def _splice(src_lines_b, span, new_text):
    """Replace the (lineno, col, end_lineno, end_col) span of a file held as a
    list of byte lines with ``new_text``; ast columns are UTF-8 byte offsets."""
    l0, c0, l1, c1 = span
    head = src_lines_b[l0 - 1][:c0]
    tail = src_lines_b[l1 - 1][c1:]
    return (src_lines_b[:l0 - 1] + [head + new_text.encode() + tail]
            + src_lines_b[l1:])


def _text_mutants(relpath, raw):
    """A file that is not Python (a workflow, pyproject.toml, the README):
    delete one line at a time."""
    out = []
    for i, line in enumerate(raw.splitlines(keepends=True), start=1):
        text = line.decode(errors='replace').strip()
        if not text or text.startswith('#'):
            continue
        body = line.rstrip(b'\r\n')
        out.append(Mutant(f'{relpath}:{i}:line', relpath, i, (i,),
                          (i, 0, i, len(body)), '', 'delete line',
                          text[:120]))
    return out


def mutants_for_file(relpath):
    """Every mutant of <checkout>/<relpath> that still parses."""
    path = os.path.join(SNAP, relpath)
    with open(path, 'rb') as f:
        raw = f.read()
    if not relpath.endswith('.py'):
        return _text_mutants(relpath, raw)
    tree = ast.parse(raw)
    lines_b = raw.splitlines(keepends=True)
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    out = []
    for node in _function_bodies(tree):
        if not hasattr(node, 'lineno') or _is_docstring(node, parents):
            continue
        # f-string pieces and annotations: mutating them proves nothing.
        if isinstance(parents.get(id(node)), (ast.JoinedStr, ast.FormattedValue)):
            continue
        span = (node.lineno, node.col_offset, node.end_lineno,
                node.end_col_offset)
        old = b''.join(lines_b[span[0] - 1:span[2]]).decode(errors='replace')
        variants = _variants(node)
        parent = parents.get(id(node))
        if isinstance(parent, (ast.If, ast.IfExp, ast.While)) \
                and parent.test is node:
            variants.append((ast.UnaryOp(ast.Not(), copy.deepcopy(node)),
                             'negate condition'))
        if isinstance(parent, ast.ExceptHandler) and parent.type is node:
            variants = [(ast.Tuple([], ast.Load()), 'except catches nothing')]
        for i, (new_node, what) in enumerate(variants):
            try:
                text = ast.unparse(new_node)
            except Exception:
                continue
            if isinstance(node, ast.expr):
                text = f'({text})'
            elif '\n' in text:
                continue
            try:    # a parenthesised expression or a one-line statement
                ast.parse(text, mode='eval' if isinstance(node, ast.expr)
                          else 'exec')
            except SyntaxError:
                continue
            mid = f'{relpath}:{node.lineno}:{node.col_offset}:{type(node).__name__}:{i}'
            out.append(Mutant(mid, relpath, node.lineno,
                              tuple(range(node.lineno, node.end_lineno + 1)),
                              span, text, what,
                              old.strip().splitlines()[0][:120] if old.strip() else ''))
    return out


# ── Coverage map ────────────────────────────────────────────────────────────

def load_coverage(path):
    """``{(relpath, line): {test nodeid}}`` from a per-test-context coverage
    file, and the set of tests that appear in it."""
    import coverage
    data = coverage.CoverageData(path)
    data.read()
    line_tests = defaultdict(set)
    tests = set()
    for fname in data.measured_files():
        rel = os.path.relpath(os.path.realpath(fname), os.path.realpath(ROOT))
        if rel.startswith('..'):
            continue
        for line, ctxs in data.contexts_by_lineno(fname).items():
            for ctx in ctxs:
                if not ctx.endswith('|run'):
                    continue
                t = ctx[:-4]
                tests.add(t)
                line_tests[(rel.replace(os.sep, '/'), line)].add(t)
    return line_tests, tests


# ── Running pytest against a mutant ─────────────────────────────────────────

_ASSERTION = re.compile(
    r'AssertionError|assert |Failed: DID NOT RAISE|DID NOT WARN|approx|'
    r'pytest\.fail|Failed:')


def _nodeid_from_case(case):
    cls = case.get('classname', '')
    name = case.get('name', '')
    parts = cls.split('.')
    # classname is "tests.test_x" or "tests.test_x.TestClass"
    mod_parts, klass = [], None
    for i, p in enumerate(parts):
        mod_parts.append(p)
        if p.startswith('test_') or p == 'conftest':
            rest = parts[i + 1:]
            klass = '::'.join(rest) if rest else None
            break
    nid = '/'.join(mod_parts) + '.py'
    if klass:
        nid += '::' + klass
    return nid + '::' + name


class Workspace:
    """A private copy of the whole checkout for one worker thread: tests run
    from it, with its src/ first on the path, so a mutant in openflo, in
    scripts/ or in a file a test reads reaches only this worker."""

    def __init__(self, base):
        self.dir = tempfile.mkdtemp(prefix='prove-ws-', dir=base)
        shutil.copytree(SNAP, self.dir, ignore=_COPY_IGNORE,
                        dirs_exist_ok=True)

    def apply(self, m):
        dst = os.path.join(self.dir, m.file)
        with open(os.path.join(SNAP, m.file), 'rb') as f:
            lines_b = f.read().splitlines(keepends=True)
        with open(dst, 'wb') as f:
            f.write(b''.join(_splice(lines_b, m.span, m.new)))
        return dst

    def restore(self, m):
        shutil.copyfile(os.path.join(SNAP, m.file),
                        os.path.join(self.dir, m.file))


def run_tests(ws, tests, timeout):
    """Run ``tests`` with ws's src first on the path. Returns
    {nodeid: 'pass' | 'assert' | 'crash' | 'skip'} or None on timeout."""
    fd, xml = tempfile.mkstemp(suffix='.xml', dir=ws.dir)
    os.close(fd)
    env = dict(os.environ)
    env['PYTHONPATH'] = (os.path.join(ws.dir, 'src') + os.pathsep
                         + env.get('PYTHONPATH', ''))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('OPENFLO_REQUIRE_GUI', '1')
    cmd = [sys.executable, '-m', 'pytest', '-p', 'no:cacheprovider',
           '-p', 'no:randomly', '-q', '-o', 'addopts=', '--no-header',
           '-W', 'ignore', f'--junitxml={xml}', *tests]
    try:
        subprocess.run(cmd, cwd=ws.dir, env=env, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        os.unlink(xml)
        return None
    res = {}
    try:
        root = ET.parse(xml).getroot()
    except Exception:
        os.unlink(xml)
        return {}
    for case in root.iter('testcase'):
        nid = _nodeid_from_case(case)
        fail = case.find('failure')
        err = case.find('error')
        if case.find('skipped') is not None:
            res[nid] = 'skip'
        elif fail is not None:
            text = (fail.get('message') or '') + (fail.text or '')
            res[nid] = 'assert' if _ASSERTION.search(text) else 'crash'
        elif err is not None:
            res[nid] = 'crash'
        else:
            res[nid] = 'pass'
    os.unlink(xml)
    return res


# ── What data each test runs on ─────────────────────────────────────────────

_DATA_KINDS = (
    # (label, pattern) in order of preference; the first that matches names
    # the test's data. Patterns are matched against the test's own source,
    # its fixtures' and the module helpers they call.
    ('real data (opt-in, skipped unless provided)',
     re.compile(r'real_wsp_path|real_fcs_dir|OPENFLO_TEST_(WSP|FCS_DIR)')),
    ('committed golden files', re.compile(r'golden')),
    ('openflo.synthetic generator', re.compile(
        r'synthetic\.|from openflo(\.synthetic)? import synthetic|'
        r'synthetic_fcs|_sample\(|make_\w+_dataset|make_dataset|'
        r'make_compensation_controls|make_fmo_controls|make_calibration_beads|'
        r'make_size_beads')),
    ('generated in the test', re.compile(
        r'default_rng|np\.random|random\.|np\.(array|arange|linspace|zeros|'
        r'ones|full|eye|column_stack|vstack|stack|meshgrid|tile|repeat)\(|'
        r'pd\.DataFrame\(|DataFrame\(|write_fcs|_write_fcs|FlowData|'
        r'create_fcs|\.fcs[\'"]|\[\[|\{\s*[\'"]')),
)


def _source_of(obj):
    import inspect
    try:
        return inspect.getsource(obj)
    except Exception:
        return ''


# Tests marked as a known, not yet fixed defect (strict xfail): they fail on
# the real code on purpose and turn red the day the defect is fixed.
XFAIL = set()


def inventory(select=''):
    """{nodeid: data label} for every collected test, from its source."""
    import inspect

    import pytest

    items = []

    class Collect:
        def pytest_collection_finish(self, session):
            items.extend(session.items)

    pytest.main(['--collect-only', '-q', '-o', 'addopts=', '-p',
                 'no:randomly', os.path.join(ROOT, 'tests')],
                plugins=[Collect()])
    out = {}
    for it in items:
        if select not in it.nodeid:
            continue
        if it.get_closest_marker('xfail') is not None:
            XFAIL.add(it.nodeid)
        fn = getattr(it, 'function', None)
        mod = getattr(it, 'module', None)
        texts = [_source_of(fn)] if fn else []
        cls = getattr(it, 'cls', None)
        if cls is not None:
            for name in ('setup_method', 'setUp', 'setup_class', 'setUpClass'):
                if hasattr(cls, name):
                    texts.append(_source_of(getattr(cls, name)))
        info = getattr(it, '_fixtureinfo', None)
        if info is not None:
            for name, defs in info.name2fixturedefs.items():
                if name.startswith('_') or name in ('request', 'monkeypatch',
                                                    'capsys', 'caplog'):
                    continue
                for d in defs:
                    texts.append(_source_of(d.func))
                    if name.startswith('tmp_path') or name == 'tmpdir':
                        pass
        # Module helpers the test (and its fixtures) call, two levels deep.
        if mod is not None:
            helpers = {n: f for n, f in vars(mod).items()
                       if inspect.isfunction(f) and not n.startswith('test')}
            seen = set()
            for _ in range(2):
                joined = '\n'.join(texts)
                for n, f in helpers.items():
                    if n not in seen and re.search(r'\b' + re.escape(n) + r'\(',
                                                   joined):
                        seen.add(n)
                        texts.append(_source_of(f))
        joined = '\n'.join(texts)
        label = 'none (checks code or UI structure, not data)'
        for name, pat in _DATA_KINDS:
            if pat.search(joined):
                label = name
                break
        if getattr(it, 'callspec', None) is not None and label.startswith('none'):
            label = 'parametrised values in the test'
        out[it.nodeid] = label
    return out


_NAME_LINES = {}


def _named_lines():
    """{name: [(file, line), ...]} over the mutable Python: every function
    (signature and body), class, and module-level assigned name."""
    if not _NAME_LINES:
        for top in MUTABLE_PY:
            folder = os.path.join(SNAP, top)
            for name in sorted(os.listdir(folder)):
                if not name.endswith('.py'):
                    continue
                rel = top + name
                with open(os.path.join(folder, name), 'rb') as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                         ast.ClassDef)):
                        names = [node.name]
                    elif isinstance(node, (ast.Assign, ast.AnnAssign)) and \
                            node in tree.body:
                        tgts = (node.targets if isinstance(node, ast.Assign)
                                else [node.target])
                        names = [t.id for t in tgts if isinstance(t, ast.Name)]
                    else:
                        continue
                    lines = [(rel, ln) for ln in range(node.lineno,
                                                       node.end_lineno + 1)]
                    for n in names:
                        _NAME_LINES.setdefault(n, []).extend(lines)
    return _NAME_LINES


_PATHISH = re.compile(r'[\w./-]+\.(?:py|ya?ml|toml|cfg|ini|txt|md|json|bat|sh)'
                      r'|openflo(?:\.\w+)+')


def _referenced_files(strings):
    """Checkout files that strings in a test name: 'scripts/typecheck.py',
    'ci.yml', 'pyproject.toml', 'openflo.editor_base' ..."""
    out = set()
    for text in strings:
        for tok in _PATHISH.findall(text):
            if tok.startswith('openflo.'):
                cand = ['src/' + tok.replace('.', '/') + '.py']
            else:
                tok = tok[2:] if tok.startswith('./') else tok.lstrip('/')
                cand = [tok, 'scripts/' + tok, '.github/workflows/' + tok,
                        'src/openflo/' + tok]
            for c in cand:
                if os.path.isfile(os.path.join(SNAP, c)) and \
                        not c.startswith('tests/'):
                    out.add(c)
                    break
    return out


def _file_lines(rel):
    """Every line of a file that has a mutant."""
    try:
        return {(rel, m.line) for m in mutants_for_file(rel)}
    except (OSError, SyntaxError, ValueError):
        return set()


_COMMON = frozenset("""
self none true false return assert import from with open read text encoding
data test tests value values name names list dict str int float len range
utf mode path root file files sample samples channel channels gate gates
pytest monkeypatch tmp_path setattr np pd array numpy pandas result results
""".split())


def test_words(nodeid):
    """Distinctive words in a test's own source and id: what it is about."""
    path, _, rest = nodeid.partition('::')
    func = re.sub(r'\[.*\]$', '', rest.split('::')[-1])
    try:
        with open(os.path.join(ROOT, path), encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return set()
    text = rest
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == func:
            text += ast.get_source_segment(
                open(os.path.join(ROOT, path), encoding='utf-8').read(),
                node) or ''
    words = {w.lower() for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]{3,}', text)}
    return words - _COMMON


_LINE_TEXT = {}


def _line_score(fl, words):
    """How many of the test's words a candidate line mentions."""
    if fl not in _LINE_TEXT:
        rel, line = fl
        try:
            with open(os.path.join(SNAP, rel), encoding='utf-8',
                      errors='replace') as f:
                for i, txt in enumerate(f, start=1):
                    _LINE_TEXT[(rel, i)] = {
                        w.lower() for w in
                        re.findall(r'[A-Za-z_][A-Za-z0-9_]{3,}', txt)}
        except OSError:
            pass
    return len(_LINE_TEXT.get(fl, set()) & words)


def fallback_lines(nodeid, test_lines):
    """Lines a test reaches by name rather than by coverage: the openflo and
    scripts/ functions, classes and constants it (or a helper it calls in its
    own module) names, and the files it names. With nothing named, what the
    rest of its test file exercises."""
    path, _, rest = nodeid.partition('::')
    leaf = rest.split('::')[-1]
    func = re.sub(r'\[.*\]$', '', leaf)
    params = leaf[len(func):]
    try:
        with open(os.path.join(ROOT, path), 'rb') as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return set()
    defs = {n.name: n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            defs.update({m.name: m for m in n.body
                         if isinstance(m, (ast.FunctionDef,
                                           ast.AsyncFunctionDef))})
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    consts[t.id] = n
    names, strings = set(), [params]
    todo, seen = [defs[func]] if func in defs else [], set()
    while todo:                               # the test, then what it uses
        node = todo.pop()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                names.add(sub.id)
                for table in (defs, consts):
                    hit = table.get(sub.id)
                    if hit is not None and id(hit) not in seen:
                        seen.add(id(hit))
                        todo.append(hit)
            elif isinstance(sub, ast.Attribute):
                names.add(sub.attr)
            elif isinstance(sub, ast.alias):
                names.update(sub.name.split('.'))
                strings.append(sub.name)
            elif isinstance(sub, ast.ImportFrom) and sub.module:
                strings.append(sub.module)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                strings.append(sub.value)
                names.update(re.findall(r'[A-Za-z_]\w+', sub.value))
    named = _named_lines()
    out = set()
    for n in names:
        out.update(named.get(n, ()))
    for rel in _referenced_files(strings):
        out.update(_file_lines(rel))
    if not out:
        for other, lines in list(test_lines.items()):
            if other.startswith(path + '::') and other != nodeid:
                out.update(lines)
    return out


# ── The sweep ───────────────────────────────────────────────────────────────

@dataclass
class Proof:
    status: str = 'unproven'     # unproven | assert | crash | survived
    mutant: Mutant | None = None
    tried: int = 0
    visited: bool = False    # each test leads the search once
    extra: dict = field(default_factory=dict)


def _test_file(nid):
    return nid.split('::', 1)[0]


def _resume(proofs, ran_against, log_path):
    """Pick up an interrupted sweep from its mutants.jsonl: the kills it
    recorded stand, and mutants it ran are not run again on the same tests."""
    try:
        f = open(log_path, encoding='utf-8')
    except OSError:
        return 0
    n = 0
    with f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            n += 1
            ran_against[r['mutant']].update(r.get('tests', ()))
            m = Mutant(r['mutant'], r['file'], r['line'], (r['line'],),
                       (0, 0, 0, 0), '', r['change'], r.get('old', ''))
            for x, k in r['killed'].items():
                if x not in proofs:
                    continue
                kind = 'assert' if k == 'assert' else 'crash'
                px = proofs[x]
                if px.status == 'unproven' or (px.status == 'crash'
                                               and kind == 'assert'):
                    px.status, px.mutant = kind, m
    return n


def write_proofs(path, proofs):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({t: {'status': p.status, 'tried': p.tried,
                       'baseline': p.extra.get('baseline'),
                       'mutant': (None if p.mutant is None else {
                           'file': p.mutant.file, 'line': p.mutant.line,
                           'change': p.mutant.what, 'old': p.mutant.old})}
                   for t, p in sorted(proofs.items())}, f, indent=1)
    os.replace(tmp, path)


def sweep(line_tests, tests, *, budget, batch_cap, workers, durations,
          out, seed, ride_s=20.0):
    rng = random.Random(seed)
    test_lines = defaultdict(set)
    for (f, line), ts in line_tests.items():
        for t in ts:
            test_lines[t].add((f, line))
    mutant_cache = {}
    cache_lock = threading.Lock()

    def file_mutants(f):
        with cache_lock:
            if f not in mutant_cache:
                ms = mutants_for_file(f)
                idx = defaultdict(list)
                for m in ms:
                    idx[m.line].append(m)
                mutant_cache[f] = idx
            return mutant_cache[f]

    proofs = {t: Proof() for t in tests}
    ran_against = defaultdict(set)   # mutant id -> tests it ran
    lock = threading.Lock()
    log_path = os.path.join(out, 'mutants.jsonl')
    resumed = _resume(proofs, ran_against, log_path)
    if resumed:
        print(f'resuming: {resumed} mutants already run, '
              f'{sum(p.status == "assert" for p in proofs.values())} tests '
              f'already proven', flush=True)
    log = open(log_path, 'a', encoding='utf-8')
    base = tempfile.mkdtemp(prefix='prove-')
    snapshot(base)
    done = [0]
    t0 = time.time()

    def candidates(t):
        """Mutants on t's lines, then on what t names (which reaches what
        coverage cannot attribute to it: a default read through
        inspect.signature, a constant, code that ran at import in another
        test, a workflow file). Within each, lines that mention the words the
        test itself uses come first, then lines few other tests reach."""
        words = test_words(t)

        def rank(fl):
            return (-_line_score(fl, words), len(line_tests.get(fl, ())),
                    rng.random())
        lines = sorted(test_lines[t], key=rank)
        named = sorted(fallback_lines(t, {}) - set(lines), key=rank)
        for f, line in [*lines, *named]:
            ms = list(file_mutants(f).get(line, ()))
            rng.shuffle(ms)
            yield from ms

    def next_test():
        with lock:
            for t in order:
                p = proofs[t]
                if p.status in ('unproven', 'crash') and not p.visited:
                    p.visited = True
                    return t
        return None

    def work():
        ws = Workspace(base)
        while True:
            t = next_test()
            if t is None:
                return
            gen = candidates(t)
            while True:
                with lock:
                    p = proofs[t]
                    if p.status == 'assert' or p.tried >= budget:
                        break
                m = next(gen, None)
                if m is None:
                    break
                with lock:
                    # A mutant t already survived (as a passenger) has
                    # nothing new to say about t.
                    if t in ran_against[m.id]:
                        continue
                    # Every not-yet-proven test on this line rides along.
                    batch = {t}
                    for ln in m.lines:
                        for x in line_tests.get((m.file, ln), ()):
                            if x in proofs and proofs[x].status in (
                                    'unproven', 'crash'):
                                batch.add(x)
                    # Quick ones first, and only as many as fit the time
                    # cap: one slow passenger would hold up every mutant.
                    riders, spent = [], 0.0
                    for x in sorted(batch - {t},
                                    key=lambda x: (durations.get(x, 2.0), x)):
                        dx = durations.get(x, 2.0)
                        if len(riders) >= batch_cap - 1 or spent + dx > ride_s:
                            break
                        riders.append(x)
                        spent += dx
                    riders = [x for x in riders if x not in ran_against[m.id]]
                    batch = [t] + riders
                    ran_against[m.id].update(batch)
                    p.tried += 1
                est = sum(durations.get(x, 2.0) for x in batch)
                timeout = 60 + 6 * est
                ws.apply(m)
                try:
                    res = run_tests(ws, batch, timeout)
                    if res is None and len(batch) > 1:
                        # Timed out: the mutant probably loops. Try t alone.
                        res = run_tests(ws, [t], 60 + 6 * durations.get(t, 2.0))
                        if res is None:
                            res = {t: 'timeout'}
                    elif res is None:
                        res = {t: 'timeout'}
                finally:
                    ws.restore(m)
                killed = {x: r for x, r in res.items()
                          if r in ('assert', 'crash', 'timeout')}
                with lock:
                    for x, r in killed.items():
                        if x not in proofs:
                            continue
                        px = proofs[x]
                        kind = 'assert' if r == 'assert' else 'crash'
                        if px.status in ('unproven', 'survived') or (
                                px.status == 'crash' and kind == 'assert'):
                            px.status = kind
                            px.mutant = m
                            px.extra = {'failure': r}
                    log.write(json.dumps({
                        'mutant': m.id, 'file': m.file, 'line': m.line,
                        'change': m.what, 'old': m.old,
                        'ran': len(batch), 'tests': batch, 'killed': killed,
                    }) + '\n')
                    log.flush()
            with lock:
                p = proofs[t]
                if p.status == 'unproven':
                    p.status = 'survived' if test_lines[t] else 'no-lines'
                done[0] += 1
                n_ok = sum(1 for q in proofs.values() if q.status == 'assert')
                if done[0] % 50 == 0:
                    write_proofs(os.path.join(out, 'proofs.json'), proofs)
                if done[0] % 10 == 0:
                    print(f'[{time.time() - t0:7.0f}s] tests visited '
                          f'{done[0]}/{len(order)}, proven by assertion '
                          f'{n_ok}', flush=True)

    # Tests with the fewest lines first: they have the narrowest choice.
    # Tests coverage saw no openflo lines for (they run openflo in a
    # subprocess, or only call into compiled code): aim at the openflo
    # functions they name, else at what the rest of their file exercises.
    for t in tests:
        if not test_lines[t]:
            for fl in fallback_lines(t, test_lines):
                test_lines[t].add(fl)
                line_tests[fl].add(t)
    # The positive half: every test runs once on the unmutated copy, exactly
    # as the mutants will run, and must pass there. A test that skips or
    # fails on the real code proves nothing by failing on a mutant.
    chunks = [tests[i:i + 60] for i in range(0, len(tests), 60)]
    base_res = {}

    def baseline():
        ws0 = Workspace(base)
        while True:
            with lock:
                if not chunks:
                    break
                chunk = chunks.pop()
            est = sum(durations.get(x, 2.0) for x in chunk)
            res = run_tests(ws0, chunk, 300 + 6 * est) or {}
            with lock:
                base_res.update(res)
        shutil.rmtree(ws0.dir, ignore_errors=True)

    pool = [threading.Thread(target=baseline) for _ in range(workers)]
    for th in pool:
        th.start()
    for th in pool:
        th.join()
    for t in tests:
        r = base_res.get(t, 'missing')
        proofs[t].extra['baseline'] = r
        if r != 'pass':
            proofs[t].status = 'skipped' if r == 'skip' else f'baseline-{r}'
            proofs[t].visited = True
    print(f'baseline: {sum(1 for t in tests if base_res.get(t) == "pass")}'
          f'/{len(tests)} pass on the real code', flush=True)
    order = sorted(tests, key=lambda t: (len(test_lines[t]), t))
    for t in tests:
        if not test_lines[t] and proofs[t].status == 'unproven':
            proofs[t].status = 'no-lines'
    threads = [threading.Thread(target=work) for _ in range(workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log.close()
    shutil.rmtree(base, ignore_errors=True)
    return proofs


# ── Re-checking earlier evidence ────────────────────────────────────────────

def recheck(proofs_path, out, wanted=None, workers=4):
    """After the code moves on, re-run each test's recorded break instead of
    searching again: find the same change on the same original text (nearest
    line) in the current file and check the test still passes without it and
    fails with it. Tests whose break no longer exists, or no longer fails
    them, are written to ``leftover.txt`` for a fresh sweep."""
    with open(proofs_path, encoding='utf-8') as f:
        old = json.load(f)
    base = tempfile.mkdtemp(prefix='prove-recheck-')
    snapshot(base)
    by_file = defaultdict(list)
    leftover = []
    for t, p in old.items():
        if wanted is not None and t not in wanted:
            continue
        m = p.get('mutant')
        if p['status'] not in ('assert', 'crash') or not m:
            leftover.append(t)
            continue
        by_file[m['file']].append((t, m))
    groups = defaultdict(list)      # mutant -> tests
    for path, items in by_file.items():
        try:
            cands = mutants_for_file(path)
        except (OSError, SyntaxError):
            leftover.extend(t for t, _ in items)
            continue
        for t, m in items:
            same = [c for c in cands
                    if c.what == m['change'] and c.old == m['old']]
            if not same:
                leftover.append(t)
                continue
            best = min(same, key=lambda c: abs(c.line - m['line']))
            groups[best.id].append(t)
    by_id = {}
    for path in by_file:
        try:
            for c in mutants_for_file(path):
                if c.id in groups:
                    by_id[c.id] = c
        except (OSError, SyntaxError):
            pass
    proofs = {}
    lock = threading.Lock()
    todo = list(groups.items())

    def work():
        ws = Workspace(base)
        while True:
            with lock:
                if not todo:
                    break
                mid, tests = todo.pop()
            m = by_id[mid]
            before = run_tests(ws, tests, 900) or {}
            ws.apply(m)
            try:
                after = run_tests(ws, tests, 900)
            finally:
                ws.restore(m)
            ev = {'file': m.file, 'line': m.line, 'change': m.what,
                  'old': m.old}
            with lock:
                for t in tests:
                    if before.get(t) != 'pass':
                        leftover.append(t)
                        continue
                    k = 'crash' if after is None else after.get(t, 'missing')
                    if k in ('assert', 'crash'):
                        proofs[t] = {'status': k, 'tried': 1,
                                     'baseline': 'pass', 'mutant': ev}
                    else:
                        leftover.append(t)
                print(f'{len(proofs)} re-proven, {len(leftover)} left over, '
                      f'{len(todo)} breaks to go', flush=True)
        shutil.rmtree(ws.dir, ignore_errors=True)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'proofs.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(proofs.items())), f, indent=1)
    with open(os.path.join(out, 'leftover.txt'), 'w', encoding='utf-8') as f:
        f.write(''.join(t + '\n' for t in sorted(set(leftover))))
    return proofs


# ── Hand-aimed breaks ───────────────────────────────────────────────────────

def aimed(spec_path, out, workers=4):
    """Prove the tests the search cannot reach with a break chosen by hand.

    ``spec_path`` is a JSON list of ``{"tests": [nodeid, ...], "file":
    path, "find": text, "replace": text, "why": text}``. Each entry's tests
    run on the real checkout, then with the first ``find`` in ``file``
    replaced (every one with ``"all": true``; ``find`` empty: ``file`` is
    created holding ``replace``). Some
    tests guard the test suite's own machinery (a fixture writer, a check
    helper, conftest's GUI probe), so ``file`` may be under tests/ too; the
    test function being proven is never the one edited.
    """
    with open(spec_path, encoding='utf-8') as f:
        spec = json.load(f)
    base = tempfile.mkdtemp(prefix='prove-aimed-')
    snapshot(base)
    proofs = {}
    lock = threading.Lock()
    todo = list(enumerate(spec))

    def one(ws, e):
        tests = e['tests']
        path = os.path.join(ws.dir, e['file'])
        src = os.path.join(SNAP, e['file'])
        before = run_tests(ws, tests, 900) or {}
        if e['find']:
            with open(src, encoding='utf-8') as f:
                text = f.read()
            if e['find'] not in text:
                raise SystemExit(f"aimed: {e['find']!r} not in {e['file']}")
            line = text[:text.index(e['find'])].count('\n') + 1
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text.replace(e['find'], e['replace'],
                                     -1 if e.get('all') else 1))
        else:
            line = 1
            with open(path, 'w', encoding='utf-8') as f:
                f.write(e['replace'])
        try:
            after = run_tests(ws, tests, 900)
        finally:
            if e['find']:
                shutil.copyfile(src, path)
            else:
                os.unlink(path)
        old = e['find'].strip().splitlines()[0][:120] if e['find'] else \
            '(file absent)'
        m = {'file': e['file'], 'line': line, 'change': e['why'], 'old': old}
        for t, r in before.items():
            if not any(t == x or t.startswith(x + '[') for x in tests):
                continue
            if r != 'pass':
                st = 'skipped' if r == 'skip' else f'baseline-{r}'
                p = {'status': st, 'tried': 0, 'baseline': r, 'mutant': None}
            else:
                k = 'timeout' if after is None else after.get(t, 'missing')
                st = {'assert': 'assert', 'crash': 'crash',
                      'timeout': 'crash'}.get(k, 'survived')
                p = {'status': st, 'tried': 1, 'baseline': r,
                     'mutant': m if st != 'survived' else None}
            with lock:
                if t not in proofs or _RANK.get(st, -1) > _RANK.get(
                        proofs[t]['status'], -1):
                    proofs[t] = p
        print(f"{e['file']}:{line} {e['why']}: "
              + ', '.join(f'{t.split("::")[-1]}={proofs[t]["status"]}'
                          for t in proofs
                          if any(t == x or t.startswith(x + '[')
                                 for x in tests)), flush=True)

    def work():
        ws = Workspace(base)
        while True:
            with lock:
                if not todo:
                    break
                _, e = todo.pop(0)
            one(ws, e)
        shutil.rmtree(ws.dir, ignore_errors=True)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'proofs.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(proofs.items())), f, indent=1)
    counts = defaultdict(int)
    for p in proofs.values():
        counts[p['status']] += 1
    print(dict(counts))
    return proofs


# ── The ledger ──────────────────────────────────────────────────────────────

_VERDICT = {
    'assert': 'yes: an assertion caught the break',
    'crash': 'yes, by an error: the break made the test error out',
    'survived': 'NO: no break tried made it fail',
    'no-lines': 'NO: found no code of its to break',
    'skipped': 'not run: skipped on this machine',
}
_RANK = {'assert': 4, 'crash': 3, 'survived': 2, 'no-lines': 1, 'skipped': 0}


def merge_proofs(paths):
    """Several proofs.json files into one; the better verdict wins (a later
    sweep with the extras installed proves what an earlier one skipped)."""
    out = {}
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for t, p in json.load(f).items():
                if t not in out or _RANK.get(p['status'], -1) > _RANK.get(
                        out[t]['status'], -1):
                    out[t] = p
    return out


def write_ledger(path, proofs, inv):
    """One CSV row per collected test: the data it runs on, whether it passes
    on the real code, and the break that made it fail."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['test', 'data it runs on', 'passes on the real code',
                    'fails when the code is broken', 'the break that failed it'])
        for t in sorted(inv or proofs):
            p = proofs.get(t)
            if t in XFAIL:
                w.writerow([t, inv.get(t, ''),
                            'known defect, marked strict xfail: it fails on '
                            'the real code until the defect is fixed',
                            'yes: it turns red once the defect is fixed', ''])
                continue
            if p is None:
                passes, verdict = 'not run in this sweep', ''
            elif p['status'] == 'skipped':
                passes, verdict = 'skipped on this machine', ''
            elif p['status'].startswith('baseline-'):
                passes, verdict = f"NO ({p['status'][9:]})", ''
            else:
                passes = 'yes'
                verdict = _VERDICT.get(p['status'], p['status'])
            m = (p or {}).get('mutant')
            ev = (f"{m['file']}:{m['line']} {m['change']} (was: {m['old']})"
                  if m else '')
            w.writerow([t, inv.get(t, ''), passes, verdict, ev])


def report(out, proof_paths):
    """ledger.csv in ``out`` from one or more sweeps' proofs.json."""
    proofs = merge_proofs(proof_paths)
    inv = inventory()
    write_ledger(os.path.join(out, 'ledger.csv'), proofs, inv)
    counts = defaultdict(int)
    for t in inv:
        counts['xfail' if t in XFAIL else proofs[t]['status']
               if t in proofs else 'not run'] += 1
    print(dict(counts))
    return counts


def read_durations(log_path):
    """{nodeid: seconds} from a pytest --durations=0 log (call phase)."""
    d = {}
    if not log_path:
        return d
    pat = re.compile(r'^\s*([\d.]+)s (call|setup|teardown)\s+(\S+)')
    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            mm = pat.match(line)
            if mm:
                d[mm.group(3)] = d.get(mm.group(3), 0.0) + float(mm.group(1))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--coverage', required=True,
                    help='coverage data file recorded with --cov-context=test')
    ap.add_argument('--durations', help='pytest log with --durations=0')
    ap.add_argument('-k', dest='select', default='',
                    help='only tests whose node id matches this regular expression')
    ap.add_argument('--tests-from', help='only the test ids listed in this '
                    'file, one per line (e.g. the unproven ones of a sweep)')
    ap.add_argument('--include-uncovered', action='store_true',
                    help='also try the collected tests coverage saw no openflo '
                    'line for (aimed by name, see fallback_lines)')
    ap.add_argument('--budget', type=int, default=12,
                    help='mutants to try per test before calling it survived')
    ap.add_argument('--batch', type=int, default=40,
                    help='most tests run against one mutant')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 2)
    ap.add_argument('--seed', type=int, default=20261014)
    ap.add_argument('--out', default=os.path.join(ROOT, 'docs',
                                                  'test-assurance'))
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ['report']:
        # prove_tests.py report OUT_DIR proofs.json [more proofs.json ...]
        report(argv[1], argv[2:])
        return 0
    if argv[:1] == ['recheck']:
        # prove_tests.py recheck OLD_PROOFS.json OUT_DIR [TESTS.txt]
        wanted = None
        if len(argv) > 3:
            with open(argv[3], encoding='utf-8') as f:
                wanted = {ln.strip() for ln in f if ln.strip()}
        recheck(argv[1], argv[2], wanted)
        return 0
    if argv[:1] == ['aimed']:
        # prove_tests.py aimed SPEC.json OUT_DIR
        aimed(argv[1], argv[2])
        return 0
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    line_tests, tests = load_coverage(args.coverage)
    if args.include_uncovered:
        tests = set(tests) | set(inventory())
    tests = sorted(t for t in tests if re.search(args.select, t))
    if args.tests_from:
        with open(args.tests_from, encoding='utf-8') as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        tests = [t for t in tests if t in wanted]
    print(f'{len(tests)} tests, {len(line_tests)} covered lines', flush=True)
    proofs = sweep(line_tests, tests, budget=args.budget,
                   batch_cap=args.batch, workers=args.workers,
                   durations=read_durations(args.durations), out=args.out,
                   seed=args.seed)
    write_proofs(os.path.join(args.out, 'proofs.json'), proofs)
    counts = defaultdict(int)
    for p in proofs.values():
        counts[p.status] += 1
    print(dict(counts))
    return 0


if __name__ == '__main__':
    sys.exit(main())
