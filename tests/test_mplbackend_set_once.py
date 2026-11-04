"""The test suite picks matplotlib's headless backend in one place.

``os.environ.setdefault('MPLBACKEND', 'Agg')`` was copied into 105 test
files, at module level and again inside helpers, so a test only ran
headless if its file remembered the line. tests/conftest.py now sets it once,
at import, before pytest imports any test module, and these tests stop the
copies coming back. Tests that hand a subprocess its own environment, or
that pin the backend for one test with monkeypatch, are not copies of the
suite-wide default and are left alone.
"""
from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

TESTS = pathlib.Path(__file__).resolve().parent


def _environ(node):
    """True for the expression ``os.environ``."""
    return (isinstance(node, ast.Attribute) and node.attr == 'environ'
            and isinstance(node.value, ast.Name) and node.value.id == 'os')


def _mpl_default_sites(tree):
    """Line numbers that set MPLBACKEND on the process's own os.environ:
    ``os.environ.setdefault('MPLBACKEND', ...)`` or
    ``os.environ['MPLBACKEND'] = ...``."""
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'setdefault'
                and _environ(node.func.value)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 'MPLBACKEND'):
            found.append(node.lineno)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and _environ(t.value)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == 'MPLBACKEND'):
                    found.append(node.lineno)
    return found


def test_conftest_sets_the_backend_at_import():
    """At module level, so it is in place before any test module imports
    matplotlib; inside a fixture or helper it would come too late for the
    files that import pyplot at the top."""
    tree = ast.parse((TESTS / 'conftest.py').read_text(encoding='utf-8'))
    top = [n.lineno for n in tree.body
           if isinstance(n, ast.Expr) and _mpl_default_sites(n)]
    assert len(top) == 1, ('tests/conftest.py no longer sets MPLBACKEND at module '
                 'level exactly once; every test that imports pyplot '
                 f'depends on it (found at lines {top})')


# Run in a fresh interpreter with MPLBACKEND unset: loading conftest must
# leave matplotlib either not yet imported or already on Agg. The AST check
# above only sees that the line exists; an import above it that pulls in
# matplotlib would leave the backend unpicked and still pass that check.
_LOAD_CONFTEST = """
import os, sys
os.environ.pop('MPLBACKEND', None)
import tests.conftest
mpl = sys.modules.get('matplotlib')
backend = None if mpl is None else mpl.rcParams._get_backend_or_none()
print(os.environ.get('MPLBACKEND'), 'unloaded' if mpl is None else backend)
"""


def _load_conftest(code=_LOAD_CONFTEST):
    env = {k: v for k, v in os.environ.items() if k != 'MPLBACKEND'}
    return subprocess.run([sys.executable, '-c', code], cwd=TESTS.parent,
                          env=env, capture_output=True, text=True,
                          timeout=120)


def test_conftest_picks_the_backend_before_matplotlib_loads():
    out = _load_conftest()
    assert out.returncode == 0, out.stderr
    env_value, backend = out.stdout.split()
    assert env_value == 'Agg'
    assert backend in ('unloaded', 'agg'), (
        f'matplotlib was imported while loading tests/conftest.py with '
        f'backend {backend!r}: something above the MPLBACKEND line imports '
        'it, so the default came too late')


def test_the_load_check_sees_matplotlib_imported_too_early():
    early = _LOAD_CONFTEST.replace(
        'import tests.conftest',
        'import matplotlib.pyplot\nimport tests.conftest')
    out = _load_conftest(early)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[1] not in ('unloaded', 'agg')


def test_no_test_file_copies_the_backend_default():
    bad = []
    for path in sorted(TESTS.glob('*.py')):
        if path.name == 'conftest.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        bad += [f'{path.name}:{ln}' for ln in _mpl_default_sites(tree)]
    assert not bad, (
        f'{bad} set MPLBACKEND again; tests/conftest.py already sets it for '
        'the whole suite. Drop the line (use monkeypatch.setenv for a test '
        'that needs a different backend)')


def test_the_scan_sees_a_planted_copy():
    planted = ast.parse(
        "import os\n"
        "os.environ.setdefault('MPLBACKEND', 'Agg')\n"
        "def f():\n"
        "    os.environ.setdefault(\"MPLBACKEND\", \"Agg\")\n"
        "    os.environ['MPLBACKEND'] = 'Agg'\n"
        "def left_alone(monkeypatch):\n"
        "    monkeypatch.setenv('MPLBACKEND', 'Agg')\n"
        "    env = dict(os.environ, MPLBACKEND='Agg')\n"
        "    env['MPLBACKEND'] = 'Agg'\n"
        "    os.environ.setdefault('OTHER', 'x')\n")
    assert sorted(_mpl_default_sites(planted)) == [2, 4, 5]


def test_the_backend_is_set_while_tests_run():
    assert os.environ.get('MPLBACKEND'), (
        'MPLBACKEND is unset inside a test; conftest should have set it')
