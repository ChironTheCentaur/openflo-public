#!/usr/bin/env python
"""Run, before a push, what CI will run (lint and the privacy gate on the
private repository, the full matrix on the public mirror's ci/** branches), so
a green run here means green CI. This is the pre-push hook
(.pre-commit-config.yaml).

    python scripts/preflight.py            # everything below
    python scripts/preflight.py --quick    # base, lint, practices, type check,
                                           # privacy gate

Steps, each matching a CI job in .github/workflows/ci.yml:

  base        The tree that must pass in full is the PR MERGED WITH master:
              a release is gated on a green mirror run for the tree master
              gets. A branch that is behind origin/master is tested against
              code master never sees, so this fails until master is merged in.
  lint        ruff check .
  practices   scripts/best_practices.py (no new docs/best-practices.md
              violations; part of CI's lint job)
  typecheck   scripts/typecheck.py (Linux + Windows, CI's pins)
  privacy     scripts/publish_public.py --check, and --check-dist on a build
  tests       pytest -n auto with OPENFLO_REQUIRE_GUI=1, under Xvfb on Linux
  slow        the serial full-pipeline and clustering-determinism tests

Why the test step is strict: every GUI test skips when Tk cannot start. With
no display (a cloud container, SSH) or no tkinter, about 500 of them skip and
`pytest` still reports green, while CI runs them under Xvfb and goes red.
That is how failures in GUI tests kept reaching CI. So, as in CI, a missing
GUI is an error here, and on Linux without $DISPLAY the tests run under
xvfb-run.

The install matrix (wheel at the lowest and highest dependency versions) is
not repeated: it needs clean environments and catches packaging, not code.

Exit 0 = all green; 1 = a step failed; 2 = this machine cannot run a step as
CI does (the message says what to install).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# As a pre-push hook this runs with git's hook exports (GIT_DIR, ...). Passed
# on to the test suite they aimed a test's `git init <tmp>` at this repository
# and left core.bare=true in its config. Every git call here uses cwd=ROOT.
for _git_var in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE',
                 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_PREFIX'):
    os.environ.pop(_git_var, None)
CI_PYTHONS = ((3, 11), (3, 12))
SLOW_TESTS = ['tests/test_cli_e2e.py', 'tests/test_clustering_reproducible.py']


class CannotRun(Exception):
    """This machine cannot run a step the way CI does; exit 2."""


def _load_typecheck():
    spec = importlib.util.spec_from_file_location(
        '_typecheck', os.path.join(ROOT, 'scripts', 'typecheck.py'))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git(*args, check=True):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True, check=check)


def check_base(fetch):
    """Fail unless origin/master is already merged into HEAD."""
    if fetch:
        r = git('fetch', '--quiet', 'origin', 'master', check=False)
        if r.returncode != 0:
            print('  (could not fetch origin/master; using the local copy)')
    if git('rev-parse', '--verify', '--quiet', 'origin/master',
           check=False).returncode != 0:
        print('  no origin/master here; skipping the base check')
        return True
    if git('merge-base', '--is-ancestor', 'origin/master', 'HEAD',
           check=False).returncode == 0:
        print('  HEAD contains origin/master: CI will test this same code')
        return True
    behind = git('rev-list', '--count', 'HEAD..origin/master').stdout.strip()
    print(f'  HEAD is {behind} commit(s) behind origin/master. CI tests the PR '
          'merged with master, so tests here would not see what CI sees.\n'
          '  Merge it first:  git merge origin/master')
    return False


def warn_if_dirty():
    if git('status', '--porcelain', '--untracked-files=no').stdout.strip():
        print('note: uncommitted changes are tested here but are not in what '
              'you push.')


def test_env(py):
    """CI's test environment, importing this checkout's src/ (an editable
    install made in another worktree would otherwise win)."""
    src = os.path.join(ROOT, 'src')
    env = {**os.environ, 'MPLBACKEND': 'Agg', 'OPENFLO_REQUIRE_GUI': '1',
           'PYTHONUTF8': '1',
           'PYTHONPATH': os.pathsep.join(
               p for p in (src, os.environ.get('PYTHONPATH')) if p)}
    where = subprocess.run(
        [py, '-c', 'import openflo, tkinter; print(openflo.__file__)'],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if where.returncode != 0:
        err = where.stderr.strip().splitlines()[-1:] or ['?']
        hint = ''
        if 'tkinter' in err[0]:
            hint = ('\nInstall Tk for this Python (Debian/Ubuntu: '
                    'apt-get install python3-tk), or the GUI tests cannot run.')
        raise CannotRun(f'preflight: {py} cannot import openflo and tkinter: '
                        f'{err[0]}{hint}')
    if not os.path.realpath(where.stdout.strip()).startswith(
            os.path.realpath(src)):
        raise CannotRun(f'preflight: openflo imports from {where.stdout.strip()}, '
                        f'not from {src}.')
    return env


def display_wrapper():
    """xvfb-run on a Linux box with no display, as CI's Linux legs do."""
    if not sys.platform.startswith('linux') or os.environ.get('DISPLAY'):
        return []
    xvfb = shutil.which('xvfb-run')
    if not xvfb:
        raise CannotRun('preflight: no $DISPLAY and no xvfb-run, so the GUI '
                        'tests cannot run (CI runs them under Xvfb). Install '
                        'it (Debian/Ubuntu: apt-get install xvfb).')
    return [xvfb, '-a']


def has_module(py, name):
    return subprocess.run([py, '-c', f'import {name}'],
                          capture_output=True).returncode == 0


def run(title, cmd, env=None, quiet=False):
    print(f'\n=== {title} ===\n  $ {" ".join(cmd)}', flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                       capture_output=quiet)
    rc = r.returncode
    if quiet and rc != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], sep='\n')
    print(f'  -> {"OK" if rc == 0 else "FAIL"} ({time.perf_counter() - t0:.0f}s)',
          flush=True)
    return rc == 0


def privacy(py):
    if not run('Privacy gate (mirror export)',
               [py, 'scripts/publish_public.py', '--check']):
        return False
    if not has_module(py, 'build'):
        raise CannotRun('preflight: the privacy gate also scrubs the built '
                        f'sdist and wheel; install `build` into {py} '
                        '(pip install build).')
    with tempfile.TemporaryDirectory() as dist:
        return (run('Build sdist + wheel',
                    [py, '-m', 'build', '--outdir', dist], quiet=True)
                and run('Privacy gate (sdist + wheel)',
                        [py, 'scripts/publish_public.py', '--check-dist', dist]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true',
                    help='skip the test steps')
    ap.add_argument('--no-fetch', action='store_true',
                    help='compare with the local origin/master without fetching')
    args = ap.parse_args(argv)

    tc = _load_typecheck()
    try:
        py = tc.choose_interpreter(None)
    except tc.EnvMismatch as exc:
        print(exc, file=sys.stderr)
        return 2
    ver = subprocess.run([py, '-c', 'import sys; print(*sys.version_info[:2])'],
                         capture_output=True, text=True).stdout.split()
    if tuple(map(int, ver)) not in CI_PYTHONS:
        print(f'note: {py} is Python {".".join(ver)}; CI tests 3.11 and 3.12.')
    warn_if_dirty()

    results = []
    try:
        print('\n=== Base: is master merged in? ===', flush=True)
        if not check_base(not args.no_fetch):
            print('\nFix the base first: every later step would test code '
                  'CI will not run.', file=sys.stderr)
            return 1
        results.append(('Base (master merged in)', True))
        results.append(('Lint (ruff)', run('Lint (ruff)',
                                           [py, '-m', 'ruff', 'check', '.'])))
        results.append(('Best practices', run(
            'Best practices (no new violations)',
            [py, 'scripts/best_practices.py'])))
        results.append(('Type check', run('Type check (as CI)',
                                          [py, 'scripts/typecheck.py',
                                           '--python', py])))
        results.append(('Privacy gate', privacy(py)))
        if not args.quick:
            env = test_env(py)
            wrap = display_wrapper()
            xdist = ['-n', 'auto'] if has_module(py, 'xdist') else []
            results.append(('Tests', run(
                'Tests (GUI required, as CI)',
                [*wrap, py, '-m', 'pytest', *xdist, '-q'], env)))
            results.append(('Slow tests', run(
                'Full-pipeline + clustering determinism (serial, as CI)',
                [*wrap, py, '-m', 'pytest', *SLOW_TESTS, '-q'],
                {**env, 'OPENFLO_RUN_SLOW_TESTS': '1'})))
    except CannotRun as exc:
        print(f'\n{exc}', file=sys.stderr)
        return 2

    print('\n===== preflight summary =====')
    for title, passed in results:
        print(f'  [{"PASS" if passed else "FAIL"}] {title}')
    if all(p for _, p in results):
        print('All green: CI should agree.' if not args.quick else
              'Quick checks green (tests not run).')
        return 0
    print('FAILURES above: fix before pushing.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
