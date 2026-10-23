"""Type-check exactly as CI does: the same pyright, the same packages, the same
interpreter, on the Linux AND Windows platforms. CI runs this script too
(.github/workflows/typecheck.yml), so there is one definition of "clean".

Why this exists: pyright's verdict depends on more than the source. It depends
on which interpreter's site-packages it reads (an import it cannot resolve
becomes Unknown, and Unknown SUPPRESSES errors), on which versions of the
dependencies are installed there (their type hints change), and on the pyright
version. Any of those drifting gives "clean locally, red in CI" -- a recurring,
expensive loop (a 2.6.1 release went out with such an error). The worst case
is silent: when [tool.pyright]'s `.venv` is missing, as in a git worktree or a
fresh clone, pyright quietly falls back to whatever `python` is on PATH.

So before type-checking, this script PROVES the environment matches CI and
exits 2 if it does not:

  1. the interpreter is chosen explicitly: --python, else the repo's .venv,
     else (in a git worktree) the main checkout's .venv;
  2. every pin in requirements.txt and requirements-typecheck.txt is
     installed in it at exactly that version;
  3. pyright is the version pinned in pyproject's dev extra;
  4. pyright's own report of its search paths names that interpreter's
     site-packages and no other, and it loaded this repo's pyproject.toml.

Then it fails on any error OR warning: a correct run is "0 errors, 0
warnings", and an "Import X could not be resolved" warning means the run is
weaker than it claims.

Usage:  python scripts/typecheck.py                  (the pre-push hook)
        python scripts/typecheck.py --python current (CI: this interpreter)
Exit 0 = both platforms clean; 1 = errors/warnings; 2 = cannot run like CI.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENT_FILES = ('requirements.txt', 'requirements-typecheck.txt')
PLATFORMS = ('Linux', 'Windows')
FIX_ENV = ('pip install -r requirements.txt && '
           'pip install --no-deps -r requirements-typecheck.txt && '
           'pip install -e ".[dev]"')


class EnvMismatch(Exception):
    """The environment would not give CI's answer; exit 2 with this message."""


def norm_name(name):
    """PEP 503 normalised distribution name."""
    return re.sub(r'[-_.]+', '-', name).lower()


def same_path(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def parse_pins(text):
    """{normalised name: version} for every `name==version` line."""
    pins = {}
    for line in text.splitlines():
        line = line.split('#', 1)[0].strip()
        m = re.match(r'^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*==\s*([^\s;]+)', line)
        if m:
            pins[norm_name(m.group(1))] = m.group(3)
    return pins


def pyright_pin():
    """The pyright version pinned in pyproject's dev extra."""
    with open(os.path.join(ROOT, 'pyproject.toml'), encoding='utf-8') as f:
        m = re.search(r'"pyright==([^"\s;]+)"', f.read())
    if not m:
        raise EnvMismatch('typecheck: no "pyright==X" pin in pyproject.toml\'s dev extra.')
    return m.group(1)


def main_checkout_root():
    """The main working tree when ROOT is a git worktree, else None."""
    try:
        r = subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                           cwd=ROOT, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    main = os.path.dirname(r.stdout.strip())
    return None if same_path(main, ROOT) else main


def venv_python(root):
    for rel in (('.venv', 'Scripts', 'python.exe'), ('.venv', 'bin', 'python')):
        p = os.path.join(root, *rel)
        if os.path.isfile(p):
            return p
    return None


def choose_interpreter(arg):
    if arg == 'current':
        return sys.executable
    if arg:
        if not os.path.isfile(arg):
            raise EnvMismatch(f'typecheck: --python {arg} does not exist.')
        return arg
    py = venv_python(ROOT)
    if py:
        return py
    main = main_checkout_root()
    py = main and venv_python(main)
    if py:
        print(f'typecheck: worktree without its own .venv; using the main checkout\'s ({py})')
        return py
    raise EnvMismatch(
        "typecheck: no .venv in the repo root (or the main checkout), so there is no "
        "interpreter that has the project's dependencies the way CI does. Create one:\n"
        f'  python -m venv .venv  then  {FIX_ENV}')


_PROBE = r'''
import json, sys, sysconfig
from importlib import metadata
dists = {}
for d in metadata.distributions():
    name = d.metadata["Name"]
    if name:
        dists[name] = d.version
paths = sysconfig.get_paths()
print(json.dumps({"dists": dists, "site": [paths["purelib"], paths["platlib"]],
                  "version": "%d.%d.%d" % sys.version_info[:3]}))
'''


def probe(py):
    r = subprocess.run([py, '-c', _PROBE], capture_output=True, text=True, encoding='utf-8')
    if r.returncode:
        raise EnvMismatch(f'typecheck: could not run {py}:\n{r.stderr[-2000:]}')
    info = json.loads(r.stdout)
    info['dists'] = {norm_name(k): v for k, v in info['dists'].items()}
    return info


def pin_mismatches(pins, installed):
    """['name: want X, have Y'] for each pin the environment does not satisfy."""
    out = []
    for name, want in sorted(pins.items()):
        have = installed.get(name)
        if have != want:
            out.append(f'{name}: want {want}, have {have or "nothing"}')
    return out


def parse_search_paths(verbose_output):
    """The paths pyright --verbose lists under 'Search paths:'."""
    paths, inside = [], False
    for line in verbose_output.splitlines():
        if line.strip() == 'Search paths:':
            inside = True
            continue
        if inside:
            if not line.startswith('    '):
                inside = False
                continue
            paths.append(line.strip())
    return paths


def foreign_site_dirs(search_paths, expected_site):
    """Third-party package dirs pyright searches that are NOT the interpreter's.

    pyright's bundled typeshed lives inside site-packages too
    ('.../site-packages/pyright/dist/typeshed-fallback'); that is not an
    environment, so it is skipped."""
    found = [p for p in search_paths
             if re.search(r'[\\/](site|dist)-packages$', p.rstrip('\\/'), re.I)
             and 'typeshed-fallback' not in p]
    return [p for p in found if not any(same_path(p, s) for s in expected_site)], found


def run_pyright(py, *args):
    env = dict(os.environ)
    # pyright-python honours these and would silently swap the pinned version.
    env.pop('PYRIGHT_PYTHON_FORCE_VERSION', None)
    env.pop('PYRIGHT_PYTHON_PYLANCE_VERSION', None)
    env['PYRIGHT_PYTHON_IGNORE_WARNINGS'] = '1'
    return subprocess.run([py, '-m', 'pyright', '--pythonpath', py, *args],
                          cwd=ROOT, capture_output=True, text=True, encoding='utf-8', env=env)


def check_environment(py):
    if os.path.exists(os.path.join(ROOT, 'pyrightconfig.json')):
        raise EnvMismatch(
            'typecheck: pyrightconfig.json exists. pyright reads it INSTEAD of '
            '[tool.pyright] in pyproject.toml, so this run would not use the '
            'project settings CI uses. Delete it.')

    info = probe(py)
    pins = {}
    for fname in REQUIREMENT_FILES:
        with open(os.path.join(ROOT, fname), encoding='utf-8') as f:
            pins.update(parse_pins(f.read()))
    want_pyright = pyright_pin()
    pins['pyright'] = want_pyright
    bad = pin_mismatches(pins, info['dists'])
    if bad:
        raise EnvMismatch(
            f'typecheck: {py} does not have the packages CI type-checks against, so '
            'pyright would see different types than CI:\n  ' + '\n  '.join(bad)
            + f'\nFix:  {FIX_ENV}')

    # Ask pyright itself where it looks. This is the check that catches a
    # silent fallback to another interpreter, whatever the reason for it.
    r = run_pyright(py, '--verbose', os.path.join('src', 'openflo', '__init__.py'))
    out = r.stdout + r.stderr
    m = re.search(r'Loading pyproject\.toml file at (.+)', out)
    if not m or not same_path(m.group(1).strip(), os.path.join(ROOT, 'pyproject.toml')):
        raise EnvMismatch('typecheck: pyright did not load this repo\'s pyproject.toml:\n'
                          + out[-2000:])
    foreign, found = foreign_site_dirs(parse_search_paths(out), info['site'])
    if foreign or not found:
        raise EnvMismatch(
            f'typecheck: pyright is not resolving imports from {py}.\n'
            f'  expected: {info["site"][0]}\n  pyright searched: {found or "no site-packages at all"}\n'
            'If a .venv in the repo root is a different environment, pyproject\'s '
            '[tool.pyright] venv setting makes pyright use it regardless of --python.')
    print(f'typecheck: python {info["version"]} at {py}; pyright {want_pyright}; '
          f'{len(pins)} pinned packages match')
    return want_pyright


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--python', help="interpreter whose packages pyright resolves "
                    "('current' = the one running this script; default: the repo's .venv)")
    args = ap.parse_args(argv)
    try:
        py = choose_interpreter(args.python)
        want_pyright = check_environment(py)
    except EnvMismatch as e:
        print(e)
        return 2

    bad = 0
    for platform in PLATFORMS:
        r = run_pyright(py, '--pythonplatform', platform, '--outputjson')
        try:
            report = json.loads(r.stdout)
        except ValueError:
            print(f'typecheck: pyright did not run ({platform}):\n{r.stdout[-2000:]}{r.stderr[-2000:]}')
            return 2
        if report.get('version') != want_pyright:
            print(f"typecheck: pyright reported version {report.get('version')}, "
                  f'not the pinned {want_pyright}.')
            return 2
        diags = [d for d in report.get('generalDiagnostics', [])
                 if d.get('severity') in ('error', 'warning')]
        s = report.get('summary', {})
        print(f"typecheck {platform:7}: {s.get('errorCount', '?')} errors, "
              f"{s.get('warningCount', '?')} warnings")
        for d in diags:
            rng = d.get('range', {}).get('start', {})
            print(f"  {os.path.relpath(d.get('file', '?'), ROOT)}:{rng.get('line', 0) + 1}: "
                  f"{d.get('severity')}: {d.get('message', '').splitlines()[0]}"
                  + (f" ({d['rule']})" if d.get('rule') else ''))
        if any(d.get('rule') == 'reportMissingImports' for d in diags):
            print('  -> unresolved imports: this run is WEAKER than it claims (Unknown types '
                  'hide errors). A new import needs its package pinned in requirements.txt '
                  'or requirements-typecheck.txt.')
        bad += len(diags)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
