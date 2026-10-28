"""Local pyright must see what CI's pyright sees.

CI's pyright runs in the interpreter that has every dependency installed, so
it resolves flowio, flowutils, igraph... Locally that only happens through the
venvPath/venv settings in [tool.pyright], and pyright IGNORES that block when a
pyrightconfig.json exists. One did, from the first release until 2.6.1: local
runs could not resolve the dependencies, typed every call into them Unknown
(which suppresses errors), and reported clean code CI then failed -- more than
once, each time a red CI matrix and a fix-up push. scripts/typecheck.py is the
gate, run by the pre-push hook AND by CI; these tests keep it in place.
"""
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from tests.test_release_mirror_gate import _gate_jobs, _jobs, _needs

ROOT = Path(__file__).resolve().parents[1]


def test_no_pyrightconfig_shadows_pyproject():
    assert not (ROOT / 'pyrightconfig.json').exists(), (
        'pyrightconfig.json makes pyright ignore [tool.pyright] in pyproject.toml '
        '(and its venv setting): local type checks become weaker than CI')


def test_pyright_resolves_imports_from_the_project_venv():
    cfg = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    pyright = cfg['tool']['pyright']
    assert (pyright.get('venvPath'), pyright.get('venv')) == ('.', '.venv')
    assert pyright.get('include') == ['src']


def test_pyright_runs_before_every_push():
    hooks = (ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
    assert 'scripts/typecheck.py' in hooks and 'stages: [pre-push]' in hooks
    assert 'default_install_hook_types: [pre-commit, pre-push]' in hooks
    assert (ROOT / 'scripts' / 'typecheck.py').is_file()


# ── scripts/typecheck.py: CI runs the same gate, against the same packages ──

_spec = importlib.util.spec_from_file_location('typecheck', ROOT / 'scripts' / 'typecheck.py')
assert _spec and _spec.loader
typecheck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(typecheck)


def test_ci_and_release_run_the_same_script_as_the_pre_push_hook():
    wf = ROOT / '.github' / 'workflows'
    gate = (wf / 'typecheck.yml').read_text(encoding='utf-8')
    assert 'python scripts/typecheck.py --python current' in gate
    assert 'pip install --no-deps -r requirements-typecheck.txt' in gate
    assert 'pip install -r requirements.txt' in gate
    assert 'uses: ./.github/workflows/typecheck.yml' in (
        wf / 'ci.yml').read_text(encoding='utf-8')
    # release.yml reaches the same gate through the whole of ci.yml (on the
    # mirror), or through the mirror's ci.yml result for the same tree.
    assert 'uses: ./.github/workflows/ci.yml' in (
        wf / 'release.yml').read_text(encoding='utf-8')
    # Bare `pyright` anywhere else in CI is the weak check this replaced.
    for path in wf.glob('*.yml'):
        for line in path.read_text(encoding='utf-8').splitlines():
            assert not re.match(r'\s*(- )?run:\s*pyright\b', line), f'{path.name}: {line}'


def _release_gates(text):
    """The full-CI gates release.yml's build job needs: 'ci.yml' (a job that
    calls ci.yml) and 'mirror' (a job that asks the mirror for ci.yml's result
    on the same tree, publish_public.py --ci-status)."""
    jobs = _jobs(text)
    needs = set(_needs(jobs['build'])) if 'build' in jobs else set()
    ci, mirror = _gate_jobs(text)
    return {kind for kind, job in (('ci.yml', ci), ('mirror', mirror))
            if job and job in needs}


def test_release_builds_only_after_the_full_ci():
    # The tests, not only the type check: a tag must not reach PyPI from a
    # tree the suite never passed on. On the mirror build waits for ci.yml
    # itself; on the private repository for the mirror's ci.yml run on the
    # same tree. One of the two is skipped in each repository, so build
    # needs both (tests/test_release_mirror_gate.py checks its if:).
    wf = ROOT / '.github' / 'workflows'
    assert _release_gates((wf / 'release.yml').read_text(encoding='utf-8')) == {
        'ci.yml', 'mirror'}
    # ci.yml can only be called that way if it declares workflow_call.
    on = (wf / 'ci.yml').read_text(encoding='utf-8').split('\njobs:', 1)[0]
    assert re.search(r'^  workflow_call:', on, re.M)


def test_the_release_gate_check_rejects_a_build_without_the_full_ci():
    # Negative half: the old wiring (build after the type check alone) has
    # neither gate; dropping the mirror check leaves a private release with
    # nothing that ran.
    old = ('on: push\njobs:\n  typecheck:\n    uses: ./.github/workflows/typecheck.yml\n\n'
           '  build:\n    name: Build\n    needs: typecheck\n\n  publish:\n'
           '    needs: build\n')
    assert _release_gates(old) == set()
    text = (ROOT / '.github' / 'workflows' / 'release.yml').read_text(encoding='utf-8')
    only_ci = text.replace('needs: [ci, mirror-ci]', 'needs: [ci]')
    assert only_ci != text and _release_gates(only_ci) == {'ci.yml'}


def test_one_pyright_pin():
    """The dev extra is the only pyright version; nothing else may name one."""
    pin = typecheck.pyright_pin()
    for path in [*(ROOT / '.github' / 'workflows').glob('*.yml'), ROOT / 'requirements.txt',
                 ROOT / 'requirements-typecheck.txt']:
        for v in re.findall(r'pyright(?:\[[^\]]*\])?==([\w.]+)', path.read_text(encoding='utf-8')):
            assert v == pin, f'{path.name} pins pyright {v}, pyproject pins {pin}'


def test_typecheck_pins_are_exact_and_distinct_from_runtime_pins():
    runtime = typecheck.parse_pins((ROOT / 'requirements.txt').read_text(encoding='utf-8'))
    extra_text = (ROOT / 'requirements-typecheck.txt').read_text(encoding='utf-8')
    extra = typecheck.parse_pins(extra_text)
    lines = [ln.split('#')[0].strip() for ln in extra_text.splitlines()]
    assert len(extra) == len([ln for ln in lines if ln]), 'every line must be name==version'
    assert {'anndata', 'cupy-cuda12x', 'phate', 'trimap', 'pacmap'} <= set(extra)
    assert not set(extra) & set(runtime)


def test_parse_pins():
    assert typecheck.parse_pins(
        'FlowIO==1.4.0\n# c\nnumpy==2.4.6   # note\npyright[nodejs]==1.1.410\n'
        'scikit_learn==1.8.0\nfoo>=1\n'
    ) == {'flowio': '1.4.0', 'numpy': '2.4.6', 'pyright': '1.1.410', 'scikit-learn': '1.8.0'}


def test_pin_mismatches():
    assert typecheck.pin_mismatches({'numpy': '2.4.6', 'phate': '2.0.0'},
                                    {'numpy': '2.4.6', 'phate': '1.0.11'}) == [
        'phate: want 2.0.0, have 1.0.11']
    assert typecheck.pin_mismatches({'anndata': '0.12.18'}, {}) == [
        'anndata: want 0.12.18, have nothing']


_VERBOSE = '''\
Loading pyproject.toml file at {root}/pyproject.toml
venv .venv subdirectory not found in venv path {root}.
Execution environment: python
  Python version: 3.11
  Search paths:
    {pyright}/typeshed-fallback/stdlib
    {root}/src
    {pyright}/typeshed-fallback/stubs/...
    /usr/lib/python3.12
    {site}
Found 1 source file
'''


def test_search_paths_must_name_the_chosen_interpreter(tmp_path):
    good = tmp_path / 'venv' / 'lib' / 'site-packages'
    other = tmp_path / 'conda' / 'Lib' / 'site-packages'
    pyright = good / 'pyright' / 'dist'
    for d in (good, other):
        d.mkdir(parents=True)
    out = _VERBOSE.format(root=tmp_path, pyright=pyright, site=good)
    assert typecheck.parse_search_paths(out)[-1] == str(good)
    foreign, found = typecheck.foreign_site_dirs(typecheck.parse_search_paths(out), [str(good)])
    assert (foreign, found) == ([], [str(good)])     # bundled typeshed is not an env

    # The silent fallback: pyright read some other interpreter's packages.
    out = _VERBOSE.format(root=tmp_path, pyright=pyright, site=other)
    foreign, _ = typecheck.foreign_site_dirs(typecheck.parse_search_paths(out), [str(good)])
    assert foreign == [str(other)]

    # No site-packages at all: nothing resolved, every import Unknown.
    out = _VERBOSE.format(root=tmp_path, pyright=pyright, site='/usr/lib/python3.12/lib-dynload')
    assert typecheck.foreign_site_dirs(typecheck.parse_search_paths(out), [str(good)])[1] == []


def test_choose_interpreter_rejects_a_missing_python(tmp_path):
    with pytest.raises(typecheck.EnvMismatch):
        typecheck.choose_interpreter(str(tmp_path / 'nope'))
