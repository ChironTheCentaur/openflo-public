"""environment.yml stays a true mirror of requirements.txt and the dev pins.

The conda file drifted once unnoticed (an old tkinterdnd2, openpyxl missing,
ruff and pyright unpinned while CI pins them). These tests compare the three
files: every version environment.yml pins must equal the requirements.txt pin
(or, for ruff and pyright, the pyproject dev pin), every runtime dependency
and the gui extra must be listed, and the dev tools must be pinned exactly.

The helpers are checked on synthetic file text first, so each check is shown
to catch the drift it names, then run once on the real files.
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# conda-forge spells a few packages differently from PyPI.
_CONDA_TO_PYPI = {'python-igraph': 'igraph'}
# Listed in environment.yml but pinned by neither file on purpose.
_UNPINNED = {'python', 'tk', 'pip', 'pytest'}
_DEV_TOOLS = ('ruff', 'pyright')


def _norm(name):
    name = re.sub(r'\[.*\]', '', name).strip().lower().replace('_', '-')
    return _CONDA_TO_PYPI.get(name, name)


def env_specs(text):
    """{package: version or None} for each dependency line of environment.yml.

    Reads both the conda list (`- name=1.2`) and the pip list
    (`- name==1.2`); `None` means listed without an exact pin.
    """
    specs = {}
    in_deps = False
    for line in text.splitlines():
        if re.match(r'\S', line):          # a top-level key starts a section
            in_deps = line.startswith('dependencies:')
            continue
        if not in_deps:
            continue
        m = re.match(r'\s*-\s+"?([A-Za-z0-9_.\-\[\]]+)\s*([<>=!].*?)?"?\s*(#.*)?$',
                     line)
        if not m or m.group(1) == 'pip:' or line.strip().startswith('- pip:'):
            continue
        name, spec = m.group(1), (m.group(2) or '').strip()
        exact = re.fullmatch(r'==?([^,<>=!\s]+)', spec)
        specs[_norm(name)] = exact.group(1) if exact else None
    return specs


def req_pins(text):
    """{package: version} for each `name==version` line of a requirements file."""
    pins = {}
    for line in text.splitlines():
        m = re.match(r'\s*([A-Za-z0-9_.\-]+)==([^\s#;]+)', line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def drift(env_text, req_text, pyproject):
    """Every way environment.yml disagrees with the other two files."""
    env = env_specs(env_text)
    reqs = req_pins(req_text)
    project = pyproject['project']
    dev = {}
    for spec in project['optional-dependencies']['dev']:
        m = re.match(r'([A-Za-z0-9_.\-\[\]]+)==(\S+)', spec)
        if m:
            dev[_norm(m.group(1))] = m.group(2)
    problems = []
    needed = project['dependencies'] + project['optional-dependencies']['gui']
    for spec in needed:
        name = _norm(re.match(r'[A-Za-z0-9_.\-\[\]]+', spec).group(0))
        if name not in env:
            problems.append(f'{name}: missing from environment.yml')
    for tool in _DEV_TOOLS:
        if env.get(tool) is None:
            problems.append(f'{tool}: not pinned exactly in environment.yml')
    for name, version in sorted(env.items()):
        if name in _UNPINNED:
            continue
        if name in _DEV_TOOLS:
            want = dev.get(name)
            if version is not None and want is not None and version != want:
                problems.append(
                    f'{name}: {version} in environment.yml, {want} expected')
            continue
        want = reqs.get(name)
        if want is None:
            problems.append(f'{name}: not pinned in requirements.txt')
        elif version is None:
            problems.append(f'{name}: unpinned, requirements say {want}')
        elif version != want:
            problems.append(f'{name}: {version} in environment.yml, {want} expected')
    return problems


_REQS = """\
alpha-io==1.0.0
Beta_Lib==2.0.0   # comment
igraph==1.0.0
dndlib==0.6.4
"""
_PYPROJECT = {'project': {
    'dependencies': ['alpha-io==1.0.0', 'beta-lib>=1.5', 'igraph==1.0.0'],
    'optional-dependencies': {
        'gui': ['dndlib==0.6.4'],
        'dev': ['pytest>=8.0', 'pyright==1.1.410', 'ruff==0.15.19'],
    },
}}
_ENV_OK = """\
name: synthetic
channels:
  - some-channel
dependencies:
  - python>=3.11,<3.13
  - alpha-io=1.0.0
  - beta_lib=2.0.0       # trailing comment
  - python-igraph=1.0.0
  - tk
  - pytest>=8.0
  - ruff=0.15.19
  - pip
  - pip:
      - dndlib==0.6.4
      - "pyright[nodejs]==1.1.410"
"""


def test_parsers_read_conda_pip_and_requirements_lines():
    assert env_specs(_ENV_OK) == {
        'python': None, 'alpha-io': '1.0.0', 'beta-lib': '2.0.0',
        'igraph': '1.0.0', 'tk': None, 'pytest': None, 'ruff': '0.15.19',
        'pip': None, 'dndlib': '0.6.4', 'pyright': '1.1.410'}
    assert req_pins(_REQS) == {'alpha-io': '1.0.0', 'beta-lib': '2.0.0',
                               'igraph': '1.0.0', 'dndlib': '0.6.4'}


def test_matching_files_report_no_drift():
    assert drift(_ENV_OK, _REQS, _PYPROJECT) == []


def test_old_version_is_reported():
    env = _ENV_OK.replace('dndlib==0.6.4', 'dndlib==0.4.3')
    assert drift(env, _REQS, _PYPROJECT) == [
        'dndlib: 0.4.3 in environment.yml, 0.6.4 expected']


def test_missing_runtime_dependency_is_reported():
    env = _ENV_OK.replace('  - beta_lib=2.0.0       # trailing comment\n', '')
    assert drift(env, _REQS, _PYPROJECT) == [
        'beta-lib: missing from environment.yml']


def test_missing_gui_extra_is_reported():
    env = _ENV_OK.replace('      - dndlib==0.6.4\n', '')
    assert drift(env, _REQS, _PYPROJECT) == [
        'dndlib: missing from environment.yml']


def test_unpinned_dev_tools_are_reported():
    env = (_ENV_OK.replace('ruff=0.15.19', 'ruff>=0.15')
           .replace('pyright[nodejs]==1.1.410', 'pyright[nodejs]>=1.1'))
    assert drift(env, _REQS, _PYPROJECT) == [
        'ruff: not pinned exactly in environment.yml',
        'pyright: not pinned exactly in environment.yml']


def test_dev_tool_pin_must_match_the_dev_extra():
    env = _ENV_OK.replace('ruff=0.15.19', 'ruff=0.15.18')
    assert drift(env, _REQS, _PYPROJECT) == [
        'ruff: 0.15.18 in environment.yml, 0.15.19 expected']


def test_package_requirements_do_not_pin_is_reported():
    env = _ENV_OK.replace('  - tk\n', '  - tk\n  - stray=1.0\n')
    assert drift(env, _REQS, _PYPROJECT) == [
        'stray: not pinned in requirements.txt']


def test_unpinned_runtime_package_is_reported():
    env = _ENV_OK.replace('alpha-io=1.0.0', 'alpha-io')
    assert drift(env, _REQS, _PYPROJECT) == [
        'alpha-io: unpinned, requirements say 1.0.0']


def test_environment_yml_matches_requirements_and_dev_pins():
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        pyproject = tomllib.load(f)
    problems = drift((ROOT / 'environment.yml').read_text(encoding='utf-8'),
                     (ROOT / 'requirements.txt').read_text(encoding='utf-8'),
                     pyproject)
    assert problems == []
