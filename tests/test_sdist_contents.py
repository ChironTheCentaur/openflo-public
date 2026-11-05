"""The sdist keeps MANIFEST.in's promise (survey T7).

MANIFEST.in says anyone holding the sdist can run `pytest` and reproduce the
golden baseline. These tests hold it to that against the exact list of files
the sdist packs: egg_info's SOURCES.txt, which `python -m build --sdist` reads
for its tarball, produced here in under a second without building anything.

* every tracked file under tests/ and scripts/ ships, unless MANIFEST.in
  excludes it by name (golden records, the independent reference and script
  assets were once left out, so 115 tests could not run from the sdist);
* nothing .gitattributes keeps off the public mirror ships either (a test
  that loads the private-only publisher once shipped without it);
* every repository file a shipped test names ships too, unless the file is
  part of the checkout itself (its CI workflows, .gitattributes, pre-commit
  config): each test that names one of those, itself or through a helper or
  constant of its module, must be marked checkout_only, which skips it in an
  unpacked sdist; so must each test that reads the git index.

The four checks against the real tree need git to know what is tracked, so
they are checkout_only themselves. The rest run anywhere, on synthetic input.
"""
import ast
import fnmatch
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest

ROOT = Path(__file__).resolve().parents[1]

# Top-level repository entries that belong to the checkout, not to the code,
# and so never ship. A shipped test that names one must be checkout_only.
# (Built, not written out, so this module does not name them itself.)
CHECKOUT_ONLY_NAMES = tuple('.' + n for n in ('github', 'gitattributes',
                                              'pre-commit-config.yaml'))
GIT_INDEX = 'ls-' + 'files'


def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def _tracked():
    return set(_git('ls-files', '-z').split('\0')) - {''}


def sdist_files(root, egg_base):
    """The repository-relative paths an sdist built from `root` would hold
    (minus the generated PKG-INFO, setup.cfg and egg-info)."""
    Path(egg_base).mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        [sys.executable, '-c', 'from setuptools import setup; setup()',
         '-q', 'egg_info', '--egg-base', str(egg_base)],
        cwd=root, capture_output=True, text=True)
    # Say why: an old setuptools rejects pyproject's metadata, and a bare
    # CalledProcessError hid that on CI.
    assert run.returncode == 0, f'egg_info failed:\n{run.stderr[-3000:]}'
    (sources,) = Path(egg_base).glob('*.egg-info/SOURCES.txt')
    return {line for line in sources.read_text(encoding='utf-8').splitlines()
            if line and '.egg-info/' not in line}


@pytest.fixture(scope='module')
def shipped(tmp_path_factory):
    return sdist_files(ROOT, tmp_path_factory.mktemp('egg'))


def manifest_excludes(text):
    """Paths MANIFEST.in drops by name with `exclude`."""
    out = set()
    for line in text.splitlines():
        words = line.split()
        if words[:1] == ['exclude']:
            out.update(words[1:])
    return out


def export_ignore_patterns(text):
    """The patterns of .gitattributes lines marked `export-ignore`."""
    return [ln.split()[0] for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith('#')
            and 'export-ignore' in ln.split()[1:]]


def export_ignored(text, tracked):
    """Tracked paths a .gitattributes `export-ignore` line covers."""
    patterns = export_ignore_patterns(text)
    return {p for p in tracked for pat in patterns
            if fnmatch.fnmatch(p, pat) or p.startswith(pat.rstrip('/') + '/')}


def unshipped_tracked(tracked, shipped, excluded):
    return sorted(p for p in tracked
                  if p.split('/', 1)[0] in ('tests', 'scripts')
                  and p not in shipped and p not in excluded)


# ── what a test names ───────────────────────────────────────────────────────

def _names_entry(text, name):
    """True when the string `text` is a path to repository entry `name`
    (`name`, `name/...`, `./name/...`), not prose that mentions it."""
    return re.match(r'(\./)?' + re.escape(name) + r'(/|$)', text) is not None


def _strings(node):
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _marked(node):
    return any('checkout_only' in ast.unparse(d) for d in node.decorator_list)


def _tests_with_reach(tree):
    """(test name, is marked, strings it reaches) for each test in a module:
    its own strings plus those of the module's helpers and constants it uses,
    followed transitively."""
    top = {}
    module_marked = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    top[t.id] = node
                    if t.id == 'pytestmark' and 'checkout_only' in ast.unparse(node):
                        module_marked = True

    def reach(node):
        seen, todo, found = set(), [node], set()
        while todo:
            n = todo.pop()
            found |= _strings(n)
            for name in {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}:
                if name in top and name not in seen:
                    seen.add(name)
                    todo.append(top[name])
        return found

    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test'):
            out.append((node.name, module_marked or _marked(node), reach(node)))
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name.startswith('test'):
                    out.append((f'{node.name}.{m.name}',
                                module_marked or _marked(node) or _marked(m),
                                reach(m) | reach(node)))
    return out


def unshipped_names(test_sources, top_names, shipped):
    """(test, what) for each test that names a repository entry the sdist
    lacks without being marked checkout_only."""
    problems = []
    for path, src in sorted(test_sources.items()):
        tree = ast.parse(src)
        strings = _strings(tree)
        for name in sorted(set(top_names) - set(CHECKOUT_ONLY_NAMES)):
            if (any(_names_entry(s, name) for s in strings)
                    and not any(f == name or f.startswith(name + '/') for f in shipped)):
                problems.append((path, name))
        for test, marked, reached in _tests_with_reach(tree):
            if marked:
                continue
            for name in CHECKOUT_ONLY_NAMES:
                if any(_names_entry(s, name) for s in reached):
                    problems.append((f'{path}::{test}', name))
            if GIT_INDEX in reached:
                problems.append((f'{path}::{test}', 'git ls-files'))
    return problems


def _shipped_tests(shipped):
    return {p: (ROOT / p).read_text(encoding='utf-8') for p in shipped
            if p.startswith('tests/') and p.endswith('.py')}


# ── the checks against the real tree ───────────────────────────────────────

@pytest.mark.checkout_only
def test_every_tracked_test_and_script_file_ships_unless_excluded(shipped):
    excluded = manifest_excludes((ROOT / 'MANIFEST.in').read_text(encoding='utf-8'))
    assert unshipped_tracked(_tracked(), shipped, excluded) == []


@pytest.mark.checkout_only
def test_nothing_kept_off_the_mirror_ships(shipped):
    attrs = (ROOT / '.gitattributes').read_text(encoding='utf-8')
    # The parse found the lines. Checked against the text, not tracked files:
    # on the public mirror (a git checkout too) the publisher is not tracked.
    assert 'scripts/publish_public.py' in export_ignore_patterns(attrs)
    assert sorted(export_ignored(attrs, _tracked()) & shipped) == []


@pytest.mark.checkout_only
def test_every_repository_file_a_shipped_test_names_ships(shipped):
    top = {p.split('/', 1)[0] for p in _tracked()}
    assert unshipped_names(_shipped_tests(shipped), top, shipped) == []


@pytest.mark.checkout_only
def test_the_golden_baseline_and_its_records_ship(shipped):
    for need in ('src/openflo/_golden.json', 'tests/test_golden_regression.py',
                 'tests/golden/population_stats_split.json',
                 'tests/reference/independent_reference.json'):
        assert need in shipped, need


# ── the helpers catch what they are for (synthetic inputs) ─────────────────

def test_the_file_helpers_flag_each_kind_of_gap():
    tracked = {'tests/test_a.py', 'tests/golden/a.json', 'scripts/s.py',
               'scripts/tool.sh', 'scripts/private.py', 'notes/x.md'}
    shipped = {'tests/test_a.py', 'scripts/s.py'}
    assert unshipped_tracked(tracked, shipped, {'scripts/private.py'}) == [
        'scripts/tool.sh', 'tests/golden/a.json']
    assert manifest_excludes('# exclude no\nexclude a b\ninclude c\nexclude d\n') == {
        'a', 'b', 'd'}
    assert export_ignored('# x export-ignore\nscripts/p.py export-ignore\n'
                          'notes export-ignore\n*.bat text eol=crlf\n',
                          tracked | {'scripts/p.py', 'a.bat'}) == {
        'scripts/p.py', 'notes/x.md'}


_SYNTHETIC_TESTS = {
    'tests/test_ok.py': (
        "def test_reads_a_shipped_file():\n"
        "    open(ROOT / 'setup.sh')\n"),
    'tests/test_missing.py': (
        "def test_reads_an_unshipped_file():\n"
        "    open(ROOT / 'environment.yml')\n"),
    'tests/test_prose.py': (
        "def test_only_mentions_names():\n"
        "    '''environment.yml and .github are only mentioned here.'''\n"),
    'tests/test_ci.py': (
        "import pytest\n"
        "CI = ROOT / '.github' / 'workflows' / 'ci.yml'\n"
        "def _jobs():\n"
        "    return CI.read_text()\n"
        "@pytest.mark.checkout_only\n"
        "def test_marked_reads_ci():\n"
        "    _jobs()\n"
        "def test_unmarked_reads_ci_through_a_helper():\n"
        "    _jobs()\n"
        "def test_unmarked_reads_nothing():\n"
        "    assert 1\n"
        "class TestCi:\n"
        "    def test_unmarked_method(self):\n"
        "        return './.github/workflows'\n"),
    'tests/test_ci_module_marked.py': (
        "import pytest\n"
        "pytestmark = pytest.mark.checkout_only\n"
        "def test_reads_ci():\n"
        "    return ROOT / '.gitattributes'\n"),
    'tests/test_index.py': (
        "def _files():\n"
        "    return run(['git', 'ls-files'])\n"
        "def test_reads_the_index():\n"
        "    _files()\n"),
}


def test_unshipped_names_flags_each_unmarked_reach_and_only_those():
    github = CHECKOUT_ONLY_NAMES[0]
    found = unshipped_names(_SYNTHETIC_TESTS,
                            {'setup.sh', 'environment.yml', *CHECKOUT_ONLY_NAMES},
                            {'setup.sh'})
    assert found == [
        ('tests/test_ci.py::test_unmarked_reads_ci_through_a_helper', github),
        ('tests/test_ci.py::TestCi.test_unmarked_method', github),
        ('tests/test_index.py::test_reads_the_index', 'git ls-files'),
        ('tests/test_missing.py', 'environment.yml')]


def test_sdist_files_lists_what_the_manifest_adds(tmp_path):
    proj = tmp_path / 'proj'
    (proj / 'pkg').mkdir(parents=True)
    (proj / 'pkg' / '__init__.py').write_text('', encoding='utf-8')
    (proj / 'data').mkdir()
    (proj / 'data' / 'kept.json').write_text('{}', encoding='utf-8')
    (proj / 'data' / 'dropped.json').write_text('{}', encoding='utf-8')
    (proj / 'pyproject.toml').write_text(
        '[build-system]\nrequires = ["setuptools"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "synthetic-pkg"\nversion = "0.1"\n'
        '[tool.setuptools]\npackages = ["pkg"]\n', encoding='utf-8')
    (proj / 'MANIFEST.in').write_text(
        'graft data\nexclude data/dropped.json\n', encoding='utf-8')
    files = sdist_files(proj, tmp_path / 'egg')
    assert 'data/kept.json' in files
    assert 'data/dropped.json' not in files
    assert 'pkg/__init__.py' in files
    assert not [f for f in files if '.egg-info/' in f]  # generated, not ours


# ── checkout_only skips outside a checkout and only there ──────────────────

def test_in_git_checkout_sees_a_git_dir_or_a_worktree_file(tmp_path):
    assert not conftest.in_git_checkout(str(tmp_path))
    (tmp_path / '.git').write_text('gitdir: elsewhere\n', encoding='utf-8')
    assert conftest.in_git_checkout(str(tmp_path))
    (tmp_path / '.git').unlink()
    (tmp_path / '.git').mkdir()
    assert conftest.in_git_checkout(str(tmp_path))


@pytest.mark.parametrize('in_checkout', [True, False])
def test_checkout_only_items_skip_only_outside_a_checkout(monkeypatch, in_checkout):
    class Item(SimpleNamespace):
        def get_closest_marker(self, name):
            return name if name in self.marks else None

        def add_marker(self, mark):
            self.added.append(mark)

    marked = Item(marks={'checkout_only'}, added=[])
    plain = Item(marks=set(), added=[])
    monkeypatch.setattr(conftest, 'in_git_checkout', lambda: in_checkout)
    conftest.pytest_collection_modifyitems(None, [marked, plain])
    assert plain.added == []
    if in_checkout:
        assert marked.added == []
    else:
        (mark,) = marked.added
        assert mark.name == 'skip' and 'sdist' in mark.kwargs['reason']


def test_this_checkout_runs_the_checkout_only_tests():
    """In the repository the marked tests run: a hook that skipped them here
    would hide these checks too while the suite stayed green."""
    if not (ROOT / '.git').exists():
        pytest.skip('an unpacked sdist, where they are meant to skip')
    added = []
    item = SimpleNamespace(get_closest_marker=lambda name: name,
                           add_marker=added.append)
    conftest.pytest_collection_modifyitems(None, [item])
    assert added == []
