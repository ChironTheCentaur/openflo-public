"""CONTRIBUTING.md tells the truth about the repository.

The guide drifted once already: it said code is formatted with a formatter
nothing runs, told contributors to run a tool the setup never installs, never
mentioned how tests are proved, and said every exception inherits a base
class most of them don't. Each check below ties one kind of claim in the guide
to the file that decides it, so the next drift fails here instead.
"""
import ast
import builtins
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'openflo'


def guide() -> str:
    return (ROOT / 'CONTRIBUTING.md').read_text(encoding='utf-8')


def section(text: str, heading: str) -> str:
    """The body of ``heading`` (any level) up to the next heading of the same
    or a higher level."""
    level = len(heading) - len(heading.lstrip('#'))
    match = re.search(rf'^{re.escape(heading)}\n(.*?)(?=^#{{1,{level}}} |\Z)',
                      text, re.M | re.S)
    assert match, f'CONTRIBUTING.md has no "{heading}" section'
    return match.group(1)


def code_lines(text: str) -> list[str]:
    """Every line inside a fenced code block, comments stripped."""
    lines = []
    for block in re.findall(r'^```[^\n]*\n(.*?)^```', text, re.M | re.S):
        for line in block.splitlines():
            line = line.split('#', 1)[0].strip()
            if line:
                lines.append(line)
    return lines


# ── paths ─────────────────────────────────────────────────────────────────

_PATH = re.compile(r'(?<![\w/.-])((?:src|scripts|tests|docs)/[\w./-]*\w'
                   r'|\.pre-commit-config\.yaml|CHANGELOG\.md)')


def kept_private() -> set[str]:
    """Paths .gitattributes keeps off the public mirror (export-ignore).
    The guide may name them; they exist only in the private checkout."""
    attrs = (ROOT / '.gitattributes').read_text(encoding='utf-8')
    return {ln.split()[0] for ln in attrs.splitlines()
            if 'export-ignore' in ln.split()[1:] and not ln.lstrip().startswith('#')}


def absent_from_checkout(path: str) -> bool:
    """True when `path` is absent and not one the mirror leaves out."""
    return not (ROOT / path).exists() and path not in kept_private()


# The guide describes the repository (docs/, the pre-commit config), which the
# sdist does not ship, so the checks of what it names run in a checkout only.
@pytest.mark.checkout_only
def test_every_repo_path_the_guide_names_exists():
    named = sorted(set(_PATH.findall(guide())))
    assert named, 'the path pattern found nothing: it no longer reads the guide'
    gone = [p for p in named if absent_from_checkout(p)]
    assert gone == [], f'CONTRIBUTING.md names files that do not exist: {gone}'


# ── commands ──────────────────────────────────────────────────────────────

def _script_accepts(source: str, word: str) -> bool:
    if word.startswith('--'):
        return re.search(rf'''add_argument\(\s*['"]{re.escape(word)}['"]''',
                         source) is not None
    # a subcommand: prove_tests.py dispatches on ``argv[:1] == ['aimed']``
    return re.search(rf'''\[\s*['"]{re.escape(word)}['"]\s*\]''', source) is not None


@pytest.mark.checkout_only
def test_every_script_command_the_guide_gives_is_accepted():
    """``python scripts/x.py sub --flag`` in a code block: the script exists
    and takes that subcommand and those flags. A private-only script is
    checked where it exists (the private checkout), not on the mirror."""
    given = checked = 0
    for line in code_lines(guide()):
        m = re.match(r'python (scripts/\w+\.py)(.*)', line)
        if not m:
            continue
        given += 1
        script = ROOT / m.group(1)
        assert not absent_from_checkout(m.group(1)), f'no such script: {m.group(1)}'
        if not script.exists():
            continue
        source = script.read_text(encoding='utf-8')
        words = m.group(2).split()
        for i, word in enumerate(words):
            is_flag = word.startswith('--')
            is_sub = i == 0 and re.fullmatch(r'[a-z]+', word) is not None
            if is_flag or is_sub:
                assert _script_accepts(source, word), \
                    f'{m.group(1)} does not take {word!r} (CONTRIBUTING: {line!r})'
                checked += 1
    assert given >= 5, 'expected the guide to give several script commands'
    assert checked >= 2, 'the commands found were not checked'


def dev_extra() -> set[str]:
    data = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    reqs = data['project']['optional-dependencies']['dev']
    return {re.split(r'[<>=!~\[ ;]', r, maxsplit=1)[0].lower() for r in reqs}


def test_dev_setup_installs_every_tool_it_runs():
    """A command in "Dev setup" (other than python, pip and the venv's own
    activate script) is a tool the same block installs, or one the dev extra
    brings."""
    lines = code_lines(section(guide(), '## Dev setup'))
    installed = set(dev_extra())
    for line in lines:
        m = re.match(r'pip install (?!-)(.*)', line)
        if m:
            installed |= {w.lower() for w in m.group(1).split()
                          if not w.startswith(('-', '"', '.'))}
    run = {line.split()[0] for line in lines}
    tools = {t for t in run if t not in {'python', 'pip'}
             and not t.startswith('.')}
    assert tools, 'the setup block runs no tool: this check reads nothing'
    missing = sorted(t for t in tools if t.lower() not in installed)
    assert missing == [], f'Dev setup runs tools it never installs: {missing}'


# ── formatting ────────────────────────────────────────────────────────────

def enforced_tools() -> str:
    """What actually runs on code: CI's workflow steps and the hook ids of
    .pre-commit-config.yaml (not its comments)."""
    ci = (ROOT / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8')
    runs = ' '.join(re.findall(r'^\s*run:\s*(.*)$', ci, re.M))
    hooks = (ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
    ids = ' '.join(re.findall(r'^\s*-\s*id:\s*(\S+)', hooks, re.M))
    return runs + ' ' + ids


@pytest.mark.checkout_only
def test_the_guide_claims_no_formatter_that_nothing_runs():
    style = section(guide(), '## Code style')
    claimed = re.findall(r'(?:formatted|formatting|PEP 8)\s+(?:via|with|by)\s+`([^`]+)`',
                         style, re.I)
    enforced = enforced_tools()
    for tool in claimed:
        hook_id = tool.replace(' ', '-')
        assert tool in enforced or hook_id in enforced, \
            f'CONTRIBUTING says code is formatted with `{tool}`, but neither CI nor a hook runs it'
    assert 'ruff check' in enforced, 'CI no longer runs ruff check'
    assert '`ruff check`' in style, 'the Code style section should name the linter CI runs'


# ── proving tests ─────────────────────────────────────────────────────────

def test_the_guide_explains_how_tests_are_proved():
    tests = section(guide(), '## Tests')
    for needle in ('`scripts/prove_tests.py`', '`docs/test-assurance/aimed.json`',
                   'generated data', 'fail'):
        assert needle in tests, f'the Tests section should mention {needle}'


# ── exceptions ────────────────────────────────────────────────────────────

_BUILTIN_EXC = {n for n, v in vars(builtins).items()
                if isinstance(v, type) and issubclass(v, BaseException)}


def exception_classes() -> dict[str, list[str]]:
    """Every exception class defined in the package: name -> base names."""
    bases: dict[str, list[str]] = {}
    for path in sorted(PACKAGE.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                names = [b.id if isinstance(b, ast.Name) else
                         b.attr if isinstance(b, ast.Attribute) else ''
                         for b in node.bases]
                bases[node.name] = names

    def is_exc(name, seen=()):
        if name in _BUILTIN_EXC:
            return True
        if name in seen or name not in bases:
            return False
        return any(is_exc(b, (*seen, name)) for b in bases[name])

    return {n: b for n, b in bases.items() if is_exc(n)}


def derives_from(name: str, root: str, classes: dict[str, list[str]]) -> bool:
    if name == root:
        return True
    return any(derives_from(b, root, classes) for b in classes.get(name, []))


def test_every_exception_class_is_in_a_documented_group():
    classes = exception_classes()
    assert 'OpenFloError' in classes and 'FcsParseError' in classes, \
        'the scan found no OpenFlo exceptions: it no longer reads the package'
    text = section(guide(), '### Exceptions')
    named = set(re.findall(r'`(\w+)`', text))
    outside = sorted(n for n in classes if not derives_from(n, 'OpenFloError', classes))
    undocumented = [n for n in outside if n not in named]
    assert undocumented == [], (
        'exceptions outside the OpenFloError tree that CONTRIBUTING '
        f'"Exceptions" does not name: {undocumented}')


def test_every_exception_the_guide_names_exists_with_that_base():
    classes = exception_classes()
    text = section(guide(), '### Exceptions')
    named = {n for n in re.findall(r'`(\w+)`', text) if n[0].isupper()}
    gone = sorted(named - classes.keys() - _BUILTIN_EXC)
    assert gone == [], f'CONTRIBUTING "Exceptions" names no such class: {gone}'
    # Each bullet's classes really have that bullet's base.
    for bullet in re.split(r'^- ', text, flags=re.M)[1:]:
        if bullet.startswith('**A failed pipeline step**'):
            root = 'OpenFloError'
        elif bullet.startswith('**Bad input'):
            root = 'ValueError'
        elif bullet.startswith('**A cancelled'):
            root = 'Exception'
        else:
            continue
        for name in re.findall(r'`([A-Z]\w+)`', bullet):
            if name in classes and name != root:
                assert derives_from(name, root, classes), \
                    f'CONTRIBUTING files {name} under {root}, but it does not subclass it'
