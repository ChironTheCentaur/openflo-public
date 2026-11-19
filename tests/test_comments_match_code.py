"""Comments and docstrings that describe code must say what the code does.

Each check here reads a comment and the thing it describes, so a comment that
drifts from its code fails a test instead of misleading the next reader.
"""
import ast
import re
import tomllib
from pathlib import Path

import numpy as np

import openflo.compare as compare

ROOT = Path(__file__).resolve().parents[1]


def _scripts():
    with open(ROOT / 'pyproject.toml', 'rb') as f:
        return tomllib.load(f)['project']['scripts']


def _usage_block(doc):
    return doc.split('Usage:', 1)[1]


def test_compare_usage_names_its_real_command():
    usage = _usage_block(compare.__doc__).strip()
    command = usage.split()[0]
    assert _scripts().get(command) == 'openflo.compare:main', command
    assert compare.__doc__.split()[0] == command


def test_compare_usage_lists_exactly_the_parser_options():
    src = Path(compare.__file__).read_text(encoding='utf-8')
    main = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    parsed = {c.args[0].value for c in ast.walk(main)
              if isinstance(c, ast.Call)
              and getattr(c.func, 'attr', '') == 'add_argument'
              and c.args[0].value.startswith('--')}
    documented = set(re.findall(r'\[(--[\w-]+)', _usage_block(compare.__doc__)))
    assert documented == parsed


def test_readme_tools_name_real_commands_or_files():
    # Only the Features list names tools; later bold names are output files.
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    features = readme.split('## Features', 1)[1].split('\n---\n', 1)[0]
    names = re.findall(r'^- \*\*`([^`]+)`\*\*', features, flags=re.M)
    assert names
    scripts = _scripts()
    for name in names:
        assert name in scripts or (ROOT / name).exists(), name


def test_ruff_ignore_note_for_e702_describes_semicolons():
    # E702 is "multiple statements on one line (semicolon)"; `import a, b`
    # on one line is E401, a different rule.
    line = next(ln for ln in (ROOT / 'pyproject.toml').read_text(
        encoding='utf-8').splitlines() if ln.strip().startswith('"E702"'))
    note = line.split('#', 1)[1]
    assert '`;`' in note
    assert 'import' not in note


def _skew(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean((x - x.mean()) ** 3) / x.std() ** 3)


def test_synthetic_fcs_docstring_skews_match_its_data(synthetic_fcs,
                                                      synthetic_channels):
    import flowio

    src = (ROOT / 'tests' / 'conftest.py').read_text(encoding='utf-8')
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == 'synthetic_fcs')
    doc = ast.get_docstring(fn)
    claims = {}
    for ln in doc.splitlines():
        m = re.match(r'\s*(FL\d-A)\s*:\s*(.*)', ln)
        if m and 'skew' in m.group(2):
            claims[m.group(1)] = m.group(2)
    assert claims

    fcs = flowio.FlowData(synthetic_fcs)
    events = np.reshape(fcs.events, (-1, fcs.channel_count))
    for ch, text in claims.items():
        s = _skew(events[:, synthetic_channels.index(ch)])
        right = 'right-skewed' in text or 'positive-skewed' in text
        left = 'left-skewed' in text or 'negative-skewed' in text
        assert right != left, text
        assert (s > 0) == right, (ch, text, s)
