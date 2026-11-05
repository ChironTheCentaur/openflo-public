"""A broad ``except Exception`` narrows or says why (survey R11, BP7).

ruff's BLE001 is on, so every ``# noqa: BLE001 - reason`` in the tree is a real
waiver. The files that held an unmarked broad handler were listed under
``[tool.ruff.lint.per-file-ignores]`` in pyproject.toml, with counts in BACKLOG
below. That list is now empty, and this test keeps it so: no file may go back
on it, and no file may gain an unmarked handler.
It also shuts the cheap ways round BLE001: a code-less or file-wide noqa
comment, and a BLE001 waiver that gives no reason. Only real comments count:
noqa text inside a string (a script run in a subprocess, say) is not a
directive ruff reads, so it waives nothing.
"""
import importlib.util
import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(importlib.util.find_spec('ruff') is None,
                                reason='ruff is in the dev extra')

# Unmarked broad handlers per file. The R11 backlog is cleared, so it is empty:
# every file is checked by BLE001, and none may be put back on the list.
BACKLOG: dict[str, int] = {}

# BLE001 waivers that give no reason after the code. The ones that predated
# BLE001 being on all have reasons now, so none is allowed.
REASONLESS_WAIVERS = 0

_UNMARKED = 'try:\n    pass\nexcept Exception:\n    pass\n'
_MARKED = 'try:\n    pass\nexcept Exception:  # noqa: BLE001 - why it is broad\n    pass\n'


def _ruff(*args, stdin=None):
    return subprocess.run([sys.executable, '-m', 'ruff', 'check', '--no-cache', *args],
                          cwd=ROOT, input=stdin, capture_output=True, text=True)


def _ignored_files():
    lint = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['ruff']['lint']
    assert 'BLE' in lint['select']
    return {f for f, codes in lint.get('per-file-ignores', {}).items() if 'BLE001' in codes}


def _python_files():
    out = subprocess.run(['git', 'ls-files', '-z', '*.py'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split('\0') if f]


_NOQA = r'#\s*' + 'no' + 'qa'  # split so this file's own patterns are not directives
_WAIVER = re.compile(_NOQA + r':\s*[A-Z0-9, ]*BLE001[A-Z0-9, ]*(.*)$')
_BLANKET = re.compile(_NOQA + r'\s*(?:#|$)')
_FILE_WIDE = re.compile(r'#\s*(?:ruff|flake8)\s*:\s*' + 'no' + 'qa', re.IGNORECASE)


def _comments(text):
    """``(line number, text)`` of each real comment in Python source `text`."""
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.start[0], tok.string


def _waiver_scan(files, root=ROOT):
    """(reasonless BLE001 waivers, blanket or file-wide noqa lines), as
    ``path:line`` lists. Only comments are read: noqa text in a string is
    not a directive."""
    reasonless, wide = [], []
    for f in files:
        for i, comment in _comments((root / f).read_text(encoding='utf-8')):
            m = _WAIVER.search(comment)
            if m and len(re.sub(r'\W', '', m.group(1))) < 4:
                reasonless.append(f'{f}:{i}')
            if _BLANKET.search(comment) or _FILE_WIDE.search(comment):
                wide.append(f'{f}:{i}')
    return reasonless, wide


def _unmarked_counts():
    """BLE001 hits per file with the backlog waiver lifted."""
    res = _ruff('--config', 'lint.per-file-ignores = {}', '--select', 'BLE001',
                '--output-format', 'json', '.')
    assert res.returncode in (0, 1), res.stderr
    return Counter(Path(d['filename']).resolve().relative_to(ROOT).as_posix()
                   for d in json.loads(res.stdout))


def test_the_backlog_in_pyproject_matches_this_list():
    assert _ignored_files() == set(BACKLOG)


def test_backlog_files_gain_no_broad_handler_and_fixed_files_leave():
    now = _unmarked_counts()
    grew = {f: (n, BACKLOG.get(f, 0)) for f, n in now.items() if n > BACKLOG.get(f, 0)}
    assert not grew, ('unmarked `except Exception` added (now, allowed); narrow it or '
                      f'add `# noqa: BLE001 - reason`: {grew}')
    done = sorted(f for f in BACKLOG if not now[f])
    assert not done, f'no unmarked handler left: drop these from the backlog: {done}'


@pytest.mark.parametrize('path, source, flagged', [
    ('src/openflo/_r11_probe.py', _UNMARKED, True),
    ('src/openflo/_r11_probe.py', _MARKED, False),
    ('tests/_r11_probe.py', _UNMARKED, True),
])
def test_ruff_flags_an_unmarked_handler_outside_the_backlog(path, source, flagged):
    res = _ruff('--stdin-filename', path, '-', stdin=source)  # the project's own rules
    assert (res.returncode == 1) == flagged, res.stdout + res.stderr
    assert ('BLE001' in res.stdout) == flagged


@pytest.mark.checkout_only
def test_no_blanket_or_file_wide_noqa_hides_a_broad_handler():
    _, wide = _waiver_scan(_python_files())
    assert not wide, ('a code-less or file-wide noqa comment silences BLE001 '
                      f'without a reason; name the code and say why: {wide}')


@pytest.mark.checkout_only
def test_reasonless_ble001_waivers_do_not_grow():
    found, _ = _waiver_scan(_python_files())
    assert len(found) <= REASONLESS_WAIVERS, (
        f'{len(found)} `# noqa: BLE001` with no reason (cap {REASONLESS_WAIVERS}); '
        f'write `# noqa: BLE001 - why it is broad`: {found}')
    assert len(found) == REASONLESS_WAIVERS, (
        f'only {len(found)} reasonless waivers left: lower REASONLESS_WAIVERS to {len(found)}')


_NQ = '# ' + 'no' + 'qa'  # split so these probes are not directives in this file
_TRY = 'try:\n    pass\nexcept Exception:  '


@pytest.mark.parametrize('source, reasonless, wide', [
    (_TRY + _NQ + ': BLE001\n    pass\n', 1, 0),
    (_TRY + _NQ + ': BLE001 - shown in a dialog\n    pass\n', 0, 0),
    ('x = 1  ' + _NQ + '\n', 0, 1),
    ('S = """\n' + _TRY + _NQ + ': BLE001\n"""\n', 0, 0),  # a string, not a comment
])
def test_the_waiver_scan_reads_comments_only(tmp_path, source, reasonless, wide):
    (tmp_path / 'probe.py').write_text(source, encoding='utf-8')
    found, blanket = _waiver_scan(['probe.py'], root=tmp_path)
    assert (len(found), len(blanket)) == (reasonless, wide), (found, blanket)
