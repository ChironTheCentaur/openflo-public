"""The full test matrix covers every Python the setup scripts accept, so the
"Python 3.11 to 3.14" on the download page and in the README is tested
(audit M8, 11-18: CI tested only 3.11 and 3.12)."""
import re
from pathlib import Path

import pytest

# ci.yml is not in the sdist.
pytestmark = pytest.mark.checkout_only

REPO = Path(__file__).resolve().parents[1]


def _setup_range(name):
    text = (REPO / name).read_text(encoding='utf-8')
    lo = int(re.search(r'PY_OLDEST_MINOR=(\d+)', text).group(1))
    hi = int(re.search(r'PY_NEWEST_MINOR=(\d+)', text).group(1))
    return [f'3.{m}' for m in range(lo, hi + 1)]


def _test_matrix(text):
    """The python list of the test job's matrix in ci.yml."""
    m = re.search(r'^\s+python: \[([^\]]*)\]\s*$', text, re.M)
    assert m, 'ci.yml has no python matrix line'
    return [v.strip().strip('"\'') for v in m.group(1).split(',')]


def test_the_ci_matrix_tests_every_python_setup_accepts():
    supported = _setup_range('setup.sh')
    assert supported == _setup_range('setup.bat')
    ci = (REPO / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8')
    assert _test_matrix(ci) == supported
    # and the check sees a matrix that stops short
    assert _test_matrix('        python: ["3.11", "3.12"]\n') != supported
