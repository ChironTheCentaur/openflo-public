"""One answer to "is this the acquisition-time channel?" (best-practices N1).

`channel_names.is_time_channel` is the only definition; the modules that
used to import pipeline's private regex, or keep their own copy of it,
now ask it, so they agree on every generated name below.
"""

from __future__ import annotations

import ast
import itertools
import random
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from openflo import bead_normalize, channel_roles, kinetics, saturation
from openflo.channel_names import TIME_PATTERN, is_time_channel

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'

TIME_NAMES = ['Time', 'TIME', 'time', 'Time (s)', 'TIMESTAMP', 'HDR-T Time',
              'time_ms', '(Time)']
OTHER_NAMES = ['FL1-A', 'FL2-H', 'SSC-A', 'FSC-W', 'LIFETIME', 'ACQTIME',
               'FL3TIME', 'Tim', 'Event_length', '141Pr', '']


def _generated_names(seed=7, n=300):
    """Random channel-like names, some holding TIME at random places."""
    rng = random.Random(seed)
    alphabet = 'ABCFHLSTW0123 -_()/'
    out = []
    for _ in range(n):
        left = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
        right = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
        word = rng.choice(['time', 'TIME', 'Time', 'tim', 'tome', 'FL'])
        out.append(left + word + right)
    return out


@pytest.mark.parametrize('name', TIME_NAMES)
def test_time_names_are_time_channels(name):
    assert is_time_channel(name)


@pytest.mark.parametrize('name', OTHER_NAMES)
def test_other_names_are_not_time_channels(name):
    assert not is_time_channel(name)


def test_non_string_names_are_read_as_text():
    assert not is_time_channel(3)
    assert not is_time_channel(None)


def test_generated_names_follow_the_rule():
    """TIME counts when no letter or digit sits right before it."""
    for name in _generated_names():
        u = name.upper()
        expected = any(u[i:i + 4] == 'TIME'
                       and (i == 0 or not u[i - 1].isalnum())
                       for i in range(len(u)))
        assert is_time_channel(name) is expected, name


def test_old_pipeline_names_are_the_same_objects():
    assert channel_roles._is_time is is_time_channel
    assert channel_roles._TIME_RE is TIME_PATTERN


def test_every_module_agrees_on_generated_columns():
    names = TIME_NAMES + OTHER_NAMES + _generated_names(seed=11, n=120)
    rng = random.Random(3)
    for cols in itertools.islice(
            (rng.sample(names, 6) for _ in itertools.count()), 200):
        want_first = next((c for c in cols if is_time_channel(c)), None)
        assert bead_normalize.find_time_channel(cols) == want_first
        assert bead_normalize.mass_channels(cols) == [
            c for c in cols if not is_time_channel(c)
            and not bead_normalize.is_gaussian_parameter(c)]
        sample = SimpleNamespace(
            data=pd.DataFrame({c: [0.0] for c in dict.fromkeys(cols)}),
            scatter_channels=[], fluor_channels=list(dict.fromkeys(cols)))
        assert kinetics.time_column(sample) == (
            next((c for c in sample.data.columns if is_time_channel(c)), None))
        assert saturation.report_channels(sample) == [
            c for c in dict.fromkeys(cols) if not is_time_channel(c)]
        for c in cols:
            assert channel_roles.is_excluded(c) == (
                any(k in c for k in channel_roles.EXCLUDE_CLUSTER)
                or is_time_channel(c))


def _time_regex_sites():
    """(file, line) of every re.compile of an upper-case TIME pattern
    under src/ (channel names are matched upper-cased)."""
    hits = []
    for path in sorted(SRC.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'compile'
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and 'TIME' in node.args[0].value):
                hits.append((path.name, node.lineno))
    return hits


def test_the_time_pattern_is_compiled_in_one_place():
    assert [f for f, _ in _time_regex_sites()] == ['channel_names.py']


def test_no_module_imports_pipelines_private_time_check():
    for path in sorted(SRC.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
                assert not names & {'_is_time', '_TIME_RE'}, path.name
