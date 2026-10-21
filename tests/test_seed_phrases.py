"""The seed setting accepts a word or phrase, not just a number.

The interesting failure here is not "wrong answer" — it is "right answer
today, different answer tomorrow". ``hash("pilot")`` is randomised per
process (PEP 456), so a phrase-seed built on it looks perfectly reproducible
inside one session and quietly stops being reproducible in the next. A test
that resolves a phrase twice in ONE process cannot see that; it passes either
way. So the load-bearing test here runs a fresh interpreter, twice, with
hash randomisation deliberately turned on.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from openflo.pipeline import resolve_seed


def test_numbers_pass_through():
    assert resolve_seed(42) == 42
    assert resolve_seed('42') == 42
    assert resolve_seed('  7 ') == 7


def test_blank_falls_back_to_the_default():
    assert resolve_seed('', default=42) == 42
    assert resolve_seed('   ', default=13) == 13


def test_a_phrase_is_stable_and_in_range():
    a = resolve_seed('pilot run 3')
    b = resolve_seed('pilot run 3')
    assert a == b
    assert 0 <= a < 2 ** 32
    assert isinstance(a, int)


def test_different_phrases_give_different_seeds():
    seeds = {resolve_seed(p) for p in
             ('pilot run 3', 'pilot run 4', 'Pilot run 3', 'pilot  run 3')}
    assert len(seeds) == 4, 'phrases must not collapse onto one seed'


def test_bool_is_not_treated_as_an_int():
    # bool is a subclass of int; True must not silently mean seed=1.
    assert resolve_seed(True) == resolve_seed('True')


@pytest.mark.parametrize('phrase', ['pilot run 3', 'donor A', 'x'])
def test_phrase_seed_survives_a_new_process(phrase):
    """The regression guard: same phrase, two fresh interpreters, hash
    randomisation ON. A ``hash()``-based implementation fails here and only
    here."""
    code = ('import sys; sys.path.insert(0, "src");'
            'from openflo.pipeline import resolve_seed;'
            f'print(resolve_seed({phrase!r}))')
    runs = [subprocess.run([sys.executable, '-R', '-c', code],
                           capture_output=True, text=True, check=True,
                           cwd=str(_repo_root())).stdout.strip()
            for _ in range(2)]
    assert runs[0] == runs[1], (
        f'phrase seed changed between processes: {runs} — the digest is '
        f'probably process-dependent (hash()?)')
    assert runs[0].isdigit()


def _repo_root():
    import pathlib
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'src' / 'openflo').is_dir():
            return parent
    raise AssertionError('repo root not found')


def test_json_round_tripped_numbers_still_mean_numbers():
    """JSON has no integer type, so a seed written as 42 can come back as
    42.0. Hashing that would give a different answer from the run it is
    supposed to reproduce."""
    assert resolve_seed(42.0) == 42
    assert resolve_seed('42.0') == 42
    import numpy as np
    assert resolve_seed(np.float64(42.0)) == 42
    assert resolve_seed(np.int64(42)) == 42


def test_none_means_the_default_not_the_word_none():
    assert resolve_seed(None, default=42) == 42
    assert resolve_seed(None, default=7) == 7
    # ... but the literal word stays a phrase
    assert resolve_seed('None') != 42


def test_bytes_are_decoded_not_repr_ed():
    assert resolve_seed(b'pilot run 3') == resolve_seed('pilot run 3')


def test_a_non_integer_number_is_still_deterministic():
    a, b = resolve_seed(1.5), resolve_seed(1.5)
    assert a == b and 0 <= a < 2 ** 32
