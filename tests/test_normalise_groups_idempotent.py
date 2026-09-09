"""`_normalise_groups` must be safe to apply twice.

Three call sites carry a comment telling the next developer to "normalise
first" — the fix for a bug class that bit `--export-wsp` and then the panel
probe, where an un-normalised `samples` spec was iterated character-by-character
and resolved ZERO samples. But normalising an ALREADY-normalised group used to
silently discard per-sample FMO overrides: the first pass flattens
``[{'name': 'd', 'fmo_set': 'X'}, 'e']`` to ``['d', 'e']`` and records the
override in `sample_fmo`; a second pass sees plain strings and overwrote that
map with the group default.

No shipped path normalised twice, so this was latent rather than live — but the
failure mode is a sample gated against the WRONG FMO control, which is silent
and scientifically wrong, and the surrounding comments actively invite a fourth
call site.
"""
import pytest

from openflo.cli import _normalise_groups

CASES = [
    pytest.param(
        [{'name': 'G', 'samples': [{'name': 'd', 'fmo_set': 'X'}, 'e'],
          'fmo_set': 'DEF'}],
        id='per-sample-override'),
    pytest.param(
        [{'name': 'G', 'samples': 'a,b,c', 'fmo_set': 'F'}],
        id='comma-string'),
    pytest.param(
        [{'name': 'G', 'samples': [{'name': 'p', 'fmo_set': ''}, 'q']}],
        id='empty-overrides'),
    pytest.param(
        [{'name': 'A', 'samples': ['s1'], 'fmo_set': 'FA'},
         {'name': 'B', 'samples': [{'name': 's2', 'fmo_set': 'FB2'}],
          'fmo_set': 'FB'}],
        id='multiple-groups'),
]


@pytest.mark.parametrize('raw', CASES)
def test_normalising_twice_changes_nothing(raw):
    once = _normalise_groups(raw)
    twice = _normalise_groups(once)
    thrice = _normalise_groups(twice)
    assert once == twice == thrice, (
        'a second pass altered the groups — the most likely casualty is a '
        'per-sample FMO override reverting to the group default')


def test_a_per_sample_override_survives_a_second_pass():
    """The specific loss: sample d is gated against FMO set X, not DEF."""
    raw = [{'name': 'G', 'samples': [{'name': 'd', 'fmo_set': 'X'}, 'e'],
            'fmo_set': 'DEF'}]
    once = _normalise_groups(raw)
    assert once[0]['sample_fmo'] == {'d': 'X', 'e': 'DEF'}
    twice = _normalise_groups(once)
    assert twice[0]['sample_fmo']['d'] == 'X', (
        "sample d's FMO override was discarded — it would be gated against "
        'the wrong control set')


def test_the_group_default_still_applies_to_new_samples():
    """Idempotence must not freeze the map: a sample added after the first
    pass still picks up the group default."""
    once = _normalise_groups(
        [{'name': 'G', 'samples': [{'name': 'd', 'fmo_set': 'X'}],
          'fmo_set': 'DEF'}])
    once[0]['samples'].append('newcomer')
    again = _normalise_groups(once)
    assert again[0]['sample_fmo'] == {'d': 'X', 'newcomer': 'DEF'}


def test_a_changed_group_default_reaches_samples_without_an_override():
    """Only overrides are sticky — the default must remain live."""
    once = _normalise_groups(
        [{'name': 'G', 'samples': [{'name': 'd', 'fmo_set': 'X'}, 'e'],
          'fmo_set': 'DEF'}])
    once[0]['fmo_set'] = 'NEWDEF'
    once[0]['sample_fmo'].pop('e')          # e had no override of its own
    again = _normalise_groups(once)
    assert again[0]['sample_fmo']['e'] == 'NEWDEF'
    assert again[0]['sample_fmo']['d'] == 'X'
