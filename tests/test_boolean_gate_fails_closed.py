"""A boolean gate must resolve its operands, and refuse events when it cannot.

`apply_region_gates` called `gate_to_mask` without the gate dictionary, so a
`boolean` gate could never resolve the ids in its `operands` list. It took the
"unresolved operands" branch, which returned all-True — and for a NOT gate that
is the worst possible answer: the complement population became the ENTIRE
sample. Measured, `NOT (CD3 > 2.0)` on 20,000 events kept all 20,000 where the
correct answer is 10,004, so every CD3-positive event was admitted into the
CD3-negative population.

Two changes. `apply_region_gates` has the whole gate list in hand and now
passes it down, so operands resolve — including operands that are disabled,
since a disabled gate still defines the population it is an operand of.

And an unresolvable boolean gate now fails CLOSED. That matches the
'cluster'/'category' branches beside it, and the fail-closed exception handler
in `apply_region_gates` whose comment already made the argument: a gate that
errors must not pass every event through, because an empty population is loud
while a superset reads as success.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, cumulative_gate_mask, gate_to_mask

N = 20_000


@pytest.fixture
def bimodal():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'CD3': np.concatenate([rng.normal(0.5, 0.4, N // 2),
                               rng.normal(4.0, 0.6, N // 2)]),
        'FSC-A': rng.normal(60_000.0, 8_000.0, N)})


def _positive(enabled=True):
    return {'id': 'g1', 'kind': 'threshold', 'channel': 'CD3', 'value': 2.0,
            'parent_id': None, 'enabled': enabled}


def _negative():
    return {'id': 'g2', 'kind': 'boolean', 'op': 'not', 'operands': ['g1'],
            'parent_id': None, 'enabled': True}


def _expected_negative(df):
    """The answer cumulative_gate_mask gives for the same gate — the path the
    editor uses, which always had the gate dictionary."""
    gates = {'g1': _positive(enabled=True), 'g2': _negative()}
    return int(np.asarray(cumulative_gate_mask(gates, 'g2', df),
                          dtype=bool).sum())


def test_a_not_gate_selects_the_complement_not_the_whole_sample(bimodal):
    expected = _expected_negative(bimodal)
    assert 0 < expected < N, 'the fixture no longer splits the sample'

    s = FlowSample.from_dataframe(bimodal.copy(), name='neg')
    s.apply_region_gates([_positive(enabled=False), _negative()])
    assert len(s.data) == expected, (
        f'the CD3-negative population came back with {len(s.data)} of {N} '
        f'events; the complement is {expected}')


def test_the_two_paths_agree(bimodal):
    """apply_region_gates and cumulative_gate_mask must not disagree about
    what a population contains."""
    s = FlowSample.from_dataframe(bimodal.copy(), name='neg')
    s.apply_region_gates([_positive(enabled=False), _negative()])
    assert len(s.data) == _expected_negative(bimodal)


def test_an_unresolvable_boolean_gate_admits_nothing(bimodal):
    """Fail closed: an empty population is visible, a superset is not."""
    orphan = {'id': 'g9', 'kind': 'boolean', 'op': 'not',
              'operands': ['does-not-exist'], 'parent_id': None}
    s = FlowSample.from_dataframe(bimodal.copy(), name='orphan')
    s.apply_region_gates([orphan])
    assert len(s.data) == 0, (
        f'{len(s.data)} of {N} events were admitted by a gate that could not '
        'be evaluated')


def test_gate_to_mask_without_the_dictionary_admits_nothing(bimodal):
    """The unit behind it. Called with no gates_by_id, a boolean gate cannot
    know what it is negating."""
    mask = np.asarray(gate_to_mask(_negative(), bimodal), dtype=bool)
    assert not mask.any(), (
        f'{int(mask.sum())} events admitted with no gate dictionary to '
        'resolve the operands against')


def test_an_and_of_a_gate_and_its_negation_is_empty(bimodal):
    """A sanity check on the semantics: apply_region_gates ANDs its gates, so
    X AND NOT X genuinely selects nothing — this must not be confused with the
    fail-closed path."""
    s = FlowSample.from_dataframe(bimodal.copy(), name='both')
    s.apply_region_gates([_positive(enabled=True), _negative()])
    assert len(s.data) == 0


def test_ordinary_gates_are_unaffected(bimodal):
    """The threshold path must still keep exactly the positive events."""
    s = FlowSample.from_dataframe(bimodal.copy(), name='pos')
    s.apply_region_gates([_positive(enabled=True)])
    assert len(s.data) == int((bimodal['CD3'] > 2.0).sum())
