"""An event with no measurement is not negative for the marker.

A threshold gate correctly leaves NaN outside: `NaN > 2.0` is False. The
boolean NOT gate then negated that mask — and swept every unmeasurable event
into the complement population. Measured on 20,000 events with 2,000 NaN in
CD3: the CD3-negative population came back as 10,004 events, 2,000 of them
(20%) with no CD3 reading at all, against a correct 8,004.

It looked right from the outside, too: the positive and negative gates summed
to exactly the sample total, which is the shape of a correct partition.

This matters more now that `transform_values` returns NaN for input it cannot
place — a non-positive value on a log scale, a non-finite one under
logicle/hyperlog. Those are exactly the dim events a "marker-negative" gate is
about, so the population most likely to absorb them is the one whose meaning
they corrupt.

Non-finite is neither positive nor negative, so such events now belong to
neither side, and the two populations no longer sum to the total — which is
the honest arithmetic when some events could not be measured.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import cumulative_gate_mask

N = 20_000
N_NAN = 2_000


def _gates():
    return {'g1': {'id': 'g1', 'kind': 'threshold', 'channel': 'CD3',
                   'value': 2.0, 'parent_id': None},
            'g2': {'id': 'g2', 'kind': 'boolean', 'op': 'not',
                   'operands': ['g1'], 'parent_id': None}}


def _cd3(with_nan):
    rng = np.random.default_rng(0)
    cd3 = np.concatenate([rng.normal(0.5, 0.4, N // 2),
                          rng.normal(4.0, 0.6, N // 2)])
    if with_nan:
        cd3[:N_NAN] = np.nan
    return pd.DataFrame({'CD3': cd3})


def _mask(gates, gid, df):
    return np.asarray(cumulative_gate_mask(gates, gid, df), dtype=bool)


@pytest.fixture
def with_nan():
    return _cd3(True)


def test_the_negative_population_holds_no_unmeasured_events(with_nan):
    negative = _mask(_gates(), 'g2', with_nan)
    unmeasured = np.isnan(with_nan['CD3'].to_numpy())
    assert int((negative & unmeasured).sum()) == 0, (
        f'{int((negative & unmeasured).sum())} events with no CD3 reading '
        'were counted as CD3-negative')


def test_the_negative_population_is_the_right_size(with_nan):
    cd3 = with_nan['CD3'].to_numpy()
    expected = int(((cd3 <= 2.0) & np.isfinite(cd3)).sum())
    assert int(_mask(_gates(), 'g2', with_nan).sum()) == expected


def test_the_positive_population_is_unchanged(with_nan):
    """The threshold side was already correct and must stay so."""
    cd3 = with_nan['CD3'].to_numpy()
    positive = _mask(_gates(), 'g1', with_nan)
    assert int(positive.sum()) == int((cd3 > 2.0).sum())
    assert int((positive & np.isnan(cd3)).sum()) == 0


def test_unmeasured_events_belong_to_neither_side(with_nan):
    gates = _gates()
    positive = _mask(gates, 'g1', with_nan)
    negative = _mask(gates, 'g2', with_nan)
    assert int((~positive & ~negative).sum()) == N_NAN
    assert int(positive.sum() + negative.sum()) == N - N_NAN, (
        'the two populations still sum to the whole sample, which is what '
        'made the old behaviour look self-consistent')


def test_a_fully_measured_channel_still_partitions_exactly():
    """The guard must not start dropping events from clean data."""
    df = _cd3(False)
    gates = _gates()
    positive = _mask(gates, 'g1', df)
    negative = _mask(gates, 'g2', df)
    assert int(positive.sum() + negative.sum()) == N
    assert not (positive & negative).any()


def test_a_nested_boolean_still_excludes_the_unmeasured(with_nan):
    """The validity requirement has to follow the operand chain, not stop at
    the first level."""
    gates = _gates()
    gates['g3'] = {'id': 'g3', 'kind': 'boolean', 'op': 'not',
                   'operands': ['g2'], 'parent_id': None}
    unmeasured = np.isnan(with_nan['CD3'].to_numpy())
    assert int((_mask(gates, 'g3', with_nan) & unmeasured).sum()) == 0


def test_a_gate_on_a_label_column_places_no_finiteness_requirement():
    """A cluster or category gate has nothing to be non-finite about, so it
    must not start excluding events."""
    df = pd.DataFrame({'cell_cycle': ['G1'] * 10 + ['S'] * 10})
    gates = {'c1': {'id': 'c1', 'kind': 'category', 'channel': 'cell_cycle',
                    'value': 'G1', 'parent_id': None},
             'c2': {'id': 'c2', 'kind': 'boolean', 'op': 'not',
                    'operands': ['c1'], 'parent_id': None}}
    assert int(_mask(gates, 'c2', df).sum()) == 10
