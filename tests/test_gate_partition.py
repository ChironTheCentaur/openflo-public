"""Adjacent interval and rect gates must partition — tested where the rule lives.

The half-open `[lo, hi)` rule was introduced because fully-open bounds silently
LOST every event sitting exactly on a boundary: adjacent intervals [0,10] and
[10,20] put a value of exactly 10 in neither, so those events vanished from
both populations and the percentages did not sum. Measured at 14.3% of events
on integer $DATATYPE I data.

That rule lives in `gate_to_mask`'s `interval` and `rect` branches and in
`FlowSample._evaluate_gate_on`. The existing partition tests in
test_polygon_gating.py call `_points_in_polygon` DIRECTLY, so they cover the
polygon implementation and never touch interval or rect at all — reverting
those two to closed bounds passes that whole file. This closes that.

Both failure directions are asserted, because they are different bugs with
opposite symptoms and a test for one is silent about the other:

    claimed twice  -> frequencies sum above 100%
    claimed never  -> events vanish, and nothing reports a loss

Integer data throughout. Floats essentially never land exactly on a bound, so
a float-only test passes no matter which rule is in force.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, gate_to_mask

STEP = 10
EDGES = [0, 10, 20, 30, 40]


@pytest.fixture
def grid():
    """Integer events sitting exactly ON every boundary the gates use."""
    xs = np.repeat(np.array(EDGES), len(EDGES))
    ys = np.tile(np.array(EDGES), len(EDGES))
    return pd.DataFrame({'X': xs.astype(np.int64), 'Y': ys.astype(np.int64)})


def _claim_counts(gates, df, evaluator):
    return sum(np.asarray(evaluator(g, df)).astype(int) for g in gates)


def _by_gate_to_mask(gate, df):
    return gate_to_mask(gate, df)


def _by_evaluate_gate_on(gate, df):
    s = FlowSample.from_dataframe(df.copy(), name='t')
    s.fluor_channels = ['X', 'Y']
    return s._evaluate_gate_on(gate, np.arange(len(df)))


EVALUATORS = [
    pytest.param(_by_gate_to_mask, id='gate_to_mask'),
    pytest.param(_by_evaluate_gate_on, id='_evaluate_gate_on'),
]


@pytest.mark.parametrize('evaluator', EVALUATORS)
def test_adjacent_intervals_claim_each_event_exactly_once(grid, evaluator):
    gates = [{'kind': 'interval', 'channel': 'X', 'lo': lo, 'hi': lo + STEP}
             for lo in EDGES[:-1]]
    counts = _claim_counts(gates, grid, evaluator)
    inside = ((grid['X'] >= EDGES[0]) & (grid['X'] < EDGES[-1])).to_numpy()

    twice = int((counts[inside] > 1).sum())
    never = int((counts[inside] < 1).sum())
    assert twice == 0, (
        f'{twice} event(s) claimed by two adjacent intervals — a frequency '
        f'can now exceed 100%')
    assert never == 0, (
        f'{never} event(s) claimed by no interval — they have vanished from '
        f'every population and nothing reports the loss')


@pytest.mark.parametrize('evaluator', EVALUATORS)
def test_adjacent_rects_claim_each_event_exactly_once(grid, evaluator):
    gates = [{'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
              'x0': x, 'x1': x + 2 * STEP, 'y0': EDGES[0], 'y1': EDGES[-1]}
             for x in (0, 20)]
    counts = _claim_counts(gates, grid, evaluator)
    inside = ((grid['X'] >= EDGES[0]) & (grid['X'] < EDGES[-1]) &
              (grid['Y'] >= EDGES[0]) & (grid['Y'] < EDGES[-1])).to_numpy()

    assert int((counts[inside] > 1).sum()) == 0, 'rects double-claim an event'
    assert int((counts[inside] < 1).sum()) == 0, (
        'rects lose an event between them')


@pytest.mark.parametrize('evaluator', EVALUATORS)
def test_four_quadrants_about_a_divider_on_the_data(grid, evaluator):
    """The case quadrant gates depend on: a divider placed exactly where
    integer events pile up."""
    div, big = 20, 10 ** 9
    gates = [{'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
              'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
             for x0, x1 in ((-big, div), (div, big))
             for y0, y1 in ((-big, div), (div, big))]
    counts = _claim_counts(gates, grid, evaluator)

    on_divider = ((grid['X'] == div) | (grid['Y'] == div)).to_numpy()
    assert int(on_divider.sum()) > 0, 'fixture no longer piles up on the divider'
    assert int((counts > 1).sum()) == 0, 'an event is in two quadrants'
    assert int((counts < 1).sum()) == 0, 'an event is in no quadrant'


def test_the_two_implementations_agree_on_every_boundary_gate(grid):
    """A divergence here means the same region gates differently depending on
    which code path evaluated it."""
    gates = (
        [{'kind': 'interval', 'channel': 'X', 'lo': lo, 'hi': lo + STEP}
         for lo in EDGES[:-1]] +
        [{'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
          'x0': x, 'x1': x + 2 * STEP, 'y0': 0, 'y1': 40} for x in (0, 20)]
    )
    for i, g in enumerate(gates):
        a = np.asarray(_by_gate_to_mask(g, grid))
        b = np.asarray(_by_evaluate_gate_on(g, grid))
        assert np.array_equal(a, b), (
            f'gate #{i} ({g["kind"]}) selects {int((a != b).sum())} different '
            f'events depending on the implementation')
