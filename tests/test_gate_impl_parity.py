"""The two gate implementations must agree, for every gate kind.

`pipeline.gate_to_mask` and `FlowSample._evaluate_gate_on` (the memory-lean
path behind `apply_region_gates`) implement every gate kind separately. That
duplication is deliberate — the second evaluates against an index subset to
avoid materialising full-length temporaries — but it means a change applied to
one and not the other makes the same gate mean two different things depending
on which code path happens to run it, with nothing to say so.

That is not hypothetical: making interval and rect bounds half-open required
editing both, and editing only one would have left the workspace disagreeing
with the editor about which events are in a population.

The fixture deliberately piles events up ON the bounds (and at exactly zero,
which `arcsinh(0)` produces), because boundary handling is where the two are
most likely to drift apart.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, gate_to_mask

# Gates whose selection must be identical under both implementations.
# NOTE the singular keys on cluster/category (`cluster_id`, `value`): getting
# those wrong makes both paths select nothing and "agree" vacuously.
GATES = {
    'threshold on a pile-up':
        {'kind': 'threshold', 'channel': 'X', 'value': 5.0},
    'threshold below zero':
        {'kind': 'threshold', 'channel': 'X', 'value': -1.0},
    'interval with bounds on pile-ups':
        {'kind': 'interval', 'channel': 'X', 'lo': 0.0, 'hi': 10.0},
    'interval open-ended below':
        {'kind': 'interval', 'channel': 'X', 'lo': -1e12, 'hi': 5.0},
    'rect with corners on pile-ups':
        {'kind': 'rect', 'x_channel': 'X', 'y_channel': 'Y',
         'x0': 0.0, 'x1': 10.0, 'y0': 0.0, 'y1': 10.0},
    'polygon square on pile-ups':
        {'kind': 'polygon', 'x_channel': 'X', 'y_channel': 'Y',
         'vertices': [[0, 0], [10, 0], [10, 10], [0, 10]]},
    'polygon concave':
        {'kind': 'polygon', 'x_channel': 'X', 'y_channel': 'Y',
         'vertices': [[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]]},
    'ellipsoid':
        {'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y',
         'mean': [5.0, 5.0], 'cov': [[9.0, 0.0], [0.0, 9.0]],
         'distance_sq': 4.0},
    'cluster membership':
        {'kind': 'cluster', 'channel': 'cluster', 'cluster_id': 2},
    'category membership':
        {'kind': 'category', 'channel': 'label', 'value': 'a'},
}

# These select nothing BY DESIGN (an undefined population must not masquerade
# as "all events"), so they are checked separately from the vacuity guard.
EMPTY_BY_DESIGN = {
    'cluster on a missing column':
        {'kind': 'cluster', 'channel': 'not_a_column', 'cluster_id': 1},
    'category on a missing column':
        {'kind': 'category', 'channel': 'not_a_column', 'value': 'a'},
}


@pytest.fixture(scope='module')
def piled_up_data():
    """Integer-valued events with deliberate pile-ups ON the gate bounds, plus
    a block at exactly zero — the arcsinh(0) case."""
    rng = np.random.RandomState(0)
    n = 6000
    base = rng.randint(-5, 16, n - 1500).astype(float)
    col = np.concatenate([base, np.zeros(500), np.full(500, 10.0),
                          np.full(500, 5.0)])
    df = pd.DataFrame({'X': col, 'Y': col[::-1].copy()})
    df['cluster'] = rng.randint(0, 4, len(df))
    df['label'] = rng.choice(['a', 'b', 'c'], len(df))
    return df


def _both(gate, df):
    gate = dict(gate, id='g')
    a = np.asarray(gate_to_mask(gate, df), dtype=bool)
    sample = FlowSample.__new__(FlowSample)
    sample.data = df
    b = np.asarray(
        sample._evaluate_gate_on(gate, np.arange(len(df)), {'g': gate}),
        dtype=bool)
    return a, b


@pytest.mark.parametrize('label', list(GATES))
def test_both_implementations_select_the_same_events(label, piled_up_data):
    a, b = _both(GATES[label], piled_up_data)
    assert a.sum() > 0, (
        f'{label}: the gate selected NOTHING, so agreement proves nothing — '
        f'fix the fixture or the gate spec')
    assert a.sum() < len(piled_up_data), (
        f'{label}: the gate selected EVERYTHING, so agreement proves little')
    differ = int((a != b).sum())
    assert differ == 0, (
        f'{label}: gate_to_mask and _evaluate_gate_on disagree on {differ} '
        f'event(s) — the same gate means two different things depending on '
        f'which code path evaluates it')


@pytest.mark.parametrize('label', list(EMPTY_BY_DESIGN))
def test_an_undefined_population_is_empty_in_both(label, piled_up_data):
    """A cluster/category gate on a column this sample does not have selects
    NOTHING in both paths — it must never fall back to all-True, which would
    silently promote an unclustered sample to 'every event'."""
    a, b = _both(EMPTY_BY_DESIGN[label], piled_up_data)
    assert not a.any() and not b.any()
