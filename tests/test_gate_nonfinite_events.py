"""How gates treat non-finite events.

Gate counts are the headline number in cytometry, and the handling of NaN /
±inf across gate kinds was neither documented nor tested — it just fell out of
numpy comparison semantics. That is fine while it is correct, and a trap the
moment someone "tidies" a comparison.

Verified behaviour, pinned here:

* ``NaN`` is OUTSIDE every gate kind. Every comparison with NaN is False, so a
  non-finite event can never be counted in a population.
* ``-inf`` is likewise outside everything, including ``X > v``.
* ``+inf`` is inside ``X > v`` — it genuinely is above the threshold — but
  outside every BOUNDED region (interval, rect, polygon), which is equally
  correct. The asymmetry is inherent to the shapes, not an accident.

Note the interaction with `transform_values`: since non-finite values now stay
NaN through logicle/hyperlog, an infinite reading on a fluorescence channel is
excluded everywhere by the time gating sees it. ±inf survives only on linearly
stored channels (FSC/SSC).
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import cumulative_gate_mask, gate_to_mask

VALUES = [1.0, 5.0, np.nan, 10.0, np.inf, -np.inf]
IDX = {'low': 0, 'mid': 1, 'nan': 2, 'high': 3, 'inf': 4, 'neg_inf': 5}


@pytest.fixture
def df():
    return pd.DataFrame({'X': VALUES, 'Y': [1.0, 5.0, 5.0, 10.0, 5.0, 5.0]})


def _mask(gate, df):
    return np.asarray(gate_to_mask(gate, df, {gate['id']: gate}, 0), dtype=bool)


THRESHOLD = {'id': 'g', 'kind': 'threshold', 'parent_id': None,
             'channel': 'X', 'op': '>', 'value': 4.0, 'name': 't'}
INTERVAL = {'id': 'g', 'kind': 'interval', 'parent_id': None, 'channel': 'X',
            'lo': 0.0, 'hi': 100.0, 'name': 'i'}
RECT = {'id': 'g', 'kind': 'rect', 'parent_id': None, 'x_channel': 'X',
        'y_channel': 'Y', 'x0': 0.0, 'x1': 100.0, 'y0': 0.0, 'y1': 100.0,
        'name': 'r'}
POLY = {'id': 'g', 'kind': 'polygon', 'parent_id': None, 'x_channel': 'X',
        'y_channel': 'Y', 'name': 'p',
        'vertices': [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]}
ALL_KINDS = [
    pytest.param(THRESHOLD, id='threshold'),
    pytest.param(INTERVAL, id='interval'),
    pytest.param(RECT, id='rect'),
    pytest.param(POLY, id='polygon'),
]


@pytest.mark.parametrize('gate', ALL_KINDS)
def test_nan_is_never_inside_a_gate(gate, df):
    """A NaN event must never be counted in a population."""
    assert not _mask(gate, df)[IDX['nan']], (
        f"a NaN event was counted inside the {gate['kind']} gate")


@pytest.mark.parametrize('gate', ALL_KINDS)
def test_negative_infinity_is_never_inside_a_gate(gate, df):
    assert not _mask(gate, df)[IDX['neg_inf']]


@pytest.mark.parametrize('gate', ALL_KINDS)
def test_ordinary_events_are_unaffected_by_their_bad_neighbours(gate, df):
    """The presence of non-finite rows must not perturb the finite ones."""
    m = _mask(gate, df)
    assert m[IDX['mid']] and m[IDX['high']], f"{gate['kind']} lost real events"


def test_positive_infinity_is_above_a_threshold(df):
    """inf > 4 is True, and that is the right answer — an infinitely bright
    event IS above the cut."""
    assert _mask(THRESHOLD, df)[IDX['inf']]


@pytest.mark.parametrize('gate', [INTERVAL, RECT, POLY])
def test_positive_infinity_is_outside_every_bounded_region(gate, df):
    """Bounded shapes cannot contain it, whatever the threshold does."""
    assert not _mask(gate, df)[IDX['inf']]


def test_a_nan_parent_excludes_its_children(df):
    """Cumulative masks AND up the chain, so an event excluded by an ancestor
    stays excluded — a NaN cannot re-enter through a child gate."""
    gates = {
        'p': dict(INTERVAL, id='p', parent_id=None),
        'c': dict(THRESHOLD, id='c', parent_id='p'),
    }
    m = np.asarray(cumulative_gate_mask(gates, 'c', df), dtype=bool)
    assert not m[IDX['nan']] and not m[IDX['inf']], (
        'a non-finite event entered a child population')
    assert m[IDX['mid']] and m[IDX['high']]


def test_an_all_nan_channel_yields_an_empty_population(df):
    allnan = pd.DataFrame({'X': [np.nan] * 6, 'Y': df['Y']})
    for gate in (THRESHOLD, INTERVAL, RECT, POLY):
        assert _mask(gate, allnan).sum() == 0, (
            f"{gate['kind']} counted events from an all-NaN channel")
