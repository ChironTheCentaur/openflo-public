"""A gate on a channel the sample does not have is NA, not 100 %.

A template drawn on FL1-A applied to a sample whose detector is named FL1-H
used to admit every event: Statistics showed the gate at 100 % and its
child gated on all events, with only an INFO log line. The region was never
evaluated, so the population is unknown (Sky's rule: uncomputable reads NA,
not a number), and so is everything gated inside it.

Generated data: 1,000 events where 300 are FL1-A positive and, of those,
120 FL2-A positive. The same events with FL1-A renamed FL1-H is the
negative sample.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from openflo.gating import population_stats
from openflo.pipeline import cumulative_gate_mask, gate_to_mask, unknown_population_reason

_GATES = {
    'g1': {'id': 'g1', 'kind': 'threshold', 'channel': 'FL1-A',
           'value': 1000.0, 'name': 'FL1+'},
    'g2': {'id': 'g2', 'kind': 'threshold', 'channel': 'FL2-A',
           'value': 1000.0, 'name': 'FL2+', 'parent_id': 'g1'},
    'g3': {'id': 'g3', 'kind': 'rect', 'x_channel': 'FL1-A',
           'y_channel': 'FL2-A', 'x0': 0, 'x1': 1e6, 'y0': 0, 'y1': 1e6,
           'name': 'box'},
    'g4': {'id': 'g4', 'kind': 'boolean', 'op': 'not', 'operands': ['g1'],
           'name': 'not FL1+'},
}
_ORDER = ['g1', 'g2', 'g3', 'g4']


def _events():
    rng = np.random.default_rng(7)
    fl1 = np.r_[rng.uniform(1500, 5000, 300), rng.uniform(10, 900, 700)]
    fl2 = np.r_[rng.uniform(1500, 5000, 120), rng.uniform(10, 900, 880)]
    return pd.DataFrame({'FSC-A': rng.uniform(1e4, 2e5, 1000),
                         'FL1-A': fl1, 'FL2-A': fl2})


def _stats(df):
    rows = population_stats('synthetic_1', df, _GATES, _ORDER, {}, [],
                            {'Count', '%Parent', '%Total'}, ())
    return {r['__gid__']: r for r in rows}


def test_present_channel_is_counted():
    rows = _stats(_events())
    assert rows['g1']['Count'] == 300
    assert rows['g2']['Count'] == 120
    assert rows['g2']['%Parent'] == pytest.approx(40.0)
    assert rows['g4']['Count'] == 700
    assert not any(r.get('__unknown__') for r in rows.values())


def test_missing_channel_reads_na_with_reason():
    df = _events().rename(columns={'FL1-A': 'FL1-H'})
    rows = _stats(df)
    for gid in ('g1', 'g2', 'g3', 'g4'):
        assert math.isnan(rows[gid]['Count']), gid
        assert math.isnan(rows[gid]['%Parent']), gid
        assert 'FL1-A' in rows[gid]['__unknown__'], gid


def test_missing_channel_selects_no_events():
    df = _events().rename(columns={'FL1-A': 'FL1-H'})
    assert not gate_to_mask(_GATES['g1'], df).any()
    assert not gate_to_mask(_GATES['g3'], df).any()
    assert not cumulative_gate_mask(_GATES, 'g2', df).any()
    # NOT over a gate that could not be evaluated is not "everything".
    assert not cumulative_gate_mask(_GATES, 'g4', df).any()


def test_reason_needs_the_sample_columns():
    # Without the sample's columns the gate list is not judged (callers that
    # have no sample keep their answer); with them the gap is named.
    assert unknown_population_reason(_GATES, 'g2') is None
    cols = _events().rename(columns={'FL1-A': 'FL1-H'}).columns
    assert unknown_population_reason(_GATES, 'g2', columns=cols) == \
        'channel FL1-A is not in this sample'
    assert unknown_population_reason(_GATES, 'g2',
                                     columns=_events().columns) is None
