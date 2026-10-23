"""A population whose gates cannot be computed is MISSING (NA), never 0.

Gates whose region is unknown fail closed: a FlowJo ellipse OpenFlo could
not convert, a boolean that lost an operand, an ellipse whose covariance
describes no ellipse -- and every gate under one or built on one. They
select nothing, so their populations used to report a measured 0 events
(0 %) in Statistics, Frequencies and its comparisons, which disagrees with
"missing is missing, not 0" for a population a sample does not have.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_frequency_missing_populations import (  # noqa: E402
    DESIGN,
    _add,
    _cols,
    _window,
)

from openflo.gating import population_stats  # noqa: E402
from openflo.pipeline import (  # noqa: E402
    cumulative_gate_mask,
    unknown_population_reason,
)

SINGULAR = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'mean': [0.5, 0.5], 'cov': [[1.0, 1.0], [1.0, 1.0]],
            'distance_sq': 4.0, 'parent_id': None, 'name': 'Ell'}


def _gates():
    return {
        'pos': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                'parent_id': None, 'name': 'FL1pos'},
        'ell': dict(SINGULAR),
        'under': {'kind': 'threshold', 'channel': 'FL2-A', 'value': 0.5,
                  'parent_id': 'ell', 'name': 'FL2pos'},
        'not_ell': {'kind': 'boolean', 'op': 'not', 'operands': ['ell'],
                    'parent_id': None, 'name': 'NotEll'},
        'lost': {'kind': 'boolean', 'op': 'and', 'operands': ['gone'],
                 'parent_id': None, 'name': 'Lost'},
        'fj': {'kind': 'flowjo_ellipse', 'parent_id': None, 'name': 'FJ'},
        'none': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 99.0,
                 'parent_id': None, 'name': 'Nobody'},
    }


def _df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FL1-A': rng.random(n), 'FL2-A': rng.random(n)})


def test_each_unknown_region_is_named():
    g = _gates()
    assert unknown_population_reason(g, 'pos') is None
    assert unknown_population_reason(g, 'none') is None   # measured empty
    assert 'covariance' in unknown_population_reason(g, 'ell')
    assert 'covariance' in unknown_population_reason(g, 'under')
    assert 'covariance' in unknown_population_reason(g, 'not_ell')
    assert 'operand' in unknown_population_reason(g, 'lost')
    assert 'FlowJo' in unknown_population_reason(g, 'fj')


def test_not_over_a_singular_ellipse_admits_nothing():
    """Measured before: NOT(singular ellipse) admitted all 200 events."""
    g, df = _gates(), _df()
    assert not cumulative_gate_mask(g, 'not_ell', df).any()


def test_unknown_populations_are_nan_and_say_why():
    g, df = _gates(), _df()
    rows = population_stats('s', df, g, list(g), {}, ['FL1-A'],
                            {'Count', '%Parent', '%Total', 'Median'},
                            ('Median',))
    by = {r['__gid__']: r for r in rows}
    for gid in ('ell', 'under', 'not_ell', 'lost', 'fj'):
        r = by[gid]
        assert r['__unknown__'], gid
        assert all(math.isnan(r[k]) for k in
                   ('Count', '%Parent', '%Total', 'Median FL1-A')), gid
    # A computable population, empty or not, keeps its measured numbers.
    assert '__unknown__' not in by['pos'] and by['pos']['Count'] > 0
    assert by['none']['Count'] == 0 and by['none']['%Total'] == 0.0
    assert '__unknown__' not in by['none']


def test_statistics_window_says_na():
    from openflo.ui_statistics import StatisticsWindow
    row = {'Sample': 's', 'Population': 'Ell', 'Count': float('nan'),
           '%Parent': float('nan'), '__unknown__': 'x'}
    assert StatisticsWindow._cell(row, 'Sample') == 's'
    assert StatisticsWindow._cell(row, 'Count') == 'NA'
    assert StatisticsWindow._cell(row, '%Parent', fmt=False) == 'NA'
    ok = {'Sample': 's', 'Count': 0, '%Parent': float('nan')}
    assert StatisticsWindow._cell(ok, 'Count') == '0'
    assert StatisticsWindow._cell(ok, '%Parent', fmt=False) == ''


def test_frequencies_treat_an_unknown_population_as_missing(monkeypatch,
                                                            tmp_path):
    """The Stim samples' FOXP3 gate sits under an ellipse with a singular
    covariance; in Ctrl it is a real gate. Stim is missing, not three 0 %."""
    def build(ed):
        for i, (nm, tr, a, b) in enumerate(DESIGN):
            gates = {'g1': {'kind': 'threshold', 'channel': 'CD3-A',
                            'value': 0.5, 'parent_id': None,
                            'name': 'CD3pos', 'enabled': True}}
            if nm.startswith('Stim'):
                gates['g2'] = {'kind': 'ellipsoid', 'x_channel': 'CD3-A',
                               'y_channel': 'FOXP3-A', 'mean': [1, 1],
                               'cov': [[0, 0], [0, 0]], 'distance_sq': 4.0,
                               'parent_id': 'g1', 'name': 'Tregs',
                               'enabled': True}
            else:
                gates['g2'] = {'kind': 'threshold', 'channel': 'FOXP3-A',
                               'value': 0.5, 'parent_id': 'g1',
                               'name': 'Tregs', 'enabled': True}
            _add(ed, nm, tr, gates, _cols(a, b, 50 + 2 * i, i))
    root, w = _window(monkeypatch, build)
    try:
        tidy = w._tidy_frame()
        tre = tidy[tidy['Population'].str.endswith('Tregs')]
        miss = tre[tre['Missing'].astype(bool)]
        assert sorted(miss['Sample']) == ['Stim_1', 'Stim_2', 'Stim_3']
        assert miss['Count'].isna().all() and miss['%Parent'].isna().all()
        # exported, a missing cell says NA
        w._tidy = tidy
        out = tmp_path / 'tidy.csv'
        monkeypatch.setattr(w, '_ask', lambda *a, **k: str(out))
        monkeypatch.setattr(w, '_done', lambda *a, **k: None)
        w._export_tidy()
        text = out.read_text()
        assert ',NA,NA,NA,True' in text
    finally:
        root.destroy()
