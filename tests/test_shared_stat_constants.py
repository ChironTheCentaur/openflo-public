"""One idea, one constant: copies of the same number cannot drift apart.

The robust SD cut points, the default statistic forms and the default
arcsinh cofactor each had a second, hand-typed copy (15.865 / 84.135 in batch
normalisation against 15.87 / 84.13 in the statistics table; 'display' /
'mad' in Preferences; 150 in the symlog view). Each test changes nothing but
holds the two places to the same answer. Generated data only (FL1-A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo import batch_normalize, gating, prefs, scales
from openflo.gating import CHANNEL_STAT_ORDER, population_stats

GATES = {'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                'value': -np.inf, 'parent_id': None, 'name': 'All',
                'enabled': True}}


def test_batch_normalisation_spread_is_the_tables_percentile_rsd():
    # A skewed population on an even grid, where the two pairs of cut
    # points (15.865 vs 15.87) give different spreads.
    vals = np.exp(np.linspace(0.0, 4.0, 1001))
    df = pd.DataFrame({'FL1-A': vals})
    row = population_stats('synthetic_1', df, GATES, ['g1'], {}, ['FL1-A'],
                           {'rSD'}, CHANNEL_STAT_ORDER,
                           stat_forms=gating.FLOWJO_STAT_FORMS)[0]
    assert batch_normalize.robust_spread(vals) == row['rSD FL1-A']


def test_preferences_fall_back_to_the_librarys_default_forms(monkeypatch):
    monkeypatch.setattr(prefs, 'read_prefs', dict)
    monkeypatch.setattr(gating, 'DEFAULT_STAT_FORMS',
                        {'geomean': 'log', 'rcv': 'percentile'})
    assert prefs.load_stat_forms() == {'geomean': 'log', 'rcv': 'percentile'}


def test_the_symlog_view_starts_from_the_default_cofactor(monkeypatch):
    monkeypatch.setattr(scales, 'ASINH_DEFAULT_COFACTOR', 77.0)
    assert scales._symlog_cofactor(lambda d: d, None) == pytest.approx(77.0)
