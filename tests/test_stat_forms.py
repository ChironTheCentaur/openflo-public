"""SD's sample form, and the GeoMean and rCV forms chosen in Preferences.

SD divides by n - 1 (the standard sample SD). GeoMean and rCV default to
FlowJo's definitions; Preferences can switch each to the textbook /
robust-statistics form. Every value here is checked against a hand-worked
answer on generated data.
"""
from __future__ import annotations

import json
import os
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from openflo import prefs
from openflo.gating import CHANNEL_STAT_ORDER, population_stats
from openflo.pipeline import (
    inverse_transform_values,
    transform_spec,
    transform_values,
)

CH = ['FL1-A']
# MAD to SD for a normal population, 1 / Phi^-1(3/4) = 1.4826022...
K = 1.0 / NormalDist().inv_cdf(0.75)
WANT = {'Count', 'Median', 'Mean', 'GeoMean', 'SD', 'rSD', 'CV', 'rCV'}
GATES = {'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                'value': -np.inf, 'parent_id': None, 'name': 'All',
                'enabled': True}}


def _row(vals, forms=None, transforms=None):
    df = pd.DataFrame({'FL1-A': np.asarray(vals, dtype=float)})
    return population_stats('synthetic_1', df, GATES, ['g1'], {}, CH, WANT,
                            CHANNEL_STAT_ORDER, transforms=transforms,
                            stat_forms=forms)[0]


def test_sd_and_cv_use_the_sample_sd():
    # {2, 4, 4, 4, 5, 5, 7, 9}: mean 5, squared deviations sum to 32, so the
    # sample SD is sqrt(32 / 7) (the population SD would be exactly 2).
    r = _row([2, 4, 4, 4, 5, 5, 7, 9])
    assert r['SD FL1-A'] == pytest.approx((32 / 7) ** 0.5, rel=1e-12)
    assert r['CV FL1-A'] == pytest.approx((32 / 7) ** 0.5 / 5 * 100,
                                          rel=1e-12)


def test_one_event_has_no_sd():
    r = _row([42.0])
    assert r['Mean FL1-A'] == 42.0
    assert np.isnan(r['SD FL1-A']) and np.isnan(r['CV FL1-A'])


def test_defaults_are_flowjos_forms():
    assert prefs.load_stat_forms() == {'geomean': 'display',
                                       'rcv': 'percentile'}
    vals = np.arange(1.0, 101.0)
    r = _row(vals)
    lo, hi = np.percentile(vals, [15.87, 84.13])
    assert r['rSD FL1-A'] == pytest.approx(0.5 * (hi - lo), rel=1e-12)


def test_mad_form_of_rsd_and_rcv():
    # 1..9: median 5, absolute deviations 4,3,2,1,0,1,2,3,4 -> MAD 2.
    r = _row(np.arange(1.0, 10.0), forms={'rcv': 'mad'})
    assert r['rSD FL1-A'] == pytest.approx(K * 2, rel=1e-12)
    assert r['rCV FL1-A'] == pytest.approx(K * 2 / 5 * 100, rel=1e-12)
    # The FlowJo form gives a different number on the same events.
    assert _row(np.arange(1.0, 10.0))['rSD FL1-A'] != pytest.approx(
        K * 2, rel=1e-6)


def test_mad_form_without_median_requested():
    df = pd.DataFrame({'FL1-A': np.arange(1.0, 10.0)})
    r = population_stats('synthetic_1', df, GATES, ['g1'], {}, CH,
                         {'rSD'}, CHANNEL_STAT_ORDER,
                         stat_forms={'rcv': 'mad'})[0]
    assert r['rSD FL1-A'] == pytest.approx(K * 2, rel=1e-12)


def test_log_form_of_geomean_on_a_transformed_channel():
    lin = np.array([-50.0, 10.0, 100.0, 1000.0])
    stored = transform_values(lin, method='logicle')
    spec = {'FL1-A': transform_spec('logicle')}
    textbook = _row(stored, forms={'geomean': 'log'}, transforms=spec)
    # exp(mean(log x)) over the positive events only: (10*100*1000)^(1/3).
    assert textbook['GeoMean FL1-A'] == pytest.approx(100.0, rel=1e-9)
    flowjo = _row(stored, transforms=spec)
    # FlowJo's form back-transforms the display-space mean of all four
    # events, the negative one included, so it reads lower.
    expected = inverse_transform_values(np.array([stored.mean()]))[0]
    assert flowjo['GeoMean FL1-A'] == pytest.approx(expected, rel=1e-9)
    assert flowjo['GeoMean FL1-A'] < 99.0


def test_forms_are_read_from_preferences():
    path = prefs._prefs_path()
    old = open(path, encoding='utf-8').read() if os.path.exists(path) else None
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stat_geomean': 'log', 'stat_rcv': 'mad'}, f)
        assert prefs.load_stat_forms() == {'geomean': 'log', 'rcv': 'mad'}
        r = _row(np.arange(1.0, 10.0))
        assert r['rSD FL1-A'] == pytest.approx(K * 2, rel=1e-12)
        # An unknown value reads as FlowJo's form.
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stat_geomean': 'nonsense', 'stat_rcv': 3}, f)
        assert prefs.load_stat_forms() == {'geomean': 'display',
                                           'rcv': 'percentile'}
    finally:
        if old is None:
            os.remove(path)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(old)
