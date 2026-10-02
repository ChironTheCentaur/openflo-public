"""Geometric mean and robust CV in population_stats.

Fluorescence is approximately log-normal, so the geometric mean — not the
arithmetic mean — is the central tendency FlowJo reports and papers cite; its
absence meant anyone comparing OpenFlo numbers to FlowJo's was comparing
different statistics. The robust (MAD-based) CV is the resistant form used in
flow QC.

These check the VALUES against independently-derived truth (a known log-normal,
a known MAD), not merely that a column appears.
"""
import numpy as np
import pandas as pd

from openflo.gating import population_stats

CH = ['CD3']
ALL = {'Count', 'Median', 'Mean', 'GeoMean', 'CV', 'rCV'}
STAT_CHAN = ('Median', 'Mean', 'GeoMean', 'CV', 'rCV')


def _rows(vals, want=ALL):
    df = pd.DataFrame({'CD3': np.asarray(vals, dtype=float)})
    gates = {'g1': {'id': 'g1', 'kind': 'threshold', 'parent_id': None,
                    'channel': 'CD3', 'op': '>', 'value': -np.inf,
                    'name': 'all', 'enabled': True}}
    return population_stats('s', df, gates, ['g1'], {}, CH, want,
                            STAT_CHAN)[0]


def test_geomean_recovers_a_known_lognormal_median():
    """For a log-normal, the geometric mean converges on the MEDIAN — and is
    well below the arithmetic mean. That separation is the whole point."""
    rng = np.random.default_rng(0)
    true_median = 5000.0
    vals = rng.lognormal(mean=np.log(true_median), sigma=0.8, size=200_000)
    row = _rows(vals)
    gm, med, mean = (row['GeoMean CD3'], row['Median CD3'], row['Mean CD3'])
    assert abs(gm - true_median) / true_median < 0.02, gm
    assert abs(gm - med) / med < 0.02, 'geomean should track the median'
    assert mean > gm * 1.2, (
        'arithmetic mean must sit well above the geometric mean for a '
        'right-skewed distribution — otherwise the new column adds nothing')


def test_geomean_of_a_constant_is_that_constant():
    row = _rows(np.full(1000, 250.0))
    assert abs(row['GeoMean CD3'] - 250.0) < 1e-9


def test_geomean_excludes_non_positive_values_rather_than_clamping():
    """Compensated data legitimately contains negatives. log() is undefined
    there; clamping to a floor would bias the result upward."""
    vals = np.concatenate([np.full(500, 100.0), np.full(500, -50.0)])
    row = _rows(vals)
    assert abs(row['GeoMean CD3'] - 100.0) < 1e-9, (
        'negatives leaked into the geometric mean')
    # The other statistics still use every finite event.
    assert abs(row['Mean CD3'] - 25.0) < 1e-9


def test_geomean_is_nan_when_nothing_is_positive():
    row = _rows(np.full(100, -5.0))
    assert np.isnan(row['GeoMean CD3'])


def test_robust_cv_matches_the_mad_definition():
    """rCV = 100 * 1.4826*MAD / median, checked against a hand computation."""
    vals = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    row = _rows(vals)
    med = 10.0
    mad = np.median(np.abs(vals - med))            # 1.0
    assert abs(row['rCV CD3'] - (1.4826 * mad / med * 100.0)) < 1e-9


def test_robust_cv_resists_an_outlier_the_ordinary_cv_does_not():
    """The reason it exists: one absurd event must not move it much."""
    base = np.full(999, 100.0)
    clean = _rows(base)
    dirty = _rows(np.append(base, 1e7))
    assert dirty['CV CD3'] > 100 * clean['CV CD3'] + 1, (
        'the ordinary CV should be wrecked by the outlier')
    assert abs(dirty['rCV CD3'] - clean['rCV CD3']) < 1.0, (
        'the robust CV moved a lot — it is not resistant')


def test_new_columns_are_opt_in():
    """Existing callers must see no new columns unless they ask."""
    row = _rows(np.linspace(1, 100, 500), want={'Count', 'Median'})
    assert 'Median CD3' in row
    assert 'GeoMean CD3' not in row and 'rCV CD3' not in row


def test_empty_population_yields_nan_not_a_crash():
    df = pd.DataFrame({'CD3': np.array([1.0, 2.0, 3.0])})
    gates = {'g1': {'id': 'g1', 'kind': 'threshold', 'parent_id': None,
                    'channel': 'CD3', 'op': '>', 'value': 1e9,
                    'name': 'none', 'enabled': True}}
    row = population_stats('s', df, gates, ['g1'], {}, CH, ALL, STAT_CHAN)[0]
    assert row['Count'] == 0
    assert np.isnan(row['GeoMean CD3']) and np.isnan(row['rCV CD3'])
