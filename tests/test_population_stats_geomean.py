"""Geometric mean and robust CV in population_stats.

Fluorescence is approximately log-normal, so the geometric mean — not the
arithmetic mean — is the central tendency FlowJo reports and papers cite; its
absence meant anyone comparing OpenFlo numbers to FlowJo's was comparing
different statistics. The robust CV is the resistant form used in flow QC.
Both follow FlowJo's published definitions.

These check the VALUES against independently-derived truth (a known log-normal,
known percentiles), not merely that a column appears.
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


def test_robust_cv_matches_flowjos_definition():
    """FlowJo: rCV = 100 * 1/2 (Intensity[P84.13] - Intensity[P15.87]) /
    Median. It used to be 1.4826*MAD/median, which agrees only for a
    symmetric distribution: on a log-normal (sigma 0.8) it read 74% where
    FlowJo reports 89%, under the same column name."""
    vals = np.arange(1.0, 1001.0)
    row = _rows(vals)
    lo, hi = np.percentile(vals, [15.87, 84.13])
    assert abs(row['rCV CD3'] - 50.0 * (hi - lo) / np.median(vals)) < 1e-9
    rng = np.random.default_rng(1)
    ln = _rows(rng.lognormal(np.log(5000.0), 0.8, 400_000))['rCV CD3']
    # analytic: 100 * (e^0.8 - e^-0.8) / 2 = 88.81
    assert abs(ln - 100 * np.sinh(0.8)) < 1.0, ln


def test_geomean_is_taken_in_display_space_on_a_transformed_channel():
    """FlowJo computes the geometric mean in the display (transform) space and
    back-transforms it, so every event counts, negatives included. The
    positive-only form put a dim population's GeoMean above its median."""
    from openflo.pipeline import (
        inverse_transform_values,
        transform_spec,
        transform_values,
    )
    lin = np.concatenate([np.full(500, 100.0), np.full(500, -50.0)])
    stored = transform_values(lin, method='logicle')
    df = pd.DataFrame({'CD3': stored})
    gates = {'g1': {'id': 'g1', 'kind': 'threshold', 'parent_id': None,
                    'channel': 'CD3', 'value': -np.inf, 'name': 'all'}}
    row = population_stats('s', df, gates, ['g1'], {}, CH, ALL, STAT_CHAN,
                           transforms={'CD3': transform_spec('logicle')})[0]
    want = inverse_transform_values(np.array([stored.mean()]))[0]
    assert abs(row['GeoMean CD3'] - want) < 1e-6
    assert 0.0 < row['GeoMean CD3'] < 100.0
    # and it tracks the geometric mean of a positive log-normal population
    rng = np.random.default_rng(2)
    ln = rng.lognormal(np.log(5000.0), 0.5, 100_000)
    df = pd.DataFrame({'CD3': transform_values(ln, method='logicle')})
    row = population_stats('s', df, gates, ['g1'], {}, CH, ALL, STAT_CHAN,
                           transforms={'CD3': transform_spec('logicle')})[0]
    gm = np.exp(np.log(ln).mean())
    assert abs(row['GeoMean CD3'] - gm) / gm < 0.02


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


def test_stats_on_transformed_data_are_on_the_linear_scale():
    """The editor hands population_stats logicle values (the loader transforms
    fluor channels in place). Gates must still be evaluated in that display
    space, but every intensity statistic must equal the formula on the LINEAR
    values, as FlowJo reports them."""
    from openflo.pipeline import transform_spec, transform_values
    rng = np.random.default_rng(3)
    lin = np.concatenate([rng.lognormal(7.0, 0.5, 4000),
                          rng.normal(0.0, 60.0, 1000)])      # some negatives
    df = pd.DataFrame({'CD3': transform_values(lin, method='logicle')})
    cut_lin = 500.0                        # gate drawn on the logicle axis
    cut = float(transform_values(np.array([cut_lin]), method='logicle')[0])
    gates = {'g1': {'id': 'g1', 'kind': 'threshold', 'parent_id': None,
                    'channel': 'CD3', 'op': '>', 'value': cut,
                    'name': 'pos', 'enabled': True}}
    row = population_stats('s', df, gates, ['g1'], {}, CH, ALL, STAT_CHAN,
                           transforms={'CD3': transform_spec('logicle')})[0]
    x = lin[lin > cut_lin]
    assert row['Count'] == x.size
    med = np.median(x)
    lo, hi = np.percentile(x, [15.87, 84.13])
    # FlowJo's GeoMean: the display-space mean, back-transformed.
    from openflo.pipeline import inverse_transform_values
    gm = inverse_transform_values(np.array([
        transform_values(x, method='logicle').mean()]))[0]
    np.testing.assert_allclose(
        [row['Median CD3'], row['Mean CD3'], row['GeoMean CD3'],
         row['CV CD3'], row['rCV CD3']],
        [med, x.mean(), gm, x.std() / x.mean() * 100,
         50.0 * (hi - lo) / med], rtol=1e-9)
    # Without the record the same frame is read as already linear.
    raw = population_stats('s', df, gates, ['g1'], {}, CH, ALL, STAT_CHAN)[0]
    assert raw['Median CD3'] < 1.0


def test_empty_population_yields_nan_not_a_crash():
    df = pd.DataFrame({'CD3': np.array([1.0, 2.0, 3.0])})
    gates = {'g1': {'id': 'g1', 'kind': 'threshold', 'parent_id': None,
                    'channel': 'CD3', 'op': '>', 'value': 1e9,
                    'name': 'none', 'enabled': True}}
    row = population_stats('s', df, gates, ['g1'], {}, CH, ALL, STAT_CHAN)[0]
    assert row['Count'] == 0
    assert np.isnan(row['GeoMean CD3']) and np.isnan(row['rCV CD3'])
