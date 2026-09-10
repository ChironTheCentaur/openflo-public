"""A cluster absent from a sample is 0%, not "not measured".

`combined_frequencies` is built from `value_counts()`, which emits no row for a
cluster with zero events. `compare_conditions` then averaged the rows directly,
so a population found in only some samples was averaged over only those
samples — the group mean silently became a conditional mean.

Measured: a cluster present in 1 of 3 samples at 10% was reported as the
condition's 10%. The group mean is 3.3%. That number goes straight into a
published comparison table and bar chart.

The frequencies are now squared off into a sample x cluster table before the
average, so every sample in the condition contributes. The standard deviation
becomes meaningful for the same reason — it was NaN over a single row — and
the sample count is reported alongside it, as the differential-abundance table
already does.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowExperiment, FlowSample


def _sample(name, clusters):
    n = len(clusters)
    df = pd.DataFrame({'FSC-A': np.linspace(1.0, 2.0, n),
                       'CD3': np.linspace(1.0, 2.0, n)})
    s = FlowSample.from_dataframe(df, name=name)
    s.data['cluster'] = list(clusters)
    s.fluor_channels = ['CD3']
    return s


def _experiment(*samples):
    exp = FlowExperiment.__new__(FlowExperiment)
    exp.samples = {s.name: s for s in samples}
    return exp


@pytest.fixture
def patchy():
    """Condition A has three samples; cluster 7 appears in only one of them,
    at 10%. Condition B has one sample carrying it at 5%."""
    return _experiment(
        _sample('a1', [0] * 90 + [7] * 10),
        _sample('a2', [0] * 100),
        _sample('a3', [0] * 100),
        _sample('b1', [0] * 95 + [7] * 5))


def _row(summary, condition, cluster):
    hit = summary[(summary['condition'] == condition)
                  & (summary['cluster'] == cluster)]
    assert len(hit) == 1, f'expected one row for {condition}/{cluster}'
    return hit.iloc[0]


def test_a_cluster_missing_from_some_samples_is_averaged_over_all_of_them(
        patchy):
    summary = patchy.compare_conditions(['a1', 'a2', 'a3'], ['b1'], plot=False)
    row = _row(summary, 'A', 7)
    assert row['mean_pct'] == pytest.approx(10.0 / 3.0, abs=1e-6), (
        f"cluster 7 reported {row['mean_pct']}% for condition A; it is present "
        'in 1 of 3 samples at 10%, so the group mean is 3.33%')


def test_the_sample_count_reflects_the_condition_not_the_detections(patchy):
    summary = patchy.compare_conditions(['a1', 'a2', 'a3'], ['b1'], plot=False)
    assert _row(summary, 'A', 7)['n_samples'] == 3
    assert _row(summary, 'A', 0)['n_samples'] == 3
    assert _row(summary, 'B', 7)['n_samples'] == 1


def test_a_cluster_present_everywhere_is_unchanged(patchy):
    """The fix must not disturb the ordinary case."""
    row = _row(patchy.compare_conditions(['a1', 'a2', 'a3'], ['b1'],
                                         plot=False), 'A', 0)
    assert row['mean_pct'] == pytest.approx((90.0 + 100.0 + 100.0) / 3.0)


def test_the_spread_is_measured_over_every_sample(patchy):
    """sd was NaN because the average ran over one row. With the zeros present
    it describes the real variability between samples."""
    row = _row(patchy.compare_conditions(['a1', 'a2', 'a3'], ['b1'],
                                         plot=False), 'A', 7)
    assert np.isfinite(row['sd_pct'])
    assert row['sd_pct'] == pytest.approx(
        float(np.std([10.0, 0.0, 0.0], ddof=1)), abs=1e-6)


def test_each_condition_still_sums_to_about_one_hundred_percent(patchy):
    """The zero-filling must not invent or lose events."""
    summary = patchy.compare_conditions(['a1', 'a2', 'a3'], ['b1'], plot=False)
    for condition in ('A', 'B'):
        total = summary[summary['condition'] == condition]['mean_pct'].sum()
        assert total == pytest.approx(100.0, abs=1e-6), (
            f'condition {condition} sums to {total}%')


def test_populations_are_still_named(patchy):
    summary = patchy.compare_conditions(['a1', 'a2', 'a3'], ['b1'], plot=False)
    assert set(summary['population']) == {'Cluster 0', 'Cluster 7'}


def test_a_cluster_seen_in_only_one_condition_is_zero_in_the_other():
    """The strongest form: B never shows cluster 7 at all, so B's mean for it
    is 0% — not a missing row that a reader would fill in themselves."""
    exp = _experiment(_sample('a1', [0] * 80 + [7] * 20),
                      _sample('b1', [0] * 100),
                      _sample('b2', [0] * 100))
    summary = exp.compare_conditions(['a1'], ['b1', 'b2'], plot=False)
    row = _row(summary, 'B', 7)
    assert row['mean_pct'] == pytest.approx(0.0)
    assert row['n_samples'] == 2
