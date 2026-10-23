"""Differential abundance / expression between sample groups."""
import types

import numpy as np
import pandas as pd
import pytest

from openflo.diffexp import (
    _benjamini_hochberg,
    cluster_abundance,
    differential_test,
    marker_expression,
)

# ── BH FDR ───────────────────────────────────────────────────────────────────

def test_bh_monotone_and_bounded():
    raw = [0.001, 0.01, 0.02, 0.5]
    adj = _benjamini_hochberg(raw)
    assert all(0.0 <= q <= 1.0 for q in adj)
    assert adj == sorted(adj)            # same order → non-decreasing
    assert adj[0] == pytest.approx(0.004)   # 0.001 * 4 / 1


def test_bh_passes_through_nan():
    adj = _benjamini_hochberg([0.01, float('nan'), 0.02])
    assert np.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


# ── differential_test ────────────────────────────────────────────────────────

def test_differential_test_detects_shift():
    rng = np.random.default_rng(0)
    a = list(rng.normal(10, 1, 8))
    b = list(rng.normal(2, 1, 8))
    rows = differential_test({'X': a, 'Y': list(rng.normal(5, 1, 8))},
                             {'X': b, 'Y': list(rng.normal(5, 1, 8))})
    by = {r['feature']: r for r in rows}
    # X differs strongly → small p; log2fc is B over A, so A>B is negative.
    assert by['X']['p'] < 0.05
    assert by['X']['log2fc'] < 0
    # Y is the same in both → not significant.
    assert by['Y']['p'] > 0.05
    # Sorted by p ascending.
    assert rows[0]['feature'] == 'X'
    assert all('p_adj' in r for r in rows)


def test_differential_test_only_common_features():
    rows = differential_test({'X': [1, 2, 3], 'A': [1, 2]},
                             {'X': [4, 5, 6], 'B': [1, 2]})
    assert {r['feature'] for r in rows} == {'X'}


# ── builders ─────────────────────────────────────────────────────────────────

def _sample(cluster_vals, markers=None):
    data = {'cluster': cluster_vals}
    if markers:
        data.update(markers)
    return types.SimpleNamespace(data=pd.DataFrame(data))


def test_cluster_abundance_pads_missing_labels_with_zero():
    # Group A samples have clusters {0,1}; group B only {0}.
    a = [_sample([0, 0, 1, 1]), _sample([0, 1, 1, 1])]
    b = [_sample([0, 0, 0, 0])]
    ga, gb = cluster_abundance(a, b)
    # cluster 1 absent in B → 0% for that sample.
    assert gb[1] == [0.0]
    # cluster 0 in A: 50% and 25%.
    assert ga[0] == pytest.approx([50.0, 25.0])
    assert set(ga) == set(gb)            # same label set in both groups


def test_marker_expression_within_population():
    a = [_sample([0, 0, 1], {'CD4': [10.0, 12.0, 99.0]})]
    b = [_sample([0, 0, 1], {'CD4': [1.0, 3.0, 99.0]})]
    ga, gb = marker_expression(a, b, ['CD4'], label_col='cluster',
                               label_value=0, stat='median')
    assert ga['CD4'] == pytest.approx([11.0])   # median of {10,12}
    assert gb['CD4'] == pytest.approx([2.0])    # median of {1,3}


def test_marker_expression_missing_channel_is_nan():
    a = [_sample([0, 1], {'CD4': [1.0, 2.0]})]
    b = [_sample([0, 1], {'CD4': [3.0, 4.0]})]
    ga, _ = marker_expression(a, b, ['NOPE'])
    assert np.isnan(ga['NOPE'][0])


def test_differential_test_log2fc_magnitude():
    """log2fc MAGNITUDE, not just sign: mean_a=8, mean_b=2 → log2(2/8)=-2.0
    (B over A, as compare_all_features and differential_abundance)."""
    r = differential_test({'X': [8.0, 8, 8, 8]}, {'X': [2.0, 2, 2, 2]})[0]
    assert r['mean_a'] == 8.0 and r['mean_b'] == 2.0
    assert r['log2fc'] == pytest.approx(-2.0, abs=1e-6)


def test_differential_abundance_proportions_and_direction():
    """prop_a/prop_b are the mean per-sample proportions (lib size = column sum);
    log2fc is group-B-vs-A. A cluster that rises in B → positive log2fc, z>0.
    Only direction/significance were checked before, never prop values."""
    from openflo.diffexp import differential_abundance
    counts = np.array([[80, 80, 50, 50],       # cluster 0 (falls in B)
                       [20, 20, 50, 50]])      # cluster 1 (rises in B)
    rows = {r['cluster']: r for r in
            differential_abundance(counts, ['A', 'A', 'B', 'B'])}
    r1 = rows[1]
    assert r1['prop_a'] == pytest.approx(0.2)  # mean(20/100, 20/100)
    assert r1['prop_b'] == pytest.approx(0.5)  # mean(50/100, 50/100)
    assert r1['log2fc'] > 0 and r1['z'] > 0    # up in B
    assert rows[0]['log2fc'] < 0               # cluster 0 down in B


def _flow_sample(name, median, record, seed):
    """An editor-like sample: CD3 lognormal around `median` (linear), stored
    on the loader's logicle, with `record` as its data_transforms."""
    import openflo.pipeline as fp
    lin = np.random.default_rng(seed).lognormal(np.log(median), 0.3, 2000)
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'CD3-A': fp.transform_values(lin, method='logicle')}),
        name=name)
    s.data_transforms = {'CD3-A': record}
    return s, float(np.median(lin))


def test_marker_expression_reports_linear_medians():
    """Editor samples hold the loader's logicle coordinates: the per-sample
    medians must be the compensated LINEAR ones (here 1000 / 100), not the
    display values (~0.6 / ~0.45), or every fold change shrinks (a 10x
    difference measured log2FC 0.51)."""
    import openflo.pipeline as fp
    logicle = fp.transform_spec('logicle')
    a = [_flow_sample('a1', 1000, logicle, 1), _flow_sample('a2', 1000, logicle, 2)]
    b = [_flow_sample('b1', 100, logicle, 3), _flow_sample('b2', 100, logicle, 4)]
    ga, gb = marker_expression([s for s, _ in a], [s for s, _ in b], ['CD3-A'])
    assert ga['CD3-A'] == pytest.approx([m for _, m in a], rel=1e-6)
    assert gb['CD3-A'] == pytest.approx([m for _, m in b], rel=1e-6)
    r = differential_test(ga, gb)[0]
    # log2fc is B over A (856218e): B's 100 against A's 1000.
    assert r['log2fc'] == pytest.approx(-np.log2(10.0), abs=0.05)


def test_marker_expression_names_the_samples_it_left_out():
    """A sample whose CD3 is of unknown scale (an older OpenFlo CSV) has no
    linear median: it went in as NaN and differential_test silently dropped
    it (audit: group_a [1009.6, 1100.2, nan] -> n_a=2, nothing said). The
    groups and every result row now name it, and a guessed one."""
    import openflo.pipeline as fp
    logicle = fp.transform_spec('logicle')
    a = [_flow_sample('a1', 1000, logicle, 1)[0],
         _flow_sample('a2', 1100, fp.guessed_spec(logicle), 2)[0],
         _flow_sample('a3_old', 1050, fp.unknown_spec('file:old.csv'), 3)[0]]
    b = [_flow_sample(f'b{i}', 100 + i, logicle, 10 + i)[0] for i in range(3)]
    ga, gb = marker_expression(a, b, ['CD3-A'])
    assert ga.omitted == {'CD3-A': ['a3_old']} and gb.omitted == {}
    assert ga.guessed == {'CD3-A': ['a2']}
    r = differential_test(ga, gb)[0]
    assert r['n_a'] == 2 and r['omitted_a'] == ['a3_old']
    assert r['omitted_b'] == [] and r['guessed_a'] == ['a2']


def test_differential_test_defaults_to_welch_on_the_values_as_given():
    """Welch's t-test by default, as the app and openflo-run; no transform,
    since the inputs may be percentages or intensities (the logit of an
    intensity above 100 is undefined)."""
    from scipy.stats import ttest_ind
    a = {'pct': [60.0, 62.0, 64.0], 'mfi': [2400.0, 2600.0, 2500.0]}
    b = {'pct': [40.0, 38.0, 42.0], 'mfi': [900.0, 1100.0, 1000.0]}
    rows = {r['feature']: r for r in differential_test(a, b)}
    for f in a:
        assert rows[f]['test'] == 'Welch t-test'
        assert rows[f]['p'] == pytest.approx(
            ttest_ind(a[f], b[f], equal_var=False).pvalue)
        assert not rows[f]['underpowered']
