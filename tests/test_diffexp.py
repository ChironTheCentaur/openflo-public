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
    # X differs strongly → small p, positive log2fc (A>B).
    assert by['X']['p'] < 0.05
    assert by['X']['log2fc'] > 0
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
