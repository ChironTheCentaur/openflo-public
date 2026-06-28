"""Tests for the multi-group statistics in openflo.stats:
multi_group_test (Kruskal-Wallis / Friedman), posthoc_pairwise, effect_size.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openflo.stats import effect_size, multi_group_test, posthoc_pairwise

# --- multi_group_test: unpaired (Kruskal-Wallis) ---------------------------

def test_kruskal_separated_groups_small_p():
    g = {
        'A': [1.0, 1.2, 0.9, 1.1, 1.0],
        'B': [5.0, 5.1, 4.9, 5.2, 5.0],
        'C': [10.0, 10.2, 9.8, 10.1, 10.0],
    }
    r = multi_group_test(g, paired=False)
    assert r['test'] == 'Kruskal-Wallis'
    assert r['k'] == 3
    assert math.isfinite(r['stat'])
    assert r['p'] < 0.05


def test_kruskal_identical_groups_no_test():
    g = {'A': [3.0, 3.0, 3.0], 'B': [3.0, 3.0, 3.0], 'C': [3.0, 3.0, 3.0]}
    r = multi_group_test(g, paired=False)
    # All values identical -> degenerate, NaN p (no crash).
    assert math.isnan(r['p'])
    assert math.isnan(r['stat'])
    assert r['k'] == 3


def test_kruskal_overlapping_groups_large_p():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 30)
    g = {'A': base.tolist(),
         'B': (base + rng.normal(0, 0.01, 30)).tolist()}
    r = multi_group_test(g, paired=False)
    assert r['p'] > 0.05


def test_kruskal_too_few_groups():
    r = multi_group_test({'only': [1.0, 2.0, 3.0]}, paired=False)
    assert r['k'] == 1
    assert math.isnan(r['p'])


def test_kruskal_drops_nans():
    g = {'A': [1.0, np.nan, 1.1], 'B': [5.0, 5.1, np.nan]}
    r = multi_group_test(g, paired=False)
    assert r['k'] == 2
    assert math.isfinite(r['p'])


# --- multi_group_test: paired (Friedman) -----------------------------------

def test_friedman_separated_small_p():
    # Each row (replicate) increases A < B < C consistently.
    g = {
        'A': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'B': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'C': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
    r = multi_group_test(g, paired=True)
    assert r['test'] == 'Friedman'
    assert r['k'] == 3
    assert r['p'] < 0.05


def test_friedman_needs_three_groups():
    g = {'A': [1.0, 2.0, 3.0], 'B': [2.0, 3.0, 4.0]}
    r = multi_group_test(g, paired=True)
    assert r['k'] == 2
    assert math.isnan(r['p'])


def test_friedman_identical_no_test():
    g = {'A': [2.0, 2.0, 2.0], 'B': [2.0, 2.0, 2.0], 'C': [2.0, 2.0, 2.0]}
    r = multi_group_test(g, paired=True)
    assert math.isnan(r['p'])


# --- posthoc_pairwise ------------------------------------------------------

def test_posthoc_pair_count_and_keys():
    g = {'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]}
    res = posthoc_pairwise(g)
    assert len(res) == 3  # AB, AC, BC
    for row in res:
        assert set(row) == {'a', 'b', 'stat', 'p', 'p_adj'}


def test_posthoc_bh_monotone_and_geq_raw():
    # Build pairs with a spread of raw p-values; check BH ordering.
    rng = np.random.default_rng(1)
    g = {
        'A': rng.normal(0, 1, 20).tolist(),
        'B': rng.normal(0.2, 1, 20).tolist(),
        'C': rng.normal(3.0, 1, 20).tolist(),
        'D': rng.normal(6.0, 1, 20).tolist(),
    }
    res = posthoc_pairwise(g)
    raw = [r['p'] for r in res]
    adj = [r['p_adj'] for r in res]
    # Each adjusted p is >= its raw p (BH only inflates).
    for p, q in zip(raw, adj, strict=True):
        assert q >= p - 1e-12
        assert q <= 1.0 + 1e-12
    # BH is monotone when traversed in raw-p sorted order.
    order = np.argsort(raw)
    sorted_adj = [adj[i] for i in order]
    for earlier, later in zip(sorted_adj, sorted_adj[1:], strict=False):
        assert earlier <= later + 1e-12


def test_posthoc_separated_pair_significant():
    g = {'A': [1, 1, 1, 1, 2], 'B': [100, 101, 102, 103, 104]}
    res = posthoc_pairwise(g)
    assert len(res) == 1
    assert res[0]['p_adj'] < 0.05


# --- effect_size -----------------------------------------------------------

def test_cliffs_delta_sign_positive():
    a = [10, 11, 12, 13]
    b = [1, 2, 3, 4]
    es = effect_size(a, b)
    assert es['cliffs_delta'] == pytest.approx(1.0)  # a strictly > b
    assert es['cohens_d'] > 0


def test_cliffs_delta_sign_negative():
    a = [1, 2, 3, 4]
    b = [10, 11, 12, 13]
    es = effect_size(a, b)
    assert es['cliffs_delta'] == pytest.approx(-1.0)
    assert es['cohens_d'] < 0


def test_cliffs_delta_no_difference():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    es = effect_size(a, b)
    assert es['cliffs_delta'] == pytest.approx(0.0)
    assert es['cohens_d'] == pytest.approx(0.0)


def test_effect_size_empty_group():
    es = effect_size([], [1, 2, 3])
    assert math.isnan(es['cliffs_delta'])
    assert math.isnan(es['cohens_d'])


def test_effect_size_zero_pooled_sd():
    # Both groups constant but different -> Cohen's d undefined (NaN),
    # Cliff's delta well-defined (+1).
    es = effect_size([5, 5, 5], [2, 2, 2])
    assert es['cliffs_delta'] == pytest.approx(1.0)
    assert math.isnan(es['cohens_d'])


def test_effect_size_drops_nans():
    es = effect_size([10, np.nan, 12], [1, 2, np.nan])
    assert es['cliffs_delta'] == pytest.approx(1.0)
