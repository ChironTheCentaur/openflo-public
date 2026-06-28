"""Tests for openflo.stats — group comparison + Prism-ready table shaping."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openflo.stats import (
    _benjamini_hochberg,
    compare_groups,
    group_kde,
    p_to_stars,
    to_prism_column,
    to_prism_grouped,
)

# ── p_to_stars ────────────────────────────────────────────────────────────────

def test_p_to_stars_thresholds():
    assert p_to_stars(0.5) == 'ns'
    assert p_to_stars(0.04) == '*'
    assert p_to_stars(0.004) == '**'
    assert p_to_stars(0.0004) == '***'
    assert p_to_stars(1e-9) == '****'
    assert p_to_stars(float('nan')) == ''
    assert p_to_stars(None) == ''


# ── compare_groups: two groups ────────────────────────────────────────────────

def test_compare_two_groups_mwu_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 20)
    b = rng.normal(20, 1, 20)
    res = compare_groups({'A': a, 'B': b})
    assert res['test'] == 'Mann-Whitney U'
    assert res['n_groups'] == 2
    assert res['p'] < 1e-3
    assert res['posthoc'] == []
    assert res['groups']['A']['n'] == 20
    assert abs(res['groups']['B']['median'] - 20) < 1.0


def test_compare_two_groups_parametric_ttest():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0.2, 1, 30)
    res = compare_groups({'A': a, 'B': b}, parametric=True)
    assert res['test'] == 'Welch t-test'
    assert np.isfinite(res['p'])


def test_compare_two_groups_no_difference_is_ns():
    rng = np.random.default_rng(2)
    a = rng.normal(5, 1, 25)
    b = rng.normal(5, 1, 25)
    res = compare_groups({'A': a, 'B': b})
    assert res['p'] > 0.05
    assert p_to_stars(res['p']) == 'ns'


# ── compare_groups: >2 groups ─────────────────────────────────────────────────

def test_compare_multi_group_kruskal_and_posthoc():
    rng = np.random.default_rng(3)
    groups = {'D3': rng.normal(5, 1, 15),
              'D6': rng.normal(10, 1, 15),
              'D9': rng.normal(20, 1, 15)}
    res = compare_groups(groups)
    assert res['test'] == 'Kruskal-Wallis'
    assert res['n_groups'] == 3
    assert res['p'] < 1e-3
    assert len(res['posthoc']) == 3           # 3 pairwise comparisons
    for pr in res['posthoc']:
        assert {'a', 'b', 'p', 'p_adj'} <= set(pr)
    d3d9 = next(pr for pr in res['posthoc']
                if {pr['a'], pr['b']} == {'D3', 'D9'})
    assert d3d9['p_adj'] < 0.05


def test_compare_multi_group_anova_parametric():
    rng = np.random.default_rng(4)
    groups = {'A': rng.normal(0, 1, 12), 'B': rng.normal(0, 1, 12),
              'C': rng.normal(3, 1, 12)}
    res = compare_groups(groups, parametric=True)
    assert res['test'] == 'One-way ANOVA'
    assert len(res['posthoc']) == 3


def test_compare_one_group_returns_no_test():
    res = compare_groups({'only': [1.0, 2.0, 3.0]})
    assert res['test'] is None
    assert res['n_groups'] == 1
    assert not np.isfinite(res['p'])


def test_compare_drops_empty_and_nan_groups():
    res = compare_groups({'A': [1.0, 2.0, np.nan], 'B': [], 'C': [3.0, 4.0]})
    assert res['n_groups'] == 2
    assert res['groups']['A']['n'] == 2


# ── _benjamini_hochberg ───────────────────────────────────────────────────────

def test_bh_monotone_and_bounded():
    adj = _benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    assert all(0.0 <= q <= 1.0 for q in adj)
    assert adj[-1] >= adj[0]


def test_bh_passes_through_nan():
    adj = _benjamini_hochberg([0.01, float('nan'), 0.04])
    assert np.isnan(adj[1])
    assert np.isfinite(adj[0]) and np.isfinite(adj[2])


# ── to_prism_column ───────────────────────────────────────────────────────────

def test_prism_column_pads_ragged_groups():
    df = to_prism_column({'Stim': [1.0, 2.0, 3.0], 'Ctrl': [4.0, 5.0]})
    assert list(df.columns) == ['Stim', 'Ctrl']
    assert len(df) == 3                          # padded to the longest group
    assert df['Stim'].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(df['Ctrl'].iloc[2])         # blank pad cell


def test_prism_column_empty():
    df = to_prism_column({})
    assert df.empty


# ── to_prism_grouped ──────────────────────────────────────────────────────────

def _tidy():
    recs = []
    for day in ('D3', 'D6'):
        for cond in ('Stim', 'Ctrl'):
            for rep, val in enumerate([10.0, 12.0]):
                recs.append({'day': day, 'cond': cond, 'value': val + rep})
    return pd.DataFrame(recs)


def test_prism_grouped_shape_and_multiindex():
    g = to_prism_grouped(_tidy(), 'day', 'cond', 'value')
    assert list(g.index) == ['D3', 'D6']
    assert isinstance(g.columns, pd.MultiIndex)
    assert g.shape == (2, 4)                      # 2 conditions × 2 replicates
    assert set(g.columns.get_level_values(0)) == {'Stim', 'Ctrl'}
    assert g.loc['D3', ('Stim', 1)] == 10.0


def test_prism_grouped_pads_uneven_replicates():
    df = _tidy().drop(index=0)                    # drop one replicate
    g = to_prism_grouped(df, 'day', 'cond', 'value')
    assert g.shape[0] == 2
    assert g.isna().to_numpy().any()


# ── group_kde ─────────────────────────────────────────────────────────────────

def test_group_kde_peaks_track_group_means():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    b = rng.normal(8, 1, 2000)
    x, dens = group_kde({'A': a, 'B': b}, gridsize=256)
    assert x.shape == (256,)
    assert set(dens) == {'A', 'B'}
    # Each density peaks near its group's mean.
    assert abs(x[np.argmax(dens['A'])] - 0) < 1.0
    assert abs(x[np.argmax(dens['B'])] - 8) < 1.0


def test_group_kde_handles_degenerate_groups():
    x, dens = group_kde({'one': [5.0], 'flat': [3.0, 3.0, 3.0],
                         'real': [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert np.allclose(dens['one'], 0.0)         # <2 points → zero curve
    assert np.allclose(dens['flat'], 0.0)        # zero variance → zero curve
    assert dens['real'].max() > 0                # a genuine density


def test_group_kde_empty():
    x, dens = group_kde({})
    assert x.size == 0 and dens == {}
