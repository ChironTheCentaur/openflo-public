"""Tests for the negative-binomial differential-abundance GLM
(openflo.diffexp.differential_abundance)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from openflo.diffexp import differential_abundance


def _synthetic_counts(seed=0, n_per=5, libsize=10000):
    """5 clusters across two groups; cluster 'up' is enriched in B,
    'down' depleted, the rest unchanged. Counts are NB-dispersed."""
    rng = np.random.default_rng(seed)
    # baseline proportions
    base = {'up': 0.08, 'down': 0.20, 'flat1': 0.30, 'flat2': 0.25,
            'flat3': 0.17}
    effect = {'up': 2.2, 'down': 0.45}     # B/A fold-change on proportion
    samples, groups = [], []
    cols = []
    for grp in ('A', 'B'):
        for i in range(n_per):
            props = {}
            for c, p in base.items():
                fc = effect.get(c, 1.0) if grp == 'B' else 1.0
                props[c] = p * fc
            tot = sum(props.values())
            col = {}
            for c, p in props.items():
                mu = libsize * p / tot
                # NB-ish noise via gamma-poisson
                col[c] = rng.poisson(rng.gamma(20, mu / 20))
            samples.append(col)
            groups.append(grp)
            cols.append(f'{grp}{i}')
    counts = pd.DataFrame(samples, index=cols).T  # clusters × samples
    return counts, np.array(groups)


def test_da_detects_up_and_down_clusters():
    counts, group = _synthetic_counts()
    rows = differential_abundance(counts, group)
    by = {r['cluster']: r for r in rows}
    # 'up' enriched in B → positive log2FC, significant.
    assert by['up']['log2fc'] > 0.5
    assert by['up']['p_adj'] < 0.05
    # 'down' depleted in B → negative log2FC, significant.
    assert by['down']['log2fc'] < -0.5
    assert by['down']['p_adj'] < 0.05


def test_da_flat_clusters_not_significant():
    counts, group = _synthetic_counts()
    rows = differential_abundance(counts, group)
    by = {r['cluster']: r for r in rows}
    for c in ('flat1', 'flat2', 'flat3'):
        assert by[c]['p_adj'] > 0.05, (c, by[c]['p_adj'])


def test_da_row_fields_and_sorting():
    counts, group = _synthetic_counts()
    rows = differential_abundance(counts, group)
    for r in rows:
        assert {'cluster', 'log2fc', 'prop_a', 'prop_b', 'z', 'p', 'p_adj',
                'dispersion', 'n_a', 'n_b'} <= set(r)
        assert 0 <= r['prop_a'] <= 1 and 0 <= r['prop_b'] <= 1
    # Sorted ascending by p-value.
    ps = [r['p'] for r in rows if np.isfinite(r['p'])]
    assert ps == sorted(ps)


def test_da_uses_library_size_offset():
    # Same proportions but very different library sizes across groups → the
    # offset must absorb it, so no spurious abundance difference.
    rng = np.random.default_rng(1)
    rows_counts, groups = [], []
    for grp, lib in (('A', 5000), ('A', 5200), ('B', 50000), ('B', 52000)):
        props = [0.5, 0.3, 0.2]
        rows_counts.append([rng.poisson(lib * p) for p in props])
        groups.append(grp)
    counts = pd.DataFrame(rows_counts, index=['s0', 's1', 's2', 's3']).T
    counts.index = ['c0', 'c1', 'c2']
    rows = differential_abundance(counts, np.array(groups))
    # No cluster should look differentially abundant (10× library size, equal
    # proportions).
    assert all(r['p_adj'] > 0.05 for r in rows if np.isfinite(r['p_adj']))


def test_da_two_groups_required():
    counts = pd.DataFrame(np.ones((3, 3)))
    import pytest
    with pytest.raises(ValueError, match="2 groups"):
        differential_abundance(counts, np.array(['A', 'B', 'C']))


def test_da_accepts_ndarray_and_names():
    Y = np.array([[100, 110, 50, 55], [200, 190, 240, 250]], dtype=float)
    rows = differential_abundance(Y, ['A', 'A', 'B', 'B'],
                                  cluster_names=['x', 'y'])
    assert {r['cluster'] for r in rows} == {'x', 'y'}


def test_rows_carry_fitted_group_direction():
    """Each row records group_a/group_b = the group order as it first appears in
    `group` (levels[0]/[1]), with prop_a and log2FC (b-vs-a) relative to THAT
    order. The diff-abundance window must label from these, not sample-load order
    — otherwise the %-headers and the log2FC sign flip."""
    counts = pd.DataFrame(
        {'s1': [50, 10], 's2': [55, 12], 's3': [10, 50], 's4': [12, 55]},
        index=['popX', 'popY'])
    rows = differential_abundance(counts, ['stim', 'stim', 'ctrl', 'ctrl'])
    assert rows and rows[0]['group_a'] == 'stim' and rows[0]['group_b'] == 'ctrl'
    byp = {r['cluster']: r for r in rows}
    # popX is high in stim (group_a) → prop_a > prop_b, and log2FC (ctrl-vs-stim) < 0
    assert byp['popX']['prop_a'] > byp['popX']['prop_b']
    assert byp['popX']['log2fc'] < 0
