"""Pure tests for the one-click cross-group comparison + volcano helpers
(openflo.stats.compare_all_features / volcano_data)."""
from __future__ import annotations

import numpy as np

from openflo.stats import compare_all_features, volcano_data

rng = np.random.default_rng(0)


def _two_group(a_mean, b_mean, n=6, sd=0.3):
    return {'A': list(rng.normal(a_mean, sd, n)),
            'B': list(rng.normal(b_mean, sd, n))}


def test_compare_all_runs_every_feature_and_bh_corrects():
    feats = {
        'pop_up':   _two_group(2.0, 8.0),    # big B>A change
        'pop_down': _two_group(8.0, 2.0),    # big A>B change
        'pop_flat': _two_group(5.0, 5.0),    # no change
    }
    res = compare_all_features(feats)
    assert len(res) == 3
    assert {r['feature'] for r in res} == set(feats)
    # every feature carries adjusted p + stars + a 2-group log2FC
    assert all('p_adj' in r and 'stars' in r and np.isfinite(r['effect'])
               for r in res)
    by = {r['feature']: r for r in res}
    assert by['pop_up']['effect'] > 1.0        # B/A up  → positive log2FC
    assert by['pop_down']['effect'] < -1.0     # B/A down → negative log2FC
    assert abs(by['pop_flat']['effect']) < 0.5
    # the changed populations are more significant than the flat one
    assert by['pop_up']['p_adj'] < by['pop_flat']['p_adj']


def test_compare_all_sorted_by_adjusted_p():
    res = compare_all_features({'a': _two_group(1, 9), 'b': _two_group(5, 5)})
    padj = [r['p_adj'] for r in res]
    assert padj == sorted(padj)                # most-significant first


def test_compare_all_multigroup_effect_is_nan():
    res = compare_all_features({'p': {'A': [1, 2, 3], 'B': [4, 5, 6],
                                      'C': [7, 8, 9]}})
    assert res[0]['n_groups'] == 3
    assert np.isnan(res[0]['effect'])          # log2FC undefined for >2 groups
    assert res[0]['test'] in ('Kruskal-Wallis', 'One-way ANOVA', None)


def test_volcano_data_flags_significant():
    feats = {f'up{i}': _two_group(2.0, 8.0) for i in range(4)}
    feats['flat'] = _two_group(5.0, 5.0)
    pts = volcano_data(compare_all_features(feats), alpha=0.05, effect_cut=1.0)
    assert {p['feature'] for p in pts} == set(feats)
    sig = {p['feature'] for p in pts if p['significant']}
    assert 'flat' not in sig                    # no effect → not significant
    assert any(f.startswith('up') for f in sig)
    # y axis is -log10(p_adj) ≥ 0; x is the log2FC
    assert all(p['y'] >= 0 for p in pts)


def test_volcano_skips_multigroup_nan_effect():
    res = compare_all_features({'p': {'A': [1, 2, 3], 'B': [4, 5, 6],
                                      'C': [7, 8, 9]}})
    assert volcano_data(res) == []              # no finite 2-group effect


def test_compare_all_window_builds(monkeypatch):
    """The CompareAllWindow (table + volcano) constructs for both the 2-group
    (volcano populated) and >2-group (graceful 'needs 2 groups') cases."""
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    import pytest
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except Exception:
        pytest.skip('no Tk display')
    root.withdraw()
    try:
        from openflo.ui_diff import CompareAllWindow
        res2 = compare_all_features({'up': _two_group(2, 8),
                                     'flat': _two_group(5, 5)})
        w2 = CompareAllWindow(root, res2, ['A', 'B'], '%Parent')
        w2.update_idletasks()
        assert w2._two and len(w2._volcano) == 2
        assert len(w2._flat_rows()) == 2
        res3 = compare_all_features({'p': {'A': [1, 2, 3], 'B': [4, 5, 6],
                                           'C': [7, 8, 9]}})
        w3 = CompareAllWindow(root, res3, ['A', 'B', 'C'], '%Parent')
        w3.update_idletasks()
        assert not w3._two and w3._volcano == []
    finally:
        root.destroy()
