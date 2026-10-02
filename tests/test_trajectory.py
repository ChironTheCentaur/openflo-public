"""Tests for openflo.trajectory — geodesic pseudotime ordering."""
from __future__ import annotations

import numpy as np
import pytest

from openflo.trajectory import (
    compute_pseudotime,
    pseudotime_trends,
    robust_root,
)


def _arc(n=1500, seed=0):
    """Cells on a noisy 1-D curve parameterized by t∈[0,1]; a 'stem' marker
    decreases with t and a 'mature' marker increases — like CD34→CD11b."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, n))
    # 2-D curved manifold so geodesic ≠ euclidean (an arc).
    x = np.cos(t * np.pi)
    y = np.sin(t * np.pi)
    X = np.column_stack([x, y]) + rng.normal(0, 0.02, (n, 2))
    stem = (1 - t) * 100 + rng.normal(0, 3, n)      # high at t=0
    mature = t * 100 + rng.normal(0, 3, n)          # high at t=1
    return X, t, stem, mature


# ── robust_root ───────────────────────────────────────────────────────────────

def test_robust_root_high_end():
    X, t, stem, _ = _arc()
    root = robust_root(X, stem, high=True)
    # Root should be a low-t (stem-high) cell.
    assert t[root] < 0.15


def test_robust_root_low_end():
    X, t, stem, _ = _arc()
    root = robust_root(X, stem, high=False)
    assert t[root] > 0.85


def test_robust_root_ignores_single_outlier():
    X, t, stem, _ = _arc()
    stem = stem.copy()
    stem[1200] = 1e6                # a far-along outlier with absurd score
    root = robust_root(X, stem, high=True)
    # Medoid-of-extremes resists the lone spike → still a low-t root.
    assert t[root] < 0.2


# ── compute_pseudotime ────────────────────────────────────────────────────────

def test_pseudotime_monotone_along_manifold():
    X, t, stem, _ = _arc()
    pt, root = compute_pseudotime(X, stem, high=True, n_neighbors=15)
    assert pt.shape == (len(X),)
    assert pt.min() >= 0.0 and pt.max() <= 1.0
    # Geodesic pseudotime should track the true ordering t strongly.
    r = np.corrcoef(pt, t)[0, 1]
    assert r > 0.9, r
    assert pt[root] == pytest.approx(0.0)


def test_pseudotime_low_root_reverses():
    X, t, stem, _ = _arc()
    pt, _ = compute_pseudotime(X, stem, high=False, n_neighbors=15)
    # Rooted at the mature end → pseudotime anti-correlates with t.
    assert np.corrcoef(pt, t)[0, 1] < -0.9


def test_pseudotime_subsample_path_covers_all():
    X, t, stem, _ = _arc(n=800)
    # Force the subsample+propagate branch with a small cap.
    pt, _ = compute_pseudotime(X, stem, high=True, max_cells=200)
    assert pt.shape == (800,)
    assert np.isfinite(pt).all()
    assert np.corrcoef(pt, t)[0, 1] > 0.8


def test_pseudotime_empty():
    pt, root = compute_pseudotime(np.empty((0, 3)), np.empty(0))
    assert pt.shape == (0,)
    assert root == 0


def test_pseudotime_is_geodesic_not_euclidean_from_root():
    """On a near-closed C-shape, the two tips are euclidean-CLOSE but arc-FAR.
    Geodesic pseudotime gives the far tip a LARGE value; a plain
    euclidean-distance-from-root implementation would give it a small one (and a
    low corr). The semicircle _arc() can't distinguish the two — both monotone."""
    rng = np.random.default_rng(0)
    n = 1500
    t = np.sort(rng.uniform(0, 1, n))
    ang = np.deg2rad(-160.0 + 320.0 * t)         # C spanning 320°, tips ~0.68 apart
    X = np.column_stack([np.cos(ang), np.sin(ang)]) + rng.normal(0, 0.02, (n, 2))
    stem = (1 - t) * 100 + rng.normal(0, 3, n)   # high at t=0 → root at a tip
    pt, root = compute_pseudotime(X, stem, high=True, n_neighbors=15)
    assert t[root] < 0.15                         # rooted at the t≈0 tip
    # far tip (t≈1) is euclidean-near the root yet must be pseudotime-far (geodesic)
    root_xy, far_xy = X[root], X[t > 0.97].mean(axis=0)
    assert np.linalg.norm(far_xy - root_xy) < 1.0     # tips ARE euclidean-close
    assert pt[t > 0.97].mean() > 0.8                   # geodesic → large (euclid → ~0.3)
    assert np.corrcoef(pt, t)[0, 1] > 0.85             # tracks arc order


# ── pseudotime_trends ─────────────────────────────────────────────────────────

def test_trends_capture_opposing_markers():
    X, t, stem, mature = _arc()
    pt, _ = compute_pseudotime(X, stem, high=True)
    centers, means = pseudotime_trends(pt, np.column_stack([stem, mature]),
                                       n_bins=10)
    assert centers.shape == (10,)
    assert means.shape == (10, 2)
    # Stem marker falls, mature marker rises across pseudotime.
    stem_trend, mat_trend = means[:, 0], means[:, 1]
    assert stem_trend[0] > stem_trend[-1]
    assert mat_trend[0] < mat_trend[-1]


def test_trends_1d_marker_and_empty_bins():
    pt = np.array([0.0, 0.05, 0.1])         # all in the first couple of bins
    centers, means = pseudotime_trends(pt, np.array([1.0, 2.0, 3.0]),
                                       n_bins=10)
    assert means.shape == (10, 1)
    assert np.isnan(means[-1, 0])           # late bins empty → NaN


def test_pseudotime_trends_bin_means_and_centers():
    """pseudotime_trends returns the per-bin MEAN of each marker at evenly-spaced
    bin centers — the actual values, not just endpoint direction."""
    pt = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    marker = np.array([[0.0], [10.0], [20.0], [30.0], [40.0]])
    centers, trend = pseudotime_trends(pt, marker, n_bins=5)
    assert np.allclose(centers, [0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(trend[:, 0], [0.0, 10.0, 20.0, 30.0, 40.0])
