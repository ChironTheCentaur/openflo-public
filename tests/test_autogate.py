"""Automated density-based gating helpers.

auto_threshold (1-D valley / Otsu) and auto_polygon_gate (2-D density
contour) are pure; tested on synthetic bimodal / single-blob data.
auto_singlet_gate (FSC-A/FSC-H ratio band) and gmm_ellipse_gates
(BIC-selected Gaussian-mixture ellipses) are checked against synthetic
data with known structure.
"""
import numpy as np
import pandas as pd

from openflo.pipeline import (
    _otsu_threshold,
    _polygon_area,
    auto_polygon_gate,
    auto_singlet_gate,
    auto_threshold,
    gate_to_mask,
    gmm_ellipse_gates,
)

# ── auto_threshold ───────────────────────────────────────────────────────────

def test_auto_threshold_bimodal_valley():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(100, 12, 4000),
                        rng.normal(1000, 80, 4000)])
    thr = auto_threshold(v)
    # Valley should sit between the two modes (closer to the tighter peak).
    assert 120 < thr < 900


def test_auto_threshold_unimodal_uses_otsu():
    rng = np.random.default_rng(1)
    v = rng.normal(500, 50, 4000)
    thr = auto_threshold(v)
    # Otsu on a single Gaussian lands somewhere within the bulk.
    assert 350 < thr < 650


def test_auto_threshold_too_few():
    assert auto_threshold([1.0, 2.0, 3.0]) is None


def test_otsu_threshold_separates_two_masses():
    centers = np.arange(10.0)
    hist = np.array([50, 50, 50, 0, 0, 0, 0, 50, 50, 50], dtype=float)
    t = _otsu_threshold(hist, centers)
    assert 2.0 <= t <= 7.0


# ── _polygon_area ────────────────────────────────────────────────────────────

def test_polygon_area_unit_square():
    assert _polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == 1.0
    assert _polygon_area([[0, 0], [1, 0]]) == 0.0   # degenerate


# ── auto_polygon_gate ────────────────────────────────────────────────────────

def test_auto_polygon_gate_wraps_main_blob():
    rng = np.random.default_rng(2)
    # One tight blob at (1000, 1000) + sparse uniform background.
    blob_x = rng.normal(1000, 40, 6000)
    blob_y = rng.normal(1000, 40, 6000)
    bg_x = rng.uniform(0, 2000, 1500)
    bg_y = rng.uniform(0, 2000, 1500)
    x = np.concatenate([blob_x, bg_x])
    y = np.concatenate([blob_y, bg_y])
    verts = auto_polygon_gate(x, y)
    assert verts is not None
    v = np.asarray(verts)
    assert v.shape[1] == 2 and len(v) >= 3
    # The polygon should sit around the blob centre, not span the whole range.
    cx, cy = v[:, 0].mean(), v[:, 1].mean()
    assert 800 < cx < 1200 and 800 < cy < 1200
    assert (v[:, 0].max() - v[:, 0].min()) < 800   # tight, not the full 0..2000


def test_auto_polygon_gate_too_few():
    assert auto_polygon_gate([1.0, 2.0], [1.0, 2.0]) is None


def test_auto_polygon_gate_vertex_cap():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(0, 1, 5000)
    verts = auto_polygon_gate(x, y, max_verts=20)
    assert verts is not None and len(verts) <= 20


# ── auto_singlet_gate ────────────────────────────────────────────────────────

def _singlet_data(n=4000, slope=2.0, seed=0):
    """Singlets on area = slope*height + noise; doublets at ~1.7x area."""
    rng = np.random.default_rng(seed)
    height = rng.uniform(2e4, 8e4, n)
    area = slope * height * rng.normal(1.0, 0.02, n)        # tight diagonal
    n_dbl = int(0.12 * n)                                   # 12% doublets
    h_d = rng.uniform(2e4, 8e4, n_dbl)
    a_d = slope * h_d * rng.normal(1.7, 0.05, n_dbl)        # above the diagonal
    return (np.concatenate([area, a_d]),
            np.concatenate([height, h_d]),
            n, n_dbl)


def test_singlet_gate_basic_shape():
    area, height, _, _ = _singlet_data()
    verts, q = auto_singlet_gate(area, height)
    assert verts is not None and q is not None
    assert len(verts) == 4 and all(len(v) == 2 for v in verts)
    assert q['slope'] > 0
    assert 0.0 < q['frac_kept'] <= 1.0
    assert q['ratio_cv'] >= 0.0


def test_singlet_gate_excludes_doublets():
    area, height, n_singlet, _ = _singlet_data()
    verts, q = auto_singlet_gate(area, height)
    df = pd.DataFrame({'FSC-A': area, 'FSC-H': height})
    gate = {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'FSC-H',
            'vertices': verts}
    mask = gate_to_mask(gate, df)
    assert mask[:n_singlet].mean() > 0.9         # keep almost all singlets
    assert mask[n_singlet:].mean() < 0.15        # reject almost all doublets
    assert abs(q['frac_kept'] - mask.mean()) < 0.05


def test_singlet_gate_too_little_data():
    assert auto_singlet_gate(np.array([1.0, 2.0]),
                             np.array([1.0, 1.0])) == (None, None)


def test_singlet_gate_handles_nonpositive_height():
    rng = np.random.default_rng(1)
    h = rng.uniform(1e4, 5e4, 500)
    a = 2.0 * h * rng.normal(1.0, 0.02, 500)    # realistic diagonal spread
    h[:50] = 0.0                       # invalid heights filtered, not crashing
    verts, q = auto_singlet_gate(a, h)
    assert verts is not None
    assert abs(q['slope'] - 2.0) < 0.1


# ── gmm_ellipse_gates ────────────────────────────────────────────────────────

def _two_blobs(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal([1e4, 1e4], [[2e6, 0], [0, 2e6]], n)
    b = rng.multivariate_normal([6e4, 6e4], [[3e6, 0], [0, 3e6]], n)
    pts = np.vstack([a, b])
    return pts[:, 0], pts[:, 1]


def test_gmm_finds_two_populations():
    x, y = _two_blobs()
    gates = gmm_ellipse_gates(x, y, max_components=6)
    assert len(gates) == 2
    for gate, info in gates:
        assert gate['kind'] == 'ellipsoid'
        assert np.shape(gate['mean']) == (2,)
        assert np.shape(gate['cov']) == (2, 2)
        assert gate['distance_sq'] > 0
        assert 0.0 < info['weight'] <= 1.0
        assert info['n_components'] == 2
        assert info['separation'] is None or info['separation'] > 2.0


def test_gmm_means_match_blobs():
    x, y = _two_blobs()
    gates = gmm_ellipse_gates(x, y, max_components=6)
    means = sorted(g['mean'][0] for g, _ in gates)
    assert abs(means[0] - 1e4) < 5e3
    assert abs(means[1] - 6e4) < 5e3


def test_gmm_ellipse_covers_its_population():
    x, y = _two_blobs()
    gates = gmm_ellipse_gates(x, y, max_components=6, coverage=0.90)
    df = pd.DataFrame({'X': x, 'Y': y})
    union = np.zeros(len(df), dtype=bool)
    for gate, _ in gates:
        union |= gate_to_mask(dict(gate, x_channel='X', y_channel='Y'), df)
    assert union.mean() > 0.8


def test_gmm_single_population():
    rng = np.random.default_rng(2)
    pts = rng.multivariate_normal([0, 0], [[1, 0], [0, 1]], 2000)
    gates = gmm_ellipse_gates(pts[:, 0], pts[:, 1], max_components=5)
    assert len(gates) == 1
    assert gates[0][1]['separation'] is None      # nothing to separate from


def test_gmm_too_little_data():
    assert gmm_ellipse_gates(np.arange(10.0), np.arange(10.0)) == []


def test_gmm_min_weight_drops_tiny_components():
    x, y = _two_blobs(n=3000)
    rng = np.random.default_rng(5)
    tiny = rng.multivariate_normal([3e4, 9e4], [[1e5, 0], [0, 1e5]], 30)
    x2 = np.concatenate([x, tiny[:, 0]])
    y2 = np.concatenate([y, tiny[:, 1]])
    gates = gmm_ellipse_gates(x2, y2, max_components=6, min_weight=0.05)
    for _, info in gates:
        assert info['weight'] >= 0.05
