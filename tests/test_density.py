"""Tests for openflo.density.event_density (extracted from the density scatter)."""
from __future__ import annotations

import numpy as np

from openflo.density import event_density, kde_density


def _edges(lo, hi, n=256):
    return np.linspace(lo, hi, n + 1)


def test_event_density_shape_and_nonneg():
    rng = np.random.default_rng(0)
    xs = rng.normal(0, 1, 2000)
    ys = rng.normal(0, 1, 2000)
    z = event_density(xs, ys, _edges(-5, 5), _edges(-5, 5))
    assert z.shape == xs.shape
    assert np.all(z >= 0.0)
    assert np.all(np.isfinite(z))


def test_event_density_higher_in_dense_region():
    rng = np.random.default_rng(1)
    # a tight central blob plus a sparse uniform scatter
    core = rng.normal(0, 0.2, (4000, 2))
    sparse = rng.uniform(-5, 5, (400, 2))
    pts = np.vstack([core, sparse])
    xs, ys = pts[:, 0], pts[:, 1]
    z = event_density(xs, ys, _edges(-5, 5), _edges(-5, 5))
    # core events (first 4000) should be markedly denser than the sparse tail
    core_med = float(np.median(z[:4000]))
    sparse_med = float(np.median(z[4000:]))
    assert core_med > sparse_med * 3


def test_event_density_deterministic():
    rng = np.random.default_rng(2)
    xs = rng.normal(0, 1, 1000)
    ys = rng.normal(0, 1, 1000)
    e = _edges(-4, 4)
    z1 = event_density(xs, ys, e, e)
    z2 = event_density(xs, ys, e, e)
    assert np.array_equal(z1, z2)


def test_kde_density_no_subsample():
    rng = np.random.default_rng(0)
    xs = rng.normal(0, 1, 800)
    ys = rng.normal(0, 1, 800)
    xs_d, ys_d, z, n_src = kde_density(xs, ys)
    assert n_src == 800                              # under both caps
    assert xs_d.size == 800 and z.shape == xs_d.shape
    assert np.all(z >= 0) and np.all(np.isfinite(z))


def test_kde_density_subsamples_source_and_display():
    rng = np.random.default_rng(1)
    n = 2000
    xs = rng.normal(0, 1, n)
    ys = rng.normal(0, 1, n)
    # tiny caps force subsampling on both source and display
    xs_d, ys_d, z, n_src = kde_density(xs, ys, max_src=200, max_display=500)
    assert n_src == 200 and n_src < n                # source capped
    assert xs_d.size == 500 and z.shape == (500,)    # display capped


def test_kde_density_higher_in_dense_region():
    rng = np.random.default_rng(2)
    core = rng.normal(0, 0.2, (1500, 2))
    sparse = rng.uniform(-5, 5, (300, 2))
    pts = np.vstack([core, sparse])
    xs_d, ys_d, z, _n = kde_density(pts[:, 0], pts[:, 1])
    # the densest displayed point should sit near the core, not the sparse rim
    peak = (xs_d[z.argmax()], ys_d[z.argmax()])
    assert abs(peak[0]) < 1.0 and abs(peak[1]) < 1.0
