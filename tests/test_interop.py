"""Tests for openflo.interop — sample-distance / MDS QC + AnnData export."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.interop import (
    mds_embed,
    sample_distance_matrix,
    to_anndata,
)


def _sample(loc, n=3000, seed=0, markers=('M1', 'M2', 'M3')):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({m: rng.normal(loc, 1.0, n) for m in markers})


def test_distance_matrix_symmetry_and_zero_diag():
    samples = {'a': _sample(0, seed=1), 'b': _sample(0, seed=2),
               'c': _sample(5, seed=3)}
    names, D = sample_distance_matrix(samples, ['M1', 'M2', 'M3'])
    assert names == ['a', 'b', 'c']
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)


def test_distance_separates_shifted_sample():
    samples = {'a': _sample(0, seed=1), 'b': _sample(0, seed=2),
               'c': _sample(8, seed=3)}
    _, D = sample_distance_matrix(samples, ['M1', 'M2', 'M3'])
    # a–b (same distribution) much closer than a–c (shifted by 8 SD).
    assert D[0, 1] < D[0, 2]
    assert D[0, 2] > 1.5
    assert D[0, 2] > 10 * D[0, 1]           # clearly separated


def test_distance_single_sample():
    names, D = sample_distance_matrix({'only': _sample(0)}, ['M1'])
    assert names == ['only']
    assert D.shape == (1, 1)


def test_mds_separates_two_batches():
    # Two batches of similar samples, far from each other → MDS splits them.
    samples = {}
    for i in range(3):
        samples[f'b1_{i}'] = _sample(0, seed=10 + i)
    for i in range(3):
        samples[f'b2_{i}'] = _sample(10, seed=20 + i)
    names, D = sample_distance_matrix(samples, ['M1', 'M2', 'M3'])
    xy = mds_embed(D)
    assert xy.shape == (6, 2)
    b1 = xy[:3].mean(0)
    b2 = xy[3:].mean(0)
    # Between-batch centroid gap exceeds within-batch spread.
    within = max(np.linalg.norm(xy[:3] - b1, axis=1).max(),
                 np.linalg.norm(xy[3:] - b2, axis=1).max())
    assert np.linalg.norm(b1 - b2) > within


def test_mds_too_few():
    assert mds_embed(np.zeros((1, 1))).shape == (1, 2)


# ── AnnData export (optional dependency) ──────────────────────────────────────

def test_to_anndata_roundtrip():
    ad = pytest.importorskip("anndata")
    samples = {'s1': _sample(0, n=100, seed=1).assign(leiden=[0] * 100),
               's2': _sample(3, n=80, seed=2).assign(leiden=[1] * 80)}
    adata = to_anndata(samples, ['M1', 'M2', 'M3'], obs_cols=['leiden'])
    assert isinstance(adata, ad.AnnData)
    assert adata.shape == (180, 3)
    assert list(adata.var_names) == ['M1', 'M2', 'M3']
    assert set(adata.obs['sample']) == {'s1', 's2'}
    assert 'leiden' in adata.obs.columns


def test_to_anndata_without_package_errors(monkeypatch):
    # Simulate anndata being absent → a clear, actionable error.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'anndata':
            raise ImportError("no anndata")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError, match="anndata"):
        to_anndata({'s': _sample(0, n=10)}, ['M1'])
