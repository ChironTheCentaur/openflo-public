"""Tests for the dimensionality-reduction comparison helper."""
from __future__ import annotations

import numpy as np

from openflo.dr_compare import available_methods, run_embeddings


def _two_blobs(n: int = 150, d: int = 8, seed: int = 0) -> np.ndarray:
    """Small two-blob synthetic matrix, (n, d) floats."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(loc=-2.0, scale=0.5, size=(half, d))
    b = rng.normal(loc=+2.0, scale=0.5, size=(n - half, d))
    return np.vstack([a, b])


def test_available_methods_includes_core_backends():
    methods = available_methods()
    # t-SNE (sklearn) and UMAP (umap-learn) are core dependencies.
    assert "tsne" in methods
    assert "umap" in methods


def test_run_embeddings_tsne_shape():
    X = _two_blobs()
    out = run_embeddings(X, methods=("tsne",))
    assert set(out.keys()) == {"coords", "index", "skipped"}
    assert "tsne" in out["coords"]
    assert out["coords"]["tsne"].shape == (len(X), 2)
    assert out["index"].shape == (len(X),)
    assert out["skipped"] == []


def test_missing_backend_skipped_gracefully():
    X = _two_blobs(n=60)
    out = run_embeddings(X, methods=("tsne", "definitely_not_a_backend"))
    # The real backend still runs...
    assert "tsne" in out["coords"]
    # ...and the bogus one is recorded as skipped, not crashed.
    skipped_names = [name for name, _reason in out["skipped"]]
    assert "definitely_not_a_backend" in skipped_names
    assert "definitely_not_a_backend" not in out["coords"]


def test_subsample_respects_max_points():
    X = _two_blobs(n=150)
    out = run_embeddings(X, methods=("tsne",), max_points=50, seed=1)
    assert out["index"].shape == (50,)
    assert out["coords"]["tsne"].shape == (50, 2)
    # Subsample indices are a valid subset of the original rows.
    assert out["index"].min() >= 0
    assert out["index"].max() < len(X)
    assert len(np.unique(out["index"])) == 50
