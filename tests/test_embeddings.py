"""Tests for the t-SNE / PHATE embedding methods on FlowSample."""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from openflo.pipeline import FlowSample


def _stub(n=400, seed=0, markers=('M1', 'M2', 'M3')):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n, len(markers))),
                   rng.normal(8, 1, (n, len(markers)))])
    df = pd.DataFrame(X, columns=pd.Index(list(markers)))
    s = types.SimpleNamespace(data=df, fluor_channels=list(markers))
    # run_tsne/run_phate call these instance helpers — bind them to the stub.
    s._embedding_input = types.MethodType(FlowSample._embedding_input, s)
    s._store_embedding = types.MethodType(FlowSample._store_embedding, s)
    return s


def test_run_tsne_writes_coords():
    s = _stub()
    FlowSample.run_tsne(s, perplexity=30, sample_n=10_000)
    assert 'TSNE1' in s.data.columns and 'TSNE2' in s.data.columns
    xy = s.data[['TSNE1', 'TSNE2']].dropna().to_numpy()
    assert len(xy) == len(s.data)              # all events embedded (no subsample)
    assert np.isfinite(xy).all()


def test_run_tsne_perplexity_clamped_small_n():
    s = _stub(n=8)                              # 16 events, perplexity must clamp
    FlowSample.run_tsne(s, perplexity=50)
    assert 'TSNE1' in s.data.columns
    assert s.data['TSNE1'].notna().all()


def test_run_tsne_too_few_events():
    s = types.SimpleNamespace(
        data=pd.DataFrame({'M1': [1.0, 2.0], 'M2': [1.0, 2.0]}),
        fluor_channels=['M1', 'M2'])
    s._embedding_input = types.MethodType(FlowSample._embedding_input, s)
    s._store_embedding = types.MethodType(FlowSample._store_embedding, s)
    FlowSample.run_tsne(s)
    assert 'TSNE1' not in s.data.columns        # skipped, not a crash


def test_run_phate_optional():
    import pytest
    pytest.importorskip("phate")
    s = _stub(n=300)
    FlowSample.run_phate(s, sample_n=10_000)
    assert 'PHATE1' in s.data.columns and 'PHATE2' in s.data.columns
