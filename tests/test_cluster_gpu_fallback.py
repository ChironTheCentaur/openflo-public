"""FlowSample.cluster's GPU branch: admission, fallback, and the glue code.

None of this ran in the suite (``_cluster_gpu`` 4% covered): CI has no RAPIDS,
so ``GPU_CLUSTERING_AVAILABLE`` is False and the branch is skipped. The
docstring makes three promises that can be checked without a GPU:

* any GPU failure falls back to CPU PhenoGraph for the same call, and says so;
* ``reproducible=True`` never takes the GPU path (cuGraph Louvain is unseeded);
* the result is always written.

The last test runs ``_cluster_gpu`` against numpy/pandas/sklearn/igraph
stand-ins for cupy/cudf/cuML/cuGraph. It pins OpenFlo's OWN glue (drop the
self-neighbour, build the edge list, put labels back in vertex order); it
says nothing about real RAPIDS behaviour.
"""
import contextlib
import io
import logging
import types

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as P
from openflo.pipeline import FlowSample

CH = ['M0-A', 'M1-A', 'M2-A']


def _blobs(n_per=150, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.4, (n_per, 3)) for c in (0.0, 8.0, 16.0)])
    return pd.DataFrame(X, columns=CH), np.repeat([0, 1, 2], n_per)


def _pure(lab, truth):
    for c in np.unique(lab):
        sub = truth[lab == c]
        if np.bincount(sub).max() / sub.size < 0.98:
            return False
    return True


@pytest.fixture
def gpu_claimed(monkeypatch):
    """Pretend RAPIDS is importable and VRAM is unknown (admits the GPU)."""
    monkeypatch.setattr(P, 'GPU_CLUSTERING_AVAILABLE', True)
    monkeypatch.setattr(P, '_vram_free_gb', lambda: None)


@pytest.fixture
def fake_phenograph(monkeypatch):
    """The CPU path's phenograph.cluster, replaced by a deterministic stand-in
    (nearest blob centre). These tests check the GPU admission and fallback
    logic, not PhenoGraph; the real unseeded Louvain is non-deterministic
    (a community of <= 10 events goes to -1, which flaked `lab >= 0`) and
    opens a cpu_count()-sized Pool on every call."""
    import phenograph
    calls = []

    def fake(data, **kwargs):
        calls.append(len(data))
        x = np.asarray(data, dtype=float)[:, 0]
        lab = np.argmin(np.abs(x[:, None] - np.array([0.0, 8.0, 16.0])[None, :]), axis=1)
        return lab, None, 0.5
    monkeypatch.setattr(phenograph, 'cluster', fake)
    return calls


def _cluster(df, **kw):
    s = FlowSample.from_dataframe(df.copy())
    with contextlib.redirect_stdout(io.StringIO()):     # phenograph is chatty
        s.cluster(channels=CH, k=15, **kw)
    return s


def test_gpu_failure_falls_back_to_cpu_and_says_so(gpu_claimed, fake_phenograph,
                                                    monkeypatch, caplog):
    calls = []

    def boom(self, X, k):
        calls.append(X.shape)
        raise RuntimeError('CUDA out of memory (simulated)')
    monkeypatch.setattr(FlowSample, '_cluster_gpu', boom)
    df, truth = _blobs()
    with caplog.at_level(logging.INFO, logger='openflo.pipeline'):
        s = _cluster(df, reproducible=False)
    assert calls == [(len(df), 3)]                      # GPU was attempted
    msgs = [r.getMessage() for r in caplog.records]
    assert any('GPU clustering failed' in m and 'RuntimeError' in m
               and 'CPU fallback' in m for m in msgs), msgs
    assert any(m.rstrip().endswith('[CPU]') for m in msgs), msgs
    lab = s.data['cluster'].to_numpy()
    assert fake_phenograph == [len(df)]                 # the CPU path ran
    assert (lab >= 0).all() and _pure(lab, truth)       # its result written


def test_reproducible_never_touches_the_gpu(gpu_claimed, monkeypatch):
    # Record rather than raise: cluster() catches any GPU exception and falls
    # back to CPU, so an exception here would be swallowed and prove nothing.
    calls = []
    monkeypatch.setattr(FlowSample, '_cluster_gpu',
                        lambda self, X, k: calls.append(1))
    df, truth = _blobs()
    s = _cluster(df, reproducible=True, use_gpu=True)
    assert calls == [], 'reproducible=True took the unseeded GPU path'
    assert _pure(s.data['cluster'].to_numpy(), truth)


def test_use_gpu_false_and_low_vram_stay_on_cpu(gpu_claimed, fake_phenograph,
                                                monkeypatch):
    calls = []
    monkeypatch.setattr(FlowSample, '_cluster_gpu',
                        lambda self, X, k: calls.append(1))
    df, _ = _blobs(n_per=60)
    _cluster(df, reproducible=False, use_gpu=False)
    monkeypatch.setattr(P, '_vram_free_gb', lambda: 0.2)
    _cluster(df, reproducible=False, use_gpu='auto', vram_admission_gb=1.0)
    assert calls == []
    assert fake_phenograph == [len(df), len(df)]         # both stayed on CPU


def _fake_kit():
    from sklearn.neighbors import NearestNeighbors

    cupy = types.SimpleNamespace(
        asarray=lambda x, dtype=None: np.asarray(x, dtype=dtype),
        float32=np.float32, int32=np.int32, repeat=np.repeat, arange=np.arange)

    class CuNN:
        def __init__(self, n_neighbors):
            self._nn = NearestNeighbors(n_neighbors=n_neighbors)

        def fit(self, X):
            self._nn.fit(X)
            return self

        def kneighbors(self, X):
            return self._nn.kneighbors(X)

    class Graph:
        def from_cudf_edgelist(self, df, source, destination, renumber):
            self.edges = df[[source, destination]].to_numpy()

    def louvain(G):
        import igraph as ig
        n = int(G.edges.max()) + 1
        assert not (G.edges[:, 0] == G.edges[:, 1]).any(), 'self-loops kept'
        g = ig.Graph(n=n, edges=G.edges.tolist(), directed=False).simplify()
        memb = np.asarray(g.community_multilevel().membership)
        # cuGraph does not promise vertex order; return it shuffled.
        order = np.random.default_rng(0).permutation(n)
        return pd.DataFrame({'vertex': order, 'partition': memb[order]}), 0.5

    return {'cupy': cupy, 'cudf': types.SimpleNamespace(DataFrame=pd.DataFrame),
            'cugraph': types.SimpleNamespace(Graph=Graph, louvain=louvain),
            'cunn': CuNN}


def test_gpu_glue_returns_labels_in_event_order(gpu_claimed, monkeypatch,
                                                caplog):
    monkeypatch.setattr(P, '_GPU_CLUSTER_KIT', _fake_kit())
    df, truth = _blobs()
    with caplog.at_level(logging.INFO, logger='openflo.pipeline'):
        s = _cluster(df, reproducible=False, use_gpu=True)
    assert any(r.getMessage().rstrip().endswith('[GPU]') for r in caplog.records)
    lab = s.data['cluster'].to_numpy()
    assert (lab >= 0).all()
    assert _pure(lab, truth)                    # labels line up with events
