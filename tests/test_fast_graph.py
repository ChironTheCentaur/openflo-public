"""The pruned ("fast") neighbour-graph option.

`_snn_jaccard_graph` links every pair of points sharing at least one
neighbour. PhenoGraph and Seurat instead link only pairs where one point is in
the other's kNN list (phenograph.core.calc_jaccard: "between i and i's direct
neighbors"). The dense form is ~14x larger at 20k events, k=30, and 6-9x
slower to partition.

Measured on 20,000 events x 10 channels, 8 planted populations:

                    well separated          heavily overlapping
                 clusters  ARI    homog   clusters  ARI    homog
    prune=False     6     0.9487  0.8964     4     0.4933  0.3891
    prune=True      8     0.9545  0.9220     8     0.3405  0.3973

Pruned recovers the planted cluster COUNT in both cases and has higher
homogeneity in both — the DENSE graph is the one that merges populations. But
on overlapping data the two disagree substantially (ARI 0.53 between them) and
pruned scores worse on ARI and completeness, so it is an option and not the
default. These tests pin that it stays optional and off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample, _snn_jaccard_graph

pytest.importorskip('igraph')

CH = [f'M{i}' for i in range(6)]
N, K = 2500, 15


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    centres = rng.uniform(3.5, 6.5, size=(4, len(CH)))
    truth = rng.choice(4, N)
    X = centres[truth] + rng.normal(0, 0.6, (N, len(CH)))
    return X, truth


def test_pruning_removes_edges_and_keeps_only_knn_pairs(data):
    X, _ = data
    dense = _snn_jaccard_graph(X, K)
    pruned = _snn_jaccard_graph(X, K, prune=True)

    assert pruned.ecount() < dense.ecount(), 'pruning removed nothing'

    # Every surviving edge must be a kNN pair in one direction or the other.
    from sklearn.neighbors import kneighbors_graph
    a = kneighbors_graph(X, K, mode='connectivity', include_self=True)
    sym = ((a + a.T) > 0).tocsr()
    for i, j in pruned.get_edgelist():
        assert sym[i, j] != 0, (
            f'edge ({i},{j}) survived pruning but neither point is in the '
            f"other's {K}-nearest-neighbour list")


def test_pruning_is_off_by_default(data):
    """Turning this on silently would change every clustering result anyone
    has already produced."""
    X, _ = data
    assert (_snn_jaccard_graph(X, K).ecount()
            == _snn_jaccard_graph(X, K, prune=False).ecount())
    assert (_snn_jaccard_graph(X, K).ecount()
            != _snn_jaccard_graph(X, K, prune=True).ecount())


def test_run_leiden_default_is_unchanged_by_the_new_parameter(data):
    """The option exists; the default path must be byte-identical to before."""
    pytest.importorskip('leidenalg')
    X, _ = data
    df = pd.DataFrame(X, columns=CH)

    def labels(**kw):
        s = FlowSample.from_dataframe(df.copy(), name='t')
        s.fluor_channels = list(CH)
        s.run_leiden(channels=CH, n_neighbors=K, max_events=N,
                     random_state=42, **kw)
        return s.data['leiden'].to_numpy()

    assert np.array_equal(labels(), labels(fast_graph=False))


def test_fast_graph_actually_changes_the_result(data):
    """Guards the opposite failure: an option that is accepted and ignored
    would make the test above pass for the wrong reason."""
    pytest.importorskip('leidenalg')
    X, _ = data
    df = pd.DataFrame(X, columns=CH)

    def labels(fast):
        s = FlowSample.from_dataframe(df.copy(), name='t')
        s.fluor_channels = list(CH)
        s.run_leiden(channels=CH, n_neighbors=K, max_events=N,
                     random_state=42, fast_graph=fast)
        return s.data['leiden'].to_numpy()

    assert not np.array_equal(labels(False), labels(True)), (
        'fast_graph=True produced the identical partition — the flag is not '
        'reaching the graph builder')


def test_fast_graph_is_still_reproducible(data):
    pytest.importorskip('leidenalg')
    X, _ = data
    df = pd.DataFrame(X, columns=CH)

    def labels():
        s = FlowSample.from_dataframe(df.copy(), name='t')
        s.fluor_channels = list(CH)
        s.run_leiden(channels=CH, n_neighbors=K, max_events=N,
                     random_state=42, fast_graph=True)
        return s.data['leiden'].to_numpy()

    assert np.array_equal(labels(), labels())
