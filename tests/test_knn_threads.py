"""Neighbour searches for clustering run on several threads and must return
exactly the single-thread neighbours, ties included (speed audit item 6).

Flow data has real ties (whole-number channels, saturated events), so the
test data is rounded to make many points equidistant. The negative partners
show that this data does separate a correct search from a wrong one.
"""
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from openflo import pipeline as P
from openflo.parallel import cpu_threads, tree_kneighbors


@pytest.fixture(autouse=True)
def _four_threads(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '4')


def _tied(seed, n, d, decimals=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4, size=(3, d))
    X = np.vstack([rng.normal(loc=c, size=(n // 3, d)) for c in centres])
    return np.round(X, decimals)


@pytest.mark.parametrize('algorithm, d', [
    ('kd_tree', 4), ('ball_tree', 4), ('auto', 6), ('auto', 20)])
def test_threaded_search_equals_single_thread(algorithm, d):
    X = _tied(0, 9000, d)
    Q = _tied(1, 6000, d)
    nn = NearestNeighbors(n_neighbors=10, algorithm=algorithm).fit(X)
    expected = nn.kneighbors(Q, 10, return_distance=False)
    np.testing.assert_array_equal(tree_kneighbors(nn, Q, 10), expected)
    np.testing.assert_array_equal(tree_kneighbors(nn, Q, 1),
                                  nn.kneighbors(Q, 1, return_distance=False))


def test_a_different_tree_breaks_ties_differently():
    """Negative partner: SciPy's cKDTree, the fast search the audit measured,
    returns different neighbours on this tied data. Swapping it in would
    change cluster labels, and the test above would catch any search that
    did."""
    from scipy.spatial import cKDTree
    X = _tied(0, 9000, 4)
    nn = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(X)
    expected = nn.kneighbors(X, 10, return_distance=False)
    other = cKDTree(X).query(X, k=10)[1]
    assert not np.array_equal(other, expected)


def test_thread_count_follows_omp_num_threads(monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '3')
    assert cpu_threads() == 3
    monkeypatch.setenv('OMP_NUM_THREADS', 'junk')
    assert cpu_threads() >= 1
    monkeypatch.delenv('OMP_NUM_THREADS')
    assert cpu_threads() >= 1


def _old_snn_edges(X, k):
    """The graph build as it was before threading (kneighbors_graph)."""
    a = kneighbors_graph(X, k, mode='connectivity', include_self=True)
    inter = (a @ a.T).tocoo()
    deg = np.asarray(a.sum(axis=1)).ravel()
    keep = inter.row < inter.col
    ii, jj, shared = inter.row[keep], inter.col[keep], inter.data[keep]
    w = shared / np.maximum(deg[ii] + deg[jj] - shared, 1e-9)
    good = w > 0
    return (list(zip(ii[good].tolist(), jj[good].tolist(), strict=True)),
            w[good].tolist())


@pytest.mark.parametrize('d', [5, 18])
def test_snn_graph_is_unchanged(d):
    """The Leiden/Louvain graph: same edges, same order, same weights."""
    X = _tied(2, 6000, d, decimals=1)
    edges, weights = _old_snn_edges(X, 15)
    g = P._snn_jaccard_graph(X, 15)
    assert g.get_edgelist() == edges
    assert g.es['weight'] == weights


def test_snn_graph_changes_with_the_data():
    """Negative partner: moving one event changes the graph, so the equality
    above is a real check."""
    X = _tied(2, 6000, 5, decimals=1)
    X2 = X.copy()
    X2[0] += 50.0
    g = P._snn_jaccard_graph(X, 15)
    g2 = P._snn_jaccard_graph(X2, 15)
    assert g.get_edgelist() != g2.get_edgelist()
