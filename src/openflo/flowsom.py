"""FlowSOM: train a self-organizing map over the marker space, assign each
event to its best-matching node, agglomerate the nodes into metaclusters, and
lay the node prototypes out as a minimal spanning tree.

Core module: numpy, with scikit-learn, scipy, igraph and the compiled
``som_kernel`` loop imported where they are used. No Tk, and it does not
import ``openflo.pipeline`` (pipeline imports it).
"""

import contextlib

import numpy as np

# ── FlowSOM (self-organizing map + metaclustering) ────────────────────────────
#
# A compact, dependency-free FlowSOM: train a rectangular SOM over the marker
# space, assign each event to its best-matching unit (node), then agglomerate
# the node prototypes into a handful of metaclusters. Pure numpy + sklearn
# (already a dependency). Not as tuned as the R FlowSOM, but the same shape:
# nodes capture fine structure, metaclusters give interpretable populations.

def som_train(X, grid=(10, 10), iters=10, max_events=50_000, seed=42,
               use_kernel=True):
    """Train a rectangular SOM. Returns (weights[n_nodes, d],
    coords[n_nodes, 2]). Online updates over a (sub-sampled) event stream
    with linearly decaying neighbourhood + learning rate. ``use_kernel=False``
    forces the pure-numpy loop (the reference the compiled one must match)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    if len(X) > max_events:
        X = X[rng.choice(len(X), max_events, replace=False)]
    gx, gy = grid
    n_nodes = gx * gy
    init = rng.choice(len(X), n_nodes, replace=(len(X) < n_nodes))
    W = X[init].astype(float).copy()
    coords = np.array([(i // gy, i % gy) for i in range(n_nodes)], dtype=float)
    sigma0 = max(gx, gy) / 2.0
    lr0 = 0.5
    # The per-event loop runs compiled when numba is there (about 10x faster,
    # identical weights; see openflo.som_kernel). Its neighbourhood table
    # holds the same numbers the loop below computes per event: gd2[b] is
    # ((coords - coords[b]) ** 2).sum(1).
    from .som_kernel import som_epoch_kernel
    epoch = som_epoch_kernel() if use_kernel else None
    if epoch is not None:
        X = np.ascontiguousarray(X)
    gd2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2)
    for t in range(iters):
        frac = t / max(1, iters)
        sigma = sigma0 * (1.0 - frac) + 0.5
        lr = lr0 * (1.0 - frac)
        two_sig2 = 2.0 * sigma * sigma
        if epoch is not None:
            epoch(X, W, rng.permutation(len(X)),
                  lr * np.exp(-gd2 / two_sig2))
            continue
        for i in rng.permutation(len(X)):
            x = X[i]
            bmu = int(np.argmin(((W - x) ** 2).sum(1)))
            gd = ((coords - coords[bmu]) ** 2).sum(1)
            h = np.exp(-gd / two_sig2)
            W += (lr * h)[:, None] * (x - W)
    return W, coords


def som_assign(X, W):
    """Best-matching-unit node index for each row of X (chunked)."""
    X = np.asarray(X, dtype=float)
    out = np.empty(len(X), dtype=int)
    step = 10_000
    for s in range(0, len(X), step):
        chunk = X[s:s + step]
        # squared distances to every node: (chunk·node) expansion
        d = (np.sum(chunk ** 2, 1)[:, None]
             - 2.0 * chunk @ W.T
             + np.sum(W ** 2, 1)[None, :])
        out[s:s + step] = np.argmin(d, axis=1)
    return out


def som_metacluster(W, n_metaclusters):
    """Agglomerate node prototypes into `n_metaclusters` labels (one per
    node). Falls back to one cluster when there are too few nodes."""
    from sklearn.cluster import AgglomerativeClustering
    k = max(1, min(int(n_metaclusters), len(W)))
    if k == 1:
        return np.zeros(len(W), dtype=int)
    return AgglomerativeClustering(n_clusters=k).fit_predict(W)


def flowsom_mst(weights):
    """Minimal-spanning-tree edges over FlowSOM node prototypes — the backbone
    of the classic FlowSOM star-tree plot.

    ``weights`` : ``(n_nodes, n_markers)`` SOM prototypes (``flowsom_result
    ['weights']``). Returns ``(edges, dist)`` where ``edges`` is a list of
    ``(i, j)`` node-index pairs (``n_nodes − 1`` of them, forming a tree) and
    ``dist`` is the full pairwise Euclidean distance matrix (so the caller can
    weight or lay out the tree)."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    W = np.asarray(weights, dtype=float)
    n = len(W)
    if n < 2:
        return [], np.zeros((n, n))
    dist = squareform(pdist(W))
    mst = minimum_spanning_tree(dist).toarray()
    i, j = np.nonzero(mst)
    edges = list(zip(i.tolist(), j.tolist(), strict=True))
    return edges, dist


@contextlib.contextmanager
def igraph_seeded(seed):
    """Run igraph's randomised routines on ``random.Random(seed)``, then hand
    igraph back its default generator, the ``random`` module itself.

    igraph has no integer seed argument. ``layout_fruchterman_reingold(seed=)``
    takes a matrix of STARTING POSITIONS, not a number, and even with one the
    layout still draws random moves from igraph's generator. So the only way
    to make a layout or a Louvain run repeat is to swap the generator in. The
    restore must be the ``random`` module and not a fresh ``random.Random()``:
    a fresh instance cuts igraph off from ``random.seed()`` for the rest of
    the process."""
    import random as _random

    import igraph as ig
    ig.set_random_number_generator(_random.Random(int(seed)))
    try:
        yield
    finally:
        ig.set_random_number_generator(_random)


def flowsom_layout(n_nodes, edges, seed=42):
    """2-D layout for the FlowSOM MST. Uses igraph's Fruchterman-Reingold on
    the tree; falls back to a circle if igraph isn't available. Returns an
    ``(n_nodes, 2)`` float array of positions; the same ``seed`` gives the
    same layout."""
    if n_nodes == 0:
        return np.zeros((0, 2))
    try:
        import igraph as ig
    except ImportError:
        ig = None
    if ig is not None:
        g = ig.Graph(n=int(n_nodes), edges=[(int(a), int(b)) for a, b in edges])
        with igraph_seeded(seed):
            lay = g.layout_fruchterman_reingold(niter=500)
        return np.asarray(lay.coords, dtype=float)
    ang = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])
