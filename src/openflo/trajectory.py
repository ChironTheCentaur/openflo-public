"""Pseudotime / trajectory ordering for developmental flow data.

Orders cells along a differentiation trajectory so maturation can be read as a
continuous axis (e.g. CD34⁺ progenitors → CD11b⁺ myeloids across a
day-series). Light and dependency-conservative — a symmetric k-nearest-neighbour
graph on the marker space, with pseudotime = the geodesic (shortest-path)
distance from a root cell, normalized to ``[0, 1]``. This is the
Wishbone/Palantir-style "geodesic pseudotime"; robust and interpretable without
a full diffusion-map stack.

Pure (numpy / scipy / sklearn). The GUI adds a ``pseudotime`` column and plots
marker trends along it; ``pseudotime_trends`` builds those trend curves.
"""
from __future__ import annotations

import numpy as np


def _standardize(X):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return np.nan_to_num((X - mu) / sd)


def robust_root(X, score, high=True, pct=5.0):
    """Pick a robust root cell for the trajectory.

    Takes the cells in the extreme ``pct``% of ``score`` (top end if ``high``,
    else bottom) and returns the index of their **medoid** — the extreme cell
    closest to the centroid of that extreme set, so a single noisy outlier
    isn't chosen. ``score`` is a per-cell scalar (e.g. a stemness marker such
    as CD34). Returns an int index into ``X``."""
    X = np.asarray(X, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(X)
    if n == 0:
        raise ValueError("empty X")
    finite = np.isfinite(score)
    if not finite.any():
        return 0
    thr = np.percentile(score[finite], 100 - pct if high else pct)
    pool = np.where(finite & ((score >= thr) if high else (score <= thr)))[0]
    if pool.size == 0:
        pool = np.where(finite)[0]
    centroid = X[pool].mean(axis=0)
    d = np.linalg.norm(X[pool] - centroid, axis=1)
    return int(pool[int(np.argmin(d))])


def _geodesic_pseudotime(X, root, n_neighbors=15):
    """Normalized shortest-path distance from ``root`` over a symmetric kNN
    graph on ``X``. Unreachable cells (separate components) take the max
    finite distance. Returns a float array in ``[0, 1]``."""
    from scipy.sparse.csgraph import dijkstra
    from sklearn.neighbors import kneighbors_graph
    n = len(X)
    if n == 1:
        return np.zeros(1)
    k = int(max(1, min(n_neighbors, n - 1)))
    g = kneighbors_graph(X, k, mode='distance')
    g = g.maximum(g.T)                      # symmetrize
    d = dijkstra(g, directed=False, indices=root)
    d = np.asarray(d, dtype=float).ravel()
    finite = d[np.isfinite(d)]
    dmax = float(finite.max()) if finite.size else 1.0
    d = np.where(np.isfinite(d), d, dmax)
    rng = float(d.max() - d.min())
    return (d - d.min()) / rng if rng > 0 else np.zeros(n)


def compute_pseudotime(X, score, high=True, n_neighbors=15,
                       max_cells=20_000, standardize=True, seed=42):
    """Pseudotime for every row of ``X``.

    ``score`` picks the root (see :func:`robust_root`); ``high`` roots at its
    upper end (e.g. CD34-high progenitors as t=0). For tractability the geodesic
    graph is built on a random subsample of up to ``max_cells`` cells (plus the
    root); every other cell inherits the pseudotime of its nearest subsample
    neighbour. Returns ``(pseudotime[n], root_index)`` with pseudotime in
    ``[0, 1]``."""
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n == 0:
        return np.zeros(0), 0
    Xs = _standardize(X) if standardize else X
    score = np.asarray(score, dtype=float)
    root = robust_root(Xs, score, high=high)

    if n <= max_cells:
        return _geodesic_pseudotime(Xs, root, n_neighbors), root

    rng = np.random.default_rng(seed)
    sub = rng.choice(n, max_cells, replace=False)
    if root not in sub:
        sub = np.append(sub, root)
    sub = np.unique(sub)
    root_local = int(np.where(sub == root)[0][0])
    pt_sub = _geodesic_pseudotime(Xs[sub], root_local, n_neighbors)
    # Propagate to all cells by nearest subsample neighbour.
    nn = NearestNeighbors(n_neighbors=1).fit(Xs[sub])
    _, idx = nn.kneighbors(Xs)
    return pt_sub[idx.ravel()], root


def pseudotime_trends(pseudotime, marker_matrix, n_bins=20):
    """Mean of each marker within equal-width pseudotime bins.

    ``marker_matrix`` is ``(n_cells, n_markers)``. Returns
    ``(centers, means)`` where ``centers`` is ``(n_bins,)`` bin midpoints and
    ``means`` is ``(n_bins, n_markers)`` — empty bins are NaN. Feeds the
    "expression vs pseudotime" trend plot (e.g. CD34 falling, CD11b rising)."""
    pt = np.asarray(pseudotime, dtype=float)
    M = np.asarray(marker_matrix, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full((n_bins, M.shape[1]), np.nan)
    idx = np.clip(np.searchsorted(edges, pt, side='right') - 1, 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            means[b] = np.nanmean(M[m], axis=0)
    return centers, means
