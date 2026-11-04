"""How many threads OpenFlo's own parallel steps may use.

The CLI and the GUI set ``OMP_NUM_THREADS`` before numpy loads (``cli.py``
gives each sample worker its share of the cores; ``gui.py`` caps it), so the
compiled libraries already obey it. Neighbour searches and the Statistics
table follow the same number, so a CLI run with ``--workers 4`` does not
start four times as many threads as there are cores.
"""
from __future__ import annotations

import os

import numpy as np


def cpu_threads() -> int:
    """``OMP_NUM_THREADS`` when it is a positive integer, else the core
    count; at least 1."""
    raw = os.environ.get('OMP_NUM_THREADS', '').strip()
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        n = os.cpu_count() or 1
    return max(1, n)


def tree_kneighbors(nn, X, k) -> np.ndarray:
    """``nn.kneighbors(X, k, return_distance=False)`` for a fitted
    scikit-learn ``NearestNeighbors``, with the queries split across threads.

    The indices are exactly the single-thread ones, ties included: every
    thread queries an identical copy of the same tree (same data, leaf size
    and metric, so the same build), and each query is answered on its own.

    Why copies: scikit-learn's own ``n_jobs`` shares one tree, and the tree
    updates its counters on every query, so threads fight over that memory and
    four of them ran no faster than one (measured: 3.0 s for 1 thread, 2.5 s
    for 4). With a tree each, 4 threads took 1.06 s. A copy costs a rebuild,
    about 50 ms for 50k events.

    Brute-force fits (scikit-learn picks them for more than 15 dimensions)
    already run on every core, so they go straight to ``nn.kneighbors``, as
    does anything too small to be worth splitting.
    """
    tree = getattr(nn, '_tree', None)
    n_threads = min(cpu_threads(), max(1, len(X) // 2000))
    fit_X = getattr(nn, '_fit_X', None)
    if tree is None or fit_X is None or n_threads < 2:
        return np.asarray(nn.kneighbors(X, k, return_distance=False))
    X = np.ascontiguousarray(X, dtype=np.float64)
    params = dict(getattr(nn, 'effective_metric_params_', None) or {})
    tree_cls = type(tree)
    chunks = np.array_split(np.arange(len(X)), n_threads)

    def run(idx):
        own = tree_cls(fit_X, nn.leaf_size, metric=nn.effective_metric_,
                       **params)
        return own.query(X[idx], k, return_distance=False)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        parts = list(ex.map(run, chunks))
    return np.vstack(parts)
