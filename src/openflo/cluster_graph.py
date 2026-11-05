"""The graph under Leiden and Louvain clustering (``snn_jaccard_graph``), the
cap that keeps phenograph's process pools to the caller's ``n_jobs`` and off
fork (``phenograph_sort_jobs``), and the switch that sends phenograph's progress
prints to the log (``phenograph_prints_to_log``).

Core module: numpy, the standard library and OpenFlo's ``parallel``; igraph,
scipy, scikit-learn and phenograph are imported only when called. No Tk, and
it does not import ``openflo.pipeline``.
"""

import builtins
import contextlib
import functools
import logging
import os
import threading

import numpy as np

from .parallel import tree_kneighbors

log = logging.getLogger(__name__)

# phenograph 1.5.7's cluster() ends with ``sort_by_size(communities,
# min_cluster_size)`` and does not pass its n_jobs on. sort_by_size then uses
# its own default of -1 and opens ``mp.Pool(cpu_count())``, so every call
# spawned a full-width Pool even at n_jobs=1. A spy on multiprocessing.Pool saw
# Pool(24) per cluster(n_jobs=1) call on a 24-core host. Under the CLI's
# parallel workers that is workers x cpu_count extra interpreters, each
# re-importing the numeric stack.
#
# Its Jaccard kernel (``parallel_jaccard_kernel``, used whenever n_jobs != 1)
# likewise opens ``Pool()`` on every core whatever n_jobs says. And both pools
# use the default start method, which is fork on Linux before Python 3.14. A
# fork copies the process with only the calling thread, so a lock any other
# thread held at that instant (logging, numba, BLAS, Tk in the GUI) stays held
# in the child for good and the clustering hangs (Python 3.12 warns).
#
# To fix this without forking phenograph's code, its module-level
# sort_by_size and parallel_jaccard_kernel are swapped (once) for wrappers.
# They apply to the OpenFlo cluster() call running on the same thread: the
# size sort runs in-process (counting labels needs no pool), and the Jaccard
# kernel runs phenograph's own calc_jaccard in a spawned Pool of exactly
# n_jobs processes. Any other caller gets phenograph's own behaviour.
_PG_SORT_JOBS = threading.local()
_PG_SORT_INSTALL_LOCK = threading.Lock()


def sort_by_size_serial(clusters, min_size=10):
    """phenograph.cluster.sort_by_size, without its process Pool.

    The Pool only counts each cluster's events, in np.unique order. Counting
    in-process gives the same sizes in the same order, and the rest is
    phenograph's own code verbatim (its argsort, its position-keyed mapping,
    its np.vectorize), so the labels come out identical."""
    _, counts = np.unique(clusters, return_counts=True)
    sizes = [int(c) for c in counts]
    o = np.argsort(sizes)[::-1]
    my_dict = {c: i for i, c in enumerate(o) if sizes[c] > min_size}
    my_dict.update({c: -1 for _i, c in enumerate(o) if sizes[c] <= min_size})
    return np.vectorize(my_dict.get)(clusters)


def pool_processes(n_jobs):
    """Processes for ``n_jobs`` by phenograph's (and joblib's) rule: -1 is
    every core, below -1 is ``cores + 1 + n_jobs``; never fewer than one."""
    cpus = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = cpus + 1 + n_jobs
    return max(1, int(n_jobs))


def jaccard_kernel_spawn(idx, n_jobs):
    """phenograph.core.parallel_jaccard_kernel in ``pool_processes(n_jobs)``
    spawned processes instead of a forked Pool on every core.

    Each row is phenograph's own calc_jaccard and the assembly is its own code,
    so the graph is identical to phenograph's (serial ``jaccard_kernel``
    included).

    Spawned workers re-import the main script, so a script that clusters at
    module level with no ``if __name__ == '__main__':`` guard kills each one
    as it starts. A multiprocessing.Pool would replace them for ever and never
    return; ProcessPoolExecutor reports the broken pool, which is raised here
    as an error that names the guard. Raises RuntimeError."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    from itertools import repeat

    import scipy.sparse as sp
    from phenograph.core import calc_jaccard

    from ._console import start_worker, worker_settings

    n = len(idx)
    procs = pool_processes(n_jobs)
    # Pool.starmap's own chunk size: executor.map's default of 1 would
    # pickle idx once per row.
    chunk = max(1, -(-n // (4 * procs)))
    try:
        # Spawned workers start with none of this process's console
        # settings (encoding, warning filters, logging): start_worker sets them.
        with ProcessPoolExecutor(
                procs, mp_context=multiprocessing.get_context('spawn'),
                initializer=start_worker,
                initargs=(worker_settings(),)) as ex:
            jaccard_values = list(ex.map(calc_jaccard, range(n), repeat(idx),
                                         chunksize=chunk))
    except BrokenProcessPool as e:
        raise RuntimeError(
            "PhenoGraph's worker processes could not start. A script that "
            "clusters with n_jobs other than 1 must run its code under "
            "if __name__ == '__main__':") from e
    graph = sp.lil_matrix((n, n), dtype=float)
    for i, tup in enumerate(jaccard_values):
        graph.rows[i] = tup[0].tolist()
        graph.data[i] = tup[1].tolist()
    i, j = graph.nonzero()
    s = graph.tocoo().data
    return i, j, s[s > 0]


def install_phenograph_sort_cap():
    """Route phenograph.cluster's sort_by_size and parallel_jaccard_kernel
    through the n_jobs cap. Runs once; a phenograph without one of those
    functions has that one left untouched."""
    import importlib
    try:
        pgc = importlib.import_module('phenograph.cluster')
    except ImportError:
        return
    with _PG_SORT_INSTALL_LOCK:
        orig = getattr(pgc, 'sort_by_size', None)
        if orig is not None and not getattr(orig, '_openflo_n_jobs_cap', False):
            pgc.sort_by_size = _capped_sort(orig)  # pyright: ignore[reportAttributeAccessIssue]
        orig = getattr(pgc, 'parallel_jaccard_kernel', None)
        if orig is not None and not getattr(orig, '_openflo_n_jobs_cap', False):
            pgc.parallel_jaccard_kernel = _capped_kernel(orig)  # pyright: ignore[reportAttributeAccessIssue]


def _capped_sort(orig):
    @functools.wraps(orig)
    def sort_by_size(clusters, min_size=10, n_jobs=-1):
        if getattr(_PG_SORT_JOBS, 'n_jobs', None) is None:  # not OpenFlo's call
            return orig(clusters, min_size, n_jobs)
        return sort_by_size_serial(clusters, min_size)

    sort_by_size._openflo_n_jobs_cap = True  # type: ignore[attr-defined]
    return sort_by_size


def _capped_kernel(orig):
    @functools.wraps(orig)
    def parallel_jaccard_kernel(idx):
        want = getattr(_PG_SORT_JOBS, 'n_jobs', None)
        if want is None:                      # not an OpenFlo cluster() call
            return orig(idx)
        return jaccard_kernel_spawn(idx, want)

    parallel_jaccard_kernel._openflo_n_jobs_cap = True  # type: ignore[attr-defined]
    return parallel_jaccard_kernel


@contextlib.contextmanager
def phenograph_sort_jobs(n_jobs):
    """While active on this thread, phenograph's size sort and Jaccard kernel
    honour ``n_jobs`` and never fork."""
    install_phenograph_sort_cap()
    prev = getattr(_PG_SORT_JOBS, 'n_jobs', None)
    _PG_SORT_JOBS.n_jobs = n_jobs
    try:
        yield
    finally:
        _PG_SORT_JOBS.n_jobs = prev


# phenograph prints its progress ("Finding 30 nearest neighbors ...",
# "PhenoGraph completed in 0.47 seconds") straight to stdout, so every
# clustering, joint clustering included, wrote seven unlabelled lines that
# --quiet could not silence and that bypassed the app's logging. Its modules
# look `print` up in their own globals before builtins, so a module-level
# `print` is put there (once). It sends the line to this module's logger
# while an OpenFlo call on the same thread has the switch on; any other
# caller, or another thread, gets the real print. sys.stdout is never
# swapped, so prints from other threads (the app's own) are untouched.
# Only phenograph's known progress lines go to DEBUG. Anything else is a
# diagnostic ("Could not find Louvain code", the Louvain convert binary's
# "stderr from convert: ...", the only clue when it fails) and goes to
# WARNING, so it is still seen by default.
_PG_PRINTS = threading.local()
_PG_PRINT_MODULES = ('phenograph.cluster', 'phenograph.core')
_PG_PROGRESS = (
    'Finding ', 'Neighbors computed in ', 'Using neighbor information from ',
    'Setting directed=False because prune=True', 'Gaussian kernel graph ',
    'Jaccard graph constructed in ', 'Running Leiden optimization',
    'Leiden completed in ', 'Sorting communities by size',
    'PhenoGraph completed in ', 'Wrote graph to binary file in ',
    'Running Louvain modularity optimization', 'After ',
    'Louvain completed ')


def _phenograph_print(*args, sep=' ', end='\n', file=None, flush=False):
    if getattr(_PG_PRINTS, 'on', False) and file is None:
        text = sep.join(map(str, args)).strip()
        if text:
            level = (logging.DEBUG if text.startswith(_PG_PROGRESS)
                     else logging.WARNING)
            log.log(level, 'phenograph: %s', text)
        return
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


# setattr: pyright allows no new attribute on a function object.
setattr(_phenograph_print, '_openflo_to_log', True)  # noqa: B010


def install_phenograph_print_to_log():
    """Give phenograph's modules the logging ``print``. Runs once per module;
    a phenograph that is not installed is left alone."""
    import importlib
    for name in _PG_PRINT_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        with _PG_SORT_INSTALL_LOCK:
            if not getattr(getattr(mod, 'print', None), '_openflo_to_log',
                           False):
                mod.print = _phenograph_print  # pyright: ignore[reportAttributeAccessIssue]


@contextlib.contextmanager
def phenograph_prints_to_log():
    """While active on this thread, phenograph's prints go to the
    ``openflo.cluster_graph`` logger instead of stdout: its progress lines at
    DEBUG, anything else (a failure diagnostic) at WARNING."""
    install_phenograph_print_to_log()
    prev = getattr(_PG_PRINTS, 'on', False)
    _PG_PRINTS.on = True
    try:
        yield
    finally:
        _PG_PRINTS.on = prev


def snn_jaccard_graph(X, k, prune=False):
    """Shared-nearest-neighbour graph with Jaccard edge weights, as igraph.

    A plain binary kNN graph makes modularity-style objectives over-split
    uniform blobs; Jaccard weighting sharpens real communities so the
    partition tracks the populations.

    NOTE this is NOT PhenoGraph's or Seurat's construction, though it is
    closely related and was described as theirs here until it was measured.
    They compute Jaccard only between i and the members of i's own kNN list
    (phenograph.core.calc_jaccard: "between i and i's direct neighbors"), so
    the edge set is bounded by n*k. ``a @ a.T`` below instead gives an edge to
    EVERY pair sharing at least one neighbour, which is far denser: on 20k
    events at k=30, 6,106,544 edges against PhenoGraph's 297,138 — 20.6x — and
    16x slower to partition (14.9s vs 0.9s per Louvain run).

    The denser graph is not obviously worse: it scored better against planted
    truth (ARI 0.488 vs 0.378). But it under-splits, finding 4 communities
    where 8 were planted and PhenoGraph found 11, which for cytometry means
    distinct populations merged. Pruning to kNN pairs would change every
    existing Leiden and Louvain result, so it has not been changed here.

    Extracted so ``run_leiden`` and ``run_louvain`` share ONE graph: they
    differ only in how the graph is partitioned, and a difference in how it is
    BUILT would silently make their results incomparable.

    ``prune`` selects between two edge sets, measured on 20,000 events x 10
    channels, 8 planted populations, k=30 (regenerate with
    ``scripts/bench_snn_prune.py``):

                        well separated            heavily overlapping
                     clusters  ARI    homog     clusters  ARI    homog
        prune=False     6     0.9487  0.8964       4     0.4933  0.3891
        prune=True      8     0.9545  0.9220       8     0.3405  0.3973

    with Leiden taking 5.0s vs 0.8s and 9.8s vs 1.1s respectively.

    Pruned is faster, recovers the planted cluster COUNT in both cases, and has
    higher homogeneity in both — the dense graph is the one that merges
    populations, finding 4 where 8 were planted. On separable data pruned is
    better on every measure and gives nearly the same partition
    (ARI 0.97 between them).

    It is nonetheless NOT the default, for two reasons. On heavily overlapping
    data the two disagree substantially (ARI 0.53) and pruned scores worse on
    ARI, AMI and completeness — it splits populations the dense graph merges,
    and when populations genuinely overlap it is not clear that is wrong so
    much as different. And switching would silently change every Leiden and
    Louvain result anyone has already produced.
    """
    import igraph as ig
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # kneighbors_graph(X, k, mode='connectivity', include_self=True), built
    # the way scikit-learn builds it but with the queries spread over threads
    # (tree_kneighbors returns the same neighbours, in the same order).
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    ind = tree_kneighbors(nn, nn._fit_X, k)
    n_q = ind.shape[0]
    a = csr_matrix((np.ones(n_q * k), ind.ravel(), np.arange(0, n_q * k + 1, k)),
                   shape=(n_q, nn.n_samples_fit_))
    inter = (a @ a.T)                        # |N(i) ∩ N(j)|
    if prune:
        # "Fast mode": keep an edge only where one point is in the other's kNN
        # list, which is what PhenoGraph and Seurat do. Without this every pair
        # sharing a single neighbour gets an edge, and the graph is ~14x denser
        # (20k events, k=30: 6,106,544 edges against 424,349).
        inter = inter.multiply((a + a.T) > 0)
    inter = inter.tocoo()
    deg = np.asarray(a.sum(axis=1)).ravel()  # |N(i)| = k+1
    keep = inter.row < inter.col             # upper triangle, no self loops
    ii = inter.row[keep]
    jj = inter.col[keep]
    shared = inter.data[keep]
    union = deg[ii] + deg[jj] - shared
    w = shared / np.maximum(union, 1e-9)
    good = w > 0
    edges = list(zip(ii[good].tolist(), jj[good].tolist(), strict=True))
    g = ig.Graph(n=int(X.shape[0]), edges=edges, directed=False)
    g.es['weight'] = w[good].tolist()
    return g
