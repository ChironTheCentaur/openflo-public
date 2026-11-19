"""An OpenFlo PhenoGraph run never forks.

phenograph 1.5.7 opens its process pools with the start method's default,
which is fork on Linux before Python 3.14: ``parallel_jaccard_kernel`` (every
cluster() call with n_jobs != 1) and ``sort_by_size``. A fork copies the
calling process with only the calling thread, so in the threaded GUI any lock
another thread held at that moment (logging, numba, BLAS, Tk) stays held in
the child forever and the clustering hangs. Python 3.12 warns about exactly
this. cluster_graph routes both through fresh (spawned) processes or no
process at all, for the calls OpenFlo makes; anyone else calling phenograph
keeps phenograph's own behaviour.

The guard below makes ``os.fork`` raise, so a fork fails the test on every
Python version (spawn starts its children without ``os.fork``). Data are
generated blobs.
"""
import contextlib
import importlib
import io
import os
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('phenograph')


class Forked(AssertionError):
    pass


def _refuse_fork(*_a, **_k):
    raise Forked('os.fork() called')


@pytest.fixture
def busy_thread():
    """A second live thread, as in the GUI, so the process is multi-threaded
    when phenograph runs."""
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    yield
    stop.set()
    t.join()


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(m, 0.3, (120, 3)) for m in (0, 5, 10)])


def _df(seed=0):
    return pd.DataFrame(_blobs(seed), columns=['A', 'B', 'C'])


def _labels(n_jobs):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(_df())
    with contextlib.redirect_stdout(io.StringIO()):
        s.cluster(channels=['A', 'B', 'C'], k=15, n_jobs=n_jobs)
    return s.data['cluster'].to_numpy()


@pytest.mark.parametrize('n_jobs', [2, -1])
def test_cluster_with_several_jobs_does_not_fork(n_jobs, busy_thread):
    serial = _labels(1)               # n_jobs=1 never opened a pool
    assert (serial >= 0).sum() > 0
    with pytest.MonkeyPatch.context() as mp:
        if hasattr(os, 'fork'):           # Windows has no fork to refuse
            mp.setattr(os, 'fork', _refuse_fork)
        got = _labels(n_jobs)
    assert np.array_equal(got, serial)


def test_no_fork_warning_from_a_threaded_process(busy_thread):
    """What Python 3.12+ says when a threaded process forks. CPython reports
    that warning as unraisable when a filter turns it into an error, so it
    is recorded here instead, regardless of pyproject's filters."""
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(_df())
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        with contextlib.redirect_stdout(io.StringIO()):
            s.cluster(channels=['A', 'B', 'C'], k=15, n_jobs=2)
    assert [str(w.message) for w in seen if 'use of fork' in str(w.message)] == []


@pytest.mark.parametrize('n_jobs', [1, 2, -1])
def test_labels_match_phenographs_serial_run(n_jobs):
    """Same labels as phenograph's own in-process path (n_jobs=1, which
    never opens a pool), so moving off fork moved no clustering."""
    import phenograph

    from openflo.cluster_graph import phenograph_sort_jobs
    X = _blobs(seed=4)
    kw = {'k': 15, 'clustering_algo': 'leiden', 'seed': 7}
    with contextlib.redirect_stdout(io.StringIO()):
        want, _, q_want = phenograph.cluster(X, n_jobs=1, **kw)
        with phenograph_sort_jobs(n_jobs):
            got, _, q_got = phenograph.cluster(X, n_jobs=n_jobs, **kw)
    assert np.array_equal(got, want)
    assert q_got == q_want


class _InProcessPool:
    """Stands in for multiprocessing.Pool: same calls, run in this process."""
    def __init__(self, *_a, **_k):
        pass

    def starmap(self, fn, args):
        return [fn(*a) for a in args]

    def close(self):
        pass


def test_spawned_jaccard_kernel_matches_phenographs_exactly(monkeypatch):
    """Same (i, j, s) as phenograph's parallel kernel, and the same graph as
    its serial one, which keeps the zero entries the parallel one drops."""
    import phenograph.core as pgcore
    import scipy.sparse as sp

    from openflo.cluster_graph import jaccard_kernel_spawn
    rng = np.random.default_rng(11)
    n, k = 200, 12
    idx = np.array([rng.choice(np.delete(np.arange(n), i), k, replace=False)
                    for i in range(n)])
    got = jaccard_kernel_spawn(idx, 2)
    monkeypatch.setattr(pgcore, 'Pool', _InProcessPool)
    want = pgcore.parallel_jaccard_kernel(idx)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.dtype
        assert np.array_equal(g, w)

    def graph(i, j, s):
        return sp.coo_matrix((s, (i, j)), shape=(n, n)).tocsr()
    diff = graph(*got) - graph(*pgcore.jaccard_kernel(idx))
    assert diff.count_nonzero() == 0


@pytest.fixture
def spawn_pool_sizes(monkeypatch):
    """Every worker count asked of a ProcessPoolExecutor, with its start
    method."""
    import concurrent.futures as cf
    seen = []
    real = cf.ProcessPoolExecutor

    def spy(max_workers=None, mp_context=None, *a, **kw):
        seen.append((max_workers, mp_context.get_start_method()))
        return real(min(max_workers or 2, 2), mp_context, *a, **kw)
    monkeypatch.setattr(cf, 'ProcessPoolExecutor', spy)
    return seen


def test_jaccard_pool_honours_n_jobs(spawn_pool_sizes):
    """phenograph's own kernel opens Pool() on every core whatever n_jobs
    says; OpenFlo's call asks for exactly n_jobs processes."""
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(_df())
    with contextlib.redirect_stdout(io.StringIO()):
        s.cluster(channels=['A', 'B', 'C'], k=15, n_jobs=2)
    assert spawn_pool_sizes == [(2, 'spawn')]


def test_pool_size_follows_phenographs_n_jobs_rules():
    from openflo.cluster_graph import pool_processes
    cpus = os.cpu_count() or 1
    assert pool_processes(2) == 2
    assert pool_processes(-1) == cpus
    assert pool_processes(-2) == max(1, cpus - 1)
    assert pool_processes(-10_000) == 1


def test_other_callers_keep_phenographs_own_kernel(monkeypatch):
    """Outside an OpenFlo cluster() call the wrapper hands straight to
    phenograph's kernel and its own Pool."""
    import phenograph.core as pgcore

    from openflo.cluster_graph import install_phenograph_sort_cap
    install_phenograph_sort_cap()
    pgc = importlib.import_module('phenograph.cluster')
    opened = []

    class Spy(_InProcessPool):
        def __init__(self, *a, **k):
            opened.append(a)
    monkeypatch.setattr(pgcore, 'Pool', Spy)
    idx = np.array([[1, 2], [0, 2], [0, 1]])
    i, j, s = pgc.parallel_jaccard_kernel(idx)
    assert opened == [()]
    assert len(i) == len(j) == len(s) == 6


UNGUARDED = """
import numpy as np, pandas as pd
from openflo.pipeline import FlowSample
rng = np.random.default_rng(0)
df = pd.DataFrame(np.vstack([rng.normal(m, .5, (150, 3)) for m in (0, 4, 8)]),
                  columns=['FL1-A', 'FL2-A', 'FL3-A'])
s = FlowSample.from_dataframe(df)
s.cluster(channels=list(df.columns), k=15, n_jobs=2)
print('DONE')
"""


def test_a_script_without_a_main_guard_fails_with_a_clear_error(tmp_path):
    """Spawned workers re-run an unguarded script and die at start. That has
    to end in an error naming the guard, not a pool that replaces its dead
    workers for ever."""
    import subprocess
    import sys
    script = tmp_path / 'unguarded.py'
    script.write_text(UNGUARDED)
    try:
        r = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.fail('an unguarded script hung instead of failing')
    assert r.returncode != 0
    assert 'DONE' not in r.stdout
    # Our message, not multiprocessing's own bootstrapping traceback (which
    # also quotes the idiom) from the dying children.
    assert ("PhenoGraph's worker processes could not start. A script that "
            "clusters with n_jobs other than 1 must run its code under "
            "if __name__ == '__main__':") in r.stderr
