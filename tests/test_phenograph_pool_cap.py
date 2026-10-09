"""FlowSample.cluster(n_jobs=1) opens no process Pool.

phenograph 1.5.7's cluster() calls sort_by_size(communities, min_cluster_size)
without its n_jobs, so the size sort fell back to sort_by_size's n_jobs=-1 and
opened mp.Pool(cpu_count()) on every call: Pool(24) on a 24-core host even at
n_jobs=1. The CLI relies on n_jobs=1 meaning one process per outer worker.
pipeline._phenograph_sort_jobs caps the sort to the caller's n_jobs; these
tests spy on multiprocessing.Pool and check the labels did not move.
"""
import contextlib
import io
import multiprocessing

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('phenograph')


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        np.vstack([rng.normal(m, 0.3, (150, 3)) for m in (0, 6)]
                  + [rng.normal(3, 2.0, (200, 3))]),
        columns=['A', 'B', 'C'])


@pytest.fixture
def pool_sizes(monkeypatch):
    """Every multiprocessing.Pool size requested, while still giving a real
    (small) Pool so callers that do open one keep working."""
    real = multiprocessing.Pool
    seen = []

    def spy(processes=None, *a, **k):
        seen.append(processes)
        return real(min(processes or 2, 2), *a, **k)
    monkeypatch.setattr(multiprocessing, 'Pool', spy)
    return seen


def test_cluster_at_n_jobs_1_opens_no_pool(pool_sizes):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(_df())
    with contextlib.redirect_stdout(io.StringIO()):      # phenograph is chatty
        s.cluster(channels=['A', 'B', 'C'], k=15, n_jobs=1)
    assert pool_sizes == []
    assert (s.data['cluster'] >= 0).sum() > 0


def test_serial_size_sort_matches_phenograph_exactly():
    """The in-process sort must relabel exactly as phenograph's Pool version,
    ties and below-min_size clusters included, so no clustering moves."""
    import importlib

    from openflo.pipeline import _install_phenograph_sort_cap, _sort_by_size_serial
    _install_phenograph_sort_cap()
    pgc = importlib.import_module('phenograph.cluster')
    orig = getattr(pgc.sort_by_size, '__wrapped__', pgc.sort_by_size)
    rng = np.random.default_rng(3)
    # Consecutive ids, as Louvain/Leiden return them, with equal-size ties
    # and clusters on both sides of min_size.
    sizes = [40, 40, 25, 11, 10, 10, 3, 40, 1, 25]
    cl = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    want = orig(cl, 10, 2)
    got = _sort_by_size_serial(cl, 10)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)


def test_other_callers_keep_phenographs_own_pool(pool_sizes):
    """Outside an OpenFlo cluster() call the wrapper passes n_jobs straight
    through, so code that calls phenograph itself sees no change."""
    import importlib

    from openflo.pipeline import _install_phenograph_sort_cap
    _install_phenograph_sort_cap()
    pgc = importlib.import_module('phenograph.cluster')
    cl = np.repeat(np.arange(3), [30, 20, 15])
    pgc.sort_by_size(cl, 10, 3)
    assert pool_sizes == [3]
