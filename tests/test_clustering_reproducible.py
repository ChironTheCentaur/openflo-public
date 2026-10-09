"""Clustering must give the same answer twice — and a different answer on
different data.

PhenoGraph's default community detection is Louvain, run through the Blondel
reference BINARIES shipped inside the package. Those expose no seed
(`community.exe` usage lists only -w -p -q -l -v -h) and are driven in a
restart loop whose stopping rule includes `(time.time() - tic) < time_limit`,
so the result depended on both an unpinnable RNG and the speed of the machine.

The seed is `srand(time(NULL) + getpid())` (`main_community.cpp:121` in the
Blondel v0.2 source), driving a shuffle of the node traversal order in
`one_level()` (`community.cpp:295-302`). Recording that here because the
obvious fix is to inject a fixed or stepped clock, and it cannot work: **the
PID half is assigned by the OS**. Confirmed before the source was read — 20
runs on identical input, all inside one second, produced 20 distinct
partitions.

Measured on 50k events with overlapping populations: two runs of the default
agreed only to **ARI 0.76**, with labels differing outright. An analysis that
changes when you re-run it is hard to put in a methods section.

Since 2.4.9 the default is PhenoGraph's *seeded Leiden* backend, which calls
`leidenalg.find_partition(..., seed=...)` in-process — deterministic by
construction, no subprocess, no timing-dependent loop. It was also ~1.5x
faster and equivalent against planted ground truth (ARI 0.4905 vs 0.4990).

Not free: Leiden and Louvain disagree with each other (ARI 0.74, 7 clusters vs
9 on that fixture), so `reproducible=False` remains available for reproducing
prior Louvain output, matching published PhenoGraph results, or using the GPU
path (cuGraph's Louvain is likewise unseeded).

Imports of openflo modules are deliberately INSIDE the tests. Importing
`openflo.cli` at module scope destabilised pytest-xdist workers badly enough to
crash one outright (an unrelated Tk test in another worker died with an xdist
INTERNALERROR), which is a good reminder that a test module's import side
effects are shared with everything else scheduled on that worker.
"""
import inspect
import io
import os
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest


def test_the_library_default_is_reproducible():
    """Cheap guard on the default itself: flipping it back to Louvain would
    restore run-to-run drift silently, since nothing else in the fast suite
    clusters at all."""
    from openflo.pipeline import FlowSample
    sig = inspect.signature(FlowSample.cluster)
    assert sig.parameters['reproducible'].default is True, (
        'FlowSample.cluster defaults to non-reproducible clustering again — '
        'the same data would give different labels on a re-run')


def test_the_workspace_default_is_reproducible():
    """The GUI and batch runs read this, so it must agree with the library."""
    from openflo.workspace import DEFAULT_RUN_CFG
    assert DEFAULT_RUN_CFG['reproducible'] is True


def test_the_cli_default_is_reproducible():
    """The CLI reaches the same default by inverting --no-reproducible; if that
    wiring breaks, batch runs silently become non-reproducible again."""
    from openflo.cli import run
    assert inspect.signature(run).parameters['reproducible'].default is True


def _fake_sample(data_seed):
    """A clustered fixture. `data_seed` varies the DATA, never the clustering
    seed — which stays at the default so the tests exercise real behaviour."""
    from openflo.pipeline import FlowSample
    rng = np.random.RandomState(data_seed)
    x = rng.normal(0, 1, (3000, 6)) + rng.choice([0, 4, 8], (3000, 1))
    df = pd.DataFrame({f'FL{i}-A': x[:, i].astype(np.float32) for i in range(6)})
    s = FlowSample.__new__(FlowSample)
    s.data = df.copy()
    s.raw = df
    s.fluor_channels = list(df.columns)
    s.channel_names = list(df.columns)
    s.channel_labels = {c: c for c in df.columns}
    s.clusters = None
    return s


def _labels(data_seed):
    s = _fake_sample(data_seed)
    with redirect_stdout(io.StringIO()):            # phenograph is chatty
        s.cluster(channels=s.fluor_channels, k=15, random_state=42)
    return s.data['cluster'].to_numpy()


slow = pytest.mark.skipif(
    os.environ.get('OPENFLO_RUN_SLOW_TESTS') != '1',
    reason='clusters several times (~35s each) — set OPENFLO_RUN_SLOW_TESTS=1')


@slow
def test_clustering_the_same_data_twice_gives_the_same_labels():
    """The property itself, end to end through the public API with DEFAULTS.

    The only check that would catch the default reverting to the unseeded
    binaries, so CI runs it on one leg with the other slow integration tests.
    """
    first, second = _labels(0), _labels(0)
    assert np.array_equal(first, second), (
        'clustering the same data twice gave different labels — the default '
        'is no longer seeded')
    assert len(set(first)) > 1, 'fixture collapsed to a single cluster'


@slow
def test_the_same_seed_on_DIFFERENT_data_still_diverges():
    """The guard that stops the test above passing vacuously.

    Fixing the seed must remove the algorithm's randomness, NOT its response to
    the data. If a bug ever made clustering ignore its input — returning a
    canned partition — the determinism test would pass perfectly while the
    tool became useless. Measured: same seed on three different datasets gave
    14, 11 and 12 clusters with different assignments.
    """
    base = _labels(0)
    for other_seed in (1, 2):
        other = _labels(other_seed)
        assert not np.array_equal(base, other), (
            f'different data (seed {other_seed}) produced IDENTICAL labels to '
            f'the baseline — clustering is not responding to its input, and '
            f'the determinism test above is therefore meaningless')
