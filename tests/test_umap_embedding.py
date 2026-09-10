"""Response controls for the UMAP embedding.

The golden baseline pins 33 numbers and none of them are UMAP's, so a change
to how the embedding is produced passes the whole suite untouched. That is not
a validation, it is a blind spot — and the change that prompted this file
(routing the neighbour search through a parallel pynndescent index) is exactly
the kind that lands wrong silently: hand UMAP a malformed ``precomputed_knn``
and it does not raise, it just embeds noise.

Coordinates cannot be pinned. UMAP layouts vary with numba, BLAS and platform,
and they are not unique even on one machine — two faithful embeddings of the
same data can look nothing alike. So these are RELATIONAL checks: things that
must hold of any correct embedding, whatever the coordinates come out as.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import FlowSample

CHANNELS = [f'M{i}' for i in range(8)]
# 4,096 is where UMAP itself switches from an exact neighbour search to
# approximate NN-descent; both sides are exercised so neither can rot.
BIG, SMALL = 4200, 600


def _sample(n, seed=0):
    """Three well-separated populations — separable enough that a correct
    embedding must keep them apart, so a broken one cannot pass."""
    rng = np.random.RandomState(seed)
    per = n // 3
    X = np.vstack([rng.normal(c, 1.0, (per, len(CHANNELS)))
                   for c in (0.0, 8.0, 16.0)])
    truth = np.repeat([0, 1, 2], per)
    df = pd.DataFrame(X, columns=CHANNELS)
    s = FlowSample.from_dataframe(df, name='t')
    s.fluor_channels = list(CHANNELS)
    return s, truth


def _embed(n, seed=42, data_seed=0):
    s, truth = _sample(n, data_seed)
    s.run_umap(channels=CHANNELS, n_neighbors=15, random_state=seed)
    return np.asarray(s.umap_coords), truth


def _label_accuracy(emb, truth, k=15):
    """Can a kNN classifier recover the populations from the 2-D coordinates
    alone? Near 1.0 for a faithful embedding of separated blobs; near chance
    (0.33) if the neighbour graph handed to UMAP was garbage."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    return float(cross_val_score(KNeighborsClassifier(k), emb, truth,
                                 cv=3).mean())


umap = pytest.importorskip('umap', reason='umap-learn not installed')


@pytest.mark.parametrize('n', [BIG, SMALL],
                         ids=['parallel-knn', 'umap-own-knn'])
def test_same_seed_gives_the_identical_embedding(n):
    a, _ = _embed(n)
    b, _ = _embed(n)
    assert np.array_equal(a, b), (
        'UMAP is meant to be reproducible at a fixed seed; identical input '
        'produced different coordinates')


@pytest.mark.parametrize('n', [BIG, SMALL],
                         ids=['parallel-knn', 'umap-own-knn'])
def test_the_seed_actually_reaches_the_embedding(n):
    """Guards the opposite failure: a seed that is accepted and ignored would
    make the test above pass for the wrong reason."""
    a, _ = _embed(n, seed=42)
    c, _ = _embed(n, seed=7)
    assert not np.array_equal(a, c), 'changing the seed changed nothing'


@pytest.mark.parametrize('n', [BIG, SMALL],
                         ids=['parallel-knn', 'umap-own-knn'])
def test_separated_populations_stay_separated(n):
    """The check that a malformed precomputed_knn fails. UMAP does not raise
    on a bad neighbour graph — it silently embeds noise — so the only way to
    catch it is to ask whether the output still means anything."""
    emb, truth = _embed(n)
    acc = _label_accuracy(emb, truth)
    assert acc > 0.90, (
        f'kNN label accuracy {acc:.3f} on three well-separated populations; '
        f'the embedding has lost the structure it is supposed to show')


def test_umap_is_not_handed_a_parallel_precomputed_knn():
    """Guards a reverted optimisation from coming back.

    Building the kNN with ``pynndescent(n_jobs=-1)`` and passing it as
    ``precomputed_knn`` runs the neighbour search on every core and is ~10%
    faster. It was tried and reverted: pynndescent's parallel NN-descent is
    deterministic only at a FIXED numba thread count, so with random_state=42
    on 5000x8, thread counts 1 / 4 / 24 produce three different neighbour
    graphs. That would make a seeded embedding depend on the machine it ran
    on — a 24-core workstation, a 4-core laptop and a CI runner disagreeing
    about a published figure.

    This is a fast structural check (UMAP is stubbed, nothing is fitted) so it
    runs on every commit. The end-to-end version below is slow and gated.
    """
    import umap as umap_lib

    seen = {}

    class _Stub:
        def __init__(self, **kw):
            seen.update(kw)

        def fit_transform(self, X):
            return np.zeros((len(X), 2))

    real = umap_lib.UMAP
    umap_lib.UMAP = _Stub
    try:
        s, _ = _sample(BIG)
        s.run_umap(channels=CHANNELS, n_neighbors=15, random_state=42)
    finally:
        umap_lib.UMAP = real

    knn = seen.get('precomputed_knn')
    assert knn in (None, (None, None, None)), (
        'run_umap handed UMAP a precomputed kNN. If that graph was built with '
        'n_jobs != 1 the embedding becomes machine-dependent at a fixed seed; '
        'see the comment in run_umap before re-introducing this.')


@pytest.mark.skipif(not os.environ.get('OPENFLO_RUN_SLOW_TESTS'),
                    reason='spawns interpreters and pays numba JIT in each '
                           '(~2 min) — set OPENFLO_RUN_SLOW_TESTS=1')
def test_embedding_does_not_depend_on_thread_count():
    """The end-to-end form of the guard above: same data, same seed, different
    numba thread counts, in separate processes. Must be byte-identical."""
    code = (
        'import os, sys, warnings; warnings.filterwarnings("ignore");'
        'sys.path.insert(0, "src");'
        'import numba; numba.set_num_threads(int(sys.argv[1]));'
        'import numpy as np, pandas as pd;'
        'from openflo.pipeline import FlowSample;'
        'ch=[f"M{i}" for i in range(8)];'
        'rng=np.random.RandomState(0);'
        'X=np.vstack([rng.normal(c,1.0,(1400,8)) for c in (0.,8.,16.)]);'
        's=FlowSample.from_dataframe(pd.DataFrame(X,columns=ch),name="t");'
        's.fluor_channels=ch;'
        's.run_umap(channels=ch,n_neighbors=15,random_state=42);'
        'e=np.asarray(s.umap_coords);'
        'print(float(np.abs(e).sum()))')
    out = []
    for threads in ('1', '4'):
        r = subprocess.run([sys.executable, '-c', code, threads],
                           capture_output=True, text=True,
                           cwd=str(_repo_root()),
                           env={**os.environ, 'MPLBACKEND': 'Agg',
                                'PYTHONUTF8': '1'})
        assert r.returncode == 0, r.stderr[-2000:]
        out.append(r.stdout.strip().splitlines()[-1])
    assert out[0] == out[1], (
        f'the embedding changed with the numba thread count: {out} — a seeded '
        f'run is no longer reproducible across machines')


def _repo_root():
    import pathlib
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'src' / 'openflo').is_dir():
            return parent
    raise AssertionError('repo root not found')
