"""Each clustering method must recover planted populations in 30-channel data.

PhenoGraph, FlowSOM and the graph methods were built for high-dimensional
cytometry (Levine 2015; Van Gassen 2015), so this is their home ground: a
30-channel synthetic sample (generic channels FL1-A to FL30-A) with eight
populations laid out as a lineage tree, each positive on two or three
markers, sharing markers with its siblings, two with intermediate
expression on a shared marker, and one rare population at 1.5% of events.

Positive half: every method recovers the planted populations (adjusted Rand
index against the truth, and each population's best-matching cluster).
Negative half: the same measures on data with no structure (every marker
negative) and on random labels stay near zero, so the thresholds can fail.

Measured when this test was written (6,000 events, seeds 0 to 3): PhenoGraph,
Leiden and Louvain ARI 0.996 to 0.998, every population F1 >= 0.99; FlowSOM
ARI 0.955 to 0.978. FlowSOM asked for exactly 8 metaclusters merges the
rare population (F1 0.22), as its authors warn; asked for 12 it finds it
(F1 0.99), so that is what the rare-population check uses.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from openflo.pipeline import FlowSample

N_CH = 30
CHANNELS = [f'FL{i}-A' for i in range(1, N_CH + 1)]
# Lineage tree: population -> its positive markers (0-based channel index).
LINEAGE = {0: [0, 1, 2], 1: [0, 1, 3], 2: [0, 4, 5], 3: [0, 4, 6],
           4: [7, 8], 5: [7, 9], 6: [10, 11, 12], 7: [13, 14]}
INTERMEDIATE = {(0, 20): 0.5, (1, 20): 0.5, (2, 21): 0.5}
FRACTIONS = np.array([.25, .18, .15, .12, .10, .10, .085, .015])
RARE = 7


def highdim_sample(seed=0, n=6000, sep=3.5, sd=0.8):
    """(events, true labels). Positive markers sit ``sep`` above negative
    ones; every channel carries Gaussian noise of ``sd``."""
    rng = np.random.default_rng(seed)
    pheno = np.zeros((len(LINEAGE), N_CH))
    for pop, markers in LINEAGE.items():
        pheno[pop, markers] = 1.0
    for (pop, marker), level in INTERMEDIATE.items():
        pheno[pop, marker] = level
    sizes = np.round(FRACTIONS * n).astype(int)
    y = np.repeat(np.arange(len(LINEAGE)), sizes)
    X = pheno[y] * sep + rng.normal(0.0, sd, (len(y), N_CH))
    return X, y


def _cluster(method, X, seed=42, n_meta=8):
    s = FlowSample.from_dataframe(pd.DataFrame(X, columns=CHANNELS))
    with contextlib.redirect_stdout(io.StringIO()):       # PhenoGraph prints
        if method == 'phenograph':
            s.cluster(channels=CHANNELS, k=30, random_state=seed)
            col = 'cluster'
        elif method == 'leiden':
            s.run_leiden(channels=CHANNELS, n_neighbors=15, random_state=seed)
            col = 'leiden'
        elif method == 'louvain':
            s.run_louvain(channels=CHANNELS, n_neighbors=15, restarts=1,
                          random_state=seed)
            col = 'louvain'
        else:
            s.run_flowsom(channels=CHANNELS, grid=(10, 10),
                          n_metaclusters=n_meta, seed=seed)
            col = 'flowsom_meta'
    return s.data[col].to_numpy()


def _best_f1(y, labels, pop):
    """F1 of the cluster that best matches planted population ``pop``."""
    truth = y == pop
    best = 0.0
    for c in np.unique(labels[labels >= 0]):
        pred = labels == c
        tp = int((truth & pred).sum())
        if tp:
            best = max(best, 2 * tp / (truth.sum() + pred.sum()))
    return best


def test_the_sample_is_built_as_designed():
    X, y = highdim_sample()
    assert X.shape == (6000, N_CH)
    assert np.bincount(y).tolist() == np.round(FRACTIONS * 6000).astype(int).tolist()
    assert (y == RARE).mean() == pytest.approx(0.015)
    for pop, markers in LINEAGE.items():
        rows = X[y == pop]
        assert rows[:, markers].mean() == pytest.approx(3.5, abs=0.15)
        negative = [c for c in range(N_CH) if c not in markers
                    and (pop, c) not in INTERMEDIATE]
        assert rows[:, negative].mean() == pytest.approx(0.0, abs=0.1)
        assert rows[:, negative].std() == pytest.approx(0.8, abs=0.05)
    assert X[y == 0, 20].mean() == pytest.approx(1.75, abs=0.15)


@pytest.mark.parametrize('method', ['phenograph', 'leiden', 'louvain'])
def test_graph_methods_recover_every_population_including_the_rare_one(method):
    X, y = highdim_sample()
    labels = _cluster(method, X)
    assert adjusted_rand_score(y, labels) >= 0.95
    for pop in LINEAGE:
        assert _best_f1(y, labels, pop) >= 0.9, pop
    assert 7 <= len(np.unique(labels[labels >= 0])) <= 10


def test_flowsom_recovers_the_populations():
    X, y = highdim_sample()
    labels = _cluster('flowsom', X)
    assert adjusted_rand_score(y, labels) >= 0.9
    for pop in LINEAGE:
        if pop != RARE:
            assert _best_f1(y, labels, pop) >= 0.9, pop


def test_flowsom_finds_the_rare_population_when_overclustering():
    X, y = highdim_sample()
    labels = _cluster('flowsom', X, n_meta=12)
    assert _best_f1(y, labels, RARE) >= 0.9


@pytest.mark.parametrize('method', ['phenograph', 'leiden', 'louvain', 'flowsom'])
def test_no_structure_means_no_recovery(method):
    # Negative half: every marker negative, so there is nothing to find and
    # the same thresholds must fail.
    X, y = highdim_sample(sep=0.0)
    labels = _cluster(method, X)
    assert adjusted_rand_score(y, labels) < 0.05
    assert max(_best_f1(y, labels, pop) for pop in LINEAGE) < 0.5


def test_random_labels_fail_the_recovery_thresholds():
    X, y = highdim_sample()
    labels = np.random.default_rng(0).integers(0, 8, len(y))
    assert adjusted_rand_score(y, labels) < 0.05
    assert _best_f1(y, labels, RARE) < 0.9
