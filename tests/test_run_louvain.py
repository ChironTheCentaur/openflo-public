"""FlowSample.run_louvain — seeded, in-process Louvain on the SNN/Jaccard graph.

It had no test at all (2% line coverage): no CLI flag or GUI path calls it, so
nothing in the suite ever ran it. These pin what its docstring promises:

* it recovers well-separated populations;
* the same ``random_state`` gives the same labels, AND the seed is actually
  consumed (on an ambiguous fixture different seeds give different
  partitions, so the determinism check cannot pass vacuously);
* ``restarts`` keeps the HIGHEST-modularity run of the seed sequence
  ``random_state, random_state + 1, ...``;
* non-finite rows get -1 and the subsample path labels every finite event.
"""
import numpy as np
import pandas as pd

from openflo.pipeline import FlowSample, _snn_jaccard_graph

CH = ['A', 'B', 'C']


def _separated(n_per=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.4, (n_per, 3)) for c in (0.0, 6.0, 12.0)])
    return pd.DataFrame(X, columns=CH), np.repeat([0, 1, 2], n_per)


def _ambiguous():
    """Four heavily overlapping populations: Louvain's local optimum depends on
    its node order, so different seeds land on different partitions (measured:
    6 distinct partitions for seeds 0..5)."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 1.2, (100, 3)) for c in (0.0, 2.5, 5.0, 7.5)])
    return pd.DataFrame(X, columns=CH)


def _louvain(df, **kw):
    s = FlowSample.from_dataframe(df.copy())
    s.run_louvain(channels=CH, **kw)
    return s.data['louvain'].to_numpy()


def _pure(lab, truth):
    for c in np.unique(lab):
        sub = truth[lab == c]
        if np.bincount(sub).max() / sub.size < 0.98:
            return False
    return True


def test_louvain_recovers_separated_populations():
    df, truth = _separated()
    lab = _louvain(df, n_neighbors=15, restarts=2)
    assert (lab >= 0).all()
    # Modularity may split a blob in two (measured: 6 clusters here), which
    # is legitimate; what must hold is that no cluster MIXES blobs.
    assert len(np.unique(lab)) >= 3
    assert _pure(lab, truth)


def test_same_seed_same_labels_and_the_seed_is_used():
    df = _ambiguous()
    a = _louvain(df, n_neighbors=10, restarts=1, random_state=3)
    b = _louvain(df, n_neighbors=10, restarts=1, random_state=3)
    assert np.array_equal(a, b)
    parts = {tuple(_louvain(df, n_neighbors=10, restarts=1, random_state=rs))
             for rs in range(6)}
    assert len(parts) >= 3, 'seed has no effect — the determinism check is vacuous'


def test_restarts_keep_the_highest_modularity_seed():
    df = _ambiguous()
    g = _snn_jaccard_graph(df[CH].to_numpy(dtype=float), 10)
    singles = [_louvain(df, n_neighbors=10, restarts=1, random_state=rs)
               for rs in range(6)]
    qs = [g.modularity(lab.tolist(), weights='weight') for lab in singles]
    best = _louvain(df, n_neighbors=10, restarts=6, random_state=0)
    assert np.array_equal(best, singles[int(np.argmax(qs))])
    assert g.modularity(best.tolist(), weights='weight') == max(qs)


def test_nonfinite_rows_are_noise_and_subsample_labels_everything():
    df, truth = _separated(n_per=300, seed=1)
    df.iloc[0, 0] = np.nan
    df.iloc[1, 2] = -np.inf
    lab = _louvain(df, n_neighbors=15, restarts=1, max_events=400)
    assert lab[0] == -1 and lab[1] == -1
    assert (lab[2:] >= 0).all()
    assert _pure(lab[2:], truth[2:])


def test_no_usable_channels_or_too_few_events_is_a_noop():
    s = FlowSample.from_dataframe(pd.DataFrame({'X': [1.0, 2.0, 3.0]}))
    s.run_louvain(channels=['missing'])
    assert 'louvain' not in s.data.columns
    s = FlowSample.from_dataframe(pd.DataFrame({'A': [1.0, 2.0], 'B': [1.0, 2.0]}))
    s.run_louvain(channels=['A', 'B'])
    assert 'louvain' not in s.data.columns
