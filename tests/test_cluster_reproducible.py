"""FlowSample.cluster(reproducible=True) routes PhenoGraph to its seeded Leiden
backend, so a re-run on identical input + random_state gives identical labels.
The default (Louvain) path is time-seeded and can drift run-to-run, so it is
intentionally NOT asserted deterministic here."""
import numpy as np
import pandas as pd
import pytest


def _have_seeded_phenograph():
    try:
        import inspect

        import phenograph
        p = set(inspect.signature(phenograph.cluster).parameters)
        return {'clustering_algo', 'seed'} <= p
    except Exception:                                   # noqa: BLE001
        return False


@pytest.mark.skipif(not _have_seeded_phenograph(),
                    reason='phenograph with seeded Leiden not available')
def test_cluster_reproducible_mode_is_deterministic():
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    # Three well-separated blobs → clustering has real structure to find.
    X = np.vstack([rng.normal(m, 0.3, (200, 4)) for m in (0.0, 6.0, 12.0)])
    df = pd.DataFrame(X, columns=[f'M{i}-A' for i in range(4)])

    def run():
        s = FlowSample.from_dataframe(df.copy(), name='s')
        s.cluster(channels=list(df.columns), k=15,
                  reproducible=True, random_state=42)
        return np.asarray(s.data['cluster'])

    a, b = run(), run()
    assert np.array_equal(a, b)              # identical labels across runs
    assert len(set(a.tolist())) >= 2         # actually recovered structure


def _blobs(n_per=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(m, 0.25, (n_per, 4)) for m in (0.0, 8.0, 16.0)])
    truth = np.repeat([0, 1, 2], n_per)
    df = pd.DataFrame(X, columns=[f'M{i}-A' for i in range(4)])
    return df, truth


def _purity_ok(labels, truth):
    """No finite event left at -1, and every CLUSTER is ~one true blob (>95%).
    Uses cluster->blob purity (not blob->label) so legitimate over-clustering
    passes, while a label-shuffle / inverted-mask / bad KD-tree assignment (which
    would MIX blobs within a cluster) fails. Requires >= n_blobs clusters so an
    all-merged-into-one degenerate result also fails."""
    if (labels < 0).any():
        return False
    for cl in np.unique(labels):
        sub = truth[labels == cl]
        _, counts = np.unique(sub, return_counts=True)
        if counts.max() / sub.size <= 0.95:
            return False
    return len(np.unique(labels)) >= len(np.unique(truth))


@pytest.mark.skipif(not _have_seeded_phenograph(),
                    reason='phenograph with seeded Leiden not available')
def test_cluster_labels_are_pure_on_separated_blobs():
    """cluster() gives one blob one label and distinct blobs distinct labels,
    and never leaves a finite event at -1 — the label-write mask was otherwise
    unasserted (only run_leiden had a purity test)."""
    from openflo.pipeline import FlowSample
    df, truth = _blobs()
    s = FlowSample.from_dataframe(df, name='s')
    s.cluster(channels=list(df.columns), k=15, reproducible=True, random_state=42)
    assert _purity_ok(np.asarray(s.data['cluster']), truth)


@pytest.mark.skipif(not _have_seeded_phenograph(),
                    reason='phenograph with seeded Leiden not available')
def test_cluster_subsample_assigns_all_events_purely():
    """cluster(max_events=N) clusters a subsample then KD-tree 1-NN assigns the
    rest; every finite event must be labelled and purity must hold (guards the
    subsample-assignment index path)."""
    from openflo.pipeline import FlowSample
    df, truth = _blobs(n_per=200, seed=1)          # 600 events
    s = FlowSample.from_dataframe(df, name='s')
    s.cluster(channels=list(df.columns), k=15, max_events=300,
              reproducible=True, random_state=42)
    assert _purity_ok(np.asarray(s.data['cluster']), truth)


@pytest.mark.skipif(not _have_seeded_phenograph(),
                    reason='phenograph with seeded Leiden not available')
def test_cluster_marks_nonfinite_as_noise():
    """Rows with NaN/Inf are excluded from clustering and labelled -1 (parity
    with run_leiden/run_flowsom, which each have a dedicated -1 test)."""
    from openflo.pipeline import FlowSample
    df, _ = _blobs(n_per=100, seed=0)
    df.iloc[5, 0] = np.nan
    df.iloc[7, 1] = np.inf
    s = FlowSample.from_dataframe(df, name='s')
    s.cluster(channels=list(df.columns), k=15, reproducible=True, random_state=42)
    lab = np.asarray(s.data['cluster'])
    assert lab[5] == -1 and lab[7] == -1
    finite = np.ones(len(lab), bool)
    finite[[5, 7]] = False
    assert (lab[finite] >= 0).all()          # every finite row got a real label
