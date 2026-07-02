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
