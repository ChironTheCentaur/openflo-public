"""FlowSOM clustering (SOM + metaclustering).

Covers the pure helpers (_som_train / _som_assign / _som_metacluster) and
FlowSample.run_flowsom end-to-end on well-separated synthetic blobs.
"""

import numpy as np
import pandas as pd

import openflo.pipeline as fp


def _three_blobs(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0], 0.3, (n, 2))
    b = rng.normal([8, 8], 0.3, (n, 2))
    c = rng.normal([0, 8], 0.3, (n, 2))
    X = np.vstack([a, b, c])
    rng.shuffle(X)
    return X


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_som_train_shapes():
    X = _three_blobs()
    W, coords = fp._som_train(X, grid=(6, 6), iters=5)
    assert W.shape == (36, 2)
    assert coords.shape == (36, 2)


def test_som_assign_returns_valid_nodes():
    X = _three_blobs()
    W, _ = fp._som_train(X, grid=(5, 5), iters=5)
    nodes = fp._som_assign(X, W)
    assert nodes.shape == (len(X),)
    assert nodes.min() >= 0 and nodes.max() < 25


def test_som_metacluster_label_range():
    W = np.random.default_rng(1).normal(size=(25, 4))
    labels = fp._som_metacluster(W, 5)
    assert labels.shape == (25,)
    assert set(np.unique(labels)) <= set(range(5))


def test_som_metacluster_handles_k1():
    W = np.zeros((4, 3))
    assert set(fp._som_metacluster(W, 1)) == {0}


# ── run_flowsom integration ──────────────────────────────────────────────────

def _stub_sample(df, fluor):
    s = fp.FlowSample.__new__(fp.FlowSample)
    s.data = df
    s.fluor_channels = list(fluor)
    s.name = 'fs'
    s.flowsom_result = None
    return s


def test_run_flowsom_writes_columns_and_separates_blobs():
    X = _three_blobs(n=1200)
    df = pd.DataFrame({'M1': X[:, 0], 'M2': X[:, 1]})
    s = _stub_sample(df, ['M1', 'M2'])
    s.run_flowsom(grid=(6, 6), n_metaclusters=3, iters=8, seed=0)

    assert 'flowsom' in s.data.columns and 'flowsom_meta' in s.data.columns
    assert s.flowsom_result['n_metaclusters'] == 3
    # Three well-separated blobs → 3 metaclusters, each dominated by one
    # spatial region. Check the metacluster assignment is pure-ish per blob:
    metas = s.data['flowsom_meta'].to_numpy()
    assert set(np.unique(metas)) <= {0, 1, 2}
    # Each metacluster should be spatially tight (a real separation, not noise).
    for mc in np.unique(metas):
        pts = X[metas == mc]
        assert pts[:, 0].std() < 2.0 and pts[:, 1].std() < 2.0


def test_run_flowsom_marks_nonfinite_as_minus_one():
    X = _three_blobs(n=400)
    df = pd.DataFrame({'M1': X[:, 0], 'M2': X[:, 1]})
    df.loc[0, 'M1'] = np.nan
    s = _stub_sample(df, ['M1', 'M2'])
    s.run_flowsom(grid=(5, 5), n_metaclusters=3, iters=5)
    assert s.data['flowsom'].iloc[0] == -1
    assert s.data['flowsom_meta'].iloc[0] == -1


def test_run_flowsom_no_channels_noop():
    s = _stub_sample(pd.DataFrame({'X': [1.0, 2.0]}), [])
    s.run_flowsom()
    assert s.flowsom_result is None
    assert 'flowsom' not in s.data.columns
