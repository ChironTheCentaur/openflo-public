"""FlowSOM training runs compiled (numba) and must give the SAME weights as
the numpy loop it replaces, bit for bit (speed audit item 4).

Each positive test has a negative partner showing the comparison it relies
on can fail: the data really does tell the summation orders apart, and the
weights really do depend on the inputs.
"""
import numpy as np
import pytest

from openflo import pipeline as P
from openflo.som_kernel import pairwise_sum_kernel, som_epoch_kernel

pytestmark = pytest.mark.skipif(som_epoch_kernel() is None,
                                reason='numba not installed')


def _blobs(seed, n, d, k=4, integer=False):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4, size=(k, d))
    X = np.vstack([rng.normal(loc=c, size=(n // k, d)) for c in centres])
    return np.round(X * 10) if integer else X


@pytest.mark.parametrize('d, integer', [
    (3, False),     # fewer than 8 values: numpy adds them left to right
    (12, False),    # 8 partial sums plus a remainder
    (12, True),     # whole numbers: exact distances, so many exact ties
    (130, False),   # over 128 values: numpy splits the sum in halves
])
def test_compiled_som_weights_equal_the_numpy_loop(d, integer):
    X = _blobs(1, 400, d, integer=integer)
    W_fast, c_fast = P._som_train(X, grid=(4, 5), iters=3, seed=7)
    W_ref, c_ref = P._som_train(X, grid=(4, 5), iters=3, seed=7,
                                use_kernel=False)
    np.testing.assert_array_equal(c_fast, c_ref)
    assert np.array_equal(W_fast, W_ref), np.abs(W_fast - W_ref).max()


def test_som_weights_depend_on_the_data_and_seed():
    """Negative partner: the equality above is not trivially true. Another
    seed or one changed event gives different weights."""
    X = _blobs(1, 400, 12)
    W, _ = P._som_train(X, grid=(4, 5), iters=3, seed=7)
    W_seed, _ = P._som_train(X, grid=(4, 5), iters=3, seed=8)
    X2 = X.copy()
    X2[0, 0] += 1e-9
    W_event, _ = P._som_train(X2, grid=(4, 5), iters=3, seed=7)
    assert not np.array_equal(W, W_seed)
    assert not np.array_equal(W, W_event)


def _numpy_epoch(X, W, order, LH):
    """One epoch of the numpy loop in pipeline._som_train, with the same
    neighbourhood table the compiled epoch takes."""
    for i in order:
        x = X[i]
        bmu = int(np.argmin(((W - x) ** 2).sum(1)))
        W += LH[bmu][:, None] * (x - W)


def _epoch_case(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 3))
    W = rng.normal(size=(5, 3))
    LH = rng.uniform(0.05, 0.6, size=(5, 5))
    return X, W, LH


def test_tied_nodes_pick_the_first_like_numpy():
    """Two identical nodes are exactly as close to every event; np.argmin
    takes the first, and so must the compiled loop (the nodes sit in
    different places on the grid, so the choice moves different weights)."""
    X, W, LH = _epoch_case(5)
    W[3] = W[1]
    X[0] = W[1] + 0.01                      # the tied pair is nearest
    ref = W.copy()
    _numpy_epoch(X, ref, np.arange(6), LH)
    som_epoch_kernel()(X, W, np.arange(6), LH)
    np.testing.assert_array_equal(W, ref)


def test_a_nan_node_is_chosen_like_numpy():
    """np.argmin returns the first NaN, so a node holding NaN (one started
    from an event with a missing value) wins every event."""
    X, W, LH = _epoch_case(6)
    W[2, 1] = np.nan
    ref = W.copy()
    _numpy_epoch(X, ref, np.arange(6), LH)
    som_epoch_kernel()(X, W, np.arange(6), LH)
    np.testing.assert_array_equal(W, ref)


def test_epoch_cases_tell_the_choices_apart():
    """Negative partner: in both cases above, choosing the other node (the
    last of the tied pair, or the nearest finite node instead of the NaN one)
    gives different weights, so the comparisons can fail."""
    X, W, LH = _epoch_case(5)
    W[3] = W[1]
    X[0] = W[1] + 0.01                      # the tied pair is nearest
    a, b = W.copy(), W.copy()
    _numpy_epoch(X, a, np.arange(6), LH)
    for i in range(6):                      # last-of-ties variant
        d = ((b - X[i]) ** 2).sum(1)
        bmu = len(d) - 1 - int(np.argmin(d[::-1]))
        b += LH[bmu][:, None] * (X[i] - b)
    assert not np.array_equal(a, b)

    X, W, LH = _epoch_case(6)
    W[2, 1] = np.nan
    a, b = W.copy(), W.copy()
    _numpy_epoch(X, a, np.arange(6), LH)
    for i in range(6):                      # NaN-skipping variant
        bmu = int(np.nanargmin(((b - X[i]) ** 2).sum(1)))
        b += LH[bmu][:, None] * (X[i] - b)
    assert not np.array_equal(a, b, equal_nan=True)


def test_nan_events_match_the_numpy_loop():
    X = _blobs(2, 200, 5)
    X[17, 2] = np.nan
    W_fast, _ = P._som_train(X, grid=(3, 3), iters=2, seed=3)
    W_ref, _ = P._som_train(X, grid=(3, 3), iters=2, seed=3,
                            use_kernel=False)
    np.testing.assert_array_equal(W_fast, W_ref)


def test_distance_sum_follows_numpys_order():
    """The compiled sum equals numpy's row sum exactly, for every length up
    to 300 (all three branches of numpy's pairwise summation)."""
    psum = pairwise_sum_kernel()
    rng = np.random.default_rng(0)
    for d in range(1, 301):
        M = rng.normal(size=(3, d)) ** 2 * 1e3
        ref = M.sum(1)
        for r in range(3):
            assert 0.0 + psum(M[r], 0, d) == ref[r], d


def test_left_to_right_sum_would_not_match():
    """Negative partner: on the same vectors a plain left-to-right sum differs
    from numpy's in the last bit, so a kernel that summed that way would be
    caught by the test above."""
    rng = np.random.default_rng(0)
    mismatches = 0
    for d in range(8, 301):
        M = rng.normal(size=(3, d)) ** 2 * 1e3
        ref = M.sum(1)
        for r in range(3):
            acc = 0.0
            for v in M[r]:
                acc += float(v)
            mismatches += acc != ref[r]
    assert mismatches > 50


def test_flowsom_node_assignments_unchanged():
    """The compiled and numpy trainings put every event in the same node."""
    X = _blobs(3, 2000, 6)
    W_fast, _ = P._som_train(X, grid=(5, 5), iters=4, seed=42)
    W_ref, _ = P._som_train(X, grid=(5, 5), iters=4, seed=42,
                            use_kernel=False)
    np.testing.assert_array_equal(P._som_assign(X, W_fast),
                                  P._som_assign(X, W_ref))


def _stub_sample(df):
    s = P.FlowSample.__new__(P.FlowSample)
    s.data = df
    s.fluor_channels = list(df.columns)
    s.name = 'synthetic'
    s.flowsom_result = None
    return s


def test_run_flowsom_labels_unchanged(monkeypatch):
    """End to end through FlowSample.run_flowsom: node and metacluster
    labels are the same with and without the compiled loop."""
    import pandas as pd

    import openflo.som_kernel as K
    X = _blobs(4, 3000, 5)
    cols = [f'FL{i}-A' for i in range(1, 6)]
    fast = _stub_sample(pd.DataFrame(X, columns=cols))
    fast.run_flowsom(grid=(6, 6), n_metaclusters=5, iters=4)
    monkeypatch.setattr(K, 'som_epoch_kernel', lambda: None)
    ref = _stub_sample(pd.DataFrame(X, columns=cols))
    ref.run_flowsom(grid=(6, 6), n_metaclusters=5, iters=4)
    for col in ('flowsom', 'flowsom_meta'):
        np.testing.assert_array_equal(fast.data[col], ref.data[col])
