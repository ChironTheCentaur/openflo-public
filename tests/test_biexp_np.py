"""The numpy logicle / hyperlog (`openflo.biexp_np`) gives FlowUtils' numbers.

It replaces FlowUtils' compiled transforms where they cannot be built, so the
bar is the reference's own output: on a dense grid over [-10T, 100T] (zero,
tiny values of both signs and +-T included) the forward transforms agree to
1e-12 of scale and the inverses to 1e-12 relative, for the default scale and
for others with a nonzero A. The reference's quirks are part of its output and
are matched too: a non-finite input maps to -1.0, zero to x1, a 1-D input
comes back float64 and a 2-D one keeps its dtype.

The round-trip tests need no FlowUtils, so they hold on the platforms the
port exists for.
"""

import numpy as np
import pytest

from openflo import biexp_np
from openflo.scales import LOGICLE_A, LOGICLE_M, LOGICLE_T, LOGICLE_W
from tests.conftest import import_flowutils_unless_browser

# (t, m, w, a): the defaults, a nonzero A of either sign, another top of
# scale, and W = 0 (logicle becomes a pure arcsinh-like scale there).
PARAMS = [
    (LOGICLE_T, LOGICLE_M, LOGICLE_W, LOGICLE_A),
    (LOGICLE_T, 4.5, 0.5, 1.0),
    (LOGICLE_T, 4.5, 0.5, -0.25),
    (2.0 ** 22, 5.4, 1.2, 0.3),
    (10000.0, 4.0, 1.0, 0.0),
]
FORWARD_TOL = 1e-12     # absolute, in scale units
INVERSE_RTOL = 1e-12    # relative, floored at 1 in data units


def _grid(t):
    """>= 1e6 linear values over [-10T, 100T] plus zero, signed zeros,
    subnormals, +-T and log-spaced tiny-to-large values of both signs."""
    dense = np.linspace(-10 * t, 100 * t, 1_000_001)
    special = [0.0, -0.0, t, -t, 1e-300, -1e-300, 5e-324, -5e-324, 1.0, -1.0]
    pos = np.logspace(-300, np.log10(100 * t), 4000)
    return np.concatenate([dense, special, pos, -pos[pos <= 10 * t]])


@pytest.fixture(scope='module')
def fu():
    return import_flowutils_unless_browser('transforms')


@pytest.mark.parametrize('params', PARAMS)
@pytest.mark.parametrize('name', ['logicle', 'hyperlog'])
def test_forward_matches_flowutils(fu, name, params):
    t, m, w, a = params
    x = _grid(t).reshape(-1, 1)
    want = getattr(fu, name)(x, [0], t, m, w, a)
    got = getattr(biexp_np, name)(x, [0], t, m, w, a)
    assert np.isfinite(want).all()
    assert np.array_equal(want == -1.0, got == -1.0)
    assert np.max(np.abs(got - want)) <= FORWARD_TOL


@pytest.mark.parametrize('params', PARAMS)
@pytest.mark.parametrize('name', ['logicle_inverse', 'hyperlog_inverse'])
def test_inverse_matches_flowutils(fu, name, params):
    t, m, w, a = params
    s = np.concatenate([np.linspace(-1.5, 1.6, 1_000_001),
                        [0.0, a / (m + a), (a + w) / (m + a), 1.0]])
    want = getattr(fu, name)(s.reshape(-1, 1), [0], t, m, w, a)
    got = getattr(biexp_np, name)(s.reshape(-1, 1), [0], t, m, w, a)
    err = np.abs(got - want) / np.maximum(np.abs(want), 1.0)
    assert np.max(err) <= INVERSE_RTOL


@pytest.mark.parametrize('name, w', [('logicle', 0.0), ('hyperlog', 1e-6)])
def test_slow_convergence_matches_flowutils(fu, name, w):
    """Scales whose iteration converges slowly for some inputs: logicle at
    W = 0 (no root solve for d) and hyperlog at a tiny W. There inputs take
    up to and past the iteration cap, so which ones read -1.0 (did not
    converge) pins the cap itself, not only the root."""
    t = LOGICLE_T
    tiny = np.logspace(-300, 9, 400_001)
    x = np.concatenate([_grid(t), tiny, -tiny])
    want = getattr(fu, name)(x, None, t, 4.5, w, 0.0)
    got = getattr(biexp_np, name)(x, None, t, 4.5, w, 0.0)
    assert (want == -1.0).any()
    assert np.array_equal(want == -1.0, got == -1.0)
    assert np.max(np.abs(got - want)) <= FORWARD_TOL


@pytest.mark.parametrize('name', ['logicle', 'hyperlog', 'logicle_inverse',
                                  'hyperlog_inverse'])
def test_edge_values_match_flowutils(fu, name):
    x = np.array([np.nan, np.inf, -np.inf, 1e308, -1e308, 0.0, -0.0, 5.0,
                  1e20])
    want = getattr(fu, name)(x, None)
    got = getattr(biexp_np, name)(x, None)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('name', ['logicle', 'hyperlog', 'logicle_inverse',
                                  'hyperlog_inverse'])
def test_array_handling_matches_flowutils(fu, name):
    # Scale values for an inverse are small, so its integers are too.
    rows = ([[1, 0], [0, -1], [0, 1]] if name.endswith('_inverse')
            else [[1, 200], [3, -400], [0, 5000]])
    ints = np.array(rows)
    floats = ints.astype(float) / 7
    for data, idx in [(ints, [1]), (ints, None), (floats, [0]),
                      (floats, None), (floats[:, 1], None),
                      (ints[:, 1], None)]:
        want = getattr(fu, name)(data, idx)
        got = getattr(biexp_np, name)(data, idx)
        assert got.dtype == want.dtype
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    # The input is never written to.
    np.testing.assert_array_equal(ints, rows)


# ── Without FlowUtils ────────────────────────────────────────────────────────

@pytest.mark.parametrize('params', PARAMS)
@pytest.mark.parametrize('name', ['logicle', 'hyperlog'])
def test_round_trip_data_scale_data(name, params):
    t, m, w, a = params
    x = _grid(t)
    fwd = getattr(biexp_np, name)(x, None, t, m, w, a)
    back = getattr(biexp_np, name + '_inverse')(fwd, None, t, m, w, a)
    assert (fwd != -1.0).all()
    err = np.abs(back - x) / np.maximum(np.abs(x), 1.0)
    assert np.max(err) <= 1e-9


@pytest.mark.parametrize('params', PARAMS)
@pytest.mark.parametrize('name', ['logicle', 'hyperlog'])
def test_round_trip_scale_data_scale(name, params):
    t, m, w, a = params
    s = np.linspace(-1.0, 1.5, 200_001)
    data = getattr(biexp_np, name + '_inverse')(s, None, t, m, w, a)
    back = getattr(biexp_np, name)(data, None, t, m, w, a)
    assert np.max(np.abs(back - s)) <= FORWARD_TOL


@pytest.mark.parametrize('name', ['logicle', 'hyperlog'])
def test_scale_landmarks(name):
    """Zero maps to x1 = (W + A) / (M + A) and T to 1; the transform is odd
    about x1 and increasing."""
    t, m, w, a = LOGICLE_T, 4.5, 0.5, 0.0
    fn = getattr(biexp_np, name)
    x1 = (w + a) / (m + a)
    zero, top = fn(np.array([0.0, t]), None, t, m, w, a)
    assert zero == x1
    assert top == pytest.approx(1.0, abs=1e-12)
    v = np.logspace(-3, 7, 1001)
    up = fn(v, None, t, m, w, a)
    down = fn(-v, None, t, m, w, a)
    np.testing.assert_allclose(up - x1, x1 - down, rtol=0, atol=1e-12)
    assert (np.diff(up) > 0).all()


@pytest.mark.parametrize('name', ['logicle', 'hyperlog'])
def test_non_finite_input_reads_minus_one(name):
    out = getattr(biexp_np, name)(np.array([np.nan, np.inf, -np.inf]), None)
    np.testing.assert_array_equal(out, [-1.0, -1.0, -1.0])
