"""Channel transforms (logicle / hyperlog / asinh / log / linear).

Covers the pure transform_values + inverse_transform_values helpers:
known points, monotonicity, and round-trip inversion.
"""
import numpy as np
import pytest

from openflo.pipeline import (
    TRANSFORM_METHODS,
    inverse_transform_values,
    transform_values,
)


def test_methods_listed():
    assert set(TRANSFORM_METHODS) == {
        'logicle', 'hyperlog', 'asinh', 'log', 'linear'}


def test_asinh_known_points_and_cofactor():
    # arcsinh(0)=0; arcsinh(x/c).
    out = transform_values(np.array([0.0, 150.0]), method='asinh', cofactor=150)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(np.arcsinh(1.0))


def test_log_clamps_nonpositive():
    out = transform_values(np.array([-5.0, 0.0, 100.0]), method='log')
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == pytest.approx(2.0)        # log10(100)


def test_linear_passthrough():
    v = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(transform_values(v, method='linear'), v)


@pytest.mark.parametrize('method', ['logicle', 'hyperlog', 'asinh'])
def test_transform_is_monotonic(method):
    v = np.linspace(-100, 50000, 500)
    out = transform_values(v, method=method)
    assert np.all(np.diff(out) >= -1e-9)       # non-decreasing


@pytest.mark.parametrize('method', ['logicle', 'hyperlog', 'asinh', 'linear'])
def test_round_trip_inverse(method):
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 50000, 2000)
    fwd = transform_values(v, method=method)
    back = inverse_transform_values(fwd, method=method)
    assert np.allclose(back, v, rtol=1e-3, atol=1.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        transform_values(np.array([1.0]), method='nope')
