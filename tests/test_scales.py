"""Tests for openflo.scales.view_funcs — axis display-scale transforms."""
from __future__ import annotations

import numpy as np
import pytest

from openflo.scales import view_funcs


def test_linear_transform_returns_none():
    # a linearly-stored channel uses matplotlib's native scale → no FuncScale
    assert view_funcs('linear', 'log') is None
    assert view_funcs('linear', 'symlog', data_sample=np.arange(10.0)) is None


def test_view_funcs_roundtrip():
    """forward maps stored→screen, inverse screen→stored; their composition is
    the identity for every display scale (the whole point — no double
    transform). Uses 'asinh' (pure maths)."""
    pytest.importorskip('flowutils')           # importing pipeline needs it
    from openflo.pipeline import transform_values
    intensity = np.array([10.0, 100.0, 1000.0, 5000.0])
    stored = transform_values(intensity, method='asinh')   # stored coords
    for scale in ('linear', 'log', 'symlog'):
        fns = view_funcs('asinh', scale, data_sample=stored)
        assert fns is not None
        fwd, inv = fns
        back = inv(fwd(stored))
        assert np.allclose(back, stored, rtol=1e-4, atol=1e-4), scale


def test_view_funcs_shape_preserved():
    """FuncScale requires shape-preserving callables (tick queries arrive as
    e.g. (1, 1))."""
    pytest.importorskip('flowutils')
    fwd, inv = view_funcs('asinh', 'log')
    out = fwd(np.array([[1.0, 2.0, 3.0]]))
    assert out.shape == (1, 3)
    assert np.all(np.isfinite(out))            # NaNs scrubbed to 0
