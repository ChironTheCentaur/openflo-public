"""GPU acceleration (CuPy) — fallback exactness + GPU/CPU parity.

The fallback tests run everywhere (no GPU needed) and lock in the guarantee that
matters most: with GPU OFF (the default), every accelerated path is byte-for-byte
the CPU path, so the golden baseline + existing sessions are unaffected. The
parity tests only run where a working CuPy + GPU is present (skipped on CI).
"""
import numpy as np
import pytest

from openflo import gpu_accel
from openflo.pipeline import transform_values

try:
    from flowutils import transforms
    _HAVE_FLOWUTILS = True
except Exception:                                   # noqa: BLE001
    _HAVE_FLOWUTILS = False


@pytest.fixture(autouse=True)
def _gpu_off():
    """Never leak enabled-state between tests (it's a module global)."""
    gpu_accel.set_enabled(False)
    yield
    gpu_accel.set_enabled(False)


def _sweep():
    return np.concatenate([
        np.linspace(-5000, 5000, 4000),
        np.logspace(0, np.log10(262144), 4000),
        -np.logspace(0, np.log10(262144 * 0.9), 4000),
    ])


# ── fallback exactness (GPU off → identical to the CPU path) ────────────────
def test_compensate_off_is_exact_numpy():
    rng = np.random.default_rng(0)
    x = rng.random((1000, 8)) * 1e5
    inv_t = np.linalg.inv(np.eye(8) + rng.random((8, 8)) * 0.05).T
    assert np.array_equal(gpu_accel.compensate(x, inv_t), x @ inv_t)


def test_arcsinh_off_is_exact_numpy():
    x = _sweep()
    assert np.array_equal(gpu_accel.arcsinh(x, 150.0), np.arcsinh(x / 150.0))


def test_interp_off_is_exact_numpy():
    xg = np.linspace(-1, 1, 500)
    yg = xg ** 3
    x = np.linspace(-1, 1, 777)
    assert np.array_equal(gpu_accel.interp(x, xg, yg), np.interp(x, xg, yg))


@pytest.mark.skipif(not _HAVE_FLOWUTILS, reason="flowutils not installed")
@pytest.mark.parametrize("method", ["logicle", "hyperlog"])
def test_transform_off_is_exact_flowutils(method):
    x = _sweep()
    fn = getattr(transforms, method)
    ref = fn(x.reshape(-1, 1), channel_indices=[0]).flatten()
    assert np.array_equal(transform_values(x, method=method), ref)


# ── GPU/CPU parity (only where a GPU + CuPy exist; skipped on CI) ────────────
_NO_GPU = not gpu_accel.gpu_available()
gpu = pytest.mark.skipif(_NO_GPU, reason="no NVIDIA GPU / CuPy available")


@gpu
def test_gpu_compensate_matches_cpu():
    rng = np.random.default_rng(0)
    x = rng.random((50000, 12)) * 1e5
    inv_t = np.linalg.inv(np.eye(12) + rng.random((12, 12)) * 0.05).T
    ref = x @ inv_t
    assert gpu_accel.set_enabled(True)
    g = gpu_accel.compensate(x, inv_t)
    # float32 on the GPU vs float64 CPU: tiny absolute error on ~1e5 values.
    assert np.allclose(g, ref, rtol=1e-3, atol=1.0)


@gpu
def test_gpu_arcsinh_matches_cpu():
    x = _sweep()
    ref = np.arcsinh(x / 150.0)
    assert gpu_accel.set_enabled(True)
    assert np.max(np.abs(gpu_accel.arcsinh(x, 150.0) - ref)) < 1e-4


@gpu
@pytest.mark.skipif(not _HAVE_FLOWUTILS, reason="flowutils not installed")
@pytest.mark.parametrize("method", ["logicle", "hyperlog"])
def test_gpu_biexp_matches_flowutils(method):
    x = _sweep()
    fn = getattr(transforms, method)
    ref = fn(x.reshape(-1, 1), channel_indices=[0]).flatten()
    assert gpu_accel.set_enabled(True)
    # LUT-interp on the GPU reproduces flowutils to ~1e-8 of scale.
    assert np.max(np.abs(transform_values(x, method=method) - ref)) < 1e-5
