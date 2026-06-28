"""Optional NVIDIA-GPU acceleration (CuPy) for the heavy load math.

Scope (first pass): the compensation matmul and the arcsinh transform — the two
operations that dominate per-sample load time and map cleanly to the GPU. The
logicle transform (flowutils' CPU C-library) and the QC histogram path stay on
the CPU for now.

OFF by default; toggled via Preferences ('use_gpu'). When disabled, unavailable,
or on ANY error, every function falls back to the exact numpy path — so enabling
the flag is the ONLY thing that can change results, and disabling it restores
bitwise-identical CPU behaviour (the golden baseline runs with it OFF).

Note on precision: the GPU path computes in float32. Consumer GPUs throttle
float64 to a fraction of float32 throughput, so float64 on the GPU would be
*slower* than the CPU; float32 is what makes it a ~10x win. That introduces
~1e-6 relative differences vs the CPU's float64 — acceptable for gating, and
exactly why this is opt-in.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_cupy: Any = None   # None = not yet probed, False = unavailable, module = ready
_enabled = False


def _probe() -> Any:
    """Return the cupy module if a working GPU + CUDA libs are present, else
    None. Cached after the first probe."""
    global _cupy
    if _cupy is None:
        try:
            import cupy as cp
            # Force the CUDA runtime + cuBLAS to actually load (a bare import
            # succeeds even when the libs are missing; a real op surfaces it).
            float(cp.asarray([1.0, 2.0]).sum())
            _cupy = cp
        except Exception:
            _cupy = False
    return _cupy or None


def gpu_available() -> bool:
    """True if CuPy + a usable CUDA GPU are present (independent of the flag)."""
    return _probe() is not None


def set_enabled(on: bool) -> bool:
    """Enable GPU acceleration iff requested AND a working GPU is present.
    Returns the effective state (so callers can report 'requested but no GPU')."""
    global _enabled
    _enabled = bool(on) and gpu_available()
    return _enabled


def enabled() -> bool:
    return _enabled


def compensate(values, inv_t):
    """``values @ inv_t`` — the compensation matmul. GPU (float32) when enabled,
    else the exact numpy path. ``values`` is (N, C); returns a host array of the
    same dtype as ``values``."""
    if _enabled:
        cp = _probe()
        if cp is not None:
            try:
                a = cp.asarray(values, dtype=cp.float32)
                b = cp.asarray(inv_t, dtype=cp.float32)
                out = cp.asnumpy(a @ b)
                return out.astype(values.dtype, copy=False)
            except Exception:
                pass
    return np.asarray(values) @ np.asarray(inv_t)


def interp(x, xgrid, ygrid):
    """1-D monotone interpolation (``xgrid`` strictly increasing). GPU when
    enabled, else numpy. Backs the LUT-based logicle / hyperlog forward
    transform: the slow per-event flowutils root-find is replaced by interp
    against a table built from flowutils' exact inverse (matches it to ~1e-8 of
    scale), which is GPU-trivial and fast on millions of events."""
    if _enabled:
        cp = _probe()
        if cp is not None:
            try:
                return cp.asnumpy(
                    cp.interp(cp.asarray(x), cp.asarray(xgrid),
                              cp.asarray(ygrid)))
            except Exception:
                pass
    return np.interp(x, xgrid, ygrid)


def arcsinh(values, cofactor):
    """``arcsinh(values / cofactor)`` — the asinh transform. GPU (float32) when
    enabled, else the exact numpy path. Returns a host array."""
    if _enabled:
        cp = _probe()
        if cp is not None:
            try:
                a = cp.asarray(values, dtype=cp.float32)
                return cp.asnumpy(cp.arcsinh(a / np.float32(cofactor)))
            except Exception:
                pass
    return np.arcsinh(np.asarray(values) / float(cofactor))
