"""Optional GPU acceleration for the heavy load math — vendor-portable.

Scope (unchanged): the compensation matmul, the LUT interp behind the logicle/
hyperlog forward transform, and the arcsinh transform — the operations that
dominate per-sample load time and map cleanly to the GPU. The logicle C-library
root-find and the QC histogram path stay on the CPU.

Backends (a *provider* wraps a vendor stack; first one that works wins under the
selected preference):

  * **cupy**  — NVIDIA / CUDA only. Fastest on NVIDIA. (native Windows + Linux)
  * **torch** — vendor-portable. Picks the best device it can find:
        cuda  → NVIDIA, and AMD via a ROCm torch build (ROCm exposes torch.cuda)
        xpu   → Intel Arc / Data Center GPU Max (Intel Extension for PyTorch)
        mps   → Apple Metal
        dml   → torch-directml: ANY Direct3D-12 GPU on Windows (AMD/Intel/NVIDIA)
    This is what gives AMD/Intel coverage — especially DirectML, the only path
    that reaches AMD/Intel GPUs on *native Windows* (ROCm is Linux-only).

Backend preference ('gpu_backend'): auto | cupy | torch | off.
  auto  → prefer cupy (NVIDIA-optimal), else torch with a non-CPU device.
The env var OPENFLO_GPU_BACKEND overrides the default at import.

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

import os
from typing import Any

import numpy as np

_enabled = False
# auto | cupy | torch | off
_backend_pref = (os.environ.get("OPENFLO_GPU_BACKEND") or "auto").strip().lower()
_provider: Any = None      # None = not yet probed, False = none available, else a _Provider
_PROBED = False


# --------------------------------------------------------------------------- #
# Providers. Each wraps one vendor stack and returns *host* numpy arrays. Any
# exception inside a provider op propagates to the caller, which falls back to
# numpy — so a provider never has to be perfectly complete to be safe.
# --------------------------------------------------------------------------- #
class _CuPyProvider:
    name = "cupy"

    def __init__(self, cp: Any):
        self._cp = cp

    def device_label(self) -> str:
        try:
            props = self._cp.cuda.runtime.getDeviceProperties(0)
            return props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        except Exception:
            return "CUDA GPU"

    def compensate(self, values, inv_t):
        cp = self._cp
        a = cp.asarray(values, dtype=cp.float32)
        b = cp.asarray(inv_t, dtype=cp.float32)
        return cp.asnumpy(a @ b).astype(np.asarray(values).dtype, copy=False)

    def interp(self, x, xgrid, ygrid):
        cp = self._cp
        return cp.asnumpy(cp.interp(cp.asarray(x), cp.asarray(xgrid), cp.asarray(ygrid)))

    def arcsinh(self, values, cofactor):
        cp = self._cp
        a = cp.asarray(values, dtype=cp.float32)
        return cp.asnumpy(cp.arcsinh(a / np.float32(cofactor)))


class _TorchProvider:
    name = "torch"

    def __init__(self, torch: Any, device: Any, label: str):
        self._torch = torch
        self._dev = device
        self._label = label

    def device_label(self) -> str:
        return self._label

    def _t(self, arr):
        return self._torch.as_tensor(np.asarray(arr), dtype=self._torch.float32, device=self._dev)

    def compensate(self, values, inv_t):
        a = self._t(values)
        b = self._t(inv_t)
        out = (a @ b).detach().to("cpu").numpy()
        return out.astype(np.asarray(values).dtype, copy=False)

    def interp(self, x, xgrid, ygrid):
        # np.interp for a strictly-increasing xgrid: bracket each x and lerp,
        # clamping to the endpoints outside the grid (np.interp's behaviour).
        torch = self._torch
        xt, xg, yg = self._t(x), self._t(xgrid), self._t(ygrid)
        n = xg.numel()
        idx = torch.searchsorted(xg, xt).clamp(1, n - 1)
        x0, x1 = xg[idx - 1], xg[idx]
        y0, y1 = yg[idx - 1], yg[idx]
        denom = x1 - x0
        frac = torch.where(denom != 0, (xt - x0) / denom, torch.zeros_like(xt))
        y = y0 + frac * (y1 - y0)
        y = torch.where(xt <= xg[0], yg[0], y)
        y = torch.where(xt >= xg[-1], yg[-1], y)
        return y.detach().to("cpu").numpy()

    def arcsinh(self, values, cofactor):
        a = self._t(values)
        return self._torch.asinh(a / float(cofactor)).detach().to("cpu").numpy()


# --------------------------------------------------------------------------- #
# Provider resolution
# --------------------------------------------------------------------------- #
def _try_cupy() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
        # Force the CUDA runtime + cuBLAS to actually load (a bare import
        # succeeds even when the libs are missing; a real op surfaces it).
        float(cp.asarray([1.0, 2.0]).sum())
        return _CuPyProvider(cp)
    except Exception:
        return None


def _torch_device(torch: Any):
    """Return (device, label) for the best available non-CPU torch device, or
    (None, None). Order: CUDA/ROCm → Intel XPU → Apple MPS → DirectML."""
    try:
        if torch.cuda.is_available():
            label = torch.cuda.get_device_name(0)
            # A ROCm build reports through the CUDA API; tag it so the label is honest.
            if getattr(torch.version, "hip", None):
                label = f"{label} (ROCm)"
            return torch.device("cuda"), label
    except Exception:
        pass
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            try:
                label = torch.xpu.get_device_name(0)
            except Exception:
                label = "Intel XPU"
            return torch.device("xpu"), label
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps"), "Apple MPS"
    except Exception:
        pass
    try:
        import torch_directml as dml  # type: ignore[import-not-found]
        if dml.is_available():
            dev = dml.device()
            try:
                label = f"{dml.device_name(0)} (DirectML)"
            except Exception:
                label = "DirectML GPU"
            return dev, label
    except Exception:
        pass
    return None, None


def _try_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
        dev, label = _torch_device(torch)
        if dev is None:
            return None
        prov = _TorchProvider(torch, dev, label or "GPU")
        # Smoke-test a real op so a broken device surfaces here, not mid-load.
        prov.arcsinh(np.asarray([1.0, 2.0], dtype=np.float32), 1.0)
        return prov
    except Exception:
        return None


def _resolve_provider() -> Any:
    """Resolve and cache the provider for the current backend preference. Returns
    a _Provider or None."""
    global _provider, _PROBED
    if _PROBED:
        return _provider or None
    pref = _backend_pref
    prov: Any = None
    if pref == "off":
        prov = None
    elif pref == "cupy":
        prov = _try_cupy()
    elif pref == "torch":
        prov = _try_torch()
    else:  # auto: NVIDIA-optimal first, then portable torch
        prov = _try_cupy() or _try_torch()
    _provider = prov if prov is not None else False
    _PROBED = True
    return prov


# --------------------------------------------------------------------------- #
# Public API (stable)
# --------------------------------------------------------------------------- #
def set_backend(name: str) -> None:
    """Select the backend preference (auto|cupy|torch|off) and re-probe."""
    global _backend_pref, _provider, _PROBED, _enabled
    _backend_pref = (name or "auto").strip().lower()
    _provider, _PROBED = None, False
    # an enabled flag may now point at a different (or no) provider
    _enabled = _enabled and (_resolve_provider() is not None)


def backend_name() -> str | None:
    """The resolved backend ('cupy'/'torch'), or None if none is available."""
    prov = _resolve_provider()
    return prov.name if prov is not None else None


def device_name() -> str | None:
    """Human-readable device label for the resolved backend, or None."""
    prov = _resolve_provider()
    return prov.device_label() if prov is not None else None


def gpu_available() -> bool:
    """True if a working GPU provider is present under the current preference
    (independent of the use_gpu flag)."""
    return _resolve_provider() is not None


def set_enabled(on: bool) -> bool:
    """Enable GPU acceleration iff requested AND a working provider is present.
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
        prov = _resolve_provider()
        if prov is not None:
            try:
                return prov.compensate(values, inv_t)
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
        prov = _resolve_provider()
        if prov is not None:
            try:
                return prov.interp(x, xgrid, ygrid)
            except Exception:
                pass
    return np.interp(x, xgrid, ygrid)


def arcsinh(values, cofactor):
    """``arcsinh(values / cofactor)`` — the asinh transform. GPU (float32) when
    enabled, else the exact numpy path. Returns a host array."""
    if _enabled:
        prov = _resolve_provider()
        if prov is not None:
            try:
                return prov.arcsinh(values, cofactor)
            except Exception:
                pass
    return np.arcsinh(np.asarray(values) / float(cofactor))
