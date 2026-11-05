"""GPU detection: is RAPIDS (cuML, and the cuGraph/cuDF/CuPy clustering kit)
importable, what the device is called, and how much VRAM is free.

The probe runs once, when this module is first imported. Core module: no Tk,
and it does not import ``openflo.pipeline`` (pipeline imports it).
"""


# ── GPU probe ─────────────────────────────────────────────────────────────────

def probe_gpu():
    """Return (available, display_name, CumlUMAP_class, cluster_kit).

    `cluster_kit` is a dict {cunn, cugraph, cupy, cudf} if the RAPIDS pieces
    needed for GPU clustering are present, else None. cuML UMAP can work
    standalone, but clustering needs the full RAPIDS stack.
    """
    try:
        from cuml.manifold import UMAP as _CU  # type: ignore[import-not-found]
        name = 'GPU'
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            h   = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetName(h)
            name = raw.decode() if isinstance(raw, bytes) else raw
        except Exception:  # noqa: BLE001 - GPU name is informational
            pass

        cluster_kit = None
        try:
            import cudf as _cudf  # type: ignore[import-not-found]
            import cugraph as _cugraph  # type: ignore[import-not-found]
            import cupy as _cupy
            from cuml.neighbors import NearestNeighbors as _CuNN  # type: ignore[import-not-found]
            cluster_kit = {
                'cunn':    _CuNN,
                'cugraph': _cugraph,
                'cupy':    _cupy,
                'cudf':    _cudf,
            }
        except ImportError:
            pass

        return True, name, _CU, cluster_kit
    except ImportError:
        return False, '', None, None

GPU_AVAILABLE, GPU_NAME, CUML_UMAP, GPU_CLUSTER_KIT = probe_gpu()
GPU_CLUSTERING_AVAILABLE = GPU_CLUSTER_KIT is not None


def vram_free_gb():
    """Free VRAM in GB via pynvml → nvidia-smi fallback. None if no GPU."""
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).free / (1024 ** 3)
    except Exception:  # noqa: BLE001 - GPU memory probe is best-effort
        pass
    try:
        import subprocess
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=4,
            creationflags=flags)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001 - GPU memory probe is best-effort
        pass
    return None
