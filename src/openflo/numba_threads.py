"""Keep two threads from running numba ``parallel=True`` code at once.

UMAP (and pynndescent under it), TriMap and PaCMAP run numba ``parallel=True``
kernels, and OpenFlo calls them from more than one thread of one process: the
embedding warm-up (openflo/warmup.py) compiles UMAP in a background thread
while the real run, or a second embedding job, may already be embedding.

Whether that is safe depends on numba's threading layer, which numba picks the
first time a parallel kernel runs: TBB, then OpenMP, then ``workqueue``. TBB
and OpenMP accept parallel launches from several threads. ``workqueue`` does
not — on a second concurrent launch it prints "Concurrent access has been
detected" and calls ``abort()``, killing the whole app with no Python
exception to catch. Linux (GNU OpenMP) and Windows get OpenMP; macOS has no
OpenMP numba can load and no TBB unless installed, so a Mac gets
``workqueue`` and was the one platform where this crashed.

So every call into those libraries goes through ``parallel_kernels()`` (or a
function decorated with ``serialized``). When the layer is thread-safe that is
a no-op; when it is ``workqueue`` it holds one process-wide lock, so the calls
take turns instead of aborting. Only the order changes, never an input, so
results are the same. tests/test_numba_threads.py runs real embeddings from
two threads under ``workqueue`` and checks every entry point is wrapped.
"""
from __future__ import annotations

import contextlib
import functools
import threading

# Re-entrant: a wrapped function may call another wrapped one.
_LOCK = threading.RLock()
_decide = threading.Lock()
_safe: bool | None = None       # None until the layer has been looked at


def _layer_is_threadsafe() -> bool:
    """Is numba's threading layer one that takes concurrent launches?

    Starts numba's thread pool to find out: the layer is chosen there and
    never changes after, and the kernel about to run would start it anyway.
    Anything unexpected answers False, which costs only the lock."""
    try:
        import numba
        from numba.np.ufunc import parallel
        parallel._launch_threads()
        return numba.threading_layer() in ('tbb', 'omp')
    except Exception:                    # noqa: BLE001 - unsure: serialize
        return False


def _needs_lock() -> bool:
    global _safe
    if _safe is None:
        with _decide:
            if _safe is None:
                _safe = _layer_is_threadsafe()
    return not _safe


def parallel_kernels():
    """Context manager around a call that may run numba parallel kernels.

    Holds the process-wide lock when the layer is ``workqueue``; does nothing
    otherwise."""
    return _LOCK if _needs_lock() else contextlib.nullcontext()


def serialized(fn):
    """Decorator form of ``parallel_kernels()``, for a function of OpenFlo's
    own that is or calls a ``parallel=True`` kernel."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with parallel_kernels():
            return fn(*args, **kwargs)
    return wrapper
