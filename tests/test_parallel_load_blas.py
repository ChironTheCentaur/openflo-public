"""Parallel sample loads compensate one at a time, and match a serial load.

The editor loads samples on a pool of threads, and every load compensates on
the CPU through OpenBLAS. Two threads inside a threaded OpenBLAS call at once
hung, crashed, or returned counts a few dozen events off on a Windows desktop
(6 of 16 session reopens), while OPENBLAS_NUM_THREADS=1 made every reopen
match. The fix holds one lock around the CPU BLAS calls on the load path.

The OpenBLAS fault itself depends on the build and the CPU, so these tests
watch the BLAS call directly: they slow it down, run the real pool function
(`LoadPoolMixin._prepared_fcs_sample`) on several threads, and count how many
threads are inside it at once. With the lock it is never more than one; with
the lock replaced by a no-op the same probe sees several (the negative check).
Every load must also equal a serial load of the same file, bit for bit.
"""
from __future__ import annotations

import contextlib
import threading
import time
import types

import numpy as np
import pytest

from openflo import gpu_accel

CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'Time']
SPILL = np.array([[1.0, 0.12, 0.03],
                  [0.08, 1.0, 0.20],
                  [0.01, 0.05, 1.0]])
N_FILES = 6
N_EVENTS = 4000


def _fcs(path, seed):
    """A synthetic FCS with three spilled fluorescence channels and $SPILL."""
    import flowio
    rng = np.random.default_rng(seed)
    true = rng.uniform(100, 60_000, (N_EVENTS, 3))
    ev = np.column_stack([rng.uniform(5e4, 1e5, N_EVENTS),
                          rng.uniform(1e4, 5e4, N_EVENTS),
                          true @ SPILL,
                          np.arange(N_EVENTS) / 10.0]).astype(np.float32)
    meta = {'SPILL': '3,FL1-A,FL2-A,FL3-A,'
                     + ','.join(map(str, SPILL.ravel()))}
    with open(path, 'wb') as f:
        flowio.create_fcs(f, ev.ravel().tolist(), CHANNELS, metadata_dict=meta)
    return str(path)


@pytest.fixture
def fcs_files(tmp_path, monkeypatch):
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, 'load_qc_enabled', lambda: False)
    monkeypatch.setattr(gpu_accel, '_enabled', False)
    return [_fcs(tmp_path / f'sample_{i}.fcs', seed=i) for i in range(N_FILES)]


def _load(path):
    """The pool's own per-file preparation (QC, compensation, transform)."""
    from openflo.editor_loadpool import LoadPoolMixin
    editor = types.SimpleNamespace(labels_str='')
    return LoadPoolMixin._prepared_fcs_sample(editor, path)


def _watch_blas(monkeypatch):
    """Slow the CPU matmul down and record the most threads inside it at once."""
    state = {'inside': 0, 'peak': 0, 'calls': 0}
    guard = threading.Lock()
    real = gpu_accel._cpu_matmul

    def slow_matmul(values, inv_t):
        with guard:
            state['inside'] += 1
            state['calls'] += 1
            state['peak'] = max(state['peak'], state['inside'])
        try:
            time.sleep(0.05)
            return real(values, inv_t)
        finally:
            with guard:
                state['inside'] -= 1

    monkeypatch.setattr(gpu_accel, '_cpu_matmul', slow_matmul)
    return state


def _load_in_parallel(paths):
    out: dict = {}
    start = threading.Barrier(len(paths))

    def work(p):
        start.wait()
        out[p] = _load(p)

    threads = [threading.Thread(target=work, args=(p,)) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert all(not t.is_alive() for t in threads), 'a parallel load hung'
    return out


def test_parallel_loads_compensate_one_at_a_time(fcs_files, monkeypatch):
    serial = {p: _load(p).data.copy() for p in fcs_files}
    state = _watch_blas(monkeypatch)
    parallel = _load_in_parallel(fcs_files)
    assert state['calls'] == N_FILES
    assert state['peak'] == 1, (
        f"{state['peak']} loader threads were inside BLAS at once")
    for p in fcs_files:
        got = parallel[p]
        assert got.comp_channels == ['FL1-A', 'FL2-A', 'FL3-A']
        assert got.data.equals(serial[p]), f'{p}: differs from a serial load'


def test_probe_sees_overlap_without_the_lock(fcs_files, monkeypatch):
    """Negative check: with the lock made a no-op, the same probe sees several
    threads inside BLAS at once, so the test above can fail."""
    monkeypatch.setattr(gpu_accel, '_CPU_BLAS_LOCK', contextlib.nullcontext())
    state = _watch_blas(monkeypatch)
    _load_in_parallel(fcs_files)
    assert state['peak'] > 1
