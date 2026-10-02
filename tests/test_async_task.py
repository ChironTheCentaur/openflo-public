"""Tests for the shared off-thread runner (openflo.async_task.run_async)."""
import threading
import time

from openflo.async_task import run_async


class _FakeWidget:
    """Captures after() callbacks; run_pending() drains them, simulating the
    Tk event loop processing what the worker thread posted back."""

    def __init__(self):
        self._pending = []
        self._lock = threading.Lock()

    def after(self, _ms, fn):
        with self._lock:
            self._pending.append(fn)

    def run_pending(self, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                fns, self._pending = self._pending, []
            for fn in fns:
                fn()
            if fns:
                return
            time.sleep(0.005)


def test_run_async_delivers_result():
    w = _FakeWidget()
    got = {}
    run_async(w, lambda: 6 * 7, on_done=lambda r: got.__setitem__('r', r))
    w.run_pending()
    assert got['r'] == 42


def test_run_async_routes_exception_to_on_error():
    w = _FakeWidget()
    seen = {}

    def boom():
        raise ValueError('nope')

    run_async(w, boom, on_done=lambda r: seen.__setitem__('done', True),
              on_error=lambda e: seen.__setitem__('err', str(e)))
    w.run_pending()
    assert seen.get('err') == 'nope' and 'done' not in seen


def test_run_async_on_finally_runs_on_both_paths():
    w = _FakeWidget()
    calls = []
    run_async(w, lambda: 1, on_done=lambda r: calls.append('done'),
              on_finally=lambda: calls.append('fin'))
    w.run_pending()
    assert calls == ['done', 'fin']

    w2 = _FakeWidget()
    calls2 = []

    def boom():
        raise RuntimeError('x')

    run_async(w2, boom, on_error=lambda e: calls2.append('err'),
              on_finally=lambda: calls2.append('fin'))
    w2.run_pending()
    assert calls2 == ['err', 'fin']
