"""Bounded FCS loader: the pool caps concurrency and the progress counters stay
exact under contention.

These exercise the real `_ensure_load_pool` / `_load_pool_worker` /
`_mark_one_done` methods bound onto a lightweight fake (no Tk window). The fake
``after`` runs its callback inline so the Tk-thread tally path is exercised
without an event loop.
"""
import threading
import time

from openflo import gui
from openflo.gui import ViewGateEditorWindow as GE


class _FakeEditor:
    """Minimal stand-in carrying just the loader state + bound pool methods."""

    def __init__(self):
        self._load_queue = gui.queue.Queue()
        self._load_pool = []
        self._load_pool_started = False
        self._load_stop = threading.Event()
        self._load_total = 0
        self._load_done = 0
        # Concurrency probes.
        self._active = 0
        self._max_active = 0
        self._probe_lock = threading.Lock()
        self._loaded = []
        # `after` is invoked from worker threads; serialise the inline call so
        # the single-writer guarantee under test isn't itself the thing racing.
        self._after_lock = threading.Lock()
        # Bind the real implementations under test.
        self._ensure_load_pool = GE._ensure_load_pool.__get__(self)
        self._load_pool_worker = GE._load_pool_worker.__get__(self)
        self._mark_one_done = GE._mark_one_done.__get__(self)

    # ── stubs ────────────────────────────────────────────────────────────
    def after(self, _ms, fn=None):
        if fn is not None:
            with self._after_lock:
                fn()

    def _update_progress_bar(self):
        pass

    def _load_worker(self, name, path):
        with self._probe_lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
        time.sleep(0.01)            # simulate a real load so overlap can occur
        with self._probe_lock:
            self._active -= 1
            self._loaded.append(name)


def _drain(f, n, timeout=10.0):
    deadline = time.time() + timeout
    while f._load_done < n and time.time() < deadline:
        time.sleep(0.005)


def test_pool_caps_concurrency_and_counts_exactly():
    f = _FakeEditor()
    f._ensure_load_pool()
    assert len(f._load_pool) == gui._LOAD_POOL_SIZE

    n = 25
    for i in range(n):
        f._load_queue.put((f'sample{i}', f'/x/sample{i}.fcs'))
        f._load_total += 1

    _drain(f, n)

    # Every job ran exactly once, the counter is exact (no lost increments),
    # and concurrency never exceeded the cap.
    assert f._load_done == n
    assert len(f._loaded) == n
    assert sorted(f._loaded) == sorted(f'sample{i}' for i in range(n))
    assert 1 <= f._max_active <= gui._LOAD_POOL_SIZE

    # Lazy spawn is one-shot.
    f._ensure_load_pool()
    assert len(f._load_pool) == gui._LOAD_POOL_SIZE

    # Sentinels let blocked workers exit cleanly.
    f._load_stop.set()
    for _ in range(gui._LOAD_POOL_SIZE):
        f._load_queue.put(None)
    for t in f._load_pool:
        t.join(timeout=1.0)
        assert not t.is_alive()


def test_done_counter_survives_failing_loads():
    """A load that raises still advances the counter (finally → tick), so the
    bar can reach N/N and auto-hide instead of hanging."""
    f = _FakeEditor()

    def boom(name, path):
        raise RuntimeError('corrupt FCS')

    f._load_worker = boom
    f._ensure_load_pool()
    n = 8
    for i in range(n):
        f._load_queue.put((f'bad{i}', f'/x/bad{i}.fcs'))
        f._load_total += 1
    _drain(f, n)
    assert f._load_done == n

    f._load_stop.set()
    for _ in range(gui._LOAD_POOL_SIZE):
        f._load_queue.put(None)
