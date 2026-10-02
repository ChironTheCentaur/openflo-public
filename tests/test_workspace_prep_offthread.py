"""Staging a run unit must not freeze the UI.

`_launch_next` used to build the unit AND serialise it to `job.pkl` on the Tk
thread. The write dominates: measured on this machine, 569 ms for a 20-sample x
200k-event run (336 MB) and 1264 ms for 8 x 1M (672 MB) — all of it blocking
the window, with the code commented "cheap (main)".

Only the WRITE moved. `prepare_unit` deliberately stays on the Tk thread: it
reads ``editor._samples``, which the user can mutate, and running it
concurrently would introduce the same read-during-write race that was fixed in
the plot path. `prep` is freshly built and unshared, so pickling it in a worker
shares nothing.

That adds a third driver state — "serialising": nothing is running yet, but the
queue must not advance and the run is not finished. These tests drive that
state machine directly.
"""
import threading
import time
import types

import pytest

from openflo import workspace as W


class _Runner:
    """The smallest object that exercises the driver's state handling."""

    def __init__(self):
        self._cur = None
        self._prep = None
        self._jobs = []
        self._err = 0
        self._total = 1
        self._status = []
        self.status_var = types.SimpleNamespace(set=self._status.append,
                                                get=lambda: self._status[-1]
                                                if self._status else '')

    _finish_prep = W.WorkspacePanel._finish_prep
    _kill_current = W.WorkspacePanel._kill_current


def _staged(tmp_path, *, alive, error=None):
    """A `_prep` record whose writer thread is alive or already done."""
    gate = threading.Event()
    jobdir = tmp_path / 'wsjob'
    jobdir.mkdir(exist_ok=True)

    def _worker():
        gate.wait(timeout=5)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    if not alive:
        gate.set()
        t.join(timeout=5)
    return gate, {
        'thread': t, 'state': {'error': error},
        'job': {'label': 'unit-1', 'cfg': {}}, 'n': 1,
        'jobdir': str(jobdir),
        'job_path': str(jobdir / 'job.pkl'),
        'result_path': str(jobdir / 'result.pkl'),
    }


def test_a_unit_still_serialising_does_not_advance_the_queue(tmp_path):
    r = _Runner()
    gate, r._prep = _staged(tmp_path, alive=True)
    try:
        assert r._finish_prep() is False, (
            'the driver treated an in-flight write as finished — it would '
            'launch the next unit while this one is still being staged')
        assert r._prep is not None and r._cur is None
    finally:
        gate.set()


def test_a_write_failure_is_reported_and_does_not_launch(tmp_path):
    """The write used to be inside a try on the Tk thread; moving it must not
    lose the error."""
    r = _Runner()
    _gate, r._prep = _staged(tmp_path, alive=False,
                             error=OSError('No space left on device'))
    assert r._finish_prep() is True
    assert r._cur is None, 'launched a child for a unit that was never written'
    assert r._prep is None
    assert r._err == 1
    assert 'No space left on device' in r.status_var.get()
    assert not (tmp_path / 'wsjob').exists(), 'the temp dir was leaked'


def test_cancelling_mid_write_reaps_the_temp_dir(tmp_path):
    """The writer is a daemon that cannot be interrupted safely, so cancel must
    let it finish and then clean up rather than leaking or racing it."""
    r = _Runner()
    gate, r._prep = _staged(tmp_path, alive=True)
    jobdir = tmp_path / 'wsjob'
    (jobdir / 'job.pkl').write_bytes(b'partial')

    r._kill_current()
    assert r._prep is None, 'cancel left the staging record in place'

    gate.set()                                   # writer finishes
    for _ in range(100):                         # reaper joins then removes
        if not jobdir.exists():
            break
        time.sleep(0.05)
    assert not jobdir.exists(), 'cancel during staging leaked the temp dir'


def test_cancel_with_nothing_staged_is_a_noop(tmp_path):
    r = _Runner()
    r._kill_current()
    assert r._prep is None and r._cur is None


@pytest.mark.parametrize('attr', ['_prep', '_cur'])
def test_the_driver_consults_both_states(attr):
    """`_cur` (running) and `_prep` (staging) must stay distinct: collapsing
    them is exactly what would let the queue advance mid-write."""
    import inspect
    body = inspect.getsource(W.WorkspacePanel._pump_jobs)
    assert attr in body, (
        f'_pump_jobs no longer consults {attr}; the staging state has been '
        'collapsed back into the running state')


def test_launch_next_returns_while_the_write_is_still_in_flight(tmp_path,
                                                                monkeypatch):
    """The property the rest of this file does not actually pin.

    Every other test here builds a `_prep` record by hand, so they exercise the
    state machine but would still pass if the write were moved back onto the Tk
    thread. This drives the REAL `_launch_next` with a write that blocks, and
    requires the call to return anyway — which is the whole point of the
    change, and the thing a future refactor could silently undo.
    """
    import collections
    import pickle

    writing = threading.Event()
    release = threading.Event()

    def blocking_dump(obj, fh, *a, **k):
        writing.set()
        assert release.wait(timeout=10), 'the writer was never released'
        pickle.dumps(obj)          # still do the work, just later

    # `_launch_next` imports pickle locally, so patch the module itself.
    monkeypatch.setattr(pickle, 'dump', blocking_dump)
    monkeypatch.setattr(W, 'prepare_unit',
                        lambda editor, members, label, cfg: {
                            'data': None, 'channels': [], 'label': label,
                            'note': '', 'n_events': 7, 'color_by': None})

    runner = _Runner()
    runner._editor = object()
    runner._out_dir = str(tmp_path)
    runner._idx = 0
    runner._jobs = collections.deque([
        {'unit': {'members': [], 'label': 'u'}, 'cfg': {}, 'label': 'u'}])

    started = time.perf_counter()
    try:
        W.WorkspacePanel._launch_next(runner)
        elapsed = time.perf_counter() - started

        assert writing.wait(timeout=10), 'the write never began'
        assert elapsed < 5.0, (
            f'_launch_next blocked for {elapsed:.1f}s — the write is back on '
            'the calling thread')
        assert runner._prep is not None, 'no staging record was left behind'
        assert runner._prep['thread'].is_alive(), (
            'the write had already finished, so it did not happen on a worker')
        assert runner._cur is None, (
            'a child process was launched before the job file was written')
    finally:
        release.set()
        pr = runner._prep
        if pr is not None:
            pr['thread'].join(timeout=10)


def test_the_staged_unit_is_reported_to_the_user(tmp_path, monkeypatch):
    """Staging is not instantaneous, so it must not look like nothing is
    happening."""
    import collections

    monkeypatch.setattr(W, 'prepare_unit',
                        lambda editor, members, label, cfg: {
                            'data': None, 'channels': [], 'label': label,
                            'note': '', 'n_events': 1234, 'color_by': None})
    runner = _Runner()
    runner._editor = object()
    runner._out_dir = str(tmp_path)
    runner._idx = 0
    runner._jobs = collections.deque([
        {'unit': {'members': [], 'label': 'u'}, 'cfg': {}, 'label': 'u'}])

    W.WorkspacePanel._launch_next(runner)
    try:
        assert '1,234' in runner.status_var.get()
    finally:
        pr = runner._prep
        if pr is not None:
            pr['thread'].join(timeout=10)
