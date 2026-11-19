"""The exit autosave must not fail silently.

`_on_close` removed `self._log_queue` from every log tee and *then* wrote the
autosave, reporting failure with a `print()` — into a sink the in-app console
could no longer receive, moments before `destroy()`. Disk full at exit meant
the session was lost, the next launch had nothing to resume, and no message
ever reached the user. The sidecar-failure flag that `_save_session` warns
about was ignored on this path too, so the autosave silently degraded to
raw-FCS-only.
"""
from types import SimpleNamespace

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tests._tk_wait import pump_until
from tests.conftest import editor_or_skip

COLS = ('FSC-A', 'SSC-A', 'CD3')


def _editor_or_skip():
    root, ed = editor_or_skip()
    df = pd.DataFrame({c: np.linspace(0, 1, 20) for c in COLS})
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=r'C:\e\s1.fcs', data=df, fluor_channels=['CD3'],
        channel_labels={c: c for c in COLS})
    ed._sample_order.append('s1')
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_gates.setdefault('s1', {})
    return root, ed


def _close_and_wait(root, ed, timeout=20.0):
    """Close, then pump the Tk loop until the autosave worker has finished
    and the close completed (the save no longer runs on the Tk thread)."""
    closed = []
    ed.destroy = lambda: closed.append(True)      # keep the window alive
    ed._on_close()
    pump_until(root, lambda: bool(closed), timeout)
    assert closed, 'the close never finished'


def _capture_dialogs(ed, monkeypatch, close_anyway=True):
    seen = {'error': [], 'warning': []}
    import openflo.editor_lifecycle as L
    monkeypatch.setattr(L.messagebox, 'showerror',
                        lambda t, m, **k: seen['error'].append((t, m)))

    def ask(t, m, **k):
        seen['error'].append((t, m))
        return close_anyway
    monkeypatch.setattr(L.messagebox, 'askyesno', ask)
    monkeypatch.setattr(L.messagebox, 'showwarning',
                        lambda t, m, **k: seen['warning'].append((t, m)))
    # The close asks first; these tests are about what Save does.
    monkeypatch.setattr(type(ed), '_ask_close_choice', lambda self: 'save')
    return seen


def test_autosave_failure_raises_a_dialog(monkeypatch):
    """A failed autosave means the work is gone — say so, visibly."""
    root, ed = _editor_or_skip()
    try:
        seen = _capture_dialogs(ed, monkeypatch)
        monkeypatch.setattr(type(ed), '_run_session_write',
                            lambda self, plan, progress=None: (_ for _ in ())
                            .throw(OSError('No space left on device')))
        monkeypatch.setattr(type(ed), '_save_geometry', lambda self: None)
        _close_and_wait(root, ed)

        assert len(seen['error']) == 1, (
            'the autosave failed and the user was not told exactly once — the '
            'session is lost and the next launch has nothing to resume',
            seen['error'])
        (title, msg), = seen['error']
        assert title == 'Autosave failed — session not saved'
        assert 'No space left on device' in msg, 'the cause must be reported'
    finally:
        root.destroy()


def test_sidecar_failure_warns_on_exit(monkeypatch):
    """Same condition _save_session warns about, previously ignored here."""
    root, ed = _editor_or_skip()
    try:
        seen = _capture_dialogs(ed, monkeypatch)

        monkeypatch.setattr(type(ed), '_run_session_write',
                            lambda self, plan, progress=None: ({}, ['s1']))
        monkeypatch.setattr(type(ed), '_save_geometry', lambda self: None)
        _close_and_wait(root, ed)

        assert len(seen['warning']) == 1, (
            'a sample lost its computed results on exit, and was not told '
            'exactly once', seen['warning'])
        (title, msg), = seen['warning']
        assert title == 'Session autosaved WITH WARNINGS'
        assert 'could not be written for:\n\n  s1\n' in msg, msg
    finally:
        root.destroy()


def test_clean_autosave_is_quiet(monkeypatch):
    """No dialogs when nothing went wrong."""
    root, ed = _editor_or_skip()
    try:
        seen = _capture_dialogs(ed, monkeypatch)
        monkeypatch.setattr(type(ed), '_run_session_write',
                            lambda self, plan, progress=None: ({}, []))
        monkeypatch.setattr(type(ed), '_save_geometry', lambda self: None)
        _close_and_wait(root, ed)
        assert not seen['error'] and not seen['warning']
    finally:
        root.destroy()


def test_autosave_runs_before_the_log_sinks_are_detached():
    """Ordering is the point: a diagnostic emitted after the tees are removed
    reaches nothing."""
    import inspect

    from openflo import editor_lifecycle
    src = inspect.getsource(editor_lifecycle.LifecycleMixin._finish_close)
    save_at = src.find('autosave failed')
    tee_at = src.find('remove_sink')
    assert save_at != -1 and tee_at != -1, 'test is stale — block moved'
    assert save_at < tee_at, (
        'the autosave runs after the log sinks are detached again, so its '
        'failure message goes nowhere the user can see')


def test_close_asks_save_discard_or_cancel(monkeypatch):
    """Saving a large session takes a while, so ✕ asks: Save and close
    autosaves, Close without saving doesn't, Cancel keeps the window; a
    second ✕ while closing is ignored."""
    root, ed = _editor_or_skip()
    try:
        _capture_dialogs(ed, monkeypatch)
        writes, closed = [], []
        monkeypatch.setattr(type(ed), '_run_session_write',
                            lambda self, plan, progress=None:
                            (writes.append(plan['path']), ({}, []))[1])
        monkeypatch.setattr(type(ed), '_save_geometry', lambda self: None)
        ed.destroy = lambda: closed.append(1)
        answer = {'v': 'cancel'}
        monkeypatch.setattr(type(ed), '_ask_close_choice',
                            lambda self: answer['v'])
        ed._on_close()
        assert writes == [] and closed == []          # Cancel: still open
        answer['v'] = 'discard'
        ed._on_close()
        assert writes == [] and closed == [1]         # closed, nothing saved
        ed._on_close()                                # a second ✕: ignored
        assert closed == [1]
        ed._closing = False
        answer['v'] = 'save'
        ed._on_close()
        pump_until(root, lambda: len(closed) >= 2, 10)   # the save is threaded
        assert len(writes) == 1 and closed == [1, 1]
    finally:
        root.destroy()


def test_the_close_question_has_three_plain_choices():
    import tkinter as tk
    from tkinter import ttk
    root, ed = _editor_or_skip()
    try:
        seen = {}

        def click():
            dlg = next(w for w in ed.winfo_children()
                       if isinstance(w, tk.Toplevel)
                       and w.title() == 'Close OpenFlo')

            def buttons(w):
                for c in w.winfo_children():
                    if isinstance(c, ttk.Button):
                        yield c
                    yield from buttons(c)
            by = {str(b.cget('text')): b for b in buttons(dlg)}
            seen['labels'] = sorted(by)
            by['Close without saving'].invoke()
        root.after(300, click)
        assert ed._ask_close_choice() == 'discard'
        assert seen['labels'] == ['Cancel', 'Close without saving',
                                  'Save and close']
    finally:
        root.destroy()


def _failing_save(monkeypatch, ed):
    monkeypatch.setattr(type(ed), '_run_session_write',
                        lambda self, plan, progress=None: (_ for _ in ())
                        .throw(OSError('No space left on device')))
    monkeypatch.setattr(type(ed), '_save_geometry', lambda self: None)


def test_cancelling_after_a_failed_save_keeps_the_window_open(monkeypatch):
    """The failed-save dialog tells the user to cancel the close. Answering
    No must keep OpenFlo open and usable: not destroyed, not stuck
    "closing", and a later close asks again."""
    root, ed = _editor_or_skip()
    try:
        seen = _capture_dialogs(ed, monkeypatch, close_anyway=False)
        _failing_save(monkeypatch, ed)
        closed = []
        ed.destroy = lambda: closed.append(True)
        lock_released = []
        monkeypatch.setattr(type(ed), '_release_session_lock',
                            lambda self: lock_released.append(1))
        ed._on_close()
        pump_until(root, lambda: bool(seen['error']), 20)
        pump_until(root, lambda: getattr(ed, '_close_thread', None) is None
                   or not ed._close_thread.is_alive(), 5)
        root.update()
        assert len(seen['error']) == 1, seen['error']
        title, msg = seen['error'][0]
        assert title == 'Autosave failed — session not saved'
        assert 'No space left on device' in msg
        assert 'Close anyway?' in msg
        assert closed == [], 'answered No, yet the editor was closed'
        assert lock_released == [], 'the session lock was dropped while open'
        assert ed._closing is False, 'the editor is stuck mid-close'
        assert ed._close_progress is None
        assert not ed._load_stop.is_set(), 'sample loads stay stopped'
        assert ed._plotcache_closed is False
        assert 's1' in ed._samples
        # A second ✕ asks again and, answered Yes, closes.
        seen['error'].clear()
        monkeypatch.setattr(
            __import__('openflo.editor_lifecycle').editor_lifecycle
            .messagebox, 'askyesno', lambda t, m, **k: True)
        ed._on_close()
        pump_until(root, lambda: bool(closed), 20)
        assert closed == [True]
    finally:
        root.destroy()


def test_closing_anyway_after_a_failed_save_closes(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        seen = _capture_dialogs(ed, monkeypatch, close_anyway=True)
        _failing_save(monkeypatch, ed)
        _close_and_wait(root, ed)
        assert len(seen['error']) == 1
    finally:
        root.destroy()


def test_queued_loads_survive_a_cancelled_close(monkeypatch):
    """A sample load still waiting when the close started runs once the
    close is cancelled: the shutdown sentinels are dropped and the loader
    pool is back to its size."""
    import threading
    root, ed = _editor_or_skip()
    try:
        _capture_dialogs(ed, monkeypatch, close_anyway=False)
        _failing_save(monkeypatch, ed)
        ed.destroy = lambda: None
        loaded, gate, started = [], threading.Event(), threading.Event()

        def fake_load(self, name, path):
            started.set()
            gate.wait(10)
            loaded.append(name)
        monkeypatch.setattr(type(ed), '_load_worker', fake_load)
        monkeypatch.setattr(type(ed), '_pool_size', lambda self: 1)
        ed._ensure_load_pool()
        ed._enqueue_load(('busy', 'busy.fcs'))     # holds the one worker
        assert started.wait(10), 'the loader never took the first job'
        ed._enqueue_load(('queued', 'queued.fcs'))
        ed._on_close()
        gate.set()                                  # finishes after the stop
        pump_until(root, lambda: not ed._closing, 20)
        pump_until(root, lambda: 'queued' in loaded, 10)
        assert loaded == ['busy', 'queued'], loaded
        alive = [t for t in ed._load_pool if t.is_alive()]
        assert len(alive) == 1, ed._load_pool  # one live pool of the set size
        ed._load_stop.set()
        ed._enqueue_load(None, priority=-1)
    finally:
        root.destroy()


def test_a_stopped_or_replaced_loader_hands_its_job_back(monkeypatch):
    """A pool worker that wakes to a job while the editor is closing, or
    after its pool was replaced, puts the job back instead of dropping it,
    so a cancelled close still loads that sample."""
    import threading
    root, ed = _editor_or_skip()
    try:
        ran = []
        monkeypatch.setattr(type(ed), '_load_worker',
                            lambda self, n, p: ran.append(n))
        ed.after = lambda *a: None   # no Tk loop runs on the worker's side

        def left():
            return [j for _p, _s, j in ed._load_queue.queue]

        # Closing: the job is put back and no load runs.
        ed._load_stop.set()
        ed._enqueue_load(('a', 'a.fcs'))
        t = threading.Thread(target=ed._load_pool_worker, daemon=True)
        t.start()
        t.join(5)
        assert not t.is_alive() and ran == [], ran
        assert left() == [('a', 'a.fcs')], left()
        ed._load_queue.get_nowait()

        # Replaced pool: a worker of the old generation hands the job back.
        ed._load_stop.clear()
        t = threading.Thread(target=ed._load_pool_worker, daemon=True)
        t.start()
        ed._enqueue_load(('b', 'b.fcs'))
        pump_until(root, lambda: ran == ['b'], 5)
        assert ran == ['b'], 'a live worker of the current pool must load'
        ed._load_gen += 1
        ed._enqueue_load(('c', 'c.fcs'))
        t.join(5)
        assert not t.is_alive(), 'a worker of a replaced pool kept running'
        assert ran == ['b'] and left() == [('c', 'c.fcs')], (ran, left())
        ed._load_stop.set()
    finally:
        root.destroy()


def test_the_failed_save_question_defaults_to_keeping_the_window(monkeypatch):
    """Enter on the failed-save question must not lose the session: No is
    the default button, and the question is an error, not a plain query."""
    root, ed = _editor_or_skip()
    try:
        import openflo.editor_lifecycle as L
        seen = []
        monkeypatch.setattr(L.messagebox, 'askyesno',
                            lambda t, m, **k: (seen.append(k), False)[1])
        assert ed._close_anyway_after_failed_save(OSError('disk full')) is False
        (kw,) = seen
        assert kw.get('default') == 'no', kw
        assert kw.get('icon') == 'error', kw
    finally:
        root.destroy()


def test_a_question_that_cannot_be_shown_keeps_the_window(monkeypatch):
    """If the failed-save question cannot be shown, the close is cancelled
    rather than losing the session without asking."""
    import tkinter as tk
    root, ed = _editor_or_skip()
    try:
        import openflo.editor_lifecycle as L

        def broken(*_a, **_k):
            raise tk.TclError('cannot show the dialog')
        monkeypatch.setattr(L.messagebox, 'askyesno', broken)
        assert ed._close_anyway_after_failed_save(OSError('x')) is False
        monkeypatch.setattr(type(ed), '_ask_close_choice', lambda self: 'save')
        _failing_save(monkeypatch, ed)
        closed = []
        ed.destroy = lambda: closed.append(True)
        ed._on_close()
        pump_until(root, lambda: not ed._closing, 20)
        assert closed == [] and ed._closing is False
    finally:
        root.destroy()


def test_a_busy_old_worker_leaves_after_a_cancelled_close(monkeypatch):
    """A loader still busy when the close is cancelled (its shutdown
    sentinel drained) belongs to the replaced pool: after its job it hands
    the next one back and leaves, so only the new pool loads."""
    import threading
    root, ed = _editor_or_skip()
    try:
        loaded = []
        gates = {n: threading.Event() for n in ('busy', 'hold')}
        started = {n: threading.Event() for n in ('busy', 'hold')}

        def fake_load(self, name, path):
            if name in gates:
                started[name].set()
                gates[name].wait(10)
            loaded.append((name, threading.current_thread()))
        monkeypatch.setattr(type(ed), '_load_worker', fake_load)
        monkeypatch.setattr(type(ed), '_pool_size', lambda self: 1)
        ed.after = lambda *a: None   # no Tk loop runs on the workers' side
        ed._ensure_load_pool()
        old = ed._load_pool[0]
        ed._enqueue_load(('busy', 'busy.fcs'))
        assert started['busy'].wait(10)
        # The close's teardown, then the user cancels it.
        ed._closing = True
        ed._load_stop.set()
        ed._enqueue_load(None, priority=-1)
        ed._reopen_after_failed_close()
        (new,) = ed._load_pool
        assert new is not old
        ed._enqueue_load(('hold', 'hold.fcs'))     # occupies the new worker
        assert started['hold'].wait(10)
        ed._enqueue_load(('queued', 'queued.fcs'))
        gates['busy'].set()          # the old worker is the only one free
        old.join(5)
        assert not old.is_alive(), 'the replaced pool\'s worker kept serving'
        gates['hold'].set()
        pump_until(root, lambda: len(loaded) == 3, 5)
        by = dict(loaded)
        assert by == {'busy': old, 'hold': new, 'queued': new}, loaded
        assert new.is_alive()
        ed._load_stop.set()
        ed._enqueue_load(None, priority=-1)
    finally:
        root.destroy()
