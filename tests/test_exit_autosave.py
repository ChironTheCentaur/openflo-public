"""The exit autosave must not fail silently.

`_on_close` removed `self._log_queue` from every log tee and *then* wrote the
autosave, reporting failure with a `print()` — into a sink the in-app console
could no longer receive, moments before `destroy()`. Disk full at exit meant
the session was lost, the next launch had nothing to resume, and no message
ever reached the user. The sidecar-failure flag that `_save_session` warns
about was ignored on this path too, so the autosave silently degraded to
raw-FCS-only.
"""
import os
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tests.conftest import gui_unavailable

COLS = ('FSC-A', 'SSC-A', 'CD3')


def _editor_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                        # noqa: BLE001
        pytest.skip(str(e))
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
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
    import time
    closed = []
    ed.destroy = lambda: closed.append(True)      # keep the window alive
    ed._on_close()
    end = time.monotonic() + timeout
    while not closed and time.monotonic() < end:
        root.update()
        time.sleep(0.01)
    assert closed, 'the close never finished'


def _capture_dialogs(ed, monkeypatch):
    seen = {'error': [], 'warning': []}
    import openflo.editor_lifecycle as L
    monkeypatch.setattr(L.messagebox, 'showerror',
                        lambda t, m, **k: seen['error'].append((t, m)))
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

        assert seen['error'], (
            'the autosave failed and the user was never told — the session is '
            'lost and the next launch has nothing to resume')
        title, msg = seen['error'][0]
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

        assert seen['warning'], (
            'a sample lost its computed results on exit, silently')
        assert 's1' in seen['warning'][0][1]
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
        import time
        end = time.monotonic() + 10                   # the save is threaded
        while len(closed) < 2 and time.monotonic() < end:
            root.update()
            time.sleep(0.01)
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
