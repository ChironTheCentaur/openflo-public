"""The editor's update check, update run and diagnostics, and the Pipeline
Workspace's Batch folder, start their work through async_task.run_async
(survey finding Q7), so they get its two guards:

- a window closed while the work ran gets no callback (its widgets are
  gone), and
- an error in the work reaches the window's error handler, or the handled
  log for the silent startup check, instead of dying with its thread.

Tk-free: each stand-in has `after`, `winfo_exists` and a `_root()` whose
`report_callback_exception` records what reached it.
"""
from __future__ import annotations

import subprocess
import threading
import types

import pytest

from openflo import errors, update, workspace_run
from openflo import workspace as W
from openflo.editor_help import HelpMixin
from openflo.editor_update import UpdateMixin


class _Window:
    """`after` queues the callback; `pump` waits for one and runs it."""

    def __init__(self, alive=True):
        self.alive = alive
        self.queued = []
        self.posted = threading.Event()
        self.status = []
        self.status_var = types.SimpleNamespace(set=self.status.append)
        self.reported = []
        self.shown = []

    def after(self, _ms, fn):
        self.queued.append(fn)
        self.posted.set()

    def winfo_exists(self):
        return self.alive

    def _root(self):
        return types.SimpleNamespace(
            report_callback_exception=lambda *a: self.reported.append(a[1]))

    def pump(self, until, timeout=30.0):
        """Run queued callbacks until `until()` holds or the work is done
        and nothing more was posted."""
        import time
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            while self.queued:
                self.queued.pop(0)()
            if until():
                return
            self.posted.wait(0.05)
            self.posted.clear()


class _Ed(_Window):
    _check_for_updates = UpdateMixin._check_for_updates
    _run_update = UpdateMixin._run_update
    _run_diagnostics = HelpMixin._run_diagnostics

    def _on_update_checked(self, res, silent):
        self.shown.append(('checked', res, silent))

    def _on_update_done(self, ok, log):
        self.shown.append(('done', ok, log))

    def _show_diagnostics_result(self, text, rc):
        self.shown.append(('diagnostics', text, rc))


def _settle(ed, threads_before):
    """Wait for every worker the call started, then run what it posted."""
    for t in threading.enumerate():
        if t not in threads_before and t.daemon:
            t.join(30)
    while ed.queued:
        ed.queued.pop(0)()


def _boom(*_a, **_k):
    raise RuntimeError("synthetic failure")


# ── errors reach someone ────────────────────────────────────────────────────

def test_an_update_check_error_asked_for_reaches_the_error_handler(monkeypatch):
    monkeypatch.setattr(update, 'check_for_update', _boom)
    ed = _Ed()
    ed._check_for_updates(silent=False)
    ed.pump(lambda: ed.reported)
    assert [str(e) for e in ed.reported] == ["synthetic failure"]
    assert ed.shown == []


def test_a_startup_update_check_error_is_logged_not_shown(monkeypatch):
    monkeypatch.setattr(update, 'check_for_update', _boom)
    seen = len(errors.noted())
    ed = _Ed()
    ed._check_for_updates(silent=True)
    ed.pump(lambda: len(errors.noted()) > seen)
    new = errors.noted()[seen:]
    assert any('check for updates at startup' in line
               and 'synthetic failure' in line for line in new), new
    assert ed.reported == [] and ed.shown == []


def test_an_update_run_that_raises_reaches_the_error_handler(monkeypatch):
    monkeypatch.setattr(update, 'run_update', _boom)
    ed = _Ed()
    ed._run_update('pip')
    ed.pump(lambda: ed.reported)
    assert [str(e) for e in ed.reported] == ["synthetic failure"]


# ── results still arrive, and only to a live window ─────────────────────────

def _fake_diagnostics(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: types.SimpleNamespace(
        stdout="OpenFlo diagnostics\nHEALTHY\n", stderr='', returncode=0))


@pytest.mark.parametrize('alive', [True, False])
def test_each_editor_job_reports_back_only_to_a_live_window(monkeypatch, alive):
    res = {'current': '1.0.0', 'latest': '1.0.0', 'available': False, 'url': ''}
    monkeypatch.setattr(update, 'check_for_update', lambda: res)
    monkeypatch.setattr(update, 'run_update',
                        lambda kind=None: (True, f"synthetic {kind} log"))
    _fake_diagnostics(monkeypatch)
    for start, want in (
            (lambda ed: ed._check_for_updates(silent=True),
             ('checked', res, True)),
            (lambda ed: ed._run_update('pip'), ('done', True, 'synthetic pip log')),
            (lambda ed: ed._run_diagnostics(),
             ('diagnostics', "OpenFlo diagnostics\nHEALTHY\n", 0))):
        ed = _Ed(alive=alive)
        before = set(threading.enumerate())
        start(ed)
        _settle(ed, before)
        assert ed.shown == ([want] if alive else []), want
        assert ed.reported == []


class _View(_Window):
    def __init__(self, alive):
        super().__init__(alive)
        self.model = types.SimpleNamespace(run_cfg=workspace_run.default_run_cfg())
        self._editor = None

    def _sync_run_cfg(self):
        pass

    def _status(self, msg):
        self.status.append(msg)

    _batch_folder = W.PipelineWorkspaceView._batch_folder


@pytest.mark.parametrize('alive', [True, False])
def test_batch_folder_reports_back_only_to_a_live_window(tmp_path, monkeypatch,
                                                         alive):
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from openflo import pipeline
    indir, outdir = tmp_path / 'in', tmp_path / 'out'
    indir.mkdir()
    outdir.mkdir()
    (indir / 's1.fcs').write_bytes(b'synthetic')
    answers = [str(indir), str(outdir)]
    monkeypatch.setattr(fd, 'askdirectory', lambda **k: answers.pop(0))
    monkeypatch.setattr(mb, 'askyesno', lambda *a, **k: True)
    monkeypatch.setattr(pipeline, 'FlowSample', _boom)   # fails at once
    viewers = []
    monkeypatch.setattr(W, 'ResultsViewer', lambda *a, **k: viewers.append(a))
    v = _View(alive)
    before = set(threading.enumerate())
    v._batch_folder()
    _settle(v, before)
    done = [m for m in v.status if m.startswith('Batch done')]
    if alive:
        assert done and done[0].startswith('Batch done: 0 ok, 1 failed'), v.status
        assert len(viewers) == 1
    else:
        assert done == [] and viewers == []
    assert v.reported == []


class _BusyEd(_Window):
    def __init__(self):
        super().__init__()
        self.busy = []

    def _begin_busy(self, msg=None):
        self.busy.append('begin')

    def _busy(self, msg):
        pass

    def _end_busy(self):
        self.busy.append('end')


@pytest.mark.parametrize('fail', ['file', 'whole job'])
def test_batch_folder_stops_the_busy_bar_when_its_view_was_rebuilt(
        tmp_path, monkeypatch, fail):
    """Load or Undo destroys the workspace view mid-batch; the editor still
    hears back, its busy bar stops, and the done status is still posted.
    An error outside the per-file guard also stops the bar."""
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from openflo import pipeline
    indir, outdir = tmp_path / 'in', tmp_path / 'out'
    indir.mkdir()
    outdir.mkdir()
    (indir / 's1.fcs').write_bytes(b'synthetic')
    answers = [str(indir), str(outdir)]
    monkeypatch.setattr(fd, 'askdirectory', lambda **k: answers.pop(0))
    monkeypatch.setattr(mb, 'askyesno', lambda *a, **k: True)
    if fail == 'file':
        monkeypatch.setattr(pipeline, 'FlowSample', _boom)
    else:                       # the job's own import fails, before any file
        monkeypatch.delattr(pipeline, 'FlowSample')
    viewers = []
    monkeypatch.setattr(W, 'ResultsViewer', lambda *a, **k: viewers.append(a))
    v = _View(alive=False)
    ed = v._editor = _BusyEd()
    before = set(threading.enumerate())
    v._batch_folder()
    _settle(ed, before)
    assert ed.busy == ['begin', 'end']
    if fail == 'file':
        assert [m for m in v.status if m.startswith('Batch done: 0 ok, 1 failed')]
        assert ed.reported == []
    else:
        assert [type(e) for e in ed.reported] == [ImportError]
