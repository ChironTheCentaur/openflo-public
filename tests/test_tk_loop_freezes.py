"""Three menu actions that froze every window (control sweep, batch 2).

* Analyze > Compare embeddings… ran the umap and scikit-learn import checks
  (child processes, seconds each) on the Tk thread the first time it opened
  (CLUSTER-2). The window now opens at once and fills its boxes in from a
  background check.
* Kinetics took any number of Bins: 2,000,000 froze the window for seconds
  (STATS-7). Above ``MAX_BINS`` it now says so and computes nothing.
* File > Export project archive… wrote the zip on the Tk thread. It now
  writes on a worker with the busy bar showing; a second export or a close
  meanwhile is refused.

Each freeze is measured as the longest gap between ticks of a 10 ms Tk timer
while the action runs, with the slow part faked slow by patching that one
call. Generated data throughout.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import pytest

from tests._tk_wait import pump_until, settle
from tests.conftest import gui_unavailable
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
)

MAX_GAP = 0.5           # s: the longest the Tk loop may stall


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path
    root.destroy()


def _longest_gap(root, action, done, timeout=30.0):
    """Run ``action()`` from the Tk loop, pump until ``done()``, and return
    the longest stall (s) of a 10 ms timer from just before ``action``."""
    ticks: list[float] = []
    state = {'ran': False, 'after': None}

    def tick():
        ticks.append(time.perf_counter())
        state['after'] = root.after(10, tick)

    def run():
        try:
            action()
        finally:
            state['ran'] = True

    tick()
    root.after(30, run)
    assert pump_until(root, lambda: state['ran'] and done(), timeout=timeout)
    root.after_cancel(state['after'])
    ticks.append(time.perf_counter())
    return float(np.max(np.diff(ticks)))


def _run_async_off_the_tk_thread(ed):
    """Give ``ed`` the editor's ``run_async`` as the app runs it: the work on
    a thread, the busy bar while it runs, the answer back on the Tk thread.
    (_new_editor runs the work inline. The app's async_task posts the answer
    with ``after`` from the worker, which needs a running mainloop; these
    tests pump ``update``, so here the Tk thread polls for it instead.)"""
    def run_async(work, on_done=None, on_error=None, busy_msg=None):
        box: dict = {}

        def worker():
            try:
                box['result'] = work()
            except Exception as exc:              # noqa: BLE001 - handed to on_error on the Tk thread, as async_task does
                box['error'] = exc

        if busy_msg is not None:
            ed._begin_busy(busy_msg)
        t = threading.Thread(target=worker, daemon=True)
        t.start()

        def poll():
            if t.is_alive():
                ed.after(20, poll)
                return
            ed.async_answers.append(work)
            if busy_msg is not None:
                ed._end_busy()
            if 'error' in box:
                on_error(box['error'])
            elif on_done is not None:
                on_done(box['result'])
        ed.after(20, poll)
        return t
    ed.async_answers = []           # each finished work(), when answered
    ed.run_async = run_async


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


# ── CLUSTER-2: Compare embeddings opens without the import check ─────────

def _boxes(dlg):
    from tkinter import ttk
    return {str(cb.cget('text')).split()[0]: cb for cb in _walk(dlg)
            if isinstance(cb, ttk.Checkbutton)}


def _dialog(ed):
    import tkinter as tk
    found = [w for w in ed.winfo_children() if isinstance(w, tk.Toplevel)
             and w.winfo_exists() and w.title() == 'Compare embeddings']
    return found[-1] if found else None


def test_compare_embeddings_opens_at_once_while_the_import_check_runs(
        env, monkeypatch):
    from openflo import capabilities, warmup
    from openflo.pipeline import FlowSample
    root, _tmp = env
    monkeypatch.setattr(warmup, 'start', lambda *a, **k: None)
    # Nothing checked yet, as in a fresh editor; the one slow call (the
    # child-process import check) takes 1.5 s per call, umap and scikit-learn
    # installed, the optional backends not.
    monkeypatch.setattr(capabilities, '_IMPORT_RESULT', {})
    monkeypatch.setattr(capabilities, '_installed', lambda m: True)
    checks = []

    def slow_check(modules):
        checks.append(tuple(modules))
        time.sleep(1.5)
        return {m: (None if m in ('umap', 'sklearn')
                    else capabilities.NOT_INSTALLED) for m in modules}
    monkeypatch.setattr(capabilities, '_import_check', slow_check)

    ed = _new_editor(root)
    _run_async_off_the_tk_thread(ed)
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(400, 3)),
                      columns=['FL1-A', 'FL2-A', 'FL3-A'])
    ed._on_loaded('synthetic_A', FlowSample.from_dataframe(
        df, name='synthetic_A'))
    ed._set_active_sample('synthetic_A')
    settle(root, 2.0)               # the editor's own first draw is done
    # Fitting a newly mapped window waits 2 s for a window manager that
    # Xvfb doesn't have, on the old code and the new alike: not measured.
    root.unbind_class('Toplevel', '<Map>')

    def ready():
        dlg = _dialog(ed)
        return dlg is not None and dlg._have is not None

    gap = _longest_gap(root, ed._open_dr_compare, ready)
    assert gap < MAX_GAP, f"the Tk loop stalled {gap:.2f} s on open"
    assert checks                                    # the check did run
    dlg = _dialog(ed)
    boxes = _boxes(dlg)
    assert set(boxes) == {'UMAP', 't-SNE', 'PHATE', 'TriMap', 'PaCMAP'}
    for name, cb in boxes.items():
        on = bool(ed.getvar(cb.cget('variable')))
        enabled = not cb.instate(['disabled'])
        if name in ('UMAP', 't-SNE'):
            assert on and enabled, name
            assert 'not installed' not in str(cb.cget('text'))
        else:
            assert not on and not enabled, name
            assert 'pip install "openflo[embed]"' in str(cb.cget('text'))
    dlg.destroy()

    # The second open has the answer cached: no check, boxes filled at once.
    n_checks = len(checks)
    gap = _longest_gap(root, ed._open_dr_compare,
                       lambda: _dialog(ed) is not None)
    assert gap < MAX_GAP
    assert len(checks) == n_checks
    dlg = _dialog(ed)
    assert dlg._have == {'umap', 'tsne'}
    on = {n for n, cb in _boxes(dlg).items()
          if ed.getvar(cb.cget('variable'))}
    assert on == {'UMAP', 't-SNE'}
    dlg.destroy()


def _embed_editor(root, monkeypatch, installed=('umap', 'sklearn'),
                  seconds=1.0):
    """An editor with one generated sample, nothing checked yet, and the
    child-process import check faked slow: ``installed`` modules import,
    the rest are not installed. Returns (editor, list of checks run)."""
    from openflo import capabilities, warmup
    from openflo.pipeline import FlowSample
    monkeypatch.setattr(warmup, 'start', lambda *a, **k: None)
    monkeypatch.setattr(capabilities, '_IMPORT_RESULT', {})
    monkeypatch.setattr(capabilities, '_installed', lambda m: True)
    checks = []

    def slow_check(modules):
        checks.append(tuple(modules))
        time.sleep(seconds)
        return {m: (None if m in installed else capabilities.NOT_INSTALLED)
                for m in modules}
    monkeypatch.setattr(capabilities, '_import_check', slow_check)
    ed = _new_editor(root)
    _run_async_off_the_tk_thread(ed)
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(size=(400, 3)),
                      columns=['FL1-A', 'FL2-A', 'FL3-A'])
    ed._on_loaded('synthetic_A', FlowSample.from_dataframe(
        df, name='synthetic_A'))
    ed._set_active_sample('synthetic_A')
    settle(root, 1.0)
    root.unbind_class('Toplevel', '<Map>')      # see the test above
    return ed, checks


def _spy_errors(root, ed, monkeypatch):
    """Everything that reaches the editor's error handler (and so the
    error window) or ``errors.report``, as a list."""
    from openflo import errors
    caught = []
    real_handler = root.report_callback_exception
    root.report_callback_exception = lambda *a: (caught.append(a[1]),
                                                 real_handler(*a))
    real_report = errors.report
    monkeypatch.setattr(errors, 'report',
                        lambda w, exc: (caught.append(exc),
                                        real_report(w, exc)))
    return caught


def test_closing_compare_embeddings_while_the_check_runs_is_quiet(
        env, monkeypatch):
    from tkinter import messagebox
    root, _tmp = env
    ed, checks = _embed_editor(root, monkeypatch)
    caught = _spy_errors(root, ed, monkeypatch)
    infos = []
    monkeypatch.setattr(messagebox, 'showinfo',
                        lambda *a, **k: infos.append(a))
    ed._open_dr_compare()
    dlg = _dialog(ed)
    assert dlg is not None and dlg._have is None     # still checking
    dlg.destroy()                                    # Cancel / close box
    assert pump_until(root, lambda: ed.async_answers, timeout=30)
    settle(root, 0.3)
    assert checks                                    # the check did answer
    assert caught == [] and ed._error_count == 0     # no error window
    assert infos == [] and _dialog(ed) is None       # nothing else opened


def test_compare_embeddings_closes_when_the_check_finds_nothing_installed(
        env, monkeypatch):
    from tkinter import messagebox
    root, _tmp = env
    ed, checks = _embed_editor(root, monkeypatch, installed=())
    infos = []
    monkeypatch.setattr(messagebox, 'showinfo',
                        lambda *a, **k: infos.append(a))
    ed._open_dr_compare()
    assert _dialog(ed) is not None                   # opened at once
    assert pump_until(root, lambda: ed.async_answers, timeout=30)
    settle(root, 0.3)
    assert checks
    assert _dialog(ed) is None                       # closed itself
    (title, message), = infos
    assert title == "Embedding comparison"
    assert message.startswith("No embedding backends are installed.")


def test_compare_embeddings_run_waits_for_the_check(env, monkeypatch):
    root, _tmp = env
    ed, _checks = _embed_editor(root, monkeypatch)
    started = []
    ed._start_embedding = lambda *a: started.append(a)
    ed._open_dr_compare()
    dlg = _dialog(ed)
    assert dlg._have is None
    dlg._run()                                       # Run while checking
    assert ed.status_var.get() == "Still checking which methods are installed…"
    assert started == [] and dlg.winfo_exists()
    assert pump_until(root, lambda: ed.async_answers, timeout=30)
    dlg.destroy()


def test_a_failed_import_check_closes_the_dialog_and_reports_it(
        env, monkeypatch):
    from openflo import dr_compare
    root, _tmp = env
    ed, _checks = _embed_editor(root, monkeypatch)
    caught = _spy_errors(root, ed, monkeypatch)
    boom = RuntimeError('import check broke')

    def broken():
        time.sleep(0.5)
        raise boom
    monkeypatch.setattr(dr_compare, 'available_methods', broken)
    ed._open_dr_compare()
    assert _dialog(ed) is not None
    assert pump_until(root, lambda: ed.async_answers, timeout=30)
    settle(root, 0.3)
    assert _dialog(ed) is None
    assert boom in caught


# ── STATS-7 (freeze only): Kinetics Bins has a maximum ───────────────────

def test_kinetics_refuses_more_bins_than_the_maximum(env, monkeypatch):
    from tkinter import messagebox

    from openflo import kinetics, ui_kinetics
    from tests.test_derived_kinetics import _fcs, _loaded
    root, tmp = env
    ed = _new_editor(root)
    ed._on_loaded('synthetic_A', _loaded(_fcs(tmp / 'a.fcs', 1)))
    ed._refresh_channel_choices()
    errors = []
    monkeypatch.setattr(messagebox, 'showerror',
                        lambda *a, **k: errors.append(a))
    used = []
    real = kinetics.trace

    def spy(*a, **k):
        used.append(k['bins'])
        return real(*a, **k)
    monkeypatch.setattr(kinetics, 'trace', spy)
    win = ui_kinetics.KineticsWindow(ed)
    win.chan_var.set(ed._fmt_channel('FL1-A'))

    win.bins_var.set('10000')                    # the maximum still runs
    used.clear()
    win._refresh()
    assert not errors and used and set(used) == {10_000}
    shown = list(win._results)

    for text in ('10001', '2000000'):
        win.bins_var.set(text)
        used.clear()
        t0 = time.perf_counter()
        win._refresh()
        assert time.perf_counter() - t0 < MAX_GAP
        assert used == []                        # nothing was computed
        assert len(errors) == 1
        _title, message = errors.pop()
        assert 'Bins' in message and '10,000' in message and text in message
        assert win._results == shown             # the last result stays
    win.destroy()


# ── File > Export project archive… writes on a worker ─────────────────────

def _archive_editor(root, tmp):
    from tests.test_session_safety import _editor
    ed = _editor(root, tmp)
    _run_async_off_the_tk_thread(ed)
    settle(root, 2.0)               # the loads' own status updates are done
    return ed


def _slow_archive(monkeypatch, seconds=1.0, fail=None):
    """Make the one zip writer slow (and, with ``fail``, raise it after)."""
    from openflo import project_archive
    real = project_archive.write_project_archive
    calls = []

    def slow(session_path, out_path):
        calls.append(out_path)
        time.sleep(seconds)
        if fail is not None:
            raise fail
        return real(session_path, out_path)
    monkeypatch.setattr(project_archive, 'write_project_archive', slow)
    return calls


def test_project_archive_is_written_off_the_tk_thread_and_reopens(
        env, monkeypatch):
    from tkinter import filedialog

    from openflo.project_archive import open_project_archive
    from tests.test_session_restore_transforms import _restore
    from tests.test_session_safety import _counts
    root, tmp = env
    scratch = tmp / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    ed = _archive_editor(root, tmp)
    before = _counts(ed)
    out = str(tmp / 'send.zip')
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: out)
    calls = _slow_archive(monkeypatch)
    busy = []
    real_begin = ed._begin_busy
    ed._begin_busy = lambda msg=None: (busy.append(msg), real_begin(msg))
    gap = _longest_gap(root, ed._export_project_archive,
                       lambda: os.path.exists(out) and not getattr(
                           ed, '_archive_exporting', False))
    assert gap < MAX_GAP, f"the Tk loop stalled {gap:.2f} s on export"
    assert calls == [out]
    assert busy == ["Exporting project archive…"]
    assert not ed._archive_export_running()
    assert ed.status_var.get() == "Exported project: 2 sample(s) → send.zip"
    # The archive is whole: it opens elsewhere with the same results.
    shutil.rmtree(tmp / 'data')
    other = tmp / 'colleague'
    other.mkdir()
    opened, _ = open_project_archive(shutil.move(out, other / 'send.zip'))
    ed2 = _restore(root, opened, raw_first=True)
    assert set(ed2._samples) == {'synthetic_A', 'synthetic_B'}
    assert _counts(ed2) == before
    # No temporary session folder is left behind (mkdtemp writes them to
    # tempfile.tempdir, pointed at ``scratch`` above).
    assert os.listdir(scratch) == []


def test_a_failed_archive_shows_a_message_and_keeps_the_window_live(
        env, monkeypatch):
    from tkinter import filedialog

    from openflo import editor_session, errors
    root, tmp = env
    ed = _archive_editor(root, tmp)
    out = str(tmp / 'send.zip')
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: out)
    _slow_archive(monkeypatch, fail=OSError(28, 'No space left on device'))
    shown, reported = [], []
    monkeypatch.setattr(editor_session, 'show_error',
                        lambda *a, **k: shown.append(a))
    monkeypatch.setattr(errors, 'report', lambda *a, **k: reported.append(a))
    gap = _longest_gap(root, ed._export_project_archive, lambda: shown)
    assert gap < MAX_GAP, f"the Tk loop stalled {gap:.2f} s on export"
    assert reported == []                        # not the traceback window
    (_parent, title, message, exc, path), = shown
    assert title == "Project export failed"
    assert message == "The project could not be exported."
    assert isinstance(exc, OSError) and path == out
    assert 'No space left' in ed.status_var.get()
    assert not os.path.exists(out)
    assert not ed._archive_export_running()


def test_an_archive_whose_session_cannot_be_read_says_so_and_leaves_nothing(
        env, monkeypatch):
    """When the Tk-thread half of File > Export project archive (reading the
    session to save) raises, the error is shown in the same message as a
    failed write, the export is not left marked as running (so a close or
    a second export is not held up), and no archive, half-written
    ``.tmp`` or temporary session folder is left behind."""
    from tkinter import filedialog

    from openflo import editor_session, errors, project_archive
    root, tmp = env
    scratch = tmp / 'scratch'
    scratch.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    ed = _archive_editor(root, tmp)
    out = str(tmp / 'send.zip')
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: out)
    written = []
    monkeypatch.setattr(project_archive, 'write_project_archive',
                        lambda *a: written.append(a))
    boom = RuntimeError('a gate could not be read')

    def snapshot_fails(path):
        raise boom
    monkeypatch.setattr(ed, '_plan_session_write', snapshot_fails)
    shown, reported = [], []
    monkeypatch.setattr(editor_session, 'show_error',
                        lambda *a, **k: shown.append(a))
    monkeypatch.setattr(errors, 'report', lambda *a, **k: reported.append(a))
    ed._export_project_archive()
    settle(root, 0.3)
    assert reported == []                        # not the traceback window
    (_parent, title, message, exc, path), = shown
    assert title == "Project export failed"
    assert message == "The project could not be exported."
    assert exc is boom and path == out
    assert ed.status_var.get() == \
        "Project export failed: a gate could not be read"
    assert not ed._archive_export_running()
    assert written == []
    assert not os.path.exists(out) and not os.path.exists(out + '.tmp')
    assert not [f for f in os.listdir(tmp) if f.endswith('.tmp')]
    assert os.listdir(scratch) == []


def test_a_second_export_or_a_close_waits_for_the_archive(env, monkeypatch):
    from tkinter import filedialog, messagebox
    root, tmp = env
    ed = _archive_editor(root, tmp)
    out = str(tmp / 'send.zip')
    asked = []
    monkeypatch.setattr(filedialog, 'asksaveasfilename',
                        lambda **k: (asked.append(k), out)[1])
    infos = []
    monkeypatch.setattr(messagebox, 'showinfo',
                        lambda *a, **k: infos.append(a))
    closes = []
    ed._start_close_save = lambda plans: closes.append(plans)
    ed._finish_close = lambda *a, **k: closes.append(a)
    calls = _slow_archive(monkeypatch, seconds=1.5)

    def during():
        ed._export_project_archive()
        root.after(200, ed._export_project_archive)  # a second one
        root.after(300, lambda: ed._on_close(ask=False))

    _longest_gap(root, during,
                 lambda: 'Exported project' in ed.status_var.get())
    assert calls == [out] and len(asked) == 1
    assert closes == []                          # the close was refused
    assert [a[0] for a in infos] == ["Project archive still being written"]
    # Once the archive is written, the close goes ahead.
    ed._on_close(ask=False)
    assert closes


def test_an_archive_missing_a_samples_fcs_file_says_so(env, monkeypatch):
    from tkinter import filedialog
    root, tmp = env
    ed = _archive_editor(root, tmp)
    warned = capture_warnings(monkeypatch)
    out = str(tmp / 'send.zip')
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: out)
    os.remove(tmp / 'data' / 'synthetic_B.fcs')
    ed._export_project_archive()
    assert pump_until(root, lambda: ed.async_answers, timeout=30)
    assert os.path.exists(out)
    (title, message), = warned
    assert title == "Project exported without some data files"
    assert 'synthetic_B' in message and 'synthetic_A' not in message
