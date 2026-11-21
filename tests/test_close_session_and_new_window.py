"""File > Close session and File > New window.

Close session closes the session this window holds, after the same
Save / Don't save / Cancel offer Open session makes, and leaves the window
as a new one opens: no samples, gates, tabs, trail, session file or lock of
the old session. New window starts another OpenFlo program, which starts
empty instead of offering to resume a session.

Generated FCS files, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import pytest

from openflo import new_window
from openflo import session_lock as _lock
from tests import test_open_session_replaces as _replaces
from tests._tk_wait import settle
from tests.test_open_session_replaces import _Asked, _comparable, _open
from tests.test_transform_records import _new_editor

# The fixtures of Open session's tests: an editor holding an old session
# with an unsaved gate (`two`), on a Tk root with ~ in a temp dir (`env`).
env = _replaces.env
two = _replaces.two


def _close(ed, answer='discard'):
    asked = _Asked(answer)
    ed._ask_choice = asked
    closed = ed._close_session()
    return closed, asked


# ── Close session ──────────────────────────────────────────────────────────

def test_close_leaves_what_a_new_window_holds(env, two):
    """The invariant: after Close session the window saves what a new
    window saves, and holds none of the old session's records."""
    root, _tmp, ed, old, _new = two
    ed._show_audit_window()
    # What opening a session or working in it sets, and closing must clear.
    ed._fmo_controls = {'FL1-A': 'old_2'}
    ed._session_channel_labels = {'FL2-A': 'L2'}
    ed._explicit_channel_labels = {'FL2-A': 'L2'}
    ed._declined_fixes = ['refit-valley']
    ed._legacy_decline_samples = ['old_1']
    ed._pending_sample_meta['p'] = {}
    ed._pending_sample_prep['p'] = {}
    ed._session_restore = {'active': 'old_1'}
    ed._redo_stack.append(ed._undo_stack[-1])
    assert ed._undo_stack and len(ed._audit_log) and ed._session_dir
    closed, asked = _close(ed)
    # Checked before Tk runs: the open's own deferred restore pass, still
    # queued, would clear it too and hide a close that did not.
    assert ed._session_restore is None
    settle(root, 0.2)
    assert closed is True
    assert [q[2] for q in asked.questions] == [['save', 'discard', 'cancel']]
    assert asked.questions[0][3] == 0           # Save is the default button
    fresh = _new_editor(root)
    try:
        assert _comparable(ed._session_state()) == _comparable(
            fresh._session_state())
    finally:
        fresh.destroy()
    assert ed._samples == {} and ed._sample_gates == {}
    assert not ed.__dict__.get('_offline_samples')
    assert ed._tabs().names() == ['All']
    assert ed._layout_doc is None
    assert ed._channels == [] and ed._channel_labels == {}
    assert ed._cluster_labels == {}
    assert ed._session_channel_labels == {}
    assert ed._explicit_channel_labels == {}
    assert ed._fmo_controls == {}
    assert ed._declined_fixes == []
    assert ed._session_restore is None
    assert ed._legacy_decline_samples is None
    assert not ed._pending_sample_meta and not ed._pending_sample_prep
    assert not ed._undo_stack and not ed._redo_stack
    assert len(ed._audit_log) == 0
    # An open History window shows the new (empty) trail.
    assert ed._audit_window._log is ed._audit_log
    assert ed._session_file is None and ed._session_read_only is None
    assert '_session_dir' not in ed.__dict__
    assert '_session_data_dir' not in ed.__dict__
    assert not ed._unsaved
    assert ed.title().startswith('Untitled session')
    # The old file is untouched by the discarded close.
    assert 'unsaved cut' not in old.read_text()


def test_close_lets_go_of_the_session_files_lock(env, two):
    _root, _tmp, ed, old, _new = two
    held = _lock.read_lock(str(old))
    assert held and held.get('token') == ed._session_lock_token()
    _close(ed)
    assert _lock.read_lock(str(old)) is None


def test_a_second_session_after_close_is_saved_alone(env, two, monkeypatch):
    """Close, then open another: Save writes only the new session, and
    the closed session's file is left as it was."""
    root, _tmp, ed, old, new = two
    before = old.read_text()
    _close(ed)
    asked = _open(root, ed, new)
    assert asked.questions == []                # nothing left to save
    assert sorted(ed._samples) == ['new_1']
    assert ed._save_session() is True
    assert [s['name'] for s in json.loads(new.read_text())['samples']] == [
        'new_1']
    assert old.read_text() == before


def test_cancel_keeps_the_session(two):
    _root, _tmp, ed, old, _new = two
    closed, asked = _close(ed, 'cancel')
    assert closed is False and len(asked.questions) == 1
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert ed._session_file == str(old) and ed._unsaved


def test_save_writes_the_session_then_closes(two):
    _root, _tmp, ed, old, _new = two
    closed, _asked = _close(ed, 'save')
    assert closed is True
    assert 'unsaved cut' in old.read_text()
    assert ed._samples == {} and ed._session_file is None


def test_a_save_that_does_not_happen_stops_the_close(two):
    _root, _tmp, ed, _old, _new = two
    ed._save_session = lambda: False            # the name dialog cancelled
    closed, _asked = _close(ed, 'save')
    assert closed is False
    assert sorted(ed._samples) == ['old_1', 'old_2']


def test_no_question_when_everything_is_saved_in_its_own_file(two):
    _root, _tmp, ed, _old, _new = two
    assert ed._save_session() is True
    closed, asked = _close(ed)
    assert closed is True and asked.questions == []
    assert ed._samples == {}


def test_a_sample_move_under_way_stops_the_close(two):
    _root, _tmp, ed, _old, _new = two
    ed._pending_move = {'id': 'x'}
    closed, asked = _close(ed)
    assert closed is False and asked.questions == []
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert 'close the session' in ed.status_var.get()


def test_an_empty_window_has_nothing_to_close(env):
    root, _tmp = env
    ed = _new_editor(root)
    try:
        settle(root, 0.1)
        closed, asked = _close(ed)
        assert closed is False and asked.questions == []
        assert 'nothing to close' in ed.status_var.get()
    finally:
        ed.destroy()


def test_a_read_only_session_with_no_samples_can_be_closed(env):
    """Opened read-only, its samples all gone: there is still a session
    (the title names it) to close."""
    root, tmp = env
    ed = _new_editor(root)
    try:
        ed._remember_session_file(None, read_only=str(tmp / 'x.flowsession'))
        closed, asked = _close(ed)
        assert closed is True and asked.questions == []
        assert ed._session_read_only is None
        assert ed.title().startswith('Untitled session')
    finally:
        ed.destroy()


def test_close_drops_the_old_sessions_queued_loads(two):
    _root, tmp, ed, _old, _new = two
    ed._enqueue_load(('gone', str(tmp / 'gone.fcs')))
    ed._loading.add('gone')
    epoch = ed.__dict__.get('_load_epoch', 0)
    _close(ed)
    assert ed._load_queue.empty() and not ed._loading
    assert ed._load_epoch == epoch + 1


# ── New window ───────────────────────────────────────────────────────────────

def test_a_bundled_build_starts_itself():
    assert new_window.new_window_command(
        r'C:\OpenFlo\OpenFlo.exe', frozen=True, platform='win32', safe_path=False) == [
        r'C:\OpenFlo\OpenFlo.exe']


def test_windows_uses_the_console_less_python(tmp_path):
    (tmp_path / 'python.exe').write_text('')
    (tmp_path / 'pythonw.exe').write_text('')
    cmd = new_window.new_window_command(
        str(tmp_path / 'python.exe'), frozen=False, platform='win32',
        safe_path=False)
    assert cmd == [str(tmp_path / 'pythonw.exe'), '-m', 'openflo.gui']


def test_windows_keeps_python_when_there_is_no_pythonw(tmp_path):
    (tmp_path / 'python.exe').write_text('')
    cmd = new_window.new_window_command(
        str(tmp_path / 'python.exe'), frozen=False, platform='win32',
        safe_path=False)
    assert cmd == [str(tmp_path / 'python.exe'), '-m', 'openflo.gui']


def test_other_systems_run_the_same_python(tmp_path):
    (tmp_path / 'pythonw.exe').write_text('')   # ignored off Windows
    exe = str(tmp_path / 'python.exe')
    assert new_window.new_window_command(
        exe, frozen=False, platform='linux', safe_path=False) == [
        exe, '-m', 'openflo.gui']


@pytest.mark.parametrize('plat', ['win32', 'linux'])
def test_started_with_minus_p_the_new_window_is_too(tmp_path, plat):
    """The installer's shortcut runs `pythonw.exe -P -m openflo.gui`, to
    keep the working folder off the import path: the new window keeps it."""
    (tmp_path / 'python.exe').write_text('')
    exe = str(tmp_path / 'python.exe')
    assert new_window.new_window_command(
        exe, frozen=False, platform=plat, safe_path=True) == [
        exe, '-P', '-m', 'openflo.gui']


@pytest.mark.parametrize('flag', [0, 1])
def test_minus_p_follows_this_programs_flag(monkeypatch, flag):
    import sys
    from types import SimpleNamespace
    monkeypatch.setattr(sys, 'flags', SimpleNamespace(safe_path=flag))
    cmd = new_window.new_window_command('py', frozen=False, platform='linux')
    assert ('-P' in cmd) is bool(flag)


def test_the_new_programs_environment():
    src = os.path.dirname(os.path.dirname(os.path.abspath(
        new_window.__file__)))
    env = new_window.new_window_env({'PYTHONPATH': 'elsewhere', 'X': 'y'})
    assert env['X'] == 'y'
    assert env[new_window.NEW_WINDOW_ENV] == '1'
    assert env['PYTHONPATH'].split(os.pathsep) == [src, 'elsewhere']
    # Already on the path: not added twice.
    again = new_window.new_window_env(env)
    assert again['PYTHONPATH'].split(os.pathsep) == [src, 'elsewhere']
    assert new_window.new_window_env({})['PYTHONPATH'] == src


def test_the_mark_is_read_once(monkeypatch):
    monkeypatch.setenv(new_window.NEW_WINDOW_ENV, '1')
    assert new_window.started_as_new_window() is True
    assert new_window.NEW_WINDOW_ENV not in os.environ
    assert new_window.started_as_new_window() is False


class _Popen:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def __call__(self, cmd, **kw):
        if self.fail is not None:
            raise self.fail
        self.calls.append((cmd, kw))


def test_new_window_starts_another_openflo(env, monkeypatch):
    root, _tmp = env
    ed = _new_editor(root)
    try:
        popen = _Popen()
        monkeypatch.setattr(subprocess, 'Popen', popen)
        assert ed._open_new_window() is True
        (cmd, kw), = popen.calls
        assert cmd == new_window.new_window_command()
        assert kw['env'][new_window.NEW_WINDOW_ENV] == '1'
        # Real output handles, not none (a console-less Python needs them).
        assert kw['stdout'] is not None and kw['stderr'] is not None
        assert kw['stdin'] == subprocess.DEVNULL
        # Its own process group off Windows (a Ctrl+C in this program's
        # terminal does not end it), no console on Windows, no handles
        # of this program's kept open in it.
        assert kw['start_new_session'] is (sys.platform != 'win32')
        assert kw['creationflags'] == getattr(subprocess,
                                              'CREATE_NO_WINDOW', 0)
        assert kw['close_fds'] is True
        assert new_window.NEW_WINDOW_ENV not in os.environ  # not this one's
        assert 'new OpenFlo window' in ed.status_var.get()
    finally:
        ed.destroy()


def test_a_new_window_that_will_not_start_is_reported(env, monkeypatch):
    root, _tmp = env
    ed = _new_editor(root)
    try:
        monkeypatch.setattr(subprocess, 'Popen',
                            _Popen(fail=OSError('no such program')))
        assert ed._open_new_window() is False
        assert 'Could not start a new window' in ed.status_var.get()
    finally:
        ed.destroy()


@pytest.mark.parametrize('marked', [True, False])
def test_a_new_window_does_not_offer_to_resume(env, monkeypatch, marked):
    root, _tmp = env
    ed = _new_editor(root)
    try:
        looked = []
        ed._find_resumable_session = lambda: looked.append(1)
        if marked:
            monkeypatch.setenv(new_window.NEW_WINDOW_ENV, '1')
        else:
            monkeypatch.delenv(new_window.NEW_WINDOW_ENV, raising=False)
        ed._offer_resume_session()
        assert looked == ([] if marked else [1])
        assert new_window.NEW_WINDOW_ENV not in os.environ
    finally:
        ed.destroy()


def test_a_held_ctrl_n_starts_one_window(env, monkeypatch):
    """The key repeats while held: one start per pause, then another."""
    root, _tmp = env
    ed = _new_editor(root)
    try:
        popen = _Popen()
        monkeypatch.setattr(subprocess, 'Popen', popen)
        clock = [100.0]
        monkeypatch.setattr('openflo.editor_session.time.monotonic',
                            lambda: clock[0])
        assert ed._open_new_window() is True
        clock[0] += ed.NEW_WINDOW_PAUSE_S - 0.1
        assert ed._open_new_window() is False
        assert 'already starting' in ed.status_var.get()
        assert len(popen.calls) == 1
        clock[0] += 0.2                         # past the pause
        assert ed._open_new_window() is True
        assert len(popen.calls) == 2
    finally:
        ed.destroy()


def test_a_window_that_did_not_start_does_not_hold_the_next(env, monkeypatch):
    root, _tmp = env
    ed = _new_editor(root)
    try:
        monkeypatch.setattr(subprocess, 'Popen', _Popen(fail=OSError('x')))
        assert ed._open_new_window() is False
        popen = _Popen()
        monkeypatch.setattr(subprocess, 'Popen', popen)
        assert ed._open_new_window() is True and len(popen.calls) == 1
    finally:
        ed.destroy()


# ── A job still working on the session stops Close and Open session ────────

_BUSY = {
    'clustering': ('_clustering_busy', 'clustering'),
    'embedding compare': ('_dr_running', 'embedding comparison'),
    'archive export': ('_archive_exporting', 'project archive'),
}


@pytest.mark.parametrize('job', sorted(_BUSY))
def test_a_running_job_stops_close_session(two, job):
    """Its results would land in the next session's same-named samples
    (review of #463: a clustering run did)."""
    _root, _tmp, ed, _old, _new = two
    attr, words = _BUSY[job]
    setattr(ed, attr, True)
    closed, asked = _close(ed)
    assert closed is False and asked.questions == []
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert words in ed.status_var.get()
    assert 'close the session' in ed.status_var.get()
    setattr(ed, attr, False)                    # finished: it closes
    assert _close(ed)[0] is True


@pytest.mark.parametrize('job', sorted(_BUSY))
def test_a_running_job_stops_open_session(two, job):
    root, _tmp, ed, _old, new = two
    attr, words = _BUSY[job]
    setattr(ed, attr, True)
    asked = _open(root, ed, new)
    assert asked.questions == []
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert words in ed.status_var.get()
    assert 'open the session' in ed.status_var.get()


def test_a_job_in_the_status_bar_stops_close_until_it_ends(two):
    root, _tmp, ed, _old, _new = two
    go = threading.Event()
    done = []
    assert ed.run_job(lambda job: go.wait(10), on_done=done.append,
                      text="Testing the strip…") is not None
    try:
        closed, asked = _close(ed)
        assert closed is False and asked.questions == []
        assert 'Testing the strip is still running' in ed.status_var.get()
        assert sorted(ed._samples) == ['old_1', 'old_2']
    finally:
        go.set()
    from tests._tk_wait import pump_until
    assert pump_until(root, lambda: not ed._job_strip.running, timeout=10)
    assert _close(ed)[0] is True


def _real_run_async(ed, monkeypatch):
    """The editor's own run_async (the test editor runs jobs inline). A
    worker's result reaches Tk with `after` from its thread, which needs a
    running mainloop; here the posts are kept and `land()` runs them on
    this thread, as the loop would. Returns `land`."""
    import queue

    from openflo import async_task
    ed.__dict__.pop('run_async', None)
    posted = queue.Queue()
    monkeypatch.setattr(async_task, '_post', lambda _w, fn: posted.put(fn))

    def land(timeout=10):
        fn = posted.get(timeout=timeout)
        fn()
    return land


def test_a_transform_change_still_landing_stops_close(two, monkeypatch):
    """Review of #463, round 2: a Transform change's result is written
    into the samples by name when it lands, so after a close it replaced
    a same-named sample's data in the next session. Close now waits until
    the job's result has been delivered."""
    _root, _tmp, ed, _old, _new = two
    land = _real_run_async(ed, monkeypatch)
    ch = 'FL1-A'
    cur = ed._channel_transform.get(ch, 'linear')
    ed._apply_channel_transforms(
        {ch: 'linear' if cur != 'linear' else 'logicle'})
    # Tk has not run, so the result cannot have landed yet.
    closed, asked = _close(ed)
    assert closed is False and asked.questions == []
    assert 'Re-transforming 1 channel(s)' in ed.status_var.get()
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert ed._busy_bar.grid_info()             # the busy bar shows
    before = ed._samples['old_2'].data[ch].to_numpy().copy()
    land()                                      # the result lands
    assert not ed._session_jobs
    assert not ed._busy_bar.grid_info()         # and is put away
    assert ed._channel_transform.get(ch, 'linear') != cur  # it was applied
    assert not (ed._samples['old_2'].data[ch].to_numpy() == before).all()
    assert _close(ed)[0] is True


@pytest.mark.parametrize('outcome', ['done', 'error'])
def test_every_background_job_is_waited_for_then_let_go(two, outcome,
                                                       monkeypatch):
    """Any job of the window's run_async counts, with a message or
    without, and stops counting once it has landed, failed or not."""
    root, _tmp, ed, _old, new = two
    land = _real_run_async(ed, monkeypatch)
    go = threading.Event()
    landed = []

    def work():
        go.wait(10)
        if outcome == 'error':
            raise ValueError('generated failure')
        return 1

    ed.run_async(work, on_done=landed.append, on_error=landed.append)
    try:
        closed, asked = _close(ed)
        assert closed is False and asked.questions == []
        assert 'a background job is still running' in ed.status_var.get()
        asked = _open(root, ed, new)            # Open session waits too
        assert asked.questions == []
        assert sorted(ed._samples) == ['old_1', 'old_2']
    finally:
        go.set()
    land()
    assert len(landed) == 1 and not ed._session_jobs
    assert _close(ed)[0] is True
