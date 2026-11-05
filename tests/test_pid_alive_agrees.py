"""Autosave resume/pruning and session locks ask the same question, "is the
process with this PID still running?", and must get the same answer.

The autosave copy used to treat a process we may not signal (PermissionError:
alive, but owned by another account) as gone, so resume could offer, and
pruning could delete, an autosave a running instance still owned, while the
session lock called the same process alive.
"""
import json
import os
import types

import pytest

from openflo import session_lock
from openflo.gui import ViewGateEditorWindow as V


def _kill_raising(exc):
    def kill(pid, sig):
        if exc is not None:
            raise exc
    return kill


@pytest.mark.parametrize('exc, alive', [
    (None, True),                                   # signal allowed: running
    (PermissionError(1, 'not permitted'), True),    # running, someone else's
    (ProcessLookupError(3, 'no such process'), False),
    (OSError(22, 'invalid argument'), False),
    (OverflowError('pid too large'), False),
])
def test_autosave_and_lock_agree_on_every_kill_outcome(monkeypatch, exc, alive):
    monkeypatch.setattr(os, 'name', 'posix')
    monkeypatch.setattr(os, 'kill', _kill_raising(exc))
    assert session_lock.pid_alive(4242) is alive
    assert V._pid_alive(4242) is alive


@pytest.mark.parametrize('pid', [None, 'x', 0, -5])
def test_no_usable_pid_is_not_running(pid):
    assert V._pid_alive(pid) is False
    assert session_lock.pid_alive(pid) is False


def _autosave_stub(ad):
    return types.SimpleNamespace(
        SESSION_EXT='.flowsession',
        _autosave_dir=lambda: str(ad),
        _pid_from_autosave=lambda p: V._pid_from_autosave(None, p),
        _pid_alive=V._pid_alive)


def test_resume_never_offers_an_autosave_another_account_still_runs(
        tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    ad = tmp_path / 'autosave'
    ad.mkdir()
    f = ad / 'session-4242-tok.flowsession'
    f.write_text(json.dumps({'format': 'openflo-session'}))
    monkeypatch.setattr(os, 'name', 'posix')
    # Control: once its owner has exited, the same file is offered.
    monkeypatch.setattr(os, 'kill',
                        _kill_raising(ProcessLookupError(3, 'no such process')))
    assert V._find_resumable_session(_autosave_stub(ad)) == str(f)
    monkeypatch.setattr(os, 'kill',
                        _kill_raising(PermissionError(1, 'not permitted')))
    assert V._find_resumable_session(_autosave_stub(ad)) is None


def test_pruning_keeps_an_old_autosave_another_account_still_runs(
        tmp_path, monkeypatch):
    ad = tmp_path / 'autosave'
    ad.mkdir()
    f = ad / 'session-4242-tok.flowsession'
    f.write_text('{}')
    dd = ad / 'session-4242-tok_data'
    dd.mkdir()
    os.utime(str(f), (1_000_000_000, 1_000_000_000))     # 2001: past cutoff
    monkeypatch.setattr(os, 'name', 'posix')
    monkeypatch.setattr(os, 'kill',
                        _kill_raising(PermissionError(1, 'not permitted')))
    V._prune_autosaves(_autosave_stub(ad))
    assert f.exists(), 'pruned an autosave a running process owns'
    assert dd.exists(), 'pruned the data folder of a running autosave'
