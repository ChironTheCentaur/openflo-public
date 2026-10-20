"""Two windows on one session never overwrite each other silently.

* A session opened or saved by name is locked (``name.flowsession.lock``),
  so a second window, or a second person on a shared folder, is told it is
  open and picks Open read-only, Open anyway or Cancel. A lock left by a
  process that is gone, on this computer, is taken over without asking.
* Before Save (and Save and close) writes over the file this window opened,
  it checks the file is still the one it read or last wrote. If someone
  saved in between, the user picks Save as a copy, Overwrite or Cancel,
  and nothing is overwritten without that word.

Generated data throughout: synthetic FCS files and the sessions written from
them. Each rule has its negative case beside it.
"""
from __future__ import annotations

import json
import os

import pytest

from tests.test_session_restore_transforms import _restore
from tests.test_session_safety import _answer, _editor, _gate, _gate_names
from tests.test_session_safety import env as env  # noqa: F401  (fixture)

# ── the lock file and the stamp, without a window ────────────────────────


def test_a_free_session_is_locked_and_released_by_its_holder(tmp_path):
    from openflo import session_lock as L
    s = str(tmp_path / 'a.flowsession')
    assert L.acquire(s, 'tok-1') is None
    held = L.read_lock(s)
    assert held['token'] == 'tok-1' and held['pid'] == os.getpid()
    assert L.acquire(s, 'tok-1') is None            # ours already
    L.release(s, 'tok-1')
    assert not os.path.exists(L.lock_path(s))


def test_a_lock_held_by_another_window_is_reported_and_kept(tmp_path):
    from openflo import session_lock as L
    s = str(tmp_path / 'a.flowsession')
    _fake_lock(s, host='lab-pc', user='someone', pid=4321, token='tok-1')
    held = L.acquire(s, 'tok-2')
    assert held is not None and held['token'] == 'tok-1'
    assert 'someone on lab-pc' in L.describe_holder(held)
    L.release(s, 'tok-2')                           # not its lock
    assert L.read_lock(s)['token'] == 'tok-1'
    assert L.acquire(s, 'tok-2', force=True) is None    # Open anyway
    assert L.read_lock(s)['token'] == 'tok-2'
    L.release(s, 'tok-1')                           # taken over: kept
    assert L.read_lock(s)['token'] == 'tok-2'


def _fake_lock(s, **info):
    from openflo import session_lock as L
    with open(L.lock_path(s), 'w', encoding='utf-8') as f:
        json.dump(info, f)


def test_another_window_of_this_process_shares_without_asking(tmp_path):
    from openflo import session_lock as L
    s = str(tmp_path / 'a.flowsession')
    L.acquire(s, 'window-1')
    assert L.held_here(L.read_lock(s))
    assert L.acquire(s, 'window-2') is None         # no question
    assert L.read_lock(s)['token'] == 'window-1'    # and not taken
    _fake_lock(s, host=L._host(), pid=os.getppid(), token='other-process')
    assert not L.held_here(L.read_lock(s))


def test_a_lock_left_by_a_dead_process_here_is_taken_over(tmp_path):
    from openflo import session_lock as L
    s = str(tmp_path / 'a.flowsession')
    _fake_lock(s, host=L._host(), pid=999_999_991, token='crashed')
    assert L.is_stale(L.read_lock(s))
    assert L.acquire(s, 'tok-1') is None
    assert L.read_lock(s)['token'] == 'tok-1'


@pytest.mark.parametrize('info', [
    {'host': 'another-computer', 'pid': 999_999_991, 'token': 'x'},
    {'pid': os.getppid(), 'token': 'x'},            # alive, here
])
def test_a_lock_from_elsewhere_or_a_live_process_is_held(tmp_path, info):
    from openflo import session_lock as L
    s = str(tmp_path / 'a.flowsession')
    if 'host' not in info:
        info = dict(info, host=L._host())
    _fake_lock(s, **info)
    assert not L.is_stale(L.read_lock(s))
    assert L.acquire(s, 'tok-1') == info


def test_an_unreadable_lock_still_counts_as_held(tmp_path):
    from openflo import session_lock as L
    s = str(tmp_path / 'a.flowsession')
    with open(L.lock_path(s), 'w', encoding='utf-8') as f:
        f.write('{"host": "cut sh')
    assert L.read_lock(s) == {}
    assert L.acquire(s, 'tok-1') == {}
    assert 'lock file' in L.describe_holder({})


def test_the_stamp_sees_a_change_a_deletion_and_nothing_else(tmp_path):
    from openflo import session_lock as L
    p = tmp_path / 'a.flowsession'
    p.write_text('{"one": 1}')
    st = L.file_stamp(str(p))
    assert not L.changed_on_disk(str(p), st)
    assert not L.changed_on_disk(str(p), None)
    p.write_text('{"one": 1, "two": 2}')
    assert L.changed_on_disk(str(p), st)
    st = L.file_stamp(str(p))
    os.utime(p, ns=(st[1] + 5_000_000_000, st[1] + 5_000_000_000))
    assert L.changed_on_disk(str(p), st)            # same size, rewritten
    p.unlink()
    assert L.changed_on_disk(str(p), st)


# ── in the editor ────────────────────────────────────────────────────────


def _choose(ed, answer):
    """Answer the editor's own choice dialog; record what it asked."""
    asked = []

    def fn(title, text, choices, default=0):
        asked.append((title, text, [v for _l, v in choices]))
        return answer
    ed._ask_choice = fn
    return asked


def _saved_session(root, tmp, monkeypatch):
    """A session saved by name in window 1, which holds its lock."""
    from openflo import session_lock as L
    ed = _editor(root, tmp)
    path = str(tmp / 'proj' / 's.flowsession')
    os.makedirs(os.path.dirname(path))
    _answer(monkeypatch, asksaveasfilename=path)
    ed._save_session()
    assert L.read_lock(path)['token'] == ed._session_lock_token()
    return ed, path


def test_saving_is_quiet_when_nobody_else_saved(env, monkeypatch):
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    asked = _choose(ed, 'cancel')
    _gate(ed, 'synthetic_A', 'second save', 0.6)
    ed._save_session()
    assert not asked
    assert 'second save' in _gate_names(path)


def _saved_elsewhere(ed, path):
    """Another window saves the session after `ed` last did."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    data['sample_gates']['synthetic_B'].append(
        {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.9,
         'parent_id': None, 'name': 'saved elsewhere', 'enabled': True,
         'id': 'gz'})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


@pytest.mark.parametrize('answer', ['cancel', 'overwrite', 'copy'])
def test_a_save_made_elsewhere_is_never_overwritten_unasked(
        env, monkeypatch, answer):
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    _saved_elsewhere(ed, path)
    copy = str(tmp / 'proj' / 's (copy).flowsession')
    asked_files = _answer(monkeypatch, asksaveasfilename=copy)
    asked = _choose(ed, answer)
    _gate(ed, 'synthetic_A', 'this window', 0.6)
    ed._save_session()
    assert [a[2] for a in asked] == [['copy', 'overwrite', 'cancel']]
    assert 'changed by another window' in asked[0][1]
    names = _gate_names(path)
    if answer == 'overwrite':
        assert 'this window' in names and 'saved elsewhere' not in names
        assert ed._open_session_file() == path
    else:
        assert 'saved elsewhere' in names and 'this window' not in names
    if answer == 'copy':
        assert len(asked_files['asksaveasfilename']) == 1
        assert 'this window' in _gate_names(copy)
        assert ed._open_session_file() == copy
    else:
        assert not os.path.exists(copy)


def test_a_session_deleted_elsewhere_is_asked_about_too(env, monkeypatch):
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    os.remove(path)
    asked = _choose(ed, 'cancel')
    ed._save_session()
    assert 'deleted or moved' in asked[0][1]
    assert not os.path.exists(path)


def test_opening_a_free_session_asks_nothing_and_locks_it(env, monkeypatch):
    from openflo import session_lock as L
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 's.flowsession')
    ed._write_session(path)                         # saved, not held
    ed2 = _restore(root, path, raw_first=True)
    assert L.read_lock(path)['token'] == ed2._session_lock_token()
    assert ed2._open_session_file() == path


def test_a_save_made_elsewhere_while_opening_is_still_caught(
        env, monkeypatch):
    """The stamp is of the file as read, so a save that lands while the
    session is being applied is not mistaken for this window's."""
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 's.flowsession')
    ed._write_session(path)
    from tests.test_transform_records import _new_editor
    ed2 = _new_editor(root)
    ed2._apply_session = lambda data: _saved_elsewhere(ed2, path)
    ed2._load_session_path(path)
    asked = _choose(ed2, 'cancel')
    ed2._samples = ed._samples
    ed2._save_session()
    assert len(asked) == 1
    assert 'saved elsewhere' in _gate_names(path)


def _held_elsewhere(path):
    """Another person's OpenFlo on another computer has `path` open."""
    _fake_lock(path, host='lab-pc', user='someone', pid=4321,
               token='elsewhere', since=1_700_000_000)


def _open_second(root, path, answer):
    """A second window opens `path`, answering the lock question."""
    from tests.test_transform_records import _new_editor
    ed2 = _new_editor(root)
    asked = _choose(ed2, answer)
    opened = []
    ed2._apply_session = lambda data: opened.append(data)
    ed2._load_session_path(path)
    return ed2, asked, opened


@pytest.mark.parametrize('answer', ['cancel', 'readonly', 'anyway'])
def test_a_session_open_in_another_window_asks_first(
        env, monkeypatch, answer):
    from openflo import session_lock as L
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    _held_elsewhere(path)
    ed2, asked, opened = _open_second(root, path, answer)
    assert [a[2] for a in asked] == [['readonly', 'anyway', 'cancel']]
    assert 'already open in someone on lab-pc' in asked[0][1]
    holder = L.read_lock(path)['token']
    if answer == 'cancel':
        assert not opened and holder == 'elsewhere'
        return
    assert len(opened) == 1
    if answer == 'readonly':
        assert holder == 'elsewhere'
        assert ed2._open_session_file() is None
        assert '(read-only)' in ed2.title()
    else:
        assert holder == ed2._session_lock_token()
        assert ed2._open_session_file() == path


def test_a_read_only_session_saves_only_under_a_new_name(env, monkeypatch):
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    _held_elsewhere(path)
    before = _gate_names(path)
    ed2 = _restore_read_only(root, path)
    asked = _answer(monkeypatch, asksaveasfilename='')
    _gate(ed2, 'synthetic_A', 'read-only edit', 0.6)
    ed2._save_session()
    assert len(asked['asksaveasfilename']) == 1
    assert _gate_names(path) == before


def _restore_read_only(root, path):
    from tests.test_transform_records import _new_editor
    real = _new_editor

    def new_editor(r):
        ed = real(r)
        _choose(ed, 'readonly')
        return ed
    import tests.test_session_restore_transforms as rt
    old = rt._new_editor
    rt._new_editor = new_editor
    try:
        return _restore(root, path, raw_first=True)
    finally:
        rt._new_editor = old


def test_a_backup_opens_without_touching_any_lock(env, monkeypatch):
    from openflo import session_lock as L
    from openflo.editor_session import session_backup_path
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    ed._save_session()                              # makes the backup
    backup = session_backup_path(path)
    ed2, asked, opened = _open_second(root, backup, 'cancel')
    assert not asked and len(opened) == 1
    assert not os.path.exists(L.lock_path(backup))


def test_closing_releases_the_lock_and_a_new_session_moves_it(
        env, monkeypatch):
    from openflo import session_lock as L
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    other = str(tmp / 'proj' / 'other.flowsession')
    _answer(monkeypatch, asksaveasfilename=other)
    ed._save_session_as()
    assert L.read_lock(path) is None                # let go of the old one
    assert L.read_lock(other)['token'] == ed._session_lock_token()
    ed._release_session_lock()
    assert L.read_lock(other) is None


@pytest.mark.parametrize('elsewhere', [False, True])
def test_save_and_close_asks_before_overwriting_a_save_made_elsewhere(
        env, monkeypatch, elsewhere):
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    if elsewhere:
        _saved_elsewhere(ed, path)
    asked = _choose(ed, 'cancel')
    plans = []
    ed._start_close_save = lambda p: plans.append(p)
    ed._on_close(ask=False)
    if elsewhere:
        assert len(asked) == 1 and not plans        # Cancel keeps it open
        assert not getattr(ed, '_closing', False)
        assert 'saved elsewhere' in _gate_names(path)
    else:
        assert not asked
        assert [p['path'] for p in plans[0]] == [
            path, ed._session_autosave_path()]


def test_save_as_onto_a_session_open_elsewhere_asks(env, monkeypatch):
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    _held_elsewhere(path)
    (tmp / 'two').mkdir()
    ed2 = _editor(root, tmp / 'two')
    with open(path, 'rb') as f:
        before = f.read()
    _answer(monkeypatch, asksaveasfilename=path)
    asked = _choose(ed2, 'cancel')
    ed2._save_session()
    assert [a[2] for a in asked] == [['overwrite', 'cancel']]
    with open(path, 'rb') as f:
        assert f.read() == before
    _fake_lock(path, host=_host(), pid=os.getpid(), token='window-1')
    ed2._save_session()                     # a window of this process
    assert len(asked) == 1


def _host():
    from openflo import session_lock as L
    return L._host()


def test_a_second_window_here_opens_without_asking(env, monkeypatch):
    from openflo import session_lock as L
    root, tmp = env
    ed, path = _saved_session(root, tmp, monkeypatch)
    ed2, asked, opened = _open_second(root, path, 'cancel')
    assert not asked and len(opened) == 1
    assert L.read_lock(path)['token'] == ed._session_lock_token()
