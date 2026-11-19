"""Closing a big session must not freeze OpenFlo.

The close-time autosave rewrote every sample's processed-data sidecar on the
Tk thread: with large samples that took about a minute, every window went
"Not Responding" with no message, and each further click on the close button
queued another full save. Now the sidecars are only rewritten when their data
changed, and the close writes on a worker thread behind a progress window.
"""
import os
import threading
import time
import types

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tests._tk_wait import pump_until, settle, wait_until
from tests.conftest import editor_or_skip
from tests.test_session import V, _session_stub

COLS = ('FSC-A', 'SSC-A', 'FL1-A')


# --------------------------------------------------------------------------
# Unchanged sidecars are not rewritten
# --------------------------------------------------------------------------

def _stub_with(tmp_path, df):
    stub = _session_stub()
    s = types.SimpleNamespace(path='/data/a.csv', data=df)
    names = ['A']
    stub._samples = {'A': s}
    stub._sample_order = names
    stub._sample_colors = {'A': '#1f77b4'}
    stub._sample_plot_enabled = {'A': True}
    stub._sample_trial = {'A': 'Day 1'}
    stub._sample_is_comp = {}
    stub._sample_gates = {'A': {}}
    stub._sample_gate_order = {'A': []}
    stub._active_sample = 'A'
    stub._session_state = lambda: V._session_state(stub)
    stub._has_computed_columns = V._has_computed_columns
    stub._sidecar_safe_name = V._sidecar_safe_name
    return stub, s


def _count_writes(monkeypatch):
    """Record each sidecar the writer ENCODES (binary or CSV) and each one it
    copies from a file already holding the same data."""
    from openflo import editor_session, session_sidecar
    calls = {'encode': [], 'copy': []}
    real_bin = session_sidecar.write_frame
    real_csv = pd.DataFrame.to_csv
    real_copy = editor_session._copy_atomic

    def counting_bin(df, path):
        calls['encode'].append(path)
        return real_bin(df, path)

    def counting_csv(self, *a, **k):
        calls['encode'].append(a[0] if a else k.get('path_or_buf'))
        return real_csv(self, *a, **k)

    def counting_copy(src, dst):
        calls['copy'].append(dst)
        return real_copy(src, dst)
    monkeypatch.setattr(session_sidecar, 'write_frame', counting_bin)
    monkeypatch.setattr(pd.DataFrame, 'to_csv', counting_csv)
    monkeypatch.setattr(editor_session, '_copy_atomic', counting_copy)
    return calls


def _frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.random(50) for c in COLS})


def _read(path):
    from openflo.session_sidecar import read_sidecar
    return read_sidecar(str(path))


def test_unchanged_sidecar_is_not_rewritten(tmp_path, monkeypatch):
    stub, _s = _stub_with(tmp_path, _frame())
    calls = _count_writes(monkeypatch)
    path = str(tmp_path / 'sess.flowsession')
    first = V._write_session(stub, path)
    second = V._write_session(stub, path)
    assert len(calls['encode']) == 1, 'an unchanged sample was written again'
    assert not calls['copy']
    # The entry still points at the sidecar, so the reload uses it.
    assert second['samples'][0]['processed_data'] == \
        first['samples'][0]['processed_data']
    side = tmp_path / 'sess_data' / 'A.npz'
    pd.testing.assert_frame_equal(_read(side), _s.data)


@pytest.mark.parametrize('change', ['value', 'column', 'replaced'])
def test_changed_data_is_written_again(tmp_path, monkeypatch, change):
    stub, s = _stub_with(tmp_path, _frame())
    calls = _count_writes(monkeypatch)
    path = str(tmp_path / 'sess.flowsession')
    V._write_session(stub, path)
    if change == 'value':                  # one event edited in place
        s.data.iloc[7, 2] = 99.0
    elif change == 'column':               # a computed column added in place
        s.data['cluster'] = 0
    else:                                  # a new frame
        s.data = s.data.iloc[:10].copy()
    V._write_session(stub, path)
    assert len(calls['encode']) == 2
    back = _read(tmp_path / 'sess_data' / 'A.npz')
    assert back.shape == s.data.shape
    np.testing.assert_allclose(back['FL1-A'], s.data['FL1-A'])


def test_sidecar_missing_on_disk_is_written_again(tmp_path, monkeypatch):
    stub, _s = _stub_with(tmp_path, _frame())
    calls = _count_writes(monkeypatch)
    path = str(tmp_path / 'sess.flowsession')
    V._write_session(stub, path)
    os.remove(tmp_path / 'sess_data' / 'A.npz')
    V._write_session(stub, path)
    assert len(calls['encode']) == 2
    assert (tmp_path / 'sess_data' / 'A.npz').is_file()


def test_a_save_elsewhere_copies_the_sidecar_it_already_wrote(
        tmp_path, monkeypatch):
    """Save As to another file needs its own copy, unchanged data or not --
    copied from the first file rather than encoded again."""
    stub, s = _stub_with(tmp_path, _frame())
    calls = _count_writes(monkeypatch)
    V._write_session(stub, str(tmp_path / 'one.flowsession'))
    V._write_session(stub, str(tmp_path / 'two.flowsession'))
    assert len(calls['encode']) == 1
    assert len(calls['copy']) == 1
    pd.testing.assert_frame_equal(_read(tmp_path / 'two_data' / 'A.npz'),
                                  s.data)


def test_a_changed_source_file_is_not_copied(tmp_path, monkeypatch):
    """The copy is only taken from a file untouched since it was written:
    one edited on disk meanwhile is not trusted, the frame is encoded."""
    stub, _s = _stub_with(tmp_path, _frame())
    calls = _count_writes(monkeypatch)
    V._write_session(stub, str(tmp_path / 'one.flowsession'))
    src = tmp_path / 'one_data' / 'A.npz'
    st = os.stat(src)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    V._write_session(stub, str(tmp_path / 'two.flowsession'))
    assert len(calls['encode']) == 2 and not calls['copy']


def test_a_column_the_binary_format_cannot_hold_falls_back_to_csv(
        tmp_path, monkeypatch):
    df = _frame()
    df['flag'] = pd.array([1, None] * 25, dtype='Int64')   # nullable
    stub, _s = _stub_with(tmp_path, df)
    data = V._write_session(stub, str(tmp_path / 'sess.flowsession'))
    assert data['samples'][0]['processed_data'] == 'sess_data/A.csv'
    assert not (tmp_path / 'sess_data' / 'A.npz').exists()
    assert _read(tmp_path / 'sess_data' / 'A.csv').shape == (50, 4)


def test_an_older_csv_sidecar_is_replaced_by_the_binary_one(tmp_path):
    """Re-saving a session an older OpenFlo wrote converts its sidecar and
    leaves no stale CSV a later restore could pick up instead."""
    stub, _s = _stub_with(tmp_path, _frame())
    (tmp_path / 'sess_data').mkdir()
    old = tmp_path / 'sess_data' / 'A.csv'
    _frame().iloc[:5].to_csv(old, index=False)
    V._write_session(stub, str(tmp_path / 'sess.flowsession'))
    assert (tmp_path / 'sess_data' / 'A.npz').is_file()
    assert not old.exists()


def test_the_csv_a_sample_was_loaded_from_is_kept(tmp_path):
    """...unless that CSV is the file the sample itself was built from."""
    (tmp_path / 'sess_data').mkdir()
    old = tmp_path / 'sess_data' / 'A.csv'
    _frame().to_csv(old, index=False)
    stub, s = _stub_with(tmp_path, _frame())
    s.path = str(old)
    V._write_session(stub, str(tmp_path / 'sess.flowsession'))
    assert (tmp_path / 'sess_data' / 'A.npz').is_file()
    assert old.is_file()


def test_a_sidecar_loaded_at_restore_is_copied_not_encoded(
        tmp_path, monkeypatch):
    """Opening a session and then autosaving (to another file) used to
    re-encode every sample once. A sidecar the restore read is remembered
    with a hash of the data it gave, so the save copies that file."""
    from openflo.editor_session import frame_digest
    from openflo.session_sidecar import read_sidecar, write_frame
    (tmp_path / 'opened_data').mkdir()
    src = tmp_path / 'opened_data' / 'A.npz'
    write_frame(_frame(), str(src))
    df = read_sidecar(str(src))
    stub, _s = _stub_with(tmp_path, df)
    V._remember_sidecar(stub, str(src), frame_digest(df))
    calls = _count_writes(monkeypatch)
    V._write_session(stub, str(tmp_path / 'autosave.flowsession'))
    assert not calls['encode']
    assert len(calls['copy']) == 1
    pd.testing.assert_frame_equal(
        _read(tmp_path / 'autosave_data' / 'A.npz'), df)


def test_the_binary_sidecar_keeps_every_column_exactly(tmp_path):
    from openflo.session_sidecar import read_sidecar, write_frame
    df = pd.DataFrame({
        'FL1-A': np.linspace(-1, 1, 6, dtype='float32'),
        'FL2-A': [np.nan, 1e-300, 1e300, -0.0, 2.5, np.inf],
        'cluster': np.arange(6), 'FL1-A_pos': [True, False] * 3,
        'cell_cycle': pd.Series(['G1', 'S', 'G2', '', 'G1', 'S'],
                                dtype=object),
        'phase': ['a', 'b', 'c', 'd', 'e', 'f']})
    path = str(tmp_path / 'x.npz')
    assert write_frame(df, path)
    pd.testing.assert_frame_equal(read_sidecar(path), df)


def test_a_crafted_sidecar_is_not_unpickled(tmp_path):
    from openflo.session_sidecar import read_sidecar
    path = tmp_path / 'x.npz'
    np.savez(path, __columns__=np.array(['a']),
             c0=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        read_sidecar(str(path))


# --------------------------------------------------------------------------
# The close keeps the windows responsive and runs one save
# --------------------------------------------------------------------------

def _editor_or_skip():
    root, ed = editor_or_skip()
    df = pd.DataFrame({c: np.linspace(0, 1, 20) for c in COLS})
    ed._samples['s1'] = types.SimpleNamespace(
        name='s1', path='/e/s1.csv', data=df, fluor_channels=['FL1-A'],
        channel_labels={c: c for c in COLS})
    ed._sample_order.append('s1')
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_gates.setdefault('s1', {})
    return root, ed


def test_close_saves_off_the_tk_thread_and_ignores_repeat_clicks(
        monkeypatch, tmp_path):
    root, ed = _editor_or_skip()
    try:
        release = threading.Event()
        started = []

        def slow_write(self, plan, progress=None):
            started.append(threading.current_thread().name)
            progress(0, 2, 's1')
            release.wait(10)
            return {}, []
        monkeypatch.setattr(type(ed), '_run_session_write', slow_write)
        monkeypatch.setattr(type(ed), '_save_geometry', lambda self: None)
        monkeypatch.setattr(type(ed), '_ask_close_choice',
                            lambda self: 'save')
        monkeypatch.setattr(type(ed), '_session_autosave_path',
                            lambda self: str(tmp_path / 'auto.flowsession'))
        closed = []
        ed.destroy = lambda: closed.append(True)

        ed._on_close()
        # The Tk loop keeps running while the save is in flight: the window
        # answers, shows progress, and is not closed yet.
        pump_until(root, lambda: bool(started), 5)
        assert started and started[0] != threading.current_thread().name, (
            'the autosave ran on the Tk thread, which freezes every window')
        settle(root, 0.2)
        msg = ed._close_progress_ui[0].get()
        assert msg == 'Saving session… 1 of 1 samples', msg
        assert not closed

        # More clicks on the close button while it saves start nothing new.
        ed._on_close()
        ed._on_close()
        release.set()
        pump_until(root, lambda: bool(closed), 5)
        assert closed == [True]
        assert len(started) == 1, 'a repeat click queued another full save'
        assert ed._close_progress is None
    finally:
        release.set()
        root.destroy()


@pytest.mark.parametrize('closing', [True, False])
def test_periodic_autosave_stands_down_while_closing(monkeypatch, closing):
    root, ed = _editor_or_skip()
    try:
        writes = []
        for fn in ('_write_session', '_start_autosave', '_run_session_write'):
            monkeypatch.setattr(type(ed), fn,
                                lambda self, *a, _fn=fn: writes.append(_fn))
        ed._primary = True
        ed._closing = closing
        ed._periodic_autosave()
        # Closing: nothing (the close writes the same file). Otherwise the
        # tick does start an autosave, so the closing case isn't vacuous.
        assert writes == ([] if closing else ['_start_autosave'])
    finally:
        ed._primary = False
        root.destroy()


# --------------------------------------------------------------------------
# The periodic autosave writes on a worker too
# --------------------------------------------------------------------------

class _WatchedLock:
    """Wraps the session writers' lock and notes when a writer starts
    waiting for it and when it gets it."""

    def __init__(self, real):
        self.real = real
        self.events = []

    def __enter__(self):
        self.events.append('waiting')
        self.real.acquire()
        self.events.append('acquired')
        return self

    def __exit__(self, *exc):
        self.real.release()


def test_periodic_autosave_writes_off_the_window_thread(tmp_path, monkeypatch):
    """The 5-minute autosave froze the windows while it wrote, as the close
    did. It now snapshots on the Tk thread and writes on a worker; a tick
    that finds the last autosave still writing skips its turn, and any other
    save (the close, Save) waits for it rather than writing the same files
    at once."""
    root, ed = _editor_or_skip()
    release = threading.Event()
    started = []
    real = type(ed)._run_session_write_locked

    def slow(self, plan, progress=None):
        started.append(threading.current_thread().name)
        release.wait(30)
        return real(self, plan, progress)
    monkeypatch.setattr(type(ed), '_run_session_write_locked', slow)
    target = str(tmp_path / 'auto.flowsession')
    monkeypatch.setattr(type(ed), '_session_autosave_path',
                        lambda self: target)
    try:
        ed._primary = True
        t0 = time.monotonic()
        ed._periodic_autosave()
        assert time.monotonic() - t0 < 2, 'the autosave blocked the Tk thread'
        assert wait_until(lambda: started == ['openflo-autosave'], pause=0.02)
        ed._periodic_autosave()                 # still writing: no second
        assert len(started) == 1

        other = {}
        plan = ed._plan_session_write(str(tmp_path / 'close.flowsession'))
        # Watch the writers' lock rather than sleeping and checking nothing
        # happened: the close must be seen queued on the lock the autosave
        # holds, so a slow runner can't pass this without proving the wait.
        lock = _WatchedLock(ed.__dict__['_session_write_lock'])
        ed.__dict__['_session_write_lock'] = lock

        def close_save():                       # as the close's worker does
            other['result'] = ed._run_session_write(plan)
        t = threading.Thread(target=close_save)
        t.start()
        assert wait_until(lambda: lock.events or len(started) > 1,
                          pause=0.005)
        assert len(started) == 1, 'a second save ran beside the autosave'
        assert lock.events == ['waiting'] and lock.real.locked(), \
            'the close did not wait for the autosave'
        release.set()
        ed._autosave_thread.join(30)
        t.join(30)
        assert len(started) == 2 and 'result' in other
        assert lock.events == ['waiting', 'acquired']
        assert os.path.isfile(target)
        assert (tmp_path / 'auto_data' / 's1.npz').is_file()
    finally:
        release.set()
        ed._primary = False
        root.destroy()


def test_a_restored_sidecar_whose_data_then_changed_is_encoded(
        tmp_path, monkeypatch):
    """The negative of the copy above: once the sample's data differs from
    what the restore read, the remembered file no longer matches and the
    save encodes the current data instead of copying stale events."""
    from openflo.editor_session import frame_digest
    from openflo.session_sidecar import read_sidecar, write_frame
    (tmp_path / 'opened_data').mkdir()
    src = tmp_path / 'opened_data' / 'A.npz'
    write_frame(_frame(), str(src))
    df = read_sidecar(str(src))
    stub, s = _stub_with(tmp_path, df)
    V._remember_sidecar(stub, str(src), frame_digest(df))
    s.data['cluster'] = 3                    # an analysis after reopening
    calls = _count_writes(monkeypatch)
    V._write_session(stub, str(tmp_path / 'autosave.flowsession'))
    assert len(calls['encode']) == 1 and not calls['copy']
    back = _read(tmp_path / 'autosave_data' / 'A.npz')
    assert (back['cluster'] == 3).all()


# --------------------------------------------------------------------------
# Repeated periodic autosave failures reach the user, once per streak
# --------------------------------------------------------------------------

def test_repeated_autosave_failures_warn_once_per_streak(tmp_path, monkeypatch):
    """A failing 5-minute autosave used to be printed to the log and nothing
    else, so a full disk or a locked folder could leave hours of work
    unprotected without the user knowing. Each failure now shows in the
    status line; the second failure in a row raises one warning, later
    failures in the same streak raise none, a working autosave ends the
    streak, and a new streak warns again. Both ways an autosave can fail
    are counted: on the worker (the write) and on the Tk thread (the plan)."""
    from openflo import tk_threads
    root, ed = _editor_or_skip()
    warnings_shown = []
    monkeypatch.setattr(
        'tkinter.messagebox.showwarning',
        lambda title, message, **k: warnings_shown.append((title, message)))
    monkeypatch.setattr(type(ed), '_session_autosave_path',
                        lambda self: str(tmp_path / 'auto.flowsession'))
    outcome = {'fail': True}

    def write(self, plan, progress=None):
        if outcome['fail']:
            raise OSError('No space left on device')
        return {}, []
    monkeypatch.setattr(type(ed), '_run_session_write', write)

    def tick():
        ed._periodic_autosave()
        ed._autosave_thread.join(10)
        tk_threads.drain()

    try:
        ed._primary = True
        tick()
        assert warnings_shown == [], 'one failure interrupted the user'
        assert ed.status_var.get() == (
            'Autosave failed (1 in a row): No space left on device')

        tick()
        assert len(warnings_shown) == 1, 'two failures in a row went unseen'
        title, message = warnings_shown[0]
        assert title == 'Autosave is failing'
        assert 'No space left on device' in message
        assert 'last 2 tries' in message and '10 minutes' in message

        tick()
        tick()
        assert len(warnings_shown) == 1, 'the warning repeated every tick'
        assert ed.status_var.get().startswith('Autosave failed (4 in a row)')

        outcome['fail'] = False
        tick()
        assert ed.status_var.get() == 'Autosave is working again.'
        assert len(warnings_shown) == 1

        # A new streak, this time failing while planning on the Tk thread.
        def bad_plan(self, path):
            raise PermissionError('folder is read-only')
        monkeypatch.setattr(type(ed), '_plan_session_write', bad_plan)
        ed._periodic_autosave()
        assert len(warnings_shown) == 1
        ed._periodic_autosave()
        assert len(warnings_shown) == 2, 'a new streak did not warn again'
        assert 'folder is read-only' in warnings_shown[1][1]
    finally:
        ed._primary = False
        root.destroy()


def test_autosave_failure_warning_stands_down_while_closing(monkeypatch):
    """The close reports its own save; a streak that reaches the warning
    as the window closes must not stack a second dialog over it."""
    root, ed = _editor_or_skip()
    monkeypatch.setattr(
        'tkinter.messagebox.showwarning',
        lambda *a, **k: pytest.fail('warned during the close'))
    try:
        ed._autosave_finished('disk full')
        ed._closing = True
        ed._autosave_finished('disk full')
        assert ed._autosave_failures == 2
    finally:
        root.destroy()


def test_autosave_failure_from_sidecars_counts_toward_the_streak(tmp_path, monkeypatch):
    """An autosave that writes the session file but not a sample's computed
    results has still lost work: it counts as a failure, names the sample,
    and two in a row warn like any other failure."""
    from openflo import tk_threads
    root, ed = _editor_or_skip()
    warnings_shown = []
    monkeypatch.setattr(
        'tkinter.messagebox.showwarning',
        lambda title, message, **k: warnings_shown.append(message))
    monkeypatch.setattr(type(ed), '_session_autosave_path',
                        lambda self: str(tmp_path / 'auto.flowsession'))
    monkeypatch.setattr(type(ed), '_run_session_write',
                        lambda self, plan, progress=None: ({}, ['A']))
    try:
        ed._primary = True
        for _ in range(2):
            ed._periodic_autosave()
            ed._autosave_thread.join(10)
            tk_threads.drain()
        assert ed._autosave_failures == 2
        assert ed.status_var.get() == (
            'Autosave failed (2 in a row): computed results could not be '
            'written for A')
        assert len(warnings_shown) == 1
        assert 'computed results could not be written for A' in \
            warnings_shown[0]
    finally:
        ed._primary = False
        root.destroy()
