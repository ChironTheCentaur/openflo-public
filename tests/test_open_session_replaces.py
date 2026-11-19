"""File > Open session opens the session in place of the one this window
holds, after offering to save it (scope audit W1).

Opening a session used to be applied on top of the current one: the old
samples, gates, trials, tabs, layout, channel settings and samples kept
offline all stayed, the save target switched to the opened file and the
unsaved mark was cleared, so Ctrl+S (or Save and close) wrote the old
session's samples and gates into the other session's file.

Now the user is offered to save first (Save / Don't save / Cancel), the old
session is closed only once the new file has read, and the result is the
same as opening the file in a new window. Loads of the old session still
queued or in flight land nowhere.

Generated FCS files, synthetic sample names and generic channels only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests._fcs_writer import write_float_fcs
from tests._tk_wait import pump_until, settle
from tests.conftest import gui_unavailable
from tests.test_transform_records import _new_editor, capture_warnings, isolate_home

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


def _fcs(path, seed):
    rng = np.random.default_rng(seed)
    ev = np.column_stack([rng.lognormal(10, .2, 300), rng.lognormal(9, .2, 300),
                          rng.lognormal(6, .5, 300), rng.lognormal(6, .5, 300)])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_float_fcs(path, ev, CH)
    return str(path)


def _thr(name, value):
    return {'kind': 'threshold', 'channel': 'FSC-A', 'op': '>',
            'value': float(value), 'name': name}


class _Asked:
    """Stands in for the editor's question dialog: records each question
    and gives `answer`."""

    def __init__(self, answer):
        self.answer = answer
        self.questions = []

    def __call__(self, title, text, choices, default=0, tips=None):
        self.questions.append((title, text, [v for _l, v in choices],
                               default, tips))
        return self.answer


def _open(root, ed, path, answer='discard'):
    """Open `path` in `ed` as File > Open session does, answering the save
    question with `answer`, and land every sample it queues."""
    asked = _Asked(answer)
    ed._ask_choice = asked
    jobs = []
    ed._queue_fcs_loads = lambda paths, front_names=(): jobs.extend(paths)
    ed._queue_processed_loads = lambda items, front_names=(): None
    ed._load_session_path(str(path))
    for p in jobs:
        ed._load_worker(ed._sample_name_for(p), p)
    names = {ed._sample_name_for(p) for p in jobs}
    if names:
        assert pump_until(root, lambda: names <= set(ed._samples),
                          timeout=10, pause=0)
        ed._on_load_settled()
        ed._apply_session_restore()
    settle(root, 0.3)
    return asked


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
    try:
        yield root, tmp_path
    finally:
        root.destroy()


def _build_session(root, tmp, stem, samples, extra=None):
    """Write `<stem>.flowsession` holding `samples` ((name, seed) pairs),
    each with one threshold gate; `extra(ed)` adds more before the save."""
    from openflo.pipeline import FlowSample
    ed = _new_editor(root)
    for name, seed in samples:
        p = _fcs(tmp / stem / f'{name}.fcs', seed)
        s = FlowSample(p)
        s.apply_transform()
        ed._on_loaded(ed._sample_name_for(p), s)
    settle(root, 0.3)
    for name, seed in samples:
        ed._set_active_sample(name)
        ed._add_gate(_thr(f'{name} cut', 20_000 + seed), audit=False)
    if extra is not None:
        extra(ed)
    path = tmp / f'{stem}.flowsession'
    ed._write_session(str(path))
    ed.destroy()
    return path


def _old_extras(ed):
    """What the old session holds besides samples and gates: channel
    settings, cluster labels, a second tab, a layout."""
    from openflo.layout_page import LayoutDoc
    ed._channel_scale['FL1-A'] = 'log'
    ed._channel_scale['SSC-A'] = 'log'
    ed._channel_range['FL2-A'] = (1.0, 5000.0)
    ed._channel_labels['FL1-A'] = 'CD901'
    ed._cluster_labels['old_1'] = {0: 'blob'}
    ed._tabs().new_tab('Old tab', members={'old_1': None})
    ed._layout_doc = LayoutDoc()


def _with_offline(path, tmp):
    """Add a sample whose file is missing to the session at `path`."""
    data = json.loads(path.read_text())
    data['samples'].append({'name': 'old_gone',
                            'path': str(tmp / 'nowhere' / 'old_gone.fcs')})
    data.setdefault('sample_gates', {})['old_gone'] = [
        dict(_thr('gone cut', 1), id='g1')]
    data.setdefault('cluster_labels', {})['old_gone'] = {'1': 'gone blob'}
    path.write_text(json.dumps(data))


@pytest.fixture
def two(env):
    """An editor holding the old session (two samples with gates, settings,
    a tab, a layout, an offline sample, and one unsaved gate), and the path
    of a new session with one sample of its own."""
    root, tmp = env
    old = _build_session(root, tmp, 'old', [('old_1', 1), ('old_2', 2)],
                         _old_extras)
    _with_offline(old, tmp)
    new = _build_session(root, tmp, 'new', [('new_1', 7)])
    ed = _new_editor(root)
    # The offline sample makes the open offer, 0.8 s later, to look for its
    # file in a real dialog; a test slower than that (a loaded machine) would
    # show it. These tests are about the save question, not that offer.
    ed._offer_locate_missing = lambda: None
    asked = _open(root, ed, old)
    assert asked.questions == []                # an empty window: no question
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert 'old_gone' in ed._offline_samples
    ed._set_active_sample('old_1')
    ed._add_gate(dict(_thr('unsaved cut', 30_000), channel='SSC-A'),
                 audit=False)
    assert ed._unsaved
    yield root, tmp, ed, old, new
    ed.destroy()


def _comparable(state):
    """A saved session minus what differs between two opens of one file
    anyway (when it was written, by which build, the audit trail)."""
    out = dict(state)
    for k in ('created', 'app_version', 'audit'):
        out.pop(k, None)
    return out


def test_open_session_replaces_the_current_one(two):
    root, _tmp, ed, old, new = two
    asked = _open(root, ed, new, 'discard')
    assert [q[2] for q in asked.questions] == [['save', 'discard', 'cancel']]
    assert asked.questions[0][3] == 0           # Save is the default button
    assert sorted(ed._samples) == ['new_1']
    assert set(ed._sample_gates) == {'new_1'}
    assert not ed.__dict__.get('_offline_samples')
    assert ed._trial_order == [ed._sample_trial['new_1']]
    assert ed._tabs().names() == ['All']
    assert ed._layout_doc is None
    assert 'FL1-A' not in ed._channel_scale
    assert 'FL2-A' not in ed._channel_range
    assert ed._channel_labels.get('FL1-A') != 'CD901'
    assert 'old_1' not in ed._cluster_labels
    assert os.path.basename(ed._session_file) == 'new.flowsession'
    # Save writes the new session only into its own file.
    assert ed._save_session() is True
    saved = json.loads(new.read_text())
    assert [s['name'] for s in saved['samples']] == ['new_1']
    assert set(saved['sample_gates']) == {'new_1'}
    # The old file still has its own samples (the unsaved gate is gone).
    kept = json.loads(old.read_text())
    assert sorted(s['name'] for s in kept['samples']) == [
        'old_1', 'old_2', 'old_gone']


def test_opening_over_a_session_saves_what_a_new_window_would(env, two):
    """The invariant: the session Save writes after opening `new` over the
    old one equals what it writes after opening `new` in a new window."""
    root, _tmp, ed, _old, new = two
    _open(root, ed, new, 'discard')
    fresh = _new_editor(root)
    try:
        _open(root, fresh, new)
        assert _comparable(ed._session_state()) == _comparable(
            fresh._session_state())
    finally:
        fresh.destroy()


def test_a_session_whose_files_are_all_missing_opens_alone(env, two,
                                                          monkeypatch):
    """With no sample to land, nothing resets the channels as a side
    effect: the old session's channels, labels and axes must still go."""
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: False)
    root, tmp, ed, _old, _new = two
    gone = tmp / 'gone.flowsession'
    gone.write_text(json.dumps({
        'format': 'openflo-session', 'version': 3,
        'samples': [{'name': 'gone_1',
                     'path': str(tmp / 'nowhere' / 'gone_1.fcs')}],
        'sample_gates': {'gone_1': [dict(_thr('g cut', 5), id='g1')]}}))
    _open(root, ed, gone, 'discard')
    assert ed._channels == []
    assert ed._channel_labels == {}
    assert (ed.x_combo.get(), ed.y_combo.get(), ed.color_combo.get()) == (
        '', '', '')
    fresh = _new_editor(root)
    try:
        _open(root, fresh, gone)
        assert _comparable(ed._session_state()) == _comparable(
            fresh._session_state())
    finally:
        fresh.destroy()


def test_a_session_of_missing_files_is_replaced_whole(env, monkeypatch):
    """The old session holds only samples whose files are missing (so
    removing loaded samples removes none): its trials and gates still go."""
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: False)
    root, tmp = env
    new = _build_session(root, tmp, 'new', [('new_1', 7)])
    gone = tmp / 'gone.flowsession'
    gone.write_text(json.dumps({
        'format': 'openflo-session', 'version': 3,
        'samples': [{'name': 'gone_1', 'trial': 'Gone trial',
                     'path': str(tmp / 'nowhere' / 'gone_1.fcs')}],
        'sample_gates': {'gone_1': [dict(_thr('g cut', 5), id='g1')]}}))
    ed = _new_editor(root)
    try:
        _open(root, ed, gone)
        ed._trial_order.append('Gone trial')    # a header the tree shows
        _open(root, ed, new, 'discard')    # saved in its own file: no question
        assert ed._trial_order == [ed._sample_trial['new_1']]
        assert not ed.__dict__.get('_offline_samples')
    finally:
        ed.destroy()


def test_cancel_keeps_the_current_session(two):
    root, _tmp, ed, _old, new = two
    before = _comparable(ed._session_state())
    asked = _open(root, ed, new, 'cancel')
    assert len(asked.questions) == 1
    assert _comparable(ed._session_state()) == before
    assert os.path.basename(ed._session_file) == 'old.flowsession'
    assert ed._unsaved


def test_save_writes_the_current_session_then_opens(two):
    root, _tmp, ed, old, new = two
    _open(root, ed, new, 'save')
    gates = json.loads(old.read_text())['sample_gates']['old_1']
    assert 'unsaved cut' in [g.get('name') for g in gates]
    assert sorted(ed._samples) == ['new_1']


def test_a_save_that_does_not_happen_stops_the_open(two):
    root, _tmp, ed, _old, new = two
    ed._save_session = lambda save_as=False: False
    _open(root, ed, new, 'save')
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert os.path.basename(ed._session_file) == 'old.flowsession'


def test_save_session_says_whether_it_saved(two, monkeypatch):
    from openflo import editor_session
    _root, tmp, ed, _old, _new = two
    ed._remember_session_file(None)
    monkeypatch.setattr(editor_session.filedialog, 'asksaveasfilename',
                        lambda **k: '')
    assert ed._save_session() is False
    out = tmp / 'named.flowsession'
    monkeypatch.setattr(editor_session.filedialog, 'asksaveasfilename',
                        lambda **k: str(out))
    assert ed._save_session() is True and out.exists()


def test_no_question_when_everything_is_saved_in_its_own_file(two):
    root, _tmp, ed, _old, new = two
    assert ed._save_session() is True
    asked = _open(root, ed, new, 'cancel')
    assert asked.questions == []
    assert sorted(ed._samples) == ['new_1']


def test_an_untitled_session_with_no_changes_is_still_asked_about(two):
    """A resumed autosave or untitled session has no file of its own, so
    opening another one asks even with no unsaved mark."""
    root, _tmp, ed, _old, new = two
    ed._remember_session_file(None)
    ed._note_saved()
    asked = _open(root, ed, new, 'cancel')
    assert len(asked.questions) == 1
    assert sorted(ed._samples) == ['old_1', 'old_2']


def test_a_file_that_will_not_open_leaves_the_session_as_it_was(two):
    root, tmp, ed, _old, _new = two
    bad = tmp / 'bad.flowsession'
    bad.write_text('{"format": "openflo-session", "samples": [')
    before = _comparable(ed._session_state())
    ed._report_unreadable_session = lambda path, exc: None
    _open(root, ed, bad, 'discard')
    assert _comparable(ed._session_state()) == before
    assert os.path.basename(ed._session_file) == 'old.flowsession'


def test_a_project_archive_asks_once(two, monkeypatch):
    from openflo.project_archive import write_project_archive
    root, tmp, ed, _old, new = two
    zpath = tmp / 'new.zip'
    write_project_archive(str(new), str(zpath))
    asked = _open(root, ed, zpath, 'discard')
    assert len(asked.questions) == 1
    assert sorted(ed._samples) == ['new_1']


def test_the_backup_offered_for_a_damaged_file_asks_once(two, monkeypatch):
    from openflo import editor_session
    root, tmp, ed, _old, new = two
    damaged = tmp / 'new2.flowsession'
    damaged.write_text('{"format": "openflo-session", "samples": [')
    backup = editor_session.session_backup_path(str(damaged))
    with open(backup, 'w', encoding='utf-8') as f:
        f.write(new.read_text())
    monkeypatch.setattr(editor_session.messagebox, 'askyesno',
                        lambda *a, **k: True)
    asked = _open(root, ed, damaged, 'discard')
    assert len(asked.questions) == 1
    assert sorted(ed._samples) == ['new_1']


def test_the_old_sessions_queued_loads_are_dropped(two, monkeypatch):
    root, tmp, ed, _old, new = two
    late = _fcs(tmp / 'late' / 'old_late.fcs', 9)
    monkeypatch.setattr(ed, '_ensure_load_pool', lambda: None)
    type(ed)._queue_fcs_loads(ed, [late])       # the real queueing
    assert ed._load_queue.qsize() == 1 and ed._loading
    _open(root, ed, new, 'discard')
    assert ed._load_queue.qsize() == 0
    assert not ed._loading
    assert (ed._load_total, ed._load_done) == (0, 0)


def _as_pool_worker(epoch, fn):
    """Run `fn` as a pool worker that took its job under session `epoch`
    (editor_loadpool._load_pool_worker records it per thread). Run on this
    thread: Tk here takes posts only from the thread running it."""
    from openflo import editor_loadpool
    job = getattr(editor_loadpool, '_JOB', None)    # absent before W1's fix
    if job is not None:
        job.epoch = epoch
    try:
        fn()
    finally:
        if job is not None:
            job.epoch = None


def _record_landings(ed):
    """Record every load result that reaches the window."""
    got = []
    ed._on_loaded = lambda name, s: got.append(('loaded', name))
    ed._on_load_error = lambda name, exc: got.append(('error', name))
    ed._mark_one_done = lambda: got.append(('tick', None))
    return got


def _late_jobs(ed, tmp):
    """Run, as pool workers, one job of each kind: a raw FCS, an FCS that
    fails, a processed CSV, a damaged CSV that falls back to its FCS, a
    CSV that fails, and the progress tick that follows every job."""
    import pandas as pd
    late = _fcs(tmp / 'late' / 'old_late.fcs', 9)
    csv = tmp / 'late' / 'old_csv.csv'
    rng = np.random.default_rng(5)
    pd.DataFrame(rng.lognormal(6, .5, (50, 4)), columns=CH).to_csv(
        csv, index=False)
    cut = tmp / 'late' / 'old_cut.csv'
    cut.write_text('')
    ed.stage_sample_prep(str(cut), source_fcs=late)
    ed._load_worker('old_late', late)
    ed._load_worker('old_bad', str(tmp / 'missing.fcs'))
    ed._load_csv_worker('old_csv', str(csv), {})
    ed._load_csv_worker('old_cut', str(cut), {})
    ed._load_csv_worker('old_nocsv', str(tmp / 'missing.csv'), {})
    post = getattr(ed, '_post_load_result', None)   # absent before W1's fix
    if post is not None:
        post(ed._mark_one_done)
    else:
        ed.after(0, ed._mark_one_done)


def test_a_load_in_flight_from_the_old_session_lands_nowhere(two):
    """A pool worker that took the old session's job before the new one
    opened posts its result after: it must not reach the new session."""
    root, tmp, ed, _old, new = two
    took = ed.__dict__.get('_load_epoch', 0)    # the job is taken
    _open(root, ed, new, 'discard')
    got = _record_landings(ed)
    _as_pool_worker(took, lambda: _late_jobs(ed, tmp))
    settle(root, 0.5)
    assert got == []
    assert sorted(ed._samples) == ['new_1']


def test_a_load_of_the_current_session_still_lands(two):
    """The guard drops only another session's loads."""
    root, tmp, ed, _old, new = two
    _open(root, ed, new, 'discard')
    got = _record_landings(ed)
    _as_pool_worker(ed._load_epoch, lambda: _late_jobs(ed, tmp))
    assert pump_until(root, lambda: len(got) == 6, timeout=10, pause=0)
    assert sorted(got, key=str) == sorted(
        [('loaded', 'old_late'), ('error', 'old_bad'), ('loaded', 'old_csv'),
         ('loaded', 'old_cut'), ('error', 'old_nocsv'), ('tick', None)],
        key=str)


def test_a_folder_watch_and_tool_windows_end_with_the_old_session(two):
    import tkinter as tk
    root, tmp, ed, _old, new = two
    ed._watch_dir = str(tmp)
    ed._watch_after = ed.after(60_000, lambda: None)
    win = tk.Toplevel(ed)
    ed.__dict__.setdefault('_tool_windows', {})['stats'] = win
    _open(root, ed, new, 'discard')
    assert ed._watch_dir is None and ed._watch_after is None
    assert not win.winfo_exists()
    assert not ed._tool_windows


def test_a_sample_move_under_way_stops_the_open(two):
    root, _tmp, ed, _old, new = two
    ed._inbound_move = {'id': 'x', 'pending': {'p'}, 'taken': []}
    asked = _open(root, ed, new, 'discard')
    assert asked.questions == []
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert 'move' in ed.status_var.get()


def test_a_pool_worker_records_the_session_its_job_belongs_to(env,
                                                             monkeypatch):
    """The real pool loop tags each job with the session epoch it was taken
    under, which _post_load_result checks when the result lands."""
    from openflo import editor_loadpool
    root, _tmp = env
    ed = _new_editor(root)
    try:
        seen = []
        monkeypatch.setattr(ed, '_lower_loader_priority', lambda: None)
        ed._load_worker = lambda name, path: seen.append(
            (name, editor_loadpool._JOB.epoch))
        ed._load_epoch = 5
        ed._enqueue_load(('job_1', 'unused.fcs'))
        ed._enqueue_load(None)                  # the shutdown sentinel
        ed._load_pool_worker()                  # runs the job, then stops
        assert seen == [('job_1', 5)]
    finally:
        editor_loadpool._JOB.epoch = None
        ed.destroy()


def test_an_old_job_taken_just_as_the_other_session_opens_lands_nowhere(two):
    """A pool worker takes the old session's job, and the other session
    opens before the worker looks at which session it is in: the job keeps
    the session it was queued under (stamped on the Tk thread), so its
    sample does not land in the new session. (Reviewer's probe P1.)"""
    import queue as _queue

    from openflo import editor_loadpool
    root, tmp, ed, _old, new = two
    late = _fcs(tmp / 'late' / 'old_late.fcs', 9)
    ed._load_queue = _queue.PriorityQueue()
    ed._enqueue_load(('old_late', late))
    ed._enqueue_load(None)                      # the shutdown sentinel
    real_get = ed._load_queue.get
    got = []

    def get_then_other_session_opens(*a, **k):
        item = real_get(*a, **k)
        if item[2] is not None:
            _open(root, ed, new, 'discard')
            ed._on_loaded = lambda name, s: got.append(('loaded', name))
            ed._mark_one_done = lambda: got.append(('tick', None))
        return item
    ed._load_queue.get = get_then_other_session_opens
    ed._lower_loader_priority = lambda: None
    try:
        ed._load_pool_worker()
    finally:
        editor_loadpool._JOB.epoch = None
    settle(root, 0.5)
    assert sorted(ed._samples) == ['new_1']
    assert got == []


def test_dropped_jobs_leave_no_session_stamp(two, monkeypatch):
    root, tmp, ed, _old, new = two
    monkeypatch.setattr(ed, '_ensure_load_pool', lambda: None)
    type(ed)._queue_fcs_loads(ed, [_fcs(tmp / 'late' / 'old_late.fcs', 9)])
    assert len(ed._load_job_epochs) == 1
    _open(root, ed, new, 'discard')
    assert ed._load_job_epochs == {}


def test_a_window_whose_samples_are_still_loading_is_asked_about(env,
                                                                 monkeypatch):
    """No sample has landed yet, the window is untitled: opening another
    session still asks, and Don't save drops those loads."""
    root, tmp = env
    new = _build_session(root, tmp, 'new', [('new_1', 7)])
    ed = _new_editor(root)
    try:
        monkeypatch.setattr(ed, '_ensure_load_pool', lambda: None)
        type(ed)._queue_fcs_loads(ed, [_fcs(tmp / 'busy' / 'busy_1.fcs', 4)])
        assert not ed._samples and ed._loading
        asked = _open(root, ed, new, 'cancel')
        assert len(asked.questions) == 1 and ed._loading
        _open(root, ed, new, 'discard')
        assert sorted(ed._samples) == ['new_1'] and not ed._loading
    finally:
        ed.destroy()


def test_an_untitled_window_of_missing_files_is_asked_about(env, monkeypatch):
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: False)
    root, tmp = env
    new = _build_session(root, tmp, 'new', [('new_1', 7)])
    gone = tmp / 'gone.flowsession'
    gone.write_text(json.dumps({
        'format': 'openflo-session', 'version': 3,
        'samples': [{'name': 'gone_1',
                     'path': str(tmp / 'nowhere' / 'gone_1.fcs')}],
        'sample_gates': {'gone_1': [dict(_thr('g cut', 5), id='g1')]}}))
    ed = _new_editor(root)
    try:
        _open(root, ed, gone)
        ed._remember_session_file(None)         # e.g. a resumed autosave
        assert not ed._samples and ed._offline_samples
        asked = _open(root, ed, new, 'cancel')
        assert len(asked.questions) == 1
        assert 'gone_1' in ed._offline_samples
    finally:
        ed.destroy()


def test_a_sample_move_out_under_way_stops_the_open(two):
    root, _tmp, ed, _old, new = two
    ed._pending_move = {'id': 'x', 'names': ['old_1'], 'clip': '{}'}
    asked = _open(root, ed, new, 'discard')
    assert asked.questions == []
    assert sorted(ed._samples) == ['old_1', 'old_2']
    assert 'move' in ed.status_var.get()


def test_an_open_history_window_shows_the_new_sessions_trail(two):
    root, _tmp, ed, _old, new = two
    ed._show_audit_window()
    win = ed._audit_window
    _open(root, ed, new, 'discard')
    assert win._log is ed._audit_log
    assert len(win._tv.get_children()) == len(ed._audit_log)


def test_a_damaged_session_is_refused_in_plain_words(two, monkeypatch):
    """Before: a raw TypeError from the gate restore, after the current
    session had been closed. Now the file is refused before anything
    closes, with a message that names the gate."""
    from openflo import editor_session
    root, tmp, ed, _old, new = two
    data = json.loads(new.read_text())
    data['sample_gates']['new_1'][0]['parent_id'] = ['x']
    damaged = tmp / 'damaged.flowsession'
    damaged.write_text(json.dumps(data))
    shown = []
    monkeypatch.setattr(editor_session.messagebox, 'showerror',
                        lambda title, text, **k: shown.append((title, text)))
    before = _comparable(ed._session_state())
    _open(root, ed, damaged, 'discard')
    assert len(shown) == 1
    title, text = shown[0]
    assert title == "Can't open session"
    assert "damaged and can't be opened" in text
    assert "gate 1 of sample 'new_1'" in text
    assert _comparable(ed._session_state()) == before


def test_a_gate_with_a_missing_bound_opens_unknown_and_is_not_drawn(two):
    """Before: drawing it raised KeyError on every redraw. It is kept (a
    save writes it back), draws nothing and its population is unknown."""
    from openflo.gate_eval import unknown_population_reason
    root, tmp, ed, _old, new = two
    data = json.loads(new.read_text())
    del data['sample_gates']['new_1'][0]['value']
    damaged = tmp / 'nobound.flowsession'
    damaged.write_text(json.dumps(data))
    _open(root, ed, damaged, 'discard')
    assert sorted(ed._samples) == ['new_1']
    ed._set_active_sample('new_1')
    (gid,) = [k for k, g in ed._gates.items() if g.get('name') == 'new_1 cut']
    ed._draw_gates('FSC-A', 'SSC-A')
    assert gid not in ed._vlines and gid not in ed._hlines
    assert 'value' in unknown_population_reason(ed._gates, gid)


@pytest.mark.parametrize('payload', ['[]', '"text"', '3'])
def test_a_file_that_is_not_a_record_is_not_a_session(two, monkeypatch,
                                                      payload):
    """Before: AttributeError on `.get` of a list, after the lock was
    taken. Now the 'Not a session file' message, nothing closed."""
    from openflo import editor_session
    root, tmp, ed, _old, _new = two
    odd = tmp / 'odd.flowsession'
    odd.write_text(payload)
    shown = []
    monkeypatch.setattr(editor_session.messagebox, 'showerror',
                        lambda title, text, **k: shown.append(title))
    before = _comparable(ed._session_state())
    _open(root, ed, odd, 'discard')
    assert shown == ['Not a session file']
    assert _comparable(ed._session_state()) == before


def test_a_damaged_tab_gate_is_refused_before_anything_closes(two, monkeypatch):
    """A tab's own gate with a list as its parent opened, then raised
    TypeError in a Tk callback while the gates were installed (#446
    review). Now the file is refused with the tab and gate named."""
    from openflo import editor_session
    root, tmp, ed, old, _new = two
    data = json.loads(old.read_text())
    tab = next(t for t in data['view_tabs']['tabs']
               if t.get('name') == 'Old tab')
    tab['sample_gates'] = {'old_1': [
        {'id': 'g1', 'parent_id': ['x'], 'kind': 'threshold',
         'channel': 'FSC-A', 'value': 1.0, 'name': 'tab cut'}]}
    damaged = tmp / 'damaged_tab.flowsession'
    damaged.write_text(json.dumps(data))
    shown = []
    monkeypatch.setattr(editor_session.messagebox, 'showerror',
                        lambda title, text, **k: shown.append(text))
    before = _comparable(ed._session_state())
    _open(root, ed, damaged, 'discard')
    assert len(shown) == 1
    assert "tab 'Old tab': gate 1 of sample 'old_1': 'parent_id'" in shown[0]
    assert _comparable(ed._session_state()) == before
