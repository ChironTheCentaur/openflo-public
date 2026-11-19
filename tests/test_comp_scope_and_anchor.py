"""Compensation window scope and anchor (scope audit W4, W5).

W4: "Apply to: this sample" flipped back to "this batch" whenever the matrix
was redrawn (Load…, Reset to identity, an Optimize result), so Apply then
recompensated every sample of the batch, with no undo. The batch also took
in samples the active tab hides.

W5: the window is non-modal and tied to the sample active when it opened,
but never named it, so Apply after picking another sample in the tray
recompensated the first one unseen; and once that sample was removed or
reloaded, Apply crashed or reached another sample of the same name.

All data is generated (synthetic FCS files with FL channels).
"""
from __future__ import annotations

import tkinter as tk

import numpy as np

from tests.conftest import gui_unavailable
from tests.test_comp_apply_to_batch import RUN, VOLTS, _write
from tests.test_compensation_editor_apply import (
    FL,
    M,
    _comp_window,
    _editor_or_skip,
    _load,
)

# ── the window alone ─────────────────────────────────────────────────────

def _window(batch, calls):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f'Tk cannot start: {exc}')
    root.withdraw()
    from openflo.ui_comp import CompensationEditorWindow
    win = CompensationEditorWindow(
        root, sample=None,
        on_apply=lambda ch, m, targets=None: calls.append(targets),
        batch=batch)
    return root, win


def _three(_channels):
    return ['FL-s1', 'FL-s2', 'FL-s3'], {}


def test_this_sample_stays_picked_through_load_reset_and_optimize():
    calls = []
    root, win = _window(_three, calls)
    try:
        win._set_matrix(FL, np.eye(3), 'file1')
        assert win.scope_var.get() == 'batch'          # the default
        win._sample_rb.invoke()                        # the user narrows
        win._set_matrix(FL, M, 'file2')                # Load… another
        assert win.scope_var.get() == 'sample'
        win._reset_identity()                          # Reset to identity
        assert win.scope_var.get() == 'sample'
        win._set_matrix(FL, M)                         # an Optimize result
        assert win.scope_var.get() == 'sample'
        win._apply()
        assert calls == [None]                         # one sample only
    finally:
        root.destroy()


def test_batch_is_the_default_until_the_user_chooses():
    calls = []
    root, win = _window(_three, calls)
    try:
        win._set_matrix(FL, np.eye(3), 'file1')
        win._set_matrix(FL, M, 'file2')
        assert win.scope_var.get() == 'batch'
        win._apply()
        assert calls == [['FL-s1', 'FL-s2', 'FL-s3']]
    finally:
        root.destroy()


def test_no_batch_means_this_sample_but_the_pick_comes_back():
    size = {'n': 3}

    def batch(_channels):
        return [f'FL-s{i}' for i in range(1, size['n'] + 1)], {}

    root, win = _window(batch, [])
    try:
        win._set_matrix(FL, np.eye(3))
        win._sample_rb.invoke()
        size['n'] = 1
        win._set_matrix(FL, M)
        assert win.scope_var.get() == 'sample'
        assert 'disabled' in win._batch_rb.state()
        size['n'] = 3
        win._set_matrix(FL, M)
        assert win.scope_var.get() == 'sample'         # still the user's
        win._batch_rb.invoke()                         # now the batch
        size['n'] = 1
        win._set_matrix(FL, M)
        assert win.scope_var.get() == 'sample'
        size['n'] = 3
        win._set_matrix(FL, M)
        assert win.scope_var.get() == 'batch'          # the user's again
    finally:
        root.destroy()


def test_apply_reaches_the_batch_as_it_is_at_apply():
    members = ['FL-s1', 'FL-s2', 'FL-s3']
    calls = []
    root, win = _window(lambda _ch: (list(members), {}), calls)
    try:
        win._set_matrix(FL, M)
        members.remove('FL-s3')               # e.g. a tab switch since
        win._apply()
        assert calls == [['FL-s1', 'FL-s2']]
        assert win._batch_var.get().startswith('2 samples')
        assert win.scope_var.get() == 'batch'
    finally:
        root.destroy()


# ── in the editor ────────────────────────────────────────────────────────

def _three_files(tmp_path, seed=7):
    rng = np.random.default_rng(seed)
    T = rng.uniform(100, 20000, (800, 3))
    # Each file's name differs from its sample's name in the session, so
    # the window must name the session's sample, not the file.
    for i, n in enumerate('abc', 1):
        _write(tmp_path / f'file_{n}.fcs', T, RUN, VOLTS, i)
    return [(n, tmp_path / f'file_{n}.fcs') for n in 'abc']


def _fl(ed, name):
    return ed._samples[name].data[FL].to_numpy(float).copy()


def test_the_batch_holds_only_what_the_active_tab_shows(tmp_path):
    root, ed = _editor_or_skip()
    try:
        _load(root, ed, _three_files(tmp_path))
        ts = ed._tabs()
        tab = ts.new_tab('FL tab', members={'a': None, 'b': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(tab)
        ed._set_active_sample('a')
        before_c = _fl(ed, 'c')
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        assert win._members == ['a', 'b']
        assert 'c: not shown in this tab' in win._batch_tip_text
        win._apply()
        assert ed._samples['b'].comp_channels == FL
        np.testing.assert_array_equal(_fl(ed, 'c'), before_c)
    finally:
        root.destroy()


def test_the_window_names_its_sample_and_applies_to_it(tmp_path):
    root, ed = _editor_or_skip()
    try:
        _load(root, ed, _three_files(tmp_path)[:2])
        ed._set_active_sample('a')
        win = _comp_window(ed)
        assert win.title() == 'Compensation matrix — a'
        assert win._sample_rb.cget('text') == 'this sample (a)'
        win._set_matrix(FL, M)
        win._sample_rb.invoke()
        before_b = _fl(ed, 'b')
        ed._set_active_sample('b')            # the tray moves on
        win._apply()
        assert ed._samples['a'].comp_channels == FL
        np.testing.assert_array_equal(_fl(ed, 'b'), before_b)
        assert ed.status_var.get().startswith("Applied 3×3 matrix to 'a'")
    finally:
        root.destroy()


def test_apply_refuses_once_its_sample_is_removed(tmp_path):
    root, ed = _editor_or_skip()
    try:
        _load(root, ed, _three_files(tmp_path)[:2])
        ed._set_active_sample('a')
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        assert win.scope_var.get() == 'batch'
        before_b = _fl(ed, 'b')
        ed._remove_samples(['a'])
        win._apply()                          # used to raise KeyError
        np.testing.assert_array_equal(_fl(ed, 'b'), before_b)
        assert ed._samples['b'].comp_channels != FL
        assert "NOT applied: 'a'" in ed.status_var.get()
        assert "NOT applied: 'a'" in win.status_var.get()
    finally:
        root.destroy()


def test_apply_refuses_once_its_sample_is_reloaded(tmp_path):
    root, ed = _editor_or_skip()
    try:
        files = _three_files(tmp_path)[:2]
        _load(root, ed, files)
        ed._set_active_sample('a')
        win = _comp_window(ed)
        win._set_matrix(FL, M)
        win._sample_rb.invoke()
        ed._remove_samples(['a'])
        _load(root, ed, files[:1])            # 'a' again, a new object
        before_a = _fl(ed, 'a')
        win._apply()
        np.testing.assert_array_equal(_fl(ed, 'a'), before_a)
        assert ed._samples['a'].comp_channels != FL
        assert "NOT applied: 'a'" in ed.status_var.get()
    finally:
        root.destroy()


def test_one_window_open_session_closes_it_another_sample_replaces_it(
        tmp_path):
    from openflo.ui_comp import CompensationEditorWindow

    def wins():
        return [w for w in ed.winfo_children()
                if isinstance(w, CompensationEditorWindow)
                and w.winfo_exists()]

    root, ed = _editor_or_skip()
    try:
        _load(root, ed, _three_files(tmp_path)[:2])
        ed._set_active_sample('a')
        first = _comp_window(ed)
        ed._open_comp_editor()                # same sample: raised, kept
        assert wins() == [first]
        ed._set_active_sample('b')
        ed._open_comp_editor()                # another sample: replaced
        now = wins()
        assert len(now) == 1 and now[0] is not first
        assert now[0].title() == 'Compensation matrix — b'
        ed._close_session_windows()           # File > Open session
        assert wins() == []
    finally:
        root.destroy()


def test_apply_never_widens_this_sample_to_a_batch_that_grew():
    members = ['FL-s1']
    calls = []
    root, win = _window(lambda _ch: (list(members), {}), calls)
    try:
        win._set_matrix(FL, M)
        assert win.scope_var.get() == 'sample'          # no batch yet
        members += ['FL-s2', 'FL-s3']                   # e.g. a tab switch
        win._apply()
        assert calls == [None]
        assert win.scope_var.get() == 'sample'
    finally:
        root.destroy()


def test_its_own_sample_stays_in_the_batch_when_the_tab_hides_it(tmp_path):
    root, ed = _editor_or_skip()
    try:
        _load(root, ed, _three_files(tmp_path))
        ed._set_active_sample('a')
        win = _comp_window(ed)                # opened on 'a' in All
        ts = ed._tabs()
        tab = ts.new_tab('FL tab', members={'b': None})
        ed._rebuild_tab_strip()
        ed._switch_tab(tab)
        before_c = _fl(ed, 'c')
        win._set_matrix(FL, M)
        assert win._members == ['a', 'b']
        win._apply()
        assert ed._samples['a'].comp_channels == FL
        assert ed._samples['b'].comp_channels == FL
        np.testing.assert_array_equal(_fl(ed, 'c'), before_c)
    finally:
        root.destroy()
