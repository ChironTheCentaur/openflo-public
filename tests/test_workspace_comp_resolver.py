"""Which compensation a Pipeline Workspace run uses (scope audit L3, part a).

Comp beads and matrices set on a row or a group were shown in the tree but
no run read them, and nothing blocked a run whose file would be reloaded
and clustered uncompensated. ``resolve_comp`` now names the compensation a
run uses: the row's own, else its group's, else the sample's matrix in the
editor now (an adjustment made after the drop counts), else the matrix
saved at drop. The tree labels come from it, and Run refuses, before any
work, a member with no compensation that is not loaded in the editor. A
loaded sample with none runs as the editor shows it, and the run says so.
Synthetic samples and generic channels only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from openflo import workspace_run as R
from openflo.workspace_run import WorkspaceModel
from tests.conftest import gui_unavailable

CH = ['FL1-A', 'FL2-A']
SAVED = np.array([[1.0, 0.1], [0.0, 1.0]])
EDITED = np.array([[1.0, 0.2], [0.0, 1.0]])


def _pl(name, matrix=None):
    return {'sample': name, 'trial': 'T', 'gate_id': None, 'color': '#000000',
            'comp_matrix': matrix, 'comp_channels': list(CH) if matrix is not None
            else []}


def _editor(**samples):
    """An editor holding ``samples``: name -> (comp_matrix or None)."""
    s = {n: SimpleNamespace(name=n, path=f'{n}.fcs', comp_matrix=m,
                            comp_channels=list(CH) if m is not None else [])
         for n, m in samples.items()}
    return SimpleNamespace(
        _samples=s, _sample_order=list(s), _sample_trial=dict.fromkeys(s, 'T'),
        _sample_colors=dict.fromkeys(s, '#1f77b4'),
        _sample_gates={n: {} for n in s}, _sample_gate_order={n: [] for n in s},
        _channel_labels={})


# ── the order ───────────────────────────────────────────────────────────────

def test_the_rows_own_comp_comes_first():
    m = WorkspaceModel()
    g = m.new_group()
    mid = m.add_item(_pl('s1', SAVED), gid=g)
    m.set_group_comp(g, {'kind': 'matrix', 'channels': CH,
                         'matrix': EDITED.tolist()})
    m.set_item_comp(mid, {'kind': 'beads', 'files': ['b1.fcs', 'b2.fcs']})
    c = R.resolve_comp(m, mid, _editor(s1=EDITED))
    assert (c['source'], c['kind'], c['files']) == (
        'row', 'beads', ['b1.fcs', 'b2.fcs'])
    assert c['label'] == 'beads:2 (row)' and c['ready']


def test_else_the_groups_comp():
    m = WorkspaceModel()
    g = m.new_group()
    mid = m.add_item(_pl('s1', SAVED), gid=g)
    m.set_group_comp(g, {'kind': 'matrix', 'channels': CH,
                         'matrix': EDITED.tolist()})
    c = R.resolve_comp(m, mid, _editor(s1=SAVED))
    assert (c['source'], c['kind']) == ('group', 'matrix')
    assert c['matrix'] == EDITED.tolist() and c['channels'] == CH
    assert c['label'] == 'matrix (grp)'


def test_else_the_matrix_in_the_editor_now():
    """An adjustment made in the editor after the drop is what runs."""
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', SAVED))
    c = R.resolve_comp(m, mid, _editor(s1=EDITED))
    assert c['source'] == 'editor' and c['live']
    np.testing.assert_array_equal(c['matrix'], EDITED)
    assert c['label'] == '2×2 (editor, changed)'
    same = R.resolve_comp(m, mid, _editor(s1=SAVED.copy()))
    assert same['label'] == '2×2 (editor)'


def test_else_the_matrix_saved_at_drop():
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', SAVED))
    for ed in (None, _editor(other=EDITED)):     # no editor; not loaded
        c = R.resolve_comp(m, mid, ed)
        assert c['source'] == 'saved' and not c['live']
        np.testing.assert_array_equal(c['matrix'], SAVED)
        assert c['label'] == '2×2' and c['ready']
        assert c['channels'] == CH


def test_a_loaded_sample_with_no_comp_is_live_but_not_ready():
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', None))
    c = R.resolve_comp(m, mid, _editor(s1=None))
    assert (c['source'], c['ready'], c['live']) == ('none', False, True)
    assert c['label'] == '⚠ none (editor)'


def test_nothing_at_all():
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', None))
    c = R.resolve_comp(m, mid, None)
    assert (c['source'], c['ready'], c['live'], c['label']) == (
        'none', False, False, '⚠')
    assert R.resolve_comp(m, 'no-such', None)['ready'] is False


def test_effective_comp_is_the_resolver():
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', SAVED))
    ed = _editor(s1=EDITED)
    assert m.effective_comp(mid, ed) == R.resolve_comp(m, mid, ed)


# ── what a run refuses ──────────────────────────────────────────────────────

def test_only_a_reload_with_no_comp_blocks_a_run():
    m = WorkspaceModel()
    gone = m.add_item(_pl('gone', None))         # not loaded, no matrix
    bare = m.add_item(_pl('bare', None))         # loaded, no matrix
    ok = m.add_item(_pl('ok', SAVED))
    beads = m.add_item(_pl('beads', None))
    m.set_item_comp(beads, {'kind': 'beads', 'files': ['b.fcs']})
    ed = _editor(bare=None, ok=SAVED)
    assert m.run_blockers(ed) == [gone]
    assert m.uncompensated_live(ed) == [bare]
    assert m.run_blockers(ed, [ok, beads, bare]) == []


# ── the window ──────────────────────────────────────────────────────────────

@pytest.fixture
def root():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        r = tk.Tk()
    except Exception as e:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except Exception:  # noqa: BLE001 - teardown; the root may already be gone
        pass


def _panel(root, editor):
    from openflo import workspace as W
    return W.WorkspacePanel(root, editor=editor)


def test_run_refuses_a_reload_with_no_comp_before_any_work(
        root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = _panel(root, _editor(s1=SAVED))
    panel.model.add_item(_pl('s1', SAVED))
    panel.model.add_item(_pl('gone', None))
    panel._run()
    msg = panel.status_var.get()
    assert msg.startswith('Not run: 1 item(s) have no compensation'), msg
    assert 'T / gone' in msg
    assert not panel._jobs
    assert not (tmp_path / 'outputs').exists()


def test_a_loaded_sample_with_no_comp_runs_and_is_named(
        root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = _panel(root, _editor(bare=None))
    panel.model.add_item(_pl('bare', None))
    monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
    panel._run()
    assert len(panel._jobs) == 1
    assert any('run uncompensated' in w and 'T / bare' in w
               for w in panel._warnings), panel._warnings


def test_the_tree_label_comes_from_the_resolver(root):
    v = _panel(root, _editor(s1=EDITED))._view
    mid = v.model.add_item(_pl('s1', SAVED))
    v._render()
    assert v.tree.set(mid, 'comp') == (
        '2×2 (editor, changed)')
