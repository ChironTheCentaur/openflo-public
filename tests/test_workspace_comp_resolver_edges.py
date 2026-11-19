"""More of which compensation a Pipeline Workspace run uses (scope audit L3a,
from its review): a comp applied in the editor after an uncompensated drop,
a bead override with no files, an override's live flag, a saved matrix read
back from JSON, and only the run's own members refusing it or named as
uncompensated. Synthetic samples and generic channels only."""
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


def _pl(name, matrix=None, **extra):
    d = {'sample': name, 'trial': 'T', 'gate_id': None, 'color': '#000000',
         'comp_matrix': matrix,
         'comp_channels': list(CH) if matrix is not None else []}
    d.update(extra)
    return d


def _editor(**samples):
    s = {n: SimpleNamespace(name=n, path=f'{n}.fcs', comp_matrix=m,
                            comp_channels=list(CH) if m is not None else [])
         for n, m in samples.items()}
    return SimpleNamespace(
        _samples=s, _sample_order=list(s), _sample_trial=dict.fromkeys(s, 'T'),
        _sample_colors=dict.fromkeys(s, '#1f77b4'),
        _sample_gates={n: {} for n in s}, _sample_gate_order={n: [] for n in s},
        _channel_labels={})


def test_comp_applied_in_the_editor_after_an_uncompensated_drop_counts():
    """Dropped uncompensated (no snapshot), then compensated in the editor:
    the run uses the editor's matrix and it is not 'changed'."""
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', None))
    c = R.resolve_comp(m, mid, _editor(s1=EDITED))
    assert c['source'] == 'editor' and c['ready'] and c['live']
    assert c['label'] == '2×2 (editor)'
    assert c['channels'] == CH


def test_a_beads_override_with_no_files_is_not_ready():
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', None))
    m.set_item_comp(mid, {'kind': 'beads', 'files': []})
    c = R.resolve_comp(m, mid, None)
    assert c['ready'] is False and c['label'] == 'beads:0 (row)'
    assert m.run_blockers(None) == [mid]


def test_an_override_on_a_loaded_sample_is_live():
    m = WorkspaceModel()
    mid = m.add_item(_pl('s1', SAVED))
    m.set_item_comp(mid, {'kind': 'beads', 'files': ['b.fcs']})
    assert R.resolve_comp(m, mid, _editor(s1=SAVED))['live'] is True
    assert R.resolve_comp(m, mid, None)['live'] is False


def test_a_saved_matrix_from_json_is_not_flagged_changed():
    m = WorkspaceModel()
    m.add_item(_pl('s1', SAVED))
    m2 = WorkspaceModel.from_dict(m.to_dict())
    mid2 = next(iter(m2.loose))
    c = R.resolve_comp(m2, mid2, _editor(s1=SAVED.copy()))
    assert c['label'] == '2×2 (editor)'


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


def test_an_unticked_row_with_no_comp_does_not_block_the_run(
        root, tmp_path, monkeypatch):
    """Only members of the run can refuse it: an unticked loose row and a
    row of an unticked group are not run, so they do not block."""
    monkeypatch.chdir(tmp_path)
    panel = _panel(root, _editor(s1=SAVED))
    panel.model.add_item(_pl('s1', SAVED))
    off = panel.model.add_item(_pl('gone', None))
    panel.model.toggle_item_display(off)
    g = panel.model.new_group()
    panel.model.add_item(_pl('gone2', None), gid=g)
    panel.model.toggle_group_display(g)
    monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
    panel._run()
    msg = panel.status_var.get()
    assert not msg.startswith('Not run'), msg
    assert len(panel._jobs) == 1
    assert not any('run uncompensated' in w for w in panel._warnings)


def test_the_bare_warning_names_only_run_members(root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = _panel(root, _editor(bare=None, off=None))
    panel.model.add_item(_pl('bare', None))
    off = panel.model.add_item(_pl('off', None))
    panel.model.toggle_item_display(off)
    monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
    panel._run()
    ws = [w for w in panel._warnings if 'run uncompensated' in w]
    assert ws and 'T / bare' in ws[0] and 'T / off' not in ws[0], ws


def test_group_row_label_agrees_with_its_members(root):
    v = _panel(root, _editor())._view
    g = v.model.new_group()
    mid = v.model.add_item(_pl('s1', None), gid=g)
    v.model.set_group_comp(g, {'kind': 'beads', 'files': ['a.fcs', 'b.fcs']})
    v._render()
    assert v.tree.set(g, 'comp') == v.tree.set(mid, 'comp') == 'beads:2 (grp)'
    v.model.set_group_comp(g, {'kind': 'matrix', 'channels': CH,
                               'matrix': SAVED.tolist()})
    v._render()
    assert v.tree.set(g, 'comp') == v.tree.set(mid, 'comp') == 'matrix (grp)'
