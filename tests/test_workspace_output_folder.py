"""Where Pipeline workspace runs write (PIPELINE-16).

Runs went to ``outputs/workspace/run_<time>`` under whatever folder OpenFlo
was started from (a shortcut, a script or a terminal), not under anything
the user chose. Now they go to an ``outputs`` folder beside the saved
workspace, beside the session file while the workspace is unsaved, under
the working folder only when there is neither, or to the folder chosen in
Preferences > Pipeline workspace. Folders are generated under tmp_path.
"""
from __future__ import annotations

import json
import os
import types
from types import SimpleNamespace

import numpy as np
import pytest

from openflo import prefs
from openflo import workspace as W
from openflo import workspace_run as R
from openflo.prefs import write_pref
from tests.conftest import editor_or_skip, gui_unavailable


@pytest.fixture(autouse=True)
def _no_chosen_folder():
    write_pref('pipeline_output_dir', '')
    yield
    write_pref('pipeline_output_dir', '')


def _editor(session_file=None):
    s1 = SimpleNamespace(name='s1', path='s1.fcs', comp_matrix=np.eye(2),
                         comp_channels=['FL1-A', 'FL2-A'])
    ed = SimpleNamespace(
        _samples={'s1': s1}, _sample_order=['s1'],
        _sample_trial={'s1': 'TrialX'}, _sample_colors={'s1': '#1f77b4'},
        _sample_gates={'s1': {}}, _sample_gate_order={'s1': []},
        _channel_labels={})
    if session_file is not None:
        ed._open_session_file = lambda: session_file
    return ed


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


def _run_dir(panel):
    """Start a run, read the folder it would write to, and cancel it."""
    panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
    panel._run()
    out = panel._out_dir
    status = panel.status_var.get()
    panel._cancel()
    panel._pump_jobs()
    return out, status


def test_run_output_root_order(tmp_path, monkeypatch):
    cwd, proj, sess = (tmp_path / 'cwd', tmp_path / 'proj',
                       tmp_path / 'sess')
    for d in (cwd, proj, sess):
        d.mkdir()
    monkeypatch.chdir(cwd)
    ws, ss = str(proj / 'w.json'), str(sess / 's.ofs')
    pick = str(tmp_path / 'picked')
    assert R.run_output_root(ws, ss, pick) == pick
    assert R.run_output_root(ws, ss, '') == str(proj / 'outputs')
    assert R.run_output_root(None, ss) == str(sess / 'outputs')
    gone = str(tmp_path / 'gone' / 'w.json')         # folder deleted since
    assert R.run_output_root(gone, ss) == str(sess / 'outputs')
    assert R.run_output_root(None, None) == str(cwd / 'outputs' / 'workspace')


def test_runs_go_beside_the_saved_workspace(root, tmp_path, monkeypatch):
    from tkinter import filedialog
    cwd, proj = tmp_path / 'cwd', tmp_path / 'proj'
    cwd.mkdir()
    proj.mkdir()
    monkeypatch.chdir(cwd)
    panel = W.WorkspacePanel(root, editor=_editor())
    monkeypatch.setattr(filedialog, 'asksaveasfilename',
                        lambda **k: str(proj / 'w.json'))
    panel._save()
    out, status = _run_dir(panel)
    assert os.path.dirname(out) == str(proj / 'outputs'), out
    assert os.path.basename(out).startswith('run_')
    assert out in status                  # the status line names the folder
    assert panel.last_run_dir == out      # what Results… opens


def test_a_loaded_workspace_runs_beside_its_file(root, tmp_path,
                                                 monkeypatch):
    from tkinter import filedialog
    cwd, other = tmp_path / 'cwd', tmp_path / 'other'
    cwd.mkdir()
    other.mkdir()
    monkeypatch.chdir(cwd)
    src = W.WorkspacePanel(root, editor=_editor())
    (other / 'w.json').write_text(json.dumps(
        src.model.to_dict()))
    panel = W.WorkspacePanel(root, editor=_editor())
    monkeypatch.setattr(filedialog, 'askopenfilename',
                        lambda **k: str(other / 'w.json'))
    panel._load()
    out, _status = _run_dir(panel)
    assert os.path.dirname(out) == str(other / 'outputs'), out


def test_an_unsaved_workspace_runs_beside_the_session(root, tmp_path,
                                                      monkeypatch):
    cwd, sess = tmp_path / 'cwd', tmp_path / 'sess'
    cwd.mkdir()
    sess.mkdir()
    monkeypatch.chdir(cwd)
    panel = W.WorkspacePanel(root, editor=_editor(str(sess / 's.ofs')))
    out, _status = _run_dir(panel)
    assert os.path.dirname(out) == str(sess / 'outputs'), out
    # Neither saved: under the working folder, as before.
    panel = W.WorkspacePanel(root, editor=_editor(None))
    out, _status = _run_dir(panel)
    assert os.path.dirname(out) == str(cwd / 'outputs' / 'workspace'), out


def test_the_folder_chosen_in_preferences_wins(root, tmp_path, monkeypatch):
    cwd, sess, pick = tmp_path / 'cwd', tmp_path / 'sess', tmp_path / 'pick'
    for d in (cwd, sess):
        d.mkdir()
    monkeypatch.chdir(cwd)
    write_pref('pipeline_output_dir', str(pick))
    panel = W.WorkspacePanel(root, editor=_editor(str(sess / 's.ofs')))
    out, _status = _run_dir(panel)
    assert os.path.dirname(out) == str(pick), out


def test_batch_folder_opens_its_output_picker_at_the_run_folder(
        tmp_path, monkeypatch):
    import tkinter.filedialog as fd
    indir, proj = tmp_path / 'in', tmp_path / 'proj'
    indir.mkdir()
    proj.mkdir()
    (indir / 'b1.fcs').write_bytes(b'x')
    asked = []

    def _ask(**k):
        asked.append(k)
        return str(indir) if len(asked) == 1 else ''   # then cancel
    monkeypatch.setattr(fd, 'askdirectory', _ask)
    v = types.SimpleNamespace(
        model=types.SimpleNamespace(run_cfg=R.default_run_cfg()),
        _editor=None, _on_batch=None,
        # The run folder's 'outputs' does not exist yet: start beside it.
        _output_root=lambda: str(proj / 'outputs'),
        _sync_run_cfg=dict)
    W.PipelineWorkspaceView._batch_folder(v)
    assert len(asked) == 2
    assert asked[1].get('initialdir') == str(proj), asked[1]


def test_preferences_choose_and_default(monkeypatch, tmp_path):
    from tkinter import filedialog

    from openflo.ui_preferences import PreferencesDialog
    root, ed = editor_or_skip()
    try:
        tips = {}
        monkeypatch.setattr(ed, '_tip', lambda w, t: tips.__setitem__(w, t),
                            raising=False)
        dlg = PreferencesDialog(ed)
        root.update()
        box, choose, reset = dlg.pipeline_output_controls
        assert dlg._pipeline_out_var.get() == dlg.PIPELINE_OUTPUT_DEFAULT
        assert "beside the saved workspace" in tips[box]
        assert tips[choose] and tips[reset]
        pick = tmp_path / 'pick'
        pick.mkdir()
        monkeypatch.setattr(filedialog, 'askdirectory',
                            lambda **k: str(pick))
        choose.invoke()
        assert prefs.load_pipeline_output_dir() == str(pick)
        assert dlg._pipeline_out_var.get() == str(pick)
        asked = []
        monkeypatch.setattr(filedialog, 'askdirectory',
                            lambda **k: asked.append(k) or '')
        choose.invoke()                       # cancelled: kept
        assert asked[-1].get('initialdir') == str(pick)   # opens there
        assert prefs.load_pipeline_output_dir() == str(pick)
        reset.invoke()
        assert prefs.load_pipeline_output_dir() == ''
        assert dlg._pipeline_out_var.get() == dlg.PIPELINE_OUTPUT_DEFAULT
        dlg.destroy()
    finally:
        root.destroy()


# ── A run folder that cannot be written is refused before any work ───────

def _refused(panel):
    panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
    panel._run()
    status = panel.status_var.get()
    assert status.startswith('Cannot write the run folder '), status
    assert 'Preferences > Pipeline workspace' in status, status
    assert not panel._running() and not panel._jobs
    assert panel._out_dir is None and panel.last_run_dir is None
    return status


def test_run_refused_when_outputs_is_a_file(root, tmp_path, monkeypatch):
    """Clustering ran first and only then failed (NotADirectoryError), once
    per unit: minutes of work on real data before the error."""
    from tkinter import filedialog
    proj = tmp_path / 'proj'
    proj.mkdir()
    (proj / 'outputs').write_text('not a folder')
    monkeypatch.chdir(tmp_path)
    panel = W.WorkspacePanel(root, editor=_editor())
    monkeypatch.setattr(filedialog, 'asksaveasfilename',
                        lambda **k: str(proj / 'w.json'))
    panel._save()
    assert 'NotADirectoryError' in _refused(panel)


def test_run_refused_in_a_read_only_folder(root, tmp_path, monkeypatch):
    ro = tmp_path / 'ro'
    ro.mkdir()
    monkeypatch.chdir(tmp_path)
    write_pref('pipeline_output_dir', str(ro / 'runs'))
    os.chmod(ro, 0o555)
    try:
        if os.access(str(ro), os.W_OK):     # root ignores the mode bits
            real = os.makedirs

            def _makedirs(path, *a, **k):
                if str(path).startswith(str(ro)):
                    raise PermissionError(13, 'Permission denied', str(path))
                return real(path, *a, **k)
            monkeypatch.setattr(W.os, 'makedirs', _makedirs)
        panel = W.WorkspacePanel(root, editor=_editor())
        assert 'PermissionError' in _refused(panel)
    finally:
        os.chmod(ro, 0o755)


# ── Small path rules of the run folder ─────────────────────────────────

def test_chosen_folder_expands_the_home_sign(monkeypatch, tmp_path):
    monkeypatch.setenv('HOME', str(tmp_path))
    assert R.run_output_root(None, None, '~/runs') == str(tmp_path / 'runs')


def test_a_session_lookup_that_raises_does_not_stop_the_run(root, tmp_path,
                                                            monkeypatch):
    cwd = tmp_path / 'cwd'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    ed = _editor()

    def _boom():
        raise RuntimeError('session lookup failed')
    ed._open_session_file = _boom
    panel = W.WorkspacePanel(root, editor=ed)
    out, _status = _run_dir(panel)
    assert os.path.dirname(out) == str(cwd / 'outputs' / 'workspace'), out


def test_a_failed_save_or_load_does_not_move_the_run_folder(root, tmp_path,
                                                            monkeypatch):
    from tkinter import filedialog
    proj = tmp_path / 'proj'
    proj.mkdir()
    monkeypatch.chdir(tmp_path)
    panel = W.WorkspacePanel(root, editor=_editor())
    monkeypatch.setattr(filedialog, 'asksaveasfilename',
                        lambda **k: str(proj / 'w.json'))
    panel._save()
    good = str(proj / 'w.json')
    assert panel._workspace_file == good
    monkeypatch.setattr(filedialog, 'asksaveasfilename',
                        lambda **k: str(tmp_path / 'gone' / 'x.json'))
    panel._save()
    assert panel.status_var.get().startswith('Save failed')
    assert panel._workspace_file == good
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    monkeypatch.setattr(filedialog, 'askopenfilename', lambda **k: str(bad))
    panel._load()
    assert panel.status_var.get().startswith('Load failed')
    assert panel._workspace_file == good


# ── Undo of a Load puts the run folder back ───────────────────────────────

def test_undo_of_a_load_puts_the_workspace_file_back(tmp_path, monkeypatch):
    """Undo of a Load kept the loaded file, so the old workspace's runs
    went beside the file Undo had taken away. Undo of an edit made before
    a Save keeps the save."""
    from tkinter import filedialog
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    monkeypatch.chdir(tmp_path)
    root, ed = editor_or_skip()
    try:
        panel = ed._ensure_workspace_panel()
        panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
        monkeypatch.setattr(filedialog, 'asksaveasfilename',
                            lambda **k: str(a / 'w.json'))
        panel._save()
        (b / 'w.json').write_text(json.dumps(R.WorkspaceModel().to_dict()))
        monkeypatch.setattr(filedialog, 'askopenfilename',
                            lambda **k: str(b / 'w.json'))
        panel._load()
        assert panel._workspace_file == str(b / 'w.json')
        assert not list(panel.model.all_items())
        ed._undo()
        assert len(list(panel.model.all_items())) == 1
        assert panel._workspace_file == str(a / 'w.json')
        assert panel._output_root() == str(a / 'outputs')
        ed._redo()
        assert panel._workspace_file == str(b / 'w.json')
        ed._undo()
        assert panel._workspace_file == str(a / 'w.json')
        # An edit, a Save, then Undo of the edit: still beside the save.
        ed._clear_undo_pending()           # a new Tk event: its own step
        panel._view._new_group()
        assert len(panel.model.groups) == 1
        monkeypatch.setattr(filedialog, 'asksaveasfilename',
                            lambda **k: str(b / 'w2.json'))
        panel._save()
        ed._undo()
        assert not panel.model.groups      # the edit was undone
        assert panel._workspace_file == str(b / 'w2.json')
    finally:
        root.destroy()
