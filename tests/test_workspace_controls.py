"""Pipeline workspace controls that did nothing, or the wrong thing.

Each test is written against the control sweep's repro: the failed unit's
reason on the final line, Cancel while a unit is staged, the 👁 column and
Run, the right-click transfer entries, Load workspace on a file that is not
one, Batch folder's Cancel, Import all twice, Remove with nothing selected,
the selection after a tree change, the undo step of Group params and
Markers, and TriMap on by default but quiet unless it is chosen. All data
is generated.
"""
from __future__ import annotations

import io
import json
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace as W
from openflo import workspace_run as R
from tests._fcs_writer import write_float_fcs
from tests.conftest import gui_unavailable


def _editor(**extra):
    """Two loaded samples, s1 (with a matrix) and s2 (without)."""
    s1 = SimpleNamespace(name='s1', path='s1.fcs', comp_matrix=np.eye(3),
                         comp_channels=['FL1-A', 'FL2-A', 'FL3-A'])
    s2 = SimpleNamespace(name='s2', path='s2.fcs', comp_matrix=None,
                         comp_channels=[])
    return SimpleNamespace(
        _samples={'s1': s1, 's2': s2}, _sample_order=['s1', 's2'],
        _sample_trial={'s1': 'TrialX', 's2': 'TrialX'},
        _sample_colors={'s1': '#1f77b4', 's2': '#ff7f0e'},
        _sample_gates={'s1': {}, 's2': {}},
        _sample_gate_order={'s1': [], 's2': []},
        _channel_labels={}, **extra)


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


def _panel(root, editor=None):
    return W.WorkspacePanel(root, editor=editor if editor is not None
                            else _editor())


# ── PIPELINE-2: a failed unit's reason stays on the final line ────────────

def test_failed_unit_reason_is_on_the_final_line(root, tmp_path,
                                                 monkeypatch):
    """The ✗ line was replaced by "Done: 0 ok, 1 failed → …" at the next
    poll, so the user never saw why the unit failed."""
    monkeypatch.chdir(tmp_path)
    panel = _panel(root)
    # A saved matrix, so the run gets past the no-compensation refusal to
    # the load failure this test is about.
    panel.model.add_item({'sample': 'ghost', 'trial': 'T', 'path': None,
                          'gate_id': None, 'comp_matrix': [[1.0]],
                          'comp_channels': ['FL1-A']})
    panel._run()
    panel._pump_jobs()                 # prepares (fails) and finishes
    final = panel.status_var.get()
    assert final.startswith('Done: 0 ok, 1 failed'), final
    assert ('[✗] T_ghost: RuntimeError: sample not loaded in editor and '
            'no readable FCS path') in final, final


# ── PIPELINE-3: Cancel and the Run guard while a unit is staged ───────────

def test_cancel_and_run_guard_while_a_unit_is_staged(root, tmp_path):
    """_running() ignored the job-file write (_prep): Cancel did nothing
    and a second Run was accepted while the last unit was being staged."""
    panel = _panel(root)
    release = threading.Event()
    writer = threading.Thread(target=release.wait, args=(10,), daemon=True)
    writer.start()
    jobdir = tmp_path / 'wsjob_x'
    jobdir.mkdir()
    panel._out_dir = str(tmp_path / 'run_a')
    panel._prep = {'thread': writer, 'state': {'error': None},
                   'job': {'label': 'u1'}, 'n': 1, 'jobdir': str(jobdir),
                   'job_path': '', 'result_path': ''}
    try:
        assert panel._running()
        panel._run()
        assert panel.status_var.get() == "A run is already in progress."
        assert panel._out_dir == str(tmp_path / 'run_a')
        panel._cancel()
        assert panel._cancel_requested
        panel._pump_jobs()
        assert panel.status_var.get().startswith('Canceled.')
        assert panel._prep is None and not panel._running()
    finally:
        release.set()


# ── PIPELINE-4: Run leaves out rows unticked in the 👁 column ────────────

def test_run_units_leave_out_unticked_items_and_groups():
    m = R.WorkspaceModel()
    pay = {'trial': 'T', 'gate_id': None}
    a = m.add_item(dict(pay, sample='a'))
    b = m.add_item(dict(pay, sample='b'))
    g1 = m.new_group('G one')
    m.add_item(dict(pay, sample='c'), g1)
    off = m.add_item(dict(pay, sample='d'), g1)
    g2 = m.new_group('G two')
    m.add_item(dict(pay, sample='e'), g2)
    m.toggle_item_display(b)
    m.toggle_item_display(off)
    m.toggle_group_display(g2)
    units = R.build_run_units(m, {})
    got = {u['label']: sorted(x['sample'] for x in u['members'])
           for u in units}
    assert got == {'G_one': ['c'], R.run_label(m.item(a)): ['a']}
    [cat] = R.build_run_units(m, {'concatenate': True})
    assert sorted(x['sample'] for x in cat['members']) == ['a', 'c']
    assert sorted(x['sample'] for x in cat['members']) == sorted(
        it['sample'] for it in m.displayed_items())


def test_run_with_every_item_unticked_says_so(root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = _panel(root)
    mid = panel.model.add_item({'sample': 's1', 'trial': 'T',
                                'gate_id': None})
    panel.model.toggle_item_display(mid)
    panel._run()
    assert panel.status_var.get() == ("Nothing to run — every item is "
                                      "unticked in the 👁 column.")
    assert not panel._running()


# ── PIPELINE-5: right-click Copy / Move hand on the item's sample ────────

def test_right_click_copy_and_move_hand_on_the_sample(root, tmp_path,
                                                      monkeypatch):
    calls = []
    fcs = tmp_path / 's1.fcs'
    fcs.write_bytes(b'x')
    ed = _editor(_copy_samples_transfer=lambda n: calls.append(('copy', n)),
                 _send_samples_transfer=lambda n: calls.append(('send', n)))
    ed._samples['s1'].path = str(fcs)
    panel = _panel(root, ed)
    v = panel._view
    v._import_all()
    mid = next(m for m, it, _g in v.model.all_items()
               if it['sample'] == 's1')
    v.tree.selection_set(mid)
    v._copy_items_transfer()
    v._send_items_transfer()
    assert calls == [('copy', ['s1']), ('send', ['s1'])]
    # Alt-drag out sends the loaded sample's FCS (tkinterdnd2 stand-in),
    # also when the path the item was added with is no longer there.
    v.model.item(mid)['path'] = str(tmp_path / 'moved' / 's1.fcs')
    monkeypatch.setitem(sys.modules, 'tkinterdnd2', types.SimpleNamespace(
        COPY='copy', DND_FILES='files'))
    v._os_drag_armed = True
    assert v._on_os_drag_init(None) == ('copy', 'files', (str(fcs),))


# ── PIPELINE-10: Load workspace… refuses a file that is not one ──────────

def test_load_workspace_refuses_a_non_workspace_and_keeps_the_open_one(
        root, tmp_path, monkeypatch):
    from tkinter import filedialog
    panel = _panel(root)
    panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
    fired = []
    panel._on_before_change = lambda: fired.append(1)
    bad = tmp_path / 'bad.json'
    for content in ({'nope': 1}, [1, 2], {'groups': 'x'}):
        bad.write_text(json.dumps(content))
        monkeypatch.setattr(filedialog, 'askopenfilename',
                            lambda **k: str(bad))
        panel._load()
        assert panel.status_var.get().startswith(
            'Load failed: ValueError: not an OpenFlo workspace'), content
        assert len(list(panel.model.all_items())) == 1 and not fired
    # A real saved workspace, and one from before the format marker, load.
    good = tmp_path / 'good.json'
    d = panel.model.to_dict()
    for saved in (d, {k: v for k, v in d.items() if k != 'format'}):
        good.write_text(json.dumps(saved))
        monkeypatch.setattr(filedialog, 'askopenfilename',
                            lambda **k: str(good))
        panel._load()
        assert panel.status_var.get().startswith('Loaded 1 item(s)')
    assert R.is_workspace_dict({'format': R.WORKSPACE_FORMAT})
    assert R.is_workspace_dict({'loose': []})       # before the marker
    assert not R.is_workspace_dict({'format': 'x', 'title': 'y'})


# ── PIPELINE-11: Batch folder… can be cancelled ───────────────────────────

class _BatchView:
    """The bits of the view _batch_folder uses, with after() queued."""

    def __init__(self, cfg):
        self.model = types.SimpleNamespace(run_cfg=cfg)
        self._editor = None
        self.queued, self.log, self.batch = [], [], []
        self._on_batch = self.batch.append

    def _sync_run_cfg(self):
        return {}

    def after(self, _ms, fn):
        self.queued.append(fn)

    def _status(self, msg):
        self.log.append(msg)

    _batch_folder = W.PipelineWorkspaceView._batch_folder


def _batch_setup(tmp_path, monkeypatch):
    """Three generated FCS files in a folder; the dialogs answered."""
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    indir, outdir = tmp_path / 'in', tmp_path / 'out'
    indir.mkdir()
    outdir.mkdir()
    rng = np.random.default_rng(3)
    for i in (1, 2, 3):
        write_float_fcs(str(indir / f'b{i}.fcs'),
                        rng.lognormal(8, .3, (300, 3)).astype(np.float32),
                        ['FSC-A', 'SSC-A', 'FL1-A'])
    answers = [str(indir), str(outdir)]
    monkeypatch.setattr(fd, 'askdirectory', lambda **k: answers.pop(0))
    monkeypatch.setattr(mb, 'askyesno', lambda *a, **k: True)
    monkeypatch.setattr(W, 'ResultsViewer', lambda *a, **k: None)


def _pump_until_status(v):
    end = time.monotonic() + 30
    while time.monotonic() < end and not v.log:
        while v.queued:
            v.queued.pop(0)()
        time.sleep(0.02)


def test_batch_folder_stops_when_cancelled(tmp_path, monkeypatch):
    """Cancel pressed while a file is being clustered stops that file and
    starts no other; before, Cancel stayed disabled for the whole batch."""
    _batch_setup(tmp_path, monkeypatch)
    started, runs = threading.Event(), []

    def _compute_run(prep, cfg, out, should_cancel=None, **_kw):
        runs.append(prep['label'])
        started.set()
        end = time.monotonic() + 30
        while time.monotonic() < end:
            if should_cancel is not None and should_cancel():
                return {'ok': False, 'cancelled': True}
            time.sleep(0.01)
        return {'ok': True}

    monkeypatch.setattr(W, 'compute_run', _compute_run)
    v = _BatchView(dict(R.default_run_cfg(), umap=False))
    v._batch_folder()
    assert started.wait(30)
    [stop] = v.batch                       # the host was told, with a stop
    assert isinstance(stop, threading.Event)
    stop.set()                             # what the panel's Cancel does
    _pump_until_status(v)
    assert runs == [R.run_label({'sample': 'b1'})]
    assert v.log[-1].startswith('Batch canceled after 0 of 3 file(s): '
                                '0 ok, 0 failed'), v.log
    assert v.batch == [stop, None]         # and told when it ended


def test_batch_folder_cancelled_between_files_starts_no_other(tmp_path,
                                                              monkeypatch):
    _batch_setup(tmp_path, monkeypatch)
    runs = []
    v = _BatchView(dict(R.default_run_cfg(), umap=False))

    def _compute_run(prep, cfg, out, should_cancel=None, **_kw):
        runs.append(prep['label'])
        if should_cancel is not None and should_cancel():
            return {'ok': False, 'cancelled': True}
        v.batch[0].set()                   # Cancel, just as the file ends
        return {'ok': True}

    monkeypatch.setattr(W, 'compute_run', _compute_run)
    v._batch_folder()
    _pump_until_status(v)
    assert runs == [R.run_label({'sample': 'b1'})]
    assert v.log[-1].startswith('Batch canceled after 1 of 3 file(s): '
                                '1 ok, 0 failed'), v.log


def test_panel_cancel_enabled_during_a_batch_and_stops_it(root):
    panel = _panel(root)
    v = panel._view
    assert str(v.btn_cancel.cget('state')) == 'disabled'
    stop = threading.Event()
    v._on_batch(stop)
    assert str(v.btn_cancel.cget('state')) == 'normal'
    assert str(v.btn_run.cget('state')) == 'disabled'
    panel._cancel()
    assert stop.is_set()
    assert panel.status_var.get().startswith('Canceling the batch')
    v._on_batch(None)
    assert str(v.btn_cancel.cget('state')) == 'disabled'
    assert str(v.btn_run.cget('state')) == 'normal'


def _cancel_on(v):
    return (str(v.btn_cancel.cget('state')) == 'normal'
            and str(v.btn_run.cget('state')) == 'disabled')


def test_a_rebuilt_view_keeps_cancel_on_during_a_batch(root):
    """Load, Undo and Pop out rebuild the view. Each new view must be told
    a batch is going, or its Cancel is off with the batch still running."""
    panel = _panel(root)
    stop = threading.Event()
    panel._view._on_batch(stop)
    panel.restore_model(panel.model.to_dict())        # what Undo / Load do
    assert _cancel_on(panel._view)
    panel._toggle_popout()
    _top, pv = panel._pop
    assert _cancel_on(pv)
    panel._redock()
    assert _cancel_on(panel._view)
    panel._view._on_batch(None)
    assert not _cancel_on(panel._view)


def test_run_is_refused_while_a_batch_goes(root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = _panel(root)
    panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
    panel._batch_state(threading.Event())
    panel._run()
    assert panel.status_var.get() == "A run is already in progress."
    assert not panel._jobs and panel._out_dir is None


def test_closing_the_panel_stops_a_running_batch(root):
    panel = _panel(root)
    stop = threading.Event()
    panel._view._on_batch(stop)
    panel.destroy()
    assert stop.is_set()


def test_a_unit_that_will_not_launch_is_on_the_final_line(root, tmp_path,
                                                          monkeypatch):
    """The child process failing to start counts the unit as failed and
    keeps why for the run's final line."""
    import subprocess
    panel = _panel(root)
    jobdir = tmp_path / 'wsjob_x'
    jobdir.mkdir()
    done = threading.Thread(target=lambda: None)
    done.start()
    done.join()
    panel._out_dir = str(tmp_path / 'run_a')
    panel._total = 1
    panel._prep = {'thread': done, 'state': {'error': None},
                   'job': {'label': 'u1'}, 'n': 1, 'jobdir': str(jobdir),
                   'job_path': '', 'result_path': ''}

    def _no_start(*_a, **_k):
        raise OSError('no such interpreter')
    monkeypatch.setattr(subprocess, 'Popen', _no_start)
    panel._pump_jobs()
    final = panel.status_var.get()
    assert final.startswith('Done: 0 ok, 1 failed'), final
    assert '[✗] u1: launch failed: no such interpreter' in final, final
    assert not jobdir.exists()


# ── PIPELINE-12, 13, 14, 15: tree edits ───────────────────────────────────

def test_import_all_twice_adds_each_sample_once(root):
    v = _panel(root)._view
    v._import_all()
    v._import_all()
    samples = sorted(it['sample'] for _m, it, _g in v.model.all_items())
    assert samples == ['s1', 's2']
    assert v.model.run_cfg is not None
    status = []
    v._on_status = status.append
    v._import_all()
    assert status == ["All 2 sample(s) are already in the workspace."]


def test_remove_selected_with_nothing_selected_says_so(root):
    v = _panel(root)._view
    v._import_all()
    status = []
    v._on_status = status.append
    v.tree.selection_set(())
    v._remove_selected()
    assert status == ["Select a group or item to remove."]
    assert len(list(v.model.all_items())) == 2


def test_selection_is_kept_after_a_tree_change(root, monkeypatch):
    v = _panel(root)._view
    gid = v.model.new_group('Trial 1')
    v._render()
    v.tree.selection_set(gid)
    monkeypatch.setattr(v, '_pick_files', lambda title: ['b1.fcs'])
    v._set_comp_picker()
    assert v.tree.selection() == (gid,)
    v._set_fmo_picker()                    # no re-select needed
    assert v.model.groups[gid]['fmo'] == {'files': ['b1.fcs']}


def test_removing_the_selected_row_keeps_the_tree_working(root):
    """The selection is put back after a render only for rows that are
    still there: putting back a removed row raised TclError."""
    v = _panel(root)._view
    v._import_all()
    mid = next(iter(v.model.loose))
    v.tree.selection_set(mid)
    v._remove_selected()
    assert v.tree.selection() == ()
    assert mid not in v.model.loose and not v.tree.exists(mid)


def test_import_all_adds_a_sample_that_is_here_only_as_a_gate(root):
    v = _panel(root)._view
    v.model.add_item({'sample': 's1', 'trial': 'TrialX', 'gate_id': 'g1',
                      'gate_path': 'P1'})
    v._import_all()
    whole = sorted(it['sample'] for _m, it, _g in v.model.all_items()
                   if not it.get('gate_id'))
    assert whole == ['s1', 's2']


def test_queued_line_says_how_many_items_were_left_out(root, tmp_path,
                                                       monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = _panel(root)
    panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
    off = panel.model.add_item({'sample': 's2', 'trial': 'T',
                                'gate_id': None})
    panel.model.toggle_item_display(off)
    panel._run()
    assert '(1 unticked item(s) left out)' in panel.status_var.get()
    panel._cancel()
    panel._pump_jobs()


def _toplevel(v, prefix):
    import tkinter as tk
    return next(w for w in v.winfo_children()
                if isinstance(w, tk.Toplevel) and w.title().startswith(prefix))


def _buttons(dlg):
    from tkinter import ttk
    out, stack = {}, [dlg]
    while stack:
        w = stack.pop()
        stack.extend(w.winfo_children())
        if isinstance(w, ttk.Button):
            out[w.cget('text')] = w
    return out


def test_group_params_and_markers_take_their_own_undo_step(root):
    ed = _editor()
    # Markers come from loaded data: give both samples marker columns.
    for s in ed._samples.values():
        s.data = pd.DataFrame(np.ones((3, 2)), columns=['FL1-A', 'FL2-A'])
        s.fluor_channels = ['FL1-A', 'FL2-A']
    v = _panel(root, ed)._view
    gid = v.model.new_group('Trial 1')
    v._render()
    fired = []
    v._on_before_change = lambda: fired.append(1)
    for label in ('Apply', 'Inherit (clear)'):
        v.tree.selection_set(gid)
        v._group_params()
        _buttons(_toplevel(v, 'Group params'))[label].invoke()
    assert len(fired) == 2
    v._pick_markers()
    _buttons(_toplevel(v, 'Clustering markers'))['Apply'].invoke()
    assert len(fired) == 3


# ── TriMap: off unless chosen; a real failure still shows when it is ─────

@pytest.fixture
def broken_trimap(monkeypatch, tmp_path):
    """A generated package installed but failing to import, as TriMap."""
    pkg = tmp_path / 'ofl_test_broken_trimap'
    pkg.mkdir()
    (pkg / '__init__.py').write_text(
        "raise ImportError('a dependency is missing')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    from openflo import capabilities
    monkeypatch.setattr(capabilities, '_IMPORT_RESULT', {})
    monkeypatch.setitem(R.EXTRA_EMBEDDINGS, 'trimap',
                        ('ofl_test_broken_trimap', 'TriMap'))


def _quiet_run(cfg, tmp_path, said=None):
    rng = np.random.RandomState(0)
    df = pd.DataFrame({c: rng.normal(0, 1, 200)
                       for c in ('FL1-A', 'FL2-A', 'FL3-A')})
    prep = {'data': df, 'channels': list(df.columns), 'label': 'b1',
            'n_events': 200, 'note': '', 'color_by': None}
    so, sys.stdout = sys.stdout, io.StringIO()       # cluster chatter
    try:
        return R.compute_run(prep, cfg, str(tmp_path),
                             progress=None if said is None else said.append)
    finally:
        sys.stdout = so


def _queued_cfg(panel):
    """Start a run of one item, read the cfg its unit was given, cancel."""
    if not list(panel.model.all_items()):
        panel.model.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None})
    panel._run()
    cfg = dict(panel._jobs[0]['cfg'])
    panel._cancel()
    panel._pump_jobs()
    return cfg


def test_trimap_on_by_default_is_quiet_unless_chosen(broken_trimap, root,
                                                     tmp_path, monkeypatch):
    """TriMap is on by default. Installed but not loading, it is offered
    until the background import check comes back, so a run started then
    warned "TriMap skipped: installed but fails to load" to a user who never
    chose it. It stays on by default; that run now skips it with a progress
    note only. Ticked by the user, the warning stays."""
    from tests._tk_wait import pump_until
    monkeypatch.chdir(tmp_path)
    assert R.default_run_cfg()['trimap'] is True
    panel = _panel(root)
    v = panel._view                        # before the import check is back
    assert v._trimap_var.get() is True
    assert _queued_cfg(panel)['unchosen_embeddings'] == ['trimap']
    cfg = dict(R.default_run_cfg(), method='phenograph', umap=False,
               max_events=0, unchosen_embeddings=['trimap'])
    said = []
    r = _quiet_run(cfg, tmp_path / 'a', said)
    assert r['ok'] and not r['trimap']
    assert not any('TriMap' in w for w in r['warnings']), r['warnings']
    assert any('TriMap skipped: installed but fails to load' in m
               and 'on by default; not chosen' in m for m in said), said
    # The user unticks and ticks it again: chosen, so it is said.
    cb, _var = v._embed_boxes['trimap']
    cb.invoke()
    cb.invoke()
    assert v._trimap_var.get() is True
    assert _queued_cfg(panel)['unchosen_embeddings'] == []
    r = _quiet_run(dict(cfg, unchosen_embeddings=[]), tmp_path / 'b')
    assert r['ok'] and not r['trimap']
    assert any(w.startswith('TriMap skipped: installed but fails to load')
               and 'a dependency is missing' in w
               for w in r['warnings']), r['warnings']
    # A choice survives the view being rebuilt (pop-out, Load, Undo).
    panel._reembed()
    assert _queued_cfg(panel)['unchosen_embeddings'] == []
    # A click in the popped-out view is recorded too.
    other = _panel(root)
    other._toggle_popout()
    pv = other._pop[1]
    pv._embed_boxes['umap'][0].invoke()
    pv._embed_boxes['umap'][0].invoke()
    assert other._embeds_chosen == {'umap'}
    other._redock()
    # Once the import check is back, the box is unticked and disabled.
    fresh = _panel(root)._view
    assert pump_until(root, lambda: not fresh._trimap_var.get())
    assert 'disabled' in fresh._embed_boxes['trimap'][0].state()


def test_only_engines_on_by_default_count_as_unchosen():
    """PHATE is off by default: ticked, the user chose it, clicked or not
    (a loaded recipe or workspace), so its failure is still said."""
    on = dict(R.default_run_cfg(), phate=True, trimap=True, pacmap=True)
    assert R.unchosen_embeddings(on, set()) == ['trimap']
    assert R.unchosen_embeddings(on, {'trimap'}) == []
    assert R.unchosen_embeddings(dict(on, trimap=False), set()) == []
