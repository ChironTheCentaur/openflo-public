"""More of how a Pipeline Workspace run reads a row's or group's comp beads
(scope audit L3b-ii, from its review): the reload note, where the channels
come from, the cache key, a failing build, the warnings, and the panel
around the bead thread (a refused run folder, the queue waiting, cancel,
a thread error, idle afterwards). Generated single-stain files and generic
channels only.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from openflo import workspace_run as R
from openflo.flow_sample import FlowSample
from openflo.workspace_run import WorkspaceModel
from tests._fcs_writer import write_float_fcs
from tests.conftest import gui_unavailable

CH = ['FL1-A', 'FL2-A']


def _stain(path, bright, spill, seed):
    rng = np.random.default_rng(seed)
    n = 2000
    sig = np.where(np.arange(n) < n // 2, rng.normal(200, 30, n),
                   rng.normal(40000, 3000, n))
    other = rng.normal(150, 30, n) + spill * sig
    cols = [sig, other] if bright == 0 else [other, sig]
    ev = np.column_stack([rng.uniform(2e4, 2e5, n), *cols])
    return write_float_fcs(str(path), ev, ['FSC-A', *CH])


@pytest.fixture
def beads(tmp_path):
    d = tmp_path / 'run1'
    d.mkdir()
    return [_stain(d / 'Beads_FL1.fcs', 0, 0.15, 1),
            _stain(d / 'Beads_FL2.fcs', 1, 0.05, 2)]


def _choice(files, source='group'):
    return {'source': source, 'kind': 'beads', 'files': list(files),
            'matrix': None, 'channels': [], 'label': f'beads:{len(files)}',
            'ready': True, 'live': False}


def _sample(tmp_path, name='s1', seed=7):
    rng = np.random.default_rng(seed)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 500),
                          rng.uniform(100, 40000, 500),
                          rng.uniform(100, 40000, 500)])
    return write_float_fcs(str(tmp_path / f'{name}.fcs'), ev, ['FSC-A', *CH])


# ── core (no Tk) ────────────────────────────────────────────────────────

def test_reload_note_names_the_bead_matrix(beads, tmp_path):
    """A reloaded member (not live) is compensated with the bead matrix and
    its note says so."""
    path = _sample(tmp_path)
    ed = SimpleNamespace(_samples={})
    item = {'sample': 's1', 'path': path, 'gate_id': None,
            'comp_channels': list(CH)}
    members = [{'item': item, 'group': 'G', 'sample': 's1',
                'comp': _choice(beads)}]
    R.build_bead_comps(ed, members)
    s, note = R.resolve_run_sample(ed, item, comp=members[0]['comp'])
    assert 'compensated with the group matrix built from 2 comp bead files' in note
    np.testing.assert_allclose(np.asarray(s.comp_matrix),
                               np.asarray(members[0]['comp']['matrix']))
    assert list(s.comp_channels) == CH


def test_channels_read_from_the_fcs_when_nothing_saved(beads, tmp_path):
    path = _sample(tmp_path)
    assert R.member_channels(SimpleNamespace(_samples={}),
                             {'sample': 's1', 'path': path}) == CH


def test_live_comp_channels_win_over_fluor_channels(tmp_path):
    live = SimpleNamespace(comp_channels=['FL2-A'], fluor_channels=CH)
    ed = SimpleNamespace(_samples={'s1': live})
    assert R.member_channels(ed, {'sample': 's1',
                                  'comp_channels': ['X']}) == ['FL2-A']


def test_cache_key_holds_the_channels(beads):
    """Same beads, different channel lists: two matrices, each over its own
    channels."""
    ed = SimpleNamespace(_samples={})
    m = [{'item': {'sample': 'a', 'comp_channels': list(CH)},
          'comp': _choice(beads)},
         {'item': {'sample': 'b', 'comp_channels': ['FL1-A']},
          'comp': _choice(beads)}]
    R.build_bead_comps(ed, m)
    assert m[0]['comp']['channels'] == CH
    assert m[1]['comp']['channels'] == ['FL1-A']


def test_no_channels_stops_the_member(beads):
    out = R.bead_comp(_choice(beads), [])
    assert 'no channels to build the group bead matrix' in out['error']


def test_a_failing_build_stops_only_that_member(beads, monkeypatch):
    ed = SimpleNamespace(_samples={})
    calls = []

    def boom(comp, chans):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('planted')
        return {**comp, 'kind': 'matrix', 'matrix': np.eye(2).tolist(),
                'channels': chans}
    monkeypatch.setattr(R, 'bead_comp', boom)
    m = [{'item': {'sample': 'a', 'comp_channels': list(CH)},
          'comp': _choice(beads[:1])},
         {'item': {'sample': 'b', 'comp_channels': list(CH)},
          'comp': _choice(beads)}]
    R.build_bead_comps(ed, m)
    assert 'ValueError: planted' in m[0]['comp']['error']
    assert m[1]['comp']['kind'] == 'matrix'


def test_a_control_without_spillover_measured_is_named(tmp_path):
    """A bead file that names FL2 but holds no FL2-A column: its row stays
    identity and the warning names it."""
    d = tmp_path / 'b'
    d.mkdir()
    f1 = _stain(d / 'Beads_FL1.fcs', 0, 0.15, 1)
    rng = np.random.default_rng(3)
    f2 = write_float_fcs(str(d / 'Beads_FL2.fcs'),
                         rng.uniform(1, 1000, (2000, 2)), ['FSC-A', 'FL1-A'])
    out = R.bead_comp(_choice([f1, f2]), CH)
    assert 'error' not in out
    assert any('no spillover could be measured' in w and 'FL2-A' in w
               for w in out['warnings']), out['warnings']


def test_two_members_of_one_unit_list_the_warning_once(beads, tmp_path):
    p1, p2 = _sample(tmp_path, 's1', 7), _sample(tmp_path, 's2', 8)
    lives = {}
    for n, p in (('s1', p1), ('s2', p2)):
        s = FlowSample(p)
        s.manual_compensate(np.eye(2), CH)
        s.apply_transform(CH)
        lives[n] = s
    ed = SimpleNamespace(_samples=lives)
    members = [{'item': {'sample': n, 'path': p, 'gate_id': None},
                'group': 'G', 'sample': n, 'comp': _choice(beads[:1])}
               for n, p in (('s1', p1), ('s2', p2))]
    R.build_bead_comps(ed, members)
    prep = R.prepare_unit(ed, members, 'u', {'channels': CH})
    hits = [w for w in prep['warnings'] if 'no group comp bead file' in w]
    assert len(hits) == 1, prep['warnings']


# ── GUI ─────────────────────────────────────────────────────────────────

@pytest.fixture
def root():
    pytest.importorskip('tkinter')
    import tkinter as tk
    try:
        r = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    r.withdraw()
    yield r
    r.destroy()


def _panel(root, beads, tmp_path, monkeypatch):
    from openflo import workspace as W
    monkeypatch.chdir(tmp_path)
    live = FlowSample(beads[0])
    live.name = 's1'
    live.manual_compensate(np.eye(2), CH)
    ed = SimpleNamespace(
        _samples={'s1': live}, _sample_order=['s1'],
        _sample_trial={'s1': 'T'}, _sample_colors={'s1': '#000000'},
        _sample_gates={'s1': {}}, _sample_gate_order={'s1': []},
        _channel_labels={})
    panel = W.WorkspacePanel(root, editor=ed)
    m = WorkspaceModel()
    g = m.new_group('G')
    m.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None,
                'path': beads[0]}, gid=g)
    m.set_group_comp(g, {'kind': 'beads', 'files': beads})
    panel.model.groups, panel.model.loose = m.groups, m.loose
    return W, panel


def test_refused_run_folder_does_not_leave_the_panel_running(
        root, beads, tmp_path, monkeypatch):
    """Run refused because the run folder cannot be made: the bead thread
    must not leave the panel 'running' (Run would then answer 'A run is
    already in progress' for good)."""
    W, panel = _panel(root, beads, tmp_path, monkeypatch)

    def no(path, *a, **k):
        raise PermissionError(13, 'Permission denied', str(path))
    monkeypatch.setattr(W.os, 'makedirs', no)
    panel._run()
    assert panel.status_var.get().startswith('Cannot write the run folder')
    t = getattr(panel, '_beads', None)
    if t is not None:
        t.join(30)
    for _ in range(20):
        root.update()
    assert not panel._running()


def test_queue_waits_for_the_bead_thread(root, beads, tmp_path, monkeypatch):
    """While the bead thread runs, no unit is prepared (it would be refused
    as 'beads were not read')."""
    _W, panel = _panel(root, beads, tmp_path, monkeypatch)
    gate = threading.Event()
    real = R.build_bead_comps

    def slow(*a):
        gate.wait(30)
        return real(*a)
    monkeypatch.setattr(R, 'build_bead_comps', slow)
    launched = []
    monkeypatch.setattr(panel, '_launch_next',
                        lambda: launched.append(1) or panel._jobs.clear())
    panel._run()
    assert panel._running()
    panel._pump_jobs()
    assert launched == [] and 'comp beads' in panel.status_var.get()
    gate.set()
    panel._beads.join(30)
    panel._pump_jobs()
    assert launched == [1]


def test_cancel_during_beads_ends_the_run(root, beads, tmp_path, monkeypatch):
    _W, panel = _panel(root, beads, tmp_path, monkeypatch)
    gate = threading.Event()
    real = R.build_bead_comps
    monkeypatch.setattr(R, 'build_bead_comps',
                        lambda *a: (gate.wait(30), real(*a))[1])
    monkeypatch.setattr(panel, 'after', lambda *a, **k: None)
    panel._run()
    panel._cancel_requested = True
    panel._pump_jobs()
    assert not panel._running()
    assert panel.status_var.get().startswith('Canceled')
    gate.set()


def test_a_bead_thread_error_is_on_the_final_line(root, beads, tmp_path,
                                                  monkeypatch):
    _W, panel = _panel(root, beads, tmp_path, monkeypatch)

    from openflo import async_task
    posted = []
    # No mainloop under pytest: Tk 'after' from the worker raises and is
    # swallowed, so collect the callbacks and run them on this thread.
    monkeypatch.setattr(async_task, '_post', lambda w, fn: posted.append(fn))

    def boom(*a):
        raise OSError('planted')
    monkeypatch.setattr(R, 'build_bead_comps', boom)
    monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
    panel._run()
    panel._beads.join(30)
    for fn in posted:
        fn()
    assert any('comp beads or FMO controls could not be read: '
               'OSError: planted' in w
               for w in panel._warnings), panel._warnings


def test_panel_is_idle_after_a_bead_run(root, beads, tmp_path, monkeypatch):
    """After the bead thread ends and the queue empties, Run is free again
    (a stale _beads would answer 'A run is already in progress' for good)."""
    _W, panel = _panel(root, beads, tmp_path, monkeypatch)
    monkeypatch.setattr(panel, 'after', lambda *a, **k: None)
    monkeypatch.setattr(panel, '_launch_next', lambda: panel._jobs.clear())
    panel._run()
    panel._beads.join(30)
    panel._pump_jobs()
    assert panel.status_var.get().startswith('Done'), panel.status_var.get()
    assert not panel._running()


def test_a_bead_error_reaches_the_member_with_its_reason(beads, tmp_path):
    """The reason (a missing file) is what the user reads, not 'were not
    read'."""
    ed = SimpleNamespace(_samples={})
    gone = str(tmp_path / 'gone.fcs')
    m = [{'item': {'sample': 's1', 'path': None,
                   'comp_channels': list(CH)},
          'comp': _choice([*beads, gone])}]
    R.build_bead_comps(ed, m)
    with pytest.raises(RuntimeError, match="gone.fcs not found"):
        R.resolve_run_sample(ed, m[0]['item'], comp=m[0]['comp'])
