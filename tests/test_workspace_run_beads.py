"""A Pipeline Workspace run builds its matrix from a row's or group's comp
beads (scope audit L3, b-ii).

Bead files set on a row or a group ("beads:N") were shown but never read.
A run now pairs each bead file with its channel by name, as the
compensation editor's Auto-fill does (now ``comp_controls.ControlMatcher``,
shared), builds the matrix with ``optimize_compensation``, and applies it as
a row or group matrix. Each bead set gives its own matrix, so two
acquisition runs in one workspace keep their own beads, and members sharing
a set compute it once. The files are read on a worker thread before the
queue starts. Beads that cannot give a matrix stop that unit with the
reason. Generated single-stain files and generic channels only.
"""
from __future__ import annotations

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
    """A single-stain file: half the events negative, half bright on
    channel ``bright`` (0 or 1), spilling ``spill`` of it into the other."""
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
    a = tmp_path / 'run1'
    b = tmp_path / 'run2'
    a.mkdir()
    b.mkdir()
    return {
        'run1': [_stain(a / 'Beads_FL1.fcs', 0, 0.15, 1),
                 _stain(a / 'Beads_FL2.fcs', 1, 0.05, 2)],
        'run2': [_stain(b / 'Beads_FL1.fcs', 0, 0.30, 3),
                 _stain(b / 'Beads_FL2.fcs', 1, 0.02, 4)],
    }


def _choice(files, source='group'):
    return {'source': source, 'kind': 'beads', 'files': list(files),
            'matrix': None, 'channels': [], 'label': f'beads:{len(files)}',
            'ready': True, 'live': False}


def test_the_beads_give_the_spillover_they_hold(beads):
    out = R.bead_comp(_choice(beads['run1']), CH)
    assert 'error' not in out, out
    assert out['kind'] == 'matrix' and out['channels'] == CH
    m = np.asarray(out['matrix'])
    assert m[0, 1] == pytest.approx(0.15, abs=0.01)
    assert m[1, 0] == pytest.approx(0.05, abs=0.01)
    assert out['bead_files'] == beads['run1'] and out['warnings'] == []


def test_each_runs_beads_give_its_own_matrix_and_a_set_is_built_once(
        beads, monkeypatch):
    from openflo import compensation_optimizer as co
    calls = []
    real = co.optimize_compensation

    def counted(channels, paths, **kw):
        calls.append(tuple(sorted(paths.values())))
        return real(channels, paths, **kw)
    monkeypatch.setattr(co, 'optimize_compensation', counted)
    ed = SimpleNamespace(_samples={})
    item = {'sample': 'x', 'comp_channels': list(CH)}
    members = [{'item': dict(item), 'comp': _choice(beads['run1'])},
               {'item': dict(item), 'comp': _choice(beads['run1'])},
               {'item': dict(item), 'comp': _choice(beads['run2'])}]
    R.build_bead_comps(ed, members)
    m1, m1b, m2 = (np.asarray(m['comp']['matrix']) for m in members)
    np.testing.assert_array_equal(m1, m1b)
    assert m1[0, 1] == pytest.approx(0.15, abs=0.01)
    assert m2[0, 1] == pytest.approx(0.30, abs=0.01)
    assert len(calls) == 2


def test_a_missing_bead_file_stops_the_member(beads, tmp_path):
    out = R.bead_comp(_choice([*beads['run1'], str(tmp_path / 'gone.fcs')]), CH)
    assert 'cannot be read' in out['error'] and 'gone.fcs' in out['error']


def test_a_file_fitting_two_channels_equally_stops_the_member(beads, tmp_path):
    ev = np.random.default_rng(5).uniform(1, 1000, (300, 3))
    both = write_float_fcs(str(tmp_path / 'Beads_FL.fcs'), ev, ['FSC-A', *CH])
    out = R.bead_comp(_choice([both]), CH)
    assert 'do not pair with the channels one to one' in out['error']


def test_files_naming_no_channel_stop_the_member(tmp_path):
    ev = np.random.default_rng(6).uniform(1, 1000, (300, 3))
    f = write_float_fcs(str(tmp_path / 'Unstained.fcs'), ev, ['FSC-A', *CH])
    out = R.bead_comp(_choice([f]), CH)
    assert 'names one of the channels' in out['error']


def test_a_channel_without_a_bead_file_is_named(beads):
    out = R.bead_comp(_choice(beads['run1'][:1]), CH)
    assert 'error' not in out
    assert any('no group comp bead file for FL2-A' in w for w in out['warnings'])
    np.testing.assert_allclose(np.asarray(out['matrix'])[1], [0.0, 1.0])


def test_the_member_runs_on_the_bead_matrix(beads, tmp_path):
    rng = np.random.default_rng(7)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 500),
                          rng.uniform(100, 40000, 500),
                          rng.uniform(100, 40000, 500)])
    path = write_float_fcs(str(tmp_path / 's1.fcs'), ev, ['FSC-A', *CH])
    live = FlowSample(path)
    live.manual_compensate(np.eye(2), CH)
    live.apply_transform(CH)
    ed = SimpleNamespace(_samples={'s1': live})
    item = {'sample': 's1', 'path': path, 'gate_id': None,
            'comp_matrix': np.eye(2), 'comp_channels': list(CH)}
    members = [{'item': item, 'group': 'G', 'sample': 's1',
                'comp': _choice(beads['run1'])}]
    R.build_bead_comps(ed, members)
    s, note = R.resolve_run_sample(ed, item, comp=members[0]['comp'])
    want = FlowSample(path)
    want.manual_compensate(np.asarray(members[0]['comp']['matrix']), CH)
    want.apply_transform(CH)
    np.testing.assert_allclose(s.data[CH].to_numpy(), want.data[CH].to_numpy(),
                               rtol=1e-6, atol=1e-6)
    assert 'compensated with the group matrix built from 2 comp bead files' in note


def test_a_bead_error_refuses_the_member():
    comp = dict(_choice(['x.fcs']), error='the group comp beads cannot be read')
    with pytest.raises(RuntimeError, match="'s1': the group comp beads"):
        R.resolve_run_sample(SimpleNamespace(_samples={}),
                             {'sample': 's1', 'path': None}, comp=comp)


def test_beads_never_read_refuse_the_member():
    """A bead choice still unread when the member runs is refused, not run
    on the editor's or the saved matrix."""
    with pytest.raises(RuntimeError, match="'s1': the group comp beads were not read"):
        R.resolve_run_sample(SimpleNamespace(_samples={}),
                             {'sample': 's1', 'path': None},
                             comp=_choice(['x.fcs']))


def test_a_unit_carries_the_bead_warnings(beads, tmp_path):
    rng = np.random.default_rng(8)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 400),
                          rng.uniform(100, 40000, 400),
                          rng.uniform(100, 40000, 400)])
    path = write_float_fcs(str(tmp_path / 's1.fcs'), ev, ['FSC-A', *CH])
    live = FlowSample(path)
    live.manual_compensate(np.eye(2), CH)
    live.apply_transform(CH)
    ed = SimpleNamespace(_samples={'s1': live})
    item = {'sample': 's1', 'path': path, 'gate_id': None}
    members = [{'item': item, 'group': 'G', 'sample': 's1',
                'comp': _choice(beads['run1'][:1])}]
    R.build_bead_comps(ed, members)
    prep = R.prepare_unit(ed, members, 'u', {'channels': CH})
    assert any('no group comp bead file for FL2-A' in w
               for w in prep['warnings']), prep['warnings']


def test_run_builds_the_bead_matrix_off_the_tk_thread(beads, tmp_path,
                                                     monkeypatch):
    pytest.importorskip('tkinter')
    import threading
    import tkinter as tk
    try:
        r = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    r.withdraw()
    try:
        from openflo import workspace as W
        monkeypatch.chdir(tmp_path)
        live = FlowSample(beads['run1'][0])
        live.name = 's1'
        live.manual_compensate(np.eye(2), CH)
        ed = SimpleNamespace(
            _samples={'s1': live}, _sample_order=['s1'],
            _sample_trial={'s1': 'T'}, _sample_colors={'s1': '#000000'},
            _sample_gates={'s1': {}}, _sample_gate_order={'s1': []},
            _channel_labels={})
        panel = W.WorkspacePanel(r, editor=ed)
        m = WorkspaceModel()
        g = m.new_group('G')
        m.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None,
                    'path': beads['run1'][0]}, gid=g)
        m.set_group_comp(g, {'kind': 'beads', 'files': beads['run1']})
        panel.model.groups, panel.model.loose = m.groups, m.loose
        seen = []
        real = R.build_run_controls
        monkeypatch.setattr(W, 'build_run_controls', lambda *a: (
            seen.append(threading.current_thread() is threading.main_thread()),
            real(*a)))
        monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
        panel._run()
        t = panel._beads
        assert t is not None
        t.join(30)
        assert seen == [False]
        (job,) = list(panel._jobs)
        comp = job['unit']['members'][0]['comp']
        assert comp['kind'] == 'matrix' and comp['bead_files'] == beads['run1']
    finally:
        r.destroy()
