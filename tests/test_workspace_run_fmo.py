"""A Pipeline Workspace run uses a row's or group's FMO set (scope audit L3, c).

FMO files set on a row or a group were shown ("N FMO") but never read.
A run now pairs each FMO file with its channel by name, reads it as the
member's events are read (QC as the member had it, compensated with the
MEMBER's matrix, transformed), takes the cutoff at the 99th percentile of
its linear values inside the member's population, and flags each event
``<channel>_pos``. The cluster table gains ``pct_pos_<channel>`` and a
``phenotype``, and ``<label>_fmo_thresholds.csv`` lists every cutoff.
FMOs annotate: clustering and the events clustered are unchanged.
Generated data and generic channels only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace_run as R
from openflo.flow_sample import FlowSample
from tests._fcs_writer import write_float_fcs
from tests.conftest import gui_unavailable

CH = ['FL1-A', 'FL2-A']
COLS = ['FSC-A', *CH]


def _write(path, fsc, fl1, fl2):
    return write_float_fcs(str(path), np.column_stack([fsc, fl1, fl2]), COLS)


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(11)
    n = 2000
    fsc = rng.uniform(5e4, 2e5, n)
    pos = np.arange(n) % 2 == 0                      # half positive on FL1
    fl1 = np.where(pos, rng.normal(20000, 1500, n), rng.normal(300, 40, n))
    stained = _write(tmp_path / 's1.fcs', fsc, fl1, rng.normal(300, 40, n))
    # The FMO: FL1 background only, plus bright debris below the FSC gate.
    debris = np.arange(n) < n // 5
    f_fsc = np.where(debris, rng.uniform(1e3, 2e4, n), rng.uniform(5e4, 2e5, n))
    f_fl1 = np.where(debris, rng.normal(30000, 3000, n), rng.normal(300, 40, n))
    fmo = _write(tmp_path / 'FMO_FL1.fcs', f_fsc, f_fl1, rng.normal(300, 40, n))
    other = _write(tmp_path / 'Unstained.fcs', f_fsc, f_fl1, f_fl1)
    return SimpleNamespace(stained=stained, fmo=fmo, other=other)


def _live(path):
    s = FlowSample(path)
    s.name = 's1'
    s.manual_compensate(np.eye(2), CH)
    s.apply_transform(CH)
    return s


def _member(path, fmo_files, gated=False):
    item = {'sample': 's1', 'path': path, 'gate_id': None, 'qc_applied': False,
            'comp_matrix': np.eye(2), 'comp_channels': list(CH)}
    if gated:
        item['gate_id'] = 'g1'
        item['gates'] = {'g1': {'id': 'g1', 'kind': 'threshold',
                                'channel': 'FSC-A', 'op': '>', 'value': 4e4,
                                'parent_id': None}}
    comp = {'source': 'editor', 'kind': 'matrix', 'matrix': np.eye(2),
            'channels': list(CH), 'files': [], 'label': '2×2 (editor)'}
    return {'item': item, 'group': 'G', 'sample': 's1', 'comp': comp,
            'fmo': {'files': list(fmo_files), 'source': 'group'}}


def test_an_fmo_flags_the_positive_events(files):
    live = _live(files.stained)
    ed = SimpleNamespace(_samples={'s1': live})
    m = _member(files.stained, [files.fmo], gated=True)
    R.build_run_controls(ed, [m])
    s, _note = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
    warned: list = []
    rows = R.apply_fmo_controls(s, m, 99.0, warned)
    assert warned == []
    (row,) = rows
    assert row['channel'] == 'FL1-A' and row['fmo_file'] == 'FMO_FL1.fcs'
    # Half the 2000 events are positive by construction; the FMO's 99th
    # percentile lets ~1% of the negatives over the cutoff, plus sampling noise.
    assert row['pct_pos'] == pytest.approx(50.0, abs=2.0)
    assert set(np.unique(s.data['FL1-A_pos'])) <= {0.0, 1.0}


def test_the_cutoff_is_taken_inside_the_population(files):
    """Without the gate, the FMO's bright debris sets the cutoff above the
    positives and calls almost nothing positive."""
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    got = {}
    for gated in (True, False):
        m = _member(files.stained, [files.fmo], gated=gated)
        R.build_run_controls(ed, [m])
        s, _n = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
        (got[gated],) = R.apply_fmo_controls(s, m, 99.0, [])
    assert got[True]['threshold_linear'] < 1000
    # As above, with the gate also dropping events (sampling noise of n=2000).
    assert got[True]['pct_pos'] == pytest.approx(50.0, abs=3.0)
    assert got[True]['fmo_events_inside'] < 2000 * 0.85
    assert got[False]['threshold_linear'] > 20000
    assert got[False]['pct_pos'] < 10
    assert got[False]['fmo_events_inside'] == 2000


def test_the_fmo_takes_the_members_compensation(files):
    m = np.array([[1.0, 0.0], [0.5, 1.0]])
    comp = {'kind': 'matrix', 'matrix': m, 'channels': list(CH)}
    fs = R.load_fmo_control(files.fmo, comp, qc=False)
    want = FlowSample(files.fmo)
    want.manual_compensate(m, CH)
    want.apply_transform()
    np.testing.assert_allclose(fs.data[CH].to_numpy(), want.data[CH].to_numpy())
    np.testing.assert_array_equal(fs.comp_matrix, m)


def test_an_fmo_naming_no_channel_is_skipped_and_named(files):
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.other])
    R.build_run_controls(ed, [m])
    s, _n = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
    warned: list = []
    assert R.apply_fmo_controls(s, m, 99.0, warned) == []
    assert any('Unstained.fcs names none of the channels' in w for w in warned)
    assert 'FL1-A_pos' not in s.data.columns


def test_a_file_shared_by_members_is_read_once(files, monkeypatch):
    calls = []
    real = R.load_fmo_control
    monkeypatch.setattr(R, 'load_fmo_control',
                        lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    ms = [_member(files.stained, [files.fmo]) for _ in range(3)]
    R.build_run_controls(ed, ms)
    assert calls == [files.fmo]
    assert all('FL1-A' in m['fmo_controls']['controls'] for m in ms)


def test_channels_read_on_the_tk_thread_are_used(files, monkeypatch):
    """The view reads a live member's channels before the worker starts;
    the worker must pair FMOs from that list, not touch the live sample."""
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.fmo])
    m['channels'] = list(CH)

    def off_thread(*_a):
        raise AssertionError('live sample read off the Tk thread')
    monkeypatch.setattr(R, 'member_channels', off_thread)
    R.build_fmo_controls(ed, [m])
    assert 'FL1-A' in m['fmo_controls']['controls']


def test_a_unit_flags_events_but_clusters_on_the_same_channels(files):
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.fmo])
    R.build_run_controls(ed, [m])
    prep = R.prepare_unit(ed, [m], 'u', {'channels': CH})
    assert prep['channels'] == CH
    assert 'FL1-A_pos' in prep['data'].columns
    assert [r['channel'] for r in prep['fmo_rows']] == ['FL1-A']


def test_the_cluster_table_gets_pct_pos_and_a_phenotype(tmp_path):
    data = pd.DataFrame({'FL1-A': np.arange(8.0), 'FL2-A': np.arange(8.0),
                         'cluster': [0, 0, 0, 0, 1, 1, 1, 1],
                         'FL1-A_pos': [1, 1, 1, 0, 0, 0, 0, np.nan]})
    s = FlowSample.from_dataframe(data, name='u')
    res = {'files': []}
    R._write_cluster_tables(s, data['cluster'], 8, str(tmp_path), 'u', None, res,
                            fmo_channels=['FL1-A'])
    table = pd.read_csv(tmp_path / 'u_clusters.csv')
    assert table['pct_pos_FL1-A'].tolist() == [75.0, 0.0]
    assert table['phenotype'].tolist() == ['FL1-A+', 'FL1-A-']


def test_the_run_writes_the_thresholds_table(tmp_path):
    res = {'files': []}
    R._write_fmo_thresholds([{'member': 's1', 'channel': 'FL1-A',
                              'fmo_file': 'FMO_FL1.fcs', 'percentile': 99.0,
                              'threshold_linear': 412.0,
                              'fmo_events_inside': 1600, 'source': 'group',
                              'pct_pos': 50.1}], str(tmp_path), 'u', res)
    t = pd.read_csv(tmp_path / 'u_fmo_thresholds.csv')
    assert t.loc[0, 'threshold_linear'] == 412.0
    assert res['files'] == [str(tmp_path / 'u_fmo_thresholds.csv')]
    R._write_fmo_thresholds([], str(tmp_path), 'v', res)
    assert not (tmp_path / 'v_fmo_thresholds.csv').exists()


def test_run_takes_the_fmo_set_and_reads_it_off_the_tk_thread(
        files, tmp_path, monkeypatch):
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
        from openflo.workspace_run import WorkspaceModel
        monkeypatch.chdir(tmp_path)
        live = _live(files.stained)
        ed = SimpleNamespace(
            _samples={'s1': live}, _sample_order=['s1'],
            _sample_trial={'s1': 'T'}, _sample_colors={'s1': '#000000'},
            _sample_gates={'s1': {}}, _sample_gate_order={'s1': []},
            _channel_labels={})
        panel = W.WorkspacePanel(r, editor=ed)
        mdl = WorkspaceModel()
        g = mdl.new_group('G')
        mdl.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None,
                      'path': files.stained}, gid=g)
        mdl.set_group_fmo(g, {'files': [files.fmo]})
        panel.model.groups, panel.model.loose = mdl.groups, mdl.loose
        seen = []
        real = R.build_run_controls
        monkeypatch.setattr(W, 'build_run_controls', lambda *a: (
            seen.append(threading.current_thread() is threading.main_thread()),
            real(*a)))
        monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
        panel._run()
        panel._beads.join(30)
        assert seen == [False]
        (job,) = list(panel._jobs)
        (member,) = job['unit']['members']
        assert member['fmo'] == {'files': [files.fmo], 'source': 'group'}
        assert 'FL1-A' in member['fmo_controls']['controls']
    finally:
        r.destroy()
