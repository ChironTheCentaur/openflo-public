"""Pipeline FMO sets (scope audit L3c), from its review: the paths the
first tests left open. A member's FMO is compensated with the member's
non-identity matrix; old ``<channel>_pos`` columns are no FMO call; the
read cache keys on every matrix value and the channels; QC follows the
member; each skip (unknown scale, empty population, missing file,
ambiguous name) is named; the run writes the thresholds table; a row's
own FMO set is reported as the row's. Generated data, generic channels.
"""
from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace_run as R
from openflo.flow_sample import FlowSample
from tests import test_workspace_run_fmo as F
from tests.conftest import gui_unavailable
from tests.test_workspace_run_fmo import CH, _live, _member, _write

files = F.files      # the generated stained / FMO / unrelated files

SPILL = np.array([[1.0, 0.5], [0.0, 1.0]])   # FL1-A spills 50% into FL2-A


def _cluster_table(s, tmp_path, fmo_channels=('FL1-A',)):
    R._write_cluster_tables(s, s.data['cluster'], len(s.data), str(tmp_path),
                            'u', None, {'files': []},
                            fmo_channels=list(fmo_channels))
    return pd.read_csv(tmp_path / 'u_clusters.csv')


# -- B1: a column the sample already had is no FMO call -----------------

def test_an_old_pos_column_without_an_fmo_is_not_reported(files, tmp_path):
    live = _live(files.stained)
    live.data['FL2-A_pos'] = live.data['FL2-A'] > 0      # a threshold gate's
    ed = SimpleNamespace(_samples={'s1': live})
    m = _member(files.stained, [])
    m.pop('fmo')
    R.build_run_controls(ed, [m])
    prep = R.prepare_unit(ed, [m], 'u', {'channels': CH})
    s = FlowSample.from_dataframe(prep['data'].copy(), name='u')
    s.data['cluster'] = np.arange(len(s.data)) % 2
    fmo_chs = sorted({r['channel'] for r in prep['fmo_rows']})
    table = _cluster_table(s, tmp_path, fmo_chs)
    assert 'phenotype' not in table.columns, table.columns.tolist()
    assert not any(c.startswith('pct_pos_') for c in table.columns)


def test_an_old_flag_is_not_pooled_with_another_members_fmo_call(files):
    """Member s1 (live) carries an old FL1-A_pos and no FMO; member s2
    gets the run's FL1-A_pos. s1's rows must hold NaN, not its old flag."""
    live = _live(files.stained)
    live.data['FL1-A_pos'] = True
    ed = SimpleNamespace(_samples={'s1': live, 's2': _live(files.stained)})
    a = _member(files.stained, [])
    a.pop('fmo')
    b = _member(files.stained, [files.fmo])
    b['item']['sample'] = b['sample'] = 's2'
    R.build_run_controls(ed, [a, b])
    prep = R.prepare_unit(ed, [a, b], 'u', {'channels': CH, 'max_events': 0})
    d = prep['data']
    src = d['__sample__'].astype(str)
    a_rows, b_rows = src.str.endswith('s1'), src.str.endswith('s2')
    assert a_rows.sum() == b_rows.sum() == 2000
    assert d.loc[a_rows, 'FL1-A_pos'].isna().all()
    assert d.loc[b_rows, 'FL1-A_pos'].notna().all()
    assert [r['member'] for r in prep['fmo_rows']] == ['s2']


def test_mixed_bool_and_float_flags_are_counted(tmp_path):
    data = pd.DataFrame({'cluster': [0, 0, 1, 1], 'FL1-A': [1.0, 2, 3, 4],
                         'FL1-A_pos': pd.Series([True, False, 1.0, np.nan],
                                                dtype=object)})
    t = _cluster_table(FlowSample.from_dataframe(data, name='u'), tmp_path)
    assert t['pct_pos_FL1-A'].tolist() == [50.0, 100.0]


# -- B2: the read cache --------------------------------------------------

def _counting_loads(monkeypatch):
    calls = []
    monkeypatch.setattr(R, 'load_fmo_control',
                        lambda p, comp, qc: calls.append((comp, qc)) or object())
    return calls


def test_large_matrices_differing_mid_matrix_are_read_apart(files, monkeypatch):
    calls = _counting_loads(monkeypatch)
    b = np.eye(40)
    b[20, 21] = 0.3                 # repr() of a 40x40 array elides this
    ms = []
    for mat in (np.eye(40), b):
        m = _member(files.stained, [files.fmo])
        m['comp'] = dict(m['comp'], matrix=mat)
        m['channels'] = list(CH)
        ms.append(m)
    R.build_fmo_controls(SimpleNamespace(_samples={}), ms)
    assert len(calls) == 2


def test_one_matrix_over_two_channel_orders_is_read_twice(files, monkeypatch):
    calls = _counting_loads(monkeypatch)
    ms = []
    for chans in (['FL1-A', 'FL2-A'], ['FL2-A', 'FL1-A']):
        m = _member(files.stained, [files.fmo])
        m['comp'] = dict(m['comp'], matrix=SPILL.copy(), channels=chans)
        m['channels'] = list(CH)
        ms.append(m)
    R.build_fmo_controls(SimpleNamespace(_samples={}), ms)
    assert len(calls) == 2


def test_qc_is_part_of_the_key_and_passed(files, monkeypatch):
    calls = _counting_loads(monkeypatch)
    ms = []
    for qc in (True, False):
        m = _member(files.stained, [files.fmo])
        m['item']['qc_applied'] = qc
        m['channels'] = list(CH)
        ms.append(m)
    R.build_fmo_controls(SimpleNamespace(_samples={}), ms)
    assert [qc for _c, qc in calls] == [True, False]


def test_a_member_whose_comp_failed_reads_no_fmo(files, monkeypatch):
    calls = _counting_loads(monkeypatch)
    m = _member(files.stained, [files.fmo])
    m['comp'] = dict(m['comp'], error='planted')
    m['channels'] = list(CH)
    R.build_fmo_controls(SimpleNamespace(_samples={}), [m])
    assert calls == [] and 'fmo_controls' not in m


# -- B3: the member's compensation, end to end ---------------------------

def test_the_fmo_is_compensated_with_the_members_matrix(tmp_path):
    """An FL2-A FMO with bright FL1-A spilling 50% into FL2-A: compensated
    with the member's matrix its FL2-A is background (cutoff near 300);
    read without it, the spill (~10000) would set the cutoff."""
    rng = np.random.default_rng(5)
    n = 2000
    fsc = rng.uniform(5e4, 2e5, n)
    fl1 = rng.normal(20000, 1500, n)
    fl2 = rng.normal(300, 40, n) + 0.5 * fl1               # measured
    stained = _write(tmp_path / 's1.fcs', fsc, fl1, fl2)
    fmo = _write(tmp_path / 'FMO_FL2.fcs', fsc, fl1, fl2)
    live = FlowSample(stained)
    live.name = 's1'
    live.manual_compensate(SPILL, CH)
    live.apply_transform(CH)
    ed = SimpleNamespace(_samples={'s1': live})
    m = _member(stained, [fmo])
    m['comp'] = dict(m['comp'], matrix=SPILL.copy(), source='row')
    m['item']['comp_matrix'] = SPILL.copy()
    R.build_run_controls(ed, [m])
    s, _n = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
    (row,) = R.apply_fmo_controls(s, m, 99.0, [])
    assert row['channel'] == 'FL2-A'
    assert row['threshold_linear'] < 1000, row     # BP15: background + 3 SD is ~420
    assert row['pct_pos'] < 5


def test_a_run_writes_the_thresholds_and_the_phenotype(files, tmp_path):
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.fmo])
    R.build_run_controls(ed, [m])
    cfg = R.default_run_cfg()
    cfg.update(method='phenograph', umap=False, tsne=False, phate=False,
               trimap=False, pacmap=False, max_events=0)
    prep = R.prepare_unit(ed, [m], 'u', dict(cfg, channels=CH))
    _so = sys.stdout
    sys.stdout = io.StringIO()                 # silence cluster chatter
    try:
        res = R.compute_run(dict(prep), cfg, str(tmp_path))
    finally:
        sys.stdout = _so
    assert res['ok'], res
    t = pd.read_csv(tmp_path / 'u_fmo_thresholds.csv')
    assert t['channel'].tolist() == ['FL1-A']
    assert str(tmp_path / 'u_fmo_thresholds.csv') in res['files']
    clusters = pd.read_csv(tmp_path / 'u_clusters.csv')
    assert 'pct_pos_FL1-A' in clusters.columns
    assert clusters['phenotype'].str.contains('FL1-A').all()


def test_a_member_value_that_is_not_finite_is_unknown_not_negative(files):
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.fmo])
    R.build_run_controls(ed, [m])
    s, _n = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
    s.data.loc[s.data.index[:10], 'FL1-A'] = np.nan
    R.apply_fmo_controls(s, m, 99.0, [])
    flag = s.data['FL1-A_pos'].to_numpy()
    assert np.isnan(flag[:10]).all()
    assert np.isfinite(flag[10:]).all()


@pytest.mark.parametrize('qc', [True, False])
def test_qc_on_the_fmo_follows_the_member(files, monkeypatch, qc):
    ran = []
    monkeypatch.setattr(FlowSample, 'run_qc', lambda self, *a, **k: ran.append(1))
    R.load_fmo_control(files.fmo, None, qc)
    assert ran == ([1] if qc else [])


# -- B3: each skip is named ---------------------------------------------

def _apply(fx, fmo_files, gated=False):
    ed = SimpleNamespace(_samples={'s1': _live(fx.stained)})
    m = _member(fx.stained, fmo_files, gated=gated)
    R.build_run_controls(ed, [m])
    s, _n = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
    warned: list = []
    return m, s, R.apply_fmo_controls(s, m, 99.0, warned), warned


def test_a_gate_on_a_channel_of_unknown_scale_skips_the_fmo(files, monkeypatch):
    from openflo import fmo_gater
    real = fmo_gater.in_stained_coordinates
    monkeypatch.setattr(fmo_gater, 'in_stained_coordinates',
                        lambda fs, s: (real(fs, s)[0], ['FSC-A']))
    _m, s, rows, warned = _apply(files, [files.fmo], gated=True)
    assert rows == [] and 'FL1-A_pos' not in s.data.columns
    assert any('FMO_FL1.fcs cannot be read through the population' in w
               for w in warned), warned


def test_an_fmo_with_no_events_in_the_population_is_skipped(files):
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.fmo], gated=True)
    m['item']['gates']['g1']['value'] = 1e12           # nothing passes
    R.build_run_controls(ed, [m])
    s, _n = R.resolve_run_sample(ed, m['item'], comp=m['comp'])
    warned: list = []
    assert R.apply_fmo_controls(s, m, 99.0, warned) == []
    assert any('FMO_FL1.fcs has no events inside the population' in w
               for w in warned), warned


def test_a_missing_fmo_file_is_named(files, tmp_path):
    gone = str(tmp_path / 'gone' / 'FMO_FL1.fcs')
    _m, _s, rows, warned = _apply(files, [gone])
    assert rows == []
    assert any('FMO FMO_FL1.fcs not found, skipped' in w for w in warned), warned


def test_an_fmo_naming_two_channels_is_named_once(files, tmp_path):
    import shutil
    both = str(tmp_path / 'FL1_FL2.fcs')
    shutil.copy(files.fmo, both)
    _m, _s, rows, warned = _apply(files, [both])
    assert rows == []
    hits = [w for w in warned if 'FL1_FL2.fcs fits FL1-A and FL2-A equally' in w]
    assert len(hits) == 1, warned


# -- cutoff, percentile, phenotype --------------------------------------

def test_the_cutoff_is_the_percentile_of_linear_values_inside(files):
    """Independent check: numpy's 99th percentile of the raw FMO FL1-A
    values with FSC-A > 4e4 (identity compensation, linear values)."""
    import flowio
    _m, s, rows, _w = _apply(files, [files.fmo], gated=True)
    (row,) = rows
    f = flowio.FlowData(files.fmo)
    ev = np.reshape(np.asarray(f.events, dtype=float), (-1, f.channel_count))
    inside = ev[:, 0] > 4e4
    want = np.percentile(ev[inside, 1], 99.0)
    # BP15: float32 file values through the transform round trip.
    assert row['threshold_linear'] == pytest.approx(want, rel=1e-6)
    assert row['fmo_events_inside'] == int(inside.sum())
    from openflo.channel_transforms import linear_values
    v = np.asarray(linear_values(s, 'FL1-A'), float)
    np.testing.assert_array_equal(s.data['FL1-A_pos'].to_numpy(),
                                  (v >= row['threshold_linear']).astype(float))


def test_the_runs_percentile_is_used(files):
    ed = SimpleNamespace(_samples={'s1': _live(files.stained)})
    m = _member(files.stained, [files.fmo], gated=True)
    R.build_run_controls(ed, [m])
    prep = R.prepare_unit(ed, [m], 'u', {'channels': CH, 'fmo_percentile': 50.0})
    (row,) = prep['fmo_rows']
    assert row['percentile'] == 50.0


def test_a_cluster_with_no_fmo_member_has_no_phenotype(tmp_path):
    data = pd.DataFrame({'FL1-A': np.arange(6.0), 'cluster': [0, 0, 0, 1, 1, 1],
                         'FL1-A_pos': [1, 1, np.nan, np.nan, np.nan, np.nan]})
    t = _cluster_table(FlowSample.from_dataframe(data, name='u'), tmp_path)
    assert t['pct_pos_FL1-A'].tolist()[0] == 100.0
    assert np.isnan(t['pct_pos_FL1-A'].tolist()[1])
    assert t['phenotype'].fillna('').tolist() == ['FL1-A+', '']


def test_half_positive_is_positive(tmp_path):
    data = pd.DataFrame({'FL1-A': np.arange(4.0), 'cluster': [0, 0, 1, 1],
                         'FL1-A_pos': [1, 0, 0, 0]})
    t = _cluster_table(FlowSample.from_dataframe(data, name='u'), tmp_path)
    assert t['phenotype'].tolist() == ['FL1-A+', 'FL1-A-']


# -- the view: a row's own set, and channels read on the Tk thread ------

def test_a_rows_own_fmo_set_is_the_rows(files, tmp_path, monkeypatch):
    pytest.importorskip('tkinter')
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
        ed = SimpleNamespace(
            _samples={'s1': _live(files.stained)}, _sample_order=['s1'],
            _sample_trial={'s1': 'T'}, _sample_colors={'s1': '#000000'},
            _sample_gates={'s1': {}}, _sample_gate_order={'s1': []},
            _channel_labels={})
        panel = W.WorkspacePanel(r, editor=ed)
        mdl = WorkspaceModel()
        g = mdl.new_group('G')
        mid = mdl.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None,
                            'path': files.stained}, gid=g)
        mdl.set_item_fmo(mid, {'files': [files.fmo]})
        panel.model.groups, panel.model.loose = mdl.groups, mdl.loose
        on_entry = []
        real = R.build_run_controls

        def spy(editor, members):
            on_entry.append([list(m.get('channels') or []) for m in members])
            return real(editor, members)
        monkeypatch.setattr(W, 'build_run_controls', spy)
        monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
        panel._run()
        panel._beads.join(30)
        assert on_entry == [[CH]]          # read before the worker started
        (job,) = list(panel._jobs)
        (member,) = job['unit']['members']
        assert member['fmo']['source'] == 'row'
        assert member['fmo_controls']['source'] == 'row'
    finally:
        r.destroy()
