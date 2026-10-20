"""Result stamps: which OpenFlo version produced each saved result
(result_stamp), so a later fix registry can find results made before a fix.

Covered: the stamp shape and version order; the v2 -> v3 session migration;
the plot cache ignoring a gate's stamp; audit entries carrying the version;
and, in a real editor, the stamp each producing step writes (gate re-fit,
auto-gate, clustering + embedding, cell cycle, calibration) and a session
round trip that writes an older result's stamp back unchanged.

All data is synthetic; channels are FL1-A, FL2-A, ...
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import result_stamp as rs
from openflo.session_format import SESSION_FORMAT, migrate_session

OLD = '2.0.0'          # a version older than any this build can be


def _bare(st):
    """A gate's stamp without the geometry digest it is sealed with
    (result_stamp.seal_gate_stamp; tested in test_fixes)."""
    return {k: v for k, v in st.items() if k != rs.GEOM_KEY}


@pytest.fixture
def now(monkeypatch):
    """Pin the running version, so stamps made in a test are recognisable."""
    monkeypatch.setattr(rs, 'app_version', lambda: '9.9.9')
    return '9.9.9'


# ── The stamp ───────────────────────────────────────────────────────────────

def test_a_new_stamp_names_the_step_version_and_inputs(now):
    st = rs.make_stamp('gate.refit', ['g2', 'FL1-A', None, 'FL2-A'])
    assert st == {'step': 'gate.refit', 'version': now,
                  'inputs': ['g2', 'FL1-A', 'FL2-A']}
    assert json.loads(json.dumps(st)) == st
    g = rs.stamp_gate({'kind': 'threshold', 'channel': 'FL1-A'}, 'x', [])
    assert _bare(rs.stamp_of(g)) == {'step': 'x', 'version': now,
                                     'inputs': []}


def test_the_running_version_is_update_current_version():
    from openflo.update import current_version
    assert rs.app_version() == str(current_version())


def test_a_stamp_without_a_version_is_unknown_not_now(now):
    assert rs.stamp_of({'stamp': {'step': 'calibration'}}) == {
        'step': 'calibration', 'version': None, 'inputs': []}
    assert rs.stamp_of({'kind': 'rect'}) is None             # no stamp
    assert rs.stamp_of({'stamp': {'version': '1.0'}}) is None  # no step
    assert rs.stamp_of({'stamp': 'garbage'}) is None
    assert rs.stamp_of(SimpleNamespace(stamp={'step': 's', 'version': '1'}))[
        'version'] == '1'


@pytest.mark.parametrize('a, b, want', [
    ('2.6.2', '2.7.0', -1), ('2.7.0', '2.6.2', 1), ('2.7.0', '2.7.0', 0),
    ('2.6.2', '2.7.0b1', -1),      # a fix in a beta is newer than 2.6.2
    ('2.7.0b1', '2.7.0', -1), ('2.7.0b2', '2.7.0b1', 1),
    (None, '0.1', -1), ('', '0.1', -1), ('0.1', None, 1), (None, None, 0),
])
def test_versions_order_with_unknown_below_everything(a, b, want):
    assert rs.compare_versions(a, b) == want


def test_made_before_flags_older_and_unknown_results():
    old = {'stamp': {'step': 's', 'version': OLD}}
    assert rs.made_before(old, '2.7.0')
    assert not rs.made_before(old, OLD)
    assert rs.made_before({'kind': 'rect'}, '0.0.1')      # never stamped
    assert rs.made_before(None, '0.0.1')
    assert not rs.made_before('3.0.0', '2.7.0')


def test_column_stamps_keep_only_the_columns_still_there(now):
    s = SimpleNamespace(data=pd.DataFrame({'FL1-A': [1.0], 'leiden': [0]}))
    rs.stamp_columns(s, ['leiden', 'UMAP1'], 'cluster.leiden', ['FL1-A'])
    assert set(s.result_stamps) == {'leiden', 'UMAP1'}
    assert rs.column_stamps(s) == {'leiden': {
        'step': 'cluster.leiden', 'version': now, 'inputs': ['FL1-A']}}
    assert rs.stamps_from_entry({'a': {'step': 'x'}, 'b': 5, 'c': {}}) == {
        'a': {'step': 'x', 'version': None, 'inputs': []}}
    assert rs.stamps_from_entry(None) == {}


# ── Session format v3 ───────────────────────────────────────────────────────

def _v2(**extra):
    return {'format': SESSION_FORMAT, 'version': 2,
            'samples': [{'name': 's1', 'path': 's1.fcs'}],
            'sample_gates': {}, **extra}


def test_v2_session_migrates_to_unknown_versions_and_no_stamps():
    data, notes = migrate_session(_v2())
    assert notes == ['upgraded session schema v2 → v3']
    assert data['version'] == 3 and data['app_version'] is None
    assert data['samples'][0]['app_version'] is None
    assert data['samples'][0]['result_stamps'] == {}


def test_v2_to_v3_is_idempotent_and_keeps_what_is_there():
    from openflo.session_format import _v2_to_v3
    st = {'leiden': {'step': 'cluster.leiden', 'version': OLD, 'inputs': []}}
    d = _v2(app_version=OLD)
    d['samples'][0].update(app_version=OLD, result_stamps=st)
    once = _v2_to_v3(json.loads(json.dumps(d)))
    assert _v2_to_v3(json.loads(json.dumps(once))) == once
    assert once['app_version'] == OLD
    assert once['samples'][0]['result_stamps'] == st


# ── Plot cache ──────────────────────────────────────────────────────────────

def test_a_gate_stamp_does_not_change_the_plot_cache_signature():
    from openflo.plotcache import chain_sigs
    gates = {'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0,
                    'parent_id': None},
             'g2': {'kind': 'threshold', 'channel': 'FL2-A', 'value': 2.0,
                    'parent_id': 'g1'}}
    before = chain_sigs(gates)
    gates['g1']['stamp'] = {'step': 'gate.refit', 'version': OLD,
                            'inputs': []}
    assert chain_sigs(gates) == before
    gates['g1']['value'] = 1.5                   # geometry still counts
    assert chain_sigs(gates)['g2'] != before['g2']


# ── Audit trail ─────────────────────────────────────────────────────────────

def test_audit_entries_carry_the_version_and_keep_it_on_reload():
    from openflo.audit import AuditLog
    log = AuditLog.from_list(
        [{'seq': 1, 'action': 'old.step', 'details': {}},
         {'seq': 2, 'action': 'older.version', 'details': {},
          'version': OLD}], version='9.9.9')
    log.record('gate.add', kind='rect', version_note='a detail')
    e = log.to_list()
    assert 'version' not in e[0]                      # recorded before
    assert e[1]['version'] == OLD                      # kept as it was
    assert e[2]['version'] == '9.9.9'
    assert e[2]['details'] == {'kind': 'rect', 'version_note': 'a detail'}
    again = AuditLog.from_list(json.loads(json.dumps(e))).to_list()
    assert again == e
    assert 'version' not in AuditLog().record('x')     # no version set


# ── In the editor ────────────────────────────────────────────────────────────

def _editor(root):
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
        on_done(work()) if on_done else work())
    return ed


@pytest.fixture
def tk_root(tmp_path, monkeypatch):
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    yield root
    root.destroy()


def _add(ed, df, name='s1', path=None):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(df, name=name, path=path or f'{name}.fcs')
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()
    ed._set_active_sample(name)
    return s


def test_refit_stamps_each_replaced_gate(tk_root, now):
    from tests.test_gate_refit import _drifting
    ed = _editor(tk_root)
    shifts, ids = _drifting(ed)
    old = ed._sample_gates['s2'][ids['s2']]
    old['stamp'] = {'step': 'gate.auto.threshold', 'version': OLD,
                    'inputs': []}
    assert ed._refit_gate_per_sample('s1', ids['s1'], review=False) == 3
    for s in shifts:
        g = ed._sample_gates[s][ids[s]]
        assert _bare(g['stamp']) == {'step': 'gate.refit', 'version': now,
                                     'inputs': [g['parent_id'], 'FL2-A']}


def test_auto_gate_gates_are_stamped_with_their_method(tk_root, now):
    ed = _editor(tk_root)
    rng = np.random.default_rng(12)
    pts = np.vstack([rng.normal([0, 0], 0.3, (2000, 2)),
                     rng.normal([3, 3], 0.3, (2000, 2))])
    _add(ed, pd.DataFrame({'FL1-A': pts[:, 0], 'FL2-A': pts[:, 1]}))
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')
    ed._auto_gate_gmm('s1', {'max_components': 3, 'coverage': 0.9,
                             'min_weight': 0.02})
    gmm = list(ed._gates.values())
    assert gmm and all(_bare(g['stamp']) == {'step': 'gate.auto.gmm',
                                      'version': now,
                                      'inputs': ['FL1-A', 'FL2-A']}
                       for g in gmm)
    parent = next(iter(ed._gates))
    ed._auto_gate_threshold('s1', parent)
    thr = [g for g in ed._gates.values() if g['kind'] == 'threshold']
    assert _bare(thr[0]['stamp']) == {'step': 'gate.auto.threshold',
                                      'version': now,
                                      'inputs': [parent, 'FL1-A']}


def test_clustering_and_embedding_columns_are_stamped(tk_root, now):
    ed = _editor(tk_root)
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(8.0 * i, 0.4, (150, 2)) for i in range(3)])
    s = _add(ed, pd.DataFrame(X, columns=['FL1-A', 'FL2-A']))

    def fake_tsne(**kw):
        s.data['TSNE1'] = 0.0
        s.data['TSNE2'] = 0.0
    s.run_tsne = fake_tsne
    ed._run_clustering(method='flowsom', all_samples=False, k=30, grid=4,
                       n_meta=3, embedding='t-SNE', channels=['FL2-A'])
    st = s.result_stamps
    want = {'step': 'cluster.flowsom', 'version': now, 'inputs': ['FL2-A']}
    assert st['flowsom'] == want and st['flowsom_meta'] == want
    fluor = [str(c) for c in s.fluor_channels]
    for c in ('TSNE1', 'TSNE2'):
        assert st[c] == {'step': 'embed.tsne', 'version': now,
                         'inputs': fluor}
    assert ed._audit_log.to_list()[-1]['version'] == now


def test_cell_cycle_and_calibration_columns_are_stamped(tk_root, now):
    ed = _editor(tk_root)
    rng = np.random.default_rng(0)
    dna = np.concatenate([rng.normal(50_000, 2_500, 6000),
                          rng.uniform(57_500, 92_500, 1000),
                          rng.normal(100_000, 4_000, 2000)])
    s = _add(ed, pd.DataFrame({'FSC-A': np.full(dna.size, 8e4),
                               'FL3-A': dna}))
    ed._sample_gates.setdefault('s1', {})
    model = s.cell_cycle_model(dna_channel='FL3-A', method='boundaries')
    ed._finish_cell_cycle('FL3-A', {'s1': s.data}, {'s1': model}, [])
    assert s.result_stamps['cell_cycle'] == {
        'step': 'cell_cycle', 'version': now,
        'inputs': ['FL3-A', f"method={s.cell_cycle_result['method']}"]}

    from tests.test_calibration_dialog import _frames, _register, _run_dialog
    ed2 = _editor(tk_root)
    beads, cells = _frames()
    _register(ed2, 'beads', beads)
    _register(ed2, 'cells', cells)
    _run_dialog(ed2)
    for nm in ('beads', 'cells'):
        assert ed2._samples[nm].result_stamps == {'MESF:FL4-A': {
            'step': 'calibration', 'version': now, 'inputs': ['FL4-A']}}


def _restore(root, path):
    ed2 = _editor(root)
    jobs = {'fcs': [], 'csv': []}
    ed2._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed2._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    ed2._load_session_path(str(path))
    for job in jobs['csv']:
        ed2._load_csv_worker(job[0], job[1], {})
    from tests.test_transform_records import _pump
    assert _pump(root, lambda: len(ed2._samples) == len(jobs['csv']) == 1)
    return ed2


def test_a_session_round_trip_writes_older_stamps_back_unchanged(
        tk_root, tmp_path, now):
    """Saved, reopened and saved again: the stamps of results an older
    version made stay that version's, while the session and the sample
    entry say which version wrote them."""
    ed = _editor(tk_root)
    rng = np.random.default_rng(4)
    df = pd.DataFrame({'FL1-A': rng.normal(0, 1, 300),
                       'FL2-A': rng.normal(0, 1, 300),
                       'leiden': rng.integers(0, 3, 300)})
    s = _add(ed, df, path=str(tmp_path / 's1.fcs'))
    old = {'step': 'cluster.leiden', 'version': OLD, 'inputs': ['FL1-A']}
    s.result_stamps = {'leiden': dict(old)}
    gate_old = {'step': 'gate.auto.threshold', 'version': OLD,
                'inputs': ['FL1-A']}
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.0,
                  'name': 'Pos', 'parent_id': None, 'stamp': dict(gate_old)})
    path = tmp_path / 'one.flowsession'
    data = ed._write_session(str(path))
    assert data['app_version'] == now
    entry = data['samples'][0]
    assert entry['app_version'] == now
    assert entry['result_stamps'] == {'leiden': old}

    ed2 = _restore(tk_root, path)
    s2 = ed2._samples['s1']
    assert s2.result_stamps == {'leiden': old}
    (g2,) = ed2._sample_gates['s1'].values()
    assert _bare(g2['stamp']) == gate_old
    again = ed2._write_session(str(tmp_path / 'two.flowsession'))
    assert again['samples'][0]['result_stamps'] == {'leiden': old}
    (saved_gate,) = again['sample_gates']['s1']
    assert _bare(saved_gate['stamp']) == gate_old


def test_pseudotime_columns_are_stamped(monkeypatch, now):
    from openflo.async_task import wait_for_job
    from tests.test_trajectory_window import _window
    root, ed, w, _truth = _window(monkeypatch)
    try:
        w._compute()
        assert wait_for_job(w._busy)
        root_ch = ed._resolve_channel(w.root_var.get())
        for name in ('Day1', 'Day3'):
            st = ed._samples[name].result_stamps['pseudotime']
            assert st['step'] == 'trajectory' and st['version'] == now
            assert st['inputs'][0] == f'root={root_ch}'
    finally:
        root.destroy()
