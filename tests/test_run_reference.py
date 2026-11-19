"""Frozen reference clusters (``openflo.run_reference``): a FlowSOM run's
map, frozen into a run template, labels every later run's events, warns
when they lie far from it, and adds each sample's share of each cluster to
the run's population table.

Generated data, generic channel and sample names only.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import run_history as H
from openflo import run_reference as F
from openflo import workspace_run as R
from openflo.flow_sample import FlowSample
from openflo.run_template import TEMPLATE_FORMAT
from tests.conftest import gui_unavailable

CH = ['FL1-A', 'FL2-A']
TF = {'FL1-A': {'method': 'asinh', 'cofactor': 150.0}}
CENTRES = np.array([[1.0, 1.0], [4.0, 1.0], [2.5, 4.0]])
KEYS = {'G / a': {'Group': 'G', 'Trial': 'Day', 'Sample': 'a'},
        'G / b': {'Group': 'G', 'Trial': 'Day', 'Sample': 'b'}}


def _blobs(shift=0.0, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 3, n)
    df = pd.DataFrame(CENTRES[lab] + rng.normal(0, 0.3, (n, 2)) + shift,
                      columns=CH)
    df['__group__'] = pd.Categorical(['G'] * n)
    df['__sample__'] = pd.Categorical(
        np.where(np.arange(n) < n // 2, 'G / a', 'G / b'))
    return df


@pytest.fixture(scope='module')
def som_run():
    s = FlowSample.from_dataframe(_blobs(), name='U')
    s.run_flowsom(channels=CH, n_metaclusters=3, max_events=3000, seed=42)
    return s, F.build_map(s.flowsom_result, s.data, TF, 'run_1 · U')


def _ref_rows():
    return [{'Group': 'G', 'Trial': 'Day', 'Sample': 'a',
             'Population': f'All events · Reference Cluster {i}',
             '%Parent': p} for i, p in enumerate((30.0, 30.0, 40.0))]


def _prep(data):
    return {'data': data, 'channels': CH, 'label': 'U',
            'data_transforms': dict(TF), 'ref_rows': _ref_rows(),
            'pop_rows': [{'Group': 'G', 'Trial': 'Day', 'Sample': 'a',
                          'Population': 'All events', 'Count': 1500}]}


def _cfg(**kw):
    return dict(R.default_run_cfg(), umap=False, n_metaclusters=3, **kw)


# ── the map ────────────────────────────────────────────────────────────────

def test_a_map_holds_what_labelling_needs(som_run):
    s, m = som_run
    assert F.check_map(json.loads(json.dumps(m))) == m
    assert m['channels'] == CH and len(m['weights']) == 100
    assert m['transforms'] == {'FL1-A': TF['FL1-A'], 'FL2-A': None}
    assert sorted(set(m['meta_of_node'])) == [0, 1, 2]
    W = np.asarray(m['weights'])
    X = s.data[CH].to_numpy()
    d = np.sqrt(((X[:, None, :] - W[None]) ** 2).sum(2)).min(1)
    assert m['far'] == pytest.approx(np.percentile(d, 95))
    assert m['n_events'] == 3000 and m['source'] == 'run_1 · U'


def test_a_non_finite_event_is_left_out_of_the_reference_distance(som_run):
    s, _m = som_run
    data = s.data.copy()
    data.loc[0, 'FL1-A'] = np.nan
    assert F.build_map(s.flowsom_result, data, TF, 'x')['n_events'] == 2999


def test_written_and_read_back(som_run, tmp_path):
    _s, m = som_run
    path = F.write_map(str(tmp_path), 'U', m)
    assert path == str(tmp_path / 'U_flowsom_map.json')
    assert F.read_map(path) == m


@pytest.mark.parametrize('change,words', [
    ({'format': 'other'}, 'not an OpenFlo FlowSOM map'),
    ({'channels': []}, 'names no channels'),
    ({'channels': ['FL1-A', 3]}, 'names no channels'),
    ({'weights': 'x'}, 'damaged'),
    ({'weights': [[1.0, 2.0, 3.0]]}, 'no node for each'),
    ({'weights': [1.0, 2.0]}, 'no node for each'),
    ({'weights': []}, 'no node for each'),
    ({'weights': [[1.0, float('nan')]]}, 'no node for each'),
    ({'meta_of_node': [0]}, 'no metacluster for each node'),
    ({'far': float('nan')}, 'no reference distance'),
    ({'far': -1.0}, 'no reference distance'),
    ({'far': None}, 'damaged'),
])
def test_a_damaged_map_is_refused(som_run, change, words):
    _s, m = som_run
    with pytest.raises(ValueError, match=words):
        F.check_map(dict(m, **change))
    with pytest.raises(ValueError, match='not an OpenFlo'):
        F.check_map([m])


# ── labelling a run ────────────────────────────────────────────────────────

def test_the_reference_run_labels_itself_as_flowsom_did(som_run):
    s, m = som_run
    labels, far = F.label_events(s.data, m, TF)
    assert (labels == s.data['flowsom_meta'].to_numpy()).all()
    # 5% by construction, within the 95th percentile's interpolation
    # between two of the 3000 events.
    assert far == pytest.approx(5.0, abs=0.1)


def test_a_new_day_on_the_same_populations_is_not_far(som_run):
    _s, m = som_run
    labels, far = F.label_events(_blobs(seed=7), m, TF)
    assert 2.0 < far < 8.0 and F.drift_warning(far) is None
    assert set(labels) == {0, 1, 2}


def test_a_shifted_day_is_far_and_warns(som_run):
    _s, m = som_run
    _labels, far = F.label_events(_blobs(shift=1.5, seed=7), m, TF)
    assert far > 50
    assert F.drift_warning(far).startswith(f'{far:.0f}% of events lie farther')


def test_a_non_finite_event_is_unclustered(som_run):
    _s, m = som_run
    data = _blobs(seed=7)
    data.loc[3, 'FL2-A'] = np.inf
    labels, _far = F.label_events(data, m, TF)
    assert labels[3] == -1 and (np.delete(labels, 3) >= 0).all()
    data[CH] = np.nan
    labels, far = F.label_events(data, m, TF)
    assert (labels == -1).all() and far == 0.0


def test_the_warning_starts_above_ten_percent():
    assert F.drift_warning(10.0) is None
    assert '(about 5% expected)' in F.drift_warning(10.5)


def test_a_missing_channel_or_another_scale_is_refused(som_run):
    _s, m = som_run
    with pytest.raises(ValueError, match='lacks: FL2-A'):
        F.label_events(_blobs().drop(columns='FL2-A'), m, TF)
    with pytest.raises(ValueError, match='FL1-A is on logicle here but on '
                       'asinh in the frozen reference'):
        F.label_events(_blobs(), m, {'FL1-A': {'method': 'logicle'}})
    with pytest.raises(ValueError, match='FL2-A is on asinh here but on linear'):
        F.label_events(_blobs(), m, dict(TF, **{'FL2-A': {'method': 'asinh'}}))
    with pytest.raises(ValueError, match='damaged'):
        F.label_events(_blobs(), dict(m, far='x'), TF)


def test_a_members_share_of_each_cluster_zero_included(som_run):
    _s, m = som_run
    rows = F.member_rows([0, 0, 2, 2], m, KEYS['G / a'], 'Cells/Big')
    assert [(r['Sample'], r['Population'], r['%Parent']) for r in rows] == [
        ('a', 'Cells/Big · Reference Cluster 0', 50.0),
        ('a', 'Cells/Big · Reference Cluster 1', 0.0),
        ('a', 'Cells/Big · Reference Cluster 2', 50.0)]
    assert rows[0]['Group'] == 'G' and rows[0]['Trial'] == 'Day'
    noisy = F.member_rows([-1, 0], m, KEYS['G / a'], 'Cells')
    assert noisy[0]['Population'] == 'Cells · Reference Unclustered (noise)'
    assert noisy[0]['%Parent'] == 50.0 and len(noisy) == 4
    assert F.member_rows([], m, KEYS['G / a'], 'Cells') == []


def test_the_maps_channel_order_is_the_one_used(som_run):
    _s, m = som_run
    flipped = dict(m, channels=CH[::-1],
                   weights=np.asarray(m['weights'])[:, ::-1].tolist())
    data = _blobs(seed=7)
    assert (F.label_events(data, flipped, TF)[0]
            == F.label_events(data, m, TF)[0]).all()


def test_a_scale_record_in_another_key_order_is_the_same_scale(som_run):
    _s, m = som_run
    reordered = {'FL1-A': dict(reversed(list(TF['FL1-A'].items())))}
    assert list(reordered['FL1-A']) != list(TF['FL1-A'])
    data = _blobs(seed=7)
    assert (F.label_events(data, m, reordered)[0]
            == F.label_events(data, m, TF)[0]).all()


# ── in a run ───────────────────────────────────────────────────────────────

def test_a_flowsom_run_writes_its_map(tmp_path):
    out = str(tmp_path / 'run_1')
    res = R.compute_run(_prep(_blobs()), _cfg(method='flowsom'), out)
    assert res['ok'], res['error']
    path = os.path.join(out, 'U_flowsom_map.json')
    assert path in res['files']
    m = F.read_map(path)
    assert m['source'] == 'run_1 · U' and m['transforms']['FL1-A'] == TF['FL1-A']
    other = str(tmp_path / 'run_2')
    assert R.compute_run(_prep(_blobs()), _cfg(method='leiden'), other)['ok']
    assert not os.path.exists(os.path.join(other, 'U_flowsom_map.json'))


def test_a_run_with_a_reference_labels_by_it(som_run, tmp_path):
    _s, m = som_run
    out = str(tmp_path / 'run_2')
    res = R.compute_run(_prep(_blobs(seed=7)), _cfg(reference=m), out)
    assert res['ok'], res['error']
    assert res['method'] == 'flowsom' and res['n_clusters'] == 3
    assert res['reference']['source'] == 'run_1 · U'
    assert 2.0 < res['reference']['far_pct'] < 8.0
    assert 'frozen reference clusters (' in res['note']
    assert not any('farther' in w for w in res['warnings'])
    assert not os.path.exists(os.path.join(out, 'U_flowsom_map.json'))
    table = pd.read_csv(os.path.join(out, 'U_population_stats.csv'))
    assert list(table['Population']) == [
        'All events'] + [r['Population'] for r in _ref_rows()]
    assert list(table['%Parent'])[1:] == [30.0, 30.0, 40.0]
    path = os.path.join(out, 'U_population_stats.csv')
    assert res['files'].count(path) == 1


def test_a_reference_run_without_a_table_still_reports_its_clusters(
        som_run, tmp_path):
    _s, m = som_run
    out = str(tmp_path / 'r')
    res = R.compute_run(dict(_prep(_blobs()), pop_rows=[]), _cfg(reference=m),
                        out)
    assert res['ok'], res['error']
    table = pd.read_csv(os.path.join(out, 'U_population_stats.csv'))
    assert len(table) == 3


def test_a_shifted_run_warns(som_run, tmp_path):
    _s, m = som_run
    res = R.compute_run(_prep(_blobs(shift=1.5, seed=7)), _cfg(reference=m),
                        str(tmp_path / 'r'))
    assert res['ok'] and any('farther from the frozen reference' in w
                             for w in res['warnings'])


def test_a_reference_the_run_cannot_use_fails_the_unit(som_run, tmp_path):
    _s, m = som_run
    prep = dict(_prep(_blobs()), data_transforms={})
    res = R.compute_run(prep, _cfg(reference=m), str(tmp_path / 'r'))
    assert not res['ok'] and 'FL1-A is on linear here' in res['error']


def _fcs_members(tmp_path):
    """One file twice in a unit: all its events, and its FL1-A high events
    (a parent and its child)."""
    from tests._fcs_writer import write_float_fcs
    rng = np.random.default_rng(11)
    lab = rng.integers(0, 3, 2000)
    ev = 1000.0 * (CENTRES[lab] + rng.normal(0, 0.3, (2000, 2)))
    path = write_float_fcs(str(tmp_path / 'a.fcs'), ev, CH)
    base = {'sample': 'a', 'trial': 'Day 1', 'path': path, 'qc_applied': False,
            'comp_matrix': None, 'comp_channels': []}
    whole = dict(base, gate_id=None, gate_chain=[], gates={})
    gate = {'kind': 'threshold', 'name': 'High', 'channel': 'FL1-A',
            'value': 0.0, 'id': 'g1', 'parent_id': None}
    high = dict(base, gate_id='g1', gate_chain=['g1'], gates={'g1': gate},
                gate_path='High')
    return [{'item': whole, 'group': 'G', 'sample': 'a'},
            {'item': high, 'group': 'G', 'sample': 'a'}]


def test_prepare_unit_tags_each_members_events(tmp_path):
    prep = R.prepare_unit(SimpleNamespace(_samples={}), _fcs_members(tmp_path),
                          'G', R.default_run_cfg())
    assert set(prep['data']['__sample__']) == {'G / a'}
    assert prep['ref_rows'] == []


def test_each_population_of_one_file_keeps_its_own_cluster_shares(tmp_path):
    ed = SimpleNamespace(_samples={})
    members = _fcs_members(tmp_path)
    whole, _n = R.resolve_run_sample(ed, members[0]['item'])
    # The gate sits on the scale the events are held on: cut at the median.
    members[1]['item']['gates']['g1']['value'] = float(
        np.median(whole.data['FL1-A']))
    high, _n = R.resolve_run_sample(ed, members[1]['item'])
    assert 0 < len(high.data) < len(whole.data)
    whole.run_flowsom(channels=CH, n_metaclusters=3, max_events=2000, seed=42)
    m = F.build_map(whole.flowsom_result, whole.data, whole.data_transforms,
                    'run_1 · a')
    prep = R.prepare_unit(ed, members, 'G', dict(R.default_run_cfg(),
                                                 reference=m, max_events=300))
    whole_name, high_name = [r['Population'] for r in prep['pop_rows']]
    assert whole_name != high_name
    key = {'Group': 'G', 'Trial': 'Day 1', 'Sample': 'a'}
    # Each share is of that population's own events, all of them (not the
    # run's 300-event subsample).
    want = (F.member_rows(F.label_events(whole.data, m, whole.data_transforms)[0],
                          m, key, whole_name)
            + F.member_rows(F.label_events(high.data, m, high.data_transforms)[0],
                            m, key, high_name))
    assert len(prep['data']) == 300
    assert [r['Population'] for r in prep['ref_rows']] == \
        [r['Population'] for r in want]
    assert [r['%Parent'] for r in prep['ref_rows']] == \
        pytest.approx([r['%Parent'] for r in want])
    assert {(r['Group'], r['Trial'], r['Sample']) for r in prep['ref_rows']} \
        == {('G', 'Day 1', 'a')}
    n = len(prep['ref_rows']) // 2
    # Parent and child differ by far more than 1 point (the child is the
    # high half of FL1-A), so pooled shares could not pass this.
    assert [r['%Parent'] for r in prep['ref_rows'][:n]] != \
        pytest.approx([r['%Parent'] for r in prep['ref_rows'][n:]], abs=1)


def test_a_member_off_the_references_scale_fails_the_unit_after_its_table(
        tmp_path, som_run):
    _s, m = som_run                      # FL1-A on asinh; the file is not
    cfg = dict(R.default_run_cfg(), reference=m, umap=False)
    prep = R.prepare_unit(SimpleNamespace(_samples={}), _fcs_members(tmp_path),
                          'G', cfg)
    assert prep['ref_rows'] == []
    assert len(prep['ref_errors']) == 2
    for err in prep['ref_errors']:                 # both members, each named
        assert err.startswith('a: FL1-A is on ')
        assert 'but on asinh in the frozen reference' in err
    out = str(tmp_path / 'run')
    res = R.compute_run(prep, cfg, out)
    assert not res['ok']
    assert res['error'].startswith('ValueError: a: FL1-A is on ')
    # The gated table needs no clustering: it is written all the same.
    table = pd.read_csv(os.path.join(out, 'G_population_stats.csv'))
    assert list(table['Population']) == [r['Population'] for r in prep['pop_rows']]


def test_one_member_the_reference_cannot_label_keeps_the_others_table(
        tmp_path, som_run):
    from tests._fcs_writer import write_float_fcs
    _s, m = som_run
    members = _fcs_members(tmp_path)
    # A second file without FL2-A, a channel the map needs.
    lacking = write_float_fcs(str(tmp_path / 'b.fcs'),
                              np.ones((50, 1)) * 1000.0, ['FL1-A'])
    members.append({'item': dict(members[0]['item'], sample='b', path=lacking),
                    'group': 'G', 'sample': 'b'})
    prep = R.prepare_unit(SimpleNamespace(_samples={}), members, 'G',
                          dict(R.default_run_cfg(), reference=m))
    assert any(e.startswith('b: the frozen reference needs channel(s) this '
                            'run lacks: FL2-A') for e in prep['ref_errors'])


def test_a_recipe_never_carries_the_reference(som_run, tmp_path, monkeypatch):
    from tkinter import filedialog

    from tests.test_workspace import _panel_view
    _s, m = som_run
    assert R.recipe_cfg({'k': 15, 'reference': m}) == {'k': 15}
    path = str(tmp_path / 'recipe.json')
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: path)
    monkeypatch.setattr(filedialog, 'askopenfilename', lambda **k: path)
    root, v = _panel_view()
    try:
        v.model.run_cfg['reference'] = m
        v._save_recipe()
        with open(path, encoding='utf-8') as fh:
            assert 'reference' not in json.load(fh)['run_cfg']
        # Loading a recipe keeps this workspace's own reference.
        v._load_recipe()
        assert v.model.run_cfg['reference'] == m
        v.model.run_cfg.pop('reference')
        v._load_recipe()
        assert 'reference' not in v.model.run_cfg
    finally:
        root.destroy()


# ── the history and the template ───────────────────────────────────────────

def _run(hdir, run_id, date, maps=('U',), status=H.STATUS_DONE):
    d = os.path.join(hdir, run_id)
    os.makedirs(d)
    H.write_run_info(d, {'run_id': run_id, 'status': status,
                         'experiment_date': date})
    for label in maps:
        with open(os.path.join(d, label + F.MAP_SUFFIX), 'w') as fh:
            fh.write('{}')
    # A finished run has written a table (H.finished_status).
    with open(os.path.join(d, 'U_population_stats.csv'), 'w',
              encoding='utf-8') as fh:
        fh.write('Group,Sample,Population,Count\nU,s,All,1\n')
    return d


def test_the_maps_of_the_active_runs_newest_first(tmp_path):
    h = str(tmp_path)
    _run(h, 'run_1', '2026-11-10')
    _run(h, 'run_2', '2026-11-11')                    # superseded by run_3
    d3 = _run(h, 'run_3', '2026-11-11', maps=('B', 'A'))
    _run(h, 'run_4', '2026-11-12', status=H.STATUS_CANCELLED)
    got = F.run_maps(h)
    assert [t for t, _p in got] == ['run_3 (2026-11-11) · A',
                                    'run_3 (2026-11-11) · B',
                                    'run_1 (2026-11-10) · U']
    assert got[0][1] == os.path.join(d3, 'A' + F.MAP_SUFFIX)
    assert F.run_maps(str(tmp_path / 'none')) == []


def _template(path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'format': TEMPLATE_FORMAT, 'schema': 1, 'title': 't',
                   'slots': [{'name': 'A', 'pattern': 'a',
                              'populations': [{'sample': 'a'}]}],
                   'run_cfg': {'method': 'flowsom'}}, fh)
    return str(path)


def test_freezing_saves_the_map_in_the_templates_run_settings(som_run,
                                                              tmp_path):
    _s, m = som_run
    tpath = _template(tmp_path / 't.json')
    before = H.template_version(F.set_template_reference(tpath, None))
    tpl = F.set_template_reference(tpath, m)
    assert tpl['run_cfg']['reference'] == m
    with open(tpath, encoding='utf-8') as fh:
        saved = json.load(fh)
    assert saved['run_cfg']['reference'] == m
    assert saved['run_cfg']['method'] == 'flowsom'
    assert H.template_version(tpl) != before
    gone = F.set_template_reference(tpath, None)
    assert 'reference' not in gone['run_cfg']
    assert H.template_version(gone) == before
    with pytest.raises(ValueError, match='not an OpenFlo FlowSOM map'):
        F.set_template_reference(tpath, {'format': 'x'})


# ── the dialog and the panel ───────────────────────────────────────────────

def _tk():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    return root


def test_the_dialog_freezes_or_unfreezes():
    from openflo.ui_run_reference import ReferenceDialog
    root = _tk()
    try:
        got = []
        maps = [('run_2 · U', '/m2'), ('run_1 · U', '/m1')]
        dlg = ReferenceDialog(root, maps, current='run_1 · U',
                              on_choose=got.append)
        assert dlg.map_var.get() == 'run_2 · U'
        assert 'run_1 · U' in dlg.now.get()
        dlg.map_var.set('run_1 · U')
        dlg.freeze()
        dlg = ReferenceDialog(root, maps, current='x', on_choose=got.append)
        dlg.unfreeze()
        assert got == ['/m1', None]
        none = ReferenceDialog(root, on_choose=got.append)
        none.freeze()                               # nothing chosen: no-op
        assert got == ['/m1', None]
        empty = ReferenceDialog(root)
        assert str(empty.freeze_btn['state']) == 'disabled'
        assert str(empty.clear_btn['state']) == 'disabled'
        assert 'Nothing frozen' in empty.now.get()
        empty.unfreeze()                            # no callback: just closes
    finally:
        root.destroy()


def test_the_panel_freezes_into_the_template_and_workspace(som_run, tmp_path,
                                                           monkeypatch):
    _s, m = som_run
    root = _tk()
    try:
        import openflo.ui_run_reference as U
        from openflo import prefs
        from openflo.workspace import WorkspacePanel
        monkeypatch.setattr(prefs, 'load_pipeline_output_dir', lambda: None)
        opened = []
        monkeypatch.setattr(U, 'ReferenceDialog',
                            lambda *a, **k: opened.append((a, k)))
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel._freeze_reference()
        assert not opened and 'New run from template' in panel.status_var.get()

        tpath = _template(tmp_path / 'course.json')
        panel.model.template = {'path': tpath, 'name': 'course',
                                'version': 'v0'}
        mpath = F.write_map(_run(str(tmp_path / 'course_history'), 'run_1',
                                 '2026-11-10', maps=()), 'U', m)
        panel._freeze_reference()
        (_args, kw), = opened
        assert _args[1] == [('run_1 (2026-11-10) · U', mpath)]
        assert kw['current'] is None

        kw['on_choose'](mpath)
        assert panel.model.run_cfg['reference'] == m
        assert panel.model.template['version'] != 'v0'
        with open(tpath, encoding='utf-8') as fh:
            assert json.load(fh)['run_cfg']['reference'] == m
        assert 'Froze run_1 · U' in panel.status_var.get()
        opened.clear()
        panel._freeze_reference()
        assert opened[0][1]['current'] == 'run_1 · U'

        panel._set_reference(None)
        assert 'reference' not in panel.model.run_cfg
        assert 'Unfroze' in panel.status_var.get()
        panel._set_reference(str(tmp_path / 'missing.json'))
        assert panel.status_var.get().startswith('Reference not changed: '
                                                 'FileNotFoundError')
        assert 'reference' not in panel.model.run_cfg
    finally:
        root.destroy()


def test_a_unit_with_one_member_off_the_references_scale_fails_after_its_table(
        tmp_path, som_run):
    # The pooled frame takes the first member's scale, so labelling the pool
    # would succeed: only ref_errors stops the unit from running with one
    # member's reference rows silently missing.
    from tests._fcs_writer import write_float_fcs
    _s, m = som_run
    rng = np.random.default_rng(5)
    lab = rng.integers(0, 3, 1500)
    on = FlowSample.from_dataframe(pd.DataFrame(
        CENTRES[lab] + rng.normal(0, 0.3, (1500, 2)), columns=CH), name='a')
    on.data_transforms = dict(TF)                  # the map's scale
    off = write_float_fcs(str(tmp_path / 'b.fcs'),
                          1000.0 * (CENTRES[lab] + rng.normal(0, .3, (1500, 2))),
                          CH)                       # reloaded: not asinh
    base = {'qc_applied': False, 'comp_matrix': None, 'comp_channels': [],
            'gate_id': None, 'gate_chain': [], 'gates': {}}
    members = [{'item': dict(base, sample='a', path=''), 'group': 'G',
                'sample': 'a'},
               {'item': dict(base, sample='b', path=off), 'group': 'G',
                'sample': 'b'}]
    cfg = dict(R.default_run_cfg(), reference=m, umap=False)
    prep = R.prepare_unit(SimpleNamespace(_samples={'a': on}), members, 'G', cfg)
    assert [e.split(':')[0] for e in prep['ref_errors']] == ['b']
    assert prep['ref_rows']                        # a was labelled
    out = str(tmp_path / 'run')
    res = R.compute_run(prep, cfg, out)
    assert not res['ok'] and res['error'].startswith('ValueError: b: ')
    table = pd.read_csv(os.path.join(out, 'G_population_stats.csv'))
    assert set(table['Sample']) == {'a', 'b'}
    assert not any('Reference Cluster' in p for p in table['Population'])
