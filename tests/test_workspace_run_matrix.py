"""A Pipeline Workspace run applies a row or group matrix (scope audit L3, b-i).

A matrix set on a row or its group was shown in the tree, but the run
clustered the editor's sample as it was, or a reload with the matrix saved
at drop. The run now puts the row's matrix, else the group's, on its own
copy of the sample (``FlowSample.recompensate``: events, transforms and
computed columns kept, as the comp editor's Apply does), or on the reload.
A matrix that cannot be applied refuses the member instead of running it on
another compensation. Gates were drawn on the editor's compensation, so a
gated member's count under both is noted, and a change of more than 5% is
a warning on the run's final line. Generated data and generic channels only.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace_run as R
from openflo.flow_sample import FlowSample
from openflo.workspace_run import WorkspaceModel
from tests._fcs_writer import write_float_fcs
from tests.conftest import gui_unavailable

CH = ['FL1-A', 'FL2-A']
M_EDITOR = np.array([[1.0, 0.05], [0.0, 1.0]])
M_ROW = np.array([[1.0, 0.40], [0.0, 1.0]])
M_GROUP = np.array([[1.0, 0.20], [0.10, 1.0]])


@pytest.fixture
def fcs(tmp_path):
    rng = np.random.default_rng(3)
    fl1 = rng.uniform(1000, 50000, 600)
    fl2 = rng.uniform(100, 2000, 600) + 0.3 * fl1
    ev = np.column_stack([rng.uniform(2e4, 2e5, 600), fl1, fl2])
    return write_float_fcs(str(tmp_path / 's1.fcs'), ev, ['FSC-A', *CH])


def _loaded(path, matrix):
    s = FlowSample(path)
    s.manual_compensate(matrix, CH)
    s.apply_transform(CH)
    return s


def _model(path, row=None, group=None):
    m = WorkspaceModel()
    g = m.new_group('G')
    mid = m.add_item({'sample': 's1', 'trial': 'T', 'gate_id': None,
                      'path': path, 'color': '#000000', 'qc_applied': False,
                      'comp_matrix': M_EDITOR, 'comp_channels': list(CH)},
                     gid=g)
    if group is not None:
        m.set_group_comp(g, {'kind': 'matrix', 'channels': CH,
                             'matrix': group.tolist()})
    if row is not None:
        m.set_item_comp(mid, {'kind': 'matrix', 'channels': CH,
                              'matrix': row.tolist()})
    return m, mid


def _run_values(path, editor, m, mid, **kw):
    comp = m.effective_comp(mid, editor)
    s, note = R.resolve_run_sample(editor, m.item(mid), comp=comp, **kw)
    return s.data[CH].to_numpy(), note


def test_a_live_member_gets_the_rows_matrix(fcs):
    live = _loaded(fcs, M_EDITOR)
    before = live.data[CH].to_numpy().copy()
    ed = SimpleNamespace(_samples={'s1': live})
    m, mid = _model(fcs, row=M_ROW, group=M_GROUP)
    got, note = _run_values(fcs, ed, m, mid)
    want = _loaded(fcs, M_ROW).data[CH].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    assert 'compensated with the row matrix' in note
    np.testing.assert_array_equal(live.data[CH].to_numpy(), before)
    np.testing.assert_array_equal(live.comp_matrix, M_EDITOR)


def test_else_the_groups_matrix(fcs):
    ed = SimpleNamespace(_samples={'s1': _loaded(fcs, M_EDITOR)})
    m, mid = _model(fcs, group=M_GROUP)
    got, note = _run_values(fcs, ed, m, mid)
    np.testing.assert_allclose(got, _loaded(fcs, M_GROUP).data[CH].to_numpy(),
                               rtol=1e-6, atol=1e-6)
    assert 'compensated with the group matrix' in note


def test_the_editors_matrix_runs_as_before(fcs):
    live = _loaded(fcs, M_EDITOR)
    ed = SimpleNamespace(_samples={'s1': live})
    m, mid = _model(fcs)
    got, note = _run_values(fcs, ed, m, mid)
    np.testing.assert_array_equal(got, live.data[CH].to_numpy())
    assert note == 'live editor sample'


def test_a_reload_gets_the_groups_matrix_not_the_saved_one(fcs):
    ed = SimpleNamespace(_samples={})
    m, mid = _model(fcs, group=M_GROUP)
    got, note = _run_values(fcs, ed, m, mid)
    want = _loaded(fcs, M_GROUP).data[CH].to_numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    assert note == 'reloaded from file, compensated with the group matrix'
    saved, _n = _run_values(fcs, ed, *_model(fcs))
    np.testing.assert_allclose(saved, _loaded(fcs, M_EDITOR).data[CH].to_numpy(),
                               rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('live', [True, False], ids=['live', 'reload'])
def test_a_matrix_that_cannot_be_applied_refuses_the_member(fcs, live):
    ed = SimpleNamespace(_samples={'s1': _loaded(fcs, M_EDITOR)} if live else {})
    m, mid = _model(fcs, row=np.array([[1.0, 1.0], [1.0, 1.0]]))  # singular
    with pytest.raises(RuntimeError, match="cannot apply the row matrix to 's1'"):
        _run_values(fcs, ed, m, mid)


def test_an_uncompensated_live_fcs_sample_gets_the_rows_matrix(fcs):
    """No comp record, but read from an FCS file: the matrix is applied
    (only a sample built from a table is refused)."""
    live = FlowSample(fcs)
    live.apply_transform(CH)
    assert not live.comp_channels
    ed = SimpleNamespace(_samples={'s1': live})
    m, mid = _model(fcs, row=M_ROW)
    got, note = _run_values(fcs, ed, m, mid)
    np.testing.assert_allclose(got, _loaded(fcs, M_ROW).data[CH].to_numpy(),
                               rtol=1e-6, atol=1e-6)
    assert 'compensated with the row matrix' in note


def test_a_sample_built_from_a_table_is_refused(fcs):
    df = _loaded(fcs, M_EDITOR).data.copy()
    table = FlowSample.from_dataframe(df, name='s1', path='s1.csv')
    ed = SimpleNamespace(_samples={'s1': table})
    m, mid = _model(fcs, row=M_ROW)
    with pytest.raises(RuntimeError, match='does not record the compensation'):
        _run_values(fcs, ed, m, mid)


def _gated(m, mid, value):
    it = m.item(mid)
    it['gate_id'] = 'g1'
    it['gates'] = {'g1': {'id': 'g1', 'kind': 'threshold', 'channel': 'FL2-A',
                          'op': '>', 'value': value, 'parent_id': None}}


def test_a_gated_member_notes_both_counts_and_warns_on_drift(fcs):
    live = _loaded(fcs, M_EDITOR)
    ed = SimpleNamespace(_samples={'s1': live})
    m, mid = _model(fcs, row=M_ROW)
    _gated(m, mid, float(np.median(live.data['FL2-A'])))
    warned: list = []
    _got, note = _run_values(fcs, ed, m, mid, warnings=warned)
    assert "with the editor's compensation)" in note
    assert ', gated ' in note
    assert len(warned) == 1 and 'the gate keeps' in warned[0]
    assert 'with the row matrix' in warned[0]


def test_no_drift_warning_for_the_same_matrix(fcs):
    live = _loaded(fcs, M_EDITOR)
    ed = SimpleNamespace(_samples={'s1': live})
    m, mid = _model(fcs, row=M_EDITOR)
    _gated(m, mid, float(np.median(live.data['FL2-A'])))
    warned: list = []
    _run_values(fcs, ed, m, mid, warnings=warned)
    assert warned == []


def test_a_unit_says_when_its_members_are_compensated_differently(fcs):
    a, b = _loaded(fcs, M_EDITOR), _loaded(fcs, M_EDITOR)
    b.name = 's2'
    ed = SimpleNamespace(_samples={'s1': a, 's2': b})
    m, mid = _model(fcs, row=M_ROW)
    mid2 = m.add_item({'sample': 's2', 'trial': 'T', 'gate_id': None,
                       'path': fcs, 'qc_applied': False,
                       'comp_matrix': M_EDITOR, 'comp_channels': list(CH)})
    members = [{'item': m.item(x), 'group': None, 'sample': n,
                'comp': m.effective_comp(x, ed)}
               for x, n in ((mid, 's1'), (mid2, 's2'))]
    prep = R.prepare_unit(ed, members, 'u', {'channels': CH})
    assert any(w.startswith('members compensated differently: ')
               and 'matrix (row) (s1)' in w and 'own matrix (s2)' in w
               for w in prep['warnings']), prep['warnings']
    assert prep['member_notes'][0].startswith('s1: live editor sample, '
                                              'compensated with the row')
    assert isinstance(prep['data'], pd.DataFrame)


def test_a_units_drift_warning_reaches_its_final_line(fcs):
    live = _loaded(fcs, M_EDITOR)
    ed = SimpleNamespace(_samples={'s1': live})
    m, mid = _model(fcs, row=M_ROW)
    _gated(m, mid, float(np.median(live.data['FL2-A'])))
    members = [{'item': m.item(mid), 'group': None, 'sample': 's1',
                'comp': m.effective_comp(mid, ed)}]
    prep = R.prepare_unit(ed, members, 'u', {'channels': CH})
    assert any('the gate keeps' in w for w in prep['warnings']), prep['warnings']
    assert "with the editor's compensation)" in prep['member_notes'][0]


@pytest.mark.parametrize('live', [True, False], ids=['live', 'reload'])
def test_an_override_in_another_channel_order_gives_the_same_events(fcs, live):
    """The row matrix lists FL2-A first; the sample and the file list FL1-A
    first. The matrix is applied by channel name, not by position."""
    ed = SimpleNamespace(_samples={'s1': _loaded(fcs, M_EDITOR)} if live else {})
    m, mid = _model(fcs)
    swapped = M_ROW[::-1, ::-1]
    m.set_item_comp(mid, {'kind': 'matrix', 'channels': CH[::-1],
                          'matrix': swapped.tolist()})
    got, _note = _run_values(fcs, ed, m, mid)
    np.testing.assert_allclose(got, _loaded(fcs, M_ROW).data[CH].to_numpy(),
                               rtol=1e-6, atol=1e-6)
    assert not np.allclose(got, _loaded(fcs, swapped).data[CH].to_numpy())


def test_the_member_notes_are_written_to_the_run_folder(tmp_path):
    res = {'files': []}
    R._write_member_notes(['s1: live editor sample, gated 10/20 (12 with the '
                           "editor's compensation)", 's2: reloaded from file'],
                          str(tmp_path), 'u', res)
    text = (tmp_path / 'u_members.txt').read_text(encoding='utf-8')
    assert text.splitlines() == [
        "s1: live editor sample, gated 10/20 (12 with the editor's compensation)",
        's2: reloaded from file']
    assert res['files'] == [str(tmp_path / 'u_members.txt')]
    R._write_member_notes([], str(tmp_path), 'v', res)
    assert not (tmp_path / 'v_members.txt').exists()


def test_each_members_line_says_which_compensation(fcs):
    """One line per kind: the editor's matrix, the matrix saved at drop
    (reloaded), a row matrix, and none (live, uncompensated)."""
    bare = FlowSample(fcs)
    bare.apply_transform(CH)
    ed = SimpleNamespace(_samples={'ed': _loaded(fcs, M_EDITOR),
                                   'row': _loaded(fcs, M_EDITOR), 'bare': bare})
    m = WorkspaceModel()
    mids = {}
    for name, saved in (('ed', M_EDITOR), ('saved', M_EDITOR),
                        ('row', M_EDITOR), ('bare', None)):
        mids[name] = m.add_item({
            'sample': name, 'trial': 'T', 'gate_id': None, 'path': fcs,
            'qc_applied': False, 'comp_matrix': saved,
            'comp_channels': list(CH) if saved is not None else []})
    m.set_item_comp(mids['row'], {'kind': 'matrix', 'channels': CH,
                                  'matrix': M_ROW.tolist()})
    members = [{'item': m.item(x), 'group': None, 'sample': n,
                'comp': m.effective_comp(x, ed)} for n, x in mids.items()]
    lines = R.prepare_unit(ed, members, 'u', {'channels': CH})['member_notes']
    assert lines == [
        "ed: live editor sample, the editor's 2×2 matrix",
        'saved: reloaded from file, the 2×2 matrix saved when it was added',
        'row: live editor sample, compensated with the row matrix',
        'bare: live editor sample, no compensation']


def _drift(live, row, value):
    """(kept, was) for the gate FL2-A > value under the row matrix and the
    editor's, counted by hand."""
    return (int((row.data['FL2-A'] > value).sum()),
            int((live.data['FL2-A'] > value).sum()))


def test_the_drift_warning_starts_above_five_percent(fcs):
    """A row matrix close to the editor's: gate cuts are found by hand where
    the count moves by 1-4.5% (no warning) and by 6-20% (a warning)."""
    near = np.array([[1.0, 0.07], [0.0, 1.0]])
    live, row = _loaded(fcs, M_EDITOR), _loaded(fcs, near)
    cuts = np.quantile(live.data['FL2-A'], np.linspace(0.02, 0.98, 400))
    rel = {float(v): abs(k - w) / max(w, 1)
           for v in cuts for k, w in [_drift(live, row, v)]}
    under = next(v for v, r in rel.items() if 0.01 < r < 0.045)
    over = next(v for v, r in rel.items() if 0.06 < r < 0.2)
    for value, warns in ((under, False), (over, True)):
        ed = SimpleNamespace(_samples={'s1': live})
        m, mid = _model(fcs, row=near)
        _gated(m, mid, value)
        warned: list = []
        _run_values(fcs, ed, m, mid, warnings=warned)
        assert bool(warned) is warns, (value, rel[value], warned)


def test_a_notes_file_that_cannot_be_written_is_a_warning(tmp_path):
    res = {'files': [], 'warnings': []}
    (tmp_path / 'u_members.txt').mkdir()          # the path is a folder
    R._write_member_notes(['s1: reloaded from file'], str(tmp_path), 'u', res)
    assert res['files'] == []
    assert any('members file not written' in w for w in res['warnings'])


def test_a_run_writes_its_member_notes(tmp_path):
    rng = np.random.default_rng(9)
    df = pd.DataFrame({c: rng.normal(0, 1, 200) for c in CH})
    prep = {'data': df, 'channels': list(CH), 'label': 'u', 'n_events': 200,
            'note': '', 'color_by': None,
            'member_notes': ['s1: live editor sample, compensated with the '
                             'row matrix']}
    cfg = R.default_run_cfg()
    cfg.update(method='phenograph', umap=False, tsne=False, phate=False,
               trimap=False, pacmap=False, max_events=0)
    res = R.compute_run(prep, cfg, str(tmp_path))
    assert res['ok'], res
    path = tmp_path / 'u_members.txt'
    assert str(path) in res['files']
    assert 'compensated with the row matrix' in path.read_text(encoding='utf-8')


def test_run_takes_each_members_comp_when_pressed(tmp_path, monkeypatch, fcs):
    """An edit made after Run is pressed does not reach a unit still
    waiting in the queue."""
    pytest.importorskip('tkinter')
    import tkinter as tk
    try:
        r = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    r.withdraw()
    try:
        from openflo import workspace as W
        monkeypatch.chdir(tmp_path)
        live = _loaded(fcs, M_EDITOR)
        ed = SimpleNamespace(
            _samples={'s1': live}, _sample_order=['s1'],
            _sample_trial={'s1': 'T'}, _sample_colors={'s1': '#000000'},
            _sample_gates={'s1': {}}, _sample_gate_order={'s1': []},
            _channel_labels={})
        panel = W.WorkspacePanel(r, editor=ed)
        m, mid = _model(fcs, row=M_ROW)
        panel.model.groups, panel.model.loose = m.groups, m.loose
        monkeypatch.setattr(panel, '_pump_jobs', lambda *a, **k: None)
        panel._run()
        (job,) = list(panel._jobs)
        (member,) = job['unit']['members']
        assert member['comp']['source'] == 'row'
        m.item(mid)['comp_override']['matrix'][0][1] = 0.9    # edited later
        assert member['comp']['matrix'][0][1] == 0.4
    finally:
        r.destroy()
