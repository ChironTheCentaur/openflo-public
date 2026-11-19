"""Run templates: a Pipeline Workspace saved without its files, so a
repeating experiment runs each new day's files with the same groups,
gates, compensation and settings (``openflo.run_template``).

Generated data, generic channel and sample names only.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from openflo import run_template as T
from openflo import workspace_run as R
from openflo.channel_transforms import transform_spec
from openflo.workspace_run import WorkspaceModel
from tests._fcs_writer import write_float_fcs
from tests.conftest import gui_unavailable

CH = ['FSC-A', 'FL1-A', 'FL2-A']
M = np.array([[1.0, 0.1], [0.05, 1.0]])
GATES = {
    'g1': {'kind': 'threshold', 'name': 'Big', 'channel': 'FSC-A',
           'value': 100000.0, 'id': 'g1', 'parent_id': None},
    'g2': {'kind': 'threshold', 'name': 'Pos', 'channel': 'FL2-A',
           'value': 0.0, 'id': 'g2', 'parent_id': 'g1'},
}


# The scales the gates were drawn on, as a template keeps them.
TF = {c: transform_spec('logicle') for c in ('FL1-A', 'FL2-A')}


def _item(sample, gid=None, path=None, **kw):
    chain = R.gate_chain(GATES, gid) if gid else []
    it = {'sample': sample, 'trial': 'Day 1', 'path': path,
          'color': '#000000', 'gate_id': gid,
          'gate_path': ' / '.join(GATES[g]['name'] for g in chain) or None,
          'is_leaf': True if gid else None, 'gate_chain': chain,
          'gates': {g: dict(GATES[g]) for g in chain},
          'comp_matrix': M, 'comp_channels': ['FL1-A', 'FL2-A'],
          'qc_applied': False, 'channel_labels': {}, 'transforms': dict(TF)}
    it.update(kw)
    return it


def _model():
    """Two groups (ctrl, treat), each a whole sample and the Big
    population of two samples named like one day's files."""
    m = WorkspaceModel('Course')
    for grp in ('ctrl', 'treat'):
        gid = m.new_group(grp.title())
        m.set_group_fmo(gid, {'files': [f'/fmo/{grp}.fcs']})
        m.set_group_comp(gid, {'kind': 'beads', 'files': [f'/beads/{grp}.fcs']})
        m.groups[gid]['cfg'] = {'k': 15}
        for rep in ('1', '2'):
            name = f'20261111_{grp}_{rep}'
            m.add_item(_item(name), gid)
            m.add_item(_item(name, 'g1'), gid)
    m.run_cfg['max_events'] = 1234
    return m


# ── patterns ───────────────────────────────────────────────────────────────

def test_name_tokens_split_on_anything_not_a_letter_or_digit():
    assert T.name_tokens('Ctrl_2h-B.fcs') == ['ctrl', '2h', 'b', 'fcs']


def test_suggested_pattern_drops_words_every_sample_shares():
    # The date every one of the day's files carries would stop the template
    # matching tomorrow's files; the group's own word stays.
    names = ['20261111_ctrl_1', '20261111_ctrl_2', '20261111_treat_1']
    assert T.suggest_pattern(names[:2], names) == 'ctrl'
    assert T.suggest_pattern(names[2:], names) == 'treat 1'
    # One group holding every sample matches any file.
    assert T.suggest_pattern(names, names) == ''


@pytest.mark.parametrize('pattern,name,fits', [
    ('', 'anything.fcs', True),
    ('ctrl', 'D2_ctrl_3.fcs', True),
    ('ctrl', 'D2_ctrlx_3.fcs', False),       # whole words, not substrings
    ('ctrl 2h', 'x_2h_CTRL.fcs', True),       # every word, any order, any case
    ('ctrl 2h', 'x_4h_ctrl.fcs', False),
    ('*_ctrl_*', 'D9_CTRL_1.fcs', True),      # a glob, case ignored
    ('*_CTRL_*', 'D9_ctrl_1.fcs', True),
    ('*_ctrl_?', 'D9_ctrl_12.fcs', False),
])
def test_pattern_matches(pattern, name, fits):
    assert T.pattern_matches(pattern, f'/data/{name}') is fits


def test_match_prefers_the_specific_slot_and_leaves_ties_unassigned():
    tpl = {'slots': [{'name': 'Any', 'pattern': ''},
                     {'name': 'Ctrl', 'pattern': 'ctrl'},
                     {'name': 'Ctrl2', 'pattern': 'ctrl 2h'},
                     {'name': 'Late', 'pattern': 'late'},
                     {'name': 'Late2', 'pattern': 'late'},
                     {'name': 'Never', 'pattern': 'zzz'}]}
    files = ['a_ctrl_4h.fcs', 'a_ctrl_2h.fcs', 'b_other.fcs', 'c_late.fcs']
    got = T.match_files(tpl, files)
    assert got['assign'] == {'a_ctrl_4h.fcs': 1, 'a_ctrl_2h.fcs': 2,
                             'b_other.fcs': 0}
    assert got['ambiguous'] == {'c_late.fcs': [3, 4]}
    assert got['unmatched'] == []
    assert got['empty'] == [3, 4, 5]
    tpl['slots'] = tpl['slots'][1:]
    assert T.match_files(tpl, ['b_other.fcs'])['unmatched'] == ['b_other.fcs']


# ── template from a workspace, and back ────────────────────────────────────

def test_template_keeps_groups_populations_and_settings_not_files():
    tpl = T.template_from_model(_model())
    assert [s['name'] for s in tpl['slots']] == ['Ctrl', 'Treat']
    ctrl = tpl['slots'][0]
    assert ctrl['pattern'] == 'ctrl'
    assert ctrl['fmo'] == {'files': ['/fmo/ctrl.fcs']}
    assert ctrl['cfg'] == {'k': 15}
    # Four rows (2 samples x 2 populations) are two populations.
    assert [p['gate_id'] for p in ctrl['populations']] == [None, 'g1']
    for p in ctrl['populations']:
        assert not {'sample', 'trial', 'path', 'color'} & set(p)
    assert tpl['run_cfg']['max_events'] == 1234


def test_ungrouped_rows_become_one_ungrouped_slot():
    m = _model()
    m.add_item(_item('20261111_spare_1', 'g2'))
    slot = T.template_from_model(m)['slots'][-1]
    assert (slot['name'], slot['grouped']) == (T.UNGROUPED, False)
    assert [p['gate_id'] for p in slot['populations']] == ['g2']


def test_an_empty_workspace_makes_no_template():
    with pytest.raises(ValueError):
        T.template_from_model(WorkspaceModel())


def test_template_survives_a_json_round_trip():
    tpl = T.template_from_model(_model())
    back = T.template_from_json(json.loads(json.dumps(T.template_to_json(tpl))))
    pop = back['slots'][0]['populations'][1]
    assert pop['gates'] == {'g1': GATES['g1']}
    assert back['run_cfg']['max_events'] == 1234
    # Settings the file lacks take the defaults.
    raw = T.template_to_json(tpl)
    del raw['run_cfg']['method']
    assert T.template_from_json(raw)['run_cfg']['method'] == \
        R.default_run_cfg()['method']
    # A slot with no populations is dropped.
    raw['slots'].append({'name': 'Hollow', 'populations': []})
    assert [s['name'] for s in T.template_from_json(raw)['slots']] == \
        ['Ctrl', 'Treat']
    with pytest.raises(ValueError):
        T.template_from_json({'format': 'openflo-workspace', 'groups': []})
    with pytest.raises(ValueError):
        T.template_from_json({'format': T.TEMPLATE_FORMAT, 'slots': []})


# ── the day's workspace ────────────────────────────────────────────────────

def test_instantiate_builds_one_row_per_file_and_population():
    tpl = T.template_from_model(_model())
    files = ['/d2/x_ctrl_2.fcs', '/d2/x_treat_1.fcs', '/d2/x_ctrl_1.fcs',
             '/other/x_ctrl_1.fcs']
    m = T.instantiate(tpl, {files[0]: 0, files[1]: 1, files[2]: 0,
                            files[3]: 0}, trial_of=lambda p: 'Day 2')
    assert [g['name'] for g in m.groups.values()] == ['Ctrl', 'Treat']
    ctrl, treat = m.groups.values()
    assert ctrl['fmo'] == {'files': ['/fmo/ctrl.fcs']}
    assert treat['comp'] == {'kind': 'beads', 'files': ['/beads/treat.fcs']}
    assert ctrl['cfg'] == {'k': 15}
    assert m.run_cfg['max_events'] == 1234
    rows = list(ctrl['items'].values())
    # Natural file order, each file once per population; two files of one
    # name get distinct sample names.
    assert [(r['sample'], r['gate_id']) for r in rows] == [
        ('x_ctrl_1', None), ('x_ctrl_1', 'g1'),
        ('x_ctrl_1 (2)', None), ('x_ctrl_1 (2)', 'g1'),
        ('x_ctrl_2', None), ('x_ctrl_2', 'g1')]
    assert {r['trial'] for r in rows} == {'Day 2'}
    assert all(r['template_slot'] == 'Ctrl' for r in rows)
    assert len(treat['items']) == 2
    # The rows' gates are copies: editing one leaves the template alone.
    rows[1]['gates']['g1']['value'] = 1.0
    assert tpl['slots'][0]['populations'][1]['gates']['g1']['value'] == 100000.0
    assert rows[3]['gates']['g1']['value'] == 100000.0


def test_instantiate_stores_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = T.instantiate(T.template_from_model(_model()), {'x_ctrl_1.fcs': 0})
    assert {it['path'] for _m, it, _g in m.all_items()} == \
        {str(tmp_path / 'x_ctrl_1.fcs')}


def test_instantiate_keeps_ungrouped_rows_loose():
    m = _model()
    m.add_item(_item('20261111_spare_1', 'g2'))
    tpl = T.template_from_model(m)
    got = T.instantiate(tpl, {'/d/s.fcs': len(tpl['slots']) - 1})
    assert not got.groups
    assert [r['gate_id'] for r in got.loose.values()] == ['g2']


# ── the preview ────────────────────────────────────────────────────────────

def _cols(path):
    if 'bad' in path:
        raise OSError('not an FCS file')
    if 'noscatter' in path:
        return ['FL1-A', 'FL2-A']
    return ['FSC-A', 'FL1-A'] if 'short' in path else list(CH)


def _spill(path):
    return 'nospill' not in path


def test_preview_rows_say_what_stops_each_file():
    tpl = T.template_from_model(_model())
    rows = T.preview_rows(tpl, ['a_ctrl_1.fcs', 'a_treat_1.fcs', 'a_x.fcs',
                                'short_ctrl.fcs', 'bad_ctrl.fcs'], _cols, _spill)
    status = [T.row_status(tpl, r) for r in rows]
    assert status[0] == status[1] == 'ready'
    assert status[2] == 'fits no slot; choose one or skip it'
    # FL2-A is no Ctrl gate channel, and no template matrix needs it.
    assert status[3] == 'ready'
    assert status[4].startswith('cannot be read (OSError: not an FCS file)')
    # A gate channel alone (FSC-A: not compensated) counts too.
    [scatterless] = T.preview_rows(tpl, ['noscatter_ctrl.fcs'], _cols, _spill)
    assert T.row_problem(tpl, scatterless).startswith('has no FSC-A,')
    with pytest.raises(ValueError, match='a_x.fcs fits no slot'):
        T.preview_assignment(tpl, rows)
    for r in rows[2:]:
        r['slot'] = T.SKIP
    assert T.preview_assignment(tpl, rows) == {'a_ctrl_1.fcs': 0,
                                               'a_treat_1.fcs': 1}
    for r in rows:
        r['slot'] = T.SKIP
    with pytest.raises(ValueError, match='no file is set to run'):
        T.preview_assignment(tpl, rows)


def test_a_file_fitting_two_slots_waits_for_a_choice():
    tpl = T.template_from_model(_model())
    tpl['slots'][1]['pattern'] = 'ctrl'
    row = T.preview_rows(tpl, ['a_ctrl_1.fcs'], _cols, _spill)[0]
    assert row['slot'] is None
    assert T.row_problem(tpl, row) == 'fits Ctrl / Treat; choose one'


# ── a template row runs on its own file ────────────────────────────────────

@pytest.fixture
def day_files(tmp_path):
    rng = np.random.default_rng(7)
    out = []
    for i in range(2):
        ev = np.column_stack([rng.uniform(2e4, 2e5, 500),
                              rng.uniform(100, 5e4, 500),
                              rng.uniform(100, 5e4, 500)])
        out.append((write_float_fcs(str(tmp_path / f'd2_ctrl_{i}.fcs'), ev, CH),
                    int((ev[:, 0] > 100000.0).sum())))
    return out


def test_file_channels_reads_the_names_only(day_files):
    assert T.file_channels(day_files[0][0]) == CH


def test_a_template_row_is_gated_on_its_own_file(day_files):
    tpl = T.template_from_model(_model())
    m = T.instantiate(tpl, {p: 0 for p, _n in day_files})
    rows = [it for _mid, it, _g in m.all_items() if it['gate_id'] == 'g1']
    for (_p, want), it in zip(day_files, rows, strict=True):
        s, note = R.resolve_run_sample(SimpleNamespace(_samples={}), it)
        assert 'reloaded from file' in note
        assert len(s.data) == want


def test_a_template_row_never_runs_on_an_editor_sample(day_files):
    path = day_files[0][0]
    tpl = T.template_from_model(_model())
    m = T.instantiate(tpl, {path: 0})
    mid, it = next((mid, it) for mid, it, _g in m.all_items())
    # Neither another day's tube of that name nor the same file open in the
    # editor (with the editor's compensation): one template, one answer.
    for src in ('/elsewhere/d2_ctrl_0.fcs', path):
        other = SimpleNamespace(path=src, source_fcs=None,
                                comp_channels=['FL9-A'])
        ed = SimpleNamespace(_samples={it['sample']: other})
        assert R.live_sample(ed, it) is None
        assert m.effective_comp(mid, ed)['live'] is False
        _s, note = R.resolve_run_sample(ed, it)
        assert 'reloaded from file' in note
        assert R.member_channels(ed, it) == ['FL1-A', 'FL2-A']
    # A row not made from a template keeps finding its sample by name.
    plain = dict(it)
    plain.pop('template_slot')
    assert R.live_sample(ed, plain) is other
    assert R.live_sample(None, plain) is None


# ── the dialog ─────────────────────────────────────────────────────────────

def test_preview_dialog_enables_load_only_when_every_file_is_settled():
    import tkinter as tk

    from openflo.ui_run_template import RunTemplatePreview
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    done = []
    try:
        tpl = T.template_from_model(_model())
        win = RunTemplatePreview(root, tpl, ['a_ctrl_1.fcs', 'a_x.fcs'],
                                 on_done=lambda a, run: done.append((a, run)),
                                 channels_of=_cols, spill_of=_spill)
        assert win.run_btn.instate(['disabled'])
        assert 'a_x.fcs fits no slot' in win.summary.get()
        win.set_rows([1], '(skip this file)')
        assert win.tree.item('1')['values'][3] == 'skipped'
        assert not win.run_btn.instate(['disabled'])
        win.set_rows([1], 'Treat')
        assert not win.run_btn.instate(['disabled'])
        assert win.tree.item('1')['values'][1] == 'Treat'
        win.run_btn.invoke()
        assert done == [({'a_ctrl_1.fcs': 0, 'a_x.fcs': 1}, True)]
        assert not win.winfo_exists()
    finally:
        root.destroy()


def test_panel_saves_a_template_and_builds_the_days_workspace(tmp_path,
                                                              monkeypatch):
    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        undo = []
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}),
                               on_before_change=lambda: undo.append(1))
        panel.model = _model()
        tfile = tmp_path / 'course.json'
        monkeypatch.setattr(filedialog, 'asksaveasfilename',
                            lambda **kw: str(tfile))
        panel._save_template()
        assert 'Saved run template course.json with 2 slot(s)' in \
            panel.status_var.get()
        saved = json.loads(tfile.read_text(encoding='utf-8'))
        assert saved['format'] == T.TEMPLATE_FORMAT

        day = [str(tmp_path / 'd2_ctrl_1.fcs'), str(tmp_path / 'd2_treat_1.fcs')]
        monkeypatch.setattr(filedialog, 'askopenfilename', lambda **kw: str(tfile))
        monkeypatch.setattr(filedialog, 'askopenfilenames', lambda **kw: day)
        opened = []
        import openflo.ui_run_template as U
        monkeypatch.setattr(U, 'file_channels', lambda p: list(CH))

        class _Spy(U.RunTemplatePreview):
            def __init__(self, *a, **kw):
                kw['channels_of'] = lambda p: list(CH)
                kw['spill_of'] = lambda p: True
                super().__init__(*a, **kw)
                opened.append(self)
        monkeypatch.setattr(U, 'RunTemplatePreview', _Spy)
        panel._new_run_from_template()
        [win] = opened
        ran = []
        monkeypatch.setattr(panel, '_run', lambda: ran.append(1))
        win.run_btn.invoke()
        assert ran == [1] and undo == [1]
        assert [g['name'] for g in panel.model.groups.values()] == ['Ctrl', 'Treat']
        assert panel._workspace_file == str(tfile)
        assert {it['path'] for _m, it, _g in panel.model.all_items()} == set(day)
        assert 'Built 4 row(s) from 2 file(s)' in panel.status_var.get()

        # "Load only" builds it without running.
        panel._new_run_from_template()
        opened[-1].load_btn.invoke()
        assert ran == [1]
    finally:
        root.destroy()


def test_panel_refuses_a_template_from_an_empty_workspace(monkeypatch):
    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        from openflo.workspace import WorkspacePanel
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        asked = []
        monkeypatch.setattr(filedialog, 'asksaveasfilename',
                            lambda **kw: asked.append(1) or '')
        panel._save_template()
        assert asked == []
        assert panel.status_var.get().startswith('No template saved')
    finally:
        root.destroy()


# ── review follow-ups: scales, matching, copies ────────────────────────────

def test_a_template_row_is_gated_on_the_scales_its_gates_were_drawn_on(
        tmp_path):
    # A threshold drawn on an asinh display. The same template and file must
    # give the same population whatever the editor shows now: a fresh app
    # (the loader's logicle), an editor on logicle, or the file open in it.
    from openflo.channel_transforms import transform_values
    from openflo.flow_sample import FlowSample
    rng = np.random.default_rng(3)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 2000),
                          rng.lognormal(7, 1.5, 2000),
                          rng.lognormal(7, 1.5, 2000)])
    path = write_float_fcs(str(tmp_path / 'd2_ctrl_1.fcs'), ev, CH)
    thr = float(transform_values(np.array([1000.0]), method='asinh')[0])
    gate = {'kind': 'threshold', 'name': 'Pos', 'channel': 'FL1-A',
            'value': thr, 'id': 'p', 'parent_id': None}
    asinh = {c: transform_spec('asinh') for c in ('FL1-A', 'FL2-A')}
    it = _item('d2_ctrl_1', path=path, gate_id='p', gate_chain=['p'],
               gates={'p': gate}, comp_matrix=None, comp_channels=[],
               transforms=asinh, template_slot='Ctrl')
    want = int((ev[:, 1] > 1000.0).sum())

    def on_logicle(s):
        s.retransform(['FL1-A', 'FL2-A'], method='logicle')
    fresh = SimpleNamespace(_samples={})
    logicle = SimpleNamespace(_samples={}, _conform_sample_transforms=on_logicle)
    live = FlowSample(path)
    live.apply_transform()
    before = live.data.copy()
    opened = SimpleNamespace(_samples={'d2_ctrl_1': live})
    for editor in (fresh, logicle, opened):
        s, _note = R.resolve_run_sample(editor, it)
        assert len(s.data) == want
    assert live.data_transforms['FL1-A']['method'] == 'logicle'   # untouched
    assert live.data.equals(before)
    # A row from no template has no record and keeps the editor's scales.
    plain = {k: v for k, v in it.items() if k not in ('transforms',
                                                       'template_slot')}
    def on_asinh(s):
        s.retransform(['FL1-A', 'FL2-A'], method='asinh')
    s, _note = R.resolve_run_sample(
        SimpleNamespace(_samples={}, _conform_sample_transforms=on_asinh), plain)
    assert s.data_transforms['FL1-A']['method'] == 'asinh'


def test_the_template_keeps_each_populations_scales():
    seen = []

    def scales(it):
        seen.append(it['sample'])
        return {'FL1-A': transform_spec('asinh')}
    m = _model()
    m.add_item(_item('20261111_spare_1', 'g2'))
    tpl = T.template_from_model(m, scales)
    assert tpl['slots'][-1]['name'] == T.UNGROUPED
    pops = [p for s in tpl['slots'] for p in s['populations']]
    assert all(p['transforms'] == {'FL1-A': transform_spec('asinh')}
               for p in pops)
    assert T.slots_without_scales(tpl) == []
    # Without a reader, a row's own record is kept; a row with none is named.
    m = _model()
    for _mid, it, _g in m.all_items():
        if it['sample'].endswith('treat_1'):
            it.pop('transforms')
    tpl = T.template_from_model(m)
    assert tpl['slots'][0]['populations'][0]['transforms'] == TF
    assert T.slots_without_scales(tpl) == ['Treat']


def test_glob_patterns_match_the_name_without_its_extension():
    assert T.pattern_matches('*_ctrl_1', '/d/D9_ctrl_1.fcs')
    assert T.pattern_matches('*_1', '/d/a_1.FCS')


def test_an_upper_case_extension_is_not_part_of_the_sample_name():
    m = T.instantiate(T.template_from_model(_model()), {'/d/X_ctrl_1.FCS': 0})
    assert {it['sample'] for _m, it, _g in m.all_items()} == {'X_ctrl_1'}


def test_a_glob_says_as_much_as_its_words():
    tpl = {'slots': [{'name': 'Star', 'pattern': 'ctrl*'},
                     {'name': 'Two', 'pattern': 'ctrl 2h'},
                     {'name': 'Glob', 'pattern': '*_ctrl_*'},
                     {'name': 'Word', 'pattern': 'ctrl'}]}
    got = T.match_files(tpl, ['ctrl_2h.fcs', 'x_ctrl_4h.fcs'])
    assert got['assign'] == {'ctrl_2h.fcs': 1}
    assert got['ambiguous'] == {'x_ctrl_4h.fcs': [2, 3]}
    # A character class is no word of its own.
    tpl = {'slots': [{'name': 'Class', 'pattern': 'ctrl_[0-9]'},
                     {'name': 'Word', 'pattern': 'ctrl'}]}
    assert T.match_files(tpl, ['ctrl_2.fcs'])['ambiguous'] == \
        {'ctrl_2.fcs': [0, 1]}


def test_a_file_only_the_catch_all_slot_takes_says_so():
    tpl = T.template_from_model(_model())
    tpl['slots'][1]['pattern'] = ''
    rows = T.preview_rows(tpl, ['a_ctrl_1.fcs', 'stray.fcs', 'bad_x.fcs'],
                          _cols, _spill)
    assert T.row_status(tpl, rows[0]) == 'ready'
    assert T.row_status(tpl, rows[1]).startswith(
        'ready (only Treat, the slot with no pattern, takes it')
    assert rows[2]['slot'] == 1
    assert T.row_status(tpl, rows[2]).startswith('cannot be read')


def test_the_template_is_stamped_with_the_app_version():
    from openflo.update import current_version
    raw = T.template_to_json(T.template_from_model(_model()))
    assert raw['app_version'] == current_version()


def test_empty_groups_make_no_slot_and_a_nameless_slot_gets_one():
    m = _model()
    m.new_group('Empty')
    tpl = T.template_from_model(m)
    assert [s['name'] for s in tpl['slots']] == ['Ctrl', 'Treat']
    raw = T.template_to_json(tpl)
    del raw['slots'][1]['name']
    assert [s['name'] for s in T.template_from_json(raw)['slots']] == \
        ['Ctrl', 'Slot 2']


def test_populations_with_other_gate_names_stay_apart():
    m = WorkspaceModel()
    gid = m.new_group('G')
    for name in ('Big', 'Small'):
        gates = {'g1': dict(GATES['g1'], name=name)}
        m.add_item(_item(f'x_{name}', 'g1', gates=gates, gate_path=None), gid)
    assert len(T.template_from_model(m)['slots'][0]['populations']) == 2


def test_each_row_gets_its_own_copies_and_each_file_a_colour():
    tpl = T.template_from_model(_model())
    tpl['slots'][0]['populations'][0]['junk'] = 1      # a newer file's key
    m = T.instantiate(tpl, {'/d/a_ctrl_1.fcs': 0, '/d/a_ctrl_2.fcs': 0})
    rows = list(m.all_items())
    rows[1][1]['transforms']['FL1-A']['t'] = 1.0
    assert tpl['slots'][0]['populations'][1]['transforms'] == TF
    assert rows[3][1]['transforms'] == TF
    assert not any('junk' in it for _m, it, _g in rows)
    colours = {it['sample']: it['color'] for _m, it, _g in rows}
    assert len(set(colours.values())) == 2


def _root():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    return root


def test_a_preview_with_no_template_lists_nothing():
    from openflo.ui_run_template import RunTemplatePreview
    root = _root()
    try:
        assert RunTemplatePreview(root, None, ['a.fcs']).rows == []
    finally:
        root.destroy()


def test_panel_saves_scales_or_refuses_without_them(tmp_path, monkeypatch):
    from tkinter import filedialog

    from openflo.flow_sample import FlowSample
    from openflo.sample_prep import stored_transforms
    root = _root()
    try:
        from openflo.workspace import WorkspacePanel
        rng = np.random.default_rng(1)
        path = write_float_fcs(str(tmp_path / 'a_ctrl_1.fcs'),
                               rng.lognormal(7, 1, (300, 3)), CH)
        live = FlowSample(path)
        live.apply_transform()
        live.retransform(['FL1-A'], method='asinh')
        panel = WorkspacePanel(root, editor=SimpleNamespace(
            _samples={'a_ctrl_1': live}))
        m = WorkspaceModel()
        gid = m.new_group('Ctrl')
        m.add_item(_item('a_ctrl_1', 'g1'), gid)
        m.add_item(_item('b_ctrl_1', 'g1'), gid)     # same population
        loose = _item('c_spare', 'g2')
        loose.pop('transforms')
        m.add_item(loose)
        panel.model = m
        out = tmp_path / 't.json'
        monkeypatch.setattr(filedialog, 'asksaveasfilename',
                            lambda **kw: str(out))
        panel._save_template()
        assert panel.status_var.get() == (
            "No template saved: the scales the gates of Ungrouped were "
            "drawn on are not known. Open one of its samples in the editor, "
            "then save again.")
        assert not out.exists()
        m.loose.clear()
        m.add_item(_item('c_spare', 'g2'))       # a template's row: its own
        panel._save_template()
        saved = json.loads(out.read_text(encoding='utf-8'))
        assert saved['slots'][0]['populations'][0]['transforms'] == \
            json.loads(json.dumps(stored_transforms(live)))
        assert saved['slots'][1]['populations'][0]['transforms'] == \
            json.loads(json.dumps(TF))
        # A group matrix held as an array is saved as nested lists.
        m.set_group_comp(gid, {'kind': 'matrix', 'matrix': np.eye(2) * 0.5,
                               'channels': ['FL1-A', 'FL2-A']})
        panel._save_template()
        saved = json.loads(out.read_text(encoding='utf-8'))
        assert saved['slots'][0]['comp']['matrix'] == [[0.5, 0.0], [0.0, 0.5]]
        # A template that will not serialise leaves the saved one whole.
        before = out.read_bytes()
        m.set_group_comp(gid, {'kind': 'matrix', 'matrix': object()})
        panel._save_template()
        assert panel.status_var.get().startswith('Template not saved: TypeError')
        assert out.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['a_ctrl_1.fcs',
                                                              't.json']
        # A file that cannot be written is said, not raised.
        monkeypatch.setattr(filedialog, 'asksaveasfilename',
                            lambda **kw: str(tmp_path / 'no' / 't.json'))
        panel._save_template()
        assert panel.status_var.get().startswith(
            'Template not saved: FileNotFoundError')
    finally:
        root.destroy()


def test_panel_warns_of_a_template_from_a_newer_openflo(tmp_path, monkeypatch):
    from tkinter import filedialog, messagebox
    root = _root()
    try:
        from openflo.workspace import WorkspacePanel
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        raw = T.template_to_json(T.template_from_model(_model()))
        raw['app_version'] = '999.0.0'
        tfile = tmp_path / 't.json'
        tfile.write_text(json.dumps(raw), encoding='utf-8')
        warned = []
        monkeypatch.setattr(filedialog, 'askopenfilename', lambda **kw: str(tfile))
        monkeypatch.setattr(filedialog, 'askopenfilenames', lambda **kw: ())
        monkeypatch.setattr(messagebox, 'showwarning',
                            lambda *a, **kw: warned.append(a[0]))
        panel._new_run_from_template()
        assert warned == ['Run template from a newer OpenFlo']
    finally:
        root.destroy()


def test_run_controls_read_no_editor_sample_for_a_template_row(day_files,
                                                                monkeypatch):
    import openflo.async_task as A
    monkeypatch.setattr(A, 'run_async', lambda *a, **kw: None)
    path = day_files[0][0]
    m = T.instantiate(T.template_from_model(_model()), {path: 0})
    it = next(it for _m, it, _g in m.all_items())
    root = _root()
    try:
        from openflo.workspace import WorkspacePanel
        for src, read in (('/elsewhere/x.fcs', False), (path, False)):
            other = SimpleNamespace(path=src, source_fcs=None,
                                    comp_channels=['FL9-A'])
            panel = WorkspacePanel(root, editor=SimpleNamespace(
                _samples={it['sample']: other}))
            member = {'item': it, 'comp': {'kind': 'beads'}}
            panel._start_run_controls([{'members': [member]}])
            assert ('channels' in member) is read
    finally:
        root.destroy()


# ── compensation: each day's file its own ──────────────────────────────────

OWN = np.array([[1.0, 0.30], [0.02, 1.0]])


def _spilled(tmp_path, name, spill=True):
    rng = np.random.default_rng(5)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 400),
                          rng.lognormal(7, 1, 400), rng.lognormal(7, 1, 400)])
    meta = {'$SPILLOVER': '2,FL1-A,FL2-A,1,0.30,0.02,1'} if spill else {}
    return write_float_fcs(str(tmp_path / name), ev, CH, metadata_dict=meta)


def test_the_template_carries_no_days_matrix():
    tpl = T.template_from_model(_model())
    for pop in (p for s in tpl['slots'] for p in s['populations']):
        assert 'comp_matrix' not in pop and 'comp_channels' not in pop
    m = T.instantiate(tpl, {'/d/a_ctrl_1.fcs': 0})
    assert all('comp_matrix' not in it for _m, it, _g in m.all_items())


def test_a_template_row_runs_on_its_own_files_spillover(tmp_path):
    # The day-2 file carries 0.30/0.02 spill; a matrix left on the row (as
    # an older template saved) is not used.
    day2 = _spilled(tmp_path, 'd2_ctrl_1.fcs')
    ed = SimpleNamespace(_samples={})
    it = _item('d2_ctrl_1', path=day2, template_slot='Ctrl')
    s, note = R.resolve_run_sample(ed, it)
    np.testing.assert_allclose(np.asarray(s.comp_matrix), OWN)
    assert note == 'reloaded from file, compensated with its own $SPILLOVER'
    # A row from no template keeps the matrix it saved.
    plain = {k: v for k, v in it.items() if k != 'template_slot'}
    s, note = R.resolve_run_sample(ed, plain)
    np.testing.assert_allclose(np.asarray(s.comp_matrix), M)
    assert note == 'reloaded from file'
    bare = _spilled(tmp_path, 'd2_ctrl_2.fcs', spill=False)
    warns = []
    it = _item('d2_ctrl_2', path=bare, template_slot='Ctrl')
    s, note = R.resolve_run_sample(ed, it, warnings=warns)
    assert not s.is_compensated()
    assert note.endswith('uncompensated (no $SPILLOVER in its file)')
    assert warns == ["'d2_ctrl_2': its file has no usable $SPILLOVER, so it "
                     "ran uncompensated"]
    R.resolve_run_sample(ed, it)                     # no list: nothing raised


def test_a_template_rows_compensation_is_its_files_own_unless_overridden():
    m = T.instantiate(T.template_from_model(_model()), {'/d/a_ctrl_1.fcs': 0})
    mid = next(mid for mid, _it, _g in m.all_items())
    # The template's slot set comp beads: those stay.
    assert m.effective_comp(mid)['kind'] == 'beads'
    m.groups[m.item_group(mid)]['comp'] = None
    c = m.effective_comp(mid)
    assert (c['source'], c['ready'], c['label']) == ('file', True,
                                                     'own $SPILLOVER')
    assert R.comp_words(c) == "the file's own $SPILLOVER"
    assert m.run_blockers() == []
    m.item(mid)['comp_override'] = {'kind': 'matrix', 'matrix': M,
                                    'channels': ['FL1-A', 'FL2-A']}
    assert m.effective_comp(mid)['source'] == 'row'


def test_the_preview_says_how_each_file_is_compensated(tmp_path):
    tpl = T.template_from_model(_model())
    tpl['slots'][0]['comp'] = {'kind': 'matrix', 'matrix': M}
    tpl['slots'][1]['comp'] = None
    rows = T.preview_rows(tpl, ['a_ctrl_1.fcs', 'a_treat_1.fcs',
                                'nospill_treat.fcs', 'bad_treat.fcs',
                                'a_x.fcs'], _cols, _spill)
    words = [T.row_comp_words(tpl, r) for r in rows]
    assert words == ["the slot's matrix", "the file's own $SPILLOVER",
                     'none: the file has no $SPILLOVER', '', '']
    assert T.row_status(tpl, rows[2]) == \
        'ready (it has no $SPILLOVER, so it runs uncompensated)'
    rows[1]['slot'] = T.SKIP
    assert T.row_comp_words(tpl, rows[1]) == ''
    tpl['slots'][0]['comp'] = {'kind': 'beads', 'files': ['b.fcs']}
    assert T.row_comp_words(tpl, rows[0]) == "the slot's comp beads"
    # A population's own override wins, as at run (resolve_comp).
    pops = tpl['slots'][1]['populations']
    pops[1]['comp_override'] = {'kind': 'matrix', 'matrix': M}
    assert T.row_comp_words(tpl, rows[2]) == (
        "by population: none: the file has no $SPILLOVER; the row's matrix")
    assert T.row_status(tpl, rows[2]) == \
        'ready (it has no $SPILLOVER, so it runs uncompensated)'
    pops[0]['comp_override'] = {'kind': 'beads', 'files': ['b.fcs']}
    pops[1]['comp_override'] = {'kind': 'beads', 'files': ['b.fcs']}
    assert T.row_comp_words(tpl, rows[2]) == "the row's comp beads"
    assert T.row_status(tpl, rows[2]) == 'ready'
    tpl['slots'][1]['comp'] = {'kind': 'matrix', 'matrix': M}
    pops[1]['comp_override'] = None
    assert T.row_comp_words(tpl, rows[2]) == (
        "by population: the row's comp beads; the slot's matrix")
    # The run agrees with the preview.
    m = T.instantiate(tpl, {'/d/a_treat_1.fcs': 1})
    assert [m.effective_comp(mid)['source'] for mid, _i, _g in
            m.all_items()] == ['row', 'group']
    a, b = _spilled(tmp_path, 'x1.fcs'), _spilled(tmp_path, 'x2.fcs', False)
    assert T.has_spillover(a) and not T.has_spillover(b)


def test_the_preview_shows_the_compensation_column():
    from openflo.ui_run_template import RunTemplatePreview
    root = _root()
    try:
        tpl = T.template_from_model(_model())
        win = RunTemplatePreview(root, tpl, ['a_ctrl_1.fcs', 'a_x.fcs'],
                                 channels_of=_cols, spill_of=_spill)
        assert win.tree.item('0')['values'][2] == "the slot's comp beads"
        assert win.tree.item('1')['values'][2] == ''
    finally:
        root.destroy()


def test_an_all_linear_scales_record_is_kept_and_used(tmp_path):
    # {} says every channel is linear: it is a record, not a missing one.
    rng = np.random.default_rng(9)
    ev = np.column_stack([rng.uniform(2e4, 2e5, 1500),
                          rng.lognormal(7, 1.5, 1500),
                          rng.lognormal(7, 1.5, 1500)])
    path = write_float_fcs(str(tmp_path / 'd2_ctrl_1.fcs'), ev, CH)
    gate = {'kind': 'threshold', 'name': 'Pos', 'channel': 'FL1-A',
            'value': 1000.0, 'id': 'p', 'parent_id': None}
    m = WorkspaceModel()
    gid = m.new_group('Ctrl')
    row = _item('d1_ctrl_1', 'p', gate_chain=['p'], gates={'p': gate})
    del row['transforms']           # only the reader gives the record
    m.add_item(row, gid)
    tpl = T.template_from_model(m, lambda it: {})
    assert tpl['slots'][0]['populations'][0]['transforms'] == {}
    assert T.slots_without_scales(tpl) == []
    back = T.template_from_json(json.loads(json.dumps(T.template_to_json(tpl))))
    day = T.instantiate(back, {path: 0})
    it = next(it for _m, it, _g in day.all_items())
    assert it['transforms'] == {}

    def on_logicle(s):
        s.retransform(['FL1-A', 'FL2-A'], method='logicle')
    want = int((ev[:, 1] > 1000.0).sum())
    for editor in (SimpleNamespace(_samples={}),
                   SimpleNamespace(_samples={},
                                   _conform_sample_transforms=on_logicle)):
        s, _note = R.resolve_run_sample(editor, it)
        assert len(s.data) == want
