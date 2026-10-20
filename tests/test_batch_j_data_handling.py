"""Batch J (Complaints 17, 26): sample metadata and data handling.

- Edit > Group samples puts samples into groups by an FCS keyword or by a
  sample-to-condition table (CSV or tab separated).
- Analyze > Plate view shows a 96-well heatmap of the event count or a
  gate's count or percent of parent; wells come from the well keyword or
  the sample's name, and a gate that cannot be computed reads NA.
- File > Export populations, all samples writes every sample's populations,
  a folder per sample.
- File > Concatenate samples joins several samples into one, each event
  tagged with its source.
- Downsampling the data (Display + data) can be undone: setting it back
  restores every event, unless the data changed since.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by
/mnt/project-files/batch-j/mutation_check.py.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import tests.test_windows_nothing_hidden as sweep
from openflo import sample_meta as sm

os.environ.setdefault('MPLBACKEND', 'Agg')


def _s(name, **kw):
    return SimpleNamespace(name=name, metadata={k: v for k, v in kw.items()})


# ── Keywords and groups ─────────────────────────────────────────────────────

def test_keywords_are_read_without_dollar_and_in_capitals():
    s = _s('x', **{'$FIL': 'x.fcs', 'tube name': ' ctrl ', 'EMPTY': ''})
    assert sm.keywords(s) == {'FIL': 'x.fcs', 'TUBE NAME': 'ctrl'}


def test_grouping_keywords_offer_only_what_splits_the_samples():
    samples = [
        _s('s1', **{'TUBE NAME': 'ctrl', 'WELL ID': 'A1', '$P1N': 'FL1-A',
                    '$DATE': '01-JAN-2026', 'CONDITION': 'x', '$TOT': '900'}),
        _s('s2', **{'TUBE NAME': 'stim', 'WELL ID': 'A2', '$P1N': 'FL1-A',
                    '$DATE': '02-JAN-2026', 'CONDITION': 'x', '$TOT': '500'}),
        _s('s3', **{'TUBE NAME': 'ctrl', 'WELL ID': 'A3', '$P1N': 'FL1-A',
                    '$DATE': '03-JAN-2026', 'CONDITION': 'x', '$TOT': '900'}),
    ]
    assert sm.grouping_keywords(samples) == [('TUBE NAME', 2)]
    assert sm.groups_by_keyword(samples, '$tube name') == {
        's1': 'ctrl', 's2': 'stim', 's3': 'ctrl'}
    samples.append(_s('s4'))
    assert sm.groups_by_keyword(samples, 'TUBE NAME')['s4'] == \
        "No TUBE NAME"


def test_condition_table_matches_names_and_file_names():
    names = ['Sample_A', 'sample_b', 'Sample_C']
    text = ("file,donor,condition\n"
            "sample_a.fcs,d1,ctrl\n"
            "SAMPLE_B,d2,stim\n"
            "nobody.fcs,d3,stim\n")
    res = sm.read_condition_table(text, names)
    assert res['groups'] == {'Sample_A': 'ctrl', 'sample_b': 'stim'}
    assert res['annotations']['Sample_A'] == {'donor': 'd1'}
    assert res['unmatched_rows'] == ['nobody.fcs']
    assert res['missing'] == ['Sample_C']
    tabbed = "sample\tgroup\nSample_C\tlate\n"
    assert sm.read_condition_table(tabbed, names)['groups'] == {
        'Sample_C': 'late'}
    with pytest.raises(ValueError):
        sm.read_condition_table("just one column\nSample_A\n", names)


# ── Wells and the plate ─────────────────────────────────────────────────────

@pytest.mark.parametrize('text, pos', [
    ('A1', (0, 0)), ('a01', (0, 0)), ('B_07', (1, 6)), ('H12', (7, 11)),
    ('Plate1_C3', (2, 2)), ('I1', None), ('A13', None), ('XA1', None),
    ('', None), (None, None)])
def test_parse_well(text, pos):
    assert sm.parse_well(text) == pos


def test_well_from_keyword_before_name_and_label():
    assert sm.well_of(_s('Plate1_C03', **{'WELL ID': 'D05'})) == (3, 4)
    assert sm.well_of(_s('Plate1_C03')) == (2, 2)
    assert sm.well_label((2, 2)) == 'C03'
    mat = sm.plate_matrix({(0, 0): 5, (7, 11): 2.5})
    assert mat.shape == (8, 12)
    assert mat[0, 0] == 5 and mat[7, 11] == 2.5
    assert np.isnan(mat).sum() == 94


# ── In the editor ──────────────────────────────────────────────────────────

@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known):
    virtual_screen(1920, 1080)
    root = sweep._root(1.0)
    try:
        yield root, sweep._editor(root)
    finally:
        root.destroy()


virtual_screen = sweep.virtual_screen
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known


def _tag(ed, **per_sample):
    for name, kw in per_sample.items():
        ed._samples[name].metadata = kw


def test_group_samples_by_keyword(editor):
    from openflo.ui_sample_groups import GroupSamplesDialog
    root, ed = editor
    _tag(ed, alpha_1={'TUBE NAME': 'ctrl'}, alpha_2={'TUBE NAME': 'stim'},
         beta_1={'TUBE NAME': 'ctrl'})
    ed._note_saved()
    dlg = GroupSamplesDialog(ed)
    root.update()
    assert [k for k, _n in dlg.choices] == ['TUBE NAME']
    dlg.keyword_combo.current(0)
    dlg.keyword_combo.event_generate('<<ComboboxSelected>>')
    root.update()
    shown = {dlg.tree.item(i, 'values')[0]: dlg.tree.item(i, 'values')[1]
             for i in dlg.tree.get_children()}
    assert shown == {'alpha_1': 'ctrl', 'alpha_2': 'stim', 'beta_1': 'ctrl'}
    dlg.apply()
    root.update()
    assert {n: ed._trial_for(n) for n in ed._sample_order} == shown
    assert sorted(ed._trial_order) == ['ctrl', 'stim']    # Day1 emptied
    assert ed.title().startswith('*')
    assert ed._audit_log.entries()[-1]['action'] == 'samples.group'


def test_group_samples_from_a_table_keeps_the_rest(editor):
    from openflo.ui_sample_groups import GroupSamplesDialog
    root, ed = editor
    dlg = GroupSamplesDialog(ed)
    res = dlg.use_table_text("sample,condition\nalpha_1.fcs,treated\n"
                             "gamma_9,treated\n", 'conditions.csv')
    assert res['unmatched_rows'] == ['gamma_9']
    assert 'gamma_9' in dlg.note_var.get()
    dlg.apply()
    root.update()
    assert ed._trial_for('alpha_1') == 'treated'
    assert ed._trial_for('alpha_2') == 'Day1'           # not in the table
    assert ed._trial_order == ['Day1', 'treated']


def _mask_count(ed, name, gate_name):
    from openflo.pipeline import cumulative_gate_mask
    gates = ed._sample_gates[name]
    gid = next(g for g, d in gates.items() if d['name'] == gate_name)
    return int(np.count_nonzero(cumulative_gate_mask(
        gates, gid, ed._samples[name].data)))


def test_plate_view_places_samples_and_computes_statistics(editor):
    from openflo.ui_plate import PlateWindow
    root, ed = editor
    _tag(ed, alpha_1={'WELL ID': 'B03'}, alpha_2={'WELL ID': 'H12'},
         beta_1={})
    win = PlateWindow(ed)
    root.update()
    assert win.wells == {'alpha_1': (1, 2), 'alpha_2': (7, 11)}
    assert win.no_well == ['beta_1']
    assert 'beta_1' in win.note_label.cget('text')
    mat = win.matrix()
    assert mat[1, 2] == 400 and mat[7, 11] == 400
    assert np.isnan(mat).sum() == 94
    assert "FL2 pos: % of parent" in win.stats
    win.stat_var.set("FL2 pos: % of parent")
    win.draw()
    want = 100 * _mask_count(ed, 'alpha_1', 'FL2 pos') \
        / _mask_count(ed, 'alpha_1', 'FL4 pos')
    vals = win.values()
    assert vals['alpha_1'] == pytest.approx(want)
    assert vals['alpha_2'] is None                 # no such gate: NA, not 0
    assert np.isnan(win.matrix()[7, 11])
    win.stat_var.set("FL4 pos: count")
    assert win.values()['alpha_1'] == _mask_count(ed, 'alpha_1', 'FL4 pos')
    rows = win.rows()
    assert [r[0] for r in rows] == ['B03', 'H12']
    win.destroy()


def test_plate_view_reads_the_well_from_the_name(editor):
    from openflo.ui_plate import PlateWindow
    root, ed = editor
    ed._samples['beta_1'].name = 'Plate1_C07'
    ed._samples['Plate1_C07'] = ed._samples.pop('beta_1')
    ed._sample_order[ed._sample_order.index('beta_1')] = 'Plate1_C07'
    win = PlateWindow(ed)
    assert win.wells.get('Plate1_C07') == (2, 6)
    win.destroy()


# ── Concatenation ──────────────────────────────────────────────────────────

def _frame(seed, n):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'FSC-A': rng.random(n) * 1e5,
                         'FL1-A': rng.random(n) * 1e4,
                         'FL2-A': rng.random(n) * 1e4})


def test_concatenate_samples_tags_each_event():
    from openflo.pipeline import FlowSample
    from openflo.ui_concatenate import SOURCE_COLUMN, concatenate_samples
    a = FlowSample.from_dataframe(_frame(1, 30), name='s_a')
    b = FlowSample.from_dataframe(_frame(2, 20), name='s_b')
    out = concatenate_samples([a, b], 'joined')
    assert len(out.data) == 50
    assert out.data[SOURCE_COLUMN].tolist() == [1] * 30 + [2] * 20
    assert out.data['FL1-A'].tolist() == \
        a.data['FL1-A'].tolist() + b.data['FL1-A'].tolist()
    assert SOURCE_COLUMN not in out.fluor_channels
    assert out.metadata['OPENFLO_CONCAT_SOURCES'] == '1=s_a; 2=s_b'
    c = FlowSample.from_dataframe(_frame(3, 10).drop(columns='FL2-A'),
                                  name='s_c')
    with pytest.raises(ValueError, match='same channels'):
        concatenate_samples([a, c], 'bad')
    d = FlowSample.from_dataframe(_frame(4, 10), name='s_d')
    d.data_transforms = {'FL1-A': {'method': 'logicle'}}
    with pytest.raises(ValueError, match='same scales'):
        concatenate_samples([a, d], 'bad')
    with pytest.raises(ValueError):
        concatenate_samples([a], 'bad')


def test_concatenate_from_the_file_menu(editor, monkeypatch):
    from openflo.pipeline import FlowSample
    from openflo.ui_concatenate import ConcatenateDialog
    root, ed = editor
    for i, n in enumerate(('alpha_1', 'alpha_2', 'beta_1')):
        s = FlowSample.from_dataframe(_frame(i, 100 + i), name=n)
        ed._samples[n] = s
    monkeypatch.setattr(type(ed), '_refresh_gate_list', lambda self: None)
    dlg = ConcatenateDialog(ed)
    dlg.listbox.selection_clear(0, 'end')
    dlg.listbox.selection_set(0)
    dlg.listbox.selection_set(2)
    dlg.apply()
    root.update()
    assert 'Concatenated' in ed._samples
    assert len(ed._samples['Concatenated'].data) == 100 + 102
    assert ed._trial_for('Concatenated') == ed._trial_for('alpha_1')
    assert ed.title().startswith('*')
    dlg = ConcatenateDialog(ed)
    assert dlg.name_var.get() == 'Concatenated 2'
    dlg.listbox.selection_clear(0, 'end')
    dlg.listbox.selection_set(0)
    dlg.apply()
    assert 'two or more' in dlg.note_var.get()
    dlg.destroy()


# ── Population export for every sample ──────────────────────────────────────

def _tot(path):
    raw = open(path, 'rb').read()
    start, end = int(raw[10:18]), int(raw[18:26])
    text = raw[start:end + 1].decode('latin-1')
    parts = text[1:].split(text[0])
    kv = dict(zip(parts[::2], parts[1::2], strict=False))
    return int(kv['$TOT'])


@pytest.fixture
def loaded(tmp_path, monkeypatch):
    import tkinter as tk

    from tests.test_beta_look import _load, _write_fcs
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    try:
        root = tk.Tk()
    except Exception as e:                                 # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    ed = _new_editor(root)
    _load(root, ed, [('sample_a', _write_fcs(tmp_path / 'a.fcs', seed=1)),
                     ('sample_b', _write_fcs(tmp_path / 'b.fcs', seed=2)),
                     ('sample_c', _write_fcs(tmp_path / 'c.fcs', seed=3))])
    yield root, ed, tmp_path
    root.destroy()


def test_export_populations_for_every_sample(loaded, monkeypatch):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    root, ed, tmp = loaded
    for name in ('sample_a', 'sample_b'):
        ed._set_active_sample(name)
        ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A', 'op': '>',
                      'value': 60_000.0, 'name': 'Big'}, audit=False)
    ed._set_active_sample('sample_a')
    out = tmp / 'pops'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs_all()
    assert wait_for_job(ed._job_strip)
    root.update()
    assert sorted(p.name for p in out.iterdir()) == ['sample_a', 'sample_b']
    for name in ('sample_a', 'sample_b'):
        (f,) = list((out / name).glob('*.fcs'))
        assert _tot(f) == _mask_count(ed, name, 'Big')
    assert '2 sample folders' in ed.status_var.get()


def test_export_populations_of_the_active_sample_only(loaded, monkeypatch):
    from openflo import editor_populations
    from openflo.async_task import wait_for_job
    root, ed, tmp = loaded
    for name in ('sample_a', 'sample_c', 'sample_b'):
        ed._set_active_sample(name)
        ed._add_gate({'kind': 'threshold', 'channel': 'FSC-A', 'op': '>',
                      'value': 60_000.0, 'name': 'Big'}, audit=False)
    out = tmp / 'one'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    assert wait_for_job(ed._job_strip)
    assert [p.name for p in out.iterdir()] == ['Big.fcs']
    (f,) = list(out.glob('*.fcs'))
    assert _tot(f) == _mask_count(ed, 'sample_b', 'Big')
    assert _tot(f) != _mask_count(ed, 'sample_c', 'Big')


# ── Undoable downsampling ──────────────────────────────────────────────────

def test_downsampling_the_data_can_be_undone(loaded):
    from tests.test_transform_records import settle
    root, ed, _tmp = loaded
    for n in ed._samples:
        ed._sample_plot_enabled[n] = True
    small = ed._samples['sample_c']
    small.data = small.data.iloc[:1000].reset_index(drop=True)
    full = {n: ed._samples[n].data.copy() for n in ('sample_a', 'sample_b')}
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    settle(root, 0.2)
    assert [len(ed._samples[n].data) for n in ('sample_a', 'sample_b')] \
        == [1000, 1000]
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    settle(root, 0.2)
    for n in ('sample_a', 'sample_b'):
        pd.testing.assert_frame_equal(ed._samples[n].data, full[n])
    assert 'Restored 2 sample(s)' in ed.status_var.get()
    # Data changed after the trim stays as it is: sample_b changed in place
    # (as clustering adds a column), sample_a replaced.
    ed._ds_mode_var.set('Display + data')
    ed._on_ds_mode_changed()
    b = ed._samples['sample_b']
    b.data['FL2-A'] = b.data['FL2-A'] * 2
    ed._data_changed('sample_b')
    a = ed._samples['sample_a']
    a.data = a.data.copy()
    ed._ds_mode_var.set('Off')
    ed._on_ds_mode_changed()
    assert len(ed._samples['sample_a'].data) == 1000
    assert len(ed._samples['sample_b'].data) == 1000
    assert 'sample_a, sample_b changed after the trim' in \
        ed.status_var.get()
