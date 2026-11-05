"""Analyze > Plate view: a 96-well heatmap of the event count or a gate's
count or percent of parent. Wells come from the well keyword or the sample's
name, and a gate that cannot be computed reads NA.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import numpy as np
import pytest

import tests._sample_editor as shared
from openflo import sample_meta as sm
from tests._sample_editor import _mask_count, _s, _tag

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


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
    _root, ed = editor
    ed._samples['beta_1'].name = 'Plate1_C07'
    ed._samples['Plate1_C07'] = ed._samples.pop('beta_1')
    ed._sample_order[ed._sample_order.index('beta_1')] = 'Plate1_C07'
    win = PlateWindow(ed)
    assert win.wells.get('Plate1_C07') == (2, 6)
    win.destroy()


def test_plate_view_reads_na_for_a_population_that_cannot_be_computed(
        editor, tmp_path, monkeypatch):
    """A gate on FL9-A, a channel the sample lacks, selects nothing
    (fail-closed), but that 0 is not a measurement: its well, and its child's
    well, read NA (grey) in the view and in the CSV, as in Statistics. A
    computable gate in the same sample still gets its real count."""
    from openflo import ui_plate
    from openflo.ui_plate import PlateWindow
    root, ed = editor
    _tag(ed, alpha_1={'WELL ID': 'A01'})
    gates = ed._sample_gates['alpha_1']
    gates['miss'] = {'kind': 'threshold', 'channel': 'FL9-A', 'op': '>',
                     'value': 0.0,
                     'parent_id': None, 'name': 'Missing chan',
                     'enabled': True}
    gates['kid'] = {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                    'value': 0.0,
                    'parent_id': 'miss', 'name': 'Kid', 'enabled': True}
    win = PlateWindow(ed)
    root.update()
    for stat in ("Missing chan: count", "Missing chan: % of parent",
                 "Kid: count", "Kid: % of parent"):
        win.stat_var.set(stat)
        win.draw()
        assert win.values()['alpha_1'] is None, stat
        assert np.isnan(win.matrix()[0, 0]), stat
    win.stat_var.set("FL4 pos: count")
    want = int((ed._samples['alpha_1'].data['FL4-A'] > 25000.0).sum())
    assert win.values()['alpha_1'] == want > 0
    win.stat_var.set("Missing chan: count")
    out = tmp_path / 'plate.csv'
    monkeypatch.setattr(ui_plate, 'ask_csv_path', lambda *a, **k: str(out))
    monkeypatch.setattr(ui_plate, 'report_saved', lambda *a, **k: None)
    win._export_csv()
    lines = out.read_text(encoding='utf-8-sig').splitlines()
    assert lines[1].split(',')[:3] == ['A01', 'alpha_1', 'NA'], lines
    win.destroy()
