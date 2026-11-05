"""File → Export gate list (CSV): every gate with its population path and its
coordinates as stored and in linear units.

Generated gates on generic channels; FL1-A carries an asinh record (cofactor
150), so its stored threshold 1.0 is 150·sinh(1.0) = 176.28 in linear units,
FSC-A has no record and stays as stored."""
from __future__ import annotations

import csv
import json
import math

from openflo.gate_table import COLUMNS, gate_rows, write_csv

GATES = {
    'g1': {'kind': 'rect', 'name': 'Cells', 'parent_id': None,
           'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
           'x0': 10.0, 'x1': 900.0, 'y0': 5.0, 'y1': 800.0},
    'g2': {'kind': 'threshold', 'name': 'Pos', 'parent_id': 'g1',
           'channel': 'FL1-A', 'value': 1.0, 'enabled': False},
}
REC = {'FL1-A': {'method': 'asinh', 'cofactor': 150.0}}


def test_rows_paths_and_linear_coordinates():
    rows = gate_rows('s1', GATES, ['g1', 'g2'], transforms=REC)
    assert [r['Population'] for r in rows] == ['Cells', 'Cells/Pos']
    assert rows[1]['Parent'] == 'Cells'
    assert rows[1]['Enabled'] is False
    assert json.loads(rows[1]['Coordinates (stored)']) == {'value': 1.0}
    lin = json.loads(rows[1]['Coordinates (linear)'])['value']
    assert math.isclose(lin, 150.0 * math.sinh(1.0), rel_tol=1e-9)
    # No record on FSC-A / SSC-A: the rectangle is the same both ways.
    assert rows[0]['Coordinates (linear)'] == rows[0]['Coordinates (stored)']


def test_unknown_scale_is_said_not_converted():
    rows = gate_rows('s1', GATES, transforms={'FL1-A': {'method': 'unknown'}})
    assert 'unknown scale, as stored: FL1-A' in rows[1]['Coordinates (linear)']


def test_csv_round_trip(tmp_path):
    path = tmp_path / 'gates.csv'
    write_csv(str(path), gate_rows('s1', GATES, transforms=REC)
              + gate_rows('s2', {'g1': GATES['g1']}))
    with open(path, newline='', encoding='utf-8') as fh:
        got = list(csv.DictReader(fh))
    assert tuple(got[0]) == COLUMNS
    assert [(r['Sample'], r['Population'], r['Enabled']) for r in got] == [
        ('s1', 'Cells', 'yes'), ('s1', 'Cells/Pos', 'no'),
        ('s2', 'Cells', 'yes')]


def test_file_menu_export_lists_every_samples_gates(tmp_path, monkeypatch):
    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.editor_export as ee
        _load(ed, 's1', 1, 300, 300.0)
        _load(ed, 's2', 2, 300, 500.0, extra_gate=True)
        out = tmp_path / 'gate_list.csv'
        monkeypatch.setattr(ee.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        ed._export_gate_list()
        with open(out, newline='', encoding='utf-8') as fh:
            got = [(r['Sample'], r['Population']) for r in csv.DictReader(fh)]
        assert got == [('s1', 'Cells'), ('s1', 'Cells/Pos'),
                       ('s2', 'Other'), ('s2', 'Cells'), ('s2', 'Cells/Pos')]
        assert ed.status_var.get().startswith('Exported 5 gate(s) of 2 ')
    finally:
        root.destroy()
