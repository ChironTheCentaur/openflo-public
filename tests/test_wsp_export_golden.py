"""Golden snapshot of a complete .wsp export.

The .wsp is what FlowJo reads, and FlowJo 10.10.2 is sensitive to its exact
form: a missing per-sample <Keywords> block, a QuadrantGate or an
EllipsoidGate each changed or dropped what FlowJo computed. None of the
numeric golden checks (openflo-selftest, test_golden_regression) look at the
exported XML, so an unintended change to it would ship silently. This pins
the whole document for one sample that uses every gate kind the writer
handles, plus Time, compensation and Keywords.

After an INTENDED change to the export, regenerate and review the diff:

    OPENFLO_UPDATE_WSP_GOLDEN=1 pytest tests/test_wsp_export_golden.py

Normalised before comparing (they differ on every run): modDate, matrix
transforms:id values, and the DataSet uri (a temporary path).
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

import openflo.pipeline as fp

GOLDEN = Path(__file__).parent / 'golden' / 'wsp_export.xml'
CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
SPILL = '2,FL1-A,FL2-A,1,0.12,0.03,1'


def _fcs(tmp_path):
    import flowio
    rng = np.random.default_rng(20261001)
    n = 400
    ev = np.column_stack([
        rng.uniform(1e4, 2e5, n), rng.uniform(5e3, 1.5e5, n),
        rng.normal(800, 600, n), rng.normal(1500, 900, n),
        np.arange(n, dtype=float) * 50.0,
    ]).astype(np.float32)
    path = tmp_path / 'golden.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), CHANNELS,
                          metadata_dict={'TIMESTEP': '0.01', 'SPILL': SPILL})
    return str(path)


def _gates():
    big = 1e12
    g = [
        {'id': 't', 'parent_id': None, 'kind': 'interval', 'channel': 'Time',
         'lo': 100.0, 'hi': 15000.0, 'name': 'Time'},
        {'id': 'cells', 'parent_id': 't', 'kind': 'polygon', 'name': 'Cells',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'vertices': [[2e4, -2000.0], [1.9e5, -2000.0], [1.9e5, 1.4e5], [2e4, 1.4e5]]},
        {'id': 'big', 'parent_id': 'cells', 'kind': 'rect', 'name': 'Large',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 5e4, 'x1': 2.6e5, 'y0': 0.0, 'y1': 2.6e5},
        {'id': 'fl1', 'parent_id': 'cells', 'kind': 'threshold', 'name': 'FL1+',
         'channel': 'FL1-A', 'value': 1000.0},
        {'id': 'ell', 'parent_id': 'cells', 'kind': 'ellipsoid', 'name': 'Blob',
         'x_channel': 'FL1-A', 'y_channel': 'FL2-A', 'mean': [800.0, 1500.0],
         'cov': [[3.6e5, 5e4], [5e4, 8.1e5]], 'distance_sq': 4.0},
    ]
    for gid, label, x0, x1, y0, y1 in [('q1', 'FL1- FL2+', -big, 900.0, 1200.0, big),
                                       ('q2', 'FL1+ FL2+', 900.0, big, 1200.0, big),
                                       ('q3', 'FL1+ FL2-', 900.0, big, -big, 1200.0),
                                       ('q4', 'FL1- FL2-', -big, 900.0, -big, 1200.0)]:
        g.append({'id': gid, 'parent_id': 'cells', 'kind': 'rect', 'label': label,
                  'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
                  'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'quad_set': 'qs',
                  'quad_origin_x': 900.0, 'quad_origin_y': 1200.0})
    g.append({'id': 'kid', 'parent_id': 'q2', 'kind': 'interval', 'name': 'Dim SSC',
              'channel': 'SSC-A', 'lo': 5000.0, 'hi': 5e4})
    return g


def _normalise(xml):
    xml = re.sub(r'modDate="[^"]*"', 'modDate="(date)"', xml)
    ids = {}
    xml = re.sub(r'transforms:id="([^"]*)"',
                 lambda m: f'transforms:id="matrix-{ids.setdefault(m.group(1), len(ids) + 1)}"',
                 xml)
    return re.sub(r'<DataSet uri="[^"]*"', '<DataSet uri="file:/(fcs)"', xml)


def test_wsp_export_matches_golden(tmp_path):
    fcs = _fcs(tmp_path)
    w = fp.WspWriter(cytometer='Golden')
    w.add_sample('golden sample', fcs, CHANNELS, _gates(),
                 compensation=fp.read_compensation_matrix(fcs))
    out = tmp_path / 'golden.wsp'
    w.write(str(out))
    got = _normalise(out.read_text(encoding='utf-8')) + '\n'
    # The warnings are part of the contract: where FlowJo will differ.
    OPEN = ("at the bottom of FlowJo's axis: FlowJo treats it as open and also "
            "counts every event below it")
    assert [m.split(': ', 1)[1] for m in w.warnings] == [
        f"gate 'Time' lower edge 1 on Time is {OPEN}",
        f"gate 'Cells' lower edge -2000 on SSC-A is {OPEN}",
        f"gate 'Large' lower edge 0 on SSC-A is {OPEN}",
        "ellipse 'Blob' exported as a 64-vertex polygon (FlowJo drops a sample "
        "with an ellipse gate); FlowJo decides polygon membership on a display "
        "grid, so counts near the edge can differ",
    ], w.warnings
    if os.environ.get('OPENFLO_UPDATE_WSP_GOLDEN'):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(got, encoding='utf-8', newline='\n')
        pytest.skip(f'golden rewritten: {GOLDEN}')
    assert GOLDEN.exists(), 'no golden yet: run with OPENFLO_UPDATE_WSP_GOLDEN=1'
    want = GOLDEN.read_text(encoding='utf-8')
    assert got == want, (
        'the .wsp export changed. If intended, regenerate with '
        'OPENFLO_UPDATE_WSP_GOLDEN=1 and review the diff before committing.')
