"""Gate coordinates cross the .wsp boundary in the right units.

A FlowJo workspace holds gate coordinates in linear scale units (compensated
intensity); OpenFlo's editor and CLI store fluorescence logicle-transformed and
keep gates in those stored coordinates. Nothing converted between the two:

* import: a FlowJo "FL1-A > 1000" was compared with logicle values that never
  exceed ~1.2 -- 7,653 events in FlowJo, 0 in OpenFlo;
* export: an OpenFlo logicle threshold of 0.454 reached FlowJo as an
  intensity of 0.454 -- 7,653 events in OpenFlo, 14,087 in FlowJo.

Every test here counts events both ways: OpenFlo's gate on the stored
(logicle) data must select the same events as the FlowJo-side gate on the
linear data.
"""
from __future__ import annotations

import types
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp

FL = ['FL1-A', 'FL2-A']


def _data(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    lin = np.column_stack([
        np.concatenate([rng.normal(0, 300, n // 2),
                        rng.lognormal(np.log(8000), 0.6, n - n // 2)]),
        np.concatenate([rng.lognormal(np.log(3000), 0.8, n // 2),
                        rng.normal(-50, 400, n - n // 2)]),
    ])
    lin_df = pd.DataFrame(lin, columns=FL)
    stored = lin_df.copy()
    for c in FL:
        stored[c] = fp.transform_values(lin_df[c].to_numpy(), method='logicle')
    return lin_df, stored


LIN, STORED = _data()
LOGICLE = {c: 'logicle' for c in FL}
TO_LIN = fp.gate_scale_fns(LOGICLE, 'to_linear')
FROM_LIN = fp.gate_scale_fns(LOGICLE, 'from_linear')

FLOWJO_GATES = [
    {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1000.0},
    {'kind': 'interval', 'channel': 'FL2-A', 'lo': -1e12, 'hi': 500.0},
    {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'x0': 1000.0, 'x1': 1e12, 'y0': -1e12, 'y1': 1000.0},
    {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'vertices': [[2000.0, -2000.0], [60000.0, -2000.0], [60000.0, 5000.0],
                  [2000.0, 5000.0]]},
]


def _n(g, df):
    return int(fp.gate_to_mask(g, df).sum())


@pytest.mark.parametrize('g', FLOWJO_GATES, ids=lambda g: g['kind'])
def test_an_imported_flowjo_gate_selects_the_same_events(g):
    want = _n(g, LIN)
    assert want > 100, 'test gate should select something'
    assert _n(g, STORED) != want, 'unconverted, the gate is wrong'
    got = _n(fp.map_gate_coords(g, FROM_LIN), STORED)
    # 1-D and rectangle bounds map exactly; a polygon's straight linear edges
    # are curves in logicle space and are traced with 16 steps per edge.
    tol = 0 if g['kind'] != 'polygon' else max(3, int(0.002 * want))
    assert abs(got - want) <= tol, (g['kind'], got, want)


@pytest.mark.parametrize('g', FLOWJO_GATES, ids=lambda g: g['kind'])
def test_an_exported_openflo_gate_selects_the_same_events(g):
    """Round trip: an OpenFlo gate (stored coords) exported to linear and
    evaluated FlowJo's way on linear data."""
    openflo_gate = fp.map_gate_coords(g, FROM_LIN)
    want = _n(openflo_gate, STORED)
    got = _n(fp.map_gate_coords(openflo_gate, TO_LIN), LIN)
    tol = 0 if g['kind'] != 'polygon' else max(3, int(0.002 * want))
    assert abs(got - want) <= tol, (g['kind'], got, want)


def test_open_bounds_stay_open_and_linear_channels_pass_through():
    g = {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'FL1-A',
         'x0': 5e4, 'x1': 2e5, 'y0': -1e12, 'y1': 0.5}
    out = fp.map_gate_coords(g, TO_LIN)
    assert out['x0'] == 5e4 and out['x1'] == 2e5      # FSC is linear
    assert out['y0'] == -1e12                           # open stays open
    assert out['y1'] == pytest.approx(
        float(fp.inverse_transform_values(np.array([0.5]))[0]))
    assert fp.map_gate_coords(g, {}) is g


def test_an_ellipse_becomes_a_polygon_that_keeps_its_events():
    rng = np.random.default_rng(3)
    pts = rng.normal([0.5, 0.4], [0.08, 0.05], size=(5000, 2))
    df = pd.DataFrame(pts, columns=FL)
    ell = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
           'mean': [0.5, 0.4], 'cov': [[0.0064, 0.0], [0.0, 0.0025]],
           'distance_sq': 4.0}
    want = _n(ell, df)
    poly = fp.map_gate_coords(ell, TO_LIN)
    assert poly['kind'] == 'polygon' and 'mean' not in poly
    lin = df.copy()
    for c in FL:
        lin[c] = fp.inverse_transform_values(df[c].to_numpy())
    assert abs(_n(poly, lin) - want) <= max(5, int(0.01 * want))


def test_wspwriter_writes_linear_units_for_transformed_channels(tmp_path):
    g = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.4543,
         'id': 'g1', 'parent_id': None, 'name': 'FL1+'}
    w = fp.WspWriter(cytometer='T')
    w.add_sample('s', '', ['FL1-A'], [g], transforms={'FL1-A': 'logicle'})
    out = tmp_path / 'x.wsp'
    w.write(str(out))
    mins = [float(v) for e in ET.parse(out).iter()
            for k, v in e.attrib.items() if k.endswith('}min')]
    want = float(fp.inverse_transform_values(np.array([0.4543]))[0])
    assert mins == [pytest.approx(want)]
    assert want > 500, 'a logicle 0.45 is an intensity near a thousand'
    # ...and reads back to the same threshold in linear units
    back = fp.WspReader(str(out)).extract_gates()
    assert back[0]['value'] == pytest.approx(want)


def test_without_transforms_the_writer_is_unchanged(tmp_path):
    g = {'kind': 'interval', 'channel': 'FL1-A', 'lo': 10.0, 'hi': 500.0,
         'id': 'g1', 'parent_id': None}
    w = fp.WspWriter(cytometer='T')
    w.add_sample('s', '', ['FL1-A'], [g])
    assert w.samples[0]['gates'][0] is g


def test_cli_export_marks_fluorescence_gates_as_logicle():
    from openflo.cli import _pipeline_gate_transforms
    tf = _pipeline_gate_transforms([
        {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A'},
        {'kind': 'threshold', 'channel': 'FL3-A'},
        {'kind': 'interval', 'channel': 'Time'}])
    assert tf == {'FL3-A': 'logicle'}


def test_stored_transforms_follow_the_samples_record():
    pytest.importorskip('tkinter')
    from openflo.editor_channels import ChannelsMixin
    ed = types.SimpleNamespace(_samples={})
    s = fp.FlowSample.from_dataframe(LIN.copy(), name='x')
    s.apply_transform(channels=['FL1-A'])
    got = ChannelsMixin._stored_transforms(ed, 'x', s)
    assert list(got) == ['FL1-A'] and got['FL1-A']['method'] == 'logicle'
    csv_like = fp.FlowSample.from_dataframe(STORED.copy(), name='y')
    csv_like.mark_transformed(['FL1-A'], method='asinh')
    assert ChannelsMixin._stored_transforms(ed, 'y', csv_like)[
        'FL1-A']['method'] == 'asinh'


def test_an_unknown_scale_is_not_converted():
    fns = fp.gate_scale_fns({'FL1-A': fp.unknown_spec('file:x.csv'),
                             'FL2-A': fp.transform_spec('logicle')},
                            'to_linear')
    assert list(fns) == ['FL2-A']


def test_bulk_import_keeps_sibling_1d_gates_on_one_channel():
    """Three intervals on FL1-A under one parent are three populations; the
    slider's replace-in-place rule kept only the last of them."""
    pytest.importorskip('tkinter')
    from openflo.editor_gating import GatingMixin

    class _Ed(GatingMixin):
        def __init__(self):
            self._gates, self._gate_id_order = {}, []
            self._seq = 0
            self._suspend_undo = True

        def _checkpoint(self):
            pass

        def _next_color(self):
            return '#000000'

        def _selected_gate_id(self):
            return None

        def _next_gate_id(self):
            self._seq += 1
            return f'g{self._seq}'

    ed = _Ed()
    bands = [(-1e12, 0.2), (0.2, 0.5), (0.5, 1e12)]
    for lo, hi in bands:
        ed._add_gate({'kind': 'interval', 'channel': 'FL1-A', 'lo': lo,
                      'hi': hi, 'parent_id': None}, replace_1d=False)
    assert len(ed._gates) == 3
    # the interactive default still moves the one gate
    ed2 = _Ed()
    for lo, hi in bands:
        ed2._add_gate({'kind': 'interval', 'channel': 'FL1-A', 'lo': lo,
                       'hi': hi, 'parent_id': None})
    assert len(ed2._gates) == 1
    # a set limits replacement to the gates that existed before
    ed3 = _Ed()
    first = ed3._add_gate({'kind': 'interval', 'channel': 'FL1-A', 'lo': 0,
                           'hi': 1, 'parent_id': None})
    before = set(ed3._gates)
    for lo, hi in bands:
        ed3._add_gate({'kind': 'interval', 'channel': 'FL1-A', 'lo': lo,
                       'hi': hi, 'parent_id': None}, replace_1d=before)
    assert len(ed3._gates) == 3 and first in ed3._gates


def test_editor_wsp_import_end_to_end(tmp_path):
    """The real editor: ingest a FlowJo workspace whose sample carries its own
    matrix (the FCS's $SPILL is a different one) and whose gates are in
    linear units, load the sample, and count each population the way FlowJo
    does (compensated linear data, linear gates)."""
    import flowio

    from tests.test_gui_smoke import _editor_or_skip
    ch = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
    rng = np.random.default_rng(5)
    n = 4000
    true = np.zeros((n, 2))
    true[: n // 2, 0] = rng.lognormal(np.log(8000), 0.5, n // 2)
    true[n // 2:, 1] = rng.lognormal(np.log(5000), 0.5, n - n // 2)
    true += rng.normal(0, 150, true.shape)
    own = np.array([[1.0, 0.25], [0.08, 1.0]])       # the workspace's matrix
    acq = np.array([[1.0, 0.05], [0.30, 1.0]])       # the file's $SPILL
    ev = np.column_stack([rng.uniform(5e4, 1e5, n), rng.uniform(1e4, 5e4, n),
                          true @ own, np.arange(n) / 10.0]).astype(np.float32)
    fcs = str(tmp_path / 'a.fcs')
    with open(fcs, 'wb') as f:
        flowio.create_fcs(f, ev.ravel().tolist(), ch, metadata_dict={
            'SPILL': '2,FL1-A,FL2-A,' + ','.join(map(str, acq.ravel()))})
    gates = [
        {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1000.0,
         'id': 'a', 'parent_id': None, 'name': 'FL1+'},
        {'kind': 'interval', 'channel': 'FL2-A', 'lo': -1e12, 'hi': 1000.0,
         'id': 'b', 'parent_id': 'a', 'name': 'FL2-'},
        {'kind': 'interval', 'channel': 'FL1-A', 'lo': -1e12, 'hi': 1000.0,
         'id': 'c', 'parent_id': None, 'name': 'FL1-'},
    ]
    w = fp.WspWriter(cytometer='T')
    w.add_sample('a', fcs, ch, gates, compensation=(FL, own))
    wsp = str(tmp_path / 'a.wsp')
    w.write(wsp)

    # FlowJo's semantics: own matrix, linear data, linear gates.
    ref = fp.FlowSample(fcs)
    ref.manual_compensate(own, FL)
    by_id = {g['id']: dict(g) for g in gates}
    want = {g['name']: int(fp.cumulative_gate_mask(by_id, g['id'],
                                                   ref.data).sum())
            for g in gates}

    root, ed, _gui = _editor_or_skip()
    try:
        queued = []
        ed._queue_fcs_loads = lambda paths, **_k: queued.extend(paths)
        ed._ingest_wsp(wsp)
        assert queued == [fcs]
        name = ed._sample_name_for(fcs)
        ed._load_worker(name, fcs)
        root.update()
        s = ed._samples[name]
        np.testing.assert_allclose(s.comp_matrix, own)
        g = ed._sample_gates[name]
        assert len(g) == 3, 'sibling 1-D gates on FL1-A must all survive'
        got = {(gd.get('name') or gd.get('label')):
               int(fp.cumulative_gate_mask(g, gid, s.data).sum())
               for gid, gd in g.items()}
        # The editor runs QC on load (FlowJo does not), so allow for the few
        # events it removes; without the conversion FL1+ held 0 events.
        for k, v in want.items():
            assert abs(got[k] - v) <= 0.02 * len(ref.data), (k, got[k], v)
            assert got[k] > 100
    finally:
        try:
            root.destroy()
        except Exception:
            pass
