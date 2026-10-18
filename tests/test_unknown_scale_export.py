"""Exporting to FlowJo, a gate on a channel of unknown scale is kept.

A column of an older OpenFlo CSV whose display transform is unknown (asinh
at some cofactor, say) has no transform to undo, so its gates used to reach
FlowJo in display units with only a log line: a cut at asinh 2.0 became an
intensity of 2.0. Now, when the sample's raw FCS can be read, the channel is
matched to it rank for rank (the display transforms are increasing, so this
recovers the transform) and the gate keeps its events in FlowJo; when it
cannot, the gate is still written, as stored. Either way the user is told.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

import openflo.pipeline as fp  # noqa: E402

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
COFACTOR = 5.0          # what an older editor might have used; not knowable


def _linear(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'FSC-A': rng.uniform(2e4, 2e5, n),
        'SSC-A': rng.uniform(1e4, 1e5, n),
        'FL1-A': np.concatenate([rng.normal(0, 40, n // 2),
                                 rng.lognormal(np.log(900), 0.5, n - n // 2)]),
        'FL2-A': rng.lognormal(np.log(300), 1.0, n),
    })


def _stored(lin):
    out = lin.copy()
    for c in ('FL1-A', 'FL2-A'):
        out[c] = np.arcsinh(lin[c].to_numpy() / COFACTOR)
    return out


LIN = _linear()
STORED = _stored(LIN)
UNKNOWN = {'FL1-A': fp.unknown_spec('file:old.csv'),
           'FL2-A': fp.unknown_spec('file:old.csv')}
CUT = {'kind': 'threshold', 'channel': 'FL1-A', 'value': 4.0,
       'id': 'g1', 'parent_id': None, 'name': 'FL1pos'}
POLY = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'vertices': [[3.0, 3.0], [7.0, 3.5], [6.5, 8.0], [3.5, 7.0]],
        'id': 'g2', 'parent_id': None, 'name': 'Blob'}


def _n(g, df):
    return int(fp.gate_to_mask(g, df).sum())


def _fns():
    return {c: fp.rank_matched_fn(STORED[c], LIN[c]) for c in ('FL1-A',
                                                             'FL2-A')}


def test_rank_matching_recovers_the_unknown_transform():
    f = _fns()['FL1-A']
    v = np.array([1.0, 3.0, 5.0, 6.0])
    want = np.sinh(v) * COFACTOR
    assert np.allclose(f(v), want, rtol=0.03, atol=2.0)
    # past the data the map keeps rising, so a bound past every event stays so
    top = STORED['FL1-A'].max()
    assert f(np.array([top + 1.0]))[0] > LIN['FL1-A'].max()
    assert fp.rank_matched_fn([1.0, 1.0], [1.0, 2.0]) is None


@pytest.mark.parametrize('g', [CUT, POLY], ids=['threshold', 'polygon'])
def test_a_matched_gate_keeps_its_events_in_flowjo(g):
    w = fp.WspWriter(cytometer='T')
    w.add_sample('s', '', COLS, [dict(g)], transforms=UNKNOWN,
                 unknown_fns=_fns())
    out = w.samples[0]['gates'][0]
    before, after = _n(g, STORED), _n(out, LIN)
    assert before > 100
    # a cut exactly; a polygon to within its re-traced edges
    assert abs(after - before) <= (0 if g is CUT else max(3, before // 200))
    assert w.scale_notes and 'rank for rank' in w.scale_notes[0]


def test_without_an_fcs_the_gate_is_written_as_stored_and_flagged():
    w = fp.WspWriter(cytometer='T')
    w.add_sample('s', '', COLS, [dict(CUT)], transforms=UNKNOWN)
    assert w.samples[0]['gates'][0]['value'] == 4.0
    assert len(w.scale_notes) == 1
    assert 'written as stored' in w.scale_notes[0]


def test_a_gate_imported_in_flowjo_units_goes_back_unchanged():
    """#42 keeps a FlowJo gate on an unknown-scale channel in FlowJo's units
    (flagged scale_unknown); only its known channels were converted."""
    tf = {'FL1-A': fp.unknown_spec(), 'FL2-A': fp.transform_spec('logicle')}
    rect = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'x0': 500.0, 'x1': 2000.0,
            'y0': float(fp.transform_values(np.array([100.0]))[0]),
            'y1': float(fp.transform_values(np.array([1000.0]))[0]),
            'scale_unknown': True, 'id': 'r', 'parent_id': None}
    w = fp.WspWriter(cytometer='T')
    w.add_sample('s', '', COLS, [rect], transforms=tf,
                 unknown_fns=_fns())
    out = w.samples[0]['gates'][0]
    assert (out['x0'], out['x1']) == (500.0, 2000.0)
    assert out['y0'] == pytest.approx(100.0, rel=1e-6)
    assert out['y1'] == pytest.approx(1000.0, rel=1e-6)
    assert not w.scale_notes


def test_editor_export_matches_to_the_source_fcs_and_warns(tmp_path,
                                                           monkeypatch):
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from test_gui_smoke import _editor_or_skip

    from openflo.fcs_export import write_fcs
    fcs = tmp_path / 'raw.fcs'
    write_fcs(LIN, str(fcs))
    root, ed, _gui = _editor_or_skip()
    out = tmp_path / 'x.wsp'
    shown = []
    monkeypatch.setattr(fd, 'asksaveasfilename', lambda *a, **k: str(out))
    monkeypatch.setattr(mb, 'askyesnocancel', lambda *a, **k: True)
    monkeypatch.setattr(mb, 'showwarning',
                        lambda title, msg, **k: shown.append(msg))
    try:
        for name, src in (('old', str(fcs)), ('lost', '')):
            s = SimpleNamespace(
                name=name, path=str(tmp_path / f'{name}.csv'),
                source_fcs=src, data=STORED.copy(),
                data_transforms=dict(UNKNOWN), qc_applied=False,
                fluor_channels=['FL1-A', 'FL2-A'],
                channel_labels={c: c for c in COLS})
            ed._samples[name] = s
            ed._sample_order.append(name)
            ed._sample_gates[name] = {'g1': dict(CUT)}
            ed._sample_gate_order[name] = ['g1']
        ed.run_async = lambda work, on_done=None, on_error=None, busy_msg=None: (
            on_done(work()) if on_done else work())
        ed._export_flowjo_wsp()
    finally:
        root.destroy()
    back = {}
    for smp in ET.parse(out).getroot().iter('Sample'):
        nm = smp.find('SampleNode').get('name')
        back[nm] = [float(v) for e in smp.iter()
                    for k, v in e.attrib.items() if k.endswith('}min')]
    want = _n(CUT, STORED)
    assert int((LIN['FL1-A'] >= back['old'][0]).sum()) == want
    assert back['lost'] == [4.0]
    assert len(shown) == 1
    assert 'old / FL1pos' in shown[0] and 'rank for rank' in shown[0]
    assert 'lost / FL1pos' in shown[0] and 'written as stored' in shown[0]
