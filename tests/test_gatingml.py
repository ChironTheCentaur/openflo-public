"""Standalone Gating-ML 2.0 import and export (Complaints item 19).

Every expected count comes from generated events and an independent
evaluation of the same region (flowutils' logicle, numpy), never from the
code under test.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from flowutils import transforms as fut

from openflo.gatingml import (
    GatingMLError,
    gates_to_stored,
    read_gatingml,
    write_gatingml,
)
from openflo.pipeline import (
    gate_to_mask,
    read_template_gates,
    transform_spec,
    transform_values,
)

NS = ('xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
      'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/'
      'transformations" '
      'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"')

TF = {'FL1-A': transform_spec('logicle'),
      'FL2-A': transform_spec('asinh', cofactor=150.0),
      'FL3-A': transform_spec('log')}


def _events(n=20_000, seed=3):
    """Linear events: scatter, three fluorescence channels with a dim and
    a bright population each."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    k = rng.random(n) < 0.4
    lin = pd.DataFrame({
        'FSC-A': rng.uniform(1e4, 2.5e5, n),
        'SSC-A': rng.uniform(5e3, 2e5, n),
        'FL1-A': np.where(k, rng.lognormal(np.log(15000), .6, n),
                          rng.normal(50, 300, n)),
        'FL2-A': np.where(rng.random(n) < .5, rng.lognormal(np.log(4000), .7, n),
                          rng.normal(20, 200, n)),
        'FL3-A': rng.lognormal(np.log(800), 1.0, n),
    })
    return lin


def _stored(lin, tf):
    out = lin.copy()
    for ch, spec in tf.items():
        out[ch] = transform_values(lin[ch].to_numpy(), **spec)
    return out


def _tree():
    return [
        {'id': 'g1', 'parent_id': None, 'kind': 'rect', 'name': 'Cells',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 3e4, 'x1': 1e12, 'y0': -1e12, 'y1': 1.5e5},
        {'id': 'g2', 'parent_id': 'g1', 'kind': 'polygon', 'name': 'FL1+ FL2+',
         'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[0.55, 2.0], [0.95, 2.4], [0.9, 4.8], [0.5, 4.2]]},
        {'id': 'g3', 'parent_id': 'g1', 'kind': 'threshold', 'name': 'FL3 hi',
         'channel': 'FL3-A', 'value': 3.0},
        {'id': 'g4', 'parent_id': 'g1', 'kind': 'interval', 'name': 'FL1 mid',
         'channel': 'FL1-A', 'lo': 0.3, 'hi': 0.7},
        {'id': 'g5', 'parent_id': 'g1', 'kind': 'ellipsoid', 'name': 'Blob',
         'x_channel': 'FL1-A', 'y_channel': 'FL2-A', 'mean': [0.75, 3.5],
         'cov': [[0.004, 0.001], [0.001, 0.3]], 'distance_sq': 4.0},
        {'id': 'g6', 'parent_id': 'g1', 'kind': 'boolean', 'name': 'Both',
         'op': 'and', 'operands': ['g2', 'g3']},
    ]


def _counts(gates, df):
    by_id = {g['id']: g for g in gates}

    def cum(g):
        m = gate_to_mask(g, df, by_id)
        p = by_id.get(g.get('parent_id'))
        return m & cum(p) if p is not None else m
    return {g.get('name') or g.get('label'): int(cum(g).sum()) for g in gates}


def _with_ids(gates):
    """Imported gates with their import ids as ids, as the editor installs
    them."""
    out = []
    for g in gates:
        d = dict(g)
        d['id'] = d.pop('_import_id')
        out.append(d)
    return out


def test_round_trip_on_the_same_scales_keeps_every_gate_and_count(tmp_path):
    """Export then import onto the same stored scales: every coordinate
    comes back (polygon vertices exactly, no edge subdivision) and every
    population counts what it did."""
    path = tmp_path / 'tree.xml'
    assert write_gatingml(str(path), _tree(), TF) == []
    got, warns = read_gatingml(str(path), TF)
    assert warns == []
    by = {g['label']: g for g in got}
    poly = by['FL1+ FL2+']
    assert poly['kind'] == 'polygon'
    np.testing.assert_allclose(poly['vertices'], _tree()[1]['vertices'],
                               rtol=1e-12)
    assert by['Blob']['kind'] == 'ellipsoid'
    np.testing.assert_allclose(by['Blob']['mean'], [0.75, 3.5], rtol=1e-12)
    assert by['Cells']['x1'] == 1e12 and by['Cells']['y0'] == -1e12
    df = _stored(_events(), TF)
    want = _counts(_tree(), df)
    assert min(want.values()) > 100            # every population is populated
    assert _counts(_with_ids(got), df) == want


def test_coordinates_are_written_on_each_channels_own_scale(tmp_path):
    """Logicle and hyperlog are written as they are; asinh(x/c) as fasinh
    and log10 as flog. Reading a value back through the Gating-ML formula
    gives the same intensity OpenFlo's transform does."""
    path = tmp_path / 'tree.xml'
    write_gatingml(str(path), _tree(), TF)
    lin, _ = read_gatingml(str(path))          # linear data units
    by = {g['label']: g for g in lin}
    # FL3 threshold at log10 = 3 is an intensity of 1000.
    assert by['FL3 hi']['value'] == pytest.approx(1000.0, rel=1e-12)
    # FL1 interval: logicle 0.3 / 0.7 back to intensities.
    want = fut.logicle_inverse(np.array([[0.3], [0.7]]), [0],
                               t=262144, m=4.5, w=0.5, a=0).ravel()
    assert [by['FL1 mid']['lo'], by['FL1 mid']['hi']] == pytest.approx(
        want.tolist(), rel=1e-9)
    # FL2 polygon vertex: asinh 2.0 with cofactor 150.
    assert by['FL1+ FL2+']['vertices'][0][1] == pytest.approx(
        150.0 * math.sinh(2.0), rel=1e-9)
    text = path.read_text(encoding='utf-8')
    assert 'transforms:fasinh' in text and 'transforms:flog' in text
    assert 'transforms:logicle' in text


def _gml(body, transforms=''):
    return (f'<?xml version="1.0"?>\n<gating:Gating-ML {NS}>{transforms}'
            f'{body}</gating:Gating-ML>')


LOGICLE = ('<transforms:transformation transforms:id="L">'
           '<transforms:logicle transforms:T="262144" transforms:W="0.5" '
           'transforms:M="4.5" transforms:A="0"/></transforms:transformation>')


def _fd(ch, ref='', comp='uncompensated', lo=None, hi=None):
    a = f' gating:transformation-ref="{ref}"' if ref else ''
    a += f' gating:compensation-ref="{comp}"'
    a += f' gating:min="{lo}"' if lo is not None else ''
    a += f' gating:max="{hi}"' if hi is not None else ''
    return (f'<gating:dimension{a}><data-type:fcs-dimension '
            f'data-type:name="{ch}"/></gating:dimension>')


def test_a_polygon_from_another_tool_keeps_straight_edges_on_its_own_axes(
        tmp_path):
    """A triangle on logicle FL1 x logicle FL2, as another tool writes it.
    Gating-ML says its edges are straight on those axes, so the events
    inside are those inside the triangle in logicle coordinates (computed
    here with flowutils directly), on linear data and on data stored on
    another scale alike."""
    tri = [(0.2, 0.2), (0.9, 0.3), (0.5, 0.95)]
    verts = ''.join(
        '<gating:vertex>' + ''.join(
            f'<gating:coordinate data-type:value="{v}"/>' for v in p)
        + '</gating:vertex>' for p in tri)
    path = tmp_path / 'tri.xml'
    path.write_text(_gml(
        f'<gating:PolygonGate gating:id="Tri">{_fd("FL1-A", "L")}'
        f'{_fd("FL2-A", "L")}{verts}</gating:PolygonGate>', LOGICLE),
        encoding='utf-8')
    lin = _events(seed=11)
    lx = fut.logicle(lin[['FL1-A']].to_numpy(), [0]).ravel()
    ly = fut.logicle(lin[['FL2-A']].to_numpy(), [0]).ravel()
    from matplotlib.path import Path
    want = int(Path(tri).contains_points(np.column_stack([lx, ly])).sum())
    assert want > 500
    # On linear stored data: the edges are subdivided on the logicle axes.
    (g,), _ = read_gatingml(str(path))
    g = dict(g, id='t')
    got_lin = int(gate_to_mask(g, lin).sum())
    assert abs(got_lin - want) <= want * 0.002
    # The same vertices joined by straight LINEAR edges miss many events.
    plain = dict(g, vertices=[
        [float(fut.logicle_inverse(np.array([[x]]), [0])[0, 0]),
         float(fut.logicle_inverse(np.array([[y]]), [0])[0, 0])]
        for x, y in tri])
    assert abs(int(gate_to_mask(plain, lin).sum()) - want) > want * 0.05
    # On data stored on the same logicle: the vertices are the file's, as is.
    tf = {'FL1-A': transform_spec('logicle'), 'FL2-A': transform_spec('logicle')}
    (g2,), _ = read_gatingml(str(path), tf)
    assert g2['vertices'] == [list(p) for p in tri]
    assert int(gate_to_mask(dict(g2, id='t'), _stored(lin, tf)).sum()) == want


def test_a_gate_openflo_cannot_read_is_left_out_with_its_children(tmp_path):
    """A QuadrantGate and a ratio dimension are left out, with the gates
    under them, and named; a boolean using a left-out gate reads NA (no
    operands), never every event."""
    path = tmp_path / 'odd.xml'
    path.write_text(_gml(
        f'<gating:RectangleGate gating:id="Cells">{_fd("FSC-A", lo=1000)}'
        '</gating:RectangleGate>'
        '<gating:QuadrantGate gating:id="Q" gating:parent_id="Cells"/>'
        f'<gating:RectangleGate gating:id="Kid" gating:parent_id="Q">'
        f'{_fd("FL1-A", lo=10)}</gating:RectangleGate>'
        '<gating:BooleanGate gating:id="NotQ"><gating:not>'
        '<gating:gateReference gating:ref="Q"/></gating:not>'
        '</gating:BooleanGate>'), encoding='utf-8')
    gates, warns = read_gatingml(str(path))
    assert [g['label'] for g in gates] == ['Cells', 'NotQ']
    assert gates[1]['operands'] == []
    text = ' '.join(warns)
    assert "'Q' left out" in text and "'Kid' left out" in text
    assert "'NotQ': an operand was left out" in text


def test_children_listed_before_their_parent_still_nest(tmp_path):
    path = tmp_path / 'order.xml'
    path.write_text(_gml(
        f'<gating:RectangleGate gating:id="Kid" gating:parent_id="Cells">'
        f'{_fd("FL1-A", lo=10)}</gating:RectangleGate>'
        f'<gating:RectangleGate gating:id="Cells">{_fd("FSC-A", lo=1000)}'
        '</gating:RectangleGate>'), encoding='utf-8')
    gates, _ = read_gatingml(str(path))
    assert [g['label'] for g in gates] == ['Cells', 'Kid']
    assert gates[1]['parent_id'] == gates[0]['_import_id']


def test_a_file_with_a_dtd_is_refused(tmp_path):
    path = tmp_path / 'dtd.xml'
    path.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]>'
                    + _gml('')[len('<?xml version="1.0"?>\n'):],
                    encoding='utf-8')
    with pytest.raises(GatingMLError):
        read_gatingml(str(path))


def test_compensation_is_written_and_read(tmp_path):
    """Fluorescence dimensions reference the sample's matrix, scatter is
    uncompensated; on import each gate records which of its channels the
    file compensates (wsp_comp_channels), so a gate on an uncompensated
    fluorescence channel of a compensated sample can be named."""
    from openflo.pipeline import wsp_uncompensated_channels
    path = tmp_path / 'comp.xml'
    m = np.array([[1.0, 0.1], [0.02, 1.0]])
    write_gatingml(str(path), _tree()[:2], TF,
                   compensation=(['FL1-A', 'FL2-A'], m))
    text = path.read_text(encoding='utf-8')
    assert 'transforms:spectrumMatrix' in text
    assert text.count('gating:compensation-ref="Spill"') == 2
    assert text.count('gating:compensation-ref="uncompensated"') == 2
    gates, warns = read_gatingml(str(path), TF)
    assert gates[1]['wsp_comp_channels'] == ['FL1-A', 'FL2-A']
    assert wsp_uncompensated_channels(gates[1], ['FL1-A', 'FL2-A']) == []
    assert any("compensation matrix Spill" in w for w in warns)
    path.write_text(text.replace('compensation-ref="Spill"',
                                 'compensation-ref="uncompensated"'),
                    encoding='utf-8')
    gates, _ = read_gatingml(str(path), TF)
    assert wsp_uncompensated_channels(gates[1], ['FL1-A', 'FL2-A']) == \
        ['FL1-A', 'FL2-A']
    assert wsp_uncompensated_channels(gates[1], []) == []


def test_unknown_scale_gates_are_converted_with_a_match_or_left_out(tmp_path):
    from openflo.pipeline import unknown_spec
    tf = dict(TF, **{'FL1-A': unknown_spec()})
    path = tmp_path / 'u.xml'
    warns = write_gatingml(str(path), _tree()[:4], tf)
    # Without a match every FL1 gate (and nothing else) is left out.
    assert any("'FL1+ FL2+' left out" in w for w in warns)
    assert any("'FL1 mid' left out" in w for w in warns)
    labels = [g['label'] for g in read_gatingml(str(path))[0]]
    assert labels == ['Cells', 'FL3 hi']
    # With a match, written in linear values and named.
    warns = write_gatingml(str(path), _tree()[:4], tf,
                           unknown_fns={'FL1-A': lambda v: np.asarray(v) * 1000})
    assert any("'FL1 mid': FL1-A has an unknown scale" in w for w in warns)
    by = {g['label']: g for g in read_gatingml(str(path))[0]}
    assert [by['FL1 mid']['lo'], by['FL1 mid']['hi']] == pytest.approx([300, 700])


def test_template_reader_and_per_sample_conversion(tmp_path):
    """read_template_gates reads a Gating-ML file with gN ids (as for a
    .wsp), keeping its scales until gates_to_stored puts it on a sample's."""
    path = tmp_path / 'tree.xml'
    write_gatingml(str(path), _tree(), TF)
    gates, labels = read_template_gates(str(path))
    assert labels is None
    assert [g['id'] for g in gates] == [f'g{i}' for i in range(1, 7)]
    assert gates[5]['operands'] == ['g2', 'g3']
    df = _stored(_events(), TF)
    assert _counts(gates_to_stored(gates, TF), df) == _counts(_tree(), df)


def test_editor_applies_and_exports_gatingml(tmp_path, monkeypatch):
    """Through the editor: a Gating-ML file applied as a template lands on
    the sample's stored logicle with the file's vertices, counts what the
    region counts on the linear events, and File > Export gates writes a
    file that reads back to the same gates."""
    import os
    import sys

    import flowio
    sys.path.insert(0, os.path.dirname(__file__))
    from test_gui_smoke import _editor_or_skip

    import openflo.editor_export as ee
    import openflo.pipeline as fp
    lin = _events(n=5000, seed=5)
    fcs = tmp_path / 's.fcs'
    with open(fcs, 'wb') as fh:
        flowio.create_fcs(fh, lin.to_numpy(np.float32).ravel().tolist(),
                          list(lin.columns))
    tf = {'FL1-A': transform_spec('logicle'), 'FL2-A': transform_spec('logicle'),
          'FL3-A': transform_spec('logicle')}
    tree = [dict(_tree()[0]),
            {'id': 'g2', 'parent_id': 'g1', 'kind': 'polygon', 'name': 'Pos',
             'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
             'vertices': [[0.55, 0.5], [0.95, 0.55], [0.9, 0.95], [0.5, 0.9]]},
            dict(_tree()[2])]               # FL3-A >= 3 on a log10 scale
    src = tmp_path / 'in.xml'
    # The file has FL3 on log10, the sample stores it on logicle.
    write_gatingml(str(src), tree, dict(tf, **{'FL3-A': transform_spec('log')}))
    root, ed, _gui = _editor_or_skip()
    try:
        s = fp.FlowSample(str(fcs))
        s.apply_transform()
        name = ed._sample_name_for(str(fcs))
        ed._on_loaded(name, s)
        ed._set_active_sample(name)
        monkeypatch.setattr(ed, '_ask_template_apply', lambda: ([name], True))
        ed._apply_template_path(str(src))
        gates = ed._sample_gates[name]
        poly = next(g for g in gates.values() if g['kind'] == 'polygon')
        np.testing.assert_allclose(poly['vertices'], tree[1]['vertices'],
                                   rtol=1e-12)
        lin32 = lin.astype(np.float32).astype(float)
        want = _counts(tree[:2], _stored(lin32, tf))
        cells = gate_to_mask(tree[0], lin32)
        want['FL3 hi'] = int((cells & (lin32['FL3-A'] >= 1000)).sum())
        got = _counts([dict(g, id=k) for k, g in gates.items()], s.data)
        assert got == want and min(want.values()) > 100
        out = tmp_path / 'out.xml'
        monkeypatch.setattr(ee.filedialog, 'asksaveasfilename',
                            lambda **k: str(out))
        ed._export_gatingml()
        back, _ = read_gatingml(str(out), tf)
        assert [g['label'] for g in back] == ['Cells', 'Pos', 'FL3 hi']
        np.testing.assert_allclose(back[1]['vertices'], tree[1]['vertices'],
                                   rtol=1e-12)
    finally:
        root.destroy()
