"""A FlowJo polygon's edges are straight on the plot's axes, not in data units.

A FlowJo workspace stores polygon vertices in data units, but FlowJo draws
and evaluates the edges straight on the transformed axes each <Sample>
declares in <Transformations> (logicle here). Evaluated with straight edges
in data units, the same vertices enclose another region.

Synthetic data: 20,000 events with FL1-A and FL2-A spread log-uniformly over
10 .. 200,000, and a triangle whose long edge runs from (150000, 300) to
(300, 150000). The expected counts are computed here independently of
OpenFlo: logicle by flowutils' own function, point-in-polygon by
matplotlib's Path. On the logicle axes the long edge bows towards the origin
in data units, so the two answers differ by thousands of events.
"""
from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path

from openflo import pipeline as fp
from openflo.compare import _compare_one_sample, _per_sample_inventory

N = 20000
LOGICLE = {'T': 262144.0, 'W': 0.5, 'M': 4.5, 'A': 0.0}
TRIANGLE = [[50.0, 50.0], [150000.0, 300.0], [300.0, 150000.0]]


def _events(seed=0):
    rng = np.random.default_rng(seed)
    return 10.0 ** rng.uniform(1.0, np.log10(2e5), size=(N, 2))


def _logicle(v):
    from flowutils import transforms
    return transforms.logicle(np.asarray(v, float).reshape(-1, 1), [0],
                              t=LOGICLE['T'], w=LOGICLE['W'],
                              m=LOGICLE['M'], a=LOGICLE['A']).ravel()


def _inside(verts, pts):
    return Path(np.asarray(verts, float)).contains_points(np.asarray(pts, float))


def _expected(ev):
    """(count with edges straight on the logicle axes, count with edges
    straight in data units), computed without OpenFlo."""
    tv = np.column_stack([_logicle(np.array(TRIANGLE)[:, 0]),
                          _logicle(np.array(TRIANGLE)[:, 1])])
    tp = np.column_stack([_logicle(ev[:, 0]), _logicle(ev[:, 1])])
    return int(_inside(tv, tp).sum()), int(_inside(TRIANGLE, ev).sum())


@pytest.fixture
def fcs(tmp_path):
    import flowio
    ev = _events()
    path = tmp_path / 'tri.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).ravel().tolist(),
                          ['FL1-A', 'FL2-A'])
    # The file holds float32: count what it holds.
    return str(path), ev.astype(np.float32).astype(float)


def _axis(kind, param):
    if kind == 'logicle':
        attrs = ' '.join(f'transforms:{k}="{v}"' for k, v in LOGICLE.items())
        return (f'<transforms:logicle transforms:length="256" {attrs}>'
                f'<data-type:parameter data-type:name="{param}"/>'
                '</transforms:logicle>')
    return ('<transforms:linear transforms:minRange="0" '
            'transforms:maxRange="262144" gain="1">'
            f'<data-type:parameter data-type:name="{param}"/></transforms:linear>')


def _wsp(path, fcs_path, axes=('logicle', 'logicle')):
    """A FlowJo 10 style workspace: one triangle on FL1-A x FL2-A, with the
    two axes `axes` declared in <Transformations> (None: not declared)."""
    decl = ''.join(_axis(k, p) for k, p in zip(axes, ('FL1-A', 'FL2-A'),
                                                strict=True) if k)
    verts = ''.join(
        f'<gating:vertex><gating:coordinate data-type:value="{x}"/>'
        f'<gating:coordinate data-type:value="{y}"/></gating:vertex>'
        for x, y in TRIANGLE)
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Workspace version="20.0" flowJoVersion="10.8.1"
  xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"
  xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
  xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">
 <SampleList><Sample>
  <DataSet uri="{fp._wsp_dataset_uri(fcs_path)}" sampleID="1"/>
  {'<Transformations>' + decl + '</Transformations>' if decl else ''}
  <SampleNode name="s1" sampleID="1" count="{N}"><Subpopulations>
   <Population name="Tri" count="0">
    <Gate gating:id="ID1"><gating:PolygonGate>
     <gating:dimension><data-type:fcs-dimension data-type:name="FL1-A"/></gating:dimension>
     <gating:dimension><data-type:fcs-dimension data-type:name="FL2-A"/></gating:dimension>
     {verts}
    </gating:PolygonGate></Gate>
   </Population>
  </Subpopulations></SampleNode>
 </Sample></SampleList>
</Workspace>''', encoding='utf-8')
    return str(path)


def _count(gates, df):
    by_id = {}
    for i, g in enumerate(gates):
        by_id[g.get('_import_id') or g.get('id') or i] = dict(g)
    (gid, _g), = [(k, g) for k, g in by_id.items() if g.get('kind') == 'polygon']
    return int(fp.cumulative_gate_mask(by_id, gid, df).sum())


def _frame(ev):
    import pandas as pd
    return pd.DataFrame({'FL1-A': ev[:, 0], 'FL2-A': ev[:, 1]})


def test_the_two_readings_differ_on_this_data(fcs):
    _path, ev = fcs
    on_axes, in_data_units = _expected(ev)
    assert in_data_units - on_axes > 1000        # the test can tell them apart


def test_compare_counts_the_polygon_on_flowjos_axes(fcs, tmp_path):
    path, ev = fcs
    wsp = _wsp(tmp_path / 'tri.wsp', path)
    (name, _uri, pops), = _per_sample_inventory(fp.WspReader(wsp))
    row, = _compare_one_sample(name, path, pops)
    assert row['openflo_count'] == _expected(ev)[0]


def test_reader_gate_counts_on_flowjos_axes(fcs, tmp_path):
    path, ev = fcs
    gates = fp.WspReader(_wsp(tmp_path / 'tri.wsp', path)).extract_gates()
    assert _count(gates, _frame(ev)) == _expected(ev)[0]


def test_one_logicle_axis_and_one_linear(fcs, tmp_path):
    """Only the logicle axis bends: x on logicle, y on a linear axis."""
    path, ev = fcs
    gates = fp.WspReader(_wsp(tmp_path / 'tri.wsp', path,
                              axes=('logicle', 'linear'))).extract_gates()
    tv = np.column_stack([_logicle(np.array(TRIANGLE)[:, 0]),
                          np.array(TRIANGLE)[:, 1]])
    want = int(_inside(tv, np.column_stack([_logicle(ev[:, 0]), ev[:, 1]])).sum())
    assert want not in _expected(ev)
    assert _count(gates, _frame(ev)) == want


@pytest.mark.parametrize('axes', [(None, None), ('linear', 'linear')])
def test_undeclared_or_linear_axes_keep_straight_edges_in_data_units(
        fcs, tmp_path, axes):
    """OpenFlo's own exports declare no axes: their polygons read back as
    written, straight in data units; so does a polygon on linear axes."""
    path, ev = fcs
    gates = fp.WspReader(_wsp(tmp_path / 'tri.wsp', path,
                              axes=axes)).extract_gates()
    assert 'flowjo_edge_axes' not in gates[0]
    assert _count(gates, _frame(ev)) == _expected(ev)[1]


def test_gui_import_onto_the_same_logicle_keeps_the_count(fcs, tmp_path):
    """The editor puts imported gates on its stored scale with
    map_gate_coords; on the same logicle the count is exact."""
    path, ev = fcs
    gates = fp.WspReader(_wsp(tmp_path / 'tri.wsp', path)).extract_gates()
    spec = fp.transform_spec('logicle', t=LOGICLE['T'], m=LOGICLE['M'],
                             w=LOGICLE['W'], a=LOGICLE['A'])
    stored = {'FL1-A': spec, 'FL2-A': spec}
    g = fp.map_gate_coords(gates[0], fp.gate_scale_fns(stored, 'from_linear'))
    assert 'flowjo_edge_axes' not in g
    df = _frame(np.column_stack([_logicle(ev[:, 0]), _logicle(ev[:, 1])]))
    assert _count([g], df) == _expected(ev)[0]


def test_import_onto_another_scale_traces_the_edges(fcs, tmp_path):
    """Onto a scale other than FlowJo's (asinh, or linear data as a template
    run keeps scatter) each edge is traced on FlowJo's axes: the count stays
    within 0.2% of the exact one, far from the data-unit reading."""
    path, ev = fcs
    gates = fp.WspReader(_wsp(tmp_path / 'tri.wsp', path)).extract_gates()
    want, wrong = _expected(ev)
    linear = fp.map_gate_coords(gates[0], {})
    assert 'flowjo_edge_axes' not in linear
    assert abs(_count([linear], _frame(ev)) - want) <= 0.002 * want
    asinh = {'FL1-A': 'asinh', 'FL2-A': 'asinh'}
    g = fp.map_gate_coords(gates[0], fp.gate_scale_fns(asinh, 'from_linear'))
    df = _frame(np.column_stack([np.arcsinh(ev[:, 0] / 150.0),
                                 np.arcsinh(ev[:, 1] / 150.0)]))
    got = _count([g], df)
    assert abs(got - want) <= 0.002 * want
    assert abs(got - wrong) > 1000


def test_export_of_an_imported_polygon_keeps_its_events(fcs, tmp_path):
    """Imported from FlowJo, written by OpenFlo (no <Transformations>) and
    read back: the polygon still holds the events FlowJo's axes gave it."""
    path, ev = fcs
    gates = fp._template_ids(
        fp.WspReader(_wsp(tmp_path / 'tri.wsp', path)).extract_gates())
    w = fp.WspWriter()
    w.add_sample('s1', path, ['FL1-A', 'FL2-A'], gates)
    out = tmp_path / 'again.wsp'
    w.write(str(out))
    back = fp.WspReader(str(out)).extract_gates()
    assert 'flowjo_edge_axes' not in back[0]
    want = _expected(ev)[0]
    assert abs(_count(back, _frame(ev)) - want) <= 0.002 * want
