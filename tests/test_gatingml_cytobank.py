"""Gating-ML 2.0 as Cytobank writes it, and the standard QuadrantGate.

Every file here is generated in the layout Cytobank's exports use (see
gatingml_cytobank's docstring): flat gates with a <cytobank> custom_info
block, populations as GateSets. Generic channel and population names only;
synthetic events. Expected counts come from numpy on the same events, with
each event first put on the plot scale's edge when it lies past it, as
Cytobank counts.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from openflo.gate_eval import gate_to_mask
from openflo.gatingml import gates_to_stored, read_gatingml, write_gatingml
from openflo.pipeline import transform_spec, transform_values
from openflo.scales import OPEN

NS = ('xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
      'xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/'
      'transformations" '
      'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"')
# Cytobank's arcsinh scale, cofactor 150: fasinh with M = 1 / ln 10 and
# T = 150 sinh(1) is asinh(x / 150).
ASINH = ('<transforms:transformation transforms:id="Tr_Arcsinh_150">'
         f'<transforms:fasinh transforms:T="{150 * math.sinh(1.0)!r}" '
         f'transforms:M="{1 / math.log(10.0)!r}" transforms:A="0.0"/>'
         '</transforms:transformation>')
LIN_SCALE = {'flag': 1, 'argument': '1', 'min': 1.0, 'max': 262144.0}
ASINH_SCALE = {'flag': 4, 'argument': '150', 'min': -200.0, 'max': 262144.0}


def _a(x):
    """Cytobank's arcsinh-150 coordinate of linear value `x`."""
    return math.asinh(x / 150.0)


def _info(name, scale=None, filename=''):
    d = json.dumps({'scale': scale or LIN_SCALE})
    return (f'<data-type:custom_info><cytobank><name>{name}</name>'
            f'<gate_id>1</gate_id><fcs_file_filename>{filename}'
            f'</fcs_file_filename><definition>{d}</definition></cytobank>'
            '</data-type:custom_info>')


def _dim(ch, lo=None, hi=None, asinh=False):
    a = ' gating:compensation-ref="FCS"'
    a += ' gating:transformation-ref="Tr_Arcsinh_150"' if asinh else ''
    a += f' gating:min="{lo!r}"' if lo is not None else ''
    a += f' gating:max="{hi!r}"' if hi is not None else ''
    return (f'<gating:dimension{a}><data-type:fcs-dimension '
            f'data-type:name="{ch}"/></gating:dimension>')


def _range(gid, name, ch, lo, hi, scale=None, asinh=False, filename=''):
    return (f'<gating:RectangleGate gating:id="{gid}">'
            f'{_info(name, scale, filename)}{_dim(ch, lo, hi, asinh)}'
            '</gating:RectangleGate>')


def _rect(gid, name, x, y, scale=None, asinh=False):
    return (f'<gating:RectangleGate gating:id="{gid}">{_info(name, scale)}'
            f'{_dim(x[0], x[1], x[2], asinh)}{_dim(y[0], y[1], y[2], asinh)}'
            '</gating:RectangleGate>')


def _poly(gid, name, xc, yc, verts, scale=None, asinh=False):
    v = ''.join('<gating:vertex>' + ''.join(
        f'<gating:coordinate data-type:value="{c!r}"/>' for c in p)
        + '</gating:vertex>' for p in verts)
    return (f'<gating:PolygonGate gating:id="{gid}">{_info(name, scale)}'
            f'{_dim(xc, asinh=asinh)}{_dim(yc, asinh=asinh)}{v}'
            '</gating:PolygonGate>')


def _set(sid, name, refs, complement=()):
    r = ''.join(f'<gating:gateReference gating:ref="{x}"'
                + (' gating:use-as-complement="true"' if x in complement else '')
                + '/>' for x in refs)
    return (f'<gating:BooleanGate gating:id="{sid}"><data-type:custom_info>'
            f'<cytobank><name>{name}</name><gate_set_id>1</gate_set_id>'
            '</cytobank></data-type:custom_info>'
            f'<gating:and>{r}</gating:and></gating:BooleanGate>')


def _quad(gid, xc, yc, xv, yv, quads, *, parent=None, asinh=False):
    """A QuadrantGate; `quads` is [(quadrant id, x location, y location)]."""
    t = ' gating:transformation-ref="Tr_Arcsinh_150"' if asinh else ''

    def div(k, ch, vals):
        v = ''.join(f'<gating:value>{x!r}</gating:value>' for x in vals)
        return (f'<gating:divider gating:id="{gid}_d{k}" '
                f'gating:compensation-ref="FCS"{t}><data-type:fcs-dimension '
                f'data-type:name="{ch}"/>{v}</gating:divider>')
    q = ''.join(
        f'<gating:Quadrant gating:id="{qid}">'
        f'<gating:position gating:divider_ref="{gid}_d1" gating:location="{lx!r}"/>'
        + (f'<gating:position gating:divider_ref="{gid}_d2" '
           f'gating:location="{ly!r}"/>' if yc else '')
        + '</gating:Quadrant>' for qid, lx, ly in quads)
    p = f' gating:parent_id="{parent}"' if parent else ''
    return (f'<gating:QuadrantGate gating:id="{gid}"{p}>{div(1, xc, xv)}'
            + (div(2, yc, yv) if yc else '') + f'{q}</gating:QuadrantGate>')


def _file(tmp_path, body, name='cb.xml'):
    p = tmp_path / name
    p.write_text(f'<?xml version="1.0"?>\n<gating:Gating-ML {NS}>{ASINH}'
                 f'{body}</gating:Gating-ML>', encoding='utf-8')
    return str(p)


def _events(n=20_000, seed=11):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'FSC-A': rng.uniform(0, 3e5, n), 'SSC-A': rng.uniform(0, 3e5, n),
        'FL1-A': np.where(rng.random(n) < .5, rng.normal(-300, 200, n),
                          rng.lognormal(np.log(5000), 1.2, n)),
        'FL2-A': np.where(rng.random(n) < .5, rng.normal(0, 400, n),
                          rng.lognormal(np.log(3000), 1.2, n)),
    })


def _cum(gates, df):
    by_id = {g['_import_id']: dict(g, id=g['_import_id']) for g in gates}

    def cum(g):
        m = np.asarray(gate_to_mask(g, df, by_id), bool)
        p = by_id.get(g.get('parent_id'))
        return m & cum(p) if p is not None else m
    return {g['label']: int(cum(by_id[g['_import_id']]).sum()) for g in gates}


# ── The tree from GateSets ───────────────────────────────────────────────────

def _cells_tree(extra=''):
    return (_range('G_c', 'cells gate', 'FSC-A', 2e4, 2.5e5)
            + _range('G_p', 'pos gate', 'FL1-A', _a(1000), _a(1e5),
                     ASINH_SCALE, asinh=True)
            + _range('G_q', 'q gate', 'FL2-A', _a(500), _a(5e4),
                     ASINH_SCALE, asinh=True)
            # Refs in any order; a root names its gate twice.
            + _set('GS_1', 'Cells', ['G_c', 'G_c'])
            + _set('GS_3', 'Both', ['G_q', 'G_c', 'G_p'])
            + _set('GS_2', 'Pos', ['G_p', 'G_c'])
            + _set('GS_4', 'Q under cells', ['G_c', 'G_q'])
            + extra)


def test_gatesets_become_the_population_tree(tmp_path):
    gates, warns = read_gatingml(_file(tmp_path, _cells_tree()))
    assert warns == []
    by = {g['label']: g for g in gates}
    assert set(by) == {'Cells', 'Pos', 'Both', 'Q under cells'}
    ids = {k: g['_import_id'] for k, g in by.items()}
    assert by['Cells']['parent_id'] is None
    assert by['Pos']['parent_id'] == ids['Cells']
    assert by['Both']['parent_id'] == ids['Pos']
    # One gate under two parents is two populations.
    assert by['Q under cells']['parent_id'] == ids['Cells']
    assert by['Both']['channel'] == by['Q under cells']['channel'] == 'FL2-A'
    assert by['Pos']['lo'] == pytest.approx(1000)
    # Counts: AND of every gate on the path, as Cytobank's GateSet says.
    df = _events()
    c = _cum(gates, df)
    fsc = (df['FSC-A'] >= 2e4) & (df['FSC-A'] <= 2.5e5)
    pos = (df['FL1-A'] >= 1000) & (df['FL1-A'] <= 1e5)
    q = (df['FL2-A'] >= 500) & (df['FL2-A'] <= 5e4)
    assert c == {'Cells': int(fsc.sum()), 'Pos': int((fsc & pos).sum()),
                 'Both': int((fsc & pos & q).sum()),
                 'Q under cells': int((fsc & q).sum())}
    assert min(c.values()) > 100


def test_a_gateset_adding_two_gates_becomes_a_chain(tmp_path):
    gates, _w = read_gatingml(_file(tmp_path, (
        _range('G_c', 'cells gate', 'FSC-A', 2e4, 2.5e5)
        + _range('G_p', 'pos gate', 'FL1-A', _a(1000), _a(1e5), asinh=True)
        + _range('G_q', 'q gate', 'FL2-A', _a(500), _a(5e4), asinh=True)
        + _set('GS_1', 'Cells', ['G_c', 'G_c'])
        + _set('GS_2', 'Deep', ['G_c', 'G_p', 'G_q']))))
    labels = [(g['label'], g['parent_id']) for g in gates]
    ids = [g['_import_id'] for g in gates]
    assert labels == [('Cells', None), ('pos gate', ids[0]), ('Deep', ids[1])]


def test_what_cannot_be_placed_is_named(tmp_path):
    gates, warns = read_gatingml(_file(tmp_path, _cells_tree(
        _range('G_t', 'cells gate', 'FSC-A', 3e4, 2.4e5, filename='one.fcs')
        + _range('G_u', 'unused', 'SSC-A', 1e3, 2e5)
        + _set('GS_5', 'Neg', ['G_c', 'G_p'], complement=['G_p'])
        + _set('GS_6', 'Ghost', ['G_c', 'G_gone'])
        # A Boolean gate with no gate_set_id is not a population.
        + _set('GB', 'Plain bool', ['G_c', 'G_u']).replace(
            '<gate_set_id>1</gate_set_id>', ''))))
    assert not {'Neg', 'Plain bool'} & {g['label'] for g in gates}
    text = ' | '.join(warns)
    assert "'Neg' left out: a gate is used as its complement" in text
    assert "'Ghost' left out: gate 'G_gone' is not in the file" in text
    assert '1 file-specific (tailored) gate(s) not imported' in text
    assert '1 gate(s) no population uses were not imported' in text
    assert ('1 Boolean gate(s) that are not Cytobank populations were not '
            'imported') in text
    # The tailored copy did not replace the experiment-wide gate.
    assert next(g for g in gates if g['label'] == 'Cells')['lo'] == 2e4


def test_two_gatesets_with_the_same_gates_are_two_populations(tmp_path):
    gates, warns = read_gatingml(_file(tmp_path, _cells_tree(
        _set('GS_9', 'Pos again', ['G_c', 'G_p']))))
    assert warns == []
    by = {g['label']: g for g in gates}
    assert by['Pos again']['parent_id'] == by['Cells']['_import_id']
    assert by['Pos again']['lo'] == by['Pos']['lo']


def test_cytobank_names_without_gatesets(tmp_path):
    gates, _w = read_gatingml(_file(tmp_path, _range('G_c', 'cells gate',
                                                     'FSC-A', 2e4, 2.5e5)))
    assert [g['label'] for g in gates] == ['cells gate']


def test_a_file_with_parent_ids_keeps_them(tmp_path):
    """A file whose gates already nest (OpenFlo's, FlowJo's) is read as it
    is, even with a BooleanGate that looks like a GateSet."""
    body = (_range('G_c', 'cells', 'FSC-A', 2e4, 2.5e5)
            + _range('G_p', 'pos', 'FL1-A', 1000, 1e5).replace(
                'gating:id="G_p"', 'gating:id="G_p" gating:parent_id="G_c"')
            + _set('GS_1', 'Set', ['G_c', 'G_p']))
    gates, _w = read_gatingml(_file(tmp_path, body))
    by = {g['label']: g for g in gates}
    assert set(by) == {'cells', 'pos', 'Set'}
    assert by['Set']['kind'] == 'boolean'
    assert by['pos']['parent_id'] == by['cells']['_import_id']


# ── Plot edges ───────────────────────────────────────────────────────────────

def test_a_side_on_the_plot_edge_takes_in_the_events_past_it(tmp_path):
    gates, _w = read_gatingml(_file(tmp_path, (
        _range('G_a', 'to top', 'FSC-A', 5e4, 262144.0)
        + _range('G_b', 'from bottom', 'FL1-A', _a(-200), _a(2000),
                 ASINH_SCALE, asinh=True)
        + _range('G_c', 'both', 'FL2-A', _a(-250), _a(3e5), ASINH_SCALE,
                 asinh=True)
        + _range('G_d', 'inside', 'SSC-A', 2.0, 2.6e5)
        + _range('G_f', 'all', 'SSC-A', 1.0, None)
        + _rect('G_e', 'box', ('FSC-A', 1.0, 1e5), ('SSC-A', 1e4, 262144.0),
                {'x': LIN_SCALE, 'y': LIN_SCALE}))))
    by = {g['label']: g for g in gates}
    assert by['to top']['kind'] == 'threshold' and by['to top']['value'] == 5e4
    assert by['from bottom']['lo'] == -OPEN
    assert by['from bottom']['hi'] == pytest.approx(2000)
    assert (by['both']['kind'], by['both']['lo'], by['both']['hi']) == (
        'interval', -OPEN, OPEN)
    assert by['inside']['lo'] == pytest.approx(2.0) and by['inside']['hi'] < OPEN
    # A lower bound on the bottom edge and no upper one: every event, in
    # the form read_gatingml gives a range with no bounds.
    assert {k: by['all'][k] for k in ('kind', 'lo', 'hi')} == {
        'kind': 'interval', 'lo': -OPEN, 'hi': OPEN}
    assert 'value' not in by['all']
    assert (by['box']['x0'], by['box']['y1']) == (-OPEN, OPEN)
    assert by['box']['x1'] == 1e5 and by['box']['y0'] == 1e4


def test_a_polygon_past_the_plot_edge_takes_in_the_piled_events(tmp_path):
    """A polygon on arcsinh axes reaching below the scale's -200: Cytobank
    puts every event below -200 at -200, so those events inside the
    polygon's cut at -200 count."""
    scale = {'x': ASINH_SCALE, 'y': ASINH_SCALE}
    lo = _a(-200)
    verts = [(_a(2000), lo - 0.5), (_a(4e4), lo - 0.5), (_a(4e4), _a(2e4)),
             (_a(2000), _a(1000))]
    gates, _w = read_gatingml(_file(tmp_path, _poly(
        'G_p', 'poly', 'FL1-A', 'FL2-A', verts, scale, asinh=True)))
    (g,) = gates
    v = np.asarray(g['vertices'])
    assert (v[:, 1] == -OPEN).sum() == 2
    df = _events()
    want_df = df.copy()
    want_df['FL2-A'] = np.maximum(df['FL2-A'], -200.0)
    ref = dict(g, id='p', vertices=None, kind='polygon')
    # The reference: the polygon as drawn, on the events piled at -200,
    # evaluated on the arcsinh axes its edges are straight on.
    a_df = pd.DataFrame({'FL1-A': np.arcsinh(want_df['FL1-A'] / 150),
                         'FL2-A': np.arcsinh(want_df['FL2-A'] / 150)})
    ref['vertices'] = [list(p) for p in verts]
    want = int(np.asarray(gate_to_mask(ref, a_df)).sum())
    got = int(np.asarray(gate_to_mask(dict(g, id='p'), df)).sum())
    piled = int(((df['FL2-A'] < -200) & (df['FL1-A'] > 2500)
                 & (df['FL1-A'] < 3.5e4)).sum())
    assert piled > 100
    assert abs(got - want) <= max(3, 0.002 * want)


def test_a_polygon_on_linear_axes_opens_at_the_plot_top(tmp_path):
    """Linear file axes stored linear (a constant scale factor): a
    polygon with two vertices on the plot's top edge still opens there."""
    scale = {'x': LIN_SCALE, 'y': LIN_SCALE}
    verts = [(5e4, 1e5), (2e5, 1e5), (2.2e5, 262144.0), (3e4, 262144.0)]
    gates, _w = read_gatingml(_file(tmp_path, _poly(
        'G_p', 'top poly', 'FSC-A', 'SSC-A', verts, scale)))
    (g,) = gates
    v = np.asarray(g['vertices'])
    assert (v[:, 1] == OPEN).any() and (v[:, 1] < 262144.0).any()
    df = _events()
    past = (df['SSC-A'] > 262144.0) & (df['FSC-A'] > 6e4) & (df['FSC-A'] < 1.9e5)
    assert past.sum() > 100
    assert np.asarray(gate_to_mask(dict(g, id='p'), df))[past.to_numpy()].all()


def _ellipse(gid, name, xc, yc, mean, cov, dsq, scale, asinh=False):
    m = ''.join(f'<gating:coordinate data-type:value="{c!r}"/>' for c in mean)
    rows = ''.join('<gating:row>' + ''.join(
        f'<gating:entry data-type:value="{e!r}"/>' for e in r) + '</gating:row>'
        for r in cov)
    return (f'<gating:EllipsoidGate gating:id="{gid}">{_info(name, scale)}'
            f'{_dim(xc, asinh=asinh)}{_dim(yc, asinh=asinh)}'
            f'<gating:mean>{m}</gating:mean><gating:covarianceMatrix>{rows}'
            f'</gating:covarianceMatrix><gating:distanceSquare '
            f'data-type:value="{dsq!r}"/></gating:EllipsoidGate>')


@pytest.mark.parametrize('edge', ['low', 'high'])
def test_a_polygon_vertex_drawn_on_the_edge_opens(tmp_path, edge):
    """A vertex drawn exactly on Cytobank's plot edge comes back from the
    arcsinh scale a rounding error inside it; it is still on the edge, so
    the events piled there count."""
    scale = {'x': ASINH_SCALE, 'y': ASINH_SCALE}
    e = _a(-200) if edge == 'low' else _a(262144.0)
    inner = _a(1000) if edge == 'low' else _a(5e4)
    verts = [(_a(2000), e), (_a(4e4), e), (_a(4e4), inner), (_a(2000), inner)]
    (g,), _w = read_gatingml(_file(tmp_path, _poly(
        'G_p', 'poly', 'FL1-A', 'FL2-A', verts, scale, asinh=True)))
    n = 500
    df = pd.DataFrame({'FL1-A': np.linspace(5000, 2e4, n),
                       'FL2-A': np.full(n, -5000.0 if edge == 'low' else 5e5)})
    assert int(np.asarray(gate_to_mask(dict(g, id='p'), df)).sum()) == n


def test_an_ellipse_past_the_plot_edge_takes_in_the_piled_events(tmp_path):
    scale = {'x': ASINH_SCALE, 'y': ASINH_SCALE}
    lo = _a(-200)
    (g,), _w = read_gatingml(_file(tmp_path, _ellipse(
        'G_e', 'blob', 'FL1-A', 'FL2-A', [_a(5000), lo], [[0.5, 0.0], [0.0, 0.5]],
        1.0, scale, asinh=True)))
    assert g['kind'] == 'polygon'
    v = np.asarray(g['vertices'])
    assert (v[:, 1] == -OPEN).sum() >= 2
    df = pd.DataFrame({'FL1-A': np.full(10, 5000.0), 'FL2-A': np.full(10, -5000.0)})
    assert int(np.asarray(gate_to_mask(dict(g, id='e'), df)).sum()) == 10


def test_the_edges_are_not_left_on_the_stored_gate(tmp_path):
    scale = {'x': ASINH_SCALE, 'y': ASINH_SCALE}
    verts = [(_a(2000), _a(1000)), (_a(4e4), _a(1000)), (_a(4e4), _a(2e4))]
    tf = {'FL1-A': transform_spec('logicle'), 'FL2-A': transform_spec('logicle')}
    (g,), _w = read_gatingml(_file(tmp_path, _poly(
        'G_p', 'poly', 'FL1-A', 'FL2-A', verts, scale, asinh=True)), tf)
    assert not [k for k in g if k.startswith('_scale')]


@pytest.mark.parametrize('defn', ['[' * 200_000, '[1, 2]', '"text"'])
def test_a_definition_that_is_not_a_scale_is_ignored(tmp_path, defn):
    """Deeply nested or non-object JSON in <definition> is no plot scale:
    the gate reads as written, and the file still reads."""
    body = _range('G_r', 'r', 'FSC-A', 1.0, 2e5).replace(
        json.dumps({'scale': LIN_SCALE}), defn)
    assert defn in body
    (g,), _w = read_gatingml(_file(tmp_path, body))
    assert (g['lo'], g['hi']) == (1.0, 2e5)


def test_a_scale_upside_down_opens_nothing(tmp_path):
    bad = {'flag': 1, 'argument': '1', 'min': 262144.0, 'max': 1.0}
    (g,), _w = read_gatingml(_file(tmp_path, _range(
        'G_r', 'r', 'FSC-A', 1.0, 262144.0, bad)))
    assert (g['kind'], g['lo'], g['hi']) == ('interval', 1.0, 262144.0)


def test_edges_hold_on_a_stored_scale_too(tmp_path):
    """Read raw, then put on logicle stored scales: the open side and the
    opened polygon count the same events as on linear values."""
    scale = {'x': ASINH_SCALE, 'y': ASINH_SCALE}
    lo = _a(-200)
    verts = [(_a(2000), lo - 0.5), (_a(4e4), lo - 0.5), (_a(4e4), _a(2e4)),
             (_a(2000), _a(1000))]
    path = _file(tmp_path, _poly('G_p', 'poly', 'FL1-A', 'FL2-A', verts,
                                 scale, asinh=True)
                 + _range('G_b', 'from bottom', 'FL1-A', _a(-200), _a(2000),
                          ASINH_SCALE, asinh=True))
    tf = {'FL1-A': transform_spec('logicle'), 'FL2-A': transform_spec('logicle')}
    raw, _w = read_gatingml(path, raw=True)
    lin = gates_to_stored(raw, None)
    stored = gates_to_stored(raw, tf)
    df = _events()
    sdf = df.copy()
    for ch, spec in tf.items():
        sdf[ch] = transform_values(df[ch].to_numpy(), **spec)
    for a, b in zip(lin, stored, strict=True):
        n_lin = int(np.asarray(gate_to_mask(dict(a, id='x'), df)).sum())
        n_st = int(np.asarray(gate_to_mask(dict(b, id='x'), sdf)).sum())
        assert abs(n_lin - n_st) <= max(3, 0.003 * n_lin), a['label']


# ── QuadrantGate (standard Gating-ML 2.0) ────────────────────────────────────

def test_a_quadrant_gate_reads_as_four_linked_rects(tmp_path):
    xv, yv = _a(1000), _a(800)
    body = (_range('G_c', 'cells', 'FSC-A', 2e4, 2.5e5)
            + _quad('QG', 'FL1-A', 'FL2-A', [xv], [yv],
                    [('UR', xv + 1, yv + 1), ('UL', xv - 1, yv + 1),
                     ('LL', xv - 1, yv - 1), ('LR', xv + 1, yv - 1)],
                    parent='G_c', asinh=True)
            + _range('G_k', 'kid', 'SSC-A', 1e3, 2e5).replace(
                'gating:id="G_k"', 'gating:id="G_k" gating:parent_id="UR"'))
    gates, warns = read_gatingml(_file(tmp_path, body))
    assert warns == []
    by = {g['label']: g for g in gates}
    cells = by['cells']['_import_id']
    ur, ll = by['UR'], by['LL']
    assert ur['kind'] == 'rect' and ur['parent_id'] == cells
    assert ur['x0'] == pytest.approx(1000) and ur['x1'] == OPEN
    assert ur['y0'] == pytest.approx(800) and ur['y1'] == OPEN
    assert (ll['x0'], ll['y0']) == (-OPEN, -OPEN)
    assert ll['x1'] == pytest.approx(1000) and ll['y1'] == pytest.approx(800)
    assert len({by[k]['quad_set'] for k in ('UR', 'UL', 'LL', 'LR')}) == 1
    assert ur['quad_origin_x'] == pytest.approx(1000)
    assert ur['quad_origin_y'] == pytest.approx(800)
    assert by['kid']['parent_id'] == ur['_import_id']
    df = _events()
    c = _cum(gates, df)
    assert sum(c[k] for k in ('UR', 'UL', 'LL', 'LR')) == c['cells']


def test_the_quadrant_centre_goes_on_the_stored_scale(tmp_path):
    xv, yv = _a(1000), _a(800)
    path = _file(tmp_path, _quad('QG', 'FL1-A', 'FL2-A', [xv], [yv],
                                 [('UR', xv + 1, yv + 1)], asinh=True))
    tf = {'FL1-A': transform_spec('logicle'), 'FL2-A': transform_spec('logicle')}
    (g,), _w = read_gatingml(path, tf)
    want = transform_values(np.array([1000.0]), **tf['FL1-A'])[0]
    assert g['quad_origin_x'] == pytest.approx(want) == pytest.approx(g['x0'])


def test_sectors_and_one_divider(tmp_path):
    """A divider with several values gives the sector a position falls in
    (no quad_set); one divider gives intervals; a quadrant gate placed
    under two populations is two sets."""
    body = (_quad('QA', 'FL1-A', 'FL2-A', [1.0, 3.0], [2.0],
                  [('mid', 2.0, 5.0), ('low', 0.5, 1.0)])
            + _quad('QB', 'FSC-A', None, [5e4], None, [('hi', 6e4, None),
                                                       ('lo', 1e4, None)])
            + _range('G_c', 'c1', 'FSC-A', 1e3, 2e5)
            + _range('G_d', 'c2', 'SSC-A', 1e3, 2e5)
            + _set('S1', 'c1', ['G_c', 'G_c']) + _set('S2', 'c2', ['G_d', 'G_d'])
            + _set('S3', 'Q@c1', ['G_c', 'mid']) + _set('S4', 'Q@c2', ['G_d', 'mid']))
    gates, _warns = read_gatingml(_file(tmp_path, body))
    by = {g['label']: g for g in gates}
    # This file has GateSets: only the populations they make are read.
    assert set(by) == {'c1', 'c2', 'Q@c1', 'Q@c2'}
    m = by['Q@c1']
    assert (m['x0'], m['x1'], m['y0'], m['y1']) == (1.0, 3.0, 2.0, OPEN)
    # Divider values in any order: the sector is between its neighbours.
    gates_d, _w = read_gatingml(_file(tmp_path, _quad(
        'QD', 'FL1-A', 'FL2-A', [3.0, 1.0], [2.0], [('mid', 2.0, 5.0)]),
        name='desc.xml'))
    (d,) = gates_d
    assert (d['x0'], d['x1']) == (1.0, 3.0)
    assert 'quad_set' not in m
    gates2, _w = read_gatingml(_file(tmp_path, _quad(
        'QB', 'FSC-A', None, [5e4], None, [('hi', 6e4, None), ('lo', 1e4, None)]),
        name='one.xml'))
    b2 = {g['label']: g for g in gates2}
    assert (b2['hi']['kind'], b2['hi']['lo'], b2['hi']['hi']) == ('interval', 5e4, OPEN)
    assert (b2['lo']['lo'], b2['lo']['hi']) == (-OPEN, 5e4)
    xv, yv = 1.0, 2.0
    q4 = [('UR', xv + 1, yv + 1), ('UL', xv - 1, yv + 1)]
    gates3, _w = read_gatingml(_file(tmp_path, (
        _range('G_c', 'c1', 'FSC-A', 1e3, 2e5) + _range('G_d', 'c2', 'SSC-A', 1e3, 2e5)
        + _quad('QC', 'FL1-A', 'FL2-A', [xv], [yv], q4)
        + _set('S1', 'c1', ['G_c', 'G_c']) + _set('S2', 'c2', ['G_d', 'G_d'])
        + _set('S3', 'UR1', ['G_c', 'UR']) + _set('S4', 'UL1', ['G_c', 'UL'])
        + _set('S5', 'UR2', ['G_d', 'UR'])), name='two.xml'))
    b3 = {g['label']: g for g in gates3}
    assert b3['UR1']['quad_set'] == b3['UL1']['quad_set'] != b3['UR2']['quad_set']


def test_a_malformed_quadrant_is_left_out(tmp_path):
    body = (_quad('QA', 'FL1-A', 'FL2-A', [1.0], [2.0], [('ok', 2.0, 3.0)])
            .replace('</gating:QuadrantGate>',
                     '<gating:Quadrant gating:id="bad"><gating:position '
                     'gating:divider_ref="nope" gating:location="1"/>'
                     '</gating:Quadrant><gating:Quadrant gating:id="half">'
                     '<gating:position gating:divider_ref="QA_d1" '
                     'gating:location="2"/></gating:Quadrant>'
                     '</gating:QuadrantGate>'))
    gates, warns = read_gatingml(_file(tmp_path, body))
    assert [g['label'] for g in gates] == ['ok']
    assert any("'bad' left out: a quadrant position on no divider" in w
               for w in warns)
    assert any("'half' left out: a quadrant not placed on every divider" in w
               for w in warns)
    three = _quad('Q3', 'FL1-A', 'FL2-A', [1.0], [2.0], [('q3', 2.0, 3.0)]).replace(
        '<gating:Quadrant ', '<gating:divider gating:id="Q3_d3" gating:'
        'compensation-ref="FCS"><data-type:fcs-dimension data-type:name="FSC-A"/>'
        '<gating:value>5.0</gating:value></gating:divider><gating:Quadrant ', 1)
    gates, warns = read_gatingml(_file(tmp_path, three, name='three.xml'))
    assert gates == []
    assert any("'q3' left out: a quadrant gate with 3 dividers" in w for w in warns)


def test_a_position_on_a_divider_value_is_in_the_sector_above(tmp_path):
    """Gating-ML 2.0 closes each sector at its lower divider value."""
    gates, _w = read_gatingml(_file(tmp_path, _quad(
        'QA', 'FL1-A', 'FL2-A', [1.0, 3.0], [2.0], [('on', 1.0, 2.0)])))
    (g,) = gates
    assert (g['x0'], g['x1'], g['y0'], g['y1']) == (1.0, 3.0, 2.0, OPEN)


def test_openflos_own_quadrants_still_round_trip(tmp_path):
    """OpenFlo writes a quadrant as four rects; they read back as four
    rects (not mistaken for a Cytobank file)."""
    rects = [{'id': f'q{k}', 'parent_id': None, 'kind': 'rect', 'name': f'Q{k}',
              'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
              'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
             for k, (x0, x1, y0, y1) in enumerate(
                 [(1e3, 1e12, 800, 1e12), (-1e12, 1e3, 800, 1e12)])]
    p = tmp_path / 'own.xml'
    assert write_gatingml(str(p), rects) == []
    gates, warns = read_gatingml(str(p))
    assert warns == [] and [g['label'] for g in gates] == ['Q0', 'Q1']
