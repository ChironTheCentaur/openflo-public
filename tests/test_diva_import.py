"""Importing a BD FACSDiva experiment exported as XML (diva_reader).

Every XML and FCS file here is generated (tests/_diva_fixtures.py): generic
channel, specimen and tube names, synthetic events. Expected values come
from FACSDiva's documented layout (see diva_reader's docstring), computed
here independently of the reader where they can be.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from openflo import diva_reader as dr
from openflo import wsp_open as wo
from openflo.channel_transforms import inverse_transform_values
from openflo.document import Document
from openflo.paths import expand_dropped_paths, is_workspace_file
from openflo.scales import OPEN
from tests._diva_fixtures import (
    COEFFS,
    boolean_gate,
    diva_xml,
    make_events,
    make_fcs,
    region_gate,
)

ROOT = 'All Events'
P1 = 'All Events\\P1'
LIN = (False, False, 0)


def _logicle_lin(frac, w):
    """Linear value at fraction `frac` of a logicle T=262144, M=4.5, A=0."""
    return float(inverse_transform_values(np.array([frac]), 'logicle',
                                          t=262144, m=4.5, w=w, a=0)[0])


def _one_tube(tmp_path, gates, **tube):
    make_fcs(tmp_path / 'a.fcs', seed=1)
    xml = diva_xml(tmp_path / 'exp.xml',
                   [('Spec_A', [dict({'name': 'Tube_1', 'file': 'a.fcs',
                                      'gates': gates}, **tube)])])
    plan = wo.plan_wsp(xml)
    assert len(plan.samples) == 1, plan
    return plan, {g['label']: g for g in plan.samples[0].gates}


# ── Coordinates ──────────────────────────────────────────────────────────────

def test_biexp_width_follows_the_scale_value():
    # W = (4.5 - log10(262144 / r)) / 2, never below 0; r = 0 gives 0.
    r = 262144 / 10 ** 3.5          # log10(T/r) = 3.5 -> W = 0.5
    assert dr.diva_biexp_w(r) == pytest.approx(0.5)
    assert dr.diva_biexp_w(-r) == pytest.approx(0.5)
    assert dr.diva_biexp_w(1.0) == 0.0
    assert dr.diva_biexp_w(0) == 0.0
    # Past the top (r > T) the width stops at M / 2, the logicle limit.
    assert dr.diva_biexp_w(262144 * 10) == pytest.approx(2.25)


def test_each_axis_kind_maps_to_linear_values(tmp_path):
    r = 262144 / 10 ** 3.5
    _plan, g = _one_tube(tmp_path, [
        region_gate('lin', ROOT, 'FSC-A', [(1000, 0), (50000, 0)],
                    rtype='INTERVAL_REGION'),
        region_gate('scaled_lin', ROOT, 'SSC-A', [(1024, 0), (2048, 0)],
                    rtype='INTERVAL_REGION', x=(True, False, 0)),
        region_gate('scaled_log', ROOT, 'FL1-A', [(1024, 0), (3072, 0)],
                    rtype='INTERVAL_REGION', x=(True, True, r)),
        region_gate('log', ROOT, 'FL2-A', [(2.5, 0), (4.0, 0)],
                    rtype='INTERVAL_REGION', x=(False, True, 0)),
    ])
    assert (g['lin']['lo'], g['lin']['hi']) == (1000.0, 50000.0)
    assert (g['scaled_lin']['lo'], g['scaled_lin']['hi']) == (65536.0, 131072.0)
    assert g['scaled_log']['lo'] == pytest.approx(_logicle_lin(0.25, 0.5))
    assert g['scaled_log']['hi'] == pytest.approx(_logicle_lin(0.75, 0.5))
    assert g['log']['lo'] == pytest.approx(10 ** 2.5)
    assert g['log']['hi'] == pytest.approx(1e4)


def test_a_side_on_the_plot_edge_is_open(tmp_path):
    _plan, g = _one_tube(tmp_path, [
        region_gate('scaled_ends', ROOT, 'FL1-A', [(0, 0), (4095, 0)],
                    rtype='INTERVAL_REGION', x=(True, True, 100)),
        # 5 decades: the log axis runs from 262144 / 1e5 = 2.62144.
        region_gate('log_ends', ROOT, 'FL2-A',
                    [(math.log10(2.62144), 0), (math.log10(262143), 0)],
                    rtype='INTERVAL_REGION', x=(False, True, 0)),
        region_gate('lin_top', ROOT, 'FSC-A', [(5000, 0), (262143, 0)],
                    rtype='INTERVAL_REGION'),
        region_gate('inside', ROOT, 'SSC-A', [(1, 0), (4094, 0)],
                    rtype='INTERVAL_REGION', x=(True, False, 0)),
    ])
    assert (g['scaled_ends']['lo'], g['scaled_ends']['hi']) == (-OPEN, OPEN)
    assert (g['log_ends']['lo'], g['log_ends']['hi']) == (-OPEN, OPEN)
    # Only the top is an edge on a linear axis: a gate there is x >= 5000.
    assert g['lin_top'] == {**g['lin_top'], 'kind': 'threshold', 'value': 5000.0}
    assert -OPEN < g['inside']['lo'] and g['inside']['hi'] < OPEN


def test_four_decades_move_the_log_floor(tmp_path):
    make_fcs(tmp_path / 'a.fcs', seed=1)
    gates = [region_gate('log_lo', ROOT, 'FL2-A',
                         [(math.log10(26.2144), 0), (4.0, 0)],
                         rtype='INTERVAL_REGION', x=(False, True, 0))]
    for dec, opened in ((4, True), (5, False)):
        xml = diva_xml(tmp_path / f'e{dec}.xml',
                       [('S', [{'name': 'T', 'file': 'a.fcs', 'gates': gates}])],
                       log_decades=dec)
        g = wo.plan_wsp(xml).samples[0].gates[0]
        assert (g['lo'] == -OPEN) is opened


def test_no_log_decades_means_five(tmp_path):
    make_fcs(tmp_path / 'a.fcs')
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [{'name': 'T', 'file': 'a.fcs'}])],
                   log_decades=None)
    assert dr.experiment_log_floor(dr.parse_diva(xml)) == pytest.approx(2.62144)


def test_a_namespaced_root_is_still_a_facsdiva_experiment(tmp_path):
    p = tmp_path / 'ns.xml'
    p.write_text('<?xml version="1.0"?><bdfacs xmlns="urn:x-made-up"><experiment/>'
                 '</bdfacs>', encoding='utf-8')
    assert dr.is_diva_xml(p) and is_workspace_file(p)


def test_a_box_is_a_rect_and_quadrants_share_their_centre(tmp_path):
    cx, cy = 2000.0, 1500.0
    s = (True, True, 150)
    quads = [region_gate(f'Q{k}', ROOT, 'FL1-A', pts, yparm='FL2-A',
                         x=s, y=s, inputs=[ROOT + '\\$c'])
             for k, pts in enumerate([
                 [(cx, cy), (0, cy), (0, 4095), (cx, 4095)],
                 [(cx, cy), (cx, 4095), (4095, 4095), (4095, cy)],
                 [(cx, cy), (0, cy), (0, 0), (cx, 0)],
                 [(cx, cy), (4095, cy), (4095, 0), (cx, 0)]], 1)]
    centre = region_gate('$c', ROOT, 'FL1-A', [(cx, cy)] * 3, yparm='FL2-A',
                         rtype='BINNER_REGION', gtype='Composite_Classifier',
                         x=s, y=s)
    _plan, g = _one_tube(tmp_path, [centre] + quads)
    assert '$c' not in g
    w = dr.diva_biexp_w(150)
    xl, yl = _logicle_lin(cx / 4096, w), _logicle_lin(cy / 4096, w)
    q1 = g['Q1']
    assert q1['kind'] == 'rect'
    assert (q1['x0'], q1['y1']) == (-OPEN, OPEN)
    assert q1['x1'] == pytest.approx(xl) and q1['y0'] == pytest.approx(yl)
    assert g['Q4']['x1'] == OPEN and g['Q4']['y0'] == -OPEN
    sets = {g[f'Q{k}']['quad_set'] for k in range(1, 5)}
    assert len(sets) == 1 and None not in sets
    assert g['Q2']['quad_origin_x'] == pytest.approx(xl)
    assert g['Q2']['quad_origin_y'] == pytest.approx(yl)
    # Three of the four left: not linked, so none moves the others.
    _plan, g3 = _one_tube(tmp_path, [centre] + quads[:3])
    assert not any('quad_set' in g3[f'Q{k}'] for k in range(1, 4))


def test_a_quadrant_centre_takes_each_axis_own_scale(tmp_path):
    """x on a biexponential plot, y on a linear one: each side of the
    centre is read on its own axis."""
    cx, cy = 2000.0, 90000.0
    sx = (True, True, 150)
    quads = [region_gate(f'Q{k}', ROOT, 'FL1-A', pts, yparm='SSC-A',
                         x=sx, inputs=[ROOT + '\\$c'])
             for k, pts in enumerate([
                 [(cx, cy), (0, cy), (0, 262143), (cx, 262143)],
                 [(cx, cy), (cx, 262143), (4095, 262143), (4095, cy)],
                 [(cx, cy), (0, cy), (0, 0), (cx, 0)],
                 [(cx, cy), (4095, cy), (4095, 0), (cx, 0)]], 1)]
    centre = region_gate('$c', ROOT, 'FL1-A', [(cx, cy)] * 3, yparm='SSC-A',
                         rtype='BINNER_REGION', gtype='Composite_Classifier',
                         x=sx)
    _plan, g = _one_tube(tmp_path, [centre] + quads)
    xl = _logicle_lin(cx / 4096, dr.diva_biexp_w(150))
    for k in range(1, 5):
        assert g[f'Q{k}']['quad_origin_x'] == pytest.approx(xl)
        assert g[f'Q{k}']['quad_origin_y'] == pytest.approx(cy)
    assert g['Q3']['y1'] == pytest.approx(cy)


def test_a_polygon_keeps_its_axes_and_opens_past_the_plot_edge(tmp_path):
    s = (True, True, 150)
    tri = [(1000, 1000), (3000, 1000), (2500, 4095), (1500, 4095)]
    _plan, g = _one_tube(tmp_path, [
        region_gate('tri', ROOT, 'FL1-A', tri, yparm='FL2-A', x=s, y=s),
        region_gate('flat', ROOT, 'FSC-A', [(1e3, 1e3), (9e4, 2e3), (5e4, 8e4)],
                    yparm='SSC-A'),
        region_gate('logged', ROOT, 'FL1-A', [(2, 1e3), (4, 2e3), (3, 8e4)],
                    yparm='SSC-A', x=(False, True, 0))])
    tri_g = g['tri']
    axes = tri_g['flowjo_edge_axes']
    assert axes[0] == {'kind': 'logicle', 'T': 262144.0,
                       'W': pytest.approx(dr.diva_biexp_w(150)),
                       'M': 4.5, 'A': 0.0}
    v = np.asarray(tri_g['vertices'])
    # The top side on the plot edge runs straight out to OPEN.
    assert len(v) == 6 and (v[:, 1] == OPEN).sum() == 2
    top = v[v[:, 1] == OPEN][:, 0]
    assert top.tolist() == pytest.approx(
        [_logicle_lin(2500 / 4096, dr.diva_biexp_w(150)),
         _logicle_lin(1500 / 4096, dr.diva_biexp_w(150))])
    assert 'flowjo_edge_axes' not in g['flat']
    assert g['logged']['flowjo_edge_axes'] == [
        {'kind': 'log', 'offset': pytest.approx(2.62144), 'decades': 1.0}, None]
    assert g['flat']['vertices'] == [[1e3, 1e3], [9e4, 2e3], [5e4, 8e4]]


def test_a_polygon_past_the_plot_edge_is_cut_there(tmp_path):
    """FACSDiva can store a vertex beyond the plot (here 4600 on a 4096
    channel axis). The plot shows the polygon cut at its edge: the sides
    keep their slope up to the cut, and the cut is opened past the edge."""
    s = (True, True, 150)
    w = dr.diva_biexp_w(150)
    past = [(1000, 1000), (2000, 800), (3000, 1000), (3000, 4600),
            (1000, 4600)]
    tilted = [(1000, 1000), (3000, 1000), (2000, 5095)]
    below = [(1000, -1000), (3000, 3095), (1000, 3095)]
    gone = [(1000, 4500), (3000, 4500), (2000, 5000)]
    plan, g = _one_tube(tmp_path, [
        region_gate('past', ROOT, 'FL1-A', past, yparm='FL2-A', x=s, y=s),
        region_gate('tilted', ROOT, 'FL1-A', tilted, yparm='FL2-A', x=s, y=s),
        region_gate('below', ROOT, 'FL1-A', below, yparm='FL2-A', x=s, y=s),
        region_gate('gone', ROOT, 'FL1-A', gone, yparm='FL2-A', x=s, y=s)])
    # Wholly past the edge: nothing on the plot, so left out and named.
    assert 'gone' not in g
    assert plan.unread == [('gone', ['POLYGON_REGION'], 0)]
    # The bottom edge (channel 0) cuts the same way: the side from
    # (1000, -1000) to (3000, 3095) crosses it at channel x = 1000 + 2000/4.095.
    b = np.asarray(g['below']['vertices'])
    assert sorted(b[b[:, 1] == -OPEN][:, 0]) == pytest.approx(
        [_logicle_lin(1000 / 4096, w), _logicle_lin((1000 + 2000 / 4.095) / 4096, w)])
    v = np.asarray(g['past']['vertices'])
    top = _logicle_lin(4095 / 4096, w)
    sides = [_logicle_lin(1000 / 4096, w), _logicle_lin(3000 / 4096, w)]
    assert sorted(v[v[:, 1] == OPEN][:, 0]) == pytest.approx(sides)
    assert sorted(v[np.isclose(v[:, 1], top)][:, 0]) == pytest.approx(sides)
    assert (v[:, 1] <= top * (1 + 1e-9)).sum() == len(v) - 2
    # The tilted sides cross the edge at x = 1500 + 0.5 and 2500 - 0.5
    # channels (a quarter of the way up their 4095-channel rise).
    t = np.asarray(g['tilted']['vertices'])
    xs = sorted(t[t[:, 1] == OPEN][:, 0])
    assert xs == pytest.approx([_logicle_lin(c / 4096, w) for c in
                                (1000 + 1000 * 3095 / 4095,
                                 3000 - 1000 * 3095 / 4095)])


# ── The tree ─────────────────────────────────────────────────────────────────

def test_parents_booleans_and_unread_populations(tmp_path):
    p1 = region_gate('P1', ROOT, 'FSC-A', [(1e3, 1e3), (2e5, 1e3), (2e5, 2e5),
                                          (1e3, 2e5)], yparm='SSC-A',
                     rtype='RECTANGLE_REGION')
    p2 = region_gate('P2', P1, 'FL1-A', [(2.0, 0), (4.0, 0)],
                     rtype='INTERVAL_REGION', x=(False, True, 0))
    p3 = region_gate('P3', P1, 'FL2-A', [(2.0, 0), (4.0, 0)],
                     gtype='SnapTo_Interval_Classifier',
                     rtype='INTERVAL_REGION', x=(False, True, 0))
    snap = region_gate('Snap', P1, 'FSC-A', [(1e3, 1e3), (9e4, 2e3), (5e4, 8e4)],
                       yparm='SSC-A', gtype='SnapTo_Classifier')
    odd = region_gate('Odd', P1, 'FL1-A', [(1, 1)], yparm='FL2-A',
                      rtype='SPIRAL_REGION')
    under_odd = region_gate('Under', P1 + '\\Odd', 'FL1-A', [(2, 0), (3, 0)],
                            rtype='INTERVAL_REGION')
    deeper = region_gate('Deeper', P1 + '\\Odd\\Under', 'FL2-A',
                         [(2, 0), (3, 0)], rtype='INTERVAL_REGION')
    both = boolean_gate('Both', P1, 'AND_Classifier', [P1 + '\\P2', P1 + '\\P3'])
    either = boolean_gate('Either', P1, 'OR_Classifier', [P1 + '\\P2', P1 + '\\P3'])
    not2 = boolean_gate('Not2', P1, 'NOT_Classifier', [P1 + '\\P2'])
    rest = boolean_gate('Rest of All Events', ROOT, 'RestOf_Classifier', [P1])
    lost = boolean_gate('Lost', P1, 'AND_Classifier', [P1 + '\\Odd', P1 + '\\P2'])
    # Children listed before their parents still land under them.
    plan, g = _one_tube(tmp_path, [deeper, under_odd, both, p2, p1, p3, odd,
                                   either, not2, rest, lost, snap])
    ids = {k: v['_import_id'] for k, v in g.items()}
    assert g['P1']['parent_id'] is None
    assert g['P2']['parent_id'] == ids['P1'] == g['P3']['parent_id']
    assert (g['Both']['op'], g['Both']['operands']) == ('and', [ids['P2'], ids['P3']])
    assert (g['Either']['op'], g['Either']['operands']) == ('or', [ids['P2'], ids['P3']])
    assert (g['Not2']['op'], g['Not2']['operands']) == ('not', [ids['P2']])
    assert (g['Rest of All Events']['op'], g['Rest of All Events']['operands']) == (
        'not', [ids['P1']])
    assert g['Rest of All Events']['parent_id'] is None
    assert g['Lost']['operands'] == []          # reads NA, never every event
    assert not {'Odd', 'Under', 'Deeper'} & set(g)
    # Left out with the two populations under it.
    assert plan.unread == [('Odd', ['SPIRAL_REGION'], 2)]
    assert g['Snap']['kind'] == 'polygon' and g['Snap']['parent_id'] == ids['P1']
    # Parents come before children in the list.
    order = [x['label'] for x in plan.samples[0].gates]
    assert order.index('P1') < order.index('P2')


def test_a_population_takes_its_name_and_a_bad_point_leaves_it_out(tmp_path):
    named = dict(region_gate('P1', ROOT, 'FSC-A', [(1e3, 1e3), (9e4, 2e3),
                                                   (5e4, 8e4)], yparm='SSC-A'),
                 title='Shown name')
    bad = region_gate('Bad', ROOT, 'FSC-A', [(1e3, 1e3), (float('nan'), 2e3),
                                             (5e4, 8e4)], yparm='SSC-A')
    plan, g = _one_tube(tmp_path, [named, bad])
    assert set(g) == {'Shown name'}
    assert plan.unread == [('Bad', ['POLYGON_REGION'], 0)]


def test_a_gate_whose_parent_is_missing_is_named(tmp_path):
    lone = region_gate('Lone', ROOT + '\\Gone', 'FL1-A', [(2, 0), (3, 0)],
                       rtype='INTERVAL_REGION')
    plan, g = _one_tube(tmp_path, [lone])
    assert g == {}
    assert plan.unread == [('Lone', ['parent not in the file'], 0)]


# ── Compensation ─────────────────────────────────────────────────────────────

def test_compensation_on_gives_the_inverse_of_the_coefficients(tmp_path):
    plan, _g = _one_tube(tmp_path, [])
    comp = plan.samples[0].comp
    assert comp['channels'] == ['FL1-A', 'FL2-A']
    c = np.array([COEFFS['FL1-A'], COEFFS['FL2-A']]).T   # column j: param j
    np.testing.assert_allclose(comp['matrix'], np.linalg.inv(c))
    assert comp['source'] == 'diva:experiment'
    assert plan.notes == []


def test_compensation_off_opens_uncompensated_and_says_so(tmp_path):
    plan, _g = _one_tube(tmp_path, [], comp_on=False)
    comp = plan.samples[0].comp
    assert comp['channels'] == ['FL1-A', 'FL2-A']
    np.testing.assert_array_equal(comp['matrix'], np.eye(2))
    assert len(plan.notes) == 1 and 'uncompensated' in plan.notes[0]
    res = Document().import_wsp(str(tmp_path / 'exp.xml'))
    assert [n for n in res['notes'] if 'compensation was off' in n] == [
        'exp.xml: ' + plan.notes[0]]


def test_a_tube_without_settings_takes_the_experiments(tmp_path):
    make_fcs(tmp_path / 'a.fcs')
    xml = diva_xml(tmp_path / 'exp.xml',
                   [('S', [{'name': 'T', 'file': 'a.fcs', 'settings': False}])],
                   shared_settings=True)
    comp = wo.plan_wsp(xml).samples[0].comp
    assert comp is not None and comp['source'] == 'diva:experiment'


def test_settings_without_parameters_fall_back_to_the_experiments(tmp_path):
    make_fcs(tmp_path / 'a.fcs')
    xml = diva_xml(tmp_path / 'exp.xml',
                   [('S', [{'name': 'T', 'file': 'a.fcs', 'settings': 'empty'}])],
                   shared_settings=True)
    comp = wo.plan_wsp(xml).samples[0].comp
    assert comp is not None and comp['source'] == 'diva:experiment'


def test_a_tube_without_settings_or_a_square_matrix_keeps_the_files(tmp_path):
    plan, _g = _one_tube(tmp_path, [], settings=False)
    assert plan.samples[0].comp is None
    plan, _g = _one_tube(tmp_path, [], coeffs={'FL1-A': [1.0, 0.1, 0.0],
                                               'FL2-A': [0.0, 1.0]})
    assert plan.samples[0].comp is None
    plan, _g = _one_tube(tmp_path, [], coeffs={'FL1-A': [1.0, 'nan'],
                                               'FL2-A': [0.0, 1.0]})
    assert plan.samples[0].comp is None


# ── Finding the files ────────────────────────────────────────────────────────

def _tube(name, file, events=4000):
    return {'name': name, 'file': file, 'events': events}


def test_files_are_found_by_name_by_fil_and_by_specimen_and_tube(tmp_path):
    a = make_fcs(tmp_path / 'a.fcs', seed=1)
    # Renamed on export; $FIL keeps FACSDiva's name.
    b = make_fcs(tmp_path / 'sub' / 'Spec_A_Tube_2.fcs', seed=2,
                 keywords={'$FIL': '0002.fcs'})
    # Renamed, $FIL elsewhere: its specimen and tube keywords tie it.
    c = make_fcs(tmp_path / 'Spec_B_Tube_1.fcs', seed=3,
                 keywords={'$FIL': 'other.fcs', '$SRC': 'Spec_B',
                           'TUBE NAME': 'Tube_1'})
    xml = diva_xml(tmp_path / 'exp.xml', [
        ('Spec_A', [_tube('Tube_1', 'a.fcs'), _tube('Tube_2', '0002.fcs'),
                    _tube('Never', 'none.fcs', events=0),
                    _tube('Lost', 'lost.fcs')]),
        ('Spec_B', [_tube('Tube_1', '0003.fcs')])])
    plan = wo.plan_wsp(xml)
    assert [(s.node_name, s.fcs_path, s.groups) for s in plan.samples] == [
        ('Spec_A / Tube_1', a, ['Spec_A']),
        ('Spec_A / Tube_2', b, ['Spec_A']),
        ('Spec_B / Tube_1', c, ['Spec_B'])]
    # A tube that recorded nothing is passed over; one that did is named.
    assert plan.unresolved == ['Spec_A / Lost']


def test_two_files_for_one_tube_are_not_guessed_between(tmp_path):
    kw = {'$SRC': 'S', 'TUBE NAME': 'T', 'EXPERIMENT NAME': 'Exp_001'}
    make_fcs(tmp_path / 'x1.fcs', keywords=kw)
    make_fcs(tmp_path / 'x2.fcs', keywords=kw)
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [_tube('T', 'gone.fcs')])])
    plan = wo.plan_wsp(xml)
    assert plan.samples == [] and plan.unresolved == ['S / T']
    # The experiment name tells them apart when one has another.
    make_fcs(tmp_path / 'x2.fcs', keywords=dict(kw, **{'EXPERIMENT NAME': 'B'}))
    plan = wo.plan_wsp(xml)
    assert [os.path.basename(s.fcs_path) for s in plan.samples] == ['x1.fcs']
    # Or when one names no experiment: the one naming this one wins.
    make_fcs(tmp_path / 'x2.fcs', keywords={'$SRC': 'S', 'TUBE NAME': 'T'})
    plan = wo.plan_wsp(xml)
    assert [os.path.basename(s.fcs_path) for s in plan.samples] == ['x1.fcs']


def test_a_file_from_another_experiment_is_never_taken(tmp_path):
    """FACSDiva's default names repeat between experiments: a lone file
    whose EXPERIMENT NAME is another's is not this tube's, by tube or by
    $FIL; one that names no experiment still is."""
    kw = {'$SRC': 'Spec_1', 'TUBE NAME': 'Tube_001',
          'EXPERIMENT NAME': 'Exp_OTHER', '$FIL': 'gone.fcs'}
    make_fcs(tmp_path / 'other.fcs', keywords=kw)
    xml = diva_xml(tmp_path / 'exp.xml',
                   [('Spec_1', [_tube('Tube_001', 'gone.fcs')])],
                   experiment='Exp_THIS')
    plan = wo.plan_wsp(xml)
    assert plan.samples == [] and plan.unresolved == ['Spec_1 / Tube_001']
    kw.pop('EXPERIMENT NAME')
    mine = make_fcs(tmp_path / 'other.fcs', keywords=kw)
    assert [s.fcs_path for s in wo.plan_wsp(xml).samples] == [mine]


def test_windows_paths_in_the_xml_and_the_file_still_match(tmp_path):
    f = make_fcs(tmp_path / 'renamed.fcs', keywords={'$FIL': 'D:\\Exp\\0007.fcs'})
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [_tube('T', 'C:\\BD\\0007.fcs')])])
    assert [s.fcs_path for s in wo.plan_wsp(xml).samples] == [f]


def test_hidden_and_very_deep_folders_are_not_searched(tmp_path):
    kw = {'$FIL': 'a.fcs'}
    make_fcs(tmp_path / '.cache' / 'x.fcs', keywords=kw)
    make_fcs(tmp_path.joinpath(*'bcdefgh') / 'y.fcs', keywords=kw)
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [_tube('T', 'a.fcs')])])
    assert wo.plan_wsp(xml).unresolved == ['S / T']
    near = make_fcs(tmp_path.joinpath(*'bcdef') / 'z.fcs', keywords=kw)
    assert [s.fcs_path for s in wo.plan_wsp(xml).samples] == [near]


def test_a_name_the_locator_finds_twice_is_named(tmp_path):
    for sub in ('p1', 'p2'):
        make_fcs(tmp_path / sub / 'a.fcs')
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [_tube('T', 'a.fcs')])])
    plan = wo.plan_wsp(xml)
    assert plan.unresolved == ['S / T']
    assert plan.ambiguous == [('a.fcs', 2)]


def test_an_event_count_other_than_the_xmls_is_noted(tmp_path):
    make_fcs(tmp_path / 'a.fcs', n=4000)
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [_tube('T', 'a.fcs', 3999)])])
    plan = wo.plan_wsp(xml)
    assert len(plan.samples) == 1
    (nm, why), = plan.mismatched
    assert nm == 'S / T' and '3999' in why and 'FACSDiva XML' in why


def test_the_fcs_folder_is_searched_by_keywords_too(tmp_path):
    from openflo.compare import FcsLocator
    other = tmp_path / 'elsewhere'
    f = make_fcs(other / 'renamed.fcs', keywords={'$FIL': 'a.fcs'})
    (tmp_path / 'xml').mkdir()
    xml = diva_xml(tmp_path / 'xml' / 'exp.xml', [('S', [_tube('T', 'a.fcs')])])
    assert wo.plan_wsp(xml).unresolved == ['S / T']
    plan = wo.plan_wsp(xml, FcsLocator(xml, str(other)))
    assert [s.fcs_path for s in plan.samples] == [f]


def test_the_callers_locator_is_asked_first(tmp_path):
    moved = make_fcs(tmp_path / 'store' / 'copy.fcs')
    xml = diva_xml(tmp_path / 'exp.xml', [('S', [_tube('T', 'a.fcs')])])
    asked = []

    def find(name):
        asked.append(name)
        return moved if name == 'a.fcs' else None
    plan = wo.plan_wsp(xml, find)
    assert asked == ['a.fcs'] and plan.samples[0].fcs_path == moved


# ── Telling the file apart ───────────────────────────────────────────────────

def test_only_a_bdfacs_xml_is_a_facsdiva_experiment(tmp_path):
    xml = diva_xml(tmp_path / 'exp.xml', [])
    gml = tmp_path / 'gates.xml'
    gml.write_text('<?xml version="1.0"?><gating:Gating-ML xmlns:gating='
                   '"http://www.isac-net.org/std/Gating-ML/v2.0/gating"/>')
    bad = tmp_path / 'bad.xml'
    bad.write_text('<bdfacs')
    assert dr.is_diva_xml(xml) and is_workspace_file(xml)
    assert not dr.is_diva_xml(gml) and not is_workspace_file(gml)
    assert not dr.is_diva_xml(bad) and not dr.is_diva_xml(tmp_path / 'no.xml')
    make_fcs(tmp_path / 'a.fcs')
    fcs, wsp = expand_dropped_paths([str(tmp_path)])
    assert wsp == [xml] and fcs == [str(tmp_path / 'a.fcs')]


def test_an_entity_declaration_is_refused(tmp_path):
    p = tmp_path / 'evil.xml'
    p.write_text('<?xml version="1.0"?><!DOCTYPE bdfacs [<!ENTITY a "aa">]>'
                 '<bdfacs><experiment/></bdfacs>')
    with pytest.raises(wo.WspUnreadable, match='entity'):
        wo.plan_wsp(str(p))


# ── End to end: counts ───────────────────────────────────────────────────────

def test_imported_gates_count_what_facsdiva_drew(tmp_path):
    """The engine opens the tube compensated with the XML's matrix and its
    gates count the events numpy counts on the same compensated values; the
    quadrants add up to their parent."""
    s = (True, True, 150)
    w = dr.diva_biexp_w(150)
    cx, cy = 1800.0, 1600.0
    gates = [
        region_gate('P1', ROOT, 'FSC-A', [(2e4, 1e4), (2.4e5, 1e4),
                                         (2.4e5, 262143), (2e4, 262143)],
                    yparm='SSC-A', rtype='RECTANGLE_REGION'),
        region_gate('P2', P1, 'FL1-A', [(cx, 0), (4095, 0)],
                    rtype='INTERVAL_REGION', x=s),
    ] + [region_gate(f'Q{k}', P1, 'FL1-A', pts, yparm='FL2-A', x=s, y=s)
         for k, pts in enumerate([
             [(cx, cy), (0, cy), (0, 4095), (cx, 4095)],
             [(cx, cy), (cx, 4095), (4095, 4095), (4095, cy)],
             [(cx, cy), (0, cy), (0, 0), (cx, 0)],
             [(cx, cy), (4095, cy), (4095, 0), (cx, 0)]], 1)]
    make_fcs(tmp_path / 'a.fcs', seed=7)
    xml = diva_xml(tmp_path / 'exp.xml',
                   [('S', [{'name': 'T', 'file': 'a.fcs', 'gates': gates}])])
    doc = Document()
    res = doc.import_wsp(xml)
    (name,) = res['samples']
    # Each gate is on the values FACSDiva compensated: nothing to warn of.
    assert res['uncompensated'] == [] and res['notes'] == []
    counts = {r['population'].split('/')[-1]: r['count']
              for r in doc.stats(name, channels=[])}

    ev = make_events(seed=7)
    c = np.array([COEFFS['FL1-A'], COEFFS['FL2-A']]).T
    fl = ev[:, 2:4].astype(float) @ c
    p1 = ((ev[:, 0] >= 2e4) & (ev[:, 0] <= 2.4e5) & (ev[:, 1] >= 1e4))
    xl, yl = _logicle_lin(cx / 4096, w), _logicle_lin(cy / 4096, w)
    assert counts['P1'] == int(p1.sum())
    assert counts['P2'] == int((p1 & (fl[:, 0] >= xl)).sum())
    quads = [counts[f'Q{k}'] for k in range(1, 5)]
    assert sum(quads) == counts['P1']
    assert counts['Q2'] == int((p1 & (fl[:, 0] >= xl) & (fl[:, 1] >= yl)).sum())
    assert counts['Q3'] == int((p1 & (fl[:, 0] < xl) & (fl[:, 1] < yl)).sum())
    assert 0 < counts['Q3'] < counts['P1']


def test_add_fcs_sends_a_facsdiva_xml_to_the_workspace_import(tmp_path,
                                                             monkeypatch):
    """The editor's Add FCS dialog offers the FACSDiva XML and imports it
    as a workspace; another .xml (Gating-ML) is not taken for one."""
    pytest.importorskip('tkinter')
    from types import SimpleNamespace

    from openflo import editor_load
    xml = diva_xml(tmp_path / 'exp.xml', [])
    gml = tmp_path / 'gates.xml'
    gml.write_text('<Gating-ML/>')
    fcs = make_fcs(tmp_path / 'a.fcs')
    seen = {}

    def ask(**kw):
        seen['types'] = kw['filetypes']
        return (xml, str(gml), fcs)
    monkeypatch.setattr(editor_load.filedialog, 'askopenfilenames', ask)
    status = []
    ed = SimpleNamespace(_open_dir=lambda: str(tmp_path), wsp=[], fcs=[],
                         status_var=SimpleNamespace(set=status.append))
    ed._ingest_wsp = ed.wsp.append
    ed._queue_fcs_loads = ed.fcs.extend
    editor_load.LoadMixin._add_samples(ed)
    assert ed.wsp == [xml]
    # The Gating-ML file is not queued as an FCS load; the status says so.
    assert ed.fcs == [fcs]
    assert len(status) == 1 and 'gates.xml' in status[0]
    assert 'not a FACSDiva experiment' in status[0]
    assert ('BD FACSDiva experiment (XML)', '*.xml') in seen['types']
