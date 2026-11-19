"""Each path through wsp_gate_coords.map_gate_coords, pinned directly.

The .wsp round-trip tests (test_wsp_gate_units, test_wsp_polygon_edge_axes)
count events through real transforms, so they cannot see which object comes
back, which keys are mapped by which axis' function, or what an ellipse with
a singular covariance leaves behind. These tests use plain arithmetic maps on
generated gates (channels FL1-A / FL2-A) so every expected value is exact.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from openflo import wsp_gate_coords
from openflo.wsp_gate_coords import map_gate_coords

OPEN = 1e12


def _double(v):
    return np.asarray(v, dtype=float) * 2.0


def _triple(v):
    return np.asarray(v, dtype=float) * 3.0


def _log10(v):
    with np.errstate(all='ignore'):
        return np.log10(np.asarray(v, dtype=float))


FNS = {'FL1-A': _double, 'FL2-A': _triple}


# ── the gate itself comes back when nothing maps ───────────────────────────

@pytest.mark.parametrize('fns', [None, {}], ids=['None', 'empty'])
def test_no_functions_returns_the_gate_itself(fns):
    g = {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'x0': 1.0, 'x1': 2.0, 'y0': 3.0, 'y1': 4.0}
    assert map_gate_coords(g, fns) is g


@pytest.mark.parametrize('channel', ['FL3-A', None, 5], ids=['other', 'None', 'int'])
def test_a_1d_gate_on_an_unmapped_channel_returns_the_gate_itself(channel):
    g = {'kind': 'threshold', 'channel': channel, 'value': 10.0}
    assert map_gate_coords(g, FNS) is g


def test_a_2d_gate_on_unmapped_channels_returns_the_gate_itself():
    g = {'kind': 'polygon', 'x_channel': 'FL3-A', 'y_channel': 'FL4-A',
         'vertices': [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}
    assert map_gate_coords(g, FNS) is g


# ── 1-D gates ──────────────────────────────────────────────────────────────

def test_a_1d_gate_maps_value_lo_and_hi_and_keeps_none_and_open_bounds():
    g = {'kind': 'interval', 'channel': 'FL1-A', 'value': 1.5, 'lo': -OPEN,
         'hi': 4.0, 'name': None}
    out = map_gate_coords(g, FNS)
    assert out is not g
    assert out == {'kind': 'interval', 'channel': 'FL1-A', 'value': 3.0,
                   'lo': -OPEN, 'hi': 8.0, 'name': None}
    assert g['hi'] == 4.0, 'the input is not changed'
    g2 = {'kind': 'threshold', 'channel': 'FL1-A', 'value': None, 'hi': 1.0}
    assert map_gate_coords(g2, FNS) == {'kind': 'threshold', 'channel': 'FL1-A',
                                        'value': None, 'hi': 2.0}


def test_a_non_finite_image_becomes_the_open_lower_sentinel():
    g = {'kind': 'interval', 'channel': 'FL1-A', 'lo': -5.0, 'hi': 0.0,
         'value': 100.0}
    out = map_gate_coords(g, {'FL1-A': _log10})
    assert out['lo'] == -OPEN and out['hi'] == -OPEN
    assert out['value'] == 2.0


# ── 2-D bounds: each key through its own axis' function ────────────────────

def test_box_keys_map_through_their_own_axis():
    g = {'kind': 'quadrant', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'x0': 1.0, 'x1': 2.0, 'quad_origin_x': 5.0,
         'y0': 1.0, 'y1': 2.0, 'quad_origin_y': 5.0}
    out = map_gate_coords(g, FNS)
    assert (out['x0'], out['x1'], out['quad_origin_x']) == (2.0, 4.0, 10.0)
    assert (out['y0'], out['y1'], out['quad_origin_y']) == (3.0, 6.0, 15.0)


# ── polygons ───────────────────────────────────────────────────────────────

TRIANGLE = [[0.0, 0.0], [16.0, 0.0], [0.0, 16.0]]


def _cube(v):
    return np.asarray(v, dtype=float) ** 3


def test_a_straight_map_moves_the_corners_and_adds_no_points():
    # Each axis scaled: an edge straight before is straight after, so the
    # polygon keeps exactly its corners, each mapped by its own axis.
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': TRIANGLE}
    out = map_gate_coords(g, FNS)
    assert out['vertices'] == [[0.0, 0.0], [32.0, 0.0], [0.0, 48.0]]
    assert g['vertices'] is TRIANGLE


def test_a_curving_map_traces_the_edges_between_the_mapped_corners():
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]]}
    v = np.array(map_gate_coords(g, {'FL1-A': _cube, 'FL2-A': _cube})['vertices'])
    corners = [[1.0, 1.0], [64.0, 1.0], [1.0, 64.0]]
    idx = [int(np.flatnonzero((v == c).all(axis=1))[0]) for c in corners]
    assert idx[0] == 0 and idx[0] < idx[1] < idx[2]
    # The two edges along an axis stay straight (y or x stays 1); the
    # diagonal one, a curve after cubing, gets points in between.
    assert idx[1] == 1 and idx[2] == len(v) - 1 > 2
    assert len(set(map(tuple, v.tolist()))) == len(v)


def test_an_edge_with_an_open_end_is_not_traced():
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[1.0, 1.0], [OPEN, 1.0], [1.0, 4.0]]}
    out = map_gate_coords(g, {'FL1-A': _cube, 'FL2-A': _cube})
    # Edges 0-1 and 1-2 touch the open vertex and stay as they are; 2-0
    # runs along x = 1 and maps straight.
    assert out['vertices'] == [[1.0, 1.0], [OPEN, 1.0], [1.0, 64.0]]


def test_a_polygon_with_under_three_vertices_is_copied_unmapped():
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[1.0, 1.0], [2.0, 2.0]], 'x0': 1.0}
    out = map_gate_coords(g, FNS)
    assert out is not g
    assert out['vertices'] == [[1.0, 1.0], [2.0, 2.0]]
    assert out['x0'] == 2.0


def test_flowjo_edge_axes_are_traced_not_subdivided_and_dropped():
    axes = [{'kind': 'log', 'offset': 1.0, 'decades': 5.0}, None]
    verts = [[10.0, 0.0], [1000.0, 0.0], [10.0, 50.0]]
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': verts, 'flowjo_edge_axes': axes}
    traced = wsp_gate_coords._trace_edges_on_axes(
        verts, wsp_gate_coords.polygon_edge_fns(g))
    out = map_gate_coords(g, FNS)
    assert 'flowjo_edge_axes' not in out and 'flowjo_edge_axes' in g
    assert len(out['vertices']) == len(traced)       # no 16-step subdivision
    assert out['vertices'] == [[2.0 * x, 3.0 * y] for x, y in traced]
    # traced even with no functions at all
    plain = map_gate_coords(g, {})
    assert plain is not g and plain['vertices'] == traced


# ── ellipses ───────────────────────────────────────────────────────────────

def test_an_ellipse_becomes_a_subdivided_mapped_polygon(monkeypatch):
    calls = []
    square = [[0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0]]

    def fake_polygon(mean, cov, distance_sq):
        calls.append((mean, cov, distance_sq))
        return square
    monkeypatch.setattr(wsp_gate_coords, '_ellipse_polygon', fake_polygon)
    g = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [8.0, 8.0], 'cov': [[1.0, 0.0], [0.0, 1.0]], 'x0': 1.0}
    out = map_gate_coords(g, FNS)
    assert calls == [([8.0, 8.0], [[1.0, 0.0], [0.0, 1.0]], 4.0)]
    assert out['kind'] == 'polygon'
    assert not {'mean', 'cov', 'distance_sq'} & set(out)
    assert out['x0'] == 2.0
    # it fell through into the polygon branch: mapped corner by corner
    assert out['vertices'] == [[0.0, 0.0], [32.0, 0.0], [32.0, 48.0], [0.0, 48.0]]
    assert g['kind'] == 'ellipsoid' and 'mean' in g


def test_an_ellipse_passes_its_own_distance(monkeypatch):
    seen = []
    monkeypatch.setattr(wsp_gate_coords, '_ellipse_polygon',
                        lambda m, c, d: seen.append(d) or [[0, 0], [1, 0], [0, 1]])
    map_gate_coords({'kind': 'ellipsoid', 'x_channel': 'FL1-A',
                     'y_channel': 'FL2-A', 'mean': [0, 0], 'cov': [[1, 0], [0, 1]],
                     'distance_sq': 2.5}, FNS)
    assert seen == [2.5]


def test_a_singular_ellipse_is_left_as_is_with_a_warning(caplog):
    g = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [1.0, 1.0], 'cov': [[1.0, 1.0], [1.0, 1.0]],
         'distance_sq': 4.0, 'name': 'E1', 'x0': 1.0}
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        out = map_gate_coords(g, FNS)
    assert out is not g
    assert out['kind'] == 'ellipsoid'
    assert out['mean'] == [1.0, 1.0] and out['cov'] == [[1.0, 1.0], [1.0, 1.0]]
    assert 'vertices' not in out
    assert out['x0'] == 2.0, 'bounds are mapped before the ellipse is tried'
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('E1' in m and 'singular covariance' in m for m in msgs)


def test_a_singular_ellipse_with_vertices_does_not_reach_the_polygon_branch():
    verts = [[0.0, 0.0], [16.0, 0.0], [0.0, 16.0]]
    g = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [1.0, 1.0], 'cov': [[0.0, 0.0], [0.0, 0.0]], 'vertices': verts}
    out = map_gate_coords(g, FNS)
    assert out['vertices'] is verts


# ── reviewer probes: types, guards and the closing edge ───────────────────

def test_1d_values_are_plain_floats():
    out = map_gate_coords({'kind': 'interval', 'channel': 'FL1-A', 'value': 1.5,
                           'lo': 1.0, 'hi': 4.0}, FNS)
    assert all(type(out[k]) is float for k in ('value', 'lo', 'hi'))


def test_singular_ellipse_warning_names_the_id_when_nameless(caplog):
    g = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': [1.0, 1.0], 'cov': [[1.0, 1.0], [1.0, 1.0]], 'name': None, 'id': 'g7'}
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        map_gate_coords(g, FNS)
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any('g7' in m for m in msgs)


def test_three_column_vertices_are_left_alone():
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[1.0, 2.0, 3.0]] * 3}
    assert map_gate_coords(g, FNS)['vertices'] == [[1.0, 2.0, 3.0]] * 3


def test_a_column_vector_image_is_flattened():
    out = map_gate_coords({'kind': 'threshold', 'channel': 'FL1-A', 'value': 1.0},
                          {'FL1-A': lambda v: (np.asarray(v) * 2).reshape(-1, 1)})
    assert out['value'] == 2.0


def test_no_call_when_every_bound_is_open():
    calls = []
    out = map_gate_coords({'kind': 'threshold', 'channel': 'FL1-A', 'value': OPEN},
                          {'FL1-A': lambda v: calls.append(v) or v})
    assert calls == [] and out['value'] == OPEN


def test_vertex_coordinates_are_plain_floats():
    g = {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'vertices': [[0.0, 0.0], [16.0, 0.0], [0.0, 16.0]]}
    assert all(type(c) is float for p in map_gate_coords(g, FNS)['vertices'] for c in p)


def test_an_ellipsoid_with_no_mean_passes_through_as_a_mapped_copy():
    g = {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
         'mean': None, 'x0': 1.0}
    out = map_gate_coords(g, FNS)
    assert out['kind'] == 'ellipsoid' and out['x0'] == 2.0 and 'vertices' not in out
