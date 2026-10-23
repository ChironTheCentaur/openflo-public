"""Tests for openflo.plotmath — pure plot/maths helpers extracted from gui.py."""
from __future__ import annotations

import numpy as np

from openflo.plotmath import (
    drop_suffix,
    ellipse_geom,
    ellipse_outline,
    ellipse_params,
    gid_from_hit,
    hist_bin_edges,
    in_box,
    point_segment_dist,
    polygon_on_screen,
    polyline_on_screen,
    reroute_on_screen,
    screen_corners,
    symlog_linthresh,
)


def test_drop_suffix():
    assert drop_suffix(0, 0) == ''
    assert drop_suffix(None, 1000) == ''
    s = drop_suffix(150, 1000)
    assert 'drops 150' in s and '15.0%' in s


def test_in_box():
    assert in_box((5, 5), (0, 0, 10, 10)) is True
    assert in_box((5, 5), (6, 6, 10, 10)) is False
    assert in_box((0, 0), (0, 0, 10, 10)) is True      # inclusive edges
    assert in_box(None, (0, 0, 1, 1)) is False
    assert in_box((1, 1), None) is False


def test_symlog_linthresh():
    assert symlog_linthresh(None) == 1.0               # default
    assert symlog_linthresh([1.0, 2.0]) == 1.0         # too few points
    data = np.concatenate([np.zeros(100), np.linspace(10, 1000, 100)])
    lt = symlog_linthresh(data)
    assert lt >= 1e-6 and lt < 1000                    # 5th pct of |nonzero|


def test_hist_bin_edges_linear_and_log():
    lin = hist_bin_edges(0.0, 10.0, 'linear', n_bins=10)
    assert len(lin) == 11 and lin[0] == 0.0 and lin[-1] == 10.0
    log = hist_bin_edges(1.0, 1000.0, 'log', n_bins=3)
    assert len(log) == 4
    # log spacing → roughly geometric (ratios ~constant), not arithmetic
    ratios = [log[i + 1] / log[i] for i in range(3)]
    assert max(ratios) - min(ratios) < 0.5
    # non-positive lo on a log axis still yields a usable, sorted grid
    safe = hist_bin_edges(-5.0, 100.0, 'log', n_bins=10)
    assert len(safe) == 11 and all(b > 0 for b in safe)


def test_ellipse_params():
    # axis-aligned covariance diag(4, 1), dist_sq=1 → full axes 2*sqrt(4)=4 and
    # 2*sqrt(1)=2 (which maps to width vs height depends on eigh's ascending
    # order, so assert the axis *set*); axis-aligned → angle a multiple of 90.
    gate = {'mean': [0.0, 0.0], 'cov': [[4.0, 0.0], [0.0, 1.0]],
            'distance_sq': 1.0}
    cx, cy, w, h, ang = ellipse_params(gate)
    assert (cx, cy) == (0.0, 0.0)
    assert {round(w, 6), round(h, 6)} == {4.0, 2.0}
    assert abs(ang) % 90.0 < 1e-6 or abs(abs(ang) % 90.0 - 90.0) < 1e-6
    # malformed → None
    assert ellipse_params({'mean': [0.0], 'cov': [[1.0]]}) is None
    assert ellipse_params({}) is None


def test_ellipse_geom():
    gate = {'mean': [1.0, 2.0], 'cov': [[4.0, 0.0], [0.0, 1.0]],
            'distance_sq': 4.0}
    res = ellipse_geom(gate)
    assert res is not None
    (cx, cy), inv, r0, (hx, hy) = res
    assert (cx, cy) == (1.0, 2.0)
    assert abs(r0 - 2.0) < 1e-9                      # sqrt(distance_sq)
    assert inv.shape == (2, 2)
    # handle sits off the centre (the rotation grip), not at it
    assert (hx, hy) != (cx, cy)
    assert ellipse_geom({}) is None
    assert ellipse_geom({'mean': [0, 0], 'cov': [[0, 0], [0, 0]],
                         'distance_sq': 1.0}) is None   # singular cov


def test_point_segment_dist():
    # point on the segment → ~0
    assert point_segment_dist(5, 0, 0, 0, 10, 0, 10, 10) < 1e-9
    # perpendicular offset, normalised by span: point (5,2) to x-axis seg,
    # span_y=10 → 0.2
    assert abs(point_segment_dist(5, 2, 0, 0, 10, 0, 10, 10) - 0.2) < 1e-9
    # degenerate segment (a == b) → distance to the point
    d = point_segment_dist(3, 4, 0, 0, 0, 0, 1, 1)
    assert abs(d - 5.0) < 1e-9                       # 3-4-5


def test_gid_from_hit():
    assert gid_from_hit(('line', 'g1')) == 'g1'
    assert gid_from_hit(('line', 'g1:lo')) == 'g1'   # threshold packs gid:edge
    assert gid_from_hit(('line', 'g1:hi')) == 'g1'
    assert gid_from_hit(None) is None
    assert gid_from_hit(('only-one',)) is None
    assert gid_from_hit(('x', 123)) is None          # non-str second


# A log-log "screen": 200 px per decade, as an axes' transData would map it.
def _to_screen(p):
    return 200.0 * np.log10(np.asarray(p, dtype=float))


def _from_screen(d):
    return 10.0 ** (np.asarray(d, dtype=float) / 200.0)


def _off_segment(p, a, b):
    """Pixel distance from each point in ``p`` to the segment ``a``-``b``."""
    ab = b - a
    s = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.hypot(*(p - (a + s[:, None] * ab)).T)


def _worst_chord_gap(verts):
    """Largest on-screen distance between a DATA-straight edge (what
    gate_to_mask tests) and the screen-straight edge (what is drawn)."""
    v = np.asarray(verts, dtype=float)
    w = np.vstack([v, v[:1]])
    s = np.linspace(0.0, 1.0, 201)[:, None]
    gap = 0.0
    for a, b in zip(w[:-1], w[1:], strict=True):
        gap = max(gap, float(_off_segment(_to_screen(a + s * (b - a)),
                                          _to_screen(a), _to_screen(b)).max()))
    return gap


def test_polygon_on_screen_matches_the_drawn_edges_and_keeps_the_vertices():
    tri = [(1e3, 1e3), (1e5, 1e3), (1e3, 1e5)]
    assert _worst_chord_gap(tri) > 50          # the bare triangle: way off
    out = polygon_on_screen(tri, _to_screen, _from_screen)
    assert _worst_chord_gap(out) <= 0.25
    # Every given vertex survives, exactly and in order; only the hypotenuse
    # bends on a log-log screen, so the two axis-parallel edges gain nothing.
    idx = [int(np.flatnonzero((out == v).all(axis=1))[0]) for v in tri]
    assert idx == [0, 1, len(out) - 1] and len(out) > 3


def test_polygon_on_screen_adds_nothing_on_a_linear_screen():
    poly = [(1.0, 2.0), (90.0, 1.5), (60.0, 70.0), (2.0, 30.0)]
    out = polygon_on_screen(poly, lambda p: 3.0 * np.asarray(p) + 7.0,
                            lambda d: (np.asarray(d) - 7.0) / 3.0)
    assert out.tolist() == [list(v) for v in poly]


def test_ellipse_outline_traces_the_gates_own_level_set():
    gate = {'mean': [1500.0, 1500.0], 'cov': [[4e5, 2e5], [2e5, 3e5]],
            'distance_sq': 4.0}
    out = ellipse_outline(ellipse_params(gate), _to_screen)
    d = out - np.asarray(gate['mean'])
    md2 = np.einsum('ij,jk,ik->i', d, np.linalg.inv(gate['cov']), d)
    assert np.allclose(md2, 4.0)                 # every point on the rim
    # ...and dense enough that the drawn (screen-straight) segments between
    # them stay within a fraction of a pixel of the curve's true image.
    cx, cy, w, h, ang = ellipse_params(gate)
    t = np.linspace(0.0, 2 * np.pi, 4001)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    u, v = 0.5 * w * np.cos(t), 0.5 * h * np.sin(t)
    true = _to_screen(np.column_stack([cx + c * u - s * v, cy + s * u + c * v]))
    d = _to_screen(np.vstack([out, out[:1]]))
    nearest = np.min([_off_segment(true, a, b)
                      for a, b in zip(d[:-1], d[1:], strict=True)], axis=0)
    assert nearest.max() <= 0.3


_TRI = [(1e3, 1e3), (1e5, 1e3), (1e3, 1e5)]


def _dense(poly):
    return polygon_on_screen(poly, _to_screen, _from_screen)


def test_screen_corners_finds_the_drawn_corners_of_a_densified_polygon():
    dense = _dense(_TRI)
    assert np.flatnonzero(screen_corners(dense, _to_screen)).tolist() == [
        0, 1, len(dense) - 1]


def test_reroute_on_screen_moves_a_corner_as_drawing_the_new_shape_would():
    """Dragging a corner of a polygon stored with points along its edges.
    Moving the one stored point leaves a notch whose data-straight edges
    bend on screen; the re-routed polygon is what drawing the moved shape
    stores, point for point."""
    dense, m = _dense(_TRI), (3e5, 3e3)
    naive = dense.copy()
    naive[1] = m
    assert _worst_chord_gap(naive) > 10
    out = reroute_on_screen(dense, 1, m, _to_screen, _from_screen)
    assert out is not None and np.array_equal(out, _dense([_TRI[0], m, _TRI[2]]))
    # A point taken from along an edge becomes a corner of its own.
    j = len(dense) // 2
    out = reroute_on_screen(dense, j, m, _to_screen, _from_screen)
    assert out is not None
    assert np.array_equal(out, _dense([_TRI[0], _TRI[1], m, _TRI[2]]))


def test_reroute_on_screen_removes_a_corner_and_refuses_below_three():
    quad = [(1e3, 1e3), (1e5, 1e3), (1e5, 1e5), (1e3, 3e4)]
    dense = _dense(quad)
    c = int(np.flatnonzero((dense == quad[2]).all(axis=1))[0])
    out = reroute_on_screen(dense, c, None, _to_screen, _from_screen)
    assert out is not None
    assert np.array_equal(out, _dense([quad[0], quad[1], quad[3]]))
    assert reroute_on_screen(_dense(_TRI), 1, None, _to_screen,
                             _from_screen) is None


def test_reroute_on_screen_on_a_linear_screen_moves_just_the_vertex():
    poly = [(1.0, 2.0), (90.0, 1.5), (60.0, 70.0), (2.0, 30.0)]
    out = reroute_on_screen(poly, 2, (50.0, 80.0),
                            lambda p: 3.0 * np.asarray(p) + 7.0,
                            lambda d: (np.asarray(d) - 7.0) / 3.0)
    assert out is not None and out.tolist() == [[1.0, 2.0], [90.0, 1.5], [50.0, 80.0], [2.0, 30.0]]


def test_a_user_vertex_on_an_edge_straight_in_data_too_is_a_corner():
    """(1e4, 1e3) lies on the screen line between its neighbours, as the
    points polygon_on_screen adds do; but that edge is horizontal, straight
    in data as well, where nothing is ever added: the user placed it, and a
    drag of its neighbour must not drop it."""
    quad = [(1e3, 1e3), (1e4, 1e3), (1e5, 1e3), (1e3, 1e5)]
    assert screen_corners(quad, _to_screen).all()
    out = reroute_on_screen(quad, 2, (2e5, 2e3), _to_screen, _from_screen)
    assert out is not None
    assert [1e4, 1e3] in out.tolist()
    assert np.array_equal(out, _dense([quad[0], quad[1], (2e5, 2e3), quad[3]]))


def test_reroute_on_screen_puts_a_new_vertex_on_an_edge_between_its_corners():
    """A new vertex on the edge between two points added along the
    hypotenuse: re-routed from the hypotenuse's corners, not raised as a
    tent between those two points."""
    dense, m = _dense(_TRI), (3e4, 3e4)
    j = len(dense) // 2
    out = reroute_on_screen(dense, j, m, _to_screen, _from_screen, on_edge=True)
    assert out is not None
    assert np.array_equal(out, _dense([_TRI[0], _TRI[1], m, _TRI[2]]))


def test_screen_corners_a_point_on_the_line_beyond_its_neighbour_is_a_corner():
    """The tip of a spike is on the screen line through its two neighbours,
    but not between them: a corner. (The line here is bent in data, so
    only the between-ness test can say so.)"""
    spike = [(1e3, 1e3), (1e5, 1e7), (1e4, 1e5), (1e5, 1e3)]
    assert screen_corners(spike, _to_screen).all()


def test_screen_corners_a_point_a_hundredth_of_a_pixel_off_its_edge_is_a_corner():
    """Added points lie on their edge's screen line to ~1e-13 px; a vertex
    0.01 px off it is the user's. (A 1 px tolerance took it for an added
    point.)"""
    dense = _dense(_TRI)
    j = len(dense) // 2
    d = _to_screen(dense)
    edge = d[j + 1] - d[j - 1]
    normal = np.array([-edge[1], edge[0]]) / np.hypot(*edge)
    nudged = dense.copy()
    nudged[j] = _from_screen(d[j] + 0.01 * normal)
    assert not screen_corners(dense, _to_screen)[j]
    assert screen_corners(nudged, _to_screen)[j]


def test_polyline_on_screen_keeps_both_ends_exactly():
    """A screen round trip moves 3e5 by ~1e-10; the chain's ends are the
    given points, not their round trips."""
    chain = [(1e3, 1e3), (3e4, 7777.7), (3e5, 2e3)]
    out = polyline_on_screen(chain, _to_screen, _from_screen)
    assert out[0].tolist() == list(chain[0])
    assert out[-1].tolist() == list(chain[-1])
    assert list(chain[1]) in out.tolist()
