"""A polygon put on another scale stays the polygon that was drawn.

`wsp_gate_coords.map_gate_coords` moves a gate between scales: stored logicle
to FlowJo's linear units for a .wsp export, back again on import, and onto
new transform parameters when a session reopens. A straight edge on one scale
is a curve on the other, so it is traced. Cutting every edge into a fixed 16
pieces moved a logicle polygon enough that events changed sides in an export
and re-import (79 of 39,495 on generated data). These tests pin the bound on
how far a traced edge may stray, where it is measured, and that a round trip
gives back the drawn polygon. Generated data and gates only (FL1-A, FL2-A).
"""
from __future__ import annotations

import numpy as np
import pytest

from openflo import gate_eval, wsp_gate_coords
from openflo.channel_transforms import transform_spec, transform_values
from openflo.document import Document
from openflo.wsp_gate_coords import gate_scale_fns, map_gate_coords
from tests.test_document import spill_fcs

# The most a traced edge may stray, as a fraction of the polygon's extent.
# A literal, not wsp_gate_coords.POLYGON_EDGE_TOL, so loosening the code's
# tolerance fails here. Twice the code's 3e-6: the code probes each piece at
# three points, these tests measure all along it.
BOUND = 6e-6
OPEN = 1e12
LOGICLE = {'FL1-A': transform_spec('logicle'), 'FL2-A': transform_spec('logicle')}
# Where linear 0 sits on the default logicle scale.
ZERO = float(transform_values(np.array([0.0]))[0])

QUAD = [[0.35, 0.2], [0.8, 0.25], [0.7, 0.75], [0.3, 0.6]]
LOW_TRIANGLE = [[0.1, 0.1], [0.6, 0.15], [0.2, 0.55]]
THIN = [[0.2, 0.2], [0.85, 0.8], [0.8, 0.85], [0.15, 0.25]]
# The first edge runs through linear 0 on both axes, symmetric about it,
# twice as steep in y: its linear image bends both ways and crosses the
# straight chord at its middle, so a single probe at the middle sees no
# stray at all.
ACROSS_ZERO = [[ZERO - 0.08, ZERO - 0.16], [ZERO + 0.08, ZERO + 0.16],
               [ZERO - 0.3, ZERO + 0.3]]


def _poly(verts):
    return {'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
            'vertices': verts}


def _stray_on_drawn_scale(drawn, linear_verts, per_piece=200):
    """Largest distance, in units of the drawn polygon's extent, between the
    exported polygon (straight pieces in linear units, as FlowJo evaluates
    it) taken back to logicle, and the drawn polygon's edges."""
    d = np.asarray(drawn, float)
    scale = np.ptp(d, axis=0)
    lin = np.asarray(linear_verts, float)
    s = np.linspace(0.0, 1.0, per_piece)[:, None]
    dense = np.concatenate([lin[i] + (lin[(i + 1) % len(lin)] - lin[i]) * s
                            for i in range(len(lin))])
    back = np.column_stack([transform_values(dense[:, 0]),
                            transform_values(dense[:, 1])]) / scale
    edges = [(d[i] / scale, d[(i + 1) % len(d)] / scale) for i in range(len(d))]

    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
        return np.hypot(*(p - (a + t[:, None] * ab)).T)
    return float(np.min([seg_dist(back, a, b) for a, b in edges], axis=0).max())


@pytest.mark.parametrize('verts', [QUAD, LOW_TRIANGLE, THIN, ACROSS_ZERO],
                         ids=['quad', 'low triangle', 'thin', 'across zero'])
def test_an_exported_polygon_strays_no_further_than_the_tolerance(verts):
    # Measured on the logicle scale it was drawn on, all along every
    # exported piece, not only at the points the tracer probes.
    out = map_gate_coords(_poly(verts), gate_scale_fns(LOGICLE, 'to_linear'))
    assert _stray_on_drawn_scale(verts, out['vertices']) <= BOUND


def test_a_narrow_polygon_is_held_to_its_own_width():
    # The tolerance is a fraction of the polygon's extent on each axis, so a
    # gate 0.02 wide in x is traced as closely, for its width, as a wide one.
    # Measured in the scale's own units it would stray ten times the bound.
    verts = [[0.3, 0.15], [0.32, 0.2], [0.312, 0.8], [0.296, 0.6]]
    out = map_gate_coords(_poly(verts), gate_scale_fns(LOGICLE, 'to_linear'))
    assert _stray_on_drawn_scale(verts, out['vertices']) <= BOUND


def test_a_fixed_16_pieces_would_have_strayed_far_more():
    # The bound above is not met by accident: the master-branch tracing
    # (16 straight pieces in logicle, mapped) strays 50x further on QUAD.
    v = np.asarray(QUAD, float)
    t = np.arange(16) / 16
    pts = np.concatenate([v[i] + (v[(i + 1) % 4] - v[i]) * t[:, None]
                          for i in range(4)])
    fixed = map_gate_coords(_poly(pts.tolist()), {})['vertices']
    lin = [[float(wsp_gate_coords.inverse_transform_values(np.array([x]))[0]),
            float(wsp_gate_coords.inverse_transform_values(np.array([y]))[0])]
           for x, y in fixed]
    assert _stray_on_drawn_scale(QUAD, lin) > 50 * BOUND


@pytest.mark.parametrize('verts', [QUAD, LOW_TRIANGLE, THIN, ACROSS_ZERO],
                         ids=['quad', 'low triangle', 'thin', 'across zero'])
def test_a_round_trip_gives_back_the_drawn_vertices_every_time(verts):
    g = _poly(verts)
    to_lin = gate_scale_fns(LOGICLE, 'to_linear')
    from_lin = gate_scale_fns(LOGICLE, 'from_linear')
    for _cycle in range(5):
        lin = map_gate_coords(g, to_lin)
        assert len(lin['vertices']) > len(verts)        # traced in linear
        g = map_gate_coords(lin, from_lin)
        assert len(g['vertices']) == len(verts)
        np.testing.assert_allclose(g['vertices'], verts, rtol=0, atol=1e-9)


def test_an_import_traces_linear_edges_onto_logicle_within_the_tolerance():
    # The other direction: a polygon drawn straight in linear units (as
    # FlowJo keeps one on a linear axis) put on logicle storage. Measured on
    # logicle, where it lands, all along each traced piece.
    lin = [[50.0, 50.0], [150000.0, 300.0], [300.0, 150000.0]]
    out = np.asarray(map_gate_coords(
        _poly(lin), gate_scale_fns(LOGICLE, 'from_linear'))['vertices'])
    scale = np.ptp(out, axis=0)
    a = np.asarray(lin, float)
    s = np.linspace(0, 1, 4000)[:, None]
    worst = 0.0
    for i in range(3):
        curve = a[i] + (a[(i + 1) % 3] - a[i]) * s
        curve = np.column_stack([transform_values(curve[:, 0]),
                                 transform_values(curve[:, 1])]) / scale
        poly = out / scale
        segs = [(poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))]
        d = np.min([np.hypot(*(curve - (p + np.clip(((curve - p) @ (q - p))
                                                      / ((q - p) @ (q - p)), 0, 1)[:, None]
                                        * (q - p))).T) for p, q in segs], axis=0)
        worst = max(worst, float(d.max()))
    assert worst <= BOUND
    # Measured on logicle, where it lands: about 800 points. Measured on
    # the linear scale it came from (the wrong one here) it over-cut to
    # over 11,000.
    assert len(out) < 3000


def test_wsp_export_and_reimport_changes_no_event(tmp_path):
    # The whole path, both apps' code: Document.export_wsp, then
    # Document.import_wsp, on generated files, compensation on and off.
    shapes = {'quad': QUAD, 'low': LOW_TRIANGLE, 'thin': THIN}
    changed = total = 0
    for seed in (1, 2, 3, 4):
        for comp in (True, False):
            doc = Document()
            s = doc.open_fcs(spill_fcs(tmp_path / f'r{seed}{comp}' / 'b.fcs',
                                       seed=seed))
            for nm, v in shapes.items():
                doc.add_gate(s, {**_poly(v), 'name': nm})
            if not comp:
                doc.set_compensation(s, False)
            out = tmp_path / f'w{seed}{comp}.wsp'
            doc.export_wsp(str(out))
            back = Document()
            back.import_wsp(str(out))
            data = doc._get(s).data.data[['FL1-A', 'FL2-A']].to_numpy(float)
            for node in back.gate_tree(s):
                vb = back.gate(s, node['id'])['vertices']
                drawn = shapes[node['name']]
                assert len(vb) == len(drawn)
                m0 = gate_eval.points_in_polygon(np.asarray(drawn), data)
                m1 = gate_eval.points_in_polygon(np.asarray(vb), data)
                total += int(m0.sum())
                changed += int((m0 != m1).sum())
    assert total > 10000
    assert changed == 0


def test_an_edge_whose_image_leaves_the_scale_is_still_cut():
    # log of a value <= 0 has no image: such an edge cannot be measured and
    # falls back to POLYGON_EDGE_STEPS pieces, the open sentinel where the
    # log is undefined, as before.
    def log10(v):
        with np.errstate(all='ignore'):
            return np.log10(np.asarray(v, dtype=float))
    g = _poly([[-10.0, 1.0], [100.0, 1.0], [100.0, 100.0]])
    v = map_gate_coords(g, {'FL1-A': log10})['vertices']
    assert len(v) >= 3 + (wsp_gate_coords.POLYGON_EDGE_STEPS - 1)


def test_a_vertex_that_turns_back_along_its_line_is_kept():
    # Only a vertex BETWEEN its neighbours is redundant. One the drawing
    # turned back at, though on the same line, is a corner the user drew.
    verts = [[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [2.0, 3.0]]
    out = map_gate_coords(_poly(verts), {'FL1-A': lambda v: np.asarray(v) * 2.0})
    assert out['vertices'] == [[0.0, 0.0], [8.0, 0.0], [4.0, 0.0], [4.0, 3.0]]


def test_a_cut_point_whose_image_is_off_scale_is_left_out():
    # A map with no image at one value (here exactly where the first cut
    # lands) must not put the open sentinel into the traced edge.
    def cube_with_a_hole(v):
        v = np.asarray(v, dtype=float)
        return np.where(v == 2.5, np.nan, v ** 3)
    g = _poly([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    v = np.asarray(map_gate_coords(g, {'FL1-A': cube_with_a_hole})['vertices'])
    assert len(v) >= 3 and np.all(np.abs(v) < OPEN)


def test_a_vertex_behind_its_previous_neighbour_is_kept():
    # The other side of the turn-back rule: (0, 0) lies on the line from
    # (2, 0) to (4, 0) but behind (2, 0), so it is a corner, not a point
    # between its neighbours.
    verts = [[2.0, 3.0], [2.0, 0.0], [0.0, 0.0], [4.0, 0.0]]
    out = map_gate_coords(_poly(verts), {'FL1-A': lambda v: np.asarray(v) * 2.0})
    assert out['vertices'] == [[4.0, 3.0], [4.0, 0.0], [0.0, 0.0], [8.0, 0.0]]


def _dest_stray(out, verts, fx, fy, per_edge=2000):
    """Largest distance, in units of the traced ring's extent, from the drawn
    edges mapped point by point to the traced ring (on the scale it lands)."""
    out = np.asarray(out, float)
    live = np.all(np.abs(out) < OPEN, axis=1)
    scale = np.ptp(out[live], axis=0)
    v = np.asarray(verts, float)
    s = np.linspace(0, 1, per_edge)[:, None]
    worst = 0.0
    segs = [(out[k] / scale, out[(k + 1) % len(out)] / scale)
            for k in range(len(out))
            if live[k] and live[(k + 1) % len(out)]]
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        if not np.all(np.abs(np.concatenate([a, b])) < OPEN):
            continue
        e = a + (b - a) * s
        curve = np.column_stack([fx(e[:, 0]), fy(e[:, 1])]) / scale
        d = np.min([np.hypot(*(curve - (p + np.clip(((curve - p) @ (q - p))
                                                      / ((q - p) @ (q - p)), 0, 1)[:, None]
                                        * (q - p))).T) for p, q in segs], axis=0)
        worst = max(worst, float(d.max()))
    return worst


def test_one_axis_without_a_way_back_is_measured_where_it_lands():
    # x can be taken back to the drawn scale, y cannot: the stray is then
    # measured where the polygon lands, on both axes alike, not half on
    # each scale (which over-cuts every curved edge to the limit).
    def double(v):
        return np.asarray(v, dtype=float) * 2.0
    double.to_drawn = lambda v: np.asarray(v, dtype=float) / 2.0

    def cube(v):
        return np.asarray(v, dtype=float) ** 3
    verts = [[1.0, 1.0], [4.0, 1.0], [1.0, 4.0]]
    out = map_gate_coords(_poly(verts), {'FL1-A': double, 'FL2-A': cube})['vertices']
    # Cut at random to the depth limit (half on each scale) it ran to
    # thousands of points.
    assert len(out) < 1000
    assert _dest_stray(out, verts, double, cube) <= BOUND


def test_an_open_vertex_does_not_loosen_the_tolerance():
    # The polygon's extent is taken over its live vertices only: with the
    # 1e12 sentinel in it, the curved live edge would never be cut.
    def ident(v):
        return np.asarray(v, dtype=float)

    def cube(v):
        return np.asarray(v, dtype=float) ** 3
    verts = [[1.0, 1.0], [OPEN, 1.0], [4.0, 1.0], [1.0, 4.0]]
    out = map_gate_coords(_poly(verts), {'FL1-A': cube, 'FL2-A': ident})['vertices']
    assert len(out) > 4
    assert _dest_stray(out, verts, cube, ident) <= BOUND


def test_a_tiny_polygon_keeps_a_corner_that_is_small_in_absolute_units():
    # Collinearity is judged against the polygon's own size: a corner
    # 1e-12 off its neighbours' line is a millionth of this polygon, a real
    # corner, though tiny next to 1.
    verts = [[0.0, 0.0], [5e-7, 1e-12], [1e-6, 0.0], [1e-6, 1e-6], [0.0, 1e-6]]
    out = map_gate_coords(_poly(verts), {'FL1-A': lambda v: np.asarray(v) * 2.0})
    assert len(out['vertices']) == 5


def test_the_corners_come_back_with_gpu_acceleration_on(monkeypatch):
    # With GPU acceleration on, logicle runs through a lookup table good to
    # about 1e-8 of scale; the round trip must still give back the corners.
    from openflo import gpu_accel
    monkeypatch.setattr(gpu_accel, 'enabled', lambda: True)
    monkeypatch.setattr(gpu_accel, 'interp',
                        lambda x, xp, fp: np.interp(np.asarray(x), xp, fp))
    g = _poly(QUAD)
    to_lin = gate_scale_fns(LOGICLE, 'to_linear')
    from_lin = gate_scale_fns(LOGICLE, 'from_linear')
    for _cycle in range(3):
        g = map_gate_coords(map_gate_coords(g, to_lin), from_lin)
        assert len(g['vertices']) == len(QUAD)
        np.testing.assert_allclose(g['vertices'], QUAD, rtol=0, atol=1e-6)


def test_an_axis_of_unknown_scale_is_measured_where_it_was_drawn_too():
    # The .wsp export matches an unknown-scale channel to its FCS values by
    # rank (rank_matched_fn). That map can be taken back, so a polygon with
    # one such axis is still measured on the scale it was drawn on.
    from openflo.wsp_gate_coords import rank_matched_fn
    rng = np.random.default_rng(3)
    linear = np.concatenate([rng.normal(0, 300, 4000),
                             10 ** rng.uniform(1, 5.3, 16000)])
    stored = transform_values(linear)
    rank = rank_matched_fn(stored, linear)
    probe = np.linspace(0.1, 0.9, 50)
    np.testing.assert_allclose(rank.to_drawn(rank(probe)), probe, atol=1e-9)
    # Past the data at both ends too, where the map carries on along its
    # end slopes.
    stored_lo, stored_hi = float(stored.min()), float(stored.max())
    tails = np.array([stored_lo - 0.2, stored_lo - 0.05,
                      stored_hi + 0.05, stored_hi + 0.2])
    np.testing.assert_allclose(rank.to_drawn(rank(tails)), tails, atol=1e-9)
    fns = dict(gate_scale_fns({'FL2-A': LOGICLE['FL2-A']}, 'to_linear'),
               **{'FL1-A': rank})
    out = map_gate_coords(_poly(QUAD), fns)['vertices']
    back = np.column_stack([rank.to_drawn(np.asarray(out)[:, 0]),
                            transform_values(np.asarray(out)[:, 1])])
    # The same check as _stray_on_drawn_scale, with x taken back by rank.
    v = np.asarray(QUAD, float)
    scale = np.ptp(v, axis=0)
    s = np.linspace(0, 1, 200)[:, None]
    lin = np.asarray(out, float)
    dense = np.concatenate([lin[i] + (lin[(i + 1) % len(lin)] - lin[i]) * s
                            for i in range(len(lin))])
    dense = np.column_stack([rank.to_drawn(dense[:, 0]),
                             transform_values(dense[:, 1])]) / scale
    edges = [(v[i] / scale, v[(i + 1) % 4] / scale) for i in range(4)]
    d = np.min([np.hypot(*(dense - (a + np.clip(((dense - a) @ (b - a))
                                                  / ((b - a) @ (b - a)), 0, 1)[:, None]
                                    * (b - a))).T) for a, b in edges], axis=0)
    assert back.shape[0] == len(out)
    assert float(d.max()) <= BOUND
