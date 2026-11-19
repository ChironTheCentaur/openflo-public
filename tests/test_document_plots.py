"""openflo.document.Document's plot calls beyond the 2-D density: the 1-D
histogram, and the gate outlines (``overlays``) a front end draws over
either plot. Generated FCS files only; channels FSC-A, SSC-A, FL1-A ...,
sample s1. No Tk.
"""
from __future__ import annotations

import numpy as np
import pytest

from openflo import polygon_np
from openflo.document import Document, DocumentError
from tests.test_document import CELLS, gated_doc, spill_fcs


def _stored(doc, name, ch):
    return doc._samples[name].data.data[ch].to_numpy(dtype=float)


# ── histogram ───────────────────────────────────────────────────────────────

def test_histogram_counts_the_population_in_bins(tmp_path):
    doc, name, _cells, pos, _mid = gated_doc(tmp_path)
    h = doc.histogram(name, 'FL2-A', gate=pos, bins=50)
    assert h['counts'].dtype == np.float32 and h['counts'].shape == (50,)
    assert h['edges'].dtype == np.float64 and h['edges'].shape == (51,)
    rows = {r['id']: r for r in doc.stats(name, channels=[])}
    assert h['events'] == h['binned'] == rows[pos]['count']
    assert h['counts'].sum() == rows[pos]['count']
    assert h['unknown'] is None
    # The axis spans the whole sample, so a population is drawn where its
    # parent is.
    v = _stored(doc, name, 'FL2-A')
    assert h['range'] == [float(v.min()), float(v.max())]
    want, _ = np.histogram(v[doc._population(doc._samples[name], pos)[0]],
                           bins=50, range=h['range'])
    np.testing.assert_array_equal(h['counts'], want)


def test_histogram_of_all_events_on_a_display_scale(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    h = doc.histogram(name, 'FSC-A', scale='log', bins=16)
    view = doc.to_view(name, 'FSC-A', _stored(doc, name, 'FSC-A'), 'log')
    assert h['events'] == 3000
    assert h['range'] == pytest.approx([float(view.min()), float(view.max())])
    want, edges = np.histogram(view, bins=16, range=h['range'])
    np.testing.assert_array_equal(h['counts'], want)
    np.testing.assert_allclose(h['edges'], edges)


def test_histogram_takes_a_range_and_refuses_bad_requests(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    h = doc.histogram(name, 'FL1-A', bins=4, x_range=[0.2, 0.6])
    assert h['range'] == [0.2, 0.6] and h['binned'] < h['events']
    for kwargs, msg in (({'bins': 0}, 'bins'), ({'bins': 5000}, 'bins'),
                        ({'x_range': [1, 0]}, 'range'),
                        ({'gate': 'nope'}, 'no gate')):
        with pytest.raises(DocumentError, match=msg):
            doc.histogram(name, 'FL1-A', **kwargs)
    with pytest.raises(DocumentError, match='no channel'):
        doc.histogram(name, 'FL9-A')


# ── overlays ────────────────────────────────────────────────────────────────

def test_a_rect_is_drawn_on_its_plot_either_way_round(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    (o,) = doc.overlays(name, 'FSC-A', 'SSC-A')
    assert (o['id'], o['name'], o['kind'], o['shape']) == (
        cells, 'cells', 'rect', 'polygon')
    pts = o['points']
    assert pts.dtype == np.float64 and pts.shape == (4 * 16 + 1, 2)
    assert (pts[0] == pts[-1]).all()
    corners = {(3e4, 1e4), (7e4, 1e4), (7e4, 5e4), (3e4, 5e4)}
    assert corners <= {tuple(p) for p in pts}
    # Every point lies on the rectangle's boundary.
    on_x = np.isin(pts[:, 0], [3e4, 7e4])
    on_y = np.isin(pts[:, 1], [1e4, 5e4])
    assert (on_x | on_y).all()
    (flip,) = doc.overlays(name, 'SSC-A', 'FSC-A')
    np.testing.assert_array_equal(flip['points'], pts[:, ::-1])
    assert doc.overlays(name, 'FSC-A', 'FL1-A') == []


def test_the_drawn_outline_holds_the_gates_events(tmp_path):
    """Events inside the outline as drawn on the display axes are the
    gate's events: the outline lies where the region is."""
    doc, name, cells, *_ = gated_doc(tmp_path)
    poly = doc.add_gate(name, {
        'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'vertices': [[0.3, 0.25], [0.58, 0.28], [0.5, 0.48], [0.36, 0.42]],
        'name': 'blob'}, cells)['id']
    count = {r['id']: r['count'] for r in doc.stats(name, channels=[])}[poly]
    cell_mask = doc._population(doc._samples[name], cells)[0]
    # Its edges cut the dense middle of the data; on the linear view they
    # bend, and an outline of the corners alone would miss ~10% of events.
    for scale in (None, 'logicle', 'asinh:c=150', 'linear'):
        got = [o for o in doc.overlays(name, 'FL1-A', 'FL2-A', parent=cells,
                                       x_scale=scale, y_scale=scale)
               if o['id'] == poly]
        assert len(got) == 1 and got[0]['shape'] == 'polygon'
        xs = doc.to_view(name, 'FL1-A', _stored(doc, name, 'FL1-A'), scale)
        ys = doc.to_view(name, 'FL2-A', _stored(doc, name, 'FL2-A'), scale)
        inside = polygon_np.points_inside(got[0]['points'],
                                          np.column_stack([xs, ys]))
        n = int((inside & cell_mask).sum())
        # Same events, give or take the few on a bent edge's chord.
        assert abs(n - count) <= 3, (scale, n, count)
    assert 500 < count < 2000


def test_1d_gates_are_bands_on_the_axis_of_their_channel(tmp_path):
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    got = {o['id']: o for o in doc.overlays(name, 'FL1-A', 'FL2-A',
                                            parent=cells, x_scale='logicle')}
    assert set(got) == {pos, mid}
    v = float(doc.to_view(name, 'FL1-A', [0.5], 'logicle')[0])
    assert got[pos] == {'id': pos, 'name': 'FL1+', 'kind': 'threshold',
                        'shape': 'band', 'axis': 'x', 'lo': v, 'hi': None}
    assert got[mid] == {'id': mid, 'name': 'FL2 mid', 'kind': 'interval',
                        'shape': 'band', 'axis': 'y', 'lo': 0.3, 'hi': 0.6}
    # On an axis other than the stored one, the band moves with it.
    lin = {o['id']: o for o in doc.overlays(name, 'FL1-A', 'FL2-A',
                                            parent=cells, x_scale='linear')}
    want = float(doc.to_view(name, 'FL1-A', [0.5], 'linear')[0])
    assert want > 10 and lin[pos]['lo'] == want
    # A histogram (no y channel) draws the 1-D gates on its own channel only.
    (h,) = doc.overlays(name, 'FL2-A', parent=cells)
    assert h['id'] == mid and h['axis'] == 'x'
    # Root gates only, unless a parent is named.
    assert [o['id'] for o in doc.overlays(name, 'FL1-A', 'FL2-A')] == []


def test_an_open_side_is_drawn_just_past_the_data(tmp_path):
    from openflo.scales import OPEN
    doc, name, *_ = gated_doc(tmp_path)
    gid = doc.add_gate(name, {'kind': 'rect', 'x_channel': 'FSC-A',
                              'y_channel': 'SSC-A', 'x0': 4e4, 'x1': OPEN,
                              'y0': -OPEN, 'y1': 2e4, 'name': 'open'})['id']
    band = doc.add_gate(name, {'kind': 'interval', 'channel': 'FSC-A',
                               'lo': -OPEN, 'hi': 6e4, 'name': 'low'})['id']
    got = {o['id']: o for o in doc.overlays(name, 'FSC-A', 'SSC-A')}
    pts = got[gid]['points']
    fsc, ssc = _stored(doc, name, 'FSC-A'), _stored(doc, name, 'SSC-A')
    assert pts[:, 0].max() > fsc.max() and pts[:, 0].max() < 1e7
    assert pts[:, 1].min() < ssc.min() and pts[:, 1].min() > -1e7
    assert got[band]['lo'] is None and got[band]['hi'] == 6e4


def test_gates_with_no_region_on_the_axes_are_left_out(tmp_path):
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    doc.add_gate(name, {'kind': 'boolean', 'op': 'and',
                        'operands': [pos, mid], 'name': 'both'}, cells)
    ell = doc.add_gate(name, {
        'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'mean': [0.5, 0.5], 'cov': [[0.01, 0.0], [0.0, 0.02]],
        'distance_sq': 4.0, 'name': 'oval'}, cells)['id']
    bad = doc.add_gate(name, {
        'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
        'mean': [0.5, 0.5], 'cov': [[0.0, 0.0], [0.0, 0.0]],
        'distance_sq': 4.0, 'name': 'flat'}, cells)['id']
    got = {o['id']: o for o in doc.overlays(name, 'FL1-A', 'FL2-A',
                                            parent=cells)}
    assert set(got) == {pos, mid, ell}
    assert bad not in got
    # An ellipse's outline circles its mean.
    pts = got[ell]['points']
    r = np.hypot((pts[:, 0] - 0.5) / 0.1, (pts[:, 1] - 0.5) / np.sqrt(0.02))
    assert 1.9 < r.min() and r.max() < 2.2


def test_overlays_refuse_unknown_names(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    with pytest.raises(DocumentError, match='no channel'):
        doc.overlays(name, 'FSC-A', 'FL9-A')
    with pytest.raises(DocumentError, match='no gate'):
        doc.overlays(name, 'FSC-A', 'SSC-A', parent='nope')


def test_a_sample_without_gates_has_no_overlays(tmp_path):
    doc = Document()
    name = doc.open_fcs(spill_fcs(tmp_path / 's1.fcs'))
    assert doc.overlays(name, 'FSC-A', 'SSC-A') == []
    assert CELLS['kind'] == 'rect'


# ── quadrants ───────────────────────────────────────────────────────────────

def test_a_quadrant_is_four_open_rects_that_split_the_parent(tmp_path):
    from openflo.scales import OPEN
    doc, name, cells, *_ = gated_doc(tmp_path)
    ids = doc.add_quadrant(name, 'FL1-A', 'FL2-A', 0.45, 0.35, cells)
    assert len(ids) == 4
    gates = [doc.gate(name, g) for g in ids]
    assert [g['name'] for g in gates] == ['Q++ (x>, y>)', 'Q+- (x>, y<)',
                                          'Q-+ (x<, y>)', 'Q-- (x<, y<)']
    assert {g['quad_set'] for g in gates} == {'qs1'}
    assert all(g['parent_id'] == cells for g in gates)
    assert [(g['x0'], g['x1'], g['y0'], g['y1']) for g in gates] == [
        (0.45, OPEN, 0.35, OPEN), (0.45, OPEN, -OPEN, 0.35),
        (-OPEN, 0.45, 0.35, OPEN), (-OPEN, 0.45, -OPEN, 0.35)]
    rows = {r['id']: r['count'] for r in doc.stats(name, channels=[])}
    assert sum(rows[g] for g in ids) == rows[cells]
    df = doc._samples[name].data.data
    cm = doc._population(doc._samples[name], cells)[0]
    want = int((cm & (df['FL1-A'] >= 0.45) & (df['FL2-A'] >= 0.35)).sum())
    assert rows[ids[0]] == want
    # Drawn as a set: each outline names it.
    drawn = doc.overlays(name, 'FL1-A', 'FL2-A', parent=cells)
    assert [o.get('quad_set') for o in drawn if o['id'] in ids] == ['qs1'] * 4
    assert all('quad_set' not in o for o in drawn if o['id'] not in ids)
    # A second quadrant gets its own set; the sample's other sets count.
    assert {doc.gate(name, g)['quad_set'] for g in
            doc.add_quadrant(name, 'FSC-A', 'SSC-A', 5e4, 3e4)} == {'qs2'}


def test_a_quadrant_refuses_what_it_cannot_place(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    with pytest.raises(DocumentError, match='no channel'):
        doc.add_quadrant(name, 'FL1-A', 'FL9-A', 0, 0)
    with pytest.raises(DocumentError, match='finite'):
        doc.add_quadrant(name, 'FL1-A', 'FL2-A', float('nan'), 0)
    with pytest.raises(DocumentError, match='no gate'):
        doc.add_quadrant(name, 'FL1-A', 'FL2-A', 0, 0, 'nope')


def test_a_quadrants_sides_come_from_its_region_not_its_name(tmp_path):
    doc, name, cells, *_ = gated_doc(tmp_path)
    ids = doc.add_quadrant(name, 'FL1-A', 'FL2-A', 0.45, 0.35, cells)
    # A plain box on the same axes: not a quadrant's, so no sides.
    doc.add_gate(name, {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
                        'x0': 0.1, 'x1': 0.2, 'y0': 0.1, 'y1': 0.2, 'name': 'box'}, cells)
    want = {ids[0]: (1, 1), ids[1]: (1, -1), ids[2]: (-1, 1), ids[3]: (-1, -1)}
    # Renamed so no name says where it lies (and two of them swapped).
    for gid, nm in zip(ids, ['d', 'c', 'b', 'a'], strict=True):
        doc.update_gate(name, gid, {**doc.gate(name, gid), 'name': nm})

    def sides(x, y):
        return {o['id']: (o['quad_x'], o['quad_y'])
                for o in doc.overlays(name, x, y, parent=cells)
                if o['id'] in ids}
    assert sides('FL1-A', 'FL2-A') == want
    # Axes swapped: each side follows its channel.
    assert sides('FL2-A', 'FL1-A') == {g: (b, a) for g, (a, b) in want.items()}
    # Not on its two channels: no sides (and no other gate gets them).
    assert sides('FL1-A', 'FL3-A') == {}
    assert all('quad_x' not in o and 'quad_y' not in o
               for o in doc.overlays(name, 'FL1-A', 'FL2-A', parent=cells)
               if o['id'] not in ids)


def test_a_quadrants_side_without_its_crossing_comes_from_its_open_bound():
    from openflo.document import _quad_side
    from openflo.scales import OPEN
    assert _quad_side({'x0': 0.4, 'x1': OPEN}, 'x') == 1
    assert _quad_side({'x0': OPEN, 'x1': 0.4}, 'x') == 1     # either order
    assert _quad_side({'y0': -OPEN, 'y1': 0.4}, 'y') == -1
    assert _quad_side({'y0': 0.4, 'y1': -OPEN}, 'y') == -1
    # The crossing wins over the bounds when it is kept.
    assert _quad_side({'x0': -OPEN, 'x1': 0.4, 'quad_origin_x': 0.4}, 'x') == -1
    assert _quad_side({'x0': 0.4, 'x1': OPEN, 'quad_origin_x': 0.4}, 'x') == 1
    assert _quad_side({'x0': 0.2, 'x1': 0.3, 'quad_origin_x': 0.4}, 'x') == -1
    assert _quad_side({'x0': 0.2, 'x1': 0.3, 'quad_origin_x': float('nan')},
                      'x') == 1


def test_the_same_population_on_another_sample(tmp_path):
    doc, s1, cells, pos, mid = gated_doc(tmp_path)
    s2 = doc.open_fcs(spill_fcs(tmp_path / 'b' / 's1.fcs'))
    s3 = doc.open_fcs(spill_fcs(tmp_path / 'c' / 's1.fcs'))
    out = doc.copy_gates(s1, [s2])
    by_name = {doc.gate(s2, g)['name']: g for g in out[s2]['ids']}
    assert doc.same_population(s1, pos, s2) == by_name['FL1+']
    assert doc.same_population(s1, mid, s2) == by_name['FL2 mid']
    assert doc.same_population(s1, cells, s2) == by_name[doc.gate(s1, cells)['name']]
    # Back the other way, and to itself.
    assert doc.same_population(s2, by_name['FL1+'], s1) == pos
    assert doc.same_population(s1, pos, s1) == pos
    # A sample without it: None.
    assert doc.same_population(s1, pos, s3) is None
    # The same name under another parent is another population.
    top = doc.add_gate(s3, {'kind': 'threshold', 'channel': 'FL1-A',
                            'value': 0.5, 'name': 'FL1+'})['id']
    assert doc.same_population(s1, pos, s3) is None
    assert doc.same_population(s3, top, s1) is None
    with pytest.raises(DocumentError, match='no gate'):
        doc.same_population(s1, 'nope', s2)
    with pytest.raises(DocumentError):
        doc.same_population(s1, pos, 'nobody')
