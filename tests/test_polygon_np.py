"""The numpy point-in-polygon (`openflo.polygon_np`) gives FlowUtils' answer
on every point, the boundary included.

`gate_eval.points_in_polygon` uses it wherever FlowUtils' compiled extension
cannot be built, so a gate must select the same events either way. Boundary
points are where two implementations of "inside" differ, so the parity tests
use an integer lattice that puts many points exactly on edges and vertices of
axis-aligned, rotated, concave and self-crossing polygons, each listed in
both winding directions; then 200,000 random points (half of them integers).

The half-open invariant (two polygons sharing an edge claim each point of it
exactly once) needs no FlowUtils, so it holds on the platforms the port is
for.
"""

import numpy as np
import pytest

from openflo.polygon_np import points_inside, winding_counts
from tests.conftest import import_flowutils_unless_browser

SQUARE = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
RECT = np.array([[1., -2.], [7., -2.], [7., 9.], [1., 9.]])
DIAMOND = np.array([[5., 0.], [10., 5.], [5., 10.], [0., 5.]])
TILTED = np.array([[2., 0.], [10., 3.], [7., 11.], [-1., 8.]])
CONCAVE = np.array([[0., 0.], [10., 0.], [10., 10.], [5., 5.], [0., 10.]])
BOWTIE = np.array([[0., 0.], [10., 10.], [10., 0.], [0., 10.]])
STAR = np.array([[5., 0.], [8., 10.], [0., 3.], [10., 3.], [2., 10.]])
SHAPES = {'square': SQUARE, 'rect': RECT, 'diamond': DIAMOND,
          'tilted': TILTED, 'concave': CONCAVE, 'bowtie': BOWTIE,
          'star': STAR}


def _lattice(lo=-3, hi=13, step=0.5):
    g = np.arange(lo, hi + step, step)
    xx, yy = np.meshgrid(g, g)
    return np.column_stack([xx.ravel(), yy.ravel()])


@pytest.fixture(scope='module')
def fu_gating():
    return import_flowutils_unless_browser('gating')


@pytest.mark.parametrize('winding', ['ccw', 'cw'])
@pytest.mark.parametrize('shape', sorted(SHAPES))
def test_lattice_matches_flowutils(fu_gating, shape, winding):
    verts = SHAPES[shape] if winding == 'ccw' else SHAPES[shape][::-1].copy()
    pts = _lattice()
    want = fu_gating.points_in_polygon(verts, pts)
    got = points_inside(verts, pts)
    assert want.any() and not want.all()
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('shape', sorted(SHAPES))
def test_winding_counts_match_flowutils(fu_gating, shape):
    from flowutils import gating_c
    verts = SHAPES[shape]
    pts = _lattice()
    want = gating_c.points_in_polygon(verts, len(verts), pts, len(pts))
    np.testing.assert_array_equal(winding_counts(verts, pts), want)


def test_random_points_match_flowutils(fu_gating):
    rng = np.random.default_rng(20261104)
    n = 200_000
    pts = rng.uniform(-2.0, 12.0, size=(n, 2))
    pts[: n // 2] = np.round(pts[: n // 2])            # integer events
    mismatches = 0
    for verts in SHAPES.values():
        want = fu_gating.points_in_polygon(verts, pts)
        mismatches += int((points_inside(verts, pts) != want).sum())
    angles = np.sort(rng.uniform(0, 2 * np.pi, 12))
    radii = rng.uniform(2.0, 6.0, 12)
    blob = np.column_stack([5 + radii * np.cos(angles),
                            5 + 1.5 * radii * np.sin(angles)])
    want = fu_gating.points_in_polygon(blob, pts)
    mismatches += int((points_inside(blob, pts) != want).sum())
    assert mismatches == 0


# ── Without FlowUtils ────────────────────────────────────────────────────────

# Pairs of polygons that share an edge: side by side, stacked, and two
# triangles splitting a square along its diagonal.
SHARED_EDGE_PAIRS = {
    'vertical': (np.array([[0., 0.], [5., 0.], [5., 10.], [0., 10.]]),
                 np.array([[5., 0.], [10., 0.], [10., 10.], [5., 10.]])),
    'horizontal': (np.array([[0., 0.], [10., 0.], [10., 4.], [0., 4.]]),
                   np.array([[0., 4.], [10., 4.], [10., 10.], [0., 10.]])),
    'diagonal': (np.array([[0., 0.], [10., 0.], [10., 10.]]),
                 np.array([[0., 0.], [10., 10.], [0., 10.]])),
}


@pytest.mark.parametrize('winding', ['ccw', 'cw'])
@pytest.mark.parametrize('pair', sorted(SHARED_EDGE_PAIRS))
def test_shared_edge_points_are_counted_exactly_once(pair, winding):
    """Every lattice point of the union's interior, the shared edge
    included, is in exactly one of the two polygons, and no point anywhere
    is in both."""
    first, second = SHARED_EDGE_PAIRS[pair]
    if winding == 'cw':
        first, second = first[::-1].copy(), second[::-1].copy()
    pts = _lattice(-2, 12, 1.0)
    counts = (points_inside(first, pts).astype(int)
              + points_inside(second, pts).astype(int))
    x, y = pts[:, 0], pts[:, 1]
    strictly_inside_union = (x > 0) & (x < 10) & (y > 0) & (y < 10)
    assert counts.max() == 1
    assert (counts[strictly_inside_union] == 1).all()
    assert (counts[(x < 0) | (x > 10) | (y < 0) | (y > 10)] == 0).all()


@pytest.mark.parametrize('shape', ['square', 'rect', 'tilted'])
def test_winding_direction_does_not_change_the_answer(shape):
    verts = SHAPES[shape]
    pts = _lattice()
    np.testing.assert_array_equal(points_inside(verts, pts),
                                  points_inside(verts[::-1].copy(), pts))


def test_axis_aligned_square_is_half_open():
    """A lower or left edge is inside, an upper or right edge outside."""
    pts = np.array([[0., 5.], [5., 0.], [10., 5.], [5., 10.], [0., 0.],
                    [5., 5.], [10., 10.], [-1e-9, 5.]])
    np.testing.assert_array_equal(
        points_inside(SQUARE, pts),
        [True, True, False, False, True, True, False, False])


def test_degenerate_inputs():
    assert points_inside(SQUARE, np.zeros((0, 2))).shape == (0,)
    assert not points_inside(SQUARE, [[np.nan, 5.], [5., np.nan]]).any()
    assert winding_counts(SQUARE, [[5., 5.]]).dtype == np.int32
