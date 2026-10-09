"""Polygon gating must partition the plane, not merely be fast.

`_points_in_polygon` decides which cells are in a population, so its behaviour
ON a gate boundary matters. A point exactly on an edge is neither strictly
inside nor outside, which means the tie-break is a CHOICE — and for gating only
one choice is defensible.

These tests originally asserted "agrees with matplotlib". That turned out to
pin the wrong reference: matplotlib double-counts shared edges, is
winding-dependent, and is not even self-consistent between an axis-aligned
square and the same square rotated. The implementation now uses flowutils'
compiled `gating_c`, and these tests assert the PROPERTIES that make it right
rather than agreement with any particular library — so they stay meaningful if
the backend is ever swapped again.

Why it matters in practice: on continuous float data both rules agree exactly.
They diverge only when events land on exact coordinates, which is precisely
what integer `$DATATYPE I` channels produce — a class this project has already
been bitten by.
"""
import numpy as np
import pytest

from openflo.pipeline import _points_in_polygon

SQUARE = np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]])
DIAMOND = np.array([[5., 0.], [10., 5.], [5., 10.], [0., 5.]])
CONCAVE = np.array([[0., 0.], [10., 0.], [10., 10.], [5., 5.], [0., 10.]])


def test_adjacent_gates_partition_a_shared_edge():
    """THE property. Two gates meeting on a line must claim each point on it
    exactly once: twice and the event is counted in two populations, never and
    it vanishes from both. This is what quadrant gates depend on."""
    left = np.array([[0., 0.], [5., 0.], [5., 10.], [0., 10.]])
    right = np.array([[5., 0.], [10., 0.], [10., 10.], [5., 10.]])
    on_edge = np.array([[5., y] for y in np.linspace(0.5, 9.5, 19)])

    in_left = _points_in_polygon(left, on_edge)
    in_right = _points_in_polygon(right, on_edge)

    both = int((in_left & in_right).sum())
    neither = int((~in_left & ~in_right).sum())
    assert both == 0, f'{both} point(s) on the shared edge are in BOTH gates'
    assert neither == 0, f'{neither} point(s) on the shared edge are in NEITHER'


def test_quadrants_sum_to_the_whole_on_integer_data():
    """Four gates meeting at a point, over integer-valued events -- the case
    where boundary behaviour stops being academic. No event may be double
    counted, or a frequency can exceed 100%."""
    rng = np.random.RandomState(0)
    pts = rng.randint(0, 21, size=(20000, 2)).astype(float)
    quads = [
        np.array([[0., 0.], [10., 0.], [10., 10.], [0., 10.]]),
        np.array([[10., 0.], [20., 0.], [20., 10.], [10., 10.]]),
        np.array([[0., 10.], [10., 10.], [10., 20.], [0., 20.]]),
        np.array([[10., 10.], [20., 10.], [20., 20.], [10., 20.]]),
    ]
    counts = np.vstack([_points_in_polygon(q, pts) for q in quads]).sum(axis=0)
    assert int((counts >= 2).sum()) == 0, (
        f'{int((counts >= 2).sum())} event(s) counted in more than one '
        f'quadrant — frequencies would sum above 100%')

    # And the other direction, which is the bug this rule was introduced for.
    # Checking only for double-counting is silent about events VANISHING —
    # fully-open bounds put a boundary event in no quadrant at all, and
    # nothing reports the loss. Restricted to events strictly inside the
    # covered region, since the outer top/right rim is excluded by the same
    # half-open convention that makes the interior partition.
    interior = (pts[:, 0] < 20.) & (pts[:, 1] < 20.)
    assert int(interior.sum()) > 0, 'fixture covers no interior events'
    lost = int((counts[interior] == 0).sum())
    assert lost == 0, (
        f'{lost} event(s) fell in NO quadrant — they have vanished from every '
        f'population and no frequency reports the loss')


@pytest.mark.parametrize('verts, label', [
    (SQUARE, 'axis-aligned square'),
    (DIAMOND, 'rotated square'),
    (CONCAVE, 'concave polygon'),
])
def test_winding_direction_does_not_change_the_answer(verts, label):
    """Users draw gates clockwise and anticlockwise. The same shape must gate
    the same way either way, including on its boundary."""
    pts = np.vstack([verts,                                   # the vertices
                     (verts + np.roll(verts, 1, axis=0)) / 2,  # edge midpoints
                     np.array([[5., 5.], [-1., -1.], [11., 11.]])])
    ccw = _points_in_polygon(verts, pts)
    cw = _points_in_polygon(verts[::-1].copy(), pts)
    assert np.array_equal(ccw, cw), (
        f'{label}: drawing the gate the other way round changes '
        f'{int((ccw != cw).sum())} event(s)')


@pytest.mark.parametrize('verts', [SQUARE, DIAMOND, CONCAVE])
def test_vertex_listing_order_does_not_change_the_answer(verts):
    """The same polygon listed from a different starting vertex is the same
    polygon."""
    pts = np.vstack([verts, (verts + np.roll(verts, 1, axis=0)) / 2,
                     np.array([[5., 5.]])])
    base = _points_in_polygon(verts, pts)
    for k in (1, 2, 3):
        assert np.array_equal(base, _points_in_polygon(
            np.roll(verts, k, axis=0), pts))


def test_translation_does_not_change_the_answer():
    """Shifting a gate and its events together by a flow-scale offset must not
    move anyone across the boundary."""
    pts = np.vstack([SQUARE, np.array([[5., 5.], [5., 0.], [0., 5.]])])
    base = _points_in_polygon(SQUARE, pts)
    moved = _points_in_polygon(SQUARE + 1e5, pts + 1e5)
    assert np.array_equal(base, moved)


def test_interior_and_exterior_are_never_in_doubt():
    """Whatever the boundary rule, points clearly inside or outside must be
    unambiguous -- and must match the reference crossing test."""
    from matplotlib.path import Path
    rng = np.random.RandomState(7)
    for _ in range(20):
        k = rng.randint(3, 30)
        theta = np.sort(rng.uniform(0, 2 * np.pi, k))
        radius = rng.uniform(0.2, 1.0, k)
        verts = np.column_stack([radius * np.cos(theta),
                                 radius * np.sin(theta)])
        pts = rng.uniform(-1.3, 1.3, (4000, 2))
        got = _points_in_polygon(verts, pts)
        ref = Path(verts).contains_points(pts)
        # Random float points never land exactly on an edge, so the rules must
        # agree everywhere. Any difference here is a real bug, not a tie-break.
        assert np.array_equal(got, ref), (
            f'{int((got != ref).sum())} interior/exterior disagreement(s) — '
            f'this is not a boundary tie-break')


def test_float_data_is_unaffected_by_the_boundary_rule():
    """The reassurance that made adopting this safe: on continuous data the
    tie-break never fires, so ordinary FCS gating is bit-identical."""
    from matplotlib.path import Path
    rng = np.random.RandomState(0)
    pts = rng.uniform(0, 20, (200000, 2))
    assert np.array_equal(_points_in_polygon(SQUARE, pts),
                          Path(SQUARE).contains_points(pts))


def test_empty_input():
    got = _points_in_polygon(SQUARE, np.zeros((0, 2)))
    assert got.shape == (0,)
    assert got.dtype == bool


def test_a_gate_far_from_the_data_is_all_false():
    rng = np.random.RandomState(1)
    pts = rng.normal(50000, 10000, (5000, 2))
    assert not _points_in_polygon(SQUARE, pts).any()
