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


# ── Known-wrong rules: each property check must catch one ────────────────────
#
# A property such as "winding does not matter" is also satisfied by a rule
# that answers False for every event. So each property test below is paired
# with a negative case: the same check, run on a rule known to be wrong in
# exactly that way, has to fail. That proves the check can fail; the positive
# case proves the real rule passes it.

def _matplotlib_rule(verts, pts):
    """The backend this replaced: winding-dependent and double-counting on
    a boundary (see _points_in_polygon's docstring)."""
    from matplotlib.path import Path
    return Path(np.asarray(verts, dtype=float)).contains_points(pts)


def _unclosed_rule(verts, pts):
    """An even-odd crossing test that forgets the edge from the last vertex
    back to the first, a classic polygon bug."""
    v = np.asarray(verts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    for (x0, y0), (x1, y1) in zip(v[:-1], v[1:], strict=True):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide='ignore', invalid='ignore'):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < xc)
    return inside


def _assert_anchored(rule, verts):
    """The rule answers the unambiguous points correctly, so a property
    holding for it is not the emptiness of a rule that answers nothing."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    far = np.array([lo - 5.0, hi + 5.0])
    assert not rule(verts, far).any()
    centre = np.array([[5., 3.]]) if verts is CONCAVE else verts.mean(axis=0)[None]
    assert rule(verts, centre).all()


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


def test_the_shared_edge_check_catches_a_double_counting_rule():
    """Negative case: matplotlib's rule puts shared-edge points in both
    gates, and the same count sees it."""
    left = np.array([[0., 0.], [5., 0.], [5., 10.], [0., 10.]])
    right = np.array([[5., 0.], [10., 0.], [10., 10.], [5., 10.]])
    on_edge = np.array([[5., y] for y in np.linspace(0.5, 9.5, 19)])
    both = _matplotlib_rule(left, on_edge) & _matplotlib_rule(right, on_edge)
    neither = ~_matplotlib_rule(left, on_edge) & ~_matplotlib_rule(right,
                                                                   on_edge)
    assert int(both.sum()) + int(neither.sum()) > 0


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
    _assert_anchored(_points_in_polygon, verts)
    ccw = _points_in_polygon(verts, pts)
    cw = _points_in_polygon(verts[::-1].copy(), pts)
    assert np.array_equal(ccw, cw), (
        f'{label}: drawing the gate the other way round changes '
        f'{int((ccw != cw).sum())} event(s)')


@pytest.mark.parametrize('verts', [SQUARE, DIAMOND, CONCAVE])
def test_the_winding_check_catches_a_winding_dependent_rule(verts):
    """Negative case: the same comparison flags matplotlib's rule, which
    answers differently on the boundary when the gate is drawn the other way."""
    pts = np.vstack([verts, (verts + np.roll(verts, 1, axis=0)) / 2,
                     np.array([[5., 5.], [-1., -1.], [11., 11.]])])
    _assert_anchored(_matplotlib_rule, verts)
    ccw = _matplotlib_rule(verts, pts)
    cw = _matplotlib_rule(verts[::-1].copy(), pts)
    assert not np.array_equal(ccw, cw)


@pytest.mark.parametrize('verts', [SQUARE, DIAMOND, CONCAVE])
def test_vertex_listing_order_does_not_change_the_answer(verts):
    """The same polygon listed from a different starting vertex is the same
    polygon."""
    pts = np.vstack([verts, (verts + np.roll(verts, 1, axis=0)) / 2,
                     np.array([[5., 5.]])])
    _assert_anchored(_points_in_polygon, verts)
    base = _points_in_polygon(verts, pts)
    for k in (1, 2, 3):
        assert np.array_equal(base, _points_in_polygon(
            np.roll(verts, k, axis=0), pts))


@pytest.mark.parametrize('verts', [SQUARE, DIAMOND, CONCAVE])
def test_the_order_check_catches_a_rule_that_forgets_the_closing_edge(verts):
    """Negative case: a crossing test that drops the closing edge answers
    differently depending on which vertex is listed first, and the same
    comparison sees it."""
    pts = np.vstack([verts, (verts + np.roll(verts, 1, axis=0)) / 2,
                     np.array([[5., 5.]])])
    base = _unclosed_rule(verts, pts)
    assert any(not np.array_equal(base, _unclosed_rule(
        np.roll(verts, k, axis=0), pts)) for k in (1, 2, 3))


def test_translation_does_not_change_the_answer():
    """Shifting a gate and its events together by a flow-scale offset must not
    move anyone across the boundary."""
    pts = np.vstack([SQUARE, np.array([[5., 5.], [5., 0.], [0., 5.]])])
    _assert_anchored(_points_in_polygon, SQUARE)
    base = _points_in_polygon(SQUARE, pts)
    moved = _points_in_polygon(SQUARE + 1e5, pts + 1e5)
    assert np.array_equal(base, moved)
    # Negative case: move only the gate and the same comparison must see
    # events change sides, or it could not have caught a shift either.
    assert not np.array_equal(base, _points_in_polygon(SQUARE + 1e5, pts))


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
