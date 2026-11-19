"""A log histogram must not throw away events it could have shown.

`hist_bin_edges` built its log floor as `max(lo, max(hi * 1e-6, 1e-12))`. The
clamp exists because a log axis cannot place a non-positive value — but taking
the max with `lo` also raised a perfectly good POSITIVE `lo` up to six decades
below `hi`. On a channel with a wide dynamic range that silently discarded
events the axis could display perfectly well: measured, an all-positive channel
spanning 1e-3 to 1e6 had its floor moved from 0.001 to 1.0, putting HALF the
events off the bottom of the histogram with no indication.

A second, smaller loss came from the round trip through `log10`: the outer
edges could land an ulp inside the requested range, dropping the single lowest
event for no reason a reader could guess.

What remains absent is only what genuinely cannot be drawn — non-positive
values on a log axis — and `unplottable_count` exists so the histogram can say
so rather than leaving the reader to notice.
"""
import numpy as np
import pytest

from openflo.plotmath import hist_bin_edges, unplottable_count

N = 100_000


@pytest.fixture
def wide_positive():
    """All positive, six decades of dynamic range — nothing a log axis cannot
    show."""
    rng = np.random.default_rng(1)
    return np.concatenate([rng.uniform(1e-3, 1.0, N // 2),
                           rng.uniform(1e3, 1e6, N // 2)])


@pytest.fixture
def with_negatives():
    """A compensated channel with a real negative population."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(-300.0, 600.0, N // 3),
                           rng.normal(500.0, 900.0, N // 3),
                           rng.normal(9_000.0, 3_000.0, N - 2 * (N // 3))])


def _edges(values, scale='log'):
    return hist_bin_edges(float(values.min()), float(values.max()), scale)


def _outside(values, edges):
    e = np.asarray(edges, dtype=float)
    v = np.asarray(values, dtype=float)
    return int(((v < e.min()) | (v > e.max())).sum())


def test_positive_data_is_not_pushed_off_the_bottom(wide_positive):
    """The headline: every event is displayable, so none may be dropped."""
    lost = _outside(wide_positive, _edges(wide_positive))
    assert lost == 0, (
        f'{lost} of {N} displayable events fell outside the log bin range')


def test_a_positive_lower_bound_is_used_as_it_stands(wide_positive):
    edges = _edges(wide_positive)
    assert edges[0] == pytest.approx(float(wide_positive.min()), rel=1e-12), (
        f'the floor was moved from {wide_positive.min():g} to {edges[0]:g}')


def test_the_extremes_are_inside_the_range(wide_positive):
    """A log10 round trip used to leave the outermost edge an ulp inside the
    requested range, silently dropping the lowest event."""
    edges = _edges(wide_positive)
    assert edges[0] <= float(wide_positive.min())
    assert edges[-1] >= float(wide_positive.max())


def test_only_the_non_positive_events_are_off_scale(with_negatives):
    """What a log axis genuinely cannot place, and nothing more."""
    lost = _outside(with_negatives, _edges(with_negatives))
    assert lost == int((with_negatives <= 0).sum()), (
        f'{lost} events off-scale, but only '
        f'{int((with_negatives <= 0).sum())} are non-positive')
    assert unplottable_count(with_negatives, 'log') == lost


def test_a_non_positive_floor_still_gets_a_synthetic_one(with_negatives):
    """The clamp must still apply where it is actually needed."""
    edges = _edges(with_negatives)
    assert edges[0] > 0
    assert edges[0] == pytest.approx(float(with_negatives.max()) * 1e-6)


def test_the_edges_are_usable_as_bins(wide_positive):
    for values in (wide_positive, np.array([1.0, 10.0, 100.0])):
        edges = _edges(values)
        assert len(edges) == 201
        assert np.all(np.diff(np.asarray(edges)) > 0), 'edges not increasing'


def test_linear_and_symlog_are_untouched(with_negatives):
    """The fix is confined to the log branch."""
    lo, hi = float(with_negatives.min()), float(with_negatives.max())
    for scale in ('linear', 'symlog'):
        edges = hist_bin_edges(lo, hi, scale)
        assert edges[0] == pytest.approx(lo)
        assert edges[-1] == pytest.approx(hi)
        assert _outside(with_negatives, edges) == 0
        assert unplottable_count(with_negatives, scale) == 0


def test_a_degenerate_range_falls_back_to_linear():
    edges = hist_bin_edges(5.0, 5.0, 'log')
    assert len(edges) == 201
    assert all(e == pytest.approx(5.0) for e in (edges[0], edges[-1]))


def test_unplottable_count_ignores_non_finite_values():
    """NaN is absent for a different reason and is reported elsewhere."""
    values = np.array([1.0, 10.0, np.nan, np.inf, -np.inf, 100.0])
    assert unplottable_count(values, 'log') == 0


def test_unplottable_count_counts_non_positives_on_a_log_axis():
    values = np.array([-5.0, 0.0, 1.0, 10.0])
    assert unplottable_count(values, 'log') == 2
    assert unplottable_count(values, 'linear') == 0
    assert unplottable_count(np.array([]), 'log') == 0
