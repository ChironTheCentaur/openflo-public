"""A corrupt event must not be placed at the end of the trajectory.

`_geodesic_pseudotime` fills unreachable cells with the MAXIMUM distance — a
deliberate, documented choice for a genuinely disconnected component. But an
event with a NaN or inf coordinate is also unreachable, so it inherited that
fill and was plotted at pseudotime 1.0: the terminus of the developmental
trajectory, indistinguishable from a real late population.

The fill for true disconnected components is unchanged. Only events whose
COORDINATES are non-finite are excluded, and they come back as NaN so they drop
out of trajectory plots and trend bins rather than inventing an endpoint.
"""
import numpy as np
import pytest

from openflo.trajectory import compute_pseudotime, pseudotime_trends


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(5, 1, (40, 4))])
    return X, X[:, 0].copy()


def test_clean_data_spans_the_full_range(data):
    X, score = data
    pt, root = compute_pseudotime(X, score)
    assert np.isfinite(pt).all()
    assert pt.min() == pytest.approx(0.0) and pt.max() == pytest.approx(1.0)
    assert 0 <= root < len(X)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_an_event_with_a_bad_coordinate_gets_no_pseudotime(data, bad):
    X, score = data
    Xb = X.copy()
    Xb[7, 2] = bad
    pt, _root = compute_pseudotime(Xb, score)
    assert np.isnan(pt[7]), (
        f'an event with a {bad} coordinate was given pseudotime {pt[7]} — at '
        'the trajectory terminus, as if it were the most differentiated cell')
    assert int(np.isnan(pt).sum()) == 1, 'the blast radius should be one event'


def test_the_good_events_are_unchanged_by_a_bad_neighbour(data):
    """Excluding bad rows must not move anyone else's pseudotime."""
    X, score = data
    clean, _ = compute_pseudotime(X, score)
    Xb = X.copy()
    Xb[7, 2] = np.nan
    dirty, _ = compute_pseudotime(Xb, score)
    keep = np.ones(len(X), dtype=bool)
    keep[7] = False
    # The graph loses one node, so allow a small shift, but the ordering and
    # the span must survive.
    assert np.nanmin(dirty[keep]) == pytest.approx(0.0)
    assert np.nanmax(dirty[keep]) == pytest.approx(1.0)
    assert np.corrcoef(clean[keep], dirty[keep])[0, 1] > 0.95


def test_the_root_is_never_a_corrupt_event(data):
    X, score = data
    Xb = X.copy()
    Xb[0, :] = np.nan            # the row robust_root would otherwise favour
    _pt, root = compute_pseudotime(Xb, score)
    assert np.isfinite(Xb[root]).all(), 'the trajectory was rooted on bad data'


def test_an_entirely_corrupt_input_yields_all_nan(data):
    _X, score = data
    pt, _root = compute_pseudotime(np.full((5, 3), np.nan), score[:5])
    assert pt.shape == (5,) and np.isnan(pt).all(), (
        'expected NaN, not a fabricated trajectory')


def test_a_truly_disconnected_component_still_takes_the_documented_fill():
    """The behaviour that is NOT changing: two far-apart clusters are a real
    finding, and the unreachable side keeps its max-distance fill."""
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (30, 3)),
                   rng.normal(500, 0.1, (30, 3))])
    pt, _root = compute_pseudotime(X, X[:, 0], n_neighbors=3)
    assert np.isfinite(pt).all(), (
        'a genuinely disconnected component should still receive a pseudotime '
        '— only bad coordinates become NaN')
    assert pt.max() == pytest.approx(1.0)


def test_an_unknown_pseudotime_does_not_contaminate_the_last_trend_bin():
    """The consumer half of the same bug, and the one that reaches a figure.

    `searchsorted` sorts NaN LAST, so every event with no pseudotime was clipped
    into the FINAL bin — dragging the terminal point of the published
    "expression vs pseudotime" curve toward whatever those events expressed.
    """
    pt = np.concatenate([np.linspace(0.0, 1.0, 10), [np.nan] * 3])
    markers = np.concatenate([np.ones(10), [1000.0] * 3])[:, None]

    _centers, means = pseudotime_trends(pt, markers, n_bins=5)

    assert means[-1, 0] == pytest.approx(1.0), (
        f'the last bin reads {means[-1, 0]:.1f} instead of 1.0 — events with '
        'no pseudotime were binned as if they were the most differentiated')
    assert np.allclose(means[:, 0], 1.0)


def test_an_empty_bin_is_still_nan_not_zero():
    """Guard the guard: excluding NaN must not turn empty bins into data."""
    pt = np.array([0.05, 0.06, 0.95])
    markers = np.array([[1.0], [1.0], [2.0]])
    _centers, means = pseudotime_trends(pt, markers, n_bins=5)
    assert means[0, 0] == pytest.approx(1.0)
    assert means[-1, 0] == pytest.approx(2.0)
    assert np.isnan(means[1:4, 0]).all(), 'an empty bin reported a value'


def test_trends_are_unchanged_when_every_pseudotime_is_known():
    pt = np.linspace(0.0, 1.0, 50)
    markers = np.linspace(0.0, 10.0, 50)[:, None]
    _centers, means = pseudotime_trends(pt, markers, n_bins=5)
    assert np.isfinite(means).all()
    assert np.all(np.diff(means[:, 0]) > 0), 'a rising marker stopped rising'


def test_an_unusable_root_channel_is_refused_not_guessed():
    """`robust_root` returned index 0 when the score had no finite values —
    silently rooting the trajectory on an arbitrary cell. The pseudotime that
    came back still looked like a trajectory, but its origin and direction
    were meaningless."""
    from openflo.trajectory import robust_root
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 3))
    with pytest.raises(ValueError, match='no finite values'):
        robust_root(X, np.full(30, np.nan))


def test_a_usable_root_channel_still_picks_a_medoid():
    """The guard must not disturb the normal path."""
    from openflo.trajectory import robust_root
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 3))
    score = X[:, 0]
    root = robust_root(X, score, high=True)
    assert 0 <= root < len(X)
    assert score[root] > np.median(score), 'root is not at the high end'


def test_a_partly_finite_root_channel_is_accepted():
    """Only a TOTAL absence of usable score values is refused."""
    from openflo.trajectory import robust_root
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 3))
    score = X[:, 0].copy()
    score[:35] = np.nan                     # only 5 usable cells
    root = robust_root(X, score, high=True)
    assert np.isfinite(score[root]), 'rooted on a cell with no score'
