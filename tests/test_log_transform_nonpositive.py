"""A non-positive value has no logarithm, and a log value is not a raw one.

Two faults in the `log` transform, in opposite directions.

FORWARD: `np.where(v > 0, log10(clip(v, 1e-6, None)), 0.0)` folded every
non-positive value onto exactly 0.0 — which on this scale means raw intensity
1.0, so the DIMMEST events landed above every genuinely positive event below
1.0. Negative values are routine in compensated data. Measured on 100,000
events, a third of them non-positive: a gate at `log > -1.0` selected 99,995
where 67,375 is correct, calling 32,620 negative events positive. The
artificial spike of a third of the sample at one coordinate is also what
`auto_threshold` and `_bimodal_valley` lock onto.

INVERSE: `np.where(v > 0, 10**v, 0.0)` confused the transformed value with the
raw one. A log-scaled value is negative for every raw intensity below 1.0 —
log10(0.5) is -0.301 — so all of those were sent to 0.0 and a round trip
destroyed them. That path runs whenever a channel's scale is changed, which
does inverse-then-forward.

The forward transform now returns NaN for non-positive input, matching what
the logicle/hyperlog branch already does for input it cannot place, so `dropna`
keeps those events out of gates and medians instead of inventing a coordinate.
The inverse is `10**v` over the whole real line.
"""
import logging

import numpy as np
import pytest

from openflo.pipeline import inverse_transform_values, transform_values

ROUND_TRIP = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 100.0, 5_000.0])


@pytest.fixture
def compensated():
    """A compensated channel with a real negative population."""
    rng = np.random.default_rng(0)
    n = 30_000
    return np.concatenate([rng.normal(-300.0, 600.0, n // 3),
                           rng.normal(500.0, 900.0, n // 3),
                           rng.normal(9_000.0, 3_000.0, n - 2 * (n // 3))])


def test_non_positive_values_have_no_logarithm(compensated):
    out = transform_values(compensated, method='log')
    non_positive = compensated <= 0
    assert non_positive.any(), 'the fixture no longer has negative events'
    assert np.isnan(out[non_positive]).all(), (
        'non-positive values were given a finite log coordinate; the largest '
        f'was {np.nanmax(out[non_positive])}')


def test_positive_values_are_untouched(compensated):
    out = transform_values(compensated, method='log')
    positive = compensated > 0
    assert np.allclose(
        out[positive],
        np.log10(np.clip(compensated[positive], 1e-6, None)))


def test_a_gate_on_a_log_channel_selects_the_right_events(compensated):
    """The harm, stated as the user sees it."""
    out = transform_values(compensated, method='log')
    for threshold in (-1.0, -0.001, 1.0):
        selected = int(np.nansum(out > threshold))
        correct = int((compensated > 10.0 ** threshold).sum())
        assert selected == correct, (
            f'gate "log > {threshold}" selected {selected} events; '
            f'{correct} are actually above 10^{threshold}')


def test_the_dimmest_events_are_not_ranked_above_the_faintest_positives():
    """The specific inversion: -3000 used to outrank +0.5."""
    out = transform_values(np.array([-3_000.0, 0.5]), method='log')
    assert np.isnan(out[0])
    assert out[1] < 0.0


def test_the_round_trip_is_exact():
    """Every raw value at or below 1.0 used to come back as 0.0."""
    back = inverse_transform_values(
        transform_values(ROUND_TRIP, method='log'), method='log')
    assert np.allclose(back, ROUND_TRIP), (
        f'round trip changed the data: {ROUND_TRIP} -> {back}')


def test_the_inverse_maps_negative_log_values_to_dim_intensities():
    """A log value below zero is an intensity below 1.0, not a missing one."""
    assert inverse_transform_values(
        np.array([-0.301029995]), method='log')[0] == pytest.approx(0.5)
    assert inverse_transform_values(
        np.array([-2.0]), method='log')[0] == pytest.approx(0.01)


def test_nan_survives_the_inverse():
    out = inverse_transform_values(np.array([np.nan, 1.0]), method='log')
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(10.0)


def test_the_user_is_told_how_many_events_have_no_log(caplog, compensated):
    with caplog.at_level(logging.WARNING):
        transform_values(compensated, method='log')
    assert any('no logarithm' in r.getMessage() for r in caplog.records), (
        'a third of the channel became NaN with nothing said'
    )


def test_an_all_positive_channel_warns_about_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        transform_values(np.array([1.0, 10.0, 100.0]), method='log')
    assert not [r for r in caplog.records if 'no logarithm' in r.getMessage()]
