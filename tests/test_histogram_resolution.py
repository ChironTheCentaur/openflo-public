"""A comb is not a bimodal distribution.

`_bimodal_valley`, `auto_threshold` and `analyze_dna` all histogram a channel
into 256 bins and look for peaks. A channel cannot fill more bins than it has
distinct values, and an integer detector ($DATATYPE I) whose bulk spans a few
tens of ADC steps therefore produces a COMB: most bins are structurally empty.

The smoothing is measured in BINS rather than data units, so a sigma of 2
cannot close a gap several bins wide. Peak finding then reads two adjacent
comb teeth as two modes with a zero between them — and a bin of zero trivially
satisfies the "valley must dip below half the shorter peak" test that
`_bimodal_valley` relies on to refuse unimodal data.

Measured: an all-live, unimodal viability channel of 43 integer levels
produced a valley at its own median, and the auto-clean viability filter
deleted 46% of the sample (27,677 of 60,000 live cells). The identical data
unrounded correctly returned no valley. It was also phase-dependent — 69
levels behaved, 43 and 51 did not — so whether half a sample survived depended
on how the histogram bins happened to line up.

The bin count is now capped at the number of distinct values in range.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import (
    _autoclean_viability_mask,
    _bimodal_valley,
    auto_threshold,
)

N = 60_000


def _integers(mu, sd, n=N, seed=0):
    return np.round(np.random.default_rng(seed).normal(mu, sd, n))


@pytest.mark.parametrize('mu, sd', [(25, 5), (40, 6), (40, 8), (100, 12)])
def test_a_unimodal_integer_channel_has_no_valley(mu, sd):
    values = _integers(mu, sd)
    assert _bimodal_valley(values) is None, (
        f'a unimodal channel of {len(np.unique(values))} integer levels was '
        'reported as bimodal')


def test_the_same_data_unrounded_also_has_no_valley():
    """The invariant behind it: rounding must not create a mode."""
    rng = np.random.default_rng(0)
    assert _bimodal_valley(rng.normal(25.0, 5.0, N)) is None


def test_a_genuinely_bimodal_integer_channel_is_still_split():
    """The guard must not make the detector blind to real structure."""
    rng = np.random.default_rng(1)
    values = np.round(np.concatenate([rng.normal(20.0, 4.0, N // 2),
                                      rng.normal(120.0, 10.0, N // 2)]))
    valley = _bimodal_valley(values)
    assert valley is not None, 'a clearly bimodal channel was called unimodal'
    assert 20.0 < valley < 120.0, (
        f'the valley landed at {valley}, outside the two modes')


def test_a_genuinely_bimodal_continuous_channel_is_still_split():
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.normal(20.0, 4.0, N // 2),
                             rng.normal(120.0, 10.0, N // 2)])
    valley = _bimodal_valley(values)
    assert valley is not None
    assert 20.0 < valley < 120.0


@pytest.mark.parametrize('seed, mu, sd', [
    # Whether the old code deleted half the sample was decided by bin PHASE:
    # the viability filter only cuts when the valley sits above the median,
    # and the comb's false valley landed either side of it depending on the
    # data. These four all landed above it — 46% of an all-live sample was
    # removed. Seeds that landed below it passed, which is why this has to be
    # parametrised rather than trusted to one draw.
    pytest.param(1, 25, 5, id='seed1-25-5'),
    pytest.param(2, 40, 6, id='seed2-40-6'),
    pytest.param(5, 30, 7, id='seed5-30-7'),
    pytest.param(11, 25, 5, id='seed11-25-5'),
])
def test_an_all_live_sample_keeps_every_event(seed, mu, sd):
    """End to end, through the filter that was deleting the events."""
    df = pd.DataFrame({'Zombie Aqua-A': _integers(mu, sd, seed=seed),
                       'FSC-A': np.random.default_rng(3).normal(
                           60_000.0, 8_000.0, N)})
    mask = np.asarray(_autoclean_viability_mask(df, {}), dtype=bool)
    assert int(mask.sum()) == N, (
        f'{N - int(mask.sum())} of {N} live cells were removed from a sample '
        'with no dead population at all')


def test_auto_threshold_is_not_led_by_the_comb():
    """auto_threshold falls back to Otsu, so it always returns a number — but
    it was taking the comb's false valley instead. Rounding the data must not
    move the threshold far."""
    rng = np.random.default_rng(4)
    continuous = rng.normal(25.0, 5.0, N)
    assert auto_threshold(np.round(continuous)) == pytest.approx(
        auto_threshold(continuous), abs=1.0)


def test_a_channel_with_almost_no_levels_is_handled():
    """The cap has a floor of 8 bins, so a near-constant channel still
    histograms rather than producing an empty one."""
    assert _bimodal_valley(np.full(N, 7.0)) is None      # no range at all

    # Three levels is far fewer than the floor. Whatever it decides, it must
    # decide it without raising, and any valley must lie inside the data.
    valley = _bimodal_valley(np.repeat([1.0, 2.0, 3.0], N // 3))
    assert valley is None or 1.0 <= valley <= 3.0
