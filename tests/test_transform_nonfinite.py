"""Non-finite values must not become real measurements.

The biexponential backends map every non-finite input to the BOTTOM of the
scale (-1.0). That is silent corruption twice over:

  * a maximally bright reading is rendered and gated as maximally DIM; and
  * a NaN becomes a FINITE coordinate, which defeats the `dropna` that
    protects the plot and the gate masks — the event then counts as a real,
    very negative measurement in every population, median and frequency.

`asinh` always propagated NaN correctly; `logicle` and `hyperlog` did not, so
the behaviour also differed by transform. These tests pin the whole family.
"""
import numpy as np
import pandas as pd
import pytest

from openflo.pipeline import transform_values

BIEXP = ('logicle', 'hyperlog')
ALL_METHODS = (*BIEXP, 'asinh')
# Real instrument range: FCS values are bounded well below 2**24.
REAL = np.array([0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 262_144.0, 1e6])


@pytest.mark.parametrize('method', ALL_METHODS)
def test_nan_stays_nan(method):
    out = transform_values(np.array([np.nan]), method=method)
    assert np.isnan(out[0]), (
        f'{method} turned NaN into {out[0]!r} — a finite coordinate that '
        'survives dropna and counts as a real event')


@pytest.mark.parametrize('method', BIEXP)
@pytest.mark.parametrize('bad', [np.inf, -np.inf])
def test_infinities_do_not_become_the_scale_minimum(method, bad):
    out = transform_values(np.array([bad]), method=method)
    assert not np.isfinite(out[0]), (
        f'{method} mapped {bad} to {out[0]:.4f}; a saturated reading would be '
        'gated as the dimmest event in the sample')


@pytest.mark.parametrize('method', ALL_METHODS)
def test_finite_values_are_untouched(method):
    """The guard must not perturb ordinary data — this is the load path for
    every fluorescence channel."""
    mixed = np.concatenate([REAL, [np.nan]])
    clean = transform_values(REAL, method=method)
    with_bad = transform_values(mixed, method=method)
    np.testing.assert_allclose(with_bad[:len(REAL)], clean, rtol=0, atol=0)


@pytest.mark.parametrize('method', ALL_METHODS)
def test_monotonic_across_the_instrument_range(method):
    out = transform_values(REAL, method=method)
    assert np.all(np.diff(out) > 0), f'{method} is not monotonic on real data'


@pytest.mark.parametrize('method', BIEXP)
def test_a_nan_event_is_still_droppable_after_transform(method):
    """The actual impact. `_get_df` calls `dropna` on the plotted channels; a
    NaN silently turned into -1.0 slips past it and is gated as a very
    negative event."""
    raw = pd.DataFrame({'FITC-A': [100.0, np.nan, 10_000.0],
                        'SSC-A': [1.0, 2.0, 3.0]})
    raw['FITC-A'] = transform_values(raw['FITC-A'].to_numpy(float),
                                     method=method)
    kept = raw.dropna(subset=['FITC-A'])
    assert len(kept) == 2, (
        f'{method}: the NaN event survived dropna and would be counted as a '
        'real measurement')


@pytest.mark.parametrize('method', ALL_METHODS)
def test_an_all_nan_channel_does_not_fabricate_data(method):
    out = transform_values(np.full(50, np.nan), method=method)
    assert np.isnan(out).all(), f'{method} invented {np.nanmin(out)} from NaN'


def test_empty_input_is_still_empty():
    for method in ALL_METHODS:
        assert transform_values(np.array([]), method=method).size == 0
