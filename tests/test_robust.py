"""The shared MAD and robust SD helpers in openflo.robust (BP1: one home for
each calculation). The former copies in voltage, pipeline, gating and
gaussian_cleanup now call these, so the checks below pin both the values and
that the copies are gone."""
import inspect

import numpy as np
import pytest

from openflo import gating, gaussian_cleanup, pipeline, spectral, voltage
from openflo.robust import MAD_TO_SD, mad, robust_sd


def test_mad_about_the_median_and_about_a_given_center():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])   # median 3, |dev| 2,1,0,1,97
    assert mad(v) == 1.0
    assert mad(v, 0.0) == 3.0                     # |v| median
    assert isinstance(mad(v), float)


def test_robust_sd_scales_the_mad_and_ignores_non_finite():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])     # MAD 1
    assert robust_sd(v) == MAD_TO_SD
    with_junk = np.concatenate([v, [np.nan, np.inf, -np.inf]])
    assert robust_sd(with_junk) == MAD_TO_SD
    assert np.isnan(robust_sd([]))
    assert np.isnan(robust_sd([np.nan, np.inf]))


def test_robust_sd_matches_the_former_inline_formula_exactly():
    rng = np.random.default_rng(7)
    for n in (5, 101, 4000):
        v = rng.lognormal(3.0, 0.8, n)
        v[::17] = np.nan
        a = v[np.isfinite(v)]
        old = MAD_TO_SD * float(np.median(np.abs(a - np.median(a))))
        assert robust_sd(v) == old                 # bit for bit, not approx


def test_former_copies_use_the_shared_helper():
    assert voltage.VoltageTitration.robust_sd(np.arange(5.0)) == MAD_TO_SD
    with pytest.warns(DeprecationWarning):
        assert pipeline.robust_sd is robust_sd
    assert not hasattr(pipeline, '_robust_sd')
    assert not hasattr(voltage, '_MAD_TO_SD')
    assert not hasattr(gaussian_cleanup, '_MAD_SD')
    for mod in (voltage, gaussian_cleanup, pipeline, gating):
        assert 'np.median(np.abs(' not in inspect.getsource(mod), mod.__name__


def test_spectral_percentile_spread_keeps_its_own_name():
    # Nguyen et al.'s P84.13 - P50 is a different statistic from MAD_TO_SD*MAD.
    assert not hasattr(spectral, '_robust_sd')
    v = np.arange(1.0, 10001.0)
    p84, p50 = np.percentile(v, [84.13, 50.0])
    assert spectral._spread_sd(v) == pytest.approx(float(p84 - p50))
