"""The median absolute deviation (MAD) and the robust SD built from it.

For a normal population the MAD is sigma * Phi^-1(3/4), so sigma is
MAD / Phi^-1(3/4) = 1.482602218505602 * MAD. The usual 1.4826 is that
value rounded; it read robust SDs about 1.5e-6 low.

A normal curve's full width at half maximum is 2 * sqrt(2 ln 2) * sigma
(about 2.3548 sigma); FWHM_TO_SD turns a measured peak width into an SD.

This is the one home for the MAD-based robust SD. FlowJo's percentile rSD
(gating.population_stats) and the Nguyen et al. spreading SD (spectral) are
different statistics and live with their callers.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

MAD_TO_SD = 1.0 / NormalDist().inv_cdf(0.75)
FWHM_TO_SD = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def mad(values, center: float | None = None) -> float:
    """Median absolute deviation of `values` from `center` (their median when
    not given), unscaled. No NaN filtering: callers pass finite values."""
    a = np.asarray(values, dtype=float)
    if center is None:
        center = np.median(a)
    return float(np.median(np.abs(a - center)))


def robust_sd(values) -> float:
    """MAD_TO_SD * MAD of the finite values: an outlier-resistant SD
    estimate. NaN when there are none."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float('nan')
    return MAD_TO_SD * mad(a)
