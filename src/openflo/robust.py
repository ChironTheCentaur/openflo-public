"""The median absolute deviation (MAD) and the robust SD built from it.

For a normal population the MAD is sigma * Phi^-1(3/4), so sigma is
MAD / Phi^-1(3/4) = 1.482602218505602 * MAD. The usual 1.4826 is that
value rounded; it read robust SDs about 1.5e-6 low.

A normal curve's full width at half maximum is 2 * sqrt(2 ln 2) * sigma
(about 2.3548 sigma); FWHM_TO_SD turns a measured peak width into an SD.

This is the one home for the MAD-based robust SD. FlowJo's percentile rSD
(gating.population_stats, batch_normalize.robust_spread) is a different
statistic: half the range between ONE_SD_PERCENTILES, defined here so both
use the same cut points. The Nguyen et al. spreading SD (spectral) lives with
its caller.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

MAD_TO_SD = 1.0 / NormalDist().inv_cdf(0.75)
FWHM_TO_SD = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
# The percentiles one SD either side of the mean of a normal population, as
# FlowJo rounds them: half the range between them is FlowJo's robust SD.
ONE_SD_PERCENTILES = (15.87, 84.13)


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


def stain_index(median_pos: float, median_neg: float, rsd_neg: float) -> float:
    """The Stain Index, ``(median_pos - median_neg) / (2 * rsd_neg)``
    (Maecker et al., Cytometry A 2004;62:169): how far a population sits
    above a negative, in units of the negative's robust spread. Signed: a
    population dimmer than its negative reads below 0. NaN when any input
    is missing or infinite, or the negative has no spread (``rsd_neg`` at
    or below 0), never an infinity."""
    vals = (median_pos, median_neg, rsd_neg)
    if not all(math.isfinite(v) for v in vals) or rsd_neg <= 0:
        return float('nan')
    return (median_pos - median_neg) / (2.0 * rsd_neg)
