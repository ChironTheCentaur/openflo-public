"""The constant that turns a median absolute deviation into an SD estimate.

For a normal population the MAD is sigma * Phi^-1(3/4), so sigma is
MAD / Phi^-1(3/4) = 1.482602218505602 * MAD. The usual 1.4826 is that
value rounded; it read robust SDs about 1.5e-6 low.
"""
from __future__ import annotations

from statistics import NormalDist

MAD_TO_SD = 1.0 / NormalDist().inv_cdf(0.75)
