"""Spreading matrix for conventional compensation (Complaints 24).

The same Spillover Spread Matrix spectral unmixing reports (Nguyen et al.,
Cytometry A 2013;83:306), on compensated single-stain controls: for the
control of channel ``j`` split into its negative and positive events on
``j``, the spread it puts into channel ``i`` is

    SSᵢⱼ = sqrt(σ²pos(i) − σ²neg(i)) / sqrt(Fpos(j) − Fneg(j))

with σ the robust SD (84th − 50th percentile) and F the median, all on
compensated linear values. A control with no separate negative uses the
unstained control as its negative, else the spread's slope over bins of
its primary channel (spectral.spillover_spread_matrix does the arithmetic).

Pure numpy.
"""
from __future__ import annotations

import numpy as np


def primary_channel(values, channels, unstained=None, cofactor=150.0):
    """The channel a single-stain control is brightest in, from compensated
    linear ``values`` (events x channels): the largest rise of its 99.5th
    percentile on arcsinh(v / cofactor) above the unstained control's (or,
    without one, above the control's own 25th percentile, its negatives).
    Spread into other channels grows only as the root of the signal, so on
    this log-like scale it stays far below the stain. None for an empty
    control."""
    v = np.asarray(values, dtype=float)
    if v.ndim != 2 or v.shape[0] < 10 or v.shape[1] != len(channels):
        return None
    z = np.arcsinh(v / cofactor)
    hi = np.nanpercentile(z, 99.5, axis=0)
    if unstained is not None and np.asarray(unstained).shape[1] == v.shape[1]:
        lo = np.nanpercentile(np.arcsinh(np.asarray(unstained, dtype=float)
                                         / cofactor), 99.5, axis=0)
    else:
        lo = np.nanpercentile(z, 25.0, axis=0)
    score = hi - lo
    if not np.isfinite(score).any():
        return None
    return channels[int(np.nanargmax(score))]


def compensation_spread_matrix(controls, channels, unstained=None,
                               n_bins=8, min_bin=30):
    """``(SSM, channels)`` from compensated single-stain ``controls``
    (``{primary channel: events x channels array}`` of compensated linear
    values in ``channels`` order) and an optional ``unstained`` array.

    ``SSM[i, j]`` is the spread INTO channel ``i`` FROM the stain whose
    primary is ``j``; the diagonal is 0, and a channel without a control
    leaves its column NaN (NA, never 0)."""
    from .spectral import spillover_spread_matrix
    n = len(channels)
    identity = np.eye(n)        # compensated values are the abundances
    return spillover_spread_matrix(controls, identity, list(channels),
                                   n_bins=n_bins, min_bin=min_bin,
                                   unstained=unstained)


def worst_pairs(ssm, channels, top=5):
    """``[(from, into, spread)]``, the ``top`` largest finite spreads,
    largest first."""
    ssm = np.asarray(ssm, dtype=float)
    out = [(channels[j], channels[i], float(ssm[i, j]))
           for i in range(ssm.shape[0]) for j in range(ssm.shape[1])
           if i != j and np.isfinite(ssm[i, j])]
    out.sort(key=lambda r: -r[2])
    return out[:top]
