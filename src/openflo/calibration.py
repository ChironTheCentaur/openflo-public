"""Fluorescence-intensity calibration to standardized units.

Calibration beads carry populations of *known* fluorescence — MESF (Molecules
of Equivalent Soluble Fluorochrome) or ABC (Antibodies Bound per Cell). Running
the beads, finding their peaks, and regressing the assigned known values on the
measured intensity gives a per-channel ``value = slope·MFI + intercept`` map
that converts raw intensities into comparable, instrument-independent units
(the fluorescence analogue of the existing FSC→µm bead-size calibration).

Pure numpy / scipy / sklearn (all core dependencies).
"""
from __future__ import annotations

import numpy as np


def detect_bead_peaks(values, n_peaks=6, seed=42):
    """Find the ``n_peaks`` bead-population peak intensities in a 1-D array by
    k-means on ``log10`` intensity (robust to the wide spacing of bead peaks).
    Returns the per-cluster **median** MFIs, ascending. Falls back to evenly
    spaced percentiles if k-means can't separate them."""
    from sklearn.cluster import KMeans
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < n_peaks:
        return np.array(sorted(v))
    lv = np.log10(v).reshape(-1, 1)
    try:
        km = KMeans(n_clusters=n_peaks, n_init='auto', random_state=seed).fit(lv)
        peaks = sorted(float(np.median(v[km.labels_ == c]))
                       for c in range(n_peaks))
        return np.array(peaks)
    except Exception:
        return np.percentile(v, np.linspace(2, 98, n_peaks))


def fit_mesf_calibration(mfi, known):
    """Least-squares calibration line ``known = slope·MFI + intercept`` from
    bead peak MFIs and their assigned MESF/ABC values. Returns
    ``{slope, intercept, r2, n}``. Raises ``ValueError`` for < 2 usable pairs.

    MESF/ABC scales are linear in MFI on a compensated linear axis, so a
    straight-line fit is the standard; ``r2`` flags a bad bead assignment."""
    mfi = np.asarray(mfi, dtype=float)
    known = np.asarray(known, dtype=float)
    m = np.isfinite(mfi) & np.isfinite(known)
    mfi, known = mfi[m], known[m]
    if mfi.size < 2:
        raise ValueError("need at least 2 (MFI, value) peak pairs")
    A = np.vstack([mfi, np.ones_like(mfi)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, known, rcond=None)
    pred = slope * mfi + intercept
    ss_res = float(np.sum((known - pred) ** 2))
    ss_tot = float(np.sum((known - known.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {'slope': float(slope), 'intercept': float(intercept),
            'r2': float(r2), 'n': int(mfi.size)}


def apply_calibration(values, slope, intercept, clip=True):
    """Convert raw intensities to calibrated units (``slope·v + intercept``);
    negatives clipped to 0 by default (MESF/ABC are non-negative)."""
    out = slope * np.asarray(values, dtype=float) + intercept
    return np.clip(out, 0.0, None) if clip else out
