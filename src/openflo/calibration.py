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


def fit_mesf_calibration(mfi, known, model='linear'):
    """Least-squares calibration from bead peak MFIs and their assigned
    MESF/ABC values. Returns ``{slope, intercept, r2, n, model}``. Raises
    ``ValueError`` for < 2 usable pairs.

    ``model='linear'``: ``known = slope·MFI + intercept``.
    ``model='loglog'``: ``log10 known = slope·log10 MFI + intercept``, the
    standard for bead sets (Schwartz et al.; NIST/ISAC): the peaks span
    decades, and a linear fit is decided by the brightest one, so a dim
    peak's small error leaves the low end off by a large factor. A slope
    near 1 says the detector is linear. Only positive pairs are used.
    ``r2`` (on the fitted scale) flags a bad bead assignment."""
    if model not in ('linear', 'loglog'):
        raise ValueError(f"Unknown calibration model {model!r}: use "
                         "'linear' or 'loglog'.")
    mfi = np.asarray(mfi, dtype=float)
    known = np.asarray(known, dtype=float)
    m = np.isfinite(mfi) & np.isfinite(known)
    if model == 'loglog':
        m &= (mfi > 0) & (known > 0)
    mfi, known = mfi[m], known[m]
    if model == 'loglog':
        mfi, known = np.log10(mfi), np.log10(known)
    if mfi.size < 2:
        raise ValueError("need at least 2 (MFI, value) peak pairs")
    A = np.vstack([mfi, np.ones_like(mfi)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, known, rcond=None)
    pred = slope * mfi + intercept
    ss_res = float(np.sum((known - pred) ** 2))
    ss_tot = float(np.sum((known - known.mean()) ** 2))
    # r2 is UNDEFINED when the assigned values have no spread (every bead given
    # the same MESF — a data-entry mistake). Reporting 1.0 there, as this did,
    # is the strongest possible "this calibration is good" signal attached to a
    # fit whose slope is ~0 and which turns every converted value into the
    # intercept. NaN is the honest answer, and it is what makes the degenerate
    # case visible in the dialog and in any `r2 > threshold` check.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'slope': float(slope), 'intercept': float(intercept),
            'r2': float(r2), 'n': int(mfi.size), 'model': model}


def apply_calibration(values, slope, intercept, clip=True, model='linear'):
    """Convert raw intensities to calibrated units: ``slope·v + intercept``
    (linear) or ``10**intercept · v**slope`` (loglog, see
    fit_mesf_calibration). Negatives are clipped to 0 by default (MESF/ABC
    are non-negative); unclipped, a log-log value at or below 0 MFI has no
    calibrated value and reads NaN."""
    v = np.asarray(values, dtype=float)
    if model == 'loglog':
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(v > 0, 10.0 ** intercept * np.abs(v) ** slope,
                           0.0 if clip else np.nan)
        return out
    out = slope * v + intercept
    return np.clip(out, 0.0, None) if clip else out


def _positive(name, value):
    """``float(value)``, raising ``ValueError`` unless it is finite and > 0."""
    v = float(value)
    if not (np.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return v


def absolute_count_per_uL(cell_events, bead_events, bead_concentration_per_uL,
                          bead_volume_uL=None, sample_volume_uL=None,
                          dilution_factor=1.0):
    """Absolute cell concentration (cells/µL of the ORIGINAL sample) from
    liquid counting beads of known concentration (CountBright / Flow-Count
    style), the standard formula:

    ``cells/µL = (cell_events / bead_events)
                 · bead_concentration_per_uL · bead_volume_uL / sample_volume_uL
                 · dilution_factor``

    ``bead_concentration_per_uL · bead_volume_uL`` is the number of beads in
    the tube; dividing by ``sample_volume_uL`` puts the count back in the
    sample's own volume, and ``dilution_factor`` undoes a dilution of the
    sample made before the beads were added (10 for 1 part in 10).

    The volumes are NOT optional in practice: 50 µL of beads at 1000/µL in
    100 µL of blood, 10,000 cell and 5,000 bead events, is 1,000 cells/µL;
    the old ``cells/beads · concentration`` reported 2,000. With neither
    volume given they are taken as equal, which is that old formula; give
    both or neither.

    Raises ``ValueError`` if ``bead_events == 0``, a volume or the dilution
    factor is not positive, or only one volume is given."""
    bead_events = float(bead_events)
    if bead_events == 0:
        raise ValueError("bead_events must be non-zero to compute a ratio")
    if (bead_volume_uL is None) != (sample_volume_uL is None):
        raise ValueError("give both bead_volume_uL and sample_volume_uL, or "
                         "neither (equal volumes)")
    ratio = 1.0
    if bead_volume_uL is not None:
        ratio = (_positive('bead_volume_uL', bead_volume_uL)
                 / _positive('sample_volume_uL', sample_volume_uL))
    return ((float(cell_events) / bead_events) * float(bead_concentration_per_uL)
            * ratio * _positive('dilution_factor', dilution_factor))


def absolute_count_from_known_beads(cell_events, bead_events, beads_added,
                                    sample_volume_uL, dilution_factor=1.0):
    """Absolute cell concentration (cells/µL of the original sample) from a
    *known number of beads added* to a *known sample volume* (the
    lyophilised-pellet variant, e.g. Trucount: beads per tube from the
    pouch label).

    ``cells/µL = (cell_events / bead_events) · beads_added / sample_volume_uL
    · dilution_factor``

    Raises ``ValueError`` if ``bead_events == 0`` or ``sample_volume_uL`` or
    ``dilution_factor`` is not positive."""
    bead_events = float(bead_events)
    if bead_events == 0:
        raise ValueError("bead_events must be non-zero to compute a ratio")
    return ((float(cell_events) / bead_events) * float(beads_added)
            / _positive('sample_volume_uL', sample_volume_uL)
            * _positive('dilution_factor', dilution_factor))


def total_cells(cell_events, bead_events, beads_added):
    """Estimated total cells in the tube from a known number of beads added:

    ``total = (cell_events / bead_events) · beads_added``

    Raises ``ValueError`` if ``bead_events == 0``."""
    bead_events = float(bead_events)
    if bead_events == 0:
        raise ValueError("bead_events must be non-zero to compute a ratio")
    return (float(cell_events) / bead_events) * float(beads_added)
