"""Compensation matrix optimizer: ``optimize_compensation`` estimates a
spillover matrix from single-stain controls.

Core module: numpy, pandas and the standard library. No Tk, and it does not
import ``openflo.pipeline``.
"""

import logging
import os

import numpy as np
import pandas as pd

from . import fcs_read as _fcs_read
from .acquisition_qc import AcquisitionQC
from .fcs_quirks import read_fcs_tolerant

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


# ── Compensation matrix optimizer ─────────────────────────────────────────────
#
# Single-stain estimate, as the FlowJo / Diva compensation wizards make it:
# split each control's primary channel into its negative and positive
# populations and take, for every other detector,
#
#     spill = (median_pos(dst) - median_neg(dst)) / (median_pos(src) - median_neg(src))
#
# Diagonals stay 1.0; rows are the source fluor, columns the detector.
#
# Not a least-squares slope of dst on src fitted within the brightest 5% of
# src, which is biased two ways (measured, true 0.20):
# * Errors in variables. Within a tight bead peak most of the src spread is
#   noise, which flattens the slope: 0.158 (and 0.124 with 90% positive
#   beads). The medians use the whole distance from the negative to the
#   positive population, where the noise is negligible: 0.1998.
# * Saturation. Events piled at the detector ceiling sit at the top of the
#   tail with their src clipped and dst not: 0.274 with 0.9% of events
#   saturated, 0.566 with 2.8%, and 0 with 9.9%. They are dropped first.
# The negative population is the WHOLE of it, not the dimmest tail: with
# autofluorescence correlated across detectors, the dimmest events by src are
# also dim in dst, which read a true 0.010 as 0.020 (bottom 5%) or 0.016
# (bottom 25%). Taken whole, 0.0105. The split is spectral's
# _split_negative_positive, which the spillover spread matrix uses too.


def _warn_skipped(src_ch, why):
    """A single stain the optimizer could not use. Its matrix row stays at
    identity, so the user is told at WARNING (which the app's log pane shows),
    not at INFO (which it hides)."""
    log.warning("[optimize] %s: skipped, %s. Its spillover row is left "
                "uncompensated (identity).", src_ch, why)


def optimize_compensation(channels, single_stain_paths,
                          positive_percentile=95.0, min_events=100):
    """Estimate an NxN spillover matrix from per-channel single-stain controls.

    Parameters
    ----------
    channels : list[str]
        Fluor channel names in the order the matrix rows / columns will use.
    single_stain_paths : dict[str, str]
        Maps each source channel name to an FCS file in which only that
        fluor is bright. Channels with no entry contribute their identity
        row only.
    positive_percentile : float
        Used only for a control whose primary channel shows no separate
        negative and positive populations: the events above this percentile
        are then compared with those below ``100 - positive_percentile``, and
        a warning says the estimate is approximate.
    min_events : int
        Minimum number of events required in each of the negative and the
        positive population to estimate a row.

    Each row ``i`` is ``(median_pos - median_neg)`` of every detector over
    that of the source channel, the populations split on the source channel
    (see the module comment above). Events piled at any matrix channel's
    ceiling (saturated) are left out first. Returns (channels, matrix);
    diagonal entries are always 1.0, negative estimates are clamped to 0.
    """
    from .spectral import _split_negative_positive
    n = len(channels)
    matrix = np.eye(n, dtype=float)
    diag_report = {}
    for i, src_ch in enumerate(channels):
        path = single_stain_paths.get(src_ch)
        if not path:
            continue
        try:
            fcs = read_fcs_tolerant(path, _fcs_read.FlowData, [])
        except Exception as exc:  # noqa: BLE001 - logged as a warning; an unreadable stain leaves its matrix row at identity
            _warn_skipped(src_ch, f"its control {os.path.basename(path)} "
                                  f"could not be read ({exc})")
            continue
        ch_names = [fcs.channels[k].get('pnn') or fcs.channels[k].get('PnN')
                    or f'Ch{k}' for k in range(1, fcs.channel_count + 1)]
        events = np.reshape(np.asarray(fcs.events),
                            (-1, fcs.channel_count)).astype(float)
        if src_ch not in ch_names:
            _warn_skipped(src_ch, f"its control {os.path.basename(path)} "
                                  f"has no {src_ch} channel")
            continue
        used = [c for c in channels if c in ch_names]
        cols = [ch_names.index(c) for c in used]
        frame = pd.DataFrame(events[:, cols], columns=pd.Index(used))
        ok = np.isfinite(frame.to_numpy()).all(axis=1)
        saturated = AcquisitionQC._margin_events(frame, used, 0.0)
        if saturated.any():
            log.info("[optimize] %s: %d of %d events saturated (at a "
                     "detector's ceiling) are left out.", src_ch,
                     int(saturated.sum()), len(frame))
        ok &= ~saturated
        x = frame[src_ch].to_numpy()[ok]
        if x.size < 2 * min_events:
            _warn_skipped(src_ch, f"only {x.size} usable events in "
                                  f"{os.path.basename(path)} (needs "
                                  f"{2 * min_events})")
            continue
        neg, pos, separated = _split_negative_positive(x)
        if not separated:
            log.warning(
                "[optimize] %s: %s shows no separate negative and positive "
                "population on %s, so its spillover is estimated from the "
                "brightest vs the dimmest %g%% of events and is approximate. "
                "A control with a distinct negative gives an exact estimate.",
                src_ch, os.path.basename(path), src_ch,
                100.0 - positive_percentile)
            pos = x > np.percentile(x, positive_percentile)
            neg = x < np.percentile(x, 100.0 - positive_percentile)
        n_pos, n_neg = int(pos.sum()), int(neg.sum())
        if n_pos < min_events or n_neg < min_events:
            _warn_skipped(src_ch, f"{n_pos} positive and {n_neg} negative "
                                  f"events in {os.path.basename(path)} "
                                  f"(needs {min_events} of each)")
            continue
        dx = float(np.median(x[pos]) - np.median(x[neg]))
        if not dx > 0:
            _warn_skipped(src_ch, "its positive and negative events do not "
                                  f"differ on {src_ch} in "
                                  f"{os.path.basename(path)}")
            continue
        diag_report[src_ch] = (n_pos, n_neg)
        for j, dst_ch in enumerate(channels):
            if j == i or dst_ch not in used:
                continue
            y = frame[dst_ch].to_numpy()[ok]
            slope = float(np.median(y[pos]) - np.median(y[neg])) / dx
            matrix[i, j] = max(0.0, slope)   # clamp negatives to 0
    log.info(f"[optimize] (positive, negative) events per source: {diag_report}")
    return list(channels), matrix
