"""Acquisition QC (``AcquisitionQC``): signal drift, flow-rate and margin
(saturation) detectors over a sample's events, combined into one clean-event
index.

Core module: numpy and pandas, matplotlib imported where a plot is drawn. No
Tk, and it does not import ``openflo.pipeline`` (pipeline imports it).
"""

import logging

import numpy as np
import pandas as pd

from .channel_roles import is_excluded as _is_excluded
from .robust import MAD_TO_SD
from .robust import mad as _mad

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see its messages.
log = logging.getLogger('openflo.pipeline')


# ══════════════════════════════════════════════════════════════════════════════
# QC MODULE
# ══════════════════════════════════════════════════════════════════════════════

class AcquisitionQC:
    """
    Acquisition QC. Three independent anomaly detectors, combined into one
    clean-event index:

      1. **Signal drift** — time bins where any channel's events sit higher
         or lower than in its other bins: the bin's mean rank is more than
         ``threshold`` SDs from the median bin's (sensor drift, settling,
         sustained instability).
      2. **Flow-rate anomalies** — time bins whose event count is a robust
         (MAD) outlier, or interior bins that are empty: clogs (rate
         collapses) and bubbles (a gap or a burst).
      3. **Margin / saturation events** — events piled up at a channel's
         ceiling (off-scale), the classic signature of a bubble/clog or
         electronic saturation. These are dropped per-event, not per-bin.

    Detectors 1–2 need a Time channel and no-op without one; detector 3
    does not. A clean acquisition trips none of them.

    Usage
    -----
        qc = AcquisitionQC(sample.data)
        clean_idx = qc.run(n_bins=200, threshold=5)
        sample.data = sample.data.loc[clean_idx].reset_index(drop=True)
        qc.plot()
        qc.report   # {'drift': n, 'flow_rate': n, 'margin': n, 'total': n}
    """

    def __init__(self, data, time_channel='Time'):
        self.data         = data
        self.time_channel = self._find_time(data, time_channel)
        self.flag         = None
        self.bin_stats    = None
        self.bin_counts   = None
        self.report       = {}

    @staticmethod
    def _find_time(data, hint):
        if hint in data.columns:
            return hint
        for col in data.columns:
            if 'time' in col.lower():
                return col
        return None

    @staticmethod
    def _mad_outliers(vals, threshold, min_scale=0.0):
        """Boolean mask of robust (median ± threshold·MAD) outliers. The
        flow-rate detector's test; drift has its own, :meth:`_drift_outliers`.

        A MAD of exactly 0 is routine here: event counts tie exactly on a
        steady acquisition. (Per-bin medians of an integer channel tied too,
        when drift used them.) An epsilon such as ``+ 1e-10`` would turn that
        into a band of width ~1e-10, so any value differing by a single
        quantisation step would be an outlier. Measured: with it, a clean,
        drift-free integer channel lost 38% of its events to "drift" while
        the identical data as float lost none.

        A zero MAD carries no scale, so one must come from outside — and it
        must NOT be derived from the deviations themselves, or a lone genuine
        outlier sets the very band meant to catch it (a 350-event burst among
        uniform 50s made its own band 1750 and escaped). ``min_scale`` is
        therefore supplied by the caller from the statistics of its own data:
        the Poisson noise for a count. It is a floor, not a replacement — a
        real MAD still wins.

        With no scale from either source there is no spread to speak of, and
        nothing can be called an outlier.
        """
        vals = np.asarray(vals, dtype=float)
        med   = np.median(vals)
        dev   = np.abs(vals - med)
        scale = max(float(np.median(dev)), float(min_scale))
        if not scale > 0:
            return np.zeros(vals.shape, dtype=bool)
        return dev > threshold * scale

    @staticmethod
    def _mid_ranks(vals, cells=65536):
        """Each value's mid-rank in its channel, in [0, 1]: the fraction of
        events below it plus half of those level with it. ``vals`` finite.

        Values are placed on a grid of ``cells`` equal steps across roughly
        the 0.1 to 99.9 percentile range (the tails go to the end cells), and
        a cell's events share its mid-rank. That is exact for an integer
        channel spanning fewer steps, and any monotone grid still gives a
        valid rank test. Placement is arithmetic, and the range comes from a
        fixed stride of ~20k values: measured on 1M events, the whole
        function takes 10 ms a channel, where a binary search over the cells
        took 120 ms and a full percentile alone 13 ms."""
        v = np.asarray(vals, dtype=float)
        if v.size == 0:
            return v
        lo, hi = np.percentile(v[::max(1, v.size // 20_000)], [0.1, 99.9])
        if not hi > lo:
            return np.full(v.size, 0.5)
        cell = v - lo
        cell *= (cells - 1) / (hi - lo)
        np.clip(cell, 0, cells - 1, out=cell)
        cell = cell.astype(np.intp)
        n_at = np.bincount(cell, minlength=cells)
        return ((np.cumsum(n_at) - 0.5 * n_at) / v.size)[cell]

    @staticmethod
    def _drift_outliers(mr, n, threshold, floor, skew, kurt, clip=3.0):
        """Boolean mask of the bins whose mean rank ``mr`` (of ``n`` events)
        is more than ``threshold`` SDs from the centre. See
        :meth:`_drift_bad_bins` for why.

        A bin's deviation is rescaled to the median bin's event count, so
        every bin has the same null SD. The SD is 1.4826·MAD of those
        deviations, at least ``floor``. The bins within ``clip`` SDs of the
        median bin are kept and the rest set aside, re-estimated until that
        set is stable, so a fault spanning many bins does not widen the band
        meant to catch it. The centre the cut is measured from is then the
        mean rank of the kept bins' events: under the null that is every
        bin's expected mean rank, and on a skewed channel the median bin sits
        below it.

        The band is widened by Cornish-Fisher terms about the mean, never
        narrowed: on its long side by (t²-1)·skew/6, ``skew`` being the null
        skewness of a bin's mean rank, and on both sides by (t³-3t)·kurt/24,
        ``kurt`` its excess kurtosis beyond what that skewness brings in a
        Poisson count (the second-order terms that would narrow the band
        are left out). ``floor``, ``skew`` and ``kurt`` may be per bin."""
        mr = np.asarray(mr, dtype=float)
        n = np.asarray(n, dtype=float)
        w = np.sqrt(n / float(np.median(n)))
        floor = np.broadcast_to(np.asarray(floor, dtype=float), mr.shape)
        skew = np.broadcast_to(np.asarray(skew, dtype=float), mr.shape)
        kurt = np.broadcast_to(np.asarray(kurt, dtype=float), mr.shape)

        def spread(d, keep):
            kept = d[keep]
            return np.maximum(MAD_TO_SD * _mad(kept), floor)

        def band(t, sd):
            g = (t * t - 1.0) * skew / 6.0
            k = t + (t ** 3 - 3.0 * t) * np.maximum(kurt, 0.0) / 24.0
            return (k + np.maximum(g, 0.0)) * sd, (k + np.maximum(-g, 0.0)) * sd

        inl = np.ones(mr.size, dtype=bool)
        for _ in range(10):
            d = (mr - float(np.median(mr[inl]))) * w
            above, below = band(clip, spread(d, inl))
            nxt = (d <= above) & (-d <= below)
            if nxt.sum() < 0.5 * mr.size or bool((nxt == inl).all()):
                break
            inl = nxt
        d = (mr - float(np.sum(n[inl] * mr[inl]) / np.sum(n[inl]))) * w
        above, below = band(threshold, spread(d, inl))
        return (d > above) | (-d > below)

    def _drift_bad_bins(self, bins, channels, n_bins, threshold):
        """Time bins whose events sit higher or lower in a channel than that
        channel's other bins do: the bin's MEAN RANK is more than
        ``threshold`` SDs from the centre. Also populates self.bin_stats
        (per-bin mean ranks).

        Every (bin, channel) pair is a test, so on a clean, stationary
        acquisition the statistic must have the same, known spread in every
        channel and bin, and the cut must hold for the whole family. The
        per-bin RAW MEDIAN at 5 unscaled MADs had neither:

        * 5 MAD is only 3.4 SD, a two-sided tail of 7.5e-4: 200 bins x 7
          channels expected about one false bin per clean sample. At 5 SD
          the tail is 5.7e-7, under 0.01 false bins even at 200 x 64.
        * A bin's raw median depends on the channel's shape. On a bimodal
          marker whose median sits at the edge of the gap between negative
          and positive events, it jumps across the gap whenever chance puts
          the bin's positive fraction past one half. The shuffled synthetic
          PBMC sample lost 7556 of 20000 events to "drift" (CD3 flagged 75
          of 200 bins).

        Measured at the loader defaults on clean, stationary data, 240
        synthetic samples (normal and log-normal, 10k-100k events, 4-16
        fluors, float and integer, a constant and a 3x falling event rate):
        the raw median removed events from 228 of them, a mean of 0.8% to
        3.7% per group; this removes none.

        A rank does not depend on the channel's shape, and the mean rank
        uses every event, so a composition change moves it smoothly. The
        mean of n ranks has a sampling SD of sqrt(var(rank) / n), so each
        bin's deviation is rescaled to the median bin's event count: a bin
        that is sparse because the rate fell is noisier, not drifting.

        :meth:`_drift_outliers` makes the cut. Each part below was needed,
        and removing any one of them fails a test in
        tests/test_acquisition_qc_ground_truth.py:

        * The SD is 1.4826·MAD across bins, so bin-to-bin spread beyond the
          sampling noise (2% gain jitter per bin) is the channel's spread,
          not drift. Unscaled, 9 of 10 such samples lost bins.
        * The SD is at least the sampling SD above, a figure from the
          events, not from the deviations being judged. The MAD of 200 bins
          is itself off by ~8%; clean log-normal samples lost a bin where it
          fell low (3 of 30 at 100k x 10 fluors).
        * Bins more than 3 SDs out are set aside before the centre and SD
          are taken, re-estimated until stable. A bin's mean rank can move
          by at most ~0.5 however large the fault, and a fault over 40% of a
          short run inflated the MAD until the 5-SD band passed that limit:
          one channel shifted by 1000 SD lost 29% of its faulty events at
          5000 events (none on one seed) and 6% at 3000. With the set-aside,
          99.9% and 96%, as with the raw median. A smooth trend is not set
          aside and is tolerated.
        * The centre is the mean rank of the kept bins' events, not the
          median bin: under the null that mean is every bin's expectation,
          and on a skewed channel the median bin sits below it.
        * On a channel mostly tied at one value, a bin's mean rank counts
          its rare events, and 5 SDs is not a 5.7e-7 tail. One-sided
          (0 but for 2% of events, 40 channels, 50k events), 11 of 50 clean
          samples lost a bin; two-sided (0.1% below, 0.1% above), 10 of 20.
          The band is widened by Cornish-Fisher skew and kurtosis terms,
          from the events' own moments over that bin's event count: taken
          at the median bin, sparse bins where the rate fell (the tube
          running dry) cost 12 of 20 clean samples a bin.

        Sensitivity, one fluor shifted by k SEs of a bin's median over 10%
        of a 50k-event run (20 seeds): 40% of the shifted events removed at
        k=4, 82% at k=5, 98% at k=6, all from k=8; outside the shift, 5
        events in all 20 runs, from the bin straddling its edge. The raw
        median took 54%, 84% and 96%, and removed 11067 events outside it.

        The widening is the price of holding the 5-SD false-positive rate on
        tied channels, and it is paid in sensitivity there (10 seeds, a 50k
        run, removed share of the faulty events; in brackets, a variant
        without the kurtosis term or per-bin moments, which lost
        events from up to 12 of 20 CLEAN samples of these kinds):

        * one-sided sparse channel, positive share 2% -> 10% over 10% of
          the run: 71%; 10% -> none: 57%;
        * two-sided (0, with rare events below and above), the upper share
          0.5% -> 4% over 10% of the run: 14% (34%); 1% -> 8%: 66% (77%);
        * an unused integer detector at 37 whose +1 glitches go 0.5% -> 5%
          over 2% of the run: 55% (82%).

        The raw median caught none of the last three (its removals there
        were its 0.5-0.7% background on clean data).

        Bins with < 10 events contribute nothing (NaN in bin_stats, which
        keeps a row per bin, len == n_bins, for the QC plot)."""
        b = np.asarray(bins)
        valid = ~pd.isna(b)
        all_valid = bool(valid.all())
        bi = (b if all_valid else b[valid]).astype(np.intp)
        n_all = np.bincount(bi, minlength=n_bins)[:n_bins]

        stats = {}
        bad = set()
        for ch in channels:
            if not pd.api.types.is_numeric_dtype(self.data[ch]):
                continue
            col = np.asarray(self.data[ch].to_numpy(), dtype=float)
            if not all_valid:
                col = col[valid]
            fin = np.isfinite(col)
            if fin.all():                    # the usual case: no copies
                vals, b_fin, n_in = col, bi, n_all
            else:
                vals, b_fin = col[fin], bi[fin]
                n_in = np.bincount(b_fin, minlength=n_bins)[:n_bins]
            rank = self._mid_ranks(vals)
            ok = n_in >= 10
            mean_rank = np.full(n_bins, np.nan)
            mean_rank[ok] = (np.bincount(b_fin, weights=rank,
                                         minlength=n_bins)[:n_bins][ok]
                             / n_in[ok])
            stats[ch] = mean_rank
            if not ok.any():
                continue
            n_ok = n_in[ok].astype(float)
            n_ref = float(np.median(n_ok))
            dev = rank - rank.mean()
            sq = dev * dev
            var = float(sq.mean())
            if not var > 0:              # every event level: nothing to drift
                continue
            # Per bin: the skewness and excess kurtosis of a mean of that
            # bin's events, the events' own over sqrt(n) and n, so larger in
            # a sparse bin. Moments by dot product: ** 3 on 1M events cost
            # ~31 ms a channel.
            g1 = float(np.dot(sq, dev)) / dev.size / var ** 1.5
            k1 = float(np.dot(sq, sq)) / dev.size / var ** 2 - 3.0 - g1 * g1
            out = self._drift_outliers(
                mean_rank[ok], n_ok, threshold, float(np.sqrt(var / n_ref)),
                g1 / np.sqrt(n_ok), k1 / n_ok)
            bad.update(np.flatnonzero(ok)[out].tolist())
        self.bin_stats = pd.DataFrame(stats, index=range(n_bins))
        return bad

    def _flowrate_bad_bins(self, bins, n_bins, threshold):
        """Time bins whose event count is a MAD outlier, plus empty
        interior bins (a gap = bubble; a collapse = clog). Edge bins are
        exempt from the 'empty' rule — acquisitions routinely start/stop
        mid-bin. Also populates self.bin_counts."""
        # Vectorised per-bin counts (bincount) — identical to the former
        # per-bin count_nonzero loop, O(N) instead of O(N·n_bins).
        b = np.asarray(bins)
        valid = ~pd.isna(b)
        counts = np.bincount(b[valid].astype(int),
                             minlength=n_bins)[:n_bins].astype(int)
        self.bin_counts = counts
        nonempty = counts[counts > 0]
        if nonempty.size < 3:
            return set()
        bad = set()
        # Count outliers among bins that actually have events.
        nz_idx = np.where(counts > 0)[0]
        # Counting statistics: a bin holding ~N events carries sqrt(N) noise
        # of its own, so that is the floor when the counts tie exactly.
        out    = self._mad_outliers(
            counts[nz_idx], threshold,
            min_scale=float(np.sqrt(max(np.median(counts[nz_idx]), 1.0))))
        bad.update(nz_idx[out].tolist())
        # Empty interior bins (gaps), ignoring leading/trailing empties.
        # Only meaningful when bins are densely populated — on a sparse
        # file an empty interior bin is expected, not an anomaly.
        first, last = int(nz_idx[0]), int(nz_idx[-1])
        if np.median(nonempty) >= 20:
            for b in range(first + 1, last):
                if counts[b] == 0:
                    bad.add(b)
        return bad

    @staticmethod
    def _margin_events(data, channels, frac, eps=1e-9):
        """Boolean per-event mask of margin/saturation events: those at a
        channel's ceiling (its max) when that ceiling is *piled up* — i.e.
        at least `frac` of events share the max value. A single off-scale
        max (continuous data) isn't a pile-up and is left alone."""
        n = len(data)
        bad = np.zeros(n, dtype=bool)
        if n == 0:
            return bad
        thresh = max(2, int(np.ceil(frac * n)))
        for ch in channels:
            col = np.asarray(data[ch].values, dtype=float)
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            # A channel with no range at all (a dead or disabled detector,
            # constant across the file) has no ceiling to pile up against:
            # every event is trivially "at the max". Flagging them deleted
            # 100% of an otherwise healthy sample because of ONE unused
            # channel, since margins are OR-ed across the panel.
            if float(finite.max() - finite.min()) <= eps:
                continue
            ceiling = finite.max()
            at_ceiling = np.isfinite(col) & (np.abs(col - ceiling) <= eps)
            if int(at_ceiling.sum()) >= thresh:
                bad |= at_ceiling
        return bad

    @staticmethod
    def _time_bins(t, n_bins):
        """Each event's equal-width Time bin (``pd.cut`` labels), NaN where
        Time is not finite; None when no Time value is. ``pd.cut`` raises on
        ±inf ("cannot specify integer `bins` when input data contains
        infinity"), so one infinite Time crashed QC, and with it the GUI load
        that runs QC on every file. Such events, like a NaN Time, get no bin:
        the time-based detectors cannot judge them, and they are kept."""
        t = np.asarray(t, dtype=float)
        fin = np.isfinite(t)
        if not fin.any():
            return None
        if not fin.all():
            t = np.where(fin, t, np.nan)
        return pd.cut(t, bins=n_bins, labels=False)

    def run(self, n_bins=200, threshold: float = 5, channels=None,
            drift=True, flow_rate=True, margins=True, flow_rate_threshold=5.0,
            margin_frac=0.01):
        """Flag drift, flow-rate and margin events and return the index of the events kept.

        Stores the per-detector removal counts in ``self.report`` and the keep mask in
        ``self.flag``. Channels default to every numeric, non-boolean detector column."""
        n = len(self.data)
        keep = np.ones(n, dtype=bool)
        report = {'drift': 0, 'flow_rate': 0, 'margin': 0}

        if channels is None:
            # Boolean columns are never detectors. The FMO step writes one
            # `<channel>_pos` flag per marker, and numpy treats bools as 0/1 —
            # so the margin detector saw "every event piled up at the ceiling"
            # and deleted every marker-POSITIVE event (measured: 49.8% of a
            # clean sample). Name-matching those away would be fragile; the
            # dtype is the honest signal.
            channels = [c for c in self.data.columns
                        if c != self.time_channel
                        and not _is_excluded(c)
                        and not pd.api.types.is_bool_dtype(self.data[c])
                        and pd.api.types.is_numeric_dtype(self.data[c])]

        bins = None
        if self.time_channel is not None and n_bins > 0 and n > 0:
            bins = self._time_bins(self.data[self.time_channel].values,
                                   n_bins)
        if bins is not None:
            bin_arr = np.asarray(bins)
            unbinned = int(pd.isna(bin_arr).sum())
            if unbinned:
                log.info("  [QC] %s events with no finite Time are not "
                         "time-binned: drift and flow-rate cannot judge "
                         "them, so they are kept.", f"{unbinned:,}")

            if drift:
                drift_bad = self._drift_bad_bins(
                    bins, channels, n_bins, threshold)
                if drift_bad:
                    drift_evt = np.isin(bin_arr, list(drift_bad))
                    report['drift'] = int(drift_evt.sum())
                    keep &= ~drift_evt

            if flow_rate:
                flow_bad = self._flowrate_bad_bins(
                    bins, n_bins, flow_rate_threshold)
                if flow_bad:
                    flow_evt = np.isin(bin_arr, list(flow_bad))
                    report['flow_rate'] = int(flow_evt.sum())
                    keep &= ~flow_evt
        else:
            log.info("  [QC] No time channel with a finite value — "
                     "time-based detectors skipped.")

        if margins:
            margin_evt = self._margin_events(self.data, channels, margin_frac)
            report['margin'] = int(margin_evt.sum())
            keep &= ~margin_evt

        report['total'] = int((~keep).sum())
        self.report = report
        self.flag   = pd.Series(keep, index=self.data.index)
        pct_rem     = (report['total'] / n * 100.0) if n else 0.0
        log.info(
            "  [QC] Removed %.1f%% events (%s total: drift %s, flow-rate %s, "
            "margin %s).",
            pct_rem, f"{report['total']:,}", f"{report['drift']:,}",
            f"{report['flow_rate']:,}", f"{report['margin']:,}")
        return self.data.index[keep]

    def plot(self):
        """Plot the event count per time bin; returns the Axes, or None before ``run``."""
        if self.bin_stats is None:
            log.info("  [QC] Run .run() first.")
            return None
        import matplotlib.pyplot as plt  # lazy: see module-top comment
        bins = self._time_bins(self.data[self.time_channel].values,
                               len(self.bin_stats))
        cnts = pd.Series(bins).value_counts().sort_index()
        _, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.asarray(cnts.index), np.asarray(cnts.values),
                lw=0.8, color='steelblue')
        ax.set_xlabel('Time bin')
        ax.set_ylabel('Event count')
        ax.set_title('Acquisition QC — event rate over time')
        plt.tight_layout()
        return ax
