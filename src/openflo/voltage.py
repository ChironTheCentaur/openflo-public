"""Voltage titration / Stain Index tool.

A small, generalized utility: point it at a *titration series* — one FCS
per PMT voltage of the same control (beads or a stained sample) — and a
channel, and it reports per-voltage **Stain Index** and the robust CV of
the negative population, then recommends the lowest voltage on the SI
plateau (maximum separation without over-volting / dynamic-range loss).

    Stain Index (SI) = (median(pos) - median(neg)) / (2 * rSD(neg))
        rSD = 1.4826 * MAD   (outlier-resistant SD estimate)

The metric layer (split_pos_neg / stain_index / robust_cv /
recommend_plateau) is pure and unit-tested on synthetic data; the IO layer
(reading $PnV from FCS metadata, loading files) sits on top. Nothing here
is specific to a panel or experiment — any channel, any file set.

Per channel: each PMT has its own optimum, so several channels can be
titrated in one run (``--channel`` repeatable, or ``--all-channels``),
each yielding its own SI curve + recommended voltage. Files sharing a PMT
voltage are pooled per channel for more robust statistics.

CLI::

    openflo-voltage v300.fcs v400.fcs v500.fcs --channel "PE-A"
    openflo-voltage *.fcs --channel CD11b --channel CD45 --plot -o si.csv
    openflo-voltage *.fcs --all-channels -o si.csv
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

_MAD_TO_SD = 1.4826   # MAD * this ≈ σ for a normal distribution


def read_pmt_voltage(metadata, channel_names, channel):
    """PMT voltage ($PnV) for `channel`, or None if absent/unparseable.

    `metadata` is the FCS TEXT dict (FlowSample.metadata); `channel_names`
    is the ordered detector list so we can map the channel to its 1-based
    parameter index n. FCS writers vary on the '$' prefix and case, so we
    probe several key spellings."""
    try:
        idx = list(channel_names).index(channel) + 1
    except ValueError:
        return None
    candidates = (f'$P{idx}V', f'P{idx}V', f'$p{idx}v', f'p{idx}v')
    for key in candidates:
        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                return None
    target = f'p{idx}v'
    for k, v in metadata.items():
        if k.lower().lstrip('$') == target:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


class VoltageTitration:
    """Stain-Index voltage walk over a titration series. See module docs."""

    # ── Pure metric layer ────────────────────────────────────────────────

    @staticmethod
    def robust_sd(arr):
        """Outlier-resistant SD estimate (1.4826 * MAD)."""
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float('nan')
        med = np.median(a)
        return _MAD_TO_SD * float(np.median(np.abs(a - med)))

    @staticmethod
    def robust_cv(arr):
        """Robust CV (%) = 100 * rSD / median. NaN when median ~ 0."""
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float('nan')
        med = float(np.median(a))
        if abs(med) < 1e-12:
            return float('nan')
        return 100.0 * VoltageTitration.robust_sd(a) / abs(med)

    @staticmethod
    def stain_index(pos, neg):
        """SI = (median(pos) - median(neg)) / (2 * rSD(neg))."""
        pos = np.asarray(pos, dtype=float)
        neg = np.asarray(neg, dtype=float)
        pos = pos[np.isfinite(pos)]
        neg = neg[np.isfinite(neg)]
        if pos.size == 0 or neg.size == 0:
            return float('nan')
        sd = VoltageTitration.robust_sd(neg)
        if not np.isfinite(sd) or sd <= 0:
            return float('nan')
        return (float(np.median(pos)) - float(np.median(neg))) / (2.0 * sd)

    @staticmethod
    def split_pos_neg(values, cofactor=150.0):
        """Split a mixed channel into (negative, positive) populations via
        a 2-component Gaussian mixture on an arcsinh-scaled axis (stable
        across decades). The lower-mean component is negative. Returns
        ``(neg, pos)`` arrays; if the data can't be split (<10 finite
        events) everything is returned as negative."""
        from sklearn.mixture import GaussianMixture
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 10:
            return v, np.array([], dtype=float)
        x = np.arcsinh(v / cofactor).reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0, n_init=2).fit(x)
        means = np.asarray(gm.means_)
        lo = int(np.argmin(means.ravel()))
        labels = gm.predict(x)
        return v[labels == lo], v[labels != lo]

    @staticmethod
    def recommend_plateau(voltages, si_values, frac=0.95):
        """Lowest voltage whose SI reaches `frac` of the maximum SI — the
        start of the plateau, where separation is near-best but the PMT
        isn't pushed harder than it needs to be. None when no usable data."""
        pairs = [(float(v), float(s))
                 for v, s in zip(voltages, si_values, strict=False)
                 if v is not None and s is not None and np.isfinite(s)]
        if not pairs:
            return None
        pairs.sort()
        max_si = max(s for _, s in pairs)
        if not np.isfinite(max_si) or max_si <= 0:
            return None
        for v, s in pairs:
            if s >= frac * max_si:
                return v
        return pairs[-1][0]

    # ── IO / orchestration ───────────────────────────────────────────────

    @staticmethod
    def _resolve_channel(sample, name):
        """Detector name, then antibody label (case-insensitive). None if
        the channel isn't in this sample."""
        if name in sample.data.columns:
            return name
        for det, lbl in sample.channel_labels.items():
            if lbl.lower() == name.lower() and det in sample.data.columns:
                return det
        return None

    @staticmethod
    def _channel_values(sample, channel):
        """(detector, values) for `channel` in `sample`, or (None, None)."""
        col = VoltageTitration._resolve_channel(sample, channel)
        if col is None:
            return None, None
        return col, np.asarray(sample.data[col].values, dtype=float)

    @staticmethod
    def _channel_set(samples, channels, all_channels):
        """The ordered list of channels to analyze. With `all_channels`,
        the ordered union of every sample's fluor detectors; otherwise the
        explicitly requested `channels`."""
        if all_channels:
            seen, out = set(), []
            for s, _ in samples:
                for c in getattr(s, 'fluor_channels', []) or []:
                    if c not in seen:
                        seen.add(c)
                        out.append(c)
            return out
        return list(channels or [])

    @classmethod
    def _channel_result(cls, channel, points, frac=0.95, voltage_round=0):
        """Build one channel's SI curve from `points` — a list of
        ``(voltage_or_None, values_ndarray)``, one per file. Files sharing
        a voltage (rounded to `voltage_round` decimals) are POOLED into a
        single curve point for more robust statistics. Pure.

        Returns ``{'channel', 'rows', 'recommended_voltage'}`` where each
        row is ``{channel, voltage, n_files, n_events, median_neg,
        median_pos, si, rcv_neg}``."""
        groups = {}
        for volt, vals in points:
            key = round(float(volt), voltage_round) if volt is not None else None
            groups.setdefault(key, []).append(np.asarray(vals, dtype=float))
        rows = []
        for key, arrs in groups.items():
            pooled = np.concatenate(arrs) if arrs else np.array([])
            neg, pos = cls.split_pos_neg(pooled)
            rows.append({
                'channel': channel,
                'voltage': key,
                'n_files': len(arrs),
                'n_events': int(np.isfinite(pooled).sum()),
                'median_neg': float(np.median(neg)) if neg.size else float('nan'),
                'median_pos': float(np.median(pos)) if pos.size else float('nan'),
                'si': cls.stain_index(pos, neg),
                'rcv_neg': cls.robust_cv(neg),
            })
        rows.sort(key=lambda r: (r['voltage'] is None, r['voltage'] or 0.0))
        rec = cls.recommend_plateau(
            [r['voltage'] for r in rows], [r['si'] for r in rows], frac)
        return {'channel': channel, 'rows': rows, 'recommended_voltage': rec}

    @classmethod
    def analyze(cls, paths, channels=None, frac=0.95, all_channels=False,
                voltage_round=0):
        """Run the voltage walk over a titration series.

        `paths`        the FCS files (any number; replicates at the same
                       voltage are pooled per channel).
        `channels`     detector names or antibody labels to titrate; ignored
                       when `all_channels=True`.
        `all_channels` analyze every fluor detector (per-channel curves).

        Each PMT has its own optimum, so results are **per channel**:
        returns ``{'frac', 'order': [channel...], 'results': {channel:
        {'channel', 'rows', 'recommended_voltage'}}}``. Files are read once
        and reused across channels."""
        from .pipeline import FlowSample
        samples = [(FlowSample(p), os.path.basename(p)) for p in paths]
        chans = cls._channel_set(samples, channels, all_channels)
        results = {}
        order = []
        for ch in chans:
            points = []
            for s, _base in samples:
                col, vals = cls._channel_values(s, ch)
                if col is None:
                    continue
                volt = read_pmt_voltage(s.metadata, s.channel_names, col)
                points.append((volt, vals))
            if not points:
                log.warning("  [voltage] channel %r not in any file — skipped",
                            ch)
                continue
            results[ch] = cls._channel_result(ch, points, frac, voltage_round)
            order.append(ch)
        return {'frac': frac, 'order': order, 'results': results}

    @staticmethod
    def plot(result, ax=None):
        """Per-channel SI vs voltage, one line per channel, each channel's
        recommended plateau voltage marked in its own colour."""
        import matplotlib.pyplot as plt
        results = result.get('results', {})
        if not results:
            log.warning("  [voltage] nothing to plot")
            return None
        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.5))
        for ch in result.get('order', list(results)):
            res = results[ch]
            rows = [r for r in res['rows'] if r['voltage'] is not None]
            if not rows:
                continue
            v  = [r['voltage'] for r in rows]
            si = [r['si'] for r in rows]
            line, = ax.plot(v, si, 'o-', label=ch)
            rec = res.get('recommended_voltage')
            if rec is not None:
                ax.axvline(rec, color=line.get_color(), ls=':', lw=1.2,
                           alpha=0.7)
        ax.set_xlabel('PMT voltage (V)')
        ax.set_ylabel('Stain Index')
        ax.set_title('Voltage titration — Stain Index per channel')
        ax.legend(loc='best', fontsize=8)
        return ax


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='openflo-voltage',
        description='Voltage titration / Stain Index over an FCS series. '
                    'Files sharing a PMT voltage are pooled per channel.')
    p.add_argument('files', nargs='+',
                   help='FCS files (titration series; globs expanded). '
                        'Replicates at the same voltage are pooled.')
    p.add_argument('--channel', '-c', action='append', metavar='CH',
                   help='Detector or antibody label to titrate. Repeatable '
                        'for several channels.')
    p.add_argument('--all-channels', action='store_true',
                   help='Titrate every fluor detector (per-channel curves).')
    p.add_argument('--frac', type=float, default=0.95,
                   help='Plateau fraction of max SI (default 0.95).')
    p.add_argument('--voltage-round', type=int, default=0,
                   help='Decimals to round $PnV to when pooling (default 0 '
                        '= nearest volt).')
    p.add_argument('--plot', action='store_true', help='Show the SI plot.')
    p.add_argument('-o', '--out', help='Write the combined table to CSV.')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args(argv)

    if not args.channel and not args.all_channels:
        p.error('give --channel CH (repeatable) or --all-channels')

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(message)s')

    paths = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) if any(c in f for c in '*?[') else [f])
    if not paths:
        p.error('no input files matched')

    result = VoltageTitration.analyze(
        paths, channels=args.channel, frac=args.frac,
        all_channels=args.all_channels, voltage_round=args.voltage_round)
    if not result['order']:
        log.warning('No requested channel was present in any file.')
        return 1

    all_rows = []
    for ch in result['order']:
        res = result['results'][ch]
        log.info('\nVoltage titration — %s', ch)
        log.info('%8s %7s %10s %10s %8s', 'voltage', 'files', 'SI',
                 'rCV_neg%', 'n')
        for r in res['rows']:
            v = f"{r['voltage']:.0f}" if r['voltage'] is not None else 'n/a'
            log.info('%8s %7d %10.2f %10.1f %8d',
                     v, r['n_files'], r['si'], r['rcv_neg'], r['n_events'])
            all_rows.append(r)
        rec = res['recommended_voltage']
        log.info('  → recommended (>= %.0f%% of max SI): %s',
                 args.frac * 100, f'{rec:g} V' if rec is not None else 'n/a')

    if args.out:
        import csv
        with open(args.out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        log.info('\nWrote %s', args.out)

    if args.plot:
        import matplotlib.pyplot as plt
        VoltageTitration.plot(result)
        plt.tight_layout()
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
