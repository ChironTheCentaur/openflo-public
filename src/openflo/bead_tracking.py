"""Bead tracking over time: Levey-Jennings charts with Westgard rules
(Complaints 25).

Each day's QC bead run gives one value per channel (the beads' median). The
first runs, or the runs marked as the baseline, set each channel's mean and
SD; every later run is judged against them by the Westgard multirule
(Westgard et al., Clin Chem 1981;27:493):

- ``1-3s``: one run beyond 3 SD (reject)
- ``2-2s``: two runs in a row beyond 2 SD on the same side (reject)
- ``R-4s``: two runs in a row more than 4 SD apart (reject)
- ``4-1s``: four runs in a row beyond 1 SD on the same side (reject)
- ``10x``: ten runs in a row on the same side of the mean (reject)
- ``1-2s``: one run beyond 2 SD (warning only)

The log is a plain CSV (date, sample, then one column per channel), so it
can be kept beside the instrument and opened anywhere. Pure Python / numpy.
"""
from __future__ import annotations

import csv
import io

import numpy as np

REJECT_RULES = ('1-3s', '2-2s', 'R-4s', '4-1s', '10x')
WARN_RULES = ('1-2s',)


def baseline(values, n=None):
    """``(mean, sd)`` of the first ``n`` finite ``values`` (all of them when
    ``n`` is None). NaN for fewer than two values or no spread: a chart
    without a baseline judges nothing (NA, never 0)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if n is not None:
        v = v[:int(n)]
    if v.size < 2:
        return float('nan'), float('nan')
    sd = float(np.std(v, ddof=1))
    if not sd > 0:
        return float(np.mean(v)), float('nan')
    return float(np.mean(v)), sd


def westgard(values, mean, sd):
    """The rules each run breaks: a list, one entry per value, of the rule
    names (REJECT_RULES then WARN_RULES order) that run violates as the last
    run of the pattern. A NaN value breaks nothing and interrupts no run of
    the multi-run rules; without a baseline (NaN mean or SD) nothing is
    judged."""
    v = np.asarray(values, dtype=float)
    out = [[] for _ in range(v.size)]
    if not (np.isfinite(mean) and np.isfinite(sd) and sd > 0):
        return out
    idx = [i for i in range(v.size) if np.isfinite(v[i])]
    z = {i: (v[i] - mean) / sd for i in idx}
    for k, i in enumerate(idx):
        zi = z[i]
        prev = [z[j] for j in idx[max(0, k - 9):k]]      # earlier, oldest first
        rules = []
        if abs(zi) > 3:
            rules.append('1-3s')
        if prev and abs(zi) > 2 and abs(prev[-1]) > 2 \
                and np.sign(zi) == np.sign(prev[-1]):
            rules.append('2-2s')
        if prev and abs(zi - prev[-1]) > 4 and np.sign(zi) != np.sign(prev[-1]) \
                and max(abs(zi), abs(prev[-1])) > 2 \
                and min(abs(zi), abs(prev[-1])) > 2:
            rules.append('R-4s')
        last4 = prev[-3:] + [zi]
        if len(last4) == 4 and all(abs(x) > 1 for x in last4) \
                and len({np.sign(x) for x in last4}) == 1:
            rules.append('4-1s')
        last10 = prev[-9:] + [zi]
        if len(last10) == 10 and all(x != 0 for x in last10) \
                and len({np.sign(x) for x in last10}) == 1:
            rules.append('10x')
        if abs(zi) > 2:
            rules.append('1-2s')
        out[i] = rules
    return out


def status(rules):
    """'reject', 'warning' or 'ok' for one run's broken ``rules``."""
    if any(r in REJECT_RULES for r in rules):
        return 'reject'
    if any(r in WARN_RULES for r in rules):
        return 'warning'
    return 'ok'


class BeadLog:
    """Runs of QC beads: ``runs`` is a list of ``{'date', 'sample',
    'values': {channel: value}}`` in run order; ``baseline_runs`` how many
    of the first runs set each channel's mean and SD."""

    def __init__(self, runs=None, baseline_runs=20):
        self.runs = list(runs or [])
        self.baseline_runs = int(baseline_runs)

    @property
    def channels(self):
        out = []
        for r in self.runs:
            for ch in r['values']:
                if ch not in out:
                    out.append(ch)
        return out

    def add(self, date, sample, values):
        self.runs.append({'date': str(date), 'sample': str(sample),
                          'values': {k: float(v) for k, v in values.items()}})

    def series(self, channel):
        return np.array([r['values'].get(channel, np.nan) for r in self.runs],
                        dtype=float)

    def judge(self, channel):
        """``{'mean', 'sd', 'values', 'rules', 'status'}`` for one channel:
        every run judged against the baseline runs (which are judged too,
        so a bad baseline shows)."""
        v = self.series(channel)
        mean, sd = baseline(v, self.baseline_runs)
        rules = westgard(v, mean, sd)
        return {'mean': mean, 'sd': sd, 'values': v, 'rules': rules,
                'status': [status(r) if np.isfinite(x) else 'NA'
                           for r, x in zip(rules, v, strict=True)]}

    def to_csv(self):
        chans = self.channels
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator='\n')
        w.writerow(['date', 'sample'] + chans)
        for r in self.runs:
            w.writerow([r['date'], r['sample']]
                       + [('NA' if ch not in r['values']
                           or not np.isfinite(r['values'][ch])
                           else f"{r['values'][ch]:.6g}") for ch in chans])
        return buf.getvalue()

    @classmethod
    def from_csv(cls, text, baseline_runs=20):
        rows = [r for r in csv.reader(io.StringIO(text.lstrip('﻿')))
                if any(c.strip() for c in r)]
        if not rows or [h.strip().lower() for h in rows[0][:2]] != \
                ['date', 'sample']:
            raise ValueError("A bead log starts with the columns date, "
                             "sample, then one column per channel.")
        chans = [h.strip() for h in rows[0][2:]]
        log = cls(baseline_runs=baseline_runs)
        for r in rows[1:]:
            r = r + [''] * (len(chans) + 2 - len(r))
            vals = {}
            for ch, x in zip(chans, r[2:], strict=False):
                try:
                    vals[ch] = float(x)
                except ValueError:
                    continue
            log.runs.append({'date': r[0].strip(), 'sample': r[1].strip(),
                             'values': vals})
        return log
