"""Bead tracking over time: Levey-Jennings charts with Westgard rules.

Each day's QC bead run gives one value per channel (the beads' median). The
first runs, or the runs marked as the baseline, set each channel's mean and
SD; every later run is judged against them by the Westgard multirule
(Westgard et al., Clin Chem 1981;27:493):

- ``1-3s``: one run beyond 3 SD (reject)
- ``2-2s``: two runs in a row beyond 2 SD on the same side (reject)
- ``R-4s``: the range exceeds 4 SD (reject), in two forms:

  * within a run (Westgard's original form): one bead level of a channel
    beyond +2 SD and another level of the same channel beyond -2 SD in the
    same run. It needs two or more bead levels (say a dim and a bright
    peak) logged per run, so a one-level log never breaks it;
  * across runs: the same level in two runs in a row beyond 2 SD on
    opposite sides. Westgard's paper does not use this form; it is on by
    default here so a one-level log still has a range check, and
    ``across_runs_r4s=False`` (Preferences > Statistics, "Westgard R-4s:
    literature form only") turns it off.
- ``4-1s``: four runs in a row beyond 1 SD on the same side (reject)
- ``10x``: ten runs in a row on the same side of the mean (reject)
- ``1-2s``: one run beyond 2 SD (warning only)

The log is a plain CSV (date, sample, then one column per channel), so it
can be kept beside the instrument and opened anywhere. A log with several
bead levels has one column per channel and level, named ``<channel> @
<level>`` (``FL1-A @ dim``); a one-level log keeps plain channel columns, so
older logs load unchanged. Each column is its own Levey-Jennings series with
its own baseline. Pure Python / numpy.
"""
from __future__ import annotations

import csv
import io

import numpy as np

REJECT_RULES = ('1-3s', '2-2s', 'R-4s', '4-1s', '10x')
WARN_RULES = ('1-2s',)

# A log column for one bead level of one channel is "<channel> @ <level>".
LEVEL_SEP = ' @ '


def level_key(channel, level=None):
    """The log column for ``channel`` at bead ``level`` (the plain channel
    name for a one-level log, ``level`` None or empty)."""
    return f"{channel}{LEVEL_SEP}{level}" if level else str(channel)


def split_level(key):
    """``(channel, level)`` of a log column; level is ``''`` for a plain
    channel column. The first separator splits: a level (a gate name) may
    contain ``' @ '``, a channel name does not."""
    channel, sep, level = str(key).partition(LEVEL_SEP)
    return (channel, level) if sep and channel and level else (str(key), '')


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


def _breaks_1_3s(zi, prev):
    """This run is beyond 3 SD."""
    return abs(zi) > 3


def _breaks_2_2s(zi, prev):
    """This run and the one before are beyond 2 SD on the same side."""
    return bool(prev) and abs(zi) > 2 and abs(prev[-1]) > 2 \
        and np.sign(zi) == np.sign(prev[-1])


def _breaks_r_4s(zi, prev):
    """This run and the one before are more than 4 SD apart, on opposite
    sides, each beyond 2 SD."""
    return bool(prev) and abs(zi - prev[-1]) > 4 \
        and np.sign(zi) != np.sign(prev[-1]) \
        and max(abs(zi), abs(prev[-1])) > 2 \
        and min(abs(zi), abs(prev[-1])) > 2


def _breaks_4_1s(zi, prev):
    """This run and the three before are beyond 1 SD on the same side."""
    last4 = prev[-3:] + [zi]
    return len(last4) == 4 and all(abs(x) > 1 for x in last4) \
        and len({np.sign(x) for x in last4}) == 1


def _breaks_10x(zi, prev):
    """This run and the nine before are on the same side of the mean."""
    last10 = prev[-9:] + [zi]
    return len(last10) == 10 and all(x != 0 for x in last10) \
        and len({np.sign(x) for x in last10}) == 1


def _breaks_1_2s(zi, prev):
    """This run is beyond 2 SD (a warning)."""
    return abs(zi) > 2


# REJECT_RULES then WARN_RULES order: westgard lists a run's rules in it.
_RULE_CHECKS = (
    ('1-3s', _breaks_1_3s),
    ('2-2s', _breaks_2_2s),
    ('R-4s', _breaks_r_4s),
    ('4-1s', _breaks_4_1s),
    ('10x', _breaks_10x),
    ('1-2s', _breaks_1_2s),
)


def _broken_rules(zi, prev, skip=()):
    """The rule names a run with z-score ``zi`` breaks, given ``prev``, the
    z-scores of up to nine earlier finite runs, oldest first; rules named in
    ``skip`` are not checked."""
    return [name for name, breaks in _RULE_CHECKS
            if name not in skip and breaks(zi, prev)]


def _in_rule_order(rules):
    """``rules`` sorted into REJECT_RULES then WARN_RULES order."""
    order = [name for name, _ in _RULE_CHECKS]
    return sorted(set(rules), key=order.index)


def westgard(values, mean, sd, across_runs_r4s=True):
    """The rules each run breaks: a list, one entry per value, of the rule
    names (REJECT_RULES then WARN_RULES order) that run violates as the last
    run of the pattern. A NaN value breaks nothing and interrupts no run of
    the multi-run rules; without a baseline (NaN mean or SD) nothing is
    judged. ``across_runs_r4s=False`` leaves out R-4s across two runs (the
    literature form judges R-4s only within a run: see
    :func:`within_run_r4s`)."""
    v = np.asarray(values, dtype=float)
    out = [[] for _ in range(v.size)]
    if not (np.isfinite(mean) and np.isfinite(sd) and sd > 0):
        return out
    idx = [i for i in range(v.size) if np.isfinite(v[i])]
    z = {i: (v[i] - mean) / sd for i in idx}
    skip = () if across_runs_r4s else ('R-4s',)
    for k, i in enumerate(idx):
        prev = [z[j] for j in idx[max(0, k - 9):k]]      # earlier, oldest first
        out[i] = _broken_rules(z[i], prev, skip)
    return out


def within_run_r4s(z_levels):
    """Westgard's within-run R-4s for each run: ``z_levels`` is a list of
    z-score series, one per bead level of one channel, each against its own
    baseline (NaN where a run lacks the level or the level has no
    baseline). Returns one bool per run: True when one level is beyond
    +2 SD and another beyond -2 SD in that run."""
    z = np.asarray(z_levels, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2:
        return [False] * (z.shape[-1] if z.ndim else 0)
    with np.errstate(invalid='ignore'):
        high = np.any(z > 2, axis=0)
        low = np.any(z < -2, axis=0)
    return [bool(h and lo) for h, lo in zip(high, low, strict=True)]


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
        """Every channel seen in any run, in first-seen order."""
        out = []
        for r in self.runs:
            for ch in r['values']:
                if ch not in out:
                    out.append(ch)
        return out

    def add(self, date, sample, values):
        """Append one run: its date, sample name and ``{channel: value}`` bead
        readings (``{level_key(channel, level): value}`` for several bead
        levels)."""
        self.runs.append({'date': str(date), 'sample': str(sample),
                          'values': {k: float(v) for k, v in values.items()}})

    def add_levels(self, date, sample, levels):
        """Append one run with several bead levels: ``levels`` is ``{level:
        {channel: value}}``. One level is logged as plain channel columns,
        the same as :meth:`add`."""
        if len(levels) == 1:
            self.add(date, sample, next(iter(levels.values())))
            return
        self.add(date, sample, {level_key(ch, lvl): v
                                for lvl, vals in levels.items()
                                for ch, v in vals.items()})

    def levels_of(self, column):
        """Every log column holding a bead level of ``column``'s channel
        (``column`` included), in first-seen order. A plain channel column
        has no other levels."""
        channel, level = split_level(column)
        if not level:
            return [column]
        return [c for c in self.channels
                if split_level(c)[1] and split_level(c)[0] == channel]

    def series(self, channel):
        """Return one channel's value per run as a float array, NaN where a run lacks it."""
        return np.array([r['values'].get(channel, np.nan) for r in self.runs],
                        dtype=float)

    def _z(self, column):
        """``column``'s z-scores per run against its own baseline (NaN with
        no baseline)."""
        v = self.series(column)
        mean, sd = baseline(v, self.baseline_runs)
        if not (np.isfinite(mean) and np.isfinite(sd) and sd > 0):
            return np.full(v.size, np.nan)
        return (v - mean) / sd

    def judge(self, channel, across_runs_r4s=True):
        """``{'mean', 'sd', 'values', 'rules', 'status'}`` for one log
        column: every run judged against the baseline runs (which are
        judged too, so a bad baseline shows). R-4s is judged within each run
        across the channel's bead levels (a run that breaks it shows it on
        every level logged in that run), and, with ``across_runs_r4s``
        (the default), also across two runs in a row of this column."""
        v = self.series(channel)
        mean, sd = baseline(v, self.baseline_runs)
        rules = westgard(v, mean, sd, across_runs_r4s=across_runs_r4s)
        levels = self.levels_of(channel)
        if len(levels) > 1:
            hits = within_run_r4s([self._z(c) for c in levels])
            for i, hit in enumerate(hits):
                # Westgard rejects the whole run, so every level logged in
                # it shows the rule.
                if hit and np.isfinite(v[i]):
                    rules[i] = _in_rule_order(rules[i] + ['R-4s'])
        return {'mean': mean, 'sd': sd, 'values': v, 'rules': rules,
                'status': [status(r) if np.isfinite(x) else 'NA'
                           for r, x in zip(rules, v, strict=True)]}

    def to_csv(self):
        """Render the log as CSV text: date, sample, then one column per channel.

        Missing or non-finite values are written as ``NA``.
        """
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
        """Build a log from CSV text in the :meth:`to_csv` layout.

        Raises ValueError when the header does not start with date, sample. Cells
        that do not parse as numbers are left out of that run.
        """
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
