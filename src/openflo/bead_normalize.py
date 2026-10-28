"""Bead-based signal normalization over a mass-cytometry run (Finck et al.
2013, Cytometry A 83:483).

A mass cytometer's sensitivity drifts while it acquires: the detector ages
and the cones foul, so the same cell reads dimmer at the end of a run than
at the start (and differently from one day to the next). Polystyrene beads
loaded with known amounts of several metals are spiked into the sample; they
are the same at every moment, so how their signal changes over Time is how
the instrument changed. The correction, as Finck et al. describe it:

1. find the bead events: high on EVERY bead channel (each channel's
   threshold is Otsu's on ``arcsinh(x / 5)``, or one the user sets), or an
   existing population the user names as beads;
2. smooth each bead channel's intensity over Time with a running median
   over a window of beads (default 500);
3. take a baseline per bead channel: the mean of those smoothed values over
   the run -- this file's own, a reference file's, or all files' together so
   several runs land on one shared level;
4. at each bead, one multiplicative factor by least squares through the
   origin across the bead channels (Finck's slope):
   ``f = sum(b_c * s_c) / sum(s_c ** 2)`` for baseline ``b`` and smoothed
   ``s``;
5. interpolate the factors to every event by its Time and multiply every
   mass channel by it. Time and the Gaussian discrimination parameters are
   not intensities and are never touched.

All arithmetic is on LINEAR values (the editor converts the display
transform back and forth). The factors are kept as a short list of (Time,
factor) knots in a JSON-able *record*, so a normalization can be re-applied
exactly to the raw file when a session is reopened, or undone, without
keeping a copy of the data. Pure numpy / pandas; no Tk.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    'COFACTOR', 'DEFAULT_WINDOW', 'MAX_KNOTS', 'find_time_channel',
    'is_gaussian_parameter', 'mass_channels', 'suggest_bead_channels',
    'otsu_split', 'bead_thresholds', 'identify_beads', 'bead_keep_mask',
    'running_median',
    'smooth_beads', 'ls_factors', 'fit_run', 'normalize_runs',
    'factors_at', 'apply_record', 'bead_trace', 'methods_text',
]

# CyTOF's usual arcsinh cofactor: bead identification looks at
# arcsinh(x / 5), where a metal's counts are log-like.
COFACTOR = 5.0
# Beads per running-median window. Finck et al. smoothed over a window of
# beads; 500 follows that and the common normalizer default. Fewer beads
# follow faster drift but carry more noise.
DEFAULT_WINDOW = 500
# Knots kept in a record. The factor curve is a running median, so it is
# smooth; 2000 points follow it closely and keep a session small.
MAX_KNOTS = 2000
# Fewest bead events a run must have before its drift is estimated.
MIN_BEADS = 20

_TIME_RE = re.compile(r'(?<![A-Z0-9])TIME')
# The Gaussian discrimination (pulse-shape) parameters of a mass cytometer,
# and its event length: standard instrument parameter names.
_GAUSSIAN = {'EVENTLENGTH', 'CENTER', 'CENTRE', 'OFFSET', 'WIDTH', 'RESIDUAL'}


def _norm(name: object) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


def find_time_channel(columns: Iterable[object]) -> str | None:
    """The first acquisition-time column of `columns` (any case), or None."""
    for c in columns:
        if _TIME_RE.search(str(c).upper()):
            return str(c)
    return None


def is_gaussian_parameter(name: object) -> bool:
    """True for Event_length, Center, Offset, Width or Residual (any case or
    separator)."""
    return _norm(name) in _GAUSSIAN


def mass_channels(columns: Iterable[object]) -> list[str]:
    """`columns` without Time and the Gaussian parameters: the intensity
    channels a normalization multiplies."""
    return [str(c) for c in columns
            if not _TIME_RE.search(str(c).upper())
            and not is_gaussian_parameter(c)]


def suggest_bead_channels(columns: Iterable[object],
                          labels: Mapping[str, str] | None = None) -> list[str]:
    """A pre-selection for the bead channels: those whose name or marker
    label says 'bead'. Only a suggestion; the user picks."""
    labels = labels or {}
    out = []
    for c in columns:
        text = f"{c} {labels.get(str(c), '')}".lower()
        if 'bead' in text:
            out.append(str(c))
    return out


# ── Bead identification ─────────────────────────────────────────────────────

def otsu_split(z: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold on the finite values of `z`, over their whole range.

    Beads are a few percent of the events, far above the cells on a bead
    channel, so the between-class variance is flat across the empty gap
    between the two; the middle of that flat top is returned rather than its
    first bin (which would sit on the cells' upper edge). NaN when there is
    nothing to split."""
    v = np.asarray(z, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float('nan')
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        return float('nan')
    hist, edges = np.histogram(v, bins=int(bins), range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    sb[~np.isfinite(sb)] = -np.inf
    best = float(sb.max())
    if not np.isfinite(best):
        return float('nan')
    top = np.flatnonzero(sb >= best * (1.0 - 1e-9))
    # The split lies above bin k: between its centre and the next one.
    k_lo, k_hi = int(top[0]), int(top[-1])
    return float(0.5 * (edges[k_lo + 1] + edges[k_hi + 1]))


def _asinh(x, cofactor: float = COFACTOR) -> np.ndarray:
    return np.arcsinh(np.asarray(x, dtype=float) / float(cofactor))


def bead_thresholds(frame: pd.DataFrame, bead_channels: Sequence[str],
                    cofactor: float = COFACTOR) -> dict[str, float]:
    """``{channel: threshold}`` on ``arcsinh(x / cofactor)``: Otsu's split
    of each bead channel's LINEAR values in `frame`."""
    return {str(c): otsu_split(_asinh(frame[c].to_numpy(), cofactor))
            for c in bead_channels}


def identify_beads(frame: pd.DataFrame, bead_channels: Sequence[str],
                   thresholds: Mapping[str, float] | None = None,
                   cofactor: float = COFACTOR,
                   ) -> tuple[np.ndarray, dict[str, float]]:
    """``(mask, thresholds)``: the events above the threshold on EVERY bead
    channel of `frame` (linear values). A channel missing from `thresholds`
    gets Otsu's (:func:`bead_thresholds`). Thresholds are on
    ``arcsinh(x / cofactor)``."""
    chans = [str(c) for c in bead_channels]
    if not chans:
        raise ValueError("choose at least one bead channel")
    missing = [c for c in chans if c not in frame.columns]
    if missing:
        raise ValueError(f"bead channel(s) {missing} not in the data")
    thr = dict(thresholds or {})
    auto = [c for c in chans if c not in thr or thr[c] is None
            or not np.isfinite(float(thr[c]))]
    thr.update(bead_thresholds(frame, auto, cofactor))
    mask = np.ones(len(frame), dtype=bool)
    for c in chans:
        t = float(thr[c])
        if not np.isfinite(t):
            mask[:] = False
            break
        mask &= _asinh(frame[c].to_numpy(), cofactor) > t
    return mask, {c: float(thr[c]) for c in chans}


def bead_keep_mask(df, params: Mapping[str, Any] | None = None) -> np.ndarray:
    """Keep-mask of the auto-clean method that removes bead events (key
    ``'beads'``): the events NOT above Otsu's split on every bead channel
    named in ``params['channels']`` (comma-separated). It works on the values
    as the editor holds them (its display scale, log-like for mass channels),
    recomputed per sample. Without a bead channel in `df` nothing is
    removed."""
    want = [c.strip() for c in str((params or {}).get('channels') or '')
            .split(',') if c.strip()]
    chans = [c for c in want if c in df.columns]
    keep = np.ones(len(df), dtype=bool)
    if not chans or len(chans) != len(want) or not len(df):
        return keep
    bead = np.ones(len(df), dtype=bool)
    for c in chans:
        v = np.asarray(df[c].values, dtype=float)
        t = otsu_split(v)
        if not np.isfinite(t):
            return keep
        bead &= v > t
    return ~bead


# ── Smoothing and the factor ────────────────────────────────────────────────

def running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running median of `values` over `window` points; the window
    shrinks at the ends (as many points as there are)."""
    v = pd.Series(np.asarray(values, dtype=float))
    w = max(1, int(window))
    return v.rolling(w, center=True, min_periods=1).median().to_numpy()


def smooth_beads(times: np.ndarray, beads: np.ndarray,
                 window: int = DEFAULT_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """``(sorted times, smoothed)`` for bead events: each column of `beads`
    (beads x channels, linear) is put in Time order and smoothed by
    :func:`running_median` over `window` beads (capped at the number of
    beads)."""
    t = np.asarray(times, dtype=float)
    b = np.asarray(beads, dtype=float)
    if b.ndim == 1:
        b = b[:, None]
    order = np.argsort(t, kind='stable')
    t, b = t[order], b[order]
    w = min(int(window), len(t)) if len(t) else 1
    sm = np.column_stack([running_median(b[:, j], w)
                          for j in range(b.shape[1])]) if len(t) else b
    return t, sm


def ls_factors(baseline: Sequence[float], smoothed: np.ndarray) -> np.ndarray:
    """Finck's slope at each row of `smoothed` (rows x bead channels): the
    one factor ``f`` that best scales the row onto `baseline` by least
    squares through the origin, ``f = sum(b * s) / sum(s ** 2)``. A row
    with no signal is NaN."""
    b = np.asarray(baseline, dtype=float)
    s = np.asarray(smoothed, dtype=float)
    if s.ndim == 1:
        s = s[None, :]
    num = (s * b[None, :]).sum(axis=1)
    den = (s * s).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = num / den
    f[~(den > 0)] = np.nan
    return f


def _thin(t: np.ndarray, f: np.ndarray, max_knots: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """Knots for np.interp: one per distinct time (mean factor of ties),
    at most `max_knots`, first and last kept."""
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if not t.size:
        return t, f
    ut, inv = np.unique(t, return_inverse=True)
    uf = np.bincount(inv, weights=f) / np.bincount(inv)
    if ut.size > max_knots:
        idx = np.unique(np.linspace(0, ut.size - 1, max_knots).round()
                        .astype(int))
        ut, uf = ut[idx], uf[idx]
    return ut, uf


def factors_at(times: np.ndarray, knots_t: Sequence[float] | np.ndarray,
               knots_f: Sequence[float] | np.ndarray) -> np.ndarray:
    """The factor at each of `times`: linear between knots, the end knots'
    factor before the first bead and after the last. An event without a
    finite Time gets factor 1 (left as it is)."""
    t = np.asarray(times, dtype=float)
    kt = np.asarray(knots_t, dtype=float)
    kf = np.asarray(knots_f, dtype=float)
    out = np.ones(t.shape, dtype=float)
    ok = np.isfinite(t)
    if kt.size == 1:
        out[ok] = kf[0]
    elif kt.size:
        out[ok] = np.interp(t[ok], kt, kf)
    return out


# ── Fit and apply ───────────────────────────────────────────────────────────

def fit_run(frame: pd.DataFrame, bead_channels: Sequence[str],
            time_channel: str | None = None,
            bead_mask: np.ndarray | None = None,
            thresholds: Mapping[str, float] | None = None,
            window: int = DEFAULT_WINDOW,
            cofactor: float = COFACTOR) -> dict[str, Any]:
    """One run's beads: their mask, Time and smoothed intensities.

    `frame` holds LINEAR values. `bead_mask` (a population the user chose)
    replaces the threshold identification. Returns ``{'mask', 'thresholds',
    'times', 'smoothed', 'raw', 'time_channel', 'bead_channels',
    'window'}``; ``times`` / ``smoothed`` / ``raw`` are in Time order."""
    chans = [str(c) for c in bead_channels]
    tcol = time_channel or find_time_channel(frame.columns)
    if tcol is None or tcol not in frame.columns:
        raise ValueError("the data has no Time channel; bead normalization "
                         "follows the signal over Time")
    if bead_mask is not None:
        mask = np.asarray(bead_mask, dtype=bool)
        if mask.shape != (len(frame),):
            raise ValueError("the bead population does not match the events")
        missing = [c for c in chans if c not in frame.columns]
        if missing:
            raise ValueError(f"bead channel(s) {missing} not in the data")
        thr: dict[str, float] = {}
    else:
        mask, thr = identify_beads(frame, chans, thresholds, cofactor)
    t = frame[tcol].to_numpy(dtype=float)[mask]
    raw = frame.loc[mask, chans].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(raw).all(axis=1)
    t, raw = t[ok], raw[ok]
    if len(t) < MIN_BEADS:
        raise ValueError(
            f"found {len(t)} bead event(s); at least {MIN_BEADS} are needed "
            "to follow the drift (check the bead channels and thresholds)")
    w = max(1, min(int(window), len(t)))
    ts, sm = smooth_beads(t, raw, w)
    order = np.argsort(t, kind='stable')
    return {'mask': mask, 'thresholds': thr, 'times': ts, 'smoothed': sm,
            'raw': raw[order], 'time_channel': str(tcol),
            'bead_channels': chans, 'window': w}


def normalize_runs(frames: Mapping[str, pd.DataFrame],
                   bead_channels: Sequence[str],
                   channels: Sequence[str] | None = None,
                   baseline: str = 'each',
                   bead_masks: Mapping[str, np.ndarray] | None = None,
                   thresholds: Mapping[str, float] | None = None,
                   window: int = DEFAULT_WINDOW,
                   time_channel: str | None = None,
                   cofactor: float = COFACTOR,
                   max_knots: int = MAX_KNOTS) -> dict[str, dict[str, Any]]:
    """Normalize each run of `frames` (``{name: LINEAR frame}``) to its beads.

    `channels` are multiplied (default: every column but Time and the
    Gaussian parameters, :func:`mass_channels`). `baseline` is 'each' (every
    run on its own beads' mean level), 'all' (one shared level: the mean
    over every run's beads) or the name of a reference run. Returns
    ``{name: {'frame': normalized copy, 'record': JSON-able record,
    'fit': fit_run's result, 'factors': per-event factor}}``; the inputs are
    never changed."""
    if not frames:
        raise ValueError("no samples to normalize")
    masks = bead_masks or {}
    fits = {n: fit_run(df, bead_channels, time_channel, masks.get(n),
                       thresholds, window, cofactor)
            for n, df in frames.items()}
    if baseline == 'all':
        pooled = np.concatenate([f['smoothed'] for f in fits.values()])
        shared = pooled.mean(axis=0)
    elif baseline == 'each':
        shared = None
    elif baseline in fits:
        shared = fits[baseline]['smoothed'].mean(axis=0)
    else:
        raise ValueError(f"baseline must be 'each', 'all' or a sample name, "
                         f"got {baseline!r}")
    out: dict[str, dict[str, Any]] = {}
    for name, df in frames.items():
        fit = fits[name]
        base = (fit['smoothed'].mean(axis=0) if shared is None else shared)
        f = ls_factors(base, fit['smoothed'])
        kt, kf = _thin(fit['times'], f, int(max_knots))
        if not kt.size or not (np.isfinite(kf).all() and (kf > 0).all()):
            raise ValueError(f"{name}: the bead signal gives no usable "
                             "factor (no signal on the bead channels?)")
        chans = ([str(c) for c in channels] if channels is not None
                 else mass_channels(df.columns))
        chans = [c for c in chans if c in df.columns
                 and c != fit['time_channel'] and not is_gaussian_parameter(c)]
        record = {
            'method': 'Finck 2013', 'time_channel': fit['time_channel'],
            'bead_channels': list(fit['bead_channels']), 'channels': chans,
            'window': int(fit['window']), 'baseline_from': str(baseline),
            'baseline': {c: float(v) for c, v in
                         zip(fit['bead_channels'], base, strict=True)},
            'thresholds': {c: float(v) for c, v in fit['thresholds'].items()},
            'cofactor': float(cofactor),
            'n_beads': len(fit['times']),
            'knots_t': [float(x) for x in kt],
            'knots_f': [float(x) for x in kf],
        }
        factors = factors_at(df[fit['time_channel']].to_numpy(), kt, kf)
        out[name] = {'frame': apply_record(df, record), 'record': record,
                     'fit': fit, 'factors': factors}
    return out


def apply_record(frame: pd.DataFrame, record: Mapping[str, Any],
                 inverse: bool = False) -> pd.DataFrame:
    """A copy of `frame` (LINEAR values) with the record's channels
    multiplied by its factors (divided, with `inverse`, which undoes it).
    Other columns are copied unchanged."""
    out = frame.copy()
    tcol = record['time_channel']
    if tcol not in frame.columns:
        raise ValueError(f"no Time channel {tcol!r} to place the factors")
    f = factors_at(frame[tcol].to_numpy(), record['knots_t'],
                   record['knots_f'])
    if inverse:
        f = 1.0 / f
    for c in record['channels']:
        if c in out.columns:
            out[c] = frame[c].to_numpy(dtype=float) * f
    return out


# ── Reporting ───────────────────────────────────────────────────────────────

def bead_trace(times: np.ndarray, values: np.ndarray,
               window: int = DEFAULT_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """``(sorted times, running median)`` of one bead channel: the curve the
    dialog draws before and after."""
    t, sm = smooth_beads(times, np.asarray(values, dtype=float)[:, None],
                         window)
    return t, sm[:, 0]


def drift_percent(smoothed: np.ndarray) -> float:
    """How far the smoothed bead signal moves over a run: the largest over
    bead channels of (max - min) / mean of its running median, in %."""
    s = np.asarray(smoothed, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    if not len(s):
        return float('nan')
    with np.errstate(divide='ignore', invalid='ignore'):
        span = (s.max(axis=0) - s.min(axis=0)) / s.mean(axis=0)
    span = span[np.isfinite(span)]
    return float(100.0 * span.max()) if span.size else float('nan')


def methods_text(records: Mapping[str, Mapping[str, Any]]) -> str:
    """A methods-report sentence for the records of one normalization."""
    from . import __version__
    if not records:
        return ''
    first = next(iter(records.values()))
    base = first.get('baseline_from', 'each')
    where = {'each': "each run's own mean bead level",
             'all': "the mean bead level of all runs together"}.get(
        base, f"the mean bead level of {base}")
    beads = ', '.join(first.get('bead_channels', []))
    n_beads = sum(int(r.get('n_beads', 0)) for r in records.values())
    lo = min(min(r['knots_f']) for r in records.values())
    hi = max(max(r['knots_f']) for r in records.values())
    return (f"Signal drift over acquisition was corrected with bead-based "
            f"normalization (Finck et al. 2013) in OpenFlo {__version__}: "
            f"{len(records)} file(s), {n_beads:,} bead events on {beads}, "
            f"intensities smoothed by a running median over "
            f"{first.get('window', DEFAULT_WINDOW)} beads and scaled to "
            f"{where} by a least-squares factor per time point (range "
            f"{lo:.3g} to {hi:.3g}), applied to "
            f"{len(first.get('channels', []))} mass channel(s).")
