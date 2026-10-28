"""Barcode deconvolution: split a pooled, barcoded tube back into samples.

Fluorescent cell barcoding (Krutzik & Nolan, 2006) and mass-cytometry
palladium barcoding (Zunder et al., 2015) stain each sample with its own
combination of barcode reagents, pool the samples into one tube, and acquire
them together. The *key* says which barcode channels are positive for each
sample: a binary code per sample over the barcode channels.

Single-cell debarcoding (Zunder et al., 2015):

1. Each barcode channel is rescaled over the events being debarcoded so its
   1st to 99th percentile maps to 0..1 (clipped), on the transformed scale
   the caller passes (arcsinh for mass data, logicle for fluorescence).
2. Per event the rescaled values are sorted. When every code in the key has
   the same number k of positives (a k-of-n scheme, e.g. 3-of-6), the
   event's positive channels are its k highest, and its *separation* is the
   k-th highest value minus the (k+1)-th. When the codes have different k,
   each k the key uses is tried and the one with the largest gap wins.
3. The positive channels form a code; the event goes to the sample with that
   code. Events whose code is in no sample, or whose separation is below the
   cutoff, stay unassigned (doublets of two barcoded cells, debris, cells
   that took up too little barcode).
4. Optionally (``second_pass``, Zunder's refinement, as in CATALYST) the
   values are rescaled again per sample: each event's barcode values are
   divided by the 95th percentile of its preliminarily assigned sample's
   positive population, channel by channel, and steps 2-3 are repeated. A
   sample stained dimmer than the others then separates as well as they do.

The preliminary assignment for the second pass takes every event whose code
matches a sample, whatever its separation, so the assignment at any cutoff
comes from one computation and :meth:`DebarcodeResult.yield_curve` can show
how the yields fall as the cutoff rises.

Tk-free, so it is tested without a display.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Separation cutoff: the same order as CATALYST's typical estimated cutoffs
# and the separations Zunder et al. used. Raising it keeps only cleanly
# separated events (purer samples, lower yield); lowering it keeps more
# events, including more doublets and misassigned cells.
DEFAULT_CUTOFF = 0.3
UNASSIGNED = 'Unassigned'
POSITIVE_QUANTILE = 95.0      # the second pass's per-sample scale
RESCALE_PERCENTILES = (1.0, 99.0)

__all__ = ['DEFAULT_CUTOFF', 'UNASSIGNED', 'BarcodeKey', 'BarcodeKeyError',
           'DebarcodeResult', 'read_key', 'debarcode', 'separation',
           'rescale']


class BarcodeKeyError(ValueError):
    """A barcode key that cannot be used, with a message for the user."""


@dataclass(frozen=True)
class BarcodeKey:
    """Sample name -> binary code over the barcode channels.

    ``codes[i, j]`` is 1 when sample ``samples[i]`` is positive for barcode
    channel ``channels[j]``."""
    samples: tuple[str, ...]
    channels: tuple[str, ...]
    codes: np.ndarray

    def __post_init__(self):
        codes = np.asarray(self.codes)
        if codes.ndim != 2 or codes.shape != (len(self.samples),
                                              len(self.channels)):
            raise BarcodeKeyError(
                f"The key has {len(self.samples)} sample(s) and "
                f"{len(self.channels)} channel(s), but its codes are shaped "
                f"{codes.shape}.")
        object.__setattr__(self, 'codes', codes.astype(np.int8))
        _validate(self)

    @classmethod
    def from_mapping(cls, mapping, channels):
        """``{sample: [0/1 per channel]}`` -> key."""
        samples = tuple(str(s) for s in mapping)
        codes = np.array([list(mapping[s]) for s in mapping], dtype=float)
        if codes.size and not np.isin(codes, (0, 1)).all():
            raise BarcodeKeyError("Every key entry must be 0 or 1.")
        return cls(samples, tuple(str(c) for c in channels),
                   codes.reshape(len(samples), len(channels)))

    @property
    def positives(self):
        """Positives per sample's code (k), as an int array."""
        return self.codes.sum(axis=1).astype(int)

    @property
    def uniform_k(self):
        """k when every code has the same number of positives, else None."""
        ks = set(self.positives.tolist())
        return ks.pop() if len(ks) == 1 else None

    def code_text(self, i):
        """Sample i's code as '101100'."""
        return ''.join(str(int(v)) for v in self.codes[i])

    def check_channels(self, available):
        """Raise BarcodeKeyError naming every key channel not in ``available``."""
        have = {str(c) for c in available}
        missing = [c for c in self.channels if c not in have]
        if missing:
            raise BarcodeKeyError(
                "The key names barcode channel(s) the data does not have: "
                + ', '.join(missing) + ".")

    def with_channels(self, channels):
        """The same codes over other channel names (a key column mapped onto
        a data channel of another name)."""
        return BarcodeKey(self.samples, tuple(str(c) for c in channels),
                          self.codes)


def _validate(key):
    if len(key.channels) < 2:
        raise BarcodeKeyError("A barcode key needs at least two barcode channels.")
    if not key.samples:
        raise BarcodeKeyError("The key lists no samples.")
    blank = [i + 1 for i, s in enumerate(key.samples) if not s.strip()]
    if blank:
        raise BarcodeKeyError(f"Sample name missing on key row(s) "
                        f"{', '.join(map(str, blank))}.")
    if UNASSIGNED.lower() in (s.strip().lower() for s in key.samples):
        raise BarcodeKeyError(f"'{UNASSIGNED}' is kept for the events no sample "
                        "matches; rename that sample in the key.")
    dup = sorted({s for s in key.samples if key.samples.count(s) > 1})
    if dup:
        raise BarcodeKeyError("Sample name(s) used twice in the key: "
                        + ', '.join(dup) + ".")
    dupc = sorted({c for c in key.channels if key.channels.count(c) > 1})
    if dupc:
        raise BarcodeKeyError("Barcode channel(s) listed twice in the key: "
                        + ', '.join(dupc) + ".")
    if not np.isin(key.codes, (0, 1)).all():
        raise BarcodeKeyError("Every key entry must be 0 or 1.")
    seen: dict[bytes, str] = {}
    clash = []
    for s, row in zip(key.samples, key.codes, strict=True):
        b = row.tobytes()
        if b in seen:
            clash.append(f"{seen[b]} and {s}")
        else:
            seen[b] = s
    if clash:
        raise BarcodeKeyError("Two samples share a barcode code, so their events "
                        "cannot be told apart: " + '; '.join(clash) + ".")


def read_key(source):
    """Read a key from a CSV path, or from CSV text when ``source`` holds a
    newline. Header row: a first column for the sample names (any title),
    then one column per barcode channel. Each row: a sample name, then 0 or
    1 per channel. Blank rows are skipped. Raises BarcodeKeyError."""
    if '\n' in str(source):
        text = str(source)
    else:
        try:
            with open(source, encoding='utf-8-sig', newline='') as fh:
                text = fh.read()
        except OSError as exc:
            raise BarcodeKeyError(f"The key file could not be read: {exc}") from exc
    try:
        dialect = csv.Sniffer().sniff(text.split('\n', 1)[0], ',;\t')
    except csv.Error:
        dialect = csv.excel                 # type: ignore[assignment]
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text),
                                                       dialect)]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        raise BarcodeKeyError("The key needs a header row (sample, then one column "
                        "per barcode channel) and at least one sample row.")
    head = rows[0]
    channels = tuple(head[1:])
    if any(not c for c in channels):
        raise BarcodeKeyError("A barcode channel column in the key has no name.")
    samples, codes = [], []
    for ln, r in enumerate(rows[1:], start=2):
        if len(r) != len(head):
            raise BarcodeKeyError(f"Key row {ln} has {len(r)} column(s); the header "
                            f"has {len(head)}.")
        vals = []
        for c, v in zip(channels, r[1:], strict=True):
            if v not in ('0', '1'):
                raise BarcodeKeyError(f"Key row {ln} ({r[0] or 'no name'}), column "
                                f"{c}: '{v}' is not 0 or 1.")
            vals.append(int(v))
        samples.append(r[0])
        codes.append(vals)
    return BarcodeKey(tuple(samples), channels, np.array(codes, dtype=np.int8))


# ── the computation ─────────────────────────────────────────────────────────

def rescale(values, percentiles=RESCALE_PERCENTILES):
    """Per column, map the percentiles' range to 0..1 and clip. NaN stays
    NaN; a column with no spread (it cannot say positive from negative) is
    all NaN."""
    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            out[:, j] = np.nan
            continue
        lo, hi = np.percentile(col[ok], percentiles)
        out[:, j] = (np.clip((col - lo) / (hi - lo), 0.0, 1.0) if hi > lo
                     else np.nan)
        out[~ok, j] = np.nan
    return out


def separation(scaled, ks):
    """Per event: ``(positive channel mask, separation)``.

    ``scaled`` is events x channels on 0..1. ``ks`` are the numbers of
    positives to try (the key's). For each k the gap is the k-th highest
    value minus the (k+1)-th, with 1 above the highest and 0 below the
    lowest, so k = 0 and k = n are defined; the k with the largest gap
    gives the positives. An event with a NaN has separation NaN and no
    positives."""
    x = np.asarray(scaled, dtype=float)
    n_ev, n_ch = x.shape
    bad: np.ndarray = np.asarray(~np.isfinite(x).all(axis=1))
    xs = np.where(bad[:, None], 0.0, x)
    order = np.argsort(-xs, axis=1, kind='stable')
    desc = np.take_along_axis(xs, order, axis=1)
    padded = np.hstack([np.ones((n_ev, 1)), desc, np.zeros((n_ev, 1))])
    ks = sorted({int(k) for k in ks})
    gaps = np.stack([padded[:, k] - padded[:, k + 1] for k in ks], axis=1)
    best = np.argmax(gaps, axis=1)
    sep = gaps[np.arange(n_ev), best]
    k_ev = np.asarray(ks)[best]
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n_ch)[None, :].repeat(n_ev, 0),
                      axis=1)
    pos = rank < k_ev[:, None]
    pos[bad] = False
    sep = np.where(bad, np.nan, sep)
    return pos, sep


def _match(key, pos):
    """Index of the key sample whose code equals each event's positives, or
    -1."""
    weights = 1 << np.arange(pos.shape[1], dtype=np.int64)
    ev = pos.astype(np.int64) @ weights
    kc = key.codes.astype(np.int64) @ weights
    order = np.argsort(kc)
    srt = kc[order]
    at = np.minimum(np.searchsorted(srt, ev), len(srt) - 1)
    return np.where(srt[at] == ev, order[at], -1).astype(np.int64)


def _second_pass_scale(key, scaled, match):
    """Divide each event's values by its preliminary sample's scale: per
    positive channel, that sample's 95th percentile there; per negative
    channel, the mean of its positive channels' scales. Unmatched events
    and samples with no events keep their values."""
    out = scaled.copy()
    for i in range(len(key.samples)):
        rows = match == i
        if not rows.any():
            continue
        posc = key.codes[i].astype(bool)
        scale = np.ones(len(key.channels))
        if posc.any():
            q = np.nanpercentile(scaled[rows][:, posc], POSITIVE_QUANTILE,
                                 axis=0)
            q = np.where(q > 0, q, 1.0)
            scale[posc] = q
            scale[~posc] = float(np.mean(q))
        out[rows] = np.clip(scaled[rows] / scale, 0.0, 1.0)
    return out


@dataclass
class DebarcodeResult:
    """Per event: the sample its code matches (``match``, an index into
    ``key.samples`` or -1) and its ``separation``. The assignment at a
    cutoff is the match where the separation reaches the cutoff."""
    key: BarcodeKey
    match: np.ndarray
    separation: np.ndarray
    cutoff: float = DEFAULT_CUTOFF
    second_pass: bool = False

    @property
    def assigned(self):
        """Per event, the sample index at ``cutoff``, or -1."""
        ok = np.isfinite(self.separation) & (self.separation >= self.cutoff)
        return np.where(ok, self.match, -1)

    def with_cutoff(self, cutoff):
        """The same result at another cutoff (no recomputation)."""
        return DebarcodeResult(self.key, self.match, self.separation,
                               float(cutoff), self.second_pass)

    def labels(self, unassigned=UNASSIGNED):
        """Per event, the sample name, or ``unassigned``."""
        names = np.array(list(self.key.samples) + [unassigned], dtype=object)
        a = self.assigned
        return names[np.where(a < 0, len(self.key.samples), a)]

    def yields(self):
        """``{sample: events}`` in key order, then ``UNASSIGNED``."""
        counts = np.bincount(self.assigned + 1,
                             minlength=len(self.key.samples) + 1)
        out = {s: int(counts[i + 1]) for i, s in enumerate(self.key.samples)}
        out[UNASSIGNED] = int(counts[0])
        return out

    def yield_curve(self, cutoffs=None):
        """Events per sample (columns, then ``UNASSIGNED``) at each cutoff
        (index): how the yields fall as the cutoff rises."""
        if cutoffs is None:
            cutoffs = np.round(np.arange(0.0, 1.0001, 0.05), 2)
        rows = [self.with_cutoff(c).yields() for c in cutoffs]
        return pd.DataFrame(rows, index=pd.Index(list(cutoffs),
                                                 name='cutoff'))


def debarcode(values, key, cutoff=DEFAULT_CUTOFF, second_pass=True):
    """Debarcode events.

    ``values``: a DataFrame holding the key's channels (by name), or an
    events x channels array in the key's channel order, on a transformed
    scale. Returns a :class:`DebarcodeResult`."""
    if isinstance(values, pd.DataFrame):
        key.check_channels(values.columns)
        x = values[list(key.channels)].to_numpy(dtype=float)
    else:
        x = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(key.channels):
            raise ValueError(f"Expected events x {len(key.channels)} "
                             f"barcode values, got shape {x.shape}.")
    ks = set(key.positives.tolist())
    if x.shape[0] == 0:
        empty = np.zeros(0)
        return DebarcodeResult(key, empty.astype(np.int64), empty,
                               float(cutoff), bool(second_pass))
    scaled = rescale(x)
    spread = np.asarray(np.isfinite(scaled).any(axis=0)).tolist()
    flat = [c for c, ok in zip(key.channels, spread, strict=True) if not ok]
    if flat:
        raise ValueError("No spread on barcode channel(s) "
                         + ", ".join(flat) + " among these events, so "
                         "positive cannot be told from negative.")
    pos, sep = separation(scaled, ks)
    match = _match(key, pos)
    if second_pass:
        scaled = _second_pass_scale(key, scaled, match)
        pos, sep = separation(scaled, ks)
        match = _match(key, pos)
    return DebarcodeResult(key, match, sep, float(cutoff), bool(second_pass))
