"""Per-sample saturation report: the percent of events at the top of each
channel's range.

An event at a detector's ceiling was clipped: its true intensity is unknown,
anywhere at or above the top, so it piles up at the edge of the plot and drags
medians and spreads. The acquisition QC's margin check removes pile-ups and
logs one pooled count; this reports, per sample and channel, how many events
sit there, so a saturated channel or a sample run at too high a voltage is
seen before analysis.

The ceiling is the top of the channel's range as the FCS file states it:
``$PnR`` - 1, the highest value the instrument records (for integer data
stored log-amplified or with a gain, that top value converted the way the
loader converts the data). Events are judged on the raw detector values,
before compensation and transforms. A channel whose range is not known (a
sample rebuilt from a CSV) reads NA, not 0: nothing was measured.
"""
from __future__ import annotations

import math

# A channel with at least this share of events at its ceiling is flagged:
# the same share at which the acquisition QC calls a ceiling piled up
# (AcquisitionQC margin_frac).
FLAG_PERCENT = 1.0


def _number(v):
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def channel_ceiling(metadata, index):
    """The highest value parameter ``index`` (1-based) can record, in the
    units the loader gives its values, or None when the file does not say.

    Float data is stored as intensities, so the top is ``$PnR`` - 1 (a
    18-bit range's 262,143). Integer data stored log-amplified
    (``$PnE f1,f2``, f1 > 0) or with a gain (``$PnG``) is converted by the
    loader (FlowSample._decode_stored_values), and its top channel ``$PnR`` - 1
    is converted the same way."""
    md = {str(k).lower().lstrip('$'): v for k, v in (metadata or {}).items()}
    rng = _number(md.get(f'p{index}r'))
    if not rng or rng <= 1:
        return None
    top = rng - 1.0
    if str(md.get('datatype', '')).strip().upper() != 'I':
        return top
    decades, log0 = 0.0, 0.0
    pne = str(md.get(f'p{index}e', '') or '')
    parts = [p for p in pne.replace(';', ',').split(',') if p.strip()]
    if len(parts) == 2:
        decades = _number(parts[0]) or 0.0
        log0 = _number(parts[1]) or 0.0
    if decades > 0:
        return (log0 or 1.0) * 10.0 ** (decades * top / rng)
    gain = _number(md.get(f'p{index}g')) or 1.0
    if gain not in (0.0, 1.0):
        return top / gain
    return top


def _parameter_index(metadata, channel_names, channel):
    """``channel``'s 1-based FCS parameter number: the ``$PnN`` naming it,
    else its position among the sample's channels."""
    md = {str(k).lower().lstrip('$'): v for k, v in (metadata or {}).items()}
    n = int(_number(md.get('par')) or 0)
    for i in range(1, n + 1):
        if str(md.get(f'p{i}n', '')).strip() == channel:
            return i
    names = list(channel_names or ())
    return names.index(channel) + 1 if channel in names else None


def report_channels(sample):
    """The detectors a sample is judged on: scatter then fluorescence, as
    the sample lists them (Time and derived columns are not detectors)."""
    from .pipeline import _is_time
    out = []
    for ch in (list(getattr(sample, 'scatter_channels', None) or ())
               + list(getattr(sample, 'fluor_channels', None) or ())):
        if ch not in out and not _is_time(str(ch)):
            out.append(ch)
    return out


def sample_saturation(name, sample, eps=1e-6):
    """One sample's report row::

        {'Sample': name, 'Events': n, 'Any channel': %, 'Removed by QC': k,
         '<channel>': %, ..., '__flagged__': [channels at or over
         FLAG_PERCENT], '__unknown__': [channels with no known range]}

    Percentages are of the events in the sample's raw data (after any QC,
    which may already have removed a pile-up: 'Removed by QC' says how many
    events it took). A channel with no known ceiling is NaN (NA)."""
    import numpy as np

    raw = getattr(sample, 'raw', None)
    if raw is None or not len(getattr(raw, 'columns', ())):
        raw = getattr(sample, 'data', None)
    n = int(len(raw)) if raw is not None else 0
    md = getattr(sample, 'metadata', None) or {}
    names = getattr(sample, 'channel_names', None) or list(
        getattr(raw, 'columns', ()))
    row = {'Sample': name, 'Events': n, 'Any channel': float('nan'),
           'Removed by QC': int(getattr(sample, 'qc_removed', 0) or 0),
           '__flagged__': [], '__unknown__': []}
    hit = np.zeros(n, dtype=bool)
    known = False
    for ch in report_channels(sample):
        if raw is None or ch not in raw.columns:
            continue
        idx = _parameter_index(md, names, ch)
        top = channel_ceiling(md, idx) if idx else None
        if top is None or n == 0:
            row[ch] = float('nan')
            row['__unknown__'].append(ch)
            continue
        col = np.asarray(raw[ch].values, dtype=float)
        # At or above the top: some instruments write a saturated event a
        # hair over $PnR - 1, and a float file can hold values past it.
        at = np.isfinite(col) & (col >= top * (1.0 - eps))
        hit |= at
        known = True
        pct = 100.0 * float(at.sum()) / n
        row[ch] = pct
        if pct >= FLAG_PERCENT:
            row['__flagged__'].append(ch)
    if known:
        row['Any channel'] = 100.0 * float(hit.sum()) / n
    return row


def saturation_report(samples):
    """Rows for ``samples`` (``[(name, sample)]``) and the column order:
    identity columns, then every channel in first-seen order."""
    rows = [sample_saturation(nm, s) for nm, s in samples]
    cols = ['Sample', 'Events', 'Any channel', 'Removed by QC']
    for r in rows:
        for k in r:
            if k not in cols and not k.startswith('__'):
                cols.append(k)
    return rows, cols
