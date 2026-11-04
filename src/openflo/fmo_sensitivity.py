"""How much an FMO-set threshold moves the result: each stained sample's %
positive with the FMO held to several percentiles (1% of FMO events above the
gate, 0.5%, ...). Tk-free; the FMO gating dialog shows it as a table.

Every value here is LINEAR (compensated intensity), as the FMO gating dialog
takes its cutoff: the percentile of the FMO's linear values, compared with the
stained sample's linear values. Positive is ``x >= cutoff``, as a threshold
gate counts it (``gate_to_mask``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .gating_helpers import fmo_threshold

# 99 = 1 % of FMO events above the gate (the usual choice), 99.5 = 0.5 %.
DEFAULT_PERCENTILES = (99.0, 99.5)

COLUMNS = ('Sample', 'Channel', 'FMO', 'FMO % above (set)', 'Threshold',
           'Events', 'Positive', '% positive', 'FMO events',
           'FMO % above (measured)')


def parse_percentiles(text: str) -> list[float]:
    """``"99, 99.5"`` -> ``[99.0, 99.5]``: sorted, without repeats. Raises
    ``ValueError`` for an empty list or a value outside 0 to 100."""
    out: set[float] = set()
    for part in str(text).replace(';', ',').replace(' ', ',').split(','):
        if not part.strip():
            continue
        v = float(part)
        if not (0.0 <= v <= 100.0) or v != v:
            raise ValueError(f"percentile {part.strip()} is outside 0 to 100")
        out.add(v)
    if not out:
        raise ValueError("no percentiles given")
    return sorted(out)


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def threshold_rows(fmos: Mapping[str, tuple[str, Iterable[float]]],
                   stained: Mapping[str, Mapping[str, Iterable[float]]],
                   percentiles: Iterable[float]) -> list[dict]:
    """One row per stained sample, channel and percentile.

    ``fmos`` is ``{channel: (FMO sample name, its linear values)}``;
    ``stained`` is ``{sample: {channel: its linear values}}``, both already
    restricted to the population the gate would sit under. A channel whose
    FMO has no finite values gives no rows. A sample with no events on a
    channel reads NA (not 0) for its % positive."""
    pcts = sorted(float(p) for p in percentiles)
    rows: list[dict] = []
    cuts: dict[str, list[tuple[float, float, float]]] = {}
    for ch, (_fmo, vals) in fmos.items():
        f = _finite(vals)
        if f.size == 0:
            continue
        cuts[ch] = []
        for p in pcts:
            cut = fmo_threshold(f, percentile=p)
            cuts[ch].append((p, cut, 100.0 * float(np.mean(f >= cut))))
    for sample, chans in stained.items():
        for ch, vals in chans.items():
            if ch not in cuts:
                continue
            v = _finite(vals)
            fmo_name, fvals = fmos[ch]
            n_fmo = int(_finite(fvals).size)
            for p, cut, fmo_pct in cuts[ch]:
                pos = int(np.count_nonzero(v >= cut))
                rows.append({
                    'Sample': sample, 'Channel': ch, 'FMO': fmo_name,
                    'FMO % above (set)': round(100.0 - p, 6),
                    'Threshold': cut,
                    'Events': int(v.size), 'Positive': pos,
                    '% positive': (100.0 * pos / v.size if v.size
                                   else float('nan')),
                    'FMO events': n_fmo,
                    'FMO % above (measured)': fmo_pct,
                })
    return rows


def spread(rows: list[dict]) -> dict[tuple[str, str], float]:
    """``{(sample, channel): max - min of % positive}`` across the
    percentiles: how far the choice of threshold moves each result."""
    by: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        v = r['% positive']
        if v == v:
            by.setdefault((r['Sample'], r['Channel']), []).append(v)
    return {k: (max(v) - min(v)) for k, v in by.items() if v}
