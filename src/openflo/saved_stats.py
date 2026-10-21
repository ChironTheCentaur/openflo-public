"""Gate counts and statistics kept with the session (Tk-free core).

Opening a session used to count every gate of every sample again, and the
Statistics window computed every row again, though nothing they depend on had
changed since the save. Each sample entry now carries the numbers it had
(``saved_stats``), and a reopen puts them back into the editor's caches
(:mod:`openflo.editor_plotcache`) when, and only when, everything they depend
on is the same:

* the **data**: a digest of the sample's whole frame (``_frame_digest``, the
  one the sidecar writer already takes). A different compensation, transform,
  QC or event set is a different digest, and nothing saved is used;
* the **gates**: each count is keyed by its gate's chain signature
  (``plotcache.chain_sigs``), so an edit to the gate, an ancestor or an
  operand misses on its own;
* the **statistic settings**: which statistics, channels, labels and
  percentiles, the sample's transform record, and every Preference that
  changes a formula (:func:`stats_settings_key`);
* the **OpenFlo version** that computed them. A new version never serves an
  old number: whether an old result needs correcting is the fix registry's
  call, not the cache's.

Only numbers are saved. A statistics row's names (the population path, its
identity key, why it is unknown) depend on gate names and are rebuilt from
the gates as they are when the row is served. Rendered plots are not saved:
they redraw from the data quickly.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

# Preferences that change what a statistic's formula gives (an n-1 SD, a
# geometric-mean method, ...). Add one name here and every saved and cached
# statistic is keyed by its value: GeoMean's and rSD/rCV's forms (prefs.py).
STATS_PREF_KEYS: tuple[str, ...] = ('stat_geomean', 'stat_rcv')

# Statistics saved per sample: the most recent sets of settings (each tick of
# a column in the Statistics window is a new set).
MAX_STATS_SETS = 6

# How an infinite value is saved (JSON has none): ``{'~inf': '+'}``.
_INF = '~inf'

# Row keys that are names, not numbers: rebuilt from the gates when served.
_NAME_KEYS = frozenset({'Sample', 'Population', '__gid__', '__key__',
                        '__unknown__'})


def stats_settings_key(prefs: Mapping[str, Any] | None) -> dict[str, Any]:
    """The Preferences that change statistic values, as ``{key: value}``."""
    prefs = prefs if isinstance(prefs, Mapping) else {}
    return {k: prefs.get(k) for k in STATS_PREF_KEYS}


def stats_key(*, sigs: Mapping[str, str], order: Iterable[str],
              select: Iterable[str] | None, labels: Mapping[str, str],
              channels: Iterable[str], want: Iterable[str],
              stat_chan: Iterable[str], transforms: Mapping | None,
              settings: Mapping[str, Any], version: str | None) -> str:
    """The key of one sample's statistics rows: everything
    ``gating.population_stats`` reads apart from the data itself and the
    gates' names. ``sigs`` are the sample's chain signatures."""
    order = [str(g) for g in order]
    channels = [str(c) for c in channels]
    body = {
        'gates': [[g, sigs.get(g)] for g in order],
        'select': None if select is None else [str(g) for g in select],
        'labels': {c: str(labels.get(c, c)) for c in channels},
        'channels': channels,
        'want': sorted(str(w) for w in want),
        'stat_chan': [str(s) for s in stat_chan],
        'transforms': {str(c): v for c, v in (transforms or {}).items()
                       if c in channels},
        'settings': dict(settings),
        'version': version,
    }
    text = json.dumps(body, sort_keys=True, default=repr,
                      separators=(',', ':'))
    return hashlib.blake2b(text.encode('utf-8', 'surrogatepass'),
                           digest_size=16).hexdigest()


# ── rows <-> numbers ─────────────────────────────────────────────────────────

def _enc(v: Any) -> Any:
    if hasattr(v, 'item') and not isinstance(v, (str, bytes)):
        try:
            v = v.item()                          # a numpy scalar
        except (TypeError, ValueError):
            pass
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if math.isinf(v):
            # JSON has no infinity, and the session writer would store the
            # open-bound sentinel (1e12) in its place: a different number.
            return {_INF: '+' if v > 0 else '-'}
        return v
    if isinstance(v, Mapping):
        return {str(k): _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    return float(v)


def _dec(v: Any) -> Any:
    if v is None:
        return float('nan')
    if isinstance(v, dict):
        if len(v) == 1 and _INF in v:
            return float('inf') if v[_INF] == '+' else float('-inf')
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [x if isinstance(x, str) else _dec(x) for x in v]
    return v


def encode_rows(rows: Iterable[Mapping[str, Any]]) -> list:
    """``[[gid, {column: number}], ...]``: a row set's numbers, JSON-ready
    (NaN as null, an infinity as ``{'~inf': '+'}`` or ``'-'``)."""
    return [[str(r.get('__gid__')),
             {str(k): _enc(v) for k, v in r.items() if k not in _NAME_KEYS}]
            for r in rows]


def decode_rows(enc: list, sample_name: str, gates: Mapping[str, Any],
                labels: Mapping[str, str], transforms: Mapping | None,
                columns: Iterable[str]) -> list[dict] | None:
    """Rows as ``population_stats`` returns them, from :func:`encode_rows`
    numbers and the gates as they are now. None if a gate is gone."""
    from .gating import population_keys, population_path
    from .pipeline import unknown_population_reason
    keys = population_keys(gates, labels)
    cols = list(columns)
    out = []
    for item in enc:
        gid, vals = item[0], item[1]
        if gid not in gates:
            return None
        row: dict[str, Any] = {
            'Sample': sample_name,
            'Population': population_path(gates, gid, transforms=transforms),
            '__gid__': gid,
            '__key__': keys.get(gid, str(gid))}
        why = unknown_population_reason(gates, gid, columns=cols)
        if why:
            row['__unknown__'] = why
        row.update({k: _dec(v) for k, v in vals.items()})
        out.append(row)
    return out


# ── the session entry ────────────────────────────────────────────────────────

def make_entry(version: str | None, digest: str, counts: Mapping[str, int],
               stats: Mapping[str, list]) -> dict:
    """A sample entry's ``saved_stats``: ``{'app_version', 'data',
    'counts': {chain signature: events}, 'stats': {stats key: rows}}``."""
    return {'app_version': version, 'data': digest,
            'counts': {str(s): int(c) for s, c in counts.items()},
            'stats': {str(k): v for k, v in stats.items()}}


def usable(entry: Any, digest: str | None, version: str | None) -> bool:
    """Whether a saved entry was computed by this version on this data."""
    return (isinstance(entry, dict) and digest is not None
            and version is not None
            and entry.get('app_version') == version
            and entry.get('data') == digest)


def entry_counts(entry: Mapping[str, Any]) -> dict[str, int]:
    """The entry's counts, malformed items dropped."""
    out = {}
    c = entry.get('counts')
    for sig, n in (c.items() if isinstance(c, dict) else ()):
        if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
            out[str(sig)] = n
    return out


def entry_stats(entry: Mapping[str, Any]) -> dict[str, list]:
    """The entry's statistics rows, malformed sets dropped."""
    out = {}
    st = entry.get('stats')
    for key, enc in (st.items() if isinstance(st, dict) else ()):
        if isinstance(enc, list) and all(
                isinstance(i, list) and len(i) == 2 and isinstance(i[0], str)
                and isinstance(i[1], dict) for i in enc):
            out[str(key)] = enc
    return out
