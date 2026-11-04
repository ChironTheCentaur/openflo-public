"""Per-sample plot and gate caches for the editor (Tk-free core).

Switching the displayed or active sample must not redo work over every event
of every displayed sample on each replot: a full-frame ``dropna``, a pandas
``df.sample`` (a full ``RandomState`` shuffle of all N rows to draw 60k), the
filter-mode gate masks and a copy of every column, and the gate-row count.
Measured uncached at 20M events: 1.5 s for a plain sample click, 6.4 s in
Filter mode, 9.9 s for a gate-row click. On pandas 2.x the ``dropna``
deep-copies the whole frame as well (5.0 s, +2.2 GB per sample per replot).

This module holds what is needed to answer those questions again without
re-reading the data:

* :func:`chain_sigs` -- a content signature per gate covering everything its
  cumulative population depends on (its own geometry, its ancestors, boolean
  operands, auto-clean recipes), so a cache entry can never outlive an edit.
  Display-only keys (name, label, colour, open, enabled, and the provenance
  stamp) are left out: a rename invalidates nothing; a reparent does.
* :func:`scan` -- one chunked pass over a sample that evaluates gate masks
  (the same ``cumulative_gate_mask`` the rest of the app uses, so counts are
  identical), valid-row masks and the filter union. Chunking bounds the
  temporaries and, in the background worker, the time the GIL is held (the
  flowutils polygon test does not release it: 0.65 s per call at 20M).
* :func:`pick_rows` -- the display subsample: ``Generator.choice`` over the
  eligible rows (O(cap), not O(N)), sorted, deterministic per seed.
* :class:`CacheStore` -- an LRU bounded by bytes.

Nothing here touches Tk or matplotlib, so the background worker can run it.
"""
from __future__ import annotations

import hashlib
import json
import threading
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

# Gate keys that never change which events a gate holds. 'stamp' is the
# provenance of a gate (result_stamp): which version and step placed it;
# 'original' is a corrected gate's geometry before the fix (fixes), kept as
# a record and never applied.
DISPLAY_ONLY_KEYS = frozenset({'name', 'label', 'color', 'open', 'enabled',
                               'stamp', 'original'})

# Rows per chunk. A power of two (so chunk bounds fall on packed-byte bounds)
# small enough that the slowest gate (a 24-vertex polygon, ~115 ns/event,
# GIL held) stays near 30 ms per chunk.
CHUNK = 1 << 18


class Cancelled(Exception):
    """Raised by a checkpoint when the work it belongs to is no longer
    wanted (superseded, the editor closed, or the sample removed)."""


# ── signatures ───────────────────────────────────────────────────────────────

def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(repr(v) for v in o)
    return repr(o)


def gate_content_sig(gate: Mapping[str, Any]) -> str:
    """Canonical text of a gate's mask-relevant content."""
    body = {str(k): v for k, v in gate.items() if k not in DISPLAY_ONLY_KEYS}
    return json.dumps(body, sort_keys=True, default=_json_default,
                      separators=(',', ':'))


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode('utf-8', 'surrogatepass'),
                           digest_size=12).hexdigest()


def chain_sigs(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """``{gid: signature}`` of each gate's cumulative population.

    A gate's signature folds in its own content, its parent's signature (the
    chain ``cumulative_gate_mask`` walks) and, for a boolean, each operand's
    signature -- everything its mask depends on. A missing parent or operand
    is part of the signature too (both change the result). A cycle, which
    ``cumulative_gate_mask`` cuts where it first repeats, folds in the whole
    gate set instead, so any edit anywhere invalidates it."""
    content = {gid: gate_content_sig(g) for gid, g in gates.items()}
    whole = None
    out: dict[str, str] = {}
    visiting: set[str] = set()

    def sig(gid: str) -> str:
        nonlocal whole
        if gid in out:
            return out[gid]
        if gid not in gates:
            return f'MISSING:{gid}'
        if gid in visiting:
            if whole is None:
                whole = _digest('\n'.join(f'{k}={v}' for k, v in
                                          sorted(content.items())))
            return f'CYCLE:{whole}'
        visiting.add(gid)
        g = gates[gid]
        parts = [content[gid]]
        pid = g.get('parent_id')
        parts.append('P:' + (sig(pid) if pid is not None else '-'))
        if g.get('kind') == 'boolean':
            parts.append('O:' + ','.join(sig(o) for o in (g.get('operands') or ())))
        visiting.discard(gid)
        s = _digest('|'.join(parts))
        out[gid] = s
        return s

    for gid in gates:
        sig(gid)
    return out


def stable_seed(*parts: Any) -> int:
    """A 32-bit seed that is the same in every process (Python's ``hash`` of
    a str is randomised per process, so a seed built on it drew a different
    subsample after every restart)."""
    return zlib.crc32('\x00'.join(map(str, parts)).encode('utf-8', 'replace'))


def chunkable(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether every gate's mask is per-event, so evaluating row slices and
    concatenating equals evaluating the whole sample. False only when a
    boolean gate could reach an auto-clean gate: auto-clean is computed over
    the whole sample (time bins), and a boolean evaluates its operands without
    the precomputed override."""
    kinds = {g.get('kind') for g in gates.values()}
    return not ('boolean' in kinds and 'autoclean' in kinds)


# ── packed masks ─────────────────────────────────────────────────────────────

def pack(mask: np.ndarray) -> np.ndarray:
    """Pack a boolean mask into bits, eight rows per byte (see ``unpack``)."""
    return np.packbits(np.asarray(mask, dtype=bool))


def unpack(packed: np.ndarray, n: int, a: int = 0, b: int | None = None) -> np.ndarray:
    """Rows ``[a, b)`` of a packed mask as bool (``a`` must be a multiple of 8)."""
    b = n if b is None else b
    seg = packed[a // 8:(b + 7) // 8]
    return np.unpackbits(seg, count=b - a).view(bool)


def popcount(packed: np.ndarray) -> int:
    """Number of set bits (True rows) in a packed mask."""
    return int(np.bitwise_count(packed).sum(dtype=np.int64))


# ── the scan ─────────────────────────────────────────────────────────────────

def _notna(values: np.ndarray) -> np.ndarray:
    """What ``DataFrame.dropna`` keeps for one column: not NaN/None/NaT.
    (``inf`` is kept, as dropna keeps it.)"""
    if values.dtype.kind in 'fc':
        return ~np.isnan(values)
    if values.dtype.kind in 'biu':
        return np.ones(values.shape[0], dtype=bool)
    import pandas as pd
    return np.asarray(pd.notna(values), dtype=bool)


def _known(gids: Iterable[str], gates: Mapping[str, Any]) -> list[str]:
    """``gids`` in first-seen order, duplicates and unknown gates dropped."""
    return [g for g in dict.fromkeys(gids) if g in gates]


def _join(parts: list[np.ndarray]) -> np.ndarray:
    """Packed chunk masks joined into one (empty when there were no chunks)."""
    return np.concatenate(parts) if parts else np.zeros(0, np.uint8)


def _scan_chunk_size(gates: Mapping[str, Any], n: int, chunk: int) -> int:
    """Rows per slice: the whole sample when the gates cannot be chunked,
    rounded up to a packed-byte bound (a multiple of 8, at least 8)."""
    if gates and not chunkable(gates):
        chunk = n                     # one slice: the whole sample
    return max(8, (int(chunk) + 7) // 8 * 8)


class _ScanParts:
    """What :func:`scan` gathers chunk by chunk, and the dict it returns."""

    def __init__(self, count_gids: list[str], keep_gids: list[str],
                 union_list: list[str] | None, vcols: list[str] | None):
        self.count_gids = count_gids
        self.keep_gids = keep_gids
        self.union_list = union_list
        self.vcols = vcols
        self.counts = dict.fromkeys(count_gids, 0)
        self.kept: dict[str, list[np.ndarray]] = {g: [] for g in keep_gids}
        self.union: list[np.ndarray] = []
        self.union_n = 0
        self.valid: list[np.ndarray] = []
        self.valid_n = 0

    @property
    def need_gates(self) -> bool:
        return bool(self.count_gids or self.keep_gids or self.union_list)

    def add_gates(self, cum: Callable[[str], np.ndarray], size: int) -> None:
        for gid in self.count_gids:
            self.counts[gid] += int(np.count_nonzero(cum(gid)))
        for gid in self.keep_gids:
            self.kept[gid].append(pack(cum(gid)))
        if self.union_list is not None:
            um = np.zeros(size, dtype=bool)
            for gid in self.union_list:
                um |= cum(gid)
            self.union_n += int(np.count_nonzero(um))
            self.union.append(pack(um))

    def add_valid(self, part, vcols: list[str]) -> None:
        vm = np.ones(len(part), dtype=bool)
        for c in vcols:
            vm &= _notna(np.asarray(part[c].to_numpy()))
        self.valid_n += int(np.count_nonzero(vm))
        self.valid.append(pack(vm))

    def result(self, n: int) -> dict[str, Any]:
        out: dict[str, Any] = {'n': n}
        if self.count_gids:
            out['counts'] = self.counts
        if self.keep_gids:
            out['masks'] = {g: _join(p) for g, p in self.kept.items()}
        if self.union_list is not None:
            out['union'] = (_join(self.union), self.union_n)
        if self.vcols is not None:
            out['valid'] = ((None, n) if self.valid_n == n
                            else (np.concatenate(self.valid), self.valid_n))
        return out


def _chunk_cum(gates: Mapping[str, Mapping[str, Any]], part, n: int,
               a: int, b: int, seeds: Mapping[str, np.ndarray],
               overrides: Mapping[str, np.ndarray] | None
               ) -> Callable[[str], np.ndarray]:
    """Cumulative mask of a gid over rows ``[a, b)`` (``part``), sharing one
    cache per chunk seeded from the known packed ``seeds``."""
    from .gate_eval import cumulative_gate_mask
    cache: dict[str, np.ndarray] = {
        gid: unpack(p, n, a, b) for gid, p in seeds.items()}
    ov = (None if not overrides else
          {gid: np.asarray(m)[a:b] for gid, m in overrides.items()})

    def cum(gid: str) -> np.ndarray:
        return np.asarray(cumulative_gate_mask(
            gates, gid, part, overrides=ov, cache=cache), dtype=bool)
    return cum


def scan(frame, *, gates: Mapping[str, Mapping[str, Any]] | None = None,
         count_gids: Iterable[str] = (), keep_gids: Iterable[str] = (),
         union_gids: Iterable[str] | None = None,
         valid_cols: Iterable[str] | None = None,
         overrides: Mapping[str, np.ndarray] | None = None,
         seed_masks: Mapping[str, np.ndarray] | None = None,
         chunk: int = CHUNK,
         checkpoint: Callable[[], None] | None = None) -> dict[str, Any]:
    """One pass over ``frame`` (all of its rows), chunk by chunk.

    Returns a dict with any of:

    * ``'counts'``: ``{gid: events in its cumulative mask}`` for ``count_gids``
    * ``'masks'``:  ``{gid: packed cumulative mask}`` for ``keep_gids``
    * ``'union'``:  ``(packed, count)`` -- OR of ``union_gids``' cumulative masks
    * ``'valid'``:  ``(packed or None, count)`` -- rows not NA in every one of
      ``valid_cols`` (``None`` when that is every row)

    Masks come from ``gate_eval.cumulative_gate_mask`` with the given auto-clean
    ``overrides`` (full-length arrays, sliced per chunk), so they are the
    masks the rest of the app computes. ``seed_masks`` are known cumulative
    masks (packed, keyed by gid) a chain walk can stop at. ``checkpoint`` is
    called between chunks; it may raise :class:`Cancelled`."""
    n = len(frame)
    gates = gates or {}
    count_list = _known(count_gids, gates)
    keep_list = _known(keep_gids, gates)
    union_list = None if union_gids is None else _known(union_gids, gates)
    vcols = None if valid_cols is None else [c for c in valid_cols
                                             if c in frame.columns]
    acc = _ScanParts(count_list, keep_list, union_list, vcols)
    chunk = _scan_chunk_size(gates, n, chunk)
    seeds = dict(seed_masks or {})
    for a in range(0, n, chunk):
        if checkpoint is not None:
            checkpoint()
        b = min(n, a + chunk)
        part = frame.iloc[a:b]
        if acc.need_gates:
            acc.add_gates(_chunk_cum(gates, part, n, a, b, seeds, overrides),
                          b - a)
        if vcols is not None:
            acc.add_valid(part, vcols)
    return acc.result(n)


def pick_rows(n: int, eligible: np.ndarray | None, m: int, cap: int | None,
              seed: int | None, chunk: int = CHUNK,
              checkpoint: Callable[[], None] | None = None) -> np.ndarray | None:
    """Row positions to draw: ``cap`` of the ``m`` eligible rows chosen by
    ``Generator(seed).choice`` and returned sorted; every eligible row when
    ``cap`` is None or ``m <= cap``. ``eligible`` is a packed mask, or None for
    "every row" (``m == n``). Returns None to mean "all ``n`` rows, in order"
    (nothing to take)."""
    if eligible is None and (cap is None or n <= cap):
        return None
    if cap is not None and m > cap:
        picks = np.sort(np.random.default_rng(seed).choice(
            m, int(cap), replace=False)).astype(np.intp, copy=False)
        if eligible is None:
            return picks
    else:
        picks = None
    out: list[np.ndarray] = []
    base = 0
    j = 0
    chunk = max(8, (int(chunk) + 7) // 8 * 8)
    for a in range(0, n, chunk):
        if checkpoint is not None:
            checkpoint()
        b = min(n, a + chunk)
        assert eligible is not None
        idx = np.flatnonzero(unpack(eligible, n, a, b))
        if picks is None:
            out.append(idx + a)
        else:
            k = int(np.searchsorted(picks, base + idx.size, side='left'))
            if k > j:
                out.append(idx[picks[j:k] - base] + a)
            j = k
        base += idx.size
    if not out:
        return np.zeros(0, dtype=np.intp)
    return np.concatenate(out).astype(np.intp, copy=False)


def edges_digest(edges) -> str:
    """Key for a set of histogram bin edges (exact bytes)."""
    return hashlib.blake2b(np.ascontiguousarray(edges, dtype=np.float64).tobytes(),
                           digest_size=12).hexdigest()


def screen_uniform_edges(tm: str, view_scale: str, lin_scale: str,
                         lo: float, hi: float, n_bins: int) -> np.ndarray | None:
    """``PlotMixin._screen_uniform_edges`` for the scales whose edges do not
    depend on the data (everything but symlog), callable off the Tk thread.
    None where they would (the caller then leaves the counts to the plot).
    A result is only ever stored under the digest of the edges the plot itself
    computes, so a divergence here costs a cache miss, never a wrong count."""
    from .plotmath import hist_bin_edges
    from .scales import view_funcs
    funcs = view_funcs(tm, view_scale, None)
    if funcs is None:
        if lin_scale == 'symlog':
            return None
        return np.asarray(hist_bin_edges(lo, hi, lin_scale, n_bins), dtype=float)
    if view_scale == 'symlog':
        return None
    fwd, inv = funcs
    slo = float(np.asarray(fwd(np.array([lo], dtype=float)))[0])
    shi = float(np.asarray(fwd(np.array([hi], dtype=float)))[0])
    if not (np.isfinite(slo) and np.isfinite(shi)) or shi <= slo:
        return np.asarray(hist_bin_edges(lo, hi, 'linear', n_bins), dtype=float)
    screen = np.linspace(slo, shi, int(n_bins) + 1)
    edges = np.unique(np.asarray(inv(screen), dtype=float))
    edges = edges[np.isfinite(edges)]
    if edges.size < 2:
        return np.asarray(hist_bin_edges(lo, hi, 'linear', n_bins), dtype=float)
    return np.asarray(edges.tolist(), dtype=float)


# ── store ────────────────────────────────────────────────────────────────────

def nbytes_of(value: Any) -> int:
    """Approximate retained bytes of a cache value."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + 112
    if isinstance(value, (tuple, list)):
        return 64 + sum(nbytes_of(v) for v in value)
    if isinstance(value, dict):
        return 240 + sum(nbytes_of(v) + 64 for v in value.values())
    return 64


class CacheStore:
    """An LRU bounded by bytes. Keys are tuples whose first item is the
    sample name, so a sample's entries can be dropped together. Used from the
    Tk thread only (the worker hands its results back through a queue)."""

    def __init__(self, budget_bytes: int = 192 << 20):
        self.budget = int(budget_bytes)
        self._d: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self.nbytes = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: tuple) -> bool:
        return key in self._d

    def get(self, key: tuple, default: Any = None) -> Any:
        """The value cached under ``key``, marking it most recently used, or
        ``default`` when absent."""
        with self._lock:
            ent = self._d.get(key)
            if ent is None:
                return default
            self._d.move_to_end(key)
            return ent[0]

    def put(self, key: tuple, value: Any, nbytes: int | None = None) -> None:
        """Cache ``value`` under ``key`` (size ``nbytes``, estimated when None),
        evicting least recently used entries to stay within budget. A value larger
        than the whole budget is not stored."""
        size = nbytes_of(value) if nbytes is None else int(nbytes)
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.nbytes -= old[1]
            if size > self.budget:
                return
            self._d[key] = (value, size)
            self.nbytes += size
            while self.nbytes > self.budget and self._d:
                _k, (_v, s) = self._d.popitem(last=False)
                self.nbytes -= s

    def purge(self, name: str | None = None) -> int:
        """Drop every entry of sample ``name`` (all entries when None)."""
        with self._lock:
            if name is None:
                k = len(self._d)
                self._d.clear()
                self.nbytes = 0
                return k
            dead = [k for k in self._d if k and k[0] == name]
            for k in dead:
                self.nbytes -= self._d.pop(k)[1]
            return len(dead)

    def items(self, name: str) -> list[tuple[tuple, Any]]:
        """Sample ``name``'s ``(key, value)`` entries, least recent first."""
        with self._lock:
            return [(k, v[0]) for k, v in self._d.items() if k and k[0] == name]

    def names(self) -> set:
        """The sample names that have at least one cached entry."""
        with self._lock:
            return {k[0] for k in self._d if k}
