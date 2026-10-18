"""Per-sample plot and gate caches for the editor (Tk-free core).

Switching the displayed or active sample used to redo work over every event of
every displayed sample on each replot: a full-frame ``dropna``, a pandas
``df.sample`` (a full ``RandomState`` shuffle of all N rows to draw 60k), the
filter-mode gate masks and a copy of every column, and the gate-row count
(twice, uncached). Measured at 20M events: 1.5 s for a plain sample click,
6.4 s in Filter mode, 9.9 s for a gate-row click. On pandas 2.x the ``dropna``
deep-copied the whole frame as well (5.0 s, +2.2 GB per sample per replot).

This module holds what is needed to answer those questions again without
re-reading the data:

* :func:`chain_sigs` -- a content signature per gate covering everything its
  cumulative population depends on (its own geometry, its ancestors, boolean
  operands, auto-clean recipes), so a cache entry can never outlive an edit.
  Display-only keys (name, label, colour, open, enabled) are left out: a
  rename invalidates nothing; a reparent does.
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

# Gate keys that never change which events a gate holds.
DISPLAY_ONLY_KEYS = frozenset({'name', 'label', 'color', 'open', 'enabled'})

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
    return np.packbits(np.asarray(mask, dtype=bool))


def unpack(packed: np.ndarray, n: int, a: int = 0, b: int | None = None) -> np.ndarray:
    """Rows ``[a, b)`` of a packed mask as bool (``a`` must be a multiple of 8)."""
    b = n if b is None else b
    seg = packed[a // 8:(b + 7) // 8]
    return np.unpackbits(seg, count=b - a).view(bool)


def popcount(packed: np.ndarray) -> int:
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

    Masks come from ``pipeline.cumulative_gate_mask`` with the given auto-clean
    ``overrides`` (full-length arrays, sliced per chunk), so they are the
    masks the rest of the app computes. ``seed_masks`` are known cumulative
    masks (packed, keyed by gid) a chain walk can stop at. ``checkpoint`` is
    called between chunks; it may raise :class:`Cancelled`."""
    from .pipeline import cumulative_gate_mask
    n = len(frame)
    gates = gates or {}
    count_gids = [g for g in dict.fromkeys(count_gids) if g in gates]
    keep_gids = [g for g in dict.fromkeys(keep_gids) if g in gates]
    union_list = (None if union_gids is None
                  else [g for g in dict.fromkeys(union_gids) if g in gates])
    vcols = None if valid_cols is None else [c for c in valid_cols
                                             if c in frame.columns]
    if gates and not chunkable(gates):
        chunk = n                     # one slice: the whole sample
    # Slice bounds must fall on packed-byte bounds (multiples of 8).
    chunk = max(8, (int(chunk) + 7) // 8 * 8)
    counts = dict.fromkeys(count_gids, 0)
    kept_parts: dict[str, list[np.ndarray]] = {g: [] for g in keep_gids}
    union_parts: list[np.ndarray] = []
    union_n = 0
    valid_parts: list[np.ndarray] = []
    valid_n = 0
    seeds = dict(seed_masks or {})
    need_gates = bool(count_gids or keep_gids or union_list)
    for a in range(0, max(n, 1), chunk):
        if n == 0:
            break
        if checkpoint is not None:
            checkpoint()
        b = min(n, a + chunk)
        part = frame.iloc[a:b]
        if need_gates:
            cache: dict[str, np.ndarray] = {
                gid: unpack(p, n, a, b) for gid, p in seeds.items()}
            ov = (None if not overrides else
                  {gid: np.asarray(m)[a:b] for gid, m in overrides.items()})

            def cum(gid, _part=part, _ov=ov, _cache=cache):
                return np.asarray(cumulative_gate_mask(
                    gates, gid, _part, overrides=_ov, cache=_cache), dtype=bool)

            for gid in count_gids:
                counts[gid] += int(np.count_nonzero(cum(gid)))
            for gid in keep_gids:
                kept_parts[gid].append(pack(cum(gid)))
            if union_list is not None:
                um = np.zeros(b - a, dtype=bool)
                for gid in union_list:
                    um |= cum(gid)
                union_n += int(np.count_nonzero(um))
                union_parts.append(pack(um))
        if vcols is not None:
            vm = np.ones(b - a, dtype=bool)
            for c in vcols:
                vm &= _notna(np.asarray(part[c].to_numpy()))
            valid_n += int(np.count_nonzero(vm))
            valid_parts.append(pack(vm))
    out: dict[str, Any] = {'n': n}
    if count_gids:
        out['counts'] = counts
    if keep_gids:
        out['masks'] = {g: (np.concatenate(p) if p else np.zeros(0, np.uint8))
                        for g, p in kept_parts.items()}
    if union_list is not None:
        out['union'] = ((np.concatenate(union_parts) if union_parts
                         else np.zeros(0, np.uint8)), union_n)
    if vcols is not None:
        if valid_n == n:
            out['valid'] = (None, n)
        else:
            out['valid'] = (np.concatenate(valid_parts), valid_n)
    return out


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
        with self._lock:
            ent = self._d.get(key)
            if ent is None:
                return default
            self._d.move_to_end(key)
            return ent[0]

    def put(self, key: tuple, value: Any, nbytes: int | None = None) -> None:
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

    def names(self) -> set:
        with self._lock:
            return {k[0] for k in self._d if k}
