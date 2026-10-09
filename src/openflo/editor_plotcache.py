"""Cached per-sample display work, and its background warm-up -- editor mixin.

What a replot needs from a sample (which rows are valid on the plotted axes,
the filter population, the display subsample, the gate counts the tree row
shows, the histogram of the X channel) used to be recomputed over every event
on every replot. It is now computed once per sample and kept in a byte-bounded
LRU (:class:`openflo.plotcache.CacheStore`), keyed by everything it depends on:

* the sample's **data version** -- a token over the FlowSample and DataFrame
  identity, length, columns, each column's buffer, a probe of each column's
  values, and an explicit counter bumped by every editor path that rewrites
  data (``_data_changed``). Any change purges the sample's entries;
* the **gate signatures** (``plotcache.chain_sigs``): content hashes, so any
  edit to a gate, its ancestors or its operands misses the cache on its own,
  with no call site to remember;
* the channels, the population (all / filter union / one gate), the cap and
  the subsample seed.

Counts and masks are computed on ALL events with the same
``cumulative_gate_mask`` as before, so every number is unchanged; only which
capped subsample is drawn differs (now ``Generator.choice``, seeded stably per
sample/axes/cap, instead of a full ``RandomState`` shuffle).

After the displayed samples are drawn, the rest are warmed in the background
(active sample first, then tree order) by one worker thread. The worker gets
snapshots (a shallow frame copy, a deep copy of the gates), runs chunked numpy
(<= ~30 ms of GIL per chunk at 20M), never calls Tk or matplotlib, and hands
results back through a queue the Tk loop polls. A result is stored only if the
sample's data version is still the one it was computed against.
"""
from __future__ import annotations

import copy
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from . import plotcache as _pcm
from .editor_base import EditorMixin

_HIST_BINS = 256
_UNCACHED_ROWS = 2_000_000      # don't keep row lists longer than this


class _WarmWorker:
    """One background thread running one task at a time."""

    def __init__(self):
        self._cv = threading.Condition()
        self._task: tuple[int, Any] | None = None
        self._busy: int | None = None
        self._cancel: set[int] = set()
        self._stop = False
        self._paused = 0                 # nesting count of foreground holds
        self.results: queue.SimpleQueue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name='openflo-plot-warmup')
        self.thread.start()

    @property
    def idle(self) -> bool:
        with self._cv:
            return self._task is None and self._busy is None

    def submit(self, tid: int, fn) -> None:
        with self._cv:
            self._task = (tid, fn)
            self._cv.notify_all()

    def cancel(self, tid: int) -> None:
        with self._cv:
            if self._task is not None and self._task[0] == tid:
                self._task = None
                self.results.put((tid, 'cancelled', None))
            elif self._busy == tid:
                self._cancel.add(tid)
            self._cv.notify_all()

    def pause(self) -> None:
        with self._cv:
            self._paused += 1

    def resume(self) -> None:
        with self._cv:
            self._paused = max(0, self._paused - 1)
            self._cv.notify_all()

    def checkpoint(self, tid: int) -> None:
        with self._cv:
            while self._paused and not self._stop and tid not in self._cancel:
                self._cv.wait(0.05)
            if self._stop or tid in self._cancel:
                raise _pcm.Cancelled()

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._task = None
            self._cv.notify_all()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._stop and self._task is None:
                    self._cv.wait()
                if self._stop:
                    return
                assert self._task is not None
                tid, fn = self._task
                self._task = None
                self._busy = tid
            try:
                out = fn(lambda _t=tid: self.checkpoint(_t))
                self.results.put((tid, 'ok', out))
            except _pcm.Cancelled:
                self.results.put((tid, 'cancelled', None))
            except Exception as exc:                       # noqa: BLE001
                self.results.put((tid, 'error', exc))
            finally:
                with self._cv:
                    self._busy = None
                    self._cancel.discard(tid)


class PlotCacheMixin(EditorMixin):
    """Per-sample caches behind ``_get_df`` / gate counts / histograms, and
    the background warm-up that fills them for samples not yet shown."""

    # ── state ────────────────────────────────────────────────────────────

    def _pc_store(self) -> _pcm.CacheStore:
        st = getattr(self, '_pc_cache', None)
        if st is None:
            st = self._pc_cache = _pcm.CacheStore()
            self._pc_explicit = {}       # name -> explicit data-change counter
            self._pc_tokens = {}         # name -> (token, version int)
            self._pc_seq = 0
        return st

    def _pc_token(self, name):
        s = self._samples.get(name)
        df = getattr(s, 'data', None)
        if df is None:
            return None
        n = len(df)
        probe = (np.linspace(0, n - 1, 64).astype(np.intp) if n
                 else np.zeros(0, np.intp))
        cols = []
        for c in df.columns:
            try:
                arr = df[c].to_numpy()
                ai = arr.__array_interface__
                ptr = (ai['data'][0], arr.dtype.str, ai.get('strides'))
                vals = arr[probe] if n else arr
                pv = (vals.tobytes() if arr.dtype.kind != 'O'
                      else repr(vals.tolist()))
            except Exception:                              # noqa: BLE001
                ptr, pv = ('?', id(df[c])), None
            cols.append((str(c), ptr, pv))
        return (id(s), id(df), n, self._pc_explicit.get(name, 0), tuple(cols))

    def _pc_version(self, name) -> int | None:
        """The sample's data version: a small int that changes whenever its
        data does (and then drops every cache entry of the sample)."""
        store = self._pc_store()
        tok = self._pc_token(name)
        if tok is None:
            return None
        cur = self._pc_tokens.get(name)
        if cur is not None and cur[0] == tok:
            return cur[1]
        self._pc_seq += 1
        if cur is not None:
            store.purge(name)
        self._pc_tokens[name] = (tok, self._pc_seq)
        return self._pc_seq

    def _pc_sigs(self, name) -> dict:
        return _pcm.chain_sigs(self._sample_gates.get(name, {}) or {})

    def _data_changed(self, names=None):
        """Sample data was replaced or rewritten in place (compensation,
        transforms, calibration, unmixing, clustering, QC/downsample trims...):
        drop everything cached for it and warm it again. Safe to over-call."""
        self._pc_store()
        if names is None:
            names = list(self._samples)
        elif isinstance(names, str):
            names = [names]
        for name in names:
            self._pc_explicit[name] = self._pc_explicit.get(name, 0) + 1
            self._pc_cache.purge(name)
            self._pc_tokens.pop(name, None)
            for c in (getattr(self, '_ac_cache', {}),
                      getattr(self, '_ac_count_cache', {}),
                      getattr(self, '_ac_method_cache', {})):
                for ck in [k for k in list(c) if k[0] == name]:
                    c.pop(ck, None)
            self._warm_cancel_sample(name)
        # Tk only from the Tk thread; off it, the next replot replans.
        if threading.current_thread() is threading.main_thread():
            self._warm_schedule()

    def _pc_forget(self, name):
        """A sample was removed: no entry, counter or queued work may stay."""
        self._pc_store().purge(name)
        self._pc_explicit.pop(name, None)
        self._pc_tokens.pop(name, None)
        self._warm_cancel_sample(name)
        q = getattr(self, '_pc_warm_names', None)
        if q is not None:
            try:
                q.remove(name)
            except ValueError:
                pass

    # ── populations ──────────────────────────────────────────────────────

    def _pc_popspec(self, name, gate_parent='auto'):
        """``(key, kind, gids)`` of the population ``_get_df`` filters to."""
        gates = self._sample_gates.get(name, {}) or {}
        if gate_parent != 'auto':
            if gate_parent is not None and gate_parent in gates:
                sig = self._pc_sigs(name)[gate_parent]
                return (('gate', sig), 'gate', [gate_parent])
            return (('all',), 'all', [])
        if self.apply_gates_var.get() and gates:
            enabled = [gid for gid, g in gates.items() if g.get('enabled', True)]
            if enabled:
                sigs = self._pc_sigs(name)
                return (('union', tuple(sorted(sigs[g] for g in enabled))),
                        'union', enabled)
        return (('all',), 'all', [])

    def _pc_internal(self, gates) -> set:
        """Gates other gates build on (parents, boolean operands): the masks
        worth keeping, packed, so an edit below them re-evaluates only the
        edited part."""
        out = set()
        for g in gates.values():
            pid = g.get('parent_id')
            if pid in gates:
                out.add(pid)
            if g.get('kind') == 'boolean':
                out.update(o for o in (g.get('operands') or ()) if o in gates)
        return out

    @staticmethod
    def _pc_chain(gates, gid) -> list:
        chain, seen, cur = [], set(), gid
        while cur is not None and cur in gates and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = gates[cur].get('parent_id')
        return chain

    def _pc_overrides(self, name, frame=None) -> dict | None:
        """``{gid: full-length keep mask}`` for the sample's auto-clean gates,
        via (and filling) the editor's existing ``_ac_cache``. ``frame`` is the
        sample's data (default: current)."""
        import pandas as pd

        from .pipeline import autoclean_keep_mask, autoclean_methods_signature
        gates = self._sample_gates.get(name, {}) or {}
        ac = [gid for gid, g in gates.items() if g.get('kind') == 'autoclean']
        if not ac:
            return None
        sd = self._samples[name].data if frame is None else frame
        out = {}
        for gid in ac:
            g = gates[gid]
            sig = autoclean_methods_signature(g)
            ent = self._ac_cache.get((name, gid))
            if ent is not None and ent[0] == id(sd) and ent[1] == sig:
                out[gid] = np.asarray(ent[2].to_numpy(), dtype=bool)
            else:
                full = np.asarray(autoclean_keep_mask(g, sd), dtype=bool)
                self._ac_cache[(name, gid)] = (id(sd), sig,
                                               pd.Series(full, index=sd.index))
                out[gid] = full
        return out

    def _pc_eval(self, name, gids, *, union=None):
        """Evaluate (and cache) the cumulative masks of ``gids`` -- counts for
        each gate on their chains, packed masks for the gates others build on,
        and the OR of ``union`` when given. Tk thread, foreground."""
        store = self._pc_store()
        ver = self._pc_version(name)
        gates = self._sample_gates.get(name, {}) or {}
        sigs = self._pc_sigs(name)
        chains = set()
        for gid in list(gids) + list(union or ()):
            chains.update(self._pc_chain(gates, gid))
        internal = self._pc_internal(gates)
        seeds = {}
        for gid in chains:
            p = store.get((name, ver, 'mask', sigs[gid]))
            if p is not None:
                seeds[gid] = p
        count = [g for g in chains
                 if (name, ver, 'count', sigs[g]) not in store]
        keep = [g for g in chains if g in internal and g not in seeds]
        if union is not None:
            ukey = (name, ver, 'union', tuple(sorted(sigs[g] for g in union)))
            want_union = ukey not in store
        else:
            ukey, want_union = None, False
        if not (count or keep or want_union):
            return
        df = self._samples[name].data
        self._warm_pause()            # foreground first: hold the worker
        try:
            res = _pcm.scan(df, gates=gates, count_gids=count, keep_gids=keep,
                            union_gids=(union if want_union else None),
                            overrides=self._pc_overrides(name),
                            seed_masks=seeds)
        finally:
            self._warm_resume()
        self._pc_put_scan(name, ver, sigs, res, ukey)

    def _pc_put_scan(self, name, ver, sigs, res, ukey=None):
        store = self._pc_store()
        for gid, c in (res.get('counts') or {}).items():
            store.put((name, ver, 'count', sigs[gid]), int(c), 64)
        for gid, p in (res.get('masks') or {}).items():
            store.put((name, ver, 'mask', sigs[gid]), p)
        if ukey is not None and 'union' in res:
            store.put(ukey, res['union'])

    def _pc_counts(self, name, gids) -> dict:
        """``{gid: events in its cumulative population}`` on all events."""
        store = self._pc_store()
        ver = self._pc_version(name)
        sigs = self._pc_sigs(name)
        out, missing = {}, []
        for gid in gids:
            c = store.get((name, ver, 'count', sigs[gid]))
            if c is None:
                p = store.get((name, ver, 'mask', sigs[gid]))
                if p is not None:
                    c = _pcm.popcount(p)
                    store.put((name, ver, 'count', sigs[gid]), c, 64)
            if c is None:
                missing.append(gid)
            else:
                out[gid] = c
        if missing:
            self._pc_eval(name, missing)
            for gid in missing:
                c = store.get((name, ver, 'count', sigs[gid]))
                if c is None:                 # evicted at once (tiny budget)
                    c = int(np.count_nonzero(self._pc_mask(name, gid)))
                out[gid] = c
        return out

    def _pc_packed_mask(self, name, gid) -> np.ndarray:
        """The cumulative mask of one gate on all events, packed (cached)."""
        store = self._pc_store()
        ver = self._pc_version(name)
        sigs = self._pc_sigs(name)
        p = store.get((name, ver, 'mask', sigs[gid]))
        if p is None:
            gates = self._sample_gates.get(name, {}) or {}
            seeds = {g: store.get((name, ver, 'mask', sigs[g]))
                     for g in self._pc_chain(gates, gid)}
            seeds = {g: m for g, m in seeds.items() if m is not None}
            res = _pcm.scan(self._samples[name].data, gates=gates,
                            keep_gids=[gid], count_gids=[gid],
                            overrides=self._pc_overrides(name),
                            seed_masks=seeds)
            self._pc_put_scan(name, ver, sigs, res)
            p = res['masks'][gid]
        return p

    def _pc_mask(self, name, gid) -> np.ndarray:
        """The cumulative mask of one gate on all events (unpacked)."""
        return _pcm.unpack(self._pc_packed_mask(name, gid),
                           len(self._samples[name].data))

    def _pc_mask_on(self, name, gid, frame) -> np.ndarray:
        """The cumulative mask of ``gid`` on the rows of ``frame`` (a frame
        ``_get_df`` returned for ``name``), from the all-events mask."""
        import pandas as pd

        from .pipeline import cumulative_gate_mask
        s = self._samples[name]
        gates = self._sample_gates.get(name, {}) or {}
        if not _pcm.chunkable(gates):
            # Not per-event (a boolean over auto-clean): evaluate on the
            # frame itself, as before.
            return np.asarray(cumulative_gate_mask(
                gates, gid, frame,
                overrides=self._autoclean_overrides(name, frame)), dtype=bool)
        full = self._pc_mask(name, gid)
        if len(frame) == len(s.data):
            return full
        idx = s.data.index
        if (isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1):
            pos = np.asarray(frame.index, dtype=np.intp)
        else:
            pos = idx.get_indexer(frame.index)
        return full[pos]

    def _pc_valid(self, name, cols):
        """``(packed or None, count)``: rows not NA in every one of ``cols``
        (what ``dropna(subset=cols)`` keeps). ``None`` = every row."""
        store = self._pc_store()
        ver = self._pc_version(name)
        key = (name, ver, 'valid', tuple(cols))
        v = store.get(key)
        if v is None:
            v = _pcm.scan(self._samples[name].data, valid_cols=cols)['valid']
            store.put(key, v)
        return v

    def _pc_population(self, name, spec):
        """``(packed or None, count)`` of a population spec on all events."""
        key, kind, gids = spec
        n = len(self._samples[name].data)
        if kind == 'all':
            return None, n
        store = self._pc_store()
        ver = self._pc_version(name)
        if kind == 'gate':
            gid = gids[0]
            return (self._pc_packed_mask(name, gid),
                    self._pc_counts(name, [gid])[gid])
        ukey = (name, ver, 'union', key[1])
        u = store.get(ukey)
        if u is None:
            self._pc_eval(name, [], union=gids)
            u = store.get(ukey)
            if u is None:                     # evicted at once (tiny budget)
                res = _pcm.scan(self._samples[name].data,
                                gates=self._sample_gates.get(name, {}),
                                union_gids=gids,
                                overrides=self._pc_overrides(name))
                u = res['union']
        return u

    @staticmethod
    def _pc_and(n, a, b):
        """AND of two ``(packed or None, count)`` row sets."""
        pa, ca = a
        pb, cb = b
        if pa is None:
            return pb, cb
        if pb is None:
            return pa, ca
        p = np.bitwise_and(pa, pb)
        return p, _pcm.popcount(p)

    def _pc_eligible(self, name, cols, spec):
        """Rows ``_get_df`` keeps before the display cap: valid on ``cols``
        and inside the population. Mirrors the old order (dropna, then gate
        masks on what is left): for per-event gates evaluating on all rows and
        intersecting is identical; for the one case that is not per-event (a
        boolean over an auto-clean gate, with NA rows present) the population
        is evaluated on the valid rows, as before."""
        n = len(self._samples[name].data)
        valid = self._pc_valid(name, cols)
        if spec[1] == 'all':
            return valid
        gates = self._sample_gates.get(name, {}) or {}
        if valid[0] is not None and not _pcm.chunkable(gates):
            return self._pc_eligible_legacy(name, cols, valid, spec)
        return self._pc_and(n, valid, self._pc_population(name, spec))

    def _pc_eligible_legacy(self, name, cols, valid, spec):
        store = self._pc_store()
        ver = self._pc_version(name)
        key = (name, ver, 'legacy', tuple(cols), spec[0])
        hit = store.get(key)
        if hit is not None:
            return hit
        import pandas as pd

        from .pipeline import cumulative_gate_mask
        df = self._samples[name].data
        n = len(df)
        pos = np.flatnonzero(_pcm.unpack(valid[0], n))
        sub = df.take(pos)
        gates = self._sample_gates.get(name, {}) or {}
        ov = self._pc_overrides(name)
        ov_sub = (None if not ov else
                  {g: pd.Series(m, index=df.index).reindex(
                      sub.index, fill_value=True).to_numpy()
                   for g, m in ov.items()})
        cache: dict = {}
        m = np.zeros(len(sub), dtype=bool)
        for gid in spec[2]:
            m |= np.asarray(cumulative_gate_mask(gates, gid, sub,
                                                 overrides=ov_sub, cache=cache),
                            dtype=bool)
        full = np.zeros(n, dtype=bool)
        full[pos[m]] = True
        out = (_pcm.pack(full), int(m.sum()))
        store.put(key, out)
        return out

    def _pc_rows(self, name, cols, spec, cap, seed, eligible=None, tag=''):
        """Row positions ``_get_df`` returns (sorted), or None for every row.
        ``eligible`` overrides the rows to choose from (``tag`` keeps that
        apart in the cache)."""
        store = self._pc_store()
        ver = self._pc_version(name)
        n = len(self._samples[name].data)
        cap, seed = self._pc_norm_cap(n, cap, seed)
        key = (name, ver, 'rows' + tag, tuple(cols), spec[0], cap, seed)
        if key in store:
            return store.get(key)
        elig, m = (self._pc_eligible(name, cols, spec) if eligible is None
                   else eligible)
        rows = _pcm.pick_rows(n, elig, m, cap, seed)
        if rows is None or rows.size <= _UNCACHED_ROWS:
            store.put(key, rows)
        return rows

    @staticmethod
    def _pc_norm_cap(n, cap, seed):
        """A cap at or above the sample size subsamples nothing: key it as no
        cap (and no seed), so every such cap shares one entry."""
        if cap is None or cap >= n:
            return None, None
        return cap, seed

    def _pc_cols(self, s, chans, alias):
        """The sample's own columns ``_get_df`` checks for NA on."""
        cols = []
        for c in chans:
            if not c:
                continue
            own = alias.get(c, c)
            if own in s.data.columns and own not in cols:
                cols.append(own)
        return cols

    def _pc_cap(self, name, for_hist=False, downsample=True, displayed=None):
        """The display cap ``_get_df`` applies. ``displayed`` (warm-up) also
        counts that sample into the downsample-to-smallest floor."""
        _dv = getattr(self, 'ds_display_var', None)
        _pv = getattr(self, 'ds_propagate_var', None)
        ds_on = ((_dv is not None and _dv.get())
                 or (_pv is not None and _pv.get()))
        cap = (self._display_point_cap()
               if (downsample and not for_hist and ds_on) else None)
        if downsample and _dv is not None and _dv.get():
            floor = self._smallest_loaded_sample_size()
            if displayed is not None and displayed in self._samples:
                own = len(self._samples[displayed].data)
                floor = own if floor is None else min(floor, own)
            if floor is not None and floor > 0:
                cap = floor if cap is None else min(cap, floor)
        return cap

    def _pc_frame(self, s, rows, alias):
        """The frame for ``rows`` (all rows when None) with alias columns.
        Never copies the whole frame: a shallow copy shares the columns."""
        df = s.data
        out = df.copy(deep=False) if rows is None else df.take(rows)
        for chosen, own in (alias or {}).items():
            out[chosen] = out[own]
        return out

    # ── histogram ────────────────────────────────────────────────────────

    def _pc_hist_spec(self, name, x, displayed=None):
        """What one sample's histogram source depends on, or None."""
        s = self._samples.get(name)
        if s is None:
            return None
        alias = self._axis_alias_for_sample(s, [x, None])
        cols = self._pc_cols(s, [x], alias)
        if not cols:
            return None
        spec = self._pc_popspec(name, 'auto')
        cap = self._pc_cap(name, for_hist=True, displayed=displayed)
        cap, seed = self._pc_norm_cap(len(s.data), cap,
                                      _pcm.stable_seed(name, x, None, cap))
        return {'col': cols[0], 'cols': cols, 'spec': spec, 'cap': cap,
                'seed': seed}

    def _pc_hist_values(self, name, hs):
        """The finite float64 values the histogram bins (all events of the
        population, or its capped subsample)."""
        gates = self._sample_gates.get(name, {}) or {}
        if hs['cap'] is None and _pcm.chunkable(gates):
            # Uncapped: the finite filter below drops every NA row anyway,
            # so the valid-row pass is not needed to know which rows count
            # (per-event gates give the same population either way).
            pop, m = self._pc_population(name, hs['spec'])
            rows = _pcm.pick_rows(len(self._samples[name].data), pop, m,
                                  None, None)
        else:
            rows = self._pc_rows(name, hs['cols'], hs['spec'], hs['cap'],
                                 hs['seed'])
        col = self._samples[name].data[hs['col']].to_numpy()
        vals = np.asarray(col if rows is None else col[rows], dtype=float)
        return vals[np.isfinite(vals)]

    def _pc_hist_key(self, name, hs, what, *extra):
        ver = self._pc_version(name)
        return (name, ver, what, hs['col'], hs['spec'][0], hs['cap'],
                hs['seed']) + extra

    def _pc_needs_anchor(self, channel) -> bool:
        """Whether the channel's bins/axis read the data (symlog does)."""
        return 'symlog' in (
            self._channel_scale.get(channel, self._default_channel_scale),
            self._channel_scale.get(channel, self._default_scale_for(channel)))

    @staticmethod
    def _pc_hist_stats(arr):
        """``(size, lo, hi)`` -- the robust range ``_plot_histogram`` unions."""
        if arr.size == 0:
            return (0, None, None)
        if arr.size >= 20:
            a, b = np.percentile(arr, (0.1, 99.9))
        else:
            a, b = float(arr.min()), float(arr.max())
        return (int(arr.size), float(a), float(b))

    # ── background warm-up ───────────────────────────────────────────────

    def _warm_worker(self):
        w = getattr(self, '_pc_worker', None)
        if w is None and not getattr(self, '_pc_closed', False):
            w = self._pc_worker = _WarmWorker()
            try:
                self.bind('<Destroy>', self._pc_on_destroy, add='+')
            except Exception:                              # noqa: BLE001
                pass
        return w

    def _pc_on_destroy(self, event=None):
        if event is None or getattr(event, 'widget', None) is self:
            self._pc_shutdown()

    def _pc_shutdown(self):
        """Stop the worker and drop every cache entry (editor closing)."""
        self._pc_closed = True
        for attr in ('_pc_poll_after', '_pc_sched_after'):
            aid = getattr(self, attr, None)
            if aid:
                try:
                    self.after_cancel(aid)
                except Exception:                          # noqa: BLE001
                    pass
            setattr(self, attr, None)
        w = getattr(self, '_pc_worker', None)
        self._pc_worker = None
        if w is not None:
            w.shutdown()
        self._pc_inflight = None
        self._pc_warm_names = deque()
        st = getattr(self, '_pc_cache', None)
        if st is not None:
            st.purge()

    def _warm_pause(self):
        w = getattr(self, '_pc_worker', None)
        if w is not None:
            w.pause()

    def _warm_resume(self):
        w = getattr(self, '_pc_worker', None)
        if w is not None:
            w.resume()

    def _warm_after_replot(self):
        """End of a replot: let the draw happen, then resume/replan."""
        try:
            self.after_idle(self._warm_resume)
        except Exception:                                  # noqa: BLE001
            self._warm_resume()
        self._warm_schedule()

    def _warm_schedule(self, delay_ms=120):
        if getattr(self, '_pc_closed', False) or not getattr(
                self, '_pc_warm_enabled', True):
            return
        aid = getattr(self, '_pc_sched_after', None)
        if aid:
            try:
                self.after_cancel(aid)
            except Exception:                              # noqa: BLE001
                pass
        try:
            self._pc_sched_after = self.after(delay_ms, self._warm_replan)
        except Exception:                                  # noqa: BLE001
            self._pc_sched_after = None

    def _warm_order(self):
        """Active sample, then the displayed ones, then tree order."""
        names = []
        if self._active_sample in self._samples:
            names.append(self._active_sample)
        for n in self._target_samples('enabled'):
            if n not in names:
                names.append(n)
        try:
            for t in self.gate_tv.get_children(''):
                stack = [t]
                while stack:
                    iid = stack.pop(0)
                    p = self._parse_iid(iid)
                    if p and p[0] == 'sample':
                        if p[1] in self._samples and p[1] not in names:
                            names.append(p[1])
                        continue
                    stack[0:0] = list(self.gate_tv.get_children(iid))
        except Exception:                                  # noqa: BLE001
            pass
        for n in self._sample_order:
            if n in self._samples and n not in names:
                names.append(n)
        return names

    def _warm_view(self):
        """The view the warm-up prepares for (Tk vars read here, not in the
        worker)."""
        return {'mode': self.mode_var.get(),
                'x': self._resolve_channel(self.x_combo.get()),
                'y': self._resolve_channel(self.y_combo.get())}

    def _warm_replan(self):
        self._pc_sched_after = None
        if getattr(self, '_pc_closed', False):
            return
        self._pc_store()
        self._pc_warm_names = deque(self._warm_order())
        inf = getattr(self, '_pc_inflight', None)
        if inf is not None:
            # Keep running work that is still wanted as planned now; cancel
            # work planned against an older version / gate set / view.
            name = inf['name']
            if (name not in self._samples
                    or self._warm_plan_key(name) != inf['key']):
                w = getattr(self, '_pc_worker', None)
                if w is not None:
                    w.cancel(inf['tid'])
            else:
                try:
                    self._pc_warm_names.remove(name)
                except ValueError:
                    pass
        self._warm_kick()

    def _warm_plan_key(self, name):
        view = self._warm_view()
        return (self._pc_version(name),
                tuple(sorted(self._pc_sigs(name).items())),
                view['mode'], view['x'], view['y'],
                bool(self.apply_gates_var.get()),
                self._pc_cap(name, displayed=name),
                self._pc_cap(name, for_hist=True, displayed=name))

    def _warm_cancel_sample(self, name):
        inf = getattr(self, '_pc_inflight', None)
        w = getattr(self, '_pc_worker', None)
        if inf is not None and inf['name'] == name and w is not None:
            w.cancel(inf['tid'])

    def _warm_kick(self):
        if getattr(self, '_pc_closed', False):
            return
        if getattr(self, '_clustering_busy', False):
            self._warm_poll_later(250)       # data is being written off-thread
            return
        if getattr(self, '_pc_inflight', None) is None:
            names = getattr(self, '_pc_warm_names', deque())
            while names:
                name = names.popleft()
                if name not in self._samples:
                    continue
                plan = self._warm_plan(name)
                if plan is None:
                    continue
                w = self._warm_worker()
                if w is None:
                    return
                self._pc_seq += 1
                plan['tid'] = self._pc_seq
                plan['t0'] = time.perf_counter()
                self._pc_inflight = plan
                w.submit(plan['tid'], plan.pop('fn'))
                break
        if getattr(self, '_pc_inflight', None) is not None:
            self._warm_poll_later()

    def _warm_poll_later(self, ms=25):
        if getattr(self, '_pc_poll_after', None) or getattr(self, '_pc_closed', False):
            return
        try:
            self._pc_poll_after = self.after(ms, self._warm_poll)
        except Exception:                                  # noqa: BLE001
            self._pc_poll_after = None

    def _warm_poll(self):
        self._pc_poll_after = None
        if getattr(self, '_pc_closed', False):
            return
        w = getattr(self, '_pc_worker', None)
        while w is not None:
            try:
                tid, status, payload = w.results.get_nowait()
            except queue.Empty:
                break
            inf = getattr(self, '_pc_inflight', None)
            if inf is None or inf['tid'] != tid:
                continue
            self._pc_inflight = None
            if status == 'ok':
                self._warm_store(inf, payload)
            elif status == 'error':
                print(f"[warm-up] {inf['name']}: {type(payload).__name__}: "
                      f"{payload}", flush=True)
            log = getattr(self, '_pc_warm_log', None)
            if log is not None:
                log.append((inf['name'], status,
                            time.perf_counter() - inf['t0']))
        self._warm_kick()

    def _warm_plan(self, name):
        """What is missing for ``name`` under the current view, as a worker
        task over snapshots; None when nothing is."""
        store = self._pc_store()
        s = self._samples[name]
        ver = self._pc_version(name)
        gates = self._sample_gates.get(name, {}) or {}
        sigs = self._pc_sigs(name)
        n = len(s.data)
        view = self._warm_view()
        internal = self._pc_internal(gates)
        count = [g for g in gates if (name, ver, 'count', sigs[g]) not in store]
        seeds = {g: store.get((name, ver, 'mask', sigs[g])) for g in gates}
        seeds = {g: p for g, p in seeds.items() if p is not None}
        # Masks are kept only alongside counts being computed anyway: an
        # evicted mask alone is never worth a pass (that would churn).
        keep = [g for g in internal if g not in seeds] if count else []
        alias = self._axis_alias_for_sample(s, [view['x'], view['y']])
        hist = view['mode'] == 'histogram'
        chans = [view['x']] if hist else [view['x'], view['y']]
        cols = self._pc_cols(s, chans, alias)
        spec = self._pc_popspec(name, 'auto')
        cap = self._pc_cap(name, for_hist=hist, displayed=name)
        cap, seed = self._pc_norm_cap(
            n, cap,
            _pcm.stable_seed(name, view['x'], None if hist else view['y'], cap))
        valid = store.get((name, ver, 'valid', tuple(cols))) if cols else None
        want_valid = bool(cols) and valid is None
        ukey = None
        union = None
        if spec[1] == 'union':
            ukey = (name, ver, 'union', spec[0][1])
            union = store.get(ukey)
        want_union = spec[1] == 'union' and union is None
        legacy = (spec[1] != 'all' and not _pcm.chunkable(gates))
        rkey = (name, ver, 'rows', tuple(cols), spec[0], cap, seed)
        want_rows = bool(cols) and rkey not in store and not legacy
        hs = hist_edges_fn = None
        want_hist = False
        if hist and cols and not legacy:
            hs = {'col': cols[0], 'cols': cols, 'spec': spec, 'cap': cap,
                  'seed': seed}
            skey = (name, ver, 'hstat', cols[0], spec[0], cap, seed)
            want_hist = skey not in store
            hist_edges_fn = self._pc_edges_fn(view['x'])
        if spec[1] == 'gate':
            want_union = False
        if not (count or keep or want_valid or want_union or want_rows
                or want_hist):
            return None
        frame = s.data.copy(deep=False)
        gsnap = copy.deepcopy(gates)
        ov_have = None
        ac = [g for g, gd in gates.items() if gd.get('kind') == 'autoclean']
        if ac:
            from .pipeline import autoclean_methods_signature
            ov_have = {}
            for gid in ac:
                ent = self._ac_cache.get((name, gid))
                if (ent is not None and ent[0] == id(s.data)
                        and ent[1] == autoclean_methods_signature(gates[gid])):
                    ov_have[gid] = np.asarray(ent[2].to_numpy(), dtype=bool)
        union_gids = spec[2] if want_union else None
        hist_scale = (self._channel_scale.get(
            view['x'], self._default_scale_for(view['x'])) if hist else None)

        def fn(checkpoint):
            from .pipeline import autoclean_keep_mask
            out: dict[str, Any] = {'ov_new': {}}
            ov = dict(ov_have or {})
            for gid in ac:
                if gid not in ov:
                    checkpoint()
                    ov[gid] = np.asarray(autoclean_keep_mask(gsnap[gid], frame),
                                         dtype=bool)
                    out['ov_new'][gid] = ov[gid]
            if count or keep or want_valid or union_gids is not None:
                res = _pcm.scan(frame, gates=gsnap, count_gids=count,
                                keep_gids=keep, union_gids=union_gids,
                                valid_cols=(cols if want_valid else None),
                                overrides=ov or None, seed_masks=seeds,
                                checkpoint=checkpoint)
                out['scan'] = res
            else:
                res = {}
            if want_rows or want_hist:
                v = res.get('valid', valid) if want_valid else valid
                if v is None:
                    v = (None, n)
                if spec[1] == 'union':
                    u = res.get('union', union)
                elif spec[1] == 'gate':
                    gid = spec[2][0]
                    p = res.get('masks', {}).get(gid, seeds.get(gid))
                    if p is None:
                        r2 = _pcm.scan(frame, gates=gsnap, keep_gids=[gid],
                                       overrides=ov or None, seed_masks=seeds,
                                       checkpoint=checkpoint)
                        p = r2['masks'][gid]
                    u = (p, _pcm.popcount(p))
                else:
                    u = (None, n)
                elig = self._pc_and(n, v, u)
                rows = _pcm.pick_rows(n, elig[0], elig[1], cap, seed,
                                      checkpoint=checkpoint)
                out['rows'] = rows
                if want_hist and hs is not None:
                    checkpoint()
                    colv = frame[hs['col']].to_numpy()
                    vals = np.asarray(colv if rows is None else colv[rows],
                                      dtype=float)
                    arr = vals[np.isfinite(vals)]
                    del vals
                    st = self._pc_hist_stats(arr)
                    out['hstat'] = st
                    if st[0]:
                        from .plotmath import unplottable_count
                        out['hhid'] = (hist_scale,
                                       int(unplottable_count(arr, hist_scale)))
                        if hist_edges_fn is not None:
                            lo, hi = self._pc_hist_range([st])
                            edges = hist_edges_fn(lo, hi, _HIST_BINS)
                            if edges is not None:
                                checkpoint()
                                out['hcnt'] = (edges, np.histogram(
                                    arr, bins=edges)[0])
            return out

        return {'name': name, 'ver': ver, 'sigs': sigs, 'ukey': ukey,
                'rkey': rkey, 'hs': hs, 'n': n, 'fn': fn,
                'key': self._warm_plan_key(name), 'data': s.data,
                'cols': cols, 'x': view['x']}

    def _warm_store(self, plan, out):
        """Keep a worker result -- only if computed against the sample's
        current data (anything else is discarded, never stored)."""
        name = plan['name']
        s = self._samples.get(name)
        if s is None or s.data is not plan['data']:
            return
        ver = self._pc_version(name)
        if ver != plan['ver']:
            return
        store = self._pc_store()
        import pandas as pd

        from .pipeline import autoclean_methods_signature
        gates = self._sample_gates.get(name, {}) or {}
        for gid, m in (out.get('ov_new') or {}).items():
            g = gates.get(gid)
            if g is not None:
                self._ac_cache[(name, gid)] = (
                    id(s.data), autoclean_methods_signature(g),
                    pd.Series(m, index=s.data.index))
        res = out.get('scan') or {}
        if res:
            self._pc_put_scan(name, ver, plan['sigs'], res, plan['ukey'])
            if 'valid' in res:
                store.put((name, ver, 'valid', tuple(plan['cols'])), res['valid'])
        if 'rows' in out:
            rows = out['rows']
            if rows is None or rows.size <= _UNCACHED_ROWS:
                store.put(plan['rkey'], rows)
        hs = plan.get('hs')
        if hs is not None and 'hstat' in out:
            store.put(self._pc_hist_key(name, hs, 'hstat'), out['hstat'], 128)
            if 'hhid' in out:
                scale, hid = out['hhid']
                store.put(self._pc_hist_key(name, hs, 'hhid', scale), hid, 64)
            if 'hcnt' in out:
                edges, counts = out['hcnt']
                # Only under the edges the plot itself would compute.
                lo, hi = self._pc_hist_range([out['hstat']])
                want = np.asarray(self._screen_uniform_edges(
                    plan['x'], lo, hi, _HIST_BINS, data_sample=None),
                    dtype=float)
                if want.shape == edges.shape and np.array_equal(want, edges):
                    store.put(self._pc_hist_key(
                        name, hs, 'hcnt', _pcm.edges_digest(edges)), counts)

    def _pc_edges_fn(self, channel):
        """A Tk/matplotlib-free stand-in for ``_screen_uniform_edges`` the
        worker can call, or None where the edges depend on data (symlog)."""
        tm = self._channel_transform.get(channel, 'linear')
        view_scale = self._channel_scale.get(channel, self._default_channel_scale)
        lin_scale = self._channel_scale.get(channel,
                                            self._default_scale_for(channel))

        def fn(lo, hi, n_bins):
            return _pcm.screen_uniform_edges(tm, view_scale, lin_scale,
                                             lo, hi, n_bins)
        return fn

    @staticmethod
    def _pc_hist_range(stats):
        """``(lo, hi)`` -- the padded union of per-sample robust ranges, as
        ``_plot_histogram`` draws it."""
        lo, hi = np.inf, -np.inf
        for st in stats:
            if not st or not st[0]:
                continue
            a, b = st[1], st[2]
            if a < lo:
                lo = float(a)
            if b > hi:
                hi = float(b)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = lo - 1.0, lo + 1.0
        else:
            pad = (hi - lo) * 0.02
            lo, hi = lo - pad, hi + pad
        return lo, hi

    def _pc_stats(self) -> dict:
        """Diagnostics: cache entries, bytes, worker state."""
        st = self._pc_store()
        w = getattr(self, '_pc_worker', None)
        return {'entries': len(st), 'bytes': st.nbytes, 'budget': st.budget,
                'worker_alive': bool(w is not None and w.thread.is_alive()),
                'inflight': (getattr(self, '_pc_inflight', None) or {}).get('name'),
                'queued': list(getattr(self, '_pc_warm_names', ()))}
