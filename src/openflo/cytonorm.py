"""CytoNorm batch normalization: FlowSOM-cluster the pooled events, then map
each batch's per-metacluster, per-channel intensity quantiles onto a shared
goal distribution with a monotone spline.

Core module: numpy, with scipy and pandas imported where they are used. No
Tk, and it does not import ``openflo.pipeline``.
"""

import logging
import warnings

import numpy as np

from .flowsom import som_assign, som_metacluster, som_train

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see this warning.
log = logging.getLogger('openflo.pipeline')


# ── CytoNorm batch normalization ──────────────────────────────────────────────
#
# CytoNorm (Van Gassen 2020; CytoNorm 2.0, Quintelier 2025): the standard for
# removing technical batch/acquisition variation in cytometry. FlowSOM-cluster
# the pooled data into metaclusters, then for each metacluster and channel,
# quantile-normalize every batch's intensity distribution onto a shared GOAL
# distribution via a monotone (PCHIP) spline. Population-aware, so it doesn't
# smear distinct populations together the way a global quantile norm can.
#
# Two modes (same engine — only the events you fit on differ):
#   • 'goal'     (CytoNorm 2.0, default): fit on ALL samples; goal = pooled
#                aggregate. No dedicated control samples required.
#   • 'controls' (classic CytoNorm): fit on per-batch CONTROL samples; the
#                fitted per-batch transform is then applied to every sample in
#                that batch. Most rigorous when you ran a shared control aliquot
#                in each batch.
# The fitted model serializes (to_dict/from_dict) so it can be applied to new
# samples later — which cyCombine-style methods can't do.

def _strict_increasing(x):
    """Nudge ties so `x` is strictly increasing (PCHIP needs strictly
    increasing knots). Tiny relative epsilon, preserves order/scale."""
    x = np.asarray(x, dtype=float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + 1e-9 * (abs(x[i - 1]) + 1.0)
    return x


def cap_events_per_batch(events_by_batch, max_events_per_batch=20_000,
                         seed=42):
    """``{batch: events}`` with each batch's events subsampled, without
    replacement, to at most `max_events_per_batch` (0 or None keeps all).
    One generator seeded with `seed` draws for the batches in order, and the
    kept rows stay in their original order, so the same input always gives
    the same events. A DataFrame stays a DataFrame (index reset); an array
    stays an array. The draw CytoNorm.fit (and so the app's batch
    normalization and ``openflo-run --batch-correct``) fits on."""
    rng = np.random.default_rng(seed)
    out = {}
    for b, ev in events_by_batch.items():
        n = len(ev)
        if max_events_per_batch and n > max_events_per_batch:
            idx = np.sort(rng.choice(n, int(max_events_per_batch),
                                     replace=False))
            ev = (ev.iloc[idx].reset_index(drop=True) if hasattr(ev, 'iloc')
                  else np.asarray(ev)[idx])
        out[b] = ev
    return out


class CytoNorm:
    """FlowSOM + per-metacluster per-channel quantile normalization.

    Usage::

        cn = CytoNorm(channels=fluor_markers, mode='goal')
        cn.fit({batch_id: events_df, ...})       # events_df = a sample (or pooled)
        corrected = cn.apply(sample.data, batch_id)
        report = cn.qc({batch_id: events_df, ...})

    ``events_by_batch`` maps a batch label to that batch's events (a DataFrame
    carrying ``channels``, or an ndarray in channel order). In 'goal' mode pass
    all samples per batch (concatenate per batch); in 'controls' mode pass the
    control sample per batch, then ``apply`` to every sample of that batch.
    ``fit`` uses at most 20,000 events a batch by default
    (cap_events_per_batch, seeded with ``seed``).
    """

    def __init__(self, channels, n_metaclusters=10, grid=(10, 10),
                 n_quantiles=101, min_cell_events=50, mode='goal', seed=42):
        self.channels = [str(c) for c in channels]
        self.n_metaclusters = int(n_metaclusters)
        self.grid = (int(grid[0]), int(grid[1]))
        self.n_quantiles = int(n_quantiles)
        if self.n_quantiles < 2:
            raise ValueError(
                "n_quantiles must be >= 2 for spline fitting, got "
                f"{self.n_quantiles}")
        self.min_cell_events = int(min_cell_events)
        self.mode = str(mode)
        self.seed = int(seed)
        self.batches = []
        self._mean = None
        self._std = None
        self._W = None
        self._node_meta = None
        self._qs = None
        self._goal_q = {}        # (meta, ch_idx) -> goal quantile array | None
        self._batch_q = {}       # (meta, batch, ch_idx) -> quantile array | None

    # -- helpers --
    def _events(self, ev):
        if hasattr(ev, 'columns'):
            missing = [c for c in self.channels if c not in ev.columns]
            if missing:
                raise ValueError(
                    f"CytoNorm: batch data is missing channel(s) {missing}; "
                    f"expected {list(self.channels)}")
            return ev[self.channels].to_numpy(dtype=float)
        return np.asarray(ev, dtype=float)

    def _z(self, X):
        return (X - self._mean) / self._std

    def _meta_of(self, X):
        assert self._node_meta is not None and self._W is not None
        return self._node_meta[som_assign(self._z(X), self._W)]

    # -- fit / apply / qc --
    def fit(self, events_by_batch, max_events_per_batch=20_000):
        """Fit on ``{batch: events}``. Each batch is first subsampled to at
        most `max_events_per_batch` events (0 or None: all) with the model's
        ``seed`` (cap_events_per_batch), so a re-fit is identical and a big
        batch doesn't outweigh the others in the goal distribution."""
        self.batches = [str(b) for b in events_by_batch]
        capped = cap_events_per_batch(events_by_batch, max_events_per_batch,
                                      seed=self.seed)
        finite = {}
        pools = []
        for b, ev in capped.items():
            X = self._events(ev)
            X = X[np.isfinite(X).all(1)]
            finite[b] = X
            if len(X):
                pools.append(X)
        if not pools:
            raise ValueError("CytoNorm.fit: no finite events to fit on.")
        pooled = np.vstack(pools)
        self._mean = pooled.mean(0)
        self._std = pooled.std(0)
        self._std[self._std == 0] = 1.0
        self._W, _coords = som_train(self._z(pooled), self.grid, seed=self.seed)
        self._node_meta = som_metacluster(self._W, self.n_metaclusters)
        self._qs = np.linspace(0.0, 1.0, self.n_quantiles)

        # Goal quantiles per (metacluster, channel) — the pooled aggregate.
        pooled_meta = self._meta_of(pooled)
        for m in np.unique(self._node_meta):
            sub = pooled[pooled_meta == m]
            for j in range(len(self.channels)):
                col = sub[:, j][np.isfinite(sub[:, j])]
                self._goal_q[(int(m), j)] = (
                    np.quantile(col, self._qs)
                    if col.size >= self.min_cell_events else None)

        # Per-batch quantiles per (metacluster, channel).
        for b, X in finite.items():
            bmeta = self._meta_of(X) if len(X) else np.array([], dtype=int)
            for m in np.unique(self._node_meta):
                sub = X[bmeta == m] if len(X) else X
                for j in range(len(self.channels)):
                    col = sub[:, j][np.isfinite(sub[:, j])] if len(sub) else sub
                    self._batch_q[(int(m), str(b), j)] = (
                        np.quantile(col, self._qs)
                        if col.size >= self.min_cell_events else None)
        return self

    def apply(self, df, batch_id):
        """Return a corrected copy of ``df`` for ``batch_id``. Rows/channels
        without a usable transform pass through unchanged."""
        from scipy.interpolate import PchipInterpolator
        out = df.copy()
        if self._W is None or not all(c in out.columns for c in self.channels):
            return out
        X = out[self.channels].to_numpy(dtype=float)
        finite = np.isfinite(X).all(1)
        meta = np.full(len(X), -1, dtype=int)
        if finite.any():
            meta[finite] = self._meta_of(X[finite])
        bkey = str(batch_id)
        for j, ch in enumerate(self.channels):
            vals = out[ch].to_numpy(dtype=float).copy()
            for m in np.unique(meta[meta >= 0]):
                bq = self._batch_q.get((int(m), bkey, j))
                gq = self._goal_q.get((int(m), j))
                if bq is None or gq is None:
                    continue
                sel = (meta == m) & np.isfinite(vals)
                if not sel.any():
                    continue
                xq = _strict_increasing(bq)
                v = np.clip(vals[sel], xq[0], xq[-1])
                nv = PchipInterpolator(xq, gq, extrapolate=False)(v)
                vals[sel] = np.where(np.isfinite(nv), nv, vals[sel])
            out[ch] = vals
        return out

    @staticmethod
    def qc_means(qc):
        """``(before, after, n_not_evaluable)`` for a :meth:`qc` result: the
        mean distances over the channels that could be measured, and how
        many could not. A channel that could not be measured is NaN in
        ``qc``; averaging it in would hide every real score behind one
        unmeasurable marker. Both means are NaN when no channel was
        measurable."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN mean
            before = float(np.nanmean([v['before'] for v in qc.values()]))
            after = float(np.nanmean([v['after'] for v in qc.values()]))
        skipped = sum(1 for v in qc.values() if not np.isfinite(v['before']))
        return before, after, skipped

    def qc(self, events_by_batch):
        """Per-channel mean Wasserstein distance batch→goal, before vs after.
        ``{channel: {'before': x, 'after': y}}`` — lower 'after' = better.

        A channel that could not be measured reports NaN, never 0.0. Zero is
        the BEST possible score on this scale, so a channel nobody could
        evaluate would be indistinguishable from a perfectly aligned one.
        An empty goal distribution is likewise reported rather than raising:
        when no event has every channel finite, the pooled reference is empty
        and scipy's wasserstein_distance would raise "Distribution can't be
        empty" out of a QC call.
        """
        import pandas as pd
        from scipy.stats import wasserstein_distance
        nan_result = {ch: {'before': float('nan'), 'after': float('nan')}
                      for ch in self.channels}
        pooled = np.vstack([self._events(ev) for ev in events_by_batch.values()])
        pooled = pooled[np.isfinite(pooled).all(1)]
        if pooled.size == 0:
            log.warning(
                "CytoNorm.qc: no event has a finite value in every channel, "
                "so there is no goal distribution to compare against — "
                "reporting the batch→goal distances as undefined.")
            return nan_result
        res = {}
        for j, ch in enumerate(self.channels):
            goal = pooled[:, j]
            goal = goal[np.isfinite(goal)]
            if goal.size == 0:
                res[ch] = dict(nan_result[ch])
                continue
            before, after = [], []
            for b, ev in events_by_batch.items():
                X = self._events(ev)
                col = X[:, j][np.isfinite(X[:, j])]
                if col.size:
                    before.append(wasserstein_distance(col, goal))
                cor = self.apply(pd.DataFrame(X, columns=pd.Index(self.channels)), b)
                cc = cor[ch].to_numpy(dtype=float)
                cc = cc[np.isfinite(cc)]
                if cc.size:
                    after.append(wasserstein_distance(cc, goal))
            res[ch] = {
                'before': float(np.mean(before)) if before else float('nan'),
                'after': float(np.mean(after)) if after else float('nan')}
        return res

    # -- serialization --
    def to_dict(self):
        """The fitted model as a JSON-serialisable dict (inverse of ``from_dict``):
        settings, scaling, SOM weights and quantiles, with tuple keys joined by ``|``."""
        def qmap(d):
            return {f'{k[0]}|{k[1]}' if len(k) == 2 else f'{k[0]}|{k[1]}|{k[2]}':
                    (None if v is None else [float(x) for x in v])
                    for k, v in d.items()}
        return {
            'format': 'openflo-cytonorm', 'version': 1,
            'channels': self.channels, 'n_metaclusters': self.n_metaclusters,
            'grid': list(self.grid), 'n_quantiles': self.n_quantiles,
            'min_cell_events': self.min_cell_events, 'mode': self.mode,
            'seed': self.seed, 'batches': self.batches,
            'mean': None if self._mean is None else self._mean.tolist(),
            'std': None if self._std is None else self._std.tolist(),
            'W': None if self._W is None else self._W.tolist(),
            'node_meta': None if self._node_meta is None
            else [int(x) for x in self._node_meta],
            'qs': None if self._qs is None else self._qs.tolist(),
            'goal_q': qmap(self._goal_q),
            'batch_q': qmap(self._batch_q),
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild a model from a dict written by ``to_dict``; missing settings fall
        back to the constructor defaults."""
        cn = cls(d['channels'], d.get('n_metaclusters', 10),
                 tuple(d.get('grid', (10, 10))), d.get('n_quantiles', 101),
                 d.get('min_cell_events', 50), d.get('mode', 'goal'),
                 d.get('seed', 42))
        cn.batches = list(d.get('batches', []))
        cn._mean = None if d.get('mean') is None else np.asarray(d['mean'])
        cn._std = None if d.get('std') is None else np.asarray(d['std'])
        cn._W = None if d.get('W') is None else np.asarray(d['W'])
        cn._node_meta = (None if d.get('node_meta') is None
                         else np.asarray(d['node_meta'], dtype=int))
        cn._qs = None if d.get('qs') is None else np.asarray(d['qs'])

        def unqmap(m):
            out = {}
            for k, v in (m or {}).items():
                parts = k.split('|')
                # batch_q keys are (meta, batch, channel); a batch label may
                # itself contain '|' (POSIX path), so take meta from the first
                # segment, channel from the last, and rejoin the middle as the
                # batch name. goal_q keys are 2-part and parsed separately.
                key = ((int(parts[0]), int(parts[1])) if len(parts) == 2
                       else (int(parts[0]), '|'.join(parts[1:-1]), int(parts[-1])))
                out[key] = None if v is None else np.asarray(v, dtype=float)
            return out
        cn._goal_q = unqmap(d.get('goal_q'))
        cn._batch_q = unqmap(d.get('batch_q'))
        return cn
