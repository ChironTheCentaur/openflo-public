"""Statistics-row collection, heatmap report, and absolute counts.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class StatsMixin(EditorMixin):
    """Collect per-population stats rows, build the heatmap HTML report, resolve marker columns, and open the absolute-counts dialog."""

    def _open_abs_counts(self):
        """Counting-bead absolute-count calculator."""
        from .ui_abscounts import AbsCountsDialog
        AbsCountsDialog(self)

    def _sample_rows(self, name, want, select=None):
        """Population rows for ONE loaded sample. `select` (a list of gids)
        restricts the emitted populations; None emits all of the sample's
        gates. Returns [] if the sample isn't loaded."""
        s = self._samples.get(name)
        if s is None:
            return []
        gates = self._sample_gates.get(name, {})
        order = self._sample_gate_order.get(name, list(gates))
        channels = [c for c in getattr(s, 'fluor_channels', [])
                    if c in s.data.columns]
        # Use THIS sample's own antibody labels (so a marker on a different
        # fluor still names its column by label and ties across samples); the
        # editor's global labels are a fallback.
        labels = dict(self._channel_labels)
        labels.update(getattr(s, 'channel_labels', {}) or {})
        sel = None if select is None else [g for g in select if g in gates]
        return self._population_stats(
            name, s.data, gates, order, labels, channels, want, select=sel)

    def _collect_stats_rows(self, want, samples=None, gate_targets=None):
        """Aggregate population rows. Three modes:
          • gate_targets : list of (sample, gid) → emit exactly those
            populations (grouped by sample, first-seen order). This is the
            curated, gate-only mode used by the stats window.
          • samples       : restrict to these sample names (all their pops).
          • neither       : every population of every loaded sample.
        `want` is the selected stat-name set. Returns (rows, columns); the
        column set excludes internal ``__``-prefixed keys (e.g. __gid__)."""
        all_rows = []
        if gate_targets is not None:
            by_sample = {}
            for nm, gid in gate_targets:
                by_sample.setdefault(nm, [])
                if gid not in by_sample[nm]:
                    by_sample[nm].append(gid)
            for name, gids in by_sample.items():
                all_rows.extend(self._sample_rows(name, want, select=gids))
        else:
            names = samples if samples is not None else [
                n for n in self._sample_order if n in self._samples]
            for name in names:
                all_rows.extend(self._sample_rows(name, want))
        # Stable column order: identity cols first, then pop-level, then
        # per-channel in first-seen order. Internal __keys never display.
        cols = ['Sample', 'Population']
        for stat in self.STAT_POP:
            if stat in want:
                cols.append(stat)
        seen = set(cols)
        for r in all_rows:
            for k in r:
                if k not in seen and not k.startswith('__'):
                    seen.add(k)
                    cols.append(k)
        return all_rows, cols

    def _stats_snapshot(self, want, samples=None, gate_targets=None):
        """Tk-thread half of a backgrounded stats run: gather the pure
        ``_population_stats`` arguments (deep-copying the small gate dicts) so the
        heavy compute can run off-thread on a stable snapshot while the user
        keeps gating. Mirrors ``_collect_stats_rows``' three target modes; the
        sample data is referenced (read-only), only the gates are copied.
        Returns a list of arg tuples — feed to ``_stats_rows_from_snapshot``."""
        import copy as _copy

        def _args(name, select=None):
            s = self._samples.get(name)
            if s is None:
                return None
            gates = self._sample_gates.get(name, {})
            order = self._sample_gate_order.get(name, list(gates))
            channels = [c for c in getattr(s, 'fluor_channels', [])
                        if c in s.data.columns]
            labels = dict(self._channel_labels)
            labels.update(getattr(s, 'channel_labels', {}) or {})
            sel = None if select is None else [g for g in select if g in gates]
            return (name, s.data, _copy.deepcopy(gates), list(order), labels,
                    channels, want, sel)

        out = []
        if gate_targets is not None:
            by_sample = {}
            for nm, gid in gate_targets:
                by_sample.setdefault(nm, [])
                if gid not in by_sample[nm]:
                    by_sample[nm].append(gid)
            for name, gids in by_sample.items():
                a = _args(name, gids)
                if a is not None:
                    out.append(a)
        else:
            names = samples if samples is not None else [
                n for n in self._sample_order if n in self._samples]
            for name in names:
                a = _args(name)
                if a is not None:
                    out.append(a)
        return out

    def _stats_rows_from_snapshot(self, snapshot, want):
        """Off-thread half: run the pure ``_population_stats`` over a snapshot
        and derive the same stable column order as ``_collect_stats_rows``.
        Returns ``(rows, columns)``."""
        all_rows = []
        for args in snapshot:
            all_rows.extend(self._population_stats(*args))
        cols = ['Sample', 'Population']
        for stat in self.STAT_POP:
            if stat in want:
                cols.append(stat)
        seen = set(cols)
        for r in all_rows:
            for k in r:
                if k not in seen and not k.startswith('__'):
                    seen.add(k)
                    cols.append(k)
        return all_rows, cols

    def _report_heatmap_html(self):
        """A cluster × marker median-expression heatmap for the active (or
        first) sample carrying a label column. Returns an ``<img>`` or None."""
        name = self._active_sample or (self._sample_order[0]
                                       if self._sample_order else None)
        s = self._samples.get(name) if name else None
        if s is None:
            return None
        col = next((c for c in ('leiden', 'cluster', 'flowsom_meta')
                    if c in s.data.columns), None)
        if col is None:
            return None
        chans = [c for c in getattr(s, 'fluor_channels', [])
                 if c in s.data.columns]
        if not chans:
            return None
        df = s.data[s.data[col] >= 0]
        if df.empty:
            return None
        med = df.groupby(col)[chans].median()
        if med.empty:
            return None
        from matplotlib.figure import Figure

        from .report import figure_html
        fig = Figure(figsize=(min(1 + 0.5 * len(chans), 10),
                              min(1 + 0.3 * len(med), 9)), dpi=120)
        ax = fig.add_subplot(111)
        # Column-z-score so markers on different scales are comparable.
        arr = med.to_numpy(dtype=float)
        mu, sd = arr.mean(0), arr.std(0)
        sd[sd == 0] = 1.0
        im = ax.imshow((arr - mu) / sd, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels([self._fmt_channel(c) for c in chans],
                           rotation=90, fontsize=7)
        ax.set_yticks(range(len(med)))
        ax.set_yticklabels([str(i) for i in med.index], fontsize=7)
        ax.set_xlabel('marker'); ax.set_ylabel(col)
        ax.set_title(f"{name}: median expression per {col} (column z-score)",
                     fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return figure_html(fig, alt='cluster heatmap')

    def _marker_column_for(self, sample, channel):
        """Resolve a chosen marker ``channel`` to the column it lives on in
        ``sample`` — the channel itself if present, else a detector carrying the
        same antibody label (cross-fluor tying). None if absent."""
        df = sample.data
        if channel in df.columns:
            return channel
        label = self._channel_labels.get(channel, channel)
        for det, lab in (getattr(sample, 'channel_labels', {}) or {}).items():
            if lab == label and det in df.columns:
                return det
        return None

    def _sample_group_label(self, name, factor, tokens=None):
        """Assign a sample to a comparison group by the chosen ``factor``:

          • ``'Trial / day'``  → its trial/day (``_sample_trial``)
          • ``'Comp vs Samples'`` → 'Comps' / 'Samples' (``_sample_is_comp``)
          • ``'Name token'``   → the first token in ``tokens`` the sample name
            contains (case-insensitive), else 'Other'
        """
        if factor == 'Comp vs Samples':
            return 'Comps' if self._sample_is_comp.get(name) else 'Samples'
        if factor == 'Name token':
            low = name.lower()
            # Most-specific (longest) matching token wins, so 'Ctrl' beats
            # 'Stim' (a substring) regardless of the order they were typed.
            matches = [t.strip() for t in (tokens or [])
                       if t.strip() and t.strip().lower() in low]
            return max(matches, key=len) if matches else 'Other'
        return self._sample_trial.get(name, 'Trial')   # Trial / day (default)
