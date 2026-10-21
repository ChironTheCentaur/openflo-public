"""Statistics-row collection, heatmap report, and absolute counts.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class _StatsArgs(tuple):
    """One sample's ``_population_stats`` arguments, plus the cache key its
    rows are kept under (``cache_key``) and the rows already kept there
    (``cached``, saved_stats numbers), if any."""
    cache_key: tuple | None = None
    cached: list | None = None


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
            name, s.data, gates, order, labels, channels, want, select=sel,
            transforms=getattr(s, 'data_transforms', None))

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

        from . import saved_stats as _ss
        from .prefs import read_prefs
        from .result_stamp import app_version
        settings = _ss.stats_settings_key(read_prefs())
        version = app_version()

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
            # The transform record rides along so the off-thread half computes
            # intensity statistics on linear values (see population_stats).
            # Data and record are taken TOGETHER: a transform change replaces
            # columns of s.data on the Tk thread, and the live frame would
            # then be inverted with the record taken before it. A shallow
            # copy keeps today's columns (copy-on-write) without copying data.
            args = _StatsArgs((
                name, s.data.copy(deep=False), _copy.deepcopy(gates),
                list(order), labels, channels, want, sel,
                dict(getattr(s, 'data_transforms', None) or {})))
            # Rows computed before for exactly these inputs (this data
            # version, these gates and settings, this OpenFlo) are served
            # again rather than recomputed; a session brings back its own.
            # An editor without the plot cache computes every time.
            if not hasattr(self, '_pc_store'):
                return args
            store = self._pc_store()
            key = _ss.stats_key(
                sigs=self._pc_sigs(name), order=args[3], select=sel,
                labels=labels, channels=channels, want=want,
                stat_chan=self.STAT_CHAN, transforms=args[8],
                settings=settings, version=version)
            args.cache_key = (name, self._pc_version(name), 'stats', key)
            args.cached = store.get(args.cache_key)
            return args

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

    def _stats_rows_from_snapshot(self, snapshot, want, mfi_reference=None,
                                  reference_snapshot=None):
        """Off-thread half: run the pure ``_population_stats`` over a snapshot
        and derive the same stable column order as ``_collect_stats_rows``.
        Returns ``(rows, columns)``.

        With the MFI ratio wanted, ``mfi_reference`` names the reference
        sample; ``reference_snapshot`` (a ``_stats_snapshot`` of it alone)
        supplies its populations when the table itself leaves them out."""
        from .gating import MFI_RATIO, add_mfi_ratio
        all_rows = []
        for args in snapshot:
            all_rows.extend(self._stats_rows_for(args))
        if MFI_RATIO in want and mfi_reference:
            ref_rows = None
            if reference_snapshot:
                ref_rows = [r for args in reference_snapshot
                            for r in self._stats_rows_for(args)]
            add_mfi_ratio(all_rows, mfi_reference, ref_rows)
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

    def _stats_rows_for(self, args):
        """``_population_stats(*args)``, served from the cache when the
        snapshot found the rows there, and kept there when computed. Safe
        off the Tk thread (the store has its own lock)."""
        from . import saved_stats as _ss
        cached = getattr(args, 'cached', None)
        if cached is not None:
            rows = _ss.decode_rows(cached, args[0], args[2], args[4],
                                   args[8], args[1].columns)
            if rows is not None:
                return rows
        rows = self._population_stats(*args)
        key = getattr(args, 'cache_key', None)
        if key is not None:
            self._pc_store().put(key, _ss.encode_rows(rows))
        return rows

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
        # Display-scale medians, z-scored: a relative picture. The
        # Statistics window has the linear medians.
        ax.set_title(f"{name}: median expression per {col} (display scale, "
                     "column z-score)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
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
          • ``'Name token'``   → the LONGEST (most-specific) token in ``tokens``
            the sample name contains (case-insensitive)

        Returns None for a sample the comparison leaves out: one set aside
        (``_sample_set_aside``) under every factor; under 'Trial /
        day' and 'Name token', a compensation control (``_is_comp``), and
        under 'Name token', a sample matching no token. Callers drop those
        samples.
        """
        # Use the same comp test as the tree (_is_comp), which falls back to the
        # name rule — else name-detected comp controls (never manually
        # dragged) get pooled in with real samples.
        if name in getattr(self, '_sample_set_aside', {}):
            # Set aside with a reason (a failed sample): in no group, under
            # any factor, until it is returned.
            return None
        if factor == 'Comp vs Samples':
            return 'Comps' if self._is_comp(name) else 'Samples'
        if self._is_comp(name):
            # A single-stain compensation control is not a biological
            # replicate. Pooled into its trial, one (median 1000 against
            # 10-14) moved the group comparison's Kruskal-Wallis p from 0.0495
            # to 0.2888.
            return None
        if factor == 'Name token':
            low = name.lower()
            # Most-specific (longest) matching token wins, so 'Ctrl' beats
            # 'Stim' (a substring) regardless of the order they were typed.
            matches = [t.strip() for t in (tokens or [])
                       if t.strip() and t.strip().lower() in low]
            # A sample matching no token is left out, as the Tokens tooltip
            # says. Grouped as 'Other' it became a third group: Kruskal-Wallis
            # instead of Stim v Ctrl Mann-Whitney, an extra Prism column, and a
            # Diff. abundance refusal ('needs exactly 2 groups').
            return max(matches, key=len) if matches else None
        return self._sample_trial.get(name, 'Trial')   # Trial / day (default)
