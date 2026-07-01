"""Analyze-menu launchers + clustering / annotation orchestration.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .editor_base import EditorMixin


class AnalysisMixin(EditorMixin):
    """Statistics / frequency / expression / QC / cluster / DR / trajectory / annotation launchers and apply."""

    def _annotate_populations(self, col):
        """Generic rename dialog for one label column's populations on the
        active sample. 'cluster' delegates to the legacy annotator; others
        edit the matching 'category' gates' names in place."""
        if col == 'cluster':
            self._annotate_clusters()
            return
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        vals = self._sample_label_values(name, col)
        if not vals:
            self.status_var.set(f"'{name}' has no '{col}' column to annotate.")
            return
        gates = self._sample_gates.get(name, {})

        def _gate_for(v):
            for g in gates.values():
                if (g.get('kind') == 'category' and g.get('channel') == col
                        and g.get('value') == v):
                    return g
            return None

        dlg = tk.Toplevel(self)
        disp = self.LABEL_COLUMNS.get(col, (col, None))[0]
        dlg.title(f"Annotate {disp} — {name}")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.geometry("360x440")

        ttk.Label(dlg, text=f"Names for '{name}':",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))
        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=(0, 6))
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        entries = {}
        for v in vals:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=f"{col} {v}", width=14).pack(side='left')
            g = _gate_for(v)
            cur = (g.get('name') if g else None) or f'{col} {v}'
            var = tk.StringVar(value=cur)
            ttk.Entry(row, textvariable=var, width=22).pack(
                side='left', fill='x', expand=True)
            entries[v] = var

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_apply():
            self._checkpoint()
            for v, var in entries.items():
                g = _gate_for(v)
                if g is not None:
                    g['name'] = var.get().strip() or f'{col} {v}'
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Updated {disp} names for '{name}'.")

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply", command=do_apply).pack(
            side='right', padx=(0, 6))

    def _annotate_clusters(self):
        """Dialog to name the active sample's clusters. Pre-fills existing
        phenotype names; on Apply, stores them in self._cluster_labels and
        renames any matching cluster gates, then refreshes the tree/plot."""
        name = self._active_sample
        if name is None:
            self.status_var.set("Select a sample first.")
            return
        ids = self._sample_cluster_ids(name)
        if not ids:
            self.status_var.set(
                f"'{name}' has no 'cluster' column to annotate.")
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Annotate clusters — {name}")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.geometry("360x440")
        dlg.minsize(300, 240)

        ttk.Label(dlg, text=f"Phenotype names for '{name}':",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=(0, 6))
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        entries = {}
        for cid in ids:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=f"Cluster {cid}", width=12).pack(side='left')
            var = tk.StringVar(value=self._cluster_display_name(name, cid))
            ttk.Entry(row, textvariable=var, width=24).pack(
                side='left', fill='x', expand=True)
            entries[cid] = var

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_apply():
            self._checkpoint()
            lbls = self._cluster_labels.setdefault(name, {})
            gates = self._sample_gates.get(name, {})
            for cid, var in entries.items():
                txt = var.get().strip()
                if txt:
                    lbls[cid] = txt
                else:
                    lbls.pop(cid, None)
                for g in gates.values():
                    if (g.get('kind') == 'cluster'
                            and g.get('cluster_id') == cid):
                        g['name'] = txt or f'Cluster {cid}'
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Updated cluster names for '{name}'.")

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply",
                   command=do_apply).pack(side='right', padx=(0, 6))

    def _open_cell_cycle_dialog(self):
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Load and select a sample first.")
            return
        from .pipeline import find_dna_channel
        s = self._samples[name]
        default = find_dna_channel(s)

        dlg = tk.Toplevel(self)
        dlg.title(f"Cell cycle — {name}")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text="DNA-content channel:").grid(
            row=0, column=0, sticky='w', padx=10, pady=(12, 4))
        combo = ttk.Combobox(dlg, width=28, state='readonly',
                             values=[self._fmt_channel(c) for c in self._channels])
        combo.grid(row=0, column=1, padx=10, pady=(12, 4))
        ttk.Label(dlg, text="Doublet cut (k):").grid(
            row=0, column=2, sticky='w', padx=10, pady=(12, 4))
        k_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(dlg, from_=0.5, to=5.0, increment=0.1, textvariable=k_var,
                    width=6).grid(row=0, column=3, sticky='w', padx=(0, 10),
                                  pady=(12, 4))
        ttk.Label(dlg, text="Singlet tol:").grid(
            row=1, column=2, sticky='w', padx=10, pady=4)
        stol_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(dlg, from_=0.05, to=1.0, increment=0.05, textvariable=stol_var,
                    width=6).grid(row=1, column=3, sticky='w', padx=(0, 10), pady=4)
        if default:
            combo.set(self._fmt_channel(default))
        elif self._channels:
            combo.set(self._fmt_channel(self._channels[0]))
        if not default:
            ttk.Label(dlg, text="(no DNA dye auto-detected — pick one)",
                      foreground='grey').grid(
                row=1, column=0, columnspan=2, sticky='w', padx=10)

        all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Run on all loaded samples",
                        variable=all_var).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=10, pady=(6, 4))

        btns = ttk.Frame(dlg)
        btns.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10,
                  pady=(6, 10))

        def do_run():
            col = self._resolve_channel(combo.get())
            k = float(k_var.get())
            stol = float(stol_var.get())
            dlg.destroy()
            if col:
                self._run_cell_cycle(col, all_var.get(), k=k, singlet_tol=stol)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Run", command=do_run).pack(
            side='right', padx=(0, 6))

    def _open_cluster_dialog(self):
        """Run unsupervised clustering (+ optional UMAP) on loaded samples,
        in a worker thread, then auto-import the resulting populations."""
        if not self._samples:
            self.status_var.set("Load a sample first.")
            return
        if getattr(self, '_clustering_busy', False):
            self.status_var.set("Clustering already running…")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Cluster")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.resizable(False, False)

        method_var = tk.StringVar(value='phenograph')
        mrow = ttk.Frame(dlg)
        mrow.grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=4)
        ttk.Label(mrow, text="Method:").pack(side='left')
        ttk.Radiobutton(mrow, text="Phenograph", value='phenograph',
                        variable=method_var).pack(side='left', padx=(6, 0))
        ttk.Radiobutton(mrow, text="FlowSOM", value='flowsom',
                        variable=method_var).pack(side='left', padx=(6, 0))
        ttk.Radiobutton(mrow, text="Leiden", value='leiden',
                        variable=method_var).pack(side='left', padx=(6, 0))

        ttk.Label(dlg, text="Phenograph/Leiden k:").grid(
            row=1, column=0, sticky='w', padx=10, pady=4)
        k_var = tk.IntVar(value=30)
        ttk.Spinbox(dlg, from_=5, to=200, textvariable=k_var, width=8).grid(
            row=1, column=1, sticky='w', padx=10, pady=4)

        ttk.Label(dlg, text="Leiden resolution:").grid(
            row=1, column=2, sticky='w', padx=10, pady=4)
        res_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(dlg, from_=0.1, to=5.0, increment=0.1, textvariable=res_var,
                    width=6).grid(row=1, column=3, sticky='w', padx=(0, 10),
                                  pady=4)

        ttk.Label(dlg, text="FlowSOM grid (NxN):").grid(row=2, column=0,
                                                        sticky='w', padx=10, pady=4)
        grid_var = tk.IntVar(value=10)
        ttk.Spinbox(dlg, from_=4, to=20, textvariable=grid_var, width=8).grid(
            row=2, column=1, sticky='w', padx=10, pady=4)

        ttk.Label(dlg, text="FlowSOM metaclusters:").grid(row=3, column=0,
                                                          sticky='w', padx=10, pady=4)
        meta_var = tk.IntVar(value=10)
        ttk.Spinbox(dlg, from_=2, to=40, textvariable=meta_var, width=8).grid(
            row=3, column=1, sticky='w', padx=10, pady=4)

        ttk.Label(dlg, text="Embedding (for visualisation):").grid(
            row=4, column=0, sticky='w', padx=10, pady=4)
        # Only offer embeddings whose backend is actually installed — picking a
        # missing one would silently produce nothing. (Independent of the
        # clustering method: any method pairs with any embedding.)
        avail_emb, missing_emb = self._available_embeddings()
        embed_var = tk.StringVar(
            value='UMAP' if 'UMAP' in avail_emb
            else (avail_emb[0] if avail_emb else 'none'))
        ttk.Combobox(dlg, textvariable=embed_var, width=10, state='readonly',
                     values=avail_emb + ['none']).grid(
            row=4, column=1, sticky='w', padx=10, pady=4)
        if missing_emb:
            ttk.Label(dlg, foreground='grey', font=('TkDefaultFont', 8),
                      text=f"({', '.join(missing_emb)} need "
                           "pip install openflo[embed])").grid(
                row=4, column=2, columnspan=2, sticky='w', padx=(0, 10))
        # Downsampling — clustering runs on FULL data by default; optionally
        # cap to the smallest sample in the group (so groups compare at equal N)
        # or to a custom number. Embeddings use the same cap when set.
        ttk.Label(dlg, text="Events / sample:").grid(
            row=5, column=0, sticky='w', padx=10, pady=4)
        dsrow = ttk.Frame(dlg)
        dsrow.grid(row=5, column=1, columnspan=3, sticky='w', padx=10, pady=4)
        ds_var = tk.StringVar(value='Full')
        ds_n_var = tk.StringVar(value='')
        ds_combo = ttk.Combobox(dsrow, textvariable=ds_var, width=18,
                                state='readonly',
                                values=['Full', 'Smallest in group', 'Custom…'])
        ds_combo.pack(side='left')
        ds_entry = ttk.Entry(dsrow, textvariable=ds_n_var, width=9)
        ds_entry.pack(side='left', padx=(6, 0))
        # Clicking or typing in the number box flips the mode to Custom
        # automatically — no second click on the dropdown needed.
        def _ds_to_custom(*_):
            if ds_var.get() != 'Custom…':
                ds_var.set('Custom…')
        ds_entry.bind('<FocusIn>', _ds_to_custom)
        ds_entry.bind('<Button-1>', _ds_to_custom)
        ds_entry.bind('<Key>', _ds_to_custom)
        _sizes = [len(s.data) for s in self._samples.values()
                  if getattr(s, 'data', None) is not None and len(s.data)]
        if _sizes:
            ttk.Label(dsrow, foreground='grey',
                      text=(f"loaded: {min(_sizes):,}–{max(_sizes):,} ev"
                            if min(_sizes) != max(_sizes)
                            else f"loaded: {_sizes[0]:,} ev")).pack(
                side='left', padx=(8, 0))

        # Markers to cluster on — defaults to ALL fluorochrome channels (the
        # historical behaviour), but you can now restrict to a subset.
        ttk.Label(dlg, text="Markers:").grid(row=6, column=0, sticky='nw',
                                             padx=10, pady=4)
        mk_frame = ttk.Frame(dlg)
        mk_frame.grid(row=6, column=1, columnspan=3, sticky='w', padx=10, pady=4)
        _act = self._samples.get(self._active_sample)
        _fluor = [str(c) for c in getattr(_act, 'fluor_channels', [])] \
            if _act is not None else []
        mk_list = tk.Listbox(mk_frame, selectmode='multiple', exportselection=False,
                             height=min(6, max(3, len(_fluor))), width=22)
        for c in _fluor:
            mk_list.insert('end', c)
        for i in range(len(_fluor)):
            mk_list.selection_set(i)          # all selected = default
        mk_sb = ttk.Scrollbar(mk_frame, orient='vertical', command=mk_list.yview)
        mk_list.config(yscrollcommand=mk_sb.set)
        mk_list.pack(side='left')
        mk_sb.pack(side='left', fill='y')
        ttk.Label(mk_frame, foreground='grey',
                  text="(all selected = default)").pack(side='left', padx=(8, 0))

        all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, text="Run on all loaded samples",
                        variable=all_var).grid(row=7, column=0, columnspan=2,
                                               sticky='w', padx=10, pady=4)

        btns = ttk.Frame(dlg)
        btns.grid(row=8, column=0, columnspan=2, sticky='ew', padx=10, pady=4)

        def do_run():
            sel = [mk_list.get(i) for i in mk_list.curselection()]
            # None = all markers (keeps the previous behaviour + pipeline default)
            channels = None if (not sel or len(sel) == len(_fluor)) else sel
            params: dict = dict(
                method=method_var.get(), all_samples=all_var.get(),
                k=int(k_var.get()), grid=int(grid_var.get()),
                n_meta=int(meta_var.get()), embedding=embed_var.get(),
                resolution=float(res_var.get()), channels=channels,
                downsample=ds_var.get(), custom_n=ds_n_var.get())
            dlg.destroy()
            self._run_clustering(**params)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Run", command=do_run).pack(
            side='right', padx=(0, 6))

    def _run_clustering(self, method, all_samples, k, grid, n_meta,
                        embedding='UMAP', resolution=1.0, channels=None,
                        downsample='Full', custom_n=''):
        targets = [n for n in (self._sample_order if all_samples
                               else [self._active_sample])
                   if n in self._samples]
        if not targets:
            self.status_var.set("No sample selected.")
            return
        # Resolve the per-sample event cap (None = full data).
        cap = None
        if downsample == 'Smallest in group':
            szs = [len(self._samples[t].data) for t in targets
                   if getattr(self._samples[t], 'data', None) is not None]
            cap = min(szs) if szs else None
        elif downsample == 'Custom…':
            try:
                cap = max(1, int(float(str(custom_n).replace(',', '').strip())))
            except (ValueError, TypeError):
                cap = None
        self._clustering_busy = True
        n = len(targets)
        cap_note = f" · {cap:,} ev/sample" if cap else " · full data"
        emb_method, emb_prefix = self._EMBEDDINGS.get(embedding, (None, None))

        def work():
            for i, name in enumerate(targets, 1):
                s = self._samples.get(name)
                if s is None:
                    continue
                self._busy(f"{method}: clustering {name} ({i}/{n})…")
                if method == 'phenograph':
                    s.cluster(channels=channels, k=k, max_events=cap)
                elif method == 'leiden':
                    s.run_leiden(channels=channels, n_neighbors=k,
                                 resolution=resolution, max_events=(cap or 200_000))
                else:
                    s.run_flowsom(channels=channels, grid=(grid, grid),
                                  n_metaclusters=n_meta)
                if emb_method:
                    self._busy(
                        f"{embedding} embedding on {name} ({i}/{n})… "
                        "first run compiles, ~30-50s — still working")
                    getattr(s, emb_method)(
                        **({'sample_n': cap} if cap else {}))

        self.run_async(
            work,
            on_done=lambda _r: self._finish_clustering(
                method, emb_prefix, targets),
            on_error=lambda exc: self._clustering_error(exc),
            busy_msg=f"{method} on {n} sample(s){cap_note}…")

    def _open_dr_compare(self):
        """Open the embedding-comparison setup dialog (pick methods + cell
        count), then run the chosen embeddings in the background."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        if getattr(self, '_dr_running', False):
            self.status_var.set("An embedding run is already in progress…")
            return
        s = self._samples[name]
        df = s.data
        chans = [c for c in (getattr(s, 'fluor_channels', None) or df.columns)
                 if c in df.columns and df[c].dtype.kind in 'fiu']
        if len(chans) < 2:
            messagebox.showinfo("Embedding comparison",
                                "Need at least 2 numeric channels.", parent=self)
            return
        from .dr_compare import available_methods
        have = available_methods()
        if not have:
            messagebox.showinfo("Embedding comparison",
                                "No embedding backends are installed.\n\n"
                                "pip install \"openflo[embed]\" adds them.",
                                parent=self)
            return
        # Pass the frame + channels (NOT a materialised array) so opening the
        # dialog is instant on a million-row sample — the heavy to_numpy() runs
        # later, in the background worker.
        from .ui_embedding import EmbeddingDialog
        EmbeddingDialog(self, name, len(df), have, df, chans)

    def _open_group_stats(self):
        """Compare a channel's per-sample median across trial groups."""
        from .ui_group_stats import GroupStatsWindow
        if len(self._samples) < 2:
            self.status_var.set(
                "Load samples in ≥2 groups (trials) to compare.")
            return
        self._show_or_raise('group_stats', lambda: GroupStatsWindow(self))

    def _open_methods_report(self):
        """Show the paper-ready Methods paragraph + run manifest."""
        from .ui_methods import MethodsWindow
        self._show_or_raise('methods', lambda: MethodsWindow(self))

    def _loaded_samples(self):
        """FlowSample objects for every loaded sample, in load order."""
        return [self._samples[n] for n in self._target_samples('all')]

    def _fluor_panel_warning(self):
        """'' when all loaded samples share a fluor panel (by antibody
        label), else a message listing the non-common labels. Cross-
        sample stats/comparison tie by label, so a sample missing a
        marker just won't contribute to that label's column."""
        samples = self._loaded_samples()
        if len(samples) < 2:
            return ''
        from .pipeline import common_fluor_warning
        return common_fluor_warning(samples)

    def _open_stats_window(self):
        from .ui_statistics import StatisticsWindow
        if not self._samples:
            self.status_var.set("Load a sample first to compute statistics.")
            return
        self._show_or_raise('stats', lambda: StatisticsWindow(self))

    def _open_frequency_window(self):
        from .ui_frequency import FrequencyComparisonWindow
        if not self._samples:
            self.status_var.set("Load samples first to compare frequencies.")
            return
        self._show_or_raise('frequency',
                            lambda: FrequencyComparisonWindow(self))

    def _open_trajectory_window(self):
        from .ui_trajectory import TrajectoryWindow
        if not self._samples:
            self.status_var.set("Load samples first to infer a trajectory.")
            return
        self._show_or_raise('trajectory', lambda: TrajectoryWindow(self))

    def _open_flowsom_tree(self):
        from .ui_flowsom_tree import FlowSOMTreeWindow
        name = self._active_sample
        s = self._samples.get(name) if name else None
        if s is None or not getattr(s, 'flowsom_result', None):
            self.status_var.set(
                "Run FlowSOM first (Cluster… → FlowSOM), then SOM tree.")
            return
        FlowSOMTreeWindow(self, name)

    def _open_annotation_window(self):
        from .ui_annotation import PopulationAnnotationWindow
        name = self._active_sample
        s = self._samples.get(name) if name else None
        if s is None:
            self.status_var.set("Select a clustered sample to annotate.")
            return
        if not any(c in s.data.columns
                   for c in ('leiden', 'cluster', 'flowsom_meta')):
            self.status_var.set(
                "Cluster the sample first (Cluster… → Phenograph/FlowSOM/"
                "Leiden), then Annotate.")
            return
        PopulationAnnotationWindow(self, name)

    def _apply_population_names(self, sample, label_col, names):
        """Write annotation names onto a sample's populations: into
        ``_cluster_labels`` (for the cluster path) and onto any existing
        cluster/category gate for that label value, then refresh the tree."""
        store = self._cluster_labels.setdefault(sample, {})
        for cid, nm in names.items():
            store[cid] = nm
        gates = self._sample_gates.get(sample, {})
        for g in gates.values():
            if g.get('kind') == 'cluster' and g.get('cluster_id') in names:
                g['name'] = names[g['cluster_id']]
            elif (g.get('kind') == 'category'
                  and g.get('channel') == label_col
                  and g.get('value') in names):
                g['name'] = names[g['value']]
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _open_expression_window(self):
        from .ui_expression import MarkerExpressionWindow
        if not self._samples:
            self.status_var.set("Load samples first to compare expression.")
            return
        self._show_or_raise('expression',
                            lambda: MarkerExpressionWindow(self))

    def _open_sample_qc_window(self):
        from .ui_sample_qc import SampleQCWindow
        if len(self._selected_samples()) < 2:
            self.status_var.set(
                "Enable ≥2 samples (☑) to compare them.")
            return
        self._show_or_raise('sample_qc', lambda: SampleQCWindow(self))
