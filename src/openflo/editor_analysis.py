"""Analyze-menu launchers + clustering / annotation orchestration.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from . import ui_errors
from .cluster_caps import LEIDEN_FAST_GRAPH_DEFAULT
from .dialog_inputs import number_or, read_int, read_number
from .editor_base import EditorMixin
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS
from .tool_window import (
    CLUSTER_RESOLUTION_TIP,
    FAST_GRAPH_TIP,
    MIN_CUSTOM_EVENTS,
    scrolled_body,
)
from .ui_logic import resolve_channel
from .ui_tips import auto_tip

# The Cluster dialog's starting FlowSOM metacluster count (the shared FlowSOM
# default; the old name stays as an alias), and the hover tip beside it (see
# _open_cluster_dialog for the measurement behind both).
FLOWSOM_DIALOG_METACLUSTERS = FLOWSOM_DEFAULT_METACLUSTERS
FLOWSOM_METACLUSTER_TIP = (
    "How many groups FlowSOM makes. Set it ABOVE the number of populations "
    "you expect: asked for exactly that many, FlowSOM tends to merge a rare "
    "population into a neighbor. Extra metaclusters are cheap: join two "
    "that are one population with an OR gate (right-click a gate > Create "
    "boolean gate…). 20 is a good start for most panels; raise it for large "
    "panels or very rare populations.")



# The Cluster dialog's Events / sample choices. Default clusters at most the
# method's own cap (cluster_caps); All events clusters every event, after a
# warning with the memory it needs for a graph method.
DEFAULT_EVENTS = 'Default'
ALL_EVENTS = 'All events'
EVENTS_CHOICES = (DEFAULT_EVENTS, ALL_EVENTS, 'Smallest in group', 'Custom…')


class ClusteringCanceled(Exception):
    """Raised inside a clustering run once its Cancel was pressed."""


def method_max_events(method, cap, uncapped, n_rows):
    """The ``max_events`` a Cluster dialog run passes `method` for a
    sample of `n_rows` rows: the user's cap, else every event (All
    events), else the method's default cap."""
    from .cluster_caps import default_cap
    if cap:
        return cap
    if uncapped:
        # FlowSOM's trainer takes a number (no None): the sample's size.
        return max(int(n_rows), 1) if method == 'flowsom' else None
    return default_cap(method)


def _cap_note(method, cap, uncapped):
    """The busy message's events part for a run."""
    from .cluster_caps import default_cap
    if cap:
        return f" · {cap:,} ev/sample"
    if uncapped:
        return " · all events"
    own = default_cap(method)
    return f" · up to {own:,} ev/sample" if own else ""


def _clustering_settings(method, k, grid, n_meta, resolution, reproducible,
                         fast_graph, emb_kw):
    """The settings a run used, for the audit trail (and so the Methods
    paragraph and a reopened session can say how populations were made)."""
    settings: dict = (
        {'grid': grid, 'n_metaclusters': n_meta} if method == 'flowsom'
        else {'k': k, 'resolution': float(resolution)})
    if method == 'phenograph':
        settings['reproducible'] = bool(reproducible)
    if method == 'leiden':
        settings['fast_graph'] = bool(fast_graph)
    if emb_kw:
        settings['embedding_components'] = emb_kw['n_components']
    return settings

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
        vals = self._sample_label_values(name, col, include_sentinel=True)
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
                  font='OpenFloHeading').pack(
            side='top', fill='x', padx=10, pady=(10, 6))
        inner = scrolled_body(dlg, pack_kw={'padx': 10, 'pady': (0, 6)})

        entries = {}
        for v in vals:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=self._population_label(col, v),
                      width=20).pack(side='left')
            g = _gate_for(v)
            cur = (g.get('name') if g else None) or self._population_label(col, v)
            var = tk.StringVar(value=cur)
            ent = ttk.Entry(row, textvariable=var, width=22)
            ent.pack(side='left', fill='x', expand=True)
            self._tip(ent, "The name this population shows in the gate "
                      "list, plots and exports, e.g. 'T cells'. Leave it "
                      f"blank to go back to '{col} {v}'.")
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
        apply_b = ttk.Button(btns, text="Apply", default='active',
                             command=do_apply)
        apply_b.pack(side='right', padx=(0, 6))
        self._tip(apply_b, f"Rename this sample's {disp} populations to the "
                  "names typed above. Only the names change, not which "
                  "events belong. One undo step.")

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
                  font='OpenFloHeading').pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        inner = scrolled_body(dlg, pack_kw={'padx': 10, 'pady': (0, 6)})

        from .cluster_labels import cluster_label
        entries = {}
        for cid in ids:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=cluster_label(cid), width=20).pack(side='left')
            var = tk.StringVar(value=self._cluster_display_name(name, cid))
            ent = ttk.Entry(row, textvariable=var, width=24)
            ent.pack(side='left', fill='x', expand=True)
            self._tip(ent, "A cell-type name for this cluster, e.g. "
                      "'B cells'. It replaces the cluster's number in the "
                      "gate list, plots and exports. Leave it blank to go "
                      "back to the number.")
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
                        g['name'] = txt or cluster_label(cid)
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Updated cluster names for '{name}'.")

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        apply_b = ttk.Button(btns, text="Apply", default='active',
                             command=do_apply)
        apply_b.pack(side='right', padx=(0, 6))
        self._tip(apply_b, "Give this sample's clusters the names typed "
                  "above. Only the names change, not which events belong. "
                  "One undo step.")

    CELL_CYCLE_MODELS = (("Dean-Jett-Fox", 'dean-jett-fox'),
                         ("Boundaries", 'boundaries'))

    def _open_cell_cycle_dialog(self):
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Load and select a sample first.")
            return
        from .cell_cycle import find_dna_channel
        s = self._samples[name]
        default = find_dna_channel(s)
        # Cluster labels, embedding coordinates, cell-cycle phases and Time
        # are not DNA content: leave them out of the list.
        from .channel_roles import is_excluded
        dna_chans = [c for c in self._channels if not is_excluded(str(c))]

        dlg = tk.Toplevel(self)
        dlg.title(f"Cell cycle — {name}")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.resizable(False, False)

        ttk.Label(dlg, text="DNA-content channel:").grid(
            row=0, column=0, sticky='w', padx=10, pady=(12, 4))
        combo = ttk.Combobox(dlg, width=28, state='readonly',
                             values=[self._fmt_channel(c) for c in dna_chans])
        combo.grid(row=0, column=1, padx=10, pady=(12, 4))
        self._tip(combo, "The channel of the DNA dye (e.g. a DAPI or PI "
                  "channel). Its intensity measures how much DNA each cell "
                  "has, which tells G1, S and G2/M apart.")
        ttk.Label(dlg, text="Doublet cut (k):").grid(
            row=0, column=2, sticky='w', padx=10, pady=(12, 4))
        k_var = tk.DoubleVar(value=1.5)
        k_spin = ttk.Spinbox(dlg, from_=0.5, to=5.0, increment=0.1,
                             textvariable=k_var, width=6)
        k_spin.grid(row=0, column=3, sticky='w', padx=(0, 10), pady=(12, 4))
        self._tip(k_spin, "How wide the G1 and G2/M peaks are taken to be, "
                  "in robust standard deviations either side of each peak "
                  "(0.5 to 5, default 1.5). Cells between the peaks count "
                  "as S phase; a larger value counts fewer as S.")
        ttk.Label(dlg, text="Singlet tol:").grid(
            row=1, column=2, sticky='w', padx=10, pady=4)
        stol_var = tk.DoubleVar(value=0.25)
        stol_spin = ttk.Spinbox(dlg, from_=0.05, to=1.0, increment=0.05,
                                textvariable=stol_var, width=6)
        stol_spin.grid(row=1, column=3, sticky='w', padx=(0, 10), pady=4)
        self._tip(stol_spin, "Two cells stuck together look like one G2/M "
                  "cell, so they are dropped first: a cell whose DNA area "
                  "to width ratio is further than this from the typical "
                  "single cell, as a fraction (0.05 to 1, default 0.25 = "
                  "25%). Smaller drops more. Needs a DNA width or height "
                  "channel.")
        if default:
            combo.set(self._fmt_channel(default))
        elif dna_chans:
            combo.set(self._fmt_channel(dna_chans[0]))
        if not default:
            ttk.Label(dlg, text="(no DNA dye auto-detected — pick one)",
                      style='Muted.TLabel').grid(
                row=1, column=0, columnspan=2, sticky='w', padx=10)

        all_var = tk.BooleanVar(value=False)
        all_cb = ttk.Checkbutton(dlg, text="Run on all loaded samples",
                                 variable=all_var)
        all_cb.grid(row=2, column=0, columnspan=2, sticky='w', padx=10,
                    pady=(6, 4))
        self._tip(all_cb, "Off: analyze only the selected sample. On: "
                  "analyze every loaded sample with these settings.")
        # The Dean-Jett-Fox fit is the literature standard:
        # S-phase cells under the G1 and G2 peaks count as S.
        ttk.Label(dlg, text="Model:").grid(row=2, column=2, sticky='w',
                                           padx=10, pady=(6, 4))
        model_var = tk.StringVar(value=self.CELL_CYCLE_MODELS[0][0])
        model_cb = ttk.Combobox(dlg, textvariable=model_var, width=14,
                                state='readonly',
                                values=[m for m, _k in self.CELL_CYCLE_MODELS])
        model_cb.grid(row=2, column=3, sticky='w', padx=(0, 10), pady=(6, 4))
        auto_tip(model_cb,
                 "Dean-Jett-Fox fits G1 and G2/M peaks and a polynomial S "
                 "phase, and reports their areas. Boundaries counts the "
                 "events between intensity cut-offs, which gives the S-phase "
                 "cells under the peaks to G1 and G2/M.")
        self._cell_cycle_model_var = model_var

        btns = ttk.Frame(dlg)
        btns.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10,
                  pady=(6, 10))

        def do_run():
            col = resolve_channel(combo.get())
            k = read_number(k_var, "Doublet cut (k)", lo=0.5, hi=5.0,
                            parent=dlg)
            if k is None:
                return
            stol = read_number(stol_var, "Singlet tol", lo=0.05, hi=1.0,
                               parent=dlg)
            if stol is None:
                return
            method = dict(self.CELL_CYCLE_MODELS)[model_var.get()]
            dlg.destroy()
            if col:
                self._run_cell_cycle(col, all_var.get(), k=k, singlet_tol=stol,
                                     method=method)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        run_b = ttk.Button(btns, text="Run", default='active', command=do_run)
        run_b.pack(side='right', padx=(0, 6))
        self._tip(run_b, "Fit the cell cycle and add G1, S and G2/M as "
                  "populations, with their percentages.")

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

        method_var = self._cluster_method_picker(dlg)
        (k_var, res_var, grid_var, meta_var, fast_var, rows, res_note,
         method_settings) = self._cluster_setting_rows(dlg)
        embed_var, dims_var = self._cluster_embedding_row(dlg)
        ds_var, ds_n_var = self._cluster_events_row(dlg)
        mk_list, _fluor, _default = self._cluster_markers_box(dlg)
        all_var, repro_var, repro_cb = self._cluster_run_options(dlg)
        # After the rows loop and before _show_method_settings first runs,
        # as when the dialog was one function.
        rows['repro'] = (repro_cb,)

        def _resolution_applies():
            return method_var.get() == 'leiden' or (
                method_var.get() == 'phenograph' and repro_var.get())

        def _show_method_settings(*_a):
            wanted = method_settings.get(method_var.get(), set())
            for key, widgets in rows.items():
                for w in widgets:
                    if key in wanted:
                        w.grid()
                    else:
                        w.grid_remove()
            rows['resolution'][1].configure(
                state='normal' if _resolution_applies() else 'disabled')
            if (method_var.get() == 'phenograph'
                    and not _resolution_applies()):
                res_note.grid()
            else:
                res_note.grid_remove()
        method_var.trace_add('write', _show_method_settings)
        repro_var.trace_add('write', _show_method_settings)
        _show_method_settings()
        self._cluster_resolution_note = res_note  # for tests
        self._cluster_method_rows = rows         # for tests

        btns = ttk.Frame(dlg)
        btns.grid(row=9, column=0, columnspan=4, sticky='ew', padx=10, pady=4)

        def do_run():
            sel = [_fluor[i] for i in mk_list.curselection()]
            # None = the pipeline default (what is pre-ticked), resolved per
            # sample when the run covers every loaded sample.
            channels = None if (not sel or set(sel) == _default) else sel
            # Only the picked method's settings are shown, so only those are
            # checked; the others keep their defaults if left unreadable.
            shown = {'phenograph': ('k', 'resolution'),
                     'leiden': ('k', 'resolution'),
                     'flowsom': ('grid', 'meta')}.get(method_var.get(), ())
            if not _resolution_applies():
                shown = tuple(x for x in shown if x != 'resolution')
            values = {}
            for key, label, var, lo, hi, integer, default in (
                    ('k', "Neighbors (k)", k_var, 5, 200, True, 30),
                    ('resolution', "Resolution", res_var, 0.1, 5.0,
                     False, 1.0),
                    ('grid', "FlowSOM grid (N×N)", grid_var, 4, 20, True, 10),
                    ('meta', "FlowSOM metaclusters", meta_var, 2, 40, True,
                     FLOWSOM_DIALOG_METACLUSTERS)):
                if key not in shown:
                    values[key] = number_or(var, default, integer=integer,
                                            lo=lo, hi=hi)
                    continue
                values[key] = read_number(var, label, integer=integer, lo=lo,
                                          hi=hi, parent=dlg)
                if values[key] is None:
                    return
            custom_n = None
            if ds_var.get() == 'Custom…':
                # A cap below the floor or a word is refused, and the dialog
                # stays open.
                custom_n = read_int(ds_n_var, "Events / sample (Custom…)",
                                    lo=MIN_CUSTOM_EVENTS, parent=dlg,
                                    thousands=True)
                if custom_n is None:
                    return
            params: dict = {
                'method': method_var.get(), 'all_samples': all_var.get(),
                'k': values['k'], 'grid': values['grid'],
                'n_meta': values['meta'], 'embedding': embed_var.get(),
                'embed_dims': (int(dims_var.get())
                               if embed_var.get() == 'UMAP' else 2),
                # Unused (greyed out) it is not sent: 1.0 changes nothing.
                'resolution': (values['resolution'] if _resolution_applies()
                               else 1.0),
                'channels': channels,
                'reproducible': repro_var.get(),
                'fast_graph': (method_var.get() == 'leiden'
                               and fast_var.get()),
                'downsample': ds_var.get(), 'custom_n': custom_n}
            dlg.destroy()
            self._run_clustering(**params)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        run_b = ttk.Button(btns, text="Run", default='active', command=do_run)
        run_b.pack(side='right', padx=(0, 6))
        self._tip(run_b, "Start clustering in the background. When it "
                  "finishes, each cluster is added as a population you can "
                  "gate, plot and rename.")

    def _cluster_method_picker(self, dlg):
        """The Cluster dialog's Method row: one radio button per
        clustering method, on a new StringVar the caller keeps."""
        method_var = tk.StringVar(value='phenograph')
        mrow = ttk.Frame(dlg)
        mrow.grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=4)
        ttk.Label(mrow, text="Method:").pack(side='left')
        for text, value, why in (
                ("PhenoGraph", 'phenograph',
                 "Groups each cell with its nearest look-alikes and finds "
                 "the communities. Good at rare populations; slower on big "
                 "samples."),
                ("FlowSOM", 'flowsom',
                 "Sorts cells onto a grid of similar types, then merges "
                 "them into metaclusters. Fast, even on millions of cells."),
                ("Leiden", 'leiden',
                 "Like PhenoGraph (a nearest-neighbor graph), with the "
                 "Leiden method for the communities. Always reproducible.")):
            rb = ttk.Radiobutton(mrow, text=text, value=value,
                                 variable=method_var)
            rb.pack(side='left', padx=(6, 0))
            self._tip(rb, why)
        return method_var

    def _cluster_setting_rows(self, dlg):
        """The Cluster dialog's per-method setting rows (spinboxes, the
        resolution note, Fast graph, the metacluster note).

        Returns their Tk variables, the rows by key, the resolution
        note and which rows each method shows; the caller keeps the
        variables (a tk.Variable left only in a local is unset when
        it goes)."""
        # Each method's own settings, shown only while it is picked:
        # neighbors and resolution for PhenoGraph and Leiden, the grid and
        # metaclusters for FlowSOM. The two graph methods share one
        # resolution, as openflo-run's --resolution and the workspace do;
        # both start at 1.0, their standard result.
        k_var = tk.IntVar(value=30)
        res_var = tk.DoubleVar(value=1.0)
        grid_var = tk.IntVar(value=10)
        meta_var = tk.IntVar(value=FLOWSOM_DIALOG_METACLUSTERS)
        rows = {}
        for row, (key, label, var, lo, hi, step) in enumerate((
                ('k', "Neighbors (k):", k_var, 5, 200, 1),
                ('resolution', "Resolution:", res_var, 0.1, 5.0, 0.1),
                ('grid', "FlowSOM grid (N×N):", grid_var, 4, 20, 1),
                ('meta', "FlowSOM metaclusters:", meta_var, 2, 40, 1)),
                start=1):
            lbl = ttk.Label(dlg, text=label)
            lbl.grid(row=row, column=0, sticky='w', padx=10, pady=4)
            spin = ttk.Spinbox(dlg, from_=lo, to=hi, increment=step,
                               textvariable=var, width=8)
            spin.grid(row=row, column=1, sticky='w', padx=10, pady=4)
            rows[key] = (lbl, spin)
        for w in rows['resolution']:
            self._tip(w, CLUSTER_RESOLUTION_TIP)
        for w in rows['k']:
            self._tip(w, "How many look-alike neighbors each cell is "
                      "linked to (5 to 200, default 30). More neighbors "
                      "give fewer, larger clusters; fewer can split off "
                      "small populations.")
        for w in rows['grid']:
            self._tip(w, "The size of FlowSOM's map: N×N cell types before "
                      "merging (4 to 20, default 10, so 100). A bigger map "
                      "picks up rarer cells but takes longer.")
        # PhenoGraph's resolution exists only on its reproducible (seeded
        # Leiden) route: its Louvain runs the Blondel binaries, which take
        # none. So with Reproducible off the box is greyed out and says why.
        res_note = ttk.Label(dlg, style='Muted.TLabel',
                             text="(PhenoGraph: needs Reproducible)")
        res_note.grid(row=2, column=2, columnspan=2, sticky='w', padx=(0, 10))
        self._tip(res_note, "PhenoGraph's non-reproducible Louvain has no "
                  "resolution setting. Tick Reproducible to use one.")
        # Leiden's pruned graph (FlowSample.run_leiden fast_graph), as in
        # the workspace and openflo-run --fast-graph. On by default: the
        # dense graph needed about 21 GB at 200,000 events (cluster_caps).
        fast_var = tk.BooleanVar(value=LEIDEN_FAST_GRAPH_DEFAULT)
        fast_cb = ttk.Checkbutton(dlg, text="Fast graph", variable=fast_var)
        fast_cb.grid(row=1, column=2, columnspan=2, sticky='w', padx=(0, 10))
        self._tip(fast_cb, FAST_GRAPH_TIP)
        rows['fast'] = (fast_cb,)
        method_settings = {'phenograph': {'k', 'resolution', 'repro'},
                           'leiden': {'k', 'resolution', 'fast'},
                           'flowsom': {'grid', 'meta'}}
        # FlowSOM asked for exactly as many metaclusters as there are
        # populations merges a rare one into a neighbour: on 30-channel
        # synthetic data with eight populations, 8 metaclusters found the
        # 1.5% population with F1 0.22 and 12 with F1 0.99
        # (tests/test_cluster_recovery_highdim.py). Its authors advise asking
        # for more than you expect and merging afterwards, so the dialog
        # starts at 20 and says why.
        meta_note = ttk.Label(dlg, style='Muted.TLabel',
                              text="(set above the populations you expect)")
        meta_note.grid(row=4, column=2, columnspan=2, sticky='w', padx=(0, 10))
        rows['meta'] += (meta_note,)
        for w in rows['meta'][1:]:
            self._tip(w, FLOWSOM_METACLUSTER_TIP)
        return (k_var, res_var, grid_var, meta_var, fast_var, rows, res_note,
                method_settings)

    def _cluster_embedding_row(self, dlg):
        """The Cluster dialog's Embedding row, offering only the
        installed embeddings, and its Components box (2 or 3, UMAP only);
        returns both variables for the caller."""
        ttk.Label(dlg, text="Embedding (for visualization):").grid(
            row=5, column=0, sticky='w', padx=10, pady=4)
        # Only offer embeddings whose backend is actually installed — picking a
        # missing one would silently produce nothing. (Independent of the
        # clustering method: any method pairs with any embedding.)
        avail_emb, missing_emb = self._available_embeddings()
        embed_var = tk.StringVar(
            value='UMAP' if 'UMAP' in avail_emb
            else (avail_emb[0] if avail_emb else 'none'))
        embrow = ttk.Frame(dlg)
        embrow.grid(row=5, column=1, sticky='w', padx=10, pady=4)
        embed_cb = ttk.Combobox(embrow, textvariable=embed_var, width=10,
                                state='readonly', values=avail_emb + ['none'])
        embed_cb.pack(side='left')
        self._tip(embed_cb, "Also make a 2-D map of the cells to look at the "
                  "clusters (UMAP is the usual choice). It is for viewing "
                  "only and does not change the clusters; none skips it "
                  "and saves time.")
        ttk.Label(embrow, text="Components:").pack(side='left', padx=(8, 2))
        dims_var = tk.StringVar(value='2')
        dims_cb = ttk.Combobox(embrow, textvariable=dims_var, width=3,
                               state='readonly', values=['2', '3'])
        dims_cb.pack(side='left')
        self._tip(dims_cb, "UMAP only. 2 (default) makes UMAP1 and UMAP2. "
                  "3 also makes UMAP3 and opens Analyze > 3D plot on the "
                  "three; the first two are then a different map from a "
                  "2-component run, so compare runs with the same number.")

        def _dims_state(*_a):
            on = embed_var.get() == 'UMAP'
            dims_cb.state(['!disabled', 'readonly'] if on else ['disabled'])
        embed_var.trace_add('write', _dims_state)
        _dims_state()
        self._cluster_dims_combo = dims_cb      # for tests
        if missing_emb:
            ttk.Label(dlg, style='Muted.TLabel', font='OpenFloSmall',
                      text=f"({', '.join(missing_emb)} need "
                           "pip install openflo[embed])").grid(
                row=5, column=2, columnspan=2, sticky='w', padx=(0, 10))
        return embed_var, dims_var

    def _cluster_events_row(self, dlg):
        """The Cluster dialog's Events / sample row (Default, All events,
        Smallest in group or a custom number); returns its two variables."""
        # Default clusters at most each method's own cap (cluster_caps);
        # optionally every event, the smallest sample's size in the group (so
        # groups compare at equal N) or a custom number. Embeddings use the
        # same cap when one of the user's own is set.
        ttk.Label(dlg, text="Events / sample:").grid(
            row=6, column=0, sticky='w', padx=10, pady=4)
        dsrow = ttk.Frame(dlg)
        dsrow.grid(row=6, column=1, columnspan=3, sticky='w', padx=10, pady=4)
        ds_var = tk.StringVar(value=DEFAULT_EVENTS)
        ds_n_var = tk.StringVar(value='')
        ds_combo = ttk.Combobox(dsrow, textvariable=ds_var, width=18,
                                state='readonly', values=list(EVENTS_CHOICES))
        ds_combo.pack(side='left')
        self._tip(ds_combo, "How many events of each sample to cluster. "
                  "Default: at most 200,000 for PhenoGraph and Leiden and "
                  "50,000 for FlowSOM; every other event gets the cluster "
                  "of its nearest clustered event. All events clusters "
                  "every event, which can need several GB of memory on a "
                  "big sample (you are warned first). Smallest in group "
                  "takes the same number from each sample, so groups "
                  "compare fairly. Custom uses the number typed to the "
                  "right. Fewer events run faster.")
        ds_entry = ttk.Entry(dsrow, textvariable=ds_n_var, width=9)
        ds_entry.pack(side='left', padx=(6, 0))
        self._tip(ds_entry, "The number of events to take from each sample "
                  f"(e.g. 50000, at least {MIN_CUSTOM_EVENTS}), picked at "
                  "random. Typing here switches the choice on the left to "
                  "Custom.")
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
            ttk.Label(dsrow, style='Muted.TLabel',
                      text=(f"loaded: {min(_sizes):,}–{max(_sizes):,} ev"
                            if min(_sizes) != max(_sizes)
                            else f"loaded: {_sizes[0]:,} ev")).pack(
                side='left', padx=(8, 0))
        return ds_var, ds_n_var

    def _cluster_markers_box(self, dlg):
        """The Cluster dialog's Markers list with the default markers
        ticked; returns the list, the channels it shows (in order)
        and the default set."""
        # Markers to cluster on. Every fluorescence channel is listed; one
        # per marker (its -A, else its -H; never -W) is ticked, as
        # FlowSample's channels=None picks them (default_cluster_channels).
        ttk.Label(dlg, text="Markers:").grid(row=7, column=0, sticky='nw',
                                             padx=10, pady=4)
        mk_frame = ttk.Frame(dlg)
        mk_frame.grid(row=7, column=1, columnspan=3, sticky='w', padx=10, pady=4)
        _act = self._samples.get(self._active_sample)
        _fluor = [str(c) for c in getattr(_act, 'fluor_channels', [])] \
            if _act is not None else []
        # Listed as "marker (channel)" where the channel has a marker name,
        # as the plot axes are, and mapped back to channels to run.
        _shown = [self._fmt_channel(c) for c in _fluor]
        mk_list = tk.Listbox(mk_frame, selectmode='multiple', exportselection=False,
                             height=min(6, max(3, len(_fluor))),
                             width=max(22, min(40, max(map(len, _shown),
                                                       default=0) + 2)))
        for c in _shown:
            mk_list.insert('end', c)
        from .channel_roles import default_cluster_channels
        _default = set(default_cluster_channels(_fluor))
        for i, c in enumerate(_fluor):
            if c in _default:
                mk_list.selection_set(i)      # ticked = default
        mk_sb = ttk.Scrollbar(mk_frame, orient='vertical', command=mk_list.yview)
        mk_list.config(yscrollcommand=mk_sb.set)
        mk_list.pack(side='left')
        self._tip(mk_list, "The markers the clustering compares cells on. "
                  "Click a marker to add or remove it. Leave out viability "
                  "and dump channels, which would group cells by the wrong "
                  "thing.")
        mk_sb.pack(side='left', fill='y')
        ttk.Label(mk_frame, style='Muted.TLabel',
                  text="(ticked = one channel per marker, the default)").pack(
            side='left', padx=(8, 0))
        self._cluster_marker_list = mk_list      # for tests
        return mk_list, _fluor, _default

    def _cluster_run_options(self, dlg):
        """The Cluster dialog's Run on all loaded samples and
        Reproducible check boxes; returns both variables and the
        Reproducible box (the caller adds it to the method rows)."""
        all_var = tk.BooleanVar(value=False)
        all_cb = ttk.Checkbutton(dlg, text="Run on all loaded samples",
                                 variable=all_var)
        all_cb.grid(row=8, column=0, columnspan=2, sticky='w', padx=10,
                    pady=4)
        self._tip(all_cb, "Off: cluster only the selected sample. On: "
                  "cluster every loaded sample in turn, each on its own, "
                  "with these settings.")
        # PhenoGraph's default Louvain is not seed-reproducible; this routes it
        # to the seeded Leiden backend so a re-run gives identical labels. On
        # by default, as in FlowSample.cluster, the CLI and the workspace since
        # 2.4.9: two unseeded runs of the same data agreed only to ARI 0.81.
        repro_var = tk.BooleanVar(value=True)
        repro_cb = ttk.Checkbutton(
            dlg, text="Reproducible (seeded PhenoGraph)", variable=repro_var)
        repro_cb.grid(row=8, column=2, columnspan=2, sticky='w', padx=10,
                      pady=4)
        self._tip(repro_cb, "On (default): running PhenoGraph again on the "
                  "same data gives exactly the same clusters. Off: its "
                  "original method, whose clusters can differ from run to "
                  "run, and which has no resolution setting.")
        return all_var, repro_var, repro_cb

    def _run_clustering(self, method, all_samples, k, grid, n_meta,
                        embedding='UMAP', resolution=1.0, channels=None,
                        reproducible=True, downsample='Default', custom_n=None,
                        fast_graph=LEIDEN_FAST_GRAPH_DEFAULT, embed_dims=2):
        targets = [n for n in (self._sample_order if all_samples
                               else [self._active_sample])
                   if n in self._samples]
        if not targets:
            self.status_var.set("No sample selected.")
            return
        # The per-sample event cap of the user's own (None: the method's
        # default cap, or every event with All events).
        ok, cap = self._clustering_cap(downsample, custom_n, targets)
        if not ok:
            return
        uncapped = downsample == ALL_EVENTS and not cap
        if not self._confirm_large_clustering(method, targets, cap, uncapped,
                                              bool(fast_graph), k):
            return
        self._clustering_busy = True
        n = len(targets)
        cap_note = _cap_note(method, cap, uncapped)
        emb_method, emb_prefix = self._EMBEDDINGS.get(embedding, (None, None))
        from .channel_roles import check_embed_components
        # Only UMAP takes a component count; the others stay 2-D.
        emb_kw = ({'n_components': check_embed_components(embed_dims)}
                  if emb_prefix == 'UMAP' else {})
        settings = _clustering_settings(method, k, grid, n_meta, resolution,
                                        reproducible, fast_graph, emb_kw)
        run = {'method': method, 'k': k, 'grid': grid, 'n_meta': n_meta,
               'resolution': resolution, 'channels': channels,
               'reproducible': reproducible, 'fast_graph': fast_graph,
               'cap': cap, 'uncapped': uncapped, 'embedding': embedding,
               'emb_method': emb_method, 'emb_prefix': emb_prefix,
               'emb_kw': emb_kw, 'settings': settings}
        # The samples the run reached, for a run that fails part way: only
        # these may have had columns written (_clustering_error).
        started: list = []
        cancel = self._show_clustering_cancel()

        def work():
            return self._clustering_work(run, targets, started, cancel)

        self.run_async(
            work,
            on_done=lambda res: self._clustering_done(
                run, targets, *res),
            on_error=lambda exc: self._clustering_error(exc, started),
            busy_msg=f"{method} on {n} sample(s){cap_note}…")

    def _clustering_cap(self, downsample, custom_n, targets):
        """``(ok, cap)`` for the Events / sample choice: the per-sample
        cap of the user's own, None for Default and All events. ``ok`` is
        False (with the reason in the status bar) for a refused Custom
        number."""
        if downsample == 'Smallest in group':
            szs = [len(self._samples[t].data) for t in targets
                   if getattr(self._samples[t], 'data', None) is not None]
            return True, (min(szs) if szs else None)
        if downsample == 'Custom…':
            # The dialog passes the whole number it checked (read_int, at
            # least MIN_CUSTOM_EVENTS); anything else is refused here, by
            # the same floor, rather than read as "no cap".
            if (not isinstance(custom_n, int) or isinstance(custom_n, bool)
                    or custom_n < MIN_CUSTOM_EVENTS):
                self.status_var.set(
                    f"Events / sample (Custom…) must be a whole number of at "
                    f"least {MIN_CUSTOM_EVENTS} (got '{custom_n}'): "
                    "clustering not started.")
                return False, None
            return True, custom_n
        return True, None

    def _confirm_large_clustering(self, method, targets, cap, uncapped,
                                  fast_graph=LEIDEN_FAST_GRAPH_DEFAULT, k=30):
        """Ask before a graph clustering of more events in one sample than
        the method's default cap (All events, or a Custom or Smallest in
        group number above it), or of more than free memory holds, with
        the memory it needs. True to go on."""
        from .cluster_caps import (
            available_memory_gb,
            default_cap,
            large_run_warning,
        )
        sizes = [len(self._samples[t].data) for t in targets
                 if getattr(self._samples[t], 'data', None) is not None]
        biggest = max(sizes, default=0)
        limit = None if uncapped else (cap or default_cap(method))
        n_events = biggest if limit is None else min(biggest, limit)
        text = large_run_warning(method, n_events, available_memory_gb(),
                                 fast_graph, k)
        if text is None:
            return True
        from tkinter import messagebox
        if messagebox.askyesno("Large clustering run", text, icon='warning',
                               default='no', parent=self):
            return True
        self.status_var.set("Clustering not started. Fewer events per "
                            "sample need less memory.")
        return False

    def _show_clustering_cancel(self):
        """A new cancel flag for a clustering run, and the status bar's
        Cancel button shown for it (when the window has one)."""
        import threading
        cancel = self._clustering_cancel = threading.Event()
        btn = self.__dict__.get('_cluster_cancel_btn')
        if btn is not None:
            try:
                btn.configure(state='normal')
                btn.grid()
            except tk.TclError:
                pass
        return cancel

    def _cancel_clustering(self):
        """The status bar's Cancel: stop the running clustering at its
        next step (Tk callback)."""
        cancel = self.__dict__.get('_clustering_cancel')
        if cancel is None or not getattr(self, '_clustering_busy', False):
            return
        cancel.set()
        btn = self.__dict__.get('_cluster_cancel_btn')
        if btn is not None:
            try:
                btn.configure(state='disabled')
            except tk.TclError:
                pass
        self.status_var.set("Canceling clustering: it stops after the "
                            "step now running…")

    def _clustering_work(self, run, targets, started, cancel):
        """The clustering run itself, off the Tk thread: ``(outcome,
        embedded, canceled)``. Cancel is checked before each sample,
        inside PhenoGraph (before its clustering and before it writes)
        and before each embedding; a sample whose labels were written
        before Cancel keeps them."""
        from .editor_compute import label_run_outcome
        method, emb_prefix = run['method'], run['emb_prefix']
        col = self._cluster_label_column(method)
        emb_col = f'{emb_prefix}1' if emb_prefix else None
        outcome, embedded, n = {}, [], len(targets)
        for i, name in enumerate(targets, 1):
            s = self._samples.get(name)
            if s is None:
                continue
            if cancel.is_set():
                return outcome, embedded, True
            started.append(name)
            self._busy(f"{method}: clustering {name} ({i}/{n})…")
            before = (s.data[col].to_numpy() if col in s.data.columns
                      else None)
            try:
                self._cluster_one(s, run, cancel)
            except ClusteringCanceled:
                return outcome, embedded, True
            outcome[name] = label_run_outcome(before, s.data, col)
            # A skipped sample is reported as 'no labels' and is neither
            # imported nor audited, so it gets no embedding either.
            if not run['emb_method'] or outcome[name] == 'skipped':
                continue
            if cancel.is_set():
                return outcome, embedded, True
            self._busy(
                f"{run['embedding']} embedding on {name} ({i}/{n})… "
                "first run compiles, ~30-50s — still working")
            emb_before = (s.data[emb_col].to_numpy()
                          if emb_col in s.data.columns else None)
            # On the channels just clustered, not every fluor channel
            # (None: the same default both use).
            getattr(s, run['emb_method'])(
                channels=run['channels'],
                **({'sample_n': run['cap']} if run['cap'] else {}),
                **run['emb_kw'])
            # As for the labels: a column this call did not write is an
            # earlier run's (an uninstalled backend writes nothing).
            if label_run_outcome(emb_before, s.data, emb_col) != 'skipped':
                embedded.append(name)
        return outcome, embedded, False

    def _cluster_one(self, s, run, cancel):
        """Cluster sample ``s`` with the run's method and settings."""
        method, channels = run['method'], run['channels']
        max_events = method_max_events(method, run['cap'], run['uncapped'],
                                       len(s.data))
        if method == 'phenograph':
            def checkpoint():
                if cancel.is_set():
                    raise ClusteringCanceled()
            s.cluster(channels=channels, k=run['k'], max_events=max_events,
                      reproducible=run['reproducible'],
                      resolution=run['resolution'], checkpoint=checkpoint)
            # None when PhenoGraph's Louvain ran (no resolution).
            run['settings']['resolution'] = (
                getattr(s, 'cluster_settings', None) or {}).get('resolution')
        elif method == 'leiden':
            s.run_leiden(channels=channels, n_neighbors=run['k'],
                         resolution=run['resolution'], max_events=max_events,
                         fast_graph=bool(run['fast_graph']))
        else:
            s.run_flowsom(channels=channels, grid=(run['grid'], run['grid']),
                          n_metaclusters=run['n_meta'],
                          max_events=max_events)

    def _clustering_done(self, run, targets, outcome, embedded, canceled):
        """A run finished, or stopped at Cancel: import what it wrote."""
        if not canceled:
            self._finish_clustering(
                run['method'], run['emb_prefix'], targets, outcome, embedded,
                channels=run['channels'], settings=run['settings'])
            return
        done = [t for t in targets if t in outcome]
        if not done:
            self._end_clustering_busy()
            self.status_var.set("Clustering canceled. No sample was "
                                "changed.")
            return
        self._finish_clustering(
            run['method'], run['emb_prefix'], done, outcome, embedded,
            channels=run['channels'], settings=run['settings'])
        self.status_var.set(
            f"Clustering canceled after {len(done)} of {len(targets)} "
            f"sample(s); those keep their clusters. "
            f"{self.status_var.get()}")

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
        # One channel per marker, as the Cluster dialog and run_umap embed.
        from .channel_roles import default_cluster_channels
        fluor = getattr(s, 'fluor_channels', None)
        chans = [c for c in (default_cluster_channels(fluor) if fluor
                             else df.columns)
                 if c in df.columns and df[c].dtype.kind in 'fiu']
        if len(chans) < 2:
            ui_errors.showinfo("Embedding comparison",
                                "Need at least 2 numeric channels.", parent=self)
            return
        # The installed methods, when already known; None while the import
        # check (seconds, the first time) has yet to run. The dialog then
        # opens at once and runs that check in the background.
        from .dr_compare import checked_methods
        have = checked_methods()
        if have is not None and not have:
            ui_errors.showinfo("Embedding comparison",
                                "No embedding backends are installed.\n\n"
                                "pip install \"openflo[embed]\" adds them.",
                                parent=self)
            return
        # Pass the frame + channels (NOT a materialised array) so opening the
        # dialog is instant on a million-row sample — the heavy to_numpy() runs
        # later, in the background worker.
        from .ui_embedding import EmbeddingDialog
        EmbeddingDialog(self, name, len(df), have, df, chans)

    def _open_view3d(self):
        """Analyze > 3D plot…: three channels or a 3D embedding, turnable."""
        from .ui_view3d import open_view3d
        return open_view3d(self)

    def _open_view3d_on(self, name, axes, color_column=None):
        """The 3D plot showing sample ``name`` on ``axes`` (three columns),
        colored by the label column ``color_column`` when given."""
        win = self._open_view3d()
        if win is not None:
            win.show(name, axes, color_column=color_column)
        return win

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
        from .label_alignment import common_fluor_warning
        return common_fluor_warning(samples)

    def _open_stats_window(self):
        from .ui_statistics import StatisticsWindow
        if not self._samples:
            self.status_var.set("Load a sample first to compute statistics.")
            return None
        return self._show_or_raise('stats', lambda: StatisticsWindow(self))

    def _open_frequency_window(self):
        """Statistics, on its Frequencies by group tab: the frequencies with
        the samples split into groups, plotted and tested."""
        if not self._samples:
            self.status_var.set("Load samples first to compare frequencies.")
            return None
        win = self._open_stats_window()
        if win is not None:
            win.show_compare()
        return win

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
        ``_cluster_labels`` and the 'cluster' gates when ``label_col`` is the
        PhenoGraph 'cluster' column, else onto that column's category gates,
        then refresh the tree."""
        # One undo step, so Ctrl+Z undoes the naming, not the step before.
        self._checkpoint()
        # _cluster_labels and kind='cluster' gates are keyed by PhenoGraph
        # cluster id only. A Leiden or FlowSOM id is a different partition of
        # the same cells, so writing its names there would relabel unrelated
        # PhenoGraph clusters (Leiden 0 named 'CD4 T' renamed PhenoGraph
        # cluster 0, which held the double-negative cells).
        if label_col == 'cluster':
            store = self._cluster_labels.setdefault(sample, {})
            for cid, nm in names.items():
                store[cid] = nm
        gates = self._sample_gates.get(sample, {})
        for g in gates.values():
            if (label_col == 'cluster' and g.get('kind') == 'cluster'
                    and g.get('cluster_id') in names):
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

    def _open_saturation_report(self):
        """Per sample, the percent of events at the top of each channel."""
        from .ui_saturation import SaturationReportWindow
        if not self._samples:
            self.status_var.set("Load samples to check them for saturation.")
            return
        self._show_or_raise('saturation',
                            lambda: SaturationReportWindow(self))

    def _open_kinetics_window(self):
        """A channel over acquisition time per sample (calcium flux)."""
        from .ui_kinetics import KineticsWindow
        if not self._samples:
            self.status_var.set("Load samples to follow a channel over time.")
            return
        self._show_or_raise('kinetics', lambda: KineticsWindow(self))

    def _open_debarcode_window(self):
        """Split a pooled, barcoded sample back into its samples."""
        from .ui_debarcode import DebarcodeWindow
        if not self._samples:
            self.status_var.set("Load the pooled sample to debarcode it.")
            return
        self._show_or_raise('debarcode', lambda: DebarcodeWindow(self))

    def _open_derived_channels(self):
        """Define channels computed from others (ratios, FRET)."""
        from .ui_derived import DerivedChannelsWindow
        if not self._samples:
            self.status_var.set("Load samples to add a derived channel.")
            return
        self._show_or_raise('derived', lambda: DerivedChannelsWindow(self))

    def _open_sample_qc_window(self):
        from .ui_sample_qc import SampleQCWindow
        if len(self._selected_samples()) < 2:
            self.status_var.set(
                "Enable ≥2 samples (☑) to compare them.")
            return
        self._show_or_raise('sample_qc', lambda: SampleQCWindow(self))
