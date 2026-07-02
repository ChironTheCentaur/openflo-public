"""Cluster/population annotation (MEM-style marker enrichment).

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np


class PopulationAnnotationWindow(tk.Toplevel):
    """Annotate clustered populations by phenotype.

    Computes MEM (Marker Enrichment Modeling) labels for each cluster of the
    active sample's chosen label column, and — given a reference
    ``name: CD3+ CD4+ CD8-`` table — assigns a best-matching cell-type name,
    writing it back onto the populations (and the cluster-label store). Exports
    the MEM table."""

    _DEFAULT_TABLE = (
        "# name: marker+ marker-  (one cell type per line)\n"
        "CD4 T: CD3+ CD4+ CD8-\n"
        "CD8 T: CD3+ CD8+ CD4-\n"
        "B cell: CD3- CD19+\n"
        "NK cell: CD3- CD56+\n"
        "Monocyte: CD14+ CD3-\n")

    def __init__(self, editor, sample):
        super().__init__(editor)
        self.editor = editor
        self.sample = sample
        self.title(f"Annotate populations — {sample}")
        self.geometry("860x620")
        self._mem = None
        self._label_col = None

        s = editor._samples[sample]
        cols = [c for c in ('leiden', 'cluster', 'flowsom_meta')
                if c in s.data.columns]
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Cluster column:").pack(side='left')
        self.col_var = tk.StringVar(value=cols[0] if cols else '')
        ttk.Combobox(ctl, textvariable=self.col_var, width=14, state='readonly',
                     values=cols).pack(side='left', padx=(2, 8))
        ttk.Label(ctl, text="MEM threshold:").pack(side='left')
        self.thr_var = tk.StringVar(value='2')
        ttk.Spinbox(ctl, from_=0, to=10, width=5,
                    textvariable=self.thr_var).pack(side='left', padx=(2, 8))
        ttk.Button(ctl, text="Compute MEM", command=self._compute).pack(
            side='left')
        ttk.Button(ctl, text="Export MEM CSV…", command=self._export).pack(
            side='right')

        cols2 = ('pop', 'n', 'mem', 'name')
        tv = ttk.Treeview(self, columns=cols2, show='headings', height=12)
        for c, w in zip(cols2, (70, 80, 430, 130), strict=True):
            tv.heading(c, text={'pop': 'Cluster', 'n': 'Events',
                                'mem': 'MEM label', 'name': 'Assigned'}[c])
            tv.column(c, width=w, anchor='w', stretch=(c == 'mem'))
        tv.pack(fill='both', expand=True, padx=6)
        self._tv = tv

        ref = ttk.LabelFrame(self, text="Reference cell-type table "
                             "(name: CD3+ CD4+ CD8-)", padding=6)
        ref.pack(fill='x', padx=6, pady=(4, 6))
        self.ref_txt = tk.Text(ref, height=6, wrap='none')
        self.ref_txt.insert('1.0', self._DEFAULT_TABLE)
        self.ref_txt.pack(fill='x', side='top')
        ttk.Button(ref, text="Assign names → populations",
                   command=self._apply).pack(side='left', pady=(4, 0))
        self.status = ttk.Label(ref, text="", foreground='#555')
        self.status.pack(side='left', padx=(10, 0), pady=(4, 0))

        if cols:
            self._compute()

    def _marker_cols(self):
        s = self.editor._samples[self.sample]
        return [c for c in getattr(s, 'fluor_channels', [])
                if c in s.data.columns]

    def _label_of(self, det):
        s = self.editor._samples[self.sample]
        return (getattr(s, 'channel_labels', {}) or {}).get(
            det, self.editor._channel_labels.get(det, det))

    def _compute(self):
        from .annotate import mem_label, mem_scores
        col = self.col_var.get()
        markers = self._marker_cols()
        if not col or not markers:
            return
        s = self.editor._samples[self.sample]
        labels = s.data[col].to_numpy()
        valid = labels >= 0
        mem = mem_scores(s.data.loc[valid, markers], labels[valid], markers)
        # Relabel detector columns to antibody markers for readability + the
        # reference table (which is written in CD names).
        mem = mem.rename(columns={d: self._label_of(d) for d in markers})
        self._mem = mem
        self._label_col = col
        try:
            thr = float(self.thr_var.get())
        except ValueError:
            thr = 2.0
        uniq, cnts = np.unique(labels[valid], return_counts=True)
        counts = {int(u): int(c) for u, c in zip(uniq, cnts, strict=True)}
        self._tv.delete(*self._tv.get_children())
        for pop, row in mem.iterrows():
            pid = int(str(pop))
            self._tv.insert('', 'end', iid=str(pid),
                            values=(pid, counts.get(pid, 0),
                                    mem_label(row, threshold=thr), ''))
        self.status.configure(text=f"MEM computed for {len(mem)} clusters.")

    def _apply(self):
        from .annotate import (
            annotate_by_reference,
            parse_signature_table,
            population_states,
        )
        if self._mem is None:
            return
        table = parse_signature_table(self.ref_txt.get('1.0', 'end'))
        if not table:
            self.status.configure(text="No valid reference rows parsed.")
            return
        try:
            thr = max(2.0, float(self.thr_var.get()))
        except ValueError:
            thr = 3.0
        states = population_states(self._mem, threshold=thr)
        ann = annotate_by_reference(states, table)
        names = {int(str(p)): ann[p]['name'] for p in ann
                 if ann[p]['name'] != 'unknown'}
        for iid in self._tv.get_children():
            pop = int(iid)
            self._tv.set(iid, 'name', names.get(pop, 'unknown'))
        if names:
            self.editor._apply_population_names(self.sample, self._label_col,
                                                names)
            self.editor._audit('annotate', sample=self.sample,
                                column=self._label_col, n_named=len(names))
        self.status.configure(
            text=f"Named {len(names)} of {len(ann)} clusters.")

    def _export(self):
        if self._mem is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv', initialfile='mem_scores.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            self._mem.to_csv(path)
            self.editor._audit('annotate.export', path=path)
            messagebox.showinfo("Annotate", f"Exported:\n{path}", parent=self)
