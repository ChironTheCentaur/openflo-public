"""Embedding (UMAP/t-SNE/PHATE/...) run dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class EmbeddingDialog(tk.Toplevel):
    """Set up an embedding comparison: pick methods (installed ones only) and
    the cell count, then run. Replaces a bare yes/no confirm so the user can
    choose backends and the subsample size instead of a hard-coded 4000."""

    _ALL = ('umap', 'tsne', 'phate', 'trimap', 'pacmap')

    def __init__(self, editor, name, n, have, df, chans):
        super().__init__(editor)
        self.title("Compare embeddings")
        self.geometry("420x360")
        self._editor = editor
        self._name, self._df, self._chans = name, df, chans
        self._have = set(have)
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=14, pady=12)
        ttk.Label(frm, justify='left',
                  text=(f"Sample: {name}\n{n:,} events. Embeddings are "
                        "compute-heavy and run in the background.")).pack(
            anchor='w', pady=(0, 8))
        ttk.Label(frm, text="Methods:",
                  font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')
        self._vars = {}
        for m in self._ALL:
            installed = m in self._have
            v = tk.BooleanVar(value=installed and m in ('umap', 'tsne'))
            cb = ttk.Checkbutton(
                frm, variable=v,
                text=(m.upper() if installed else f"{m.upper()}  (not "
                      "installed — pip install \"openflo[embed]\")"))
            cb.pack(anchor='w')
            if not installed:
                cb.state(['disabled'])
            self._vars[m] = v
        crow = ttk.Frame(frm)
        crow.pack(anchor='w', pady=(10, 4))
        ttk.Label(crow, text="Cells to embed (subsample):").pack(side='left')
        self._cap = tk.StringVar(value=str(min(5000, n)))
        ttk.Spinbox(crow, from_=200, to=200000, increment=1000, width=10,
                    textvariable=self._cap).pack(side='left', padx=6)
        bar = ttk.Frame(frm)
        bar.pack(side='bottom', fill='x', pady=(8, 0))
        ttk.Button(bar, text="Run", command=self._run).pack(side='right',
                                                            padx=6)
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side='right')

    def _run(self):
        methods = [m for m, v in self._vars.items()
                   if v.get() and m in self._have]
        if not methods:
            self._editor.status_var.set("Pick at least one installed method.")
            return
        try:
            cap = max(50, int(float(self._cap.get())))
        except (TypeError, ValueError):
            cap = 5000
        self.destroy()
        self._editor._start_embedding(self._name, self._df, self._chans,
                                      tuple(methods), cap)
