"""Embedding (UMAP/t-SNE/PHATE/...) run dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .dialog_inputs import read_int
from .dr_compare import METHOD_NAMES
from .ui_tips import auto_tip


class EmbeddingDialog(tk.Toplevel):
    """Set up an embedding comparison: pick methods (installed ones only) and
    the cell count, then run. Replaces a bare yes/no confirm so the user can
    choose backends and the subsample size instead of a hard-coded 4000."""

    _ALL = ('umap', 'tsne', 'phate', 'trimap', 'pacmap')
    _TIPS = {
        'umap': "UMAP: keeps local neighbourhoods and much of the global "
                "layout. A good first choice.",
        'tsne': "t-SNE: separates local clusters sharply; distances between "
                "clusters mean little.",
        'phate': "PHATE: follows continuous trajectories, such as "
                 "differentiation.",
        'trimap': "TriMap: keeps the global arrangement of clusters better "
                  "than t-SNE.",
        'pacmap': "PaCMAP: balances local and global structure, and is "
                  "fast.",
    }

    def __init__(self, editor, name, n, have, df, chans):
        super().__init__(editor)
        self.title("Compare embeddings")
        self._editor = editor
        self._name, self._df, self._chans = name, df, chans
        self._have = set(have)
        # Start compiling UMAP's numba kernels NOW, while the user is still
        # picking methods and a cell count. That compile costs ~18s and is
        # cached in no process; paying it during the seconds someone spends in
        # this dialog means Run starts embedding immediately instead of
        # sitting still. UMAP is ticked by default, so the bet is usually
        # right — and when it is wrong the kernels stay compiled for the rest
        # of the session anyway, so nothing is actually wasted. Cannot affect
        # results: see openflo/warmup.py.
        from . import warmup
        warmup.start({m: True for m in self._have})

        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=14, pady=12)
        # The button row is packed first, so a short window squeezes the
        # list rather than hiding Run and Cancel; the window takes the size
        # its content asks for (no fixed geometry), which fit_window keeps
        # on screen.
        bar = ttk.Frame(frm)
        bar.pack(side='bottom', fill='x', pady=(8, 0))
        ttk.Label(frm, justify='left',
                  text=(f"Sample: {name}\n{n:,} events. Embeddings are "
                        "compute-heavy and run in the background.")).pack(
            anchor='w', pady=(0, 8))
        ttk.Label(frm, text="Methods:",
                  font='OpenFloHeading').pack(anchor='w')
        self._vars = {}
        for m in self._ALL:
            installed = m in self._have
            v = tk.BooleanVar(value=installed and m in ('umap', 'tsne'))
            cb = ttk.Checkbutton(
                frm, variable=v,
                text=(METHOD_NAMES.get(m, m) if installed
                      else f"{METHOD_NAMES.get(m, m)}  (not installed: pip "
                           "install \"openflo[embed]\")"))
            cb.pack(anchor='w')
            auto_tip(cb, self._TIPS.get(m, '') if installed else
                     f"{m.upper()} is an optional extra. Install it with "
                     "pip install \"openflo[embed]\" and reopen this dialog.")
            if not installed:
                cb.state(['disabled'])
            self._vars[m] = v
        crow = ttk.Frame(frm)
        crow.pack(anchor='w', pady=(10, 4))
        ttk.Label(crow, text="Cells to embed (subsample):").pack(side='left')
        self._cap = tk.StringVar(value=str(min(5000, n)))
        _cap = ttk.Spinbox(crow, from_=200, to=200000, increment=1000,
                           width=10, textvariable=self._cap)
        _cap.pack(side='left', padx=6)
        auto_tip(_cap,
                 "Maximum events fed to the embedding; more are randomly "
                 "subsampled. Embeddings scale badly with event count, so "
                 "this is the main control on how long the run takes. The "
                 "sample itself is not modified.")
        # Main action, then Cancel, at the right (tool_window).
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(side='right')
        _rb = ttk.Button(bar, text="Run", default='active', command=self._run)
        _rb.pack(side='right', padx=6)
        auto_tip(_rb,
                 "Compute the ticked embeddings and add their coordinates as "
                 "new columns, so they can be plotted and gated like any "
                 "other channel.")

    def _run(self):
        methods = [m for m, v in self._vars.items()
                   if v.get() and m in self._have]
        if not methods:
            self._editor.status_var.set("Pick at least one installed method.")
            return
        cap = read_int(self._cap, "Cells to embed", lo=50,
                          hi=200000, parent=self)
        if cap is None:
            return
        self.destroy()
        self._editor._start_embedding(self._name, self._df, self._chans,
                                      tuple(methods), cap)
