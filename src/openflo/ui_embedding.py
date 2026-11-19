"""Embedding (UMAP/t-SNE/PHATE/...) run dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .dialog_inputs import read_int
from .dr_compare import METHOD_NAMES
from .ui_tips import auto_tip


class EmbeddingDialog(tk.Toplevel):
    """Set up an embedding comparison: pick methods (installed ones only) and
    the cell count, then run. Replaces a bare yes/no confirm so the user can
    choose backends and the subsample size instead of a hard-coded 4000."""

    _ALL = ('umap', 'tsne', 'phate', 'trimap', 'pacmap')
    _TIPS = {
        'umap': "UMAP: keeps local neighborhoods and much of the global "
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
        """``have``: the installed methods, or None while that is unknown.
        Finding out imports umap and scikit-learn in a child process, which
        takes seconds the first time; the window then opens at once with the
        boxes "checking…" and fills them in from a background check."""
        super().__init__(editor)
        self.title("Compare embeddings")
        self._editor = editor
        self._name, self._df, self._chans = name, df, chans
        self._have: set | None = None

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
        self._boxes = {}
        for m in self._ALL:
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(frm, variable=v)
            cb.pack(anchor='w')
            auto_tip(cb, lambda m=m: self._tip_for(m))
            self._vars[m] = v
            self._boxes[m] = cb
        if have is None:
            for m, cb in self._boxes.items():
                cb.configure(text=f"{METHOD_NAMES.get(m, m)}  (checking…)")
                cb.state(['disabled'])
            self._check_methods()
        else:
            self._set_have(have)
        crow = ttk.Frame(frm)
        crow.pack(anchor='w', pady=(10, 4))
        ttk.Label(crow, text="Cells to embed (subsample):").pack(side='left')
        self._cap = tk.StringVar(value=str(min(5000, n)))
        _cap = ttk.Spinbox(crow, from_=50, to=200000, increment=1000,
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
                 "Compute the ticked embeddings on a subsample and show them "
                 "side by side in a figure window, to compare the layouts. "
                 "Nothing is added to the sample: to plot and gate an "
                 "embedding, add it with Cluster… instead.")

    def _tip_for(self, m):
        if self._have is None:
            return ("Checking whether this method is installed; the box "
                    "is ready in a few seconds.")
        if m in self._have:
            return self._TIPS.get(m, '')
        return (f"{m.upper()} is an optional extra. Install it with "
                "pip install \"openflo[embed]\" and reopen this dialog.")

    def _set_have(self, have):
        """Fill in the method boxes: installed ones enabled, UMAP and t-SNE
        ticked; the rest disabled with the install hint. Also starts
        compiling UMAP's kernels when UMAP is there."""
        self._have = set(have)
        for m, cb in self._boxes.items():
            installed = m in self._have
            self._vars[m].set(installed and m in ('umap', 'tsne'))
            cb.configure(
                text=(METHOD_NAMES.get(m, m) if installed
                      else f"{METHOD_NAMES.get(m, m)}  (not installed: pip "
                           "install \"openflo[embed]\")"))
            cb.state(['!disabled'] if installed else ['disabled'])
        # Start compiling UMAP's numba kernels NOW, while the user is still
        # picking methods and a cell count. That compile costs ~18s and is
        # cached in no process; paying it during the seconds someone spends in
        # this dialog means Run starts embedding immediately instead of
        # sitting still. UMAP is ticked by default, so the bet is usually
        # right — and when it is wrong the kernels stay compiled for the rest
        # of the session anyway, so nothing is actually wasted. Cannot affect
        # results: see openflo/warmup.py.
        from . import warmup
        warmup.start(dict.fromkeys(self._have, True))

    def _check_methods(self):
        """Run the import check off the Tk thread (on it, the first open
        froze the window for 3-17 s) and fill the boxes in when it answers.
        Nothing installed: say so and close, as before."""
        from . import dr_compare

        def done(have):
            if not self.winfo_exists():
                return
            if not have:
                self.destroy()
                messagebox.showinfo("Embedding comparison",
                                    "No embedding backends are installed.\n\n"
                                    "pip install \"openflo[embed]\" adds them.",
                                    parent=self._editor)
                return
            self._set_have(have)

        def failed(exc):
            if self.winfo_exists():
                self.destroy()
            from .errors import report
            report(self._editor, exc)

        run = getattr(self._editor, 'run_async', None)
        if run is None:                 # a bare parent window
            from .async_task import run_async
            run_async(self, dr_compare.available_methods, on_done=done,
                      on_error=failed)
        else:
            run(dr_compare.available_methods, on_done=done, on_error=failed)

    def _run(self):
        if self._have is None:
            self._editor.status_var.set(
                "Still checking which methods are installed…")
            return
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
