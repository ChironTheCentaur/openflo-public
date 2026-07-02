"""Multi-panel figure layout/export dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class FigureLayoutDialog(tk.Toplevel):
    """Configure a multi-panel publication figure built from the current
    plot selection. Calls ``on_apply(opts)`` with a dict::

        {layout, ncols, pairs, gates}

    where ``layout`` is one of ``single`` / ``per_sample`` / ``per_pair`` /
    ``grid``. The plot mode, colouring and (for the single/per-sample
    layouts) the channels come from the live plot controls."""

    def __init__(self, parent, n_samples, mode, default_pairs, on_apply):
        super().__init__(parent)
        self.title("Figure layout")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(
            body, justify='left',
            text=(f"{n_samples} sample(s) enabled · mode: {mode}\n"
                  "Build a multi-panel figure from the current plot. "
                  "Channel pairs apply to the\npair / grid layouts "
                  "(e.g. \"CD34/CD11b, CD11b/CD45\"; markers or "
                  "channel names).")).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

        ttk.Label(body, text="Layout:", font=('TkDefaultFont', 9, 'bold')
                  ).grid(row=1, column=0, sticky='w')
        self.layout_var = tk.StringVar(value='per_sample')
        layouts = [
            ('One panel per sample (current channels)', 'per_sample'),
            ('One panel per channel pair (samples overlaid)', 'per_pair'),
            ('Grid: samples × channel pairs', 'grid'),
            ('Single panel (current view)', 'single'),
        ]
        lf = ttk.Frame(body)
        lf.grid(row=2, column=0, columnspan=2, sticky='w', pady=(2, 8))
        for lbl, val in layouts:
            ttk.Radiobutton(lf, text=lbl, value=val,
                            variable=self.layout_var,
                            command=self._sync_enabled).pack(anchor='w')

        ttk.Label(body, text="Channel pairs:").grid(
            row=3, column=0, sticky='nw', pady=(4, 0))
        self.pairs_txt = tk.Text(body, height=3, width=42, wrap='word')
        self.pairs_txt.insert('1.0', default_pairs)
        self.pairs_txt.grid(row=3, column=1, sticky='w', pady=(4, 0))

        ttk.Label(body, text="Columns:").grid(
            row=4, column=0, sticky='w', pady=(8, 0))
        self.ncols_var = tk.StringVar(value='3')
        self.ncols_spin = ttk.Spinbox(body, from_=1, to=12, width=6,
                                      textvariable=self.ncols_var)
        self.ncols_spin.grid(row=4, column=1, sticky='w', pady=(8, 0))

        self.gates_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(body, text="Draw gates on panels",
                        variable=self.gates_var).grid(
            row=5, column=0, columnspan=2, sticky='w', pady=(8, 0))

        bb = ttk.Frame(body)
        bb.grid(row=6, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="Build", command=self._apply).pack(
            side='right', padx=(0, 6))

        self._sync_enabled()
        try:
            self.grab_set()
        except Exception:
            pass

    def _sync_enabled(self):
        layout = self.layout_var.get()
        needs_pairs = layout in ('per_pair', 'grid')
        self.pairs_txt.configure(
            state=('normal' if needs_pairs else 'disabled'))
        # Grid derives its column count from the number of pairs.
        self.ncols_spin.configure(
            state=('disabled' if layout == 'grid' else 'normal'))

    def _apply(self):
        try:
            ncols = max(1, int(self.ncols_var.get()))
        except (TypeError, ValueError):
            ncols = 3
        opts = {
            'layout': self.layout_var.get(),
            'ncols': ncols,
            'pairs': self.pairs_txt.get('1.0', 'end').strip(),
            'gates': bool(self.gates_var.get()),
        }
        self.on_apply(opts)
        self.destroy()
