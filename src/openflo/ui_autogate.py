"""Automatic gating dialog.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class AutoGateDialog(tk.Toplevel):
    """Choose an automated-gating method for the active sample. Calls
    ``on_apply(opts)`` with a dict whose ``method`` is one of:

      • ``singlet``   — FSC-A/FSC-H ratio-band polygon (+ ``k``, ``area``,
                        ``height``)
      • ``gmm``       — BIC-selected Gaussian-mixture ellipses on the current
                        X/Y plot (+ ``max_components``, ``coverage``,
                        ``min_weight``)
      • ``threshold`` — 1-D valley/Otsu split on the current X channel

    Each proposal is added as an ordinary undoable gate the user reviews."""

    def __init__(self, parent, has_singlet, area, height, cur_x, cur_y,
                 on_apply):
        super().__init__(parent)
        self.title("Auto-gate")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply
        self._area = area
        self._height = height

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(
            body, justify='left',
            text=("Propose gates from the data — each is added as an ordinary,\n"
                  "editable gate you can accept, tweak or delete. Quality is\n"
                  "reported in the status bar.")).grid(
            row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))

        self.method_var = tk.StringVar(
            value='singlet' if has_singlet else 'gmm')

        mf = ttk.Frame(body)
        mf.grid(row=1, column=0, columnspan=2, sticky='w')
        singlet_lbl = ("Singlet gate (FSC-A vs FSC-H ratio band)"
                       if has_singlet
                       else "Singlet gate — needs an FSC-A + FSC-H pair")
        self._singlet_rb = ttk.Radiobutton(
            mf, text=singlet_lbl, value='singlet',
            variable=self.method_var, command=self._sync)
        if not has_singlet:
            self._singlet_rb.configure(state='disabled')
        self._singlet_rb.pack(anchor='w')
        if area and height:
            ttk.Label(mf, text=f"    {area}  vs  {height}",
                      foreground='#666').pack(anchor='w')

        ttk.Radiobutton(
            mf, text=f"Find populations (GMM ellipses) on  {cur_x or '?'} × "
                     f"{cur_y or '?'}",
            value='gmm', variable=self.method_var,
            command=self._sync).pack(anchor='w', pady=(4, 0))
        ttk.Radiobutton(
            mf, text=f"1-D threshold on  {cur_x or '?'}  (valley / Otsu)",
            value='threshold', variable=self.method_var,
            command=self._sync).pack(anchor='w', pady=(4, 0))

        # ── Singlet params ──
        self.singlet_frame = ttk.LabelFrame(body, text="Singlet band",
                                            padding=8)
        self.singlet_frame.grid(row=2, column=0, columnspan=2, sticky='ew',
                                pady=(10, 0))
        ttk.Label(self.singlet_frame, text="Band width (× robust σ):").grid(
            row=0, column=0, sticky='w')
        self.k_var = tk.StringVar(value='3.0')
        k_sp = ttk.Spinbox(self.singlet_frame, from_=1.0, to=6.0,
                           increment=0.5, width=6, textvariable=self.k_var)
        k_sp.grid(row=0, column=1, sticky='w', padx=(6, 0))
        self._singlet_inputs = [k_sp]

        # ── GMM params ──
        self.gmm_frame = ttk.LabelFrame(body, text="GMM ellipses", padding=8)
        self.gmm_frame.grid(row=3, column=0, columnspan=2, sticky='ew',
                            pady=(8, 0))
        ttk.Label(self.gmm_frame, text="Max populations:").grid(
            row=0, column=0, sticky='w')
        self.kmax_var = tk.StringVar(value='6')
        kmax_sp = ttk.Spinbox(self.gmm_frame, from_=1, to=12, width=6,
                              textvariable=self.kmax_var)
        kmax_sp.grid(row=0, column=1, sticky='w', padx=(6, 12))
        ttk.Label(self.gmm_frame, text="Coverage %:").grid(
            row=0, column=2, sticky='w')
        self.cov_var = tk.StringVar(value='90')
        cov_sp = ttk.Spinbox(self.gmm_frame, from_=50, to=99, width=6,
                             textvariable=self.cov_var)
        cov_sp.grid(row=0, column=3, sticky='w', padx=(6, 0))
        ttk.Label(self.gmm_frame, text="Min population %:").grid(
            row=1, column=0, sticky='w', pady=(6, 0))
        self.minw_var = tk.StringVar(value='2')
        minw_sp = ttk.Spinbox(self.gmm_frame, from_=0, to=25, width=6,
                              textvariable=self.minw_var)
        minw_sp.grid(row=1, column=1, sticky='w', padx=(6, 0), pady=(6, 0))
        self._gmm_inputs = [kmax_sp, cov_sp, minw_sp]

        bb = ttk.Frame(body)
        bb.grid(row=4, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="Propose", command=self._apply).pack(
            side='right', padx=(0, 6))

        self._sync()
        try:
            self.grab_set()
        except Exception:
            pass

    def _sync(self):
        m = self.method_var.get()
        for sp in self._singlet_inputs:
            sp.configure(state=('normal' if m == 'singlet' else 'disabled'))
        for sp in self._gmm_inputs:
            sp.configure(state=('normal' if m == 'gmm' else 'disabled'))

    def _apply(self):
        method = self.method_var.get()
        opts: dict = {'method': method}
        if method == 'singlet':
            opts['area'] = self._area
            opts['height'] = self._height
            try:
                opts['k'] = float(self.k_var.get())
            except ValueError:
                opts['k'] = 3.0
        elif method == 'gmm':
            try:
                opts['max_components'] = max(1, int(self.kmax_var.get()))
            except ValueError:
                opts['max_components'] = 6
            try:
                opts['coverage'] = min(0.999, max(0.5,
                                   float(self.cov_var.get()) / 100.0))
            except ValueError:
                opts['coverage'] = 0.90
            try:
                opts['min_weight'] = max(0.0,
                                   float(self.minw_var.get()) / 100.0)
            except ValueError:
                opts['min_weight'] = 0.02
        self.on_apply(opts)
        self.destroy()
