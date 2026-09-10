"""Spectral unmixing setup dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .ui_tips import auto_tip


class SpectralUnmixDialog(tk.Toplevel):
    """Assign loaded samples to roles for spectral unmixing — single-stain
    (→ fluorophore), unstained, or ignore — and pick the detector channels.
    Calls ``on_apply(singles {name: fluor}, unstained_name|None, detectors,
    nonneg)`` on Build & Apply."""

    def __init__(self, parent, sample_names, detectors, on_apply):
        super().__init__(parent)
        self.title("Spectral unmixing")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(
            body, justify='left',
            text=("Designate the single-stain control samples (→ fluorophore) "
                  "and one unstained\ncontrol. Every other loaded sample is "
                  "unmixed into per-fluor 'U:' channels.")).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))
        ttk.Label(body, text="Sample", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=0, sticky='w')
        ttk.Label(body, text="Role", font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=1, sticky='w', padx=8)
        ttk.Label(body, text="Fluorophore",
                  font=('TkDefaultFont', 9, 'bold')).grid(
            row=1, column=2, sticky='w')

        self.rows = []
        roles = ['Ignore', 'Single-stain', 'Unstained']
        for i, nm in enumerate(sample_names):
            ttk.Label(body, text=(nm[:34])).grid(row=2 + i, column=0, sticky='w')
            rv = tk.StringVar(value='Ignore')
            _rc = ttk.Combobox(body, textvariable=rv, values=roles,
                               state='readonly', width=12)
            _rc.grid(row=2 + i, column=1, padx=8, pady=1)
            auto_tip(_rc,
                     "What this sample is: an unstained control, a "
                     "single-stain reference for one fluorophore, or a "
                     "full-stain sample to be unmixed. The reference spectra "
                     "come from the single stains.")
            fv = tk.StringVar(value='')
            _fe = ttk.Entry(body, textvariable=fv, width=22)
            _fe.grid(row=2 + i, column=2, sticky='w')
            auto_tip(_fe,
                     "Name of the fluorophore this single stain defines. It "
                     "becomes the name of the unmixed channel.")
            self.rows.append((nm, rv, fv))

        r = 2 + len(sample_names)
        ttk.Label(body, text="Detectors:").grid(
            row=r, column=0, sticky='ne', pady=(8, 0))
        self.det_txt = tk.Text(body, height=3, width=46, wrap='word')
        self.det_txt.insert('1.0', ', '.join(detectors))
        self.det_txt.grid(row=r, column=1, columnspan=2, sticky='w', pady=(8, 0))
        self.nonneg_var = tk.BooleanVar(value=True)
        _nn = ttk.Checkbutton(body, text="Non-negative abundances",
                              variable=self.nonneg_var)
        auto_tip(_nn,
                 "Constrain unmixed abundances to be >= 0. Physically a "
                 "negative amount of dye is meaningless, but the constraint "
                 "also hides genuine over-unmixing, which shows up as "
                 "negative values. Leave off while diagnosing a bad unmix.")
        _nn.grid(
            row=r + 1, column=1, columnspan=2, sticky='w', pady=(6, 0))

        bb = ttk.Frame(body)
        bb.grid(row=r + 2, column=0, columnspan=3, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        _ba = ttk.Button(bb, text="Build & Apply", command=self._apply)
        _ba.pack(side='right', padx=(0, 6))
        auto_tip(_ba,
                 "Build the reference spectra from the single stains, solve "
                 "the unmixing and apply it to the full-stain samples.")
        try:
            self.grab_set()
        except Exception:
            pass

    def _apply(self):
        singles, unstained = {}, None
        for nm, rv, fv in self.rows:
            role = rv.get()
            if role == 'Single-stain':
                singles[nm] = fv.get().strip() or nm
            elif role == 'Unstained':
                unstained = nm
        dets = [d.strip() for d in
                self.det_txt.get('1.0', 'end').replace('\n', ' ').split(',')
                if d.strip()]
        if not singles:
            messagebox.showwarning(
                "Spectral unmixing",
                "Assign at least one single-stain control to a fluorophore.",
                parent=self)
            return
        if len(dets) < 2:
            messagebox.showwarning(
                "Spectral unmixing", "Need at least 2 detector channels.",
                parent=self)
            return
        self.on_apply(singles, unstained, dets, bool(self.nonneg_var.get()))
        self.destroy()
