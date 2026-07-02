"""Absolute-count (counting-bead) dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class AbsCountsDialog(tk.Toplevel):
    """Counting-bead absolute counts: cells/µL from cell vs bead event counts."""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Absolute counts")
        self.geometry("440x300")
        self._editor = editor
        self._cell = tk.StringVar()
        self._bead = tk.StringVar()
        self._conc = tk.StringVar()
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=12, pady=10)
        ttk.Label(frm, justify='left',
                  text="Counting-bead absolute count:\n"
                       "cells/µL = (cell events / bead events) × bead "
                       "concentration (beads/µL).").pack(anchor='w', pady=(0, 8))
        for lbl, var in (("Cell events:", self._cell),
                         ("Bead events:", self._bead),
                         ("Bead concentration (beads/µL):", self._conc)):
            row = ttk.Frame(frm)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=lbl, width=28).pack(side='left')
            ttk.Entry(row, textvariable=var, width=14).pack(side='left')
        self._result = ttk.Label(frm, text="", font=('TkDefaultFont', 11, 'bold'))
        self._result.pack(anchor='w', pady=10)
        bar = ttk.Frame(frm)
        bar.pack(fill='x', side='bottom')
        ttk.Button(bar, text="Compute", command=self._compute).pack(side='left')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')

    def _compute(self):
        from .calibration import absolute_count_per_uL
        try:
            cells = float(self._cell.get())
            beads = float(self._bead.get())
            conc = float(self._conc.get())
            val = absolute_count_per_uL(cells, beads, conc)
            self._result.configure(text=f"= {val:,.1f} cells/µL")
        except Exception as exc:
            self._result.configure(text=f"— {exc}")
