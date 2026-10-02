"""Cell-cycle (DNA-content) phase window.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
)
from .tool_window import figure_panel


class CellCycleWindow(tk.Toplevel):
    """DNA-content histogram with G1/S/G2M boundaries + phase percentages
    for one sample's cell-cycle result."""

    def __init__(self, editor, sample_name):
        super().__init__(editor)
        self.title(f"Cell cycle — {sample_name}")
        self.geometry("720x500")
        self.minsize(480, 320)

        s   = editor._samples[sample_name]
        res = getattr(s, 'cell_cycle_result', None)
        if not res or not res.get('ok'):
            ttk.Label(self, text="No cell-cycle result for this sample.").pack(
                padx=20, pady=20)
            return

        col   = res['channel']
        phase = np.asarray(s.data['cell_cycle'].values)
        vals  = np.asarray(s.data[col].values, dtype=float)
        keep  = (phase != 'NA') & np.isfinite(vals)
        v     = vals[keep]

        # dark=False: themed below after the axes are drawn.
        _frame, fig, canvas = figure_panel(self, figsize=(7, 4), dark=False)
        ax  = fig.add_subplot(111)
        if v.size:
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(v, bins=200, range=(float(lo), float(hi)),
                    color='#999999', alpha=0.65)
        # Phase means (solid) + G1|S and S|G2M boundaries (dashed).
        ax.axvline(res['g1_mean'], color='#4363d8', lw=1.4, label='G1')
        ax.axvline(res['g2_mean'], color='#e6194b', lw=1.4, label='G2/M')
        ax.axvline(res['g1_hi'], color='#3cb44b', ls='--', lw=1)
        ax.axvline(res['g2_lo'], color='#3cb44b', ls='--', lw=1)
        ax.set_xlabel(editor._fmt_channel(col))
        ax.set_ylabel('events')
        ax.set_title(f"Cell cycle — {sample_name}")
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()

        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        canvas.draw()

        summary = (
            f"G1 {res['pct_g1']:.1f}%      "
            f"S {res['pct_s']:.1f}%      "
            f"G2/M {res['pct_g2m']:.1f}%        "
            f"({res['n_cycling']:,} cycling of {res['n_singlet']:,} singlets)")
        ttk.Label(self, text=summary,
                  font=('TkDefaultFont', 10, 'bold')).pack(pady=(4, 4))
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 8))


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS TABLE (FlowJo-style population statistics)
# ══════════════════════════════════════════════════════════════════════════════
