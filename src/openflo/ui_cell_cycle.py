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

        from .pipeline import linear_values
        col   = res['channel']
        phase = np.asarray(s.data['cell_cycle'].values)
        # The model's means and boundaries are on the linear DNA scale
        # (FlowSample.cell_cycle), so the histogram under them must be too.
        vals  = linear_values(s, col)
        keep  = (phase != 'NA') & np.isfinite(vals)
        v     = vals[keep]

        # dark=False: themed below after the axes are drawn.
        _frame, fig, canvas = figure_panel(self, figsize=(7, 4), dark=False)
        self.fig = fig
        ax  = fig.add_subplot(111)
        if v.size:
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(v, bins=200, range=(float(lo), float(hi)),
                    color='#999999', alpha=0.65)
        # Phase means (solid) + G1|S and S|G2M boundaries (dashed).
        ax.axvline(res['g1_mean'], color='#4363d8', lw=1.4, label='G1')
        ax.axvline(res['g2_mean'], color='#e6194b', lw=1.4, label='G2/M')
        if res.get('djf_params') and v.size:
            # The fitted Dean-Jett-Fox components, in events per bin.
            from .cellcycle_djf import _components
            x = np.linspace(float(lo), float(hi), 400)
            width = (float(hi) - float(lo)) / 200
            g1c, sc, g2c, _d = _components(np.asarray(res['djf_params']), x,
                                           width)
            ax.plot(x, g1c, color='#4363d8', lw=1, ls=':')
            ax.plot(x, sc, color='#3cb44b', lw=1.2, label='S (fit)')
            ax.plot(x, g2c, color='#e6194b', lw=1, ls=':')
            ax.plot(x, g1c + sc + g2c, color='black', lw=1, label='Fit')
        else:
            ax.axvline(res['g1_hi'], color='#3cb44b', ls='--', lw=1)
            ax.axvline(res['g2_lo'], color='#3cb44b', ls='--', lw=1)
        ax.set_xlabel(f"{editor._fmt_channel(col)} (linear)")
        ax.set_ylabel('events')
        ax.set_title(f"Cell cycle — {sample_name}")
        ax.legend(fontsize=8, loc='best')
        fig.tight_layout()

        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        canvas.draw()

        method = res.get('method', 'boundaries')
        summary = (
            f"{method[:1].upper()}{method[1:]}:  "
            f"G1 {res['pct_g1']:.1f}%      "
            f"S {res['pct_s']:.1f}%      "
            f"G2/M {res['pct_g2m']:.1f}%        "
            f"({res['n_cycling']:,} cycling of {res['n_singlet']:,} singlets)")
        ttk.Label(self, text=summary,
                  font='OpenFloHeading').pack(pady=(4, 4))
        b = res.get('boundaries')
        if res.get('djf_params') and b:
            ttk.Label(self, style='Muted.TLabel', text=(
                f"Fit: G2/G1 {res['g2_g1_ratio']:.2f}, G1 CV "
                f"{res['g1_cv']:.1f}%, reduced chi-square {res['rcs']:.2f}. "
                f"Boundaries gave G1 {b['pct_g1']:.1f}% / S "
                f"{b['pct_s']:.1f}% / G2/M {b['pct_g2m']:.1f}%.")).pack()
        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 8))


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS TABLE (FlowJo-style population statistics)
# ══════════════════════════════════════════════════════════════════════════════
