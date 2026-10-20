"""The NxN unmixed-pairs plot of a spectral unmix (Complaints 24).

Every pair of unmixed fluors of one sample, a histogram on the diagonal
(spectral_af.unmixed_pairs_figure); opened from Spectral QC.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .tool_window import background_selector, export_figure
from .ui_tips import auto_tip


class UnmixedPairsWindow(tk.Toplevel):
    """The pairs figure, scrolled when large, with export."""

    def __init__(self, parent, A, fluors, sample=''):
        super().__init__(parent)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        from .spectral_af import unmixed_pairs_figure
        from .theme import _dialog_dark_on, _theme_figure_dark
        self.title(f"Unmixed pairs — {sample}" if sample else "Unmixed pairs")
        self.minsize(520, 520)
        self.fig = unmixed_pairs_figure(A, list(fluors), title=sample)
        if _dialog_dark_on(self):
            _theme_figure_dark(self.fig)
        bar = ttk.Frame(self, padding=(12, 4, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        self.bg_var, bg = background_selector(bar)
        b = ttk.Button(bar, text="Export figure…", command=self._export)
        b.pack(side='left')
        auto_tip(b, "Save the grid as PNG, SVG or PDF.")
        bg.pack(side='left', padx=(6, 0))
        ttk.Label(self, style='Muted.TLabel', wraplength=600, justify='left',
                  padding=(12, 4),
                  text="Each fluor against each other on arcsinh axes. A "
                       "pair that leans or bends instead of forming a "
                       "cross points at a wrong or missing spectrum.").pack(
            side='bottom', fill='x')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self.canvas.draw_idle()

    def _export(self):
        export_figure(self, self.fig, background=self.bg_var.get(),
                      initialfile='unmixed_pairs.png')
