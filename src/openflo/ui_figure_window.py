"""Generic prebuilt-figure pop-out window.

Shows a ready-made matplotlib Figure with an Export (PNG/PDF/SVG) + Close bar;
used by the compensation-QC, gating-tree and embedding-comparison views.
Extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
import warnings
from tkinter import ttk

from .theme import match_dialog
from .tool_window import background_selector, export_figure
from .ui_tips import auto_tip


class FigureWindow(tk.Toplevel):
    """Show a prebuilt matplotlib Figure with an Export (PNG/PDF/SVG) + Close
    bar. Used by the compensation-QC, gating-tree and DR-comparison views."""

    def __init__(self, parent, fig, title, geometry='860x660'):
        super().__init__(parent)
        self.title(title)
        self.geometry(geometry)
        self._fig = fig
        # Dark preview when pop-ups should be dark (the toggle or Midnight
        # theme), so the pop-up isn't a blinding white rectangle.
        dark = match_dialog(fig, self)
        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        self._bg, bg_combo = background_selector(bar)
        ttk.Button(bar, text="Close", command=self.destroy).pack(
            side='right', padx=(0, 6), pady=4)
        bg_combo.pack(side='right', padx=(0, 4), pady=4)
        ttk.Label(bar, text="Background:").pack(side='right', padx=(0, 2))
        _xb = ttk.Button(bar, text="Export figure…", command=self._export_figure)
        _xb.pack(side='right', padx=(0, 6), pady=4)
        auto_tip(_xb,
                 "Save this figure as PNG, PDF or SVG at export resolution. "
                 "PDF and SVG stay vector, so they scale without pixelating.")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        cf = ttk.Frame(self)
        cf.pack(fill='both', expand=True)
        canvas = FigureCanvasTkAgg(fig, master=cf)
        _w = canvas.get_tk_widget()
        if dark:
            from .theme import current_palette
            _w.configure(bg=current_palette()['bg'], highlightthickness=0)
        _w.pack(fill='both', expand=True)
        canvas.draw()

    def _export_figure(self):
        export_figure(self, self._fig, background=self._bg.get())


def __getattr__(name):
    # The old private name of FigureWindow, served with a DeprecationWarning.
    if name == '_FigureWindow':
        warnings.warn('_FigureWindow is deprecated and will be removed in a '
                      'future release; use FigureWindow', DeprecationWarning,
                      stacklevel=2)
        return FigureWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
