"""The review grid for gates re-fitted on each sample: one panel per sample
showing where the gate was (grey, dashed) and where the re-fit puts it
(red), with the share of the parent inside before and after. Nothing
changes until Accept.

`draw_review` draws the figure without Tk; `RefitReviewWindow` wraps it.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from .gate_refit import outline
from .tool_window import ToolWindow, figure_panel
from .ui_tips import tip


def _pct(v):
    return 'NA' if v != v else f'{100.0 * v:.1f}%'


def draw_review(fig, proposals):
    """One panel per proposal (``{'sample', 'old', 'new', 'why', 'data',
    'before', 'after'}``). Returns the axes."""
    fig.clear()
    n = max(1, len(proposals))
    cols = min(4, n)
    rows = math.ceil(n / cols)
    axes = []
    for i, p in enumerate(proposals):
        ax = fig.add_subplot(rows, cols, i + 1)
        axes.append(ax)
        old, new = p['old'], p['new']
        if old.get('kind') in ('threshold', 'interval'):
            v = np.asarray(p['data'][old['channel']], dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                ax.hist(v, bins=128, color='#90a4ae', histtype='stepfilled')
            keys = ('value',) if old['kind'] == 'threshold' else ('lo', 'hi')
            for k in keys:
                if abs(float(old[k])) < 1e12:
                    ax.axvline(float(old[k]), color='grey', ls='--', lw=1)
                if abs(float(new[k])) < 1e12:
                    ax.axvline(float(new[k]), color='#d62728', lw=1.4)
        else:
            x = np.asarray(p['data'][old['x_channel']], dtype=float)
            y = np.asarray(p['data'][old['y_channel']], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 1:
                ax.hist2d(x[ok], y[ok], bins=96, cmap='Greys', cmin=1)
            for g, style in ((old, dict(color='grey', ls='--', lw=1)),
                             (new, dict(color='#d62728', lw=1.4))):
                xy = outline(g)
                if xy is not None:
                    ax.plot(xy[0], xy[1], **style)
        title = f"{p['sample']}\n{_pct(p['before'])} -> {_pct(p['after'])}"
        if p.get('why'):
            title += f"\nkept: {p['why']}"
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    return axes


class RefitReviewWindow(ToolWindow):
    """Shows every sample's re-fit; Accept installs the ticked ones."""

    def __init__(self, editor, proposals=(), on_accept=None,
                 title="Re-place on each sample"):
        proposals = list(proposals)
        super().__init__(editor, title, geometry='960x680', editor=editor)
        self.proposals = proposals
        self.on_accept = on_accept
        side = ttk.Frame(self, padding=(10, 8))
        side.pack(side='right', fill='y')
        ttk.Label(side, text="Accept for:").pack(anchor='w')
        self.vars = {}
        for p in proposals:
            var = tk.BooleanVar(value=not p.get('why'))
            self.vars[p['sample']] = var
            cb = ttk.Checkbutton(side, text=p['sample'], variable=var)
            cb.pack(anchor='w')
            tip(editor, cb, "Ticked: this sample's gate moves to the red "
                "placement. A sample whose landmark was not found starts "
                "unticked and keeps its gate.")
        ttk.Button(side, text="Cancel", command=self.destroy).pack(
            side='bottom', fill='x')
        acc = ttk.Button(side, text="Accept", command=self.accept)
        acc.pack(side='bottom', fill='x', pady=(0, 6))
        tip(editor, acc, "Move the gate on every ticked sample (one undo "
            "step); the rest keep theirs.")
        _f, self.fig, self.canvas = figure_panel(self, figsize=(8, 6),
                                                 dark=False)
        draw_review(self.fig, proposals)
        self.canvas.draw_idle()

    def accept(self):
        chosen = [p for p in self.proposals if self.vars[p['sample']].get()]
        self.destroy()
        if self.on_accept is not None:
            self.on_accept(chosen)
