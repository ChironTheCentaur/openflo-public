"""The polyfunctionality window: SPICE-style bars and pies of every
combination of a set of gates, with the frequency table.

The figure is drawn by `draw_polyfunctionality` (no Tk) so it can be tested
and exported; `PolyfunctionWindow` is the window around it.
"""
from __future__ import annotations

import csv
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .gate_combos import degree_fractions, group_means, sign_patterns
from .tool_window import (
    ToolWindow,
    ask_csv_path,
    export_figure,
    figure_panel,
)
from .ui_tips import tip

_PIE_COLORS = ('#cfd8dc', '#4e79a7', '#59a14f', '#f28e2b', '#e15759',
               '#b07aa1', '#76b7b2')


def bar_order(labels, show_negative=False):
    """The combinations the bars show, in SPICE order (most positives
    first); the all-negative one only when asked for."""
    from .gate_combos import combination_name
    pats = sign_patterns(len(labels))
    if not show_negative:
        pats = [p for p in pats if any(p)]
    return [(p, combination_name(labels, p)) for p in pats]


def draw_polyfunctionality(fig, rows, labels, show_negative=False):
    """Draw bars (mean % of parent per group, a dot per sample), the +/-
    grid under them, and a pie per group of how many gates each positive
    event is in. Returns the axes ``(bars, grid, [pies])``."""
    fig.clear()
    order = bar_order(labels, show_negative)
    means = group_means(rows)
    groups = list(dict.fromkeys(r['Group'] for r in rows)) or ['']
    n_g = len(groups)
    gs = fig.add_gridspec(2, max(n_g, 1) + 3,
                          height_ratios=[3, max(1, len(labels)) * 0.35])
    ax = fig.add_subplot(gs[0, :3])
    grid = fig.add_subplot(gs[1, :3], sharex=ax)
    x = np.arange(len(order))
    width = 0.8 / n_g
    for i, grp in enumerate(groups):
        off = (i - (n_g - 1) / 2) * width
        vals = [means.get(grp, {}).get(name, np.nan) for _p, name in order]
        ax.bar(x + off, vals, width=width * 0.95,
               label=grp or 'All samples', color=f'C{i}', alpha=0.85)
        for j, (_p, name) in enumerate(order):
            pts = [r['% of parent'] for r in rows
                   if r['Group'] == grp and r['Combination'] == name
                   and r['% of parent'] == r['% of parent']]
            if pts:
                ax.plot([x[j] + off] * len(pts), pts, 'o', ms=3,
                        color='black', alpha=0.6)
    ax.set_ylabel('% of parent')
    ax.tick_params(axis='x', labelbottom=False)
    if n_g > 1 or groups[0]:
        ax.legend(fontsize=8, frameon=False)
    for k in range(len(labels)):
        yk = len(labels) - 1 - k
        for j, (p, _name) in enumerate(order):
            grid.plot(x[j], yk, 'o' if p[k] else '_', ms=6 if p[k] else 8,
                      color='black' if p[k] else 'grey')
    grid.set_yticks(range(len(labels)))
    grid.set_yticklabels(list(reversed(labels)), fontsize=8)
    grid.set_ylim(-0.6, len(labels) - 0.4)
    grid.set_xticks([])
    for side in ('top', 'right', 'bottom'):
        grid.spines[side].set_visible(False)
    pies = []
    deg = degree_fractions(rows, len(labels))
    for i, grp in enumerate(groups):
        pa = fig.add_subplot(gs[:, 3 + i])
        samples = list(dict.fromkeys(r['Sample'] for r in rows
                                     if r['Group'] == grp))
        known = [deg[s] for s in samples if np.isfinite(deg[s]).all()]
        pos = (np.mean(known, axis=0)[1:] if known
               else np.array([], dtype=float))
        if pos.size and pos.sum() > 0:
            pa.pie(pos / pos.sum(), colors=_PIE_COLORS[1:len(pos) + 1],
                   labels=[str(k) for k in range(1, len(pos) + 1)],
                   textprops={'fontsize': 8})
        else:
            pa.text(0.5, 0.5, 'no positive events', ha='center',
                    va='center', fontsize=8, transform=pa.transAxes)
            pa.axis('off')
        pa.set_title(f"{grp or 'All samples'}\ngates positive", fontsize=8)
        pies.append(pa)
    fig.tight_layout()
    return ax, grid, pies


def write_table(path, rows):
    """The frequency table as CSV, one row per sample and combination;
    NA where a frequency cannot be computed."""
    cols = ['Sample', 'Group', 'Combination', 'Positives', 'Count',
            'Parent count', '% of parent']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in ('Count', 'Parent count', '% of parent'):
                v = out[k]
                if isinstance(v, float) and v != v:
                    out[k] = 'NA'
                elif k == '% of parent':
                    out[k] = f'{v:.4f}'
            w.writerow(out)


class PolyfunctionWindow(ToolWindow):
    """Bars, +/- grid and pies of a set of combination gates."""

    def __init__(self, editor, rows=(), labels=(), parent_name='the parent'):
        rows, labels = list(rows), list(labels)
        super().__init__(editor, f"Polyfunctionality — {', '.join(labels)}",
                         geometry='980x620', editor=editor)
        self.rows, self.labels = rows, labels
        top = ttk.Frame(self, padding=(10, 8, 10, 0))
        top.pack(side='top', fill='x')
        ttk.Label(top, text=(f"{2 ** len(labels)} combinations of "
                             f"{len(labels)} gates, as % of {parent_name}")
                  ).pack(side='left')
        self.neg_var = tk.BooleanVar(value=False)
        neg = ttk.Checkbutton(top, text="Show all-negative",
                              variable=self.neg_var, command=self.redraw)
        neg.pack(side='right')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')
        fig_b = ttk.Button(btns, text="Export figure…",
                           command=lambda: export_figure(
                               self, self.fig,
                               initialfile='polyfunctionality.png'))
        fig_b.pack(side='right', padx=(0, 6))
        tab_b = ttk.Button(btns, text="Export table…",
                           command=self.export_table)
        tab_b.pack(side='right', padx=(0, 6))
        tip(editor, neg, "Add the bar of events in none of the gates (off by "
            "default: it is usually most of the parent and dwarfs the rest).")
        tip(editor, fig_b, "Save the bars and pies as an image.")
        tip(editor, tab_b, "Save every sample's count and % of parent for each "
            "combination as CSV; NA where it cannot be computed.")
        _frame, self.fig, self.canvas = figure_panel(self, figsize=(9.5, 5.5),
                                                     dark=False)
        self.redraw()

    def redraw(self):
        if not self.labels:
            return
        draw_polyfunctionality(self.fig, self.rows, self.labels,
                               show_negative=bool(self.neg_var.get()))
        self.canvas.draw_idle()

    def export_table(self, path=None):
        path = path or ask_csv_path(self, title="Export combination table",
                                    initialfile='combinations.csv')
        if not path:
            return None
        try:
            write_table(path, self.rows)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self)
            return None
        return path
