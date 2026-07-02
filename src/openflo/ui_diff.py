"""Differential-abundance + compare-all result windows.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
)
from .tool_window import figure_panel


class CompareAllWindow(tk.Toplevel):
    """All-population group comparison: a sortable results table (per-group
    means, log2 fold-change, BH-adjusted p, stars) beside a volcano plot
    (log2FC vs −log10 adjusted-p, significant populations highlighted). One
    click compares every population at once; export the full table or the
    volcano figure. The volcano needs the two-group case (log2FC); with >2
    groups the table still shows the omnibus Kruskal-Wallis / ANOVA result."""

    def __init__(self, parent, results, groups, metric):
        super().__init__(parent)
        self.title("Compare all populations")
        self.geometry("1040x620")
        self._results = results
        self._groups = groups
        self._metric = metric
        self._two = len(groups) == 2
        from .stats import volcano_data
        self._volcano = volcano_data(results)

        ttk.Label(self, padding=6, font=('TkDefaultFont', 9, 'bold'),
                  text=(f"{metric} across {len(groups)} groups "
                        f"({', '.join(groups)}) — {len(results)} populations, "
                        f"BH-adjusted.")).pack(fill='x', side='top')
        bar = ttk.Frame(self)
        bar.pack(fill='x')
        ttk.Button(bar, text="Results CSV…", command=self._export_csv).pack(
            side='right', padx=6, pady=4)
        ttk.Button(bar, text="Volcano figure…",
                   command=self._export_figure).pack(side='right', pady=4)

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True)

        # ── left: results table ──
        tbl = ttk.Frame(paned)
        paned.add(tbl, weight=1)
        if self._two:
            cols = ('pop', 'a', 'b', 'log2fc', 'p', 'padj', 'sig')
            heads = {'pop': 'Population', 'a': 'mean ' + groups[0],
                     'b': 'mean ' + groups[1], 'log2fc': 'log2FC',
                     'p': 'p', 'padj': 'p(adj)', 'sig': ''}
            widths = (210, 90, 90, 70, 70, 70, 36)
        else:
            cols = ('pop', 'p', 'padj', 'sig')
            heads = {'pop': 'Population', 'p': 'p (omnibus)',
                     'padj': 'p(adj)', 'sig': ''}
            widths = (320, 100, 90, 40)
        tv = ttk.Treeview(tbl, columns=cols, show='headings')
        for c, w in zip(cols, widths, strict=True):
            tv.heading(c, text=heads[c])
            tv.column(c, width=w, anchor='w', stretch=(c == 'pop'))
        tv.pack(fill='both', expand=True)

        def _f(x, fmt):
            return fmt.format(x) if x is not None and x == x else 'n/a'
        for r in results:
            g = r['groups']
            if self._two:
                ma = g.get(groups[0], {}).get('mean')
                mb = g.get(groups[1], {}).get('mean')
                tv.insert('', 'end', values=(
                    r['feature'], _f(ma, '{:.3g}'), _f(mb, '{:.3g}'),
                    _f(r['effect'], '{:+.2f}'), _f(r['p'], '{:.2g}'),
                    _f(r['p_adj'], '{:.2g}'), r['stars']))
            else:
                tv.insert('', 'end', values=(
                    r['feature'], _f(r['p'], '{:.2g}'),
                    _f(r['p_adj'], '{:.2g}'), r['stars']))

        # ── right: volcano ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        # dark=False: _draw_volcano themes the figure itself.
        _frame, self._fig, self._canvas = figure_panel(
            right, figsize=(5.2, 4.8), dpi=100, dark=False)
        self._draw_volcano()

    def _draw_volcano(self):
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        if not self._two or not self._volcano:
            ax.text(0.5, 0.5, "Volcano needs exactly 2 groups\n"
                    "(log2 fold-change).", ha='center', va='center',
                    fontsize=9, color='#666', transform=ax.transAxes)
            ax.set_axis_off()
            if _dialog_dark_on(self):
                _theme_figure_dark(fig)
            self._canvas.draw()
            return
        import numpy as _np
        xs = _np.array([p['x'] for p in self._volcano])
        ys = _np.array([p['y'] for p in self._volcano])
        sig = _np.array([p['significant'] for p in self._volcano])
        ax.scatter(xs[~sig], ys[~sig], s=18, c='#bbb', linewidths=0,
                   label='ns')
        ax.scatter(xs[sig], ys[sig], s=22, c='#d62728', linewidths=0,
                   label='significant')
        ax.axhline(-_np.log10(0.05), color='#888', ls='--', lw=.7)
        for xc in (-1.0, 1.0):
            ax.axvline(xc, color='#888', ls=':', lw=.7)
        # label the most significant populations
        for p in sorted(self._volcano, key=lambda d: -d['y'])[:6]:
            if p['significant']:
                ax.annotate(p['feature'], (p['x'], p['y']), fontsize=7,
                            xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(f"log2 fold-change ({self._groups[1]} / {self._groups[0]})",
                      fontsize=9)
        ax.set_ylabel("−log10 adjusted p", fontsize=9)
        ax.set_title("Volcano", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        try:
            fig.tight_layout()
        except Exception:
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _flat_rows(self):
        out = []
        for r in self._results:
            row = {'population': r['feature'], 'test': r['test'],
                   'log2FC': r['effect'], 'p': r['p'], 'p_adj': r['p_adj'],
                   'sig': r['stars']}
            for gname, gs in r['groups'].items():
                row[f'mean_{gname}'] = gs.get('mean')
                row[f'n_{gname}'] = gs.get('n')
            out.append(row)
        return out

    def _export_csv(self):
        import pandas as pd
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='compare_all_populations.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            pd.DataFrame(self._flat_rows()).to_csv(path, index=False)
            messagebox.showinfo("Compare all populations",
                                f"Exported:\n{path}", parent=self)

    def _export_figure(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='volcano.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'),
                       ('All files', '*.*')])
        if path:
            self._fig.savefig(path, dpi=200, bbox_inches='tight')
            messagebox.showinfo("Compare all populations",
                                f"Saved:\n{path}", parent=self)


class DiffAbundanceWindow(tk.Toplevel):
    """Results table for the negative-binomial differential-abundance test
    (log2FC of group-B vs group-A proportion per population, with adjusted
    p-values and significance stars), with CSV export."""

    def __init__(self, parent, rows, groups):
        super().__init__(parent)
        self.title("Differential abundance (NB-GLM)")
        self.geometry("780x460")
        self._rows = rows
        from .stats import p_to_stars
        ttk.Label(self, padding=6,
                  text=(f"{groups[1]} vs {groups[0]} — negative-binomial GLM on "
                        f"counts (library-size offset). {len(rows)} populations."),
                  font=('TkDefaultFont', 9, 'bold')).pack(fill='x', side='top')
        bar = ttk.Frame(self)
        bar.pack(fill='x')
        ttk.Button(bar, text="Export CSV…", command=self._export).pack(
            side='right', padx=6, pady=4)
        cols = ('pop', 'log2fc', 'pa', 'pb', 'p', 'padj', 'sig')
        heads = {'pop': 'Population', 'log2fc': 'log2FC', 'pa': '%' + groups[0],
                 'pb': '%' + groups[1], 'p': 'p', 'padj': 'p(adj)',
                 'sig': ''}
        widths = (300, 70, 70, 70, 80, 80, 40)
        tv = ttk.Treeview(self, columns=cols, show='headings')
        for c, w in zip(cols, widths, strict=True):
            tv.heading(c, text=heads[c])
            tv.column(c, width=w, anchor='w', stretch=(c == 'pop'))
        tv.pack(fill='both', expand=True)
        for r in rows:
            tv.insert('', 'end', values=(
                r['cluster'], f"{r['log2fc']:+.2f}",
                f"{r['prop_a'] * 100:.2f}", f"{r['prop_b'] * 100:.2f}",
                f"{r['p']:.2g}" if r['p'] == r['p'] else 'n/a',
                f"{r['p_adj']:.2g}" if r['p_adj'] == r['p_adj'] else 'n/a',
                p_to_stars(r['p_adj'])))

    def _export(self):
        import pandas as pd
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='differential_abundance.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            pd.DataFrame(self._rows).to_csv(path, index=False)
            messagebox.showinfo("Differential abundance",
                                f"Exported:\n{path}", parent=self)
