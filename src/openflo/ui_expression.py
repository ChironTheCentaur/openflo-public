"""Per-marker expression comparison across populations/groups.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    savefig_background,
)
from .tool_window import background_selector, figure_panel


class MarkerExpressionWindow(tk.Toplevel):
    """Marker-expression distributions by group — violin or ridgeline.

    Pools each enabled sample's per-cell values for a chosen marker (resolving
    the marker across fluors by antibody label), groups the samples by a factor
    (trial/day, comp-vs-samples, or a name token), and draws a violin or
    ridgeline plot per group. Significance comes from a per-SAMPLE-median
    comparison (so the test treats each sample as a replicate, not each cell),
    which is also what the **GraphPad Prism** Column export contains."""

    FACTORS = ('Trial / day', 'Comp vs Samples', 'Name token')
    PER_GROUP_CAP = 40_000

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Marker expression by group")
        self.geometry("1000x720")
        self._percell = {}
        self._medians = {}
        self._groups = []

        chans = [c for c in editor._channels]
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Marker:").pack(side='left')
        self.marker_var = tk.StringVar()
        disp = [editor._fmt_channel(c) for c in chans]
        ttk.Combobox(ctl, textvariable=self.marker_var, width=20,
                     state='readonly', values=disp).pack(side='left', padx=(2, 8))
        fluor = next((editor._fmt_channel(c) for c in chans
                      if c in (self._first_fluor() or [])), disp[0] if disp
                     else '')
        self.marker_var.set(fluor)
        ttk.Label(ctl, text="Group by:").pack(side='left')
        self.factor_var = tk.StringVar(value='Trial / day')
        ttk.Combobox(ctl, textvariable=self.factor_var, width=14,
                     state='readonly', values=self.FACTORS).pack(
            side='left', padx=(2, 4))
        ttk.Label(ctl, text="Tokens:").pack(side='left')
        self.tokens_var = tk.StringVar(value='Stim, Ctrl')
        ttk.Entry(ctl, textvariable=self.tokens_var, width=14).pack(
            side='left', padx=(2, 8))
        ttk.Label(ctl, text="Plot:").pack(side='left')
        self.plot_var = tk.StringVar(value='Violin')
        ttk.Combobox(ctl, textvariable=self.plot_var, width=9, state='readonly',
                     values=['Violin', 'Ridgeline']).pack(side='left', padx=(2, 8))
        self.param_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Parametric",
                        variable=self.param_var).pack(side='left', padx=(0, 8))
        ttk.Button(ctl, text="Update", command=self._rebuild).pack(side='left')

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        ttk.Button(exp, text="Prism Column (medians)…",
                   command=self._export_prism).pack(side='left')
        ttk.Button(exp, text="Stats summary…",
                   command=self._export_summary).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Figure…",
                   command=self._export_figure).pack(side='left', padx=(4, 0))
        self.bg_var, _bg_combo = background_selector(exp)
        _bg_combo.pack(side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        # dark=False: themed in _draw after rendering (preserves prior behaviour).
        _frame, self._fig, self._canvas = figure_panel(
            self, figsize=(9.5, 5.0), dpi=100, dark=False)
        self._summary = tk.Text(self, height=5, wrap='word')
        self._summary.pack(fill='x', side='bottom')
        self._rebuild()

    def _first_fluor(self):
        for n in self.editor._sample_order:
            s = self.editor._samples.get(n)
            if s is not None:
                return getattr(s, 'fluor_channels', []) or []
        return []

    def _samples(self):
        names = self.editor._selected_samples() or (
            [self.editor._active_sample] if self.editor._active_sample else [])
        return [n for n in names if n in self.editor._samples]

    def _tokens(self):
        return [t for t in self.tokens_var.get().split(',') if t.strip()]

    def _collect(self):
        import numpy as _np
        ch = self.editor._resolve_channel(self.marker_var.get())
        factor = self.factor_var.get()
        tokens = self._tokens() if factor == 'Name token' else None
        order = {n: i for i, n in enumerate(self.editor._sample_order)}
        percell, medians, groups = {}, {}, []
        for n in sorted(self._samples(), key=lambda x: order.get(x, 1e9)):
            s = self.editor._samples[n]
            col = self.editor._marker_column_for(s, ch)
            if not col:
                continue
            vals = _np.asarray(s.data[col].values, dtype=float)
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                continue
            g = self.editor._sample_group_label(n, factor, tokens)
            if g not in percell:
                percell[g] = []
                medians[g] = []
                groups.append(g)
            percell[g].append(vals)
            medians[g].append(float(_np.median(vals)))
        rng = _np.random.default_rng(0)
        pooled = {}
        for g in groups:
            allv = _np.concatenate(percell[g])
            if allv.size > self.PER_GROUP_CAP:
                allv = allv[rng.choice(allv.size, self.PER_GROUP_CAP,
                                       replace=False)]
            pooled[g] = allv
        self._percell, self._medians, self._groups = pooled, medians, groups

    def _rebuild(self):
        from .stats import compare_groups
        self._collect()
        if not self._groups:
            self._fig.clear()
            if _dialog_dark_on(self):
                _theme_figure_dark(self._fig)
            self._canvas.draw()
            return
        res = compare_groups(self._medians, parametric=self.param_var.get())
        self._draw(res)
        self._write_summary(res)

    def _draw(self, res):
        import numpy as _np
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        groups = self._groups
        marker = self.marker_var.get()
        if self.plot_var.get() == 'Ridgeline':
            self._draw_ridgeline(ax, groups, marker)
        else:
            data = [self._percell[g] for g in groups]
            pos = list(range(1, len(groups) + 1))
            parts = ax.violinplot(data, positions=pos, showmedians=True,
                                  widths=0.8)
            # parts['bodies'] is a list of PolyCollection at runtime; the
            # matplotlib stub types it as a non-iterable Collection.
            bodies: list = list(parts.get('bodies') or [])  # type: ignore
            for b in bodies:
                b.set_alpha(0.6)
            ax.set_xticks(pos)
            ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
            ax.set_ylabel(marker, fontsize=9)
            self._draw_sig(ax, [_np.asarray(self._medians[g]) for g in groups],
                           groups, res)
        ax.set_title(f"{marker} by {self.factor_var.get()}", fontsize=9)
        try:
            fig.tight_layout()
        except Exception:
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _draw_ridgeline(self, ax, groups, marker):
        from .stats import group_kde
        x, dens = group_kde(self._percell)
        if x.size == 0:
            return
        peak = max((d.max() for d in dens.values() if d.size), default=1.0) or 1.0
        step = 0.8
        for i, g in enumerate(groups):
            d = dens.get(g)
            if d is None:
                continue
            base = i * step
            y = base + d / peak * step * 1.6
            ax.fill_between(x, base, y, alpha=0.7, zorder=len(groups) - i)
            ax.plot(x, y, lw=0.8, color='black', alpha=0.5)
        ax.set_yticks([i * step for i in range(len(groups))])
        ax.set_yticklabels(groups, fontsize=8)
        ax.set_xlabel(marker, fontsize=9)

    def _draw_sig(self, ax, medians, groups, res):
        from .stats import p_to_stars
        finite = [m[np.isfinite(m)] for m in medians]
        finite = [m for m in finite if len(m)]
        if not finite:
            return
        ymax = max(float(m.max()) for m in finite)
        span = (ymax - min(float(m.min()) for m in finite)) or (abs(ymax) or 1.0)
        h = span * 0.05
        base = ymax + span * 0.10
        idx = {g: i for i, g in enumerate(groups)}
        pairs = []
        if len(groups) == 2:
            s = p_to_stars(res.get('p'))
            if s:
                pairs = [(0, 1, s)]
        else:
            for pr in res.get('posthoc', []):
                s = p_to_stars(pr.get('p_adj'))
                if s and s != 'ns' and pr['a'] in idx and pr['b'] in idx:
                    pairs.append((idx[pr['a']], idx[pr['b']], s))
            pairs.sort(key=lambda t: abs(t[1] - t[0]))
            pairs = pairs[:6]
        for k, (i, j, s) in enumerate(pairs):
            y = base + k * h * 2.4
            x1, x2 = i + 1, j + 1
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0,
                    color='black')
            ax.text((x1 + x2) / 2.0, y + h, s, ha='center', va='bottom',
                    fontsize=9)

    def _write_summary(self, res):
        lines = [f"Marker: {self.marker_var.get()}    "
                 f"(significance from per-sample medians, n = samples)"]
        if res.get('test'):
            lines.append(f"Test: {res['test']}    p = {res.get('p'):.4g}")
        else:
            lines.append("Test: (need ≥2 groups with samples)")
        for g, st in res.get('groups', {}).items():
            lines.append(f"  {g}: n={st['n']}  median-of-medians="
                         f"{st['median']:.3g}")
        self._summary.configure(state='normal')
        self._summary.delete('1.0', 'end')
        self._summary.insert('1.0', "\n".join(lines))
        self._summary.configure(state='disabled')

    def _export_prism(self):
        from .stats import to_prism_column
        if not self._medians:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='expression_medians_prism.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            to_prism_column(self._medians).to_csv(path, index=False)
            self._done(path, 'prism_column')

    def _export_summary(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.md',
            initialfile='expression_stats.md',
            filetypes=[('Markdown', '*.md'), ('Text', '*.txt')])
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._summary.get('1.0', 'end'))
            self._done(path, 'summary')

    def _export_figure(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='expression.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if path:
            savefig_background(self._fig, path, background=self.bg_var.get())
            self._done(path, 'figure')

    def _done(self, path, kind):
        try:
            self.editor._audit('expression.export', path=path, kind=kind,
                               marker=self.marker_var.get())
        except Exception:
            pass
        messagebox.showinfo("Expression", f"Exported:\n{path}", parent=self)
