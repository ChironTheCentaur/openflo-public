"""Population-frequency comparison across samples/groups (plots + table).

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    current_palette,
    savefig_background,
)
from .tool_window import background_selector, figure_panel


class FrequencyComparisonWindow(tk.Toplevel):
    """Population-frequency & group-comparison view.

    Collects each loaded sample's per-population frequency (reusing the editor's
    ``_collect_stats_rows``), groups the samples by a chosen factor (trial/day,
    comp-vs-samples, or a name token like Stim), and for a selected population +
    metric draws a box/strip comparison with significance annotations plus an
    all-population overview. Exports tidy CSV, **GraphPad Prism**-ready Column
    and Grouped tables, a stats summary, and the figure."""

    METRICS = ('%Parent', '%Total', 'Count')
    FACTORS = ('Trial / day', 'Comp vs Samples', 'Name token')

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        # Paint the window dark immediately and keep it HIDDEN until the dark
        # figure + loading placeholder are in place, so it never maps in a white
        # state and flashes white→dark (an epilepsy-safety concern).
        # Anti-flash: paint dark + map the window fully TRANSPARENT (alpha 0) so
        # it lays out and renders the figure off-screen-invisible, then a render
        # listener (_on_canvas_drawn) flips it opaque once the first real plot is
        # painted. The first VISIBLE frame is the rendered dark plot — never a
        # white flash. _revealed guards the one-shot reveal; the 2.5 s timer is a
        # safety net if the draw_event never arrives.
        self._dark = _dialog_dark_on(editor)
        self._revealed = not self._dark
        self._has_real_draw = False
        self._draw_cid = None
        if self._dark:
            try:
                self.configure(bg=current_palette()['bg'])
                self.attributes('-alpha', 0.0)
            except Exception:
                self._revealed = True   # opacity unsupported → just show normally
        self.title("Population frequencies & group comparison")
        self.geometry("1120x760")
        self._rows = []
        self._tidy = None
        self._last_res = None

        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Population:").pack(side='left')
        self.pop_var = tk.StringVar()
        self.pop_combo = ttk.Combobox(ctl, textvariable=self.pop_var, width=26,
                                      state='readonly')
        self.pop_combo.pack(side='left', padx=(2, 8))
        ttk.Label(ctl, text="Metric:").pack(side='left')
        self.metric_var = tk.StringVar(value='%Parent')
        ttk.Combobox(ctl, textvariable=self.metric_var, width=8,
                     state='readonly', values=self.METRICS).pack(
            side='left', padx=(2, 8))
        ttk.Label(ctl, text="Group by:").pack(side='left')
        self.factor_var = tk.StringVar(value='Trial / day')
        ttk.Combobox(ctl, textvariable=self.factor_var, width=14,
                     state='readonly', values=self.FACTORS).pack(
            side='left', padx=(2, 4))
        ttk.Label(ctl, text="Tokens:").pack(side='left')
        self.tokens_var = tk.StringVar(value='Stim, Ctrl')
        ttk.Entry(ctl, textvariable=self.tokens_var, width=14).pack(
            side='left', padx=(2, 8))
        self.param_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Parametric", variable=self.param_var).pack(
            side='left', padx=(0, 8))
        ttk.Button(ctl, text="Update", command=self._rebuild).pack(side='left')
        for w in (self.pop_combo,):
            w.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        ttk.Button(exp, text="Tidy CSV…",
                   command=self._export_tidy).pack(side='left')
        ttk.Button(exp, text="Prism Column…",
                   command=self._export_prism_column).pack(
            side='left', padx=(4, 0))
        ttk.Button(exp, text="Prism Grouped…",
                   command=self._export_prism_grouped).pack(
            side='left', padx=(4, 0))
        ttk.Button(exp, text="Stats summary…",
                   command=self._export_summary).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Diff. abundance…",
                   command=self._diff_abundance).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Compare all…",
                   command=self._compare_all).pack(side='left', padx=(4, 0))
        ttk.Button(exp, text="Figure…",
                   command=self._export_figure).pack(side='left', padx=(4, 0))
        self.bg_var, _bg_combo = background_selector(
            exp, var=tk.StringVar(value='White'))
        _bg_combo.pack(side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        # dark=False: themed in _draw after rendering (preserves prior behaviour).
        self._figframe, self._fig, self._canvas = figure_panel(
            self, figsize=(10.5, 4.8), dpi=100, dark=False)

        # Lazy-loader placeholder: a solid dark panel with a "Loading…" label
        # laid OVER the canvas while the (off-thread) collect runs. Guarantees
        # the window never shows a white slab or flashes white→dark before the
        # first real draw (an epilepsy-safety concern). Destroyed once data is in.
        self._summary = tk.Text(self, height=6, wrap='word')
        self._summary.pack(fill='x', side='bottom')

        # Listen for the figure's first real render, then reveal (see _reveal).
        if not self._revealed:
            try:
                self._draw_cid = self._canvas.mpl_connect(
                    'draw_event', self._on_canvas_drawn)
            except Exception:
                self._draw_cid = None
            self.after(2500, self._reveal)   # safety net
        if self.editor is not None:
            try:
                self.editor.status_var.set("Computing frequencies…")
            except Exception:
                pass
        self.after(30, lambda: self._collect(self._after_collect))

    def _on_canvas_drawn(self, _event=None):
        """draw_event callback: reveal the window on the first render that
        follows the real-data draw (the empty-figure draw during build sets no
        _has_real_draw, so it's ignored)."""
        if self._revealed or not self._has_real_draw:
            return
        self._reveal()

    def _reveal(self):
        """Make the window opaque + ensure it's shown — once. Idempotent."""
        if getattr(self, '_revealed', True):
            return
        self._revealed = True
        try:
            if self._draw_cid is not None:
                self._canvas.mpl_disconnect(self._draw_cid)
                self._draw_cid = None
        except Exception:
            pass
        try:
            self.attributes('-alpha', 1.0)
        except Exception:
            pass
        try:
            self.deiconify()
            self.lift()
        except Exception:
            pass

    def _after_collect(self):
        pops = sorted({r['Population'] for r in self._rows})
        self.pop_combo['values'] = pops
        if pops:
            self.pop_var.set(pops[0])
        # Mark that the upcoming rebuild's draw is the REAL one, so the render
        # listener reveals the window once it paints.
        self._has_real_draw = True
        self._rebuild()
        # Fallback: if no draw_event arrived (e.g. backend quirk), reveal now.
        if not self._revealed:
            self._reveal()

    # ── data ─────────────────────────────────────────────────────────────
    def _collect(self, then=None):
        """Collect frequency rows off the Tk thread (snapshot-then-background)
        so opening the window over many samples/gates doesn't freeze it; ``then``
        runs on the Tk thread once the rows are in."""
        from .async_task import run_async
        want = {'Count', '%Parent', '%Total'}
        try:
            snap = self.editor._stats_snapshot(want)
        except Exception as exc:
            print(f"[frequencies] snapshot failed: {exc}", flush=True)
            self._rows = []
            if then:
                then()
            return

        def _done(res):
            self._rows = res[0]
            if then:
                then()

        def _err(exc):
            print(f"[frequencies] collect failed: {exc}", flush=True)
            self._rows = []
            if then:
                then()

        run_async(self,
                  lambda: self.editor._stats_rows_from_snapshot(snap, want),
                  on_done=_done, on_error=_err)

    def _tokens(self):
        return [t for t in self.tokens_var.get().split(',') if t.strip()]

    def _tidy_frame(self):
        import pandas as pd
        factor = self.factor_var.get()
        tokens = self._tokens() if factor == 'Name token' else None
        recs = []
        for r in self._rows:
            nm = r['Sample']
            recs.append({
                'Sample': nm,
                'Group': self.editor._sample_group_label(nm, factor, tokens),
                'Population': r['Population'],
                'Count': r.get('Count'),
                '%Parent': r.get('%Parent'),
                '%Total': r.get('%Total')})
        return pd.DataFrame(recs)

    def _ordered_groups(self, tidy):
        """Groups in stable sample-load order (so day series stay chronological
        as loaded)."""
        order = {n: i for i, n in enumerate(self.editor._sample_order)}
        seen = {}
        for _, r in tidy.sort_values(
                'Sample', key=lambda s: s.map(lambda n: order.get(n, 1e9))
                ).iterrows():
            seen.setdefault(r['Group'], None)
        return list(seen)

    def _values_by_group(self, tidy, pop, metric):
        sub = tidy[tidy['Population'] == pop]
        vbg = {}
        for g in self._ordered_groups(tidy):
            vals = sub[sub['Group'] == g][metric].astype(float).tolist()
            vbg[g] = vals
        return vbg

    # ── rebuild + draw ───────────────────────────────────────────────────
    def _rebuild(self):
        from .stats import compare_groups
        self._tidy = self._tidy_frame()
        pop = self.pop_var.get()
        metric = self.metric_var.get()
        if not pop or self._tidy.empty:
            self._fig.clear()
            if _dialog_dark_on(self):
                from .theme import THEMES
                self._fig.set_facecolor(THEMES['midnight']['plot_bg'])
            self._canvas.draw()
            return
        vbg = self._values_by_group(self._tidy, pop, metric)
        res = compare_groups(vbg, parametric=self.param_var.get())
        self._last_res = res
        self._draw(vbg, res, self._tidy, pop, metric)
        self._write_summary(res, pop, metric)

    def _draw(self, vbg, res, tidy, pop, metric):
        import numpy as _np
        fig = self._fig
        fig.clear()
        # clear() resets the facecolor to matplotlib-white; re-darken it BEFORE
        # any drawing so an intermediate paint can never flash white in dark UI.
        if _dialog_dark_on(self):
            from .theme import THEMES
            fig.set_facecolor(THEMES['midnight']['plot_bg'])
        axA = fig.add_subplot(1, 2, 1)
        axB = fig.add_subplot(1, 2, 2)
        groups = list(vbg)
        data = [_np.asarray(vbg[g], float) for g in groups]
        data = [d[_np.isfinite(d)] for d in data]
        pos = list(range(1, len(groups) + 1))
        if any(len(d) for d in data):
            axA.boxplot(data, positions=pos, widths=0.55, showfliers=False)
            rng = _np.random.default_rng(0)
            for i, d in zip(pos, data, strict=True):
                if len(d):
                    jit = i + (rng.random(len(d)) - 0.5) * 0.16
                    axA.scatter(jit, d, s=16, alpha=0.75, color='#1f77b4',
                                zorder=3, linewidths=0)
        axA.set_xticks(pos)
        axA.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
        axA.set_ylabel(metric, fontsize=9)
        axA.set_title(pop, fontsize=9)
        self._draw_sig(axA, data, groups, res)

        # Panel B — all-population overview: mean metric per group (top pops).
        self._draw_overview(axB, tidy, metric, groups)
        try:
            fig.tight_layout()
        except Exception:
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _draw_sig(self, ax, data, groups, res):
        from .stats import p_to_stars
        finite = [d for d in data if len(d)]
        if not finite:
            return
        ymax = max(float(d.max()) for d in finite)
        ymin = min(float(d.min()) for d in finite)
        span = (ymax - ymin) or (abs(ymax) or 1.0)
        h = span * 0.05
        base = ymax + span * 0.08
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
        if pairs:
            ax.set_ylim(top=base + len(pairs) * h * 2.4 + span * 0.12)

    def _draw_overview(self, ax, tidy, metric, groups):
        import numpy as _np
        means = (tidy.groupby(['Population', 'Group'])[metric]
                 .mean().reset_index())
        # Rank populations by overall mean; cap to keep the chart readable.
        overall = (means.groupby('Population')[metric].mean()
                   .sort_values(ascending=False))
        pops = list(overall.index[:10])
        capped = len(overall) > len(pops)
        x = _np.arange(len(pops))
        n = max(1, len(groups))
        w = 0.8 / n
        for gi, g in enumerate(groups):
            vals = []
            for p in pops:
                row = means[(means['Population'] == p) & (means['Group'] == g)]
                vals.append(float(row[metric].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (gi - (n - 1) / 2.0) * w, vals, width=w, label=str(g))
        ax.set_xticks(x)
        ax.set_xticklabels([p.split('/')[-1] for p in pops], rotation=40,
                           ha='right', fontsize=7)
        ax.set_ylabel(f"mean {metric}", fontsize=9)
        ax.set_title("All populations" + (" (top 10)" if capped else ""),
                     fontsize=9)
        if len(groups) <= 8:
            ax.legend(fontsize=7, framealpha=0.85)

    def _write_summary(self, res, pop, metric):
        lines = [f"Population: {pop}    Metric: {metric}"]
        if res.get('test'):
            lines.append(f"Test: {res['test']}    p = {res.get('p'):.4g}")
        else:
            lines.append("Test: (need ≥2 non-empty groups)")
        for g, st in res.get('groups', {}).items():
            lines.append(f"  {g}: n={st['n']}  mean={st['mean']:.3g}  "
                         f"median={st['median']:.3g}  sd={st['sd']:.3g}")
        if res.get('posthoc'):
            lines.append("Pairwise (BH-adjusted):")
            from .stats import p_to_stars
            for pr in res['posthoc']:
                lines.append(f"  {pr['a']} vs {pr['b']}: "
                             f"p_adj={pr.get('p_adj'):.4g} "
                             f"{p_to_stars(pr.get('p_adj'))}")
        self._summary.configure(state='normal')
        self._summary.delete('1.0', 'end')
        self._summary.insert('1.0', "\n".join(lines))
        self._summary.configure(state='disabled')

    # ── exports ──────────────────────────────────────────────────────────
    def _ask(self, default, ftypes):
        return filedialog.asksaveasfilename(
            parent=self, defaultextension=os.path.splitext(default)[1],
            initialfile=default, filetypes=ftypes + [('All files', '*.*')])

    def _export_tidy(self):
        if self._tidy is None or self._tidy.empty:
            return
        path = self._ask('frequencies_tidy.csv', [('CSV', '*.csv')])
        if path:
            self._tidy.to_csv(path, index=False)
            self._done('frequencies.export', path, kind='tidy')

    def _export_prism_column(self):
        from .stats import to_prism_column
        pop, metric = self.pop_var.get(), self.metric_var.get()
        if self._tidy is None or not pop:
            return
        vbg = self._values_by_group(self._tidy, pop, metric)
        path = self._ask('prism_column.csv', [('CSV', '*.csv')])
        if path:
            to_prism_column(vbg).to_csv(path, index=False)
            self._done('frequencies.export', path, kind='prism_column')

    def _export_prism_grouped(self):
        from .stats import to_prism_grouped
        pop, metric = self.pop_var.get(), self.metric_var.get()
        if self._tidy is None or not pop:
            return
        tokens = self._tokens()
        if not tokens:
            messagebox.showinfo(
                "Prism Grouped",
                "Set comma-separated condition Tokens (e.g. 'Stim, Ctrl') — "
                "the Grouped table is Day (rows) × condition (columns).",
                parent=self)
            return
        sub = self._tidy[self._tidy['Population'] == pop].copy()
        sub['Day'] = [self.editor._sample_group_label(s, 'Trial / day')
                      for s in sub['Sample']]
        sub['Cond'] = [self.editor._sample_group_label(s, 'Name token', tokens)
                       for s in sub['Sample']]
        path = self._ask('prism_grouped.csv', [('CSV', '*.csv')])
        if path:
            to_prism_grouped(sub, 'Day', 'Cond', metric).to_csv(path)
            self._done('frequencies.export', path, kind='prism_grouped')

    def _export_summary(self):
        path = self._ask('frequencies_stats.md', [('Markdown', '*.md'),
                                                  ('Text', '*.txt')])
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._summary.get('1.0', 'end'))
        self._done('frequencies.export', path, kind='summary')

    def _export_figure(self):
        path = self._ask('frequencies.png',
                         [('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if not path:
            return
        savefig_background(self._fig, path, background=self.bg_var.get())
        self._done('frequencies.export', path, kind='figure',
                   background=self.bg_var.get())

    def _done(self, action, path, **details):
        try:
            self.editor._audit(action, path=path, **details)
        except Exception:
            pass
        messagebox.showinfo("Frequencies", f"Exported:\n{path}", parent=self)

    def _diff_abundance(self):
        """Run the negative-binomial differential-abundance GLM over the
        populations between the two grouping levels, using each sample's total
        event count as the library-size offset, and show the results table."""
        from .diffexp import differential_abundance
        if self._tidy is None or self._tidy.empty:
            return
        tidy = self._tidy
        groups = self._ordered_groups(tidy)
        if len(groups) != 2:
            messagebox.showinfo(
                "Differential abundance",
                "Differential abundance needs exactly 2 groups — pick a "
                "Group-by / tokens that yield two (e.g. ctrl vs treat).",
                parent=self)
            return
        # counts: populations × samples (Count); group + library size per sample.
        wide = tidy.pivot_table(index='Population', columns='Sample',
                                values='Count', aggfunc='first', fill_value=0)
        samples = list(wide.columns)
        grp = [tidy.loc[tidy['Sample'] == s, 'Group'].iloc[0] for s in samples]
        col_sums = wide.to_numpy(dtype=float).sum(axis=0)   # per-sample totals
        lib = []
        for k, s in enumerate(samples):
            csum = int(col_sums[k])
            ev = (len(self.editor._samples[s].data)
                  if s in self.editor._samples else csum)
            lib.append(max(ev, csum, 1))
        try:
            rows = differential_abundance(wide, grp, lib_sizes=lib)
        except Exception as exc:
            messagebox.showerror("Differential abundance",
                                 f"Failed: {exc}", parent=self)
            return
        # Label from the direction the GLM actually fitted: rows carry
        # group_a/group_b (= the pivot's alphabetical sample-column order), which
        # differs from the load-order `groups`. Using `groups` mislabels the
        # %-headers AND flips the log2FC sign whenever load order != pivot order.
        disp = [rows[0]['group_a'], rows[0]['group_b']] if rows else groups
        try:
            self.editor._audit('diff_abundance', n_populations=len(rows),
                               group_a=disp[0], group_b=disp[1])
        except Exception:
            pass
        from .ui_diff import DiffAbundanceWindow
        DiffAbundanceWindow(self, rows, disp)

    def _compare_all(self):
        """Compare EVERY population across the current grouping in one pass
        (BH-corrected across populations) and open a results table + volcano —
        instead of stepping through populations one at a time."""
        from .stats import compare_all_features
        from .ui_diff import CompareAllWindow
        if self._tidy is None or self._tidy.empty:
            return
        tidy = self._tidy
        metric = self.metric_var.get()
        groups = self._ordered_groups(tidy)
        if len(groups) < 2:
            messagebox.showinfo(
                "Compare all populations",
                "Need at least 2 groups — pick a Group-by / tokens that yield "
                "two or more (e.g. Stim vs Ctrl).", parent=self)
            return
        pops = sorted({r['Population'] for r in self._rows})
        vbf = {p: self._values_by_group(tidy, p, metric) for p in pops}
        res = compare_all_features(vbf, parametric=self.param_var.get())
        try:
            self.editor._audit('compare_all_populations', metric=metric,
                               n_populations=len(res), groups=','.join(groups))
        except Exception:
            pass
        CompareAllWindow(self, res, groups, metric)
