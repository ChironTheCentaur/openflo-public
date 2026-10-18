"""Population-frequency comparison across samples/groups (plots + table).

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .group_test_choice import add_test_controls, pair_key, power_lines, test_kwargs
from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    current_palette,
    plot_ink,
    savefig_background,
)
from .tool_window import background_selector, figure_panel
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import tip


class FrequencyComparisonWindow(tk.Toplevel):
    """Population-frequency & group-comparison view.

    Collects each loaded sample's per-population frequency (reusing the editor's
    ``_collect_stats_rows``), groups the samples by a chosen factor (trial/day,
    comp-vs-samples, or a name token like Stim), and for a selected population +
    metric draws a box/strip comparison with significance annotations plus an
    all-population overview. Exports tidy CSV, **GraphPad Prism**-ready Column
    and Grouped tables, a stats summary, and the figure."""

    METRICS = ('%Parent', '%Total', 'Count')
    # The scale "t-test, transformed" compares each metric on.
    TRANSFORM = {'%Parent': 'logit', '%Total': 'logit', 'Count': 'log1p'}
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
        self._left_out = []          # samples the current grouping leaves out

        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Population:").pack(side='left')
        self.pop_var = tk.StringVar()
        self.pop_combo = ttk.Combobox(ctl, textvariable=self.pop_var, width=26,
                                      state='readonly')
        self.pop_combo.pack(side='left', padx=(2, 8))
        tip(editor, self.pop_combo,
            "Which gated population's frequency to report. The list is every "
            "population defined in the gate tree.")
        ttk.Label(ctl, text="Metric:").pack(side='left')
        self.metric_var = tk.StringVar(value='%Parent')
        _mc = ttk.Combobox(ctl, textvariable=self.metric_var, width=8,
                           state='readonly', values=self.METRICS)
        _mc.pack(side='left', padx=(2, 8))
        tip(editor, _mc,
            "What the percentage is OF: %Parent divides by the immediate "
            "parent gate, %Total by all events in the sample. They differ "
            "sharply for a deep gate, so state which one a figure uses.")
        # Grouping and Update on a second row: on one row the controls were
        # wider than the window and the packer dropped Update off the end.
        ctl2 = ttk.Frame(self, padding=(6, 0, 6, 6))
        ctl2.pack(fill='x', side='top')
        ttk.Label(ctl2, text="Group by:").pack(side='left')
        self.factor_var = tk.StringVar(value='Trial / day')
        _fc = ttk.Combobox(ctl2, textvariable=self.factor_var, width=16,
                           state='readonly', values=self.FACTORS)
        _fc.pack(side='left', padx=(2, 4))
        tip(editor, _fc,
            "How samples are split into groups for comparison — by trial/day, "
            "or by matching the tokens typed to the right against sample "
            "names. This defines what the statistics compare.")
        ttk.Label(ctl2, text="Tokens:").pack(side='left')
        self.tokens_var = tk.StringVar(value='Stim, Ctrl')
        _te = ttk.Entry(ctl2, textvariable=self.tokens_var, width=14)
        _te.pack(side='left', padx=(2, 8))
        tip(editor, _te,
            "Comma-separated substrings matched against sample names to form "
            "groups, e.g. 'Stim, Ctrl'. A sample matching none is left out. "
            "Commits on Enter or when focus leaves.")
        self.test_var, self.paired_var = add_test_controls(
            editor, ctl2, on_change=self._rebuild)
        _ub = ttk.Button(ctl2, text="Update", command=self._rebuild)
        _ub.pack(side='left')
        tip(editor, _ub,
            "Recompute the table from the current gates and settings. The "
            "population, grouping and token boxes already refresh on change; "
            "use this after editing gates in the editor.")
        # Rebuild the tidy frame whenever the GROUPING changes — not just on the
        # population combo / Update — so Diff. abundance / Compare all can't run
        # on a stale grouping (typed tokens commit on <Return> or focus-out,
        # which also fires just before a clicked export button's command).
        _fc.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())
        _te.bind('<Return>', lambda *_: self._rebuild())
        _te.bind('<FocusOut>', lambda *_: self._rebuild())
        self.pop_combo.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        _b_export_tidy = ttk.Button(exp, text="Tidy CSV…",
                   command=self._export_tidy)
        _b_export_tidy.pack(side='left')
        tip(editor, _b_export_tidy, "One row per sample x population: the long-form table for R, pandas or any stats package.")
        _b_prism_col = ttk.Button(exp, text="Prism Column…",
                                  command=self._export_prism_column)
        _b_prism_col.pack(side='left', padx=(4, 0))
        tip(editor, _b_prism_col,
            "GraphPad Prism Column layout — one column per group, one row per "
            "replicate. Paste straight into a Column table for a t-test or "
            "ANOVA.")
        _b_prism_grp = ttk.Button(exp, text="Prism Grouped…",
                                  command=self._export_prism_grouped)
        _b_prism_grp.pack(side='left', padx=(4, 0))
        tip(editor, _b_prism_grp,
            "GraphPad Prism Grouped layout — groups as columns, populations "
            "as row blocks. Use when comparing several populations across the "
            "same groups.")
        _b_export_summary = ttk.Button(exp, text="Stats summary…",
                   command=self._export_summary)
        _b_export_summary.pack(side='left', padx=(4, 0))
        tip(editor, _b_export_summary, "The computed test results themselves: n, medians or means, the test used, and p-values — what a methods section needs.")
        _b_diff_abundance = ttk.Button(exp, text="Diff. abundance…",
                   command=self._diff_abundance)
        _b_diff_abundance.pack(side='left', padx=(4, 0))
        tip(editor, _b_diff_abundance, "Tests every population at once for a difference between groups, with multiple-testing correction, instead of one population at a time.")
        _b_compare_all = ttk.Button(exp, text="Compare all…",
                   command=self._compare_all)
        _b_compare_all.pack(side='left', padx=(4, 0))
        tip(editor, _b_compare_all, "Side-by-side comparison of every population across the groups, as one overview window.")
        _b_export_figure = ttk.Button(exp, text="Export figure…",
                   command=self._export_figure)
        _b_export_figure.pack(side='left', padx=(4, 0))
        tip(editor, _b_export_figure, "Save the current plot as a publication figure (PNG/PDF/SVG) at export resolution rather than a screenshot.")
        # On the shorter second row, at its right: at the end of the export
        # row it was cut off whenever the window was narrower than that row.
        self.bg_var, _bg_combo = background_selector(
            ctl2, var=tk.StringVar(value='White'))
        _bg_combo.pack(side='right')
        ttk.Label(ctl2, text="Fig background:").pack(side='right',
                                                     padx=(0, 2))

        # dark=False: themed in _draw after rendering (preserves prior behaviour).
        self._figframe, self._fig, self._canvas = figure_panel(
            self, figsize=(10.5, 4.8), dpi=100, dark=False)

        # Lazy-loader placeholder: a solid dark panel with a "Loading…" label
        # laid OVER the canvas while the (off-thread) collect runs. Guarantees
        # the window never shows a white slab or flashes white→dark before the
        # first real draw (an epilepsy-safety concern). Destroyed once data is in.
        # Summary text with a scrollbar: multi-group + BH-pairwise output can far
        # exceed 6 lines; without the scrollbar the overflow was silently hidden.
        _sumf = ttk.Frame(self)
        # Packed ahead of the figure: last in line, it was what a short window
        # hid.
        _sumf.pack(fill='x', side='bottom', before=self._figframe)
        _sumsb = ttk.Scrollbar(_sumf, orient='vertical')
        _sumsb.pack(side='right', fill='y')
        self._summary = tk.Text(_sumf, height=6, wrap='word',
                                yscrollcommand=_sumsb.set)
        self._summary.pack(side='left', fill='both', expand=True)
        _sumsb.config(command=self._summary.yview)

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
        self._index_populations()
        pops = sorted(set(self._pop_names.values()))
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

    def _index_populations(self):
        """Name each cross-sample population once.

        Rows are matched across samples by `__key__` (gating.population_keys:
        the gate's name, or its kind and markers and sibling rank), not by the
        displayed path, which carries an unnamed gate's coordinates -- so a
        cut nudged in one sample no longer splits one population into several,
        each missing from every other sample. A population is shown under the
        path most of its samples use; `_renamed` keeps the ones whose samples
        disagree, for the summary."""
        from collections import Counter
        paths: dict = {}
        for r in self._rows:
            key = r.get('__key__') or r['Population']
            paths.setdefault(key, Counter())[r['Population']] += 1
        names, used, renamed = {}, set(), {}
        for key in sorted(paths, key=str):
            c = paths[key]
            name = sorted(c, key=lambda p: (-c[p], p))[0]
            base, k = name, 2
            while name in used:          # two populations, one display text
                name = f"{base} [{k}]"
                k += 1
            used.add(name)
            names[key] = name
            if len(c) > 1:
                renamed[name] = sorted(c)
        self._pop_names = names
        self._renamed = renamed

    def _tidy_frame(self):
        """One row per (sample, population) for every sample in the grouping.

        A population a sample does not have gets a row too, with no values
        and Missing = True: it is MISSING, not 0 events. So does one whose
        gates cannot be computed (a gate whose region is unknown selects
        nothing, and that 0 was never measured). The differential-
        abundance table used to fill such gaps with a count of 0, so a gate
        drawn only in one group's samples read as a population that vanished
        in the other group."""
        import pandas as pd
        factor = self.factor_var.get()
        tokens = self._tokens() if factor == 'Name token' else None
        self._index_populations()        # cheap; never stale against _rows
        names = self._pop_names
        recs, left_out, groups = [], [], {}
        for r in self._rows:
            nm = r['Sample']
            grp = self.editor._sample_group_label(nm, factor, tokens)
            if grp is None:              # comp control / no matching token
                if nm not in left_out:
                    left_out.append(nm)
                continue
            groups[nm] = grp
            recs.append({
                'Sample': nm,
                'Group': grp,
                'Population': names.get(r.get('__key__') or r['Population'],
                                        r['Population']),
                'Count': r.get('Count'),
                '%Parent': r.get('%Parent'),
                '%Total': r.get('%Total'),
                # A population whose gates cannot be computed has no values
                # (gating.population_stats): missing, like one it lacks.
                'Missing': bool(r.get('__unknown__'))})
        # Every grouped sample (also one with no gates at all) owes a row to
        # every population.
        for nm in getattr(self.editor, '_sample_order', []):
            if nm in groups or nm in left_out or nm not in getattr(
                    self.editor, '_samples', {}):
                continue
            grp = self.editor._sample_group_label(nm, factor, tokens)
            if grp is None:
                left_out.append(nm)
            else:
                groups[nm] = grp
        have = {(r['Sample'], r['Population']) for r in recs}
        nan = float('nan')
        for pop in sorted(set(names.values())):
            for nm, grp in groups.items():
                if (nm, pop) not in have:
                    recs.append({'Sample': nm, 'Group': grp,
                                 'Population': pop, 'Count': nan,
                                 '%Parent': nan, '%Total': nan,
                                 'Missing': True})
        self._left_out = left_out
        return pd.DataFrame(recs, columns=['Sample', 'Group', 'Population',
                                           'Count', '%Parent', '%Total',
                                           'Missing'])

    def _missing_by_group(self, tidy, pop):
        """{group: [samples]} that do not have population `pop`."""
        sub = tidy[(tidy['Population'] == pop) & tidy['Missing'].astype(bool)]
        out: dict = {}
        for _, r in sub.iterrows():
            out.setdefault(r['Group'], []).append(r['Sample'])
        return out

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

    def _test_args(self):
        """compare_groups' test arguments from the Test box and Metric."""
        return test_kwargs(self.test_var.get(),
                           self.TRANSFORM.get(self.metric_var.get()))

    def _pair_keys(self, tidy, pop):
        """{group: [donor key per value]} aligned with _values_by_group, or
        None when Paired is off."""
        if not self.paired_var.get():
            return None
        sub = tidy[(tidy['Population'] == pop)
                   & ~tidy['Missing'].astype(bool)]
        tokens = (self._tokens() if self.factor_var.get() == 'Name token'
                  else ())
        return {g: [pair_key(n, g, tokens)
                    for n in sub[sub['Group'] == g]['Sample']]
                for g in self._ordered_groups(tidy)}

    def _sample_blocks(self, samples, groups):
        """Donor key per sample, for differential abundance's block."""
        tokens = (self._tokens() if self.factor_var.get() == 'Name token'
                  else ())
        return [pair_key(n, g, tokens) for n, g in zip(samples, groups,
                                                        strict=True)]

    def _values_by_group(self, tidy, pop, metric):
        """{group: [value per sample that HAS `pop`]}. A sample without the
        population contributes nothing (not a 0); `_missing_by_group` names
        them."""
        sub = tidy[(tidy['Population'] == pop)
                   & ~tidy['Missing'].astype(bool)]
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
        res = compare_groups(vbg, pair_keys=self._pair_keys(self._tidy, pop),
                             **self._test_args())
        self._last_res = res
        missing = self._missing_by_group(self._tidy, pop)
        self._draw(vbg, res, self._tidy, pop, metric, missing)
        self._write_summary(res, pop, metric, missing)

    def _draw(self, vbg, res, tidy, pop, metric, missing=None):
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
        n_miss = sum(len(v) for v in (missing or {}).values())
        axA.set_title(pop + (f"\n(missing in {n_miss} sample"
                             f"{'' if n_miss == 1 else 's'})" if n_miss else ''),
                      fontsize=9)
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
        _ink = plot_ink(self)         # brackets + stars legible on either canvas
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
                    color=_ink)
            ax.text((x1 + x2) / 2.0, y + h, s, ha='center', va='bottom',
                    fontsize=9, color=_ink)
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
                # No samples of this group have the population: no bar, not
                # a bar of 0.
                vals.append(float(row[metric].iloc[0]) if len(row)
                            else float('nan'))
            ax.bar(x + (gi - (n - 1) / 2.0) * w, vals, width=w, label=str(g))
        ax.set_xticks(x)
        ax.set_xticklabels([p.split('/')[-1] for p in pops], rotation=40,
                           ha='right', fontsize=7)
        ax.set_ylabel(f"mean {metric}", fontsize=9)
        ax.set_title("All populations" + (" (top 10)" if capped else ""),
                     fontsize=9)
        if len(groups) <= 8:
            ax.legend(fontsize=7, framealpha=0.85)

    def _write_summary(self, res, pop, metric, missing=None):
        lines = [f"Population: {pop}    Metric: {metric}"]
        for g, names in (missing or {}).items():
            lines.append(f"  Missing in {g} (not counted, not 0): "
                         f"{', '.join(names)}")
        paths = getattr(self, '_renamed', {}).get(pop)
        if paths:
            lines.append("  Matched across samples by gate structure; "
                         "drawn as: " + ' | '.join(paths))
        if res.get('test'):
            lines.append(f"Test: {res['test']}    p = {res.get('p'):.4g}")
        else:
            lines.append("Test: (need ≥2 non-empty groups)")
        lines.extend(power_lines(res))
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
        aside = set(getattr(self.editor, '_sample_set_aside', {}))
        other = [n for n in self._left_out if n not in aside]
        if other:
            lines.append("Left out (compensation control or no matching "
                         f"token): {', '.join(other)}")
        for line in self.editor._set_aside_lines(
                [n for n in self._left_out if n in aside]):
            lines.append(f"Set aside: {line}")
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
        path = self._ask('frequencies_tidy.csv',
                         [('CSV', '*.csv'), ('Excel workbook', '*.xlsx')])
        if path:
            # A missing population's cells say NA (R and pandas read it as
            # missing), not a blank that reads as forgotten.
            from .table_export import about_rows, is_excel_path, write_frame_csv, write_workbook
            if is_excel_path(path):
                write_workbook(path, {'Frequencies': self._tidy},
                               about=about_rows())
            else:
                write_frame_csv(path, self._tidy, na_rep='NA')
            self._done('frequencies.export', path, kind='tidy')

    def _export_prism_column(self):
        from .stats import to_prism_column
        pop, metric = self.pop_var.get(), self.metric_var.get()
        if self._tidy is None or not pop:
            return
        vbg = self._values_by_group(self._tidy, pop, metric)
        path = self._ask('prism_column.csv', [('CSV', '*.csv')])
        if path:
            from .table_export import pandas_csv_format
            sep, dec, enc = pandas_csv_format()
            to_prism_column(vbg).to_csv(path, index=False, sep=sep,
                                        decimal=dec, encoding=enc)
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
        sub = self._tidy[(self._tidy['Population'] == pop)
                         & ~self._tidy['Missing'].astype(bool)].copy()
        days = [self.editor._sample_group_label(s, 'Trial / day')
                for s in sub['Sample']]
        conds = [self.editor._sample_group_label(s, 'Name token', tokens)
                 for s in sub['Sample']]
        sub['Day'] = days
        sub['Cond'] = conds
        # Same exclusions as the comparison itself (comp controls, samples
        # matching no token): None in either key means left out.
        sub = sub[[d is not None and c is not None
                   for d, c in zip(days, conds, strict=True)]]
        path = self._ask('prism_grouped.csv', [('CSV', '*.csv')])
        if path:
            from .table_export import pandas_csv_format
            sep, dec, enc = pandas_csv_format()
            to_prism_grouped(sub, 'Day', 'Cond', metric).to_csv(
                path, sep=sep, decimal=dec, encoding=enc)
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
        report_saved(self, path, "Exported")

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
        # A population a sample does not have stays NaN -- differential_
        # abundance fits it on the samples that have it. fill_value=0 made a
        # gate drawn in one group's samples only read as a population that
        # vanished from the other group.
        wide = tidy.pivot_table(index='Population', columns='Sample',
                                values='Count', aggfunc='first',
                                dropna=False)
        samples = list(wide.columns)
        grp = [tidy.loc[tidy['Sample'] == s, 'Group'].iloc[0] for s in samples]
        import numpy as _np
        # per-sample totals (NaN = population missing in that sample)
        col_sums = _np.nansum(wide.to_numpy(dtype=float), axis=0)
        lib = []
        for k, s in enumerate(samples):
            csum = int(col_sums[k])
            ev = (len(self.editor._samples[s].data)
                  if s in self.editor._samples else csum)
            # ev is the sample's TRUE total event count (the GLM offset); csum
            # sums Count over hierarchical/overlapping populations (root+nested),
            # so it double-counts and would understate the displayed proportions.
            # Keep csum only as the not-loaded fallback already baked into `ev`.
            lib.append(max(ev, 1))
        block = (self._sample_blocks(samples, grp)
                 if self.paired_var.get() else None)
        try:
            rows = differential_abundance(wide, grp, lib_sizes=lib,
                                          block=block)
        except Exception as exc:
            show_error(self, "Differential abundance",
                       "The differential abundance test could not run.",
                       exc)
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
        pops = sorted(set(tidy['Population']))
        vbf = {p: self._values_by_group(tidy, p, metric) for p in pops}
        pk = ({p: self._pair_keys(tidy, p) for p in pops}
              if self.paired_var.get() else None)
        res = compare_all_features(vbf, pair_keys=pk, **self._test_args())
        try:
            self.editor._audit('compare_all_populations', metric=metric,
                               n_populations=len(res), groups=','.join(groups))
        except Exception:
            pass
        CompareAllWindow(self, res, groups, metric)
