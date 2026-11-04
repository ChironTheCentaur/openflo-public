"""Population-frequency comparison across samples/groups (plots + table).

The "Frequencies by group" tab of the Statistics window (ui_statistics):
a panel, not a window of its own, since the table and the group comparison
work on the same Count / %Parent / %Total numbers.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .errors import note
from .export_audit import record_export
from .group_test_choice import add_test_controls, pair_key, power_lines, test_kwargs
from .plotmath import draw_significance
from .theme import (
    dialog_dark_on,
    match_dialog,
    plot_ink,
    savefig_background,
)
from .tool_window import background_selector, figure_panel, set_readonly_text
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import tip


def _fmt_3g_or_na(x):
    """A summary number to 3 significant figures, or NA when it cannot be
    computed (the SD of a single sample, a group with no values)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 'NA'
    return f"{x:.3g}" if x == x and abs(x) != float('inf') else 'NA'


class FrequencyPanel(ttk.Frame):
    """Population-frequency & group-comparison view (a tab of Statistics).

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

    def __init__(self, master, editor):
        super().__init__(master)
        self.editor = editor
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
        self.metric_combo = _mc
        tip(editor, _mc,
            "What the percentage is OF: %Parent divides by the immediate "
            "parent gate, %Total by all events in the sample. They differ "
            "sharply for a deep gate, so state which one a figure uses.")
        # Update ends the first row; grouping takes the second row and the
        # test and figure background the third. Each row stays narrower than
        # the Statistics window this panel sits in (on two rows they were
        # wider, and the packer squeezed Fig background away).
        _ub = ttk.Button(ctl, text="Update", command=self._update)
        self.update_button = _ub
        _ub.pack(side='left')
        tip(editor, _ub,
            "Recompute the frequencies from the current gates and settings. "
            "The population, metric, grouping and test boxes already "
            "refresh on change; use this after editing gates in the editor.")
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
        ctl3 = ttk.Frame(self, padding=(6, 0, 6, 6))
        ctl3.pack(fill='x', side='top')
        self.test_var, self.paired_var = add_test_controls(
            editor, ctl3, on_change=self._rebuild)
        # Rebuild the tidy frame whenever the GROUPING changes — not just on the
        # population combo / Update — so Diff. abundance / Compare all can't run
        # on a stale grouping (typed tokens commit on <Return> or focus-out,
        # which also fires just before a clicked export button's command).
        _fc.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())
        _te.bind('<Return>', lambda *_: self._rebuild())
        _te.bind('<FocusOut>', lambda *_: self._rebuild())
        self.pop_combo.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())
        # The Metric box redraws too: unbound, picking %Total or Count left
        # the plot and summary on the previous metric until Update.
        _mc.bind('<<ComboboxSelected>>', lambda *_: self._rebuild())

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
        # The two all-population analyses sit at the right of the grouping
        # row, which they depend on; the exports keep a row of their own.
        _b_diff_abundance = ttk.Button(ctl2, text="Diff. abundance…",
                   command=self._diff_abundance)
        tip(editor, _b_diff_abundance, "Tests every population at once for a difference between groups, with multiple-testing correction, instead of one population at a time.")
        _b_compare_all = ttk.Button(ctl2, text="Compare all…",
                   command=self._compare_all)
        _b_compare_all.pack(side='right')
        _b_diff_abundance.pack(side='right', padx=(0, 4))
        tip(editor, _b_compare_all, "Side-by-side comparison of every population across the groups, as one overview window.")
        _b_export_figure = ttk.Button(exp, text="Export figure…",
                   command=self._export_figure)
        _b_export_figure.pack(side='left', padx=(4, 0))
        tip(editor, _b_export_figure, "Save the current plot as a publication figure (PNG/PDF/SVG) at export resolution rather than a screenshot.")
        # Close on this tab too: the Table tab's Close is out of sight here.
        ttk.Button(exp, text="Close",
                   command=lambda: self.winfo_toplevel().destroy()).pack(
            side='right')
        # On the short test row, at its right: at the end of the export row
        # it was cut off whenever the window was narrower than that row.
        self.bg_var, _bg_combo = background_selector(ctl3)
        _bg_combo.pack(side='right')
        ttk.Label(ctl3, text="Fig background:").pack(side='right',
                                                     padx=(0, 2))

        # dark=False: themed in _draw after rendering. figure_panel still paints
        # the EMPTY figure dark in a dark UI, so the tab never shows a white
        # slab before the first draw (an epilepsy-safety concern).
        self._figframe, self._fig, self._canvas = figure_panel(
            self, figsize=(9.6, 4.2), dpi=100, dark=False)

        # Summary text with a scrollbar: multi-group + BH-pairwise output can far
        # exceed 6 lines; without the scrollbar the overflow is silently hidden.
        _sumf = ttk.Frame(self)
        # Packed ahead of the figure: last in line, it would be what a short
        # window hides.
        _sumf.pack(fill='x', side='bottom', before=self._figframe)
        _sumsb = ttk.Scrollbar(_sumf, orient='vertical')
        _sumsb.pack(side='right', fill='y')
        self._summary = tk.Text(_sumf, height=6, wrap='word',
                                yscrollcommand=_sumsb.set)
        self._summary.pack(side='left', fill='both', expand=True)
        _sumsb.config(command=self._summary.yview)

        if self.editor is not None:
            try:
                self.editor.status_var.set("Computing frequencies…")
            except (tk.TclError, AttributeError):
                pass
        # Closed mid-collect, the collect's result is dropped (the panel is
        # gone), so its message would stay in the editor's status bar.
        self.bind('<Destroy>', lambda e: (
            self._clear_status() if e.widget is self else None), add='+')
        self.after(30, lambda: self._collect(self._after_collect))

    def _update(self):
        """Update: collect the rows again from the current gates, then redraw.
        Regrouping alone (what the boxes do) keeps the numbers collected at
        open, so after a gate edit the plot, tests and exports were stale."""
        try:
            self.editor.status_var.set("Computing frequencies…")
        except (tk.TclError, AttributeError):
            pass
        self._collect(self._after_collect)

    def _after_collect(self):
        self._clear_status()
        self._index_populations()
        pops = sorted(set(self._pop_names.values()))
        self.pop_combo['values'] = pops
        if self.pop_var.get() not in pops:
            # Keep the user's pick if it remains; with no population left
            # (every gate deleted), none is picked.
            self.pop_var.set(pops[0] if pops else '')
        self._rebuild()

    def _clear_status(self):
        """Take down the editor's "Computing frequencies…" once the rows are
        in (or the collect failed): left up, it reads as a window still busy.
        Only that message is cleared, so a newer one stays."""
        try:
            if self.editor.status_var.get() == "Computing frequencies…":
                self.editor.status_var.set("")
        except (tk.TclError, AttributeError):
            pass

    # ── data ─────────────────────────────────────────────────────────────
    def _collect(self, then=None):
        """Collect frequency rows off the Tk thread (snapshot-then-background)
        so opening the window over many samples/gates doesn't freeze it; ``then``
        runs on the Tk thread once the rows are in."""
        from .async_task import run_async
        want = {'Count', '%Parent', '%Total'}
        # Only the newest collect may land: an Update pressed while the
        # opening collect still runs must not be undone by it.
        self._request = req = getattr(self, '_request', 0) + 1
        try:
            snap = self.editor._stats_snapshot(want)
        except Exception as exc:  # noqa: BLE001 - a failed snapshot is printed and the window shows an empty table
            print(f"[frequencies] snapshot failed: {exc}", flush=True)
            self._rows = []
            if then:
                then()
            return

        def _done(res):
            if req != self._request:
                return
            self._rows = res[0]
            if then:
                then()

        def _err(exc):
            if req != self._request:
                return
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
        cut nudged in one sample does not split one population into several,
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
        abundance table must not fill such gaps with a count of 0: a gate
        drawn only in one group's samples would read as a population that
        vanished in the other group."""
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
            if dialog_dark_on(self):
                from .theme import THEMES
                self._fig.set_facecolor(THEMES['midnight']['plot_bg'])
            self._canvas.draw()
            # The summary goes too: left up, it described a plot no longer
            # shown, and Stats summary… saved it.
            self._last_res = None
            self._write_text("No samples in this grouping." if pop
                             else "No populations to compare.")
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
        if dialog_dark_on(self):
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
        draw_significance(axA, data, groups, res, plot_ink(self), gap=0.08,
                          fit_top=True)

        # Panel B — all-population overview: mean metric per group (top pops).
        self._draw_overview(axB, tidy, metric, groups)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
            pass
        match_dialog(fig, self)
        self._canvas.draw()

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
            f3 = _fmt_3g_or_na
            lines.append(f"  {g}: n={st['n']}  mean={f3(st['mean'])}  "
                         f"median={f3(st['median'])}  sd={f3(st['sd'])}")
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
        self._write_text("\n".join(lines))

    def _write_text(self, text):
        set_readonly_text(self._summary, text)

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
            # A boolean-list row filter reads as Series | DataFrame | ndarray
            # in the stubs; sub is the filtered tidy DataFrame.
            to_prism_grouped(sub, 'Day', 'Cond', metric).to_csv(  # pyright: ignore[reportArgumentType]
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
        if self._tidy is not None and 'Sample' in self._tidy:
            details.setdefault('samples', [str(n) for n in dict.fromkeys(
                self._tidy['Sample'])])
        record_export(self.editor, action, path, **details)
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
        # abundance fits it on the samples that have it. fill_value=0 would
        # make a gate drawn in one group's samples only read as a population
        # that vanished from the other group.
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
        except Exception as exc:  # noqa: BLE001 - a test that cannot run is shown in an error dialog
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
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("write the audit entry", exc)
        from .ui_diff import DiffAbundanceWindow
        DiffAbundanceWindow(self, rows, disp, samples=samples)

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
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("write the audit entry", exc)
        CompareAllWindow(self, res, groups, metric,
                         samples=[str(n) for n in dict.fromkeys(
                             tidy['Sample'])])
