"""Per-marker expression comparison across populations/groups.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from .async_task import BusyStrip
from .errors import note
from .group_test_choice import add_test_controls, pair_key, power_lines, test_kwargs
from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    plot_ink,
    savefig_background,
)
from .tool_window import background_selector, figure_panel
from .ui_notice import report_saved
from .ui_tips import tip


class MarkerExpressionWindow(tk.Toplevel):
    """Marker-expression distributions by group — violin or ridgeline.

    Pools every loaded sample's per-cell values (ticked in the tree or not)
    for a chosen marker (resolving the marker across fluors by antibody
    label) inside a chosen population
    (or all events), groups the samples by a factor (trial/day,
    comp-vs-samples, or a name token), and draws a violin or ridgeline plot
    per group. Significance comes from a per-SAMPLE-median comparison (so the
    test treats each sample as a replicate, not each cell), which is also what
    the **GraphPad Prism** Column export contains."""

    FACTORS = ('Trial / day', 'Comp vs Samples', 'Name token')
    PER_GROUP_CAP = 40_000
    ALL_EVENTS = 'All events'

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Marker expression by group")
        self.geometry("1000x720")
        self._percell = {}
        self._medians = {}           # per-sample LINEAR medians (tested/exported)
        self._disp_medians = {}      # the same, display scale (bracket placement)
        self._groups = []
        self._left_out = []          # samples the grouping leaves out
        self._omitted = []           # samples on an unknown scale
        self._guessed = []           # samples whose scale is a guess
        self._no_pop = []            # samples without the chosen population

        chans = [c for c in editor._channels]
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Marker:").pack(side='left')
        self.marker_var = tk.StringVar()
        disp = [editor._fmt_channel(c) for c in chans]
        _mk = ttk.Combobox(ctl, textvariable=self.marker_var, width=20,
                           state='readonly', values=disp)
        _mk.pack(side='left', padx=(2, 8))
        tip(editor, _mk,
            "Which channel's expression to plot. Compensated and transformed "
            "exactly as in the main plot, so the shapes match what you gated "
            "on.")
        fluor = next((editor._fmt_channel(c) for c in chans
                      if c in (self._first_fluor() or [])), disp[0] if disp
                     else '')
        self.marker_var.set(fluor)
        ttk.Label(ctl, text="Population:").pack(side='left')
        self.pop_var = tk.StringVar(value=self.ALL_EVENTS)
        self._pop_combo = ttk.Combobox(ctl, textvariable=self.pop_var,
                                       width=22, state='readonly',
                                       values=self._population_choices())
        self._pop_combo.pack(side='left', padx=(2, 8))
        tip(editor, self._pop_combo,
            "Which events to compare. 'All events' takes every event of each "
            "sample, debris included; a gated population takes only the "
            "events inside that gate and its parents. Populations are matched "
            "across samples by their path in the gate tree.")
        ttk.Label(ctl, text="Group by:").pack(side='left')
        self.factor_var = tk.StringVar(value='Trial / day')
        _fx = ttk.Combobox(ctl, textvariable=self.factor_var, width=16,
                           state='readonly', values=self.FACTORS)
        _fx.pack(side='left', padx=(2, 4))
        tip(editor, _fx,
            "How samples are split into groups — by trial/day, or by matching "
            "the tokens to the right against sample names. This defines what "
            "the test compares.")
        ttk.Label(ctl, text="Tokens:").pack(side='left')
        self.tokens_var = tk.StringVar(value='Stim, Ctrl')
        _tk = ttk.Entry(ctl, textvariable=self.tokens_var, width=14)
        _tk.pack(side='left', padx=(2, 8))
        tip(editor, _tk,
            "Comma-separated substrings matched against sample names to form "
            "groups, e.g. 'Stim, Ctrl'. A sample matching none is left out.")
        # A second row: on one row the controls are wider than the window,
        # and the packer dropped Update and the options at its end.
        ctl2 = ttk.Frame(self, padding=(6, 0, 6, 6))
        ctl2.pack(fill='x', side='top')
        ttk.Label(ctl2, text="Plot:").pack(side='left')
        self.plot_var = tk.StringVar(value='Violin')
        _pl = ttk.Combobox(ctl2, textvariable=self.plot_var, width=9,
                           state='readonly', values=['Violin', 'Ridgeline'])
        _pl.pack(side='left', padx=(2, 8))
        tip(editor, _pl,
            "Violin shows each group's full distribution side by side; "
            "Ridgeline stacks them as overlapping curves, which reads better "
            "for many groups or a time course.")
        self.test_var, self.paired_var = add_test_controls(editor, ctl2)
        _up = ttk.Button(ctl2, text="Update", command=self._rebuild)
        _up.pack(side='left')
        tip(editor, _up,
            "Recompute from the current gates of the chosen population, the "
            "marker and the grouping.")

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        _e1 = ttk.Button(exp, text="Prism Column (linear medians)…",
                         command=self._export_prism)
        _e1.pack(side='left')
        tip(editor, _e1,
            "GraphPad Prism Column layout of the per-sample MEDIAN intensity "
            "on the compensated linear scale (as the Statistics window "
            "reports it) — one column per group, one row per sample. The "
            "median, not the mean, because intensity distributions are "
            "skewed.")
        _e2 = ttk.Button(exp, text="Stats summary…",
                         command=self._export_summary)
        _e2.pack(side='left', padx=(4, 0))
        tip(editor, _e2,
            "The test results themselves: n, centres, the test used and "
            "p-values — what a methods section needs.")
        _e3 = ttk.Button(exp, text="Export figure…", command=self._export_figure)
        _e3.pack(side='left', padx=(4, 0))
        tip(editor, _e3,
            "Save this plot as a publication figure (PNG/PDF/SVG) at export "
            "resolution rather than a screenshot.")
        self.bg_var, _bg_combo = background_selector(exp)
        _bg_combo.pack(side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        # The summary before the figure: packed last, it was what a short
        # window hid.
        self._summary = tk.Text(self, height=5, wrap='word')
        self._summary.pack(fill='x', side='bottom')
        # The reading and the test run off the window thread, shown here.
        self._busy = BusyStrip(self, padding=(6, 2))
        self._busy.pack(fill='x', side='bottom')
        # dark=False: themed in _draw after rendering (preserves prior behaviour).
        _frame, self._fig, self._canvas = figure_panel(
            self, figsize=(9.5, 5.0), dpi=100, dark=False)
        self._rebuild()

    def _first_fluor(self):
        for n in self.editor._sample_order:
            s = self.editor._samples.get(n)
            if s is not None:
                return getattr(s, 'fluor_channels', []) or []
        return []

    def _samples(self):
        # Every loaded sample, ticked or not, as in Frequencies and Group
        # comparison: the tree's tick only controls the plot.
        return [n for n in self.editor._sample_order
                if n in self.editor._samples]

    def _tokens(self):
        return [t for t in self.tokens_var.get().split(',') if t.strip()]

    def _population_gids(self, name):
        """{population path: gate id} for one sample, in gate-tree order.
        Group containers are skipped: they never filter events."""
        ed = self.editor
        gates = ed._sample_gates.get(name, {})
        # Bounds in intensity, as the gate list shows them. Samples sharing
        # a gate's coordinates (and so their transforms) share its path.
        transforms = ed._display_transforms(name)
        out = {}
        for gid in ed._sample_gate_order.get(name, list(gates)):
            g = gates.get(gid)
            if g is None or g.get('kind') == 'group':
                continue
            out.setdefault(ed._population_path(gates, gid,
                                               transforms=transforms), gid)
        return out

    def _population_choices(self):
        seen = {}
        for n in self.editor._sample_order:
            if n in self.editor._samples:
                for path in self._population_gids(n):
                    seen.setdefault(path, None)
        return [self.ALL_EVENTS] + list(seen)

    def _plan(self):
        """What `_compute` reads, gathered on the window thread: each
        sample's frame, column, group and population (its gate id), and the
        samples left out on the way."""
        ch = self.editor._resolve_channel(self.marker_var.get())
        factor = self.factor_var.get()
        tokens = self._tokens() if factor == 'Name token' else None
        pop = self.pop_var.get() or self.ALL_EVENTS
        order = {n: i for i, n in enumerate(self.editor._sample_order)}
        items, left_out, no_pop = [], [], []
        for n in sorted(self._samples(), key=lambda x: order.get(x, 1e9)):
            s = self.editor._samples[n]
            col = self.editor._marker_column_for(s, ch)
            if not col:
                continue
            g = self.editor._sample_group_label(n, factor, tokens)
            if g is None:                # comp control / no matching token
                left_out.append(n)
                continue
            gid = None
            if pop != self.ALL_EVENTS:
                gid = self._population_gids(n).get(pop)
                if gid is None:
                    no_pop.append(n)
                    continue
            items.append((n, s, col, g, gid,
                          dict(self.editor._sample_gates.get(n, {}))))
        return {'items': items, 'left_out': left_out, 'no_pop': no_pop,
                'test': test_kwargs(self.test_var.get(), 'asinh'),
                'paired': bool(self.paired_var.get()),
                'tokens': (tokens or ())}

    def _compute(self, plan, job=None):
        """The per-cell values, per-sample medians and the group test for
        `plan`; reads the samples only, so it runs off the window thread."""
        import numpy as _np

        from .pipeline import (
            UnknownScaleError,
            cumulative_gate_mask,
            is_guessed_spec,
            linear_values,
        )
        from .stats import compare_groups
        percell, medians, groups, keys = {}, {}, [], {}
        disp_medians, omitted, guessed = {}, [], []
        for i, (n, s, col, g, gid, gates) in enumerate(plan['items']):
            if job is not None:
                job.progress(i, len(plan['items']), f"Reading {n}…")
            vals = _np.asarray(s.data[col].values, dtype=float)
            if gid is not None:
                # The window used to read every event whatever the gates: with
                # 100 debris events beside 100 cells, the per-sample medians
                # (and the Prism export) came out at half the gated values.
                vals = vals[_np.asarray(
                    cumulative_gate_mask(gates, gid, s.data), dtype=bool)]
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                continue
            # The distributions are drawn on the display scale, like the main
            # plot; the per-sample median that is tested, summarised and
            # exported is the compensated LINEAR one, the number the
            # Statistics window reports for the same events. A sample whose
            # column is on an unknown scale has neither: it is left out.
            try:
                lin = linear_values(s, col, vals)
            except UnknownScaleError:
                omitted.append(n)
                continue
            if is_guessed_spec((getattr(s, 'data_transforms', None)
                                or {}).get(col)):
                guessed.append(n)            # its median is a guess: say so
            if g not in percell:
                percell[g] = []
                medians[g] = []
                disp_medians[g] = []
                keys[g] = []
                groups.append(g)
            percell[g].append(vals)
            medians[g].append(float(_np.median(lin[_np.isfinite(lin)])))
            keys[g].append(pair_key(n, g, plan.get('tokens', ())))
            disp_medians[g].append(float(_np.median(vals)))
        rng = _np.random.default_rng(0)
        pooled = {}
        for g in groups:
            allv = _np.concatenate(percell[g])
            if allv.size > self.PER_GROUP_CAP:
                allv = allv[rng.choice(allv.size, self.PER_GROUP_CAP,
                                       replace=False)]
            pooled[g] = allv
        res = (compare_groups(medians,
                              pair_keys=keys if plan.get('paired') else None,
                              **plan['test'])
               if groups else {})
        return {'percell': pooled, 'medians': medians, 'groups': groups,
                'disp_medians': disp_medians, 'omitted': omitted,
                'guessed': guessed, 'left_out': plan['left_out'],
                'no_pop': plan['no_pop'], 'res': res}

    def _take(self, out):
        self._percell, self._medians = out['percell'], out['medians']
        self._groups = out['groups']
        self._disp_medians = out['disp_medians']
        self._omitted, self._guessed = out['omitted'], out['guessed']
        self._left_out, self._no_pop = out['left_out'], out['no_pop']

    def _collect(self):
        """`_plan`, `_compute` and keep the result, on this thread."""
        out = self._compute(self._plan())
        self._take(out)
        return out

    def _show(self, out):
        self._take(out)
        if not self._groups:
            self._fig.clear()
            if _dialog_dark_on(self):
                _theme_figure_dark(self._fig)
            self._canvas.draw()
            self._write_summary({})          # still say what was left out
            return
        self._draw(out['res'])
        self._write_summary(out['res'])

    def _rebuild(self):
        """Recompute and redraw. The reading and the test run off the window
        thread (every event of every sample, through its gates), shown with
        progress and Cancel; a newer Update replaces a running one."""
        # Gates may have changed since the window opened.
        choices = self._population_choices()
        self._pop_combo['values'] = choices
        if self.pop_var.get() not in choices:
            self.pop_var.set(self.ALL_EVENTS)
        plan = self._plan()

        def failed(exc):
            self._write_text(f"Failed: {type(exc).__name__}: {exc}")

        self._busy.run(lambda job: self._compute(plan, job),
                       on_done=self._show, on_error=failed,
                       text="Reading the samples…")

    def _draw(self, res):
        import numpy as _np
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        groups = self._groups
        marker = self.marker_var.get()
        axis_label = f"{marker} (display scale)"
        if self.plot_var.get() == 'Ridgeline':
            self._draw_ridgeline(ax, groups, axis_label)
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
            ax.set_ylabel(axis_label, fontsize=9)
            # Brackets sit on THIS axis, so they are placed by the display-
            # scale medians; the stars come from the linear-median test.
            self._draw_sig(ax, [_np.asarray(self._disp_medians[g])
                                for g in groups], groups, res)
        ax.set_title(f"{marker} by {self.factor_var.get()}", fontsize=9)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
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
            ax.plot(x, y, lw=0.8, color=plot_ink(self), alpha=0.5)
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

    def _write_summary(self, res):
        pop = self.pop_var.get() or self.ALL_EVENTS
        lines = [f"Marker: {self.marker_var.get()}    Population: {pop}"
                 + (" (ungated)" if pop == self.ALL_EVENTS else "")
                 + "    (significance from per-sample medians on the "
                   "compensated linear scale, n = samples; plot on the "
                   "display scale)"]
        if res.get('test'):
            lines.append(f"Test: {res['test']}    p = {res.get('p'):.4g}")
        else:
            lines.append("Test: (need ≥2 groups with samples)")
        lines.extend(power_lines(res))
        for g, st in res.get('groups', {}).items():
            lines.append(f"  {g}: n={st['n']}  median-of-medians="
                         f"{st['median']:.3g}")
        aside = set(getattr(self.editor, '_sample_set_aside', {}))
        other = [n for n in self._left_out if n not in aside]
        if other:
            lines.append("Left out (compensation control or no matching "
                         f"token): {', '.join(other)}")
        for line in self.editor._set_aside_lines(
                [n for n in self._left_out if n in aside]):
            lines.append(f"Set aside: {line}")
        if self._no_pop:
            lines.append(f"No '{pop}' population: {', '.join(self._no_pop)}")
        if getattr(self, '_omitted', None):
            lines.append("[!] Left out, scale unknown (older OpenFlo CSV): "
                         + ', '.join(self._omitted))
        if getattr(self, '_guessed', None):
            lines.append("[!] Linear medians guessed (no transform record, "
                         "logicle assumed): " + ', '.join(self._guessed))
        self._write_text("\n".join(lines))

    def _write_text(self, text):
        self._summary.configure(state='normal')
        self._summary.delete('1.0', 'end')
        self._summary.insert('1.0', text)
        self._summary.configure(state='disabled')

    def _export_prism(self):
        from .stats import to_prism_column
        if not self._medians:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            initialfile='expression_linear_medians_prism.csv',
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
                               marker=self.marker_var.get(),
                               samples=self._samples())
        except Exception as exc:
            note("recording the export in the audit trail", exc)
        report_saved(self, path, "Exported")
