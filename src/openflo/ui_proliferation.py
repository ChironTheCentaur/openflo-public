"""Analyze, Proliferation: dye-dilution generations and the standard
indices (Complaints 21).

The active sample's dye channel, within a population, is fitted as
generations one halving apart (proliferation.fit_generations). The
undivided peak is found automatically, or taken from an unstimulated
control sample's median in the same population.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np

from .dialog_inputs import read_int
from .tool_window import (
    ask_csv_path,
    background_selector,
    export_figure,
    figure_panel,
)
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import auto_tip

ALL_EVENTS = "All events"
AUTO_PEAK = "Find automatically"


def population_values(sample, gates, channel, population):
    """Linear ``channel`` values of ``sample`` in ``population`` (a gate
    name, or ALL_EVENTS). None when the sample has no such gate."""
    from .pipeline import cumulative_gate_mask, linear_values
    v = np.asarray(linear_values(sample, channel), dtype=float)
    if population == ALL_EVENTS:
        return v
    gid = next((g for g, d in gates.items() if d.get('name') == population),
               None)
    if gid is None:
        return None
    return v[np.asarray(cumulative_gate_mask(gates, gid, sample.data),
                        dtype=bool)]


def no_sample(win, text):
    """Fill ``win`` with ``text`` and a Close button: the window was opened
    with no sample to work on."""
    ttk.Label(win, text=text, padding=20).pack(side='top')
    ttk.Button(win, text="Close", command=win.destroy).pack(side='bottom',
                                                           pady=(0, 10))


class ProliferationWindow(tk.Toplevel):
    """[Channel] [Population] [Undivided peak] [Generations] [Fit] /
    histogram with generation curves / generations table and indices."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.sample = editor._active_sample
        self.title(f"Proliferation — {self.sample}")
        self.minsize(720, 560)
        self.result: dict[str, Any] | None = None
        self.fig: Any = None
        s = editor._samples.get(self.sample)
        if s is None:
            no_sample(self, "Load and select a sample to fit.")
            return
        chans = [c for c in (getattr(s, 'fluor_channels', None)
                             or editor._channels) if c in s.data.columns]
        gates = editor._sample_gates.get(self.sample, {})
        pops = [ALL_EVENTS] + [g['name'] for g in gates.values()
                               if g.get('name') and g.get('kind') != 'autoclean']
        others = [n for n in editor._sample_order
                  if n in editor._samples and n != self.sample]

        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(side='top', fill='x')
        self.channel_var = tk.StringVar(value=chans[0] if chans else '')
        self.population_var = tk.StringVar(value=pops[-1] if len(pops) > 1
                                           else ALL_EVENTS)
        self.control_var = tk.StringVar(value=AUTO_PEAK)
        self.gen_var = tk.IntVar(value=8)
        rows = (("Dye channel:", ttk.Combobox(
                    top, textvariable=self.channel_var, values=chans,
                    state='readonly', width=22),
                 "The dye that halves with each division."),
                ("Population:", ttk.Combobox(
                    top, textvariable=self.population_var, values=pops,
                    state='readonly', width=22),
                 "Fit the cells of this gate (live singlets, say)."),
                ("Undivided peak:", ttk.Combobox(
                    top, textvariable=self.control_var,
                    values=[AUTO_PEAK] + others, state='readonly', width=22),
                 "The generation-0 position: the brightest clear peak, or "
                 "an unstimulated control's median in the same population."),
                ("Generations:", ttk.Spinbox(
                    top, from_=1, to=12, textvariable=self.gen_var, width=5),
                 "The most divisions to model."))
        for i, (label, widget, tip) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=i // 2, column=(i % 2) * 2,
                                            sticky='e', padx=(0, 4), pady=2)
            widget.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky='w',
                        padx=(0, 12), pady=2)
            auto_tip(widget, tip)
        fit = ttk.Button(top, text="Fit", command=self.fit, default='active')
        fit.grid(row=0, column=4, rowspan=2, padx=(6, 0))
        auto_tip(fit, "Fit the generations and compute the indices.")

        bar = ttk.Frame(self, padding=(12, 4, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        self.bg_var, bg = background_selector(bar)
        b = ttk.Button(bar, text="Export figure…", command=self._export_png)
        b.pack(side='left')
        auto_tip(b, "Save the histogram and fit as PNG, SVG or PDF.")
        bg.pack(side='left', padx=(6, 12))
        c = ttk.Button(bar, text="Export CSV…", command=self._export_csv)
        c.pack(side='left')
        auto_tip(c, "Cells per generation, precursor frequencies and the "
                    "indices.")
        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, wraplength=680,
                  justify='left', padding=(12, 2)).pack(side='bottom',
                                                        fill='x')
        self.table = ttk.Treeview(self, columns=('cells', 'pct', 'prec'),
                                  show='tree headings', height=6)
        self.table.heading('#0', text='Generation')
        self.table.heading('cells', text='Cells')
        self.table.heading('pct', text='% of cells')
        self.table.heading('prec', text='Precursor %')
        for col in ('#0', 'cells', 'pct', 'prec'):
            self.table.column(col, width=110, anchor='e')
        self.table.pack(side='bottom', fill='x', padx=12)
        _frame, self.fig, self.canvas = figure_panel(self, figsize=(7, 3.6))
        if chans:
            self.fit()

    def _values(self, name):
        e = self.editor
        return population_values(e._samples[name],
                                 e._sample_gates.get(name, {}),
                                 self.channel_var.get(),
                                 self.population_var.get())

    def fit(self):
        from .proliferation import fit_generations
        v = self._values(self.sample)
        if v is None:
            self.status_var.set(f"{self.sample} has no population "
                                f"'{self.population_var.get()}'.")
            return
        gens = read_int(self.gen_var, "Generations", lo=1, hi=12, parent=self)
        if gens is None:
            return
        undivided = None
        ctrl = self.control_var.get()
        if ctrl != AUTO_PEAK:
            cv = self._values(ctrl)
            cv = cv[np.isfinite(cv) & (cv > 0)] if cv is not None else None
            if cv is None or not cv.size:
                self.status_var.set(f"{ctrl} has no events in "
                                    f"'{self.population_var.get()}'.")
                return
            undivided = float(np.median(cv))
        self.values = v
        self.result = fit_generations(v, undivided=undivided,
                                      max_generations=gens)
        self.draw()

    def draw(self):
        from .theme import _dialog_dark_on, _theme_figure_dark
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for row in self.table.get_children():
            self.table.delete(row)
        res = self.result
        if not res or not res.get('ok'):
            self.status_var.set("Too few positive events to fit (50 or more "
                                "are needed). The indices read NA.")
            self.canvas.draw_idle()
            return
        x = np.log2(self.values[np.isfinite(self.values) & (self.values > 0)])
        hist, edges = np.histogram(x, bins=160)
        width = edges[1] - edges[0]
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centres, hist, width=width, color='0.7', linewidth=0)
        grid = np.linspace(edges[0], edges[-1], 600)
        total = np.zeros_like(grid)
        for k, (m, w) in enumerate(zip(res['means'], res['weights'],
                                       strict=True)):
            curve = (w * x.size * width / (res['sd'] * np.sqrt(2 * np.pi))
                     * np.exp(-0.5 * ((grid - np.log2(m)) / res['sd']) ** 2))
            total += curve
            if w > 0.002:
                ax.plot(grid, curve, lw=1, color=f"C{k % 10}")
        ax.plot(grid, total, lw=1.4, color='black', label='Fit')
        ax.set_xlabel(f"{self.editor._fmt_channel(self.channel_var.get())} "
                      "(log2)")
        ax.set_ylabel('cells')
        ax.legend(fontsize=8)
        self.fig.tight_layout()
        if _dialog_dark_on(self):
            _theme_figure_dark(self.fig)
        self.canvas.draw_idle()
        n = max(sum(res['counts']), 1)
        for k, (c, p) in enumerate(zip(res['counts'],
                                       res['precursor_frequency'],
                                       strict=True)):
            self.table.insert('', 'end', text=str(k),
                              values=(f"{c:,}", f"{100 * c / n:.1f}",
                                      f"{100 * p:.1f}"))
        self.status_var.set(self.summary())

    def summary(self):
        res = self.result or {}

        def f(key, fmt):
            v = res.get(key)
            return 'NA' if v is None or not np.isfinite(v) else fmt.format(v)
        return (f"Divided {f('pct_divided', '{:.1f}%')}   ·   "
                f"Division index {f('division_index', '{:.2f}')}   ·   "
                f"Proliferation index {f('proliferation_index', '{:.2f}')}   ·"
                f"   Expansion index {f('expansion_index', '{:.2f}')}   ·   "
                f"Spacing {f('spacing', '{:.2f}')} log2 per division")

    def rows(self):
        """CSV rows: one per generation, then the indices."""
        res = self.result or {}
        if not res.get('ok'):
            return [['note', 'too few events; indices NA']]
        out: list[list[Any]] = [['generation', 'cells',
                                 'precursor_frequency']]
        for k, (c, p) in enumerate(zip(res['counts'],
                                       res['precursor_frequency'],
                                       strict=True)):
            out.append([k, c, f"{p:.6g}"])
        out.append([])
        for key in ('pct_divided', 'division_index', 'proliferation_index',
                    'expansion_index', 'undivided', 'spacing'):
            v = res.get(key)
            out.append([key, 'NA' if v is None or not np.isfinite(v)
                        else f"{v:.6g}"])
        return out

    def _export_csv(self):
        path = ask_csv_path(self, initialfile=f"{self.sample}_proliferation"
                                             ".csv")
        if not path:
            return
        import csv
        try:
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerows(self.rows())
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Proliferation", "The table could not be saved.",
                       exc, path)
            return
        report_saved(self, path)

    def _export_png(self):
        if self.fig is None:
            return
        export_figure(self, self.fig, background=self.bg_var.get(),
                      initialfile=f"{self.sample}_proliferation.png")
