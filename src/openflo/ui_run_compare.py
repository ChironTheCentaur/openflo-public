"""Run history: a run template's runs compared within one run and across
every run (``run_compare`` does the work).

The plot shows one population's statistic per experiment date, a line per
group (or trial, or sample), with a dashed line where the template
changed. Below it, the across-run test (each run day a replicate, paired
by day) and the within-run test of the chosen run (each sample a
replicate), and any day whose detector voltages or template changed.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any

from .errors import note
from .theme import savefig_background
from .tool_window import figure_panel, set_readonly_text
from .ui_tips import tip


class RunCompareWindow(tk.Toplevel):
    """Compare a run template's runs from its history folder `hdir`.
    Opened with no folder (or one with no history yet), it says so."""

    def __init__(self, editor: Any, hdir: str | None = None):
        super().__init__(editor)
        self._editor = editor
        self.title("Run history")
        self.geometry('900x720')
        self.hdir = hdir
        self.df = None

        top = ttk.Frame(self, padding=6)
        top.pack(fill='x')
        self.where = tk.StringVar()
        ttk.Label(top, textvariable=self.where, style='Muted.TLabel',
                  wraplength=600, justify='left').pack(side='left')
        ob = ttk.Button(top, text="Open history folder…",
                        command=self._choose_folder)
        ob.pack(side='right')
        tip(editor, ob, "Show another run template's history: its "
            "'<template>_history' folder, beside the template.")

        ctl = ttk.Frame(self, padding=(6, 0))
        ctl.pack(fill='x')
        self.pop_var = tk.StringVar()
        self.stat_var = tk.StringVar()
        self.by_var = tk.StringVar(value='Group')
        self.pop_cb = self._combo(ctl, "Population:", self.pop_var, 28,
                                  "The gated population to follow over the "
                                  "runs (All events for whole-sample rows).")
        self.stat_cb = self._combo(ctl, "Statistic:", self.stat_var, 18,
                                   "Count, %Parent, %Total, or a channel's "
                                   "Median or GeoMean (linear values). Compare "
                                   "MFI across days only while the voltages "
                                   "stay the same.")
        self.by_cb = self._combo(ctl, "Lines:", self.by_var, 9,
                                 "One line per group, trial or sample. Each "
                                 "line's point for a day is the mean of its "
                                 "samples in that day's run.")
        self.all_var = tk.BooleanVar(value=False)
        allb = ttk.Checkbutton(ctl, text="Include superseded runs",
                               variable=self.all_var, command=self.reload)
        allb.pack(side='left', padx=(8, 0))
        tip(editor, allb, "Off (safe default): a day run more than once "
            "shows its newest run only. On: its older runs are plotted too, "
            "each its own point named by its run. The across-run test always "
            "uses each day's newest run.")

        self._figframe, self.fig, self.canvas = figure_panel(
            self, figsize=(8.6, 3.8), dpi=100)

        runrow = ttk.Frame(self, padding=(6, 4))
        runrow.pack(fill='x')
        self.run_var = tk.StringVar()
        self.run_cb = self._combo(runrow, "Within run:", self.run_var, 34,
                                  "The run whose groups are compared with "
                                  "each sample as a replicate (newest "
                                  "first).")
        eb = ttk.Button(runrow, text="Export table…", command=self.export_table)
        eb.pack(side='right')
        tip(editor, eb, "Save the plotted values as CSV: one row per "
            "plotted run (by experiment date), one column per line.")
        pb = ttk.Button(runrow, text="Save plot…", command=self.save_plot)
        pb.pack(side='right', padx=4)
        tip(editor, pb, "Save the plot as a PNG with a transparent "
            "background.")

        self.summary = tk.Text(self, height=12, wrap='word')
        self.summary.pack(fill='both', expand=False, padx=6, pady=(0, 6))
        for cb in (self.pop_cb, self.stat_cb, self.by_cb, self.run_cb):
            cb.bind('<<ComboboxSelected>>', lambda _e: self.redraw())
        self.reload()

    def _combo(self, parent: tk.Misc, label: str, var: tk.StringVar,
               width: int, help_text: str) -> ttk.Combobox:
        ttk.Label(parent, text=label).pack(side='left')
        cb = ttk.Combobox(parent, textvariable=var, width=width,
                          state='readonly')
        cb.pack(side='left', padx=(2, 8))
        tip(self._editor, cb, help_text)
        return cb

    # -- data --
    def reload(self) -> None:
        """Read the history again (after a run, or the superseded box)."""
        from .run_compare import load_history, populations, runs, series_choices, stat_names
        if not self.hdir:
            self.where.set("No run history open. Run a workspace built with "
                           "New run from template, or open a history folder.")
            self.df = None
            self._show_text('')
            return
        self.df = load_history(self.hdir, self.all_var.get())
        self.where.set(f"{self.hdir}  ·  {len(runs(self.df))} run(s)")
        for cb, var, values in (
                (self.pop_cb, self.pop_var, populations(self.df)),
                (self.stat_cb, self.stat_var, stat_names(self.df)),
                (self.by_cb, self.by_var, list(series_choices(self.df))),
                (self.run_cb, self.run_var,
                 [f'{r} ({d})' for r, d in runs(self.df)])):
            cb['values'] = values
            if values and var.get() not in values:
                var.set(values[0])
        self.redraw()

    def _choice(self) -> tuple[str, str, str] | None:
        if self.df is None or self.df.empty:
            return None
        pop, stat, by = self.pop_var.get(), self.stat_var.get(), self.by_var.get()
        if not (pop and stat and by):
            return None
        return pop, stat, by

    def redraw(self) -> None:
        """Plot the chosen population and statistic over the runs and write
        the tests."""
        from .run_compare import (
            compare_across,
            compare_within,
            label_date,
            per_run_means,
            result_lines,
            version_changes,
            within_rows,
        )
        self.fig.clear()
        choice = self._choice()
        if choice is None:
            self.canvas.draw()
            self._show_text("No finished runs in this history yet."
                            if self.hdir else '')
            return
        pop, stat, by = choice
        table = per_run_means(self.df, pop, stat, by)
        ax = self.fig.add_subplot(1, 1, 1)
        xs = list(range(len(table.index)))
        for col in table.columns:
            ax.plot(xs, table[col].values, marker='o', label=str(col))
        marks = set(version_changes(self.df))
        dates = [label_date(row) for row in table.index]
        for x, date in enumerate(dates):
            # Once per date: its first point (superseded runs share it).
            if date in marks and date not in dates[:x]:
                ax.axvline(x - 0.5, linestyle='--', linewidth=1, color='0.5')
        ax.set_xticks(xs)
        ax.set_xticklabels([str(d) for d in table.index], rotation=30,
                           ha='right', fontsize=8)
        ax.set_ylabel(stat)
        ax.set_title(pop, fontsize=10)
        if len(table.columns):
            ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        lines = ["Across runs (each run day one replicate):"]
        lines += result_lines(compare_across(self.df, pop, stat, by))
        run = self.run_var.get().split(' (')[0]
        if run:
            lines += ['', f"Within {self.run_var.get()} (each sample one "
                      "replicate):"]
            from_runs = sorted(set(within_rows(self.df, run)['Run']))
            if len(from_runs) > 1:
                lines.append("  (the day's active rows, from runs "
                             + ', '.join(map(str, from_runs)) + ")")
            lines += result_lines(compare_within(self.df, run, pop, stat, by))
        if marks:
            lines += ['', "The template changed on " + ', '.join(sorted(marks))
                      + " (dashed lines): compare across those with care."]
        lines += self._voltage_lines()
        self._show_text('\n'.join(lines))

    def _voltage_lines(self) -> list[str]:
        from .run_history import changes_since_last, load_runs, mark_active
        try:
            runs = mark_active(load_runs(self.hdir or ''))
        except OSError as exc:
            note("read the run records", exc)
            return []
        out = []
        for r in runs:
            if r.get('active'):
                out += [f"{r['experiment_date']}: {w}"
                        for w in changes_since_last(runs, r['run_id'])
                        if 'voltages' in w]
        return [''] + out if out else []

    def _show_text(self, text: str) -> None:
        set_readonly_text(self.summary, text)

    # -- actions --
    def _choose_folder(self) -> None:
        d = filedialog.askdirectory(parent=self, title="Open a run history folder")
        if d:
            self.hdir = d
            self.reload()

    def export_table(self) -> None:
        """Save the plotted values as CSV."""
        from .run_compare import export_table
        choice = self._choice()
        if choice is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Export run history table",
            defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if path:
            export_table(self.df, *choice, path)
            self._append(f"\nSaved {os.path.basename(path)}.")

    def save_plot(self) -> None:
        """Save the plot as PNG (transparent background)."""
        path = filedialog.asksaveasfilename(
            parent=self, title="Save run history plot",
            defaultextension='.png', filetypes=[('PNG', '*.png')])
        if path:
            # Exports are transparent by default, as every OpenFlo figure.
            savefig_background(self.fig, path, dpi=150)
            self._append(f"\nSaved {os.path.basename(path)}.")

    def _append(self, text: str) -> None:
        self._show_text(self.summary.get('1.0', 'end-1c') + text)
