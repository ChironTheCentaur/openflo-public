"""Tools, Bead tracking: a Levey-Jennings chart of QC bead runs with
Westgard rules (Complaints 25).

Each run adds the active sample's median per fluorescence channel (in a
population, the bead singlets say) to a log kept as a CSV beside the
instrument; the chart shows one channel's runs against the baseline mean
and 1, 2 and 3 SD, each run coloured by the rules it breaks
(bead_tracking.westgard).

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from .bead_tracking import BeadLog
from .tool_window import background_selector, export_figure, figure_panel
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_proliferation import ALL_EVENTS, population_values
from .ui_tips import auto_tip

STATUS_COLOURS = {'ok': '#2a7f3f', 'warning': '#c98a00', 'reject': '#c0392b',
                  'NA': '#999999'}


def run_values(sample, gates, population, channels):
    """``{channel: median}`` of linear values in ``population``; a channel
    with no finite values is left out (NA)."""
    out = {}
    for ch in channels:
        v = population_values(sample, gates, ch, population)
        if v is None:
            return None
        v = v[np.isfinite(v)]
        if v.size:
            out[ch] = float(np.median(v))
    return out


def run_date(sample):
    from .sample_meta import keywords
    kw = keywords(sample)
    return kw.get('DATE') or kw.get('BTIM') or ''


class BeadTrackingWindow(tk.Toplevel):
    """[Open log] [Save log] [Population] [Add active sample] / chart /
    runs table."""

    def __init__(self, editor, log=None):
        super().__init__(editor)
        self.editor = editor
        self.title("Bead tracking")
        self.minsize(760, 560)
        self.log = log or BeadLog()
        self.path = None
        s = editor._samples.get(editor._active_sample)
        gates = editor._sample_gates.get(editor._active_sample, {})
        pops = [ALL_EVENTS] + [g['name'] for g in gates.values()
                               if g.get('name') and g.get('kind') != 'autoclean']

        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(side='top', fill='x')
        o = ttk.Button(top, text="Open log…", command=self.open_log)
        o.pack(side='left')
        auto_tip(o, "Open a bead log (CSV: date, sample, one column per "
                    "channel).")
        sv = ttk.Button(top, text="Save log…", command=self.save_log)
        sv.pack(side='left', padx=(6, 12))
        auto_tip(sv, "Save the log as CSV to keep adding runs to it.")
        ttk.Label(top, text="Population:").pack(side='left')
        self.population_var = tk.StringVar(value=pops[-1])
        pc = ttk.Combobox(top, textvariable=self.population_var, values=pops,
                          state='readonly', width=18)
        pc.pack(side='left', padx=4)
        auto_tip(pc, "The beads' gate in the active sample (singlet beads).")
        add = ttk.Button(top, text="Add active sample", command=self.add_active)
        add.pack(side='left', padx=(6, 0))
        auto_tip(add, "Add the active sample's medians as the next run.")
        if s is None:
            add.state(['disabled'])

        mid = ttk.Frame(self, padding=(12, 0, 12, 4))
        mid.pack(side='top', fill='x')
        ttk.Label(mid, text="Channel:").pack(side='left')
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(mid, textvariable=self.channel_var,
                                          state='readonly', width=22)
        self.channel_combo.pack(side='left', padx=4)
        self.channel_combo.bind('<<ComboboxSelected>>', lambda _e: self.draw())
        auto_tip(self.channel_combo, "The channel to chart.")
        ttk.Label(mid, text="Baseline runs:").pack(side='left', padx=(12, 0))
        self.baseline_var = tk.IntVar(value=self.log.baseline_runs)
        bs = ttk.Spinbox(mid, from_=2, to=100, textvariable=self.baseline_var,
                         width=5, command=self.draw)
        bs.pack(side='left', padx=4)
        bs.bind('<Return>', lambda _e: self.draw())
        auto_tip(bs, "How many of the first runs set the mean and SD (20 is "
                     "the usual; fewer while a new lot settles).")

        bar = ttk.Frame(self, padding=(12, 4, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        self.bg_var, bg = background_selector(bar)
        b = ttk.Button(bar, text="Export figure…", command=self._export_png)
        b.pack(side='left')
        auto_tip(b, "Save the chart as PNG, SVG or PDF.")
        bg.pack(side='left', padx=(6, 0))
        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, wraplength=720,
                  justify='left', padding=(12, 2)).pack(side='bottom',
                                                        fill='x')
        self.table = ttk.Treeview(self, columns=('date', 'value', 'rules'),
                                  show='tree headings', height=6)
        self.table.heading('#0', text='Run')
        self.table.heading('date', text='Date')
        self.table.heading('value', text='Median')
        self.table.heading('rules', text='Westgard rules')
        self.table.column('#0', width=160)
        self.table.column('rules', width=200)
        for st, colour in STATUS_COLOURS.items():
            self.table.tag_configure(st, foreground=colour)
        self.table.pack(side='bottom', fill='x', padx=12)
        _frame, self.fig, self.canvas = figure_panel(self, figsize=(7, 3.4))
        self.refresh()

    def refresh(self):
        chans = self.log.channels
        self.channel_combo.configure(values=chans)
        if chans and self.channel_var.get() not in chans:
            self.channel_var.set(chans[0])
        self.draw()

    def add_active(self):
        e = self.editor
        name = e._active_sample
        s = e._samples[name]
        chans = [c for c in (getattr(s, 'fluor_channels', None) or [])
                 if c in s.data.columns]
        vals = run_values(s, e._sample_gates.get(name, {}),
                          self.population_var.get(), chans)
        if not vals:
            self.status_var.set(f"{name} has no events in "
                                f"'{self.population_var.get()}'.")
            return
        self.log.add(run_date(s), name, vals)
        self.refresh()
        self.status_var.set(f"Added {name} as run {len(self.log.runs)}. "
                            "Save log… keeps it.")

    def judged(self):
        self.log.baseline_runs = max(2, int(self.baseline_var.get() or 20))
        ch = self.channel_var.get()
        return self.log.judge(ch) if ch else None

    def draw(self):
        from .theme import _dialog_dark_on, _theme_figure_dark
        self.fig.clear()
        for row in self.table.get_children():
            self.table.delete(row)
        res = self.judged()
        if res is None or not self.log.runs:
            self.status_var.set("Open a log or add a bead run to start.")
            self.canvas.draw_idle()
            return
        ax = self.fig.add_subplot(111)
        x = np.arange(1, len(res['values']) + 1)
        ax.plot(x, res['values'], color='0.5', lw=0.8)
        for st, colour in STATUS_COLOURS.items():
            sel = np.array([s == st for s in res['status']])
            if sel.any():
                ax.scatter(x[sel], res['values'][sel], color=colour, s=22,
                           zorder=3, label=st)
        m, sd = res['mean'], res['sd']
        if np.isfinite(m) and np.isfinite(sd):
            ax.axhline(m, color='black', lw=1)
            for k, ls in ((1, ':'), (2, '--'), (3, '-')):
                for sgn in (1, -1):
                    ax.axhline(m + sgn * k * sd, color='0.4', lw=0.7, ls=ls)
            self.status_var.set(
                f"Baseline of the first {min(self.log.baseline_runs, len(x))}"
                f" runs: mean {m:,.4g}, SD {sd:,.3g} "
                f"(CV {100 * sd / m:.1f}%)." if m else "")
        else:
            self.status_var.set("Two or more baseline runs are needed for a "
                                "mean and SD; until then nothing is judged.")
        ax.set_xlabel('run')
        ax.set_ylabel(f"{self.editor._fmt_channel(self.channel_var.get())} "
                      "median")
        ax.legend(fontsize=7, loc='best')
        self.fig.tight_layout()
        if _dialog_dark_on(self):
            _theme_figure_dark(self.fig)
        self.canvas.draw_idle()
        for i, (run, v, rules, st) in enumerate(zip(
                self.log.runs, res['values'], res['rules'], res['status'],
                strict=True)):
            self.table.insert('', 'end', text=f"{i + 1}. {run['sample']}",
                              values=(run['date'],
                                      'NA' if not np.isfinite(v)
                                      else f"{v:,.4g}",
                                      ', '.join(rules) or st),
                              tags=(st,))

    def open_log(self):
        path = filedialog.askopenfilename(
            parent=self, title="Open bead log",
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                self.log = BeadLog.from_csv(fh.read(),
                                            baseline_runs=self.log.baseline_runs)
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Bead tracking", "The log could not be opened.",
                       exc, path)
            return
        self.path = path
        self.refresh()

    def save_log(self):
        path = self.path or filedialog.asksaveasfilename(
            parent=self, title="Save bead log", defaultextension='.csv',
            initialfile='bead_log.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8', newline='') as fh:
                fh.write(self.log.to_csv())
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Bead tracking", "The log could not be saved.",
                       exc, path)
            return
        self.path = path
        report_saved(self, path)

    def _export_png(self):
        export_figure(self, self.fig, background=self.bg_var.get(),
                      initialfile='bead_tracking.png')
