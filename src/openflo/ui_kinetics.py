"""Kinetics window: a channel over acquisition time per sample, overlaid,
with each sample's response summarised (openflo.kinetics).

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from . import kinetics
from .tool_window import (
    ask_csv_path,
    background_selector,
    export_figure,
    figure_panel,
)
from .ui_errors import show_error
from .ui_tips import tip

COLS = (('Sample', 'Sample'), ('baseline', 'Baseline'), ('peak', 'Peak'),
        ('time_to_peak', 'Time to peak'), ('auc', 'AUC'),
        ('fold', 'Peak / baseline'), ('n_events', 'Events'))


def _num(text):
    try:
        v = float(str(text).strip())
    except ValueError:
        return None
    return v if np.isfinite(v) else None


class KineticsWindow(tk.Toplevel):
    """Channel, population, statistic, bins, baseline end and stimulus time;
    one trace per sample and a table of their responses."""

    ALL_EVENTS = 'All events'

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Kinetics")
        self.geometry("1000x720")
        self.minsize(640, 480)
        self._results: list[dict] = []
        self._unit = 's'

        ctl = ttk.Frame(self, padding=6)
        ctl.pack(side='top', fill='x')
        ttk.Label(ctl, text="Channel:").pack(side='left')
        chans = [c for c in editor._channels
                 if not self._is_time_name(c)]
        self._chan_cols = chans
        disp = [editor._fmt_channel(c) for c in chans]
        self.chan_var = tk.StringVar(master=self, value=next(
            (d for c, d in zip(chans, disp, strict=True) if str(c).startswith('D:')),
            disp[0] if disp else ''))
        _c = ttk.Combobox(ctl, textvariable=self.chan_var, width=20,
                          state='readonly', values=disp)
        _c.pack(side='left', padx=(2, 8))
        tip(editor, _c, "The channel to follow over time, on its "
                        "compensated linear values. A derived ratio "
                        "(Tools > Derived channels) is the usual read-out "
                        "for a ratio dye.")
        ttk.Label(ctl, text="Population:").pack(side='left')
        self.pop_var = tk.StringVar(master=self, value=self.ALL_EVENTS)
        _p = ttk.Combobox(ctl, textvariable=self.pop_var, width=18,
                          state='readonly',
                          values=self._population_choices())
        _p.pack(side='left', padx=(2, 8))
        tip(editor, _p, "Follow only the events in this gated population.")
        ttk.Label(ctl, text="Statistic:").pack(side='left')
        self.stat_var = tk.StringVar(master=self, value='median')
        _s = ttk.Combobox(ctl, textvariable=self.stat_var, width=8,
                          state='readonly', values=kinetics.STATS)
        _s.pack(side='left', padx=(2, 8))
        tip(editor, _s, "What each time bin shows: the median of its "
                        "events (the usual choice, robust to outliers) or "
                        "their mean.")
        ttk.Label(ctl, text="Bins:").pack(side='left')
        self.bins_var = tk.StringVar(master=self, value='100')
        _e = ttk.Entry(ctl, textvariable=self.bins_var, width=5)
        _e.pack(side='left', padx=(2, 8))
        tip(editor, _e, "How many equal time bins each sample's run is cut "
                        "into. Fewer bins, smoother traces.")

        ctl2 = ttk.Frame(self, padding=(6, 0, 6, 6))
        ctl2.pack(side='top', fill='x')
        ttk.Label(ctl2, text="Baseline ends at:").pack(side='left')
        self.base_var = tk.StringVar(master=self, value='')
        _e = ttk.Entry(ctl2, textvariable=self.base_var, width=8)
        _e.pack(side='left', padx=(2, 8))
        tip(editor, _e, "Events before this time are the baseline. Empty: "
                        "the first tenth of each sample's run.")
        ttk.Label(ctl2, text="Stimulus at:").pack(side='left')
        self.stim_var = tk.StringVar(master=self, value='')
        _e = ttk.Entry(ctl2, textvariable=self.stim_var, width=8)
        _e.pack(side='left', padx=(2, 8))
        tip(editor, _e, "When the stimulus went in; the response (peak, "
                        "time to peak, area) is read from here on. Empty: "
                        "where the baseline ends.")
        self.unit_label = ttk.Label(ctl2, text='')
        self.unit_label.pack(side='left', padx=(0, 8))
        _b = ttk.Button(ctl2, text="Plot", command=self._refresh)
        _b.pack(side='left')
        tip(editor, _b, "Redraw the traces and the table with these "
                        "settings.")

        self.status_var = tk.StringVar(master=self, value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=960, justify='left',
                  padding=(10, 0, 10, 4)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(side='bottom', fill='x')
        _b = ttk.Button(btns, text="Export table…", command=self._export_csv)
        _b.pack(side='left')
        tip(editor, _b, "Save each sample's response as a CSV.")
        _b = ttk.Button(btns, text="Export traces…",
                        command=self._export_traces)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Save every sample's binned trace as a CSV: time, "
                        "statistic and events per bin.")
        self.bg_var, _bg = background_selector(btns)
        _bg.pack(side='left', padx=(12, 4))
        _b = ttk.Button(btns, text="Export figure…",
                        command=self._export_figure)
        _b.pack(side='left')
        tip(editor, _b, "Save the traces as PNG, PDF or SVG.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        holder = ttk.Frame(self, padding=(10, 0, 10, 6))
        holder.pack(side='bottom', fill='x')
        self.tv = ttk.Treeview(holder, show='headings', height=6,
                               columns=[c for c, _ in COLS])
        for c, t in COLS:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=160 if c == 'Sample' else 100,
                           anchor='w' if c == 'Sample' else 'e')
        self.tv.pack(fill='x')

        plot = ttk.Frame(self)
        plot.pack(side='top', fill='both', expand=True)
        _f, self.fig, self.canvas = figure_panel(plot, figsize=(9, 4.5))
        self._refresh()

    @staticmethod
    def _is_time_name(c):
        from .pipeline import _is_time
        return _is_time(str(c))

    # population helpers, as in the expression window
    def _population_gids(self, name):
        ed = self.editor
        gates = ed._sample_gates.get(name, {})
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
        seen: dict = {}
        for n in self.editor._sample_order:
            if n in self.editor._samples:
                for path in self._population_gids(n):
                    seen.setdefault(path, None)
        return [self.ALL_EVENTS] + list(seen)

    def _channel(self):
        disp = self.chan_var.get()
        for c in self._chan_cols:
            if self.editor._fmt_channel(c) == disp:
                return c
        return self.editor._resolve_channel(disp)

    def _compute(self):
        """Per sample: its trace and summary. Returns (results, notes)."""
        from .pipeline import (
            UnknownScaleError,
            cumulative_gate_mask,
            linear_values,
        )
        ed = self.editor
        ch = self._channel()
        pop = self.pop_var.get() or self.ALL_EVENTS
        stat = self.stat_var.get()
        bins = _num(self.bins_var.get())
        bins = int(bins) if bins and bins >= 2 else 100
        base_end, stim = _num(self.base_var.get()), _num(self.stim_var.get())
        results, skipped = [], []
        units = set()
        for n in ed._sample_order:
            s = ed._samples.get(n)
            if s is None:
                continue
            col = ed._marker_column_for(s, ch) if ch else None
            if not col:
                skipped.append((n, 'no such channel'))
                continue
            try:
                t, unit = kinetics.sample_time(s)
                v = linear_values(s, col)
            except UnknownScaleError:
                skipped.append((n, 'scale unknown'))
                continue
            except ValueError:
                skipped.append((n, 'no Time channel'))
                continue
            v = np.asarray(v, dtype=float)
            if pop != self.ALL_EVENTS:
                gid = self._population_gids(n).get(pop)
                if gid is None:
                    skipped.append((n, 'population not drawn'))
                    continue
                m = np.asarray(cumulative_gate_mask(
                    ed._sample_gates.get(n, {}), gid, s.data), dtype=bool)
                t, v = t[m], v[m]
            ok = np.isfinite(t) & np.isfinite(v)
            if not ok.any():
                skipped.append((n, 'no events'))
                continue
            t0, t1 = float(t[ok].min()), float(t[ok].max())
            be = base_end if base_end is not None else t0 + 0.1 * (t1 - t0)
            rng = (t0, t1)
            centres, tr, counts = kinetics.trace(t, v, bins=bins, stat=stat,
                                                 t_range=rng)
            summ = kinetics.summarize(t, v, be, stimulus=stim, bins=bins,
                                      stat=stat, t_range=rng)
            units.add(unit)
            results.append({'Sample': n, 'centres': centres, 'trace': tr,
                            'counts': counts, 'unit': unit,
                            'stimulus': be if stim is None else stim,
                            **summ})
        return results, skipped, units

    def _refresh(self):
        results, skipped, units = self._compute()
        self._results = results
        self._unit = units.pop() if len(units) == 1 else 'mixed'
        self.unit_label.configure(
            text=f"time in {self._unit}" if results else '')
        self._draw()
        self.tv.delete(*self.tv.get_children())
        for r in results:
            self.tv.insert('', 'end', values=[self._cell(r, c)
                                              for c, _ in COLS])
        msg = (f"{len(results)} sample(s)." if results else
               "No sample to show: load samples with a Time channel.")
        if self._unit == 'ticks':
            msg += ("  The files don't say how long a Time tick is "
                    "($TIMESTEP), so time is in ticks.")
        elif self._unit == 'mixed':
            msg += ("  Some files give time in seconds and some only in "
                    "ticks; compare them with care.")
        if skipped:
            msg += "  Left out: " + "; ".join(f"{n} ({why})"
                                              for n, why in skipped) + "."
        if any(not np.isfinite(r['baseline']) for r in results):
            msg += ("  NA baseline: no events before the baseline end.")
        self.status_var.set(msg)

    @staticmethod
    def _cell(r, c, fmt=True):
        v = r.get(c, '')
        if isinstance(v, float):
            if not np.isfinite(v):
                return 'NA'
            return f"{v:.4g}" if fmt else v
        return v

    def _draw(self):
        fig = self.fig
        fig.clear()
        ax = fig.add_subplot(111)
        for r in self._results:
            ax.plot(r['centres'], r['trace'], lw=1.2, label=r['Sample'])
        stims = {round(r['stimulus'], 9) for r in self._results}
        if len(stims) == 1:
            ax.axvline(stims.pop(), color='0.5', ls='--', lw=1)
        ax.set_xlabel(f"Time ({self._unit})")
        ax.set_ylabel(f"{self.stat_var.get()} {self.chan_var.get()}")
        if self._results and len(self._results) <= 12:
            ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        try:
            from .theme import _dialog_dark_on, _theme_figure_dark
            if _dialog_dark_on(self):
                _theme_figure_dark(fig)
        except Exception:
            pass
        self.canvas.draw_idle()

    def _table_lines(self):
        head = [t if c not in ('time_to_peak',) else
                f"{t} ({self._unit})" for c, t in COLS]
        return [head] + [[self._cell(r, c, fmt=False) for c, _ in COLS]
                         for r in self._results]

    def _write_csv(self, rows, title, initial):
        path = ask_csv_path(self, title=title, initialfile=initial)
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)
            self.status_var.set(f"Exported → {os.path.basename(path)}")
        except Exception as exc:
            show_error(self, "Export failed",
                       "The kinetics could not be exported.", exc, path)

    def _export_csv(self):
        if not self._results:
            self.status_var.set("Nothing to export.")
            return
        self._write_csv(self._table_lines(), "Export kinetics table",
                        'kinetics.csv')

    def _export_traces(self):
        if not self._results:
            self.status_var.set("Nothing to export.")
            return
        rows: list[list] = [['Sample', f'Time ({self._unit})',
                             self.stat_var.get(), 'Events']]
        for r in self._results:
            for t, v, k in zip(r["centres"], r["trace"], r["counts"], strict=True):
                rows.append([r['Sample'], float(t),
                             float(v) if np.isfinite(v) else 'NA', int(k)])
        self._write_csv(rows, "Export kinetics traces",
                        'kinetics_traces.csv')

    def _export_figure(self):
        export_figure(self, self.fig, background=self.bg_var.get(),
                      initialfile='kinetics.png')
