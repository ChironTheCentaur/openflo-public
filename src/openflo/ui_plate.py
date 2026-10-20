"""Analyze, Plate view: a 96-well plate heatmap of one statistic per well
(Complaints 17).

Each sample's well comes from its well keyword (WELL ID and the like) or,
failing that, from its name (Plate1_B07); samples with no well are listed
under the plate. The statistic is the event count, or a gate's count or
percent of its parent, the gate found by name in each sample.

An index-sorted sample (Complaints 28) can be shown instead: each sorted
cell in the well it went to, coloured by a channel's value
(index_sort.well_values).

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from .sample_meta import (
    PLATE_COLS,
    PLATE_ROWS,
    plate_matrix,
    well_label,
    well_of,
)
from .theme import plt_get_cmap
from .tool_window import (
    ask_csv_path,
    background_selector,
    export_figure,
    figure_panel,
)
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import auto_tip

EVENTS = "Events"
SAMPLES = "Samples"
INDEX_PREFIX = "Index sort: "


def _gate_by_name(gates, name):
    for gid, g in gates.items():
        if g.get('name') == name:
            return gid
    return None


def statistic(sample, gates, stat):
    """``stat`` for one sample: EVENTS, "<gate>: count" or
    "<gate>: % of parent". None when the sample has no such gate or the
    parent is empty (NA, never 0)."""
    df = getattr(sample, 'data', None)
    if df is None:
        return None
    if stat == EVENTS:
        return float(len(df))
    gate, _sep, kind = stat.rpartition(': ')
    gid = _gate_by_name(gates, gate)
    if gid is None:
        return None
    from .pipeline import cumulative_gate_mask
    n = int(np.count_nonzero(cumulative_gate_mask(gates, gid, df)))
    if kind == 'count':
        return float(n)
    parent = gates[gid].get('parent_id')
    if parent is None or parent not in gates:
        total = len(df)
    else:
        total = int(np.count_nonzero(cumulative_gate_mask(gates, parent, df)))
    return 100.0 * n / total if total else None


class PlateWindow(tk.Toplevel):
    """[Statistic] / plate heatmap / samples without a well / buttons."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Plate view")
        self.minsize(640, 460)
        names = [n for n in editor._sample_order if n in editor._samples]
        self.wells = {}                     # sample -> (row, col)
        self.no_well = []
        for n in names:
            pos = well_of(editor._samples[n])
            if pos is None:
                self.no_well.append(n)
            else:
                self.wells[n] = pos
        gate_names = []
        for n in names:
            for g in editor._sample_gates.get(n, {}).values():
                nm = g.get('name')
                if nm and nm not in gate_names and g.get('kind') != 'autoclean':
                    gate_names.append(nm)
        self.stats = [EVENTS] + [f"{g}: {k}" for g in gate_names
                                 for k in ('% of parent', 'count')]
        from .index_sort import has_index_sort
        self.index_samples = [n for n in names
                              if has_index_sort(editor._samples[n])]

        top = ttk.Frame(self, padding=(12, 10, 12, 4))
        top.pack(side='top', fill='x')
        self.source_var = tk.StringVar(value=SAMPLES)
        if self.index_samples:
            ttk.Label(top, text="Show:").pack(side='left')
            self.source_combo = ttk.Combobox(
                top, textvariable=self.source_var, state='readonly', width=26,
                values=[SAMPLES] + [INDEX_PREFIX + n
                                    for n in self.index_samples])
            self.source_combo.pack(side='left', padx=(6, 12))
            self.source_combo.bind('<<ComboboxSelected>>',
                                   lambda _e: self._source_changed())
            auto_tip(self.source_combo,
                     "One value per sample, or the cells of an index-sorted "
                     "sample, each in the well it was sorted into.")
        ttk.Label(top, text="Statistic:").pack(side='left')
        self.stat_var = tk.StringVar(value=EVENTS)
        self.stat_combo = ttk.Combobox(top, textvariable=self.stat_var,
                                       values=self.stats, state='readonly',
                                       width=32)
        self.stat_combo.pack(side='left', padx=6)
        self.stat_combo.bind('<<ComboboxSelected>>', lambda _e: self.draw())
        auto_tip(self.stat_combo,
                 "Events per sample, or a gate's count or percent of its "
                 "parent (the gate found by name in each sample). A well "
                 "whose sample lacks the gate reads NA and stays grey.")

        bar = ttk.Frame(self, padding=(12, 4, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        self.bg_var, bg = background_selector(bar)
        b = ttk.Button(bar, text="Export figure…", command=self._export_png)
        b.pack(side='left')
        auto_tip(b, "Save the plate as PNG, SVG or PDF.")
        bg.pack(side='left', padx=(6, 12))
        c = ttk.Button(bar, text="Export CSV…", command=self._export_csv)
        c.pack(side='left')
        auto_tip(c, "Each sample's well and value, one row per sample.")
        self.status_var = tk.StringVar(value='')
        note = ("Samples without a well: " + ", ".join(self.no_well)
                if self.no_well else "")
        self.note_label = ttk.Label(self, text=note, style='Muted.TLabel',
                                    wraplength=600, justify='left',
                                    padding=(12, 0))
        self.note_label.pack(side='bottom', fill='x')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  padding=(12, 0)).pack(side='bottom', fill='x')
        _frame, self.fig, self.canvas = figure_panel(self, figsize=(7.5, 4.6))
        self.draw()

    def index_sample(self):
        """The index-sorted sample shown, or None for one value per
        sample."""
        src = self.source_var.get()
        return src[len(INDEX_PREFIX):] if src.startswith(INDEX_PREFIX) \
            else None

    def _source_changed(self):
        name = self.index_sample()
        if name is None:
            values = self.stats
        else:
            s = self.editor._samples[name]
            values = [c for c in self.editor._channels if c in s.data.columns]
        self.stat_combo.configure(values=values)
        self.stat_var.set(values[0] if values else '')
        self.draw()

    def cells(self):
        """``[(position, label, value)]`` shown: a sample per well, or each
        index-sorted cell (labelled by its event number) in its well."""
        name = self.index_sample()
        if name is None:
            vals = self.values()
            return [(self.wells[n], n, vals[n]) for n in self.wells]
        from .index_sort import event_wells
        from .pipeline import linear_values
        s = self.editor._samples[name]
        rows, cols = event_wells(s)
        v = np.asarray(linear_values(s, self.stat_var.get()), dtype=float)
        return [((int(rows[i]), int(cols[i])), f"{name} event {i + 1}",
                 float(v[i]) if np.isfinite(v[i]) else None)
                for i in range(len(v))
                if 0 <= rows[i] < len(PLATE_ROWS) and 0 <= cols[i] < PLATE_COLS]

    def values(self):
        """``{sample: value}`` of the chosen statistic for the samples with
        a well (None where it is NA)."""
        stat = self.stat_var.get()
        return {n: statistic(self.editor._samples[n],
                             self.editor._sample_gates.get(n, {}), stat)
                for n in self.wells}

    def matrix(self):
        """The 8 x 12 plate of the chosen statistic; a well holding two
        samples shows the later one, and both are listed in the CSV."""
        return plate_matrix({pos: v for pos, _lbl, v in self.cells()
                             if v is not None})

    def draw(self):
        from .theme import _dialog_dark_on, _theme_figure_dark
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        mat = self.matrix()
        cmap = plt_get_cmap('viridis')
        cmap.set_bad('#d9d9d9')
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect='equal')
        ax.set_xticks(range(PLATE_COLS), [str(c + 1) for c in
                                          range(PLATE_COLS)])
        ax.set_yticks(range(len(PLATE_ROWS)), list(PLATE_ROWS))
        ax.tick_params(length=0)
        ax.xaxis.tick_top()
        finite = mat[np.isfinite(mat)]
        mid = float(np.nanmean(finite)) if finite.size else 0.0
        pct = '%' in self.stat_var.get() and self.index_sample() is None
        for (r, c), v in np.ndenumerate(mat):
            if np.isfinite(v):
                ax.text(c, r, f"{v:.1f}" if pct else f"{v:,.0f}",
                        ha='center', va='center', fontsize=6,
                        color='white' if v < mid else 'black')
        cb = self.fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(self.stat_var.get())
        name = self.index_sample()
        ax.set_title(f"{len(self.cells())} cells index-sorted from {name}"
                     if name else f"{len(self.wells)} samples on the plate",
                     fontsize=9, pad=18)
        if _dialog_dark_on(self):
            _theme_figure_dark(self.fig)
        self.canvas.draw_idle()

    def rows(self):
        """``[(well, sample, value)]`` for the CSV, in plate order."""
        out = [(well_label(pos), lbl, v) for pos, lbl, v in self.cells()]
        out.sort(key=lambda r: (r[0][0], int(r[0][1:]), r[1]))
        return out

    def _export_csv(self):
        path = ask_csv_path(self, initialfile='plate.csv')
        if not path:
            return
        import csv
        try:
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(['well', 'cell' if self.index_sample() else
                            'sample', self.stat_var.get()])
                for well, n, v in self.rows():
                    w.writerow([well, n, 'NA' if v is None else f"{v:g}"])
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Plate view", "The table could not be saved.",
                       exc, path)
            return
        report_saved(self, path)

    def _export_png(self):
        export_figure(self, self.fig, background=self.bg_var.get(),
                      initialfile='plate.png')
