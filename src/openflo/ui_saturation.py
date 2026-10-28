"""Saturation report window: per sample, the percent of events at the top of
each channel's range (openflo.saturation).

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import ttk

from .saturation import FLAG_PERCENT, saturation_report
from .tool_window import ask_csv_path
from .ui_errors import show_error
from .ui_tips import tip


class SaturationReportWindow(tk.Toplevel):
    """One row per loaded sample: events, the percent at the ceiling of any
    channel, events QC already removed, then one column per channel."""

    _ID_COLS = ('Sample', 'Events', 'Removed by QC')

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Saturation report")
        self.geometry("900x420")
        self.minsize(560, 280)
        self._rows, self._cols = [], []

        # Buttons and status first, at the bottom, so a short window shrinks
        # the table rather than hiding them.
        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=860, justify='left',
                  padding=(10, 0, 10, 4)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(side='bottom', fill='x')
        _b = ttk.Button(btns, text="Refresh", command=self._refresh)
        _b.pack(side='left')
        tip(editor, _b, "Recompute for the samples loaded now.")
        _b = ttk.Button(btns, text="Export CSV…", command=self._export_csv)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Save the table, full precision, as a CSV.")
        _b = ttk.Button(btns, text="Copy table", command=self._copy_table)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Copy the table, tab separated, to paste into a "
                        "spreadsheet or document.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        ttk.Label(self, padding=(10, 8, 10, 4), wraplength=860,
                  justify='left',
                  text="Percent of each sample's events at the top of a "
                       "channel's range ($PnR - 1, raw detector values "
                       "before compensation). Those events were clipped by "
                       "the detector. 'Any channel' counts an event once.",
                  ).pack(side='top', fill='x')

        holder = ttk.Frame(self, padding=(10, 0, 10, 6))
        holder.pack(side='top', fill='both', expand=True)
        self.tv = ttk.Treeview(holder, show='headings', height=10)
        ysb = ttk.Scrollbar(holder, orient='vertical', command=self.tv.yview)
        xsb = ttk.Scrollbar(holder, orient='horizontal',
                            command=self.tv.xview)
        self.tv.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tv.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        xsb.grid(row=1, column=0, sticky='ew')
        holder.rowconfigure(0, weight=1)
        holder.columnconfigure(0, weight=1)
        from .theme import flag_tints
        bg, fg = flag_tints()['warn']      # readable on light and dark
        self.tv.tag_configure('flag', background=bg, foreground=fg)
        self._refresh()

    def _samples(self):
        ed = self.editor
        return [(nm, ed._samples[nm]) for nm in ed._sample_order
                if nm in ed._samples]

    def _refresh(self):
        samples = self._samples()
        if not samples:
            self._show([], [])
            self.status_var.set("Load samples to see their saturation.")
            return
        from .async_task import run_async
        self.status_var.set("Checking…")
        run_async(self, lambda: saturation_report(samples),
                  on_done=lambda res: self._show(*res),
                  on_error=lambda exc: self.status_var.set(
                      f"Saturation report failed: {exc}"))

    def _label(self, col):
        """A channel column's title: its antibody label beside the
        detector when it has one."""
        if col in self._ID_COLS:
            return col
        if col == 'Any channel':
            return 'Any channel %'
        lbl = self.editor._channel_labels.get(col) if hasattr(
            self.editor, '_channel_labels') else None
        return f"{col} {lbl} %" if lbl and lbl != col else f"{col} %"

    @staticmethod
    def _cell(row, col, fmt=True):
        val = row.get(col, '')
        if isinstance(val, float):
            if val != val:
                return 'NA'
            return f"{val:.2f}" if fmt else val
        if fmt and isinstance(val, int) and col != 'Sample':
            return f"{val:,}"
        return val

    def _show(self, rows, cols):
        self._rows, self._cols = rows, cols
        self.tv.delete(*self.tv.get_children())
        self.tv['columns'] = cols
        for c in cols:
            self.tv.heading(c, text=self._label(c))
            self.tv.column(c, anchor='w' if c == 'Sample' else 'e',
                           width=160 if c == 'Sample' else 110,
                           stretch=False)
        for r in rows:
            self.tv.insert('', 'end',
                           values=[self._cell(r, c) for c in cols],
                           tags=('flag',) if r['__flagged__'] else ())
        if not rows:
            return
        flagged = [r for r in rows if r['__flagged__']]
        msg = (f"{len(rows)} sample(s). " + (
            f"[!] {len(flagged)} with {FLAG_PERCENT:g} % or more of events "
            "at a channel's top: " + "; ".join(
                f"{r['Sample']} ({', '.join(r['__flagged__'])})"
                for r in flagged) + "."
            if flagged else
            f"No channel has {FLAG_PERCENT:g} % or more of its events at "
            "the top."))
        unknown = [r['Sample'] for r in rows if r['__unknown__']]
        if unknown:
            msg += ("  NA = the file gives no range for that channel "
                    f"({', '.join(unknown)}).")
        if any(r.get('Removed by QC') for r in rows):
            msg += ("  Events QC removed (a pile-up among them) are not "
                    "counted here.")
        self.status_var.set(msg)

    def _table_lines(self):
        out: list[list] = [[self._label(c) for c in self._cols]]
        for r in self._rows:
            out.append([self._cell(r, c, fmt=False) for c in self._cols])
        return out

    def _copy_table(self):
        if not self._rows:
            self.status_var.set("Nothing to copy.")
            return
        from .clipboard_export import copy_table
        copy_table(self, self._table_lines())
        self.status_var.set(f"Copied {len(self._rows)} row(s) to the "
                            "clipboard.")

    def _export_csv(self):
        if not self._rows:
            self.status_var.set("Nothing to export.")
            return
        path = ask_csv_path(self, title="Export saturation report",
                            initialfile='saturation_report.csv')
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(self._table_lines())
            self.status_var.set(f"Exported {len(self._rows)} row(s) → "
                                f"{os.path.basename(path)}")
        except Exception as exc:
            show_error(self, "Export failed",
                       "The saturation report could not be exported.", exc,
                       path)
