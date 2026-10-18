"""A read-only results table: a note, the rows, Copy table and Export CSV.

Self-contained Tk window (see ui_*.py convention), shared by the FMO
threshold comparison and Compare with saved session.
"""
from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import ttk

from .tool_window import ask_csv_path
from .ui_errors import show_error
from .ui_tips import tip


def cell_text(val, fmt=True):
    """A cell as shown (``fmt``) or as copied/exported: NaN reads NA, floats
    get 3 decimals on screen and full precision in a copy or CSV."""
    if isinstance(val, float):
        if val != val:
            return 'NA'
        return f"{val:,.3f}" if fmt else val
    if isinstance(val, bool):
        return 'yes' if val else ''
    if isinstance(val, int) and fmt:
        return f"{val:,}"
    return '' if val is None else val


class TableWindow(tk.Toplevel):
    """``rows`` (dicts) under ``columns``; rows whose ``flag_key`` is truthy
    are tinted. ``csv_name`` is the suggested export file name."""

    def __init__(self, editor, title='Table', columns=(), rows=(), note='',
                 status='', csv_name='table.csv', flag_key=None):
        super().__init__(editor)
        self.editor = editor
        self.title(title)
        self.geometry("980x460")
        self.minsize(520, 260)
        self._cols = list(columns)
        self._rows = list(rows)
        self._csv_name = csv_name

        self.status_var = tk.StringVar(value=status)
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=940, justify='left',
                  padding=(10, 0, 10, 4)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(side='bottom', fill='x')
        _b = ttk.Button(btns, text="Export CSV…", command=self._export_csv)
        _b.pack(side='left')
        tip(editor, _b, "Save the table, full precision, as a CSV.")
        _b = ttk.Button(btns, text="Copy table", command=self._copy_table)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Copy the table, tab separated, to paste into a "
                        "spreadsheet, Prism or a document.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')
        if note:
            ttk.Label(self, padding=(10, 8, 10, 4), wraplength=940,
                      justify='left', text=note).pack(side='top', fill='x')

        holder = ttk.Frame(self, padding=(10, 0, 10, 6))
        holder.pack(side='top', fill='both', expand=True)
        self.tv = ttk.Treeview(holder, show='headings', height=12,
                               columns=self._cols)
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
        bg, fg = flag_tints()['warn']
        self.tv.tag_configure('flag', background=bg, foreground=fg)
        for i, c in enumerate(self._cols):
            self.tv.heading(c, text=c)
            self.tv.column(c, anchor='w' if i < 2 else 'e',
                           width=170 if i < 2 else 120, stretch=False)
        for r in self._rows:
            self.tv.insert('', 'end',
                           values=[cell_text(r.get(c)) for c in self._cols],
                           tags=('flag',) if flag_key and r.get(flag_key)
                           else ())

    def table_lines(self):
        out: list[list] = [list(self._cols)]
        for r in self._rows:
            out.append([cell_text(r.get(c), fmt=False) for c in self._cols])
        return out

    def _copy_table(self):
        if not self._rows:
            self.status_var.set("Nothing to copy.")
            return
        from .clipboard_export import copy_table
        copy_table(self, self.table_lines())
        self.status_var.set(f"Copied {len(self._rows)} row(s) to the "
                            "clipboard.")

    def _export_csv(self):
        if not self._rows:
            self.status_var.set("Nothing to export.")
            return
        path = ask_csv_path(self, title=self.title(),
                            initialfile=self._csv_name)
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(self.table_lines())
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Export failed",
                       "The table could not be exported.", exc, path)
            return
        self.status_var.set(f"Exported {len(self._rows)} row(s) → "
                            f"{os.path.basename(path)}")
