"""New run from template: the preview that puts the day's files on the
template's slots before anything runs.

Each file is a row: the slot it matched, and what stops it running (fits
no slot, fits two, lacks a channel the slot gates on, cannot be read). The
user sets a slot, or skips the file, for the selected rows. Load and
"Load and run" stay off until every row is ready or skipped: a run never
starts on a guess. The matching and the words are Tk-free
(``run_template``).
"""
from __future__ import annotations

import os
import tkinter as tk
from collections.abc import Callable, Mapping, Sequence
from tkinter import ttk
from typing import Any

from .run_template import (
    SKIP,
    file_channels,
    has_spillover,
    preview_assignment,
    preview_rows,
    row_comp_words,
    row_problem,
    row_status,
)
from .theme import current_palette
from .ui_tips import auto_tip

_SKIP_LABEL = '(skip this file)'


class RunTemplatePreview(tk.Toplevel):
    """Match the day's files to a run template's slots, then load (and
    optionally run) the workspace the template builds from them.

    ``on_done(assignment, run)`` gets ``{path: slot index}`` and whether
    "Load and run" was pressed. Opened with no template (a stand-alone
    check), it shows an empty list."""

    def __init__(self, editor: Any,
                 template: Mapping[str, Any] | None = None,
                 paths: Sequence[str] = (),
                 on_done: Callable[[dict[str, int], bool], object] | None = None,
                 channels_of: Callable[[str], list[str]] = file_channels,
                 parent: tk.Misc | None = None,
                 spill_of: Callable[[str], bool] = has_spillover):
        super().__init__(parent or editor)
        self._editor = editor
        self._template = template or {'title': '', 'slots': []}
        self._on_done = on_done
        self.title(f"New run from template: {self._template.get('title') or ''}"
                   .rstrip(': '))
        self.geometry('820x420')
        self.rows = preview_rows(self._template, list(paths), channels_of,
                                 spill_of) \
            if template else []

        ttk.Label(self, justify='left', wraplength=680, padding=(10, 8, 10, 4),
                  text=("Each file goes to the template slot its name "
                        "matches. Check the list, set a slot or skip any "
                        "file that is not ready, then load.")).pack(anchor='w')

        body = ttk.Frame(self, padding=(10, 0))
        body.pack(fill='both', expand=True)
        cols = ('file', 'slot', 'comp', 'status')
        self.tree = ttk.Treeview(body, columns=cols, show='headings',
                                 selectmode='extended', height=10)
        for c, w, txt in (('file', 220, 'File'), ('slot', 120, 'Slot'),
                          ('comp', 170, 'Compensation'),
                          ('status', 260, 'Status')):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, stretch=(c != 'slot'))
        sb = ttk.Scrollbar(body, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='left', fill='y')
        self.tree.tag_configure('problem', foreground=current_palette()['error'])

        setrow = ttk.Frame(self, padding=(10, 6))
        setrow.pack(fill='x')
        ttk.Label(setrow, text="Slot for the selected file(s):").pack(side='left')
        names = [s['name'] for s in self._template.get('slots', [])]
        self._slot_var = tk.StringVar(value=names[0] if names else _SKIP_LABEL)
        self._slot_box = ttk.Combobox(setrow, textvariable=self._slot_var,
                                      values=[*names, _SKIP_LABEL],
                                      state='readonly', width=24)
        self._slot_box.pack(side='left', padx=6)
        auto_tip(self._slot_box,
                 "The slot (template group) the selected files run in, with "
                 "its gates, compensation and settings. Choose "
                 f"'{_SKIP_LABEL}' to leave a file out of this run.")
        setb = ttk.Button(setrow, text="Set", command=self._set_selected)
        setb.pack(side='left')
        auto_tip(setb, "Put the selected files in the slot chosen at left.")

        self.summary = tk.StringVar()
        ttk.Label(self, textvariable=self.summary, wraplength=680,
                  justify='left', padding=(10, 0)).pack(anchor='w')

        btns = ttk.Frame(self, padding=(10, 8))
        btns.pack(fill='x')
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side='right')
        self.run_btn = ttk.Button(btns, text="Load and run",
                                  command=lambda: self._finish(True))
        self.run_btn.pack(side='right', padx=6)
        auto_tip(self.run_btn,
                 "Build the workspace from these files and run it now, "
                 "with the template's run settings.")
        self.load_btn = ttk.Button(btns, text="Load only",
                                   command=lambda: self._finish(False))
        self.load_btn.pack(side='right')
        auto_tip(self.load_btn,
                 "Build the workspace from these files without running, to "
                 "check or change it first. It replaces the open workspace "
                 "(Undo brings that back).")
        self._refresh()

    # -- state --
    def _slot_text(self, row: Mapping[str, Any]) -> str:
        if row['slot'] == SKIP:
            return _SKIP_LABEL
        if row['slot'] is None:
            return ''
        return self._template['slots'][row['slot']]['name']

    def _refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        n_problem = 0
        for i, r in enumerate(self.rows):
            bad = bool(row_problem(self._template, r))
            n_problem += bad
            self.tree.insert('', 'end', iid=str(i),
                             values=(os.path.basename(r['path']),
                                     self._slot_text(r),
                                     row_comp_words(self._template, r),
                                     row_status(self._template, r)),
                             tags=('problem',) if bad else ())
        try:
            assignment = preview_assignment(self._template, self.rows)
        except ValueError as exc:
            self.summary.set(f"Not ready: {exc}." if self.rows
                             else "No files to match.")
            ok = False
        else:
            used = set(assignment.values())
            empty = [s['name'] for i, s in enumerate(self._template['slots'])
                     if i not in used]
            self.summary.set(
                f"{len(assignment)} file(s) ready"
                + (f"; no file for {', '.join(empty)}" if empty else '')
                + '.')
            ok = True
        for b in (self.run_btn, self.load_btn):
            b.state(['!disabled'] if ok else ['disabled'])

    def set_rows(self, indices: Sequence[int], slot_name: str) -> None:
        """Set the rows at `indices` to the slot named `slot_name` (or skip
        them for the skip entry), and refresh the list."""
        names = [s['name'] for s in self._template.get('slots', [])]
        slot = SKIP if slot_name == _SKIP_LABEL else names.index(slot_name)
        for i in indices:
            self.rows[i]['slot'] = slot
        self._refresh()

    def _set_selected(self) -> None:
        picked = [int(i) for i in self.tree.selection()]
        if picked:
            self.set_rows(picked, self._slot_var.get())

    def _finish(self, run: bool) -> None:
        try:
            assignment = preview_assignment(self._template, self.rows)
        except ValueError:
            self._refresh()
            return
        self.destroy()
        if self._on_done is not None:
            self._on_done(assignment, run)
