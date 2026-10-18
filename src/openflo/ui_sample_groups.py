"""Edit, Group samples: put the loaded samples into groups (trials) by an
FCS keyword or by a sample-to-condition table (Complaints 17).

The groups are the trials the sample tree, the group statistics and the
exports already use, so grouping here is the same as dragging each sample
onto its group. A preview shows each sample's group before Apply.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from .sample_meta import grouping_keywords, groups_by_keyword, read_condition_table
from .ui_errors import show_error
from .ui_tips import auto_tip


class GroupSamplesDialog(tk.Toplevel):
    """[By keyword: (combobox)] [From a table…] / preview / [Apply] [Cancel]."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Group samples")
        self.minsize(460, 360)
        self.samples = [editor._samples[n] for n in editor._sample_order
                        if n in editor._samples]
        self.groups: dict[str, str] = {}
        self.source = ''

        top = ttk.Frame(self, padding=(12, 12, 12, 6))
        top.pack(side='top', fill='x')
        ttk.Label(top, text="By keyword:").pack(side='left')
        self.choices = grouping_keywords(self.samples)
        self.keyword_var = tk.StringVar(value='')
        self.keyword_combo = ttk.Combobox(
            top, textvariable=self.keyword_var, state='readonly', width=24,
            values=[f"{k} ({n} groups)" for k, n in self.choices])
        self.keyword_combo.pack(side='left', padx=(6, 10))
        self.keyword_combo.bind('<<ComboboxSelected>>',
                                lambda _e: self._use_keyword())
        auto_tip(self.keyword_combo,
                 "A keyword the FCS files carry, such as the tube, well or "
                 "condition the acquisition software wrote. Each value "
                 "becomes a group.")
        table = ttk.Button(top, text="From a table…",
                           command=self._use_table)
        table.pack(side='left')
        auto_tip(table,
                 "A CSV or tab-separated table with a header: the sample "
                 "(its name or file name) in the first column, its group in "
                 "the column headed condition, group or trial (else the "
                 "second).")

        bar = ttk.Frame(self, padding=(12, 6, 12, 12))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(
            side='right')
        self.apply_button = ttk.Button(bar, text="Apply", default='active',
                                       command=self.apply, state='disabled')
        self.apply_button.pack(side='right', padx=(0, 6))
        auto_tip(self.apply_button,
                 "Move each sample to the group shown; samples showing "
                 "(unchanged) stay where they are.")
        self.note_var = tk.StringVar(value=(
            "Pick a keyword, or load a table." if self.choices else
            "The files carry no keyword that splits them into groups; "
            "load a table instead."))
        ttk.Label(self, textvariable=self.note_var, style='Muted.TLabel',
                  wraplength=430, justify='left', padding=(12, 0)).pack(
            side='bottom', fill='x')

        mid = ttk.Frame(self, padding=(12, 0))
        mid.pack(side='top', fill='both', expand=True)
        self.tree = ttk.Treeview(mid, columns=('sample', 'group'),
                                 show='headings', height=12)
        self.tree.heading('sample', text="Sample")
        self.tree.heading('group', text="Group")
        self.tree.column('sample', width=230)
        self.tree.column('group', width=180)
        sb = ttk.Scrollbar(mid, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self._show({})

    # ── sources ────────────────────────────────────────────────────────────
    def _use_keyword(self):
        i = self.keyword_combo.current()
        if i < 0:
            return
        key = self.choices[i][0]
        self.source = f"keyword {key}"
        self._show(groups_by_keyword(self.samples, key))
        n = len(set(self.groups.values()))
        self.note_var.set(f"{len(self.groups)} samples in {n} groups by "
                          f"{key}.")

    def use_table_text(self, text, name='the table'):
        """Read a sample-to-condition table (the From a table… button reads
        the file into this)."""
        res = read_condition_table(text, [s.name for s in self.samples])
        self.source = name
        self._show(res['groups'])
        note = (f"{len(res['groups'])} samples in "
                f"{len(set(res['groups'].values()))} groups from {name}.")
        if res['missing']:
            note += (f" Not in the table, kept in their group: "
                     f"{', '.join(res['missing'][:6])}"
                     f"{'…' if len(res['missing']) > 6 else ''}.")
        if res['unmatched_rows']:
            note += (f" Rows naming no loaded sample: "
                     f"{', '.join(res['unmatched_rows'][:6])}"
                     f"{'…' if len(res['unmatched_rows']) > 6 else ''}.")
        self.note_var.set(note)
        return res

    def _use_table(self):
        path = filedialog.askopenfilename(
            parent=self, title="Sample-to-condition table",
            filetypes=[('Tables', '*.csv *.tsv *.txt'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8-sig') as fh:
                text = fh.read()
            import os
            self.use_table_text(text, os.path.basename(path))
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Group samples",
                       "The table could not be read.", exc, path)

    # ── preview and apply ─────────────────────────────────────────────────
    def _show(self, groups):
        self.groups = dict(groups)
        self.tree.delete(*self.tree.get_children())
        for s in self.samples:
            cur = self.editor._trial_for(s.name)
            new = self.groups.get(s.name)
            self.tree.insert('', 'end', values=(
                s.name, new if new else f"{cur} (unchanged)"))
        self.apply_button.configure(
            state='normal' if self.groups else 'disabled')

    def apply(self):
        if not self.groups:
            return
        self.editor._apply_sample_groups(self.groups, self.source)
        self.destroy()
