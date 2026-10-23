"""Derived channels window: define a channel computed from others
(openflo.derived), e.g. the ratio of two detectors for a ratio dye.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from . import derived
from .ui_tips import tip


class DerivedChannelsWindow(tk.Toplevel):
    """Name + formula, a channel list to insert from, and the derived
    channels already defined, each removable."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Derived channels")
        self.geometry("720x460")
        self.minsize(520, 340)

        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=680, justify='left',
                  padding=(10, 0, 10, 4)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(side='bottom', fill='x')
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        ttk.Label(self, padding=(10, 8, 10, 4), wraplength=680,
                  justify='left',
                  text="A new channel computed per event from the others, "
                       "on their compensated linear values: the ratio of "
                       "two detectors for a ratio dye, FRET, or one marker "
                       "per cell over another. Name channels in square "
                       "brackets; + - * / ** and log, log10, log2, sqrt, "
                       "exp, abs, asinh, min, max can be used. Events where "
                       "the result is not a number (0 / 0) are left out of "
                       "plots and statistics on it."
                  ).pack(side='top', fill='x')

        form = ttk.Frame(self, padding=(10, 0, 10, 4))
        form.pack(side='top', fill='x')
        ttk.Label(form, text="Name:").grid(row=0, column=0, sticky='w')
        self.name_var = tk.StringVar(master=self, value='Ratio')
        e = ttk.Entry(form, textvariable=self.name_var, width=24)
        e.grid(row=0, column=1, sticky='w', padx=(4, 0))
        tip(editor, e, "Shown as D:<name> beside the measured channels.")
        ttk.Label(form, text="Formula:").grid(row=1, column=0, sticky='w',
                                              pady=(4, 0))
        self.formula_var = tk.StringVar(master=self, value='')
        self.formula_entry = ttk.Entry(form, textvariable=self.formula_var,
                                       width=60)
        self.formula_entry.grid(row=1, column=1, sticky='ew', padx=(4, 0),
                                pady=(4, 0))
        tip(editor, self.formula_entry,
            "For example [FL1-A] / [FL2-A], or log10([FL1-A] / [FL2-A]). A "
            "marker name works in place of its detector.")
        form.columnconfigure(1, weight=1)
        _b = ttk.Button(form, text="Add to all samples", command=self._add)
        _b.grid(row=2, column=1, sticky='w', padx=(4, 0), pady=(6, 0))
        tip(editor, _b, "Compute the channel for every loaded sample. Saved "
                        "with the session, and recomputed from the FCS "
                        "files when it is reopened from them.")

        body = ttk.Frame(self, padding=(10, 4, 10, 6))
        body.pack(side='top', fill='both', expand=True)
        left = ttk.LabelFrame(body, text="Channels (double-click to insert)",
                              padding=4)
        left.pack(side='left', fill='both', expand=True)
        self.chan_list = tk.Listbox(left, height=10, exportselection=False)
        self.chan_list.pack(fill='both', expand=True)
        self.chan_list.bind('<Double-Button-1>', self._insert_channel)
        right = ttk.LabelFrame(body, text="Derived channels", padding=4)
        right.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.tv = ttk.Treeview(right, show='headings', height=8,
                               columns=('channel', 'formula', 'samples'))
        for c, t, w in (('channel', 'Channel', 120),
                        ('formula', 'Formula', 200),
                        ('samples', 'Samples', 70)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor='w')
        self.tv.pack(fill='both', expand=True)
        _b = ttk.Button(right, text="Remove", command=self._remove)
        _b.pack(anchor='w', pady=(4, 0))
        tip(editor, _b, "Delete the selected derived channel from every "
                        "sample. Not allowed while a gate is drawn on it.")
        self._fill()

    def _measured(self):
        ed = self.editor
        return [c for c in ed._channels if not derived.is_derived(c)
                and not str(c).startswith(('MESF:', 'U:'))]

    def _fill(self):
        ed = self.editor
        self.chan_list.delete(0, 'end')
        for c in self._measured():
            self.chan_list.insert('end', ed._fmt_channel(c))
        self.tv.delete(*self.tv.get_children())
        for col, (text, n) in self._defined().items():
            self.tv.insert('', 'end', iid=col, values=(col, text, n))

    def _defined(self):
        """{column: (formula, number of samples carrying it)}."""
        out: dict = {}
        for s in self.editor._samples.values():
            for col, text in (getattr(s, 'derived', None) or {}).items():
                prev = out.get(col, (text, 0))
                out[col] = (prev[0], prev[1] + 1)
        return out

    def _insert_channel(self, _e=None):
        sel = self.chan_list.curselection()
        if not sel:
            return
        ch = self._measured()[sel[0]]
        self.formula_entry.insert('insert', f'[{ch}]')
        self.formula_entry.focus_set()

    def _add(self):
        ed = self.editor
        if not ed._samples:
            self.status_var.set("Load samples first.")
            return
        name, text = self.name_var.get().strip(), self.formula_var.get()
        try:
            col, applied, failed = derived.apply_to_samples(
                ed._samples, name, text, getattr(ed, '_channel_labels', {}))
        except derived.FormulaError as exc:
            self.status_var.set(str(exc))
            return
        for nm in applied:
            ed._data_changed(nm)
        if applied:
            ed._refresh_channel_choices()       # picks up the new column
            ed._audit('derived channel', column=col, formula=text,
                      n_samples=len(applied))
        msg = f"'{col}' added to {len(applied)} sample(s)."
        if failed:
            msg += "  Not added: " + "; ".join(f"{n} ({why})"
                                               for n, why in failed)
        self.status_var.set(msg)
        self._fill()

    def _remove(self):
        ed = self.editor
        sel = self.tv.selection()
        if not sel:
            self.status_var.set("Select a derived channel to remove.")
            return
        col = sel[0]
        users = derived.gates_using(getattr(ed, '_sample_gates', {}), col)
        if users:
            self.status_var.set(
                f"Gates on '{col}' in {', '.join(users)}: delete them "
                "first.")
            return
        if not messagebox.askyesno(
                "Remove derived channel",
                f"Remove '{col}' from every sample?", parent=self):
            return
        for nm, s in ed._samples.items():
            if col in s.data.columns or col in (getattr(s, 'derived', None)
                                                or {}):
                derived.remove_from_sample(s, col)
                ed._data_changed(nm)
        if col in ed._channels:
            ed._channels.remove(col)
        ed._refresh_channel_choices()
        ed._audit('derived channel removed', column=col)
        self.status_var.set(f"Removed '{col}'.")
        self._fill()
