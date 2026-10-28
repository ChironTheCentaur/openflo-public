"""File, Concatenate samples: join the events of several loaded samples into
one new sample (Complaints 26; the command line has done this with
``--concat``).

The new sample keeps every event of the samples picked, each tagged with
its source in a ``__sample__`` column (1, 2, ... in the order listed, the
names kept in its metadata), so it can be gated, clustered or exported as
one. The samples must share their channels and display transforms; their
gates stay with them.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pandas as pd

from .ui_tips import auto_tip

SOURCE_COLUMN = '__sample__'


def concatenate_samples(samples, name):
    """A new FlowSample holding every event of ``samples`` (FlowSample-like:
    ``.name``, ``.data``, ``.channel_labels``, ``.data_transforms``), with a
    ``__sample__`` column numbering the source from 1. Raises ValueError
    when they do not share channels or transforms."""
    from .pipeline import FlowSample
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("Pick two or more samples to concatenate.")
    cols = list(samples[0].data.columns)
    tf = getattr(samples[0], 'data_transforms', {}) or {}
    for s in samples[1:]:
        if set(s.data.columns) != set(cols):
            raise ValueError(f"{s.name} does not have the same channels as "
                             f"{samples[0].name}.")
        if (getattr(s, 'data_transforms', {}) or {}) != tf:
            raise ValueError(f"{s.name} is not on the same scales as "
                             f"{samples[0].name}; apply the same transforms "
                             "first.")
    frames = []
    for i, s in enumerate(samples, 1):
        df = s.data[cols].copy()
        df[SOURCE_COLUMN] = i
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    labels = dict(getattr(samples[0], 'channel_labels', {}) or {})
    labels[SOURCE_COLUMN] = 'Source sample'
    meta = {'OPENFLO_CONCAT_SOURCES': '; '.join(
        f"{i}={s.name}" for i, s in enumerate(samples, 1))}
    out = FlowSample.from_dataframe(data, name=name, labels=labels,
                                    metadata=meta)
    out.data_transforms = {k: dict(v) for k, v in tf.items()}
    return out


class ConcatenateDialog(tk.Toplevel):
    """Samples to join (all ticked samples picked), the new name, and
    [Concatenate] [Cancel]."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Concatenate samples")
        self.names = [n for n in editor._sample_order if n in editor._samples]
        frm = ttk.Frame(self, padding=(12, 12, 12, 6))
        frm.pack(side='top', fill='both', expand=True)
        ttk.Label(frm, text="Samples to join:").pack(anchor='w')
        self.listbox = tk.Listbox(frm, selectmode='multiple',
                                  exportselection=False,
                                  height=min(12, max(4, len(self.names))),
                                  width=40)
        for i, n in enumerate(self.names):
            self.listbox.insert('end', n)
            if editor._sample_plot_enabled.get(n, False):
                self.listbox.selection_set(i)
        self.listbox.pack(fill='both', expand=True, pady=(4, 8))
        row = ttk.Frame(frm)
        row.pack(fill='x')
        ttk.Label(row, text="New sample name:").pack(side='left')
        self.name_var = tk.StringVar(value=self._free_name("Concatenated"))
        ent = ttk.Entry(row, textvariable=self.name_var, width=28)
        ent.pack(side='left', padx=6)
        auto_tip(ent, "The joined sample's name. Each event keeps its source "
                 "in the __sample__ column (1, 2, ... in this list's order).")
        self.note_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.note_var, style='Error.TLabel',
                  wraplength=380, padding=(12, 0)).pack(fill='x')
        bar = ttk.Frame(self, padding=(12, 6, 12, 12))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Cancel", command=self.destroy).pack(
            side='right')
        go = ttk.Button(bar, text="Concatenate", default='active',
                        command=self.apply)
        go.pack(side='right', padx=(0, 6))
        auto_tip(go, "Add the joined sample to the session; the samples "
                 "picked stay as they are.")

    def _free_name(self, base):
        name, i = base, 2
        while name in self.editor._samples:
            name, i = f"{base} {i}", i + 1
        return name

    def apply(self):
        picked = [self.names[i] for i in self.listbox.curselection()]
        name = self.name_var.get().strip()
        if not name or name in self.editor._samples:
            self.note_var.set("Give the new sample a name no loaded sample "
                              "has.")
            return
        try:
            new = concatenate_samples(
                [self.editor._samples[n] for n in picked], name)
        except ValueError as exc:
            self.note_var.set(str(exc))
            return
        self.editor._add_concatenated_sample(new, picked)
        self.destroy()
