"""Group comparison: Kruskal-Wallis + pairwise Mann-Whitney.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np


class GroupStatsWindow(tk.Toplevel):
    """Compare a channel's per-sample median across trial groups: Kruskal-Wallis
    omnibus + pairwise Mann-Whitney (BH-adjusted) + effect sizes."""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Group comparison")
        self.geometry("660x540")
        self._editor = editor
        chans = []
        for s in editor._samples.values():
            for c in s.data.columns:
                if c not in chans:
                    chans.append(c)
        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=10, pady=8)
        ttk.Label(bar, text="Channel:").pack(side='left')
        self._ch = tk.StringVar(value=chans[0] if chans else '')
        ttk.Combobox(bar, textvariable=self._ch, values=chans,
                     state='readonly', width=26).pack(side='left', padx=6)
        ttk.Button(bar, text="Compare groups", command=self._run).pack(
            side='left', padx=6)
        self._txt = tk.Text(self, wrap='word')
        self._txt.pack(fill='both', expand=True, padx=10, pady=(0, 6))
        ttk.Label(
            self, foreground='grey', wraplength=620, justify='left',
            text=("Per-sample median of the channel, grouped by trial. "
                  "Kruskal-Wallis omnibus across groups, then pairwise "
                  "Mann-Whitney U (Benjamini-Hochberg adjusted) with Cliff's "
                  "delta effect size.")).pack(anchor='w', padx=10, pady=(0, 8))

    def _run(self):

        from .stats import effect_size, multi_group_test, posthoc_pairwise
        ch = self._ch.get()
        ed = self._editor
        groups = {}
        for name, s in ed._samples.items():
            if ch not in s.data.columns:
                continue
            tr = ed._sample_trial.get(name, '(ungrouped)')
            groups.setdefault(tr, []).append(
                float(np.nanmedian(s.data[ch].to_numpy(dtype=float))))
        groups = {k: v for k, v in groups.items() if v}
        self._txt.configure(state='normal')
        self._txt.delete('1.0', 'end')
        if len(groups) < 2:
            self._txt.insert('end', "Need at least 2 groups (trials) with "
                                    "samples to compare.")
            self._txt.configure(state='disabled')
            return
        omni = multi_group_test(groups)
        lines = [f"Channel: {ch}   (per-sample median, grouped by trial)", ""]
        for gname, vals in groups.items():
            lines.append(f"  {gname}: n={len(vals)}  "
                         f"median={np.median(vals):.4g}")
        lines += ["", f"Omnibus {omni['test']}: stat={omni['stat']:.4g}, "
                      f"p={omni['p']:.4g}  (k={omni['k']} groups)", ""]
        ph = posthoc_pairwise(groups)
        if ph:
            lines.append("Pairwise (Mann-Whitney U, BH-adjusted):")
            for r in ph:
                es = effect_size(groups[r['a']], groups[r['b']])
                lines.append(
                    f"  {r['a']} vs {r['b']}:  p={r['p']:.4g}  "
                    f"p_adj={r['p_adj']:.4g}  Cliff's δ={es['cliffs_delta']:.2f}")
        self._txt.insert('end', "\n".join(lines))
        self._txt.configure(state='disabled')
