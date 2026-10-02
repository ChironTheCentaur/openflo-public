"""FMO-control gating dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui_tips import auto_tip


class FMOGatingDialog(tk.Toplevel):
    """Map marker channels to FMO control samples; place threshold gates on the
    active (stained) sample at each FMO's percentile cutoff. Copy them to other
    samples afterwards via Edit → Copy gates to…."""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("FMO gating")
        self.geometry("540x540")
        self._editor = editor
        active = editor._active_sample
        # `_open_fmo_gating` already refuses without an active sample, so this
        # is defence in depth rather than a live path — but `editor._samples[
        # None]` raising a bare `KeyError: None` from a CONSTRUCTOR is a poor
        # failure for a precondition the dialog can state itself.
        sample_obj = editor._samples.get(active)
        if sample_obj is None:
            self._ok = False
            ttk.Label(self, padding=16, justify='left',
                      text=("Select the stained sample to gate first, then "
                            "reopen FMO gating.")).pack()
            return
        self._ok = True
        df = sample_obj.data
        channels = list(getattr(sample_obj, 'fluor_channels', None)
                        or list(df.columns))
        names = list(editor._samples.keys())

        ttk.Label(
            self, justify='left',
            text=(f"Stained sample:  {active}\n\nMap each marker to its FMO "
                  "control (the tube stained for everything EXCEPT that "
                  "marker). A threshold gate is placed on the stained sample "
                  "at the FMO percentile.")).pack(anchor='w', padx=10,
                                                  pady=(10, 6))
        prow = ttk.Frame(self)
        prow.pack(anchor='w', padx=10)
        ttk.Label(prow, text="Percentile:").pack(side='left')
        self._pct = tk.StringVar(value='99')
        _pc = ttk.Spinbox(prow, from_=90, to=100, increment=0.5, width=6,
                          textvariable=self._pct)
        _pc.pack(side='left', padx=6)
        auto_tip(_pc,
                 "Where in the FMO control the positive threshold is placed. "
                 "99 means 1% of FMO events fall above the gate -- the "
                 "conventional choice. Lower it and more FMO spread is called "
                 "positive; raise it and you trade sensitivity for "
                 "specificity. Applies to every channel below.")

        body = ttk.Frame(self)
        body.pack(fill='both', expand=True, padx=10, pady=6)
        cv = tk.Canvas(body, highlightthickness=0)
        sb = ttk.Scrollbar(body, orient='vertical', command=cv.yview)
        inner = ttk.Frame(cv)
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        _win = cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))
        # Stretch the inner frame to the canvas width so there's no dead
        # column of blank space to the right of the controls.
        cv.bind('<Configure>', lambda e: cv.itemconfigure(_win, width=e.width))
        self._map = {}
        opts = ['(none)'] + [s for s in names if s != active]
        for ch in channels:
            row = ttk.Frame(inner)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text=str(ch), width=20).pack(side='left')
            var = tk.StringVar(value='(none)')
            _cb = ttk.Combobox(row, textvariable=var, values=opts,
                               state='readonly')
            _cb.pack(side='left', fill='x', expand=True, padx=(0, 8))
            auto_tip(_cb,
                     f"The FMO control for {ch}: a sample stained with every "
                     "marker EXCEPT this one. Its upper edge is where the "
                     "positive gate goes, because it shows the spread this "
                     "channel has when nothing real is in it. '(none)' leaves "
                     "the channel without an FMO reference.")
            self._map[ch] = var

        bar = ttk.Frame(self)
        bar.pack(side='bottom', fill='x', pady=6)
        _b = ttk.Button(bar, text="Apply", command=self._apply)
        _b.pack(side='right', padx=8)
        auto_tip(_b,
                 "Record these FMO assignments and use them for threshold "
                 "suggestions on the matching channels.")
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')

    def _apply(self):
        from .gating_helpers import fmo_threshold_gate
        if not getattr(self, '_ok', True):
            return                      # built without a sample; nothing to do
        ed = self._editor
        try:
            pct = float(self._pct.get())
        except ValueError:
            pct = 99.0
        if not (0.0 <= pct <= 100.0):
            # Out of range would make np.percentile raise for EVERY channel,
            # which the per-channel except swallows into a misleading "nothing
            # added" — validate up front and tell the user the real reason.
            ed.status_var.set("FMO percentile must be between 0 and 100.")
            return
        added = 0
        for ch, var in self._map.items():
            fmo = var.get()
            if fmo == '(none)' or fmo not in ed._samples:
                continue
            fdf = ed._samples[fmo].data
            if ch not in fdf.columns:
                continue
            try:
                gate = fmo_threshold_gate(fdf, ch, percentile=pct)
            except Exception as exc:
                print(f"[fmo] {ch}: {exc}", flush=True)
                continue
            gate.pop('id', None)
            ed._add_gate(gate)
            added += 1
        if added:
            ed._refresh_gate_list()
            ed._schedule_replot(0)
            ed.status_var.set(
                f"Added {added} FMO threshold gate(s) at the {pct:g}th "
                f"percentile to {ed._active_sample}.")
            self.destroy()
        else:
            ed.status_var.set("No FMO mappings chosen — nothing added.")
