"""FMO-control gating dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from .async_task import BusyStrip
from .dialog_inputs import read_number
from .fmo_sensitivity import COLUMNS, DEFAULT_PERCENTILES
from .ui_tips import auto_tip


def _in_stained_coordinates(fmo, stained):
    """``(frame, unconvertible)``: the FMO's events in the STAINED sample's
    display coordinates -- where the stained sample's gates live -- and the
    columns that cannot be put there because either side's transform is
    unknown. Columns both samples carry the same way are not copied."""
    from .pipeline import (
        is_unknown_spec,
        to_linear,
        transform_params,
        transform_values,
    )
    f_rec = getattr(fmo, 'data_transforms', None) or {}
    s_rec = getattr(stained, 'data_transforms', None) or {}
    out = fmo.data.copy(deep=False)
    bad = []
    for c in out.columns:
        a, b = f_rec.get(c), s_rec.get(c)
        if a == b:
            continue
        if is_unknown_spec(a) or is_unknown_spec(b):
            bad.append(c)
            continue
        lin = to_linear(out[c].values, a)
        out[c] = transform_values(lin, **transform_params(b)) if b else lin
    return out, bad


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
        pid = editor._selected_gate_id()
        if pid is not None:
            from .gating import population_path
            path = population_path(
                editor._gates, pid,
                transforms=editor._display_transforms(active))
            where = (f" under the selected gate ({path}), with the FMO "
                     "percentile taken inside that same population.")
        else:
            where = " at the FMO percentile (no gate selected: a root gate)."

        ttk.Label(
            self, justify='left', wraplength=500,
            text=(f"Stained sample:  {active}\n\nMap each marker to its FMO "
                  "control (the tube stained for everything EXCEPT that "
                  "marker). A threshold gate is placed on the stained sample"
                  + where)).pack(anchor='w', padx=10, pady=(10, 6))
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
            known = getattr(editor, '_fmo_controls', {}).get(ch)
            var = tk.StringVar(value=known if known in opts else '(none)')
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
        # The FMO controls are read off the window thread, shown here.
        self._busy = BusyStrip(self, padding=(10, 0))
        self._busy.pack(side='bottom', fill='x')
        # Main action, then Close, at the right (tool_window).
        ttk.Button(bar, text="Close", command=self.destroy).pack(
            side='right', padx=(0, 8))
        _b = self._apply_btn = ttk.Button(bar, text="Apply", default='active',
                                          command=self._apply)
        _b.pack(side='right', padx=(0, 6))
        auto_tip(_b,
                 "Record these FMO assignments and use them for threshold "
                 "suggestions on the matching channels.")
        _c = self._compare_btn = ttk.Button(
            bar, text="Compare thresholds…", command=self._compare)
        _c.pack(side='left', padx=8)
        auto_tip(_c,
                 "Without adding gates: every stained sample's % positive "
                 "on each mapped channel with the threshold set at each of "
                 "the percentiles below, to see how much the choice of "
                 "cutoff moves the result.")
        self._cmp_pcts = tk.StringVar(value=', '.join(
            f'{p:g}' for p in DEFAULT_PERCENTILES))
        _e = ttk.Entry(bar, textvariable=self._cmp_pcts, width=12)
        _e.pack(side='left')
        auto_tip(_e,
                 "Percentiles to compare, separated by commas. 99 puts 1% "
                 "of FMO events above the gate, 99.5 puts 0.5% above.")

    def _pairs(self):
        """``[(channel, FMO name, FMO sample)]`` for the mapped channels."""
        ed = self._editor
        pairs = []
        for ch, var in self._map.items():
            fmo = var.get()
            if fmo == '(none)' or fmo not in ed._samples:
                continue
            fs = ed._samples[fmo]
            if ch not in fs.data.columns:
                continue
            pairs.append((ch, fmo, fs))
        return pairs

    def _parent(self):
        """``(pid, parent_gates, parent_chs)`` of the selected gate."""
        from .gating import gate_channels
        ed = self._editor
        pid = ed._selected_gate_id()
        parent_chs = set()
        if pid is not None:
            for g in ed._gates.values():
                parent_chs |= gate_channels(g)
        import copy as _copy
        parent_gates = _copy.deepcopy(ed._gates) if pid is not None else {}
        return pid, parent_gates, parent_chs

    def _compare(self):
        """Every stained sample's % positive at each percentile, in a table
        (no gate is added)."""
        from .fmo_sensitivity import parse_percentiles
        if not getattr(self, '_ok', True):
            return
        ed = self._editor
        try:
            pcts = parse_percentiles(self._cmp_pcts.get())
        except ValueError as exc:
            ed.status_var.set(f"Compare thresholds: {exc}.")
            return
        pairs = self._pairs()
        if not pairs:
            ed.status_var.set("Map at least one channel to its FMO control "
                              "to compare thresholds.")
            return
        pid, parent_gates, parent_chs = self._parent()
        stained = ed._samples[ed._active_sample]
        fmo_names = {fmo for _ch, fmo, _fs in pairs}
        others = [ed._samples[n] for n in ed._sample_order
                  if n in ed._samples and n not in fmo_names]
        self._busy.run(
            lambda job: self._sensitivity(job, pairs, others, pid,
                                          parent_gates, parent_chs, stained,
                                          pcts),
            on_done=self._show_sensitivity,
            on_error=lambda exc: ed.status_var.set(
                f"Compare thresholds failed: {type(exc).__name__}: {exc}"),
            on_cancel=lambda: ed.status_var.set("Compare thresholds "
                                                "cancelled."),
            text="Comparing thresholds…",
            disable=[self._apply_btn, self._compare_btn])

    @staticmethod
    def _in_parent(sample, stained, pid, parent_gates, parent_chs):
        """Boolean rows of `sample` inside the selected parent gate, read in
        the stained sample's coordinates (all rows with no parent)."""
        from .pipeline import UnknownScaleError, cumulative_gate_mask
        if pid is None:
            return np.ones(len(sample.data), dtype=bool)
        if sample is stained:
            frame, bad = sample.data, []
        else:
            frame, bad = _in_stained_coordinates(sample, stained)
        blocked = sorted(set(bad) & parent_chs)
        if blocked:
            raise UnknownScaleError(
                f"{', '.join(blocked)} of '{sample.name}' is on an unknown "
                "scale, so the parent gate cannot be read on it")
        return np.asarray(cumulative_gate_mask(parent_gates, pid, frame),
                          dtype=bool)

    @classmethod
    def _sensitivity(cls, job, pairs, others, pid, parent_gates, parent_chs,
                     stained, pcts):
        """``(rows, refused)`` off the window thread."""
        from .fmo_sensitivity import threshold_rows
        from .pipeline import UnknownScaleError, linear_values
        refused = []
        fmos = {}
        masks = {}

        def rows_of(s):
            if s.name not in masks:
                masks[s.name] = cls._in_parent(s, stained, pid, parent_gates,
                                               parent_chs)
            return masks[s.name]

        n = len(pairs) + len(others)
        for i, (ch, fmo, fs) in enumerate(pairs):
            job.progress(i, n, f"Reading {fmo} for {ch}…")
            try:
                rows = rows_of(fs)
                fmos[ch] = (fmo, linear_values(
                    fs, ch, fs.data[ch].to_numpy(float)[rows]))
            except UnknownScaleError as exc:
                refused.append(f"{ch} from {fmo} ({exc})")
        stained_vals = {}
        for j, s in enumerate(others):
            job.progress(len(pairs) + j, n, f"Reading {s.name}…")
            chans = {}
            try:
                rows = rows_of(s)
            except UnknownScaleError as exc:
                refused.append(f"{s.name} ({exc})")
                continue
            for ch in fmos:
                if ch not in s.data.columns:
                    continue
                try:
                    chans[ch] = linear_values(
                        s, ch, s.data[ch].to_numpy(float)[rows])
                except UnknownScaleError as exc:
                    refused.append(f"{ch} of {s.name} ({exc})")
            if chans:
                stained_vals[s.name] = chans
        return threshold_rows(fmos, stained_vals, pcts), refused

    def _show_sensitivity(self, res):
        from .fmo_sensitivity import spread
        from .ui_table import TableWindow
        rows, refused = res
        ed = self._editor
        if not rows:
            ed.status_var.set("Compare thresholds: nothing to compare."
                              + (f" Not read: {'; '.join(refused)}."
                                 if refused else ''))
            return
        pid = ed._selected_gate_id()
        if pid is not None:
            from .gating import population_path
            under = "inside " + population_path(
                ed._gates, pid,
                transforms=ed._display_transforms(ed._active_sample))
        else:
            under = "on all events (no gate selected)"
        moved = spread(rows)
        worst = max(moved.items(), key=lambda kv: kv[1]) if moved else None
        status = (f"{len(rows)} row(s), {under}."
                  + (f" The threshold moves {worst[0][0]} / {worst[0][1]} "
                     f"most: {worst[1]:.2f} percentage points."
                     if worst else '')
                  + (f"  [!] Not read: {'; '.join(refused)}."
                     if refused else ''))
        TableWindow(
            ed, "Compare FMO thresholds", COLUMNS, rows,
            note=("Each stained sample's % positive with the gate set at "
                  "each FMO percentile, " + under + ". Threshold is linear "
                  "(compensated) intensity; positive is at or above it. "
                  "No gate was added."),
            status=status, csv_name='fmo_threshold_comparison.csv')

    def _apply(self):
        from .gating import gate_channels
        if not getattr(self, '_ok', True):
            return                      # built without a sample; nothing to do
        ed = self._editor
        pct = read_number(self._pct, "Percentile", parent=self)
        if pct is None:
            return
        if not (0.0 <= pct <= 100.0):
            # Out of range would make np.percentile raise for EVERY channel,
            # which the per-channel except swallows into a misleading "nothing
            # added" — validate up front and tell the user the real reason.
            ed.status_var.set("FMO percentile must be between 0 and 100.")
            return
        # The gates go under the selected gate, as every other new gate does
        # (_add_gate's default). The FMO must then be read through that SAME
        # parent population: a cutoff taken over every FMO event lets events
        # the parent excludes set it. Measured with a 'cells' parent and a
        # bright-debris FMO: 0.662 over all events against 0.317 inside the
        # parent, which called 16.8% of a 50%-positive stained sample CD3+.
        pid = ed._selected_gate_id()
        # The cutoff is a LINEAR quantity: the FMO's percentile on its linear
        # values, mapped into the stained sample's display coordinates through
        # the stained sample's own record. Taken in the FMO's display
        # coordinates it was only right when both tubes happened to share one
        # transform -- an FMO on asinh cofactor 5 beside a stained sample on
        # cofactor 150 called 5.36% positive where 50.52% is right, and an
        # FMO of unknown scale (an older OpenFlo CSV) 0.0%.
        stained = ed._samples[ed._active_sample]
        st_rec = getattr(stained, 'data_transforms', None) or {}
        parent_chs = set()
        if pid is not None:
            for g in ed._gates.values():
                parent_chs |= gate_channels(g)
        import copy as _copy
        parent_gates = _copy.deepcopy(ed._gates) if pid is not None else {}
        pairs = self._pairs()           # (channel, FMO name, FMO sample)
        if pairs:
            # Remembered (and saved with the session) so histograms of these
            # channels can show the FMO beside the stained sample.
            ed._set_fmo_controls({ch: fmo for ch, fmo, _fs in pairs})
        if not pairs:
            ed.status_var.set("No FMO mappings chosen — nothing added.")
            return
        self._busy.run(
            lambda job: self._cutoffs(job, pairs, pid, parent_gates,
                                      parent_chs, stained, st_rec, pct),
            on_done=lambda res: self._add_gates(res, pid, pct, stained.name),
            on_error=lambda exc: ed.status_var.set(
                f"FMO gating failed: {type(exc).__name__}: {exc}"),
            on_cancel=lambda: ed.status_var.set(
                "FMO gating cancelled; nothing added."),
            text="Reading the FMO controls…", disable=[self._apply_btn])

    @staticmethod
    def _cutoffs(job, pairs, pid, parent_gates, parent_chs, stained, st_rec,
                 pct):
        """Each mapped channel's cutoff in the stained sample's display
        coordinates, off the window thread: ``(cuts, refused)``."""
        from .gating_helpers import fmo_threshold
        from .pipeline import (
            UnknownScaleError,
            is_unknown_spec,
            linear_values,
            transform_params,
            transform_values,
        )
        inside = {}                     # FMO sample -> its in-parent rows
        refused = []                    # "channel (why)" for the status
        cuts = []
        for i, (ch, fmo, fs) in enumerate(pairs):
            job.progress(i, len(pairs), f"Reading {fmo} for {ch}…")
            try:
                rows = np.ones(len(fs.data), dtype=bool)
                if pid is not None:
                    if fmo not in inside:
                        # The parent gate lives in the STAINED sample's
                        # coordinates: read the FMO through it there.
                        from .pipeline import cumulative_gate_mask
                        frame, bad = _in_stained_coordinates(fs, stained)
                        blocked = sorted(set(bad) & parent_chs)
                        if blocked:
                            raise UnknownScaleError(
                                f"{', '.join(blocked)} of '{fmo}' is on an "
                                "unknown scale, so the parent gate cannot "
                                "be read on it")
                        inside[fmo] = np.asarray(
                            cumulative_gate_mask(parent_gates, pid, frame),
                            dtype=bool)
                    rows = inside[fmo]
                lin = linear_values(fs, ch, fs.data[ch].to_numpy(float)[rows])
                spec = st_rec.get(ch)
                if is_unknown_spec(spec):
                    raise UnknownScaleError(
                        f"'{ch}' in the stained sample '{stained.name}' "
                        "holds values on an unknown display scale")
                cut = fmo_threshold(lin, percentile=pct)
                if spec:
                    cut = float(transform_values(
                        np.array([cut]), **transform_params(spec))[0])
            except UnknownScaleError as exc:
                refused.append(f"{ch} from {fmo} ({exc})")
                continue
            except Exception as exc:
                print(f"[fmo] {ch}: {exc}", flush=True)
                continue
            cuts.append((ch, cut))
        return cuts, refused

    def _add_gates(self, res, pid, pct, stained_name):
        """Place the gates `_cutoffs` found (window thread)."""
        cuts, refused = res
        ed = self._editor
        if ed._active_sample != stained_name:
            ed.status_var.set(f"The active sample changed from {stained_name}"
                              " while the FMO controls were read; no FMO "
                              "gate added.")
            return
        if pid is not None and pid not in ed._gates:
            ed.status_var.set("The selected parent gate is gone; no FMO gate "
                              "added.")
            return
        added = 0
        for ch, cut in cuts:
            ed._add_gate({'kind': 'threshold', 'channel': ch, 'value': cut,
                          'parent_id': pid, 'color': '#d62728',
                          'enabled': True})
            added += 1
        not_gated = (f"  [!] Not gated, scale unknown: {'; '.join(refused)}."
                     if refused else '')
        if added:
            ed._refresh_gate_list()
            ed._schedule_replot(0)
            under = ''
            if pid is not None:
                from .gating import population_path
                under = " under " + population_path(
                    ed._gates, pid,
                    transforms=ed._display_transforms(ed._active_sample))
            ed.status_var.set(
                f"Added {added} FMO threshold gate(s) at the {pct:g}th "
                f"percentile to {ed._active_sample}{under}." + not_gated)
            self.destroy()
        elif refused:
            ed.status_var.set("No FMO gate added." + not_gated)
        else:
            ed.status_var.set("No FMO mappings chosen — nothing added.")
