"""MESF calibration-bead dialog.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import messagebox, ttk

from .dialog_inputs import parse_number, read_int
from .scope_pick import ScopePickMixin
from .theme import current_palette
from .ui_logic import resolve_channel
from .ui_tips import auto_tip

BEAD_TIP = ("The calibration-bead acquisition -- a tube of beads "
            "with known assigned values, run on the same "
            "settings as the samples. Settings must match or the "
            "calibration does not transfer.")

# "Apply to" choices: (value, radio text).
SCOPES = (('batch', "this batch"), ('tab', "every sample in this tab"),
          ('bead', "bead sample only"))


class CalibrationDialog(ScopePickMixin, tk.Toplevel):
    """Fluorescence-intensity calibration to standardized units (MESF / ABC).

    Detect the bead peaks in a channel, paste each peak's assigned value from
    the bead datasheet, fit ``value = slope·MFI + intercept``, then apply it to
    a channel across all samples as a ``MESF:<marker>`` column."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Fluorescence calibration (MESF / ABC)")
        self.geometry("560x540")
        self._cal = None
        body = ttk.Frame(self, padding=10)
        body.pack(fill='both', expand=True)

        r1 = ttk.Frame(body)
        r1.pack(fill='x')
        ttk.Label(r1, text="Bead sample:").pack(side='left')
        self.bead_var = tk.StringVar(
            value=editor._active_sample or (editor._sample_order[0]
                                            if editor._sample_order else ''))
        bead_cb = ttk.Combobox(r1, textvariable=self.bead_var, width=20,
                               state='readonly', values=editor._sample_order)
        bead_cb.pack(side='left', padx=(2, 8))
        bead_cb.bind('<<ComboboxSelected>>', lambda *_: self._sync_channels())
        auto_tip(bead_cb, BEAD_TIP)
        ttk.Label(r1, text="Channel:").pack(side='left')
        self.chan_var = tk.StringVar()
        self.chan_cb = ttk.Combobox(r1, textvariable=self.chan_var, width=16,
                                    state='readonly')
        self.chan_cb.pack(side='left', padx=(2, 8))
        auto_tip(self.chan_cb,
                 "Channel to read the bead peaks in. Calibration is "
                 "per detector, so each channel needs its own fit.")
        # Peak count and Detect on their own row: on one row with the two
        # boxes they would run past the window's edge and not be shown.
        r1b = ttk.Frame(body)
        r1b.pack(fill='x', pady=(6, 0))
        ttk.Label(r1b, text="Peaks:").pack(side='left')
        self.npk_var = tk.StringVar(value='6')
        _sp = ttk.Spinbox(r1b, from_=2, to=12, width=4,
                          textvariable=self.npk_var)
        _sp.pack(side='left', padx=(2, 8))
        auto_tip(_sp,
                 "How many bead populations to find -- the number of peaks "
                 "this bead lot has, from its insert. Asking for more than "
                 "exist splits real peaks.")
        _db = ttk.Button(r1b, text="Detect peaks", command=self._detect)
        _db.pack(side='left')
        auto_tip(_db,
                 "Find the peak MFIs automatically and fill the table below. "
                 "Check them against the lot sheet before fitting -- a "
                 "missed or split peak shifts the whole curve.")

        ttk.Label(body, justify='left', wraplength=520,
                  text="Peaks — one per line as  MFI (compensated linear) "
                       "<tab/comma> assigned value (MESF/ABC from the bead "
                       "lot):").pack(
            anchor='w', pady=(8, 2))
        self.txt = tk.Text(body, height=9, wrap='none')
        self.txt.pack(fill='x')

        r2 = ttk.Frame(body)
        r2.pack(fill='x', pady=(6, 0))
        # Log-log is the bead-calibration standard.
        self.model_var = tk.StringVar(value='loglog')
        for text, val in (("Log-log", 'loglog'), ("Linear", 'linear')):
            rb = ttk.Radiobutton(r2, text=text, value=val,
                                 variable=self.model_var,
                                 command=self._model_changed)
            rb.pack(side='left', padx=(0, 6))
            auto_tip(rb, "Log-log fits log MESF against log MFI, the "
                         "standard for bead sets spanning decades; linear "
                         "fits MESF against MFI, decided by the brightest "
                         "peak.")
        _fb = ttk.Button(r2, text="Fit", command=self._fit)
        _fb.pack(side='left')
        auto_tip(_fb,
                 "Fit MFI against assigned value across the peaks. "
                 "The reported slope and R-squared say whether the "
                 "calibration is trustworthy: a poor R-squared "
                 "usually means a mis-assigned peak.")
        self.result = ttk.Label(r2, text="")
        self.result.pack(side='left', padx=(10, 0))

        r3 = ttk.Frame(body)
        r3.pack(fill='x', pady=(10, 0))
        ttk.Label(r3, text="Apply to channel:").pack(side='left')
        self.apply_var = tk.StringVar()
        self.apply_cb = ttk.Combobox(r3, textvariable=self.apply_var, width=16,
                                     state='readonly')
        self.apply_cb.pack(side='left', padx=(2, 8))
        auto_tip(self.apply_cb,
                 "Channel the fitted calibration is written to. "
                 "Usually the same channel it was fitted on.")
        _ab = ttk.Button(r3, text="Add MESF column",
                         command=self._apply)
        _ab.pack(side='left')
        auto_tip(_ab,
                 "Add a MESF: column converting this channel's arbitrary "
                 "units into standardized molecule counts, so intensities "
                 "can be compared across instruments and days. Asks "
                 "before replacing a MESF column a sample already has.")
        self.apply_cb.bind('<<ComboboxSelected>>',
                           lambda *_: self._refresh_scope())
        self._build_scope_row(body)

        self._sync_channels()
        try:
            self.grab_set()
        except tk.TclError:
            pass

    # ── which samples Add MESF column writes to ─────────────────────────
    def _build_scope_row(self, body):
        """Apply to: this batch / every sample in this tab / bead sample
        only, with who the batch holds (and on hover, why each other
        sample is left out)."""
        self._scope_pick = None       # the user's choice, once made
        self._setting_scope = False
        self.scope_var = tk.StringVar(master=self, value='batch')
        self.scope_var.trace_add('write', self._scope_written)
        r4 = ttk.Frame(body)
        r4.pack(fill='x', pady=(6, 0))
        ttk.Label(r4, text="Apply to:").pack(side='left')
        tips = {
            'batch': "The samples acquired like the bead sample (same "
                     "instrument, date and detector voltage on this "
                     "channel), in this tab. Read from the FCS keywords.",
            'tab': "Every sample this tab shows, acquired like the beads "
                   "or not; the bead sample only if the tab shows it. "
                   "Settings must match or the calibration does not "
                   "transfer.",
            'bead': "Only the bead sample, to check the column before "
                    "writing it into samples.",
        }
        self._scope_rbs = {}
        for value, text in SCOPES:
            rb = ttk.Radiobutton(r4, text=text, value=value,
                                 variable=self.scope_var)
            rb.pack(side='left', padx=(4, 0))
            auto_tip(rb, tips[value])
            self._scope_rbs[value] = rb
        self._batch_var = tk.StringVar(master=self, value='')
        lbl = ttk.Label(body, textvariable=self._batch_var,
                        foreground=current_palette()['muted'],
                        wraplength=520, justify='left')
        lbl.pack(anchor='w')
        self._batch_tip_text = ''
        auto_tip(lbl, lambda: self._batch_tip_text)

    def _batch(self, ch):
        """``(members, left_out)``: the bead sample's batch for channel
        `ch`, in the active tab (the editor's ``_batch_in_tab``, which
        compensation uses too); nothing without a bead sample or channel."""
        ed, bead = self.editor, self.bead_var.get()
        if bead not in ed._samples or not ch:
            return [], {}
        return ed._batch_in_tab(bead, [ch])

    def _refresh_scope(self):
        """Show the batch for the bead sample and target channel. The
        scope becomes "this batch" only while the user has not chosen one,
        and "bead sample only" when no other sample is in the batch (never
        replacing a pick of the whole tab)."""
        if not hasattr(self, 'scope_var'):
            return
        members, left_out = self._batch(resolve_channel(self.apply_var.get()))
        if len(members) > 1:
            shown = ', '.join(members[:6]) + (
                f' and {len(members) - 6} more' if len(members) > 6 else '')
            self._batch_var.set(f"Batch: {len(members)} samples: {shown}")
            self._scope_rbs['batch'].state(['!disabled'])
            self._set_scope(self._scope_pick or 'batch')
        else:
            self._batch_var.set(
                "Batch: no other sample was acquired like the bead sample")
            self._scope_rbs['batch'].state(['disabled'])
            if self.scope_var.get() == 'batch':
                self._set_scope('bead')
        why = '\n'.join(f"{n}: {r}" for n, r in list(left_out.items())[:15])
        more = len(left_out) - 15
        self._batch_tip_text = (
            "Same instrument, acquisition date and detector voltage on this "
            "channel (and stored spillover, when both have one)."
            + (f"\n\nLeft out:\n{why}" if why else '')
            + (f"\n…and {more} more" if more > 0 else ''))

    def _targets(self, ch):
        """The samples Add MESF column writes to for channel `ch` under the
        scope chosen now, read again at the click (never widened: the
        choice is not changed here), and the words for the status line."""
        ed, bead = self.editor, self.bead_var.get()
        scope = self.scope_var.get()
        if scope == 'tab':
            names, how = ed._in_tab(list(ed._sample_order)), "in this tab"
        elif scope == 'batch':
            names, how = self._batch(ch)[0], "in this batch"
        else:
            names, how = [bead], "(the bead sample)"
        names = [n for n in names if n in ed._samples
                 and ch in ed._samples[n].data.columns]
        return names, how

    def _fluor_channels(self):
        s = self.editor._samples.get(self.bead_var.get())
        if s is None:
            return []
        return [c for c in getattr(s, 'fluor_channels', [])
                if c in s.data.columns]

    def _sync_channels(self):
        chans = self._fluor_channels()
        disp = [self.editor._fmt_channel(c) for c in chans]
        self.chan_cb['values'] = disp
        self.apply_cb['values'] = disp
        if disp:
            self.chan_var.set(disp[0])
            self.apply_var.set(disp[0])
        self._refresh_scope()

    def _detect(self):
        from .calibration import detect_bead_peaks
        from .channel_transforms import UnknownScaleError, linear_values
        s = self.editor._samples.get(self.bead_var.get())
        ch = resolve_channel(self.chan_var.get())
        if s is None or not ch or ch not in s.data.columns:
            self.result.configure(text="Pick a bead sample + channel.")
            return
        n = read_int(self.npk_var, "Peaks", lo=2, hi=12,
                        parent=self)
        if n is None:
            return
        # MESF is linear in MFI only on the compensated LINEAR axis; the
        # loader's logicle values give peaks of 0.2-0.9, a fit with R² 0.59
        # and cells at 58,164 MESF where 6,000 is right. Peaks, fit and the
        # applied column are all on linear values.
        try:
            lin = linear_values(s, ch)
        except UnknownScaleError as exc:
            self.result.configure(text=f"Can't calibrate: {exc}")
            return
        peaks = detect_bead_peaks(lin, n_peaks=n)
        self.txt.delete('1.0', 'end')
        self.txt.insert('1.0', '\n'.join(f"{p:.1f}\t" for p in peaks))
        self.result.configure(text=f"Detected {len(peaks)} peaks — enter the "
                                   "MESF/ABC value after each.")

    def _parse(self):
        """The (MFI, value) pairs typed in the box."""
        return [(m, v) for _i, m, v in self._parse_lines()[0]]

    def _parse_lines(self):
        """``(pairs, left_out)``: ``(line number, MFI, value)`` for each line
        that holds two finite numbers, and a short note for each non-blank
        line that does not, so the fit can say which peaks it left out (a
        dropped bead peak shifts the whole curve)."""
        pairs, left_out = [], []
        for i, line in enumerate(self.txt.get('1.0', 'end').splitlines(), 1):
            toks = [t for t in re.split(r'[,\s]+', line.strip()) if t]
            if not toks:
                continue
            if len(toks) < 2:
                left_out.append(f"line {i} has no value")
                continue
            mfi, val = parse_number(toks[0]), parse_number(toks[1])
            if mfi is None or val is None:
                bad = toks[0] if mfi is None else toks[1]
                left_out.append(f"line {i}: “{bad}” is not a number")
                continue
            pairs.append((i, mfi, val))
        return pairs, left_out

    def _fit(self):
        from .calibration import fit_mesf_calibration
        pairs, left_out = self._parse_lines()
        if self.model_var.get() == 'loglog':
            # fit_mesf_calibration leaves these out of a log-log fit; say so.
            left_out += [f"line {i}: log-log needs values above 0"
                         for i, m, v in pairs if m <= 0 or v <= 0]
        note = f"  Left out: {'; '.join(left_out)}." if left_out else ''
        if len(pairs) < 2:
            self.result.configure(text="Enter ≥2 peaks as 'MFI value'."
                                  + note)
            return
        mfi = [p[1] for p in pairs]
        known = [p[2] for p in pairs]
        try:
            self._cal = fit_mesf_calibration(mfi, known,
                                             model=self.model_var.get())
        except ValueError as exc:
            self.result.configure(text=str(exc) + note)
            return
        c = self._cal
        import math
        if not math.isfinite(c['r2']):
            # Every bead was assigned the SAME value, so the fit carries no
            # information: the slope collapses to ~0 and every converted value
            # would come out as the intercept. Refuse rather than show a
            # number the user would reasonably read as a good fit.
            self._cal = None
            self.result.configure(
                text="All peaks were given the same value — the calibration "
                     "is degenerate (slope ≈ 0). Check the MESF/ABC column.")
            return
        if c['model'] == 'loglog':
            text = (f"log value = {c['slope']:.4g}·log MFI + "
                    f"{c['intercept']:.4g}   (R²={c['r2']:.4f}, n={c['n']})")
        else:
            text = (f"value = {c['slope']:.4g}·MFI + {c['intercept']:.4g}   "
                    f"(R²={c['r2']:.4f}, n={c['n']})")
        self.result.configure(text=text + note)

    def _model_changed(self):
        """Log-log / Linear after a fit: fit again with the model now
        picked, so the result line and Add MESF column never use the other
        one. A refit that fails leaves no fit."""
        if self._cal is None:
            return
        self._cal = None
        self._fit()

    def _apply(self):
        from .calibration import apply_calibration
        from .channel_transforms import UnknownScaleError, linear_values
        from .result_stamp import stamp_columns
        if self._cal is None:
            self.result.configure(text="Fit a calibration first.")
            return
        ch = resolve_channel(self.apply_var.get())
        if not ch:
            return
        label = self.editor._channel_labels.get(ch, ch)
        col = f'MESF:{label}'
        names, how = self._targets(ch)
        lin, unknown = {}, []  # no linear MFI exists for unknown: not applied
        for nm in names:
            try:
                lin[nm] = linear_values(self.editor._samples[nm], ch)
            except UnknownScaleError:
                unknown.append(nm)
        if not lin:
            self.result.configure(
                text=f"No sample {how} has {label} to calibrate."
                + (f"  Scale unknown: {', '.join(unknown)}." if unknown
                   else ''))
            return
        if not self._may_replace(col, lin):
            return
        for nm, values in lin.items():
            s = self.editor._samples[nm]
            s.data[col] = apply_calibration(
                values, self._cal['slope'], self._cal['intercept'],
                model=self._cal.get('model', 'linear'))
            stamp_columns(s, [col], 'calibration', [ch])
            self.editor._data_changed(nm)
        done = list(lin)
        self.editor._refresh_channel_choices()
        self.editor._audit('calibration', channel=ch, column=col,
                           slope=round(self._cal['slope'], 4),
                           model=self._cal.get('model', 'linear'),
                           r2=round(self._cal['r2'], 4), n_samples=len(done),
                           scope=self.scope_var.get(),
                           bead_sample=self.bead_var.get(), samples=done)
        shown = ', '.join(done[:6]) + ('…' if len(done) > 6 else '')
        self.result.configure(
            text=f"Applied to {len(done)} sample(s) {how} → '{col}' "
                 f"({shown}; now a plottable channel)."
                 + (f"  Not applied, scale unknown: {', '.join(unknown)}."
                    if unknown else ''))

    def _may_replace(self, col, targets):
        """Whether to write `col` into `targets`: asked (default No) when
        some already have one, which may come from their own beads; No
        applies nothing."""
        have = [n for n in targets
                if col in self.editor._samples[n].data.columns]
        if not have:
            return True
        shown = ', '.join(have[:4]) + ('…' if len(have) > 4 else '')
        if messagebox.askyesno(
                "Replace MESF column",
                f"Replace '{col}' in {len(have)} sample(s) ({shown}) with "
                "this calibration? Their column may come from their own "
                "beads.", default=messagebox.NO, parent=self):
            return True
        self.result.configure(text=f"Nothing applied: '{col}' kept in "
                                   f"{shown}.")
        return False
