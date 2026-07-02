"""MESF calibration-bead dialog.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk


class CalibrationDialog(tk.Toplevel):
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
        ttk.Label(r1, text="Channel:").pack(side='left')
        self.chan_var = tk.StringVar()
        self.chan_cb = ttk.Combobox(r1, textvariable=self.chan_var, width=16,
                                    state='readonly')
        self.chan_cb.pack(side='left', padx=(2, 8))
        ttk.Label(r1, text="Peaks:").pack(side='left')
        self.npk_var = tk.StringVar(value='6')
        ttk.Spinbox(r1, from_=2, to=12, width=4,
                    textvariable=self.npk_var).pack(side='left', padx=(2, 8))
        ttk.Button(r1, text="Detect peaks",
                   command=self._detect).pack(side='left')

        ttk.Label(body, justify='left',
                  text="Peaks — one per line as  MFI <tab/comma> assigned "
                       "value (MESF/ABC from the bead lot):").pack(
            anchor='w', pady=(8, 2))
        self.txt = tk.Text(body, height=9, wrap='none')
        self.txt.pack(fill='x')

        r2 = ttk.Frame(body)
        r2.pack(fill='x', pady=(6, 0))
        ttk.Button(r2, text="Fit", command=self._fit).pack(side='left')
        self.result = ttk.Label(r2, text="", foreground='#333')
        self.result.pack(side='left', padx=(10, 0))

        r3 = ttk.Frame(body)
        r3.pack(fill='x', pady=(10, 0))
        ttk.Label(r3, text="Apply to channel:").pack(side='left')
        self.apply_var = tk.StringVar()
        self.apply_cb = ttk.Combobox(r3, textvariable=self.apply_var, width=16,
                                     state='readonly')
        self.apply_cb.pack(side='left', padx=(2, 8))
        ttk.Button(r3, text="Apply calibration → MESF: column",
                   command=self._apply).pack(side='left')

        self._sync_channels()
        try:
            self.grab_set()
        except Exception:
            pass

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

    def _detect(self):
        from .calibration import detect_bead_peaks
        s = self.editor._samples.get(self.bead_var.get())
        ch = self.editor._resolve_channel(self.chan_var.get())
        if s is None or not ch or ch not in s.data.columns:
            self.result.configure(text="Pick a bead sample + channel.")
            return
        try:
            n = max(2, int(self.npk_var.get()))
        except ValueError:
            n = 6
        peaks = detect_bead_peaks(s.data[ch].to_numpy(dtype=float), n_peaks=n)
        self.txt.delete('1.0', 'end')
        self.txt.insert('1.0', '\n'.join(f"{p:.1f}\t" for p in peaks))
        self.result.configure(text=f"Detected {len(peaks)} peaks — enter the "
                                   "MESF/ABC value after each.")

    def _parse(self):
        pairs = []
        for line in self.txt.get('1.0', 'end').splitlines():
            toks = [t for t in re.split(r'[,\s]+', line.strip()) if t]
            if len(toks) >= 2:
                try:
                    pairs.append((float(toks[0]), float(toks[1])))
                except ValueError:
                    continue
        return pairs

    def _fit(self):
        from .calibration import fit_mesf_calibration
        pairs = self._parse()
        if len(pairs) < 2:
            self.result.configure(text="Enter ≥2 peaks as 'MFI value'.")
            return
        mfi = [p[0] for p in pairs]
        known = [p[1] for p in pairs]
        try:
            self._cal = fit_mesf_calibration(mfi, known)
        except ValueError as exc:
            self.result.configure(text=str(exc))
            return
        c = self._cal
        self.result.configure(
            text=f"value = {c['slope']:.4g}·MFI + {c['intercept']:.4g}   "
                 f"(R²={c['r2']:.4f}, n={c['n']})")

    def _apply(self):
        from .calibration import apply_calibration
        if self._cal is None:
            self.result.configure(text="Fit a calibration first.")
            return
        ch = self.editor._resolve_channel(self.apply_var.get())
        if not ch:
            return
        label = self.editor._channel_labels.get(ch, ch)
        col = f'MESF:{label}'
        n_applied = 0
        for s in self.editor._samples.values():
            if ch in s.data.columns:
                s.data[col] = apply_calibration(
                    s.data[ch].to_numpy(dtype=float),
                    self._cal['slope'], self._cal['intercept'])
                n_applied += 1
        self.editor._refresh_channel_choices()
        self.editor._audit('calibration', channel=ch, column=col,
                           slope=round(self._cal['slope'], 4),
                           r2=round(self._cal['r2'], 4), n_samples=n_applied)
        self.result.configure(
            text=f"Applied to {n_applied} sample(s) → '{col}' "
                 "(now a plottable channel).")
