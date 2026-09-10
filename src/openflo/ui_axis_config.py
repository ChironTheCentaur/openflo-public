"""Per-axis scale/range configuration dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .ui_tips import auto_tip

# What each scale is FOR. The names are jargon; the choice changes where events
# appear, so it deserves a sentence rather than an expansion of the label.
_SCALE_HELP = {
    'linear': "Even spacing. Right for scatter (FSC/SSC) and any channel "
              "without a large dynamic range.",
    'log': "Log spacing for wide-range fluorescence. Cannot show zero or "
           "negative values, which compensated data routinely contains -- "
           "those events drop off the plot.",
    'symlog': "Log for large values but linear near zero, so negatives and "
              "zeros stay visible. A safer default than log for compensated "
              "channels.",
    'logicle': "The flow-cytometry standard: log-like at high intensity, "
               "linear through zero. Keeps the negative population as a "
               "single tight blob instead of smearing it down the axis.",
    'asinh': "Inverse hyperbolic sine -- log-like when large, linear near "
             "zero. Similar intent to logicle with one tunable parameter.",
}


class AxisConfigDialog(tk.Toplevel):
    """Tiny modal: pick scale (linear / log) and optionally a fixed
    (min, max) display range for the given channel. Calls
    ``on_apply(scale_str, range_tuple_or_None, link_bool)`` when the user hits OK.
    (Symlog is backend-only and not offered here.)
    """

    def __init__(self, parent, channel, scale, rng, on_apply, show_link=False):
        super().__init__(parent)
        self.title(f"Axis: {channel}")
        self.transient(parent)
        self.resizable(False, False)
        self.on_apply = on_apply
        self.link_var = tk.BooleanVar(value=False)
        self._show_link = show_link

        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)

        ttk.Label(body, text=f"Channel: {channel}",
                  font=('TkDefaultFont', 9, 'bold')).grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        # Scale radios.
        ttk.Label(body, text="Scale:").grid(row=1, column=0,
                                            sticky='e', padx=(0, 6))
        # Symlog is intentionally not offered here (backend-only — its density
        # binning is artefact-prone on some scatter views). A channel that
        # still carries a legacy 'symlog' scale shows as Log in the picker.
        self.scale_var = tk.StringVar(
            value=scale if scale in ('linear', 'log') else 'log')
        for i, (lbl, val) in enumerate([('Linear', 'linear'),
                                        ('Log',    'log')]):
            _rb = ttk.Radiobutton(body, text=lbl, value=val,
                                  variable=self.scale_var)
            _rb.grid(row=1, column=1 + i, sticky='w', padx=(0, 8))
            auto_tip(_rb, _SCALE_HELP.get(val, f"Use the {lbl} scale."))

        # Range section.
        self.auto_var = tk.BooleanVar(value=(rng is None))
        _ar = ttk.Checkbutton(body, text="Auto-range",
                              variable=self.auto_var,
                              command=self._toggle_range)
        auto_tip(_ar,
                 "Fit the axis to the data on every replot. Turn OFF and set "
                 "min/max to hold one range across samples -- necessary when "
                 "plots are compared side by side, since an auto range "
                 "silently rescales each one.")
        _ar.grid(
            row=2, column=0, columnspan=4, sticky='w', pady=(8, 2))

        ttk.Label(body, text="Min:").grid(row=3, column=0,
                                          sticky='e', padx=(0, 6))
        self.min_var = tk.StringVar(value=(f"{rng[0]:g}" if rng else ''))
        self.min_entry = ttk.Entry(body, textvariable=self.min_var, width=12)
        self.min_entry.grid(row=3, column=1, sticky='w', padx=(0, 12))
        auto_tip(self.min_entry,
                 "Lower axis limit, used when Auto-range is off. On a "
                 "log scale this must be positive.")

        ttk.Label(body, text="Max:").grid(row=3, column=2,
                                          sticky='e', padx=(0, 6))
        self.max_var = tk.StringVar(value=(f"{rng[1]:g}" if rng else ''))
        self.max_entry = ttk.Entry(body, textvariable=self.max_var, width=12)
        self.max_entry.grid(row=3, column=3, sticky='w')
        auto_tip(self.max_entry,
                 "Upper axis limit, used when Auto-range is off.")
        self._toggle_range()

        # Link X & Y — apply this scale + range to both axes at once.
        if show_link:
            _lk = ttk.Checkbutton(
                body, text="Link X & Y (apply these settings to both axes)",
                variable=self.link_var)
            auto_tip(_lk,
                     "Apply this scale and range to BOTH axes. Useful for "
                     "FSC-A/FSC-H or any pair meant to be compared on equal "
                     "footing.")
            _lk.grid(
                row=4, column=0, columnspan=4, sticky='w', pady=(8, 0))

        # Buttons.
        bot = ttk.Frame(self, padding=(12, 0, 12, 12))
        bot.pack(fill='x')
        ttk.Button(bot, text="Cancel",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="OK",
                   command=self._on_ok).pack(side='right', padx=(0, 6))

        # Status line for parse errors.
        self.err_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.err_var,
                  foreground='red', padding=(12, 0, 12, 4)).pack(
            side='bottom', fill='x')

    def _toggle_range(self):
        state = 'disabled' if self.auto_var.get() else 'normal'
        self.min_entry.configure(state=state)
        self.max_entry.configure(state=state)

    def _on_ok(self):
        rng = None
        if not self.auto_var.get():
            try:
                lo = float(self.min_var.get().strip())
                hi = float(self.max_var.get().strip())
            except ValueError:
                self.err_var.set("Min/Max must be numbers.")
                return
            if not (lo < hi):
                self.err_var.set("Min must be < Max.")
                return
            if self.scale_var.get() == 'log' and lo <= 0:
                self.err_var.set("Log scale needs Min > 0.")
                return
            rng = (lo, hi)
        try:
            self.on_apply(self.scale_var.get(), rng, bool(self.link_var.get()))
        except Exception as exc:
            self.err_var.set(f"Apply failed: {exc}")
            return
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# COMPENSATION MATRIX EDITOR
# ══════════════════════════════════════════════════════════════════════════════
