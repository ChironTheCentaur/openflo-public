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
    """Tiny modal: pick a scale (logicle with its width, arcsinh with its
    cofactor, log, linear) and optionally a fixed
    (min, max) display range for the given channel. Calls
    ``on_apply(scale_str, range_tuple_or_None, link_bool)`` when the user hits OK.
    The scale passed back is a scales.make_scale string ('logicle:w=1').
    (Symlog is backend-only and not offered here.)

    `rng` and the range passed back are in the units the axis is LABELLED
    in: linear intensity for a transformed (e.g. logicle) channel, which the
    caller converts at this boundary; `intensity` says so in the dialog.
    """

    def __init__(self, parent, channel, scale, rng, on_apply, show_link=False,
                 intensity=False):
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
                  font='OpenFloHeading').grid(
            row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))

        # Scale radios. Symlog is intentionally not offered (backend-only —
        # its density binning is artefact-prone on some scatter views); a
        # channel that still carries a legacy 'symlog' shows as Logicle.
        from .scales import (
            ASINH_DEFAULT_COFACTOR,
            LOGICLE_DEFAULTS,
            PICKABLE_SCALES,
            parse_scale,
        )
        kind, params = parse_scale(scale)
        picks = [k for k, _ in PICKABLE_SCALES]
        ttk.Label(body, text="Scale:").grid(row=1, column=0,
                                            sticky='e', padx=(0, 6))
        self.scale_var = tk.StringVar(
            value=kind if kind in picks else 'logicle')
        radios = ttk.Frame(body)
        radios.grid(row=1, column=1, columnspan=3, sticky='w')
        for val, lbl in PICKABLE_SCALES:
            _rb = ttk.Radiobutton(radios, text=lbl.split(' (')[0], value=val,
                                  variable=self.scale_var,
                                  command=self._toggle_param)
            _rb.pack(side='left', padx=(0, 8))
            auto_tip(_rb, _SCALE_HELP.get(val, f"Use the {lbl} scale."))

        # The one parameter of logicle (width W) and arcsinh (cofactor).
        # Changing it changes only how the axis is drawn: gates stay on the
        # same events, since they are kept in the channel's own coordinates.
        self.param_label = ttk.Label(body, text="")
        self.param_label.grid(row=2, column=0, sticky='e', padx=(0, 6),
                              pady=(6, 0))
        self.w_var = tk.StringVar(
            value=f"{params.get('w', LOGICLE_DEFAULTS['w']):g}")
        self.c_var = tk.StringVar(
            value=f"{params.get('c', ASINH_DEFAULT_COFACTOR):g}")
        self.param_entry = ttk.Spinbox(body, width=10)
        self.param_entry.grid(row=2, column=1, sticky='w', pady=(6, 0))
        self.param_hint = ttk.Label(body, text="", style='Muted.TLabel')
        self.param_hint.grid(row=2, column=2, columnspan=2, sticky='w',
                             padx=(8, 0), pady=(6, 0))
        self._toggle_param()

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
            row=3, column=0, columnspan=4, sticky='w', pady=(8, 2))

        ttk.Label(body, text="Min:").grid(row=4, column=0,
                                          sticky='e', padx=(0, 6))
        self.min_var = tk.StringVar(value=(f"{rng[0]:.6g}" if rng else ''))
        self.min_entry = ttk.Entry(body, textvariable=self.min_var, width=12)
        self.min_entry.grid(row=4, column=1, sticky='w', padx=(0, 12))
        auto_tip(self.min_entry,
                 "Lower axis limit, used when Auto-range is off. On a "
                 "log scale this must be positive.")

        ttk.Label(body, text="Max:").grid(row=4, column=2,
                                          sticky='e', padx=(0, 6))
        self.max_var = tk.StringVar(value=(f"{rng[1]:.6g}" if rng else ''))
        self.max_entry = ttk.Entry(body, textvariable=self.max_var, width=12)
        self.max_entry.grid(row=4, column=3, sticky='w')
        auto_tip(self.max_entry,
                 "Upper axis limit, used when Auto-range is off.")
        if intensity:
            # The channel is stored transformed; Min / Max are what the axis
            # labels say, not the stored coordinate.
            ttk.Label(body, text="Min / Max in intensity (linear units), as "
                                 "the axis is labelled.",
                      style='Muted.TLabel').grid(
                row=6, column=0, columnspan=4, sticky='w', pady=(4, 0))
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
                row=5, column=0, columnspan=4, sticky='w', pady=(8, 0))

        # Buttons.
        bot = ttk.Frame(self, padding=(12, 0, 12, 12))
        bot.pack(fill='x')
        ttk.Button(bot, text="Cancel",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="OK", default='active',
                   command=self._on_ok).pack(side='right', padx=(0, 6))

        # Status line for parse errors.
        self.err_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.err_var, style='Error.TLabel',
                  padding=(12, 0, 12, 4)).pack(
            side='bottom', fill='x')
        self.bind('<Return>', lambda _e: self._on_ok())
        self.bind('<Escape>', lambda _e: self.destroy())

    def _toggle_param(self):
        """Show the chosen scale's parameter (logicle W, arcsinh cofactor),
        or grey the row out for a scale that has none."""
        kind = self.scale_var.get()
        if kind == 'logicle':
            self.param_label.configure(text="Width (W):")
            self.param_entry.configure(textvariable=self.w_var, from_=0.0,
                                       to=2.0, increment=0.1,
                                       state='normal')
            self.param_hint.configure(
                text="decades of linear region around 0 (default 0.5)")
            auto_tip(self.param_entry,
                     "How much of the axis around zero is linear. Raise it "
                     "when a negative population is split or smeared into "
                     "two; lower it when it is squashed into a line.")
        elif kind == 'asinh':
            self.param_label.configure(text="Cofactor:")
            self.param_entry.configure(textvariable=self.c_var, from_=0.1,
                                       to=100000, increment=10,
                                       state='normal')
            self.param_hint.configure(
                text="~150 for fluorescence, ~5 for mass cytometry")
            auto_tip(self.param_entry,
                     "Intensity at which the axis turns from linear to "
                     "log-like. Larger values compress the area near zero.")
        else:
            self.param_label.configure(text="")
            self.param_entry.configure(state='disabled')
            self.param_hint.configure(text="")
            auto_tip(self.param_entry,
                     "This scale has no parameter to set.")

    def _scale_string(self):
        """The scale the dialog describes, parameter included (see
        scales.make_scale), or raise ValueError with a message to show."""
        from .scales import ASINH_C_RANGE, LOGICLE_W_RANGE, make_scale
        kind = self.scale_var.get()
        if kind == 'logicle':
            try:
                w = float(self.w_var.get().strip())
            except ValueError:
                raise ValueError("Width must be a number.") from None
            lo, hi = LOGICLE_W_RANGE
            if not (lo <= w <= hi):
                raise ValueError(f"Width must be between {lo:g} and {hi:g}.")
            return make_scale('logicle', w=w)
        if kind == 'asinh':
            try:
                c = float(self.c_var.get().strip())
            except ValueError:
                raise ValueError("Cofactor must be a number.") from None
            lo, hi = ASINH_C_RANGE
            if not (lo <= c <= hi):
                raise ValueError(
                    f"Cofactor must be between {lo:g} and {hi:g}.")
            return make_scale('asinh', c=c)
        return kind

    def _toggle_range(self):
        state = 'disabled' if self.auto_var.get() else 'normal'
        self.min_entry.configure(state=state)
        self.max_entry.configure(state=state)

    def _on_ok(self):
        try:
            scale = self._scale_string()
        except ValueError as exc:
            self.err_var.set(str(exc))
            return
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
            self.on_apply(scale, rng, bool(self.link_var.get()))
        except Exception as exc:
            self.err_var.set(f"Apply failed: {exc}")
            return
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# COMPENSATION MATRIX EDITOR
# ══════════════════════════════════════════════════════════════════════════════
