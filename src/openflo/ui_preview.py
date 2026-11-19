"""
ui_preview.py — Quick single-sample QC preview dialog.

A lightweight :class:`tk.Toplevel` that lets the user eyeball the *raw* data
of a single FCS file BEFORE running it through the full pipeline. It wraps the
existing single-sample helpers in :mod:`openflo.preview`:

  • ``density_scatter``   — KDE-coloured scatter of two channels
  • ``add_threshold_lines`` — optional X/Y quadrant gate lines
  • ``annotate_quadrants``  — % labels for whichever thresholds are set

The file is read with :class:`openflo.flow_sample.FlowSample` (``.data`` /
``.channel_names``) so the channel dropdowns reflect the actual FCS detectors.

This dialog is a child of the View/Gate editor window. It expects the editor to
provide:
  • ``status_var`` (``tk.StringVar``)   — shared status-bar text
  • ``_theme_var`` (``tk.StringVar``) — the theme, whose plot background
    the preview follows
"""

import tkinter as tk
from tkinter import filedialog, ttk

from . import preview as _preview
from . import ui_errors
from .dialog_inputs import number_message, parse_number, read_number
from .paths import FCS_FILE_PATTERNS
from .tool_window import figure_panel
from .ui_errors import show_error
from .ui_tips import auto_tip


class QuickPreviewDialog(tk.Toplevel):
    """Raw-data QC scatter for one FCS file.

    Pick a file → choose X/Y channels → Plot a density scatter, optionally with
    quadrant threshold lines. Export the figure to PNG/SVG/PDF.
    """

    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor
        self._sample = None          # openflo.flow_sample.FlowSample
        self._fcs_path = None

        self.title("Quick preview — raw QC scatter")
        self.geometry("760x680")
        try:
            self.transient(editor)
        except tk.TclError:
            pass

        self._build_ui()

    # ── status helper ───────────────────────────────────────────────────────
    def _status(self, msg):
        sv = getattr(self._editor, 'status_var', None)
        if sv is not None:
            try:
                sv.set(msg)
            except tk.TclError:
                pass

    # ── layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side='top', fill='x')

        # File row
        filerow = ttk.Frame(top)
        filerow.pack(fill='x', pady=(0, 6))
        _pf = ttk.Button(filerow, text="Pick FCS…",
                         command=self._pick_file)
        _pf.pack(side='left')
        auto_tip(_pf,
                 "Open an .fcs file just to look at it -- it is not added to "
                 "the session and nothing here changes your gates.")
        self._file_var = tk.StringVar(value="(no file loaded)")
        ttk.Label(filerow, textvariable=self._file_var,
                  style='Muted.TLabel').pack(side='left', padx=8)

        # Channel / threshold row
        ctl = ttk.Frame(top)
        ctl.pack(fill='x')

        ttk.Label(ctl, text="X").grid(row=0, column=0, sticky='w', padx=(0, 2))
        self._x_combo = ttk.Combobox(ctl, state='readonly', width=18)
        self._x_combo.grid(row=0, column=1, padx=(0, 10))
        auto_tip(self._x_combo, "Channel on the X axis.")

        ttk.Label(ctl, text="Y").grid(row=0, column=2, sticky='w', padx=(0, 2))
        self._y_combo = ttk.Combobox(ctl, state='readonly', width=18)
        self._y_combo.grid(row=0, column=3, padx=(0, 10))
        auto_tip(self._y_combo, "Channel on the Y axis.")
        # A new channel redraws at once once a file is loaded (_plot
        # returns while none is).
        for combo in (self._x_combo, self._y_combo):
            combo.bind('<<ComboboxSelected>>',
                       lambda _e: self._plot(quiet=True))

        ttk.Label(ctl, text="X thresh").grid(row=1, column=0, sticky='w',
                                             padx=(0, 2), pady=(6, 0))
        self._xthr_var = tk.StringVar(value='')
        _xt = ttk.Entry(ctl, textvariable=self._xthr_var, width=10)
        _xt.grid(row=1, column=1, sticky='w', pady=(6, 0))
        auto_tip(_xt,
                 "Optional X threshold drawn on the preview, to try a cutoff "
                 "before committing to a gate. Leave blank for none.")

        ttk.Label(ctl, text="Y thresh").grid(row=1, column=2, sticky='w',
                                             padx=(0, 2), pady=(6, 0))
        self._ythr_var = tk.StringVar(value='')
        _yt = ttk.Entry(ctl, textvariable=self._ythr_var, width=10)
        _yt.grid(row=1, column=3, sticky='w', pady=(6, 0))
        auto_tip(_yt,
                 "Optional Y threshold drawn on the preview. With both set, "
                 "the quadrant percentages are shown.")

        ttk.Label(ctl, text="(blank = no gate line)", style='Muted.TLabel').grid(
            row=1, column=4, sticky='w', padx=8, pady=(6, 0))

        # Button row
        btns = ttk.Frame(top)
        btns.pack(fill='x', pady=(8, 0))
        self._plot_btn = ttk.Button(btns, text="Plot", command=self._plot,
                                    state='disabled')
        self._plot_btn.pack(side='left')
        auto_tip(self._plot_btn,
                 "Draw the preview with the current thresholds. A new X or "
                 "Y channel redraws by itself.")
        self._export_btn = ttk.Button(btns, text="Export figure…",
                                      command=self._export_figure, state='disabled')
        self._export_btn.pack(side='left', padx=6)
        auto_tip(self._export_btn,
                 "Save the preview plot as an image.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        # Figure — themed by _apply_dark_if_on() in the draw path, so dark=False.
        _frame, self._fig, self._canvas = figure_panel(
            self, figsize=(6.2, 5.2), dpi=100, dark=False,
            pack_kw={'side': 'top', 'fill': 'both', 'expand': True, 'padx': 8, 'pady': 8})
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title("Pick an FCS file, then Plot")
        try:
            self._canvas.draw()
        except Exception:  # noqa: BLE001 - preview canvas may already be gone
            pass

    # ── file load ───────────────────────────────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select an FCS file",
            filetypes=[('FCS files', FCS_FILE_PATTERNS), ('All files', '*.*')])
        if not path:
            return
        self._load_sample(path)

    def _load_sample(self, path):
        try:
            from .flow_sample import FlowSample
            sample = FlowSample(path)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is reported in the status line and an error dialog
            self._status(f"Quick preview: load failed: {exc}")
            show_error(self, "Load failed",
                       "The FCS file could not be read.", exc, path)
            return

        if sample.data is None or sample.data.empty:
            ui_errors.showwarning(
                "Empty sample", "That FCS file has no events to plot.",
                parent=self)
            return

        self._sample = sample
        self._fcs_path = path

        # Use the data columns; fall back to declared channel names if there
        # are none (no numeric filter — all columns are offered).
        cols = list(sample.data.columns)
        if not cols:
            cols = list(sample.channel_names)
        self._x_combo['values'] = cols
        self._y_combo['values'] = cols
        if cols:
            self._x_combo.set(cols[0])
            self._y_combo.set(cols[1] if len(cols) > 1 else cols[0])

        import os
        self._file_var.set(f"{os.path.basename(path)}  ({len(sample.data):,} events)")
        self._plot_btn.config(state='normal')
        self._status(f"Quick preview: loaded {os.path.basename(path)}")

    # ── threshold parsing ───────────────────────────────────────────────────
    @staticmethod
    def _parse_thr(raw):
        """A threshold as a finite number, or None (blank or not a number)."""
        raw = (raw or '').strip()
        return parse_number(raw) if raw else None

    def _read_thr(self, var, label, quiet=False):
        """``(ok, value)``: a blank box is no threshold (``(True, None)``);
        text that is not a finite number is refused with a message naming
        the box, ``(False, None)``, so nothing is drawn without the line the
        user asked for. ``quiet`` (a redraw the user did not ask for with
        Plot): the refusal goes to the status line, not a dialog."""
        raw = var.get() or ''
        if not raw.strip():
            return True, None
        if quiet:
            value = parse_number(raw)
            if value is None:
                self._status("Quick preview not redrawn: "
                             + number_message(label, raw))
            return value is not None, value
        value = read_number(var, label, parent=self,
                            title="Check the threshold")
        return value is not None, value

    # ── plot ────────────────────────────────────────────────────────────────
    def _plot(self, quiet=False):
        """Draw the preview. ``quiet`` (a new X or Y channel): a threshold
        box that is not a number leaves the last plot and says so on the
        status line instead of in a dialog."""
        if self._sample is None:
            return
        xcol = self._x_combo.get()
        ycol = self._y_combo.get()
        df = self._sample.data
        if xcol not in df.columns or ycol not in df.columns:
            ui_errors.showwarning(
                "Pick channels", "Choose valid X and Y channels first.",
                parent=self)
            return

        ok_x, xthr = self._read_thr(self._xthr_var, "X thresh", quiet)
        if not ok_x:
            return
        ok_y, ythr = self._read_thr(self._ythr_var, "Y thresh", quiet)
        if not ok_y:
            return

        try:
            import numpy as np
            x = np.asarray(df[xcol].values, dtype=float)
            y = np.asarray(df[ycol].values, dtype=float)

            self._fig.clear()
            ax = self._fig.add_subplot(111)
            self._ax = ax

            from .prefs import load_density_binned
            _preview.density_scatter(ax, x, y, alpha=0.5,
                                     binned=load_density_binned())

            def _disp(col):
                s = self._sample
                if s is None or not getattr(s, 'channel_labels', None):
                    return col
                return s.channel_labels.get(col, col)

            ax.set_xlabel(_disp(xcol))
            ax.set_ylabel(_disp(ycol))
            ax.set_title(self._sample.name)

            if xthr is not None or ythr is not None:
                _preview.add_threshold_lines(ax, xthr, ythr)
            # Percentages for whichever thresholds exist: four quadrants
            # when both gate lines are present, a two-way split when only
            # one is, nothing when neither. The helper owns that decision
            # so the CLI and the GUI cannot drift apart on it.
            _preview.annotate_quadrants(ax, df, xcol, ycol, xthr, ythr)

            self._apply_dark_if_on()
            self._fig.tight_layout()
            self._canvas.draw()
            self._export_btn.config(state='normal')
            self._status(f"Quick preview: plotted {xcol} vs {ycol}")
        except Exception as exc:  # noqa: BLE001 - a plot failure is reported in the status line and an error dialog
            self._status(f"Quick preview: plot failed: {exc}")
            show_error(self, "Plot failed", "The preview could not be drawn.",
                       exc)

    # ── dark-mode theming ───────────────────────────────────────────────────
    def _dark_on(self):
        try:
            from .theme import dialog_dark_on
            return dialog_dark_on(self)
        except Exception:  # noqa: BLE001 - dialog theming is cosmetic
            return False

    def _apply_dark_if_on(self):
        if not self._dark_on():
            return
        try:
            from .theme import theme_figure_dark
            theme_figure_dark(self._fig)
        except Exception:  # noqa: BLE001 - dialog theming is cosmetic
            pass

    # ── export ──────────────────────────────────────────────────────────────
    def _export_figure(self):
        if self._sample is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export preview image",
            defaultextension='.png',
            initialfile='openflo_preview.png',
            filetypes=[('PNG image', '*.png'), ('SVG vector', '*.svg'),
                       ('PDF', '*.pdf'), ('All files', '*.*')])
        if not path:
            return
        try:
            from .theme import savefig_background
            savefig_background(self._fig, path)
            import os
            self._status(f"Quick preview: saved → {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 - a failed export is reported in the status line and an error dialog
            self._status(f"Quick preview: export failed: {exc}")
            show_error(self, "Export failed",
                       "The preview could not be exported.", exc, path)
