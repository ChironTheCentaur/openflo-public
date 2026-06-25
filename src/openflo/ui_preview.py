"""
ui_preview.py — Quick single-sample QC preview dialog.

A lightweight :class:`tk.Toplevel` that lets the user eyeball the *raw* data
of a single FCS file BEFORE running it through the full pipeline. It wraps the
existing single-sample helpers in :mod:`openflo.preview`:

  • ``density_scatter``   — KDE-coloured scatter of two channels
  • ``add_threshold_lines`` — optional X/Y quadrant gate lines
  • ``annotate_quadrants``  — per-quadrant % labels when both thresholds set

The file is read with :class:`openflo.pipeline.FlowSample` (``.data`` /
``.channel_names``) so the channel dropdowns reflect the actual FCS detectors.

This dialog is a child of the View/Gate editor window. It expects the editor to
provide:
  • ``status_var`` (``tk.StringVar``)   — shared status-bar text
  • ``_dark_figs``  (``tk.BooleanVar``) — "Dark figures in pop-ups" toggle
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import preview as _preview


class QuickPreviewDialog(tk.Toplevel):
    """Raw-data QC scatter for one FCS file.

    Pick a file → choose X/Y channels → Plot a density scatter, optionally with
    quadrant threshold lines. Export the figure to PNG/SVG/PDF.
    """

    def __init__(self, editor):
        super().__init__(editor)
        self._editor = editor
        self._sample = None          # openflo.pipeline.FlowSample
        self._fcs_path = None

        self.title("Quick preview — raw QC scatter")
        self.geometry("760x680")
        try:
            self.transient(editor)
        except Exception:
            pass

        self._build_ui()

    # ── status helper ───────────────────────────────────────────────────────
    def _status(self, msg):
        sv = getattr(self._editor, 'status_var', None)
        if sv is not None:
            try:
                sv.set(msg)
            except Exception:
                pass

    # ── layout ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side='top', fill='x')

        # File row
        filerow = ttk.Frame(top)
        filerow.pack(fill='x', pady=(0, 6))
        ttk.Button(filerow, text="Pick FCS…",
                   command=self._pick_file).pack(side='left')
        self._file_var = tk.StringVar(value="(no file loaded)")
        ttk.Label(filerow, textvariable=self._file_var,
                  foreground='grey').pack(side='left', padx=8)

        # Channel / threshold row
        ctl = ttk.Frame(top)
        ctl.pack(fill='x')

        ttk.Label(ctl, text="X").grid(row=0, column=0, sticky='w', padx=(0, 2))
        self._x_combo = ttk.Combobox(ctl, state='readonly', width=18)
        self._x_combo.grid(row=0, column=1, padx=(0, 10))

        ttk.Label(ctl, text="Y").grid(row=0, column=2, sticky='w', padx=(0, 2))
        self._y_combo = ttk.Combobox(ctl, state='readonly', width=18)
        self._y_combo.grid(row=0, column=3, padx=(0, 10))

        ttk.Label(ctl, text="X thresh").grid(row=1, column=0, sticky='w',
                                             padx=(0, 2), pady=(6, 0))
        self._xthr_var = tk.StringVar(value='')
        ttk.Entry(ctl, textvariable=self._xthr_var, width=10).grid(
            row=1, column=1, sticky='w', pady=(6, 0))

        ttk.Label(ctl, text="Y thresh").grid(row=1, column=2, sticky='w',
                                             padx=(0, 2), pady=(6, 0))
        self._ythr_var = tk.StringVar(value='')
        ttk.Entry(ctl, textvariable=self._ythr_var, width=10).grid(
            row=1, column=3, sticky='w', pady=(6, 0))

        ttk.Label(ctl, text="(blank = no gate line)", foreground='grey').grid(
            row=1, column=4, sticky='w', padx=8, pady=(6, 0))

        # Button row
        btns = ttk.Frame(top)
        btns.pack(fill='x', pady=(8, 0))
        self._plot_btn = ttk.Button(btns, text="Plot", command=self._plot,
                                    state='disabled')
        self._plot_btn.pack(side='left')
        self._export_btn = ttk.Button(btns, text="Export…",
                                      command=self._export, state='disabled')
        self._export_btn.pack(side='left', padx=6)
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        # Figure
        self._fig = Figure(figsize=(6.2, 5.2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_title("Pick an FCS file, then Plot")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(side='top', fill='both', expand=True,
                                          padx=8, pady=8)
        try:
            self._canvas.draw()
        except Exception:
            pass

    # ── file load ───────────────────────────────────────────────────────────
    def _pick_file(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select an FCS file",
            filetypes=[('FCS files', '*.fcs'), ('All files', '*.*')])
        if not path:
            return
        self._load_sample(path)

    def _load_sample(self, path):
        try:
            from .pipeline import FlowSample
            sample = FlowSample(path)
        except Exception as exc:
            self._status(f"Quick preview: load failed: {exc}")
            messagebox.showerror(
                "Load failed",
                f"Could not read FCS file:\n{type(exc).__name__}: {exc}",
                parent=self)
            return

        if sample.data is None or sample.data.empty:
            messagebox.showwarning(
                "Empty sample", "That FCS file has no events to plot.",
                parent=self)
            return

        self._sample = sample
        self._fcs_path = path

        # Prefer numeric data columns; fall back to declared channel names.
        cols = [c for c in sample.data.columns]
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
        raw = (raw or '').strip()
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    # ── plot ────────────────────────────────────────────────────────────────
    def _plot(self):
        if self._sample is None:
            return
        xcol = self._x_combo.get()
        ycol = self._y_combo.get()
        df = self._sample.data
        if xcol not in df.columns or ycol not in df.columns:
            messagebox.showwarning(
                "Pick channels", "Choose valid X and Y channels first.",
                parent=self)
            return

        xthr = self._parse_thr(self._xthr_var.get())
        ythr = self._parse_thr(self._ythr_var.get())

        try:
            import numpy as np
            x = np.asarray(df[xcol].values, dtype=float)
            y = np.asarray(df[ycol].values, dtype=float)

            self._fig.clear()
            ax = self._fig.add_subplot(111)
            self._ax = ax

            _preview.density_scatter(ax, x, y, alpha=0.5)

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
            # Quadrant % labels only when both gate lines are present.
            if xthr is not None and ythr is not None:
                _preview.annotate_quadrants(ax, df, xcol, ycol, xthr, ythr)

            self._apply_dark_if_on()
            self._fig.tight_layout()
            self._canvas.draw()
            self._export_btn.config(state='normal')
            self._status(f"Quick preview: plotted {xcol} vs {ycol}")
        except Exception as exc:
            self._status(f"Quick preview: plot failed: {exc}")
            messagebox.showerror(
                "Plot failed", f"{type(exc).__name__}: {exc}", parent=self)

    # ── dark-mode theming ───────────────────────────────────────────────────
    def _dark_on(self):
        var = getattr(self._editor, '_dark_figs', None)
        try:
            return bool(var.get()) if var is not None else False
        except Exception:
            return False

    def _apply_dark_if_on(self):
        if not self._dark_on():
            return
        try:
            from .gui import _theme_figure_dark
            _theme_figure_dark(self._fig)
        except Exception:
            pass

    # ── export ──────────────────────────────────────────────────────────────
    def _export(self):
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
            from .gui import savefig_background
            background = 'Dark' if self._dark_on() else 'White'
            savefig_background(self._fig, path, background=background, dpi=300)
            import os
            self._status(f"Quick preview: saved → {os.path.basename(path)}")
        except Exception as exc:
            self._status(f"Quick preview: export failed: {exc}")
            messagebox.showerror(
                "Export failed", f"{type(exc).__name__}: {exc}", parent=self)
