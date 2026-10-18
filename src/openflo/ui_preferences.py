"""Preferences dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .editor_loadpool import _default_pool_size
from .prefs import read_prefs, write_pref
from .tool_window import scrolled_body
from .ui_tips import tip


class PreferencesDialog(tk.Toplevel):
    """One place for the scattered settings: theme, hover tips, export-provenance
    stamp, loader concurrency, and GPU acceleration. (Theme + hover tips also
    remain on the View menu.)"""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Preferences")
        self.geometry("420x560")
        self._editor = editor
        # Close first, at the bottom, so it always shows; the settings above
        # it scroll when the screen is too short for all of them.
        bar = ttk.Frame(self, padding=(16, 6, 16, 12))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        frm = scrolled_body(self, padding=(16, 14, 16, 0))

        ttk.Label(frm, text="Appearance",
                  font='OpenFloHeading').pack(anchor='w')
        trow = ttk.Frame(frm)
        trow.pack(anchor='w', pady=(4, 10))
        ttk.Label(trow, text="Windows:").pack(side='left')
        # Window colors and plot background are two choices, stored together
        # as one theme key (theme.theme_key), as the View menu does.
        from .theme import (
            CHROME_NAMES,
            PLOT_BACKGROUND_NAMES,
            theme_key,
            theme_parts,
        )
        chrome0, plot0 = theme_parts(editor._theme_var.get())
        self._chrome_name = tk.StringVar(value=CHROME_NAMES[chrome0])
        self._plot_name = tk.StringVar(value=PLOT_BACKGROUND_NAMES[plot0])
        combo = ttk.Combobox(trow, textvariable=self._chrome_name,
                             state='readonly', width=8,
                             values=list(CHROME_NAMES.values()))
        combo.pack(side='left', padx=6)
        ttk.Label(trow, text="Plot background:").pack(side='left',
                                                      padx=(10, 0))
        pcombo = ttk.Combobox(trow, textvariable=self._plot_name,
                              state='readonly', width=8,
                              values=list(PLOT_BACKGROUND_NAMES.values()))
        pcombo.pack(side='left', padx=6)
        self.theme_combos = (combo, pcombo)

        def _on_theme(_e=None):
            chrome = {v: k for k, v in CHROME_NAMES.items()}.get(
                self._chrome_name.get(), 'light')
            plot = {v: k for k, v in PLOT_BACKGROUND_NAMES.items()}.get(
                self._plot_name.get(), 'light')
            editor._theme_var.set(theme_key(chrome, plot))
            editor._set_theme()
        combo.bind('<<ComboboxSelected>>', _on_theme)
        pcombo.bind('<<ComboboxSelected>>', _on_theme)

        def _follow(*_a):
            c, p = theme_parts(editor._theme_var.get())
            self._chrome_name.set(CHROME_NAMES[c])
            self._plot_name.set(PLOT_BACKGROUND_NAMES[p])
        # Follow a theme picked from the View menu while this is open.
        _trace = editor._theme_var.trace_add('write', _follow)
        self.bind('<Destroy>', lambda e: (
            e.widget is self and editor._theme_var.trace_remove(
                'write', _trace)), add='+')
        tip(editor, combo,
            "Color of the windows, applied immediately.")
        tip(editor, pcombo,
            "Background of the plots in the main window and of the figures "
            "in analysis windows. Exports keep the background chosen when "
            "exporting.")

        _tips_cb = ttk.Checkbutton(
            frm, text="Show hover tips & menu help",
            variable=editor._tooltips_enabled,
            command=lambda: write_pref(
                'tooltips', bool(editor._tooltips_enabled.get())))
        _tips_cb.pack(anchor='w', pady=2)
        tip(editor, _tips_cb,
            "Master switch for these hover tips, in every window. Off hides "
            "them everywhere; the setting is remembered between sessions.")

        # Legend placement. matplotlib's automatic ("best") placement tests
        # candidate corners against every plotted point, so it costs ~1.1 s on
        # 8 samples x 50k events and runs on every replot.
        lrow = ttk.Frame(frm)
        lrow.pack(anchor='w', pady=2)
        ttk.Label(lrow, text="Legend position:").pack(side='left')
        self._legend_labels = {
            'cached': 'Automatic (reuse position)',
            'best':   'Automatic (recompute each time)',
            'fixed':  'Always top-right',
        }
        self._legend_var = tk.StringVar(
            value=self._legend_labels.get(editor._legend_placement.get(),
                                          self._legend_labels['cached']))
        lcombo = ttk.Combobox(lrow, textvariable=self._legend_var,
                              state='readonly', width=28,
                              values=list(self._legend_labels.values()))
        lcombo.pack(side='left', padx=6)

        def _on_legend(_e=None):
            rev = {v: k for k, v in self._legend_labels.items()}
            key = rev.get(self._legend_var.get(), 'cached')
            editor._legend_placement.set(key)
            editor._legend_pos_cache.clear()     # stale under the new mode
            write_pref('legend_placement', key)
            try:
                editor._replot()
            except Exception:                    # noqa: BLE001
                pass
        lcombo.bind('<<ComboboxSelected>>', _on_legend)
        tip(editor, lcombo,
            "Where the plot legend goes. Recomputing each time avoids covering "
            "data but tests every corner against every point (~1 s on 8 "
            "samples x 50k events, on every replot); reusing keeps the last "
            "solved position and only re-solves when the axes or samples "
            "change.")
        ttk.Label(
            frm, style='Muted.TLabel', wraplength=350, justify='left',
            text="Automatic placement avoids covering data but is costly on "
                 "big plots. Reusing the position keeps that placement and "
                 "only re-solves when the axes or sample set change.").pack(
            anchor='w', pady=(0, 6))

        ttk.Label(frm, text="Export", font='OpenFloHeading').pack(
            anchor='w', pady=(12, 0))
        self._prov_var = tk.BooleanVar(
            value=bool(read_prefs().get('export_provenance', True)))
        _prov_cb = ttk.Checkbutton(
            frm, text="Stamp OpenFlo version on exported figures",
            variable=self._prov_var,
            command=lambda: write_pref(
                'export_provenance', bool(self._prov_var.get())))
        _prov_cb.pack(anchor='w', pady=2)
        tip(editor, _prov_cb,
            "Writes the OpenFlo version into the corner of every exported "
            "figure, so a figure in a paper or slide deck can be traced back "
            "to the version that produced it.")
        self._editable_var = tk.BooleanVar(
            value=bool(read_prefs().get('export_editable_text', True)))
        _ed_cb = ttk.Checkbutton(
            frm, text="Keep text editable in PDF and SVG",
            variable=self._editable_var,
            command=lambda: write_pref(
                'export_editable_text', bool(self._editable_var.get())))
        _ed_cb.pack(anchor='w', pady=2)
        tip(editor, _ed_cb,
            "Saves labels as real text (TrueType in PDF, <text> in SVG), so "
            "Illustrator or Inkscape can edit them. Off draws each letter as "
            "a shape, which looks the same everywhere even without the font, "
            "but cannot be edited.")

        from .table_export import CSV_STYLES, csv_style
        crow = ttk.Frame(frm)
        crow.pack(anchor='w', pady=2)
        ttk.Label(crow, text="CSV tables:").pack(side='left')
        _csv_labels = dict(CSV_STYLES)
        self._csv_var = tk.StringVar(value=_csv_labels[csv_style()])
        ccombo = ttk.Combobox(
            crow, textvariable=self._csv_var, state='readonly', width=42,
            values=[lbl for _k, lbl in CSV_STYLES])
        ccombo.pack(side='left', padx=6)
        ccombo.bind('<<ComboboxSelected>>', lambda _e: write_pref(
            'csv_style', {lbl: k for k, lbl in CSV_STYLES}.get(
                self._csv_var.get(), 'standard')))
        tip(editor, ccombo,
            "How exported CSV tables and copied tables write numbers. Where "
            "Excel uses a semicolon and a decimal comma (most of Europe), a "
            "plain comma CSV opens as one column; the second choice writes "
            "this computer's own separator and decimal mark. Excel "
            "workbooks (.xlsx) open correctly either way.")

        from .theme import EXPORT_DPI_CHOICES, export_dpi
        drow = ttk.Frame(frm)
        drow.pack(anchor='w', pady=2)
        ttk.Label(drow, text="Export resolution:").pack(side='left')
        self._dpi_var = tk.StringVar(value=f"{export_dpi()} dpi")
        dcombo = ttk.Combobox(
            drow, textvariable=self._dpi_var, state='readonly', width=10,
            values=[f"{d} dpi" for d in EXPORT_DPI_CHOICES])
        dcombo.pack(side='left', padx=6)
        dcombo.bind('<<ComboboxSelected>>', lambda _e: write_pref(
            'export_dpi', int(self._dpi_var.get().split()[0])))
        tip(editor, dcombo,
            "Pixels per inch of saved and copied figures (PNG, TIFF, and the "
            "dense dots inside a PDF or SVG). 300 suits most journals; 600 or "
            "more for line art. The layout editor keeps its own setting.")

        frow = ttk.Frame(frm)
        frow.pack(anchor='w', pady=2)
        ttk.Label(frow, text="Figure font:").pack(side='left')
        from .theme import apply_figure_font, installed_figure_fonts
        default_label = "Default (DejaVu Sans)"
        fonts = [default_label] + [f for f in installed_figure_fonts()
                                   if f != 'DejaVu Sans']
        self._font_var = tk.StringVar(
            value=read_prefs().get('figure_font') or default_label)
        fcombo = ttk.Combobox(frow, textvariable=self._font_var,
                              state='readonly', width=22, values=fonts)
        fcombo.pack(side='left', padx=6)

        def _on_font(_e=None):
            name = self._font_var.get()
            write_pref('figure_font', '' if name == default_label else name)
            apply_figure_font()
            try:
                editor._replot()
            except Exception:                    # noqa: BLE001
                pass
        fcombo.bind('<<ComboboxSelected>>', _on_font)
        tip(editor, fcombo,
            "Font of the text in plots, on screen and in every export. Only "
            "fonts installed on this computer are listed; many journals ask "
            "for Arial or Helvetica.")

        self._layout_view_var = tk.BooleanVar(
            value=read_prefs().get('layout_drop_view', 'last') == 'current')
        _lv_cb = ttk.Checkbutton(
            frm, text="Layout: drop plots as the current graph is shown",
            variable=self._layout_view_var,
            command=lambda: write_pref(
                'layout_drop_view',
                'current' if self._layout_view_var.get() else 'last'))
        _lv_cb.pack(anchor='w', pady=2)
        tip(editor, _lv_cb,
            "A population dragged into the layout editor normally keeps the "
            "axes, plot type, axis scales and zoom it was last displayed "
            "with. On: it takes those of the graph showing right now "
            "instead.")

        ttk.Label(frm, text="Loading",
                  font='OpenFloHeading').pack(
            anchor='w', pady=(12, 0))
        from .prefs import load_qc_enabled
        self._qc_var = tk.BooleanVar(value=load_qc_enabled())
        _qc_cb = ttk.Checkbutton(
            frm, text="Run acquisition QC on load (drift, flow rate, "
                      "margin events)",
            variable=self._qc_var,
            command=lambda: write_pref('load_qc', bool(self._qc_var.get())))
        _qc_cb.pack(anchor='w', pady=2)
        tip(editor, _qc_cb,
            "Removes events in time bins where the signal drifts or the flow "
            "rate jumps (clogs, bubbles) and events piled at a channel's "
            "ceiling, before any gate. Off by default: FlowJo has no such "
            "step, so with it on the same gates count fewer events here than "
            "in FlowJo. Applies to files loaded from now on; a sample keeps "
            "the choice it was loaded with.")

        ttk.Label(frm, text="Performance",
                  font='OpenFloHeading').pack(
            anchor='w', pady=(12, 0))
        wrow = ttk.Frame(frm)
        wrow.pack(anchor='w', pady=2)
        ttk.Label(wrow, text="Concurrent file loaders:").pack(side='left')
        auto_n = _default_pool_size()
        cur = read_prefs().get('load_workers')
        self._workers_var = tk.StringVar(
            value=(f'Auto ({auto_n})' if cur is None else str(int(cur))))
        wcombo = ttk.Combobox(
            wrow, textvariable=self._workers_var, state='readonly', width=10,
            values=[f'Auto ({auto_n})'] + [str(i) for i in range(1, 9)])
        wcombo.pack(side='left', padx=6)

        def _on_workers(_e=None):
            v = self._workers_var.get()
            if v.startswith('Auto'):
                write_pref('load_workers', None)     # None ⇒ hardware default
            else:
                try:
                    write_pref('load_workers', max(1, min(8, int(v))))
                except ValueError:
                    pass
        wcombo.bind('<<ComboboxSelected>>', _on_workers)
        tip(editor, wcombo,
            "How many FCS files load in parallel. Higher is faster on a "
            "many-core machine but uses proportionally more RAM. 'Auto' picks "
            "from this machine's CPU and memory. Applies to the next load, "
            "not one already running.")
        ttk.Label(
            frm, style='Muted.TLabel', wraplength=350, justify='left',
            text=("How many files load in parallel. Higher is faster on a "
                  "many-core / high-RAM machine but uses more memory. 'Auto' "
                  "picks from your CPU & RAM; a change applies to the next "
                  "load.")).pack(anchor='w', pady=(2, 0))

        # GPU acceleration (opt-in): compensation matmul + arcsinh transform.
        # Backend auto-selects CuPy (NVIDIA) or the portable PyTorch backend
        # (NVIDIA/AMD/Intel/Apple, incl. DirectML on native Windows).
        gpu_ok = False
        gpu_dev = None
        try:
            from . import gpu_accel
            gpu_accel.set_backend(str(read_prefs().get('gpu_backend', 'auto')))
            gpu_ok = gpu_accel.gpu_available()
            gpu_dev = gpu_accel.device_name() if gpu_ok else None
        except Exception:
            gpu_ok = False
        self._gpu_var = tk.BooleanVar(
            value=bool(read_prefs().get('use_gpu', False)) and gpu_ok)

        def _on_gpu():
            on = bool(self._gpu_var.get())
            write_pref('use_gpu', on)
            try:
                from . import gpu_accel
                if on and not gpu_accel.set_enabled(True):
                    self._gpu_var.set(False)   # requested but no usable GPU
                elif not on:
                    gpu_accel.set_enabled(False)
            except Exception:
                pass
        gpu_cb = ttk.Checkbutton(
            frm, text="Use GPU acceleration",
            variable=self._gpu_var, command=_on_gpu)
        gpu_cb.pack(anchor='w', pady=(8, 0))
        tip(editor, gpu_cb, lambda: (
            "Runs compensation and the arcsinh transform on the GPU instead "
            "of the CPU (~10x on large samples). Results are unchanged. "
            + (f"Detected: {gpu_dev}." if gpu_dev else
               "Disabled because no usable GPU device was detected.")))
        if not gpu_ok:
            gpu_cb.state(['disabled'])
            self._gpu_var.set(False)

        # Backend picker. `gpu_backend` was read at startup and here, but
        # nothing ever wrote it — so the portable PyTorch backend
        # (AMD / Intel / Apple / DirectML) was reachable only through the
        # undocumented OPENFLO_GPU_BACKEND environment variable.
        brow = ttk.Frame(frm)
        brow.pack(anchor='w', pady=(4, 0))
        ttk.Label(brow, text="GPU backend:").pack(side='left')
        self._backend_labels = {
            'auto':  'Auto',
            'cupy':  'CuPy (NVIDIA)',
            'torch': 'PyTorch (NVIDIA/AMD/Intel/Apple)',
            'off':   'Off',
        }
        _cur_be = str(read_prefs().get('gpu_backend', 'auto')).lower()
        self._backend_var = tk.StringVar(
            value=self._backend_labels.get(_cur_be,
                                           self._backend_labels['auto']))
        bcombo = ttk.Combobox(brow, textvariable=self._backend_var,
                              state='readonly', width=30,
                              values=list(self._backend_labels.values()))
        bcombo.pack(side='left', padx=6)

        def _on_backend(_e=None):
            rev = {v: k for k, v in self._backend_labels.items()}
            key = rev.get(self._backend_var.get(), 'auto')
            write_pref('gpu_backend', key)
            try:
                from . import gpu_accel
                gpu_accel.set_backend(key)
                now_ok = gpu_accel.gpu_available()
                dev = gpu_accel.device_name() if now_ok else None
                self._backend_status.configure(
                    text=(f"resolved: {gpu_accel.backend_name()} — {dev}"
                          if now_ok else
                          "no usable device for that backend"))
                if not now_ok:
                    self._gpu_var.set(False)
                    gpu_accel.set_enabled(False)
                    gpu_cb.state(['disabled'])
                else:
                    gpu_cb.state(['!disabled'])
            except Exception:                          # noqa: BLE001
                self._backend_status.configure(text='backend probe failed')
        bcombo.bind('<<ComboboxSelected>>', _on_backend)
        tip(editor, bcombo,
            "Which GPU library to use. CuPy is NVIDIA-only and fastest there; "
            "PyTorch also reaches AMD, Intel and Apple GPUs (and any "
            "Direct3D-12 GPU on Windows via DirectML). 'Auto' tries CuPy "
            "first, then PyTorch. The line below shows what actually "
            "resolved.")
        self._backend_status = ttk.Label(frm, style='Muted.TLabel')
        self._backend_status.pack(anchor='w')
        if gpu_ok:
            try:
                from . import gpu_accel as _ga
                self._backend_status.configure(
                    text=f"resolved: {_ga.backend_name()} — {gpu_dev}")
            except Exception:                          # noqa: BLE001
                pass
        if gpu_ok and gpu_dev:
            avail = "Offloads compensation + arcsinh to the GPU "
            avail += f"(~10x on large samples). Detected: {gpu_dev}. "
        elif gpu_ok:
            avail = ("Offloads compensation + arcsinh to the GPU "
                     "(~10x on large samples). ")
        else:
            avail = ("No GPU backend detected. Install one: "
                     "`pip install openflo[gpu]` (NVIDIA/CuPy), "
                     "`[gpu-torch]` (NVIDIA/AMD/Intel/Apple via PyTorch), or "
                     "`[gpu-dml]` (any GPU on native Windows via DirectML). ")
        ttk.Label(
            frm, style='Muted.TLabel', wraplength=350, justify='left',
            text=(avail
                  + "GPU results differ microscopically (float32) from the CPU "
                    "path; a change applies to the next load.")).pack(
            anchor='w', pady=(2, 0))

        ttk.Label(
            frm, style='Muted.TLabel', wraplength=350, justify='left',
            text=("Window size/position and recent sessions are remembered "
                  "automatically.")).pack(anchor='w', pady=(10, 0))
