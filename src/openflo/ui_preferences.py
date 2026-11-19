"""Preferences dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from .editor_loadpool import _default_pool_size
from .errors import note
from .prefs import read_prefs, write_pref
from .tool_window import scrolled_body
from .ui_tips import tip

# The GeoMean and rCV forms are one global choice, not saved with a session
# (Sky's call, scope audit W3), so the tip says it reaches what is already
# shown, not only statistics computed later.
STAT_FORM_SCOPE = (
    " This choice is global and not saved with a session: changing it "
    "changes these numbers everywhere they are shown, statistics already "
    "on screen, layout stats boxes and exports included (on their next "
    "redraw), and a session opened later computes them with the choice "
    "set then.")


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

        self._build_appearance(frm, editor)
        self._build_export(frm, editor)
        self._build_statistics(frm, editor)
        self._build_pipeline_output(frm, editor)
        self._build_loading(frm, editor)
        self._build_updates(frm, editor)
        self._build_performance(frm, editor)
        gpu_ok, gpu_dev, gpu_cb = self._build_gpu(frm, editor)
        self._build_gpu_backend(frm, editor, gpu_ok, gpu_dev, gpu_cb)

        ttk.Label(
            frm, style='Muted.TLabel', wraplength=350, justify='left',
            text=("Window size/position and recent sessions are remembered "
                  "automatically.")).pack(anchor='w', pady=(10, 0))

    def _build_appearance(self, frm, editor):
        """Preferences > Appearance: window and plot theme, hover tips, legend position."""
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

        self._build_accessibility(frm, editor)

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
            except Exception:  # noqa: BLE001 - replot after a setting change is best-effort
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

    def _build_accessibility(self, frm, editor):
        """Preferences > Appearance: text size, colour-blind safe colours and
        high contrast, each applied at once."""
        from .prefs import (
            TEXT_SIZES,
            load_high_contrast,
            load_safe_colors,
            load_text_size,
        )
        srow = ttk.Frame(frm)
        srow.pack(anchor='w', pady=2)
        ttk.Label(srow, text="Text size:").pack(side='left')
        labels = {k: lbl for k, lbl, _pts in TEXT_SIZES}
        self._text_size_var = tk.StringVar(value=labels[load_text_size()])
        self.text_size_combo = scombo = ttk.Combobox(
            srow, textvariable=self._text_size_var, state='readonly',
            width=10, values=list(labels.values()))
        scombo.pack(side='left', padx=6)
        scombo.bind('<<ComboboxSelected>>',
                    lambda _e: self._text_size_changed())
        tip(editor, scombo,
            "Size of the text in every OpenFlo window: menus, buttons, "
            "lists and dialogs. Default follows the window width (10 pt in "
            "a wide window, 9 pt in a narrower one, never smaller); each "
            "step up adds 2 pt. Larger sizes need larger windows: some "
            "windows then scroll or wrap. Text in plots and exported "
            "figures does not change.")

        self._safe_colors_var = tk.BooleanVar(value=load_safe_colors())
        self.safe_colors_cb = ttk.Checkbutton(
            frm, text="Color-blind safe colors (gates, backgating, density)",
            variable=self._safe_colors_var,
            command=self._safe_colors_changed)
        self.safe_colors_cb.pack(anchor='w', pady=2)
        tip(editor, self.safe_colors_cb,
            "Draws the main window's plots in colors that stay distinct "
            "under red-green and blue-yellow color blindness: gates that "
            "have their automatic color, and backgated populations without "
            "one of their own, in the Okabe-Ito colors, and the density "
            "coloring in Viridis (it overrides View > Density colors while "
            "on). Gates keep their own colors underneath, so turning it off "
            "brings those back; a color you picked yourself is never "
            "changed. Export plot image and Copy plot save the plot as it "
            "is shown; layouts and FlowJo or Gating-ML exports keep each "
            "gate's own color. Off by default.")

        self._high_contrast_var = tk.BooleanVar(value=load_high_contrast())
        self.high_contrast_cb = ttk.Checkbutton(
            frm, text="High contrast (stronger text, borders and gate lines)",
            variable=self._high_contrast_var,
            command=self._high_contrast_changed)
        self.high_contrast_cb.pack(anchor='w', pady=2)
        tip(editor, self.high_contrast_cb,
            "Draws hints, headings and borders nearly as dark (or, on dark "
            "windows, as light) as the main text, and the gates on the main "
            "window's plots twice as thick and solid. Applies to both window "
            "colors. Export plot image and Copy plot save the plot as it "
            "is shown, thick gates included; layouts and other exports are "
            "not changed. Off by default.")

    def _text_size_changed(self):
        """Save the Text size choice and resize the windows' text now."""
        from .prefs import TEXT_SIZES
        key = {lbl: k for k, lbl, _pts in TEXT_SIZES}.get(
            self._text_size_var.get(), 'default')
        write_pref('text_size', key)
        try:
            self._editor._apply_chrome_scale(force=True)
        except Exception as exc:  # noqa: BLE001 - noted; the choice is saved and applies at the next start
            note("applying the text size", exc)

    def _safe_colors_changed(self):
        """Save the colour-blind safe colours choice and redraw."""
        write_pref('safe_colors', bool(self._safe_colors_var.get()))
        try:
            self._editor._schedule_replot(0)
            self._editor._refresh_gate_list()
        except Exception as exc:  # noqa: BLE001 - noted; the choice is saved and shows at the next redraw
            note("redrawing with the new colors", exc)

    def _high_contrast_changed(self):
        """Save the high contrast choice and re-apply the theme (which reads
        it) and redraw."""
        write_pref('high_contrast', bool(self._high_contrast_var.get()))
        try:
            self._editor._set_theme()
        except Exception as exc:  # noqa: BLE001 - noted; the choice is saved and applies at the next start
            note("applying high contrast", exc)

    def _build_export(self, frm, editor):
        """Preferences > Export: provenance stamp, editable text, CSV style,
        FCS channel names, resolution, figure font and layout drop view."""
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

        from .prefs import (
            FCS_CHANNEL_NAME_CHOICES,
            load_fcs_export_channel_names,
        )
        nrow = ttk.Frame(frm)
        nrow.pack(anchor='w', pady=2)
        ttk.Label(nrow, text="FCS export channel names:").pack(side='left')
        _name_labels = dict(FCS_CHANNEL_NAME_CHOICES)
        self._fcs_names_var = tk.StringVar(
            value=_name_labels[load_fcs_export_channel_names()])
        self.fcs_names_combo = ncombo = ttk.Combobox(
            nrow, textvariable=self._fcs_names_var, state='readonly',
            width=24, values=[lbl for _k, lbl in FCS_CHANNEL_NAME_CHOICES])
        ncombo.pack(side='left', padx=6)
        ncombo.bind('<<ComboboxSelected>>', lambda _e: write_pref(
            'fcs_export_channel_names',
            {lbl: k for k, lbl in FCS_CHANNEL_NAME_CHOICES}.get(
                self._fcs_names_var.get(), 'session')))
        tip(editor, ncombo,
            "The marker name each channel gets in an exported FCS file "
            "(populations, barcoded samples, normalized copies). Session "
            "names: a channel you relabeled here (for example from vial "
            "labels) or a template named carries that name in every file; "
            "every other channel keeps each file's own. Each file's original "
            "names: every file keeps the names it came with, as if you never "
            "relabeled. The data in the file is the same either way.")

        from .layout_page import EXPORT_DPI_CHOICES
        from .theme import export_dpi
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
            except Exception:  # noqa: BLE001 - replot after a setting change is best-effort
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

    def _build_statistics(self, frm, editor):
        """Preferences > Statistics: the GeoMean and rCV forms and the Westgard
        R-4s rule."""
        ttk.Label(frm, text="Statistics",
                  font='OpenFloHeading').pack(
            anchor='w', pady=(12, 0))
        from .prefs import GEOMEAN_FORMS, RCV_FORMS, load_stat_forms
        _forms = load_stat_forms()
        for key, pref, label, choices, tiptext in (
                ('geomean', 'stat_geomean', "GeoMean:", GEOMEAN_FORMS,
                 "How the GeoMean column is computed. FlowJo's form (the "
                 "default) averages each event's value on the channel's "
                 "display axis and converts back, so it counts the negative "
                 "events compensation produces, but it depends on that "
                 "axis: a dim population's GeoMean can differ by tens of "
                 "percent between two logicle settings. Samples imported "
                 "from a FlowJo workspace use the workspace's axis for each "
                 "channel it declares, so they match FlowJo; others use "
                 "OpenFlo's display axis. The GeoMean column heading names "
                 "a FlowJo axis when one was used. The textbook form, "
                 "exp(mean of log x), has no axis but leaves out every "
                 "event at or below zero, which makes a dim population read "
                 "high. Keep FlowJo's unless a method asks for the textbook "
                 "one."),
                ('rcv', 'stat_rcv', "rCV and rSD:", RCV_FORMS,
                 "How the robust spread is computed. The robust-statistics "
                 "form (the default), 1.4826 x MAD, tolerates up to half "
                 "the events being outliers. FlowJo's form is half the "
                 "15.87 to 84.13 percentile range; pick it to match "
                 "FlowJo. Both equal the SD for a normal population; on a "
                 "skewed one they differ (for example 74% against "
                 "FlowJo's 89%).")):
            row = ttk.Frame(frm)
            row.pack(anchor='w', pady=2)
            ttk.Label(row, text=label).pack(side='left')
            labels = dict(choices)
            var = tk.StringVar(value=labels[_forms[key]])
            setattr(self, f'_stat_{key}_var', var)
            combo = ttk.Combobox(row, textvariable=var, state='readonly',
                                 width=72, values=list(labels.values()))
            combo.pack(side='left', padx=6)
            if pref == 'stat_geomean':
                combo.bind('<<ComboboxSelected>>', lambda _e, v=var: write_pref(
                    'stat_geomean', {lbl: k for k, lbl in GEOMEAN_FORMS}.get(
                        v.get(), 'display')))
            else:
                combo.bind('<<ComboboxSelected>>', lambda _e, v=var: write_pref(
                    'stat_rcv', {lbl: k for k, lbl in RCV_FORMS}.get(
                        v.get(), 'mad')))
            # Then an open Statistics window recomputes with the new form
            # (editor_watch watches 'stat_forms').
            combo.bind('<<ComboboxSelected>>',
                       lambda _e: self._stat_form_changed(), add='+')
            tip(editor, combo, tiptext + STAT_FORM_SCOPE)

        from .prefs import load_westgard_r4s_across
        self._r4s_literature_var = tk.BooleanVar(
            value=not load_westgard_r4s_across())
        _r4s_cb = ttk.Checkbutton(
            frm, text="Westgard R-4s: literature form only (within run)",
            variable=self._r4s_literature_var,
            command=lambda: write_pref(
                'westgard_r4s_literature',
                bool(self._r4s_literature_var.get())))
        _r4s_cb.pack(anchor='w', pady=2)
        self.r4s_literature_box = _r4s_cb
        tip(editor, _r4s_cb,
            "Bead tracking's R-4s rule rejects a run whose range exceeds "
            "4 SD. Westgard's form looks within one run: one bead level "
            "(say the dim peak) beyond +2 SD and another (the bright peak) "
            "beyond -2 SD. Off (the default): OpenFlo also flags the same "
            "level beyond 2 SD on opposite sides in two runs in a row, so "
            "a log with one bead level per run still has a range check. "
            "On: only the within-run form, which needs two or more bead "
            "levels per run; a one-level log then never breaks R-4s. "
            "Leave it off unless your lab follows the published rules "
            "exactly.")

    def _build_loading(self, frm, editor):
        """Preferences > Loading: acquisition QC on load."""
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

    def _build_updates(self, frm, editor):
        """Preferences > Updates: the update check at start."""
        ttk.Label(frm, text="Updates",
                  font='OpenFloHeading').pack(anchor='w', pady=(12, 0))
        from .prefs import load_check_updates_on_start
        self._upd_var = tk.BooleanVar(value=load_check_updates_on_start())
        cb = ttk.Checkbutton(
            frm, text="Check for updates when OpenFlo starts",
            variable=self._upd_var,
            command=lambda: write_pref('check_updates_on_start',
                                       bool(self._upd_var.get())))
        cb.pack(anchor='w', pady=2)
        tip(editor, cb,
            "On (the default), OpenFlo asks GitHub for the newest release a "
            "few seconds after it starts, without slowing the start, and "
            "notes a newer one in the status bar. Nothing from your files is "
            "sent. Off, it only checks when you choose Help > Check for "
            "updates.")

    def _build_performance(self, frm, editor):
        """Preferences > Performance: concurrent loaders and the binned KDE
        density switch (the GPU rows follow in their own builders)."""
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

        from .prefs import load_density_binned
        self._density_binned_var = tk.BooleanVar(value=load_density_binned())

        def _on_density_binned():
            write_pref('density_binned', bool(self._density_binned_var.get()))
            try:
                editor._replot()
            except Exception:  # noqa: BLE001 - replot after a setting change is best-effort
                pass
        self.density_binned_cb = ttk.Checkbutton(
            frm, text="Faster Gaussian KDE density (binned, approximate)",
            variable=self._density_binned_var, command=_on_density_binned)
        self.density_binned_cb.pack(anchor='w', pady=(8, 0))
        tip(editor, self.density_binned_cb,
            "Speed against accuracy for the exact density coloring: the "
            "'True Gaussian KDE (slow)' plot option and the Quick preview "
            "scatter. Off (the default) computes the exact Gaussian KDE, "
            "which takes seconds per redraw on a large sample, and minutes "
            "in Quick preview, which uses every event. On spreads the events "
            "onto a fine grid and smooths it with the same kernel: "
            "milliseconds, with colors within about 0.5% of the exact "
            "ones. Leave it off when the exact density matters. The standard "
            "density plot is already fast and does not change.")

    def _build_gpu(self, frm, editor):
        """Performance: probe the GPU backend and add the GPU acceleration box.

        Returns ``(gpu_ok, gpu_dev, gpu_cb)``: the probe result, the device
        name, and the checkbox, which the backend picker reads and enables."""
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
        except Exception:  # noqa: BLE001 - noted; the GPU option is shown as unavailable
            note("detecting a GPU backend")
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
            except Exception as exc:  # noqa: BLE001 - noted; the preference is still saved even if the live switch fails
                note("switching GPU acceleration", exc)
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
        return gpu_ok, gpu_dev, gpu_cb

    def _build_gpu_backend(self, frm, editor, gpu_ok, gpu_dev, gpu_cb):
        """Performance: the GPU backend picker, its status line and the note
        under it, from the probe results ``_build_gpu`` returned."""
        # Backend picker. It writes `gpu_backend` (read at startup and here),
        # so the portable PyTorch backend (AMD / Intel / Apple / DirectML) is
        # reachable without the undocumented OPENFLO_GPU_BACKEND environment
        # variable.
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
            except Exception:                          # noqa: BLE001 - a backend probe can fail many ways (driver, import); the status label says it failed
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
            except Exception:  # noqa: BLE001 - backend status label is cosmetic
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

    def _stat_form_changed(self):
        """A GeoMean / rCV form was saved: tell the editor."""
        notify = getattr(self._editor, '_notify_changed', None)
        if notify is not None:
            notify()

    # Shown in the folder box while no folder is chosen.
    PIPELINE_OUTPUT_DEFAULT = "Beside the saved workspace (default)"

    def _build_pipeline_output(self, frm, editor):
        """Preferences > Pipeline workspace: the folder runs write to."""
        from tkinter import filedialog

        from .prefs import load_pipeline_output_dir
        ttk.Label(frm, text="Pipeline workspace",
                  font='OpenFloHeading').pack(anchor='w', pady=(12, 0))
        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x', pady=2)
        ttk.Label(row, text="Run outputs:").pack(side='left')
        self._pipeline_out_var = tk.StringVar(
            value=load_pipeline_output_dir() or self.PIPELINE_OUTPUT_DEFAULT)
        box = ttk.Entry(row, textvariable=self._pipeline_out_var,
                        state='readonly', width=30)
        box.pack(side='left', padx=6, fill='x', expand=True)

        def _choose():
            cur = load_pipeline_output_dir()
            d = filedialog.askdirectory(
                parent=self, title="Folder for Pipeline run outputs",
                initialdir=cur if cur and os.path.isdir(cur) else None)
            if d:
                write_pref('pipeline_output_dir', d)
                self._pipeline_out_var.set(d)

        def _default():
            write_pref('pipeline_output_dir', '')
            self._pipeline_out_var.set(self.PIPELINE_OUTPUT_DEFAULT)
        choose = ttk.Button(row, text="Choose…", command=_choose)
        choose.pack(side='left')
        reset = ttk.Button(row, text="Default", command=_default)
        reset.pack(side='left', padx=(4, 0))
        self.pipeline_output_controls = (box, choose, reset)
        what = ("Where each Pipeline run writes its run_<date>_<time> "
                "folder (cluster tables, embedding pictures, per-event "
                "files). Default: an 'outputs' folder beside the saved "
                "workspace file; before the workspace is saved, beside the "
                "session file; with neither, under the folder OpenFlo was "
                "started from. The status line and Results… show the "
                "folder each run used.")
        tip(editor, box, what)
        tip(editor, choose, "Pick one folder for every Pipeline run's "
            "outputs instead. " + what)
        tip(editor, reset, "Go back to the default: an 'outputs' folder "
            "beside the saved workspace.")
