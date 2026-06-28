"""Voltage-titration dialog for the OpenFlo editor.

A thin Tk front-end over :class:`openflo.voltage.VoltageTitration`. The user
points it at a folder of FCS files (one per PMT voltage of the same control),
picks a channel, and runs the Stain-Index voltage walk. The SI-vs-voltage
curve is embedded as a matplotlib figure and the recommended plateau voltage
is shown as text; the figure can be exported (respecting the editor's dark
preference).

The analysis runs in a background thread so the UI never freezes; results are
marshalled back onto the Tk thread with ``self.after(0, ...)``.

Open it from the editor with::

    from .ui_voltage import VoltageDialog
    VoltageDialog(self)
"""
from __future__ import annotations

import glob
import os
import tkinter as tk
from tkinter import filedialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .async_task import run_async
from .voltage import VoltageTitration

_FCS_GLOB = ('*.fcs', '*.FCS')


def _find_fcs(folder):
    """Sorted, de-duplicated list of FCS paths directly under `folder`."""
    seen, out = set(), []
    for pat in _FCS_GLOB:
        for p in glob.glob(os.path.join(folder, pat)):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                out.append(p)
    out.sort()
    return out


class VoltageDialog(tk.Toplevel):
    """Folder-driven voltage-titration / Stain-Index dialog.

    Constructed empty: ``VoltageDialog(editor)``. The user picks a folder and
    channel inside the dialog, so construction never blocks on a file picker.
    """

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title('Voltage optimization')
        self.transient(editor)
        self.resizable(True, True)

        self._folder = None
        self._paths = []
        self._result = None
        self._running = False
        self._canvas = None

        self._build_ui()
        self._set_status('Pick a folder of FCS files (a voltage-titration '
                         'series), then choose a channel.')

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill='both', expand=True)

        # Folder row
        top = ttk.Frame(outer)
        top.pack(fill='x')
        ttk.Button(top, text='Folder…', command=self._pick_folder).pack(
            side='left')
        self._folder_var = tk.StringVar(value='(no folder chosen)')
        ttk.Label(top, textvariable=self._folder_var).pack(
            side='left', padx=6)

        # Channel row
        mid = ttk.Frame(outer)
        mid.pack(fill='x', pady=(6, 0))
        ttk.Label(mid, text='Channel:').pack(side='left')
        self._channel_var = tk.StringVar()
        self._channel_box = ttk.Combobox(
            mid, textvariable=self._channel_var, state='disabled', width=28)
        self._channel_box.pack(side='left', padx=6)

        self._run_btn = ttk.Button(
            mid, text='Run', command=self._run, state='disabled')
        self._run_btn.pack(side='left', padx=(6, 0))

        # Figure
        self._fig = Figure(figsize=(7.0, 4.2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlabel('PMT voltage (V)')
        self._ax.set_ylabel('Stain Index')
        self._canvas = FigureCanvasTkAgg(self._fig, master=outer)
        self._canvas.get_tk_widget().pack(fill='both', expand=True, pady=6)
        self._maybe_theme()        # dark the initial empty plot under Midnight

        # Recommendation text
        self._rec_var = tk.StringVar(value='')
        ttk.Label(outer, textvariable=self._rec_var,
                  wraplength=620, justify='left').pack(fill='x')

        # Local status line (mirrors editor status; works even when detached)
        self._status_var = tk.StringVar(value='')
        ttk.Label(outer, textvariable=self._status_var, foreground='grey',
                  wraplength=620, justify='left').pack(fill='x', pady=(4, 0))

        # Buttons
        btns = ttk.Frame(outer)
        btns.pack(fill='x', pady=(6, 0))
        self._export_btn = ttk.Button(
            btns, text='Export…', command=self._export, state='disabled')
        self._export_btn.pack(side='left')
        ttk.Button(btns, text='Close', command=self.destroy).pack(side='right')

    # ── Status helpers ────────────────────────────────────────────────────

    def _set_status(self, msg):
        """Update both the dialog's own line and the editor's status bar."""
        try:
            self._status_var.set(msg)
        except Exception:
            pass
        try:
            self.editor.status_var.set(msg)
        except Exception:
            pass

    # ── Folder / channel selection ────────────────────────────────────────

    def _pick_folder(self):
        folder = filedialog.askdirectory(
            parent=self, title='Choose a folder of titration FCS files')
        if not folder:
            return
        paths = _find_fcs(folder)
        if not paths:
            self._folder = None
            self._paths = []
            self._folder_var.set('(no FCS files found)')
            self._channel_box.configure(values=(), state='disabled')
            self._channel_var.set('')
            self._run_btn.configure(state='disabled')
            self._set_status(f'No .fcs files directly under {folder}.')
            return
        self._folder = folder
        self._paths = paths
        self._folder_var.set(f'{os.path.basename(folder) or folder} '
                             f'({len(paths)} FCS)')
        self._populate_channels(paths[0])

    def _populate_channels(self, first_path):
        """Read one FCS to list its channels (detector + antibody label)."""
        try:
            from .pipeline import FlowSample
            sample = FlowSample(first_path)
        except Exception as exc:
            self._channel_box.configure(values=(), state='disabled')
            self._channel_var.set('')
            self._run_btn.configure(state='disabled')
            self._set_status(f'Could not read {os.path.basename(first_path)}: '
                             f'{exc}')
            return
        names = list(getattr(sample, 'channel_names', []) or [])
        labels = getattr(sample, 'channel_labels', {}) or {}
        # Prefer the fluor channels at the top; fall back to all names.
        fluor = list(getattr(sample, 'fluor_channels', []) or [])
        ordered = fluor + [n for n in names if n not in fluor]
        display, self._chan_lookup = [], {}
        for n in ordered:
            lbl = labels.get(n, n)
            shown = f'{n}  ({lbl})' if lbl and lbl != n else n
            display.append(shown)
            self._chan_lookup[shown] = n
        if not display:
            self._channel_box.configure(values=(), state='disabled')
            self._run_btn.configure(state='disabled')
            self._set_status('That FCS exposes no channels to titrate.')
            return
        self._channel_box.configure(values=display, state='readonly')
        # Default to the first fluor channel.
        self._channel_var.set(display[0])
        self._run_btn.configure(state='normal')
        self._set_status('Channel ready — click Run.')

    def _selected_channel(self):
        shown = self._channel_var.get()
        return self._chan_lookup.get(shown, shown) if shown else None

    # ── Run analysis (background thread) ──────────────────────────────────

    def _run(self):
        if self._running:
            return
        if not self._paths:
            self._set_status('Pick a folder of FCS files first.')
            return
        channel = self._selected_channel()
        if not channel:
            self._set_status('Choose a channel first.')
            return

        self._running = True
        self._run_btn.configure(state='disabled')
        self._export_btn.configure(state='disabled')
        try:
            self.editor._begin_busy('Voltage titration…')
        except Exception:
            pass

        paths = list(self._paths)
        run_async(self,
                  lambda: VoltageTitration.analyze(paths, channels=[channel]),
                  on_done=lambda result: self._on_done(result, channel),
                  on_error=self._on_fail,
                  on_finally=self._finish_busy)

    def _finish_busy(self):
        self._running = False
        self._run_btn.configure(state='normal')
        try:
            self.editor._end_busy()
        except Exception:
            pass

    def _on_fail(self, exc):
        self._set_status(f'Voltage analysis failed: {exc}')

    def _on_done(self, result, channel):
        self._result = result
        order = (result or {}).get('order') or []
        if not order:
            self._rec_var.set('')
            self._set_status(f'Channel {channel!r} was not found in any FCS '
                             'in that folder.')
            return
        self._draw(result)
        self._export_btn.configure(state='normal')

        parts = []
        for ch in order:
            res = result['results'][ch]
            rec = res.get('recommended_voltage')
            n_pts = len([r for r in res['rows'] if r['voltage'] is not None])
            if rec is not None:
                parts.append(f'{ch}: recommended {rec:g} V '
                             f'(>= {result.get("frac", 0.95) * 100:.0f}% of '
                             f'max SI, {n_pts} voltage point(s))')
            else:
                parts.append(f'{ch}: no recommendation (no usable SI / '
                             'voltages — is $PnV present in the FCS?)')
        self._rec_var.set('\n'.join(parts))
        self._set_status('Voltage titration done.')

    def _draw(self, result):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        VoltageTitration.plot(result, ax=ax)
        if not ax.has_data():
            ax.set_xlabel('PMT voltage (V)')
            ax.set_ylabel('Stain Index')
            ax.text(0.5, 0.5, 'No voltage points to plot\n'
                    '($PnV missing from FCS metadata?)',
                    ha='center', va='center', transform=ax.transAxes)
        self._ax = ax
        self._maybe_theme()
        self._fig.tight_layout()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _maybe_theme(self):
        """Apply the dark figure palette when pop-ups should be dark (the
        'Dark figures in pop-ups' toggle, or the Midnight theme), and match the
        Tk canvas widget background so no white margin shows around the plot."""
        try:
            from .theme import THEMES, _dialog_dark_on, _theme_figure_dark
            dark = _dialog_dark_on(self)
            if dark:
                _theme_figure_dark(self._fig)
            if self._canvas is not None:
                self._canvas.get_tk_widget().configure(
                    bg=(THEMES['midnight']['plot_bg'] if dark else 'white'),
                    highlightthickness=0)
        except Exception:
            pass

    # ── Export ────────────────────────────────────────────────────────────

    def _export(self):
        if self._result is None:
            self._set_status('Run an analysis before exporting.')
            return
        path = filedialog.asksaveasfilename(
            parent=self, title='Export Stain-Index plot',
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'),
                       ('SVG', '*.svg'), ('TIFF', '*.tiff')])
        if not path:
            return
        try:
            from .theme import _dialog_dark_on, savefig_background
            background = 'Dark' if _dialog_dark_on(self) else 'White'
            savefig_background(self._fig, path, background=background)
        except Exception as exc:  # noqa: BLE001 - report any failure
            self._set_status(f'Export failed: {exc}')
            return
        self._set_status(f'Saved plot → {os.path.basename(path)}')
