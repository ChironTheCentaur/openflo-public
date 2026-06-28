"""Preferences dialog.

Self-contained Tk window(s) extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .editor_loadpool import _default_pool_size
from .prefs import read_prefs, write_pref


class PreferencesDialog(tk.Toplevel):
    """One place for the scattered settings: theme + hover tips. (Both also
    remain on the View menu.)"""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Preferences")
        self.geometry("400x520")
        self._editor = editor
        frm = ttk.Frame(self)
        frm.pack(fill='both', expand=True, padx=16, pady=14)

        ttk.Label(frm, text="Appearance",
                  font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')
        trow = ttk.Frame(frm)
        trow.pack(anchor='w', pady=(4, 10))
        ttk.Label(trow, text="Theme:").pack(side='left')
        combo = ttk.Combobox(trow, textvariable=editor._theme_var,
                             state='readonly', width=22,
                             values=['light', 'dark', 'midnight'])
        combo.pack(side='left', padx=6)
        combo.bind('<<ComboboxSelected>>', lambda _e: editor._set_theme())

        ttk.Checkbutton(
            frm, text="Show hover tips & menu help",
            variable=editor._tooltips_enabled,
            command=lambda: write_pref(
                'tooltips', bool(editor._tooltips_enabled.get()))).pack(
            anchor='w', pady=2)

        ttk.Label(frm, text="Export", font=('TkDefaultFont', 9, 'bold')).pack(
            anchor='w', pady=(12, 0))
        self._prov_var = tk.BooleanVar(
            value=bool(read_prefs().get('export_provenance', True)))
        ttk.Checkbutton(
            frm, text="Stamp OpenFlo version on exported figures",
            variable=self._prov_var,
            command=lambda: write_pref(
                'export_provenance', bool(self._prov_var.get()))).pack(
            anchor='w', pady=2)

        ttk.Label(frm, text="Performance",
                  font=('TkDefaultFont', 9, 'bold')).pack(
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
        ttk.Label(
            frm, foreground='grey', wraplength=350, justify='left',
            text=("How many files load in parallel. Higher is faster on a "
                  "many-core / high-RAM machine but uses more memory. 'Auto' "
                  "picks from your CPU & RAM; a change applies to the next "
                  "load.")).pack(anchor='w', pady=(2, 0))

        # GPU acceleration (opt-in): compensation matmul + arcsinh transform.
        gpu_ok = False
        try:
            from . import gpu_accel
            gpu_ok = gpu_accel.gpu_available()
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
            frm, text="Use GPU acceleration (NVIDIA / CuPy)",
            variable=self._gpu_var, command=_on_gpu)
        gpu_cb.pack(anchor='w', pady=(8, 0))
        if not gpu_ok:
            gpu_cb.state(['disabled'])
            self._gpu_var.set(False)
        ttk.Label(
            frm, foreground='grey', wraplength=350, justify='left',
            text=(("Offloads compensation + arcsinh transform to the GPU "
                   "(~10x on large samples). "
                   if gpu_ok else
                   "No NVIDIA GPU / CuPy detected — `pip install cupy-cuda12x` "
                   "to enable. ")
                  + "GPU results differ microscopically (float32) from the CPU "
                    "path; a change applies to the next load.")).pack(
            anchor='w', pady=(2, 0))

        ttk.Label(
            frm, foreground='grey', wraplength=350, justify='left',
            text=("Window size/position and recent sessions are remembered "
                  "automatically.")).pack(anchor='w', pady=(10, 0))
        ttk.Button(frm, text="Close", command=self.destroy).pack(
            side='bottom', anchor='e')
