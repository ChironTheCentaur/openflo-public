"""GUI dialog exposing the synthetic-dataset generators (``openflo.synthetic``)
and loading the written ``.fcs`` files straight into the editor.

``SyntheticDialog`` is a :class:`tkinter.Toplevel` child of the view/gate editor.
It offers a dataset-type selector mapped to the ``make_*`` generators, the
parameter fields each generator actually takes, and an output-directory picker
(default a temp folder under ``~/.openflo``). "Generate" runs the chosen
generator on a background thread and, on success, hands the written FCS paths to
the editor's ``_queue_fcs_loads`` and closes.

Editor contract (provided by ``gui.ViewGateEditorWindow``): ``status_var``,
``_begin_busy(msg)`` / ``_end_busy()`` and ``_queue_fcs_loads(paths)``.
"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import synthetic

# Dataset type → spec. Each spec lists which parameter fields to show and the
# callable that runs the generator (returns a flat list of written .fcs paths).
# The callables normalise the various return shapes of the make_* functions
# (some return paths, some (paths, csv), some a single fcs path) to a list.


def _immuno(out_dir, params):
    return synthetic.make_immunophenotyping_dataset(
        out_dir, donors=params['donors'], n=params['n'], seed=params['seed'])


def _diff(out_dir, params):
    return synthetic.make_differentiation_dataset(
        out_dir, reps=params['reps'], n=params['n'], seed=params['seed'])


def _cellcycle(out_dir, params):
    return synthetic.make_cell_cycle_dataset(
        out_dir, samples=params['samples'], n=params['n'], seed=params['seed'])


def _spectral(out_dir, params):
    paths, _controls = synthetic.make_spectral_dataset(
        out_dir, n=params['n'], seed=params['seed'])
    return paths


def _beads(out_dir, params):
    fcs, _csv = synthetic.make_calibration_beads(
        out_dir, n=params['n'], seed=params['seed'])
    return [fcs]


def _full(out_dir, params):
    info = synthetic.make_dataset(
        out_dir, n=params['n'], seed=params['seed'])
    # make_dataset writes into sub-folders and only returns counts; collect the
    # .fcs it produced by walking the tree.
    paths = []
    for root, _dirs, files in os.walk(info['out_dir']):
        for f in files:
            if f.lower().endswith('.fcs'):
                paths.append(os.path.join(root, f))
    return sorted(paths)


# fields: list of (key, label, default) — only the params the generator uses.
_N = ('n', 'Events per sample', 5000)
_SEED = ('seed', 'Random seed', 0)

DATASETS = {
    'PBMC immunophenotyping': {
        'fn': _immuno,
        'fields': [_N, _SEED, ('donors', 'Donors per group', 3)],
        'subdir': 'pbmc',
    },
    'Differentiation time-course': {
        'fn': _diff,
        'fields': [_N, _SEED, ('reps', 'Replicates per day x condition', 2)],
        'subdir': 'diff',
    },
    'Cell cycle': {
        'fn': _cellcycle,
        'fields': [_N, _SEED, ('samples', 'Samples', 2)],
        'subdir': 'cellcycle',
    },
    'Spectral controls': {
        'fn': _spectral,
        'fields': [('n', 'Events per control', 5000), _SEED],
        'subdir': 'spectral',
    },
    'Calibration beads': {
        'fn': _beads,
        'fields': [('n', 'Events', 8000), _SEED],
        'subdir': 'calibration',
    },
    'Everything (full)': {
        'fn': _full,
        'fields': [_N, _SEED],
        'subdir': 'synthetic_data',
    },
}

_INT_FIELDS = {'n', 'seed', 'donors', 'reps', 'samples'}


def _default_out_dir():
    """Default output root: ``~/.openflo/synthetic``."""
    return os.path.join(os.path.expanduser('~'), '.openflo', 'synthetic')


class SyntheticDialog(tk.Toplevel):
    """Generate a synthetic FCS dataset and load it into the editor."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title('Generate synthetic dataset')
        self.transient(editor)
        self.resizable(False, False)

        self._busy = False
        self._field_vars: dict[str, tk.StringVar] = {}

        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')

        # Dataset-type selector.
        ttk.Label(frm, text='Dataset type:').grid(
            row=0, column=0, sticky='w', pady=(0, 4))
        self._type_var = tk.StringVar(value=next(iter(DATASETS)))
        self._type_cb = ttk.Combobox(
            frm, textvariable=self._type_var, state='readonly',
            values=list(DATASETS), width=32)
        self._type_cb.grid(row=0, column=1, columnspan=2, sticky='ew',
                           pady=(0, 4))
        self._type_cb.bind('<<ComboboxSelected>>', lambda _e: self._rebuild_fields())

        # Parameter fields (rebuilt when the dataset type changes).
        self._fields_frame = ttk.Frame(frm)
        self._fields_frame.grid(row=1, column=0, columnspan=3, sticky='ew',
                                pady=(6, 6))

        # Output-directory picker.
        ttk.Label(frm, text='Output folder:').grid(
            row=2, column=0, sticky='w', pady=(4, 0))
        self._out_var = tk.StringVar(value=_default_out_dir())
        ttk.Entry(frm, textvariable=self._out_var, width=34).grid(
            row=2, column=1, sticky='ew', pady=(4, 0))
        ttk.Button(frm, text='Browse...', command=self._pick_dir).grid(
            row=2, column=2, sticky='ew', padx=(6, 0), pady=(4, 0))

        # Buttons.
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=3, sticky='e', pady=(12, 0))
        self._gen_btn = ttk.Button(btns, text='Generate', command=self._generate)
        self._gen_btn.grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text='Close', command=self.destroy).grid(
            row=0, column=1)

        frm.columnconfigure(1, weight=1)
        self._rebuild_fields()

        try:
            self.update_idletasks()
            self.grab_set()
        except Exception:
            pass

    # ── UI helpers ───────────────────────────────────────────────────────────

    def _rebuild_fields(self):
        for child in self._fields_frame.winfo_children():
            child.destroy()
        self._field_vars.clear()
        spec = DATASETS[self._type_var.get()]
        for r, (key, label, default) in enumerate(spec['fields']):
            ttk.Label(self._fields_frame, text=f'{label}:').grid(
                row=r, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=str(default))
            ttk.Entry(self._fields_frame, textvariable=var, width=12).grid(
                row=r, column=1, sticky='w', padx=(6, 0), pady=2)
            self._field_vars[key] = var
        self._fields_frame.columnconfigure(1, weight=1)

    def _pick_dir(self):
        cur = self._out_var.get().strip() or _default_out_dir()
        chosen = filedialog.askdirectory(
            parent=self, title='Choose output folder',
            initialdir=cur if os.path.isdir(cur) else os.path.expanduser('~'))
        if chosen:
            self._out_var.set(chosen)

    def _collect_params(self):
        """Read + validate the field values into a params dict, or raise
        ``ValueError`` with a friendly message."""
        params = {}
        for key, var in self._field_vars.items():
            raw = var.get().strip()
            if key in _INT_FIELDS:
                try:
                    val = int(raw)
                except ValueError as exc:
                    raise ValueError(f'{key!r} must be an integer.') from exc
                if key != 'seed' and val < 1:
                    raise ValueError(f'{key!r} must be >= 1.')
                params[key] = val
            else:
                params[key] = raw
        return params

    # ── Generation ───────────────────────────────────────────────────────────

    def _generate(self):
        if self._busy:
            return
        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror('Invalid value', str(exc), parent=self)
            return

        type_name = self._type_var.get()
        spec = DATASETS[type_name]
        root = self._out_var.get().strip() or _default_out_dir()
        out_dir = os.path.join(root, spec['subdir'])

        self._busy = True
        try:
            self._gen_btn.config(state='disabled')
        except Exception:
            pass
        try:
            self.editor._begin_busy(f'Generating {type_name}...')
        except Exception:
            pass

        fn = spec['fn']

        def _worker():
            try:
                os.makedirs(out_dir, exist_ok=True)
                paths = fn(out_dir, params)
            except Exception as exc:  # noqa: BLE001 - report any failure
                self.after(0, lambda e=exc: self._on_error(e))
                return
            self.after(0, lambda: self._on_done(paths, type_name))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, paths, type_name):
        self._busy = False
        try:
            self.editor._end_busy()
        except Exception:
            pass
        paths = [p for p in (paths or []) if str(p).lower().endswith('.fcs')]
        if not paths:
            try:
                self.editor.status_var.set(
                    f'{type_name}: no FCS files were produced.')
            except Exception:
                pass
            self._restore_button()
            return
        try:
            self.editor.status_var.set(
                f'Generated {len(paths)} {type_name} file(s); loading...')
        except Exception:
            pass
        try:
            self.editor._queue_fcs_loads(paths)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror('Load failed', str(exc), parent=self)
            self._restore_button()
            return
        self.destroy()

    def _on_error(self, exc):
        self._busy = False
        try:
            self.editor._end_busy()
        except Exception:
            pass
        try:
            self.editor.status_var.set(f'Generate failed: {exc}')
        except Exception:
            pass
        messagebox.showerror('Generate failed', str(exc), parent=self)
        self._restore_button()

    def _restore_button(self):
        try:
            self._gen_btn.config(state='normal')
        except Exception:
            pass
