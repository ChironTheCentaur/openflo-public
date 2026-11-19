"""Compensation-matrix editor + single-stain optimizer dialogs.

Two self-contained Tk window classes extracted from gui.py:
``CompensationEditorWindow`` (view/edit a spillover matrix) and
``OptimizeCompensationDialog`` (derive one from single-stain controls). They
take the editor as parent and call back via ``on_apply``; pipeline maths is
imported lazily inside the methods that need it.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from .async_task import BusyStrip
from .comp_controls import ControlMatcher
from .dialog_inputs import parse_number
from .errors import note
from .paths import FCS_FILE_PATTERNS, is_single_fcs_file
from .scope_pick import ScopePickMixin
from .theme import current_palette
from .ui_tips import auto_tip


class CompensationEditorWindow(ScopePickMixin, tk.Toplevel):
    """Window for viewing and editing a spillover matrix (non-modal).

    Auto-imports on open by trying, in order:
      1. the sample's already-applied comp_matrix (if any),
      2. the active sample's FCS file ($SPILL keyword),
      3. a .wsp sitting alongside the FCS file (any matrix it carries), and
      4. a sibling compensation.csv / spillover.csv / comp.csv.
    Otherwise opens with whatever was loaded explicitly via Load…, or an
    identity matrix when the user clicks Reset.

    `on_apply(channels, matrix)` is called when the user clicks Apply. It
    may return ``(applied, message)``; the window then shows that message
    (what was really done) instead of a plain "Applied.".
    With ``batch`` -- ``batch(channels) -> (members, left_out)``, see
    :func:`openflo.acquisition.batch_of` -- the window offers "Apply to:
    this sample / this batch" and calls ``on_apply(channels, matrix,
    targets)`` with the samples chosen. Once the user picks a scope it
    stays picked: a Load…, Reset or Optimize result redraws the matrix but
    never flips "this sample" back to "this batch" (the batch default
    applies only until the user chooses). The batch is read again at Apply,
    so the targets are the samples the label names at that moment.

    The window is tied to ``sample`` for its whole life (the tray selection
    may move on), so the title and the "this sample" choice name it
    (``name``, the sample's name in the session; else ``sample.name``).
    """

    def __init__(self, parent, sample=None, on_apply=None, batch=None,
                 name=None):
        super().__init__(parent)
        # The sample's name in the session (``sample.name`` is its file's).
        self.sample_name = name or (getattr(sample, 'name', None)
                                    if sample is not None else None)
        self.title(f"Compensation matrix — {self.sample_name}"
                   if self.sample_name else "Compensation matrix")
        self.transient(parent)
        self.geometry("780x520")
        self.minsize(560, 360)
        self.sample   = sample
        self.on_apply = on_apply
        self.batch    = batch
        self._members = []           # the batch for the current channels
        self.channels = []
        self.matrix   = None
        self.entries  = {}     # (i, j) -> (StringVar, Entry)
        self._applied_shown = False  # the status line reports an Apply
        self._scope_pick = None      # 'sample'/'batch' once the user chose
        self._setting_scope = False  # the window, not the user, sets it
        self._build()
        if sample is not None:
            self._auto_import()

    def _sample_label(self):
        """The "this sample" choice, naming the sample it means."""
        return (f"this sample ({self.sample_name})" if self.sample_name
                else "this sample")

    # ── UI ───────────────────────────────────────────────────────────────

    def _build(self):
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(side='top', fill='x')
        _c_load = ttk.Button(top, text="Load (CSV/WSP/FCS)…",
                             command=self._load)
        _c_load.pack(side='left')
        auto_tip(_c_load,
            "Import a spillover matrix — from a CSV, from a FlowJo .wsp, or "
            "from the $SPILL keyword embedded in an FCS file by the "
            "cytometer.")
        _c_save = ttk.Button(top, text="Save…", command=self._save)
        _c_save.pack(side='left', padx=(4, 0))
        auto_tip(_c_save,
            "Write the current matrix to CSV so it can be reused on another "
            "session or shared with the rest of the lab.")
        _c_reset = ttk.Button(top, text="Reset to identity",
                              command=self._reset_identity)
        _c_reset.pack(side='left', padx=(4, 0))
        auto_tip(_c_reset,
            "Clear all spillover — 1 on the diagonal, 0 elsewhere, i.e. no "
            "compensation applied. Use as a starting point, or to check what "
            "the uncompensated data looks like.")
        _c_opt = ttk.Button(top, text="Optimize from single-stains…",
                            command=self._open_optimize)
        _c_opt.pack(side='left', padx=(12, 0))
        auto_tip(_c_opt,
            "Compute the matrix from single-stain controls instead of typing "
            "it: point at one control per channel and OpenFlo solves the "
            "spillover coefficients.")

        self.status_var = tk.StringVar(
            value="Load a matrix or click Reset to start from identity.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground=current_palette()['muted'], wraplength=740,
                  padding=(8, 0, 8, 0)).pack(side='top', fill='x')

        body = ttk.Frame(self, padding=(8, 4, 8, 4))
        body.pack(side='top', fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        # Scrollable matrix area.
        cv = tk.Canvas(body, highlightthickness=0)
        cv.grid(row=0, column=0, sticky='nsew')
        vbar = ttk.Scrollbar(body, orient='vertical', command=cv.yview)
        vbar.grid(row=0, column=1, sticky='ns')
        hbar = ttk.Scrollbar(body, orient='horizontal', command=cv.xview)
        hbar.grid(row=1, column=0, sticky='ew')
        cv.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self._cv = cv
        self.matrix_frame = ttk.Frame(cv)
        _mtx_win = cv.create_window((0, 0), window=self.matrix_frame, anchor='nw')
        self.matrix_frame.bind(
            '<Configure>',
            lambda e: cv.configure(scrollregion=cv.bbox('all')))
        # Stretch the matrix to fill the canvas width so a small NxN matrix isn't
        # jammed into the top-left with dead space beside it; it still scrolls when
        # a panel has more channels than fit.
        cv.bind('<Configure>',
                lambda e: cv.itemconfigure(
                    _mtx_win,
                    width=max(e.width, self.matrix_frame.winfo_reqwidth())))

        bot = ttk.Frame(self, padding=8)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="Close",
                   command=self.destroy).pack(side='right')
        _c_apply = ttk.Button(bot, text="Apply", default='active',
                              command=self._apply)
        _c_apply.pack(side='right', padx=(0, 4))
        auto_tip(_c_apply,
            "Apply this matrix to the samples chosen under 'Apply to' and "
            "replot. Gates keep their coordinates, so a gate drawn on "
            "uncompensated data may need revisiting afterwards.")
        # Apply to: this sample / every sample acquired with it. The batch
        # is read from the FCS keywords (instrument, date, voltages, stored
        # spillover), not folders, which say only what was exported together.
        self.scope_var = tk.StringVar(value='sample')
        self.scope_var.trace_add('write', self._scope_written)
        if self.batch is not None:
            scope = ttk.Frame(self, padding=(8, 0, 8, 0))
            scope.pack(side='bottom', fill='x')
            ttk.Label(scope, text="Apply to:").pack(side='left')
            self._sample_rb = ttk.Radiobutton(
                scope, text=self._sample_label(), value='sample',
                variable=self.scope_var)
            self._sample_rb.pack(side='left', padx=(4, 0))
            self._batch_rb = ttk.Radiobutton(
                scope, text="this batch", value='batch',
                variable=self.scope_var)
            self._batch_rb.pack(side='left', padx=(4, 0))
            self._batch_var = tk.StringVar(value='')
            self._batch_lbl = ttk.Label(
                scope, textvariable=self._batch_var,
                foreground=current_palette()['muted'])
            self._batch_lbl.pack(side='left', padx=(8, 0), fill='x',
                                 expand=True)
            self._batch_tip_text = ''
            auto_tip(self._batch_lbl, lambda: self._batch_tip_text)

        self._render_matrix()

    def _refresh_batch(self, keep_scope=False):
        """Recompute the batch for the current channels and show it: who
        is in it, and on hover why each other sample isn't. The scope
        becomes "this batch" only while the user has not chosen one, and
        "this sample" when there is no batch. ``keep_scope`` never widens
        it: Apply reads the batch again, and a batch that grew since the
        redraw must not turn "this sample" into "this batch"."""
        if self.batch is None:
            return
        members, left_out = ([], {})
        if self.channels:
            members, left_out = self.batch(list(self.channels))
        self._members = list(members)
        if len(members) > 1:
            shown = ', '.join(members[:6]) + (
                f' and {len(members) - 6} more' if len(members) > 6 else '')
            self._batch_var.set(f"{len(members)} samples: {shown}")
            self._batch_rb.state(['!disabled'])
            if not keep_scope:
                self._set_scope(self._scope_pick or 'batch')
        else:
            self._batch_var.set("no other sample was acquired like this one")
            self._batch_rb.state(['disabled'])
            self._set_scope('sample')
        why = '\n'.join(f"{n}: {r}" for n, r in list(left_out.items())[:15])
        more = len(left_out) - 15
        tip = ("Same instrument, acquisition date and detector voltages"
               " (and stored spillover, when both have one)."
               + (f"\n\nLeft out:\n{why}" if why else '')
               + (f"\n…and {more} more" if more > 0 else ''))
        self._batch_tip_text = tip

    def _render_matrix(self):
        self._refresh_batch()
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self.entries = {}
        if self.matrix is None or not self.channels:
            ttk.Label(self.matrix_frame,
                      text="(no matrix loaded)",
                      foreground=current_palette()['muted']).grid(row=0, column=0, padx=20, pady=20)
            return
        n = len(self.channels)
        # Theme-aware colours: header + non-zero ("used") cells take the normal
        # foreground so they're legible on any theme (a fixed near-black is
        # invisible on the dark Midnight canvas); zero cells are muted so
        # the meaningful spillover stands out. Values carry the header colour.
        pal = current_palette()
        hdr_fg = pal.get('fg', '#20242b')        # headers + used cells
        muted_fg = pal.get('muted', '#888888')   # de-emphasised zeros
        # Top-left corner cell hint
        ttk.Label(self.matrix_frame, text="src \\ dst",
                  foreground=muted_fg,
                  font='OpenFloNote').grid(
            row=0, column=0, padx=4, pady=2, sticky='e')
        # Destination column headers (dst across the top).
        for j, ch in enumerate(self.channels):
            ttk.Label(self.matrix_frame, text=ch, foreground=hdr_fg,
                      font='OpenFloHeading').grid(
                row=0, column=j + 1, padx=2, pady=2)
        # Source row labels + entry cells.
        for i, ch in enumerate(self.channels):
            ttk.Label(self.matrix_frame, text=ch, foreground=hdr_fg,
                      font='OpenFloHeading').grid(
                row=i + 1, column=0, padx=4, pady=1, sticky='e')
            for j in range(n):
                val = float(self.matrix[i, j])
                var = tk.StringVar(value=f"{val:.6f}")
                var.trace_add('write', self._cell_edited)
                # Zero = no spillover → muted; any non-zero value (incl. the
                # diagonal 1.0) carries the header colour so it reads clearly.
                e = ttk.Entry(self.matrix_frame, textvariable=var, width=10,
                              justify='right',
                              foreground=(muted_fg if abs(val) < 1e-9
                                          else hdr_fg))
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                self.entries[(i, j)] = (var, e)

    # ── State helpers ────────────────────────────────────────────────────

    def _cell_edited(self, *_args):
        """A cell changed after an Apply: the matrix on screen is no longer
        the one in the data, so the Apply result must not stay up."""
        if self._applied_shown:
            self._applied_shown = False
            self.status_var.set("Matrix edited, not applied yet. Press Apply "
                                "to use it.")

    def _read_matrix_from_entries(self):
        if not self.entries:
            return None
        n = len(self.channels)
        m = np.zeros((n, n), dtype=float)
        for (i, j), (var, _e) in self.entries.items():
            # parse_number refuses words, 'nan' and 'inf'; a matrix holding
            # nan or inf inverts to all NaN and would wipe every compensated
            # value.
            v = parse_number(var.get())
            if v is None:
                self._applied_shown = False
                self.status_var.set(
                    f"Cell {self.channels[i]} → {self.channels[j]} must be a "
                    f"finite number (got '{var.get().strip()}').")
                return None
            m[i, j] = v
        return m

    def _set_matrix(self, channels, matrix, source_label=''):
        self.channels = list(channels)
        self.matrix   = np.asarray(matrix, dtype=float)
        self._render_matrix()
        if source_label:
            self.status_var.set(
                f"Loaded {len(self.channels)}×{len(self.channels)} matrix "
                f"from {source_label}.")

    # ── Auto-import ──────────────────────────────────────────────────────

    def _auto_import(self):
        from .compensation_io import read_compensation_matrix
        # 0) If a matrix is already applied to this sample, show that — so
        #    edits build on the active matrix and it "stays loaded" across
        #    reopens, rather than silently reverting to the FCS $SPILL.
        cm = getattr(self.sample, 'comp_matrix', None)
        cc = getattr(self.sample, 'comp_channels', None)
        if cm is not None and cc:
            self._set_matrix(list(cc), cm, source_label='currently applied')
            return
        path = getattr(self.sample, 'path', None)
        if not path:
            return
        # 1) Try embedded $SPILL in the FCS itself (the sample's own data
        #    set of a file holding several).
        try:
            ch, m = read_compensation_matrix(
                path, getattr(self.sample, 'set_number', None))
            if m is not None:
                self._set_matrix(ch, m,
                                 source_label=f'FCS $SPILL ({os.path.basename(path)})')
                return
        except Exception as exc:  # noqa: BLE001 - noted; the search goes on to sibling files
            note("reading the compensation matrix from the FCS file", exc)
        # 2) Try a sibling .wsp in the same folder.
        folder = os.path.dirname(path)
        if folder and os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith('.wsp'):
                    try:
                        ch, m = read_compensation_matrix(
                            os.path.join(folder, fn))
                        if m is not None:
                            self._set_matrix(ch, m,
                                             source_label=f'sibling .wsp ({fn})')
                            return
                    except Exception as exc:  # noqa: BLE001 - noted; an unreadable sibling workspace is skipped and the search goes on
                        note("reading a compensation matrix from a sibling file", exc)
                        continue
        # 3) Try a sibling compensation.csv / spillover.csv.
        unlabelled = ''
        for name in ('compensation.csv', 'spillover.csv', 'comp.csv'):
            p = os.path.join(folder, name) if folder else name
            if os.path.isfile(p):
                try:
                    ch, m = read_compensation_matrix(p)
                    if m is None:
                        continue
                    if ch is None:        # headerless csv
                        ch = self._headerless_channels(m.shape[0])
                        if ch is None:
                            unlabelled = f" {self._unlabelled_note(name, m.shape[0])}"
                            continue
                    self._set_matrix(ch, m, source_label=f'{name}')
                    return
                except Exception as exc:  # noqa: BLE001 - noted; an unreadable matrix file is skipped and the search goes on
                    note("reading a compensation matrix file", exc)
                    continue
        self.status_var.set(
            "No embedded matrix found. Load… or Reset to identity." + unlabelled)

    def _headerless_channels(self, n):
        """Names for a header-less n x n matrix: the sample's fluorescence
        channels when there are exactly n of them, else None. Spillover runs
        between fluorescence detectors; the first n data columns would put
        scatter on the matrix (FSC-A, SSC-A, FLA-A for a 3x3), so Apply would
        'compensate' FSC-A and leave FLB and FLC uncompensated."""
        fl = list(getattr(self.sample, 'fluor_channels', None) or [])
        return fl if len(fl) == n else None

    def _unlabelled_note(self, name, n):
        k = len(getattr(self.sample, 'fluor_channels', None) or [])
        return (f"{name} has no header row naming its channels, and its "
                f"{n}×{n} matrix does not fit this sample's {k} fluorescence "
                "channel(s). Add a header row and Load… it.")

    # ── Button handlers ──────────────────────────────────────────────────

    def _load(self):
        from .compensation_io import read_compensation_matrix
        path = filedialog.askopenfilename(
            title="Load compensation matrix",
            filetypes=[('Compensation',
                        f'*.wsp *.csv *.tsv {FCS_FILE_PATTERNS}'),
                       ('FlowJo workspace', '*.wsp'),
                       ('CSV', '*.csv'),
                       ('TSV', '*.tsv'),
                       ('FCS (embedded $SPILL)', FCS_FILE_PATTERNS),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            ch, m = read_compensation_matrix(path)
        except Exception as exc:  # noqa: BLE001 - a file that will not load is reported in the status line
            self.status_var.set(f"Load failed: {exc}")
            return
        if m is None:
            self.status_var.set(
                f"No matrix found in {os.path.basename(path)}")
            return
        if ch is None:
            if self.sample is None:
                ch = [f'ch{i}' for i in range(m.shape[0])]
            else:
                ch = self._headerless_channels(m.shape[0])
                if ch is None:
                    self.status_var.set(
                        self._unlabelled_note(os.path.basename(path), m.shape[0]))
                    return
        self._set_matrix(ch, m, source_label=os.path.basename(path))

    def _save(self):
        from .compensation_io import write_compensation_matrix
        m = self._read_matrix_from_entries()
        if m is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save compensation matrix",
            defaultextension='.csv',
            initialfile='compensation.csv',
            filetypes=[('CSV', '*.csv'),
                       ('TSV', '*.tsv'),
                       ('FlowJo workspace', '*.wsp')])
        if not path:
            return
        try:
            write_compensation_matrix(path, m, self.channels)
            self.status_var.set(f"Saved → {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001 - a failed save is reported in the status line
            self.status_var.set(f"Save failed: {exc}")

    def _reset_identity(self):
        # If no channels yet, prefer the active sample's columns.
        if not self.channels:
            data = getattr(self.sample, 'data', None)
            if data is not None:
                # Default to the fluor channels if the sample classified
                # them; otherwise every column.
                fl = getattr(self.sample, 'fluor_channels', None) or list(data.columns)
                self.channels = list(fl)
            else:
                self.status_var.set("Load a sample before resetting to identity.")
                return
        n = len(self.channels)
        self.matrix = np.eye(n, dtype=float)
        self._render_matrix()
        self.status_var.set(f"Reset to {n}×{n} identity matrix.")

    def _open_optimize(self):
        if not self.channels:
            self.status_var.set(
                "Load a matrix or pick a sample first so the channel "
                "list is known.")
            return
        OptimizeCompensationDialog(
            self, channels=self.channels,
            on_complete=self._set_matrix)

    def _apply(self):
        m = self._read_matrix_from_entries()
        if m is None:
            # The main status line may still report an earlier Apply
            # ("Applied 3×3 matrix to ..."): say this one was refused.
            main = getattr(self.master, 'status_var', None)
            if main is not None:
                why = self.status_var.get()
                try:
                    main.set("Compensation matrix not applied: "
                             + why[:1].lower() + why[1:])
                except tk.TclError:
                    pass
            return
        result = None
        if self.on_apply:
            if self.batch is None:
                result = self.on_apply(self.channels, m)
            else:
                # The batch as it is now (a tab switch or a removed sample
                # since the last redraw changes it), so Apply reaches the
                # samples the label names.
                self._refresh_batch(keep_scope=True)
                targets = (list(self._members)
                           if self.scope_var.get() == 'batch'
                           and len(self._members) > 1 else None)
                result = self.on_apply(self.channels, m, targets)
        if isinstance(result, tuple):
            ok, msg = result
            self.status_var.set(msg or ("Applied." if ok else "Not applied."))
        else:
            self.status_var.set("Applied.")
        self._applied_shown = True


class OptimizeCompensationDialog(ControlMatcher, tk.Toplevel):
    """Per-fluor file picker → single-stain regression. Calls
    `on_complete(channels, matrix)` when the user runs the optimisation."""

    def __init__(self, parent, channels, on_complete):
        super().__init__(parent)
        self.title("Optimize compensation matrix")
        self.transient(parent)
        self.grab_set()
        self.geometry("680x420")
        self.minsize(560, 280)
        self.channels    = list(channels)
        self.on_complete = on_complete
        self.path_vars   = {}     # channel -> StringVar
        self._build()

    def _build(self):
        ttk.Label(self, text="Optimize compensation from single-stain controls",
                  font='OpenFloHeading',
                  padding=(10, 10, 10, 2)).pack(side='top', fill='x')
        ttk.Label(self,
                  text="For each fluor channel, point to a single-stain FCS "
                       "where ONLY that dye is bright, with a negative "
                       "population in the same file. Each control is split "
                       "into its negative and positive events on the source "
                       "channel; the spillover into every other channel is "
                       "the difference of their medians there over the "
                       "difference on the source channel. Saturated events "
                       "are left out.",
                  foreground=current_palette()['muted'],
                  wraplength=620,
                  padding=(10, 0, 10, 8),
                  justify='left').pack(side='top', fill='x')

        body = ttk.Frame(self, padding=(10, 0, 10, 0))
        body.pack(side='top', fill='both', expand=True)
        body.columnconfigure(1, weight=1)
        for i, ch in enumerate(self.channels):
            var = tk.StringVar()
            self.path_vars[ch] = var
            ttk.Label(body, text=ch).grid(row=i, column=0, sticky='e', padx=4, pady=2)
            _e = ttk.Entry(body, textvariable=var)
            _e.grid(row=i, column=1, sticky='ew', padx=4, pady=2)
            auto_tip(_e, f"The single-stain control FCS file for {ch}: only "
                         "this dye bright, with unstained events in the same "
                         "file. Leave empty to skip this channel.")
            _b = ttk.Button(body, text="Browse…", width=10,
                            command=lambda c=ch: self._browse(c))
            _b.grid(row=i, column=2, padx=4, pady=2)
            auto_tip(_b, f"Pick the single-stain control file for {ch}.")

        # Convenience: pick a directory and try to auto-match filenames.
        _ad = ttk.Button(body, text="Auto-detect from a folder…",
                         command=self._autofill_from_dir)
        _ad.grid(row=len(self.channels), column=1, sticky='w', pady=(8, 0))
        auto_tip(_ad, "Pick a folder of single-stain controls (.fcs or .lmd "
                      "files) and fill in each channel whose dye the file "
                      "name spells. Unclear "
                      "matches are left empty, never guessed; check the "
                      "result before running.")

        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var,
                  foreground=current_palette()['muted'], padding=(10, 0, 10, 4)).pack(
            side='top', fill='x')

        bot = ttk.Frame(self, padding=10)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="Cancel",
                   command=self.destroy).pack(side='right')
        self._run_btn = ttk.Button(bot, text="Run optimization",
                                   default='active', command=self._run)
        self._run_btn.pack(side='right', padx=(0, 4))
        auto_tip(self._run_btn,
                 "Read the chosen control files and work out the spillover "
                 "matrix. The result replaces the matrix in the compensation "
                 "editor, where you can review it before applying.")
        # The controls are read off the window thread, shown here.
        self._busy = BusyStrip(self, padding=(10, 0))
        self._busy.pack(side='bottom', fill='x')

    def _browse(self, ch):
        p = filedialog.askopenfilename(
            title=f"Single-stain control for '{ch}'",
            filetypes=[('FCS', FCS_FILE_PATTERNS), ('All files', '*.*')])
        if p:
            self.path_vars[ch].set(p)

    def _autofill_from_dir(self):
        """Match single-stain FCS files in a chosen folder to channels.

        A channel considers only the files that spell its LONGEST key found
        in the folder, and a file goes only to channels whose key is the
        longest any channel finds in it: the FLB-Cy7 control goes to FLB-Cy7-A,
        not FLB-A, and with no FLC-Cy7 control FLB-Cy7-A does not take one
        spelling only 'cy7'. Channels of one dye (FLA-A, FLA-H, or FLB on
        two lasers) share its file. A file that two different dyes find
        equally goes to the one dye it names entirely (FLC.fcs to FLC-A, not
        FLC-Cy7-A, when there is no FLC-Cy7 control), else to neither.
        When a channel is left with several files, the one with the least of
        its name left over wins ('FLB.fcs' over 'FLB-Cy7.fcs' when there is no
        FLB-Cy7 channel); a tie leaves the channel empty and is reported,
        never guessed.

        Dividing a match by the whole channel name would let a marker name
        decide ('CD4 FLB-Cy7-A' would take FLB.fcs), and whole tokens alone
        miss joined names (AF647, CD3FLA, eF450).
        """
        folder = filedialog.askdirectory(
            title="Pick a folder with single-stain FCS files")
        if not folder:
            return
        files = sorted(f for f in os.listdir(folder) if is_single_fcs_file(f))
        results, ambiguous, missed = self._match_controls(self.channels, files)
        for ch, f in results.items():
            self.path_vars[ch].set(os.path.join(folder, f))
        self.status_var.set(self._autofill_message(
            len(results), len(self.channels), ambiguous, missed))

    def _run(self):
        paths = {ch: v.get().strip()
                 for ch, v in self.path_vars.items()
                 if v.get().strip()}
        if not paths:
            self.status_var.set("Pick at least one single-stain file.")
            return
        from .compensation_optimizer import optimize_compensation
        channels = list(self.channels)

        def work(job):
            # Reads every control file: off the window thread. Cancel takes
            # effect when the files are read, and nothing is applied.
            res = optimize_compensation(channels, paths)
            job.check()
            return res

        def done(res):
            self.on_complete(*res)
            self.destroy()

        self.status_var.set("")
        self._busy.run(
            work, on_done=done,
            on_error=lambda exc: self.status_var.set(
                f"Optimization failed: {exc}"),
            on_cancel=lambda: self.status_var.set(
                "Optimization canceled; the matrix is unchanged."),
            text=f"Reading {len(paths)} single-stain control(s)…",
            disable=[self._run_btn])
