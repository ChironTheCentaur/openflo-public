"""Compensation-matrix editor + single-stain optimizer dialogs.

Two self-contained Tk window classes extracted from gui.py:
``CompensationEditorWindow`` (view/edit a spillover matrix) and
``OptimizeCompensationDialog`` (derive one from single-stain controls). They
take the editor as parent and call back via ``on_apply``; pipeline maths is
imported lazily inside the methods that need it.
"""
from __future__ import annotations

import os
import re
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from .theme import current_palette
from .ui_tips import auto_tip


class CompensationEditorWindow(tk.Toplevel):
    """Window for viewing and editing a spillover matrix (non-modal).

    Auto-imports on open by trying, in order:
      1. the sample's already-applied comp_matrix (if any),
      2. the active sample's FCS file ($SPILL keyword),
      3. a .wsp sitting alongside the FCS file (any matrix it carries), and
      4. a sibling compensation.csv / spillover.csv / comp.csv.
    Otherwise opens with whatever was loaded explicitly via Load…, or an
    identity matrix when the user clicks Reset.

    `on_apply(channels, matrix)` is called when the user clicks Apply.
    """

    def __init__(self, parent, sample=None, on_apply=None):
        super().__init__(parent)
        self.title("Compensation Matrix")
        self.transient(parent)
        self.geometry("780x520")
        self.minsize(560, 360)
        self.sample   = sample
        self.on_apply = on_apply
        self.channels = []
        self.matrix   = None
        self.entries  = {}     # (i, j) -> (StringVar, Entry)
        self._build()
        if sample is not None:
            self._auto_import()

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
                  foreground=current_palette()['muted'],
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
        _c_apply = ttk.Button(bot, text="Apply", command=self._apply)
        _c_apply.pack(side='right', padx=(0, 4))
        auto_tip(_c_apply,
            "Apply this matrix to every loaded sample and replot. Gates keep "
            "their coordinates, so a gate drawn on uncompensated data may "
            "need revisiting afterwards.")

        self._render_matrix()

    def _render_matrix(self):
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
        # foreground so they're legible on any theme (the hardcoded near-black
        # was invisible on the dark Midnight canvas); zero cells are muted so
        # the meaningful spillover stands out. Values carry the header colour.
        pal = current_palette()
        hdr_fg = pal.get('fg', '#20242b')        # headers + used cells
        muted_fg = pal.get('muted', '#888888')   # de-emphasised zeros
        # Top-left corner cell hint
        ttk.Label(self.matrix_frame, text="src \\ dst",
                  foreground=muted_fg,
                  font=('TkDefaultFont', 8, 'italic')).grid(
            row=0, column=0, padx=4, pady=2, sticky='e')
        # Destination column headers (dst across the top).
        for j, ch in enumerate(self.channels):
            ttk.Label(self.matrix_frame, text=ch, foreground=hdr_fg,
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=0, column=j + 1, padx=2, pady=2)
        # Source row labels + entry cells.
        for i, ch in enumerate(self.channels):
            ttk.Label(self.matrix_frame, text=ch, foreground=hdr_fg,
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=i + 1, column=0, padx=4, pady=1, sticky='e')
            for j in range(n):
                val = float(self.matrix[i, j])
                var = tk.StringVar(value=f"{val:.6f}")
                # Zero = no spillover → muted; any non-zero value (incl. the
                # diagonal 1.0) carries the header colour so it reads clearly.
                e = ttk.Entry(self.matrix_frame, textvariable=var, width=10,
                              justify='right',
                              foreground=(muted_fg if abs(val) < 1e-9
                                          else hdr_fg))
                e.grid(row=i + 1, column=j + 1, padx=1, pady=1)
                self.entries[(i, j)] = (var, e)

    # ── State helpers ────────────────────────────────────────────────────

    def _read_matrix_from_entries(self):
        if not self.entries:
            return None
        n = len(self.channels)
        m = np.zeros((n, n), dtype=float)
        for (i, j), (var, _e) in self.entries.items():
            try:
                m[i, j] = float(var.get())
            except ValueError:
                self.status_var.set(f"Cell [{i}, {j}] is not a number.")
                return None
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
        from .pipeline import read_compensation_matrix
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
        # 1) Try embedded $SPILL in the FCS itself.
        try:
            ch, m = read_compensation_matrix(path)
            if m is not None:
                self._set_matrix(ch, m,
                                 source_label=f'FCS $SPILL ({os.path.basename(path)})')
                return
        except Exception:
            pass
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
                    except Exception:
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
                except Exception:
                    continue
        self.status_var.set(
            "No embedded matrix found. Load… or Reset to identity." + unlabelled)

    def _headerless_channels(self, n):
        """Names for a header-less n x n matrix: the sample's fluorescence
        channels when there are exactly n of them, else None. Spillover runs
        between fluorescence detectors; the first n data columns used before
        put scatter on the matrix (FSC-A, SSC-A, FLA-A for a 3x3), so Apply
        'compensated' FSC-A and left FLB and FLC uncompensated."""
        fl = list(getattr(self.sample, 'fluor_channels', None) or [])
        return fl if len(fl) == n else None

    def _unlabelled_note(self, name, n):
        k = len(getattr(self.sample, 'fluor_channels', None) or [])
        return (f"{name} has no header row naming its channels, and its "
                f"{n}×{n} matrix does not fit this sample's {k} fluorescence "
                "channel(s). Add a header row and Load… it.")

    # ── Button handlers ──────────────────────────────────────────────────

    def _load(self):
        from .pipeline import read_compensation_matrix
        path = filedialog.askopenfilename(
            title="Load compensation matrix",
            filetypes=[('Compensation', '*.wsp *.csv *.tsv *.fcs'),
                       ('FlowJo workspace', '*.wsp'),
                       ('CSV', '*.csv'),
                       ('TSV', '*.tsv'),
                       ('FCS (embedded $SPILL)', '*.fcs'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            ch, m = read_compensation_matrix(path)
        except Exception as exc:
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
        from .pipeline import write_compensation_matrix
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
        except Exception as exc:
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
            return
        if self.on_apply:
            self.on_apply(self.channels, m)
        self.status_var.set("Applied.")


class OptimizeCompensationDialog(tk.Toplevel):
    """Per-fluor file picker → single-stain regression. Calls
    `on_complete(channels, matrix)` when the user runs the optimisation."""

    def __init__(self, parent, channels, on_complete):
        super().__init__(parent)
        self.title("Optimize Compensation Matrix")
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
                  font=('TkDefaultFont', 10, 'bold'),
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
            ttk.Entry(body, textvariable=var).grid(
                row=i, column=1, sticky='ew', padx=4, pady=2)
            ttk.Button(body, text="Browse…", width=10,
                       command=lambda c=ch: self._browse(c)).grid(
                row=i, column=2, padx=4, pady=2)

        # Convenience: pick a directory and try to auto-match filenames.
        ttk.Button(body, text="Auto-detect from a folder…",
                   command=self._autofill_from_dir).grid(
            row=len(self.channels), column=1, sticky='w', pady=(8, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var,
                  foreground=current_palette()['muted'], padding=(10, 0, 10, 4)).pack(
            side='top', fill='x')

        bot = ttk.Frame(self, padding=10)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="Cancel",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="Run optimisation",
                   command=self._run).pack(side='right', padx=(0, 4))

    def _browse(self, ch):
        p = filedialog.askopenfilename(
            title=f"Single-stain control for '{ch}'",
            filetypes=[('FCS', '*.fcs'), ('All files', '*.*')])
        if p:
            self.path_vars[ch].set(p)

    # Most labs name a control after the dye, not the brand.
    _VENDOR_PREFIXES = ('Horizon ', 'BD ', 'eFluor ', 'Brilliant ',
                        'Alexa Fluor ', 'AF', 'Super Bright ')

    # A word ends at a separator, between letters and digits, and where
    # lower case turns upper: 'CD3FLA' -> CD 3 FLA, 'compFLB' -> comp FLB,
    # 'PerCP-Cy5.5' -> Per CP Cy 5 5. An all-capitals run stays whole.
    _WORD = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+')

    @classmethod
    def _name_tokens(cls, name):
        """Lower-case words of `name` (see `_WORD`): 'FLB-Cy7 Stained' ->
        ['flb', 'cy', '7', 'stained']. Separators never matter, so 'FLB-Cy7',
        'FLB_Cy7', 'FLB Cy7' and 'FLBCy7' compare equal."""
        return [w.lower() for w in cls._WORD.findall(name)]

    @staticmethod
    def _runs(tokens):
        """Every run of consecutive whole `tokens`, joined."""
        return {''.join(tokens[i:j]) for i in range(len(tokens))
                for j in range(i + 1, len(tokens) + 1)}

    @classmethod
    def _channel_keys(cls, channel):
        """(dye, keys, whole) for `channel`. `dye` is the channel name without
        its -A/-H/-W suffix and any bracketed detector or laser ('FLB (YG)-A'),
        joined ('FLB-Cy7-A' -> 'flbcy7'): channels with the same dye are one
        fluor and share a control. A key is a run of the dye's words
        ('flbcy7', 'flbcy', 'flb', 'cy7', 'cy'); a file matches it only as a run
        of WHOLE words, so 'flb' never matches inside 'flbcy7' or 'xflb'.
        The vendor prefix is not a key on its own ('Horizon' would claim every
        Horizon dye), and neither is a single character or a digit run inside
        a longer dye name (the '5' of PerCP-Cy5-5). `whole` holds the keys
        that name the entire dye, with and without its vendor prefix."""
        s = re.sub(r'\s*\([^)]*\)', '', channel).strip()
        s = re.sub(r'-[AHW]$', '', s, flags=re.IGNORECASE)
        dye = ''.join(cls._name_tokens(s))
        for prefix in cls._VENDOR_PREFIXES:
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):]
                break
        toks = cls._name_tokens(s)
        keys = {dye} if dye else set()
        for i in range(len(toks)):
            for j in range(i + 1, len(toks) + 1):
                k = ''.join(toks[i:j])
                if j - i == len(toks) or (len(k) >= 2 and not k.isdigit()):
                    keys.add(k)
        return dye, keys, {dye, ''.join(toks)} - {''}

    # A detector name: letters, then a wavelength or a number ('B530',
    # 'YG582', 'UV379', 'BL1'), optionally a band width ('B530/30').
    _DETECTOR = re.compile(r'[A-Za-z]{1,3}\d+(?:[-/]\d+)?')

    @classmethod
    def _bracket_dye_keys(cls, channel):
        """`_channel_keys` of the dye a channel names in brackets after a
        detector ('B530 (FLA)-A' -> FLA, 'YG582 (FLB)-A' -> FLB), or None
        for a channel not written that way. _channel_keys strips the bracket,
        which is right for a laser ('FLB (YG)-A') or a marker ('FLA (CD3)-A')
        but left these with only the detector."""
        m = re.fullmatch(r'\s*(.*?)\s*\(([^)]*)\)\s*(-[AHW])?\s*', channel,
                         re.IGNORECASE)
        if (not m or not m.group(2).strip()
                or not cls._DETECTOR.fullmatch(m.group(1))):
            return None
        return cls._channel_keys(m.group(2).strip() + (m.group(3) or ''))

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

        Dividing a match by the whole channel name let a marker name decide
        ('CD4 FLB-Cy7-A' took FLB.fcs), and whole tokens alone missed joined
        names (AF647, CD3FLA, eF450).
        """
        folder = filedialog.askdirectory(
            title="Pick a folder with single-stain FCS files")
        if not folder:
            return
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith('.fcs'))
        info = {ch: self._channel_keys(ch) for ch in self.channels}
        words = {f: self._name_tokens(os.path.splitext(f)[0]) for f in files}
        runs = {f: self._runs(words[f]) for f in files}
        # A detector outside the brackets that names no file here, and a dye
        # inside them ('B530 (FLA)-A'): the channel is that dye's. A dye
        # outside ('FLB (YG)-A', 'FLA (CD3)-A') is no detector name and keeps
        # its keys; one that looks like one ('FLD (CD3)-A') keeps them
        # whenever a file here names it.
        for ch in self.channels:
            alt = self._bracket_dye_keys(ch)
            if alt and not any(k in runs[f] for k in info[ch][1]
                               for f in files):
                info[ch] = alt
        # The longest key any channel finds in each file.
        longest = {f: max((len(k) for _d, ks, _w in info.values() for k in ks
                           if k in runs[f]), default=0) for f in files}
        claims = {}                        # file -> {dye: [channels]}
        for ch, (dye, ks, _whole) in info.items():
            top = max((len(k) for k in ks if any(k in runs[f] for f in files)),
                      default=0)
            for f in files:
                if top and longest[f] == top and any(
                        len(k) == top and k in runs[f] for k in ks):
                    claims.setdefault(f, {}).setdefault(dye, []).append(ch)
        cands = {ch: [] for ch in self.channels}      # ch -> [(leftover, file)]
        contested = {ch: [] for ch in self.channels}  # ch -> [(file, rivals)]
        for f, by_dye in claims.items():
            if len(by_dye) > 1:
                # The dye this file names entirely, if exactly one: 'flc' is
                # all of FLC-A but only part of FLC-Cy7-A.
                named = [d for d, chs in by_dye.items()
                         if any(len(w) == longest[f] and w in runs[f]
                                for w in info[chs[0]][2])]
                if len(named) == 1:
                    by_dye = {named[0]: by_dye[named[0]]}
            chans = [ch for chs in by_dye.values() for ch in chs]
            for ch in chans:
                if len(by_dye) == 1:
                    cands[ch].append((len(''.join(words[f])) - longest[f], f))
                else:
                    contested[ch].append((f, chans))

        results, ambiguous, missed = {}, [], []
        for ch in self.channels:
            got = sorted(cands[ch])
            if got and (len(got) == 1 or got[0][0] < got[1][0]):
                results[ch] = got[0][1]
            elif got:
                ambiguous.append(f"{ch} matched {len(got)} files")
            elif contested[ch]:
                f, rivals = contested[ch][0]
                ambiguous.append(f"{f} fits {' and '.join(rivals)} equally")
            else:
                missed.append(ch)

        for ch, f in results.items():
            self.path_vars[ch].set(os.path.join(folder, f))

        msg_parts = [f"Matched {len(results)}/{len(self.channels)} channels"]
        if ambiguous:
            msg_parts.append("ambiguous, left empty: "
                             + '; '.join(dict.fromkeys(ambiguous[:3])))
        if missed:
            msg_parts.append(f"unmatched: {', '.join(missed)}")
        self.status_var.set('  •  '.join(msg_parts))

    def _run(self):
        paths = {ch: v.get().strip()
                 for ch, v in self.path_vars.items()
                 if v.get().strip()}
        if not paths:
            self.status_var.set("Pick at least one single-stain file.")
            return
        from .pipeline import optimize_compensation
        try:
            ch, m = optimize_compensation(self.channels, paths)
        except Exception as exc:
            self.status_var.set(f"Optimization failed: {exc}")
            return
        self.on_complete(ch, m)
        self.destroy()


# ── App themes ───────────────────────────────────────────────────────────────
# Two chrome palettes. The matplotlib plot stays light in BOTH (flow-cytometry
# field norm) — these only colour the surrounding Tk/ttk chrome.
# Each palette carries chrome colours plus four `plot_*` keys for the
# matplotlib canvas. Light and dark chrome both keep a WHITE plot (field norm
# — scatters read best on white); 'midnight' is dark chrome with a dark plot
# too. (All values are strings — `_DARK_MODES` below tracks chrome darkness so
# the dict stays str-typed for the many `pal[...]` widget-colour calls.)
