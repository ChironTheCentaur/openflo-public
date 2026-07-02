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

from .theme import current_palette


class CompensationEditorWindow(tk.Toplevel):
    """Modal window for viewing and editing a spillover matrix.

    Auto-imports on open by trying, in order:
      1. the active sample's FCS file ($SPILL keyword), and
      2. a .wsp sitting alongside the FCS file (any matrix it carries).
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
        ttk.Button(top, text="Load (CSV/WSP/FCS)…",
                   command=self._load).pack(side='left')
        ttk.Button(top, text="Save…",
                   command=self._save).pack(side='left', padx=(4, 0))
        ttk.Button(top, text="Reset to identity",
                   command=self._reset_identity).pack(side='left', padx=(4, 0))
        ttk.Button(top, text="Optimize from single-stains…",
                   command=self._open_optimize).pack(side='left', padx=(12, 0))

        self.status_var = tk.StringVar(
            value="Load a matrix or click Reset to start from identity.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground='grey',
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
        cv.create_window((0, 0), window=self.matrix_frame, anchor='nw')
        self.matrix_frame.bind(
            '<Configure>',
            lambda e: cv.configure(scrollregion=cv.bbox('all')))

        bot = ttk.Frame(self, padding=8)
        bot.pack(side='bottom', fill='x')
        ttk.Button(bot, text="Close",
                   command=self.destroy).pack(side='right')
        ttk.Button(bot, text="Apply",
                   command=self._apply).pack(side='right', padx=(0, 4))

        self._render_matrix()

    def _render_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()
        self.entries = {}
        if self.matrix is None or not self.channels:
            ttk.Label(self.matrix_frame,
                      text="(no matrix loaded)",
                      foreground='grey').grid(row=0, column=0, padx=20, pady=20)
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
        for name in ('compensation.csv', 'spillover.csv', 'comp.csv'):
            p = os.path.join(folder, name) if folder else name
            if os.path.isfile(p):
                try:
                    ch, m = read_compensation_matrix(p)
                    if m is None:
                        continue
                    if ch is None:        # headerless csv → use sample channels
                        data = getattr(self.sample, 'data', None)
                        if data is not None and m.shape[0] <= len(data.columns):
                            ch = list(data.columns)[:m.shape[0]]
                    if ch is not None:
                        self._set_matrix(ch, m, source_label=f'{name}')
                        return
                except Exception:
                    continue
        self.status_var.set(
            "No embedded matrix found. Load… or Reset to identity.")

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
            data = getattr(self.sample, 'data', None)
            if data is not None and m.shape[0] <= len(data.columns):
                ch = list(data.columns)[:m.shape[0]]
            else:
                ch = [f'ch{i}' for i in range(m.shape[0])]
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
                       "where ONLY that dye is bright. The regression uses "
                       "the brightest 2% of events on the source channel to "
                       "estimate spillover into every other channel.",
                  foreground='grey',
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
                  foreground='grey', padding=(10, 0, 10, 4)).pack(
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

    @staticmethod
    def _channel_tokens(channel):
        """Yield candidate filename tokens for ``channel``, longest /
        most-specific first.

        Examples
        --------
        'BV421-A' -> ['bv421', 'horizonbv421']
        'PE-Cy7-A'        -> ['pecy7', 'cy7', 'pe']
        'APC-Cy7-A'       -> ['apccy7', 'cy7', 'apc']
        'PerCP-Cy5.5-A'   -> ['percpcy5.5', 'cy5.5', 'percp']
        'APC-A'           -> ['apc']
        'FITC-A'          -> ['fitc']

        The naïve ``ch.split('-')[0]`` heuristic this replaces would
        return ``'pe'`` for ``PE-Cy7-A`` — too generic; it picks up any
        FCS that happens to contain the letters ``pe``, e.g. a
        ``compensation_pe-cy7.fcs`` would be matched for the PE channel
        even though it belongs to PE-Cy7.
        """
        import re as _re
        s = channel
        # Strip vendor prefixes — most labs annotate the dye, not the brand.
        for prefix in ('Horizon ', 'BD ', 'eFluor ', 'Brilliant ',
                       'Alexa Fluor ', 'AF', 'Super Bright '):
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):]
                break
        # Strip trailing -A / -H / -W (PMT area / height / width).
        s = _re.sub(r'-[AHW]$', '', s, flags=_re.IGNORECASE)
        parts = [p for p in s.split('-') if p]
        if not parts:
            return []
        # Most specific = the joined / collapsed form (no separators).
        full = ''.join(parts).lower()
        # Then individual parts, longest first. Skip pure-digit and
        # very short fragments (less than 3 chars) — too ambiguous.
        sub = sorted({p.lower() for p in parts
                      if len(p) >= 3 and not p.isdigit()},
                     key=len, reverse=True)
        out = []
        if len(full) >= 3 and full not in sub:
            out.append(full)
        out.extend(sub)
        # De-dup while preserving order.
        seen, uniq = set(), []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _autofill_from_dir(self):
        """Match channel names against filename tokens in the chosen
        directory. Uses :meth:`_channel_tokens` to generate ordered
        candidate tokens per channel; first token that yields exactly
        one file wins. Multiple matches surface a warning, none → skip.
        """
        folder = filedialog.askdirectory(
            title="Pick a folder with single-stain FCS files")
        if not folder:
            return
        files = [f for f in os.listdir(folder) if f.lower().endswith('.fcs')]
        flow = [f.lower() for f in files]

        results, ambiguous, missed = {}, [], []
        for ch in self.channels:
            best = None
            for token in self._channel_tokens(ch):
                hits = [files[i] for i, name in enumerate(flow)
                        if token in name]
                if len(hits) == 1:
                    best = hits[0]
                    break
                if len(hits) > 1 and best is None:
                    # Remember the first ambiguous hit-set in case no
                    # subsequent token disambiguates.
                    best = hits[0]   # heuristic: alphabetically-first match
                    ambiguous.append((ch, token, hits))
                    # Keep trying — a more specific token may still narrow.
            if best:
                results[ch] = best
            else:
                missed.append(ch)

        for ch, f in results.items():
            self.path_vars[ch].set(os.path.join(folder, f))

        msg_parts = [f"Matched {len(results)}/{len(self.channels)} channels"]
        if ambiguous:
            amb_summary = '; '.join(
                f"{ch} matched {len(hits)} files via '{tok}'"
                for ch, tok, hits in ambiguous[:3])
            msg_parts.append(f"ambiguous: {amb_summary}")
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
