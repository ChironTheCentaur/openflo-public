"""Population statistics window: per-population counts + per-channel stats.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class StatisticsWindow(tk.Toplevel):
    """Population-statistics table over the editor's loaded samples.

    Rows = sample × population (gate node, evaluated as the cumulative
    gate chain). Columns are modular — toggle Count / %Parent / %Total
    and per-channel Median / Mean / CV. Export the current table to CSV.
    Recomputes on toggle or Refresh; uses the full sample data (not the
    plot's downsample).
    """

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Population statistics")
        self.geometry("1000x560")
        self.minsize(640, 360)

        self._cols = []     # current Treeview column ids
        self._rows = []     # current row dicts
        # Curated target set: an ordered list of (sample, gid) POPULATIONS —
        # statistics is population-based, so only gate rows are accepted (never
        # whole samples). Drag from either tree APPENDS; the Import buttons
        # OVERRIDE; Clear empties it. ``_gate_sources`` maps (sample, gid) →
        # {'editor','workspace'} for the Source column.
        #
        # ``_user_curated`` separates "fresh window → show every population as a
        # convenience" (False) from "the user has touched the set" (True). Once
        # True, the table shows EXACTLY ``_gate_targets`` — so Clear → empty
        # table that STAYS empty instead of auto-repopulating with all pops.
        self._gate_targets = []     # [(sample, gid)] in display order
        self._gate_sources = {}     # (sample, gid) -> {source}
        self._user_curated = False

        # ── Column-selection checkboxes ──────────────────────────────────
        opts = ttk.Frame(self, padding=(10, 8, 10, 4))
        opts.pack(side='top', fill='x')
        ttk.Label(opts, text="Columns:",
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left',
                                                          padx=(0, 8))
        self._stat_vars = {}
        defaults = {'Count', '%Parent', '%Total', 'Median'}
        for stat in (*editor.STAT_POP, *editor.STAT_CHAN):
            var = tk.BooleanVar(value=(stat in defaults))
            self._stat_vars[stat] = var
            ttk.Checkbutton(opts, text=stat, variable=var,
                            command=self._refresh).pack(side='left',
                                                        padx=(0, 6))

        # ── Table ────────────────────────────────────────────────────────
        table_holder = ttk.Frame(self, padding=(10, 0, 10, 6))
        table_holder.pack(side='top', fill='both', expand=True)
        self.tv = ttk.Treeview(table_holder, show='headings', height=18)
        ysb = ttk.Scrollbar(table_holder, orient='vertical',
                            command=self.tv.yview)
        xsb = ttk.Scrollbar(table_holder, orient='horizontal',
                            command=self.tv.xview)
        self.tv.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tv.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        xsb.grid(row=1, column=0, sticky='ew')
        table_holder.rowconfigure(0, weight=1)
        table_holder.columnconfigure(0, weight=1)

        # ── Buttons + status ─────────────────────────────────────────────
        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, foreground='grey',
                  padding=(10, 0, 10, 2)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        ttk.Button(btns, text="Refresh", command=self._refresh).pack(side='left')
        ttk.Button(btns, text="Export CSV…",
                   command=self._export_csv).pack(side='left', padx=(6, 0))
        # Per-side bulk import — each REPLACES the curated set (override).
        # Dragging a gate row from either tree APPENDS instead.
        ttk.Button(btns, text="Import S&G gates",
                   command=self._import_all_editor).pack(side='left', padx=(12, 0))
        ttk.Button(btns, text="Import workspace",
                   command=self._import_all_workspace).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text="Clear",
                   command=self._clear_targets).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

        self.status_var.set("Showing all populations. Drag a gate from the "
                            "panel or a gated workspace item here to add it; "
                            "the Import buttons replace the set.")
        self._refresh()

    # ── Curated population set (drag = append, import = override) ─────────
    def add_targets(self, targets, source):
        """APPEND (sample, gid) populations to the curated set, tagging each
        with the side it came from ('editor' | 'workspace'). Switches the table
        out of the default 'all populations' mode. Called by the editor /
        workspace gate-drag handlers."""
        added = 0
        for t in targets:
            if not (isinstance(t, tuple) and len(t) == 2 and all(t)):
                continue
            if t not in self._gate_targets:
                self._gate_targets.append(t)
            self._gate_sources.setdefault(t, set()).add(source)
            added += 1
        if added:
            self._user_curated = True
            self._refresh()
            try:
                self.lift()
                self.focus_set()
            except Exception:
                pass

    def _set_targets(self, targets, source):
        """REPLACE the curated set with `targets` (override). Used by Import."""
        self._gate_targets = []
        self._gate_sources = {}
        for t in targets:
            if t not in self._gate_targets:
                self._gate_targets.append(t)
            self._gate_sources.setdefault(t, set()).add(source)
        self._user_curated = True
        self._refresh()
        try:
            self.lift()
            self.focus_set()
        except Exception:
            pass

    def _clear_targets(self):
        """Empty the table and keep it empty (does NOT revert to showing every
        population). Use Refresh or an Import button to repopulate."""
        self._gate_targets = []
        self._gate_sources = {}
        self._user_curated = True
        self._refresh()

    def _import_all_editor(self):
        """Override the set with every gate of every loaded editor sample."""
        targets = []
        for name in self.editor._sample_order:
            if name not in self.editor._samples:
                continue
            order = (self.editor._sample_gate_order.get(name)
                     or list(self.editor._sample_gates.get(name, {})))
            for gid in order:
                targets.append((name, gid))
        if not targets:
            self.status_var.set("No gates in the loaded samples to import.")
            return
        self._set_targets(targets, 'editor')

    def _import_all_workspace(self):
        """Override the set with every gated population in the workspace."""
        panel = getattr(self.editor, '_workspace_panel', None)
        model = getattr(panel, 'model', None) if panel is not None else None
        if model is None:
            self.status_var.set("No Pipeline Workspace is open.")
            return
        targets = []
        for _mid, it, _gid in model.all_items():
            nm, gid = it.get('sample'), it.get('gate_id')
            if nm and gid and (nm, gid) not in targets:
                targets.append((nm, gid))
        if not targets:
            self.status_var.set("The workspace has no gated populations to import.")
            return
        self._set_targets(targets, 'workspace')

    def _selected_stats(self):
        return {s for s, v in self._stat_vars.items() if v.get()}

    def _refresh(self):
        want = self._selected_stats()
        # Curated mode (the user has dragged/imported/cleared): show EXACTLY the
        # targeted (sample, gid) populations — an empty set stays empty. A fresh
        # window (not yet curated) shows every population as a convenience.
        curated = self._user_curated
        # Snapshot the inputs on the Tk thread (cheap: data refs + copied gate
        # dicts), then compute off-thread so a big sample/gate set doesn't freeze
        # the window on every toggle. The snapshot is stable even if the user
        # keeps gating in the main editor meanwhile.
        try:
            if curated:
                snap = self.editor._stats_snapshot(
                    want, gate_targets=self._gate_targets)
            else:
                snap = self.editor._stats_snapshot(want)
        except Exception as exc:
            self.status_var.set(f"Stats failed: {exc}")
            return
        from .async_task import run_async
        self.status_var.set("Computing statistics…")
        run_async(
            self,
            lambda: self.editor._stats_rows_from_snapshot(snap, want),
            on_done=lambda res: self._populate(res[0], res[1], curated),
            on_error=lambda exc: self.status_var.set(f"Stats failed: {exc}"))

    def _populate(self, rows, cols, curated):
        # Annotate each row with the side(s) it came from (keyed on the gate).
        if curated:
            cols = list(cols)
            if 'Source' not in cols:
                cols.insert(2, 'Source')   # after Sample, Population
            for r in rows:
                srcs = self._gate_sources.get((r.get('Sample'), r.get('__gid__')))
                r['Source'] = '+'.join(sorted(srcs)) if srcs else ''
        self._rows, self._cols = rows, cols

        self.tv.delete(*self.tv.get_children())
        self.tv['columns'] = cols
        for c in cols:
            self.tv.heading(c, text=c)
            anchor = 'w' if c in ('Sample', 'Population', 'Source') else 'e'
            width = (220 if c == 'Population'
                     else 120 if c == 'Sample'
                     else 80 if c == 'Source' else 90)
            self.tv.column(c, anchor=anchor, width=width, stretch=False)

        for r in rows:
            values = [self._fmt(c, r.get(c, '')) for c in cols]
            self.tv.insert('', 'end', values=values)
        msg = (f"{len(rows)} population row(s) across "
               f"{len({r['Sample'] for r in rows})} sample(s)"
               + (" (curated populations)." if curated else "."))
        # Per-channel columns are named by antibody label, so a marker on
        # different fluors across samples already merges into one column.
        # Flag any label that isn't shared by every sample.
        if self.editor._fluor_panel_warning():
            msg += ("  [!] samples differ in fluor panel — non-common "
                    "labels are blank where absent.")
        if curated:
            missing = {nm for nm, _gid in self._gate_targets
                       if nm not in self.editor._samples}
            if missing:
                msg += (f"  [!] {len(missing)} sample(s) not loaded in the "
                        "editor — Add FCS to include their populations.")
        self.status_var.set(msg)

    @staticmethod
    def _fmt(col, val):
        if val == '' or val is None:
            return ''
        if col == 'Count':
            try:
                return f"{int(val):,}"
            except (TypeError, ValueError):
                return str(val)
        if isinstance(val, float):
            if val != val:        # NaN
                return ''
            return f"{val:.3g}"
        return str(val)

    def _export_csv(self):
        if not self._rows:
            self.status_var.set("Nothing to export — no populations.")
            return
        path = filedialog.asksaveasfilename(
            title="Export statistics to CSV",
            defaultextension='.csv',
            initialfile='population_stats.csv',
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not path:
            return
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=self._cols, extrasaction='ignore')
                w.writeheader()
                for r in self._rows:
                    # Write raw numeric values (not the display-formatted
                    # strings) so the CSV is analysis-ready.
                    w.writerow({c: ('' if (isinstance(r.get(c), float)
                                           and r.get(c) != r.get(c))
                                    else r.get(c, '')) for c in self._cols})
            self.status_var.set(f"Exported {len(self._rows)} row(s) → "
                                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            messagebox.showerror(
                "Export statistics failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}", parent=self)


# ══════════════════════════════════════════════════════════════════════════════
# AXIS CONFIG DIALOG (per-channel scale + range)
# ══════════════════════════════════════════════════════════════════════════════
