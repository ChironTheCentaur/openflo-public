"""Population statistics window: per-population counts + per-channel stats.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

from .errors import note
from .ui_errors import show_error
from .ui_tips import tip

# What each statistic column MEANS, for the column checkboxes. A tooltip that
# only expands the label ("%Parent: percent of parent") is not worth showing;
# these say what the number is computed against, which is the part that gets
# misread between figures.
_STAT_HELP = {
    'Count': "Number of events in the population, after gating and any "
             "downsampling currently applied.",
    '%Parent': "Percent of the IMMEDIATE parent gate. The usual number for "
               "describing a single gating step.",
    '%Grandparent': "Percent of the parent's parent gate, e.g. a subset as "
                    "a percent of all lymphocytes when its parent is a "
                    "singlet gate. Blank for a gate drawn on all events, "
                    "which has no grandparent.",
    '%Total': "Percent of all events in the sample. Differs sharply from "
              "%Parent for a deep gate, so say which one a figure uses.",
    'Median': "Median intensity — the robust centre. Preferred over the mean "
              "for flow data, which is rarely symmetric. On the compensated "
              "linear scale, not the plot's display transform.",
    'Mean': "Arithmetic mean intensity. Sensitive to a bright tail; compare "
            "against Median before reporting it. On the compensated linear "
            "scale.",
    'GeoMean': "Geometric mean. By default the mean in the channel's "
               "display (transform) space, back-transformed, as FlowJo "
               "computes it, so it depends on that axis: a sample imported "
               "from a FlowJo workspace uses the workspace's axis for each "
               "channel it declares (the column heading names it), any "
               "other OpenFlo's display axis. Preferences can switch it to "
               "the textbook exp(mean of log x) over positive events. Suits "
               "the log-normal-ish distributions of stained channels.",
    'CV': "Coefficient of variation (SD / mean, %). Spread measure used for "
          "bead QC and to judge how tight a population is. On the "
          "compensated linear scale.",
    'rCV': "Robust CV: 100 x rSD / median, so a few outliers do not "
           "inflate it. rSD follows the form chosen in Preferences.",
    'SD': "Standard deviation of intensity (the sample SD, dividing by "
          "n - 1; CV = SD / Mean). NA for a single event. On the "
          "compensated linear scale.",
    'rSD': "Robust SD, the spread rCV divides by the median. By default "
           "FlowJo's 1/2 (P84.13 - P15.87); Preferences can switch it to "
           "1.4826 x MAD. A few outliers do not inflate it.",
    'MAD': "Median absolute deviation from the median, unscaled (x 1.4826 "
           "estimates the SD of a normal population). Robust spread.",
    'Percentiles': "One column per percentile per channel, e.g. P5 and P95 "
                   "bound the middle 90 % of the population. Type the "
                   "percentiles you want, separated by commas.",
    'MFI ratio': "The population's median divided by the SAME population's "
                 "median in the reference sample, per channel: the usual "
                 "stained over unstained (or FMO) MFI ratio. Pick the "
                 "control as the reference. Blank where the reference lacks "
                 "the population or its median is not positive.",
}

# The percentiles a fresh window offers.
DEFAULT_PERCENTILES = '5, 25, 75, 95'



class StatisticsWindow(tk.Toplevel):
    """Population-statistics table over the editor's loaded samples.

    Rows = sample × population (gate node, evaluated as the cumulative
    gate chain). Columns are modular — toggle Count / %Parent / %Total
    and per-channel Median / Mean / GeoMean / CV / rCV. Export the current
    table to CSV.
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
        # Three short rows rather than one long one, which ran off a laptop
        # screen once the list grew past eight statistics.
        self._stat_vars = {}
        defaults = {'Count', '%Parent', '%Total', 'Median'}
        for i, (title, stats) in enumerate((
                ("Populations:", editor.STAT_POP),
                ("Per channel:", editor.STAT_CHAN))):
            opts = ttk.Frame(self, padding=(10, 8 if i == 0 else 0, 10, 2))
            opts.pack(side='top', fill='x')
            ttk.Label(opts, text=title, width=12,
                      font='OpenFloHeading').pack(side='left',
                                                              padx=(0, 8))
            for stat in stats:
                var = tk.BooleanVar(value=(stat in defaults))
                self._stat_vars[stat] = var
                _cb = ttk.Checkbutton(opts, text=stat, variable=var,
                                      command=self._refresh)
                _cb.pack(side='left', padx=(0, 6))
                tip(editor, _cb, _STAT_HELP.get(
                    stat, f"Show the {stat} column in the table."))
        extra = ttk.Frame(self, padding=(10, 0, 10, 4))
        extra.pack(side='top', fill='x')
        ttk.Label(extra, text="", width=12).pack(side='left', padx=(0, 8))
        self._pct_on = tk.BooleanVar(value=False)
        _cb = ttk.Checkbutton(extra, text="Percentiles", variable=self._pct_on,
                              command=self._refresh)
        _cb.pack(side='left')
        tip(editor, _cb, _STAT_HELP['Percentiles'])
        self._pct_text = tk.StringVar(value=DEFAULT_PERCENTILES)
        _ent = ttk.Entry(extra, textvariable=self._pct_text, width=16)
        _ent.pack(side='left', padx=(4, 14))
        _ent.bind('<Return>', lambda _e: self._refresh())
        _ent.bind('<FocusOut>', lambda _e: (self._pct_on.get()
                                            and self._refresh()))
        tip(editor, _ent, "Which percentiles, separated by commas (0 to "
                          "100). Press Enter to apply.")
        self._ratio_on = tk.BooleanVar(value=False)
        _cb = ttk.Checkbutton(extra, text="MFI ratio vs", command=self._refresh,
                              variable=self._ratio_on)
        _cb.pack(side='left')
        tip(editor, _cb, _STAT_HELP['MFI ratio'])
        self._ratio_ref = tk.StringVar(value='')
        self._ratio_combo = ttk.Combobox(extra, textvariable=self._ratio_ref,
                                         state='readonly', width=22)
        self._ratio_combo.pack(side='left', padx=(4, 0))
        self._ratio_combo.bind('<<ComboboxSelected>>',
                               lambda _e: self._refresh())
        tip(editor, self._ratio_combo,
            "The reference sample for the MFI ratio, usually the unstained "
            "or FMO control.")

        # ── Buttons + status ─────────────────────────────────────────────
        # Packed BEFORE the table: the packer gives space in packing order, so
        # whatever is packed last is what disappears when the window is short.
        # That was the button row; now it is a few table rows.
        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  padding=(10, 0, 10, 2)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        _b_refresh = ttk.Button(btns, text="Refresh", command=self._refresh)
        _b_refresh.pack(side='left')
        tip(editor, _b_refresh,
            "Recompute every row against the current gates and samples. "
            "Needed after editing gates in the editor; the column toggles "
            "refresh on their own.")
        _b_csv = ttk.Button(btns, text="Export CSV…", command=self._export_csv)
        _b_csv.pack(side='left', padx=(6, 0))
        tip(editor, _b_csv,
            "Save the table exactly as shown — same rows, same columns, same "
            "order — for a paper table or a stats package. The separator "
            "and decimal mark follow Preferences → Export.")
        _b_xlsx = ttk.Button(btns, text="Save as Excel…",
                             command=self._export_xlsx)
        _b_xlsx.pack(side='left', padx=(6, 0))
        tip(editor, _b_xlsx,
            "Save an Excel workbook: the whole table on one sheet, then one "
            "sheet per statistic with populations down the side and samples "
            "across. Opens correctly in any language of Excel.")
        _b_copy = ttk.Button(btns, text="Copy table", command=self._copy_table)
        _b_copy.pack(side='left', padx=(6, 0))
        tip(editor, _b_copy,
            "Copy the table, with its headings, to the clipboard, ready to "
            "paste into Excel, Prism or a document. The values are the "
            "exported ones (full precision), tab separated.")
        # Per-side bulk import — each REPLACES the curated set (override).
        # Dragging a gate row from either tree APPENDS instead.
        _b_imp_ed = ttk.Button(btns, text="Import S&G gates",
                               command=self._import_all_editor)
        _b_imp_ed.pack(side='left', padx=(12, 0))
        tip(editor, _b_imp_ed,
            "REPLACE the current row set with every gate from the Samples & "
            "Gates tree. This overrides a curated selection rather than "
            "adding to it.")
        _b_imp_ws = ttk.Button(btns, text="Import workspace",
                               command=self._import_all_workspace)
        _b_imp_ws.pack(side='left', padx=(6, 0))
        tip(editor, _b_imp_ws,
            "REPLACE the current row set with everything in the Pipeline "
            "Workspace. Also an override, not an addition.")
        _b_clear = ttk.Button(btns, text="Clear", command=self._clear_targets)
        _b_clear.pack(side='left', padx=(6, 0))
        tip(editor, _b_clear,
            "Drop the curated row set and go back to showing all "
            "populations. Does not delete any gate.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(side='right')

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
            except tk.TclError:
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
        except tk.TclError:
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
        """The statistic names ticked, with each percentile asked for
        (``P5``, ...) and the MFI ratio when it has a reference."""
        from .gating import MFI_RATIO, parse_percentiles
        want = {s for s, v in self._stat_vars.items() if v.get()}
        self._pct_error = ''
        if self._pct_on.get():
            try:
                want.update(parse_percentiles(self._pct_text.get()))
            except ValueError as exc:
                self._pct_error = f"Percentiles: {exc}."
        if self._ratio_on.get() and self._ratio_ref.get():
            want.add(MFI_RATIO)
        return want

    def _sync_ratio_choices(self):
        """Offer the loaded samples as MFI-ratio references; keep the
        choice while it is loaded, else take the first sample."""
        names = [n for n in self.editor._sample_order
                 if n in self.editor._samples]
        self._ratio_combo['values'] = names
        if self._ratio_ref.get() not in names:
            self._ratio_ref.set(names[0] if names else '')

    def _refresh(self):
        self._sync_ratio_choices()
        want = self._selected_stats()
        from .gating import MFI_RATIO
        ref = self._ratio_ref.get() if MFI_RATIO in want else None
        ref_snap = None
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
            # A curated table may leave the reference's populations out;
            # the ratio still needs their medians.
            if ref and curated:
                ref_snap = self.editor._stats_snapshot(want, samples=[ref])
        except Exception as exc:
            self.status_var.set(f"Stats failed: {exc}")
            return
        from .async_task import run_async
        self.status_var.set("Computing statistics…")
        run_async(
            self,
            lambda: self.editor._stats_rows_from_snapshot(
                snap, want, mfi_reference=ref, reference_snapshot=ref_snap),
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
        # Samples set aside from the comparisons stay in the table (their
        # numbers are real), marked with the reason.
        aside = getattr(self.editor, '_sample_set_aside', {}) or {}
        if aside:
            cols = list(cols)
            if 'Set aside' not in cols:
                cols.insert(3 if 'Source' in cols else 2, 'Set aside')
            for r in rows:
                r['Set aside'] = aside.get(r.get('Sample'), '')
        self._rows, self._cols = rows, cols

        self.tv.delete(*self.tv.get_children())
        self.tv['columns'] = cols
        shown = [{c: (self._population_label(r) if c == 'Population'
                      else self._cell(r, c)) for c in cols} for r in rows]
        for c in cols:
            self.tv.heading(c, text=self._heading(c))
            anchor = 'w' if c in self._ID_COLS else 'e'
            width = (self._population_width([v['Population'] for v in shown])
                     if c == 'Population'
                     else 120 if c == 'Sample'
                     else 80 if c == 'Source'
                     else 140 if c == 'Set aside'
                     # Wide enough for its heading ('Median FL2-A (linear)'),
                     # which was cut to 'Median FL' at a fixed 90.
                     else self._text_width([self._heading(c)], 90, 240))
            self.tv.column(c, anchor=anchor, width=width, stretch=False)

        for v in shown:
            self.tv.insert('', 'end', values=[v[c] for c in cols])
        msg = (f"{len(rows)} population row(s) across "
               f"{len({r['Sample'] for r in rows})} sample(s)"
               + (" (curated populations)." if curated else "."))
        # The plot shows fluor channels through a display transform (logicle
        # by default); the numbers here are not in those units. Say which.
        from .gating import MFI_RATIO, is_channel_stat_column
        if any(is_channel_stat_column(c, self.editor.STAT_CHAN)
               for c in cols):
            msg += ("  Intensity statistics are on the compensated linear "
                    "scale.")
            # A column from an older OpenFlo CSV whose display transform is
            # unknown has no linear values: its cells are blank, and why.
            unknown = {}
            for r in rows:
                for lbl in r.get('__scale_unknown__', ()):
                    names = unknown.setdefault(r['Sample'], [])
                    if lbl not in names:
                        names.append(lbl)
            if unknown:
                msg += ("  [!] Left blank, scale unknown (older OpenFlo CSV): "
                        + "; ".join(f"{nm}: {', '.join(lbls)}"
                                    for nm, lbls in unknown.items()) + ".")
            # FlowJo's GeoMean is a mean on the display axis: say which axis
            # each sample's was taken on when a FlowJo import set one.
            from .gating import OPENFLO_AXIS_TEXT
            geo_axes = {}
            for r in rows:
                for lbl, ax in (r.get('__geomean_axis__') or {}).items():
                    geo_axes.setdefault((ax, lbl), set()).add(r['Sample'])
            if any(ax != OPENFLO_AXIS_TEXT for ax, _l in geo_axes):
                msg += ("  GeoMean axis: " + "; ".join(
                    f"{lbl} on the {ax} ({', '.join(sorted(map(str, ss)))})"
                    for (ax, lbl), ss in sorted(geo_axes.items())) + ".")
            from .gating import GUESSED_TAG
            if any(c.endswith(f' {GUESSED_TAG}') for c in cols):
                msg += ("  [!] '(linear, guessed logicle)' columns come from "
                        "CSVs with no transform record, logicle assumed: a "
                        "guess, not a measurement.")
        if any(str(c).startswith(f'{MFI_RATIO} ') for c in cols):
            msg += (f"  MFI ratio = median / the same population's median "
                    f"in {self._ratio_ref.get()}.")
        if getattr(self, '_pct_error', ''):
            msg += f"  [!] {self._pct_error}"
        # Per-channel columns are named by antibody label, so a marker on
        # different fluors across samples already merges into one column.
        # Flag any label that isn't shared by every sample.
        if self.editor._fluor_panel_warning():
            msg += ("  [!] samples differ in fluor panel — non-common "
                    "labels are blank where absent.")
        # A population whose gates cannot be computed reads NA, not 0.
        unknown_pops = {}
        for r in rows:
            why = r.get('__unknown__')
            if why:
                unknown_pops.setdefault(why, []).append(
                    f"{r['Sample']}: {r['Population']}")
        if unknown_pops:
            msg += ("  [!] NA = population cannot be computed ("
                    + "; ".join(f"{why}: {len(p)}"
                                for why, p in unknown_pops.items()) + ").")
        if curated:
            missing = {nm for nm, _gid in self._gate_targets
                       if nm not in self.editor._samples}
            if missing:
                msg += (f"  [!] {len(missing)} sample(s) not loaded in the "
                        "editor — Add FCS to include their populations.")
        self.status_var.set(msg)

    # The Population column: the gate's own name, indented under its parent,
    # rather than the whole path. With the path, a parent and its child began
    # with the same text and read alike in a column too narrow to reach where
    # they differ. The exported CSV keeps the full path, which is unambiguous
    # on its own.
    _INDENT = '\u2003'            # an em space; Treeview keeps it, unlike ' '
    _POP_WIDTH_MIN, _POP_WIDTH_MAX = 220, 520

    def _population_label(self, row):
        """``row``'s Population cell: the gate's name, indented one step per
        ancestor, or the full path when the gate is not in the editor."""
        path = str(row.get('Population', ''))
        gates = self.editor._sample_gates.get(row.get('Sample'), {})
        gid = row.get('__gid__')
        if gid not in gates:
            return path
        from .gating import population_path
        sample = self.editor._samples.get(row.get('Sample'))
        leaf = population_path({gid: gates[gid]}, gid, transforms=getattr(
            sample, 'data_transforms', None))
        depth, seen, cur = 0, {gid}, gates[gid].get('parent_id')
        while cur in gates and cur not in seen:
            seen.add(cur)
            depth += 1
            cur = gates[cur].get('parent_id')
        if not depth:
            return leaf
        return self._INDENT * (2 * depth - 1) + '\u2514 ' + leaf

    def _population_width(self, labels):
        """Pixels for the Population column: its longest label, within
        bounds (beyond the maximum the column scrolls sideways)."""
        return self._text_width(labels, self._POP_WIDTH_MIN,
                                self._POP_WIDTH_MAX)

    @staticmethod
    def _text_width(texts, lo, hi):
        """Pixels a column needs for the longest of ``texts``, within
        ``lo``..``hi``, in the font the table draws with."""
        try:
            import tkinter.font as tkfont
            font = tkfont.nametofont('TkDefaultFont')
            need = max((font.measure(t) for t in texts), default=0) + 24
        except Exception:
            return lo
        return int(min(max(need, lo), hi))

    def _heading(self, col):
        """A column's title, on screen and in the exported CSV. Per-channel
        intensity statistics name the scale they are computed on: the CSV
        leaves the window, and with it the status bar that says so."""
        from .gating import GUESSED_TAG, MFI_RATIO, is_channel_stat_column
        if str(col).startswith(f'{MFI_RATIO} '):
            return f"{col} (vs {self._ratio_ref.get()})"
        if is_channel_stat_column(col, self.editor.STAT_CHAN):
            # A guessed transform's numbers are never titled as certain.
            if col.endswith(f' {GUESSED_TAG}'):
                return (f"{col[:-len(GUESSED_TAG) - 1]} "
                        "(linear, guessed logicle)")
            axis = self._geomean_axis(col)
            if axis:
                return f"{col} (linear, {axis})"
            return f"{col} (linear)"
        return col

    def _geomean_axis(self, col):
        """For a GeoMean column whose FlowJo-form values were averaged on a
        FlowJo workspace's axis (gating.population_stats), that axis' name;
        'axis differs by sample' when its rows used different axes; None
        when every row used OpenFlo's display axis (the heading stays
        'GeoMean X (linear)', as exports have always named it) or the
        textbook form, which has no axis."""
        from .gating import geomean_axis_note
        if not str(col).startswith('GeoMean '):
            return None
        return geomean_axis_note(getattr(self, '_rows', None),
                                 str(col)[len('GeoMean '):])

    # Identity columns; every other column is a statistic.
    _ID_COLS = ('Sample', 'Population', 'Source', 'Set aside')

    @classmethod
    def _cell(cls, row, col, fmt=True):
        """A row's value for `col` as shown (fmt) or exported. A population
        that cannot be computed (``__unknown__``) reads 'NA' in every
        statistic: it is missing, and a blank or 0 would read as measured."""
        if row.get('__unknown__') and col not in cls._ID_COLS:
            return 'NA'
        val = row.get(col, '')
        if fmt:
            return cls._fmt(col, val)
        return '' if isinstance(val, float) and val != val else val

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
            if str(col).startswith('%'):
                # A frequency: one decimal, as everywhere else ('93.0', not
                # '93' or '92.96'); the column header carries the %.
                return f"{val:.1f}"
            # Linear intensities run to 10^5; '.3g' would show 1,097 as
            # '1.1e+03'. Whole units with separators from 1,000 up.
            if abs(val) >= 1000:
                return f"{val:,.0f}"
            return f"{val:.3g}"
        return str(val)

    def _table_lines(self):
        """The table as exported: a heading row, then one row per
        population, raw values (NA where a population cannot be
        computed)."""
        out = [[self._heading(c) for c in self._cols]]
        for r in self._rows:
            out.append([self._cell(r, c, fmt=False) for c in self._cols])
        return out

    def _copy_table(self):
        """Copy the table, tab separated, to the clipboard."""
        if not self._rows:
            self.status_var.set("Nothing to copy — no populations.")
            return
        from .clipboard_export import copy_table
        copy_table(self, self._table_lines())
        self.status_var.set(f"Copied {len(self._rows)} row(s) to the "
                            "clipboard.")

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
            from .table_export import write_rows_csv
            # Raw numeric values (not the display-formatted strings), so the
            # CSV is analysis-ready.
            lines = self._table_lines()
            write_rows_csv(path, lines[0], lines[1:], na_rep='')
            self.status_var.set(f"Exported {len(self._rows)} row(s) → "
                                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            show_error(self, "Export statistics failed",
                       "The statistics could not be exported.", exc, path)
            return
        self._audit_export(path, 'csv')

    def _audit_export(self, path, kind):
        """Record a written export in the editor's audit log, with the
        samples its rows come from (see fixes.stale_exports)."""
        names = list(dict.fromkeys(str(r.get('Sample')) for r in self._rows
                                   if r.get('Sample') is not None))
        try:
            self.editor._audit('statistics.export', path=path, kind=kind,
                               n_rows=len(self._rows), samples=names)
        except Exception as exc:
            note("recording the export in the audit trail", exc)

    def _id_column_count(self):
        """How many leading columns identify a row rather than measure it."""
        n = 0
        for c in self._cols:
            if c not in self._ID_COLS:
                break
            n += 1
        return n

    def _export_xlsx(self):
        """Save the table as an Excel workbook, a sheet per statistic."""
        if not self._rows:
            self.status_var.set("Nothing to export — no populations.")
            return
        path = filedialog.asksaveasfilename(
            title="Save statistics as an Excel workbook",
            defaultextension='.xlsx',
            initialfile='population_stats.xlsx',
            filetypes=[('Excel workbook', '*.xlsx'), ('All files', '*.*')])
        if not path:
            return
        try:
            from .table_export import about_rows, statistics_sheets, write_workbook
            lines = self._table_lines()
            sheets = statistics_sheets(lines[0], lines[1:],
                                       self._id_column_count())
            write_workbook(path, sheets, about=about_rows(), na_rep='')
            self.status_var.set(
                f"Saved {len(self._rows)} row(s), {len(sheets)} sheet(s) → "
                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            show_error(self, "Save Excel workbook failed",
                       "The workbook could not be saved.", exc, path)
            return
        self._audit_export(path, 'xlsx')


# ══════════════════════════════════════════════════════════════════════════════
# AXIS CONFIG DIALOG (per-channel scale + range)
# ══════════════════════════════════════════════════════════════════════════════
