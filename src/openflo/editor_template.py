"""Gating-template application and lossy-export summaries.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from .editor_base import EditorMixin

BASE = os.path.dirname(os.path.abspath(__file__))


class TemplateMixin(EditorMixin):
    """Apply a saved gating template to samples (with channel-mismatch checks) and summarise lossy .wsp exports."""

    def _apply_template_path(self, path):
        """Read a template/.wsp at ``path`` and apply it to chosen samples.
        Shared by the file-picker and the built-in library menu.

        Native JSON schema: {"gates": [gate_dict, …], "labels": {ch: lbl, …}}.
        """
        # Read the file first (separate from applying it, so a parse
        # failure reports cleanly before we pop the target dialog).
        try:
            sys.path.insert(0, BASE)
            from .pipeline import read_template_gates
            gate_dicts, labels = read_template_gates(path)
        except Exception as exc:
            self.status_var.set(f"Failed to read template: {exc}")
            messagebox.showerror(
                "Load template failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}", parent=self)
            return

        kind_label = 'FlowJo .wsp' if path.lower().endswith('.wsp') \
                     else 'JSON template'
        source = f'{kind_label} ({os.path.basename(path)})'

        if self._active_sample is None or not self._samples:
            self.status_var.set("Load a sample first, then load a template.")
            return

        if labels:
            self._channel_labels.update(
                {k: str(v) for k, v in labels.items()})
            if self._channels:
                self._populate_channel_combos()

        # Ask which loaded samples to apply to + whether to overwrite or
        # add to each target's existing gates.
        choice = self._ask_template_apply()
        if choice is None:
            return                              # cancelled
        targets, overwrite = choice
        if not targets:
            self.status_var.set("No target samples selected.")
            return

        saved_active = self._active_sample
        mismatches = {}
        lost = []
        from_wsp = path.lower().endswith('.wsp')
        for name in targets:
            lost += self._apply_template_to_sample(name, gate_dicts, overwrite,
                                                   from_wsp=from_wsp)
            miss = self._count_channel_mismatches(name, gate_dicts)
            if miss:
                mismatches[name] = miss
        # Restore whatever sample was active before the batch apply.
        if saved_active in self._samples:
            self._set_active_sample(saved_active)
        self._refresh_gate_list()
        self._schedule_replot(0)

        kinds = ', '.join(sorted({g.get('kind', '?') for g in gate_dicts})) \
                or 'none'
        verb = 'overwrote' if overwrite else 'added to'
        from .pipeline import _flowjo_placeholder_note
        self.status_var.set(
            f"Applied {len(gate_dicts)} gate(s) [{kinds}] from {source}: "
            f"{verb} {len(targets)} sample(s){self._lost_operand_note(lost)}."
            + _flowjo_placeholder_note(gate_dicts))

        # Channel-mismatch warning — gates referencing channels a target
        # sample doesn't have will sit inert (gate_to_mask no-ops them).
        if mismatches:
            lines = '\n'.join(f"  • {n}: {c} gate(s)"
                              for n, c in mismatches.items())
            messagebox.showwarning(
                "Template channels missing in some samples",
                "Some gates reference channels that aren't present in "
                "these samples — they'll be inactive (no-op) there until "
                "the channels exist:\n\n" + lines, parent=self)

    def _count_channel_mismatches(self, name, gate_dicts):
        """How many of `gate_dicts` reference a channel absent from
        sample `name`'s data."""
        s = self._samples.get(name)
        if s is None:
            return 0
        try:
            cols = set(s.data.columns)
        except Exception:
            return 0
        return sum(1 for g in gate_dicts
                   if self._gate_channels(g) - cols)

    def _apply_template_to_sample(self, name, gate_dicts, overwrite,
                                  from_wsp=False):
        """Install a template's gates into sample `name`.

        Uses the set-active → install → (caller restores) pattern, the
        same as the .wsp-ingest path. Each sample gets independent gate
        ids, so the source ids are rewired per sample.

        overwrite=True  → replace that sample's gate tree
        overwrite=False → append (keeps existing gates; threshold/
                          interval gates with a matching (channel, parent)
                          are replaced in place by _add_gate, as usual).

        Returns the names of the boolean gates installed with no operands
        (one could not be resolved), which select nothing.
        """
        if name not in self._samples:
            return []
        self._checkpoint()      # applying a template is one undoable step
        self._set_active_sample(name)
        if overwrite:
            # The id counter carries on: restarted at g1, the template's
            # gates took the ids of the gates they replaced, and a clip of one
            # of those, pasted back under its id (_paste_gate_tree), became
            # the operand of the template's boolean.
            self._gates.clear()
            del self._gate_id_order[:]
        # Label-first retargeting: a template gate stamped with an
        # antibody label retargets to THIS sample's detector for that
        # label, so a CD901b gate applies wherever CD901b sits in each
        # sample (different fluors across panels). Built from the
        # target sample's own detector↔label map.
        from .pipeline import (
            _sample_fluor_labels,
            gate_scale_fns,
            map_gate_coords,
            relabel_gate_for_sample,
        )
        label_to_det = _sample_fluor_labels(self._samples[name])
        # Rewrite each source parent_id (.wsp `_import_id` or template
        # `id`) to the fresh editor id _add_gate returns, in parent-first
        # order.
        old_to_new = {}
        added = []
        # A 1-D template gate may replace a matching gate the sample already
        # had (add mode), never a sibling from this same template.
        before = set(self._gates)
        # A .wsp template holds FlowJo's linear scale units; this sample's
        # fluorescence is stored transformed (see map_gate_coords).
        from_flowjo = (gate_scale_fns(self._stored_transforms(name),
                                      'from_linear')
                       if from_wsp else None)
        for raw in gate_dicts:
            g = relabel_gate_for_sample(raw, label_to_det)
            if from_flowjo:
                g = map_gate_coords(g, from_flowjo)
            src_id = g.pop('_import_id', None) or g.pop('id', None)
            parent = g.get('parent_id')
            if parent is not None:
                g['parent_id'] = old_to_new.get(parent)
            gid = self._add_gate(g, replace_1d=before)
            added.append(g)
            if src_id is not None:
                old_to_new[src_id] = gid
        # Boolean operands are template ids as well. Rewritten after the
        # loop, when every id is known; with 'Add to existing' the old ids
        # named the sample's own gates (measured: 88 events, 5,035 right).
        from .gating import remap_operands
        from .pipeline import describe_gate
        for g in added:
            remap_operands(g, old_to_new)
        return [g.get('name') or describe_gate(g) for g in added
                if g.get('kind') == 'boolean' and not g.get('operands')]

    def _ask_template_apply(self):
        """Modal dialog: choose target samples (multiselect) + apply mode
        (overwrite vs add-to). Returns (targets:list[str], overwrite:bool)
        or None if cancelled. Blocks until dismissed."""
        loaded = [n for n in self._sample_order if n in self._samples]
        result: dict[str, tuple[list[str], bool] | None] = {'value': None}

        dlg = tk.Toplevel(self)
        dlg.title("Apply template to…")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.geometry("420x460")
        dlg.minsize(360, 320)

        ttk.Label(dlg, text="Apply the template to these samples:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        # Scrollable checkbox list.
        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=(0, 6))
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))

        cb_vars = {}
        for name in loaded:
            # Default: every loaded sample checked (batch intent).
            var = tk.BooleanVar(value=True)
            cb_vars[name] = var
            label = name + ('  (active)' if name == self._active_sample else '')
            ttk.Checkbutton(inner, text=label, variable=var).pack(
                side='top', anchor='w', padx=2, pady=1)

        # Mode radio.
        mode_frame = ttk.Frame(dlg)
        mode_frame.pack(side='top', fill='x', padx=10, pady=(0, 6))
        mode_var = tk.StringVar(value='overwrite')
        ttk.Label(mode_frame, text="Mode:").pack(side='left')
        ttk.Radiobutton(mode_frame, text="Overwrite gates",
                        value='overwrite', variable=mode_var).pack(
            side='left', padx=(6, 0))
        ttk.Radiobutton(mode_frame, text="Add to existing",
                        value='append', variable=mode_var).pack(
            side='left', padx=(6, 0))

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)
        ttk.Button(btns, text="Select all",
                   command=lambda: [v.set(True) for v in cb_vars.values()]
                   ).pack(side='left')
        ttk.Button(btns, text="Deselect all",
                   command=lambda: [v.set(False) for v in cb_vars.values()]
                   ).pack(side='left', padx=(4, 0))

        def do_apply():
            result['value'] = (
                [n for n, v in cb_vars.items() if v.get()],
                mode_var.get() == 'overwrite',
            )
            dlg.destroy()

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply",
                   command=do_apply).pack(side='right', padx=(0, 4))
        dlg.bind('<Escape>', lambda *_: dlg.destroy())

        self.wait_window(dlg)
        return result['value']

    def _wsp_lossy_summary(self):
        """List the OpenFlo-only state that a FlowJo .wsp export can't
        carry, given the CURRENT editor state. Empty list → a clean
        export with nothing surprising lost.

        Gate geometry (incl. ellipsoid / quadrant) and the compensation
        matrix DO survive — those aren't reported. We only flag state
        that has no slot in the FlowJo schema AND is actually present:
          - custom per-channel axis scales / ranges (set via the ⚙ dialog)
          - disabled gates (a .wsp would write them as live populations,
            silently changing the analysis)
          - cluster phenotype labels
          - events the load's acquisition QC removed (FlowJo counts them)
        Gate colours are mentioned too, but on their own don't trigger
        the warning (FlowJo reassigns its own colours; not surprising).
        """
        items = []
        # Custom axis scales (anything the user changed off the default).
        custom_scales = [ch for ch, sc in self._channel_scale.items()
                         if sc != self._default_channel_scale]
        if custom_scales:
            items.append(
                f"per-channel axis scale for {len(custom_scales)} channel(s) "
                f"({', '.join(custom_scales[:3])}"
                f"{'…' if len(custom_scales) > 3 else ''})")
        if self._channel_range:
            items.append(
                f"custom display range for {len(self._channel_range)} channel(s)")
        # Disabled gates across every sample (cluster/category populations
        # are reported on their own lines below, so exclude them here).
        n_disabled = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if not g.get('enabled', True)
            and g.get('kind') not in ('cluster', 'category', 'boolean',
                                      'autoclean'))
        if n_disabled:
            items.append(
                f"{n_disabled} disabled gate(s) — FlowJo would treat them as "
                "active populations")
        # Cluster populations have no FlowJo geometry — they're dropped on
        # export (the phenotype names go with them).
        n_cluster = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'cluster')
        if n_cluster:
            items.append(
                f"{n_cluster} cluster population(s) — no FlowJo equivalent")
        elif self._cluster_labels:
            items.append("cluster phenotype labels")
        # Category populations (e.g. cell-cycle phases) — no FlowJo geometry.
        n_category = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'category')
        if n_category:
            items.append(
                f"{n_category} category population(s) (e.g. cell-cycle) — "
                "no FlowJo equivalent")
        n_boolean = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'boolean')
        if n_boolean:
            items.append(
                f"{n_boolean} boolean gate(s) (AND/OR/NOT) — not exported")
        n_autoclean = sum(
            1 for gates in self._sample_gates.values()
            for g in gates.values()
            if g.get('kind') == 'autoclean')
        if n_autoclean:
            items.append(
                f"{n_autoclean} auto-clean gate(s) — recomputed per sample, "
                "no FlowJo equivalent")
        # The load's acquisition QC removed these events before any gate;
        # the .wsp has no QC population, so FlowJo counts them (measured:
        # 20,000 events -> 19,614 here; a gate at 6,371 in FlowJo read
        # 5,985, %Total 31.86 vs 30.51).
        qc = [int(getattr(s, 'qc_removed', 0) or 0)
              for s in (getattr(self, '_samples', None) or {}).values()]
        qc = [n for n in qc if n]
        if qc:
            items.append(
                f"acquisition QC removed {sum(qc):,} event(s) in {len(qc)} "
                "sample(s) before gating — FlowJo has no QC gate and counts "
                "them, so its counts and %Total will differ")
        return items
