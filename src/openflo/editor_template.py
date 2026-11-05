"""Gating-template application and lossy-export summaries.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from .editor_base import EditorMixin
from .errors import note
from .tool_window import scrolled_body
from .ui_errors import show_error

BASE = os.path.dirname(os.path.abspath(__file__))


def _read_template(path):
    """Read the template/.wsp/Gating-ML at ``path``.

    Returns ``(gate_dicts, labels, gml_notes, wsp_axes)``; raises on a file
    that cannot be read.
    """
    sys.path.insert(0, BASE)
    from .gate_templates import read_template_gates
    gml_notes = []
    if path.lower().endswith('.xml'):
        from .gate_templates import template_ids
        from .gatingml import read_gatingml
        raw, gml_notes = read_gatingml(path, raw=True)
        gate_dicts, labels = template_ids(raw), None
    else:
        gate_dicts, labels = read_template_gates(path)
    # A .wsp's axes for the sample its gates come from: FlowJo's
    # GeoMean is a mean on them (gating.population_stats).
    wsp_axes = {}
    if path.lower().endswith('.wsp'):
        from .gate_templates import wsp_template_stat_axes
        wsp_axes = wsp_template_stat_axes(path)
    return gate_dicts, labels, gml_notes, wsp_axes


def _template_source(path):
    """'<kind> (<file name>)' for the status line."""
    low = path.lower()
    kind_label = ('FlowJo .wsp' if low.endswith('.wsp') else
                  'Gating-ML' if low.endswith('.xml') else 'JSON template')
    return f'{kind_label} ({os.path.basename(path)})'


def _note_uncompensated(ed, name, gate_dicts, gml_notes):
    """Add a note to ``gml_notes`` for each Gating-ML gate drawn on a
    channel's uncompensated values that sample ``name`` has compensated: it
    counts compensated ones there."""
    from .wsp_gate_coords import wsp_uncompensated_channels
    comp = list(getattr(ed._samples.get(name), 'comp_channels', None) or [])
    for g in gate_dicts:
        for ch in wsp_uncompensated_channels(g, comp):
            msg = (f"'{g.get('label') or '?'}' is drawn on "
                   f"uncompensated {ch}; {name} is compensated "
                   "there, so it counts compensated values")
            if msg not in gml_notes:
                gml_notes.append(msg)


def _merge_stat_axes(smp, wsp_axes):
    """Give ``smp`` the .wsp's stat axes for the channels it has."""
    smp.stat_axes = {
        **(getattr(smp, 'stat_axes', None) or {}),
        **{c: dict(a) for c, a in wsp_axes.items()
           if c in smp.data.columns}}


def _apply_to_targets(ed, targets, gate_dicts, overwrite, path, gml_notes,
                      wsp_axes):
    """Install the template on each of ``targets``.

    Returns ``(mismatches, lost, built_on)``: per sample the count of gates
    on channels it lacks, and the lost / built-on boolean names.
    """
    low = path.lower()
    from_wsp = low.endswith('.wsp')
    from_gml = low.endswith('.xml')
    mismatches = {}
    lost, built_on = [], []
    for name in targets:
        if from_gml:
            _note_uncompensated(ed, name, gate_dicts, gml_notes)
        got_lost, got_built = ed._apply_template_to_sample(
            name, gate_dicts, overwrite, from_wsp=from_wsp,
            from_gml=from_gml)
        smp = ed._samples.get(name)
        if wsp_axes and smp is not None:
            _merge_stat_axes(smp, wsp_axes)
        lost += got_lost
        built_on += got_built
        miss = ed._count_channel_mismatches(name, gate_dicts)
        if miss:
            mismatches[name] = miss
    return mismatches, lost, built_on


def _applied_status(ed, gate_dicts, source, overwrite, targets, lost,
                    built_on):
    """The status line after a template is applied."""
    kinds = ', '.join(sorted({g.get('kind', '?') for g in gate_dicts})) \
            or 'none'
    verb = 'overwrote' if overwrite else 'added to'
    from .pipeline import _flowjo_placeholder_note
    return (f"Applied {len(gate_dicts)} gate(s) [{kinds}] from {source}: "
            f"{verb} {len(targets)} sample(s)"
            f"{ed._lost_operand_note(lost, built_on)}."
            + _flowjo_placeholder_note(gate_dicts))


def _warn_after_apply(ed, gml_notes, mismatches):
    """Show the Gating-ML import notes and the channel-mismatch warning."""
    if gml_notes:
        shown = gml_notes[:12]
        more = len(gml_notes) - len(shown)
        messagebox.showwarning(
            "Gating-ML import notes",
            '\n'.join(f"  • {w}" for w in shown)
            + (f"\n  … and {more} more (see log)" if more else ''),
            parent=ed)

    # Channel-mismatch warning — gates referencing channels a target
    # sample doesn't have will sit inert (gate_to_mask no-ops them).
    if mismatches:
        lines = '\n'.join(f"  • {n}: {c} gate(s)"
                          for n, c in mismatches.items())
        messagebox.showwarning(
            "Template channels missing in some samples",
            "Some gates reference channels that aren't present in "
            "these samples — they'll be inactive (no-op) there until "
            "the channels exist:\n\n" + lines, parent=ed)


def _count_kind(gates_by_sample, kind):
    """How many gates across every sample have this kind."""
    return sum(1 for gates in gates_by_sample.values()
               for g in gates.values()
               if g.get('kind') == kind)


def _axis_items(ed):
    """Lossy-export lines for custom axis scales and display ranges."""
    items = []
    # Custom axis scales (anything the user changed off the default).
    custom_scales = [ch for ch, sc in ed._channel_scale.items()
                     if sc != ed._default_scale_for(ch)]
    if custom_scales:
        items.append(
            f"per-channel axis scale for {len(custom_scales)} channel(s) "
            f"({', '.join(custom_scales[:3])}"
            f"{'…' if len(custom_scales) > 3 else ''})")
    if ed._channel_range:
        items.append(
            f"custom display range for {len(ed._channel_range)} channel(s)")
    return items


def _disabled_count(gates_by_sample):
    """Disabled gates across every sample (cluster/category populations
    are reported on their own lines, so they are left out here)."""
    return sum(
        1 for gates in gates_by_sample.values()
        for g in gates.values()
        if not g.get('enabled', True)
        and g.get('kind') not in ('cluster', 'category', 'boolean',
                                  'autoclean'))


def _unexportable_boolean_count(gates_by_sample):
    """Booleans export as FlowJo AND/OR/NOT nodes; count only those
    FlowJo cannot express (a NOT over several gates, or one over a gate
    that is itself left out)."""
    from .gate_text import wsp_unexportable
    return sum(
        1 for gates in gates_by_sample.values()
        for gid in wsp_unexportable(gates)
        if gates[gid].get('kind') == 'boolean')


def _gate_items(ed):
    """Lossy-export lines for gates a .wsp can't carry."""
    items = []
    gates_by_sample = ed._sample_gates
    n_disabled = _disabled_count(gates_by_sample)
    if n_disabled:
        items.append(
            f"{n_disabled} disabled gate(s) — FlowJo would treat them as "
            "active populations")
    # Cluster populations have no FlowJo geometry — they're dropped on
    # export (the phenotype names go with them).
    n_cluster = _count_kind(gates_by_sample, 'cluster')
    if n_cluster:
        items.append(
            f"{n_cluster} cluster population(s) — no FlowJo equivalent")
    elif ed._cluster_labels:
        items.append("cluster phenotype labels")
    # Category populations (e.g. cell-cycle phases) — no FlowJo geometry.
    n_category = _count_kind(gates_by_sample, 'category')
    if n_category:
        items.append(
            f"{n_category} category population(s) (e.g. cell-cycle) — "
            "no FlowJo equivalent")
    n_boolean = _unexportable_boolean_count(gates_by_sample)
    if n_boolean:
        items.append(
            f"{n_boolean} boolean gate(s) FlowJo cannot express (a NOT "
            "over several gates, or over a gate not exported) — not "
            "exported")
    n_autoclean = _count_kind(gates_by_sample, 'autoclean')
    if n_autoclean:
        items.append(
            f"{n_autoclean} auto-clean gate(s) — recomputed per sample, "
            "no FlowJo equivalent")
    return items


def _qc_items(samples):
    """The acquisition-QC line: the load's QC removed these events before
    any gate; the .wsp has no QC population, so FlowJo counts them
    (measured: 20,000 events -> 19,614 here; a gate at 6,371 in FlowJo
    read 5,985, %Total 31.86 vs 30.51)."""
    qc = [int(getattr(s, 'qc_removed', 0) or 0)
          for s in (samples or {}).values()]
    qc = [n for n in qc if n]
    if not qc:
        return []
    return [f"acquisition QC removed {sum(qc):,} event(s) in {len(qc)} "
            "sample(s) before gating — FlowJo has no QC gate and counts "
            "them, so its counts and %Total will differ"]


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
            gate_dicts, labels, gml_notes, wsp_axes = _read_template(path)
        except Exception as exc:  # noqa: BLE001 - shown in the status line and an error dialog before anything is applied
            self.status_var.set(f"Failed to read template: {exc}")
            show_error(self, "Load template failed",
                       "The gate template could not be read.", exc, path)
            return

        source = _template_source(path)

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
        mismatches, lost, built_on = _apply_to_targets(
            self, targets, gate_dicts, overwrite, path, gml_notes, wsp_axes)
        # Restore whatever sample was active before the batch apply.
        if saved_active in self._samples:
            self._set_active_sample(saved_active)
        self._refresh_gate_list()
        self._schedule_replot(0)

        self.status_var.set(_applied_status(
            self, gate_dicts, source, overwrite, targets, lost, built_on))
        _warn_after_apply(self, gml_notes, mismatches)

    def _count_channel_mismatches(self, name, gate_dicts):
        """How many of `gate_dicts` reference a channel absent from
        sample `name`'s data."""
        s = self._samples.get(name)
        if s is None:
            return 0
        try:
            cols = set(s.data.columns)
        except Exception as exc:  # noqa: BLE001 - noted; a sample without readable columns counts as no mismatches
            note("reading a sample's columns to match a template", exc)
            return 0
        return sum(1 for g in gate_dicts
                   if self._gate_channels(g) - cols)

    def _apply_template_to_sample(self, name, gate_dicts, overwrite,
                                  from_wsp=False, from_gml=False):
        """Install a template's gates into sample `name`.

        Uses the set-active → install → (caller restores) pattern, the
        same as the .wsp-ingest path. Each sample gets independent gate
        ids, so the source ids are rewired per sample.

        overwrite=True  → replace that sample's gate tree
        overwrite=False → append (keeps existing gates; threshold/
                          interval gates with a matching (channel, parent)
                          are replaced in place by _add_gate, as usual).

        Returns ``(lost, built_on)``: the names of the boolean gates installed
        with no operands (one could not be resolved), and of those built on
        one, all of which select nothing (gate_eval.broken_booleans).
        """
        if name not in self._samples:
            return [], []
        self._checkpoint()      # applying a template is one undoable step
        self._set_active_sample(name)
        if overwrite:
            # The id counter carries on: restarted at g1, the template's
            # gates would take the ids of the gates they replaced, and a clip
            # of one of those, pasted back under its id (_paste_gate_tree),
            # would become the operand of the template's boolean.
            self._gates.clear()
            del self._gate_id_order[:]
        # Label-first retargeting: a template gate stamped with an
        # antibody label retargets to THIS sample's detector for that
        # label, so a CD901b gate applies wherever CD901b sits in each
        # sample (different fluors across panels). Built from the
        # target sample's own detector↔label map.
        from .label_alignment import relabel_gate_for_sample, sample_fluor_labels
        from .wsp_gate_coords import gate_scale_fns, map_gate_coords, mark_unknown_scale
        label_to_det = sample_fluor_labels(self._samples[name])
        # Rewrite each source parent_id (.wsp `_import_id` or template
        # `id`) to the fresh editor id _add_gate returns, in parent-first
        # order.
        old_to_new = {}
        added, added_ids = [], []
        # A 1-D template gate may replace a matching gate the sample already
        # had (add mode), never a sibling from this same template.
        before = set(self._gates)
        # A .wsp template holds FlowJo's linear scale units; this sample's
        # fluorescence is stored transformed (see map_gate_coords).
        stored = (self._stored_transforms(name) if from_wsp or from_gml
                  else None)
        from_flowjo = (gate_scale_fns(stored, 'from_linear')
                       if from_wsp else None)
        if from_gml:
            # A Gating-ML file names each axis's scale; its gates go onto
            # this sample's (a polygon on the same scale keeps its vertices
            # exactly). A channel of unknown scale takes linear values and
            # reads NA (mark_unknown_scale).
            from .gatingml import gates_to_stored
            gate_dicts = [mark_unknown_scale(g, stored)
                          for g in gates_to_stored(gate_dicts, stored)]
        for raw in gate_dicts:
            g = relabel_gate_for_sample(raw, label_to_det)
            if from_wsp:
                # A channel of unknown scale cannot take FlowJo's units: the
                # gate is kept but reads NA (mark_unknown_scale).
                g = mark_unknown_scale(map_gate_coords(g, from_flowjo or {}),
                                       stored)
            src_id = g.pop('_import_id', None) or g.pop('id', None)
            parent = g.get('parent_id')
            if parent is not None:
                g['parent_id'] = old_to_new.get(parent)
            gid = self._add_gate(g, replace_1d=before)
            added.append(g)
            added_ids.append(gid)
            if src_id is not None:
                old_to_new[src_id] = gid
        # Boolean operands are template ids as well. Rewritten after the
        # loop, when every id is known; left as they are, with 'Add to
        # existing' they would name the sample's own gates (measured: 88
        # events, 5,035 right).
        from .gate_eval import broken_booleans
        from .gate_text import describe_gate
        from .gating import remap_operands
        for g in added:
            remap_operands(g, old_to_new)
        lost_ids, built_ids = broken_booleans(self._gates)
        mine = set(added_ids)

        def names(ids):
            return [self._gates[i].get('name') or describe_gate(self._gates[i])
                    for i in ids if i in mine]
        return names(lost_ids), names(built_ids)

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
                  font='OpenFloHeading').pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        # Scrollable checkbox list.
        inner = scrolled_body(dlg, pack_kw={'padx': 10, 'pady': (0, 6)})

        cb_vars = {}
        for name in loaded:
            # Default: every loaded sample checked (batch intent).
            var = tk.BooleanVar(value=True)
            cb_vars[name] = var
            label = name + ('  (active)' if name == self._active_sample else '')
            cb = ttk.Checkbutton(inner, text=label, variable=var)
            cb.pack(side='top', anchor='w', padx=2, pady=1)
            self._tip(cb, "Ticked: this sample gets the template's gates "
                      "when you press Apply. Untick it to leave this "
                      "sample's gates as they are.")

        # Mode radio.
        mode_frame = ttk.Frame(dlg)
        mode_frame.pack(side='top', fill='x', padx=10, pady=(0, 6))
        mode_var = tk.StringVar(value='overwrite')
        ttk.Label(mode_frame, text="Mode:").pack(side='left')
        rb_over = ttk.Radiobutton(mode_frame, text="Overwrite gates",
                                  value='overwrite', variable=mode_var)
        rb_over.pack(side='left', padx=(6, 0))
        self._tip(rb_over, "Delete every gate the ticked samples have now, "
                  "then put the template's gates in their place. Undo "
                  "(Ctrl+Z) brings the old gates back.")
        rb_add = ttk.Radiobutton(mode_frame, text="Add to existing",
                                 value='append', variable=mode_var)
        rb_add.pack(side='left', padx=(6, 0))
        self._tip(rb_add, "Keep the gates the ticked samples already have "
                  "and add the template's gates alongside them. A sample "
                  "can end up with two copies of a gate.")

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)
        b_all = ttk.Button(btns, text="Select all",
                           command=lambda: [v.set(True)
                                            for v in cb_vars.values()])
        b_all.pack(side='left')
        self._tip(b_all, "Tick every sample in the list, so Apply changes "
                  "all of them.")
        b_none = ttk.Button(btns, text="Deselect all",
                            command=lambda: [v.set(False)
                                             for v in cb_vars.values()])
        b_none.pack(side='left', padx=(4, 0))
        self._tip(b_none, "Untick every sample, to start again and pick "
                  "just the few you want.")

        def do_apply():
            result['value'] = (
                [n for n, v in cb_vars.items() if v.get()],
                mode_var.get() == 'overwrite',
            )
            dlg.destroy()

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        b_apply = ttk.Button(btns, text="Apply", default='active',
                             command=do_apply)
        b_apply.pack(side='right', padx=(0, 4))
        self._tip(b_apply, "Put the template's gates on every ticked "
                  "sample, in the mode chosen above. One undo step.")
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
        return (_axis_items(self) + _gate_items(self)
                + _qc_items(self._samples))
