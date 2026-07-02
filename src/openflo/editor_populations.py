"""Cluster / label / cell-cycle import and population FCS export.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np

from .editor_base import EditorMixin


class PopulationsMixin(EditorMixin):
    """Import clusters / label columns / cell-cycle calls as populations, wire the Populations menu, and export populations to FCS."""

    def _sample_cluster_ids(self, name):
        """Sorted unique cluster ids in `name`'s data, or [] when the
        sample isn't clustered."""
        s = self._samples.get(name)
        if s is None or getattr(s, 'data', None) is None:
            return []
        df = s.data
        if 'cluster' not in df.columns:
            return []
        vals = df['cluster'].dropna().unique()
        out = []
        for v in vals:
            try:
                out.append(int(v))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))

    def _cluster_display_name(self, name, cid):
        """Phenotype label for one cluster, falling back to 'Cluster N'."""
        lbls = self._cluster_labels.get(name) or {}
        nm = lbls.get(cid)
        if nm is None:
            nm = lbls.get(str(cid))
        return nm or f'Cluster {cid}'

    def _label_columns_present(self):
        """Known label columns that at least one loaded sample carries."""
        present = []
        for col in self.LABEL_COLUMNS:
            for s in self._samples.values():
                df = getattr(s, 'data', None)
                if df is not None and col in df.columns:
                    present.append(col)
                    break
        return present

    def _sample_label_values(self, name, col):
        """Sorted distinct values of `col` in `name`'s data, minus the
        unassigned sentinel. [] when the column is absent."""
        s = self._samples.get(name)
        if s is None or getattr(s, 'data', None) is None:
            return []
        df = s.data
        if col not in df.columns:
            return []
        skip = self.LABEL_COLUMNS.get(col, (None, None))[1]
        vals = [v for v in df[col].dropna().unique() if v != skip]
        try:
            return sorted(vals)
        except TypeError:
            return sorted(vals, key=str)

    def _fill_populations_menu(self, menu):
        """(Re)build the Edit → Populations cascade on open — one entry per
        label column present (clusters / FlowSOM / cell-cycle)."""
        menu.delete(0, 'end')
        present = self._label_columns_present()
        if not present:
            menu.add_command(label="No cluster / FlowSOM columns loaded",
                             state='disabled')
            return
        for col in present:
            disp = self.LABEL_COLUMNS[col][0]
            menu.add_command(
                label=f"Import {disp} as populations",
                command=lambda c=col: self._import_populations(c))
            menu.add_command(
                label=f"Annotate {disp}…",
                command=lambda c=col: self._annotate_populations(c))
            menu.add_separator()

    def _open_clusters_menu(self):
        """Popup at the pointer (the 'Pops' button): import / annotate any
        label column present as populations."""
        menu = tk.Menu(self, tearoff=0)
        self._fill_populations_menu(menu)
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def _import_populations(self, col):
        """Import one label column as selectable populations. 'cluster' uses
        the legacy cluster-gate path (kept for session back-compat); every
        other column becomes 'category' gates via the generic importer."""
        if col == 'cluster':
            self._import_clusters()
        else:
            self._import_label_populations(col)

    def _import_population_column(self, *, group_for, disp, items_for,
                                  is_member, member_key, make_gate, none_msg,
                                  noun):
        """Shared core for importing per-sample populations from a label column
        (category values, cluster ids, …). For every loaded sample it: collects
        the column's items via ``items_for(name)``; nests new populations under
        ONE collapsed 'group' container (like auto-clean); creates a gate per
        not-yet-present item via ``make_gate(name, item, idx, grp_id)`` (started
        disabled); and keeps the group label showing the count. Idempotent —
        ``is_member``/``member_key`` dedupe against existing gates.

        Factored out of the previously copy-pasted cluster / label importers."""
        self._checkpoint()
        total_new = 0
        n_samples = 0
        for name in self._sample_order:
            items = items_for(name)
            if not items:
                continue
            n_samples += 1
            gates = self._sample_gates.setdefault(name, {})
            order = self._sample_gate_order.setdefault(name, [])
            existing = {member_key(g) for g in gates.values() if is_member(g)}
            grp_id = next((gid for gid, g in gates.items()
                           if g.get('kind') == 'group'
                           and g.get('group_for') == group_for), None)
            if grp_id is None and any(it not in existing for it in items):
                grp_id = self._next_gate_id_for(name)
                gates[grp_id] = {
                    'kind': 'group', 'group_for': group_for, 'name': disp,
                    'parent_id': None, 'open': False,   # collapsed by default
                    'color': '#808080', 'enabled': True,
                }
                order.append(grp_id)
            for i, it in enumerate(items):
                if it in existing:
                    continue
                gid = self._next_gate_id_for(name)
                g = make_gate(name, it, i, grp_id)
                g.setdefault('parent_id', grp_id)
                g.setdefault('enabled', False)
                gates[gid] = g
                order.append(gid)
                total_new += 1
            if grp_id is not None and grp_id in gates:
                n = sum(1 for g in gates.values() if is_member(g))
                gates[grp_id]['name'] = f'{disp} ({n})'
        if n_samples == 0:
            self.status_var.set(none_msg)
            return
        self._refresh_gate_list()
        self.status_var.set(
            f"Imported {total_new} new {noun} across {n_samples} sample(s). "
            "Toggle them in the tree.")

    def _import_label_populations(self, col):
        """Create a 'category' population per distinct value of ``col`` across
        every loaded sample that carries it. Idempotent; start disabled."""
        from .pipeline import GATE_PALETTE
        disp = self.LABEL_COLUMNS.get(col, (col, None))[0]

        def make(name, v, i, grp_id):
            return {'kind': 'category', 'channel': col, 'value': v,
                    'name': f'{col} {v}',
                    'color': GATE_PALETTE[i % len(GATE_PALETTE)]}

        self._import_population_column(
            group_for=col, disp=disp,
            items_for=lambda name: self._sample_label_values(name, col),
            is_member=lambda g: (g.get('kind') == 'category'
                                 and g.get('channel') == col),
            member_key=lambda g: g.get('value'),
            make_gate=make,
            none_msg=f"No samples carry a '{col}' column.",
            noun=f"{disp} population(s)")

    def _import_clusters(self):
        """Create a cluster-gate per clustering label for every loaded sample
        that carries a 'cluster' column. Idempotent; start disabled."""
        from .pipeline import GATE_PALETTE
        disp = self.LABEL_COLUMNS.get('cluster', ('clusters', None))[0]

        def make(name, cid, i, grp_id):
            return {'kind': 'cluster', 'channel': 'cluster', 'cluster_id': cid,
                    'name': self._cluster_display_name(name, cid),
                    'color': GATE_PALETTE[cid % len(GATE_PALETTE)]}

        self._import_population_column(
            group_for='cluster', disp=disp,
            items_for=lambda name: self._sample_cluster_ids(name),
            is_member=lambda g: g.get('kind') == 'cluster',
            member_key=lambda g: g.get('cluster_id'),
            make_gate=make,
            none_msg=("No clustered samples loaded. Run the pipeline with "
                      "clustering, or load a session that has a 'cluster' "
                      "column."),
            noun="cluster population(s)")

    def _run_cell_cycle(self, dna_channel, all_samples, k=1.5, singlet_tol=0.25):
        targets = (list(self._sample_order) if all_samples
                   else [self._active_sample])
        done = []
        for name in targets:
            s = self._samples.get(name)
            if s is None:
                continue
            try:
                s.cell_cycle(dna_channel=dna_channel, k=k, singlet_tol=singlet_tol)
            except Exception as exc:
                self.status_var.set(f"Cell cycle failed for {name}: {exc}")
                continue
            res = getattr(s, 'cell_cycle_result', None)
            if res and res.get('ok'):
                self._import_cell_cycle(name)
                done.append(name)
        self._refresh_gate_list()
        if not done:
            self.status_var.set(
                "Cell cycle: no usable DNA peaks found "
                f"on '{dna_channel}'.")
            return
        self.status_var.set(
            f"Cell cycle done for {len(done)} sample(s) on '{dna_channel}'. "
            "Phases added as populations; toggle them in the tree.")
        active = self._active_sample
        if active in done:
            try:
                from .ui_cell_cycle import CellCycleWindow
                CellCycleWindow(self, active)
            except Exception as exc:
                self.status_var.set(f"Cell-cycle plot failed: {exc}")

    def _import_cell_cycle(self, name):
        """Create a category population per cell-cycle phase present in
        `name`'s data. Idempotent (re-running only adds new phases).
        Populations start disabled, like imported clusters."""
        from .pipeline import CELL_CYCLE_PHASES, GATE_PALETTE
        s = self._samples.get(name)
        if s is None or 'cell_cycle' not in s.data.columns:
            return
        self._checkpoint()
        present = set(s.data['cell_cycle'].unique())
        phases = [p for p in CELL_CYCLE_PHASES if p in present]
        gates = self._sample_gates.setdefault(name, {})
        order = self._sample_gate_order.setdefault(name, [])
        existing = {g.get('value') for g in gates.values()
                    if g.get('kind') == 'category'
                    and g.get('channel') == 'cell_cycle'}
        # Nest the phase populations under ONE collapsed 'group' container
        # (like clusters / auto-clean), created on first import.
        disp = self.LABEL_COLUMNS.get('cell_cycle', ('cell-cycle phases', None))[0]
        grp_id = next((gid for gid, g in gates.items()
                       if g.get('kind') == 'group'
                       and g.get('group_for') == 'cell_cycle'), None)
        if grp_id is None and any(ph not in existing for ph in phases):
            grp_id = self._next_gate_id_for(name)
            gates[grp_id] = {
                'kind': 'group', 'group_for': 'cell_cycle', 'name': disp,
                'parent_id': None, 'open': False,
                'color': '#808080', 'enabled': True,
            }
            order.append(grp_id)
        for i, ph in enumerate(phases):
            if ph in existing:
                continue
            gid = self._next_gate_id_for(name)
            gates[gid] = {
                'kind': 'category',
                'channel': 'cell_cycle',
                'value': ph,
                'name': ph,
                'parent_id': grp_id,
                'color': self.PHASE_COLORS.get(
                    ph, GATE_PALETTE[i % len(GATE_PALETTE)]),
                'enabled': False,
            }
            order.append(gid)
        if grp_id is not None and grp_id in gates:
            n_ph = sum(1 for g in gates.values()
                       if g.get('kind') == 'category'
                       and g.get('channel') == 'cell_cycle')
            gates[grp_id]['name'] = f'{disp} ({n_ph})'

    def _export_populations_fcs(self):
        """Write each gated population of the active sample to its own FCS."""
        import numpy as np
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        gates = self._gates
        if not gates:
            messagebox.showinfo("Export populations",
                                "This sample has no gates to export.",
                                parent=self)
            return
        out = filedialog.askdirectory(
            title="Export gated populations (FCS) to…")
        if not out:
            return
        from .pipeline import gate_to_mask
        df = self._samples[name].data

        def _cumulative_mask(gid):
            mask = np.ones(len(df), dtype=bool)
            cur, seen = gid, set()
            while cur is not None and cur in gates and cur not in seen:
                seen.add(cur)
                g = gates[cur]
                try:
                    mask &= np.asarray(gate_to_mask(g, df), dtype=bool)
                except Exception:
                    pass
                cur = g.get('parent_id')
            return mask

        pops = {}
        for gid, g in gates.items():
            if g.get('kind') == 'autoclean':
                continue
            label = g.get('name') or f"{g.get('kind', 'gate')}_{gid}"
            sub = df[_cumulative_mask(gid)]
            if len(sub):
                pops[label] = sub
        if not pops:
            self.status_var.set("No non-empty populations to export.")
            return
        from .fcs_export import export_populations
        labels = getattr(self._samples[name], 'channel_labels', None)
        try:
            paths = export_populations(pops, out, channel_labels=labels)
        except Exception as exc:
            messagebox.showerror("Export failed",
                                 f"{type(exc).__name__}: {exc}", parent=self)
            return
        self.status_var.set(
            f"Exported {len(paths)} population FCS file(s) → {out}")

    def _export_population_fcs(self, name, gid):
        """Write the events inside a population (the gate's cumulative mask) to
        a standalone .fcs, re-importable in FlowJo / FCS Express. Exports the
        sample's RAW detector values when they align with the gated rows (so
        the file isn't in transformed coordinates), else the processed data."""
        from .pipeline import cumulative_gate_mask, write_fcs
        s = self._samples.get(name)
        gates = self._sample_gates.get(name, {})
        if s is None or gid not in gates:
            self.status_var.set("Select a gated population to export.")
            return
        mask = np.asarray(cumulative_gate_mask(gates, gid, s.data), dtype=bool)
        n = int(mask.sum())
        if n == 0:
            self.status_var.set("That population is empty — nothing to export.")
            return
        # Prefer raw detector values (untransformed) when row-aligned with data.
        raw = getattr(s, 'raw', None)
        if raw is not None and len(raw) == len(s.data) and not raw.empty:
            export_df = raw.iloc[mask]
            labels = getattr(s, 'channel_labels', {}) or {}
        else:
            export_df = s.data[mask]
            labels = dict(self._channel_labels)
            labels.update(getattr(s, 'channel_labels', {}) or {})

        pop = self._population_path(gates, gid)
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', f"{name}_{pop}").strip('_')
        path = filedialog.asksaveasfilename(
            parent=self, title="Export population as FCS",
            defaultextension='.fcs', initialfile=f"{safe}.fcs",
            filetypes=[('FCS', '*.fcs'), ('All files', '*.*')])
        if not path:
            return
        try:
            written = write_fcs(path, export_df, channel_labels=labels)
        except Exception as exc:
            messagebox.showerror(
                "Export FCS", f"Could not write FCS:\n{type(exc).__name__}: "
                f"{exc}", parent=self)
            return
        self._audit('population.export_fcs', sample=name, population=pop,
                    n_events=written, path=path)
        self.status_var.set(
            f"Exported {written:,} events of '{pop}' → "
            f"{os.path.basename(path)}")
