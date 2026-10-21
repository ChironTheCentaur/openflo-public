"""Combination gates (every Boolean combination of chosen gates) and the
polyfunctionality window -- editor mixin.

The gate logic is in gate_combos (Tk-free); this is the dialog, the
per-sample install and the hand-off to ui_polyfunction.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .editor_base import EditorMixin


class CombinationsMixin(EditorMixin):
    """Gates > Combination gates…: 2^N boolean populations in one step."""

    def _combinable_gate_ids(self, name):
        """The active sample's gates a combination may use: geometric and
        label gates (not booleans or containers), in tree order."""
        gates = self._sample_gates.get(name, {})
        return [g for g in self._sample_gate_order.get(name, [])
                if g in gates and gates[g].get('kind') not in
                ('boolean', 'group', 'autoclean', 'flowjo_ellipse')]

    def _open_combination_dialog(self):
        """Choose 2 to 6 gates; every combination of them is made as a
        boolean population under their common parent, on the active sample
        and (by default) every other sample that has the same gates."""
        from .gate_combos import MAX_COMBINATION_GATES, gate_label
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return None
        gates = self._sample_gates.get(name, {})
        order = self._combinable_gate_ids(name)
        if len(order) < 2:
            self.status_var.set("Need at least two gates to combine.")
            return None
        transforms = self._display_transforms(name)

        dlg = tk.Toplevel(self)
        dlg.title(f"Combination gates — {name}")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.geometry("420x480")
        ttk.Label(dlg, wraplength=390, justify='left', text=(
            f"Choose 2 to {MAX_COMBINATION_GATES} positive gates. Every "
            "combination of them (all positive to all negative) is made as a "
            "population under the gates' common parent.")).pack(
            side='top', fill='x', padx=10, pady=(10, 4))
        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=4)
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))
        from .gating import population_path
        cb_vars = {}
        for gid in order:
            var = tk.BooleanVar(value=False)
            cb_vars[gid] = var
            ttk.Checkbutton(inner, variable=var, text=population_path(
                gates, gid, transforms=transforms)).pack(side='top',
                                                        anchor='w')
        info = tk.StringVar(value='')
        ttk.Label(dlg, textvariable=info, wraplength=390,
                  justify='left').pack(side='top', fill='x', padx=10)
        all_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, variable=all_var, text=(
            "Also make them on every other sample with these gates")).pack(
            side='top', anchor='w', padx=10, pady=(4, 0))

        def chosen():
            return [g for g in order if cb_vars[g].get()]

        def update_info(*_):
            from .gate_combos import check_combinable, common_parent
            ids = chosen()
            why = check_combinable(gates, ids) if ids else None
            if not ids:
                info.set('')
            elif why:
                info.set(why)
            else:
                pid = common_parent(gates, ids)
                where = (population_path(gates, pid, transforms=transforms)
                         if pid else 'all events')
                info.set(f"{2 ** len(ids)} combinations of "
                         f"{', '.join(gate_label(gates, g) for g in ids)}, "
                         f"as % of {where}.")
        for v in cb_vars.values():
            v.trace_add('write', update_info)

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=(4, 10))

        def do_create():
            ids = chosen()
            from .gate_combos import check_combinable
            why = check_combinable(gates, ids)
            if why:
                info.set(why)
                return
            dlg.destroy()
            self._make_combinations(ids, every_sample=bool(all_var.get()))

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Create", command=do_create).pack(
            side='right', padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: dlg.destroy())
        return dlg

    def _make_combinations(self, ids, every_sample=True, show=True):
        """Make the combinations of the active sample's gates ``ids`` and,
        with ``every_sample``, of the same gates (matched by population, as
        Compare-all matches them) in each other loaded sample. One undo step.
        Returns ``{sample: build_combinations result}``."""
        from .gate_combos import build_combinations, gate_label
        from .gating import population_keys
        name = self._active_sample
        gates = self._sample_gates.get(name, {})
        labels = [gate_label(gates, g) for g in ids]
        self._checkpoint()
        built, skipped = {}, []
        targets = [name] + ([n for n in self._sample_order
                             if n in self._samples and n != name]
                            if every_sample else [])
        keys = population_keys(gates, self._channel_labels)
        want = [keys[g] for g in ids]
        for n in targets:
            sg = self._sample_gates.setdefault(n, {})
            if n == name:
                tids = list(ids)
            else:
                rev = {k: g for g, k in population_keys(
                    sg, self._channel_labels).items()}
                tids = [rev.get(k) for k in want]
                if any(t is None for t in tids):
                    skipped.append(n)
                    continue

            def add(gate, _n=n):
                return self._add_gate_to_sample(_n, gate)
            try:
                built[n] = build_combinations(sg, tids, add, labels=labels)
            except ValueError:
                skipped.append(n)
        self._refresh_gate_list()
        self._schedule_replot(0)
        added = sum(b['added'] for b in built.values())
        self._audit('gate.combinations', sample=name, gates=labels,
                    samples=list(built), added=added)
        msg = (f"{2 ** len(ids)} combinations of {', '.join(labels)} on "
               f"{len(built)} sample(s); {added} gate(s) added.")
        if skipped:
            msg += (f"  {len(skipped)} sample(s) lack these gates: "
                    f"{', '.join(self._short_sample(s, 14) for s in skipped[:3])}"
                    f"{'…' if len(skipped) > 3 else ''}.")
        self.status_var.set(msg)
        if show and built:
            self._show_polyfunctionality(built, labels)
        return built

    def _add_gate_to_sample(self, name, gate):
        """Store ``gate`` in sample ``name`` with a fresh id there (the
        active sample goes through _add_gate, which also keeps its id order).
        Returns the id."""
        if name == self._active_sample:
            return self._add_gate(gate, parent_id=gate.get('parent_id'),
                                  audit=False, replace_1d=False)
        gate.setdefault('color', self._next_color())
        gate.setdefault('enabled', True)
        gid = self._next_gate_id_for(name)
        while gid in self._sample_gates[name]:
            gid = self._next_gate_id_for(name)
        gate['id'] = gid
        self._sample_gates[name][gid] = gate
        self._sample_gate_order.setdefault(name, []).append(gid)
        return gid

    def _combination_rows(self, built, labels):
        """The frequency rows (gate_combos.frequency_rows) for ``built``,
        grouped by trial."""
        from .gate_combos import combination_counts, frequency_rows
        per = {}
        for n, b in built.items():
            df = getattr(self._samples.get(n), 'data', None)
            if df is None:
                continue
            gates = self._sample_gates.get(n, {})
            n_par, counts = combination_counts(gates, df, b['combos'],
                                               b['parent'])
            per[n] = (n_par, b['combos'], counts)
        groups = {n: str(self._trial_for(n) or '') for n in per}
        if len(set(groups.values())) <= 1:
            groups = {n: '' for n in per}
        return frequency_rows(per, labels, groups)

    def _show_polyfunctionality(self, built, labels):
        from .gating import population_path
        from .ui_polyfunction import PolyfunctionWindow
        rows = self._combination_rows(built, labels)
        first = next(iter(built))
        pid = built[first]['parent']
        where = (population_path(self._sample_gates.get(first, {}), pid,
                                 transforms=self._display_transforms(first))
                 if pid else 'all events')
        return PolyfunctionWindow(self, rows, labels, where)
