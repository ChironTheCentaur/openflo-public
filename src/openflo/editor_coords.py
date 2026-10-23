"""Edit coordinates… -- type a gate's bounds or vertices in axis units.

The conversions are in gate_coords (Tk-free); this is the dialog and the
install, which is one undo step.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .editor_base import EditorMixin


class CoordsMixin(EditorMixin):
    """A gate's right-click > Edit coordinates…"""

    def _edit_gate_coordinates(self, name=None, gid=None):
        """Open the coordinates dialog for gate ``gid`` of sample ``name``
        (default: the selected gate of the active sample)."""
        from . import gate_coords as gc
        name = name or self._active_sample
        gid = gid or self._selected_gate_id()
        gate = self._sample_gates.get(name, {}).get(gid)
        if gate is None:
            self.status_var.set("Select a gate first.")
            return None
        if not gc.editable(gate):
            self.status_var.set("This gate has no coordinates to type "
                                "(rectangle, quadrant, polygon, ellipse, "
                                "threshold or interval gates do).")
            return None
        transforms = self._display_transforms(name)
        rows = gc.fields(gate, transforms)
        from .pipeline import describe_gate
        dlg = tk.Toplevel(self)
        dlg.title(f"Edit coordinates — {describe_gate(gate, transforms=transforms)}")
        dlg.transient(self)  # type: ignore[arg-type]
        body = ttk.Frame(dlg, padding=10)
        body.pack(fill='both', expand=True)
        entries = {}
        for r, (key, label, text) in enumerate(rows):
            ttk.Label(body, text=label).grid(row=r, column=0, sticky='nw',
                                             padx=(0, 8), pady=2)
            if key == 'vertices':
                w = tk.Text(body, width=34, height=min(14, text.count('\n') + 2))
                w.insert('1.0', text)
                entries[key] = w
            else:
                var = tk.StringVar(value=text)
                w = ttk.Entry(body, textvariable=var, width=18)
                entries[key] = var
            w.grid(row=r, column=1, sticky='we', pady=2)
        body.columnconfigure(1, weight=1)
        note = ["Values are the axis units (intensity); a blank bound is "
                "open."]
        unknown = gc.unknown_channels(gate, transforms)
        if unknown:
            note.append(f"{', '.join(unknown)}: scale unknown, so the values "
                        "are the stored ones.")
        ttk.Label(body, text=' '.join(note), wraplength=320,
                  justify='left').grid(row=len(rows), column=0, columnspan=2,
                                       sticky='w', pady=(8, 0))
        all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(body, variable=all_var, text=(
            "Set the same gate in every other sample too")).grid(
            row=len(rows) + 1, column=0, columnspan=2, sticky='w', pady=(4, 0))
        err = tk.StringVar(value='')
        ttk.Label(body, textvariable=err, style='Error.TLabel', wraplength=320,
                  justify='left').grid(row=len(rows) + 2, column=0,
                                       columnspan=2, sticky='w')

        def typed():
            out = {}
            for key, w in entries.items():
                out[key] = (w.get('1.0', 'end-1c') if isinstance(w, tk.Text)
                            else w.get())
            return out

        def do_ok():
            try:
                n = self._apply_gate_coordinates(name, gid, typed(),
                                                 every_sample=all_var.get())
            except ValueError as exc:
                err.set(str(exc))
                return
            dlg.destroy()
            self.status_var.set(f"Gate coordinates set on {n} sample(s).")

        btns = ttk.Frame(body)
        btns.grid(row=len(rows) + 3, column=0, columnspan=2, sticky='e',
                  pady=(8, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="OK", command=do_ok).pack(side='right',
                                                        padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: dlg.destroy())
        dlg.bind('<Return>', lambda e: (None if isinstance(e.widget, tk.Text)
                                        else do_ok()))
        return dlg

    def _apply_gate_coordinates(self, name, gid, typed, every_sample=False):
        """Apply typed coordinates to gate ``gid`` of ``name`` and, with
        ``every_sample``, to the same population in every other sample (each
        converted with that sample's own transforms). Validates every sample
        before changing any; one undo step. Returns how many samples changed.
        ValueError says what is wrong with the input."""
        from . import gate_coords as gc
        from .gating import population_keys
        gates = self._sample_gates[name]
        plan = [(name, gid)]
        if every_sample:
            key = population_keys(gates, self._channel_labels)[gid]
            for n in self._sample_order:
                if n == name or n not in self._sample_gates:
                    continue
                for og, k in population_keys(self._sample_gates[n],
                                             self._channel_labels).items():
                    if k == key and gc.editable(self._sample_gates[n][og]):
                        plan.append((n, og))
                        break
        new = []
        for n, g in plan:
            new.append((n, g, gc.apply(self._sample_gates[n][g], typed,
                                       self._display_transforms(n))))
        self._checkpoint()
        for n, g, ng in new:
            old = self._sample_gates[n][g]
            if old.get('quad_set'):
                gc.set_quadrant_origin(self._sample_gates[n], old['quad_set'],
                                       ng['quad_origin_x'], ng['quad_origin_y'])
            else:
                old.clear()
                old.update(ng)
        from .pipeline import describe_gate
        self._audit('gate.coordinates', sample=name, id=gid,
                    gate=describe_gate(gates[gid]),
                    samples=[n for n, _g, _ng in new])
        self._refresh_gate_list()
        self._schedule_replot(0)
        return len(new)
