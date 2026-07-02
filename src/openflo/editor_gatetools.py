"""Interactive gate drawing, boolean/copy dialogs, and gate redraw.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .editor_base import EditorMixin
from .ui_logic import has_real_gates


class GateToolsMixin(EditorMixin):
    """Rectangle / ellipse / polygon / lasso gate tools, boolean & copy-gate dialogs, singlet / FMO gates, and the gate-overlay redraw."""

    def _next_gate_id_for(self, name):
        """Allocate a fresh gate id for `name`'s gate set, keeping the
        per-sample sequence (and the active-sample shortcut) in sync."""
        seq = self._sample_gate_seq.get(name, 0) + 1
        self._sample_gate_seq[name] = seq
        if name == self._active_sample:
            self._gate_id_seq = seq
        return f'g{seq}'

    def _open_copy_gates_dialog(self):
        """Pop a modal dialog: pick which loaded samples should receive a
        copy of the active sample's gate tree. Includes Select all /
        Deselect all helpers. Copies APPEND (don't overwrite) so the
        target sample's existing gates are preserved."""
        if self._active_sample is None or not self._gates:
            self.status_var.set("Active sample has no gates to copy.")
            return
        others = [n for n in self._sample_order
                  if n in self._samples and n != self._active_sample]
        if not others:
            self.status_var.set("No other samples loaded to copy into.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Copy gates to samples")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.geometry("420x440")
        dlg.minsize(360, 280)

        ttk.Label(dlg,
                  text=f"Copy {len(self._gates)} gate(s) from "
                       f"'{self._active_sample}' to:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 6))

        # Scrollable list of checkboxes (one per other sample).
        canvas_holder = ttk.Frame(dlg)
        canvas_holder.pack(side='top', fill='both', expand=True,
                           padx=10, pady=(0, 6))
        cv = tk.Canvas(canvas_holder, highlightthickness=0)
        sb = ttk.Scrollbar(canvas_holder, orient='vertical',
                           command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor='nw')

        def _on_inner_configure(_):
            cv.configure(scrollregion=cv.bbox('all'))
        inner.bind('<Configure>', _on_inner_configure)

        cb_vars = {}
        for name in others:
            var = tk.BooleanVar(value=False)
            cb_vars[name] = var
            ttk.Checkbutton(inner, text=name, variable=var).pack(
                side='top', anchor='w', padx=2, pady=1)

        freeze_var = tk.BooleanVar(value=False)
        has_ac = any(g.get('kind') == 'autoclean' for g in self._gates.values())

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        # Freeze toggle sits just above the buttons. Only meaningful when the
        # source has an auto-clean gate; disabled otherwise.
        ttk.Checkbutton(
            dlg, variable=freeze_var,
            state=('normal' if has_ac else 'disabled'),
            text=("Freeze auto-clean cuts to "
                  f"'{self._active_sample}'s values (else recompute "
                  "per sample)")).pack(side='bottom', anchor='w',
                                       padx=10, pady=(0, 2))

        def select_all():
            for v in cb_vars.values():
                v.set(True)

        def select_none():
            for v in cb_vars.values():
                v.set(False)

        ttk.Button(btns, text="Select all",
                   command=select_all).pack(side='left')
        ttk.Button(btns, text="Deselect all",
                   command=select_none).pack(side='left', padx=(4, 0))

        def do_copy():
            targets = [n for n, v in cb_vars.items() if v.get()]
            frz = bool(freeze_var.get())
            dlg.destroy()
            if targets:
                self._copy_gates_to(targets, append=True, freeze=frz)

        ttk.Button(btns, text="Cancel",
                   command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Copy",
                   command=do_copy).pack(side='right', padx=(0, 4))

        dlg.bind('<Escape>', lambda *_: dlg.destroy())

    def _copy_gates_to(self, target_names, append=True, freeze=False):
        """Copy the active sample's gates into each target sample.

        When `append=True` (default), target's existing gates are kept
        and the copied tree is added on top (each copied gate gets a
        fresh id; parent_id pointers within the copied subset are
        remapped). When False, the target's gates are replaced.

        When `freeze=True`, auto-clean gates are FROZEN to the source
        sample's computed cuts before copying (fixed `min_fsc` / dye
        `max_signal`), so targets apply identical thresholds instead of
        recomputing — see `pipeline.freeze_autoclean_gate`.
        """
        import copy as _copy
        src = self._active_sample
        if src is None or src not in self._sample_gates:
            return
        src_gates = self._sample_gates[src]
        src_order = list(self._sample_gate_order.get(src, []))
        if not src_order:
            return
        # Pre-freeze the source's auto-clean gates once (same frozen values for
        # every target); other gates copy verbatim.
        frozen = {}
        n_frozen = 0
        if freeze:
            from .pipeline import freeze_autoclean_gate
            sdata = getattr(self._samples.get(src), 'data', None)
            labels = getattr(self._samples.get(src), 'channel_labels', {}) or {}
            if sdata is not None:
                for gid, g in src_gates.items():
                    if g.get('kind') == 'autoclean':
                        frozen[gid] = freeze_autoclean_gate(g, sdata, labels)
                        n_frozen += 1
        self._checkpoint()

        copied_to = 0
        for target in target_names:
            if target == src or target not in self._samples:
                continue
            if not append:
                self._sample_gates[target]     = {}
                self._sample_gate_order[target] = []
                self._sample_gate_seq[target]   = 0
            # Ensure containers exist (target may have been gate-less).
            self._sample_gates.setdefault(target, {})
            self._sample_gate_order.setdefault(target, [])
            self._sample_gate_seq.setdefault(target, 0)

            # If the target IS the currently-active sample, mutate the
            # already-bound containers (so self._gates etc. stay in sync).
            if target == self._active_sample:
                target_gates = self._gates
                target_order = self._gate_id_order
            else:
                target_gates = self._sample_gates[target]
                target_order = self._sample_gate_order[target]

            old_to_new = {}
            for old_id in src_order:
                g = _copy.deepcopy(frozen.get(old_id, src_gates[old_id]))
                self._sample_gate_seq[target] += 1
                new_id = f'g{self._sample_gate_seq[target]}'
                if target == self._active_sample:
                    self._gate_id_seq = self._sample_gate_seq[target]
                pid = g.get('parent_id')
                g['parent_id'] = old_to_new.get(pid) if pid else None
                target_gates[new_id] = g
                target_order.append(new_id)
                old_to_new[old_id] = new_id
            copied_to += 1

        froze_msg = (f" · froze {n_frozen} auto-clean cut(s) to '{src}'"
                     if freeze and n_frozen else "")
        self.status_var.set(
            f"Copied {len(src_order)} gate(s) from '{src}' to "
            f"{copied_to} sample(s) (appended; existing gates preserved)"
            f"{froze_msg}.")
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _next_gate_id(self):
        self._gate_id_seq += 1
        # Counter is per-sample; the int rebind doesn't propagate via the
        # shared-reference trick we use for dicts, so mirror it explicitly.
        if self._active_sample is not None:
            self._sample_gate_seq[self._active_sample] = self._gate_id_seq
        return f'g{self._gate_id_seq}'

    def _selected_gate_id(self):
        """Gate_id of the currently-selected tree row IF it belongs to the
        active sample; otherwise None. Used by _add_gate to decide what
        parent a new shape should attach to."""
        if not hasattr(self, 'gate_tv'):
            return None
        sel = self.gate_tv.selection()
        if not sel:
            return None
        parsed = self._parse_iid(sel[0])
        if parsed is None or parsed[0] != 'gate':
            return None
        sample_name, gid = parsed[1], parsed[2]
        if sample_name != self._active_sample:
            return None
        return gid if gid in self._gates else None

    def _has_real_gates(self):
        """True if the active sample carries any positive gate. Auto-clean
        gates don't count — they're negative selections (events to drop), not
        populations to highlight or filter to."""
        return has_real_gates(self._sample_gates.get(self._active_sample, {}))

    def _gates_topological_for(self, gates_dict):
        """Generic topo-sort over an arbitrary per-sample gates dict."""
        seen = set()
        out = []

        def visit(gid):
            if gid in seen or gid not in gates_dict:
                return
            seen.add(gid)
            parent = gates_dict[gid].get('parent_id')
            if parent and parent in gates_dict and parent not in seen:
                visit(parent)
            out.append(gid)

        for gid in gates_dict:
            visit(gid)
        return out

    def _pick_gate_color(self, name, gid):
        """Open a colour chooser for one gate/population and apply the chosen
        colour (used by the highlight overlay, backgate, and the tree swatch)."""
        g = self._sample_gates.get(name, {}).get(gid)
        if g is None:
            return
        from tkinter import colorchooser
        _rgb, hexv = colorchooser.askcolor(
            color=g.get('color') or '#e6194b', parent=self,
            title='Gate colour')
        if not hexv:
            return
        self._checkpoint()
        g['color'] = hexv
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _insert_polygon_vertex(self, gid, idx, x, y):
        """Insert a vertex (x, y) at position ``idx`` in polygon ``gid``."""
        g = self._gates.get(gid)
        if g is None or g.get('kind') != 'polygon':
            return
        if x is None or y is None:
            return
        verts = list(g.get('vertices') or [])
        idx = max(0, min(int(idx), len(verts)))
        verts.insert(idx, [float(x), float(y)])
        g['vertices'] = verts
        self._redraw_only_gates()
        self._refresh_gate_list()

    def _polygon_under_point(self, x, y):
        """Return gate_id of the polygon containing (x, y), or — if none
        contain it — the nearest polygon by vertex distance. None if no
        polygon gates exist in this view's channels."""
        if x is None or y is None:
            return None
        from matplotlib.path import Path as _MplPath
        x_ch = self._resolve_channel(self.x_combo.get())
        y_ch = self._resolve_channel(self.y_combo.get())
        candidates = []
        for gid, g in self._gates.items():
            if g.get('kind') != 'polygon':
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            verts = g.get('vertices') or []
            if len(verts) < 3:
                continue
            try:
                arr = np.asarray(verts, dtype=float)
                if _MplPath(arr).contains_point((x, y)):
                    return gid
                # distance to nearest vertex (axis-fraction units)
                dx = arr[:, 0] - x
                dy = arr[:, 1] - y
                d = float(np.sqrt((dx * dx + dy * dy).min()))
                candidates.append((d, gid))
            except Exception:
                continue
        if candidates:
            candidates.sort()
            return candidates[0][1]
        return None

    def _move_gate_to_sample(self, src_sample, src_gid,
                             dst_sample, new_parent_in_dst):
        """Move a gate AND every descendant from src_sample to dst_sample.

        Fresh gate ids are assigned in the destination; parent_id
        pointers within the moved subtree are remapped to those new ids.
        The dragged gate's root becomes a child of `new_parent_in_dst`
        (None = root of the destination sample).

        Returns the new gate id of the dragged gate's root in dst, or
        None on failure.
        """
        import copy as _copy
        src_gates = self._sample_gates.get(src_sample, {})
        src_order = self._sample_gate_order.get(src_sample, [])
        if src_gid not in src_gates:
            return None

        # Collect the subtree rooted at src_gid (BFS, preserves any
        # ordering invariants children had relative to each other).
        subtree_ids = []
        queue       = [src_gid]
        seen        = set()
        while queue:
            cur = queue.pop(0)
            if cur in seen or cur not in src_gates:
                continue
            seen.add(cur)
            subtree_ids.append(cur)
            for other_gid, og in src_gates.items():
                if og.get('parent_id') == cur:
                    queue.append(other_gid)

        # Ensure destination containers exist.
        self._sample_gates.setdefault(dst_sample, {})
        self._sample_gate_order.setdefault(dst_sample, [])
        self._sample_gate_seq.setdefault(dst_sample, 0)
        dst_gates = self._sample_gates[dst_sample]
        dst_order = self._sample_gate_order[dst_sample]

        # Assign fresh ids in dst for every subtree member.
        old_to_new = {}
        for old_id in subtree_ids:
            self._sample_gate_seq[dst_sample] += 1
            old_to_new[old_id] = f'g{self._sample_gate_seq[dst_sample]}'
        if dst_sample == self._active_sample:
            self._gate_id_seq = self._sample_gate_seq[dst_sample]

        # Transplant each gate (deep-copied to be safe), remapping parents.
        for old_id in subtree_ids:
            g = _copy.deepcopy(src_gates[old_id])
            new_id = old_to_new[old_id]
            if old_id == src_gid:
                g['parent_id'] = new_parent_in_dst
            else:
                old_parent = g.get('parent_id')
                g['parent_id'] = old_to_new.get(old_parent, new_parent_in_dst)
            dst_gates[new_id] = g
            dst_order.append(new_id)

        # Remove the subtree from src.
        for old_id in subtree_ids:
            src_gates.pop(old_id, None)
            if old_id in src_order:
                src_order.remove(old_id)

        return old_to_new[src_gid]

    def _collect_gate_subtree(self, sample_name, root_gid):
        """Deep-copy the subtree rooted at `root_gid` from `sample_name`.
        Each returned dict carries a temporary `_clip_id` (the original
        gate_id, so paste can rewire `parent_id` references). BFS order
        so paste can iterate parents-before-children safely."""
        import copy as _copy
        sgates = self._sample_gates.get(sample_name, {})
        if root_gid not in sgates:
            return []
        collected = []
        queue = [root_gid]
        seen  = set()
        while queue:
            cur = queue.pop(0)
            if cur in seen or cur not in sgates:
                continue
            seen.add(cur)
            g = _copy.deepcopy(sgates[cur])
            g['_clip_id'] = cur
            collected.append(g)
            for other_gid, og in sgates.items():
                if og.get('parent_id') == cur:
                    queue.append(other_gid)
        return collected

    def _is_descendant_of(self, sample_name, ancestor_gid, candidate_gid):
        """True if `candidate_gid` is `ancestor_gid` itself or anywhere
        below it in `sample_name`'s tree. Cycle-safe."""
        sgates = self._sample_gates.get(sample_name, {})
        cur = candidate_gid
        seen = set()
        while cur is not None and cur not in seen:
            if cur == ancestor_gid:
                return True
            seen.add(cur)
            g = sgates.get(cur)
            if g is None:
                return False
            cur = g.get('parent_id')
        return False

    def _open_boolean_dialog(self):
        """Build a boolean population (AND / OR / NOT) from the active
        sample's existing gates. The result is a root 'boolean' gate whose
        operands are the chosen gates (evaluated as their cumulative
        masks)."""
        from .pipeline import describe_gate
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        gates = self._sample_gates.get(name, {})
        order = [g for g in self._sample_gate_order.get(name, [])
                 if g in gates and gates[g].get('kind') != 'boolean']
        if len(order) < 1:
            self.status_var.set("Need at least one gate to combine.")
            return

        dlg = tk.Toplevel(self)
        dlg.title(f"Boolean gate — {name}")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.geometry("360x420")

        ttk.Label(dlg, text="Combine these gates:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 4))

        op_var = tk.StringVar(value='and')
        oprow = ttk.Frame(dlg)
        oprow.pack(side='top', fill='x', padx=10)
        for lbl, val in [('AND', 'and'), ('OR', 'or'), ('NOT', 'not')]:
            ttk.Radiobutton(oprow, text=lbl, value=val,
                            variable=op_var).pack(side='left', padx=(0, 8))

        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=6)
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
        for gid in order:
            var = tk.BooleanVar(value=False)
            cb_vars[gid] = var
            ttk.Checkbutton(inner, text=describe_gate(gates[gid]),
                            variable=var).pack(side='top', anchor='w')

        namerow = ttk.Frame(dlg)
        namerow.pack(side='top', fill='x', padx=10, pady=(4, 0))
        ttk.Label(namerow, text="Name:").pack(side='left')
        name_var = tk.StringVar(value='')
        ttk.Entry(namerow, textvariable=name_var).pack(
            side='left', fill='x', expand=True)

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_create():
            operands = [g for g, v in cb_vars.items() if v.get()]
            if not operands:
                self.status_var.set("Select at least one gate.")
                return
            op = op_var.get()
            nm = name_var.get().strip() or (
                f"{op.upper()} ({len(operands)})")
            dlg.destroy()
            self._add_gate({'kind': 'boolean', 'op': op,
                            'operands': operands, 'name': nm,
                            'parent_id': None})
            self.status_var.set(f"Boolean gate '{nm}' created.")
            self._schedule_replot(0)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Create", command=do_create).pack(
            side='right', padx=(0, 6))

    def _ask_clear_all(self, total, n_samples, n_autoclean):
        """Confirm Clear-all. Returns True (also clear auto-clean), False (keep
        auto-clean), or None (cancelled). With no auto-clean gates present it's
        a plain yes/no (returns False/None)."""
        if n_autoclean == 0:
            ok = messagebox.askyesno(
                "Clear all gates",
                f"Remove all {total} gate(s) from {n_samples} sample(s)?\n"
                "Samples stay loaded. (Undoable.)",
                parent=self)
            return False if ok else None

        dlg = tk.Toplevel(self)
        dlg.title("Clear all gates")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.resizable(False, False)
        ttk.Label(
            dlg, justify='left', padding=(12, 12, 12, 6),
            text=(f"Remove gates from {n_samples} sample(s)?\n"
                  "Samples stay loaded. (Undoable.)")).pack(anchor='w')
        inc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            dlg, variable=inc_var, padding=(12, 0),
            text=(f"Also clear the {n_autoclean} auto-clean gate(s) "
                  "(off = keep cleaning)")).pack(anchor='w')
        result: dict[str, object] = {'val': None}

        def _ok():
            result['val'] = bool(inc_var.get())
            dlg.destroy()

        btns = ttk.Frame(dlg, padding=12)
        btns.pack(anchor='e')
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(
            side='right')
        ttk.Button(btns, text="Clear", command=_ok).pack(
            side='right', padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: dlg.destroy())
        dlg.bind('<Return>', lambda _e: _ok())
        try:
            dlg.grab_set()
            self.wait_window(dlg)
        except Exception:
            pass
        return result['val']

    def _activate_gate_tool(self, *_):
        """Wire up the matplotlib region selector matching the current
        tool. Called when the user switches tool, when _replot rebuilds
        the axes, or when the plot mode changes."""
        # Tear down any prior selector.
        if self._selector is not None:
            try:
                self._selector.set_active(False)
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None

        tool = self.gate_tool_var.get()
        mode = self.mode_var.get()
        # Update the gesture hint label next to the tool radios.
        if hasattr(self, 'tool_hint_var'):
            self.tool_hint_var.set(self._TOOL_HINTS.get(tool, ''))
        # Region tools don't make sense in histogram mode.
        # Edit mode also has no matplotlib Selector — its gestures are
        # handled by `_on_press` against the existing hit-test results.
        if tool in ('quadrant', 'edit') or mode == 'histogram':
            return
        # Need an X+Y channel for any region tool.
        if not (self._resolve_channel(self.x_combo.get())
                and self._resolve_channel(self.y_combo.get())):
            return

        from matplotlib.widgets import (
            EllipseSelector,
            LassoSelector,
            PolygonSelector,
            RectangleSelector,
        )
        props = dict(color='red', linestyle='--', linewidth=1.3, alpha=0.9)
        try:
            if tool == 'rectangle':
                self._selector = RectangleSelector(
                    self.ax, self._on_rect_select,
                    useblit=True,
                    minspanx=3, minspany=3, spancoords='pixels',
                    interactive=False,
                    props=props)
            elif tool == 'ellipse':
                self._selector = EllipseSelector(
                    self.ax, self._on_ellipse_select,
                    useblit=True,
                    minspanx=3, minspany=3, spancoords='pixels',
                    interactive=False,
                    props=props)
            elif tool == 'polygon':
                # matplotlib's runtime PolygonSelector passes onselect(verts)
                # (list of (x,y)) — its stubs incorrectly declare (x, y) pair.
                self._selector = PolygonSelector(
                    self.ax,
                    self._on_poly_select,  # type: ignore[arg-type]
                    useblit=True, props=props)
            elif tool == 'lasso':
                self._selector = LassoSelector(
                    self.ax,
                    self._on_lasso_select,  # type: ignore[arg-type]
                    useblit=True, props=props)
        except Exception as exc:
            # Older matplotlibs may not accept `props=`; fall back without it.
            print(f"[gate-tool] selector init failed ({exc}); retrying "
                  f"without props", flush=True)
            try:
                if tool == 'rectangle':
                    self._selector = RectangleSelector(
                        self.ax, self._on_rect_select,
                        useblit=True,
                        minspanx=3, minspany=3, spancoords='pixels',
                        interactive=False)
                elif tool == 'ellipse':
                    self._selector = EllipseSelector(
                        self.ax, self._on_ellipse_select,
                        useblit=True,
                        minspanx=3, minspany=3, spancoords='pixels',
                        interactive=False)
                elif tool == 'polygon':
                    self._selector = PolygonSelector(
                        self.ax,
                        self._on_poly_select,  # type: ignore[arg-type]
                        useblit=True)
                elif tool == 'lasso':
                    self._selector = LassoSelector(
                        self.ax,
                        self._on_lasso_select,  # type: ignore[arg-type]
                        useblit=True)
            except Exception as exc2:
                print(f"[gate-tool] selector init still failed: {exc2}",
                      flush=True)
                self._selector = None

    def _on_rect_select(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or self.mode_var.get() == 'histogram':
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        self._add_gate_multi({'kind': 'rect', 'x_channel': x, 'y_channel': y,
                              'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1})
        self._schedule_replot(0)

    def _on_ellipse_select(self, eclick, erelease):
        """EllipseSelector gives the press + release corners of the
        bounding box. Build an axis-aligned ellipsoid gate inscribed in
        that box: mean = box centre, and a diagonal covariance whose
        semi-axes equal the box half-extents at distance_sq = 1
        (so (x-µ)²/a² + (y-µ)²/b² ≤ 1 is exactly the inscribed ellipse).
        """
        if eclick.xdata is None or erelease.xdata is None:
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or self.mode_var.get() == 'histogram':
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        a = (x1 - x0) / 2.0    # semi-axis along x
        b = (y1 - y0) / 2.0    # semi-axis along y
        if a <= 0 or b <= 0:
            return
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self._add_gate_multi({
            'kind': 'ellipsoid', 'x_channel': x, 'y_channel': y,
            'mean': [cx, cy],
            'cov':  [[a * a, 0.0], [0.0, b * b]],
            'distance_sq': 1.0,
        })
        self._schedule_replot(0)

    def _on_poly_select(self, verts):
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or len(verts) < 3:
            return
        self._add_gate_multi({'kind': 'polygon', 'x_channel': x, 'y_channel': y,
                              'vertices': [[float(vx), float(vy)]
                                           for vx, vy in verts]})
        self._schedule_replot(0)

    def _on_lasso_select(self, verts):
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y or len(verts) < 3:
            return
        self._add_gate_multi({'kind': 'polygon', 'x_channel': x, 'y_channel': y,
                              'vertices': [[float(vx), float(vy)]
                                           for vx, vy in verts]})
        self._schedule_replot(0)

    def _redraw_only_gates(self):
        """Cheap refresh of the gate overlays without redoing the density
        plot. Used while dragging sliders."""
        # Remove existing artists.
        for line in list(self._vlines.values()) + list(self._hlines.values()):
            try:
                line.remove()
            except Exception:
                pass
        for patch in list(self._shape_artists.values()):
            try:
                patch.remove()
            except Exception:
                pass
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        self._draw_gates(x, y if self.mode_var.get() != 'histogram' else None)
        self.canvas.draw_idle()

    def _add_singlet_gate(self):
        """Add an FSC-A vs FSC-H singlet gate to the active sample."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first to add a singlet gate.")
            return
        df = self._samples[name].data
        cols = list(df.columns)

        def _find(suffix):
            if f'FSC{suffix}' in cols:
                return f'FSC{suffix}'
            return next((c for c in cols if c.upper().replace('-', '')
                         == f'FSC{suffix}'.replace('-', '')), None)
        area, height = _find('-A'), _find('-H')
        if not area or not height:
            messagebox.showinfo(
                "Singlet gate",
                "A singlet gate needs FSC-A and FSC-H channels; this sample "
                "doesn't have both.", parent=self)
            return
        try:
            from .gating_helpers import singlet_gate
            gate = singlet_gate(df, area=area, height=height)
        except Exception as exc:
            self.status_var.set(f"Singlet gate failed: {exc}")
            return
        gate.pop('id', None)             # _add_gate assigns the registry id
        gate['parent_id'] = None         # singlets is a root gate
        self._add_gate(gate)
        self._refresh_gate_list()
        self._schedule_replot(0)
        self.status_var.set(f"Added singlet gate ({area} vs {height}).")

    def _open_fmo_gating(self):
        """FMO-based threshold gating: map marker channels to FMO control
        samples; thresholds are placed at the FMO percentile on the active
        sample (copy to others with Edit → Copy gates to…)."""
        if self._active_sample is None or self._active_sample not in self._samples:
            self.status_var.set("Select the stained sample to gate first.")
            return
        if len(self._samples) < 2:
            messagebox.showinfo(
                "FMO gating",
                "Load your FMO control sample(s) as well as the stained "
                "sample, then map each marker to its FMO control here.",
                parent=self)
            return
        from .ui_fmo import FMOGatingDialog
        FMOGatingDialog(self)
