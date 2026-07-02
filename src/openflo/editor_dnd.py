"""Tree drag-and-drop, checkbox toggling, and subtree insertion.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin
from .theme import current_palette


class DnDMixin(EditorMixin):
    """Treeview interaction: insert sample subtrees, checkbox clicks, and drag-drop regrouping / drops onto workspace and stats windows."""

    def _insert_sample_subtree(self, name, parent_iid):
        """Insert one sample row (with its event count) plus its full gate
        hierarchy under ``parent_iid`` (a trial or a Comps/Samples subgroup)."""
        from .pipeline import describe_gate
        sample_iid = self._sample_iid(name)
        plot_on = self._sample_plot_enabled.get(name, True)
        # Neutral until displayed; the swatch colour matches the plot overlay
        # only while the sample is actually shown.
        fg = self._color_for(name) if plot_on else current_palette()['fg']
        sample_tag = f'sample_col_{name}'
        # Normal weight (not bold) so the ☑/☐ matches the clean auto-clean
        # rows; the sample stays distinguished by its colour, not heft.
        self.gate_tv.tag_configure(sample_tag, foreground=fg)
        shown, total = self._sample_display_count(name)
        cnt = (f'  ({shown:,}/{total:,})' if shown < total else f'  ({total:,})')
        # A sample staged for a cross-instance MOVE shows a ✄ marker + the
        # 'pending_move' tag, until the move completes (pasted) or is cancelled.
        moving = name in getattr(self, '_pending_move_names', set())
        mark = '✄' if moving else '■'
        tags = (sample_tag, 'pending_move') if moving else (sample_tag,)
        self.gate_tv.insert(
            parent_iid, 'end', iid=sample_iid,
            text=f'{mark} {name}{cnt}',
            values=('☑' if plot_on else '☐',),
            open=True, tags=tags)

        sample_gates = self._sample_gates.get(name, {})
        order        = self._sample_gate_order.get(name, [])
        inserted_gates = set()

        def insert_gate(gid, _name=name, _sg=sample_gates,
                        _siid=sample_iid, _ins=inserted_gates):
            # _ins default-binds the per-iteration set so the closure is fully
            # self-contained (silences B023).
            if gid in _ins:
                return
            g = _sg.get(gid)
            if g is None:
                return
            parent_gid = g.get('parent_id')
            if (parent_gid and parent_gid in _sg
                    and parent_gid not in _ins):
                insert_gate(parent_gid)
            if parent_gid and parent_gid in _ins:
                parent_tree_iid = self._gate_iid(_name, parent_gid)
            else:
                parent_tree_iid = _siid
            on = g.get('enabled', True)
            gate_color = g.get('color', '#000000')
            color_tag = f'col_{_name}_{gid}'
            self.gate_tv.tag_configure(color_tag, foreground=gate_color)
            tags = (color_tag,) if on else ('off',)
            # Auto-clean gates carry a "drops N (X%)" readout so the cleaning's
            # effect is visible without switching to Filter mode.
            ac_counts = (self._autoclean_counts(_name, gid)
                         if g.get('kind') == 'autoclean' else None)
            gate_text = describe_gate(g)
            if ac_counts is not None:
                gate_text += self._drop_suffix(ac_counts[1], ac_counts[0])
            self.gate_tv.insert(
                parent_tree_iid, 'end',
                iid=self._gate_iid(_name, gid),
                text=gate_text,
                values=('☑' if on else '☐',),
                open=g.get('open', True), tags=tags)
            _ins.add(gid)
            # An auto-clean gate is a GROUP: render one synthetic child row per
            # cleaning method (each toggleable). 'open' default False = collapsed.
            if g.get('kind') == 'autoclean':
                g_iid = self._gate_iid(_name, gid)
                total = ac_counts[0] if ac_counts else None
                per_method = ac_counts[2] if ac_counts else {}
                reasons = ac_counts[3] if ac_counts and len(ac_counts) > 3 else {}
                for m in g.get('methods', []):
                    mkey = m.get('key', '')
                    mon = m.get('enabled', True)
                    mtext = '   ' + (m.get('label') or mkey)
                    # Show the debris method's EFFECTIVE size cut, so it's clear
                    # 'bead' mode with no bead file in the run silently uses the
                    # valley cut (why switching bead↔valley then changes nothing).
                    if mkey == 'debris':
                        mp = m.get('params') or {}
                        if mp.get('mode', 'bead') == 'bead':
                            mtext += ('  [beads]' if mp.get('bead_fsc')
                                      else '  [beads→valley: no bead file]')
                        else:
                            mtext += '  [valley]'
                    if per_method.get(mkey) is not None:
                        mtext += self._drop_suffix(per_method[mkey], total)
                        # Explain a silent 0-drop (only for enabled methods).
                        if mon and per_method[mkey] == 0 and reasons.get(mkey):
                            mtext += f'  ·  {reasons[mkey]}'
                    self.gate_tv.insert(
                        g_iid, 'end',
                        iid=self._method_iid(_name, gid, mkey),
                        text=mtext,
                        values=('☑' if mon else '☐',),
                        open=True, tags=(() if mon else ('off',)))

        for gid in order:
            insert_gate(gid)
        for gid in list(sample_gates):
            insert_gate(gid)

    def _on_tv_press(self, event):
        """Record where the press started so we can disambiguate a click
        from a drag in the matching release/motion handlers. Doesn't
        suppress Treeview's default selection."""
        self._press_iid = self.gate_tv.identify_row(event.y)
        self._press_col = self.gate_tv.identify_column(event.x)
        self._press_x   = event.x
        self._press_y   = event.y
        self._drag_active = False
        # Snapshot the multi-selection BEFORE Tk's class binding collapses it
        # to the clicked row, so bulk display-toggle and multi-drag see it.
        self._press_selection = tuple(self.gate_tv.selection())
        # ALT held at press arms an OS (cross-instance) file-drag instead of the
        # in-app drag — consumed by _on_tree_os_drag_init. Plain drags leave this
        # False, so tkdnd aborts and the normal in-app drag runs unchanged. Alt
        # (not Shift/Ctrl) so it never triggers the tree's range/toggle select.
        # Alt mask: 0x20000 on Windows, 0x0008 (Mod1) on X11.
        _st = getattr(event, 'state', 0)
        self._os_drag_armed = bool(_st & 0x20000) or bool(_st & 0x0008)

    def _on_tree_os_drag_init(self, event):
        """tkdnd drag-source for the Samples & Gates tree. When ALT was held at
        press, hand the dragged sample(s)' FCS path(s) to the OS as a file drag,
        so they can be dropped onto ANOTHER OpenFlo window (which loads them via
        its file-drop handler) — "universal" cross-instance drag.

        Returns '' when NOT Alt-armed, which aborts the OS drag and leaves the
        plain drag as the in-app regroup / workspace / stats drag, untouched."""
        if not getattr(self, '_os_drag_armed', False):
            return ''
        import os as _os
        names = []
        try:
            names = list(self._target_samples('selected'))
        except Exception:
            names = []
        if not names and getattr(self, '_press_iid', None):
            parsed = self._parse_iid(self._press_iid)
            if parsed and parsed[0] == 'sample':
                names = [parsed[1]]
        paths = []
        for nm in names:
            s = self._samples.get(nm)
            p = getattr(s, 'path', '') if s else ''
            if p and p.lower().endswith('.fcs') and _os.path.isfile(p):
                paths.append(p)
        if not paths:
            return ''
        try:
            from tkinterdnd2 import COPY, DND_FILES
        except Exception:
            return ''
        self.status_var.set(
            f"Dragging {len(paths)} sample(s) — drop on another OpenFlo window "
            f"to copy them there.")
        return (COPY, DND_FILES, tuple(paths))

    def _on_tv_motion(self, event):
        """Once the cursor moves past the threshold on a gate or sample row, the
        gesture becomes a drag (cursor changes to fleur): a gate reparents, a
        sample regroups (move to another day, or between the Comps/Samples
        subgroups). The checkbox column and empty space aren't draggable."""
        if self._press_iid is None or self._drag_active:
            return
        if (abs(event.x - self._press_x) <= self._drag_threshold
                and abs(event.y - self._press_y) <= self._drag_threshold):
            return
        parsed = self._parse_iid(self._press_iid)
        if parsed is None:
            return
        if parsed[0] == 'trial':
            # Trial rows become draggable only to ferry their members to an open
            # Pipeline Workspace (no in-editor reparenting of a whole trial).
            if not self._workspace_open():
                return
        elif parsed[0] not in ('sample', 'gate'):
            return
        # Don't initiate drag from the toggle column (avoids accidental
        # drags during checkbox clicks).
        if self._press_col == '#1':
            return
        self._drag_active = True
        try:
            self.gate_tv.config(cursor='fleur')
        except Exception:
            pass

    def _on_tv_release(self, event):
        """Either:
          • the user just clicked (no drag): toggle ☑/☐ if they pressed
            the toggle column; otherwise let default selection stand.
          • the user dragged: reparent the source gate onto the row under
            the cursor (sample row → make it a root; gate row → make it
            that gate's child). Cycles and cross-sample drops are refused.
        """
        drag = self._drag_active
        press_iid = self._press_iid
        press_col = self._press_col
        self._press_iid = None
        self._press_col = None
        self._drag_active = False
        try:
            self.gate_tv.config(cursor='')
        except Exception:
            pass

        if drag:
            # Cross-window: a drag that ends over the Pipeline Workspace hands
            # that node to the workspace (with its cumulative gate chain + comp
            # matrix) instead of reparenting inside the editor.
            if self._maybe_drop_to_workspace(press_iid, event):
                return
            # …or over an open Statistics window → add its sample(s) there.
            if self._maybe_drop_to_stats(press_iid, event):
                return
            target_iid = self.gate_tv.identify_row(event.y)
            self._handle_drag_drop(press_iid, target_iid)
            return

        # Plain click. Toggle on press_col '#1' (the ☑ column).
        if press_iid and press_col == '#1':
            self._handle_checkbox_click(press_iid)

    def _display_toggle_target_state(self, parsed):
        """The new on/off state implied by clicking ``parsed``'s checkbox.
        Trial: off if any member is currently on, else on. Sample/gate: flip."""
        if parsed[0] == 'trial':
            members = self._trial_members(parsed[1])
            return not any(self._sample_plot_enabled.get(n, True) for n in members)
        if parsed[0] == 'subgroup':
            members = self._subgroup_members(parsed[1], parsed[2])
            return not any(self._sample_plot_enabled.get(n, True) for n in members)
        if parsed[0] == 'sample':
            return not self._sample_plot_enabled.get(parsed[1], True)
        if parsed[0] == 'gate':
            g = self._sample_gates.get(parsed[1], {}).get(parsed[2])
            return not (g.get('enabled', True) if g else True)
        if parsed[0] == 'method':
            g = self._sample_gates.get(parsed[1], {}).get(parsed[2])
            if g and g.get('kind') == 'autoclean':
                for m in g.get('methods', []):
                    if m.get('key') == parsed[3]:
                        return not m.get('enabled', True)
            return True
        return True

    def _handle_checkbox_click(self, row_id):
        clicked = self._parse_iid(row_id)
        if clicked is None:
            return
        # New state comes from the clicked row; with a live multi-selection
        # (captured pre-click) the same state is applied to every selected row.
        new_state = self._display_toggle_target_state(clicked)
        sel = self._press_selection or ()
        rows = sel if (row_id in sel and len(sel) > 1) else (row_id,)

        changed_plot = False
        changed_gate = False
        for r in rows:
            p = self._parse_iid(r)
            if p is None:
                continue
            if p[0] == 'trial':
                for n in self._trial_members(p[1]):
                    self._sample_plot_enabled[n] = new_state
                    changed_plot = True
            elif p[0] == 'subgroup':
                for n in self._subgroup_members(p[1], p[2]):
                    self._sample_plot_enabled[n] = new_state
                    changed_plot = True
            elif p[0] == 'sample':
                self._sample_plot_enabled[p[1]] = new_state
                changed_plot = True
            elif p[0] == 'gate':
                g = self._sample_gates.get(p[1], {}).get(p[2])
                if g is not None:
                    g['enabled'] = new_state
                    changed_gate = True
            elif p[0] == 'method' and len(p) == 4:
                g = self._sample_gates.get(p[1], {}).get(p[2])
                if g is not None and g.get('kind') == 'autoclean':
                    for m in g.get('methods', []):
                        if m.get('key') == p[3]:
                            m['enabled'] = new_state
                            changed_gate = True
                            break

        self._refresh_gate_list()
        if changed_plot:
            self._schedule_replot(0)
        elif changed_gate:
            mode = getattr(self, 'gate_display_var', None)
            show_removed = getattr(self, 'show_removed_var', None)
            # A full replot is needed when the display actually filters/highlights
            # on gates, OR when the cleaned-out-events overlay is showing — toggling
            # an auto-clean gate/method changes which events are removed, and the
            # overlay (unlike plain gate geometry) only redraws on a replot.
            # Otherwise just repaint the gate lines (cheap).
            if ((mode is not None and mode.get() in ('filter', 'highlight'))
                    or (show_removed is not None and bool(show_removed.get()))):
                self._schedule_replot(0)
            else:
                self._redraw_only_gates()

    def _maybe_drop_to_workspace(self, src_iid, event):
        """If a tree drag ended over a Pipeline Workspace tree (the docked pane's
        active tab, or a popped-out window), hand the dragged sample/leaf to that
        workspace instead of reparenting in the editor. The workspace snapshots
        the node's gate chain + linked comp matrix itself. Returns True iff the
        drop was consumed."""
        panel = getattr(self, '_workspace_panel', None)
        if panel is None:
            return False
        # Ferry the whole multi-selection if the dragged row is part of it,
        # else just the dragged row. Trials expand to their member samples.
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        nodes = []
        for s in srcs:
            p = self._parse_iid(s)
            if p is None:
                continue
            if p[0] in ('sample', 'gate'):
                nodes.append(p)
            elif p[0] == 'trial':
                nodes.extend(('sample', n) for n in self._trial_members(p[1]))
        if not nodes:
            return False
        try:
            # Geometry hit-test (not winfo_containing): during the drag's
            # pointer grab winfo_containing returns None over a popped-out
            # workspace toplevel, so a drop onto the FLOATING workspace wouldn't
            # register. point_in_target works docked OR floating.
            if panel.point_in_target(event.x_root, event.y_root) is None:
                return False
            # The workspace routes by the column under the pointer (drop on the
            # Comp/FMO column to assign beads/FMOs; elsewhere adds populations).
            return bool(panel.drop_at(self, nodes, event.x_root, event.y_root))
        except Exception:
            return False

    def _stats_window_under(self, x_root, y_root):
        """The open StatisticsWindow whose Toplevel contains the screen point,
        or None. Used as a cross-window drop target (editor tree + workspace)."""
        from .ui_statistics import StatisticsWindow
        try:
            w = self.winfo_containing(x_root, y_root)
        except Exception:
            w = None
        while w is not None:
            if isinstance(w, StatisticsWindow):
                return w
            w = getattr(w, 'master', None)
        # Geometry fallback (grab-proof): the stats window is always a separate
        # toplevel, where winfo_containing can return None mid-drag.
        win = getattr(self, '_tool_windows', {}).get('stats')
        try:
            if (win is not None and win.winfo_exists()
                    and win.winfo_ismapped()):
                x, y = win.winfo_rootx(), win.winfo_rooty()
                if (x <= x_root < x + win.winfo_width()
                        and y <= y_root < y + win.winfo_height()):
                    return win
        except Exception:
            pass
        return None

    def _dragged_gate_targets(self, src_iid):
        """(sample, gid) population targets implied by a tree drag: the dragged
        row plus the rest of the live multi-selection if the dragged row is
        part of it. ONLY gate rows qualify — whole-sample and trial rows are
        ignored (statistics is population-based). De-duplicated, order-keeping;
        only gates of loaded samples are returned."""
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        targets = []
        for s in srcs:
            p = self._parse_iid(s) if s else None
            if p and p[0] == 'gate':
                nm, gid = p[1], p[2]
                if nm in self._samples and (nm, gid) not in targets:
                    targets.append((nm, gid))
        return targets

    def _maybe_drop_to_stats(self, src_iid, event):
        """If a tree drag ends over an open Statistics window, add the dragged
        population(s) to it. Statistics accepts gate rows only — dropping a
        whole sample/trial is consumed with an explanatory status, never
        falls through to a reparent. Returns True iff over a stats window."""
        try:
            win = self._stats_window_under(event.x_root, event.y_root)
        except Exception:
            win = None
        if win is None:
            return False
        targets = self._dragged_gate_targets(src_iid)
        if not targets:
            self.status_var.set("Statistics accepts gate/population rows only "
                                "(not whole samples) — drag a gate.")
            return True
        try:
            win.add_targets(targets, 'editor')
            self.status_var.set(f"Added {len(targets)} population(s) to statistics.")
        except Exception:
            pass
        return True

    def _handle_drag_drop(self, src_iid, target_iid):
        """Reparent (within sample) or MOVE (across samples) the dragged
        gate, including its entire subtree, onto the drop target.

        Within-sample: just rewrites the dragged gate's parent_id.
        Cross-sample: copies the whole subtree to the destination with
        fresh gate ids + remapped parent_id pointers, then removes the
        originals from the source.

        Cycles (drop onto own descendant within the same sample) are
        refused. Same-parent drops are no-ops.
        """
        if not src_iid or not target_iid or src_iid == target_iid:
            return
        src = self._parse_iid(src_iid)
        tgt = self._parse_iid(target_iid)
        if not src or not tgt:
            return
        # A dragged SAMPLE regroups (move day / switch Comps<->Samples) rather
        # than reparenting gates.
        if src[0] == 'sample':
            self._regroup_dragged_samples(src_iid, target_iid)
            return
        if src[0] != 'gate':
            return
        # Only a sample or gate row is a valid reparent target — dropping onto a
        # trial / Comps-Samples subgroup / method header is ambiguous (no owning
        # sample), so ignore it.
        if tgt[0] not in ('sample', 'gate'):
            return
        src_sample, src_gid = src[1], src[2]
        tgt_sample = tgt[1]
        self._checkpoint()

        # Compute new parent in the *destination* sample.
        if tgt[0] == 'sample':
            new_parent = None
        else:  # gate tuple is ('gate', sample, gid)
            new_parent = tgt[2] if len(tgt) > 2 else None

        # ── Same-sample reparent ─────────────────────────────────────────
        if tgt_sample == src_sample:
            if new_parent == src_gid:
                return
            if (new_parent is not None
                    and self._is_descendant_of(src_sample, src_gid, new_parent)):
                self.status_var.set(
                    "Drop refused: would create a cycle in the gate tree.")
                return
            sgates = self._sample_gates.get(src_sample, {})
            g = sgates.get(src_gid)
            if g is None:
                return
            if g.get('parent_id') == new_parent:
                return  # no-op
            g['parent_id'] = new_parent
            self._refresh_gate_list()
            try:
                self.gate_tv.selection_set(self._gate_iid(src_sample, src_gid))
                self.gate_tv.see(self._gate_iid(src_sample, src_gid))
            except Exception:
                pass
            if self.gate_display_var.get() in ('filter', 'highlight'):
                self._schedule_replot(0)
            from .pipeline import describe_gate
            self.status_var.set(
                f"Reparented {describe_gate(g)} → "
                f"{'root of ' + src_sample if new_parent is None else 'under ' + new_parent}.")
            return

        # ── Cross-sample move ────────────────────────────────────────────
        moved_root_new_gid = self._move_gate_to_sample(
            src_sample, src_gid, tgt_sample, new_parent)
        if moved_root_new_gid is None:
            return
        self._refresh_gate_list()
        try:
            self.gate_tv.selection_set(
                self._gate_iid(tgt_sample, moved_root_new_gid))
            self.gate_tv.see(
                self._gate_iid(tgt_sample, moved_root_new_gid))
        except Exception:
            pass
        if self.gate_display_var.get() in ('filter', 'highlight'):
            self._schedule_replot(0)
        else:
            self._redraw_only_gates()
        self.status_var.set(
            f"Moved gate from '{src_sample}' to '{tgt_sample}' "
            f"(now as {'root' if new_parent is None else 'child of ' + new_parent}).")

    def _regroup_target(self, tgt):
        """Resolve a drop target row into (new_trial, new_comp) for sample
        regrouping. Either may be None ('leave as-is'):
          • trial row     → (that trial, None)
          • subgroup row   → (its trial, True/False for Comps/Samples)
          • sample/gate/method row → that row's owning sample's (trial, comp)."""
        if tgt[0] == 'trial':
            return tgt[1], None
        if tgt[0] == 'subgroup':
            return tgt[2], (tgt[1] == 'comp')
        if tgt[0] in ('sample', 'gate', 'method'):
            ref = tgt[1]
            if ref in self._samples:
                return self._trial_for(ref), self._is_comp(ref)
        return None, None

    def _regroup_dragged_samples(self, src_iid, target_iid):
        """Move the dragged sample(s) to the drop target's day and/or
        Comps↔Samples subgroup. Honors a live multi-selection. Reassigning the
        Comps/Samples side sets a manual override (`_sample_is_comp`)."""
        tgt = self._parse_iid(target_iid)
        if tgt is None:
            return
        new_trial, new_comp = self._regroup_target(tgt)
        if new_trial is None and new_comp is None:
            return
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        names = []
        for s in srcs:
            p = self._parse_iid(s) if s else None
            if p and p[0] == 'sample' and p[1] in self._samples \
                    and p[1] not in names:
                names.append(p[1])
        if not names:
            return
        moved = 0
        for n in names:
            changed = False
            if new_trial is not None and self._trial_for(n) != new_trial:
                self._sample_trial[n] = new_trial
                if new_trial not in self._trial_order:
                    self._trial_order.append(new_trial)
                changed = True
            if new_comp is not None and self._is_comp(n) != new_comp:
                self._sample_is_comp[n] = new_comp
                changed = True
            if changed:
                moved += 1
        if not moved:
            return
        # Drop now-empty trials from the order list.
        self._trial_order = [t for t in self._trial_order
                             if any(self._trial_for(n) == t
                                    for n in self._samples)]
        self._refresh_gate_list()
        self._schedule_replot(0)
        dest = new_trial if new_trial is not None else self._trial_for(names[0])
        side = ('' if new_comp is None
                else ' / Comps' if new_comp else ' / Samples')
        self.status_var.set(f"Moved {moved} sample(s) → {dest}{side}.")

    def _toggle_all_sample_plots(self):
        """Header ✓ click: if ANY loaded sample is plot-on, turn them all
        off; otherwise turn them all on. Gates are untouched."""
        if not self._samples:
            return
        any_on = any(self._sample_plot_enabled.get(n, True)
                     for n in self._samples)
        new_state = not any_on
        for name in self._samples:
            self._sample_plot_enabled[name] = new_state
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _toggle_expand_all(self):
        """#0-heading click: expand every group / subgroup / trial if anything
        is currently collapsed, otherwise collapse them all. A tidy one-click
        way to fold up clustered-population groups, Samples/Comps subgroups,
        and trials at once."""
        any_collapsed = False

        def walk(parent):
            nonlocal any_collapsed
            for iid in self.gate_tv.get_children(parent):
                if self.gate_tv.get_children(iid) and not self.gate_tv.item(
                        iid, 'open'):
                    any_collapsed = True
                walk(iid)
        walk('')
        self._set_all_expanded(any_collapsed)

    def _set_all_expanded(self, expanded):
        """Expand (or collapse) every container. Gate containers persist via
        their gate dict's 'open'; trials/subgroups persist because
        _refresh_gate_list snapshots the live tree's open state, so we set it
        there before rebuilding."""
        for gates in self._sample_gates.values():
            for g in gates.values():
                g['open'] = expanded
        # Set open on every live item too, so _refresh_gate_list's snapshot
        # of trials / subgroups / auto-clean groups picks up the new state.
        def walk(parent):
            for iid in self.gate_tv.get_children(parent):
                self.gate_tv.item(iid, open=expanded)
                walk(iid)
        walk('')
        self.gate_tv.heading(
            '#0', text=('▾ ' if expanded else '▸ ') + 'Samples & Gates')
        self._refresh_gate_list()
