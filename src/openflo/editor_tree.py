"""Samples-and-gates tree: build + interaction.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk

from .editor_base import EditorMixin
from .theme import current_palette


class _TreeRecording:
    """Stands in for the Treeview while the tree's rows are built: records
    each ``insert`` and ``tag_configure`` so the build can be compared with
    the one the tree already shows, and replayed onto it when it differs."""

    def __init__(self):
        self.ops = []

    def insert(self, parent, index, **kw):
        self.ops.append(('insert', parent, index, tuple(sorted(kw.items()))))
        return kw.get('iid')

    def tag_configure(self, tag, **kw):
        self.ops.append(('tag', tag, tuple(sorted(kw.items()))))

    def replay(self, tree):
        for op in self.ops:
            if op[0] == 'insert':
                tree.insert(op[1], op[2], **dict(op[3]))
            else:
                tree.tag_configure(op[1], **dict(op[2]))


class TreeMixin(EditorMixin):
    """Tree rebuild, selection, double-click, right-click menu, find, delete."""

    def _on_tv_double_click(self, event):
        """Double-clicking an auto-clean gate row (or one of its method rows)
        opens the parameter dialog. Other rows are unaffected."""
        try:
            row = self.gate_tv.identify_row(event.y)
        except Exception:
            return
        p = self._parse_iid(row) if row else None
        if p is None:
            return
        if p[0] == 'method':
            self._edit_autoclean_params(p[1], p[2])
            return 'break'
        if p[0] == 'gate':
            g = self._sample_gates.get(p[1], {}).get(p[2])
            if g is not None and g.get('kind') == 'autoclean':
                self._edit_autoclean_params(p[1], p[2])
                return 'break'

    def _on_tree_select(self, *_):
        """Row selection switched: update active sample (= owner of the
        selected row, sample row or gate row alike) and replot if it
        actually changed. Also show the row's event count (and % of parent
        for a gate) in the status bar."""
        sel = self.gate_tv.selection()
        if not sel:
            return
        parsed = self._parse_iid(sel[0])
        if parsed is None:
            return
        if parsed[0] in ('trial', 'subgroup'):
            return  # group header — expand/collapse + display toggle only
        owning_sample = parsed[1]
        echo = getattr(self, '_tree_select_echo', None)
        # One event consumes the echo mark, whichever row it is for.
        self._tree_select_echo = None
        if sel[0] == echo and owning_sample == self._active_sample:
            # The tree rebuild re-selecting the row it already had (see
            # _refresh_gate_list): nothing changed, so nothing to redo.
            return
        # The layout editor drops a population "as it was last displayed":
        # remember how the plot looked while the previous row was selected.
        self._remember_population_view()
        self._viewed_population = (
            (parsed[1], parsed[2]) if parsed[0] == 'gate'
            else (parsed[1], None) if parsed[0] == 'sample' else None)
        if owning_sample and owning_sample != self._active_sample:
            self._set_active_sample(owning_sample)
            # Don't re-refresh the tree (would clobber selection); just
            # repaint the canvas.
            self._schedule_replot(50)
        if owning_sample and self._plot_sample_row(owning_sample):
            self._schedule_replot(50)
        if parsed[0] == 'gate':
            self._show_gate_count(parsed[1], parsed[2])
        elif parsed[0] == 'sample':
            self._show_sample_count(parsed[1])

    def _plot_sample_row(self, name):
        """Tick ``name`` on for plotting if it isn't, updating its row in
        place. Gates, compensation and clustering act on the highlighted
        sample, so it must be one the user can see: highlighting an unticked
        sample left the plot on another one, and a gate drawn there went to
        the plotted samples and the highlighted one. True when it changed."""
        if self._sample_plot_enabled.get(name, True):
            return False
        self._sample_plot_enabled[name] = True
        try:
            self.gate_tv.item(self._sample_iid(name), values=('☑',))
            self.gate_tv.tag_configure(f'sample_col_{name}',
                                       foreground=self._color_for(name))
        except Exception:
            pass
        return True

    def _show_gate_count(self, sname, gid):
        """Status-bar readout for the selected gate: event count and % of its
        parent population (or % of all events for a root gate) — the headline
        number in cytometry. Best-effort; never raises into the selection."""
        try:
            from .gating import format_gate_count
            from .pipeline import unknown_population_reason
            s = self._samples.get(sname)
            sample_gates = self._sample_gates.get(sname, {})
            g = sample_gates.get(gid)
            if s is None or g is None:
                return
            why = unknown_population_reason(sample_gates, gid,
                                            columns=s.data.columns)
            if why:
                # It selects nothing, but that 0 was never measured.
                self.status_var.set(f"{g.get('name', 'gate')}:  n = NA   "
                                    f"(cannot be computed: {why})")
                return
            # gating.gate_counts' answer, from the per-sample count cache
            # (computed on all events; one chain walk shared by the gate and
            # its parent; usually already warmed in the background).
            pid = g.get('parent_id')
            has_parent = bool(pid) and pid in sample_gates
            counts = self._pc_counts(sname, [gid, pid] if has_parent else [gid])
            if has_parent:
                n_gate, n_parent, of = counts[gid], counts[pid], 'parent'
            else:
                n_gate, n_parent, of = counts[gid], len(s.data), 'all'
            self.status_var.set(
                format_gate_count(g.get('name', 'gate'), n_gate, n_parent, of))
        except Exception as exc:
            print(f"[gate-count] {sname}/{gid}: "
                  f"{type(exc).__name__}: {exc}", flush=True)

    # Gate kinds whose description is their bounds: an unnamed one is named
    # by what it gates on instead, so its row has room for its count.
    _BOUNDS_KINDS = ('threshold', 'interval', 'rect', 'polygon', 'ellipsoid')

    def _tree_gate_text(self, sname, gid, transforms=None, counts=None):
        """A gate's row in the tree: its name and its population, as
        'Lymphocytes  18,650 · 93.2%'. The bounds the row used to show
        ('R  FSC-A x SSC-A  FSC-A in [11.4K, 215K] x ...') were cut off at
        the panel's width and hid the count; the plot shows them. An
        unnamed gate is named by its channels ('FSC-A x SSC-A'), or for a
        1-D one by its cut ('CD3-A >= 987'). The count is left off until it
        has been computed (rows are filled in as counts arrive), and reads
        NA when the population cannot be computed
        (unknown_population_reason), never a 0 that was not measured."""
        from .gating import format_percent
        from .pipeline import describe_gate, unknown_population_reason
        gates = self._sample_gates.get(sname, {}) or {}
        g = gates.get(gid) or {}
        kind = g.get('kind')
        if kind not in self._BOUNDS_KINDS:
            text = describe_gate(g, transforms=transforms)
        else:
            text = str(g.get('name') or g.get('label') or '')
            if not text:
                x, y = g.get('x_channel'), g.get('y_channel')
                if x and y:
                    text = f"{x} x {y}"
                else:
                    # 'T  CD3-A >= 987' -> 'CD3-A >= 987'
                    text = describe_gate(
                        g, transforms=transforms).split('  ', 1)[-1]
        if kind in ('group', 'autoclean'):
            return text          # a container, or its own "drops N" readout
        s = self._samples.get(sname)
        cols = getattr(getattr(s, 'data', None), 'columns', None)
        if cols is None:
            return text
        if unknown_population_reason(gates, gid, columns=cols):
            return f"{text}  NA"
        counts = counts or {}
        n = counts.get(gid)
        if n is None:
            return text
        pid = g.get('parent_id')
        n_parent = (counts.get(pid) if pid and pid in gates
                    else len(s.data))
        if n_parent is None:
            return f"{text}  {n:,}"
        return f"{text}  {n:,} · {format_percent(n, n_parent)}"

    def _tree_counts_changed(self, sname):
        """New gate counts were cached for `sname`: refresh its gate rows'
        text in place, once per idle pass however many arrive."""
        pend = self.__dict__.setdefault('_tree_count_pending', set())
        if sname in pend:
            return
        pend.add(sname)
        try:
            self.after_idle(self._tree_fill_counts)
        except Exception:                                  # noqa: BLE001
            pend.discard(sname)

    def _tree_fill_counts(self):
        """Rewrite the text of the gate rows of the samples whose counts
        arrived (see `_tree_counts_changed`); updated, never rebuilt."""
        names = self.__dict__.pop('_tree_count_pending', set())
        tv = getattr(self, 'gate_tv', None)
        if tv is None:
            return
        for sname in names:
            gates = self._sample_gates.get(sname, {}) or {}
            if sname not in self._samples or not gates:
                continue
            counts = self._pc_peek_counts(sname, list(gates))
            transforms = self._display_transforms(sname)
            for gid, g in gates.items():
                if g.get('kind') in ('group', 'autoclean'):
                    continue
                iid = self._gate_iid(sname, gid)
                try:
                    if tv.exists(iid):
                        tv.item(iid, text=self._tree_gate_text(
                            sname, gid, transforms, counts))
                except Exception:                          # noqa: BLE001
                    pass
        self._fit_tree_column()

    def _fit_tree_column(self):
        """Widen the tree column to its longest row so no row is cut
        off; the tree scrolls sideways when that is wider than the panel."""
        tv = getattr(self, 'gate_tv', None)
        if tv is None:
            return
        try:
            import tkinter.font as tkfont
            font = tkfont.nametofont('TkDefaultFont')
            indent = 20                     # ttk.Treeview's default indent
            widest = 0
            stack = [(iid, 1) for iid in tv.get_children('')]
            while stack:
                iid, depth = stack.pop()
                widest = max(widest, font.measure(tv.item(iid, 'text'))
                             + depth * indent + 12)
                # Collapsed rows too, so expanding one never cuts it off.
                stack.extend((c, depth + 1) for c in tv.get_children(iid))
            tv.column('#0', width=max(212, widest))
        except Exception:                                  # noqa: BLE001
            pass

    def _show_sample_count(self, sname):
        """Status-bar readout for a selected sample row: total event count."""
        try:
            s = self._samples.get(sname)
            if s is not None:
                self.status_var.set(f"{sname}:  {len(s.data):,} events")
        except Exception:
            pass

    def _target_samples(self, mode='selected'):
        """The single place that turns UI state into a list of sample names,
        always in load order, de-duplicated, and present in ``_samples``:

          'selected' : the tree selection — sample/gate/method rows → owning
                       sample, trial/subgroup rows → members — falling back to
                       the active sample so a plain click still targets one.
          'enabled'  : samples checked for plotting (☑).
          'all'      : every loaded sample.
          'active'   : just the active sample (or []).

        Replaces the half-dozen ad-hoc selection→samples reimplementations; the
        named wrappers below (``_selected_sample_names`` / ``_selected_samples``
        / ``_loaded_samples``) delegate here."""
        order = self._sample_order
        present = self._samples
        if mode == 'active':
            return [self._active_sample] if self._active_sample in present else []
        if mode == 'all':
            return [n for n in order if n in present]
        if mode == 'enabled':
            # A tray tab plots only its own samples.
            return [n for n in order
                    if n in present and self._sample_plot_enabled.get(n, True)
                    and self._tab_has_sample(n)]
        # 'selected'
        picked = set()
        try:
            sel = self.gate_tv.selection()
        except Exception:
            sel = ()
        for iid in sel:
            p = self._parse_iid(iid)
            if not p:
                continue
            if p[0] in ('sample', 'gate', 'method'):
                picked.add(p[1])
            elif p[0] == 'trial':
                picked.update(self._trial_members(p[1]))
            elif p[0] == 'subgroup':
                picked.update(self._subgroup_members(p[1], p[2]))
        out = [n for n in order if n in picked and n in present]
        if not out and self._active_sample in present:
            out = [self._active_sample]
        return out

    def _selected_sample_names(self):
        """Samples covered by the current tree selection (trial/subgroup rows
        expanded to members), or the active sample if none. Used by per-sample
        actions (e.g. Auto-clean) so selecting several applies to all."""
        return self._target_samples('selected')

    def _refresh_gate_list(self):
        """Rebuild the samples-and-gates tree, grouped by trial:
        trial row → its samples → each sample's gate hierarchy. The trial row's
        ☑ column toggles plot-display for every sample in that trial; clicking
        the disclosure triangle collapses/expands the trial. iid encoding lives
        with `_trial_iid` / `_sample_iid` / `_gate_iid`."""

        sel_iid = None
        cur = self.gate_tv.selection()
        if cur:
            sel_iid = cur[0]

        # Preserve which trials the user had collapsed across the rebuild,
        # plus each Comps/Samples subgroup's open state (default: Samples open,
        # Comps collapsed).
        collapsed = set()
        sub_open = {}        # (kind, trial) -> bool
        for iid in self.gate_tv.get_children(''):
            p = self._parse_iid(iid)
            if p and p[0] == 'trial':
                if not self.gate_tv.item(iid, 'open'):
                    collapsed.add(iid)
                for sg_iid in self.gate_tv.get_children(iid):
                    sp = self._parse_iid(sg_iid)
                    if sp and sp[0] == 'subgroup':
                        sub_open[(sp[1], sp[2])] = bool(
                            self.gate_tv.item(sg_iid, 'open'))
        # Persist each auto-clean group's expand/collapse choice back into its
        # gate dict so a rebuild keeps the user's state (default: collapsed).
        for nm, gates in self._sample_gates.items():
            for gid, g in gates.items():
                if g.get('kind') == 'autoclean':
                    giid = self._gate_iid(nm, gid)
                    if self.gate_tv.exists(giid):
                        g['open'] = bool(self.gate_tv.item(giid, 'open'))
        # The rows are built into a recording first and replayed onto the
        # tree only when they differ from what it already shows: a replot
        # (every zoom and axis change) asks for a rebuild, and redoing an
        # identical tree cost a delete, re-insert and re-measure of every row.
        tv = _TreeRecording()

        # No font override: inherit the Treeview's real default font so the
        # day's aggregate box matches samples / auto-clean rows exactly. (A
        # ('TkDefaultFont', N) tuple is WRONG here — it's read as a missing
        # family name, falling back to a different font with different glyph
        # metrics, which made the day checkboxes look mis-sized.) The ▦ prefix
        # still marks the header; foreground is the themed default.
        tv.tag_configure('trial_row',
                                   foreground=current_palette()['fg'])
        tv.tag_configure('subgroup_row',
                                   foreground=current_palette()['muted'])
        # Still-loading sample rows (⏳) are muted so it's visually clear they
        # aren't ready yet.
        tv.tag_configure('loading',
                                   foreground=current_palette()['muted'])

        def _agg_mark(names):
            en = [self._sample_plot_enabled.get(n, True) for n in names]
            return '☑' if all(en) else ('☐' if not any(en) else '▣')

        # Still-loading samples (queued, not yet in `_samples`) are shown as
        # persistent ⏳ rows under their trial. Without this, the first
        # `_on_loaded` refresh would rebuild the tree from loaded samples only,
        # so every not-yet-ready sample's placeholder would vanish and pop back
        # one-by-one — making a multi-file (or big-file) load look stalled. We
        # derive each loading sample's trial from its queued path so it lands in
        # the right group immediately.
        from .workspace import derive_trial_name
        loading_by_trial: dict[str, list[str]] = {}
        for nm in self._loading:
            if nm in self._samples:
                continue
            # Prefer a trial we already know (set from session metadata during
            # restore); otherwise derive it from the queued file path.
            pth = self._name_to_path.get(nm)
            tr = (self._sample_trial.get(nm)
                  or (derive_trial_name(pth) if pth else ''))
            loading_by_trial.setdefault(tr, []).append(nm)

        ordered_trials = list(self._ordered_trials())
        for tr in loading_by_trial:
            if tr not in ordered_trials:
                ordered_trials.append(tr)

        # The active tab's Presort zone, on top while it holds anything.
        self._insert_presort_zone(tv)

        for trial in ordered_trials:
            members = [n for n in self._tab_sample_order()
                       if n in self._samples and self._trial_for(n) == trial
                       and self._tab_has_sample(n)]
            pending = loading_by_trial.get(trial, [])
            if not members and not pending:
                continue
            t_iid = self._trial_iid(trial)
            count = len(members) + len(pending)
            tv.insert(
                '', 'end', iid=t_iid,
                text=f'▦ {trial}  ({count})',
                values=(_agg_mark(members) if members else '☐',),
                open=(t_iid not in collapsed), tags=('trial_row',))

            # Split into Comps vs Samples. Only introduce the subgroup headers
            # when comps are actually present (otherwise list samples directly
            # under the trial, as before). Samples first (expanded), Comps
            # second (collapsed by default).
            comps = [n for n in members if self._is_comp(n)]
            if comps:
                samps = [n for n in members if not self._is_comp(n)]
                for kind, sub, default_open, label in (
                        ('samp', samps, True, 'Samples'),
                        ('comp', comps, False, 'Comps')):
                    if not sub:
                        continue
                    sg_iid = self._subgroup_iid(kind, trial)
                    tv.insert(
                        t_iid, 'end', iid=sg_iid,
                        text=f'{label}  ({len(sub)})',
                        values=(_agg_mark(sub),),
                        open=sub_open.get((kind, trial), default_open),
                        tags=('subgroup_row',))
                    for name in sub:
                        self._insert_sample_subtree(name, sg_iid, tv)
            else:
                for name in members:
                    self._insert_sample_subtree(name, t_iid, tv)

            # Still-loading placeholders, last, directly under the trial row.
            for nm in pending:
                tv.insert(
                    t_iid, 'end', iid=self._sample_iid(nm),
                    text=f'⏳ {nm}', values=('',), tags=('loading',))

        if tv.ops == getattr(self, '_tree_shown_ops', None):
            return
        for iid in self.gate_tv.get_children(''):
            self.gate_tv.delete(iid)
        tv.replay(self.gate_tv)
        self._tree_shown_ops = tv.ops

        if sel_iid:
            # Re-selecting fires <<TreeviewSelect>> again (queued, after this
            # returns). Mark it as an echo so _on_tree_select doesn't redo the
            # gate count for a row that was already handled; the event itself
            # clears the mark. It used to be cleared at idle, but on Windows
            # the queued event can arrive AFTER idle: it then rewrote the
            # status bar over whatever was set meanwhile (an FMO result, a
            # refusal). The timer only covers an event that never comes.
            self._tree_select_echo = sel_iid
            try:
                prev = getattr(self, '_tree_select_echo_after', None)
                if prev is not None:
                    self.after_cancel(prev)
                self._tree_select_echo_after = self.after(
                    500, self._clear_tree_select_echo)
            except Exception:
                self._tree_select_echo = None
            try:
                self.gate_tv.selection_set(sel_iid)
            except Exception:
                pass
        self._fit_tree_column()

    def _clear_tree_select_echo(self):
        self._tree_select_echo = None
        self._tree_select_echo_after = None

    def _find_in_tree(self):
        """Find box: select + scroll to the first tree row (sample or gate)
        whose name contains the query, expanding its ancestors."""
        q = self._find_var.get().strip().lower()
        if not q:
            return

        def _walk(parent=''):
            for iid in self.gate_tv.get_children(parent):
                yield iid
                yield from _walk(iid)
        for iid in _walk():
            try:
                if q in self.gate_tv.item(iid, 'text').lower():
                    p = self.gate_tv.parent(iid)
                    while p:
                        self.gate_tv.item(p, open=True)
                        p = self.gate_tv.parent(p)
                    self.gate_tv.see(iid)
                    self.gate_tv.selection_set(iid)
                    self.gate_tv.focus(iid)
                    return
            except Exception:
                continue

    def _on_delete_key(self, event=None):
        """Delete key: clear the selected gate (cascade), or — on a TRIAL row —
        remove that trial's samples + gates (confirmed). Plain sample rows are
        ignored to avoid accidental keyboard deletion; use the Remove button."""
        sel = self.gate_tv.selection()
        if not sel:
            return 'break'
        parsed = self._parse_iid(sel[0])
        if parsed and parsed[0] == 'gate':
            self._clear_selected_gate()
        elif parsed and parsed[0] == 'trial':
            self._remove_selected()
        return 'break'

    def _theme_menu(self, menu):
        """Colour a popup ``tk.Menu`` from the palette. tk menus don't follow
        ttk styles, so an un-themed context menu renders with the OS default
        colours — on a dark theme that makes disabled items (e.g. a greyed-out
        Paste) look garbled. Best-effort; a failure just leaves OS defaults."""
        pal = current_palette()
        try:
            menu.configure(
                bg=pal['bg'], fg=pal['fg'],
                activebackground=pal['active'], activeforeground=pal['fg'],
                disabledforeground=pal.get('muted', '#9aa0a6'),
                bd=0, relief='flat', activeborderwidth=0)
        except Exception:
            pass

    def _on_right_click(self, event):
        """Pop a context menu appropriate to whatever was right-clicked
        (gate row / sample row / empty space). Defensive — any failure
        building the menu just prints and bails (no crash)."""
        try:
            row_id = self.gate_tv.identify_row(event.y)
        except Exception:
            return 'break'
        if row_id:
            try:
                # Keep an existing multi-selection intact when right-clicking
                # one of its rows (so "backgate" can act on all selected
                # populations); only collapse to the clicked row when it isn't
                # already part of the selection.
                if row_id not in self.gate_tv.selection():
                    self.gate_tv.selection_set(row_id)
            except Exception:
                pass
        parsed = self._parse_iid(row_id) if row_id else None
        if row_id and str(row_id).startswith('PRE:'):
            return self._presort_menu(event)

        menu = tk.Menu(self.gate_tv, tearoff=0)
        self._theme_menu(menu)
        try:
            paste_avail = bool(self._clip_payload) or bool(
                self._paste_fcs_from_clipboard())
        except Exception:
            paste_avail = False
        paste_state = 'normal' if paste_avail else 'disabled'
        try:
            transfer_state = ('normal' if self._read_transfer_bundle()
                              is not None else 'disabled')
        except Exception:
            transfer_state = 'disabled'

        if parsed and parsed[0] == 'gate':
            g = self._sample_gates.get(parsed[1], {}).get(parsed[2])
            if g is not None and g.get('kind') == 'autoclean':
                menu.add_command(
                    label="Edit auto-clean parameters…",
                    command=lambda n=parsed[1], gd=parsed[2]:
                        self._edit_autoclean_params(n, gd))
                menu.add_separator()
            menu.add_command(
                label="Set color…",
                command=lambda n=parsed[1], gd=parsed[2]:
                    self._pick_gate_color(n, gd))
            menu.add_separator()
            menu.add_command(label="Copy gate (Ctrl+C)",
                             command=self._on_copy)
            menu.add_command(label="Cut gate (Ctrl+X)",
                             command=self._on_cut)
            menu.add_command(label="Paste (Ctrl+V)",
                             command=self._on_paste, state=paste_state)
            menu.add_separator()
            from .gate_coords import editable as _coords_ok
            from .gate_refit import refittable as _refit_ok
            if g is not None and _coords_ok(g):
                menu.add_command(
                    label="Edit coordinates…",
                    command=lambda n=parsed[1], gd=parsed[2]:
                        self._edit_gate_coordinates(n, gd))
            if g is not None and _refit_ok(g):
                menu.add_command(
                    label="Re-place on each sample…",
                    command=lambda n=parsed[1], gd=parsed[2]:
                        self._refit_gate_per_sample(n, gd))
            menu.add_command(label="Create boolean gate…",
                             command=self._open_boolean_dialog)
            menu.add_command(label="Combination gates…",
                             command=self._open_combination_dialog)
            menu.add_separator()
            menu.add_command(label="Backgate (show on plot)",
                             command=self._backgate_selected)
            if self._backgate:
                menu.add_command(label="Clear backgating",
                                 command=self._clear_backgate)
            menu.add_separator()
            menu.add_command(
                label="Export population as FCS…",
                command=lambda n=parsed[1], gd=parsed[2]:
                    self._export_population_fcs(n, gd))
            menu.add_separator()
            _is_ac = g is not None and g.get('kind') == 'autoclean'
            menu.add_command(
                label=("Remove auto-clean gate" if _is_ac
                       else "Delete gate (cascade)"),
                command=self._clear_selected_gate)
        elif parsed and parsed[0] == 'method':
            n, gd, mkey = parsed[1], parsed[2], parsed[3]
            if mkey == 'debris':
                _g, m = self._autoclean_method(n, gd, 'debris')
                from .pipeline import _debris_mode
                mp = (m.get('params') if m else {}) or {}
                mode = _debris_mode(mp)
                sub = tk.Menu(menu, tearoff=0)
                self._theme_menu(sub)
                sub.add_command(
                    label=("• " if mode == 'bead' else "    ")
                          + "Beads (absolute size)",
                    command=lambda: self._autoclean_set_debris_mode(n, gd, 'bead'))
                sub.add_command(
                    label=("• " if mode == 'valley' else "    ")
                          + "Valley (density; can cut small cells)",
                    command=lambda: self._autoclean_set_debris_mode(n, gd, 'valley'))
                menu.add_cascade(label="Debris method", menu=sub)
                menu.add_command(
                    label="Bead size (µm)…",
                    command=lambda: self._autoclean_prompt_float(
                        n, gd, 'debris', 'bead_um', "Bead size",
                        "Calibration bead diameter (µm):", 8.0, 0.1))
                menu.add_command(
                    label="Min cell size (µm)…",
                    command=lambda: self._autoclean_prompt_float(
                        n, gd, 'debris', 'min_um', "Min cell size",
                        "Smallest size to keep (µm):", 4.0, 0.0))
                menu.add_command(
                    label="Re-detect bead reference",
                    command=lambda: self._autoclean_set_debris_mode(n, gd, 'bead'))
                menu.add_separator()
            elif mkey == 'viability':
                _g, m = self._autoclean_method(n, gd, 'viability')
                cur = ((m.get('params') if m else {}) or {}).get('channel')
                sd = getattr(self._samples.get(n), 'data', None)
                sub = tk.Menu(menu, tearoff=0)
                self._theme_menu(sub)
                sub.add_command(
                    label=("• " if not cur else "    ") + "Auto-detect",
                    command=lambda: self._autoclean_set_viability_channel(n, gd, None))
                if sd is not None:
                    for col in list(sd.columns):
                        cl = str(col)
                        if (cl.lower() == 'time' or cl.endswith('_pos')
                                or cl.upper().startswith(('FSC', 'SSC'))):
                            continue
                        sub.add_command(
                            label=("• " if col == cur else "    ") + cl,
                            command=lambda c=col:
                                self._autoclean_set_viability_channel(n, gd, c))
                menu.add_cascade(label="Viability channel", menu=sub)
                menu.add_separator()
            menu.add_command(
                label="Edit auto-clean parameters…",
                command=lambda: self._edit_autoclean_params(n, gd))
        elif parsed and parsed[0] == 'sample':
            menu.add_command(
                label="Set color…",
                command=lambda n=parsed[1]: self._pick_sample_color(n))
            menu.add_separator()
            _sel = self._target_samples('selected') or [parsed[1]]
            if parsed[1] in self._sample_set_aside:
                menu.add_command(
                    label="Return to comparisons",
                    command=lambda s=_sel: self._return_sample(s))
                menu.add_command(
                    label="Change set-aside reason…",
                    command=lambda s=_sel: self._prompt_set_aside(s))
            else:
                menu.add_command(
                    label="Set aside from comparisons…",
                    command=lambda s=_sel: self._prompt_set_aside(s))
            menu.add_command(
                label=("Edit note…" if parsed[1] in self._sample_notes
                       else "Add note…"),
                command=lambda n=parsed[1]: self._prompt_sample_note(n))
            menu.add_separator()
            menu.add_command(label="Create boolean gate…",
                             command=self._open_boolean_dialog)
            menu.add_command(label="Combination gates…",
                             command=self._open_combination_dialog)
            menu.add_separator()
            menu.add_command(label="Copy gates (Ctrl+C)",
                             command=self._on_copy)
            menu.add_command(label="Paste (Ctrl+V)",
                             command=self._on_paste, state=paste_state)
            menu.add_separator()
            menu.add_command(label="Copy sample file path",
                             command=lambda: self._copy_sample_path(parsed[1]))
            menu.add_command(label="Copy gates to other samples…",
                             command=self._open_copy_gates_dialog)
            menu.add_separator()
            menu.add_command(
                label="Copy sample(s) for another instance (with gates)",
                command=lambda n=parsed[1]: self._copy_samples_transfer(
                    self._target_samples('selected') or [n]))
            menu.add_command(
                label="Send sample(s) to another instance (move)",
                command=lambda n=parsed[1]: self._send_samples_transfer(
                    self._target_samples('selected') or [n]))
            menu.add_command(label="Paste sample(s) from another instance",
                             command=self._paste_samples_transfer,
                             state=transfer_state)
            menu.add_separator()
            menu.add_command(label="Clear sample's gates",
                             command=self._clear_selected_gate)
            menu.add_command(label="Remove sample",
                             command=self._remove_selected)
        elif parsed and parsed[0] == 'trial':
            menu.add_command(label="Clear trial's gates (samples kept)",
                             command=self._clear_selected_gate)
            menu.add_command(label="Remove trial (samples + gates)",
                             command=self._remove_selected)
        else:
            menu.add_command(label="Add FCS…",
                             command=self._add_samples)
            menu.add_command(label="Paste files (Ctrl+V)",
                             command=self._on_paste, state=paste_state)
            menu.add_command(label="Paste sample(s) from another instance",
                             command=self._paste_samples_transfer,
                             state=transfer_state)
        if parsed and parsed[0] in ('gate', 'sample', 'trial', 'subgroup'):
            try:
                self._add_tab_menu_items(menu)
            except Exception as exc:                  # noqa: BLE001
                print(f"[context menu] tab items: {exc}", flush=True)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        except Exception as exc:
            print(f"[context menu] popup failed: {exc}", flush=True)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass
        return 'break'
