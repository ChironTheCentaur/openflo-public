"""Tabs in the Samples & Gates tray (beta).

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
Each tab is a saved view (see ``view_tabs``): its samples and its own
gates, their on/off states, and the plot's axes, type, colouring, axis
types and zoom. Selecting a tab makes its gates the live ones, shows only
its members and restores its plot. A gate drawn or edited in one tab stays
in that tab. The strip lives inside the tray's frame, so it pops out with
the tray.
"""
from __future__ import annotations

import copy
import tkinter as tk
from tkinter import simpledialog, ttk

from .editor_base import EditorMixin
from .errors import note
from .tool_window import bind_right_click
from .view_tabs import TabSet, apply_rows, copy_gates, rename_gate_ids


class TabsMixin(EditorMixin):
    """The tray's tab strip, switching, copying rows to tabs and applying
    their on/off state to other tabs."""

    # ── state ────────────────────────────────────────────────────────────
    def _tabs(self) -> TabSet:
        ts = getattr(self, '_view_tabs', None)
        if not isinstance(ts, TabSet):
            ts = TabSet()
            self._view_tabs = ts
        return ts

    def _tab_has_sample(self, name):
        ts = getattr(self, '_view_tabs', None)
        return True if ts is None else ts.active.has_sample(name)

    def _tab_shown_gates(self, name, gates):
        """Gate ids the active tab shows for ``name``, or None for all."""
        ts = getattr(self, '_view_tabs', None)
        if ts is None or ts.active.is_all:
            return None
        return ts.active.shown_gates(name, gates)

    # ── each tab's own gates ─────────────────────────────────────────────
    def _rebind_active_gates(self):
        """Point the active sample's shortcuts at the live gates, after they
        were swapped for another tab's (the id counter is shared by every
        tab, so a new gate's id is never one another tab holds)."""
        a = self._active_sample
        if a is None or a not in self._samples:
            self._gates = {}
            self._gate_id_order = []
            return
        self._gates = self._sample_gates.setdefault(a, {})
        self._gate_id_order = self._sample_gate_order.setdefault(a, [])

    def _activate_tab(self, idx, park=True):
        """Make tab ``idx`` the active one: the live gates go back to the
        tab that had them (unless ``park`` is False: that tab is gone), and
        the new tab's own gates become the live ones. A loaded sample the
        new tab holds no gates for takes a copy of the old live ones."""
        ts = self._tabs()
        old_g, old_o = self._sample_gates, self._sample_gate_order
        if park and ts.active.gates is None:
            ts.active.gates, ts.active.gate_order = old_g, old_o
        ts.current = idx
        tab = ts.active
        gates = tab.gates if tab.gates is not None else {}
        order = tab.gate_order if tab.gate_order is not None else {}
        if gates is not old_g:
            for n in self._samples:
                if n not in gates:
                    gates[n] = copy.deepcopy(old_g.get(n) or {})
                    order[n] = list(old_o.get(n) or gates[n])
        tab.gates = tab.gate_order = None
        self._sample_gates, self._sample_gate_order = gates, order
        self._rebind_active_gates()

    def _copy_gates_to_tab(self, idx, rows):
        """Copy the rows' gates from the live ones into tab ``idx``'s own,
        so they are there when it is shown (no-op for the active tab)."""
        ts = self._tabs()
        tab = ts.tabs[idx]
        if idx == ts.current or tab.gates is None:
            return 0
        self._fill_tab_gates(tab)
        return copy_gates(rows, self._sample_gates, self._sample_gate_order,
                          tab.gates, tab.gate_order, shown=tab.shown_gates)

    def _fill_tab_gates(self, tab):
        """Give an inactive tab a copy of the live gates of each loaded
        sample it holds none for (what it would take when next shown)."""
        if tab.gates is None:
            return
        if tab.gate_order is None:
            tab.gate_order = {}
        for n in self._samples:
            if n not in tab.gates:
                tab.gates[n] = copy.deepcopy(self._sample_gates.get(n) or {})
                tab.gate_order[n] = list(self._sample_gate_order.get(n)
                                         or tab.gates[n])

    def _live_gate_copy(self):
        """A copy of the live gates and their order, for a new tab."""
        return (copy.deepcopy(self._sample_gates),
                copy.deepcopy(self._sample_gate_order))

    def _tabs_sample_arrived(self, name, rescale=None, id_map=None):
        """Sample ``name`` has loaded, its gates in the live (active tab's)
        store. Every other tab gets its own: those a session saved for it,
        else a copy of the live ones (a session saved before tabs kept their
        own gates, a new load or a workspace import). A session restore
        hands out fresh gate ids (``id_map``: saved -> new); each tab's
        members follow its gates."""
        ts = getattr(self, '_view_tabs', None)
        if not isinstance(ts, TabSet):
            return
        live_g = self._sample_gates.get(name) or {}
        live_o = self._sample_gate_order.get(name) or list(live_g)
        active = ts.active
        # One saved -> new id map for the sample across every tab: a gate
        # several tabs saved under one id keeps one id in all of them.
        seen = dict(id_map or {})
        for t in ts.tabs:
            if t is active:
                continue
            if t.gates is None:
                t.gates, t.gate_order = {}, {}
            if t.gate_order is None:
                t.gate_order = {}
            if name in t.saved_gates:
                g, o, m = self._install_tab_gates(
                    name, t.saved_gates.pop(name), rescale, seen)
                t.gates[name], t.gate_order[name] = g, o
                t.remap_gate_ids(name, m)
                for old, new in m.items():
                    seen.setdefault(old, new)
            elif name not in t.gates:
                t.gates[name] = copy.deepcopy(live_g)
                t.gate_order[name] = list(live_o)
                t.remap_gate_ids(name, id_map or {})
        if name in active.saved_gates:
            # Switched to before its sample arrived: its own gates, not the
            # session's (those were the tab active when it was saved).
            g, o, m = self._install_tab_gates(
                name, active.saved_gates.pop(name), rescale, seen)
            self._sample_gates[name], self._sample_gate_order[name] = g, o
            active.remap_gate_ids(name, m)
            self._rebind_active_gates()
        else:
            active.remap_gate_ids(name, id_map or {})

    def _install_tab_gates(self, name, saved, rescale=None, align=None):
        """A tab's saved gates for sample ``name``, put through the same
        restore as the session's own (units, fresh ids, booleans).

        A gate the session's own gates or a tab restored before held under
        its saved id (ids are never reused within a sample, so it is the
        same gate, perhaps edited in this tab) takes the id that one was
        given (``align``: saved -> new), so Copy to tab still knows it for
        the same gate.
        Returns ``(gates, order, saved id -> new id)``."""
        pending = self._session_pending_gates({'gates': list(saved)},
                                              rescale) or []
        live = (self._sample_gates, self._sample_gate_order)
        saved_active = self._active_sample
        prev_suspend = self._suspend_undo
        self._sample_gates, self._sample_gate_order = {name: {}}, {name: []}
        self._suspend_undo = True
        id_map: dict = {}
        try:
            self._set_active_sample(name)
            id_map = self._add_pending_gates(pending) or {}
            gates = self._sample_gates[name]
            order = self._sample_gate_order[name]
        finally:
            self._suspend_undo = prev_suspend
            self._sample_gate_seq[name] = max(
                self._sample_gate_seq.get(name, 0), self._gate_id_seq)
            self._sample_gates, self._sample_gate_order = live
            if saved_active is not None and saved_active in self._samples:
                self._set_active_sample(saved_active)
            else:
                self._active_sample = None
                self._gates, self._gate_id_order = {}, []
                self._gate_id_seq = 0
        align = align or {}
        rename = {new: align[old] for old, new in id_map.items()
                  if old in align}
        gates = rename_gate_ids(gates, order, rename)
        return gates, order, {old: rename.get(new, new)
                              for old, new in id_map.items()}

    # ── strip ────────────────────────────────────────────────────────────
    def _build_tab_strip(self, parent):
        """The tab row above the tree: one tab per saved view, a + button."""
        row = ttk.Frame(parent)
        self._tab_nb = ttk.Notebook(row, height=1)
        self._tab_nb.pack(side='left', fill='x', expand=True)
        self._tab_nb.bind('<<NotebookTabChanged>>',
                          lambda _e: self._on_tab_changed())
        bind_right_click(self._tab_nb, self._on_tab_strip_menu)
        self._tab_nb.bind('<Double-Button-1>', lambda _e: self._rename_tab())
        add = ttk.Button(row, text="＋", width=3, style='Side.TButton',
                         command=self._new_tab_from_selection)
        add.pack(side='left', padx=(2, 0))
        self._tip(add, "New tab. It takes the selected samples and a copy "
                  "of their gates (or starts empty) and keeps its own gates, "
                  "on/off states, axes, plot type, scales and zoom: a gate "
                  "drawn or changed in one tab stays in that tab. Right-click a tab to rename, "
                  "duplicate or delete it.")
        self._tip(self._tab_nb, "Saved views of this tray. Selecting a tab "
                  "shows only its samples and gates and restores its plot. "
                  "'All' always holds every sample. Each tab has its own "
                  "gates; Copy to tab copies gates into another tab.")
        self._rebuild_tab_strip()
        return row

    def _rebuild_tab_strip(self):
        nb = getattr(self, '_tab_nb', None)
        if nb is None:
            return
        ts = self._tabs()
        self._tab_strip_busy = True
        try:
            for t in nb.tabs():
                nb.forget(t)
            for tab in ts.tabs:
                nb.add(ttk.Frame(nb, height=1), text=tab.name)
            nb.select(ts.current)
        finally:
            self._tab_strip_busy = False
        self._sync_tab_remove_button()

    def _on_tab_changed(self):
        if getattr(self, '_tab_strip_busy', False):
            return
        try:
            idx = self._tab_nb.index('current')
        except tk.TclError:
            return
        self._switch_tab(idx)

    # ── switching ────────────────────────────────────────────────────────
    def _tab_snapshot(self):
        """Store the live state into the active tab."""
        ts = self._tabs()
        tab = ts.active
        names = tab.samples(self._sample_order)
        for n in names:
            tab.plot_enabled[n] = bool(self._sample_plot_enabled.get(n, True))
            gates = self._sample_gates.get(n, {}) or {}
            shown = tab.shown_gates(n, gates)
            tab.gate_enabled[n] = {gid: bool(gates[gid].get('enabled', True))
                                   for gid in shown if gid in gates}
        try:
            view = self._current_plot_view()
        except Exception:                             # noqa: BLE001 - tab zoom view is cosmetic
            view = {}
        if view:
            # The tab keeps the user's zoom, not whatever autoscale showed.
            kz = getattr(self, '_kept_zoom', None)
            mine = kz if kz and kz.get('key') == self._zoom_key() else None
            view['xlim'] = mine.get('xlim') if mine else None
            view['ylim'] = mine.get('ylim') if mine else None
            tab.view = view
        try:
            sel = self.gate_tv.selection()
            tab.selected = sel[0] if sel else None
        except tk.TclError:
            pass
        try:
            self._remember_population_view()
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("remember the tab's population view", exc)

    def _tab_restore(self):
        """Apply the active tab: on/off states, then the plot's view."""
        tab = self._tabs().active
        self._sync_tab_remove_button()
        for n, on in tab.plot_enabled.items():
            if n in self._samples:
                self._sample_plot_enabled[n] = bool(on)
        for n, states in tab.gate_enabled.items():
            gates = self._sample_gates.get(n, {}) or {}
            for gid, on in states.items():
                if gid in gates:
                    gates[gid]['enabled'] = bool(on)
        v = tab.view or {}
        try:
            if v.get('x') in self._channels:
                self.x_combo.set(self._fmt_channel(v['x']))
            if v.get('y') in self._channels:
                self.y_combo.set(self._fmt_channel(v['y']))
            if v.get('mode'):
                self.mode_var.set(v['mode'])
            if v.get('color'):
                self.color_combo.set(v['color'])
            for ch, sc in ((v.get('x'), v.get('xscale')),
                           (v.get('y'), v.get('yscale'))):
                if ch and sc:
                    self._channel_scale[ch] = sc
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("restore the tab's plot settings", exc)
        self._refresh_gate_list()
        sel = tab.selected
        try:
            if sel and self.gate_tv.exists(sel):
                self.gate_tv.selection_set(sel)
                self.gate_tv.see(sel)
        except tk.TclError:
            pass
        # The tab's zoom is kept through this and every later replot of
        # its view (see PlotMixin._apply_kept_zoom), not just set once.
        self._keep_zoom(v)
        try:
            self._replot()
        except Exception:                             # noqa: BLE001 - a failed immediate replot falls back to a scheduled one
            self._schedule_replot(0)

    def _switch_tab(self, idx):
        ts = self._tabs()
        if idx == ts.current or not 0 <= idx < len(ts.tabs):
            return
        self._tab_snapshot()
        self._activate_tab(idx)
        # Keep the strip on the same tab, or its pending tab-changed event
        # would switch straight back.
        nb = getattr(self, '_tab_nb', None)
        if nb is not None:
            try:
                if nb.index('current') != idx:
                    self._tab_strip_busy = True
                    try:
                        nb.select(idx)
                    finally:
                        self._tab_strip_busy = False
            except tk.TclError:
                pass
        self._tab_restore()
        try:
            self.status_var.set(
                f"Tab '{ts.active.name}': "
                f"{len(ts.active.samples(self._sample_order))} sample(s).")
        except Exception:                             # noqa: BLE001 - status text is cosmetic
            pass

    # ── tab actions ──────────────────────────────────────────────────────
    def _selected_tab_rows(self):
        """``[(sample, gid|None), ...]`` for the tree selection: sample rows,
        gate rows, and the members of trial / subgroup rows."""
        rows = []
        try:
            sel = self.gate_tv.selection()
        except tk.TclError:                           # tree widget gone, no rows
            sel = ()
        for iid in sel:
            p = self._parse_iid(iid)
            if p is None:
                continue
            if p[0] == 'gate':
                r = [(p[1], p[2])]
            elif p[0] == 'sample':
                r = [(p[1], None)]
            elif p[0] == 'trial':
                r = [(n, None) for n in self._trial_members(p[1])]
            elif p[0] == 'subgroup':
                r = [(n, None) for n in self._subgroup_members(p[1], p[2])]
            else:
                r = []
            rows += [x for x in r if x[0] in self._samples and x not in rows]
        return rows

    def _row_states(self, rows, plot_before=None):
        """``(plot, gates)``: the on/off state of the sample rows and of the
        gate rows in ``rows``. ``plot_before`` (``{sample: on}``, from
        _on_right_click) gives the samples' ticks as they were before the
        right-click selected, and so ticked, the clicked row."""
        before = plot_before or {}
        plot = {s: bool(before.get(s, self._sample_plot_enabled.get(s, True)))
                for s, g in rows if g is None}
        gates = {}
        for s, g in rows:
            if g is not None:
                gd = (self._sample_gates.get(s, {}) or {}).get(g)
                if gd is not None:
                    gates[(s, g)] = bool(gd.get('enabled', True))
        return plot, gates

    def _new_tab_from_selection(self):
        ts = self._tabs()
        self._tab_snapshot()
        rows = self._selected_tab_rows()
        idx = ts.new_tab()
        tab = ts.tabs[idx]
        # Its own copy of the gates, from here on apart from this tab's.
        tab.gates, tab.gate_order = self._live_gate_copy()
        plot, gates = self._row_states(rows)
        apply_rows([tab], rows, plot, gates)
        tab.view = dict(ts.active.view)
        self._activate_tab(idx)
        self._rebuild_tab_strip()
        self._tab_restore()

    def _copy_rows_to_tab(self, idx, rows=None, plot_before=None):
        """Copy the selected samples/gates into tab ``idx`` (they keep their
        on/off state there; ``plot_before`` as in _row_states)."""
        rows = self._selected_tab_rows() if rows is None else rows
        if not rows:
            return
        ts = self._tabs()
        plot, gates = self._row_states(rows, plot_before)
        self._copy_gates_to_tab(idx, rows)
        apply_rows([ts.tabs[idx]], rows, plot, gates)
        self.status_var.set(f"Copied {len(rows)} row(s) to tab "
                            f"'{ts.tabs[idx].name}'.")

    def _copy_rows_to_new_tab(self, plot_before=None):
        rows = self._selected_tab_rows()
        ts = self._tabs()
        idx = ts.new_tab()
        ts.tabs[idx].gates, ts.tabs[idx].gate_order = self._live_gate_copy()
        self._copy_rows_to_tab(idx, rows, plot_before)
        self._rebuild_tab_strip()
        self._switch_tab(idx)

    def _apply_rows_to_tabs(self, idxs, rows=None, plot_before=None):
        """Push the selected rows' on/off state (adding them where missing)
        to the tabs ``idxs`` (``plot_before`` as in _row_states)."""
        rows = self._selected_tab_rows() if rows is None else rows
        if not rows:
            return
        ts = self._tabs()
        plot, gates = self._row_states(rows, plot_before)
        idxs = [i for i in idxs if 0 <= i < len(ts.tabs) and i != ts.current]
        for i in idxs:
            self._copy_gates_to_tab(i, rows)
        targets = [ts.tabs[i] for i in idxs]
        apply_rows(targets, rows, plot, gates)
        self.status_var.set(f"Applied {len(rows)} row(s) to "
                            f"{len(targets)} tab(s).")

    def _apply_rows_to_all_tabs(self, plot_before=None):
        ts = self._tabs()
        self._apply_rows_to_tabs(range(len(ts.tabs)), plot_before=plot_before)

    def _ask_apply_rows_to_tabs(self, plot_before=None):
        """Pick the tabs to apply the selected rows to."""
        ts = self._tabs()
        others = [(i, t.name) for i, t in enumerate(ts.tabs)
                  if i != ts.current]
        if not others:
            return
        rows = self._selected_tab_rows()
        dlg = tk.Toplevel(self)
        dlg.title("Apply to tabs")
        dlg.transient(dlg.master.winfo_toplevel())
        body = ttk.Frame(dlg, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text=f"Apply {len(rows)} selected row(s), with their "
                  "on/off state, to:").pack(anchor='w', pady=(0, 6))
        vars_ = []
        for i, name in others:
            v = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(body, text=name, variable=v)
            cb.pack(anchor='w')
            self._tip(cb, "Tick to copy the selected rows into this tab, "
                      "each switched on or off as it is here. Unticked "
                      "tabs are left alone.")
            vars_.append((i, v))

        def ok():
            idxs = [i for i, v in vars_ if v.get()]
            dlg.destroy()
            self._apply_rows_to_tabs(idxs, rows, plot_before)
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(10, 0))
        ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side='right')
        apply_b = ttk.Button(bb, text="Apply", default='active', command=ok)
        apply_b.pack(side='right', padx=(0, 6))
        self._tip(apply_b, "Add the selected rows to every ticked tab and "
                  "give them this tab's on/off state there. Nothing is "
                  "deleted from those tabs.")

    def _population_key(self, sample, gid):
        from .layout_render import SampleContext
        return SampleContext(self, sample).keys.get(gid)

    def _gid_for_key(self, sample, key, gates=None):
        """``sample``'s gate id for population ``key`` in the live gates, or
        in ``gates`` (``{sample: {gid: gate}}``, a tray tab's own)."""
        from .layout_render import SampleContext
        if key is None:
            return None
        own = None if gates is None else gates.get(sample) or {}
        return SampleContext(self, sample, gates=own).by_key.get(key)

    def _apply_rows_within_tab(self, idx, targets, rows=None,
                               plot_before=None):
        """Apply the selected rows' on/off state to chosen samples of tab
        ``idx`` only. Gates are matched by population name; ``plot_before``
        as in _row_states."""
        from .view_tabs import apply_within
        rows = self._selected_tab_rows() if rows is None else rows
        ts = self._tabs()
        if not rows or not 0 <= idx < len(ts.tabs) or idx == ts.current:
            return 0
        plot, gates = self._row_states(rows, plot_before)
        tab = ts.tabs[idx]
        self._fill_tab_gates(tab)
        # That tab's own gate for each population, not this tab's.
        n = apply_within(tab, rows, list(targets), plot, gates,
                         self._population_key,
                         lambda s, k: self._gid_for_key(s, k, tab.gates))
        self.status_var.set(f"Applied to {len(targets)} sample(s) in tab "
                            f"'{ts.tabs[idx].name}' ({n} change(s)).")
        return n

    def _ask_apply_rows_within_tab(self, plot_before=None):
        """Pick one other tab, then find and tick the samples in it to apply
        the selected rows to (not every sample in that tab)."""
        ts = self._tabs()
        others = [(i, t.name) for i, t in enumerate(ts.tabs)
                  if i != ts.current]
        rows = self._selected_tab_rows()
        if not others or not rows:
            return
        dlg = tk.Toplevel(self)
        dlg.title("Apply within a tab")
        dlg.transient(dlg.master.winfo_toplevel())
        body = ttk.Frame(dlg, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text=f"Apply {len(rows)} selected row(s), with their "
                  "on/off state, to these samples only:").pack(
            anchor='w', pady=(0, 6))
        top = ttk.Frame(body)
        top.pack(fill='x')
        ttk.Label(top, text="Tab:").pack(side='left')
        tab_var = tk.StringVar(value=others[0][1])
        tab_cb = ttk.Combobox(top, textvariable=tab_var, state='readonly',
                              values=[n for _i, n in others], width=24)
        tab_cb.pack(side='left', padx=(4, 10))
        self._tip(tab_cb, "The tab to change. Its samples are listed below; "
                  "choosing another tab clears the samples picked so far.")
        ttk.Label(top, text="Find:").pack(side='left')
        find_var = tk.StringVar()
        find_e = ttk.Entry(top, textvariable=find_var, width=20)
        find_e.pack(side='left', padx=(4, 0), fill='x', expand=True)
        self._tip(find_e, "Type part of a sample name to shorten the list. "
                  "Samples you already picked stay picked while hidden.")
        lf = ttk.Frame(body)
        lf.pack(fill='both', expand=True, pady=(6, 0))
        lb = tk.Listbox(lf, selectmode='extended', height=12, width=48,
                        exportselection=False)
        sb = ttk.Scrollbar(lf, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left', fill='both', expand=True)
        self._tip(lb, "Click the samples to change; Ctrl+click or "
                  "Shift+click picks several. Only these samples in the tab "
                  "get the rows' on/off state.")
        sb.pack(side='right', fill='y')
        shown: list = []
        picked: set = set()

        def tab_idx():
            return next(i for i, n in others if n == tab_var.get())

        def on_select(_e=None):
            # Picks made under another Find filter are kept.
            sel = {shown[i] for i in lb.curselection()}
            for n in shown:
                if n in sel:
                    picked.add(n)
                else:
                    picked.discard(n)
        lb.bind('<<ListboxSelect>>', on_select)

        def fill(*_):
            q = find_var.get().strip().lower()
            names = ts.tabs[tab_idx()].samples(
                [n for n in self._sample_order if n in self._samples])
            shown[:] = [n for n in names if q in n.lower()]
            lb.delete(0, 'end')
            for i, n in enumerate(shown):
                lb.insert('end', n)
                if n in picked:
                    lb.selection_set(i)
        tab_cb.bind('<<ComboboxSelected>>',
                    lambda _e: (picked.clear(), shown.clear(), fill()))
        find_var.trace_add('write', fill)
        fill()

        def ok():
            on_select()
            idx = tab_idx()
            dlg.destroy()
            if picked:
                self._apply_rows_within_tab(idx, sorted(
                    picked, key=self._sample_order.index), rows, plot_before)
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(10, 0))
        ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side='right')
        apply_b = ttk.Button(bb, text="Apply", default='active', command=ok)
        apply_b.pack(side='right', padx=(0, 6))
        self._tip(apply_b, "Switch the matching gates on or off, as the "
                  "selected rows are here, in the picked samples of that "
                  "tab only. Gates are matched by population name.")

    def _remove_rows_from_tab(self):
        """Take the selected rows off the active tab only. The samples stay
        loaded, with their gates, and stay in the All tab and every other tab
        that holds them. (Remove from session is the full removal.)"""
        ts = self._tabs()
        if ts.active.is_all:
            self.status_var.set(
                "The All tab always holds every loaded sample. To take a "
                "sample out of the session, use Remove from session.")
            return
        rows = self._selected_tab_rows()
        if not rows:
            self.status_var.set("Select a sample, trial or gate row to "
                                "remove from this tab.")
            return
        tab = ts.active

        def shown():
            # What the tree shows per selected sample: a listed gate brings
            # its ancestors and subtree, so compare what is shown, not lists.
            return {s: (tab.shown_gates(s, self._sample_gates.get(s) or {})
                        if tab.has_sample(s) else None)
                    for s in {r[0] for r in rows}}
        before = shown()
        for s, g in rows:
            tab.remove(s, g)
        after = shown()
        gone_s = [s for s, v in before.items()
                  if v is not None and after[s] is None]
        gone_g = 0
        for s, was in before.items():
            now = after[s]
            if was is not None and now is not None:
                gone_g += len(was - now)
        parts = ([f"{len(gone_s)} sample(s)"] if gone_s else []) + (
            [f"{gone_g} gate(s)"] if gone_g else [])
        if parts:
            self.status_var.set(
                f"Removed {' and '.join(parts)} from tab '{tab.name}'; "
                f"still loaded (see the All tab).")
        else:
            self.status_var.set(
                f"Nothing removed: tab '{tab.name}' still shows the selected "
                f"gate(s), because it shows all of that sample's gates or a "
                f"gate above or below them. Remove that gate or the sample "
                f"from this tab instead.")
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _sync_tab_remove_button(self):
        """Enable the tray's Remove from tab button only on a tab other than
        All (the All tab always holds every loaded sample)."""
        b = getattr(self, '_tab_remove_btn', None)
        if b is None:
            return
        try:
            b.state(['disabled'] if self._tabs().active.is_all
                    else ['!disabled'])
        except tk.TclError:                           # button gone with the window
            pass

    def _tab_copy_cascade(self, menu, label="Copy to tab"):
        """``label`` ▸ every other tab by name, then New tab…: copy the
        selected rows there."""
        ts = self._tabs()
        before = self._menu_plot_states     # see _on_right_click
        cp = self._ctx_submenu(menu)
        for i, t in enumerate(ts.tabs):
            if i == ts.current or t.is_all:
                continue
            cp.add_command(label=t.name,
                           command=lambda i=i: self._copy_rows_to_tab(
                               i, plot_before=before))
        cp.add_command(label="New tab…", command=lambda: (
            self._copy_rows_to_new_tab(plot_before=before)))
        menu.add_cascade(label=label, menu=cp)

    def _tab_apply_cascade(self, menu, label="Apply to other tabs"):
        """``label`` ▸ All tabs / Choose tabs… / Within a tab…: push the
        selected rows' on/off states to other tabs (greyed with one tab)."""
        ts = self._tabs()
        before = self._menu_plot_states     # see _on_right_click
        ap = self._ctx_submenu(menu)
        ap.add_command(label="All tabs", command=lambda: (
            self._apply_rows_to_all_tabs(plot_before=before)))
        ap.add_command(label="Choose tabs…", command=lambda: (
            self._ask_apply_rows_to_tabs(plot_before=before)))
        ap.add_command(label="Within a tab (choose samples)…",
                       command=lambda: self._ask_apply_rows_within_tab(
                           plot_before=before))
        menu.add_cascade(label=label, menu=ap,
                         state='normal' if len(ts.tabs) > 1 else 'disabled')

    def _tabs_submenu(self, menu):
        """Tabs ▸ Copy to tab ▸ / Apply to other tabs ▸ / Remove from this
        tab: the tab entries of a gate row's right-click menu, in one place."""
        sub = self._ctx_submenu(menu)
        self._tab_copy_cascade(sub)
        self._tab_apply_cascade(sub)
        if not self._tabs().active.is_all:
            sub.add_command(label="Remove from this tab",
                            command=self._remove_rows_from_tab)
        menu.add_cascade(label="Tabs", menu=sub)

    def _add_tab_menu_items(self, menu, with_remove=True):
        """The tab entries of the tree's right-click menu for trial and
        subgroup rows. ``with_remove`` False leaves out Remove from this tab,
        for menus that already list it beside Remove from session."""
        ts = self._tabs()
        menu.add_separator()
        self._tab_copy_cascade(menu)
        self._tab_apply_cascade(menu)
        if with_remove and not ts.active.is_all:
            menu.add_command(label="Remove from this tab",
                             command=self._remove_rows_from_tab)

    def _on_tab_strip_menu(self, event):
        try:
            idx = self._tab_nb.index(f'@{event.x},{event.y}')
        except tk.TclError:
            return
        ts = self._tabs()
        if idx != ts.current:
            self._tab_nb.select(idx)
        m = tk.Menu(self, tearoff=0)
        self._theme_menu(m)
        is_all = ts.tabs[idx].is_all
        m.add_command(label="Rename…", command=self._rename_tab,
                      state='disabled' if is_all else 'normal')
        m.add_command(label="Duplicate", command=self._duplicate_tab)
        m.add_command(label="Delete", command=self._delete_tab,
                      state='disabled' if is_all else 'normal')
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    def _rename_tab(self):
        ts = self._tabs()
        if ts.active.is_all:
            return
        name = simpledialog.askstring("Rename tab", "Tab name:",
                                      initialvalue=ts.active.name,
                                      parent=self)
        if name:
            ts.rename(ts.current, name)
            self._rebuild_tab_strip()

    def _duplicate_tab(self):
        ts = self._tabs()
        self._tab_snapshot()
        dup = ts.duplicate(ts.current,
                           samples=[n for n in self._sample_order
                                    if n in self._samples],
                           live=(self._sample_gates, self._sample_gate_order))
        self._activate_tab(dup)
        self._rebuild_tab_strip()
        self._tab_restore()

    def _delete_tab(self):
        ts = self._tabs()
        if ts.active.is_all:
            return
        # The deleted tab's gates (the live ones) go with it.
        ts.remove(ts.current)
        self._activate_tab(ts.current, park=False)
        self._rebuild_tab_strip()
        self._tab_restore()

    # ── dropping rows onto a tab; the Presort zone ───────────────────────
    def _tab_sample_order(self):
        """The editor's samples in the active tab's own order."""
        ts = getattr(self, '_view_tabs', None)
        if ts is None:
            return list(self._sample_order)
        return ts.active.samples(self._sample_order) + [
            n for n in self._sample_order if not ts.active.has_sample(n)]

    def _tab_at(self, x_root, y_root):
        """Index of the tab under a screen point, or None (geometry, as a
        drag's pointer grab blinds winfo_containing)."""
        nb = getattr(self, '_tab_nb', None)
        if nb is None:
            return None
        try:
            x, y = x_root - nb.winfo_rootx(), y_root - nb.winfo_rooty()
            if not (0 <= x < nb.winfo_width() and 0 <= y < nb.winfo_height()):
                return None
            return int(nb.index(f'@{x},{y}'))
        except (tk.TclError, ValueError):
            return None

    def _drop_rows_on_tab(self, idx, rows):
        """Rows dropped on tab ``idx``: into its Presort zone when the tab
        is arranged (groups or its own order), else to the top of it."""
        ts = self._tabs()
        tab = ts.tabs[idx]
        rows = [r for r in rows if r[0] in self._samples]
        if not rows:
            return None
        grouped = (not tab.is_all
                   and tab.is_grouped(self._trial_for, self._sample_order))
        self._copy_gates_to_tab(idx, rows)
        where = tab.drop(rows, grouped)
        if idx == ts.current:
            self._refresh_gate_list()
            self._schedule_replot(0)
        self.status_var.set(
            f"{len(rows)} row(s) "
            + (f"wait in the Presort zone of '{tab.name}'; drag them into "
               "place there." if where == 'presort'
               else f"added to the top of '{tab.name}'."))
        return where

    def _maybe_drop_to_tab(self, src_iid, event):
        idx = self._tab_at(event.x_root, event.y_root)
        if idx is None:
            return False
        sel = self._press_selection or ()
        srcs = list(sel) if (src_iid in sel and len(sel) > 1) else [src_iid]
        rows = []
        for s in srcs:
            p = self._parse_iid(s) if s else None
            if p is None:
                continue
            if p[0] == 'gate':
                r = [(p[1], p[2])]
            elif p[0] == 'sample':
                r = [(p[1], None)]
            elif p[0] == 'trial':
                r = [(n, None) for n in self._trial_members(p[1])]
            else:
                r = []
            rows += [x for x in r if x not in rows]
        self._drop_rows_on_tab(idx, rows)
        return True

    def _over_tree(self, x_root, y_root):
        tv = getattr(self, 'gate_tv', None)
        try:
            if tv is None or not tv.winfo_ismapped():
                return False
            x, y = tv.winfo_rootx(), tv.winfo_rooty()
            return (x <= x_root < x + tv.winfo_width()
                    and y <= y_root < y + tv.winfo_height())
        except tk.TclError:
            return False

    def _drop_from_workspace(self, rows, x_root, y_root):
        """Pipeline items dragged back to the tray: onto a tab on the strip
        (as a tree drag lands there), or onto the tree, where they join the
        tab shown -- placed before the sample they land on."""
        rows = [r for r in rows if r[0] in self._samples]
        idx = self._tab_at(x_root, y_root)
        if idx is not None:
            if rows:
                self._drop_rows_on_tab(idx, rows)
            return True
        if not self._over_tree(x_root, y_root):
            return False
        if not rows:
            return True
        tab = self._tabs().active
        if tab.is_all:
            self.status_var.set("Those are already in the All tab.")
        else:
            target = self.gate_tv.identify_row(
                y_root - self.gate_tv.winfo_rooty())
            p = self._parse_iid(target) if target else None
            before = p[1] if p and p[0] in ('sample', 'gate') else None
            tab.place(rows, before=before)
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Added {len(rows)} row(s) to tab "
                                f"'{tab.name}'.")
        try:
            from .tree_ids import gate_iid, sample_iid
            s, g = rows[0]
            iid = sample_iid(s) if g is None else gate_iid(s, g)
            if self.gate_tv.exists(iid):
                self.gate_tv.see(iid)
                self.gate_tv.selection_set(iid)
        except Exception:                             # noqa: BLE001 - tree highlight is cosmetic
            pass
        return True

    def _insert_presort_zone(self, tv=None):
        """The active tab's waiting rows, as a zone at the top of the tree
        (gone when empty). It grows with what it holds."""
        ts = getattr(self, '_view_tabs', None)
        if ts is None or not ts.active.presort:
            return
        if tv is None:
            tv = self.gate_tv
        tv.insert('', 0, iid='PRE:zone', open=True, values=('',),
                  text=f'⇣ Presort ({len(ts.active.presort)}) — drag into '
                       'place', tags=('subgroup_row',))
        for i, (s, gid) in enumerate(ts.active.presort):
            if gid is None:
                text = f'■ {s}'
            else:
                from .gating import population_path
                text = (f'{s}  ›  '
                        + population_path(self._sample_gates.get(s, {}), gid))
            tv.insert('PRE:zone', 'end', iid=f'PRE:{i}', text=text,
                      values=('',))

    def _presort_rows(self, iid):
        ts = self._tabs()
        pre = ts.active.presort
        if iid == 'PRE:zone':
            return list(pre)
        try:
            return [pre[int(str(iid)[4:])]]
        except (ValueError, IndexError):
            return []

    def _place_presort_row(self, iid, target_iid):
        """A Presort row dropped in the tree: it joins the tab, its sample
        placed before the sample it was dropped on (or the top). Dropped
        back on the zone, it waits."""
        rows = self._presort_rows(iid)
        if not rows or (target_iid and str(target_iid).startswith('PRE:')):
            return
        p = self._parse_iid(target_iid) if target_iid else None
        before = p[1] if p and p[0] in ('sample', 'gate', 'method') else None
        self._tabs().active.place(rows, before=before,
                                  top=before is None)
        self._refresh_gate_list()
        self._schedule_replot(0)
        self.status_var.set(f"Placed {len(rows)} row(s).")

    def _presort_menu(self, event):
        ts = self._tabs()
        m = tk.Menu(self.gate_tv, tearoff=0)
        self._theme_menu(m)
        m.add_command(label="Place all at the top",
                      command=lambda: (ts.active.place(list(
                          ts.active.presort), top=True),
                          self._refresh_gate_list(),
                          self._schedule_replot(0)))
        m.add_command(label="Clear the Presort zone",
                      command=lambda: (ts.active.presort.clear(),
                                       self._refresh_gate_list()))
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()
        return 'break'

    # ── sessions ─────────────────────────────────────────────────────────
    def _tabs_state(self):
        """The tabs as saved in a session (after storing the live view in
        the active tab)."""
        if getattr(self, '_view_tabs', None) is None:
            return None
        try:
            self._tab_snapshot()
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("snapshot the active tab", exc)
        from . import result_stamp as _stamps
        for t in self._tabs().tabs:
            for gates in (t.gates or {}).values():
                _stamps.settle_gate_stamps(gates)   # hand-edited: the user's
        return self._tabs().to_dict()

    def _restore_tabs_state(self, d):
        """Tabs from a session. The session's own view and gates are the
        active tab's, so only membership needs applying here; every other
        tab's own gates wait for their samples (``_tabs_sample_arrived``)."""
        self._view_tabs = TabSet.from_dict(d) if d else TabSet()
        self._rebuild_tab_strip()
