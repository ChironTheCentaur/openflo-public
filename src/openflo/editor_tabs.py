"""Tabs in the Samples & Gates tray (beta).

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
Each tab is a saved view (see ``view_tabs``): its samples and gates, their
on/off states, and the plot's axes, type, colouring, axis types and zoom.
Selecting a tab shows only its members and restores its plot. The strip
lives inside the tray's frame, so it pops out with the tray.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog, ttk

from .editor_base import EditorMixin
from .view_tabs import TabSet, apply_rows


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

    # ── strip ────────────────────────────────────────────────────────────
    def _build_tab_strip(self, parent):
        """The tab row above the tree: one tab per saved view, a + button."""
        row = ttk.Frame(parent)
        self._tab_nb = ttk.Notebook(row, height=1)
        self._tab_nb.pack(side='left', fill='x', expand=True)
        self._tab_nb.bind('<<NotebookTabChanged>>',
                          lambda _e: self._on_tab_changed())
        self._tab_nb.bind('<Button-3>', self._on_tab_strip_menu)
        self._tab_nb.bind('<Double-Button-1>', lambda _e: self._rename_tab())
        add = ttk.Button(row, text="＋", width=3, style='Side.TButton',
                         command=self._new_tab_from_selection)
        add.pack(side='left', padx=(2, 0))
        self._tip(add, "New tab. It takes the selected samples and gates "
                  "(or starts empty) and keeps its own on/off states, axes, "
                  "plot type, scales and zoom. Right-click a tab to rename, "
                  "duplicate or delete it.")
        self._tip(self._tab_nb, "Saved views of this tray. Selecting a tab "
                  "shows only its samples and gates and restores its plot. "
                  "'All' always holds every sample.")
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
        except Exception:                             # noqa: BLE001
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
        except Exception:                             # noqa: BLE001
            pass
        try:
            self._remember_population_view()
        except Exception:                             # noqa: BLE001
            pass

    def _tab_restore(self):
        """Apply the active tab: on/off states, then the plot's view."""
        tab = self._tabs().active
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
        except Exception:                             # noqa: BLE001
            pass
        self._refresh_gate_list()
        sel = tab.selected
        try:
            if sel and self.gate_tv.exists(sel):
                self.gate_tv.selection_set(sel)
                self.gate_tv.see(sel)
        except Exception:                             # noqa: BLE001
            pass
        # The tab's zoom is kept through this and every later replot of
        # its view (see PlotMixin._apply_kept_zoom), not just set once.
        self._keep_zoom(v)
        try:
            self._replot()
        except Exception:                             # noqa: BLE001
            self._schedule_replot(0)

    def _switch_tab(self, idx):
        ts = self._tabs()
        if idx == ts.current or not 0 <= idx < len(ts.tabs):
            return
        self._tab_snapshot()
        ts.current = idx
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
        except Exception:                             # noqa: BLE001
            pass

    # ── tab actions ──────────────────────────────────────────────────────
    def _selected_tab_rows(self):
        """``[(sample, gid|None), ...]`` for the tree selection: sample rows,
        gate rows, and the members of trial / subgroup rows."""
        rows = []
        try:
            sel = self.gate_tv.selection()
        except Exception:                             # noqa: BLE001
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

    def _row_states(self, rows):
        plot = {s: bool(self._sample_plot_enabled.get(s, True))
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
        plot, gates = self._row_states(rows)
        apply_rows([tab], rows, plot, gates)
        tab.view = dict(ts.active.view)
        ts.current = idx
        self._rebuild_tab_strip()
        self._tab_restore()

    def _copy_rows_to_tab(self, idx, rows=None):
        """Copy the selected samples/gates into tab ``idx`` (they keep their
        on/off state there)."""
        rows = self._selected_tab_rows() if rows is None else rows
        if not rows:
            return
        ts = self._tabs()
        plot, gates = self._row_states(rows)
        apply_rows([ts.tabs[idx]], rows, plot, gates)
        self.status_var.set(f"Copied {len(rows)} row(s) to tab "
                            f"'{ts.tabs[idx].name}'.")

    def _copy_rows_to_new_tab(self):
        rows = self._selected_tab_rows()
        ts = self._tabs()
        idx = ts.new_tab()
        self._copy_rows_to_tab(idx, rows)
        self._rebuild_tab_strip()
        self._switch_tab(idx)

    def _apply_rows_to_tabs(self, idxs, rows=None):
        """Push the selected rows' on/off state (adding them where missing)
        to the tabs ``idxs``."""
        rows = self._selected_tab_rows() if rows is None else rows
        if not rows:
            return
        ts = self._tabs()
        plot, gates = self._row_states(rows)
        targets = [ts.tabs[i] for i in idxs
                   if 0 <= i < len(ts.tabs) and i != ts.current]
        apply_rows(targets, rows, plot, gates)
        self.status_var.set(f"Applied {len(rows)} row(s) to "
                            f"{len(targets)} tab(s).")

    def _apply_rows_to_all_tabs(self):
        ts = self._tabs()
        self._apply_rows_to_tabs(range(len(ts.tabs)))

    def _ask_apply_rows_to_tabs(self):
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
            ttk.Checkbutton(body, text=name, variable=v).pack(anchor='w')
            vars_.append((i, v))

        def ok():
            idxs = [i for i, v in vars_ if v.get()]
            dlg.destroy()
            self._apply_rows_to_tabs(idxs, rows)
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(10, 0))
        ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(bb, text="Apply", default='active', command=ok).pack(side='right',
                                                      padx=(0, 6))

    def _population_key(self, sample, gid):
        from .layout_render import SampleContext
        return SampleContext(self, sample).keys.get(gid)

    def _gid_for_key(self, sample, key):
        from .layout_render import SampleContext
        if key is None:
            return None
        return SampleContext(self, sample).by_key.get(key)

    def _apply_rows_within_tab(self, idx, targets, rows=None):
        """Apply the selected rows' on/off state to chosen samples of tab
        ``idx`` only. Gates are matched by population name."""
        from .view_tabs import apply_within
        rows = self._selected_tab_rows() if rows is None else rows
        ts = self._tabs()
        if not rows or not 0 <= idx < len(ts.tabs) or idx == ts.current:
            return 0
        plot, gates = self._row_states(rows)
        n = apply_within(ts.tabs[idx], rows, list(targets), plot, gates,
                         self._population_key, self._gid_for_key)
        self.status_var.set(f"Applied to {len(targets)} sample(s) in tab "
                            f"'{ts.tabs[idx].name}' ({n} change(s)).")
        return n

    def _ask_apply_rows_within_tab(self):
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
        ttk.Label(top, text="Find:").pack(side='left')
        find_var = tk.StringVar()
        ttk.Entry(top, textvariable=find_var, width=20).pack(
            side='left', padx=(4, 0), fill='x', expand=True)
        lf = ttk.Frame(body)
        lf.pack(fill='both', expand=True, pady=(6, 0))
        lb = tk.Listbox(lf, selectmode='extended', height=12, width=48,
                        exportselection=False)
        sb = ttk.Scrollbar(lf, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left', fill='both', expand=True)
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
                    picked, key=self._sample_order.index), rows)
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(10, 0))
        ttk.Button(bb, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(bb, text="Apply", default='active', command=ok).pack(side='right',
                                                      padx=(0, 6))

    def _remove_rows_from_tab(self):
        ts = self._tabs()
        if ts.active.is_all:
            return
        for s, g in self._selected_tab_rows():
            ts.active.remove(s, g)
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _add_tab_menu_items(self, menu):
        """The tab entries of the tree's right-click menu."""
        ts = self._tabs()
        cp = tk.Menu(menu, tearoff=0)
        self._theme_menu(cp)
        for i, t in enumerate(ts.tabs):
            if i == ts.current or t.is_all:
                continue
            cp.add_command(label=t.name,
                           command=lambda i=i: self._copy_rows_to_tab(i))
        cp.add_command(label="New tab…", command=self._copy_rows_to_new_tab)
        ap = tk.Menu(menu, tearoff=0)
        self._theme_menu(ap)
        ap.add_command(label="All tabs", command=self._apply_rows_to_all_tabs)
        ap.add_command(label="Choose tabs…",
                       command=self._ask_apply_rows_to_tabs)
        ap.add_command(label="Within a tab (choose samples)…",
                       command=self._ask_apply_rows_within_tab)
        menu.add_separator()
        menu.add_cascade(label="Copy to tab", menu=cp)
        menu.add_cascade(label="Apply to other tabs", menu=ap,
                         state='normal' if len(ts.tabs) > 1 else 'disabled')
        if not ts.active.is_all:
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
        ts.current = ts.duplicate(ts.current,
                                  samples=[n for n in self._sample_order
                                           if n in self._samples])
        self._rebuild_tab_strip()
        self._tab_restore()

    def _delete_tab(self):
        ts = self._tabs()
        if ts.active.is_all:
            return
        ts.remove(ts.current)
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
        except Exception:                             # noqa: BLE001
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
        except Exception:                             # noqa: BLE001
            pass
        return self._tabs().to_dict()

    def _restore_tabs_state(self, d):
        """Tabs from a session. The session's own view is the active tab's,
        so only membership needs applying here."""
        self._view_tabs = TabSet.from_dict(d) if d else TabSet()
        self._rebuild_tab_strip()
