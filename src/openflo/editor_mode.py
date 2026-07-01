"""Plot-mode and gate-display option controls.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class ModeMixin(EditorMixin):
    """Switch plot mode (dot/pseudocolor/contour/histogram), sync the mode option widgets, and apply the gate-display mode."""

    def _on_gate_display_changed(self, *_):
        """Display-mode radios (all / highlight / filter). Keep the back-compat
        apply_gates_var in sync (filter ⇒ apply) and replot."""
        self.apply_gates_var.set(self.gate_display_var.get() == 'filter')
        self._schedule_replot(0)

    def _sync_display_mode_availability(self):
        """Grey out Highlight / Filter when there are no real gates to act on,
        and fall back to 'All events' if one was selected — so a stale mode
        can't keep drawing gates that were deleted."""
        radios = getattr(self, '_display_radios', None)
        if not radios:
            return
        has = self._has_real_gates()
        for key in ('highlight', 'filter'):
            r = radios.get(key)
            if r is not None:
                try:
                    r.state(['!disabled'] if has else ['disabled'])
                except Exception:
                    pass
        if not has and self.gate_display_var.get() in ('highlight', 'filter'):
            self.gate_display_var.set('all')
            self.apply_gates_var.set(False)

    def _apply_display_mode(self):
        """Sync apply-gates with the current display mode and replot. Shared by
        the View → Display radios and the Ctrl+1/2/3 shortcuts."""
        self.apply_gates_var.set(self.gate_display_var.get() == 'filter')
        self._schedule_replot(0)

    def _set_display_mode(self, mode):
        """Set a specific display mode (Ctrl+1/2/3) and show it in the status
        bar so the keyboard action is visible."""
        if mode not in self._DISPLAY_MODES:
            return
        try:
            self.gate_display_var.set(mode)
        except Exception:
            return
        self._apply_display_mode()
        self.status_var.set(f"Display: {self._DISPLAY_LABELS[mode]}")

    def _update_quad_set(self, qs_id, new_x=None, new_y=None):
        """Move the shared origin of a 4-rect quadrant set. Each member's
        origin-corner is identified by its `label` (Q++ / Q+- / Q-+ / Q--);
        the corresponding x and/or y bound is rewritten to the new value
        while the outer extent is left alone. Pass new_x and/or new_y;
        the unspecified axis keeps its current origin coord."""
        members = [g for g in self._gates.values()
                   if g.get('quad_set') == qs_id]
        if not members:
            return
        cur_x = float(members[0].get('quad_origin_x', 0.0))
        cur_y = float(members[0].get('quad_origin_y', 0.0))
        nx = cur_x if new_x is None else float(new_x)
        ny = cur_y if new_y is None else float(new_y)
        for g in members:
            label = g.get('label', '') or ''
            # Map quadrant label → which (x, y) corner of the rect is the
            # SHARED origin (the others stay put as the outer extents).
            if   'Q++' in label: g['x0'], g['y0'] = nx, ny
            elif 'Q+-' in label: g['x0'], g['y1'] = nx, ny
            elif 'Q-+' in label: g['x1'], g['y0'] = nx, ny
            elif 'Q--' in label: g['x1'], g['y1'] = nx, ny
            else: continue
            g['quad_origin_x'] = nx
            g['quad_origin_y'] = ny

    def _on_mode_changed(self):
        """Switching plot mode rebuilds the canvas and toggles the
        histogram slider panel — replot first, then sync the slider gate."""
        self._sync_hist_y_combo()
        self._schedule_replot(0)

    def _sync_hist_y_combo(self):
        """Refresh which mode-specific options are shown (also covers session
        restore + programmatic mode changes, which call through here)."""
        self._update_mode_options()

    def _update_mode_options(self):
        """Show only the plot options relevant to the current mode, so the
        control bar stays uncluttered: KDE for pseudocolor, scatter/outliers
        for contour, Hist-Y for histogram."""
        of = getattr(self, '_opt_frame', None)
        if of is None:
            return
        try:
            for w in of.winfo_children():
                w.pack_forget()
            mode = self.mode_var.get()
            if mode == 'pseudocolor':
                self._kde_cb.pack(side='left', padx=(0, 12))
            if mode == 'contour':
                self._cscatter_cb.pack(side='left', padx=(0, 12))
                self._coutliers_cb.pack(side='left', padx=(0, 8))
            if mode == 'histogram':
                self._histy_lbl.pack(side='left', padx=(0, 2))
                self.hist_y_combo.pack(side='left')
        except Exception:
            pass
