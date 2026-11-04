"""Plot-mode and gate-display option controls.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk
import warnings

from .editor_base import EditorMixin


class _ApplyGatesShim:
    """Deprecated stand-in for the old ``apply_gates_var`` BooleanVar. It holds
    no state of its own: it reads and writes ``gate_display_var``."""

    def __init__(self, editor):
        self._editor = editor

    @staticmethod
    def _warn():
        warnings.warn('apply_gates_var is deprecated; read '
                      '_filters_to_gated() or set gate_display_var instead. '
                      'It will be removed in a future release',
                      DeprecationWarning, stacklevel=3)

    def get(self):
        self._warn()
        return self._editor._filters_to_gated()

    def set(self, value):
        self._warn()
        display = self._editor.gate_display_var
        if value:
            display.set('filter')
        elif display.get() == 'filter':
            display.set('all')


class ModeMixin(EditorMixin):
    """Switch plot mode (dot/pseudocolor/contour/histogram), sync the mode option widgets, and apply the gate-display mode."""

    def _filters_to_gated(self) -> bool:
        """True when the display mode is 'Filter to gated', so plots show only
        events inside the enabled gates."""
        return self.gate_display_var.get() == 'filter'

    @property
    def apply_gates_var(self):
        """Deprecated: use :meth:`_filters_to_gated` (read) or
        ``gate_display_var`` (write)."""
        return _ApplyGatesShim(self)

    def _on_gate_display_changed(self, *_):
        """Display-mode radios (all / highlight / filter): replot."""
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
                except tk.TclError:
                    pass
        if not has and self.gate_display_var.get() in ('highlight', 'filter'):
            self.gate_display_var.set('all')

    def _apply_display_mode(self):
        """Replot for the current display mode. Shared by the View → Display
        radios and the Ctrl+1/2/3 shortcuts."""
        self._schedule_replot(0)

    def _set_display_mode(self, mode):
        """Set a specific display mode (Ctrl+1/2/3) and show it in the status
        bar so the keyboard action is visible."""
        if mode not in self._DISPLAY_MODES:
            return
        try:
            self.gate_display_var.set(mode)
        except tk.TclError:
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
        histogram slider panel — sync the mode-specific options first, then
        schedule a replot."""
        self._update_mode_options()
        self._schedule_replot(0)

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
        except tk.TclError:
            pass
        self._sync_color_combo()

    def _sync_color_combo(self):
        """Only dot mode reads Color: grey the combo out in the other modes,
        so it never shows a setting the plot isn't using (its hover tip says
        why). The value is kept for when dot mode comes back."""
        combo = getattr(self, 'color_combo', None)
        if combo is None:
            return
        try:
            combo.state(['!disabled'] if self.mode_var.get() == 'dot'
                        else ['disabled'])
        except tk.TclError:
            pass

    _COLOR_OFF_WHY = {
        'pseudocolor': "pseudocolor always colors by event density",
        'contour': "contour draws each sample in its own color",
        'histogram': "histogram draws each sample in its own color",
    }

    def _color_combo_tip(self):
        """Hover text for the Color combo, saying why it is greyed out in the
        modes that don't use it."""
        base = ("Color the dots by density, by sample, or by a channel / "
                "cluster column.")
        why = self._COLOR_OFF_WHY.get(self.mode_var.get())
        if why is None:
            return base
        return (f"Not used in this mode: {why}. Switch Mode to dot to color "
                "by density, by sample, or by a channel / cluster column.")
