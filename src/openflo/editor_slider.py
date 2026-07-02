"""Threshold-slider side panel and per-axis scale/range dialog.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import numpy as np

from .editor_base import EditorMixin


class SliderMixin(EditorMixin):
    """The 1-D threshold slider panel (kind, lo/hi) and the axis scale/range dialog that drives it."""

    def _show_slider_panel(self, show):
        if show:
            self.slider_panel.grid(row=2, column=0, sticky='ew', pady=(4, 0))
        else:
            self.slider_panel.grid_remove()

    def _sync_slider_panel(self):
        """Visibility + range for the slider panel, based on current mode
        and X channel. Called from _replot at the end."""
        self._sync_hist_y_combo()
        mode = self.mode_var.get()
        if mode != 'histogram':
            self._show_slider_panel(False)
            return
        x = self._resolve_channel(self.x_combo.get())
        if not x:
            self._show_slider_panel(False)
            return
        self._show_slider_panel(True)
        # Range: derived from the currently-displayed data's x range.
        try:
            xl, xh = self.ax.get_xlim()
        except Exception:
            xl, xh = 0.0, 1.0
        if not np.isfinite(xl) or not np.isfinite(xh) or xh <= xl:
            xl, xh = 0.0, 1.0
        self.slider_lo.configure(from_=float(xl), to=float(xh))
        self.slider_hi.configure(from_=float(xl), to=float(xh))
        self._slider_axis_label.configure(
            text=f"channel: {self._fmt_channel(x)}")
        # If the X channel changed, seed the slider value(s) from the
        # current gate-on-this-channel (or the axis midpoint).
        if x != self._slider_channel:
            self._slider_channel = x
            self._slider_gate_id = self._find_1d_gate_id_for(x)
            self._seed_sliders_from_gate(xl, xh)
        # UI-only refresh: show/hide the hi slider, update labels. We
        # explicitly DO NOT call _commit_slider_to_gate here — the
        # slider panel should not create gates on its own when the user
        # merely switches mode or channel. Only a user-driven slider
        # drag commits.
        self._update_slider_ui()

    def _find_1d_gate_id_for(self, ch):
        for gid, g in self._gates.items():
            if g.get('channel') == ch and g.get('kind') in ('threshold', 'interval'):
                return gid
        return None

    def _seed_sliders_from_gate(self, xl, xh):
        """Set slider positions from the existing gate, falling back to
        axis-midpoint quartiles when no gate exists yet."""
        self._slider_updating = True
        try:
            g = (self._gates.get(self._slider_gate_id)
                 if self._slider_gate_id else None)
            if g is None:
                mid = (xl + xh) * 0.5
                span = (xh - xl) * 0.25
                self.slider_lo.set(mid - span)
                self.slider_hi.set(mid + span)
            elif g['kind'] == 'threshold':
                self.slider_lo.set(float(g['value']))
                self.slider_hi.set(float(g['value']))
                self.slider_kind_var.set('threshold')
            elif g['kind'] == 'interval':
                self.slider_lo.set(float(g['lo']))
                self.slider_hi.set(float(g['hi']))
                self.slider_kind_var.set('interval')
        finally:
            self._slider_updating = False

    def _update_slider_ui(self):
        """Refresh the slider panel UI (hi slider visibility + labels).
        Touches NO gate state — safe to call when entering histogram mode
        or switching the Threshold/Interval radio. The user has to drag a
        slider to commit a gate; that path runs _commit_slider_to_gate."""
        kind = self.slider_kind_var.get()
        if kind == 'interval':
            self.slider_hi.grid(row=2, column=0, columnspan=3,
                                sticky='ew', padx=(0, 6), pady=(2, 4))
            self.slider_hi_lbl.grid(row=2, column=3, sticky='e', pady=(2, 4))
        else:
            self.slider_hi.grid_remove()
            self.slider_hi_lbl.grid_remove()
        lo = float(self.slider_lo.get())
        hi = float(self.slider_hi.get())
        if kind == 'threshold':
            self.slider_lo_lbl.configure(text=f"{lo:.3g}")
            self.slider_hi_lbl.configure(text='—')
        else:
            if lo > hi:
                lo, hi = hi, lo
            self.slider_lo_lbl.configure(text=f"lo {lo:.3g}")
            self.slider_hi_lbl.configure(text=f"hi {hi:.3g}")

    def _commit_slider_to_gate(self):
        """Build (or update) the 1D gate that matches the current slider
        state. Only the user-drag handlers and the explicit kind-switch
        path (when an existing gate would be silently mis-interpreted)
        should call this — entering histogram mode does NOT."""
        self._update_slider_ui()
        ch = self._slider_channel
        if not ch:
            return
        kind = self.slider_kind_var.get()
        lo = float(self.slider_lo.get())
        hi = float(self.slider_hi.get())
        if kind == 'threshold':
            new_gate = {'kind': 'threshold', 'channel': ch, 'value': lo}
        else:
            if lo > hi:
                lo, hi = hi, lo
            new_gate = {'kind': 'interval', 'channel': ch, 'lo': lo, 'hi': hi}
        # _add_gate replaces the existing 1D gate on this (channel, parent).
        self._slider_gate_id = self._add_gate(new_gate)
        self._refresh_gate_list()
        self._redraw_only_gates()

    def _on_slider_kind_changed(self):
        """Threshold ↔ Interval radio toggle. Updates the slider UI; only
        re-commits to a gate if one already exists for this channel (so
        the user's intent of 'change kind of my gate' takes effect)."""
        self._update_slider_ui()
        if self._slider_gate_id and self._slider_gate_id in self._gates:
            self._commit_slider_to_gate()

    def _on_slider_lo(self, *_):
        if self._slider_updating:
            return
        self._commit_slider_to_gate()
        if self.apply_gates_var.get():
            self._schedule_replot(150)

    def _on_slider_hi(self, *_):
        if self._slider_updating:
            return
        self._commit_slider_to_gate()
        if self.apply_gates_var.get():
            self._schedule_replot(150)

    def _open_axis_dialog(self, axis_letter):
        """Open the AxisConfigDialog for the channel currently bound to
        the X or Y combo. Updates per-channel state + replots on OK."""
        combo = self.x_combo if axis_letter == 'x' else self.y_combo
        other_combo = self.y_combo if axis_letter == 'x' else self.x_combo
        ch = self._resolve_channel(combo.get())
        if not ch:
            self.status_var.set(
                f"Pick a {axis_letter.upper()} channel before configuring its axis.")
            return
        other = self._resolve_channel(other_combo.get())
        from .ui_axis_config import AxisConfigDialog
        AxisConfigDialog(
            self,
            channel=ch,
            scale=self._channel_scale.get(ch, self._default_scale_for(ch)),
            rng=self._channel_range.get(ch),
            show_link=bool(other and other != ch),
            on_apply=lambda s, r, linked: self._set_axis_config(
                ch, s, r, other_channel=(other if linked else None)))

    def _set_axis_config(self, channel, scale, rng, other_channel=None):
        """Persist a channel's scale + range (optionally mirrored to the other
        axis's channel when the dialog's Link X & Y is on) and replot."""
        for chn in (channel, other_channel):
            if not chn:
                continue
            self._channel_scale[chn] = scale
            if rng is None:
                self._channel_range.pop(chn, None)
            else:
                self._channel_range[chn] = (float(rng[0]), float(rng[1]))
        self._schedule_replot(0)
