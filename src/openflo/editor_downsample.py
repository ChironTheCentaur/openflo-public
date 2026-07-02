"""Display down-sampling controls and propagation.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class DownsampleMixin(EditorMixin):
    """Max-points / down-sample mode controls, visibility sync, and propagation of the display cap across samples."""

    def _on_max_points_changed(self):
        """Max-points control edited: refresh the per-sample event counts in
        the tree (they show shown/total) and replot with the new cap."""
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _smallest_loaded_sample_size(self):
        """Smallest in-memory FlowSample.data length across all loaded
        samples that are currently enabled for plotting. None when no
        samples qualify."""
        sizes = []
        for n in self._sample_order:
            if n not in self._samples:
                continue
            if not self._sample_plot_enabled.get(n, False):
                continue
            try:
                sizes.append(len(self._samples[n].data))
            except Exception:
                continue
        if not sizes:
            return None
        return min(sizes)

    def _on_downsample_display_toggled(self):
        """Display auto-downsample toggled: replot AND refresh the tree so the
        per-sample event counts reflect the new scaled-down numbers."""
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _on_ds_mode_changed(self):
        """Downsample dropdown → drive the backing booleans. 'Display + data'
        trims FlowSample.data (destructive, via the propagate handler)."""
        mode = self._ds_mode_var.get()
        self.ds_display_var.set(mode != 'Off')
        new_prop = (mode == 'Display + data')
        prop_changed = (self.ds_propagate_var.get() != new_prop)
        self.ds_propagate_var.set(new_prop)
        self._update_ds_visibility()
        self._on_downsample_display_toggled()
        if prop_changed:
            self._on_downsample_propagate_toggled()

    def _update_ds_visibility(self):
        """Max points only makes sense while downsampling is on — show it for
        Display / Display+data, hide it when Off."""
        lbl = getattr(self, '_mp_label', None)
        combo = getattr(self, '_mp_combo', None)
        if lbl is None or combo is None:
            return
        on = self.ds_display_var.get() or self.ds_propagate_var.get()
        try:
            if on:
                lbl.pack(side='left', padx=(8, 2))
                combo.pack(side='left')
            else:
                lbl.pack_forget()
                combo.pack_forget()
        except Exception:
            pass

    def _sync_ds_mode_var(self):
        """Set the dropdown label from the backing booleans (e.g. after a
        session restore sets them directly)."""
        if not hasattr(self, '_ds_mode_var'):
            return
        if self.ds_propagate_var.get():
            self._ds_mode_var.set('Display + data')
        elif self.ds_display_var.get():
            self._ds_mode_var.set('Display only')
        else:
            self._ds_mode_var.set('Off')

    def _on_downsample_propagate_toggled(self):
        """Propagate toggle handler.

        Turning ON: trims every loaded FlowSample.data to the smallest
        loaded sample's size (seeded random subsample). NOT reversible
        from the GUI — the user must re-add the samples to restore the
        full event count. Surfaces a confirmation in the status bar.

        Turning OFF: no immediate effect on already-trimmed samples
        (we can't restore lost rows), but newly-added samples won't be
        trimmed going forward.
        """
        if not self.ds_propagate_var.get():
            self.status_var.set(
                "Propagate OFF — new samples load full-size. "
                "Already-trimmed samples are not restored (re-add to undo).")
            return
        floor = self._smallest_loaded_sample_size()
        if floor is None or floor <= 0:
            self.status_var.set(
                "Propagate ON — no samples loaded yet; will trim on add.")
            return
        trimmed = 0
        for n in list(self._sample_order):
            s = self._samples.get(n)
            if s is None:
                continue
            if len(s.data) > floor:
                s.data = s.data.sample(floor, random_state=42).reset_index(
                    drop=True)
                trimmed += 1
        self.status_var.set(
            f"Propagate ON — trimmed {trimmed} sample(s) to {floor:,} events.")
        self._schedule_replot(0)
