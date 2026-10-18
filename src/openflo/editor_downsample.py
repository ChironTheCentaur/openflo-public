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
        ENABLED (checked-for-plotting) sample's size — a loaded-but-unchecked
        smaller sample is ignored (seeded random subsample). Each sample's
        full frame is kept (``_ds_full_data``) so the trim can be undone.

        Turning OFF: puts back the full frame of every sample still as the
        trim left it (Complaints 26); a sample whose data changed since
        (compensation, a transform, clustering) keeps its trimmed data,
        since the kept frame no longer matches it, and is named.
        """
        if not self.ds_propagate_var.get():
            restored, kept = self._restore_downsampled()
            msg = "Downsampling off: new samples load in full."
            if restored:
                msg += f" Restored {restored} sample(s) to all their events."
            if kept:
                msg += (f" {', '.join(kept)} changed after the trim and "
                        "stay trimmed (add again to restore).")
            self.status_var.set(msg)
            if restored:
                self._refresh_gate_list()
                self._schedule_replot(0)
            return
        floor = self._smallest_loaded_sample_size()
        if floor is None or floor <= 0:
            self.status_var.set(
                "Propagate ON — no samples loaded yet; will trim on add.")
            return
        trimmed, floor = self._apply_propagate_downsample()
        self.status_var.set(
            f"Propagate ON — trimmed {trimmed} sample(s) to {floor:,} events.")
        self._schedule_replot(0)

    def _apply_propagate_downsample(self):
        """Trim every loaded sample to the smallest ENABLED sample's size.

        Returns ``(n_trimmed, floor)``; ``(0, None)`` when there is nothing to
        trim to.

        This is the ONLY place the propagate trim happens, and it is applied to
        the whole set at once. It used to also run per-arrival in `_on_loaded`,
        comparing each incoming sample against the floor of whatever had
        loaded SO FAR — which made the result depend on the order samples
        happened to finish loading, and that order comes from a thread pool.

        Measured on four samples of 50k/40k/30k/20k events, all enabled:

            arrival a,b,c,d -> 50000, 40000, 30000, 20000   (nothing trimmed)
            arrival d,c,b,a -> 20000, 20000, 20000, 20000   (all trimmed)
            arrival b,d,a,c -> 20000, 40000, 20000, 20000   (neither)

        Only a descending arrival order ever trimmed anything, so the same
        session resumed twice could give different event counts — and every
        frequency and gate count computed from them. Taking the floor once,
        over the final set, is order-independent by construction.

        Convergent, so it is safe to call more than once: a smaller floor
        trims further, and the same floor is a no-op.
        """
        floor = self._smallest_loaded_sample_size()
        if floor is None or floor <= 0:
            return 0, None
        trimmed = 0
        full = self.__dict__.setdefault('_ds_full_data', {})
        trimmed_to = self.__dict__.setdefault('_ds_trimmed_to', {})
        for name in list(self._sample_order):
            sample = self._samples.get(name)
            if sample is None:
                continue
            if len(sample.data) > floor:
                full.setdefault(name, sample.data)
                sample.data = sample.data.sample(
                    floor, random_state=42).reset_index(drop=True)
                self._data_changed(name, keep_undo=True)
                trimmed_to[name] = sample.data
                trimmed += 1
        return trimmed, floor

    def _restore_downsampled(self):
        """Put back each trimmed sample's full frame. Returns ``(n_restored,
        [samples kept trimmed because their data changed since])``."""
        full = self.__dict__.get('_ds_full_data', {})
        trimmed_to = self.__dict__.get('_ds_trimmed_to', {})
        restored, kept = 0, []
        for name in list(trimmed_to):
            sample = self._samples.get(name)
            frame = full.pop(name, None)
            trimmed = trimmed_to.pop(name)
            if sample is None:
                continue
            if frame is None or sample.data is not trimmed:
                kept.append(name)
                continue
            sample.data = frame
            self._data_changed(name, keep_undo=True)
            restored += 1
        full.clear()
        return restored, kept
