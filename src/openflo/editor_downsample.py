"""Display down-sampling controls and propagation.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import tkinter as tk

from .editor_base import EditorMixin

# The seed every propagate trim samples with. Recorded with each trim in the
# session (``trim_record_from_entry``), so a reopen replays the trim it was
# saved with even if this ever changes.
TRIM_SEED = 42


def replay_trim(frame, steps, seed=TRIM_SEED):
    """``frame`` cut down as the propagate trim cut it: one seeded subsample
    per step (a later, smaller floor trims the already-trimmed frame again).
    The rows picked depend only on the frame's length, so the same length
    and steps give the same events."""
    for k in steps:
        if len(frame) > k:
            frame = frame.sample(k, random_state=seed).reset_index(drop=True)
    return frame


def trim_record_from_entry(rec):
    """A session entry's ``downsample_trim`` as ``{'from', 'steps', 'seed'}``,
    or None when it is missing or malformed (the sample then loads as it is
    stored, as before the record existed)."""
    if not isinstance(rec, dict):
        return None
    try:
        n_from = int(rec['from'])
        steps = [int(k) for k in rec['steps']]
        seed = int(rec.get('seed', TRIM_SEED))
    except (KeyError, TypeError, ValueError):
        return None
    # Each step cuts the frame left by the one before: strictly smaller.
    if n_from <= 0 or not steps or any(
            k <= 0 or k >= prev
            for prev, k in zip([n_from] + steps[:-1], steps, strict=True)):
        return None
    return {'from': n_from, 'steps': steps, 'seed': seed}


def trim_entry(editor, name):
    """The trim record a session saves for sample `name`, or None when its
    data is not a propagate trim. The trim is taken over the samples present
    and ticked at the time; recomputing it on reopen from those present and
    ticked THEN gave a sample whose smallest partner had been removed or
    unticked all its events back (600 saved, 1,500 reopened). Only while its
    length is still the trimmed one: a sample rebuilt whole since
    (re-compensated from its FCS) is not trimmed any more."""
    rec = (getattr(editor, '_ds_trims', None) or {}).get(name)
    sample = (getattr(editor, '_samples', None) or {}).get(name)
    if rec is None or sample is None:
        return None
    if len(sample.data) != rec['steps'][-1]:
        return None
    return {'from': int(rec['from']),
            'steps': [int(k) for k in rec['steps']],
            'seed': int(rec.get('seed', TRIM_SEED))}


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
            except Exception:  # noqa: BLE001 - sample size is only a hint
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
        except tk.TclError:
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
        since the kept frame no longer matches it, and is named. A sample
        trimmed before a session was saved and reopened has no kept frame:
        it is rebuilt from its FCS file (``_ds_rebuild_full``), or named
        with the reason it cannot be.
        """
        if not self.ds_propagate_var.get():
            restored, kept, unrestored = self._restore_downsampled()
            msg = "Downsampling off: new samples load in full."
            if restored:
                msg += f" Restored {restored} sample(s) to all their events."
            if kept:
                msg += (f" {', '.join(kept)} changed after the trim and "
                        "stay trimmed (add again to restore).")
            for name, why in unrestored:
                msg += (f" {name} stays trimmed to "
                        f"{len(self._samples[name].data):,} events: {why}.")
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
        trims = self.__dict__.setdefault('_ds_trims', {})
        for name in list(self._sample_order):
            sample = self._samples.get(name)
            if sample is None:
                continue
            if len(sample.data) > floor:
                # The trim's record, which the session saves (see
                # `trim_entry`): the length it started from and each
                # floor it was cut to. A further cut of a trimmed sample is
                # another step on the same record; a sample whose length no
                # longer matches its record (rebuilt whole) starts anew.
                rec = trims.get(name)
                if rec is None or rec['steps'][-1] != len(sample.data):
                    rec = {'from': len(sample.data), 'steps': [],
                           'seed': TRIM_SEED}
                rec['steps'].append(int(floor))
                trims[name] = rec
                full.setdefault(name, sample.data)
                sample.data = replay_trim(sample.data, [floor])
                self._data_changed(name, keep_undo=True)
                trimmed_to[name] = sample.data
                trimmed += 1
        return trimmed, floor

    def _ds_restore_trim(self, name, sample, rec):
        """A session sample arriving with its saved trim record `rec`
        (``trim_record_from_entry``). One reloaded from its raw FCS holds all
        its events: the saved trim is replayed on it, so it holds the SAME
        events it was saved with, and its full frame is kept so turning the
        trim off restores it as before the save. One from its processed
        sidecar is already trimmed: only the record is kept (turning the
        trim off rebuilds it from its FCS). Returns a note for the user when
        the loaded frame matches neither, else ''."""
        n = len(sample.data)
        trims = self.__dict__.setdefault('_ds_trims', {})
        if n == rec['from']:
            full = sample.data
            sample.data = replay_trim(full, rec['steps'], rec['seed'])
            self.__dict__.setdefault('_ds_full_data', {})[name] = full
            self.__dict__.setdefault('_ds_trimmed_to', {})[name] = sample.data
            trims[name] = rec
            return ''
        if n == rec['steps'][-1]:
            trims[name] = rec
            return ''
        return (f"{name}: its saved downsample (to {rec['steps'][-1]:,} of "
                f"{rec['from']:,} events) was not applied, because it loaded "
                f"with {n:,} events; it holds all of them")

    def _restore_downsampled(self):
        """Put back each trimmed sample's full frame. Returns ``(n_restored,
        [samples kept trimmed because their data changed since],
        [(sample, why) trimmed before a reopen that could not be rebuilt
        from their FCS])``."""
        full = self.__dict__.get('_ds_full_data', {})
        trimmed_to = self.__dict__.get('_ds_trimmed_to', {})
        trims = self.__dict__.setdefault('_ds_trims', {})
        restored, kept, unrestored = 0, [], []
        for name in list(dict.fromkeys(list(trimmed_to) + list(trims))):
            sample = self._samples.get(name)
            frame = full.pop(name, None)
            in_memory = name in trimmed_to
            trimmed = trimmed_to.pop(name, None)
            if sample is None:
                trims.pop(name, None)
                continue
            if in_memory:
                if frame is None or sample.data is not trimmed:
                    kept.append(name)
                    continue
                sample.data = frame
                self._data_changed(name, keep_undo=True)
                trims.pop(name, None)
                restored += 1
                continue
            # Trimmed before the session was saved and reopened from its
            # sidecar: the full frame lived only in memory and the sidecar
            # holds the trimmed one, so turning the trim off used to leave
            # it trimmed without a word while raw-FCS samples came back whole.
            if trim_entry(self, name) is None:
                trims.pop(name, None)       # not trimmed any more
                continue
            frame, why = self._ds_rebuild_full(sample, trims[name])
            if frame is None:
                unrestored.append((name, why))
                continue
            sample.data = frame
            self._data_changed(name, keep_undo=True)
            trims.pop(name, None)
            restored += 1
        full.clear()
        return restored, kept, unrestored

    def _ds_rebuild_full(self, sample, rec):
        """All of a trimmed sample's events, rebuilt from its FCS file as the
        loader prepares it (QC as it ran, its compensation matrix) and put on
        the transforms its data carries. Returns ``(frame, '')``, or
        ``(None, why)`` when that cannot give back the sample it was.

        Only the FCS's own channels (and derived channels, recomputed from
        their formulas) can be rebuilt. A column computed on the trimmed
        events (clusters, an embedding) has values for the kept events only;
        blanks for the rest would plot and gate as a population of their own,
        so such a sample stays trimmed and is named. The rebuilt frame must
        also give back the sample's current values at the trim's positions;
        when it does not (its data changed since), it stays trimmed rather
        than swapping in other values."""
        import numpy as np
        import pandas as pd

        path = str(getattr(sample, 'path', '') or '')
        fcs = getattr(sample, 'source_fcs', None) or (
            path if path.lower().endswith('.fcs') else None)
        if not fcs or not os.path.isfile(fcs):
            return None, "its FCS file was not found to rebuild it from"
        comp = None
        if (getattr(sample, 'comp_channels', None)
                and getattr(sample, 'comp_matrix', None) is not None):
            comp = {'matrix': np.asarray(sample.comp_matrix, dtype=float),
                    'channels': list(sample.comp_channels),
                    'source': getattr(sample, 'compensation_source', None)}
        try:
            fresh = self._prepared_fcs_sample(fcs, {
                'qc': bool(getattr(sample, 'qc_applied', False)),
                'comp': comp})
            saved = getattr(sample, 'data_transforms', None)
            if saved is not None:
                self._restore_sample_transforms(fresh, dict(saved))
            if getattr(sample, 'derived', None):
                from .derived import restore as _restore_derived
                _restore_derived(fresh, dict(sample.derived))
        except Exception as exc:          # noqa: BLE001 - told to the user
            return None, ("rebuilding it from its FCS file failed "
                          f"({type(exc).__name__})")
        if len(fresh.data) != rec['from']:
            return None, (f"its FCS file gives {len(fresh.data):,} events, "
                          f"not the {rec['from']:,} it was trimmed from")
        cols = list(sample.data.columns)
        missing = [str(c) for c in cols if c not in fresh.data.columns]
        if missing:
            return None, (f"{', '.join(missing)} "
                          f"{'was' if len(missing) == 1 else 'were'} computed "
                          "on the kept events only (add the sample again "
                          "and re-run on all its events)")
        pos = replay_trim(pd.DataFrame({'pos': np.arange(rec['from'])}),
                          rec['steps'], rec['seed'])['pos'].to_numpy()
        kept = fresh.data[cols].iloc[pos]
        for c in cols:
            a = np.asarray(pd.to_numeric(kept[c], errors='coerce'),
                           dtype=float)
            b = np.asarray(pd.to_numeric(sample.data[c], errors='coerce'),
                           dtype=float)
            if not np.allclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True):
                return None, ("its values changed after the trim, so its "
                              "FCS file no longer rebuilds them")
        return fresh.data[cols].reset_index(drop=True), ''
