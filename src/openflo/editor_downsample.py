"""Display down-sampling controls and propagation.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import tkinter as tk
from contextlib import suppress
from tkinter import messagebox

from .downsample_trim import TRIM_SEED, replay_trim
from .editor_base import EditorMixin
from .paths import is_fcs_file


def trim_entry(editor, name):
    """The trim record a session saves for sample `name`, or None when its
    data is not a propagate trim. The trim is taken over the samples present
    and ticked at the time; recomputing it on reopen from those present and
    ticked THEN would give a sample whose smallest partner had been removed or
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


def _smallest_size(samples, names):
    """Smallest data length among ``names`` in ``samples``, or None."""
    sizes = []
    for n in names:
        try:
            sizes.append(len(samples[n].data))
        except Exception:  # noqa: BLE001 - sample size is only a hint
            continue
    return min(sizes) if sizes else None


def short_cap(cap, uncapped):
    """A Max points cap in a few characters: 'all', '1.2k', '60k', '1.5m'."""
    if cap == uncapped:
        return 'all'
    # The unit is picked after rounding to three digits: 999,600 is '1m',
    # never '1e+03k'.
    if cap >= 999_500:
        m = cap / 1_000_000
        return f"{m:.3g}m" if m < 999.5 else f"{m:.0f}m"
    return f"{cap / 1000:.3g}k"


# Short forms of the downsample modes for the Events dropdown's label.
EVENTS_SHORT_MODES = {'Off': 'Off', 'Display only': 'Display',
                      'Display + data': 'Display+data'}

# Caps whose short forms are the widest the Events button is sized for
# ('999k', '1.23m', '12.3m', '123m', '999m'); a cap past 999m still shows,
# clipped at the end.
WIDEST_LABEL_CAPS = (999_499, 1_234_567, 12_345_678, 123_456_789,
                     999_000_000)


def events_label(mode, cap, overlay):
    """The Events dropdown's label: the downsample mode, the cap (``cap``,
    an int, or None when the text is not one) while downsampling is on
    (60k, 1.2m, all), and ⚠ while the cleaned-out overlay is drawn."""
    from .editor_plot import POINT_CAP_UNCAPPED
    text = f"Events: {EVENTS_SHORT_MODES.get(mode, mode)}"
    if mode != 'Off' and cap is not None:
        text += " · " + short_cap(cap, POINT_CAP_UNCAPPED)
    if overlay:
        text += " ⚠"
    return text


def events_button_chars(font):
    """Width, in the characters a Tk button counts (each the width of a
    '0' in ``font``), that holds the longest Events label in ``font``,
    with one character to spare. Measured rather than fixed: the label
    has symbols and proportional letters, so how many '0's it spans
    depends on the font (DejaVu Sans needs more than Inter)."""
    from .editor_plot import POINT_CAP_UNCAPPED
    zero = max(1, font.measure('0'))
    widest = max(font.measure(events_label('Display + data', cap, True))
                 for cap in (*WIDEST_LABEL_CAPS, POINT_CAP_UNCAPPED))
    return -(-widest // zero) + 1


def widget_font(widget):
    """The font a ttk widget draws its text in: its own ``font`` option
    when set, else its style's, else TkDefaultFont."""
    from tkinter import font as tkfont
    from tkinter import ttk
    name = ''
    with suppress(tk.TclError):
        name = str(widget.cget('font'))
    if not name:
        with suppress(tk.TclError):
            name = str(ttk.Style(widget).lookup(widget.winfo_class(), 'font'))
    name = name or 'TkDefaultFont'
    try:
        return tkfont.nametofont(name, root=widget)
    except tk.TclError:
        return tkfont.Font(root=widget, font=name)


class DownsampleMixin(EditorMixin):
    """Max-points / down-sample mode controls, visibility sync, and propagation of the display cap across samples."""

    def _on_max_points_changed(self):
        """Max-points cap committed (a preset or Custom… in the Events
        dropdown):
        refresh the per-sample event counts in the tree (they show
        shown/total) and replot with the new cap. Text that is not a valid
        cap is refused on the status line and the cap goes back to the last
        valid value, so what the dropdown shows is always the cap in use."""
        from .editor_plot import POINT_CAP_RULE, parse_point_cap
        text = str(self.max_points_var.get()).strip()
        if parse_point_cap(text) is None:
            good = getattr(self, '_mp_last_good', '60000')
            self.max_points_var.set(good)
            self.status_var.set(f"{POINT_CAP_RULE} (it says “{text}”); "
                                f"kept {good}.")
            return
        if text == getattr(self, '_mp_last_good', None):
            return                      # nothing changed (e.g. focus-out)
        self._mp_last_good = text
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _smallest_loaded_sample_size(self):
        """The display floor: smallest in-memory FlowSample.data length among
        the loaded samples the active tab shows (every one on the All tab)
        that are enabled for plotting. A tab draws only its own samples, so
        a small sample in another tab doesn't cap this one. None when no
        samples qualify."""
        return _smallest_size(self._samples, (
            n for n in self._sample_order if n in self._samples
            and self._tab_has_sample(n)
            and self._sample_plot_enabled.get(n, False)))

    def _data_floor_sample(self):
        """``(size, name)`` of the smallest sample the "Display + data" trim
        cuts every sample to, or ``(None, None)``. Sample data is shared by
        every tab, so this floor doesn't depend on the active tab: it is the
        smallest loaded sample enabled for plotting on the All tab (the live
        flags while All is active, else the All tab's kept ones). Re-applied
        after loads and on session restore, it gives the same floor whichever
        tab is active."""
        ts = self._tabs()
        kept = None if ts.active.is_all else ts.tabs[0].plot_enabled
        best = (None, None)
        for n in self._sample_order:
            if n not in self._samples:
                continue
            on = self._sample_plot_enabled.get(n, False)
            if kept is not None:
                on = kept.get(n, on)
            if not on:
                continue
            size = len(self._samples[n].data)
            if best[0] is None or size < best[0]:
                best = (size, n)
        return best

    def _on_downsample_display_toggled(self):
        """Display auto-downsample toggled: replot AND refresh the tree so the
        per-sample event counts reflect the new scaled-down numbers."""
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _on_ds_mode_changed(self):
        """Downsample dropdown → drive the backing booleans. 'Display + data'
        trims FlowSample.data (destructive, via the propagate handler)."""
        mode = self._ds_mode_var.get()
        # Kept so a No to the Display + data question (asked from a tab)
        # puts the mode back as it was.
        self._ds_prev_display = bool(self.ds_display_var.get())
        self.ds_display_var.set(mode != 'Off')
        new_prop = (mode == 'Display + data')
        prop_changed = (self.ds_propagate_var.get() != new_prop)
        self.ds_propagate_var.set(new_prop)
        self._update_ds_visibility()
        self._on_downsample_display_toggled()
        if prop_changed:
            self._on_downsample_propagate_toggled()

    def _update_ds_visibility(self):
        """Max points only makes sense while downsampling is on: its entries
        in the Events dropdown are greyed when Off. Also refreshes the
        dropdown's label."""
        menu = getattr(self, '_events_menu', None)
        if menu is not None:
            on = self.ds_display_var.get() or self.ds_propagate_var.get()
            state = 'normal' if on else 'disabled'
            first = self._events_cap_index()
            try:
                for i in range(first, first + len(self._EVENTS_CAP_PRESETS) + 1):
                    menu.entryconfigure(i, state=state)  # presets + Custom…
            except tk.TclError:
                pass
        self._sync_events_button()

    def _events_cap_index(self):
        """Menu index of the first Max points preset in the Events dropdown:
        after its Downsample heading, the modes and a separator, and the
        Max points heading."""
        return len(self._EVENTS_DS_MODES) + 3

    def _sync_events_button(self):
        """The Events dropdown's label (``events_label``: the downsample
        mode, the cap while downsampling is on, ⚠ while the cleaned-out
        overlay is drawn). The button keeps one width whatever it says,
        the width its font needs for the longest label
        (``events_button_chars``), so no setting widens the control row."""
        mb = getattr(self, '_events_mb', None)
        if mb is None:
            return
        from .editor_plot import parse_point_cap
        text = events_label(self._ds_mode_var.get(),
                            parse_point_cap(self.max_points_var.get()),
                            self.show_removed_var.get())
        try:
            mb.configure(text=text, width=events_button_chars(widget_font(mb)))
        except tk.TclError:
            pass

    def _ask_max_points(self):
        """Events > Custom…: type a Max points cap. It goes through the same
        check as a preset (_on_max_points_changed): a bad entry is refused
        on the status line and the cap in use is kept."""
        from tkinter import simpledialog

        from .editor_plot import POINT_CAP_RULE
        text = simpledialog.askstring(
            "Max points", POINT_CAP_RULE + ".",
            initialvalue=str(self.max_points_var.get()), parent=self)
        if text is None:
            return
        self.max_points_var.set(text.strip())
        self._on_max_points_changed()

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
        smaller sample is ignored (seeded random subsample). From a tab
        other than All it asks first (``_confirm_data_trim_in_tab``). Each sample's
        full frame is kept (``_ds_full_data``) so the trim can be undone.

        Turning OFF: puts back the full frame of every sample still as the
        trim left it; a sample whose data changed since
        (compensation, a transform, clustering) keeps its trimmed data,
        since the kept frame no longer matches it, and is named. A sample
        trimmed before a session was saved and reopened has no kept frame:
        it is rebuilt from its FCS file (``_ds_rebuild_full``), or named
        with the reason it cannot be.
        """
        if not self.ds_propagate_var.get():
            restored, kept, unrestored = self._restore_downsampled()
            # Off for the data only, when the plot still downsamples
            # ('Display only'): saying "Downsampling off" there misleads.
            msg = ("Data downsampling off: samples keep every event and new "
                   "ones load in full; the plot still shows a subsample."
                   if self.ds_display_var.get() else
                   "Downsampling off: new samples load in full.")
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
        floor, smallest = self._data_floor_sample()
        if floor is None or floor <= 0:
            self.status_var.set(
                "Propagate ON — no samples loaded yet; will trim on add.")
            return
        if not self._confirm_data_trim_in_tab(floor, smallest):
            self.ds_propagate_var.set(False)
            self.ds_display_var.set(getattr(self, '_ds_prev_display', True))
            self._sync_ds_mode_var()
            self._update_ds_visibility()
            self._on_downsample_display_toggled()
            self.status_var.set("Display + data not turned on; no sample "
                                "was trimmed.")
            return
        trimmed, floor = self._apply_propagate_downsample()
        self.status_var.set(
            f"Propagate ON — trimmed {trimmed} sample(s) to {floor:,} events.")
        self._schedule_replot(0)

    def _confirm_data_trim_in_tab(self, floor, smallest):
        """Turning "Display + data" on from a tab other than All: ask first,
        because the trim cuts the data of every loaded sample in every tab,
        not only the ones this tab shows (default No). True on the All tab
        or when there is nothing to trim."""
        if self._tabs().active.is_all:
            return True
        cut = [n for n in self._sample_order if n in self._samples
               and len(self._samples[n].data) > floor]
        if not cut:
            return True
        hidden = [n for n in cut if not self._tab_has_sample(n)]
        where = ("" if self._tab_has_sample(smallest)
                 else ", which this tab doesn't show")
        extra = (f"\n{len(hidden)} of them this tab doesn't show."
                 if hidden else "")
        return messagebox.askyesno(
            "Display + data",
            f"This trims the events of {len(cut)} loaded sample(s) in every "
            f"tab, not only this one, to {floor:,} events: the size of "
            f"'{smallest}'{where}.{extra}\nClustering and statistics then "
            "use the trimmed data. Turning it off restores them.\n\n"
            "Trim them?", default=messagebox.NO, parent=self)

    def _apply_propagate_downsample(self):
        """Trim every loaded sample to the smallest ENABLED sample's size
        (``_data_floor_sample``: the same whichever tab is active).

        Returns ``(n_trimmed, floor)``; ``(0, None)`` when there is nothing to
        trim to.

        This is the ONLY place the propagate trim happens, and it is applied to
        the whole set at once. It must not also run per-arrival in
        `_on_loaded`, comparing each incoming sample against the floor of
        whatever had loaded SO FAR: that makes the result depend on the order
        samples happen to finish loading, and that order comes from a thread
        pool.

        Trimming per arrival, measured on four samples of 50k/40k/30k/20k
        events, all enabled:

            arrival a,b,c,d -> 50000, 40000, 30000, 20000   (nothing trimmed)
            arrival d,c,b,a -> 20000, 20000, 20000, 20000   (all trimmed)
            arrival b,d,a,c -> 20000, 40000, 20000, 20000   (neither)

        Only a descending arrival order trims anything, so the same
        session resumed twice could give different event counts — and every
        frequency and gate count computed from them. Taking the floor once,
        over the final set, is order-independent by construction.

        Convergent, so it is safe to call more than once: a smaller floor
        trims further, and the same floor is a no-op.
        """
        floor = self._data_floor_sample()[0]
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
                self._data_changed(name, keep_undo=True,
                                   keep_batch_norm=True)
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
                self._data_changed(name, keep_undo=True,
                                   keep_batch_norm=True)
                trims.pop(name, None)
                restored += 1
                continue
            # Trimmed before the session was saved and reopened from its
            # sidecar: the full frame lived only in memory and the sidecar
            # holds the trimmed one, so it is rebuilt from its FCS file here:
            # left alone it would stay trimmed without a word while raw-FCS
            # samples came back whole.
            if trim_entry(self, name) is None:
                trims.pop(name, None)       # not trimmed any more
                continue
            frame, why = self._ds_rebuild_full(sample, trims[name])
            if frame is None:
                unrestored.append((name, why))
                continue
            sample.data = frame
            self._data_changed(name, keep_undo=True,
                               keep_batch_norm=True)
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
            path if is_fcs_file(path) else None)
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
                'comp': comp,
                'set_number': getattr(sample, 'set_number', None)})
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
