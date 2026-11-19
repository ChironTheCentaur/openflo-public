"""Channel-name formatting and filterable channel combos.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin
from .ui_logic import (
    filter_choices,
    format_channel,
    resolve_channel,
    resolve_choice,
)


class ChannelsMixin(EditorMixin):
    """Format / resolve channel display names and populate the filterable X / Y / colour channel comboboxes."""

    def _fmt_channel(self, det):
        return format_channel(det, self._channel_labels)

    # ── Setting channel (antibody) labels ──────────────────────────────────

    def _set_channel_labels(self, labels, source, explicit=True):
        """Label channels for this session: ``labels`` is ``{detector:
        label}``. Each label also goes into the session's own labels
        (``_session_channel_labels``), which a reopened session puts back
        over the files' $PnS. One Undo step reverses it (the labels changed
        ride on the gate checkpoint, ``_label_layer``). ``source`` names
        where they came from in the provenance trail.

        ``explicit`` (the default) makes each label one the user set, kept
        in ``_explicit_channel_labels``: an FCS export writes it for that
        channel in every file (_export_channel_labels). It is recorded even
        when the channel already shows that label, so "name it as the
        active file does" reaches the other files too. ``explicit=False``
        (Label channels from vial labels, "Revert to the file's labels")
        takes the channels out of that set again.

        Returns the ``{detector: label}`` that changed, in the view or in
        the explicit set (``_label_set_changes``)."""
        changes = self._label_set_changes(labels, explicit)
        if not changes:
            return {}
        if not self._suspend_undo:
            self._checkpoint()
            top = self._undo_stack[-1] if self._undo_stack else None
            if top is not None and 'barrier' not in top:
                # The labels before the change, on this event's checkpoint
                # (a second change in the same event adds its own channels).
                layer = self._label_layer(changes)
                old = top.get('labels')
                if old is not None:
                    layer['channel'].update(old['channel'])
                    layer['session'] = old['session']
                    layer['explicit'] = old.get('explicit', {})
                top['labels'] = layer
        session = dict(getattr(self, '_session_channel_labels', None) or {})
        session.update(changes)
        self._session_channel_labels = session
        chosen = dict(self._explicit_channel_labels)
        for d, v in changes.items():
            if explicit:
                chosen[d] = v
            else:
                chosen.pop(d, None)
        self._explicit_channel_labels = chosen
        self._channel_labels.update(changes)
        self._audit('channel_labels.set', source=str(source),
                    labels=dict(changes),
                    explicit={d: chosen[d] for d in changes if d in chosen})
        self._refresh_channel_names()
        return changes

    def _export_channel_labels(self, sample, own=None):
        """The ``$PnS`` labels an FCS export of ``sample`` writes, by
        Preferences > Export, "FCS export channel names". ``own`` is what
        that export wrote before the preference (default: the sample's own
        ``channel_labels``); "Each file's original names" returns it as is.

        Session names (the default): for each channel, the label set
        explicitly in this session (``_explicit_channel_labels``: a relabel
        with ``_set_channel_labels``, a template's label, or one of those
        rebuilt from a reopened session's trail), else the file's own label
        (the sample's ``channel_labels``). The view's labels seeded from
        the first file's ``$PnS`` are that file's names, never another
        file's (fcs_export.export_channel_labels)."""
        from .fcs_export import export_channel_labels
        from .prefs import load_fcs_export_channel_names
        file_own = dict(getattr(sample, 'channel_labels', None) or {})
        return export_channel_labels(
            file_own if own is None else own,
            load_fcs_export_channel_names(), file_own=file_own,
            explicit=self._explicit_channel_labels)

    def _label_changes(self, labels):
        """The part of ``labels`` (``{detector: label}``) that differs from
        the channel labels the view shows now, so a dialog or a status note
        can name what a relabel changes on screen before it happens. A
        label the view already shows can still become an explicit session
        name (``_label_set_changes``) without being named here."""
        return {str(d): str(v) for d, v in labels.items()
                if self._channel_labels.get(d, d) != str(v)}

    def _label_set_changes(self, labels, explicit=True):
        """The part of ``labels`` that ``_set_channel_labels`` changes: in
        the view, or in the explicit set (``explicit``: see there)."""
        chosen = self._explicit_channel_labels
        changes = {}
        for d, v in labels.items():
            d, v = str(d), str(v)
            same_view = self._channel_labels.get(d, d) == v
            same_set = (chosen.get(d) == v) if explicit else d not in chosen
            if not (same_view and same_set):
                changes[d] = v
        return changes

    def _label_layer(self, dets):
        """The channel labels of ``dets`` now (None: no label), the
        session's own labels and the explicit ones: what an Undo or Redo of
        a label change puts back (``_restore_label_layer``)."""
        return {'session': dict(getattr(self, '_session_channel_labels', None)
                                or {}),
                'explicit': dict(self._explicit_channel_labels),
                'channel': {d: self._channel_labels.get(d) for d in dets}}

    def _restore_label_layer(self, layer):
        """Put back the labels a ``_label_layer`` recorded, and record it in
        the provenance trail (with which channels are explicit after it),
        so a reopened session rebuilds the explicit labels Undo and Redo
        left (explicit_labels_from_trail)."""
        self._session_channel_labels = dict(layer['session'])
        chosen = dict(layer.get('explicit') or {})
        self._explicit_channel_labels = chosen
        for det, lbl in layer['channel'].items():
            if lbl is None:
                self._channel_labels.pop(det, None)
            else:
                self._channel_labels[det] = lbl
        self._audit('channel_labels.set', source='Undo or Redo',
                    labels={d: (lbl if lbl is not None else d)
                            for d, lbl in layer['channel'].items()},
                    explicit={d: chosen[d] for d in layer['channel']
                              if d in chosen})
        self._refresh_channel_names()

    def _refresh_channel_names(self):
        """Show the current channel labels wherever channels are named: the
        X / Y / colour pickers (keeping the channels chosen), the gate tree
        and the plot."""
        chosen = [(combo, resolve_channel(combo.get()))
                  for combo in (self.x_combo, self.y_combo, self.color_combo)]
        if self._channels:
            self._refresh_channel_choices()
        for combo, det in chosen:
            if det and det in self._channels:
                combo.set(self._fmt_channel(det))
                combo._last_valid = combo.get()
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _stored_transforms(self, name, sample=None):
        """``{channel: transform_spec}`` for the channels sample `name` stores
        off the linear scale -- the space its gates' coordinates are in --
        from the sample's own ``data_transforms`` record."""
        from .sample_prep import stored_transforms
        s = sample if sample is not None else self._samples.get(name)
        return stored_transforms(s)

    def _display_transforms(self, name):
        """``{channel: transform_spec}`` to SAY sample `name`'s gate bounds
        and values with (gate_text.describe_gate / gate_value_text): what its
        stored values carry, so a logicle cut stored at 0.453 reads as the
        intensity 987 that the axis shows there. The sample's own
        ``data_transforms`` record (unknown-scale entries included, so they
        are marked rather than given an invented intensity); for a sample
        without one, the methods the axes assume (``_channel_transform``),
        since that is how it is drawn."""
        rec = getattr(self._samples.get(name), 'data_transforms', None)
        if rec is not None:
            return dict(rec)
        return {ch: {'method': m}
                for ch, m in (self._channel_transform or {}).items()
                if m and m != 'linear'}

    def _gate_text(self, name, gate):
        """A gate of sample `name` as the gate list and status lines show
        it: describe_gate with its bounds in intensity."""
        from .gate_text import describe_gate
        return describe_gate(gate, transforms=self._display_transforms(name))

    def _populate_channel_combos(self):
        display = [self._fmt_channel(c) for c in self._channels]
        self._xy_choices = display
        self._color_choices = ['By sample', 'By density'] + display
        self.x_combo['values']     = display
        self.y_combo['values']     = display
        self.color_combo['values'] = self._color_choices

        # Sensible defaults — FSC-A on X, SSC-A on Y if available
        fsc = next((c for c in self._channels if 'FSC' in c.upper()
                    and '-A' in c.upper()), None)
        if not fsc:
            fsc = next((c for c in self._channels if 'FSC' in c.upper()),
                       self._channels[0])
        ssc = next((c for c in self._channels if 'SSC' in c.upper()
                    and '-A' in c.upper()), None)
        if not ssc:
            ssc = next((c for c in self._channels if 'SSC' in c.upper()),
                       self._channels[min(1, len(self._channels) - 1)])
        self.x_combo.set(self._fmt_channel(fsc))
        self.y_combo.set(self._fmt_channel(ssc))
        self.color_combo.set('By sample')

    def _make_filterable(self, combo, choices_attr):
        """Turn a channel combobox into a type-to-filter one: typing narrows
        the dropdown to matching entries; on commit it snaps to an exact (or
        best) match so an invalid channel can never be selected. `choices_attr`
        names the instance attr holding the full value list."""
        combo.configure(state='normal')
        combo._last_valid = combo.get()

        def _full():
            return list(getattr(self, choices_attr, None) or combo['values'])

        def _on_key(event):
            if event.keysym in ('Up', 'Down', 'Return', 'Escape', 'Tab',
                                'Left', 'Right'):
                return
            combo['values'] = filter_choices(combo.get(), _full())

        def _commit(replot):
            full = _full()
            match = resolve_choice(combo.get(), full,
                                   getattr(combo, '_last_valid', ''))
            changed = bool(match) and match != combo._last_valid
            if match:
                combo.set(match)
                combo._last_valid = match
            combo['values'] = full
            if replot and changed:
                self._on_axis_channel_change()

        combo.bind('<KeyRelease>', _on_key, add='+')
        combo.bind('<FocusOut>', lambda _e: _commit(True), add='+')
        combo.bind('<Return>', lambda _e: (_commit(True), 'break')[1], add='+')
        # Test/automation hooks: the filter + commit logic, callable without
        # synthetic key-event timing (used by tests/test_gui_ux.py).
        combo._filter_type = lambda: _on_key(type('E', (), {'keysym': 'a'})())
        combo._filter_commit = lambda replot=False: _commit(replot)
