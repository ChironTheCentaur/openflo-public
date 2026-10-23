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

    def _stored_transforms(self, name, sample=None):
        """``{channel: transform_spec}`` for the channels sample `name` stores
        off the linear scale -- the space its gates' coordinates are in --
        from the sample's own ``data_transforms`` record."""
        s = sample if sample is not None else self._samples.get(name)
        rec = getattr(s, 'data_transforms', None) or {}
        return {str(ch): dict(spec) for ch, spec in rec.items()
                if spec and spec.get('method', 'linear') != 'linear'}

    def _display_transforms(self, name):
        """``{channel: transform_spec}`` to SAY sample `name`'s gate bounds
        and values with (pipeline.describe_gate / gate_value_text): what its
        stored values carry, so a logicle cut stored at 0.453 reads as the
        intensity 987 that the axis shows there. The sample's own
        ``data_transforms`` record (unknown-scale entries included, so they
        are marked rather than given a made-up intensity); for a sample
        without one, the methods the axes assume (``_channel_transform``),
        since that is how it is drawn."""
        rec = getattr(self._samples.get(name), 'data_transforms', None)
        if rec is not None:
            return dict(rec)
        return {ch: {'method': m}
                for ch, m in (getattr(self, '_channel_transform', None)
                              or {}).items()
                if m and m != 'linear'}

    def _gate_text(self, name, gate):
        """A gate of sample `name` as the gate list and status lines show
        it: describe_gate with its bounds in intensity."""
        from .pipeline import describe_gate
        return describe_gate(gate, transforms=self._display_transforms(name))

    def _resolve_channel(self, display):
        return resolve_channel(display)

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
