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
