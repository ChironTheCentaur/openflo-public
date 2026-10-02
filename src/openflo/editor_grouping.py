"""Trial / subgroup membership and per-sample colour assignment.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class GroupingMixin(EditorMixin):
    """Resolve trial / subgroup membership and assign stable per-sample and per-gate colours."""

    def _next_color(self):
        """Pick the next colour from flow_pipeline.GATE_PALETTE, cycling."""
        from .pipeline import GATE_PALETTE
        return GATE_PALETTE[self._gate_id_seq % len(GATE_PALETTE)]

    def _pick_sample_color(self, name):
        """Open a colour chooser for a sample and apply it (the sample's plot
        overlay colour and its tree swatch)."""
        from tkinter import colorchooser
        _rgb, hexv = colorchooser.askcolor(
            color=self._sample_colors.get(name) or self._color_for(name),
            parent=self, title=f'Colour for {name}')
        if not hexv:
            return
        self._sample_colors[name] = hexv
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _color_for(self, name):
        """Palette color for a sample, assigned lazily on first display so
        undisplayed samples stay neutral (and don't burn palette slots)."""
        c = self._sample_colors.get(name)
        if c is None:
            idx = len(self._sample_colors) % len(self.SAMPLE_PALETTE)
            c = self.SAMPLE_PALETTE[idx]
            self._sample_colors[name] = c
        return c

    def _trial_members(self, trial):
        """Loaded samples belonging to ``trial``, in display order."""
        return [n for n in self._sample_order
                if n in self._samples and self._trial_for(n) == trial]

    def _is_comp(self, name):
        """Whether ``name`` belongs in the Comps subgroup — a manual drag
        override if present, else the name-based guess."""
        from .workspace import is_comp_sample
        if name in self._sample_is_comp:
            return bool(self._sample_is_comp[name])
        return is_comp_sample(name)

    def _subgroup_members(self, kind, trial):
        """Loaded samples in ``trial`` of the given subgroup ``kind`` —
        'comp' (compensation controls) or 'samp' (everything else)."""
        want_comp = (kind == 'comp')
        return [n for n in self._trial_members(trial)
                if self._is_comp(n) == want_comp]

    def _trial_for(self, name):
        return self._sample_trial.get(name, 'Trial')

    def _ordered_trials(self):
        """Trials that currently have at least one loaded sample, in first-seen
        order (with any stragglers not yet in _trial_order appended)."""
        loaded = [n for n in self._sample_order if n in self._samples]
        trials, seen = [], set()
        for t in self._trial_order:
            if any(self._trial_for(n) == t for n in loaded):
                trials.append(t)
                seen.add(t)
        for n in loaded:
            t = self._trial_for(n)
            if t not in seen:
                trials.append(t)
                seen.add(t)
        # Day-organised groups sort numerically (Day 0 < Day 3 < … < Day 15);
        # any non-day trials keep first-seen order, after the day groups.
        from .workspace import trial_day_number
        idx = {t: i for i, t in enumerate(trials)}
        return sorted(
            trials,
            key=lambda t: (0, trial_day_number(t), 0)
            if trial_day_number(t) is not None else (1, 0, idx[t]))
