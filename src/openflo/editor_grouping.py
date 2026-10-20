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
            parent=self, title=f'Color for {name}')
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
            from .theme import group_colors
            palette = group_colors()
            c = palette[len(self._sample_colors) % len(palette)]
            self._sample_colors[name] = c
        return c

    def _set_group_palette(self):
        """View → Group colors: remember the palette and recolour the
        samples that wear an automatic colour from another palette, each to
        the same position in the new one. A colour picked by hand is kept."""
        from .prefs import write_pref
        from .theme import GROUP_PALETTES, group_colors
        new = str(self._group_palette_var.get())
        write_pref('group_palette', new)
        target = group_colors(new)
        for other, _lbl in GROUP_PALETTES:
            if other == new:
                continue
            old = [c.lower() for c in group_colors(other)]
            for name, c in list(self._sample_colors.items()):
                lc = str(c).lower()
                if lc in old and lc not in (t.lower() for t in target):
                    self._sample_colors[name] = target[
                        old.index(lc) % len(target)]
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _trial_members(self, trial):
        """Loaded samples belonging to ``trial``, in display order."""
        return [n for n in self._sample_order
                if n in self._samples and self._trial_for(n) == trial]

    def _is_comp(self, name):
        """Whether ``name`` is a compensation control — a manual drag
        override if present, else the name rule (``is_comp_sample``). The one
        classification behind the tree's Comps subgroup, the 'Comp vs
        Samples' grouping and the samples the statistics leave out."""
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

    # ── Setting a sample aside ──────────────────────────────────────────
    def _set_sample_aside(self, names, reason):
        """Leave ``names`` out of every group comparison, with ``reason``
        (kept in the session, the tree and the methods text). The sample
        and its gates stay; only the comparisons skip it."""
        reason = str(reason).strip()
        if not reason:
            return 0
        done = [n for n in names if n in self._samples]
        if not done:
            return 0
        for n in done:
            self._sample_set_aside[n] = reason
        self._audit('sample.set_aside', samples=list(done), reason=reason)
        self._refresh_gate_list()
        self.status_var.set(
            f"Set aside {len(done)} sample(s) from group comparisons: "
            f"{reason}")
        return len(done)

    def _return_sample(self, names):
        """Bring set-aside samples back into the comparisons."""
        done = [n for n in names if self._sample_set_aside.pop(n, None)
                is not None]
        if done:
            self._audit('sample.returned', samples=list(done))
            self._refresh_gate_list()
            self.status_var.set(
                f"Returned {len(done)} sample(s) to group comparisons.")
        return len(done)

    def _set_sample_note(self, name, note):
        """A free-text note on a sample; empty removes it."""
        note = str(note).strip()
        if note:
            self._sample_notes[name] = note
        else:
            self._sample_notes.pop(name, None)
        self._refresh_gate_list()

    def _prompt_set_aside(self, names):
        """Ask for the reason, then set ``names`` aside."""
        from tkinter import simpledialog
        names = [n for n in names if n in self._samples]
        if not names:
            return
        what = names[0] if len(names) == 1 else f"{len(names)} samples"
        reason = simpledialog.askstring(
            "Set aside from comparisons",
            f"Why is {what} set aside? (e.g. failed QC: clogged, too few "
            "events)\n\nIt stays in the session and the statistics table, "
            "but group comparisons leave it out and the methods text lists "
            "it with this reason.",
            initialvalue=self._sample_set_aside.get(names[0], ''),
            parent=self)
        if reason is not None and reason.strip():
            self._set_sample_aside(names, reason)

    def _prompt_sample_note(self, name):
        from tkinter import simpledialog
        note = simpledialog.askstring(
            "Sample note", f"Note for {name} (empty removes it):",
            initialvalue=self._sample_notes.get(name, ''), parent=self)
        if note is not None:
            self._set_sample_note(name, note)

    def _set_aside_items(self, names=None):
        """(name, reason) for each set-aside sample (of ``names``, if
        given), in sample order."""
        want = None if names is None else set(names)
        return [(n, self._sample_set_aside[n]) for n in self._sample_order
                if n in self._sample_set_aside and (want is None or n in want)]

    def _set_aside_lines(self, names=None):
        """'name: reason' for each set-aside sample, for a summary."""
        return [f"{n}: {r}" for n, r in self._set_aside_items(names)]
