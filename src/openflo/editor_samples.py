"""Sample metadata and sets: grouping, the plate view and concatenation
(Complaints 17, 26) — editor mixin.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class SamplesMixin(EditorMixin):
    """Group samples by keyword or table, show them on a plate, and join
    several into one."""

    def _open_group_samples(self):
        """Edit, Group samples: by an FCS keyword or a condition table."""
        if not self._samples:
            self.status_var.set("Load samples first.")
            return
        from .ui_sample_groups import GroupSamplesDialog
        GroupSamplesDialog(self)

    def _apply_sample_groups(self, groups, source=''):
        """Move each sample in ``{sample: group}`` to that group (trial), as
        dragging it there would; groups left empty are dropped."""
        moved = 0
        for name, group in groups.items():
            if name not in self._samples or not group:
                continue
            group = str(group)
            if self._trial_for(name) != group:
                self._sample_trial[name] = group
                moved += 1
            if group not in self._trial_order:
                self._trial_order.append(group)
        self._trial_order = [t for t in self._trial_order
                             if any(self._trial_for(n) == t
                                    for n in self._samples)]
        self._audit('samples.group', source=source, moved=moved,
                    groups=len(set(groups.values())))
        if moved:
            self._note_unsaved()
        self._refresh_gate_list()
        self._schedule_replot(0)
        n = len({str(g) for g in groups.values() if g})
        self.status_var.set(
            f"Grouped {len(groups)} sample(s) into {n} group(s)"
            + (f" by {source}" if source else "") + f"; {moved} moved.")

    def _open_plate_view(self):
        """Analyze, Plate view: a 96-well heatmap of one statistic."""
        if not self._samples:
            self.status_var.set("Load samples first.")
            return
        from .ui_plate import PlateWindow
        PlateWindow(self)

    def _open_concatenate(self):
        """File, Concatenate samples: join several samples into one."""
        if len(self._samples) < 2:
            self.status_var.set("Load two or more samples to concatenate.")
            return
        from .ui_concatenate import ConcatenateDialog
        ConcatenateDialog(self)

    def _add_concatenated_sample(self, sample, sources):
        """Add the joined ``sample`` (ui_concatenate.concatenate_samples) to
        the session, in its first source's group."""
        name = sample.name
        trial = self._trial_for(sources[0]) if sources else None
        self._on_loaded(name, sample)
        if trial:
            self._sample_trial[name] = trial
            if trial not in self._trial_order:
                self._trial_order.append(trial)
        self._audit('samples.concatenate', name=name, sources=list(sources),
                    events=len(sample.data))
        self._note_unsaved()
        self._refresh_gate_list()
        self.status_var.set(
            f"Joined {len(sources)} samples into {name} "
            f"({len(sample.data):,} events). Copy gates to… applies a "
            "sample's gates to it.")
