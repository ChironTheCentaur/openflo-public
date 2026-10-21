"""Specialist analysis windows: proliferation, the compensation spreading
matrix and bead tracking (Complaints 21, 24, 25) — editor mixin.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class SpecialistMixin(EditorMixin):
    """Open the specialist analysis windows on the active sample."""

    def _need_active(self):
        if self._active_sample not in self._samples:
            self.status_var.set("Load and select a sample first.")
            return False
        return True

    def _open_proliferation(self):
        """Analyze, Proliferation: dye-dilution generations."""
        if not self._need_active():
            return
        from .ui_proliferation import ProliferationWindow
        ProliferationWindow(self)

    def _open_spread_matrix(self):
        """Tools, Spreading matrix: spread of conventional compensation."""
        if not self._need_active():
            return
        from .ui_comp_spread import SpreadMatrixWindow
        SpreadMatrixWindow(self)

    def _open_bead_tracking(self):
        """Tools, Bead tracking: Levey-Jennings chart of QC bead runs."""
        if not self._need_active():
            return
        from .ui_bead_tracking import BeadTrackingWindow
        BeadTrackingWindow(self)

    def _open_unmixed_pairs(self):
        """Spectral QC, Unmixed pairs: every pair of the last unmix."""
        pairs = getattr(self, '_last_unmix_pairs', None)
        if not pairs:
            self.status_var.set("Run Unmix first: no unmixed sample yet.")
            return
        from .ui_unmixed_pairs import UnmixedPairsWindow
        name, A, fluors = pairs
        UnmixedPairsWindow(self, A, fluors, sample=name)
