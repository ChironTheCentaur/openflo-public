"""Specialist analysis windows: proliferation, the compensation spreading
matrix, the panel check and bead tracking (Complaints 21, 24, 25) — editor mixin.

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

    def _open_panel_check(self):
        """Tools, Panel check: predicted similarity of a planned panel."""
        from .ui_panel_check import PanelCheckWindow
        PanelCheckWindow(self)

    def _open_unmixed_pairs(self):
        """Spectral QC, Unmixed pairs: every pair of the last unmix."""
        pairs = getattr(self, '_last_unmix_pairs', None)
        if not pairs:
            self.status_var.set("Run Unmix first: no unmixed sample yet.")
            return
        from .ui_unmixed_pairs import UnmixedPairsWindow
        name, A, fluors = pairs
        UnmixedPairsWindow(self, A, fluors, sample=name)

    # Mass cytometry (ui_mass_cytometry).
    def _open_bead_normalize(self):
        """Tools, Mass cytometry, Bead normalization (Finck et al. 2013)."""
        if not self._need_active():
            return
        from .ui_mass_cytometry import BeadNormalizeWindow
        BeadNormalizeWindow(self)

    def _restore_bead_norm(self):
        """Tools, Mass cytometry, Undo bead normalization: every normalized
        sample back on its values from before."""
        from .ui_mass_cytometry import restore_bead_normalization
        done = restore_bead_normalization(self)
        self.status_var.set(
            f"Bead normalization undone: {len(done)} sample(s) back on their "
            "values from before." if done else
            "No bead normalization to undo.")
        return done

    def _open_gaussian_cleanup(self):
        """Tools, Mass cytometry, Gaussian parameter clean-up."""
        if not self._need_active():
            return
        from .ui_mass_cytometry import GaussianCleanupWindow
        GaussianCleanupWindow(self)
