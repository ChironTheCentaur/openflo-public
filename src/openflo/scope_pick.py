"""The "Apply to:" scope a window offers, remembered once the user picks it.

Compensation and calibration both offer a scope (this sample or batch, this
batch or the whole tab, ...) that the window sets to a default each time it
redraws, until the user chooses one; after that a redraw keeps the user's
choice. The window keeps ``scope_var`` (a StringVar traced with
``_scope_written``), ``_scope_pick`` (None until the user chooses) and
``_setting_scope`` (False).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:          # no Tk at import: this module is not a GUI one
    import tkinter as tk


class ScopePickMixin:
    """`_scope_written` and `_set_scope` for a window with a scope radio."""

    scope_var: tk.StringVar
    _scope_pick: str | None
    _setting_scope: bool

    def _scope_written(self, *_args):
        """Remember a scope the user chose, so a redraw keeps it."""
        if not self._setting_scope:
            self._scope_pick = self.scope_var.get()

    def _set_scope(self, value):
        """Set the scope as the window's own default, not as a user pick."""
        self._setting_scope = True
        try:
            self.scope_var.set(value)
        finally:
            self._setting_scope = False
