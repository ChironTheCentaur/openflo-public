"""Provenance audit log and history window.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin
from .errors import note


class AuditMixin(EditorMixin):
    """Append actions to the provenance audit log and show the history / audit window."""

    def _audit(self, action, **details):
        """Append an operation to the session's audit trail (stamped with the
        wall-clock time and the running OpenFlo version) and live-refresh the
        History window if it's open. Cheap and best-effort — a logging
        failure must never break the operation being logged."""
        try:
            from datetime import datetime

            from .result_stamp import app_version
            if self._audit_log.version is None:
                self._audit_log.version = app_version()
            ts = datetime.now().isoformat(timespec='seconds')
            self._audit_log.record(action, time=ts, details=details)
            win = self._audit_window
            if win is not None and win.winfo_exists():
                win.refresh()
            self._notify_changed()       # Methods follows the trail
        except Exception as exc:  # noqa: BLE001 - noted; an audit-trail failure must never break the audited operation
            note("audit trail: an action was not recorded", exc)

    def _show_audit_window(self):
        """Open (or focus) the provenance / audit-trail viewer."""
        win = self._audit_window
        if win is not None and win.winfo_exists():
            win.refresh()
            win.lift()
            win.focus_set()
            return
        from .ui_audit import AuditWindow
        self._audit_window = AuditWindow(self, self._audit_log)
