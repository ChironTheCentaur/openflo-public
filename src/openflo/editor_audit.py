"""Provenance audit log and history window.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class AuditMixin(EditorMixin):
    """Append actions to the provenance audit log and show the history / audit window."""

    def _audit(self, action, **details):
        """Append an operation to the session's audit trail (stamped with the
        wall-clock time) and live-refresh the History window if it's open.
        Cheap and best-effort — a logging failure must never break the
        operation being logged."""
        try:
            from datetime import datetime
            ts = datetime.now().isoformat(timespec='seconds')
            self._audit_log.record(action, time=ts, details=details)
            win = getattr(self, '_audit_window', None)
            if win is not None and win.winfo_exists():
                win.refresh()
        except Exception as exc:
            print(f"[audit] {type(exc).__name__}: {exc}", flush=True)

    def _show_audit_window(self):
        """Open (or focus) the provenance / audit-trail viewer."""
        win = getattr(self, '_audit_window', None)
        if win is not None and win.winfo_exists():
            win.refresh()
            win.lift()
            win.focus_set()
            return
        from .ui_audit import AuditWindow
        self._audit_window = AuditWindow(self, self._audit_log)
