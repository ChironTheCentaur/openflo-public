"""Record a written export in the editor's audit trail.

One helper for every results window that writes a file, so the trail entry
(and ``fixes.stale_exports``, which reads its ``path`` and ``samples``) is
made the same way everywhere. Tk-free.
"""
from __future__ import annotations

from .errors import note


def record_export(editor, action, path, **details):
    """Append ``action`` with ``path`` and ``details`` to ``editor``'s audit
    trail. Without an editor, or one with no audit trail, it does nothing. A
    failure to record is noted in the log and never breaks the export."""
    audit = getattr(editor, '_audit', None)
    if audit is None:
        return
    try:
        audit(action, path=path, **details)
    except Exception as exc:  # noqa: BLE001 - noted; recording must never break the export
        note("recording the export in the audit trail", exc)
