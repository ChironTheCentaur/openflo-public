"""Versioning + migration for the ``.flowsession`` save format.

Kept deliberately Tk-free so the GUI, a standalone upgrade script
(``scripts/migrate_session.py``), and the test suite can all import it without
a display.

**Continuity contract.** ``SESSION_KEYS`` and ``SESSION_VERSION`` pin the
schema that :meth:`openflo.gui.ViewGateEditorWindow._session_state` emits. The
continuity test (``tests/test_session_continuity.py``) fails the moment the
writer's key set or version drifts from these constants — so a schema change
can never be accidental. When you *do* change the schema on purpose:

1. add/remove the key in ``_session_state`` and update ``SESSION_KEYS``;
2. bump ``SESSION_VERSION``;
3. register a ``v -> v+1`` step in ``_SESSION_MIGRATIONS`` so old saved
   sessions keep loading (downstream users are not stranded);
4. note it in ``CHANGELOG.md`` under the new version.

The loader auto-upgrades older files on open and tells the user; the standalone
script upgrades files in place for batch/headless use.
"""
from __future__ import annotations

from collections.abc import Callable

SESSION_FORMAT = 'openflo-session'
SESSION_VERSION = 1

# Every top-level key a CURRENT session carries. Locked by the continuity test.
SESSION_KEYS = frozenset({
    'format', 'version', 'created', 'active_sample', 'samples', 'sample_gates',
    'channel_scale', 'channel_range', 'channel_labels', 'plot_mode',
    'x_channel', 'y_channel', 'color_channel', 'downsample_display',
    'downsample_propagate', 'max_points', 'show_removed', 'contour_scatter',
    'contour_outliers', 'hist_y_mode', 'cluster_labels', 'audit',
})


class SessionVersionError(ValueError):
    """Raised when a session can't be migrated (e.g. written by a newer
    OpenFlo, or a missing migration step)."""


# Ordered upgrade steps: ``_SESSION_MIGRATIONS[v]`` lifts a v-schema dict to
# v+1. Empty today (we're at v1); the framework forces every future bump to
# ship its migration. Each step must be pure and idempotent on its own version.
_SESSION_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def migrate_session(data: dict) -> tuple[dict, list[str]]:
    """Upgrade a parsed ``.flowsession`` dict to ``SESSION_VERSION`` in place.

    Returns ``(data, notes)`` where ``notes`` lists the steps applied (empty if
    the file was already current). Raises :class:`SessionVersionError` if the
    file is newer than this build, or an intermediate migration is missing.
    """
    if data.get('format') != SESSION_FORMAT:
        raise SessionVersionError(
            f"Not an OpenFlo session (format={data.get('format')!r}).")
    notes: list[str] = []
    try:
        v = int(data.get('version', 1) or 1)
    except (TypeError, ValueError):
        v = 1
    if v > SESSION_VERSION:
        raise SessionVersionError(
            f"This session was written by a newer OpenFlo "
            f"(schema v{v}; this build supports v{SESSION_VERSION}). "
            "Update OpenFlo to open it.")
    while v < SESSION_VERSION:
        step = _SESSION_MIGRATIONS.get(v)
        if step is None:
            raise SessionVersionError(
                f"No migration registered from session schema v{v} "
                f"to v{v + 1}.")
        data = step(data)
        notes.append(f"upgraded session schema v{v} → v{v + 1}")
        v += 1
    data['version'] = SESSION_VERSION
    data.setdefault('format', SESSION_FORMAT)
    return data, notes
