"""Continuity guard for the persisted ``.flowsession`` save format.

Downstream users keep saved sessions across OpenFlo versions, so the schema is
a contract. These tests fail the moment that contract drifts — making any
change to the save format *intentional, noted, and migration-backed* rather
than a silent break:

* the writer's emitted key set and version must match the locked constants in
  ``openflo.session_format`` (Tk-guarded — it builds a real editor);
* a frozen older-version fixture must still migrate + load;
* the migrator must be idempotent on a current file and refuse a newer one.

If you change the save format on purpose: update ``SESSION_KEYS`` /
``SESSION_VERSION``, add the ``v -> v+1`` migration, note it in CHANGELOG, and
these tests go green again.
"""
from __future__ import annotations

import os

import pytest

from openflo.session_format import (
    SESSION_FORMAT,
    SESSION_KEYS,
    SESSION_VERSION,
    SessionVersionError,
    migrate_session,
)


def _minimal_session(version=SESSION_VERSION, **extra):
    """A schema-complete session dict at ``version`` (values are placeholders;
    the migrator/loader only cares about shape)."""
    base = {k: None for k in SESSION_KEYS}
    base.update({
        'format': SESSION_FORMAT, 'version': version,
        'samples': [], 'sample_gates': {}, 'channel_scale': {},
        'channel_range': {}, 'channel_labels': {}, 'cluster_labels': {},
        'audit': [],
    })
    base.update(extra)
    return base


# ── Migration contract ──────────────────────────────────────────────────────

def test_current_session_is_unchanged_and_no_notes():
    data, notes = migrate_session(_minimal_session())
    assert notes == []                                  # already current
    assert data['version'] == SESSION_VERSION


def test_migration_is_idempotent():
    once, _ = migrate_session(_minimal_session())
    twice, notes = migrate_session(dict(once))
    assert notes == [] and twice['version'] == SESSION_VERSION


def test_newer_session_is_refused():
    with pytest.raises(SessionVersionError):
        migrate_session(_minimal_session(version=SESSION_VERSION + 1))


def test_non_session_is_refused():
    with pytest.raises(SessionVersionError):
        migrate_session({'format': 'something-else', 'version': 1})


def test_old_version_migrates_when_steps_exist():
    """If/when SESSION_VERSION > 1 there MUST be a path from v1 to current; a
    v1 fixture must migrate cleanly. Below v1 there's nothing to do."""
    if SESSION_VERSION == 1:
        pytest.skip("still at schema v1 — no older version to migrate yet")
    data, notes = migrate_session(_minimal_session(version=1))
    assert data['version'] == SESSION_VERSION
    assert notes and len(notes) == SESSION_VERSION - 1


# ── Schema lock (Tk-guarded: builds a real editor) ───────────────────────────

def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                              # noqa: BLE001
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def test_writer_schema_matches_locked_contract():
    """The actual session writer must emit exactly SESSION_KEYS at
    SESSION_VERSION. Drift here means a downstream-visible format change."""
    root, ed = _editor_or_skip()
    try:
        state = ed._session_state()
        emitted = set(state)
        assert emitted == set(SESSION_KEYS), (
            "Session schema drifted from the locked contract.\n"
            f"  added:   {sorted(emitted - set(SESSION_KEYS))}\n"
            f"  removed: {sorted(set(SESSION_KEYS) - emitted)}\n"
            "If intentional: update SESSION_KEYS, bump SESSION_VERSION, add a "
            "migration in _SESSION_MIGRATIONS, and note it in CHANGELOG.")
        assert state['version'] == SESSION_VERSION
        assert state['format'] == SESSION_FORMAT
    finally:
        root.destroy()


def test_round_trip_through_migrator_loads():
    """A written session, passed through the migrator, still applies."""
    root, ed = _editor_or_skip()
    try:
        state = ed._session_state()
        migrated, notes = migrate_session(dict(state))
        assert notes == []                              # fresh write is current
        ed._apply_session(migrated)                     # must not raise
    finally:
        root.destroy()
