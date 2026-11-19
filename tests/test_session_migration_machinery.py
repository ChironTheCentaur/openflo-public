"""The session migration machinery, exercised for the first time.

`migrate_session` is what protects a user's saved work across an upgrade, but
`SESSION_VERSION` is still 1 and `_SESSION_MIGRATIONS` is empty, so the loop
inside it has never executed — `test_session_continuity` honestly SKIPS for
lack of an older version to migrate from. The first real migration would
therefore run untested code on real sessions.

These tests inject migrations to drive the machinery itself, without inventing
a schema version in the shipped code. When a genuine `v1 -> v2` step is added,
these keep working unchanged.
"""
import pytest

from openflo import session_format as SF
from openflo.session_format import (
    SESSION_FORMAT,
    SessionVersionError,
    migrate_session,
)


def _doc(version=1, **extra):
    return {'format': SESSION_FORMAT, 'version': version, **extra}


def _with_schema(monkeypatch, version, migrations):
    monkeypatch.setattr(SF, 'SESSION_VERSION', version)
    monkeypatch.setattr(SF, '_SESSION_MIGRATIONS', migrations)


def test_current_file_is_untouched_and_reports_no_steps():
    data, notes = migrate_session(_doc(version=SF.SESSION_VERSION, a=1))
    assert notes == []
    assert data['a'] == 1
    assert data['version'] == SF.SESSION_VERSION


def test_a_single_step_runs_and_is_reported(monkeypatch):
    def v1_to_v2(d):
        d['added_by_migration'] = True
        return d
    _with_schema(monkeypatch, 2, {1: v1_to_v2})

    data, notes = migrate_session(_doc(version=1))
    assert data['added_by_migration'] is True
    assert data['version'] == 2
    assert notes == ['upgraded session schema v1 → v2']


def test_multi_step_chain_applies_in_order(monkeypatch):
    order = []

    def s1(d):
        order.append(1)
        d['seen'] = [1]
        return d

    def s2(d):
        order.append(2)
        d['seen'] = d.get('seen', []) + [2]
        return d
    _with_schema(monkeypatch, 3, {1: s1, 2: s2})

    data, notes = migrate_session(_doc(version=1))
    assert order == [1, 2], 'steps ran out of order'
    assert data['seen'] == [1, 2]
    assert data['version'] == 3
    assert len(notes) == 2


def test_a_missing_intermediate_step_is_refused(monkeypatch):
    """Silently skipping a gap would hand the app a half-upgraded session."""
    _with_schema(monkeypatch, 3, {1: lambda d: d})     # no 2 -> 3
    with pytest.raises(SessionVersionError, match='No migration registered'):
        migrate_session(_doc(version=1))


def test_a_newer_session_is_refused_not_downgraded():
    with pytest.raises(SessionVersionError, match='newer OpenFlo'):
        migrate_session(_doc(version=SF.SESSION_VERSION + 5))


def test_a_foreign_file_is_refused():
    with pytest.raises(SessionVersionError, match='Not an OpenFlo session'):
        migrate_session({'format': 'something-else', 'version': 1})
    with pytest.raises(SessionVersionError):
        migrate_session({'version': 1})                 # no format at all


@pytest.mark.parametrize('bad', [None, '', 'abc', [], {}])
def test_an_unreadable_version_falls_back_to_v1(monkeypatch, bad):
    """A corrupt version field must not crash the open; v1 is the documented
    floor."""
    def v1_to_v2(d):
        d['migrated'] = True
        return d
    _with_schema(monkeypatch, 2, {1: v1_to_v2})

    data, _notes = migrate_session({'format': SESSION_FORMAT, 'version': bad})
    assert data['migrated'] is True, f'version={bad!r} was not treated as v1'


def test_migration_is_idempotent(monkeypatch):
    """Re-migrating an already-upgraded document must be a no-op — the app
    migrates on manual open AND on autosave resume."""
    calls = []

    def v1_to_v2(d):
        calls.append(1)
        d['n'] = d.get('n', 0) + 1
        return d
    _with_schema(monkeypatch, 2, {1: v1_to_v2})

    data, _ = migrate_session(_doc(version=1))
    again, notes = migrate_session(data)
    assert len(calls) == 1, 'the step ran a second time on a current document'
    assert again['n'] == 1
    assert notes == []


def test_the_document_keeps_its_payload_across_a_migration(monkeypatch):
    """A step that only adds a field must not lose the user's data."""
    _with_schema(monkeypatch, 2,
                 {1: lambda d: {**d, 'schema_note': 'x'}})
    payload = {'samples': [{'name': 's1', 'gates': [{'id': 'g1'}]}],
               'display': {'theme': 'midnight'}}
    data, _ = migrate_session(_doc(version=1, **payload))
    assert data['samples'] == payload['samples']
    assert data['display'] == payload['display']


def test_format_is_restored_if_a_step_drops_it(monkeypatch):
    """migrate_session backfills `format`, so a careless step cannot produce a
    document the next open would reject as foreign."""
    _with_schema(monkeypatch, 2,
                 {1: lambda d: {k: v for k, v in d.items() if k != 'format'}})
    data, _ = migrate_session(_doc(version=1))
    assert data['format'] == SESSION_FORMAT
