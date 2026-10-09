"""Tests for openflo.prefs — the read-modify-write must not lose data on a
transient read failure (the recurring Windows AV/indexer file-lock case)."""
from __future__ import annotations

import builtins
import json

import openflo.prefs as prefs


def _point_prefs_at(tmp_path, monkeypatch):
    p = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    return p


def test_write_pref_preserves_other_keys(tmp_path, monkeypatch):
    p = _point_prefs_at(tmp_path, monkeypatch)
    p.write_text(json.dumps({'a': 1, 'b': 2, 'theme': 'light'}), encoding='utf-8')
    prefs.write_pref('theme', 'dark')
    assert json.loads(p.read_text(encoding='utf-8')) == {
        'a': 1, 'b': 2, 'theme': 'dark'}


def test_write_pref_aborts_on_transient_read_error(tmp_path, monkeypatch):
    # A locked/permission-denied read (OSError) must NOT wipe the existing prefs
    # down to the single key being written.
    p = _point_prefs_at(tmp_path, monkeypatch)
    original = {'a': 1, 'b': 2, 'theme': 'light'}
    p.write_text(json.dumps(original), encoding='utf-8')

    real_open = builtins.open

    def locked_open(f, *a, **k):
        mode = a[0] if a else k.get('mode', 'r')
        if str(f) == str(p) and 'r' in mode and 'w' not in mode:
            raise OSError('locked by antivirus')
        return real_open(f, *a, **k)

    monkeypatch.setattr(builtins, 'open', locked_open)
    prefs.write_pref('theme', 'midnight')       # should abort, not clobber
    monkeypatch.setattr(builtins, 'open', real_open)

    assert json.loads(p.read_text(encoding='utf-8')) == original


def test_write_pref_resets_a_corrupt_file(tmp_path, monkeypatch):
    # An already-corrupt (unparseable) prefs file is safe to reset — the write
    # should still succeed and persist the new key.
    p = _point_prefs_at(tmp_path, monkeypatch)
    p.write_text('{not valid json', encoding='utf-8')
    prefs.write_pref('theme', 'dark')
    assert json.loads(p.read_text(encoding='utf-8')) == {'theme': 'dark'}
