"""Files OpenFlo keeps (preferences, run records, run templates, FlowSOM
maps, the transforms sidecar) are written whole or not at all
(``openflo.atomic_write``): a write that fails halfway leaves the old file
as it was and no temporary file behind.

Generated data, generic channel names only.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import openflo.atomic_write as AW


class _HalfJson:
    """``json`` whose dump writes part of the file, then fails."""
    dumps = staticmethod(json.dumps)

    @staticmethod
    def dump(obj, fh, **kw):
        fh.write('{"half": ')
        raise OSError('disk full')


@pytest.fixture
def half_json(monkeypatch):
    # Only the shared writer's json, so a writer that bypasses it does not
    # fail and the test says so.
    monkeypatch.setattr(AW, 'json', _HalfJson)


def _only(tmp_path):
    return sorted(p.name for p in tmp_path.iterdir())


def test_a_file_is_replaced_only_when_the_write_ends(tmp_path):
    p = str(tmp_path / 'a.json')
    AW.atomic_write_json(p, {'v': 1})
    with pytest.raises(TypeError):
        AW.atomic_write_json(p, {'v': object()})
    assert json.loads(open(p, encoding='utf-8').read()) == {'v': 1}
    assert _only(tmp_path) == ['a.json']
    with pytest.raises(ValueError), AW.atomic_open(p) as fh:
        fh.write('partial')
        raise ValueError('stop')
    assert json.loads(open(p, encoding='utf-8').read()) == {'v': 1}
    assert _only(tmp_path) == ['a.json']
    with AW.atomic_open(p) as fh:
        fh.write('{"v": 2}')
    assert json.loads(open(p, encoding='utf-8').read()) == {'v': 2}


def test_numpy_values_are_written_as_numbers():
    assert AW.json_default(np.eye(2)) == [[1.0, 0.0], [0.0, 1.0]]
    assert AW.json_default(np.float32(0.5)) == 0.5
    with pytest.raises(TypeError):
        AW.json_default(object())


def test_preferences_survive_a_failed_save(tmp_path, monkeypatch, half_json):
    import openflo.prefs as prefs
    p = tmp_path / 'prefs.json'
    p.write_text('{"theme": "light"}', encoding='utf-8')
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    prefs.write_pref('theme', 'dark')            # noted, not raised
    assert json.loads(p.read_text(encoding='utf-8')) == {'theme': 'light'}
    assert _only(tmp_path) == ['prefs.json']


def test_the_transforms_sidecar_is_whole_or_absent(tmp_path, half_json):
    from openflo import transform_sidecar as S
    csv = tmp_path / 'events.csv'
    csv.write_text('FL1-A\n1\n', encoding='utf-8')
    with pytest.raises(OSError):
        S.write_transforms_sidecar(str(csv), {'FL1-A': {'method': 'asinh'}},
                                   ['FL1-A'], 1)
    assert _only(tmp_path) == ['events.csv']


def test_a_run_template_survives_a_failed_reference_save(tmp_path, half_json):
    from openflo import run_reference as F
    from openflo.run_template import TEMPLATE_FORMAT
    t = tmp_path / 't.json'
    t.write_text(json.dumps({'format': TEMPLATE_FORMAT, 'schema': 1,
                             'title': 't', 'slots': [
                                 {'name': 'A', 'pattern': 'a',
                                  'populations': [{'sample': 'a'}]}]}),
                 encoding='utf-8')
    before = t.read_bytes()
    with pytest.raises(OSError):
        F.set_template_reference(str(t), None)
    assert t.read_bytes() == before
    assert _only(tmp_path) == ['t.json']


def test_a_flowsom_map_is_whole_or_absent(tmp_path, half_json):
    from openflo import run_reference as F
    with pytest.raises(OSError):
        F.write_map(str(tmp_path), 'A', {'format': 'x'})
    assert _only(tmp_path) == []
    assert not os.path.exists(tmp_path / ('A' + F.MAP_SUFFIX))


# ── #404 review follow-ups ─────────────────────────────────────────────────

def test_the_new_file_is_flushed_to_the_disk_before_it_replaces(
        tmp_path, monkeypatch):
    events = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(AW.os, 'fsync',
                        lambda fd: events.append('fsync') or real_fsync(fd))
    monkeypatch.setattr(AW.os, 'replace',
                        lambda a, b: events.append('replace') or real_replace(a, b))
    AW.atomic_write_json(str(tmp_path / 'a.json'), {'v': 1})
    assert events[:2] == ['fsync', 'replace']


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permissions')
def test_the_replaced_file_keeps_its_permissions(tmp_path):
    p = tmp_path / 'a.json'
    p.write_text('{}', encoding='utf-8')
    os.chmod(p, 0o600)
    AW.atomic_write_json(str(p), {'v': 1})
    assert (os.stat(p).st_mode & 0o777) == 0o600
    umask = os.umask(0)
    os.umask(umask)
    AW.atomic_write_json(str(tmp_path / 'new.json'), {'v': 1})
    assert (os.stat(tmp_path / 'new.json').st_mode & 0o777) == 0o666 & ~umask


@pytest.mark.skipif(os.name == 'nt', reason='symbolic links need rights')
def test_a_symbolic_link_keeps_pointing_at_the_written_file(tmp_path):
    real = tmp_path / 'real.json'
    real.write_text('{}', encoding='utf-8')
    link = tmp_path / 'link.json'
    os.symlink(real, link)
    AW.atomic_write_json(str(link), {'v': 2})
    assert os.path.islink(link)
    assert json.loads(real.read_text(encoding='utf-8')) == {'v': 2}
    assert _only(tmp_path) == ['link.json', 'real.json']


def test_two_writers_at_once_do_not_share_a_temporary_file(tmp_path):
    p = str(tmp_path / 'prefs.json')
    with AW.atomic_open(p) as first:
        first.write('{"from": "first"}')
        with AW.atomic_open(p) as second:
            second.write('{"from": "second"}')
    assert json.loads(open(p, encoding='utf-8').read()) == {'from': 'first'}
    assert _only(tmp_path) == ['prefs.json']


def test_the_permissions_are_given_after_the_write(tmp_path, monkeypatch):
    # A read-only file is replaced (as a non-root user too): its mode goes
    # on the new file only once that is written.
    p = tmp_path / 'a.json'
    p.write_text('{}', encoding='utf-8')
    events = []
    real = os.chmod
    monkeypatch.setattr(AW.os, 'chmod',
                        lambda f, m: events.append('chmod') or real(f, m))
    with AW.atomic_open(str(p)) as fh:
        fh.write('{"v": 1}')
        events.append('written')
    assert events == ['written', 'chmod']


@pytest.mark.skipif(getattr(os, 'geteuid', lambda: 0)() == 0,
                    reason='a read-only file stops only a non-root user')
def test_a_read_only_file_is_replaced_and_stays_read_only(tmp_path):
    p = tmp_path / 'a.json'
    p.write_text('{}', encoding='utf-8')
    os.chmod(p, 0o444)
    AW.atomic_write_json(str(p), {'v': 1})
    assert json.loads(p.read_text(encoding='utf-8')) == {'v': 1}
    assert (os.stat(p).st_mode & 0o777) == 0o444
    assert _only(tmp_path) == ['a.json']
