"""OpenFlo project files (``.flowproject``): one file with a session, its
data and its FCS files, opened on another machine or in the phone app
(``Document.unpack_project``), and every archive that is not one OpenFlo
wrote refused before a byte is written. Generated data only."""
from __future__ import annotations

import json
import os
import struct
import zipfile

import numpy as np
import pytest

from openflo import project_archive as pa
from openflo.document import Document, DocumentError
from tests.test_document import gated_doc, spill_fcs

# ── a session with its files, round trip ────────────────────────────────────


def _project(tmp_path):
    """A saved session of two samples whose FCS files share one name (in
    two folders): s1 gated, its compensation turned off; the other on. The
    session is written, then packed as a project; the originals removed."""
    doc, name, *_ = gated_doc(tmp_path)
    other = doc.open_fcs(spill_fcs(tmp_path / 'raw2' / 's1.fcs', seed=7))
    doc.add_gate(other, {'kind': 'threshold', 'channel': 'FL2-A',
                         'value': 0.4, 'name': 'FL2+'})
    doc.set_compensation(name, on=False)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    out = tmp_path / ('run' + pa.ARCHIVE_EXT)
    manifest = pa.write_project_archive(str(sess), str(out))
    want = {s: (doc.stats(s), doc.compensation(s)['on'])
            for s in (name, other)}
    return out, manifest, want


def _values(rows):
    """Every channel statistic of `rows`, flat, in one order."""
    return [v for r in rows for _c, st in sorted(r['channels'].items())
            for _k, v in sorted(st.items())]


def _counts(rows):
    return [(r['population'], r['count']) for r in rows]


def test_a_project_unpacks_and_reopens_with_gates_and_compensation(tmp_path):
    out, manifest, want = _project(tmp_path)
    assert out.suffix == '.flowproject'
    assert sorted(manifest['samples']) == sorted(want)
    import shutil
    for d in ('raw', 'raw2', 'proj'):
        shutil.rmtree(tmp_path / d)
    dest = tmp_path / 'phone' / '7'
    r = Document().unpack_project(str(out), str(dest))
    assert r == {'session': str(dest / 'project.flowsession'), 'missing': []}
    # The two files of one name are both there, under names of their own.
    assert sorted(os.listdir(dest / 'fcs')) == ['s1 (2).fcs', 's1.fcs']
    doc = Document()
    listed = doc.data_files(r['session'])
    assert all(f['found'] for f in listed['files']) and len(listed['files']) == 2
    opened = doc.open_session(r['session'])
    assert opened['missing'] == []
    assert sorted(opened['samples']) == sorted(want)
    for s, (stats, on) in want.items():
        assert doc.compensation(s)['on'] is on
        assert _counts(doc.stats(s)) == _counts(stats)
        # Every statistic as before (to float rounding: a sample whose
        # compensation was turned off is put back through the matrix).
        assert _values(doc.stats(s)) == pytest.approx(_values(stats),
                                                      rel=1e-6)


def test_a_sample_saved_without_its_file_is_listed_missing(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    os.remove(tmp_path / 'raw' / 's1.fcs')
    out = tmp_path / 'run.flowproject'
    pa.write_project_archive(str(sess), str(out))
    r = Document().unpack_project(str(out), str(tmp_path / 'u'))
    assert r['missing'] == [name]
    opened = Document().open_session(r['session'])
    assert opened['missing'] == [name] and opened['samples'] == []


def test_the_desktop_export_saves_a_flowproject_and_opens_old_zips(tmp_path):
    assert pa.ARCHIVE_EXT == '.flowproject'
    assert '.zip' in pa.ARCHIVE_EXTS
    out, _manifest, _want = _project(tmp_path)
    old = tmp_path / 'old.zip'
    os.rename(out, old)
    assert pa.is_project_archive(str(old))
    sess, _ = pa.open_project_archive(str(old))
    assert os.path.isfile(sess)


def test_fcs_files_are_packed_where_it_pays_and_come_back_exact(tmp_path):
    from tests._fcs_writer import write_fcs
    doc, _name, *_ = gated_doc(tmp_path)        # full-precision floats
    rng = np.random.default_rng(4)
    ints = rng.integers(0, 1024, (20000, 4)).astype(float)
    ints[:, 2:] = np.minimum(ints[:, 2:], 300)  # ADC steps, few levels
    (tmp_path / 'raw3').mkdir()
    small = write_fcs(str(tmp_path / 'raw3' / 'int.fcs'), ints,
                      ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'], datatype='I',
                      bits=16)
    # Noise in every bit: packing cannot pay, so it goes in as it is.
    noise = rng.integers(0, 2 ** 32, (20000, 4), dtype=np.uint64)
    (tmp_path / 'raw4').mkdir()
    loud = write_fcs(str(tmp_path / 'raw4' / 'noise.fcs'), noise,
                     ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'], datatype='I',
                     bits=32)
    doc.open_fcs(small)
    doc.open_fcs(loud)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    out = tmp_path / 'run.flowproject'
    manifest = pa.write_project_archive(str(sess), str(out))
    files = {f['path']: f for f in manifest['files']}
    assert {p: (f['member'], f['codec']) for p, f in files.items()} == {
        'fcs/s1.fcs': ('fcs/s1.fcs.ofz', 'fcs-shuffle-deflate'),
        'fcs/int.fcs': ('fcs/int.fcs.ofz', 'fcs-shuffle-deflate'),
        'fcs/noise.fcs': ('fcs/noise.fcs', 'stored')}
    originals = {'s1.fcs': tmp_path / 'raw' / 's1.fcs', 'int.fcs': small,
                 'noise.fcs': loud}
    with zipfile.ZipFile(out) as zf:
        info = {i.filename: i for i in zf.infolist()}
    for arc, orig in originals.items():
        f = files['fcs/' + arc]
        assert f['bytes'] == os.path.getsize(orig)
        member = info[f['member']]
        assert member.compress_type == zipfile.ZIP_STORED
        assert member.file_size <= pa.KEEP_RATIO * f['bytes'] or \
            f['codec'] == 'stored'
    assert info['fcs/int.fcs.ofz'].file_size < 0.7 * files['fcs/int.fcs'][
        'bytes']
    assert not [n for n in os.listdir(tmp_path) if n.startswith('.openflo')]
    pa.open_project_archive(str(out), str(tmp_path / 'back'))
    assert sorted(os.listdir(tmp_path / 'back' / 'fcs')) == sorted(originals)
    for arc, orig in originals.items():
        assert (tmp_path / 'back' / 'fcs' / arc).read_bytes() == \
            open(orig, 'rb').read()
    # And it opens, with the same counts.
    reopened = Document()
    r = reopened.open_session(str(tmp_path / 'back' / 'project.flowsession'))
    assert r['missing'] == [] and len(r['samples']) == 3


def _packed_zip(tmp_path, name, size, packed, *, member=None, path=None,
                listed=None):
    """A project zip whose manifest lists one packed FCS file `path` of
    `size` bytes in member `listed`, held in `member` with the bytes
    `packed` (both ``<path>.ofz`` unless given)."""
    path = path or 'fcs/a.fcs'
    member = member or path + '.ofz'
    return _zip(tmp_path / name, {member: packed}, manifest={
        'session': pa.SESSION_NAME,
        'files': [{'path': path, 'member': listed or path + '.ofz',
                   'bytes': size,
                   'codec': 'fcs-shuffle-deflate'}]})


def _packed_bytes(tmp_path):
    from tests._fcs_writer import write_fcs
    ev = np.random.default_rng(2).integers(0, 1024, (3000, 3)).astype(float)
    (tmp_path / 'src').mkdir(exist_ok=True)
    src = write_fcs(str(tmp_path / 'src' / 'a.fcs'), ev,
                    ['FSC-A', 'SSC-A', 'FL1-A'])
    import io

    from openflo.fcs_pack import pack_fcs
    b = io.BytesIO()
    pack_fcs(src, b)
    return src, b.getvalue()


def test_a_packed_file_unpacks_only_as_its_manifest_lists_it(tmp_path):
    src, packed = _packed_bytes(tmp_path)
    size = os.path.getsize(src)
    good = _packed_zip(tmp_path, 'g.flowproject', size, packed)
    Document().unpack_project(good, str(tmp_path / 'ok'))
    assert (tmp_path / 'ok' / 'fcs' / 'a.fcs').read_bytes() == \
        open(src, 'rb').read()
    # Not listed, listed for another file, or with a size that is no size.
    z = _zip(tmp_path / 'u.flowproject', {'fcs/a.fcs.ofz': packed})
    _refused(tmp_path, z, 'does not list')
    z = _packed_zip(tmp_path, 'w.flowproject', size, packed,
                    member='fcs/b.fcs.ofz')
    _refused(tmp_path, z, 'does not list')
    for bad in (-1, '12', True, None):
        z = _packed_zip(tmp_path, f'b{bad}.flowproject', bad, packed)
        _refused(tmp_path, z, 'unreadable project description')
    z = _packed_zip(tmp_path, 'n.flowproject', size, packed,
                    path='project_data/a.fcs')
    _refused(tmp_path, z, 'unreadable project description')
    # Listed under a member that is not the file's own name + .ofz.
    z = _packed_zip(tmp_path, 'm.flowproject', size, packed,
                    member='fcs/other.ofz', listed='fcs/other.ofz')
    _refused(tmp_path, z, 'unreadable project description')
    # The size it lists is held to: a packed file of another size fails.
    z = _packed_zip(tmp_path, 's.flowproject', size - 1, packed)
    _refused(tmp_path, z, 'size differs')
    # A packed file and a plain one that land on one file.
    z = _zip(tmp_path / 'd.flowproject', {'fcs/a.fcs.ofz': packed,
                                         'fcs/A.fcs': b'x'}, manifest={
        'session': pa.SESSION_NAME,
        'files': [{'path': 'fcs/a.fcs', 'member': 'fcs/a.fcs.ofz',
                   'bytes': size, 'codec': 'fcs-shuffle-deflate'}]})
    _refused(tmp_path, z, 'twice')


def test_a_packed_file_counts_at_its_unpacked_size_against_the_cap(
        tmp_path):
    src, packed = _packed_bytes(tmp_path)
    size = os.path.getsize(src)
    assert len(packed) < size
    z = _packed_zip(tmp_path, 'c.flowproject', size, packed)
    _refused(tmp_path, z, 'unpacks to', max_bytes=size - 1)
    Document().unpack_project(z, str(tmp_path / 'fits'), max_bytes=size + 10_000)


# ── archives refused ───────────────────────────────────────────────────────


def _zip(path, members, manifest=None):
    """A zip at `path`: the manifest (unless `manifest` is False), a session,
    then `members` ``{name: bytes}``."""
    with zipfile.ZipFile(path, 'w') as zf:
        if manifest is not False:
            zf.writestr(pa.MANIFEST, json.dumps(
                manifest or {'format': 'openflo-project', 'version': 1,
                             'session': pa.SESSION_NAME}))
        zf.writestr(pa.SESSION_NAME, '{}')
        for name, data in members.items():
            zf.writestr(name, data)
    return str(path)


def _nothing_left(tmp_path, dest):
    assert not os.path.exists(dest)
    assert not [n for n in os.listdir(os.path.dirname(dest))
                if n.startswith('.openflo-unpack-')]


def _refused(tmp_path, path, match, **kw):
    dest = str(tmp_path / 'out' / 'x')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with pytest.raises(DocumentError, match=match):
        Document().unpack_project(path, dest, **kw)
    _nothing_left(tmp_path, dest)


def test_a_good_minimal_archive_unpacks(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 50,
                                         'project_data/s/a.npz': b'y'})
    r = Document().unpack_project(z, str(tmp_path / 'o'))
    assert open(tmp_path / 'o' / 'fcs' / 'a.fcs', 'rb').read() == b'x' * 50
    assert r['session'] == str(tmp_path / 'o' / 'project.flowsession')


@pytest.mark.parametrize('name', [
    '../evil.fcs', 'fcs/../../evil.fcs', '/abs.fcs', 'C:/x.fcs',
    'fcs\\x.fcs', 'fcs/a\x01.fcs', 'project_data/' + '/'.join('d' * 9),
    'project_data/' + 'n' * 300,
])
def test_an_unsafe_name_is_refused(tmp_path, name):
    z = _zip(tmp_path / 'a.flowproject', {name: b'x'})
    _refused(tmp_path, z, 'outside its folder')


@pytest.mark.parametrize('name', ['notes.txt', 'fcs/sub/a.fcs',
                                  'other/a.fcs', 'run.flowsession',
                                  'project_dataX/a.npz'])
def test_a_part_no_project_holds_is_refused(tmp_path, name):
    z = _zip(tmp_path / 'a.flowproject', {name: b'x'})
    _refused(tmp_path, z, 'not part of an OpenFlo project')


def test_a_zip_without_the_manifest_is_refused(tmp_path):
    z = _zip(tmp_path / 'a.zip', {'fcs/a.fcs': b'x'}, manifest=False)
    _refused(tmp_path, z, 'not an OpenFlo project archive')


def test_two_names_that_land_on_one_file_are_refused(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/A.fcs': b'1',
                                         'fcs/a.fcs': b'2'})
    _refused(tmp_path, z, 'twice')
    # Unicode forms too: e + combining accent, and the precomposed letter.
    z = _zip(tmp_path / 'b.flowproject', {'fcs/e\u0301.fcs': b'1',
                                         'fcs/\u00e9.fcs': b'2'})
    _refused(tmp_path, z, 'twice')


def test_names_windows_reads_as_one_are_refused(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'project_data/s/a': b'1',
                                         'project_data/s/a.': b'2'})
    _refused(tmp_path, z, 'twice')


def test_a_file_and_a_folder_of_one_name_are_refused_plainly(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'project_data/a': b'1',
                                         'project_data/a/b': b'2'})
    dest = str(tmp_path / 'out' / 'x')
    os.makedirs(os.path.dirname(dest))
    with pytest.raises(DocumentError, match='could not be unpacked') as e:
        Document().unpack_project(z, dest)
    assert '.openflo-unpack-' not in str(e.value)
    _nothing_left(tmp_path, dest)


def test_folder_entries_from_a_re_zip_are_skipped(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/': b'', 'project_data/s/': b'',
                                         'fcs/a.fcs': b'x'})
    Document().unpack_project(z, str(tmp_path / 'o'))
    assert (tmp_path / 'o' / 'fcs' / 'a.fcs').read_bytes() == b'x'


def test_too_many_files_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, 'MAX_MEMBERS', 3)
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'1',
                                         'fcs/b.fcs': b'2'})
    _refused(tmp_path, z, 'more than 3')


def test_an_archive_above_the_cap_is_refused(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 3_000_000})
    _refused(tmp_path, z, 'unpacks to 4 MB.*up to 2 MB', max_bytes=2_000_000)
    Document().unpack_project(z, str(tmp_path / 'ok'),
                             max_bytes=4_000_000)
    assert os.path.getsize(tmp_path / 'ok' / 'fcs' / 'a.fcs') == 3_000_000


def test_no_room_on_the_disk_is_refused(tmp_path, monkeypatch):
    import shutil
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 1000})
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(pa.shutil, 'disk_usage',
                        lambda p: usage._replace(free=pa.DISK_MARGIN))
    dest = str(tmp_path / 'out' / 'x')
    os.makedirs(os.path.dirname(dest))
    with pytest.raises(pa.ProjectArchiveError, match='MB free'):
        pa.open_project_archive(z, dest)
    _nothing_left(tmp_path, dest)


def test_a_manifest_from_a_newer_openflo_or_too_large_is_refused(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {}, manifest={
        'version': pa.ARCHIVE_VERSION + 1, 'session': pa.SESSION_NAME})
    _refused(tmp_path, z, 'newer OpenFlo')
    z = _zip(tmp_path / 'v.flowproject', {}, manifest={
        'version': '1', 'session': pa.SESSION_NAME})
    _refused(tmp_path, z, 'newer OpenFlo')
    z = _zip(tmp_path / 'b.flowproject', {}, manifest={
        'session': pa.SESSION_NAME, 'pad': 'x' * pa.MAX_MANIFEST_BYTES})
    _refused(tmp_path, z, 'too large')
    with zipfile.ZipFile(tmp_path / 'c.flowproject', 'w') as zf:
        zf.writestr(pa.MANIFEST, '[not json')
        zf.writestr(pa.SESSION_NAME, '{}')
    _refused(tmp_path, str(tmp_path / 'c.flowproject'), 'unreadable')


def test_a_session_the_manifest_names_outside_the_top_is_refused(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/s.flowsession': b'{}'},
             manifest={'session': 'fcs/s.flowsession'})
    _refused(tmp_path, z, 'no session file')


def _patch_central(path, name, *, flag=None, method=None, csize=None,
                   usize=None):
    """Overwrite fields of member `name`'s central directory record (and
    the flag/method in its local header) in the zip at `path`."""
    data = bytearray(open(path, 'rb').read())
    enc = name.encode()
    at = 0
    while True:
        at = data.index(b'PK\x01\x02', at)
        nlen = struct.unpack_from('<H', data, at + 28)[0]
        if bytes(data[at + 46:at + 46 + nlen]) == enc:
            break
        at += 4
    local = struct.unpack_from('<I', data, at + 42)[0]
    if flag is not None:
        struct.pack_into('<H', data, at + 8, flag)
        struct.pack_into('<H', data, local + 6, flag)
    if method is not None:
        struct.pack_into('<H', data, at + 10, method)
        struct.pack_into('<H', data, local + 8, method)
    if csize is not None:
        struct.pack_into('<I', data, at + 20, csize)
    if usize is not None:
        struct.pack_into('<I', data, at + 24, usize)
    open(path, 'wb').write(bytes(data))


def test_an_encrypted_or_oddly_packed_member_is_refused(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 10})
    _patch_central(z, 'fcs/a.fcs', flag=0x1)
    _refused(tmp_path, z, 'encrypted')
    z = _zip(tmp_path / 'b.flowproject', {'fcs/a.fcs': b'x' * 10})
    _patch_central(z, 'fcs/a.fcs', method=zipfile.ZIP_BZIP2)
    _refused(tmp_path, z, 'packed in a way')


def test_members_claiming_more_than_the_file_holds_are_refused(tmp_path):
    # A zip bomb's overlapping members: their packed sizes add up past
    # the archive's own size.
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 10})
    _patch_central(z, 'fcs/a.fcs', csize=10 ** 8)
    _refused(tmp_path, z, 'claims more data')


def test_a_member_larger_than_it_says_leaves_nothing(tmp_path):
    # Declared 10 bytes, holding 4000: never more than 10 written, and the
    # whole unpack undone when the check at its end fails.
    with zipfile.ZipFile(tmp_path / 'a.flowproject', 'w',
                         zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(pa.MANIFEST, json.dumps({'session': pa.SESSION_NAME}))
        zf.writestr(pa.SESSION_NAME, '{}')
        zf.writestr('project_data/a.npz', b'z' * 4000)
    z = str(tmp_path / 'a.flowproject')
    _patch_central(z, 'project_data/a.npz', usize=10)
    _refused(tmp_path, z, 'not a readable OpenFlo project|larger than')


def test_a_member_smaller_than_it_says_leaves_nothing(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 10})
    _patch_central(z, 'fcs/a.fcs', usize=20)
    _refused(tmp_path, z, 'not a readable OpenFlo project|smaller than')


def test_a_file_that_is_no_zip_is_refused(tmp_path):
    p = tmp_path / 'a.flowproject'
    p.write_bytes(b'not a zip at all')
    _refused(tmp_path, str(p), 'not a readable OpenFlo project')


def test_an_existing_folder_is_never_unpacked_into(tmp_path):
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x'})
    (tmp_path / 'there').mkdir()
    with pytest.raises(DocumentError, match='already exists'):
        Document().unpack_project(z, str(tmp_path / 'there'))
    assert os.listdir(tmp_path / 'there') == []


def test_a_member_is_never_written_past_its_declared_size(tmp_path,
                                                          monkeypatch):
    # zipfile stops at the declared size itself; the unpack counts too, so
    # a reader that does not (a future zipfile, another library) cannot
    # fill the disk or the phone's memory.
    import io
    z = _zip(tmp_path / 'a.flowproject', {'fcs/a.fcs': b'x' * 10})
    real = zipfile.ZipFile.open

    def overlong(self, info, *a, **kw):
        name = getattr(info, 'filename', info)
        if name == 'fcs/a.fcs':
            return io.BytesIO(b'x' * 5000)
        return real(self, info, *a, **kw)
    monkeypatch.setattr(zipfile.ZipFile, 'open', overlong)
    _refused(tmp_path, z, 'larger than the archive says')


# ── expanding a project back to FCS files and a FlowJo workspace ───────────


def test_a_project_expands_to_its_fcs_files_and_a_flowjo_workspace(
        tmp_path):
    import shutil
    out, _manifest, want = _project(tmp_path)
    originals = {'s1.fcs': (tmp_path / 'raw' / 's1.fcs').read_bytes(),
                 's1 (2).fcs': (tmp_path / 'raw2' / 's1.fcs').read_bytes()}
    for d in ('raw', 'raw2', 'proj'):
        shutil.rmtree(tmp_path / d)
    r = pa.expand_project(str(out), str(tmp_path / 'x' / 'run'))
    folder = tmp_path / 'x' / 'run'
    assert r['folder'] == str(folder) and r['missing'] == []
    assert r['wsp'] == str(folder / 'run.wsp') and r['left_out'] == 0
    assert r['fcs'] == sorted(originals)
    for name, raw in originals.items():
        assert (folder / 'fcs' / name).read_bytes() == raw
    doc = Document()
    imported = doc.import_wsp(r['wsp'])
    assert imported['unresolved'] == []
    # FlowJo names a sample after its file ('s1 (2)' for the second s1).
    # Its copy counts every population as the session did, on the matrix
    # the sample was compensated with (none when turned off).
    got = sorted((_counts(doc.stats(s, channels=[], stats=())),
                  doc.compensation(s)['on']) for s in imported['samples'])
    assert got == sorted((_counts(stats), on)
                         for stats, on in want.values())


def test_a_failed_expand_leaves_no_folder(tmp_path, monkeypatch):
    out, _manifest, _want = _project(tmp_path)

    def broken(self, path, within=None):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(Document, 'export_wsp', broken)
    with pytest.raises(OSError, match='No space'):
        pa.expand_project(str(out), str(tmp_path / 'x' / 'run'))
    assert os.listdir(tmp_path / 'x') == []


def test_a_sample_saved_without_its_file_is_named_when_expanding(tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    os.remove(tmp_path / 'raw' / 's1.fcs')
    out = tmp_path / 'run.flowproject'
    pa.write_project_archive(str(sess), str(out))
    r = pa.expand_project(str(out), str(tmp_path / 'x'))
    assert r['missing'] == [name] and r['fcs'] == []
    assert os.path.isfile(r['wsp'])


def test_a_sample_opened_from_its_saved_data_alone_stays_out_of_the_wsp(
        tmp_path):
    # Saved with its processed data but not its FCS: the expanded session
    # still opens it (from the data), but FlowJo has no file for it, so the
    # workspace leaves it out rather than naming a path from the computer
    # the archive was made on, and the expand names it as missing.
    from openflo.session_sidecar import write_frame
    doc, name, *_ = gated_doc(tmp_path)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    side = tmp_path / 'proj' / 'run_data' / 's1.npz'
    side.parent.mkdir()
    assert write_frame(doc._samples[name].data.data, str(side))
    d = json.loads(sess.read_text('utf-8'))
    d['samples'][0]['processed_data'] = 'run_data/s1.npz'
    sess.write_text(json.dumps(d), 'utf-8')
    raw = tmp_path / 'raw'
    os.remove(raw / 's1.fcs')
    out = tmp_path / 'run.flowproject'
    assert pa.write_project_archive(str(sess), str(out))['missing'] == [name]
    # Back where it was (a drive plugged in again): still not named, the
    # workspace holding only what the expanded folder has.
    spill_fcs(raw / 's1.fcs')
    r = pa.expand_project(str(out), str(tmp_path / 'x'))
    assert r['missing'] == [name] and r['fcs'] == []
    text = open(r['wsp'], encoding='utf-8').read()
    assert str(raw) not in text and 's1.fcs' not in text


def test_the_workspace_carries_each_population_count(tmp_path):
    import xml.etree.ElementTree as ET
    doc, name, *_ = gated_doc(tmp_path)
    doc.export_wsp(str(tmp_path / 'a.wsp'))
    rec = doc._samples[name]
    want = {rec.gates[r['id']]['name']: r['count']
            for r in doc.stats(name, channels=[], stats=(), all_events=False)}
    got = {e.get('name'): int(e.get('count'))
           for e in ET.parse(tmp_path / 'a.wsp').getroot().iter('Population')}
    assert got == want and len(got) == 3


def test_export_wsp_writes_only_samples_whose_file_is_in_the_folder(
        tmp_path):
    doc, name, *_ = gated_doc(tmp_path)
    r = doc.export_wsp(str(tmp_path / 'a.wsp'))
    assert (r['samples'], r['not_written']) == (1, [])
    r = doc.export_wsp(str(tmp_path / 'b.wsp'),
                       within=str(tmp_path / 'elsewhere'))
    assert (r['samples'], r['not_written'], r['gates']) == (0, [name], 0)
    os.remove(tmp_path / 'raw' / 's1.fcs')
    r = doc.export_wsp(str(tmp_path / 'c.wsp'))
    assert (r['samples'], r['not_written']) == (0, [name])


def test_the_workspace_leaves_out_what_flowjo_cannot_hold_and_keeps_the_rest(
        tmp_path):
    # A NOT over two gates (FlowJo's NotNode takes one) with a gate under
    # it: the NOT is left out and the gate under it moves up to the NOT's
    # own parent, as the editor's File > Export FlowJo .wsp does.
    doc, name, cells, pos, mid = gated_doc(tmp_path)
    nb = doc.add_gate(name, {'kind': 'boolean', 'op': 'not',
                             'operands': [pos, mid], 'name': 'neither'},
                      cells)['id']
    doc.add_gate(name, {'kind': 'threshold', 'channel': 'FL3-A',
                        'value': 0.2, 'name': 'under'}, nb)
    r = doc.export_wsp(str(tmp_path / 'o.wsp'))
    assert (r['gates'], r['left_out']) == (4, 1)
    back = Document()
    s = back.import_wsp(str(tmp_path / 'o.wsp'))['samples'][0]
    tree = back.gate_tree(s)
    assert [g['name'] for g in tree] == ['cells']
    assert [g['name'] for g in tree[0]['children']] == ['FL1+', 'FL2 mid',
                                                        'under']
