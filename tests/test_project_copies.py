"""Opening a project archive again reuses the copy unpacked before (a big
project on a USB stick took minutes every time, written beside it again as
"<name> (2)"); a new copy of an archive on a removable drive goes to the
internal project folder; the unpack counts its megabytes; and a session
copied next to the project it came from opens with that project's samples.
Generated data only."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from openflo import project_archive as pa
from openflo import project_copies as pc
from openflo.document import Document
from openflo.session_open import plan_samples
from tests.test_document import CELLS, spill_fcs


def _project(tmp_path, stem='run', seed=0):
    """A session of two samples whose FCS files share one name (in two
    folders), saved in proj/ and packed as `<stem>.flowproject` in usb/;
    the raw files then removed. Returns (archive, session, sample names)."""
    doc = Document()
    names = []
    for i, sub in enumerate(('d1', 'd2')):
        nm = doc.open_fcs(spill_fcs(tmp_path / 'raw' / sub / 'tube.fcs',
                                    seed=seed + i + 1, n=800))
        doc.add_gate(nm, CELLS)
        names.append(nm)
    (tmp_path / 'proj').mkdir()
    sess = tmp_path / 'proj' / 'run.flowsession'
    doc.save_session(str(sess))
    (tmp_path / 'usb').mkdir()
    arc = tmp_path / 'usb' / (stem + pa.ARCHIVE_EXT)
    pa.write_project_archive(str(sess), str(arc))
    shutil.rmtree(tmp_path / 'raw')
    return arc, sess, names


@pytest.fixture
def home(monkeypatch, tmp_path):
    h = tmp_path / 'home'
    h.mkdir()
    monkeypatch.setenv('HOME', str(h))
    monkeypatch.setenv('USERPROFILE', str(h))
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: False)
    return h


# ── reusing a copy ─────────────────────────────────────────────────────────


def test_a_copy_unpacked_before_is_found(tmp_path, home):
    arc, _sess, _names = _project(tmp_path)
    assert pc.find_copy(str(arc)) is None
    sess, _ = pa.open_project_archive(str(arc), pc.unpack_destination(
        str(arc)))
    folder = os.path.dirname(sess)
    assert folder == str(tmp_path / 'usb' / 'run')
    assert pc.find_copy(str(arc)) == pc.ArchiveCopy(folder, sess, False)


def test_a_copy_saved_since_is_told_apart_and_an_unchanged_one_preferred(
        tmp_path, home):
    arc, _sess, _names = _project(tmp_path)
    s1, _ = pa.open_project_archive(str(arc), pc.unpack_destination(str(arc)))
    with open(s1, 'a', encoding='utf-8') as fh:
        fh.write('\n')
    assert pc.find_copy(str(arc)) == pc.ArchiveCopy(
        os.path.dirname(s1), s1, True)
    s2, _ = pa.open_project_archive(str(arc), pc.unpack_destination(str(arc)))
    assert os.path.basename(os.path.dirname(s2)) == 'run (2)'
    assert pc.find_copy(str(arc)) == pc.ArchiveCopy(
        os.path.dirname(s2), s2, False)


@pytest.mark.parametrize('fault', ['fcs gone', 'fcs cut short',
                                   'other manifest', 'session gone'])
def test_a_folder_that_is_not_a_whole_copy_is_not_reused(tmp_path, home,
                                                         fault):
    arc, _sess, _names = _project(tmp_path)
    sess, manifest = pa.open_project_archive(
        str(arc), pc.unpack_destination(str(arc)))
    folder = os.path.dirname(sess)
    fcs = os.path.join(folder, *manifest['files'][1]['path'].split('/'))
    if fault == 'fcs gone':
        os.remove(fcs)
    elif fault == 'fcs cut short':
        with open(fcs, 'r+b') as fh:
            fh.truncate(os.path.getsize(fcs) - 1)
    elif fault == 'other manifest':
        m = os.path.join(folder, pa.MANIFEST)
        with open(m, 'a', encoding='utf-8') as fh:
            fh.write(' ')
    else:
        os.remove(sess)
    assert pc.find_copy(str(arc)) is None


def test_a_copy_in_the_internal_folder_is_found(tmp_path, home, monkeypatch):
    arc, _sess, _names = _project(tmp_path)
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: True)
    dest = pc.unpack_destination(str(arc))
    assert dest == os.path.join(str(home), '.openflo', 'projects', 'run')
    sess, _ = pa.open_project_archive(str(arc), dest)
    assert not os.path.exists(tmp_path / 'usb' / 'run')
    assert pc.find_copy(str(arc)) == pc.ArchiveCopy(dest, sess, False)


def test_a_new_copy_goes_beside_unless_removable_or_read_only(
        tmp_path, home, monkeypatch):
    arc = str(tmp_path / 'p.flowproject')
    store = os.path.join(str(home), '.openflo', 'projects')
    assert pc.unpack_destination(arc) == str(tmp_path / 'p')
    (tmp_path / 'p').mkdir()
    assert pc.unpack_destination(arc) == str(tmp_path / 'p (2)')
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: True)
    assert pc.unpack_destination(arc) == os.path.join(store, 'p')
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: False)
    real = os.access
    monkeypatch.setattr(pc.os, 'access',
                        lambda p, mode: False if p == str(tmp_path)
                        else real(p, mode))
    assert pc.unpack_destination(arc) == os.path.join(store, 'p')


@pytest.mark.skipif(os.name == 'nt', reason='POSIX mount points')
def test_removable_mount_points(monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert pc.on_removable_drive('/media/stick/a.flowproject')
    assert pc.on_removable_drive('/run/media/me/stick/a.flowproject')
    assert not pc.on_removable_drive('/home/me/a.flowproject')
    monkeypatch.setattr(sys, 'platform', 'darwin')
    assert pc.on_removable_drive('/Volumes/STICK/a.flowproject')
    assert not pc.on_removable_drive('/Users/me/a.flowproject')


def test_the_unpack_counts_its_bytes(tmp_path, home):
    arc, _sess, _names = _project(tmp_path)
    seen = []
    _s, manifest = pa.open_project_archive(
        str(arc), str(tmp_path / 'out'),
        progress=lambda d, t: seen.append((d, t)))
    total = seen[0][1]
    assert seen[0] == (0, total) and seen[-1] == (total, total)
    assert all(t == total for _d, t in seen)
    assert [d for d, _t in seen] == sorted(d for d, _t in seen)
    # every member after the first report, the FCS files at full size
    assert len(seen) == 1 + sum(len(fs) for _r, _d, fs in os.walk(
        tmp_path / 'out'))
    assert total >= sum(f['bytes'] for f in manifest['files'])


# ── a session's samples, in the project it came from ──────────────────────


def _session_beside(tmp_path, sess):
    """The session copied alone into usb/, beside the archive."""
    out = tmp_path / 'usb' / 'copied.flowsession'
    shutil.copy(sess, out)
    shutil.rmtree(tmp_path / 'proj')
    return out


def test_a_session_beside_its_unpacked_project_opens_with_its_samples(
        tmp_path, home):
    arc, sess, names = _project(tmp_path)
    pa.open_project_archive(str(arc), pc.unpack_destination(str(arc)))
    copied = _session_beside(tmp_path, sess)
    opened = Document().open_session(str(copied))
    assert opened['missing'] == []
    assert sorted(opened['samples']) == sorted(names)


def test_a_session_beside_an_unpacked_project_alone_finds_it(tmp_path,
                                                             home):
    """The unpacked folder copied without its archive is still found."""
    arc, sess, _names = _project(tmp_path)
    pa.open_project_archive(str(arc), pc.unpack_destination(str(arc)))
    os.remove(arc)
    copied = _session_beside(tmp_path, sess)
    assert Document().open_session(str(copied))['missing'] == []


def test_a_session_beside_an_archive_unpacked_internally_finds_it(
        tmp_path, home, monkeypatch):
    arc, sess, _names = _project(tmp_path)
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: True)
    pa.open_project_archive(str(arc), pc.unpack_destination(str(arc)))
    copied = _session_beside(tmp_path, sess)
    opened = Document().open_session(str(copied))
    assert opened['missing'] == []


def test_without_the_project_search_the_samples_stay_missing(tmp_path,
                                                             home):
    arc, sess, _names = _project(tmp_path)
    pa.open_project_archive(str(arc), pc.unpack_destination(str(arc)))
    copied = _session_beside(tmp_path, sess)
    data = json.loads(copied.read_text(encoding='utf-8'))
    d = str(copied.parent)
    off = plan_samples(data['samples'], {}, d, find_projects=False)
    on = plan_samples(data['samples'], {}, d)
    assert [p.raw_found for p in off] == [False, False]
    assert [p.raw_found for p in on] == [True, True]
    # each sample takes its own file, the renamed one included
    assert sorted(os.path.basename(p.raw_path) for p in on) == [
        'tube (2).fcs', 'tube.fcs']
    assert len({p.raw_path for p in on}) == 2
    # and is re-linked afresh at the next save
    assert all('rel_path' not in p.entry for p in on)


def test_same_file_name():
    assert pc.same_file_name('Tube.fcs', 'tube.FCS')
    assert pc.same_file_name('tube.fcs', 'tube (2).fcs')
    assert not pc.same_file_name('tube.fcs', 'tube (2).lmd')
    assert not pc.same_file_name('tube.fcs', 'tube2.fcs')
    assert not pc.same_file_name('tube (2).fcs', 'tube.fcs')


def _stats(digest):
    """A saved_stats record of data `digest`."""
    return {'app_version': '0', 'data': digest, 'counts': {}, 'stats': {}}


def _fake_project(folder, samples, digest='d1'):
    """An unpacked project in `folder` holding `samples`: {name: (file
    name, set number)}, each saved with numbers of data `digest`."""
    os.makedirs(os.path.join(folder, pa.FCS_DIR))
    entries = []
    for name, (fn, n) in samples.items():
        open(os.path.join(folder, pa.FCS_DIR, fn), 'wb').close()
        e = {'name': name, 'path': pa.FCS_DIR + '/' + fn,
             'rel_path': pa.FCS_DIR + '/' + fn, 'basename': fn,
             'saved_stats': _stats(digest)}
        if n is not None:
            e['set_number'] = n
        entries.append(e)
    with open(os.path.join(folder, pa.SESSION_NAME), 'w',
              encoding='utf-8') as fh:
        json.dump({'format': 'openflo-session', 'samples': entries}, fh)
    with open(os.path.join(folder, pa.MANIFEST), 'w', encoding='utf-8') as fh:
        fh.write('{}')
    return folder


def _found(matches):
    """`project_matches`' answer as {name: [(file, folder name, verified)]}."""
    return {n: [(os.path.basename(m.path), os.path.basename(m.folder),
                 m.verified) for m in ms] for n, ms in matches.items()}


def test_project_matches_by_name_file_name_and_data_set(tmp_path):
    a = _fake_project(str(tmp_path / 'a'), {
        's1': ('x.fcs', None), 's2': ('y.fcs', None), 's3': ('z.fcs', 2)})
    b = _fake_project(str(tmp_path / 'b'), {
        's1': ('x.fcs', None), 's2': ('y.fcs', None), 's3': ('z.fcs', 3),
        's4': ('w.fcs', None)})
    st = _stats('d1')
    want = [{'name': 's1', 'path': r'F:\run\x.fcs', 'saved_stats': st},
            {'name': 's2', 'basename': 'other.fcs', 'saved_stats': st},
            {'name': 's3', 'path': 'C:/d/z.fcs', 'set_number': 3,
             'saved_stats': st},
            {'name': 's4', 'path': '/old/w.fcs',
             'saved_stats': _stats('d2')}]
    got = _found(pc.project_matches(want, [a, b]))
    # s2 is another file; s3 is another data set in a; s4's numbers were of
    # other data, so it is not verified
    assert got == {'s1': [('x.fcs', 'a', True), ('x.fcs', 'b', True)],
                   's3': [('z.fcs', 'b', True)],
                   's4': [('w.fcs', 'b', False)]}
    m = pc.project_matches(want, [a, b])
    assert {n: os.path.basename(v.folder)
            for n, v in pc.verified_files(m).items()} == {'s1': 'a',
                                                          's3': 'b'}
    assert pc.folder_files(m, a) == {'s1': os.path.join(a, 'fcs', 'x.fcs')}
    assert pc.candidate_folders(m) == [(b, 3), (a, 1)]


def test_a_saved_file_identity_decides(tmp_path):
    a = _fake_project(str(tmp_path / 'a'), {'s1': ('x.fcs', None)})
    f = os.path.join(a, 'fcs', 'x.fcs')
    with open(f, 'wb') as fh:
        fh.write(b'FCS3.1 day A')
    ident = pc.fcs_identity(f)
    entry = {'name': 's1', 'path': 'x.fcs', 'trial': 'other'}
    # the same recording: verified, though the entries differ
    assert _found(pc.project_matches([dict(entry, fcs_id=ident)], [a])) == {
        's1': [('x.fcs', 'a', True)]}
    # another recording of that name: not a match at all
    other = dict(ident, head='0' * 32)
    assert pc.project_matches([dict(entry, fcs_id=other)], [a]) == {}
    bigger = dict(ident, bytes=ident['bytes'] + 1)
    assert pc.project_matches([dict(entry, fcs_id=bigger)], [a]) == {}


def test_candidate_folders_keep_a_tie_in_found_order(tmp_path):
    z = _fake_project(str(tmp_path / 'z'), {'s1': ('x.fcs', None)})
    a = _fake_project(str(tmp_path / 'a'), {'s1': ('x.fcs', None)})
    m = pc.project_matches([{'name': 's1', 'path': 'x.fcs'}], [z, a])
    assert pc.candidate_folders(m) == [(z, 1), (a, 1)]
    assert pc.verified_files(m) == {}


@pytest.mark.parametrize('mine', [None, {}, {'data': ''}, {'data': None},
                                  'not a record'])
def test_no_saved_digest_is_not_verified(tmp_path, mine):
    a = _fake_project(str(tmp_path / 'a'), {'s1': ('x.fcs', None)},
                      digest='')
    b = _fake_project(str(tmp_path / 'b'), {'s1': ('x.fcs', None)})
    e = {'name': 's1', 'path': 'x.fcs', 'saved_stats': mine}
    assert _found(pc.project_matches([e], [a, b])) == {
        's1': [('x.fcs', 'a', False), ('x.fcs', 'b', False)]}
    # a project entry without numbers does not verify a session's either
    e = dict(e, saved_stats=_stats(''))
    assert not pc.project_matches([e], [a])['s1'][0].verified


def test_two_data_sets_of_one_file_both_match(tmp_path):
    a = _fake_project(str(tmp_path / 'a'), {'w1': ('acc.c6', 1),
                                            'w2': ('acc.c6', 2)})
    got = _found(pc.project_matches(
        [{'name': 'w1', 'path': 'acc.c6', 'set_number': 1,
          'saved_stats': _stats('d1')},
         {'name': 'w2', 'path': 'acc.c6', 'set_number': 2,
          'saved_stats': _stats('d1')}], [a]))
    assert got == {'w1': [('acc.c6', 'a', True)],
                   'w2': [('acc.c6', 'a', True)]}


def test_a_listed_file_that_is_not_there_is_not_matched(tmp_path):
    a = _fake_project(str(tmp_path / 'a'), {'s1': ('x.fcs', None)})
    os.remove(os.path.join(a, 'fcs', 'x.fcs'))
    assert pc.project_matches([{'name': 's1', 'path': 'x.fcs'}], [a]) == {}


# ── another day's project of the same names (review B1) ───────────────────


def _digests(data, day):
    for e in data['samples']:
        e['saved_stats'] = _stats(f"{day}-{e['name']}")


def _digests_in(folder, day):
    p = folder / pa.SESSION_NAME
    data = json.loads(p.read_text(encoding='utf-8'))
    _digests(data, day)
    p.write_text(json.dumps(data), encoding='utf-8')


def _two_days(tmp_path, monkeypatch, *, mine_too, old_session):
    """Day B's session copied onto a stick (usb/) holding day A's project
    (same sample and file names, other events), unpacked as a_other/; with
    `mine_too`, day B's own project too, unpacked as z_mine/. With
    `old_session`, the session lacks the file identities older OpenFlo
    did not save. Returns (session path, day B's first FSC-A value)."""
    a_root, b_root = tmp_path / 'dayA', tmp_path / 'dayB'
    a_root.mkdir()
    b_root.mkdir()
    arc_a, _s, _n = _project(a_root, seed=50)
    arc_b, sess_b, _n = _project(b_root, seed=0)
    usb = tmp_path / 'stick'
    usb.mkdir()
    pa.open_project_archive(str(arc_a), str(usb / 'a_other'))
    if mine_too:
        pa.open_project_archive(str(arc_b), str(usb / 'z_mine'))
    data = json.loads(sess_b.read_text(encoding='utf-8'))
    if old_session:
        for e in data['samples']:
            e.pop('fcs_id')
        if old_session == 'with numbers':
            # numbers saved with each day's data digest, in the session
            # and in each project's own copy of its session
            _digests(data, 'B')
            _digests_in(usb / 'a_other', 'A')
            if mine_too:
                _digests_in(usb / 'z_mine', 'B')
    out = usb / 'dayB.flowsession'
    out.write_text(json.dumps(data), encoding='utf-8')
    return out


@pytest.mark.parametrize('old_session', [False, 'with numbers',
                                         'without numbers'])
def test_another_days_project_is_never_loaded_on_its_own(
        tmp_path, home, monkeypatch, old_session):
    sess = _two_days(tmp_path, monkeypatch, mine_too=False,
                     old_session=old_session)
    opened = Document().open_session(str(sess))
    assert sorted(opened['missing']) == ['tube', 'tube [d2]']
    assert opened['samples'] == []


@pytest.mark.parametrize('old_session', [False, 'with numbers'])
def test_the_sessions_own_project_wins_over_one_named_first(
        tmp_path, home, monkeypatch, old_session):
    sess = _two_days(tmp_path, monkeypatch, mine_too=True,
                     old_session=old_session)
    doc = Document()
    opened = doc.open_session(str(sess))
    assert opened['missing'] == []
    data = json.loads(sess.read_text(encoding='utf-8'))
    paths = {e['name']: e['path'] for e in doc._session_data(
        str(sess.parent))[0]['samples']}
    assert all(os.sep + 'z_mine' + os.sep in p for p in paths.values())
    assert len(data['samples']) == 2
    # and the user is told where they came from
    assert [n for n in opened['notes'] if 'loaded from project z_mine' in n]


def test_an_old_session_without_numbers_is_left_to_ask(tmp_path, home,
                                                       monkeypatch):
    """Nothing shows its own project is the same recordings: missing, so
    the editor asks (test_project_copies_editor)."""
    sess = _two_days(tmp_path, monkeypatch, mine_too=True,
                     old_session='without numbers')
    assert sorted(Document().open_session(str(sess))['missing']) == [
        'tube', 'tube [d2]']


def test_a_saved_session_records_each_files_identity(tmp_path, home):
    _arc, sess, _names = _project(tmp_path)
    data = json.loads(sess.read_text(encoding='utf-8'))
    ids = [e['fcs_id'] for e in data['samples']]
    assert all(set(i) == {'bytes', 'head'} for i in ids)
    assert ids[0] != ids[1]           # two recordings of one file name


# ── folders that are not projects ──────────────────────────────────────────


def test_a_half_written_unpack_folder_is_not_a_project(tmp_path, home):
    """A crashed unpack leaves .openflo-unpack-*: the manifest, the session
    and full-size FCS files, zero-filled where they were not written yet.
    It must not be read as a project."""
    arc, _sess, _names = _project(tmp_path)
    d = str(tmp_path / 'usb')
    pa.open_project_archive(str(arc), os.path.join(d, '.openflo-unpack-x'))
    assert pc.project_folders(d) == []
    os.rename(os.path.join(d, '.openflo-unpack-x'), os.path.join(d, 'kept'))
    assert pc.project_folders(d) == [os.path.join(d, 'kept')]


def test_a_crowded_folder_is_looked_at_only_so_far(tmp_path, home,
                                                   monkeypatch):
    for i in range(5):
        _fake_project(str(tmp_path / f'p{i}'), {'s': ('x.fcs', None)})
    assert len(pc.project_folders(str(tmp_path))) == 5
    monkeypatch.setattr(pc, 'MAX_SCAN', 3)
    assert len(pc.project_folders(str(tmp_path))) <= 3


def test_an_archive_naming_a_session_outside_its_copy_is_not_reused(
        tmp_path, home):
    import zipfile
    arc = tmp_path / 'x.flowproject'
    with zipfile.ZipFile(arc, 'w') as zf:
        zf.writestr(pa.MANIFEST, json.dumps({'session': '..\\x.json',
                                             'files': []}))
        zf.writestr('..\\x.json', '{}')
    (tmp_path / 'x').mkdir()
    (tmp_path / 'x' / pa.MANIFEST).write_bytes(
        zipfile.ZipFile(arc).read(pa.MANIFEST))
    (tmp_path / 'x' / '..\\x.json').write_text('{}')
    assert pc.find_copy(str(arc)) is None


def test_archives_listing_the_samples_are_found_most_first(tmp_path, home):
    arc, _sess, names = _project(tmp_path, stem='b')
    other = tmp_path / 'usb' / 'a.flowproject'
    import zipfile
    with zipfile.ZipFile(other, 'w') as zf:
        zf.writestr(pa.MANIFEST, json.dumps({'samples': [names[0]]}))
        zf.writestr(pa.SESSION_NAME, '{}')
    d = str(tmp_path / 'usb')
    assert pc.project_archives(d, names) == [str(arc), str(other)]
    assert pc.project_archives(d, ['nobody']) == []
