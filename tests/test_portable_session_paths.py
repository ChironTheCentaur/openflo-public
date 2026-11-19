"""Saved paths travel between computers and operating systems.

A session saved on a Windows PC is opened on Linux, macOS or a phone that
works on a copy of the project in its own storage, and saved again there;
spectral controls generated on one computer are used on another. These
tests pin the shared helpers (``paths``), the reader
(``session_open.resolve_sample_path`` / ``offline_entry`` /
``resolve_processed_path``), the project archive and the spectral
``controls.json``. Old sessions written by earlier savers, with the fields
those wrote for a Windows path re-saved on Linux, still open.

Synthetic paths (``C:\\data\\run1\\FL_sample_01.fcs``) and generated files
only; no Tk.
"""
from __future__ import annotations

import json
import logging
import ntpath
import os
import shutil

import pytest

from openflo import paths as P
from openflo import session_open as so
from tests._open_fixtures import make_fcs

WIN = 'C:\\data\\run1\\FL_sample_01.fcs'
UNC = '\\\\server\\share\\run1\\FL_sample_02.fcs'


def _touch(path):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('x')
    return str(path)


# ── the shared helpers ──────────────────────────────────────────────────────

@pytest.mark.parametrize('saved, name', [
    (WIN, 'FL_sample_01.fcs'), (UNC, 'FL_sample_02.fcs'),
    ('C:/data/FL_sample_03.fcs', 'FL_sample_03.fcs'),
    ('/data/run1/FL_sample_04.fcs', 'FL_sample_04.fcs'),
    ('raw\\FL_sample_05.fcs', 'FL_sample_05.fcs'),
    ('FL_sample_06.fcs', 'FL_sample_06.fcs'), ('', ''),
    ('E:FL_sample_07.fcs', 'FL_sample_07.fcs'),
    ('run1/a:FL_sample_08.fcs', 'a:FL_sample_08.fcs')])
def test_the_file_name_is_split_on_either_separator(saved, name):
    assert P.portable_basename(saved) == name


@pytest.mark.parametrize('saved, absolute', [
    (WIN, True), (UNC, True), ('C:/data/x.fcs', True), ('/data/x.fcs', True),
    ('raw\\x.fcs', False), ('raw/x.fcs', False), ('x.fcs', False),
    # drive-relative, and a POSIX name with a colon: neither is absolute
    ('E:x.fcs', False), ('a:b.fcs', False), ('run1/a:b.fcs', False),
    ('ab:/x.fcs', False)])
def test_a_path_absolute_on_any_os_is_known_as_such(saved, absolute):
    assert P.is_foreign_absolute(saved) is absolute


def test_a_relative_link_never_comes_from_the_working_folder(
        tmp_path, monkeypatch):
    """A Windows path read on Linux (or a POSIX one on Windows) is not a
    path on this computer: no link, and nothing of the working folder."""
    monkeypatch.chdir(tmp_path)
    sess = str(tmp_path / 'proj')
    foreign = UNC if os.name == 'nt' else WIN
    assert P.portable_rel_path(foreign, sess) is None
    if os.name != 'nt':
        assert P.portable_rel_path('/data/run1/x.fcs', sess) == (
            os.path.relpath('/data/run1/x.fcs', sess))
    here = str(tmp_path / 'proj' / 'raw' / 'x.fcs')
    assert P.portable_rel_path(here, sess) == 'raw/x.fcs'


def test_the_saved_hints_of_a_foreign_path_are_its_name_only(tmp_path,
                                                             monkeypatch):
    monkeypatch.chdir(tmp_path)
    foreign = UNC if os.name == 'nt' else WIN
    hints = P.relink_hints(foreign, str(tmp_path / 'proj'))
    assert hints == {'basename': P.portable_basename(foreign)}


def test_a_kept_link_is_carried_to_the_folder_saved_in(tmp_path):
    """An offline sample keeps the link it was opened with: unchanged when
    saved in the same folder, carried over when saved elsewhere."""
    proj = str(tmp_path / 'proj')
    kept = {'kept_rel': 'raw\\FL_sample_01.fcs', 'kept_from': proj}
    assert P.relink_hints(WIN, proj, **kept) == {
        'basename': 'FL_sample_01.fcs', 'rel_path': 'raw/FL_sample_01.fcs'}
    assert P.relink_hints(WIN, str(tmp_path / 'out'), **kept)['rel_path'] == (
        '../proj/raw/FL_sample_01.fcs')


def test_a_local_path_is_linked_relative_to_the_session_folder(tmp_path,
                                                               monkeypatch):
    sub = tmp_path / 'work'
    sub.mkdir()
    monkeypatch.chdir(sub)
    f = _touch(tmp_path / 'proj' / 'raw' / 'a.fcs')
    assert P.relink_hints(f, str(tmp_path / 'proj')) == {
        'basename': 'a.fcs', 'rel_path': 'raw/a.fcs'}
    # A sample loaded by a name relative to where OpenFlo runs.
    rel_here = os.path.relpath(f, sub)
    assert P.relink_hints(rel_here, str(tmp_path / 'proj'))['rel_path'] == (
        'raw/a.fcs')


def test_a_saved_file_name_field_is_read_whole_but_a_bare_drive_path_not():
    """A ``basename`` field is a file name (``a:b.fcs`` is a POSIX one);
    only a saved path of the bare drive-relative shape loses its drive."""
    assert P.saved_file_name({'basename': 'a:b.fcs', 'path': 'E:x.fcs'}) == (
        'a:b.fcs')
    assert P.saved_file_name({'basename': WIN}) == 'FL_sample_01.fcs'
    assert P.saved_file_name({'path': 'E:x.fcs'}) == 'x.fcs'
    assert P.saved_file_name({}) == ''


@pytest.mark.skipif(os.name == 'nt', reason='a colon is not a Windows '
                    'file-name character')
def test_a_posix_name_with_a_colon_is_linked_and_found_by_its_name(
        tmp_path, monkeypatch):
    sess = tmp_path / 'proj'
    here = _touch(sess / 'a:b.fcs')
    monkeypatch.chdir(sess)
    assert P.relink_hints('a:b.fcs', str(sess)) == {
        'basename': 'a:b.fcs', 'rel_path': 'a:b.fcs'}
    assert so.resolve_sample_path(
        {'path': '/gone/a:b.fcs', 'basename': 'a:b.fcs'}, str(sess)) == here


# ── the Windows branches, run on any OS ─────────────────────────────────────

@pytest.fixture
def on_windows(monkeypatch):
    """The helpers' path flavour set to Windows' for this test only (the
    module's own `_osp`; ``os`` itself is not touched)."""
    monkeypatch.setattr(P, '_osp', ntpath)


def test_on_windows_a_kept_link_to_another_drive_falls_back_to_the_path(
        on_windows):
    """Saved onto the drive the file is on: linked from its path."""
    kept = {'kept_rel': 'raw/FL_sample_01.fcs', 'kept_from': 'D:\\proj'}
    assert P.relink_hints('E:\\data\\FL_sample_01.fcs', 'E:\\out',
                          **kept) == {'basename': 'FL_sample_01.fcs',
                                      'rel_path': '../data/FL_sample_01.fcs'}
    # Same drive: carried over as on any OS.
    assert P.relink_hints('F:\\gone\\FL_sample_01.fcs', 'D:\\copies',
                          **kept)['rel_path'] == (
        '../proj/raw/FL_sample_01.fcs')


def test_on_windows_no_link_when_nothing_is_on_the_drive_saved_to(
        on_windows):
    """Kept link on D:, file on F:, saved on E: no link at all."""
    assert P.relink_hints('F:\\gone\\FL_sample_01.fcs', 'E:\\out',
                          kept_rel='raw/FL_sample_01.fcs',
                          kept_from='D:\\proj') == {
        'basename': 'FL_sample_01.fcs'}


def test_on_windows_a_path_without_a_drive_is_not_linked(on_windows):
    """``\\data\\x.fcs`` names a folder on whichever drive is current."""
    assert P.portable_rel_path('\\data\\x.fcs', '\\proj') is None
    assert P.portable_rel_path('/data/x.fcs', '/proj') is None
    assert P.portable_rel_path('\\\\server\\share\\run\\x.fcs',
                               '\\\\server\\share\\proj') == '../run/x.fcs'
    assert P.portable_rel_path('C:\\proj\\raw\\x.fcs', 'C:\\proj') == (
        'raw/x.fcs')


# ── reading a session saved on another OS ───────────────────────────────────

@pytest.mark.parametrize('saved', [WIN, UNC, 'C:/data/run1/FL_sample_01.fcs'])
def test_a_windows_path_is_found_by_its_name_beside_the_session(tmp_path,
                                                                saved):
    sess = tmp_path / 'proj'
    here = _touch(sess / P.portable_basename(saved))
    assert so.resolve_sample_path({'path': saved}, str(sess)) == here


def test_an_old_save_with_a_whole_windows_path_as_basename_opens(tmp_path):
    """What the Linux/macOS saver wrote before for a Windows path: the whole
    path as ``basename``, and a link through the working folder."""
    sess = tmp_path / 'proj'
    here = _touch(sess / 'FL_sample_01.fcs')
    old = {'path': WIN, 'basename': WIN,
           'rel_path': '../../work/C:\\data\\run1\\FL_sample_01.fcs'}
    assert so.resolve_sample_path(old, str(sess)) == here


def test_the_order_is_path_then_link_then_name(tmp_path):
    sess = tmp_path / 'proj'
    linked = _touch(sess / 'raw' / 'FL_sample_01.fcs')
    _touch(sess / 'FL_sample_01.fcs')
    got = so.resolve_sample_path(
        {'path': WIN, 'rel_path': 'raw\\FL_sample_01.fcs', 'basename': WIN},
        str(sess))
    assert got == os.path.normpath(linked)


def test_a_wrong_case_link_is_not_matched_on_a_case_sensitive_system(
        tmp_path):
    sess = tmp_path / 'proj'
    _touch(sess / 'raw' / 'FL_sample_01.fcs')
    if os.path.exists(sess / 'RAW' / 'fl_sample_01.FCS'):
        pytest.skip('case-insensitive file system')
    entry = {'path': WIN, 'rel_path': 'raw\\fl_sample_01.FCS',
             'basename': 'fl_sample_01.FCS'}
    assert so.resolve_sample_path(entry, str(sess)) == WIN


@pytest.mark.parametrize('saved', [WIN, UNC])
def test_a_missing_windows_sample_is_named_by_its_file_name(tmp_path, saved):
    (p,) = so.plan_samples([{'path': saved}], {}, str(tmp_path))
    assert p.label == P.portable_basename(saved) and not p.staged


@pytest.mark.parametrize('saved', [WIN, UNC])
def test_an_offline_windows_path_is_kept_as_saved(tmp_path, saved):
    """Not joined onto the session folder as if it were relative."""
    e = so.offline_entry({'name': 'x', 'path': saved,
                          'rel_path': 'raw\\x.fcs'}, str(tmp_path))
    assert e['path'] == saved
    assert e['rel_path'] == 'raw\\x.fcs'


def test_an_offline_relative_path_is_still_made_absolute(tmp_path):
    e = so.offline_entry({'name': 'x', 'path': 'raw/x.fcs'}, str(tmp_path))
    assert e['path'] == os.path.join(str(tmp_path), 'raw', 'x.fcs')


def test_an_offline_drive_relative_path_is_kept_as_saved(tmp_path):
    """``E:x.fcs`` read on Linux is not ``<session folder>/E:x.fcs``."""
    e = so.offline_entry({'name': 'x', 'path': 'E:x.fcs',
                          'basename': 'x.fcs'}, str(tmp_path))
    assert e['path'] == 'E:x.fcs'


def test_an_offline_drive_relative_path_naming_a_local_file_is_linked(
        tmp_path):
    """A POSIX file that is really called ``E:x.fcs`` in the session folder
    is read from there."""
    if os.name == 'nt':
        pytest.skip('a file name cannot hold a colon on Windows')
    here = _touch(tmp_path / 'E:x.fcs')
    e = so.offline_entry({'name': 'x', 'path': 'E:x.fcs'}, str(tmp_path))
    assert e['path'] == here


def test_a_missing_drive_relative_sample_is_still_found_by_name_after_a_save(
        tmp_path):
    """Kept offline and saved, the entry still names ``x.fcs``: put the file
    beside the saved session and it is found."""
    from openflo.document import Document
    sess = tmp_path / 'pc' / 'run.flowsession'
    sess.parent.mkdir()
    sess.write_text(json.dumps({
        'format': 'openflo-session', 'version': 3, 'samples': [
            {'name': 'x', 'path': 'E:x.fcs', 'basename': 'x.fcs'}],
        'sample_gates': {}}), encoding='utf-8')
    doc = Document()
    assert doc.open_session(str(sess))['missing'] == ['x']
    out = tmp_path / 'proj' / 'run.flowsession'
    out.parent.mkdir()
    doc.save_session(str(out))
    (entry,) = json.loads(out.read_text(encoding='utf-8'))['samples']
    assert (entry['path'], entry['basename']) == ('E:x.fcs', 'x.fcs')
    make_fcs(out.parent / 'x.fcs')
    assert Document().open_session(str(out))['samples'] == ['x']


def test_a_nameless_entry_is_reported_by_its_file_name(tmp_path):
    """An older saver wrote a whole Windows path as ``basename``."""
    (p,) = so.plan_samples([{'basename': WIN}], {}, str(tmp_path))
    assert p.label == 'FL_sample_01.fcs'


# ── the processed-data pointer stays in the session folder ─────────────────

def test_a_windows_sidecar_pointer_is_read(tmp_path):
    side = _touch(tmp_path / 's_data' / 'A.npz')
    got = so.resolve_processed_path({'processed_data': 's_data\\A.npz'}, 'A',
                                    str(tmp_path))
    assert os.path.samefile(got, side)


def test_a_folder_whose_name_starts_like_the_session_folder_is_outside(
        tmp_path, caplog):
    """'proj_x' begins with 'proj' but is not inside it."""
    sess = tmp_path / 'proj'
    sess.mkdir()
    _touch(tmp_path / 'proj_x' / 'A.npz')
    assert so.resolve_processed_path(
        {'processed_data': '../proj_x/A.npz'}, 'A', str(sess)) == ''
    assert 'out-of-tree' in caplog.text


def test_a_sidecar_pointer_through_a_link_out_of_the_folder_is_refused(
        tmp_path, caplog):
    sess = tmp_path / 'proj'
    sess.mkdir()
    _touch(tmp_path / 'elsewhere' / 'A.npz')
    try:
        os.symlink(tmp_path / 'elsewhere', sess / 'lnk')
    except (OSError, NotImplementedError):
        pytest.skip('no symbolic links here')
    assert so.resolve_processed_path({'processed_data': 'lnk/A.npz'}, 'A',
                                     str(sess)) == ''
    assert 'out-of-tree' in caplog.text


def test_a_session_folder_reached_through_a_link_still_reads(tmp_path):
    """A phone's shared storage is often a link to the real folder: both
    sides are resolved before they are compared."""
    real = tmp_path / 'real'
    side = _touch(real / 'proj' / 's_data' / 'A.npz')
    try:
        os.symlink(real, tmp_path / 'link')
    except (OSError, NotImplementedError):
        pytest.skip('no symbolic links here')
    got = so.resolve_processed_path({'processed_data': 's_data/A.npz'}, 'A',
                                    str(tmp_path / 'link' / 'proj'))
    assert os.path.samefile(got, side)


@pytest.mark.parametrize('pointer', [WIN, UNC, '/data/A.npz'])
def test_an_absolute_sidecar_pointer_from_any_os_is_refused(tmp_path,
                                                            pointer, caplog):
    assert so.resolve_processed_path({'processed_data': pointer}, 'A',
                                     str(tmp_path)) == ''
    assert 'out-of-tree' in caplog.text


def test_a_linked_data_folder_is_read_by_its_usual_name_and_says_so(
        tmp_path, caplog):
    """``<stem>_data`` itself a link out of the session folder: the pointer
    through it is refused, the sidecar is read by its usual name, and the
    log says that rather than that it was ignored."""
    sess = tmp_path / 'proj'
    sess.mkdir()
    _touch(tmp_path / 'elsewhere' / 'A.npz')
    try:
        os.symlink(tmp_path / 'elsewhere', sess / 's_data')
    except (OSError, NotImplementedError):
        pytest.skip('no symbolic links here')
    with caplog.at_level(logging.WARNING, logger=so.log.name):
        got = so.resolve_processed_path(
            {'processed_data': 's_data/A.npz'}, 'A', str(sess),
            data_dir=str(sess / 's_data'))
    assert got == str(sess / 's_data' / 'A.npz')
    assert 'out-of-tree' not in caplog.text
    assert "'s_data/A.npz' leads outside the session folder" in caplog.text
    assert 'usual name' in caplog.text


def test_a_sidecar_that_is_a_link_out_of_its_data_folder_is_not_read(
        tmp_path, caplog):
    """``<stem>_data/A.npz`` itself a link to a file elsewhere: neither the
    pointer nor the by-name fallback reads it."""
    sess = tmp_path / 'proj'
    out = _touch(tmp_path / 'elsewhere' / 'A.npz')
    (sess / 's_data').mkdir(parents=True)
    try:
        os.symlink(out, sess / 's_data' / 'A.npz')
    except (OSError, NotImplementedError):
        pytest.skip('no symbolic links here')
    with caplog.at_level(logging.WARNING, logger=so.log.name):
        for entry in ({'processed_data': 's_data/A.npz'}, {}):
            assert so.resolve_processed_path(
                entry, 'A', str(sess), data_dir=str(sess / 's_data')) == ''
    assert 'usual name' not in caplog.text
    assert "ignoring out-of-tree processed_data: 's_data/A.npz'" in \
        caplog.text


def test_the_log_says_why_a_pointer_was_refused(tmp_path, caplog):
    """A ``..`` escape is not said to be through a symbolic link; a link
    is."""
    sess = tmp_path / 'proj'
    sess.mkdir()
    _touch(tmp_path / 'x' / 'B.npz')
    with caplog.at_level(logging.WARNING, logger=so.log.name):
        so.resolve_processed_path({'processed_data': '../x/B.npz'}, 'B',
                                  str(sess))
    assert 'leads outside the session folder)' in caplog.text
    assert 'symbolic link' not in caplog.text
    caplog.clear()
    try:
        os.symlink(tmp_path / 'x', sess / 'lnk')
    except (OSError, NotImplementedError):
        pytest.skip('no symbolic links here')
    with caplog.at_level(logging.WARNING, logger=so.log.name):
        so.resolve_processed_path({'processed_data': 'lnk/B.npz'}, 'B',
                                  str(sess))
    assert 'through a symbolic link' in caplog.text


# ── the project archive ─────────────────────────────────────────────────────

def test_an_archive_packs_a_windows_saved_sample_found_by_its_link(tmp_path):
    from openflo.project_archive import write_project_archive
    proj = tmp_path / 'proj'
    make_fcs(proj / 'raw' / 'FL_sample_01.fcs', seed=1)
    sess = proj / 'p.flowsession'
    sess.write_text(json.dumps({'samples': [
        {'name': 'FL_sample_01', 'path': WIN,
         'rel_path': 'raw\\FL_sample_01.fcs', 'basename': 'FL_sample_01.fcs'}]}),
        encoding='utf-8')
    manifest = write_project_archive(str(sess), str(tmp_path / 'p.zip'))
    assert manifest['samples'] == ['FL_sample_01'] and not manifest['missing']


def test_an_archive_reads_a_relative_saved_path_in_the_session_folder(
        tmp_path, monkeypatch):
    """Not in the working folder, even when a file of that name is there
    too (an archive itself saves ``fcs/<name>.fcs``)."""
    from openflo.project_archive import open_project_archive, write_project_archive
    proj = tmp_path / 'proj'
    mine = make_fcs(proj / 'fcs' / 'FL_sample_01.fcs', seed=1)
    work = tmp_path / 'work'
    make_fcs(work / 'fcs' / 'FL_sample_01.fcs', seed=2)
    monkeypatch.chdir(work)
    sess = proj / 'p.flowsession'
    sess.write_text(json.dumps({'samples': [
        {'name': 'FL_sample_01', 'path': 'fcs/FL_sample_01.fcs'}]}),
        encoding='utf-8')
    out = tmp_path / 'p.zip'
    manifest = write_project_archive(str(sess), str(out))
    assert manifest['samples'] == ['FL_sample_01']
    open_project_archive(str(out), str(tmp_path / 'back'))
    with open(mine, 'rb') as fh:
        assert (tmp_path / 'back' / 'fcs' / 'FL_sample_01.fcs').read_bytes() \
            == fh.read()


# ── spectral controls.json ──────────────────────────────────────────────────

def test_spectral_controls_name_their_files_relative_to_the_json(tmp_path):
    from openflo.cli_unmix import load_unmix_controls
    from openflo.synthetic import make_spectral_dataset
    made, cj = make_spectral_dataset(str(tmp_path / 'spectral'), n=300,
                                     seed=3)
    with open(cj, encoding='utf-8') as fh:
        raw = json.load(fh)
    assert all(not os.path.isabs(v) and os.sep not in v and '/' not in v
               for v in raw.values()), raw
    # Moved (to another computer, say) and read from somewhere else.
    moved = tmp_path / 'moved' / 'spectral'
    shutil.copytree(tmp_path / 'spectral', moved)
    shutil.rmtree(tmp_path / 'spectral')
    elsewhere = tmp_path / 'cwd'
    elsewhere.mkdir()
    old = os.getcwd()
    os.chdir(elsewhere)
    try:
        got = load_unmix_controls(str(moved / 'controls.json'))
    finally:
        os.chdir(old)
    assert set(got) == set(raw)
    assert all(os.path.isfile(p) for p in got.values())
    assert {os.path.basename(p) for p in got.values()} == {
        os.path.basename(p) for p in made if 'mixed' not in p}


def test_old_absolute_and_windows_relative_controls_entries_still_read(
        tmp_path):
    from openflo.cli_unmix import load_unmix_controls
    absolute = make_fcs(tmp_path / 'abs' / 'ss_a.fcs', seed=1)
    make_fcs(tmp_path / 'ctl' / 'sub' / 'ss_b.fcs', seed=2)
    cj = tmp_path / 'ctl' / 'controls.json'
    cj.write_text(json.dumps({'FL1': absolute, 'FL2': 'sub\\ss_b.fcs'}),
                  encoding='utf-8')
    got = load_unmix_controls(str(cj))
    assert got['FL1'] == absolute
    assert os.path.samefile(got['FL2'], tmp_path / 'ctl' / 'sub' / 'ss_b.fcs')


def test_an_older_controls_file_is_read_from_the_working_folder(
        tmp_path, monkeypatch):
    """An entry not beside the file is read where earlier versions read
    it, the working folder; one in both places is read beside the file."""
    from openflo.cli_unmix import load_unmix_controls
    work = tmp_path / 'work'
    make_fcs(work / 'data' / 'ss_a.fcs', seed=1)
    make_fcs(work / 'data' / 'ss_b.fcs', seed=2)
    beside = make_fcs(tmp_path / 'ctl' / 'data' / 'ss_b.fcs', seed=3)
    cj = tmp_path / 'ctl' / 'controls.json'
    cj.write_text(json.dumps({'FL1': 'data/ss_a.fcs', 'FL2': 'data/ss_b.fcs',
                              'FL3': 'data/ss_c.fcs'}), encoding='utf-8')
    monkeypatch.chdir(work)
    got = load_unmix_controls(str(cj))
    assert got['FL1'] == 'data/ss_a.fcs'
    assert os.path.samefile(got['FL1'], work / 'data' / 'ss_a.fcs')
    assert os.path.samefile(got['FL2'], beside)
    # Found nowhere: named beside the file, where the error will point.
    assert got['FL3'] == os.path.join(str(tmp_path / 'ctl'), 'data',
                                      'ss_c.fcs')


def test_inline_controls_stay_relative_to_the_working_folder():
    from openflo.cli_unmix import load_unmix_controls
    assert load_unmix_controls('{"FL1": "ss_a.fcs"}') == {'FL1': 'ss_a.fcs'}
