"""A session saved on Windows, opened and saved again on another OS.

The phone (and a Mac or Linux PC) opens sessions saved on a Windows PC and
works on a copy of the project in its own storage. Here a project is saved
by the real editor, its sample entries are rewritten as a Windows PC writes
them (drive-letter and UNC paths, backslash links and sidecar pointers),
the folder (session, its _data folder and the FCS files) is copied to a new
place, and the copy is opened and saved again from another working folder.

The saved links must be relative to the session's own folder, file names
must be file names, and a sample whose raw file is missing (kept offline,
or loaded from its sidecar) must keep its Windows path and its good link. The reopened copy must find every file.

Generated FCS files and synthetic paths only.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from tests._tk_wait import settle
from tests.conftest import gui_unavailable
from tests.test_session_restore_transforms import _fcs_without_spill, _loaded
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
)

NAMES = ('FL_sample_01', 'FL_sample_02', 'FL_sample_03', 'FL_sample_04',
         'FL_sample_05')


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: False)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path
    root.destroy()


def _windows_project(root, tmp):
    """A project saved by the editor, then given the sample entries a
    Windows PC writes. 01: drive-letter path, backslash link into raw\\;
    02: UNC path, found by its name beside the session; 03: a POSIX path
    from another computer, linked into raw\\; 04: missing everywhere; 05:
    loaded from its sidecar (backslash pointer), its raw file not copied."""
    proj = tmp / 'pc' / 'proj'
    (proj / 'raw').mkdir(parents=True)
    ed = _new_editor(root)
    for i, nm in enumerate(NAMES):
        p = _fcs_without_spill(proj / 'raw' / f'{nm}.fcs', i + 1)
        ed._on_loaded(nm, _loaded(p))
        ed._set_active_sample(nm)
        ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
                      'parent_id': None, 'name': f'{nm} pos',
                      'enabled': True})
    s5 = ed._samples['FL_sample_05']
    s5.data['cluster'] = np.arange(len(s5.data)) % 3
    sess = proj / 'run.flowsession'
    ed._write_session(str(sess))
    ed.destroy()
    data = json.loads(sess.read_text(encoding='utf-8'))
    by = {e['name']: e for e in data['samples']}
    for nm in NAMES[:4]:
        side = by[nm].pop('processed_data', None)
        if side:
            os.remove(proj / side)
    by['FL_sample_01'].update(path='C:\\data\\run1\\FL_sample_01.fcs',
                              rel_path='raw\\FL_sample_01.fcs',
                              basename='FL_sample_01.fcs')
    shutil.move(str(proj / 'raw' / 'FL_sample_02.fcs'),
                proj / 'FL_sample_02.fcs')
    by['FL_sample_02'].update(path='\\\\server\\share\\FL_sample_02.fcs',
                              basename='FL_sample_02.fcs')
    by['FL_sample_02'].pop('rel_path')
    by['FL_sample_03'].update(path='/data/run1/FL_sample_03.fcs',
                              rel_path='raw\\FL_sample_03.fcs')
    os.remove(proj / 'raw' / 'FL_sample_04.fcs')
    by['FL_sample_04'].update(path='E:\\gone\\FL_sample_04.fcs',
                              rel_path='raw\\FL_sample_04.fcs',
                              basename='FL_sample_04.fcs')
    os.remove(proj / 'raw' / 'FL_sample_05.fcs')
    by['FL_sample_05'].update(
        path='C:\\data\\run1\\FL_sample_05.fcs',
        rel_path='raw\\FL_sample_05.fcs', basename='FL_sample_05.fcs',
        processed_data=by['FL_sample_05']['processed_data'].replace('/', '\\'))
    sess.write_text(json.dumps(data), encoding='utf-8')
    return proj


def _open(root, path):
    """A fresh editor on session `path`, its loads run here in queue order."""
    ed = _new_editor(root)
    jobs = []
    ed._ensure_load_pool = lambda: None
    ed._enqueue_load = lambda payload, priority=1: jobs.append(
        (priority, len(jobs), payload))
    ed._eq_jobs = jobs
    ed._load_session_path(str(path))
    _run_loads(root, ed)
    return ed


def _run_loads(root, ed):
    """Run the loads `ed` queued, here, in queue order, and settle."""
    jobs = ed._eq_jobs
    while jobs:
        jobs.sort(key=lambda j: j[:2])
        name, fpath, *rest = jobs.pop(0)[2]
        if rest and rest[0] == 'csv':
            ed._load_csv_worker(name, fpath, rest[1])
        else:
            ed._load_worker(name, fpath)
    settle(root, 1.5)


def _entries(path):
    with open(path, encoding='utf-8') as fh:
        return {e['name']: e for e in json.load(fh)['samples']}


def test_a_windows_session_copied_elsewhere_opens_and_saves_portably(
        env, monkeypatch):
    root, tmp = env
    proj = _windows_project(root, tmp)
    phone = tmp / 'phone' / 'files' / 'proj'
    shutil.copytree(proj, phone)
    shutil.rmtree(tmp / 'pc')
    work = tmp / 'some' / 'working' / 'folder'
    work.mkdir(parents=True)
    monkeypatch.chdir(work)

    ed = _open(root, phone / 'run.flowsession')
    assert set(ed._samples) == set(NAMES) - {'FL_sample_04'}
    for nm in ('FL_sample_01', 'FL_sample_03'):
        assert os.path.samefile(ed._samples[nm].path,
                                phone / 'raw' / f'{nm}.fcs')
    assert os.path.samefile(ed._samples['FL_sample_02'].path,
                            phone / 'FL_sample_02.fcs')
    saved = phone / 'run.flowsession'
    ed._write_session(str(saved))
    ed.destroy()

    text = saved.read_text(encoding='utf-8')
    assert str(work) not in text and 'some/working' not in text
    by = _entries(saved)
    assert by['FL_sample_01']['rel_path'] == 'raw/FL_sample_01.fcs'
    assert by['FL_sample_02']['rel_path'] == 'FL_sample_02.fcs'
    assert by['FL_sample_03']['rel_path'] == 'raw/FL_sample_03.fcs'
    for nm in NAMES:
        assert by[nm]['basename'] == f'{nm}.fcs', by[nm]
    # The missing sample: its Windows path and its link, as saved there.
    assert by['FL_sample_04']['path'] == 'E:\\gone\\FL_sample_04.fcs'
    assert by['FL_sample_04']['rel_path'] == 'raw/FL_sample_04.fcs'
    # From its sidecar, its raw file not copied: its link kept too.
    assert by['FL_sample_05']['path'] == 'C:\\data\\run1\\FL_sample_05.fcs'
    assert by['FL_sample_05']['rel_path'] == 'raw/FL_sample_05.fcs'

    # The missing files turn up beside their links; the copy, copied again
    # and opened from yet another folder, finds every file.
    _fcs_without_spill(phone / 'raw' / 'FL_sample_04.fcs', 4)
    _fcs_without_spill(phone / 'raw' / 'FL_sample_05.fcs', 5)
    again = tmp / 'again' / 'proj'
    shutil.copytree(phone, again)
    monkeypatch.chdir(tmp)
    ed = _open(root, again / 'run.flowsession')
    assert set(ed._samples) == set(NAMES)
    assert os.path.samefile(ed._samples['FL_sample_04'].path,
                            again / 'raw' / 'FL_sample_04.fcs')
    # 05 loads from its sidecar, told the raw file its link finds.
    assert os.path.samefile(ed._samples['FL_sample_05'].source_fcs,
                            again / 'raw' / 'FL_sample_05.fcs')
    ed.destroy()


def test_an_offline_sample_saved_elsewhere_keeps_a_working_link(env):
    """Save As into another folder: the missing sample's link is carried
    over, so it still names raw/ beside the original session."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    ed = _open(root, proj / 'run.flowsession')
    assert 'FL_sample_04' not in ed._samples
    out = tmp / 'pc' / 'copies' / 'run2.flowsession'
    out.parent.mkdir()
    ed._write_session(str(out))
    ed.destroy()
    e = _entries(out)['FL_sample_04']
    assert e['rel_path'] == '../proj/raw/FL_sample_04.fcs'
    assert e['path'] == 'E:\\gone\\FL_sample_04.fcs'


def test_an_old_save_with_the_buggy_fields_still_opens(env):
    """What the Linux/macOS saver wrote before for a missing Windows-path
    sample: the session folder joined onto the Windows path, and that whole
    path as the file name. With the file beside the session it opens."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    sess = proj / 'run.flowsession'
    data = json.loads(sess.read_text(encoding='utf-8'))
    for e in data['samples']:
        if e['name'] == 'FL_sample_04':
            joined = os.path.join(str(proj), 'E:\\gone\\FL_sample_04.fcs')
            e.update(path=joined, basename='E:\\gone\\FL_sample_04.fcs',
                     rel_path='E:\\gone\\FL_sample_04.fcs')
    sess.write_text(json.dumps(data), encoding='utf-8')
    _fcs_without_spill(proj / 'FL_sample_04.fcs', 4)
    ed = _open(root, sess)
    assert set(ed._samples) == set(NAMES)
    assert os.path.samefile(ed._samples['FL_sample_04'].path,
                            proj / 'FL_sample_04.fcs')
    ed.destroy()


def test_locate_missing_finds_a_file_named_by_an_old_whole_windows_path(
        env):
    """An offline sample whose saved file name is a whole Windows path (an
    older Linux/macOS save) is still found by name in the chosen folder."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    sess = proj / 'run.flowsession'
    data = json.loads(sess.read_text(encoding='utf-8'))
    for e in data['samples']:
        if e['name'] == 'FL_sample_04':
            e.update(basename='E:\\gone\\FL_sample_04.fcs')
    sess.write_text(json.dumps(data), encoding='utf-8')
    ed = _open(root, sess)
    found_dir = tmp / 'found' / 'deeper'
    found_dir.mkdir(parents=True)
    _fcs_without_spill(found_dir / 'FL_sample_04.fcs', 4)
    assert ed._relink_missing(str(tmp / 'found')) == (['FL_sample_04'], [])
    ed.destroy()


def test_a_sample_loaded_anew_under_a_kept_name_gets_its_own_link(env):
    """The kept link belongs to the file the session named: a file loaded
    since under that sample's name is linked from where it is."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    ed = _open(root, proj / 'run.flowsession')
    assert 'FL_sample_04' not in ed._samples
    (proj / 'other').mkdir()
    other = _fcs_without_spill(proj / 'other' / 'FL_sample_04.fcs', 9)
    ed._on_loaded('FL_sample_04', _loaded(other))
    ed._write_session(str(proj / 'run.flowsession'))
    ed.destroy()
    e = _entries(proj / 'run.flowsession')['FL_sample_04']
    assert e['rel_path'] == 'other/FL_sample_04.fcs'


def test_a_file_found_again_where_it_was_is_linked_from_there(env):
    """Locate missing finds the file at its saved path after all: the link
    the session was opened with no longer applies."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    sess = proj / 'run.flowsession'
    gone = tmp / 'pc' / 'gone'
    data = json.loads(sess.read_text(encoding='utf-8'))
    for e in data['samples']:
        if e['name'] == 'FL_sample_04':
            e.update(path=str(gone / 'FL_sample_04.fcs'))
    sess.write_text(json.dumps(data), encoding='utf-8')
    ed = _open(root, sess)
    assert 'FL_sample_04' not in ed._samples
    gone.mkdir()
    _fcs_without_spill(gone / 'FL_sample_04.fcs', 4)
    assert ed._relink_missing(str(gone)) == (['FL_sample_04'], [])
    _run_loads(root, ed)
    assert 'FL_sample_04' in ed._samples
    ed._write_session(str(sess))
    ed.destroy()
    assert _entries(sess)['FL_sample_04']['rel_path'] == (
        '../gone/FL_sample_04.fcs')


def test_a_link_that_cannot_be_worked_out_is_dropped_not_kept_stale(
        env, monkeypatch):
    """Save As where no link can be worked out (on Windows: kept link,
    file and new folder on three drives): the missing sample's old link,
    relative to the folder it was opened from, is not written for the new
    folder, where it could name another file of that name."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    ed = _open(root, proj / 'run.flowsession')
    assert 'FL_sample_04' not in ed._samples
    from openflo import paths
    monkeypatch.setattr(paths, 'portable_rel_path', lambda path, folder: None)
    out = tmp / 'pc' / 'copies' / 'run2.flowsession'
    out.parent.mkdir()
    ed._write_session(str(out))
    ed.destroy()
    e = _entries(out)['FL_sample_04']
    assert 'rel_path' not in e, e
    assert e['basename'] == 'FL_sample_04.fcs'
    assert e['path'] == 'E:\\gone\\FL_sample_04.fcs'


def test_a_sample_found_at_its_saved_path_is_linked_from_there(env):
    """Its saved link is stale (the file was moved back, say): found by its
    path, it is linked from where it is, not by the stale link."""
    root, tmp = env
    proj = _windows_project(root, tmp)
    sess = proj / 'run.flowsession'
    data = json.loads(sess.read_text(encoding='utf-8'))
    for e in data['samples']:
        if e['name'] == 'FL_sample_03':
            e.update(path=str(proj / 'raw' / 'FL_sample_03.fcs'),
                     rel_path='old\\FL_sample_03.fcs')
    sess.write_text(json.dumps(data), encoding='utf-8')
    ed = _open(root, sess)
    assert os.path.samefile(ed._samples['FL_sample_03'].path,
                            proj / 'raw' / 'FL_sample_03.fcs')
    ed._write_session(str(sess))
    ed.destroy()
    assert _entries(sess)['FL_sample_03']['rel_path'] == (
        'raw/FL_sample_03.fcs')
