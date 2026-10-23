"""A session survives a bad save, Save writes back to the file opened, and a
project travels as one archive.

* Saving over a session keeps the previous save as ``name.backup.flowsession``
  (autosaves don't need one), and a session that can't be read offers it.
* Resume used to look only at the newest orphan autosave, so one cut short
  by a crash hid every older one that was fine.
* Ctrl+S asked for a file name every time, and Save and close wrote only the
  autosave, so the file the user opened went stale.
* A project archive packs the session, its data and every FCS file into one
  zip that opens on another machine.

Generated data throughout: synthetic FCS files and the sessions written from
them. Each behaviour has its negative case beside it.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile

import pytest

from tests.test_session_restore_transforms import (
    _fcs_without_spill,
    _loaded,
    _restore,
)
from tests.test_transform_records import (
    _new_editor,
    capture_warnings,
    isolate_home,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path
    root.destroy()


def _editor(root, tmp, names=('synthetic_A', 'synthetic_B')):
    data = tmp / 'data'
    data.mkdir(exist_ok=True)
    ed = _new_editor(root)
    for i, nm in enumerate(names):
        ed._on_loaded(nm, _loaded(_fcs_without_spill(data / f'{nm}.fcs',
                                                     i + 1)))
        _gate(ed, nm, f'{nm} pos')
    return ed


def _gate(ed, sample, name, value=0.5):
    ed._set_active_sample(sample)
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': value,
                  'parent_id': None, 'name': name, 'enabled': True},
                 replace_1d=False)


def _gate_names(path):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return sorted(g.get('name') for gl in d['sample_gates'].values()
                  for g in gl)


def _answer(monkeypatch, **answers):
    """Patch the editor's dialogs; record what each was asked."""
    from tkinter import filedialog, messagebox
    asked = {k: [] for k in ('askyesno', 'showerror', 'asksaveasfilename')}
    for mod, name in ((messagebox, 'askyesno'), (messagebox, 'showerror'),
                      (filedialog, 'asksaveasfilename')):
        def fn(*a, _n=name, **k):
            asked[_n].append((a, k))
            return answers.get(_n)
        monkeypatch.setattr(mod, name, fn)
    return asked


# ── item 7: backup of the previous save ──────────────────────────────────


def test_saving_over_a_session_keeps_the_previous_save(env):
    from openflo.editor_session import session_backup_path
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 'proj' / 'two.flowsession')
    os.makedirs(os.path.dirname(path))
    ed._write_session(path)
    first = _gate_names(path)
    _gate(ed, 'synthetic_A', 'added later', 0.7)
    ed._write_session(path)
    backup = session_backup_path(path)
    assert backup.endswith('two.backup.flowsession')
    assert _gate_names(backup) == first
    assert 'added later' in _gate_names(path)
    assert 'added later' not in _gate_names(backup)


def test_no_backup_on_a_first_save_or_an_autosave(env):
    from openflo.editor_session import session_backup_path
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 'one.flowsession')
    ed._write_session(path)
    assert not os.path.exists(session_backup_path(path))
    auto = ed._session_autosave_path()
    ed._write_session(auto)
    ed._write_session(auto)
    assert not os.path.exists(session_backup_path(auto))


def test_a_damaged_previous_file_never_replaces_a_good_backup(env):
    from openflo.editor_session import session_backup_path
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 'one.flowsession')
    ed._write_session(path)
    ed._write_session(path)                 # backup = a good save
    good = _gate_names(session_backup_path(path))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('{"format": "openflo-sess')  # cut short
    ed._write_session(path)
    assert _gate_names(session_backup_path(path)) == good


# ── item 7: resume tries older autosaves ─────────────────────────────────


def _orphans(ed, tmp, damaged_newest):
    """Two orphan autosaves (their PIDs are not running), the older one
    holding gate 'older'. The newer one is cut short when asked."""
    auto = ed._autosave_dir()
    older = os.path.join(auto, 'session-999990001-a.flowsession')
    newer = os.path.join(auto, 'session-999990002-b.flowsession')
    ed._write_session(older)
    ed._write_session(newer)
    if damaged_newest:
        with open(newer, 'r+', encoding='utf-8') as f:
            f.truncate(40)
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    return older, newer


def test_resume_offers_an_older_autosave_when_the_newest_is_damaged(env):
    root, tmp = env
    ed = _editor(root, tmp)
    older, _newer = _orphans(ed, tmp, damaged_newest=True)
    assert ed._find_resumable_session() == older


def test_resume_offers_the_newest_autosave_when_it_reads(env):
    root, tmp = env
    ed = _editor(root, tmp)
    _older, newer = _orphans(ed, tmp, damaged_newest=False)
    assert ed._find_resumable_session() == newer


def test_resume_offers_nothing_when_every_autosave_is_damaged(env):
    root, tmp = env
    ed = _editor(root, tmp)
    older, _newer = _orphans(ed, tmp, damaged_newest=True)
    with open(older, 'w', encoding='utf-8') as f:
        f.write('')
    assert ed._find_resumable_session() is None


# ── item 7: a damaged session offers its backup ──────────────────────────


def test_a_damaged_session_offers_and_opens_its_backup(env, monkeypatch):
    from openflo.editor_session import session_backup_path
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 'two.flowsession')
    ed._write_session(path)
    ed._write_session(path)
    with open(path, 'r+', encoding='utf-8') as f:
        f.truncate(100)
    asked = _answer(monkeypatch, askyesno=True)
    ed2 = _new_editor(root)
    opened = []
    real = ed2._apply_session
    ed2._apply_session = lambda data: (opened.append(data), real(data))
    ed2._load_session_path(path)
    assert len(asked['askyesno']) == 1
    title, msg = asked['askyesno'][0][0][:2]
    assert 'damaged or incomplete' in msg and 'previous save' in msg
    assert not asked['showerror']
    assert len(opened) == 1 and len(opened[0]['samples']) == 2
    # Save after opening the backup asks for a name: it must not write over
    # the backup, the only good copy.
    assert ed2._open_session_file() is None
    assert os.path.isfile(session_backup_path(path))


def test_a_damaged_session_without_a_backup_says_so_plainly(env, monkeypatch):
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 'two.flowsession')
    ed._write_session(path)
    with open(path, 'r+', encoding='utf-8') as f:
        f.truncate(100)
    asked = _answer(monkeypatch)
    _new_editor(root)._load_session_path(path)
    assert not asked['askyesno']
    msg = asked['showerror'][0][0][1]
    assert 'damaged or incomplete' in msg
    assert 'No backup' in msg


# ── item 8: Save writes to the open session ──────────────────────────────


def test_save_asks_once_then_writes_to_that_file(env, monkeypatch):
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 'named.flowsession')
    asked = _answer(monkeypatch, asksaveasfilename=path)
    ed._save_session()
    assert len(asked['asksaveasfilename']) == 1
    _gate(ed, 'synthetic_B', 'second save', 0.6)
    ed._save_session()
    assert len(asked['asksaveasfilename']) == 1      # not asked again
    assert 'second save' in _gate_names(path)
    assert ed.title().startswith('named.flowsession')


def test_save_as_always_asks_and_moves_later_saves(env, monkeypatch):
    root, tmp = env
    ed = _editor(root, tmp)
    one, two = str(tmp / 'one.flowsession'), str(tmp / 'two.flowsession')
    asked = _answer(monkeypatch, asksaveasfilename=one)
    ed._save_session()
    asked2 = _answer(monkeypatch, asksaveasfilename=two)
    ed._save_session_as()
    assert len(asked2['asksaveasfilename']) == 1
    _gate(ed, 'synthetic_A', 'after save as', 0.6)
    ed._save_session()
    assert len(asked2['asksaveasfilename']) == 1
    assert 'after save as' in _gate_names(two)
    assert 'after save as' not in _gate_names(one)
    assert asked['asksaveasfilename']


def test_save_writes_back_to_the_session_opened(env, monkeypatch):
    root, tmp = env
    ed = _editor(root, tmp)
    path = tmp / 'proj' / 's.flowsession'
    path.parent.mkdir()
    ed._write_session(str(path))
    ed2 = _restore(root, path, raw_first=True)
    asked = _answer(monkeypatch, asksaveasfilename=None)
    _gate(ed2, 'synthetic_A', 'after reopening', 0.6)
    ed2._save_session()
    assert not asked['asksaveasfilename']
    assert 'after reopening' in _gate_names(path)


def test_a_new_session_still_asks_for_a_name(env, monkeypatch):
    root, tmp = env
    ed = _editor(root, tmp)
    asked = _answer(monkeypatch, asksaveasfilename='')
    ed._save_session()
    assert len(asked['asksaveasfilename']) == 1
    assert ed._open_session_file() is None


@pytest.mark.parametrize('opened', [True, False])
def test_save_and_close_writes_the_open_file_and_the_autosave(
        env, monkeypatch, opened):
    root, tmp = env
    ed = _editor(root, tmp)
    path = str(tmp / 's.flowsession')
    if opened:
        _answer(monkeypatch, asksaveasfilename=path)
        ed._save_session()
    plans = []
    ed._start_close_save = lambda p: plans.append(p)
    ed._on_close(ask=False)
    written = [p['path'] for p in plans[0]]
    want = ([path] if opened else []) + [ed._session_autosave_path()]
    assert written == want


# ── item 9: project archive ──────────────────────────────────────────────


def _counts(ed):
    rows, _ = ed._collect_stats_rows({'Count'})
    return {(r['Sample'], r.get('Population')): r['Count'] for r in rows}


def test_a_project_archive_opens_elsewhere_with_the_same_results(env):
    from openflo.project_archive import (
        is_project_archive,
        open_project_archive,
        write_project_archive,
    )
    root, tmp = env
    ed = _editor(root, tmp)
    before = _counts(ed)
    sess = tmp / 'proj' / 's.flowsession'
    sess.parent.mkdir()
    ed._write_session(str(sess))
    out = str(tmp / 'send.zip')
    manifest = write_project_archive(str(sess), out)
    assert manifest['samples'] == ['synthetic_A', 'synthetic_B']
    assert manifest['missing'] == []
    assert is_project_archive(out)
    shutil.rmtree(tmp / 'data')                     # the originals are gone
    shutil.rmtree(tmp / 'proj')
    other = tmp / 'colleague'
    other.mkdir()
    moved = shutil.move(out, other / 'send.zip')
    opened, _ = open_project_archive(str(moved))
    assert os.path.dirname(opened) == str(other / 'send')
    ed2 = _restore(root, opened, raw_first=True)
    assert set(ed2._samples) == {'synthetic_A', 'synthetic_B'}
    assert _counts(ed2) == before
    for s in ed2._samples.values():
        assert os.path.dirname(s.path) == str(other / 'send' / 'fcs')


def test_open_session_opens_an_archive_directly(env, monkeypatch):
    from openflo.project_archive import write_project_archive
    root, tmp = env
    ed = _editor(root, tmp)
    sess = tmp / 's.flowsession'
    ed._write_session(str(sess))
    out = str(tmp / 'p.zip')
    write_project_archive(str(sess), out)
    ed2 = _new_editor(root)
    seen = []
    ed2._apply_session = lambda data: seen.append(data)
    ed2._load_session_path(out)
    assert len(seen) == 1
    assert [e['rel_path'] for e in seen[0]['samples']] == [
        'fcs/synthetic_A.fcs', 'fcs/synthetic_B.fcs']


def test_an_archive_lists_samples_whose_file_is_missing(env):
    from openflo.project_archive import write_project_archive
    root, tmp = env
    ed = _editor(root, tmp)
    sess = tmp / 's.flowsession'
    ed._write_session(str(sess))
    os.remove(tmp / 'data' / 'synthetic_B.fcs')
    manifest = write_project_archive(str(sess), str(tmp / 'p.zip'))
    assert manifest['samples'] == ['synthetic_A']
    assert manifest['missing'] == ['synthetic_B']
    with zipfile.ZipFile(tmp / 'p.zip') as zf:
        assert [n for n in zf.namelist() if n.startswith('fcs/')] == [
            'fcs/synthetic_A.fcs']


def test_a_plain_zip_is_not_opened_as_a_project(env):
    from openflo.project_archive import (
        ProjectArchiveError,
        is_project_archive,
        open_project_archive,
    )
    _root, tmp = env
    z = tmp / 'plain.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr('notes.txt', 'synthetic')
    assert not is_project_archive(str(z))
    with pytest.raises(ProjectArchiveError):
        open_project_archive(str(z))
    assert not (tmp / 'plain').exists()


def test_an_archive_writing_outside_its_folder_is_refused(env):
    from openflo.project_archive import (
        MANIFEST,
        SESSION_NAME,
        ProjectArchiveError,
        open_project_archive,
    )
    _root, tmp = env
    z = tmp / 'bad.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        zf.writestr(MANIFEST, json.dumps({'session': SESSION_NAME}))
        zf.writestr(SESSION_NAME, '{}')
        zf.writestr('../escaped.txt', 'synthetic')
    with pytest.raises(ProjectArchiveError):
        open_project_archive(str(z))
    assert not (tmp / 'escaped.txt').exists()
    assert not (tmp / 'bad').exists()
