"""The editor side of project_copies: opening a project again reuses the
copy unpacked before (asking when its session was saved since), the
status line counts the megabytes of an unpack, and a session whose
samples are missing offers to load them from the project archive beside
it. Generated data only."""
from __future__ import annotations

import os
import re
import shutil

import pytest

from openflo import project_archive as pa
from openflo import project_copies as pc
from tests import test_open_session_files as _osf
from tests._tk_wait import pump_until
from tests.test_project_copies import _project

# the real-open editor of test_open_session_files
real_editor = _osf.real_editor


@pytest.fixture
def counted(monkeypatch):
    """Count the real unpacks."""
    calls = []
    real = pa.open_project_archive

    def unpack(path, dest=None, **k):
        calls.append(dest)
        return real(path, dest, **k)
    monkeypatch.setattr(pa, 'open_project_archive', unpack)
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: False)
    return calls


def _open(ed, path):
    told = []
    assert ed._load_session_path(str(path), confirmed=True,
                                 then=told.append) is None
    assert pump_until(ed, lambda: told, timeout=30), ed.status_var.get()
    return told[0]


def test_opening_a_project_again_reuses_its_copy(real_editor, tmp_path,
                                                 counted):
    ed, errors, _queued = real_editor
    arc, _sess, _names = _project(tmp_path)
    assert _open(ed, arc) is True
    folder = str(tmp_path / 'usb' / 'run')
    assert counted == [folder]
    assert _open(ed, arc) is True
    assert counted == [folder]                  # not unpacked again
    assert not os.path.exists(tmp_path / 'usb' / 'run (2)')
    assert ed.status_var.get() == (
        "Opened project run.flowproject from the copy already unpacked in "
        f"{folder}")
    assert ed._open_session_file() == os.path.join(folder, pa.SESSION_NAME)
    assert errors == []


@pytest.mark.parametrize('choice', ['copy', 'fresh', 'cancel'])
def test_a_copy_saved_since_is_asked_about(real_editor, tmp_path, counted,
                                           monkeypatch, choice):
    ed, errors, _queued = real_editor
    arc, _sess, _names = _project(tmp_path)
    sess, _ = pa.open_project_archive(str(arc), str(tmp_path / 'usb' / 'run'))
    with open(sess, 'a', encoding='utf-8') as fh:
        fh.write('\n')
    counted.clear()
    asked = []
    monkeypatch.setattr(ed, '_ask_choice',
                        lambda title, *a, **k: asked.append(title) or choice)
    opened = _open(ed, arc)
    assert asked == ["Project already unpacked"]
    if choice == 'copy':
        assert opened is True and counted == []
        assert ed._open_session_file() == sess
    elif choice == 'fresh':
        fresh = str(tmp_path / 'usb' / 'run (2)')
        assert opened is True and counted == [fresh]
        assert ed._open_session_file() == os.path.join(fresh, pa.SESSION_NAME)
    else:
        assert opened is False and counted == []
        assert not ed.__dict__.get('_session_file')
        assert ed.status_var.get() == 'Project run.flowproject was not opened.'
        assert not ed.__dict__.get('_project_opening')
    assert errors == []


def test_an_unpack_counts_its_megabytes(real_editor, tmp_path, counted,
                                        monkeypatch):
    ed, errors, _queued = real_editor
    arc, _sess, _names = _project(tmp_path)
    said = []
    monkeypatch.setattr(ed, '_busy', said.append)
    assert _open(ed, arc) is True
    form = re.compile(r'^Unpacking project run\.flowproject: (\d+) of '
                      r'(\d+) MB$')
    hits = [form.match(m) for m in said]
    assert said and all(hits), said
    assert hits[0].group(1) == '0'
    assert all(a != b for a, b in zip(said, said[1:], strict=False))   # only on a change
    assert hits[-1].group(1) == hits[-1].group(2) or int(
        hits[-1].group(2)) == 1        # a small project: under 1 MB
    assert errors == []


def test_a_project_refused_after_its_unpack_is_not_left_opening(
        real_editor, tmp_path, counted, monkeypatch):
    """The save question comes after the unpack; answered Cancel, the
    status line (counting megabytes until then) says it was not opened."""
    ed, errors, _queued = real_editor
    arc, _sess, _names = _project(tmp_path)
    # refused while the status line still counts the megabytes
    monkeypatch.setattr(
        ed, '_ok_to_replace_session',
        lambda path: ed.status_var.set(
            'Unpacking project run.flowproject: 1 of 1 MB') or False)
    told = []
    ed._load_session_path(str(arc), then=told.append)
    assert pump_until(ed, lambda: told, timeout=30)
    assert told == [False] and len(counted) == 1
    assert ed.status_var.get() == 'Project run.flowproject was not opened.'
    assert errors == []


def test_a_project_on_a_removable_drive_unpacks_internally(
        real_editor, tmp_path, counted, monkeypatch):
    ed, errors, _queued = real_editor
    monkeypatch.setattr(pc, 'on_removable_drive', lambda p: True)
    arc, _sess, _names = _project(tmp_path)
    assert _open(ed, arc) is True
    store = os.path.join(pc.store_dir(), 'run')
    assert counted == [store]
    assert not os.path.exists(tmp_path / 'usb' / 'run')
    assert ed.status_var.get() == (
        f"Opened project archive run.flowproject; its files are in {store}")
    assert errors == []


def _session_beside_archive(tmp_path):
    arc, sess, names = _project(tmp_path)
    out = tmp_path / 'usb' / 'copied.flowsession'
    shutil.copy(sess, out)
    shutil.rmtree(tmp_path / 'proj')
    return arc, out, names


@pytest.mark.parametrize('choice', ['project', 'project, unpacked before',
                                    'folder', 'cancel'])
def test_missing_samples_offer_the_project_beside_the_session(
        real_editor, tmp_path, counted, monkeypatch, choice):
    import openflo.editor_session as ES
    ed, errors, queued = real_editor
    arc, copied, names = _session_beside_archive(tmp_path)
    folder = str(tmp_path / 'usb' / 'run')
    monkeypatch.setattr(ed, '_offer_locate_missing', lambda: None)
    assert ed._load_session_path(str(copied), confirmed=True) is True
    assert sorted(ed._offline_samples) == sorted(names)
    before = choice == 'project, unpacked before'
    if before:
        # unpacked (to the internal folder) since the session opened: the
        # offer finds that copy and does not unpack again
        folder = os.path.join(pc.store_dir(), 'run')
        pa.open_project_archive(str(arc), folder)
        counted.clear()
        choice = 'project'
    asked, picked = [], []

    def ask(title, text, choices, **k):
        asked.append((title, [v for _l, v in choices]))
        return choices[0][1] if choice == 'project' else choice
    monkeypatch.setattr(ed, '_ask_choice', ask)
    monkeypatch.setattr(ES.filedialog, 'askdirectory',
                        lambda **k: picked.append(k) or '')
    type(ed)._offer_locate_missing(ed)
    # the one source offered: the copy when there is one, else the archive
    assert asked == [("Sample files not found",
                      [folder if before else str(arc), 'folder', 'cancel'])]
    if choice == 'project':
        assert pump_until(ed, lambda: not ed._offline_samples, timeout=30)
        assert counted == ([] if before else [folder])
        assert picked == []
        loaded = sorted(os.path.relpath(j[1], folder) for j in queued)
        assert loaded == [os.path.join('fcs', 'tube (2).fcs'),
                          os.path.join('fcs', 'tube.fcs')]
        assert pump_until(ed, lambda: ed.status_var.get().startswith(
            'Found all 2 missing'), timeout=10), ed.status_var.get()
    else:
        ed.update()
        assert sorted(ed._offline_samples) == sorted(names)
        assert counted == [] and queued == []
        assert len(picked) == (choice == 'folder')
    assert errors == []


def test_a_session_opens_with_a_copy_unpacked_internally(
        real_editor, tmp_path, counted):
    """A copy of the archive beside the session, unpacked to the internal
    folder (it was on a USB stick), is found when the session opens."""
    ed, errors, queued = real_editor
    arc, copied, _names = _session_beside_archive(tmp_path)
    folder = os.path.join(pc.store_dir(), 'run')
    pa.open_project_archive(str(arc), folder)
    asked = []
    ed._offer_locate_missing = lambda: asked.append(1)
    assert ed._load_session_path(str(copied), confirmed=True) is True
    ed.update()
    assert not ed.__dict__.get('_offline_samples')
    assert sorted(os.path.dirname(j[1]) for j in queued) == [
        os.path.join(folder, 'fcs')] * 2
    assert errors == []


def _other_days(tmp_path, n, old=True):
    """A session (day B) beside `n` unpacked projects of other days with
    the same sample and file names: p0/, p1/, …; `old`: saved before
    sessions held file identities (no fcs_id)."""
    from tests.test_project_copies import _project as proj
    b = tmp_path / 'B'
    b.mkdir()
    _arc, sess, names = proj(b)
    usb = tmp_path / 'stick'
    usb.mkdir()
    for i in range(n):
        d = tmp_path / f'other{i}'
        d.mkdir()
        arc, _s, _n = proj(d, seed=100 * (i + 1))
        pa.open_project_archive(str(arc), str(usb / f'p{i}'))
    out = usb / 'dayB.flowsession'
    import json
    data = json.loads(sess.read_text(encoding='utf-8'))
    for e in data['samples']:
        if old:
            e.pop('fcs_id')
    out.write_text(json.dumps(data), encoding='utf-8')
    shutil.rmtree(b / 'proj')
    return out, names


def test_another_days_files_are_not_offered_to_a_new_session(
        real_editor, tmp_path, counted, monkeypatch):
    """A session that saved its files' identities knows those files are
    other recordings: they are not offered; the folder question is."""
    import openflo.editor_session as ES
    ed, errors, queued = real_editor
    sess, _names = _other_days(tmp_path, 1, old=False)
    monkeypatch.setattr(ed, '_offer_locate_missing', lambda: None)
    ed._load_session_path(str(sess), confirmed=True)
    asked, yes = [], []
    monkeypatch.setattr(ed, '_ask_choice',
                        lambda *a, **k: asked.append(a) or 'cancel')
    monkeypatch.setattr(ES.messagebox, 'askyesno',
                        lambda *a, **k: yes.append(a) or False)
    type(ed)._offer_locate_missing(ed)
    assert asked == [] and len(yes) == 1 and queued == []
    assert errors == []


@pytest.mark.parametrize('n', [1, 2])
def test_another_days_project_is_offered_never_loaded(
        real_editor, tmp_path, counted, monkeypatch, n):
    ed, errors, queued = real_editor
    sess, names = _other_days(tmp_path, n)
    monkeypatch.setattr(ed, '_offer_locate_missing', lambda: None)
    assert ed._load_session_path(str(sess), confirmed=True) is True
    ed.update()
    assert sorted(ed._offline_samples) == sorted(names) and queued == []
    asked = []

    def ask(title, text, choices, default=0, **k):
        asked.append(([v for _l, v in choices], default, text))
        return 'cancel'
    monkeypatch.setattr(ed, '_ask_choice', ask)
    type(ed)._offer_locate_missing(ed)
    folders = [str(sess.parent / f'p{i}') for i in range(n)]
    (values, default, text), = asked
    assert values == folders + ['folder', 'cancel']
    assert '(2 of them)' in text and 'cannot tell' in text
    # one project: it is the default; two as likely: neither is
    assert default == (0 if n == 1 else len(values) - 1)
    assert queued == [] and errors == []


def test_a_project_chosen_loads_its_files(real_editor, tmp_path, counted,
                                         monkeypatch):
    ed, errors, queued = real_editor
    sess, _names = _other_days(tmp_path, 1)
    counted.clear()
    monkeypatch.setattr(ed, '_offer_locate_missing', lambda: None)
    ed._load_session_path(str(sess), confirmed=True)
    monkeypatch.setattr(ed, '_ask_choice',
                        lambda title, text, choices, **k: choices[0][1])
    type(ed)._offer_locate_missing(ed)
    assert pump_until(ed, lambda: not ed._offline_samples, timeout=30)
    folder = str(sess.parent / 'p0')
    assert sorted(os.path.dirname(j[1]) for j in queued) == [
        os.path.join(folder, 'fcs')] * 2
    assert counted == [] and errors == []


def test_a_session_opened_meanwhile_is_left_alone(real_editor, tmp_path,
                                                  counted, monkeypatch):
    """The user chose a project; while it unpacked, another session was
    opened. Nothing is loaded into that one."""
    ed, errors, queued = real_editor
    arc, copied, names = _session_beside_archive(tmp_path)
    monkeypatch.setattr(ed, '_offer_locate_missing', lambda: None)
    ed._load_session_path(str(copied), confirmed=True)
    # the other session is opened while the work runs: stand in by
    # swapping the offline record before the result is delivered
    held = ed._offline_samples
    ed._load_missing_from_project(str(arc))
    ed._offline_samples = dict(held)
    assert pump_until(ed, lambda: 'another session' in ed.status_var.get(),
                      timeout=30), ed.status_var.get()
    assert queued == [] and sorted(ed._offline_samples) == sorted(names)
    assert errors == []


def test_samples_found_in_their_project_are_said(real_editor, tmp_path,
                                                 counted):
    ed, errors, _queued = real_editor
    arc, copied, _names = _session_beside_archive(tmp_path)
    pa.open_project_archive(str(arc), str(tmp_path / 'usb' / 'run'))
    statuses = []
    real_set = ed.status_var.set
    ed.status_var.set = lambda v: (statuses.append(v), real_set(v))
    ed._offer_locate_missing = lambda: None
    assert ed._load_session_path(str(copied), confirmed=True) is True
    assert [m for m in statuses if m.startswith('Loading session: 2 ') and
            '2 found in project run (the same recordings)' in m], statuses
    assert errors == []


def test_an_archive_chosen_reuses_its_copy(real_editor, tmp_path, counted,
                                           monkeypatch):
    """Loading from an archive whose copy was unpacked before reads that
    copy: nothing is unpacked again."""
    ed, errors, queued = real_editor
    arc, copied, _names = _session_beside_archive(tmp_path)
    monkeypatch.setattr(ed, '_offer_locate_missing', lambda: None)
    ed._load_session_path(str(copied), confirmed=True)
    folder = os.path.join(pc.store_dir(), 'run')
    pa.open_project_archive(str(arc), folder)
    counted.clear()
    ed._load_missing_from_project(str(arc))
    assert pump_until(ed, lambda: not ed._offline_samples, timeout=30)
    assert counted == []
    assert sorted(os.path.dirname(j[1]) for j in queued) == [
        os.path.join(folder, 'fcs')] * 2
    assert errors == []
