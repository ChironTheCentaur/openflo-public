"""Session and project files open from a file drop and from the command line.

A .flowsession or .flowproject dropped onto the editor used to be ignored,
because the drop only looked for FCS files and workspaces; and the window ignored a file
named on its command line, so a double-clicked session could not open in it.
"""
from __future__ import annotations

import os
import re
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from openflo.paths import expand_dropped_paths, is_session_file
from openflo.project_archive import MANIFEST
from tests._tk_wait import pump_until

REPO = Path(__file__).resolve().parents[1]


def _touch(p, data=b'x'):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _archive_zip(p):
    with zipfile.ZipFile(p, 'w') as zf:
        zf.writestr(MANIFEST, '{}')
    return p


# ── paths: telling session files apart ─────────────────────────────────────

def test_is_session_file(tmp_path):
    assert is_session_file('a.flowsession')
    assert is_session_file('B.FLOWPROJECT')
    assert not is_session_file('a.fcs')
    assert not is_session_file('a.wsp')
    assert is_session_file(_archive_zip(tmp_path / 'old.zip'))
    with zipfile.ZipFile(tmp_path / 'other.zip', 'w') as zf:
        zf.writestr('readme.txt', 'not a project')
    assert not is_session_file(tmp_path / 'other.zip')


def test_dropped_session_files_are_collected_in_drop_order(tmp_path):
    s2 = _touch(tmp_path / 'b.flowsession')
    s1 = _touch(tmp_path / 'a b {c}.flowproject')
    fcs = _touch(tmp_path / 'x.fcs')
    sessions: list[str] = []
    got = expand_dropped_paths([str(s2), str(fcs), f'"{s1}"', str(s2)],
                               sessions=sessions)
    assert sessions == [str(s2), str(s1)]
    assert got == ([str(fcs)], [])


def test_a_session_inside_a_dropped_folder_is_not_collected(tmp_path):
    # a trial folder's own autosave or archive must not replace its samples
    _touch(tmp_path / 'trial' / 'a.fcs')
    _touch(tmp_path / 'trial' / 'old.flowsession')
    sessions: list[str] = []
    fcs, _wsp = expand_dropped_paths([str(tmp_path / 'trial')],
                                     sessions=sessions)
    assert sessions == []
    assert [os.path.basename(p) for p in fcs] == ['a.fcs']


def test_without_the_sessions_list_a_session_file_is_ignored(tmp_path):
    # the command-line group reader passes no list: unchanged for it
    s = _touch(tmp_path / 'a.flowsession')
    assert expand_dropped_paths([str(s)]) == ([], [])


# ── the editor: a drop opens the session ───────────────────────────────────

@pytest.fixture
def editor(monkeypatch, tmp_path):
    from tests.conftest import editor_or_skip
    from tests.test_transform_records import isolate_home
    isolate_home(monkeypatch, tmp_path)
    root, ed = editor_or_skip()
    opened, queued = [], []
    def load(path, confirmed=False, then=None):
        opened.append(path)
        if then is not None:
            then(True)
        return True
    monkeypatch.setattr(ed, '_load_session_path', load)
    monkeypatch.setattr(ed, '_enqueue_load',
                        lambda job, priority=1: queued.append(job))
    yield ed, opened, queued
    root.destroy()


def test_dropping_a_session_opens_it(editor, tmp_path):
    ed, opened, queued = editor
    s = _touch(tmp_path / 'my run {1}' / 'day 4.flowsession')
    # tkdnd's file list: a path with spaces or braces arrives braced
    event = SimpleNamespace(data='{' + str(s) + '}', action='copy')
    assert ed._on_dnd_drop(event) == 'copy'
    # not inside the drop handler (Windows waits on it): once it returned
    assert opened == []
    ed.update()
    assert opened == [str(s)]
    assert queued == []


def test_dropping_a_project_archive_opens_it(editor, tmp_path):
    ed, opened, _queued = editor
    p = _touch(tmp_path / 'run.flowproject')
    ed._import_dropped_paths([str(p)])
    ed.update()
    assert opened == [str(p)]


def test_a_session_dropped_with_samples_opens_alone(editor, tmp_path):
    ed, opened, queued = editor
    s1 = _touch(tmp_path / 'one.flowsession')
    s2 = _touch(tmp_path / 'two.flowproject')
    f = _touch(tmp_path / 'a.fcs')
    ed._import_dropped_paths([str(f), str(s1), str(s2)])
    ed.update()
    assert opened == [str(s1)]
    assert queued == []
    assert ed.status_var.get() == (
        'Opened one.flowsession. A session opens on its own, so 2 other '
        'dropped file(s) were not loaded.')


def test_a_lone_session_drop_leaves_the_status_to_the_open(editor, tmp_path):
    ed, _opened, _queued = editor
    ed.status_var.set('before')
    ed._import_dropped_paths([str(_touch(tmp_path / 'a.flowsession'))])
    ed.update()
    assert ed.status_var.get() == 'before'


def test_a_drop_with_nothing_to_open_says_so(editor, tmp_path):
    ed, opened, queued = editor
    ed._import_dropped_paths([str(_touch(tmp_path / 'notes.txt'))])
    ed.update()
    assert opened == [] and queued == []
    assert 'sessions' in ed.status_var.get()


# ── a dropped session that does not open (the real open, not a stand-in) ──

@pytest.fixture
def real_editor(monkeypatch, tmp_path):
    """An editor whose session open is the real one; error dialogs are
    recorded instead of shown, and sample loads are recorded, not run."""
    import openflo.editor_session as ES
    from tests.conftest import editor_or_skip
    from tests.test_transform_records import isolate_home
    isolate_home(monkeypatch, tmp_path)
    root, ed = editor_or_skip()
    errors, queued = [], []
    monkeypatch.setattr(ES, 'show_error',
                        lambda parent, title, *a, **k: errors.append(title))
    monkeypatch.setattr(ES.messagebox, 'showerror',
                        lambda title, *a, **k: errors.append(title))
    monkeypatch.setattr(ed, '_enqueue_load',
                        lambda job, priority=1: queued.append(job))
    # a question would wait for a click that never comes: record and cancel
    monkeypatch.setattr(ed, '_ask_choice',
                        lambda title, *a, **k: errors.append(title) or 'cancel')
    # run_async as the app runs it (work on a thread), answered by polling
    # because these tests pump update() rather than mainloop
    from tests.test_tk_loop_freezes import _run_async_off_the_tk_thread
    _run_async_off_the_tk_thread(ed)
    yield ed, errors, queued
    root.destroy()


def test_a_broken_session_dropped_with_samples_says_why(real_editor,
                                                        tmp_path):
    ed, errors, queued = real_editor
    bad = _touch(tmp_path / 'bad.flowsession', b'{"format": ')
    f = _touch(tmp_path / 'a.fcs')
    ed._import_dropped_paths([str(bad), str(f)])
    ed.update()
    status = ed.status_var.get()
    assert status.startswith('Load session failed'), status
    assert 'Opened' not in status and 'not loaded' not in status
    assert errors == ["Session can't be read"]
    assert queued == []


def test_a_cancelled_session_drop_leaves_the_status_as_it_was(
        real_editor, monkeypatch, tmp_path):
    ed, errors, queued = real_editor
    asked = []
    monkeypatch.setattr(ed, '_ok_to_replace_session',
                        lambda path: asked.append(path) and False)
    s = _touch(tmp_path / 's.flowsession', b'{}')
    f = _touch(tmp_path / 'a.fcs')
    ed.status_var.set('before')
    ed._import_dropped_paths([str(s), str(f)])
    ed.update()
    assert asked == [str(s)]
    assert ed.status_var.get() == 'before'
    assert errors == [] and queued == []


def test_a_session_that_opens_keeps_its_note_and_counts_the_rest(
        real_editor, tmp_path):
    ed, errors, queued = real_editor
    s = tmp_path / 'empty.flowsession'
    ed._write_session(str(s))
    ed._import_dropped_paths([str(s), str(_touch(tmp_path / 'a.fcs'))])
    ed.update()
    assert errors == [] and queued == []
    assert ed._open_session_file() == str(s)
    # the open's own progress line stays, with the note after it
    note = 'A session opens on its own, so 1 other dropped file(s) were ' \
        'not loaded.'
    status = ed.status_var.get()
    assert status.endswith('. ' + note) and len(status) > len(note) + 2, \
        status


def test_a_drop_adds_its_note_after_the_opens_own_message(
        real_editor, monkeypatch, tmp_path):
    ed, _errors, _queued = real_editor
    real = ed._load_session_path

    def load(path, confirmed=False, then=None):
        ok = real(path, confirmed)
        ed.status_var.set('Opened a session from an older OpenFlo.')
        then(ok)
        return ok
    monkeypatch.setattr(ed, '_load_session_path', load)
    s = tmp_path / 'old.flowsession'
    ed._write_session(str(s))
    ed._import_dropped_paths([str(s), str(_touch(tmp_path / 'b.fcs'))])
    ed.update()
    assert ed.status_var.get() == (
        'Opened a session from an older OpenFlo. A session opens on its '
        'own, so 1 other dropped file(s) were not loaded.')


def test_a_project_whose_session_does_not_read_is_not_called_opened(
        real_editor, monkeypatch, tmp_path):
    import openflo.project_archive as PA
    ed, errors, _queued = real_editor
    bad = _touch(tmp_path / 'unpacked' / 'run.flowsession', b'not json')
    monkeypatch.setattr(PA, 'open_project_archive',
                        lambda path, dest=None, **k: (str(bad), {}))
    proj = _archive_zip(tmp_path / 'run.flowproject')
    told = []
    assert ed._load_session_path(str(proj), then=told.append) is None
    assert pump_until(ed, lambda: told, timeout=10)
    assert told == [False]
    assert ed.status_var.get().startswith('Load session failed')
    assert errors == ["Session can't be read"]


def test_a_project_that_does_not_unpack_is_reported(real_editor,
                                                    monkeypatch, tmp_path):
    import openflo.project_archive as PA
    ed, errors, _queued = real_editor

    tries = []

    def broken(path, dest=None, **k):
        tries.append(path)
        raise OSError('the stick was pulled out')
    monkeypatch.setattr(PA, 'open_project_archive', broken)
    proj = _archive_zip(tmp_path / 'run.flowproject')
    for n in (1, 2):                 # a failed open does not block the next
        told = []
        ed._load_session_path(str(proj), then=told.append)
        assert pump_until(ed, lambda t=told: t, timeout=10)
        assert told == [False] and len(tries) == n
        assert ed.status_var.get() == (
            'Open project failed: the stick was pulled out')
    assert errors == ["Can't open project"] * 2


@pytest.mark.parametrize('case', ['open elsewhere', 'not a session',
                                  'from a newer OpenFlo'])
def test_a_refused_open_in_a_drop_is_not_called_opened(
        real_editor, monkeypatch, tmp_path, case):
    """Each way the open is refused after the save question (Cancel on
    "open elsewhere", no format marker, a newer schema) leaves the drop's
    status line as it was: no "Opened", no note on the other files."""
    import json

    import openflo.editor_session as ES
    from openflo.session_format import SESSION_VERSION
    ed, errors, queued = real_editor
    data = {'format': 'openflo-session', 'version': SESSION_VERSION + 1}
    title = "Can't open session"
    if case == 'open elsewhere':
        monkeypatch.setattr(ES._lock, 'acquire',
                            lambda *a, **k: {'pid': 1, 'host': 'other'})
        data['version'] = SESSION_VERSION
        title = "Session open elsewhere"
    elif case == 'not a session':
        data = {'samples': []}
        title = "Not a session file"
    s = _touch(tmp_path / 's.flowsession', json.dumps(data).encode())
    told = []
    real = ed._load_session_path
    monkeypatch.setattr(
        ed, '_load_session_path',
        lambda path, confirmed=False, then=None: real(
            path, confirmed, lambda ok: (told.append(ok), then(ok))))
    ed.status_var.set('before')
    ed._import_dropped_paths([str(s), str(_touch(tmp_path / 'a.fcs'))])
    ed.update()
    assert told == [False]
    assert errors == [title]
    assert ed.status_var.get() == 'before'
    assert not ed.__dict__.get('_session_file') and queued == []


def test_a_slow_project_unpack_keeps_the_window_answering(
        real_editor, monkeypatch, tmp_path):
    """A big archive on a slow USB stick took minutes to unpack on the Tk
    thread, inside the drop: Windows reported the window hung. The unpack
    now runs in the background and the drop returns at once."""
    import openflo.project_archive as PA
    ed, errors, queued = real_editor
    sess = tmp_path / 'unpacked' / 'run.flowsession'
    sess.parent.mkdir()
    ed._write_session(str(sess))
    started, release = threading.Event(), threading.Event()

    def slow(path, dest=None, **k):
        started.set()
        assert release.wait(30)
        return str(sess), {}
    monkeypatch.setattr(PA, 'open_project_archive', slow)
    proj = _archive_zip(tmp_path / 'run.flowproject')
    ed.status_var.set('before')
    ed._import_dropped_paths([str(proj), str(_touch(tmp_path / 'a.fcs'))])
    assert not started.is_set()           # nothing ran inside the drop
    assert ed.status_var.get() == 'before'
    ticks = []

    def tick():
        ticks.append(1)
        ed.after(10, tick)
    tick()
    try:
        assert pump_until(ed, lambda: started.is_set() and len(ticks) > 20,
                          timeout=10)
        assert not ed.__dict__.get('_session_file')
        assert ed.status_var.get() == 'Opening project run.flowproject…'
    finally:
        release.set()
    note = ('A session opens on its own, so 1 other dropped file(s) were '
            'not loaded.')
    assert pump_until(ed, lambda: ed.status_var.get().endswith(note),
                      timeout=10), ed.status_var.get()
    assert ed._open_session_file() == str(sess)
    assert ed.status_var.get().startswith(
        'Opened project archive run.flowproject; its files are in ')
    assert errors == [] and queued == []


def _held_unpack(monkeypatch, sess):
    """open_project_archive stands in as an unpack that waits until
    released; returns (started, release, calls)."""
    import openflo.project_archive as PA
    started, release, calls = threading.Event(), threading.Event(), []

    def slow(path, dest=None, **k):
        calls.append(path)
        started.set()
        assert release.wait(30)
        return str(sess), {}
    monkeypatch.setattr(PA, 'open_project_archive', slow)
    return started, release, calls


@pytest.mark.parametrize('answer', [True, False])
def test_the_save_question_comes_after_the_unpack(
        real_editor, monkeypatch, tmp_path, answer):
    """The window keeps answering while a project unpacks, so whatever is
    done meanwhile must be asked about: the offer to save comes once the
    unpack is done, not before it starts."""
    ed, errors, _queued = real_editor
    sess = tmp_path / 'unpacked' / 'run.flowsession'
    sess.parent.mkdir()
    ed._write_session(str(sess))
    started, release, _calls = _held_unpack(monkeypatch, sess)
    asked = []
    monkeypatch.setattr(ed, '_ok_to_replace_session',
                        lambda path: asked.append(path) or answer)
    told = []
    proj = _archive_zip(tmp_path / 'run.flowproject')
    assert ed._load_session_path(str(proj), then=told.append) is None
    try:
        assert pump_until(ed, started.is_set, timeout=10)
        ed.update()
        assert asked == [] and told == []
    finally:
        release.set()
    assert pump_until(ed, lambda: told, timeout=10)
    assert asked == [str(sess)]
    assert told == [answer]
    if answer:
        assert ed._open_session_file() == str(sess)
    else:
        assert not ed.__dict__.get('_session_file')
        # not left saying it is still opening
        assert ed.status_var.get() == 'Project run.flowproject was not opened.'
    assert errors == []


def test_a_second_project_waits_for_the_first(real_editor, monkeypatch,
                                              tmp_path):
    ed, errors, _queued = real_editor
    sess = tmp_path / 'unpacked' / 'run.flowsession'
    sess.parent.mkdir()
    ed._write_session(str(sess))
    started, release, calls = _held_unpack(monkeypatch, sess)
    first = _archive_zip(tmp_path / 'run.flowproject')
    second = _archive_zip(tmp_path / 'other.flowproject')
    told1, told2 = [], []
    ed._load_session_path(str(first), then=told1.append)
    try:
        assert pump_until(ed, started.is_set, timeout=10)
        assert ed._load_session_path(str(second), then=told2.append) is False
        assert told2 == [False]
        assert ed.status_var.get() == (
            'A project is still opening (run.flowproject): wait for it to '
            'open, then open other.flowproject.')
    finally:
        release.set()
    assert pump_until(ed, lambda: told1, timeout=10)
    assert told1 == [True] and calls == [str(first)]
    # once it has opened, another may open
    told3 = []
    ed._load_session_path(str(second), then=told3.append)
    assert pump_until(ed, lambda: told3, timeout=10)
    assert told3 == [True] and calls == [str(first), str(second)]
    assert errors == []


def test_a_refused_session_in_a_project_does_not_stay_opening(
        real_editor, monkeypatch, tmp_path):
    ed, errors, queued = real_editor
    bad = _touch(tmp_path / 'unpacked' / 'run.flowsession',
                 b'{"samples": []}')
    _started, release, _calls = _held_unpack(monkeypatch, bad)
    release.set()
    proj = _archive_zip(tmp_path / 'run.flowproject')
    ed._import_dropped_paths([str(proj), str(_touch(tmp_path / 'a.fcs'))])
    assert pump_until(ed, lambda: errors, timeout=10)
    ed.update()
    assert errors == ["Not a session file"]
    assert ed.status_var.get() == 'Project run.flowproject was not opened.'
    assert queued == []


def test_a_deeply_nested_session_is_reported_not_raised(real_editor,
                                                        tmp_path):
    ed, errors, _queued = real_editor
    deep = _touch(tmp_path / 'deep.flowsession', b'[' * 100_000)
    assert ed._load_session_path(str(deep)) is False
    assert ed.status_var.get().startswith('Load session failed')
    assert errors == ["Session can't be read"]


# ── the command line: a double-clicked file opens on start ─────────────────

def test_command_line_paths_keeps_only_existing_ones(tmp_path):
    from openflo.gui import _command_line_paths
    s = _touch(tmp_path / 'a b.flowsession')
    assert _command_line_paths(['', str(s), str(tmp_path / 'gone.fcs'),
                                '--flag']) == [str(s)]


@pytest.mark.parametrize('skip', [True, False])
def test_a_file_to_open_skips_the_resume_offer(editor, monkeypatch, skip):
    ed, _opened, _queued = editor
    looked = []
    monkeypatch.setattr(ed, '_find_resumable_session',
                        lambda: looked.append(True))
    ed._skip_resume_offer = skip
    ed._offer_resume_session()
    assert looked == ([] if skip else [True])


def test_main_opens_the_file_named_on_its_command_line(monkeypatch, tmp_path):
    from tests.conftest import gui_unavailable
    from tests.test_transform_records import isolate_home
    isolate_home(monkeypatch, tmp_path)
    try:
        import tkinter as tk
        tk.Tk().destroy()
    except Exception as e:  # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f'Tk cannot initialise without a display: {e}')
    import openflo.gui as G
    monkeypatch.setattr(threading, 'excepthook', threading.excepthook)
    s = _touch(tmp_path / 'my sessions' / 'day 4.flowsession')
    imported, offered, welcomed = [], [], []
    monkeypatch.setattr(G.ViewGateEditorWindow, '_import_dropped_paths',
                        lambda self, paths: imported.append(list(paths)))
    monkeypatch.setattr(G.ViewGateEditorWindow, '_maybe_welcome',
                        lambda self: welcomed.append(True))
    real_offer = G.ViewGateEditorWindow._offer_resume_session

    def offer(self):
        offered.append(self._skip_resume_offer)
        return real_offer(self)
    monkeypatch.setattr(G.ViewGateEditorWindow, '_offer_resume_session',
                        offer)

    def pump_then_close(self, n=0):
        pump_until(self, lambda: imported, timeout=20)
        self.destroy()
    monkeypatch.setattr(G.tk.Misc, 'mainloop', pump_then_close)

    assert G.main(['--not-a-file', str(s)]) == 0
    assert imported == [[str(s)]]
    assert welcomed == []
    assert offered == [True]


# ── the installer and setup.bat register the two file types ────────────────

_TYPES = (('.flowsession', 'OpenFlo.Session'),
          ('.flowproject', 'OpenFlo.Project'))


def test_installer_registers_and_removes_the_file_types():
    nsi = (REPO / 'packaging' / 'windows' / 'openflo.nsi').read_text()
    for ext, progid in _TYPES:
        assert f'!insertmacro REGISTER_TYPE "{ext}" "{progid}"' in nsi
        assert f'!insertmacro UNREGISTER_TYPE "{ext}" "{progid}"' in nsi
    macro = nsi[nsi.index('!macro REGISTER_TYPE'):]
    macro = macro[:macro.index('!macroend')]
    assert re.search(r'shell\\open\\command" "" \'"\$INSTDIR\\python\\'
                     r'pythonw\.exe" -P -m \$\{GUI_MODULE\} "%1"\'', macro)
    assert 'WriteRegStr HKCU "${CLASSES}\\${EXT}" "" "${PROGID}"' in macro
    assert '!define CLASSES "Software\\Classes"' in nsi
    assert 'HKLM' not in macro


def test_the_uninstaller_removes_only_what_still_points_at_openflo():
    """OpenFlo's own ProgID always goes; the extension's default value (and
    then its key, if empty) only while it still names that ProgID, so a
    program that has taken the file type over since keeps it."""
    nsi = (REPO / 'packaging' / 'windows' / 'openflo.nsi').read_text()
    macro = nsi[nsi.index('!macro UNREGISTER_TYPE EXT PROGID'):]
    macro = macro[:macro.index('!macroend')]
    lines = [ln.strip() for ln in macro.splitlines()[1:]
             if ln.strip() and not ln.strip().startswith(';')]
    assert lines == [
        'DeleteRegKey HKCU "${CLASSES}\\${PROGID}"',
        'ReadRegStr $0 HKCU "${CLASSES}\\${EXT}" ""',
        '${If} $0 == "${PROGID}"',
        'DeleteRegValue HKCU "${CLASSES}\\${EXT}" ""',
        'DeleteRegKey /ifempty HKCU "${CLASSES}\\${EXT}"',
        '${EndIf}',
    ]
    # the macro runs in the uninstall section, after the files are gone
    uninst = nsi[nsi.index('Section "Uninstall"'):]
    for ext, progid in _TYPES:
        assert f'!insertmacro UNREGISTER_TYPE "{ext}" "{progid}"' in uninst


def test_setup_bat_registers_the_file_types_for_the_launcher():
    bat = (REPO / 'setup.bat').read_text()
    # an interactive setup registers them; "setup.bat filetypes" only that
    assert 'if /i not "%MODE%"=="auto" call :register_file_types\n' in bat
    block = ('if /i "%MODE%"=="filetypes" (\n'
             '    call :register_file_types\n'
             '    endlocal\n'
             '    exit /b 0\n'
             ')\n')
    assert block in bat
    # ...and stops there, before the first install step
    assert bat.index(block) < bat.index('set "VPY=')
    body = bat[bat.index('\n:register_file_types\n'):]
    assert 'HKCU:\\Software\\Classes' in body
    for ext, progid in _TYPES:
        assert f"@('{ext}', '{progid}'," in body
    # %%1 in a .bat is the literal %1 the shell replaces with the file
    assert "'openflo-gui.bat'" in body and "'%%1'" in body


def test_launcher_passes_the_file_on():
    bat = (REPO / 'openflo-gui.bat').read_text()
    assert 'set "OPENFLO_OPEN=%~1"' in bat
    assert '$a += $q + $env:OPENFLO_OPEN + $q' in bat
    assert '-ArgumentList $a ' in bat
