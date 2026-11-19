"""The desktop first run: a welcome window on the first launch, the bundled
sample it opens, and the getting-started tour.

* The welcome shows once: on the first launch of openflo-gui (gui.main),
  not after it was shown or turned off, not while a test runs, not with
  OPENFLO_NO_WELCOME set, not over samples already open.
* Help, Getting started tour and Help, Open the sample data reopen what it
  offers.
* The sample ships in the package (pyproject package-data; the sdist check
  is in tests/test_sdist_contents.py), opens as two gated samples, and is
  unpacked under ~/.openflo/sample, never beside the installed copy. The
  shipped copy is the one scripts/make_desktop_sample.py writes.
* The tour steps forward and back, points at a real editor widget at each
  step and closes on Done; it stays beside that widget when the editor
  moves after it was placed (macOS settles a window late).

Generated data only (openflo.tutorial_sample); HOME is a temporary folder
(tests/conftest.py), so the prefs file written here is the test's own.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import tkinter.messagebox

import pytest

from openflo import first_run as fr
from openflo import prefs
from openflo.document import Document
from openflo.tutorial_sample import GATES
from tests._tk_wait import pump_until, settle
from tests.conftest import editor_or_skip


@contextlib.contextmanager
def _user_state_kept():
    """Put the test home's prefs file and sample folder back as they were.

    The test home is shared by every test in a worker. Opening the sample
    records its copy under File > Open recent, and the copy stays on disk,
    so a later test that walks the menus (test_windows_nothing_hidden)
    invoked that entry over its own samples, which asks first in a dialog
    that waits for an answer: the run hung there."""
    path = prefs._prefs_path()
    try:
        with open(path, 'rb') as f:
            saved = f.read()
    except FileNotFoundError:
        saved = None
    root = fr.sample_root()
    copies = set(os.listdir(root)) if os.path.isdir(root) else set()
    try:
        yield
    finally:
        if saved is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            with open(path, 'wb') as f:
                f.write(saved)
        if os.path.isdir(root):
            for name in set(os.listdir(root)) - copies:
                shutil.rmtree(os.path.join(root, name))


@pytest.fixture(autouse=True)
def user_state_kept():
    """Every test here leaves the shared test home as it found it."""
    with _user_state_kept():
        yield


@pytest.fixture
def clean_prefs():
    """No welcome flag stored, and the test's own run not hiding it."""
    path = prefs._prefs_path()
    if os.path.exists(path):
        os.remove(path)
    yield
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def first_launch(monkeypatch, clean_prefs):
    """As on a first launch outside pytest: no flag, nothing to hide it.
    Call it in the test body: pytest sets PYTEST_CURRENT_TEST again for
    the call phase, after the fixtures ran."""
    def launch():
        monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
        monkeypatch.delenv(fr.NO_WELCOME_ENV, raising=False)
    return launch


@pytest.fixture
def ed():
    root, editor = editor_or_skip()
    try:
        yield root, editor
    finally:
        root.destroy()


def _windows(editor, cls):
    return [w for w in editor.winfo_children()
            if isinstance(w, cls) and w.winfo_exists()]


def _open_sample(root, editor, **kw):
    """Help, Open the sample data, its loads run on this thread."""
    jobs = []
    editor._queue_fcs_loads = lambda paths, front_names=(): jobs.extend(paths)
    ok = editor._open_sample_data(**kw)
    for job in jobs:
        editor._load_worker(editor._sample_name_for(job), job)
        root.update()
    assert pump_until(root, lambda: len(editor._samples) == len(jobs))
    return ok


# ── the flag ───────────────────────────────────────────────────────────────

def test_the_welcome_shows_only_until_it_was_shown_or_hidden():
    assert fr.should_welcome({}, {}) is True
    assert fr.should_welcome({'theme': 'dark'}, {}) is True
    assert fr.should_welcome([], {}) is True            # a damaged prefs file
    assert fr.should_welcome({fr.PREF_KEY: True}, {}) is False
    assert fr.should_welcome({}, {fr.NO_WELCOME_ENV: '1'}) is False
    assert fr.should_welcome({}, {'PYTEST_CURRENT_TEST': 'x'}) is False


def test_showing_it_is_stored_with_the_other_prefs(ed, clean_prefs):
    root, editor = ed
    prefs.write_pref('theme', 'dark')
    assert fr.should_welcome(prefs.read_prefs(), {}) is True
    editor._show_welcome().destroy()
    root.update()
    assert fr.should_welcome(prefs.read_prefs(), {}) is False
    assert prefs.read_prefs()['theme'] == 'dark'


def test_a_test_run_never_sees_the_welcome(ed, clean_prefs):
    _root, editor = ed
    assert fr.should_welcome({}) is False                # PYTEST_CURRENT_TEST
    assert editor._maybe_welcome() is None


# ── the welcome window ─────────────────────────────────────────────────────

def test_the_welcome_shows_once(ed, first_launch):
    from openflo.ui_first_run import WelcomeWindow
    root, editor = ed
    first_launch()
    win = editor._maybe_welcome()
    assert isinstance(win, WelcomeWindow)
    win.destroy()                                       # the window's X
    root.update()
    assert editor._maybe_welcome() is None
    assert _windows(editor, WelcomeWindow) == []


def test_dont_show_again_closes_it_for_good(ed, first_launch):
    from openflo.ui_first_run import WelcomeWindow
    root, editor = ed
    first_launch()
    win = editor._maybe_welcome()
    win.dismiss_btn.invoke()
    root.update()
    assert not win.winfo_exists()
    assert prefs.read_prefs().get(fr.PREF_KEY) is True
    assert editor._maybe_welcome() is None
    assert _windows(editor, WelcomeWindow) == []


def test_no_welcome_over_open_samples_or_with_the_switch(ed, first_launch,
                                                         monkeypatch):
    _root, editor = ed
    first_launch()
    monkeypatch.setenv(fr.NO_WELCOME_ENV, '1')
    assert editor._maybe_welcome() is None
    monkeypatch.delenv(fr.NO_WELCOME_ENV)
    monkeypatch.setitem(editor._samples, 'x', object())
    assert editor._maybe_welcome() is None
    assert prefs.read_prefs().get(fr.PREF_KEY) is None   # still to come


def test_the_welcome_buttons_open_the_sample_or_files_then_the_tour(
        ed, first_launch, monkeypatch):
    _root, editor = ed
    calls = []
    monkeypatch.setattr(type(editor), '_open_sample_data',
                        lambda self, tour=False: calls.append(('sample', tour)))
    monkeypatch.setattr(type(editor), '_add_samples',
                        lambda self: calls.append(('files',)))
    monkeypatch.setattr(type(editor), '_start_tour',
                        lambda self, start=0: calls.append(('tour',)))
    win = editor._show_welcome()
    win.sample_btn.invoke()
    assert not win.winfo_exists()
    win = editor._show_welcome()
    win.files_btn.invoke()
    assert not win.winfo_exists()
    assert calls == [('sample', True), ('files',), ('tour',)]


def test_the_first_launch_shows_the_welcome(first_launch, monkeypatch):
    """gui.main schedules it once the window is up."""
    first_launch()
    try:
        import tkinter as tk
        tk.Tk().destroy()
    except Exception as e:  # noqa: BLE001 - any Tk start failure means no GUI
        from tests.conftest import gui_unavailable
        gui_unavailable(f'Tk cannot initialise: {e}')
    from openflo import gui
    from openflo.ui_first_run import WelcomeWindow
    made = []
    make_root = gui._make_root

    def root_without_loop():
        root = make_root()
        root.mainloop = lambda *a: None
        made.append(root)
        return root
    monkeypatch.setattr(gui, '_make_root', root_without_loop)
    monkeypatch.setattr(threading, 'excepthook', threading.excepthook)
    assert gui.main([]) == 0  # [] : not pytest's own command line
    (root,) = made
    try:
        (editor,) = [w for w in root.winfo_children()
                     if isinstance(w, gui.ViewGateEditorWindow)]
        assert pump_until(root, lambda: _windows(editor, WelcomeWindow),
                          timeout=10)
    finally:
        root.destroy()


def _launch_with_an_autosave(monkeypatch, tmp_path, resume):
    """gui.main on a first launch where an autosave from the last session
    waits (an existing user's first launch of this version). The user
    takes a second to answer "Resume last session?", and answers `resume`;
    Tk keeps running timers meanwhile, as the question's own wait does.
    Returns (root, editor, seen): seen['welcome_during_question'] is
    whether a welcome window was open while the question was."""
    try:
        import tkinter as tk
        tk.Tk().destroy()
    except Exception as e:  # noqa: BLE001 - any Tk start failure means no GUI
        from tests.conftest import gui_unavailable
        gui_unavailable(f'Tk cannot initialise: {e}')
    from openflo import gui
    from openflo.ui_first_run import WelcomeWindow
    auto = tmp_path / 'autosave.flowsession'
    auto.write_text(json.dumps({'format': 'openflo-session',
                                'created': 'yesterday',
                                'samples': [{'name': 'S1'}]}))
    monkeypatch.setattr(gui.ViewGateEditorWindow, '_find_resumable_session',
                        lambda self: str(auto))
    applied = []
    monkeypatch.setattr(gui.ViewGateEditorWindow, '_apply_session',
                        lambda self, data: applied.append(data))
    seen = {}

    def askyesno(title, msg, parent=None, **k):
        pump_until(parent.master, lambda: False, timeout=1.0)
        seen['welcome_during_question'] = bool(
            _windows(parent, WelcomeWindow))
        return resume
    monkeypatch.setattr(tkinter.messagebox, 'askyesno', askyesno)
    made = []
    make_root = gui._make_root

    def root_without_loop():
        root = make_root()
        root.mainloop = lambda *a: None
        made.append(root)
        return root
    monkeypatch.setattr(gui, '_make_root', root_without_loop)
    monkeypatch.setattr(threading, 'excepthook', threading.excepthook)
    assert gui.main([]) == 0  # [] : not pytest's own command line
    (root,) = made
    (editor,) = [w for w in root.winfo_children()
                 if isinstance(w, gui.ViewGateEditorWindow)]
    assert pump_until(root, lambda: 'welcome_during_question' in seen,
                      timeout=10)
    assert bool(applied) is resume
    return root, editor, seen


def test_the_welcome_waits_for_resume_last_session(first_launch,
                                                   monkeypatch, tmp_path):
    """Not over the open question; once it is answered No, it shows."""
    from openflo.ui_first_run import WelcomeWindow
    first_launch()
    root, editor, seen = _launch_with_an_autosave(monkeypatch, tmp_path,
                                                  resume=False)
    try:
        assert seen['welcome_during_question'] is False
        assert pump_until(root, lambda: _windows(editor, WelcomeWindow),
                          timeout=10)
    finally:
        root.destroy()


def test_no_welcome_over_a_resumed_session(first_launch, monkeypatch,
                                           tmp_path):
    """Answered Yes: the resumed session's samples are still loading, so
    the editor looks empty, but the welcome stays away (and is still to
    come on a later launch)."""
    from openflo.ui_first_run import WelcomeWindow
    first_launch()
    root, editor, seen = _launch_with_an_autosave(monkeypatch, tmp_path,
                                                  resume=True)
    try:
        assert seen['welcome_during_question'] is False
        assert not editor._samples
        settle(root, 0.5)
        assert _windows(editor, WelcomeWindow) == []
        assert prefs.read_prefs().get(fr.PREF_KEY) is None
    finally:
        root.destroy()


def test_what_waits_for_the_resume_step_runs_once_it_is_over(ed,
                                                            monkeypatch):
    """_after_resume_step: held while the step is to come, run once it is
    over (even when it failed), and at once after that."""
    root, editor = ed
    # The editor's own resume step (200 ms after it opens) is over first,
    # so nothing else runs what waits.
    assert pump_until(root, lambda: editor._resume_step_done, timeout=10)
    ran = []
    editor._resume_step_done = False
    editor._after_resume = []
    editor._after_resume_step(lambda: ran.append('first'))
    root.update()
    assert ran == []

    def broken(self):
        raise RuntimeError("damaged autosave")
    monkeypatch.setattr(type(editor), '_offer_resume_session', broken)
    with pytest.raises(RuntimeError):
        editor._maybe_resume_session()
    assert pump_until(root, lambda: ran == ['first'])
    editor._after_resume_step(lambda: ran.append('later'))
    assert pump_until(root, lambda: ran == ['first', 'later'])


# ── What's new on an upgrade's first start ────────────────────────────────

@pytest.mark.parametrize('stored, want', [
    ({'geometry': '900x600+10+10'}, ['whats-new']),     # 3.0.0's own prefs
    ({}, ['welcome']),                                  # a first launch
])
def test_whats_new_waits_for_the_resume_question_and_replaces_the_welcome(
        ed, first_launch, monkeypatch, stored, want):
    """On the start What's new opens it waits until "Resume last session?"
    is answered, as the welcome does, and the welcome does not open too:
    one window, not two over the question."""
    from openflo import editor_update, update
    root, editor = ed
    assert pump_until(root, lambda: editor._resume_step_done, timeout=10)
    first_launch()
    for k, v in stored.items():
        prefs.write_pref(k, v)
    monkeypatch.setattr(update, 'current_version', lambda: '9.2.0')
    monkeypatch.setattr(prefs, 'load_check_updates_on_start', lambda: False)
    opened = []
    cls = type(editor)
    monkeypatch.setattr(cls, '_show_whats_new',
                        lambda self: opened.append('whats-new'))
    monkeypatch.setattr(cls, '_show_welcome',
                        lambda self: opened.append('welcome'))
    monkeypatch.setattr(cls, '_offer_resume_session', lambda self: None)
    assert editor_update.startup_popups_allowed()
    # As gui.main starts it, with the question still open.
    editor._resume_step_done = False
    editor._after_resume = []
    editor._schedule_startup_checks()
    editor._after_resume_step(editor._maybe_welcome)
    settle(root, 1.2)                   # past What's new's 800 ms
    assert opened == []
    editor._maybe_resume_session()      # the question is answered
    assert pump_until(root, lambda: opened == want)
    settle(root, 0.3)
    assert opened == want


def test_the_welcome_flag_is_no_sign_of_an_earlier_openflo(ed, clean_prefs):
    """What the welcome saves when it opens (first_run.PREF_KEY) never makes
    a fresh install look like an upgrade (whats_new.ran_before)."""
    from openflo import whats_new
    root, editor = ed
    editor._show_welcome().destroy()
    root.update()
    stored = prefs.read_prefs()
    assert stored == {fr.PREF_KEY: True}
    assert whats_new.ran_before(stored) is False


# ── Help reopens it ────────────────────────────────────────────────────────

def _help_entry(editor, label):
    btn = next(b for b in editor._menubar_buttons if b.cget('text') == 'Help')
    menu = btn.nametowidget(btn.cget('menu'))
    return menu, menu.index(label)


def test_help_reopens_the_tour_and_the_sample(ed):
    from openflo.ui_first_run import TourPanel
    root, editor = ed
    menu, i = _help_entry(editor, 'Getting started tour')
    menu.invoke(i)
    root.update()
    (panel,) = _windows(editor, TourPanel)
    panel.next_btn.invoke()
    assert panel.index == 1
    menu.invoke(i)                         # again: the same panel, step 1
    root.update()
    assert _windows(editor, TourPanel) == [panel] and panel.index == 0
    jobs = []
    editor._queue_fcs_loads = lambda paths, front_names=(): jobs.extend(paths)
    menu, i = _help_entry(editor, 'Open the sample data')
    menu.invoke(i)
    assert len(jobs) == len(fr.SAMPLE_NAMES)
    assert os.path.realpath(jobs[0]).startswith(
        os.path.realpath(fr.sample_root()) + os.sep)


# ── the sample ─────────────────────────────────────────────────────────────

def test_the_sample_is_package_data():
    text = open('pyproject.toml', encoding='utf-8').read()
    line = next(ln for ln in text.splitlines() if ln.startswith('openflo = ['))
    assert '"sample_data/*.flowproject"' in line
    assert fr.bundled_sample().is_file()


def test_opening_the_sample_loads_two_gated_samples_in_a_user_folder(ed):
    root, editor = ed
    shipped = os.fspath(fr.bundled_sample())
    pkg = os.path.dirname(shipped)
    before = sorted(os.listdir(pkg))
    assert _open_sample(root, editor) is True
    assert list(editor._samples) == list(fr.SAMPLE_NAMES)
    for name in fr.SAMPLE_NAMES:
        gates = editor._sample_gates[name]
        assert {g['name'] for g in gates.values()} == GATES
        assert len(editor._samples[name].data) == fr.SAMPLE_EVENTS
        assert editor._samples[name].comp_channels    # $SPILLOVER applied
        path = os.path.realpath(editor._samples[name].path)
        assert path.startswith(os.path.realpath(fr.sample_root()) + os.sep)
    assert sorted(os.listdir(pkg)) == before          # nothing written there
    # The control is on the plot as it opens: a gated plot at once.
    assert pump_until(root, lambda: editor._sample_plot_enabled.get(
        'Control') is True, timeout=10)
    settle(root, 0.3)
    assert editor._sample_plot_enabled == {'Control': True, 'Treated': False}


def test_opening_the_sample_asks_before_replacing_the_session(ed,
                                                             monkeypatch):
    _root, editor = ed
    asked = []

    def ok(self, path):
        asked.append(path)
        return False                                   # the user cancels
    monkeypatch.setattr(type(editor), '_ok_to_replace_session', ok)

    def copies():
        root = fr.sample_root()
        return sorted(os.listdir(root)) if os.path.isdir(root) else []
    before = copies()
    assert editor._open_sample_data() is False
    assert asked == [fr.SAMPLE_COPY_NAME]
    assert copies() == before                          # nothing unpacked


@pytest.mark.parametrize('had_prefs', [False, True])
def test_opening_the_sample_leaves_nothing_for_later_tests(had_prefs):
    """What opening the sample records (its copy under Open recent, the
    copy itself) is undone after each test here, so no later test in the
    same worker finds a recent session to open: with no prefs file before,
    and with one."""
    path = prefs._prefs_path()
    if os.path.exists(path):
        os.remove(path)
    if had_prefs:
        prefs.write_pref('recent_sessions', [])

    def state():
        root = fr.sample_root()
        return (os.path.exists(path), prefs.read_prefs(),
                sorted(os.listdir(root)) if os.path.isdir(root) else [])
    before = state()
    assert before[0] is had_prefs
    with _user_state_kept():
        root, editor = editor_or_skip()
        try:
            _open_sample(root, editor)
        finally:
            root.destroy()
        _exists, stored, copies = state()
        recent = stored.get('recent_sessions')
        # The undo is needed: opening the sample did record both.
        assert recent and os.path.realpath(recent[0]).startswith(
            os.path.realpath(fr.sample_root()) + os.sep)
        assert len(copies) == len(before[2]) + 1
    assert state() == before


def test_a_second_copy_gets_its_own_folder(tmp_path):
    first = fr.unpack_sample(str(tmp_path))
    second = fr.unpack_sample(str(tmp_path))
    assert os.path.dirname(first) != os.path.dirname(second)
    assert sorted(os.listdir(tmp_path)) == [
        fr.SAMPLE_COPY_NAME, fr.SAMPLE_COPY_NAME + ' (2)']


def test_opening_the_sample_from_the_welcome_starts_the_tour(ed):
    from openflo.ui_first_run import TourPanel
    root, editor = ed
    _open_sample(root, editor, tour=True)
    (panel,) = _windows(editor, TourPanel)
    assert panel.index == 0


def test_opening_the_sample_says_where_the_copy_is(ed):
    _root, editor = ed
    editor._queue_fcs_loads = lambda paths, front_names=(): None
    assert editor._open_sample_data() is True
    folder = os.path.dirname(editor._session_file)
    assert os.path.basename(folder) == 'OpenFlo sample'
    assert editor.status_var.get() == (
        "Opened the sample: a control and a treated sample, gated. "
        f"Your copy is in {folder}.")


def test_opening_the_sample_over_open_samples_asks_once(ed, monkeypatch):
    """Over a session with unsaved changes: one question, answered for
    real (Don't save), then the sample opens in its place."""
    root, editor = ed
    _open_sample(root, editor)
    first = editor._session_file
    editor._unsaved = True
    asked = []

    def ask_choice(title, *a, **k):
        asked.append(title)
        return 'discard'
    monkeypatch.setattr(editor, '_ask_choice', ask_choice)
    assert _open_sample(root, editor) is True
    assert asked == ["Open another session"]
    second = editor._session_file
    assert os.path.dirname(second) != os.path.dirname(first)
    assert list(editor._samples) == list(fr.SAMPLE_NAMES)
    for name in fr.SAMPLE_NAMES:
        assert os.path.realpath(editor._samples[name].path).startswith(
            os.path.realpath(os.path.dirname(second)) + os.sep)


def test_a_sample_session_that_will_not_load_is_not_called_opened(
        ed, monkeypatch):
    """The session itself does not load (it says why itself): no
    "Opened the sample", no tour, and False."""
    from openflo.ui_first_run import TourPanel
    _root, editor = ed
    editor.status_var.set("Load session failed: damaged")
    monkeypatch.setattr(type(editor), '_load_session_path',
                        lambda self, path, confirmed=False: None)
    assert editor._open_sample_data(tour=True) is False
    assert editor.status_var.get() == "Load session failed: damaged"
    assert _windows(editor, TourPanel) == []


def test_a_failed_sample_load_over_an_open_session_is_not_called_opened(
        ed, monkeypatch, tmp_path):
    """The user's own session file stays open when the sample session does
    not load, so the check is that the sample itself is open, not merely
    that some session is: False, no "Opened the sample", no tour."""
    from openflo.ui_first_run import TourPanel
    _root, editor = ed
    own = str(tmp_path / 'own.flowproject')
    editor._session_file = own
    editor.status_var.set("Load session failed: damaged")
    monkeypatch.setattr(type(editor), '_ok_to_replace_session',
                        lambda self, name: True)
    monkeypatch.setattr(type(editor), '_load_session_path',
                        lambda self, path, confirmed=False: None)
    assert editor._open_sample_data(tour=True) is False
    assert editor._session_file == own
    assert editor.status_var.get() == "Load session failed: damaged"
    assert _windows(editor, TourPanel) == []


def test_a_sample_that_will_not_open_is_reported(ed, monkeypatch):
    _root, editor = ed
    shown = []
    monkeypatch.setattr('openflo.ui_errors.show_error',
                        lambda *a, **k: shown.append(a[1]))

    def broken(root=None):
        raise OSError("disk full")
    monkeypatch.setattr(fr, 'unpack_sample', broken)
    assert editor._open_sample_data() is False
    assert shown == ["Can't open the sample"]
    assert 'disk full' in editor.status_var.get()
    assert not editor._samples


def _make_desktop_sample():
    """scripts/make_desktop_sample.py, the script that writes the sample."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'scripts', 'make_desktop_sample.py')
    spec = importlib.util.spec_from_file_location('make_desktop_sample', path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _summary(path, tmp):
    doc = Document()
    session = doc.unpack_project(path, str(tmp))['session']
    with open(session, encoding='utf-8') as f:
        plotted = [e['name'] for e in json.load(f)['samples']
                   if e.get('plot_enabled')]
    doc.open_session(session)
    return {'plotted': plotted, 'counts': {
        s['name']: {r['population'].rsplit('/', 1)[-1]: r['count']
                    for r in doc.stats(s['name'], channels=[])}
        for s in doc.samples()}}


def test_the_shipped_sample_is_the_one_the_script_writes(tmp_path):
    written = _make_desktop_sample().main(str(tmp_path / 'new.flowproject'))
    summary = _summary(os.fspath(fr.bundled_sample()), tmp_path / 'a')
    assert summary == _summary(written, tmp_path / 'b')
    assert summary['plotted'] == ['Control']
    shipped = summary['counts']
    assert list(shipped) == list(fr.SAMPLE_NAMES)
    for counts in shipped.values():
        assert counts['All events'] == fr.SAMPLE_EVENTS
        assert set(counts) - {'All events'} == GATES
    # The treatment switches the activation marker on.
    share = {s: c['CD69+'] / c['T cells'] for s, c in shipped.items()}
    assert share['Treated'] > 2 * share['Control']


# ── the tour ───────────────────────────────────────────────────────────────

def test_the_tour_steps_forward_and_back_and_closes_on_done(ed):
    from openflo.ui_first_run import TourPanel, tour_target
    root, editor = ed
    editor.deiconify()
    root.update()
    panel = editor._start_tour()
    assert isinstance(panel, TourPanel)
    n = len(fr.TOUR_STEPS)
    assert 3 <= n <= 5
    assert panel.step_label.cget('text') == f"Step 1 of {n}"
    assert panel.back_btn.instate(['disabled'])
    seen = []
    for i in range(n):
        assert panel.index == i
        assert panel.title_label.cget('text') == fr.TOUR_STEPS[i][1]
        target = panel.target()
        assert target is not None and target.winfo_ismapped()
        assert target is tour_target(editor, fr.TOUR_STEPS[i][0])
        assert len(panel._outline) == 4                  # the outline
        seen.append(target)
        if i < n - 1:
            assert str(panel.next_btn.cget('text')) == "Next"
            panel.next_btn.invoke()
    assert str(panel.next_btn.cget('text')) == "Done"
    assert not panel.back_btn.instate(['disabled'])
    assert seen[1] is editor.gate_tv
    assert seen[2] is editor.x_combo.master.master
    panel.back_btn.invoke()
    assert panel.index == n - 2
    outline = list(panel._outline)
    panel.next_btn.invoke()
    assert all(not f.winfo_exists() for f in outline)     # moved on
    outline = list(panel._outline)
    panel.next_btn.invoke()                               # Done
    root.update()
    assert not panel.winfo_exists()
    assert all(not f.winfo_exists() for f in outline)


def test_the_tour_does_not_hold_the_editor(ed):
    root, editor = ed
    panel = editor._start_tour()
    root.update()
    assert panel.grab_current() is None
    editor.status_var.set("still usable")
    assert editor.status_var.get() == "still usable"
    panel.close_btn.invoke()
    assert not panel.winfo_exists()


def test_escape_closes_the_welcome_and_the_tour(ed):
    root, editor = ed
    editor.deiconify()
    for win in (editor._show_welcome(), editor._start_tour()):
        root.update()
        win.focus_force()
        root.update()
        win.event_generate('<Escape>', when='now')
        root.update()
        assert not win.winfo_exists()


def test_the_tour_and_the_welcome_take_the_keys(ed):
    """Each puts the keyboard on a button of its own when it opens, so Escape
    reaches it on every OS (on macOS the tour had none and kept none)."""
    root, editor = ed
    editor.deiconify()
    welcome = editor._show_welcome()
    root.update()
    assert welcome.focus_lastfor() is welcome.sample_btn
    welcome.destroy()
    tour = editor._start_tour()
    root.update()
    assert tour.focus_lastfor() is tour.next_btn


def test_the_tour_keeps_to_its_steps(ed):
    _root, editor = ed
    panel = editor._start_tour()
    panel.show(len(fr.TOUR_STEPS) + 3)
    assert panel.index == len(fr.TOUR_STEPS) - 1
    panel.show(-2)
    assert panel.index == 0
    panel.destroy()


def test_the_tour_sits_below_its_widget_or_above_without_room(ed,
                                                             monkeypatch):
    root, editor = ed
    editor.deiconify()
    editor.geometry('900x600+0+300')
    root.update()
    panel = editor._start_tour(1)                  # the population tree
    root.update()
    widget = panel.target()

    def top():
        root.update()
        return int(panel.geometry().split('+')[2])
    monkeypatch.setattr(type(panel), 'winfo_screenheight',
                        lambda self: 100000)
    panel.show(1)
    assert top() >= widget.winfo_rooty() + widget.winfo_height()
    monkeypatch.setattr(
        type(panel), 'winfo_screenheight',
        lambda self: widget.winfo_rooty() + widget.winfo_height() + 10)
    panel.show(1)
    assert top() + panel.winfo_reqheight() <= widget.winfo_rooty()
    panel.destroy()


def test_the_tour_follows_its_widget_when_the_editor_moves_late(ed,
                                                                monkeypatch):
    """macOS settles a window after Tk has placed it (CI saw the editor land
    38 px lower once the tour card had placed itself, so the card covered the
    population tree by 30 px). Moving the editor after the card is placed
    must bring the card back below its widget, and it must not cover it."""
    root, editor = ed
    editor.deiconify()
    editor.geometry('900x600+0+262')
    root.update()
    panel = editor._start_tour(1)                  # the population tree
    monkeypatch.setattr(type(panel), 'winfo_screenheight',
                        lambda self: 100000)
    panel.show(1)
    root.update()
    widget = panel.target()
    for y in (300, 340):                           # the late move, then a drag
        editor.geometry(f'+0+{y}')
        root.update()
        root.update()
        top = int(panel.geometry().split('+')[2])
        assert widget.winfo_rooty() >= y           # the move really happened
        assert top == widget.winfo_rooty() + widget.winfo_height() + 8, y
    panel.destroy()


def test_a_burst_of_editor_moves_re_places_the_tour_once(ed, monkeypatch):
    """A drag sends many moves before Tk is idle: the card schedules one
    re-place for the burst, not one per move."""
    root, editor = ed
    editor.deiconify()
    editor.geometry('900x600+0+262')
    root.update()
    panel = editor._start_tour(1)
    root.update()
    calls = []
    monkeypatch.setattr(type(panel), '_refollow',
                        lambda self: calls.append(1))
    for y in (280, 300, 320):
        editor.event_generate('<Configure>', x=0, y=y, width=900, height=600)
    root.update()
    assert calls == [1]
    panel.destroy()


def test_a_part_of_the_card_closing_does_not_stop_it_following(ed):
    """Only the card closing stops it following the editor: a widget inside
    the card being destroyed (its Destroy event reaches the card too) does
    not."""
    root, editor = ed
    editor.deiconify()
    editor.geometry('900x600+0+262')
    root.update()
    panel = editor._start_tour(1)
    panel.winfo_screenheight = lambda: 100000
    root.update()
    import tkinter as tk
    tk.Frame(panel).destroy()
    root.update()
    widget = panel.target()
    editor.geometry('+0+330')
    root.update()
    root.update()
    top = int(panel.geometry().split('+')[2])
    assert top == widget.winfo_rooty() + widget.winfo_height() + 8
    panel.destroy()


def test_closing_the_tour_keeps_the_editors_own_move_handling(ed):
    """The card follows the editor through a <Configure> binding of its own;
    closing the card removes that one only, and the editor's own binding
    (the chrome scaling) stays, with no command or pending re-place left."""
    root, editor = ed
    before = editor.bind('<Configure>')
    assert before.strip()
    panel = editor._start_tour(1)
    root.update()
    fid = panel._follow_id
    assert fid in editor.bind('<Configure>')
    editor.event_generate('<Configure>')           # a re-place now pending
    job = panel._follow_job
    assert job is not None
    panel.destroy()
    assert editor.bind('<Configure>').strip() == before.strip()
    assert not editor.tk.call('info', 'commands', fid)
    assert job not in editor.tk.splitlist(editor.tk.call('after', 'info'))
