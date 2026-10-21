"""Shared pytest fixtures.

`synthetic_fcs` writes a tiny in-memory FCS via FlowIO so the suite can
exercise FlowSample / compensation / gating without checking real patient
data into the repo.

Every test runs on generated data; none reads a real sample.
"""
from __future__ import annotations

import importlib
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import pytest

# ── Keep the suite's git commands off the developer's repository ─────────────
#
# Run from a git hook (the pre-push preflight), the suite inherits the GIT_DIR /
# GIT_WORK_TREE / GIT_INDEX_FILE git exports to hooks. A test's
# `git init <tmp>` then re-initialised the repository GIT_DIR names instead of
# <tmp>, with no work tree, and left core.bare=true in its config: afterwards
# every worktree reported "this operation must be run in a work tree" (seen
# 2026-10-09 after a hooked push). Tests that want a repository make their own.
for _git_var in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE',
                 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_PREFIX'):
    os.environ.pop(_git_var, None)

# ── Keep the suite out of the developer's real home ──────────────────────────
#
# OpenFlo keeps per-user state under ~/.openflo — prefs.json, the error report
# and its key file, autosaves, transfer markers, example and synthetic data —
# and resolves every one of those paths with os.path.expanduser('~') at call
# time. So any test that reaches that code used the real home of whoever ran
# the suite. It did: test editors resumed the developer's own last session
# (loading their real FCS data, once into a MemoryError) and pruned their
# autosaves, test keys landed in their prefs.json, and test tracebacks in their
# error report.
#
# So, before any test module is imported, every variable expanduser consults
# is pointed at a fresh temporary directory; subprocesses (the CLI tests)
# inherit it, and each xdist worker makes its own. And because "nothing writes
# there any more" is a
# claim, the run checks it: the real ~/.openflo is snapshotted here and compared
# when the session ends (pytest_sessionfinish), and any difference fails the run.

# The home whose .openflo the guard watches. Exported so that xdist workers
# (and any nested pytest run), which inherit the redirected HOME, guard the real
# home rather than their parent's temporary one. Set it yourself to aim the
# guard at a scratch directory — that is how the guard itself is tested.
_REAL_HOME_ENV = 'OPENFLO_TEST_REAL_HOME'
REAL_HOME = os.path.abspath(
    os.environ.get(_REAL_HOME_ENV) or os.path.expanduser('~'))
os.environ[_REAL_HOME_ENV] = REAL_HOME


def _home_snapshot(home):
    """``{path under .openflo: (size, mtime_ns)}`` for ``home``/.openflo and
    everything in it ('.' is the directory itself), or None if it does not
    exist.

    Read-only by construction — lstat and directory listings, nothing that can
    create, change or delete — because that directory belongs to the
    developer, not to the suite. A directory's mtime moves when an entry is
    added, removed or renamed, so a file created and deleted again within the
    run (an atomic-write .tmp) still shows."""
    root = os.path.join(home, '.openflo')
    try:
        st = os.lstat(root)
    except OSError:
        return None
    snap = {'.': (st.st_size, st.st_mtime_ns)}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except OSError:         # removed mid-walk; the next snapshot says so
                continue
            snap[os.path.relpath(path, root)] = (st.st_size, st.st_mtime_ns)
    return snap


def _home_changes(before, after):
    """One line per entry created, removed or modified between two snapshots."""
    before, after = before or {}, after or {}
    changes = []
    for path in sorted(set(before) | set(after)):
        if path not in after:
            changes.append(f"removed   {path}")
        elif path not in before:
            changes.append(f"created   {path}")
        elif before[path] != after[path]:
            changes.append(f"modified  {path}")
    return changes


_REAL_HOME_BEFORE = _home_snapshot(REAL_HOME)

# The one deliberate exception: CuPy's compiled-kernel cache stays in the real
# ~/.cupy. It is a pure cache keyed on the kernel source, and starting it cold
# costs a GPU host ~15 s of compiles per run. Set here, before the redirect, so
# CuPy uses it whether it is imported before this file (zarr's pytest plugin
# imports it) or after; subprocesses and xdist workers inherit it.
os.environ.setdefault('CUPY_CACHE_DIR', os.path.join(
    os.path.expanduser('~'), '.cupy', 'kernel_cache'))

TEST_HOME = tempfile.mkdtemp(prefix='openflo-test-home-')
os.environ['HOME'] = os.environ['USERPROFILE'] = TEST_HOME
if os.name == 'nt':
    # expanduser falls back to these when USERPROFILE is unset; other tools
    # (git, Tcl) read them or HOME directly. Keep them all in agreement.
    os.environ['HOMEDRIVE'], os.environ['HOMEPATH'] = os.path.splitdrive(
        TEST_HOME)
# Prove the redirect took rather than assume it: a platform that resolves the
# home some other way would otherwise put the whole suite back on the real one.
if os.path.expanduser('~') != TEST_HOME:
    raise RuntimeError(
        f"tests/conftest.py could not redirect the home directory: "
        f"expanduser('~') is {os.path.expanduser('~')!r}, not {TEST_HOME!r}")


_REAL_HOME_CHANGES: list[str] = []


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session):
    """Fail the run if the real ~/.openflo changed while it ran.

    A session hook, not a fixture: a fixture's teardown error is swallowed
    when the test it lands on is marked xfail, and a fixture never runs at all
    when every test is skipped or deselected. Under xdist only the controller
    checks, once every worker has finished. Writes made after this point
    (atexit, a thread or child process that outlives the session) are beyond
    any check the run can make."""
    if os.environ.get('PYTEST_XDIST_WORKER'):
        return
    _REAL_HOME_CHANGES[:] = _home_changes(_REAL_HOME_BEFORE,
                                          _home_snapshot(REAL_HOME))
    if not _REAL_HOME_CHANGES:
        return
    if session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    # Printed in red just above the final counts, which would otherwise read
    # as a clean pass.
    session.shouldfail = (f"FAILED: the run changed the real "
                          f"{os.path.join(REAL_HOME, '.openflo')} "
                          f"(listed under 'real home changed')")
    if session.config.pluginmanager.get_plugin('terminalreporter') is None:
        print(_real_home_report(), file=sys.stderr)


def pytest_terminal_summary(terminalreporter):
    if _REAL_HOME_CHANGES:
        terminalreporter.write_sep('=', 'real home changed', red=True)
        terminalreporter.write_line(_real_home_report())


def _real_home_report():
    return (f"The test run changed {os.path.join(REAL_HOME, '.openflo')}:\n  "
            + "\n  ".join(_REAL_HOME_CHANGES)
            + "\nThe suite points HOME/USERPROFILE at a temporary directory "
            "(tests/conftest.py), so this was reached another way: a path "
            "captured before the redirect, a hard-coded path, or a subprocess "
            "given its own environment. If OpenFlo or another test run was "
            "using that directory at the same time, that is the likelier "
            "cause; a re-run tells them apart. Nothing there was changed "
            "back.")


def pytest_unconfigure():
    shutil.rmtree(TEST_HOME, ignore_errors=True)


# ── Tk dialogs in tests: patched by the test, or an error ────────────────────

def _dialog_functions():
    """The ask*/show* functions of Tk's dialog modules, as first imported."""
    found = {}
    for name in ('tkinter.messagebox', 'tkinter.filedialog',
                 'tkinter.simpledialog', 'tkinter.colorchooser'):
        try:
            mod = importlib.import_module(name)
        except ImportError:                 # no Tk here: nothing to leak
            continue
        for attr, value in vars(mod).items():
            if attr.startswith(('ask', 'show')) and callable(value):
                found[(mod, attr)] = value
    return found


_DIALOG_FUNCTIONS = _dialog_functions()
_UNPATCHED_DIALOG_CALLS: list[str] = []


def _refuse_dialog(name):
    """A stand-in for one Tk dialog function during a test. The real one is
    modal and would hang the run waiting for a click nobody will make."""
    def refuse(*args, **kwargs):
        call = f"{name}({kwargs.get('title', args[0] if args else None)!r})"
        _UNPATCHED_DIALOG_CALLS.append(call)
        pytest.fail(f"{call} was called during a test without being patched: "
                    f"a real dialog would block the run. Patch it in the test "
                    f"(monkeypatch.setattr) with the answer the test expects.")
    return refuse


_DIALOG_STAND_INS = {
    (mod, attr): _refuse_dialog(f"{mod.__name__.rsplit('.', 1)[-1]}.{attr}")
    for mod, attr in _DIALOG_FUNCTIONS}


@pytest.fixture(autouse=True)
def _scope_dialog_patches():
    """Every Tk dialog function raises during a test unless the test patched
    it; whatever the test patched is put back afterwards.

    About thirty GUI test helpers answer every question with
    ``gui.messagebox.askyesno = lambda *a, **k: True`` and never undo it, so
    whichever ran first changed tkinter for the rest of the process. That
    includes the "Resume last session?" prompt every editor schedules on open:
    editors in later tests, in any file, resumed whatever session they found.
    Tests that patched nothing passed only because an earlier file's patch was
    still in place, and with the real dialog they would block.

    So each test starts with stand-ins that fail it, naming the dialog and its
    title, and ends with the originals restored, which makes those helpers'
    assignments last exactly one test. A call is also recorded, because a
    failure raised inside a Tk callback or an ``except`` block is swallowed;
    the recording fails the test at teardown instead. A patch made at import
    time is undone after the module's first test; use a fixture for a
    module-wide one."""
    _UNPATCHED_DIALOG_CALLS.clear()
    for (mod, attr), stand_in in _DIALOG_STAND_INS.items():
        setattr(mod, attr, stand_in)
    yield
    for (mod, attr), value in _DIALOG_FUNCTIONS.items():
        setattr(mod, attr, value)
    if _UNPATCHED_DIALOG_CALLS:
        calls = ', '.join(_UNPATCHED_DIALOG_CALLS)
        _UNPATCHED_DIALOG_CALLS.clear()
        pytest.fail(f"Unpatched Tk dialog(s) called during this test: {calls}",
                    pytrace=False)


BROWSER_OPENS: list = []


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch):
    """No test opens the real browser. A menu sweep that clicked Help >
    Online documentation used to start the system browser, which then wrote
    a whole browser profile into the redirected test home after it had been
    removed, and opened tabs on the desktop of whoever ran the tests. URLs
    asked for are kept in BROWSER_OPENS; a test that checks them patches
    ``webbrowser.open`` itself, as before."""
    import webbrowser

    def _record(url, *_a, **_k):
        BROWSER_OPENS.append(url)
        return True
    for name in ('open', 'open_new', 'open_new_tab'):
        monkeypatch.setattr(webbrowser, name, _record)
    if hasattr(os, 'startfile'):                          # Windows only
        monkeypatch.setattr(os, 'startfile',
                            lambda path, *a, **k: BROWSER_OPENS.append(path))


@pytest.fixture(autouse=True)
def _error_dialog_as_message_box(monkeypatch):
    """OpenFlo's error dialog (ui_errors) waits for a click like a message
    box. In tests it reports through ``tkinter.messagebox.showerror`` instead,
    looked up at the call, so the stand-in above fails a test that did not
    expect an error, and a test that patched ``showerror`` sees the title and
    message as before. The dialog itself is tested directly."""
    try:
        from tkinter import messagebox

        from openflo import ui_errors
    except ImportError:                     # no Tk here
        return
    monkeypatch.setattr(
        ui_errors, '_present',
        lambda parent, title, message, details: messagebox.showerror(
            title, message, parent=parent))


# ── Synthetic data ────────────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def synthetic_channels():
    return ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']


@pytest.fixture(scope='session')
def synthetic_fcs(tmp_path_factory, synthetic_channels):
    """Create a tiny on-disk FCS with 5 channels and 1000 events.

    Channels:
      FSC-A, SSC-A : log-normal scatter (positive scalar)
      FL1-A      : bimodal (negative + positive populations)
      FL2-A        : negative-skewed
      FL3-A     : positive-skewed

    Returns the path as a string.
    """
    import flowio
    rng = np.random.default_rng(seed=0)
    n_events = 1000

    fsc = rng.lognormal(mean=10, sigma=0.3, size=n_events)
    ssc = rng.lognormal(mean=9, sigma=0.4, size=n_events)
    fl1 = np.concatenate([
        rng.normal(loc=100, scale=20, size=n_events // 2),
        rng.normal(loc=5000, scale=500, size=n_events - n_events // 2),
    ])
    fl2 = rng.exponential(scale=200, size=n_events) + 50
    fl3 = rng.exponential(scale=500, size=n_events) + 100

    events = np.column_stack([fsc, ssc, fl1, fl2, fl3]).astype(np.float32)
    flat = events.flatten().tolist()

    out = tmp_path_factory.mktemp('fcs') / 'synthetic.fcs'
    with open(out, 'wb') as f:
        flowio.create_fcs(
            f, flat, synthetic_channels,
            opt_channel_names=['', '', 'CD901b', 'CD902', 'CD903'])
    return str(out)


# ── Tk interpreters die on the main thread ──────────────────────────────────
#
# A Tk interpreter freed by the garbage collector on a background thread
# aborts the process ("Tcl_AsyncDelete: async handler deleted by the wrong
# thread"), which xdist reports as a crashed worker during whichever test was
# running. tests/tk_main_thread.py explains; this holds every interpreter and
# frees it here after each test, and destroys the windows tests left open when
# each test module ends.
try:
    import tkinter as _tkinter  # noqa: F401
except ImportError:                     # no Tk at all: nothing to protect
    _tk_main_thread = None
else:
    from tests import tk_main_thread as _tk_main_thread
    _tk_main_thread.install()


@pytest.fixture(autouse=True)
def _popup_menus_wait_for_a_click_on_windows(monkeypatch, request):
    """Fail a test that opens a popup menu it does not answer.

    On Windows ``Menu.tk_popup`` and ``Menu.post`` run the system's own menu
    loop and return only when someone clicks; under Xvfb they return at
    once. So a test that reaches one passes on Linux and hangs the Windows
    CI job (#78's dialog sweep, at the Layout editor's pages button). Here
    the popup returns at once, as under Xvfb, and the test fails at the end
    unless it replaced ``tk_popup``/``post`` itself (as a stub that answers,
    the way tests stub the message boxes and file pickers)."""
    if _tk_main_thread is None:
        yield
        return
    import tkinter as tk
    opened = []

    def guard(name):
        def popup(self, *args, **kwargs):
            opened.append(f"{name} on {self}")
        return popup

    guards = {'tk_popup': guard('tk_popup'), 'post': guard('post')}
    for name, fn in guards.items():
        monkeypatch.setattr(tk.Menu, name, fn)
    yield
    if opened:
        pytest.fail(
            "a popup menu was opened and not answered; on Windows it waits "
            "for a click and the CI job hangs. Replace tk.Menu.tk_popup/post "
            "in this test (monkeypatch) with a stub: " + "; ".join(opened[:3]),
            pytrace=False)


@pytest.fixture(autouse=True)
def _free_tk_interpreters_on_the_main_thread():
    yield
    if _tk_main_thread is not None:
        _tk_main_thread.release()


@pytest.fixture(autouse=True, scope='module')
def _free_tk_windows_left_in_use():
    # Autouse, so set up before the module's own fixtures and torn down after
    # them: no module-scoped window is still wanted by then.
    yield
    if _tk_main_thread is not None:
        _tk_main_thread.release_leftovers()


# ── GUI availability: skip politely, but FAIL where a GUI was promised ──────

def gui_unavailable(reason):
    """Tk could not start. Skip on a headless dev box; FAIL in CI.

    Every GUI test guards itself with "skip if Tk is unavailable", which is
    right on a headless machine — the GUI genuinely cannot be tested there and
    a red suite would be noise. But that guard is FAIL-OPEN: if Tk stopped
    initialising everywhere, every GUI test would skip and the suite would go
    green having tested no GUI at all. That is the same "check that cannot
    fail" shape this project keeps finding in its own code, sitting in the test
    harness.

    It is not hypothetical. On windows-latest / py3.11 a test skipped with
    "Can't find a usable init.tcl", and nothing in the run said so — the leg
    was green with that file's coverage silently missing.

    So CI, which provides Xvfb on Linux and a desktop session on Windows, sets
    ``OPENFLO_REQUIRE_GUI=1`` and a Tk failure becomes an error there. Local
    runs are unaffected.
    """
    if os.environ.get('OPENFLO_REQUIRE_GUI') != '1':
        pytest.skip(reason)

    # Before failing, establish whether Tk is BROKEN or merely hiccuped.
    # Creating a Tk root fails transiently when several xdist workers do it at
    # once — observed locally within an hour of turning this guard on, and on
    # windows/py3.11 in CI ("Can't find a usable init.tcl"), while the same
    # test passes alone and under xdist within its own file. Failing on that
    # would trade a silent coverage hole for flaky red CI, which is a bad
    # trade: nobody trusts a suite that cries wolf.
    #
    # So try once more. If Tk comes up, the environment is fine and this run
    # simply lost a race — skip, and say so. If it does not, Tk is genuinely
    # unavailable where it was promised, which is what this guard is for.
    exc = _tk_probe_error()
    if exc is not None:
        pytest.fail(
            f"Tk was required in this environment but is unavailable: {reason}"
            f"\n{_TK_PROBES} more attempts also failed ({type(exc).__name__}: "
            f"{exc}), so this is not a transient race."
            "\nOPENFLO_REQUIRE_GUI=1 is set, so this is an error rather than a "
            "skip — otherwise the GUI tests would pass by not running.")
    pytest.skip(f"{reason} — {_TK_TRANSIENT}, so the "
                f"environment is fine and this run lost a race")


# Matches the ways Tk announces it cannot start. Deliberately narrow: this
# only decides whether to PROBE, and the probe decides the outcome.
_TK_FAILURE = re.compile(
    r"tclerror|tkinter|init\.tcl|no display|\$DISPLAY|"
    r"couldn't connect to display|can't find a usable|"
    r"tk could not start|no usable tk",
    re.I)


# How many fresh roots to try, with a growing pause between them, before
# calling Tk broken. ONE retry was not enough: on windows-latest a root fails
# reading Tcl's own library files ("couldn't read file .../init.tcl: No
# error", ".../ttk/fonts.tcl: no such file or directory") while other xdist
# workers start Tk at the same moment, and the single retry lost the same race
# often enough to red master and every PR in a row. A Tk that is really
# broken fails every attempt, so retrying costs the guard nothing.
_TK_PROBES = 4
_TK_PROBE_PAUSE_S = 0.5


def _tk_probe_error(attempts=_TK_PROBES, pause=_TK_PROBE_PAUSE_S):
    """None if a fresh Tk root comes up within `attempts` tries, else the
    last error."""
    import time
    last = None
    for i in range(attempts):
        if i:
            time.sleep(pause * i)
        try:
            import tkinter as tk
            probe = tk.Tk()
            probe.destroy()
            return None
        except Exception as exc:                             # noqa: BLE001
            last = exc
    return last


def _tk_really_broken():
    """True only if Tk genuinely will not start."""
    return _tk_probe_error() is not None


# gui_unavailable's own skip once Tk came up on retry. The environment was
# PROVEN fine there, so probing again here only re-runs the race it already
# lost: that second probe failing is what turned "TRANSIENT" skips into
# failures and red CI.
_TK_TRANSIENT = 'TRANSIENT: Tk came up on retry'


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Turn a Tk-caused skip into a failure when a GUI was promised.

    ``gui_unavailable`` does this for tests that call it. Most do not: 7 files
    create a root and skip on their own, and 10 more route only ImportError
    through the guard. Measured under a deliberately broken Tk, 55 GUI tests
    skipped silently while the suite still reported success for them.

    Doing it here catches every skip however it was written, so a new test
    cannot reintroduce the hole by forgetting to use the helper.
    """
    outcome = yield
    report = outcome.get_result()

    # 'setup' matters as much as 'call': a skip raised inside a fixture
    # (which is where several of these live) reports at setup, and
    # handling only 'call' would leave exactly those bypassed.
    if report.when not in ('setup', 'call') or not report.skipped:
        return
    if os.environ.get('OPENFLO_REQUIRE_GUI') != '1':
        return

    # For a skip, longrepr is (path, lineno, "Skipped: <reason>").
    reason = ''
    lr = getattr(report, 'longrepr', None)
    if isinstance(lr, tuple) and len(lr) == 3:
        reason = str(lr[2])
    elif lr is not None:
        reason = str(lr)
    if not _TK_FAILURE.search(reason) or _TK_TRANSIENT in reason:
        return

    # Only convert if Tk is genuinely down. Roots fail transiently when xdist
    # workers race, and a suite that reds on a race stops being trusted.
    if not _tk_really_broken():
        return

    report.outcome = 'failed'
    report.longrepr = (
        f"{reason}\n\n"
        "This skip was converted to a failure: OPENFLO_REQUIRE_GUI=1 promises "
        f"a working GUI in this environment, and {_TK_PROBES} more Tk roots "
        "also failed, so this is not a transient xdist race.\n"
        "Skipping here would let the GUI tests pass by not running — the "
        "failure mode tests/conftest.py:gui_unavailable exists to prevent, "
        "which only protected tests that remembered to call it.")


def skip_unless_the_screen_holds(root, width, height):
    """Skip a test whose virtual screen is bigger than the real one.

    Tests pin ``winfo_screenwidth/height`` to a virtual screen so the result
    does not depend on the machine's display. Under Xvfb, with no window
    manager, a window can be any size; on Windows the system will not make a
    window larger than the real screen (the windows-latest runner's is about
    1024x768), so a virtual 1366x768 or 1920x1080 cannot be laid out there
    and the check would measure the clamp, not the window. Probes with a real
    window at the largest size a fitted window takes (the screen less the
    margins ``tool_window.fit_window`` keeps)."""
    import tkinter as tk
    probe = tk.Toplevel(root)
    try:
        probe.geometry(f"{width - 16}x{height - 72}+0+0")
        probe.update()
        got = (probe.winfo_width(), probe.winfo_height())
    finally:
        probe.destroy()
    if got != (width - 16, height - 72):
        pytest.skip(f"the real screen ({root.tk.call('winfo', 'screenwidth', '.')}"
                    f"x{root.tk.call('winfo', 'screenheight', '.')}) cannot hold "
                    f"a {width}x{height} virtual screen's windows: a window "
                    f"asked to be {width - 16}x{height - 72} came out "
                    f"{got[0]}x{got[1]}")
