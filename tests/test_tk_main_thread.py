"""A Tk interpreter dropped by a test is never freed on a background thread.

The Windows CI legs lost an xdist worker in a different test each run. The
cause: a background thread's garbage collection freed a Tk interpreter that an
earlier GUI test had dropped, and Tcl aborts the process when that happens
("Tcl_AsyncDelete: async handler deleted by the wrong thread").
tests/tk_main_thread.py holds every interpreter until the main thread frees
it; these tests check that, and that the scenario really aborts without it.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from tests import tk_main_thread
from tests.conftest import gui_unavailable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A GUI test's leftovers: a root in a reference cycle, with a Tk variable and
# a widget, destroyed and dropped. Then a background thread collects garbage,
# as the suite's warm-up, autosave and reader threads do mid-run.
SCENARIO = textwrap.dedent('''
    import gc, sys, threading, tkinter as tk
    sys.path.insert(0, {root!r})
    if sys.argv[1] == 'protected':
        from tests import tk_main_thread
        tk_main_thread.install()
    gc.disable()          # only the background thread's collection may run
    for _ in range(5):
        root = tk.Tk()
        root.withdraw()
        root.me = root                      # the cycle a real window has
        var = tk.StringVar(root, 'x')
        tk.Label(root, textvariable=var).pack()
        root.update()
        root.destroy()
        del root, var
        worker = threading.Thread(target=gc.collect)
        worker.start()
        worker.join()
        if sys.argv[1] == 'protected':
            assert tk_main_thread.held() == 1, tk_main_thread.held()
            tk_main_thread.release()
            assert tk_main_thread.held() == 0, tk_main_thread.held()
    print('survived')
''').format(root=ROOT)


@pytest.fixture
def tk_works():
    try:
        import tkinter as tk
        probe = tk.Tk()
        probe.destroy()
    except Exception as exc:                                 # noqa: BLE001
        gui_unavailable(f"Tk could not start: {exc}")


def _run(mode):
    return subprocess.run(
        [sys.executable, '-X', 'faulthandler', '-c', SCENARIO, mode],
        capture_output=True, text=True, timeout=120, cwd=ROOT)


def _new_root():
    """A withdrawn root, retried the way conftest retries: on a loaded
    Windows runner Tk now and then fails to read its own library files
    ("Can't find a usable tk.tcl") and comes up on the next try. A failed
    try is still held, so it is released before the count is taken."""
    import tkinter as tk
    for _ in range(4):
        try:
            root = tk.Tk()
        except tk.TclError:
            tk_main_thread.release()
            time.sleep(0.5)
        else:
            root.withdraw()
            return root
    root = tk.Tk()                      # the last try: let its error show
    root.withdraw()
    return root


def _in_a_clean_process(name):
    """Run test ``name`` again in a fresh pytest process, and say so.

    The two in-process tests below need a garbage collection to really run
    when they ask for one. In the long-lived suite process it may not: a
    leftover thread can sit inside a collection for many seconds (a Tk
    variable's finaliser waiting on the main thread), and ``gc.collect()``
    called meanwhile on any thread returns at once without collecting. A
    fresh process has no such thread, and its conftest installs the hold
    just as the suite's does."""
    if os.environ.get('OPENFLO_TK_CLEAN_PROCESS'):
        return False
    out = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
         '-p', 'no:xdist', '-p', 'no:randomly', f'tests/{os.path.basename(__file__)}::{name}'],
        env={**os.environ, 'OPENFLO_TK_CLEAN_PROCESS': '1'},
        capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-2000:]
    return True


def test_a_background_collection_of_a_dropped_root_aborts_without_protection(
        tk_works):
    """The control: the scenario reproduces the CI crash when nothing holds
    the interpreter, so the next test is not passing by luck."""
    out = _run('bare')
    assert out.returncode != 0, out.stdout + out.stderr
    assert 'survived' not in out.stdout


def test_a_background_collection_of_a_dropped_root_is_harmless_with_protection(
        tk_works):
    out = _run('protected')
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip() == 'survived'


def test_the_suite_holds_each_root_until_the_main_thread_releases_it(tk_works):
    """In this process: conftest installed the hold, so a background
    collection leaves a dropped root's interpreter alone, and release frees
    it on the main thread."""
    if _in_a_clean_process(
            'test_the_suite_holds_each_root_until_the_main_thread_releases_it'):
        return
    tk_main_thread.release()
    before = tk_main_thread.held()
    root = _new_root()
    root.me = root
    interp = root.tk
    root.destroy()
    del root
    assert tk_main_thread.held() == before + 1

    worker = threading.Thread(target=gc.collect)
    worker.start()
    worker.join()
    # Our name, the hold, and getrefcount's argument: the background
    # collection could not have freed it.
    assert sys.getrefcount(interp) == 3
    del interp

    tk_main_thread.release()
    assert tk_main_thread.held() == before


def test_release_collects_the_cycles_that_still_hold_a_dropped_root(tk_works):
    """No background collection came first: the dropped root still sits in
    its cycle, and release must break that cycle itself to free it."""
    if _in_a_clean_process(
            'test_release_collects_the_cycles_that_still_hold_a_dropped_root'):
        return
    tk_main_thread.release()
    before = tk_main_thread.held()
    gc.disable()
    try:
        root = _new_root()
        root.me = root
        root.destroy()
        del root
        assert tk_main_thread.held() == before + 1
        tk_main_thread.release()
        assert tk_main_thread.held() == before
    finally:
        gc.enable()


class _Owner:
    """What an editor is to its window: it holds the root, and a callback
    bound to it is registered with Tcl, which holds it from C."""

    def __init__(self, root):
        self.root = root
        self.var = None

    def changed(self, *_args):
        pass


def test_release_frees_a_destroyed_window_whose_callbacks_tcl_still_holds(
        tk_works):
    """A trace and a registered command keep the window's owner alive from C,
    where the collector cannot see, and a held interpreter is polled by every
    later event loop. Once the window is destroyed nothing can call them, so
    release deletes them and frees the interpreter."""
    if _in_a_clean_process(
            'test_release_frees_a_destroyed_window_whose_callbacks_tcl_still_holds'):
        return
    import tkinter as tk
    tk_main_thread.release()
    before = tk_main_thread.held()
    root = _new_root()
    owner = _Owner(root)
    owner.var = tk.StringVar(root, 'x')
    owner.var.trace_add('write', owner.changed)
    root.register(owner.changed)
    root.tk.createcommand('mpl_blit_test', owner.changed)  # matplotlib's kind
    root.update()
    root.destroy()
    del root, owner
    assert tk_main_thread.held() == before + 1
    tk_main_thread.release()
    assert tk_main_thread.held() == before


def test_a_variable_freed_on_another_thread_leaves_its_tcl_clean_up_to_the_main_thread(
        tk_works):
    """Freed on a background thread, a Tk variable asks the main thread to
    unset it and waits for an answer that does not come while the main thread
    is busy, inside a collection the whole time. It must queue the clean-up
    for release instead, and release must run it."""
    if _in_a_clean_process(
            'test_a_variable_freed_on_another_thread_leaves_its_tcl_clean_up_to_the_main_thread'):
        return
    import tkinter as tk
    root = _new_root()
    names = []
    gc.disable()
    try:
        for i in range(3):
            var = tk.StringVar(root, str(i))
            var.me = var                    # only the collector frees it
            names.append(str(var))
            del var
        worker = threading.Thread(target=gc.collect)
        worker.start()
        worker.join()
    finally:
        gc.enable()
    assert len(tk_main_thread._deferred) == 3
    assert all(root.getboolean(root.tk.call('info', 'exists', n)) for n in names)
    tk_main_thread.release()
    assert not tk_main_thread._deferred
    assert not any(root.getboolean(root.tk.call('info', 'exists', n)) for n in names)
    root.destroy()
    del root
    tk_main_thread.release()


def test_leftover_windows_are_destroyed_and_freed_when_the_module_ends(tk_works):
    """A window a test left open (never destroyed) is destroyed and freed at
    the end of its module, so it is not held and polled for the rest of the
    run."""
    if _in_a_clean_process(
            'test_leftover_windows_are_destroyed_and_freed_when_the_module_ends'):
        return
    import tkinter as tk
    tk_main_thread.release()
    before = tk_main_thread.held()
    root = _new_root()
    owner = _Owner(root)
    owner.var = tk.StringVar(root, 'x')
    owner.var.trace_add('write', owner.changed)
    tk.Button(root, command=owner.changed).pack()
    root.update()
    del root, owner
    tk_main_thread.release()
    assert tk_main_thread.held() == before + 1      # still open: kept
    tk_main_thread.release_leftovers()
    assert tk_main_thread.held() == before


def test_a_test_that_made_no_window_pays_for_no_collection(tk_works, monkeypatch):
    """A full collection can take seconds in a long-lived worker, so release
    collects only when a window was made since the last one (or a finaliser
    is queued), not after every test while some window is still held."""
    import types
    calls = []
    fake_gc = types.SimpleNamespace(
        collect=lambda *a: calls.append(1) or gc.collect(*a))
    monkeypatch.setattr(tk_main_thread, 'gc', fake_gc)
    tk_main_thread.release(force=True)
    root = _new_root()                     # held: this test still uses it
    calls.clear()
    tk_main_thread.release()
    assert calls, 'a window was made, so release must collect'
    calls.clear()
    tk_main_thread.release()
    assert not calls, 'no window made since: nothing to collect'
    tk_main_thread.release(force=True)
    assert calls, 'force collects anyway'
    root.destroy()
    del root
    tk_main_thread.release(force=True)
