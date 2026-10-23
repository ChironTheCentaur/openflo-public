"""Every window opened from the menus runs its buttons without an error that
nothing handled, with its own defaults and with nonsense typed into it.

The dialogs pass what the user typed and picked into OpenFlo's analysis code.
The analysis code has its own tests; this sweep checks the wiring: each
window opens, each action button runs, and any failure is either handled
(a message, a status line) or, if not, shows up here. An error in a Tk
callback never reaches the caller of ``invoke`` (Tk hands it to
``report_callback_exception``) and an error in a background thread only
prints, so the sweep collects both directly.

Message boxes, file and color pickers and popup menus answer at once (Cancel,
no file, no color, closed), so actions that save or export stop at the
picker. Buttons that only close, cancel or open another picker are skipped.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import tkinter as tk

import pytest

from tests import test_windows_nothing_hidden as sweep
from tests.test_derived_kinetics import _fcs, _loaded
from tests.test_transform_records import isolate_home
from tests.test_windows_nothing_hidden import (
    NOT_A_WINDOW,
    _menu_items,
    _root,
    _settle,
    _toplevels,
    _walk,
)

SKIP_BUTTONS = ('Close', 'Cancel', 'Done', 'Help', '?', 'Browse', 'Choose',
                'Open', 'Save', 'Export', 'Copy', 'Report', 'Documentation',
                'Online', 'Pick', 'Load', 'Add file', 'Remove', 'Delete')

# The sweep's fixtures, shared.
no_dialogs = sweep.no_dialogs
engines_known = sweep.engines_known
virtual_screen = sweep.virtual_screen

ACTIONS = ('TButton', 'Button')
INPUTS = ('TEntry', 'Entry', 'TSpinbox', 'Spinbox')

NONSENSE = 'x1,;'


@pytest.fixture
def no_color_picker(monkeypatch):
    """The color picker answers at once (no color), like the other pickers,
    and a popup menu closes at once: on Windows ``tk_popup`` runs the
    system's own menu loop, which waits for a click (the Layout editor's
    pages button stopped the sweep there)."""
    from tkinter import colorchooser
    monkeypatch.setattr(colorchooser, 'askcolor', lambda *a, **k: (None, None))
    monkeypatch.setattr(tk.Menu, 'tk_popup', lambda self, *a, **k: None)
    monkeypatch.setattr(tk.Menu, 'post', lambda self, *a, **k: None)


@pytest.fixture
def unhandled(monkeypatch):
    """Collect every error a Tk callback or a thread did not handle."""
    class Seen(list):
        on_tk = None
    seen = Seen()

    def on_tk(root):
        root.report_callback_exception = (
            lambda exc, val, tb: seen.append(f"{exc.__name__}: {val}"))

    monkeypatch.setattr(
        threading, 'excepthook',
        lambda args: seen.append(
            f"thread {args.exc_type.__name__}: {args.exc_value}"))
    seen.on_tk = on_tk
    return seen


# Threads that keep running on purpose (compiling or drawing ahead of time);
# a button that starts one has still finished its own work.
LONG_LIVED = ('openflo-warmup', 'openflo-plot-warmup',
              'dialog-sweep-watchdog')


def _wait_for_work(root, cap_s=8.0):
    """Let jobs a button started finish: background threads the button made,
    and the event-loop callbacks that deliver their results. Capped, so a
    slow runner spends at most ``cap_s`` per button (an error that arrives
    later is still collected, by the next button)."""
    end = time.monotonic() + cap_s
    while time.monotonic() < end:
        root.update()
        time.sleep(0.01)
        if not [t for t in threading.enumerate()
                if t.ident not in _BASE_THREADS and t.is_alive()
                and not t.name.startswith(LONG_LIVED)]:
            break
    _settle(root, ms=50, still_ms=50, cap_ms=1000)


_BASE_THREADS: set = set()

# A button still running after this long is stuck (each normally finishes in
# well under ``_wait_for_work``'s cap): the watchdog prints which window and
# button, with every thread's stack, and ends the run instead of letting it
# sit until the CI job's own limit.
STUCK_S = 180


def _real_stderr(request):
    """The process's own stderr, which pytest's capture replaces during a
    test (a stuck test's output written to the captured one is lost when
    the worker is ended)."""
    capman = request.config.pluginmanager.getplugin('capturemanager')
    err = getattr(getattr(capman, '_global_capturing', None), 'err', None)
    fd = getattr(err, 'targetfd_save', None)
    return fd if isinstance(fd, int) else sys.__stderr__.fileno()


class _Watchdog:
    """Ends the process, saying where, if one step runs past ``STUCK_S``."""

    def __init__(self, fd):
        self.fd, self.timer = fd, None

    def step(self, where):
        self.stop()
        self.timer = threading.Timer(STUCK_S, self._stuck, args=(where,))
        self.timer.name = 'dialog-sweep-watchdog'
        self.timer.daemon = True
        self.timer.start()

    def stop(self):
        if self.timer is not None:
            self.timer.cancel()

    def _stuck(self, where):
        head = f"\n[dialog sweep] stuck for {STUCK_S} s at: {where}\n"
        os.write(self.fd, head.encode('utf-8', 'replace'))
        faulthandler.dump_traceback(file=self.fd, all_threads=True)
        # On Windows a worker's stderr does not reach the CI log, so the
        # report also goes to a file, which CI prints when a job fails.
        folder = os.environ.get('OPENFLO_TEST_REPORTS')
        if folder:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f'dialog-sweep-stuck-{os.getpid()}.txt')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(head)
                f.flush()
                faulthandler.dump_traceback(file=f, all_threads=True)
        os._exit(3)


def _screen_that_fits(root, width, height):
    """``width`` x ``height``, or the real screen's size when the real screen
    cannot hold a window that big (Windows will not make a window larger
    than the screen, the windows-latest runner's is about 1024x768, and a
    window fitted to a screen it cannot have is resized by the system)."""
    probe = tk.Toplevel(root)
    try:
        probe.geometry(f"{width - 16}x{height - 72}+0+0")
        probe.update()
        got = (probe.winfo_width(), probe.winfo_height())
    finally:
        probe.destroy()
    if got == (width - 16, height - 72):
        return width, height
    return (int(root.tk.call('winfo', 'screenwidth', '.')),
            int(root.tk.call('winfo', 'screenheight', '.')))


def _buttons(win):
    for w in _walk(win):
        if w.winfo_class() not in ACTIONS or w.winfo_toplevel() is not win:
            continue
        try:
            text = str(w.cget('text')).strip()
            state = str(w.cget('state'))
        except tk.TclError:
            continue
        if not text or state == 'disabled' or text.startswith(SKIP_BUTTONS):
            continue
        yield w, text


def _editor(root, tmp_path):
    """A shown editor holding three loader-processed generated samples (QC,
    compensation, logicle; with Time and marker names), a gate and a child."""
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    for i, nm in enumerate(('alpha_1', 'alpha_2', 'beta_1')):
        path = _fcs(tmp_path / f'{nm}.fcs', i + 1)
        ed._on_loaded(nm, _loaded(path))
    ed.update()
    ed._set_active_sample('alpha_1')
    parent = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
                           'value': 1000.0, 'name': 'FL1 pos'}, audit=False)
    ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'op': '>',
                  'value': 1000.0, 'name': 'FL2 pos'}, parent_id=parent,
                 audit=False)
    ed.update()
    return ed


def _sweep(nonsense, tmp_path, monkeypatch, unhandled, virtual_screen,
           request):
    isolate_home(monkeypatch, tmp_path)
    root = _root()
    virtual_screen(*_screen_that_fits(root, 1920, 1080))
    watchdog = _Watchdog(_real_stderr(request))
    failures, pressed = [], 0
    try:
        watchdog.step("building the editor")
        ed = _editor(root, tmp_path)
        unhandled.on_tk(root)
        for menu, i, label in list(_menu_items(ed)):
            if label.startswith(NOT_A_WINDOW):
                continue
            before = set(_toplevels(root))
            del unhandled[:]
            watchdog.step(f"opening {label}")
            menu.invoke(i)
            _settle(root, ms=50, still_ms=50, cap_ms=1500)
            if unhandled:
                failures += [f"{label} (opening): {e}" for e in unhandled]
                del unhandled[:]
            for key, win in list(_toplevels(root).items()):
                if key in before or not win.winfo_exists():
                    continue
                if nonsense:
                    for w in _walk(win):
                        if w.winfo_class() in INPUTS:
                            try:
                                if str(w.cget('state')) == 'normal':
                                    w.delete(0, 'end')
                                    w.insert(0, NONSENSE)
                            except tk.TclError:
                                pass
                for btn, text in list(_buttons(win)):
                    if not (win.winfo_exists() and btn.winfo_exists()):
                        break
                    watchdog.step(f"{label} [{text}]")
                    _BASE_THREADS.clear()
                    _BASE_THREADS.update(t.ident for t in threading.enumerate())
                    try:
                        btn.invoke()
                    except tk.TclError:
                        continue          # the button's window went away
                    pressed += 1
                    _wait_for_work(root)
                    failures += [f"{label} [{text}]: {e}" for e in unhandled]
                    del unhandled[:]
                watchdog.step(f"closing {label}")
                for key2, w2 in list(_toplevels(root).items()):
                    if key2 not in before and w2.winfo_exists():
                        w2.destroy()
                root.update()
        assert pressed >= 30, pressed     # guard the guard
        assert not failures, "\n".join(failures)
    finally:
        watchdog.stop()
        root.destroy()


def test_every_window_runs_its_buttons_with_its_defaults(
        tmp_path, monkeypatch, no_dialogs, no_color_picker, engines_known,
        unhandled, virtual_screen, request):
    _sweep(False, tmp_path, monkeypatch, unhandled, virtual_screen, request)


def test_every_window_handles_nonsense_typed_into_it(
        tmp_path, monkeypatch, no_dialogs, no_color_picker, engines_known,
        unhandled, virtual_screen, request):
    _sweep(True, tmp_path, monkeypatch, unhandled, virtual_screen, request)
