"""``update()`` returns on macOS while a test's editor is withdrawn.

On aqua, Tk's own ``update`` never returned in GUI tests whose editor was
withdrawn: each poll for window events queues more idle redisplay for the
hidden window, so idle work never runs out (tests/tk_aqua_update.py
explains). There is no Mac here, so these tests drive the replacement
``tkinter.Misc.update`` with a fake interpreter that behaves that way, and
check that Tk's own update loop, run against the same fake, does not return.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

_tkinter = pytest.importorskip('_tkinter')
import tkinter  # noqa: E402

from tests import tk_aqua_update  # noqa: E402

WINDOW, FILE, TIMER, IDLE = (_tkinter.WINDOW_EVENTS, _tkinter.FILE_EVENTS,
                             _tkinter.TIMER_EVENTS, _tkinter.IDLE_EVENTS)

# Far more events than any update of these fakes needs; reaching it means the
# loop would have run for ever.
GUARD = 100_000


class Hung(Exception):
    pass


class FakeInterp:
    """Just enough of a Tk interpreter for ``update``: ``dooneevent`` with
    Tcl's flags and order (window, file and timer events, then the idle
    handlers queued so far, all at once), and ``call`` for ``tk
    windowingsystem``, ``update`` and ``update idletasks`` (Tk's
    ``Tk_UpdateObjCmd`` loop).

    ``withdrawn`` models aqua with a withdrawn editor: every poll for window
    events queues the hidden window's redisplay as idle work again."""

    def __init__(self, windowingsystem='aqua', withdrawn=True):
        self.windowingsystem = windowingsystem
        self.withdrawn = withdrawn
        self.window_events: list = []       # names, handled in order
        self.timers: list = []              # callables
        self.idle_callbacks: list = []      # callables (after_idle)
        self.redisplay = False              # Tk-internal idle work queued
        self.handled: list = []
        self.calls: list = []               # what was asked of call()
        self.events = 0

    def dooneevent(self, flags):
        self.events += 1
        if self.events > GUARD:
            raise Hung(f"no return after {GUARD} events")
        if flags & WINDOW:
            if self.withdrawn and self.windowingsystem == 'aqua':
                self.redisplay = True       # Aqua's poll of the application
            if self.window_events:
                self.handled.append(self.window_events.pop(0))
                return 1
        if flags & TIMER and self.timers:
            self.timers.pop(0)()
            return 1
        if flags & IDLE and (self.redisplay or self.idle_callbacks):
            batch, self.idle_callbacks = self.idle_callbacks, []
            self.redisplay = False
            for callback in batch:
                callback()
            return 1
        return 0

    def call(self, *args):
        self.calls.append(args)
        if args == ('tk', 'windowingsystem'):
            return self.windowingsystem
        if args == ('update',):
            return self._tk_update(_tkinter.ALL_EVENTS | _tkinter.DONT_WAIT)
        if args == ('update', 'idletasks'):
            return self._tk_update(IDLE)
        raise AssertionError(f"unexpected Tcl call {args}")

    def _tk_update(self, flags):
        while True:
            while self.dooneevent(flags):
                pass
            if not self.dooneevent(flags):  # after XSync
                return ''


def _with_real_events(interp):
    """Queue a window event, a timer and an idle callback; the timer queues
    another idle callback, which queues another window event, so real events
    keep arriving after idle passes."""
    interp.window_events.append('configure')
    interp.idle_callbacks.append(lambda: interp.handled.append('after_idle'))

    def timer():
        interp.handled.append('after 0')
        interp.idle_callbacks.append(chained_idle)

    def chained_idle():
        interp.handled.append('chained after_idle')
        interp.window_events.append('expose')
    interp.timers.append(timer)
    return interp


EXPECTED = ['configure', 'after 0', 'after_idle', 'chained after_idle',
            'expose']


@pytest.fixture(autouse=True)
def _not_forced(monkeypatch):
    """These tests choose the windowing system themselves."""
    monkeypatch.delenv(tk_aqua_update.FORCE_ENV, raising=False)


def test_the_fake_is_the_hang_tk_update_has_on_aqua():
    """The premise: against this fake, Tk's own update loop never returns
    while the editor is withdrawn, and does once it is shown; update
    idletasks returns either way, as measured on aqua."""
    with pytest.raises(Hung):
        _with_real_events(FakeInterp()).call('update')
    shown = _with_real_events(FakeInterp(withdrawn=False))
    shown.call('update')
    assert shown.handled == EXPECTED
    FakeInterp().call('update', 'idletasks')


def test_update_returns_on_aqua_with_a_withdrawn_editor():
    interp = _with_real_events(FakeInterp())
    tkinter.Misc.update(SimpleNamespace(tk=interp))
    assert interp.handled == EXPECTED
    assert interp.events < 1000
    assert ('update',) not in interp.calls


def test_update_stops_only_after_idle_work_alone_for_a_while():
    """It keeps going through idle passes while real events still arrive, and
    returns only after IDLE_ONLY_PASSES of idle work with nothing else."""
    interp = FakeInterp()
    left = tk_aqua_update.update_until_only_redisplay_is_left(
        _with_real_events(interp))
    assert left == tk_aqua_update.IDLE_ONLY_PASSES
    assert interp.handled == EXPECTED


def test_update_on_aqua_with_every_window_shown_ends_like_tks():
    """Nothing withdrawn: idle work runs out, and the replacement ends where
    Tk's update does, once nothing at all is pending."""
    interp = _with_real_events(FakeInterp(withdrawn=False))
    left = tk_aqua_update.update_until_only_redisplay_is_left(interp)
    assert interp.handled == EXPECTED
    assert left < tk_aqua_update.IDLE_ONLY_PASSES
    assert ('update', 'idletasks') in interp.calls


@pytest.mark.parametrize('system', ['x11', 'win32'])
def test_other_windowing_systems_keep_tks_own_update(system):
    interp = _with_real_events(FakeInterp(system, withdrawn=False))
    tkinter.Misc.update(SimpleNamespace(tk=interp))
    assert interp.calls == [('tk', 'windowingsystem'), ('update',)]
    assert interp.handled == EXPECTED


def test_an_interpreter_without_tk_keeps_tks_own_update():
    """tkinter.Tcl(): no ``tk`` command to ask, so no aqua."""
    interp = FakeInterp('x11', withdrawn=False)

    def call(*args):
        if args == ('tk', 'windowingsystem'):
            raise tkinter.TclError('invalid command name "tk"')
        return FakeInterp.call(interp, *args)
    interp.call = call                      # type: ignore[method-assign]
    tkinter.Misc.update(SimpleNamespace(tk=interp))
    assert interp.calls == [('update',)]


def test_the_environment_can_force_the_aqua_path(monkeypatch):
    """How the replacement is run against the real GUI tests off a Mac."""
    monkeypatch.setenv(tk_aqua_update.FORCE_ENV, '1')
    interp = _with_real_events(FakeInterp('x11', withdrawn=False))
    tkinter.Misc.update(SimpleNamespace(tk=interp))
    assert ('update',) not in interp.calls
    assert interp.handled == EXPECTED
