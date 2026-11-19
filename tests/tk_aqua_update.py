"""``update()`` returns on macOS even while a test's editor is withdrawn.

GUI tests build their editor hidden: the helpers call ``root.withdraw()`` and
then ``ed.withdraw()`` on the ``ViewGateEditorWindow`` itself
(tests/test_transform_records.py ``_new_editor``, test_layout_editor.py
``_editor_or_skip``, test_exit_autosave.py, test_saved_stats.py and many
more, imported across the suite), so no window flashes up on a developer's
desktop. The app itself never does that: it withdraws only its empty root
and a tooltip, and shows the editor.

On macOS (Tk's "aqua" windowing system) a withdrawn toplevel full of widgets
makes Tk's own ``update`` loop for ever. Measured on GitHub's macos runners
(Tk 8.6.16): about 5,000 events a second, all Tk-internal idle work, no Python
callback and no ``after`` job among them. Run alone, that idle work runs
out; but ``update`` also polls for window events, and on aqua every poll of
the application queues more of it, because the editor's views still need
displaying and a withdrawn window can never draw. It stops the moment the
editor is deiconified. So ``settle()``, ``_pump()``, ``_reopen``,
``_restore_session`` and the other helpers that call ``root.update()`` never
returned, and the macOS job hung in about nine files.

``install`` therefore replaces ``tkinter.Misc.update`` for the test run. On
any interpreter whose ``tk windowingsystem`` is not aqua (asked at each call;
Linux and Windows) the original runs, unchanged. On aqua it runs the same
loop as Tk's ``update`` (tkCmds.c ``Tk_UpdateObjCmd``) one event at a time,
window, file and timer events before idle work as ``Tcl_DoOneEvent`` orders
them, and returns where Tk's would: when nothing at all is pending. It also
returns when nothing but idle work has arrived for ``IDLE_ONLY_PASSES``
passes in a row: every real event has been handled by then, and what keeps
coming is the hidden windows' redisplay, which no amount of waiting will
finish. A Python idle callback, a timer or a window event that turns up in
between starts the count again, so callbacks still run as under ``update``.

Setting ``OPENFLO_TEST_FORCE_AQUA_UPDATE=1`` takes the aqua path on any
system, which is how the replacement is checked against the real GUI tests
without a Mac.
"""
from __future__ import annotations

import os

# Passes of idle work alone, after the last real event, before calling the
# rest the hidden windows' redisplay. One idle pass runs every idle handler
# queued so far, so a burst of widget redraws is one pass, and a geometry
# change that ripples down nested frames is one pass per level: tens at most.
# At the rate measured on aqua, a hundred passes is about 20 ms.
IDLE_ONLY_PASSES = 100

FORCE_ENV = 'OPENFLO_TEST_FORCE_AQUA_UPDATE'

_installed = False


def _flags():
    import _tkinter
    real = (_tkinter.WINDOW_EVENTS | _tkinter.FILE_EVENTS
            | _tkinter.TIMER_EVENTS | _tkinter.DONT_WAIT)
    idle = _tkinter.IDLE_EVENTS | _tkinter.DONT_WAIT
    return real, idle, _tkinter.ALL_EVENTS | _tkinter.DONT_WAIT


def on_aqua(interp):
    """True if ``interp``'s Tk draws through aqua (or the run forces it)."""
    if os.environ.get(FORCE_ENV) == '1':
        return True
    from tkinter import TclError
    try:
        return interp.call('tk', 'windowingsystem') == 'aqua'
    except TclError:               # a tkinter.Tcl() interpreter: no Tk loaded
        return False


def update_until_only_redisplay_is_left(interp):
    """Tk's ``update``, except that it stops once only idle work keeps
    arriving (see the module docstring). Returns the number of passes that
    found idle work alone at the end, for the tests."""
    real, idle, every = _flags()
    idle_only = 0
    while idle_only < IDLE_ONLY_PASSES:
        if interp.dooneevent(real):
            idle_only = 0          # a real event: carry on, as update would
            continue
        if interp.dooneevent(idle):
            idle_only += 1
            continue
        # Nothing pending at all, which is where Tk's update ends. Like it,
        # sync with the display (``update idletasks`` does) and look once
        # more before returning.
        interp.call('update', 'idletasks')
        if not interp.dooneevent(every):
            break
        # It found something, but perhaps only idle work the sync queued:
        # count the pass, so this branch cannot keep the loop going alone.
        idle_only += 1
    return idle_only


def install():
    """Make ``Misc.update`` return on aqua while editors are withdrawn."""
    global _installed
    if _installed:
        return
    import tkinter
    real_update = tkinter.Misc.update

    def update(self):
        if on_aqua(self.tk):
            update_until_only_redisplay_is_left(self.tk)
        else:
            real_update(self)

    update.__wrapped__ = real_update                 # type: ignore[attr-defined]
    update.__doc__ = real_update.__doc__
    tkinter.Misc.update = update                     # type: ignore[method-assign]
    _installed = True
