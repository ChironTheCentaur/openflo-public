"""Free every Tk interpreter the suite makes on the main thread, never another.

A Tk root always sits in reference cycles (the root's ``children`` dict and
each widget's ``master``, callbacks bound to the objects that own them), so
the cyclic garbage collector, not a refcount, frees its interpreter. The
collector runs on whichever thread happens to allocate when a collection is
due, and the suite has plenty of background threads that outlive their test:
warm-up, autosave, update checks, job readers, async tasks. When one of those
threads frees a Tk interpreter, Tcl aborts the whole process::

    Tcl_AsyncDelete: async handler deleted by the wrong thread
    Fatal Python error: Aborted

Under pytest-xdist that is "[gw0] node down: Not properly terminated" and
"worker 'gw0' crashed while running <whatever test was running then>", which
is why the Windows CI legs crashed in a different, innocent test each time.
The same collections, when they free a Tk variable or image rather than the
interpreter, show up as "Exception ignored in Variable.__del__ / Image.__del__:
RuntimeError: main thread is not in main loop" warnings.

``install`` keeps a reference to the interpreter of every ``tkinter.Tk`` made
(``tkinter.Tcl`` and OpenFlo's own window are ``Tk`` too), so no other thread
can drop the last one. ``release``, run on the main thread after each test,
collects the cycles there and lets go of each interpreter that nothing else
still holds, so it is freed right there; one still in use is kept for a later
release. After a test that made no interpreter it skips the collection, which
can take seconds in a long-lived worker (13% of the suite's time on four
workers).

Two things kept most interpreters held for good, and a held interpreter is
not free: its display connection stays open and every later test's event
loop polls it, so the suite ran about four times slower (13 minutes became
47 on four workers):

* A Python callback registered with Tcl (a variable trace, a widget command)
  is held from C, where the collector cannot see it, and it holds the window
  that holds the interpreter. Once a window is destroyed nothing can call
  those callbacks any more, so ``release`` deletes them (and any pending
  ``after``) from a destroyed window's interpreter, and collects again.
* A Tk variable or image freed on a background thread asks the main thread
  to delete it from Tcl and waits up to a second for an answer that never
  comes, with that thread inside a collection the whole time; meanwhile
  ``gc.collect()`` on the main thread returns at once without collecting.
  ``install`` makes those finalisers hand their Tcl clean-up to the main
  thread instead, and ``release`` runs it.

``release_leftovers``, run when a test module ends, destroys windows its
tests left open and frees them the same way.
"""
from __future__ import annotations

import gc
import sys
import weakref

from openflo import tk_threads

_interpreters: list = []        # (interpreter, weak reference to its Tk root)
_cut: set = set()               # id() of held interpreters already cut loose
_made = 0                       # interpreters made since the last release
_installed = False


def _on_main_thread():
    return tk_threads.on_main_thread()


def install():
    """Hold every Tk interpreter made from now on until ``release``."""
    global _installed
    if _installed:
        return
    import tkinter
    real_init = tkinter.Tk.__init__

    def init(self, *args, **kwargs):
        global _made
        try:
            real_init(self, *args, **kwargs)
        finally:
            # Also when the root failed part-way (Tk loses start-up races
            # under xdist): the interpreter exists and must die here too.
            interp = self.__dict__.get('tk')
            if interp is not None:
                _interpreters.append((interp, weakref.ref(self)))
                _made += 1

    init.__wrapped__ = real_init                     # type: ignore[attr-defined]
    tkinter.Tk.__init__ = init                       # type: ignore[method-assign]
    # Tk variable and image finalisers hand their Tcl clean-up to the main
    # thread; the app installs the same (openflo.tk_threads).
    tk_threads.install()
    _installed = True


def _run_deferred():
    tk_threads.drain()


def _free_unused():
    """Collect on this thread, then let go of each held interpreter nothing
    else uses, so it is freed right here."""
    # Break the cycles that hold interpreters (and their Tk variables and
    # images) while this thread still owns the only reference that counts.
    # Finalisers queued by other threads first: each holds its object (and so
    # its window) until it has run.
    _run_deferred()
    gc.collect()
    still_used = []
    while _interpreters:
        interp, root = _interpreters.pop()
        # Two references: this name and getrefcount's argument.
        if sys.getrefcount(interp) > 2:
            still_used.append((interp, root))
        else:
            _cut.discard(id(interp))
        del interp                     # the last reference: freed right here
    _interpreters.extend(still_used)


def _destroyed(interp):
    from tkinter import TclError
    try:
        return not interp.getboolean(interp.call('winfo', 'exists', '.'))
    except TclError:
        return True                    # "application has been destroyed"


def _cut_loose(interp):
    """Drop what Tcl holds from C for a destroyed window: pending ``after``
    callbacks and every Python command registered with its interpreter."""
    from tkinter import TclError
    try:
        for after_id in interp.splitlist(interp.call('after', 'info')):
            interp.call('after', 'cancel', after_id)
    except TclError:
        pass
    try:
        commands = interp.splitlist(interp.call('info', 'commands'))
    except TclError:
        return
    # tkinter names each command it registers repr(id(callback)) + its name,
    # and Tcl's own commands never start with a digit. matplotlib's Tk
    # backend registers one more per interpreter, its blit.
    for name in commands:
        if name[:1].isdigit() or name.startswith('mpl_blit_'):
            try:
                interp.deletecommand(name)
            except TclError:
                pass


def _cut_destroyed():
    """Cut loose each held, destroyed window not cut yet; say if any was."""
    cut = False
    for interp, _root in _interpreters:
        if id(interp) not in _cut and _destroyed(interp):
            _cut_loose(interp)
            _cut.add(id(interp))
            cut = True
    return cut


def release(force=False):
    """Free, on this thread, every held interpreter nothing else uses.

    Must run on the main thread, the one that made the interpreters.

    After a test that made no interpreter and left no finaliser queued it
    does nothing unless ``force``: a full collection can take seconds in a
    long-lived worker, and an interpreter still held is safe where it is
    until the next GUI test or the end of the module."""
    global _made
    if not tk_threads.pending() and not (_interpreters and (_made or force)):
        return
    assert _on_main_thread()
    _made = 0
    _free_unused()
    if _cut_destroyed():
        _free_unused()


def release_leftovers():
    """Destroy, on this thread, every window a finished test left open, then
    free what that lets go.

    Run on the main thread once nothing still needs the windows: when a test
    module ends, after its own module fixtures are torn down. An interpreter
    still in use after that stays held."""
    if not _interpreters:
        return
    assert _on_main_thread()
    from tkinter import TclError
    for _interp, root_ref in list(_interpreters):
        root = root_ref()
        if root is not None:
            try:
                root.destroy()
            except TclError:
                pass                    # already destroyed
        del root
    del _interp, root_ref              # the loop's names would keep one held
    release(force=True)


def held():
    """How many interpreters are waiting for a main-thread release."""
    return len(_interpreters)
