"""Keep Tk's clean-up on the main thread while background threads run.

A Tk variable (``StringVar`` and friends) or image is freed by the garbage
collector, and the collector runs on whichever thread happens to allocate
when a collection is due. In OpenFlo that is often a background thread: the
embedding warm-up, autosave, the plot-cache worker, sample loading, update
checks. tkinter's ``Variable.__del__`` and ``Image.__del__`` then make a Tcl
call from that thread, which tkinter hands to the main thread and waits for.
While the main thread is busy the background thread sits inside the
collection, holding up everything it was doing (the warm-up was seen stuck
for seconds), and when the main loop is not running the call fails with
"RuntimeError: main thread is not in main loop".

The same queue carries anything else a background thread needs done on the
main thread (``call_on_main_thread``), such as reporting an error that escaped
it (openflo.errors).

``install`` makes those two finalisers queue their Tcl clean-up when they run
off the main thread, and ``drain`` runs the queue on the main thread. The
editor drains it from its event loop (``schedule_drain``); the test suite
drains it after each test (tests/tk_main_thread.py). A finaliser that runs on
the main thread is unchanged.
"""
from __future__ import annotations

import collections
import functools
import threading

_deferred: collections.deque = collections.deque()   # finalisers for the main thread
_installed = False

# How often the editor's event loop runs the queued finalisers. The queue is
# almost always empty, so the check costs nothing.
DRAIN_MS = 500


def on_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def _finalise_on_main_thread(real_del):
    """Wrap a tkinter ``__del__`` so its Tcl calls only run on the main
    thread: elsewhere it is queued (keeping the object alive) for ``drain``."""
    def __del__(self):
        if on_main_thread():
            try:
                real_del(self)
            except Exception:           # noqa: BLE001 - its Tcl state is gone
                pass
        else:
            _deferred.append(functools.partial(real_del, self))
    __del__.__wrapped__ = real_del                   # type: ignore[attr-defined]
    return __del__


def install() -> None:
    """Route Tk variable and image finalisers to the main thread. Idempotent."""
    global _installed
    if _installed:
        return
    import tkinter
    tkinter.Variable.__del__ = _finalise_on_main_thread(    # type: ignore[method-assign]
        tkinter.Variable.__del__)
    tkinter.Image.__del__ = _finalise_on_main_thread(       # type: ignore[method-assign]
        tkinter.Image.__del__)
    _installed = True


def call_on_main_thread(fn) -> None:
    """Run ``fn()`` on the main thread at the next ``drain``. Any thread."""
    _deferred.append(fn)


def pending() -> int:
    """How many finalisers are waiting for the main thread."""
    return len(_deferred)


def drain() -> int:
    """Run every queued finaliser. Main thread only. Returns how many ran."""
    assert on_main_thread()
    ran = 0
    while _deferred:
        finalise = _deferred.popleft()
        try:
            finalise()
        except Exception:               # noqa: BLE001 - its Tcl state is gone
            pass
        del finalise
        ran += 1
    return ran


def schedule_drain(widget) -> None:
    """Drain the queue now and every ``DRAIN_MS`` from ``widget``'s event
    loop, for as long as the widget exists."""
    def tick():
        drain()
        try:
            widget.after(DRAIN_MS, tick)
        except Exception:               # noqa: BLE001 - the window is gone
            pass
    tick()
