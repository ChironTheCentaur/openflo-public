"""The waits every test shares: run the Tk event loop until something holds
or for a while, and wait on a worker thread with no event loop.

pump_until: run the event loop until ``done()`` holds or time runs out.
settle: run the event loop for a fixed time, so deferred callbacks ran.
wait_until: poll ``done()`` with no event loop (worker threads, pools).
tests/test_tk_waits_shared.py keeps tests from growing their own."""
from __future__ import annotations

import time


def pump_until(widget, done, timeout=30.0, pause=0.01):
    """Run ``widget``'s Tk event loop until ``done()`` holds or ``timeout``
    seconds pass, and return ``done()``'s last answer, so a caller writes
    ``assert pump_until(root, lambda: ...)``. ``done()`` is asked before each
    round, so nothing is pumped when it already holds."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and not done():
        widget.update()
        time.sleep(pause)
    return done()


def settle(widget, seconds=1.0, pause=0.01):
    """Run ``widget``'s Tk event loop for ``seconds``, so deferred callbacks
    (load settle, status rewrites) have run before anything is asserted -- a
    note that only lived in the status bar is gone by then."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        widget.update()
        time.sleep(pause)


def wait_until(done, timeout=10.0, pause=0.01):
    """Poll ``done()`` until it holds or ``timeout`` seconds pass, with no
    event loop; return its last answer."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if done():
            return True
        time.sleep(pause)
    return done()
