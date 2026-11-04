"""One place for errors OpenFlo catches, so none of them disappears unseen.

Three kinds of error reach here:

* ``note``: an error OpenFlo handled and carried on from (a preference that
  would not save, an optional engine that would not load). It is not worth
  interrupting anyone for, but it must leave a line in the log, so a problem
  that keeps happening can be found. Use it instead of ``except Exception:
  pass`` wherever the failure could matter to a result or a file.
* ``report``: an error nothing handled. It goes to the window's error
  handler (``report_callback_exception``, which the editor sets to its error
  window and the scrubbed problem report), or to stderr without a window.
* ``install_thread_hook``: an error that escapes a background thread's
  function, which Python would otherwise only print. It is handed to the
  window's error handler on the main thread, the same as a Tk callback's.

tests/test_error_handling.py keeps new code from swallowing errors silently.
"""
from __future__ import annotations

import sys
import threading
import traceback

# Errors noted since start-up, newest last (kept short: tests and the
# diagnostics read it, nothing depends on its length).
_noted: list = []
_NOTED_MAX = 200


def note(what: str, exc: BaseException | None = None) -> None:
    """Record an error that was handled: one line on stderr (the editor's
    log pane mirrors stderr) and an entry in ``noted()``. Never raises."""
    if exc is None:
        exc = sys.exc_info()[1]
    line = f"[handled] {what}"
    if exc is not None:
        line += f": {type(exc).__name__}: {exc}"
    _noted.append(line)
    del _noted[:-_NOTED_MAX]
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:                   # noqa: BLE001 - no usable stderr
        pass


def noted() -> list:
    """The lines ``note`` recorded, oldest first."""
    return list(_noted)


def report(widget, exc: BaseException) -> None:
    """Hand an exception nothing else handled to ``widget``'s error handler.
    Main thread only. Falls back to printing the traceback."""
    handler = None
    try:
        handler = widget._root().report_callback_exception
    except Exception:                   # noqa: BLE001 - no window: print it
        pass
    if handler is not None:
        handler(type(exc), exc, exc.__traceback__)
    else:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def install_thread_hook(widget) -> None:
    """Send exceptions that escape a background thread to ``widget``'s error
    handler, on the main thread. (The handler prints the traceback, so this
    replaces Python's own printing rather than adding to it.)"""
    from . import tk_threads
    previous = threading.excepthook

    def hook(args):
        if args.exc_type is SystemExit:
            return
        if args.exc_value is None:
            previous(args)
            return
        exc = args.exc_value
        tk_threads.call_on_main_thread(lambda: report(widget, exc))

    threading.excepthook = hook
