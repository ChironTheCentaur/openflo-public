"""Run a blocking job off the Tk thread and marshal the result back.

A single helper that replaces the ad-hoc ``threading.Thread(target=work).start()``
+ ``self.after(0, lambda: …)`` pattern scattered across the editor and its
dialogs. The work runs on a daemon thread; ``on_done`` / ``on_error`` /
``on_finally`` are delivered on the Tk event loop via ``widget.after``, so they
can safely touch widgets. ``widget`` only needs an ``after(ms, fn)`` method.

Why a free function (not just a mixin method): the editor uses it through a
``ComputeMixin.run_async`` wrapper that also drives the busy bar, but the
standalone tool windows (ui_voltage / ui_synth / ui_compare / ui_statistics …)
are plain ``tk.Toplevel`` subclasses with no access to that mixin — they call
this directly.
"""
from __future__ import annotations

import threading


def run_async(widget, work, on_done=None, on_error=None, on_finally=None):
    """Run ``work()`` on a daemon thread; deliver its return value to
    ``on_done(result)`` — or any exception to ``on_error(exc)`` — back on
    ``widget``'s Tk thread. ``on_finally()`` always runs last on the Tk thread
    (use it to stop a spinner). Returns the started ``Thread``.

    All callbacks are optional and best-effort: if the widget is gone (window
    closed mid-run) the ``after`` call is swallowed rather than raising on the
    worker thread."""
    def _worker():
        try:
            result = work()
        except Exception as exc:                       # noqa: BLE001
            def _fail(e=exc):
                # The window may have been closed while we computed: its `after`
                # callback can still fire but its child widgets are gone, so
                # touching them raises a stale-command TclError. Skip the
                # callbacks entirely if the widget no longer exists.
                if not _alive(widget):
                    return
                try:
                    if on_error is not None:
                        on_error(e)
                finally:
                    if on_finally is not None:
                        on_finally()
            _post(widget, _fail)
            return

        def _ok(r=result):
            if not _alive(widget):
                return
            try:
                if on_done is not None:
                    on_done(r)
            finally:
                if on_finally is not None:
                    on_finally()
        _post(widget, _ok)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def _alive(widget):
    """True if the widget still exists (or doesn't expose winfo_exists)."""
    try:
        return bool(widget.winfo_exists())
    except Exception:
        return True


def _post(widget, fn):
    """Schedule ``fn`` on the widget's Tk thread; ignore a dead widget."""
    try:
        widget.after(0, fn)
    except Exception:
        pass
