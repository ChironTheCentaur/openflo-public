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
import tkinter as tk
from tkinter import ttk
from typing import Any


def run_async(widget, work, on_done=None, on_error=None, on_finally=None):
    """Run ``work()`` on a daemon thread; deliver its return value to
    ``on_done(result)`` — or any exception to ``on_error(exc)`` — back on
    ``widget``'s Tk thread. ``on_finally()`` always runs last on the Tk thread
    (use it to stop a spinner). Returns the started ``Thread``.

    All callbacks are optional and best-effort: if the widget is gone (window
    closed mid-run) the ``after`` call is swallowed rather than raising on the
    worker thread. Without ``on_error`` an exception goes to the window's
    error handler (``openflo.errors.report``), never nowhere."""
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
                    else:
                        from .errors import report
                        report(widget, e)
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
    except Exception:  # noqa: BLE001 - widget may be gone; treat as still alive
        return True


def _post(widget, fn):
    """Schedule ``fn`` on the widget's Tk thread; ignore a dead widget."""
    try:
        widget.after(0, fn)
    except Exception:  # noqa: BLE001 - widget or Tk loop may already be gone
        pass


# ── long jobs with progress and Cancel ──────────────────────────────────────

class JobCancelled(Exception):
    """Raised by ``Job.check()`` once the user pressed Cancel."""


class Job:
    """What a long job's work function is handed: whether Cancel was
    pressed, and a way to report progress. Both are safe off the Tk
    thread; the strip showing the job reads the progress on its own."""

    def __init__(self):
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._progress = None              # (done, total, text) or None
        self.finished = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    def check(self):
        """Stop here (raise `JobCancelled`) if Cancel was pressed."""
        if self._cancel.is_set():
            raise JobCancelled()

    def progress(self, done, total=None, text=None):
        """``done`` of ``total`` steps (``total`` None: no known length),
        with an optional line of text. Also a cancel point."""
        with self._lock:
            self._progress = (done, total, text)
        self.check()

    def read_progress(self):
        with self._lock:
            return self._progress


class BusyStrip(ttk.Frame):
    """A progress bar, a line of text and a Cancel button for one window's
    long job, shown only while the job runs (idle, the strip is empty and
    takes no room). ``run`` starts a job off the Tk thread; the window stays
    usable meanwhile, and Cancel stops the job at its next cancel point
    (``Job.check`` or ``Job.progress``), so nothing after it happens."""

    POLL_MS = 100

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.columnconfigure(1, weight=1)
        self._bar = ttk.Progressbar(self, mode='indeterminate', length=150)
        self._text = tk.StringVar(value='')
        self._label = ttk.Label(self, textvariable=self._text,
                                style='Muted.TLabel')
        self._cancel_btn = ttk.Button(self, text="Cancel", command=self.cancel)
        self._job = None
        self._disabled = []
        self._poll_id = None

    @property
    def running(self):
        return self._job is not None

    @property
    def text(self):
        return self._text.get()

    def run(self, work, on_done=None, on_error=None, on_cancel=None,
            text="Working…", disable=()):
        """Run ``work(job)`` off the Tk thread. On the Tk thread afterwards:
        ``on_done(result)``, ``on_error(exc)`` or, after Cancel,
        ``on_cancel()``. A job already running is cancelled first (the newer
        request wins). ``disable`` lists widgets greyed out while it runs."""
        job = self._start(text, disable)
        self._callbacks = (on_done, on_error, on_cancel)

        # The worker never touches Tk: it leaves its result on the job, and
        # the strip's own poll (on the Tk thread) delivers it.
        def _work():
            try:
                job.result = work(job)
            except BaseException as exc:                   # noqa: BLE001
                job.error = exc
            finally:
                job.finished.set()

        threading.Thread(target=_work, daemon=True,
                         name='openflo-job').start()
        self._poll_id = self.after(self.POLL_MS // 4, self._poll)
        return job

    def run_steps(self, steps, on_done=None, on_error=None, on_cancel=None,
                  text="Working…", disable=()):
        """Run the generator ``steps`` on the Tk thread, one step per turn
        of the event loop, for work that must stay on it (it draws through
        the editor). Each ``yield (done, total, text)`` reports progress and
        lets the window answer clicks, Cancel included; Cancel closes the
        generator at that yield (its ``finally`` and ``with`` blocks run).
        Its return value goes to ``on_done``."""
        job = self._start(text, disable)
        self._callbacks = (on_done, on_error, on_cancel)
        self._steps = steps

        def _step():
            self._poll_id = None
            if job is not self._job:
                return
            if job.cancelled:
                job.error = JobCancelled()
                self._deliver(job)
                return
            try:
                p = next(steps)
            except StopIteration as stop:
                job.result = stop.value
                self._deliver(job)
                return
            except Exception as exc:                       # noqa: BLE001
                job.error = exc
                self._deliver(job)
                return
            if p is not None:
                with job._lock:
                    job._progress = tuple(p)
                self._show_progress(job)
            self._poll_id = self.after(1, _step)

        self._poll_id = self.after(1, _step)
        return job

    def _start(self, text, disable):
        if self._job is not None:
            self.cancel()
            self._finish()
        job = self._job = Job()
        self._disabled = []
        for w in disable:
            try:
                if str(w.cget('state')) != 'disabled':
                    w.configure(state='disabled')
                    self._disabled.append(w)
            except tk.TclError:                              # noqa: BLE001
                pass
        self._text.set(text)
        self._base_text = text
        self._bar.configure(mode='indeterminate', value=0)
        self._bar.grid(row=0, column=0, padx=(0, 8), pady=2)
        self._label.grid(row=0, column=1, sticky='w')
        self._cancel_btn.grid(row=0, column=2, padx=(8, 0), pady=2)
        self._cancel_btn.configure(state='normal')
        self._bar.start(12)
        return job

    def cancel(self):
        job = self._job
        if job is not None:
            job.cancel()
            self._text.set("Cancelling…")
            try:
                self._cancel_btn.configure(state='disabled')
            except tk.TclError:                              # noqa: BLE001
                pass

    def _poll(self):
        self._poll_id = None
        job = self._job
        if job is None:
            return
        if job.finished.is_set():
            self._deliver(job)
            return
        self._show_progress(job)
        try:
            self._poll_id = self.after(self.POLL_MS, self._poll)
        except Exception:                                  # noqa: BLE001
            self._poll_id = None

    def _show_progress(self, job):
        p = job.read_progress()
        if p is None or job.cancelled:
            return
        done, total, text = p
        if total:
            if str(self._bar.cget('mode')) != 'determinate':
                self._bar.stop()
                self._bar.configure(mode='determinate', maximum=total)
            self._bar.configure(value=min(done, total))
        self._text.set(text or (f"{self._base_text}  {done} of {total}"
                                if total else self._base_text))

    def _deliver(self, job):
        on_done, on_error, on_cancel = self._callbacks
        self._finish()
        # Cancel takes effect at the work's next cancel point; work that
        # got past its last one finished, and its result is delivered.
        if isinstance(job.error, JobCancelled):
            if on_cancel is not None:
                on_cancel()
        elif job.error is not None:
            if on_error is not None:
                on_error(job.error)
            else:
                raise job.error
        elif on_done is not None:
            on_done(job.result)

    def _finish(self):
        self._job = None
        steps, self._steps = getattr(self, '_steps', None), None
        if steps is not None:
            steps.close()           # a superseded step job ends here
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except tk.TclError:                              # noqa: BLE001
                pass
            self._poll_id = None
        try:
            self._bar.stop()
            for w in (self._bar, self._label, self._cancel_btn):
                w.grid_remove()
        except Exception:  # noqa: BLE001 - busy strip may already be destroyed
            pass
        for w in self._disabled:
            try:
                w.configure(state='normal')
            except tk.TclError:                              # noqa: BLE001
                pass
        self._disabled = []


def wait_for_job(strip, timeout=60.0):
    """Pump the Tk loop until ``strip`` has no job running (tests, and a
    window closing that must let its job settle). False on timeout."""
    import time
    end = time.monotonic() + timeout
    try:
        while strip.running:
            if time.monotonic() > end:
                return False
            strip.update()
            time.sleep(0.005)
        strip.update()
    except tk.TclError:
        pass                    # the window closed once its job was done
    return True
