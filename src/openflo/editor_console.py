"""Log pane + in-app Python console — a mixin for the gate editor.

First slice of decomposing ``ViewGateEditorWindow``: a cohesive method group
moved to its own module. It's a *mixin* (mixed into the editor), so the methods
still operate on the editor's widgets/state — the ``TYPE_CHECKING`` block below
declares the attributes/methods the editor provides so the type checker is
satisfied without runtime cost.
"""
from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import tkinter as tk


class ConsoleLogMixin:
    """Bottom log pane (mirrors stdout/stderr) + a persistent Python REPL.
    Mixed into :class:`ViewGateEditorWindow`, which builds the widgets and sets
    the state these methods use."""

    if TYPE_CHECKING:                       # provided by the composed editor
        _samples: dict
        _console: Any                       # code.InteractiveConsole | None
        _console_entry: tk.Entry
        _console_history: list
        _console_hist_idx: int
        _console_prompt: tk.StringVar
        _log_text: tk.Text
        _log_frame: tk.Widget
        _log_queue: queue.Queue
        _show_log_var: tk.BooleanVar

        def after(self, ms, func=None, *args): ...  # tk.Misc

    def _toggle_log(self):
        """Show/hide the log + console at the bottom of the left column."""
        if self._show_log_var.get():
            self._log_frame.grid()
        else:
            self._log_frame.grid_remove()

    def _toggle_log_shortcut(self):
        """Ctrl+` — flip the log/console visibility (keeps the menu var in
        sync, since _toggle_log reads it)."""
        try:
            self._show_log_var.set(not self._show_log_var.get())
            self._toggle_log()
        except Exception:
            pass

    def _make_console(self):
        """Build the persistent interpreter, pre-binding handy live objects."""
        import code
        ns = {'__name__': '__console__', 'editor': self, 'self': self,
              'samples': self._samples, 'np': np}
        try:
            import pandas as _pd
            ns['pd'] = _pd
        except Exception:
            pass
        return code.InteractiveConsole(locals=ns)

    def _console_run(self, event=None):
        """Run the entered line through the interpreter. Output / the repr of
        expressions / tracebacks all surface in the log via the stdout/stderr
        tee. A continuation (e.g. an open `def`) flips the prompt to `...`."""
        line = self._console_entry.get()
        self._console_entry.delete(0, 'end')
        if line.strip():
            self._console_history.append(line)
        self._console_hist_idx = len(self._console_history)
        self._append_log(f"{self._console_prompt.get()} {line}\n")
        if self._console is None:
            self._console = self._make_console()
        try:
            more = self._console.push(line)
        except SystemExit:
            more = False
        except BaseException:           # noqa: BLE001 — console must never crash the GUI
            more = False
        self._console_prompt.set('...' if more else '>>>')
        self._drain_log()               # flush output/repr immediately
        return 'break'

    def _console_history_prev(self, event=None):
        if not self._console_history:
            return 'break'
        self._console_hist_idx = max(0, self._console_hist_idx - 1)
        self._console_entry.delete(0, 'end')
        self._console_entry.insert(0, self._console_history[self._console_hist_idx])
        return 'break'

    def _console_history_next(self, event=None):
        if not self._console_history:
            return 'break'
        self._console_hist_idx = min(len(self._console_history),
                                     self._console_hist_idx + 1)
        self._console_entry.delete(0, 'end')
        if self._console_hist_idx < len(self._console_history):
            self._console_entry.insert(
                0, self._console_history[self._console_hist_idx])
        return 'break'

    def _clear_log(self):
        t = getattr(self, '_log_text', None)
        if t is None:
            return
        t.config(state='normal')
        t.delete('1.0', 'end')
        t.config(state='disabled')

    def _drain_log(self):
        """Append any queued stdout/stderr lines to the pane (main thread).
        Reschedules itself; cheap when the queue is empty."""
        try:
            chunks = []
            while True:
                try:
                    chunks.append(self._log_queue.get_nowait())
                except queue.Empty:
                    break
            if chunks:
                self._append_log(''.join(chunks))
        except Exception:
            pass
        finally:
            try:
                self.after(300, self._drain_log)
            except Exception:
                pass

    def _append_log(self, text):
        t = getattr(self, '_log_text', None)
        if t is None or not text:
            return
        try:
            t.config(state='normal')
            t.insert('end', text)
            # Cap the buffer so a long session doesn't grow unbounded.
            last = int(t.index('end-1c').split('.')[0])
            if last > 500:
                t.delete('1.0', f'{last - 500}.0')
            t.see('end')
            t.config(state='disabled')
        except Exception:
            pass
