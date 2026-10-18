"""Error messages people can read: what went wrong in plain words, with the
technical description one click away under "Show details".

Every failure a window reports goes through ``show_error``. The message box
used to show the exception itself ("KeyError: 'x'"); now the first line says
what OpenFlo was doing, the next says why in plain words (``plain_error``),
and the exception and its traceback wait behind the Details toggle, with a
button to copy them into a bug report.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import errno
import os
import tkinter as tk
import traceback
from tkinter import ttk


def _where(exc):
    name = getattr(exc, 'filename', None)
    return f" ({os.path.basename(str(name))})" if name else ""


def plain_error(exc) -> str:
    """One or two sentences saying, without programming terms, why ``exc``
    happened."""
    where = _where(exc)
    if isinstance(exc, FileNotFoundError):
        return f"The file or folder could not be found{where}."
    if isinstance(exc, PermissionError):
        return ("OpenFlo was not allowed to open or write the file" + where
                + ". It may be open in another program, or the folder may "
                  "be read-only.")
    if isinstance(exc, IsADirectoryError):
        return f"A folder was given where a file was expected{where}."
    if isinstance(exc, NotADirectoryError):
        return f"A file was given where a folder was expected{where}."
    if isinstance(exc, MemoryError):
        return ("There is not enough memory for this. Close other programs, "
                "or load fewer samples or events, and try again.")
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "The disk is full. Free some space and try again."
    if isinstance(exc, UnicodeError):
        return ("The file has characters OpenFlo could not read as text; it "
                "may not be the kind of file expected.")
    if isinstance(exc, KeyError):
        key = exc.args[0] if exc.args else ''
        return f"Something this needs was not found: {key!s}."
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        mod = getattr(exc, 'name', None) or ''
        return ("A component this needs is not installed"
                + (f" ({mod})" if mod else "") + ". Help, Environment "
                "lists what is installed.")
    text = str(exc).strip()
    if not text:
        return "Something went wrong that OpenFlo did not expect."
    text = text[0].upper() + text[1:]
    return text if text.endswith(('.', '!', '?', ')')) else text + '.'


def error_details(exc) -> str:
    """The technical description: the exception and, when it was raised,
    its traceback."""
    if exc is None:
        return ''
    if getattr(exc, '__traceback__', None) is not None:
        return ''.join(traceback.format_exception(type(exc), exc,
                                                  exc.__traceback__)).strip()
    return f"{type(exc).__name__}: {exc}"


class ErrorDialog(tk.Toplevel):
    """The message, then [Show details] [Copy details] ... [Close]. Details
    start hidden; Return and Escape close it."""

    def __init__(self, parent, title="Error", message='', details=''):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, False)
        self.details = details
        try:
            top = parent.winfo_toplevel() if parent is not None else None
            if top is not None and str(top) != '.':
                self.transient(top)
        except Exception:
            pass
        body = ttk.Frame(self, padding=(16, 14, 16, 8))
        body.pack(side='top', fill='both', expand=True)
        self.message_label = ttk.Label(body, text=message, wraplength=440,
                                       justify='left')
        self.message_label.pack(side='top', anchor='w')

        bar = ttk.Frame(self, padding=(16, 0, 16, 12))
        bar.pack(side='bottom', fill='x')
        self.close_button = ttk.Button(bar, text="Close", default='active',
                                       command=self.destroy)
        self.close_button.pack(side='right')
        self.details_button = self.copy_button = None
        self._box = tk.Text(body, height=10, width=72, wrap='none',
                            font='OpenFloMono')
        if details:
            self._box.insert('1.0', details)
            self._box.configure(state='disabled')
            self.details_button = ttk.Button(bar, text="Show details",
                                             command=self.toggle_details)
            self.details_button.pack(side='left')
            self.copy_button = ttk.Button(bar, text="Copy details",
                                          command=self.copy_details)
            self.copy_button.pack(side='left', padx=(6, 0))
            from .ui_tips import auto_tip
            auto_tip(self.details_button, "The technical description of the "
                     "error, for a bug report.")
            auto_tip(self.copy_button, "Copy the technical description, to "
                     "paste into a bug report (Help, Report a problem…).")
        self.bind('<Escape>', lambda _e: self.destroy())
        self.bind('<Return>', lambda _e: self.destroy())
        self.close_button.focus_set()

    @property
    def details_shown(self):
        return bool(self._box.winfo_manager())

    def toggle_details(self):
        if self.details_shown:
            self._box.pack_forget()
            self.resizable(True, False)
            if self.details_button is not None:
                self.details_button.configure(text="Show details")
        else:
            self._box.pack(side='top', fill='both', expand=True,
                           pady=(10, 0))
            self.resizable(True, True)
            if self.details_button is not None:
                self.details_button.configure(text="Hide details")
        self.update_idletasks()
        self.geometry('')                      # fit the new content

    def copy_details(self):
        try:
            self.clipboard_clear()
            self.clipboard_append(self.details)
        except Exception:
            pass


def _present(parent, title, message, details):
    """Show the dialog and wait until it is closed, as a message box would.
    (Tests replace this: see tests/conftest.py.)"""
    dlg = ErrorDialog(parent, title, message, details)
    try:
        dlg.grab_set()
    except Exception:
        pass
    dlg.wait_window()


def error_message(message='', exc=None, path=None) -> str:
    """The text shown: what failed, why in plain words, and the file."""
    parts = [p for p in (message.strip() if message else '',
                         plain_error(exc) if exc is not None else '') if p]
    text = "\n\n".join(parts)
    if path:
        text += f"\n\nFile: {path}"
    return text


def show_error(parent, title, message='', exc=None, path=None):
    """Report a failure: ``message`` says what OpenFlo was doing ("The
    session could not be saved."), ``exc`` why (in plain words, with the
    technical description under Show details), ``path`` the file."""
    _present(parent, title, error_message(message, exc, path),
             error_details(exc))
