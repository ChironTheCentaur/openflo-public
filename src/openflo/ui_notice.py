"""Saying a file was written, without a pop-up to dismiss (UI 35).

Every export used to end in a "Saved: <path>" message box, about forty of
them, each one more click. Now the window's own status line and the editor's
status bar say what was written, and the status bar offers Open folder.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk


def _editor_of(widget):
    """The editor that owns ``widget``'s window (it has ``_show_saved``)."""
    w = widget
    for _ in range(12):
        if w is None:
            return None
        if hasattr(w, '_show_saved'):
            return w
        w = getattr(w, 'master', None)
    return None


def saved_text(path, what="Saved"):
    return f"{what} {os.path.basename(str(path))}"


def report_saved(widget, path, what="Saved"):
    """Say ``path`` was written: in ``widget``'s window's status line when it
    has one (``status_var``), and in the editor's status bar with an Open
    folder button. Returns the text shown."""
    text = saved_text(path, what)
    try:
        top = widget.winfo_toplevel()
    except Exception:
        top = None
    for w in (widget, top):
        var = getattr(w, 'status_var', None)
        if var is not None and _editor_of(w) is not w:
            try:
                var.set(text)
            except tk.TclError:
                pass
            break
    ed = _editor_of(widget)
    if ed is not None:
        try:
            ed._show_saved(str(path), text)
        except Exception:  # noqa: BLE001 - saved-file notice is a nicety
            pass
    return text


def folder_command(path):
    """The command that shows ``path`` in the system's file manager."""
    path = os.path.abspath(str(path))
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if sys.platform == 'darwin':
        return ['open', '-R', path] if os.path.exists(path) else ['open',
                                                                  folder]
    if sys.platform == 'win32':
        return (['explorer', f'/select,{path}'] if os.path.isfile(path)
                else ['explorer', folder])
    return ['xdg-open', folder]


def open_folder(path):
    """Show ``path``'s folder (the file selected where the system can)."""
    try:
        subprocess.Popen(folder_command(path), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return True
    except OSError:
        return False
