"""File > New window: how another OpenFlo is started beside this one.

The new window is a separate program, as a second launch is: its own
autosave and session locks (editor_session, session_lock), so two sessions
are worked on side by side without sharing state. It is told it was opened
from a window (``NEW_WINDOW_ENV``), so it starts empty instead of offering
to resume a session.
"""
from __future__ import annotations

import os
import sys

# Set in the new program's environment; the editor reads and removes it
# before offering to resume a session (LifecycleMixin._offer_resume_session).
NEW_WINDOW_ENV = 'OPENFLO_NEW_WINDOW'


def new_window_command(executable: str | None = None,
                       frozen: bool | None = None,
                       platform: str | None = None,
                       safe_path: bool | None = None) -> list[str]:
    """The command that starts another OpenFlo window.

    The same Python runs ``-m openflo.gui`` (the Windows installer's
    shortcut starts its private Python that way); on Windows its
    console-less ``pythonw.exe`` is used when it sits beside ``python.exe``,
    so no console window opens with it. When this program was started with
    ``-P`` (``sys.flags.safe_path``, as the installer's shortcut does, to
    keep the working folder off the import path) the new one is too. A
    bundled build (``sys.frozen``; none ships today) is started as itself."""
    exe = sys.executable if executable is None else executable
    frozen = bool(getattr(sys, 'frozen', False)) if frozen is None else frozen
    plat = sys.platform if platform is None else platform
    if safe_path is None:
        safe_path = bool(getattr(sys.flags, 'safe_path', False))
    if frozen:
        return [exe]
    if plat == 'win32':
        folder, name = os.path.split(exe)
        if name.lower() == 'python.exe':
            quiet = os.path.join(folder, 'pythonw.exe')
            if os.path.isfile(quiet):
                exe = quiet
    return [exe, *(['-P'] if safe_path else []), '-m', 'openflo.gui']


def new_window_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The new program's environment: this one's, marked as a new window,
    with the folder this ``openflo`` package was imported from first on
    ``PYTHONPATH`` so ``-m openflo.gui`` finds the same code (a run from the
    source folder is not installed)."""
    env = dict(os.environ if base is None else base)
    env[NEW_WINDOW_ENV] = '1'
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p]
    if src not in parts:
        parts.insert(0, src)
    env['PYTHONPATH'] = os.pathsep.join(parts)
    return env


def started_as_new_window() -> bool:
    """Whether this program was started by File > New window. The mark is
    removed once read, so programs this one starts (pipeline runs) do not
    inherit it."""
    return os.environ.pop(NEW_WINDOW_ENV, None) == '1'
