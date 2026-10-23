"""Keep two windows from silently overwriting one session file.

Two layers, both about a session the user opened or saved by name:

* A **lock file** beside the session (``name.flowsession.lock``) names the
  computer, user and process that has it open, so a second window, or a
  second person on a shared or synced folder, is told before opening it.
  It is a courtesy: a synced folder may copy it late, and a crash leaves it
  behind (one left by a process that is gone, on this computer, is taken
  over without asking).
* A **stamp** (size and modified time) of the file as this window last read
  or wrote it. Before a save, a different stamp means someone else saved in
  between, and the save asks instead of overwriting. This is the layer that
  protects the data, wherever the lock file got to.

Pure stdlib and no Tk, so it is tested without a display.
"""
from __future__ import annotations

import getpass
import json
import os
import socket
import time
import uuid

LOCK_SUFFIX = '.lock'


def lock_path(session_path):
    """The lock file for the session at `session_path`."""
    return os.fspath(session_path) + LOCK_SUFFIX


def file_stamp(path):
    """``(size, mtime_ns)`` of `path`, or None when it is missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def changed_on_disk(path, stamp):
    """True when the file at `path` is not the one `stamp` recorded: it was
    replaced, changed or deleted since. A None `stamp` (nothing recorded)
    is never a change."""
    if stamp is None:
        return False
    return file_stamp(path) != tuple(stamp)


def new_token():
    """A token naming one window's hold on a lock."""
    return uuid.uuid4().hex


def _user():
    try:
        return getpass.getuser()
    except Exception:                               # noqa: BLE001 - user/host name is informational
        return ''


def _host():
    try:
        return socket.gethostname()
    except Exception:                               # noqa: BLE001 - user/host name is informational
        return ''


def pid_alive(pid):
    """Best-effort: is a process with this PID running on this computer?"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        if os.name == 'nt':
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if h:
                ctypes.windll.kernel32.CloseHandle(h)
                return True
            return False
        os.kill(pid, 0)
        return True
    except PermissionError:                         # alive, someone else's
        return True
    except Exception:  # noqa: BLE001 - unknown process state counts as not running
        return False


def read_lock(session_path):
    """The holder recorded in the session's lock file, or None when there
    is none. An unreadable lock file reads as a holder with no details, so
    it is still reported rather than ignored."""
    lp = lock_path(session_path)
    try:
        with open(lp, encoding='utf-8') as f:
            info = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:                               # noqa: BLE001
        return {}
    return info if isinstance(info, dict) else {}


def is_stale(info):
    """A lock left by a process on this computer that is no longer running
    (OpenFlo crashed or was killed) holds nothing."""
    return (bool(info) and info.get('host') == _host()
            and not pid_alive(info.get('pid')))


def held_here(info):
    """A lock this OpenFlo process holds for another of its windows. Not a
    conflict to ask about: windows in one process are one user, and the
    changed-on-disk check still keeps them from overwriting each other."""
    return (bool(info) and info.get('host') == _host()
            and info.get('pid') == os.getpid())


def _write_lock(session_path, token, exclusive):
    info = {'host': _host(), 'user': _user(), 'pid': os.getpid(),
            'token': token, 'since': time.time()}
    lp = lock_path(session_path)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive
                                        else os.O_TRUNC)
    fd = os.open(lp, flags, 0o644)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(info, f)
    return info


def acquire(session_path, token, force=False):
    """Take the lock on `session_path` for `token`.

    Returns None when this window now holds it (it was free, already ours,
    or stale) or another window of this process does (held_here), else the other holder's details (a dict, possibly empty when
    the lock file can't be read). ``force=True`` takes it over regardless
    ("Open anyway"). A folder where no lock can be written (read-only media)
    is not an error: there is nothing to warn others with, so None."""
    held = read_lock(session_path)
    if held is not None and not force:
        if held_here(held) and held.get('token') != token:
            return None
        if held.get('token') == token or is_stale(held):
            force = True
        else:
            return held
    try:
        _write_lock(session_path, token, exclusive=not force)
    except FileExistsError:
        # Another window took it between the read and the write.
        return read_lock(session_path) or {}
    except OSError:
        return None
    return None


def release(session_path, token):
    """Remove the lock on `session_path` if `token` holds it. A lock some
    other window took over ("Open anyway") is left to that window."""
    if not session_path or not token:
        return
    held = read_lock(session_path)
    if held is None or held.get('token') != token:
        return
    try:
        os.remove(lock_path(session_path))
    except OSError:
        pass


def describe_holder(info):
    """One line naming who holds a lock, for a dialog."""
    if not info:
        return "another OpenFlo window (its lock file can't be read)"
    who = info.get('user') or 'someone'
    host = info.get('host') or 'another computer'
    since = info.get('since')
    when = ''
    if isinstance(since, (int, float)):
        when = time.strftime(' since %Y-%m-%d %H:%M', time.localtime(since))
    here = ' on this computer' if host == _host() else f' on {host}'
    return f"{who}{here}{when}"
