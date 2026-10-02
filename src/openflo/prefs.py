"""User-preferences persistence (``~/.openflo/prefs.json``).

Tiny, dependency-free helpers shared by the editor and its dialogs — kept out
of gui.py so a dialog module doesn't import the whole GUI just to read a flag.
"""
from __future__ import annotations

import json
import os


def _prefs_path():
    d = os.path.join(os.path.expanduser('~'), '.openflo')
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, 'prefs.json')


def read_prefs():
    try:
        with open(_prefs_path(), encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def write_pref(key, value):
    # Read the existing prefs, but distinguish failure modes: a transient
    # lock/permission error (OSError — common on Windows with AV/indexers)
    # must NOT be treated as "no prefs", or we'd overwrite the whole file with
    # just this one key and wipe every other stored preference. Only a genuinely
    # absent or already-corrupt file is safe to start from empty.
    try:
        with open(_prefs_path(), encoding='utf-8') as f:
            prefs = json.load(f)
    except FileNotFoundError:
        prefs = {}
    except json.JSONDecodeError:
        prefs = {}          # existing file is already garbage — safe to reset
    except OSError:
        return              # transient lock — abort rather than clobber
    if not isinstance(prefs, dict):
        prefs = {}
    prefs[key] = value
    # Write atomically (tmp + replace) so a crash mid-dump can't leave a
    # truncated/corrupt prefs.json that the next read turns into {}.
    try:
        path = _prefs_path()
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass
