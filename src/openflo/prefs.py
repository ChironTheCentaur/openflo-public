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
    prefs = read_prefs()
    prefs[key] = value
    try:
        with open(_prefs_path(), 'w', encoding='utf-8') as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass
