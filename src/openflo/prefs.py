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
    except OSError:
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
    except Exception as exc:
        from .errors import note
        note("saving preferences", exc)


def load_qc_enabled():
    """Whether the editor runs acquisition QC (drift, flow-rate and margin
    events) on every file it loads. Off unless turned on in Preferences:
    QC removes events before any gate, FlowJo has no such step, so with it on
    the same gates count fewer events here than in FlowJo."""
    return bool(read_prefs().get('load_qc', False))


# How two statistics are computed (Preferences > Statistics). Each defaults
# to FlowJo's definition, so the same gates give FlowJo's numbers; the
# second choice is the textbook / robust-statistics form.
# FlowJo's GeoMean depends on the display axis; a sample imported from a
# FlowJo workspace takes it on that workspace's axes (pipeline.wsp_stat_axes).
GEOMEAN_FORMS = (
    ('display', 'Mean on the display axis, back-transformed (FlowJo; '
                'FlowJo imports use the workspace axis)'),
    ('log', 'exp(mean of log x) over positive events (textbook)'),
)
RCV_FORMS = (
    ('percentile', 'Half the 15.87 to 84.13 percentile range (FlowJo)'),
    ('mad', '1.4826 x median absolute deviation (robust statistics)'),
)


def load_stat_forms():
    """``{'geomean': key, 'rcv': key}`` from Preferences. A missing or
    unknown value reads as FlowJo's form."""
    prefs = read_prefs()
    geo = prefs.get('stat_geomean')
    rcv = prefs.get('stat_rcv')
    return {
        'geomean': geo if geo in dict(GEOMEAN_FORMS) else 'display',
        'rcv': rcv if rcv in dict(RCV_FORMS) else 'percentile',
    }
