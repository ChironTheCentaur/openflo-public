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
    """Return the stored preferences dict, or ``{}`` when the file is missing or unreadable."""
    try:
        with open(_prefs_path(), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):  # an unreadable or corrupt prefs file reads as defaults ({})
        return {}


def write_pref(key, value):
    """Store one preference, keeping the others, and save the file atomically.

    A transient read error skips the save instead of overwriting the file;
    a failed write is noted, not raised.
    """
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
    except Exception as exc:  # noqa: BLE001 - noted; a failed save must not break the setting change
        from .errors import note
        note("saving preferences", exc)


def load_qc_enabled():
    """Whether the editor runs acquisition QC (drift, flow-rate and margin
    events) on every file it loads. Off unless turned on in Preferences:
    QC removes events before any gate, FlowJo has no such step, so with it on
    the same gates count fewer events here than in FlowJo."""
    return bool(read_prefs().get('load_qc', False))


def load_density_binned():
    """Whether the exact Gaussian-KDE density colouring (the 'True Gaussian
    KDE' plot option and the Quick preview scatter) is evaluated on a binned
    grid instead. Off unless turned on in Preferences > Performance: the
    binned form is much faster but approximate (within about 0.5% of the
    exact density's peak)."""
    return bool(read_prefs().get('density_binned', False))


# How two statistics are computed (Preferences > Statistics). Each defaults
# to FlowJo's definition, so the same gates give FlowJo's numbers; the
# second choice is the textbook / robust-statistics form.
# FlowJo's GeoMean depends on the display axis; a sample imported from a
# FlowJo workspace takes it on that workspace's axes (wsp_gate_coords.wsp_stat_axes).
GEOMEAN_FORMS = (
    ('display', 'Mean on the display axis, back-transformed (FlowJo; '
                'FlowJo imports use the workspace axis)'),
    ('log', 'exp(mean of log x) over positive events (textbook)'),
)
# rCV defaults to the robust-statistics form (the literature standard);
# FlowJo's percentile form stays a choice.
RCV_FORMS = (
    ('mad', '1.4826 x median absolute deviation (robust statistics)'),
    ('percentile', 'Half the 15.87 to 84.13 percentile range (FlowJo)'),
)


def load_stat_forms():
    """``{'geomean': key, 'rcv': key}`` from Preferences. A missing or
    unknown value reads as the default (gating.DEFAULT_STAT_FORMS): FlowJo's
    GeoMean, the MAD form of rCV."""
    prefs = read_prefs()
    geo = prefs.get('stat_geomean')
    rcv = prefs.get('stat_rcv')
    return {
        'geomean': geo if geo in dict(GEOMEAN_FORMS) else 'display',
        'rcv': rcv if rcv in dict(RCV_FORMS) else 'mad',
    }


def load_westgard_r4s_across():
    """Whether bead tracking also judges Westgard R-4s across two runs in a
    row (Preferences > Statistics). On by default, so a log with one bead
    level per run still has a range check; the literature form (the box
    "Westgard R-4s: literature form only" ticked) judges R-4s only within a
    run, across two or more bead levels."""
    return not bool(read_prefs().get('westgard_r4s_literature', False))
