"""Versioning + migration for the ``.flowsession`` save format.

Kept deliberately Tk-free so the GUI, a standalone upgrade script
(``scripts/migrate_session.py``), and the test suite can all import it without
a display.

**Continuity contract.** ``SESSION_KEYS`` and ``SESSION_VERSION`` pin the
schema that :meth:`openflo.gui.ViewGateEditorWindow._session_state` emits. The
continuity test (``tests/test_session_continuity.py``) fails the moment the
writer's key set or version drifts from these constants — so a schema change
can never be accidental. When you *do* change the schema on purpose:

1. add/remove the key in ``_session_state`` and update ``SESSION_KEYS``;
2. bump ``SESSION_VERSION``;
3. register a ``v -> v+1`` step in ``_SESSION_MIGRATIONS`` so old saved
   sessions keep loading (downstream users are not stranded);
4. note it in ``CHANGELOG.md`` under the new version.

The loader auto-upgrades older files on open and tells the user; the standalone
script upgrades files in place for batch/headless use.

It also holds the parts of a sample entry that come from the sample's data
alone (``column_records``, ``qc_records``, ``comp_entry``) and the JSON
repair every session write goes through (``json_safe``), so the gate editor
and a headless document write a session the same way.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

from .errors import note
from .scales import OPEN
from .session_check import session_structure_problem

SESSION_FORMAT = 'openflo-session'
SESSION_VERSION = 3

# Every top-level key a CURRENT session carries. Locked by the continuity test.
SESSION_KEYS = frozenset({
    'format', 'version', 'created', 'active_sample', 'samples', 'sample_gates',
    'channel_scale', 'channel_range', 'channel_labels', 'plot_mode',
    'x_channel', 'y_channel', 'color_channel', 'downsample_display',
    'downsample_propagate', 'max_points', 'show_removed', 'contour_scatter',
    'contour_outliers', 'hist_y_mode', 'cluster_labels', 'audit',
    'view_tabs', 'layout', 'app_version', 'fixes_applied', 'fixes_declined',
})


class SessionError(ValueError):
    """A session that cannot be opened; the message says why in plain
    words. Callers that open a session catch this one class."""


class SessionVersionError(SessionError):
    """Raised when a session can't be migrated (e.g. written by a newer
    OpenFlo, or a missing migration step)."""


class SessionDamagedError(SessionError):
    """Raised when a session's structure is damaged
    (``session_check.session_structure_problem``): the message names the
    sample or gate and the field."""


def _v1_to_v2(data: dict) -> dict:
    """v2 adds the tray's tabs (``view_tabs``) and the layout editor's
    pages (``layout``). A v1 session had neither: one All tab, no layout."""
    data.setdefault('view_tabs', None)
    data.setdefault('layout', None)
    return data


def _v2_to_v3(data: dict) -> dict:
    """v3 records which OpenFlo wrote the session (``app_version``) and, in
    each sample entry, which wrote that sample (``app_version``), the
    stamp of each computed column (``result_stamps``, see result_stamp) and
    its gate counts and statistics (``saved_stats``, see saved_stats). A v2
    session says none of it: its versions are unknown (None, older than
    every version), its columns carry no stamps and it saved no numbers.

    It also records the fixes (fixes.py) applied to the session's results
    (``fixes_applied``, fix ids read from the results, informational) and
    declined (``fixes_declined``: ``{'fix', 'sample', 'result', 'key'}`` per
    result, or a bare fix id from a session saved before declines were per
    result, which declines that fix for every result), and in
    each sample entry the originals of corrected columns
    (``result_originals``, ``{column: {'stamp', 'column'}}``). A v2 session
    has had none applied."""
    data.setdefault('app_version', None)
    data.setdefault('fixes_applied', [])
    data.setdefault('fixes_declined', [])
    for entry in data.get('samples') or []:
        if isinstance(entry, dict):
            entry.setdefault('app_version', None)
            entry.setdefault('result_stamps', {})
            entry.setdefault('result_originals', {})
            entry.setdefault('saved_stats', None)
    return data


# Ordered upgrade steps: ``_SESSION_MIGRATIONS[v]`` lifts a v-schema dict to
# v+1; every bump ships its migration. Each step must be pure and idempotent
# on its own version.
_SESSION_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    1: _v1_to_v2, 2: _v2_to_v3}


def migrate_session(data: dict) -> tuple[dict, list[str]]:
    """Upgrade a parsed ``.flowsession`` dict to ``SESSION_VERSION`` in place.

    Returns ``(data, notes)`` where ``notes`` lists the steps applied (empty if
    the file was already current). Raises :class:`SessionVersionError` if the
    file is newer than this build, or an intermediate migration is missing,
    and :class:`SessionDamagedError` if the upgraded session's structure is
    damaged (``session_check``).
    """
    if not isinstance(data, dict) or data.get('format') != SESSION_FORMAT:
        fmt = data.get('format') if isinstance(data, dict) else None
        raise SessionVersionError(
            f"Not an OpenFlo session (format={fmt!r}).")
    notes: list[str] = []
    try:
        v = int(data.get('version', 1) or 1)
    except (TypeError, ValueError):
        v = 1
    if v > SESSION_VERSION:
        raise SessionVersionError(
            f"This session was written by a newer OpenFlo "
            f"(schema v{v}; this build supports v{SESSION_VERSION}). "
            "Update OpenFlo to open it.")
    while v < SESSION_VERSION:
        step = _SESSION_MIGRATIONS.get(v)
        if step is None:
            raise SessionVersionError(
                f"No migration registered from session schema v{v} "
                f"to v{v + 1}.")
        data = step(data)
        notes.append(f"upgraded session schema v{v} → v{v + 1}")
        v += 1
    data['version'] = SESSION_VERSION
    data.setdefault('format', SESSION_FORMAT)
    problem = session_structure_problem(data)
    if problem:
        raise SessionDamagedError(
            f"This session file is damaged and can't be opened: {problem}.")
    return data, notes


# ── Per-sample compensation (an additive field inside each sample entry) ──
#
# A sample entry records the matrix the sample was compensated with, so a
# reopened session, or a sample handed to another window, is compensated the
# same way. Without it a reload would fall back to the FCS's own $SPILL, which
# would silently replace a matrix the user had applied (bead-derived, edited, or
# imported from a .wsp) under gates drawn on the other one. Entries written
# before this field existed have none and keep that $SPILL behaviour.

def comp_entry(sample) -> dict | None:
    """The compensation applied to `sample` as a JSON-ready dict
    ``{'channels', 'matrix', 'source'}``, or None when it has none."""
    m = getattr(sample, 'comp_matrix', None)
    ch = list(getattr(sample, 'comp_channels', None) or [])
    if m is None or not ch:
        return None
    return {'channels': [str(c) for c in ch],
            'matrix': [[float(v) for v in row] for row in m],
            'source': getattr(sample, 'compensation_source', None) or 'manual'}


def comp_from_entry(d) -> dict | None:
    """Inverse of :func:`comp_entry`, validated: a square, finite matrix with
    one row per channel. Anything else is None (the sample then falls back to
    its $SPILL) rather than an exception that aborts a whole restore."""
    if not isinstance(d, dict):
        return None
    import numpy as np
    ch = d.get('channels')
    try:
        m = np.asarray(d.get('matrix'), dtype=float)
    except (TypeError, ValueError):
        return None
    if (not isinstance(ch, list) or not ch or m.ndim != 2
            or m.shape != (len(ch), len(ch)) or not np.isfinite(m).all()):
        return None
    return {'channels': [str(c) for c in ch], 'matrix': m,
            'source': str(d.get('source') or 'manual')}


# ── A sample entry's data records, and JSON-safe writing ──

def column_records(s) -> dict[str, Any]:
    """Per-column records of sample `s`: its transforms, its FlowJo stat
    axes and its compensation."""
    entry = {}
    # The transform each column carries. A sample reloaded from its
    # raw FCS comes back on the loader's logicle; without this, one
    # that landed first would put the whole editor on logicle and
    # gates drawn in asinh coordinates would select nothing. Additive, inside
    # the sample entry (like basename / rel_path / processed_data), so
    # SESSION_KEYS and SESSION_VERSION are unchanged and an older
    # OpenFlo ignores it.
    rec = getattr(s, 'data_transforms', None)
    if rec is not None:
        entry['transforms'] = {str(c): dict(spec)
                               for c, spec in rec.items()
                               if c in s.data.columns}
    # The axes a FlowJo workspace declared for it, which its
    # FlowJo-form GeoMean is taken on (wsp_gate_coords.wsp_stat_axes): a
    # reload from the raw FCS cannot get them back. Additive, like
    # 'transforms'; an older OpenFlo ignores it.
    axes = getattr(s, 'stat_axes', None)
    if axes:
        entry['stat_axes'] = {str(c): dict(a) for c, a in axes.items()
                              if c in s.data.columns}
    comp = comp_entry(s)
    if comp is not None:
        entry['comp'] = comp
    entry.update(instrument_records(s))
    return entry


def instrument_records(s) -> dict[str, Any]:
    """Which copy of an .lmd file (or which data set of a file holding
    several) sample `s` holds, and whether the
    instrument already compensated its values: a sample reopened from its
    sidecar reads that copy's keywords and still refuses a second matrix
    (sample_prep.sidecar_sample). Additive keys, written only when set."""
    entry: dict[str, Any] = {}
    which = getattr(s, 'lmd_data_set', None)
    if which:
        entry['lmd_data_set'] = str(which)
    # Which data set of a file holding several the values are: reopened
    # from the raw file or its sidecar, it is that data set again.
    number = getattr(s, 'set_number', None)
    if number:
        entry['set_number'] = int(number)
        member = container_member(s, int(number))
        if member:
            entry['container_member'] = member
    if getattr(s, 'instrument_compensated', False):
        entry['instrument_compensated'] = True
    return entry


def container_member(s, number: int) -> str:
    """The archive name of the FCS member sample `s` was read from, when
    its file is a BD Accuri file (fcs_containers), else ''. Saved with its
    number, so the session reopens that well even if the file gained wells
    since (fcs_containers.resolve_member)."""
    from .fcs_containers import member_name
    from .paths import is_fcs_container
    path = str(getattr(s, 'source_fcs', None) or getattr(s, 'path', '') or '')
    if not is_fcs_container(path) or not os.path.isfile(path):
        return ''
    return member_name(path, number)


def qc_records(s) -> dict[str, Any]:
    """Whether acquisition QC ran on sample `s`, and what it removed."""
    # Whether acquisition QC ran on it, so a reload from the raw FCS
    # gives the same events whatever Preferences says by then.
    entry: dict[str, Any] = {'qc_applied': bool(getattr(s, 'qc_applied', False))}
    # And how many events it removed: a sample reopened from its
    # processed sidecar holds only the events kept, so this count
    # (the load status, the .wsp export's QC note) cannot be
    # recomputed from it. Additive, like 'qc_applied'.
    qc_removed = int(getattr(s, 'qc_removed', 0) or 0)
    if qc_removed:
        entry['qc_removed'] = qc_removed
    return entry


def report_nonfinite(where: str, value: Any, replacement: Any) -> None:
    """Note (``errors.note``) a value `json_safe` replaced at `where`."""
    note(f"session: {where} was {value}, which JSON cannot represent; "
         f"stored as {replacement}", None)


def json_safe(obj, report=None, _where='state'):
    """Replace values RFC 8259 cannot carry, reporting each one.

    ``Infinity`` / ``-Infinity`` become the ±1e12 sentinel this codebase
    already uses for an open gate bound; ``NaN`` becomes ``null``, since a
    missing value is what it means. A numpy scalar becomes its Python value:
    an int64 a Leiden / FlowSOM population takes from its column would make
    every save of the session fail ("not JSON serializable"). Everything else is
    returned unchanged.
    """
    import numpy as np
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj):
            if report:
                report(_where, 'NaN', 'null')
            return None
        if math.isinf(obj):
            sentinel = -OPEN if obj < 0 else OPEN
            if report:
                report(_where, obj, sentinel)
            return sentinel
        return obj
    if isinstance(obj, dict):
        return {(k.item() if isinstance(k, np.generic) else k):
                json_safe(v, report, f'{_where}.{k}')
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, report, f'{_where}[{i}]')
                for i, v in enumerate(obj)]
    return obj
