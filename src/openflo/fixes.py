"""Fixes that reach saved sessions: the registry, and the scan that finds
the results a fix would change (Tk-free).

A saved session keeps its results (re-fitted gates, cluster and cell-cycle
columns, ...) as they were made, so a bug fixed later never reaches them by
itself. Each result carries a stamp (result_stamp): the step that made it
and the OpenFlo version that ran it. Each :class:`Fix` here names the step
it affects and the version it was fixed in; a result is affected when its
step matches and it was made before that version (or by an unknown one).

Rules, from the design agreed for this:

* **Applied once, never forced.** The editor shows what a fix would change
  and the user picks. A corrected result's stamp moves to the running
  version, so it no longer matches; the session also records the fix id in
  ``fixes_applied`` (or ``fixes_declined``) and the scan skips it from then
  on.
* **By version range.** A result stamped 1.1 matches every fix whose
  ``fixed_in`` is after 1.1; once corrected (stamp now 2.6), the jump to 2.7
  matches only the fixes in (2.6, 2.7]. Fixes on one result apply oldest
  ``fixed_in`` first.
* **Recompute or shift.** A ``recompute`` fix reruns the step with the
  current code from the raw inputs, so running it again gives the same
  answer. A ``shift`` fix applies a known change between versions and
  assumes the earlier fixes are in: it lists them in ``requires`` and is
  offered only together with them (or on a result made after them).
* **Default changes are flagged, not applied** (``kind='default-change'``).
* A fix only counts once this build has it: the running version must be at
  or after ``fixed_in``.

``redo(model, target)`` gets the editor (anything with ``_sample_gates``,
``_samples`` and ``_get_df``) and a target whose ``'current'`` is the result
as it stands after the earlier fixes in its chain (a gate dict, or a
column's values); it returns ``{'new': ..., 'why': None | reason}``.
"""
from __future__ import annotations

import copy
import functools
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import result_stamp as rs

KINDS = ('error', 'default-change')
MODES = ('recompute', 'shift')


@dataclass(frozen=True)
class Fix:
    """One fix in the registry."""
    id: str
    title: str
    step: str                       # the stamp step; ending '.' is a prefix
    fixed_in: str                   # the first version with the fix
    kind: str = 'error'             # 'error' | 'default-change'
    mode: str = 'recompute'         # 'recompute' | 'shift'
    requires: tuple[str, ...] = ()  # a shift fix: the fixes it builds on
    redo: Callable[[Any, dict], dict] | None = None
    note: str = ''

    @property
    def informational(self) -> bool:
        return self.kind == 'default-change'

    def matches_step(self, step) -> bool:
        step = str(step or '')
        if self.step.endswith('.'):
            return step.startswith(self.step)
        return step == self.step

    def affects(self, stamp) -> bool:
        """Whether a result with ``stamp`` was made by this fix's step before
        it was fixed (an unknown version counts as before)."""
        return (stamp is not None and self.matches_step(stamp.get('step'))
                and rs.made_before(stamp.get('version'), self.fixed_in))


# ── Where a result's inputs come from ───────────────────────────────────────

def gate_channels(gate) -> list:
    return [c for c in (gate.get('channel'), gate.get('x_channel'),
                        gate.get('y_channel')) if c]


def parent_events(model, sample, gate):
    """The events of ``gate``'s parent population on ``sample``, all of them
    (no display downsample), with the gate's channels."""
    chans = gate_channels(gate)
    return model._get_df(sample, chans[0], chans[1] if len(chans) > 1 else None,
                         downsample=False, gate_parent=gate.get('parent_id'))


def gate_numbers(model, sample, gate):
    """``(events in gate, % of parent)`` for ``gate`` on ``sample``, or
    ``(None, None)`` when its parent's events cannot be had."""
    import numpy as np

    from .pipeline import gate_to_mask
    try:
        df = parent_events(model, sample, gate)
        if any(c not in df.columns for c in gate_channels(gate)):
            return None, None
        m = np.asarray(gate_to_mask(gate, df), dtype=bool)
    except Exception:                                    # noqa: BLE001
        return None, None
    n = int(m.sum())
    return n, (100.0 * n / m.size if m.size else float('nan'))


def column_summary(values) -> str:
    """A short summary of a column's values for the review window."""
    import numpy as np
    import pandas as pd
    s = pd.Series(values)
    if len(s) == 0:
        return 'empty'
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 12:
        v = s.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 'no finite values'
        return f"median {np.median(v):.4g}, mean {v.mean():.4g}"
    top = s.value_counts(normalize=True).head(3)
    return ', '.join(f"{k}: {100 * p:.1f}%" for k, p in top.items())


# ── The registry ────────────────────────────────────────────────────────────

def _redo_refit(model, target) -> dict:
    """Re-run the re-fit of one gate on its sample's parent population.

    It calls gate_refit.refit, the module function, so it runs whatever
    re-fit code this build has. PR #78 ("Re-fit per sample missed the valley
    before a bright population") is what this fix exists for; until #78 is
    merged the redo reproduces the old placement, and once it is, this picks
    up #78's code with no change here."""
    import numpy as np

    from . import gate_refit
    gate = target['current']
    df = parent_events(model, target['sample'], gate)
    chans = gate_channels(gate)
    if any(c not in df.columns for c in chans):
        return {'new': copy.deepcopy(gate), 'why': 'its channels are missing'}
    data = {c: df[c].to_numpy(dtype=float) for c in chans}
    new, why = gate_refit.refit(gate, data)
    if not np.isfinite([float(new.get(k, 0.0)) for k in ('value', 'lo', 'hi')
                        if k in new]).all():
        return {'new': copy.deepcopy(gate), 'why': 'the re-fit was not finite'}
    return {'new': new, 'why': why}


REGISTRY: list[Fix] = [
    # PR #78. fixed_in is the release after the version this was written in
    # (2.7.0b1); the fix is in effect from that release on.
    Fix(id='refit-valley', title="Re-fit per sample missed the valley before "
        "a bright population", step='gate.refit', fixed_in='2.7.0b2',
        kind='error', mode='recompute', redo=_redo_refit),
]


def by_id(registry=None) -> dict:
    return {f.id: f for f in (REGISTRY if registry is None else registry)}


def _version_key():
    return functools.cmp_to_key(rs.compare_versions)


def in_effect(fix, running=None) -> bool:
    """Whether this build has ``fix`` (it runs ``fixed_in`` or later)."""
    running = rs.app_version() if running is None else running
    return not rs.made_before(running, fix.fixed_in)


# ── The results a session holds ─────────────────────────────────────────────

def results_of(source) -> list:
    """Every stamped result of ``source`` (an editor, or a parsed session
    dict): ``{'sample', 'kind': 'gate'|'column', 'key', 'stamp', 'name'}``.
    A gate moved by hand since it was stamped is not a result any more."""
    out = []
    if isinstance(source, dict):
        for name, gl in (source.get('sample_gates') or {}).items():
            for g in gl or []:
                st = rs.live_gate_stamp(g) if isinstance(g, dict) else None
                if st is not None:
                    out.append({'sample': name, 'kind': 'gate',
                                'key': str(g.get('id')), 'stamp': st,
                                'name': g.get('name') or ''})
        for e in source.get('samples') or []:
            if not isinstance(e, dict):
                continue
            for col, st in rs.stamps_from_entry(
                    e.get('result_stamps')).items():
                out.append({'sample': e.get('name'), 'kind': 'column',
                            'key': col, 'stamp': st, 'name': col})
        return out
    for name in list(getattr(source, '_sample_order', None)
                     or getattr(source, '_sample_gates', {})):
        gates = (getattr(source, '_sample_gates', {}) or {}).get(name) or {}
        order = (getattr(source, '_sample_gate_order', {}) or {}).get(
            name, list(gates))
        for gid in order:
            g = gates.get(gid)
            st = rs.live_gate_stamp(g) if g is not None else None
            if g is not None and st is not None:
                out.append({'sample': name, 'kind': 'gate', 'key': gid,
                            'stamp': st, 'name': g.get('name') or ''})
        s = (getattr(source, '_samples', {}) or {}).get(name)
        if s is not None:
            for col, st in rs.column_stamps(s).items():
                out.append({'sample': name, 'kind': 'column', 'key': col,
                            'stamp': st, 'name': col})
    return out


def settled(source) -> tuple[list, list]:
    """``(fixes_applied, fixes_declined)`` of a session dict or editor."""
    if isinstance(source, dict):
        a, d = source.get('fixes_applied'), source.get('fixes_declined')
    else:
        a = getattr(source, '_applied_fixes', None)
        d = getattr(source, '_declined_fixes', None)
    return ([str(x) for x in a or [] if x], [str(x) for x in d or [] if x])


def _has(known, rid, offered, stamp) -> bool:
    """Whether a result with ``stamp`` has, or is offered, fix ``rid``."""
    r_fix = known.get(rid)
    if r_fix is None:
        return False
    return rid in offered or not rs.made_before(stamp.get('version'),
                                                r_fix.fixed_in)


def scan(source, registry=None, *, include_settled=False,
         running=None) -> list:
    """The results of ``source`` (an editor or a session dict) a fix would
    change, one target per (result, fix), each result's fixes oldest
    ``fixed_in`` first::

        {'fix', 'title', 'kind', 'mode', 'informational', 'sample',
         'result': 'gate'|'column', 'key', 'name', 'stamp', 'version',
         'unknown', 'status': 'open'|'applied'|'declined'}

    Fixes already in the session's ``fixes_applied`` or ``fixes_declined``
    are left out unless ``include_settled`` (the editor's "Review fixes"),
    where they come back with that status."""
    reg = [f for f in (REGISTRY if registry is None else registry)
           if in_effect(f, running)]
    known = by_id(registry)
    applied, declined = settled(source)
    key = _version_key()
    out = []
    for r in results_of(source):
        st = r['stamp']
        hits = []
        for f in reg:
            if not f.affects(st):
                continue
            status = ('applied' if f.id in applied else
                      'declined' if f.id in declined else 'open')
            if status != 'open' and not include_settled:
                continue
            hits.append((f, status))
        hits.sort(key=lambda h: key(h[0].fixed_in))
        # A shift fix stands on the fixes it requires: offered with them, or
        # on a result made after them. Dropping one can strand another.
        changed = True
        while changed:
            changed = False
            offered = {f.id for f, _s in hits}
            for f, s in list(hits):
                if f.mode == 'shift' and not all(
                        _has(known, q, offered, st) for q in f.requires):
                    hits.remove((f, s))
                    changed = True
                    break
        for f, status in hits:
            out.append({'fix': f.id, 'title': f.title, 'kind': f.kind,
                        'mode': f.mode, 'informational': f.informational,
                        'requires': list(f.requires),
                        'sample': r['sample'], 'result': r['kind'],
                        'key': r['key'], 'name': r['name'], 'stamp': st,
                        'version': st.get('version'),
                        'unknown': st.get('version') is None,
                        'status': status})
    return out


# ── Proposals, applying them, and what follows ──────────────────────────────

def _current(model, sample, kind, key):
    if kind == 'gate':
        return copy.deepcopy(model._sample_gates[sample][key])
    return model._samples[sample].data[key].to_numpy(copy=True)


def _numbers(model, sample, kind, value):
    if kind == 'gate':
        return gate_numbers(model, sample, value)
    return column_summary(value)


def plan(model, targets, registry=None) -> list:
    """Rows for the review window, one per result: each result's open fixes
    run in turn (oldest first), each on the previous one's output::

        {'sample', 'result', 'key', 'name', 'stamp', 'fixes': [ids],
         'titles', 'old', 'new', 'why', 'before', 'after', 'informational',
         'unknown', 'status', 'changed'}

    A default-change target becomes its own informational row (no ``new``).
    ``before`` / ``after`` are ``(events, % of parent)`` for a gate, a short
    summary for a column."""
    known = by_id(registry)
    rows, groups = [], {}
    for t in targets:
        if t['informational']:
            rows.append(dict(t, fixes=[t['fix']], titles=[t['title']],
                             old=None, new=None, why=None, before=None,
                             after=None, changed=False))
            continue
        g = groups.get((t['sample'], t['result'], t['key']))
        if g is None:
            g = groups[(t['sample'], t['result'], t['key'])] = []
            rows.append({'_group': g, **t})
        g.append(t)
    out = []
    for row in rows:
        g = row.pop('_group', None)
        if g is None:
            out.append(row)
            continue
        sample, kind, key = row['sample'], row['result'], row['key']
        try:
            old = _current(model, sample, kind, key)
        except Exception as exc:                          # noqa: BLE001
            out.append(dict(row, fixes=[t['fix'] for t in g],
                            titles=[t['title'] for t in g], old=None,
                            new=None, why=f"not loaded ({exc})", before=None,
                            after=None, changed=False))
            continue
        cur, whys = old, []
        for t in g:
            fix = known[t['fix']]
            try:
                res = fix.redo(model, dict(t, current=copy.deepcopy(cur)))
            except Exception as exc:                      # noqa: BLE001
                res = {'new': cur, 'why': f"could not redo: {exc}"}
            if res.get('why'):
                whys.append(str(res['why']))
            if res.get('new') is not None:
                cur = res['new']
        changed = (rs.geometry_sig(cur) != rs.geometry_sig(old)
                   if kind == 'gate' else not _same_values(cur, old))
        out.append(dict(row, fixes=[t['fix'] for t in g],
                        titles=[t['title'] for t in g],
                        status=('open' if any(t['status'] == 'open' for t in g)
                                else g[0]['status']),
                        old=old, new=cur, why='; '.join(whys) or None,
                        before=_numbers(model, sample, kind, old),
                        after=_numbers(model, sample, kind, cur),
                        changed=changed))
    return out


def _same_values(a, b) -> bool:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(a, b))


def _geometry(gate) -> dict:
    return {k: copy.deepcopy(v) for k, v in gate.items()
            if k not in rs._NOT_GEOMETRY and not str(k).startswith('_')}


def apply_rows(model, rows) -> list:
    """Put each row's corrected result in place on ``model`` and return the
    ``(sample, result, key)`` changed. The result's stamp moves to the
    running version (same step and inputs); its original is kept, once:
    a gate's geometry and stamp in ``gate['original']``, a column's values
    under :func:`result_stamp.original_column` in the sample's data and its
    stamp in ``sample.result_originals``. Undo is the caller's (one
    checkpoint before)."""
    done = []
    for row in rows:
        if row.get('informational') or row.get('new') is None:
            continue
        sample, kind, key = row['sample'], row['result'], row['key']
        old_st = row.get('stamp') or {}
        if kind == 'gate':
            g = model._sample_gates[sample][key]
            if rs.ORIGINAL_KEY not in g:
                orig = _geometry(g)
                orig['stamp'] = {k: v for k, v in old_st.items()
                                 if k != rs.GEOM_KEY}
                g[rs.ORIGINAL_KEY] = orig
            for k in list(_geometry(g)):
                g.pop(k, None)
            g.update(_geometry(row['new']))
            rs.stamp_gate(g, old_st.get('step') or 'gate', old_st.get('inputs'))
        else:
            s = model._samples[sample]
            originals = dict(getattr(s, rs.ORIGINALS_ATTR, None) or {})
            if key not in originals:
                kept = rs.original_column(key, old_st.get('version'))
                s.data[kept] = s.data[key].to_numpy(copy=True)
                originals[key] = {'stamp': {k: v for k, v in old_st.items()
                                            if k != rs.GEOM_KEY},
                                  'column': kept}
                setattr(s, rs.ORIGINALS_ATTR, originals)
            s.data[key] = row['new']
            rs.stamp_columns(s, [key], old_st.get('step') or 'column',
                             old_st.get('inputs'))
        done.append((sample, kind, key))
    return done


def record(model, applied=(), declined=()) -> None:
    """Add fix ids to ``model``'s ``_applied_fixes`` / ``_declined_fixes``
    (the session's ``fixes_applied`` / ``fixes_declined``; exports stamp the
    applied ones). An id applied is no longer declined."""
    a, d = settled(model)
    for f in applied:
        if f not in a:
            a.append(f)
    d = [f for f in d if f not in a]
    for f in declined:
        if f not in a and f not in d:
            d.append(f)
    model._applied_fixes = a
    model._declined_fixes = d


# Steps with a re-run the review window can start; the rest are named, with
# where to re-run them and the editor method that opens that dialog (the
# window's Open…). Slow steps start unticked.
_RERUN_WHERE = (('cluster.', 'Analyze > Cluster…', '_open_cluster_dialog'),
                ('embed.', 'Analyze > Cluster… (embedding)',
                 '_open_cluster_dialog'),
                ('cell_cycle', 'Analyze > Cell cycle…',
                 '_open_cell_cycle_dialog'),
                ('calibration', 'Tools > Calibration…',
                 '_open_calibration_dialog'),
                ('trajectory', 'Analyze > Trajectory…',
                 '_open_trajectory_window'),
                ('gate.auto.', 'Gate > Auto-gate', '_auto_gate'))
_SLOW = ('embed.', 'cluster.', 'trajectory')


def downstream(model, corrected) -> list:
    """Results built on a corrected one, for a follow-up re-run (never
    started by itself): stamped gates under a corrected gate (via
    ``parent_id``, all the way down), and stamped results whose ``inputs``
    name a corrected gate or column. Statistics, plots and layout pages are
    recomputed from the gates, so they need nothing. Each row::

        {'sample', 'result', 'key', 'name', 'step', 'stamp', 'because',
         'rerun': bool, 'where': str | None, 'opens': str | None,
         'slow': bool}

    ``opens`` names the editor method that opens the dialog re-running the
    step (None for a re-fit, which re-runs from the window).
    """
    results = results_of(model)
    by_sample: dict = {}
    for r in results:
        by_sample.setdefault(r['sample'], []).append(r)
    seen = set(corrected)
    queue = list(corrected)
    out = []
    while queue:
        sample, kind, key = queue.pop(0)
        gates = model._sample_gates.get(sample, {})
        for r in by_sample.get(sample, []):
            rk = (sample, r['kind'], r['key'])
            if rk in seen:
                continue
            under = (kind == 'gate' and r['kind'] == 'gate'
                     and (gates.get(r['key']) or {}).get('parent_id') == key)
            if not (under or key in r['stamp'].get('inputs', [])):
                continue
            seen.add(rk)
            queue.append(rk)
            step = r['stamp']['step']
            where, opens = next(((w, o) for p, w, o in _RERUN_WHERE
                                 if step.startswith(p)), (None, None))
            out.append({'sample': sample, 'result': r['kind'],
                        'key': r['key'], 'name': r['name'], 'step': step,
                        'stamp': r['stamp'], 'because': key,
                        'rerun': step == 'gate.refit', 'where': where,
                        'opens': opens, 'slow': step.startswith(_SLOW)})
    return out


# Audit actions that write a file from the session's results. Each records
# the file's ``path`` (an export of several files, also ``files``) and the
# samples in it (``samples``, or ``sample`` for one). Spectral QC and the
# compliance record are left out: they are made from the unmixing controls
# and the audit trail, not from the results a fix corrects.
EXPORT_ACTIONS = ('report.export', 'export.gate_list', 'figure.export',
                  'layout.export', 'population.export_fcs', 'anndata.export',
                  'expression.export', 'trajectory.export', 'annotate.export',
                  'flowsom_tree.export', 'project.export', 'plot.export',
                  'gatingml.export', 'wsp.export', 'statistics.export',
                  'frequencies.export', 'kinetics.export',
                  'compare_wsp.export', 'compare_all.export',
                  'diff_abundance.export')


def _export_samples(details):
    """The samples an export entry names, or None when it names none."""
    names = details.get('samples')
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    if details.get('sample'):
        return [str(details['sample'])]
    return None


def stale_exports(entries, exists=os.path.exists) -> list:
    """Files exported before the latest ``fix.apply`` in the audit
    ``entries`` that involve a sample it corrected (or do not say which
    samples): they still hold the uncorrected numbers. Nothing is re-exported
    here; the export settings are not recorded. One row per file, from its
    latest export, newest first::

        {'path', 'time', 'version', 'action', 'samples', 'exists'}
    """
    fix = next((e for e in reversed(entries) if e['action'] == 'fix.apply'),
               None)
    if fix is None:
        return []
    corrected = {str(r.get('sample')) for r in
                 fix['details'].get('results') or () if isinstance(r, dict)}
    if not corrected:
        return []
    rows: dict = {}
    for e in entries:
        if e['seq'] >= fix['seq'] or e['action'] not in EXPORT_ACTIONS:
            continue
        d = e['details']
        names = _export_samples(d)
        if names is not None and not corrected.intersection(names):
            continue
        for p in d.get('files') or ([d['path']] if d.get('path') else []):
            rows[str(p)] = {'path': str(p), 'time': e.get('time'),
                            'version': e.get('version'),
                            'action': e['action'], 'samples': names,
                            'seq': e['seq']}
    out = sorted(rows.values(), key=lambda r: -r['seq'])
    for r in out:
        del r['seq']
        r['exists'] = bool(exists(r['path']))
    return out


def rerun_row(model, follow) -> dict:
    """A plan row re-running a follow-up (``downstream``) with the current
    code; only re-fitted gates can be re-run from here."""
    t = {'sample': follow['sample'], 'result': follow['result'],
         'key': follow['key'], 'stamp': follow['stamp']}
    old = _current(model, follow['sample'], follow['result'], follow['key'])
    res = _redo_refit(model, dict(t, current=copy.deepcopy(old)))
    return dict(t, name=follow['name'], fixes=[], titles=[],
                informational=False, old=old, new=res['new'],
                why=res.get('why'),
                before=_numbers(model, t['sample'], t['result'], old),
                after=_numbers(model, t['sample'], t['result'], res['new']),
                changed=rs.geometry_sig(res['new']) != rs.geometry_sig(old))


# ── Sessions on disk (openflo-run check-sessions) ───────────────────────────

def check_session_file(path, registry=None, running=None) -> list:
    """The targets :func:`scan` finds in the session file at ``path``. Reads
    only; the file is never written. Raises OSError / ValueError when it
    cannot be read."""
    from .session_format import migrate_session
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError('not an OpenFlo session')
    data, _notes = migrate_session(data)
    return scan(data, registry, running=running)


def find_sessions(folder) -> list:
    """Every ``.flowsession`` under ``folder`` (a session's backup copy
    excluded), sorted by path."""
    out = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if (fn.endswith('.flowsession')
                    and not fn.endswith('.backup.flowsession')):
                out.append(os.path.join(root, fn))
    return sorted(out)


def describe_target(t) -> str:
    """One line for a target: where it is, the fix and who made it."""
    what = (f"gate '{t['name']}' ({t['key']})" if t['result'] == 'gate'
            and t['name'] else f"gate {t['key']}" if t['result'] == 'gate'
            else f"column {t['key']}")
    made = t['version'] or 'an unknown version'
    tag = ' [default change, flagged only]' if t['informational'] else ''
    return (f"{t['sample']}: {what} made by OpenFlo {made} -- "
            f"{t['fix']}: {t['title']}{tag}")
