"""Run history for a run template: every day's run of a repeating
experiment, kept together so they can be compared.

A template's runs go in one history folder (``<template name>_history``
beside the template, or under the folder chosen in Preferences > Pipeline
workspace > Run outputs). Each run is its own ``run_<time>`` folder with
the run's usual outputs and a ``run_info.json`` saying what it ran: the
template version (a hash of the template), the experiment date (the FCS
files' $DATE, else the day it ran), each file, its slot and its detector
voltages. ``history.csv`` joins every finished run's population tables
(``<label>_population_stats.csv``) into one long table, one row per run,
sample and population. It is rebuilt from the run folders after every run,
so the run folders are what counts: delete a run folder and the next
rebuild leaves it out.

Running a day again does not add a second set of points for that date:
for each experiment date and group, the newest finished run with rows of
that group is the active one, and earlier rows of that group (or of a
sample it moved there from another group) stay on disk marked superseded.
So a re-run of one group of a day replaces only that group, and a run that
wrote no population table (every unit failed before its table, or none can
be read) is marked failed and replaces nothing (see :func:`active_rows`).

These tables are read back by OpenFlo, so they are always written in the
standard CSV style (comma separated, decimal point), whatever CSV style is
chosen in Preferences, and each is written whole or not at all.

Core module: no Tk.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

RUN_INFO = 'run_info.json'
HISTORY_CSV = 'history.csv'
HISTORY_SUFFIX = '_history'
STATUS_STARTED = 'started'
STATUS_DONE = 'done'
STATUS_CANCELLED = 'cancelled'
STATUS_FAILED = 'failed'        # finished, but wrote no population table
POP_TABLE_GLOB = '*_population_stats.csv'
# The table's columns that name rows, read as text ('007' stays '007').
KEY_COLUMNS = ('Group', 'Trial', 'Sample', 'Population')
# The CSV style the tables are kept in (see the module docstring).
TABLE_STYLE = 'standard'

# The separators a table may have been written with (read_history_table).
_SEPARATORS = (',', ';', '\t')

# A number written with a decimal comma ('12,5', '-0,25', '1,5E3').
_DECIMAL_COMMA = re.compile(r'[-+]?\d+,\d+(?:[eE][-+]?\d+)?')

# history.csv's own columns, before the population table's.
HISTORY_HEAD = ('Run', 'Experiment date', 'Active', 'Template version')


def template_version(template: Mapping[str, Any]) -> str:
    """A short hash of what `template` does (slots, patterns, populations,
    compensation, FMO sets, settings). The OpenFlo version that saved it is
    left out, so re-saving an unchanged template keeps its version."""
    from .run_template import template_to_json
    d = {k: v for k, v in template_to_json(template).items()
         if k != 'app_version'}
    blob = json.dumps(d, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:12]


def history_dir(template_path: str, chosen: str | None = None) -> str:
    """The history folder of the template at `template_path`: beside it
    (``<name>_history``), or in the folder `chosen` in Preferences when
    there is one. There, templates of one file name from two folders would
    meet, so the name carries a short hash of the template's full path
    (``<name>_<hash>_history``): each template keeps its own runs."""
    full = os.path.abspath(template_path)
    stem = os.path.splitext(os.path.basename(full))[0]
    if not (chosen and chosen.strip()):
        return os.path.join(os.path.dirname(full), stem + HISTORY_SUFFIX)
    tag = hashlib.sha256(os.path.normcase(full).encode('utf-8')).hexdigest()[:8]
    return os.path.join(os.path.abspath(os.path.expanduser(chosen.strip())),
                        f'{stem}_{tag}{HISTORY_SUFFIX}')


# ── What a run ran ─────────────────────────────────────────────────────────

def read_fcs_keywords(path: str) -> dict[str, str]:
    """FCS file `path`'s TEXT keywords, upper-cased names, reading only its
    TEXT segment. Raises as the loader does."""
    import functools

    from . import fcs_read
    from .fcs_quirks import read_fcs_tolerant
    fcs = read_fcs_tolerant(path, functools.partial(fcs_read.FlowData,
                                                    only_text=True), [])
    # The reader gives keyword names without their '$', in lower case.
    return {str(k).upper(): str(v) for k, v in fcs.text.items()}


_DATE_FORMATS = ('%d-%b-%Y', '%Y-%m-%d', '%d-%b-%y', '%m/%d/%Y', '%d/%m/%Y',
                 '%Y/%m/%d', '%d.%m.%Y')


def parse_fcs_date(text: str | None) -> str | None:
    """An FCS $DATE as ISO ``YYYY-MM-DD``, or None when it is empty or in
    no form this reads. FCS 3.x writes ``dd-mmm-yyyy`` ('11-NOV-2026');
    the others are forms instruments are known to write. A day-first and a
    month-first reading of the same slashed date both parse: the
    month-first one is taken, as US instruments write it."""
    t = (text or '').strip()
    if not t:
        return None
    for fmt in _DATE_FORMATS:
        try:
            # %b reads a month name in any case ('NOV', 'Nov').
            return datetime.datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def voltages(keywords: Mapping[str, str]) -> dict[str, float]:
    """Each detector's voltage ($PnV) by its name ($PnN), from `keywords`
    as :func:`read_fcs_keywords` gives them. Detectors with no or no numeric
    voltage are left out."""
    out: dict[str, float] = {}
    try:
        n = int(keywords.get('PAR', '0'))
    except ValueError:
        return out
    for i in range(1, n + 1):
        name, v = keywords.get(f'P{i}N'), keywords.get(f'P{i}V')
        if not name or v is None:
            continue
        try:
            out[name.strip()] = float(v)
        except ValueError:
            continue
    return out


def run_files(model: Any) -> list[dict[str, Any]]:
    """The files of a template-built workspace, once each, in row order:
    ``{'path', 'sample', 'slot', 'ran'}``. Unticked rows are included (the
    record says what the day had); ``ran`` says whether any of the file's
    rows is ticked, so runs, and only those date the run."""
    ticked = {id(it) for it in model.displayed_items()}
    seen: dict[str, dict[str, Any]] = {}
    for _mid, it, _gid in model.all_items():
        p = it.get('path')
        if not p:
            continue
        if p not in seen:
            seen[p] = {'path': str(p), 'sample': str(it.get('sample') or ''),
                       'slot': str(it.get('template_slot') or ''),
                       'ran': False}
        seen[p]['ran'] = seen[p]['ran'] or id(it) in ticked
    return list(seen.values())


def experiment_date(dates: Sequence[str]) -> tuple[str | None, str | None]:
    """``(date, note)``: the run's experiment date from the $DATEs of the
    files it ran (ISO): the latest of them, and a note when they disagree,
    or ``(None, None)`` when there are none. Files carried over from an
    earlier day, however many, must not move the run back onto that day
    (where it would supersede that day's own run)."""
    if not dates:
        return None, None
    counts: dict[str, int] = {}
    for d in dates:
        counts[d] = counts.get(d, 0) + 1
    date = max(counts)
    if len(counts) == 1:
        return date, None
    shown = ', '.join(f'{d}: {counts[d]} file(s)' for d in sorted(counts))
    return date, (f"the files' $DATE differ ({shown}): the run is dated "
                  f"{date}, the latest of them")


def build_run_info(run_id: str, template: Mapping[str, Any],
                   files: Sequence[Mapping[str, Any]],
                   read_keywords: Callable[[str], Mapping[str, str]] = read_fcs_keywords,
                   today: Callable[[], datetime.date] = datetime.date.today,
                   stat_forms: Mapping[str, Any] | None = None
                   ) -> dict[str, Any]:
    """The ``run_info.json`` of a run `run_id` of `template` (its
    ``{'path', 'name', 'version'}``) on `files` (:func:`run_files`), with
    the GeoMean forms its population tables used (`stat_forms`). The
    experiment date comes from the $DATE of the files that ran
    (:func:`experiment_date`), else `today`; each file keeps its voltages.
    A file whose keywords cannot be read is noted, not fatal."""
    recs, dates, notes = [], [], []
    for f in files:
        rec: dict[str, Any] = dict(f)
        try:
            kw = read_keywords(f['path'])
        except Exception as exc:  # noqa: BLE001 - any unreadable header is noted in the record; the run itself reloads the file and says why it fails
            notes.append(f"{os.path.basename(f['path'])}: keywords not read "
                         f"({type(exc).__name__}: {exc})")
            kw = {}
        rec['date'] = parse_fcs_date(kw.get('DATE'))
        rec['voltages'] = voltages(kw)
        if rec['date'] and rec.get('ran', True):
            dates.append(rec['date'])
        recs.append(rec)
    date, mixed = experiment_date(dates)
    if mixed:
        notes.append(mixed)
    return {'run_id': run_id, 'status': STATUS_STARTED,
            'template': dict(template),
            'experiment_date': date or today().isoformat(),
            'date_from': 'files' if date else 'run day',
            'stat_forms': dict(stat_forms or {}),
            'files': recs, 'notes': notes}


def write_run_info(out_dir: str, info: Mapping[str, Any]) -> str:
    """Write `info` as ``run_info.json`` in `out_dir`, whole or not at
    all; returns its path."""
    from .atomic_write import atomic_write_json
    path = os.path.join(out_dir, RUN_INFO)
    atomic_write_json(path, info)
    return path


def _unread_note(run_id: str, path: str, exc: BaseException) -> str:
    """How a population table that cannot be read is said."""
    return (f"{run_id}/{os.path.basename(path)} not read "
            f"({type(exc).__name__}: {exc})")


def finished_status(out_dir: str, done: bool,
                    notes: list[str] | None = None) -> str:
    """The status a run in `out_dir` ends with: ``cancelled`` when it was
    stopped; ``done`` when it ran to the end and wrote at least one
    population table with a row; else ``failed`` (no unit got as far as
    its table, or none of its tables can be read), which the history leaves
    out. Each table that cannot be read is said in `notes` (only a failed
    run's: a finished run's are said when the history reads them)."""
    if not done:
        return STATUS_CANCELLED
    unread = []
    for p in sorted(glob.glob(os.path.join(glob.escape(out_dir),
                                           POP_TABLE_GLOB))):
        try:
            if len(read_history_table(p)):
                return STATUS_DONE
        except (OSError, ValueError) as exc:
            unread.append(_unread_note(os.path.basename(out_dir), p, exc))
    if notes is not None:
        notes.extend(unread)
    return STATUS_FAILED


def failed_line(notes: Sequence[str]) -> str:
    """The final line's words for a failed run (:func:`finished_status`),
    `notes` being what it said: no table written, or none readable."""
    what = ('no population table of this run could be read' if notes
            else 'no population table with a row was written')
    return (f"{what}: this run is not in the history and replaces no "
            "earlier run.")


def set_run_status(out_dir: str, status: str) -> None:
    """Mark the run in `out_dir` ``done``, ``failed`` or ``cancelled``."""
    path = os.path.join(out_dir, RUN_INFO)
    with open(path, encoding='utf-8') as fh:
        info = json.load(fh)
    info['status'] = status
    write_run_info(out_dir, info)


# ── The history ────────────────────────────────────────────────────────────

def load_runs(hdir: str) -> list[dict[str, Any]]:
    """Every run in history folder `hdir` that has a readable
    ``run_info.json``, oldest first, each with its ``dir``. A run folder
    whose record is missing or damaged is passed over."""
    runs = []
    for d in sorted(glob.glob(os.path.join(glob.escape(hdir), 'run_*'))):
        try:
            with open(os.path.join(d, RUN_INFO), encoding='utf-8') as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(info, dict) and info.get('run_id'):
            runs.append(dict(info, dir=d))
    runs.sort(key=lambda r: str(r['run_id']))
    return runs


def write_history_table(path: str, df) -> None:
    """Write a population table or ``history.csv`` (DataFrame `df`) to
    `path` in :data:`TABLE_STYLE`, whole or not at all."""
    from .atomic_write import atomic_path
    from .table_export import write_frame_csv
    with atomic_path(path) as tmp:
        write_frame_csv(tmp, df, style=TABLE_STYLE)


def read_history_table(path: str):
    """A table :func:`write_history_table` wrote, as a DataFrame, its
    :data:`KEY_COLUMNS` as text. One written in a semicolon or tab CSV
    style (before these tables were kept in the standard style; Windows'
    own style may use either) is read in that style, its decimal commas as
    decimals. Raises OSError, or ValueError
    when the file is empty or not a table."""
    import pandas as pd
    with open(path, encoding='utf-8-sig') as fh:
        head = fh.readline()
        body = fh.read(1 << 14)
    # The separator the header holds most of; a comma when it holds none
    # (one column). Ties go to the comma, then the semicolon.
    sep = max(_SEPARATORS, key=head.count)
    decimal = ',' if sep != ',' and _decimal_commas(head, body, sep) else '.'
    return pd.read_csv(path, sep=sep, decimal=decimal, encoding='utf-8-sig',
                       converters=dict.fromkeys(KEY_COLUMNS, str))


def _decimal_commas(head: str, body: str, sep: str) -> bool:
    """Whether a table in separator `sep` (header line `head`, then the
    start of its `body`) writes its numbers with decimal commas: a value
    cell (not one of :data:`KEY_COLUMNS`, whose names may hold '1,2') is a
    number with a decimal comma. A body cut short may end in part of a
    row; a part of a value cell holds a decimal comma only when the whole
    cell does, so that row is read too."""
    import csv
    names = next(csv.reader([head], delimiter=sep), [])
    keys = {i for i, n in enumerate(names) if n in KEY_COLUMNS}
    lines = body.splitlines()
    for row in csv.reader(lines, delimiter=sep):
        if any(i not in keys and _DECIMAL_COMMA.fullmatch(v.strip())
               for i, v in enumerate(row)):
            return True
    return False


def run_table(run: Mapping[str, Any], notes: list[str] | None = None):
    """Run `run`'s population tables joined, or None when it has none. A
    table that cannot be read is left out and said in `notes`."""
    import pandas as pd
    frames = []
    for p in sorted(glob.glob(os.path.join(glob.escape(run['dir']),
                                           POP_TABLE_GLOB))):
        try:
            frames.append(read_history_table(p))
        except (OSError, ValueError) as exc:
            if notes is not None:
                notes.append(_unread_note(str(run['run_id']), p, exc))
    return pd.concat(frames, ignore_index=True) if frames else None


def _key(df, col: str) -> list[str]:
    """Column `col` of `df` as stripped text, '' throughout when absent."""
    if col not in df.columns:
        return [''] * len(df)
    return [str(v).strip() for v in df[col].fillna('')]


def active_rows(runs: Sequence[Mapping[str, Any]],
                tables: Mapping[str, Any]) -> dict[str, list[bool]]:
    """Which rows of each finished run's table (`tables`, by run id, as
    :func:`read_tables` gives them: finished runs only) are active. A row
    is superseded when a newer run of the same experiment date has

    - rows of its group (a re-run of that group; an ungrouped row, Group
      '', is no group: only its own sample replaces it), or
    - its group and sample (a re-run of that ungrouped sample), or
    - its sample in a group this run did not have it in (the sample moved
      to another group). A sample name that is in the newer run's group
      just as it was in this run is no move, so it does not replace a
      sample of the same name in another group of this run. When this run
      had the name in two groups and the newer run has it in a third,
      which of the two moved cannot be told from the name: both are
      replaced (the rows carry no file identity).

    A run with no table is in no list."""
    later: dict[str, tuple[set[str], set[tuple[str, str]]]] = {}
    out: dict[str, list[bool]] = {}
    for r in sorted(runs, key=lambda r: str(r['run_id']), reverse=True):
        t = tables.get(str(r['run_id']))
        if t is None:
            continue
        groups, pairs = later.setdefault(str(r.get('experiment_date')),
                                         (set(), set()))
        mine = list(zip(_key(t, 'Group'), _key(t, 'Sample'), strict=True))
        own = set(mine)
        moved = {s for g, s in pairs if s and (g, s) not in own}
        out[str(r['run_id'])] = [not ((g and g in groups) or (g, s) in pairs
                                      or s in moved) for g, s in mine]
        groups.update(g for g, _s in mine)
        pairs.update(own)
    return out


def read_tables(runs: Iterable[Mapping[str, Any]],
                notes: list[str] | None = None) -> dict[str, Any]:
    """:func:`run_table` of each finished run in `runs` that has one, by
    run id."""
    return {str(r['run_id']): t for r in runs
            if r.get('status') == STATUS_DONE
            and (t := run_table(r, notes)) is not None}


def mark_active(runs: Iterable[dict[str, Any]],
                tables: Mapping[str, Any] | None = None
                ) -> list[dict[str, Any]]:
    """`runs` with ``active`` set: True on a finished run any of whose
    rows is still active (:func:`active_rows`), False on every other (a run
    every row of which a newer run of its date replaced; an unfinished,
    cancelled or failed run, or one with no table, never counts). `tables`
    are the runs' :func:`read_tables`, read here when not given."""
    runs = list(runs)
    if tables is None:
        tables = read_tables(runs)
    _set_active(runs, active_rows(runs, tables))
    return runs


def _set_active(runs: Iterable[dict[str, Any]],
                rows: Mapping[str, list[bool]]) -> None:
    """Set each run's ``active``: any of its rows (`rows`) active."""
    for r in runs:
        r['active'] = any(rows.get(str(r['run_id'])) or ())


def rebuild_history(hdir: str, notes: list[str] | None = None
                    ) -> tuple[str | None, int]:
    """Rebuild ``history.csv`` in `hdir` from its finished runs' tables,
    each row marked active or superseded (:func:`active_rows`). Returns
    ``(path, rows)``; the path is None (and an old file removed) when no
    finished run has a table. A table that cannot be read is left out and
    said in `notes`."""
    path, n, _runs = update_history(hdir, notes)
    return path, n


def update_history(hdir: str, notes: list[str] | None = None
                   ) -> tuple[str | None, int, list[dict[str, Any]]]:
    """:func:`rebuild_history`, plus the runs of `hdir` (:func:`load_runs`)
    marked active (:func:`mark_active`) from the same tables, read once."""
    import pandas as pd
    runs = load_runs(hdir)
    tables = read_tables(runs, notes)
    rows = active_rows(runs, tables)
    _set_active(runs, rows)
    frames = []
    for r in runs:
        t = tables.get(str(r['run_id']))
        if t is None:
            continue
        t = t.copy()
        for i, (col, val) in enumerate(zip(HISTORY_HEAD, (
                r['run_id'], r.get('experiment_date'), rows[str(r['run_id'])],
                (r.get('template') or {}).get('version', '')), strict=True)):
            t.insert(i, col, val)
        frames.append(t)
    path = os.path.join(hdir, HISTORY_CSV)
    if not frames:
        if os.path.exists(path):
            os.remove(path)
        return None, 0, runs
    df = pd.concat(frames, ignore_index=True)
    write_history_table(path, df)
    return path, int(len(df)), runs


def changes_since_last(runs: Sequence[Mapping[str, Any]],
                       run_id: str) -> list[str]:
    """What changed between run `run_id` and the active run before it (of
    an earlier experiment date): the template version, and any detector
    whose voltage differs, as warnings to show with the run. Comparing
    across days by MFI means little once a voltage moved."""
    runs = sorted(runs, key=lambda r: str(r['run_id']))
    me = next((r for r in runs if r['run_id'] == run_id), None)
    if me is None:
        return []
    # The day before, not a run of `run_id`'s own date: a re-run of part
    # of a day leaves that day's earlier run active for the other groups.
    day = str(me.get('experiment_date'))
    before = [r for r in runs if r.get('active') and str(r['run_id']) < run_id
              and str(r.get('experiment_date')) != day]
    if not before:
        return []
    prev = before[-1]
    out = []
    v_old = (prev.get('template') or {}).get('version')
    v_new = (me.get('template') or {}).get('version')
    if v_old and v_new and v_old != v_new:
        out.append(f"the template changed since the {prev['experiment_date']} "
                   "run: compare across that change with care")

    def volts(r):
        merged: dict[str, set] = {}
        for f in r.get('files') or []:
            for ch, v in (f.get('voltages') or {}).items():
                merged.setdefault(ch, set()).add(v)
        return merged
    # A record from before the forms were kept has none: no comparison.
    if 'stat_forms' in prev and 'stat_forms' in me \
            and prev['stat_forms'] != me['stat_forms']:
        out.append(f"the statistic forms (Preferences: GeoMean) changed since "
                   f"the {prev['experiment_date']} run: GeoMean is not "
                   "comparable across that change")
    old, new = volts(prev), volts(me)
    moved = sorted(ch for ch in set(old) & set(new) if old[ch] != new[ch])
    if moved:
        shown = ', '.join(moved[:4]) + (f" and {len(moved) - 4} more"
                                        if len(moved) > 4 else '')
        out.append(f"detector voltages changed since the "
                   f"{prev['experiment_date']} run ({shown}): MFI is not "
                   "comparable across that change")
    return out
