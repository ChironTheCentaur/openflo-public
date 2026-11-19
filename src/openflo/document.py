"""A headless gating document: open, gate, count, save -- Tk-free.

``Document`` is what a front end without the Tk gate editor (the phone
app, driving it from a Pyodide worker, or a script) holds: a set of
samples, each with its gate tree, opened from FCS files, an OpenFlo
session or a FlowJo workspace. It opens them with the shared openers the
editor uses (``sample_prep``, ``session_open``, ``wsp_open``), changes
gate trees with ``gate_store``, counts with ``gating.population_stats``
and saves a session the editor opens with the same gates and numbers.

Its answers are plain data a JavaScript bridge passes as it is: dicts,
lists, strings, numbers (None for a value that is not known) and numpy
arrays. The data gates and statistics read stays float64; only the binned
counts of a plot (``density``, ``histogram``), copies drawn on screen, are
float32. ``overlays`` gives a plot's gate outlines on its display axes.

It never writes to a file it read. It takes no lock, saves nothing in
place and keeps no autosave: ``save_session`` writes a new session to the
path it is given (and its ``<stem>_data`` sidecar folder when a sample
needs one) and refuses a path naming any file it read (a session, a
workspace, an FCS file, a processed-data sidecar or its transform record),
by any spelling, link or hard link.

Unlike the editor it keeps no window-wide transforms: each sample stays on
the transforms its data carries (a session's saved ones), with its gates as
drawn on them.

Core module: no Tk.
"""
from __future__ import annotations

import copy
import json
import math
import os
import secrets
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from . import gate_store
from .gate_record import gate_record_problem
from .session_format import SESSION_FORMAT

# The per-channel statistics a Document reports by default, with the keys
# they are reported under.
DEFAULT_CHANNEL_STATS = ('Median', 'Mean', 'GeoMean', 'CV')

# The overlay colour a session gives a sample never displayed
# (session_open ignores it on the way back in).
_PLACEHOLDER_COLOR = '#1f77b4'

# Bins of a density grid along one axis, at most.
MAX_BINS = 1024

# A sample entry's keys that a save works out again from the sample: those
# read from a session are dropped first, so a stale one is never kept.
_REWRITTEN = ('path', 'basename', 'rel_path', 'processed_data',
              'processed_csv', 'transforms', 'stat_axes', 'comp',
              'qc_applied', 'qc_removed', 'app_version', 'saved_stats',
              'lmd_data_set', 'instrument_compensated', 'set_number')

# The longest marker label set_channel_label takes, and the control
# characters it reads as white space (tab, line break, vertical tab, form
# feed, carriage return).
MAX_LABEL = 64
_LABEL_SPACES = frozenset('\t\n\v\f\r')

# A session's top-level view settings, as a Document saves them when it
# opened no session (an editor with nothing changed). With the keys
# _session_data writes itself these are session_format.SESSION_KEYS
# (test_document pins that).
_VIEW_DEFAULTS: dict[str, Any] = {
    'channel_scale': {}, 'channel_range': {}, 'channel_labels': {},
    'plot_mode': None, 'x_channel': None, 'y_channel': None,
    'color_channel': None, 'downsample_display': True,
    'downsample_propagate': False, 'max_points': None,
    'show_removed': False, 'contour_scatter': True,
    'contour_outliers': True, 'hist_y_mode': None, 'cluster_labels': {},
    'audit': [], 'view_tabs': None, 'layout': None, 'fixes_applied': [],
    'fixes_declined': [],
}



def _read_session_json(path: str) -> dict[str, Any]:
    """The parsed OpenFlo session at `path`; ``DocumentError`` for a file
    that is not one."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except ValueError as exc:
        raise DocumentError(
            f"{os.path.basename(path)} is not a readable session "
            f"({exc})") from exc
    if not isinstance(data, dict) or data.get('format') != SESSION_FORMAT:
        raise DocumentError(
            f"{os.path.basename(path)} isn't an OpenFlo session "
            "(missing format marker).")
    return data



def _fcs_within(fcs: str, within: str | None) -> bool:
    """Whether `fcs` is an FCS file that exists, and, given `within`, lies
    inside that folder."""
    if not fcs or not os.path.isfile(fcs):
        return False
    if within is None:
        return True
    root = os.path.realpath(within)
    return os.path.commonpath([root, os.path.realpath(fcs)]) == root

class DocumentError(ValueError):
    """A request the document cannot carry out; the message says why."""


@dataclass
class _Sample:
    """One loaded sample: its data, gate set (``{id: gate}`` and the ids in
    order) and where it came from. ``entry`` is the session entry it was
    opened from (None for a file opened on its own), ``session_dir`` that
    session's folder, ``sidecar`` the processed-data file it loaded from
    ('' for a raw FCS) and ``trial`` the group a workspace put it in."""

    name: str
    data: Any
    gates: dict[str, gate_store.Gate] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    entry: dict[str, Any] | None = None
    session_dir: str = ''
    sidecar: str = ''
    trial: str | None = None
    # ``(path, _stamp(path))`` of the sidecar and its transform record as
    # they were when read; a save refuses to copy them once changed.
    stamps: tuple[tuple[str, Any], ...] = ()


# The case folding of `_key`: os.path.normcase, which folds case on
# Windows and is the identity elsewhere (a module name so a test can stand
# in Windows's folding).
_normcase = os.path.normcase


def _key(path: str) -> str:
    """`path` as two spellings of one file compare (normcased, absolute,
    links followed). For a file that exists, ``_file_id`` also catches a
    hard link and, on a file system that folds case where normcase does
    not (macOS), another spelling, which names the same inode there (not
    tested: the tests run on file systems that tell case apart)."""
    return _normcase(os.path.realpath(os.path.abspath(path)))


def _file_id(path: str) -> tuple[int, int] | None:
    """``(st_dev, st_ino)`` of the file `path` names (links followed), as
    ``os.path.samefile`` compares them; None when there is no such file or
    the file system gives no inode number."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _by_name_files(samples: list[dict[str, Any]], data_dir: str,
                   dests: list[str]) -> list[str]:
    """The files the opener would find by name in `data_dir` (and their
    transform records) for each session entry in `samples` written with
    no sidecar pointer (``session_open.sidecar_name_candidates``). One
    that is a sidecar copy in `dests` (compared with case folded on every
    OS: a session written where case is told apart may be reopened where
    it is not) raises ``DocumentError``: that sample would reopen holding
    another sample's events."""
    from .session_open import sidecar_name_candidates
    from .transform_sidecar import transforms_sidecar_path
    folded = {_key(d).casefold() for d in dests}
    out: list[str] = []
    for entry in samples:
        name = entry.get('name')
        if not name or entry.get('_sidecar'):
            continue
        for cand in sidecar_name_candidates(str(name), data_dir):
            if _key(cand).casefold() in folded:
                raise DocumentError(
                    f"sample {name} would reopen from "
                    f"{os.path.basename(cand)}, another sample's saved "
                    "data; remove one of the two samples from the "
                    "document to save it")
            out += [cand, transforms_sidecar_path(cand)]
    return out


def _refuse_others_files(data_dir: str, touched: list[str]) -> None:
    """Raise ``DocumentError`` naming the files of `touched` (every file
    the save would remove or replace in `data_dir`) that exist,
    for a save to a path with no session yet: they are another session's
    sharing the folder (``run.json`` beside ``run.flowsession``) or an
    orphan's, never an earlier save's, and are not removed."""
    found = sorted({os.path.basename(c) for c in touched
                    if os.path.lexists(c)})
    if found:
        raise DocumentError(
            f"{os.path.basename(data_dir)} already holds "
            f"{', '.join(found)}, which this save would remove "
            "(another session's data?); save under another name")


def _stamp(path: str) -> tuple[int, int, int, int] | None:
    """``(st_dev, st_ino, st_size, st_mtime_ns)`` of the file `path` names,
    or None when there is none: two stamps differ once the file is
    rewritten or replaced. The inode is kept although some file systems
    renumber files that did not change (``save_session`` lists them): a
    false refusal asks for a reopen, while a replacement with the same
    size and time that went unseen would be copied as the data read."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _new_temp(dst: str, staged: list[tuple[str, str]]) -> Any:
    """A new, empty binary file beside `dst`, open for writing, its path
    added to `staged` (as ``(path, dst)``) the moment it exists, so the
    caller's cleanup removes it whatever stops the save (an error or an
    interrupt). Made with O_EXCL under a random name, so it is never a
    file already there, nor a link planted there to send the write
    elsewhere. The name is 7 characters longer than ``<dst>.tmp``."""
    folder, base = os.path.split(dst)
    flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL
             | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    for _ in range(100):
        tmp = os.path.join(folder, f'{base}.{secrets.token_hex(3)}.tmp')
        try:
            fd = os.open(tmp, flags, 0o666)
        except FileExistsError:
            continue
        try:
            staged.append((tmp, dst))
            return os.fdopen(fd, 'wb')
        except BaseException:
            os.close(fd)
            raise
    raise FileExistsError(f"no free temporary name beside {base}")


def _write_json(f: Any, data: Any) -> None:
    """Write session dict `data` to binary file `f`."""
    f.write(json.dumps(data, indent=2, ensure_ascii=False,
                       allow_nan=False).encode('utf-8'))


def _copy_file(f: Any, src: str) -> None:
    """Copy file `src` into binary file `f`."""
    import shutil
    with open(src, 'rb') as inp:
        shutil.copyfileobj(inp, f)


def _heading_note(heading: str, col: str) -> str:
    """What ``gating.stat_column_heading`` put after column `col`'s title
    (``'(linear)'`` from ``'Median FL1-A (linear)'``); '' when nothing."""
    return heading[len(col):].strip() if heading.startswith(col) else ''


def _stat_value(v: Any) -> Any:
    """A statistic as plain data: a float, an int for a whole count, or
    None for one that is not known (NaN)."""
    if v is None:
        return None
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return int(v)
    f = float(v)
    return None if math.isnan(f) else f


class Document:
    """Samples and their gate trees, without a window (see the module
    docstring).

    `labels` is a ``'DetA=LabelA;DetB=LabelB'`` string applied to every
    FCS file opened, as the editor's labels setting; `qc` whether
    acquisition QC runs on a file whose session or workspace does not say
    (the editor's Preferences default is off); `stat_forms` the GeoMean
    and rCV forms (``gating.DEFAULT_STAT_FORMS`` when None, as the
    editor's Preferences default)."""

    def __init__(self, *, labels: str = '', qc: bool = False,
                 stat_forms: Mapping[str, str] | None = None) -> None:
        self.labels = labels
        self.qc = bool(qc)
        self.stat_forms = dict(stat_forms) if stat_forms else None
        self._samples: dict[str, _Sample] = {}
        self._seqs: dict[str, int] = {}
        # Samples of an opened session whose files were not found, kept so
        # a save writes them back: name -> {'entry', 'gates'}.
        self._offline: dict[str, dict[str, Any]] = {}
        # Every file read (sessions, workspaces, FCS, sidecars and their
        # transform records): never written. _key spellings, and the
        # _file_id of each that existed when read.
        self._sources: set[str] = set()
        self._source_ids: set[tuple[int, int]] = set()
        # The top-level view settings of the first session opened.
        self._view: dict[str, Any] = {}
        # Marker labels named in this document (set_channel_label): over
        # every session's and file's, and saved with the session's own.
        self._labels: dict[str, str] = {}
        # This document's own provenance records (channel_labels.set), put
        # after the opened session's trail when it is saved.
        self._trail: list[dict[str, Any]] = []
        self._active: str | None = None

    # ── opening ───────────────────────────────────────────────────────────

    def _read_from(self, path: str) -> None:
        """Record `path` as a file this document read (never written)."""
        self._sources.add(_key(path))
        fid = _file_id(path)
        if fid is not None:
            self._source_ids.add(fid)

    def _is_source(self, path: str) -> bool:
        """Whether `path` names a file this document read: the same
        spelling once normalised and links followed, or (for a file that
        exists) the same file by device and inode."""
        if _key(path) in self._sources:
            return True
        fid = _file_id(path)
        return fid is not None and fid in self._source_ids

    def open_fcs(self, path: str) -> str:
        """Open the FCS file at `path` as a new sample, prepared as the
        editor's loader prepares one (``sample_prep.prepare_fcs_sample``:
        QC when on, the file's own $SPILL, the default transform, labels).
        Returns the name it is listed under (``sample_prep.sample_name_for``).

        `path` is kept absolute, as the editor keeps it. Opening one file
        twice gives a second sample (``s1 [folder]``), unlike the editor,
        which keeps one sample per file."""
        from .sample_prep import prepare_fcs_sample, sample_name_for
        path = os.path.abspath(str(path))
        s = prepare_fcs_sample(path, None, labels=self.labels,
                               qc_default=self.qc)
        self._read_from(path)
        name = sample_name_for(path, self._taken)
        self._add(_Sample(name=name, data=s))
        return name

    def data_files(self, path: str) -> dict[str, Any]:
        """The data files the OpenFlo session or FlowJo workspace at `path`
        names, for a front end that must be handed them before opening it
        (a phone copies each picked file into the folder of the session or
        workspace, where ``open_session`` and ``import_wsp`` look for it).

        Returns ``{'kind': 'session' | 'workspace', 'files': [{'name',
        'found'}]}``, one row per file name in the order saved: `name` is
        the file name (from either separator, whatever OS saved it) and
        `found` whether opening `path` now would find that file, by the
        same lookup the open uses. Reads `path` only; nothing is opened.
        Raises ``DocumentError`` for a file that is neither."""
        from .compare import FcsLocator, uri_path
        from .paths import portable_basename, saved_file_name
        from .session_open import resolve_sample_path
        path = str(path)
        rows: dict[str, bool] = {}
        if path.lower().endswith('.wsp'):
            from .wsp_open import WspUnreadable, plan_wsp
            locator = FcsLocator(path)
            uris: list[str] = []

            def note(uri: str | None) -> None:
                if uri:
                    uris.append(uri)
            try:
                plan_wsp(path, note)
            except WspUnreadable as exc:
                raise DocumentError(
                    f"{os.path.basename(path)} is not a readable FlowJo "
                    f"workspace ({exc})") from exc
            for uri in uris:
                name = portable_basename(uri_path(uri))
                if name:
                    rows[name] = rows.get(name, False) or (
                        locator.find(uri) is not None)
            kind = 'workspace'
        else:
            data = _read_session_json(path)
            sess_dir = os.path.dirname(os.path.abspath(path))
            for e in data.get('samples') or []:
                if not isinstance(e, Mapping):
                    continue
                name = saved_file_name(e)
                if name:
                    rows[name] = rows.get(name, False) or os.path.isfile(
                        resolve_sample_path(e, sess_dir))
            kind = 'session'
        return {'kind': kind,
                'files': [{'name': n, 'found': f} for n, f in rows.items()]}

    def export_wsp(self, path: str, within: str | None = None,
                   names_only: bool = False) -> dict[str, Any]:
        """Write the loaded samples, their gates and compensation as a
        FlowJo workspace at `path`, as the editor's File > Export FlowJo
        .wsp does (``wsp_export``: the same gate list, matrices and
        unknown-scale matching; each population's count). Each sample
        points at the raw FCS it was read from. A sample whose FCS file is
        not there (one opened from its saved data alone), or, with
        `within`, is not inside that folder, is left out: FlowJo could not
        open it, and its path may belong to another computer.

        With `names_only` (the phone, whose files sit in its own storage)
        each sample names its FCS file by name alone, relative to the
        workspace: kept in the folder with those files, it finds them there.
        Samples from different files of one name are all written, with a
        warning: FlowJo would find one file for all of them.

        Returns ``{'samples': n written, 'gates': n written, 'left_out': n
        gates FlowJo cannot carry, 'not_written': [samples left out],
        'warnings': [where FlowJo will count differently]}``."""
        import math

        from .sample_prep import stored_transforms
        from .wsp_export import (
            match_unknown_scales,
            own_compensation,
            sample_fcs,
            wsp_gate_list,
        )
        from .wsp_writer import WspWriter
        w = WspWriter(cytometer='OpenFlo', file_names_only=names_only)
        gates_out = left_out = 0
        by_file: dict[str, dict[str, list[str]]] = {}
        not_written: list[str] = []
        for name, rec in self._samples.items():
            sample, data = rec.data, rec.data.data
            if not _fcs_within(sample_fcs(sample), within):
                not_written.append(name)
                continue
            fcs = sample_fcs(sample)
            by_file.setdefault(os.path.basename(fcs), {}).setdefault(
                os.path.realpath(fcs), []).append(name)
            gate_list = wsp_gate_list(rec.gates, rec.order)
            left_out += len(rec.order) - len(gate_list)
            own = own_compensation(sample)
            stored = stored_transforms(sample)
            counts: dict[str, float] = {}
            if gate_list:
                for r in self.stats(name, gates=[g['id'] for g in gate_list],
                                    channels=[], stats=(), all_events=False):
                    counts[r['id']] = (math.nan if r['count'] is None
                                       else r['count'])
            w.add_sample(
                name=name, fcs_path=sample_fcs(sample),
                channels=list(data.columns), gates=gate_list,
                compensation=own, transforms=stored,
                unknown_fns=match_unknown_scales(
                    sample, stored, gate_list, *(own or (None, None))),
                group=rec.trial, counts=counts, events=len(data),
                set_number=getattr(sample, 'set_number', None))
            gates_out += len(gate_list)
        w.write(str(path))
        shared = [f"{' and '.join(n for ns in files.values() for n in ns)} "
                  f"come from different files named {file}: FlowJo finds "
                  "one file for all of them"
                  for file, files in by_file.items()
                  if names_only and len(files) > 1]
        return {'samples': len(self._samples) - len(not_written),
                'gates': gates_out, 'left_out': left_out,
                'not_written': not_written,
                'warnings': [str(x) for x in w.warnings] + shared}

    def unpack_project(self, path: str, dest: str,
                       max_bytes: int | None = None) -> dict[str, Any]:
        """Unpack the OpenFlo project archive at `path` (a ``.flowproject``,
        or an older ``.zip``) into the new folder `dest`, checked first
        (``project_archive.check_project_archive``; `max_bytes` caps its
        unpacked size). Nothing is opened: the session it holds opens with
        ``open_session``, its FCS files found beside it.

        Returns ``{'session': path of the session, 'missing': [samples the
        archive was saved without the FCS file of]}``. Raises
        ``DocumentError`` for a file that is not a project archive, or not
        a safe one; nothing is left in `dest` then."""
        import zipfile
        import zlib

        from .project_archive import ProjectArchiveError, open_project_archive
        path, dest = str(path), str(dest)
        if os.path.exists(dest):
            raise DocumentError(f"{dest} already exists")
        try:
            session, manifest = open_project_archive(
                path, dest, max_bytes=max_bytes)
        except ProjectArchiveError as exc:
            raise DocumentError(str(exc)) from exc
        except (zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error,
                EOFError) as exc:
            raise DocumentError(
                f"{os.path.basename(path)} is not a readable OpenFlo project "
                f"({exc})") from exc
        except OSError as exc:
            # e.g. a file and a folder of one name: say what, not where (the
            # hidden unpack folder means nothing to the user).
            raise DocumentError(
                f"{os.path.basename(path)} could not be unpacked "
                f"({exc.strerror or type(exc).__name__})") from exc
        missing = manifest.get('missing')
        return {'session': session,
                'missing': [str(m) for m in missing]
                if isinstance(missing, list) else []}

    def open_session(self, path: str) -> dict[str, Any]:
        """Open the OpenFlo session at `path` (``session_open.plan_session``):
        each sample from its processed-data sidecar when it has one, else
        its raw FCS (prepared with the session's matrix and QC choice), put
        back on the transforms it was saved with, with its saved gates.

        Returns ``{'samples': [names opened], 'missing': [names whose files
        were not found], 'notes': [text]}`` (``notes`` with each sample's
        load warnings, as the desktop lists them). A missing sample is
        kept, gates and all, so a save writes it back. Raises ``DocumentError`` for a
        file that is not a session, was written by a newer OpenFlo or is
        damaged (``session_check``)."""
        from .channel_label_trail import explicit_labels_from_trail
        from .session_format import SessionError
        from .session_open import plan_session
        from .wsp_notes import sample_load_notes
        path = str(path)
        data = _read_session_json(path)
        sess_dir = os.path.dirname(os.path.abspath(path))
        data_dir = os.path.splitext(os.path.abspath(path))[0] + '_data'
        try:
            plan = plan_session(data, sess_dir, data_dir=data_dir)
        except SessionError as exc:
            raise DocumentError(str(exc)) from exc
        self._read_from(path)
        # The labels it set (relabels, a template's), not read from a file:
        # an FCS export writes them (channel_label_trail). One set here
        # already wins.
        for ch, lab in explicit_labels_from_trail(
                plan.data.get('audit'), plan.data.get('channel_labels')).items():
            self._labels.setdefault(ch, lab)
        if not self._view:
            self._view = {k: copy.deepcopy(plan.data[k])
                          for k in _VIEW_DEFAULTS if k in plan.data}
        else:
            # A later session adds the channel labels the first did not
            # set; on a channel both label, the first session's wins.
            more = plan.data.get('channel_labels')
            if isinstance(more, Mapping) and more:
                mine = self._view.get('channel_labels')
                mine = dict(mine) if isinstance(mine, Mapping) else {}
                for k, v in more.items():
                    mine.setdefault(k, v)
                self._view['channel_labels'] = mine
        out: dict[str, Any] = {'samples': [], 'missing': [],
                               'notes': list(plan.notes)}
        for p in plan.samples:
            if not p.staged:
                # An entry naming no sample: opened from its raw file as
                # the editor queues it, nothing restored onto it.
                if p.raw_found:
                    opened = self.open_fcs(p.raw_path)
                    out['samples'].append(opened)
                    out['notes'].extend(sample_load_notes(
                        opened, self._samples[opened].data))
                continue
            name = self._free_name(p.name)
            s, sidecar, stamps = self._load_planned(p, name, out['notes'])
            if s is None:
                from .session_open import offline_entry
                self._offline[name] = {
                    'entry': dict(offline_entry(p.entry, sess_dir),
                                  name=name),
                    'gates': [dict(g) for g in p.gates],
                    'kept': (p.entry.get('rel_path') or '', sess_dir)}
                out['missing'].append(p.label)
                continue
            meta = p.meta or {}
            self._restore_meta(name, s, meta, out['notes'])
            self._add(_Sample(name=name, data=s, entry=dict(p.entry),
                              session_dir=sess_dir, sidecar=sidecar,
                              trial=meta.get('trial'), stamps=stamps),
                      gates=meta.get('gates'))
            out['samples'].append(name)
            out['notes'].extend(sample_load_notes(name, s))
        if plan.active in self._samples:
            self._active = plan.active
        return out

    def _load_planned(self, p: Any, name: str, notes: list[str]
                      ) -> tuple[Any, str, tuple[tuple[str, Any], ...]]:
        """``(sample, sidecar it loaded from, stamps)`` for planned session
        sample `p`, or ``(None, '', ())`` when neither its sidecar nor its
        raw FCS can be read. A sidecar that cannot be used gives way to the
        raw FCS, as in the editor. `stamps` are the sidecar's and its
        transform record's ``_stamp``, taken before they are read."""
        from .cli_panel import parse_labels
        from .sample_prep import (
            prepare_fcs_sample,
            read_sidecar_checked,
            sidecar_sample,
        )
        from .transform_sidecar import transforms_sidecar_path
        prep = p.prep or {}
        if p.processed_path and p.load_found:
            # Read here: the sidecar and its transform record.
            record = transforms_sidecar_path(p.processed_path)
            stamps = ((p.processed_path, _stamp(p.processed_path)),
                      (record, _stamp(record)))
            self._read_from(p.processed_path)
            self._read_from(record)
            df, problem = read_sidecar_checked(p.processed_path)
            if not problem:
                fcs = str(prep.get('source_fcs') or '')
                if fcs:
                    # sidecar_sample reads its keywords.
                    self._read_from(fcs)
                s, assumed = sidecar_sample(
                    df, p.processed_path, prep, name=name,
                    labels=parse_labels(self.labels) or None)
                if assumed:
                    notes.append(assumed)
                return s, p.processed_path, stamps
            fcs = prep.get('source_fcs') or ''
            notes.append(f"{name}: its saved data was not used ({problem}); "
                         + ("reloaded from its FCS file" if fcs
                            and os.path.isfile(fcs)
                            else "its FCS file was not found"))
            if not (fcs and os.path.isfile(fcs)):
                return None, '', ()
            raw = fcs
        elif p.raw_found:
            raw = p.raw_path
        else:
            return None, '', ()
        s = prepare_fcs_sample(raw, prep, labels=self.labels,
                               qc_default=self.qc)
        self._read_from(raw)
        return s, '', ()

    @staticmethod
    def _restore_meta(name: str, s: Any, meta: Mapping[str, Any],
                      notes: list[str]) -> None:
        """Put back on sample `s` what its session saved that changes its
        values or statistics: its transforms, derived channels and the
        workspace axes its GeoMean is taken on."""
        from .sample_prep import restore_sample_transforms
        if meta.get('transforms') is not None:
            restore_sample_transforms(s, meta['transforms'])
        if meta.get('stat_axes'):
            s.stat_axes = {c: a for c, a in meta['stat_axes'].items()
                           if c in s.data.columns}
        if meta.get('derived'):
            from .derived import restore as restore_derived
            for col, why in restore_derived(s, meta['derived']):
                notes.append(f"{name}: derived channel {col} not restored "
                             f"({why})")

    def import_wsp(self, path: str,
                   locator: Callable[[str | None], str | None] | Any = None
                   ) -> dict[str, Any]:
        """Import the FlowJo workspace at `path` (``wsp_open.plan_wsp``):
        every sample whose FCS file `locator` finds (an object with
        ``find(uri)`` or a function of the uri; None looks beside the
        workspace and in its subfolders) is opened with the workspace's own
        matrix when it has one, and gets the workspace's gates on its
        stored units (``wsp_open.gates_on_sample``).

        Returns ``{'samples', 'unresolved', 'mismatched', 'unread',
        'ambiguous', 'uncompensated', 'notes'}`` as plain lists (see
        ``wsp_open.WspPlan``; ``uncompensated`` is ``[gate, channel]`` per
        gate FlowJo evaluates on an uncompensated parameter). ``notes`` is
        what the import says to the user, in the desktop's words
        (``wsp_notes``): the desktop editor's load notes for this
        workspace, then each sample's load warnings (a matrix that could
        not be used, an event count other than $TOT, a channel named
        twice), then one per boolean gate built on an ellipse that was not
        converted, then one per gate on an uncompensated parameter;
        each text once, as the desktop lists them. Raises
        ``wsp_open.WspUnreadable`` when the workspace cannot be read."""
        from .flowjo_placeholders import placeholder_booleans
        from .sample_prep import (
            prepare_fcs_sample,
            sample_name_for,
            stored_transforms,
        )
        from .wsp_notes import (
            placeholder_boolean_note,
            sample_load_notes,
            uncompensated_gate_note,
            wsp_load_notes,
        )
        from .wsp_open import gates_on_sample, plan_wsp
        path = str(path)
        plan = plan_wsp(path, locator)
        self._read_from(path)
        names: list[str] = []
        uncompensated: list[list[str]] = []
        in_several: list[tuple[str, list[str]]] = []
        booleans: list[str] = []
        warned: list[str] = []
        for rec in plan.samples:
            prep: dict[str, Any] = {'comp': rec.comp} if rec.comp else {}
            if rec.set_number is not None:
                prep['set_number'] = rec.set_number
            s = prepare_fcs_sample(rec.fcs_path, prep or None,
                                   labels=self.labels, qc_default=self.qc)
            self._read_from(rec.fcs_path)
            name = sample_name_for(rec.fcs_path, self._taken,
                                   base=rec.set_name)
            if rec.stat_axes:
                s.stat_axes = {c: dict(a) for c, a in rec.stat_axes.items()
                               if c in s.data.columns}
            gates, unc = gates_on_sample(rec.gates, stored_transforms(s), s)
            uncompensated.extend([g, ch] for g, ch in unc)
            self._add(_Sample(name=name, data=s,
                              trial=rec.groups[0] if rec.groups else None),
                      gates=gates)
            names.append(name)
            warned.extend(sample_load_notes(name, s))
            if len(rec.groups) > 1:
                in_several.append((rec.node_name or name, list(rec.groups)))
            booleans.extend(placeholder_booleans(rec.gates))
        wsp = os.path.basename(path)
        # In the desktop's order: the workspace's notes, each sample's load
        # warnings as it loads, then what the load says once it settles.
        notes = wsp_load_notes(wsp, plan, in_several) + warned
        notes.extend(placeholder_boolean_note(wsp, b)
                     for b in dict.fromkeys(booleans))
        notes.extend(uncompensated_gate_note(g, ch) for g, ch in uncompensated)
        # Once each, as the desktop's load-notes dialog lists them.
        notes = list(dict.fromkeys(notes))
        return {'samples': names,
                'unresolved': list(plan.unresolved),
                'mismatched': [list(m) for m in plan.mismatched],
                'unread': [[pop, list(tags), int(n)]
                           for pop, tags, n in plan.unread],
                'ambiguous': [[f, int(n)] for f, n in plan.ambiguous],
                'uncompensated': uncompensated,
                'notes': notes}

    def _taken(self, name: str) -> bool:
        return name in self._samples or name in self._offline

    def _free_name(self, name: str) -> str:
        """`name`, or `name (2)`, `name (3)`, ... when it is taken."""
        out, n = name, 2
        while self._taken(out):
            out = f'{name} ({n})'
            n += 1
        return out

    def _add(self, rec: _Sample, gates: Iterable[gate_store.Gate] | None
             = None) -> None:
        """Hold `rec`, installing `gates` (a saved or imported list) with
        fresh ids (``gate_store.install_gates``)."""
        self._samples[rec.name] = rec
        self._seqs.setdefault(rec.name, 0)
        if gates:
            gate_store.install_gates(rec.gates, rec.order, list(gates),
                                     self._new_id(rec.name))
        if self._active is None:
            self._active = rec.name

    def _new_id(self, name: str) -> Callable[[], str]:
        return lambda: gate_store.next_gate_id(self._seqs, name)

    def remove_sample(self, sample: str) -> None:
        """Forget sample `sample` (loaded or missing) and its gates."""
        if self._samples.pop(sample, None) is None:
            if self._offline.pop(sample, None) is None:
                raise DocumentError(f"no sample named {sample!r}")
        if self._active == sample:
            self._active = next(iter(self._samples), None)

    # ── listing ───────────────────────────────────────────────────────────

    def _get(self, sample: str) -> _Sample:
        rec = self._samples.get(sample)
        if rec is None:
            why = (" (its file was not found)" if sample in self._offline
                   else "")
            raise DocumentError(f"no loaded sample named {sample!r}{why}")
        return rec

    def samples(self) -> list[dict[str, Any]]:
        """One record per sample, loaded ones first: ``{'name', 'loaded',
        'path', 'events', 'channels', 'gates', 'compensated'}``. A sample
        whose file was not found has ``loaded`` False, its saved path and
        None for what only its data can say."""
        out = []
        for name, rec in self._samples.items():
            s = rec.data
            out.append({
                'name': name, 'loaded': True,
                'path': str(getattr(s, 'source_fcs', '')
                            or getattr(s, 'path', '') or ''),
                'events': int(len(s.data)),
                'channels': int(len(s.data.columns)),
                'gates': len(rec.order),
                'compensated': bool(getattr(s, 'comp_channels', None))})
        for name, off in self._offline.items():
            out.append({'name': name, 'loaded': False,
                        'path': str(off['entry'].get('path') or ''),
                        'events': None, 'channels': None,
                        'gates': len(off['gates']), 'compensated': None})
        return out

    def keywords(self, sample: str) -> dict[str, Any]:
        """What sample `sample`'s FCS file says about itself, for a viewer:
        ``keyword_view.keyword_view`` of its TEXT keywords (a summary, one
        row per channel, every keyword) and ``'file'``, the file's name.

        Read from the file as written (``fcs_text.fcs_text_pairs``: keys
        keep their case and '$'); when the file cannot be read any more (gone,
        or changed and cut short since), from the keywords the loader kept
        (FlowIO's lower-case, $-less keys)."""
        from .fcs_data_sets import data_set_location
        from .fcs_text import fcs_text_pairs
        from .keyword_view import keyword_view
        s = self._get(sample).data
        path = str(getattr(s, 'source_fcs', '') or getattr(s, 'path', '') or '')
        number = getattr(s, 'set_number', None)
        try:
            # The sample's own data set of a file holding several (or its
            # own well of a BD Accuri file).
            pairs = fcs_text_pairs(*data_set_location(path, number))
        except (OSError, ValueError, IndexError):   # gone, changed, cut short
            pairs = [(str(k), '' if v is None else str(v))
                     for k, v in (getattr(s, 'metadata', None) or {}).items()]
        return {'file': os.path.basename(path), **keyword_view(pairs)}

    @property
    def active(self) -> str | None:
        """The sample a front end shows first: the one the session was saved
        with active, else the first opened."""
        return self._active

    def channels(self, sample: str) -> list[dict[str, Any]]:
        """Sample `sample`'s channels in data order: ``{'name', 'label',
        'kind', 'transform'}``. ``kind`` is 'scatter', 'fluor' or 'other'
        (time, a computed column); ``transform`` the method its stored values
        carry ('linear' when none, 'unknown' when it cannot be known).
        A label set in the session (its ``channel_labels``) wins over the
        file's, as the editor shows it, and one set here
        (``set_channel_label``) over both."""
        s = self._get(sample).data
        scatter = set(getattr(s, 'scatter_channels', None) or ())
        fluor = set(getattr(s, 'fluor_channels', None) or ())
        labels = self._labels_of(self._get(sample))
        rec = getattr(s, 'data_transforms', None) or {}
        out = []
        for ch in s.data.columns:
            spec = rec.get(ch) or {}
            out.append({
                'name': str(ch), 'label': str(labels.get(ch) or ch),
                'kind': ('scatter' if ch in scatter
                         else 'fluor' if ch in fluor else 'other'),
                'transform': str(spec.get('method', 'linear'))})
        return out

    def set_channel_label(self, channel: str, label: str) -> None:
        """Name the marker on detector `channel` `label` for every sample of
        this document, as the editor's channel labels do: ``channels`` lists
        it over the session's and the file's label, and a saved session
        keeps it in its ``channel_labels``. Runs of white space (tabs and
        line breaks too) become one space and those round it are dropped;
        an empty `label` drops the name set here, so the channel shows the
        label its session or file gave it again (the desktop's revert to
        the files' labels, for one channel). A channel no open sample has,
        a label longer than ``MAX_LABEL`` characters or one holding any
        other control character (C0, DEL, C1) is refused (DocumentError)
        and nothing changes."""
        if not isinstance(channel, str) or not isinstance(label, str):
            raise DocumentError("a channel and its label are text")
        if not any(channel in rec.data.data.columns
                   for rec in self._samples.values()):
            raise DocumentError(f"no open sample has a channel {channel!r}")
        if any(ch not in _LABEL_SPACES and (ord(ch) < 32 or 127 <= ord(ch) < 160)
               for ch in label):
            raise DocumentError("a marker label cannot hold a control "
                                "character")
        text = ' '.join(label.split())
        if len(text) > MAX_LABEL:
            raise DocumentError(f"a marker label is at most {MAX_LABEL} "
                                "characters")
        if text:
            self._labels[channel] = text
        else:
            self._labels.pop(channel, None)
        from datetime import datetime
        # In the provenance trail as the editor records a relabel, so a
        # saved session reopened (here or on the desktop) knows the label
        # was set, not read from a file, and an FCS export writes it.
        self._trail.append({'time': datetime.now().isoformat(timespec='seconds'),
                            'action': 'channel_labels.set',
                            'details': {'source': 'Label a channel', 'labels': {channel: text},
                                        'explicit': {channel: text} if text else {}}})

    def gate(self, sample: str, gid: str) -> dict[str, Any]:
        """A copy of gate `gid` of sample `sample`."""
        rec = self._get(sample)
        if gid not in rec.gates:
            raise DocumentError(f"{sample}: no gate {gid!r}")
        return copy.deepcopy(dict(rec.gates[gid], id=gid))

    def gate_tree(self, sample: str) -> list[dict[str, Any]]:
        """Sample `sample`'s gates as a tree in the order they were added:
        root nodes ``{'id', 'name', 'kind', 'parent_id', 'enabled',
        'color', 'children': [nodes]}``. A gate whose parent is gone is a
        root (``gatetree.build_tree``)."""
        from .gatetree import build_tree
        from .gating import population_names
        rec = self._get(sample)
        ordered = {gid: rec.gates[gid] for gid in rec.order
                   if gid in rec.gates}
        transforms = getattr(rec.data, 'data_transforms', None)

        def node(n: Mapping[str, Any]) -> dict[str, Any]:
            g = ordered[n['id']]
            name = population_names(ordered, n['id'], transforms)[-1]
            return {'id': n['id'], 'name': name,
                    'kind': g.get('kind'), 'parent_id': g.get('parent_id'),
                    'enabled': bool(g.get('enabled', True)),
                    'color': g.get('color'),
                    'children': [node(c) for c in n['children']]}
        return [node(n) for n in build_tree(ordered)]

    # ── editing ───────────────────────────────────────────────────────────

    @staticmethod
    def _parent_ok(rec: _Sample, parent_id: Any) -> None:
        """Refuse a parent that is not a gate of the sample. A parent that
        is the gate itself or under it is refused by ``_operands_ok``
        (``gate_store.depends_on`` follows parents as well as operands)."""
        if parent_id is None:
            return
        if parent_id not in rec.gates:
            raise DocumentError(f"{rec.name}: no gate {parent_id!r} to put "
                                "a gate under")

    @staticmethod
    def _operands_ok(rec: _Sample, g: Mapping[str, Any],
                     gid: str | None = None, restore: bool = False) -> None:
        """Refuse boolean gate `g` (to be stored as `gid`, None for a new
        gate) whose operands are missing (other than those gate `gid`
        names already, or any with `restore`) or would make a population
        rest on itself (``gate_store.operand_problem``, the rule the
        desktop's Boolean gate dialog keeps too)."""
        why = gate_store.operand_problem(rec.gates, g, gid, restore)
        if why is not None:
            raise DocumentError(f"{rec.name}: {why}")

    @staticmethod
    def _checked(gate: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(gate, Mapping) or not gate.get('kind'):
            raise DocumentError("a gate is a dict with a 'kind'")
        g = copy.deepcopy(dict(gate))
        g.pop('id', None)
        return g

    def add_gate(self, sample: str, gate: Mapping[str, Any],
                 parent_id: str | None = None) -> dict[str, Any]:
        """Add `gate` (a gate dict, see ``gate_eval``) to sample `sample`,
        under `parent_id` unless it names its own parent
        (``gate_store.add_gate``: a threshold or interval gate replaces the
        one on its channel under the same parent). A boolean gate's
        operands must be gates of the sample. Returns ``{'id',
        'replaced'}``."""
        rec = self._get(sample)
        g = self._checked(gate)
        self._parent_ok(rec, g.get('parent_id', parent_id))
        self._operands_ok(rec, g)
        added = gate_store.add_gate(rec.gates, rec.order, g,
                                    self._new_id(sample), parent_id)
        return {'id': added.id, 'replaced': bool(added.replaced)}

    def add_quadrant(self, sample: str, x: str, y: str, x0: float,
                     y0: float, parent_id: str | None = None,
                     ) -> list[str]:
        """Add a quadrant on channels `x`, `y` crossing at (`x0`, `y0`)
        (stored coordinates) under `parent_id`, as the editor's quadrant
        tool does: four rectangle gates with open outer sides, so together
        they hold every event of the parent, sharing a new quadrant-set id
        (``quad_set``) and named Q++ (x>, y>), Q+- (x>, y<), Q-+ (x<, y>)
        and Q-- (x<, y<). Returns their ids in that order."""
        from .scales import OPEN
        rec = self._get(sample)
        for ch in (x, y):
            if ch not in rec.data.data.columns:
                raise DocumentError(f"{sample}: no channel {ch!r}")
        xc, yc = float(x0), float(y0)
        if not (math.isfinite(xc) and math.isfinite(yc)):
            raise DocumentError("a quadrant's center is two finite numbers")
        self._parent_ok(rec, parent_id)
        taken = set().union(*(gate_store.quad_set_ids(r.gates.values())
                              for r in self._samples.values()))
        qs = f'qs{gate_store.highest_quad_set_number(taken) + 1}'
        ids = []
        for label, ax0, ax1, ay0, ay1 in (
                ('Q++ (x>, y>)', xc, OPEN, yc, OPEN),
                ('Q+- (x>, y<)', xc, OPEN, -OPEN, yc),
                ('Q-+ (x<, y>)', -OPEN, xc, yc, OPEN),
                ('Q-- (x<, y<)', -OPEN, xc, -OPEN, yc)):
            g = {'kind': 'rect', 'x_channel': x, 'y_channel': y,
                 'x0': ax0, 'x1': ax1, 'y0': ay0, 'y1': ay1,
                 'name': label, 'quad_set': qs,
                 'quad_origin_x': xc, 'quad_origin_y': yc}
            ids.append(gate_store.add_gate(rec.gates, rec.order, g,
                                           self._new_id(sample),
                                           parent_id).id)
        return ids

    def copy_gates(self, sample: str, to: Sequence[str] | None = None, *,
                   replace: bool = True) -> dict[str, dict[str, Any]]:
        """Copy sample `sample`'s gates to the samples `to` (None: every
        other loaded sample): the whole tree, parents before children, each
        target giving them ids of its own (``gate_store.install_gates``,
        with a boolean's operands and a quadrant set followed). With
        `replace` a target's own gates are removed first, so its tree ends
        up the same as the source's; else they are added after them.
        Returns per target ``{'ids': [new ids in the source's order],
        'removed': n, 'missing': [channels the gates use that the target
        lacks]}`` (gates on a missing channel are copied and select
        nothing, as on the desktop)."""
        src = self._get(sample)
        targets = ([n for n in self._samples if n != sample]
                   if to is None else list(dict.fromkeys(to)))
        for t in targets:
            self._get(t)
            if t == sample:
                raise DocumentError(f"{sample}: a sample's gates cannot be "
                                    "copied onto itself")
        pending = [dict(copy.deepcopy(src.gates[gid]), id=gid)
                   for gid in src.order if gid in src.gates]
        from .gating import gate_channels
        used = {ch for g in pending for ch in gate_channels(g)}
        out: dict[str, dict[str, Any]] = {}
        for t in targets:
            rec = self._get(t)
            removed = (gate_store.clear_gates(rec.gates, rec.order)
                       if replace else 0)
            ids = gate_store.install_gates(
                rec.gates, rec.order, copy.deepcopy(pending),
                self._new_id(t))
            cols = set(rec.data.data.columns)
            out[t] = {'ids': [ids[g['id']] for g in pending if g['id'] in ids],
                      'removed': removed,
                      'missing': sorted(c for c in used if c not in cols)}
        return out

    def restore_gates(self, sample: str, gates: Sequence[Mapping[str, Any]]
                      ) -> dict[str, str]:
        """Put `gates` back in sample `sample`, as they were saved (each
        with its ``id``; parents before children): Undo of a delete or of a
        replace. Each takes a new id. A ``parent_id`` or boolean operand
        naming one of `gates` follows it to its new id, whatever the order
        (a boolean may come before a gate it combines); one naming a gate
        of the sample stays. An operand naming a gate in neither stays as
        it was: the boolean is back as it was deleted, still lost and
        selecting nothing (``lost_operands``), and a later Undo that puts
        that gate back points it there. A gate's id is never given again in
        a document, so such an operand names no other gate. A parent in
        neither is refused, and then nothing changes. Returns ``{saved id:
        new id}``."""
        rec = self._get(sample)
        pending = [self._checked(g) | {'_saved': g.get('id')} for g in gates]
        saved = {g['_saved'] for g in pending}
        for g in pending:
            parent = g.get('parent_id')
            if parent is not None and parent not in saved:
                self._parent_ok(rec, parent)
        new_id = self._new_id(sample)
        old_to_new: dict[str, str] = {}
        added = []
        for g in pending:
            src = g.pop('_saved')
            gid = gate_store.add_gate(rec.gates, rec.order, g, new_id,
                                      replace_1d=False).id
            added.append(gid)
            if src is not None:
                old_to_new[src] = gid
        # Once every id is known: a parent or an operand listed after the
        # gate that names it is followed as well.
        for gid in added:
            g = rec.gates[gid]
            parent = g.get('parent_id')
            if parent is not None:
                g['parent_id'] = old_to_new.get(parent, parent)
            if g.get('kind') == 'boolean' and g.get('operands'):
                g['operands'] = [old_to_new.get(o, o) for o in g['operands']]
        return old_to_new

    def apply_gatingml(self, path: str, samples: Sequence[str], *,
                       overwrite: bool = True) -> dict[str, Any]:
        """Put the gates of the Gating-ML file at `path` on each of
        `samples`, as the desktop's Apply template does with one
        (``template_apply``): retargeted by marker, onto each sample's
        stored scales, a boolean's operands followed. With `overwrite` a
        sample's own gates are removed first; else the template's are added
        after them (a 1-D gate replacing the sample's own on its channel
        and parent). Returns ``{'status': the desktop's status line,
        'notes': the file's import notes, then the samples lacking a
        channel the gates read, 'ids': {sample: new ids in the file's
        order}}``. A file that cannot be read changes nothing."""
        from .gate_templates import template_ids
        from .gatingml import read_gatingml
        from .sample_prep import stored_transforms
        from .template_apply import (
            MISMATCH_HEAD,
            applied_status,
            broken_template_booleans,
            channel_mismatches,
            mismatch_lines,
            note_uncompensated,
            onto_sample_scales,
            retarget_template,
            template_source,
        )
        targets = [self._get(n).name for n in dict.fromkeys(samples)]
        if not targets:
            raise DocumentError("no sample to apply the gates to")
        try:
            raw, notes = read_gatingml(path, raw=True)
        except Exception as exc:  # noqa: BLE001 - any parse failure is the file's; said, and nothing changes
            raise DocumentError(f"{os.path.basename(path)} could not be "
                                f"read as Gating-ML: {exc}") from exc
        gate_dicts = template_ids(raw)
        lost: list[str] = []
        built_on: list[str] = []
        mismatches: dict[str, int] = {}
        ids: dict[str, list[str]] = {}
        for name in targets:
            rec = self._get(name)
            mine = retarget_template(gate_dicts, rec.data, legacy_label=False)
            note_uncompensated(name, rec.data, mine, notes)
            added = self._install_template(
                rec, onto_sample_scales(mine, stored_transforms(rec.data),
                                        from_gml=True), overwrite)
            got_lost, got_built = broken_template_booleans(rec.gates, added)
            lost += got_lost
            built_on += got_built
            miss = channel_mismatches(rec.data.data.columns, mine)
            if miss:
                mismatches[name] = miss
            ids[name] = added
        if mismatches:
            notes += [MISMATCH_HEAD, *mismatch_lines(mismatches)]
        return {'status': applied_status(gate_dicts, template_source(path),
                                         overwrite, len(targets), lost,
                                         built_on),
                'notes': notes, 'ids': ids}

    def _install_template(self, rec: _Sample, gates: list[gate_store.Gate],
                          overwrite: bool) -> list[str]:
        """Install a template's `gates` (on `rec`'s scales, parents first)
        in sample `rec` (``template_apply.install_template``), after
        removing its own with `overwrite`. A 1-D gate may replace one the
        sample had, never a sibling from the template. Returns the new ids.
        (A Gating-ML file holds no quadrant sets, so none needs a new id.)"""
        from .template_apply import install_template
        if overwrite:
            gate_store.clear_gates(rec.gates, rec.order)
        before = set(rec.gates)
        new_id = self._new_id(rec.name)

        def add(g: gate_store.Gate) -> str:
            return gate_store.add_gate(rec.gates, rec.order, g, new_id,
                                       replace_1d=before).id
        return install_template(gates, add)

    def update_gate(self, sample: str, gid: str,
                    gate: Mapping[str, Any], *,
                    restore: bool = False) -> dict[str, Any]:
        """Put `gate` in the place of gate `gid` (its id and place in the
        tree kept; its colour, on/off state, name and expand state kept
        unless `gate` gives them; its parent kept unless `gate` names
        one). A gate that would rest on itself, through its parents or a
        boolean's operands, is refused. A boolean operand no gate of the
        sample has is refused unless gate `gid` names it already, or
        `restore` (Undo putting the gate back as it was saved: it comes
        back still lost, selecting nothing, as ``restore_gates`` keeps
        such an operand). Returns the stored gate, as ``gate`` returns
        it."""
        rec = self._get(sample)
        if gid not in rec.gates:
            raise DocumentError(f"{sample}: no gate {gid!r}")
        g = self._checked(gate)
        g.setdefault('parent_id', rec.gates[gid].get('parent_id'))
        self._parent_ok(rec, g['parent_id'])
        self._operands_ok(rec, g, gid, restore)
        keep = [k for k in ('color', 'enabled', 'name', 'open')
                if k not in g]
        gate_store.update_gate(rec.gates, gid, g, keep)
        return self.gate(sample, gid)

    def delete_gate(self, sample: str, gid: str) -> list[str]:
        """Delete gate `gid` of sample `sample` and every gate under it
        (``gate_store.remove_gate``). Returns the ids removed."""
        rec = self._get(sample)
        if gid not in rec.gates:
            raise DocumentError(f"{sample}: no gate {gid!r}")
        return gate_store.remove_gate(rec.gates, rec.order, gid)

    # ── numbers ───────────────────────────────────────────────────────────

    def same_population(self, sample: str, gate: str, other: str
                        ) -> str | None:
        """The gate of sample `other` that is population `gate` of
        `sample`, or None when it has none: matched by
        ``gating.population_keys`` (the gate's name and its place in the
        tree, or for an unnamed gate its kind, markers and sibling rank),
        as the editor's Compare all and every-sample edits match them."""
        from .gating import population_keys
        rec, them = self._get(sample), self._get(other)
        if gate not in rec.gates:
            raise DocumentError(f"{sample}: no gate {gate!r}")
        key = population_keys(rec.gates, self._labels_of(rec))[gate]
        for gid, k in population_keys(them.gates,
                                      self._labels_of(them)).items():
            if k == key:
                return gid
        return None

    def _labels_of(self, rec: _Sample) -> dict[str, str]:
        """Sample `rec`'s channel labels as ``channels`` reports them."""
        return {**(getattr(rec.data, 'channel_labels', None) or {}),
                **(self._view.get('channel_labels') or {}),
                **self._labels}

    def lost_operands(self, sample: str) -> dict[str, Any]:
        """The boolean gates of sample `sample` that select nothing because
        an operand is gone (a gate they combine was deleted), and those
        built on one: ``{'lost': [names], 'built_on': [names], 'note':
        text}``, `note` being the desktop's status-line suffix for them
        (``gate_text.lost_operand_note``; '' when there are none)."""
        from .gate_text import broken_boolean_names, lost_operand_note
        lost, built_on = broken_boolean_names(self._get(sample).gates)
        return {'lost': lost, 'built_on': built_on,
                'note': lost_operand_note(lost, built_on)}

    def stats(self, sample: str, *, channels: Sequence[str] | None = None,
              gates: Sequence[str] | None = None,
              stats: Sequence[str] = DEFAULT_CHANNEL_STATS,
              all_events: bool = True) -> list[dict[str, Any]]:
        """Counts and statistics of sample `sample`'s populations
        (``gating.population_stats``, as the editor's Statistics window).

        One row per gate (in tree order, or the ids in `gates`), after an
        'All events' row unless `all_events` is False: ``{'id' (None for
        all events), 'population', 'count', 'pct_parent', 'pct_total',
        'unknown', 'name', 'depth', 'scale', 'channels'}``. ``name`` is the
        population's own name (``gating.population_names``: a name may hold
        a '/'), ``depth`` how many gates down it sits (0 for all events).
        ``scale`` maps each channel to ``{stat: note}``, the note the
        Statistics window and the CSV put after that column's title
        (``gating.stat_column_heading``: ``'(linear)'``, ``'(linear,
        guessed logicle)'``, a GeoMean's FlowJo axis, and the form that
        computed it when not the default, such as ``'FlowJo percentile
        form'``), the same on every row. ``channels`` maps each channel in
        `channels` (default: the fluorescence channels) to ``{stat: value}``
        for each of `stats` (``Median``, ``Mean``, ``GeoMean``, ``CV`` by
        default; any of ``gating.CHANNEL_STAT_ORDER``), keyed in lower case.
        Statistics are on the compensated linear values. A value that is
        not known (an empty population, one resting on a gate of unknown
        region: ``unknown`` says why) is None."""
        from .channel_transforms import is_guessed_spec
        from .gating import (
            CHANNEL_STAT_ORDER,
            GUESSED_TAG,
            population_names,
            population_stats,
            stat_column_heading,
        )
        rec = self._get(sample)
        s = rec.data
        cols = list(s.data.columns)
        chans = [str(c) for c in (channels if channels is not None
                                  else getattr(s, 'fluor_channels', []))
                 if channels is not None or c in cols]
        bad = [c for c in chans if c not in cols]
        if bad:
            raise DocumentError(f"{sample}: no channel {bad[0]!r}")
        wanted = [str(x) for x in stats]
        odd = [x for x in wanted if x not in CHANNEL_STAT_ORDER]
        if odd:
            raise DocumentError(f"unknown statistic {odd[0]!r}; one of "
                                f"{', '.join(CHANNEL_STAT_ORDER)}")
        if gates is not None:
            missing = [g for g in gates if g not in rec.gates]
            if missing:
                raise DocumentError(f"{sample}: no gate {missing[0]!r}")
        transforms = getattr(s, 'data_transforms', None)
        rows = population_stats(
            sample, s.data, rec.gates, rec.order, {c: c for c in chans},
            chans, {'Count', '%Parent', '%Total', *wanted},
            CHANNEL_STAT_ORDER, select=None if gates is None else list(gates),
            transforms=transforms, stat_forms=self.stat_forms,
            stat_axes=getattr(s, 'stat_axes', None), all_events=all_events)
        tag = {c: (f'{c} {GUESSED_TAG}'
                   if is_guessed_spec((transforms or {}).get(c)) else c)
               for c in chans}
        scale = {c: {st.lower(): _heading_note(
                         stat_column_heading(f'{st} {tag[c]}', rows, CHANNEL_STAT_ORDER), f'{st} {c}')
                     for st in wanted} for c in chans}
        out = []
        for r in rows:
            gid = r.get('__gid__')
            names = population_names(rec.gates, gid, transforms) if gid is not None else []
            out.append({
                'id': gid, 'population': r.get('Population'),
                'name': names[-1] if names else r.get('Population'), 'depth': len(names),
                'scale': scale,
                'count': _stat_value(r.get('Count')),
                'pct_parent': _stat_value(r.get('%Parent')),
                'pct_total': _stat_value(r.get('%Total')),
                'unknown': r.get('__unknown__') or None,
                'channels': {c: {st.lower(): _stat_value(r.get(f'{st} {tag[c]}'))
                                 for st in wanted} for c in chans}})
        return out

    def export_population_fcs(self, sample: str, gate: str, path: str
                              ) -> dict[str, Any]:
        """Write the events of population `gate` of `sample` to a NEW FCS
        3.1 file at `path`, as the editor's Export population as FCS writes
        one (``fcs_export``): each channel's compensated LINEAR values (not
        the display coordinates, which a re-import would transform again,
        nor the raw detectors), the source's $PnR ranges, voltages and
        keywords (``source_keywords``: $DATE, $CYT, $TIMESTEP...; the
        applied matrix in OpenFlo's own keywords, never a $SPILLOVER), and
        each channel's marker as $PnS: the file's own, or the one set in
        this document (``set_channel_label``), as the editor's default
        "session names" preference writes them.

        Returns ``{'path', 'events', 'dropped'}`` (``dropped``: columns an
        FCS cannot carry). Refuses a population that is empty or cannot be
        computed (it rests on a gate of unknown region), and a `path` that
        is a file this document read."""
        from .fcs_export import (
            export_channel_labels,
            population_export_frame,
            source_keywords,
            write_fcs,
        )
        rec = self._get(sample)
        if gate not in rec.gates:
            raise DocumentError(f"{sample}: no gate {gate!r}")
        if self._is_source(str(path)):
            raise DocumentError(
                f"{os.path.basename(str(path))} is a file this document read;"
                " the export goes to a new file")
        mask, unknown = self._population(rec, gate)
        if unknown:
            raise DocumentError(f"that population cannot be computed: {unknown}")
        if not mask.any():
            raise DocumentError("that population is empty: nothing to export")
        s = rec.data
        frame, ranges, dropped = population_export_frame(s, mask)
        file_own = dict(getattr(s, 'channel_labels', None) or {})
        labels = export_channel_labels(file_own, 'session', file_own=file_own,
                                       explicit=self._labels)
        keywords, voltages = source_keywords(s)
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        n = write_fcs(frame, str(path), channel_labels=labels, ranges=ranges,
                      keywords=keywords, voltages=voltages)
        return {'path': str(path), 'events': int(n), 'dropped': list(dropped)}

    def export_stats_csv(self, path: str, samples: Sequence[str] | None = None,
                         *, channels: Sequence[str] | None = None,
                         stats: Sequence[str] = DEFAULT_CHANNEL_STATS,
                         separator: str = ',') -> dict[str, Any]:
        """Write the populations' statistics of `samples` (default: every
        loaded sample) as a NEW CSV at `path`, as the editor's Statistics
        window exports them: ``Sample``, ``Population``, ``Count``,
        ``%Parent``, ``%Total``, then ``<stat> <channel>`` for each of
        `channels` (default: each sample's fluorescence channels) and
        `stats`, titled as the window titles it
        (``gating.stat_column_heading``: ``(linear)``, the FlowJo axis
        a GeoMean was taken on, or the form of a statistic not computed in
        the default form); raw values, empty where a value is not a number, ``NA`` in
        every statistic of a population that cannot be computed.

        `separator` ``','`` (the default) writes commas and decimal points,
        the CSV every analysis tool reads; ``';'`` writes semicolons and
        decimal commas, with a byte-order mark, as Excel reads a CSV where
        the decimal mark is a comma (``table_export.SEMICOLON_STYLE``).
        Preferences are not read.

        Returns ``{'path', 'rows', 'columns'}``. Refuses a `path` that is a
        file this document read."""
        from .gating import CHANNEL_STAT_ORDER, stat_column_heading
        from .table_export import SEMICOLON_STYLE, write_rows_csv
        styles = {',': 'standard', ';': SEMICOLON_STYLE}
        if separator not in styles:
            raise DocumentError(f"separator {separator!r}: ',' or ';'")
        rows, cols = self._stats_rows(path, samples, channels, stats)

        def cell(r: Mapping[str, Any], col: str) -> Any:
            if r.get('__unknown__') and col not in ('Sample', 'Population'):
                return 'NA'
            return r.get(col, '')

        write_rows_csv(str(path),
                       [stat_column_heading(c, rows, CHANNEL_STAT_ORDER)
                        for c in cols],
                       [[cell(r, c) for c in cols] for r in rows],
                       style=styles[separator], na_rep='')
        return {'path': str(path), 'rows': len(rows), 'columns': len(cols)}

    def _stats_rows(self, path: str, samples: Sequence[str] | None,
                    channels: Sequence[str] | None, stats: Sequence[str]
                    ) -> tuple[list[dict[str, Any]], list[str]]:
        """The statistics rows of `samples` and their columns in order, as
        export_stats_csv and export_stats_plt write them; refuses a `path`
        that is a file this document read."""
        from .gating import CHANNEL_STAT_ORDER, population_stats
        names = list(self._samples) if samples is None else [str(n) for n in samples]
        if not names:
            raise DocumentError("no loaded sample to export")
        wanted = [str(x) for x in stats]
        odd = [x for x in wanted if x not in CHANNEL_STAT_ORDER]
        if odd:
            raise DocumentError(f"unknown statistic {odd[0]!r}; one of "
                                f"{', '.join(CHANNEL_STAT_ORDER)}")
        if self._is_source(str(path)):
            raise DocumentError(
                f"{os.path.basename(str(path))} is a file this document read;"
                " the export goes to a new file")
        cols: list[str] = []
        rows: list[dict[str, Any]] = []
        for name in names:
            rec = self._get(name)
            s = rec.data
            have = list(s.data.columns)
            chans = [str(c) for c in (channels if channels is not None
                                      else getattr(s, 'fluor_channels', []))
                     if c in have]
            for r in population_stats(
                    name, s.data, rec.gates, rec.order, {}, chans,
                    {'Count', '%Parent', '%Total', *wanted},
                    CHANNEL_STAT_ORDER, transforms=getattr(s, 'data_transforms', None),
                    stat_forms=self.stat_forms,
                    stat_axes=getattr(s, 'stat_axes', None), all_events=True):
                rows.append(r)
                cols += [k for k in r if not k.startswith('__') and k not in cols]
        return rows, cols

    def export_stats_plt(self, path: str, samples: Sequence[str] | None = None,
                         *, channels: Sequence[str] | None = None,
                         stats: Sequence[str] = DEFAULT_CHANNEL_STATS) -> dict[str, Any]:
        """Write the same statistics as export_stats_csv (same rows, same
        columns, titled the same) as a Pocket Lab Table at `path`, the file
        the Statistics app opens with its columns already set
        (stats_plt.stats_table: the population as the analyte, each
        sample's workspace group as the group when there are two or more).

        Returns ``{'path', 'rows', 'columns', 'name'}``, `name` the file
        name to send it under."""
        from .gating import CHANNEL_STAT_ORDER, stat_column_heading
        from .stats_plt import plt_file_name, stats_table, write_stats_plt
        from .update import current_version
        rows, cols = self._stats_rows(path, samples, channels, stats)
        heads = {c: stat_column_heading(c, rows, CHANNEL_STAT_ORDER) for c in cols}
        groups = {name: (str(rec.trial) if rec.trial else None)
                  for name, rec in self._samples.items()}
        table = stats_table(rows, cols, heads, CHANNEL_STAT_ORDER, groups=groups,
                            version=current_version())
        write_stats_plt(str(path), table)
        return {'path': str(path), 'rows': len(rows),
                'columns': len(table['columns']), 'name': plt_file_name()}

    def _view_fns(self, s: Any, channel: str, scale: str | None
                  ) -> tuple[Callable[[Any], np.ndarray],
                             Callable[[Any], np.ndarray]]:
        """``(forward, inverse)`` between `channel`'s stored values (the
        coordinates its gates are in) and the axis of display scale `scale`
        (``'linear'``, ``'log'``, ``'logicle[:w=..]'``, ``'asinh[:c=..]'``,
        see ``scales.parse_scale``). None or ``'stored'`` is the stored
        coordinate itself. Taken through the channel's linear values with
        its own transform parameters."""
        from .channel_transforms import (
            is_unknown_spec,
            to_linear,
            transform_params,
            transform_values,
        )
        from .scales import LOG_VIEW_MIN, intensity_view, scale_kind

        def same(v: Any) -> np.ndarray:
            return np.asarray(v, dtype=float)
        if scale in (None, '', 'stored'):
            return same, same
        spec = (getattr(s, 'data_transforms', None) or {}).get(channel)
        if is_unknown_spec(spec):
            raise DocumentError(
                f"{channel}: its scale is unknown, so it can only be shown "
                "as stored")
        params: dict[str, Any] = (transform_params(spec) if spec
                                  else {'method': 'linear'})

        def to_lin(v: Any) -> np.ndarray:
            return np.asarray(to_linear(np.asarray(v, dtype=float), spec),
                              dtype=float)

        def from_lin(v: Any) -> np.ndarray:
            arr = np.asarray(v, dtype=float)
            if params.get('method', 'linear') == 'linear':
                return arr
            return np.asarray(transform_values(arr.ravel(), **params),
                              dtype=float).reshape(arr.shape)
        kind = scale_kind(scale)
        view = intensity_view(scale)
        if view is not None:
            fwd, inv = view
            return (lambda v: np.asarray(fwd(to_lin(v)), dtype=float),
                    lambda p: from_lin(inv(np.asarray(p, dtype=float))))
        if kind == 'log':
            return (lambda v: np.log10(np.clip(to_lin(v), LOG_VIEW_MIN,
                                               None)),
                    lambda p: from_lin(10.0 ** np.asarray(p, dtype=float)))
        if kind == 'linear':
            return to_lin, from_lin
        raise DocumentError(f"unknown display scale {scale!r}")

    def compensation(self, sample: str) -> dict[str, Any]:
        """What compensation sample `sample` carries: ``{'on', 'source',
        'channels', 'matrix', 'file'}``. ``matrix`` is the spillover applied
        over ``channels`` (float64, rows = source, columns = detector; None
        with no channels), ``source`` where it came from ('$SPILL', a
        workspace, 'manual', 'off'), ``on`` whether it is anything but the
        identity, and ``file`` the file's own $SPILL as ``{'channels',
        'matrix'}`` or None (what ``set_compensation(on=True)`` applies)."""
        s = self._get(sample).data
        chans = list(getattr(s, 'comp_channels', None) or [])
        m = getattr(s, 'comp_matrix', None)
        matrix = None if m is None or not chans else np.asarray(m, float)
        f = s._parse_spillover()
        return {'on': matrix is not None
                and not np.array_equal(matrix, np.eye(len(chans))),
                'source': getattr(s, 'compensation_source', None),
                'channels': chans, 'matrix': matrix,
                'file': None if f is None else {
                    'channels': list(f['channels']),
                    'matrix': np.asarray(f['matrix'], float)}}

    def set_compensation(self, sample: str, on: bool) -> dict[str, Any]:
        """Turn sample `sample`'s compensation off (the identity over the
        channels it compensates, as the desktop editor's Reset does) or on
        (the file's own $SPILL), through ``FlowSample.recompensate``: every
        event and gate is kept, and stored values move. Returns
        ``compensation`` as it now is, with ``'lost'``: how many events
        gained an unknown value."""
        s = self._get(sample).data
        now = self.compensation(sample)
        if bool(on) == now['on']:
            return {**now, 'lost': 0}
        try:
            if on:
                if now['file'] is None:
                    raise DocumentError(f"{sample}: the file has no "
                                        "spillover matrix to turn on")
                f = now['file']
                lost = s.recompensate(f['matrix'], f['channels'],
                                      source='$SPILL')
            else:
                n = len(now['channels'])
                lost = s.recompensate(np.eye(n), now['channels'],
                                      source='off')
        except DocumentError:
            raise
        except Exception as e:  # noqa: BLE001 - the engine's reason, to the caller
            raise DocumentError(f"{sample}: {e}") from e
        return {**self.compensation(sample), 'lost': int(lost)}

    def to_view(self, sample: str, channel: str, values: Any,
                scale: str | None = None, *, inverse: bool = False
                ) -> np.ndarray:
        """`values` of `channel` (stored coordinates, as gates hold them)
        on the axis of display scale `scale` (see ``density``), as float64;
        with `inverse`, axis positions back to stored coordinates (a gate
        drawn on screen)."""
        s = self._get(sample).data
        if channel not in s.data.columns:
            raise DocumentError(f"{sample}: no channel {channel!r}")
        fwd, inv = self._view_fns(s, channel, scale)
        return (inv if inverse else fwd)(values)

    def density(self, sample: str, x: str, y: str, *,
                gate: str | None = None, x_scale: str | None = None,
                y_scale: str | None = None, bins: int | Sequence[int] = 128,
                x_range: Sequence[float] | None = None,
                y_range: Sequence[float] | None = None) -> dict[str, Any]:
        """A binned 2-D density of channels `x` and `y` for a plot: the
        events of population `gate` (None: all events) counted on a
        `bins` grid (one number, or ``(nx, ny)``) over each axis' display
        scale (`x_scale`, `y_scale`: None for the stored coordinate the
        gates are in, else ``'linear'``, ``'log'``, ``'logicle'``,
        ``'asinh'``, with parameters as ``scales.parse_scale`` reads them).

        The ranges default to the finite extent of the whole sample on
        that axis, so a population is drawn where its parent is. Returns
        ``{'counts', 'x_edges', 'y_edges', 'events', 'binned', 'x_range',
        'y_range', 'unknown'}``: ``counts`` is a float32 ``(ny, nx)`` array (row = y
        bin, from the low end), the edges float64 arrays on the display
        axes, ``events`` the population's size and ``binned`` how many of
        them fall on the grid. ``unknown`` is None, or why population
        `gate` cannot be computed (``gate_eval.unknown_population_reason``:
        it rests on a gate of unknown region, on a channel the sample
        lacks, or nests past what can be followed); its ``events`` 0 is
        then not a measurement."""
        rec = self._get(sample)
        s = rec.data
        for ch in (x, y):
            if ch not in s.data.columns:
                raise DocumentError(f"{sample}: no channel {ch!r}")
        nb = ([int(bins)] * 2 if isinstance(bins, (int, np.integer))
              else [int(b) for b in bins])
        if len(nb) != 2 or any(not 1 <= b <= MAX_BINS for b in nb):
            raise DocumentError(f"bins must be 1 to {MAX_BINS} per axis")
        nx, ny = nb
        xs = self._view_fns(s, x, x_scale)[0](s.data[x].values)
        ys = self._view_fns(s, y, y_scale)[0](s.data[y].values)
        mask, unknown = self._population(rec, gate)
        xr = self._axis_range(xs, x_range)
        yr = self._axis_range(ys, y_range)
        keep = mask & np.isfinite(xs) & np.isfinite(ys)
        hist, xe, ye = np.histogram2d(xs[keep], ys[keep], bins=[nx, ny],
                                      range=[xr, yr])
        return {'counts': np.ascontiguousarray(hist.T, dtype=np.float32),
                'x_edges': xe, 'y_edges': ye,
                'events': int(mask.sum()), 'binned': int(hist.sum()),
                'x_range': list(xr), 'y_range': list(yr),
                'unknown': unknown}

    def density_events(self, sample: str, x: str, y: str, *,
                       gate: str | None = None, x_scale: str | None = None,
                       y_scale: str | None = None,
                       max_events: int = 50_000,
                       x_range: Sequence[float] | None = None,
                       y_range: Sequence[float] | None = None,
                       highlight: str | None = None,
                       ) -> dict[str, Any]:
        """The events of population `gate` as the desktop's pseudocolor
        plot draws them: each a dot at its display position, coloured by
        the smoothed density around it (``density.event_density_numpy``
        on a 256 x 256 grid over the axes, then the desktop's colour
        spread: ``(z / max z) ** 0.4``). Axes and scales as ``density``.

        At most `max_events` dots (a population larger than that gives a
        fixed, seeded subsample, so the same request gives the same dots).
        Returns ``{'x', 'y', 'c', 'events', 'shown', 'x_range', 'y_range',
        'unknown'}``: float32 display copies (``c`` in 0..1) ordered by
        density, densest last, so they are drawn on top; ``events`` is the
        population's size and ``shown`` how many dots there are.

        With `highlight` (a gate of the sample), also the events of that
        population as the desktop's backgate draws them, to be drawn on
        top in its own colour: ``'hx'``, ``'hy'`` (float32 display
        positions; subsampled at the same rate as the dots, so the two
        read alike) and ``'h_events'``, its size. It need not lie inside
        `gate`: its events are shown wherever they fall on these axes."""
        from .density import event_density_numpy
        rec = self._get(sample)
        s = rec.data
        for ch in (x, y):
            if ch not in s.data.columns:
                raise DocumentError(f"{sample}: no channel {ch!r}")
        cap = int(max_events)
        if cap < 1:
            raise DocumentError("max_events is at least 1")
        xs = self._view_fns(s, x, x_scale)[0](s.data[x].values)
        ys = self._view_fns(s, y, y_scale)[0](s.data[y].values)
        mask, unknown = self._population(rec, gate)
        xr = self._axis_range(xs, x_range)
        yr = self._axis_range(ys, y_range)
        idx = np.flatnonzero(mask & np.isfinite(xs) & np.isfinite(ys))
        if idx.size > cap:
            idx = np.sort(np.random.default_rng(0).choice(idx, cap,
                                                          replace=False))
        px, py = xs[idx], ys[idx]
        if idx.size:
            z = event_density_numpy(px, py, np.linspace(*xr, 257),
                                    np.linspace(*yr, 257))
            top = float(z.max())
            c = (z / top) ** 0.4 if top > 0 else np.zeros_like(z)
            order = np.argsort(z, kind='stable')
            px, py, c = px[order], py[order], c[order]
        else:
            c = np.zeros(0)
        out = {'x': px.astype(np.float32), 'y': py.astype(np.float32),
               'c': c.astype(np.float32), 'events': int(mask.sum()),
               'shown': int(idx.size), 'x_range': list(xr),
               'y_range': list(yr), 'unknown': unknown}
        if highlight is not None:
            hmask = self._population(rec, highlight)[0]
            hidx = np.flatnonzero(hmask & np.isfinite(xs) & np.isfinite(ys))
            whole = int(np.count_nonzero(mask & np.isfinite(xs)
                                         & np.isfinite(ys)))
            rate = idx.size / whole if whole else 1.0
            keep = max(1, int(round(hidx.size * rate))) if hidx.size else 0
            if hidx.size > keep:
                hidx = np.sort(np.random.default_rng(1).choice(
                    hidx, keep, replace=False))
            out.update(hx=xs[hidx].astype(np.float32),
                       hy=ys[hidx].astype(np.float32),
                       h_events=int(hmask.sum()))
        return out

    def _population(self, rec: _Sample, gate: str | None
                    ) -> tuple[np.ndarray, str | None]:
        """``(mask, unknown)`` of population `gate` of `rec` (None: all
        events): its events, and why it cannot be computed or None
        (``gate_eval.unknown_population_reason``)."""
        from .gate_eval import cumulative_gate_mask, unknown_population_reason
        n = len(rec.data.data)
        if gate is None:
            return np.ones(n, dtype=bool), None
        if gate not in rec.gates:
            raise DocumentError(f"{rec.name}: no gate {gate!r}")
        unknown = unknown_population_reason(
            rec.gates, gate, columns=list(rec.data.data.columns))
        mask = np.asarray(cumulative_gate_mask(rec.gates, gate, rec.data.data,
                                               cache={}), dtype=bool)
        return mask, unknown

    def histogram(self, sample: str, channel: str, *,
                  gate: str | None = None, scale: str | None = None,
                  bins: int = 256, x_range: Sequence[float] | None = None,
                  highlight: str | None = None) -> dict[str, Any]:
        """A binned 1-D histogram of `channel` for a plot: the events of
        population `gate` (None: all events) counted in `bins` equal bins
        over display scale `scale` (as ``density``), across `x_range`
        (default: the finite extent of the whole sample, so a population
        is drawn where its parent is). Returns ``{'counts', 'edges',
        'events', 'binned', 'range', 'unknown'}`` as ``density`` does, with
        ``counts`` a float32 ``(bins,)`` array. With `highlight` (a gate of
        the sample), also ``'h_counts'``: that population's events in the
        same bins (wherever it lies; the desktop's backgate) and
        ``'h_events'``, its size."""
        rec = self._get(sample)
        s = rec.data
        if channel not in s.data.columns:
            raise DocumentError(f"{sample}: no channel {channel!r}")
        nb = int(bins)
        if not 1 <= nb <= MAX_BINS:
            raise DocumentError(f"bins must be 1 to {MAX_BINS}")
        vs = self._view_fns(s, channel, scale)[0](s.data[channel].values)
        mask, unknown = self._population(rec, gate)
        r = self._axis_range(vs, x_range)
        keep = mask & np.isfinite(vs)
        hist, edges = np.histogram(vs[keep], bins=nb, range=r)
        out = {'counts': np.ascontiguousarray(hist, dtype=np.float32),
               'edges': edges, 'events': int(mask.sum()),
               'binned': int(hist.sum()), 'range': list(r),
               'unknown': unknown}
        if highlight is not None:
            hmask = self._population(rec, highlight)[0]
            h = np.histogram(vs[hmask & np.isfinite(vs)], bins=edges)[0]
            out.update(h_counts=np.ascontiguousarray(h, dtype=np.float32),
                       h_events=int(hmask.sum()))
        return out

    def overlays(self, sample: str, x: str, y: str | None = None, *,
                 parent: str | None = None, x_scale: str | None = None,
                 y_scale: str | None = None) -> list[dict[str, Any]]:
        """The outlines of the gates drawn on a plot of `x` (and `y`; None
        for a histogram) whose parent is population `parent` (None: the
        root gates), on the plot's display axes, for a front end to draw
        over ``density`` or ``histogram``.

        One entry per gate that can be drawn there, in tree order:
        ``{'id', 'name', 'kind', 'shape', ...}`` (and ``quad_set`` for a
        quadrant's gates, with ``quad_x`` and ``quad_y``: 1 or -1, the side
        of the crossing it lies on along this plot's x and y). ``shape`` is
        ``'polygon'`` with ``points`` (a float64 ``(n, 2)`` array of
        display x, y; a rectangle is its four corners, an ellipse its
        outline, and an edge is followed through the display scales so it
        lies where the gate's region does), or ``'band'`` with ``axis``
        ('x' or 'y') and ``lo``, ``hi`` (None for an open side). A 2-D
        gate is drawn on a plot of its two channels, either way round; a
        1-D gate (threshold, interval) on any plot with its channel as an
        axis. Gates with no region on these axes (boolean, cluster) are
        left out, and so are gates whose record cannot be evaluated
        (``gate_record.gate_record_problem``)."""
        from .gate_refit import outline
        from .scales import OPEN
        rec = self._get(sample)
        s = rec.data
        axes = [c for c in (x, y) if c is not None]
        for ch in axes:
            if ch not in s.data.columns:
                raise DocumentError(f"{sample}: no channel {ch!r}")
        if parent is not None and parent not in rec.gates:
            raise DocumentError(f"{sample}: no gate {parent!r}")
        fx = self._view_fns(s, x, x_scale)[0]
        fy = self._view_fns(s, y, y_scale)[0] if y is not None else None
        names = {n['id']: n['name'] for n in _walk(self.gate_tree(sample))}

        def side(f: Callable[[Any], np.ndarray], v: Any) -> float | None:
            if v is None or abs(float(v)) >= OPEN:
                return None
            out = float(f([float(v)])[0])
            return out if math.isfinite(out) else None

        out: list[dict[str, Any]] = []
        for gid in rec.order:
            g = rec.gates.get(gid)
            # A gate of a kind this version does not know, or missing a
            # field (gate_record_problem), has no region to draw.
            if (g is None or g.get('parent_id') != parent
                    or gate_record_problem(g)):
                continue
            base = {'id': gid, 'name': names.get(gid, gid),
                    'kind': g.get('kind')}
            if g.get('quad_set'):
                base['quad_set'] = g['quad_set']
            kind = g.get('kind')
            if g.get('quad_set') and kind == 'rect' and y is not None:
                base.update(_quad_sides(g, x, y))
            if kind in ('threshold', 'interval'):
                ch = g.get('channel')
                lo = g.get('value') if kind == 'threshold' else g.get('lo')
                hi = None if kind == 'threshold' else g.get('hi')
                for axis, name, f in (('x', x, fx), ('y', y, fy)):
                    if ch == name and f is not None:
                        out.append({**base, 'shape': 'band', 'axis': axis,
                                    'lo': side(f, lo), 'hi': side(f, hi)})
                        break
                continue
            if y is None or kind not in ('rect', 'polygon', 'ellipsoid'):
                continue
            pair = (g.get('x_channel'), g.get('y_channel'))
            if pair not in ((x, y), (y, x)):
                continue
            corners = _outline_corners(g, outline)
            if corners is None:
                continue
            if pair == (y, x):
                corners = corners[:, ::-1]
            assert fy is not None
            pts = _densified(np.column_stack([
                _open_clipped(corners[:, 0], s.data[x].values),
                _open_clipped(corners[:, 1], s.data[y].values)]))
            px, py = fx(pts[:, 0]), fy(pts[:, 1])
            out.append({**base, 'shape': 'polygon',
                        'points': np.column_stack([px, py])})
        return out

    @staticmethod
    def _axis_range(v: np.ndarray, given: Sequence[float] | None
                    ) -> tuple[float, float]:
        if given is not None:
            lo, hi = float(given[0]), float(given[1])
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise DocumentError("a range is two finite numbers, low "
                                    "first")
            return lo, hi
        fin = v[np.isfinite(v)]
        if not fin.size:
            return 0.0, 1.0
        lo, hi = float(fin.min()), float(fin.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return lo, hi

    # ── saving ────────────────────────────────────────────────────────────

    def _entry(self, rec: _Sample, sess_dir: str) -> dict[str, Any]:
        """Sample `rec`'s entry in a session saved in `sess_dir`: what the
        session it came from held (models, records this document does not
        change), with everything taken from its data worked out again
        (``session_format.column_records``, ``qc_records``) and its relink
        hints for the new folder (``paths.relink_hints``)."""
        from .paths import relink_hints, session_relative
        from .result_stamp import app_version, column_originals, column_stamps
        from .session_format import column_records, qc_records
        s = rec.data
        base = {k: v for k, v in (rec.entry or {}).items()
                if k not in _REWRITTEN}
        src = str(getattr(s, 'source_fcs', '') or '')
        kept_rel = kept_from = ''
        if rec.sidecar and not (src and os.path.isfile(src)):
            # Loaded from its sidecar, its raw file not found (source_fcs
            # is then the saved path, or none): the saved path is from the
            # computer that last had it, and its link is kept (carried over
            # to the folder saved in), as the editor keeps it.
            src = session_relative(str((rec.entry or {}).get('path') or ''),
                                   rec.session_dir)
            kept_rel = str((rec.entry or {}).get('rel_path') or '')
            kept_from = rec.session_dir if kept_rel else ''
        elif not src:
            src = str(getattr(s, 'path', '') or '')
        entry = {'name': rec.name, 'path': src,
                 'color': base.pop('color', _PLACEHOLDER_COLOR),
                 'plot_enabled': bool(base.pop('plot_enabled', False))}
        if rec.trial:
            entry['trial'] = str(rec.trial)
        entry.update(base)
        entry.update(column_records(s))
        entry.update(qc_records(s))
        entry['app_version'] = app_version()
        entry['result_stamps'] = (column_stamps(s)
                                  or dict(base.get('result_stamps') or {}))
        entry['result_originals'] = (column_originals(s) or dict(
            base.get('result_originals') or {}))
        entry['saved_stats'] = None
        entry.update(relink_hints(src, sess_dir, kept_rel=kept_rel,
                                  kept_from=kept_from))
        return entry

    def _offline_entry(self, off: Mapping[str, Any], sess_dir: str
                       ) -> dict[str, Any]:
        """A missing sample's entry, as read, with its link carried over to
        `sess_dir`."""
        from .paths import relink_hints
        entry = dict(off['entry'])
        entry.pop('rel_path', None)
        kept_rel, kept_from = off.get('kept') or ('', '')
        hints = relink_hints(str(entry.get('path') or ''), sess_dir,
                             kept_rel=kept_rel,
                             kept_from=kept_from if kept_rel else '')
        if not hints.get('basename'):
            hints.pop('basename')
        entry.update(hints)
        return entry

    def _session_data(self, sess_dir: str) -> tuple[
            dict[str, Any], list[tuple[str, str, tuple[tuple[str, Any], ...]]]]:
        """``(session dict, [(sidecar to copy, its name in the data
        folder, its stamps when read)])`` for a session saved in
        `sess_dir` (see ``save_session``)."""
        from datetime import datetime

        from .paths import safe_sidecar_name
        from .result_stamp import app_version, settle_gate_stamps
        from .session_format import SESSION_VERSION
        from .view_tabs import gate_lists
        samples, copies = [], []
        for name, rec in self._samples.items():
            settle_gate_stamps(rec.gates)
            entry = self._entry(rec, sess_dir)
            if rec.sidecar:
                ext = os.path.splitext(rec.sidecar)[1].lower()
                stem = safe_sidecar_name(name) + ext
                copies.append((rec.sidecar, stem, rec.stamps))
                entry['_sidecar'] = stem
            samples.append(entry)
        sample_gates = gate_lists({n: r.gates for n, r in self._samples.items()},
                                  {n: r.order for n, r in self._samples.items()})
        for name, off in self._offline.items():
            samples.append(self._offline_entry(off, sess_dir))
            sample_gates.setdefault(name, [dict(g) for g in off['gates']])
        view = {k: copy.deepcopy(self._view.get(k, v))
                for k, v in _VIEW_DEFAULTS.items()}
        if self._trail:
            trail = view['audit'] if isinstance(view['audit'], list) else []
            view['audit'] = trail + [dict(e, seq=len(trail) + i)
                                     for i, e in enumerate(self._trail, 1)]
        if self._labels:
            labels = view['channel_labels']
            view['channel_labels'] = {
                **(labels if isinstance(labels, Mapping) else {}),
                **self._labels}
        data = {'format': SESSION_FORMAT, 'version': SESSION_VERSION,
                'app_version': app_version(),
                'created': datetime.now().isoformat(timespec='seconds'),
                'active_sample': self._active, 'samples': samples,
                'sample_gates': sample_gates, **view}
        return data, copies

    def save_session(self, path: str) -> dict[str, Any]:
        """Write this document as a NEW OpenFlo session at `path`, which the
        gate editor opens with the same samples, gates and statistics.

        A sample loaded from a session's processed-data sidecar gets a copy
        of it (and of its transform record, when it had one) in
        ``<stem>_data`` beside `path`; any other sample is reopened from its
        raw FCS with the matrix, QC choice and transforms saved for it.
        Missing samples are written back as they were read.

        The copies are taken from disk, so a sidecar or transform record
        that changed since this document read it (a re-save by the
        desktop, a sync) raises ``DocumentError`` before anything is
        written: its data would not be the data the gates were drawn on.

        Nothing this document read is ever written or removed: a `path`, a
        sidecar copy or its transform record naming one of those files (by
        any spelling, symbolic link or hard link) raises ``DocumentError``
        before anything is written. Every file is first written whole to a
        new temporary file beside it (made with O_EXCL, so a file or link
        already there is never written through). When a write fails, or the
        save is interrupted, the temporary files (and a ``<stem>_data``
        folder the save made) are removed and nothing already there has
        changed.

        Only when all are written are files changed in place, in this
        order: the files an earlier save left that the reopen would pair
        with this save's samples are removed, then each sidecar copy is
        moved in followed by its record, then the session last. Removed
        are the transform record already beside each copy's destination
        (the desktop saver's ``discard_transforms_sidecar`` rule), the
        other format's file of each copy's own name (``s1.npz`` for a copy
        ``s1.csv`` and the reverse, as the desktop's ``_write_sidecar``
        does) and, for each sample written with no sidecar (loaded from
        its raw FCS, or offline), every file the opener would find for it
        by name in ``<stem>_data`` and their records
        (``session_open.sidecar_name_candidates``). One that stayed would
        describe other data (a re-save over an earlier save). Those names
        are checked like every file written: one naming a file this
        document read is refused, and a sample whose by-name file would be
        another sample's copy (the legacy stem is not one-to-one; compared
        with case folded, so a raw ``s1`` beside a copied ``S1`` is
        refused on every OS; two copies differing only in case are not
        checked yet) is refused, naming it. When no session is at `path`
        yet, any file this save would remove or replace that is already
        in ``<stem>_data`` belongs to another session sharing the folder
        (``run.json`` beside ``run.flowsession``) or to none: the save is
        refused, naming them, and nothing is removed. A
        regular file where ``<stem>_data`` should be is refused too. A
        removal that fails (a file held open on Windows) fails the save
        before anything is moved in. The order means a move that fails
        partway never pairs a sidecar with another one's record: at worst
        a sidecar is left with no record (reopened on the transform
        assumed for it, with a note), and the earlier session at `path`,
        if any, is left pointing at those files.

        A change is told by each file's ``_stamp``: device, inode, size
        and modification time. On some file systems these change when the
        file did not, and the save is then refused ("open the session
        again"), never wrong: a CIFS share mounted with ``noserverino``
        (inode numbers made by the client can change when its cache
        drops them), an NFS remount (a new device number), and in Pyodide
        a worker file system made again between open and save (an IDBFS
        ``syncfs`` from storage gives new inode numbers, and writing the
        same bytes again a new modification time). A front end must
        not reload its file system while a document is open, or must
        open the session again first. Same-size rewrites in place inside
        the modification-time granularity (FAT's 2 s), with the inode
        kept, are not seen.
        Returns ``{'path', 'samples', 'sidecars'}``."""
        import contextlib

        from .session_format import json_safe, report_nonfinite
        from .session_sidecar import LEGACY_EXT, SIDECAR_EXT
        from .transform_sidecar import transforms_sidecar_path
        target = os.path.abspath(str(path))
        sess_dir = os.path.dirname(target)
        data_dir = os.path.splitext(target)[0] + '_data'
        data, copies = self._session_data(sess_dir)
        plan = []                       # (source, destination)
        discard = []                    # files to remove before moving
        stamps: list[tuple[str, Any]] = []
        for src, stem, read in copies:
            dst = os.path.join(data_dir, stem)
            plan.append((src, dst))
            rec_src = transforms_sidecar_path(src)
            if dict(read).get(rec_src) is not None:
                plan.append((rec_src, transforms_sidecar_path(dst)))
            discard.append(transforms_sidecar_path(dst))
            # The other format's file of the same stem, as the desktop's
            # _write_sidecar removes it: when the pointer is refused (a
            # data folder linked out of the session folder) the opener
            # finds the copy by name, .npz first.
            base, ext = os.path.splitext(dst)
            discard.append(base + (SIDECAR_EXT if ext.lower() == LEGACY_EXT
                                   else LEGACY_EXT))
            stamps.extend(read)
        # A sample written with no sidecar pointer (loaded from its raw
        # FCS, or offline) is looked for by name in the data folder when
        # reopened: an earlier save's file there would be read as its data.
        # Compared with case folded on every OS: a session written where
        # case is told apart may be reopened where it is not.
        by_name = _by_name_files(data['samples'], data_dir,
                                 [d for _s, d in plan])
        discard += by_name
        if ((plan or discard) and os.path.lexists(data_dir)
                and not os.path.isdir(data_dir)):
            raise DocumentError(
                f"{os.path.basename(data_dir)} is a file, not a folder; "
                "save under another name")
        for t in [target] + [d for _s, d in plan] + discard:
            if self._is_source(t):
                raise DocumentError(
                    f"{os.path.basename(t)} is a file this document was "
                    "opened from; save to a new file")
        if not os.path.exists(target):
            # Every file this save would remove or replace in the data
            # folder, not only the by-name ones: the other format's file
            # and the record beside a copy may be another session's too.
            _refuse_others_files(
                data_dir, discard + [d for _s, d in plan])

        def unchanged() -> None:
            for f, st in stamps:
                if _stamp(f) != st:
                    raise DocumentError(
                        f"{os.path.basename(f)} has changed since this "
                        "document opened it; open the session again to "
                        "save its current data")
        unchanged()
        for entry in data['samples']:
            stem = entry.pop('_sidecar', None)
            if stem:
                entry['processed_data'] = (
                    f"{os.path.basename(data_dir)}/{stem}")
        data = cast('dict[str, Any]', json_safe(data, report_nonfinite))
        made_dir = bool(plan) and not os.path.isdir(data_dir)
        staged: list[tuple[str, str]] = []       # (temporary, destination)

        def stage(dst: str, write: Callable[[Any], None]) -> None:
            with _new_temp(dst, staged) as f:
                write(f)
        try:
            if plan:
                os.makedirs(data_dir, exist_ok=True)
            for src, dst in plan:
                stage(dst, lambda f, src=src: _copy_file(f, src))
            stage(target, lambda f: _write_json(f, data))
            unchanged()                 # nor while it was being copied
            for rec in discard:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(rec)
            for tmp, dst in staged:
                os.replace(tmp, dst)
        except BaseException:
            self._unstage(staged, data_dir if made_dir else '')
            raise
        return {'path': target, 'samples': len(data['samples']),
                'sidecars': [d for _s, d in plan
                             if not d.endswith('.json')]}

    @staticmethod
    def _unstage(staged: list[tuple[str, str]], made_dir: str) -> None:
        """Remove the temporary files in `staged` not yet moved, and the
        data folder `made_dir` the save made ('' for none) when empty."""
        for tmp, _dst in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if made_dir:
            try:
                os.rmdir(made_dir)
            except OSError:
                pass


# Points per edge when a gate's outline is followed through the display
# scales (a straight edge in stored coordinates bends on another scale).
_EDGE_POINTS = 16


def _walk(nodes: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for n in nodes:
        yield n
        yield from _walk(n.get('children') or ())


def _quad_sides(g: Mapping[str, Any], x: str, y: str) -> dict[str, int]:
    """A quadrant gate's ``quad_x`` and ``quad_y`` on a plot of channels
    `x` and `y` (either way round), or nothing when the plot is not on
    its two channels."""
    sides = {g.get('x_channel'): _quad_side(g, 'x'),
             g.get('y_channel'): _quad_side(g, 'y')}
    if x in sides and y in sides:
        return {'quad_x': sides[x], 'quad_y': sides[y]}
    return {}


def _quad_side(g: Mapping[str, Any], axis: str) -> int:
    """Which side of its quadrant's crossing gate `g` lies on along `axis`
    ('x' or 'y'): 1 above it, -1 below. Read from the region (its bounds
    against ``quad_origin_<axis>``, else which bound is open), never from
    the gate's name, which the user may change."""
    from .scales import OPEN
    lo, hi = sorted((float(g[f'{axis}0']), float(g[f'{axis}1'])))
    origin = g.get(f'quad_origin_{axis}')
    if origin is not None and math.isfinite(float(origin)):
        return 1 if (lo + hi) / 2 >= float(origin) else -1
    return -1 if lo <= -OPEN and hi < OPEN else 1


def _outline_corners(g: Mapping[str, Any], outline: Callable[[Any], Any]
                     ) -> np.ndarray | None:
    """Gate `g`'s closed outline in stored coordinates (last point = first):
    a float64 ``(n, 2)`` array, or None when it has none (a malformed
    gate). Open sides stay at the sentinel."""
    if g.get('kind') == 'rect':
        try:
            x0, x1 = float(g['x0']), float(g['x1'])
            y0, y1 = float(g['y0']), float(g['y1'])
        except (KeyError, TypeError, ValueError):
            return None
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    else:
        try:
            got = outline(g)
        except (KeyError, TypeError, ValueError):
            return None
        if got is None:
            return None
        corners = np.column_stack(got).astype(float)
    if len(corners) < 2 or not np.isfinite(corners).all():
        return None
    return corners


def _densified(corners: np.ndarray) -> np.ndarray:
    """A closed outline with each edge split in ``_EDGE_POINTS`` steps."""
    t = np.linspace(0.0, 1.0, _EDGE_POINTS, endpoint=False)[:, None]
    steps = [a + t * (b - a) for a, b in zip(corners[:-1], corners[1:], strict=True)]
    return np.vstack([*steps, corners[-1:]])


def _open_clipped(v: np.ndarray, data: np.ndarray) -> np.ndarray:
    """`v` with the open-side sentinel (``scales.OPEN``) brought in to just
    past the data's finite extent, so an open rectangle side is drawn at
    the plot's edge instead of at 1e12."""
    from .scales import OPEN
    fin = data[np.isfinite(data)]
    if not fin.size:
        return np.clip(v, -1.0, 1.0)
    lo, hi = float(fin.min()), float(fin.max())
    pad = max(hi - lo, 1.0)
    out = np.asarray(v, dtype=float).copy()
    out[out <= -OPEN] = lo - pad
    out[out >= OPEN] = hi + pad
    return out
