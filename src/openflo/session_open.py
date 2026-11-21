"""Opening a saved OpenFlo session -- Tk-free.

A ``.flowsession`` file lists its samples, each with where its data was
(an absolute path, a path relative to the session file, the file name) and
what the session knew about it: gates, group, compensation, QC, transforms,
models. Opening one means working out, per sample, which file to read and
how to prepare it. ``plan_session`` does that without touching any window,
so the Tk gate editor and a headless document open a session the same way:

* the processed-data sidecar (``<session>_data/<name>.npz``, or the CSV an
  older session wrote) when there is one -- it carries clusters and
  embeddings the raw file does not -- else the raw FCS;
* the raw FCS is looked for at its saved absolute path, then at its saved
  relative path under the session folder, then by file name in the session
  folder, so a project moved to another computer still finds its data; the
  relative path and file name are read with either separator, so a session
  saved on Windows opens on Linux, macOS or a phone (``paths``);
* a sidecar pointer must stay inside the session folder once symbolic
  links are followed: a crafted session cannot make the sidecar reader
  open a file outside it (a raw FCS may be in a sibling folder, with a
  note in the log);
* a raw FCS still not found is looked for in the project the session came
  from, unpacked beside it (``project_copies.project_matches``), and taken
  from there when it is verified to be the same recording: a session
  copied next to its project opens with that project's samples, and never
  with another day's file of the same name;
* a sample whose file is not found is kept, with its gates, as an offline
  entry (``offline_entry``), so the next save writes it back unchanged.

The editor keeps what is a window's business: dialogs, the status line, the
load queue and the session lock.

``rescaled_gates`` puts a restored sample's saved gates onto the data it
reopened on, when the reopen put a channel on other units.

Core module: no Tk.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from . import result_stamp as _stamps
from . import session_models as _models
from .downsample_trim import trim_record_from_entry
from .fcs_quirks import LMD_COPIES
from .gate_store import Gate
from .paths import (
    is_fcs_container,
    is_fcs_file,
    is_foreign_absolute,
    legacy_sidecar_name,
    portable_join,
    safe_sidecar_name,
    sample_key,
    saved_file_name,
    session_relative,
)
from .session_format import comp_from_entry, migrate_session
from .session_sidecar import SIDECAR_EXT

log = logging.getLogger(__name__)

_Entry = Mapping[str, Any]

# The overlay colour a session saves for a sample never displayed: not a
# colour the user chose, so it is not restored.
_PLACEHOLDER_COLOR = '#1f77b4'


@dataclass(frozen=True)
class PlannedSample:
    """How to reopen one sample entry of a session.

    ``name`` is '' for an entry that names no sample: it is still queued
    from its raw file when that is found, but nothing is staged for it.
    ``processed_path`` is its sidecar ('' when none), ``raw_path`` where its
    raw FCS was looked for ('' when nowhere; it may not exist, see
    ``raw_found``). ``meta`` is what the session restores onto the sample
    once it has loaded (gates, group, records) and ``prep`` how its file is
    prepared (``sample_prep.prepare_fcs_sample``'s staging record:
    ``comp``, ``source_fcs``, ``qc``, ``qc_removed``, ``lmd_data_set``,
    ``instrument_compensated``, ``set_number``); both are None for an
    entry not staged. ``fmo_of`` and ``color`` are its FMO-control channels
    and the overlay colour to give it (None: none saved, or the
    placeholder). ``found_in`` is the unpacked project its raw file was
    found in when it was not where the session saved it ('' otherwise;
    ``plan_samples``)."""

    entry: _Entry
    name: str
    gates: list[Gate]
    processed_path: str
    raw_path: str
    raw_found: bool
    load_found: bool
    meta: dict[str, Any] | None
    prep: dict[str, Any] | None
    fmo_of: list[str] | None
    color: str | None
    found_in: str = ''

    @property
    def load_path(self) -> str:
        """The file the sample loads from: its sidecar, else its raw FCS."""
        return self.processed_path or self.raw_path

    @property
    def set_number(self) -> int | None:
        """Which data set of its raw FCS the sample is (a file holding
        several, or a well of a BD Accuri file), or None: read the file as
        one sample."""
        return planned_set_number(self.entry, self.raw_path)

    @property
    def key(self) -> str:
        """``load_path`` as the loader keys staged records
        (``paths.sample_key``: normcased, absolute, and the data set of a
        raw FCS holding several), or '' when there is none."""
        p = self.load_path
        return sample_key(p, self.set_number) if p else ''

    @property
    def staged(self) -> bool:
        """Whether a meta record and a staging record were made for it."""
        return self.meta is not None

    @property
    def label(self) -> str:
        """The name to report it under: its name, else its file's."""
        e = self.entry
        return e.get('name') or saved_file_name(e) or '?'


@dataclass(frozen=True)
class SessionPlan:
    """A session, upgraded to the current format, and its samples' plans
    in the session's order. ``notes`` are the format upgrades applied."""

    data: dict[str, Any]
    notes: list[str]
    samples: list[PlannedSample]

    @property
    def active(self) -> str | None:
        """The sample the session was saved with active."""
        return self.data.get('active_sample')


def plan_session(data: dict[str, Any], session_dir: str, *,
                 data_dir: str = '') -> SessionPlan:
    """The plan for opening the parsed session `data` saved in
    `session_dir`.

    `data` is upgraded in place to the current format first
    (``session_format.migrate_session``, which raises
    ``SessionVersionError`` for one written by a newer OpenFlo).
    `data_dir` is the session's conventional sidecar folder
    (``<session file stem>_data``), searched when a sample's sidecar
    pointer is missing or stale."""
    data, notes = migrate_session(data)
    return SessionPlan(data, notes, plan_samples(
        data.get('samples', []), data.get('sample_gates') or {},
        session_dir, data_dir=data_dir))


def plan_samples(entries: Iterable[_Entry],
                 sample_gates: Mapping[Any, list[Gate]], session_dir: str,
                 *, data_dir: str = '',
                 find_projects: bool = True) -> list[PlannedSample]:
    """``plan_sample`` for each of a session's sample `entries`, in order.

    With `find_projects`, a sample whose raw file was not found is looked
    for in the project it came from, unpacked beside the session
    (``project_copies``: a project folder there, or a copy of a project
    archive there). One found there AND verified to be the recording the
    session had (``project_copies.project_matches``) is planned from that
    file instead: its entry's ``path`` that file, its old ``rel_path``
    dropped (the next save links it afresh) and ``found_in`` the project
    folder, for the caller to say so. A match not verified (a file of the
    same name, maybe another day's) is left missing, for the caller to
    offer."""
    plan = [plan_sample(e, sample_gates, session_dir, data_dir=data_dir)
            for e in entries]
    lost = [p for p in plan if not p.raw_found and p.name]
    if not (find_projects and lost and session_dir):
        return plan
    from .project_copies import (
        project_folders,
        project_matches,
        verified_files,
    )
    found = verified_files(project_matches(
        [p.entry for p in lost], project_folders(session_dir)))
    if not found:
        return plan
    out = []
    for p in plan:
        m = found.get(p.name) if not p.raw_found else None
        if m is None:
            out.append(p)
            continue
        e = dict(p.entry, path=m.path)
        e.pop('rel_path', None)
        log.info("[session] %s: raw file found in project folder %s",
                 p.name, m.folder)
        out.append(replace(plan_sample(e, sample_gates, session_dir,
                                       data_dir=data_dir),
                           found_in=m.folder))
    return out


def plan_sample(entry: _Entry, sample_gates: Mapping[Any, list[Gate]],
                session_dir: str, *, data_dir: str = '') -> PlannedSample:
    """How to reopen the saved sample `entry` (see ``PlannedSample``);
    `sample_gates` is the session's ``{sample name: [gate, ...]}``."""
    nm = entry.get('name') or ''
    gates = list(sample_gates.get(entry.get('name'), []))
    pcsv = resolve_processed_path(entry, nm, session_dir,
                                  data_dir) if nm else ''
    raw = resolve_sample_path(entry, session_dir)
    raw_found = bool(raw) and os.path.isfile(raw)
    load_path = pcsv or raw
    meta: dict[str, Any] | None = None
    prep: dict[str, Any] | None = None
    fmo_of: list[str] | None = None
    color: str | None = None
    if nm and load_path:
        meta = {'gates': list(gates)}
        if entry.get('trial'):
            meta['trial'] = entry['trial']
        if 'is_comp' in entry:
            meta['is_comp'] = bool(entry['is_comp'])
        meta.update(analysis_records(entry, nm))
        meta.update(result_records(entry))
        # A session written before 'qc_applied' existed was loaded with QC
        # (it then always ran), so that is what a raw reload repeats. The
        # matrix is the one the sample was compensated with: a raw-FCS
        # reload applies it instead of $SPILL; a processed CSV, already
        # compensated, gets it attached, and is told its raw FCS.
        fcs = raw if pcsv else ''
        prep = {'comp': comp_from_entry(entry.get('comp')),
                'source_fcs': (fcs if fcs and is_fcs_file(fcs)
                               else None),
                'qc': bool(entry.get('qc_applied', True)),
                # Which .lmd copy the saved values are, and whether the
                # instrument compensated them (an older session has
                # neither: the .lmd file is read again to find out).
                'lmd_data_set': (entry['lmd_data_set']
                                 if entry.get('lmd_data_set') in LMD_COPIES
                                 else None),
                'instrument_compensated': bool(
                    entry.get('instrument_compensated')),
                # Which data set of a file holding several it is.
                'set_number': planned_set_number(entry, raw),
                'qc_removed': (int(entry['qc_removed'])
                               if isinstance(entry.get('qc_removed'),
                                             (int, float)) else None)}
        if isinstance(entry.get('fmo_of'), list):
            fmo_of = [str(ch) for ch in entry['fmo_of']]
        color = _saved_color(entry)
    return PlannedSample(
        entry=entry, name=nm, gates=gates, processed_path=pcsv,
        raw_path=raw, raw_found=raw_found,
        load_found=bool(load_path) and os.path.isfile(load_path),
        meta=meta, prep=prep, fmo_of=fmo_of, color=color)


def saved_set_number(entry: _Entry) -> int | None:
    """The data set number (from 1) a session `entry` saved for a sample
    read from one data set of a file holding several, or None (an entry
    without one, or with one that is not a whole number above 0)."""
    n = entry.get('set_number')
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        return None
    return n


def planned_set_number(entry: _Entry, raw_path: str) -> int | None:
    """``saved_set_number``, checked against the BD Accuri file at
    `raw_path` when the entry also saved its member's name
    (``container_member``): the number that member has in the file now
    (``fcs_containers.resolve_member``), so a session reopens the well it
    saved even when the file gained wells since."""
    n = saved_set_number(entry)
    member = entry.get('container_member')
    if (n and isinstance(member, str) and member and raw_path
            and is_fcs_container(raw_path) and os.path.isfile(raw_path)):
        from .fcs_containers import resolve_member
        return resolve_member(raw_path, n, member)
    return n


def _saved_color(entry: _Entry) -> str | None:
    """The overlay colour saved for a sample, or None for none. A sample
    never displayed is saved with the placeholder and keeps getting its
    colour when it is first shown."""
    color = entry.get('color')
    if (isinstance(color, str) and color
            and (entry.get('plot_enabled') or color != _PLACEHOLDER_COLOR)):
        return color
    return None


def raw_loads(plan: Iterable[PlannedSample], processed: Iterable[str] = ()
              ) -> tuple[list[tuple[str, int | None]], list[PlannedSample]]:
    """``(raw FCS loads, samples not found)`` of a session's planned
    samples, leaving out those named in `processed` (they load from their
    sidecars). Each load is ``(path, set number)``: the data set of a file
    holding several, or None to read the file as one sample. A sample not
    found is to be kept offline (``offline_entry``)."""
    skip = set(processed)
    paths: list[tuple[str, int | None]] = []
    missing: list[PlannedSample] = []
    for p in plan:
        if p.entry.get('name') in skip:
            continue
        if p.raw_found:
            paths.append((p.raw_path, p.set_number))
        else:
            missing.append(p)
    return paths, missing


def offline_entry(entry: _Entry, session_dir: str) -> dict[str, Any]:
    """The session entry to keep for a sample whose file was not found, so
    that a save writes the sample back. Its sidecar pointer is dropped (it
    is relative to the session it came from, and was not found either); a
    relative raw path is made absolute against `session_dir`, but not a
    path absolute on another OS (``C:\\...`` read on Linux), nor a
    drive-relative one (``E:x.fcs``) that names no file there
    (``paths.session_relative``)."""
    e = dict(entry)
    for key in ('processed_data', 'processed_csv'):
        e.pop(key, None)
    if e.get('path'):
        e['path'] = session_relative(e['path'], session_dir)
    return e


def _under(root: str, path: str) -> bool:
    """Whether `path` is `root` or inside it once symbolic links are
    followed (a link in the folder pointing out of it is outside)."""
    top = os.path.realpath(root)
    real = os.path.realpath(path)
    return real == top or real.startswith(top.rstrip(os.sep) + os.sep)


def sidecar_name_candidates(name: str, data_dir: str) -> list[str]:
    """The files sample `name`'s sidecar is looked for under in `data_dir`
    when its pointer is missing or stale (``_sidecar_by_name``), in the
    order they are tried: the current stem first, then the
    plain-substitution one sidecars had before their names carried a
    digest, each as ``.npz`` and then ``.csv``. A writer that leaves a
    sample with no pointer removes these (and their transform records), so
    the reopen cannot find an earlier save's file."""
    out: list[str] = []
    for stem in (safe_sidecar_name(name), legacy_sidecar_name(name)):
        for ext in (SIDECAR_EXT, '.csv'):
            guess = os.path.join(data_dir, stem + ext)
            if guess not in out:
                out.append(guess)
    return out


def _sidecar_by_name(name: str, data_dir: str) -> str:
    """Sample `name`'s sidecar under its usual name in `data_dir`
    (``sidecar_name_candidates``; two names could map to one file under
    the legacy stem), or ''. A file there that is a symbolic link out of
    the folder is not read (the folder itself may be a link: it is the
    real folder the file must be in)."""
    for guess in sidecar_name_candidates(name, data_dir):
        if os.path.isfile(guess):
            if not _under(data_dir, guess):
                log.warning("[session] ignoring out-of-tree sidecar %r: "
                            "a symbolic link out of %s", guess, data_dir)
                continue
            return guess
    return ''


def _pointer_refusal(session_dir: str, pointer: str, cand: str) -> str:
    """Why the sidecar `pointer` (resolved to `cand`) is refused, or ''
    when it stays inside `session_dir`."""
    if is_foreign_absolute(pointer):
        return 'is an absolute path'
    top = os.path.abspath(session_dir)
    here = os.path.abspath(cand)
    if not (here == top or here.startswith(top.rstrip(os.sep) + os.sep)):
        return 'leads outside the session folder'
    if not _under(session_dir, cand):
        return ('leads outside the session folder through a symbolic '
                'link')
    return ''


def resolve_processed_path(entry: _Entry, name: str, session_dir: str,
                           data_dir: str = '') -> str:
    """Absolute path of sample `name`'s processed-data sidecar (binary
    ``.npz``, or the CSV older sessions wrote), or '' if none exists.

    The session's recorded pointer (``processed_data``, or an older
    session's ``processed_csv``, with either separator) is resolved against
    `session_dir` and must stay inside it, symbolic links followed; when it
    is missing or stale the conventional ``<data_dir>/<safe name>.npz`` /
    ``.csv`` is looked for, so computed columns (clusters, UMAP) are still
    recovered when the pointer was dropped (for example by a racing exit
    autosave whose sample had already reloaded the raw FCS). That file must
    be inside `data_dir` once links are followed (`data_dir` itself may be
    a link, ``_sidecar_by_name``).

    A refused pointer is noted in the log, with why (absolute, outside the
    folder by its own ``..`` steps, or through a symbolic link): as ignored
    when nothing is found, or, when the sidecar is then found by its usual
    name (the ``<stem>_data`` folder itself a symbolic link out of the
    session folder, say), as read from there instead."""
    refused, found, by_name = [], '', False
    for key in ('processed_data', 'processed_csv'):
        pcsv = entry.get(key) or ''
        if not pcsv:
            continue
        # Either separator, as rel_path; with no session folder, read
        # against the working folder.
        cand = portable_join(session_dir, pcsv)
        why = _pointer_refusal(session_dir, pcsv, cand) if session_dir else ''
        if why:
            # The pointer is always app-generated relative to the session
            # folder; an absolute (on any OS) or escaping one is refused.
            refused.append((key, pcsv, why))
        elif os.path.isfile(cand):
            found = cand
            break
    if not found and data_dir:
        found = _sidecar_by_name(name, data_dir)
        by_name = bool(found)
    for key, pcsv, why in refused:
        if by_name:
            log.warning("[session] %s %r %s; read the sidecar by its usual "
                        "name instead: %s", key, pcsv, why, found)
        else:
            log.warning("[session] ignoring out-of-tree %s: %r (%s)", key,
                        pcsv, why)
    return found


def resolve_sample_path(entry: _Entry, session_dir: str, *,
                        path_in_session: bool = False) -> str:
    """The raw FCS path for a session sample `entry`, relinking a moved or
    copied project: the saved ``path``, then ``rel_path`` under
    `session_dir`, then ``basename`` in `session_dir`; the first that
    exists. A relative saved ``path`` is read as it is (against the working
    folder, as the editor loaded it), or, with `path_in_session`, against
    `session_dir` (``session_relative``; the project archive). The relative
    path and the file name are read with either separator, whatever OS
    saved them (a Windows path's file name is found on Linux); names are
    matched as the file system does (case-sensitive on Linux). When none is
    found, the saved path ('' when there is none)."""
    p = entry.get('path') or ''
    if path_in_session:
        p = session_relative(p, session_dir)
    if p and os.path.isfile(p):
        return p
    if session_dir:
        # Forward slashes are written; a Windows-written session's
        # backslashes are read too.
        rel = entry.get('rel_path') or ''
        if rel:
            cand = portable_join(session_dir, rel)
            if not _under(session_dir, cand):
                # Raw files kept in a sibling folder are a real project
                # layout, so this is allowed; but a session reaching
                # outside its own tree could name any local file: noted.
                log.warning("[session] note: rel_path resolves outside the "
                            "session dir: %s", cand)
            if os.path.isfile(cand):
                return cand
        # Older saves on Linux/macOS wrote a Windows path whole here.
        base = saved_file_name(entry)
        if base:
            cand = os.path.join(session_dir, base)
            if os.path.isfile(cand):
                return cand
    return p


def analysis_records(entry: _Entry, name: str) -> dict[str, Any]:
    """Sample `name`'s saved analysis records from its entry: transforms,
    derived channels, stat axes, normalisations, models. An unreadable one
    is logged and left out."""
    m: dict[str, Any] = {}
    if 'transforms' in entry:
        from .transform_sidecar import transform_record_from_json
        try:
            m['transforms'] = transform_record_from_json(entry['transforms'])
        except ValueError as exc:
            log.warning("[session] %s: transform record ignored (%s)",
                        name, exc)
    if entry.get('derived'):
        m['derived'] = dict(entry['derived'])
    if 'stat_axes' in entry:
        from .wsp_gate_coords import stat_axes_from_json
        axes = stat_axes_from_json(entry['stat_axes'])
        if axes:
            m['stat_axes'] = axes
        elif entry['stat_axes']:
            log.warning("[session] %s: stat_axes ignored (not understood)",
                        name)
    if isinstance(entry.get('bead_norm'), dict):
        m['bead_norm'] = dict(entry['bead_norm'])
    # The cell-cycle / FlowSOM models and a spectral unmix: staged as read,
    # checked against the reopened sample's columns once it has loaded.
    for key in (_models.CELL_CYCLE_KEY, _models.FLOWSOM_KEY,
                _models.UNMIX_KEY):
        if isinstance(entry.get(key), dict):
            m[key] = entry[key]
        elif key in entry:
            log.warning("[session] %s: %s ignored (not a record)", name, key)
    if isinstance(entry.get('batch_norm'), dict):
        m['batch_norm'] = dict(entry['batch_norm'])
    return m


def result_records(entry: _Entry) -> dict[str, Any]:
    """A saved sample entry's result stamps, kept originals, saved
    statistics and down-sample trim (each only when present and valid)."""
    m: dict[str, Any] = {}
    stamps = _stamps.stamps_from_entry(entry.get('result_stamps'))
    if stamps:
        m['result_stamps'] = stamps
    originals = _stamps.originals_from_entry(entry.get('result_originals'))
    if originals:
        m['result_originals'] = originals
    if isinstance(entry.get('saved_stats'), dict):
        m['saved_stats'] = entry['saved_stats']
    trim = trim_record_from_entry(entry.get('downsample_trim'))
    if trim is not None:
        m['downsample_trim'] = trim
    return m


def rescaled_gates(gates: list[Gate] | None,
                   rescale: Mapping[str, Any] | None) -> list[Gate] | None:
    """A restored sample's saved `gates` on the units its data reopened on.

    `rescale` maps a channel to the function taking its saved units to the
    current ones (``wsp_gate_coords.gate_rescale_fns``); empty or None
    leaves the gates as saved. A change of units is not a hand edit, so a
    stamped gate keeps its stamp, resealed on the new coordinates. None
    when there are no gates."""
    pending = gates or None
    if pending and rescale:
        from .result_stamp import carry_gate_stamp
        from .wsp_gate_coords import map_gate_coords
        pending = [carry_gate_stamp(g, map_gate_coords(g, rescale))
                   for g in pending]
    return pending
