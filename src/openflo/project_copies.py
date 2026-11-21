"""Unpacked copies of project archives: reusing one, where a new one goes,
and finding a session's samples in one. Core module: no Tk.

Opening a ``.flowproject`` unpacks it into a folder (``project_archive.
open_project_archive``). Opening the same archive again used to unpack it
again, into ``<name> (2)``, ``(3)``…: minutes each time for a big project
on a USB stick. Here:

* ``find_copy`` finds a folder already unpacked from the archive (beside
  it, or in the internal project folder), so it opens at once. A folder
  is a copy of the archive when it holds the archive's manifest byte for
  byte, the session the manifest names and every FCS file the manifest
  lists, each of the size it lists. Whether the session there was saved
  since (``ArchiveCopy.changed``) is told apart, so the caller can ask.
* ``unpack_destination`` puts a new copy beside the archive, as before,
  unless the archive is on a removable drive (a USB stick or memory card:
  slow to write, and taken away) or its folder cannot be written: then in
  the internal project folder, ``~/.openflo/projects``.
* ``project_matches`` finds the raw files of a session's samples that were
  not found in an unpacked copy of the project they came from: a folder
  beside the session holding one, or a copy of an archive beside it. A
  sample is matched by its name in the project's own session and by its
  file name (the archive may have renamed it ``x (2).fcs``, when two
  samples had one file name). Instruments name files alike every day, so
  a match opens on its own only when it is verified to be the same
  recording (its saved ``fcs_id``, or, for an older session, the same data
  digest saved with its numbers in both sessions); any other is offered
  to the user, who picks the project.

Reuse trusts the manifest and file sizes, not the files' contents: a copy
whose FCS file was changed in place at the same size is still reused (the
manifest carries no hashes).
"""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .paths import (
    fcs_identity,
    portable_basename,
    same_fcs_identity,
    saved_file_name,
)
from .project_archive import (
    ARCHIVE_EXT,
    FCS_DIR,
    MANIFEST,
    MAX_MANIFEST_BYTES,
    SESSION_NAME,
    safe_member,
    unpack_folder_for,
)

# Copies looked for beside an archive: '<name>', '<name> (2)' … this.
MAX_COPIES = 50
# Folders and files looked at in a session's folder for projects (a USB
# stick's top folder can hold many): beyond this, the rest are not.
MAX_SCAN = 500
# A project's own session larger than this is not read for matching.
MAX_SESSION_BYTES = 64 << 20


def store_dir() -> str:
    """The internal folder unpacked projects go to when they cannot go
    beside their archive (``unpack_destination``)."""
    return os.path.join(os.path.expanduser('~'), '.openflo', 'projects')


def on_removable_drive(path: str) -> bool:
    """Whether `path` is on a removable drive: on Windows a drive Windows
    reports as removable (USB sticks, memory cards); on macOS a volume
    under /Volumes; on Linux one mounted under /media or /run/media. A
    network path, or one that cannot be told, is not."""
    p = os.path.abspath(path)
    if sys.platform == 'win32':
        drive = os.path.splitdrive(p)[0]
        if not drive or drive.startswith(('\\\\', '//')):
            return False
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            return int(kernel32.GetDriveTypeW(drive + '\\')) == 2
        except (AttributeError, OSError):
            return False
    if sys.platform == 'darwin':
        return p.startswith('/Volumes/')
    return p.startswith(('/media/', '/run/media/'))


def unpack_destination(archive_path: str) -> str:
    """A new, empty folder to unpack the archive at `archive_path` into:
    beside it (``project_archive.unpack_folder_for``), or, when it is on a
    removable drive or its folder cannot be written, in ``store_dir``."""
    here = os.path.dirname(os.path.abspath(archive_path))
    if on_removable_drive(archive_path) or not os.access(here, os.W_OK):
        return unpack_folder_for(
            os.path.join(store_dir(), os.path.basename(archive_path)))
    return unpack_folder_for(archive_path)


def copy_folders(archive_path: str) -> list[str]:
    """The folders that may hold a copy of the archive at `archive_path`,
    in the order they are tried: its name and numbered names
    (``unpack_folder_for``) beside it, then in ``store_dir``. Only those
    that exist."""
    name = os.path.splitext(os.path.basename(archive_path))[0]
    out = []
    for parent in (os.path.dirname(os.path.abspath(archive_path)),
                   store_dir()):
        for i in range(1, MAX_COPIES + 1):
            cand = os.path.join(parent, name if i == 1 else f"{name} ({i})")
            if os.path.isdir(cand) and cand not in out:
                out.append(cand)
    return out


@dataclass(frozen=True)
class ArchiveCopy:
    """A folder unpacked from an archive: its ``session`` file, and
    whether that session differs from the archive's (``changed``: saved
    there since it was unpacked)."""

    folder: str
    session: str
    changed: bool


def _read_bounded(path: str, limit: int) -> bytes | None:
    try:
        if os.path.getsize(path) > limit:
            return None
        with open(path, 'rb') as fh:
            return fh.read()
    except OSError:
        return None


def _archive_identity(archive_path: str) -> tuple[bytes, dict, str, bytes] | None:
    """``(manifest bytes, manifest, session member name, session bytes)``
    of the archive, or None when it cannot be read as a project."""
    try:
        with zipfile.ZipFile(archive_path) as zf:
            info = zf.getinfo(MANIFEST)
            if info.file_size > MAX_MANIFEST_BYTES:
                return None
            raw = zf.read(MANIFEST)
            manifest = json.loads(raw.decode('utf-8'))
            if not isinstance(manifest, dict):
                return None
            session = manifest.get('session') or SESSION_NAME
            if (not isinstance(session, str) or '/' in session
                    or not safe_member(session)):
                return None
            sinfo = zf.getinfo(session)
            if sinfo.file_size > MAX_SESSION_BYTES:
                return None
            return raw, manifest, session, zf.read(session)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile,
            UnicodeDecodeError):
        return None


def _holds_files(folder: str, manifest: Mapping[str, Any]) -> bool:
    """Whether every FCS file `manifest` lists is in `folder` at the size
    it lists."""
    files = manifest.get('files')
    if not isinstance(files, list):
        return False
    for f in files:
        if not isinstance(f, dict):
            return False
        path, size = f.get('path'), f.get('bytes')
        if (not isinstance(path, str) or not path.startswith(FCS_DIR + '/')
                or not isinstance(size, int) or isinstance(size, bool)):
            return False
        full = os.path.join(folder, *path.split('/'))
        try:
            if os.path.getsize(full) != size:
                return False
        except OSError:
            return False
    return True


def find_copy(archive_path: str) -> ArchiveCopy | None:
    """A folder already unpacked from the archive at `archive_path` (see
    the module docstring), or None. One whose session is as in the archive
    is preferred over one saved since; the first of each in
    ``copy_folders`` order."""
    ident = _archive_identity(archive_path)
    if ident is None:
        return None
    raw, manifest, session, sess_bytes = ident
    changed = None
    for folder in copy_folders(archive_path):
        if _read_bounded(os.path.join(folder, MANIFEST),
                         MAX_MANIFEST_BYTES) != raw:
            continue
        sess = os.path.join(folder, session)
        if not os.path.isfile(sess) or not _holds_files(folder, manifest):
            continue
        if _read_bounded(sess, MAX_SESSION_BYTES) == sess_bytes:
            return ArchiveCopy(folder, sess, False)
        if changed is None:
            changed = ArchiveCopy(folder, sess, True)
    return changed


# ── a session's missing samples, in the project it came from ──────────────


def _is_project_folder(folder: str) -> bool:
    return (os.path.isfile(os.path.join(folder, MANIFEST))
            and os.path.isfile(os.path.join(folder, SESSION_NAME)))


def project_folders(session_dir: str) -> list[str]:
    """Unpacked projects a session in `session_dir` may have come from:
    folders in it holding a project (manifest and session), and copies
    (``find_copy``) of the project archives in it; each once, in name
    order. At most ``MAX_SCAN`` entries of `session_dir` are looked at."""
    out: list[str] = []
    archives: list[str] = []
    try:
        with os.scandir(session_dir) as it:
            entries = []
            for i, e in enumerate(it):
                if i >= MAX_SCAN:
                    break
                entries.append(e)
    except OSError:
        return out
    for e in sorted(entries, key=lambda e: e.name.lower()):
        try:
            if e.is_dir() and not e.name.startswith('.'):
                if _is_project_folder(e.path):
                    out.append(e.path)
            elif e.is_file() and e.name.lower().endswith(ARCHIVE_EXT):
                archives.append(e.path)
        except OSError:
            continue
    for a in archives:
        c = find_copy(a)
        if c is not None and c.folder not in out:
            out.append(c.folder)
    return out


def project_archives(session_dir: str, names: Iterable[str]) -> list[str]:
    """The project archives in `session_dir` whose manifest lists any of
    the sample `names`, most of them first (then by name)."""
    want = set(names)
    found: list[tuple[int, str]] = []
    try:
        with os.scandir(session_dir) as it:
            files = [e.path for i, e in enumerate(it)
                     if i < MAX_SCAN and e.name.lower().endswith(ARCHIVE_EXT)]
    except OSError:
        return []
    for a in sorted(files, key=str.lower):
        ident = _archive_identity(a)
        if ident is None:
            continue
        listed = ident[1].get('samples')
        if not isinstance(listed, list):
            continue
        n = len(want & {str(s) for s in listed})
        if n:
            found.append((-n, a))
    return [a for _n, a in sorted(found, key=lambda t: t[0])]


_NUMBERED = re.compile(r'^(?P<stem>.*) \((?P<n>\d+)\)$')


def same_file_name(saved: str, packed: str) -> bool:
    """Whether the file name `packed` in a project is the file `saved`
    (case aside): the same, or the name the archive gave a second file of
    one name (``x (2).fcs`` for ``x.fcs``)."""
    a, b = saved.lower(), packed.lower()
    if a == b:
        return True
    sa, ea = os.path.splitext(a)
    sb, eb = os.path.splitext(b)
    m = _NUMBERED.match(sb)
    return ea == eb and m is not None and m.group('stem') == sa


def _project_samples(folder: str) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """``{sample name: (its raw file in the project, its entry)}`` of the
    unpacked project in `folder`, for the files that are there."""
    raw = _read_bounded(os.path.join(folder, SESSION_NAME), MAX_SESSION_BYTES)
    if raw is None:
        return {}
    try:
        data = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError, RecursionError):
        return {}
    out: dict[str, tuple[str, Mapping[str, Any]]] = {}
    samples = data.get('samples') if isinstance(data, dict) else None
    for e in samples if isinstance(samples, list) else []:
        if not isinstance(e, dict) or not isinstance(e.get('name'), str):
            continue
        rel = e.get('rel_path') or e.get('path') or ''
        if not isinstance(rel, str) or not rel.startswith(FCS_DIR + '/'):
            continue
        parts = rel.split('/')
        if len(parts) != 2 or parts[1] in ('', '.', '..'):
            continue
        full = os.path.join(folder, FCS_DIR, parts[1])
        if os.path.isfile(full):
            out[e['name']] = (full, e)
    return out


def _same_data(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two session entries saved numbers (``saved_stats``) for the
    same data: equal, non-empty data digests. The digest is of the
    sample's events as loaded, so two recordings never share one."""
    da, db = a.get('saved_stats'), b.get('saved_stats')
    if not (isinstance(da, Mapping) and isinstance(db, Mapping)):
        return False
    d = da.get('data')
    return isinstance(d, str) and bool(d) and d == db.get('data')


@dataclass(frozen=True)
class ProjectMatch:
    """A missing sample's raw file found in an unpacked project: ``path``
    in project ``folder``; ``verified`` when it is shown to be the very
    recording the session had (``project_matches``)."""

    path: str
    folder: str
    verified: bool


def project_matches(entries: Iterable[Mapping[str, Any]],
                    folders: Iterable[str]
                    ) -> dict[str, list[ProjectMatch]]:
    """``{sample name: [ProjectMatch, ...]}`` for the session sample
    `entries` found in the unpacked projects `folders`, in folder order:
    a sample of the same name there whose file name is the entry's
    (``same_file_name``) and whose data set number (a file holding
    several) is the same. Instruments name files alike every day, so that
    alone does not show it is the same recording. A match is ``verified``
    when the entry saved its file's identity (``fcs_id``) and the file
    there has it, or, for a session saved before identities were, when the
    project's own entry for the sample saved numbers for the same data
    (``_same_data``: the data digest of ``saved_stats``). A file whose
    saved identity differs is not a match at all. Two samples
    may take one file: two data sets of it."""
    want = {e['name']: e for e in entries
            if isinstance(e.get('name'), str) and e['name']}
    out: dict[str, list[ProjectMatch]] = {}
    if not want:
        return out
    for folder in folders:
        for name, (path, pe) in _project_samples(folder).items():
            e = want.get(name)
            if e is None:
                continue
            saved = saved_file_name(e)
            if not saved or not same_file_name(saved,
                                               portable_basename(path)):
                continue
            if e.get('set_number') != pe.get('set_number'):
                continue
            if e.get('fcs_id') is not None:
                if not same_fcs_identity(e['fcs_id'], fcs_identity(path)):
                    continue
                verified = True
            else:
                verified = _same_data(e, pe)
            out.setdefault(name, []).append(
                ProjectMatch(path, folder, verified))
    return out


def verified_files(matches: Mapping[str, list[ProjectMatch]]
                   ) -> dict[str, ProjectMatch]:
    """``{sample name: match}`` of the samples with a verified match: the
    first, in folder order (any verified one is the same recording)."""
    out: dict[str, ProjectMatch] = {}
    for name, ms in matches.items():
        for m in ms:
            if m.verified:
                out[name] = m
                break
    return out


def folder_files(matches: Mapping[str, list[ProjectMatch]], folder: str
                 ) -> dict[str, str]:
    """``{sample name: file}`` of the matches in project `folder`,
    verified or not (the user chose that project)."""
    out: dict[str, str] = {}
    for name, ms in matches.items():
        for m in ms:
            if m.folder == folder:
                out[name] = m.path
                break
    return out


def candidate_folders(matches: Mapping[str, list[ProjectMatch]]
                      ) -> list[tuple[str, int]]:
    """``[(project folder, samples it holds), ...]`` of the folders holding
    any match, most samples first; folders holding as many keep the order
    they were found in (the user picks between them, nothing here does)."""
    count: dict[str, int] = {}
    for ms in matches.values():
        for f in dict.fromkeys(m.folder for m in ms):
            count[f] = count.get(f, 0) + 1
    return sorted(count.items(), key=lambda t: -t[1])
