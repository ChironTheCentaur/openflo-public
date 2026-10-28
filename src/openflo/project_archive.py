"""One-file project archives: a session, its saved data and its FCS files.

A ``.flowsession`` is JSON beside a ``<stem>_data/`` folder, and points at
raw FCS files wherever they were on the machine that saved it. Sent to a
colleague, the FCS paths break (FlowJo workspaces have the same problem;
its answer is a zipped ``.acs``). A project archive is one zip:

    project.flowsession      the session, sample paths made relative
    project_data/...         its processed-data sidecars and records
    fcs/<file>.fcs           every sample's raw FCS file
    openflo-project.json     what the archive holds (and what it lacks)

`open_project_archive` unpacks one into a folder and returns the session to
open; the session's own relink (``rel_path``) finds the FCS files there.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile

ARCHIVE_EXT = '.zip'
MANIFEST = 'openflo-project.json'
SESSION_NAME = 'project.flowsession'
DATA_DIR = 'project_data'
FCS_DIR = 'fcs'


class ProjectArchiveError(ValueError):
    """The file is not an OpenFlo project archive, or not a safe one."""


def is_project_archive(path):
    """True when `path` is a zip holding an OpenFlo project manifest."""
    try:
        with zipfile.ZipFile(path) as zf:
            return MANIFEST in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _unique(name, taken):
    stem, ext = os.path.splitext(name)
    out, i = name, 2
    while out.lower() in taken:
        out = f"{stem} ({i}){ext}"
        i += 1
    taken.add(out.lower())
    return out


def write_project_archive(session_path, out_path):
    """Pack the session saved at `session_path` (with its ``<stem>_data``
    folder) and every sample's raw FCS file into the zip `out_path`.

    Sample paths are rewritten relative to the archive, so the session reads
    the same on any machine. Returns the manifest: ``samples`` packed,
    ``missing`` (samples whose FCS file was not found and is not in the
    archive) and ``bytes`` of FCS data."""
    with open(session_path, encoding='utf-8') as f:
        data = json.load(f)
    src_data = os.path.splitext(session_path)[0] + '_data'
    sess_dir = os.path.dirname(os.path.abspath(session_path))
    taken: set[str] = set()
    packed, missing, total = [], [], 0
    fcs_files = []
    for entry in data.get('samples', []):
        name = entry.get('name', '')
        src = entry.get('path') or ''
        if src and not os.path.isabs(src):
            src = os.path.join(sess_dir, src)
        processed = (entry.get('processed_data') or '').replace('\\', '/')
        if processed:
            # The sidecars move under DATA_DIR with the session's new name.
            entry['processed_data'] = (
                DATA_DIR + '/' + processed.split('/', 1)[-1])
        if not (src and os.path.isfile(src)):
            missing.append(name)
            continue
        arc = FCS_DIR + '/' + _unique(os.path.basename(src), taken)
        fcs_files.append((src, arc))
        entry['path'] = arc
        entry['rel_path'] = arc
        entry['basename'] = arc.split('/', 1)[1]
        packed.append(name)
        total += os.path.getsize(src)
    manifest = {'format': 'openflo-project', 'version': 1,
                'session': SESSION_NAME, 'samples': packed,
                'missing': missing, 'bytes': total}
    tmp = out_path + '.tmp'
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
            zf.writestr(SESSION_NAME, json.dumps(data, indent=2,
                                                 ensure_ascii=False))
            if os.path.isdir(src_data):
                for root, _dirs, files in os.walk(src_data):
                    for fn in sorted(files):
                        full = os.path.join(root, fn)
                        rel = os.path.relpath(full, src_data).replace(
                            os.sep, '/')
                        zf.write(full, DATA_DIR + '/' + rel)
            for src, arc in fcs_files:
                # FCS data barely compresses; storing it is much faster.
                zf.write(src, arc, compress_type=zipfile.ZIP_STORED)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return manifest


def _safe_member(name):
    """A member name that stays inside the folder it is unpacked into."""
    norm = name.replace('\\', '/')
    parts = [p for p in norm.split('/') if p not in ('', '.')]
    return not (norm.startswith('/') or ':' in norm or '..' in parts
                or not parts)


def unpack_folder_for(archive_path):
    """A new, empty folder beside `archive_path` named after it."""
    base = os.path.splitext(os.path.abspath(archive_path))[0]
    out, i = base, 2
    while os.path.exists(out):
        out = f"{base} ({i})"
        i += 1
    return out


def open_project_archive(archive_path, dest=None):
    """Unpack the project archive at `archive_path` into `dest` (a new
    folder beside it by default) and return ``(session_path, manifest)``.

    Refuses a zip without the manifest, or one with a member that would
    land outside `dest`, before writing anything."""
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if MANIFEST not in names:
            raise ProjectArchiveError(
                f"{os.path.basename(archive_path)} is a zip file but not an "
                "OpenFlo project archive.")
        bad = [n for n in names if not _safe_member(n)]
        if bad:
            raise ProjectArchiveError(
                f"{os.path.basename(archive_path)} holds a file that would "
                f"be written outside its folder ({bad[0]}), so it was not "
                "opened.")
        manifest = json.loads(zf.read(MANIFEST).decode('utf-8'))
        session = manifest.get('session') or SESSION_NAME
        if session not in names or not _safe_member(session):
            raise ProjectArchiveError(
                f"{os.path.basename(archive_path)} has no session file.")
        dest = dest or unpack_folder_for(archive_path)
        parent = os.path.dirname(os.path.abspath(dest)) or '.'
        os.makedirs(parent, exist_ok=True)
        work = tempfile.mkdtemp(prefix='.openflo-unpack-', dir=parent)
        try:
            zf.extractall(work)
            os.replace(work, dest)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    return os.path.join(dest, *session.split('/')), manifest
