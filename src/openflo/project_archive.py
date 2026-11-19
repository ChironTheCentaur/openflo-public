"""One-file project archives: a session, its saved data and its FCS files.

A ``.flowsession`` is JSON beside a ``<stem>_data/`` folder, and points at
raw FCS files wherever they were on the machine that saved it. Sent to a
colleague, the FCS paths break (FlowJo workspaces have the same problem;
its answer is a zipped ``.acs``). A project archive is one zip, saved as
``.flowproject`` (older ones as ``.zip``), the manifest first:

    project.flowsession      the session, sample paths made relative
    project_data/...         its processed-data sidecars and records
    fcs/<file>.fcs.ofz       every sample's raw FCS file, packed
                             (``fcs_pack``: unpacks byte for byte) where that
                             saves 10%, else as it is (fcs/<file>.fcs)
    openflo-project.json     what the archive holds (and what it lacks):
                             each FCS file's member, size and codec

`open_project_archive` checks one (``check_project_archive``: only these
parts, safe names, bounded sizes) and unpacks it into a folder, returning
the session to open; the session's own relink (``rel_path``) finds the FCS
files there. The phone app unpacks the same way, into its memory
(``Document.unpack_project``).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile
from typing import NoReturn

from .fcs_pack import CODEC as FCS_CODEC
from .fcs_pack import SUFFIX as FCS_SUFFIX

# Its own ending, so that phones offer OpenFlo to open it (and not to
# unzip it); older archives were saved as .zip, and still open.
ARCHIVE_EXT = '.flowproject'
ARCHIVE_EXTS = (ARCHIVE_EXT, '.zip')
ARCHIVE_VERSION = 1
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


# An FCS file goes in packed (``fcs_pack``: its events byte-shuffled, then
# deflated) when that makes it at most this share of its size; else as it
# is. Packing runs on this many threads at once (zlib lets go of the GIL).
KEEP_RATIO = 0.9
PACK_THREADS = 4


def _pack_all(fcs_files, work):
    """Pack each ``(src, arc)`` FCS file into folder `work`; for each, the
    packed copy's path, or None where packing does not pay."""
    from concurrent.futures import ThreadPoolExecutor

    from .fcs_pack import pack_fcs

    def one(i_src):
        i, src = i_src
        out = os.path.join(work, f'{i}{FCS_SUFFIX}')
        with open(out, 'wb') as fh:
            pack_fcs(src, fh)
        if os.path.getsize(out) <= KEEP_RATIO * os.path.getsize(src):
            return out
        os.remove(out)
        return None
    jobs = list(enumerate(src for src, _ in fcs_files))
    if sys.platform == 'emscripten' or len(jobs) < 2:
        return [one(j) for j in jobs]               # (no threads there)
    with ThreadPoolExecutor(PACK_THREADS) as pool:
        return list(pool.map(one, jobs))


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
    archive), ``bytes`` of FCS data and ``files``: for each FCS file, the
    ``path`` it unpacks to, the zip ``member`` holding it, its ``bytes``
    and ``codec`` (``fcs_pack.CODEC``, member ``<path>.ofz``, or 'stored',
    the file as it is)."""
    with open(session_path, encoding='utf-8') as f:
        data = json.load(f)
    src_data = os.path.splitext(session_path)[0] + '_data'
    from .session_open import resolve_sample_path
    sess_dir = os.path.dirname(os.path.abspath(session_path))
    taken: set[str] = set()
    packed, missing, total = [], [], 0
    fcs_files = []
    for entry in data.get('samples', []):
        name = entry.get('name', '')
        # Its saved path (a relative one under the session folder), then
        # its link, then its file name: moved, or saved on another OS (a
        # Windows path read on Linux), it is found as opening would.
        src = resolve_sample_path(entry, sess_dir, path_in_session=True)
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
    work = tempfile.mkdtemp(prefix='.openflo-pack-',
                            dir=os.path.dirname(os.path.abspath(out_path)))
    tmp = out_path + '.tmp'
    try:
        packs = _pack_all(fcs_files, work)
        files = [{'path': arc, 'member': arc + FCS_SUFFIX if pk else arc,
                  'bytes': os.path.getsize(src),
                  'codec': FCS_CODEC if pk else 'stored'}
                 for (src, arc), pk in zip(fcs_files, packs, strict=True)]
        manifest = {'format': 'openflo-project', 'version': ARCHIVE_VERSION,
                    'session': SESSION_NAME, 'samples': packed,
                    'missing': missing, 'bytes': total, 'files': files}
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
            zf.writestr(SESSION_NAME, json.dumps(data, indent=2,
                                                 ensure_ascii=False))
            if os.path.isdir(src_data):
                for root, _dirs, names in os.walk(src_data):
                    for fn in sorted(names):
                        full = os.path.join(root, fn)
                        rel = os.path.relpath(full, src_data).replace(
                            os.sep, '/')
                        zf.write(full, DATA_DIR + '/' + rel)
            for (src, _arc), pk, f in zip(fcs_files, packs, files,
                                          strict=True):
                # Packed or not, deflating it again gains nothing.
                zf.write(pk or src, f['member'],
                         compress_type=zipfile.ZIP_STORED)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


# What an archive may hold, and how much: anything else is not one this
# app wrote, and is refused before a byte is written.
MAX_MEMBERS = 10_000
MAX_MANIFEST_BYTES = 1 << 20
MAX_NAME_CHARS = 255
MAX_DEPTH = 8
# Room left on the disk beyond the unpacked files.
DISK_MARGIN = 64 << 20
CHUNK = 1 << 20


def _safe_member(name):
    """A member name that stays inside the folder it is unpacked into: no
    absolute path, drive, ``..``, backslash or control character, and not
    too long or too deep."""
    if not name or '\\' in name or ':' in name or name.startswith('/'):
        return False
    if any(ord(c) < 32 for c in name) or len(name) > 4 * MAX_NAME_CHARS:
        return False
    parts = [p for p in name.split('/') if p not in ('', '.')]
    return bool(parts) and '..' not in parts and len(parts) <= MAX_DEPTH \
        and all(len(p) <= MAX_NAME_CHARS for p in parts)


def _expected_member(name, session):
    """Whether `name` is one of the parts an OpenFlo project archive holds:
    its manifest, its session, its saved data, or an FCS file."""
    if name in (MANIFEST, session):
        return True
    if name.startswith(DATA_DIR + '/'):
        return True
    rest = name[len(FCS_DIR) + 1:] if name.startswith(FCS_DIR + '/') else ''
    return bool(rest) and '/' not in rest


def _fold(name):
    """`name` as a file system that ignores case (and Unicode forms) sees
    it: two members alike here would land on one file. Windows also drops
    trailing dots and spaces from each part."""
    import unicodedata
    folded = unicodedata.normalize('NFC', name).casefold()
    return '/'.join(p.rstrip('. ') for p in folded.split('/'))


def _refuse(archive_path, why) -> NoReturn:
    raise ProjectArchiveError(
        f"{os.path.basename(archive_path)} {why}, so it was not opened.")


def _read_manifest(zf, archive_path):
    info = zf.getinfo(MANIFEST)
    if info.file_size > MAX_MANIFEST_BYTES:
        _refuse(archive_path, "has a project description too large to be "
                "one OpenFlo wrote")
    try:
        manifest = json.loads(zf.read(MANIFEST).decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        _refuse(archive_path, "has an unreadable project description")
    if not isinstance(manifest, dict):
        _refuse(archive_path, "has an unreadable project description")
    version = manifest.get('version', 1)
    if not isinstance(version, int) or version > ARCHIVE_VERSION:
        _refuse(archive_path, "was made by a newer OpenFlo; update OpenFlo "
                "to open it")
    return manifest


def _packed_files(manifest, archive_path):
    """``{member: (path, bytes)}`` for each packed FCS file the manifest
    lists (``files``, codec ``FCS_CODEC``): each must name a member
    ``<path>.ofz`` of a top-level FCS file and a whole number of bytes."""
    out = {}
    files = manifest.get('files') or []
    if not isinstance(files, list):
        _refuse(archive_path, "has an unreadable project description")
    for f in files:
        if not isinstance(f, dict) or f.get('codec') != FCS_CODEC:
            continue
        path, member, size = f.get('path'), f.get('member'), f.get('bytes')
        if (not isinstance(path, str) or member != path + FCS_SUFFIX
                or not _expected_member(path, '')
                or not path.startswith(FCS_DIR + '/')
                or not isinstance(size, int) or isinstance(size, bool)
                or size < 0):
            _refuse(archive_path, "has an unreadable project description")
        out[member] = (path, size)
    return out


def _target(info, packed, session, archive_path):
    """``(path it unpacks to, its size, size of a packed file or None)``
    for member `info`, refused when no project holds it."""
    name = info.filename
    if name.startswith(FCS_DIR + '/') and name.endswith(FCS_SUFFIX):
        if name not in packed:
            _refuse(archive_path, f"holds {name!r}, a packed file its "
                    "project description does not list")
        path, size = packed[name]
        return path, size, size
    if not _expected_member(name, session):
        _refuse(archive_path, f"holds {name!r}, which is not part of an "
                "OpenFlo project")
    return name, info.file_size, None


def check_project_archive(zf, archive_path, max_bytes=None):
    """The members of the open zip `zf` to unpack, checked: the manifest
    and the session it names are there, every member is a part an OpenFlo
    project holds (``_expected_member``) under a safe name, no two land on
    one file, none is encrypted or packed in a way this reads, they claim
    no more bytes than the archive could hold (overlapping members, a zip
    bomb) and, unpacked, they come to at most `max_bytes` (None: no cap),
    a packed FCS file counted at the size its manifest entry gives (the
    unpack holds it to that). Returns ``(manifest, session member,
    members, unpacked bytes)``, each member ``(info, path it unpacks to,
    size of a packed FCS file or None)``;
    raises ``ProjectArchiveError`` naming the first fault."""
    infos = zf.infolist()
    names = [i.filename for i in infos]
    if MANIFEST not in names:
        raise ProjectArchiveError(
            f"{os.path.basename(archive_path)} is a zip file but not an "
            "OpenFlo project archive.")
    if len(infos) > MAX_MEMBERS:
        _refuse(archive_path, f"holds more than {MAX_MEMBERS:,} files")
    # The name as stored: on Windows zipfile turns a backslash into '/' and
    # cuts at a NUL, which would let a name refused elsewhere through.
    bad = [i.orig_filename for i in infos if not _safe_member(i.orig_filename)]
    if bad:
        _refuse(archive_path, "holds a file that would be written outside "
                f"its folder ({bad[0]!r})")
    manifest = _read_manifest(zf, archive_path)
    session = manifest.get('session') or SESSION_NAME
    if (not isinstance(session, str) or session not in names
            or not _safe_member(session) or '/' in session):
        raise ProjectArchiveError(
            f"{os.path.basename(archive_path)} has no session file.")
    packed_files = _packed_files(manifest, archive_path)
    seen, members, unpacked, packed = set(), [], 0, 0
    for info in infos:
        name = info.filename
        if info.is_dir():
            continue
        target, size, declared = _target(info, packed_files, session,
                                         archive_path)
        if info.flag_bits & 0x1:
            _refuse(archive_path, f"holds {name!r} encrypted")
        if info.compress_type not in (zipfile.ZIP_STORED,
                                      zipfile.ZIP_DEFLATED):
            _refuse(archive_path, f"holds {name!r} packed in a way OpenFlo "
                    "does not read")
        key = _fold(target)
        if key in seen:
            _refuse(archive_path, f"holds {target!r} twice")
        seen.add(key)
        members.append((info, target, declared))
        unpacked += size
        packed += info.compress_size
    try:
        size = os.path.getsize(archive_path)
    except (OSError, TypeError):
        size = None
    if size is not None and packed > size:
        _refuse(archive_path, "claims more data than the file holds (it is "
                "damaged, or crafted to exhaust the memory of the program "
                "that opens it)")
    if max_bytes is not None and unpacked > max_bytes:
        mb = (unpacked + 999_999) // 1_000_000
        cap = max_bytes // 1_000_000
        _refuse(archive_path, f"unpacks to {mb:,} MB and this device opens "
                f"projects of up to {cap:,} MB; open it in OpenFlo on a "
                "computer")
    return manifest, session, members, unpacked


def _extract(zf, info, root, target=None, declared=None):
    """Write member `info` of `zf` under `root` (at `target`, default its
    own name), in chunks, never more than the size it declares (zipfile
    also checks its CRC at the end). A packed FCS file (`declared` its
    size) is unpacked to the original, held to `declared` bytes."""
    out = os.path.join(root, *(target or info.filename).split('/'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    left = info.file_size if declared is None else declared
    with zf.open(info) as src, open(out, 'wb') as dst:
        # Sized once: an in-memory file system then never regrows it.
        dst.truncate(left)
        dst.seek(0)
        if declared is not None:
            from .fcs_pack import FcsPackError, unpack_fcs
            try:
                unpack_fcs(src, dst, declared)
            except FcsPackError as exc:
                raise ProjectArchiveError(f"{info.filename}: {exc}.") from exc
            return
        while True:
            chunk = src.read(min(CHUNK, left + 1))
            if not chunk:
                break
            left -= len(chunk)
            if left < 0:
                raise ProjectArchiveError(
                    f"{info.filename} is larger than the archive says.")
            dst.write(chunk)
    if left:
        raise ProjectArchiveError(
            f"{info.filename} is smaller than the archive says.")


def expand_project(archive_path, dest):
    """Turn the project archive at `archive_path` back into ordinary files
    in the new folder `dest`: each FCS file as it was, the session (with
    its saved data) and ``<archive name>.wsp``, a FlowJo workspace of the
    samples' gates and compensation (``Document.export_wsp``, the same
    writer as File > Export FlowJo .wsp), its samples pointing at the FCS
    files there. Returns ``{'folder', 'session', 'wsp', 'fcs': [file
    names], 'missing': [samples saved without their FCS file], 'gates',
    'left_out', 'warnings'}``; a sample without its FCS file there is left
    out of the workspace. The folder is removed when the workspace cannot
    be written."""
    from .document import Document
    sess, manifest = open_project_archive(archive_path, dest)
    folder = os.path.dirname(sess)
    try:
        doc = Document()
        opened = doc.open_session(sess)
        stem = os.path.splitext(os.path.basename(archive_path))[0]
        wsp = os.path.join(folder, stem + '.wsp')
        written = doc.export_wsp(wsp, within=folder)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    fcs_dir = os.path.join(folder, FCS_DIR)
    fcs = sorted(os.listdir(fcs_dir)) if os.path.isdir(fcs_dir) else []
    return {'folder': folder, 'session': sess, 'wsp': wsp, 'fcs': fcs,
            'missing': _all_missing(manifest, opened),
            'gates': written['gates'],
            'left_out': written['left_out'],
            'warnings': written['warnings']}


def _all_missing(manifest, opened):
    """Samples of an expanded project without their FCS file: those the
    archive was saved without (also when the session opens them from
    their saved data) and those the session could not find, each once."""
    saved = manifest.get('missing')
    names = [str(m) for m in saved] if isinstance(saved, list) else []
    names += [str(m) for m in opened['missing']]
    return list(dict.fromkeys(names))


def unpack_folder_for(archive_path):
    """A new, empty folder beside `archive_path` named after it."""
    base = os.path.splitext(os.path.abspath(archive_path))[0]
    out, i = base, 2
    while os.path.exists(out):
        out = f"{base} ({i})"
        i += 1
    return out


def open_project_archive(archive_path, dest=None, *, max_bytes=None):
    """Unpack the project archive at `archive_path` into `dest` (a new
    folder beside it by default) and return ``(session_path, manifest)``.

    Everything is checked first (``check_project_archive``; `max_bytes`
    caps the unpacked size, as a phone holding it in memory needs); with no
    cap, the disk must have room for it. Nothing is left behind when an
    archive is refused or fails part way: the files go to a hidden folder
    renamed to `dest` only once all are written."""
    with zipfile.ZipFile(archive_path) as zf:
        manifest, session, members, unpacked = check_project_archive(
            zf, archive_path, max_bytes)
        dest = dest or unpack_folder_for(archive_path)
        parent = os.path.dirname(os.path.abspath(dest)) or '.'
        os.makedirs(parent, exist_ok=True)
        if max_bytes is None:
            free = shutil.disk_usage(parent).free
            if unpacked + DISK_MARGIN > free:
                _refuse(archive_path, f"unpacks to "
                        f"{(unpacked + 999_999) // 1_000_000:,} MB and the "
                        f"disk has {free // 1_000_000:,} MB free")
        work = tempfile.mkdtemp(prefix='.openflo-unpack-', dir=parent)
        try:
            for info, target, declared in members:
                _extract(zf, info, work, target, declared)
            os.replace(work, dest)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    return os.path.join(dest, session), manifest
