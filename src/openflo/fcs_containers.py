"""BD Accuri C6 files: several FCS files in one zip archive.

The BD Accuri C6 software saves a whole run in one file (``.c6``) that
"contains instrument settings, FCS files, and plot layouts" (BD Accuri C6
Software User Guide, section 3.11). The file is a zip archive: the
BSD-licensed accuri2fcs converter opens it as one, finds each well's data in
a member named like ``Sample A01.fcs``, and reads the well's sample name
from a member ``collectManager.txt`` that lists wells as
``$SAMPLEIDENTIFIER/A1/$SAMPLENAME/<name>/``. FCS Express opens ``.ci`` and
``.cfl`` files with the same "C6 files" filter, so they are read the same
way. BD does not publish the layout; this module was written from those
public descriptions, then checked against public C6 files (Zenodo records
4972042 and 15351644). Those hold, besides ``collectManager.txt`` and one
``Sample A01.fcs`` member per well, about twenty settings and layout
members (and sometimes two members of one name), all left alone here. A
plate's wells that were never run are kept as FCS members with no events
($TOT 0): they are left out, with a note.

OpenFlo opens such a file as one sample per FCS member (``list_members``),
named the way the C6 software names its own FCS export: the well, then the
sample name (``A01 unst``). A member is numbered from 1 in well order; the
editor keeps that number as the sample's ``set_number``, as for a data set
of an FCS file holding several (fcs_data_sets), and a session also keeps
the member's name, so it reopens the same well even if the file gained
wells since (``resolve_member``).

A member is read by copying it, once, into OpenFlo's own cache folder
(``member_file``); nothing is ever written next to the user's file. The
archive is untrusted: the member count, each member's size and the index's
size are capped, and a member is copied with its bytes counted, whatever
size the archive declares.

Core module: standard library only. No Tk.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass

from .fcs_data_sets import HEADER_SIZE, DataSet, MissingDataSetError, parse_text
from .paths import FCS_EXTENSIONS, is_fcs_container

# The most members (files) an archive may list. A 96-well plate is 96 FCS
# files and a few more; far more is not a C6 file.
MAX_MEMBERS = 2000
# The largest FCS member read (bytes), counted as it is copied.
MAX_MEMBER_BYTES = 2 * 1024 ** 3
# The most bytes all members may claim together.
MAX_TOTAL_BYTES = 64 * 1024 ** 3
# The largest well index (collectManager.txt) read.
MAX_INDEX_BYTES = 4 * 1024 ** 2
# The largest TEXT segment of a member read for its keywords.
MAX_TEXT_BYTES = 1024 ** 2
# The end of the note on a file's wells with no events (list_members).
_EMPTY_NOTE = 'empty well(s) with no events were left out'
# The longest sample name taken from the index.
_MAX_NAME = 80
# Copied members are kept this long after their file was last read.
CACHE_DAYS = 7

INDEX_NAME = 'collectmanager.txt'
_INDEX_ENTRY = re.compile(
    r'\$SAMPLEIDENTIFIER/([A-Za-z])0*(\d{1,3})/\$SAMPLENAME/([^/]*)/',
    re.IGNORECASE)
# A well at the end of a member's name: 'Sample A01', 'A1', 'B12'.
_MEMBER_WELL = re.compile(r'(?:^|[\s_-])([A-Za-z])0*(\d{1,3})$')


class ContainerError(ValueError):
    """A BD Accuri file that cannot be opened: not a zip archive, damaged,
    or beyond the caps. The message is for the user."""


@dataclass(frozen=True)
class Member(DataSet):
    """One FCS file of a container: ``number`` (from 1, in well order),
    ``member`` its name in the archive, ``label`` the sample name it opens
    under, ``size`` its declared size in bytes."""

    member: str = ''
    label: str = ''
    size: int = 0


def _export_hint() -> str:
    return ("In the Accuri software, choose File > Export ALL Samples as "
            "FCS, then open those FCS files.")


def _name(path) -> str:
    return os.path.basename(str(path))


def _open_zip(path) -> zipfile.ZipFile:
    """The archive at `path`, or ContainerError (OSError when the file
    cannot be read at all)."""
    name = _name(path)
    if not zipfile.is_zipfile(path):
        raise ContainerError(
            f"{name} is not a BD Accuri file OpenFlo can read: it is not a "
            f"zip archive. {_export_hint()}")
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ContainerError(f"{name} is damaged: {exc}. "
                             f"{_export_hint()}") from exc


def _files(zf: zipfile.ZipFile, name: str) -> list[zipfile.ZipInfo]:
    """The archive's file members, within the caps."""
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_MEMBERS:
        raise ContainerError(
            f"{name} holds {len(infos)} files, more than the {MAX_MEMBERS} "
            f"OpenFlo opens from one archive. {_export_hint()}")
    if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
        raise ContainerError(
            f"{name} says it holds more than "
            f"{MAX_TOTAL_BYTES // 1024 ** 3} GB of data, which OpenFlo does "
            f"not unpack. {_export_hint()}")
    return infos


def _starts_fcs(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    """Whether member `info` is an FCS file: unencrypted, at least a HEADER
    long, within the size cap, and starting 'FCS'."""
    if info.flag_bits & 0x1 or not HEADER_SIZE <= info.file_size:
        return False
    try:
        with zf.open(info) as fh:
            return fh.read(3) == b'FCS'
    except (OSError, zipfile.BadZipFile, NotImplementedError, RuntimeError):
        return False


def _member_keywords(zf: zipfile.ZipFile, info: zipfile.ZipInfo
                     ) -> dict[str, str]:
    """The TEXT keywords of FCS member `info` (keys upper-case, '$' kept),
    read from its first bytes only; {} when its TEXT cannot be read or
    ends past MAX_TEXT_BYTES. A workspace sample is matched to its well by them
    (fcs_data_sets.data_set_matches)."""
    try:
        with zf.open(info) as fh:
            hdr = fh.read(HEADER_SIZE)
            t0, t1 = int(hdr[10:18]), int(hdr[18:26])
            if not HEADER_SIZE <= t0 <= t1 < MAX_TEXT_BYTES:
                return {}
            fh.read(t0 - HEADER_SIZE)
            raw = fh.read(t1 - t0 + 1)
        return {str(k).upper(): str(v) for k, v in parse_text(raw)}
    except (OSError, ValueError, IndexError, zipfile.BadZipFile,
            NotImplementedError, RuntimeError):
        return {}


def _well_names(zf: zipfile.ZipFile, infos) -> dict[str, str]:
    """``{well ('A01'): sample name}`` from the archive's well index, or
    {} when it has none that can be read."""
    index = next((i for i in infos
                  if i.filename.replace('\\', '/').rsplit('/', 1)[-1].lower()
                  == INDEX_NAME), None)
    if index is None or index.file_size > MAX_INDEX_BYTES:
        return {}
    try:
        with zf.open(index) as fh:
            raw = fh.read(MAX_INDEX_BYTES + 1)
    except (OSError, zipfile.BadZipFile, NotImplementedError, RuntimeError):
        return {}
    text = raw[:MAX_INDEX_BYTES].decode('utf-8', 'replace')
    text = text.replace('\r', '').replace('\n', '')
    return {_well(row, col): _clean(nm)
            for row, col, nm in _INDEX_ENTRY.findall(text)}


def _well(row: str, col: str) -> str:
    """'A', '1' -> 'A01'."""
    return f'{row.upper()}{int(col):02d}'


def _clean(text: str) -> str:
    """`text` as a sample name: control characters dropped, white space
    collapsed, length capped."""
    t = ''.join(ch if ch.isprintable() else ' ' for ch in str(text))
    return ' '.join(t.split())[:_MAX_NAME].strip()


def _stem(member: str) -> str:
    """A member's file name without its folder and extension."""
    base = member.replace('\\', '/').rsplit('/', 1)[-1]
    return os.path.splitext(base)[0]


def _member_well(member: str) -> str:
    """The well a member's name ends in ('Sample A01.fcs' -> 'A01'), or ''."""
    m = _MEMBER_WELL.search(_stem(member))
    return _well(m.group(1), m.group(2)) if m else ''


def _sort_key(member: str) -> tuple:
    """Wells first, by row then column; then any other member by name."""
    well = _member_well(member)
    if well:
        return (0, well[0], int(well[1:]), member)
    return (1, '', 0, member)


# The listings read lately, by file and version (see _file_ident): a
# session save asks for the member name of every well.
_LISTINGS: dict[str, tuple[list[Member], str]] = {}
_MAX_LISTINGS = 32


def _file_ident(path) -> str:
    """`path`, absolute, with its size and modification time: another
    value once the file changes."""
    st = os.stat(path)
    return f'{os.path.abspath(str(path))}|{st.st_size}|{st.st_mtime_ns}'


def list_members(path) -> tuple[list[Member], str]:
    """``list_members_uncached``, remembered per version of the file."""
    ident = _file_ident(path)
    got = _LISTINGS.get(ident)
    if got is None:
        got = list_members_uncached(path)
        if len(_LISTINGS) >= _MAX_LISTINGS:
            _LISTINGS.clear()
        _LISTINGS[ident] = got
    return list(got[0]), got[1]


def list_members_uncached(path) -> tuple[list[Member], str]:
    """``(members, note)``: the FCS files of the BD Accuri file at `path`,
    numbered from 1 in well order, each labelled with its well and sample
    name ('A01 unst'; the well alone when the index names none; the
    member's file name when it names no well). `note` says, for the user,
    how many FCS members were left out for being over the size cap or for
    holding no events ('' when none). Empty when the archive holds no FCS
    file with events. Raises ContainerError
    for a file that is not such an archive or is beyond the caps."""
    name = _name(path)
    with _open_zip(path) as zf:
        infos = _files(zf, name)
        big = [i.filename for i in infos if i.file_size > MAX_MEMBER_BYTES]
        fcs = sorted((i for i in infos if i.file_size <= MAX_MEMBER_BYTES
                      and _starts_fcs(zf, i)),
                     key=lambda i: _sort_key(i.filename))
        wells = _well_names(zf, infos)
        keywords = [_member_keywords(zf, i) for i in fcs]
    kept = [(i, kw) for i, kw in zip(fcs, keywords, strict=True)
            if not _is_empty(kw)]
    out = []
    for n, (info, kw) in enumerate(kept, 1):
        well = _member_well(info.filename)
        if well:
            sample = wells.get(well, '')
            label = f'{well} {sample}' if sample else well
        else:
            label = _clean(_stem(info.filename)) or f'{name} #{n}'
        out.append(Member(number=n, offset=0, version='', keywords=kw,
                          member=info.filename, label=label,
                          size=info.file_size))
    notes = []
    if big:
        notes.append(f"{len(big)} file(s) larger than "
                     f"{MAX_MEMBER_BYTES // 1024 ** 3} GB were not opened")
    empty = len(fcs) - len(kept)
    if empty:
        notes.append(f"{empty} {_EMPTY_NOTE}")
    return out, (f"{name}: " + '; '.join(notes)) if notes else ''


def _is_empty(keywords: dict[str, str]) -> bool:
    """Whether a member's TEXT says it holds no events ($TOT 0): the C6
    software keeps such a member for a plate's wells that hold no data."""
    try:
        return int(keywords.get('$TOT', '').strip()) == 0
    except ValueError:
        return False


def _member(path, number) -> tuple[Member, int]:
    """Member `number` of the file at `path`, and how many it holds.
    Raises ContainerError (no FCS member at all) or MissingDataSetError."""
    members, note = list_members(path)
    if not members and _EMPTY_NOTE in note:
        raise ContainerError(
            f"{_name(path)} holds no events: every well in it is empty.")
    if not members:
        raise ContainerError(
            f"{_name(path)} holds no FCS data. {_export_hint()}")
    n = 1 if number is None else int(number)
    if not 1 <= n <= len(members):
        held = ('one sample' if len(members) == 1
                else f'{len(members)} samples')
        raise MissingDataSetError(
            f"its sample {n} was not found: the file now holds {held}")
    return members[n - 1], len(members)


def cache_root() -> str:
    """Where copied members go: ``~/.openflo/cache/containers``."""
    return os.path.join(os.path.expanduser('~'), '.openflo', 'cache',
                        'containers')


def _cache_dir(path) -> str:
    """The cache folder of the file at `path` as it is now: a new one when
    the file changes (its size or modification time)."""
    ident = _file_ident(path)
    return os.path.join(cache_root(),
                        hashlib.sha256(ident.encode('utf-8')).hexdigest()[:20])


def _safe(label: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ' -_' else '_'
                   for ch in label).strip() or 'sample'


def prune_cache(keep: str = '', now: float | None = None) -> None:
    """Remove the cache folders not used for CACHE_DAYS (but never
    `keep`). Errors are ignored: a folder in use stays."""
    root = cache_root()
    try:
        names = os.listdir(root)
    except OSError:
        return
    limit = (time.time() if now is None else now) - CACHE_DAYS * 86400
    for nm in names:
        d = os.path.join(root, nm)
        try:
            if d != keep and os.path.getmtime(d) < limit:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            continue


def member_file(path, number=None) -> tuple[str, Member, int]:
    """``(copy, member, count)``: the path of a copy of FCS member
    `number` (from 1; None: the first) of the BD Accuri file at `path`, in
    OpenFlo's cache folder, that member, and how many FCS members the file
    holds. The copy is made once per version of the file. Raises
    ContainerError or MissingDataSetError."""
    member, count = _member(path, number)
    d = os.path.join(_cache_dir(path), str(member.number))
    out = os.path.join(d, _safe(member.label) + FCS_EXTENSIONS[0])
    if os.path.isfile(out) and os.path.getsize(out) == member.size:
        os.utime(os.path.dirname(d))
        return out, member, count
    os.makedirs(d, exist_ok=True)
    _copy_member(path, member, out)
    prune_cache(keep=os.path.dirname(d))
    return out, member, count


def _copy_member(path, member: Member, out: str) -> None:
    """Copy `member` of the archive at `path` to `out`, counting its bytes
    against the cap and its declared size; written to a temporary name of
    its own first (unique to this call: loader threads may copy one well
    at once), then put in place in one step, so a copy cut short is never
    taken for a whole one, and a copy another call put there first is
    simply replaced by the same bytes."""
    name = _name(path)
    fd, tmp = tempfile.mkstemp(suffix='.part', dir=os.path.dirname(out))
    limit = min(member.size, MAX_MEMBER_BYTES)
    done = 0
    try:
        with os.fdopen(fd, 'wb') as dst, _open_zip(path) as zf, \
                zf.open(member.member) as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                done += len(chunk)
                if done > limit:
                    raise ContainerError(
                        f"{name}: {member.label} is larger than the archive "
                        "says, so it was not opened. The file may be "
                        "damaged.")
                dst.write(chunk)
        if done != member.size:
            raise ContainerError(f"{name}: {member.label} is cut short.")
        os.replace(tmp, out)
    except (zipfile.BadZipFile, KeyError, RuntimeError) as exc:
        raise ContainerError(f"{name}: {member.label} could not be read "
                             f"({exc}). The file may be damaged.") from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def member_name(path, number) -> str:
    """The archive name of FCS member `number` of the file at `path`, or ''
    when it cannot be read."""
    try:
        return _member(path, number)[0].member
    except (OSError, ValueError):
        return ''


def item_set_number(item) -> int | None:
    """The data set number to reload a workspace item (``workspace_run``)
    from: its ``set_number``, checked against its BD Accuri file by the
    ``container_member`` it saved (``resolve_member``)."""
    number = item.get('set_number')
    path = str(item.get('path') or '')
    member = item.get('container_member')
    if (number and isinstance(member, str) and member
            and is_fcs_container(path) and os.path.isfile(path)):
        return resolve_member(path, number, member)
    return number


def resolve_member(path, number: int | None, member: str) -> int | None:
    """The number of the FCS member a session saved as `number` and
    archive name `member`, in the file at `path` as it is now: `number`
    when that member still has it, else the member's new number, else a
    number past the last, so the load says the sample is missing. `number`
    alone (no saved name, or a file that cannot be read) is kept."""
    if not number or not member:
        return number
    try:
        members, _note = list_members(path)
    except (OSError, ValueError):
        return number
    names = [m.member for m in members]
    if 1 <= number <= len(names) and names[number - 1] == member:
        return number
    if member in names:
        return names.index(member) + 1
    return len(names) + 1
