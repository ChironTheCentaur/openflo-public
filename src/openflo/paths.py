"""Filesystem path helpers lifted out of the GUI — Tk-free, testable.

Drop-target expansion, sidecar-name sanitisation and the portable relink
hints a session keeps per sample: pure path logic the editor uses for
drag-and-drop import, processed-data sidecars and session files.

Portable paths: a session saved on one computer is opened on another, of
any OS (a Windows lab PC, a Mac, a phone working on a copy in its own
storage). So the paths a session stores are read whatever separator they
were written with, and a path that is absolute on another OS (``C:\\...``,
``\\\\server\\share\\...``, ``/...``) is never taken for one relative to
the folder it is read in.
"""
from __future__ import annotations

import hashlib
import ntpath
import os
import re
from collections.abc import Mapping
from typing import Any


def safe_sidecar_name(name) -> str:
    """Filesystem-safe stem for a sample's processed-data sidecar. The same
    mapping is used by the session writer and the loader's fallback location,
    so a sidecar is found even if the recorded pointer is missing.

    Mapping every character outside ``[alnum-_]`` to ``_`` is not injective.
    ``Tube 01 Rep A`` and ``Tube_01_Rep_A`` would produce the SAME stem, as
    would ``Day 1``/``Day.1``, ``Patient 01``/``Patient#01`` and
    ``Live/Dead``/``Live Dead``. The session writer would then hand both
    samples that one file and record it on both entries, and restore PREFERS
    the sidecar over the raw FCS — so one sample would come back holding the
    other's events, clusters and compensated values, with nothing said.

    A short digest of the ORIGINAL name is therefore appended whenever
    sanitisation actually changed it. A name that was already filesystem-safe
    keeps its plain stem, so sidecars written by earlier versions are still
    found by the loader's fallback; and two names can share a stem only if
    they were identical to begin with.
    """
    text = str(name)
    safe = ''.join(ch if (ch.isalnum() or ch in '-_') else '_' for ch in text)
    if not safe:
        return 'sample'          # one empty name, so nothing to disambiguate
    if safe == text:
        return safe
    return f'{safe}_{hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]}'


def legacy_sidecar_name(name) -> str:
    """The pre-digest sidecar stem: a plain character substitution.

    Kept ONLY so the loader can still find sidecars written by earlier
    versions when the recorded pointer is missing. It is not injective — which
    is why it was replaced — so nothing should ever WRITE with it.
    """
    return ''.join(ch if (ch.isalnum() or ch in '-_') else '_'
                   for ch in str(name)) or 'sample'


# The event-data files OpenFlo opens: FCS, and Beckman Coulter list-mode
# (.lmd, an FCS file holding two data sets; see fcs_quirks.read_fcs_tolerant).
FCS_EXTENSIONS = ('.fcs', '.lmd')
# Files that hold several FCS files, one sample each: the BD Accuri C6
# software's file (.c6), and the C6 Plus files FCS Express opens with it
# (.ci, .cfl). See fcs_containers.
CONTAINER_EXTENSIONS = ('.c6', '.ci', '.cfl')
# As Tk file-dialog patterns: the FCS files every picker lists, and the
# Accuri files only the editor's Add FCS lists (it opens one sample per
# well; a tool that reads one sample per file does not take them).
FCS_FILE_PATTERNS = '*.fcs *.lmd'
CONTAINER_FILE_PATTERNS = '*.c6 *.ci *.cfl'


def is_fcs_container(path) -> bool:
    """True for a file name of a BD Accuri file (``.c6``, ``.ci`` or
    ``.cfl``, any case): several FCS files in one, opened one sample per
    well (fcs_containers)."""
    return str(path).lower().endswith(CONTAINER_EXTENSIONS)


def is_single_fcs_file(path) -> bool:
    """True for a file name that is one FCS file (``.fcs`` or ``.lmd``, any
    case), not a BD Accuri file holding several: what the command line, the
    workspace batch and the folder watch read, one sample per file."""
    return str(path).lower().endswith(FCS_EXTENSIONS)


def is_fcs_file(path) -> bool:
    """True for a file name OpenFlo reads FCS data from: ``.fcs`` or
    ``.lmd``, or a BD Accuri file holding several (``is_fcs_container``),
    any case."""
    return str(path).lower().endswith(FCS_EXTENSIONS + CONTAINER_EXTENSIONS)


def sample_key(path, set_number=None) -> str:
    """The key the editor files what it staged for a sample under: `path`
    normalised (absolute, and case-folded where the OS ignores case), and,
    for one data set of an FCS file holding several (`set_number`, from
    1), or one well of a BD Accuri file (its number, fcs_containers),
    that number after a '#'. Such a key never names a file: it does not
    end in a data file's extension. A sidecar's path is keyed alone, whatever data set
    its values came from."""
    key = os.path.normcase(os.path.abspath(str(path)))
    if set_number and is_fcs_file(path):
        key = f'{key}#{int(set_number)}'
    return key


def fcs_stem(name) -> str:
    """The base name of ``name`` without its data file extension (``.fcs``,
    ``.lmd``, ``.c6``...); any other name is returned whole (base name
    only)."""
    base = os.path.basename(str(name))
    return os.path.splitext(base)[0] if is_fcs_file(base) else base


def is_workspace_file(path) -> bool:
    """True for a FlowJo workspace (``.wsp``) or a BD FACSDiva experiment
    exported as XML (an ``.xml`` file whose root is ``<bdfacs>``; another
    ``.xml``, such as a Gating-ML file, is not one)."""
    low = str(path).lower()
    if low.endswith('.wsp'):
        return True
    if low.endswith('.xml'):
        from .diva_reader import is_diva_xml
        return is_diva_xml(path)
    return False


SESSION_EXTENSIONS = ('.flowsession', '.flowproject')


def is_session_file(path) -> bool:
    """True for a file the editor opens as a whole session: an OpenFlo
    session (``.flowsession``) or project archive (``.flowproject``, or an
    older archive saved as ``.zip``, recognised by its manifest), any
    case."""
    low = str(path).lower()
    if low.endswith(SESSION_EXTENSIONS):
        return True
    if low.endswith('.zip'):
        from .project_archive import is_project_archive
        return is_project_archive(path)
    return False


def expand_dropped_paths(paths, vendor: list[str] | None = None,
                         sessions: list[str] | None = None
                         ) -> tuple[list[str], list[str]]:
    """Resolve dropped paths (files and/or folders) into the flat FCS
    (``.fcs`` / ``.lmd``, or a BD Accuri ``.c6``) files and workspaces
    (``.wsp``, or a FACSDiva experiment ``.xml``; see
    ``is_workspace_file``) they contain. Folders are walked recursively, so
    dropping a trial folder (or a parent of several) surfaces every sample
    inside.

    Returns ``(fcs_paths, wsp_paths)``, each de-duplicated and sorted for a
    deterministic load order. Instrument files OpenFlo cannot read
    (``vendor_formats``, by extension) are appended, sorted, to `vendor`
    when it is given, so the caller can say how to export them as FCS.
    Session and project files (``is_session_file``) dropped as files, not
    found inside a dropped folder, are appended in drop order to
    `sessions` when it is given; a folder's own autosave or archive never
    replaces the samples it was dropped for.
    """
    from .vendor_formats import vendor_format
    fcs, wsp, other = set(), set(), set()

    def _add_file(fp):
        if is_fcs_file(fp):
            fcs.add(fp)
        elif is_workspace_file(fp):
            wsp.add(fp)
        elif vendor_format(fp) is not None:
            other.add(fp)

    for p in paths:
        p = (p or '').strip().strip('"').strip("'")
        if not p:
            continue
        if os.path.isdir(p):
            for dirpath, _dirs, files in os.walk(p):
                for fn in files:
                    _add_file(os.path.join(dirpath, fn))
        elif os.path.isfile(p):
            if sessions is not None and is_session_file(p):
                if p not in sessions:
                    sessions.append(p)
            else:
                _add_file(p)
    if vendor is not None:
        vendor.extend(sorted(other))
    return sorted(fcs), sorted(wsp)


def dialog_start_dir(last_dir=None, package_dir=None) -> str:
    """The folder a file dialog opens in: the last data folder used when it
    still exists, else the current folder, else the home folder. Never the
    installed package's own folder (``src/openflo``, or site-packages after a
    pip install), which a fresh home would otherwise give the Add FCS
    dialog: a current folder inside the package -- a shortcut that starts
    OpenFlo there -- falls back to the home folder too."""
    if last_dir and os.path.isdir(last_dir):
        return str(last_dir)
    home = os.path.expanduser('~')
    try:
        cwd = os.getcwd()
    except OSError:                       # the current folder was deleted
        return home
    if package_dir:
        pkg = os.path.normcase(os.path.abspath(package_dir))
        here = os.path.normcase(os.path.abspath(cwd))
        if here == pkg or here.startswith(pkg + os.sep):
            return home
    return cwd


# ── Portable session paths ─────────────────────────────────────────────────

# This computer's path flavour. The helpers below read it rather than
# ``os.path`` so a test can run their Windows branches on any OS by giving
# this module ``ntpath`` (never by patching ``os`` itself).
_osp = os.path

# How a saved path's own shape is read, whatever OS wrote or reads it:
# a single ASCII letter and a colon at the very start is a Windows drive;
# followed by a separator (``C:\\x``, ``C:/x``) the path is absolute, with
# none (``E:x.fcs``, drive-relative) it is not, and the drive is not part of
# the file name. A colon anywhere else (``run1/a:b.fcs``) is a character of
# a POSIX name. A saved path that is just ``a:b.fcs`` is read as
# drive-relative (by shape alone it cannot be told from a POSIX name; a
# local file of that name is linked from where it is, and a saved
# ``basename`` field is read whole).
_DRIVE = re.compile(r'[A-Za-z]:')
_DRIVE_ROOT = re.compile(r'[A-Za-z]:[\\/]')


def portable_basename(path: str, *, strip_drive: bool = True) -> str:
    """The file name at the end of a saved `path`, split on both ``\\`` and
    ``/`` whatever OS reads it ('' for ''), and without a leading Windows
    drive (``E:x.fcs`` -> ``x.fcs``) unless `strip_drive` is false: a saved
    ``basename`` field is already a file name taken from a whole path, so
    one that starts ``a:`` is a POSIX name and is read with
    ``strip_drive=False``."""
    name = path.replace('\\', '/').rsplit('/', 1)[-1]
    if strip_drive and name == path and _DRIVE.match(path):
        name = path[2:]
    return name


def is_foreign_absolute(path: str) -> bool:
    """Whether `path` is absolute on Windows or on POSIX, whichever OS
    reads it: a drive letter and a separator (``C:\\x``, ``C:/x``), a UNC
    share (``\\\\server\\share``) or a leading separator. Such a path is
    never relative to the folder it is read in. A drive-relative
    ``E:x.fcs`` and a POSIX name with a colon (``a:b.fcs``) are not
    absolute."""
    return (path.startswith(('/', '\\'))
            or bool(_DRIVE_ROOT.match(path)))


def portable_join(folder: str, rel: str) -> str:
    """The normalised path a saved relative path `rel` (written with either
    separator) names under `folder`."""
    return _osp.normpath(_osp.join(folder, rel.replace('\\', '/')))


def portable_rel_path(path: str, folder: str) -> str | None:
    """`path` relative to `folder`, with ``/`` separators, or None when it
    cannot be: `path` is not an absolute path on this computer (a Windows
    path read on Linux, a path without a drive on Windows) or is on another
    drive than `folder`. Never relative to the process's working folder."""
    if not path or not _osp.isabs(path) or (
            _osp is ntpath and not ntpath.splitdrive(path)[0]):
        return None
    try:
        rel = _osp.relpath(path, folder)
    except ValueError:              # another drive (Windows)
        return None
    return rel.replace(_osp.sep, '/')


# A raw file's identity, saved with its sample (``fcs_id``): its size and
# a hash of its first bytes, which hold the FCS header and keywords (date,
# times, event count, file name) and the first events. Enough to tell two
# recordings of one file name apart, and cheap on a slow USB stick.
FCS_ID_HEAD = 1 << 16
_FCS_IDS: dict[tuple[str, int, int], dict[str, Any]] = {}


def fcs_identity(path: str) -> dict[str, Any] | None:
    """``{'bytes': size, 'head': hash}`` of the file at `path` (see
    ``FCS_ID_HEAD``), or None when it cannot be read. Remembered per path,
    size and modification time, so saving again does not read it again."""
    try:
        st = os.stat(path)
        key = (_osp.abspath(path), st.st_size, st.st_mtime_ns)
        if key in _FCS_IDS:
            return dict(_FCS_IDS[key])
        with open(path, 'rb') as fh:
            head = fh.read(FCS_ID_HEAD)
    except OSError:
        return None
    out = {'bytes': st.st_size,
           'head': hashlib.sha256(head).hexdigest()[:32]}
    if len(_FCS_IDS) > 4096:
        _FCS_IDS.clear()
    _FCS_IDS[key] = out
    return dict(out)


def same_fcs_identity(saved: Any, found: Mapping[str, Any] | None) -> bool:
    """Whether a saved ``fcs_id`` names the file whose identity is
    `found`."""
    return (isinstance(saved, Mapping) and found is not None
            and saved.get('bytes') == found['bytes']
            and saved.get('head') == found['head'])


def relink_hints(path: str, session_dir: str, *, kept_rel: str = '',
                 kept_from: str = '') -> dict[str, Any]:
    """The portable relink keys a session saved in `session_dir` writes for
    a sample at `path`: ``basename`` (its file name), ``fcs_id`` (the
    file's identity, ``fcs_identity``, when it is there to read; a sample
    whose file is missing keeps the one it was saved with) and, when it can
    be worked out, ``rel_path`` (relative to the session's folder, ``/``
    separators; see ``session_open.resolve_sample_path`` for the reading).

    A sample whose raw file was not found keeps the ``rel_path`` it was
    saved with (`kept_rel`, relative to `kept_from`, the folder of the
    session it came from), carried over to `session_dir`: its saved `path`
    is from the computer that last had the file, and would overwrite a
    good project-relative link. When the kept link cannot be carried over
    (to another drive), the link is worked out from `path` as for any
    sample. A `path` that is relative on this computer but names an
    existing file (a sample loaded by a relative name) is taken from where
    it is. When no link can be worked out (`path` is on yet another drive,
    or not a path on this computer), there is no ``rel_path`` key, and the
    saver drops any ``rel_path`` the entry held: it is relative to another
    folder."""
    if (path and not _osp.isabs(path) and not is_foreign_absolute(path)
            and os.path.isfile(path)):
        path = _osp.abspath(path)
    out: dict[str, Any] = {
        'basename': portable_basename(path) if path else ''}
    rel = None
    if kept_rel and kept_from:
        rel = portable_rel_path(portable_join(kept_from, kept_rel),
                                session_dir)
    if rel is None and path:
        rel = portable_rel_path(path, session_dir)
    if rel is not None:
        out['rel_path'] = rel
    ident = fcs_identity(path) if path and _osp.isfile(path) else None
    if ident is not None:
        out['fcs_id'] = ident
    return out


def saved_file_name(entry: Mapping[str, Any]) -> str:
    """The file name a saved session sample `entry` names: its ``basename``
    field (split on either separator, as older savers wrote a whole
    Windows path there), else its ``path``'s ('' when it has neither)."""
    saved = entry.get('basename') or ''
    if saved:
        return portable_basename(saved, strip_drive=False)
    return portable_basename(entry.get('path') or '')


def session_relative(path: str, session_dir: str) -> str:
    """A saved sample `path` read against `session_dir` when it is relative
    on this computer and not absolute on another OS (``C:\\...`` read on
    Linux is left as it is), made absolute; any other `path` unchanged.

    A drive-relative Windows path (``E:x.fcs``) is left as saved too,
    unless it names a file in `session_dir`: made absolute, it would become
    ``<session folder>/E:x.fcs`` on Linux or a path on one Windows PC's
    current folder of drive E:, and a save would write that over the file
    name the session had."""
    if not path or _osp.isabs(path) or is_foreign_absolute(path):
        return path
    joined = _osp.join(session_dir or '', path)
    if _DRIVE.match(path) and not (joined != path
                                   and os.path.isfile(joined)):
        return path
    return _osp.abspath(joined)
