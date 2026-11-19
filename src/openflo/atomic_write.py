"""Writing a file so it is whole or untouched: the content goes to a
temporary file beside it, which is flushed to the disk (``fsync``) and then
replaces it (``os.replace``). A crash, a full disk or a value that will not
serialise halfway through leaves the old file as it was, and no temporary
file behind; once the write has returned, a power cut leaves the new file
(on a filesystem that keeps what ``fsync`` says it kept).

The replaced file keeps its permissions, and a path that is a symbolic
link has the file it points to replaced, the link kept. Each write has its
own temporary name, so two OpenFlo windows saving one file at once do not
write into each other's temporary file (the last to finish wins). On
Windows a file another program holds open (Excel) still cannot be
replaced: the write raises, and the old file is kept.

Core module: no Tk.
"""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import stat
from collections.abc import Callable, Iterator
from typing import IO, Any


def _fsync_path(path: str, directory: bool = False) -> None:
    """Flush the file (or folder) at `path` to the disk. A folder cannot
    be opened for this on Windows: skipped there."""
    if directory and os.name == 'nt':
        return
    fd = os.open(path, os.O_RDONLY if directory else os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _new_temp(target: str) -> str:
    """Create an empty temporary file beside `target` (the new-file
    permissions this process gives, its umask applied); return its path."""
    while True:
        folder, name = os.path.split(target)
        tmp = os.path.join(folder, f'.{secrets.token_hex(4)}.{name}.tmp')
        try:
            os.close(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            continue
        return tmp


def _keep_mode(tmp: str, target: str) -> None:
    """Give `tmp` the permissions of `target`, when it exists. Done after
    the write, so a read-only `target` is still replaced."""
    with contextlib.suppress(FileNotFoundError):
        os.chmod(tmp, stat.S_IMODE(os.stat(target).st_mode))


@contextlib.contextmanager
def atomic_path(path: str) -> Iterator[str]:
    """A temporary path beside `path` (``.<random>.<name>.tmp``, already
    created, empty) for a writer that takes a path: when the ``with``
    block ends without an error it is flushed to the disk and replaces
    `path` (the file a symbolic link `path` points to); when it does not,
    it is removed."""
    target = os.path.realpath(path)
    tmp = _new_temp(target)
    try:
        yield tmp
        _fsync_path(tmp)
        _keep_mode(tmp, target)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    # The new name is in the folder's entry once the folder is flushed.
    with contextlib.suppress(OSError):
        _fsync_path(os.path.dirname(target) or '.', directory=True)


@contextlib.contextmanager
def atomic_open(path: str, mode: str = 'w', **open_kw: Any) -> Iterator[IO]:
    """``open(path, mode)`` for writing, through :func:`atomic_path`."""
    with atomic_path(path) as tmp, open(tmp, mode, **open_kw) as fh:
        yield fh


def json_default(obj: Any) -> Any:
    """``json.dump``'s ``default`` for numpy values (an array, as nested
    lists; a scalar, as its Python number). Anything else still raises."""
    tolist = getattr(obj, 'tolist', None)
    if callable(tolist):
        return tolist()
    raise TypeError(f'{type(obj).__name__} is not JSON serializable')


def atomic_write_json(path: str, obj: Any, *, indent: int | None = 2,
                      default: Callable[[Any], Any] | None = json_default
                      ) -> None:
    """Write `obj` as JSON (UTF-8) to `path`, whole or not at all."""
    with atomic_open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=indent, default=default)
