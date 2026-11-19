"""A smaller, lossless form of an FCS file for project archives: the file's
own bytes, with its event data byte-shuffled before deflate.

Event values are fixed-width numbers (``$PnB`` bits each, the same for
every parameter in nearly every file). Putting the first byte of every
value together, then the second, and so on, lines up the slowly changing
high bytes, which deflate then packs far better than the file as it is
(about 15% smaller for full-precision floats, 45% for whole-number ones,
on generated data; the plain file deflates by 6% and 30%). The shuffle is
done in blocks, so neither packing nor unpacking holds more than a block
beyond the file itself, and unpacking gives back the file byte for byte
(its CRC-32 is checked).

Packed form (``pack_fcs``)::

    MAGIC  b'OFZ1'
    u32    length of the JSON header that follows (little-endian)
    JSON   {'size', 'crc32', 'begin', 'end', 'width'}: the original's size
           and CRC-32, the event data's first and last byte, the value
           width in bytes (0: not shuffled)
    zlib   the original's bytes, those in [begin, end] shuffled per block

Only OpenFlo reads it (``unpack_fcs``); a project archive lists each packed
member's codec in its manifest.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from typing import IO, Any

MAGIC = b'OFZ1'
CODEC = 'fcs-shuffle-deflate'
SUFFIX = '.ofz'
# Shuffled in blocks of this many bytes (a multiple of every width).
BLOCK = 4 << 20
LEVEL = 6
CHUNK = 1 << 20
MAX_HEADER = 1 << 16


class FcsPackError(ValueError):
    """A packed FCS file that is damaged, or not one."""


def _int(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def data_layout(path: str) -> tuple[int, int, int]:
    """``(begin, end, width)`` of the event data in the FCS file at `path`:
    its first and last byte, and each value's width in bytes when every
    parameter has the same 2, 4 or 8 (else 0: the data is not shuffled).
    ``(0, -1, 0)`` when the file has no data segment this can find."""
    from .fcs_text import fcs_text_pairs
    size = os.path.getsize(path)
    try:
        with open(path, 'rb') as fh:
            hdr = fh.read(58)
        begin, end = _int(hdr[26:34]), _int(hdr[34:42])
        keys = {k.upper(): v for k, v in fcs_text_pairs(path)}
    except (OSError, ValueError, IndexError):
        return 0, -1, 0
    if not begin or not end:
        # Too far into the file for the header's 8 digits: in TEXT.
        begin, end = _int(keys.get('$BEGINDATA')), _int(keys.get('$ENDDATA'))
    if begin is None or end is None or not 0 < begin <= end < size:
        return 0, -1, 0
    n = _int(keys.get('$PAR')) or 0
    widths = {_int(keys.get(f'$P{i}B')) for i in range(1, n + 1)}
    one = widths.pop() if len(widths) == 1 else None
    width = one // 8 if one else 0
    if width not in (2, 4, 8) or keys.get('$DATATYPE', '').upper() == 'A':
        width = 0
    return begin, end, width


def _shuffle(block: bytes, width: int) -> bytes:
    import numpy as np
    whole = len(block) - len(block) % width
    a = np.frombuffer(block, np.uint8, whole).reshape(-1, width)
    return np.ascontiguousarray(a.T).tobytes() + block[whole:]


def _unshuffle(block: bytes, width: int) -> bytes:
    import numpy as np
    whole = len(block) - len(block) % width
    a = np.frombuffer(block, np.uint8, whole).reshape(width, -1)
    return np.ascontiguousarray(a.T).tobytes() + block[whole:]


def _pieces(begin: int, end: int, size: int, width: int):
    """``(start, stop, shuffled)`` spans covering a file of `size` bytes:
    before the data, the data in blocks, after it."""
    if not width:
        yield 0, size, False
        return
    yield 0, begin, False
    for s in range(begin, end + 1, BLOCK):
        yield s, min(s + BLOCK, end + 1), True
    yield end + 1, size, False


def pack_fcs(src: str, out: IO[bytes]) -> dict[str, Any]:
    """Write the FCS file `src` packed (see the module docstring) to the
    open binary file `out`; returns its header."""
    size = os.path.getsize(src)
    begin, end, width = data_layout(src)
    crc = 0
    with open(src, 'rb') as fh:
        while chunk := fh.read(CHUNK):
            crc = zlib.crc32(chunk, crc)
    head = {'size': size, 'crc32': crc, 'begin': begin, 'end': end,
            'width': width}
    raw = json.dumps(head).encode()
    out.write(MAGIC + struct.pack('<I', len(raw)) + raw)
    comp = zlib.compressobj(LEVEL)
    with open(src, 'rb') as fh:
        for start, stop, shuffled in _pieces(begin, end, size, width):
            fh.seek(start)
            left = stop - start
            while left > 0:
                piece = fh.read(min(left, BLOCK))
                if not piece:
                    raise FcsPackError(f"{src} changed while it was packed.")
                left -= len(piece)
                out.write(comp.compress(_shuffle(piece, width) if shuffled
                                        else piece))
    out.write(comp.flush())
    return head


def _read_head(src: IO[bytes]) -> dict[str, Any]:
    pre = src.read(8)
    if len(pre) != 8 or pre[:4] != MAGIC:
        raise FcsPackError("not a packed FCS file")
    n = struct.unpack('<I', pre[4:])[0]
    if n > MAX_HEADER:
        raise FcsPackError("packed FCS header too large")
    try:
        head = json.loads(src.read(n).decode('utf-8'))
        size, crc = int(head['size']), int(head['crc32'])
        begin, end, width = (int(head['begin']), int(head['end']),
                             int(head['width']))
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise FcsPackError("packed FCS header unreadable") from exc
    if size < 0 or width not in (0, 2, 4, 8) or (
            width and not 0 <= begin <= end < size):
        raise FcsPackError("packed FCS header out of range")
    return {'size': size, 'crc32': crc, 'begin': begin, 'end': end,
            'width': width}


def _inflate(src: IO[bytes], dec: Any, want: int) -> bytes:
    """The next `want` bytes out of `dec`, fed from `src` as it needs."""
    buf = bytearray()
    while len(buf) < want:
        if dec.unconsumed_tail:
            data = dec.unconsumed_tail
        else:
            data = src.read(CHUNK)
            if not data:
                raise FcsPackError("packed FCS file is cut short")
        try:
            buf += dec.decompress(data, want - len(buf))
        except zlib.error as exc:
            raise FcsPackError("packed FCS data damaged") from exc
    return bytes(buf)


def unpack_fcs(src: IO[bytes], out: IO[bytes],
               declared: int | None = None) -> int:
    """Write the original FCS file packed in the open binary file `src` to
    `out`; returns its size. Raises ``FcsPackError`` if it is damaged, is
    not the size `declared` (the archive's manifest says), would grow past
    it, or its CRC-32 does not match: nothing more than its size is ever
    written."""
    head = _read_head(src)
    size = head['size']
    if declared is not None and size != declared:
        raise FcsPackError("packed FCS size differs from the manifest's")
    dec = zlib.decompressobj()
    crc, done = 0, 0
    for start, stop, shuffled in _pieces(head['begin'], head['end'], size,
                                         head['width']):
        left = stop - start
        while left > 0:
            # A shuffled block is gathered whole to unshuffle it; the rest
            # streams through in CHUNK slices, never the whole file at once.
            n = left if shuffled else min(left, CHUNK)
            piece = _inflate(src, dec, n)
            if shuffled:
                piece = _unshuffle(piece, head['width'])
            crc = zlib.crc32(piece, crc)
            out.write(piece)
            done += n
            left -= n
    rest = dec.unconsumed_tail + src.read(CHUNK)
    try:
        more = dec.decompress(rest, 1) if rest and not dec.eof else b''
    except zlib.error as exc:
        raise FcsPackError("packed FCS data damaged") from exc
    if done != size or more or dec.unused_data or src.read(1):
        raise FcsPackError("packed FCS file holds more than its size")
    if not dec.eof:
        raise FcsPackError("packed FCS file is cut short")
    if crc != head['crc32']:
        raise FcsPackError("packed FCS file damaged (checksum differs)")
    return done
