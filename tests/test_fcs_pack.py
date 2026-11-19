"""The packed FCS form of project archives (openflo.fcs_pack): byte for
byte back, smaller where the events line up, and every damaged or
oversized packed file refused. Generated data only."""
from __future__ import annotations

import io
import json
import os
import struct
import zlib

import numpy as np
import pytest

from openflo import fcs_pack as fp
from tests._fcs_writer import write_fcs

NAMES = ['FSC-A', 'SSC-A', 'FL1-A']


def _fcs(tmp_path, name='a.fcs', datatype='I', bits=16, n=5000, **kw):
    rng = np.random.default_rng(len(name) + bits)
    ev = (rng.lognormal(6, 1, (n, 3)) if datatype in 'FD'
          else rng.integers(0, 1000, (n, 3)).astype(float))
    return write_fcs(str(tmp_path / name), ev, NAMES, datatype=datatype,
                     bits=bits, **kw)


def _pack(path):
    b = io.BytesIO()
    head = fp.pack_fcs(path, b)
    return head, b.getvalue()


def _unpack(packed, declared=None):
    out = io.BytesIO()
    fp.unpack_fcs(io.BytesIO(packed), out, declared)
    return out.getvalue()


@pytest.mark.parametrize('datatype,bits,width', [
    ('I', 16, 2), ('I', 32, 4), ('F', 32, 4), ('D', 64, 8)])
def test_every_value_width_comes_back_byte_for_byte(tmp_path, datatype, bits,
                                                    width):
    p = _fcs(tmp_path, datatype=datatype, bits=bits)
    head, packed = _pack(p)
    assert head['width'] == width
    raw = open(p, 'rb').read()
    assert raw[head['begin']:head['end'] + 1]          # the event data
    assert head['end'] - head['begin'] + 1 == 5000 * 3 * width
    assert _unpack(packed, len(raw)) == raw


def test_shuffling_packs_events_smaller_than_plain_deflate(tmp_path):
    p = _fcs(tmp_path, datatype='F', bits=32, n=40000)
    raw = open(p, 'rb').read()
    _head, packed = _pack(p)
    assert len(packed) < 0.95 * len(zlib.compress(raw, fp.LEVEL))


def test_data_spanning_many_blocks_and_an_odd_tail(tmp_path, monkeypatch):
    # Blocks smaller than the data, of a size that splits values: still
    # whole values per block, and the bytes after the data come back.
    monkeypatch.setattr(fp, 'BLOCK', 1000)
    p = _fcs(tmp_path, datatype='F', bits=32, n=2000)
    with open(p, 'ab') as fh:
        fh.write(b'trailing bytes after the data segment')
    head, packed = _pack(p)
    assert head['width'] == 4
    assert _unpack(packed) == open(p, 'rb').read()


def test_mixed_widths_or_no_data_are_packed_unshuffled(tmp_path):
    p = _fcs(tmp_path, bits=16, extra={'$P2B': '32'})
    assert fp.data_layout(p)[2] == 0
    head, packed = _pack(p)
    assert head['width'] == 0
    assert _unpack(packed) == open(p, 'rb').read()
    junk = tmp_path / 'junk.fcs'
    junk.write_bytes(b'not an fcs file at all')
    assert fp.data_layout(str(junk)) == (0, -1, 0)
    assert _unpack(_pack(str(junk))[1]) == junk.read_bytes()


def test_data_offsets_too_large_for_the_header_are_read_from_text(tmp_path):
    p = _fcs(tmp_path)
    begin, end, width = fp.data_layout(p)
    raw = bytearray(open(p, 'rb').read())
    raw[26:42] = b'       0       0'                   # header says 0, 0
    text = raw[int(raw[10:18]):int(raw[18:26]) + 1].decode('latin-1')
    assert f'$BEGINDATA{text[0]}' in text.upper()
    (tmp_path / 'big.fcs').write_bytes(bytes(raw))
    assert fp.data_layout(str(tmp_path / 'big.fcs')) == (begin, end, width)


def _corrupt(packed, at, value=None):
    b = bytearray(packed)
    b[at] = (b[at] ^ 0xFF) if value is None else value
    return bytes(b)


def test_damaged_packed_files_are_refused(tmp_path):
    p = _fcs(tmp_path, n=20000)
    size = os.path.getsize(p)
    _head, packed = _pack(p)
    start = 8 + struct.unpack('<I', packed[4:8])[0]
    cases = {
        'magic': (b'XXXX' + packed[4:], 'not a packed'),
        'cut': (packed[:-50], 'cut short'),
        'extra': (packed + b'\x00' * 10, 'more than its size'),
        'body': (_corrupt(packed, start + len(packed[start:]) // 2),
                 'damaged'),
        'declared': (packed, 'differs from the manifest'),
    }
    for what, (data, match) in cases.items():
        with pytest.raises(fp.FcsPackError, match=match):
            _unpack(data, size - 1 if what == 'declared' else size)


def test_a_header_that_lies_is_refused_before_writing(tmp_path):
    p = _fcs(tmp_path)
    head, packed = _pack(p)
    body = packed[8 + struct.unpack('<I', packed[4:8])[0]:]

    def with_head(h):
        raw = json.dumps(h).encode()
        return fp.MAGIC + struct.pack('<I', len(raw)) + raw + body
    for bad, match in (({**head, 'width': 3}, 'out of range'),
                       ({**head, 'begin': head['size']}, 'out of range'),
                       ({**head, 'size': -1}, 'out of range'),
                       ({'size': 1}, 'unreadable'),
                       ({**head, 'crc32': head['crc32'] ^ 1}, 'checksum'),
                       ({**head, 'size': head['size'] + 5}, 'cut short')):
        out = io.BytesIO()
        with pytest.raises(fp.FcsPackError, match=match):
            fp.unpack_fcs(io.BytesIO(with_head(bad)), out)
        assert out.tell() <= head['size']
    huge = fp.MAGIC + struct.pack('<I', fp.MAX_HEADER + 1)
    with pytest.raises(fp.FcsPackError, match='too large'):
        _unpack(huge)


def test_a_packed_file_never_writes_past_its_size(tmp_path):
    # Its stream holds more than its header says: the unpack stops at the
    # size and refuses, having written no more than that.
    p = _fcs(tmp_path)
    head, _packed = _pack(p)
    raw = open(p, 'rb').read()
    body = zlib.compress(raw + b'x' * 100_000, fp.LEVEL)
    h = json.dumps({**head, 'width': 0}).encode()
    out = io.BytesIO()
    with pytest.raises(fp.FcsPackError, match='more than its size'):
        fp.unpack_fcs(io.BytesIO(fp.MAGIC + struct.pack('<I', len(h)) + h
                                 + body), out)
    assert out.tell() == len(raw)


def test_a_stream_that_never_ends_is_refused(tmp_path):
    # Every byte there, but the deflate stream not closed: a file cut at a
    # flush point.
    p = _fcs(tmp_path)
    head, _packed = _pack(p)
    raw = open(p, 'rb').read()
    c = zlib.compressobj(fp.LEVEL)
    body = c.compress(raw) + c.flush(zlib.Z_SYNC_FLUSH)
    h = json.dumps({**head, 'width': 0}).encode()
    with pytest.raises(fp.FcsPackError, match='cut short'):
        _unpack(fp.MAGIC + struct.pack('<I', len(h)) + h + body)


def test_ascii_data_is_never_shuffled(tmp_path):
    # $DATATYPE A with 16-bit-looking widths: characters, not numbers.
    p = _fcs(tmp_path, extra={'$DATATYPE': 'A'})
    assert fp.data_layout(p)[2] == 0


def test_an_unshuffled_file_unpacks_without_holding_it_whole(tmp_path):
    # Width 0 (here: no FCS layout at all) is one piece the size of the
    # file; it must stream through, not be gathered in memory.
    import tracemalloc
    big = tmp_path / 'plain.bin'
    big.write_bytes(bytes(range(256)) * (16 << 12))     # 16 MiB
    packed = tmp_path / 'plain.ofz'
    with open(packed, 'wb') as fh:
        assert fp.pack_fcs(str(big), fh)['width'] == 0
    with open(packed, 'rb') as src, open(tmp_path / 'back', 'wb') as out:
        tracemalloc.start()
        try:
            fp.unpack_fcs(src, out)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert (tmp_path / 'back').read_bytes() == big.read_bytes()
    assert peak < 4 * fp.CHUNK, peak
