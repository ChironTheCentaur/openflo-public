"""The FCS writers every test shares. Synthetic data only.

write_float_fcs: float32 list mode through flowio.create_fcs, as most
tests need. write_fcs: a minimal FCS 3.1 writer for what flowio.create_fcs
cannot write: integer ($DATATYPE I) data, and any $PnE / $PnG / $PnR.
tests/test_fcs_writers_shared.py keeps tests from growing their own."""
from __future__ import annotations

import flowio
import numpy as np


def write_float_fcs(path, events, names, **create_kw):
    """Write `events` (events x channels) to `path` as float32 with
    flowio.create_fcs; `create_kw` passes through (opt_channel_names,
    metadata_dict). Returns the path as a string."""
    flat = np.asarray(events, dtype=np.float32).ravel().tolist()
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, flat, names, **create_kw)
    return str(path)


def write_fcs(path, data, names, *, datatype='I', bits=16, pne=None,
              png=None, pnr=None, extra=None, drop=(), version='3.1'):
    """Write `data` (events x channels) to `path` as a little-endian list
    mode FCS file (FCS 3.1 unless `version` says otherwise, e.g. '2.0').
    `pne`, `png` and `pnr` are per-channel lists (None keeps the default:
    linear '0,0', no gain, 1024 for integer data); `extra` adds or overrides
    keywords and `drop` leaves keywords out."""
    data = np.asarray(data)
    n_ev, n_ch = data.shape
    if datatype == 'F':
        dt, bits = '<f4', 32
    elif datatype == 'D':
        dt, bits = '<f8', 64
    else:
        dt = {8: '<u1', 16: '<u2', 32: '<u4'}[bits]
    raw = data.astype(dt).tobytes()
    byteord = {8: '1', 16: '1,2', 32: '1,2,3,4', 64: '1,2,3,4,5,6,7,8'}[bits]
    kw = {'$BYTEORD': byteord, '$DATATYPE': datatype, '$MODE': 'L',
          '$NEXTDATA': '0', '$PAR': str(n_ch), '$TOT': str(n_ev)}
    for i, name in enumerate(names, 1):
        kw[f'$P{i}N'] = name
        kw[f'$P{i}B'] = str(bits)
        kw[f'$P{i}R'] = str(pnr[i - 1] if pnr else
                            (1024 if datatype == 'I' else 262144))
        kw[f'$P{i}E'] = pne[i - 1] if pne else '0,0'
        if png:
            kw[f'$P{i}G'] = str(png[i - 1])
    kw.update(extra or {})
    for key in drop:                         # a keyword a damaged file lacks
        kw.pop(key, None)

    def text(begin, end):
        k = dict(kw, **{'$BEGINDATA': str(begin), '$ENDDATA': str(end),
                        '$BEGINANALYSIS': '0', '$ENDANALYSIS': '0',
                        '$BEGINSTEXT': '0', '$ENDSTEXT': '0'})
        return ('/' + '/'.join(f'{a}/{b}' for a, b in k.items()) + '/'
                ).encode('utf-8')

    begin = 0
    for _ in range(6):                       # settle the self-referring offsets
        nxt = 58 + len(text(begin, begin + len(raw) - 1)) + 8
        if nxt == begin:
            break
        begin = nxt
    txt = text(begin, begin + len(raw) - 1)
    end_text = 58 + len(txt) - 1
    header = f'FCS{version}    '.encode() + b''.join(
        str(v).rjust(8).encode() for v in
        (58, end_text, begin, begin + len(raw) - 1, 0, 0))
    with open(path, 'wb') as fh:
        fh.write(header + txt + b' ' * (begin - end_text - 1) + raw)
    return path


def write_lmd(path, first, second, names, *, first_kw=None,
              second_kw=None):
    """Write a Beckman Coulter style list-mode file to `path`: an FCS 2.0
    data set of 10-bit integer `first` (events x channels), whose $NEXTDATA
    points at an FCS 3.0 data set of float32 `second`. `first_kw` and
    `second_kw` are more write_fcs options for each (e.g. `extra`, `drop`).
    Returns `path`."""
    import pathlib
    path = pathlib.Path(path)
    a = path.with_name(path.name + '.part1')
    b = path.with_name(path.name + '.part2')
    kw1 = dict(first_kw or {})
    kw1['extra'] = {**kw1.get('extra', {}), '$NEXTDATA': '0' * 10}
    write_fcs(a, first, names, version='2.0', **kw1)
    write_fcs(b, second, names, datatype='F', version='3.0',
              **(second_kw or {}))
    raw = a.read_bytes()
    raw = raw.replace(b'$NEXTDATA/' + b'0' * 10 + b'/',
                      b'$NEXTDATA/' + str(len(raw)).rjust(10, '0').encode()
                      + b'/')
    path.write_bytes(raw + b.read_bytes())
    a.unlink()
    b.unlink()
    return path


def write_multi_fcs(path, sets, names, *, kws=None, version='3.1', drop=()):
    """Write an FCS file holding several data sets to `path`: one float32
    data set per array in `sets` (events x channels), each chained to the
    next by $NEXTDATA counted from its own start (the FCS 2.0 rule; for the
    second data set that is also its place in the file). `kws` is one dict
    of extra keywords per data set (e.g. ``{'$FIL': 'A01.fcs'}``); `drop`
    leaves keywords out of every data set (write_fcs). Returns `path`."""
    import pathlib
    path = pathlib.Path(path)
    kws = list(kws or [{}] * len(sets))
    blobs = []
    for i, (data, kw) in enumerate(zip(sets, kws, strict=True)):
        part = path.with_name(f'{path.name}.part{i}')
        write_fcs(part, data, names, datatype='F', version=version,
                  extra={**kw, '$NEXTDATA': '0' * 10}, drop=drop)
        blobs.append(part.read_bytes())
        part.unlink()
    out = b''
    for i, raw in enumerate(blobs):
        nxt = len(raw) if i < len(blobs) - 1 else 0
        out += raw.replace(b'$NEXTDATA/' + b'0' * 10 + b'/',
                           b'$NEXTDATA/' + str(nxt).rjust(10, '0').encode()
                           + b'/')
    path.write_bytes(out)
    return path


def set_nextdata(path, which, value):
    """Overwrite the $NEXTDATA of data set `which` (from 1, in file order of
    the keyword) of the file at `path` with `value` (written as ten digits,
    a leading '-' for a negative one), as a damaged writer leaves it."""
    import pathlib
    import re
    path = pathlib.Path(path)
    raw = path.read_bytes()
    m = list(re.finditer(rb'\$NEXTDATA/(-?\d{10}|-\d{9})/', raw))[which - 1]
    text = (f'-{abs(int(value)):09d}' if int(value) < 0
            else str(int(value)).rjust(10, '0'))
    path.write_bytes(raw[:m.start(1)] + text.encode() + raw[m.end(1):])
    return path
