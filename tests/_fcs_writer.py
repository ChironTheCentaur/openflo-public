"""A minimal FCS 3.1 writer for tests that need what flowio.create_fcs
cannot write: integer ($DATATYPE I) data, and any $PnE / $PnG / $PnR.
Synthetic data only."""
from __future__ import annotations

import numpy as np


def write_fcs(path, data, names, *, datatype='I', bits=16, pne=None,
              png=None, pnr=None, extra=None, drop=()):
    """Write `data` (events x channels) to `path` as a little-endian list
    mode FCS 3.1 file. `pne`, `png` and `pnr` are per-channel lists (None
    keeps the default: linear '0,0', no gain, 1024 for integer data);
    `extra` adds or overrides keywords and `drop` leaves keywords out."""
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
    header = b'FCS3.1    ' + b''.join(
        str(v).rjust(8).encode() for v in
        (58, end_text, begin, begin + len(raw) - 1, 0, 0))
    with open(path, 'wb') as fh:
        fh.write(header + txt + b' ' * (begin - end_text - 1) + raw)
    return path
