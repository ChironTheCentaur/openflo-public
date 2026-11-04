"""An FCS file's TEXT keywords in file order (``fcs_text_pairs``) and its
seconds per Time tick (``fcs_timestep``), the Time units a FlowJo .wsp holds
(``wsp_time_gain``, ``scale_time_coords``), and the small value helpers they
share (``positive_float``, ``xml_safe``).

Core module: numpy. No Tk, and it does not import ``openflo.pipeline``
(pipeline imports it).
"""

import os

import numpy as np

from . import fcs_read as _fcs_read
from .errors import note
from .fcs_quirks import read_fcs_tolerant
from .scales import OPEN

# ── FCS TEXT and Time units at the .wsp boundary ─────────────────────────────
#
# OpenFlo keeps the Time channel as the FCS stores it: raw ticks. FlowJo
# 10.10.2 holds Time in SECONDS (measured with synthetic files that differ
# only in that keyword): seconds = ticks x $TIMESTEP, with $TIMESTEP read from
# the FCS file -- a different value in the workspace <Keywords> is ignored.
# A file with no $TIMESTEP gets ($ETIM - $BTIM) / last tick. So the writer
# multiplies Time coordinates by that factor and the reader divides by it.

def positive_float(v):
    """Return `v` as a float when it is finite and above zero, else None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) and f > 0 else None


def fcs_text_pairs(path):
    """The FCS TEXT segment as (key, value) pairs in file order, each key in
    its original case with its '$' kept: the form FlowJo 10 writes into a
    sample's <Keywords>. flowio cannot supply this (it lower-cases keys and
    strips every '$'). A doubled delimiter is an escaped literal delimiter."""
    with open(path, 'rb') as fh:
        hdr = fh.read(58)
        t0, t1 = int(hdr[10:18]), int(hdr[18:26])
        fh.seek(t0)
        raw = fh.read(t1 - t0 + 1)
    try:
        s = raw.decode('utf-8')
    except UnicodeDecodeError:
        s = raw.decode('latin-1')
    delim, parts, cur, i = s[0], [], [], 1
    while i < len(s):
        c = s[i]
        if c == delim:
            if i + 1 < len(s) and s[i + 1] == delim:     # escaped delimiter
                cur.append(delim)
                i += 2
                continue
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    if cur:
        parts.append(''.join(cur))
    # strict=False: a malformed TEXT with a trailing key and no value drops it
    return list(zip(parts[0::2], parts[1::2], strict=False))


def xml_safe(v):
    """Drop the characters XML 1.0 cannot carry (C0 controls other than tab,
    newline and CR); an FCS TEXT value may contain them."""
    return ''.join(ch for ch in v
                   if (ch in '\t\n\r' or ord(ch) >= 0x20)
                   and ch not in '\ufffe\uffff')


def _clock_seconds(v):
    """$BTIM/$ETIM -> seconds since midnight. FCS 3.1 writes hh:mm:ss[.cc],
    FCS 3.0 hh:mm:ss[:tt] with tt in 1/60 s. None if unparseable."""
    try:
        f = str(v).strip().split(':')
        sec = int(f[0]) * 3600 + int(f[1]) * 60 + float(f[2])
        if len(f) == 4:
            sec += int(f[3]) / 60.0
        return sec
    except (ValueError, IndexError):
        return None


def fcs_timestep(path, pairs=None):
    """Seconds per Time tick for the FCS at `path`, as FlowJo derives it:
    $TIMESTEP; failing that ($ETIM - $BTIM) / last Time tick. None when the
    file is missing or neither is usable. `pairs` = its fcs_text_pairs."""
    if not path or not os.path.isfile(path):
        return None
    try:
        kw = {k.upper(): v for k, v in (pairs if pairs is not None
                                         else fcs_text_pairs(path))}
    except (OSError, ValueError, IndexError):
        return None
    ts = positive_float(kw.get('$TIMESTEP'))
    if ts:
        return ts
    b, e = _clock_seconds(kw.get('$BTIM')), _clock_seconds(kw.get('$ETIM'))
    if b is None or e is None:
        return None
    span = e - b if e > b else e - b + 86400.0     # acquisition over midnight
    try:
        fd = read_fcs_tolerant(path, _fcs_read.FlowData, [])
        names = [fd.channels[k]['pnn'] for k in sorted(fd.channels, key=int)]
        ti = [n.lower() for n in names].index('time')
        if fd.events is None:
            raise ValueError('no DATA segment')
        events = np.asarray(fd.events, dtype=float).reshape(-1, fd.channel_count)
        last = float(events[:, ti].max())
    except Exception as exc:  # noqa: BLE001 - noted; an unreadable time channel gives no timestep (None)
        note("reading the time channel from an FCS file", exc)
        return None
    return positive_float(span / last) if last > 0 else None


def wsp_time_gain(sample_elem):
    """The `gain` of the linear axis a (namespace-stripped) FlowJo <Sample>
    declares for its Time parameter in <Transformations>: the seconds per
    tick FlowJo shows Time in. None when there is no such axis, or its gain
    is missing, not a positive number, or 1 (which says nothing about
    seconds; the FCS decides, see WspReader.sample_timestep)."""
    if sample_elem is None:
        return None
    for block in sample_elem.findall('Transformations'):
        for t in block:
            p = t.find('parameter')
            if t.tag != 'linear' or p is None or (p.get('name') or '').lower() != 'time':
                continue
            g = positive_float(t.get('gain'))
            return g if g and g != 1.0 else None
    return None


def scale_time_coords(g, factor):
    """Copy of gate dict `g` with every coordinate on a Time axis multiplied
    by `factor`; `g` itself when there is no factor or no Time axis. The
    ±1e12 open-bound sentinel stays the sentinel."""
    if not factor or factor == 1.0:
        return g

    def k(ch):
        # Exact 'Time': TIME_MSW and TIME_LSW would both be scaled.
        return factor if isinstance(ch, str) and ch.lower() == 'time' else 1.0

    def sc(v, f):
        v = float(v)
        return v if f == 1.0 or abs(v) >= OPEN else v * f

    if 'channel' in g:
        f = k(g['channel'])
        if f == 1.0:
            return g
        g = dict(g)
        for key in ('value', 'lo', 'hi'):
            if g.get(key) is not None:
                g[key] = sc(g[key], f)
        return g
    fx, fy = k(g.get('x_channel')), k(g.get('y_channel'))
    if fx == 1.0 and fy == 1.0:
        return g
    g = dict(g)
    for key, f in (('x0', fx), ('x1', fx), ('quad_origin_x', fx),
                   ('y0', fy), ('y1', fy), ('quad_origin_y', fy)):
        if g.get(key) is not None:
            g[key] = sc(g[key], f)
    if g.get('vertices') is not None:
        g['vertices'] = [[sc(vx, fx), sc(vy, fy)] for vx, vy in g['vertices']]
    if g.get('mean') is not None:
        g['mean'] = [sc(g['mean'][0], fx), sc(g['mean'][1], fy)]
    if g.get('cov') is not None:
        f2 = (fx, fy)
        g['cov'] = [[float(g['cov'][i][j]) * f2[i] * f2[j] for j in range(2)]
                    for i in range(2)]
    return g
