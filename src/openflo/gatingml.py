"""Standalone Gating-ML 2.0 import and export.

Gating-ML 2.0 is the ISAC standard for exchanging gates between tools
(Spidlen et al., Cytometry A 2015). OpenFlo already speaks it inside a FlowJo
``.wsp``; this module reads and writes the standalone ``.xml`` form, one gate
tree per file.

Units. OpenFlo keeps gate coordinates on each channel's stored (display)
scale. A standalone file says which scale each dimension is on with a
``transformation-ref``:

* Export writes every coordinate on the channel's own scale, with a
  transformation that reproduces it (logicle and hyperlog as they are, asinh
  as ``fasinh``, log as ``flog``), so a polygon's edges stay straight on the
  axes it was drawn on. A linear channel has no transformation. A channel of
  unknown scale has no transform to name: its coordinates are converted to
  linear values with a caller-supplied function (see
  ``pipeline.rank_matched_fn``), and the export warns, as the FlowJo export
  does; without one the gate is left out and the export says so.
* Import returns every coordinate in linear data units (the same as
  ``WspReader.extract_gates``), so callers put them on their own stored
  scale with ``pipeline.gate_scale_fns(..., 'from_linear')``. Vertices are
  converted one by one, so an imported polygon keeps its Gating-ML vertices.
  An ellipsoid on a transformed axis is not an ellipse on any other scale;
  it comes back as a 72-vertex polygon traced on its own axes.

Compensation. A dimension's ``compensation-ref`` names a ``spectrumMatrix``
in the file, ``FCS`` (the file's own spillover) or ``uncompensated``. Export
writes the sample's matrix when it has one. Import records the compensated
channels as ``wsp_comp_channels`` (see ``pipeline.wsp_uncompensated_channels``)
and warns when a gate is on uncompensated values or on a matrix carried in the
file, since OpenFlo gates each sample's own compensated data.
"""
from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

log = logging.getLogger(__name__)

NS = {
    'gating':     'http://www.isac-net.org/std/Gating-ML/v2.0/gating',
    'transforms': 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations',
    'data-type':  'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes',
}

# OpenFlo writes a missing rectangle side as this sentinel (JSON-safe, unlike
# inf; see pipeline._finite_bound). Gating-ML writes an open side by leaving
# the bound out.
_OPEN = 1e12

# Sub-segments per polygon edge when a Gating-ML polygon goes onto another
# scale. Near zero a logicle axis is close to linear and far out it is
# logarithmic, so one straight edge bends hard; 16 (the editor's
# pipeline._POLYGON_EDGE_STEPS) missed 0.7% of a test triangle's events, 64
# misses under 0.1%.
_EDGE_STEPS = 64



def _q(prefix, local):
    return f'{{{NS[prefix]}}}{local}'


# ── Transforms ───────────────────────────────────────────────────────────────
# Gating-ML 2.0 parameterised scales (Transformations spec, section 6). Each
# entry: forward (linear -> scale) and inverse (scale -> linear).

def _fasinh(x, T, M, A):
    ln10 = math.log(10.0)
    return ((np.arcsinh(np.asarray(x, float) * math.sinh(M * ln10) / T)
             + A * ln10) / ((M + A) * ln10))


def _fasinh_inv(y, T, M, A):
    ln10 = math.log(10.0)
    return T * np.sinh((np.asarray(y, float) * (M + A) - A) * ln10) \
        / math.sinh(M * ln10)


def _gml_inverse(spec):
    """``f(scale values) -> linear values`` for a parsed transformation."""
    kind, p = spec['kind'], spec['params']
    if kind == 'flin':
        return lambda y: np.asarray(y, float) * (p['T'] + p['A']) - p['A']
    if kind == 'flog':
        return lambda y: p['T'] * np.power(
            10.0, (np.asarray(y, float) - 1.0) * p['M'])
    if kind == 'fasinh':
        return lambda y: _fasinh_inv(y, p['T'], p['M'], p['A'])
    if kind in ('logicle', 'hyperlog'):
        from .pipeline import inverse_transform_values
        return lambda y: inverse_transform_values(
            np.atleast_1d(np.asarray(y, float)), method=kind,
            t=p['T'], m=p['M'], w=p['W'], a=p['A'])
    raise ValueError(f'unsupported transformation {kind!r}')


def _export_scale(spec):
    """How a channel's stored coordinates go to Gating-ML:
    ``(transformation element or None, factor)`` where the Gating-ML value is
    ``factor * stored``. None for a linear channel (no transformation-ref);
    raises ValueError for a scale Gating-ML cannot name."""
    from .pipeline import transform_params
    params = transform_params(spec if isinstance(spec, dict)
                              else {'method': spec})
    method = params.get('method') or 'linear'
    if method == 'linear':
        return None, 1.0
    if method in ('logicle', 'hyperlog'):
        return ((method, {'T': params.get('t', 262144), 'W': params.get('w', 0.5),
                          'M': params.get('m', 4.5), 'A': params.get('a', 0)}),
                1.0)
    if method == 'asinh':
        # OpenFlo's asinh(x / c). fasinh with A = 0, M = 1 and
        # T = c * sinh(ln 10) is asinh(x / c) / ln 10.
        c = float(params.get('cofactor', 150.0))
        return (('fasinh', {'T': c * math.sinh(math.log(10.0)), 'M': 1.0,
                            'A': 0.0}), 1.0 / math.log(10.0))
    if method == 'log':
        # OpenFlo's log10(x). flog with T = 10, M = 1 is log10(x / 10) + 1.
        return ('flog', {'T': 10.0, 'M': 1.0}), 1.0
    raise ValueError(f'no Gating-ML scale for {method!r}')


# ── Export ───────────────────────────────────────────────────────────────────

_NCNAME_BAD = re.compile(r'[^A-Za-z0-9_.\-]')


def _gate_ids(gates):
    """A unique xs:ID per gate, from its name where it has one."""
    out, used = {}, set()
    for i, g in enumerate(gates, 1):
        name = str(g.get('name') or g.get('label') or f'gate{i}')
        base = _NCNAME_BAD.sub('_', name.strip()) or f'gate{i}'
        if not re.match(r'[A-Za-z_]', base):
            base = f'g_{base}'
        gid, n = base, 2
        while gid in used:
            gid, n = f'{base}_{n}', n + 1
        used.add(gid)
        out[g['id']] = gid
    return out


def _fmt(v):
    return repr(float(v))


def write_gatingml(path, gates, transforms=None, compensation=None,
                   unknown_fns=None):
    """Write `gates` (one sample's tree, parents before children, each with
    ``id`` / ``parent_id``) as a standalone Gating-ML 2.0 file at `path`.

    `transforms` is the sample's ``{channel: spec}`` stored-scale record,
    `compensation` its ``(channels, spillover matrix)`` or None, and
    `unknown_fns` ``{channel: f(stored) -> linear}`` for channels of unknown
    scale (see the module docstring). Returns the warnings, one string per
    place another tool will not count what OpenFlo counts."""
    from .pipeline import is_unknown_spec
    transforms = dict(transforms or {})
    unknown_fns = dict(unknown_fns or {})
    warnings = []
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)
    root = ET.Element(_q('gating', 'Gating-ML'))

    comp_chans = set()
    if compensation is not None:
        c_chans, matrix = compensation
        c_chans = list(c_chans)
        matrix = np.asarray(matrix, float)
        comp_chans = set(c_chans)
        sm = ET.SubElement(root, _q('transforms', 'spectrumMatrix'),
                           {_q('transforms', 'id'): 'Spill'})
        for tag in ('fluorochromes', 'detectors'):
            el = ET.SubElement(sm, _q('transforms', tag))
            for ch in c_chans:
                ET.SubElement(el, _q('data-type', 'fcs-dimension'),
                              {_q('data-type', 'name'): ch})
        for row in matrix:
            sp = ET.SubElement(sm, _q('transforms', 'spectrum'))
            for v in row:
                ET.SubElement(sp, _q('transforms', 'coefficient'),
                              {_q('transforms', 'value'): _fmt(v)})

    # One transformation per distinct scale, named after its first channel.
    xform_ids, xform_els = {}, []
    scale = {}          # channel -> (ref or None, factor) | ('unknown', f)

    def channel_scale(ch):
        if ch in scale:
            return scale[ch]
        spec = transforms.get(ch)
        if spec is not None and is_unknown_spec(
                spec if isinstance(spec, dict) else {'method': spec}):
            scale[ch] = ('unknown', unknown_fns.get(ch))
            return scale[ch]
        xf, factor = _export_scale(spec or 'linear')
        ref = None
        if xf is not None:
            key = (xf[0], tuple(sorted(xf[1].items())))
            if key not in xform_ids:
                tid = _NCNAME_BAD.sub('_', f'{xf[0]}_{ch}')
                xform_ids[key] = tid
                xform_els.append((tid, xf))
            ref = xform_ids[key]
        scale[ch] = (ref, factor)
        return scale[ch]

    def coord(ch, v):
        ref, f = channel_scale(ch)
        if ref == 'unknown':
            return float(np.asarray(f(np.atleast_1d(float(v))))[0])
        return float(v) * f

    ids = _gate_ids(gates)
    skipped = set()
    gate_els = []
    for g in gates:
        gname = g.get('name') or g.get('label') or ids[g['id']]
        kind = g.get('kind')
        pid = g.get('parent_id')
        if pid in skipped:
            skipped.add(g['id'])
            warnings.append(f"'{gname}' left out: its parent was left out")
            continue
        chans = [g[k] for k in ('channel', 'x_channel', 'y_channel') if g.get(k)]
        unk = [c for c in chans if channel_scale(c)[0] == 'unknown']
        if unk and (kind == 'ellipsoid'
                    or any(channel_scale(c)[1] is None for c in unk)):
            skipped.add(g['id'])
            warnings.append(f"'{gname}' left out: {', '.join(unk)} has an "
                            "unknown scale that could not be matched to the "
                            "FCS file")
            continue
        if unk:
            warnings.append(f"'{gname}': {', '.join(unk)} has an unknown "
                            "scale; written in linear values matched to the "
                            "FCS file, check it in the other tool")
        attrs = {_q('gating', 'id'): ids[g['id']]}
        if pid is not None and pid in ids:
            attrs[_q('gating', 'parent_id')] = ids[pid]
        el = _gate_element(g, kind, attrs, ids, coord)
        if el is None:
            skipped.add(g['id'])
            warnings.append(f"'{gname}' left out: Gating-ML has no "
                            f"{kind!r} gate")
            continue
        info = ET.SubElement(el, _q('data-type', 'custom_info'))
        ET.SubElement(info, 'openflo', {'name': str(gname)})
        # custom_info must come first in a gate.
        el.remove(info)
        el.insert(0, info)
        for dim in el.iter(_q('gating', 'dimension')):
            fd = dim.find(_q('data-type', 'fcs-dimension'))
            ch = fd.get(_q('data-type', 'name')) if fd is not None else None
            dim.set(_q('gating', 'compensation-ref'),
                    'Spill' if ch in comp_chans else 'uncompensated')
            ref = channel_scale(ch)[0]
            if ref and ref != 'unknown':
                dim.set(_q('gating', 'transformation-ref'), ref)
        gate_els.append(el)

    for tid, (kind, params) in xform_els:
        t = ET.SubElement(root, _q('transforms', 'transformation'),
                          {_q('transforms', 'id'): tid})
        ET.SubElement(t, _q('transforms', kind),
                      {_q('transforms', k): _fmt(v) for k, v in params.items()})
    root.extend(gate_els)
    if hasattr(ET, 'indent'):
        ET.indent(root, space='  ')
    body = ET.tostring(root, encoding='unicode')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + body + '\n')
    for w in warnings:
        log.warning('[Gating-ML export] %s', w)
    return warnings


def _dimension(parent, ch, lo=None, hi=None):
    attrs = {}
    if lo is not None:
        attrs[_q('gating', 'min')] = _fmt(lo)
    if hi is not None:
        attrs[_q('gating', 'max')] = _fmt(hi)
    d = ET.SubElement(parent, _q('gating', 'dimension'), attrs)
    ET.SubElement(d, _q('data-type', 'fcs-dimension'),
                  {_q('data-type', 'name'): ch})
    return d


def _bound(v, lo_side):
    """None for OpenFlo's open-side sentinel, else the value."""
    if v is None:
        return None
    v = float(v)
    if (lo_side and v <= -_OPEN) or (not lo_side and v >= _OPEN):
        return None
    return v


def _gate_element(g, kind, attrs, ids, coord):
    if kind == 'threshold':
        el = ET.Element(_q('gating', 'RectangleGate'), attrs)
        _dimension(el, g['channel'], lo=coord(g['channel'], g['value']))
        return el
    if kind == 'interval':
        ch = g['channel']
        lo, hi = _bound(g.get('lo'), True), _bound(g.get('hi'), False)
        el = ET.Element(_q('gating', 'RectangleGate'), attrs)
        _dimension(el, ch, lo=None if lo is None else coord(ch, lo),
                   hi=None if hi is None else coord(ch, hi))
        return el
    if kind == 'rect':
        el = ET.Element(_q('gating', 'RectangleGate'), attrs)
        for ch, a, b in ((g['x_channel'], 'x0', 'x1'),
                         (g['y_channel'], 'y0', 'y1')):
            lo, hi = _bound(g.get(a), True), _bound(g.get(b), False)
            _dimension(el, ch, lo=None if lo is None else coord(ch, lo),
                       hi=None if hi is None else coord(ch, hi))
        return el
    if kind == 'polygon':
        xc, yc = g['x_channel'], g['y_channel']
        el = ET.Element(_q('gating', 'PolygonGate'), attrs)
        _dimension(el, xc)
        _dimension(el, yc)
        for vx, vy in g['vertices']:
            v = ET.SubElement(el, _q('gating', 'vertex'))
            for ch, val in ((xc, vx), (yc, vy)):
                ET.SubElement(v, _q('gating', 'coordinate'),
                              {_q('data-type', 'value'): _fmt(coord(ch, val))})
        return el
    if kind == 'ellipsoid':
        xc, yc = g['x_channel'], g['y_channel']
        # Stored -> Gating-ML is a per-axis factor here (unknown scales were
        # refused above), so the ellipse maps exactly: mean scaled, the
        # covariance by the same factors on both sides.
        k = np.array([coord(xc, 1.0), coord(yc, 1.0)])
        mean = np.asarray(g['mean'], float) * k
        cov = np.asarray(g['cov'], float) * np.outer(k, k)
        el = ET.Element(_q('gating', 'EllipsoidGate'), attrs)
        _dimension(el, xc)
        _dimension(el, yc)
        m = ET.SubElement(el, _q('gating', 'mean'))
        for v in mean:
            ET.SubElement(m, _q('gating', 'coordinate'),
                          {_q('data-type', 'value'): _fmt(v)})
        cm = ET.SubElement(el, _q('gating', 'covarianceMatrix'))
        for row in cov:
            r = ET.SubElement(cm, _q('gating', 'row'))
            for v in row:
                ET.SubElement(r, _q('gating', 'entry'),
                              {_q('data-type', 'value'): _fmt(v)})
        ET.SubElement(el, _q('gating', 'distanceSquare'),
                      {_q('data-type', 'value'):
                       _fmt(g.get('distance_sq', 4.0))})
        return el
    if kind == 'boolean':
        op = (g.get('op') or 'and').lower()
        operands = [o for o in (g.get('operands') or []) if o in ids]
        if op not in ('and', 'or', 'not') or not operands:
            return None
        el = ET.Element(_q('gating', 'BooleanGate'), attrs)
        o = ET.SubElement(el, _q('gating', op))
        for ref in operands[:1] if op == 'not' else operands:
            ET.SubElement(o, _q('gating', 'gateReference'),
                          {_q('gating', 'ref'): ids[ref]})
        return el
    return None


# ── Import ───────────────────────────────────────────────────────────────────

class GatingMLError(ValueError):
    """The file is not a Gating-ML 2.0 document OpenFlo can read."""


def _local(tag):
    return tag.rsplit('}', 1)[-1]


def _attr(el, name):
    """An attribute by local name, whatever its namespace."""
    for k, v in el.attrib.items():
        if _local(k) == name:
            return v
    return None


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_gatingml(path):
    """True when `path` is an XML file whose root is a Gating-ML element."""
    try:
        for _ev, el in ET.iterparse(path, events=('start',)):
            return _local(el.tag) == 'Gating-ML'
    except (ET.ParseError, OSError):
        return False
    return False


def _parse_root(path):
    with open(path, 'rb') as fh:
        head = fh.read(1 << 20).lower()
    # Same guard as WspReader: no DTD or entity declarations (entity
    # expansion), which a Gating-ML document never needs.
    if b'<!doctype' in head or b'<!entity' in head:
        raise GatingMLError(f'refusing a Gating-ML file with a DTD or entity '
                            f'declaration: {path}')
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise GatingMLError(f'not valid XML ({path}): {e}') from e
    if _local(root.tag) != 'Gating-ML':
        raise GatingMLError(f'not a Gating-ML document (root <{_local(root.tag)}>)')
    return root


def read_gatingml(path, transforms=None, *, raw=False):
    """Read a standalone Gating-ML 2.0 file.

    Returns ``(gates, warnings)``: `gates` in OpenFlo's gate-dict form,
    parents before children, each with an ``_import_id`` and a ``parent_id``
    naming another's (as ``WspReader.extract_gates`` returns), named by
    ``label``; `warnings` lists what was left out or will count differently.
    A gate that cannot be read is left out together with every gate under
    it, as the .wsp import does.

    Coordinates are put on `transforms` (``{channel: spec}``, the stored
    scales of the data the gates go on; None or ``{}`` means linear data
    units) with `gates_to_stored`. ``raw=True`` keeps them on the file's own
    scales instead, for a caller that converts per sample later."""
    from .pipeline import _strip_comp_marker
    root = _parse_root(path)
    warnings = []

    xforms = {}
    for t in root.iter():
        if _local(t.tag) != 'transformation':
            continue
        tid = _attr(t, 'id')
        for fn in t:
            kind = _local(fn.tag)
            params = {k: _num(_attr(fn, k)) for k in ('T', 'W', 'M', 'A')
                      if _attr(fn, k) is not None}
            xforms[tid] = {'kind': kind, 'params': params}
    matrices = {_attr(m, 'id') for m in root.iter()
                if _local(m.tag) == 'spectrumMatrix'}
    used_matrices = set()

    def dim_info(dim):
        fd = next((c for c in dim.iter() if _local(c.tag) == 'fcs-dimension'),
                  None)
        if fd is None:
            raise ValueError('a dimension that is not an FCS parameter '
                             '(ratio or new dimension)')
        ch = _strip_comp_marker(_attr(fd, 'name') or '')[0]
        if not ch:
            raise ValueError('a dimension with no parameter name')
        spec = None
        ref = _attr(dim, 'transformation-ref')
        if ref:
            spec = xforms.get(ref)
            if spec is None:
                raise ValueError(f'transformation {ref!r} is not in the file')
            _gml_inverse(spec)          # ValueError for one OpenFlo can't undo
        comp = _attr(dim, 'compensation-ref') or 'uncompensated'
        if comp in matrices:
            used_matrices.add(comp)
        elif comp not in ('uncompensated', 'FCS'):
            raise ValueError(f'compensation {comp!r} is not in the file')
        return ch, spec, comp != 'uncompensated'

    gate_els = [(_attr(el, 'id'), _attr(el, 'parent_id'), el, _local(el.tag))
                for el in root if _local(el.tag).endswith('Gate')]

    def name_of(gid, el):
        for info in _children(el, 'custom_info'):
            for c in info.iter():
                if _local(c.tag) == 'openflo' and c.get('name'):
                    return c.get('name')
        return gid or 'unnamed'

    def parse(el, tag, gname):
        dim_els = _children(el, 'dimension')
        dims = [dim_info(d) for d in dim_els]
        if tag == 'RectangleGate':
            b = [(ch, _num(_attr(d, 'min')), _num(_attr(d, 'max')))
                 for (ch, _s, _c), d in zip(dims, dim_els, strict=True)]
            if len(b) == 1:
                ch, lo, hi = b[0]
                if lo is not None and hi is None:
                    g = {'kind': 'threshold', 'channel': ch, 'value': lo}
                else:
                    g = {'kind': 'interval', 'channel': ch,
                         'lo': -1e12 if lo is None else lo,
                         'hi': 1e12 if hi is None else hi}
            elif len(b) == 2:
                (xc, x0, x1), (yc, y0, y1) = b
                g = {'kind': 'rect', 'x_channel': xc, 'y_channel': yc,
                     'x0': -1e12 if x0 is None else x0,
                     'x1': 1e12 if x1 is None else x1,
                     'y0': -1e12 if y0 is None else y0,
                     'y1': 1e12 if y1 is None else y1}
            else:
                raise ValueError(f'a {len(b)}-dimensional rectangle')
        elif tag == 'PolygonGate':
            if len(dims) != 2:
                raise ValueError('a polygon needs two dimensions')
            verts = []
            for v in _children(el, 'vertex'):
                cs = [_num(_attr(c, 'value')) for c in _children(v, 'coordinate')]
                if len(cs) != 2 or None in cs:
                    raise ValueError('a vertex without two coordinates')
                verts.append(cs)
            if len(verts) < 3:
                raise ValueError('a polygon with fewer than 3 vertices')
            g = {'kind': 'polygon', 'x_channel': dims[0][0],
                 'y_channel': dims[1][0], 'vertices': verts}
        elif tag == 'EllipsoidGate':
            if len(dims) != 2:
                raise ValueError('an ellipsoid in other than two dimensions')
            mean_el = _children(el, 'mean')
            cov_el = _children(el, 'covarianceMatrix')
            dsq_el = _children(el, 'distanceSquare')
            if not mean_el or not cov_el or not dsq_el:
                raise ValueError('an ellipsoid without mean, covariance or '
                                 'distanceSquare')
            mean = [_num(_attr(c, 'value')) for c in
                    _children(mean_el[0], 'coordinate')]
            cov = [[_num(_attr(e, 'value')) for e in _children(r, 'entry')]
                   for r in _children(cov_el[0], 'row')]
            dsq = _num(_attr(dsq_el[0], 'value'))
            if (len(mean) != 2 or None in mean or len(cov) != 2
                    or any(len(r) != 2 or None in r for r in cov)
                    or dsq is None or dsq <= 0):
                raise ValueError('a malformed ellipsoid')
            if np.any(np.linalg.eigvalsh(np.asarray(cov, float)) <= 0):
                raise ValueError('an ellipsoid whose covariance is not '
                                 'positive definite')
            g = {'kind': 'ellipsoid', 'x_channel': dims[0][0],
                 'y_channel': dims[1][0], 'mean': mean, 'cov': cov,
                 'distance_sq': dsq}
        elif tag == 'BooleanGate':
            ops = [c for c in el if _local(c.tag) in ('and', 'or', 'not')]
            if len(ops) != 1:
                raise ValueError('a boolean gate without one and/or/not')
            refs = _children(ops[0], 'gateReference')
            if any((_attr(r, 'use-as-complement') or '').lower() == 'true'
                   for r in refs):
                raise ValueError('a boolean operand used as its complement')
            g = {'kind': 'boolean', 'op': _local(ops[0].tag),
                 'operands': [_attr(r, 'ref') for r in refs]}
        else:
            raise ValueError(f'a {tag}, which OpenFlo does not read')
        if dims:
            g['_gml_axes'] = {ch: spec for ch, spec, _c in dims}
            # Whether a gate on uncompensated values counts differently
            # depends on the sample it goes on: the caller checks with
            # pipeline.wsp_uncompensated_channels.
            g['wsp_comp_channels'] = sorted({ch for ch, _s, c in dims if c})
        return g

    by_id = {gid: (pid, el, tag) for gid, pid, el, tag in gate_els}
    order, done = [], set()

    def visit(gid, trail=()):
        if gid in done:
            return
        if gid in trail:
            raise GatingMLError(f'gate {gid!r} is its own ancestor')
        pid = by_id[gid][0]
        if pid is not None and pid in by_id:
            visit(pid, trail + (gid,))
        done.add(gid)
        order.append(gid)

    for gid, *_ in gate_els:
        visit(gid)

    gates, imp, dropped = [], {}, set()
    for gid in order:
        pid, el, tag = by_id[gid]
        gname = name_of(gid, el)
        if pid is not None and (pid not in by_id or pid in dropped):
            dropped.add(gid)
            why = ('its parent was left out' if pid in dropped
                   else f'its parent {pid!r} is not in the file')
            warnings.append(f"'{gname}' left out: {why}")
            continue
        try:
            g = parse(el, tag, gname)
        except ValueError as e:
            dropped.add(gid)
            warnings.append(f"'{gname}' left out: {e}")
            continue
        g['label'] = gname
        g['_import_id'] = f'gml_{len(imp)}'
        imp[gid] = g['_import_id']
        g['parent_id'] = imp.get(pid) if pid is not None else None
        gates.append(g)
    if used_matrices:
        warnings.append(f"the file's gates use its compensation matrix "
                        f"{', '.join(sorted(used_matrices))}; OpenFlo gates "
                        "each sample with its own compensation")
    # Boolean operands name Gating-ML ids; point them at the import ids. An
    # operand that was left out empties the boolean, which then reads NA
    # (pipeline.broken_booleans), never every event.
    for g in gates:
        if g['kind'] == 'boolean':
            ops = g['operands']
            if all(o in imp for o in ops):
                g['operands'] = [imp[o] for o in ops]
            else:
                g['operands'] = []
                warnings.append(f"'{g['label']}': an operand was left out, "
                                "so it reads NA")
    for w in warnings:
        log.warning('[Gating-ML import] %s', w)
    if not raw:
        gates = gates_to_stored(gates, transforms)
    return gates, warnings


def _same_scale(file_spec, stored_spec):
    """The factor ``k`` with stored = file / k when the file's scale for a
    channel IS the stored one (so coordinates map by a constant), else
    None."""
    try:
        xf, k = _export_scale(stored_spec or 'linear')
    except ValueError:
        return None
    if xf is None or file_spec is None:
        return k if xf is None and file_spec is None else None
    kind, params = xf
    if kind != file_spec['kind']:
        return None
    got = file_spec['params']
    for key, v in params.items():
        g = got.get(key, 0.0 if key == 'A' else None)
        if g is None or not math.isclose(float(g), float(v), rel_tol=1e-9,
                                         abs_tol=1e-12):
            return None
    return k


def _axis_fn(file_spec, stored_spec):
    """``(f, k)``: `f` maps the file's coordinates on a channel to the stored
    scale `stored_spec`; `k` is the constant factor (stored = file / k) when
    the two scales are one, else None. A stored scale that is unknown takes
    linear values (pipeline.mark_unknown_scale then flags the gate)."""
    from .pipeline import is_unknown_spec, transform_params, transform_values
    spec = stored_spec if isinstance(stored_spec, dict) else (
        {'method': stored_spec} if stored_spec else None)
    if spec is not None and is_unknown_spec(spec):
        spec = None
    k = _same_scale(file_spec, spec)
    if k is not None:
        return (lambda v, _k=k: np.asarray(v, float) / _k), k
    inv = _gml_inverse(file_spec) if file_spec is not None else (
        lambda v: np.asarray(v, float))
    params = transform_params(spec or {})
    if (params.get('method') or 'linear') == 'linear':
        return inv, None
    return (lambda v: np.asarray(transform_values(
        np.atleast_1d(inv(np.atleast_1d(np.asarray(v, float)))),
        **params), float)), None


def gates_to_stored(gates, transforms=None):
    """Gates read with ``read_gatingml(raw=True)`` put on the stored scales
    `transforms` (``{channel: spec}``; None or ``{}`` for linear values).

    Bounds map exactly (every scale is monotone). A polygon or ellipse whose
    axes are on the stored scale already maps by a constant and keeps its
    shape exactly. On another scale, a Gating-ML edge is straight on the
    file's axes, so each polygon edge is subdivided there before mapping,
    and an ellipse is traced as a polygon there; the warning names it."""
    from .pipeline import _ellipse_polygon
    transforms = transforms or {}
    out = []
    for g in gates:
        g = dict(g)
        axes = g.pop('_gml_axes', None)
        if not axes:
            out.append(g)
            continue
        fns = {ch: _axis_fn(spec, transforms.get(ch))
               for ch, spec in axes.items()}

        def m(ch, v, fns=fns):
            if v is None or abs(float(v)) >= 1e12:
                return v
            return float(np.asarray(fns[ch][0](np.atleast_1d(float(v))))[0])

        kind = g['kind']
        if kind == 'threshold':
            g['value'] = m(g['channel'], g['value'])
        elif kind == 'interval':
            g['lo'] = m(g['channel'], g['lo'])
            g['hi'] = m(g['channel'], g['hi'])
        elif kind == 'rect':
            for key, ch in (('x0', 'x_channel'), ('x1', 'x_channel'),
                            ('y0', 'y_channel'), ('y1', 'y_channel')):
                g[key] = m(g[ch], g[key])
        elif kind in ('polygon', 'ellipsoid'):
            (fx, kx), (fy, ky) = fns[g['x_channel']], fns[g['y_channel']]
            if kx is not None and ky is not None:
                if kind == 'ellipsoid':
                    s = np.array([1.0 / kx, 1.0 / ky])
                    g['mean'] = (np.asarray(g['mean'], float) * s).tolist()
                    g['cov'] = (np.asarray(g['cov'], float)
                                * np.outer(s, s)).tolist()
                else:
                    g['vertices'] = [[v[0] / kx, v[1] / ky]
                                     for v in g['vertices']]
            else:
                if kind == 'ellipsoid':
                    v = np.asarray(_ellipse_polygon(
                        g.pop('mean'), g.pop('cov'), g.pop('distance_sq')),
                        float)
                    g['kind'] = 'polygon'
                else:
                    v = np.asarray(g['vertices'], float)
                    t = (np.arange(_EDGE_STEPS) / _EDGE_STEPS)[:, None]
                    v = np.concatenate([a + (b - a) * t for a, b in
                                        zip(v, np.roll(v, -1, axis=0),
                                            strict=True)])
                xs = np.asarray(fx(v[:, 0]), float)
                ys = np.asarray(fy(v[:, 1]), float)
                g['vertices'] = [[float(a), float(b)]
                                 for a, b in zip(xs, ys, strict=True)]
        out.append(g)
    return out
