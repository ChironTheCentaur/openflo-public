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
  ``wsp_gate_coords.rank_matched_fn``), and the export warns, as the FlowJo export
  does; without one the gate is left out and the export says so.
* Import returns every coordinate in linear data units (the same as
  ``WspReader.extract_gates``), so callers put them on their own stored
  scale with ``wsp_gate_coords.gate_scale_fns(..., 'from_linear')``. Vertices are
  converted one by one, so an imported polygon keeps its Gating-ML vertices.
  An ellipsoid on a transformed axis is not an ellipse on any other scale;
  it comes back as a 72-vertex polygon traced on its own axes.

Compensation. A dimension's ``compensation-ref`` names a ``spectrumMatrix``
in the file, ``FCS`` (the file's own spillover) or ``uncompensated``. Export
writes the sample's matrix when it has one. Import records the compensated
channels as ``wsp_comp_channels`` (see ``wsp_gate_coords.wsp_uncompensated_channels``)
and warns when a gate is on uncompensated values or on a matrix carried in the
file, since OpenFlo gates each sample's own compensated data.
"""
from __future__ import annotations

import logging
import math
import re
import warnings
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import numpy as np

from .scales import ASINH_DEFAULT_COFACTOR, LOGICLE_DEFAULTS, OPEN

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import numpy.typing as npt

    _Path = str | os.PathLike[str]
    # An OpenFlo gate dict (see gate_eval) and a gate's or a sample's
    # stored-scale record entry (a transform spec dict, or a bare method).
    _Gate = dict[str, Any]
    _StoredSpec = dict[str, Any] | str
    # A parsed Gating-ML transformation, ``{'kind': ..., 'params': {...}}``
    # (_read_transformations), and the one export writes, ``(kind, params)``.
    _FileSpec = Mapping[str, Any]
    _Xform = tuple[str, dict[str, Any]]
    # A read <dimension>: (channel, file scale spec or None, on a matrix).
    _Dim = tuple[str, _FileSpec | None, bool]
    # (parent id, element, local tag) of each top-level gate element.
    _GateEl = tuple[str | None, ET.Element, str]
    # Maps an array of values from one scale to another.
    _AxisFn = Callable[[npt.ArrayLike], np.ndarray]
    # A caller's ``f(stored) -> linear`` for a channel of unknown scale.
    _UnknownFn = Callable[[np.ndarray], npt.ArrayLike]
    # An export scale: ``(transformation id or None, factor)``, or
    # ``('unknown', f or None)``; the second item is a float factor or an
    # _UnknownFn by the first, which pyright cannot tell apart once unpacked.
    _Scale = tuple[str | None, Any]

log = logging.getLogger(__name__)

NS = {
    'gating':     'http://www.isac-net.org/std/Gating-ML/v2.0/gating',
    'transforms': 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations',
    'data-type':  'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes',
}

# OpenFlo writes a missing rectangle side as the OPEN sentinel (JSON-safe,
# unlike inf; see wsp_reader._finite_bound). Gating-ML writes an open side by
# leaving the bound out.

# Sub-segments per polygon edge when a Gating-ML polygon goes onto another
# scale. Near zero a logicle axis is close to linear and far out it is
# logarithmic, so one straight edge bends hard; 16 (the editor's
# wsp_gate_coords.POLYGON_EDGE_STEPS) missed 0.7% of a test triangle's events, 64
# misses under 0.1%.
_EDGE_STEPS = 64



def _q(prefix: str, local: str) -> str:
    return f'{{{NS[prefix]}}}{local}'


# ── Transforms ───────────────────────────────────────────────────────────────
# Gating-ML 2.0 parameterised scales (Transformations spec, section 6). Each
# entry: forward (linear -> scale) and inverse (scale -> linear).

def _fasinh(x: npt.ArrayLike, T: float, M: float, A: float) -> np.ndarray:
    ln10 = math.log(10.0)
    return ((np.arcsinh(np.asarray(x, float) * math.sinh(M * ln10) / T)
             + A * ln10) / ((M + A) * ln10))


def _fasinh_inv(y: npt.ArrayLike, T: float, M: float, A: float) -> np.ndarray:
    ln10 = math.log(10.0)
    return T * np.sinh((np.asarray(y, float) * (M + A) - A) * ln10) \
        / math.sinh(M * ln10)


def _gml_inverse(spec: _FileSpec) -> _AxisFn:
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
        from .channel_transforms import inverse_transform_values
        return lambda y: inverse_transform_values(
            np.atleast_1d(np.asarray(y, float)), method=kind,
            t=p['T'], m=p['M'], w=p['W'], a=p['A'])
    raise ValueError(f'unsupported transformation {kind!r}')


def _export_scale(spec: _StoredSpec) -> tuple[_Xform | None, float]:
    """How a channel's stored coordinates go to Gating-ML:
    ``(transformation element or None, factor)`` where the Gating-ML value is
    ``factor * stored``. None for a linear channel (no transformation-ref);
    raises ValueError for a scale Gating-ML cannot name."""
    from .channel_transforms import transform_params
    params = transform_params(spec if isinstance(spec, dict)
                              else {'method': spec})
    method = params.get('method') or 'linear'
    if method == 'linear':
        return None, 1.0
    if method in ('logicle', 'hyperlog'):
        d = LOGICLE_DEFAULTS
        return ((method, {'T': params.get('t', d['t']), 'W': params.get('w', d['w']),
                          'M': params.get('m', d['m']), 'A': params.get('a', d['a'])}),
                1.0)
    if method == 'asinh':
        # OpenFlo's asinh(x / c). fasinh with A = 0, M = 1 and
        # T = c * sinh(ln 10) is asinh(x / c) / ln 10.
        c = float(params.get('cofactor', ASINH_DEFAULT_COFACTOR))
        return (('fasinh', {'T': c * math.sinh(math.log(10.0)), 'M': 1.0,
                            'A': 0.0}), 1.0 / math.log(10.0))
    if method == 'log':
        # OpenFlo's log10(x). flog with T = 10, M = 1 is log10(x / 10) + 1.
        return ('flog', {'T': 10.0, 'M': 1.0}), 1.0
    raise ValueError(f'no Gating-ML scale for {method!r}')


# ── Export ───────────────────────────────────────────────────────────────────

_NCNAME_BAD = re.compile(r'[^A-Za-z0-9_.\-]')


def _gate_ids(gates: Iterable[Mapping[str, Any]]) -> dict[str, str]:
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


def _fmt(v: float) -> str:
    return repr(float(v))


def _gate_name(g: Mapping[str, Any], ids: Mapping[str, str]) -> str:
    return g.get('name') or g.get('label') or ids[g['id']]


def write_gatingml(path: _Path, gates: Sequence[Mapping[str, Any]],
                   transforms: Mapping[str, _StoredSpec] | None = None,
                   compensation: tuple[Iterable[str], npt.ArrayLike] | None = None,
                   unknown_fns: Mapping[str, _UnknownFn | None] | None = None,
                   corrections: Iterable[Any] | None = None) -> list[str]:
    """Write `gates` (one sample's tree, parents before children, each with
    ``id`` / ``parent_id``) as a standalone Gating-ML 2.0 file at `path`.

    `transforms` is the sample's ``{channel: spec}`` stored-scale record,
    `compensation` its ``(channels, spillover matrix)`` or None, and
    `unknown_fns` ``{channel: f(stored) -> linear}`` for channels of unknown
    scale (see the module docstring). A comment names the OpenFlo version
    and ``corrections`` (export_stamp). Returns the warnings, one string per
    place another tool will not count what OpenFlo counts."""
    scales = _ExportScales(dict(transforms or {}), dict(unknown_fns or {}))
    warnings = []
    for prefix, uri in NS.items():
        ET.register_namespace(prefix, uri)
    root = ET.Element(_q('gating', 'Gating-ML'))
    comp_chans = _write_spectrum_matrix(root, compensation)

    ids = _gate_ids(gates)
    skipped = set()
    gate_els = []
    for g in gates:
        gname = _gate_name(g, ids)
        if g.get('parent_id') in skipped:
            warnings.append(f"'{gname}' left out: its parent was left out")
            el = None
        else:
            el = _export_gate(g, gname, ids, scales, comp_chans, warnings)
        if el is None:
            skipped.add(g['id'])
        else:
            gate_els.append((g, el))
    _drop_dangling(gate_els, gates, ids, warnings)

    for tid, (kind, params) in scales.transformations:
        t = ET.SubElement(root, _q('transforms', 'transformation'),
                          {_q('transforms', 'id'): tid})
        ET.SubElement(t, _q('transforms', kind),
                      {_q('transforms', k): _fmt(v) for k, v in params.items()})
    root.extend(el for _g, el in gate_els)
    _write_document(path, root, corrections)
    for w in warnings:
        log.warning('[Gating-ML export] %s', w)
    return warnings


def _write_spectrum_matrix(root: ET.Element,
                           compensation: tuple[Iterable[str], npt.ArrayLike] | None,
                           ) -> set[str]:
    """The sample's spillover as the ``Spill`` spectrumMatrix; returns the
    compensated channels (none without a matrix)."""
    if compensation is None:
        return set()
    c_chans, matrix = compensation
    c_chans = list(c_chans)
    matrix = np.asarray(matrix, float)
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
    return set(c_chans)


class _ExportScales:
    """Each channel's export scale, ``(ref or None, factor)`` or
    ``('unknown', f)``, with one transformation per distinct scale, named
    after its first channel."""

    def __init__(self, transforms: Mapping[str, _StoredSpec],
                 unknown_fns: Mapping[str, _UnknownFn | None]) -> None:
        self._transforms = transforms
        self._unknown_fns = unknown_fns
        self._scale: dict[str, _Scale] = {}
        self._ids: dict[tuple[str, tuple[tuple[str, Any], ...]], str] = {}
        # (id, (kind, params)), first use first
        self.transformations: list[tuple[str, _Xform]] = []

    def __call__(self, ch: str) -> _Scale:
        if ch not in self._scale:
            self._scale[ch] = self._lookup(ch)
        return self._scale[ch]

    def _lookup(self, ch: str) -> _Scale:
        from .channel_transforms import is_unknown_spec
        spec = self._transforms.get(ch)
        if spec is not None and is_unknown_spec(
                spec if isinstance(spec, dict) else {'method': spec}):
            return 'unknown', self._unknown_fns.get(ch)
        xf, factor = _export_scale(spec or 'linear')
        if xf is None:
            return None, factor
        key = (xf[0], tuple(sorted(xf[1].items())))
        if key not in self._ids:
            tid = _NCNAME_BAD.sub('_', f'{xf[0]}_{ch}')
            self._ids[key] = tid
            self.transformations.append((tid, xf))
        return self._ids[key], factor

    def coord(self, ch: str, v: float) -> float:
        ref, f = self(ch)
        if ref == 'unknown':
            return float(np.asarray(f(np.atleast_1d(float(v))))[0])
        return float(v) * f


def _export_gate(g: Mapping[str, Any], gname: str, ids: Mapping[str, str],
                 scales: _ExportScales, comp_chans: set[str],
                 warnings: list[str]) -> ET.Element | None:
    """`g` as a Gating-ML gate element, or None when it is left out; appends
    to `warnings` why, or where another tool may count differently."""
    kind = g.get('kind')
    chans = [g[k] for k in ('channel', 'x_channel', 'y_channel') if g.get(k)]
    unk = [c for c in chans if scales(c)[0] == 'unknown']
    if unk and (kind == 'ellipsoid'
                or any(scales(c)[1] is None for c in unk)):
        warnings.append(f"'{gname}' left out: {', '.join(unk)} has an "
                        "unknown scale that could not be matched to the "
                        "FCS file")
        return None
    if unk:
        warnings.append(f"'{gname}': {', '.join(unk)} has an unknown "
                        "scale; written in linear values matched to the "
                        "FCS file, check it in the other tool")
    attrs = {_q('gating', 'id'): ids[g['id']]}
    pid = g.get('parent_id')
    if pid is not None and pid in ids:
        attrs[_q('gating', 'parent_id')] = ids[pid]
    el = _gate_element(g, kind, attrs, ids, scales.coord)
    if el is None:
        warnings.append(f"'{gname}' left out: Gating-ML has no "
                        f"{kind!r} gate")
        return None
    info = ET.SubElement(el, _q('data-type', 'custom_info'))
    ET.SubElement(info, 'openflo', {'name': str(gname)})
    # custom_info must come first in a gate.
    el.remove(info)
    el.insert(0, info)
    _set_dimension_refs(el, scales, comp_chans)
    return el


def _set_dimension_refs(el: ET.Element, scales: _ExportScales,
                        comp_chans: set[str]) -> None:
    """Name each dimension's compensation and transformation."""
    for dim in el.iter(_q('gating', 'dimension')):
        fd = dim.find(_q('data-type', 'fcs-dimension'))
        ch = fd.get(_q('data-type', 'name')) if fd is not None else None
        dim.set(_q('gating', 'compensation-ref'),
                'Spill' if ch in comp_chans else 'uncompensated')
        # Every dimension here was written by _dimension, with a named
        # fcs-dimension, so ch is a channel name, never None.
        ref = scales(ch)[0]  # pyright: ignore[reportArgumentType]
        if ref and ref != 'unknown':
            dim.set(_q('gating', 'transformation-ref'), ref)


def _drop_dangling(gate_els: list[tuple[Mapping[str, Any], ET.Element]],
                   gates: Iterable[Mapping[str, Any]], ids: Mapping[str, str],
                   warnings: list[str]) -> None:
    """A boolean naming a gate that is not in the file (left out above, or
    never one of `gates`) makes the whole file unreadable to other tools.
    Leave such a boolean out of `gate_els`, and everything under it; repeat,
    since a boolean may name a boolean left out here."""
    names = {g['id']: _gate_name(g, ids) for g in gates}
    changed = True
    while changed:
        changed = False
        written = {g['id'] for g, _el in gate_els}
        for g, el in list(gate_els):
            reason = _dangling_reason(g, ids, written, names)
            if reason is None:
                continue
            gate_els.remove((g, el))
            changed = True
            warnings.append(f"'{names[g['id']]}' left out: {reason}")


def _dangling_reason(g: Mapping[str, Any], ids: Mapping[str, str],
                     written: set[str], names: Mapping[str, str]) -> str | None:
    """Why `g` cannot be written when only `written` gates are, or None."""
    gone = [o for o in (g.get('operands') or [])
            if g.get('kind') == 'boolean' and o not in written]
    if gone:
        which = ', '.join(f"'{names.get(o, o)}'" for o in gone)
        return f'its operand {which} is not in the file'
    if g.get('parent_id') in ids and g.get('parent_id') not in written:
        return 'its parent was left out'
    return None


def _write_document(path: _Path, root: ET.Element,
                    corrections: Iterable[Any] | None) -> None:
    if hasattr(ET, 'indent'):
        ET.indent(root, space='  ')
    body = ET.tostring(root, encoding='unicode')
    from .export_stamp import xml_comment
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + xml_comment(corrections)
                + '\n' + body + '\n')


def _dimension(parent: ET.Element, ch: str, lo: float | None = None,
               hi: float | None = None) -> ET.Element:
    attrs = {}
    if lo is not None:
        attrs[_q('gating', 'min')] = _fmt(lo)
    if hi is not None:
        attrs[_q('gating', 'max')] = _fmt(hi)
    d = ET.SubElement(parent, _q('gating', 'dimension'), attrs)
    ET.SubElement(d, _q('data-type', 'fcs-dimension'),
                  {_q('data-type', 'name'): ch})
    return d


def _bound(v: float | None, lo_side: bool) -> float | None:
    """None for OpenFlo's open-side sentinel, else the value."""
    if v is None:
        return None
    v = float(v)
    if (lo_side and v <= -OPEN) or (not lo_side and v >= OPEN):
        return None
    return v


def _gate_element(g: Mapping[str, Any], kind: str | None, attrs: dict[str, str],
                  ids: Mapping[str, str], coord: Callable[[str, float], float],
                  ) -> ET.Element | None:
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
        operands = list(g.get('operands') or [])
        # An operand not among the gates written is dropped with its boolean
        # by write_gatingml, never from the operand list: a boolean over the
        # remaining operands would be another population.
        if op not in ('and', 'or', 'not') or not operands:
            return None
        el = ET.Element(_q('gating', 'BooleanGate'), attrs)
        complement = {}
        if op == 'not' and len(operands) > 1:
            # OpenFlo's NOT over several gates is NOT(any of them), so an
            # event in none of them. Gating-ML's <not> takes one operand;
            # the same population is an <and> of every operand used as its
            # complement (it was written as NOT of the first operand only).
            op = 'and'
            complement = {_q('gating', 'use-as-complement'): 'true'}
        o = ET.SubElement(el, _q('gating', op))
        for ref in operands:
            ET.SubElement(o, _q('gating', 'gateReference'),
                          {_q('gating', 'ref'): ids.get(ref, str(ref)),
                           **complement})
        return el
    return None


# ── Import ───────────────────────────────────────────────────────────────────

class GatingMLError(ValueError):
    """The file is not a Gating-ML 2.0 document OpenFlo can read."""


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _attr(el: ET.Element, name: str) -> str | None:
    """An attribute by local name, whatever its namespace."""
    for k, v in el.attrib.items():
        if _local(k) == name:
            return v
    return None


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def _finite_float(v: object) -> float | None:
    try:
        # Any value: what float() refuses (None, a non-numeric string) is
        # caught below, so the static ConvertibleToFloat check does not apply.
        f = float(v)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def is_gatingml(path: _Path) -> bool:
    """True when `path` is an XML file whose root is a Gating-ML element.

    Deprecated: unused by OpenFlo and will be removed in a future release."""
    warnings.warn('is_gatingml is deprecated and will be removed in a future '
                  'release', DeprecationWarning, stacklevel=2)
    try:
        for _ev, el in ET.iterparse(path, events=('start',)):
            return _local(el.tag) == 'Gating-ML'
    except (ET.ParseError, OSError):
        return False
    return False


def _parse_root(path: _Path) -> ET.Element:
    from .wsp_reader import xml_declares_entities
    with open(path, 'rb') as fh:
        head = fh.read(1 << 20)
    # Same guard as WspReader: no DTD internal subset or entity declaration
    # (entity expansion), which a Gating-ML document never needs. A bare
    # <!DOCTYPE name> declares nothing and is allowed.
    if xml_declares_entities(head):
        raise GatingMLError(f'refusing a Gating-ML file with a DTD or entity '
                            f'declaration: {path}')
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise GatingMLError(f'not valid XML ({path}): {e}') from e
    if _local(root.tag) != 'Gating-ML':
        raise GatingMLError(f'not a Gating-ML document (root <{_local(root.tag)}>)')
    return root


def _or_open(v: float | None, side: int) -> float:
    """`v`, or the OPEN sentinel on `side` (-1 low, +1 high) when missing."""
    return side * OPEN if v is None else v


def _read_rectangle(el: ET.Element, dims: Sequence[_Dim],
                    dim_els: Sequence[ET.Element]) -> _Gate:
    """A RectangleGate: a threshold or interval in 1-D, a rect in 2-D."""
    b = [(ch, _finite_float(_attr(d, 'min')), _finite_float(_attr(d, 'max')))
         for (ch, _s, _c), d in zip(dims, dim_els, strict=True)]
    if len(b) == 1:
        ch, lo, hi = b[0]
        if lo is not None and hi is None:
            return {'kind': 'threshold', 'channel': ch, 'value': lo}
        return {'kind': 'interval', 'channel': ch,
                'lo': _or_open(lo, -1), 'hi': _or_open(hi, 1)}
    if len(b) == 2:
        (xc, x0, x1), (yc, y0, y1) = b
        return {'kind': 'rect', 'x_channel': xc, 'y_channel': yc,
                'x0': _or_open(x0, -1), 'x1': _or_open(x1, 1),
                'y0': _or_open(y0, -1), 'y1': _or_open(y1, 1)}
    raise ValueError(f'a {len(b)}-dimensional rectangle')


def _read_polygon(el: ET.Element, dims: Sequence[_Dim],
                  _dim_els: Sequence[ET.Element]) -> _Gate:
    if len(dims) != 2:
        raise ValueError('a polygon needs two dimensions')
    verts = []
    for v in _children(el, 'vertex'):
        cs = [_finite_float(_attr(c, 'value')) for c in _children(v, 'coordinate')]
        if len(cs) != 2 or None in cs:
            raise ValueError('a vertex without two coordinates')
        verts.append(cs)
    if len(verts) < 3:
        raise ValueError('a polygon with fewer than 3 vertices')
    return {'kind': 'polygon', 'x_channel': dims[0][0],
            'y_channel': dims[1][0], 'vertices': verts}


def _ellipsoid_parts(el: ET.Element) -> tuple[list[float | None],
                                              list[list[float | None]],
                                              float | None]:
    """The mean, covariance rows and distanceSquare of an EllipsoidGate,
    each value a float or None."""
    mean_el = _children(el, 'mean')
    cov_el = _children(el, 'covarianceMatrix')
    dsq_el = _children(el, 'distanceSquare')
    if not mean_el or not cov_el or not dsq_el:
        raise ValueError('an ellipsoid without mean, covariance or '
                         'distanceSquare')
    mean = [_finite_float(_attr(c, 'value')) for c in
            _children(mean_el[0], 'coordinate')]
    cov = [[_finite_float(_attr(e, 'value')) for e in _children(r, 'entry')]
           for r in _children(cov_el[0], 'row')]
    return mean, cov, _finite_float(_attr(dsq_el[0], 'value'))


def _read_ellipsoid(el: ET.Element, dims: Sequence[_Dim],
                    _dim_els: Sequence[ET.Element]) -> _Gate:
    if len(dims) != 2:
        raise ValueError('an ellipsoid in other than two dimensions')
    mean, cov, dsq = _ellipsoid_parts(el)
    if (len(mean) != 2 or None in mean or len(cov) != 2
            or any(len(r) != 2 or None in r for r in cov)
            or dsq is None or dsq <= 0):
        raise ValueError('a malformed ellipsoid')
    if np.any(np.linalg.eigvalsh(np.asarray(cov, float)) <= 0):
        raise ValueError('an ellipsoid whose covariance is not '
                         'positive definite')
    return {'kind': 'ellipsoid', 'x_channel': dims[0][0],
            'y_channel': dims[1][0], 'mean': mean, 'cov': cov,
            'distance_sq': dsq}


def _read_boolean(el: ET.Element, _dims: Sequence[_Dim],
                  _dim_els: Sequence[ET.Element]) -> _Gate:
    ops = [c for c in el if _local(c.tag) in ('and', 'or', 'not')]
    if len(ops) != 1:
        raise ValueError('a boolean gate without one and/or/not')
    refs = _children(ops[0], 'gateReference')
    comp = [(_attr(r, 'use-as-complement') or '').lower() == 'true'
            for r in refs]
    op = _local(ops[0].tag)
    if any(comp) and op == 'not':
        raise ValueError('a "not" over an operand used as its '
                         'complement')
    g = {'kind': 'boolean', 'op': op,
         'operands': [_attr(r, 'ref') for r in refs]}
    if any(comp):
        # Placed as OpenFlo "not" gates by _complement_helpers.
        g['_complements'] = comp
    return g


# Each Gating-ML gate element's reader: (el, dims, dimension elements) ->
# gate dict, with dims as read_gatingml's dim_info gives them. A ValueError
# leaves the gate out.
_GATE_READERS: dict[str, Callable[[ET.Element, Sequence[_Dim], Sequence[ET.Element]],
                                  _Gate]] = {
    'RectangleGate': _read_rectangle,
    'PolygonGate': _read_polygon,
    'EllipsoidGate': _read_ellipsoid,
    'BooleanGate': _read_boolean,
}


def _read_transformations(root: ET.Element) -> dict[str | None, _FileSpec]:
    """Each <transformation>'s id -> ``{'kind', 'params'}`` of its function."""
    xforms = {}
    for t in root.iter():
        if _local(t.tag) != 'transformation':
            continue
        tid = _attr(t, 'id')
        for fn in t:
            kind = _local(fn.tag)
            params = {k: _finite_float(_attr(fn, k)) for k in ('T', 'W', 'M', 'A')
                      if _attr(fn, k) is not None}
            xforms[tid] = {'kind': kind, 'params': params}
    return xforms


def _read_matrices(root: ET.Element) -> dict[str | None, dict[str, str]]:
    """Each matrix's fluorochrome -> detector names. A dimension on a matrix
    may be named by its fluorochrome ('FL1 dye' for 'FL1-H'), as FlowKit
    reads it; the data has only the detector, so the gate is put there."""
    matrices = {}
    for m in root.iter():
        if _local(m.tag) != 'spectrumMatrix':
            continue

        def dims_of(tag: str, m: ET.Element = m) -> list[str]:
            return [_attr(d, 'name') or '' for el in _children(m, tag)
                    for d in el if _local(d.tag) == 'fcs-dimension']
        fl, det = dims_of('fluorochromes'), dims_of('detectors')
        matrices[_attr(m, 'id')] = (dict(zip(fl, det, strict=True))
                                    if len(fl) == len(det) else {})
    return matrices


class _DimReader:
    """Reads a gate's <dimension> as ``(channel, file scale spec or None,
    on a matrix)``, raising ValueError for one OpenFlo cannot use, and
    collects the ids of the matrices the dimensions read so far are on."""

    def __init__(self, xforms: Mapping[str | None, _FileSpec],
                 matrices: Mapping[str | None, Mapping[str, str]]) -> None:
        self.xforms = xforms
        self.matrices = matrices
        self.used_matrices: set[str] = set()

    def __call__(self, dim: ET.Element) -> _Dim:
        from .wsp_reader import strip_comp_marker
        fd = next((c for c in dim.iter() if _local(c.tag) == 'fcs-dimension'),
                  None)
        if fd is None:
            raise ValueError('a dimension that is not an FCS parameter '
                             '(ratio or new dimension)')
        ch = strip_comp_marker(_attr(fd, 'name') or '')[0]
        if not ch:
            raise ValueError('a dimension with no parameter name')
        dye_to_detector = self.matrices.get(_attr(dim, 'compensation-ref') or '')
        if dye_to_detector and ch in dye_to_detector \
                and ch not in dye_to_detector.values():
            ch = dye_to_detector[ch]
        return ch, self._spec(dim), self._on_matrix(dim)

    def _spec(self, dim: ET.Element) -> _FileSpec | None:
        ref = _attr(dim, 'transformation-ref')
        if not ref:
            return None
        spec = self.xforms.get(ref)
        if spec is None:
            raise ValueError(f'transformation {ref!r} is not in the file')
        _gml_inverse(spec)          # ValueError for one OpenFlo can't undo
        return spec

    def _on_matrix(self, dim: ET.Element) -> bool:
        comp = _attr(dim, 'compensation-ref') or 'uncompensated'
        if comp in self.matrices:
            self.used_matrices.add(comp)
        elif comp not in ('uncompensated', 'FCS'):
            raise ValueError(f'compensation {comp!r} is not in the file')
        return comp != 'uncompensated'


def _nested_gates_warning(root: ET.Element) -> str | None:
    """Gating-ML puts every gate directly under <Gating-ML>. Gates inside
    another element are not read; the warning says so (None when there are
    none), rather than import nothing without a word."""
    parent_of = {c: p for p in root.iter() for c in p}
    nested = [el for el, p in parent_of.items()
              if _local(el.tag).endswith('Gate') and p is not root
              and not _local(p.tag).endswith('Gate')]
    if not nested:
        return None
    where = sorted({_local(parent_of[el].tag) for el in nested})
    return (f"{len(nested)} gate(s) inside <{'>, <'.join(where)}> "
            "not read: Gating-ML gates are direct children of "
            "<Gating-ML>")


def _imported_name(gid: str | None, el: ET.Element) -> str:
    """The OpenFlo name a gate was written with, else its Gating-ML id."""
    for info in _children(el, 'custom_info'):
        for c in info.iter():
            if _local(c.tag) == 'openflo' and c.get('name'):
                # Truthy just above; pyright does not narrow a repeated call.
                return c.get('name')  # pyright: ignore[reportReturnType]
    return gid or 'unnamed'


def _parse_gate(el: ET.Element, tag: str,
                dim_info: Callable[[ET.Element], _Dim]) -> _Gate:
    """One gate element as an OpenFlo gate dict; ValueError leaves it out."""
    dim_els = _children(el, 'dimension')
    dims = [dim_info(d) for d in dim_els]
    reader = _GATE_READERS.get(tag)
    if reader is None:
        raise ValueError(f'a {tag}, which OpenFlo does not read')
    g = reader(el, dims, dim_els)
    if dims:
        g['_gml_axes'] = {ch: spec for ch, spec, _c in dims}
        # Whether a gate on uncompensated values counts differently
        # depends on the sample it goes on: the caller checks with
        # wsp_gate_coords.wsp_uncompensated_channels.
        g['wsp_comp_channels'] = sorted({ch for ch, _s, c in dims if c})
    return g


def _parents_first(gate_els: Iterable[tuple[str | None, str | None, ET.Element, str]],
                   by_id: Mapping[str | None, _GateEl]) -> list[str | None]:
    """The gate ids of `gate_els`, each after its parent (file order
    otherwise). A parent cycle raises GatingMLError."""
    order, done = [], set()

    def visit(gid: str | None, trail: tuple[str | None, ...] = ()) -> None:
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
    return order


def _import_gates(order: Iterable[str | None], by_id: Mapping[str | None, _GateEl],
                  dim_info: Callable[[ET.Element], _Dim], warnings: list[str],
                  ) -> tuple[list[_Gate], dict[str | None, str]]:
    """The gates in `order` as gate dicts with labels, import ids and
    parent ids, plus ``{Gating-ML id: import id}``. A gate that cannot be
    read is left out with every gate under it, each with a warning."""
    gates, imp, dropped = [], {}, set()
    for gid in order:
        pid, el, tag = by_id[gid]
        gname = _imported_name(gid, el)
        if pid is not None and (pid not in by_id or pid in dropped):
            dropped.add(gid)
            why = ('its parent was left out' if pid in dropped
                   else f'its parent {pid!r} is not in the file')
            warnings.append(f"'{gname}' left out: {why}")
            continue
        try:
            g = _parse_gate(el, tag, dim_info)
        except ValueError as e:
            dropped.add(gid)
            warnings.append(f"'{gname}' left out: {e}")
            continue
        g['label'] = gname
        g['parent_id'] = imp.get(pid) if pid is not None else None
        comp = g.pop('_complements', None)
        if comp:
            g['operands'] = _complement_helpers(
                g, comp, gid, imp, gates,
                lambda r: _imported_name(r, by_id[r][1]) if r in by_id else r)
        g['_import_id'] = f'gml_{len(imp)}'
        imp[gid] = g['_import_id']
        gates.append(g)
    return gates, imp


def _remap_boolean_operands(gates: Iterable[_Gate], imp: Mapping[str | None, str],
                            warnings: list[str]) -> None:
    """Boolean operands name Gating-ML ids; point them at the import ids. An
    operand that was left out empties the boolean, which then reads NA
    (gate_eval.broken_booleans), never every event."""
    for g in gates:
        if g['kind'] == 'boolean':
            ops = g['operands']
            if all(o in imp for o in ops):
                g['operands'] = [imp[o] for o in ops]
            else:
                g['operands'] = []
                warnings.append(f"'{g['label']}': an operand was left out, "
                                "so it reads NA")


def read_gatingml(path: _Path, transforms: Mapping[str, _StoredSpec] | None = None,
                  *, raw: bool = False) -> tuple[list[_Gate], list[str]]:
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
    root = _parse_root(path)
    warnings = []
    dim_info = _DimReader(_read_transformations(root), _read_matrices(root))
    gate_els = [(_attr(el, 'id'), _attr(el, 'parent_id'), el, _local(el.tag))
                for el in root if _local(el.tag).endswith('Gate')]
    nested = _nested_gates_warning(root)
    if nested:
        warnings.append(nested)
    by_id = {gid: (pid, el, tag) for gid, pid, el, tag in gate_els}
    order = _parents_first(gate_els, by_id)
    gates, imp = _import_gates(order, by_id, dim_info, warnings)
    if dim_info.used_matrices:
        warnings.append(f"the file's gates use its compensation matrix "
                        f"{', '.join(sorted(dim_info.used_matrices))}; OpenFlo "
                        "gates each sample with its own compensation")
    _remap_boolean_operands(gates, imp, warnings)
    for w in warnings:
        log.warning('[Gating-ML import] %s', w)
    if not raw:
        gates = gates_to_stored(gates, transforms)
    return gates, warnings


def _complement_helpers(g: _Gate, comp: Sequence[bool], gid: str | None,
                        imp: dict[str | None, str], gates: list[_Gate],
                        name: Callable[[str], str]) -> list[str]:
    """Operands for boolean `g` whose Gating-ML operands ``comp`` marks
    use-as-complement. OpenFlo's boolean gates have no per-operand
    complement, but its "not" gate is NOT(any operand), so each complement
    becomes a "not" gate beside `g` (same parent), appended to `gates`:

    - and: A and not B and not C = A and NOT(B or C), one helper;
    - or:  A or not B or not C, one helper per complemented operand.

    Returns `g`'s new operand list, naming each helper by a key in `imp`
    that the operand mapping in read_gatingml resolves."""
    ops = g['operands']
    if g['op'] == 'and':
        groups = [[r for r, c in zip(ops, comp, strict=True) if c]]
        kept = [r for r, c in zip(ops, comp, strict=True) if not c]
    else:
        groups = [[r] for r, c in zip(ops, comp, strict=True) if c]
        kept = [r for r, c in zip(ops, comp, strict=True) if not c]
    for k, refs in enumerate(groups):
        names = ' or '.join(name(r) for r in refs)
        key = f'{gid}#not{k}'
        imp[key] = f'gml_{len(imp)}'
        gates.append({'kind': 'boolean', 'op': 'not', 'operands': refs,
                      'label': f"not {names}" if len(refs) == 1
                      else f"not ({names})",
                      '_import_id': imp[key], 'parent_id': g['parent_id']})
        kept.append(key)
    return kept


def _same_scale(file_spec: _FileSpec | None, stored_spec: _StoredSpec | None,
                ) -> float | None:
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


def _axis_fn(file_spec: _FileSpec | None, stored_spec: _StoredSpec | None,
             ) -> tuple[_AxisFn, float | None]:
    """``(f, k)``: `f` maps the file's coordinates on a channel to the stored
    scale `stored_spec`; `k` is the constant factor (stored = file / k) when
    the two scales are one, else None. A stored scale that is unknown takes
    linear values (wsp_gate_coords.mark_unknown_scale then flags the gate)."""
    from .channel_transforms import is_unknown_spec, transform_params, transform_values
    spec: dict[str, Any] | None = stored_spec if isinstance(stored_spec, dict) else (
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


def gates_to_stored(gates: Iterable[Mapping[str, Any]],
                    transforms: Mapping[str, _StoredSpec] | None = None,
                    ) -> list[_Gate]:
    """Gates read with ``read_gatingml(raw=True)`` put on the stored scales
    `transforms` (``{channel: spec}``; None or ``{}`` for linear values).

    Bounds map exactly (every scale is monotone). A polygon or ellipse whose
    axes are on the stored scale already maps by a constant and keeps its
    shape exactly. On another scale, a Gating-ML edge is straight on the
    file's axes, so each polygon edge is subdivided there before mapping,
    and an ellipse is traced as a polygon there; the warning names it."""
    from .wsp_reader import ellipse_polygon
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

        def m(ch: str, v: float | None,
              fns: Mapping[str, tuple[_AxisFn, float | None]] = fns) -> float | None:
            if v is None or abs(float(v)) >= OPEN:
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
                    v = np.asarray(ellipse_polygon(
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
