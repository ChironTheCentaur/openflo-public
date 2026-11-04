"""Gate coordinates across the FlowJo .wsp boundary: ``map_gate_coords``
(a gate dict put on another scale, polygon edges traced and ellipses turned
into polygons), the per-channel functions it takes (``gate_scale_fns``,
``gate_rescale_fns``, ``rank_matched_fn``), unknown-scale marking
(``mark_unknown_scale``), the FlowJo axes a workspace's statistics use
(``wsp_stat_axes``, ``stat_axes_from_json``, ``edge_axis_fns``) and the
channels FlowJo evaluates uncompensated (``wsp_uncompensated_channels``).

Core module: numpy only. No Tk, and it does not import ``openflo.pipeline``
(pipeline imports it).
"""

import logging
from typing import Any

import numpy as np

from .channel_transforms import (
    inverse_transform_values,
    is_unknown_spec,
    to_linear,
    transform_params,
    transform_values,
)
from .scales import OPEN
from .wsp_reader import ellipse_polygon as _ellipse_polygon
from .wsp_reader import flowjo_biex_lut as _flowjo_biex_lut
from .wsp_reader import flowjo_edge_axis as _flowjo_edge_axis
from .wsp_reader import strip_comp_marker as _strip_comp_marker

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


# ── Gate coordinates across the .wsp boundary ───────────────────────────────
#
# A FlowJo workspace holds gate coordinates in the parameter's SCALE units:
# compensated linear intensity (compare.py evaluates them on linear data and
# agrees with FlowJo's counts). OpenFlo's editor and CLI store fluorescence
# logicle-transformed in place and keep gates in those stored coordinates. The
# two were passed across unconverted in both directions: an imported FlowJo
# "FL1-A > 1000" was evaluated against logicle values that never exceed ~1.2
# (7,653 events in FlowJo, 0 in OpenFlo), and an exported OpenFlo threshold of
# 0.454 (logicle) reached FlowJo as an intensity of 0.454 (7,653 events in
# OpenFlo, 14,087 in FlowJo). `map_gate_coords` converts a gate between the
# two, given a function per transformed channel.


POLYGON_EDGE_STEPS = 16   # sub-segments per polygon edge when re-mapped


def map_gate_coords(g, fns):
    """Copy of gate dict `g` with every coordinate on a channel in `fns` passed
    through that channel's function; `g` itself when no axis maps.

    `fns` is ``{channel: f}`` with `f` a vectorised, increasing map (e.g.
    ``transform_values`` / ``inverse_transform_values`` for that channel's
    method). Bounds map exactly because the transforms are monotone, and the
    ±1e12 open-bound sentinels stay open. A non-finite image (a non-positive
    bound under 'log') becomes the open lower sentinel, as it lies below every
    value the channel can hold.

    A straight polygon edge in one space is a curve in the other, so each
    edge is subdivided before its vertices are mapped; an ellipse is not an
    ellipse in the other space, so it is traced as a polygon first (area-
    preserving, see `_ellipse_polygon`) and leaves as kind 'polygon'.

    A polygon imported from FlowJo with 'flowjo_edge_axes' has its edges
    straight on FlowJo's axes, not in `g`'s units: each edge is traced there
    instead (even when no axis maps), and the copy, straight in the units it
    leaves in, drops the key."""
    edge_fns = polygon_edge_fns(g) if g.get('kind') == 'polygon' else None
    if not fns and edge_fns is None:
        return g
    fns = fns or {}

    def fn_for(ch):
        return fns.get(ch) if isinstance(ch, str) else None

    def m(f, vals):
        v = np.asarray(vals, dtype=float).copy()
        live = np.abs(v) < OPEN
        if f is not None and live.any():
            got = np.asarray(f(v[live]), dtype=float).reshape(-1)
            v[live] = np.where(np.isfinite(got), got, -OPEN)
        return v

    if 'channel' in g:                              # threshold / interval
        f = fn_for(g['channel'])
        if f is None:
            return g
        g = dict(g)
        for key in ('value', 'lo', 'hi'):
            if g.get(key) is not None:
                g[key] = float(m(f, [g[key]])[0])
        return g
    fx, fy = fn_for(g.get('x_channel')), fn_for(g.get('y_channel'))
    if fx is None and fy is None and edge_fns is None:
        return g
    g = dict(g)
    kind = g.get('kind')
    if edge_fns is not None:
        g.pop('flowjo_edge_axes', None)
        g['vertices'] = _trace_edges_on_axes(g['vertices'], edge_fns)
    for key, f in (('x0', fx), ('x1', fx), ('quad_origin_x', fx),
                   ('y0', fy), ('y1', fy), ('quad_origin_y', fy)):
        if g.get(key) is not None:
            g[key] = float(m(f, [g[key]])[0])
    if kind == 'ellipsoid' and g.get('mean') is not None:
        verts = _ellipse_polygon(g['mean'], g['cov'],
                                 g.get('distance_sq', 4.0))
        if verts is None:
            log.warning("  [gate] %s: ellipse with a singular covariance "
                        "cannot be converted between scales — left as is.",
                        g.get('name') or g.get('id') or 'ellipse')
            return g
        for key in ('mean', 'cov', 'distance_sq'):
            g.pop(key, None)
        g['kind'] = kind = 'polygon'
        g['vertices'] = verts
    if kind == 'polygon' and g.get('vertices') is not None:
        v = np.asarray(g['vertices'], dtype=float)
        if v.ndim == 2 and v.shape[1] == 2 and len(v) >= 3:
            pts = []
            for i in range(len(v)):
                a, b = v[i], v[(i + 1) % len(v)]
                pts.append(a)
                if edge_fns is None and np.all(
                        np.abs(np.concatenate([a, b])) < OPEN):
                    t = np.arange(1, POLYGON_EDGE_STEPS) / POLYGON_EDGE_STEPS
                    pts.extend(a + (b - a) * t[:, None])
            p = np.asarray(pts, dtype=float)
            xs, ys = m(fx, p[:, 0]), m(fy, p[:, 1])
            g['vertices'] = [[float(x), float(y)] for x, y in zip(xs, ys, strict=True)]
    return g


def _trace_edges_on_axes(vertices, edge_fns, steps=None):
    """Polygon `vertices` (data units) with each edge traced as `steps`
    segments straight on the axes `edge_fns` names (see polygon_edge_fns);
    an edge with an open (±1e12) end is kept as it is."""
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2 or len(v) < 3:
        return vertices
    steps = steps or _EDGE_AXIS_STEPS
    t = (np.arange(steps) / steps)[:, None]
    out = []
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        if not np.all(np.abs(np.concatenate([a, b])) < OPEN):
            out.append(a[None, :])
            continue
        ea, eb = a.copy(), b.copy()
        for k, f in enumerate(edge_fns):
            if f is not None:
                ea[k], eb[k] = (float(x) for x in f[0](np.array([a[k], b[k]])))
        seg = ea + (eb - ea) * t
        for k, f in enumerate(edge_fns):
            if f is not None:
                seg[:, k] = f[1](seg[:, k])
        seg[0] = a                     # the drawn vertex exactly
        out.append(seg)
    p = np.concatenate(out)
    return [[float(x), float(y)] for x, y in p]


def gate_scale_fns(transforms, direction):
    """``{channel: f}`` for `map_gate_coords` from ``{channel: method}`` (or
    ``{channel: params dict}`` as FlowSample._transformed_channels returns):
    `direction` 'to_linear' inverts the stored transform (export to FlowJo),
    'from_linear' applies it (import from FlowJo). Linear channels are left
    out, so gates on them pass through untouched."""
    if direction not in ('to_linear', 'from_linear'):
        raise ValueError(f"direction must be 'to_linear' or 'from_linear', "
                         f"not {direction!r}")
    out = {}
    for ch, spec in (transforms or {}).items():
        params = dict(spec) if isinstance(spec, dict) else {'method': spec}
        if (params.get('method') or 'linear') == 'linear':
            continue
        if is_unknown_spec(params):
            # No transform to invert or apply: the gate stays as it is, which
            # is wrong in one of the two spaces, so say so.
            log.warning("  [gate] %s: display scale unknown -- gate "
                        "coordinates on it are NOT converted for FlowJo.", ch)
            continue
        params = transform_params(params)   # not 'origin' / 'guessed'
        fn = (inverse_transform_values if direction == 'to_linear'
              else transform_values)
        out[ch] = (lambda v, _p=params, _f=fn:
                   np.asarray(_f(np.asarray(v, dtype=float), **_p)))
    return out


def gate_rescale_fns(before, after):
    """``{channel: f}`` for `map_gate_coords` that moves gate coordinates
    from the transforms `before` to `after` (two ``data_transforms``
    records): undo the old transform, apply the new one. Only channels whose
    parameters differ are listed; a channel of unknown scale on either side
    is left out, as nothing converts to or from it. Used when a sample's data
    is re-transformed under gates drawn on its old transform (a cofactor-5
    asinh sample put on the editor's cofactor 150): unconverted, a cut that
    held 240 of 800 events held none."""
    before, after = before or {}, after or {}
    lin = {'method': 'linear'}
    out = {}
    for ch in set(before) | set(after):
        b, a = before.get(ch) or lin, after.get(ch) or lin
        if is_unknown_spec(b) or is_unknown_spec(a):
            continue
        if transform_params(b) == transform_params(a):
            continue
        a_params = transform_params(a)
        if a.get('method', 'linear') == 'linear':
            out[ch] = (lambda v, _b=b: to_linear(v, _b))
        else:
            out[ch] = (lambda v, _b=b, _p=a_params: np.asarray(
                transform_values(to_linear(v, _b), **_p)))
    return out


def rank_matched_fn(stored, linear, n=1024):
    """``f`` for `map_gate_coords` taking a channel of UNKNOWN scale
    (`stored`, its values as kept) to the linear units FlowJo will see
    (`linear`, the same channel of the events read from the FCS): the
    stored value at each quantile goes to the linear value at the same
    quantile. The display transforms are increasing, so when both hold the
    same events this is the transform itself and a cut keeps exactly its
    events either side; past the data it carries on with the end slopes, so
    a bound beyond every event stays beyond every event. ``None`` when
    either side has fewer than two distinct finite values."""
    s = np.asarray(stored, dtype=float)
    x = np.asarray(linear, dtype=float)
    s, x = s[np.isfinite(s)], x[np.isfinite(x)]
    if np.unique(s).size < 2 or np.unique(x).size < 2:
        return None
    q = np.linspace(0.0, 1.0, int(n))
    sq, xq = np.quantile(s, q), np.quantile(x, q)
    sq, keep = np.unique(sq, return_index=True)     # np.interp needs xp rising
    xq = xq[keep]
    if sq.size < 2:
        return None

    span = (xq[-1] - xq[0]) / (sq[-1] - sq[0])      # > 0: both sides vary

    def _slope(i, j):
        d = (xq[j] - xq[i]) / (sq[j] - sq[i])
        return d if d > 0 else span

    lo_k, hi_k = _slope(0, 1), _slope(-2, -1)

    def f(v):
        v = np.asarray(v, dtype=float)
        out = np.interp(v, sq, xq)
        lo, hi = v < sq[0], v > sq[-1]
        out[lo] = xq[0] + (v[lo] - sq[0]) * lo_k
        out[hi] = xq[-1] + (v[hi] - sq[-1]) * hi_k
        return out
    return f


def mark_unknown_scale(gate: dict, transforms) -> dict:
    """`gate` (a copy) flagged ``scale_unknown`` when it is in FlowJo's
    linear units on a channel whose display scale is unknown
    (``transforms``: the sample's stored transforms), which gate_scale_fns
    cannot convert: compared as stored, its coordinates name another region.
    gate_to_mask then fails closed and statistics say NA. A boolean or
    placeholder gate has no coordinates of its own and is returned as is."""
    from .gating import gate_channels
    if gate.get('kind') in ('boolean', 'flowjo_ellipse'):
        return gate
    if any(is_unknown_spec((transforms or {}).get(ch))
           for ch in gate_channels(gate)):
        out: dict = dict(gate)
        out['scale_unknown'] = True
        return out
    return gate


# An imported polygon's 'flowjo_edge_axes' (the FlowJo axes its edges are
# straight on) are read by wsp_reader; see the note before
# wsp_reader.flowjo_edge_axis.
_EDGE_AXIS_STEPS = 64      # sub-segments per edge when a polygon is re-mapped


def wsp_stat_axes(sample_elem):
    """{data column: FlowJo axis} that a (namespace-stripped) workspace
    <Sample> declares in <Transformations>, as `_flowjo_edge_axis` specs.
    FlowJo's GeoMean is a mean on this axis (gating.population_stats).

    A compensated parameter ('Comp-FL1-A', '<FL1-A>') wins over the bare,
    uncompensated one, as the data column holds compensated values. A linear
    axis, and one not understood, give no entry: that channel keeps OpenFlo's
    own display transform."""
    names = {}
    for block in (sample_elem.findall('Transformations')
                  if sample_elem is not None else []):
        for t in block:
            p = t.find('parameter')
            name = p.get('name') if p is not None else None
            if not name:
                continue
            col, marked = _strip_comp_marker(name)
            if marked or col not in names:
                names[col] = name
    out = {}
    for col, name in names.items():
        axis = _flowjo_edge_axis(sample_elem, name)
        if axis is not None:
            out[col] = axis
    return out


def stat_axes_from_json(rec):
    """A saved ``stat_axes`` record (see wsp_stat_axes) as {channel: axis},
    keeping only the axes `edge_axis_fns` reads; {} for anything else."""
    if not isinstance(rec, dict):
        return {}
    return {str(c): dict(a) for c, a in rec.items()
            if isinstance(a, dict) and edge_axis_fns(a) is not None}


def edge_axis_fns(spec):
    """(forward, inverse) between data units and a 'flowjo_edge_axes' axis
    (see `_flowjo_edge_axis`), both vectorised and increasing; None for a
    spec that names no axis this reads."""
    if not isinstance(spec, dict):
        return None
    kind = spec.get('kind')
    try:
        if kind == 'logicle':
            p: dict[str, Any] = {'t': float(spec['T']), 'w': float(spec['W']),
                                 'm': float(spec['M']), 'a': float(spec['A'])}
            return ((lambda v: transform_values(v, 'logicle', **p)),
                    (lambda y: inverse_transform_values(y, 'logicle', **p)))
        if kind == 'biex':
            lut = _flowjo_biex_lut(float(spec['neg']), float(spec['width']),
                                   float(spec['pos']), float(spec['maxRange']))
            if lut is None:
                return None
            data, chan = lut
            # FlowJo's table; an event off either end sits at that end, as
            # FlowJo and FlowKit put it.
            return ((lambda v: np.interp(np.asarray(v, dtype=float), data, chan)),
                    (lambda y: np.interp(np.asarray(y, dtype=float), chan, data)))
        if kind == 'log':
            off, dec = float(spec['offset']), float(spec['decades'])
            # FlowJo's log axis: a value below the offset sits at the bottom.
            return ((lambda v: np.log10(np.maximum(np.asarray(v, dtype=float), off)
                                        / off) / dec),
                    (lambda y: off * np.power(10.0, np.asarray(y, dtype=float) * dec)))
        if kind == 'fasinh':
            from .gatingml import _fasinh, _fasinh_inv
            T, M, A = float(spec['T']), float(spec['M']), float(spec['A'])
            return ((lambda v: _fasinh(v, T, M, A)),
                    (lambda y: _fasinh_inv(y, T, M, A)))
    except (KeyError, TypeError, ValueError):
        return None
    return None


def polygon_edge_fns(gate):
    """[(forward, inverse) or None for x, the same for y] for a polygon
    carrying 'flowjo_edge_axes' (None: that axis is in data units); None
    when the gate carries none this reads."""
    axes = gate.get('flowjo_edge_axes')
    if not isinstance(axes, (list, tuple)) or len(axes) != 2:
        return None
    fns = [edge_axis_fns(a) if a is not None else None for a in axes]
    return fns if any(f is not None for f in fns) else None


def wsp_uncompensated_channels(gate, compensated):
    """The channels of imported gate `gate` that FlowJo evaluates on the
    UNcompensated parameter although `compensated` (the sample's compensated
    channels) holds them: named without 'Comp-' / <...> in the workspace.
    [] for a gate without 'wsp_comp_channels' -- one not read from a .wsp
    (drawn, from a template, hand-built), whose names mean the data's
    columns (see wsp_reader.wsp_comp_channels)."""
    if 'wsp_comp_channels' not in gate:
        return []
    marked = set(gate.get('wsp_comp_channels') or ())
    comp = set(compensated or ())
    return sorted({gate.get(k) for k in ('channel', 'x_channel', 'y_channel')
                   if gate.get(k) in comp and gate.get(k) not in marked})
