"""Typed gate coordinates -- Tk-free.

Gates are stored in each channel's STORED coordinates (logicle or asinh for
fluorescence), but people read and type the axis's units: the intensity the
ticks show. These helpers turn a gate into the fields a dialog shows, in
intensity, and the typed text back into a gate, so a published gate can be
reproduced, a gate matched exactly between experiments, or a gate set
without a mouse.

A channel whose display scale is unknown has no intensity to type: its
fields are the stored values, and the dialog says so. An empty bound of a
rectangle or interval is an open side (the +-1e12 sentinel the quadrant tool
and imported open-sided gates use). A field left as it was shown keeps the
stored value exactly, so opening the dialog and pressing OK changes nothing.
"""
from __future__ import annotations

import math
import re

import numpy as np

OPEN = 1e12

# Kinds the dialog edits. A quadrant is four 'rect' gates sharing a
# `quad_set`; its fields are the shared origin.
EDITABLE_KINDS = ('threshold', 'interval', 'rect', 'polygon', 'ellipsoid')

_SUFFIX = {'k': 1e3, 'm': 1e6, 'g': 1e9}


def editable(gate) -> bool:
    return gate.get('kind') in EDITABLE_KINDS and not gate.get('scale_unknown')


def parse_number(text) -> float | None:
    """A typed value: '1500', '1,500', '1.5K', '-200', '2e3', '0.453'.
    None for an empty field (an open side); ValueError for anything else."""
    s = str(text).strip().replace(',', '').replace('−', '-')
    if not s:
        return None
    m = re.fullmatch(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([kKmMgG]?)', s)
    if not m:
        raise ValueError(f"'{text}' is not a number")
    v = float(m.group(1)) * _SUFFIX.get(m.group(2).lower(), 1.0)
    if not math.isfinite(v):
        raise ValueError(f"'{text}' is not a finite number")
    return v


def format_number(v) -> str:
    """A value as the field shows it: plain digits, 6 significant figures."""
    v = float(v)
    if v == 0:
        return '0'
    return f'{v:.6g}'


def _spec(transforms, ch):
    return (transforms or {}).get(ch)


def _unknown(spec) -> bool:
    from .pipeline import is_unknown_spec
    return bool(spec) and is_unknown_spec(spec)


def to_axis(value, spec) -> float:
    """A stored coordinate as the axis value a user types (intensity)."""
    from .pipeline import to_linear
    if spec is None or _unknown(spec):
        return float(value)
    return float(to_linear(np.array([float(value)]), spec)[0])


def from_axis(value, spec) -> float:
    """A typed axis value (intensity) as the stored coordinate."""
    from .pipeline import transform_params, transform_values
    if (spec is None or _unknown(spec)
            or (spec.get('method') or 'linear') == 'linear'):
        return float(value)
    out = float(np.asarray(transform_values(
        np.array([float(value)]), **transform_params(spec))).ravel()[0])
    if not math.isfinite(out):
        raise ValueError(f"{format_number(value)} is off this channel's scale")
    return out


def _axis_text(value, spec) -> str:
    if value is None or abs(float(value)) >= OPEN:
        return ''
    return format_number(to_axis(value, spec))


def ellipse_params(mean, cov, dsq):
    """``(cx, cy, half_width, half_height, angle_deg)`` of an ellipsoid gate,
    on the plotted (stored) scale: the half-axes of its boundary and the
    angle of the first from the x axis."""
    cov = np.asarray(cov, dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    a, b = np.sqrt(np.maximum(vals, 0.0) * float(dsq))
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    return float(mean[0]), float(mean[1]), float(a), float(b), float(ang)


def ellipse_cov(a, b, angle_deg, dsq):
    """The covariance whose `dsq` boundary has half-axes ``a``, ``b`` at
    ``angle_deg`` (ellipse_params inverted)."""
    t = math.radians(angle_deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    d = np.diag([a * a / dsq, b * b / dsq])
    return (r @ d @ r.T).tolist()


def fields(gate, transforms=None) -> list[tuple[str, str, str]]:
    """The dialog's rows for ``gate``: ``(key, label, text)``. A polygon has
    one row, ``'vertices'``, whose text is one 'x, y' pair per line."""
    k = gate.get('kind')
    if k == 'threshold':
        ch = gate['channel']
        return [('value', f"{ch} at or above",
                 _axis_text(gate['value'], _spec(transforms, ch)))]
    if k == 'interval':
        ch = gate['channel']
        s = _spec(transforms, ch)
        return [('lo', f"{ch} from (inside)", _axis_text(gate['lo'], s)),
                ('hi', f"{ch} below", _axis_text(gate['hi'], s))]
    xc, yc = gate.get('x_channel'), gate.get('y_channel')
    sx, sy = _spec(transforms, xc), _spec(transforms, yc)
    if k == 'rect' and gate.get('quad_set'):
        return [('quad_origin_x', f"{xc} divider",
                 _axis_text(gate.get('quad_origin_x', 0.0), sx)),
                ('quad_origin_y', f"{yc} divider",
                 _axis_text(gate.get('quad_origin_y', 0.0), sy))]
    if k == 'rect':
        return [('x0', f"{xc} from", _axis_text(gate['x0'], sx)),
                ('x1', f"{xc} below", _axis_text(gate['x1'], sx)),
                ('y0', f"{yc} from", _axis_text(gate['y0'], sy)),
                ('y1', f"{yc} below", _axis_text(gate['y1'], sy))]
    if k == 'polygon':
        lines = [f"{_axis_text(x, sx) or 'open'}, {_axis_text(y, sy) or 'open'}"
                 for x, y in gate.get('vertices') or []]
        return [('vertices', f"Vertices ({xc}, {yc}), one per line",
                 '\n'.join(lines))]
    if k == 'ellipsoid':
        cx, cy, a, b, ang = ellipse_params(gate['mean'], gate['cov'],
                                           gate.get('distance_sq', 4.0))
        return [('cx', f"Centre {xc}", _axis_text(cx, sx)),
                ('cy', f"Centre {yc}", _axis_text(cy, sy)),
                ('a', "Half-width on the plot scale", format_number(a)),
                ('b', "Half-height on the plot scale", format_number(b)),
                ('angle', "Angle (degrees)", format_number(ang))]
    raise ValueError(f"a {k} gate has no typed coordinates")


def unknown_channels(gate, transforms=None) -> list:
    """The gate's channels whose fields are stored values (unknown scale)."""
    return [ch for ch in (gate.get('channel'), gate.get('x_channel'),
                          gate.get('y_channel'))
            if ch and _unknown(_spec(transforms, ch))]


def _value(text, shown, stored, spec, *, allow_open=False, what='value'):
    """The stored coordinate for one field: the original ``stored`` when the
    text is unchanged, the open sentinel for an empty bound, else the typed
    axis value converted."""
    if str(text).strip() == str(shown).strip() and stored is not None:
        return float(stored)
    v = parse_number(text)
    if v is None:
        if allow_open is False:
            raise ValueError(f"{what} needs a value")
        return allow_open
    return from_axis(v, spec)


def apply(gate, typed: dict, transforms=None) -> dict:
    """A copy of ``gate`` with the typed fields (``{key: text}``, the keys
    `fields` gave) applied. ValueError says what is wrong with the input."""
    import copy
    shown = {k: t for k, _lab, t in fields(gate, transforms)}
    g = copy.deepcopy(gate)
    k = g.get('kind')

    stored = dict(gate)
    if k == 'ellipsoid':
        stored['cx'], stored['cy'] = gate['mean'][0], gate['mean'][1]

    def val(key, spec, **kw):
        return _value(typed.get(key, shown[key]), shown[key], stored.get(key),
                      spec, **kw)
    if k == 'threshold':
        g['value'] = val('value', _spec(transforms, g['channel']),
                         what='The cut')
        return g
    if k == 'interval':
        s = _spec(transforms, g['channel'])
        lo = val('lo', s, allow_open=-OPEN, what='From')
        hi = val('hi', s, allow_open=OPEN, what='Below')
        if not lo < hi:
            raise ValueError("'from' must be below 'below'")
        g['lo'], g['hi'] = lo, hi
        return g
    sx = _spec(transforms, g.get('x_channel'))
    sy = _spec(transforms, g.get('y_channel'))
    if k == 'rect' and g.get('quad_set'):
        g['quad_origin_x'] = val('quad_origin_x', sx, what='The x divider')
        g['quad_origin_y'] = val('quad_origin_y', sy, what='The y divider')
        return g
    if k == 'rect':
        x0 = val('x0', sx, allow_open=-OPEN)
        x1 = val('x1', sx, allow_open=OPEN)
        y0 = val('y0', sy, allow_open=-OPEN)
        y1 = val('y1', sy, allow_open=OPEN)
        if not (x0 < x1 and y0 < y1):
            raise ValueError("each 'from' must be below its 'below'")
        g.update(x0=x0, x1=x1, y0=y0, y1=y1)
        return g
    if k == 'polygon':
        text = typed.get('vertices', shown['vertices'])
        if str(text).strip() == shown['vertices'].strip():
            return g
        verts = []
        for n, line in enumerate(str(text).splitlines(), 1):
            if not line.strip():
                continue
            parts = [p for p in re.split(r'[,;\t]|\s+', line.strip()) if p]
            if len(parts) != 2:
                raise ValueError(f"line {n}: give 'x, y'")
            xy = []
            for p, s in zip(parts, (sx, sy), strict=True):
                v = parse_number(p) if p.lower() != 'open' else None
                if v is None:
                    raise ValueError(f"line {n}: '{p}' is not a number")
                xy.append(from_axis(v, s))
            verts.append(xy)
        if len(verts) < 3:
            raise ValueError("a polygon needs at least three vertices")
        g['vertices'] = verts
        return g
    if k == 'ellipsoid':
        cx = val('cx', sx, what='The centre')
        cy = val('cy', sy, what='The centre')
        same = all(str(typed.get(key, shown[key])).strip() == shown[key]
                   for key in ('a', 'b', 'angle'))
        g['mean'] = [cx, cy]
        if not same:
            a = parse_number(typed.get('a', shown['a']))
            b = parse_number(typed.get('b', shown['b']))
            ang = parse_number(typed.get('angle', shown['angle']))
            if a is None or b is None or ang is None or a <= 0 or b <= 0:
                raise ValueError("the half-width and half-height must be "
                                 "above 0, and the angle a number")
            g['cov'] = ellipse_cov(a, b, ang, float(g.get('distance_sq', 4.0)))
        return g
    raise ValueError(f"a {k} gate has no typed coordinates")


def set_quadrant_origin(gates, quad_set, x, y) -> int:
    """Move the shared origin of quadrant set ``quad_set`` in ``gates`` (one
    sample's ``{gid: gate}``) to ``(x, y)``, stored coordinates: each
    member's inner corner moves and its outer extent stays. Returns how
    many members moved (the editor's drag does the same, _update_quad_set)."""
    corner = {'Q++': ('x0', 'y0'), 'Q+-': ('x0', 'y1'),
              'Q-+': ('x1', 'y0'), 'Q--': ('x1', 'y1')}
    n = 0
    for g in gates.values():
        if g.get('quad_set') != quad_set:
            continue
        label = g.get('label', '') or ''
        hit = next((v for k, v in corner.items() if k in label), None)
        if hit is None:
            continue
        g[hit[0]], g[hit[1]] = float(x), float(y)
        g['quad_origin_x'], g['quad_origin_y'] = float(x), float(y)
        n += 1
    return n
