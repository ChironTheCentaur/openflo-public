"""Gates as text: a gate's coordinates and bounds as a user reads them
(``gate_value_text``, ``describe_gate``), and the names and boolean nodes a
gate's population takes in a FlowJo workspace (``wsp_population_name``,
``WSP_BOOLEAN_OPS``, ``resolve_wsp_dependent``, ...).

Core module: numpy, no Tk, and it does not import ``openflo.pipeline``
(pipeline imports it).
"""

import warnings

import numpy as np

from .channel_transforms import is_unknown_spec, to_linear
from .scales import OPEN


def gate_value_text(value, spec=None):
    """A gate coordinate as a user should read it, ASCII.

    Gates live in the channel's STORED coordinates, which for fluorescence
    are logicle: a cut at an intensity of ~1,000 is stored as 0.453, and the
    gate list, the population names and the slider printed exactly that.
    With the channel's ``transform_spec`` (`spec`, a ``data_transforms``
    entry) this is the linear intensity the coordinate stands for, written as
    the axis ticks write it (scales.format_intensity_text: '987', '1.23K',
    '50K'); a linear channel's value (`spec` None) is its intensity already
    and is written alike. A channel whose transform is unknown has no
    intensity to give: its stored value, '.3g', which the caller marks."""
    v = float(value)
    if is_unknown_spec(spec):
        return f'{v:.3g}'
    from .scales import format_intensity_text
    return format_intensity_text(float(to_linear(np.array([v]), spec)[0]))


def _gate_axis_text(channel, lo, hi, spec):
    """One axis of a gate, as `describe_gate` says it with transforms:
    'CD3-A >= 1K', 'CD3-A < 2.5K', 'CD3-A in [1K, 2.5K)' (gates are half
    open, lower bound inside); '' when both ends are open (the axis does not
    constrain). An end at the +-1e12 sentinel (or None) is open: printed as a
    number it would read '[-1e+12, 0.453]'."""
    lo_open = lo is None or lo <= -OPEN
    hi_open = hi is None or hi >= OPEN
    if (lo is not None and lo >= OPEN) or (hi is not None
                                           and hi <= -OPEN):
        return f"{channel} (no events)"
    if lo_open and hi_open:
        return ''
    if lo_open:
        text = f"{channel} < {gate_value_text(hi, spec)}"
    elif hi_open:
        text = f"{channel} >= {gate_value_text(lo, spec)}"
    else:
        text = (f"{channel} in [{gate_value_text(lo, spec)}, "
                f"{gate_value_text(hi, spec)})")
    if is_unknown_spec(spec):
        text += " (stored, scale unknown)"
    return text


def wsp_population_name(gate):
    """The name a gate's population takes in an exported FlowJo workspace:
    the user's name or label when it has one, else what it gates on
    ('FSC-A x SSC-A', 'CD3-A'), so the same gate is named the same in every
    sample. Never the editor's gate id, which differs between samples."""
    nm = gate.get('label') or gate.get('name')
    if nm:
        return str(nm)
    x, y = gate.get('x_channel'), gate.get('y_channel')
    if x and y:
        return f"{x} x {y}"
    if gate.get('channel'):
        return str(gate['channel'])
    return str(gate.get('kind') or 'Population').replace('_', ' ').title()


# FlowJo's element for each boolean operation, and back.
WSP_BOOLEAN_TAGS = {'and': 'AndNode', 'or': 'OrNode', 'not': 'NotNode'}
WSP_BOOLEAN_OPS = {v: k for k, v in WSP_BOOLEAN_TAGS.items()}


# Gate kinds with no FlowJo geometry: left out of a .wsp export.
WSP_NO_GEOMETRY = ('cluster', 'category', 'autoclean', 'group')


def wsp_unexportable(gates_by_id):
    """Ids of the gates a FlowJo .wsp cannot carry: the kinds with no
    geometry (WSP_NO_GEOMETRY), and the boolean gates FlowJo cannot express
    -- a NOT over more than one operand (FlowJo's NotNode takes one), an
    unknown operation, or an operand that is missing or itself left out."""
    out = {gid for gid, g in gates_by_id.items()
           if g.get('kind') in WSP_NO_GEOMETRY}
    changed = True
    while changed:
        changed = False
        for gid, g in gates_by_id.items():
            if gid in out or g.get('kind') != 'boolean':
                continue
            op = str(g.get('op', 'and')).lower()
            ops = g.get('operands') or []
            unwritable = (op not in WSP_BOOLEAN_TAGS or not ops
                          or (op == 'not' and len(ops) != 1)
                          or any(o not in gates_by_id or o in out
                                 for o in ops))
            if unwritable:
                out.add(gid)
                changed = True
    return out


def wsp_dependent_name(own_path, target_path):
    """The <Dependent> name of a FlowJo boolean node at `own_path` for the
    population at `target_path` (both tuples of population names from the
    sample down): '..' steps up from the boolean node itself, so a sibling
    operand reads '../CD4+', as FlowJo writes it.

    Deprecated: unused by OpenFlo and will be removed in a future release."""
    warnings.warn('wsp_dependent_name is deprecated and will be removed in a '
                  'future release', DeprecationWarning, stacklevel=2)
    own, target = list(own_path), list(target_path)
    common = 0
    while (common < len(own) - 1 and common < len(target)
           and own[common] == target[common]):
        common += 1
    return '/'.join(['..'] * (len(own) - common) + target[common:])


def resolve_wsp_dependent(name, own_path, known):
    """The population path a FlowJo <Dependent> `name` names, or None.

    `own_path` is the boolean node's path and `known` the population paths
    read for its sample. FlowJo writes a path relative to the boolean node
    ('../CD4+' is a sibling); a name with no '..' is tried below the
    boolean's parent, then from the sample, then as a unique population
    name anywhere in the sample."""
    parts = [p for p in str(name or '').split('/') if p not in ('', '.')]
    if not parts:
        return None
    base = list(own_path)
    for p in parts:
        if p == '..':
            if not base:
                return None
            base.pop()
        else:
            base.append(p)
    tried = [tuple(base)]
    plain = [p for p in parts if p != '..']
    tried += [tuple(own_path[:-1]) + tuple(plain), tuple(plain)]
    for t in tried:
        if t in known:
            return t
    # A population name may itself hold a '/' ('CD3+ / CD4+'), which the
    # split above cut apart: match the written path as a whole.
    written = '/'.join(plain)
    whole = [k for k in known
             if '/'.join(k) == written or '/'.join(k).endswith('/' + written)]
    if len(whole) == 1:
        return whole[0]
    last = [k for k in known if k and k[-1] == plain[-1]] if plain else []
    return last[0] if len(last) == 1 else None


def unique_population_name(name, taken):
    """`name`, or `name (2)`, `name (3)`... when a sibling already holds it
    (FlowJo population paths must be unique). Adds the result to `taken`."""
    out, n = name, 2
    while out in taken:
        out, n = f"{name} ({n})", n + 1
    taken.add(out)
    return out


def describe_gate(gate, transforms=None):
    """Short human-readable label, for logs and the GUI gate list.
    ASCII-only so it survives cp1252 stdout when callers haven't
    reconfigured (e.g. ad-hoc scripts importing flow_pipeline).

    `transforms` ({channel: transform_spec}, the gated sample's
    ``data_transforms``) makes it the user's text: the bounds of a
    threshold, interval or rectangle are the intensities they stand for
    (``gate_value_text``) rather than stored logicle coordinates, open sides
    are said as open ('T  CD3-A >= 987', 'I  CD3-A < 2.5K',
    'R  Q1  FSC-A x CD3-A  FSC-A < 52.3K, CD3-A >= 1K'), and a channel of
    unknown scale keeps its stored value, marked so. Without it the text is
    as it always was (stored coordinates), for logs and library callers."""
    k = gate.get('kind')
    if transforms is not None and k in ('threshold', 'interval', 'rect'):
        return _describe_bounds(gate, k, transforms)
    # Compared with == in order, not hashed: any kind that equals one of
    # these names is described as that kind, as the if-chain this replaced did.
    describe = next((f for kind, f in _DESCRIBE_KIND.items() if k == kind),
                    None)
    return describe(gate) if describe else f"?  {k}"


def _name_prefix(gate):
    """The gate's name or label and two spaces, or '' (`describe_gate`)."""
    nm = gate.get('name') or gate.get('label')
    return f"{nm}  " if nm else ""


def _describe_threshold(gate):
    return f"T  {gate['channel']} >= {float(gate['value']):.3g}"


def _describe_interval(gate):
    return (f"I  {gate['channel']} in "
            f"[{float(gate['lo']):.3g}, {float(gate['hi']):.3g}]")


def _describe_rect(gate):
    pre = _name_prefix(gate)
    return (f"R  {pre}{gate['x_channel']} x {gate['y_channel']}  "
            f"[{float(gate['x0']):.3g},{float(gate['x1']):.3g}] x "
            f"[{float(gate['y0']):.3g},{float(gate['y1']):.3g}]")


def _describe_polygon(gate):
    pre = _name_prefix(gate)
    return (f"P  {pre}{gate['x_channel']} x {gate['y_channel']}  "
            f"({len(gate.get('vertices', []))} verts)")


def _describe_ellipsoid(gate):
    return (f"E  {_name_prefix(gate)}"
            f"{gate.get('x_channel')} x {gate.get('y_channel')}")


def _describe_flowjo_ellipse(gate):
    return (f"E!  {_name_prefix(gate)}"
            f"{gate.get('x_channel')} x {gate.get('y_channel')}  "
            "(FlowJo ellipse, not converted: admits no events)")


def _describe_cluster(gate):
    nm = gate.get('name') or gate.get('label')
    return f"C  {nm}" if nm else f"C  cluster {gate.get('cluster_id')}"


def _describe_category(gate):
    nm = gate.get('name') or gate.get('label') or gate.get('value')
    return f"=  {nm}"


def _describe_boolean(gate):
    nm = gate.get('name')
    if nm:
        return f"B  {nm}"
    op = (gate.get('op') or 'and').upper()
    return f"B  {op}({len(gate.get('operands', []))})"


def _describe_autoclean(gate):
    nm = gate.get('name') or 'autocleaned sample'
    on = sum(1 for m in (gate.get('methods') or [])
             if m.get('enabled', True))
    return f"AC  {nm}  ({on} on)"


def _describe_group(gate):
    return gate.get('name') or 'group'


# `describe_gate` without transforms, per gate kind.
_DESCRIBE_KIND = {
    'threshold': _describe_threshold,
    'interval': _describe_interval,
    'rect': _describe_rect,
    'polygon': _describe_polygon,
    'ellipsoid': _describe_ellipsoid,
    'flowjo_ellipse': _describe_flowjo_ellipse,
    'cluster': _describe_cluster,
    'category': _describe_category,
    'boolean': _describe_boolean,
    'autoclean': _describe_autoclean,
    'group': _describe_group,
}


def _describe_bounds(gate, kind, transforms):
    """`describe_gate` of a threshold / interval / rect in intensity."""
    def axis(ch, lo, hi):
        return _gate_axis_text(ch, None if lo is None else float(lo),
                               None if hi is None else float(hi),
                               transforms.get(ch))
    if kind in ('threshold', 'interval'):
        ch = gate['channel']
        text = (axis(ch, gate['value'], None) if kind == 'threshold'
                else axis(ch, gate['lo'], gate['hi']))
        return f"{'T' if kind == 'threshold' else 'I'}  {text or ch + ' (all events)'}"
    nm = gate.get('name') or gate.get('label')
    pre = f"{nm}  " if nm else ""
    xc, yc = gate['x_channel'], gate['y_channel']
    sides = [t for t in (axis(xc, gate['x0'], gate['x1']),
                         axis(yc, gate['y0'], gate['y1'])) if t]
    return f"R  {pre}{xc} x {yc}  {', '.join(sides) or '(all events)'}"
