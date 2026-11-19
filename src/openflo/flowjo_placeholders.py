"""FlowJo ellipse placeholders: the boolean gates built on a 'flowjo_ellipse'
placeholder (``placeholder_booleans``) and the status-bar note that names them
(``flowjo_placeholder_note``).

Core module: no Tk, and it does not import ``openflo.pipeline``, which serves
``placeholder_booleans`` under the same name and ``_flowjo_placeholder_note``,
deprecated, under its old name.
"""
from .gate_eval import depends_on_flowjo_placeholder
from .gate_text import describe_gate


def placeholder_booleans(gates):
    """Names of the boolean gates in `gates` (gate dicts of ONE id namespace:
    a template, or one sample's saved gates; ids in '_import_id' or 'id')
    that rest on a 'flowjo_ellipse' placeholder and so select nothing."""
    by_id = {}
    for g in gates:
        gid = g.get('_import_id') or g.get('id')
        if gid is not None:
            by_id[gid] = g
    return [g.get('name') or describe_gate(g) for g in gates
            if g.get('kind') == 'boolean' and any(
                depends_on_flowjo_placeholder(by_id, op)
                for op in g.get('operands') or ())]


def flowjo_placeholder_note(gates, booleans=None):
    """Status-bar note for the 'flowjo_ellipse' placeholders in `gates`
    (gate dicts), or '' when there are none. It also names the boolean gates
    built on them, which select nothing too: `booleans`, or when None those
    `placeholder_booleans` finds in `gates`."""
    names = [g.get('label') or g.get('name') or '?' for g in gates
             if g.get('kind') == 'flowjo_ellipse']
    if not names:
        return ''
    shown = ', '.join(names[:3]) + (f' (+{len(names) - 3})' if len(names) > 3 else '')
    note = (f" — {len(names)} FlowJo ellipse(s) not converted ({shown}): "
            "they and their subpopulations count no events (see log)")
    booleans = list(dict.fromkeys(placeholder_booleans(gates)
                                  if booleans is None else booleans))
    if booleans:
        bshown = ', '.join(f"'{b}'" for b in booleans[:3]) + (
            f' (+{len(booleans) - 3})' if len(booleans) > 3 else '')
        note += f"; so do the boolean gate(s) built on them: {bshown}"
    return note
