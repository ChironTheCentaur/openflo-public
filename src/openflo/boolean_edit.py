"""Making and changing a boolean gate (AND / OR / NOT) -- Tk-free.

The rules behind the desktop's Boolean gate dialog
(editor_gatetools._open_boolean_dialog), the same ones the phone's Combine
dialog keeps (mobile/src/boolean.ts): a boolean gate may combine boolean
gates too (as FlowJo nests them, and as gate_eval counts them), and a
boolean gate's choice, gates and name can be changed afterwards, never so
that it rests on itself (``gate_store.depends_on``, the engine's rule).

Core module: no Tk.
"""
from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

from .gate_lineage import lineage
from .gate_store import Gate, depends_on

# The choices, in the dialog's order, with what each keeps.
BOOL_OPS: tuple[tuple[str, str, str], ...] = (
    ('AND', 'and', "Keep events inside EVERY ticked gate (the overlap)."),
    ('OR', 'or', "Keep events inside ANY ticked gate (all of them "
     "together)."),
    ('NOT', 'not', "Keep events OUTSIDE the ticked gates (with several "
     "ticked: outside all of them)."),
)
_OPS = tuple(op for _label, op, _why in BOOL_OPS)


def default_name(op: str, n: int) -> str:
    """The name a boolean gate gets when none is typed: 'AND (2)'."""
    return f"{op.upper()} ({n})"


def combinable(gates: Mapping[str, Gate], order: Iterable[str],
               blocked: Collection[str] = ()) -> list[str]:
    """The ids of `order` a boolean gate may combine, in that order: every
    gate of `gates` (a boolean one too), but a FlowJo ellipse OpenFlo could
    not convert and every gate under it (they count nothing, so NOT of one
    would read as every event), and but the gates in `blocked` (for a
    boolean being changed: ``resting_on`` it)."""
    return [gid for gid in order
            if gid in gates and gid not in blocked
            and not any(gates[cur].get('kind') == 'flowjo_ellipse'
                        for cur in lineage(gates, gid))]


def resting_on(gates: Mapping[str, Gate], target: str) -> set[str]:
    """The gates of `gates` whose population rests on gate `target`,
    `target` too: the gates under it and the boolean gates combining any
    of them, and so on. A boolean changed to combine one of them would
    rest on itself."""
    return {gid for gid in gates if depends_on(gates, [gid], target)}


def editing(before: Mapping[str, Any]) -> tuple[str, list[str], str]:
    """The choice, operands and name boolean gate `before` is shown with
    for a change: its name blank when it is the one it would get by
    default, so the default follows the new choice."""
    op = str(before['op']) if before.get('op') in _OPS else 'and'
    raw = before.get('operands')
    operands: list[str] = (list(raw) if isinstance(raw, (list, tuple))
                           else [])
    name = str(before.get('name') or '')
    return op, operands, ('' if name == default_name(op, len(operands))
                          else name)


def unshown_operands(operands: Iterable[str], present: Collection[str],
                     offered: Collection[str]
                     ) -> tuple[list[str], list[str]]:
    """The operands of a boolean gate being changed that the dialog cannot
    show: (lost, kept). `lost` name no gate of the sample (`present`) any
    more, and Save leaves them out; `kept` are gates of the sample it does
    not offer (`offered`: an unconverted FlowJo ellipse, or a gate under
    one), and Save keeps them, so an AND over one is never quietly
    widened."""
    ops = list(operands)
    return ([o for o in ops if o not in present],
            [o for o in ops if o in present and o not in offered])


def unshown_note(lost: int, kept: int) -> str:
    """What the dialog says of a changed gate's unshown operands ('' for
    none)."""
    parts = []
    if lost:
        parts.append("A gate it combined was deleted; Save leaves it out."
                     if lost == 1 else
                     f"{lost} gates it combined were deleted; Save leaves "
                     "them out.")
    if kept:
        which = ("A gate it combines is" if kept == 1
                 else f"{kept} gates it combines are")
        parts.append(f"{which} not listed (an unconverted FlowJo ellipse, "
                     f"or a gate under one); Save keeps "
                     f"{'it' if kept == 1 else 'them'}.")
    return ' '.join(parts)


def kept_operands(was: Iterable[str], ticked: Iterable[str],
                  kept: Iterable[str]) -> list[str]:
    """The operands a changed boolean gate is saved with: of those it had
    (`was`), the ticked and the `kept` ones in their order, then those
    newly ticked."""
    was, ticked = list(was), list(ticked)
    keep = {*ticked, *kept}
    return ([o for o in was if o in keep]
            + [o for o in ticked if o not in was])


def new_boolean(op: str, operands: Iterable[str], typed: str
                ) -> Gate | str:
    """The boolean gate to add (a root gate), or why there is none."""
    ops = list(operands)
    if not ops:
        return "Select at least one gate."
    name = typed.strip() or default_name(op, len(ops))
    return {'kind': 'boolean', 'op': op, 'operands': ops, 'name': name,
            'parent_id': None}


def changed_boolean(before: Mapping[str, Any], op: str,
                    operands: Iterable[str], typed: str) -> Gate | str:
    """Boolean gate `before` (as stored) with the choice, operands and
    name of the edit, or why there is none. Its place in the tree, colour
    and the rest are kept; an imported gate's label follows the name."""
    made = new_boolean(op, operands, typed)
    if isinstance(made, str):
        return made
    gate = {k: v for k, v in before.items() if k != 'id'}
    gate.update(op=op, operands=made['operands'], name=made['name'])
    if 'label' in gate:
        gate['label'] = made['name']
    return gate


def same_boolean(before: Mapping[str, Any], after: Mapping[str, Any]
                 ) -> bool:
    """Whether `after` changes boolean gate `before` at all."""
    return (before.get('op') == after.get('op')
            and before.get('name') == after.get('name')
            and list(before.get('operands') or [])
            == list(after.get('operands') or []))
