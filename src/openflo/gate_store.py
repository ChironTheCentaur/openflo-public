"""Adding, replacing and removing gates in a sample's gate set -- Tk-free.

A sample's gates are a dict ``{gate id: gate}`` plus a list of the ids in
the order they were added; a gate names its parent by ``parent_id`` (None
for a gate on all events). Gate ids are ``g1``, ``g2``, ... from a
per-sample counter that only goes up, so an id is never handed out twice
in a sample, even after a delete.

These functions hold the rules for changing such a set, so every front end
(the Tk gate editor, a headless document) keeps them the same way:

* a new threshold or interval gate replaces the one already on its channel
  under the same parent (a histogram slider moves one gate instead of
  stacking new ones), keeping that gate's colour, on/off state, name and
  expand state;
* deleting a gate deletes the gates under it;
* a gate list from elsewhere (a reopened session, a view tab, an imported
  workspace) is installed with fresh ids, its parents and boolean operands
  remapped (``install_gates``);
* within a sample, a quadrant-set id (``quad_set``, shared by the four
  rects of one quadrant, which move together) names one set only: gates
  arriving from elsewhere keep theirs unless another set there has it
  (``separate_quad_sets``).

Undo, the audit trail and redrawing stay with the caller. The dict and the
list are changed in place, never rebound, so a caller holding them (the
editor's active-sample shortcuts) stays in step.

Core module: no Tk.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, MutableMapping
from typing import Any, NamedTuple

from .gate_lineage import subtree

# A gate: a dict of its kind, channels, bounds, parent and look (see
# gate_eval for the keys each kind has).
Gate = dict[str, Any]
_GateSet = MutableMapping[str, Gate]

# The 1-D gate kinds: a gate on one channel (a threshold or an interval).
# Within a set, one of them is unique per (channel, parent).
ONE_D_KINDS = ('threshold', 'interval')

# What a replaced 1-D gate keeps of the gate it replaces: everything the user
# set on it, so nudging it changes only its position.
_KEEP_ON_REPLACE = ('color', 'enabled', 'name', 'open')


class Added(NamedTuple):
    """What ``add_gate`` did: the gate's id, and whether it replaced a 1-D
    twin (True) or was stored under a new id from the counter (False)."""
    id: str
    replaced: bool


def depends_on(gates: Mapping[str, Gate], start: Iterable[Any],
               target: str) -> bool:
    """Whether gate `target` is among the ids in `start` or anything their
    populations rest on: each gate's parent and, for a boolean, its
    operands, followed all the way up. A gate given `start` as its parent
    and operands would rest on itself when this is True. Ids not in
    `gates` (and None) are skipped; cycle-safe."""
    seen: set[str] = set()
    stack = [g for g in start if g is not None]
    while stack:
        gid = stack.pop()
        if gid == target:
            return True
        if gid in seen or gid not in gates:
            continue
        seen.add(gid)
        g = gates[gid]
        if g.get('parent_id') is not None:
            stack.append(g['parent_id'])
        if g.get('kind') == 'boolean':
            stack.extend(op for op in (g.get('operands') or ())
                         if op is not None)
    return False


def operand_problem(gates: Mapping[str, Gate], g: Mapping[str, Any],
                    gid: str | None = None,
                    restore: bool = False) -> str | None:
    """Why gate `g` (to be stored as `gid`, None for a new gate) cannot go
    in `gates`, or None when it can: a boolean gate naming no operands or
    one no gate of `gates` is (other than those gate `gid` names already,
    or any with `restore`), or a gate that would rest on itself: an
    operand that is the gate or depends on it, or a parent that is the
    gate, is under it or depends on it through an operand
    (``depends_on``, which follows both).

    A boolean that lost an operand (or every one) keeps naming it: a change
    to it (a rename, Undo pointing it back) names what it named already,
    and only a newly missing gate is refused. `restore` (Undo putting the
    gate back as it was saved) also accepts an operand no gate is, and no
    operands on a gate stored as a boolean: it comes back still lost,
    selecting nothing, as a deleted one is put back."""
    ops: list[Any] = []
    stored = gates.get(gid) if gid is not None else None
    had = (list(stored.get('operands') or [])
           if stored is not None and stored.get('kind') == 'boolean'
           else None)
    keep = g.get('operands')
    if restore and isinstance(keep, (list, tuple)) and keep:
        had = [*(had or ()), *keep]
    if g.get('kind') == 'boolean':
        raw = g.get('operands')
        if not isinstance(raw, (list, tuple)) or (
                not raw and had != [] and not (restore and had is not None)):
            return "a boolean gate needs the gates it combines ('operands')"
        ops = list(raw)
        for op in ops:
            if op not in gates and op not in (had or ()):
                return f"no gate {op!r} to combine"
    if gid is not None and depends_on(gates, [g.get('parent_id'), *ops], gid):
        return (f"gate {gid!r} cannot rest on itself or on a gate that "
                "rests on it")
    return None


def gate_id(seq: int) -> str:
    """The gate id for counter value `seq` (``g<seq>``)."""
    return f'g{seq}'


def next_gate_id(seqs: MutableMapping[str, int], sample: str) -> str:
    """Advance `sample`'s counter in `seqs` (0 when it has none) and return
    the id it now gives."""
    seqs[sample] = seqs.get(sample, 0) + 1
    return gate_id(seqs[sample])


def insert_gate(gates: _GateSet, order: list[str], gid: str,
                gate: Gate) -> None:
    """Store `gate` under the new id `gid`, last in `order`."""
    gates[gid] = gate
    order.append(gid)


def one_d_twin(gates: _GateSet, gate: Gate,
               among: bool | set[str] = True) -> str | None:
    """The id of the gate a new 1-D `gate` replaces: the first threshold or
    interval gate on the same channel under the same parent. None when
    `gate` is not a 1-D gate or there is none. `among` limits the search:
    True looks at every gate, False at none, a set at those ids only."""
    if gate.get('kind') not in ONE_D_KINDS or not among:
        return None
    ch, pid = gate['channel'], gate['parent_id']
    for gid, g in gates.items():
        if among is not True and gid not in among:
            continue
        if (g.get('kind') in ONE_D_KINDS and g.get('channel') == ch
                and g.get('parent_id') == pid):
            return gid
    return None


def update_gate(gates: _GateSet, gid: str, gate: Gate,
                keep: Iterable[str] = _KEEP_ON_REPLACE, *,
                in_place: bool = False) -> Gate:
    """Put `gate` in the place of gate `gid`, keeping the old gate's values
    of the keys in `keep` (those it has). Returns the stored gate.

    By default `gate` itself is stored (and given the kept values). With
    `in_place` the old gate's dict is kept and refilled from `gate`, which
    is left as it was: for a gate other code may already hold."""
    old = gates[gid]
    kept = {k: old[k] for k in keep if k in old}
    if in_place:
        old.clear()
        old.update(gate, **kept)
        return old
    gate.update(kept)
    gates[gid] = gate
    return gate


def add_gate(gates: _GateSet, order: list[str], gate: Gate,
             new_id: Callable[[], str], parent_id: str | None = None,
             replace_1d: bool | set[str] = True) -> Added:
    """Add `gate` to the set; returns its id and whether it replaced a
    twin (``Added``).

    `gate` gets `parent_id` when it names no parent, and is switched on
    unless it says otherwise. A threshold or interval gate replaces its
    twin (``one_d_twin``) in place and returns the twin's id; `replace_1d`
    limits that for bulk installs, whose sibling 1-D gates on one channel
    are distinct populations: False never replaces, and a set of ids
    replaces only those, each at most once (a replaced id leaves the set).
    Otherwise `new_id()` is called for the gate's id, which is stored last
    in `order`, and the result says it was not a replacement even if that
    id was already taken (the counter's caller decides what that means)."""
    gate.setdefault('parent_id', parent_id)
    gate.setdefault('enabled', True)
    twin = one_d_twin(gates, gate, replace_1d)
    if twin is not None:
        update_gate(gates, twin, gate)
        if isinstance(replace_1d, set):
            replace_1d.discard(twin)
        return Added(twin, True)
    gid = new_id()
    insert_gate(gates, order, gid, gate)
    return Added(gid, False)


def remove_gate(gates: _GateSet, order: list[str], gid: str,
                cascade: bool = True) -> list[str]:
    """Remove gate `gid` and, with `cascade`, every gate under it. Returns
    the ids removed, in the set's order; empty when `gid` is not in it.
    Without `cascade` the gates under it are left to the caller (they then
    name a parent that is gone)."""
    if gid not in gates:
        return []
    victims = set(subtree(gates, gid)) if cascade else {gid}
    removed = [g for g in gates if g in victims]
    for victim in removed:
        del gates[victim]
        if victim in order:
            order.remove(victim)
    return removed


def clear_gates(gates: _GateSet, order: list[str]) -> int:
    """Remove every gate and empty `order`; returns how many gates there
    were."""
    n = len(gates)
    gates.clear()
    order.clear()
    return n


def install_gates(gates_by_id: dict[str, Gate], order: list[str],
                  pending: list[Gate], new_id: Callable[[], str], *,
                  fill: Callable[[Gate], None] | None = None
                  ) -> dict[str, str]:
    """Add the gate list `pending` (a restored session's or an imported
    workspace's) to a sample's gate set; returns ``{saved id: new id}``.

    Each gate is copied and given a fresh id from `new_id` (as from
    ``next_gate_id``); its
    saved id is ``_import_id`` or ``id``. ``parent_id`` and a boolean's
    operands are remapped to the new ids; a parent not added before its
    child is lost (None). Gates without a saved on/off state come in on.
    Sibling 1-D gates on one channel are distinct populations: none
    replaces another. `fill`, when given, fills in the caller's defaults
    on each gate (a colour, a parent) just before it is added.
    A quadrant set keeps its saved ``quad_set`` id unless a set already in
    the sample, or another one in `pending`, has it; it then takes a
    ``qsN`` past every id there (``separate_quad_sets``)."""
    from .gating import remap_operands
    taken = quad_set_ids(gates_by_id.values())
    old_to_new: dict[str, str] = {}
    added = []
    for raw in pending:
        g = dict(raw)
        src_id = g.pop('_import_id', None) or g.pop('id', None)
        parent = g.get('parent_id')
        if parent is not None:
            g['parent_id'] = old_to_new.get(parent)
        g.setdefault('enabled', True)
        if fill is not None:
            fill(g)
        gid = add_gate(gates_by_id, order, g, new_id, replace_1d=False).id
        added.append(g)
        if src_id is not None:
            old_to_new[src_id] = gid
    # A session saves each gate's editor id and the restore hands out fresh
    # ones, so once a gate has been deleted the ids shift and a boolean's
    # saved operands would name other gates. Remapped after the loop, when
    # every id is known.
    for g in added:
        remap_operands(g, old_to_new)
    separate_quad_sets(added, taken)
    return old_to_new


_QUAD_SET_ID = re.compile(r'qs(\d+)')


def quad_set_key(gate: Gate) -> tuple:
    """What names one quadrant set in a sample: its ``quad_set`` id,
    channel pair and parent. A drag or typed divider moves the members with
    the key of the one grabbed; a reopened session can hold two sets under
    one id on other axes or parents."""
    return (gate.get('quad_set'), gate.get('x_channel'),
            gate.get('y_channel'), gate.get('parent_id'))


def quad_set_ids(gates: Iterable[Gate]) -> set[str]:
    """The quadrant-set ids (``quad_set``) the `gates` use."""
    return {g['quad_set'] for g in gates if g.get('quad_set')}


def highest_quad_set_number(ids: Iterable[str]) -> int:
    """The highest N of the ``qsN`` quadrant-set ids in `ids`; 0 when there
    is none. A new set's id goes past it, so no set there has it."""
    nums = [int(m.group(1)) for m in map(_QUAD_SET_ID.fullmatch, ids) if m]
    return max(nums, default=0)


def separate_quad_sets(added: Iterable[Gate], taken: Collection[str],
                       new_id: Callable[[], str] | None = None) -> int:
    """Give each quadrant set among the gates just `added` to a sample an
    id no other set in that sample has, changing their ``quad_set`` in
    place. Returns how many sets took a new id.

    A set is the members that share a ``quad_set`` id, channel pair and
    parent. It keeps its id unless `taken` (the ids of the sample's other
    gates) or a set before it in `added` has it; then it takes `new_id()`
    (by default ``qsN`` past every id here). Dragging a quadrant's centre
    moves every member with its id, so a pasted, copied, templated or
    reopened quadrant sharing the id of one already there (on other axes,
    perhaps) would move that one as well, unseen."""
    added = list(added)
    used = set(taken)
    n = highest_quad_set_number(used | quad_set_ids(added))

    def past_every_id() -> str:
        nonlocal n
        n += 1
        return f'qs{n}'
    next_id = new_id or past_every_id
    given: dict[tuple, str] = {}
    renamed = 0
    for g in added:
        qs = g.get('quad_set')
        if not qs:
            continue
        key = quad_set_key(g)
        if key not in given:
            fresh = qs
            while fresh in used:
                fresh = next_id()
            renamed += fresh != qs
            given[key] = fresh
            used.add(fresh)
        g['quad_set'] = given[key]
    return renamed
