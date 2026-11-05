"""Walking a gate up to its root -- Tk-free.

A sample's gates are a dict ``{gate id: gate}``; each gate names its parent
by ``parent_id`` (None, or missing, for a gate drawn on all events). Every
module that needs a gate's ancestry walks it with these, so the rules are
the same everywhere:

* the walk stops at an id that is not in ``gates`` (a root's empty parent,
  or a parent that was deleted);
* it stops at an id it has already met, so a broken file whose gates form a
  cycle still ends.
"""
from __future__ import annotations

from collections.abc import Mapping


def lineage(gates: Mapping, gid) -> list:
    """``gid`` and its ancestors, nearest first: ``[gid, parent, ...,
    root]``. Empty when ``gid`` is not in ``gates``. Cycle-safe."""
    out, seen, cur = [], set(), gid
    while cur in gates and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = gates[cur].get('parent_id')
    return out


def ancestors(gates: Mapping, gid) -> list:
    """``gid``'s ancestors, nearest first, ending at a root gate (``gid``
    itself left out). Cycle-safe."""
    return lineage(gates, gid)[1:]


def root_first(gates: Mapping, gid) -> list:
    """``gid``'s chain from the root down to ``gid``: the order its gates
    are applied in. Cycle-safe."""
    return lineage(gates, gid)[::-1]


def descendants(gates: Mapping, gid) -> set:
    """Every gate under ``gid`` (children, theirs, ...), ``gid`` left out.
    Cycle-safe."""
    out: set = set()
    frontier = [gid] if gid in gates else []
    while frontier:
        pid = frontier.pop()
        for cid, g in gates.items():
            if g.get('parent_id') == pid and cid not in out and cid != gid:
                out.add(cid)
                frontier.append(cid)
    return out
