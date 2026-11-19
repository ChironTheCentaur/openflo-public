"""Reading the Gating-ML 2.0 that Cytobank exports -- Tk-free.

Cytobank writes standard Gating-ML 2.0 with its own conventions, as its
exports show and the open-source CytoML package reads them (no CytoML code
is used here):

* Every gate sits at the top level, without a ``parent_id``. Each gate's
  ``<data-type:custom_info><cytobank>`` holds its ``<name>`` and a JSON
  ``<definition>`` with the plot ``scale`` it was drawn on (linear ``min``
  and ``max``, per axis ``x`` / ``y`` for a 2-D gate).
* A population is a "GateSet": a ``<BooleanGate>`` whose ``<and>`` names
  every gate on the population's path from all events (in any order; a
  population of one gate names it twice), with the population's name in its
  own ``<cytobank>`` block. The tree follows from the sets: a population's
  parent is the largest other GateSet whose gates are all among its own,
  and its gate is the one gate it adds.
* A gate drawn for one file only (a "tailored" gate) names that file in
  ``<fcs_file_filename>``; the experiment-wide gate leaves it empty.
* A gate side on the edge of its plot scale takes in the events piled on
  that edge, as Cytobank counts them.

``cytobank_tree`` turns such a file's gate elements into the parent-linked
list ``gatingml.read_gatingml`` reads; ``open_scale_edges`` opens a read
gate's sides on its plot edges.

Core module: no Tk.
"""
from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from .gatingml import xml_attr, xml_local
from .scales import OPEN

# (id, parent id, element, local tag), as gatingml.read_gatingml lists the
# gate elements of a file.
GateEl = tuple[str | None, str | None, ET.Element, str]


def cytobank_info(el: ET.Element) -> ET.Element | None:
    """The ``<cytobank>`` block of `el`'s custom_info, or None."""
    for info in el:
        if xml_local(info.tag) != 'custom_info':
            continue
        for c in info:
            if xml_local(c.tag) == 'cytobank':
                return c
    return None


def cytobank_name(el: ET.Element) -> str | None:
    """The name Cytobank gave gate element `el`, or None."""
    info = cytobank_info(el)
    name = (info.findtext('name') or '').strip() if info is not None else ''
    return name or None


def _definition(el: ET.Element) -> dict[str, Any]:
    info = cytobank_info(el)
    text = info.findtext('definition') if info is not None else None
    try:
        d = json.loads(text) if text else {}
    except (ValueError, RecursionError):
        # Bad JSON, or JSON nested too deep to parse: no definition.
        return {}
    return d if isinstance(d, dict) else {}


def _tailored(el: ET.Element) -> bool:
    info = cytobank_info(el)
    return bool(info is not None and (info.findtext('fcs_file_filename') or '').strip())


def _gate_set(el: ET.Element) -> tuple[list[str], bool] | None:
    """(gate ids, any used as complement) of a Cytobank GateSet, or None
    when `el` is not one."""
    info = cytobank_info(el)
    if info is None or info.find('gate_set_id') is None:
        return None
    ands = [c for c in el if xml_local(c.tag) == 'and']
    if len(ands) != 1:
        return None
    refs = [r for r in ands[0] if xml_local(r.tag) == 'gateReference']
    comp = any((xml_attr(r, 'use-as-complement') or '').lower() == 'true'
               for r in refs)
    ids = list(dict.fromkeys(xml_attr(r, 'ref') or '' for r in refs))
    return ids, comp


def named(el: ET.Element, name: str) -> ET.Element:
    """A copy of gate element `el` that read_gatingml names `name`."""
    out = copy.deepcopy(el)
    info = ET.Element('custom_info')
    ET.SubElement(info, 'openflo', {'name': name})
    out.insert(0, info)
    return out


class _Tree:
    """Builds the population list of one Cytobank file (see cytobank_tree)."""

    def __init__(self, gate_els: Sequence[GateEl],
                 quadrants: MutableMapping[ET.Element, ET.Element]) -> None:
        self.quadrants = quadrants
        self.by_id = {gid: (el, tag) for gid, _p, el, tag in gate_els
                      if tag != 'BooleanGate' and not _tailored(el)}
        self.out: list[GateEl] = []
        self.done: list[tuple[str, frozenset[str]]] = []
        self.used: set[str] = set()

    def _parent(self, refs: frozenset[str]) -> tuple[str | None, frozenset[str]]:
        best: tuple[str | None, frozenset[str]] = (None, frozenset())
        for sid, s in self.done:
            if s < refs and len(s) > len(best[1]):
                best = (sid, s)
        return best

    def add(self, sid: str, name: str, ids: list[str]) -> None:
        pid, pset = self._parent(frozenset(ids))
        diff = [r for r in ids if r not in pset]
        for k, ref in enumerate(diff):
            el, tag = self.by_id[ref]
            last = k == len(diff) - 1
            nid = sid if last else f'{sid}#{k}'
            gname = name if last else (cytobank_name(el) or ref)
            copy_el = named(el, gname)
            if tag == 'Quadrant' and el in self.quadrants:
                self.quadrants[copy_el] = self.quadrants[el]
            self.out.append((nid, pid, copy_el, tag))
            pid = nid
        self.used.update(ids)
        self.done.append((sid, frozenset(ids)))


def cytobank_tree(gate_els: Sequence[GateEl],
                  quadrants: MutableMapping[ET.Element, ET.Element],
                  warnings: list[str]) -> list[GateEl]:
    """`gate_els` as a parent-linked population list when they are a
    Cytobank export (GateSets and no parent ids); otherwise unchanged.

    Each GateSet becomes one population: a copy of the gate it adds to its
    parent GateSet, named after the GateSet. A GateSet that adds several
    gates becomes a chain, the gates before the last named after
    themselves. `quadrants` maps a Quadrant element to its QuadrantGate;
    copies of a Quadrant are added to it. Appends to `warnings` what was
    not imported: a GateSet with a gate used as its complement or naming a
    gate that is not in the file, Boolean gates that are not GateSets,
    tailored (per-file) gates, and gates no population uses."""
    if any(pid for _g, pid, _e, _t in gate_els):
        return list(gate_els)
    sets = [(gid, el, _gate_set(el)) for gid, _p, el, tag in gate_els
            if tag == 'BooleanGate']
    sets = [(gid, el, s) for gid, el, s in sets if s is not None]
    if not sets:
        return list(gate_els)
    tree = _Tree(gate_els, quadrants)
    for gid, el, (ids, comp) in sorted(sets, key=lambda t: len(t[2][0])):
        name = cytobank_name(el) or gid or 'unnamed'
        missing = [r for r in ids if r not in tree.by_id]
        if comp or missing or not ids:
            why = ('a gate is used as its complement' if comp else
                   f"gate {missing[0]!r} is not in the file" if missing
                   else 'it names no gate')
            warnings.append(f"'{name}' left out: {why}")
            continue
        tree.add(gid or name, name, ids)
    others = sum(1 for _g, _p, el, tag in gate_els
                 if tag == 'BooleanGate' and _gate_set(el) is None)
    if others:
        warnings.append(f"{others} Boolean gate(s) that are not Cytobank "
                        "populations were not imported")
    _note_unused(gate_els, tree, warnings)
    return tree.out


def _note_unused(gate_els: Sequence[GateEl], tree: _Tree,
                 warnings: list[str]) -> None:
    tailored = sum(1 for _g, _p, el, tag in gate_els
                   if tag != 'BooleanGate' and _tailored(el))
    if tailored:
        warnings.append(f"{tailored} file-specific (tailored) gate(s) not "
                        "imported; each population uses its experiment-wide "
                        "gate")
    unused = [g for g in tree.by_id if g not in tree.used]
    if unused:
        warnings.append(f"{len(unused)} gate(s) no population uses were not "
                        "imported")


# ── Plot edges ───────────────────────────────────────────────────────────────

def _scale(defn: Mapping[str, Any], axis: str | None) -> tuple[float, float] | None:
    s = defn.get('scale')
    if isinstance(s, dict) and axis is not None:
        s = s.get(axis)
    if not isinstance(s, dict):
        return None
    try:
        lo, hi = float(s['min']), float(s['max'])
    except (KeyError, TypeError, ValueError):
        return None
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) and lo < hi else None


def scale_edges(el: ET.Element, chans: Sequence[str],
                ) -> dict[str, tuple[float, float]]:
    """``{channel: (min, max)}``, in linear values, of the Cytobank plot
    scale each of a 2-D gate's channels `chans` (x, y) was drawn on; {}
    when the element gives none."""
    defn = _definition(el)
    out = {}
    for ch, axis in zip(chans, ('x', 'y'), strict=False):
        sc = _scale(defn, axis) if defn else None
        if sc is not None:
            out[ch] = sc
    return out


def _past(v: float, edge: float, low: bool) -> bool:
    tol = 1e-9 * max(1.0, abs(edge))
    return v <= edge + tol if low else v >= edge - tol


def open_scale_edges(g: dict[str, Any], el: ET.Element,
                     to_linear: Mapping[str, Callable[[float], float]]) -> None:
    """Open, in place, each side of read gate `g` (a threshold, interval or
    rect on the file's own scales) that lies on the edge of the Cytobank
    plot scale gate element `el` was drawn on. `to_linear` maps each
    channel's file coordinate to a linear value, as the plot scale is
    given."""
    defn = _definition(el)
    if not defn:
        return
    kind = g.get('kind')
    if kind in ('threshold', 'interval'):
        sides = [('value', 'channel', None, True)] if kind == 'threshold' else [
            ('lo', 'channel', None, True), ('hi', 'channel', None, False)]
    elif kind == 'rect':
        sides = [('x0', 'x_channel', 'x', True), ('x1', 'x_channel', 'x', False),
                 ('y0', 'y_channel', 'y', True), ('y1', 'y_channel', 'y', False)]
    else:
        return
    for key, ch_key, axis, low in sides:
        sc = _scale(defn, axis)
        v = g.get(key)
        if sc is None or v is None or abs(float(v)) >= OPEN:
            continue
        lin = to_linear[g[ch_key]](float(v))
        if _past(lin, sc[0] if low else sc[1], low):
            g[key] = -OPEN if low else OPEN
    _as_read(g)


def _as_read(g: dict[str, Any]) -> None:
    """A 1-D gate in the form read_gatingml gives one with the same open
    sides: a threshold when only its upper side is open."""
    if g.get('kind') == 'threshold' and g['value'] <= -OPEN:
        g.update(kind='interval', lo=-OPEN, hi=OPEN)
        del g['value']
    elif g.get('kind') == 'interval' and g['hi'] >= OPEN and g['lo'] > -OPEN:
        g.update(kind='threshold', value=g.pop('lo'))
        del g['hi']
