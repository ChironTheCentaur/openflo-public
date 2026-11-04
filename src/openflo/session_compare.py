"""Compare with saved session: each population's count with this session's
gates beside its count with a saved session's gates, on the same loaded
events, so a rerun or a re-gating shows exactly which numbers moved.
Tk-free.

Samples are matched by name and populations by their cross-sample key
(``gating.population_keys``: place in the tree and name), never by gate id.
Both sides are counted on the events loaded NOW, so a difference is the
gates'. A saved gate on a channel whose stored scale differs from the loaded
sample's is converted through linear units (its saved transform record), or,
when either scale is unknown, reads NA rather than a count on the wrong
scale.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np

from .errors import note
from .gate_lineage import lineage

COLUMNS = ('Sample', 'Population', 'Count now', 'Count saved', 'Change',
           '% change', '%Parent now', '%Parent saved', 'Note')


def saved_gates(data: Mapping, name: str) -> dict:
    """``{id: gate}`` of sample ``name`` in a parsed session dict."""
    out = {}
    for i, g in enumerate((data.get('sample_gates') or {}).get(name) or []):
        if isinstance(g, dict):
            gid = str(g.get('id') or f'saved{i}')
            out[gid] = {k: v for k, v in g.items() if k != 'id'}
    return out


def saved_transforms(data: Mapping, name: str) -> dict | None:
    """The ``transforms`` record of sample ``name`` in a parsed session dict, or
    None when the sample or its record is missing."""
    for s in data.get('samples') or []:
        if isinstance(s, dict) and s.get('name') == name:
            rec = s.get('transforms')
            return dict(rec) if isinstance(rec, dict) else None
    return None


def _convert(gates: dict, src: Mapping | None, dst: Mapping | None):
    """``(gates, unknown channels)``: ``gates`` (stored on ``src``'s scales)
    moved onto ``dst``'s, channel by channel where the two differ."""
    from .channel_transforms import is_unknown_spec
    from .wsp_gate_coords import gate_scale_fns, map_gate_coords
    if src is None or dst is None:
        return gates, set()
    differ = {ch for ch in set(src) | set(dst) if src.get(ch) != dst.get(ch)}
    if not differ:
        return gates, set()
    unknown = {ch for ch in differ
               if is_unknown_spec(src.get(ch)) or is_unknown_spec(dst.get(ch))}
    lin = gate_scale_fns({c: src[c] for c in differ - unknown if c in src},
                         'to_linear')
    back = gate_scale_fns({c: dst[c] for c in differ - unknown if c in dst},
                          'from_linear')
    out = {}
    for gid, g in gates.items():
        g2 = map_gate_coords(g, lin) if lin else g
        out[gid] = map_gate_coords(g2, back) if back else g2
    return out, unknown


def _counts(gates: dict, frame) -> dict:
    """``{gid: (count, parent count)}``; a gate that cannot be evaluated
    (a missing channel, say) maps to None."""
    from .gate_eval import cumulative_gate_mask
    cache: dict = {}
    n = len(frame)
    out: dict = {}
    for gid, g in gates.items():
        try:
            m = np.asarray(cumulative_gate_mask(gates, gid, frame,
                                                cache=cache), dtype=bool)
            pid = g.get('parent_id')
            pm = (np.asarray(cumulative_gate_mask(gates, pid, frame,
                                                  cache=cache), dtype=bool)
                  if pid in gates else None)
            out[gid] = (int(m.sum()), int(pm.sum()) if pm is not None else n)
        except Exception:                             # noqa: BLE001 - noted; a gate that cannot be counted on this frame shows as None in the comparison
            note("counting events in a gate for comparison")
            out[gid] = None
    return out


def compare_sessions(current: Mapping[str, tuple], data: Mapping,
                     progress: Callable[[int, int, str], None] | None = None
                     ) -> list[dict]:
    """Rows for every population of every sample loaded now that the saved
    session (``data``, parsed and migrated) also has.

    ``current`` is ``{sample: (gates, frame, transforms)}``: this session's
    gate dict, the loaded events and the sample's ``data_transforms``."""
    rows: list[dict] = []
    names = [n for n in current if _in_saved(data, n)]
    for i, name in enumerate(names):
        if progress:
            progress(i, len(names), f"Counting {name}…")
        rows.extend(_sample_rows(name, current[name], data))
    return rows


def _in_saved(data: Mapping, name) -> bool:
    """Whether the saved session has gates for, or a sample record of,
    ``name``."""
    return bool(saved_gates(data, name)) or name in {
        s.get('name') for s in data.get('samples') or []
        if isinstance(s, dict)}


class _Sides(NamedTuple):
    """One sample on both sides: this session's gates, the saved ones as
    stored (``old_raw``) and moved onto the loaded scales (``old``), the
    channels whose scale is unknown, the loaded transforms and both sides'
    ``_counts``."""
    gates: dict
    old_raw: dict
    old: dict
    unknown: set
    rec: Mapping | None
    now_c: dict
    old_c: dict


def _sample_rows(name: str, entry: tuple, data: Mapping) -> list[dict]:
    """The rows of one sample: this session's populations in their order,
    then those only the saved session has."""
    from .gating import population_keys
    gates, frame, rec = entry
    old_raw = saved_gates(data, name)
    old, unknown = _convert(old_raw, saved_transforms(data, name), rec)
    now_c, old_c = _counts(gates, frame), _counts(old, frame)
    sides = _Sides(gates, old_raw, old, unknown, rec, now_c, old_c)
    now_k = {k: gid for gid, k in population_keys(gates).items()}
    old_k = {k: gid for gid, k in population_keys(old).items()}
    return [_population_row(name, now_k.get(key), old_k.get(key), sides)
            for key in list(now_k) + [k for k in old_k if k not in now_k]]


def _population_row(name: str, a, b, sides: _Sides) -> dict:
    """The row of one population: ``a`` its gate id now, ``b`` its saved
    one (None on the side that lacks it)."""
    from .gating import gate_channels, population_path
    path = (population_path(sides.gates, a, transforms=sides.rec) if a
            else population_path(sides.old, str(b), transforms=sides.rec))
    na = sides.now_c.get(a) if a else None
    nb = sides.old_c.get(b) if b else None
    notes = []
    if a is None:
        notes.append('only in the saved session')
    elif b is None:
        notes.append('only in this session')
    if b is not None and _chain_channels(sides.old, b, gate_channels) \
            & sides.unknown:
        nb = None
        notes.append('saved gate on a channel of unknown scale')
    if a and b and _chain(sides.gates, a) != _chain(sides.old_raw, b) \
            and not sides.unknown:
        notes.append('gate changed')
    return _row(name, path, na, nb, notes)


def _row(name: str, path, na, nb, notes: list) -> dict:
    """A table row from both sides' ``(count, parent count)`` (None where a
    side has no count)."""
    cnt_a = na[0] if na else None
    cnt_b = nb[0] if nb else None
    change = (cnt_a - cnt_b if cnt_a is not None and cnt_b is not None
              else None)
    return {
        'Sample': name, 'Population': path,
        'Count now': cnt_a, 'Count saved': cnt_b,
        'Change': change,
        '% change': _pct(change, cnt_b),
        '%Parent now': _pct(cnt_a, na[1] if na else None),
        '%Parent saved': _pct(cnt_b, nb[1] if nb else None),
        'Note': '; '.join(notes),
        # A count on one side only (a population, or a gate that
        # cannot be counted, missing from the other) moved too.
        'moved': bool(change) or (cnt_a is None) != (cnt_b is None),
    }


def _pct(part, whole) -> float:
    """``100 * part / whole``, NaN when either is missing or ``whole`` is
    0."""
    return 100.0 * part / whole if part is not None and whole else float('nan')


def _chain(gates: dict, gid) -> list:
    """The geometry of ``gid`` and its ancestors, ids left out, to tell
    whether a population is defined the same way on both sides."""
    return [{k: v for k, v in gates[cur].items()
             if k not in ('parent_id', 'color', 'enabled', 'label',
                          '_import_id', 'name')}
            for cur in lineage(gates, gid)]


def _chain_channels(gates, gid, gate_channels) -> set:
    chans: set = set()
    for cur in lineage(gates, gid):
        chans |= set(gate_channels(gates[cur]))
    return chans
