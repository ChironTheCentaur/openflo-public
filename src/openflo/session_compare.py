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

import numpy as np

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
    for s in data.get('samples') or []:
        if isinstance(s, dict) and s.get('name') == name:
            rec = s.get('transforms')
            return dict(rec) if isinstance(rec, dict) else None
    return None


def _convert(gates: dict, src: Mapping | None, dst: Mapping | None):
    """``(gates, unknown channels)``: ``gates`` (stored on ``src``'s scales)
    moved onto ``dst``'s, channel by channel where the two differ."""
    from .pipeline import gate_scale_fns, is_unknown_spec, map_gate_coords
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
    from .pipeline import cumulative_gate_mask
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
        except Exception:                             # noqa: BLE001
            out[gid] = None
    return out


def compare_sessions(current: Mapping[str, tuple], data: Mapping,
                     progress: Callable[[int, int, str], None] | None = None
                     ) -> list[dict]:
    """Rows for every population of every sample loaded now that the saved
    session (``data``, parsed and migrated) also has.

    ``current`` is ``{sample: (gates, frame, transforms)}``: this session's
    gate dict, the loaded events and the sample's ``data_transforms``."""
    from .gating import gate_channels, population_keys, population_path
    rows: list[dict] = []
    names = [n for n in current if saved_gates(data, n) or n in {
        s.get('name') for s in data.get('samples') or []
        if isinstance(s, dict)}]
    for i, name in enumerate(names):
        if progress:
            progress(i, len(names), f"Counting {name}…")
        gates, frame, rec = current[name]
        old_raw = saved_gates(data, name)
        old, unknown = _convert(old_raw, saved_transforms(data, name), rec)
        now_c, old_c = _counts(gates, frame), _counts(old, frame)
        now_k = {k: gid for gid, k in population_keys(gates).items()}
        old_k = {k: gid for gid, k in population_keys(old).items()}
        for key in list(now_k) + [k for k in old_k if k not in now_k]:
            a, b = now_k.get(key), old_k.get(key)
            path = (population_path(gates, a, transforms=rec) if a
                    else population_path(old, str(b), transforms=rec))
            na = now_c.get(a) if a else None
            nb = old_c.get(b) if b else None
            notes = []
            if a is None:
                notes.append('only in the saved session')
            elif b is None:
                notes.append('only in this session')
            if b is not None and _chain_channels(old, b, gate_channels) \
                    & unknown:
                nb = None
                notes.append('saved gate on a channel of unknown scale')
            if a and b and _chain(gates, a) != _chain(old_raw, b) \
                    and not unknown:
                notes.append('gate changed')
            cnt_a = na[0] if na else None
            cnt_b = nb[0] if nb else None
            change = (cnt_a - cnt_b if cnt_a is not None and cnt_b is not None
                      else None)
            rows.append({
                'Sample': name, 'Population': path,
                'Count now': cnt_a, 'Count saved': cnt_b,
                'Change': change,
                '% change': (100.0 * change / cnt_b if change is not None
                             and cnt_b else float('nan')),
                '%Parent now': (100.0 * na[0] / na[1] if na and na[1]
                                else float('nan')),
                '%Parent saved': (100.0 * nb[0] / nb[1] if nb and nb[1]
                                  else float('nan')),
                'Note': '; '.join(notes),
                # A count on one side only (a population, or a gate that
                # cannot be counted, missing from the other) moved too.
                'moved': bool(change) or (cnt_a is None) != (cnt_b is None),
            })
    return rows


def _chain(gates: dict, gid) -> list:
    """The geometry of ``gid`` and its ancestors, ids left out, to tell
    whether a population is defined the same way on both sides."""
    out, seen = [], set()
    while gid in gates and gid not in seen:
        seen.add(gid)
        g = gates[gid]
        out.append({k: v for k, v in g.items()
                    if k not in ('parent_id', 'color', 'enabled', 'label',
                                 '_import_id', 'name')})
        gid = g.get('parent_id')
    return out


def _chain_channels(gates, gid, gate_channels) -> set:
    chans: set = set()
    seen = set()
    while gid in gates and gid not in seen:
        seen.add(gid)
        chans |= set(gate_channels(gates[gid]))
        gid = gates[gid].get('parent_id')
    return chans
