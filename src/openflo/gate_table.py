"""Every sample's gates as one table: population path, kind, channels and
coordinates, both as stored and in linear (compensated intensity) units, the
units a FlowJo workspace or Gating-ML file holds. Tk-free.

The row a script would otherwise build by reading the gate tree back out of
a saved .wsp.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping

COLUMNS = ('Sample', 'Population', 'Gate', 'Kind', 'Parent', 'Enabled',
           'Channels', 'Coordinates (stored)', 'Coordinates (linear)')

# The keys of a gate dict that carry its geometry.
_COORD_KEYS = ('value', 'lo', 'hi', 'x0', 'x1', 'y0', 'y1', 'vertices',
               'mean', 'cov', 'quad_origin_x', 'quad_origin_y', 'operands',
               'op', 'clusters', 'categories')


def _coords(g: Mapping) -> str:
    """The gate's geometry as compact JSON (keys in a fixed order)."""
    out = {k: g[k] for k in _COORD_KEYS if g.get(k) is not None}
    return json.dumps(out, separators=(',', ':'), default=str)


def gate_rows(sample: str, gates: Mapping[str, Mapping],
              order: Iterable[str] | None = None,
              transforms: Mapping | None = None) -> list[dict]:
    """One row per gate of one sample, in ``order`` (default: dict order).
    ``transforms`` is the sample's ``data_transforms``: with it, the linear
    column has every coordinate on a transformed channel converted back to
    intensity; a channel of unknown scale stays as stored, and says so."""
    from .gating import gate_channels, population_path
    from .pipeline import gate_scale_fns, is_unknown_spec, map_gate_coords
    rec = dict(transforms or {})
    fns = gate_scale_fns(rec, 'to_linear') if rec else {}
    unknown = {ch for ch, sp in rec.items()
               if is_unknown_spec(sp if isinstance(sp, dict)
                                  else {'method': sp})}
    rows = []
    ids = [gid for gid in (order or gates) if gid in gates]
    ids += [gid for gid in gates if gid not in ids]
    for gid in ids:
        g = gates[gid]
        chans = sorted(gate_channels(g))
        pid = g.get('parent_id')
        kind = g.get('kind') or ''
        if kind in ('boolean', 'cluster', 'category'):
            linear = _coords(g)
        else:
            try:
                linear = _coords(map_gate_coords(g, fns) if fns else g)
            except Exception as exc:                  # noqa: BLE001
                linear = f'NA ({type(exc).__name__})'
            stuck = sorted(set(chans) & unknown)
            if stuck:
                linear += f"  [unknown scale, as stored: {', '.join(stuck)}]"
        rows.append({
            'Sample': sample,
            'Population': population_path(gates, gid, transforms=rec or None),
            'Gate': str(g.get('name') or gid),
            'Kind': kind,
            'Parent': (population_path(gates, str(pid), transforms=rec or None)
                       if pid in gates else ''),
            'Enabled': bool(g.get('enabled', True)),
            'Channels': ', '.join(chans),
            'Coordinates (stored)': _coords(g),
            'Coordinates (linear)': linear,
        })
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'Enabled': 'yes' if r['Enabled'] else 'no'})
