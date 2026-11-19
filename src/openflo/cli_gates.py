"""The gates openflo-run applies: the ``--gates`` JSON list and the
``--gates-from`` file (OpenFlo template, GUI session, FlowJo .wsp or
Gating-ML), put on the scale the run gates on, and the one population of a
gate tree that ``--gate-population`` picks.

`--gates` accepts a JSON list of gate dicts (see flow_pipeline.gate_to_mask).
'threshold' kinds feed gate_overrides (which hooks into the existing FMO
threshold merge); everything else feeds region_gates and is applied as a
compound mask via FlowSample.apply_region_gates.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import json
import os

from .gate_lineage import root_first


def parse_gates_arg(s, units='stored'):
    """Returns (gate_overrides_dict, region_gates_list). Empty input → ({}, []).

    `units` 'stored' takes coordinates on the scale the run applies
    (logicle on fluorescence, linear scatter and Time); 'data' takes them in
    data units (intensities) and puts them on that scale."""
    if not s:
        return {}, []
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        print(f"[gates] JSON parse error: {e}", flush=True)
        return {}, []
    if not isinstance(data, list):
        print(f"[gates] expected a JSON list of gate dicts, got "
              f"{type(data).__name__}", flush=True)
        return {}, []
    if units == 'data':
        data = data_units_to_stored([g for g in data if isinstance(g, dict)])

    overrides = {}
    region_gates = []
    for g in data:
        if not isinstance(g, dict):
            print(f"[gates] dropped malformed gate (not a dict): {g!r}",
                  flush=True)
            continue
        kind = g.get('kind')
        if kind == 'threshold':
            try:
                overrides[str(g['channel'])] = float(g['value'])
            except (KeyError, TypeError, ValueError):
                print(f"[gates] dropped threshold gate with missing/invalid "
                      f"channel or value: {g!r}", flush=True)
                continue
        elif kind in ('interval', 'rect', 'polygon', 'ellipsoid'):
            region_gates.append(g)
        elif kind == 'flowjo_ellipse':
            # A FlowJo ellipse OpenFlo could not convert: it admits no events,
            # and skipping it would admit every event it excluded. Kept, so
            # --export-wsp also writes it back as FlowJo drew it.
            print(f"[gates] FlowJo ellipse {g.get('label') or g.get('name') or ''!r} "
                  "was not converted: it admits no events", flush=True)
            region_gates.append(g)
        else:
            print(f"[gates] unknown gate kind {kind!r} — skipped", flush=True)
    return overrides, region_gates


def data_units_to_stored(gates):
    """`gates` with coordinates given in data units (intensities, as the
    FCS file and FlowJo hold them) put on the scale the run gates on
    (pipeline_gate_transforms). Bounds map exactly; a polygon's edges stay
    straight in data units (they are subdivided, wsp_gate_coords.map_gate_coords)."""
    from .wsp_gate_coords import gate_scale_fns, map_gate_coords
    fns = gate_scale_fns(pipeline_gate_transforms(gates), 'from_linear')
    return [map_gate_coords(dict(g), fns) for g in gates]


def pipeline_transform_specs(gates):
    """``{channel: spec}`` of the run's scale for the channels `gates` use
    (see pipeline_gate_transforms), as full transform specs."""
    from .channel_transforms import transform_spec
    return {ch: transform_spec(m)
            for ch, m in pipeline_gate_transforms(gates).items()}


def gate_name(g):
    """Return a display name for a gate dict: its name, label or id, else ``'?'``."""
    return str(g.get('name') or g.get('label') or g.get('id') or '?')


def session_gates(path, data):
    """One sample's gates from a parsed GUI session, on the run's scale:
    the active sample's when it has gates, else the first sample's that
    does. Returns ``(gates, note)``."""
    from .session_format import migrate_session
    from .transform_sidecar import transform_record_from_json
    from .wsp_gate_coords import gate_rescale_fns, map_gate_coords
    data, _notes = migrate_session(data)
    by_sample = data.get('sample_gates') or {}
    order = [s.get('name') for s in data.get('samples') or []
             if isinstance(s, dict)]
    active = data.get('active_sample')
    pick = active if by_sample.get(active) else next(
        (n for n in order + list(by_sample) if by_sample.get(n)), None)
    if pick is None:
        raise ValueError(f"{os.path.basename(path)}: no sample in the "
                         "session has gates")
    entry = next((s for s in data.get('samples') or []
                  if isinstance(s, dict) and s.get('name') == pick), {})
    saved = transform_record_from_json(entry.get('transforms') or {})
    gates = [dict(g) for g in by_sample[pick] if isinstance(g, dict)]
    # Gates sit on the transforms the sample was saved with; the run puts
    # fluorescence on its default logicle and keeps scatter and Time linear.
    fns = gate_rescale_fns(saved, pipeline_transform_specs(gates))
    gates = [map_gate_coords(g, fns) for g in gates]
    others = sum(1 for n, gl in by_sample.items() if gl and n != pick)
    note = (f"the gates of sample '{pick}'"
            + (f"; {others} other sample(s) hold their own, not used"
               if others else ''))
    return gates, note


def read_gate_file(path):
    """Every gate in a gate file, on the run's scale, with ids and
    parent_ids. Reads an OpenFlo JSON template (coordinates as stored, as
    --gates takes them), a GUI session (.flowsession), a FlowJo .wsp and a
    Gating-ML 2.0 .xml. Returns ``(gates, note)``."""
    from .gate_templates import read_template_gates
    from .wsp_gate_coords import gate_scale_fns, map_gate_coords
    low = path.lower()
    if low.endswith('.xml'):
        from .gate_templates import template_ids
        from .gatingml import gates_to_stored, read_gatingml
        raw, warns = read_gatingml(path, raw=True)
        for w in warns:
            print(f"[gates] {os.path.basename(path)}: {w}", flush=True)
        gates = template_ids(gates_to_stored(
            raw, pipeline_transform_specs(raw)))
        return gates, 'Gating-ML'
    if low.endswith('.wsp'):
        gates, _labels = read_template_gates(path)
        fns = gate_scale_fns(pipeline_gate_transforms(gates), 'from_linear')
        return [map_gate_coords(g, fns) for g in gates], 'FlowJo workspace'
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and data.get('format') == 'openflo-session':
        return session_gates(path, data)
    gates, _labels = read_template_gates(path) if low.endswith('.json') \
        else (None, None)
    if gates is None:
        raise ValueError(f"{os.path.basename(path)}: not a gating template, "
                         "session, .wsp or Gating-ML file")
    return gates, 'template'


def population_gates(gates, population=None):
    """The gates that define one population of a gate tree, root first: the
    population's gate and its ancestors, which the run applies together
    (FlowSample.apply_region_gates ANDs them). A boolean gate's operands,
    and their ancestors, come along disabled so it can resolve them.

    `population` names the population; without one the tree must be a
    single chain. Raises ValueError naming the choices otherwise."""
    by_id = {g['id']: g for g in gates if g.get('id') is not None}
    if population:
        want = population.strip().lower()
        hits = [g for g in gates if gate_name(g).lower() == want]
        if len(hits) != 1:
            names = ', '.join(sorted({gate_name(g) for g in gates}))
            raise ValueError(
                (f"no population named {population!r}" if not hits else
                 f"{len(hits)} populations are named {population!r}")
                + f"; the file has: {names}")
        leaf = hits[0]
    else:
        parents = [g.get('parent_id') for g in gates]
        leaves = [g for g in gates if g.get('id') not in parents]
        if len(leaves) > 1:
            raise ValueError(
                "the file holds several populations; choose one with "
                "--gate-population: " + ', '.join(gate_name(g) for g in leaves))
        if not leaves:
            return []
        leaf = leaves[0]

    def chain(g):
        if g is None:
            return []
        return [by_id[gid] for gid in root_first(by_id, g.get('id'))]

    path = chain(leaf)
    keep = {g['id'] for g in path}
    extra, todo = [], [g for g in path if g.get('kind') == 'boolean']
    while todo:
        b = todo.pop()
        for op in b.get('operands') or []:
            for g in chain(by_id.get(op)):
                if g['id'] not in keep:
                    keep.add(g['id'])
                    extra.append(dict(g, enabled=False))
                    if g.get('kind') == 'boolean':
                        todo.append(g)
    return extra + [dict(g) for g in path]


def pipeline_gate_transforms(gates):
    """``{channel: 'logicle'}`` for the fluorescence channels `gates` use.
    The run applies the default logicle to every fluorescence channel
    (FlowSample._classify_channels + apply_transform) BEFORE region gates and
    FMO thresholds, so that is the space their coordinates are in; scatter and
    Time stay linear. WspWriter converts these to FlowJo's linear units."""
    from .channel_roles import is_excluded, is_scatter
    chans = set()
    for g in gates or []:
        for k in ('channel', 'x_channel', 'y_channel'):
            v = g.get(k)
            if isinstance(v, str):
                chans.add(v)
    return {c: 'logicle' for c in chans
            if not is_scatter(c) and not is_excluded(c)}
