"""Gate templates: ``read_template_gates`` reads the gate tree a template
file (v2 JSON, FlowJo ``.wsp`` or Gating-ML 2.0 ``.xml``) holds, and
``wsp_template_stat_axes`` the GeoMean axes of a ``.wsp`` template's sample.

Core module: the standard library. No Tk, and it does not import
``openflo.pipeline``.
"""

import logging
import os

from .wsp_gate_coords import wsp_stat_axes
from .wsp_reader import WspReader

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


# ── Gate templates ────────────────────────────────────────────────────────────


def template_ids(raw):
    """Imported gates (`_import_id` keys, see WspReader.extract_gates) with
    ids rewritten to the gN form, so the returned shape is consistent
    regardless of source."""
    from .gating import remap_operands
    imp_to_id = {g.get('_import_id'): f'g{i}' for i, g in enumerate(raw, 1)}
    gates = []
    for g in raw:
        d = dict(g)
        d.pop('_import_id', None)
        d['id'] = imp_to_id[g.get('_import_id')]
        pid = g.get('parent_id')
        d['parent_id'] = imp_to_id.get(pid) if pid else None
        if d.get('kind') == 'boolean':
            remap_operands(d, imp_to_id)
        gates.append(d)
    return gates


def _wsp_template_node(reader):
    """(gates, <SampleNode>) of the sample a `.wsp` template takes its gate
    tree from: the first that has gates; ([], None) when none has."""
    for sn in (reader.root.iter('SampleNode')
               if reader.root is not None else ()):
        raw = reader.extract_gates(sample_node=sn)
        if raw:
            return raw, sn
    return [], None


def wsp_template_stat_axes(path):
    """`wsp_stat_axes` of the sample a `.wsp` template's gates come from
    (see read_template_gates), so a sample the template is applied to takes
    its GeoMean on the same axes; {} when no sample holds gates."""
    reader = WspReader(path)
    _raw, used = _wsp_template_node(reader)
    if used is None or reader.root is None:
        return {}
    for sample in reader.root.iter('Sample'):
        if any(sn is used for sn in sample.iter('SampleNode')):
            return wsp_stat_axes(sample)
    return {}


def read_template_gates(path):
    """Read gates from a v2 JSON template, a FlowJo `.wsp` workspace or a
    standalone Gating-ML 2.0 `.xml` file.

    Returns `(gates, labels)` where:
      * `gates` is a list[dict] of gate definitions. Each entry has an
        `id` field (allocated `g1`, `g2`, … as needed) and a `parent_id`
        that references another id within the same list (or None for a
        root). Schema otherwise matches `gate_to_mask`.
      * `labels` is a {detector: antibody_label} dict from the template,
        or None if the source doesn't carry labels.

    Callers that want their own id namespace (the editor allocates ids
    per sample) can remap by walking the list in order and substituting
    parent_id pointers via an old->new map.

    A `.wsp` holds a gate tree per sample; the template is ONE of them, the
    first sample's that has gates (the log names it). Concatenating every
    sample's tree would give a 2-sample workspace of 5 gates as 10, and each
    target sample would get every population twice.

    `.wsp` gates are in linear units. Gating-ML gates stay on the file's own
    scales, each with its '_gml_axes': put them on a sample's stored scales
    with `gatingml.gates_to_stored`.
    """
    import json as _json
    p = path.lower()
    if p.endswith('.xml'):
        # Standalone Gating-ML 2.0: one gate tree, kept on the file's own
        # scales (each gate's '_gml_axes'); gatingml.gates_to_stored puts it
        # on a sample's.
        from .gatingml import read_gatingml
        raw, _warnings = read_gatingml(path, raw=True)
        return template_ids(raw), None
    if p.endswith('.wsp'):
        reader = WspReader(path)
        nodes = (list(reader.root.iter('SampleNode'))
                 if reader.root is not None else [])
        raw, used = _wsp_template_node(reader)
        if used is None:
            # No sample holds a <Population> tree: fall back to the
            # reader's flat scan of bare gates.
            raw = reader.extract_gates()
        else:
            others = sum(1 for sn in nodes if sn is not used
                         and next(sn.iter('Population'), None) is not None)
            log.info("[template] %s: the gate tree of sample '%s' (%d "
                     "gate(s))%s", os.path.basename(path),
                     used.get('name') or '?', len(raw),
                     f"; {others} other sample(s) hold their own tree, not "
                     "used" if others else '')
        return template_ids(raw), None

    if p.endswith('.json'):
        with open(path, encoding='utf-8') as f:
            data = _json.load(f)
        if not isinstance(data, dict):
            raise ValueError("template JSON must be an object")
        gates_field = data.get('gates')
        if not isinstance(gates_field, list):
            raise ValueError(
                "template JSON must have a 'gates' field of type list")
        labels = (data.get('labels')
                  if isinstance(data.get('labels'), dict) else None)
        out = [dict(g) for g in gates_field
               if isinstance(g, dict) and g.get('kind')]
        return out, labels

    raise ValueError(f"unsupported template format: {path}")
