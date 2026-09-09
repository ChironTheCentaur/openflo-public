"""Real-WSP gate-extraction regression. Skips when no .wsp is present
(see conftest::real_wsp_path). Covers:

  - extract_gates returns the same kinds as the editor's v2 template
    expects (threshold / interval / rect / polygon / ellipsoid / quadrant)
  - parent_id chain is internally consistent
  - serialising to v2 template + reloading preserves the kind histogram
"""
import json
import os
import tempfile
from collections import Counter

import openflo.pipeline as fp


def test_real_wsp_extract_and_serialise(real_wsp_path):
    reader = fp.WspReader(real_wsp_path)
    gates = reader.extract_gates()
    assert gates, "real WSP should extract at least one gate"

    kinds = Counter(g.get('kind') for g in gates)
    # The kinds the editor + writer actually understand.
    known = {'threshold', 'interval', 'rect', 'polygon',
             'ellipsoid', 'quadrant'}
    unknown = {k for k in kinds if k not in known and k is not None}
    assert not unknown, f"unexpected gate kinds: {unknown}"

    # Internal parent-id consistency (each _import_id is unique, each
    # parent_id refers to a real gate or None).
    ids = {g['_import_id'] for g in gates}
    assert len(ids) == len(gates), "_import_id must be unique per gate"
    for g in gates:
        p = g.get('parent_id')
        assert p is None or p in ids, (
            f"gate {g.get('name', g['_import_id'])} has dangling "
            f"parent_id={p!r}")

    # Serialise like the editor's Save Template would.
    imp_to_eid = {}
    out_gates = []
    for i, g in enumerate(gates):
        eid = f"g{i+1}"
        imp_to_eid[g['_import_id']] = eid
        copy = dict(g)
        copy.pop('_import_id', None)
        copy['id'] = eid
        out_gates.append(copy)
    for g in out_gates:
        pid = g.get('parent_id')
        g['parent_id'] = imp_to_eid.get(pid) if pid else None

    template = {
        'name': 'roundtrip',
        'version': 2,
        'gates': out_gates,
    }
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(template, f)
        tmp_path = f.name
    try:
        with open(tmp_path, encoding='utf-8') as f:
            reloaded = json.load(f)['gates']
        assert len(reloaded) == len(gates)
        rk = Counter(g.get('kind') for g in reloaded)
        assert rk == kinds, f"kind histogram drifted: {dict(kinds)} -> {dict(rk)}"
    finally:
        os.unlink(tmp_path)
