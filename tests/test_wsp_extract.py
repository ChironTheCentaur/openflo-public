"""WSP gate extraction, on a generated workspace. Covers:

  - extract_gates returns exactly the kinds that were written, of those the
    editor and writer understand
  - the parent_id chain is the one that was written, with no dangling ids
  - serialising to a v2 template and reloading preserves the kind histogram

This used to run only on a real FlowJo workspace someone had to supply, so it
never ran. The workspace is now written by OpenFlo itself from a gate tree
whose shape is known, which lets the checks be exact.
"""
import json
from collections import Counter

import openflo.pipeline as fp

# A nested tree: cells > singlets > {FL1+, FL2 mid, blob}. The ellipse goes
# out as a polygon (FlowJo drops a sample with an ellipse gate).
GATES = [
    {'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
     'x0': 5e4, 'x1': 2.5e5, 'y0': 3e4, 'y1': 2e5,
     'id': 'g1', 'parent_id': None, 'name': 'cells'},
    {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'FSC-H',
     'vertices': [[1e4, 5e3], [2.5e5, 1e5], [2.5e5, 1.4e5], [1e4, 9e3]],
     'id': 'g2', 'parent_id': 'g1', 'name': 'singlets'},
    {'kind': 'threshold', 'channel': 'FL1-A', 'value': 1000.0,
     'id': 'g3', 'parent_id': 'g2', 'name': 'FL1+'},
    {'kind': 'interval', 'channel': 'FL2-A', 'lo': 100.0, 'hi': 5000.0,
     'id': 'g4', 'parent_id': 'g2', 'name': 'FL2 mid'},
    {'kind': 'ellipsoid', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
     'mean': [2000.0, 1000.0], 'cov': [[4e5, 0.0], [0.0, 1e5]],
     'distance_sq': 4.0, 'id': 'g5', 'parent_id': 'g2', 'name': 'blob'},
]
CHANNELS = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A']
EXPECTED_KINDS = Counter({'polygon': 2, 'rect': 1, 'threshold': 1,
                          'interval': 1})


def generated_workspace(path, gates=GATES):
    w = fp.WspWriter(cytometer='generated')
    w.add_sample('generated-1', fcs_path='', channels=CHANNELS, gates=gates)
    w.write(str(path))
    return str(path)


def _depths(gates):
    """Each gate's depth in the tree, by following parent_id."""
    by_id = {g['_import_id']: g for g in gates}

    def depth(g):
        d = 0
        while g.get('parent_id') is not None:
            g = by_id[g['parent_id']]
            d += 1
        return d
    return sorted(depth(g) for g in gates)


def test_extract_gives_back_the_tree_that_was_written(tmp_path):
    """Positive: the kinds, the nesting and the template round trip."""
    gates = fp.WspReader(generated_workspace(tmp_path / 'g.wsp')).extract_gates()
    kinds = Counter(g.get('kind') for g in gates)
    assert kinds == EXPECTED_KINDS

    ids = {g['_import_id'] for g in gates}
    assert len(ids) == len(gates), "_import_id must be unique per gate"
    for g in gates:
        p = g.get('parent_id')
        assert p is None or p in ids, f"dangling parent_id={p!r}"
    # One root, one child of it, three grandchildren.
    assert _depths(gates) == [0, 1, 2, 2, 2]

    # Serialise like the editor's Save Template would, and reload.
    imp_to_eid = {g['_import_id']: f"g{i + 1}" for i, g in enumerate(gates)}
    out = []
    for g in gates:
        c = {k: v for k, v in g.items() if k != '_import_id'}
        c['id'] = imp_to_eid[g['_import_id']]
        c['parent_id'] = imp_to_eid.get(g.get('parent_id')) \
            if g.get('parent_id') else None
        out.append(c)
    path = tmp_path / 't.json'
    path.write_text(json.dumps({'name': 'rt', 'version': 2, 'gates': out}),
                    encoding='utf-8')
    reloaded = json.loads(path.read_text(encoding='utf-8'))['gates']
    assert Counter(g.get('kind') for g in reloaded) == kinds


def test_extract_does_not_invent_nesting_or_gates(tmp_path):
    """Negative: the same gates written flat come back flat, and a sample
    with no gates gives none."""
    flat = [dict(g, parent_id=None) for g in GATES]
    gates = fp.WspReader(generated_workspace(tmp_path / 'f.wsp',
                                             flat)).extract_gates()
    assert Counter(g.get('kind') for g in gates) == EXPECTED_KINDS
    assert _depths(gates) == [0, 0, 0, 0, 0]

    empty = fp.WspReader(generated_workspace(tmp_path / 'e.wsp', [])
                         ).extract_gates()
    assert empty == []
