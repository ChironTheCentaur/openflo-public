"""A session moves between the headless Document (the phone) and the gate
editor (the desktop) with the same gates, counts and statistics, both ways.

* Document -> desktop: the Document opens FCS files and gates them, saves a
  session, and the real editor opens it through its own session restore
  (``_load_session_path`` and its loaders, run on this thread).
* Desktop -> Document: the editor loads and gates the files and saves (a
  compensated sample gets its processed-data sidecar), and the Document
  opens that session; then the Document saves it again (copying the
  sidecar) and the editor opens that.

Counts, %Parent, %Total and the Median, Mean, GeoMean and CV of every
fluorescence channel are compared value for value, and the gates field by
field. Generated FCS files; channels FSC-A, SSC-A, FL1-A ...; sample names
s1, s2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from openflo.document import Document
from tests.conftest import gui_unavailable
from tests.test_document import FL, spill_fcs
from tests.test_session_restore_transforms import _restore
from tests.test_transform_records import _new_editor, capture_warnings, isolate_home

WANT = {'Count', '%Parent', '%Total', 'Median', 'Mean', 'GeoMean', 'CV'}

# Gates in the editor's schema: a scatter rectangle, a polygon and a
# threshold under it, an interval, a boolean of two of them, and booleans
# nested on it (NOT of the boolean, with a gate under it; OR of that and
# the polygon), as the phone makes them and FlowJo nests them.
GATES = [
    ({'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
      'x0': 3e4, 'x1': 7e4, 'y0': 1e4, 'y1': 5e4, 'name': 'cells'}, None),
    ({'kind': 'polygon', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
      'vertices': [[0.3, 0.2], [0.9, 0.2], [0.9, 0.8], [0.3, 0.8]],
      'name': 'double'}, 0),
    ({'kind': 'threshold', 'channel': 'FL1-A', 'value': 0.5,
      'name': 'FL1+'}, 0),
    ({'kind': 'interval', 'channel': 'FL2-A', 'lo': 0.3, 'hi': 0.6,
      'name': 'FL2 mid'}, None),
    ({'kind': 'boolean', 'op': 'and', 'operands': [2, 3],
      'name': 'FL1+ and mid'}, 0),
    ({'kind': 'boolean', 'op': 'not', 'operands': [4],
      'name': 'not both'}, None),
    ({'kind': 'threshold', 'channel': 'FL3-A', 'value': 0.4,
      'name': 'FL3+'}, 5),
    ({'kind': 'boolean', 'op': 'or', 'operands': [6, 1],
      'name': 'FL3+ or double'}, None),
]


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)       # never a real (blocking) modal
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, tmp_path
    root.destroy()


def _files(tmp):
    """s1 compensated by its $SPILLOVER, s2 without one."""
    return {'s1': spill_fcs(tmp / 'raw' / 's1.fcs', seed=1),
            's2': spill_fcs(tmp / 'raw' / 's2.fcs', seed=2, spill=False)}


def _concrete(gate, ids):
    """`gate` with its boolean operands (positions in GATES) as ids."""
    g = dict(gate)
    if 'operands' in g:
        g['operands'] = [ids[i] for i in g['operands']]
    return g


def _for(name, gate):
    """`gate` as sample `name` takes it."""
    if name == 's2' and gate['kind'] == 'polygon':
        return dict(gate, y_channel='FL3-A')
    return gate


def _gate_doc(doc, name):
    ids = []
    for gate, parent in GATES:
        gate = _for(name, gate)
        ids.append(doc.add_gate(name, _concrete(gate, ids),
                                None if parent is None else ids[parent])['id'])


def _gate_editor(ed, name):
    ed._set_active_sample(name)
    ids = []
    for gate, parent in GATES:
        gate = _for(name, gate)
        ids.append(ed._add_gate(_concrete(gate, ids),
                                None if parent is None else ids[parent]))


def _shape(gates, order):
    """A gate set as comparable data: per gate in order, its fields but the
    look (colour, expand state, stamps) with ids replaced by positions."""
    pos = {gid: i for i, gid in enumerate(order)}
    out = []
    for gid in order:
        g = {k: v for k, v in gates[gid].items()
             if k not in ('color', 'open', 'enabled') and not k.startswith('_')}
        g['parent_id'] = pos.get(g.get('parent_id'))
        if 'operands' in g:
            g['operands'] = [pos[o] for o in g['operands']]
        out.append(g)
    return out


def _nan(v):
    return float('nan') if v is None else float(v)


def _doc_numbers(doc, name):
    out = {}
    for r in doc.stats(name, all_events=False):
        out[r['population']] = (
            [_nan(r['count']), _nan(r['pct_parent']), _nan(r['pct_total'])]
            + [_nan(r['channels'][ch][st]) for ch in FL
               for st in ('median', 'mean', 'geomean', 'cv')])
    return out


def _editor_numbers(ed, name):
    out = {}
    for r in ed._sample_rows(name, WANT):
        out[r['Population']] = (
            [_nan(r['Count']), _nan(r['%Parent']), _nan(r['%Total'])]
            + [_nan(r.get(f'{st} {ch}')) for ch in FL
               for st in ('Median', 'Mean', 'GeoMean', 'CV')])
    return out


def _assert_same_numbers(a, b):
    assert sorted(a) == sorted(b)
    for pop in a:
        for x, y in zip(a[pop], b[pop], strict=True):
            assert (math.isnan(x) and math.isnan(y)) or x == y, (pop, a[pop],
                                                                  b[pop])


def _assert_same(doc, ed, names):
    for name in names:
        rec = doc._samples[name]
        assert _shape(rec.gates, rec.order) == _shape(
            ed._sample_gates[name], ed._sample_gate_order[name]), name
        numbers = _doc_numbers(doc, name)
        assert len(numbers) == len(GATES)
        assert numbers['cells'][0] > 100            # the gates select
        _assert_same_numbers(numbers, _editor_numbers(ed, name))


def test_a_document_session_opens_on_the_desktop_the_same(env):
    root, tmp = env
    files = _files(tmp)
    doc = Document()
    for name in ('s1', 's2'):
        assert doc.open_fcs(files[name]) == name
        _gate_doc(doc, name)
    path = tmp / 'phone.flowsession'
    doc.save_session(str(path))
    ed = _restore(root, path, raw_first=True)
    assert sorted(ed._samples) == ['s1', 's2']
    assert ed._samples['s1'].comp_channels == FL
    _assert_same(doc, ed, ['s1', 's2'])


def test_a_desktop_session_opens_in_a_document_the_same(env):
    root, tmp = env
    from tests.test_session_restore_transforms import _loaded
    files = _files(tmp)
    ed = _new_editor(root)
    for name in ('s1', 's2'):
        ed._on_loaded(name, _loaded(files[name]))
        _gate_editor(ed, name)
    path = tmp / 'desk.flowsession'
    data = ed._write_session(str(path))
    # The compensated sample reopens from its sidecar, the other from FCS.
    assert [bool(e.get('processed_data')) for e in data['samples']] == [
        True, False]
    doc = Document()
    assert doc.open_session(str(path))['samples'] == ['s1', 's2']
    assert doc._samples['s1'].sidecar
    _assert_same(doc, ed, ['s1', 's2'])
    # Saved again by the Document, elsewhere: the sidecar goes with it.
    (tmp / 'copy').mkdir()
    again = tmp / 'copy' / 'phone.flowsession'
    doc.save_session(str(again))
    ed2 = _restore(root, again, raw_first=False)
    assert ed2._samples['s1'].path.startswith(str(tmp / 'copy'))
    _assert_same(doc, ed2, ['s1', 's2'])
    np.testing.assert_array_equal(ed2._samples['s1'].data.to_numpy(),
                                  ed._samples['s1'].data.to_numpy())
