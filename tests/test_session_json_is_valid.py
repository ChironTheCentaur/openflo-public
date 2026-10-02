"""A session file must be JSON that other tools can read.

`Infinity`, `-Infinity` and `NaN` are Python extensions to JSON. `json.dump`
writes them and `json.load` reads them back, so a session containing one is
perfectly valid *here* and rejected by `JSON.parse`, `jq`, and every other
RFC 8259 reader — a failure that can only ever surface somewhere else.

They arrive by a legitimate route. Gating-ML expresses an open bound as
`min="-INF"`, which `float()` turns into `-inf`. The MISSING-bound case a few
lines away in the same parser already used a ±1e12 sentinel for exactly this
reason, with the comment "JSON-safe, unlike -inf"; the EXPLICIT-INF case did
not.

Two defences: the parser clamps non-finite bounds and vertices, and the writer
sanitises whatever reaches it and reports what it changed. The writer repairs
rather than raising — refusing to save would cost a user their work over a
value we can represent perfectly well.
"""
import json
import math
import os
import tempfile

import pytest

from openflo import pipeline as fp
from openflo.editor_session import _json_safe

OPEN_BOUNDS_WSP = '''<?xml version="1.0" encoding="UTF-8"?>
<Workspace xmlns:gating="g" xmlns:data-type="d">
 <SampleList><Sample><SampleNode name="s1"><Subpopulations>
  <Population name="Open">
   <Gate><gating:RectangleGate>
     <gating:dimension gating:min="-INF" gating:max="1000">
       <data-type:fcs-dimension data-type:name="FSC-A"/></gating:dimension>
     <gating:dimension gating:min="500" gating:max="INF">
       <data-type:fcs-dimension data-type:name="SSC-A"/></gating:dimension>
   </gating:RectangleGate></Gate>
  </Population>
 </Subpopulations></SampleNode></Sample></SampleList></Workspace>'''


def _strict(text):
    """Parse the way a non-Python reader does: no Infinity, no NaN."""
    def reject(constant):
        raise ValueError(f'not valid JSON: {constant}')
    return json.loads(text, parse_constant=reject)


@pytest.fixture
def open_bounds_gates():
    path = os.path.join(tempfile.mkdtemp(), 'inf.wsp')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(OPEN_BOUNDS_WSP)
    gates, _ = fp.read_template_gates(path)
    return gates


def test_open_gating_ml_bounds_do_not_become_infinity(open_bounds_gates):
    assert len(open_bounds_gates) == 1
    rect = open_bounds_gates[0]
    for key in ('x0', 'x1', 'y0', 'y1'):
        assert math.isfinite(rect[key]), f'{key} came back as {rect[key]}'


def test_such_a_gate_serialises_to_readable_json(open_bounds_gates):
    _strict(json.dumps(open_bounds_gates, allow_nan=False))


def test_the_open_side_is_still_effectively_unbounded(open_bounds_gates):
    """The sentinel has to be far enough out to behave like no bound at all
    on any real channel."""
    rect = open_bounds_gates[0]
    assert rect['x0'] <= -1e12
    assert rect['y1'] >= 1e12
    assert rect['x1'] == pytest.approx(1000.0)   # the real bounds are intact
    assert rect['y0'] == pytest.approx(500.0)


def test_the_sanitiser_replaces_what_json_cannot_carry():
    reported = []
    state = {'a': float('inf'), 'b': float('-inf'), 'c': float('nan'),
             'd': [1.0, float('inf')], 'e': 'text', 'f': 3}
    safe = _json_safe(state, lambda where, value, rep:
                      reported.append(where))

    assert safe['a'] == 1e12
    assert safe['b'] == -1e12
    assert safe['c'] is None
    assert safe['d'] == [1.0, 1e12]
    assert safe['e'] == 'text' and safe['f'] == 3
    assert sorted(reported) == ['state.a', 'state.b', 'state.c', 'state.d[1]']

    _strict(json.dumps(safe, allow_nan=False))


def test_the_sanitiser_leaves_ordinary_state_alone():
    state = {'gates': [{'lo': -1.5, 'hi': 2.0}], 'name': 'A', 'n': 7,
             'flag': True, 'missing': None}
    assert _json_safe(state) == state


def test_a_polygon_vertex_at_infinity_is_not_written():
    """The same hazard in the other geometry parser."""
    wsp = OPEN_BOUNDS_WSP.replace(
        '<gating:RectangleGate>', '<gating:PolygonGate>').replace(
        '</gating:RectangleGate>', '</gating:PolygonGate>')
    path = os.path.join(tempfile.mkdtemp(), 'poly.wsp')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(wsp)
    gates, _ = fp.read_template_gates(path)
    _strict(json.dumps(gates, allow_nan=False))
