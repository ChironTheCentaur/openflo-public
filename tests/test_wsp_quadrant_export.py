"""WspWriter must export an editor quadrant the way FlowJo itself stores one.

The editor draws a quadrant as four rect gates that share a ``quad_set``.
FlowJo 10.x stores a quadrant as four sibling ``<Population>`` elements, each
with its own ``<gating:RectangleGate>``; it does not write a
``<gating:QuadrantGate>`` element. WspWriter used to fold the four rects into
one ``"Quadrants"`` population that held a ``<gating:QuadrantGate>``. That
element does not match the Gating-ML 2.0 QuadrantGate type: its dividers have
no id, they wrap the fcs-dimension in a ``<gating:dimension>``, and there is
no ``<gating:Quadrant>``. The fold also lost the four sector names, and it
moved a child of ONE sector onto the whole quadrant. On re-import that child
went under a different sector.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import openflo.pipeline as fp

G = '{http://www.isac-net.org/std/Gating-ML/v2.0/gating}'
D = '{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}'
XDIV, YDIV = 100.0, 200.0
BIG = 1e6
# id, label, x0, x1, y0, y1
SECTORS = [
    ('q_ul', 'UL', -BIG, XDIV, YDIV, BIG),
    ('q_ur', 'UR', XDIV, BIG, YDIV, BIG),
    ('q_lr', 'LR', XDIV, BIG, -BIG, YDIV),
    ('q_ll', 'LL', -BIG, XDIV, -BIG, YDIV),
]
CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


def _gates():
    gates = [{'id': 'root', 'parent_id': None, 'name': 'Root',
              'kind': 'interval', 'channel': 'FSC-A', 'lo': 0.0, 'hi': BIG}]
    for gid, label, x0, x1, y0, y1 in SECTORS:
        gates.append({'id': gid, 'parent_id': 'root', 'label': label,
                      'kind': 'rect', 'x_channel': 'FL1-A',
                      'y_channel': 'FL2-A',
                      'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                      'quad_set': 'qs1',
                      'quad_origin_x': XDIV, 'quad_origin_y': YDIV})
    # A child of the upper-left sector ONLY.
    gates.append({'id': 'kid', 'parent_id': 'q_ul', 'name': 'Child of UL',
                  'kind': 'interval', 'channel': 'SSC-A',
                  'lo': 0.0, 'hi': 500.0})
    return gates


def _write(tmp_path):
    w = fp.WspWriter()
    w.add_sample('s1', fcs_path='', channels=CHANNELS, gates=_gates())
    out = tmp_path / 'quad.wsp'
    w.write(str(out))
    return out


def _children(elem):
    return [p for s in elem.findall('Subpopulations')
            for p in s.findall('Population')]


def test_quadrant_exports_as_four_rectangle_populations(tmp_path):
    root = ET.parse(_write(tmp_path)).getroot()
    sample_node = root.find('SampleList/Sample/SampleNode')
    (parent,) = [p for p in _children(sample_node) if p.get('name') == 'Root']
    sectors = {p.get('name'): p for p in _children(parent)}

    assert sorted(sectors) == sorted(s[1] for s in SECTORS), (
        f"expected the four sector populations under 'Root', got "
        f"{sorted(sectors)}")
    for _gid, label, x0, x1, y0, y1 in SECTORS:
        rect = sectors[label].find(f'Gate/{G}RectangleGate')
        assert rect is not None, f'sector {label} has no RectangleGate'
        dims = [(d.find(f'{D}fcs-dimension').get(f'{D}name'),
                 float(d.get(f'{G}min')), float(d.get(f'{G}max')))
                for d in rect.findall(f'{G}dimension')]
        assert dims == [('FL1-A', x0, x1), ('FL2-A', y0, y1)], label
    # The child is nested under the sector it was drawn in, and nowhere else.
    assert [c.get('name') for c in _children(sectors['UL'])] == ['Child of UL']
    for label in ('UR', 'LR', 'LL'):
        assert _children(sectors[label]) == [], label
    assert not list(root.iter(f'{G}QuadrantGate'))


def test_quadrant_child_keeps_its_sector_through_wsp_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    n = 4000
    df = pd.DataFrame({
        'FSC-A': rng.uniform(0.0, 1000.0, n),
        'SSC-A': rng.uniform(0.0, 1000.0, n),
        'FL1-A': rng.uniform(-500.0, 700.0, n),
        'FL2-A': rng.uniform(-500.0, 900.0, n),
    })
    before = {g['id']: g for g in _gates()}
    want = np.asarray(fp.cumulative_gate_mask(before, 'kid', df), dtype=bool)

    back, _ = fp.read_template_gates(str(_write(tmp_path)))
    after = {g['id']: g for g in back}
    (kid_id,) = [g['id'] for g in back
                 if (g.get('label') or g.get('name')) == 'Child of UL']
    got = np.asarray(fp.cumulative_gate_mask(after, kid_id, df), dtype=bool)

    assert 0 < int(want.sum()) < n
    assert int(got.sum()) == int(want.sum()), (
        f"child of the UL sector selects {int(got.sum())} events after "
        f"export + re-import, {int(want.sum())} before")
    assert np.array_equal(got, want)
