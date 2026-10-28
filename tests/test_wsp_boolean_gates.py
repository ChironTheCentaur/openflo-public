"""FlowJo AND / OR / NOT populations survive a .wsp import and export.

FlowJo writes a boolean population as an <AndNode>, <OrNode> or <NotNode>
whose <Dependent> elements name the populations it combines by their path
from the sample ('Cells/A'; OpenFlo's earlier exports wrote '../A' for a
sibling, which still reads). The reader only walked <Population> elements, so these vanished
on import with no warning, together with everything gated under them, and
the writer left OpenFlo's boolean gates out of every export.

Generated data: 2,000 events with FL1-A and FL2-A positives at known rows,
so each population's count is computed independently with numpy. The
negative cases: an operand FlowJo's file does not have reads NA rather
than a number, and booleans FlowJo cannot express are still left out.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.gating import population_stats

_GATES = [
    {'id': 'p', 'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
     'x0': 1000.0, 'x1': 2e5, 'y0': 1000.0, 'y1': 2e5, 'parent_id': None,
     'name': 'Cells'},
    {'id': 'a', 'kind': 'threshold', 'channel': 'FL1-A', 'value': 1000.0,
     'parent_id': 'p', 'name': 'A'},
    {'id': 'b', 'kind': 'threshold', 'channel': 'FL2-A', 'value': 1000.0,
     'parent_id': 'p', 'name': 'B'},
    {'id': 'and', 'kind': 'boolean', 'op': 'and', 'operands': ['a', 'b'],
     'parent_id': 'p', 'name': 'A and B'},
    {'id': 'or', 'kind': 'boolean', 'op': 'or', 'operands': ['a', 'b'],
     'parent_id': 'p', 'name': 'A or B'},
    {'id': 'not', 'kind': 'boolean', 'op': 'not', 'operands': ['a'],
     'parent_id': 'p', 'name': 'not A'},
    {'id': 'kid', 'kind': 'threshold', 'channel': 'FSC-A', 'value': 50000.0,
     'parent_id': 'and', 'name': 'big'},
]


def _events():
    rng = np.random.default_rng(11)
    n = 2000
    fsc = rng.uniform(2000, 1e5, n)
    fsc[:100] = 10.0                                  # outside Cells
    fl1 = np.where(rng.random(n) < 0.4, rng.uniform(2e3, 9e3, n),
                   rng.uniform(1, 500, n))
    fl2 = np.where(rng.random(n) < 0.3, rng.uniform(2e3, 9e3, n),
                   rng.uniform(1, 500, n))
    return pd.DataFrame({'FSC-A': fsc, 'SSC-A': rng.uniform(2e3, 1e5, n),
                         'FL1-A': fl1, 'FL2-A': fl2})


def _truth(df):
    cells = (df['FSC-A'] >= 1000) & (df['SSC-A'] >= 1000)
    a = cells & (df['FL1-A'] >= 1000)
    b = cells & (df['FL2-A'] >= 1000)
    return {'A and B': int((a & b).sum()), 'A or B': int((a | b).sum()),
            'not A': int((cells & ~a).sum()),
            'big': int((a & b & (df['FSC-A'] >= 50000)).sum())}


def _write(tmp_path, gates=_GATES):
    w = fp.WspWriter(cytometer='bool-test')
    w.add_sample('synthetic_1', fcs_path='',
                 channels=['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'],
                 gates=[dict(g) for g in gates])
    out = tmp_path / 'bool.wsp'
    w.write(str(out))
    return out


def _counts(gates, df):
    by_id = {g['id']: g for g in gates}
    rows = population_stats('synthetic_1', df, by_id, list(by_id), {}, [],
                            {'Count'}, ())
    return {(by_id[r['__gid__']].get('name')
             or by_id[r['__gid__']].get('label')): r for r in rows}


def test_writer_emits_flowjo_boolean_nodes(tmp_path):
    root = ET.parse(_write(tmp_path)).getroot()
    tags = {el.get('name'): el.tag for el in root.iter()
            if el.tag in ('AndNode', 'OrNode', 'NotNode')}
    assert tags == {'A and B': 'AndNode', 'A or B': 'OrNode',
                    'not A': 'NotNode'}
    deps = {el.get('name'): [d.get('name') for d in el.iter('Dependent')]
            for el in root.iter() if el.tag in ('AndNode', 'NotNode')}
    assert deps == {'A and B': ['Cells/A', 'Cells/B'], 'not A': ['Cells/A']}


def test_round_trip_counts_match(tmp_path):
    df = _events()
    back, _ = fp.read_template_gates(str(_write(tmp_path)))
    got = _counts(back, df)
    truth = _truth(df)
    for name, n in truth.items():
        assert got[name]['Count'] == n, name
        assert not got[name].get('__unknown__'), name
    assert truth['A and B'] > 0 and truth['big'] > 0


def test_flowjo_order_operand_after_the_boolean(tmp_path):
    # FlowJo may list the boolean before the populations it names.
    path = _write(tmp_path)
    tree = ET.parse(path)
    for sub in tree.getroot().iter('Subpopulations'):
        kids = list(sub)
        if any(k.tag == 'AndNode' for k in kids):
            for k in kids:
                sub.remove(k)
            kids.sort(key=lambda k: k.tag != 'AndNode')
            sub.extend(kids)
    tree.write(path, encoding='utf-8', xml_declaration=True)
    df = _events()
    back, _ = fp.read_template_gates(str(path))
    assert _counts(back, df)['A and B']['Count'] == _truth(df)['A and B']


def test_operand_missing_from_the_file_reads_na(tmp_path):
    path = _write(tmp_path)
    tree = ET.parse(path)
    for dep in tree.getroot().iter('Dependent'):
        if dep.get('name') == 'Cells/B':
            dep.set('name', 'Cells/Nowhere')
    tree.write(path, encoding='utf-8', xml_declaration=True)
    back, _ = fp.read_template_gates(str(path))
    got = _counts(back, _events())
    assert math.isnan(got['A and B']['Count'])
    assert got['A and B']['__unknown__']
    assert math.isnan(got['big']['Count'])          # gated inside it
    assert got['not A']['Count'] == _truth(_events())['not A']


@pytest.mark.parametrize('bad', [
    {'id': 'x', 'kind': 'boolean', 'op': 'not', 'operands': ['a', 'b'],
     'parent_id': 'p', 'name': 'not A nor B'},
    {'id': 'x', 'kind': 'boolean', 'op': 'and', 'operands': ['a', 'c'],
     'parent_id': 'p', 'name': 'A and cluster'},
], ids=['multi-operand-not', 'operand-has-no-geometry'])
def test_booleans_flowjo_cannot_express_are_left_out(bad):
    gates = {g['id']: g for g in _GATES}
    gates['c'] = {'id': 'c', 'kind': 'cluster', 'channel': 'cluster',
                  'cluster_id': 1, 'parent_id': 'p'}
    gates['x'] = bad
    out = fp.wsp_unexportable(gates)
    assert 'x' in out and 'c' in out
    assert not out & {'p', 'a', 'b', 'and', 'or', 'not', 'kid'}
