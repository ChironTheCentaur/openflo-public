"""A .wsp used as a gating template gives ONE sample's gate tree.

FlowJo stores a gate tree per <SampleNode>, normally the same tree on every
sample. read_template_gates walked every SampleNode and returned the trees
concatenated: a 2-sample workspace of 5 gates came back as 10, and applying
it as a template gave each target sample every population twice (two 'Cells',
two 'FL1 bright', ...), and a 20-sample workspace twenty times over.
"""
from __future__ import annotations

import logging

import openflo.pipeline as fp

CH = ['FSC-A', 'SSC-A', 'FL1-A']


def _tree(lo):
    return [
        {'id': 'cells', 'parent_id': None, 'kind': 'rect', 'label': 'Cells',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 3e4, 'x1': 1.9e5, 'y0': 2e4, 'y1': 1.4e5},
        {'id': 'neg', 'parent_id': 'cells', 'kind': 'interval',
         'label': 'FL1 neg', 'channel': 'FL1-A', 'lo': -800.0, 'hi': lo},
        {'id': 'pos', 'parent_id': 'cells', 'kind': 'threshold',
         'label': 'FL1 pos', 'channel': 'FL1-A', 'value': lo},
    ]


def _write(tmp_path, trees):
    w = fp.WspWriter()
    for i, t in enumerate(trees):
        w.add_sample(f's{i}', '', CH, t)
    p = tmp_path / 't.wsp'
    w.write(str(p))
    return str(p)


def test_multi_sample_wsp_template_is_one_tree(tmp_path, caplog):
    """3 samples x 3 gates: 3 gates come back (were 9), the first sample's,
    with its parent links, and the log says which sample was used."""
    path = _write(tmp_path, [_tree(600.0), _tree(700.0), _tree(800.0)])
    with caplog.at_level(logging.INFO, logger='openflo.pipeline'):
        gates, labels = fp.read_template_gates(path)
    assert labels is None
    assert [g['label'] for g in gates] == ['Cells', 'FL1 neg', 'FL1 pos']
    by_id = {g['id']: g for g in gates}
    assert [by_id[g['parent_id']]['label'] if g['parent_id'] else None
            for g in gates] == [None, 'Cells', 'Cells']
    assert gates[2]['value'] == 600.0               # sample s0's, not s1's
    assert 's0' in caplog.text and '3' in caplog.text


def test_single_sample_wsp_template_unchanged(tmp_path):
    path = _write(tmp_path, [_tree(600.0)])
    gates, _ = fp.read_template_gates(path)
    assert len(gates) == 3


def test_first_sample_without_gates_is_skipped(tmp_path):
    """A workspace whose first sample has no gates yet: the template is the
    first tree there is, not an empty one."""
    path = _write(tmp_path, [[], _tree(700.0)])
    gates, _ = fp.read_template_gates(path)
    assert [g['label'] for g in gates] == ['Cells', 'FL1 neg', 'FL1 pos']
    assert gates[2]['value'] == 700.0
