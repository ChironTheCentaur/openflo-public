"""Headless tests for the gating-tree renderer (openflo.gatetree).

matplotlib runs on the Agg backend; nothing is shown."""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

from matplotlib.figure import Figure

from openflo.gatetree import build_tree, gate_tree_figure


def _gates():
    """Root -> child chain (g1 -> g2) plus a sibling leaf (g3) under g1."""
    return {
        'g1': {'kind': 'rect', 'channel': 'FSC-A', 'parent_id': None,
               'color': '#1f77b4', 'enabled': True, 'id': 'g1'},
        'g2': {'kind': 'category', 'channel': 'CD3', 'parent_id': 'g1',
               'color': '#ff7f0e', 'enabled': True, 'id': 'g2',
               'name': 'CD3+ T cells'},
        'g3': {'kind': 'category', 'channel': 'CD19', 'parent_id': 'g1',
               'color': '#2ca02c', 'enabled': True, 'id': 'g3'},
    }


def test_build_tree_counts():
    roots = build_tree(_gates())
    assert len(roots) == 1
    root = roots[0]
    assert root['id'] == 'g1'
    assert len(root['children']) == 2
    child_ids = {c['id'] for c in root['children']}
    assert child_ids == {'g2', 'g3'}


def test_build_tree_labels():
    roots = build_tree(_gates())
    root = roots[0]
    # No name -> "<kind> on <channel>".
    assert root['label'] == 'rect on FSC-A'
    labels = {c['id']: c['label'] for c in root['children']}
    # Explicit name wins.
    assert labels['g2'] == 'CD3+ T cells'
    assert labels['g3'] == 'category on CD19'


def test_build_tree_empty():
    assert build_tree(None) == []
    assert build_tree({}) == []


def test_build_tree_orphan_is_root():
    gates = {
        'a': {'kind': 'rect', 'channel': 'X', 'parent_id': 'missing'},
    }
    roots = build_tree(gates)
    assert len(roots) == 1
    assert roots[0]['id'] == 'a'


def test_build_tree_cycle_defensive():
    # a -> b -> a cycle: every gate still appears exactly once.
    gates = {
        'a': {'kind': 'rect', 'channel': 'X', 'parent_id': 'b'},
        'b': {'kind': 'rect', 'channel': 'Y', 'parent_id': 'a'},
    }
    roots = build_tree(gates)
    seen = []

    def collect(node):
        seen.append(node['id'])
        for c in node['children']:
            collect(c)

    for r in roots:
        collect(r)
    assert sorted(seen) == ['a', 'b']
    assert len(roots) >= 1


def test_gate_tree_figure_returns_figure():
    fig = gate_tree_figure(_gates(), sample_name='Sample A')
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_gate_tree_figure_empty():
    fig = gate_tree_figure({})
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
