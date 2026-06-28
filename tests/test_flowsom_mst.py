"""Tests for the FlowSOM MST helpers (pipeline.flowsom_mst / flowsom_layout)."""
from __future__ import annotations

import numpy as np

from openflo.pipeline import flowsom_layout, flowsom_mst


def test_mst_is_a_spanning_tree():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 5))
    edges, dist = flowsom_mst(W)
    # A tree over n nodes has exactly n-1 edges.
    assert len(edges) == 11
    assert dist.shape == (12, 12)
    # Connected: every node appears in some edge.
    seen = set()
    for a, b in edges:
        seen.add(a)
        seen.add(b)
    assert seen == set(range(12))


def test_mst_connects_clusters_via_bridge():
    # Two tight clusters far apart → the MST must contain exactly one bridging
    # edge between them (the rest are intra-cluster).
    a = np.zeros((4, 2)) + np.array([0.0, 0.0])
    a += np.linspace(0, 0.1, 4)[:, None]
    b = np.zeros((4, 2)) + np.array([100.0, 0.0])
    b += np.linspace(0, 0.1, 4)[:, None]
    W = np.vstack([a, b])
    edges, _ = flowsom_mst(W)
    bridges = [(i, j) for i, j in edges if (i < 4) != (j < 4)]
    assert len(bridges) == 1


def test_mst_too_few_nodes():
    edges, dist = flowsom_mst(np.zeros((1, 3)))
    assert edges == []
    assert dist.shape == (1, 1)


def test_layout_shape_and_fallback():
    edges, _ = flowsom_mst(np.random.default_rng(1).normal(size=(8, 4)))
    pos = flowsom_layout(8, edges)
    assert pos.shape == (8, 2)
    assert np.isfinite(pos).all()
    # Empty graph → empty layout.
    assert flowsom_layout(0, []).shape == (0, 2)
