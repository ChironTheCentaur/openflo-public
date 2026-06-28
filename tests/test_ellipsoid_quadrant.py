"""FlowJo ellipsoid + quadrant gate support.

Covers the Phase-1 additions:
  - gate_to_mask evaluates 'ellipsoid' (squared Mahalanobis test)
  - WspWriter emits EllipsoidGate / QuadrantGate
  - WspReader parses them back (ellipsoid → 'ellipsoid'; quadrant →
    4 linked 'rect' gates sharing a quad_set)
  - Full write → read round-trip is self-consistent

IMPORTANT — these tests assert OpenFlo's *own* round-trip
(WspWriter ↔ WspReader). FlowJo v10's exact EllipsoidGate /
QuadrantGate serialization (foci/edge hints, divider value as attr vs
child element) is NOT validated here — that needs a real FlowJo .wsp
containing those gate kinds. See the note in WspReader.extract_gates.
"""
import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp

# ── gate_to_mask: ellipsoid ──────────────────────────────────────────────────

def test_ellipsoid_mask_unit_circle():
    """An axis-aligned unit-variance ellipsoid with distance_sq=1 is the
    unit circle centred at the mean: points within radius 1 are inside."""
    gate = {
        'kind': 'ellipsoid',
        'x_channel': 'X', 'y_channel': 'Y',
        'mean': [0.0, 0.0],
        'cov': [[1.0, 0.0], [0.0, 1.0]],
        'distance_sq': 1.0,
    }
    df = pd.DataFrame({
        'X': [0.0, 0.5, 0.9, 1.1, 0.0, 2.0],
        'Y': [0.0, 0.0, 0.0, 0.0, 1.1, 0.0],
    })
    mask = fp.gate_to_mask(gate, df)
    # inside: (0,0),(0.5,0),(0.9,0); outside: (1.1,0),(0,1.1),(2,0)
    assert list(mask) == [True, True, True, False, False, False]


def test_ellipsoid_mask_respects_covariance_orientation():
    """A correlated covariance tilts the ellipse — a point off the major
    axis that would be inside a circle can fall outside."""
    gate = {
        'kind': 'ellipsoid',
        'x_channel': 'X', 'y_channel': 'Y',
        'mean': [0.0, 0.0],
        # Wide along x (var 4), narrow along y (var 0.25).
        'cov': [[4.0, 0.0], [0.0, 0.25]],
        'distance_sq': 1.0,
    }
    df = pd.DataFrame({
        'X': [1.9, 0.0, 0.0],
        'Y': [0.0, 0.49, 0.6],
    })
    mask = fp.gate_to_mask(gate, df)
    # (1.9,0): 1.9²/4 = 0.9025 ≤ 1 → inside
    # (0,0.49): 0.49²/0.25 = 0.96 ≤ 1 → inside
    # (0,0.6): 0.6²/0.25 = 1.44 > 1 → outside
    assert list(mask) == [True, True, False]


def test_ellipsoid_mask_singular_cov_is_noop():
    gate = {
        'kind': 'ellipsoid', 'x_channel': 'X', 'y_channel': 'Y',
        'mean': [0.0, 0.0], 'cov': [[0.0, 0.0], [0.0, 0.0]],
        'distance_sq': 1.0,
    }
    df = pd.DataFrame({'X': [0.0, 5.0], 'Y': [0.0, 5.0]})
    mask = fp.gate_to_mask(gate, df)
    assert list(mask) == [True, True]   # all-True no-op


def test_ellipsoid_mask_missing_channel_is_noop():
    gate = {
        'kind': 'ellipsoid', 'x_channel': 'NOPE', 'y_channel': 'Y',
        'mean': [0.0, 0.0], 'cov': [[1.0, 0.0], [0.0, 1.0]],
        'distance_sq': 1.0,
    }
    df = pd.DataFrame({'X': [0.0], 'Y': [0.0]})
    assert list(fp.gate_to_mask(gate, df)) == [True]


# ── gate_to_mask: cluster (#43) ──────────────────────────────────────────────

def test_cluster_mask_selects_one_label():
    """A cluster gate selects exactly the events whose 'cluster' id matches."""
    gate = {'kind': 'cluster', 'channel': 'cluster', 'cluster_id': 2}
    df = pd.DataFrame({'cluster': [0, 1, 2, 2, 3]})
    assert list(fp.gate_to_mask(gate, df)) == [False, False, True, True, False]


def test_cluster_mask_missing_column_is_empty():
    """Unlike geometric gates (all-True no-op), a missing cluster column
    means the population is undefined → selects NOTHING."""
    gate = {'kind': 'cluster', 'channel': 'cluster', 'cluster_id': 0}
    df = pd.DataFrame({'X': [1.0, 2.0, 3.0]})
    assert list(fp.gate_to_mask(gate, df)) == [False, False, False]


def test_cluster_describe_uses_name():
    assert fp.describe_gate(
        {'kind': 'cluster', 'cluster_id': 3, 'name': 'T cells'}) == 'C  T cells'
    assert fp.describe_gate(
        {'kind': 'cluster', 'cluster_id': 3}) == 'C  cluster 3'


# ── gate_to_mask: boolean (AND/OR/NOT) ───────────────────────────────────────

def _bool_gates():
    return {
        'g1': {'kind': 'threshold', 'channel': 'X', 'value': 5,
               'parent_id': None},
        'g2': {'kind': 'threshold', 'channel': 'Y', 'value': 5,
               'parent_id': None},
    }


def _bool_df():
    # rows: (0,0) (10,0) (10,10) (0,10)
    return pd.DataFrame({'X': [0, 10, 10, 0], 'Y': [0, 0, 10, 10]})


def test_boolean_and():
    gates = _bool_gates()
    g = {'kind': 'boolean', 'op': 'and', 'operands': ['g1', 'g2']}
    assert list(fp.gate_to_mask(g, _bool_df(), gates)) == \
        [False, False, True, False]


def test_boolean_or():
    gates = _bool_gates()
    g = {'kind': 'boolean', 'op': 'or', 'operands': ['g1', 'g2']}
    assert list(fp.gate_to_mask(g, _bool_df(), gates)) == \
        [False, True, True, True]


def test_boolean_not():
    gates = _bool_gates()
    g = {'kind': 'boolean', 'op': 'not', 'operands': ['g1']}  # NOT X>5
    assert list(fp.gate_to_mask(g, _bool_df(), gates)) == \
        [True, False, False, True]


def test_boolean_without_gates_dict_is_noop():
    g = {'kind': 'boolean', 'op': 'and', 'operands': ['g1']}
    assert list(fp.gate_to_mask(g, _bool_df())) == [True, True, True, True]


def test_boolean_participates_in_cumulative_chain():
    # A child gate under a boolean parent ANDs with the boolean's mask.
    gates = _bool_gates()
    gates['b'] = {'kind': 'boolean', 'op': 'or', 'operands': ['g1', 'g2'],
                  'parent_id': None}
    gates['c'] = {'kind': 'threshold', 'channel': 'X', 'value': 5,
                  'parent_id': 'b'}    # X>5 within (g1 OR g2)
    mask = fp.cumulative_gate_mask(gates, 'c', _bool_df())
    # OR = rows 1,2,3; AND X>5 (rows 1,2) → rows 1,2
    assert list(mask) == [False, True, True, False]


def test_boolean_cycle_is_guarded():
    # b references itself indirectly; must terminate, not recurse forever.
    gates = {'b': {'kind': 'boolean', 'op': 'and', 'operands': ['b'],
                   'parent_id': None}}
    mask = fp.gate_to_mask(gates['b'], _bool_df(), gates)
    assert len(mask) == 4   # returned something finite


def test_boolean_describe_gate():
    assert fp.describe_gate(
        {'kind': 'boolean', 'name': 'live CD4'}) == 'B  live CD4'
    assert fp.describe_gate(
        {'kind': 'boolean', 'op': 'or', 'operands': ['a', 'b']}) == 'B  OR(2)'


# ── Ellipsoid WSP round-trip ─────────────────────────────────────────────────

def test_ellipsoid_wsp_round_trip(tmp_path):
    gate = {
        'kind': 'ellipsoid', 'x_channel': 'BV421-A', 'y_channel': 'APC-A',
        'mean': [1000.0, 2000.0],
        'cov': [[50000.0, 1200.0], [1200.0, 80000.0]],
        'distance_sq': 4.0,
        'id': 'e1', 'parent_id': None, 'name': 'blast-ellipse',
    }
    w = fp.WspWriter(cytometer='ellipsoid-test')
    w.add_sample('s', fcs_path='', channels=['BV421-A', 'APC-A'], gates=[gate])
    out = tmp_path / 'ell.wsp'
    w.write(str(out))

    back, _ = fp.read_template_gates(str(out))
    ells = [g for g in back if g.get('kind') == 'ellipsoid']
    assert len(ells) == 1, f"expected 1 ellipsoid, got {[g.get('kind') for g in back]}"
    e = ells[0]
    assert e['x_channel'] == 'BV421-A'
    assert e['y_channel'] == 'APC-A'
    np.testing.assert_allclose(e['mean'], gate['mean'], rtol=1e-9)
    np.testing.assert_allclose(e['cov'], gate['cov'], rtol=1e-9)
    assert e['distance_sq'] == pytest.approx(4.0)


# ── Quadrant WSP round-trip ──────────────────────────────────────────────────

def _make_quad_set(xc, yc, xdiv, ydiv):
    """Build the editor's 4-rect quadrant representation."""
    big = 1e12
    qs = 'qs1'
    rects = []
    for i, (label, x0, x1, y0, y1) in enumerate([
            ('Q++', xdiv,  big,  ydiv,  big),
            ('Q+-', xdiv,  big, -big,  ydiv),
            ('Q-+', -big,  xdiv, ydiv,  big),
            ('Q--', -big,  xdiv, -big,  ydiv)]):
        rects.append({
            'kind': 'rect', 'x_channel': xc, 'y_channel': yc,
            'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'label': label,
            'quad_set': qs, 'quad_origin_x': xdiv, 'quad_origin_y': ydiv,
            'id': f'q{i}', 'parent_id': None,
        })
    return rects


def test_quadrant_collapses_to_single_gate_then_expands(tmp_path):
    """4 linked rects → write → one QuadrantGate → read → 4 rects again,
    with the divider position preserved."""
    rects = _make_quad_set('FSC-A', 'SSC-A', xdiv=1000.0, ydiv=2000.0)
    w = fp.WspWriter(cytometer='quad-test')
    w.add_sample('s', fcs_path='', channels=['FSC-A', 'SSC-A'], gates=rects)
    out = tmp_path / 'quad.wsp'
    w.write(str(out))

    # On disk it should be ONE QuadrantGate (count opening tags only —
    # the substring appears in both <...QuadrantGate> and </...QuadrantGate>).
    xml = out.read_text(encoding='utf-8')
    assert xml.count('<gating:QuadrantGate') == 1, (
        "quad_set should collapse to exactly one QuadrantGate")
    assert '<gating:RectangleGate' not in xml, (
        "collapsed quadrant must not leave stray RectangleGates")

    back, _ = fp.read_template_gates(str(out))
    quad_rects = [g for g in back if g.get('quad_set')]
    assert len(quad_rects) == 4, (
        f"QuadrantGate should expand back to 4 rects, got {len(quad_rects)}")
    # All 4 share one quad_set and the same divider origin.
    assert len({g['quad_set'] for g in quad_rects}) == 1
    for g in quad_rects:
        assert g['quad_origin_x'] == pytest.approx(1000.0)
        assert g['quad_origin_y'] == pytest.approx(2000.0)
        assert g['x_channel'] == 'FSC-A'
        assert g['y_channel'] == 'SSC-A'


def test_quadrant_masks_partition_the_plane(tmp_path):
    """The 4 expanded rects should partition events into 4 disjoint,
    exhaustive groups around the divider point."""
    rects = _make_quad_set('FSC-A', 'SSC-A', xdiv=1000.0, ydiv=2000.0)
    w = fp.WspWriter(cytometer='quad-test')
    w.add_sample('s', fcs_path='', channels=['FSC-A', 'SSC-A'], gates=rects)
    out = tmp_path / 'quad.wsp'
    w.write(str(out))
    back, _ = fp.read_template_gates(str(out))

    df = pd.DataFrame({
        'FSC-A': [1500, 1500, 500, 500],   # hi, hi, lo, lo
        'SSC-A': [2500, 1500, 2500, 1500],  # hi, lo, hi, lo
    })
    # Each event should land in exactly one quadrant.
    quad_rects = [g for g in back if g.get('quad_set')]
    hit_counts = np.zeros(len(df), dtype=int)
    for g in quad_rects:
        hit_counts += fp.gate_to_mask(g, df).astype(int)
    assert list(hit_counts) == [1, 1, 1, 1], (
        f"each event should be in exactly one quadrant, got {hit_counts}")
