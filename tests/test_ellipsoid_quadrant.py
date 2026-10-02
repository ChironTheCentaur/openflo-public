"""FlowJo ellipsoid + quadrant gate support.

Covers the Phase-1 additions:
  - gate_to_mask evaluates 'ellipsoid' (squared Mahalanobis test)
  - WspWriter emits an ellipse as a 64-vertex PolygonGate (FlowJo 10.10.2
    silently drops a sample carrying a Gating-ML EllipsoidGate), and an
    editor quadrant as 4 RectangleGates (the form FlowJo itself writes)
  - WspReader parses an EllipsoidGate (mean/covariance form) and a
    QuadrantGate (→ 4 'rect' gates)

IMPORTANT — FlowJo v10's own EllipsoidGate serialization (foci/edge in
display space) is NOT validated here. See the note in
WspReader.extract_gates.
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


def test_boolean_without_gates_dict_admits_nothing():
    """A boolean gate references other gates by id. With no dictionary to
    resolve them it cannot know what it is combining, and it used to no-op to
    all-True — which for a NOT gate handed back the ENTIRE sample as the
    complement population (measured: 20,000 events where 10,004 was correct).

    It now fails CLOSED, like the 'cluster'/'category' branches beside it and
    the fail-closed handler in apply_region_gates: an empty population is
    visible to the user, a superset reads as success. Both real callers pass
    the dictionary, so this branch means "could not be evaluated"."""
    g = {'kind': 'boolean', 'op': 'and', 'operands': ['g1']}
    assert list(fp.gate_to_mask(g, _bool_df())) == [False, False, False, False]


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

_ELLIPSE = {
    'kind': 'ellipsoid', 'x_channel': 'BV421-A', 'y_channel': 'APC-A',
    'mean': [1000.0, 2000.0],
    'cov': [[50000.0, 1200.0], [1200.0, 80000.0]],
    'distance_sq': 4.0,
    'id': 'e1', 'parent_id': None, 'name': 'blast-ellipse',
}


def test_ellipsoid_exports_as_a_polygon_flowjo_keeps(tmp_path):
    # FlowJo 10.10.2 removed every sample that carried a Gating-ML
    # EllipsoidGate from the saved workspace, without a notification
    # (measured). The export must never contain one.
    w = fp.WspWriter(cytometer='ellipsoid-test')
    w.add_sample('s', fcs_path='', channels=['BV421-A', 'APC-A'], gates=[dict(_ELLIPSE)])
    out = tmp_path / 'ell.wsp'
    w.write(str(out))
    xml = out.read_text(encoding='utf-8')
    assert '<gating:EllipsoidGate' not in xml
    assert xml.count('<gating:PolygonGate') == 1
    assert any('ellipse' in m and 'polygon' in m for m in w.warnings), w.warnings

    back, _ = fp.read_template_gates(str(out))
    (poly,) = back
    assert poly['kind'] == 'polygon' and (poly.get('label') or poly.get('name')) == 'blast-ellipse'
    assert (poly['x_channel'], poly['y_channel']) == ('BV421-A', 'APC-A')
    assert len(poly['vertices']) == 64
    # The area-preserving 64-gon selects (almost) the ellipse's events:
    # the boundaries differ by < 0.1% of the radius.
    rng = np.random.default_rng(3)
    L = np.linalg.cholesky(np.asarray(_ELLIPSE['cov']))
    z = rng.normal(size=(2, 200_000)) * 1.2
    pts = np.asarray(_ELLIPSE['mean'])[:, None] + L @ z
    df = pd.DataFrame({'BV421-A': pts[0], 'APC-A': pts[1]})
    want = fp.gate_to_mask(dict(_ELLIPSE), df)
    got = fp.gate_to_mask(poly, df)
    assert 0.3 * len(df) < want.sum() < 0.9 * len(df)
    assert (want != got).sum() <= 0.002 * want.sum(), (int((want != got).sum()), int(want.sum()))


def test_singular_ellipsoid_exports_as_all_pass_rectangle(tmp_path):
    # OpenFlo passes every event through a singular ellipse (see
    # test_ellipsoid_mask_singular_cov_is_noop); the export keeps that.
    g = dict(_ELLIPSE, cov=[[1.0, 1.0], [1.0, 1.0]])
    w = fp.WspWriter(cytometer='ellipsoid-test')
    w.add_sample('s', fcs_path='', channels=['BV421-A', 'APC-A'], gates=[g])
    out = tmp_path / 'ell.wsp'
    w.write(str(out))
    assert '<gating:EllipsoidGate' not in out.read_text(encoding='utf-8')
    (rect,) = fp.read_template_gates(str(out))[0]
    assert rect['kind'] == 'rect'
    assert (rect['x0'], rect['x1'], rect['y0'], rect['y1']) == (-1e6, 1e6, -1e6, 1e6)
    assert any('singular' in m for m in w.warnings), w.warnings


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


def test_quadrant_exports_as_four_rectangle_gates(tmp_path):
    """4 linked rects → write → 4 RectangleGates (FlowJo's own quadrant
    form; FlowJo never writes a QuadrantGate) → read → the same 4 rects."""
    rects = _make_quad_set('FSC-A', 'SSC-A', xdiv=1000.0, ydiv=2000.0)
    w = fp.WspWriter(cytometer='quad-test')
    w.add_sample('s', fcs_path='', channels=['FSC-A', 'SSC-A'], gates=rects)
    out = tmp_path / 'quad.wsp'
    w.write(str(out))

    xml = out.read_text(encoding='utf-8')
    assert '<gating:QuadrantGate' not in xml
    assert xml.count('<gating:RectangleGate') == 4

    back, _ = fp.read_template_gates(str(out))
    got = sorted((g['label'], g['x0'], g['x1'], g['y0'], g['y1'])
                 for g in back if g.get('kind') == 'rect')
    want = sorted((g['label'], g['x0'], g['x1'], g['y0'], g['y1'])
                  for g in rects)
    assert got == want
    for g in back:
        assert g['x_channel'] == 'FSC-A'
        assert g['y_channel'] == 'SSC-A'


def test_quadrant_masks_partition_the_plane(tmp_path):
    """The 4 re-imported rects should partition events into 4 disjoint,
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
    quad_rects = [g for g in back if g.get('kind') == 'rect']
    assert len(quad_rects) == 4
    hit_counts = np.zeros(len(df), dtype=int)
    for g in quad_rects:
        hit_counts += fp.gate_to_mask(g, df).astype(int)
    assert list(hit_counts) == [1, 1, 1, 1], (
        f"each event should be in exactly one quadrant, got {hit_counts}")


def test_indefinite_ellipsoid_is_flagged_not_called_singular(tmp_path):
    # Invertible but not positive definite: OpenFlo's quadratic form then
    # describes no ellipse, and does not pass every event either. The export
    # cannot reproduce it, and must say so rather than claim "singular".
    g = dict(_ELLIPSE, cov=[[1.0, 2.0], [2.0, 1.0]])
    w = fp.WspWriter(cytometer='ellipsoid-test')
    w.add_sample('s', fcs_path='', channels=['BV421-A', 'APC-A'], gates=[g])
    w.write(str(tmp_path / 'ell.wsp'))
    (msg,) = [m for m in w.warnings if 'blast-ellipse' in m]
    assert 'not positive definite' in msg and 'FlowJo will differ' in msg
    assert 'singular' not in msg


def test_correlated_nested_ellipse_keeps_orientation_and_parent(tmp_path):
    # A strongly tilted ellipse (correlation 0.9) under a parent gate: the
    # polygon must follow the tilt (a transposed Cholesky factor would not)
    # and stay under its parent, or FlowJo stops applying the parent.
    parent = {'id': 'p', 'parent_id': None, 'kind': 'interval', 'channel': 'BV421-A',
              'lo': -1e5, 'hi': 1e5, 'name': 'Parent'}
    cov = [[1e6, 9e5], [9e5, 1e6]]
    ell = dict(_ELLIPSE, cov=cov, id='e1', parent_id='p')
    w = fp.WspWriter(cytometer='ellipsoid-test')
    w.add_sample('s', fcs_path='', channels=['BV421-A', 'APC-A'], gates=[parent, ell])
    out = tmp_path / 'ell.wsp'
    w.write(str(out))
    back, _ = fp.read_template_gates(str(out))
    (p_back,) = [g for g in back if g['kind'] == 'interval']
    (poly,) = [g for g in back if g['kind'] == 'polygon']
    assert poly['parent_id'] == p_back['id']
    rng = np.random.default_rng(5)
    L = np.linalg.cholesky(np.asarray(cov))
    pts = np.asarray(_ELLIPSE['mean'])[:, None] + L @ (rng.normal(size=(2, 200_000)) * 1.2)
    df = pd.DataFrame({'BV421-A': pts[0], 'APC-A': pts[1]})
    want = fp.gate_to_mask(dict(ell), df)
    got = fp.gate_to_mask(poly, df)
    assert (want != got).sum() <= 0.002 * want.sum(), (int((want != got).sum()), int(want.sum()))


def test_ellipse_polygon_has_the_ellipse_area():
    # Area-preserving by construction: an inscribed 64-gon would hold 0.16%
    # less area, a systematic undercount at the edge. Shoelace area must equal
    # pi * sqrt(det cov) * distance_sq.
    cov = np.array([[1e6, 9e5], [9e5, 1e6]])
    verts = np.asarray(fp._ellipse_polygon([5.0, -3.0], cov, 4.0))
    x, y = verts[:, 0], verts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert area == pytest.approx(np.pi * np.sqrt(np.linalg.det(cov)) * 4.0, rel=1e-9)
