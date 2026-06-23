"""WspWriter <-> WspReader round-trip.

The synthetic-data variant always runs and is the CI safety net: build
a small gate set, write it, read it back, verify nothing was lost.

The real-WSP variant runs only when OPENFLO_TEST_WSP points at an
actual FlowJo workspace — that one catches subtler regressions like
namespace handling on real v10 files.
"""

import numpy as np
import pytest

import openflo.pipeline as fp


def _gates_equivalent(a, b, tol=1e-6):
    if a.get('kind') != b.get('kind'):
        return False
    k = a.get('kind')
    if k == 'threshold':
        return (a.get('channel') == b.get('channel')
                and abs(float(a['value']) - float(b['value'])) < tol)
    if k == 'interval':
        return (a.get('channel') == b.get('channel')
                and abs(float(a['lo']) - float(b['lo'])) < tol
                and abs(float(a['hi']) - float(b['hi'])) < tol)
    if k == 'rect':
        return (a.get('x_channel') == b.get('x_channel')
                and a.get('y_channel') == b.get('y_channel')
                and all(abs(float(a[ax]) - float(b[ax])) < tol
                        for ax in ('x0', 'x1', 'y0', 'y1')))
    if k == 'polygon':
        if a.get('x_channel') != b.get('x_channel'): return False
        if a.get('y_channel') != b.get('y_channel'): return False
        va = a.get('vertices') or []
        vb = b.get('vertices') or []
        if len(va) != len(vb): return False
        for (ax, ay), (bx, by) in zip(va, vb, strict=True):
            if abs(float(ax) - float(bx)) > tol: return False
            if abs(float(ay) - float(by)) > tol: return False
        return True
    return False


def test_wsp_synthetic_round_trip(tmp_path):
    """Build a tiny gate set, write a .wsp, read it back, verify it survived."""
    gates = [
        {'kind': 'threshold', 'channel': 'BV421-A', 'value': 1000.0,
         'id': 'g1', 'parent_id': None, 'name': 'CD11b+'},
        {'kind': 'rect',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 50_000, 'x1': 250_000, 'y0': 30_000, 'y1': 200_000,
         'id': 'g2', 'parent_id': None, 'name': 'cells'},
    ]

    w = fp.WspWriter(cytometer='SyntheticTest')
    w.add_sample('synthetic', fcs_path='', channels=['FSC-A', 'SSC-A', 'BV421-A'],
                 gates=gates)

    chans = ['BV421-A', 'APC-A', 'PE-Cy7-A']
    mtx = np.array([[1.0, 0.05, 0.0],
                    [0.0, 1.0, 0.01],
                    [0.0, 0.02, 1.0]])
    w.set_compensation(chans, mtx)

    out = tmp_path / 'synthetic.wsp'
    w.write(str(out))
    assert out.is_file() and out.stat().st_size > 0

    # Read back
    back, _ = fp.read_template_gates(str(out))
    assert len(back) == len(gates)

    # gate-by-gate (positional; ids may have been re-emitted)
    for orig, got in zip(gates, back, strict=True):
        assert _gates_equivalent(orig, got), (
            f"gate mismatch: orig={orig} got={got}")

    # compensation survived
    r = fp.WspReader(str(out))
    m = r.get_matrix()
    assert m is not None
    assert list(m['channels']) == chans
    np.testing.assert_allclose(m['matrix'], mtx)


def test_wsp_parse_error_on_garbage(tmp_path):
    """Bogus XML must raise WspParseError, not silently parse zero gates."""
    bad = tmp_path / 'garbage.wsp'
    bad.write_text("this is definitely not xml <unclosed>")
    with pytest.raises(fp.WspParseError):
        fp.WspReader(str(bad))


def test_wsp_missing_file_raises(tmp_path):
    with pytest.raises(fp.WspParseError):
        fp.WspReader(str(tmp_path / 'nope.wsp'))


# ── Compensation round-trip ──────────────────────────────────────────────────
# Regression: the editor's "Export to FlowJo .wsp" used to skip
# `set_compensation`, so reopening the workspace lost the spillover.
# This test mirrors what the GUI / CLI do now — set_compensation +
# add_sample + write, then read_compensation_matrix(path) on the result.

def test_compensation_round_trips_through_export(tmp_path):
    chans = ['BV421-A', 'APC-A', 'PE-Cy7-A']
    mtx = np.array([
        [1.00, 0.05, 0.00],
        [0.00, 1.00, 0.01],
        [0.00, 0.02, 1.00],
    ])
    w = fp.WspWriter(cytometer='OpenFlo-roundtrip')
    w.set_compensation(chans, mtx)
    w.add_sample('sample-A', fcs_path='', channels=chans, gates=[])
    out = tmp_path / 'with_comp.wsp'
    w.write(str(out))

    # The high-level reader used by FlowSample.compensate_from_wsp.
    chans_back, mat_back = fp.read_compensation_matrix(str(out))
    assert chans_back == chans
    assert mat_back is not None
    np.testing.assert_allclose(mat_back, mtx, atol=1e-9)


# ── Per-sample extract via WspReader.extract_gates(sample_node=...) ─────────
# The kwarg was added during the gate-editor WSP-ingest work so each
# sample's <SampleNode> subtree can be walked independently. These tests
# build a multi-sample workspace synthetically, then verify the
# per-sample walk returns the right subset and the default walk still
# flattens across all samples.

def _sample_nodes(reader):
    """Type-narrowed helper — every test calls reader.root.iter() and
    pyright otherwise complains because root is Optional. Wrapping
    once here keeps the test bodies clean."""
    assert reader.root is not None
    return list(reader.root.iter('SampleNode'))


def _build_multisample_wsp(tmp_path):
    """Return (path, expected_per_sample) where expected_per_sample is
    a dict {sample_name: set of (kind, distinguishing_value)} that
    uniquely identifies each gate in that sample."""
    w = fp.WspWriter(cytometer='multisample-test')
    sample_a_gates = [
        {'kind': 'threshold', 'channel': 'BV421-A', 'value': 1000.0,
         'id': 'a1', 'parent_id': None, 'name': 'A-CD11b+'},
        {'kind': 'rect',
         'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'x0': 50_000, 'x1': 250_000, 'y0': 30_000, 'y1': 200_000,
         'id': 'a2', 'parent_id': None, 'name': 'A-cells'},
    ]
    sample_b_gates = [
        {'kind': 'polygon',
         'x_channel': 'APC-A', 'y_channel': 'PE-Cy7-A',
         'vertices': [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
         'id': 'b1', 'parent_id': None, 'name': 'B-poly'},
    ]
    sample_c_gates = [
        {'kind': 'interval', 'channel': 'BV421-A',
         'lo': 100.0, 'hi': 500.0,
         'id': 'c1', 'parent_id': None, 'name': 'C-interval'},
        {'kind': 'threshold', 'channel': 'APC-A', 'value': 50.0,
         'id': 'c2', 'parent_id': 'c1', 'name': 'C-child'},
    ]
    w.add_sample('sample-A', fcs_path='', channels=['FSC-A', 'SSC-A', 'BV421-A'],
                 gates=sample_a_gates)
    w.add_sample('sample-B', fcs_path='', channels=['APC-A', 'PE-Cy7-A'],
                 gates=sample_b_gates)
    w.add_sample('sample-C', fcs_path='', channels=['BV421-A', 'APC-A'],
                 gates=sample_c_gates)
    out = tmp_path / 'multi.wsp'
    w.write(str(out))

    # Identify gates by (kind, distinguishing-value) tuples so we don't
    # have to compare _import_ids (which are reader-internal).
    expected = {
        'sample-A': {('threshold', 'BV421-A', 1000.0),
                     ('rect', 'FSC-A', 'SSC-A')},
        'sample-B': {('polygon', 'APC-A', 'PE-Cy7-A')},
        'sample-C': {('interval', 'BV421-A', 100.0, 500.0),
                     ('threshold', 'APC-A', 50.0)},
    }
    return str(out), expected


def _gate_signature(g):
    """Stable identifier-tuple for a gate dict — matches the values in
    the expected sets above."""
    k = g.get('kind')
    if k == 'threshold':
        return (k, g['channel'], float(g['value']))
    if k == 'interval':
        return (k, g['channel'], float(g['lo']), float(g['hi']))
    if k == 'rect':
        return (k, g['x_channel'], g['y_channel'])
    if k == 'polygon':
        return (k, g['x_channel'], g['y_channel'])
    return (k,)


def test_extract_gates_per_sample_returns_only_that_samples_gates(tmp_path):
    path, expected = _build_multisample_wsp(tmp_path)
    reader = fp.WspReader(path)

    by_name = {}
    for sn in _sample_nodes(reader):
        name = sn.get('name')
        if name in expected:
            by_name[name] = reader.extract_gates(sample_node=sn)

    assert set(by_name) == set(expected), (
        f"missing samples: expected {set(expected)}, got {set(by_name)}")

    for sample_name, gates in by_name.items():
        sigs = {_gate_signature(g) for g in gates}
        assert sigs == expected[sample_name], (
            f"{sample_name}: expected {expected[sample_name]}, got {sigs}")


def test_extract_gates_per_sample_preserves_parent_chain(tmp_path):
    """Sample-C has a threshold child of an interval parent — that
    relationship must survive the per-sample walk."""
    path, _ = _build_multisample_wsp(tmp_path)
    reader = fp.WspReader(path)

    sample_c = next(sn for sn in _sample_nodes(reader)
                    if sn.get('name') == 'sample-C')
    gates = reader.extract_gates(sample_node=sample_c)

    # Find the threshold-on-APC-A and verify its parent_id resolves to
    # an interval-on-BV421-A in this same returned list.
    sig_by_id = {g['_import_id']: _gate_signature(g) for g in gates}
    children = [g for g in gates if g.get('parent_id') is not None]
    assert children, "per-sample walk lost the parent_id chain"
    for child in children:
        assert child['parent_id'] in sig_by_id, (
            f"parent_id {child['parent_id']!r} doesn't resolve within "
            f"the per-sample subset")
        # Specifically: the APC-A threshold's parent should be the
        # BV421-A interval.
        if _gate_signature(child) == ('threshold', 'APC-A', 50.0):
            assert sig_by_id[child['parent_id']] == (
                'interval', 'BV421-A', 100.0, 500.0)


def test_extract_gates_default_still_flattens(tmp_path):
    """Without the kwarg, the existing behaviour (walk every SampleNode)
    must be unchanged — read_template_gates relies on this."""
    path, expected = _build_multisample_wsp(tmp_path)
    reader = fp.WspReader(path)
    flat = reader.extract_gates()

    # Total gate count = sum of per-sample counts.
    total_expected = sum(len(s) for s in expected.values())
    assert len(flat) == total_expected, (
        f"flat extract should return all {total_expected} gates across "
        f"samples; got {len(flat)}")

    # Every signature from every sample is represented.
    flat_sigs = {_gate_signature(g) for g in flat}
    expected_sigs = set().union(*expected.values())
    assert flat_sigs == expected_sigs


def test_extract_gates_empty_sample_node(tmp_path):
    """A SampleNode with no Subpopulations / no Population children
    must return an empty list, not raise."""
    w = fp.WspWriter(cytometer='empty-sample-test')
    w.add_sample('lonely', fcs_path='', channels=['BV421-A'], gates=[])
    out = tmp_path / 'lonely.wsp'
    w.write(str(out))
    reader = fp.WspReader(str(out))
    sn = _sample_nodes(reader)[0]
    gates = reader.extract_gates(sample_node=sn)
    assert gates == []


def test_flowsample_persists_comp_matrix_after_apply(synthetic_fcs):
    """`_apply_comp` must stash the matrix on the FlowSample so the
    GUI / CLI export paths can register it on the writer. Before this
    fix the matrix was applied to .data and immediately discarded."""
    s = fp.FlowSample(synthetic_fcs)
    # Synthetic FCS has no $SPILL; apply a manual matrix instead.
    chans = ['BV421-A', 'APC-A', 'PE-Cy7-A']
    mtx = np.array([
        [1.00, 0.05, 0.00],
        [0.00, 1.00, 0.01],
        [0.00, 0.02, 1.00],
    ])
    s.manual_compensate(mtx, chans)
    assert s.comp_channels == chans, (
        "comp_channels not stored after manual_compensate — "
        "GUI export would skip spillover")
    assert s.comp_matrix is not None
    np.testing.assert_allclose(s.comp_matrix, mtx)


# ── Opt-in real-data round-trip ──────────────────────────────────────────────

def test_real_wsp_round_trip(real_wsp_path, tmp_path):
    """Run only when a real .wsp is available locally — see conftest."""
    orig_gates, _ = fp.read_template_gates(real_wsp_path)
    assert orig_gates, "real WSP should contain gates"

    w = fp.WspWriter(cytometer='LSRFortessa-roundtrip')
    channels = sorted({g.get('channel', '') for g in orig_gates if g.get('channel')}
                      | {g.get('x_channel', '') for g in orig_gates if g.get('x_channel')}
                      | {g.get('y_channel', '') for g in orig_gates if g.get('y_channel')})
    w.add_sample('roundtrip_all_gates', fcs_path='', channels=channels, gates=orig_gates)

    out = tmp_path / 'roundtrip.wsp'
    w.write(str(out))
    back, _ = fp.read_template_gates(str(out))
    assert len(back) == len(orig_gates), (
        f"count mismatch: orig {len(orig_gates)} / round {len(back)}")
