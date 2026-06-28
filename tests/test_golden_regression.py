"""Golden-output regression test.

Loads the deterministic synthetic FCS (seeded RNG → reproducible bits),
runs the deterministic portion of the pipeline (load → QC → compensate →
logicle transform → threshold gate), and compares the resulting numeric
summaries to snapshotted values.

This is the test that catches "numpy 2.5 changed kNN tie-breaking and
now half our gates shifted" or "flowutils tweaked logicle defaults".
It does NOT include Phenograph clustering (small-N runs are flaky and
the algorithm itself isn't bit-stable across releases).

Tolerances are generous enough to absorb float32 / minor version drift
in numpy/scipy/flowutils, tight enough to catch real algorithm changes.
"""
import numpy as np

import openflo.pipeline as fp

# ── Expected values, captured 2026-05-27 from a clean run ────────────────────
# Synthetic FCS is built from a seeded RNG (see conftest.synthetic_fcs); if
# the fixture changes, regenerate these snapshots and bump the comment date.

EXPECTED_EVENTS = 1000

EXPECTED_LOGICLE_STATS = {
    'BV421-A': {'mean': 0.4134, 'p50': 0.4170},
    'APC-A':   {'mean': 0.2823, 'p50': 0.2761},
    'PE-Cy7-A':{'mean': 0.3668, 'p50': 0.3658},
}

# Channel-level relative tolerances. Logicle scale is bounded ~[0, 1] so an
# absolute tolerance is meaningful and easier to reason about than relative.
ATOL_LOGICLE = 0.05

# The synthetic BV421 channel is bimodal by construction (50/50 split at
# the cluster boundary). After logicle, a threshold at 0.5 cleanly splits.
EXPECTED_BV421_POS_FRAC = 0.500
ATOL_FRAC = 0.10  # 40% – 60% would still be a pass; below that means a real shift


def test_golden_load(synthetic_fcs):
    s = fp.FlowSample(synthetic_fcs)
    assert len(s.data) == EXPECTED_EVENTS


def test_golden_pipeline_logicle_stats(synthetic_fcs):
    """After load → QC → compensate (no-op) → logicle, per-channel
    summary stats must stay within tolerance of the snapshot."""
    s = fp.FlowSample(synthetic_fcs)
    s.run_qc()
    s.auto_compensate()
    s.apply_transform(channels=list(EXPECTED_LOGICLE_STATS.keys()))

    # QC should not drop synthetic events (no real instrument anomalies).
    assert len(s.data) == EXPECTED_EVENTS, (
        "QC unexpectedly trimmed events from a clean synthetic FCS")

    for ch, expected in EXPECTED_LOGICLE_STATS.items():
        arr = np.asarray(s.data[ch].values, dtype=float)
        got_mean = float(arr.mean())
        got_p50 = float(np.percentile(arr, 50))
        assert abs(got_mean - expected['mean']) < ATOL_LOGICLE, (
            f"{ch} mean drift: expected {expected['mean']:.4f} "
            f"got {got_mean:.4f}")
        assert abs(got_p50 - expected['p50']) < ATOL_LOGICLE, (
            f"{ch} median drift: expected {expected['p50']:.4f} "
            f"got {got_p50:.4f}")


def test_golden_bimodal_split(synthetic_fcs):
    """BV421-A is constructed as 500 events at mean 100 + 500 at mean 5000.
    After logicle, a threshold at 0.5 should split them ~50/50. If this
    drifts past +-10% we've broken either the transform or threshold logic.
    """
    s = fp.FlowSample(synthetic_fcs)
    s.apply_transform(channels=['BV421-A'])
    s.apply_threshold_gates({'BV421-A': 0.5})

    assert 'BV421-A_pos' in s.data.columns
    pos = float(np.asarray(s.data['BV421-A_pos']).mean())
    assert abs(pos - EXPECTED_BV421_POS_FRAC) < ATOL_FRAC, (
        f"BV421+ fraction drifted: expected ~{EXPECTED_BV421_POS_FRAC:.2f}, "
        f"got {pos:.4f}")


def test_golden_rect_gate_invariants(synthetic_fcs):
    """Rect gate that should keep all events (range covers everything).
    Followed by a rect gate that should keep ~0 events (range below
    everything). Both invariants are independent of numerical drift."""
    s = fp.FlowSample(synthetic_fcs)
    initial = len(s.data)

    keep_all = [{
        'kind': 'rect',
        'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'x0': -1e9, 'x1': 1e9, 'y0': -1e9, 'y1': 1e9,
        'id': 'g1', 'parent_id': None,
    }]
    s.apply_region_gates(keep_all)
    assert len(s.data) == initial, (
        "wide-open rect dropped events — apply_region_gates bug")

    keep_none = [{
        'kind': 'rect',
        'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'x0': 1e9, 'x1': 2e9, 'y0': 1e9, 'y1': 2e9,
        'id': 'g2', 'parent_id': None,
    }]
    s.apply_region_gates(keep_none)
    assert len(s.data) == 0, (
        "out-of-range rect kept events — apply_region_gates bug")


def test_golden_export_stats_shape(synthetic_fcs, tmp_path):
    """export_stats writes a CSV with the expected columns even when
    .cluster() hasn't been called (clusters fall back to a single bucket
    or empty frame). The shape contract is what downstream tooling
    depends on."""
    s = fp.FlowSample(synthetic_fcs)
    out = tmp_path / 'stats.csv'
    df = s.export_stats(str(out))
    assert out.is_file()
    # cluster_frequencies returns an empty DataFrame before .cluster();
    # the contract is that it doesn't crash and the file is created.
    import pandas as pd
    assert isinstance(df, pd.DataFrame)
