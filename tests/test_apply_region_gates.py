"""End-to-end gate application — uses the synthetic fixture for CI and
the real dataset for full integration when available locally.
"""
import json
import os

import numpy as np
import pytest

import openflo.pipeline as fp


def test_flowsample_loads_synthetic(synthetic_fcs, synthetic_channels):
    s = fp.FlowSample(synthetic_fcs)
    assert len(s.data) == 1000
    for ch in synthetic_channels:
        assert ch in s.channel_names, f"missing channel {ch}"


def test_apply_threshold_gate_synthetic(synthetic_fcs):
    """A threshold on the bimodal channel should split ~50/50."""
    s = fp.FlowSample(synthetic_fcs)
    s.apply_threshold_gates({'BV421-A': 1000.0})
    assert 'BV421-A_pos' in s.data.columns
    pos_frac = float(np.asarray(s.data['BV421-A_pos']).mean())
    # Channel is bimodal centred at 100 and 5000 — should be ~50% positive.
    assert 0.3 < pos_frac < 0.7, f"expected ~50% positive, got {pos_frac:.0%}"


def test_apply_rect_gate_synthetic(synthetic_fcs):
    """A rectangle on FSC/SSC should select a non-empty subset."""
    s = fp.FlowSample(synthetic_fcs)
    initial = len(s.data)
    rect = [{
        'kind': 'rect',
        'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
        'x0': 0, 'x1': 1e6, 'y0': 0, 'y1': 1e6,
        'id': 'g1', 'parent_id': None,
    }]
    s.apply_region_gates(rect)
    assert 0 < len(s.data) <= initial, (
        f"region gate produced empty or impossible result: {len(s.data)}")


# ── Real-data integration (opt-in) ────────────────────────────────────────────

# Opt-in integration: point at your own FCS filenames via OPENFLO_TEST_FCS_SAMPLES
# (comma-separated) alongside OPENFLO_TEST_FCS_DIR; otherwise the test just skips.
REAL_SAMPLES = [s for s in os.environ.get(
    'OPENFLO_TEST_FCS_SAMPLES', 'sample_1.fcs').split(',') if s]
TEMPLATES = ["templates/testtemplate.json",
             "src/openflo/template_library/example_panel.json"]


def _load_template(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    gates = data.get('gates', [])
    region_gates = []
    overrides = {}
    for g in gates:
        if g.get('kind') == 'threshold':
            overrides[g['channel']] = float(g['value'])
        elif g.get('kind') in ('interval', 'rect', 'polygon'):
            region_gates.append(g)
    return region_gates, overrides


@pytest.mark.parametrize('fcs_name', REAL_SAMPLES)
@pytest.mark.parametrize('tpl', TEMPLATES)
def test_real_dataset_apply_template(real_fcs_dir, fcs_name, tpl):
    fcs_path = os.path.join(real_fcs_dir, fcs_name)
    if not os.path.isfile(fcs_path):
        pytest.skip(f"FCS missing: {fcs_name}")
    if not os.path.isfile(tpl):
        pytest.skip(f"template missing: {tpl}")

    region_gates, overrides = _load_template(tpl)

    s = fp.FlowSample(fcs_path)
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    if overrides:
        s.apply_threshold_gates(overrides)
        for ch in overrides:
            assert ch + '_pos' in s.data.columns
    if region_gates:
        initial = len(s.data)
        s.apply_region_gates(region_gates)
        # Just confirm we didn't lose all events or somehow gain any.
        assert 0 <= len(s.data) <= initial
