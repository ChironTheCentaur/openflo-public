"""Cell-cycle DNA-content modelling.

Covers the pure core (find_dna_channel, analyze_dna, assign_phase), the
FlowSample.cell_cycle method on a synthetic bimodal DNA distribution, and
the 'category' gate kind that surfaces phases as populations.
"""
import types

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp


def _dna_values(n_g1=6000, n_s=1000, n_g2=2000, seed=0):
    """Classic cell-cycle DNA histogram: G1 at 100, G2/M at 200 (2× DNA),
    S phase spread uniformly between."""
    rng = np.random.default_rng(seed)
    g1 = rng.normal(100, 5, n_g1)
    g2 = rng.normal(200, 8, n_g2)
    s  = rng.uniform(115, 185, n_s)
    v = np.concatenate([g1, s, g2])
    rng.shuffle(v)
    return v


# ── find_dna_channel ─────────────────────────────────────────────────────────

def test_find_dna_channel_by_label_prefers_area():
    s = types.SimpleNamespace(
        data=pd.DataFrame({'FSC-A': [], 'FxCycle-W': [], 'FxCycle-A': []}),
        channel_labels={'FxCycle-A': 'FxCycle Violet', 'FxCycle-W': 'FxCycle'})
    assert fp.find_dna_channel(s) == 'FxCycle-A'


def test_find_dna_channel_recognises_several_dyes():
    for det, lbl in [('V450-A', 'DAPI'), ('B695-A', '7-AAD'),
                     ('YG610-A', 'Propidium Iodide'), ('R670-A', 'DRAQ5')]:
        s = types.SimpleNamespace(
            data=pd.DataFrame({det: [], 'FSC-A': []}),
            channel_labels={det: lbl})
        assert fp.find_dna_channel(s) == det


def test_find_dna_channel_pi_is_word_bounded():
    # 'PE' / 'APC' must NOT be mistaken for the 'pi' DNA token.
    s = types.SimpleNamespace(
        data=pd.DataFrame({'PE-A': [], 'APC-A': []}),
        channel_labels={'PE-A': 'CD11b', 'APC-A': 'CD34'})
    assert fp.find_dna_channel(s) is None


# ── analyze_dna ──────────────────────────────────────────────────────────────

def test_analyze_dna_finds_peaks_and_fractions():
    m = fp.analyze_dna(_dna_values())
    assert m['ok']
    assert m['g1_mean'] == pytest.approx(100, abs=8)
    assert m['g2_mean'] == pytest.approx(200, abs=15)
    # G1 dominates; G2M present; S smallest. Fractions sum ~100.
    assert m['pct_g1'] > m['pct_g2m'] > m['pct_s']
    assert m['pct_g1'] + m['pct_s'] + m['pct_g2m'] == pytest.approx(100, abs=1e-6)


def test_analyze_dna_too_few_events():
    m = fp.analyze_dna([1.0, 2.0, 3.0])
    assert m['ok'] is False
    assert np.isnan(m['pct_g1'])


def test_assign_phase_uses_model_boundaries():
    m = fp.analyze_dna(_dna_values())
    labels = fp.assign_phase(
        np.array([100.0, 150.0, 200.0, np.nan, 1.0]), m)
    assert labels[0] == 'G1'
    assert labels[2] == 'G2M'
    assert labels[3] == 'NA'          # non-finite
    assert labels[4] == 'sub-G1'      # well below G1


def test_assign_phase_invalid_model_all_na():
    bad = fp.analyze_dna([1.0])       # ok == False
    labels = fp.assign_phase(np.array([1.0, 2.0]), bad)
    assert list(labels) == ['NA', 'NA']


# ── FlowSample.cell_cycle ────────────────────────────────────────────────────

def _stub_sample(df, labels):
    """A FlowSample with __init__ bypassed, carrying just what cell_cycle
    touches."""
    s = fp.FlowSample.__new__(fp.FlowSample)
    s.data = df
    s.channel_labels = labels
    s.name = 'cc'
    s.cell_cycle_result = None
    return s


def test_flowsample_cell_cycle_autodetect_and_column():
    df = pd.DataFrame({'FSC-A': np.ones(9000), 'FxCycle-A': _dna_values()})
    s = _stub_sample(df, {'FxCycle-A': 'FxCycle', 'FSC-A': 'FSC-A'})
    s.cell_cycle()
    assert s.cell_cycle_result['channel'] == 'FxCycle-A'
    assert 'cell_cycle' in s.data.columns
    assert set(s.data['cell_cycle'].unique()) <= set(fp.CELL_CYCLE_PHASES) | {'NA'}
    assert s.cell_cycle_result['pct_g1'] > 40


def test_flowsample_cell_cycle_no_dye_is_noop():
    df = pd.DataFrame({'PE-A': np.arange(100.0)})
    s = _stub_sample(df, {'PE-A': 'CD11b'})
    s.cell_cycle()
    assert s.cell_cycle_result is None
    assert 'cell_cycle' not in s.data.columns


# ── 'category' gate kind ─────────────────────────────────────────────────────

def test_category_gate_mask_selects_value():
    gate = {'kind': 'category', 'channel': 'cell_cycle', 'value': 'G2M'}
    df = pd.DataFrame({'cell_cycle': ['G1', 'S', 'G2M', 'G2M', 'NA']})
    assert list(fp.gate_to_mask(gate, df)) == [False, False, True, True, False]


def test_category_gate_missing_column_empty():
    gate = {'kind': 'category', 'channel': 'cell_cycle', 'value': 'G1'}
    df = pd.DataFrame({'X': [1.0, 2.0]})
    assert list(fp.gate_to_mask(gate, df)) == [False, False]


def test_category_describe_gate():
    assert fp.describe_gate(
        {'kind': 'category', 'value': 'G1', 'name': 'G1 phase'}) == '=  G1 phase'
