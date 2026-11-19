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
from openflo.cell_cycle import CELL_CYCLE_PHASES


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
    # 'FL5' / 'FL2' must NOT be mistaken for the 'pi' DNA token — and neither must
    # 'PI3K' (a signalling marker): a digit right after 'pi' used to satisfy the
    # boundary and wrongly tag it as the PI DNA stain.
    s = types.SimpleNamespace(
        data=pd.DataFrame({'FL5-A': [], 'FL2-A': [], 'PI3K-A': []}),
        channel_labels={'FL5-A': 'CD901b', 'FL2-A': 'CD902', 'PI3K-A': 'PI3K'})
    assert fp.find_dna_channel(s) is None
    # …but a real PI channel still matches.
    s2 = types.SimpleNamespace(
        data=pd.DataFrame({'PI-A': []}), channel_labels={'PI-A': 'PI'})
    assert fp.find_dna_channel(s2) == 'PI-A'


# ── analyze_dna ──────────────────────────────────────────────────────────────

def test_analyze_dna_finds_peaks_and_fractions():
    m = fp.analyze_dna(_dna_values())
    assert m['ok']
    # Peaks are smoothed-histogram modes of sd-5 (G1) and sd-8 (G2) populations; seeds
    # 0-7 land within 0.7 and 3. The bounds (~1.5-2 peak SDs) check the right peak is found.
    assert m['g1_mean'] == pytest.approx(100, abs=8)
    # G2/M: 2,000 events at sd 8, mode noisier than G1's
    assert m['g2_mean'] == pytest.approx(200, abs=15)
    # G1 dominates; G2M present; S smallest. Fractions sum ~100.
    assert m['pct_g1'] > m['pct_g2m'] > m['pct_s']
    assert m['pct_g1'] + m['pct_s'] + m['pct_g2m'] == pytest.approx(100, abs=1e-6)


def test_analyze_dna_too_few_events():
    m = fp.analyze_dna([1.0, 2.0, 3.0])
    assert m['ok'] is False
    assert np.isnan(m['pct_g1'])


def test_analyze_dna_refuses_a_single_broad_bell():
    # A scatter channel (one bell, CV 20 %) has no G1 or G2/M peak. It was
    # reported as a fit with G1 100 %, S 0 %, G2/M 0 %.
    rng = np.random.default_rng(3)
    v = rng.normal(35_000, 7_000, 20_000)
    m = fp.analyze_dna(v)
    assert m['ok'] is False
    assert np.isnan(m['pct_g1']) and np.isnan(m['pct_g2m'])
    assert 'too wide' in m['reason']
    cv = float(m['reason'].split('CV ')[1].split(' %')[0])
    # The CV it reports is the bell's own (true 20 %, sample SD over mean),
    # read from the half-maximum width of a smoothed 256-bin histogram and
    # printed as a whole percent: within 2 points.
    assert cv == pytest.approx(100 * np.std(v, ddof=1) / np.mean(v), abs=2)
    assert set(fp.assign_phase(v, m)) == {'NA'}


def _aneuploid_values(di, seed):
    """A DNA-aneuploid tumour: half diploid cells (G0/G1 at 50,000, few
    cycling) and half an aneuploid clone at DNA index ``di`` (its own G1,
    S and G2/M, the G2/M at 2 * di times the diploid G1). CV 5 %."""
    rng = np.random.default_rng(seed)
    g1, cv = 50_000.0, 0.05

    def s_phase(lo, n):
        s = rng.uniform(lo, 2 * lo, n)
        return s + rng.normal(0, cv * s)
    a1 = di * g1
    return np.concatenate([
        rng.normal(g1, cv * g1, 9_000), rng.normal(2 * g1, cv * 2 * g1, 500),
        s_phase(g1, 500),
        rng.normal(a1, cv * a1, 5_500), rng.normal(2 * a1, cv * 2 * a1, 2_000),
        s_phase(a1, 2_500)])


@pytest.mark.parametrize('di', [1.2, 1.3, 1.5])
def test_analyze_dna_fits_an_aneuploid_histogram(di):
    # The clone's G1 sits between the diploid G1 and G2/M and its G2/M far
    # above them: still a DNA histogram, fitted on the tallest (diploid) G1.
    for seed in range(3):
        v = _aneuploid_values(di, seed)
        m = fp.analyze_dna(v)
        assert m['ok'] is True, (di, seed, m['reason'])
        assert m['reason'] == ''
        assert m['g1_mean'] == pytest.approx(50_000, abs=2_500)  # 1 G1 SD
        # The phase % are the events between the model's own boundaries,
        # counted here again from the values (the same sums: exact).
        g1_lo = m['g1_mean'] - (m['g1_hi'] - m['g1_mean'])
        g2_hi = m['g2_mean'] + (m['g2_mean'] - m['g2_lo'])
        n_g1 = np.count_nonzero((v >= g1_lo) & (v <= m['g1_hi']))
        n_s = np.count_nonzero((v > m['g1_hi']) & (v < m['g2_lo']))
        n_g2 = np.count_nonzero((v >= m['g2_lo']) & (v <= g2_hi))
        tot = n_g1 + n_s + n_g2
        assert m['pct_g1'] == 100 * n_g1 / tot
        assert m['pct_s'] == 100 * n_s / tot
        assert m['pct_g2m'] == 100 * n_g2 / tot


def test_analyze_dna_keeps_a_non_cycling_g1_peak():
    # Resting cells: one narrow G1 peak (CV 5 %) and no G2/M. Still a DNA
    # histogram; G2/M is placed at 2x G1 and the model says it was not found.
    rng = np.random.default_rng(5)
    v = rng.normal(50_000, 2_500, 20_000)
    m = fp.analyze_dna(v)
    assert m['ok'] and m['reason'] == ''
    assert m['g2_found'] is False
    assert m['g2_mean'] == pytest.approx(2 * m['g1_mean'])
    assert m['pct_g2m'] == 0
    assert fp.analyze_dna(_dna_values())['g2_found'] is True


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
    assert set(s.data['cell_cycle'].unique()) <= set(CELL_CYCLE_PHASES) | {'NA'}
    assert s.cell_cycle_result['pct_g1'] > 40


def test_flowsample_cell_cycle_no_dye_is_noop():
    df = pd.DataFrame({'FL5-A': np.arange(100.0)})
    s = _stub_sample(df, {'FL5-A': 'CD901b'})
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


def test_dna_width_partner_matches_lowercase_suffix():
    """The -W/-H partner lookup is case-insensitive — it finds a lowercase
    'DNA-w' column for a 'DNA-A' DNA channel (used for doublet exclusion)."""
    s = fp.FlowSample.from_dataframe(
        pd.DataFrame({'DNA-A': [1.0, 2.0], 'DNA-w': [1.0, 1.0]}), name='x')
    assert s._dna_width_partner('DNA-A') == 'DNA-w'
