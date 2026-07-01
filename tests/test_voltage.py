"""Voltage titration / Stain Index tool.

The pure metric layer is the part worth pinning down — robust SD/CV, the
Stain Index formula, the GMM neg/pos split, the plateau recommendation,
and reading $PnV out of FCS metadata. The IO orchestration (analyze over
files) is a thin loop over these + FlowSample and isn't re-tested here.
"""
import types

import numpy as np
import pytest

from openflo.voltage import VoltageTitration as VT
from openflo.voltage import read_pmt_voltage

# ── robust stats ─────────────────────────────────────────────────────────────

def test_robust_sd_matches_mad_scaling():
    a = np.array([0.0, 0.0, 0.0, 0.0, 10.0])   # MAD = 0 → rSD 0 (robust)
    assert VT.robust_sd(a) == 0.0
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])     # MAD = 1 → rSD = 1.4826
    assert VT.robust_sd(b) == pytest.approx(1.4826, rel=1e-4)


def test_robust_cv_percent():
    a = np.array([8.0, 10.0, 10.0, 10.0, 12.0])  # median 10, MAD 0
    assert VT.robust_cv(a) == 0.0
    b = np.full(50, 0.0)
    assert np.isnan(VT.robust_cv(np.concatenate([b])))  # median 0 → NaN


# ── stain index ──────────────────────────────────────────────────────────────

def test_stain_index_formula():
    rng = np.random.default_rng(0)
    neg = rng.normal(100, 10, 5000)
    pos = rng.normal(1000, 50, 5000)
    si = VT.stain_index(pos, neg)
    # (1000-100) / (2 * ~10) ≈ 45; robust SD of N(.,10) ≈ 10.
    expected = (np.median(pos) - np.median(neg)) / (2 * VT.robust_sd(neg))
    assert si == pytest.approx(expected, rel=1e-9)
    assert 35 < si < 60


def test_stain_index_nan_guards():
    assert np.isnan(VT.stain_index([], [1, 2, 3]))
    assert np.isnan(VT.stain_index([1, 2], []))
    assert np.isnan(VT.stain_index([5, 5], [3, 3]))   # rSD(neg)=0


# ── GMM split ────────────────────────────────────────────────────────────────

def test_split_pos_neg_separates_bimodal():
    rng = np.random.default_rng(1)
    neg = rng.normal(50, 8, 3000)
    pos = rng.normal(5000, 400, 2000)
    vals = np.concatenate([neg, pos])
    rng.shuffle(vals)
    got_neg, got_pos = VT.split_pos_neg(vals)
    # Roughly recovers the 3000/2000 split and orders by magnitude.
    assert np.median(got_neg) < np.median(got_pos)
    assert abs(got_neg.size - 3000) < 300
    assert abs(got_pos.size - 2000) < 300


def test_split_pos_neg_too_few_events():
    neg, pos = VT.split_pos_neg([1.0, 2.0, 3.0])
    assert pos.size == 0 and neg.size == 3


# ── plateau recommendation ───────────────────────────────────────────────────

def test_recommend_plateau_picks_lowest_on_plateau():
    volts = [300, 400, 500, 600, 700]
    si    = [10, 18, 19.5, 20, 19]      # max 20 at 600; 95% = 19 reached at 500
    assert VT.recommend_plateau(volts, si, frac=0.95) == 500


def test_recommend_plateau_ignores_nan_and_none():
    volts = [None, 400, 500]
    si    = [float('nan'), 8.0, 10.0]
    assert VT.recommend_plateau(volts, si, frac=0.95) == 500


def test_recommend_plateau_empty():
    assert VT.recommend_plateau([], []) is None
    assert VT.recommend_plateau([400], [float('nan')]) is None


# ── $PnV reading ─────────────────────────────────────────────────────────────

def test_read_pmt_voltage_resolves_index_and_key_forms():
    names = ['FSC-A', 'SSC-A', 'PE-A']
    # PE-A is parameter 3 → $P3V.
    assert read_pmt_voltage({'$P3V': '450'}, names, 'PE-A') == 450.0
    assert read_pmt_voltage({'P3V': '450'}, names, 'PE-A') == 450.0
    assert read_pmt_voltage({'$p3v': '450'}, names, 'PE-A') == 450.0


def test_read_pmt_voltage_missing_returns_none():
    names = ['FSC-A', 'PE-A']
    assert read_pmt_voltage({}, names, 'PE-A') is None
    assert read_pmt_voltage({'$P2V': 'x'}, names, 'PE-A') is None   # unparseable
    assert read_pmt_voltage({'$P2V': '400'}, names, 'NOPE') is None  # no channel


# ── per-channel curve + replicate pooling ────────────────────────────────────

def _bimodal(neg_n, pos_n, seed):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(50, 8, neg_n),
                           rng.normal(5000, 400, pos_n)])


def test_channel_result_pools_same_voltage_replicates():
    # Two files at 400 V (replicates) + one at 500 V → two curve points,
    # and the 400 V point pools both files' events.
    points = [
        (400.0, _bimodal(1500, 1000, 1)),
        (400.4, _bimodal(1500, 1000, 2)),   # rounds to 400 → pooled
        (500.0, _bimodal(1500, 1200, 3)),
    ]
    res = VT._channel_result('PE-A', points, frac=0.95, voltage_round=0)
    by_v = {r['voltage']: r for r in res['rows']}
    assert set(by_v) == {400.0, 500.0}
    assert by_v[400.0]['n_files'] == 2          # both 400 V files pooled
    assert by_v[500.0]['n_files'] == 1
    # Pooled point carries both files' events (~2 * 2500).
    assert by_v[400.0]['n_events'] == pytest.approx(5000, abs=10)
    assert np.isfinite(res['recommended_voltage'])


def test_channel_result_none_voltage_excluded_from_recommendation():
    points = [(None, _bimodal(1500, 1000, 1)),
              (450.0, _bimodal(1500, 1200, 2))]
    res = VT._channel_result('PE-A', points)
    # The None-voltage point still appears as a row but can't drive the
    # recommendation (only the 450 V point can).
    assert res['recommended_voltage'] == 450.0
    assert any(r['voltage'] is None for r in res['rows'])


def test_channel_set_all_channels_is_ordered_union():
    a = types.SimpleNamespace(fluor_channels=['CD11b', 'CD45'])
    b = types.SimpleNamespace(fluor_channels=['CD45', 'CD34'])
    got = VT._channel_set([(a, 'a'), (b, 'b')], None, all_channels=True)
    assert got == ['CD11b', 'CD45', 'CD34']      # first-seen order, deduped


def test_channel_set_explicit_passthrough():
    assert VT._channel_set([], ['PE-A', 'APC-A'], all_channels=False) == \
        ['PE-A', 'APC-A']
