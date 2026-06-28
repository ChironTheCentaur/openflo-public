"""Tests for openflo.calibration (MESF/ABC fluorescence calibration) using the
synthetic rainbow-bead generator so the true MESF=2*MFI+100 line is known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.calibration import (
    apply_calibration,
    detect_bead_peaks,
    fit_mesf_calibration,
)
from openflo.pipeline import FlowSample
from openflo.synthetic import (
    CAL_CHANNEL,
    CAL_PEAK_MESF,
    make_calibration_beads,
)


def test_fit_recovers_line():
    mfi = np.array([100, 800, 3000, 11000, 40000])
    known = 2.0 * mfi + 100.0
    cal = fit_mesf_calibration(mfi, known)
    assert cal['slope'] == pytest.approx(2.0, rel=1e-6)
    assert cal['intercept'] == pytest.approx(100.0, abs=1e-3)
    assert cal['r2'] == pytest.approx(1.0)
    assert cal['n'] == 5


def test_fit_too_few_pairs():
    with pytest.raises(ValueError, match="2"):
        fit_mesf_calibration([100.0], [200.0])


def test_apply_calibration_clips():
    out = apply_calibration([0.0, 1000.0], slope=2.0, intercept=-500.0)
    assert out[0] == 0.0                       # 2*0-500 < 0 → clipped
    assert out[1] == 1500.0


def test_detect_peaks_finds_six():
    rng = np.random.default_rng(0)
    centres = [200, 800, 3000, 11000, 40000, 150000]
    v = np.concatenate([rng.normal(c, c * 0.05, 2000) for c in centres])
    peaks = detect_bead_peaks(v, n_peaks=6)
    assert len(peaks) == 6
    # Recovered peak medians track the true centres (ascending).
    for got, want in zip(peaks, centres, strict=True):
        assert abs(got - want) / want < 0.15


def test_synthetic_beads_calibrate(tmp_path):
    fcs, csv = make_calibration_beads(str(tmp_path), n=12000, seed=1)
    s = FlowSample(fcs)
    s.run_qc()
    assert CAL_CHANNEL in s.channel_names
    # Detect peaks from the bead FCS, assign the known MESF values, fit.
    peaks = detect_bead_peaks(s.raw[CAL_CHANNEL].to_numpy(),
                              n_peaks=len(CAL_PEAK_MESF))
    cal = fit_mesf_calibration(peaks, CAL_PEAK_MESF)
    # The generator used MESF = 2*MFI + 100 → recovered slope ≈ 2.
    assert cal['slope'] == pytest.approx(2.0, rel=0.05)
    assert cal['r2'] > 0.999
    # The peaks CSV the generator wrote matches.
    pk = pd.read_csv(csv)
    assert list(pk['MESF']) == CAL_PEAK_MESF
