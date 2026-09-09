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


def test_detect_bead_peaks_small_input_returns_sorted():
    """Fewer finite positive values than requested peaks → the sorted values
    themselves (the small-input fallback branch), not a k-means fit."""
    peaks = detect_bead_peaks([300.0, 100.0, 200.0], n_peaks=6)
    assert list(peaks) == [100.0, 200.0, 300.0]


def test_apply_calibration_no_clip_keeps_negative():
    """clip=False leaves sub-zero calibrated values intact — only the default
    clip=True path was asserted."""
    out = apply_calibration([0.0, 1000.0], slope=2.0, intercept=-500.0, clip=False)
    assert out[0] == -500.0 and out[1] == 1500.0


def test_calibration_r2_flags_bad_bead_assignment():
    """r2 must DROP on a mis-assigned / non-linear bead set (its documented job
    as a bad-assignment flag) — it was only ever asserted ~1.0 on perfect data."""
    good = fit_mesf_calibration([100.0, 200, 300, 400], [300.0, 500, 700, 900])
    bad = fit_mesf_calibration([100.0, 200, 300, 400], [1000.0, 2000, 9000, 4000])
    assert good['r2'] == pytest.approx(1.0, abs=1e-6)
    assert bad['r2'] < 0.5                    # bad assignment flagged (~0.34)


def test_r2_is_nan_when_the_assigned_values_have_no_spread():
    """r2 is UNDEFINED with no variance in the response, and it used to return
    1.0 there — the strongest "good fit" signal possible, attached to a fit
    whose slope is ~0 and which turns every converted value into the
    intercept. A user who mis-entered the bead column saw R²=1.0000 and applied
    a garbage calibration to every loaded sample."""
    import math

    from openflo.calibration import fit_mesf_calibration
    out = fit_mesf_calibration([100.0, 400.0, 1600.0], [500.0, 500.0, 500.0])
    assert not math.isfinite(out['r2']), (
        f"r2={out['r2']} for a degenerate calibration — it reads as a perfect "
        'fit')
    assert abs(out['slope']) < 1e-9, 'expected the degenerate ~0 slope'


def test_r2_is_still_exact_for_a_real_calibration():
    """The guard must not disturb the normal path — this is the golden's
    calibration.r2 metric."""
    from openflo.calibration import fit_mesf_calibration
    out = fit_mesf_calibration([100.0, 400.0, 1600.0, 6400.0],
                               [300.0, 900.0, 3300.0, 12900.0])
    assert out['r2'] == pytest.approx(1.0, abs=1e-9)
    assert out['slope'] == pytest.approx(2.0, abs=1e-9)
    assert out['intercept'] == pytest.approx(100.0, abs=1e-9)
    assert out['n'] == 4


def test_a_noisy_calibration_reports_r2_below_one():
    """Guard the guard: r2 must still be able to say 'bad fit', which is the
    job the docstring claims for it."""
    from openflo.calibration import fit_mesf_calibration
    out = fit_mesf_calibration([100.0, 200.0, 300.0, 400.0],
                               [300.0, 9000.0, 400.0, 12000.0])
    assert 0.0 <= out['r2'] < 0.95, f"r2={out['r2']} for scattered peaks"
