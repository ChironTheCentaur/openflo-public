"""optimize_compensation must recover the spillover planted in its controls.

It fitted a least-squares slope of the destination on the source within the
brightest 5% of the source channel. Measured against a planted FL4 -> FL5 of
0.20:

* tight bead peaks (CV 4%, photon noise): 0.158 -- inside the peak most of
  the source spread is noise, which flattens the slope (errors in variables);
  with 90% of the beads positive, 0.124;
* events piled at the detector ceiling: 0.274 with 0.9% of the events
  saturated, 0.632 with 3.2% -- their source value is clipped, their
  destination value is not.

It now takes median(positive) - median(negative) on every detector over the
same on the source channel, the populations split on the source channel and
saturated events left out first. Each test plants a known matrix and checks
the estimate against it.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import openflo.pipeline as fp
from openflo.synthetic import _write_fcs

DETS = ['FL4-A', 'FL5-A']
CEIL = 262143.0
N = 20_000


def _photon(rng, sig, k=20.0):
    return sig + rng.normal(0, 1, sig.shape) * np.sqrt(k * np.clip(sig, 0, None))


def _beads(rng, src, spill, pos_frac=0.5):
    """Negative beads at 300 and positive beads at 50,000 (CV 4%) in `src`,
    measured through `spill` with photon and electronic noise."""
    true = np.clip(rng.normal(300, 80, (N, 2)), 1, None)
    k = int(N * pos_frac)
    true[:k, src] = rng.normal(50_000, 2_000, k)
    return _photon(rng, true @ spill) + rng.normal(0, 30, (N, 2))


def _estimate(tmp_path, controls, dets=DETS):
    paths = {}
    for det, ev in controls.items():
        p = tmp_path / f'{det}.fcs'
        _write_fcs(str(p), pd.DataFrame(ev, columns=dets))
        paths[det] = str(p)
    chans, m = fp.optimize_compensation(dets, paths)
    assert chans == dets
    return m


def test_tight_bead_peaks_give_the_planted_matrix(tmp_path):
    """Before: FL4->FL5 0.158, FL5->FL4 0.025. The matrix is asymmetric, so
    a transposed result fails too."""
    spill = np.array([[1.0, 0.20], [0.03, 1.0]])
    rng = np.random.default_rng(0)
    m = _estimate(tmp_path, {'FL4-A': _beads(rng, 0, spill),
                             'FL5-A': _beads(rng, 1, spill)})
    np.testing.assert_allclose(m, spill, atol=0.002)


def test_mostly_positive_beads(tmp_path):
    """90% positive: the brightest 5% is deep inside the positive peak.
    Before: 0.124."""
    spill = np.array([[1.0, 0.20], [0.0, 1.0]])
    m = _estimate(tmp_path, {'FL4-A': _beads(np.random.default_rng(1), 0,
                                              spill, pos_frac=0.9)})
    assert abs(m[0, 1] - 0.20) < 0.002


@pytest.mark.parametrize('sat_frac', [0.01, 0.03, 0.10])
def test_saturated_events_do_not_bias_the_estimate(tmp_path, sat_frac):
    """A broad bright population clipped at the ceiling. Before: 0.27 at 1%
    saturated, 0.63 at 3%, and 0 at 10%."""
    from scipy.stats import norm
    rng = np.random.default_rng(5)
    true = np.clip(rng.normal(300, 80, (N, 2)), 1, None)
    mu = CEIL - norm.ppf(1 - 2 * sat_frac) * 50_000
    true[: N // 2, 0] = rng.normal(mu, 50_000, N // 2)
    ev = np.minimum(true @ np.array([[1.0, 0.20], [0.0, 1.0]])
                    + rng.normal(0, 30, (N, 2)), CEIL)
    assert abs((ev[:, 0] >= CEIL).mean() - sat_frac) < 0.01
    m = _estimate(tmp_path, {'FL4-A': ev})
    assert abs(m[0, 1] - 0.20) < 0.003


@pytest.mark.parametrize('pos_frac', [0.02, 0.2, 0.5])
def test_correlated_autofluorescence_is_not_spillover(tmp_path, pos_frac):
    """Cells whose autofluorescence is correlated across the detectors (0.64)
    with a small true spill of 0.010. The negative population must be taken
    whole: the dimmest 5% by FL4 are dim in FL5 too, and a median difference
    against them read 0.020 (0.39 with 2% positive cells). The old tail
    regression read 0.000 with 2% positive cells."""
    rng = np.random.default_rng(int(pos_frac * 100))
    base = rng.normal(0, 1, N)
    true = np.clip(300 + 80 * (0.8 * base[:, None]
                               + 0.6 * rng.normal(0, 1, (N, 2))), 1, None)
    k = int(N * pos_frac)
    true[:k, 0] += rng.lognormal(np.log(5000), 0.5, k)
    ev = (_photon(rng, true @ np.array([[1.0, 0.01], [0.0, 1.0]]), k=5)
          + rng.normal(0, 20, (N, 2)))
    m = _estimate(tmp_path, {'FL4-A': ev})
    assert abs(m[0, 1] - 0.010) < 0.0015


def test_a_control_without_a_negative_population_is_flagged(tmp_path, caplog):
    """All beads positive: there is nothing to subtract, so the estimate is
    approximate, and the log says so."""
    spill = np.array([[1.0, 0.20], [0.0, 1.0]])
    ev = _beads(np.random.default_rng(3), 0, spill, pos_frac=1.0)
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        m = _estimate(tmp_path, {'FL4-A': ev})
    assert 'no separate negative and positive' in caplog.text
    assert 0.0 < m[0, 1] < 0.25


def test_a_clean_control_is_not_flagged(tmp_path, caplog):
    spill = np.array([[1.0, 0.20], [0.0, 1.0]])
    ev = _beads(np.random.default_rng(4), 0, spill)
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        _estimate(tmp_path, {'FL4-A': ev})
    assert 'no separate negative' not in caplog.text
