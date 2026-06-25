"""Compensation / spillover QC.

Exercises spillover_metrics on a small known matrix (a 3x3 spillover with a
single 0.2 leak) and checks comp_qc_figure returns a headless Figure with the
expected axes. Agg backend; no figures are shown.
"""
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from openflo.comp_qc import comp_qc_figure, spillover_metrics

CHANNELS = ["FITC", "PE", "APC"]


def _known_matrix():
    """Identity diagonal with a single 0.2 spill: FITC -> PE."""
    m = np.eye(3)
    m[0, 1] = 0.2  # source FITC (row 0) leaks into destination PE (col 1)
    return m


# ── spillover_metrics ─────────────────────────────────────────────────────────

def test_metrics_basic():
    m = spillover_metrics(_known_matrix(), CHANNELS)
    assert m["n_channels"] == 3
    assert m["max_offdiag"] == pytest.approx(0.2)
    assert m["max_pair"] == ("FITC", "PE")
    # Mean over the 6 off-diagonal entries: only one is 0.2.
    assert m["mean_offdiag"] == pytest.approx(0.2 / 6)


def test_metrics_strong_pairs():
    m = spillover_metrics(_known_matrix(), CHANNELS)
    assert m["strong_pairs"] == [("FITC", "PE", pytest.approx(0.2))]


def test_metrics_strong_pairs_sorted_desc():
    mat = np.eye(3)
    mat[0, 1] = 0.15
    mat[2, 0] = 0.40
    mat[1, 2] = 0.05  # below 0.10 threshold -> excluded
    m = spillover_metrics(mat, CHANNELS)
    vals = [v for _, _, v in m["strong_pairs"]]
    assert vals == sorted(vals, reverse=True)
    assert m["strong_pairs"][0] == ("APC", "FITC", pytest.approx(0.40))
    assert len(m["strong_pairs"]) == 2  # the 0.05 leak is dropped


def test_metrics_no_spillover():
    m = spillover_metrics(np.eye(3), CHANNELS)
    assert m["max_offdiag"] == pytest.approx(0.0)
    assert m["strong_pairs"] == []


def test_metrics_rejects_none():
    with pytest.raises(ValueError):
        spillover_metrics(None, CHANNELS)


def test_metrics_rejects_non_square():
    with pytest.raises(ValueError):
        spillover_metrics(np.zeros((2, 3)), ["a", "b"])


def test_metrics_rejects_channel_mismatch():
    with pytest.raises(ValueError):
        spillover_metrics(np.eye(3), ["only", "two"])


# ── comp_qc_figure ─────────────────────────────────────────────────────────────

def test_figure_returns_figure_with_axes():
    fig = comp_qc_figure(_known_matrix(), CHANNELS, title="QC")
    assert isinstance(fig, Figure)
    # One heatmap axes + one colorbar axes.
    assert len(fig.axes) == 2
    heatmap = fig.axes[0]
    assert len(heatmap.get_xticks()) == 3
    assert len(heatmap.get_yticks()) == 3


def test_figure_rejects_bad_input():
    with pytest.raises(ValueError):
        comp_qc_figure(None, CHANNELS)
