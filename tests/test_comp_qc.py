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

CHANNELS = ["FL4", "FL5", "FL2"]


def _known_matrix():
    """Identity diagonal with a single 0.2 spill: FL4 -> FL5."""
    m = np.eye(3)
    m[0, 1] = 0.2  # source FL4 (row 0) leaks into destination FL5 (col 1)
    return m


# ── spillover_metrics ─────────────────────────────────────────────────────────

def test_metrics_basic():
    m = spillover_metrics(_known_matrix(), CHANNELS)
    assert m["n_channels"] == 3
    assert m["max_offdiag"] == pytest.approx(0.2)
    assert m["max_pair"] == ("FL4", "FL5")
    # Mean over the 6 off-diagonal entries: only one is 0.2.
    assert m["mean_offdiag"] == pytest.approx(0.2 / 6)


def test_metrics_strong_pairs():
    m = spillover_metrics(_known_matrix(), CHANNELS)
    assert m["strong_pairs"] == [("FL4", "FL5", pytest.approx(0.2))]


def test_metrics_strong_pairs_sorted_desc():
    mat = np.eye(3)
    mat[0, 1] = 0.15
    mat[2, 0] = 0.40
    mat[1, 2] = 0.05  # below 0.10 threshold -> excluded
    m = spillover_metrics(mat, CHANNELS)
    vals = [v for _, _, v in m["strong_pairs"]]
    assert vals == sorted(vals, reverse=True)
    assert m["strong_pairs"][0] == ("FL2", "FL4", pytest.approx(0.40))
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


def test_strong_pairs_includes_exact_threshold():
    """An off-diagonal exactly at STRONG_THRESHOLD (0.10) is 'strong' per the
    docstring ('at or above') and must not be dropped by a strict '>'."""
    from openflo.comp_qc import STRONG_THRESHOLD
    mat = np.eye(3)
    mat[0, 1] = STRONG_THRESHOLD          # exactly 0.10
    m = spillover_metrics(mat, ['A', 'B', 'C'])
    assert ('A', 'B', pytest.approx(STRONG_THRESHOLD)) in m['strong_pairs']


def test_metrics_reject_non_finite():
    """A matrix with NaN/inf must raise, not silently propagate into the metrics
    (and the QC figure)."""
    mat = np.eye(3)
    mat[0, 1] = np.nan
    with pytest.raises(ValueError):
        spillover_metrics(mat, ['A', 'B', 'C'])


# ── units and sign ────────────────────────────────────────────────────────────

_FRACTIONS = np.array([[1.0, 0.12, 0.005],
                       [0.03, 1.0, 0.07],
                       [0.0, 0.002, 1.0]])


def test_percent_matrix_gives_the_same_metrics_as_fractions():
    """Each row is read relative to its diagonal. On raw entries the percent
    form reported max_offdiag 12.0 and listed the 0.5% leak, the 3% and the
    7% as 'strong' (>= 0.10)."""
    pct = spillover_metrics(_FRACTIONS * 100, CHANNELS)
    frac = spillover_metrics(_FRACTIONS, CHANNELS)
    assert pct['max_offdiag'] == pytest.approx(0.12)
    assert pct['strong_pairs'] == [('FL4', 'FL5', pytest.approx(0.12))]
    assert pct == {k: (pytest.approx(v) if isinstance(v, float) else v)
                   for k, v in frac.items()}


def test_a_large_negative_entry_is_the_largest_spill():
    """-0.35 is an over-compensation by 35% of FL4's signal. It ranked below
    +0.04 on the signed values, and nothing was 'strong'."""
    mat = np.eye(3)
    mat[0, 1] = -0.35
    mat[1, 2] = 0.04
    m = spillover_metrics(mat, CHANNELS)
    assert m['max_offdiag'] == pytest.approx(0.35)
    assert m['max_pair'] == ('FL4', 'FL5')
    assert m['strong_pairs'] == [('FL4', 'FL5', pytest.approx(-0.35))]
    assert m['mean_offdiag'] == pytest.approx((0.35 + 0.04) / 6)


def test_a_row_is_relative_to_its_own_diagonal():
    """A diagonal of 0.5 doubles that row's relative spill."""
    mat = np.eye(3)
    mat[1, 1] = 0.5
    mat[1, 0] = 0.06
    m = spillover_metrics(mat, CHANNELS)
    assert m['strong_pairs'] == [('FL5', 'FL4', pytest.approx(0.12))]


def test_a_zero_diagonal_is_refused():
    mat = np.eye(3)
    mat[2, 2] = 0.0
    with pytest.raises(ValueError, match='diagonal'):
        spillover_metrics(mat, CHANNELS)


def test_figure_colours_a_negative_entry_by_its_magnitude():
    mat = np.eye(3)
    mat[0, 1] = -0.35
    fig = comp_qc_figure(mat, CHANNELS)
    shown = np.asarray(fig.axes[0].get_images()[0].get_array())
    assert shown[0, 1] == pytest.approx(0.35)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert '-0.35' in texts
