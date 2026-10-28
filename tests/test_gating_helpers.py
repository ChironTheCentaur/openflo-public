"""Tests for openflo.gating_helpers (headless singlet + FMO gating)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openflo.gating_helpers import fmo_threshold, fmo_threshold_gate, singlet_gate
from openflo.pipeline import gate_to_mask


def _make_singlet_doublet_df(n_singlets=4000, n_doublets=1000, seed=0):
    """Singlets: height ~= area * slope + noise (tight diagonal).
    Doublets: same height range but much larger area (off-diagonal)."""
    rng = np.random.default_rng(seed)
    slope = 0.95  # height / area

    # Singlets: pick a height, derive area = height / slope, add mild noise.
    h_s = rng.uniform(20_000, 120_000, size=n_singlets)
    a_s = h_s / slope + rng.normal(0, 1500, size=n_singlets)

    # Doublets: similar heights but ~1.7x the area (fall below the diagonal).
    h_d = rng.uniform(20_000, 120_000, size=n_doublets)
    a_d = (h_d / slope) * 1.7 + rng.normal(0, 1500, size=n_doublets)

    area = np.concatenate([a_s, a_d])
    height = np.concatenate([h_s, h_d])
    is_singlet = np.concatenate(
        [np.ones(n_singlets, dtype=bool), np.zeros(n_doublets, dtype=bool)]
    )
    df = pd.DataFrame({"FSC-A": area, "FSC-H": height, "is_singlet": is_singlet})
    return df


def test_singlet_gate_schema():
    df = _make_singlet_doublet_df()
    g = singlet_gate(df, area="FSC-A", height="FSC-H", id="singlets")

    assert set(g) == {
        "kind",
        "x_channel",
        "y_channel",
        "vertices",
        "parent_id",
        "color",
        "enabled",
        "id",
    }
    assert g["kind"] == "polygon"
    assert g["x_channel"] == "FSC-A"
    assert g["y_channel"] == "FSC-H"
    assert g["id"] == "singlets"
    assert g["parent_id"] is None
    assert g["enabled"] is True
    assert isinstance(g["color"], str) and g["color"].startswith("#")

    verts = g["vertices"]
    assert isinstance(verts, list) and len(verts) >= 3
    for v in verts:
        assert len(v) == 2
        assert all(isinstance(c, float) for c in v)


def test_singlet_gate_keeps_singlets_drops_doublets():
    df = _make_singlet_doublet_df()
    g = singlet_gate(df, area="FSC-A", height="FSC-H")
    mask = gate_to_mask(g, df)

    truth = df["is_singlet"].to_numpy()
    # Keeps the vast majority of true singlets.
    assert mask[truth].mean() > 0.9
    # Rejects the vast majority of doublets.
    assert mask[~truth].mean() < 0.1


def test_singlet_gate_custom_parent_and_color():
    df = _make_singlet_doublet_df()
    g = singlet_gate(df, parent_id="root", color="#abcdef", id="s1")
    assert g["parent_id"] == "root"
    assert g["color"] == "#abcdef"
    assert g["id"] == "s1"


def test_singlet_gate_missing_channel_raises():
    df = _make_singlet_doublet_df()
    # pandas would raise a bare KeyError('NOPE') on its own; the message must
    # say which role is missing, so a user knows which channel to pick.
    with pytest.raises(KeyError, match="area channel 'NOPE'"):
        singlet_gate(df, area="NOPE", height="FSC-H")
    with pytest.raises(KeyError, match="height channel 'NOPE'"):
        singlet_gate(df, area="FSC-A", height="NOPE")


def _wide_singlet_df(seed=3):
    """Singlets on FSC-A = 2 x FSC-H with a 15 % spread, plus doublets at
    FSC-A = 3.1 x FSC-H: wide enough that a band fitted on height/area
    instead of area/height reaches the doublets."""
    rng = np.random.default_rng(seed)
    h_s = rng.lognormal(10.0, 0.4, 20_000)
    a_s = 2.0 * h_s * rng.normal(1.0, 0.15, h_s.size)
    h_d = rng.lognormal(10.2, 0.4, 1_500)
    a_d = 3.1 * h_d * rng.normal(1.0, 0.15, h_d.size)
    return pd.DataFrame({"FSC-A": np.r_[a_s, a_d], "FSC-H": np.r_[h_s, h_d]})


def test_singlet_gate_band_is_area_over_height():
    # Expected corners worked out here from the definition, not from the
    # code under test: median +/- 3 robust SD of FSC-A/FSC-H, the rays
    # area = r * height clipped to the 0.1-99.9th height percentiles.
    df = _wide_singlet_df()
    a, h = df["FSC-A"].to_numpy(), df["FSC-H"].to_numpy()
    r = a / h
    med = np.median(r)
    sd = np.median(np.abs(r - med)) / 0.6744897501960817
    r_lo, r_hi = med - 3 * sd, med + 3 * sd
    h_lo, h_hi = np.percentile(h, [0.1, 99.9])
    expected = [[r_lo * h_lo, h_lo], [r_lo * h_hi, h_hi],
                [r_hi * h_hi, h_hi], [r_hi * h_lo, h_lo]]

    g = singlet_gate(df, area="FSC-A", height="FSC-H")
    np.testing.assert_allclose(g["vertices"], expected, rtol=1e-9)


def test_singlet_gate_matches_the_auto_gate_proposal():
    # Edit > Add singlet gate and the auto-gate dialog must draw one gate.
    from openflo.pipeline import auto_singlet_gate

    df = _wide_singlet_df(seed=7)
    for k in (2.0, 3.0, 4.5):
        g = singlet_gate(df, area="FSC-A", height="FSC-H", k=k)
        verts, _q = auto_singlet_gate(df["FSC-A"], df["FSC-H"], k=k)
        np.testing.assert_allclose(g["vertices"], verts, rtol=1e-12)


def test_singlet_gate_too_few_events_raises():
    df = pd.DataFrame({"FSC-A": [2.0, 4.0, 6.0], "FSC-H": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="singlet_gate"):
        singlet_gate(df, area="FSC-A", height="FSC-H")


def test_fmo_threshold_matches_percentile():
    rng = np.random.default_rng(7)
    vals = rng.normal(100.0, 15.0, size=50_000)
    for pct in (90.0, 95.0, 99.0, 99.9):
        cut = fmo_threshold(vals, percentile=pct)
        assert cut == pytest.approx(float(np.percentile(vals, pct)), rel=1e-9)


def test_fmo_threshold_ignores_nan():
    vals = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.inf])
    cut = fmo_threshold(vals, percentile=50.0)
    assert cut == pytest.approx(np.percentile([1.0, 2.0, 3.0, 4.0], 50.0))


def test_fmo_threshold_empty_raises():
    with pytest.raises(ValueError):
        fmo_threshold([np.nan, np.inf])


def test_fmo_threshold_gate_schema_and_value():
    rng = np.random.default_rng(11)
    fmo_df = pd.DataFrame({"CD3": rng.normal(50.0, 10.0, size=20_000)})
    g = fmo_threshold_gate(fmo_df, "CD3", percentile=99.0)

    assert set(g) == {
        "kind",
        "channel",
        "value",
        "parent_id",
        "color",
        "enabled",
        "id",
    }
    assert g["kind"] == "threshold"
    assert g["channel"] == "CD3"
    assert g["id"] == "CD3+"
    assert g["parent_id"] is None
    assert g["enabled"] is True
    assert isinstance(g["value"], float)
    assert g["value"] == pytest.approx(
        float(np.percentile(fmo_df["CD3"].to_numpy(), 99.0)), rel=1e-9
    )


def test_fmo_threshold_gate_positive_above_cutoff():
    # FMO background near 0; a "stained" sample sits well above the cutoff.
    rng = np.random.default_rng(3)
    fmo_df = pd.DataFrame({"CD8": rng.normal(0.0, 5.0, size=20_000)})
    g = fmo_threshold_gate(fmo_df, "CD8", percentile=99.0, id="cd8pos")
    assert g["id"] == "cd8pos"

    stained = pd.DataFrame(
        {"CD8": np.concatenate([rng.normal(0.0, 5.0, 5000), rng.normal(80.0, 5.0, 5000)])}
    )
    mask = gate_to_mask(g, stained)
    # The bright population (last 5000) is overwhelmingly called positive,
    # the background (first 5000) is overwhelmingly negative.
    assert mask[5000:].mean() > 0.95
    assert mask[:5000].mean() < 0.05


def test_fmo_threshold_gate_missing_channel_raises():
    fmo_df = pd.DataFrame({"FL1-A": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError, match="channel 'FL2-A' not in fmo_df"):
        fmo_threshold_gate(fmo_df, "FL2-A")
