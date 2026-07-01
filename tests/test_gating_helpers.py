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
    with pytest.raises(KeyError):
        singlet_gate(df, area="NOPE", height="FSC-H")


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
    fmo_df = pd.DataFrame({"CD3": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        fmo_threshold_gate(fmo_df, "CD4")
