"""Tests for openflo.calibration counting-bead absolute-count helpers using
hand-computed numbers, plus the zero-bead guard."""
from __future__ import annotations

import pytest

from openflo.calibration import (
    absolute_count_from_known_beads,
    absolute_count_per_uL,
    total_cells,
)


def test_absolute_count_per_uL_handcomputed():
    # 5000 cells / 1000 beads = 5; * 1010 beads/µL = 5050 cells/µL.
    assert absolute_count_per_uL(5000, 1000, 1010.0) == pytest.approx(5050.0)


def test_absolute_count_per_uL_zero_beads_raises():
    with pytest.raises(ValueError, match="bead_events"):
        absolute_count_per_uL(5000, 0, 1010.0)


def test_absolute_count_from_known_beads_handcomputed():
    # (8000/2000) * 50000 beads / 200 µL = 4 * 50000 / 200 = 1000 cells/µL.
    got = absolute_count_from_known_beads(8000, 2000, 50000, 200.0)
    assert got == pytest.approx(1000.0)


def test_from_known_beads_zero_beads_raises():
    with pytest.raises(ValueError, match="bead_events"):
        absolute_count_from_known_beads(8000, 0, 50000, 200.0)


def test_from_known_beads_zero_volume_raises():
    with pytest.raises(ValueError, match="sample_volume_uL"):
        absolute_count_from_known_beads(8000, 2000, 50000, 0.0)


def test_total_cells_handcomputed():
    # (8000/2000) * 50000 = 4 * 50000 = 200000 cells.
    assert total_cells(8000, 2000, 50000) == pytest.approx(200000.0)


def test_total_cells_zero_beads_raises():
    with pytest.raises(ValueError, match="bead_events"):
        total_cells(8000, 0, 50000)
