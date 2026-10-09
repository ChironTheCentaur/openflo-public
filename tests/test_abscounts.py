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


# ── the standard formula: bead volume, sample volume, dilution ───────────────

def _acquire(rng, cells_per_uL, sample_uL, beads_in_tube, dilution=1.0,
             fraction=0.2):
    """A tube made up and acquired: `sample_uL` of a sample that was diluted
    `dilution`-fold beforehand (so it holds cells_per_uL / dilution per µL),
    plus `beads_in_tube` beads; the cytometer reads a random `fraction` of
    the tube's contents. Returns (cell events, bead events)."""
    cells_in_tube = int(round(cells_per_uL / dilution * sample_uL))
    return (int(rng.binomial(cells_in_tube, fraction)),
            int(rng.binomial(beads_in_tube, fraction)))


def test_liquid_beads_count_against_the_sample_volume():
    """50 µL of beads at 1000/µL added to 100 µL of blood holding 1000
    cells/µL. The old formula, (cells / beads) x concentration, ignored both
    volumes and reported about 2000 cells/µL."""
    import numpy as np
    rng = np.random.default_rng(0)
    cells, beads = _acquire(rng, 1000.0, 100.0, 50 * 1000)
    got = absolute_count_per_uL(cells, beads, 1000.0, bead_volume_uL=50.0,
                                sample_volume_uL=100.0)
    assert got == pytest.approx(1000.0, rel=0.03)
    assert absolute_count_per_uL(cells, beads, 1000.0) == \
        pytest.approx(2000.0, rel=0.03)            # what it used to say
    # The audit's hand numbers: 10,000 cells, 5,000 beads -> 1000, not 2000.
    assert absolute_count_per_uL(10000, 5000, 1000, 50, 100) == \
        pytest.approx(1000.0)


def test_dilution_factor_reports_the_undiluted_sample():
    """Blood diluted 1:10 before 100 µL of it went into a Trucount tube of
    50,000 beads: the result is per µL of the blood, not of the dilution."""
    import numpy as np
    rng = np.random.default_rng(1)
    cells, beads = _acquire(rng, 5000.0, 100.0, 50_000, dilution=10.0)
    got = absolute_count_from_known_beads(cells, beads, 50_000, 100.0,
                                          dilution_factor=10.0)
    assert got == pytest.approx(5000.0, rel=0.03)
    liquid = absolute_count_per_uL(cells, beads, 1000.0, 50.0, 100.0,
                                   dilution_factor=10.0)
    assert liquid == pytest.approx(got)


def test_equal_volumes_keep_the_old_result():
    assert absolute_count_per_uL(5000, 1000, 1010.0, 100.0, 100.0) == \
        pytest.approx(absolute_count_per_uL(5000, 1000, 1010.0))


def test_one_volume_alone_is_refused():
    with pytest.raises(ValueError, match="both"):
        absolute_count_per_uL(5000, 1000, 1010.0, bead_volume_uL=50.0)
    with pytest.raises(ValueError, match="dilution_factor"):
        absolute_count_per_uL(5000, 1000, 1010.0, dilution_factor=0)
    with pytest.raises(ValueError, match="sample_volume_uL"):
        absolute_count_per_uL(5000, 1000, 1010.0, 50.0, -1.0)


def _dialog_or_skip():
    import os
    os.environ.setdefault('MPLBACKEND', 'Agg')
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001
        pytest.skip(str(e))
    root.withdraw()
    from openflo.ui_abscounts import AbsCountsDialog
    return root, AbsCountsDialog(root)


def test_dialog_uses_the_volumes_and_says_when_it_assumed_them():
    root, d = _dialog_or_skip()
    try:
        v = d._vars
        v["Cell events:"].set("10000")
        v["Bead events:"].set("5000")
        v["Bead concentration (beads/µL):"].set("1000")
        v["Bead volume added (µL):"].set("50")
        v["Sample volume (µL):"].set("100")
        d._compute()
        assert d._result.cget('text') == "= 1,000.0 cells/µL"
        v["Bead volume added (µL):"].set("")
        v["Sample volume (µL):"].set("")
        d._compute()
        assert d._result.cget('text') == "= 2,000.0 cells/µL"
        assert 'EQUAL' in d._detail.cget('text')
        # A pellet tube: beads per tube instead of concentration x volume.
        v["Bead concentration (beads/µL):"].set("")
        v["Beads per tube:"].set("50000")
        v["Sample volume (µL):"].set("100")
        v["Sample dilution factor:"].set("10")
        d._compute()
        assert d._result.cget('text') == "= 10,000.0 cells/µL"
        v["Bead concentration (beads/µL):"].set("1000")
        d._compute()
        assert 'not both' in d._result.cget('text')
    finally:
        root.destroy()
