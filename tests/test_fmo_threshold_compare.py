"""FMO gating → Compare thresholds: each stained sample's % positive with the
cutoff at several FMO percentiles, without adding a gate.

Generated data: an FMO whose channel values are 1..1000 and a stained sample
with 1..2000, both linear, so every cutoff and count has a closed form
(np.percentile of the FMO, then the stained events at or above it)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from openflo.async_task import wait_for_job
from openflo.fmo_sensitivity import (
    COLUMNS,
    parse_percentiles,
    spread,
    threshold_rows,
)

FMO = np.arange(1, 1001, dtype=float)
STAINED = np.arange(1, 2001, dtype=float)


def test_rows_match_closed_form():
    rows = threshold_rows({'FL1-A': ('fmo1', FMO)},
                          {'s1': {'FL1-A': STAINED}}, [99.5, 99.0])
    assert [r['FMO % above (set)'] for r in rows] == [1.0, 0.5]
    for r, p in zip(rows, (99.0, 99.5), strict=True):
        cut = float(np.percentile(FMO, p))
        assert r['Threshold'] == pytest.approx(cut)
        assert r['Positive'] == int(np.sum(STAINED >= cut))
        assert r['% positive'] == pytest.approx(
            100.0 * np.sum(STAINED >= cut) / STAINED.size)
        assert r['FMO % above (measured)'] == pytest.approx(
            100.0 * np.mean(FMO >= cut))
        assert set(r) == set(COLUMNS)
    # 99th: cut 990.01, so 1010 of 2000 are positive; 99.5th: 995.005, 1005.
    assert [r['Positive'] for r in rows] == [1010, 1005]
    assert spread(rows) == {('s1', 'FL1-A'): pytest.approx(0.25)}


def test_events_on_the_cut_are_positive():
    # A flat FMO puts the cut exactly on 5.0; a threshold gate keeps x >= 5.
    rows = threshold_rows({'FL1-A': ('fmo1', np.full(50, 5.0))},
                          {'s1': {'FL1-A': np.array([4.0, 5.0, 5.0, 6.0])}},
                          [99.0])
    assert (rows[0]['Threshold'], rows[0]['Positive']) == (5.0, 3)
    assert rows[0]['FMO % above (measured)'] == 100.0


def test_empty_sample_reads_na_and_missing_channel_gives_no_rows():
    rows = threshold_rows({'FL1-A': ('fmo1', FMO)},
                          {'empty': {'FL1-A': np.array([np.nan])},
                           'other': {'FL2-A': STAINED}}, [99.0])
    assert len(rows) == 1
    assert rows[0]['Sample'] == 'empty'
    assert math.isnan(rows[0]['% positive'])


def test_parse_percentiles():
    assert parse_percentiles('99.5, 99 99;99') == [99.0, 99.5]
    for bad in ('', '101', '-1', 'x'):
        with pytest.raises(ValueError):
            parse_percentiles(bad)


def test_dialog_compares_every_stained_sample_and_adds_no_gate():
    from tests.test_fmo_gating_dialog import _editor_or_skip, _register, _setup
    root, ed = _editor_or_skip()
    try:
        fmo, stained = _setup(ed)
        rng = np.random.default_rng(9)
        other = stained.copy()
        other['CD3'] = rng.normal(0.4, 0.1, len(other))
        _register(ed, 'stained2', other)
        from openflo.ui_fmo import FMOGatingDialog
        from openflo.ui_table import TableWindow
        dlg = FMOGatingDialog(ed)
        dlg._map['CD3'].set('fmo_cd3')
        dlg._cmp_pcts.set('99, 99.5')
        dlg._compare()
        assert wait_for_job(dlg._busy)
        root.update()
        wins = [w for w in ed.winfo_children() if isinstance(w, TableWindow)]
        assert len(wins) == 1
        rows = wins[0]._rows
        assert {r['Sample'] for r in rows} == {'stained', 'stained2'}
        for r in rows:
            df = stained if r['Sample'] == 'stained' else other
            cut = float(np.percentile(fmo['CD3'], 100 - r['FMO % above (set)']))
            assert r['% positive'] == pytest.approx(
                100.0 * np.mean(df['CD3'].to_numpy() >= cut))
        assert not ed._gates
        assert wins[0].table_lines()[0] == list(COLUMNS)
    finally:
        root.destroy()


def test_apply_remembers_the_fmo_for_histograms():
    from tests.test_fmo_gating_dialog import _editor_or_skip, _setup
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        from openflo.ui_fmo import FMOGatingDialog
        dlg = FMOGatingDialog(ed)
        dlg._map['CD3'].set('fmo_cd3')
        dlg._apply()
        assert wait_for_job(dlg._busy)
        assert ed._fmo_for_channel('CD3') == 'fmo_cd3'
        assert ed._fmo_for_channel('CD4') is None
        # The next dialog starts from it.
        assert FMOGatingDialog(ed)._map['CD3'].get() == 'fmo_cd3'
    finally:
        root.destroy()
