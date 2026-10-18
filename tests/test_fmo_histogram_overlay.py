"""View → FMO on histograms: a histogram of a channel with a known FMO
control draws that FMO in grey behind the sample, and the legend gives each
one's % at or above the channel's threshold gate.

Generated data (tests/test_fmo_gating_dialog._data): an FMO and a stained
sample on CD3 whose % above any cut is counted directly from the frames."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_fmo_gating_dialog import _editor_or_skip, _setup

CUT = 0.45


def _render(ed):
    from matplotlib.figure import Figure
    ax = Figure().add_subplot()
    ed._render_into(ax, ['stained'], 'CD3', None, 'histogram', None,
                    population=None)
    return {ln.get_label(): ln for ln in ax.get_lines()
            if not ln.get_label().startswith('_')}


def _gate(ed):
    ed._add_gate({'kind': 'threshold', 'channel': 'CD3', 'value': CUT,
                  'name': 'CD3+', 'parent_id': None, 'enabled': True,
                  'color': '#d62728'})


def test_fmo_drawn_with_percent_above_the_gate():
    root, ed = _editor_or_skip()
    try:
        fmo, stained = _setup(ed)
        _gate(ed)
        ed._set_fmo_controls({'CD3': 'fmo_cd3'})
        lines = _render(ed)
        pct_s = 100.0 * np.mean(stained['CD3'].to_numpy() >= CUT)
        pct_f = 100.0 * np.mean(fmo['CD3'].to_numpy() >= CUT)
        assert set(lines) == {f'stained; ≥ CD3+: {pct_s:.1f}%',
                              f'FMO: fmo_cd3; ≥ CD3+: {pct_f:.1f}%'}
        assert lines[f'FMO: fmo_cd3; ≥ CD3+: {pct_f:.1f}%'] \
            .get_linestyle() == '--'
        assert pct_s == pytest.approx(50.0, abs=1.5)    # the data's design
    finally:
        root.destroy()


def test_no_fmo_when_switched_off_or_unknown():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        _gate(ed)
        assert set(_render(ed)) == {'stained'}          # no FMO known
        ed._set_fmo_controls({'CD3': 'fmo_cd3'})
        ed._fmo_hist_var.set(False)
        assert set(_render(ed)) == {'stained'}
    finally:
        root.destroy()


def test_fmo_assignment_is_saved_on_the_fmo_sample():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed._set_fmo_controls({'CD3': 'fmo_cd3', 'CD4': 'fmo_cd3'})
        entries = {e['name']: e for e in ed._session_state()['samples']}
        assert entries['fmo_cd3']['fmo_of'] == ['CD3', 'CD4']
        assert 'fmo_of' not in entries['stained']
    finally:
        root.destroy()


def test_percent_above_counts_events_on_the_cut():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        assert ed._hist_above_text('stained', 'CD3', arr, [('g', 2.0)]) \
            == '; ≥ g: 75.0%'
        assert ed._hist_above_text('stained', 'CD3', arr[:0], [('g', 2.0)]) \
            == '; ≥ g: NA'
    finally:
        root.destroy()
