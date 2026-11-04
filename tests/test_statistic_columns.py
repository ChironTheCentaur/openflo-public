"""More statistics (complaint 16): % of grandparent, SD, rSD and MAD,
percentiles and the MFI ratio against a reference sample, in the
statistics table and the Statistics window.

Every number is checked against a value worked out here from generated
data (synthetic sample names, generic FL channels), and each check has a
negative partner: a robust spread that must NOT move under outliers, a
reference with no usable value (blank).
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

import tests._inline_editor as shared
from openflo.gating import (
    CHANNEL_STAT_ORDER,
    MFI_RATIO,
    add_mfi_ratio,
    parse_percentiles,
    population_stats,
)
from tests._inline_editor import _register, _sample

editor = shared.editor


CH = ['FL1-A']


def _gate(parent, value, name):
    return {'kind': 'threshold', 'channel': 'FL1-A', 'op': '>',
            'value': value, 'parent_id': parent, 'name': name,
            'enabled': True}


def _stats(vals, want, gates=None, order=None, name='synthetic_1'):
    df = pd.DataFrame({'FL1-A': np.asarray(vals, dtype=float)})
    gates = gates or {'g1': _gate(None, -np.inf, 'All')}
    order = order or list(gates)
    return population_stats(name, df, gates, order, {}, CH, set(want),
                            CHANNEL_STAT_ORDER)


def _tree():
    # 0..999: 900 events at 100 or over, 500 at 500, 100 at 900.
    return ({'g1': _gate(None, 100, 'A'), 'g2': _gate('g1', 500, 'B'),
             'g3': _gate('g2', 900, 'C')}, ['g1', 'g2', 'g3'])


def test_grandparent_percent_counts_against_the_parents_parent():
    gates, order = _tree()
    rows = {r['__gid__']: r for r in _stats(
        np.arange(1000), {'Count', '%Parent', '%Grandparent'}, gates, order)}
    # A gate on all events has no grandparent: blank, not a number.
    assert np.isnan(rows['g1']['%Grandparent'])
    # B's grandparent is all events; C's is A.
    assert rows['g2']['%Grandparent'] == pytest.approx(500 / 1000 * 100)
    assert rows['g3']['%Grandparent'] == pytest.approx(100 / 900 * 100)
    # And %Parent is unchanged beside it.
    assert rows['g3']['%Parent'] == pytest.approx(100 / 500 * 100)


def test_grandparent_is_absent_unless_asked():
    gates, order = _tree()
    rows = _stats(np.arange(1000), {'Count', '%Parent'}, gates, order)
    assert all('%Grandparent' not in r for r in rows)


def test_sd_rsd_and_mad_match_their_definitions():
    rng = np.random.default_rng(1)
    vals = rng.normal(1000.0, 100.0, 200_000)
    r = _stats(vals, {'Mean', 'SD', 'rSD', 'CV', 'rCV', 'MAD',
                      'Median'})[0]
    p_lo, p_hi = np.percentile(vals, [15.87, 84.13])
    med = np.median(vals)
    assert r['SD FL1-A'] == pytest.approx(np.std(vals, ddof=1), rel=1e-12)
    mad = np.median(np.abs(vals - med))
    # The default rSD is the MAD form; FlowJo's is half the 15.87-84.13 range.
    assert r['rSD FL1-A'] == pytest.approx(
        mad / NormalDist().inv_cdf(0.75), rel=1e-12)
    flowjo = population_stats('synthetic_1', pd.DataFrame({'FL1-A': vals}),
                              {'g1': _gate(None, -np.inf, 'All')}, ['g1'],
                              {}, CH, {'rSD'}, CHANNEL_STAT_ORDER,
                              stat_forms={'rcv': 'percentile'})[0]
    assert flowjo['rSD FL1-A'] == pytest.approx(0.5 * (p_hi - p_lo),
                                                rel=1e-12)
    assert r['MAD FL1-A'] == pytest.approx(mad, rel=1e-12)
    # CV and rCV are these spreads over the centre.
    assert r['CV FL1-A'] == pytest.approx(
        r['SD FL1-A'] / r['Mean FL1-A'] * 100, rel=1e-12)
    assert r['rCV FL1-A'] == pytest.approx(
        r['rSD FL1-A'] / r['Median FL1-A'] * 100, rel=1e-12)
    # For a normal population all three estimate sigma = 100.
    assert r['SD FL1-A'] == pytest.approx(100, rel=0.02)
    # MAD-based rSD of 200,000 events: SE ~0.3%; seeds read within 0.5%.
    assert r['rSD FL1-A'] == pytest.approx(100, rel=0.02)
    # Same MAD estimate scaled to sigma: SE ~0.3%.
    assert r['MAD FL1-A'] * 1.4826 == pytest.approx(100, rel=0.02)


def test_robust_spreads_ignore_outliers_that_inflate_the_sd():
    rng = np.random.default_rng(2)
    clean = rng.normal(1000.0, 100.0, 100_000)
    dirty = np.concatenate([clean, np.full(1_000, 1e6)])     # 1 % off scale
    a = _stats(clean, {'SD', 'rSD', 'MAD'})[0]
    b = _stats(dirty, {'SD', 'rSD', 'MAD'})[0]
    assert b['SD FL1-A'] > 50 * a['SD FL1-A']
    # 1% of events off scale shift the median and MAD by ~1.1-1.3% (a fixed
    # bias, seeds 2, 5, 6); 5% leaves room while the SD moves 50-fold.
    assert b['rSD FL1-A'] == pytest.approx(a['rSD FL1-A'], rel=0.05)
    # Same ~1.2% bias from the 1% off-scale events.
    assert b['MAD FL1-A'] == pytest.approx(a['MAD FL1-A'], rel=0.05)


def test_percentiles_give_one_column_each():
    vals = np.arange(1, 1001, dtype=float)
    r = _stats(vals, {'Median', 'P5', 'P50', 'P97.5'})[0]
    for name, q in (('P5', 5), ('P50', 50), ('P97.5', 97.5)):
        assert r[f'{name} FL1-A'] == pytest.approx(np.percentile(vals, q))
    assert r['P50 FL1-A'] == r['Median FL1-A']
    assert not any(k.startswith('P25') for k in r)


def test_percentile_text_is_parsed_and_checked():
    assert parse_percentiles('5, 25,75 95') == ['P5', 'P25', 'P75', 'P95']
    assert parse_percentiles('2.5;97.5%, 2.5') == ['P2.5', 'P97.5']
    assert parse_percentiles('  ') == []
    for bad in ('101', '-1', 'five'):
        with pytest.raises(ValueError):
            parse_percentiles(bad)


def _two_samples(want):
    gates = {'g1': _gate(None, -np.inf, 'All'), 'g2': _gate('g1', 0, 'Pos')}
    ctrl = np.linspace(90.0, 110.0, 201)              # median 100
    rows = _stats(ctrl, want, gates, name='control_1')
    rows += _stats(ctrl * 8.0, want, gates, name='stained_1')
    return rows


def test_mfi_ratio_divides_by_the_same_population_in_the_reference():
    rows = add_mfi_ratio(_two_samples({'Median', MFI_RATIO}), 'control_1')
    by = {(r['Sample'], r['Population']): r for r in rows}
    assert by[('stained_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(8.0)
    assert by[('control_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(1.0)
    # A different reference gives a different ratio.
    rows = add_mfi_ratio(_two_samples({MFI_RATIO}), 'stained_1')
    by = {(r['Sample'], r['Population']): r for r in rows}
    assert by[('control_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(
        1 / 8.0)


def test_mfi_ratio_shows_without_the_median_column():
    rows = add_mfi_ratio(_two_samples({MFI_RATIO}), 'control_1')
    assert all('Median FL1-A' not in r for r in rows)
    assert all('MFI ratio FL1-A' in r for r in rows)


def test_mfi_ratio_is_blank_without_a_usable_reference():
    rows = _two_samples({MFI_RATIO})
    # Reference rows taken from elsewhere (a curated table): only 'All'.
    ref = [r for r in rows if r['Sample'] == 'control_1'
           and r['Population'] == 'All']
    stained = [r for r in rows if r['Sample'] == 'stained_1']
    add_mfi_ratio(stained, 'control_1', reference_rows=ref)
    by = {r['__gid__']: r for r in stained}
    assert by['g1']['MFI ratio FL1-A'] == pytest.approx(8.0)
    assert np.isnan(by['g2']['MFI ratio FL1-A'])       # reference lacks it
    # A reference median at or below zero has no ratio.
    gates = {'g1': _gate(None, -np.inf, 'All')}
    neg = _stats(np.full(11, -5.0), {MFI_RATIO}, gates, name='control_1')
    pos = _stats(np.full(11, 50.0), {MFI_RATIO}, gates, name='stained_1')
    add_mfi_ratio(neg + pos, 'control_1')
    assert np.isnan(pos[0]['MFI ratio FL1-A'])
    # No reference sample loaded at all: blank, never 1 or 0.
    rows = add_mfi_ratio(_two_samples({MFI_RATIO}), 'missing_1')
    assert all(np.isnan(r['MFI ratio FL1-A']) for r in rows)


def test_statistics_window_offers_the_new_columns_and_copies(editor):
    from openflo.ui_statistics import StatisticsWindow
    gates = {'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                    'parent_id': None, 'name': 'All', 'enabled': True},
             'g2': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 5.0,
                    'parent_id': 'g1', 'name': 'Upper', 'enabled': True}}
    ctrl = np.linspace(90.0, 110.0, 21)
    _register(editor, _sample('control_1', ctrl), gates)
    _register(editor, _sample('stained_1', ctrl * 4), gates)
    w = StatisticsWindow(editor)
    w.withdraw()
    for st in ('%Grandparent', 'SD', 'rSD', 'MAD'):
        w._stat_vars[st].set(True)
    w._pct_on.set(True)
    w._pct_text.set('10, 90')
    w._ratio_on.set(True)
    w._sync_ratio_choices()
    assert w._ratio_ref.get() == 'control_1'           # first loaded
    w._refresh()
    for col in ('%Grandparent', 'SD FL1-A', 'rSD FL1-A', 'MAD FL1-A',
                'P10 FL1-A', 'P90 FL1-A', 'MFI ratio FL1-A'):
        assert col in w._cols, col
    assert w._heading('P10 FL1-A') == 'P10 FL1-A (linear)'
    assert w._heading('MFI ratio FL1-A') == 'MFI ratio FL1-A (vs control_1)'
    by = {(r['Sample'], r['Population']): r for r in w._rows}
    assert by[('stained_1', 'All')]['MFI ratio FL1-A'] == pytest.approx(4.0)
    assert by[('stained_1', 'All')]['P90 FL1-A'] == pytest.approx(
        np.percentile(ctrl * 4, 90))
    assert 'MFI ratio = median' in w.status_var.get()

    w._copy_table()
    lines = w.clipboard_get().splitlines()
    assert lines[0].split('\t')[:2] == ['Sample', 'Population']
    assert 'MFI ratio FL1-A (vs control_1)' in lines[0].split('\t')
    assert len(lines) == 1 + len(w._rows)

    # A curated table without the reference still gets its ratio.
    w._set_targets([('stained_1', 'g1')], 'editor')
    assert [r['Sample'] for r in w._rows] == ['stained_1']
    assert w._rows[0]['MFI ratio FL1-A'] == pytest.approx(4.0)

    # A typo in the percentiles is said, and adds no column.
    w._pct_text.set('10, ninety')
    w._refresh()
    assert not any(c.startswith('P10 ') for c in w._cols)
    assert 'Percentiles' in w.status_var.get()
    w.destroy()
