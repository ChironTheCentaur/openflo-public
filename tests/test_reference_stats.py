"""ΔMFI and the Stain Index beside the MFI ratio in the Statistics window:
each row against a negative, either the same population in a control
sample (FMO or unstained) or a negative gate in the row's own sample.

Every number is worked out here from generated data (synthetic sample
names, generic FL channels) with numpy and the textbook formulas, not with
OpenFlo: ΔMFI = median - negative median, Stain Index = ΔMFI / (2 x rSD of
the negative) (Maecker et al., Cytometry A 2004). Each check has a
negative partner: a negative missing, without spread or too small.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

import tests._inline_editor as shared
from openflo.gating import (
    CHANNEL_STAT_ORDER,
    DELTA_MFI,
    MFI_RATIO,
    REFERENCE_STATS,
    STAIN_INDEX,
    THIN_NEGATIVE_EVENTS,
    add_reference_stats,
    population_stats,
)
from openflo.robust import stain_index
from tests._inline_editor import _register, _sample

editor = shared.editor

CH = ['FL1-A']
# The normal-consistency constant for the MAD, from its definition.
K = 1.0 / NormalDist().inv_cdf(0.75)
ALL = set(REFERENCE_STATS)


def _all():
    return {'kind': 'threshold', 'channel': 'FL1-A', 'value': -1e12,
            'parent_id': None, 'name': 'All', 'enabled': True}


def _band(lo, hi, name):
    return {'kind': 'interval', 'channel': 'FL1-A', 'lo': lo, 'hi': hi,
            'parent_id': 'g1', 'name': name, 'enabled': True}


def _stats(vals, want, gates, name, forms=None):
    df = pd.DataFrame({'FL1-A': np.asarray(vals, dtype=float)})
    return population_stats(name, df, gates, list(gates), {}, CH, set(want),
                            CHANNEL_STAT_ORDER, stat_forms=forms)


def _mad_rsd(a):
    a = np.asarray(a, dtype=float)
    return K * float(np.median(np.abs(a - np.median(a))))


def _pct_rsd(a):
    lo, hi = np.percentile(np.asarray(a, dtype=float), (15.87, 84.13))
    return 0.5 * float(hi - lo)


def _by(rows):
    """Rows by (sample, gate name): the last step of the population path."""
    return {(r['Sample'], r['Population'].rsplit('/', 1)[-1]): r
            for r in rows}


# ── against the same population in a control sample ─────────────────────────

CTRL = np.linspace(90.0, 110.0, 201)           # median 100, MAD 5
STAINED = np.concatenate([CTRL + 300.0, np.full(20, 5000.0)])


def _control_and_stained(want, forms=None):
    gates = {'g1': _all()}
    return (_stats(CTRL, want, gates, 'control_1', forms)
            + _stats(STAINED, want, gates, 'stained_1', forms))


def test_delta_mfi_and_stain_index_against_the_control_sample():
    rows = add_reference_stats(_control_and_stained(ALL), ALL,
                               reference='control_1')
    r = _by(rows)[('stained_1', 'All')]
    d = float(np.median(STAINED)) - float(np.median(CTRL))
    assert r['ΔMFI FL1-A'] == pytest.approx(d)
    assert r['Stain index FL1-A'] == pytest.approx(d / (2 * _mad_rsd(CTRL)))
    assert r['MFI ratio FL1-A'] == pytest.approx(
        np.median(STAINED) / np.median(CTRL))
    # The control against itself: no separation.
    c = _by(rows)[('control_1', 'All')]
    assert c['ΔMFI FL1-A'] == 0.0
    assert c['Stain index FL1-A'] == 0.0


def test_stain_index_uses_the_rsd_form_chosen_in_preferences():
    # FlowJo's percentile rSD differs from 1.4826 x MAD on this uniform
    # control, so the Stain Index must follow the form asked for.
    assert _pct_rsd(CTRL) != pytest.approx(_mad_rsd(CTRL), rel=0.05)
    rows = add_reference_stats(
        _control_and_stained({STAIN_INDEX},
                             {'rcv': 'percentile', 'geomean': 'display'}),
        {STAIN_INDEX}, reference='control_1')
    r = _by(rows)[('stained_1', 'All')]
    d = float(np.median(STAINED)) - float(np.median(CTRL))
    assert r['Stain index FL1-A'] == pytest.approx(d / (2 * _pct_rsd(CTRL)))


def test_only_the_statistics_asked_for_get_columns():
    for one in REFERENCE_STATS:
        rows = add_reference_stats(_control_and_stained({one}), {one},
                                   reference='control_1')
        keys = {k for r in rows for k in r if not k.startswith('__')}
        assert f'{one} FL1-A' in keys
        others = [f'{s} FL1-A' for s in REFERENCE_STATS if s != one]
        assert not keys & {*others, 'Median FL1-A', 'rSD FL1-A'}, one


def test_a_missing_or_unusable_negative_leaves_cells_blank():
    gates = {'g1': _all(), 'g2': _band(200, 1e12, 'Bright')}
    ctrl = _stats(CTRL, ALL, {'g1': _all()}, 'control_1')
    stained = _stats(STAINED, ALL, gates, 'stained_1')
    add_reference_stats(ctrl + stained, ALL, reference='control_1')
    bright = _by(stained)[('stained_1', 'Bright')]
    # The control has no 'Bright' population: blank, never 0 or 1.
    for stat in REFERENCE_STATS:
        assert np.isnan(bright[f'{stat} FL1-A']), stat
    assert not np.isnan(_by(stained)[('stained_1', 'All')]['ΔMFI FL1-A'])
    # A negative with no spread: the Stain Index is blank, not infinite;
    # ΔMFI is still measured. A negative median at or below 0 has no
    # ratio, but its ΔMFI and Stain Index stand.
    flat = _stats(np.full(150, 40.0), ALL, {'g1': _all()}, 'control_1')
    pos = _stats(STAINED, ALL, {'g1': _all()}, 'stained_1')
    add_reference_stats(flat + pos, ALL, reference='control_1')
    assert np.isnan(pos[0]['Stain index FL1-A'])
    assert pos[0]['ΔMFI FL1-A'] == pytest.approx(np.median(STAINED) - 40.0)
    below = _stats(CTRL - 120.0, ALL, {'g1': _all()}, 'control_1')
    pos = _stats(STAINED, ALL, {'g1': _all()}, 'stained_1')
    add_reference_stats(below + pos, ALL, reference='control_1')
    assert np.isnan(pos[0]['MFI ratio FL1-A'])
    assert pos[0]['ΔMFI FL1-A'] == pytest.approx(
        np.median(STAINED) - np.median(CTRL - 120.0))
    assert pos[0]['Stain index FL1-A'] > 0


def test_a_population_dimmer_than_its_negative_reads_below_zero():
    gates = {'g1': _all()}
    rows = (_stats(CTRL, ALL, gates, 'control_1')
            + _stats(CTRL - 30.0, ALL, gates, 'dim_1'))
    add_reference_stats(rows, ALL, reference='control_1')
    r = _by(rows)[('dim_1', 'All')]
    assert r['ΔMFI FL1-A'] == pytest.approx(-30.0)
    assert r['Stain index FL1-A'] == pytest.approx(-30.0 / (2 * _mad_rsd(CTRL)))


# ── against a negative gate in each sample ──────────────────────────────────

def _titration(neg_shift):
    neg = np.linspace(0.0, 20.0, 201) + neg_shift
    pos = np.linspace(400.0, 440.0, 101)
    return neg, pos, np.concatenate([neg, pos])


def test_each_row_is_compared_with_the_negative_gate_of_its_own_sample():
    gates = {'g1': _all(), 'g2': _band(-1e12, 100, 'Neg'),
             'g3': _band(100, 1e12, 'Pos')}
    rows, expect = [], {}
    for name, shift in (('titr_1', 0.0), ('titr_2', 40.0)):
        neg, pos, vals = _titration(shift)
        rows += _stats(vals, ALL, gates, name)
        d = float(np.median(pos)) - float(np.median(neg))
        expect[name] = (d, d / (2 * _mad_rsd(neg)))
    neg_key = _by(rows)[('titr_1', 'Neg')]['__key__']
    add_reference_stats(rows, ALL, negative=neg_key)
    by = _by(rows)
    for name, (d, si) in expect.items():
        assert by[(name, 'Pos')]['ΔMFI FL1-A'] == pytest.approx(d)
        assert by[(name, 'Pos')]['Stain index FL1-A'] == pytest.approx(si)
    # Each sample's own negative: the two differ only in it.
    assert expect['titr_1'][0] != pytest.approx(expect['titr_2'][0])


def test_a_sample_without_the_negative_gate_has_blank_cells():
    gates = {'g1': _all(), 'g2': _band(-1e12, 100, 'Neg'),
             'g3': _band(100, 1e12, 'Pos')}
    _neg, _pos, vals = _titration(0.0)
    rows = _stats(vals, ALL, gates, 'titr_1')
    other = _stats(vals, ALL, {'g1': _all(), 'g3': gates['g3']}, 'titr_2')
    neg_key = _by(rows)[('titr_1', 'Neg')]['__key__']
    add_reference_stats(rows + other, ALL, negative=neg_key)
    assert not np.isnan(_by(rows)[('titr_1', 'Pos')]['ΔMFI FL1-A'])
    for stat in REFERENCE_STATS:
        assert np.isnan(_by(other)[('titr_2', 'Pos')][f'{stat} FL1-A'])


def test_negatives_left_out_of_the_table_come_from_the_reference_rows():
    gates = {'g1': _all(), 'g2': _band(-1e12, 100, 'Neg'),
             'g3': _band(100, 1e12, 'Pos')}
    neg, pos, vals = _titration(0.0)
    rows = _stats(vals, ALL, gates, 'titr_1')
    neg_row = [r for r in rows if r['Population'] == 'All/Neg']
    shown = [r for r in rows if r['Population'] == 'All/Pos']
    add_reference_stats(shown, ALL, negative=neg_row[0]['__key__'],
                        reference_rows=neg_row)
    assert shown[0]['ΔMFI FL1-A'] == pytest.approx(
        np.median(pos) - np.median(neg))


def test_a_stain_index_on_a_small_negative_is_flagged():
    for n, thin in ((THIN_NEGATIVE_EVENTS - 1, True),
                    (THIN_NEGATIVE_EVENTS, False)):
        ctrl = _stats(np.linspace(90.0, 110.0, n), ALL, {'g1': _all()},
                      'control_1')
        pos = _stats(STAINED, ALL, {'g1': _all()}, 'stained_1')
        add_reference_stats(ctrl + pos, ALL, reference='control_1')
        assert bool(pos[0].get('__thin_negative__')) is thin, n
    # A negative whose size was not kept is not vouched for.
    ctrl = _stats(CTRL, ALL, {'g1': _all()}, 'control_1')
    del ctrl[0]['__events__']
    pos = _stats(STAINED, ALL, {'g1': _all()}, 'stained_1')
    add_reference_stats(ctrl + pos, ALL, reference='control_1')
    assert pos[0].get('__thin_negative__')
    # The negative's own size counts, not its sample's: a small negative
    # gate inside a large sample is flagged.
    gates = {'g1': _all(), 'g2': _band(-1e12, 100, 'Neg'),
             'g3': _band(100, 1e12, 'Pos')}
    vals = np.concatenate([np.linspace(0.0, 20.0, 50),
                           np.linspace(400.0, 440.0, 950)])
    rows = _stats(vals, ALL, gates, 'titr_1')
    add_reference_stats(rows, ALL, negative=_by(rows)[('titr_1', 'Neg')]
                        ['__key__'])
    assert _by(rows)[('titr_1', 'Pos')].get('__thin_negative__')
    # Only the Stain Index rests on the negative's spread.
    ctrl = _stats(np.linspace(90.0, 110.0, 10), {DELTA_MFI}, {'g1': _all()},
                  'control_1')
    pos = _stats(STAINED, {DELTA_MFI}, {'g1': _all()}, 'stained_1')
    add_reference_stats(ctrl + pos, {DELTA_MFI}, reference='control_1')
    assert not pos[0].get('__thin_negative__')


def test_stain_index_formula_and_guards():
    assert stain_index(410.0, 10.0, 5.0) == pytest.approx(40.0)
    assert stain_index(0.0, 10.0, 5.0) == pytest.approx(-1.0)
    for bad in ((410.0, 10.0, 0.0), (410.0, 10.0, -1.0),
                (float('nan'), 10.0, 5.0), (410.0, float('inf'), 5.0),
                (410.0, 10.0, float('nan'))):
        assert np.isnan(stain_index(*bad)), bad


# ── the Statistics window ───────────────────────────────────────────────────

def test_statistics_window_offers_delta_mfi_and_stain_index(editor):
    from openflo.ui_statistics import NEG_GATE, StatisticsWindow
    gates = {'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                    'parent_id': None, 'name': 'All', 'enabled': True},
             'g2': {'kind': 'interval', 'channel': 'FL1-A', 'lo': -1e12,
                    'hi': 100.0, 'parent_id': 'g1', 'name': 'Neg',
                    'enabled': True},
             'g3': {'kind': 'interval', 'channel': 'FL1-A', 'lo': 100.0,
                    'hi': 1e12, 'parent_id': 'g1', 'name': 'Pos',
                    'enabled': True}}
    neg, pos, vals = _titration(0.0)
    _register(editor, _sample('control_1', neg), gates)
    _register(editor, _sample('stained_1', vals), gates)
    w = StatisticsWindow(editor)
    w.withdraw()
    w._delta_on.set(True)
    w._si_on.set(True)
    w._refresh()
    assert {'ΔMFI FL1-A', 'Stain index FL1-A'} <= set(w._cols)
    assert MFI_RATIO + ' FL1-A' not in w._cols
    assert w._heading('ΔMFI FL1-A') == 'ΔMFI FL1-A (vs control_1)'
    by = _by(w._rows)
    # Same population ('All') in the control: the control is all negative.
    d = float(np.median(vals)) - float(np.median(neg))
    assert by[('stained_1', 'All')]['ΔMFI FL1-A'] == pytest.approx(d)
    assert by[('stained_1', 'All')]['Stain index FL1-A'] == pytest.approx(
        d / (2 * _mad_rsd(neg)))
    assert 'Negative = the same population in control_1' in (
        w.status_var.get())

    assert w._ratio_combo.winfo_manager() == 'pack'
    assert w._neg_gate_combo.winfo_manager() == ''

    # A negative gate in each sample instead. None is picked for the user
    # (the first gate is a parent, never a negative): the columns stay out
    # and the status line asks for one.
    w._neg_mode.set(NEG_GATE)
    w._on_neg_mode()
    assert w._neg_gate.get() == ''
    assert w._ratio_combo.winfo_manager() == ''
    assert w._neg_gate_combo.winfo_manager() == 'pack'
    assert not any(c.startswith(('ΔMFI ', 'Stain index ')) for c in w._cols)
    assert 'Choose the negative gate' in w.status_var.get()
    # Each population once, though both samples have it.
    assert list(w._neg_gates) == ['All', 'All/Neg', 'All/Pos']
    w._neg_gate.set('All/Neg')
    w._refresh()
    w._refresh()                     # the pick stays across refreshes
    assert w._neg_gate.get() == 'All/Neg'
    assert 'Choose the negative gate' not in w.status_var.get()
    by = _by(w._rows)
    si = by[('stained_1', 'Pos')]['Stain index FL1-A']
    assert si == pytest.approx((np.median(pos) - np.median(neg))
                               / (2 * _mad_rsd(neg)))
    assert w._heading('Stain index FL1-A') == (
        'Stain index FL1-A (vs All/Neg in each sample)')
    assert "All/Neg in each row's own sample" in w.status_var.get()

    # A curated table without the negative gate still gets its values.
    w._set_targets([('stained_1', 'g3')], 'editor')
    assert [r['Population'] for r in w._rows] == ['All/Pos']
    assert w._rows[0]['Stain index FL1-A'] == pytest.approx(si)

    # Copied with the negative in the heading.
    w._copy_table()
    head = w.clipboard_get().splitlines()[0].split('\t')
    assert 'ΔMFI FL1-A (vs All/Neg in each sample)' in head
    w.destroy()


def test_statistics_window_warns_on_a_small_negative(editor):
    from openflo.ui_statistics import StatisticsWindow
    gates = {'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                    'parent_id': None, 'name': 'All', 'enabled': True}}
    _register(editor, _sample('control_1', np.linspace(90, 110, 40)), gates)
    _register(editor, _sample('stained_1', STAINED), gates)
    w = StatisticsWindow(editor)
    w.withdraw()
    w._delta_on.set(True)
    w._refresh()
    assert 'uncertain' not in w.status_var.get()      # ΔMFI needs no spread
    w._si_on.set(True)
    w._refresh()
    assert 'under 100 events' in w.status_var.get()
    w.destroy()


def test_stain_index_heading_names_a_non_default_rsd_form(editor, monkeypatch):
    from openflo.ui_statistics import StatisticsWindow
    gates = {'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                    'parent_id': None, 'name': 'All', 'enabled': True}}
    _register(editor, _sample('control_1', CTRL), gates)
    _register(editor, _sample('stained_1', STAINED), gates)
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, 'load_stat_forms', lambda: {
        'rcv': 'percentile', 'geomean': 'display'})
    w = StatisticsWindow(editor)
    w.withdraw()
    w._si_on.set(True)
    w._delta_on.set(True)
    w._refresh()
    assert w._heading('Stain index FL1-A') == (
        'Stain index FL1-A (vs control_1, rSD FlowJo percentile form)')
    assert w._heading('ΔMFI FL1-A') == 'ΔMFI FL1-A (vs control_1)'
    w.destroy()


def test_every_reference_statistic_has_a_status_formula_and_a_tip():
    from openflo.ui_statistics import _REFERENCE_FORMULAS, _STAT_HELP
    assert set(_REFERENCE_FORMULAS) == set(REFERENCE_STATS)
    assert set(REFERENCE_STATS) <= set(_STAT_HELP)


def test_a_negative_that_cannot_be_computed_is_not_used():
    # Sample mode: the control's population reads NA (its gate's channel is
    # missing there), so the stained rows have no negative.
    gates = {'g1': _all()}
    ctrl = _stats(CTRL, ALL, gates, 'control_1')
    ctrl[0]['__unknown__'] = 'channel missing'
    pos = _stats(STAINED, ALL, gates, 'stained_1')
    add_reference_stats(ctrl + pos, ALL, reference='control_1')
    for stat in REFERENCE_STATS:
        assert np.isnan(pos[0][f'{stat} FL1-A']), stat
    # Gate mode: the same for a negative gate that reads NA.
    gates = {'g1': _all(), 'g2': _band(-1e12, 100, 'Neg'),
             'g3': _band(100, 1e12, 'Pos')}
    _neg, _pos, vals = _titration(0.0)
    rows = _stats(vals, ALL, gates, 'titr_1')
    neg_row = _by(rows)[('titr_1', 'Neg')]
    neg_row['__unknown__'] = 'channel missing'
    add_reference_stats(rows, ALL, negative=neg_row['__key__'])
    for stat in REFERENCE_STATS:
        assert np.isnan(_by(rows)[('titr_1', 'Pos')][f'{stat} FL1-A']), stat


def test_negative_gate_chooser_keys_and_labels(editor):
    """The chooser keys each gate as the statistics rows do (with the
    sample's own channel labels), skips an order entry with no gate, and
    keeps two populations whose paths read alike apart."""
    from openflo.ui_statistics import NEG_GATE, StatisticsWindow
    # Unnamed gates: their identity is the marker, so the sample's own
    # label for FL1-A is part of the key.
    gates = {'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                    'parent_id': None, 'name': 'All', 'enabled': True},
             'g2': {'kind': 'interval', 'channel': 'FL1-A', 'lo': -1e12,
                    'hi': 100.0, 'parent_id': 'g1', 'enabled': True},
             'g3': {'kind': 'interval', 'channel': 'FL1-A', 'lo': 100.0,
                    'hi': 1e12, 'parent_id': 'g1', 'name': 'Pos',
                    'enabled': True},
             'g4': {'kind': 'interval', 'channel': 'FL1-A', 'lo': 150.0,
                    'hi': 1e12, 'parent_id': 'g1', 'name': 'Pos',
                    'enabled': True}}
    neg, pos, vals = _titration(0.0)
    s = _sample('stained_1', vals)
    s.channel_labels = {'FL1-A': 'marker one'}
    _register(editor, s, gates)
    # The session's label differs; the sample's own wins in its rows.
    editor._channel_labels = {'FL1-A': 'session name'}
    editor._sample_gate_order['stained_1'] = [
        'g1', 'g2', 'ghost', 'g3', 'g4']
    w = StatisticsWindow(editor)
    w.withdraw()
    w._sync_ratio_choices()
    # Two 'All/Pos' gates (different bounds) are both offered; 'ghost' is
    # in the order but not a gate.
    labels = list(w._neg_gates)
    assert len(labels) == 4, labels
    assert sum(lb.startswith('All/Pos') for lb in labels) == 2
    neg_label = next(lb for lb in labels
                     if lb != 'All' and not lb.startswith('All/Pos'))
    w._delta_on.set(True)
    w._neg_mode.set(NEG_GATE)
    w._neg_gate.set(neg_label)
    w._on_neg_mode()
    row = next(r for r in w._rows if r['__gid__'] == 'g3')
    assert row['ΔMFI marker one'] == pytest.approx(
        np.median(pos) - np.median(neg))
    w.destroy()
