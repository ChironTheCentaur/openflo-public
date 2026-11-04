"""Westgard's R-4s within a run, across two or more bead levels (Sky's
option D): one level beyond +2 SD and another beyond -2 SD in the same run
rejects it. The check across two runs in a row stays on by default;
Preferences > Statistics, "Westgard R-4s: literature form only", turns it
off. One-level logs load and judge as before.

Generated data only, with synthetic sample names and generic channels
(FL1-A, FL2-A); every check is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import tests._sample_editor as shared
from openflo import bead_tracking as bt
from openflo import prefs
from tests._sample_editor import _set

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

# Baseline runs alternate 90/110 (mean 100, SD 10.26 with n-1) for the dim
# level and 990/1010 (mean 1000, SD 10.26) for the bright level.
_DIM_BASE = [90.0, 110.0] * 5
_BRIGHT_BASE = [990.0, 1010.0] * 5
_SD = float(np.std(_DIM_BASE, ddof=1))


def _two_level_log(dim_tail, bright_tail):
    log = bt.BeadLog(baseline_runs=10)
    for i, (d, b) in enumerate(zip(_DIM_BASE + dim_tail,
                                   _BRIGHT_BASE + bright_tail, strict=True)):
        log.add_levels(f'2026-02-{i + 1:02d}', f'beads_{i}',
                       {'dim': {'FL1-A': d}, 'bright': {'FL1-A': b}})
    return log


def test_level_columns_name_channel_and_level():
    assert bt.level_key('FL1-A', 'dim') == 'FL1-A @ dim'
    assert bt.level_key('FL1-A') == 'FL1-A'
    assert bt.split_level('FL1-A @ dim') == ('FL1-A', 'dim')
    assert bt.split_level('FL1-A') == ('FL1-A', '')
    assert bt.split_level('FL1-A @ ') == ('FL1-A @ ', '')
    # A gate name may hold the separator; the channel is before the first.
    assert bt.split_level('FL1-A @ peak @ 2') == ('FL1-A', 'peak @ 2')
    assert bt.split_level(bt.level_key('FL1-A', 'peak @ 2')) == \
        ('FL1-A', 'peak @ 2')


def test_within_run_r4s_needs_opposite_sides_beyond_2sd():
    assert bt.within_run_r4s([[2.5, 2.5, 1.9, 2.5], [-2.5, 2.5, -2.5, -1.9]]) \
        == [True, False, False, False]
    # Exactly 2 SD is not beyond it.
    assert bt.within_run_r4s([[2.0, 2.01], [-2.01, -2.01]]) == [False, True]
    # A third level: any pair on opposite sides fires.
    assert bt.within_run_r4s([[0.0], [2.2], [-2.2]]) == [True]
    # NaN (level missing from the run, or no baseline) breaks nothing.
    assert bt.within_run_r4s([[np.nan, 2.5], [-2.5, np.nan]]) == [False, False]
    # One level never breaks it.
    assert bt.within_run_r4s([[2.5, -2.5]]) == [False, False]


def test_two_levels_on_opposite_sides_in_one_run_reject_it():
    # Run 11: dim at +2.5 SD, bright at -2.5 SD -> R-4s within the run,
    # though neither level breaks R-4s across runs (both runs 10 in range).
    d, b = 100 + 2.5 * _SD, 1000 - 2.5 * _SD
    log = _two_level_log([d, 100.0], [b, 1000.0])
    assert log.channels == ['FL1-A @ dim', 'FL1-A @ bright']
    for col in log.channels:
        j = log.judge(col)
        assert j['rules'][10] == ['R-4s', '1-2s'], col
        assert j['status'][10] == 'reject'
        assert j['rules'][11] == []
    # Same side: no R-4s, only the 1-2s warning.
    log = _two_level_log([d, 100.0], [1000 + 2.5 * _SD, 1000.0])
    assert log.judge('FL1-A @ dim')['rules'][10] == ['1-2s']


def test_a_run_rejected_within_shows_on_every_level():
    log = bt.BeadLog(baseline_runs=10)
    tails = {'dim': 100 + 2.5 * _SD, 'mid': 500.0, 'bright': 1000 - 2.5 * _SD}
    bases = {'dim': _DIM_BASE, 'mid': [490.0, 510.0] * 5,
             'bright': _BRIGHT_BASE}
    for i in range(11):
        log.add_levels('d', f's{i}', {lvl: {'FL1-A': (bases[lvl][i] if i < 10
                                                      else tails[lvl])}
                                      for lvl in bases})
    assert log.judge('FL1-A @ mid')['rules'][10] == ['R-4s']
    assert log.judge('FL1-A @ mid')['status'][10] == 'reject'


def test_levels_of_other_channels_do_not_mix():
    log = bt.BeadLog(baseline_runs=10)
    for i in range(11):
        hi = i == 10
        log.add_levels('d', f's{i}', {
            'dim': {'FL1-A': 100 + 2.5 * _SD if hi else _DIM_BASE[i],
                    'FL2-A': 100.0 if hi else _DIM_BASE[i]},
            'bright': {'FL1-A': 1000.0 if hi else _BRIGHT_BASE[i],
                       'FL2-A': 1000 - 2.5 * _SD if hi else _BRIGHT_BASE[i]}})
    assert log.levels_of('FL1-A @ dim') == ['FL1-A @ dim', 'FL1-A @ bright']
    assert log.judge('FL1-A @ dim')['rules'][10] == ['1-2s']
    assert log.judge('FL2-A @ bright')['rules'][10] == ['1-2s']


def test_a_plain_column_is_not_a_level_of_its_channel():
    # A one-level run of FL1-A, then two-level runs: the plain FL1-A column
    # is its own series, not a third level.
    log = bt.BeadLog(baseline_runs=10)
    log.add('d', 'old_beads', {'FL1-A': 500.0})
    for i in range(11):
        hi = i == 10
        log.add_levels('d', f's{i}', {
            'dim': {'FL1-A': 100 + 2.5 * _SD if hi else _DIM_BASE[i]},
            'bright': {'FL1-A': 1000 - 2.5 * _SD if hi else _BRIGHT_BASE[i]}})
    assert log.channels == ['FL1-A', 'FL1-A @ dim', 'FL1-A @ bright']
    assert log.levels_of('FL1-A') == ['FL1-A']
    assert log.levels_of('FL1-A @ dim') == ['FL1-A @ dim', 'FL1-A @ bright']
    plain = log.judge('FL1-A')
    assert plain['status'] == ['ok'] + ['NA'] * 11
    assert np.isnan(plain['sd'])                  # one value: no baseline
    dim = log.judge('FL1-A @ dim')
    assert dim['status'][0] == 'NA'
    assert dim['rules'][11] == ['R-4s', '1-2s']


def test_levels_with_the_separator_in_their_name_group_by_channel():
    log = bt.BeadLog(baseline_runs=10)
    for i in range(11):
        hi = i == 10
        log.add_levels('d', f's{i}', {
            'dim @ x': {'FL1-A': 100 + 2.5 * _SD if hi else _DIM_BASE[i]},
            'bright @ x': {'FL1-A': 1000 - 2.5 * _SD if hi else _BRIGHT_BASE[i]}})
    assert log.levels_of('FL1-A @ dim @ x') == ['FL1-A @ dim @ x',
                                                'FL1-A @ bright @ x']
    assert log.judge('FL1-A @ bright @ x')['rules'][10] == ['R-4s', '1-2s']


def test_across_runs_r4s_is_on_by_default_and_can_be_turned_off():
    # One level: +2.5 SD then -2.5 SD in two runs in a row.
    v = [100.0, 125.0, 75.0]
    assert bt.westgard(v, 100.0, 10.0)[-1] == ['R-4s', '1-2s']
    assert bt.westgard(v, 100.0, 10.0, across_runs_r4s=False)[-1] == ['1-2s']
    # Other rules are untouched by the switch.
    assert bt.westgard([100, 131], 100.0, 10.0, across_runs_r4s=False)[-1] \
        == ['1-3s', '1-2s']
    log = _two_level_log([100 + 2.5 * _SD, 100 - 2.5 * _SD], [1000.0, 1000.0])
    assert log.judge('FL1-A @ dim')['rules'][11] == ['R-4s', '1-2s']
    assert log.judge('FL1-A @ dim', across_runs_r4s=False)['rules'][11] == \
        ['1-2s']
    # The within-run form still fires with the across-run form off.
    log = _two_level_log([100 + 2.5 * _SD], [1000 - 2.5 * _SD])
    assert log.judge('FL1-A @ dim', across_runs_r4s=False)['rules'][10] == \
        ['R-4s', '1-2s']


def test_one_level_logs_load_and_judge_as_before():
    text = ("date,sample,FL1-A,FL2-A\n"
            + ''.join(f"2026-03-{i + 1:02d},beads_{i},{x},50\n"
                      for i, x in enumerate(_DIM_BASE + [125.6, 74.4])))
    log = bt.BeadLog.from_csv(text, baseline_runs=10)
    assert log.channels == ['FL1-A', 'FL2-A']
    assert log.levels_of('FL1-A') == ['FL1-A']
    assert log.judge('FL1-A')['rules'][-1] == ['R-4s', '1-2s']
    # add_levels with one level writes plain channel columns.
    log.add_levels('2026-03-13', 'beads_12', {'All events': {'FL1-A': 100.0}})
    assert log.runs[-1]['values'] == {'FL1-A': 100.0}


def test_multi_level_log_csv_round_trip():
    log = _two_level_log([100.0], [1000.0])
    text = log.to_csv()
    assert text.splitlines()[0] == 'date,sample,FL1-A @ dim,FL1-A @ bright'
    back = bt.BeadLog.from_csv(text, baseline_runs=10)
    assert back.runs == log.runs


def test_preference_reads_default_and_literature(tmp_path, monkeypatch):
    p = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    assert prefs.load_westgard_r4s_across() is True
    p.write_text(json.dumps({'westgard_r4s_literature': True}))
    assert prefs.load_westgard_r4s_across() is False
    p.write_text(json.dumps({'westgard_r4s_literature': False}))
    assert prefs.load_westgard_r4s_across() is True


def test_preferences_box_stores_literature_form(editor, tmp_path, monkeypatch):
    from openflo.ui_preferences import PreferencesDialog
    p = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    root, ed = editor
    dlg = PreferencesDialog(ed)
    root.update()
    box = dlg.r4s_literature_box
    assert 'literature form only' in box.cget('text')
    assert not dlg._r4s_literature_var.get()          # safe default: off
    box.invoke()
    assert json.loads(p.read_text())['westgard_r4s_literature'] is True
    assert prefs.load_westgard_r4s_across() is False
    box.invoke()
    assert prefs.load_westgard_r4s_across() is True
    dlg.destroy()


def test_window_logs_two_levels_and_follows_the_preference(editor, tmp_path,
                                                          monkeypatch):
    from openflo import ui_bead_tracking as ub
    p = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    root, ed = editor
    _set(ed, 'alpha_1', **{'FL1-A': np.full(400, 100.0)})
    ed._sample_gates['alpha_1'] = {
        'g1': {'name': 'dim', 'kind': 'range', 'x': 'FL1-A',
               'xmin': 0.0, 'xmax': 1e9},
        'g2': {'name': 'bright', 'kind': 'range', 'x': 'FL1-A',
               'xmin': 0.0, 'xmax': 1e9}}
    win = ub.BeadTrackingWindow(ed)
    root.update()
    win.baseline_var.set(10)
    # Feed each level's median straight in: the gates select every event, so
    # the run's medians are set by the data per level below.
    medians = {'dim': _DIM_BASE + [100 + 2.5 * _SD],
               'bright': _BRIGHT_BASE + [1000 - 2.5 * _SD]}
    real = ub.run_values
    monkeypatch.setattr(
        ub, 'run_values',
        lambda s, g, pop, ch: {'FL1-A': medians[pop][len(win.log.runs)]}
        if pop in medians else real(s, g, pop, ch))
    win.set_populations(['dim', 'bright'])
    assert win.selected_populations() == ['dim', 'bright']
    for _ in range(11):
        win.add_active()
    assert win.log.channels == ['FL1-A @ dim', 'FL1-A @ bright']
    win.channel_var.set('FL1-A @ bright')
    win.draw()
    assert win.fig.axes[0].get_ylabel().endswith('median (bright)')
    rows = win.table.get_children()
    assert win.table.item(rows[-1])['values'][2] == 'R-4s, 1-2s'
    assert 'within each run across bead levels, and across' in \
        win.r4s_var.get()
    p.write_text(json.dumps({'westgard_r4s_literature': True}))
    win.draw()
    assert win.table.item(win.table.get_children()[-1])['values'][2] == \
        'R-4s, 1-2s'
    assert win.r4s_var.get().startswith('R-4s: literature form, within')
    win.set_populations([])
    win.add_active()
    assert 'Pick the beads' in win.status_var.get()
    assert len(win.log.runs) == 11
    win.destroy()


def test_window_says_one_level_literature_never_breaks_r4s(editor, tmp_path,
                                                          monkeypatch):
    from openflo import ui_bead_tracking as ub
    p = tmp_path / 'prefs.json'
    p.write_text(json.dumps({'westgard_r4s_literature': True}))
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(p))
    root, ed = editor
    log = bt.BeadLog(baseline_runs=2)
    for i, x in enumerate([90.0, 110.0, 131.0, 69.0]):
        log.add('d', f'beads_{i}', {'FL1-A': x})
    win = ub.BeadTrackingWindow(ed, log=log)
    root.update()
    win.baseline_var.set(2)
    win.draw()
    assert 'one bead level, so R-4s is never broken' in win.r4s_var.get()
    assert win.fig.axes[0].get_ylabel().endswith('median')
    last = win.table.item(win.table.get_children()[-1])['values'][2]
    assert 'R-4s' not in last
    p.write_text('{}')
    win.draw()
    assert 'R-4s' in win.table.item(win.table.get_children()[-1])['values'][2]
    assert win.r4s_var.get().startswith('R-4s: across two runs')
    win.destroy()


def test_a_picked_gate_with_no_events_adds_no_run(editor):
    from openflo import ui_bead_tracking as ub
    root, ed = editor
    _set(ed, 'alpha_1', **{'FL1-A': np.full(400, 100.0),
                           'FL2-A': np.full(400, 50.0)})
    ed._sample_gates['alpha_1'] = {
        'g1': {'name': 'dim', 'kind': 'rect', 'x_channel': 'FL1-A',
               'y_channel': 'FL2-A', 'x0': 0.0, 'x1': 1e6,
               'y0': 0.0, 'y1': 1e6},
        'g2': {'name': 'nothing', 'kind': 'rect', 'x_channel': 'FL1-A',
               'y_channel': 'FL2-A', 'x0': 1e5, 'x1': 1e6,
               'y0': 0.0, 'y1': 1e6}}
    win = ub.BeadTrackingWindow(ed)
    root.update()
    win.set_populations(['dim', 'nothing'])
    win.add_active()
    assert "alpha_1 has no events in 'nothing'." == win.status_var.get()
    assert win.log.runs == []
    # The gate that holds events alone logs a one-level run.
    win.set_populations(['dim'])
    win.add_active()
    assert len(win.log.runs) == 1
    assert win.log.runs[0]['values']['FL1-A'] == pytest.approx(100.0)
    win.destroy()
