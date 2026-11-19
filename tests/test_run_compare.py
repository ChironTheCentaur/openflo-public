"""Comparisons over a run template's history: within one run (samples the
replicates) and across runs (run days the replicates, paired by day)
(``openflo.run_compare``), and the Run history window.

Generated data, generic channel and sample names only.
"""
from __future__ import annotations

import csv
import os
from types import SimpleNamespace

import pytest

from openflo import run_compare as C
from openflo import run_history as H
from openflo import stats
from tests.conftest import gui_unavailable

HEAD = ['Run', 'Experiment date', 'Active', 'Template version', 'Group',
        'Trial', 'Sample', 'Population', 'Count', '%Parent', '%Total',
        'Median FL1-A', 'GeoMean FL1-A']


def _history(tmp_path, rows):
    path = tmp_path / H.HISTORY_CSV
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        w.writerows(rows)
    return str(tmp_path)


def _row(run, date, group, sample, pct, active='True', version='v1',
         pop='Big', count=100):
    return [run, date, active, version, group, 'Day', sample, pop, count,
            pct, pct / 2, 1000 + pct, 900 + pct]


@pytest.fixture
def hist(tmp_path):
    rows = []
    for i, (run, date) in enumerate((('run_1', '2026-11-10'),
                                     ('run_2', '2026-11-11'),
                                     ('run_3', '2026-11-12'))):
        for rep in (1, 2, 3):
            rows.append(_row(run, date, 'Ctrl', f'c{rep}', 20 + i + rep))
            rows.append(_row(run, date, 'Treat', f't{rep}', 40 + 2 * i + rep,
                             version='v2' if run == 'run_3' else 'v1'))
    rows.append(_row('run_0', '2026-11-10', 'Ctrl', 'old', 99.0,
                     active='False'))
    rows.append(_row('run_1', '2026-11-10', 'Ctrl', 'c1', 5.0, pop='Other'))
    return _history(tmp_path, rows)


def test_load_keeps_active_runs_unless_asked(hist):
    assert set(C.load_history(hist)['Run']) == {'run_1', 'run_2', 'run_3'}
    assert 'run_0' in set(C.load_history(hist, include_superseded=True)['Run'])


def test_superseded_runs_are_their_own_points_never_averaged(hist):
    df = C.load_history(hist, include_superseded=True)
    t = C.per_run_means(df, 'Big', '%Parent')
    assert list(t.index) == ['2026-11-10 · run_0', '2026-11-10 · run_1',
                             '2026-11-11', '2026-11-12']
    assert t.loc['2026-11-10 · run_0', 'Ctrl'] == 99.0
    assert t.loc['2026-11-10 · run_1', 'Ctrl'] == pytest.approx(22.0)
    assert [C.label_date(x) for x in t.index] == [
        '2026-11-10', '2026-11-10', '2026-11-11', '2026-11-12']
    # The across-run test pairs each day's newest run only.
    assert C.compare_across(df, 'Big', '%Parent') == \
        C.compare_across(C.load_history(hist), 'Big', '%Parent')


def test_a_series_missing_a_day_is_tested_on_the_days_it_has(tmp_path):
    rows = []
    for i, (run, date) in enumerate((('run_1', '2026-11-10'),
                                     ('run_2', '2026-11-11'),
                                     ('run_3', '2026-11-12'))):
        rows.append(_row(run, date, 'Ctrl', 'c', 20.0 + i))
        if run != 'run_2':
            rows.append(_row(run, date, 'Treat', 't', 40.0 + 3 * i))
    df = C.load_history(_history(tmp_path, rows))
    res = C.compare_across(df, 'Big', '%Parent')
    assert res['groups']['Treat']['n'] == 2
    assert res['groups']['Treat']['mean'] == pytest.approx(43.0)
    assert res['groups']['Ctrl']['n'] == 3
    # Paired on the days both have, each value with its own day.
    days = ['2026-11-10', '2026-11-11', '2026-11-12']
    assert res == stats.compare_groups(
        {'Ctrl': [20.0, 21.0, 22.0], 'Treat': [40.0, 46.0]}, transform='logit',
        pair_keys={'Ctrl': days, 'Treat': [days[0], days[2]]})


def test_no_history_is_an_empty_table(tmp_path):
    df = C.load_history(str(tmp_path))
    assert df.empty and C.runs(df) == [] and C.version_changes(df) == []


def test_stats_populations_runs_and_series(hist):
    df = C.load_history(hist)
    assert C.stat_names(df) == ['Count', '%Parent', '%Total', 'Median FL1-A',
                                'GeoMean FL1-A']
    assert C.populations(df) == ['Big', 'Other']
    assert C.runs(df) == [('run_3', '2026-11-12'), ('run_2', '2026-11-11'),
                          ('run_1', '2026-11-10')]
    assert list(C.series_choices(df)) == ['Group', 'Sample']


@pytest.mark.parametrize('stat,scale', [('%Parent', 'logit'), ('%Total', 'logit'),
                                        ('Count', 'log1p'),
                                        ('Median FL1-A', 'asinh'),
                                        ('GeoMean FL1-A', 'asinh')])
def test_each_statistic_is_tested_on_its_scale(stat, scale):
    assert C.scale_for(stat) == scale


def test_per_run_means_average_a_groups_samples_per_day(hist):
    t = C.per_run_means(C.load_history(hist), 'Big', '%Parent')
    assert list(t.index) == ['2026-11-10', '2026-11-11', '2026-11-12']
    assert list(t.columns) == ['Ctrl', 'Treat']
    assert t.loc['2026-11-11', 'Ctrl'] == pytest.approx(23.0)    # 22, 23, 24
    assert t.loc['2026-11-12', 'Treat'] == pytest.approx(46.0)   # 45, 46, 47
    with pytest.raises(ValueError, match='no Mean FL1-A'):
        C.per_run_means(C.load_history(hist), 'Big', 'Mean FL1-A')


def test_a_day_is_the_mean_of_its_samples_in_first_seen_order(tmp_path):
    rows = [_row('run_2', '2026-11-11', 'Treat', 't1', 1.0),
            _row('run_2', '2026-11-11', 'Treat', 't2', 2.0),
            _row('run_2', '2026-11-11', 'Treat', 't3', 6.0),
            _row('run_1', '2026-11-10', 'Ctrl', 'c1', 4.0),
            _row('run_1', '2026-11-10', 'Treat', 't1', 5.0)]
    t = C.per_run_means(C.load_history(_history(tmp_path, rows)), 'Big',
                        '%Parent')
    assert list(t.columns) == ['Treat', 'Ctrl']
    assert list(t.index) == ['2026-11-10', '2026-11-11']
    assert t.loc['2026-11-11', 'Treat'] == pytest.approx(3.0)   # not the median


def test_a_blank_active_cell_is_not_active(tmp_path):
    rows = [_row('run_1', '2026-11-10', 'Ctrl', 'c1', 4.0),
            _row('run_2', '2026-11-11', 'Ctrl', 'c1', 5.0, active='')]
    assert set(C.load_history(_history(tmp_path, rows))['Run']) == {'run_1'}


def test_within_a_run_the_samples_are_the_replicates(hist):
    df = C.load_history(hist)
    got = C.compare_within(df, 'run_2', 'Big', '%Parent')
    want = stats.compare_groups({'Ctrl': [22.0, 23.0, 24.0],
                                 'Treat': [43.0, 44.0, 45.0]},
                                transform='logit')
    assert got['test'] == want['test'] and not got['paired']
    assert got['p'] == pytest.approx(want['p'])
    assert got['groups']['Ctrl']['n'] == 3


def test_across_runs_the_days_are_paired_replicates(hist):
    df = C.load_history(hist)
    got = C.compare_across(df, 'Big', '%Parent')
    want = stats.compare_groups(
        {'Ctrl': [22.0, 23.0, 24.0], 'Treat': [42.0, 44.0, 46.0]},
        transform='logit',
        pair_keys={'Ctrl': ['2026-11-10', '2026-11-11', '2026-11-12'],
                   'Treat': ['2026-11-10', '2026-11-11', '2026-11-12']})
    assert got['paired'] and got['n_pairs'] == 3
    assert got['p'] == pytest.approx(want['p'])


def test_the_dates_where_the_template_changed(hist):
    assert C.version_changes(C.load_history(hist)) == ['2026-11-12']


def _rerun_history(tmp_path):
    """Day 2 run twice: first on the old template, then again after it
    changed; only the re-run is active."""
    rows = []
    for run, date, active, version in (
            ('run_1', '2026-11-10', 'True', 'v1'),
            ('run_2', '2026-11-11', 'False', 'v1'),
            ('run_3', '2026-11-11', 'True', 'v2')):
        rows.append(_row(run, date, 'Ctrl', 'c', 20.0, active=active,
                         version=version))
        rows.append(_row(run, date, 'Treat', 't', 40.0, active=active,
                         version=version))
    return _history(tmp_path, rows)


def test_a_dates_version_is_its_active_runs(tmp_path):
    h = _rerun_history(tmp_path)
    for superseded in (False, True):
        assert C.version_changes(C.load_history(h, superseded)) == \
            ['2026-11-11']


def test_the_template_line_is_drawn_once_before_its_date(tmp_path):
    from openflo.ui_run_compare import RunCompareWindow
    root = _tk()
    try:
        win = RunCompareWindow(root, _rerun_history(tmp_path))
        win.all_var.set(True)
        win.reload()
        win.stat_var.set('%Parent')
        win.redraw()
        ax = win.fig.axes[0]
        dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == '--']
        assert len(ax.get_lines()) == 3 and len(dashed) == 1
        assert list(dashed[0].get_xdata()) == [0.5, 0.5]
    finally:
        root.destroy()


def test_result_lines_say_the_test_groups_and_pairs():
    res = stats.compare_groups({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0],
                                'C': [7.0, 8.0, 9.0]})
    lines = C.result_lines(res)
    assert lines[0].startswith(f"{res['test']}: p = ")
    assert '  A: n = 3, mean = 2' in lines
    assert any(line.startswith('  A vs B: p(adj) = ') for line in lines)
    none = C.result_lines(stats.compare_groups({'A': [1.0]}))
    assert none[0].startswith('Not enough data')
    paired = C.result_lines(stats.compare_groups(
        {'A': [1.0, 2.0, 3.0], 'B': [2.0, 4.0, 5.0]},
        pair_keys={'A': [1, 2, 3], 'B': [1, 2, 3]}))
    assert ', paired by run day: p = ' in paired[0]


def test_export_table_writes_dates_down_series_across(hist, tmp_path):
    out = str(tmp_path / 'x.csv')
    C.export_table(C.load_history(hist), 'Big', '%Parent', 'Group', out)
    with open(out, encoding='utf-8-sig', newline='') as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ['Experiment date', 'Ctrl', 'Treat']
    assert rows[1][0] == '2026-11-10' and len(rows) == 4


# ── the window ─────────────────────────────────────────────────────────────

def _tk():
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    return root


def test_the_window_plots_and_tests_the_history(hist, tmp_path, monkeypatch):
    from tkinter import filedialog

    from openflo.ui_run_compare import RunCompareWindow
    root = _tk()
    try:
        win = RunCompareWindow(root, hist)
        assert (win.pop_var.get(), win.stat_var.get(), win.by_var.get()) == \
            ('Big', 'Count', 'Group')
        assert win.run_var.get() == 'run_3 (2026-11-12)'
        win.stat_var.set('%Parent')
        win.redraw()
        ax = win.fig.axes[0]
        assert [ln.get_label() for ln in ax.get_lines()][:2] == ['Ctrl', 'Treat']
        assert list(ax.get_lines()[0].get_ydata()) == pytest.approx([22, 23, 24])
        text = win.summary.get('1.0', 'end')
        assert 'Across runs (each run day one replicate):' in text
        assert ', paired by run day: p = ' in text
        assert 'Within run_3 (2026-11-12) (each sample one replicate):' in text
        assert 'The template changed on 2026-11-12' in text
        # The dashed line sits before the changed day.
        assert len(ax.lines) == 3
        assert 'run(s)' in win.where.get()

        win.pop_var.set('Gone')                  # a choice the history lacks
        win.reload()
        assert win.pop_var.get() == 'Big'
        win.all_var.set(True)
        win.reload()
        assert 'run_0 (2026-11-10)' in win.run_cb['values']

        out = str(tmp_path / 'p.png')
        monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: out)
        win.save_plot()
        assert os.path.getsize(out) > 0
        csv_out = str(tmp_path / 't.csv')
        monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **k: csv_out)
        win.export_table()
        assert os.path.exists(csv_out)
        assert 'Saved t.csv.' in win.summary.get('1.0', 'end')
    finally:
        root.destroy()


def test_the_window_says_when_there_is_no_history(tmp_path, monkeypatch):
    from tkinter import filedialog

    from openflo.ui_run_compare import RunCompareWindow
    root = _tk()
    try:
        win = RunCompareWindow(root)
        assert win.where.get().startswith('No run history open')
        win.redraw()
        win.export_table()                       # nothing to export: no-op
        win = RunCompareWindow(root, str(tmp_path))
        assert 'No finished runs' in win.summary.get('1.0', 'end')
        monkeypatch.setattr(filedialog, 'askdirectory', lambda **k: '')
        win._choose_folder()
        assert win.hdir == str(tmp_path)
        other = str(tmp_path / 'other')
        monkeypatch.setattr(filedialog, 'askdirectory', lambda **k: other)
        win._choose_folder()
        assert win.hdir == other
    finally:
        root.destroy()


def _volt_history(h, runs):
    for run, date, volts in runs:
        d = os.path.join(h, run)
        os.makedirs(d)
        H.write_run_info(d, {'run_id': run, 'status': H.STATUS_DONE,
                             'experiment_date': date,
                             'template': {'version': 'v1'},
                             'files': [{'voltages': {'FL1-A': volts}}]})
        with open(os.path.join(d, 'G_population_stats.csv'), 'w',
                  encoding='utf-8') as fh:
            fh.write('Group,Trial,Sample,Population,Count\n'
                     f'G,Day,s,Big,{int(volts)}\n')
    H.rebuild_history(h)


def test_the_window_reports_voltage_changes(tmp_path):
    from openflo.ui_run_compare import RunCompareWindow
    moved = str(tmp_path / 'moved')
    _volt_history(moved, (('run_1', '2026-11-10', 400.0),
                          ('run_2', '2026-11-11', 450.0)))
    # run_2 changed the voltage but run_3 re-ran its day at the old one:
    # only the active run counts.
    rerun = str(tmp_path / 'rerun')
    _volt_history(rerun, (('run_1', '2026-11-10', 400.0),
                          ('run_2', '2026-11-11', 500.0),
                          ('run_3', '2026-11-11', 400.0)))
    root = _tk()
    try:
        text = RunCompareWindow(root, moved).summary.get('1.0', 'end')
        assert '2026-11-11: detector voltages changed since the 2026-11-10 ' \
            'run (FL1-A)' in text
        text = RunCompareWindow(root, rerun).summary.get('1.0', 'end')
        assert 'voltages changed' not in text
    finally:
        root.destroy()


def test_the_panel_opens_its_templates_history(tmp_path, monkeypatch):
    root = _tk()
    try:
        import openflo.ui_run_compare as U
        from openflo import prefs
        from openflo.workspace import WorkspacePanel
        monkeypatch.setattr(prefs, 'load_pipeline_output_dir', lambda: None)
        opened = []
        monkeypatch.setattr(U, 'RunCompareWindow',
                            lambda parent, hdir: opened.append(hdir))
        panel = WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel._open_history()
        panel.model.template = {'path': str(tmp_path / 'course.json'),
                                'name': 'course', 'version': 'v'}
        panel._open_history()
        assert opened == [None, str(tmp_path / 'course_history')]
    finally:
        root.destroy()


# ── a day run partly again (#404 review follow-ups) ────────────────────────

def _split_day(tmp_path):
    """Day 2 run, then its group Ctrl run again on a changed template: the
    day's active rows come from two runs."""
    rows = []
    for run, date, group, active, version in (
            ('run_1', '2026-11-10', 'Ctrl', 'True', 'v1'),
            ('run_1', '2026-11-10', 'Treat', 'True', 'v1'),
            ('run_2', '2026-11-11', 'Treat', 'True', 'v1'),
            ('run_2', '2026-11-11', 'Ctrl', 'False', 'v1'),
            ('run_3', '2026-11-11', 'Ctrl', 'True', 'v2')):
        for rep in (1, 2):
            rows.append(_row(run, date, group, f'{group}{rep}',
                             (20.0 if group == 'Ctrl' else 40.0) + rep,
                             active=active, version=version))
    return _history(tmp_path, rows)


def test_within_a_rerun_its_day_is_compared(tmp_path):
    h = _split_day(tmp_path)
    df = C.load_history(h)
    got = C.compare_within(df, 'run_3', 'Big', '%Parent')
    assert list(got['groups']) == ['Treat', 'Ctrl']
    assert got['groups']['Treat']['n'] == 2
    # A superseded run, shown on request, is its own rows only.
    every = C.load_history(h, include_superseded=True)
    assert sorted(set(C.within_rows(every, 'run_2')['Run'])) == ['run_2',
                                                               'run_3']
    # The day's superseded rows (run_2's Ctrl) stay out.
    assert len(C.within_rows(every, 'run_3')) == 4
    old = every[every['Run'] == 'run_2'].assign(Active=False)
    assert set(C.within_rows(old, 'run_2')['Run']) == {'run_2'}
    assert len(C.within_rows(every, 'run_1')) == 4


def test_a_day_holding_two_versions_is_marked(tmp_path):
    h = _split_day(tmp_path)
    for superseded in (False, True):
        assert C.version_changes(C.load_history(h, superseded)) == \
            ['2026-11-11']


def test_a_superseded_version_does_not_mark_the_next_day(tmp_path):
    h = _history(tmp_path, [
        _row('run_1', '2026-11-10', 'Ctrl', 'c', 20.0, version='v2'),
        _row('run_2', '2026-11-11', 'Ctrl', 'c', 20.0, active='False',
             version='v1'),
        _row('run_3', '2026-11-11', 'Ctrl', 'c', 20.0, version='v2'),
        _row('run_4', '2026-11-12', 'Ctrl', 'c', 20.0, version='v2')])
    assert C.version_changes(C.load_history(h, True)) == []


def test_the_window_says_a_days_rows_came_from_two_runs(tmp_path):
    from openflo.ui_run_compare import RunCompareWindow
    root = _tk()
    try:
        win = RunCompareWindow(root, _split_day(tmp_path))
        win.stat_var.set('%Parent')
        win.run_var.set('run_3 (2026-11-11)')
        win.redraw()
        text = win.summary.get('1.0', 'end')
        assert ("Within run_3 (2026-11-11) (each sample one replicate):\n"
                "  (the day's active rows, from runs run_2, run_3)\n") in text
        win.run_var.set('run_1 (2026-11-10)')
        win.redraw()
        assert "the day's active rows" not in win.summary.get('1.0', 'end')
    finally:
        root.destroy()


@pytest.mark.parametrize('first', ['v1', 'v2'])
def test_a_day_that_adds_a_version_is_marked(tmp_path, first):
    # Whichever of the day's two versions its rows list first.
    second = 'v2' if first == 'v1' else 'v1'
    h = _history(tmp_path, [
        _row('run_1', '2026-11-10', 'Ctrl', 'c', 20.0, version='v1'),
        _row('run_2', '2026-11-11', 'Ctrl', 'c', 20.0, version=first),
        _row('run_3', '2026-11-11', 'Treat', 't', 20.0, version=second)])
    assert C.version_changes(C.load_history(h)) == ['2026-11-11']
