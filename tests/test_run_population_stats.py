"""Every Pipeline Workspace run writes a population table
(``<label>_population_stats.csv``): each member's gated population (or
whole sample) with Count, %Parent and %Total, and the Median and GeoMean of
each scatter and fluorescence channel on its linear values, all taken on
every event before the run's subsample cap.

Generated data, generic channel and sample names only.
"""
from __future__ import annotations

import csv
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace_run as R
from openflo.flow_sample import FlowSample
from tests._fcs_writer import write_float_fcs
from tests.conftest import gui_unavailable

CH = ['FSC-A', 'FL1-A', 'FL2-A', 'Time']
GATES = {
    'g1': {'kind': 'threshold', 'name': 'Big', 'channel': 'FSC-A',
           'value': 100000.0, 'id': 'g1', 'parent_id': None},
    'g2': {'kind': 'threshold', 'name': 'Bigger', 'channel': 'FSC-A',
           'value': 150000.0, 'id': 'g2', 'parent_id': 'g1'},
}
N = 800


@pytest.fixture
def fcs(tmp_path):
    rng = np.random.default_rng(11)
    ev = np.column_stack([rng.uniform(2e4, 2e5, N),
                          rng.lognormal(7, 0.6, N),
                          rng.lognormal(8, 0.4, N),
                          np.arange(N, dtype=float)])
    path = write_float_fcs(str(tmp_path / 'day_ctrl_1.fcs'), ev, CH)
    return path, ev


def _item(path, gid=None):
    chain = R.gate_chain(GATES, gid) if gid else []
    return {'sample': 'day_ctrl_1', 'trial': 'Day 2', 'path': path,
            'gate_id': gid, 'gate_chain': chain,
            'gates': {g: dict(GATES[g]) for g in chain},
            'qc_applied': False, 'comp_matrix': None, 'comp_channels': []}


def _loaded(path):
    s = FlowSample(path)
    s.apply_transform()
    return s


def test_stat_channels_are_scatter_and_fluorescence_not_time(fcs):
    assert R.stat_channels(_loaded(fcs[0])) == ['FSC-A', 'FL1-A', 'FL2-A']


def test_a_whole_sample_row_is_all_events_on_linear_values(fcs):
    path, ev = fcs
    s = _loaded(path)
    assert s.data_transforms.get('FL1-A')      # stored on a display scale
    [row] = R.population_table_rows(s, _item(path), 'day_ctrl_1')
    assert row['Population'] == 'All events'
    assert row['Count'] == N and row['%Total'] == 100.0
    assert np.isnan(row['%Parent'])
    np.testing.assert_allclose(row['Median FL1-A'], np.median(ev[:, 1]),
                               rtol=1e-6)
    assert row['GeoMean FL2-A'] > 0
    assert not [k for k in row if k.startswith('__')]


def test_a_gated_row_counts_against_its_parent_and_the_whole(fcs):
    path, ev = fcs
    [row] = R.population_table_rows(_loaded(path), _item(path, 'g2'), 'x')
    big = int((ev[:, 0] > 100000.0).sum())
    bigger = int((ev[:, 0] > 150000.0).sum())
    assert row['Count'] == bigger
    assert row['%Parent'] == pytest.approx(bigger / big * 100)
    assert row['%Total'] == pytest.approx(bigger / N * 100)
    np.testing.assert_allclose(
        row['Median FL1-A'], np.median(ev[ev[:, 0] > 150000.0, 1]), rtol=1e-6)


def test_the_stat_forms_reach_the_statistics(fcs, monkeypatch):
    from openflo import gating
    seen = []
    real = gating.population_stats

    def spy(*a, **kw):
        seen.append(kw.get('stat_forms'))
        return real(*a, **kw)
    monkeypatch.setattr(gating, 'population_stats', spy)
    forms = {'geomean': 'display', 'rcv': 'percentile'}
    R.population_table_rows(_loaded(fcs[0]), _item(fcs[0]), 'x', forms)
    assert seen == [forms]


def test_prepare_unit_takes_the_table_before_the_cap(fcs):
    path, ev = fcs
    members = [{'item': _item(path, 'g1'), 'group': 'Ctrl',
                'sample': 'day_ctrl_1'},
               {'item': _item(path), 'group': 'Ctrl', 'sample': 'day_ctrl_1'}]
    cfg = dict(R.default_run_cfg(), max_events=50)
    prep = R.prepare_unit(SimpleNamespace(_samples={}), members, 'Ctrl', cfg)
    assert prep['n_events'] == 50
    big, whole = prep['pop_rows']
    assert big['Count'] == int((ev[:, 0] > 100000.0).sum())
    assert whole['Count'] == N
    assert (big['Group'], big['Trial'], big['Population']) == \
        ('Ctrl', 'Day 2', 'Big')
    # A unit with no group (a loose row) says so with an empty Group.
    loose = R.prepare_unit(SimpleNamespace(_samples={}),
                           [{'item': _item(path), 'group': None,
                             'sample': 'day_ctrl_1'}], 'x', cfg)
    assert loose['pop_rows'][0]['Group'] == ''


def _read(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.reader(fh))


def test_compute_run_writes_the_table_even_when_it_cannot_cluster(tmp_path):
    rows = [{'Group': 'Ctrl', 'Trial': 'Day 2', 'Sample': 's1',
             'Median FL1-A': 12.5, 'Population': 'Big', 'Count': 3,
             '%Parent': 50.0, '%Total': 25.0}]
    prep = {'data': pd.DataFrame({'FL1-A': [1.0, 2.0]}), 'channels': ['FL1-A'],
            'label': 'Ctrl', 'pop_rows': rows}
    out = str(tmp_path / 'run')
    res = R.compute_run(prep, R.default_run_cfg(), out)
    assert not res['ok'] and 'too few events' in res['error']
    path = os.path.join(out, 'Ctrl_population_stats.csv')
    assert path in res['files']
    head, row = _read(path)
    assert head == ['Group', 'Trial', 'Sample', 'Population', 'Count',
                    '%Parent', '%Total', 'Median FL1-A']
    assert row[:5] == ['Ctrl', 'Day 2', 's1', 'Big', '3']
    rec, = R.find_run_outputs(out)
    assert rec['label'] == 'Ctrl' and rec['populations'] == path


def test_compute_run_without_rows_writes_no_table(tmp_path):
    prep = {'data': pd.DataFrame({'FL1-A': [1.0, 2.0]}),
            'channels': ['FL1-A'], 'label': 'Ctrl'}
    out = str(tmp_path / 'run')
    R.compute_run(prep, R.default_run_cfg(), out)
    assert not os.path.exists(os.path.join(out, 'Ctrl_population_stats.csv'))


def test_the_panel_hands_the_preferred_stat_forms_to_the_run(monkeypatch):
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as e:
        gui_unavailable(f'no display: {e}')
    root.withdraw()
    try:
        import openflo.workspace as W
        from openflo import prefs
        forms = {'geomean': 'display', 'rcv': 'percentile'}
        monkeypatch.setattr(prefs, 'load_stat_forms', lambda: forms)
        seen = []
        monkeypatch.setattr(W, 'build_run_units',
                            lambda model, cfg: seen.append(cfg) or [])
        panel = W.WorkspacePanel(root, editor=SimpleNamespace(_samples={}))
        panel._run()
        assert seen and seen[0]['stat_forms'] == forms
        assert 'stat_forms' not in panel.model.run_cfg
    finally:
        root.destroy()


def test_prepare_unit_passes_the_runs_stat_forms(fcs, monkeypatch):
    seen = []
    real = R.population_table_rows

    def spy(sample, item, name, stat_forms=None):
        seen.append(stat_forms)
        return real(sample, item, name, stat_forms)
    monkeypatch.setattr(R, 'population_table_rows', spy)
    forms = {'geomean': 'display', 'rcv': 'percentile'}
    R.prepare_unit(SimpleNamespace(_samples={}),
                   [{'item': _item(fcs[0]), 'group': 'Ctrl', 'sample': 'x'}],
                   'Ctrl', dict(R.default_run_cfg(), stat_forms=forms))
    assert seen == [forms]


def test_no_rows_no_file(tmp_path):
    res = {'files': []}
    R._write_population_stats([], str(tmp_path), 'Ctrl', res)
    assert res['files'] == [] and not os.listdir(tmp_path)
