"""Analyze -> Statistics: the exported CSV carries the numbers, unrounded.

ui_statistics.StatisticsWindow._export_csv ran at 22 % under the suite (only
its "nothing to export" branch). Here one sample with a hand-checkable channel
is exported with every statistic switched on, and each cell is compared with
the formula the help text describes. The scale question is pinned separately
because the editor's loader hands the window logicle values.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

CD903 = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, -3.0, 5.0])


def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        r = work()
    except Exception as exc:                      # noqa: BLE001
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(r)
    finally:
        if on_finally:
            on_finally()


def _register(ed, s, gates, order):
    name = s.name
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = gates
    ed._sample_gate_order[name] = order
    ed._sample_gate_seq[name] = len(order)
    ed._channels = list(s.data.columns)
    ed._channel_labels = dict(s.channel_labels)
    ed._active_sample = name


def _export(ed, monkeypatch, path, stats=('Mean', 'GeoMean', 'CV', 'rCV')):
    import openflo.ui_statistics as us
    from openflo.ui_statistics import StatisticsWindow
    w = StatisticsWindow(ed)
    w.withdraw()
    for st in stats:
        w._stat_vars[st].set(True)
    w._refresh()
    monkeypatch.setattr(us.filedialog, 'asksaveasfilename', lambda **k: str(path))
    w._export_csv()
    return {r['Population']: r for r in csv.DictReader(open(path, encoding='utf-8'))}


def test_exported_statistics_match_their_formulas(monkeypatch, tmp_path):
    import openflo.async_task as _at
    from openflo.pipeline import FlowSample
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, _gui = _editor_or_skip()
    try:
        s = FlowSample.from_dataframe(
            pd.DataFrame({'FSC-A': np.arange(1.0, 11.0), 'CD903-A': CD903}),
            name='s1', labels={'CD903-A': 'CD903'})
        gates = {
            'g1': {'kind': 'threshold', 'channel': 'FSC-A', 'value': 0.0,
                   'parent_id': None, 'name': 'All', 'enabled': True},
            'g2': {'kind': 'threshold', 'channel': 'CD903-A', 'value': 3.0,
                   'parent_id': 'g1', 'name': 'Bright', 'enabled': True},
            'g3': {'kind': 'threshold', 'channel': 'CD903-A', 'value': 1e9,
                   'parent_id': 'g1', 'name': 'Empty', 'enabled': True}}
        _register(ed, s, gates, ['g1', 'g2', 'g3'])
        rows = _export(ed, monkeypatch, tmp_path / 'stats.csv')

        v = CD903
        b = v[v > 3.0]
        assert rows['All']['Count'] == '10' and rows['All/Bright']['Count'] == str(b.size)
        assert float(rows['All/Bright']['%Parent']) == pytest.approx(100 * b.size / 10)
        for pop, x in (('All', v), ('All/Bright', b)):
            r = rows[pop]
            med = np.median(x)
            assert float(r['Median CD903 (linear)']) == pytest.approx(med, rel=1e-12)
            assert float(r['Mean CD903 (linear)']) == pytest.approx(x.mean(), rel=1e-12)
            # Geometric mean over POSITIVE values only (-3 is excluded).
            pos = x[x > 0]
            assert float(r['GeoMean CD903 (linear)']) == pytest.approx(
                np.exp(np.log(pos).mean()), rel=1e-12)
            assert float(r['CV CD903 (linear)']) == pytest.approx(
                x.std() / x.mean() * 100, rel=1e-12)
            lo, hi = np.percentile(x, [15.87, 84.13])   # FlowJo's rCV
            assert float(r['rCV CD903 (linear)']) == pytest.approx(
                50.0 * (hi - lo) / med, rel=1e-12)
        # An empty population exports blanks, not 'nan' or 0.
        assert rows['All/Empty']['Count'] == '0'
        assert rows['All/Empty']['Median CD903 (linear)'] == ''
        assert rows['All/Empty']['GeoMean CD903 (linear)'] == ''
    finally:
        root.destroy()


def test_geomean_of_an_editor_loaded_fcs_is_on_the_linear_scale(monkeypatch, tmp_path):
    """The loader logicle-transforms fluor channels; the statistics must still
    come out on the compensated linear scale (GeoMean ~1100, not ~0.46)."""
    import flowio

    import openflo.async_task as _at
    import openflo.pipeline as fp
    monkeypatch.setattr(_at, 'run_async', _inline)
    rng = np.random.default_rng(0)
    n = 5000
    fl2 = rng.lognormal(7.0, 0.5, n)
    ev = np.column_stack([rng.normal(8e4, 8e3, n), rng.normal(4e4, 5e3, n), fl2,
                          np.linspace(0, 100, n)]).astype(np.float32)
    path = tmp_path / 'cd903.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(),
                          ['FSC-A', 'SSC-A', 'FL2-A', 'Time'],
                          opt_channel_names=['', '', 'CD903', ''])
    s = fp.FlowSample(str(path))           # what editor_loadpool._load_worker does
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    root, ed, _gui = _editor_or_skip()
    try:
        _register(ed, s, {'g1': {'kind': 'threshold', 'channel': 'FSC-A',
                                 'value': 0.0, 'parent_id': None, 'name': 'All',
                                 'enabled': True}}, ['g1'])
        rows = _export(ed, monkeypatch, tmp_path / 'stats.csv', stats=('GeoMean',))
        true_gm = float(np.exp(np.log(fl2).mean()))        # ~ e^7 ~ 1097
        assert float(rows['All']['GeoMean CD903 (linear)']) == pytest.approx(true_gm, rel=0.03)
    finally:
        root.destroy()


def _editor_loaded_lognormal(tmp_path):
    import flowio

    import openflo.pipeline as fp
    rng = np.random.default_rng(0)
    n = 5000
    fl2 = rng.lognormal(7.0, 0.5, n)
    ev = np.column_stack([rng.normal(8e4, 8e3, n), rng.normal(4e4, 5e3, n), fl2,
                          np.linspace(0, 100, n)]).astype(np.float32)
    path = tmp_path / 'cd903.fcs'
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(),
                          ['FSC-A', 'SSC-A', 'FL2-A', 'Time'],
                          opt_channel_names=['', '', 'CD903', ''])
    s = fp.FlowSample(str(path))           # what editor_loadpool._load_worker does
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    return s, fl2


def test_every_intensity_statistic_is_linear_and_the_window_says_so(
        monkeypatch, tmp_path):
    """Median, Mean, GeoMean, CV and rCV of a loader-processed sample are all
    the linear-scale formulas, and the window labels the scale."""
    import openflo.async_task as _at
    from openflo.pipeline import inverse_transform_values
    monkeypatch.setattr(_at, 'run_async', _inline)
    s, fl2 = _editor_loaded_lognormal(tmp_path)
    x = inverse_transform_values(s.data['FL2-A'].to_numpy(float),
                                 method='logicle')
    root, ed, _gui = _editor_or_skip()
    try:
        _register(ed, s, {'g1': {'kind': 'threshold', 'channel': 'FSC-A',
                                 'value': 0.0, 'parent_id': None, 'name': 'All',
                                 'enabled': True}}, ['g1'])
        rows = _export(ed, monkeypatch, tmp_path / 'stats.csv')
        r = rows['All']
        med = np.median(x)
        lo, hi = np.percentile(x, [15.87, 84.13])
        # FlowJo's definitions: GeoMean is the display-space (logicle) mean
        # back-transformed; rCV comes from the 15.87/84.13 percentiles.
        gm = inverse_transform_values(np.array([
            s.data['FL2-A'].to_numpy(float).mean()]), method='logicle')[0]
        for col, want in (('Median', med), ('Mean', x.mean()),
                          ('GeoMean', gm),
                          ('CV', x.std() / x.mean() * 100),
                          ('rCV', 50.0 * (hi - lo) / med)):
            assert float(r[f'{col} CD903 (linear)']) == pytest.approx(want, rel=1e-9), col
        # ...and they are the generating distribution's, not logicle units.
        assert float(r['Median CD903 (linear)']) == pytest.approx(np.median(fl2), rel=0.03)
        assert float(r['CV CD903 (linear)']) == pytest.approx(
            fl2.std() / fl2.mean() * 100, rel=0.05)
        from openflo.ui_statistics import StatisticsWindow
        w = StatisticsWindow(ed)
        w.withdraw()
        assert 'compensated linear scale' in w.status_var.get()
        assert w._fmt('Median CD903', 1097.4) == '1,097'
    finally:
        root.destroy()


def test_every_exported_intensity_header_names_its_scale(monkeypatch, tmp_path):
    """The exported CSV leaves the window; its headers must carry the scale
    the numbers are on, not just the status bar. Identity and population
    columns are not intensities and stay as they were."""
    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    s, _fl2 = _editor_loaded_lognormal(tmp_path)
    root, ed, _gui = _editor_or_skip()
    try:
        _register(ed, s, {'g1': {'kind': 'threshold', 'channel': 'FSC-A',
                                 'value': 0.0, 'parent_id': None, 'name': 'All',
                                 'enabled': True}}, ['g1'])
        path = tmp_path / 'stats.csv'
        _export(ed, monkeypatch, path)
        header = next(csv.reader(open(path, encoding='utf-8')))
        stat_cols = [h for h in header
                     if h.split(' ', 1)[0] in ed.STAT_CHAN]
        assert stat_cols, header
        assert all(h.endswith(' (linear)') for h in stat_cols), header
        assert {'Sample', 'Population', 'Count', '%Parent'} <= set(header)
    finally:
        root.destroy()
