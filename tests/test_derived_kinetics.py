"""Derived channels (a channel computed from others, derived.py) and
kinetics (a channel over acquisition time, kinetics.py).

Ground truth is built in: every sample is synthetic, with the ratio of two
detectors or the response over time known in advance.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import derived, kinetics
from tests.test_transform_records import (
    _new_editor,
    _pump,
    capture_warnings,
    isolate_home,
)

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'Time']
MK = ['', '', 'CD901b', 'CD902', '']


def _fcs(path, seed, timestep='0.01', n=1200):
    """FL1 and FL2 lognormal, Time 0..n-1 ticks; FL1 steps up 3x halfway."""
    import flowio
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    fl2 = rng.lognormal(7, .2, n)
    fl1 = fl2 * np.where(t < n // 2, 1.0, 3.0) * rng.lognormal(0, .05, n)
    ev = np.column_stack([rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
                          fl1, fl2, t])
    meta = {'TIMESTEP': timestep} if timestep else None
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(), CH,
                          opt_channel_names=MK, metadata_dict=meta)
    return str(path)


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)
    s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    return s


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, _new_editor(root), tmp_path
    root.destroy()


# ── Formula ─────────────────────────────────────────────────────────────────

def test_formula_evaluates_the_arithmetic():
    f = derived.Formula('log10([A] / [B]) + 2 * [A] ** 2 - max([A], [B])')
    a, b = np.array([10., 100.]), np.array([1., 10.])
    np.testing.assert_allclose(f({'A': a, 'B': b}),
                               np.log10(a / b) + 2 * a ** 2
                               - np.maximum(a, b))
    assert f.refs == ['A', 'B']


@pytest.mark.parametrize('text', [
    "__import__('os').getcwd()", "[A].real", "open([A])", "[A] if 1 else 0",
    "[A] < 2", "'x'", "log([A], base=2)", "max([A])", "[A", "2 + 3", "",
    "lambda: [A]"])
def test_formula_refuses_anything_but_arithmetic(text):
    with pytest.raises(derived.FormulaError):
        derived.Formula(text)


def test_an_event_with_no_finite_result_is_nan_not_a_number():
    f = derived.Formula('[A] / [B]')
    out = f({'A': np.array([0., 1., -1., 4.]), 'B': np.array([0., 0., 0., 2.])})
    assert np.isnan(out[:3]).all() and out[3] == 2.0
    out = derived.Formula('log([A])')({'A': np.array([-1., np.e])})
    assert np.isnan(out[0]) and out[1] == pytest.approx(1.0)


def test_a_marker_label_resolves_to_its_detector():
    text = derived.resolve_refs('[CD901b] / [FL2-A] + [nope]',
                                ['FL1-A', 'FL2-A'], {'FL1-A': 'CD901b'})
    assert text == '[FL1-A] / [FL2-A] + [nope]'


# ── On a loaded sample ──────────────────────────────────────────────────────

def test_ratio_is_computed_on_linear_values_and_stored_linear(tmp_path):
    from openflo.pipeline import linear_values
    p = _fcs(tmp_path / 'a.fcs', 1)
    s = _loaded(p)
    assert s.data_transforms.get('FL1-A'), 'loader should transform FL1-A'
    col = derived.add_to_sample(s, 'Ratio', '[FL1-A] / [FL2-A]')
    assert col == 'D:Ratio'
    want = linear_values(s, 'FL1-A') / linear_values(s, 'FL2-A')
    np.testing.assert_allclose(s.data[col].values, want, rtol=1e-9)
    np.testing.assert_allclose(linear_values(s, col), want, rtol=1e-9)
    assert col not in s.data_transforms
    assert s._derived_column(col)               # not clustered by default
    # Redefined over a column that carried a display transform: the record
    # goes, or the linear ratio would be read as logicle coordinates.
    from openflo.pipeline import transform_spec
    s.data_transforms[col] = transform_spec('logicle')
    derived.add_to_sample(s, 'Ratio', '[FL1-A] / [FL2-A]')
    np.testing.assert_allclose(linear_values(s, col), want, rtol=1e-9)
    assert s.derived == {col: '[FL1-A] / [FL2-A]'}
    # The step: ratio ~1 before, ~3 after.
    n = len(want) // 2
    assert np.median(want[:n]) == pytest.approx(1.0, rel=.05)
    assert np.median(want[n:]) == pytest.approx(3.0, rel=.05)


def test_apply_to_samples_checks_the_formula_before_touching_any():
    df = pd.DataFrame({'FL1-A': [1., 2.], 'FL2-A': [1., 1.]})
    a = SimpleNamespace(name='a', data=df.copy(), channel_labels={})
    with pytest.raises(derived.FormulaError):
        derived.apply_to_samples({'a': a}, 'R', '[FL1-A] / ')
    assert list(a.data.columns) == ['FL1-A', 'FL2-A']


def test_apply_to_samples_names_the_samples_it_could_not_do():
    a = SimpleNamespace(name='a', data=pd.DataFrame({'FL1-A': [2.],
                                                     'FL2-A': [1.]}),
                        channel_labels={'FL1-A': 'CD901b'})
    b = SimpleNamespace(name='b', data=pd.DataFrame({'FL2-A': [1.]}),
                        channel_labels={})
    col, ok, bad = derived.apply_to_samples({'a': a, 'b': b}, 'R',
                                            '[CD901b] / [FL2-A]')
    assert (col, ok) == ('D:R', ['a'])
    assert [n for n, _ in bad] == ['b']
    assert a.data['D:R'].tolist() == [2.0]
    assert 'D:R' not in b.data.columns


def test_restore_recomputes_a_missing_column_from_the_formula(tmp_path):
    p = _fcs(tmp_path / 'a.fcs', 2)
    s = _loaded(p)
    want = derived.compute(s, '[FL1-A] - [FL2-A]')
    s2 = _loaded(p)                         # the raw FCS again, no column
    failed = derived.restore(s2, {'D:Diff': '[FL1-A] - [FL2-A]',
                                  'D:Bad': '[FL9-A] * 2'})
    np.testing.assert_allclose(s2.data['D:Diff'].values, want)
    assert [c for c, _ in failed] == ['D:Bad']
    assert 'D:Bad' not in s2.data.columns


def test_a_derived_column_in_an_old_csv_without_a_record_is_linear():
    """A CSV with no transform record: a D: column whose values happen to
    sit in the logicle range is still linear, as OpenFlo wrote it."""
    from openflo.editor_loadpool import mark_processed_transforms
    df = pd.DataFrame({'FL1-A': [0.2, 0.5, 0.7], 'D:R': [0.2, 0.5, 0.7]})
    s = SimpleNamespace(name='old', data=df, fluor_channels=['FL1-A', 'D:R'],
                        data_transforms={})
    mark_processed_transforms(s, None, provenance='session')
    assert s.data_transforms.get('D:R', {}).get('method',
                                                'linear') == 'linear'
    assert s.data_transforms['FL1-A']['method'] != 'linear'


def test_gates_using_and_remove():
    gates = {'a': {'g1': {'x_channel': 'D:R', 'y_channel': 'FL2-A'}},
             'b': {'g1': {'kind': 'category', 'channel': 'D:R'}},
             'c': {'g1': {'x_channel': 'FL1-A', 'y_channel': 'FL2-A'}}}
    assert derived.gates_using(gates, 'D:R') == ['a', 'b']
    s = SimpleNamespace(data=pd.DataFrame({'D:R': [1.], 'FL1-A': [1.]}),
                        derived={'D:R': '[FL1-A]'})
    derived.remove_from_sample(s, 'D:R')
    assert list(s.data.columns) == ['FL1-A'] and s.derived == {}


# ── Session ─────────────────────────────────────────────────────────────────

def _restore_session(root, session_path):
    ed2 = _new_editor(root)
    jobs = {'fcs': [], 'csv': []}
    ed2._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed2._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    ed2._load_session_path(str(session_path))
    for job in jobs['fcs']:
        ed2._load_worker(ed2._sample_name_for(job), job)
        root.update()
    for job in jobs['csv']:
        ed2._load_csv_worker(job[0], job[1], {})
        root.update()
    want = len(jobs['fcs']) + len(jobs['csv'])
    assert _pump(root, lambda: len(ed2._samples) == want), \
        ed2.status_var.get()
    return ed2, jobs


@pytest.mark.parametrize('sidecar', [True, False])
def test_a_derived_channel_survives_a_session_round_trip(env, sidecar):
    """Saved with its formula; reopened from the processed sidecar, or (the
    sidecar gone) recomputed from the raw FCS."""
    root, ed, tmp = env
    p = _fcs(tmp / 'kin.fcs', 3)
    s = _loaded(p)
    ed._on_loaded('kin', s)
    derived.add_to_sample(s, 'Ratio', '[FL1-A] / [FL2-A]')
    want = s.data['D:Ratio'].to_numpy(copy=True)
    path = tmp / 'kin.flowsession'
    data = ed._write_session(str(path))
    entry = data['samples'][0]
    assert entry['derived'] == {'D:Ratio': '[FL1-A] / [FL2-A]'}
    if not sidecar:
        side = entry.get('processed_data')
        assert side, entry
        side_path = side if os.path.isabs(side) else os.path.join(tmp, side)
        os.remove(side_path)
    ed2, jobs = _restore_session(root, path)
    if not sidecar:
        assert jobs['fcs'] and not jobs['csv'], jobs
    s2 = ed2._samples['kin']
    assert 'D:Ratio' in s2.data.columns
    np.testing.assert_allclose(s2.data['D:Ratio'].values, want, rtol=1e-5)
    assert s2.derived == {'D:Ratio': '[FL1-A] / [FL2-A]'}


# ── Windows ─────────────────────────────────────────────────────────────────

def test_derived_window_adds_to_every_sample_and_refuses_a_gated_removal(
        env, monkeypatch):
    from tkinter import messagebox

    from openflo.ui_derived import DerivedChannelsWindow
    root, ed, tmp = env
    for nm, seed in (('A', 1), ('B', 2)):
        ed._on_loaded(nm, _loaded(_fcs(tmp / f'{nm}.fcs', seed)))
    win = DerivedChannelsWindow(ed)
    win.name_var.set('Ratio')
    win.formula_var.set('[CD901b] / [FL2-A]')
    win._add()
    assert 'D:Ratio' in ed._channels, win.status_var.get()
    for nm in ('A', 'B'):
        assert 'D:Ratio' in ed._samples[nm].data.columns
    assert win.tv.exists('D:Ratio')
    ed._sample_gates['B'] = {'g': {'x_channel': 'D:Ratio',
                                   'y_channel': 'FL2-A'}}
    win.tv.selection_set('D:Ratio')
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: True)
    win._remove()
    assert 'D:Ratio' in ed._samples['A'].data.columns
    assert 'B' in win.status_var.get()
    ed._sample_gates['B'] = {}
    win.tv.selection_set('D:Ratio')
    win._remove()
    assert 'D:Ratio' not in ed._channels
    for nm in ('A', 'B'):
        assert 'D:Ratio' not in ed._samples[nm].data.columns
    win.formula_var.set('[FL1-A] /')
    win._add()
    assert "Can't read" in win.status_var.get()
    win.destroy()


# ── Kinetics ────────────────────────────────────────────────────────────────

def test_trace_gives_the_median_of_each_bin():
    t = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
    v = np.array([1., 3., 2., 10., 20., 30., 5., np.nan])
    c, med, n = kinetics.trace(t, v, bins=2, t_range=(0, 8))
    np.testing.assert_allclose(c, [2., 6.])
    np.testing.assert_allclose(med, [2.5, 20.])     # 1,3,2,10 / 20,30,5
    assert n.tolist() == [4, 3]
    _c, mean, _n = kinetics.trace(t, v, bins=2, stat='mean', t_range=(0, 8))
    np.testing.assert_allclose(mean, [4., 55 / 3])
    _c, med, n = kinetics.trace(t, v, bins=4, t_range=(0, 16))
    assert n.tolist() == [4, 3, 0, 0] and np.isnan(med[2:]).all()
    _c, _m, n = kinetics.trace(t, v, bins=2, t_range=(1, 3))
    assert n.tolist() == [1, 2]                 # events outside left out


def test_summary_on_a_hand_worked_trace():
    t = np.arange(10, dtype=float)
    v = np.array([5., 1, 1, 1, 2, 3, 4, 3, 2, 1])
    r = kinetics.summarize(t, v, baseline_end=4., bins=10, t_range=(0, 10))
    assert r['n_baseline'] == 4                 # t < 4, not t <= 4
    assert r['baseline'] == 1.0                 # median of 5, 1, 1, 1
    assert r['peak'] == 4.0                     # not the 5 before it
    assert r['time_to_peak'] == pytest.approx(2.5)    # 6.5 - 4
    assert r['auc'] == pytest.approx(8.5)       # trapezoids of 1,2,3,2,1,0
    assert r['fold'] == pytest.approx(4.0)
    r = kinetics.summarize(t, v, baseline_end=4., stimulus=9.2, bins=10,
                           t_range=(0, 10))
    assert r['peak'] == 1.0 and np.isnan(r['auc'])    # one bin: no width


def _step(base=100., level=400., t_stim=20., t_end=60., n=6000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, t_end, n))
    v = np.where(t < t_stim, base, level) + rng.normal(0, 1, n)
    return t, v


def test_summary_reads_baseline_peak_time_and_area():
    t, v = _step()
    r = kinetics.summarize(t, v, baseline_end=20., bins=60, t_range=(0, 60))
    assert r['baseline'] == pytest.approx(100., abs=.5)
    assert r['peak'] == pytest.approx(400., abs=2)
    assert 0 <= r['time_to_peak'] <= 40
    # 300 above baseline from 20.5 (first bin centre) to 59.5: 39 s.
    assert r['auc'] == pytest.approx(300 * 39, rel=.01)
    assert r['fold'] == pytest.approx(4.0, rel=.01)
    assert r['n_baseline'] == int((t < 20).sum())


def test_summary_time_to_peak_counts_from_the_stimulus():
    t = np.arange(0, 100, .01)
    v = np.where((t > 70) & (t < 72), 50., 1.)
    r = kinetics.summarize(t, v, baseline_end=30., stimulus=40., bins=100,
                           t_range=(0, 100))
    assert r['peak'] == 50.
    assert r['time_to_peak'] == pytest.approx(30.5, abs=1)


def test_summary_is_nan_where_it_cannot_be_computed():
    t, v = _step(base=0.)
    r = kinetics.summarize(t, v, baseline_end=20., bins=60, t_range=(0, 60))
    assert np.isnan(r['fold'])                      # baseline 0
    assert np.isfinite(r['auc'])
    r = kinetics.summarize(t, v, baseline_end=-1., bins=60, t_range=(0, 60))
    assert np.isnan(r['baseline']) and np.isnan(r['auc'])
    assert r['n_baseline'] == 0
    r = kinetics.summarize(t, v, baseline_end=10., stimulus=99., bins=60,
                           t_range=(0, 60))
    assert np.isnan(r['peak']) and np.isnan(r['time_to_peak'])


def test_time_is_in_seconds_when_the_file_gives_a_timestep(tmp_path):
    s = _loaded(_fcs(tmp_path / 'a.fcs', 1, timestep='0.01'))
    t, unit = kinetics.sample_time(s)
    assert unit == 's'
    raw = s.data['Time'].to_numpy(dtype=float)
    np.testing.assert_allclose(t, raw * 0.01)
    assert t.max() == pytest.approx(11.99, abs=.02)
    s = _loaded(_fcs(tmp_path / 'b.fcs', 1, timestep=None))
    t, unit = kinetics.sample_time(s)
    assert unit == 'ticks' and t.max() == pytest.approx(1199, abs=2)


def test_a_sample_without_time_is_refused():
    s = SimpleNamespace(name='x', data=pd.DataFrame({'FL1-A': [1.]}),
                        metadata={})
    with pytest.raises(ValueError):
        kinetics.sample_time(s)


def test_kinetics_window_reads_each_sample_and_leaves_out_one_without_time(
        env):
    from openflo.ui_kinetics import KineticsWindow
    root, ed, tmp = env
    for nm, seed in (('A', 1), ('B', 2)):
        s = _loaded(_fcs(tmp / f'{nm}.fcs', seed))
        derived.add_to_sample(s, 'Ratio', '[FL1-A] / [FL2-A]')
        ed._on_loaded(nm, s)
    ed._refresh_channel_choices()
    s = _loaded(_fcs(tmp / 'C.fcs', 3))
    derived.add_to_sample(s, 'Ratio', '[FL1-A] / [FL2-A]')
    s.data = s.data.drop(columns=['Time'])
    ed._on_loaded('C', s)
    win = KineticsWindow(ed)
    win.chan_var.set(ed._fmt_channel('D:Ratio'))
    win.base_var.set('5')
    win.bins_var.set('24')
    win._refresh()
    rows = {r['Sample']: r for r in win._results}
    assert set(rows) == {'A', 'B'}
    assert 'C (no Time channel)' in win.status_var.get()
    for r in rows.values():
        assert r['unit'] == 's'
        assert r['baseline'] == pytest.approx(1.0, rel=.05)
        assert r['peak'] == pytest.approx(3.0, rel=.1)
        assert r['fold'] == pytest.approx(3.0, rel=.1)
        # The step is at 6 s, 1 s after the baseline ends (the stimulus).
        assert 0.7 <= r['time_to_peak'] <= 7.0
        assert r['stimulus'] == 5.0
    assert len(win.tv.get_children()) == 2
    lines = win._table_lines()
    assert lines[0][3] == 'Time to peak (s)' and len(lines) == 3
    # A measured channel is read on its linear values, not the display.
    win.chan_var.set(ed._fmt_channel('FL1-A'))
    win._refresh()
    for r in win._results:
        assert r['fold'] == pytest.approx(3.0, rel=.15)
    win.destroy()
