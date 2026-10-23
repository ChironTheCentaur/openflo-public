"""Gate counts and statistics saved with the session (saved_stats).

A reopened session must show the counts and statistics it saved without
computing them again, and must never show a saved number once anything it
depends on has changed: a gate or one of its ancestors, the data
(compensation, transform, events), a Preference that changes a formula, or
the OpenFlo version. Every number served has to equal a fresh computation.

All data is synthetic; channels are FSC-A, SSC-A, FL1-A, ...
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from openflo import saved_stats as ss  # noqa: E402

WANT = {'Count', '%Parent', '%Grandparent', '%Total', 'Median', 'Mean',
        'GeoMean', 'CV', 'rCV', 'P5', 'P95'}
CHANS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'FSC-A': rng.normal(100.0, 30.0, n),
        'SSC-A': rng.normal(50.0, 20.0, n),
        'FL1-A': rng.lognormal(5.0, 1.0, n),
        'FL2-A': rng.lognormal(4.0, 1.2, n),
        'FL3-A': rng.normal(300.0, 80.0, n),
    })


def _gate_list(df):
    """Five gates in a chain, at the data's own quantiles (so they hold
    events on whatever scale the data is stored), plus one that holds none."""
    q = {c: np.nanquantile(df[c].to_numpy(), [0.2, 0.5, 0.8]) for c in df}
    th = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    fx, sy = q['FSC-A'], q['SSC-A']
    poly = [[float(fx[1] + (fx[2] - fx[0]) * np.cos(t)),
             float(sy[1] + (sy[2] - sy[0]) * np.sin(t))] for t in th]
    return [
        {'kind': 'polygon', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
         'vertices': poly, 'parent': None, 'name': 'Cells'},
        {'kind': 'threshold', 'channel': 'FL1-A', 'value': float(q['FL1-A'][1]),
         'parent': 0, 'name': 'FL1pos'},
        {'kind': 'threshold', 'channel': 'FL2-A', 'value': float(q['FL2-A'][1]),
         'parent': 1, 'name': 'FL2pos'},
        {'kind': 'interval', 'channel': 'FL3-A', 'lo': float(q['FL3-A'][0]),
         'hi': float(q['FL3-A'][2]), 'parent': 1, 'name': 'FL3mid'},
        {'kind': 'threshold', 'channel': 'FL3-A',
         'value': float(np.nanmax(df['FL3-A'].to_numpy())) + 1.0,
         'parent': 2, 'name': 'Empty'},
    ]


def _add_gates(ed, name):
    ed._set_active_sample(name)
    ids = []
    for g in _gate_list(ed._samples[name].data):
        g = dict(g)
        p = g.pop('parent')
        g['parent_id'] = None if p is None else ids[p]
        ids.append(ed._add_gate(g, replace_1d=False))
    return ids


def _same(a, b):
    """Equal, NaN equal to NaN, key order included."""
    if isinstance(a, dict):
        return (isinstance(b, dict) and list(a) == list(b)
                and all(_same(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b, strict=True)))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def _rows(ed, want=WANT):
    return ed._stats_rows_from_snapshot(ed._stats_snapshot(want), want)[0]


def _fresh_rows(ed, want=WANT):
    """The rows computed now, past every cache."""
    rows = []
    for args in ed._stats_snapshot(want):
        rows.extend(ed._population_stats(*args))
    return rows


def _fresh_counts(ed, name):
    from openflo.gating import gate_counts
    s = ed._samples[name]
    gates = ed._sample_gates[name]
    return {gid: gate_counts(gates, gid, s.data,
                             ed._autoclean_overrides(name, s.data))[0]
            for gid in gates}


def _counters(monkeypatch, fail=False):
    """Count (or refuse) every gate scan and statistics computation."""
    import openflo.gating as gating
    import openflo.plotcache as pcm
    calls = {'scan': [], 'stats': 0}
    real_scan, real_stats = pcm.scan, gating.population_stats

    def scan(*a, **k):
        if fail:
            raise AssertionError("gate counts were computed again")
        calls['scan'].append(list(k.get('count_gids') or ()))
        return real_scan(*a, **k)

    def stats(*a, **k):
        if fail:
            raise AssertionError("statistics were computed again")
        calls['stats'] += 1
        return real_stats(*a, **k)
    monkeypatch.setattr(pcm, 'scan', scan)
    monkeypatch.setattr(gating, 'population_stats', stats)
    return calls


# ── pure: rows, keys, entries ───────────────────────────────────────────────

def test_rows_come_back_from_their_saved_numbers():
    from openflo.gating import CHANNEL_STAT_ORDER, population_stats
    df = _frame(3000, 1)
    gates, ids = {}, []
    for i, g in enumerate(_gate_list(df)):
        g = dict(g)
        p = g.pop('parent')
        g['parent_id'] = None if p is None else ids[p]
        ids.append(f'g{i + 1}')
        gates[ids[-1]] = g
    labels = {c: c for c in CHANS}
    chans = ['FL1-A', 'FL2-A', 'FL3-A']
    rows = population_stats('s1', df, gates, ids, labels, chans, WANT,
                            CHANNEL_STAT_ORDER)
    rows[1]['CV FL1-A'] = float('inf')         # JSON has no infinity either
    assert any(math.isnan(v) for v in rows[-1].values()
               if isinstance(v, float))         # the empty gate's NaNs
    enc = json.loads(json.dumps(ss.encode_rows(rows), allow_nan=False))
    back = ss.decode_rows(enc, 's1', gates, labels, None, df.columns)
    assert _same(back, rows)
    # Names are not saved: a renamed gate is served under its new name.
    gates['g2']['name'] = 'Renamed'
    back = ss.decode_rows(enc, 's1', gates, labels, None, df.columns)
    assert 'Renamed' in back[1]['Population']
    del gates['g5']
    assert ss.decode_rows(enc, 's1', gates, labels, None, df.columns) is None


def _key(**over):
    base = dict(sigs={'g1': 'a', 'g2': 'b'}, order=['g1', 'g2'], select=None,
                labels={'FL1-A': 'FL1-A'}, channels=['FL1-A'],
                want={'Median'}, stat_chan=('Median',), transforms={},
                settings={}, version='9.9.9')
    base.update(over)
    return ss.stats_key(**base)


@pytest.mark.parametrize('over', [
    {'sigs': {'g1': 'a', 'g2': 'c'}}, {'order': ['g2', 'g1']},
    {'select': ['g2']}, {'labels': {'FL1-A': 'Marker'}},
    {'channels': ['FL1-A', 'FL2-A']}, {'want': {'Median', 'P5'}},
    {'transforms': {'FL1-A': {'method': 'asinh', 'cofactor': 150.0}}},
    {'settings': {'stats_option': 1}}, {'version': '9.9.10'},
])
def test_every_input_of_the_statistics_changes_their_key(over):
    assert _key(**over) != _key()


def test_a_stats_preference_is_one_name_away(monkeypatch):
    assert 'stats_option' not in ss.stats_settings_key({'stats_option': 1})
    monkeypatch.setattr(ss, 'STATS_PREF_KEYS', ('stats_option',))
    assert ss.stats_settings_key({'stats_option': 1}) == {'stats_option': 1}
    assert ss.stats_settings_key(None) == {'stats_option': None}


def test_an_entry_is_used_only_by_its_version_on_its_data():
    e = ss.make_entry('9.9.9', 'd1', {'sig': 5}, {})
    assert ss.usable(e, 'd1', '9.9.9')
    assert not ss.usable(e, 'd2', '9.9.9')
    assert not ss.usable(e, 'd1', '9.9.10')
    assert not ss.usable(e, None, '9.9.9')
    assert not ss.usable(None, 'd1', '9.9.9')
    assert ss.entry_counts({'counts': {'a': 3, 'b': -1, 'c': 'x',
                                       'd': True}}) == {'a': 3}
    assert ss.entry_stats({'stats': {'k': [['g1', {}]], 'bad': [['g1']]}}) \
        == {'k': [['g1', {}]]}


def test_v2_session_saved_no_numbers():
    from openflo.session_format import SESSION_FORMAT, migrate_session
    data, _ = migrate_session({'format': SESSION_FORMAT, 'version': 2,
                               'samples': [{'name': 's1'}],
                               'sample_gates': {}})
    assert data['samples'][0]['saved_stats'] is None


# ── in an editor ────────────────────────────────────────────────────────────

@pytest.fixture
def tk_root(tmp_path, monkeypatch):
    from tests.test_transform_records import capture_warnings, isolate_home
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    import openflo.async_task as _at

    def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
        try:
            res = work()
        except Exception as exc:                  # noqa: BLE001
            if on_error:
                on_error(exc)
            else:
                raise
        else:
            if on_done:
                on_done(res)
        finally:
            if on_finally:
                on_finally()
    monkeypatch.setattr(_at, 'run_async', _inline)
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                        # noqa: BLE001
        from tests.conftest import gui_unavailable
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    yield root
    root.destroy()


def _editor(root):
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    ed._pc_warm_enabled = False
    return ed


def _pump(root, done, timeout=30.0):
    from tests.test_transform_records import _pump as pump
    return pump(root, done, timeout)


def _add_frame(ed, name, df):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(df, name=name, path=f'{name}.csv')
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(df.columns)
    ed._channel_labels = {c: c for c in df.columns}
    ed._populate_channel_combos()
    return s


def _load_fcs(ed, root, path):
    name = ed._sample_name_for(path)
    ed._load_worker(name, path)
    assert _pump(root, lambda: name in ed._samples)
    return name


@pytest.fixture
def ed(tk_root):
    e = _editor(tk_root)
    _add_frame(e, 'A', _frame(4000, 1))
    _add_frame(e, 'B', _frame(3000, 2))
    for name in ('A', 'B'):
        _add_gates(e, name)
    return e


def _saved_entry(ed, name):
    """What a save writes for the sample, through JSON."""
    ed._pc_counts(name, list(ed._sample_gates[name]))
    _rows(ed)
    entry = ed._saved_stats_finish(ed._saved_stats_plan(name))
    return json.loads(json.dumps(entry, allow_nan=False))


def _reopen(ed, name, entry):
    """As a reopen does: nothing cached, then the saved numbers."""
    ed._pc_forget(name)
    return ed._saved_stats_restore(name, entry)


def _restore_session(root, path):
    ed2 = _editor(root)
    jobs = {'fcs': [], 'csv': []}
    ed2._queue_fcs_loads = lambda paths, front_names=(): jobs['fcs'].extend(
        paths)
    ed2._queue_processed_loads = lambda items, front_names=(): jobs[
        'csv'].extend(items)
    ed2._load_session_path(str(path))
    for p in jobs['fcs']:
        ed2._load_worker(ed2._sample_name_for(p), p)
    for job in jobs['csv']:
        ed2._load_csv_worker(job[0], job[1], {})
    n = len(jobs['fcs']) + len(jobs['csv'])
    assert _pump(root, lambda: len(ed2._samples) == n)
    return ed2


@pytest.mark.parametrize('source', ['fcs', 'sidecar'])
def test_a_reopened_session_shows_its_saved_numbers_without_computing(
        tk_root, tmp_path, monkeypatch, source):
    ed1 = _editor(tk_root)
    if source == 'fcs':
        from openflo.synthetic import _write_fcs
        for i in range(2):
            _write_fcs(str(tmp_path / f'sample_{i}.fcs'), _frame(4000, i))
        names = [_load_fcs(ed1, tk_root, str(tmp_path / f'sample_{i}.fcs'))
                 for i in range(2)]
    else:
        names = ['A', 'B']
        for i, name in enumerate(names):
            _add_frame(ed1, name, _frame(4000, i))
    for name in names:
        _add_gates(ed1, name)
        ed1._pc_counts(name, list(ed1._sample_gates[name]))
    rows1 = _rows(ed1)
    counts1 = {n: ed1._pc_peek_counts(n, list(ed1._sample_gates[n]))
               for n in names}
    path = tmp_path / 'one.flowsession'
    ed1._write_session(str(path))
    saved = json.loads(path.read_text())
    for entry in saved['samples']:
        st = entry['saved_stats']
        assert set(st) == {'app_version', 'data', 'counts', 'stats'}
        assert len(st['counts']) == 5 and len(st['stats']) == 1

    ed2 = _restore_session(tk_root, path)
    _counters(monkeypatch, fail=True)            # nothing may be computed
    for n in names:
        assert ed2._pc_counts(n, list(ed2._sample_gates[n])) == counts1[n]
    rows2 = _rows(ed2)
    assert _same(rows2, rows1)
    monkeypatch.undo()
    assert _same(rows2, _fresh_rows(ed2))
    for n in names:
        assert counts1[n] == _fresh_counts(ed2, n)


def test_the_statistics_window_shows_saved_rows_without_computing(
        ed, monkeypatch):
    from openflo.ui_statistics import StatisticsWindow
    w = StatisticsWindow(ed)                      # computes, and keeps
    first = [dict(r) for r in w._rows]
    w.destroy()
    entries = {n: _saved_entry(ed, n) for n in ('A', 'B')}
    for n, entry in entries.items():
        assert _reopen(ed, n, entry)
    _counters(monkeypatch, fail=True)
    w = StatisticsWindow(ed)
    try:
        assert w._rows and _same(w._rows, first)
    finally:
        w.destroy()


def _edit_gate(ed, which):
    gates = ed._sample_gates['A']
    ids = ed._sample_gate_order['A']
    if which == 'gate':
        g = gates[ids[3]]                         # a leaf: FL3mid
        g['hi'] = g['lo'] + (g['hi'] - g['lo']) / 2
    else:
        gates[ids[0]]['vertices'][0] = [0.0, 0.0]  # the root everything is in


@pytest.mark.parametrize('change', ['gate', 'parent', 'compensation',
                                    'transform', 'events', 'preference',
                                    'version'])
def test_any_change_misses_and_recomputes(ed, monkeypatch, change):
    entry = _saved_entry(ed, 'A')
    ids = list(ed._sample_gate_order['A'])
    s = ed._samples['A']
    if change in ('gate', 'parent'):
        _edit_gate(ed, change)
    elif change == 'compensation':
        s._apply_comp(np.array([[1.0, 0.2], [0.0, 1.0]]), ['FL1-A', 'FL2-A'])
        ed._data_changed('A')
    elif change == 'transform':
        ed._apply_channel_transforms({'FL1-A': 'asinh'})
    elif change == 'events':
        s.data = s.data.iloc[::2].reset_index(drop=True)
    elif change == 'preference':
        import openflo.prefs as prefs
        monkeypatch.setattr(ss, 'STATS_PREF_KEYS', ('stats_option',))
        monkeypatch.setattr(prefs, 'read_prefs',
                            lambda: {'stats_option': 'other'})
    elif change == 'version':
        import openflo.result_stamp as rs
        monkeypatch.setattr(rs, 'app_version', lambda: '999.0.0')
    calls = _counters(monkeypatch)
    restored = _reopen(ed, 'A', entry)
    counts = ed._pc_counts('A', ids)
    rows = [r for r in _rows(ed) if r['Sample'] == 'A']
    assert calls['stats'] >= 1                    # the statistics missed
    recounted = {g for c in calls['scan'] for g in c}
    if change == 'gate':                          # only that gate's count
        assert recounted == {ids[3]}
    elif change == 'parent':                      # all of them
        assert recounted == set(ids)
    elif change == 'preference':                  # counts don't depend on it
        assert recounted == set()
    else:                                         # nothing saved was used
        assert restored == 0 and recounted == set(ids)
    monkeypatch.setattr(ss, 'STATS_PREF_KEYS', ss.STATS_PREF_KEYS)
    assert counts == _fresh_counts(ed, 'A')
    assert _same(rows, [r for r in _fresh_rows(ed) if r['Sample'] == 'A'])


def test_a_rename_still_serves_the_saved_numbers(ed, monkeypatch):
    entry = _saved_entry(ed, 'A')
    gid = ed._sample_gate_order['A'][1]
    ed._sample_gates['A'][gid]['name'] = 'Renamed'
    _reopen(ed, 'A', entry)
    _counters(monkeypatch, fail=True)
    rows = [r for r in _rows(ed) if r['Sample'] == 'A']
    assert any('Renamed' in r['Population'] for r in rows)
    monkeypatch.undo()
    assert _same(rows, [r for r in _fresh_rows(ed) if r['Sample'] == 'A'])


def test_statistic_form_preferences_key_saved_numbers():
    """Switching GeoMean or rSD/rCV to the other form in Preferences changes
    the key, so a number saved under one form is never served for the
    other."""
    base = {'stat_geomean': 'display', 'stat_rcv': 'percentile'}
    key = ss.stats_settings_key(base)
    assert ss.stats_settings_key({**base, 'stat_geomean': 'log'}) != key
    assert ss.stats_settings_key({**base, 'stat_rcv': 'mad'}) != key
