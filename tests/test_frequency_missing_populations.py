"""A population a sample does not have is MISSING, never 0 events -- and
fewer populations go missing in the first place.

Two defects, one shape:

* Populations were lined up across samples by their displayed path, and an
  unnamed gate's path carries its coordinates ('T  CD3-A >= 0.453'). A cut
  nudged in one sample, or a marker read on another detector in a second
  panel, split one population into several, each present in a single sample.
* The differential-abundance table then filled every gap with a count of 0,
  so a gate drawn only in one group's samples read as a population that had
  vanished from the other group.

Populations are now matched by what the gate is (gating.population_keys);
what still has no counterpart is shown, tested and exported as missing.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_frequency_window import DESIGN, N, _inline  # noqa: E402
from test_gui_smoke import _editor_or_skip  # noqa: E402

from openflo.diffexp import differential_abundance  # noqa: E402
from openflo.gating import population_keys, population_stats  # noqa: E402

# ── the identity key ────────────────────────────────────────────────────────

def test_a_nudged_unnamed_gate_keeps_its_identity():
    a = {'g1': {'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.453,
                'parent_id': None}}
    b = {'x9': {'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.471,
                'parent_id': None}}
    assert population_keys(a)['g1'] == population_keys(b)['x9']


def test_a_marker_on_another_detector_keeps_its_identity():
    """Panel 1 reads CD901b on FL1, panel 2 on FL7: one population."""
    a = {'g': {'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FSC-A',
               'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1, 'parent_id': None}}
    b = {'g': {'kind': 'rect', 'x_channel': 'FL7-A', 'y_channel': 'FSC-A',
               'x0': 0.1, 'x1': 1, 'y0': 0, 'y1': 1, 'parent_id': None}}
    ka = population_keys(a, {'FL1-A': 'CD901b'})['g']
    kb = population_keys(b, {'FL7-A': 'CD901b'})['g']
    assert ka == kb
    # ...but not when the markers differ
    assert ka != population_keys(b, {'FL7-A': 'CD14'})['g']


def test_sibling_bands_stay_distinct_and_names_are_normalised():
    gates = {
        'hi': {'kind': 'interval', 'channel': 'CD3-A', 'lo': 0.6, 'hi': 1e12,
               'parent_id': None},
        'lo': {'kind': 'interval', 'channel': 'CD3-A', 'lo': -1e12, 'hi': 0.2,
               'parent_id': None},
        'mid': {'kind': 'interval', 'channel': 'CD3-A', 'lo': 0.2, 'hi': 0.6,
                'parent_id': None},
        'n': {'kind': 'threshold', 'channel': 'CD4-A', 'value': 1.0,
              'parent_id': 'hi', 'name': ' CD4  Pos '}}
    k = population_keys(gates)
    assert len({k['lo'], k['mid'], k['hi']}) == 3
    # ranked low to high, so the same three bands match in another sample
    assert k['lo'].endswith('#1') and k['hi'].endswith('#3')
    other = {'h': dict(gates['hi'], parent_id=None),
             'n2': dict(gates['n'], parent_id='h', name='cd4 pos')}
    other['l'] = dict(gates['lo'])
    other['m'] = dict(gates['mid'])
    assert population_keys(other)['n2'] == k['n']


def test_an_empty_parent_gives_no_fraction_not_zero():
    df = pd.DataFrame({'CD3-A': np.zeros(50), 'CD4-A': np.ones(50)})
    gates = {'a': {'kind': 'threshold', 'channel': 'CD3-A', 'value': 5.0,
                   'parent_id': None, 'name': 'CD3pos'},
             'b': {'kind': 'threshold', 'channel': 'CD4-A', 'value': 0.5,
                   'parent_id': 'a', 'name': 'CD4pos'}}
    rows = population_stats('s', df, gates, ['a', 'b'], {}, [],
                            {'Count', '%Parent'}, ())
    by = {r['Population']: r for r in rows}
    assert by['CD3pos']['%Parent'] == 0.0           # a measured 0 of 50
    assert math.isnan(by['CD3pos/CD4pos']['%Parent'])   # 0 of 0: undefined


# ── differential abundance with missing samples ─────────────────────────────

def test_a_missing_count_is_fitted_on_the_samples_that_have_it():
    rng = np.random.default_rng(0)
    lib = np.full(8, 10_000.0)
    group = ['A'] * 4 + ['B'] * 4
    Y = rng.poisson(lib * np.array([[0.10] * 4 + [0.20] * 4,
                                    [0.30] * 8]))
    Yn = Y.astype(float)
    Yn[0, 1] = np.nan                    # population 0 missing in one A sample
    rows = {r['cluster']: r for r in differential_abundance(
        Yn, group, lib_sizes=lib, cluster_names=['p0', 'p1'])}
    keep = [0, 2, 3, 4, 5, 6, 7]
    ref = differential_abundance(Y[[0]][:, keep], [group[i] for i in keep],
                                 lib_sizes=lib[keep], cluster_names=['p0'])[0]
    for k in ('log2fc', 'z', 'p', 'dispersion', 'df', 'prop_a', 'prop_b'):
        assert rows['p0'][k] == pytest.approx(ref[k], rel=1e-9, nan_ok=True), k
    assert rows['p0']['n_a'] == 3 and rows['p0']['n_missing'] == 1
    assert rows['p1']['n_missing'] == 0 and rows['p1']['n_a'] == 4


def test_a_population_missing_from_a_whole_group_is_not_a_loss():
    Y = np.array([[50.0, 60.0, 55.0, np.nan, np.nan, np.nan]])
    r = differential_abundance(Y, ['A'] * 3 + ['B'] * 3,
                               lib_sizes=[1000] * 6)[0]
    assert math.isnan(r['p']) and math.isnan(r['log2fc'])
    assert math.isnan(r['prop_b']) and r['n_b'] == 0 and r['n_missing'] == 3


# ── the Frequencies window ──────────────────────────────────────────────────

def _add(ed, name, trial, gates, cols):
    from openflo.pipeline import FlowSample
    s = FlowSample.from_dataframe(pd.DataFrame(cols), name=name)
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = gates
    ed._sample_gate_order[name] = list(gates)
    ed._sample_gate_seq[name] = len(gates)


def _cols(a, b, c, seed):
    cd3, cd4, foxp3 = np.zeros(N), np.zeros(N), np.zeros(N)
    cd3[:a] = 1.0
    cd4[:b] = 1.0
    foxp3[:c] = 1.0
    perm = np.random.default_rng(seed).permutation(N)
    return {'CD3-A': cd3[perm], 'CD4-A': cd4[perm], 'FOXP3-A': foxp3[perm]}


def _window(monkeypatch, build):
    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *a, **k: None)
    build(ed)
    ed._channels = ['CD3-A', 'CD4-A', 'FOXP3-A']
    ed._channel_labels = {c: c for c in ed._channels}
    ed._active_sample = ed._sample_order[0]
    from openflo.ui_frequency import FrequencyComparisonWindow
    w = FrequencyComparisonWindow(ed)
    w.withdraw()
    w._collect(w._after_collect)
    w.factor_var.set('Name token')
    w.tokens_var.set('Stim, Ctrl')
    w.metric_var.set('%Parent')
    w._rebuild()                   # the grouping above takes effect
    return root, w


def test_unnamed_gates_nudged_per_sample_are_one_population(monkeypatch):
    def build(ed):
        for i, (nm, tr, a, b) in enumerate(DESIGN):
            cut = 0.5 + 0.01 * i            # each sample's cut drawn anew
            _add(ed, nm, tr, {'g1': {'kind': 'threshold', 'channel': 'CD3-A',
                                     'value': cut, 'parent_id': None,
                                     'enabled': True}},
                 _cols(a, b, 0, i))
    root, w = _window(monkeypatch, build)
    try:
        tidy = w._tidy_frame()
        assert tidy['Population'].nunique() == 1, (
            'one gate, nudged per sample, split into '
            f"{tidy['Population'].nunique()} populations")
        assert not tidy['Missing'].any()
        pop = tidy['Population'].iloc[0]
        w.pop_var.set(pop)
        w._rebuild()
        assert w._last_res['groups']['Stim']['n'] == 3
        assert w._last_res['groups']['Stim']['mean'] == pytest.approx(62.0)
        assert 'Matched across samples by gate structure' in \
            w._summary.get('1.0', 'end')
    finally:
        root.destroy()


def _tregs_build(ctrl_with_tregs=()):
    """CD3pos in every sample; Tregs (FOXP3+ inside CD3pos) drawn in every
    Stim sample and only in the Ctrl samples named."""
    def build(ed):
        for i, (nm, tr, a, b) in enumerate(DESIGN):
            gates = {'g1': {'kind': 'threshold', 'channel': 'CD3-A',
                            'value': 0.5, 'parent_id': None,
                            'name': 'CD3pos', 'enabled': True}}
            if nm.startswith('Stim') or nm in ctrl_with_tregs:
                gates['g2'] = {'kind': 'threshold', 'channel': 'FOXP3-A',
                               'value': 0.5, 'parent_id': 'g1',
                               'name': 'Tregs', 'enabled': True}
            _add(ed, nm, tr, gates, _cols(a, b, 50 + 2 * i, i))
    return build


def test_a_population_absent_from_a_group_is_missing_not_zero(monkeypatch):
    root, w = _window(monkeypatch, _tregs_build())
    try:
        tidy = w._tidy_frame()
        tre = tidy[tidy['Population'] == 'CD3pos/Tregs']
        miss = tre[tre['Missing']]
        assert sorted(miss['Sample']) == ['Ctrl_1', 'Ctrl_2', 'Ctrl_3']
        assert miss['Count'].isna().all() and miss['%Parent'].isna().all()

        w.pop_var.set('CD3pos/Tregs')
        w._rebuild()
        # Ctrl has no values (not three zeros), so there is nothing to test.
        assert w._last_res['groups'].get('Ctrl', {'n': 0})['n'] == 0
        assert not w._last_res.get('test')
        text = w._summary.get('1.0', 'end')
        assert 'Missing in Ctrl' in text and 'Ctrl_2' in text

        import openflo.ui_diff as ud
        seen = {}
        monkeypatch.setattr(ud, 'DiffAbundanceWindow',
                            lambda parent, rows, disp: seen.update(rows=rows))
        w._diff_abundance()
        r = {x['cluster']: x for x in seen['rows']}['CD3pos/Tregs']
        # Filled with 0 this read as a complete, highly significant loss.
        assert math.isnan(r['p']) and r['n_missing'] == 3

        # the exported tidy table says so too
        out = w._tidy_frame()
        assert set(out.columns) >= {'Missing'}
    finally:
        root.destroy()


def test_a_population_missing_from_some_samples_uses_the_rest(monkeypatch):
    root, w = _window(monkeypatch, _tregs_build(('Ctrl_1', 'Ctrl_3')))
    try:
        import openflo.ui_diff as ud
        seen = {}
        monkeypatch.setattr(ud, 'DiffAbundanceWindow',
                            lambda parent, rows, disp: seen.update(rows=rows))
        w._diff_abundance()
        r = {x['cluster']: x for x in seen['rows']}['CD3pos/Tregs']
        assert r['n_missing'] == 1 and r['n_a'] + r['n_b'] == 5
        assert np.isfinite(r['p'])

        w.pop_var.set('CD3pos/Tregs')
        w._rebuild()
        assert w._last_res['groups']['Ctrl']['n'] == 2
    finally:
        root.destroy()
