"""The Population-frequencies window computes the numbers it shows.

Until now the window (now ui_frequency.FrequencyPanel) was only ever
CONSTRUCTED on an empty editor (test_dialog_tooltips), so none of its data path
ran: collect -> tidy frame -> grouping -> compare_groups -> Prism exports ->
differential abundance -> compare-all. Here six synthetic samples carry gates
whose counts are fixed by construction, so every number has a hand-computed
answer:

    sample  trial  CD3+ (a)  CD4+ inside CD3+ (b)     N = 1000 events each
    Stim_1  D1     600       300
    Ctrl_1  D1     400       100
    Stim_2  D2     620       310
    Ctrl_2  D2     380        95
    Stim_3  D3     640       330
    Ctrl_3  D3     420       105
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

DESIGN = [('Stim_1', 'D1', 600, 300), ('Ctrl_1', 'D1', 400, 100),
          ('Stim_2', 'D2', 620, 310), ('Ctrl_2', 'D2', 380, 95),
          ('Stim_3', 'D3', 640, 330), ('Ctrl_3', 'D3', 420, 105)]
N = 1000


def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        res = work()
    except Exception as exc:                      # noqa: BLE001 - stands in for async_task.run_async, which hands any error from the work to on_error
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(res)
    finally:
        if on_finally:
            on_finally()


def _add(ed, name, trial, a, b, seed):
    from openflo.pipeline import FlowSample
    cd3, cd4 = np.zeros(N), np.zeros(N)
    cd3[:a] = 1.0
    cd4[:b] = 1.0
    perm = np.random.default_rng(seed).permutation(N)
    s = FlowSample.from_dataframe(
        pd.DataFrame({'CD3-A': cd3[perm], 'CD4-A': cd4[perm]}), name=name)
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = trial
    if trial not in ed._trial_order:
        ed._trial_order.append(trial)
    ed._sample_plot_enabled[name] = True
    ed._sample_gates[name] = {
        'g1': {'kind': 'threshold', 'channel': 'CD3-A', 'value': 0.5,
               'parent_id': None, 'name': 'CD3pos', 'enabled': True},
        'g2': {'kind': 'threshold', 'channel': 'CD4-A', 'value': 0.5,
               'parent_id': 'g1', 'name': 'CD4pos', 'enabled': True}}
    ed._sample_gate_order[name] = ['g1', 'g2']
    ed._sample_gate_seq[name] = 2


@pytest.fixture
def freq_window(monkeypatch):
    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *a, **k: None)
    try:
        for i, (nm, tr, a, b) in enumerate(DESIGN):
            _add(ed, nm, tr, a, b, seed=i)
        ed._channels = ['CD3-A', 'CD4-A']
        ed._channel_labels = {'CD3-A': 'CD3-A', 'CD4-A': 'CD4-A'}
        ed._active_sample = 'Stim_1'
        from tests.conftest import frequency_panel
        w = frequency_panel(ed)
        w._collect(w._after_collect)         # the window schedules this via after()
        w.pop_var.set('CD3pos')
        w.metric_var.set('%Parent')
        w.factor_var.set('Name token')
        w.tokens_var.set('Stim, Ctrl')
        w._rebuild()
        yield w
    finally:
        root.destroy()


COMP = 'Compensation Controls_FL4 Stained Control'


def _window_with_extras(monkeypatch, extras):
    """The six DESIGN samples plus ``extras`` (name, trial, a, b), opened
    the same way as the freq_window fixture. Returns (root, window)."""
    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    for i, (nm, tr, a, b) in enumerate(list(DESIGN) + list(extras)):
        _add(ed, nm, tr, a, b, seed=i)
    ed._channels = ['CD3-A', 'CD4-A']
    ed._channel_labels = {'CD3-A': 'CD3-A', 'CD4-A': 'CD4-A'}
    ed._active_sample = 'Stim_1'
    from tests.conftest import frequency_panel
    w = frequency_panel(ed)
    w._collect(w._after_collect)
    w.pop_var.set('CD3pos')
    w.metric_var.set('%Parent')
    return root, w


def test_a_sample_matching_no_token_is_left_out(monkeypatch, tmp_path):
    # The Tokens tooltip says a sample matching none is left out; it used to
    # form an 'Other' group (Kruskal-Wallis, a 3rd Prism column, DA refused).
    root, w = _window_with_extras(monkeypatch, [('Bead_X', 'D1', 990, 10)])
    try:
        w.factor_var.set('Name token')
        w.tokens_var.set('Stim, Ctrl')
        w._rebuild()
        assert set(w._tidy['Sample']) == {nm for nm, *_ in DESIGN}
        res = w._last_res
        assert list(res['groups']) == ['Stim', 'Ctrl']
        # 3 v 3 fully separated: the default t-test on the logit scale
        # finds it (Mann-Whitney could only ever say 0.1 here).
        assert res['test'] == 'Welch t-test (logit)' and res['p'] < 0.05
        assert [g['n'] for g in res['groups'].values()] == [3, 3]
        assert 'Left out' in w._summary.get('1.0', 'end')
        assert 'Bead_X' in w._summary.get('1.0', 'end')
        col = tmp_path / 'col.csv'
        w._ask = lambda default, ftypes: str(col)
        w._export_prism_column()
        assert list(pd.read_csv(col).columns) == ['Stim', 'Ctrl']
        grp = tmp_path / 'grp.csv'
        w._ask = lambda default, ftypes: str(grp)
        w._export_prism_grouped()
        tab = pd.read_csv(grp, header=[0, 1], index_col=0)
        assert {c for c, _ in tab.columns} == {'Stim', 'Ctrl'}
    finally:
        root.destroy()


def test_a_compensation_control_is_not_a_trial_replicate(monkeypatch):
    root, w = _window_with_extras(monkeypatch, [(COMP, 'D1', 990, 10)])
    try:
        w.factor_var.set('Trial / day')
        w._rebuild()
        d1 = set(w._tidy.loc[w._tidy['Group'] == 'D1', 'Sample'])
        assert d1 == {'Stim_1', 'Ctrl_1'}
        assert COMP in w._summary.get('1.0', 'end')
        # 'Comp vs Samples' is the one grouping that compares them.
        w.factor_var.set('Comp vs Samples')
        w._rebuild()
        assert set(w._tidy.loc[w._tidy['Group'] == 'Comps', 'Sample']) == {COMP}
    finally:
        root.destroy()


# Real samples whose names hold 'comp', 'control' or 'stain', which the old
# substring rule (workspace_run.is_comp_sample) took for comp controls.
CONTROL_TREATED = [('Control_1', 'D1', 400, 100), ('Treated_1', 'D1', 600, 300),
                   ('Control_2', 'D2', 410, 100), ('Treated_2', 'D2', 610, 300),
                   ('Control_3', 'D3', 420, 100), ('Treated_3', 'D3', 620, 300)]
LOOKALIKES = [('Vehicle control', 'D1', 500, 100),
              ('Compound_A_10uM', 'D1', 500, 100),
              ('CFSE stained day 3', 'D1', 500, 100),
              ('Complete media', 'D1', 500, 100)]


def test_control_samples_are_not_comp_controls(monkeypatch, tmp_path):
    # 'Control_*' matched the old rule, so every Control sample was left out
    # and the window refused ('need >=2 non-empty groups') instead of the
    # 3 v 3 Mann-Whitney the user asked for. The FlowJo comp control holds
    # the token 'Control' too, and must still stay out of the Control group.
    root, w = _window_with_extras(
        monkeypatch, CONTROL_TREATED + [(COMP, 'D1', 990, 10)])
    try:
        w.factor_var.set('Name token')
        w.tokens_var.set('Control, Treated')
        w._rebuild()
        assert set(w._tidy['Sample']) == {nm for nm, *_ in CONTROL_TREATED}
        assert COMP in w._left_out
        res = w._last_res
        assert list(res['groups']) == ['Control', 'Treated']
        # 3 v 3 fully separated: the default t-test on the logit scale
        # finds it (Mann-Whitney could only ever say 0.1 here).
        assert res['test'] == 'Welch t-test (logit)' and res['p'] < 0.05
        assert [g['n'] for g in res['groups'].values()] == [3, 3]
        col = tmp_path / 'col.csv'
        w._ask = lambda default, ftypes: str(col)
        w._export_prism_column()
        assert list(pd.read_csv(col).columns) == ['Control', 'Treated']
        grp = tmp_path / 'grp.csv'
        w._ask = lambda default, ftypes: str(grp)
        w._export_prism_grouped()
        tab = pd.read_csv(grp, header=[0, 1], index_col=0)
        assert {c for c, _ in tab.columns} == {'Control', 'Treated'}
        assert tab.notna().sum().sum() == len(CONTROL_TREATED)
    finally:
        root.destroy()


def test_trial_day_leaves_out_only_comp_controls(monkeypatch):
    unstained = 'day1unstained_004'
    root, w = _window_with_extras(
        monkeypatch, CONTROL_TREATED + LOOKALIKES
        + [(COMP, 'D1', 990, 10), (unstained, 'D1', 5, 1)])
    try:
        w.factor_var.set('Trial / day')
        w._rebuild()
        d1 = set(w._tidy.loc[w._tidy['Group'] == 'D1', 'Sample'])
        assert d1 == ({'Stim_1', 'Ctrl_1', 'Control_1', 'Treated_1'}
                      | {nm for nm, *_ in LOOKALIKES})
        assert w._left_out == [COMP, unstained]
        # The user's Comps/Samples flag overrides the name in both directions.
        ed = w.editor
        ed._sample_is_comp['Vehicle control'] = True
        ed._sample_is_comp[COMP] = False
        w._rebuild()
        d1 = set(w._tidy.loc[w._tidy['Group'] == 'D1', 'Sample'])
        assert 'Vehicle control' not in d1 and COMP in d1
        assert w._left_out == ['Vehicle control', unstained]
    finally:
        root.destroy()


def test_an_unticked_sample_still_counts(freq_window):
    # The tree's tick only controls the plot; every loaded sample counts.
    ed = freq_window.editor
    ed._sample_plot_enabled['Ctrl_1'] = False
    freq_window._collect(freq_window._after_collect)
    freq_window._rebuild()
    assert set(freq_window._tidy['Sample']) == {nm for nm, *_ in DESIGN}
    assert freq_window._last_res['groups']['Ctrl']['n'] == 3


def test_frequencies_match_the_constructed_counts(freq_window):
    tidy = freq_window._tidy
    assert len(tidy) == 2 * len(DESIGN)
    for nm, _tr, a, b in DESIGN:
        rows = tidy[tidy['Sample'] == nm].set_index('Population')
        assert rows.loc['CD3pos', 'Count'] == a
        assert rows.loc['CD3pos', '%Parent'] == pytest.approx(a / 10)
        assert rows.loc['CD3pos', '%Total'] == pytest.approx(a / 10)
        assert rows.loc['CD3pos/CD4pos', 'Count'] == b
        assert rows.loc['CD3pos/CD4pos', '%Parent'] == pytest.approx(100 * b / a)
        assert rows.loc['CD3pos/CD4pos', '%Total'] == pytest.approx(b / 10)


def test_two_group_test_and_summary(freq_window):
    res = freq_window._last_res
    assert list(res['groups']) == ['Stim', 'Ctrl']
    assert res['groups']['Stim']['mean'] == pytest.approx(62.0)
    assert res['groups']['Ctrl']['median'] == pytest.approx(40.0)
    # Default: Welch's t-test on the logit of the percentages.
    from scipy.stats import ttest_ind

    from openflo.stats import transform_values as tv
    exp = ttest_ind(tv([60, 62, 64], 'logit'), tv([40, 38, 42], 'logit'),
                    equal_var=False).pvalue
    assert res['test'] == 'Welch t-test (logit)'
    assert res['p'] == pytest.approx(exp, rel=1e-12)
    text = freq_window._summary.get('1.0', 'end')
    assert 'Welch t-test (logit)' in text and "can't give p below" not in text

    # Rank test: 3 v 3, completely separated: exact two-sided Mann-Whitney
    # p = 2/20, the best it can do, and the summary says so.
    freq_window.test_var.set('Rank test')
    freq_window._rebuild()
    res = freq_window._last_res
    assert res['test'] == 'Mann-Whitney U' and res['p'] == pytest.approx(0.1)
    assert res['min_p'] == pytest.approx(0.1) and res['underpowered']
    text = freq_window._summary.get('1.0', 'end')
    assert 'Mann-Whitney U' in text and 'p = 0.1' in text
    assert "can't give p below 0.1" in text

    freq_window.test_var.set('t-test, raw values')
    freq_window._rebuild()
    exp = ttest_ind([60, 62, 64], [40, 38, 42], equal_var=False).pvalue
    assert freq_window._last_res['test'] == 'Welch t-test'
    assert freq_window._last_res['p'] == pytest.approx(exp, rel=1e-12)


def test_prism_column_and_grouped_exports(freq_window, tmp_path):
    col = tmp_path / 'col.csv'
    freq_window._ask = lambda default, ftypes: str(col)
    freq_window._export_prism_column()
    got = pd.read_csv(col)
    assert list(got.columns) == ['Stim', 'Ctrl']
    assert got['Stim'].tolist() == [60.0, 62.0, 64.0]
    assert got['Ctrl'].tolist() == [40.0, 38.0, 42.0]

    grp = tmp_path / 'grp.csv'
    freq_window._ask = lambda default, ftypes: str(grp)
    freq_window._export_prism_grouped()
    tab = pd.read_csv(grp, header=[0, 1], index_col=0)
    want = {('D1', 'Ctrl'): 40, ('D1', 'Stim'): 60, ('D2', 'Ctrl'): 38,
            ('D2', 'Stim'): 62, ('D3', 'Ctrl'): 42, ('D3', 'Stim'): 64}
    for (day, cond), v in want.items():
        assert float(tab.loc[day, (cond, '1')]) == pytest.approx(v)


def test_differential_abundance_direction_and_fold_change(freq_window, monkeypatch):
    import openflo.ui_diff as ud
    seen = {}
    monkeypatch.setattr(ud, 'DiffAbundanceWindow',
                        lambda parent, rows, disp, **kw: seen.update(rows=rows, disp=disp))
    freq_window._diff_abundance()
    rows, disp = seen['rows'], seen['disp']
    # The GLM's groups follow the pivot's (alphabetical) sample order: Ctrl
    # is A, Stim is B, and the header must say so.
    assert disp == ['Ctrl', 'Stim']
    r = {x['cluster']: x for x in rows}['CD3pos']
    assert r['group_a'] == 'Ctrl' and r['group_b'] == 'Stim'
    # Equal library sizes -> the MLE fold change is the ratio of mean counts.
    assert r['log2fc'] == pytest.approx(math.log2(1860 / 1200), abs=1e-6)
    assert r['prop_a'] == pytest.approx(0.40)
    assert r['prop_b'] == pytest.approx(0.62)


def test_compare_all_effect_follows_group_order(freq_window, monkeypatch):
    import openflo.ui_diff as ud
    seen = {}
    monkeypatch.setattr(ud, 'CompareAllWindow',
                        lambda parent, res, groups, metric, **kw: seen.update(
                            res=res, groups=groups, metric=metric))
    freq_window._compare_all()
    assert seen['groups'] == ['Stim', 'Ctrl']          # load order
    by = {r['feature']: r for r in seen['res']}
    assert set(by) == {'CD3pos', 'CD3pos/CD4pos'}
    # effect = log2(mean of 2nd group / mean of 1st) = log2(40 / 62)
    assert by['CD3pos']['effect'] == pytest.approx(math.log2(40 / 62), abs=1e-6)
    from scipy.stats import ttest_ind

    from openflo.stats import transform_values as tv
    exp = ttest_ind(tv([60, 62, 64], 'logit'), tv([40, 38, 42], 'logit'),
                    equal_var=False).pvalue
    assert by['CD3pos']['p'] == pytest.approx(exp, rel=1e-9)


def test_picking_a_metric_redraws_without_update(freq_window):
    # Picking a metric in the box must redraw the plot and summary at once,
    # as the population and grouping boxes do. Unbound, the window kept
    # showing %Parent after %Total or Count was picked.
    w = freq_window
    w.update_idletasks()
    for metric, want in (('%Total', [60.0, 62.0, 64.0]),
                         ('Count', [600.0, 620.0, 640.0])):
        w.metric_var.set(metric)
        w.metric_combo.event_generate('<<ComboboxSelected>>')
        w.update()
        assert w._fig.axes[0].get_ylabel() == metric
        summary = w._summary.get('1.0', 'end')
        assert f"Metric: {metric}" in summary.splitlines()[0]
        # The test ran on the picked metric's values: Stim is 600/620/640
        # events of 1000, so a mean of 62 (% of total) or 620 (count).
        stim = w._last_res['groups']['Stim']
        assert stim['n'] == 3
        assert stim['mean'] == pytest.approx(sum(want) / 3)


def test_status_message_clears_once_frequencies_are_in(monkeypatch):
    # "Computing frequencies…" went up on open and was never taken down, so
    # the window looked stuck after its numbers had arrived.
    root, w = _window_with_extras(monkeypatch, [])
    try:
        assert w.editor.status_var.get() != "Computing frequencies…"
        assert w._rows                       # the numbers did arrive
        # A newer message (another tool's) is left alone.
        w.editor.status_var.set("Something else")
        w._after_collect()
        assert w.editor.status_var.get() == "Something else"
    finally:
        root.destroy()


def test_one_sample_group_reads_na_not_nan(monkeypatch):
    # The SD of a single sample cannot be computed: the summary says NA.
    root, w = _window_with_extras(monkeypatch, [])
    try:
        w.factor_var.set('Name token')
        w.tokens_var.set('Stim_1, Ctrl')
        w._rebuild()
        lines = w._summary.get('1.0', 'end').splitlines()
        one = [ln for ln in lines if ln.strip().startswith('Stim_1:')]
        assert one == ['  Stim_1: n=1  mean=60  median=60  sd=NA']
        ctrl = [ln for ln in lines if ln.strip().startswith('Ctrl:')]
        assert ctrl and 'sd=NA' not in ctrl[0] and 'nan' not in ctrl[0]
    finally:
        root.destroy()


def _stim1_cd3(w):
    t = w._tidy
    row = t[(t['Sample'] == 'Stim_1') & (t['Population'] == 'CD3pos')]
    return float(row['Count'].iloc[0])


def test_update_recomputes_after_a_gate_edit(freq_window):
    # Update said "use this after editing gates" but only regrouped the rows
    # collected at open, so the plot, tests and exports kept the old counts.
    w = freq_window
    ed = w.editor
    w.pop_var.set('CD3pos/CD4pos')               # a pick Update must keep
    w._rebuild()
    assert _stim1_cd3(w) == 600
    # Stim_1's CD3 values are 0 or 1: a cut at 1.5 leaves no CD3+ events.
    ed._sample_gates['Stim_1']['g1']['value'] = 1.5
    w._rebuild()                                  # regrouping alone: stale
    assert _stim1_cd3(w) == 600
    w.update_button.invoke()                      # the Update button
    assert _stim1_cd3(w) == 0
    assert w.pop_var.get() == 'CD3pos/CD4pos'
    stim = w._last_res['groups']['Stim']
    # CD4+ inside an empty CD3+ has no %Parent: Stim_1 drops out (n=2), and
    # the mean is the other two Stim samples' 310/620 and 330/640.
    assert stim['n'] == 2
    assert stim['mean'] == pytest.approx((310 / 620 + 330 / 640) * 50)
    assert w.editor.status_var.get() != "Computing frequencies…"


def test_an_older_collect_finishing_late_is_dropped(freq_window, monkeypatch):
    # Two collects in flight (the opening one and an Update): the newer one's
    # rows stand even when the older one finishes last.
    import openflo.async_task as _at
    w = freq_window
    pending = []
    monkeypatch.setattr(_at, 'run_async',
                        lambda widget, work, on_done=None, on_error=None,
                        on_finally=None: pending.append((work, on_done)))
    w._collect()                                  # older: full gates
    ed = w.editor
    ed._sample_gates['Stim_1']['g1']['value'] = 1.5
    w._collect()                                  # newer: edited gate
    (old_work, old_done), (new_work, new_done) = pending
    new_done(new_work())
    old_done(old_work())                          # lands last
    w._rebuild()
    assert _stim1_cd3(w) == 0


def test_no_group_left_clears_the_summary(freq_window):
    # Clearing the tokens leaves every sample out: the plot went blank but
    # the old test result stayed in the summary (and in Stats summary…).
    w = freq_window
    assert 'Test:' in w._summary.get('1.0', 'end')
    w.tokens_var.set('')
    w._rebuild()
    text = w._summary.get('1.0', 'end').strip()
    assert text == 'No samples in this grouping.'
    assert w._last_res is None
    assert not w._fig.axes
