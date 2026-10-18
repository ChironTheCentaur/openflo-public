"""Group comparisons that can come out significant at the sizes flow
experiments use, a paired (donor) design, and a plain warning when a test
can't.

Mann-Whitney's smallest two-sided p is 2 / C(n_a + n_b, n_a): 0.1 at 3 vs
3, so two groups that don't overlap at all read "not significant". The
windows now default to Welch's t-test on transformed values (percentages
as log odds, counts as log, intensities as arcsinh), offer a paired test
matched by donor, put the donor in the negative-binomial abundance model,
and say when the chosen test can't reach 0.05 at all.

Generated data throughout, with hand-computable answers. Each rule has its
negative case beside it.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
from scipy.stats import ttest_ind, ttest_rel, wilcoxon

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

# ── the smallest p a test can give ───────────────────────────────────────


@pytest.mark.parametrize('test, sizes, want', [
    ('Mann-Whitney U', [3, 3], 0.1),
    ('Mann-Whitney U', [2, 2], 1 / 3),
    ('Mann-Whitney U', [4, 4], 2 / 70),
    ('Mann-Whitney U', [3, 0], 1.0),
    ('Wilcoxon signed-rank', [4], 0.125),
    ('Wilcoxon signed-rank', [6], 2 / 64),
    ('Welch t-test', [3, 3], 0.0),
    ('Paired t-test', [3], 0.0),
])
def test_smallest_p_is_the_exact_floor(test, sizes, want):
    from openflo.stats import smallest_p
    assert smallest_p(test, sizes) == pytest.approx(want)


def test_the_floor_matches_what_scipy_actually_returns():
    """The floor is reached by complete separation, never beaten."""
    from openflo.stats import mann_whitney, smallest_p
    assert mann_whitney([1, 2, 3], [7, 8, 9])[1] == pytest.approx(
        smallest_p('Mann-Whitney U', [3, 3]))
    assert wilcoxon([1, 2, 3, 4, 5, 6],
                    [11, 12, 13, 14, 15, 16]).pvalue == pytest.approx(
        smallest_p('Wilcoxon signed-rank', [6]))
    assert math.isnan(smallest_p('Kruskal-Wallis', [3, 3, 3]))


# ── transforms ───────────────────────────────────────────────────────────


def test_logit_takes_percentages_and_clips_the_edges():
    from openflo.stats import LOGIT_FLOOR, transform_values
    got = transform_values([50.0, 0.0, 100.0, 75.0, -1.0, 101.0, np.nan],
                           'logit')
    edge = math.log(LOGIT_FLOOR / (1 - LOGIT_FLOOR))
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(edge) and got[2] == pytest.approx(-edge)
    assert got[3] == pytest.approx(math.log(3))
    assert np.isnan(got[4:]).all()           # not a percentage: left out


def test_log1p_and_asinh_and_none():
    from openflo.stats import transform_values
    assert transform_values([0, 9, -1], 'log1p')[:2].tolist() == \
        pytest.approx([0.0, math.log(10)])
    assert np.isnan(transform_values([-1], 'log1p')[0])
    assert transform_values([-150, 150], 'asinh').tolist() == pytest.approx(
        [-math.asinh(1), math.asinh(1)])
    assert transform_values([3, 4], None).tolist() == [3, 4]
    with pytest.raises(ValueError):
        transform_values([1], 'sqrt')


# ── two groups, unpaired ─────────────────────────────────────────────────

SEPARATED = {'Stim': [60.0, 62.0, 64.0], 'Ctrl': [40.0, 38.0, 42.0]}


def test_the_rank_test_says_it_cannot_reach_significance_at_3_v_3():
    from openflo.stats import compare_groups
    r = compare_groups(SEPARATED)
    assert r['test'] == 'Mann-Whitney U' and r['p'] == pytest.approx(0.1)
    assert r['min_p'] == pytest.approx(0.1) and r['underpowered']
    r4 = compare_groups({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
    assert r4['p'] == pytest.approx(2 / 70) and not r4['underpowered']


def test_a_t_test_on_the_logit_scale_finds_the_separation():
    from openflo.stats import compare_groups, transform_values
    r = compare_groups(SEPARATED, parametric=True, transform='logit')
    want = ttest_ind(transform_values(SEPARATED['Stim'], 'logit'),
                     transform_values(SEPARATED['Ctrl'], 'logit'),
                     equal_var=False).pvalue
    assert r['test'] == 'Welch t-test (logit)'
    assert r['p'] == pytest.approx(want) and r['p'] < 0.05
    assert r['min_p'] == 0.0 and not r['underpowered']
    # The summary stays in the values' own units.
    assert r['groups']['Stim']['mean'] == pytest.approx(62.0)
    raw = compare_groups(SEPARATED, parametric=True)
    assert raw['test'] == 'Welch t-test' and raw['p'] != pytest.approx(r['p'])


# ── paired by donor ──────────────────────────────────────────────────────

# Donors differ a lot; the treatment adds 5 points in every donor.
CTRL = [10.0, 30.0, 50.0, 70.0]
STIM = [15.5, 34.5, 55.2, 74.8]
KEYS = ['d1', 'd2', 'd3', 'd4']


def test_pairing_takes_the_donor_out():
    from openflo.stats import compare_groups
    unpaired = compare_groups({'Ctrl': CTRL, 'Stim': STIM}, parametric=True)
    paired = compare_groups({'Ctrl': CTRL, 'Stim': STIM}, parametric=True,
                            pair_keys={'Ctrl': KEYS, 'Stim': KEYS})
    assert unpaired['p'] > 0.5
    assert paired['test'] == 'Paired t-test' and paired['paired']
    assert paired['n_pairs'] == 4 and paired['unpaired'] == []
    assert paired['p'] == pytest.approx(ttest_rel(STIM, CTRL).pvalue)
    assert paired['p'] < 0.001


def test_pairs_are_matched_by_key_not_by_position():
    from openflo.stats import compare_groups
    order = [3, 1, 0, 2]
    r = compare_groups(
        {'Ctrl': CTRL, 'Stim': [STIM[i] for i in order]}, parametric=True,
        pair_keys={'Ctrl': KEYS, 'Stim': [KEYS[i] for i in order]})
    assert r['p'] == pytest.approx(ttest_rel(STIM, CTRL).pvalue)


def test_a_donor_without_a_partner_or_given_twice_is_left_out_and_named():
    from openflo.stats import compare_groups
    r = compare_groups(
        {'Ctrl': CTRL + [90.0, 1.0], 'Stim': STIM + [5.0]}, parametric=True,
        pair_keys={'Ctrl': KEYS + ['d5', 'd6'], 'Stim': KEYS + ['d6']})
    assert r['n_pairs'] == 5 and r['unpaired'] == ['d5']
    r = compare_groups(
        {'Ctrl': CTRL + [90.0], 'Stim': STIM + [95.0, 96.0]},
        parametric=True,
        pair_keys={'Ctrl': KEYS + ['d5'], 'Stim': KEYS + ['d5', 'd5']})
    assert r['n_pairs'] == 4 and r['unpaired'] == ['d5']
    assert r['p'] == pytest.approx(ttest_rel(STIM, CTRL).pvalue)


def test_the_rank_test_paired_is_wilcoxon_with_its_own_floor():
    from openflo.stats import compare_groups
    r = compare_groups({'Ctrl': CTRL, 'Stim': STIM},
                       pair_keys={'Ctrl': KEYS, 'Stim': KEYS})
    assert r['test'] == 'Wilcoxon signed-rank'
    assert r['p'] == pytest.approx(wilcoxon(STIM, CTRL).pvalue)
    assert r['min_p'] == pytest.approx(0.125) and r['underpowered']


def test_paired_degenerate_cases_never_claim_a_difference():
    from openflo.stats import compare_groups
    same = compare_groups({'a': CTRL, 'b': CTRL}, parametric=True,
                          pair_keys={'a': KEYS, 'b': KEYS})
    assert same['p'] == 1.0
    shift = compare_groups({'a': CTRL, 'b': [x + 5 for x in CTRL]},
                           parametric=True, pair_keys={'a': KEYS, 'b': KEYS})
    assert math.isnan(shift['p'])            # no spread: t undefined
    rank = compare_groups({'a': CTRL, 'b': CTRL},
                          pair_keys={'a': KEYS, 'b': KEYS})
    assert rank['p'] == 1.0


def test_paired_keys_must_line_up_with_values():
    from openflo.stats import compare_groups
    with pytest.raises(ValueError):
        compare_groups({'a': CTRL, 'b': STIM}, parametric=True,
                       pair_keys={'a': KEYS[:3], 'b': KEYS})


def test_three_paired_groups_rm_anova_equals_paired_t_for_two():
    """With two groups the repeated-measures F is the paired t squared;
    with three it runs, with paired post-hoc tests."""
    from openflo.stats import _rm_anova, compare_groups
    F, p, dg, de = _rm_anova(np.array([CTRL, STIM]))
    t = ttest_rel(STIM, CTRL)
    assert F == pytest.approx(t.statistic ** 2) and p == pytest.approx(
        t.pvalue)
    assert (dg, de) == (1, 3)
    third = [x + 9.0 + (i % 2) for i, x in enumerate(CTRL)]
    r = compare_groups({'a': CTRL, 'b': STIM, 'c': third}, parametric=True,
                       pair_keys={'a': KEYS, 'b': KEYS, 'c': KEYS})
    assert r['test'] == 'Repeated-measures ANOVA' and r['p'] < 0.001
    assert len(r['posthoc']) == 3
    fr = compare_groups({'a': CTRL, 'b': STIM, 'c': third},
                        pair_keys={'a': KEYS, 'b': KEYS, 'c': KEYS})
    assert fr['test'] == 'Friedman' and 0 < fr['p'] < 0.05


def test_compare_all_features_passes_the_choice_and_flags_rows():
    from openflo.stats import compare_all_features
    vbf = {'p1': SEPARATED, 'p2': {'Stim': [1.0, 2.0, 3.0],
                                   'Ctrl': [1.5, 2.5, 2.0]}}
    rank = compare_all_features(vbf)
    assert all(r['underpowered'] for r in rank)
    t = compare_all_features(vbf, parametric=True, transform='logit')
    by = {r['feature']: r for r in t}
    assert by['p1']['p'] < 0.05 and not by['p1']['underpowered']
    keys = {f: {'Stim': ['x', 'y', 'z'], 'Ctrl': ['x', 'y', 'z']}
            for f in vbf}
    pr = compare_all_features(vbf, parametric=True, pair_keys=keys)
    assert all(r['n_pairs'] == 3 for r in pr)


# ── differential_test (the CLI's comparison CSV) ─────────────────────────


def test_differential_test_t_on_logit_and_the_floor():
    from openflo.diffexp import differential_test
    from openflo.stats import transform_values
    a = {'c1': [60.0, 62.0, 64.0], 'c2': [5.0, 5.0, 5.0]}
    b = {'c1': [40.0, 38.0, 42.0], 'c2': [5.0, 5.0, 5.0]}
    rank = {r['feature']: r for r in differential_test(a, b)}
    assert rank['c1']['p'] == pytest.approx(0.1)
    assert rank['c1']['test'] == 'Mann-Whitney U'
    assert rank['c1']['underpowered']
    t = {r['feature']: r for r in differential_test(
        a, b, parametric=True, transform='logit')}
    want = ttest_ind(transform_values(a['c1'], 'logit'),
                     transform_values(b['c1'], 'logit'),
                     equal_var=False).pvalue
    assert t['c1']['p'] == pytest.approx(want) and t['c1']['p'] < 0.05
    assert t['c1']['test'] == 'Welch t-test (logit)'
    assert not t['c1']['underpowered']
    assert t['c2']['p'] == 1.0               # identical: no difference
    # Fold change stays in the values' own units.
    assert t['c1']['log2fc'] == pytest.approx(math.log2(40 / 62))


# ── the donor in the negative-binomial abundance model ───────────────────


def _donor_counts(effect, n_donors, seed, n_pops=1, alpha=0.05):
    rng = np.random.default_rng(seed)
    donor = rng.lognormal(0.0, 0.8, n_donors)
    lib = np.full(2 * n_donors, 20_000.0)
    rows = []
    for _ in range(n_pops):
        base = 0.02 * donor * 20_000.0
        r = 1.0 / alpha
        a = rng.negative_binomial(r, r / (r + base))
        b = rng.negative_binomial(r, r / (r + base * effect))
        rows.append(np.r_[a, b])
    group = ['ctrl'] * n_donors + ['stim'] * n_donors
    block = [f'd{i}' for i in range(n_donors)] * 2
    return np.array(rows, dtype=float), group, lib, block


def test_the_donor_term_finds_a_paired_effect_the_plain_model_misses():
    from openflo.diffexp import differential_abundance
    Y, group, lib, block = _donor_counts(1.5, 4, seed=3, alpha=0.01)
    plain = differential_abundance(Y, group, lib)[0]
    blocked = differential_abundance(Y, group, lib, block=block)[0]
    assert plain['p'] > 0.1
    assert blocked['p'] < 0.05
    assert blocked['df'] == 8 - (1 + 4) and blocked['n_blocks'] == 4
    assert blocked['log2fc'] == pytest.approx(math.log2(1.5), abs=0.25)
    assert 'n_blocks' not in plain


def test_the_donor_term_keeps_false_positives_near_5_percent():
    from openflo.diffexp import differential_abundance
    ps = []
    for seed in range(40):
        Y, group, lib, block = _donor_counts(1.0, 4, seed=100 + seed,
                                             n_pops=5, alpha=0.1)
        ps += [r['p'] for r in differential_abundance(Y, group, lib,
                                                      block=block)]
    rate = float(np.mean(np.array(ps) < 0.05))
    assert 0.01 <= rate <= 0.10, rate


def test_the_donor_term_edge_cases():
    from openflo.diffexp import differential_abundance
    Y, group, lib, block = _donor_counts(1.5, 3, seed=2)
    with pytest.raises(ValueError):
        differential_abundance(Y, group, lib, block=block[:-1])
    # One sample per donor and group, every donor its own block in only
    # one group: no residual df, so no p.
    r = differential_abundance(Y[:, [0, 3]], ['ctrl', 'stim'], lib[:2],
                               block=['d0', 'd0'])[0]
    assert math.isnan(r['p']) and r['df'] == 0
    # A population with no events in one group: fold change is infinite.
    Z = Y.copy()
    Z[0, 3:] = 0
    r = differential_abundance(Z, group, lib, block=block)[0]
    assert r['log2fc'] == float('-inf')


# ── the windows ──────────────────────────────────────────────────────────


def test_pair_key_reads_the_donor_from_the_name():
    from openflo.group_test_choice import pair_key
    assert pair_key('D1_Stim', 'Stim', ['Stim', 'Ctrl']) == 'd1'
    assert pair_key('D1_Ctrl.fcs', 'Ctrl', ['Stim', 'Ctrl']) == 'd1'
    assert pair_key('donor 2 - stim', 'Stim') == 'donor 2'
    assert pair_key('D1_Stim', 'Stim') != pair_key('D2_Stim', 'Stim')


def test_the_test_choices():
    from openflo.group_test_choice import RANK, T_RAW, T_TRANSFORMED, test_kwargs
    assert test_kwargs(T_TRANSFORMED, 'logit') == {'parametric': True,
                                                   'transform': 'logit'}
    assert test_kwargs(RANK, 'logit') == {'parametric': False,
                                          'transform': None}
    assert test_kwargs(T_RAW, 'logit') == {'parametric': True,
                                           'transform': None}


def test_power_lines_say_what_the_test_cannot_do():
    from openflo.group_test_choice import power_lines
    from openflo.stats import compare_groups
    text = '\n'.join(power_lines(compare_groups(SEPARATED)))
    assert "can't give p below 0.1" in text
    assert power_lines(compare_groups(SEPARATED, parametric=True)) == []
    paired = compare_groups({'Ctrl': CTRL + [1.0], 'Stim': STIM},
                            parametric=True,
                            pair_keys={'Ctrl': KEYS + ['d9'], 'Stim': KEYS})
    text = '\n'.join(power_lines(paired))
    assert '4 donor(s)' in text and 'd9' in text


PAIRED_DESIGN = [('D1_Stim', 'T', 300, 100), ('D1_Ctrl', 'T', 250, 80),
                 ('D2_Stim', 'T', 520, 100), ('D2_Ctrl', 'T', 470, 80),
                 ('D3_Stim', 'T', 750, 100), ('D3_Ctrl', 'T', 700, 80),
                 ('D4_Stim', 'T', 410, 100), ('D4_Ctrl', 'T', 360, 80)]


@pytest.fixture
def paired_window(monkeypatch):
    from test_frequency_window import _add, _inline
    from test_gui_smoke import _editor_or_skip

    import openflo.async_task as _at
    monkeypatch.setattr(_at, 'run_async', _inline)
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    try:
        for i, (nm, tr, a, b) in enumerate(PAIRED_DESIGN):
            _add(ed, nm, tr, a, b, seed=i)
        ed._channels = ['CD3-A', 'CD4-A']
        ed._channel_labels = {'CD3-A': 'CD3-A', 'CD4-A': 'CD4-A'}
        ed._active_sample = 'D1_Stim'
        from openflo.ui_frequency import FrequencyComparisonWindow
        w = FrequencyComparisonWindow(ed)
        w.withdraw()
        w._collect(w._after_collect)
        w.pop_var.set('CD3pos')
        w.metric_var.set('%Parent')
        w.factor_var.set('Name token')
        w.tokens_var.set('Stim, Ctrl')
        w._rebuild()
        yield w
    finally:
        root.destroy()


def test_frequencies_default_is_the_t_test_and_paired_matches_donors(
        paired_window):
    from openflo.stats import transform_values as tv
    w = paired_window
    res = w._last_res
    assert res['test'] == 'Welch t-test (logit)' and not res['paired']
    unpaired_p = res['p']
    stim = [30.0, 52.0, 75.0, 41.0]
    ctrl = [25.0, 47.0, 70.0, 36.0]
    w.paired_var.set(True)
    w._rebuild()
    res = w._last_res
    assert res['test'] == 'Paired t-test (logit)' and res['n_pairs'] == 4
    assert res['p'] == pytest.approx(
        ttest_rel(tv(ctrl, 'logit'), tv(stim, 'logit')).pvalue)
    assert res['p'] < 0.01 < unpaired_p
    assert 'Paired by donor: 4 donor(s)' in w._summary.get('1.0', 'end')


def test_frequencies_paired_puts_the_donor_in_the_abundance_model(
        paired_window, monkeypatch):
    import openflo.diffexp as de
    import openflo.ui_diff as ud
    w = paired_window
    seen = {}
    real = de.differential_abundance

    def spy(*a, **k):
        seen['block'] = k.get('block')
        return real(*a, **k)
    monkeypatch.setattr(de, 'differential_abundance', spy)
    monkeypatch.setattr(ud, 'DiffAbundanceWindow',
                        lambda parent, rows, groups: seen.update(rows=rows))
    w._diff_abundance()
    assert seen['block'] is None
    w.paired_var.set(True)
    w._diff_abundance()
    assert sorted(set(seen['block'])) == ['d1', 'd2', 'd3', 'd4']
    assert all(r['n_blocks'] == 4 for r in seen['rows'])


def test_frequencies_compare_all_follows_the_choice(paired_window,
                                                    monkeypatch):
    import openflo.ui_diff as ud
    w = paired_window
    seen = {}
    monkeypatch.setattr(ud, 'CompareAllWindow',
                        lambda parent, res, groups, metric: seen.update(
                            res=res))
    w.test_var.set('Rank test')
    w._compare_all()
    assert all(r['test'] == 'Mann-Whitney U' for r in seen['res'])
    w.test_var.set('t-test, transformed')
    w.paired_var.set(True)
    w._compare_all()
    assert all(r['test'] == 'Paired t-test (logit)' and r['n_pairs'] == 4
               for r in seen['res'])


def test_compare_all_csv_carries_the_floor():
    from openflo.stats import compare_all_features
    from openflo.ui_diff import CompareAllWindow
    res = compare_all_features({'p1': SEPARATED})
    rows = CompareAllWindow._flat_rows(type('W', (), {'_results': res})())
    assert rows[0]['min_possible_p'] == pytest.approx(0.1)
    assert rows[0]['underpowered'] is True


def test_group_comparison_window_warns_when_no_pair_can_reach_005():
    from test_group_stats_window import _run, _setup
    root, ed = _setup()
    try:
        txt = _run(ed, 'CD4-A')
        assert "Mann-Whitney can't give p below 0.1" in txt
    finally:
        root.destroy()
    import pandas as pd

    from openflo.pipeline import FlowSample
    root, ed = _setup()
    try:
        for name, trial, med in [('A4', 'D1', 16), ('B4', 'D2', 26)]:
            cd4 = med + np.concatenate([np.zeros(51),
                                        np.linspace(1, 100, 50)])
            ed._samples[name] = FlowSample.from_dataframe(
                pd.DataFrame({'CD4-A': cd4, 'CD8-A': np.arange(101.0)}),
                name=name)
            ed._sample_order.append(name)
            ed._sample_trial[name] = trial
        txt = _run(ed, 'CD4-A')
        assert "can't give p below" not in txt
    finally:
        root.destroy()
