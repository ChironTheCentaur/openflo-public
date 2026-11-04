"""Every path through `compare_groups`, checked against numbers worked out
here rather than read back from the code under test.

The expected values come from SciPy called directly on the same arrays
(Welch / paired t, Mann-Whitney U, Wilcoxon, one-way ANOVA, Kruskal-Wallis,
Friedman), from the `statistics` module for the group summaries (sample SD,
n - 1 in the denominator), and from a Benjamini-Hochberg adjustment and a
repeated-measures ANOVA written out longhand below. The smallest-p floors
are 2 / C(n_a + n_b, n_a) for Mann-Whitney and 2 / 2^n for Wilcoxon.

Tolerances are rel=1e-12: the same arithmetic in a different order, never
sampling noise. All data are small literal arrays; groups A, B, C, D and
donors d1, d2, ...
"""
from __future__ import annotations

import math
import statistics
import warnings
from math import comb

import numpy as np
import pytest
import scipy.stats
from scipy.stats import f as f_dist
from scipy.stats import (
    f_oneway,
    friedmanchisquare,
    kruskal,
    mannwhitneyu,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

from openflo import stats
from openflo.stats import compare_groups

REL = 1e-12

RESULT_KEYS = ['test', 'statistic', 'p', 'parametric', 'n_groups',
               'transform', 'paired', 'n_pairs', 'unpaired', 'min_p',
               'underpowered', 'groups', 'posthoc']

A = [1.0, 2.0, 4.0, 3.5, 2.5]
B = [5.0, 6.5, 6.0, 8.0, 7.0]
C = [3.0, 9.0, 4.5, 6.0, 5.5]
D = [0.5, 1.5, 1.0, 2.2, 0.8]
DONORS = ['d1', 'd2', 'd3', 'd4', 'd5']


def _bh_by_hand(pvals):
    """Benjamini-Hochberg written out: NaNs skipped, m = finite count."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    m = len(idx)
    ranked = sorted(idx, key=lambda i: pvals[i])
    out = [float('nan')] * len(pvals)
    running = 1.0
    for rank in range(m, 0, -1):
        i = ranked[rank - 1]
        running = min(running, pvals[i] * m / rank)
        out[i] = running
    return out


def _summary(values):
    """{n, mean, median, sd} of the finite values, via `statistics`."""
    v = [x for x in values if math.isfinite(x)]
    return {'n': len(v), 'mean': statistics.mean(v),
            'median': statistics.median(v),
            'sd': statistics.stdev(v) if len(v) > 1 else float('nan')}


def _assert_summaries(res, expected_values):
    """res['groups'] matches _summary of each group, in the given order."""
    assert list(res['groups']) == list(expected_values)
    for g, vals in expected_values.items():
        want = _summary(vals)
        got = res['groups'][g]
        assert list(got) == ['n', 'mean', 'median', 'sd']
        assert got['n'] == want['n']
        for k in ('mean', 'median'):
            assert got[k] == pytest.approx(want[k], rel=REL)
        if math.isnan(want['sd']):
            assert math.isnan(got['sd'])
        else:
            assert got['sd'] == pytest.approx(want['sd'], rel=REL)


def _assert_posthoc(res, pairs, raw_p):
    """Posthoc rows in this (a, b) order with these raw p and hand BH."""
    rows = res['posthoc']
    assert [(r['a'], r['b']) for r in rows] == pairs
    assert all(list(r) == ['a', 'b', 'p', 'p_adj'] for r in rows)
    want_adj = _bh_by_hand(raw_p)
    for r, p, q in zip(rows, raw_p, want_adj, strict=True):
        if math.isnan(p):
            assert math.isnan(r['p']) and math.isnan(r['p_adj'])
        else:
            assert r['p'] == pytest.approx(p, rel=REL)
            assert r['p_adj'] == pytest.approx(q, rel=REL)


# ── fewer than two usable groups ─────────────────────────────────────────


def test_one_usable_group_gives_no_test_but_keeps_its_summary():
    """An all-NaN group is not counted: no test, NaN p, summary of the
    other group only, defaults in every other field."""
    res = compare_groups({'A': [1.0, 2.0, 4.0], 'B': [np.nan, np.nan]})
    assert list(res) == RESULT_KEYS
    assert res['test'] is None
    assert math.isnan(res['statistic']) and math.isnan(res['p'])
    assert math.isnan(res['min_p']) and res['underpowered'] is False
    assert res['n_groups'] == 1 and res['posthoc'] == []
    assert res['paired'] is False and res['n_pairs'] is None
    assert res['unpaired'] == []
    _assert_summaries(res, {'A': [1.0, 2.0, 4.0]})


def test_one_usable_group_paired_returns_before_pairing():
    """With pair keys but one usable group: paired True, still no test and
    no pairing done (n_pairs None)."""
    res = compare_groups({'A': [1.0, 2.0], 'B': [np.nan, np.nan]},
                         pair_keys={'A': ['d1', 'd2'], 'B': ['d1', 'd2']})
    assert res['test'] is None and res['paired'] is True
    assert res['n_pairs'] is None and res['n_groups'] == 1


# ── unpaired, two groups ─────────────────────────────────────────────────


def test_two_groups_welch():
    """Two groups, parametric: Welch's t (unequal variances), t-test floor
    0, not underpowered, no posthoc, raw-unit summaries."""
    res = compare_groups({'A': A, 'B': B}, parametric=True)
    want = ttest_ind(A, B, equal_var=False)
    assert list(res) == RESULT_KEYS
    assert res['test'] == 'Welch t-test'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)  # pyright: ignore[reportAttributeAccessIssue]
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)  # pyright: ignore[reportAttributeAccessIssue]
    assert res['statistic'] < 0                 # A below B
    assert res['min_p'] == 0.0 and res['underpowered'] is False
    assert res['parametric'] is True and res['transform'] is None
    assert res['n_groups'] == 2 and res['posthoc'] == []
    assert res['paired'] is False and res['n_pairs'] is None
    _assert_summaries(res, {'A': A, 'B': B})


def test_two_groups_mann_whitney_floor_and_underpowered():
    """Two groups, non-parametric: Mann-Whitney U against SciPy, floor
    2 / C(6, 3) = 0.1 at 3 v 3 (underpowered), 2 / C(10, 5) at 5 v 5
    (not)."""
    a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    res = compare_groups({'A': a, 'B': b}, parametric=False)
    want = mannwhitneyu(a, b, alternative='two-sided')
    assert res['test'] == 'Mann-Whitney U'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
    assert res['min_p'] == pytest.approx(2 / comb(6, 3), rel=REL)
    assert res['underpowered'] is True and res['parametric'] is False

    res5 = compare_groups({'A': A, 'B': B}, parametric=False)
    want5 = mannwhitneyu(A, B, alternative='two-sided')
    assert res5['p'] == pytest.approx(float(want5.pvalue), rel=REL)
    assert res5['min_p'] == pytest.approx(2 / comb(10, 5), rel=REL)
    assert res5['underpowered'] is False


def test_two_groups_transform_named_applied_and_summaries_raw():
    """log1p: the test name says so, the test runs on log1p values with
    the negative one (NaN after log1p) dropped, the summary keeps raw
    units and still counts that value."""
    a = [-1.0, 1.0, 3.0, 7.0]
    b = [10.0, 20.0, 15.0]
    res = compare_groups({'A': a, 'B': b}, parametric=True,
                         transform='log1p')
    want = ttest_ind(np.log1p([1.0, 3.0, 7.0]), np.log1p(b), equal_var=False)
    assert res['test'] == 'Welch t-test (log1p)'
    assert res['transform'] == 'log1p'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)  # pyright: ignore[reportAttributeAccessIssue]
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)  # pyright: ignore[reportAttributeAccessIssue]
    _assert_summaries(res, {'A': a, 'B': b})

    mw = compare_groups({'A': a, 'B': b}, parametric=False,
                        transform='log1p')
    assert mw['test'] == 'Mann-Whitney U (log1p)'
    assert mw['min_p'] == pytest.approx(2 / comb(6, 3), rel=REL)  # 3 v 3


def test_two_groups_test_raising_value_error_leaves_no_test(monkeypatch):
    """If the two-group test raises ValueError the result has no test, NaN
    statistic/p/min_p, but keeps the summaries."""
    def boom(*_a, **_k):
        raise ValueError('cannot compute')
    monkeypatch.setattr(scipy.stats, 'ttest_ind', boom)
    monkeypatch.setattr(stats, 'mann_whitney', boom)
    for parametric in (True, False):
        res = compare_groups({'A': A, 'B': B}, parametric=parametric)
        assert res['test'] is None
        assert math.isnan(res['statistic']) and math.isnan(res['p'])
        assert math.isnan(res['min_p']) and res['underpowered'] is False
        assert res['n_groups'] == 2
        _assert_summaries(res, {'A': A, 'B': B})


# ── unpaired, more than two groups ───────────────────────────────────────


def test_many_groups_anova_and_welch_posthoc_in_input_order():
    """Four groups, parametric: one-way ANOVA against SciPy, Welch posthoc
    in upper-triangular order of the input dict (not sorted), hand BH."""
    groups = {'C': C, 'A': A, 'D': D, 'B': B}
    res = compare_groups(groups, parametric=True)
    want = f_oneway(C, A, D, B)
    assert list(res) == RESULT_KEYS
    assert res['test'] == 'One-way ANOVA'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
    assert math.isnan(res['min_p']) and res['underpowered'] is False
    assert res['n_groups'] == 4
    names = list(groups)
    pairs = [(names[i], names[j]) for i in range(4) for j in range(i + 1, 4)]
    raw = [float(ttest_ind(groups[x], groups[y], equal_var=False).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
           for x, y in pairs]
    assert pairs[0] == ('C', 'A') and pairs[-1] == ('D', 'B')
    _assert_posthoc(res, pairs, raw)
    _assert_summaries(res, groups)


def test_many_groups_kruskal_and_mann_whitney_posthoc():
    """Three groups, non-parametric: Kruskal-Wallis against SciPy,
    Mann-Whitney posthoc with hand BH, transform suffix on the name."""
    groups = {'A': A, 'B': B, 'C': C}
    res = compare_groups(groups, parametric=False, transform='asinh')
    t = {g: np.arcsinh(np.asarray(v) / stats.ASINH_COFACTOR)
         for g, v in groups.items()}
    want = kruskal(t['A'], t['B'], t['C'])
    assert res['test'] == 'Kruskal-Wallis (asinh)'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
    pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    raw = [float(mannwhitneyu(t[x], t[y], alternative='two-sided').pvalue)
           for x, y in pairs]
    _assert_posthoc(res, pairs, raw)
    _assert_summaries(res, groups)


def test_many_groups_failed_pair_is_nan_and_left_out_of_bh(monkeypatch):
    """A pair whose test raises ValueError gets NaN p and p_adj; BH runs
    over the other pairs only (m = 2)."""
    real = stats.mann_whitney

    def flaky(a, b):
        if np.array_equal(a, A) and np.array_equal(b, C):
            raise ValueError('cannot compute')
        return real(a, b)
    monkeypatch.setattr(stats, 'mann_whitney', flaky)
    res = compare_groups({'A': A, 'B': B, 'C': C}, parametric=False)
    raw = [float(mannwhitneyu(A, B, alternative='two-sided').pvalue),
           float('nan'),
           float(mannwhitneyu(B, C, alternative='two-sided').pvalue)]
    _assert_posthoc(res, [('A', 'B'), ('A', 'C'), ('B', 'C')], raw)
    assert res['test'] == 'Kruskal-Wallis'


def test_many_groups_omnibus_raising_keeps_posthoc(monkeypatch):
    """If the omnibus test raises ValueError there is no test, but the
    pairwise posthoc is still computed."""
    def boom(*_a, **_k):
        raise ValueError('cannot compute')
    monkeypatch.setattr(scipy.stats, 'f_oneway', boom)
    res = compare_groups({'A': A, 'B': B, 'C': C}, parametric=True)
    assert res['test'] is None and math.isnan(res['p'])
    pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    g = {'A': A, 'B': B, 'C': C}
    raw = [float(ttest_ind(g[x], g[y], equal_var=False).pvalue)  # pyright: ignore[reportAttributeAccessIssue]
           for x, y in pairs]
    _assert_posthoc(res, pairs, raw)


# ── paired, two groups ───────────────────────────────────────────────────


PA = [10.0, 12.0, 9.0, 14.0, 11.0, 13.0]
PB = [12.5, 13.0, 12.0, 15.5, 14.8, 13.4]
PK = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']


def test_paired_t_two_groups():
    """Paired t of B against A (positive when B is higher), n_pairs,
    t-test floor 0, nothing unpaired, raw summaries."""
    res = compare_groups({'A': PA, 'B': PB}, parametric=True,
                         pair_keys={'A': PK, 'B': PK})
    want = ttest_rel(PB, PA)
    assert list(res) == RESULT_KEYS
    assert res['test'] == 'Paired t-test' and res['paired'] is True
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['statistic'] > 0
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
    assert res['n_pairs'] == 6 and res['unpaired'] == []
    assert res['min_p'] == 0.0 and res['underpowered'] is False
    assert res['posthoc'] == []
    _assert_summaries(res, {'A': PA, 'B': PB})


def test_paired_wilcoxon_floor_and_unpaired_keys():
    """Wilcoxon against SciPy on the matched donors; a donor missing from
    one group (d7) and one given twice in a group (d8) are left out and
    named, sorted; floor 2 / 2^n: underpowered at 5, not at 6."""
    a = PA + [20.0, 30.0, 31.0]
    b = PB + [40.0]
    ka = PK + ['d7', 'd8', 'd8']
    kb = PK + ['d8']
    res = compare_groups({'A': a, 'B': b}, parametric=False,
                         pair_keys={'A': ka, 'B': kb})
    want = wilcoxon(PB, PA)
    assert res['test'] == 'Wilcoxon signed-rank'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)  # pyright: ignore[reportAttributeAccessIssue]
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)  # pyright: ignore[reportAttributeAccessIssue]
    assert res['n_pairs'] == 6 and res['unpaired'] == ['d7', 'd8']
    assert res['min_p'] == pytest.approx(2 / 2 ** 6, rel=REL)
    assert res['underpowered'] is False
    _assert_summaries(res, {'A': a, 'B': b})

    five = compare_groups({'A': PA[:5], 'B': PB[:5]}, parametric=False,
                          pair_keys={'A': PK[:5], 'B': PK[:5]})
    assert five['min_p'] == pytest.approx(2 / 2 ** 5, rel=REL)
    assert five['underpowered'] is True


def test_paired_transform_aligns_with_raw_values():
    """The transform runs on the raw values so the keys still line up: a
    NaN (d2) and a value log1p can't take (d3) drop those donors; the
    paired t runs on log1p of the rest."""
    a = [1.0, np.nan, 3.0, 4.0, 6.0]
    b = [2.0, 5.0, -1.0, 9.0, 8.0]
    res = compare_groups({'A': a, 'B': b}, parametric=True,
                         transform='log1p',
                         pair_keys={'A': DONORS, 'B': DONORS})
    keep_a, keep_b = [1.0, 4.0, 6.0], [2.0, 9.0, 8.0]
    want = ttest_rel(np.log1p(keep_b), np.log1p(keep_a))
    assert res['test'] == 'Paired t-test (log1p)'
    assert res['n_pairs'] == 3 and res['unpaired'] == ['d2', 'd3']
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
    _assert_summaries(res, {'A': a, 'B': b})


def test_paired_group_missing_from_pair_keys_raises():
    """A usable group with no entry in pair_keys has 0 keys for its values,
    which is a length mismatch."""
    with pytest.raises(ValueError, match='0 pair keys'):
        compare_groups({'A': PA, 'B': PB}, pair_keys={'A': PK})


def test_paired_length_mismatch_raises():
    """Fewer pair keys than values in a group raises ValueError naming the
    counts."""
    with pytest.raises(ValueError, match='6 values but 5 pair keys'):
        compare_groups({'A': PA, 'B': PB},
                       pair_keys={'A': PK, 'B': PK[:5]})


# ── paired, more than two groups ─────────────────────────────────────────


PC = [11.0, 15.0, 10.5, 14.2, 16.0, 12.1]


def _rm_anova_by_hand(rows):
    """Repeated-measures F and p from the sums of squares, longhand."""
    k, n = len(rows), len(rows[0])
    grand = sum(sum(r) for r in rows) / (k * n)
    group_means = [sum(r) / n for r in rows]
    subj_means = [sum(r[j] for r in rows) / k for j in range(n)]
    ss_group = n * sum((m - grand) ** 2 for m in group_means)
    ss_subj = k * sum((m - grand) ** 2 for m in subj_means)
    ss_total = sum((x - grand) ** 2 for r in rows for x in r)
    ss_err = ss_total - ss_group - ss_subj
    df_g, df_e = k - 1, (k - 1) * (n - 1)
    F = (ss_group / df_g) / (ss_err / df_e)
    return F, float(f_dist.sf(F, df_g, df_e))


def test_paired_many_rm_anova_and_paired_posthoc():
    """Three paired groups, parametric: repeated-measures F and p by hand,
    no simple floor (min_p NaN), paired-t posthoc with hand BH."""
    g = {'A': PA, 'B': PB, 'C': PC}
    res = compare_groups(g, parametric=True,
                         pair_keys={'A': PK, 'B': PK, 'C': PK})
    F, p = _rm_anova_by_hand([PA, PB, PC])
    assert res['test'] == 'Repeated-measures ANOVA'
    assert res['statistic'] == pytest.approx(F, rel=REL)
    assert res['p'] == pytest.approx(p, rel=REL)
    assert res['n_pairs'] == 6 and res['unpaired'] == []
    assert math.isnan(res['min_p']) and res['underpowered'] is False
    pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    raw = [float(ttest_rel(g[y], g[x]).pvalue) for x, y in pairs]
    _assert_posthoc(res, pairs, raw)
    _assert_summaries(res, g)


def test_paired_many_friedman_and_wilcoxon_posthoc():
    """Three paired groups, non-parametric, with a transform: Friedman
    against SciPy on the transformed matrix, Wilcoxon posthoc with hand
    BH, suffix on the name."""
    g = {'A': PA, 'B': PB, 'C': PC}
    res = compare_groups(g, parametric=False, transform='log1p',
                         pair_keys={'A': PK, 'B': PK, 'C': PK})
    t = {k: np.log1p(v) for k, v in g.items()}
    want = friedmanchisquare(t['A'], t['B'], t['C'])
    assert res['test'] == 'Friedman (log1p)'
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
    assert res['n_pairs'] == 6
    pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
    raw = [float(wilcoxon(t[y], t[x]).pvalue) for x, y in pairs]  # pyright: ignore[reportAttributeAccessIssue]
    _assert_posthoc(res, pairs, raw)


def test_friedman_with_every_subject_constant_is_nan():
    """Every donor has the same value in each group: Friedman is undefined,
    NaN statistic and p (the test name is still recorded), decided before
    SciPy is asked, so no divide-by-zero RuntimeWarning escapes."""
    v = [1.0, 2.0, 3.0]
    keys = ['d1', 'd2', 'd3']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        res = compare_groups({'A': v, 'B': v, 'C': v}, parametric=False,
                             pair_keys={'A': keys, 'B': keys, 'C': keys})
    assert res['test'] == 'Friedman'
    assert math.isnan(res['statistic']) and math.isnan(res['p'])
    assert res['n_pairs'] == 3


def test_paired_many_with_one_pair_is_nan():
    """Only one donor in every group (n < 2): Friedman and the
    repeated-measures ANOVA both give NaN statistic and p."""
    g = {'A': [1.0, 5.0], 'B': [2.0, np.nan], 'C': [4.0, 6.0]}
    keys = {k: ['d1', 'd2'] for k in g}
    for parametric, name in ((False, 'Friedman'),
                             (True, 'Repeated-measures ANOVA')):
        res = compare_groups(g, parametric=parametric, pair_keys=keys)
        assert res['test'] == name
        assert res['n_pairs'] == 1 and res['unpaired'] == ['d2']
        assert math.isnan(res['statistic']) and math.isnan(res['p'])


def test_mann_whitney_floor_uses_both_group_sizes():
    """Unequal groups (2 and 4 values): the Mann-Whitney floor is
    2 / C(6, 2), so it needs the size of each group, not one of them
    twice."""
    a, b = [1.0, 2.0], [3.0, 4.0, 5.0, 6.0]
    res = compare_groups({'A': a, 'B': b}, parametric=False)
    assert res['test'] == 'Mann-Whitney U'
    assert res['min_p'] == pytest.approx(2.0 / comb(6, 2), rel=REL)
    assert res['underpowered'] is True
    assert res['p'] == pytest.approx(
        float(mannwhitneyu(a, b, alternative='two-sided').pvalue), rel=REL)


def test_friedman_runs_with_two_pairs():
    """Three groups, two donors: two pairs are enough for Friedman, so
    statistic and p come from SciPy rather than NaN."""
    g = {'A': [1.0, 4.0], 'B': [2.0, 6.0], 'C': [3.0, 5.0]}
    keys = {k: ['d1', 'd2'] for k in g}
    res = compare_groups(g, parametric=False, pair_keys=keys)
    want = friedmanchisquare(g['A'], g['B'], g['C'])
    assert res['n_pairs'] == 2
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)


def test_friedman_runs_when_only_some_subjects_are_constant():
    """One donor has the same value in every group, the others do not:
    Friedman is still defined and is run (only an all-constant matrix is
    left as NaN)."""
    g = {'A': [1.0, 2.0, 7.0], 'B': [1.0, 4.0, 5.0], 'C': [1.0, 3.0, 9.0]}
    keys = {k: ['d1', 'd2', 'd3'] for k in g}
    res = compare_groups(g, parametric=False, pair_keys=keys)
    want = friedmanchisquare(g['A'], g['B'], g['C'])
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)


def test_paired_group_with_no_finite_value_is_left_out_of_pairing():
    """A third group with only NaNs is not a usable group: the other two
    are paired on their own (a paired t-test over all five donors), not
    cut down to the donors the empty group shares (none)."""
    nan5 = [np.nan] * 5
    res = compare_groups({'A': A, 'B': B, 'C': nan5},
                         pair_keys={'A': DONORS, 'B': DONORS, 'C': DONORS})
    want = ttest_rel(B, A)
    assert res['n_groups'] == 2
    assert res['test'] == 'Paired t-test'
    assert res['n_pairs'] == 5 and res['unpaired'] == []
    assert res['statistic'] == pytest.approx(float(want.statistic), rel=REL)
    assert res['p'] == pytest.approx(float(want.pvalue), rel=REL)
