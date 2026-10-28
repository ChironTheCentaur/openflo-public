"""Group-comparison statistics + GraphPad Prism-ready table shaping.

Two layers, both pure (numpy / scipy / pandas — no Tk):

  * ``compare_groups`` — pick the right test for the number of groups
    (Mann-Whitney U / Welch t for two; Kruskal-Wallis / one-way ANOVA for
    more, with BH-adjusted pairwise post-hoc), and report a tidy result dict.
  * ``to_prism_column`` / ``to_prism_grouped`` — reshape per-replicate values
    into the Column and Grouped table layouts you paste straight into GraphPad
    Prism (columns = groups, rows = replicates), padding ragged groups with
    blanks. ``p_to_stars`` maps a p-value to the usual significance asterisks.

These feed the frequency / comparison and violin/ridgeline plots, but are
generic — give them any per-sample metric.
"""
from __future__ import annotations

import numpy as np


def _benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values (NaNs passed through). Aligned to input."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U of ``a`` against ``b``: ``(U, p)``, the same
    on every supported SciPy.

    When every value in both groups is the same, the groups cannot differ
    and p is 1. SciPy gave 1 there until 1.18, which returns NaN instead, so
    a cluster with an identical share in both conditions turned from "no
    difference" into NA depending on the installed SciPy. Only that case is
    changed; anything else, an error included, is SciPy's own answer."""
    from scipy.stats import mannwhitneyu
    r = mannwhitneyu(a, b, alternative='two-sided')
    u, p = float(r.statistic), float(r.pvalue)
    if np.isnan(p):
        both = np.concatenate([np.ravel(np.asarray(a, dtype=float)),
                               np.ravel(np.asarray(b, dtype=float))])
        if both.size and np.all(both == both[0]):
            p = 1.0
    return u, p


# ── transforms and the smallest p a test can give ─────────────────────────
#
# Rank tests can't reach significance with the group sizes flow experiments
# use: Mann-Whitney's smallest two-sided p is 2 / C(n_a + n_b, n_a), 0.1 at
# 3 vs 3, so two groups that don't overlap at all read "not significant".
# A t-test on transformed values has no such floor, and is the usual
# choice at those sizes: proportions on the logit scale (Warton and Hui
# 2011), intensities on the arcsinh scale, counts on the log scale.

TRANSFORMS = ('logit', 'log1p', 'asinh')
LOGIT_FLOOR = 1e-4          # 0.01 %: where 0 % and 100 % are clipped to
ASINH_COFACTOR = 150.0
ALPHA = 0.05


def transform_values(values, transform):
    """`values` on the scale a test should compare them on. ``'logit'`` takes
    percentages (0 to 100) to log odds, clipping 0 % and 100 % to
    LOGIT_FLOOR from the edge; ``'log1p'`` takes counts to log(1 + n);
    ``'asinh'`` takes intensities (negative ones too, after compensation)
    to arcsinh(x / 150). None leaves them as they are. NaN stays NaN, and a
    value the transform can't take (a negative percentage or count) becomes
    NaN, so the test leaves it out."""
    a = np.asarray(values, dtype=float)
    if transform is None:
        return a
    with np.errstate(invalid='ignore', divide='ignore'):
        if transform == 'logit':
            ok = (a >= 0) & (a <= 100)
            f = np.clip(a / 100.0, LOGIT_FLOOR, 1.0 - LOGIT_FLOOR)
            return np.where(ok, np.log(f / (1.0 - f)), np.nan)
        if transform == 'log1p':
            return np.where(a >= 0, np.log1p(a), np.nan)
        if transform == 'asinh':
            return np.arcsinh(a / ASINH_COFACTOR)
    raise ValueError(f"unknown transform {transform!r}; "
                     f"one of {TRANSFORMS} or None")


def smallest_p(test, sizes):
    """The smallest two-sided p-value `test` can give with these group
    sizes (`sizes`: the two group sizes, or the number of pairs as one
    item), from its exact null distribution: Mann-Whitney U
    ``2 / C(n_a + n_b, n_a)``, Wilcoxon signed-rank ``2 / 2^n``. A t-test
    has no floor (0). NaN for a test without a simple floor."""
    from math import comb
    name = str(test or '')
    try:
        if name.startswith('Mann-Whitney'):
            na, nb = (int(x) for x in sizes)
            return min(1.0, 2.0 / comb(na + nb, na)) if na and nb else 1.0
        if name.startswith('Wilcoxon'):
            n = int(sizes[0])
            return min(1.0, 2.0 / 2.0 ** n) if n else 1.0
    except (TypeError, ValueError):
        return float('nan')
    if 't-test' in name:
        return 0.0
    return float('nan')


def pair_values(values_by_group, pair_keys):
    """Line up paired values by their keys (a donor, say).

    ``values_by_group`` ``{group: [values]}`` and ``pair_keys`` ``{group:
    [key per value]}``, aligned. Returns ``(keys, matrix, dropped)``:
    the keys every group has exactly once with a finite value, a
    ``(n_groups, n_keys)`` array of their values in group order, and the
    keys left out (missing in a group, or given twice in one)."""
    groups = list(values_by_group)
    per = []
    every = set()
    for g in groups:
        vals = np.asarray(values_by_group[g], dtype=float)
        keys = list((pair_keys or {}).get(g) or [])
        if len(keys) != vals.size:
            raise ValueError(f"group {g!r}: {vals.size} values but "
                             f"{len(keys)} pair keys")
        seen, twice = {}, set()
        for k, v in zip(keys, vals, strict=True):
            if k in seen:
                twice.add(k)
            seen[k] = v
        for k in twice:
            seen.pop(k)
        per.append({k: v for k, v in seen.items() if np.isfinite(v)})
        every.update(keys)
    common = [k for k in sorted(every, key=str)
              if all(k in d for d in per)]
    mat = np.array([[d[k] for k in common] for d in per], dtype=float)
    dropped = [k for k in sorted(every, key=str) if k not in common]
    return common, mat.reshape(len(groups), len(common)), dropped


def _rm_anova(mat):
    """One-way repeated-measures ANOVA of a ``(k groups, n subjects)``
    matrix: ``(F, p, df_group, df_error)``."""
    from scipy.stats import f as f_dist
    k, n = mat.shape
    grand = mat.mean()
    ss_group = n * np.sum((mat.mean(axis=1) - grand) ** 2)
    ss_subj = k * np.sum((mat.mean(axis=0) - grand) ** 2)
    ss_total = np.sum((mat - grand) ** 2)
    ss_err = ss_total - ss_group - ss_subj
    df_g, df_e = k - 1, (k - 1) * (n - 1)
    if df_e <= 0:
        return float('nan'), float('nan'), df_g, df_e
    if ss_err <= 0:
        return ((float('inf'), 0.0, df_g, df_e) if ss_group > 0
                else (float('nan'), 1.0, df_g, df_e))
    F = (ss_group / df_g) / (ss_err / df_e)
    return float(F), float(f_dist.sf(F, df_g, df_e)), df_g, df_e


def _paired_two(a, b, parametric):
    """Paired test of matched arrays: ``(test name, statistic, p)``."""
    from scipy.stats import ttest_rel, wilcoxon
    d = b - a
    if parametric:
        if d.size < 2:
            return 'Paired t-test', float('nan'), float('nan')
        if np.all(d == d[0]):
            # No spread in the differences: t is 0/0 or x/0. Every pair
            # equal is no difference (p 1, as mann_whitney); a shift the
            # same in every pair has no t to test, so it is NA.
            return 'Paired t-test', float('nan'), (1.0 if d[0] == 0
                                                   else float('nan'))
        r = ttest_rel(b, a)
        return ('Paired t-test', float(r.statistic),  # pyright: ignore[reportAttributeAccessIssue]
                float(r.pvalue))  # pyright: ignore[reportAttributeAccessIssue]
    if d.size < 1:
        return 'Wilcoxon signed-rank', float('nan'), float('nan')
    if np.all(d == 0):
        return 'Wilcoxon signed-rank', 0.0, 1.0
    r = wilcoxon(b, a)
    return ('Wilcoxon signed-rank', float(r.statistic),  # pyright: ignore[reportAttributeAccessIssue]
            float(r.pvalue))  # pyright: ignore[reportAttributeAccessIssue]


def p_to_stars(p):
    """GraphPad-style significance asterisks for a p-value.
    ≤1e-4 ****, ≤1e-3 ***, ≤1e-2 **, ≤0.05 *, else 'ns' (NaN → '')."""
    if p is None or not np.isfinite(p):
        return ''
    if p <= 1e-4:
        return '****'
    if p <= 1e-3:
        return '***'
    if p <= 1e-2:
        return '**'
    if p <= 0.05:
        return '*'
    return 'ns'


def _clean_groups(groups):
    """Drop non-finite values; keep only groups with >=1 finite value.
    Returns ``(names, arrays)`` with input order preserved."""
    names, arrs = [], []
    for k, v in groups.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            names.append(k)
            arrs.append(a)
    return names, arrs


def multi_group_test(groups, paired=False):
    """Omnibus test across two or more groups.

    ``groups`` : ``{label: [values]}`` (>=2 groups, one value per
    replicate/sample). ``paired=False`` (default) runs Kruskal-Wallis;
    ``paired=True`` runs the Friedman test (>=3 groups; groups are TRUNCATED to
    the shortest length by position, not required to be equal-length). Non-finite
    values are dropped per group. For the paired case, after truncation, rows
    with any NaN are dropped listwise so the groups stay aligned.

    Returns ``{'test', 'stat', 'p', 'k'}`` where ``k`` is the number of usable
    groups. Degenerate inputs (<2 usable groups, all values identical, too few
    groups/replicates for the chosen test) return ``stat``/``p`` as NaN with a
    descriptive ``test`` label rather than raising.
    """
    from scipy.stats import friedmanchisquare, kruskal

    if paired:
        # Listwise-complete rows so every group has the same aligned length.
        labels = list(groups)
        raw = [np.asarray(groups[k], dtype=float) for k in labels]
        keep = [r for r in raw if r.size]
        labels = [k for k, r in zip(labels, raw, strict=True) if r.size]
        out = {'test': 'Friedman', 'stat': float('nan'),
               'p': float('nan'), 'k': len(labels)}
        if len(keep) < 3:
            return out  # Friedman needs >=3 related groups
        n = min(r.size for r in keep)
        mat = np.vstack([r[:n] for r in keep])           # (k, n)
        mask = np.all(np.isfinite(mat), axis=0)
        mat = mat[:, mask]
        if mat.shape[1] < 2:
            return out
        if np.all(np.ptp(mat, axis=0) == 0):   # no within-subject variation → undefined
            return out
        try:
            r = friedmanchisquare(*list(mat))
            out['stat'] = float(r.statistic)
            out['p'] = float(r.pvalue)
        except ValueError:
            pass
        return out

    names, arrs = _clean_groups(groups)
    out = {'test': 'Kruskal-Wallis', 'stat': float('nan'),
           'p': float('nan'), 'k': len(names)}
    if len(names) < 2:
        return out
    allv = np.concatenate(arrs)
    if np.allclose(allv, allv[0]):                       # all values identical
        return out
    try:
        r = kruskal(*arrs)
        out['stat'] = float(r.statistic)
        out['p'] = float(r.pvalue)
    except ValueError:
        pass
    return out


def posthoc_pairwise(groups):
    """All pairwise two-sided Mann-Whitney U tests with BH-adjusted p-values.

    ``groups`` : ``{label: [values]}``. Non-finite values are dropped per
    group; groups with no finite values are skipped. The Benjamini-Hochberg
    adjustment is applied across the full set of pairwise p-values (reusing
    :func:`_benjamini_hochberg`).

    Returns a list of ``{'a', 'b', 'stat', 'p', 'p_adj'}`` in upper-triangular
    order of the (usable) group labels. ``stat``/``p`` are NaN for any pair the
    test can't be computed for; two groups of one identical value give
    p = 1 (see mann_whitney).
    """
    names, arrs = _clean_groups(groups)
    pairs, raw = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            stat, p = float('nan'), float('nan')
            try:
                stat, p = mann_whitney(arrs[i], arrs[j])
            except ValueError:
                pass
            pairs.append({'a': names[i], 'b': names[j], 'stat': stat, 'p': p})
            raw.append(p)
    for pr, q in zip(pairs, _benjamini_hochberg(raw), strict=True):
        pr['p_adj'] = q
    return pairs


def effect_size(a, b):
    """Effect sizes for two samples ``a`` vs ``b`` (non-finite values dropped).

    Returns ``{'cliffs_delta', 'cohens_d'}``:

      * Cliff's delta — nonparametric, in ``[-1, 1]``: ``P(a>b) - P(a<b)``.
        Positive means ``a`` tends to exceed ``b``.
      * Cohen's d — standardized mean difference ``(mean(a)-mean(b))/s_pooled``
        using the pooled SD. Positive means ``a`` has the larger mean.

    Either value is NaN when it can't be computed (empty group, or zero pooled
    SD for Cohen's d).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    res = {'cliffs_delta': float('nan'), 'cohens_d': float('nan')}
    if x.size == 0 or y.size == 0:
        return res
    # Cliff's delta via pairwise sign comparison (broadcast).
    diff = np.sign(x[:, None] - y[None, :])
    res['cliffs_delta'] = float(diff.sum() / (x.size * y.size))
    nx, ny = x.size, y.size
    if nx + ny > 2:
        vx = np.var(x, ddof=1) if nx > 1 else 0.0
        vy = np.var(y, ddof=1) if ny > 1 else 0.0
        sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        if sp > 0:
            res['cohens_d'] = float((np.mean(x) - np.mean(y)) / sp)
    return res


def compare_groups(values_by_group, parametric=True, transform=None,
                   pair_keys=None):
    """Compare a numeric value across two or more groups.

    ``values_by_group`` : ``{group_name: [values]}`` (one value per replicate /
    sample). With exactly two groups runs Welch's t-test
    (``parametric=True``, default) or a two-sided Mann-Whitney U rank test
    (``parametric=False``). With more than two it runs one-way ANOVA or
    Kruskal-Wallis, then every pairwise post-hoc test with BH-adjusted
    p-values. The t-test is the default because at 3 to 5 samples a group
    a rank test can barely, or never, reach p < 0.05 (smallest_p).

    ``transform`` (one of TRANSFORMS) puts the values on the scale the test
    compares them on (transform_values). It is None by default because bare
    values don't say what they are; pass ``'logit'`` for percentages and
    ``'log1p'`` (counts) or ``'asinh'`` (intensities) otherwise to get the
    test the app's comparison windows run by default. The group summaries
    stay in the values' own units.

    ``pair_keys`` ``{group: [key per value]}`` (a donor or animal per
    sample, aligned with the values) makes it a paired design: values are
    matched by key, keys missing from a group are left out (``unpaired``),
    and two groups get a paired t-test (parametric) or Wilcoxon
    signed-rank; more get a repeated-measures ANOVA or Friedman, with
    paired post-hoc tests.

    Returns::

        {test, statistic, p, parametric, n_groups, transform, paired,
         n_pairs, unpaired, min_p, underpowered,
         groups: {name: {n, mean, median, sd}},
         posthoc: [{a, b, p, p_adj}]}      # empty for the two-group case

    ``min_p`` is the smallest p the test could give with these group sizes
    (smallest_p; 0 for a t-test, NaN where there is no simple floor), and
    ``underpowered`` is True when even that is above 0.05: the comparison
    cannot come out significant, however different the groups are.
    ``test`` is None / ``p`` NaN when there isn't enough data to compute one.
    Group order is preserved from the input dict.
    """
    from scipy.stats import f_oneway, friedmanchisquare, kruskal, ttest_ind

    clean = {}
    for k, v in values_by_group.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            clean[k] = a
    names = list(clean)
    # The sample SD of ONE value is undefined (ddof=1 divides by n-1 = 0).
    # Returning 0.0 there reads as "this group had zero variability" — a
    # confident, wrong claim, and it is rendered straight into the group
    # summary as `sd=0`. NaN is the honest answer and displays as `nan`.
    summ = {k: {'n': int(v.size), 'mean': float(np.mean(v)),
                'median': float(np.median(v)),
                'sd': (float(np.std(v, ddof=1)) if v.size > 1
                       else float('nan'))}
            for k, v in clean.items()}
    paired = pair_keys is not None
    out = {'test': None, 'statistic': float('nan'), 'p': float('nan'),
           'parametric': bool(parametric), 'n_groups': len(names),
           'transform': transform, 'paired': paired, 'n_pairs': None,
           'unpaired': [], 'min_p': float('nan'), 'underpowered': False,
           'groups': summ, 'posthoc': []}
    if len(names) < 2:
        return out
    suffix = f" ({transform})" if transform else ''

    def _done(test, stat, p, sizes):
        out['test'] = test + suffix
        out['statistic'], out['p'] = float(stat), float(p)
        out['min_p'] = smallest_p(test, sizes)
        out['underpowered'] = bool(np.isfinite(out['min_p'])
                                   and out['min_p'] > ALPHA)
        return out

    def _bh_pairs(pairs):
        for pr, q in zip(pairs, _benjamini_hochberg([x['p'] for x in pairs]),
                         strict=True):
            pr['p_adj'] = q
        out['posthoc'] = pairs

    if paired:
        tv = {g: transform_values(values_by_group[g], transform)
              for g in values_by_group
              if g in clean}
        keys = {g: (pair_keys or {}).get(g) or [] for g in tv}
        _keys, mat, dropped = pair_values(tv, keys)
        n = mat.shape[1]
        out['n_pairs'], out['unpaired'] = n, dropped
        if len(names) == 2:
            test, stat, p = _paired_two(mat[0], mat[1], parametric)
            return _done(test, stat, p, [n])
        try:
            if parametric:
                F, p, _dg, _de = _rm_anova(mat)
                test, stat = 'Repeated-measures ANOVA', F
            else:
                test, stat, p = 'Friedman', float('nan'), float('nan')
                if n >= 2 and not np.all(np.ptp(mat, axis=0) == 0):
                    r = friedmanchisquare(*list(mat))
                    stat = float(r.statistic)  # pyright: ignore[reportAttributeAccessIssue]
                    p = float(r.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
        except ValueError:
            test, stat, p = ('Repeated-measures ANOVA' if parametric
                             else 'Friedman'), float('nan'), float('nan')
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                _t, _s, pp = _paired_two(mat[i], mat[j], parametric)
                pairs.append({'a': names[i], 'b': names[j], 'p': pp})
        _bh_pairs(pairs)
        return _done(test, stat, p, [n])

    arrs = []
    for nm in names:
        t = transform_values(clean[nm], transform)
        arrs.append(t[np.isfinite(t)])

    def _pv(result):
        # scipy result objects expose .pvalue at runtime; some (TtestResult)
        # aren't fully declared in the type stubs, hence the targeted ignore.
        return float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]

    def _pair_p(a, b):
        """Two-sample p-value (the chosen test), NaN if it can't be computed."""
        try:
            if parametric:
                return _pv(ttest_ind(a, b, equal_var=False))
            return mann_whitney(a, b)[1]
        except ValueError:
            return float('nan')

    if len(names) == 2:
        a, b = arrs
        sizes = [a.size, b.size]
        try:
            if parametric:
                r = ttest_ind(a, b, equal_var=False)
                return _done('Welch t-test',
                             r.statistic,  # pyright: ignore[reportAttributeAccessIssue]
                             _pv(r), sizes)
            stat, p = mann_whitney(a, b)
            return _done('Mann-Whitney U', stat, p, sizes)
        except ValueError:
            return out

    # > 2 groups: omnibus test ...
    try:
        r = f_oneway(*arrs) if parametric else kruskal(*arrs)
        _done('One-way ANOVA' if parametric else 'Kruskal-Wallis',
              r.statistic,  # pyright: ignore[reportAttributeAccessIssue]
              _pv(r), [x.size for x in arrs])
    except ValueError:
        pass
    # ... + pairwise post-hoc with BH correction.
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append({'a': names[i], 'b': names[j],
                          'p': _pair_p(arrs[i], arrs[j])})
    _bh_pairs(pairs)
    return out


def log2_fold_change(mean_a, mean_b, pseudocount=None):
    """``log2(mean_b / mean_a)``: B over A, the one direction every OpenFlo
    fold change uses (differential_test, compare_all_features,
    differential_abundance, and the volcano's 'log2 fold-change (B / A)').

    No pseudo-count by default. A quantity absent from one group (mean 0)
    gives ``+inf`` (absent from A) or ``-inf`` (absent from B); 0 in both
    gives NaN. The fixed 1e-9 floor this replaced turned a population at 0 %
    in one group and 0.6 % in the other into log2FC +29.2, a number nothing
    in the data supports, which then set the volcano's x range. A negative
    mean (a dim, compensated intensity) has no fold change: NaN, since the
    ratio of two negatives reads like a real one.

    ``pseudocount``, when given, is added to both means first: a choice the
    caller makes, in the data's own units."""
    a, b = float(mean_a), float(mean_b)
    if pseudocount is not None:
        a, b = a + float(pseudocount), b + float(pseudocount)
    if not (np.isfinite(a) and np.isfinite(b)) or a < 0 or b < 0:
        return float('nan')
    if a == 0:
        return float('nan') if b == 0 else float('inf')
    if b == 0:
        return float('-inf')
    return float(np.log2(b / a))


def compare_all_features(values_by_feature, parametric=True, eps=None,
                         transform=None, pair_keys=None):
    """Run :func:`compare_groups` for EVERY feature at once (one click instead
    of stepping through populations/markers by hand) and BH-correct the omnibus
    p across features.

    ``values_by_feature`` : ``{feature: {group: [per-sample values]}}`` — e.g.
    ``{population: {group: [freqs]}}`` or ``{marker: {group: [medians]}}``.

    Returns a list (sorted by adjusted p, then raw p) of::

        {feature, test, p, p_adj, stars, n_groups, groups: {name: {...}},
         effect}              # effect = log2 fold-change (groupB/groupA) for the
                              # two-group case, else NaN — the volcano x-axis.

    ``effect`` is :func:`log2_fold_change` of the two group means: ±inf for
    a feature absent (mean 0) from one group, NaN for a negative mean. It is
    given only when the input has exactly two groups and both hold data, so
    B and A are always the second and first group passed in. ``eps``, when
    given, is a pseudo-count added to both means (none by default).

    The BH correction is across features, so ``p_adj`` already accounts for
    testing many populations/markers in one sweep.

    ``parametric``, ``transform`` and ``pair_keys`` (``{feature: {group:
    [keys]}}``) are passed to compare_groups for each feature: Welch's
    t-test / ANOVA by default, the rank tests with ``parametric=False``.
    Pass ``transform`` (``'logit'`` for percentages, ``'log1p'`` or
    ``'asinh'`` for counts or intensities) to match the app's default test.
    Each row also carries its ``min_p``, ``underpowered``, ``n_pairs`` and
    ``unpaired``.
    """
    feats = list(values_by_feature)
    results, raw_p = [], []
    for f in feats:
        r = compare_groups(values_by_feature[f], parametric=parametric,
                           transform=transform,
                           pair_keys=(None if pair_keys is None
                                      else pair_keys.get(f, {})))
        effect = float('nan')
        if r['n_groups'] == 2 and len(values_by_feature[f]) == 2:
            ga, gb = list(r['groups'])                # input order preserved
            ma, mb = r['groups'][ga]['mean'], r['groups'][gb]['mean']
            effect = log2_fold_change(ma, mb, eps)
        results.append({'feature': f, 'test': r['test'], 'p': r['p'],
                        'n_groups': r['n_groups'], 'groups': r['groups'],
                        'effect': effect, 'min_p': r['min_p'],
                        'underpowered': r['underpowered'],
                        'n_pairs': r['n_pairs'], 'unpaired': r['unpaired']})
        raw_p.append(r['p'])
    for res, q in zip(results, _benjamini_hochberg(raw_p), strict=True):
        res['p_adj'] = q
        res['stars'] = p_to_stars(q)
    results.sort(key=lambda d: (
        np.inf if not np.isfinite(d['p_adj']) else d['p_adj'],
        np.inf if not np.isfinite(d['p']) else d['p']))
    return results


def volcano_data(results, alpha=0.05, effect_cut=1.0):
    """Volcano-plot points from :func:`compare_all_features` results: ``x`` =
    effect (log2 fold-change), ``y`` = ``-log10(p_adj)``, ``significant`` =
    ``p_adj <= alpha and |effect| >= effect_cut``. Features without a finite
    two-group effect + adjusted p are skipped, including one absent from a
    group (effect ±inf: it has no place on a log axis; the results table
    lists it). Returns a list of ``{feature, x, y, significant}``."""
    pts = []
    for r in results:
        eff, padj = r.get('effect'), r.get('p_adj')
        if eff is None or padj is None or not (np.isfinite(eff)
                                               and np.isfinite(padj)):
            continue
        y = -float(np.log10(max(float(padj), 1e-300)))
        pts.append({'feature': r['feature'], 'x': float(eff), 'y': y,
                    'significant': bool(padj <= alpha and abs(eff) >= effect_cut)})
    return pts


def to_prism_column(values_by_group):
    """Prism **Column** table: one column per group, rows = replicates, ragged
    groups padded with NaN to the longest. Paste into a Prism Column table for
    an unpaired t-test / one-way ANOVA. Returns a ``pandas.DataFrame``."""
    import pandas as pd
    maxn = max((len(v) for v in values_by_group.values()), default=0)
    data = {str(k): list(v) + [np.nan] * (maxn - len(v))
            for k, v in values_by_group.items()}
    return pd.DataFrame(data)


def group_kde(values_by_group, gridsize=256, pad_frac=0.05):
    """Kernel-density estimate per group over a shared x-grid — the data behind
    a ridgeline / violin plot.

    ``values_by_group`` : ``{group: [per-cell values]}``. Returns
    ``(x, {group: density})`` where ``x`` is a ``(gridsize,)`` grid spanning all
    groups (padded by ``pad_frac`` of the range) and each density is evaluated
    on it. Groups with <2 finite values or zero variance get an all-zero curve
    (so the caller can still lay out a row for them). Empty input → ``([], {})``.
    """
    from scipy.stats import gaussian_kde
    clean = {}
    for k, v in values_by_group.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            clean[k] = a
    if not clean:
        return np.array([]), {}
    allv = np.concatenate(list(clean.values()))
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * pad_frac + 1e-9
    x = np.linspace(lo - pad, hi + pad, gridsize)
    out = {}
    for k, v in clean.items():
        if v.size < 2 or float(np.std(v)) == 0.0:
            out[k] = np.zeros_like(x)
            continue
        try:
            out[k] = gaussian_kde(v)(x)
        except Exception:
            out[k] = np.zeros_like(x)
    return x, out


def to_prism_grouped(df, row_factor, col_factor, value):
    """Prism **Grouped** table from a tidy DataFrame.

    Rows = the levels of ``row_factor`` (e.g. Day); column groups = the levels
    of ``col_factor`` (e.g. Stim vs Ctrl); within each column group the
    replicate values become sub-columns (padded with NaN). Returns a DataFrame
    with a 2-level ``MultiIndex`` on the columns — written to CSV it produces
    the two-row header Prism's Grouped paste expects."""
    import pandas as pd
    # Keep replicates whose row/col factor is missing under an explicit
    # "(unassigned)" level instead of silently dropping them from the table —
    # a sample with a blank Day/condition still shows up, clearly flagged.
    df = df.copy()
    for _fac in (row_factor, col_factor):
        if df[_fac].isna().any():
            df[_fac] = df[_fac].fillna('(unassigned)')
    rows = sorted(df[row_factor].unique(), key=str)
    cols = sorted(df[col_factor].unique(), key=str)
    grouped = df.groupby([row_factor, col_factor])[value].apply(list)
    maxrep = max((len(v) for v in grouped), default=1)
    data = {(str(c), r + 1): [] for c in cols for r in range(maxrep)}
    for r in rows:
        for c in cols:
            vals = grouped.get((r, c), [])
            for k in range(maxrep):
                data[(str(c), k + 1)].append(vals[k] if k < len(vals)
                                             else np.nan)
    res = pd.DataFrame(data, index=pd.Index(rows, name=row_factor))
    res.columns = pd.MultiIndex.from_tuples(res.columns,
                                            names=[col_factor, 'rep'])
    return res
