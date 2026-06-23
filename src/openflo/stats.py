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


def compare_groups(values_by_group, parametric=False):
    """Compare a numeric value across two or more groups.

    ``values_by_group`` : ``{group_name: [values]}`` (one value per replicate /
    sample). With exactly two groups runs a two-sided Mann-Whitney U
    (``parametric=False``, default) or Welch's t-test (``parametric=True``).
    With more than two it runs Kruskal-Wallis (non-parametric) or one-way
    ANOVA, then every pairwise post-hoc test with BH-adjusted p-values.

    Returns::

        {test, statistic, p, parametric, n_groups,
         groups: {name: {n, mean, median, sd}},
         posthoc: [{a, b, p, p_adj}]}      # empty for the two-group case

    ``test`` is None / ``p`` NaN when there isn't enough data to compute one.
    Group order is preserved from the input dict.
    """
    from scipy.stats import f_oneway, kruskal, mannwhitneyu, ttest_ind

    clean = {}
    for k, v in values_by_group.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            clean[k] = a
    names = list(clean)
    summ = {k: {'n': int(v.size), 'mean': float(np.mean(v)),
                'median': float(np.median(v)),
                'sd': float(np.std(v, ddof=1)) if v.size > 1 else 0.0}
            for k, v in clean.items()}
    out = {'test': None, 'statistic': float('nan'), 'p': float('nan'),
           'parametric': bool(parametric), 'n_groups': len(names),
           'groups': summ, 'posthoc': []}
    if len(names) < 2:
        return out

    def _pv(result):
        # scipy result objects expose .pvalue at runtime; some (TtestResult)
        # aren't fully declared in the type stubs, hence the targeted ignore.
        return float(result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]

    def _pair_p(a, b):
        """Two-sample p-value (the chosen test), NaN if it can't be computed."""
        try:
            if parametric:
                return _pv(ttest_ind(a, b, equal_var=False))
            return _pv(mannwhitneyu(a, b, alternative='two-sided'))
        except ValueError:
            return float('nan')

    arrs = [clean[n] for n in names]
    if len(names) == 2:
        a, b = arrs
        try:
            if parametric:
                r = ttest_ind(a, b, equal_var=False)
                out['test'] = 'Welch t-test'
            else:
                r = mannwhitneyu(a, b, alternative='two-sided')
                out['test'] = 'Mann-Whitney U'
            out['statistic'] = float(r.statistic)  # pyright: ignore[reportAttributeAccessIssue]
            out['p'] = _pv(r)
        except ValueError:
            pass
        return out

    # > 2 groups: omnibus test ...
    try:
        r = f_oneway(*arrs) if parametric else kruskal(*arrs)
        out['test'] = 'One-way ANOVA' if parametric else 'Kruskal-Wallis'
        out['statistic'] = float(r.statistic)  # pyright: ignore[reportAttributeAccessIssue]
        out['p'] = _pv(r)
    except ValueError:
        pass
    # ... + pairwise post-hoc with BH correction.
    pairs, raw = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pp = _pair_p(arrs[i], arrs[j])
            pairs.append({'a': names[i], 'b': names[j], 'p': pp})
            raw.append(pp)
    for pr, q in zip(pairs, _benjamini_hochberg(raw), strict=True):
        pr['p_adj'] = q
    out['posthoc'] = pairs
    return out


def compare_all_features(values_by_feature, parametric=False, eps=1e-9):
    """Run :func:`compare_groups` for EVERY feature at once (one click instead
    of stepping through populations/markers by hand) and BH-correct the omnibus
    p across features.

    ``values_by_feature`` : ``{feature: {group: [per-sample values]}}`` — e.g.
    ``{population: {group: [freqs]}}`` or ``{marker: {group: [medians]}}``.

    Returns a list (sorted by adjusted p, then raw p) of::

        {feature, test, p, p_adj, stars, n_groups, groups: {name: {...}},
         effect}              # effect = log2 fold-change (groupB/groupA) for the
                              # two-group case, else NaN — the volcano x-axis.

    The BH correction is across features, so ``p_adj`` already accounts for
    testing many populations/markers in one sweep.
    """
    feats = list(values_by_feature)
    results, raw_p = [], []
    for f in feats:
        r = compare_groups(values_by_feature[f], parametric=parametric)
        effect = float('nan')
        if r['n_groups'] == 2:
            ga, gb = list(r['groups'])                # input order preserved
            ma, mb = r['groups'][ga]['mean'], r['groups'][gb]['mean']
            effect = float(np.log2((mb + eps) / (ma + eps)))
        results.append({'feature': f, 'test': r['test'], 'p': r['p'],
                        'n_groups': r['n_groups'], 'groups': r['groups'],
                        'effect': effect})
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
    two-group effect + adjusted p are skipped. Returns a list of
    ``{feature, x, y, significant}``."""
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
    rows = sorted(df[row_factor].dropna().unique(), key=str)
    cols = sorted(df[col_factor].dropna().unique(), key=str)
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
