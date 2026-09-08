"""Differential abundance / expression between two groups of samples.

The OMIQ/Cytobank/diffcyt-style analysis OpenFlo was missing: given two
groups of samples (e.g. treated vs control), compare each population's
abundance, or each marker's expression within a population, and rank by
significance.

Two layers:

  * Pure stats — ``differential_test`` (Mann-Whitney U + log2 fold-change +
    Benjamini-Hochberg FDR) over per-sample feature values. Generic: feed
    it abundances OR expressions.
  * Builders — ``cluster_abundance`` / ``marker_expression`` turn a set of
    FlowSamples into the per-sample feature dicts the test consumes.

Everything here is numpy/scipy and unit-tested without Tk.
"""
from __future__ import annotations

import numpy as np


def _benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values for a list of raw p-values (NaNs passed
    through as NaN). Returns a list aligned to the input."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))
    # enforce monotonicity from the largest p downward
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out[order] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def differential_test(group_a, group_b, eps=1e-9):
    """Compare two groups feature-by-feature.

    `group_a` / `group_b` are dicts ``{feature: [value per sample]}`` (e.g.
    a cluster's % abundance across the samples in each group). For every
    feature present in both groups, computes the group means, log2 fold-
    change (A vs B), a two-sided Mann-Whitney U test, and a BH-adjusted
    p-value. Returns a list of row dicts sorted by raw p-value:

        {feature, mean_a, mean_b, log2fc, u, p, p_adj, n_a, n_b}
    """
    from scipy.stats import mannwhitneyu
    feats = sorted(set(group_a) & set(group_b), key=lambda f: str(f))
    rows = []
    for f in feats:
        a = np.asarray(group_a[f], dtype=float)
        b = np.asarray(group_b[f], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 1 or b.size < 1:
            continue
        ma, mb = float(np.mean(a)), float(np.mean(b))
        log2fc = float(np.log2((ma + eps) / (mb + eps)))
        try:
            u, p = mannwhitneyu(a, b, alternative='two-sided')
            u, p = float(u), float(p)
        except ValueError:           # e.g. all values identical
            u, p = float('nan'), float('nan')
        rows.append({'feature': f, 'mean_a': ma, 'mean_b': mb,
                     'log2fc': log2fc, 'u': u, 'p': p,
                     'n_a': int(a.size), 'n_b': int(b.size)})
    adj = _benjamini_hochberg([r['p'] for r in rows])
    for r, q in zip(rows, adj, strict=False):
        r['p_adj'] = q
    rows.sort(key=lambda r: (not np.isfinite(r['p']), r['p']))
    return rows


def _poisson_irls(y, X, offset, iters=50):
    """IRLS fit of a Poisson GLM (log link) with a fixed offset. Returns the
    coefficient vector."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(float(np.mean(y)), 1e-3))
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta + offset, -30, 30))
        try:
            step = np.linalg.solve((X.T * mu) @ X, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def _nb_irls(y, X, offset, alpha, iters=50):
    """IRLS fit of a negative-binomial GLM (log link, dispersion ``alpha``;
    Var = mu + alpha·mu²) with a fixed offset. Returns ``(beta, cov, mu)``."""
    beta = _poisson_irls(y, X, offset)
    w = np.exp(np.clip(X @ beta + offset, -30, 30))
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta + offset, -30, 30))
        w = mu / (1.0 + alpha * mu)
        try:
            step = np.linalg.solve((X.T * w) @ X, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = np.exp(np.clip(X @ beta + offset, -30, 30))
    w = mu / (1.0 + alpha * mu)
    cov = np.linalg.pinv((X.T * w) @ X)
    return beta, cov, mu


def _common_dispersion(counts, X, offset):
    """Method-of-moments common NB dispersion across all clusters (a stable,
    edgeR-common-dispersion-style shrinkage given few samples): fit each
    cluster with Poisson, then ``alpha = Σ((y-µ)² - µ) / Σµ²``."""
    num = den = 0.0
    for y in counts:
        beta = _poisson_irls(y, X, offset)
        mu = np.exp(np.clip(X @ beta + offset, -30, 30))
        num += float(np.sum((y - mu) ** 2 - mu))
        den += float(np.sum(mu ** 2))
    if den <= 0:
        return 0.1
    return float(min(max(num / den, 1e-6), 1e3))


def differential_abundance(counts, group, lib_sizes=None, cluster_names=None):
    """diffcyt-style differential abundance of cluster proportions between two
    groups via a **negative-binomial GLM** on counts.

    ``counts`` : ``(n_clusters, n_samples)`` array or a DataFrame (rows =
        clusters, columns = samples) of per-sample cluster counts.
    ``group``  : per-sample group label (length n_samples); exactly two
        distinct levels.
    ``lib_sizes`` : total events per sample (the GLM offset / library size).
        Defaults to each sample's column sum — correct when the clusters
        partition the cells, which accounts for composition.

    Each cluster is modelled ``count ~ group`` with ``log(library size)`` as
    offset and a shared (method-of-moments) dispersion; the group coefficient
    is Wald-tested and BH-adjusted. Returns rows sorted by p-value::

        {cluster, log2fc, prop_a, prop_b, z, p, p_adj, dispersion, n_a, n_b}

    where ``prop_*`` are the mean per-sample proportions in each group and
    ``log2fc`` is group-B-vs-A on the log2 scale."""
    from scipy.stats import norm
    if hasattr(counts, 'to_numpy'):
        cluster_names = (list(counts.index) if cluster_names is None
                         else cluster_names)
        Y = counts.to_numpy(dtype=float)
    else:
        Y = np.asarray(counts, dtype=float)
    if Y.ndim != 2:
        raise ValueError("counts must be 2-D (clusters × samples)")
    n_clusters, n = Y.shape
    if cluster_names is None:
        cluster_names = list(range(n_clusters))
    group = np.asarray(group)
    levels = list(dict.fromkeys(group.tolist()))
    if len(levels) != 2:
        raise ValueError(f"need exactly 2 groups, got {len(levels)}: {levels}")
    g01 = (group == levels[1]).astype(float)
    if lib_sizes is None:
        lib_sizes = Y.sum(axis=0)
    lib_sizes = np.clip(np.asarray(lib_sizes, dtype=float), 1.0, None)
    offset = np.log(lib_sizes)
    X = np.column_stack([np.ones(n), g01])

    alpha = _common_dispersion([Y[k] for k in range(n_clusters)], X, offset)
    a_mask, b_mask = g01 == 0, g01 == 1
    rows = []
    for k in range(n_clusters):
        y = Y[k]
        beta, cov, _mu = _nb_irls(y, X, offset, alpha)
        b = float(beta[1])
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        z = b / se if se > 0 else 0.0
        p = float(2 * norm.sf(abs(z))) if se > 0 else float('nan')
        prop_a = float(np.mean(y[a_mask] / lib_sizes[a_mask]))
        prop_b = float(np.mean(y[b_mask] / lib_sizes[b_mask]))
        rows.append({'cluster': cluster_names[k],
                     'log2fc': b / np.log(2.0),
                     'prop_a': prop_a, 'prop_b': prop_b,
                     'z': z, 'p': p, 'dispersion': alpha,
                     'n_a': int(a_mask.sum()), 'n_b': int(b_mask.sum()),
                     'group_a': str(levels[0]), 'group_b': str(levels[1])})
    adj = _benjamini_hochberg([r['p'] for r in rows])
    for r, q in zip(rows, adj, strict=False):
        r['p_adj'] = q
    rows.sort(key=lambda r: (not np.isfinite(r['p']), r['p']))
    return rows


def _sample_label_fractions(sample, label_col):
    """{label_value: fraction of events} for one sample's `label_col`."""
    df = sample.data
    if label_col not in df.columns or len(df) == 0:
        return {}
    vc = df[label_col].value_counts(normalize=True)
    return {k: float(v) * 100.0 for k, v in vc.items()}


def cluster_abundance(samples_a, samples_b, label_col='cluster'):
    """Per-population abundance (% of events) per sample, grouped.

    Returns ``(group_a, group_b)`` dicts ``{label_value: [pct per sample]}``
    suitable for ``differential_test``. Labels absent from a sample count
    as 0% for that sample (so a population present in one group but not the
    other still shows up)."""
    groups = []
    all_labels = set()
    per_group_fracs = []
    for samples in (samples_a, samples_b):
        fracs = [_sample_label_fractions(s, label_col) for s in samples]
        per_group_fracs.append(fracs)
        for f in fracs:
            all_labels.update(f)
    for fracs in per_group_fracs:
        g = {lbl: [f.get(lbl, 0.0) for f in fracs] for lbl in all_labels}
        groups.append(g)
    return groups[0], groups[1]


def marker_expression(samples_a, samples_b, channels, label_col=None,
                      label_value=None, stat='median'):
    """Per-marker expression per sample, grouped.

    For each channel, take the per-sample summary statistic (`median` or
    `mean`) over events — optionally restricted to a single population
    (`label_col == label_value`). Returns ``(group_a, group_b)`` dicts
    ``{channel: [value per sample]}`` for ``differential_test``."""
    agg = np.median if stat == 'median' else np.mean
    groups = []
    for samples in (samples_a, samples_b):
        g = {ch: [] for ch in channels}
        for s in samples:
            df = s.data
            if label_col is not None and label_col in df.columns:
                df = df[df[label_col] == label_value]
            for ch in channels:
                if ch in df.columns and len(df):
                    vals = np.asarray(df[ch].values, dtype=float)
                    vals = vals[np.isfinite(vals)]
                    g[ch].append(float(agg(vals)) if vals.size else float('nan'))
                else:
                    g[ch].append(float('nan'))
        groups.append(g)
    return groups[0], groups[1]
