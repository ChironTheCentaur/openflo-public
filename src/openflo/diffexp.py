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


def differential_test(group_a, group_b, eps=None):
    """Compare two groups feature-by-feature.

    `group_a` / `group_b` are dicts ``{feature: [value per sample]}`` (e.g.
    a cluster's % abundance across the samples in each group). For every
    feature present in both groups, computes the group means, log2 fold-
    change (B over A), a two-sided Mann-Whitney U test, and a BH-adjusted
    p-value. Returns a list of row dicts sorted by raw p-value:

        {feature, mean_a, mean_b, log2fc, u, p, p_adj, n_a, n_b}

    ``log2fc = log2(mean_b / mean_a)`` (stats.log2_fold_change), the
    direction of compare_all_features, differential_abundance and the
    volcano's axis; until 2.6.1 this one function alone gave A over B, so
    one comparison had opposite signs in two places. A fold change needs
    positive quantities: a negative group mean (a dim, compensated
    intensity) gives NaN; a mean of 0 in one group gives ±inf (no 1e-9
    floor, which turned 0 % vs 0.6 % into -29.2). ``eps``, when given, is a
    pseudo-count added to both means."""
    from scipy.stats import mannwhitneyu

    from .stats import log2_fold_change
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
        log2fc = log2_fold_change(ma, mb, eps)
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


def _nb_group_rates(Y, lib, labels, n_groups, alpha, lr=None, iters=100):
    """Maximum-likelihood negative-binomial rate (events per library event)
    of every group, for every population at once.

    This is the fit of ``count ~ group`` with a ``log(library size)``
    offset: with one coefficient per group (the intercept and the group
    term of a two-group design), each group's log rate has its own score,
    ``sum_i (y_i - mu_i) / (1 + alpha mu_i)``, solved here by Fisher scoring
    (information ``sum_i mu_i / (1 + alpha mu_i)``).

    The NB weight ``1/(1 + alpha mu)`` on the residual is what the earlier
    IRLS left out: it solved the POISSON score, so with library sizes that
    differ inside a group it returned the Poisson rate (group effect 0.643
    where the NB MLE is 0.170, at alpha 0.3) and called it NB.

    ``Y`` (K, n); ``lib`` (n,); ``labels`` (n,) ints in ``range(n_groups)``;
    ``alpha`` (K,). Returns log rates (K, n_groups); ``-inf`` for a group
    with no events in that population (its MLE rate is 0). ``lr`` warm
    starts the iteration."""
    K = Y.shape[0]
    out = np.empty((K, n_groups))
    a = np.asarray(alpha, dtype=float)[:, None]
    for g in range(n_groups):
        m = labels == g
        y, L = Y[:, m], lib[m]
        tot = y.sum(axis=1)
        has = tot > 0
        # The Poisson MLE, exact when alpha = 0 or the library sizes are equal.
        start = np.log(np.where(has, tot, 1.0) / L.sum())
        cur = start if lr is None else np.where(np.isfinite(lr[:, g]),
                                                lr[:, g], start)
        for _ in range(iters):
            mu = L[None, :] * np.exp(cur)[:, None]
            den = 1.0 + a * mu
            score = np.sum((y - mu) / den, axis=1)
            info = np.sum(mu / den, axis=1)
            step = np.clip(score / info, -2.0, 2.0)
            cur = cur + step
            if np.max(np.abs(step[has]), initial=0.0) < 1e-10:
                break
        out[:, g] = np.where(has, cur, -np.inf)
    return out


def _nb_mu(lr, lib, labels):
    """Fitted means ``lib_i * rate_group(i)`` (K, n) from log rates."""
    return lib[None, :] * np.exp(lr[:, labels])


def _pearson(Y, mu, alpha):
    """Pearson statistic ``sum (y-mu)^2 / (mu (1 + alpha mu))`` per
    population; a cell fitted at exactly 0 (a group with no events)
    contributes nothing."""
    a = np.asarray(alpha, dtype=float)[:, None]
    pos = mu > 0
    safe = np.where(pos, mu, 1.0)
    return np.sum(np.where(pos, (Y - mu) ** 2 / (safe * (1.0 + a * safe)),
                           0.0), axis=1)


def _nb_dispersion(Y, lib, labels, n_groups):
    """Each population's own NB dispersion, and its residual degrees of
    freedom: the ``alpha`` at which the Pearson statistic of the
    ``count ~ group`` fit equals those df (``alpha`` 0 when even Poisson
    variance is more than the data show; NaN with no df).

    The df are ``n_g - 1`` summed over the groups the population has events
    in: a group with none is fitted exactly (mean 0) and has no residuals
    to measure spread with.

    It replaced ONE method-of-moments dispersion for every population, which
    had no df correction (true alpha 0.1 estimated 0.044 at 3 vs 3) and
    which nested gates (Cells, Singlets, Live: biological CV 2-3 %) pulled
    to 0.0008, so a rare subset (CV 40 %) was tested as almost Poisson: 84 %
    of null comparisons came out significant at the 5 % level. Per
    population, with a t reference on these df (differential_abundance),
    the same simulations give 4-6 %."""
    K = Y.shape[0]
    df = np.zeros(K)
    for g in range(n_groups):
        m = labels == g
        df += np.where(Y[:, m].sum(axis=1) > 0, m.sum() - 1, 0)
    zero = np.zeros(K)
    lr = _nb_group_rates(Y, lib, labels, n_groups, zero)
    p0 = _pearson(Y, _nb_mu(lr, lib, labels), zero)
    alpha = np.where(df > 0, 0.0, np.nan)
    need = (df > 0) & (p0 > df)
    if need.any():
        # Bisection on log10(alpha) for every population at once: the
        # Pearson statistic falls as alpha grows.
        Yn, dn, lrn = Y[need], df[need], lr[need]
        lo = np.full(Yn.shape[0], -10.0)
        hi = np.full(Yn.shape[0], 4.0)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            a = 10.0 ** mid
            lrn = _nb_group_rates(Yn, lib, labels, n_groups, a, lr=lrn)
            above = _pearson(Yn, _nb_mu(lrn, lib, labels), a) > dn
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        alpha[need] = 10.0 ** (0.5 * (lo + hi))
    return alpha, df


def differential_abundance(counts, group, lib_sizes=None, cluster_names=None):
    """diffcyt-style differential abundance of cluster proportions between two
    groups via a **negative-binomial GLM** on counts.

    ``counts`` : ``(n_clusters, n_samples)`` array or a DataFrame (rows =
        clusters, columns = samples) of per-sample cluster counts.
    ``group``  : per-sample group label (length n_samples); exactly two
        distinct levels. The first level to appear is group A, the other B.
    ``lib_sizes`` : total events per sample (the GLM offset / library size).
        Defaults to each sample's column sum — correct when the clusters
        partition the cells, which accounts for composition. Pass the total
        events when the rows are nested gates (Cells > Live > CD3 > ...).

    Each cluster is modelled ``count ~ group`` with ``log(library size)`` as
    offset and ITS OWN dispersion (Var = mu + alpha mu^2), estimated so the
    Pearson statistic equals the residual df (``_nb_dispersion``). The group
    effect is tested by likelihood ratio against a t reference on those df
    (``F(1, df)``), and BH-adjusted. Returns rows sorted by p-value::

        {cluster, log2fc, prop_a, prop_b, z, p, p_adj, dispersion, df,
         n_a, n_b, group_a, group_b}

    ``log2fc`` is log2(rate B / rate A) of the fitted NB rates (B over A,
    as every fold change in OpenFlo; ±inf when the population has no events
    in one group, NaN in neither). ``prop_*`` are the mean per-sample
    proportions in each group. ``z`` is the signed square root of the
    likelihood-ratio statistic and ``p = 2 P(T_df > |z|)``. ``p`` is NaN
    when there are no residual df (one sample per group): with no
    replicates there is nothing to measure the spread between samples with.

    Why a t reference: with 3 vs 3 there are 4 df, and a dispersion from 4
    df is noisy. Treating the Wald z as normal gave 12 % false positives at
    the 5 % level in seeded NB simulations even with the dispersion
    estimated correctly; the t (F) reference gives 4-6 % at 3 vs 3, 4 vs 4
    and 6 vs 6, alpha 0.05-0.3, library sizes varying 2x, and for nested
    gates (tests/test_diffabundance_calibration.py)."""
    from scipy.stats import t as t_dist
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
    labels = (group == levels[1]).astype(int)
    if lib_sizes is None:
        lib_sizes = np.nansum(Y, axis=0)
    lib_sizes = np.clip(np.asarray(lib_sizes, dtype=float), 1.0, None)
    # A population a sample does not have is NaN there (MISSING -- never a
    # count of 0, which is a measurement). Each population is fitted on the
    # samples that have it; populations sharing the same set of samples are
    # fitted together, so the fit stays vectorised.
    present = np.isfinite(Y)
    patterns: dict = {}
    for k in range(n_clusters):
        patterns.setdefault(present[k].tobytes(), []).append(k)
    rows = []
    for ks in patterns.values():
        cols = present[ks[0]]
        sub = _nb_abundance_rows(
            Y[np.ix_(ks, np.flatnonzero(cols))], labels[cols], lib_sizes[cols],
            [cluster_names[k] for k in ks], levels, t_dist)
        for r in sub:
            r['n_missing'] = int((~cols).sum())
        rows.extend(sub)
    adj = _benjamini_hochberg([r['p'] for r in rows])
    for r, q in zip(rows, adj, strict=False):
        r['p_adj'] = q
    rows.sort(key=lambda r: (not np.isfinite(r['p']), r['p']))
    return rows


def _nb_abundance_rows(Y, labels, lib_sizes, cluster_names, levels, t_dist):
    """differential_abundance's fit and test for populations present in
    every one of these samples. Rows without p_adj (added by the caller,
    across all populations)."""
    n_clusters, n = Y.shape
    nan = float('nan')
    if not (labels == 0).any() or not (labels == 1).any():
        # Every sample of one group lacks these populations: nothing to
        # compare them with. Said, not tested -- and not called a loss.
        rows = []
        for k in range(n_clusters):
            y = Y[k]
            pa = y[labels == 0] / lib_sizes[labels == 0]
            pb = y[labels == 1] / lib_sizes[labels == 1]
            rows.append({'cluster': cluster_names[k], 'log2fc': nan,
                         'prop_a': float(np.mean(pa)) if pa.size else nan,
                         'prop_b': float(np.mean(pb)) if pb.size else nan,
                         'z': nan, 'p': nan, 'dispersion': nan, 'df': 0,
                         'n_a': int((labels == 0).sum()),
                         'n_b': int((labels == 1).sum()),
                         'group_a': str(levels[0]),
                         'group_b': str(levels[1])})
        return rows
    a_mask, b_mask = labels == 0, labels == 1

    alpha, df = _nb_dispersion(Y, lib_sizes, labels, 2)
    a_fit = np.where(np.isfinite(alpha), alpha, 0.0)
    lr1 = _nb_group_rates(Y, lib_sizes, labels, 2, a_fit)
    lr0 = _nb_group_rates(Y, lib_sizes, np.zeros(n, dtype=int), 1, a_fit)
    mu1 = _nb_mu(lr1, lib_sizes, labels)
    mu0 = _nb_mu(lr0, lib_sizes, np.zeros(n, dtype=int))
    # Likelihood ratio at the population's dispersion. The terms of the NB
    # log-likelihood that do not involve mu cancel between the two fits, and
    # writing r log(r/(r+mu)) as -log1p(alpha mu)/alpha keeps it exact as
    # alpha -> 0 (where it is the Poisson -mu).
    a2 = a_fit[:, None]
    safe1 = np.where(mu1 > 0, mu1, 1.0)
    safe0 = np.where(mu0 > 0, mu0, 1.0)
    ylog = np.where(Y > 0, Y * np.log(safe1 / safe0), 0.0)
    d_log1p = np.log1p(a2 * mu1) - np.log1p(a2 * mu0)
    pos = a2 > 0
    shape = np.where(pos, d_log1p / np.where(pos, a2, 1.0), mu1 - mu0)
    lr_stat = 2.0 * np.sum(ylog - Y * d_log1p - shape, axis=1)
    lr_stat = np.clip(lr_stat, 0.0, None)
    with np.errstate(invalid='ignore'):
        beta = lr1[:, 1] - lr1[:, 0]          # NaN when both groups are empty
    rows = []
    for k in range(n_clusters):
        y = Y[k]
        b = float(beta[k])
        testable = df[k] > 0 and np.isfinite(lr_stat[k]) and not np.isnan(b)
        z = (float(np.sign(b) * np.sqrt(lr_stat[k])) if testable
             else float('nan'))
        p = (float(2.0 * t_dist.sf(abs(z), df[k])) if testable
             else float('nan'))
        prop_a = float(np.mean(y[a_mask] / lib_sizes[a_mask]))
        prop_b = float(np.mean(y[b_mask] / lib_sizes[b_mask]))
        rows.append({'cluster': cluster_names[k],
                     'log2fc': float(b / np.log(2.0)),
                     'prop_a': prop_a, 'prop_b': prop_b,
                     'z': z, 'p': p, 'dispersion': float(alpha[k]),
                     'df': int(df[k]),
                     'n_a': int(a_mask.sum()), 'n_b': int(b_mask.sum()),
                     'group_a': str(levels[0]), 'group_b': str(levels[1])})
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
    ``{channel: [value per sample]}`` for ``differential_test``.

    The statistic is taken on the compensated LINEAR values (the sample's
    recorded display transform undone, pipeline.linear_values): on the
    loader's logicle values a 10x difference measured log2FC 0.51 instead of
    3.32. Negative values are kept (compensated intensities are legitimately
    negative); a channel of unknown display scale gives NaN."""
    from .pipeline import UnknownScaleError, linear_values
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
                    try:
                        vals = linear_values(s, ch, df[ch].values)
                    except UnknownScaleError:
                        g[ch].append(float('nan'))
                        continue
                    vals = vals[np.isfinite(vals)]
                    g[ch].append(float(agg(vals)) if vals.size else float('nan'))
                else:
                    g[ch].append(float('nan'))
        groups.append(g)
    return groups[0], groups[1]
