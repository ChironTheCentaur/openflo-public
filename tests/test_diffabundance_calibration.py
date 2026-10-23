"""differential_abundance's p-values must mean what they say.

The NB GLM behind the Frequencies window's "Differential abundance" was
badly anti-conservative, for three reasons that compounded:

  * its IRLS solved the POISSON score (no 1/(1 + alpha mu) weight on the
    residual), so with library sizes differing inside a group the "NB" group
    effect was the Poisson one: 0.643 where the NB MLE is 0.170;
  * one method-of-moments dispersion for all populations, with no df
    correction (true alpha 0.1 estimated 0.044 at 3 vs 3);
  * that shared dispersion, pulled to 0.0008 by big nested gates (Cells,
    Singlets, Live), was then applied to rare subsets with CV 30-40 %, and
    the Wald z was read against a normal on 4 df.

Measured before the fix at nominal 5 %: 24 % false positives for pure NB
3 vs 3, and 64 % / 84 % / 83 % for CD3 / Treg / a rare subset under nested
gating. Each check below is against independent truth: scipy's NB likelihood
for the fit, seeded simulation with no planted effect for the error rate.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import nbinom

from openflo.diffexp import (
    _nb_group_rates,
    _nb_mu,
    differential_abundance,
)


def _nb_counts(rng, mu, alpha):
    r = 1.0 / alpha
    return rng.negative_binomial(r, r / (r + mu)).astype(float)


def test_group_effect_is_the_negative_binomial_mle():
    """Unequal library sizes inside each group: the NB MLE of the group
    effect differs from the Poisson one (ratio of pooled sums). The old IRLS
    returned the Poisson 0.643 at every alpha; scipy's optimum of the NB
    likelihood is 0.170 at alpha 0.3."""
    y = np.array([50., 400., 60., 30., 900., 40.])
    lib = np.array([1e4, 1e5, 2e4, 1e4, 1e5, 2e4])
    lab = np.array([0, 0, 0, 1, 1, 1])
    for alpha in (0.05, 0.3, 2.0):
        def nll(b, alpha=alpha):
            r = 1.0 / alpha
            mu = lib * np.exp(b[0] + b[1] * lab)
            return -nbinom.logpmf(y, r, r / (r + mu)).sum()
        ref = minimize(nll, [np.log(y.sum() / lib.sum()), 0.0],
                       method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12,
                                'maxiter': 20000}).x
        lr = _nb_group_rates(y[None], lib, lab, 2, np.array([alpha]))
        assert lr[0, 1] - lr[0, 0] == pytest.approx(ref[1], abs=1e-5)
    poisson = np.log(y[3:].sum() / lib[3:].sum() / (y[:3].sum() / lib[:3].sum()))
    assert poisson == pytest.approx(0.643, abs=1e-3)
    assert ref[1] != pytest.approx(poisson, abs=0.2)


def test_dispersion_makes_pearson_equal_the_residual_df_and_z_is_the_lr():
    """The reported dispersion is the df-corrected moment estimate (Pearson
    = n - 2), and z^2 is the likelihood ratio computed straight from scipy's
    NB pmf at that dispersion."""
    y = np.array([50., 400., 60., 30., 900., 40.])
    lib = np.array([1e4, 1e5, 2e4, 1e4, 1e5, 2e4])
    lab = np.array([0, 0, 0, 1, 1, 1])
    row = differential_abundance(y[None], ['A'] * 3 + ['B'] * 3,
                                 lib_sizes=lib)[0]
    alpha = row['dispersion']
    assert row['df'] == 4
    lr1 = _nb_group_rates(y[None], lib, lab, 2, np.array([alpha]))
    mu1 = _nb_mu(lr1, lib, lab)[0]
    assert np.sum((y - mu1) ** 2 / (mu1 * (1 + alpha * mu1))) == \
        pytest.approx(4.0, rel=1e-6)
    zero = np.zeros(6, dtype=int)
    mu0 = _nb_mu(_nb_group_rates(y[None], lib, zero, 1, np.array([alpha])),
                 lib, zero)[0]
    r = 1.0 / alpha
    lr_stat = 2 * (nbinom.logpmf(y, r, r / (r + mu1)).sum()
                   - nbinom.logpmf(y, r, r / (r + mu0)).sum())
    assert row['z'] ** 2 == pytest.approx(lr_stat, rel=1e-6)
    assert row['log2fc'] == pytest.approx((lr1[0, 1] - lr1[0, 0]) / np.log(2))


def _null_rate(n_per, alpha, reps, rng, n_clusters=20):
    """Share of p < 0.05 with NO group effect: NB counts, library sizes
    spread 2x (50k-100k), clusters partitioning the cells."""
    hits = tests = 0
    g = np.r_[np.zeros(n_per), np.ones(n_per)]
    for _ in range(reps):
        lib = rng.uniform(50_000, 100_000, 2 * n_per)
        props = rng.dirichlet(np.full(n_clusters, 2.0))
        Y = _nb_counts(rng, props[:, None] * lib[None, :], alpha)
        ps = np.array([r['p'] for r in differential_abundance(Y, g)])
        hits += int(np.sum(ps < 0.05))
        tests += ps.size
    return hits / tests


@pytest.mark.parametrize('n_per', [3, 4, 6])
def test_null_false_positive_rate_is_nominal(n_per):
    """Measured with this seed: 3v3 0.051/0.054/0.051, 4v4 0.054/0.054/0.059,
    6v6 0.048/0.052/0.048 at alpha 0.05/0.1/0.3 (300 runs each); 0.24 at
    3 vs 3 before the fix."""
    rng = np.random.default_rng(100 + n_per)
    for alpha in (0.05, 0.1, 0.3):
        rate = _null_rate(n_per, alpha, reps=50, rng=rng)
        assert rate <= 0.07, (n_per, alpha, rate)


def _nested_counts(rng, n_per):
    """Gate counts as the Frequencies window passes them: each gate a
    binomial draw from its parent with a per-donor logit-normal proportion
    (biological spread small for scatter gates, large for rare subsets);
    the library size is the sample's total events."""
    chain = [(0.90, 0.2), (0.85, 0.2), (0.80, 0.3), (0.45, 0.4),
             (0.10, 0.4), (0.15, 0.5)]   # Cells, Singlets, Live, CD3, Treg, rare
    lib = rng.integers(50_000, 150_000, 2 * n_per).astype(float)
    Y = np.zeros((len(chain), 2 * n_per))
    for j in range(2 * n_per):
        parent = lib[j]
        for k, (p, sd) in enumerate(chain):
            lp = np.log(p / (1 - p)) + rng.normal(0.0, sd)
            Y[k, j] = parent = rng.binomial(int(parent), 1 / (1 + np.exp(-lp)))
    return Y, lib


def test_nested_gates_are_tested_on_their_own_dispersion():
    """Before: one dispersion for all six gates (0.0008, set by the scatter
    gates) made 64-84 % of null CD3 / Treg / rare comparisons significant.
    Measured with this seed over 1000 runs: 0.048 overall, worst gate 0.056
    at 3 vs 3."""
    rng = np.random.default_rng(7)
    g = np.r_[np.zeros(3), np.ones(3)]
    reps = 250
    hits = np.zeros(6)
    for _ in range(reps):
        Y, lib = _nested_counts(rng, 3)
        for r in differential_abundance(Y, g, lib_sizes=lib):
            hits[r['cluster']] += r['p'] < 0.05
    rates = hits / reps
    assert rates.max() <= 0.09, rates          # 250 runs: SE 0.014 per gate
    assert rates.mean() <= 0.065, rates


def test_a_two_fold_change_is_found_at_four_versus_four():
    """Power at an exact 2x change (alpha 0.1, the biological CV of ~32 %):
    measured 0.73 at 4 vs 4 and 0.92 at 6 vs 6, the same as a t-test on log
    proportions (0.73, 0.92); an oracle that knew alpha reached 0.88 / 0.97.
    The changed clusters must also come out with the right sign."""
    rng = np.random.default_rng(11)
    for n_per, floor in ((4, 0.6), (6, 0.85)):
        found = signed = total = 0
        g = np.r_[np.zeros(n_per), np.ones(n_per)]
        for _ in range(40):
            lib = rng.uniform(50_000, 100_000, 2 * n_per)
            props = rng.dirichlet(np.full(20, 2.0))
            fc = np.ones((20, 2 * n_per))
            fc[:5, g == 1] = 2.0
            Y = _nb_counts(rng, props[:, None] * lib[None, :] * fc, 0.1)
            rows = {r['cluster']: r
                    for r in differential_abundance(Y, g, lib_sizes=lib)}
            for k in range(5):
                total += 1
                found += rows[k]['p'] < 0.05
                signed += rows[k]['p'] < 0.05 and rows[k]['log2fc'] > 0
        assert found / total >= floor, (n_per, found / total)
        assert signed == found


def test_one_sample_per_group_gives_no_p_value():
    """With one sample per group there is no between-sample spread to
    measure, so no p-value; the fold change is still reported."""
    rows = differential_abundance(np.array([[100, 60], [50, 90]]), ['a', 'b'])
    assert all(np.isnan(r['p']) and np.isnan(r['p_adj']) and r['df'] == 0
               for r in rows)
    by = {r['cluster']: r for r in rows}
    assert by[0]['log2fc'] < 0 < by[1]['log2fc']


def test_a_population_absent_from_one_group():
    """0 events in every A sample: the fold change is +inf (not a finite
    number set by an iteration cap), and the likelihood ratio still gives a
    p-value from the B samples' spread (df = n_B - 1)."""
    Y = np.array([[0, 0, 0, 5, 8, 6], [100, 120, 90, 100, 110, 95]], float)
    rows = {r['cluster']: r for r in differential_abundance(
        Y, ['A'] * 3 + ['B'] * 3, lib_sizes=np.full(6, 1000.0))}
    assert rows[0]['log2fc'] == float('inf')
    assert rows[0]['df'] == 2
    assert 0 < rows[0]['p'] < 0.05
    empty = differential_abundance(np.zeros((1, 6)), ['A'] * 3 + ['B'] * 3,
                                   lib_sizes=np.full(6, 1000.0))[0]
    assert np.isnan(empty['log2fc']) and np.isnan(empty['p'])
