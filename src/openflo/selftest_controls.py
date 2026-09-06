"""Response controls — prove the pipeline REACTS to its input.

The golden baseline (:mod:`openflo.selftest`) answers one question: *did the
answer change?* It cannot answer the other one: *is the answer real?* Every
golden metric runs a single fixed dataset and compares against a pinned number,
so an implementation that ignored its input entirely and returned those seven
constants would pass all seven checks. Frozen wrongness is indistinguishable
from preserved correctness.

This module is the other half, and it is built on one rule:

    **NOTHING HERE IS A PINNED NUMBER.**

Every check is a *relationship* between an input the caller varies and the
output the pipeline produces — a negative control that must come back at zero,
a titration that must track its dose, a planted effect that must be found in
the planted direction, a null comparison that must find nothing. A hardcoded
constant cannot satisfy a negative control and a titration at the same time,
so these checks fail loudly for exactly the class of bug the golden is blind
to. They are the analytical equivalent of the unstained tube and the isotype
control: the golden proves the run reproduced, these prove it measured.

    openflo-selftest --controls          # run, print a PASS/FAIL table, exit 0/1
    openflo-selftest --controls --json   # emit the raw observations as JSON

Because the assertions are relational, this file needs no baseline to update
and cannot drift: a change that breaks one is a real behaviour change, not a
re-baselining chore.
"""
from __future__ import annotations

import tempfile

import numpy as np

# Tolerances below are DELIBERATELY loose. They are correctness floors ("did
# the estimate land near the planted truth", "is the effect in the right
# direction"), not frozen observations — tightening them into precise expected
# values would recreate the very problem this module exists to solve.
_ZERO_SPILL = 0.02      # a negative control must come back below this
_SPILL_ABS = 0.03       # recovered spill must land this close to planted
_FRAC_ABS = 0.015       # recovered event fraction, absolute (1.5 percentage pts)
_ALPHA = 0.05           # significance for the planted-effect controls


# ── compensation: negative control + titration ────────────────────────────────

def _recover_spill(leaks, seed=700, n=8000):
    """Plant `leaks`, write single-stain controls, estimate the matrix back."""
    from .pipeline import optimize_compensation
    from .synthetic import PBMC_MARKER_DET, PBMC_MARKERS, make_compensation_controls
    dets = [PBMC_MARKER_DET[m] for m in PBMC_MARKERS]
    with tempfile.TemporaryDirectory() as tmp:
        paths, _csv = make_compensation_controls(tmp, n=n, seed=seed,
                                                 leaks=leaks)
        by_channel = {d: p for d, p in zip(dets, paths, strict=True)}
        # optimize_compensation returns (channels, matrix) — index the matrix
        # with the channel order IT reports, not the one we passed in.
        chans, est = optimize_compensation(dets, by_channel)
    return list(chans), np.asarray(est, dtype=float)


def _off_diagonal_max(est):
    m = np.asarray(est, dtype=float).copy()
    np.fill_diagonal(m, 0.0)
    return float(np.abs(m).max()) if m.size else 0.0


def _compensation_controls():
    """No spill in → no spill out; more spill in → proportionally more out."""
    out = []

    # NEGATIVE CONTROL — the unstained tube. Nothing planted, nothing found.
    _dets, est = _recover_spill({})
    worst = _off_diagonal_max(est)
    out.append(('compensation.negative_control', worst < _ZERO_SPILL,
                f'{worst:.4f}', f'< {_ZERO_SPILL}',
                'Spill-free controls estimate ~zero spillover'))

    # TITRATION — recovered coefficient must track the planted dose.
    src, dst = 'APC-A', 'APC-Fire-A'
    planted, recovered = [0.05, 0.15, 0.30], []
    dets = None
    for v in planted:
        dets, est = _recover_spill({(src, dst): v})
        recovered.append(float(est[dets.index(src), dets.index(dst)]))
    close = max(abs(r - p) for r, p in zip(recovered, planted, strict=True))
    rising = all(b > a for a, b in zip(recovered, recovered[1:], strict=False))
    got = ' → '.join(f'{r:.3f}' for r in recovered)
    out.append(('compensation.titration_tracks', close < _SPILL_ABS and rising,
                got, ' → '.join(f'{p:.3f}' for p in planted),
                'Recovered spillover follows the planted dose'))
    return out


# ── compensation APPLIER: recovery at every spill level ───────────────────────

def _compensation_apply_controls():
    """Applying a spillover matrix must RECOVER the true signal.

    Distinct from the estimator controls above, and the distinction matters:
    the 2.2.1 bug was in the APPLIER (`observed @ inv(M).T` instead of
    `observed @ inv(M)`), which the estimator checks cannot see. The matrices
    here are deliberately ASYMMETRIC — a transposed matmul is invisible against
    a symmetric one.
    """
    import pandas as pd

    from .pipeline import FlowSample
    chans = ['A', 'B', 'C', 'D']
    rng = np.random.default_rng(4)
    true = rng.uniform(100.0, 6000.0, size=(3000, 4))
    scale = float(np.abs(true).max())

    def residual(spill):
        obs = true @ spill
        s = FlowSample.from_dataframe(
            pd.DataFrame(obs, columns=pd.Index(chans)), name='ctl')
        s.manual_compensate(spill.copy(), chans)
        out = np.asarray(s.data[chans].to_numpy(), dtype=float)
        return float(np.abs(out - true).max() / scale)

    # NEGATIVE CONTROL — no spillover to remove, so nothing may change.
    ident = residual(np.eye(4))

    # TITRATION — recovery must hold at every level, not just the one baked
    # into the fixtures.
    errs = []
    for lvl in (0.05, 0.15, 0.30):
        m = np.eye(4)
        m[0, 1] = lvl              # A -> B
        m[2, 3] = lvl * 0.6        # C -> D
        m[1, 2] = lvl * 0.3        # B -> C, makes the matrix asymmetric
        errs.append(residual(m))
    worst = max(errs)
    return [
        ('compensation.identity_is_a_noop', ident < 1e-9,
         f'{ident:.2e}', '< 1e-9',
         'Identity compensation leaves the data untouched'),
        ('compensation.recovers_true_signal', worst < 1e-9,
         ' / '.join(f'{e:.1e}' for e in errs), '< 1e-9 at every level',
         'Applying compensation recovers the true signal'),
    ]


# ── auto-clean: titration of a planted artefact fraction ──────────────────────

def _doublet_controls():
    """Plant a known doublet fraction; the cleaner must remove about that much."""
    from .pipeline import autoclean_keep_mask, default_autoclean_methods
    from .synthetic import PBMC_CHANNELS

    def sample_with_doublets(frac, n=40000, seed=11):
        """Singlets at FSC-H≈FSC-A/2, doublets at FSC-H≈FSC-A/3.1 (as the
        shipped generator builds them), mixed at a known ratio."""
        import pandas as pd
        rng = np.random.default_rng(seed)
        n_dbl = int(n * frac)
        n_sgl = n - n_dbl
        s_fsc = np.clip(rng.normal(60000, 8000, n_sgl), 1.0, None)
        d_fsc = np.clip(rng.normal(118000, 12000, n_dbl), 1.0, None)
        fsc = np.concatenate([s_fsc, d_fsc])
        fsh = np.concatenate([
            s_fsc / 2.0 * rng.normal(1.0, 0.02, n_sgl),
            d_fsc / 3.1 * rng.normal(1.0, 0.04, n_dbl)])
        df = pd.DataFrame({c: np.zeros(n) for c in PBMC_CHANNELS})
        df['FSC-A'], df['FSC-H'] = fsc, fsh
        df['SSC-A'] = np.clip(rng.normal(50000, 9000, n), 1.0, None)
        return df

    method = next(m for m in default_autoclean_methods()
                  if m['key'] == 'doublets')
    planted, removed = [0.02, 0.08, 0.16], []
    for frac in planted:
        df = sample_with_doublets(frac)
        gate = {'kind': 'autoclean',
                'methods': [{**method, 'enabled': True}]}
        keep = np.asarray(autoclean_keep_mask(gate, df), dtype=bool)
        removed.append(float((~keep).mean()))
    close = max(abs(r - p) for r, p in zip(removed, planted, strict=True))
    rising = all(b > a for a, b in zip(removed, removed[1:], strict=False))
    got = ' → '.join(f'{r * 100:.1f}%' for r in removed)
    exp = ' → '.join(f'{p * 100:.1f}%' for p in planted)
    return [('autoclean.doublet_titration', close < _FRAC_ABS and rising,
             got, exp, 'Doublets removed tracks the planted fraction')]


# ── differential abundance: planted effect + null comparison ──────────────────

def _population_counts(df):
    """Count the six true PBMC populations by thresholding their bright marker.

    The generator draws positives at _HI and negatives at _LO, orders of
    magnitude apart, so a midpoint cut recovers the planted composition without
    needing the discarded per-event labels.
    """
    from .synthetic import PBMC_MARKER_DET
    cut = 1200.0                     # between _LO (150) and _HI (5000)
    cd3 = df[PBMC_MARKER_DET['CD3']].to_numpy(float) > cut
    cd4 = df[PBMC_MARKER_DET['CD4']].to_numpy(float) > cut
    cd8 = df[PBMC_MARKER_DET['CD8']].to_numpy(float) > cut
    cd19 = df[PBMC_MARKER_DET['CD19']].to_numpy(float) > cut
    cd56 = df[PBMC_MARKER_DET['CD56']].to_numpy(float) > cut
    cd14 = df[PBMC_MARKER_DET['CD14']].to_numpy(float) > cut
    return {'CD4 T': int((cd3 & cd4).sum()), 'CD8 T': int((cd3 & cd8).sum()),
            'B cell': int(cd19.sum()), 'NK cell': int(cd56.sum()),
            'Monocyte': int(cd14.sum())}


def _abundance_table(arms, donors=4, n=20000, seed0=0):
    """counts (populations x samples) + per-sample arm labels.

    ``arms`` is ``[(arm_label, generator_group), ...]``. The two are separate on
    purpose: the null control needs two DISTINCT arm labels (the GLM requires
    exactly two levels) drawn from the SAME generator composition, so that any
    difference it reports is manufactured rather than planted.
    """
    import pandas as pd

    from .synthetic import immunophenotyping_sample
    cols, labels = {}, []
    for ai, (arm, group) in enumerate(arms):
        for d in range(donors):
            df = immunophenotyping_sample(n=n, seed=seed0 + ai * 100 + d,
                                          group=group)
            cols[f'{arm}{d}'] = _population_counts(df)
            labels.append(arm)
    return pd.DataFrame(cols), labels


def _abundance_controls():
    """ctrl vs treat must FIND the planted shift; ctrl vs ctrl must find none."""
    from .diffexp import differential_abundance
    out = []

    # POSITIVE CONTROL — the generator plants NK +6pp, CD8 +4pp, CD4 -10pp.
    # `differential_abundance` returns a list of dicts and reports log2fc as
    # b-relative-to-a, with group_a/group_b naming the arms, so the expected
    # sign depends on which arm landed in b.
    counts, labels = _abundance_table(
        [('ctrl', 'ctrl'), ('treat', 'treat')])
    res = differential_abundance(counts, labels,
                                 cluster_names=list(counts.index))
    by_pop = {r['cluster']: r for r in res}
    treat_is_b = bool(res) and res[0].get('group_b') == 'treat'
    direction, all_ok = {}, True
    for pop, up_in_treat in (('NK cell', True), ('CD8 T', True),
                             ('CD4 T', False)):
        r = by_pop.get(pop)
        if r is None:                                    # pragma: no cover
            direction[pop], all_ok = 'missing', False
            continue
        lfc, p = float(r['log2fc']), float(r['p_adj'])
        want_positive = up_in_treat if treat_is_b else not up_in_treat
        ok = (p < _ALPHA) and ((lfc > 0) == want_positive)
        all_ok = all_ok and ok
        direction[pop] = f'log2fc={lfc:+.2f} q={p:.1e}'
    got = '; '.join(f'{k} {v}' for k, v in direction.items())
    out.append(('diffabundance.planted_effect', all_ok, got,
                'NK up, CD8 up, CD4 down (q<0.05)',
                'The planted composition shift is found, correctly signed'))

    # NULL CONTROL — two ctrl arms differing only by donor seed. A
    # "significant" hit here would mean the test manufactures effects.
    counts0, labels0 = _abundance_table(
        [('armA', 'ctrl'), ('armB', 'ctrl')], seed0=500)
    res0 = differential_abundance(counts0, labels0,
                                  cluster_names=list(counts0.index))
    sig = [r['cluster'] for r in res0 if float(r['p_adj']) < _ALPHA]
    out.append(('diffabundance.null_control', not sig,
                f'{len(sig)} significant'
                + (f' ({", ".join(sig)})' if sig else ''),
                '0 significant',
                'ctrl vs ctrl finds no false differences'))
    return out


# ── clustering: found structure vs a permutation control ──────────────────────

def _leiden_labels(df, fluor, resolution=0.5):
    """Run Leiden on a duck-typed sample (as the golden does) and return the
    per-event cluster id."""
    import types
    from typing import cast

    from .pipeline import FlowSample
    s = types.SimpleNamespace(data=df.copy(), fluor_channels=list(fluor))
    FlowSample.run_leiden(cast('FlowSample', s), resolution=resolution)
    return s.data['leiden'].to_numpy()


def _clustering_controls():
    """Clusters must be PURE with respect to the true populations — and that
    purity must vanish when the population structure is destroyed.

    The golden pins a cluster COUNT (18, against 6 true populations), which
    says nothing about whether the clusters correspond to anything real. These
    two checks bracket it: a high score on real data is meaningless without the
    permutation control showing the score can fall.
    """
    from sklearn.metrics import completeness_score, homogeneity_score

    from .synthetic import PBMC_MARKER_DET, PBMC_MARKERS, immunophenotyping_sample
    fluor = [PBMC_MARKER_DET[m] for m in PBMC_MARKERS]
    df, truth = immunophenotyping_sample(n=6000, seed=5, return_labels=True)

    # Score on the live singlets — the population structure the clustering is
    # meant to recover. Artefacts are a cleaning problem, not a clustering one.
    live = ~truth.isin(['dead', 'debris', 'doublet'])
    real = df.loc[live.to_numpy()].reset_index(drop=True)
    real_truth = np.asarray(truth[live])

    hom = float(homogeneity_score(real_truth, _leiden_labels(real, fluor)))
    comp = float(completeness_score(real_truth, _leiden_labels(real, fluor)))

    # PERMUTATION CONTROL — shuffle each marker column independently. Every
    # channel keeps its exact distribution; only the joint structure (which
    # cell carries which combination) is destroyed. Real structure-finding must
    # collapse here; an implementation that "finds" clusters regardless will
    # not.
    rng = np.random.default_rng(1234)
    scrambled = real.copy()
    for c in fluor:
        scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    hom_null = float(homogeneity_score(real_truth,
                                       _leiden_labels(scrambled, fluor)))

    return [
        ('cluster.recovers_populations', hom > 0.80 and hom > 3 * hom_null,
         f'homogeneity={hom:.2f} (completeness={comp:.2f})',
         '> 0.80 and >> permuted',
         'Clusters are pure w.r.t. the true populations'),
        ('cluster.permutation_control', hom_null < 0.25,
         f'homogeneity={hom_null:.2f}', '< 0.25',
         'Purity collapses when the structure is scrambled'),
    ]


# ── driver ────────────────────────────────────────────────────────────────────

def run_controls():
    """Run every response control. Returns a list of result tuples
    ``(key, passed, observed, expected, label)``."""
    results = []
    for fn in (_compensation_controls, _compensation_apply_controls,
               _doublet_controls, _abundance_controls,
               _clustering_controls):
        try:
            results.extend(fn())
        except Exception as exc:                          # noqa: BLE001
            results.append((f'{fn.__name__.strip("_")}.ERROR', False,
                            f'{type(exc).__name__}: {exc}', 'no exception',
                            'Control raised instead of running'))
    return results


def format_table(results):
    lines = ['', 'OpenFlo response controls — the pipeline must REACT to input',
             '(relational checks only; no pinned values, nothing to re-baseline)',
             '']
    width = max((len(r[4]) for r in results), default=10)
    for _key, ok, got, exp, label in results:
        mark = '✓' if ok else '✗'
        lines.append(f'  {mark} {label:<{width}}  {got}   (expect {exp})')
    n_ok = sum(1 for r in results if r[1])
    lines.append('')
    lines.append(f'  {n_ok}/{len(results)} passed'
                 + (' — the pipeline responds to its input.' if n_ok == len(results)
                    else ' — a control FAILED: an output is not tracking its input.'))
    return '\n'.join(lines)


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(
        prog='openflo-selftest --controls',
        description='Response controls: prove outputs track their inputs.')
    ap.add_argument('--json', action='store_true', help='emit raw JSON')
    args = ap.parse_args(argv)
    results = run_controls()
    if args.json:
        print(json.dumps([{'key': k, 'passed': ok, 'observed': got,
                           'expected': exp, 'label': lab}
                          for k, ok, got, exp, lab in results], indent=2))
    else:
        print(format_table(results))
    return 0 if all(r[1] for r in results) else 1


if __name__ == '__main__':                                # pragma: no cover
    raise SystemExit(main())
