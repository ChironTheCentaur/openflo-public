"""End-user self-test — confirm a build reproduces OpenFlo's reference behavior.

Runs the **seeded** synthetic dataset (:mod:`openflo.synthetic`) through the
core feature paths — auto-clean (debris / viability / doublets), Leiden
clustering, MESF calibration, compensation — and compares each metric against
the committed golden baseline (``_golden.json``). Because the generators are
seeded, a green run means your install/code reproduces the baseline; a red
metric pinpoints exactly which feature's behavior changed.

    openflo-selftest            # run, print a PASS/FAIL table, exit 0/1
    openflo-selftest --json     # emit the raw metrics as JSON
    openflo-selftest --update   # rewrite _golden.json from the current run

This is the end-user counterpart to the pytest suite: no test framework, no
real data, one command. The same metrics back the ``test_selftest`` regression
test, so the golden file is the single source of truth for both.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import cast

import numpy as np

_GOLDEN = os.path.join(os.path.dirname(__file__), '_golden.json')


# ── metric computations (each pure + deterministic from a fixed seed) ───────────

def _autoclean_metrics():
    from .pipeline import (
        autoclean_keep_mask,
        default_autoclean_methods,
        find_viability_channel,
        transform_values,
    )
    from .synthetic import (
        PBMC_LABELS,
        immunophenotyping_sample,
        size_bead_sample,
    )
    bead_fsc = float(np.median(size_bead_sample(seed=950)['FSC-A']))
    raw = immunophenotyping_sample(n=20000, seed=42, group='ctrl')
    via = find_viability_channel(list(raw.columns), PBMC_LABELS)
    d = raw.copy()
    d[via] = transform_values(raw[via].to_numpy(float), method='logicle')
    methods = default_autoclean_methods()
    for m in methods:
        if m['key'] == 'debris':
            m['params']['bead_fsc'] = bead_fsc
        if m['key'] == 'viability':
            m['params']['channel'] = via

    def pct(key):
        m = next(x for x in methods if x['key'] == key)
        solo = {'kind': 'autoclean', 'methods': [{**m, 'enabled': True}]}
        return round(100.0 * (~np.asarray(autoclean_keep_mask(solo, d),
                                          bool)).mean(), 3)
    return {'autoclean.debris_pct': pct('debris'),
            'autoclean.viability_pct': pct('viability'),
            'autoclean.doublets_pct': pct('doublets')}


def _cluster_metric():
    import types

    from .pipeline import FlowSample
    from .synthetic import immunophenotyping_sample
    df = immunophenotyping_sample(n=6000, seed=5)
    s = types.SimpleNamespace(
        data=df.copy(),
        fluor_channels=['BV510-A', 'FITC-A', 'APC-A', 'PE-A', 'BV605-A',
                        'APC-Fire-A'])
    # run_leiden only touches .data + .fluor_channels; a duck-typed stub keeps
    # the cluster channels fixed (so the golden count is reproducible).
    FlowSample.run_leiden(cast(FlowSample, s), resolution=0.5)
    return {'cluster.leiden_n': int(s.data['leiden'].nunique())}


def _calibration_metric():
    import tempfile

    from .calibration import detect_bead_peaks, fit_mesf_calibration
    from .pipeline import FlowSample
    from .synthetic import CAL_CHANNEL, CAL_PEAK_MESF, make_calibration_beads
    with tempfile.TemporaryDirectory() as td:
        fcs, _ = make_calibration_beads(td, seed=900)
        vals = FlowSample(fcs).data[CAL_CHANNEL].to_numpy(float)
        peaks = detect_bead_peaks(vals, n_peaks=len(CAL_PEAK_MESF))
        fit = fit_mesf_calibration(peaks, CAL_PEAK_MESF)
    return {'calibration.slope': round(fit['slope'], 4),
            'calibration.r2': round(fit['r2'], 4)}


def _compensation_metric():
    import tempfile

    from .pipeline import read_compensation_matrix
    from .synthetic import make_compensation_controls
    with tempfile.TemporaryDirectory() as td:
        _, csv = make_compensation_controls(td, n=4000, seed=8)
        chans, mat = read_compensation_matrix(csv)
    assert chans is not None and mat is not None      # synthetic CSV always parses
    i, j = chans.index('APC-A'), chans.index('APC-Fire-A')
    return {'compensation.apc_leak': round(float(mat[i, j]), 4)}


# Order = print order. Each returns a {metric_key: value} dict.
_METRIC_GROUPS = (_autoclean_metrics, _cluster_metric, _calibration_metric,
                  _compensation_metric)


def compute_metrics():
    """Run every feature path and return ``{metric_key: value}``. A group that
    raises (e.g. an optional clustering backend is missing) is reported with
    each of its metrics as ``None`` rather than aborting the whole run."""
    out: dict[str, float | int | str | None] = {}
    for group in _METRIC_GROUPS:
        try:
            out.update(group())
        except Exception as exc:                       # noqa: BLE001
            # Tag the group's expected keys (from golden) as failed/None.
            for k in load_golden():
                pre = group.__name__.split('_')[1]      # 'autoclean'/'cluster'/…
                if k.startswith(pre) and k not in out:
                    out[k] = None
            out[f'_error.{group.__name__}'] = str(exc)
    return out


def load_golden():
    with open(_GOLDEN, encoding='utf-8') as f:
        return json.load(f)['metrics']


# ── compare + report ────────────────────────────────────────────────────────────

def run_selftest():
    """Compute metrics, compare to the golden baseline. Returns
    ``(results, ok)`` where ``results`` is a list of per-metric dicts and
    ``ok`` is True iff every metric is within tolerance."""
    golden = load_golden()
    metrics = compute_metrics()
    results = []
    ok = True
    for key, spec in golden.items():
        got = metrics.get(key)
        exp, tol = spec['value'], spec['tol']
        passed = got is not None and abs(float(got) - float(exp)) <= float(tol)
        ok = ok and passed
        results.append({'key': key, 'label': spec.get('label', key),
                        'unit': spec.get('unit', ''), 'expected': exp,
                        'tol': tol, 'got': got, 'passed': passed})
    return results, ok


def _format_table(results):
    lines = []
    for r in results:
        mark = '✓' if r['passed'] else '✗'
        # Only show a unit symbol for '%'; 'count'/'' render as bare numbers.
        unit = r['unit'] if r['unit'] == '%' else ''
        got = '—' if r['got'] is None else f"{r['got']:g}{unit}"
        exp = f"{r['expected']:g}{unit}"
        lines.append(
            f"  {mark} {r['label']:<44} {got:>9}   "
            f"(exp {exp} ±{r['tol']:g})")
    n_ok = sum(r['passed'] for r in results)
    lines.append('')
    lines.append(f"  {n_ok}/{len(results)} passed — "
                 + ('behavior matches baseline.'
                    if n_ok == len(results)
                    else 'BEHAVIOR CHANGED (see ✗ rows above).'))
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='openflo-selftest', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', action='store_true',
                    help='Emit the raw computed metrics as JSON and exit.')
    ap.add_argument('--update', action='store_true',
                    help='Rewrite _golden.json from the current run (keeps '
                         'each metric\'s tolerance/label). Use only after an '
                         'intended behavior change.')
    args = ap.parse_args(argv)

    if args.json:
        print(json.dumps(compute_metrics(), indent=2))
        return 0

    if args.update:
        golden = load_golden()
        metrics = compute_metrics()
        for k, spec in golden.items():
            if metrics.get(k) is not None:
                spec['value'] = metrics[k]
        with open(_GOLDEN, encoding='utf-8') as f:
            full = json.load(f)
        full['metrics'] = golden
        with open(_GOLDEN, 'w', encoding='utf-8') as f:
            json.dump(full, f, indent=2)
            f.write('\n')
        print(f"Updated golden baseline → {_GOLDEN}")
        return 0

    print("OpenFlo self-test — seeded synthetic data vs golden baseline\n")
    results, ok = run_selftest()
    print(_format_table(results))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
