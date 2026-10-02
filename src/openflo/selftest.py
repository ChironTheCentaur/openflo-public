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

from ._console import force_utf8_streams

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
    """Compensation must REMOVE baked-in spillover and PRESERVE the signal.

    The previous version of this metric wrote the generator's own planted
    matrix to a CSV, read it straight back, and asserted the value it had just
    written. It invoked no compensation code whatsoever, so it stayed green
    through the 2.2.1 transpose bug that corrupted every real compensated
    analysis — a metric named "Compensation APC -> APC-Fire spill" that could
    not observe compensation being broken.

    Now it applies compensation to data with a known spillover baked in and
    reports two numbers that cannot both be satisfied by a degenerate answer:
    the residual error against the true signal (a transposed or wrong-direction
    matmul leaves this large) and the fraction of signal retained (zeroing the
    data would drive this to 0 while flattering the residual).
    """
    import pandas as pd

    from .pipeline import FlowSample
    from .synthetic import DEFAULT_SPILL_LEAKS, PBMC_MARKER_DET, PBMC_MARKERS
    dets = [PBMC_MARKER_DET[m] for m in PBMC_MARKERS]
    k = len(dets)
    spill = np.eye(k)
    for (src, dst), v in DEFAULT_SPILL_LEAKS.items():
        spill[dets.index(src), dets.index(dst)] = v
    rng = np.random.default_rng(8)
    true = rng.uniform(100.0, 6000.0, size=(4000, k))
    observed = true @ spill                    # instrument sees the spillover
    s = FlowSample.from_dataframe(
        pd.DataFrame(observed, columns=pd.Index(dets)), name='selftest')
    s.manual_compensate(spill.copy(), dets)
    out = np.asarray(s.data[dets].to_numpy(), dtype=float)
    # Scale-free so the number means the same thing at any signal level.
    resid = float(np.abs(out - true).max() / np.abs(true).max())
    retained = float(np.median(out) / np.median(true))
    return {'compensation.residual_err': round(resid, 6),
            'compensation.signal_retained': round(retained, 4)}


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
    force_utf8_streams()
    ap = argparse.ArgumentParser(
        prog='openflo-selftest', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', action='store_true',
                    help='Emit the raw computed metrics as JSON and exit.')
    ap.add_argument('--update', action='store_true',
                    help='Rewrite _golden.json from the current run (keeps '
                         'each metric\'s tolerance/label). Use only after an '
                         'intended behavior change.')
    ap.add_argument('--controls', action='store_true',
                    help='Run the RESPONSE controls instead of the golden '
                         'baseline: negative controls, titrations and a null '
                         'comparison that prove each output tracks its input. '
                         'The golden proves the answer did not change; these '
                         'prove the answer is real. No baseline to update.')
    ap.add_argument('--all', action='store_true',
                    help='Run the golden baseline AND the response controls.')
    ap.add_argument('--force', action='store_true',
                    help='With --update: re-pin the baseline even though a '
                         'response control is failing. Only for a change you '
                         'have verified is intended.')
    args = ap.parse_args(argv)

    if args.controls and not args.all:
        from .selftest_controls import main as controls_main
        return controls_main(['--json'] if args.json else [])

    if args.json:
        print(json.dumps(compute_metrics(), indent=2))
        return 0

    if args.update:
        # Re-baselining is the one operation that can turn a bug into the
        # expected answer: --update rewrites every pinned value from the
        # CURRENT run, so a broken pipeline blesses itself and the golden
        # then defends the breakage forever. Gate it on the response
        # controls, which assert relationships rather than values and so
        # cannot be re-baselined — if an output has stopped tracking its
        # input, there is nothing legitimate to re-pin.
        if not args.force:
            from .selftest_controls import format_table, run_controls
            creq = run_controls()
            if not all(r[1] for r in creq):
                print(format_table(creq))
                print("\nREFUSING to update the golden baseline: a response "
                      "control is failing, which means an output is no longer "
                      "tracking its input. Re-pinning now would record the "
                      "broken behaviour as correct.\n"
                      "Fix the control first, or pass --force if you have "
                      "verified this is an intended change.")
                return 1

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

    if args.all:
        from .selftest_controls import format_table, run_controls
        creq = run_controls()
        print(format_table(creq))
        ok = ok and all(r[1] for r in creq)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
