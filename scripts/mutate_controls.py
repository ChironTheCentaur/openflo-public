#!/usr/bin/env python
"""Attack the response controls: break the pipeline on purpose, and check that
something notices.

`openflo.selftest_controls` exists because the golden baseline cannot tell
preserved correctness from frozen wrongness. But the controls themselves are
code, and a control that cannot fail is exactly the theatre they were written
to replace. This script closes that loop: for each subsystem it substitutes a
deliberately broken implementation and reports whether any control failed.

It earned its keep immediately. A containment check written for the gating
controls drew the child gate wholly INSIDE its parent — so the child was
already a subset, and a chain that never intersected with the parent at all
still passed. The mutation caught it; reading the code had not.

Two rules this script follows, both learned the hard way:

  * A fake must match the real function's output shape before it is trusted.
    An early attempt returned a dict with a 'population' key where the real
    code emits 'cluster'. Every control "failed" — on a KeyError from the
    fake, proving nothing at all. Shapes are asserted against a real call
    before any attack runs.

  * An attack that is NOT caught is the interesting result, so it is reported
    loudly rather than counted as a pass.

Not part of the test suite: each attack re-runs a full control group, so the
whole sweep takes minutes. Run it after changing a control, after adding one,
or when you want to know the safety net is still attached.

Usage:
    python scripts/mutate_controls.py            # every attack
    python scripts/mutate_controls.py gating qc  # only matching attacks
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402

import openflo.annotate as annotate  # noqa: E402
import openflo.diffexp as diffexp  # noqa: E402
import openflo.pipeline as pipeline  # noqa: E402
import openflo.selftest_controls as controls  # noqa: E402

GROUPS = {
    'compensation': controls._compensation_controls,
    'compensation_apply': controls._compensation_apply_controls,
    'doublets': controls._doublet_controls,
    'qc': controls._acquisition_qc_controls,
    'gating': controls._gating_controls,
    'mem': controls._mem_controls,
    'abundance': controls._abundance_controls,
    'clustering': controls._clustering_controls,
}


def _failures(group):
    """Keys of the controls in `group` that did NOT pass."""
    try:
        return [key for key, ok, *_ in GROUPS[group]() if not ok]
    except Exception as exc:                                  # noqa: BLE001
        # A raise is a legitimate way for a control to catch a break, but it
        # is also how a malformed fake looks — so it is labelled, not hidden.
        return [f'RAISED {type(exc).__name__}: {exc}']


class Attack:
    """One deliberate break, applied and then undone."""

    def __init__(self, name, group, apply, undo):
        self.name, self.group, self.apply, self.undo = name, group, apply, undo

    def run(self):
        self.apply()
        try:
            return _failures(self.group)
        finally:
            self.undo()


def _patch(obj, attr, value):
    """(apply, undo) pair that swaps one attribute."""
    original = getattr(obj, attr)
    return (lambda: setattr(obj, attr, value),
            lambda: setattr(obj, attr, original))


def _abundance_fake(log2fc, p_adj):
    """A differential-abundance result with a FIXED verdict for every cluster.

    Built from a real result row so the keys cannot drift out of sync with
    `differential_abundance` — see the module docstring.
    """
    counts, labels = controls._abundance_table(
        [('ctrl', 'ctrl'), ('treat', 'treat')])
    template = diffexp.differential_abundance(
        counts, labels, cluster_names=list(counts.index))[0]

    def fake(counts, group, lib_sizes=None, cluster_names=None):
        names = (list(cluster_names) if cluster_names is not None
                 else list(counts.index))
        group_a, group_b = sorted(set(group))
        rows = []
        for name in names:
            row = dict(template)
            row.update({'cluster': name, 'log2fc': log2fc, 'p': p_adj,
                        'p_adj': p_adj, 'group_a': group_a,
                        'group_b': group_b})
            rows.append(row)
        return rows

    probe = fake(counts, labels, cluster_names=list(counts.index))[0]
    if set(probe) != set(template):
        raise AssertionError(
            'the abundance fake no longer matches the real output shape; '
            'attacks using it would prove nothing. '
            f'missing={set(template) - set(probe)} '
            f'extra={set(probe) - set(template)}')
    return fake


def _mem_ignores_the_data(real):
    """Every population scores the same on every marker."""
    def fake(data, labels, markers, **kw):
        table = real(data, labels, markers, **kw)
        table.loc[:, :] = 10.0
        return table
    return fake


def _mem_drops_the_sign(real):
    """Scores keep their magnitude but lose which way the marker went."""
    def fake(data, labels, markers, **kw):
        return real(data, labels, markers, **kw).abs()
    return fake


def _keep_everything(gates, df, **_kw):
    return np.ones(len(df), dtype=bool)


def _qc_noop(self, *_a, **_kw):
    return self


def _qc_fixed_tenth(self, *_a, **_kw):
    self.data = self.data.iloc[:int(len(self.data) * 0.90)]
    self.data = self.data.reset_index(drop=True)
    return self


def _one_cluster(self, *_a, **_kw):
    self.data['leiden'] = 0
    return self


def _random_clusters(self, *_a, **_kw):
    rng = np.random.default_rng(0)
    self.data['leiden'] = rng.integers(0, 6, len(self.data))
    return self


def _identity_spillover(*args, **_kw):
    channels = list(args[1]) if len(args) > 1 else []
    return channels, np.eye(max(len(channels), 1))


def _gate_selects_everything(_gates, _gid, df, **_kw):
    return np.ones(len(df), dtype=bool)


def _gate_ignores_parent(gates, gid, df, **_kw):
    """Evaluate the gate's OWN region without intersecting its ancestors."""
    rooted = {gid: {**gates[gid], 'parent_id': None}}
    return np.asarray(controls_real_mask(rooted, gid, df), dtype=bool)


def _gate_off_by_one(gates, gid, df, **_kw):
    mask = np.asarray(
        controls_real_mask(gates, gid, df.reset_index(drop=True)), dtype=bool)
    return np.roll(mask, 1) if mask.size else mask


controls_real_mask = pipeline.cumulative_gate_mask


def build_attacks():
    FlowSample = pipeline.FlowSample
    return [
        Attack('compensation is never applied', 'compensation_apply',
               *_patch(FlowSample, 'manual_compensate',
                       lambda self, *a, **k: self)),
        Attack('spillover is estimated as the identity', 'compensation',
               *_patch(pipeline, 'optimize_compensation',
                       _identity_spillover)),
        Attack('doublet cleaning is a no-op', 'doublets',
               *_patch(pipeline, 'autoclean_keep_mask', _keep_everything)),
        Attack('QC never removes an event', 'qc',
               *_patch(FlowSample, 'run_qc', _qc_noop)),
        Attack('QC always removes exactly 10%', 'qc',
               *_patch(FlowSample, 'run_qc', _qc_fixed_tenth)),
        Attack('a gate ignores its own vertices', 'gating',
               *_patch(pipeline, 'cumulative_gate_mask',
                       _gate_selects_everything)),
        Attack('a child gate is not intersected with its parent', 'gating',
               *_patch(pipeline, 'cumulative_gate_mask',
                       _gate_ignores_parent)),
        Attack('gated events are misaligned by one row', 'gating',
               *_patch(pipeline, 'cumulative_gate_mask', _gate_off_by_one)),
        Attack('every population is called significant', 'abundance',
               *_patch(diffexp, 'differential_abundance',
                       _abundance_fake(0.5, 1e-9))),
        Attack('no population is ever significant', 'abundance',
               *_patch(diffexp, 'differential_abundance',
                       _abundance_fake(0.0, 1.0))),
        Attack('clustering puts every event in one cluster', 'clustering',
               *_patch(FlowSample, 'run_leiden', _one_cluster)),
        Attack('MEM scores every population identically', 'mem',
               *_patch(annotate, 'mem_scores',
                       _mem_ignores_the_data(annotate.mem_scores))),
        Attack('MEM loses the direction of the marker', 'mem',
               *_patch(annotate, 'mem_scores',
                       _mem_drops_the_sign(annotate.mem_scores))),
        Attack('clustering returns random labels', 'clustering',
               *_patch(FlowSample, 'run_leiden', _random_clusters)),
    ]


def main(argv=None):
    logging.disable(logging.CRITICAL)
    argv = list(sys.argv[1:] if argv is None else argv)

    print('\nBaseline — nothing broken; every group must be clean.\n')
    clean = True
    for name in GROUPS:
        failed = _failures(name)
        print(f'  {name:22} {"clean" if not failed else "FAILING: " + str(failed)}')
        clean = clean and not failed
    if not clean:
        print('\n  A control is already failing. Fix that before reading the '
              'attacks below — they cannot be interpreted against a red '
              'baseline.\n')

    attacks = [a for a in build_attacks()
               if not argv or any(t in a.name or t == a.group for t in argv)]
    print(f'\nAttacks — each must be CAUGHT by at least one control. '
          f'({len(attacks)} selected)\n')

    survived = []
    for attack in attacks:
        failed = attack.run()
        if failed:
            print(f'  caught   {attack.name}')
            print(f'           by {", ".join(failed)}')
        else:
            print(f'  SURVIVED {attack.name}')
            print(f'           nothing in "{attack.group}" noticed — that '
                  'control cannot fail for this break')
            survived.append(attack.name)

    print()
    if survived:
        print(f'  {len(survived)} of {len(attacks)} attacks went unnoticed:')
        for name in survived:
            print(f'    - {name}')
        print('\n  A control that survives a real break is theatre. Either '
              'tighten it, or\n  add one that covers this case.\n')
        return 1
    print(f'  All {len(attacks)} attacks were caught — the controls have '
          'teeth.\n')
    return 0 if clean else 1


if __name__ == '__main__':
    raise SystemExit(main())
