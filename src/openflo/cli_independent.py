"""openflo-run's independent mode: each trial folder is its own analysis,
its samples processed (in parallel with --workers > 1) and its groups
compared, into a sub-folder per trial when there is more than one.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import os

from . import cli_compare, cli_parallel
from .cli_args import DEFAULT_DPI
from .cli_samples import fcs_path
from .cli_trials import (
    already_done,
    build_fmo_thresholds_for_groups,
    calibrate_bead_micron,
    trial_label,
)
from .shared_clustering import cluster_settings


def run_independent(trial_dirs, out_dir, groups, fmo_sets,
                     k, fmo_percentile, labels, gate_overrides, region_gates,
                     skip_existing,
                     workers, n_jobs, max_events, admission_gb,
                     vram_admission_gb, filter_debris_um, bead_um,
                     doublet_tol, report, random_state=42, pairs=None,
                     cytonorm=None, cluster_cfg=None, qc=False, dpi=DEFAULT_DPI,
                     keep_open=None):
    """Process each trial folder as a separate analysis; results go into
    per-trial sub-dirs. Samples within a trial are processed in parallel
    when workers > 1. Every group with >= 1 sample gets a UMAP; every
    pair of groups gets a comparison. Returns False when a trial's
    comparisons were not written (the other trials still run)."""
    settings = cluster_settings(k, n_jobs, max_events, vram_admission_gb,
                                random_state, cluster_cfg)
    settings['dpi'] = dpi           # every PNG the tasks and groups write
    n_trials = len(trial_dirs)
    published = True

    for trial_dir in trial_dirs:
        tl = trial_label(trial_dir)
        trial_out = os.path.join(out_dir, tl) if n_trials > 1 else out_dir

        # FMO thresholds for each set actually referenced by the groups
        fmo_thresh = build_fmo_thresholds_for_groups(
            trial_dir, groups, fmo_sets, fmo_percentile, gate_overrides,
            report, trial_label_prefix=tl)

        # Bead-based µm/FSC calibration (used only if debris filter is on).
        debris_min_fsc = None
        if filter_debris_um and filter_debris_um > 0:
            factor, _ = calibrate_bead_micron(trial_dir, fmo_sets, bead_um)
            if factor:
                debris_min_fsc = float(filter_debris_um) / factor
                print(f"[Calibration] Debris cut-off: {filter_debris_um:.1f} µm "
                      f"→ FSC-A >= {debris_min_fsc:,.0f}", flush=True)
            else:
                print(f"[Calibration] No readable bead file in {tl} — "
                      f"debris filter disabled for this trial.", flush=True)

        # Build per-sample tasks. A sample --skip-existing finds done is not
        # processed again, but its task is kept so the group comparison can
        # reload its events (see write_group_outputs).
        tasks, reused = [], {}
        for g in groups:
            for name in g['samples']:
                # Per-sample FMO set (override → group default → ungated).
                sfmo = g.get('sample_fmo', {}).get(name, g['fmo_set'])
                thresh = fmo_thresh.get(sfmo, {})
                done = skip_existing and already_done(trial_out, g['name'], name)
                if done:
                    report(f'Skipping {name} (output already exists)')
                # By-day groups carry their own source folder; fall back to
                # the run's trial_dir for the classic single-folder layout.
                src_dir = g.get('trial_dir') or trial_dir
                try:
                    paths, gone = [(name, fcs_path(src_dir, name))], []
                except FileNotFoundError as e:
                    if not done:
                        report(f'[{tl}] {name}: {e}')
                        continue
                    paths, gone = [], [str(e)]   # kept, but cannot reload
                task = dict(settings,
                            name=name, group=g['name'], out=trial_out,
                            paths=paths, missing_inputs=gone,
                            thresholds=thresh,
                            region_gates=region_gates, labels=labels,
                            debris_min_fsc=debris_min_fsc,
                            doublet_tol=doublet_tol, cytonorm=cytonorm,
                            batch_id=src_dir, qc=qc)
                if done:
                    reused.setdefault(g['name'], []).append(task)
                else:
                    tasks.append(task)

        failures = {}
        results = cli_parallel.run_sample_tasks(tasks, workers, admission_gb, report,
                                    failures)
        published &= cli_compare.write_group_outputs(
            groups, results, reused, trial_out, report, settings,
            random_state=random_state, pairs=pairs, tl=tl,
            failures=failures, workers=workers, keep_open=keep_open,
            tasks=tasks + [t for ts in reused.values() for t in ts])
    return published
