"""openflo-run's concatenate mode: each sample is loaded from every trial
folder, its events merged with their trial of origin kept, and clustered
once; FMO thresholds and the bead calibration come from the first trial.

Moved out of cli.py unchanged; the old cli name still works, deprecated.
"""
from __future__ import annotations

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


def run_concatenated(trial_dirs, out_dir, groups, fmo_sets,
                     k, fmo_percentile, labels, gate_overrides, region_gates,
                     skip_existing,
                     workers, n_jobs, max_events, admission_gb,
                     vram_admission_gb, filter_debris_um, bead_um,
                     doublet_tol, report, random_state=42, pairs=None,
                     cytonorm=None, cluster_cfg=None, qc=False, dpi=DEFAULT_DPI,
                     keep_open=None):
    """Load each sample from every trial, concatenate with trial-origin labels,
    then cluster / UMAP once on the merged data per sample. Different
    samples are processed in parallel when workers > 1. FMO thresholds
    come from the first (reference) trial only."""
    settings = cluster_settings(k, n_jobs, max_events, vram_admission_gb,
                               random_state, cluster_cfg)
    settings['dpi'] = dpi           # every PNG the tasks and groups write
    ref_dir = trial_dirs[0]

    # FMO thresholds for each set actually referenced by the groups
    fmo_thresh = build_fmo_thresholds_for_groups(
        ref_dir, groups, fmo_sets, fmo_percentile, gate_overrides,
        report, trial_label_prefix='ref trial')

    # Bead-based µm/FSC calibration from the reference trial.
    debris_min_fsc = None
    if filter_debris_um and filter_debris_um > 0:
        factor, _ = calibrate_bead_micron(ref_dir, fmo_sets, bead_um)
        if factor:
            debris_min_fsc = float(filter_debris_um) / factor
            print(f"[Calibration] Debris cut-off: {filter_debris_um:.1f} µm "
                  f"→ FSC-A >= {debris_min_fsc:,.0f}", flush=True)
        else:
            print("[Calibration] No readable bead file in reference trial — "
                  "debris filter disabled.", flush=True)

    # Build per-sample tasks — paths span every trial folder. A sample
    # --skip-existing finds done keeps its task for the comparison's reload.
    tasks, reused = [], {}
    for g in groups:
        for name in g['samples']:
            # Per-sample FMO set (override → group default → ungated).
            sfmo = g.get('sample_fmo', {}).get(name, g['fmo_set'])
            thresh = fmo_thresh.get(sfmo, {})
            done = skip_existing and already_done(out_dir, g['name'], name)
            if done:
                report(f'Skipping {name} (output already exists)')
            # A by-day group names its own folder: resolve its samples there.
            # Searching the run's folders instead (collapsed to the parent
            # for by-day runs) found the same filename in every day folder
            # and took the first one — day 0's file — for every day.
            src_dirs = [g['trial_dir']] if g.get('trial_dir') else trial_dirs
            # A trial folder without the sample's file: the sample is still
            # processed from the others (with a warning), but it is missing
            # data, so no comparison is written (not_in_this_run). A kept
            # sample was reloaded from the remaining folders without a word:
            # Treat fell from 60 % to 30 % beside a t1_stats.csv made from
            # both trials.
            paths, gone = [], []
            for trial_dir in src_dirs:
                try:
                    paths.append((trial_label(trial_dir),
                                  fcs_path(trial_dir, name)))
                except FileNotFoundError as e:
                    gone.append(str(e))
                    if not done:
                       print(f"  [!] {e}")
            if not paths and not done:
                report(f'{name}: not found in any trial folder')
                continue
            task = dict(settings,
                       name=name, group=g['name'], out=out_dir,
                       paths=paths, missing_inputs=gone, thresholds=thresh,
                       region_gates=region_gates, labels=labels,
                       debris_min_fsc=debris_min_fsc,
                       doublet_tol=doublet_tol, cytonorm=cytonorm,
                       batch_id=g.get('trial_dir') or ref_dir, qc=qc)
            if done:
                reused.setdefault(g['name'], []).append(task)
            else:
                tasks.append(task)

    failures = {}
    results = cli_parallel.run_sample_tasks(tasks, workers, admission_gb, report,
                               failures)
    return cli_compare.write_group_outputs(
        groups, results, reused, out_dir, report, settings,
        random_state=random_state, pairs=pairs, failures=failures,
        workers=workers, keep_open=keep_open,
        tasks=tasks + [t for ts in reused.values() for t in ts])
