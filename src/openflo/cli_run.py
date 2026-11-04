"""openflo-run's programmatic entry point: run() checks the groups, fits
the batch correction, sizes the workers and hands the trials to the
independent or concatenate mode.

Moved out of cli.py unchanged; openflo.cli.run is still this function, and
the old cli name of the GPU check still works, deprecated.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from .cli_args import DEFAULT_DPI
from .cli_concatenate import run_concatenated
from .cli_cytonorm import fit_cytonorm
from .cli_defaults import DEFAULT_FMO_SETS
from .cli_groups import group_spec_problems, sample_file_problems
from .cli_independent import run_independent
from .cli_plots import show_saved_group_plots
from .cli_samples import normalise_groups
from .som_kernel import FLOWSOM_DEFAULT_METACLUSTERS


def gpu_present():
    """True if an NVIDIA GPU is visible via nvidia-smi (driver-level check).
    Independent of whether RAPIDS cuML is installed."""
    try:
        import subprocess
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=6,
                             creationflags=flags)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001 - no usable nvidia-smi means no GPU
        return False


# ── Programmatic entry point ──────────────────────────────────────────────────
# `run(...)` takes keyword arguments and is suitable for importing from other
# Python code. `cli.main()` is the no-arg argparse-driven entry point
# that the `openflo-run` console script invokes.

def run(trial_dirs, out_dir, groups, fmo_sets=None, batch_mode='independent',
        k=30, fmo_percentile=99.0, labels=None, gate_overrides=None,
        region_gates=None,
        show_plots=False, skip_existing=False, workers=1, max_events=None,
        admission_gb=4.0, vram_admission_gb=1.0,
        filter_debris_um=0.0, bead_um=8.0, doublet_tol=0.0,
        random_state=42, palette_name='auto', pairs=None,
        batch_correct=False, cytonorm_mode='goal', cytonorm_metaclusters=10,
        cytonorm_control='',
        cluster_method='phenograph', resolution=1.0,
        n_metaclusters=FLOWSOM_DEFAULT_METACLUSTERS,
        reproducible=True, qc=False, dpi=DEFAULT_DPI, fast_graph=False):
    """Run the full pipeline for a list of user-defined sample groups.

    groups
        List of dicts: ``[{'name': 'Day 1', 'samples': ['d1m1','d1m2'],
        'fmo_set': 'standard'}, ...]``. Supports any number of named groups
        (group 1 vs group 2, day 1 .. day 8, treated vs control, etc).

    fmo_sets
        Dict mapping a set name to a ``{detector_channel: fcs_basename}``
        dict that the FMO threshold builder needs. Defaults to
        ``DEFAULT_FMO_SETS`` (the Late / Early pair).

    fast_graph
        Leiden only: cluster on the pruned neighbour graph (PhenoGraph and
        Seurat's form) instead of the default dense one. Faster, but the
        clusters differ. Off by default.

    qc
        Run acquisition QC (FlowSample.run_qc) on every file as it loads.
        Off by default, as in the app.

    Returns 0, or 1 when group comparisons asked for were not written: a
    sample was missing, the shared clustering failed, or a file in use kept
    the new set out (the run still finishes every trial).
    """

    if isinstance(trial_dirs, str):
        trial_dirs = [trial_dirs]
    trial_dirs = [d for d in trial_dirs if d]

    if fmo_sets is None:
        fmo_sets = DEFAULT_FMO_SETS
    groups = normalise_groups(groups)
    if not groups:
        print("[!] No groups defined — nothing to do.", flush=True)
        return 0
    problems = (group_spec_problems(groups)
                + sample_file_problems(groups, trial_dirs))
    if problems:
        raise ValueError('invalid groups: ' + ' '.join(problems))

    os.makedirs(out_dir, exist_ok=True)

    # Batch correction (CytoNorm) — fit once across batches up front; the
    # serialized model rides along on each task and is applied in the worker.
    cytonorm_dict = None
    if batch_correct:
        default_trial = trial_dirs[0] if trial_dirs else '.'
        cytonorm_dict, _ = fit_cytonorm(
            groups, default_trial, mode=cytonorm_mode,
            n_metaclusters=cytonorm_metaclusters,
            control_token=cytonorm_control, qc=qc)

    from .gpu_probe import GPU_AVAILABLE, GPU_CLUSTERING_AVAILABLE, GPU_NAME
    if GPU_AVAILABLE:
        clu = "+ clustering (cuML + cuGraph)" if GPU_CLUSTERING_AVAILABLE else "(UMAP only)"
        print(f"[GPU] {GPU_NAME} detected — running UMAP {clu}", flush=True)
        if GPU_CLUSTERING_AVAILABLE:
            print(f"[GPU] Per-worker VRAM admission: ≥ {vram_admission_gb:.1f} GB free "
                  "to use GPU clustering (else CPU Phenograph).", flush=True)
    elif gpu_present():
        print("[GPU] NVIDIA GPU detected, but RAPIDS cuML is not installed — "
              "UMAP & clustering run on CPU. Install cuML+cuGraph (Linux/WSL2) "
              "for GPU acceleration.", flush=True)
    else:
        print("[GPU] No CUDA GPU detected — UMAP will run on CPU (numba JIT)", flush=True)

    # Parallel sample processing.
    #
    # When running multiple outer workers in parallel we deliberately set
    # n_jobs=1 inside Phenograph instead of `cores // workers`. The reason:
    # at any n_jobs other than 1, phenograph's Jaccard step spawns a
    # multiprocessing.Pool() of cpu_count() processes, and its size sort a
    # Pool of n_jobs (FlowSample.cluster caps that one; phenograph alone
    # would always use cpu_count()). On Windows each inner child must
    # re-import the full numpy/scipy/sklearn stack (~250 MB) AND receive a
    # fresh copy of the events array, which multiplies the per-worker RAM
    # footprint. Forcing n_jobs=1 keeps every outer worker to a single
    # process; total parallelism is simply `workers` (each one runs Louvain
    # on its own core, which is the single-threaded bottleneck anyway).
    workers = max(1, int(workers))
    cores   = os.cpu_count() or 2
    n_jobs  = -1 if workers <= 1 else 1

    # RAM-aware throttling — cap workers to what free RAM can actually hold.
    try:
        import psutil
        free_gb  = psutil.virtual_memory().available / (1024 ** 3)
        # Per-worker peak estimate (calibrated against observed Windows runs).
        # Each outer worker holds, at peak:
        #   • imports (numpy/scipy/sklearn/phenograph/igraph)         ~0.5 GB
        #   • kNN graph (indices + distances)                          events × ~30 B
        #   • Jaccard sparse graph + intermediate copies              events × ~80 B
        #   • Louvain native binary-file scratch + igraph state       events × ~30 B
        #   • Phenograph's pickled return buffers + igraph leiden     events × ~15 B
        # → ≈ 0.5 GB + events × 4 µ-GB.  For 500 k events that's ~2.5 GB,
        # which matched the working-set we saw before the pagefile gave out.
        events_per_worker = max_events if max_events else 3_000_000
        per_worker_gb     = max(1.0, 0.5 + events_per_worker * 4.0e-6)
        # Reserve 2 GB for the parent process + OS headroom.
        budget_gb         = max(1.0, free_gb - 2.0)
        max_safe          = max(1, int(budget_gb / per_worker_gb))
        if workers > max_safe:
            print(f"[RAM-throttle] Free RAM {free_gb:.1f} GB, per-worker estimate "
                  f"~{per_worker_gb:.1f} GB → reducing workers {workers} → {max_safe}",
                  flush=True)
            workers = max_safe
    except ImportError:
        pass

    n_trials = len(trial_dirs)
    cap_msg  = f"  |  cluster cap: {max_events:,} events" if max_events else ""
    group_msg = " / ".join(f"{g['name']}({len(g['samples'])})" for g in groups)
    print(f"[Trials] {n_trials} trial folder(s)  |  mode: {batch_mode}  |  "
          f"parallel workers: {workers}  ({cores} cores){cap_msg}", flush=True)
    print(f"[Groups] {group_msg}", flush=True)

    # Step total — one [STEP N/M] line per unit of work the runners will do:
    #   • one per unique FMO set built  (per trial in independent mode,
    #     once in concatenate mode)
    #   • one per sample processed (or skipped)
    #   • one per group that gets a UMAP run (every group with >= 1
    #     processed sample; counting only groups with >= 2 printed
    #     [STEP 5/3] for two single-sample groups)
    #   • one for the shared clustering the group comparisons use
    #   • one per pair of groups (pairwise condition comparison)
    # A step that does not happen (failed sample, UMAP left to an earlier
    # run) only leaves the count short of the total, never past it.
    # Unique FMO sets across group defaults + per-sample overrides
    # (one build step each; '' = ungated builds nothing).
    _all_sets = set()
    for g in groups:
        _all_sets.add(g.get('fmo_set', ''))
        _all_sets.update(g.get('sample_fmo', {}).values())
    n_used_sets = len({s for s in _all_sets if s})
    n_samples   = sum(len(g['samples']) for g in groups)
    n_umaps     = len(groups)       # normalised groups all hold >= 1 sample
    n_compares  = (len(groups) * (len(groups) - 1)) // 2
    n_shared    = 1 if n_compares else 0
    per_trial_steps = (n_used_sets + n_samples + n_umaps + n_shared
                       + n_compares)
    total = (per_trial_steps if batch_mode == 'concatenate'
             else n_trials * per_trial_steps)
    total = max(total, 1)   # avoid 0/0 in the progress bar

    step = [0]

    def report(msg):
        step[0] += 1
        print(f'[STEP {step[0]}/{total}] {msg}', flush=True)

    # Propagate the palette choice to the worker side too — if main()
    # is being called from a script that didn't go through __main__,
    # the palette set earlier in __main__ won't apply here.
    if palette_name and palette_name != 'auto':
        from .pipeline import set_default_palette
        set_default_palette(palette_name)

    _cluster_cfg = {'method': cluster_method, 'resolution': resolution,
                    'n_metaclusters': n_metaclusters,
                    'reproducible': reproducible,
                    'fast_graph': bool(fast_graph)}
    if batch_mode == 'concatenate':
        published = run_concatenated(
            trial_dirs, out_dir, groups, fmo_sets,
            k, fmo_percentile, labels, gate_overrides, region_gates,
            skip_existing, workers, n_jobs, max_events,
            admission_gb, vram_admission_gb,
            filter_debris_um, bead_um, doublet_tol, report,
            random_state=random_state, pairs=pairs,
            cytonorm=cytonorm_dict, cluster_cfg=_cluster_cfg, qc=qc, dpi=dpi,
            keep_open=bool(show_plots))
    else:
        published = run_independent(
            trial_dirs, out_dir, groups, fmo_sets,
            k, fmo_percentile, labels, gate_overrides, region_gates,
            skip_existing, workers, n_jobs, max_events,
            admission_gb, vram_admission_gb,
            filter_debris_um, bead_um, doublet_tol, report,
            random_state=random_state, pairs=pairs,
            cytonorm=cytonorm_dict, cluster_cfg=_cluster_cfg, qc=qc, dpi=dpi,
            keep_open=bool(show_plots))

    print(f"\nDone. Results in: {out_dir}")
    if not published:
        print("[!] Some group comparisons were not written; see the messages "
              "above.", flush=True)
    if show_plots:
        # If nothing fresh was drawn (e.g. every sample was --skip-existing'd
        # and no group-UMAPs ran because there weren't enough live samples
        # to UMAP), reload the saved group-level PNGs from disk so the user
        # can still review them interactively.
        if not plt.get_fignums():
            opened = show_saved_group_plots(out_dir)
            if opened == 0:
                print("[Show plots] No figures to display — nothing fresh "
                      "this run and no saved group-level PNGs found in "
                      f"{out_dir}.", flush=True)
        plt.show()
    return 0 if published else 1
