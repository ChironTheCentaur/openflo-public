"""openflo-run's argparse-driven entry point: main() parses the options,
resolves the groups, gates and channel labels, calls run() and writes the
workspace, the --xlsx workbook and the run manifest.

Moved out of cli.py unchanged; openflo.cli.main is still this function, and
the old private cli name of the workbook writer still works, deprecated.
Never import openflo.cli from here: cli imports this module after capping
the BLAS threads and choosing the matplotlib backend.
"""
from __future__ import annotations

import json
import logging
import os

from ._console import force_utf8_streams, hide_future_warnings
from .cli_defaults import DEFAULT_FMO_SETS, DEFAULT_GROUPS
from .cli_gates import (
    gate_name,
    parse_gates_arg,
    population_gates,
    read_gate_file,
)
from .cli_groups import (
    auto_groups_by_day,
    group_spec_problems,
    sample_file_problems,
)
from .cli_manifest import write_run_manifest
from .cli_panel import find_panel_xlsx, parse_labels, read_staining_panel
from .cli_plots import parse_pairs
from .cli_run import run
from .cli_samples import fcs_path, normalise_groups
from .cli_unmix import run_batch_unmix
from .cli_wsp import export_pipeline_workspace
from .pipeline import FlowSample


def write_tables_workbook(out_dir, since):
    """--xlsx: every CSV table this run wrote, in one workbook."""
    from .table_export import csvs_to_workbook
    path = os.path.join(out_dir, 'openflo_tables.xlsx')
    try:
        included, skipped = csvs_to_workbook(out_dir, path, since=since)
    except OSError as exc:
        print(f"[xlsx] not written: {exc}", flush=True)
        return
    if included:
        print(f"[xlsx] {len(included)} table(s) → {path}", flush=True)
    else:
        print("[xlsx] no CSV tables to collect", flush=True)
    for note in skipped:
        print(f"[xlsx] left out {note}", flush=True)


def main() -> int:
    """argparse-driven entry point for the ``openflo-run`` console script.

    Mirrors the historical ``python run_analysis.py ...`` invocation
    exactly. Returns 0 on clean exit; non-zero from argparse or an
    unhandled exception bubbles via sys.exit / SystemExit.
    """
    force_utf8_streams()
    hide_future_warnings()
    import multiprocessing
    multiprocessing.freeze_support()

    from .cli_args import build_parser, check_args
    ap = build_parser()
    args = ap.parse_args()
    check_args(ap, args)

    # Configure the root logger ONCE based on -v / -q, before any
    # `from .pipeline import ...` runs in this process. Worker processes
    # start fresh and copy this level and format (_console.start_worker).
    # Users wanting JSON / file output should configure `logging`
    # themselves before importing openflo.
    if args.quiet:
        _log_level = logging.WARNING
    elif args.verbose >= 2:
        _log_level = logging.DEBUG
    else:
        _log_level = logging.INFO
    logging.basicConfig(
        level=_log_level,
        format='%(asctime)s %(levelname)-7s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        force=True,    # override any prior configuration
    )

    # Spectral batch-unmixing is a self-contained mode — run it and exit
    # before the standard analysis-pipeline argument resolution.
    if args.unmix:
        return run_batch_unmix(args)

    # Resolve trial dirs: --trials takes precedence, else --fcs
    raw_trials = args.trials or args.fcs or '.'
    trial_dirs = [t.strip() for t in raw_trials.split(',') if t.strip()]

    # Resolve groups / fmo_sets — four paths in order of preference:
    #   1. --groups (explicit) → use as-is. Pair with --fmo-sets if given.
    #   2. --samples (legacy) → auto-split by "late" prefix into two groups.
    #   3. Neither, but folders contain FCS → by-DAY auto-grouping: each
    #      folder (incl. sub-folders of a parent you point at) becomes a
    #      day/group, sampled independently, compared across days in one
    #      analysis. This is the new default.
    #   4. Fallback → a single generic group over the named samples.
    if args.groups:
        groups = json.loads(args.groups)
        fmo_sets = (json.loads(args.fmo_sets)
                    if args.fmo_sets else DEFAULT_FMO_SETS)
    elif args.samples:
        names = [s.strip() for s in args.samples.split(',') if s.strip()]
        # Optional two-group split on a name prefix (override with --groups).
        group_b = [n for n in names if n.lower().startswith('late')]
        group_a = [n for n in names if not n.lower().startswith('late')]
        groups = []
        if group_a:
            groups.append({'name': 'Group A',
                           'samples': group_a, 'fmo_set': 'Set A'})
        if group_b:
            groups.append({'name': 'Group B',
                           'samples': group_b, 'fmo_set': 'Set B'})
        fmo_sets = (json.loads(args.fmo_sets)
                    if args.fmo_sets else DEFAULT_FMO_SETS)
    else:
        auto = auto_groups_by_day(trial_dirs)
        if auto:
            groups = auto
            fmo_sets = (json.loads(args.fmo_sets)
                        if args.fmo_sets else DEFAULT_FMO_SETS)
            print(f"[grouping] By-day: {len(auto)} folder(s) → "
                  f"{', '.join(g['name'] for g in auto)}", flush=True)
            # Each group carries its own trial_dir, so collapse the run to
            # a SINGLE pass (one combined analysis comparing days). The
            # representative trial_dir below is only used for labels /
            # bead calibration fallbacks.
            trial_dirs = [trial_dirs[0] if trial_dirs else '.']
        else:
            groups   = DEFAULT_GROUPS
            fmo_sets = DEFAULT_FMO_SETS

    # Refuse an ambiguous group spec before anything reads an FCS (the
    # --panel probe below does), naming where the groups came from.
    norm = normalise_groups(groups)
    problems = (group_spec_problems(norm)
                + sample_file_problems(norm, trial_dirs))
    if problems:
        source = ('--groups' if args.groups else
                  '--samples' if args.samples else 'groups from the folders')
        for p in problems:
            print(f'[!] {source}: {p}', flush=True)
        return 2

    gate_overrides, region_gates = parse_gates_arg(args.gates,
                                                   units=args.gate_units)
    if args.gates_from:
        try:
            file_gates, note = read_gate_file(args.gates_from)
            picked = population_gates(file_gates, args.gate_population)
        except (OSError, ValueError) as exc:
            print(f'[!] --gates-from {os.path.basename(args.gates_from)}: '
                  f'{exc}', flush=True)
            return 2
        # Their own id space, so they cannot collide with --gates ids.
        from .gating import remap_operands
        ids = {g['id']: f"file:{g['id']}" for g in picked}
        for g in picked:
            g['id'] = ids[g['id']]
            g['parent_id'] = ids.get(g.get('parent_id'))
            if g.get('kind') == 'boolean':
                remap_operands(g, ids)
        shown = [gate_name(g) for g in picked if g.get('enabled', True)]
        print(f"[gates] {os.path.basename(args.gates_from)} ({note}): "
              f"{' > '.join(shown) or 'no gates'}", flush=True)
        region_gates = list(region_gates) + picked
    parsed_overrides = gate_overrides   # keep the dict form for the wsp export
    if not gate_overrides and not region_gates:
        gate_overrides = None   # preserve the original "no gates" semantics

    # Channel labels: staining-panel xlsx first, then --labels overrides.
    labels = {}
    panel_path = args.panel
    if panel_path == 'auto':
        panel_path = find_panel_xlsx(trial_dirs)
        if panel_path:
            print(f"[panel] Auto-detected: {panel_path}", flush=True)
    if panel_path:
        try:
            # Normalise first — the raw --groups spec may carry `samples` as a
            # comma-string or per-sample dicts, so groups[0]['samples'][0] would
            # be a single char / a dict and the probe would silently fail.
            _ng = normalise_groups(groups)
            probe_dir = _ng[0].get('trial_dir') or trial_dirs[0]
            probe = fcs_path(probe_dir, _ng[0]['samples'][0])
            channels = list(FlowSample(probe).data.columns)
            labels.update(read_staining_panel(panel_path, channels))
        except Exception as exc:  # noqa: BLE001 - reported; --labels still apply
            print(f"[panel] Could not apply panel: "
                  f"{type(exc).__name__}: {exc}", flush=True)
    labels.update(parse_labels(args.labels))   # explicit --labels win
    pairs = parse_pairs(args.pairs)
    if args.fast_graph and args.cluster_method != 'leiden':
        print(f"[!] --fast-graph applies only to --cluster-method leiden; "
              f"{args.cluster_method} runs on its usual graph.", flush=True)

    # Apply the user's palette choice process-wide before any plot is
    # rendered. Worker processes start fresh and copy it from this process
    # (_console.start_worker).
    if args.palette and args.palette != 'auto':
        from .pipeline import set_default_palette
        set_default_palette(args.palette)

    import datetime as _dt
    started = _dt.datetime.now(_dt.UTC)
    rc = run(
        trial_dirs     = trial_dirs,
        out_dir        = args.out,
        groups         = groups,
        fmo_sets       = fmo_sets,
        batch_mode     = args.batch_mode,
        k              = args.k,
        cluster_method = args.cluster_method,
        resolution     = args.resolution,
        n_metaclusters = args.n_metaclusters,
        reproducible   = not args.no_reproducible,
        fast_graph     = args.fast_graph,
        dpi            = args.dpi,
        fmo_percentile = args.fmo_percentile,
        labels         = labels,
        gate_overrides = gate_overrides,
        region_gates   = region_gates,
        show_plots     = args.show_plots,
        skip_existing  = args.skip_existing,
        workers        = args.workers,
        max_events     = args.max_events or None,
        admission_gb   = args.admission_gb,
        vram_admission_gb = args.vram_admission_gb,
        filter_debris_um = args.filter_debris_um,
        bead_um          = args.bead_um,
        doublet_tol      = args.doublet_tol,
        random_state     = args.seed,
        palette_name     = args.palette,
        pairs            = pairs,
        batch_correct    = args.batch_correct,
        cytonorm_mode    = args.cytonorm_mode,
        cytonorm_control = args.cytonorm_control,
        cytonorm_metaclusters = args.cytonorm_metaclusters,
        qc               = args.qc,
    )

    if args.export_wsp:
        export_pipeline_workspace(
            out_path       = args.export_wsp,
            trial_dirs     = trial_dirs,
            groups         = groups,
            gate_overrides = parsed_overrides,
            region_gates   = region_gates)

    if args.xlsx:
        # A little slack for file systems that store coarse times.
        write_tables_workbook(args.out, started.timestamp() - 2)

    try:
        write_run_manifest(args, trial_dirs, groups, parsed_overrides,
                           region_gates, rc, started)
    except OSError as exc:
        print(f"[manifest] not written: {exc}", flush=True)
    return rc
