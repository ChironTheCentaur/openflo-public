"""
openflo command-line runner.
----------------------------
Panel-agnostic flow-cytometry analysis: point it at one or more FCS trials,
describe your groups / samples / FMO controls, and it compensates, gates,
clusters, and exports per-sample / per-condition stats, plots and a `.wsp`.

The constants below (example panel, FMO sets, groups) are just defaults used
when you don't pass --groups / --samples / --fmo-sets — replace them with your
own, or supply the flags per run.

Usage (single trial):
    openflo-run --trials /path/to/fcs/ --samples sample_1,sample_2 --out results/

Usage (multiple trials — independent):
    openflo-run --trials /trial1,/trial2,/trial3 --samples sample_1 --batch-mode independent

Usage (multiple trials — concatenated, cells retain trial origin):
    openflo-run --trials /trial1,/trial2 --samples sample_1 --batch-mode concatenate
"""

import os
import sys
import warnings

# Imported before _cap_blas_threads, as it always was, although cli no longer
# uses it: numpy reads the BLAS thread variables on import, so this keeps the
# main process's BLAS pool as it was (the cap reaches the workers only).
import numpy as np  # noqa: F401  # pyright: ignore[reportUnusedImport]


# Cap each process's BLAS thread pool to a fair share of cores. Must run
# BEFORE any numpy / scipy / sklearn import — those libraries read these
# env vars at load time and never re-check. Each parallel sample worker
# inherits this via os.environ, so on a 24-core box with --workers 4 each
# worker gets 6 BLAS threads instead of all spawning 24 (= 96 total threads,
# which is what was triggering the 'OpenBLAS error after 10 retries' kills).
def _cap_blas_threads():
    try:
        workers = 1
        if '--workers' in sys.argv:
            i = sys.argv.index('--workers')
            if i + 1 < len(sys.argv):
                workers = max(1, int(sys.argv[i + 1]))
        cores = os.cpu_count() or 2
        n = max(1, cores // max(1, workers))
        for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ.setdefault(var, str(n))
    except Exception:  # noqa: BLE001 - thread-count hint is best-effort
        pass
_cap_blas_threads()

# UTF-8 stdout/stderr (so ≥, →, … don't crash a cp1252 console) is applied at
# the top of main(), NOT here.
#
# It must not run at import: reconfiguring a global stream is a side effect no
# library should have on import. It breaks pytest-xdist outright — workers
# talk to the controller over stdout via execnet, so merely importing this
# module inside a worker would corrupt the channel and kill it with
# "EOFError: expected 1 bytes, got 0", taking an unrelated test in that
# worker down with it. The same hazard applies to any embedding host or
# notebook that imports openflo.cli as a library.
#
# main() is the right place: it is the only entry point that owns the process's
# streams, and calling it there (before the parser is built) also covers
# `--help`.
# E402 noqa on the imports below: backend selection MUST happen after
# `import matplotlib` and BEFORE matplotlib.pyplot is imported (the cli_*
# modules below import it), so they have to be interleaved with logic.
import matplotlib  # noqa: E402

from ._console import (  # noqa: E402
    force_utf8_streams as force_utf8_streams,  # still a public cli name
)
from ._console import (  # noqa: E402
    hide_future_warnings as hide_future_warnings,  # still a public cli name
)

# Headless backend for the main run (unless --show-plots) and for every
# multiprocessing child ('__mp_main__'). Left untouched when imported elsewhere
# (e.g. preview.py) so that interactive previews still open a window.
if __name__ in ('__main__', '__mp_main__') and '--show-plots' not in sys.argv:
    matplotlib.use('Agg')

from .cli_defaults import (  # noqa: E402
    CD902 as CD902,  # still a public cli name
)
from .cli_defaults import (  # noqa: E402
    CD903 as CD903,  # still a public cli name
)
from .cli_defaults import (  # noqa: E402
    DEFAULT_FMO_SETS as DEFAULT_FMO_SETS,  # still a public cli name
)
from .cli_defaults import (  # noqa: E402
    DEFAULT_GROUPS as DEFAULT_GROUPS,  # still a public cli name
)
from .cli_defaults import (  # noqa: E402
    FMOS_SET_A as FMOS_SET_A,  # still a public cli name
)
from .cli_defaults import (  # noqa: E402
    FMOS_SET_B as FMOS_SET_B,  # still a public cli name
)
from .cli_defaults import (  # noqa: E402
    CD901b as CD901b,  # still a public cli name
)
from .cli_gates import (  # noqa: E402
    gate_name as gate_name,  # still a public cli name
)
from .cli_gates import (  # noqa: E402
    parse_gates_arg as parse_gates_arg,  # still a public cli name
)
from .cli_gates import (  # noqa: E402
    population_gates as population_gates,  # still a public cli name
)
from .cli_gates import (  # noqa: E402
    read_gate_file as read_gate_file,  # still a public cli name
)
from .cli_groups import (  # noqa: E402
    auto_groups_by_day as auto_groups_by_day,  # still a public cli name
)
from .cli_groups import (  # noqa: E402
    group_spec_problems as group_spec_problems,  # still a public cli name
)
from .cli_groups import output_name  # noqa: E402
from .cli_groups import (  # noqa: E402
    sample_file_problems as sample_file_problems,  # still a public cli name
)
from .cli_main import (  # noqa: E402
    main as main,  # still a public cli name
)
from .cli_manifest import (  # noqa: E402
    MANIFEST_NAME as MANIFEST_NAME,  # still a public cli name
)
from .cli_manifest import (  # noqa: E402
    write_run_manifest as write_run_manifest,  # still a public cli name
)
from .cli_panel import (  # noqa: E402
    find_panel_xlsx as find_panel_xlsx,  # still a public cli name
)
from .cli_panel import (  # noqa: E402
    parse_labels as parse_labels,  # still a public cli name
)
from .cli_panel import (  # noqa: E402
    read_staining_panel as read_staining_panel,  # still a public cli name
)
from .cli_plots import (  # noqa: E402
    DEFAULT_SCATTER_PAIRS as DEFAULT_SCATTER_PAIRS,  # still a public cli name
)
from .cli_plots import (  # noqa: E402
    parse_pairs as parse_pairs,  # still a public cli name
)
from .cli_plots import (  # noqa: E402
    run_group_umap as run_group_umap,  # still a public cli name
)
from .cli_plots import (  # noqa: E402
    save_group_pair_scatters as save_group_pair_scatters,  # still a public cli name
)
from .cli_plots import (  # noqa: E402
    save_plots as save_plots,  # still a public cli name
)
from .cli_run import (  # noqa: E402
    run as run,  # still a public cli name
)
from .cli_samples import (  # noqa: E402
    fcs_path as fcs_path,  # still a public cli name
)
from .cli_samples import (  # noqa: E402
    normalise_groups as normalise_groups,  # still a public cli name
)
from .cli_trials import (  # noqa: E402
    build_fmo_thresholds as build_fmo_thresholds,  # still a public cli name
)
from .cli_unmix import (  # noqa: E402
    run_batch_unmix as run_batch_unmix,  # still a public cli name
)
from .cli_wsp import (  # noqa: E402
    export_pipeline_workspace as export_pipeline_workspace,  # still a public cli name
)
from .pipeline import (  # noqa: E402
    FlowSample as FlowSample,  # still a public cli name
)

# Old names of what shared_clustering now holds, deprecated and served by
# __getattr__: {old cli name: new shared_clustering name}.
_MOVED_TO_SHARED_CLUSTERING = {
    '_cluster_settings': 'cluster_settings',
    '_cluster_by_method': 'cluster_by_method',
    '_equal_quota': 'equal_quota',
    '_fold_label': 'fold_label',
    '_fluor_tokens': 'fluor_tokens',
    '_label_key': 'label_key',
    '_align_shared_channels': 'align_shared_channels',
    'align_shared_channels': 'align_shared_channels',
    '_shared_cluster_labels': 'shared_cluster_labels',
    '_shared_pool_cap': 'shared_pool_cap',
}

# Old names of what cli_gates now holds, deprecated and served by
# __getattr__: {old cli name: new cli_gates name}.
_MOVED_TO_CLI_GATES = {
    '_parse_gates_arg': 'parse_gates_arg',
    '_data_units_to_stored': 'data_units_to_stored',
    '_pipeline_transform_specs': 'pipeline_transform_specs',
    '_gate_name': 'gate_name',
    '_session_gates': 'session_gates',
    '_read_gate_file': 'read_gate_file',
    '_population_gates': 'population_gates',
    '_pipeline_gate_transforms': 'pipeline_gate_transforms',
}

# Old names of what cli_unmix now holds, deprecated and served by
# __getattr__: {old cli name: new cli_unmix name}. run_batch_unmix keeps
# its name and is imported above.
_MOVED_TO_CLI_UNMIX = {
    '_load_unmix_controls': 'load_unmix_controls',
    '_write_unmix_qc': 'write_unmix_qc',
}

# Old names of what cli_samples now holds, deprecated and served by
# __getattr__: {old cli name: new cli_samples name}. fcs_path keeps its
# name and is imported above.
_MOVED_TO_CLI_SAMPLES = {
    '_normalise_groups': 'normalise_groups',
    '_scan_for_token': 'scan_for_token',
    '_find_unstained': 'find_unstained',
}

# Old names of what cli_wsp now holds, deprecated and served by
# __getattr__: {old cli name: new cli_wsp name}.
_MOVED_TO_CLI_WSP = {
    '_build_export_gate_list': 'build_export_gate_list',
    '_own_spillover': 'own_spillover',
    '_export_pipeline_workspace': 'export_pipeline_workspace',
}

# Old names of what cli_manifest now holds, deprecated and served by
# __getattr__: {old cli name: new cli_manifest name}. MANIFEST_NAME keeps
# its name and is imported above.
_MOVED_TO_CLI_MANIFEST = {
    '_MANIFEST_PACKAGES': 'MANIFEST_PACKAGES',
    '_sha256': 'sha256',
    '_file_entry': 'file_entry',
    '_run_input_files': 'run_input_files',
    '_write_run_manifest': 'write_run_manifest',
}

# Old names of what cli_groups now holds, deprecated and served by
# __getattr__: {old cli name: new cli_groups name}. output_name keeps its
# name and is imported above.
_MOVED_TO_CLI_GROUPS = {
    '_compare_base': 'compare_base',
    '_group_spec_problems': 'group_spec_problems',
    '_sample_file_problems': 'sample_file_problems',
    '_discover_day_folders': 'discover_day_folders',
    '_auto_groups_by_day': 'auto_groups_by_day',
}

# Old names of what cli_plots now holds, deprecated and served by
# __getattr__: {old cli name: new cli_plots name}. DEFAULT_SCATTER_PAIRS,
# parse_pairs, run_group_umap, save_group_pair_scatters and save_plots keep
# their names and are imported above. run() passes --show-plots down as
# keep_open; KEEP_FINAL_PLOTS is only the default when keep_open is left
# out, and setting the old cli name has no effect.
_MOVED_TO_CLI_PLOTS = {
    '_resolve_pair': 'resolve_pair',
    '_intensity_axes': 'intensity_axes',
    '_show_saved_group_plots': 'show_saved_group_plots',
    '_KEEP_FINAL_PLOTS': 'KEEP_FINAL_PLOTS',
    '_STAGING_PREFIX': 'STAGING_PREFIX',
}

# Old names of what cli_parallel now holds, deprecated and served by
# __getattr__: {old cli name: new cli_parallel name}. cli calls them on
# cli_parallel, so that is where a test patches them.
_MOVED_TO_CLI_PARALLEL = {
    '_load_task_sample': 'load_task_sample',
    '_process_sample_task': 'process_sample_task',
    '_write_sample_outputs': 'write_sample_outputs',
    '_run_sample_tasks': 'run_sample_tasks',
}

# Old names of what cli_trials now holds, deprecated and served by
# __getattr__: {old cli name: new cli_trials name}. build_fmo_thresholds
# keeps its name and is imported above.
_MOVED_TO_CLI_TRIALS = {
    '_trial_label': 'trial_label',
    '_calibrate_bead_micron': 'calibrate_bead_micron',
    '_already_done': 'already_done',
    '_build_fmo_thresholds_for_groups': 'build_fmo_thresholds_for_groups',
}

# Old names of what cli_compare now holds, deprecated and served by
# __getattr__: {old cli name: new cli_compare name}. cli and
# cli_independent call write_group_outputs on cli_compare, so that is where
# a test patches it (and compare_group_pair, run_group_umap and
# save_group_pair_scatters, which it calls).
_MOVED_TO_CLI_COMPARE = {
    '_SHARED_IDS_NOTE': 'SHARED_IDS_NOTE',
    '_COMPARE_OWN': 'COMPARE_OWN',
    '_compare_files': 'compare_files',
    '_compare_group_pair': 'compare_group_pair',
    '_add_pvalues': 'add_pvalues',
    '_shared_n_jobs': 'shared_n_jobs',
    '_STAGING_MARK': 'STAGING_MARK',
    '_is_staging': 'is_staging',
    '_clear_stale_staging': 'clear_stale_staging',
    '_remove_earlier_set': 'remove_earlier_set',
    '_publish_compare_set': 'publish_compare_set',
    '_not_in_this_run': 'not_in_this_run',
    '_samples_to_compare': 'samples_to_compare',
    '_group_plots_job': 'group_plots_job',
    '_draw_group_plots': 'draw_group_plots',
    '_write_group_outputs': 'write_group_outputs',
}

# Old names of what cli_independent now holds, deprecated and served by
# __getattr__: {old cli name: new cli_independent name}.
_MOVED_TO_CLI_INDEPENDENT = {
    '_run_independent': 'run_independent',
}

# Old names of what cli_concatenate now holds, deprecated and served by
# __getattr__: {old cli name: new cli_concatenate name}.
_MOVED_TO_CLI_CONCATENATE = {
    '_run_concatenated': 'run_concatenated',
}

# Old names of what cli_cytonorm now holds, deprecated and served by
# __getattr__: {old cli name: new cli_cytonorm name}.
_MOVED_TO_CLI_CYTONORM = {
    '_fit_cytonorm': 'fit_cytonorm',
}

# Old names of what cli_run now holds, deprecated and served by
# __getattr__: {old cli name: new cli_run name}.
_MOVED_TO_CLI_RUN = {
    '_gpu_present': 'gpu_present',
}

# Old names of what cli_panel now holds, deprecated and served by
# __getattr__: {old cli name: new cli_panel name}.
_MOVED_TO_CLI_PANEL = {
    '_norm_token': 'norm_token',
}

# Old names of what cli_main now holds, deprecated and served by
# __getattr__: {old cli name: new cli_main name}.
_MOVED_TO_CLI_MAIN = {
    '_write_tables_workbook': 'write_tables_workbook',
}


def __getattr__(name):
    # The old name _safe_filename is deprecated and served here.
    if name == '_safe_filename':
        warnings.warn('_safe_filename is deprecated and will be removed in a '
                      'future release; use output_name', DeprecationWarning,
                      stacklevel=2)
        return output_name
    if name in _MOVED_TO_SHARED_CLUSTERING:
        new = _MOVED_TO_SHARED_CLUSTERING[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use shared_clustering.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import shared_clustering
        return getattr(shared_clustering, new)
    if name in _MOVED_TO_CLI_GATES:
        new = _MOVED_TO_CLI_GATES[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_gates.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_gates
        return getattr(cli_gates, new)
    if name in _MOVED_TO_CLI_UNMIX:
        new = _MOVED_TO_CLI_UNMIX[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_unmix.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_unmix
        return getattr(cli_unmix, new)
    if name in _MOVED_TO_CLI_SAMPLES:
        new = _MOVED_TO_CLI_SAMPLES[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_samples.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_samples
        return getattr(cli_samples, new)
    if name in _MOVED_TO_CLI_WSP:
        new = _MOVED_TO_CLI_WSP[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_wsp.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_wsp
        return getattr(cli_wsp, new)
    if name in _MOVED_TO_CLI_MANIFEST:
        new = _MOVED_TO_CLI_MANIFEST[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_manifest.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_manifest
        return getattr(cli_manifest, new)
    if name in _MOVED_TO_CLI_GROUPS:
        new = _MOVED_TO_CLI_GROUPS[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_groups.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_groups
        return getattr(cli_groups, new)
    if name in _MOVED_TO_CLI_PLOTS:
        new = _MOVED_TO_CLI_PLOTS[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_plots.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_plots
        return getattr(cli_plots, new)
    if name in _MOVED_TO_CLI_PARALLEL:
        new = _MOVED_TO_CLI_PARALLEL[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_parallel.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_parallel
        return getattr(cli_parallel, new)
    if name in _MOVED_TO_CLI_TRIALS:
        new = _MOVED_TO_CLI_TRIALS[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_trials.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_trials
        return getattr(cli_trials, new)
    if name in _MOVED_TO_CLI_COMPARE:
        new = _MOVED_TO_CLI_COMPARE[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_compare.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_compare
        return getattr(cli_compare, new)
    if name in _MOVED_TO_CLI_INDEPENDENT:
        new = _MOVED_TO_CLI_INDEPENDENT[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_independent.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_independent
        return getattr(cli_independent, new)
    if name in _MOVED_TO_CLI_CONCATENATE:
        new = _MOVED_TO_CLI_CONCATENATE[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_concatenate.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_concatenate
        return getattr(cli_concatenate, new)
    if name in _MOVED_TO_CLI_CYTONORM:
        new = _MOVED_TO_CLI_CYTONORM[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_cytonorm.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_cytonorm
        return getattr(cli_cytonorm, new)
    if name in _MOVED_TO_CLI_RUN:
        new = _MOVED_TO_CLI_RUN[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_run.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_run
        return getattr(cli_run, new)
    if name in _MOVED_TO_CLI_PANEL:
        new = _MOVED_TO_CLI_PANEL[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_panel.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_panel
        return getattr(cli_panel, new)
    if name in _MOVED_TO_CLI_MAIN:
        new = _MOVED_TO_CLI_MAIN[name]
        warnings.warn(f'cli.{name} is deprecated and will be removed in a '
                      f'future release; use cli_main.{new}',
                      DeprecationWarning, stacklevel=2)
        from . import cli_main
        return getattr(cli_main, new)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_check_sessions(folder, out=None) -> int:
    """``openflo-run check-sessions <folder>``: for each .flowsession under
    ``folder``, the fixes (fixes.py) that would apply to its saved results
    when it is next opened. Reads only. Returns 0, or 2 when ``folder`` is
    not a folder."""
    from . import fixes
    from .result_stamp import app_version
    out = out or sys.stdout
    if not os.path.isdir(folder):
        print(f"[!] check-sessions: {folder} is not a folder", file=out)
        return 2
    paths = fixes.find_sessions(folder)
    print(f"OpenFlo {app_version()}: {len(paths)} session(s) under "
          f"{os.path.basename(os.path.abspath(folder)) or folder}", file=out)
    total = 0
    for p in paths:
        rel = os.path.relpath(p, folder)
        try:
            targets = fixes.check_session_file(p)
        except Exception as exc:                          # noqa: BLE001 - a session that will not read is reported on its own line; the other sessions are still checked
            print(f"{rel}: can't read ({exc})", file=out)
            continue
        if not targets:
            print(f"{rel}: nothing to fix", file=out)
            continue
        ids = sorted({t['fix'] for t in targets})
        print(f"{rel}: {len(targets)} result(s), fix(es) {', '.join(ids)}",
              file=out)
        for t in targets:
            print(f"  {fixes.describe_target(t)}", file=out)
        total += len(targets)
    print(f"{total} result(s) to review; nothing was changed.", file=out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
