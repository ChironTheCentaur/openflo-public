"""openflo-run's per-sample work: load, QC, compensate, transform and gate
each output sample's FCS files, cluster it, and write its plots and stats,
one sample at a time or on parallel worker processes with RAM admission
control.

Moved out of cli.py unchanged; the old cli names still work, deprecated.
"""
from __future__ import annotations

import copy
import logging
import os

from ._console import start_worker, worker_settings
from .cli_args import DEFAULT_DPI
from .cli_groups import output_name
from .cli_plots import save_plots
from .label_alignment import concatenate
from .pipeline import FlowSample
from .shared_clustering import cluster_by_method

# The logger a sample's "Stats →" line goes to: openflo-run has always shown
# it as openflo.cli's.
LOGGER_NAME = 'openflo.cli'


def load_task_sample(task):
    """Load every input FCS for one output sample, QC it (with --qc),
    compensate, transform and gate it, and concatenate the files when there is more than one
    (tagging each cell with its trial-of-origin). Does NOT cluster. Returns
    the FlowSample, or None when the task has no input file.

    Separate from the clustering so --skip-existing can reload a finished
    sample's events for the group comparison without clustering it again."""
    name = task['name']
    pieces = []
    for origin, path in task['paths']:
        s = FlowSample(path)
        if task.get('qc'):
            s.run_qc()
        # Debris + doublet filtering, applied to raw (un-transformed)
        # FSC channels. Order matters: drop debris first so the
        # doublet ratio's median is computed on cell-sized events.
        if task.get('debris_min_fsc') is not None:
            s.filter_debris(min_fsc=task['debris_min_fsc'])
        if task.get('doublet_tol'):
            s.filter_doublets(tol=float(task['doublet_tol']))
        s.auto_compensate()
        s.apply_transform()
        if task['labels']:
            s.set_labels(task['labels'])
        s.apply_threshold_gates(task['thresholds'])
        # Compound region gates (rect/polygon/interval) — filter events,
        # applied after transform so coordinates match post-logicle space.
        if task.get('region_gates'):
            s.apply_region_gates(task['region_gates'])
        s.name = origin                     # → sample_origin column
        pieces.append(s)
    if not pieces:
        return None
    combined = copy.copy(pieces[0])
    if len(pieces) > 1:
        combined.data = concatenate(pieces)
    combined.name = name      # display / filename stem
    # Batch correction (CytoNorm) — apply the pre-fitted model after
    # transform, before clustering, so clusters/UMAP/stats use the
    # batch-aligned values.
    if task.get('cytonorm'):
        from .cytonorm import CytoNorm
        combined.data = CytoNorm.from_dict(task['cytonorm']).apply(
            combined.data, task.get('batch_id', ''))
    return combined


def process_sample_task(task):
    """Top-level worker (picklable — runs in a child process).

    Loads every input FCS for one output sample, concatenates them when there
    is more than one (tagging each cell with its trial-of-origin), gates and
    clusters. Returns (key, FlowSample | None, error_msg | None, written) where
    `key` uniquely identifies the task (group + name) so the dispatcher
    can bucket results correctly even when two groups (e.g. two days)
    contain identically-named FCS files.

    A task with ``write_outputs`` also draws the sample's plots and writes
    its stats here, in the worker (write_sample_outputs), so a parallel run
    spreads that work over the workers instead of doing it one sample at a
    time in the main process while they wait; the fourth item then says the
    files are written.
    """
    name = task['name']
    key  = task.get('key', name)
    try:
        combined = load_task_sample(task)
        if combined is None:
            return (key, None, 'no input files found', False)
        cluster_by_method(combined, task)
        written = False
        if task.get('write_outputs'):
            write_sample_outputs(combined, task)
            written = True
        return (key, combined, None, written)
    except Exception as e:  # noqa: BLE001 - returned for parent to report
        return (key, None, f'{type(e).__name__}: {e}', False)


def write_sample_outputs(s, task):
    """One sample's own files: its plots (save_plots) and its per-cluster
    stats CSV, in the task's group folder."""
    group_dir = output_name(task['group'])
    file_nm   = output_name(task['name'])
    save_plots(s, os.path.join(task['out'], group_dir),
               dpi=task.get('dpi', DEFAULT_DPI))
    stats_path = os.path.join(task['out'], group_dir, f"{file_nm}_stats.csv")
    s.cluster_stats().to_csv(stats_path, index=False)
    logging.getLogger(LOGGER_NAME).info(f"  Stats → {stats_path}")


def run_sample_tasks(tasks, workers, admission_gb, report, failures=None):
    """Process sample tasks — in parallel when workers > 1, with admission
    control: a new worker is only submitted when free RAM ≥ `admission_gb`.
    In-flight workers are NEVER paused (suspending would not free RAM) — they
    complete naturally; only the *launcher* waits. Each completed sample's
    plots + stats are written immediately so --skip-existing works after a
    watchdog restart. Returns {group_name: [FlowSample, ...]} keyed by the
    'group' field on each task; `failures`, when given, gets each failed
    task's key -> error."""
    results  = {}                                  # group_name -> [samples]
    # Dispatch key MUST be unique per task. Bare sample name collides
    # when two groups (e.g. two day-folders) hold identically-named FCS
    # — that silently bucketed both into one group. Key by group+name.
    for t in tasks:
        t['key'] = f"{t['group']}\x1f{t['name']}"
    by_key   = {t['key']: t for t in tasks}

    def _handle(result):
        key, s, err, written = result
        t = by_key[key]
        nm = t['name']                              # for display + filenames
        if s is None:
            report(f"Failed: {nm} ({err})")
            if failures is not None:
                failures[key] = err
            # Detect the unmistakable Windows commit-charge / pagefile signature
            # and tell the user how to fix it. RAM might look fine in psutil
            # while commit is exhausted — the watchdog won't catch this.
            e = (err or '').lower()
            if ('paging file' in e or 'pagefile' in e
                    or 'winerror 1455' in e or 'winerror 8' in e
                    or 'not enough memory resources' in e):
                print(
                    "\n  [!] WINDOWS PAGEFILE TOO SMALL.\n"
                    "      Your physical RAM is fine but the OS commit charge\n"
                    "      ran out. Increase the pagefile:\n"
                    "        Win -> View advanced system settings\n"
                    "        Performance -> Settings -> Advanced -> Virtual memory -> Change\n"
                    "        Uncheck 'Automatically manage' -> pick C: -> Custom size\n"
                    "        Initial + Maximum >= 32768 MB (32 GB) -> Set -> OK -> restart\n",
                    flush=True)
            return
        if not written:
            write_sample_outputs(s, t)
        results.setdefault(t['group'], []).append(s)
        report(f"Processed {nm}  ({len(s.data):,} events)")

    if not tasks:
        return results

    if workers <= 1 or len(tasks) == 1:
        for t in tasks:
            _handle(process_sample_task(t))
        return results

    # Parallel path with admission control.
    import multiprocessing
    import time
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    try:
        import psutil
    except ImportError:
        psutil = None

    n = min(workers, len(tasks))
    # This module's own console lines stay ASCII. The CLI makes its streams
    # UTF-8 first (_console.force_utf8_streams), but a script that calls this
    # directly keeps its own: on Windows, output redirected to a file or pipe
    # is cp1252, which has no '>=' sign, so printing one stopped the run.
    print(f"\n-- Processing {len(tasks)} sample(s) on {n} parallel worker(s) "
          f"({tasks[0]['n_jobs']} clustering thread(s) each).", flush=True)
    if admission_gb > 0 and psutil is not None:
        print(f"   Admission: a new worker starts only when >= {admission_gb:.1f} GB RAM is free.",
              flush=True)
    else:
        print("   Admission: disabled (psutil unavailable or threshold <= 0).",
              flush=True)

    waiting     = False
    no_progress = 0       # consecutive 2 s polls with no completion

    # ProcessPoolExecutor workers are NON-daemonic, so Phenograph can still
    # spawn its own Jaccard-kernel pool inside each worker. (multiprocessing.Pool
    # workers are daemonic and would crash with "daemonic processes are not
    # allowed to have children".)
    # Fresh processes on every system, as on Windows and macOS: Linux's
    # default fork copies a process whose numba/BLAS threads may hold a
    # lock (Python 3.12 warns, 3.14 stops forking by default).
    # Spawned workers start with none of this process's console settings
    # (encoding, warning filters, logging, palette): start_worker sets them.
    with ProcessPoolExecutor(max_workers=n,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=start_worker,
                             initargs=(worker_settings(),)) as ex:
        pending   = list(tasks)
        in_flight = {}                            # future → sample name

        while pending or in_flight:
            # ── Top up workers, admission-gated ──────────────────────────
            while pending and len(in_flight) < n:
                if admission_gb > 0 and psutil is not None:
                    free_gb = psutil.virtual_memory().available / (1024 ** 3)
                else:
                    free_gb = float('inf')
                if free_gb < admission_gb:
                    if not in_flight:
                        if not waiting:
                            print(f"   [Admission] Free RAM {free_gb:.1f} GB "
                                  f"< {admission_gb:.1f} GB - waiting ...",
                                  flush=True)
                            waiting = True
                        time.sleep(3)
                        continue
                    break
                if waiting:
                    print(f"   [Admission] Free RAM {free_gb:.1f} GB >= "
                          f"{admission_gb:.1f} GB - resuming submission.",
                          flush=True)
                    waiting = False
                t = pending.pop(0)
                # Plots and stats are drawn in the worker.
                t['write_outputs'] = True
                fut = ex.submit(process_sample_task, t)
                in_flight[fut] = t['name']   # display name for status only

            if not in_flight:
                continue

            # ── Drain completions (timeout lets us re-poll RAM regularly) ─
            try:
                done, _ = wait(in_flight.keys(),
                               timeout=2.0, return_when=FIRST_COMPLETED)
            except Exception as e:  # noqa: BLE001 - pool broken: report all, stop
                # Pool broken etc. — surface and stop, don't hang silently.
                for nm in in_flight.values():
                    report(f"Failed: {nm} (pool broken: {e})")
                return results

            if done:
                no_progress = 0
                for fut in done:
                    nm = in_flight.pop(fut)
                    try:
                        _handle(fut.result())
                    except Exception as e:  # noqa: BLE001 - reported; others go on
                        report(f"Failed: {nm} ({type(e).__name__}: {e})")
            else:
                no_progress += 1
                # 90 polls × 2 s = ~3 minutes of silence → log a warning.
                # Repeats every 3 minutes so a hang can't go undetected.
                if no_progress % 90 == 0:
                    free_gb = (psutil.virtual_memory().available / (1024**3)
                               if psutil is not None else 0)
                    stuck = ', '.join(in_flight.values())
                    print(f"   [Stuck?] {no_progress*2//60} min with no "
                          f"completions. In flight: {stuck}. "
                          f"Free RAM {free_gb:.1f} GB. "
                          f"If this persists, Cancel and retry with fewer "
                          f"workers or a smaller --max-events cap.",
                          flush=True)

    return results
