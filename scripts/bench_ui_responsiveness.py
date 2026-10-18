#!/usr/bin/env python
"""Does loading freeze the window? Measure it instead of guessing.

FCS loading runs on a thread pool, and the Python glue between the numpy ops
holds the GIL. Too many workers and the Tk event loop is starved, which a user
experiences as the window going dead during a big resume. `_default_pool_size`
caps the pool at 2 for exactly this reason.

"Is it still responsive?" is not answerable by reading the code, and it is not
answerable by timing the load either — a load that finishes fast can still have
frozen the window for a second in the middle. So this measures the thing the
user actually feels: a 10 ms heartbeat is scheduled on the Tk thread, and every
tick records how late it ACTUALLY fired while the pool runs underneath it.

Interpreting the lateness numbers:

    < 50 ms    imperceptible — a dropped frame or three
    ~100 ms    a visible hitch
    > 200 ms   the window reads as frozen

Measured on 2026-09-08 (8 files, 300k events x 18 channels, 172 MB total), the
shipped 2-worker default came back at median 1.7 ms, p95 27 ms, max 59 ms, with
no tick over 100 ms — i.e. no freeze to fix. That result is why the long-parked
"process-based loader" idea was closed rather than built: it would have cost
pickling every DataFrame across a process boundary to solve a problem that does
not show up in the numbers. Re-run this before reopening that idea.

Generates its own synthetic FCS files, so it needs no data to run.

Usage:
    python scripts/bench_ui_responsiveness.py                # sweep 1,2,4,8
    python scripts/bench_ui_responsiveness.py --workers 2
    python scripts/bench_ui_responsiveness.py --files 16 --events 500000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
warnings.filterwarnings('ignore')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The loader narrates each file ("No spillover in FCS metadata for ..."), which
# would bury the table. Errors still surface.
logging.getLogger('openflo').setLevel(logging.ERROR)

CHANNELS = ['FSC-A', 'FSC-H', 'SSC-A', 'Time'] + [f'CD{i}-A' for i in range(14)]


def make_fcs(directory, n_files, n_events):
    """Write synthetic FCS files and return their paths."""
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n_files):
        frame = pd.DataFrame(
            {c: np.abs(rng.normal(5000.0, 1500.0, n_events)) for c in CHANNELS})
        frame['Time'] = np.linspace(0.0, 300.0, n_events)
        path = os.path.join(directory, f'sample{i:02d}.fcs')
        write_fcs(frame, path, {c: c for c in CHANNELS})
        paths.append(path)
    return paths


def load_one(path):
    """Exactly what `LoadPoolMixin._load_worker` does per sample."""
    from openflo.pipeline import FlowSample
    sample = FlowSample(path)
    sample.run_qc()
    sample.auto_compensate()
    sample.apply_transform()
    return sample


def measure(paths, workers):
    """Load `paths` on `workers` threads; return (wall_seconds, lateness_ms)."""
    import tkinter as tk
    from concurrent.futures import ThreadPoolExecutor

    root = tk.Tk()
    root.withdraw()
    lateness = []
    running = True
    pending = [None]

    def tick(expected):
        lateness.append((time.perf_counter() - expected) * 1000.0)
        if running:
            pending[0] = root.after(10, tick, time.perf_counter() + 0.010)

    def run():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(load_one, paths))

    started = time.perf_counter()
    pending[0] = root.after(10, tick, started + 0.010)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    # Drive the event loop by hand: this stands in for the user's window, and
    # a real mainloop would not return until the load finished.
    while worker.is_alive():
        root.update()
        time.sleep(0.001)
    running = False
    wall = time.perf_counter() - started
    # Cancel the armed tick before tearing the interpreter down, or Tk fires it
    # against a destroyed widget and prints `invalid command name ...tick`.
    if pending[0] is not None:
        try:
            root.after_cancel(pending[0])
        except Exception:
            pass
    root.destroy()
    return wall, np.asarray(lateness)


def report(workers, wall, late, default_workers):
    mark = '  <- shipped default' if workers == default_workers else ''
    verdict = ('responsive' if late.max() < 100 else
               'visible hitch' if late.max() < 200 else 'FREEZES')
    print(f'  {workers:>2} worker(s)  wall {wall:5.1f}s   '
          f'median {np.median(late):5.1f}  p95 {np.percentile(late, 95):6.1f}  '
          f'p99 {np.percentile(late, 99):6.1f}  max {late.max():6.1f} ms   '
          f'{verdict}{mark}')
    over = int((late > 100).sum())
    if over:
        print(f'{"":14}{over} of {late.size} ticks later than 100 ms')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--workers', type=int, default=None,
                    help='pool size to test (default: sweep 1, 2, 4, 8)')
    ap.add_argument('--files', type=int, default=8)
    ap.add_argument('--events', type=int, default=300_000)
    args = ap.parse_args(argv)

    from openflo.editor_loadpool import _default_pool_size
    default_workers = _default_pool_size()

    print(f'\n  Tk responsiveness during a threaded FCS load'
          f'\n  {args.files} files x {args.events:,} events x {len(CHANNELS)} '
          f'channels   (this machine defaults to {default_workers} workers)\n')

    with tempfile.TemporaryDirectory() as tmp:
        print('  generating synthetic FCS …', flush=True)
        paths = make_fcs(tmp, args.files, args.events)
        total_mb = sum(os.path.getsize(p) for p in paths) / 1e6
        print(f'  {total_mb:.0f} MB written\n')

        sweep = ([args.workers] if args.workers
                 else [1, 2, 4, 8])
        for workers in sweep:
            wall, late = measure(paths, workers)
            report(workers, wall, late, default_workers)

    print('\n  Lateness is how long the Tk event loop went unserviced.'
          '\n  Under ~50 ms is imperceptible; over ~200 ms reads as frozen.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
