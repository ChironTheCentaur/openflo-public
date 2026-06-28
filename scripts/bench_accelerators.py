#!/usr/bin/env python
"""Side-by-side accelerator benchmark for OpenFlo's heavy numeric paths.

Run this on a machine with the NVIDIA GPU to decide whether adding **CuPy**
(GPU array math, works on native Windows) is worth it, and how it compares to
the **RAPIDS** path OpenFlo already ships (cuML/cuGraph — Linux/WSL2 only, so it
is dormant on native Windows).

The two accelerate DIFFERENT stages, so the script reports both:

  1. LOAD MATH  — per-sample compensation (spillover matmul) + transform
     (arcsinh) + a QC reduction. This is the load pipeline that froze the GUI.
     CuPy can accelerate it; RAPIDS does NOT touch it.
       backends: numpy (CPU)  vs  cupy (GPU)
       also reports host<->device TRANSFER time — the per-sample overhead that
       decides whether GPU offload is a net win.

  2. UMAP       — the embedding RAPIDS currently accelerates.
       backends: umap-learn (CPU)  vs  cuml.UMAP (GPU, RAPIDS)

Missing backends are skipped with a note (so it runs anywhere and tells you
exactly what is / isn't available on this box).

Usage:
    python scripts/bench_accelerators.py                  # defaults
    python scripts/bench_accelerators.py --rows 2000000 --channels 24
    python scripts/bench_accelerators.py --umap-rows 100000 --repeats 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time


def _time(fn, repeats, warmup=1, sync=None):
    """Return (median_seconds, runs). Runs `warmup` discarded passes first;
    `sync()` (if given) is called inside the timed region so async GPU work is
    actually waited on."""
    for _ in range(max(0, warmup)):
        fn()
        if sync:
            sync()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs), runs


def _fmt(label, secs, base=None):
    s = f"  {label:<34} {secs * 1e3:9.1f} ms"
    if base and secs > 0:
        s += f"   ({base / secs:5.2f}x vs CPU)"
    return s


def bench_load_math(rows, channels, repeats):
    """Compensation (X @ inv(S)) + arcsinh transform + a QC column reduction —
    the per-sample load math, on numpy (CPU) vs cupy (GPU)."""
    import numpy as np
    print(f"\n== LOAD MATH  (compensation + arcsinh + QC) — "
          f"{rows:,} events x {channels} channels, float32 ==")

    rng = np.random.default_rng(0)
    X_host = rng.random((rows, channels), dtype=np.float32) * 1e5
    S = np.eye(channels, dtype=np.float32) + rng.random(
        (channels, channels), dtype=np.float32) * 0.05
    Sinv_host = np.linalg.inv(S).astype(np.float32)

    def work(xp, X, Sinv):
        Xc = X @ Sinv                      # compensation
        Xt = xp.arcsinh(Xc / 150.0)        # asinh transform (CuPy-able)
        # a couple of QC-style reductions (finite mask + per-channel quantile)
        _ = xp.isfinite(Xt).all(axis=1)
        _ = xp.quantile(Xt, 0.99, axis=0)
        return Xt

    # CPU
    cpu, _ = _time(lambda: work(np, X_host, Sinv_host), repeats)
    print(_fmt("numpy (CPU)", cpu))

    # GPU (CuPy)
    try:
        import cupy as cp
    except Exception as exc:                       # noqa: BLE001
        print(f"  cupy (GPU)                         — not available ({exc})")
        return

    sync = cp.cuda.Stream.null.synchronize
    X_dev = cp.asarray(X_host)
    Sinv_dev = cp.asarray(Sinv_host)
    sync()

    # compute-only (data already on GPU)
    gpu, _ = _time(lambda: work(cp, X_dev, Sinv_dev), repeats, sync=sync)
    print(_fmt("cupy (GPU, compute only)", gpu, base=cpu))

    # end-to-end incl. host->device upload + device->host download of the result
    def roundtrip():
        xd = cp.asarray(X_host)
        out = work(cp, xd, Sinv_dev)
        return cp.asnumpy(out)
    e2e, _ = _time(roundtrip, repeats, sync=sync)
    print(_fmt("cupy (GPU, incl. H<->D transfer)", e2e, base=cpu))
    print("  -> if 'incl. transfer' isn't faster than numpy, GPU offload of the\n"
          "     load math is NOT worth it at this sample size.")


def bench_clustering(rows, channels, k):
    """Clustering via OpenFlo's OWN FlowSample.cluster(): PhenoGraph (CPU) vs
    cuGraph Louvain (GPU/RAPIDS). Exercises the real code path, so the numbers
    reflect what users actually get."""
    import numpy as np
    import pandas as pd
    print(f"\n== CLUSTERING  (FlowSample.cluster: PhenoGraph CPU vs cuGraph GPU)"
          f" — {rows:,} events x {channels} channels, k={k} ==")
    try:
        from openflo.pipeline import GPU_CLUSTERING_AVAILABLE, FlowSample
    except Exception as exc:                       # noqa: BLE001
        print(f"  openflo.pipeline import failed — {exc}")
        return
    # Gaussian blobs so there's real structure to find.
    rng = np.random.default_rng(0)
    centers = rng.random((6, channels)) * 12.0
    lbl = rng.integers(0, 6, rows)
    X = centers[lbl] + rng.standard_normal((rows, channels))
    cols = [f'M{i}' for i in range(channels)]
    df = pd.DataFrame(X.astype(np.float32), columns=cols)

    def run(use_gpu):
        s = FlowSample.from_dataframe(df.copy(), name='bench')
        s.cluster(channels=cols, k=k, use_gpu=use_gpu, max_events=rows)
        return s

    cpu = None
    try:
        cpu, _ = _time(lambda: run(False), repeats=1, warmup=0)
        print(_fmt("PhenoGraph (CPU)", cpu))
    except Exception as exc:                       # noqa: BLE001
        print(f"  PhenoGraph (CPU)                   — failed ({exc})")

    if not GPU_CLUSTERING_AVAILABLE:
        print("  cuGraph Louvain (GPU/RAPIDS)       — not available")
        print("  (RAPIDS cuML+cuGraph ships Linux/WSL2 only — run this script "
              "inside WSL2 with RAPIDS installed to time the GPU path.)")
        return
    try:
        import cupy as cp
        sync = cp.cuda.Stream.null.synchronize
    except Exception:
        sync = None
    g, _ = _time(lambda: run(True), repeats=1, warmup=1, sync=sync)
    print(_fmt("cuGraph Louvain (GPU/RAPIDS)", g, base=cpu))


def bench_umap(rows, channels, repeats):
    """UMAP embedding: umap-learn (CPU) vs cuML (RAPIDS GPU)."""
    import numpy as np
    print(f"\n== UMAP  — {rows:,} events x {channels} channels ==")
    rng = np.random.default_rng(0)
    X = rng.random((rows, channels), dtype=np.float32)

    # CPU (umap-learn)
    cpu = None
    try:
        from umap import UMAP
        cpu, _ = _time(lambda: UMAP(n_neighbors=15, n_components=2,
                                    random_state=42).fit_transform(X),
                       repeats=1, warmup=0)
        print(_fmt("umap-learn (CPU)", cpu))
    except Exception as exc:                       # noqa: BLE001
        print(f"  umap-learn (CPU)                   — not available ({exc})")

    # GPU (cuML / RAPIDS)
    try:
        from cuml.manifold import UMAP as cuUMAP
    except Exception as exc:                       # noqa: BLE001
        print(f"  cuml.UMAP (GPU/RAPIDS)             — not available ({exc})")
        print("  (RAPIDS ships Linux/WSL2 only — expected to be absent on "
              "native Windows.)")
        return
    try:
        import cupy as cp
        sync = cp.cuda.Stream.null.synchronize
    except Exception:
        sync = None
    g, _ = _time(lambda: cuUMAP(n_neighbors=15, n_components=2).fit_transform(X),
                 repeats=1, warmup=1, sync=sync)
    print(_fmt("cuml.UMAP (GPU/RAPIDS)", g, base=cpu))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rows', type=int, default=2_000_000,
                    help='events for the load-math benchmark (default 2,000,000)')
    ap.add_argument('--channels', type=int, default=20)
    ap.add_argument('--umap-rows', type=int, default=50_000,
                    help='events for the UMAP benchmark (default 50,000)')
    ap.add_argument('--cluster-rows', type=int, default=50_000,
                    help='events for the clustering benchmark (default 50,000)')
    ap.add_argument('--k', type=int, default=30, help='neighbours k for clustering')
    ap.add_argument('--repeats', type=int, default=5,
                    help='timed passes for the load-math benchmark (median reported)')
    ap.add_argument('--skip-umap', action='store_true')
    ap.add_argument('--skip-cluster', action='store_true')
    ap.add_argument('--skip-load', action='store_true')
    args = ap.parse_args(argv)

    print("OpenFlo accelerator side-by-side")
    print(f"python {sys.version.split()[0]}")
    try:
        import numpy as np
        print(f"numpy {np.__version__}")
    except Exception:
        print("numpy not installed — cannot benchmark.")
        return 1
    for name in ('cupy', 'cuml'):
        try:
            mod = __import__(name)
            print(f"{name} {getattr(mod, '__version__', '?')}  AVAILABLE")
        except Exception:
            print(f"{name}  not available")

    if not args.skip_load:
        bench_load_math(args.rows, args.channels, args.repeats)
    if not args.skip_cluster:
        bench_clustering(args.cluster_rows, args.channels, args.k)
    if not args.skip_umap:
        bench_umap(args.umap_rows, args.channels, args.repeats)

    print("\nVerdict guide:")
    print("  CuPy and RAPIDS accelerate DIFFERENT stages — not either/or:")
    print("  * LOAD MATH    -> CuPy (native Windows + Linux). Already wired in,")
    print("    opt-in via Preferences.")
    print("  * UMAP/CLUSTER -> RAPIDS cuML/cuGraph (Linux/WSL2 only). Already in")
    print("    pipeline.cluster()/run_umap(), auto-used when present.")
    print("  Run this in WSL2 with RAPIDS installed to fill the GPU rows for")
    print("  clustering + UMAP and compare RAPIDS vs the CPU 'universal' path.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
