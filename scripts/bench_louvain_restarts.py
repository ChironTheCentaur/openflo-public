#!/usr/bin/env python
"""Regenerate the numbers quoted in ``FlowSample.run_louvain``'s docstring.

Docstring measurements get quoted, compared against, and used to justify
defaults, so they have to be reproducible from the information given. The
previous table was not: it stated conditions without the dimensionality, and
none of its figures could be reproduced by running the shipped function —
they appear to have come from a prototype that partitioned PhenoGraph's graph
rather than the one ``_snn_jaccard_graph`` builds.

Reports, for a sweep of restart counts:

    time        wall clock for the whole run_louvain call
    ARI         agreement with the PLANTED labels — the comparable measure
    clusters    how many communities came out (8 were planted)

and the same for the PhenoGraph binary it is compared against.

Modularity is reported per method but NOT compared between them. Q is a
property of a graph; the two partition different graphs, so their Q values are
not commensurable. That comparison was the previous table's headline claim.

Run:  python scripts/bench_louvain_restarts.py [--events N] [--channels D]
                                               [--k K] [--restarts 1,5,20]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')


def make_data(n, d, seed=0):
    """8 overlapping populations of unequal size — the shape where community
    detection methods actually differ."""
    rng = np.random.RandomState(seed)
    centres = rng.uniform(3.5, 6.5, size=(8, d))
    which = rng.choice(8, n, p=rng.dirichlet(np.ones(8) * 0.6))
    return (centres[which] + rng.normal(0, 1.4, (n, d))).astype(np.float64), which


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--events', type=int, default=20_000)
    ap.add_argument('--channels', type=int, default=10)
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--restarts', default='1,5,20')
    args = ap.parse_args()

    from sklearn.metrics import adjusted_rand_score as ari

    from openflo.pipeline import FlowSample, _snn_jaccard_graph

    X, truth = make_data(args.events, args.channels)
    ch = [f'M{i}' for i in range(args.channels)]
    df = pd.DataFrame(X, columns=ch)
    print(f'  {args.events:,} events x {args.channels} channels, '
          f'8 overlapping populations, k={args.k}, {os.cpu_count()} cores\n')

    # Built once and reused: modularity is measured on OUR graph, and
    # rebuilding it per row cost more than the measurements themselves.
    graph = _snn_jaccard_graph(X, args.k)
    print(f'  our graph: {graph.vcount():,} nodes, {graph.ecount():,} edges\n')

    print(f'  {"restarts":>9} {"time":>9} {"ARI":>8} {"Q(ours)":>9} clusters')
    for r in [int(x) for x in args.restarts.split(',')]:
        s = FlowSample.from_dataframe(df.copy(), name='bench')
        s.fluor_channels = ch
        t0 = time.perf_counter()
        s.run_louvain(channels=ch, n_neighbors=args.k, restarts=r,
                      max_events=args.events, random_state=42)
        dt = time.perf_counter() - t0
        lab = s.data['louvain'].to_numpy()
        q = float(graph.modularity(lab.tolist(), weights='weight'))
        print(f'  {r:>9} {dt:8.1f}s {ari(truth, lab):8.4f} {q:9.4f} '
              f'{len(np.unique(lab[lab >= 0])):>8}')

    # Reproducibility is the whole point of the function; assert it, do not
    # assume it.
    a = FlowSample.from_dataframe(df.copy(), name='a'); a.fluor_channels = ch
    b = FlowSample.from_dataframe(df.copy(), name='b'); b.fluor_channels = ch
    for smp in (a, b):
        smp.run_louvain(channels=ch, n_neighbors=args.k, restarts=5,
                        max_events=args.events, random_state=42)
    same = np.array_equal(a.data['louvain'].to_numpy(),
                          b.data['louvain'].to_numpy())
    print(f'\n  identical across two runs at the same seed: {same}')

    try:
        import phenograph
    except ImportError:
        print('  (phenograph not installed — skipping the comparison)')
        return
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        comm, _g, q_pg = phenograph.cluster(X, k=args.k, n_jobs=1)
    dt = time.perf_counter() - t0
    comm = np.asarray(comm)
    print(f'  PhenoGraph binary: {dt:.1f}s  ARI {ari(truth, comm):.4f}  '
          f'Q(its own graph) {q_pg:.4f}  '
          f'{len(np.unique(comm))} clusters, NOT reproducible')
    print('  (the two Q values are on different graphs and are not comparable)')


if __name__ == '__main__':
    # PhenoGraph's cluster() always spawns a Pool. On Windows its children
    # re-import this file, and without the guard they re-run the whole
    # benchmark inside every worker.
    multiprocessing.freeze_support()
    main()
