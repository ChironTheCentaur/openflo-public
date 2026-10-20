#!/usr/bin/env python
"""Regenerate the fast-graph comparison quoted in ``_snn_jaccard_graph``.

The shipped graph links every pair of points sharing at least one neighbour.
PhenoGraph and Seurat instead link only pairs where one point is in the other's
kNN list. The pruned form is far smaller and much faster to partition, but it
gives DIFFERENT clusters, so it is an option rather than the default and the
difference has to be measurable rather than asserted.

Reported for both graphs, on both Leiden and Louvain:

    ARI          agreement with the planted labels
    AMI          same question, different bias about cluster counts
    homogeneity  is each cluster ONE population, or several merged? This is
                 the one that matters in cytometry, and the one where a
                 metric like ARI can mislead: merging two genuinely
                 inseparable populations makes FEWER wrong pair decisions
                 than splitting them, so under-splitting can score well
                 while losing populations outright.
    completeness the converse — is each population kept in one cluster?

plus the ARI BETWEEN the two partitions, which says how different the answers
are rather than which is better.

Run:  python scripts/bench_snn_prune.py [noise_sd] [--events N] [--k K]

The noise argument matters: at sd 0.6 the populations are separable and the
pruned graph wins on every measure; at 1.4 they genuinely overlap and the two
disagree. A single fixture would support whichever conclusion it happened to
produce.
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import random
import time
import warnings

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import completeness_score, homogeneity_score

warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')


def make_data(n, d, spread, seed=0):
    rng = np.random.RandomState(seed)
    centres = rng.uniform(3.5, 6.5, size=(8, d))
    truth = rng.choice(8, n, p=rng.dirichlet(np.ones(8) * 0.6))
    X = (centres[truth] + rng.normal(0, spread, (n, d))).astype(np.float64)
    return X, truth


def partition(graph, method, seed=42):
    if method == 'louvain':
        import igraph
        igraph.set_random_number_generator(random.Random(seed))
        part = graph.community_multilevel(weights='weight')
    else:
        import leidenalg
        part = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=1.0, seed=seed)
    return np.asarray(part.membership)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('spread', nargs='?', type=float, default=1.4,
                    help='population noise sd: 0.6 separable, 1.4 overlapping')
    ap.add_argument('--events', type=int, default=20_000)
    ap.add_argument('--channels', type=int, default=10)
    ap.add_argument('--k', type=int, default=30)
    args = ap.parse_args()

    from openflo.pipeline import _snn_jaccard_graph

    X, truth = make_data(args.events, args.channels, args.spread)
    kind = 'separable' if args.spread < 1.0 else 'heavily overlapping'
    print(f'  {args.events:,} events x {args.channels} channels, 8 planted '
          f'populations ({kind}, noise sd={args.spread}), k={args.k}\n')

    graphs, labels = {}, {}
    for name, prune in (('default (all co-neighbour pairs)', False),
                        ('pruned  (kNN pairs only)', True)):
        t0 = time.perf_counter()
        g = _snn_jaccard_graph(X, args.k, prune=prune)
        print(f'  {name:<34} {g.ecount():>10,} edges  '
              f'build {time.perf_counter() - t0:5.1f}s')
        graphs[name] = g

    print()
    for name, g in graphs.items():
        for method in ('louvain', 'leiden'):
            t0 = time.perf_counter()
            lab = partition(g, method)
            dt = time.perf_counter() - t0
            labels[(name, method)] = lab
            print(f'  {name[:24]:<25} {method:<8} {dt:6.1f}s  '
                  f'ARI={ari(truth, lab):.4f}  AMI={ami(truth, lab):.4f}  '
                  f'homog={homogeneity_score(truth, lab):.4f}  '
                  f'compl={completeness_score(truth, lab):.4f}  '
                  f'clusters={len(set(lab.tolist()))}')

    print('\n  how DIFFERENT the two are (1.0 = the same partition):')
    for method in ('louvain', 'leiden'):
        a = labels[('default (all co-neighbour pairs)', method)]
        b = labels[('pruned  (kNN pairs only)', method)]
        print(f'    {method:<8} ARI(default, pruned) = {ari(a, b):.4f}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
