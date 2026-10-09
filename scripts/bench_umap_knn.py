#!/usr/bin/env python
"""Re-measure the UMAP parallel-kNN change: speed AND embedding quality.

``umap-learn`` forces ``n_jobs=1`` whenever ``random_state`` is set, because
its layout optimisation is a lock-free parallel SGD — threads race on the
shared embedding array, and thread interleaving is not an RNG, so no seed can
pin it. We want the determinism, so we take the override.

But UMAP is two stages and only the second is unseedable:

    [ approximate kNN search ]  ->  [ layout optimisation ]
      parallel AND seedable            lock-free, unseedable

The override applies to both, so the neighbour search is needlessly serial.
``run_umap`` now builds it separately with every core and hands it in via
``precomputed_knn``.

The resulting embedding is NOT identical to the serial one. That is expected —
UMAP layouts are not unique — but it means speed alone cannot justify the
change: a faster route to a worse answer is not an improvement. So this scores
both embeddings against the SOURCE DATA and the planted labels, never against
each other:

    trustworthiness    do points that are neighbours in the embedding
                       deserve to be? (penalises false neighbours)
    kNN preservation   of each point's true high-D neighbours, how many
                       survive the projection? (penalises lost structure)
    label accuracy     can a kNN classifier recover the planted population
                       from the 2-D coordinates alone?
    silhouette         are the planted populations separated?

Run:  python scripts/bench_umap_knn.py [--events N] [--channels D] [--k K]
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('MPLBACKEND', 'Agg')

SEED = 42


def make_data(n, d, rng_seed=0):
    """Overlapping populations of unequal size — the shape where two
    embeddings are actually distinguishable. Well-separated blobs would score
    identically no matter what, and prove nothing."""
    rng = np.random.RandomState(rng_seed)
    centres = rng.uniform(3.5, 6.5, size=(8, d))
    which = rng.choice(8, n, p=rng.dirichlet(np.ones(8) * 0.6))
    return (centres[which] + rng.normal(0, 1.4, (n, d))).astype(np.float32), which


def timed(fn):
    t0 = time.perf_counter()
    return fn(), time.perf_counter() - t0


def serial(X, k):
    import umap
    return np.asarray(umap.UMAP(n_neighbors=k, min_dist=0.3,
                                random_state=SEED).fit_transform(X))


def parallel_knn(X, k):
    import umap
    from pynndescent import NNDescent
    index = NNDescent(X, n_neighbors=k, metric='euclidean',
                      random_state=SEED, n_jobs=-1, low_memory=True)
    idx, dist = index.neighbor_graph
    return np.asarray(umap.UMAP(n_neighbors=k, min_dist=0.3,
                                random_state=SEED,
                                precomputed_knn=(idx, dist, index)
                                ).fit_transform(X))


def knn_preservation(X, emb, k=15, sample=4000):
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(0)
    sel = rng.choice(len(X), min(sample, len(X)), replace=False)
    hi = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X[sel], return_distance=False)[:, 1:]
    lo = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(
        emb[sel], return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(hi, lo, strict=True)]))


def label_accuracy(emb, labels, k=15, sample=6000):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.RandomState(0)
    sel = rng.choice(len(emb), min(sample, len(emb)), replace=False)
    return float(cross_val_score(KNeighborsClassifier(k), emb[sel],
                                 labels[sel], cv=3).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--events', type=int, default=30_000)
    ap.add_argument('--channels', type=int, default=12)
    ap.add_argument('--k', type=int, default=30)
    args = ap.parse_args()

    from sklearn.manifold import trustworthiness
    from sklearn.metrics import silhouette_score

    X, truth = make_data(args.events, args.channels)
    print(f'{args.events:,} events x {args.channels} channels, 8 overlapping '
          f'populations, n_neighbors={args.k}, {os.cpu_count()} cores\n')

    # UMAP is numba-compiled and the FIRST call in a process pays ~30s of JIT
    # regardless of which path it takes. Timing the two paths cold makes
    # whichever ran first look terrible and inflates the speed-up to nearly
    # whatever you want it to be. Burn the JIT on a tiny array first, so what
    # is timed below is the work and not the compiler.
    # The warm-up MUST cross UMAP's ~4096-row exact/approximate kNN boundary.
    # Below it UMAP uses sklearn's exact search and never touches the
    # pynndescent kernels a real run spends its compile time on — measured, a
    # 600-row warm-up saves nothing (25k fit: 28.5s cold, 24.7s "warmed"),
    # while a 5000-row warm-up takes it to 11.9s.
    warm = X[:5000]
    serial(warm, 15)
    parallel_knn(warm, 15)
    print('  (numba warmed up — timings below exclude JIT)')
    print()

    runs = {'serial (umap builds its own kNN)': timed(lambda: serial(X, args.k)),
            'parallel kNN (precomputed_knn)': timed(lambda: parallel_knn(X, args.k))}

    rng = np.random.RandomState(0)
    sub = rng.choice(len(X), min(3000, len(X)), replace=False)  # trust is O(n^2)

    print(f'  {"embedding":<34}{"time":>8}{"trust":>8}{"kNN-pres":>10}'
          f'{"label-acc":>11}{"silhou":>9}')
    scores = {}
    for name, (emb, t) in runs.items():
        row = (trustworthiness(X[sub], emb[sub], n_neighbors=15),
               knn_preservation(X, emb),
               label_accuracy(emb, truth),
               silhouette_score(emb[sub], truth[sub]))
        scores[name] = row
        print(f'  {name:<34}{t:7.1f}s{row[0]:8.4f}{row[1]:10.4f}'
              f'{row[2]:11.4f}{row[3]:9.4f}')

    (a_emb, a_t), (b_emb, b_t) = runs.values()
    a, b = scores.values()
    print(f'\n  speed: {a_t / b_t:.2f}x faster\n')

    names = ('trustworthiness', 'kNN preservation', 'label accuracy',
             'silhouette')
    worse = better = 0
    for nm, va, vb in zip(names, a, b, strict=True):
        d = vb - va
        worse += d < -0.005
        better += d > 0.005
        tag = 'BETTER' if d > 0.005 else ('WORSE' if d < -0.005 else 'same')
        print(f'    {nm:<20} serial={va:.4f}  parallel={vb:.4f}  '
              f'({d:+.4f})  {tag}')

    print()
    if worse:
        print(f'  VERDICT: parallel kNN is WORSE on {worse}/4 — the speed-up '
              f'costs accuracy and should not be kept.')
    elif better:
        print(f'  VERDICT: parallel kNN is BETTER on {better}/4 and faster.')
    else:
        print('  VERDICT: equivalent quality — the speed-up is free; the two '
              'layouts simply differ, as UMAP layouts do.')


if __name__ == '__main__':
    main()
