"""Cross-sample QC and single-cell-ecosystem interop.

  * **Sample-similarity** — a pairwise distance between samples (mean over
    markers of the 1-D Wasserstein / Earth-Mover's distance, each marker scaled
    by its pooled spread), plus an MDS embedding so batch effects and outlier
    samples show up at a glance.
  * **AnnData export** — bundle the concatenated events (× markers) with
    ``obs`` (sample + any label columns) and ``var`` (markers) into an
    ``.h5ad`` for the scanpy / single-cell Python ecosystem. The ``anndata``
    package is an optional dependency, imported lazily.

Distance + MDS are pure (scipy / sklearn, already dependencies).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _matrix(df, markers, max_events, rng):
    cols = [m for m in markers if m in df.columns]
    X = df[cols].to_numpy(dtype=float)
    if max_events and len(X) > max_events:
        X = X[rng.choice(len(X), max_events, replace=False)]
    return X


def sample_distance_matrix(samples, markers, max_events=20_000, seed=42):
    """Pairwise distance between samples.

    ``samples`` : ``{name: DataFrame}`` (each carrying ``markers`` columns).
    The distance is the mean over markers of the 1-D Wasserstein (EMD) distance
    between the two samples' marker distributions, each marker divided by its
    pooled standard deviation so markers on different scales contribute
    comparably. Returns ``(names, D)`` with ``D`` an ``(n, n)`` symmetric
    matrix, zero diagonal."""
    from scipy.stats import wasserstein_distance
    names = list(samples)
    rng = np.random.default_rng(seed)
    arrs = {nm: _matrix(samples[nm], markers, max_events, rng) for nm in names}
    present = [m for m in markers
               if all(m in samples[nm].columns for nm in names)]
    if not present or len(names) < 2:
        return names, np.zeros((len(names), len(names)))
    pooled = np.vstack([arrs[nm] for nm in names])
    sd = np.nanstd(pooled, axis=0)
    sd[sd == 0] = 1.0
    cols = {nm: [m in samples[nm].columns for m in markers] for nm in names}
    idx = {m: k for k, m in enumerate(markers)}
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d, used = 0.0, 0
            for m in present:
                ci = sum(cols[names[i]][:idx[m]])
                cj = sum(cols[names[j]][:idx[m]])
                a = arrs[names[i]][:, ci]
                b = arrs[names[j]][:, cj]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size and b.size:
                    d += float(wasserstein_distance(a, b)) / sd[idx[m]]
                    used += 1
            D[i, j] = D[j, i] = d / used if used else 0.0
    return names, D


def mds_embed(D, seed=42):
    """2-D MDS embedding of a precomputed distance matrix. Returns
    ``(n, 2)`` coordinates (zeros when there are fewer than 2 samples)."""
    D = np.asarray(D, dtype=float)
    if len(D) < 2:
        return np.zeros((len(D), 2))
    from sklearn.manifold import MDS
    m = MDS(n_components=2, dissimilarity='precomputed', random_state=seed,
            normalized_stress='auto')
    return np.asarray(m.fit_transform(D), dtype=float)


def to_anndata(samples, markers, obs_cols=None, max_events=None, seed=42):
    """Build an ``AnnData`` of concatenated events × markers.

    ``obs`` carries a ``sample`` column plus any ``obs_cols`` (e.g.
    ``['leiden', 'cluster']``) present in a sample. Requires the optional
    ``anndata`` package (``pip install anndata``)."""
    try:
        import anndata as ad
    except ImportError as e:
        raise ImportError(
            "AnnData export needs the 'anndata' package — pip install anndata "
            "(or: pip install 'openflo[interop]')") from e
    rng = np.random.default_rng(seed)
    obs_cols = list(obs_cols or [])
    Xs, obs = [], []
    for nm, df in samples.items():
        cols = [m for m in markers if m in df.columns]
        sub = df
        if max_events and len(df) > max_events:
            sub = df.iloc[rng.choice(len(df), max_events, replace=False)]
        Xs.append(sub[cols].to_numpy(dtype=float))
        o = pd.DataFrame({'sample': [nm] * len(sub)})
        for c in obs_cols:
            if c in sub.columns:
                o[c] = sub[c].to_numpy()
        obs.append(o)
    X = np.vstack(Xs) if Xs else np.zeros((0, len(markers)))
    obs_df = pd.concat(obs, ignore_index=True) if obs else pd.DataFrame()
    obs_df.index = obs_df.index.astype(str)
    var = pd.DataFrame(index=pd.Index([str(m) for m in markers]))
    return ad.AnnData(X=X, obs=obs_df, var=var)


def write_h5ad(path, samples, markers, obs_cols=None, max_events=None):
    """Write the samples to an ``.h5ad`` file (see :func:`to_anndata`).
    Returns the number of events written."""
    adata = to_anndata(samples, markers, obs_cols=obs_cols,
                       max_events=max_events)
    adata.write_h5ad(path)
    return int(adata.n_obs)
