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

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Cluster-like obs columns that carry the -1 noise sentinel — a labelled sibling
# is added on AnnData export so scanpy users see "Unclustered (noise)" without
# losing the raw integer id.
_CLUSTER_OBS_COLS = ('cluster', 'leiden', 'flowsom_meta')


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
    present = [m for m in markers
               if all(m in samples[nm].columns for nm in names)]
    dropped = [m for m in markers if m not in present]
    if dropped:
        log.warning("sample_distance_matrix: %d marker(s) omitted (missing from "
                    "some samples): %s", len(dropped),
                    ', '.join(map(str, dropped)))
    if not present or len(names) < 2:
        D = np.zeros((len(names), len(names)))
        if not present and len(names) >= 2:
            # ≥2 samples but NO shared markers: the distance is UNDEFINED, not
            # zero — NaN off-diagonal so it can't masquerade as "identical
            # samples" (and collapse every point onto the MDS origin).
            D[:] = np.nan
            np.fill_diagonal(D, 0.0)
        return names, D
    # Restrict EVERY sample to the shared markers so the pooled std and the
    # per-marker column index are over one consistent column set. Selecting
    # per-sample present columns instead would give ragged widths (np.vstack
    # fails) and mis-index `sd` when a marker is missing from some samples.
    rng = np.random.default_rng(seed)
    arrs = {nm: _matrix(samples[nm], present, max_events, rng) for nm in names}
    pooled = np.vstack([arrs[nm] for nm in names])
    sd = np.nanstd(pooled, axis=0)
    sd[sd == 0] = 1.0
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d, used = 0.0, 0
            for k in range(len(present)):
                a = arrs[names[i]][:, k]
                b = arrs[names[j]][:, k]
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                if a.size and b.size:
                    d += float(wasserstein_distance(a, b)) / sd[k]
                    used += 1
            # A pair with no usable (finite) marker overlap is UNDEFINED → NaN,
            # not 0.0 (which would read as "identical").
            D[i, j] = D[j, i] = d / used if used else np.nan
    return names, D


def mds_embed(D, seed=42):
    """2-D MDS embedding of a precomputed distance matrix. Returns
    ``(n, 2)`` coordinates (zeros when there are fewer than 2 samples)."""
    D = np.asarray(D, dtype=float).copy()
    if len(D) < 2:
        return np.zeros((len(D), 2))
    # Undefined pairs (NaN — no shared/usable markers) can't go into MDS. Fill
    # them with the LARGEST finite distance so those samples read as maximally
    # dissimilar (pushed to the edge) rather than collapsed onto the origin.
    if not np.all(np.isfinite(D)):
        finite = D[np.isfinite(D)]
        fill = float(finite.max()) if finite.size else 1.0
        if not fill > 0:               # e.g. 2 samples, only the zero diagonal
            fill = 1.0                  # → push apart, don't collapse to origin
        D[~np.isfinite(D)] = fill
        np.fill_diagonal(D, 0.0)
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
    # Use the markers shared by ALL samples so every block of X has the same
    # width (ragged per-sample columns would break np.vstack) and matches `var`.
    present = [m for m in markers
               if all(m in df.columns for df in samples.values())] \
        if samples else list(markers)
    dropped = [m for m in markers if m not in present]
    if dropped:
        log.warning("to_anndata: %d marker(s) omitted (missing from some "
                    "samples): %s", len(dropped), ', '.join(map(str, dropped)))
    Xs, obs = [], []
    for nm, df in samples.items():
        sub = df
        if max_events and len(df) > max_events:
            sub = df.iloc[rng.choice(len(df), max_events, replace=False)]
        Xs.append(sub[present].to_numpy(dtype=float))
        o = pd.DataFrame({'sample': [nm] * len(sub)})
        for c in obs_cols:
            if c in sub.columns:
                o[c] = sub[c].to_numpy()
        obs.append(o)
    X = np.vstack(Xs) if Xs else np.zeros((0, len(present)))
    obs_df = pd.concat(obs, ignore_index=True) if obs else pd.DataFrame()
    obs_df.index = obs_df.index.astype(str)
    # Add a labelled sibling for cluster-like obs columns so the -1 noise
    # sentinel reads "Unclustered (noise)" for scanpy users, while the raw
    # integer id is preserved for their own joins/plots.
    for c in obs_cols:
        if c in obs_df.columns and c in _CLUSTER_OBS_COLS:
            from .pipeline import cluster_label
            obs_df[f'{c}_label'] = [cluster_label(v) for v in obs_df[c]]
    var = pd.DataFrame(index=pd.Index([str(m) for m in present]))
    adata = ad.AnnData(X=X, obs=obs_df, var=var)
    if dropped:
        adata.uns['dropped_markers'] = list(dropped)
    return adata


def _h5ad_safe_strings(adata):
    """Coerce pandas string-backed columns/indices to plain object dtype.

    pandas 3.x gives string columns a ``StringArray`` / ``ArrowStringArray``
    backing, and anndata refuses to write those unless
    ``settings.allow_write_nullable_strings`` is opted into — so
    ``write_h5ad`` raised ``RuntimeError`` on the obs index alone, making the
    whole ``.h5ad`` export unusable. Converting to object dtype produces the
    same on-disk representation anndata has always written, and does not
    depend on a global setting or an anndata version.
    """
    import pandas as _pd

    def _fix(frame):
        if frame is None:
            return
        frame.index = _pd.Index(
            [str(v) for v in frame.index], dtype=object, name=frame.index.name)
        for col in frame.columns:
            s = frame[col]
            if _pd.api.types.is_string_dtype(s) and s.dtype != object:
                frame[col] = s.astype(object)

    _fix(getattr(adata, 'obs', None))
    _fix(getattr(adata, 'var', None))
    return adata


def write_h5ad(path, samples, markers, obs_cols=None, max_events=None):
    """Write the samples to an ``.h5ad`` file (see :func:`to_anndata`).
    Returns the number of events written."""
    adata = to_anndata(samples, markers, obs_cols=obs_cols,
                       max_events=max_events)
    _h5ad_safe_strings(adata)
    adata.write_h5ad(path)
    return int(adata.n_obs)
