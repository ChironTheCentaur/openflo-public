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
    return np.asarray(MDS(**_mds_params(MDS, seed)).fit_transform(D),
                      dtype=float)


def _mds_params(MDS, seed):
    """Every `MDS` setting `mds_embed` relies on, spelled out so a
    scikit-learn release that changes a default can't move the map: 1.9
    takes `n_init` from 4 to 1, 1.10 takes `init` from 'random' to
    'classical_mds', and 1.8 cut `eps` from 1e-3 to 1e-6. The values are
    scikit-learn 1.8's defaults, so the layout matches the release we pin."""
    import inspect
    params = {'n_components': 2, 'n_init': 4, 'max_iter': 300, 'eps': 1e-6,
              'random_state': seed, 'normalized_stress': 'auto'}
    if 'metric_mds' in inspect.signature(MDS).parameters:
        # 1.8+: `metric` names the distance and `dissimilarity` is
        # deprecated; `init` became a constructor setting.
        params.update(metric='precomputed', metric_mds=True, init='random')
    else:
        # Before 1.8 `metric` is the metric/non-metric switch.
        params.update(dissimilarity='precomputed', metric=True)
    return params


def _marker_transform(transforms, names, marker):
    """``(method, spec)`` the samples' records give `marker`: ``'linear'``
    with no record entry, the method when every sample agrees (spec as JSON),
    else ``'mixed'`` (and no spec)."""
    import json
    specs = [(transforms.get(nm) or {}).get(marker) or {} for nm in names]
    keys = {json.dumps(sp, sort_keys=True) for sp in specs}
    if len(keys) > 1:
        return 'mixed', ''
    sp = specs[0] if specs else {}
    return sp.get('method', 'linear'), (json.dumps(sp, sort_keys=True)
                                        if sp else '')


def _import_anndata():
    """The ``anndata`` module, or an ImportError saying how to install it."""
    try:
        import anndata as ad
    except ImportError as e:
        raise ImportError(
            "AnnData export needs the 'anndata' package — pip install anndata "
            "(or: pip install 'openflo[interop]')") from e
    return ad


def _shared_markers(samples, markers):
    """``(present, dropped)``: the markers every sample has, in `markers`
    order, and the rest (logged as a warning). Shared markers keep every
    block of X the same width (ragged per-sample columns would break
    np.vstack) and matching `var`."""
    present = [m for m in markers
               if all(m in df.columns for df in samples.values())] \
        if samples else list(markers)
    dropped = [m for m in markers if m not in present]
    if dropped:
        log.warning("to_anndata: %d marker(s) omitted (missing from some "
                    "samples): %s", len(dropped), ', '.join(map(str, dropped)))
    return present, dropped


def _subsample(df, max_events, rng):
    """`df`, or `max_events` of its rows drawn without replacement when it
    has more (one `rng` draw only in that case)."""
    if max_events and len(df) > max_events:
        return df.iloc[rng.choice(len(df), max_events, replace=False)]
    return df


def _linear_block(x, rec, present, n_rows):
    """The linear (float32) values of one sample's block `x`, column by
    column through that marker's transform record in `rec`."""
    from .channel_transforms import to_linear
    if not present:
        return np.zeros((n_rows, 0), np.float32)
    return np.column_stack(
        [to_linear(x[:, j], rec.get(m)).astype(np.float32)
         for j, m in enumerate(present)])


def _obs_block(nm, sub, obs_cols):
    """One sample's ``obs`` rows: its name plus each of `obs_cols` it has."""
    o = pd.DataFrame({'sample': [nm] * len(sub)})
    for c in obs_cols:
        if c in sub.columns:
            o[c] = sub[c].to_numpy()
    return o


def _obs_frame(obs, obs_cols):
    """The blocks joined with a string index, plus a labelled sibling for
    cluster-like obs columns so the -1 noise sentinel reads "Unclustered
    (noise)" for scanpy users, while the raw integer id is preserved for
    their own joins/plots."""
    obs_df = pd.concat(obs, ignore_index=True) if obs else pd.DataFrame()
    obs_df.index = obs_df.index.astype(str)
    for c in obs_cols:
        if c in obs_df.columns and c in _CLUSTER_OBS_COLS:
            from .cluster_labels import cluster_label
            obs_df[f'{c}_label'] = [cluster_label(v) for v in obs_df[c]]
    return obs_df


def _transform_var(var, transforms, names, present, lins):
    """Fill ``var['transform']`` / ``var['transform_spec']`` and return the
    ``layers`` dict holding the linear values."""
    tf = [_marker_transform(transforms, names, m) for m in present]
    var['transform'] = [t[0] for t in tf]
    var['transform_spec'] = [t[1] for t in tf]
    return {'linear': (np.vstack(lins) if lins
                       else np.zeros((0, len(present)), np.float32))}


def to_anndata(samples, markers, obs_cols=None, max_events=None, seed=42,
               transforms=None):
    """Build an ``AnnData`` of concatenated events × markers.

    ``obs`` carries a ``sample`` column plus any ``obs_cols`` (e.g.
    ``['leiden', 'cluster']``) present in a sample. Requires the optional
    ``anndata`` package (``pip install anndata``).

    ``X`` holds the values as given. ``transforms`` (``{sample: its
    data_transforms}``) says which display transform each sample's columns
    carry, as the editor's data does (logicle 0-1 coordinates by default).
    Without it an editor export was a matrix of 0.2-0.9 values with nothing
    saying they were not intensities. With it, ``var['transform']`` /
    ``var['transform_spec']`` name each marker's transform ('mixed' when the
    samples differ) and ``layers['linear']`` holds the compensated linear
    values (float32; NaN where the scale is unknown) for intensity work."""
    ad = _import_anndata()
    rng = np.random.default_rng(seed)
    obs_cols = list(obs_cols or [])
    present, dropped = _shared_markers(samples, markers)
    Xs, lins, obs = [], [], []
    for nm, df in samples.items():
        sub = _subsample(df, max_events, rng)
        Xs.append(sub[present].to_numpy(dtype=float))
        if transforms is not None:
            lins.append(_linear_block(Xs[-1], transforms.get(nm) or {},
                                      present, len(sub)))
        obs.append(_obs_block(nm, sub, obs_cols))
    X = np.vstack(Xs) if Xs else np.zeros((0, len(present)))
    obs_df = _obs_frame(obs, obs_cols)
    var = pd.DataFrame(index=pd.Index([str(m) for m in present]))
    layers = None
    if transforms is not None:
        layers = _transform_var(var, transforms, list(samples), present, lins)
    adata = ad.AnnData(X=X, obs=obs_df, var=var, layers=layers)
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


def write_h5ad(path, samples, markers, obs_cols=None, max_events=None,
               transforms=None):
    """Write the samples to an ``.h5ad`` file (see :func:`to_anndata`).
    Returns the number of events written."""
    adata = to_anndata(samples, markers, obs_cols=obs_cols,
                       max_events=max_events, transforms=transforms)
    _h5ad_safe_strings(adata)
    adata.write_h5ad(path)
    return int(adata.n_obs)
