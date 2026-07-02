"""Spectral unmixing for full-spectrum cytometers (Cytek Aurora, BD S8…).

Conventional compensation subtracts spillover between a few detectors;
spectral cytometers instead record each event's full emission spectrum
across many detectors and *unmix* it into per-fluorophore abundances using
reference spectra measured from single-stain controls (plus an
autofluorescence spectrum from an unstained control).

Two layers, both pure (numpy):

  * ``build_reference_spectra`` — turn single-stain (and unstained) control
    arrays into a reference spectra matrix S (n_fluors × n_detectors).
  * ``unmix`` — solve, per event, the least-squares abundances A such that
    A · S ≈ Y (the raw detector matrix). OLS by default (what Cytek/SpectroFlo
    use); optional non-negativity.

``apply_unmixing`` wires it onto a FlowSample, adding one abundance column
per fluor.
"""
from __future__ import annotations

import numpy as np


def _normalize(spec, mode='max'):
    spec = np.asarray(spec, dtype=float)
    if mode == 'l2':
        n = np.linalg.norm(spec)
        return spec / n if n > 0 else spec
    if mode == 'max':
        mx = spec.max()
        return spec / mx if mx > 0 else spec
    return spec


def build_reference_spectra(single_stains, unstained=None, bright_pct=90.0,
                            normalize='max'):
    """Build the reference spectra matrix from control arrays.

    `single_stains` : ``{fluor_name: ndarray (n_events, n_detectors)}`` —
                      each a single-stain control over the SAME detectors.
    `unstained`     : optional ndarray (m, n_detectors); its mean is the
                      autofluorescence spectrum, subtracted from every
                      single-stain spectrum and added as its own endmember.
    `bright_pct`    : use only the brightest events (by total signal) of each
                      single stain to estimate its spectrum (avoids the
                      negative population dragging the mean).

    Returns ``(S, fluors)`` where ``S`` is ``(n_fluors, n_detectors)`` (rows
    L-normalised per `normalize`) and `fluors` is the row label list (with
    ``'Autofluorescence'`` last when `unstained` is given)."""
    auto = None
    if unstained is not None and len(unstained):
        auto = np.mean(np.asarray(unstained, dtype=float), axis=0)

    fluors, rows = [], []
    for name, arr in single_stains.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.all(np.isfinite(arr), axis=1)]   # drop non-finite events
        if arr.size == 0:
            continue
        total = arr.sum(axis=1)
        thr = np.percentile(total, bright_pct)
        bright = arr[total >= thr] if np.any(total >= thr) else arr
        spec = bright.mean(axis=0)
        if auto is not None:
            spec = spec - auto
        spec = np.clip(spec, 0.0, None)
        fluors.append(name)
        rows.append(_normalize(spec, normalize))
    if auto is not None:
        fluors.append('Autofluorescence')
        rows.append(_normalize(np.clip(auto, 0.0, None), normalize))
    return np.asarray(rows, dtype=float), fluors


def unmix(Y, S, nonneg=False):
    """Unmix raw detector signals into fluorophore abundances.

    `Y` : ``(n_events, n_detectors)`` raw signal.
    `S` : ``(n_fluors, n_detectors)`` reference spectra.
    Solves ``A · S ≈ Y`` for ``A`` (n_events, n_fluors) by ordinary least
    squares (vectorised). With `nonneg`, negatives are clipped to 0 (a fast
    approximation to NNLS, adequate once spectra are clean)."""
    Y = np.asarray(Y, dtype=float)
    S = np.asarray(S, dtype=float)
    # Solve S.T (det × fluor) @ A.T (fluor × events) = Y.T (det × events).
    A = np.linalg.lstsq(S.T, Y.T, rcond=None)[0].T
    if nonneg:
        A = np.clip(A, 0.0, None)
    return A


def apply_unmixing(sample, S, fluors, detectors, nonneg=False, prefix='U:'):
    """Unmix `sample` in place: read the `detectors` columns, unmix against
    spectra `S` (labelled `fluors`), and add one ``f'{prefix}{fluor}'``
    abundance column per fluor. Returns the list of new column names."""
    cols = [d for d in detectors if d in sample.data.columns]
    if len(cols) != S.shape[1]:
        raise ValueError(
            f"spectra have {S.shape[1]} detectors but {len(cols)} of the "
            f"requested detectors are present in the sample")
    Y = sample.data[cols].to_numpy(dtype=float)
    A = unmix(Y, S, nonneg=nonneg)
    new_cols = []
    for j, f in enumerate(fluors):
        name = f'{prefix}{f}'
        sample.data[name] = A[:, j]
        new_cols.append(name)
    return new_cols


# ── Unmixing quality control ──────────────────────────────────────────────────
#
# Two complementary diagnostics for how trustworthy an unmix will be:
#   • the spectral SIMILARITY matrix — purely a function of the reference
#     spectra; flags fluorophore pairs whose signatures are nearly collinear
#     (hard to resolve, the unmix amplifies their noise);
#   • the Spillover Spread Matrix (SSM, Nguyen 2013 / Cytek) — measured from
#     the single-stain controls; quantifies the spreading error each stain
#     introduces into every other fluor's unmixed channel.

def spectral_similarity_matrix(S):
    """Cosine-similarity matrix between reference-spectra rows.

    ``M[i, j] = <Sᵢ, Sⱼ> / (|Sᵢ|·|Sⱼ|)`` — in ``[0, 1]`` for the non-negative
    spectra ``build_reference_spectra`` produces. Diagonal is 1. A high
    off-diagonal value (≳ 0.98) means the two fluorophores are spectrally
    almost indistinguishable, so unmixing them is ill-conditioned and their
    abundances will be noisy/anti-correlated. Returns an ``(n, n)`` ndarray."""
    S = np.asarray(S, dtype=float)
    norm = np.linalg.norm(S, axis=1)
    norm[norm == 0] = 1.0
    U = S / norm[:, None]
    M = np.clip(U @ U.T, -1.0, 1.0)
    np.fill_diagonal(M, 1.0)   # self-similarity is 1 even for a zero-norm spectrum
    return M


def spectral_condition_number(S):
    """2-norm condition number of the spectra matrix ``S`` (n_fluors ×
    n_detectors): ``σ_max / σ_min`` of its singular values. A large value
    (≳ 100) means the unmixing system is ill-posed — small detector noise is
    amplified into large abundance errors. ``inf`` for a rank-deficient S."""
    S = np.asarray(S, dtype=float)
    # More fluors than detectors → the unmix is underdetermined (abundances not
    # uniquely identifiable). svd returns only min(shape) singular values, which
    # can all be nonzero, so cond would look finite/small; report inf instead.
    if S.ndim != 2 or S.shape[0] > S.shape[1]:
        return float('inf')
    sv = np.linalg.svd(S, compute_uv=False)
    if sv.size == 0 or sv.size < min(S.shape):
        return float('inf')
    smax = float(sv.max())
    # Treat singular values below the standard numerical-rank tolerance as
    # zero, so a (near-)rank-deficient matrix reports inf rather than a
    # meaningless 1e16 from a residual float.
    tol = smax * max(S.shape) * np.finfo(float).eps
    smin = float(sv.min())
    if smax <= 0 or smin <= tol:
        return float('inf')
    return smax / smin


def spillover_spread_matrix(single_stains, S, fluors, nonneg=False,
                            n_bins=8, min_bin=30):
    """Spillover Spread Matrix (Nguyen et al. 2013; Cytek SpectroFlo).

    For each single-stain control ``j`` (an ``(events, detectors)`` array in
    ``single_stains[fluor_j]``), unmix it against ``S`` and measure how much
    spreading error it injects into every other fluor ``i``: within quantile
    bins of the primary abundance ``A[:, j]``, the standard deviation of the
    spillover channel ``A[:, i]`` grows like ``SSᵢⱼ · sqrt(primary)``. So
    ``SSᵢⱼ`` is the robust median of ``SD(Aᵢ) / sqrt(median primary)`` over
    the bins — the spreading-error coefficient.

    Returns ``(SSM, fluors)`` where ``SSM`` is ``(n_fluors, n_fluors)`` in the
    row-order of ``fluors`` (``SSM[i, j]`` = spread into ``i`` from stain
    ``j``; diagonal 0). A stain absent from ``single_stains`` (or with too few
    events) leaves its column ``NaN``."""
    S = np.asarray(S, dtype=float)
    nf = len(fluors)
    SSM = np.full((nf, nf), np.nan, dtype=float)
    idx = {f: i for i, f in enumerate(fluors)}
    for jname, arr in single_stains.items():
        if jname not in idx:
            continue
        j = idx[jname]
        Y = np.asarray(arr, dtype=float)
        if Y.ndim != 2 or Y.shape[1] != S.shape[1] or len(Y) < min_bin * 2:
            continue
        A = unmix(Y, S, nonneg=nonneg)
        primary = A[:, j]
        pos = primary > 0
        if int(pos.sum()) < min_bin * 2:
            continue
        p = primary[pos]
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 3:
            continue
        binidx = np.clip(np.searchsorted(edges, p, side='right') - 1,
                         0, edges.size - 2)
        SSM[j, j] = 0.0
        for i in range(nf):
            if i == j:
                continue
            ai = A[pos, i]
            ratios = []
            for b in range(edges.size - 1):
                m = binidx == b
                if int(m.sum()) < min_bin:
                    continue
                prim_med = float(np.median(p[m]))
                if prim_med <= 0:
                    continue
                ratios.append(float(np.std(ai[m])) / np.sqrt(prim_med))
            if ratios:
                SSM[i, j] = float(np.median(ratios))
    return SSM, list(fluors)


def unmixing_qc(single_stains, S, fluors, nonneg=False, sim_threshold=0.98,
                top_spread=5):
    """Bundle the spectral-unmixing diagnostics into one report dict.

    Returns ``{fluors, similarity, ssm, condition_number, similar_pairs,
    worst_spread}`` where ``similar_pairs`` lists fluor pairs whose spectral
    cosine similarity is ≥ ``sim_threshold`` (descending), and
    ``worst_spread`` lists the ``top_spread`` largest finite SSM entries
    (``into``/``from`` fluor + value) — the pairs most worth scrutinizing."""
    sim = spectral_similarity_matrix(S)
    cond = spectral_condition_number(S)
    ssm, _ = spillover_spread_matrix(single_stains, S, fluors, nonneg=nonneg)
    nf = len(fluors)

    similar_pairs = []
    for i in range(nf):
        for j in range(i + 1, nf):
            if sim[i, j] >= sim_threshold:
                similar_pairs.append({'fluor_a': fluors[i], 'fluor_b': fluors[j],
                                      'similarity': float(sim[i, j])})
    similar_pairs.sort(key=lambda d: d['similarity'], reverse=True)

    spread = []
    for i in range(nf):
        for j in range(nf):
            v = ssm[i, j]
            if i != j and np.isfinite(v):
                spread.append({'into': fluors[i], 'from': fluors[j],
                               'spread': float(v)})
    spread.sort(key=lambda d: d['spread'], reverse=True)

    return {'fluors': list(fluors), 'similarity': sim, 'ssm': ssm,
            'condition_number': cond, 'similar_pairs': similar_pairs,
            'worst_spread': spread[:top_spread]}
