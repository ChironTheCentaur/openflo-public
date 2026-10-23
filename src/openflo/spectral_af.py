"""Multiple autofluorescence signatures and the NxN unmixed-pairs plot
(Complaints 24).

Tissue and blood samples hold cell types with different autofluorescence
spectra; one averaged AF spectrum leaves the others' AF in the fluor
channels. Here the unstained control's spectra are clustered into ``k``
signatures, and every cell is unmixed once per signature and keeps the one
that explains it best (smallest residual), as AutoSpectral and SpectroFlo's
"multiple AF" do. Each cell's chosen signature is reported as its AF index.

Pure numpy; matplotlib only inside unmixed_pairs_figure.
"""
from __future__ import annotations

import numpy as np


def af_signatures(unstained, k=3, seed=0, iters=50, n_max=20000):
    """Up to ``k`` autofluorescence spectra (rows, max-normalised) from the
    unstained control's ``(events, detectors)`` linear signal: k-means on
    each event's normalised spectrum, the bright quarter of events only
    (dim events are mostly noise), each signature the median spectrum of
    its cluster. Fewer rows come back when clusters end up empty or
    duplicate (cosine similarity above 0.995)."""
    U = np.asarray(unstained, dtype=float)
    U = U[np.isfinite(U).all(axis=1)]
    if U.ndim != 2 or len(U) < 4 * k:
        raise ValueError("The unstained control has too few events for "
                         f"{k} autofluorescence signatures.")
    rng = np.random.default_rng(seed)
    total = U.sum(axis=1)
    U = U[total >= np.percentile(total, 75)]
    if len(U) > n_max:
        U = U[rng.choice(len(U), n_max, replace=False)]
    norm = U / np.maximum(np.abs(U).max(axis=1, keepdims=True), 1e-12)
    # k-means++ seeding, then Lloyd iterations.
    centres = [norm[rng.integers(len(norm))]]
    for _ in range(1, k):
        d = np.min([((norm - c) ** 2).sum(axis=1) for c in centres], axis=0)
        if not d.sum() > 0:
            break
        centres.append(norm[rng.choice(len(norm), p=d / d.sum())])
    C = np.array(centres)
    lab = np.zeros(len(norm), dtype=int)
    for _ in range(iters):
        lab = np.argmin(((norm[:, None, :] - C[None]) ** 2).sum(axis=2), axis=1)
        new = np.array([norm[lab == c].mean(axis=0) if (lab == c).any()
                        else C[c] for c in range(len(C))])
        if np.allclose(new, C):
            break
        C = new
    sigs = []
    for c in range(len(C)):
        if not (lab == c).any():
            continue
        s = np.median(U[lab == c], axis=0)
        s = s / max(float(np.abs(s).max()), 1e-12)
        if all(np.dot(s, t) / (np.linalg.norm(s) * np.linalg.norm(t)) < 0.995
               for t in sigs):
            sigs.append(s)
    return np.array(sigs)


def unmix_multi_af(Y, S, af, nonneg=False):
    """Unmix ``Y`` (events x detectors) against the fluor spectra ``S``
    (fluors x detectors, no AF row) plus, per cell, the one AF signature of
    ``af`` (signatures x detectors) with the smallest residual.

    Returns ``(A, af_abundance, af_index)``: ``A`` (events x fluors), each
    cell's AF abundance, and its signature's index (from 0; -1 for an event
    with non-finite detector values, whose abundances are NaN)."""
    Y = np.asarray(Y, dtype=float)
    S = np.asarray(S, dtype=float)
    af = np.atleast_2d(np.asarray(af, dtype=float))
    n, nf = len(Y), S.shape[0]
    good = np.isfinite(Y).all(axis=1)
    best_r = np.full(n, np.inf)
    A = np.full((n, nf), np.nan)
    af_ab = np.full(n, np.nan)
    idx = np.full(n, -1, dtype=int)
    Yg = Y[good]
    for k, sig in enumerate(af):
        M = np.vstack([S, sig[None, :]])
        coef, *_ = np.linalg.lstsq(M.T, Yg.T, rcond=None)
        coef = coef.T
        resid = ((Yg - coef @ M) ** 2).sum(axis=1)
        rows = np.flatnonzero(good)
        better = resid < best_r[rows]
        r = rows[better]
        best_r[r] = resid[better]
        A[r] = coef[better, :nf]
        af_ab[r] = coef[better, nf]
        idx[r] = k
    if nonneg:
        A = np.where(np.isfinite(A), np.clip(A, 0.0, None), A)
    return A, af_ab, idx


def unmixed_pairs_figure(A, fluors, max_events=3000, seed=0, title=''):
    """An NxN grid of every pair of unmixed fluors (``A``: events x fluors,
    linear abundances), on arcsinh axes so negatives and spread show, the
    diagonal a histogram of each fluor. A skewed or bent pair points at a
    spectrum problem. Returns the matplotlib Figure (never shown)."""
    from matplotlib.figure import Figure
    A = np.asarray(A, dtype=float)
    A = A[np.isfinite(A).all(axis=1)]
    n = len(fluors)
    if len(A) > max_events:
        A = A[np.random.default_rng(seed).choice(len(A), max_events,
                                                 replace=False)]
    cof = np.array([max(float(np.percentile(np.abs(A[:, j]), 50)), 1.0)
                    for j in range(n)]) if len(A) else np.ones(n)
    Z = np.arcsinh(A / cof)
    size = min(1.3 * n + 1.0, 16.0)
    fig = Figure(figsize=(size, size))
    axes = fig.subplots(n, n, squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == j:
                ax.hist(Z[:, j], bins=60, color='0.35')
            else:
                ax.scatter(Z[:, j], Z[:, i], s=0.6, alpha=0.4,
                           linewidths=0, color='C0', rasterized=True)
            if i == n - 1:
                ax.set_xlabel(fluors[j], fontsize=6, rotation=45, ha='right')
            if j == 0:
                ax.set_ylabel(fluors[i], fontsize=6, rotation=0, ha='right',
                              va='center')
    if title:
        fig.suptitle(title, fontsize=9)
    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.98, top=0.95,
                        wspace=0.05, hspace=0.05)
    return fig
