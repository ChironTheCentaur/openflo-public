"""Proliferation modelling from a dye-dilution histogram (Complaints 21).

Each division halves a cell's dye (CFSE, CellTrace and the like), so the
generations sit one log2 unit apart below the undivided peak. The model is a
mixture of Gaussians on log2 intensity with the generation means at
``m0 - k*d`` (``d`` near 1, fitted), one shared width, and a weight per
generation, fitted by expectation-maximisation.

From the cells counted in each generation ``n_k`` come the standard indices
(Roederer, Cytometry A 2011;79:95), each on the cells that started:

- precursor frequency of generation k: ``(n_k / 2^k) / sum_j (n_j / 2^j)``
- percent divided: ``100 * (1 - precursor fraction of generation 0)``
- division index: mean divisions of all starting cells,
  ``sum_k k n_k/2^k / sum_k n_k/2^k``
- proliferation index: mean divisions of the cells that divided,
  ``sum_{k>=1} k n_k/2^k / sum_{k>=1} n_k/2^k``
- expansion index: cells now per starting cell,
  ``sum_k n_k / sum_k n_k/2^k``

Pure numpy.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def proliferation_indices(counts):
    """The indices above from the cells counted in generations 0, 1, ...
    ``counts``. NaN where they are undefined (no cells, or none divided)."""
    n = np.asarray(counts, dtype=float)
    k = np.arange(n.size)
    start = n / 2.0 ** k                       # starting cells per generation
    total = start.sum()
    nan = float('nan')
    if total <= 0:
        return {'precursor_frequency': [nan] * n.size, 'pct_divided': nan,
                'division_index': nan, 'proliferation_index': nan,
                'expansion_index': nan}
    divided = start[1:].sum()
    return {
        'precursor_frequency': [float(f) for f in start / total],
        'pct_divided': float(100.0 * divided / total),
        'division_index': float((k * start).sum() / total),
        'proliferation_index': (float((k[1:] * start[1:]).sum() / divided)
                                if divided > 0 else nan),
        'expansion_index': float(n.sum() / total),
    }


def fit_generations(values, undivided=None, max_generations=8, iters=300,
                    tol=1e-7):
    """Fit the generation model to linear dye intensities ``values``.

    ``undivided`` is the undivided peak's intensity (an unstimulated
    control's median, say), and stays fixed; without it the brightest
    well-populated peak starts the fit and its position is fitted too.
    Returns ``{'ok', 'undivided', 'spacing', 'sd', 'weights',
    'counts', 'means', 'labels'}`` plus the indices of
    proliferation_indices; ``labels`` gives each value's generation (-1 for
    a non-positive value)."""
    v = np.asarray(values, dtype=float)
    pos = np.isfinite(v) & (v > 0)
    out: dict[str, Any] = {'ok': False}
    if pos.sum() < 50:
        return out
    x = np.log2(v[pos])
    if undivided is None or not (np.isfinite(undivided) and undivided > 0):
        hist, edges = np.histogram(x, bins=128)
        centres = 0.5 * (edges[:-1] + edges[1:])
        smooth = np.convolve(hist, np.ones(5) / 5.0, mode='same')
        peaks = [i for i in range(1, len(smooth) - 1)
                 if smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]
                 and smooth[i] >= 0.05 * smooth.max()]
        m0 = float(centres[peaks[-1]]) if peaks else float(np.percentile(x, 95))
    else:
        m0 = float(np.log2(undivided))
    fixed = undivided is not None and np.isfinite(undivided) and undivided > 0
    ng = int(max_generations) + 1
    k = np.arange(ng)
    d = 1.0
    sd = 0.25
    w = np.full(ng, 1.0 / ng)
    prev = -np.inf
    resp = np.zeros((ng, x.size))
    for _ in range(iters):
        means = m0 - k * d
        z = (x[None, :] - means[:, None]) / sd
        logp = np.log(np.maximum(w, 1e-300))[:, None] - 0.5 * z * z \
            - np.log(sd)
        top = logp.max(axis=0)
        p = np.exp(logp - top)
        tot = p.sum(axis=0)
        resp = p / tot
        ll = float((np.log(tot) + top).sum())
        nk = resp.sum(axis=1)
        w = nk / x.size
        # m0 and d by weighted least squares on mean_k = m0 - k d.
        sw = resp.sum()
        swk = (resp * k[:, None]).sum()
        swkk = (resp * (k * k)[:, None]).sum()
        swx = (resp * x[None, :]).sum()
        swkx = (resp * k[:, None] * x[None, :]).sum()
        det = sw * swkk - swk * swk
        if fixed and swkk > 0:
            # Undivided peak given: only the spacing is fitted.
            d = float(np.clip((m0 * swk - swkx) / swkk, 0.6, 1.4))
        elif det > 0:
            m0 = (swkk * swx - swk * swkx) / det
            d = float(np.clip(-(sw * swkx - swk * swx) / det, 0.6, 1.4))
        means = m0 - k * d
        sd = float(np.sqrt(max((resp * (x[None, :] - means[:, None]) ** 2)
                               .sum() / x.size, 1e-4)))
        if abs(ll - prev) < tol * abs(ll):
            break
        prev = ll
    counts = np.round(resp.sum(axis=1)).astype(int)
    labels = np.full(v.size, -1)
    labels[pos] = np.argmax(resp, axis=0)
    out.update(ok=True, undivided=float(2.0 ** m0), spacing=float(d),
               sd=float(sd), weights=list(map(float, w)),
               counts=list(map(int, counts)),
               means=[float(2.0 ** m) for m in (m0 - k * d)],
               labels=labels)
    out.update(proliferation_indices(counts))
    return out
