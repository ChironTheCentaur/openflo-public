"""Compiled inner loop for FlowSOM training (``pipeline._som_train``).

The online SOM visits every event in a Python loop, each visit making five
small numpy calls: about 10 s per sample at the defaults (50k events x 10
iterations). This module runs the same loop under numba, which umap-learn
already installs, so nothing new ships.

It must give the SAME weights, bit for bit, not merely close ones: FlowSOM
node and metacluster labels follow from them, and a result that moves when
someone upgrades OpenFlo is a result nobody can trust. Three details keep it
exact:

  * The distance to each node is summed in numpy's own order. ``(W - x) ** 2``
    then ``.sum(1)`` adds each row with numpy's pairwise summation (eight
    running partial sums for rows of 8 to 128 values, split in halves above
    that), so :func:`_pairwise_sum` repeats that order exactly. A plain
    left-to-right loop differs in the last bit and can flip a near-tied best
    node.
  * The neighbourhood weights ``lr * exp(-d2 / (2 sigma^2))`` are computed by
    numpy, not here. numpy's ``exp`` may use its own SIMD code, which need not
    agree with the C library's ``exp`` numba calls. The caller passes them in
    as a node-by-node table, one per iteration (grid distances take only a few
    values, so this costs nothing).
  * No fast-math: numba then keeps each multiply and add as separate IEEE
    operations, the same ones numpy performs.

``tests/test_som_kernel.py`` pins all three against the numpy loop.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_kernel = None
_tried = False


def _build():
    import numba
    import numpy as np

    def jit(fn):
        # cache=True saves the compile (about a second) for the next process.
        # Where no cache folder is writable numba refuses caching at
        # decoration time; compile per process then.
        try:
            return numba.njit(cache=True)(fn)
        except Exception:
            return numba.njit(cache=False)(fn)

    @jit
    def _pairwise_sum(a, s, n):
        # numpy's pairwise_sum for float64 (numpy/_core/src/umath/loops_utils),
        # term for term.
        if n < 8:
            res = 0.0
            for i in range(s, s + n):
                res += a[i]
            return res
        if n <= 128:
            r0 = a[s]
            r1 = a[s + 1]
            r2 = a[s + 2]
            r3 = a[s + 3]
            r4 = a[s + 4]
            r5 = a[s + 5]
            r6 = a[s + 6]
            r7 = a[s + 7]
            i = 8
            while i < n - (n % 8):
                r0 += a[s + i]
                r1 += a[s + i + 1]
                r2 += a[s + i + 2]
                r3 += a[s + i + 3]
                r4 += a[s + i + 4]
                r5 += a[s + i + 5]
                r6 += a[s + i + 6]
                r7 += a[s + i + 7]
                i += 8
            res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while i < n:
                res += a[s + i]
                i += 1
            return res
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum(a, s, n2) + _pairwise_sum(a, s + n2, n - n2)

    @jit
    def som_epoch(X, W, order, LH):
        """One pass over ``X`` in ``order``, updating ``W`` in place.
        ``LH[b, n]`` is ``lr * exp(-griddist2(b, n) / (2 sigma^2))``."""
        n_nodes, d = W.shape
        sq = np.empty(d)
        for i in order:
            best = 0
            best_d = np.inf
            for n in range(n_nodes):
                for j in range(d):
                    e = W[n, j] - X[i, j]
                    sq[j] = e * e
                dist = 0.0 + _pairwise_sum(sq, 0, d)
                if dist != dist:
                    # np.argmin returns the first NaN
                    best = n
                    break
                if n == 0 or dist < best_d:
                    best = n
                    best_d = dist
            for n in range(n_nodes):
                h = LH[best, n]
                for j in range(d):
                    W[n, j] = W[n, j] + h * (X[i, j] - W[n, j])

    return som_epoch, _pairwise_sum


def _kernels():
    global _kernel, _tried
    if not _tried:
        _tried = True
        try:
            _kernel = _build()
        except Exception as e:  # numba missing or broken: keep working
            log.info("FlowSOM: numba unavailable (%s); using the numpy loop", e)
            _kernel = None
    return _kernel


def som_epoch_kernel():
    """The compiled epoch function, or ``None`` when numba is unavailable (the
    caller then runs its numpy loop, which gives the same weights)."""
    k = _kernels()
    return k[0] if k else None


def pairwise_sum_kernel():
    """The compiled ``_pairwise_sum(a, start, n)`` (for tests), or ``None``."""
    k = _kernels()
    return k[1] if k else None
