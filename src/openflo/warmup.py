"""Compile the embedding kernels ahead of time, while something else is running.

UMAP is numba-compiled and ships with ``cache=True`` on none of its 114 njit
kernels, so EVERY process that embeds pays the compile again from scratch —
measured here at ~18-29s. The workspace runs each unit in a fresh
``python -m openflo.workspace`` child, so a 20-file run pays that bill twenty
times: six or seven minutes of a long run spent compiling the same functions
over and over.

None of it has to be on the critical path. A run does its clustering first, and
clustering is compiled C (PhenoGraph's Louvain binaries, igraph's Leiden) that
releases the GIL — so the compile can happen in a background thread while the
clustering runs, and the embedding then starts against warm kernels.

This is not speculative: the child already knows from ``cfg`` whether an
embedding was requested, so the work is never wasted. It is only reordered.

Two measured constraints shape the warm-up payload:

  * It MUST be larger than 4096 rows. Below that UMAP uses an exact neighbour
    search and never touches the approximate-search kernels a real run spends
    its compile time in — a 600-row warm-up measured 28.5s -> 24.7s (nothing),
    a 5000-row one 28.5s -> 11.9s.
  * ``n_epochs=1`` is enough. We want the layout kernel COMPILED, not run;
    dropping the epochs cuts the warm-up's own cost without weakening it.

Nothing here can change a result. Compilation is not computation: the same
function is called with the same inputs either way. The one way this module
could affect a run is by disturbing global random state, so it never touches
the global numpy RNG — the throwaway array comes from a local ``RandomState``
and the seed is passed explicitly. tests/test_warmup.py pins that.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_started = threading.Lock()
_done = False

# Above UMAP's 4096-row exact/approximate boundary — see the module docstring.
_WARM_ROWS = 5000
_WARM_COLS = 4

# cfg keys whose embeddings are numba-compiled and so benefit from this.
_NUMBA_EMBEDDINGS = ('umap', 'trimap', 'pacmap')


def wants_embedding(cfg) -> bool:
    """Does this run config ask for a numba-backed embedding?"""
    if not cfg:
        return False
    return any(bool(cfg.get(k)) for k in _NUMBA_EMBEDDINGS)


def _warm() -> None:
    global _done
    try:
        import numpy as np
        import umap
        # A LOCAL RandomState. Touching np.random.* here would perturb the
        # global stream that the actual analysis may draw from, and a warm-up
        # that changes results is far worse than a slow one.
        rng = np.random.RandomState(0)
        X = rng.normal(0, 1, (_WARM_ROWS, _WARM_COLS))
        umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=0,
                  n_epochs=1).fit_transform(X)
        _done = True
        log.debug("  [warmup] embedding kernels compiled")
    except Exception as exc:            # noqa: BLE001
        # Never let this reach the user. Its only job is to make something
        # faster; failing at that must cost time, never the run.
        log.debug("  [warmup] skipped (%s: %s)", type(exc).__name__, exc)


def start(cfg=None, force: bool = False) -> threading.Thread | None:
    """Start compiling the embedding kernels in a daemon thread.

    Returns the thread, or None if there is nothing to do. Safe to call more
    than once — only the first call in a process does any work. Never raises,
    never blocks the caller.
    """
    global _done
    if not force and not wants_embedding(cfg):
        return None
    if not _started.acquire(blocking=False):
        return None                      # another call already owns the warm-up
    if _done:
        return None
    t = threading.Thread(target=_warm, name='openflo-warmup', daemon=True)
    t.start()
    return t
