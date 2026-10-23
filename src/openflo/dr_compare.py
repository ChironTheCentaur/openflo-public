"""Side-by-side dimensionality-reduction comparison (pure compute, headless).

Runs several 2-D embedding algorithms on the same marker matrix so their
layouts can be compared on equal footing. The package ships UMAP (``umap-learn``)
and t-SNE (``sklearn.manifold.TSNE``) in its core deps; the optional ``embed``
extra adds PHATE, TriMap and PaCMAP. Backends that aren't installed are probed
and skipped gracefully rather than crashing the run.

No plotting and no matplotlib import — callers (the GUI, notebooks) own the
visualization. Every backend is imported lazily inside the run so that importing
this module never pulls in a heavy or missing dependency.
"""
from __future__ import annotations

import numpy as np

from . import numba_threads

# All methods this helper knows how to drive, in a stable display order.
ALL_METHODS: tuple[str, ...] = ("umap", "tsne", "phate", "trimap", "pacmap")

# Each method by its published name, for display.
METHOD_NAMES = {"umap": "UMAP", "tsne": "t-SNE", "phate": "PHATE",
                "trimap": "TriMap", "pacmap": "PaCMAP"}

# Map each method name -> the top-level module that must import for it to run.
_BACKEND_MODULE: dict[str, str] = {
    "umap": "umap",
    "tsne": "sklearn",
    "phate": "phate",
    "trimap": "trimap",
    "pacmap": "pacmap",
}


# Settings given to ONE backend only, by default: FlowSample.run_umap's UMAP
# neighbours and min_dist, so the comparison draws the UMAP the rest of the
# app does. t-SNE (perplexity 30) and PHATE take their own defaults; neither
# accepts UMAP's arguments. Override per method with run_embeddings(method_kw=).
METHOD_DEFAULTS: dict[str, dict] = {
    "umap": {"n_neighbors": 30, "min_dist": 0.3},
}


def _backend_importable(method: str) -> bool:
    """Return True if ``method``'s backend module can be imported right now.

    The same import check (``capabilities.engine_unavailable``, cached) that
    Help → Environment, the Cluster dialog and the Pipeline use, so they all
    agree on what can run."""
    module = _BACKEND_MODULE.get(method)
    if module is None:
        return False
    from .capabilities import engine_unavailable
    return engine_unavailable(module) is None


def available_methods() -> list[str]:
    """Which of ``ALL_METHODS`` are importable in the current environment.

    Probes each backend with a try/except import and returns the installed
    subset in canonical order. ``tsne`` and ``umap`` are always present via the
    core dependencies; the rest depend on the optional ``embed`` extra.
    """
    return [m for m in ALL_METHODS if _backend_importable(m)]


def _embed_one(method: str, X: np.ndarray, seed: int, **kw) -> np.ndarray:
    """Run a single backend on ``X`` and return an (n, 2) float array.

    Each backend is imported lazily here. ``n_components=2`` and
    ``random_state=seed`` are passed where the backend supports them; extra
    ``**kw`` are forwarded verbatim. The numba-backed ones (UMAP, TriMap,
    PaCMAP) take turns where numba needs it: see openflo/numba_threads.py.
    """
    if method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed, **kw)
        with numba_threads.parallel_kernels():
            return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE perplexity must be strictly < n_samples; clamp so small inputs
        # don't error. The floor is 1.0 (NOT 5.0 — a 5.0 floor exceeds n for
        # 3<=n<=5 and t-SNE would still raise), and (n-1)/3 < n for all n.
        n = len(X)
        perplexity = kw.pop("perplexity", 30.0)
        perplexity = min(perplexity, max(1.0, (n - 1) / 3.0))
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity, **kw)
        return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == "phate":
        import phate

        reducer = phate.PHATE(n_components=2, random_state=seed, verbose=False, **kw)
        return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == "trimap":
        import trimap

        # TriMap has no random_state arg; it is deterministic given inputs.
        reducer = trimap.TRIMAP(n_dims=2, **kw)
        with numba_threads.parallel_kernels():
            return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == "pacmap":
        import pacmap

        reducer = pacmap.PaCMAP(n_components=2, random_state=seed, **kw)
        with numba_threads.parallel_kernels():
            return np.asarray(reducer.fit_transform(X, init="pca"),
                              dtype=float)

    raise ValueError(f"unknown method: {method!r}")


def run_embeddings(
    X,
    methods=("umap", "tsne", "phate"),
    seed: int = 42,
    max_points: int | None = None,
    method_kw: dict[str, dict] | None = None,
    **kw,
) -> dict:
    """Embed ``X`` with several DR backends and return their 2-D coordinates.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Float feature matrix (e.g. arcsinh-transformed marker intensities).
    methods : iterable of str
        Subset of ``ALL_METHODS`` to attempt. Any whose backend is not
        installed is skipped (recorded in the returned ``skipped`` list) rather
        than raising.
    seed : int
        Seed used for ``random_state`` (and for the subsample draw). 42, the
        seed the rest of OpenFlo uses (0 up to 2.7.0).
    max_points : int or None
        If set and ``n > max_points``, a seeded random subsample of
        ``max_points`` rows is drawn and embedded; the chosen row indices are
        returned in ``index``. If None, all rows are used.
    method_kw : dict or None
        Per-method keyword args, ``{method: {arg: value}}``, given to that
        backend only, over ``METHOD_DEFAULTS`` (UMAP 30 neighbours, min_dist
        0.3).
    **kw
        Extra keyword args forwarded to every backend constructor.

    Returns
    -------
    dict
        ``{"coords": {method: ndarray (m, 2)}, "index": ndarray (m,),
        "skipped": [(method, reason), ...]}`` where ``m`` is the number of rows
        actually embedded (all rows, or the subsample). ``coords`` only contains
        methods that ran successfully.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, d); got shape {X.shape}")
    n = len(X)

    # Seeded subsample if requested.
    if max_points is not None and n > max_points:
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(n, size=max_points, replace=False))
    else:
        index = np.arange(n)
    Xsub = X[index]

    coords: dict[str, np.ndarray] = {}
    skipped: list[tuple[str, str]] = []
    for method in methods:
        if method not in _BACKEND_MODULE:
            skipped.append((method, "unknown method"))
            continue
        if not _backend_importable(method):
            skipped.append((method, "backend not installed"))
            continue
        try:
            own = {**METHOD_DEFAULTS.get(method, {}), **kw,
                   **(method_kw or {}).get(method, {})}
            coords[method] = _embed_one(method, Xsub, seed, **own)
        except Exception as exc:  # noqa: BLE001 - report, never crash the batch
            skipped.append((method, f"{type(exc).__name__}: {exc}"))

    return {"coords": coords, "index": index, "skipped": skipped}
