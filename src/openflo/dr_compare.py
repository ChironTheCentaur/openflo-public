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

# All methods this helper knows how to drive, in a stable display order.
ALL_METHODS: tuple[str, ...] = ("umap", "tsne", "phate", "trimap", "pacmap")

# Map each method name -> the top-level module that must import for it to run.
_BACKEND_MODULE: dict[str, str] = {
    "umap": "umap",
    "tsne": "sklearn",
    "phate": "phate",
    "trimap": "trimap",
    "pacmap": "pacmap",
}


def _backend_importable(method: str) -> bool:
    """Return True if ``method``'s backend module can be imported right now."""
    module = _BACKEND_MODULE.get(method)
    if module is None:
        return False
    try:
        __import__(module)
    except Exception:
        return False
    return True


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
    ``**kw`` are forwarded verbatim.
    """
    if method == "umap":
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed, **kw)
        return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == "tsne":
        from sklearn.manifold import TSNE

        # t-SNE perplexity must be < n_samples; clamp so small inputs don't error.
        n = len(X)
        perplexity = kw.pop("perplexity", 30.0)
        perplexity = min(perplexity, max(5.0, (n - 1) / 3.0))
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
        return np.asarray(reducer.fit_transform(X), dtype=float)

    if method == "pacmap":
        import pacmap

        reducer = pacmap.PaCMAP(n_components=2, random_state=seed, **kw)
        return np.asarray(reducer.fit_transform(X, init="pca"), dtype=float)

    raise ValueError(f"unknown method: {method!r}")


def run_embeddings(
    X,
    methods=("umap", "tsne", "phate"),
    seed: int = 0,
    max_points: int | None = None,
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
        Seed used for ``random_state`` (and for the subsample draw).
    max_points : int or None
        If set and ``n > max_points``, a seeded random subsample of
        ``max_points`` rows is drawn and embedded; the chosen row indices are
        returned in ``index``. If None, all rows are used.
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
            coords[method] = _embed_one(method, Xsub, seed, **kw)
        except Exception as exc:  # noqa: BLE001 - report, never crash the batch
            skipped.append((method, f"{type(exc).__name__}: {exc}"))

    return {"coords": coords, "index": index, "skipped": skipped}
