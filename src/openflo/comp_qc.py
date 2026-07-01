"""Compensation / spillover QC.

Standalone, headless helpers for inspecting a spillover (compensation) matrix.
A spillover matrix ``M`` is square with ``M[i, j]`` the fraction of signal from
fluorophore ``i`` (the *source*) that leaks into detector/channel ``j`` (the
*destination*). The diagonal is ~1.0; large off-diagonal entries flag channel
pairs that spill heavily and warrant attention before analysis.

matplotlib is imported lazily inside :func:`comp_qc_figure` (Agg-safe; the
figure is returned, never shown), so importing this module pulls in only numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["spillover_metrics", "comp_qc_figure"]

# Off-diagonal spillover at or above this fraction is considered "strong".
STRONG_THRESHOLD = 0.10


def _as_matrix(matrix, channels) -> tuple[np.ndarray, list[str]]:
    """Validate inputs and return a (square float ndarray, channel list)."""
    if matrix is None:
        raise ValueError("spillover matrix is None; no compensation to inspect")
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"spillover matrix must be square 2-D; got shape {arr.shape}"
        )
    k = arr.shape[0]
    chans = list(channels) if channels is not None else []
    if len(chans) != k:
        raise ValueError(
            f"channels has {len(chans)} names but matrix is {k}x{k}; "
            "they must match"
        )
    return arr, chans


def spillover_metrics(matrix, channels) -> dict:
    """Summarise off-diagonal spillover in a compensation matrix.

    ``matrix[i, j]`` = fraction of source channel ``i`` leaking into
    destination channel ``j``. The diagonal (self-signal, ~1.0) is ignored.

    Returns a dict with::

        {'max_offdiag': float,                 # largest off-diagonal entry
         'max_pair': (src_channel, dst_channel),
         'mean_offdiag': float,                # mean of all off-diagonal entries
         'strong_pairs': [(src, dst, value), ...],  # value > 0.10, desc
         'n_channels': k}

    Raises ``ValueError`` for None / non-square inputs or a channel-count
    mismatch.
    """
    arr, chans = _as_matrix(matrix, channels)
    k = arr.shape[0]

    off_mask = ~np.eye(k, dtype=bool)
    off_vals = arr[off_mask]

    if off_vals.size == 0:  # 1x1 matrix: no off-diagonal entries
        return {
            "max_offdiag": 0.0,
            "max_pair": None,
            "mean_offdiag": 0.0,
            "strong_pairs": [],
            "n_channels": k,
        }

    flat_idx = int(np.argmax(arr[off_mask]))
    rows, cols = np.where(off_mask)
    src_i, dst_j = int(rows[flat_idx]), int(cols[flat_idx])

    # Strong pairs: off-diagonal entries strictly above the threshold, desc.
    strong: list[tuple[str, str, float]] = []
    for i, j in zip(*np.where(off_mask & (arr > STRONG_THRESHOLD)), strict=True):
        strong.append((chans[i], chans[j], float(arr[i, j])))
    strong.sort(key=lambda t: t[2], reverse=True)

    return {
        "max_offdiag": float(off_vals.max()),
        "max_pair": (chans[src_i], chans[dst_j]),
        "mean_offdiag": float(off_vals.mean()),
        "strong_pairs": strong,
        "n_channels": k,
    }


def comp_qc_figure(matrix, channels, title: str = ""):
    """Render a spillover-matrix heatmap and return the matplotlib Figure.

    Channels label both axes (source = rows, destination = columns); each cell
    is annotated with its value. The colour scale is clipped to [0, 0.3] so the
    near-1.0 diagonal does not wash out the off-diagonal spillover that matters
    for QC. The caller is responsible for saving or embedding the figure; this
    function never calls ``show()``.
    """
    arr, chans = _as_matrix(matrix, channels)
    k = arr.shape[0]

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    # Size grows with channel count but stays within reason.
    side = max(4.0, min(0.6 * k + 2.0, 16.0))
    fig = Figure(figsize=(side, side))
    ax = fig.add_subplot(111)

    # Highlight off-diagonal spillover: clip the colour scale low so the
    # ~1.0 diagonal saturates and small leaks remain visible.
    vmax = max(STRONG_THRESHOLD * 3.0, 1e-6)  # 0.3 by default
    im = ax.imshow(arr, cmap="magma", vmin=0.0, vmax=vmax, aspect="equal")

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(chans, rotation=90, fontsize=8)
    ax.set_yticklabels(chans, fontsize=8)
    ax.set_xlabel("destination channel (spill into)")
    ax.set_ylabel("source channel (spill from)")
    ax.set_title(title or "Spillover matrix")

    # Annotate each cell; pick a contrasting text colour against the cell.
    thresh = vmax * 0.5
    for i in range(k):
        for j in range(k):
            val = arr[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if val < thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="spillover fraction (clipped)")
    fig.tight_layout()
    return fig
