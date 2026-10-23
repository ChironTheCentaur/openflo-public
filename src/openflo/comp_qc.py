"""Compensation / spillover QC.

Standalone, headless helpers for inspecting a spillover (compensation) matrix.
A spillover matrix ``M`` is square with ``M[i, j]`` the fraction of signal from
fluorophore ``i`` (the *source*) that leaks into detector/channel ``j`` (the
*destination*). The diagonal is ~1.0; large off-diagonal entries flag channel
pairs that spill heavily and warrant attention before analysis.

Everything here is on the spill as a fraction of the source's OWN signal (each
row divided by its diagonal entry) and ranked by magnitude. On the raw signed
entries a percent matrix (diagonal 100) listed a 0.5% leak among its "strong"
pairs and reported 12.0 as the largest spill, and an over-compensating -0.35
entry ranked below a +0.04 one, so that matrix read as "no strong spillover".

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
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "spillover matrix contains non-finite (NaN/inf) values"
        )
    k = arr.shape[0]
    chans = list(channels) if channels is not None else []
    if len(chans) != k:
        raise ValueError(
            f"channels has {len(chans)} names but matrix is {k}x{k}; "
            "they must match"
        )
    return arr, chans


def _relative(arr, chans) -> np.ndarray:
    """Each row as a fraction of its diagonal entry (the source's own
    signal). A diagonal that is not positive is not a spillover matrix."""
    d = np.diag(arr)
    bad = [chans[i] for i in range(len(d)) if not d[i] > 0]
    if bad:
        raise ValueError(
            f"spillover matrix diagonal is not positive for {bad}; the "
            "diagonal holds each source's own signal (1, or 100 in percent)")
    return arr / d[:, None]


def spillover_metrics(matrix, channels) -> dict:
    """Summarise off-diagonal spillover in a compensation matrix.

    ``matrix[i, j]`` = fraction of source channel ``i`` leaking into
    destination channel ``j``. Each row is first divided by its diagonal
    entry, so the values are fractions of the source's own signal whatever
    the file's units (a percent matrix gives the same metrics as its
    fraction form). Pairs are ranked by MAGNITUDE: a negative entry is an
    over-compensation as large as its absolute value.

    Returns a dict with::

        {'max_offdiag': float,                 # largest |off-diagonal|
         'max_pair': (src_channel, dst_channel),
         'mean_offdiag': float,                # mean |off-diagonal|
         'strong_pairs': [(src, dst, value), ...],  # |value| >= 0.10,
                                                    # signed, |value| desc
         'n_channels': k}

    Raises ``ValueError`` for None / non-square inputs, a channel-count
    mismatch, or a diagonal entry that is not positive.
    """
    arr, chans = _as_matrix(matrix, channels)
    k = arr.shape[0]
    rel = _relative(arr, chans)

    off_mask = ~np.eye(k, dtype=bool)
    off_vals = np.abs(rel[off_mask])

    if off_vals.size == 0:  # 1x1 matrix: no off-diagonal entries
        return {
            "max_offdiag": 0.0,
            "max_pair": None,
            "mean_offdiag": 0.0,
            "strong_pairs": [],
            "n_channels": k,
        }

    flat_idx = int(np.argmax(off_vals))
    rows, cols = np.where(off_mask)
    src_i, dst_j = int(rows[flat_idx]), int(cols[flat_idx])

    # Strong pairs: off-diagonal magnitudes at or above the threshold, desc.
    strong: list[tuple[str, str, float]] = []
    for i, j in zip(*np.where(off_mask & (np.abs(rel) >= STRONG_THRESHOLD)),
                    strict=True):
        strong.append((chans[i], chans[j], float(rel[i, j])))
    strong.sort(key=lambda t: abs(t[2]), reverse=True)

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
    is annotated with its spill as a fraction of the source's own signal (the
    row divided by its diagonal entry, signed). The colour is the MAGNITUDE,
    clipped to [0, 0.3] so the 1.0 diagonal does not wash out the
    off-diagonal spillover that matters for QC, and a -0.35 entry colours as
    strongly as +0.35 rather than as 0. The caller is responsible for saving
    or embedding the figure; this function never calls ``show()``.
    """
    arr, chans = _as_matrix(matrix, channels)
    arr = _relative(arr, chans)
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
    im = ax.imshow(np.abs(arr), cmap="magma", vmin=0.0, vmax=vmax,
                   aspect="equal")

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
                color="white" if abs(val) < thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="|spillover| / source signal (clipped)")
    fig.tight_layout()
    return fig
