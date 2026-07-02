"""Headless gating helpers for OpenFlo.

Pure numpy/pandas utilities that build ready-to-insert *gate dicts* matching
the runtime schema evaluated by :func:`openflo.pipeline.gate_to_mask`:

- ``'polygon'``: ``x_channel``, ``y_channel``, ``vertices`` (list of ``[x, y]``)
- ``'threshold'``: ``channel``, ``value`` (an event is positive when the
  channel value is strictly ``> value``)

Every emitted gate also carries the common tree-metadata keys the GUI and
pipeline expect: ``id``, ``parent_id``, ``color``, ``enabled``.

No Tk, no matplotlib, no I/O — safe to call from scripts, tests, or the GUI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["singlet_gate", "fmo_threshold", "fmo_threshold_gate"]


def singlet_gate(
    df: pd.DataFrame,
    area: str = "FSC-A",
    height: str = "FSC-H",
    id: str = "singlets",
    parent_id: str | None = None,
    color: str = "#1f77b4",
    k: float = 3.0,
    height_pct: tuple[float, float] = (0.1, 99.9),
) -> dict:
    """Build a singlet-discrimination ``'polygon'`` gate in (area, height) space.

    For single cells the area channel (e.g. ``FSC-A``) is proportional to the
    height channel (``FSC-H``); doublets and aggregates carry more area per unit
    height and fall *off* that diagonal. We take the robust ratio
    ``r = height / area``, keep a band around its median
    (``median ± k · 1.4826 · MAD``), and express that band as a quadrilateral in
    ``(area, height)`` coordinates clipped to the populated height range.

    Args:
        df: event table containing the ``area`` and ``height`` columns.
        area: name of the area channel (the gate's ``x_channel``).
        height: name of the height channel (the gate's ``y_channel``).
        id: gate id.
        parent_id: id of the parent gate (``None`` = root).
        color: hex colour for the gate.
        k: half-width of the kept ratio band in robust standard deviations.
        height_pct: low/high percentiles of height used to clip the polygon to
            the populated range.

    Returns:
        A ``'polygon'`` gate dict ready to insert into the gate tree.

    Raises:
        KeyError: if ``area`` or ``height`` is not a column of ``df``.
        ValueError: if the ratio band is undefined (too little spread/data).

    Example:
        >>> g = singlet_gate(df, area="FSC-A", height="FSC-H")
        >>> mask = gate_to_mask(g, df)  # doctest: +SKIP
    """
    if area not in df.columns:
        raise KeyError(f"area channel {area!r} not in df columns")
    if height not in df.columns:
        raise KeyError(f"height channel {height!r} not in df columns")

    a = np.asarray(df[area].to_numpy(), dtype=float)
    h = np.asarray(df[height].to_numpy(), dtype=float)
    finite = np.isfinite(a) & np.isfinite(h) & (a > 0)
    a, h = a[finite], h[finite]
    if a.size < 2:
        raise ValueError("singlet_gate: not enough finite (area, height) events")

    ratio = h / a  # height / area; ~constant for proportional singlets
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 2:
        raise ValueError("singlet_gate: ratio undefined")

    med = float(np.median(ratio))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("singlet_gate: non-positive median ratio")

    mad = float(np.median(np.abs(ratio - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(ratio))
    if sigma <= 0:
        raise ValueError("singlet_gate: zero spread in ratio")

    r_lo = max(med - k * sigma, 1e-12)
    r_hi = med + k * sigma

    h_lo, h_hi = (float(v) for v in np.percentile(h, list(height_pct)))
    if h_hi <= h_lo:
        raise ValueError("singlet_gate: degenerate height range")

    # The band lies between the two rays  height = r_lo * area  and
    # height = r_hi * area, i.e.  area = height / r.  Clip to the populated
    # height range to form a quadrilateral (x = area, y = height). Wind the
    # vertices so the polygon is non-self-intersecting.
    verts = [
        [h_lo / r_hi, h_lo],
        [h_hi / r_hi, h_hi],
        [h_hi / r_lo, h_hi],
        [h_lo / r_lo, h_lo],
    ]
    verts = [[float(x), float(y)] for x, y in verts]

    return {
        "kind": "polygon",
        "x_channel": area,
        "y_channel": height,
        "vertices": verts,
        "parent_id": parent_id,
        "color": color,
        "enabled": True,
        "id": id,
    }


def fmo_threshold(fmo_values, percentile: float = 99.0) -> float:
    """Positivity cutoff from a Fluorescence-Minus-One (FMO) control.

    The FMO control lacks the marker of interest, so its channel signal is
    background. The given ``percentile`` of that background (default 99th) is
    the cutoff above which a stained sample is called positive.

    Args:
        fmo_values: 1-D array-like of the FMO control's channel values.
        percentile: percentile of the FMO distribution to use (0–100).

    Returns:
        The cutoff value as a float.

    Raises:
        ValueError: if there are no finite values.
    """
    vals = np.asarray(fmo_values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("fmo_threshold: no finite values")
    return float(np.percentile(vals, percentile))


def fmo_threshold_gate(
    fmo_df: pd.DataFrame,
    channel: str,
    percentile: float = 99.0,
    id: str | None = None,
    parent_id: str | None = None,
    color: str = "#d62728",
) -> dict:
    """Build a ``'threshold'`` gate from an FMO control on ``channel``.

    Computes the FMO cutoff (see :func:`fmo_threshold`) on ``fmo_df[channel]``
    and returns a 'threshold' gate; positive = above the cutoff, matching
    ``gate_to_mask`` (``value`` semantics are strictly greater-than).

    Args:
        fmo_df: the FMO control event table.
        channel: the marker channel to threshold.
        percentile: percentile of the FMO distribution for the cutoff.
        id: gate id; defaults to ``"<channel>+"`` when ``None``.
        parent_id: id of the parent gate (``None`` = root).
        color: hex colour for the gate.

    Returns:
        A ``'threshold'`` gate dict ready to insert into the gate tree.

    Raises:
        KeyError: if ``channel`` is not a column of ``fmo_df``.
        ValueError: if the channel has no finite values.
    """
    if channel not in fmo_df.columns:
        raise KeyError(f"channel {channel!r} not in fmo_df columns")
    cutoff = fmo_threshold(fmo_df[channel].to_numpy(), percentile=percentile)
    gate_id = id if id is not None else f"{channel}+"
    return {
        "kind": "threshold",
        "channel": channel,
        "value": cutoff,
        "parent_id": parent_id,
        "color": color,
        "enabled": True,
        "id": gate_id,
    }
