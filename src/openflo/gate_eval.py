"""Gate evaluation: ``gate_to_mask`` evaluates one gate dict against a
DataFrame, ``cumulative_gate_mask`` a gate and all its ancestors, and the
unknown-population checks (``unknown_population_reason``,
``broken_booleans``, ...) say which populations cannot be computed.

Core module: numpy, pandas and the standard library. No Tk, and it does not
import ``openflo.pipeline``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .autoclean import autoclean_keep_mask
from .gate_lineage import lineage
from .scales import OPEN
from .wsp_gate_coords import polygon_edge_fns as _polygon_edge_fns

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
    from typing import Any

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# these messages.
log = logging.getLogger('openflo.pipeline')


# ── Gate model & evaluator ────────────────────────────────────────────────────
#
# A gate is a JSON-friendly dict describing a region in event space. The same
# schema is consumed by the pipeline (this module) and authored by the GUI
# editor. Five kinds today:
#
#   {"kind": "threshold", "channel": str, "value": float}
#       1D one-sided:  x >= value  (a Gating-ML min-only RectangleGate)
#
#   {"kind": "interval",  "channel": str, "lo": float, "hi": float}
#       1D two-sided:  lo <= x < hi  (half-open)
#
#   {"kind": "rect",      "x_channel": str, "y_channel": str,
#                          "x0": float, "x1": float, "y0": float, "y1": float}
#       2D axis-aligned rectangle, half-open on both axes.
#
#   An open side is the sentinel -1e12 / +1e12 (JSON cannot carry +-inf; see
#   wsp_reader._finite_bound), not +-inf. Data never comes near 1e12, so
#   for every finite event an open side is unbounded. A non-finite event
#   stays outside every bounded comparison: +inf fails `x < 1e12`, as it
#   fails any finite upper bound, and is inside only a threshold (tests/
#   test_gate_nonfinite_events.py pins this).
#
#   {"kind": "polygon",   "x_channel": str, "y_channel": str,
#                          "vertices": [[x, y], ...]}
#       2D polygon (>=3 vertices). Membership via matplotlib.path.Path.
#
# Coordinates are in the SAME space as the data being gated (typically
# post-transform — logicle / log). FlowJo .wsp gates round-trip in this space
# already, matching apply_threshold_gates() semantics.

def depends_on_flowjo_placeholder(gates_by_id: Mapping[str, Mapping] | None,
                                  gid: str,
                                  _depth: int = 0) -> bool:
    """True when gate `gid`'s population rests on a 'flowjo_ellipse'
    placeholder (a FlowJo ellipse OpenFlo could not convert): `gid` is one,
    one is among its ancestors, or, for a boolean, one of its operands rests
    on one. Such a population is unknown, not empty. Cycle-safe: booleans
    nested past gate_to_mask's own depth limit cannot be followed, so they
    count as unknown too (True), never as shown not to rest on one."""
    if _depth > 20:
        return True
    gates_by_id = gates_by_id or {}
    for cur in lineage(gates_by_id, gid):
        g = gates_by_id[cur]
        if g.get('kind') == 'flowjo_ellipse':
            return True
        if g.get('kind') == 'boolean' and any(
                depends_on_flowjo_placeholder(gates_by_id, op, _depth + 1)
                for op in g.get('operands') or ()):
            return True
    return False


def rests_on_broken_boolean(gates_by_id: Mapping[str, Mapping] | None,
                            gid: str,
                            _depth: int = 0) -> bool:
    """True when gate `gid`'s population rests on a boolean gate that cannot
    be evaluated -- one with no operands, or naming a gate not in
    `gates_by_id`, which gate_to_mask makes select nothing: `gid` is one, one
    is among its ancestors, or, for a boolean, one of its operands rests on
    one. remap_operands leaves exactly that shape when it cannot resolve an
    operand. Such a population is unknown, not empty: NOT over it admitted
    every event, and OR with it was silently the other operand alone.
    Cycle-safe; nested past gate_to_mask's depth limit it counts as unknown
    (True), as in depends_on_flowjo_placeholder."""
    if _depth > 20:
        return True
    gates_by_id = gates_by_id or {}
    for cur in lineage(gates_by_id, gid):
        g = gates_by_id[cur]
        if g.get('kind') == 'boolean':
            ops = g.get('operands') or []
            unresolved = not ops or any(op not in gates_by_id for op in ops)
            if unresolved or any(
                    rests_on_broken_boolean(gates_by_id, op, _depth + 1)
                    for op in ops):
                return True
    return False


def broken_booleans(gates_by_id: Mapping[str, Mapping],
                    ) -> tuple[list[str], list[str]]:
    """``(lost, built_on)``: ids of the boolean gates in `gates_by_id` that
    select nothing because they cannot be evaluated. `lost` have no operands
    or name a gate that is not there; `built_on` rest on such a gate
    (rests_on_broken_boolean)."""
    lost, built_on = [], []
    for gid, g in gates_by_id.items():
        if g.get('kind') != 'boolean':
            continue
        ops = g.get('operands') or []
        if not ops or any(op not in gates_by_id for op in ops):
            lost.append(gid)
        elif rests_on_broken_boolean(gates_by_id, gid):
            built_on.append(gid)
    return lost, built_on


def _own_unknown_reason(gates_by_id: Mapping[str, Mapping], g: Mapping,
                        columns: Collection[str] | None = None,
                        ) -> str | None:
    """Why gate `g` itself cannot be evaluated (None when it can): the
    regions gate_to_mask fails closed on because they are unknown. With
    `columns` (the sample's), a gate on a channel the sample lacks is one."""
    kind = g.get('kind')
    if g.get('scale_unknown'):
        return 'imported onto a channel of unknown scale'
    if columns is not None:
        gone = missing_gate_channels(g, columns)
        if gone:
            return ('channel ' + ', '.join(gone) + ' is not in this sample')
        if kind in ('cluster', 'category'):
            # A population of a label column the sample does not have (its
            # clustering was not restored with a session, say) selects
            # nothing, but nobody measured that 0: every cluster of a
            # reopened autosave read 0 events under a group of 20,000.
            col = g.get('channel') or ('cluster' if kind == 'cluster' else '')
            if col not in set(columns):
                return f"column '{col}' is not in this sample"
    if kind == 'flowjo_ellipse':
        return 'FlowJo ellipse not converted'
    if kind == 'boolean':
        ops = g.get('operands') or []
        if not ops or any(op not in gates_by_id for op in ops):
            return 'boolean gate lost an operand'
    if kind == 'ellipsoid':
        mean, cov = g.get('mean'), g.get('cov')
        if mean is not None and cov is not None:
            cov = np.asarray(cov, dtype=float)
            if cov.shape == (2, 2) and not _positive_definite(cov):
                return 'ellipse has no valid covariance'
    return None


def unknown_population_reason(gates_by_id: Mapping[str, Mapping] | None,
                              gid: str,
                              _depth: int = 0,
                              columns: Collection[str] | None = None,
                              ) -> str | None:
    """Why gate `gid`'s population cannot be computed, or None when it can.

    A population is unknown when it, an ancestor, or (for a boolean) an
    operand rests on a gate whose region is unknown: a FlowJo ellipse
    OpenFlo could not convert, a boolean that lost an operand, or an ellipse
    whose covariance describes no ellipse. gate_to_mask makes such gates
    select nothing (fail-closed), so their count is 0 events -- but that 0 is
    not a measurement, and statistics report the population as missing (NA)
    with this reason. Cycle-safe; nested past gate_to_mask's depth limit a
    population counts as unknown too, as in depends_on_flowjo_placeholder.

    `columns`, the sample's columns, adds gates on a channel the sample
    lacks (missing_gate_channels), and cluster / category populations of a
    label column it lacks; without it they are not judged."""
    if _depth > 20:
        return 'gates nest too deep to follow'
    gates_by_id = gates_by_id or {}
    for cur in lineage(gates_by_id, gid):
        g = gates_by_id[cur]
        why = _own_unknown_reason(gates_by_id, g, columns)
        if why:
            return why
        if g.get('kind') == 'boolean':
            for op in g.get('operands') or ():
                why = unknown_population_reason(gates_by_id, op, _depth + 1,
                                                columns=columns)
                if why:
                    return why
    return None


def _measured_in(df: pd.DataFrame, gates_by_id: Mapping[str, Mapping] | None,
                 operand_ids: Iterable[str] | None,
                 _depth: int = 0) -> np.ndarray:
    """Events that HAVE a measurement in every channel the given gates touch.

    Used by boolean NOT: an event with no value in the operand's channel is
    not "negative for it", it is unmeasured, and negating a mask would
    otherwise adopt it. Recurses through nested boolean operands. Channels
    that are absent or non-numeric (a cluster or category label) place no
    requirement — there is nothing to be non-finite about.

    The channel keys are read inline rather than via gating.gate_channels:
    gating imports from here, and one shared helper is not worth the cycle.
    """
    n = len(df)
    valid = np.ones(n, dtype=bool)
    if gates_by_id is None or _depth > 20:
        return valid
    for gid in operand_ids or ():
        gate = gates_by_id.get(gid)
        if gate is None:
            continue
        if gate.get('kind') == 'boolean':
            valid &= _measured_in(df, gates_by_id, gate.get('operands'),
                                  _depth + 1)
            continue
        for key in ('channel', 'x_channel', 'y_channel'):
            ch = gate.get(key)
            if not ch:
                continue
            if ch not in df.columns:
                continue
            col = df[ch]
            if not pd.api.types.is_numeric_dtype(col):
                continue
            valid &= np.isfinite(np.asarray(col.values, dtype=float))

    return valid


def points_in_polygon(verts: Any, pts: Any) -> np.ndarray:
    """Point-in-polygon for gating, via flowutils' compiled ``gating_c``.

    Not ``matplotlib.path.Path.contains_points``: the choice rests on which
    rule is actually CORRECT on the boundary, not merely faster -- a point
    exactly on an edge is neither strictly inside nor outside, so the tie-break
    is a choice, and matplotlib's choice is wrong for gating:

    * **It double-counts a shared edge.** Two gates meeting on a line both
      claim every point on it. Across a quadrant, 902 of 20 000 integer events
      landed in two populations at once, so the four quadrants summed above
      100%. flowutils assigns each such point to exactly one gate.
    * **It depends on WINDING.** The same square drawn clockwise and
      anticlockwise gives different answers on its boundary. Users draw gates
      in both directions, so the same shape could gate differently depending
      on the mouse path that produced it. flowutils is winding-invariant.
    * **Its rule is not even self-consistent**: for an axis-aligned square it
      includes three edges and excludes the bottom; for the same square rotated
      45 degrees it includes all four.

    flowutils implements the standard half-open convention -- a point on a
    "lower/left" edge is inside, one on an "upper/right" edge is outside -- so
    adjacent gates tile without gaps or overlaps, which is the property gating
    actually needs.

    Impact: on continuous float data the two agree EXACTLY (0 of 200 000
    events differed). They diverge only where events land on exact
    coordinates, i.e. integer ``$DATATYPE I`` channels (6.96% of events), and
    there matplotlib is the one double-counting. The golden baseline is the
    same under either.

    It is also 5-12x faster, which is a side benefit rather than the reason:
    at 1M events a normal gate takes 32 ms against matplotlib's 172, and a
    tight gate 14 ms against 163. A bounding-box prefilter does not pay on top
    -- building the mask costs more than the crossing test it saves (48 vs
    32 ms).
    """
    verts = np.asarray(verts, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    try:
        from flowutils import gating as _fu_gating
        return np.asarray(_fu_gating.points_in_polygon(verts, pts), dtype=bool)
    except Exception:                                        # noqa: BLE001 - any flowutils failure falls back to matplotlib with a logged warning, so the gate still evaluates
        # flowutils is a hard pinned dependency, so this should not happen;
        # falling back keeps gating working rather than failing the run, at
        # the cost of the boundary behaviour documented above.
        from matplotlib.path import Path as _MplPath
        log.warning("  [gate] flowutils polygon test unavailable — "
                    "falling back to matplotlib (boundary events may differ)")
        return _MplPath(verts).contains_points(pts)


# The channel keys of the geometric gate kinds: a gate naming a channel the
# data lacks cannot be evaluated on it.
_GEOMETRIC_CHANNEL_KEYS: dict[Any, tuple[str, ...]] = {
    'threshold': ('channel',), 'interval': ('channel',),
    'rect': ('x_channel', 'y_channel'),
    'polygon': ('x_channel', 'y_channel'),
    'ellipsoid': ('x_channel', 'y_channel')}


def missing_gate_channels(gate: Mapping[str, Any],
                          columns: Iterable[str]) -> list[Any]:
    """The channels geometric gate `gate` names that `columns` lacks."""
    cols = set(columns)
    return [gate.get(k) for k in _GEOMETRIC_CHANNEL_KEYS.get(gate.get('kind'), ())
            if gate.get(k) and gate.get(k) not in cols]


def _channel_missing(gate: Mapping[str, Any], channels: Sequence[str], n: int,
                     df: pd.DataFrame | None = None) -> np.ndarray:
    """All-False mask for a geometric gate whose channel the data lacks.

    Admitting every event instead would make a sample whose FL1-A is named
    FL1-H report the gated population at 100 % and gate its children on all
    events. The region cannot be evaluated: fail closed, and statistics show
    the population as NA with the reason (unknown_population_reason, given
    the sample's columns)."""
    if df is not None:
        channels = [c for c in channels if c not in df.columns]
    log.warning("  [gate] %s: channel(s) %s not in this sample -- admitting "
                "no events (fail-closed)", gate.get('name') or gate.get('kind'),
                ", ".join(map(repr, channels)))
    return np.zeros(n, dtype=bool)


def _two_channel_points(gate: Mapping[str, Any], df: pd.DataFrame) -> np.ndarray:
    """The events' (x, y) values on a 2D gate's channels, as an n x 2 float
    array."""
    return np.column_stack([
        np.asarray(df[gate['x_channel']].values, dtype=float),
        np.asarray(df[gate['y_channel']].values, dtype=float),
    ])


def _lacks_two_channels(gate: Mapping[str, Any], df: pd.DataFrame) -> bool:
    """True when the data lacks a 2D gate's x or y channel."""
    return (gate['x_channel'] not in df.columns
            or gate['y_channel'] not in df.columns)


def _threshold_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                    gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                    ) -> np.ndarray:
    ch = gate['channel']
    if ch not in df.columns:
        return _channel_missing(gate, [ch], n)
    # AT OR ABOVE the cut. A threshold is what a min-only Gating-ML
    # RectangleGate imports as and what WspWriter writes back as
    # `gating:min`; Gating-ML's rectangle is min <= x, and FlowJo counts
    # it so. `x > value` would leave out every event sitting on the cut:
    # integer events 995..1005 with a minimum of 1000 would give 5 per
    # value where FlowJo gives 6. It also agrees with the interval
    # [value, open) of the same gate.
    # `.values` is typed ndarray | ExtensionArray; a gated channel is numeric.
    return np.asarray(df[ch].values >= float(gate['value']))  # pyright: ignore[reportOperatorIssue]


def _interval_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                   gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                   ) -> np.ndarray:
    ch = gate['channel']
    if ch not in df.columns:
        return _channel_missing(gate, [ch], n)
    vals = np.asarray(df[ch].values, dtype=float)
    # HALF-OPEN [lo, hi): the lower bound is inside, the upper bound is
    # not. Fully-open bounds would silently LOSE every event sitting
    # exactly on a boundary -- adjacent intervals [0,10] and [10,20] would
    # put a value of exactly 10 in NEITHER, so those events would vanish
    # from both populations and the percentages not sum. Measured at 14.3% of
    # events on integer $DATATYPE I data; float data is unaffected (0 of
    # 200k), because continuous values never land exactly on a bound.
    # Also matches Gating-ML's RectangleGate convention and the polygon
    # rule in points_in_polygon, so the same region gates identically
    # however it is expressed.
    return (vals >= float(gate['lo'])) & (vals < float(gate['hi']))


def _rect_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
               gates_by_id: Mapping[str, Mapping] | None, _depth: int,
               ) -> np.ndarray:
    if _lacks_two_channels(gate, df):
        return _channel_missing(gate, [gate['x_channel'], gate['y_channel']],
                                n, df)
    xs = np.asarray(df[gate['x_channel']].values, dtype=float)
    ys = np.asarray(df[gate['y_channel']].values, dtype=float)
    # Half-open on both axes -- see the reasoning on `interval` above.
    # Adjacent rectangles must partition, and a rect must agree with the
    # identical region drawn as a polygon.
    return ((xs >= float(gate['x0'])) & (xs < float(gate['x1'])) &
            (ys >= float(gate['y0'])) & (ys < float(gate['y1'])))


def _polygon_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                  gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                  ) -> np.ndarray:
    if _lacks_two_channels(gate, df):
        return _channel_missing(gate, [gate['x_channel'], gate['y_channel']],
                                n, df)
    verts = np.asarray(gate['vertices'], dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        log.info(f"  [gate] polygon: malformed vertices (shape={verts.shape}) — skipped")
        return np.ones(n, dtype=bool)
    pts = _two_channel_points(gate, df)
    edge_fns = _polygon_edge_fns(gate)
    if edge_fns is not None:
        # Imported from FlowJo: the edges are straight on FlowJo's axes
        # (see _flowjo_edge_axis), so vertices and events go there first.
        verts = verts.copy()
        for i, f in enumerate(edge_fns):
            if f is not None:
                verts[:, i] = f[0](np.clip(verts[:, i], -OPEN, OPEN))
                pts[:, i] = f[0](pts[:, i])
    return points_in_polygon(verts, pts)


def _ellipsoid_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                    gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                    ) -> np.ndarray:
    # Gating-ML 2.0 EllipsoidGate: an event is inside when its
    # squared Mahalanobis distance from the mean is within
    # `distance_sq`:  (p-µ)ᵀ Σ⁻¹ (p-µ) ≤ distance_sq.
    if _lacks_two_channels(gate, df):
        return _channel_missing(gate, [gate['x_channel'], gate['y_channel']],
                                n, df)
    mean = np.asarray(gate['mean'], dtype=float)
    cov  = np.asarray(gate['cov'], dtype=float)
    dist_sq = float(gate.get('distance_sq', 4.0))
    if mean.shape != (2,) or cov.shape != (2, 2):
        log.info("  [gate] ellipsoid: malformed mean/cov — skipped")
        return np.ones(n, dtype=bool)
    # A covariance that is singular, not positive definite or not finite
    # describes no ellipse. A singular one was 'skipped' and admitted
    # every event, and so did a negative-definite one, whose quadratic
    # form is never positive. Its region is unknown: fail closed, as a
    # FlowJo ellipse that cannot be converted does (the WSP exporter
    # says so for both kinds, _wsp_export_gate).
    if not _positive_definite(cov):
        log.warning("  [gate] ellipsoid: covariance is singular or not "
                    "positive definite — admitting no events (fail-closed)")
        return np.zeros(n, dtype=bool)
    pts = _two_channel_points(gate, df)
    try:
        # flowutils' compiled gating_c, the same extension backing
        # points_in_polygon. Agrees with the hand-rolled einsum quadratic
        # form on real data — 0 of 1,000,000 random events differ — and is
        # ~1.4x faster: re-measured 2026-09-10, 42 -> 31 ms at 1M events.
        #
        # They are NOT bit-identical on the boundary itself: of 512
        # points constructed to sit exactly
        # on the ellipse, the two disagree on 92. Those points compute d^2
        # in the range 3.99999999999999867 .. 4.00000000000000178, i.e.
        # they straddle dist_sq by one or two ulp, and the two orderings of
        # the same arithmetic break the tie differently.
        #
        # That is harmless HERE, and it is worth being clear why, because
        # the identical situation was NOT harmless for rect and interval
        # gates. An ellipse contour is not lattice-aligned: integer
        # $DATATYPE I events land exactly on an interval or rectangle
        # bound constantly (14.3% of events, which is why those became
        # half-open), but on 200,000 integer events the two ellipsoid
        # paths disagreed 0 times, and a 13x13 integer lattice holds no
        # point at exactly d^2 == dist_sq at all. So the fallback below
        # cannot silently move a real event across this boundary. It takes the
        # covariance itself and inverts internally, so the singular case
        # still surfaces as LinAlgError and is handled below. Using one
        # library for both polygon and ellipsoid keeps
        # gate geometry semantics in a single place.
        from flowutils import gating as _fu_gating
        return np.asarray(
            _fu_gating.points_in_ellipsoid(cov, mean, dist_sq, pts),
            dtype=bool)
    except np.linalg.LinAlgError:
        log.warning("  [gate] ellipsoid: singular covariance — admitting "
                    "no events (fail-closed)")
        return np.zeros(n, dtype=bool)
    except Exception:                                    # noqa: BLE001 - any other flowutils failure falls back to the numpy quadratic form, so the gate is never lost
        # Never lose a gate to an unexpected backend failure: fall back to
        # the quadratic form, which needs only numpy.
        inv = np.linalg.inv(cov)
        d = pts - mean
        return np.einsum('ij,jk,ik->i', d, inv, d) <= dist_sq


def _cluster_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                  gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                  ) -> np.ndarray:
    # Membership in one clustering label. Unlike the geometric gates,
    # a missing column means the population is undefined for this
    # sample, so it selects NOTHING (empty) rather than no-op all-True
    # — an unclustered sample shouldn't masquerade as "all events".
    ch = gate.get('channel', 'cluster')
    if ch not in df.columns:
        log.info(f"  [gate] cluster: column '{ch}' not in data — empty")
        return np.zeros(n, dtype=bool)
    return np.asarray(df[ch].values == gate.get('cluster_id'))


def _category_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                   gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                   ) -> np.ndarray:
    # Membership in a categorical label column (e.g. cell-cycle phase
    # in a 'cell_cycle' column). Like 'cluster', a missing column means
    # the population is undefined → selects nothing.
    ch = gate.get('channel')
    if not ch or ch not in df.columns:
        log.info(f"  [gate] category: column '{ch}' not in data — empty")
        return np.zeros(n, dtype=bool)
    return np.asarray(df[ch].values == gate.get('value'))


def _flowjo_ellipse_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                         gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                         ) -> np.ndarray:
    # An ellipse drawn in FlowJo on an axis OpenFlo cannot convert (see
    # wsp_ellipse_gate). Its region is unknown, so it selects nothing:
    # the unknown-kind default (every event) would hand each population
    # under it every event the ellipse excluded.
    log.info("  [gate] flowjo_ellipse: FlowJo ellipse not converted — empty")
    return np.zeros(n, dtype=bool)


def _unusable_operands_warning(operands: Sequence[str],
                               gates_by_id: Mapping[str, Mapping],
                               df: pd.DataFrame) -> str | None:
    """The warning to log when a boolean gate's operands, all present in
    `gates_by_id`, cannot be combined, or None when they can."""
    # An operand resting on a FlowJo ellipse OpenFlo could not convert
    # (the ellipse, a gate under it, or a boolean built on one) has an
    # unknown region, and so does every combination of it. The ellipse
    # selects nothing, which keeps it and its subpopulations empty, but
    # its complement is not "everything" nor its union "the other
    # operand": measured, NOT(ellipse) admitted all 20,000 events and
    # OR(CD3+, ellipse) all of CD3+.
    unknown = [gid for gid in operands
               if depends_on_flowjo_placeholder(gates_by_id, gid)]
    if unknown:
        return (f"  [gate] boolean: operand(s) {unknown} rest on a "
                "FlowJo ellipse that was not converted (or nest too "
                "deep to follow) — admitting no events (fail-closed)")
    # Likewise an operand resting on a boolean that cannot be evaluated
    # (no operands, or one deleted): that boolean fails closed, and its
    # emptiness is unknown, not a measured zero. Measured: with B empty,
    # NOT(B) admitted all 10,000 events, OR(CD3+, B) was CD3+ alone and
    # NOT(CD3+, B) was NOT CD3+.
    broken = [gid for gid in operands
              if rests_on_broken_boolean(gates_by_id, gid)]
    if broken:
        return (f"  [gate] boolean: operand(s) {broken} rest on a "
                "boolean gate that cannot be evaluated — admitting "
                "no events (fail-closed)")
    # And an operand resting on an ellipse whose covariance describes no
    # ellipse: that ellipse selects nothing for the same reason, so NOT
    # over it admitted every event (unknown_population_reason).
    unknown = [gid for gid in operands
               if unknown_population_reason(gates_by_id, gid,
                                            columns=df.columns)]
    if unknown:
        return (f"  [gate] boolean: operand(s) {unknown} rest on a "
                "gate whose region is unknown — admitting no events "
                "(fail-closed)")
    return None


def _boolean_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                  gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                  ) -> np.ndarray:
    # Combine OTHER gates' cumulative masks. op ∈ {and, or, not}; 'not'
    # negates the OR of its operands (so a single operand → plain NOT).
    op = gate.get('op', 'and')
    operands = gate.get('operands', []) or []
    # A boolean gate that cannot resolve its operands must NOT admit every
    # event. Returning all-True made a "NOT X" population the ENTIRE
    # sample — measured, `NOT (CD3 > 2.0)` kept 20,000 events where the
    # correct answer was 10,002, so every CD3-positive event was admitted
    # into the CD3-negative population. Fail CLOSED, matching both the
    # 'cluster'/'category' missing-column branches above and the
    # fail-closed exception handler in apply_region_gates: an empty
    # population is loud, a superset reads as success.
    if gates_by_id is None or not operands or _depth > 20:
        log.warning("  [gate] boolean: operands could not be resolved — "
                    "admitting no events (fail-closed)")
        return np.zeros(n, dtype=bool)
    # ONE missing operand is enough to fail closed. Dropping it and
    # combining the rest changes the population without a word: measured,
    # deleting CD4+ turned AND(CD3+, CD4+) from 4,983 events into CD3+
    # alone, 10,044; a NOT over two operands becomes the wider NOT of one,
    # and an OR quietly narrows.
    missing = [gid for gid in operands if gid not in gates_by_id]
    if missing:
        log.warning(f"  [gate] boolean: operand(s) {missing} no longer "
                    "exist — admitting no events (fail-closed)")
        return np.zeros(n, dtype=bool)
    warning = _unusable_operands_warning(operands, gates_by_id, df)
    if warning:
        log.warning(warning)
        return np.zeros(n, dtype=bool)
    masks = [cumulative_gate_mask(gates_by_id, gid, df, _depth + 1)
             for gid in operands]
    out = masks[0].copy()
    if op not in ('or', 'not'):    # default: and
        for m in masks[1:]:
            out &= m
        return out
    for m in masks[1:]:
        out |= m
    if op == 'or':
        return out
    # Negation turns "could not be measured" into "belongs here". An
    # event whose value is NaN fails `> threshold`, so it is correctly
    # OUTSIDE the positive gate — and then the complement swept it in.
    # Measured: 2,000 unmeasurable events made up 20% of a CD3-negative
    # population, and because the two gates still summed to the sample
    # total the result looked self-consistent. Non-finite is not
    # negative, so those events belong to neither side.
    return ~out & _measured_in(df, gates_by_id, operands)


def _autoclean_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                    gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                    ) -> np.ndarray:
    # Recipe gate (no coordinates): the AND of every enabled cleaning
    # method, recomputed from THIS df. See autoclean_keep_mask.
    return autoclean_keep_mask(gate, df)


def _group_mask(gate: Mapping[str, Any], df: pd.DataFrame, n: int,
                gates_by_id: Mapping[str, Mapping] | None, _depth: int,
                ) -> np.ndarray:
    # Pure organisational container (e.g. a 'Phenograph (N)' folder over
    # cluster populations): no geometry, never filters — children carry
    # the actual masks.
    return np.ones(n, dtype=bool)


# Each gate kind's evaluator: (gate, df, n, gates_by_id, _depth) -> mask.
_MASK_BY_KIND = {
    'threshold': _threshold_mask,
    'interval': _interval_mask,
    'rect': _rect_mask,
    'polygon': _polygon_mask,
    'ellipsoid': _ellipsoid_mask,
    'cluster': _cluster_mask,
    'category': _category_mask,
    'flowjo_ellipse': _flowjo_ellipse_mask,
    'boolean': _boolean_mask,
    'autoclean': _autoclean_mask,
    'group': _group_mask,
}


def gate_to_mask(gate: Mapping[str, Any], df: pd.DataFrame,
                 gates_by_id: Mapping[str, Mapping] | None = None,
                 _depth: int = 0) -> np.ndarray:
    """Evaluate one gate dict against a DataFrame. Returns a 1-D bool ndarray
    aligned to df. A gate whose channel the data lacks selects nothing
    (fail-closed) and its population is unknown (unknown_population_reason
    with `columns`); a geometric gate also logs a warning. Unknown kinds
    no-op.

    `gates_by_id` is only needed for 'boolean' gates, whose operands are
    OTHER gates in the same sample (resolved via their cumulative masks);
    `_depth` guards against operand cycles."""
    kind = gate.get('kind')
    n    = len(df)
    if gate.get('scale_unknown'):
        # Imported in FlowJo's linear units onto a channel whose display
        # scale is unknown: its coordinates could not be converted, and
        # compared as stored they name some other region. Fail closed; the
        # population is unknown (unknown_population_reason), not empty.
        log.warning("  [gate] %s: imported onto a channel of unknown scale "
                    "-- admitting no events (fail-closed)",
                    gate.get('name') or kind)
        return np.zeros(n, dtype=bool)
    evaluate = _MASK_BY_KIND.get(kind) if isinstance(kind, str) else None
    if evaluate is None:
        log.info(f"  [gate] unknown kind {kind!r} — skipped")
        return np.ones(n, dtype=bool)
    return evaluate(gate, df, n, gates_by_id, _depth)


def cumulative_gate_mask(gates_by_id: Mapping[str, Mapping], gid: str,
                         df: pd.DataFrame, _depth: int = 0,
                         overrides: Mapping[str, np.ndarray] | None = None,
                         cache: MutableMapping[str, np.ndarray] | None = None,
                         ) -> np.ndarray:
    """AND every gate's mask from `gid` up the parent chain to the root.
    Cycle-safe.

    The chain ALWAYS includes every ancestor regardless of each one's
    `enabled` flag: the toggle controls visibility ("draw this gate's
    highlight overlay") and pipeline inclusion separately, not whether a
    gate participates in defining the population at this node. So the
    cumulative meaning of leaf C inside parent P stays `P AND C` even
    when P's highlight is hidden.

    `gates_by_id` is a dict of gate_id -> gate_dict where each gate_dict
    may carry a 'parent_id' field naming another key (None = root).

    `overrides` (optional) maps gate_id -> a precomputed df-aligned bool mask
    used INSTEAD of evaluating that gate. The GUI uses it to inject cached
    auto-clean masks (whose recompute is expensive) so a chain that nests
    populations under an auto-clean root doesn't re-run the cleaning per node.

    `cache` (optional) memoises the CUMULATIVE mask per gate id. Walking a
    chain stops at the nearest cached ancestor, so evaluating a whole tree
    costs one `gate_to_mask` per gate instead of one per gate *per descendant*
    — the difference between O(gates x depth) and O(gates). This matters
    because a polygon gate is ~250x the cost of a threshold gate
    (`Path.contains_points` over every event), and real gating hierarchies are
    polygon-based: a 60-gate panel over 200k events drops from ~4.7 s to
    ~0.7 s.

    The caller OWNS the dict and MUST discard it whenever `df`, `gates_by_id`
    or `overrides` change — nothing here can detect that. Masks returned from
    a cache are SHARED; treat them as read-only. Omit `cache` (the default) and
    every call computes a freshly allocated mask.
    """
    if cache is not None:
        hit = cache.get(gid)
        if hit is not None:
            return hit

    # Walk up to the root (or to the nearest cached ancestor), recording the
    # chain. Re-applying it top-down is what lets every intermediate level be
    # cached on the way back — AND is associative, so the result is unchanged.
    chain = lineage(gates_by_id, gid)
    base = None
    if cache is not None:
        for i, cur in enumerate(chain):
            hit = cache.get(cur)
            if hit is not None:
                base, chain = hit, chain[:i]
                break

    mask = np.ones(len(df), dtype=bool) if base is None else base
    for cur_id in reversed(chain):
        g = gates_by_id[cur_id]
        if overrides is not None and cur_id in overrides:
            mask = mask & overrides[cur_id]
        else:
            mask = mask & gate_to_mask(g, df, gates_by_id, _depth)
        if cache is not None:
            cache[cur_id] = mask
    return mask


def _positive_definite(cov: Any) -> bool:
    """Whether `cov` is a covariance that describes an ellipse: finite,
    symmetric and positive definite (it has a Cholesky factor)."""
    cov = np.asarray(cov, dtype=float)
    finite_symmetric = (bool(np.isfinite(cov).all())
                        and bool(np.allclose(cov, cov.T)))
    try:
        np.linalg.cholesky(cov)
        factored = True
    except np.linalg.LinAlgError:
        factored = False
    return finite_symmetric and factored
