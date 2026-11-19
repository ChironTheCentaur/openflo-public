"""Pure-numpy point-in-polygon: a drop-in for
``flowutils.gating.points_in_polygon`` on platforms where FlowUtils'
compiled extension cannot be built (a browser runtime such as Pyodide).
``gate_eval.points_in_polygon`` uses FlowUtils when it imports and this
module otherwise, so both give the same answer on every point, the
boundary included.

The test is a vectorised port of FlowUtils' ``gate_helpers.c`` (Dan
Sunday's winding-number algorithm): the same bounding-box prefilter, the
same upward / downward crossing comparisons in the same order, and the same
``is_left`` expression, so a point exactly on an edge gets the reference's
half-open answer (two polygons sharing an edge claim each point on it once)
whichever way the polygon is wound. As in FlowUtils, a point is inside when
its winding count is odd.

Core module: numpy only. No Tk.

Ported from FlowUtils (https://github.com/whitews/FlowUtils), BSD-licensed,
Copyright (c) 2013-14, Duke University (the full notice, from FlowUtils'
LICENSE file, is in ``openflo.biexp_np``); its winding-number code is based on the C
implementation by Dan Sunday, Copyright 2000 softSurfer, 2012 Dan Sunday.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _bounding_box(verts: np.ndarray) -> tuple[float, float, float, float]:
    """(min_x, max_x, min_y, max_y), scanned as the reference scans it: a
    vertex that lowers the minimum is not tested against the maximum."""
    min_x = max_x = verts[0, 0]
    min_y = max_y = verts[0, 1]
    for vx, vy in verts[1:]:
        if vx < min_x:
            min_x = vx
        elif vx > max_x:
            max_x = vx
        if vy < min_y:
            min_y = vy
        elif vy > max_y:
            max_y = vy
    return min_x, max_x, min_y, max_y


def winding_counts(poly_vertices: Any, points: Any) -> np.ndarray:
    """The winding count of each of `points` (N x 2) about the closed
    polygon `poly_vertices` (V x 2); 0 for a point outside its bounding
    box. Same int32 values as FlowUtils' ``gating_c.points_in_polygon``."""
    verts = np.ascontiguousarray(poly_vertices, dtype=np.float64).reshape(-1, 2)
    pts = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    counts = np.zeros(len(pts), dtype=np.int32)
    if len(verts) == 0 or len(pts) == 0:
        return counts
    min_x, max_x, min_y, max_y = _bounding_box(verts)
    px = pts[:, 0]
    py = pts[:, 1]
    outside = (px < min_x) | (px > max_x) | (py < min_y) | (py > max_y)
    idx = np.flatnonzero(~outside)
    px = px[idx]
    py = py[idx]
    wind = np.zeros(len(idx), dtype=np.int32)
    n = len(verts)
    # Huge coordinates overflow is_left to inf / nan, as in the C code,
    # which is silent; the comparisons below give the same answers.
    with np.errstate(over='ignore', invalid='ignore'):
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[0] if i >= n - 1 else verts[i + 1]
            is_left = ((bx - ax) * (py - ay)) - ((px - ax) * (by - ay))
            starts_below = ay <= py
            # An upward crossing with the point left of the edge counts +1, a
            # downward crossing with it right of the edge -1.
            wind += (starts_below & (py < by) & (is_left > 0))
            wind -= (~starts_below & (by <= py) & (is_left < 0))
    counts[idx] = wind
    return counts


def points_inside(poly_vertices: Any, points: Any) -> np.ndarray:
    """Boolean mask of `points` (N x 2) inside polygon `poly_vertices`
    (V x 2): an odd winding count, FlowUtils' rule."""
    return winding_counts(poly_vertices, points) % 2 != 0
