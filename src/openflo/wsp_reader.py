"""The FlowJo .wsp workspace reader: ``WspReader`` (compensation matrices,
gates, sample groups and Time units of a workspace), the per-sample checks the
importers share (``wsp_sample_events``, ``wsp_event_mismatch``,
``xml_declares_entities``), and the gate-element helpers the reader and
openflo-compare both use: FlowJo's axis floors (``wsp_axis_floors``), its
display-space ellipses (``wsp_ellipse_gate``), polygon edge axes
(``wsp_polygon_edge_axes``) and compensated-parameter names
(``wsp_comp_channels``).

Core module: numpy and the standard library's ElementTree. No Tk, and it does
not import ``openflo.pipeline`` (pipeline imports it).
"""

from __future__ import annotations

import copy
import functools
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import numpy as np

from .channel_transforms import inverse_transform_values
from .exceptions import WspParseError
from .fcs_text import fcs_text_pairs as _fcs_text_pairs
from .fcs_text import fcs_timestep as _fcs_timestep
from .fcs_text import positive_float as _positive_float
from .fcs_text import scale_time_coords
from .fcs_text import wsp_time_gain as _wsp_time_gain
from .gate_text import WSP_BOOLEAN_OPS, describe_gate, resolve_wsp_dependent
from .scales import OPEN

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import numpy.typing as npt

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# the reader's messages.
log = logging.getLogger('openflo.pipeline')


def _finite_bound(value: float | str | None, sentinel: float) -> float | None:
    """A gate bound that JSON can carry.

    Gating-ML expresses an open side as "-INF"/"INF"; float() makes that ±inf,
    and json.dump writes `Infinity`, which RFC 8259 forbids. Python's reader
    accepts it, so such a session file is valid locally and unreadable to
    JSON.parse, jq, or any other consumer. ±1e12 is the sentinel this module
    already uses for a MISSING bound, for the same stated reason.
    """
    if value is None:
        return None
    v = float(value)
    if np.isnan(v):
        return None
    if np.isinf(v):
        return float(sentinel if v < 0 else abs(sentinel))
    return v


def _wsp_uri_path(uri: str | None) -> str | None:
    """Local path of a <DataSet uri> ('file:/C:/a%20b/x.fcs', 'file:C:/...',
    '/data/x.fcs' or a bare path), or None when it is empty."""
    if not uri:
        return None
    from urllib.parse import unquote
    p = unquote(uri[len('file:'):]) if uri.startswith('file:') else uri
    p = p.replace('\\', '/')
    if p.startswith('//'):                       # file:///C:/... or file:///data
        p = p[2:]
    if len(p) > 2 and p[0] == '/' and p[2] == ':':  # /C:/... on Windows
        p = p[1:]
    return p


# The FlowJo version an export says it is in the file's flowJoVersion: the
# version whose own files the export's form follows (full-path <Dependent>
# names, a <Transformations> block per sample). Other readers (FlowKit) parse
# it as a number, so a value such as 'OpenFlo-export-1.0' stops them. Which OpenFlo
# wrote the file is in the root's WSP_OPENFLO_ATTR attribute instead.
WSP_FLOWJO_VERSION = '10.8.1'
WSP_OPENFLO_ATTR = 'openfloVersion'


def wsp_written_by_openflo(root: ET.Element | None) -> bool:
    """Whether a parsed workspace `root` is one OpenFlo exported: it carries
    WSP_OPENFLO_ATTR, or (exports before that attribute) a flowJoVersion
    that starts 'OpenFlo'."""
    return root is not None and (bool(root.get(WSP_OPENFLO_ATTR)) or str(
        root.get('flowJoVersion') or '').startswith('OpenFlo'))


_ELLIPSE_VERTICES = 64


# FlowJo 10.10.2's default biex axis (width -100): a rectangle lower edge
# at or below -2000 was opened, one at -1000 or above kept (measured; see
# the export's floor note in wsp_writer, beside _OPEN_BOUND).
BIEX_OPENED_UPTO = -2000.0
BIEX_KEPT_FROM = -1000.0


def ellipse_polygon(mean: npt.ArrayLike, cov: npt.ArrayLike, distance_sq: float,
                    n: int = _ELLIPSE_VERTICES) -> list[list[float]] | None:
    """Vertices of an n-gon tracing {p : (p-mean)' cov^-1 (p-mean) = distance_sq},
    scaled so its area equals the ellipse's (it neither systematically gains
    nor loses events at the edge). None when `cov` is not positive definite."""
    try:
        L = np.linalg.cholesky(np.asarray(cov, dtype=float))
    except np.linalg.LinAlgError:
        return None
    r = np.sqrt(float(distance_sq)) * np.sqrt(2 * np.pi / (n * np.sin(2 * np.pi / n)))
    t = 2 * np.pi * np.arange(n) / n
    pts = np.asarray(mean, dtype=float)[:, None] + r * (L @ np.vstack([np.cos(t), np.sin(t)]))
    return [[float(x), float(y)] for x, y in pts.T]


# Import side of the same convention, for RECTANGLES only (1-D and 2-D),
# keyed by the axis each FlowJo <Sample> declares. Measured on linear axes of
# maxRange 262144, gain 1, for these minRange values -> (edge opened at or
# below, edge kept from):
_LINEAR_FLOORS_MEASURED = {
    0.0: (20.125, 64.0),       # 20.125 s on Time and 10 on SSC opened; 64 kept
    3.0: (10.0, 64.0),
    500.0: (500.0, 1000.0),
    -2000.0: (-2000.0, -1500.0),
}
# Every measured axis opened an edge AT its minRange; for another minRange
# only that much is applied and anything within 64 above it is logged.
# Polygons stay exact on import: FlowJo opens them from a different height
# than rectangles, and its in-range polygon rule is not identified (it is
# not exact point-in-polygon). On the six real round trips, clamping polygon
# events to FlowJo's axis range moved the largest FlowJo-OpenFlo gap only
# from 5.41% to 5.39%.


def wsp_axis_floors(sample_elem: ET.Element | None) -> dict[str, tuple[float, float]]:
    """{FlowJo parameter name: (open_at_or_below, kept_from)} for the axes a
    FlowJo ``<Sample>`` declares in ``<Transformations>`` (namespace-stripped
    tree), for the axis kinds the floor was measured on. Empty without such a
    block. A workspace OpenFlo exported (`wsp_written_by_openflo`) is not
    floored by the callers: it keeps OpenFlo's literal gate semantics on
    re-import, though it declares linear axes."""
    out: dict[str, tuple[float, float]] = {}
    if sample_elem is None:
        return out
    for block in sample_elem.findall('Transformations'):
        for t in block:
            p = t.find('parameter')
            name = p.get('name') if p is not None else None
            if not name:
                continue
            try:
                if (t.tag == 'linear' and float(t.get('maxRange', 'nan')) == 262144
                        and float(t.get('gain', 1)) == 1):
                    m = float(t.get('minRange', 0))
                    out[name] = _LINEAR_FLOORS_MEASURED.get(m, (m, m + 64.0))
                elif (t.tag == 'biex' and float(t.get('width', 'nan')) == -100
                      and float(t.get('maxRange', 'nan')) == 262144
                      and float(t.get('neg', 'nan')) == 0
                      and abs(float(t.get('pos', 'nan')) - 4.418539922) < 1e-6):
                    out[name] = (BIEX_OPENED_UPTO, BIEX_KEPT_FROM)
            except ValueError:
                continue
    return out


def wsp_axis_floor_lo(param: str | None, lo: float | None,
                      floors: Mapping[str, tuple[float, float]]) -> float | None:
    """A FlowJo rectangle lower bound as OpenFlo must apply it: -1e12 (open)
    when FlowJo opens it, else unchanged. Between the measured bounds the
    edge stays literal and the uncertainty is logged."""
    f = floors.get(param) if param else None
    if lo is None or f is None or lo <= -OPEN:
        return lo
    open_at, kept_from = f
    if lo <= open_at:
        return -OPEN
    if lo < kept_from:
        log.info(f"[WspReader] {param}: lower edge {lo:g} lies where FlowJo's "
                 f"axis floor is not measured ({open_at:g}..{kept_from:g}); kept literal")
    return lo


# An ellipse drawn in FlowJo is not a Gating-ML mean/covariance ellipse: FlowJo
# writes two <foci> and four <edge> points in its 256 x 256 display space, and
# each display coordinate / 256 is the fraction of that axis's TRANSFORMED range
# (the form FlowKit's WSPEllipsoidGate documents; FlowKit, BSD-3-Clause, is the
# reference for the geometry below). Converting one to data units therefore
# needs the axis transform its <Sample> declares. Measured against FlowJo's own
# counts in two FlowJo-written workspaces (FlowKit's test data):
# - linear 0..262144 axes: 51 of FlowJo's 51 events. A linear axis maps display
#   to data affinely, so the ellipse is exactly an ellipse in data units too.
# - logicle axes (T 262144, W 1, M 4.42, A 0): 5869 where FlowJo counts 6002,
#   under a parent OpenFlo counts 0.18% below FlowJo. Matching that parent
#   closed 27 of the 133 events; the rest is not explained, so the import says
#   so. FlowKit's own test of the same workspace (another sample) counts 7023
#   where FlowJo counts 7256. 64, 128 and 512 polygon vertices gave the same
#   count.
# - biex axes (FlowJo's default for fluorescence) use FlowJo's biex lookup
#   table, see flowjo_biex_lut. No FlowJo-drawn ellipse on a biex axis was
#   available to count against FlowJo. Against FlowKit, event for event on
#   the real ICS sample, with the ellipse at 20 placements on biex axes of
#   several parameter sets (some at the axis bottom or a corner): up to 139
#   events differ, at most 0.18% of the gate, all at the outline, with
#   OpenFlo slightly higher (a 64-gon of equal area here, FlowKit's 128-gon
#   inscribed).
# Any other axis -- log, fasinh, a linear gain other than 1, or no declared
# transform -- is not converted, and the gate becomes a 'flowjo_ellipse'
# placeholder: it admits no events, keeps its children under it, and keeps
# FlowJo's own points and axes so the export can write it back unchanged.
_FLOWJO_DISPLAY_BINS = 256.0
# FlowJo's biex lookup table has 4096 channels across the 256 display bins.
_FLOWJO_BIEX_CHANNELS = 4096


def _flowjo_log_root(b: float, w: float) -> float:
    """Root of 2 ln(d) + w d = 2 ln(b) - w b on (0, b], by Newton steps
    guarded by bisection. Ported line for line from FlowKit's `_log_root`
    (FlowKit, BSD-3-Clause, Copyright (c) 2018 Scott White), which ports
    cytolib's port of FlowJo's legacy Java code; kept identical so the table
    below matches FlowKit's and FlowJo's (see flowjo_biex_lut)."""
    x_lo, x_hi = 0.0, b
    d = (x_lo + x_hi) / 2
    dx = abs(int(x_lo - x_hi))
    dx_last = dx
    fb = -2 * np.log(b) + w * b
    f = 2. * np.log(d) + w * b + fb
    df = 2 / d + w
    if w == 0:
        return b
    for _ in range(100):
        if (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0 \
                or abs(2 * f) > abs(dx_last * df):
            dx = (x_hi - x_lo) / 2
            d = x_lo + dx
            if d == x_lo:
                return d
        else:
            dx = f / df
            t = d
            d -= dx
            if d == t:
                return d
        if abs(dx) < 1.0E-12:
            return d
        dx_last = dx
        f = 2 * np.log(d) + w * d + fb
        df = 2 / d + w
        if f < 0:
            x_lo = d
        else:
            x_hi = d
    return d


@functools.lru_cache(maxsize=16)
def flowjo_biex_lut(neg: float, width: float, pos: float, max_range: float,
                    channels: int = _FLOWJO_BIEX_CHANNELS,
                    ) -> tuple[np.ndarray, np.ndarray] | None:
    """FlowJo 10's biex axis as a lookup table: (data value at each channel,
    channel numbers 0..channels). FlowJo maps between them by linear
    interpolation, and so does `_flowjo_display_axis`. `neg`, `width`, `pos`
    and `max_range` are the <transforms:biex> attributes of the same names
    (`width` is FlowJo's negative "width basis", eg -100).

    The arithmetic is FlowKit's `generate_biex_lut` (FlowKit, BSD-3-Clause,
    Copyright (c) 2018 Scott White; ported from cytolib, itself ported from
    FlowJo's legacy Java code), reproduced step for step: bit for bit equal
    to FlowKit's table for five parameter sets. It matches the lookup table
    FlowJo itself exports for width -7.943282, neg 1 (FlowKit's test data) at
    every one of its 4096 entries within one unit of the sixth significant
    digit FlowJo prints (largest relative difference 4.8e-6; FlowKit's table
    differs identically). Returns None for parameters it is not defined for."""
    if not (width < 0 and max_range > 0 and np.isfinite([neg, pos]).all()):
        return None
    ln10 = np.log(10.0)
    w = np.log10(-width)
    decades = pos - w / 2
    extra = max(neg, 0.0) + w / 2
    zero_point = int((extra * channels) / (extra + decades))
    zero_point = int(min(zero_point, channels / 2))
    if zero_point > 0:
        decades = extra * channels / zero_point
    w = w / (2 * decades)
    positive_range = ln10 * decades
    minimum = max_range / np.exp(positive_range)
    negative_range = _flowjo_log_root(positive_range, w)
    n = channels + 1
    idx = np.arange(n)
    positive = np.exp(idx / float(n) * positive_range)
    negative = np.exp(idx / float(n) * -negative_range)
    negative *= np.exp((positive_range + negative_range) * (w + extra / decades))
    s = positive[zero_point] - negative[zero_point]
    positive[zero_point:] = minimum * (positive[zero_point:] - negative[zero_point:] - s)
    below = np.arange(zero_point)
    positive[below] = -positive[2 * zero_point - below]
    if not (np.isfinite(positive).all() and (np.diff(positive) > 0).all()):
        return None
    return positive, idx.astype(float)


def _wsp_axis_elem(sample_elem: ET.Element | None, param: str | None) -> ET.Element | None:
    """The <Transformations> child that sets FlowJo parameter `param`'s axis
    in a (namespace-stripped) <Sample>, or None."""
    for block in (sample_elem.findall('Transformations') if sample_elem is not None else []):
        for t in block:
            p = t.find('parameter')
            if p is not None and p.get('name') == param:
                return t
    return None


def _flowjo_display_axis(
        sample_elem: ET.Element | None, param: str | None,
) -> tuple[str | None, Callable[[npt.ArrayLike], np.ndarray] | None, str]:
    """(kind, display -> data function, why) for one axis of FlowJo's display
    space: kind 'linear', 'logicle' or 'biex', or None with `why` when this
    axis cannot be converted. `param` is FlowJo's parameter name ('Comp-'
    kept), as the sample's <Transformations> names it."""
    t = _wsp_axis_elem(sample_elem, param)
    if t is None:
        return None, None, f"the sample declares no axis transform for {param}"
    try:
        if t.tag == 'linear' and float(t.get('gain', 1)) == 1:
            lo = float(t.get('minRange', 'nan'))
            hi = float(t.get('maxRange', 'nan'))
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                # Display 0 is minRange. FlowJo 10.10.2 measured: on linear
                # axes it saved with minRange 500 and -2000, a rectangle edge
                # AT minRange was the axis bottom (opened) and one 500 above
                # was kept (see _LINEAR_FLOORS_MEASURED). FlowKit's
                # (x + minRange) / (maxRange + minRange) would put those axes'
                # bottoms at -500 and +2000 instead. With minRange 0 the two
                # agree.
                return ('linear',
                        lambda d: lo + (np.asarray(d, dtype=float)
                                        / _FLOWJO_DISPLAY_BINS) * (hi - lo),
                        '')
        if t.tag == 'logicle':
            lp: dict[str, Any] = {k.lower(): float(t.get(k, 'nan'))
                                  for k in ('T', 'W', 'M', 'A')}
            if all(np.isfinite(v) for v in lp.values()):
                return ('logicle',
                        lambda d: inverse_transform_values(
                            np.asarray(d, dtype=float) / _FLOWJO_DISPLAY_BINS,
                            'logicle', **lp),
                        '')
        if t.tag == 'biex' and float(t.get('length', 'nan')) == _FLOWJO_DISPLAY_BINS:
            lut = flowjo_biex_lut(*(float(t.get(k, 'nan'))
                                     for k in ('neg', 'width', 'pos', 'maxRange')))
            if lut is not None:
                data, chan = lut
                # Off the axis, the table's end value: FlowJo has no table
                # entry beyond either end.
                return ('biex',
                        lambda d: np.interp(np.asarray(d, dtype=float)
                                            * (_FLOWJO_BIEX_CHANNELS / _FLOWJO_DISPLAY_BINS),
                                            chan, data),
                        '')
    except (TypeError, ValueError):
        pass
    kind = t.tag + (f" (gain {t.get('gain')})" if t.tag == 'linear' else '')
    return None, None, f"{param} is on a FlowJo {kind} axis OpenFlo cannot convert"


# A FlowJo workspace stores a polygon's vertices in data units, but FlowJo
# draws and evaluates its edges straight on the plot's axes: the transform
# each <Sample> declares per parameter in <Transformations>. On a logicle or
# biex axis a straight edge there is a curve in data units, so evaluating the
# same vertices with straight edges in data units counts another region
# (measured on public FlowJo 10 workspaces with logicle axes: 6 to 114 events
# per polygon differed from FlowKit, which evaluates on the axes; with the
# edges on the axes, none). An imported polygon therefore carries
# 'flowjo_edge_axes': per axis (x, y) the FlowJo axis its edges are straight
# on, or None for data units. gate_to_mask evaluates it there;
# map_gate_coords, which puts a gate on another scale, traces each edge on
# those axes first and drops the key. A linear axis, at any gain, keeps edges
# straight in data units; an undeclared axis (OpenFlo's own exports declare
# none) keeps them so too, so a workspace OpenFlo wrote reads back as written.


def flowjo_edge_axis(sample_elem: ET.Element | None, param: str | None) -> dict[str, Any] | None:
    """The axis FlowJo draws `param`'s polygon edges straight on, as a JSON
    spec (see wsp_gate_coords' `edge_axis_fns`), or None when they are straight in
    data units (a linear axis, or no or an unreadable declaration). `param` is FlowJo's
    parameter name ('Comp-' kept), as <Transformations> names it."""
    t = _wsp_axis_elem(sample_elem, param) if param else None
    if t is None or t.tag == 'linear':
        return None

    def num(*keys: str) -> list[float]:
        vals = [float(t.get(k, 'nan')) for k in keys]
        if not all(np.isfinite(v) for v in vals):
            raise ValueError(f"{t.tag} axis of {param}: missing attribute")
        return vals

    try:
        if t.tag == 'logicle':
            T, W, M, A = num('T', 'W', 'M', 'A')
            return {'kind': 'logicle', 'T': T, 'W': W, 'M': M, 'A': A}
        if t.tag == 'biex':
            neg, width, pos, max_range = num('neg', 'width', 'pos', 'maxRange')
            if flowjo_biex_lut(neg, width, pos, max_range) is not None:
                return {'kind': 'biex', 'neg': neg, 'width': width,
                        'pos': pos, 'maxRange': max_range}
        elif t.tag == 'log':
            offset, decades = num('offset', 'decades')
            if offset > 0 and decades > 0:
                return {'kind': 'log', 'offset': offset, 'decades': decades}
        elif t.tag == 'fasinh':
            T, M, A = num('T', 'M', 'A')
            if T > 0 and M > 0:
                return {'kind': 'fasinh', 'T': T, 'M': M, 'A': A}
    except ValueError:
        pass
    log.warning("[WspReader] %s: FlowJo %s axis not understood; polygon "
                "edges on it are taken as straight in data units", param, t.tag)
    return None


def wsp_polygon_edge_axes(gate_elem: ET.Element, sample_elem: ET.Element | None,
                          ) -> list[dict[str, Any] | None] | None:
    """'flowjo_edge_axes' for a FlowJo <PolygonGate> in `sample_elem` (both
    namespace-stripped): [x axis, y axis], or None when its edges are
    straight in data units on both."""
    if sample_elem is None:
        return None
    params: list[str | None] = []
    for d in list(gate_elem.iter('dimension'))[:2]:
        fd = next(d.iter('fcs-dimension'), None)
        params.append((fd.get('name') or fd.get('PnN')) if fd is not None else None)
    axes = [flowjo_edge_axis(sample_elem, p) for p in params]
    return axes if any(a is not None for a in axes) else None


def _flowjo_ellipse_points(ell_elem: ET.Element) -> tuple[np.ndarray, np.ndarray] | None:
    """(foci, edge) of a FlowJo-drawn EllipsoidGate as (n, 2) float arrays in
    display units, or None when they are missing or not numbers."""
    def points(tag: str) -> np.ndarray | None:
        el = ell_elem.find(tag)
        if el is None:
            return None
        out = []
        for v in el.findall('vertex'):
            cs = [c.get('value') for c in v.findall('coordinate')]
            if len(cs) != 2:
                return None
            try:
                # A missing value is None: float() raises TypeError, caught.
                out.append([float(cs[0]), float(cs[1])])  # pyright: ignore[reportArgumentType]
            except (TypeError, ValueError):
                return None
        return np.asarray(out, dtype=float).reshape(-1, 2)

    foci, edge = points('foci'), points('edge')
    if foci is None or edge is None:
        return None
    return foci, edge


def _flowjo_ellipse_display(foci: npt.ArrayLike, edge: npt.ArrayLike,
                            ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """(centre, major-axis unit vector, a, b) of a FlowJo-drawn ellipse in
    display units, or None when its foci/edge points do not define one.

    The centre is mid-foci and the major axis runs through the foci. An edge
    point's distance from the centre is at most `a`, reached at the ends of the
    major axis, which FlowJo stores first; `b` follows from the foci
    (b^2 = a^2 - c^2), as in FlowKit, which found FlowJo's stored minor-axis
    points unreliable."""
    foci, edge = np.asarray(foci, dtype=float), np.asarray(edge, dtype=float)
    if (foci.shape != (2, 2) or edge.ndim != 2 or len(edge) == 0
            or not np.isfinite(foci).all() or not np.isfinite(edge).all()):
        return None
    centre = foci.mean(axis=0)
    half = (foci[1] - foci[0]) / 2.0
    c = float(np.hypot(*half))
    a = float(np.max(np.hypot(*(edge - centre).T)))
    if not a > c:
        return None
    u = half / c if c > 0 else np.array([1.0, 0.0])
    return centre, u, a, float(np.sqrt(a * a - c * c))


def _flowjo_native_ellipse(foci: npt.ArrayLike, edge: npt.ArrayLike,
                           sample_elem: ET.Element | None, x_param: str | None,
                           y_param: str | None) -> tuple[dict[str, Any] | None, str]:
    """A FlowJo-drawn EllipsoidGate (foci + edge, no mean) as a gate body in
    data units: {'kind': 'ellipsoid', ...} on two linear axes (exact),
    {'kind': 'polygon', ...} when an axis is logicle or biex, else None.
    Returns (body, why): `why` says what was approximated, or why there is no
    body."""
    geom = _flowjo_ellipse_display(foci, edge)
    if geom is None:
        return None, "its foci/edge points do not define an ellipse"
    centre, u, a, b = geom
    v = np.array([-u[1], u[0]])
    axes = [_flowjo_display_axis(sample_elem, p) for p in (x_param, y_param)]
    for kind, _f, why in axes:
        if kind is None:
            return None, why
    disp_cov = a * a * np.outer(u, u) + b * b * np.outer(v, v)
    (kx, fx, _), (ky, fy, _) = axes
    if kx == ky == 'linear':
        # Affine per axis, so exact: data = lo + s * display.
        # fx and fy are None only with a None kind, returned above.
        lo = np.array([float(fx(0.0)), float(fy(0.0))])  # pyright: ignore[reportOptionalCall]
        s = np.array([float(fx(1.0)), float(fy(1.0))]) - lo  # pyright: ignore[reportOptionalCall]
        cov = np.outer(s, s) * disp_cov
        return {'kind': 'ellipsoid',
                'mean': [float(v_) for v_ in lo + s * centre],
                'cov': [[float(e) for e in row] for row in cov],
                'distance_sq': 1.0}, ''
    verts = ellipse_polygon(centre, disp_cov, 1.0)
    if verts is None:
        return None, "its foci/edge points do not define an ellipse"
    vs = np.asarray(verts, dtype=float)
    n_drawn = len(vs)
    # A biex axis is a table: an event off either end sits at that end of the
    # axis (FlowKit clamps it there, as FlowJo puts off-scale events on the
    # axis edge). So the part of the ellipse beyond an end is cut off at the
    # end, and the cut is extended straight out to +-1e12, taking in every
    # event beyond that end between the two cut points. Mapping the off-axis
    # vertices to the table's end value instead left those events out: 352
    # where FlowKit counts 359 on an ellipse crossing the bottom of a width
    # -10 axis.
    opened = []
    for i, kind in enumerate((kx, ky)):
        if kind == 'biex':
            vs = _clip_polygon_axis(vs, i, 0.0, _FLOWJO_DISPLAY_BINS)
            opened.append(i)
    if len(vs) < 3:
        return None, "its outline lies entirely off the axis"
    data = np.column_stack([fx(vs[:, 0]), fy(vs[:, 1])])  # pyright: ignore[reportOptionalCall]
    if not np.isfinite(data).all():
        return None, "its outline falls outside the axis transform's range"
    for i in opened:
        for bound, far in ((0.0, -OPEN), (_FLOWJO_DISPLAY_BINS, OPEN)):
            vs, data = open_polygon_at(vs, data, i, bound, far)
    return ({'kind': 'polygon', 'vertices': data.tolist()},
            f"converted from FlowJo's display space on {kx}/{ky} axes as a "
            f"{n_drawn}-vertex polygon; FlowJo's own count can differ (on the "
            "real FlowJo logicle ellipses measured, OpenFlo counted a few "
            "percent fewer events than FlowJo)")


def open_polygon_at(vs: np.ndarray, data: np.ndarray, axis: int, bound: float,
                    far: float) -> tuple[np.ndarray, np.ndarray]:
    """Extend polygon `data` (data units; `vs` the same vertices in display
    units) beyond a cut at display coordinate `bound` on `axis`: each run of
    vertices on the cut keeps its two ends and gains copies moved to `far`
    on that axis, so the region beyond the axis end runs straight out from
    the cut points."""
    on = vs[:, axis] == bound
    if not on.any():
        return vs, data
    n = len(vs)
    out_v, out_d = [], []
    for k in range(n):
        if not on[k]:
            out_v.append(vs[k])
            out_d.append(data[k])
            continue
        moved = data[k].copy()
        moved[axis] = far
        pts = ([data[k]] if not on[k - 1] else []) + [moved] \
            + ([data[k]] if not on[(k + 1) % n] else [])
        out_v.extend([vs[k]] * len(pts))
        out_d.extend(pts)
    return np.asarray(out_v, dtype=float), np.asarray(out_d, dtype=float)


def _clip_polygon_axis(vs: npt.ArrayLike, axis: int, lo: float, hi: float) -> np.ndarray:
    """Polygon `vs` ((n, 2) array) cut to lo <= coordinate `axis` <= hi
    (Sutherland-Hodgman, one axis); the cut points lie exactly on lo / hi."""
    def cut(pts: list[np.ndarray], keep: Callable[[np.ndarray], bool],
            bound: float) -> list[np.ndarray]:
        out = []
        for k in range(len(pts)):
            p, q = pts[k - 1], pts[k]
            if keep(q) != keep(p):
                r = p + (bound - p[axis]) / (q[axis] - p[axis]) * (q - p)
                r[axis] = bound
                out.append(r)
            if keep(q):
                out.append(q)
        return out
    pts = cut(list(np.asarray(vs, dtype=float)), lambda p: p[axis] >= lo, lo)
    pts = cut(pts, lambda p: p[axis] <= hi, hi) if pts else pts
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def strip_comp_marker(name: str) -> tuple[str, bool]:
    """(data column, compensated?) for a FlowJo parameter name. FlowJo names
    a legacy-compensated parameter 'Comp-FL5-A' and a natively compensated
    one '<FL5-A>' (angle brackets); both strip to the FCS column 'FL5-A'. The
    bare name is the UNcompensated parameter of a compensated sample, which
    FlowJo keeps beside the compensated one. (A fully configurable comp
    prefix/suffix is a rarer case not handled here.)"""
    marked = False
    if name.startswith('Comp-'):
        name, marked = name[len('Comp-'):], True
    if len(name) >= 2 and name[0] == '<' and name[-1] == '>':
        name, marked = name[1:-1], True
    return name, marked


def wsp_comp_channels(gate_elem: ET.Element) -> list[str]:
    """Sorted data columns that gate element `gate_elem` (a RectangleGate,
    PolygonGate, EllipsoidGate or QuadrantGate, namespace-stripped) names
    with FlowJo's compensated marker. Every gate read from a .wsp carries
    this as 'wsp_comp_channels' (its presence says the gate came from a
    workspace, where a bare name means something): stripping the marker merges
    'Comp-FL2-A' and the uncompensated 'FL2-A' into one column, and FlowJo
    evaluates the two on different values (a gate at 2,500: 5,216 events
    compensated, 8,413 not). openflo-compare evaluates the unmarked ones on
    the uncompensated data; the editor, which holds only compensated
    values, warns about them."""
    out = set()
    for fd in gate_elem.iter('fcs-dimension'):
        ch, marked = strip_comp_marker(fd.get('name') or fd.get('PnN') or '')
        if marked and ch:
            out.add(ch)
    return sorted(out)


def _unread_tags(unread: Iterable[tuple[str, list[str], int]]) -> set[str]:
    """Gate tags named by `unread` (see _log_unread)."""
    return {t for _, tags, _ in unread for t in tags}


def _log_unread(where: str, unread: Sequence[tuple[str, list[str], int]]) -> None:
    """Warn about the populations WspReader.extract_gates left out because
    their gate could not be read (see _GateWalk.walk_population)."""
    if not unread:
        return
    under = sum(n for _, _, n in unread)
    shown = ', '.join(f"'{name}' ({'/'.join(tags)})"
                      for name, tags, _ in unread[:5])
    more = f' (+{len(unread) - 5})' if len(unread) > 5 else ''
    log.warning(
        "[WspReader] %s: %d population(s) with a gate OpenFlo cannot read "
        "were not imported: %s%s%s. They are left out rather than counted "
        "without their gate.", where, len(unread), shown, more,
        f"; nor were the {under} population(s) under them" if under else '')


def wsp_ellipse_gate(ell_elem: ET.Element, sample_elem: ET.Element | None,
                     ) -> tuple[dict[str, Any] | None, str]:
    """A <EllipsoidGate> of a (namespace-stripped) FlowJo workspace as a gate
    body without ids: (body, why), body None when the element is unreadable.

    - Gating-ML 2.0 form (<mean>, <covarianceMatrix>, <distanceSquare>, a
      squared Mahalanobis radius): 'ellipsoid'.
    - Drawn in FlowJo (two <foci>, four <edge> points in display space):
      converted with the sample's axes by _flowjo_native_ellipse, `why`
      naming any approximation; where it cannot be converted, a
      'flowjo_ellipse' placeholder that admits no events and keeps FlowJo's
      points, element attributes and the two axes (`why` says why). Dropping
      it instead attached its children to the grandparent, so each child
      silently counted every event the ellipse had excluded (12,358 where
      2,405 were right).

    `sample_elem` is the <Sample> holding the gate (its <Transformations>);
    None converts nothing."""
    dims = list(ell_elem.findall('dimension'))
    if len(dims) != 2:
        return None, "it does not have two dimensions"
    xc = WspReader._channel_name(dims[0])
    yc = WspReader._channel_name(dims[1])
    if not xc or not yc:
        return None, "a dimension names no parameter"
    mean_elem = ell_elem.find('mean')
    if mean_elem is None and ell_elem.find('foci') is not None:
        # Keyed by FlowJo's own parameter names ('Comp-' kept), as
        # <Transformations> names them.
        params = []
        for d in dims:
            fd = next(d.iter('fcs-dimension'), None)
            params.append((fd.get('name') or fd.get('PnN')) if fd is not None else None)
        pts = _flowjo_ellipse_points(ell_elem)
        if pts is None:
            # Unreadable, but still FlowJo's ellipse: dropping it attached
            # its children to the grandparent, every event of 40,000 inside.
            body, why = None, "its foci/edge points cannot be read"
        else:
            body, why = _flowjo_native_ellipse(*pts, sample_elem, *params)
        if body is not None:
            return {**body, 'x_channel': xc, 'y_channel': yc}, why
        axes = []
        for p in params:
            t = _wsp_axis_elem(sample_elem, p)
            axes.append(None if t is None else
                        {'kind': t.tag, 'attrs': dict(t.attrib.items())})
        raw = copy.copy(ell_elem)
        raw.tail = None
        return {'kind': 'flowjo_ellipse', 'x_channel': xc, 'y_channel': yc,
                'flowjo': {'parameters': params,
                           'foci': None if pts is None else pts[0].tolist(),
                           'edge': None if pts is None else pts[1].tolist(),
                           'axes': axes, 'why': why,
                           # For the export: the element exactly as read.
                           'xml': ET.tostring(raw, encoding='unicode')}}, why
    if mean_elem is None:
        return None, "it has neither a mean nor foci"
    # A missing value or covarianceMatrix raises TypeError / AttributeError, caught.
    try:
        mean_vals = [float(c.get('value'))  # pyright: ignore[reportArgumentType]
                     for c in mean_elem.findall('coordinate')]
        cov = [[float(e.get('value')) for e in row.findall('entry')]  # pyright: ignore[reportArgumentType]
               for row in ell_elem.find('covarianceMatrix').findall('row')]  # pyright: ignore[reportOptionalMemberAccess]
    except (AttributeError, TypeError, ValueError):
        return None, "its mean or covariance is not numbers"
    if len(mean_vals) != 2 or len(cov) != 2 or any(len(r) != 2 for r in cov):
        return None, "its mean or covariance is not 2-D"
    dsq_elem = ell_elem.find('distanceSquare')
    try:
        dist_sq = (float(dsq_elem.get('value'))  # pyright: ignore[reportArgumentType]
                   if dsq_elem is not None else 4.0)
    except (TypeError, ValueError):
        dist_sq = 4.0
    return {'kind': 'ellipsoid', 'x_channel': xc, 'y_channel': yc,
            'mean': mean_vals, 'cov': cov, 'distance_sq': dist_sq}, ''


# ── The workspace reader ──────────────────────────────────────────────────────

def wsp_sample_events(sample_elem: ET.Element | None) -> int | None:
    """The event count a (namespace-stripped) FlowJo <Sample> records for
    its file (FlowJo writes it as the <SampleNode>'s `count`), or None when
    it records none. 0 and -1 are not counts: an OpenFlo workspace may hold
    0 where nothing was counted, and FlowJo writes -1 for a sample it did
    not compute."""
    sn = sample_elem.find('SampleNode') if sample_elem is not None else None
    if sn is None:
        return None
    try:
        n = int(sn.get('count'))  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def fcs_event_total(path: str | os.PathLike[str]) -> int | None:
    """$TOT of the FCS file at `path`, or None when it cannot be read."""
    try:
        kw = {k.upper(): v for k, v in _fcs_text_pairs(path)}
        n = int(str(kw.get('$TOT', '')).strip())
    except (OSError, ValueError, IndexError):
        return None
    return n if n >= 0 else None


def wsp_event_mismatch(expected: int | None, actual: int | None) -> str:
    """'' when a workspace sample's recorded event count `expected` (see
    wsp_sample_events) agrees with the `actual` events of the FCS file it
    was matched to, or either is unknown; else what to tell the user. A
    workspace can hold several samples whose files share a name, so a file
    found by name may be another sample's."""
    if expected is None or actual is None or int(expected) == int(actual):
        return ''
    return (f"the workspace records {int(expected)} events for this sample "
            f"but the FCS file has {int(actual)}: it may be a different "
            "file of the same name")


_DTD_SUBSET = re.compile(rb'<!doctype[^>\[]*\[', re.IGNORECASE)


def xml_declares_entities(head: bytes) -> bool:
    """Whether the start of an XML file (bytes) declares a DTD internal
    subset (``<!DOCTYPE x [ ... ]>``) or an entity, the vehicles of an
    entity-expansion attack. A bare ``<!DOCTYPE name>`` declares neither."""
    return bool(_DTD_SUBSET.search(head)) or b'<!entity' in head.lower()


def _pre_flowjo10_version(root: ET.Element) -> str:
    """The FlowJo version of a parsed workspace root when it is older than
    FlowJo 10 (its ``flowJoVersion``, else its ``version``, below 10; FlowJo
    9 writes version "2.0" and flowJoVersion "9.x", FlowJo 10 "20.0" and
    "10.x"), else ''. A version that is not a number (earlier OpenFlo
    exports say "OpenFlo-export-1.0") is not judged."""
    tag = re.sub(r'\{.*?\}', '', root.tag)
    if tag != 'Workspace':
        return ''
    for key in ('flowJoVersion', 'version'):
        v = (root.get(key) or '').strip()
        m = re.match(r'(\d+)(?:\.\d+)*', v)
        if m:
            return v if int(m.group(1)) < 10 else ''
    return ''


def _sample_owners(root: ET.Element) -> dict[ET.Element | tuple[str, str | None], ET.Element]:
    """SampleNode -> its <Sample> (ET elements have no parent pointer). A
    caller may pass a SampleNode from its own parse of the same file (the
    editor does), so each sampleID also maps to its <Sample>."""
    owner: dict[ET.Element | tuple[str, str | None], ET.Element] = {}
    for s_el in root.iter('Sample'):
        for sn_el in s_el.findall('SampleNode'):
            owner[sn_el] = s_el
            if sn_el.get('sampleID') is not None:
                owner.setdefault(('sampleID', sn_el.get('sampleID')), s_el)
    return owner


def _log_extracted(gates: Sequence[Mapping[str, Any]]) -> None:
    """Log the gates WspReader.extract_gates read, indented so the hierarchy
    is visible."""
    if not gates:
        log.info("[WspReader] No supported gates found.")
        return
    id_to_depth: dict[str, int] = {}
    for g in gates:
        pid = g.get('parent_id')
        id_to_depth[g['_import_id']] = (
            id_to_depth.get(pid, -1) + 1 if pid else 0)
    log.info(f"[WspReader] Extracted {len(gates)} gate(s):")
    for g in gates:
        d = id_to_depth.get(g['_import_id'], 0)
        log.info(f"   {'  ' * d}{describe_gate(g)}")


class _GateWalk:
    """One WspReader.extract_gates call: walks a workspace's population tree
    and collects its gates (see extract_gates for what is read and how)."""

    def __init__(self, reader: WspReader) -> None:
        self.reader = reader
        # extract_gates builds a walk only once the reader has a root.
        self.root: ET.Element = reader.root  # pyright: ignore[reportAttributeAccessIssue]
        self.gates: list[dict[str, Any]] = []
        self.next_id = 0
        self.owner = _sample_owners(reader.root)  # pyright: ignore[reportArgumentType]
        # FlowJo's axis floors for the sample being walked.
        self.floors: dict[str, tuple[float, float]] = {}
        # The <Sample> being walked: a FlowJo-drawn ellipse needs its axis
        # transforms. Gate tags walked but not imported, for the warning, and
        # the populations imported as 'flowjo_ellipse' placeholders.
        self.cur_sample: ET.Element | None = None
        self.not_imported: set[str] = set()
        self.placeholders: list[str] = []
        # (population, gate tags, populations under it) whose gate could not
        # be read: left out with their subtree. Also kept on the reader for
        # a caller's status line.
        self.unread: list[tuple[str, list[str], int]] = []
        self.literal = wsp_written_by_openflo(reader.root)
        # Per sample: population path -> import id, and the boolean nodes
        # read, whose operands are resolved when the sample is done.
        self.pop_paths: dict[tuple[str, ...], str | None] = {}
        self.booleans: list[tuple[dict[str, Any], tuple[str, ...], list[str | None]]] = []

    # ── per-sample context ───────────────────────────────────────────────

    def sample_of(self, sn: ET.Element) -> ET.Element | None:
        s_el = self.owner.get(sn)
        return s_el if s_el is not None else self.owner.get(
            ('sampleID', sn.get('sampleID')))

    def floors_for(self, sn: ET.Element) -> dict[str, tuple[float, float]]:
        # OpenFlo's own export declares linear axes FlowJo would floor;
        # OpenFlo counted its edges as written, so they read back so.
        return {} if self.literal else wsp_axis_floors(self.sample_of(sn))

    def timestep_for(self, sn: ET.Element) -> float | None:
        s_el = self.sample_of(sn)
        return self.reader.sample_timestep(s_el) if s_el is not None else None

    def to_ticks(self, start: int, ts: float | None) -> None:
        """Gates appended since `start`: FlowJo's Time seconds -> ticks."""
        if ts:
            self.gates[start:] = [scale_time_coords(g, 1.0 / ts)
                                  for g in self.gates[start:]]

    def make_id(self) -> str:
        i = self.next_id
        self.next_id += 1
        return f'imp_{i}'

    # ── gate elements ────────────────────────────────────────────────────

    def _rect_dimension(self, dim: ET.Element) -> tuple[str, float | None, float | None] | None:
        """(channel, lo, hi) of a RectangleGate <dimension>, or None when it
        names no channel or a bound is not a number."""
        ch = self.reader._channel_name(dim)
        if not ch:
            return None
        min_attr = dim.get('min')
        max_attr = dim.get('max')
        try:
            lo = float(min_attr) if min_attr is not None else None
            hi = float(max_attr) if max_attr is not None else None
        except (TypeError, ValueError):
            return None
        # Gating-ML writes an open bound as "-INF"/"INF", which float()
        # turns into ±inf — and json.dump then writes `-Infinity`, which
        # RFC 8259 does not allow. Python's own reader accepts it, so
        # the session file would look fine locally while being unreadable
        # by JSON.parse, jq, or anything else. The MISSING-bound case in
        # _one_d_rect uses ±1e12 for exactly this reason
        # ("JSON-safe, unlike -inf"); the EXPLICIT-INF case does too.
        lo = _finite_bound(lo, -OPEN)
        hi = _finite_bound(hi, OPEN)
        # Keyed by FlowJo's own parameter name (before _channel_name
        # strips 'Comp-'), as <Transformations> names it.
        fd = next(dim.iter('fcs-dimension'), None)
        lo = wsp_axis_floor_lo(fd.get('name') if fd is not None else None,
                               lo, self.floors)
        return ch, lo, hi

    def _one_d_rect(self, ch: str, lo: float | None, hi: float | None,
                    parent_imp_id: str | None) -> dict[str, Any] | None:
        if hi is not None:
            # Max-only 1-D rect (x < hi): represent as an interval with
            # a large negative sentinel lo (JSON-safe, unlike -inf) so
            # the constraint — and this population's children — stay in
            # the tree instead of being silently re-parented up to the
            # grandparent (which drops `x < hi` from every descendant).
            return {'kind': 'interval', 'channel': ch,
                    'lo': -OPEN if lo is None else lo, 'hi': hi,
                    'parent_id': parent_imp_id,
                    '_import_id': self.make_id()}
        if lo is not None:
            return {'kind': 'threshold', 'channel': ch, 'value': lo,
                    'parent_id': parent_imp_id,
                    '_import_id': self.make_id()}
        return None

    def _two_d_rect(self, x: tuple[str, float | None, float | None],
                    y: tuple[str, float | None, float | None],
                    parent_imp_id: str | None) -> dict[str, Any]:
        (xc, x0, x1), (yc, y0, y1) = x, y
        # A missing bound is an open side: FlowJo writes each of a
        # quadrant's four sectors with only its divider bound per
        # dimension. Skipping them dropped every FlowJo quadrant and
        # lifted its children out of their sector.
        return {'kind': 'rect',
                'x_channel': xc, 'y_channel': yc,
                'x0': -OPEN if x0 is None else x0,
                'x1': OPEN if x1 is None else x1,
                'y0': -OPEN if y0 is None else y0,
                'y1': OPEN if y1 is None else y1,
                'parent_id': parent_imp_id,
                '_import_id': self.make_id()}

    def parse_rect(self, rect_elem: ET.Element, parent_imp_id: str | None) -> dict[str, Any] | None:
        parsed = [p for p in map(self._rect_dimension,
                                 rect_elem.iter('dimension'))
                  if p is not None]
        if len(parsed) == 1:
            return self._one_d_rect(*parsed[0], parent_imp_id)
        if len(parsed) == 2:
            return self._two_d_rect(parsed[0], parsed[1], parent_imp_id)
        return None

    @staticmethod
    def _polygon_vertex(v: ET.Element) -> list[float] | None:
        """[x, y] of a PolygonGate <vertex>, or None when it is not two
        finite-or-infinite numbers."""
        coords = list(v.iter('coordinate'))
        if len(coords) < 2:
            return None
        vx_attr = coords[0].get('value')
        vy_attr = coords[1].get('value')
        if vx_attr is None or vy_attr is None:
            return None
        try:
            vx = float(vx_attr)
            vy = float(vy_attr)
        except (TypeError, ValueError):
            return None
        # A vertex at infinity carries the same JSON hazard as an open
        # rectangle bound, and a NaN vertex has no place in a polygon
        # at all — skip it rather than writing an unreadable session.
        vx = _finite_bound(vx, -OPEN if vx < 0 else OPEN)
        vy = _finite_bound(vy, -OPEN if vy < 0 else OPEN)
        if vx is None or vy is None:
            return None
        return [vx, vy]

    def parse_polygon(self, poly_elem: ET.Element,
                      parent_imp_id: str | None) -> dict[str, Any] | None:
        dims = list(poly_elem.iter('dimension'))
        if len(dims) != 2:
            return None
        xc = self.reader._channel_name(dims[0])
        yc = self.reader._channel_name(dims[1])
        if not xc or not yc:
            return None
        verts = [p for p in map(self._polygon_vertex, poly_elem.iter('vertex'))
                 if p is not None]
        if len(verts) < 3:
            return None
        g = {'kind': 'polygon',
             'x_channel': xc, 'y_channel': yc,
             'vertices': verts,
             'parent_id': parent_imp_id,
             '_import_id': self.make_id()}
        axes = wsp_polygon_edge_axes(poly_elem, self.cur_sample)
        if axes is not None:
            g['flowjo_edge_axes'] = axes
        return g

    def parse_ellipsoid(self, ell_elem: ET.Element, parent_imp_id: str | None,
                        pop_name: str | None = None) -> dict[str, Any] | None:
        """EllipsoidGate, in either form (see wsp_ellipse_gate). An
        event is inside a Gating-ML one when
        (x-µ)ᵀ Σ⁻¹ (x-µ) ≤ distanceSquare. A FlowJo-drawn one that cannot
        be converted is a 'flowjo_ellipse' placeholder, reported here and
        in the walk's summary as imported, not skipped."""
        body, why = wsp_ellipse_gate(ell_elem, self.cur_sample)
        if body is None:
            return None
        name = pop_name or 'unnamed'
        if body['kind'] == 'flowjo_ellipse':
            self.placeholders.append(name)
            log.warning(
                f"[WspReader] FlowJo ellipse '{name}' could not be "
                f"converted ({why}); imported as a placeholder that "
                "admits no events, so it and every population under it "
                "stay empty. Redraw it in OpenFlo; a FlowJo export "
                "writes it back as FlowJo drew it.")
        elif why:
            log.warning(f"[WspReader] FlowJo ellipse '{name}': {why}.")
        return {**body, 'parent_id': parent_imp_id,
                '_import_id': self.make_id()}

    def _divider(self, d: ET.Element) -> tuple[str, float] | None:
        """(channel, threshold) of a QuadrantGate <divider>, or None."""
        ch = self.reader._channel_name(d)
        # Divider threshold is a <value> child element (Gating-ML)
        # or a value= attribute (defensive fallback).
        val = None
        v_elem = d.find('value')
        if v_elem is not None and (v_elem.text or '').strip():
            val = v_elem.text.strip()  # pyright: ignore[reportOptionalMemberAccess]
        if val is None:
            val = d.get('value')
        try:
            val = float(val) if val is not None else None
        except (TypeError, ValueError):
            val = None
        if not ch or val is None:
            return None
        return ch, val

    def parse_quadrant(self, quad_elem: ET.Element,
                       parent_imp_id: str | None) -> list[dict[str, Any]] | None:
        """Gating-ML 2.0 QuadrantGate: two <divider> elements, each
        naming a dimension + a threshold value. Expanded into FOUR
        rect gates sharing a `quad_set` id + `quad_origin`, spanning
        ±1e12 on their open sides — matching the editor's internal
        quadrant representation so the round-trip stays in one model.

        Returns a LIST of 4 gate dicts (not a single dict)."""
        divs = quad_elem.findall('divider')
        if len(divs) != 2:
            return None
        parsed = []
        for d in divs:
            div = self._divider(d)
            if div is None:
                return None
            parsed.append(div)
        (xc, xdiv), (yc, ydiv) = parsed
        big = OPEN
        # Rect bounds are half-open [lo, hi), so a divider used as the
        # LOWER bound of the upper quadrant and the UPPER bound of the
        # lower one already assigns an event sitting exactly on it to
        # exactly one side. The lower bound must NOT be nudged down by
        # one ULP (nextafter): an event at exactly nextafter(div, -inf)
        # would then satisfy both `>= xlo` above and `< xdiv` below,
        # i.e. be double-counted.
        xlo, ylo = xdiv, ydiv
        qs_id = self.make_id()
        out = []
        for label, x0, x1, y0, y1 in [
                ('Q++ (x>, y>)',  xlo,   big,  ylo,   big),
                ('Q+- (x>, y<)',  xlo,   big, -big,  ydiv),
                ('Q-+ (x<, y>)', -big,  xdiv,  ylo,   big),
                ('Q-- (x<, y<)', -big,  xdiv, -big,  ydiv)]:
            out.append({'kind': 'rect',
                        'x_channel': xc, 'y_channel': yc,
                        'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                        'label': label,
                        'quad_set': qs_id,
                        'quad_origin_x': xdiv, 'quad_origin_y': ydiv,
                        'parent_id': parent_imp_id,
                        '_import_id': self.make_id()})
        return out

    def _read_gate(self, child: ET.Element, parent_imp_id: str | None,
                   pop_name: str | None) -> list[dict[str, Any]] | None:
        """The gate(s) one element inside a <Gate> holds, as a list ready to
        append, or None when it is not a gate this reads."""
        if child.tag == 'QuadrantGate':
            quad = self.parse_quadrant(child, parent_imp_id)
            if not quad:
                return None
            # parse_quadrant returns 4 linked rects.
            cc = wsp_comp_channels(child)
            for q in quad:
                q['wsp_comp_channels'] = list(cc)
            return quad
        if child.tag == 'RectangleGate':
            g = self.parse_rect(child, parent_imp_id)
        elif child.tag == 'PolygonGate':
            g = self.parse_polygon(child, parent_imp_id)
        elif child.tag == 'EllipsoidGate':
            g = self.parse_ellipsoid(child, parent_imp_id, pop_name)
        else:
            return None
        if g is None:
            return None
        # `setdefault` so a parser that already derived a label keeps it.
        if pop_name:
            g.setdefault('label', pop_name)
        g['wsp_comp_channels'] = wsp_comp_channels(child)
        return [g]

    # ── population tree ──────────────────────────────────────────────────

    def _population_gate(self, pop_elem: ET.Element, parent_imp_id: str | None,
                         pop_name: str | None) -> str | None:
        """Append the first readable gate under `pop_elem`'s <Gate>s and
        return its import id (the first of a quadrant's four), else None."""
        for gate_wrapper in pop_elem.findall('Gate'):
            for child in gate_wrapper:
                read = self._read_gate(child, parent_imp_id, pop_name)
                if read:
                    self.gates.extend(read)
                    return read[0]['_import_id']
        return None

    def walk_population(self, pop_elem: ET.Element, parent_imp_id: str | None,
                        path: tuple[str, ...] = ()) -> None:
        """Process a <Population>: parse its <Gate> (if any), then recurse
        into its <Subpopulations>. A population with no gate passes its
        children to its parent. One whose gate could not be read is left
        out WITH everything under it (named in the warning): attaching
        its children to the grandparent, as this did, made each count
        every event the unread gate excluded, a plausible wrong number.
        """
        # The population's NAME lives on the <Population> element, not on
        # the <Gate> inside it — so parsing only the gate discarded every
        # label in the file. A 40-population FlowJo workspace imported
        # fully unnamed, and nothing downstream put the names back.
        pop_name = pop_elem.get('name')
        this_id = self._population_gate(pop_elem, parent_imp_id, pop_name)
        gate_tags = [c.tag for gw in pop_elem.findall('Gate') for c in gw]
        if this_id is None and gate_tags:
            self.not_imported.update(gate_tags)
            self.unread.append(
                (pop_name or 'unnamed', sorted(set(gate_tags)),
                 sum(1 for _ in pop_elem.iter('Population')) - 1))
            return
        attach_to = this_id if this_id is not None else parent_imp_id
        own = tuple(path) + (pop_name or '',)
        self.pop_paths[own] = attach_to
        self.walk_children(pop_elem, attach_to, own)

    def walk_boolean(self, node: ET.Element, parent_imp_id: str | None,
                     path: tuple[str, ...]) -> None:
        """A FlowJo AndNode / OrNode / NotNode: a boolean gate over the
        populations its <Dependent> elements name, resolved once the
        whole sample is read (an operand may come after it). These used
        to be skipped without a word, together with everything under
        them."""
        name = node.get('name') or ''
        g = {'kind': 'boolean', 'op': WSP_BOOLEAN_OPS[node.tag],
             'operands': [], 'parent_id': parent_imp_id,
             '_import_id': self.make_id()}
        if name:
            g['label'] = name
        self.gates.append(g)
        own = tuple(path) + (name,)
        self.pop_paths[own] = g['_import_id']
        deps = [d.get('name') for dn in node.findall('Dependents')
                for d in dn.findall('Dependent')]
        self.booleans.append((g, own, deps))
        self.walk_children(node, g['_import_id'], own)

    def walk_children(self, elem: ET.Element, parent_imp_id: str | None,
                      path: tuple[str, ...]) -> None:
        for sub in elem.findall('Subpopulations'):
            for child in sub:
                if child.tag == 'Population':
                    self.walk_population(child, parent_imp_id, path)
                elif child.tag in WSP_BOOLEAN_OPS:
                    self.walk_boolean(child, parent_imp_id, path)

    def _resolve_operand(self, g: dict[str, Any], own: tuple[str, ...], dep: str | None,
                         sample_label: str) -> None:
        hit = resolve_wsp_dependent(dep, own, self.pop_paths)
        target = self.pop_paths.get(hit) if hit is not None else None
        if target is None or target == g['_import_id']:
            log.warning("[WspReader] %s: boolean %r: operand %r "
                        "not found -- the population reads NA",
                        sample_label, g.get('label') or g['op'], dep)
            g['operands'].append(f'unresolved:{dep}')
        else:
            g['operands'].append(target)

    def resolve_booleans(self, sample_label: str) -> None:
        """Point each boolean read since the last call at its operands'
        import ids. An operand that cannot be found leaves the boolean
        with an operand no gate has, so it reads NA (fail-closed), and
        is named in the log."""
        for g, own, deps in self.booleans:
            for dep in deps:
                self._resolve_operand(g, own, dep, sample_label)
            if not deps:
                log.warning("[WspReader] %s: boolean %r names no operand "
                            "-- the population reads NA", sample_label,
                            g.get('label') or g['op'])
        self.booleans.clear()
        self.pop_paths.clear()

    # ── one sample / the whole document ──────────────────────────────────

    def walk_sample(self, sample_node: ET.Element, timestep: float | None) -> None:
        """Per-sample walk: just this <SampleNode>'s subpopulations. Skip
        the fallback flat scan + the cross-document unsupported-kinds
        warning (walk_document still runs both for the full-document case),
        but name what this sample's walk could not import: this is the
        editor's import path."""
        self.floors = self.floors_for(sample_node)
        self.cur_sample = self.sample_of(sample_node)
        self.walk_children(sample_node, None, ())
        label = sample_node.get('name') or 'sample'
        self.resolve_booleans(label)
        self.to_ticks(0, _positive_float(timestep) if timestep is not None
                      else self.timestep_for(sample_node))
        # One warning per skipped gate: the unread populations' names
        # their gate types (_log_unread); any other type is named here.
        rest = self.not_imported - _unread_tags(self.unread)
        if rest:
            log.warning(
                "[WspReader] %s: skipped unsupported gate types: %s",
                label, sorted(rest))
        _log_unread(label, self.unread)
        if self.placeholders:
            log.warning(
                "[WspReader] %s: %d FlowJo ellipse(s) imported as empty "
                "placeholders (not converted): %s",
                label, len(self.placeholders), ', '.join(self.placeholders))

    def _walk_document_sample(self, sn: ET.Element) -> bool:
        """Walk one <SampleNode> of the document; whether it held any
        population hierarchy."""
        start = len(self.gates)
        self.floors = self.floors_for(sn)
        self.cur_sample = self.sample_of(sn)
        self.walk_children(sn, None, ())
        walked = len(self.gates) > start or any(
            True for sub in sn.findall('Subpopulations') for _ in sub)
        self.resolve_booleans(sn.get('name') or 'sample')
        self.to_ticks(start, self.timestep_for(sn))
        return walked

    def _flat_scan(self) -> None:
        """Fallback when the .wsp has gates but no Population wrappers:
        every RectangleGate, then every PolygonGate, as roots."""
        self.floors = {}          # no owning sample for a flat scan
        for tag, parse in (('RectangleGate', self.parse_rect),
                           ('PolygonGate', self.parse_polygon)):
            for elem in self.root.iter(tag):
                g = parse(elem, None)
                if g is not None:
                    self.gates.append(g)
        # The flat scan reads neither of these.
        self.not_imported.update(gt for gt in ('EllipsoidGate', 'QuadrantGate')
                                 if any(True for _ in self.root.iter(gt)))

    def walk_document(self) -> None:
        """Walk the hierarchy under every SampleNode (the flat scan when
        none has one), then warn once about what was not imported."""
        walked = False
        for sn in self.root.iter('SampleNode'):
            walked = self._walk_document_sample(sn) or walked
        if not walked:
            self._flat_scan()
        # Unsupported kinds: report once. EllipsoidGate and QuadrantGate
        # are parsed, so they are named only when one was not imported;
        # listing them whenever present called a correctly imported
        # ellipse "skipped".
        # The unread populations' own warning names their gate types.
        covered = _unread_tags(self.unread)
        skipped = set(self.not_imported) - covered
        for gt in ('CurlyQuad', 'BooleanGate'):
            if gt not in covered and any(True for _ in self.root.iter(gt)):
                skipped.add(gt)
        if skipped:
            log.warning(
                "[WspReader] Skipped unsupported gate types: %s",
                sorted(skipped))
        _log_unread('workspace', self.unread)


class WspReader:
    """
    Parse compensation matrices from a FlowJo v10 .wsp workspace file.

    Usage
    -----
        reader = WspReader('experiment.wsp')
        reader.print_matrices()
        m = reader.get_matrix()           # first matrix
        m = reader.get_matrix('My Comp') # by name
        # m keys: matrix (ndarray), channels (list), prefix, suffix
    """

    def __init__(self, wsp_path: str | os.PathLike[str]) -> None:
        self.path     = wsp_path
        self.matrices: dict[str, dict[str, Any]] = {}
        self.root: ET.Element | None = None        # populated by _parse; needed for gate extraction
        # Populations the last extract_gates() left out (unreadable gate).
        self.unread_populations: list[tuple[str, list[str], int]] = []
        self._parse()

    def _parse(self) -> None:
        try:
            # DTD/entity guard: stdlib etree doesn't resolve external entities
            # (no XXE file-read), but IS vulnerable to internal entity-expansion
            # (billion-laughs / quadratic-blowup) amplification — a few-KB .wsp
            # ballooning to GBs. Reject an internal DTD subset or any ENTITY
            # declaration before parsing (no new dep). A bare
            # `<!DOCTYPE Workspace>`, as FlowJo 9 writes, declares nothing and
            # is let through so the version check below can name the file.
            with open(self.path, 'rb') as _fh:
                _head = _fh.read(1 << 20)        # DOCTYPE must be in the prolog
            if xml_declares_entities(_head):
                raise WspParseError(
                    "refusing to parse a WSP with a DTD/entity declaration "
                    f"(possible entity-expansion attack): {self.path}")
            tree = ET.parse(self.path)
        except FileNotFoundError as e:
            raise WspParseError(f"WSP file not found: {self.path}") from e
        except ET.ParseError as e:
            raise WspParseError(
                f"WSP file is not valid XML ({self.path}): {e}") from e
        root = tree.getroot()
        old = _pre_flowjo10_version(root)
        if old:
            raise WspParseError(
                f"{os.path.basename(self.path)} is a FlowJo {old} workspace. "
                "FlowJo 9 (and older) workspaces are not supported yet; "
                "re-save it in FlowJo 10 and open that file.")
        # Strip XML namespace from BOTH element tags AND attribute keys.
        # FlowJo v10's Gating-ML v2 .wsp files namespace attributes too
        # (gating:min, data-type:name, data-type:value, ...), and
        # ElementTree expands those to {namespace_uri}localname. Without
        # this attribute-side strip, our find/get calls all miss and the
        # reader silently extracts zero matrices and zero gates.
        ns_re = re.compile(r'\{.*?\}')
        for elem in root.iter():
            elem.tag = ns_re.sub('', elem.tag)
            if elem.attrib:
                stripped = {ns_re.sub('', k): v for k, v in elem.attrib.items()}
                elem.attrib.clear()
                elem.attrib.update(stripped)
        self.root = root
        for tag in self._MATRIX_TAGS:
            for node in root.iter(tag):
                self._extract_matrix(node)
        if not self.matrices:
            log.info("[WspReader] No compensation matrices found.")

    # FlowJo's own groups: every sample is in 'All Samples', and FlowJo puts
    # compensation controls in 'Compensation'. Neither is a user's grouping.
    _DEFAULT_GROUPS = frozenset({'all samples', 'compensation'})

    def sample_groups(self) -> dict[str, list[str]]:
        """``{sampleID: [group name, ...]}`` from the workspace's <Groups>,
        in the order the workspace lists them, leaving out FlowJo's own
        'All Samples' and 'Compensation' groups."""
        out: dict[str, list[str]] = {}
        if self.root is None:
            return out
        for gn in self.root.iter('GroupNode'):
            name = (gn.get('name') or '').strip()
            if not name or name.lower() in self._DEFAULT_GROUPS:
                continue
            for ref in gn.iter('SampleRef'):
                sid = ref.get('sampleID')
                if sid is not None and name not in out.setdefault(sid, []):
                    out[sid].append(name)
        return out

    @staticmethod
    def _channel_name(dim_elem: ET.Element) -> str | None:
        """Pull the FCS channel name from a gating <dimension>, stripping
        FlowJo's 'Comp-' prefix so it matches FlowSample.data columns."""
        ch_elem = dim_elem.find('fcs-dimension')
        if ch_elem is None:
            ch_elem = next(dim_elem.iter('fcs-dimension'), None)
        if ch_elem is None:
            return None
        ch_name = ch_elem.get('name') or ch_elem.get('PnN')
        if not ch_name:
            return None
        return strip_comp_marker(ch_name)[0]

    @staticmethod
    def sample_timestep(sample_elem: ET.Element | None,
                        fcs_path: str | os.PathLike[str] | None = None) -> float | None:
        """Seconds per Time tick for a <Sample>.

        When the sample's <Transformations> puts Time on a linear axis with
        a `gain` other than 1, that gain: FlowJo writes there the seconds
        per tick it used. FlowJo 10.6 and 10.8 used ($ETIM - $BTIM) / last
        tick, not the file's $TIMESTEP, and wrote that as the gain (0.0100853
        where $TIMESTEP said 0.01); with the gain, FlowJo's Time counts come
        out exactly, with $TIMESTEP they do not.

        Otherwise as FlowJo 10.10.2 derives it, measured on workspaces
        without such a gain: from the FCS file -- its $TIMESTEP, else
        ($ETIM - $BTIM) / last tick -- ignoring a different $TIMESTEP in the
        workspace <Keywords> (FlowJo applied the file's 0.05 where the
        workspace said 0.01). `fcs_path` is the sample's FCS; None resolves
        the <DataSet> uri. Only when the file cannot be read does the
        <Keywords> value stand in. None when nothing gives one."""
        if sample_elem is None:
            return None
        gain = _wsp_time_gain(sample_elem)
        if gain:
            return gain
        if fcs_path is None:
            ds = sample_elem.find('DataSet')
            fcs_path = _wsp_uri_path(ds.get('uri') if ds is not None else None)
        ts = _fcs_timestep(fcs_path)
        if ts:
            return ts
        if fcs_path and os.path.isfile(fcs_path):
            return None              # the file itself gives FlowJo no factor
        for kw in sample_elem.findall('Keywords/Keyword'):
            if (kw.get('name') or '').upper() == '$TIMESTEP':
                return _positive_float(kw.get('value'))
        return None

    def extract_gates(self, *, sample_node: ET.Element | None = None,
                      timestep: float | None = None) -> list[dict[str, Any]]:
        """Extract supported gates from the .wsp as a list of gate dicts in
        topological order (parents before children).

        Parameters
        ----------
        sample_node : ET.Element | None
            When None (default), walk every ``<SampleNode>`` in the
            document and return a single flattened list — preserves the
            historical behaviour used by :func:`read_template_gates` and
            the GUI's per-template loader.
            When given a specific ``<SampleNode>`` element, walk only
            that sample's gate tree. Used by the gate editor's
            ``Add FCS / Workspace`` to attach the right gate tree to
            the right sample when opening a multi-sample .wsp.
        timestep : float | None
            Seconds per Time tick for `sample_node`. FlowJo writes Time gate
            coordinates in seconds; they are divided by this so they come
            back in the raw ticks OpenFlo's data holds. None derives it as
            FlowJo does, from the sample's FCS (see `sample_timestep`); pass
            it when the caller located the FCS somewhere else. No factor
            means no conversion.

        Each returned gate carries an `_import_id` (a temporary string, stable
        within this call) and a `parent_id` referencing another gate's
        `_import_id` (or None for roots). The GUI editor remaps these to its
        own gate_ids on load.

        Hierarchy comes from FlowJo's nested <Population>/<Subpopulations>
        structure. If no such hierarchy is found, falls back to a flat scan
        of every RectangleGate / PolygonGate in the document.

        Supports:
          • 1-D RectangleGate (min only)   → 'threshold'
          • 1-D RectangleGate (min + max)  → 'interval'
          • 2-D RectangleGate              → 'rect'
          • PolygonGate                    → 'polygon'
          • EllipsoidGate (Gating-ML)      → 'ellipsoid'
          • EllipsoidGate (drawn in FlowJo: foci + edge in display space)
                                           → 'ellipsoid' on linear axes,
                                             'polygon' on logicle / biex axes;
                                             on any other axis a
                                             'flowjo_ellipse' placeholder, which
                                             admits no events (see
                                             wsp_ellipse_gate)
          • QuadrantGate                   → 4 linked 'rect' (quad_set)
          • AndNode / OrNode / NotNode     → 'boolean' over the populations
                                             its <Dependent>s name
        Skipped (with a warning): CurlyQuad, Gating-ML BooleanGate, and any
        EllipsoidGate / QuadrantGate that could not be read. A population
        whose gate is skipped is left out together with the populations
        under it (`self.unread_populations` lists them for this call); a
        population with no <Gate> at all passes its children to its parent.

        Each gate carries 'wsp_comp_channels': its channels FlowJo names with
        the compensated marker ('Comp-' or <...>; see wsp_comp_channels).
        """
        if self.root is None:
            return []
        walk = _GateWalk(self)
        self.unread_populations = walk.unread
        if sample_node is not None:
            walk.walk_sample(sample_node, timestep)
        else:
            walk.walk_document()
        _log_extracted(walk.gates)
        return walk.gates

    def _extract_matrix(self, node: ET.Element) -> None:
        """Parse a matrix element and register it in `self.matrices` under its
        name. A name already holding a DIFFERENT matrix gets a numbered key
        instead: FlowJo calls every file's own SPILL "Acquisition-defined",
        and keying by bare name kept only the last of them, so a lookup by
        name handed every sample the last file's matrix."""
        m = self._parse_matrix(node)
        if m is None:
            return
        name, key, k = m['name'], m['name'], 1
        while key in self.matrices:
            old = self.matrices[key]
            if (old['channels'] == m['channels']
                    and np.array_equal(old['matrix'], m['matrix'])):
                return                   # a sample's copy of a listed matrix
            k += 1
            key = f'{name} ({k})'
        self.matrices[key] = m

    def _parse_matrix(self, node: ET.Element) -> dict[str, Any] | None:
        """A compensation-matrix element as dict(matrix, channels, prefix,
        suffix, name), or None when it holds no usable values."""
        name   = node.get('name') or node.get('matrixName') or 'unnamed'
        prefix = node.get('prefix', 'Comp-')
        suffix = node.get('suffix', '')
        channels = [p.get('name') or p.get('PnN')
                    for p in node.iter('parameter')
                    if p.get('name') or p.get('PnN')]
        if not channels:
            return None
        n      = len(channels)
        values = []
        for vn in node.iter('spilloverValues'):
            try:
                values = [float(v) for v in vn.get('values', '').split(',')]
            except ValueError:
                pass
        matrix = None
        if len(values) == n * n:
            matrix = np.array(values).reshape(n, n)
        else:
            rows = []
            for row in node.iter('spilloverRow'):
                try:
                    rows.append([float(v) for v in row.get('values','').split(',')])
                except ValueError:
                    pass
            if len(rows) == n:
                matrix = np.array(rows)
        # Gating-ML v2 layout (FlowJo v10 .wsp):
        #   <spillover parameter="src"><coefficient parameter="dst" value="x"/></spillover>
        # Build the matrix row-by-row from the source/destination channel
        # pairs against our channel index. Identity diagonal as the default.
        if matrix is None:
            ch_idx = {c: i for i, c in enumerate(channels)}
            m = np.eye(n)
            seen = 0
            declared = set()
            for sp in node.iter('spillover'):
                src = sp.get('parameter')
                if src not in ch_idx:
                    continue
                declared.add(src)
                i = ch_idx[src]
                for coef in sp.iter('coefficient'):
                    dst = coef.get('parameter')
                    val = coef.get('value')
                    if dst in ch_idx and val is not None:
                        try:
                            m[i, ch_idx[dst]] = float(val)
                            seen += 1
                        except ValueError:
                            pass
            if seen >= n:    # at minimum the diagonal entries should show up
                # `seen` counts COEFFICIENTS, not channels, so a file in which
                # a couple of detectors carry a full row clears the bar while
                # others are never mentioned at all. Those keep the identity
                # row they were initialised with, which silently means "this
                # detector has no spillover" — and spillover from every other
                # channel then leaks into it uncorrected. Verified: 2 of 4
                # channels declaring 8 coefficients was accepted with the
                # other 2 left uncompensated. Still accepted, because the
                # declared rows are usable and a partial matrix beats none,
                # but not in silence.
                missing = [c for c in channels if c not in declared]
                if missing:
                    log.warning(
                        "[WspReader] '%s': no spillover declared for %d of %d "
                        "channels (%s). They are left UNCOMPENSATED — check "
                        "the workspace if you expected them to be corrected.",
                        name, len(missing), n,
                        ', '.join(missing))  # pyright: ignore[reportArgumentType, reportCallIssue]
                matrix = m
        if matrix is None:
            log.info(f"[WspReader] Could not parse values for '{name}'.")
            return None
        return {'matrix': matrix, 'channels': channels, 'prefix': prefix,
                'suffix': suffix, 'name': name}

    _MATRIX_TAGS = ('CompensationMatrix', 'spilloverMatrix', 'Compensation')

    def sample_matrix(self, sample_elem: ET.Element | None) -> dict[str, Any] | None:
        """The matrix a <Sample> carries for itself (FlowJo writes a copy of
        it directly inside the Sample, after <DataSet>), parsed; None when
        the sample has none -- FlowJo then applies its Acquisition-defined
        matrix, i.e. the FCS file's own $SPILL."""
        if sample_elem is None:
            return None
        for child in sample_elem:
            if child.tag in self._MATRIX_TAGS:
                m = self._parse_matrix(child)
                if m is not None:
                    return m
        return None

    def matrix_for_fcs(self, fcs_path: str | os.PathLike[str] | None,
                       ) -> tuple[bool, dict[str, Any] | None]:
        """`(listed, matrix)` for the FCS at `fcs_path`: `listed` is True when
        a <Sample> of this workspace has that file as its DataSet (matched by
        full path, else by an unambiguous file name), and `matrix` is that
        sample's own matrix (see `sample_matrix`) or None when it has none."""
        if self.root is None or not fcs_path:
            return False, None
        want = os.path.normcase(os.path.abspath(str(fcs_path)))
        base = os.path.basename(want)
        exact, by_name = [], []
        for sample_elem in self.root.iter('Sample'):
            ds = sample_elem.find('DataSet')
            p = _wsp_uri_path(ds.get('uri') if ds is not None else None)
            if not p:
                continue
            p = os.path.normcase(os.path.abspath(p))
            if p == want:
                exact.append(sample_elem)
            elif os.path.basename(p) == base:
                by_name.append(sample_elem)
        hits = exact or by_name
        if len(hits) != 1:
            if len(hits) > 1:
                log.warning("[WspReader] %d samples in %s point at %s -- not "
                            "choosing between them.", len(hits),
                            os.path.basename(str(self.path)), base)
            return False, None
        return True, self.sample_matrix(hits[0])

    def distinct_matrix_count(self) -> int:
        """Number of compensation matrices read from the workspace."""
        return len(self.matrices)

    def get_matrix(self, name: str | None = None) -> dict[str, Any]:
        """The matrix dict called ``name``, or the first one when None. Raises
        RuntimeError when there are none and KeyError for an unknown name."""
        if not self.matrices:
            raise RuntimeError("No matrices available.")
        if name is None:
            return next(iter(self.matrices.values()))
        if name not in self.matrices:
            raise KeyError(f"'{name}' not found. Available: {list(self.matrices)}")
        return self.matrices[name]

    def print_matrices(self) -> None:
        """Log each matrix's name and channel list."""
        for name, m in self.matrices.items():
            log.info(f"  '{name}' — {len(m['channels'])} ch: {m['channels']}")
