"""Importing a BD FACSDiva experiment exported as XML -- Tk-free.

FACSDiva's "Export > Experiment" writes an XML file whose root is
``<bdfacs>``: the experiment holds specimens, each specimen holds tubes, and
each tube names its FCS file (``<data_filename>``), its instrument settings
(per parameter: log display, biexponential scale, compensation
coefficients) and its gates (``<gates>``, one ``<gate>`` per population,
parents named by path such as ``All Events\\P1``).

``plan_diva`` reads all of it into the same records ``wsp_open.plan_wsp``
gives for a FlowJo workspace (``WspPlan`` / ``WspSample``), so the desktop
editor and the headless document import a FACSDiva experiment the way they
import a workspace; ``wsp_open.plan_wsp`` sends a ``<bdfacs>`` file here.

The reader follows the layout of the exported XML, as the open-source CytoML
package documents and reads it; no CytoML code is used. Coordinates:

* A gate axis whose ``is_<x|y>_parameter_scaled`` is true is on FACSDiva's
  0..4096 plot scale. On a log axis that is a biexponential (logicle) scale
  with T = 262144, M = 4.5, A = 0 and W from the gate's own
  ``<x|y>_parameter_scale_value`` r: W = (M - log10(T / r)) / 2, at least 0
  (the formula CytoML gives, from BD). On a linear axis it is 0..262144.
* A log axis that is not scaled holds log10 of the value.
* A linear axis that is not scaled holds the value.

Gates come back in linear data units, as ``WspReader.extract_gates`` gives
them; polygon edges keep FACSDiva's shape through ``flowjo_edge_axes``. A
gate side on the edge of FACSDiva's plot (0 or 4095 on the plot scale, the
top or bottom decade on a log axis, 262143 on a linear one) is open, as
FACSDiva counts the events piled on that edge: the four quadrants of a
quadrant gate add up to their parent.

Compensation: a tube with compensation on carries the coefficients of its
compensation matrix (the inverse of the spillover matrix); the sample is
opened with the spillover matrix they give. A tube with compensation off is
opened uncompensated, as it was gated, and the load notes say so.

Core module: no Tk.
"""
from __future__ import annotations

import logging
import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .exceptions import WspParseError
from .gatingml import xml_local
from .scales import OPEN

if TYPE_CHECKING:
    from .gate_store import Gate
    from .wsp_open import WspPlan, WspSample

log = logging.getLogger(__name__)

# FACSDiva's plot scale, and the biexponential it puts a log axis on there.
DIVA_PLOT_CHANNELS = 4096.0
DIVA_T = 262144.0
DIVA_M = 4.5
# The top of a linear axis (an 18-bit value).
DIVA_LINEAR_TOP = 262143.0
_ROOT_EVENTS = 'All Events'
# The comp record's source for a tube FACSDiva did not compensate.
DIVA_COMP_OFF = 'diva:compensation off'
_SEP = '\\'


class DivaUnreadable(WspParseError):
    """The FACSDiva XML could not be read; the message says why."""


def is_diva_xml(path: str | os.PathLike[str]) -> bool:
    """True when `path` is an XML file whose root element is ``<bdfacs>``
    (a FACSDiva experiment export). Reads only the start of the file."""
    try:
        for _ev, el in ET.iterparse(path, events=('start',)):
            return xml_local(el.tag) == 'bdfacs'
    except (ET.ParseError, OSError):
        return False
    return False


def parse_diva(path: str | os.PathLike[str]) -> ET.Element:
    """The ``<bdfacs>`` root of the FACSDiva XML at `path`. Raises
    DivaUnreadable for a file that is missing, not XML, declares entities
    (as WspReader refuses them) or is not a FACSDiva export."""
    from .wsp_reader import xml_declares_entities
    try:
        with open(path, 'rb') as fh:
            head = fh.read(1 << 20)
        if xml_declares_entities(head):
            raise DivaUnreadable(
                f'refusing a FACSDiva XML with a DTD or entity declaration: '
                f'{os.path.basename(str(path))}')
        root = ET.parse(path).getroot()
    except OSError as e:
        raise DivaUnreadable(f'cannot read {os.path.basename(str(path))}: {e}') from e
    except ET.ParseError as e:
        raise DivaUnreadable(f'not valid XML ({os.path.basename(str(path))}): {e}') from e
    if root.tag != 'bdfacs':
        raise DivaUnreadable(f'not a FACSDiva experiment (root <{root.tag}>)')
    return root


# ── Axes ─────────────────────────────────────────────────────────────────────

def diva_biexp_w(r: float) -> float:
    """W of FACSDiva's biexponential plot scale for scale value `r`,
    between 0 and M / 2 (the widest a logicle scale can be)."""
    r = abs(float(r))
    if r == 0:
        return 0.0
    return min(DIVA_M / 2.0, max(0.0, (DIVA_M - math.log10(DIVA_T / r)) / 2.0))


@dataclass(frozen=True)
class DivaAxis:
    """How one gate axis is stored: on the 0..4096 plot scale (`scaled`),
    on a log display (`log`), with the biexponential scale value `r`."""
    scaled: bool
    log: bool
    r: float
    log_floor: float = 2.6     # the lowest value a log axis shows

    def logicle(self) -> dict[str, float]:
        """The logicle parameters (T, W, M, A) a scaled log axis shows."""
        return {'T': DIVA_T, 'W': diva_biexp_w(self.r), 'M': DIVA_M, 'A': 0.0}

    def to_linear(self, c: np.ndarray) -> np.ndarray:
        """Stored coordinates `c` as linear data values."""
        from .channel_transforms import inverse_transform_values
        c = np.asarray(c, dtype=float)
        if self.scaled and self.log:
            p = self.logicle()
            return np.asarray(inverse_transform_values(
                c / DIVA_PLOT_CHANNELS, 'logicle', t=int(p['T']), m=p['M'],
                w=p['W'], a=int(p['A'])), dtype=float)
        if self.scaled:
            return c / DIVA_PLOT_CHANNELS * DIVA_T
        if self.log:
            return np.power(10.0, c)
        return c

    def ends(self) -> tuple[float | None, float | None]:
        """The stored coordinates of the plot's bottom and top edge; a gate
        side at or past one is open. None: that end is not an edge."""
        if self.scaled:
            return 0.0, DIVA_PLOT_CHANNELS - 1.0
        if self.log:
            return math.log10(self.log_floor), math.log10(DIVA_LINEAR_TOP)
        return None, DIVA_LINEAR_TOP

    def edge_axis(self) -> dict[str, Any] | None:
        """The axis FACSDiva draws polygon edges straight on, as a
        'flowjo_edge_axes' spec; None when straight in data units."""
        if self.scaled and self.log:
            return {'kind': 'logicle', **self.logicle()}
        if self.log:
            return {'kind': 'log', 'offset': self.log_floor, 'decades': 1.0}
        return None


def _truthy(el: ET.Element, tag: str) -> bool:
    return (el.findtext(tag) or '').strip().lower() == 'true'


def _float(text: str | None, default: float = 0.0) -> float:
    try:
        v = float(text)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def gate_axis(gate_el: ET.Element, which: str, log_floor: float) -> DivaAxis:
    """The `which` ('x' or 'y') axis of a FACSDiva <gate>."""
    return DivaAxis(scaled=_truthy(gate_el, f'is_{which}_parameter_scaled'),
                    log=_truthy(gate_el, f'is_{which}_parameter_log'),
                    r=_float(gate_el.findtext(f'{which}_parameter_scale_value')),
                    log_floor=log_floor)


def experiment_log_floor(root: ET.Element) -> float:
    """The lowest value FACSDiva's log axes show: the top (262144) less the
    experiment's ``<log_decades>`` (4 or 5; 5 when not given)."""
    dec = _float(root.findtext('experiment/log_decades'), 5.0)
    return DIVA_T / 10.0 ** (dec if dec > 0 else 5.0)


# ── Regions ──────────────────────────────────────────────────────────────────

def _points(region: ET.Element) -> np.ndarray:
    pts = [[_float(p.get('x'), math.nan), _float(p.get('y'), math.nan)]
           for p in region.iter('point')]
    a = np.asarray(pts, dtype=float).reshape(-1, 2)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('a region without numeric points')
    return a


def _open_side(axis: DivaAxis, c: float, lo_side: bool) -> bool:
    lo, hi = axis.ends()
    if lo_side:
        return lo is not None and c <= lo
    return hi is not None and c >= hi


def _bound(axis: DivaAxis, c: float, lo_side: bool) -> float:
    if _open_side(axis, c, lo_side):
        return -OPEN if lo_side else OPEN
    return float(axis.to_linear(np.array([c]))[0])


def _interval(ch: str, axis: DivaAxis, xs: np.ndarray) -> Gate:
    lo = _bound(axis, float(xs.min()), True)
    hi = _bound(axis, float(xs.max()), False)
    if hi >= OPEN and lo > -OPEN:
        return {'kind': 'threshold', 'channel': ch, 'value': lo}
    return {'kind': 'interval', 'channel': ch, 'lo': lo, 'hi': hi}


def _is_box(pts: np.ndarray) -> bool:
    """Four points with two x and two y values: an axis-aligned box."""
    return (len(pts) == 4 and len(np.unique(pts[:, 0])) == 2
            and len(np.unique(pts[:, 1])) == 2)


def _rect(xc: str, yc: str, ax: DivaAxis, ay: DivaAxis, pts: np.ndarray) -> Gate:
    return {'kind': 'rect', 'x_channel': xc, 'y_channel': yc,
            'x0': _bound(ax, float(pts[:, 0].min()), True),
            'x1': _bound(ax, float(pts[:, 0].max()), False),
            'y0': _bound(ay, float(pts[:, 1].min()), True),
            'y1': _bound(ay, float(pts[:, 1].max()), False)}


def _polygon(xc: str, yc: str, ax: DivaAxis, ay: DivaAxis, pts: np.ndarray) -> Gate:
    """A polygon in data units, its edges straight on FACSDiva's axes,
    cut at each plot edge it reaches (as the plot shows it) and opened past
    the cut, so the events FACSDiva piles on that edge are inside."""
    from .wsp_reader import clip_polygon_axis, open_polygon_at
    vs = pts.copy()
    for i, axis in enumerate((ax, ay)):
        lo, hi = axis.ends()
        vs = clip_polygon_axis(vs, i, -np.inf if lo is None else lo,
                               np.inf if hi is None else hi)
    if len(vs) < 3:
        raise ValueError('a polygon wholly past the plot edge')
    data = np.column_stack([ax.to_linear(vs[:, 0]), ay.to_linear(vs[:, 1])])
    for i, axis in enumerate((ax, ay)):
        for bound, far in zip(axis.ends(), (-OPEN, OPEN), strict=True):
            if bound is not None:
                vs, data = open_polygon_at(vs, data, i, bound, far)
    g: Gate = {'kind': 'polygon', 'x_channel': xc, 'y_channel': yc,
               'vertices': data.tolist()}
    edges = [ax.edge_axis(), ay.edge_axis()]
    if any(e is not None for e in edges):
        g['flowjo_edge_axes'] = edges
    return g


def region_gate(gate_el: ET.Element, log_floor: float) -> Gate:
    """The OpenFlo gate of a FACSDiva region gate (a rectangle, polygon,
    snap-to or interval region), in linear data units. ValueError for a
    region OpenFlo does not read."""
    region = gate_el.find('region')
    if region is None:
        raise ValueError('a region gate without a <region>')
    rtype = region.get('type') or ''
    xc, yc = region.get('xparm'), region.get('yparm')
    if not xc:
        raise ValueError('a region without an x parameter')
    pts = _points(region)
    ax = gate_axis(gate_el, 'x', log_floor)
    if rtype == 'INTERVAL_REGION':
        return _interval(xc, ax, pts[:, 0])
    if rtype not in ('RECTANGLE_REGION', 'POLYGON_REGION'):
        raise ValueError(f'a {rtype or "untyped"} region')
    if not yc:
        raise ValueError(f'a {rtype} without a y parameter')
    ay = gate_axis(gate_el, 'y', log_floor)
    if _is_box(pts):
        return _rect(xc, yc, ax, ay, pts)
    if rtype == 'RECTANGLE_REGION' or len(pts) < 3:
        raise ValueError(f'a {rtype} with {len(pts)} points')
    return _polygon(xc, yc, ax, ay, pts)


# ── A tube's gate tree ───────────────────────────────────────────────────────

_BOOLEAN_TYPES = {'AND_Classifier': 'and', 'OR_Classifier': 'or',
                  'NOT_Classifier': 'not'}
# A region gate, possibly drawn with the snap-to tool (the region holds the
# outline it settled on).
_REGION_TYPES = ('Region_Classifier', 'SnapTo_Classifier',
                 'SnapTo_Interval_Classifier')
# The source of all events, and a quadrant's centre (its four quadrants are
# region gates of their own).
_SKIPPED_TYPES = ('EventSource_Classifier', 'Composite_Classifier')


def _path(text: str | None) -> str:
    return (text or '').strip()


class _TubeGates:
    """Reads one tube's <gates> into gate dicts (see tube_gates)."""

    def __init__(self, log_floor: float) -> None:
        self.log_floor = log_floor
        self.gates: list[Gate] = []
        self.ids: dict[str, str] = {_ROOT_EVENTS: ''}
        self.dropped: set[str] = set()
        self.unread: list[tuple[str, list[str], int]] = []
        self.booleans: list[tuple[Gate, list[str]]] = []
        self.children: dict[str, list[str]] = {}
        # A quadrant's centre (linear x, y) by its path, and its quadrants.
        self.centres: dict[str, tuple[float, float]] = {}
        self.quadrants: dict[str, list[Gate]] = {}

    def _subtree_size(self, path: str) -> int:
        kids = self.children.get(path, [])
        return len(kids) + sum(self._subtree_size(k) for k in kids)

    def _leave_out(self, path: str, name: str, what: str) -> None:
        self.dropped.add(path)
        self.unread.append((name, [what], self._subtree_size(path)))

    def read(self, gate_els: list[ET.Element]) -> None:
        for el in gate_els:
            parent = _path(el.findtext('parent'))
            if parent:
                self.children.setdefault(parent, []).append(
                    _path(el.get('fullname')))
        # Parents first: a gate's path runs through its parent's.
        for el in sorted(gate_els,
                         key=lambda e: _path(e.get('fullname')).count(_SEP)):
            self._one(el)
        self._resolve_booleans()
        self._link_quadrants()

    def _one(self, el: ET.Element) -> None:
        gtype = el.get('type') or ''
        path = _path(el.get('fullname'))
        if gtype == 'Composite_Classifier':
            self._centre(el, path)
        if gtype in _SKIPPED_TYPES or not path:
            return
        name = (el.findtext('name') or path.rsplit(_SEP, 1)[-1]).strip()
        parent = _path(el.findtext('parent')) or _ROOT_EVENTS
        if parent in self.dropped:
            self.dropped.add(path)
            return
        if parent not in self.ids:
            self._leave_out(path, name, 'parent not in the file')
            return
        try:
            g = self._gate(el, gtype)
        except ValueError as e:
            log.warning("[FACSDiva] population '%s' not imported: %s", name, e)
            region = el.find('region')
            what = region.get('type') if region is not None else None
            self._leave_out(path, name, what or gtype or 'gate')
            return
        g['label'] = name
        g['parent_id'] = self.ids[parent] or None
        g['_import_id'] = f'diva_{len(self.gates)}'
        self.ids[path] = g['_import_id']
        self.gates.append(g)
        for i in el.findall('input'):
            if g['kind'] == 'rect':
                self.quadrants.setdefault(_path(i.text), []).append(g)

    def _gate(self, el: ET.Element, gtype: str) -> Gate:
        if gtype in _REGION_TYPES:
            return region_gate(el, self.log_floor)
        if gtype in _BOOLEAN_TYPES or gtype == 'RestOf_Classifier':
            inputs = [_path(i.text) for i in el.findall('input')]
            op = _BOOLEAN_TYPES.get(gtype, 'not')
            if not inputs:
                raise ValueError(f'a {gtype} without inputs')
            g: Gate = {'kind': 'boolean', 'op': op, 'operands': []}
            self.booleans.append((g, inputs))
            return g
        raise ValueError(f'a {gtype or "untyped"} gate')

    def _centre(self, el: ET.Element, path: str) -> None:
        """Note the centre of a quadrant gate (a binner region)."""
        region = el.find('region')
        if region is None or region.get('type') != 'BINNER_REGION':
            return
        try:
            pt = _points(region)[0]
        except ValueError:
            return
        ax = gate_axis(el, 'x', self.log_floor)
        ay = gate_axis(el, 'y', self.log_floor)
        self.centres[path] = (float(ax.to_linear(np.array([pt[0]]))[0]),
                              float(ay.to_linear(np.array([pt[1]]))[0]))

    def _link_quadrants(self) -> None:
        """The four rects of one FACSDiva quadrant gate share a `quad_set`
        and its centre, as a FlowJo quadrant imports, so they move
        together in OpenFlo."""
        for k, (path, rects) in enumerate(self.quadrants.items()):
            if path not in self.centres or len(rects) != 4:
                continue
            cx, cy = self.centres[path]
            for g in rects:
                g['quad_set'] = f'diva_quad_{k}'
                g['quad_origin_x'], g['quad_origin_y'] = cx, cy

    def _resolve_booleans(self) -> None:
        """Point each boolean at its inputs' import ids. An input not
        imported leaves the boolean with no operands, so it reads NA
        (gate_eval.broken_booleans), never every event."""
        for g, inputs in self.booleans:
            got = [self.ids.get(p) for p in inputs]
            if all(got):
                g['operands'] = got
            else:
                log.warning("[FACSDiva] boolean '%s': an input was not "
                            "imported, so it reads NA", g.get('label'))


def tube_gates(tube: ET.Element, log_floor: float,
               ) -> tuple[list[Gate], list[tuple[str, list[str], int]]]:
    """A tube's gates, parents before children, each with an `_import_id`
    and a `parent_id` naming another's (as WspReader.extract_gates gives
    them), and ``(population, [gate type], n under)`` for each population
    left out (with everything under it) because OpenFlo cannot read it."""
    walk = _TubeGates(log_floor)
    walk.read(tube.findall('gates/gate'))
    return walk.gates, walk.unread


# ── Compensation ─────────────────────────────────────────────────────────────

def _settings(*holders: ET.Element | None) -> ET.Element | None:
    """The first <instrument_settings> among `holders` (tube, specimen,
    experiment), as FACSDiva falls back to the shared settings."""
    for h in holders:
        s = h.find('instrument_settings') if h is not None else None
        if s is not None and s.find('parameter') is not None:
            return s
    return None


def tube_compensation(settings: ET.Element | None) -> dict[str, Any] | None:
    """The comp record (``{'channels', 'matrix', 'source'}``, a spillover
    matrix) a tube's gates were drawn on, or None to keep the FCS file's
    own matrix. Compensation on: the spillover matrix whose inverse is the
    tube's compensation coefficients (parameter j's coefficients are column
    j). Compensation off: the identity over the parameters FACSDiva could
    compensate, so the sample opens uncompensated, as gated."""
    if settings is None:
        return None
    params = [p for p in settings.findall('parameter') if p.get('name')]
    on = _truthy(settings, 'compensation_enabled')
    comp = [p for p in params if p.find('compensation') is not None]
    if not on:
        names = [p.get('name') for p in params
                 if _truthy(p, 'can_be_compensated')]
        if not names:
            return None
        return {'channels': names, 'matrix': np.eye(len(names)).tolist(),
                'source': DIVA_COMP_OFF}
    if not comp:
        return None
    cols = [[_float(c.text, math.nan) for c in p.iter('compensation_coefficient')]
            for p in comp]
    n = len(comp)
    if any(len(c) != n for c in cols):
        log.warning("[FACSDiva] compensation coefficients do not form a "
                    "square matrix; the FCS file's own matrix is used")
        return None
    cmat = np.asarray(cols, dtype=float).T
    if not np.isfinite(cmat).all():
        return None
    try:
        spill = np.linalg.inv(cmat)
    except np.linalg.LinAlgError:
        log.warning("[FACSDiva] the compensation matrix cannot be inverted; "
                    "the FCS file's own matrix is used")
        return None
    return {'channels': [p.get('name') for p in comp],
            'matrix': spill.tolist(), 'source': 'diva:experiment'}


# ── Finding the FCS files ────────────────────────────────────────────────────

def _key(v: str | None) -> str:
    return (v or '').strip().lower()


class DivaFcsIndex:
    """The FCS files under `roots`, by what FACSDiva writes into each: its
    original file name ($FIL, the name the XML's <data_filename> gives) and
    its specimen ($SRC), tube (TUBE NAME) and experiment (EXPERIMENT NAME).
    An exported file is often renamed after its specimen and tube
    while its XML entry keeps the original name; these keywords still tie
    the two. Built on first use, reading only each file's TEXT segment."""

    MAX_DEPTH = 6

    def __init__(self, roots: list[str]) -> None:
        self.roots = [r for r in dict.fromkeys(roots) if r and os.path.isdir(r)]
        self._rows: list[tuple[str, dict[str, str]]] | None = None

    def _files(self) -> list[str]:
        out: list[str] = []
        for root in self.roots:
            base = os.path.abspath(root).rstrip(os.sep).count(os.sep)
            for dirpath, dirnames, filenames in os.walk(root):
                depth = os.path.abspath(dirpath).count(os.sep) - base
                dirnames[:] = ([] if depth >= self.MAX_DEPTH else
                               [d for d in dirnames if not d.startswith('.')])
                out.extend(os.path.join(dirpath, f) for f in filenames
                           if f.lower().endswith('.fcs'))
        return list(dict.fromkeys(out))

    def rows(self) -> list[tuple[str, dict[str, str]]]:
        """(path, TEXT keywords) of each FCS file found, read once."""
        if self._rows is None:
            from .fcs_text import fcs_text_pairs
            self._rows = []
            for f in self._files():
                try:
                    kw = {k.strip().upper(): v.strip() for k, v in fcs_text_pairs(f)}
                except (OSError, ValueError, IndexError):
                    continue
                self._rows.append((f, kw))
        return self._rows

    def find(self, data_filename: str, specimen: str, tube: str,
             experiment: str) -> str | None:
        """The one file FACSDiva wrote for this tube, or None when none or
        several match. A file that names another experiment (EXPERIMENT
        NAME) is never this tube's: FACSDiva's default specimen and tube
        names repeat from one experiment to the next."""
        exp = _key(experiment)
        rows = [(f, kw) for f, kw in self.rows()
                if not exp or _key(kw.get('EXPERIMENT NAME')) in ('', exp)]
        want = _key(os.path.basename(data_filename.replace('\\', '/')))
        by_fil = [f for f, kw in rows
                  if want and _key(os.path.basename(kw.get('$FIL', '').replace('\\', '/'))) == want]
        if len(by_fil) == 1:
            return by_fil[0]
        cands = by_fil or [f for f, _kw in rows]
        kws = dict(rows)
        hits = [f for f in cands
                if _key(kws[f].get('$SRC')) == _key(specimen)
                and _key(kws[f].get('TUBE NAME')) == _key(tube)]
        if len(hits) > 1 and experiment:
            hits = [f for f in hits
                    if _key(kws[f].get('EXPERIMENT NAME')) == _key(experiment)]
        return hits[0] if len(hits) == 1 else None


# ── The plan ─────────────────────────────────────────────────────────────────

def _root_events(tube: ET.Element) -> int | None:
    for g in tube.findall('gates/gate'):
        if g.get('type') == 'EventSource_Classifier':
            n = _float(g.findtext('num_events'), -1.0)
            return int(n) if n > 0 else None
    return None


def _finder(locator: Any) -> Callable[[str | None], str | None] | None:
    from .wsp_open import FcsFinder
    if locator is None:
        return None
    return locator.find if isinstance(locator, FcsFinder) else locator


def plan_diva(path: str, locator: Any = None) -> WspPlan:
    """Read the FACSDiva experiment XML at `path` as a ``wsp_open.WspPlan``:
    one ``WspSample`` per tube whose FCS file was found, in the order the
    XML lists them, its gates in linear units (``wsp_open.gates_on_sample``
    puts them on a sample's stored units), its group the tube's specimen
    and its comp record from the tube's settings (``tube_compensation``).

    A tube's file is looked for by its <data_filename> with `locator` (as
    ``plan_wsp`` takes it; None uses ``compare.FcsLocator`` on the XML's
    folder), then among the FCS files under the XML's folder (and the
    locator's FCS folder) by the keywords FACSDiva writes (``DivaFcsIndex``).
    A tube that recorded no events and has no file is passed over. Raises
    DivaUnreadable when the file cannot be read."""
    from .wsp_open import WspPlan
    root = parse_diva(path)
    if locator is None:
        from .compare import FcsLocator
        locator = FcsLocator(path)
    find = _finder(locator)
    roots = [os.path.dirname(os.path.abspath(path)),
             getattr(locator, 'fcs_dir', None) or '']
    index = DivaFcsIndex(roots)
    plan = WspPlan()
    exp = root.find('experiment')
    if exp is None:
        return plan
    floor = experiment_log_floor(root)
    for specimen in exp.findall('specimen'):
        for tube in specimen.findall('tube'):
            rec = _plan_tube(exp, specimen, tube, floor, find, index, plan)
            if rec is not None:
                plan.samples.append(rec)
    plan.ambiguous = list(getattr(locator, 'ambiguous', None) or [])
    return plan


def _plan_tube(exp: ET.Element, specimen: ET.Element, tube: ET.Element,
               floor: float, find: Callable[[str | None], str | None] | None,
               index: DivaFcsIndex, plan: WspPlan) -> WspSample | None:
    from .wsp_open import WspSample
    from .wsp_reader import fcs_event_total, wsp_event_mismatch
    sname = specimen.get('name') or ''
    tname = tube.get('name') or ''
    label = f'{sname} / {tname}' if sname else tname or '(unnamed)'
    data_filename = (tube.findtext('data_filename') or '').strip()
    events = _root_events(tube)
    fcs = find(data_filename) if (find and data_filename) else None
    if fcs is None:
        fcs = index.find(data_filename, sname, tname, exp.get('name') or '')
    if fcs is None:
        if events:
            plan.unresolved.append(label)
        return None
    why = wsp_event_mismatch(events, fcs_event_total(fcs))
    if why:
        plan.mismatched.append((label, why.replace('workspace', 'FACSDiva XML')))
    gates, unread = tube_gates(tube, floor)
    plan.unread.extend(unread)
    comp = tube_compensation(_settings(tube, specimen, exp))
    if comp is not None and comp['source'] == DIVA_COMP_OFF:
        plan.notes.append(f"'{label}': compensation was off for this tube in "
                          "FACSDiva, so it is opened uncompensated, as it "
                          "was gated there")
    return WspSample(node_name=label, fcs_path=fcs, gates=gates,
                     stat_axes={}, groups=[sname] if sname else [], comp=comp)
