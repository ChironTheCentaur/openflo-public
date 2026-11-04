"""The FlowJo .wsp workspace writer: ``WspWriter`` builds a FlowJo v10
workspace (Gating-ML 2.0 gates, compensation, sample groups) from gate dicts,
plus the export checks it runs on FlowJo's axis floors and Time edges.

Core module: numpy and the standard library's ElementTree. No Tk, and it does
not import ``openflo.pipeline`` (pipeline imports it).
"""

import logging
import os
import re
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from . import fcs_read as _fcs_read
from .channel_names import is_time_channel
from .channel_transforms import is_unknown_spec
from .fcs_text import fcs_text_pairs as _fcs_text_pairs
from .fcs_text import fcs_timestep as _fcs_timestep
from .fcs_text import positive_float as _positive_float
from .fcs_text import scale_time_coords
from .fcs_text import xml_safe as _xml_safe
from .gate_text import WSP_BOOLEAN_TAGS as _WSP_BOOLEAN_TAGS
from .gate_text import unique_population_name, wsp_population_name
from .scales import OPEN
from .wsp_gate_coords import gate_scale_fns, map_gate_coords
from .wsp_reader import BIEX_KEPT_FROM as _BIEX_KEPT_FROM
from .wsp_reader import BIEX_OPENED_UPTO as _BIEX_OPENED_UPTO
from .wsp_reader import WSP_FLOWJO_VERSION, WSP_OPENFLO_ATTR, WspReader, fcs_event_total
from .wsp_reader import ellipse_polygon as _ellipse_polygon

# Name kept as 'openflo.pipeline' so existing log filters and caplog tests see
# the writer's messages.
log = logging.getLogger('openflo.pipeline')


# ── FlowJo .wsp writer ────────────────────────────────────────────────────────
#
# Emits Gating-ML v2 XML that FlowJo v10 reads. Mirrors the structure WspReader
# expects (and that the real workspaces in our test set use), so a round-trip
# `extract_gates → write → extract_gates` preserves every gate.
#
# Usage:
#     w = WspWriter(cytometer='LSRFortessa')
#     w.set_compensation(['FL1-A', 'FL2-A', 'FL3-A'], spillover_matrix)
#     w.add_sample('sample_1', '/path/to/sample_1.fcs', channels=[...],
#                  gates=[{'kind': 'polygon', 'id': 'g1', ...}, ...])
#     w.write('out.wsp')


_WSP_NS = {
    'gating':     'http://www.isac-net.org/std/Gating-ML/v2.0/gating',
    'transforms': 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations',
    'data-type':  'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes',
    'xsi':        'http://www.w3.org/2001/XMLSchema-instance',
}


def _q(prefix, local):
    """Build a Clark-notation tag/attr name (`{namespace}localname`) for ET."""
    return f'{{{_WSP_NS[prefix]}}}{local}'


def _wsp_dataset_uri(path):
    """`<DataSet uri>` in the form FlowJo writes: 'file:/' + the absolute
    path with forward slashes, percent-encoded except '/', ':' and ','
    (Windows: 'file:/C:/a%20b,c/x.fcs'; POSIX: 'file:/data/a%20b/x.fcs')."""
    from urllib.parse import quote
    p = os.path.abspath(path).replace('\\', '/')
    if not p.startswith('/'):
        p = '/' + p
    return 'file:' + quote(p, safe='/:,')


class WspWriter:
    """Build a FlowJo-compatible .wsp workspace from in-memory gate dicts.

    Round-trips through WspReader: every gate kind we author (threshold,
    interval, rect, polygon) survives extract → write → extract.

    Out of scope: boolean gates only (ellipsoid DOES round-trip).
    A 'flowjo_ellipse' placeholder (a FlowJo-drawn ellipse OpenFlo could not
    convert) is written back as the EllipsoidGate FlowJo drew, with the axes
    it was drawn on, so FlowJo counts it as it did when drawn.
    An editor quadrant (4 rects sharing a ``quad_set``) is written as 4
    sibling RectangleGate populations — the form FlowJo itself writes — so
    a child stays under its own sector. Re-import gives 4 plain rects.
    Event counts: each <SampleNode> carries the FCS file's event total
    ($TOT, as FlowJo writes it; `events` when the file gives none) and each
    population the count the caller passes in `counts` (see `add_sample`),
    left out when the caller counted only part of the file.
    A count that is not known, or a population that cannot be computed, gets
    no ``count`` attribute: never a 0 that was not measured. FlowJo
    recomputes them all on opening.
    """

    def __init__(self, *, cytometer='Generic',
                 flowjo_version=WSP_FLOWJO_VERSION):
        self.cytometer       = cytometer
        self.flowjo_version  = flowjo_version
        self.samples         = []   # list of dicts (see add_sample)
        self.matrix          = None # (name, channels, np.ndarray)
        # Filled by write(): where FlowJo will not count what OpenFlo counts.
        self.warnings        = []
        # Filled by add_sample: gates on a channel of unknown scale.
        self.scale_notes     = []
        # Fix ids / labels the stamp comment names (export_stamp).
        self.corrections: list[str] | None = None

    @staticmethod
    def _check_matrix(channels, matrix):
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] != len(channels):
            raise ValueError(
                f"matrix shape {m.shape} doesn't match {len(channels)} channels")
        if not np.all(np.isfinite(m)):
            # FlowJo cannot apply a NaN/inf coefficient; never write one.
            raise ValueError("matrix has a cell that is not a finite number")
        return m

    def set_compensation(self, channels, matrix, name='Acquisition-defined'):
        """Register a workspace-level spillover matrix, listed in <Matrices>
        only. `matrix` is an NxN numpy array whose rows are source channels
        in `channels` order, columns destination channels in the same order.
        Diagonals are usually 1.0.

        It is NOT copied into the samples: a matrix inside a <Sample> is what
        FlowJo applies to that sample, as given, so a copy of one file's
        matrix would override every other file's own. A sample with no
        matrix of its own (see `add_sample`) gets FlowJo's
        "Acquisition-defined" matrix from its own FCS (SPILL)."""
        self.matrix = (name, list(channels),
                       self._check_matrix(channels, matrix))

    def add_sample(self, name, fcs_path, channels, gates, compensation=None,
                   timestep=None, transforms=None, unknown_fns=None,
                   group=None, counts=None, events=None):
        """Register one sample's gating tree. `gates` is a list of gate
        dicts in the shared schema (see `gate_to_mask`) with `id` and
        `parent_id` fields resolved within the list (use
        `read_template_gates` if you have a .wsp / template to convert).
        `compensation` is the `(channels, matrix)` OpenFlo applies to THIS
        sample; it is written inside the sample, where FlowJo applies it.

        `transforms` is ``{channel: method}`` for the channels whose gate
        coordinates are in a transformed space (the editor's logicle, say).
        Those coordinates are converted to the linear scale units a FlowJo
        workspace holds (see `map_gate_coords`); without it every coordinate
        is taken to be linear already.

        A channel of UNKNOWN scale in `transforms` has no transform to undo.
        `unknown_fns` (``{channel: f}``, see `rank_matched_fn`) converts it
        where the caller could match it to the FCS; a gate on one it cannot
        is written as stored. Either way it is listed in `self.scale_notes`
        for the user. A gate flagged ``scale_unknown`` was imported from
        FlowJo and is still in FlowJo's units on such a channel, so only its
        other channels are converted.

        `timestep` is seconds per Time tick, used only when the FCS gives
        none: FlowJo scales Time with the file's own value (see
        `_fcs_timestep`), so that wins, and a different `timestep` is logged.
        Time gates are written in seconds with it. The FCS TEXT goes into
        the sample's <Keywords>. `group` names a FlowJo group the sample is
        put in, besides 'All Samples' (OpenFlo's group of the sample).

        `counts` is ``{gate id: events in its population}`` as the caller
        counted them (the GUI: its Statistics count); None for a population
        that cannot be computed. `events` is the number of events the caller
        counted in, written on the <SampleNode> only when the FCS TEXT has no
        $TOT: FlowJo records the file's total there, and a re-import checks
        the file by it. When `events` differs from that total (events dropped
        by QC or downsampling), the counts are of a subset of the file and
        are left out, since FlowJo counts every event. What is not given is
        left without a ``count``."""
        comp = None
        if compensation is not None:
            c_chans, c_matrix = compensation
            comp = (list(c_chans), self._check_matrix(c_chans, c_matrix))
        keywords = None
        if fcs_path and os.path.isfile(fcs_path):
            try:
                keywords = _fcs_text_pairs(fcs_path)
            except (OSError, ValueError, IndexError) as exc:
                log.info(f"[WspWriter] {fcs_path}: FCS TEXT unreadable "
                         f"({exc}); sample written without <Keywords>")
        ts = _fcs_timestep(fcs_path, keywords)
        given = _positive_float(timestep) if timestep is not None else None
        if ts and given and not np.isclose(ts, given, rtol=1e-9):
            log.warning(f"[WspWriter] {name}: timestep {given:g} differs from "
                        f"the FCS's {ts:g}; FlowJo uses the file's, so it is used")
        transforms = transforms or {}
        unknown = {ch for ch, sp in transforms.items()
                   if is_unknown_spec(sp if isinstance(sp, dict)
                                      else {'method': sp})}
        unknown_fns = {ch: f for ch, f in (unknown_fns or {}).items()
                       if ch in unknown}
        to_linear = gate_scale_fns(
            {ch: sp for ch, sp in transforms.items() if ch not in unknown},
            'to_linear')
        gates = [self._gate_to_linear(name, g, to_linear, unknown_fns, unknown)
                 for g in gates]
        total = fcs_event_total(fcs_path) if keywords is not None else None
        if total is None:
            total = events
        counts = dict(counts or {})
        if counts and events is not None and total != events:
            log.info(f"[WspWriter] {name}: counted {events} of the file's "
                     f"{total} events; population counts left out")
            counts = {}
        self.samples.append({
            'name':         name,
            'fcs_path':     fcs_path or '',
            'channels':     list(channels),
            'gates':        list(gates),
            'compensation': comp,
            'keywords':     keywords,
            'timestep':     ts or given,
            'group':        str(group) if group else None,
            'counts':       counts,
            'events':       total,
        })

    def _gate_to_linear(self, sample, g, to_linear, unknown_fns, unknown):
        """`g` in FlowJo's linear units (see `add_sample`), noting in
        `scale_notes` a gate that rests on a channel of unknown scale."""
        from .gating import gate_channels
        # map_gate_coords also traces a polygon imported with FlowJo's edge
        # axes onto straight edges in data units, which is how the workspace
        # written here declares them (no <Transformations>), so it is called
        # even when no channel maps.
        if (g.get('scale_unknown')
                or g.get('kind') in ('boolean', 'flowjo_ellipse')):
            return map_gate_coords(g, to_linear)
        label = g.get('name') or g.get('id') or 'gate'
        for ch in sorted(set(gate_channels(g)) & unknown):
            how = ('converted by matching its values to the FCS rank for rank'
                   if ch in unknown_fns else
                   'written as stored (no FCS to match it to), so FlowJo '
                   'will likely count another region')
            note = f"{sample} / {label}: {ch} has an unknown scale; {how}."
            log.warning("[WspWriter] %s", note)
            self.scale_notes.append(note)
        fns = dict(to_linear, **unknown_fns)
        return map_gate_coords(g, fns)

    def write(self, out_path):
        """Render the workspace and write it to `out_path`."""
        xml_str = self._build_xml()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    # ── XML construction (private) ────────────────────────────────────────

    def _build_xml(self):
        self.warnings = []
        for prefix, uri in _WSP_NS.items():
            ET.register_namespace(prefix, uri)

        from .export_stamp import version as openflo_version
        root = ET.Element('Workspace', {
            'version':       '20.0',
            'modDate':       _now_for_wsp(),
            'flowJoVersion': self.flowjo_version,
            WSP_OPENFLO_ATTR: openflo_version(),
            _q('xsi', 'schemaLocation'): (
                f"{_WSP_NS['gating']} {_WSP_NS['gating']}/Gating-ML.v2.0.xsd "
                f"{_WSP_NS['transforms']} "
                f"{_WSP_NS['transforms']}/Transformations.v2.0.xsd "
                f"{_WSP_NS['data-type']} "
                f"{_WSP_NS['data-type']}/DataTypes.v2.0.xsd"),
        })

        # ── Matrices ──────────────────────────────────────────────────────
        # FlowJo lists each matrix here AND puts a copy (same transforms:id)
        # inside every <Sample> it compensates, right after <DataSet>.
        plan, sample_matrix = self._matrix_plan()
        matrices = ET.SubElement(root, 'Matrices')
        for entry in plan:
            self._emit_matrix(matrices, *entry)

        # ── Cytometers (minimal — FlowJo wants the element present) ───────
        cyts = ET.SubElement(root, 'Cytometers')
        ET.SubElement(cyts, 'Cytometer', {
            'name':          self.cytometer,
            'cyt':           self.cytometer,
            'transformType': 'BIEX',
            'manufacturer':  '',
            'serialnumber':  '',
        })

        # ── Groups (one default group, sample refs only) ──────────────────
        groups = ET.SubElement(root, 'Groups')
        grp_node = ET.SubElement(groups, 'GroupNode', {
            'name':         'All Samples',
            'owningGroup':  'All Samples',
            'expanded':     '1',
            'sortPriority': '10',
        })
        grp = ET.SubElement(grp_node, 'Group', {'name': 'All Samples'})
        refs = ET.SubElement(grp, 'SampleRefs')
        for i in range(1, len(self.samples) + 1):
            ET.SubElement(refs, 'SampleRef', {'sampleID': str(i)})
        # One more group per OpenFlo group, so FlowJo (and OpenFlo reading
        # the workspace back) keeps the samples grouped.
        named = {}
        for i, s in enumerate(self.samples, 1):
            if s.get('group') and s['group'].lower() not in \
                    WspReader._DEFAULT_GROUPS:
                named.setdefault(s['group'], []).append(i)
        for gname, ids in named.items():
            node = ET.SubElement(groups, 'GroupNode', {
                'name': gname, 'owningGroup': gname, 'expanded': '1',
                'sortPriority': '10'})
            g_el = ET.SubElement(node, 'Group', {'name': gname})
            g_refs = ET.SubElement(g_el, 'SampleRefs')
            for i in ids:
                ET.SubElement(g_refs, 'SampleRef', {'sampleID': str(i)})

        # ── SampleList ────────────────────────────────────────────────────
        sample_list = ET.SubElement(root, 'SampleList')
        for i, s in enumerate(self.samples, 1):
            sample = ET.SubElement(sample_list, 'Sample')
            if s['fcs_path']:
                ET.SubElement(sample, 'DataSet', {
                    'uri':      _wsp_dataset_uri(s['fcs_path']),
                    'sampleID': str(i),
                })
            own = sample_matrix[i - 1]
            if own is not None:
                self._emit_matrix(sample, *own)
            gates = [self._flowjo_form(g, s) for g in s['gates']]
            self._emit_axes(sample, s, gates)
            self._emit_keywords(sample, s)
            sn_attrs = {
                'name':         s['name'],
                'sampleID':     str(i),
                'owningGroup':  '',
                'expanded':     '1',
                'sortPriority': '10',
                **_count_attr(s.get('events')),
            }
            sample_node = ET.SubElement(sample, 'SampleNode', sn_attrs)
            self._emit_gate_tree(sample_node, gates, s.get('counts') or {})

        # Pretty-print (Python 3.9+).
        if hasattr(ET, 'indent'):
            ET.indent(root, space='  ')
        body = ET.tostring(root, encoding='unicode')
        # Which OpenFlo wrote it, and with which corrections, as a comment
        # too (the root's WSP_OPENFLO_ATTR names the version alone).
        from .export_stamp import xml_comment
        return ('<?xml version="1.0" encoding="UTF-8"?>\n'
                + xml_comment(self.corrections) + '\n' + body)

    def _flowjo_form(self, g, s):
        """Gate `g` of sample `s` as FlowJo must receive it, recording in
        `self.warnings` every place FlowJo will count differently:

        - an ellipse becomes a polygon: FlowJo 10.10.2 silently drops the
          whole sample for a Gating-ML (mean/covariance) EllipsoidGate;
        - Time coordinates become seconds (see `scale_time_coords`);
        - a channel this sample's own matrix compensates is named
          'Comp-<channel>': OpenFlo gated the compensated values, and in
          FlowJo the bare name is the raw parameter;
        - a lower edge FlowJo opens (or may open) is reported, not changed."""
        name = g.get('label') or g.get('name') or g.get('id')
        if g.get('kind') == 'ellipsoid':
            base = {k: g[k] for k in ('id', 'parent_id', 'name', 'label', '_import_id')
                    if k in g}
            verts = _ellipse_polygon(g['mean'], g['cov'], g.get('distance_sq', 4.0))
            if verts is not None:
                g = dict(base, kind='polygon', x_channel=g['x_channel'],
                         y_channel=g['y_channel'], vertices=verts)
                self._warn(f"{s['name']}: ellipse '{name}' exported as a "
                           f"{len(verts)}-vertex polygon (FlowJo drops a sample "
                           "with an ellipse gate); FlowJo decides polygon "
                           "membership on a display grid, so counts near the "
                           "edge can differ")
            else:
                # No ellipse, so gate_to_mask admits no events through it
                # (fail-closed). No gate shape is known that FlowJo counts as
                # empty, so it still goes out as the all-pass rectangle, and
                # the warning says the counts will differ.
                try:
                    np.linalg.inv(np.asarray(g['cov'], dtype=float))
                    why = "a covariance that is not positive definite"
                except np.linalg.LinAlgError:
                    why = "a singular covariance"
                g = dict(base, kind='rect', x_channel=g['x_channel'],
                         y_channel=g['y_channel'], x0=-1e6, x1=1e6, y0=-1e6, y1=1e6)
                self._warn(f"{s['name']}: ellipse '{name}' has {why}, so OpenFlo "
                           "admits no events through it; exported as an "
                           "all-pass rectangle, so FlowJo will differ and count "
                           "every event")
        elif g.get('kind') == 'flowjo_ellipse':
            self._warn(f"{s['name']}: FlowJo ellipse '{name}' written back as "
                       "FlowJo drew it, on the axes it was drawn on; OpenFlo "
                       "could not convert it and counts it, and every "
                       "population under it, as empty, but FlowJo will count "
                       "them")
        if s['timestep'] and _touches_time(g):
            self._check_time_edge(g, s, name)
        g = scale_time_coords(g, s['timestep'])
        if not s['timestep'] and _touches_time(g):
            self._warn(f"{s['name']}: gate '{name}' is on Time but the FCS gives "
                       "no $TIMESTEP or $BTIM/$ETIM; exported in ticks, and "
                       "FlowJo's Time scale for it is unknown")
        for ch, edge, certain in _edges_flowjo_opens(g, self._display(s)):
            self._warn(f"{s['name']}: gate '{name}' lower edge {edge:g} on {ch} "
                       + ("is at the bottom of FlowJo's axis: FlowJo treats it "
                          "as open and also counts every event below it" if certain
                          else "is near the bottom of FlowJo's axis, where FlowJo "
                          "may treat it as open (not measured)"))
        comp = set(s['compensation'][0]) if s.get('compensation') else set()
        named = {k: 'Comp-' + g[k] for k in ('channel', 'x_channel', 'y_channel')
                 if g.get(k) in comp}
        return dict(g, **named) if named else g

    def _check_time_edge(self, g, s, name):
        """Warn when a Time gate's upper edge looks converted to seconds
        before it reached the writer, which keeps Time in ticks and converts
        again (so FlowJo gets it `timestep` times too small). The mark: the
        edge (in ticks) ends the gate within the first `timestep` fraction
        of the run (the first 1% at the usual 0.01 s per tick), exactly where
        the run's length in seconds lands. A deliberate Time cut trims the
        start or the end of a run and keeps most of it, so it never lands
        there."""
        hi = _time_upper_edge(g)
        ts = float(s['timestep'])
        if hi is None or not (0 < ts < 0.5):
            return
        times = self._time_ticks(s)
        if times is None or not times.size:
            return
        last = float(np.nanmax(times))
        if not (0 <= hi < last and hi <= last * ts * 1.05):
            return
        cut = int(np.count_nonzero(times >= hi))
        self._warn(f"{s['name']}: Time gate '{name}' ends at {hi:g} ticks, "
                   f"where the run's last event is at {last:g}, so it cuts "
                   f"{cut:,} of {times.size:,} events "
                   f"({100.0 * cut / times.size:.1f}%). The run is "
                   f"{last * ts:g} s long, "
                   "so the edge was probably given in seconds: OpenFlo keeps "
                   "Time in ticks and multiplies by the timestep "
                   f"({ts:g} s) for FlowJo, which then gets {hi * ts:g} s "
                   f"where {hi:g} s was meant. Check this gate before "
                   "trusting FlowJo's counts.")

    @staticmethod
    def _time_ticks(s):
        """The Time channel of sample `s`'s FCS (raw ticks), read once and
        kept on the sample; None without a readable FCS or Time channel."""
        if '_time_ticks' in s:
            return s['_time_ticks']
        out = None
        path = s.get('fcs_path')
        if path and os.path.isfile(path):
            try:
                fd = _fcs_read.FlowData(path)
                names = [fd.channels[k]['pnn'].lower()
                         for k in sorted(fd.channels, key=int)]
                if 'time' in names and fd.events is not None:
                    ev = np.asarray(fd.events, dtype=float).reshape(
                        -1, fd.channel_count)
                    out = ev[:, names.index('time')]
            except Exception as exc:                  # noqa: BLE001 - logged; Time ticks are optional, the export goes on without them
                log.info(f"[WspWriter] {path}: Time not read ({exc})")
        s['_time_ticks'] = out
        return out

    def _warn(self, msg):
        self.warnings.append(msg)
        log.warning(f"[WspWriter] {msg}")

    @staticmethod
    def _polygon_channels(s):
        """The channels (bare names) of sample `s`'s polygon gates, ellipses
        included (they go out as polygons): their axes are written linear,
        see `_emit_axes`."""
        out = set()
        for g in s['gates']:
            if g.get('kind') in ('polygon', 'ellipsoid'):
                # Exact 'Time', as the axis rule in _emit_axes.
                out.update(c for c in (g.get('x_channel'), g.get('y_channel'))
                           if isinstance(c, str) and c.lower() != 'time')
        return out

    def _display(self, s):
        """{channel: 'LIN'/'LOG'}: the FCS's P<n>DISPLAY (see
        `_flowjo_axis_is_linear`), with the polygon channels `_emit_axes`
        writes linear marked 'LIN', so the warnings follow the axes FlowJo
        will be given."""
        out = _fcs_display(s.get('keywords'))
        ellipse = {c for g in s['gates'] if g.get('kind') == 'flowjo_ellipse'
                   for c, ax in zip((g.get('x_channel'), g.get('y_channel')),
                                    (g.get('flowjo') or {}).get('axes') or (),
                                    strict=False) if ax}
        out.update(dict.fromkeys(self._polygon_channels(s) - ellipse, 'LIN'))
        return out

    @staticmethod
    def _max_range(s, ch):
        """$PnR of channel `ch` (bare name) in sample `s`'s FCS TEXT, the
        top of FlowJo's linear axis for it; 262144 when not known."""
        kw = {str(k).upper().lstrip('$'): v for k, v in (s.get('keywords') or [])}
        for k, v in kw.items():
            m = re.fullmatch(r'P(\d+)N', k)
            if m and str(v).strip() == ch:
                r = _positive_float(kw.get(f'P{m.group(1)}R'))
                if r:
                    return r
        return 262144.0

    def _emit_axes(self, sample, s, gates):
        """<Transformations>, which FlowJo writes in every sample (other
        readers, FlowKit among them, require it). `gates` are in FlowJo form
        (`_flowjo_form`), so a parameter is named as its gate names it
        ('Comp-' where this sample is compensated). Declared:

        - the axes a 'flowjo_ellipse' is drawn on: its points are fractions
          of those axes, so FlowJo must get the axes it was drawn on, as
          FlowJo writes them (every attribute in the transforms namespace but
          linear's `gain`);
        - a linear axis (0 to the FCS's $PnR, gain 1) for each other gated
          parameter FlowJo shows linear anyway (scatter, a P<n>DISPLAY of
          LIN) and for each polygon's parameters. A polygon goes out with
          edges straight in data units (`_gate_to_linear` traces one drawn on
          another scale), which is what FlowJo does on a linear axis; on its
          default biex axis FlowJo would bend them. OpenFlo reads a linear or
          undeclared axis alike, so the export reads back as written.

        Time is left to FlowJo, which puts it in seconds from the FCS; other
        parameters keep FlowJo's defaults."""
        axes = {}
        for g in gates:
            if g.get('kind') != 'flowjo_ellipse':
                continue
            fj = g.get('flowjo') or {}
            for ch, ax in zip((g.get('x_channel'), g.get('y_channel')),
                              fj.get('axes') or (), strict=False):
                if not ch or not ax:
                    continue      # drawn on FlowJo's default axis: keep it
                if ch in axes and axes[ch] != ax:
                    self._warn(f"{s['name']}: FlowJo ellipses on {ch} were drawn on "
                               "different axes; the first one's axis is written, "
                               "so FlowJo will place the others differently")
                    continue
                axes[ch] = ax
        display = self._display(s)
        for g in gates:
            if g.get('kind') in ('boolean', 'flowjo_ellipse'):
                continue
            for ch in (g.get('channel'), g.get('x_channel'), g.get('y_channel')):
                # Exact 'Time': FlowJo's Time axis was measured only on a
                # channel named Time (see _flowjo_axis_is_linear).
                if (not isinstance(ch, str) or ch in axes
                        or ch.lower() == 'time'
                        or not _flowjo_axis_is_linear(ch, display)):
                    continue
                bare = ch[5:] if ch.startswith('Comp-') else ch
                axes[ch] = {'kind': 'linear', 'attrs': {
                    'minRange': '0', 'maxRange': f'{self._max_range(s, bare):g}',
                    'gain': '1'}}
        block = ET.SubElement(sample, 'Transformations')
        for ch, ax in axes.items():
            t = ET.SubElement(block, _q('transforms', str(ax['kind'])), {
                (k if k == 'gain' else _q('transforms', k)): str(v)
                for k, v in (ax.get('attrs') or {}).items()})
            ET.SubElement(t, _q('data-type', 'parameter'), {
                _q('data-type', 'name'): ch})

    @staticmethod
    def _emit_keywords(sample, s):
        """<Keywords>: FlowJo's FJ_FCS_VERSION, then the FCS TEXT in file
        order with the original key case. FlowJo 10.10.2 computes no
        population of a sample without them (measured: it keeps the
        sample and shows "parameters missing"). With no readable FCS, only a
        known $TIMESTEP is recorded, so OpenFlo's reader can convert Time
        back; it is never added next to an FCS TEXT that lacks it (FlowJo
        reads $TIMESTEP from the file, not from here)."""
        if s.get('keywords'):
            kel = ET.SubElement(sample, 'Keywords')
            ET.SubElement(kel, 'Keyword', {'name': 'FJ_FCS_VERSION', 'value': '3'})
            for k, v in s['keywords']:
                ET.SubElement(kel, 'Keyword', {
                    'name': _xml_safe(k), 'value': _xml_safe(v)})
        elif s.get('timestep'):
            kel = ET.SubElement(sample, 'Keywords')
            ET.SubElement(kel, 'Keyword', {
                'name': '$TIMESTEP', 'value': repr(s['timestep'])})

    def _matrix_plan(self):
        """Return `(plan, per_sample)`. `plan` lists one
        `(name, id, channels, matrix)` per distinct matrix for <Matrices>:
        the `set_compensation` matrix first, then each sample's own.
        `per_sample[k]` is sample k's OWN entry, or None -- never the
        workspace matrix (see `set_compensation`)."""
        plan = []

        def entry(name, channels, matrix):
            for e in plan:
                if e[2] == channels and np.array_equal(e[3], matrix):
                    return e
            if any(e[0] == name for e in plan):
                name = f'{name}-{len(plan) + 1}'
            e = (name, str(uuid.uuid4()), channels, matrix)
            plan.append(e)
            return e

        if self.matrix is not None:
            entry(*self.matrix)
        base = self.matrix[0] if self.matrix is not None else 'Acquisition-defined'
        per_sample = [entry(base, *s['compensation'])
                      if s.get('compensation') is not None else None
                      for s in self.samples]
        return plan, per_sample

    def _emit_matrix(self, parent, name, mid, channels, matrix):
        sm = ET.SubElement(parent, _q('transforms', 'spilloverMatrix'), {
            'spectral':               '0',
            'weightOptAlgorithmType': 'OLS',
            'prefix':                 'Comp-',
            'name':                   name,
            'editable':               '0',
            'color':                  '#c0c0c0',
            'version':                f'FlowJo-{self.flowjo_version}',
            'status':                 'FINALIZED',
            _q('transforms', 'id'):   mid,
            'suffix':                 '',
        })
        params = ET.SubElement(sm, _q('data-type', 'parameters'))
        for ch in channels:
            ET.SubElement(params, _q('data-type', 'parameter'), {
                _q('data-type', 'name'):  ch,
                'userProvidedCompInfix':  f'Comp-{ch}',
            })
        for i, src in enumerate(channels):
            sp = ET.SubElement(sm, _q('transforms', 'spillover'), {
                _q('data-type', 'parameter'): src,
                'userProvidedCompInfix':      f'Comp-{src}',
            })
            for j, dst in enumerate(channels):
                ET.SubElement(sp, _q('transforms', 'coefficient'), {
                    _q('data-type', 'parameter'): dst,
                    _q('transforms', 'value'):    repr(float(matrix[i, j])),
                })

    def _emit_gate_tree(self, sample_node, gates, counts=None):
        """Build <Subpopulations><Population><Gate>...</Gate><Subpopulations>...
        recursively from a flat list of gate dicts linked by parent_id.

        Quadrant rects (`quad_set`) are NOT folded into a QuadrantGate:
        FlowJo writes a quadrant as 4 RectangleGate populations, and folding
        moved a child of one sector onto the whole quadrant."""
        if not gates:
            return
        # Tolerate gates that carry only the reader's `_import_id` (a direct
        # WspReader→WspWriter round-trip, with no GUI id-assignment in between):
        # synthesise an `id` from it so the hierarchy still builds. parent_id in
        # reader output references `_import_id`, so this stays consistent.
        for _i, g in enumerate(gates):
            if 'id' not in g:
                g['id'] = g.get('_import_id') or f'_g{_i}'
        by_id    = {g['id']: g for g in gates}
        children = {}
        for g in gates:
            children.setdefault(g.get('parent_id'), []).append(g['id'])
        roots = children.get(None, [])
        if not roots:
            return
        sub = ET.SubElement(sample_node, 'Subpopulations')
        taken = set()
        # Population paths (names from the sample down) by gate id, and the
        # boolean nodes whose <Dependent> names wait for them: an operand may
        # be written after the boolean that names it.
        paths, pending = {}, []
        for gid in roots:
            self._emit_population(sub, gid, by_id, children, taken, (),
                                  paths, pending, counts or {})
        # Each operand by its path from the sample ('A/B/C'), as FlowJo 10
        # writes it. The '../C' form (relative to the boolean) is read back
        # by OpenFlo but not by other readers (FlowKit).
        for deps, operands, _own in pending:
            for op in operands:
                ET.SubElement(deps, 'Dependent', {'name': '/'.join(paths[op])})

    def _emit_population(self, parent, gid, by_id, children, taken,
                         parent_path=(), paths=None, pending=None,
                         counts=None):
        g = by_id[gid]
        paths = {} if paths is None else paths
        pending = [] if pending is None else pending
        # FlowJo ties a population across samples by its name, and needs it
        # unique among siblings: an unnamed gate is named for what it gates
        # on, not for the editor's per-sample id ('g2' here, 'g3' there).
        pop_name = unique_population_name(wsp_population_name(g), taken)
        own = tuple(parent_path) + (str(pop_name),)
        paths[gid] = own
        attrs = {
            'name':         str(pop_name),
            **_count_attr((counts or {}).get(gid)),
            'owningGroup':  '',
            'expanded':     '1',
            'sortPriority': '10',
        }
        if g.get('kind') == 'boolean':
            # FlowJo's boolean populations: AndNode / OrNode / NotNode, each
            # naming its operands as <Dependent> population paths.
            pop = ET.SubElement(parent, _WSP_BOOLEAN_TAGS[
                str(g.get('op', 'and')).lower()], attrs)
            deps = ET.SubElement(pop, 'Dependents')
            pending.append((deps, [op for op in g.get('operands') or []
                                   if op in by_id], own))
        else:
            # owningGroup="" = the sample's own gate. Naming "All Samples"
            # here would claim a gate the (empty) group node does not carry.
            pop = ET.SubElement(parent, 'Population', attrs)
            gate_wrap = ET.SubElement(pop, 'Gate', {
                _q('gating', 'id'): gid,
            })
            self._emit_gate_xml(gate_wrap, g)
        child_taken = set()
        for child in children.get(gid, []):
            sub = pop.find('Subpopulations')
            if sub is None:
                sub = ET.SubElement(pop, 'Subpopulations')
            self._emit_population(sub, child, by_id, children, child_taken,
                                  own, paths, pending, counts)

    def _emit_gate_xml(self, gate_wrap, g):
        kind = g.get('kind')
        if kind == 'threshold':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            dim  = ET.SubElement(rect, _q('gating', 'dimension'), {
                _q('gating', 'min'): repr(float(g['value'])),
            })
            ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                _q('data-type', 'name'): g['channel'],
            })
        elif kind == 'interval':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            dim  = ET.SubElement(rect, _q('gating', 'dimension'),
                                 _bound_attrs(float(g['lo']), float(g['hi'])))
            ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                _q('data-type', 'name'): g['channel'],
            })
        elif kind == 'rect':
            rect = ET.SubElement(gate_wrap, _q('gating', 'RectangleGate'))
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            for ch, lo, hi in (
                    (g['x_channel'], x0, x1),
                    (g['y_channel'], y0, y1)):
                dim = ET.SubElement(rect, _q('gating', 'dimension'),
                                    _bound_attrs(lo, hi))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
        elif kind == 'polygon':
            poly = ET.SubElement(gate_wrap, _q('gating', 'PolygonGate'))
            for ch in (g['x_channel'], g['y_channel']):
                dim = ET.SubElement(poly, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
            for vx, vy in g.get('vertices', []):
                vert = ET.SubElement(poly, _q('gating', 'vertex'))
                ET.SubElement(vert, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(vx)),
                })
                ET.SubElement(vert, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(vy)),
                })
        elif kind == 'ellipsoid':
            ell = ET.SubElement(gate_wrap, _q('gating', 'EllipsoidGate'))
            for ch in (g['x_channel'], g['y_channel']):
                dim = ET.SubElement(ell, _q('gating', 'dimension'))
                ET.SubElement(dim, _q('data-type', 'fcs-dimension'), {
                    _q('data-type', 'name'): ch,
                })
            mean = ET.SubElement(ell, _q('gating', 'mean'))
            for mv in g['mean']:
                ET.SubElement(mean, _q('gating', 'coordinate'), {
                    _q('data-type', 'value'): repr(float(mv)),
                })
            cov = ET.SubElement(ell, _q('gating', 'covarianceMatrix'))
            for row_vals in g['cov']:
                row = ET.SubElement(cov, _q('gating', 'row'))
                for entry_val in row_vals:
                    ET.SubElement(row, _q('gating', 'entry'), {
                        _q('data-type', 'value'): repr(float(entry_val)),
                    })
            ET.SubElement(ell, _q('gating', 'distanceSquare'), {
                _q('data-type', 'value'): repr(float(g.get('distance_sq', 4.0))),
            })
        elif kind == 'flowjo_ellipse':
            # The element as FlowJo wrote it (read namespace-stripped), with
            # its namespaces put back and its two dimensions named as this
            # export names them ('Comp-' where the sample is compensated).
            raw = ET.fromstring((g.get('flowjo') or {}).get('xml') or '<EllipsoidGate/>')
            _emit_flowjo_element(gate_wrap, raw, [g['x_channel'], g['y_channel']])
        else:
            log.info(f"[WspWriter] unknown gate kind {kind!r} — skipped")


def _count_attr(n):
    """``{'count': 'n'}`` for a known event count, else ``{}``: a population
    whose count is unknown (not given, or not computable) carries no count
    rather than a 0 nobody measured."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return {}
    if not np.isfinite(n) or n < 0 or n != int(n):
        return {}
    return {'count': str(int(n))}


def _emit_flowjo_element(parent, el, names):
    """Re-namespace `el`, a FlowJo gate element read namespace-stripped, under
    `parent` as FlowJo writes it: <fcs-dimension> and the `name`/`value` of
    <fcs-dimension>/<coordinate> in data-type, FlowJo's gating attributes
    (distance, min, max, id, parent_id) and every other tag in gating, the
    rest of the attributes plain. The <fcs-dimension>s take `names` in order."""
    tag = el.tag
    attrs = {}
    for k, v in el.attrib.items():
        if (tag, k) in (('fcs-dimension', 'name'), ('coordinate', 'value')):
            attrs[_q('data-type', k)] = v
        elif k in ('distance', 'min', 'max', 'id', 'parent_id'):
            attrs[_q('gating', k)] = v
        else:
            attrs[k] = v
    if tag == 'fcs-dimension' and names:
        attrs[_q('data-type', 'name')] = names.pop(0)
    new = ET.SubElement(parent, _q('data-type' if tag == 'fcs-dimension' else 'gating', tag),
                        attrs)
    if el.text and el.text.strip():
        new.text = el.text
    for child in el:
        _emit_flowjo_element(new, child, names)


# How FlowJo 10.10.2 treats the bottom of an axis. Measured by having FlowJo
# open, compute and save synthetic probe workspaces whose probe values each
# carried a distinct power-of-two number of events, so every saved count
# decodes to an exact set of events:
# - A rectangle's lower edge at or below a floor is open: FlowJo also counts
#   every event below the axis, however far, whatever the upper edge. On the
#   linear 0..262144 axis FlowJo gives a parameter when a workspace has no
#   <Transformations> (as OpenFlo exports) -- _flowjo_axis_is_linear says
#   which parameters get one -- an edge at 20.125 or below was opened and one
#   at 64 or above kept; in between is not measured. On its default biex
#   (width -100) an edge at -100 was kept and the floor lies in [-2000, -1000).
# - A polygon opens from higher up: its bottom at 1000 was opened on an axis
#   starting at 500, where a rectangle edge at 1000 was kept. FlowJo decides
#   polygon membership on a display grid (256 cells per axis by default,
#   1024 units on 0..262144); the polygon cases fit "bottom inside the first
#   cell", which is what the export warns about.
# Edges at or below _OPEN_BOUND are OpenFlo's own "open" sentinels.
_LINEAR_OPENED_UPTO = 20.125
_LINEAR_KEPT_FROM = 64.0
_POLYGON_LINEAR_OPENED_UPTO = 1.0     # bottoms at 1 and at -2000 were opened
_POLYGON_LINEAR_KEPT_FROM = 1024.0
# (_BIEX_OPENED_UPTO and _BIEX_KEPT_FROM, the biex floor, are imported from
# wsp_reader, which floors imported rectangles with them.)
_OPEN_BOUND = -1e6


def _bound_attrs(lo, hi):
    """gating:min/max for one dimension. A bound at OpenFlo's open sentinel
    (|v| >= 1e12) is left out -- FlowJo's own form for an open side, which it
    accepted for quadrant sectors; WspReader reads a missing bound back as
    the sentinel. Both are kept when both are open (a dimension needs one)."""
    lo_open, hi_open = lo <= -OPEN, hi >= OPEN
    if lo_open and hi_open:
        lo_open = hi_open = False
    a = {}
    if not lo_open:
        a[_q('gating', 'min')] = repr(lo)
    if not hi_open:
        a[_q('gating', 'max')] = repr(hi)
    return a


def _time_upper_edge(g):
    """The highest bounded coordinate gate `g` puts on a Time axis, or None
    (no Time axis, only a lower bound, or an open side)."""
    # Exact 'Time', as scale_time_coords: this edge is the one it converts.
    vals = []
    if isinstance(g.get('channel'), str) and g['channel'].lower() == 'time':
        if g.get('hi') is not None:
            vals.append(g['hi'])
    else:
        for axis, keys, i in (('x_channel', ('x0', 'x1'), 0),
                              ('y_channel', ('y0', 'y1'), 1)):
            ch = g.get(axis)
            if not (isinstance(ch, str) and ch.lower() == 'time'):
                continue
            vals.extend(g[k] for k in keys if g.get(k) is not None)
            vals.extend(v[i] for v in g.get('vertices') or ())
    vals = [float(v) for v in vals]
    vals = [v for v in vals if np.isfinite(v) and abs(v) < OPEN]
    return max(vals) if vals else None


def _touches_time(g):
    # Exact 'Time', as scale_time_coords: says whether it converts `g`.
    return any(isinstance(g.get(k), str) and g[k].lower() == 'time'
               for k in ('channel', 'x_channel', 'y_channel'))


def _flowjo_axis_is_linear(ch, display=None):
    """Whether FlowJo 10.10.2 shows `ch` on a linear axis in a workspace with
    no <Transformations> (as OpenFlo exports). It follows the FCS's
    P<n>DISPLAY keyword (BD FACSDiva writes LIN or LOG per parameter):
    measured, scatter marked LOG got a biex axis and scatter marked LIN a
    linear one. Without the keyword, scatter and Time are linear and the rest
    biex. `display` is {channel name: DISPLAY value} from the FCS TEXT;
    `ch` is a channel name (str)."""
    d = (display or {}).get(ch[5:] if ch.startswith('Comp-') else ch)
    # Exact 'Time': FlowJo 10.10.2 was measured only on a channel named Time.
    if ch.lower() != 'time' and d in ('LIN', 'LOG'):
        return d == 'LIN'
    return ch.lower() == 'time' or ch.upper().startswith(('FSC', 'SSC'))


def _fcs_display(keywords):
    """{channel name: 'LIN'/'LOG'} from FCS TEXT pairs ($PnN + PnDISPLAY)."""
    kw = {k.upper().lstrip('$'): v for k, v in (keywords or [])}
    out = {}
    for k, v in kw.items():
        m = re.fullmatch(r'P(\d+)N', k)
        if m:
            d = kw.get(f'P{m.group(1)}DISPLAY', '').strip().upper()
            if d in ('LIN', 'LOG'):
                out[v] = d
    return out


def _edges_flowjo_opens(g, display=None):
    """(channel, lower edge, certain) for each lower edge of gate `g` (in
    FlowJo units) that FlowJo opens (certain) or may open (not measured),
    while OpenFlo keeps it literal. See the thresholds above."""
    kind = g.get('kind')
    edges = []
    if kind == 'threshold':
        edges.append((g.get('channel'), g.get('value')))
    elif kind == 'interval':
        edges.append((g.get('channel'), g.get('lo')))
    elif kind == 'rect':
        edges.append((g.get('x_channel'), min(float(g['x0']), float(g['x1']))))
        edges.append((g.get('y_channel'), min(float(g['y0']), float(g['y1']))))
    elif kind == 'polygon' and g.get('vertices'):
        vs = np.asarray(g['vertices'], dtype=float)
        edges.append((g.get('x_channel'), float(vs[:, 0].min())))
        edges.append((g.get('y_channel'), float(vs[:, 1].min())))
    out = []
    for ch, v in edges:
        if v is None or not isinstance(ch, str):
            continue
        v = float(v)
        if v <= _OPEN_BOUND or (is_time_channel(ch) and v <= 0):
            continue                     # meant open / nothing lies below 0 s
        if _flowjo_axis_is_linear(ch, display):
            opened, kept = ((_POLYGON_LINEAR_OPENED_UPTO, _POLYGON_LINEAR_KEPT_FROM)
                            if kind == 'polygon' else (_LINEAR_OPENED_UPTO, _LINEAR_KEPT_FROM))
        elif kind == 'polygon':
            opened, kept = -np.inf, _BIEX_KEPT_FROM      # polygons on biex: not measured
        else:
            opened, kept = _BIEX_OPENED_UPTO, _BIEX_KEPT_FROM
        if v < kept:
            out.append((ch, v, v <= opened))
    return out


def _now_for_wsp():
    """FlowJo's modDate format, eg 'Fri Mar 20 14:22:19 EDT 2026'."""
    import time
    return time.strftime('%a %b %d %H:%M:%S %Z %Y')
