"""compare_workspace.py — Side-by-side review tool.

Reads a FlowJo .wsp workspace, locates each referenced FCS, re-applies
every Population gate (and every boolean AndNode / OrNode / NotNode) via
OpenFlo's evaluator, and compares the resulting per-population event counts
against the counts FlowJo wrote into the .wsp itself (the `count="…"`
attribute on each population).

The output is a CSV (one row per sample × population) plus a tiny HTML
summary; both name the population's path from the sample, FlowJo's count,
OpenFlo's count, and the delta (absolute + relative).

Usage:
    python compare_workspace.py path/to/workspace.wsp \\
        [--fcs-dir DIR]         # override the DataSet uri root
        [--csv  report.csv]     # default <wsp_basename>_compare.csv
        [--html report.html]    # default <wsp_basename>_compare.html
"""
import argparse
import csv
import html
import os
import re
import sys
from collections import defaultdict
from urllib.parse import unquote, urlparse

from ._console import force_utf8_streams
from .pipeline import (
    _WSP_BOOLEAN_OPS,
    FlowSample,
    WspParseError,
    WspReader,
    _scale_time_coords,
    _wsp_axis_floor_lo,
    _wsp_axis_floors,
    _wsp_comp_channels,
    _wsp_ellipse_gate,
    _wsp_polygon_edge_axes,
    cumulative_gate_mask,
    resolve_wsp_dependent,
    wsp_event_mismatch,
    wsp_sample_events,
    wsp_uncompensated_channels,
    wsp_written_by_openflo,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_fcs_uri(uri, fcs_dir=None):
    """Map a FlowJo DataSet uri (eg 'file:/path/to/sample.fcs' or plain path)
    onto a local path. If the original doesn't exist and `fcs_dir` is
    set, look for a basename match there."""
    if not uri:
        return None
    p = _uri_path(uri).replace('/', os.sep)
    if os.path.isfile(p):
        return p
    if fcs_dir:
        candidate = os.path.join(fcs_dir, os.path.basename(p))
        if os.path.isfile(candidate):
            return candidate
    return None


class FcsLocator:
    """Find the FCS files a workspace names: its uri as written, then
    `fcs_dir`, then the workspace's own folder, then, by file name, any
    subfolder of those two (a workspace kept beside a folder per day or
    plate). Two files of that name in the subfolders are told apart by how
    many of the uri's folder names they share; when that cannot decide, the
    sample is not located and `ambiguous` names it, never a guess."""

    MAX_DEPTH = 6

    def __init__(self, wsp_path=None, fcs_dir=None):
        self.fcs_dir = fcs_dir if fcs_dir and os.path.isdir(fcs_dir) else None
        self.wsp_dir = (os.path.dirname(os.path.abspath(wsp_path))
                        if wsp_path else None)
        self._index = None
        self.ambiguous = []        # (file name, number of candidates)

    def _roots(self):
        out = []
        for r in (self.fcs_dir, self.wsp_dir):
            if r and os.path.normcase(os.path.abspath(r)) not in {
                    os.path.normcase(os.path.abspath(o)) for o in out}:
                out.append(r)
        return out

    def _build_index(self):
        index = {}
        seen = set()
        for root in self._roots():
            base_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep)
            for dirpath, dirnames, filenames in os.walk(root):
                depth = os.path.abspath(dirpath).count(os.sep) - base_depth
                dirnames[:] = ([] if depth >= self.MAX_DEPTH else
                               [d for d in dirnames if not d.startswith('.')])
                for f in filenames:
                    if not f.lower().endswith('.fcs'):
                        continue
                    full = os.path.join(dirpath, f)
                    key = os.path.normcase(os.path.abspath(full))
                    if key not in seen:
                        seen.add(key)
                        index.setdefault(f.lower(), []).append(full)
        return index

    def find(self, uri):
        fcs = _resolve_fcs_uri(uri, self.fcs_dir)
        if fcs is not None or not uri:
            return fcs
        raw = _uri_path(uri)
        name = os.path.basename(raw)
        if self.wsp_dir:
            cand = os.path.join(self.wsp_dir, name)
            if os.path.isfile(cand):
                return cand
        if self._index is None:
            self._index = self._build_index()
        hits = self._index.get(name.lower(), [])
        if len(hits) == 1:
            return hits[0]
        if not hits:
            return None
        want = [p.lower() for p in re.split(r'[\\/]', raw)[:-1] if p][::-1]

        def shared(path):
            have = [p.lower() for p in
                    os.path.dirname(os.path.abspath(path)).split(os.sep)][::-1]
            n = 0
            for a, b in zip(want, have, strict=False):
                if a != b:
                    break
                n += 1
            return n
        scores = sorted(((shared(h), h) for h in hits), reverse=True)
        if scores[0][0] > scores[1][0]:
            return scores[0][1]
        self.ambiguous.append((name, len(hits)))
        return None


def _uri_path(uri):
    """The local path a DataSet uri names, before any lookup."""
    if uri.startswith('file:'):
        p = unquote(urlparse(uri).path or uri[len('file:'):])
        # urlparse('file:G:/...') puts the drive letter into netloc; recover.
        if not p:
            p = unquote(uri[len('file:'):])
    else:
        p = uri
    # FlowJo on Windows often emits leading slashes like '/C:/...'
    if p.startswith('/') and len(p) > 2 and p[2] == ':':
        p = p[1:]
    return p


# Per-sample raw inventory: every Population with its declared FlowJo
# count, parent linkage, and the underlying gate dict (in our schema).
def _per_sample_inventory(reader, fcs_dir=None, locate=None):
    """Walk the .wsp again — this time keeping track of (sample_name,
    fcs_uri, [(pop_name, flowjo_count, gate_dict, parent_pop_name), ...])
    Returns a list of (sample_name, fcs_uri, pops_list). `fcs_dir` is where
    to look for an FCS whose uri does not resolve (its Time scale is read
    from the file, as FlowJo does)."""
    samples = []
    if reader.root is None:
        return samples
    for sample_elem in reader.root.iter('Sample'):
        ds = sample_elem.find('DataSet')
        uri = ds.get('uri') if ds is not None else None
        sn = sample_elem.find('SampleNode')
        if sn is None:
            continue
        sample_name = sn.get('name') or 'unnamed'
        # An OpenFlo export's edges are literal (see _wsp_axis_floors).
        floors = ({} if wsp_written_by_openflo(reader.root)
                  else _wsp_axis_floors(sample_elem))
        pops = []          # (uid, pop_name, count, gate_dict, parent_uid)
        uid_ctr = [0]      # per-occurrence id so same-named pops don't collide
        known = {}         # population path (names from the sample) -> uid
        booleans = []      # (gate dict, own path, <Dependent> names)

        def count_of(elem):
            try:
                count_attr = elem.get('count')
                return int(count_attr) if count_attr is not None else None
            except ValueError:
                return None

        def walk(pop_elem, parent_uid, path):
            uid_ctr[0] += 1                 # noqa: B023 (walk runs same-iteration)
            uid = uid_ctr[0]                # noqa: B023
            pop_name = pop_elem.get('name') or 'unnamed'
            own = tuple(path) + (pop_elem.get('name') or '',)
            count = count_of(pop_elem)
            # Pull this Population's gate via the same parser the reader uses.
            g = _gate_from_population(pop_elem, reader, floors,  # noqa: B023
                                      sample_elem)  # noqa: B023
            tags = sorted({c.tag for gw in pop_elem.findall('Gate') for c in gw})
            if g is None and tags:
                # A gate compare cannot read (a QuadrantGate, a type it does
                # not know): its row, and every row under it, says so. Left
                # out, each child became a root and was counted against
                # every event the unread gate excluded, with no error.
                g = {'kind': _UNREAD, 'why': '/'.join(tags)}
            if g is not None:
                g = dict(g, _wsp_path=own)
                # B023: `pops` is loop-local but `walk` is consumed within
                # the same iteration — safe in practice.
                pops.append((uid, pop_name, count, g, parent_uid))  # noqa: B023
            # A population with no gate at all selects its parent's events:
            # its children are its parent's.
            attach = uid if g is not None else parent_uid
            known[own] = attach                                 # noqa: B023
            walk_children(pop_elem, attach, own)

        def walk_boolean(node, parent_uid, path):
            """A FlowJo AndNode / OrNode / NotNode, as WspReader reads it: a
            boolean over the populations its <Dependent> elements name,
            resolved once the whole sample is walked. These, and every
            population under them, used to be left out of the report."""
            uid_ctr[0] += 1                                     # noqa: B023
            uid = uid_ctr[0]                                    # noqa: B023
            own = tuple(path) + (node.get('name') or '',)
            g = {'kind': 'boolean', 'op': _WSP_BOOLEAN_OPS[node.tag],
                 'operands': [], 'wsp_comp_channels': [], '_wsp_path': own}
            pops.append((uid, node.get('name') or 'unnamed',   # noqa: B023
                         count_of(node), g, parent_uid))
            known[own] = uid                                    # noqa: B023
            booleans.append((g, uid, own,                       # noqa: B023
                             [d.get('name') for dn in node.findall('Dependents')
                              for d in dn.findall('Dependent')]))
            walk_children(node, uid, own)

        def walk_children(elem, parent_uid, path):
            for sub in elem.findall('Subpopulations'):
                for child in sub:
                    if child.tag == 'Population':
                        walk(child, parent_uid, path)
                    elif child.tag in _WSP_BOOLEAN_OPS:
                        walk_boolean(child, parent_uid, path)

        walk_children(sn, None, ())
        for g, uid, own, deps in booleans:
            missing = []
            for dep in deps:
                hit = resolve_wsp_dependent(dep, own, known)
                target = known.get(hit) if hit is not None else None
                if target is None or target == uid:
                    missing.append(str(dep))
                else:
                    g['operands'].append(target)
            if missing or not deps:
                why = (f"boolean operand {', '.join(map(repr, missing))} not found"
                       if missing else 'boolean names no operand')
                g.clear()
                g.update({'kind': _UNREAD, 'why': why, '_wsp_path': own})
        # FlowJo's Time gates are in seconds; FlowSample's Time is in ticks.
        ts = WspReader.sample_timestep(
            sample_elem, (locate or (lambda u: _resolve_fcs_uri(u, fcs_dir)))(uri))
        if ts:
            pops[:] = [(u, n, c, g if g.get('kind') == _UNREAD
                        else _scale_time_coords(g, 1.0 / ts), p)
                       for u, n, c, g, p in pops]
        if pops:
            samples.append(_SampleEntry((sample_name, uri, pops),
                                        wsp_sample_events(sample_elem)))
    return samples


class _SampleEntry(tuple):
    """(sample_name, fcs_uri, pops), carrying as `events` the event count
    the workspace records for the sample (None when it records none)."""

    events: int | None

    def __new__(cls, fields, events=None):
        self = super().__new__(cls, fields)
        self.events = events
        return self

    def __reduce__(self):
        return (_SampleEntry, (tuple(self), self.events))


# Marks a population whose gate compare cannot read (see _per_sample_inventory):
# never evaluated, reported as an error row with everything under it.
_UNREAD = '_unread'


def _gate_from_population(pop_elem, reader, floors=None, sample_elem=None):
    """Extract this Population's own gate using WspReader's existing
    rect/polygon parsers (avoids duplicating logic). Returns a gate dict
    or None if the gate kind is unsupported. `floors` are the sample's
    axis floors (see pipeline._wsp_axis_floor_lo); `sample_elem` is the
    <Sample> whose axes a FlowJo-drawn ellipse is converted with.

    An ellipse goes through the reader's own parser (_wsp_ellipse_gate): one
    it cannot convert is a 'flowjo_ellipse' that counts no events, so its
    children stay under it. Unparsed, it made each child a root, counted
    against every event the ellipse excluded.

    Like WspReader's, the gate carries 'wsp_comp_channels': the channels
    FlowJo names compensated ('Comp-FL2-A'); a compensated channel named
    bare is evaluated on the uncompensated data (see _compare_one_sample)."""
    g = _parse_population_gate(pop_elem, reader, floors, sample_elem)
    if g is not None:
        gate_elem = next((c for gw in pop_elem.findall('Gate') for c in gw), None)
        g['wsp_comp_channels'] = (_wsp_comp_channels(gate_elem)
                                  if gate_elem is not None else [])
    return g


def _parse_population_gate(pop_elem, reader, floors=None, sample_elem=None):
    for gw in pop_elem.findall('Gate'):
        for child in gw:
            if child.tag == 'EllipsoidGate':
                body, _why = _wsp_ellipse_gate(child, sample_elem)
                if body is not None:
                    return body
                continue
            if child.tag == 'RectangleGate':
                parsed = []
                for dim in child.iter('dimension'):
                    ch = reader._channel_name(dim)
                    if not ch:
                        continue
                    mn = dim.get('min')
                    mx = dim.get('max')
                    try:
                        lo = float(mn) if mn is not None else None
                        hi = float(mx) if mx is not None else None
                    except (TypeError, ValueError):
                        continue
                    fd = next(dim.iter('fcs-dimension'), None)
                    lo = _wsp_axis_floor_lo(fd.get('name') if fd is not None else None,
                                            lo, floors or {})
                    parsed.append((ch, lo, hi))
                if len(parsed) == 1:
                    ch, lo, hi = parsed[0]
                    if lo is not None and hi is not None:
                        return {'kind': 'interval', 'channel': ch, 'lo': lo, 'hi': hi}
                    if lo is not None:
                        return {'kind': 'threshold', 'channel': ch, 'value': lo}
                    if hi is not None:   # max-only: open below, as WspReader reads it
                        return {'kind': 'interval', 'channel': ch, 'lo': -1e12, 'hi': hi}
                if len(parsed) == 2:
                    # A missing bound is an open side (FlowJo's quadrant
                    # sectors carry only their divider bound), as in WspReader.
                    (xc, x0, x1), (yc, y0, y1) = parsed
                    return {'kind': 'rect',
                            'x_channel': xc, 'y_channel': yc,
                            'x0': -1e12 if x0 is None else x0,
                            'x1': 1e12 if x1 is None else x1,
                            'y0': -1e12 if y0 is None else y0,
                            'y1': 1e12 if y1 is None else y1}
            elif child.tag == 'PolygonGate':
                dims = list(child.iter('dimension'))
                if len(dims) != 2:
                    continue
                xc = reader._channel_name(dims[0])
                yc = reader._channel_name(dims[1])
                if not xc or not yc:
                    continue
                verts = []
                for v in child.iter('vertex'):
                    coords = list(v.iter('coordinate'))
                    if len(coords) < 2:
                        continue
                    vx = coords[0].get('value')
                    vy = coords[1].get('value')
                    if vx is None or vy is None:
                        continue
                    try:
                        verts.append([float(vx), float(vy)])
                    except ValueError:
                        continue
                if len(verts) >= 3:
                    g = {'kind': 'polygon', 'x_channel': xc, 'y_channel': yc,
                         'vertices': verts}
                    # Edges straight on FlowJo's axes, as the reader's.
                    axes = _wsp_polygon_edge_axes(child, sample_elem)
                    if axes is not None:
                        g['flowjo_edge_axes'] = axes
                    return g
    return None


# ── Per-sample comparison ────────────────────────────────────────────────────

def _compare_one_sample(sample_name, fcs_path, pops, wsp_path=None,
                        expected_events=None):
    """Apply each population's cumulative gate to the FCS data, report
    the OpenFlo count alongside FlowJo's declared count, and yield rows
    suitable for the CSV/HTML report.

    If `wsp_path` is provided, the workspace's spillover matrix is
    applied to the data BEFORE gate evaluation — essential for any
    gate on a fluorescence channel, since FlowJo authors gate
    thresholds against compensated values. Without this step, CD901b+ /
    CD903+ / etc. gates compare against raw data and miss almost
    everything.

    `expected_events` is the event count the workspace records for the
    sample (_SampleEntry.events). When the FCS file holds another number,
    every row says so in its error: a workspace can name several files
    alike, and the one found may not be the one FlowJo gated."""
    s = FlowSample(fcs_path)
    comp_err = ''
    if wsp_path:
        try:
            s.compensate_from_wsp(wsp_path)
        except Exception as exc:
            # Gates on fluorescence channels compare against COMPENSATED values;
            # without comp the counts are meaningless. Flag every row (not just a
            # stdout print) so the FlowJo-vs-OpenFlo table can't read as valid.
            comp_err = f'uncompensated (compensation failed: {type(exc).__name__})'
            print(f"    [compensation skipped] {type(exc).__name__}: {exc}")
    total = len(s.data)
    mismatch = wsp_event_mismatch(expected_events, total)
    if mismatch:
        comp_err = '; '.join(e for e in (comp_err, mismatch) if e)
    # Key gates by the per-occurrence uid threaded from the walk, so populations
    # that SHARE a name (Q1-Q4, 'Single Cells', copied gates) don't overwrite
    # each other. Parent linkage uses the parent's uid directly (the walk
    # already passes a gateless population's children to its parent).
    gates_by_id = {uid: dict(gdict) for uid, _pn, _cnt, gdict, _puid in pops}
    for uid, _pn, _cnt, _gdict, parent_uid in pops:
        gates_by_id[uid]['parent_id'] = (
            parent_uid if parent_uid in gates_by_id else None)
    # FlowJo keeps a compensated sample's uncompensated parameters beside the
    # compensated ones: 'FL2-A' is the measured value, 'Comp-FL2-A' the
    # compensated one. WspReader strips the prefix, so both name one column;
    # a gate on the bare name of a channel compensated here reads that
    # channel's measured values instead (a gate at 2,500 counted 5,216 where
    # FlowJo counts 8,413). Nothing here filters events, so `raw` and `data`
    # are row for row the same events.
    data = s.data
    if s.is_compensated() and len(s.raw) == len(data):
        extra = {}
        for g in gates_by_id.values():
            if g.get('kind') == _UNREAD:
                continue
            bare = wsp_uncompensated_channels(g, s.comp_channels)
            for key in ('channel', 'x_channel', 'y_channel'):
                ch = g.get(key)
                if ch in bare:
                    alias = f'{ch} (uncompensated)'
                    extra[alias] = s.raw[ch].to_numpy(dtype=float)
                    g[key] = alias
        if extra:
            data = data.assign(**extra)

    rows = []
    # One cache for the whole sweep: this walks EVERY population in the
    # workspace against the same sample, so without it each one re-evaluates
    # its full ancestor chain — and a polygon gate costs ~250x a threshold
    # gate. `gates_by_id` and `s.data` are fixed here, which is the cache's
    # precondition.
    mask_cache = {}

    def unconverted(uid):
        """Why OpenFlo counts nothing under `uid`: the first 'flowjo_ellipse'
        on its chain, or ''."""
        seen = set()
        while uid in gates_by_id and uid not in seen:
            seen.add(uid)
            g = gates_by_id[uid]
            if g.get('kind') == 'flowjo_ellipse':
                why = (g.get('flowjo') or {}).get('why') or 'its axis'
                return f'FlowJo ellipse not converted ({why}); OpenFlo counts it as empty'
            uid = g.get('parent_id')
        return ''

    names = {uid: pn for uid, pn, _c, _g, _p in pops}

    def unread(uid):
        """Why `uid` cannot be counted: the gate on its chain, or on a
        boolean operand's chain, that compare could not read, or ''."""
        seen, first, todo = set(), uid, [uid]
        while todo:
            uid = todo.pop()
            if uid not in gates_by_id or uid in seen:
                continue
            seen.add(uid)
            g = gates_by_id[uid]
            if g.get('kind') == _UNREAD:
                what = f"gate not read ({g.get('why') or '?'})"
                return (what if uid == first else
                        f"depends on '{names.get(uid, '?')}', whose {what}")
            todo.append(g.get('parent_id'))
            if g.get('kind') == 'boolean':
                todo.extend(g.get('operands') or [])
        return ''

    for uid, pname, flowjo_count, gdict, _parent_uid in pops:
        path = '/'.join(gdict.get('_wsp_path') or (pname,))
        why_unread = unread(uid)
        if why_unread:
            rows.append({
                'sample': sample_name, 'population': pname, 'path': path,
                'flowjo_count': flowjo_count, 'openflo_count': None,
                'delta': None, 'rel_delta': None, 'total_events': total,
                'error': '; '.join(e for e in (comp_err, why_unread) if e)})
            continue
        try:
            mask = cumulative_gate_mask(gates_by_id, uid, data,
                                        cache=mask_cache)
            openflo_count = int(mask.sum())
        except Exception as exc:
            openflo_count = None
            err = f'{type(exc).__name__}: {exc}'
            if comp_err:
                err = f'{comp_err}; {err}'
        else:
            err = '; '.join(e for e in (comp_err, unconverted(uid)) if e)
        rows.append({
            'sample':        sample_name,
            'population':    pname,
            'path':          path,
            'flowjo_count':  flowjo_count,
            'openflo_count': openflo_count,
            'delta':         (openflo_count - flowjo_count
                              if openflo_count is not None
                              and flowjo_count is not None else None),
            'rel_delta':     (
                (openflo_count - flowjo_count) / flowjo_count
                if openflo_count is not None and flowjo_count not in (None, 0)
                else None),
            'total_events':  total,
            'error':         err,
        })
    return rows


# ── Report writers ───────────────────────────────────────────────────────────

# `path` is the population's names from the sample down ('Lymphs/Single
# Cells'): populations of one name under different parents are told apart.
CSV_FIELDS = ['sample', 'population', 'path', 'flowjo_count', 'openflo_count',
              'delta', 'rel_delta', 'total_events', 'error']

def _write_csv(rows, path):
    fields = CSV_FIELDS
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: '' if r.get(k) is None else r.get(k)
                        for k in fields})


def _write_html(rows, path, wsp_path):
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r['sample']].append(r)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>OpenFlo vs FlowJo — {html.escape(os.path.basename(wsp_path))}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:1em auto;color:#222}",
        "table{border-collapse:collapse;margin:0.5em 0 1.5em 0;width:100%;font-size:13px}",
        "th,td{padding:4px 8px;border-bottom:1px solid #ddd;text-align:right}",
        "th:first-child,td:first-child{text-align:left}",
        "tr.bad{background:#ffe9e6}", "tr.warn{background:#fff7d9}",
        "h2{margin-top:1.5em;font-size:1.1em;color:#444;font-weight:600}",
        ".tot{color:#777}.err{color:#b00;font-style:italic}",
        "</style></head><body>",
        "<h1>OpenFlo vs FlowJo</h1>",
        f"<p>Workspace: <code>{html.escape(wsp_path)}</code><br>"
        f"{len(rows)} populations across {len(by_sample)} sample(s).</p>",
    ]
    for sample, srows in by_sample.items():
        parts.append(f"<h2>{html.escape(sample)}</h2>")
        parts.append("<table><thead><tr>"
                     "<th>Population</th><th>Path</th>"
                     "<th>FlowJo</th><th>OpenFlo</th>"
                     "<th>Δ</th><th>Δ %</th><th>Total events</th><th>Error</th>"
                     "</tr></thead><tbody>")
        for r in srows:
            cls = ''
            rd  = r.get('rel_delta')
            if r.get('error'):
                cls = 'bad'
            elif rd is not None:
                ard = abs(rd)
                if ard > 0.05:
                    cls = 'bad'
                elif ard > 0.01:
                    cls = 'warn'
            row_open = f"<tr class='{cls}'>" if cls else "<tr>"
            fc = r['flowjo_count']
            oc = r['openflo_count']
            dl = r['delta']
            rl = r['rel_delta']
            fc_s = '' if fc is None else f'{fc:,}'
            oc_s = '' if oc is None else f'{oc:,}'
            dl_s = '' if dl is None else f'{dl:+,}'
            rl_s = '' if rl is None else f'{rl * 100:+.2f}%'
            parts.append(row_open + (
                f"<td>{html.escape(r['population'])}</td>"
                f"<td>{html.escape(r.get('path') or r['population'])}</td>"
                f"<td>{fc_s}</td>"
                f"<td>{oc_s}</td>"
                f"<td>{dl_s}</td>"
                f"<td>{rl_s}</td>"
                f"<td class='tot'>{r['total_events']:,}</td>"
                f"<td class='err'>{html.escape(r['error'])}</td>"
                f"</tr>"))
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))


# ── Running the samples ──────────────────────────────────────────────────────

def _compare_workers(asked, n_samples):
    """How many processes compare samples: ``asked`` when given (at least
    1), else up to 4 and never more than the cores; never more than the
    samples."""
    if asked and asked > 0:
        n = asked
    else:
        n = min(4, os.cpu_count() or 1)
    return max(1, min(n, n_samples))


def _compare_sample_at(wsp_path, fcs_dir, index):
    """Compare the workspace's ``index``-th sample, in a worker process:
    ``(lines to print, rows)``. The workspace is read again here, since its
    parsed XML does not cross process boundaries."""
    reader = WspReader(wsp_path)
    locator = FcsLocator(wsp_path, fcs_dir)
    samples = _per_sample_inventory(reader, fcs_dir, locate=locator.find)
    return _compare_entry(wsp_path, fcs_dir, samples[index], locator)


def _compare_entry(wsp_path, fcs_dir, entry, locator):
    """One sample's report lines and rows; a failure is a line, not a
    crash, so the other samples are still compared."""
    sample_name, uri, pops = entry
    fcs = locator.find(uri)
    if fcs is None:
        return (f"  [skip] {sample_name}: FCS not located "
                f"(uri={uri!r}, --fcs-dir={fcs_dir or ''!r})", [])
    line = (f"  {sample_name}: {os.path.basename(fcs)} "
            f"({len(pops)} population(s))")
    try:
        rows = _compare_one_sample(sample_name, fcs, pops, wsp_path=wsp_path,
                                   expected_events=getattr(entry, 'events', None))
    except Exception as exc:
        return (f"{line}\n    [error] {type(exc).__name__}: {exc}", [])
    return line, rows


def _compare_samples(wsp_path, fcs_dir, samples, locator, workers):
    """``(line, rows)`` for every sample, in workspace order. With more
    than one worker the samples run in a process pool (each is independent:
    its own FCS, compensation and gates), and the results are still yielded
    in order, so the CSV and HTML are the same as from one process."""
    if workers <= 1 or len(samples) <= 1:
        for entry in samples:
            yield _compare_entry(wsp_path, fcs_dir, entry, locator)
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_compare_sample_at, wsp_path, fcs_dir, i)
                for i in range(len(samples))]
        for fut in futs:
            yield fut.result()


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    force_utf8_streams()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('wsp', help='Path to the FlowJo .wsp workspace')
    ap.add_argument('--fcs-dir', default='',
                    help='Fallback directory for FCS files when the .wsp '
                         'DataSet uri does not resolve.')
    ap.add_argument('--csv', default='',
                    help='Output CSV path (default: <wsp>_compare.csv).')
    ap.add_argument('--html', default='',
                    help='Output HTML path (default: <wsp>_compare.html).')
    ap.add_argument('--workers', type=int, default=0,
                    help='Samples compared at once, each in its own process '
                         '(default: up to 4, never more than the cores). '
                         'The report is the same whatever the number.')
    args = ap.parse_args()

    if not os.path.isfile(args.wsp):
        sys.exit(f"workspace not found: {args.wsp}")
    base = os.path.splitext(args.wsp)[0]
    csv_path  = args.csv  or f'{base}_compare.csv'
    html_path = args.html or f'{base}_compare.html'

    print(f"Reading workspace: {args.wsp}")
    try:
        reader = WspReader(args.wsp)
    except WspParseError as exc:
        sys.exit(str(exc))
    locator = FcsLocator(args.wsp, args.fcs_dir or None)
    samples = _per_sample_inventory(reader, args.fcs_dir or None,
                                    locate=locator.find)
    if not samples:
        sys.exit("no SampleNodes with parseable gates found in the workspace.")

    all_rows = []
    for line, rows in _compare_samples(args.wsp, args.fcs_dir or None,
                                       samples, locator,
                                       _compare_workers(args.workers,
                                                        len(samples))):
        print(line)
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("nothing compared — either no FCS resolved, or no "
                 "populations carried event counts.")

    _write_csv(all_rows, csv_path)
    _write_html(all_rows, html_path, args.wsp)
    print(f"\nWrote {csv_path}  ({len(all_rows)} rows)")
    print(f"Wrote {html_path}")


if __name__ == '__main__':
    main()
