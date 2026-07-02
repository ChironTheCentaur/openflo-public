"""compare_workspace.py — Side-by-side review tool.

Reads a FlowJo .wsp workspace, locates each referenced FCS, re-applies
every Population gate via OpenFlo's evaluator, and compares the
resulting per-population event counts against the counts FlowJo wrote
into the .wsp itself (the `count="…"` attribute on each <Population>).

The output is a CSV (one row per sample × population) plus a tiny HTML
summary; both name FlowJo's count, OpenFlo's count, and the delta
(absolute + relative).

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
import sys
from collections import defaultdict
from urllib.parse import unquote, urlparse

from .pipeline import FlowSample, WspReader, cumulative_gate_mask

# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_fcs_uri(uri, fcs_dir=None):
    """Map a FlowJo DataSet uri (eg 'file:/path/to/sample.fcs' or plain path)
    onto a local path. If the original doesn't exist and `fcs_dir` is
    set, look for a basename match there."""
    if not uri:
        return None
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
    p = p.replace('/', os.sep)
    if os.path.isfile(p):
        return p
    if fcs_dir:
        candidate = os.path.join(fcs_dir, os.path.basename(p))
        if os.path.isfile(candidate):
            return candidate
    return None


# Per-sample raw inventory: every Population with its declared FlowJo
# count, parent linkage, and the underlying gate dict (in our schema).
def _per_sample_inventory(reader):
    """Walk the .wsp again — this time keeping track of (sample_name,
    fcs_uri, [(pop_name, flowjo_count, gate_dict, parent_pop_name), ...])
    Returns a list of (sample_name, fcs_uri, pops_list)."""
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
        pops = []          # (uid, pop_name, count, gate_dict, parent_uid)
        uid_ctr = [0]      # per-occurrence id so same-named pops don't collide

        def walk(pop_elem, parent_uid):
            uid_ctr[0] += 1                 # noqa: B023 (walk runs same-iteration)
            uid = uid_ctr[0]                # noqa: B023
            pop_name = pop_elem.get('name') or 'unnamed'
            try:
                count_attr = pop_elem.get('count')
                count = int(count_attr) if count_attr is not None else None
            except ValueError:
                count = None
            # Pull this Population's gate via the same parser the reader uses.
            g = _gate_from_population(pop_elem, reader)
            if g is not None:
                # B023: `pops` is loop-local but `walk` is consumed within
                # the same iteration — safe in practice.
                pops.append((uid, pop_name, count, g, parent_uid))  # noqa: B023
            for sub in pop_elem.findall('Subpopulations'):
                for child in sub.findall('Population'):
                    walk(child, uid)

        for sub in sn.findall('Subpopulations'):
            for pop in sub.findall('Population'):
                walk(pop, None)
        if pops:
            samples.append((sample_name, uri, pops))
    return samples


def _gate_from_population(pop_elem, reader):
    """Extract this Population's own gate using WspReader's existing
    rect/polygon parsers (avoids duplicating logic). Returns a gate dict
    or None if the gate kind is unsupported."""
    for gw in pop_elem.findall('Gate'):
        for child in gw:
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
                    parsed.append((ch, lo, hi))
                if len(parsed) == 1:
                    ch, lo, hi = parsed[0]
                    if lo is not None and hi is not None:
                        return {'kind': 'interval', 'channel': ch, 'lo': lo, 'hi': hi}
                    if lo is not None:
                        return {'kind': 'threshold', 'channel': ch, 'value': lo}
                if len(parsed) == 2:
                    (xc, x0, x1), (yc, y0, y1) = parsed
                    if None not in (x0, x1, y0, y1):
                        return {'kind': 'rect',
                                'x_channel': xc, 'y_channel': yc,
                                'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
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
                    return {'kind': 'polygon',
                            'x_channel': xc, 'y_channel': yc,
                            'vertices': verts}
    return None


# ── Per-sample comparison ────────────────────────────────────────────────────

def _compare_one_sample(sample_name, fcs_path, pops, wsp_path=None):
    """Apply each population's cumulative gate to the FCS data, report
    the OpenFlo count alongside FlowJo's declared count, and yield rows
    suitable for the CSV/HTML report.

    If `wsp_path` is provided, the workspace's spillover matrix is
    applied to the data BEFORE gate evaluation — essential for any
    gate on a fluorescence channel, since FlowJo authors gate
    thresholds against compensated values. Without this step, CD11b+ /
    CD45+ / etc. gates compare against raw data and miss almost
    everything."""
    s = FlowSample(fcs_path)
    if wsp_path:
        try:
            s.compensate_from_wsp(wsp_path)
        except Exception as exc:
            print(f"    [compensation skipped] {type(exc).__name__}: {exc}")
    total = len(s.data)
    # Key gates by the per-occurrence uid threaded from the walk, so populations
    # that SHARE a name (Q1-Q4, 'Single Cells', copied gates) don't overwrite
    # each other. Parent linkage uses the parent's uid directly; a parent that
    # produced no gate (or the root) resolves to None — same as before.
    gates_by_id = {uid: dict(gdict) for uid, _pn, _cnt, gdict, _puid in pops}
    for uid, _pn, _cnt, _gdict, parent_uid in pops:
        gates_by_id[uid]['parent_id'] = (
            parent_uid if parent_uid in gates_by_id else None)

    rows = []
    for uid, pname, flowjo_count, _gdict, _parent_uid in pops:
        try:
            mask = cumulative_gate_mask(gates_by_id, uid, s.data)
            openflo_count = int(mask.sum())
        except Exception as exc:
            openflo_count = None
            err = f'{type(exc).__name__}: {exc}'
        else:
            err = ''
        rows.append({
            'sample':        sample_name,
            'population':    pname,
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

def _write_csv(rows, path):
    fields = ['sample', 'population', 'flowjo_count', 'openflo_count',
              'delta', 'rel_delta', 'total_events', 'error']
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
                     "<th>Population</th><th>FlowJo</th><th>OpenFlo</th>"
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


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('wsp', help='Path to the FlowJo .wsp workspace')
    ap.add_argument('--fcs-dir', default='',
                    help='Fallback directory for FCS files when the .wsp '
                         'DataSet uri does not resolve.')
    ap.add_argument('--csv', default='',
                    help='Output CSV path (default: <wsp>_compare.csv).')
    ap.add_argument('--html', default='',
                    help='Output HTML path (default: <wsp>_compare.html).')
    args = ap.parse_args()

    if not os.path.isfile(args.wsp):
        sys.exit(f"workspace not found: {args.wsp}")
    base = os.path.splitext(args.wsp)[0]
    csv_path  = args.csv  or f'{base}_compare.csv'
    html_path = args.html or f'{base}_compare.html'

    print(f"Reading workspace: {args.wsp}")
    reader = WspReader(args.wsp)
    samples = _per_sample_inventory(reader)
    if not samples:
        sys.exit("no SampleNodes with parseable gates found in the workspace.")

    all_rows = []
    for sample_name, uri, pops in samples:
        fcs = _resolve_fcs_uri(uri, args.fcs_dir or None)
        if fcs is None:
            print(f"  [skip] {sample_name}: FCS not located "
                  f"(uri={uri!r}, --fcs-dir={args.fcs_dir!r})")
            continue
        print(f"  {sample_name}: {os.path.basename(fcs)} "
              f"({len(pops)} population(s))")
        try:
            rows = _compare_one_sample(sample_name, fcs, pops,
                                       wsp_path=args.wsp)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"    [error] {type(exc).__name__}: {exc}")

    if not all_rows:
        sys.exit("nothing compared — either no FCS resolved, or no "
                 "populations carried event counts.")

    _write_csv(all_rows, csv_path)
    _write_html(all_rows, html_path, args.wsp)
    print(f"\nWrote {csv_path}  ({len(all_rows)} rows)")
    print(f"Wrote {html_path}")


if __name__ == '__main__':
    main()
