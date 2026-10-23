"""Render a layout page (``layout_page.Layout``) for one sample (beta).

Plots go through the editor's own plotting code (``_render_into``), so a
layout panel looks like the live plot: same scales, density colouring and
gate outlines. A plot can overlay several samples and populations, keep the
axis types and zoom it was taken with, and colour by a cluster column.
Charts (violin / box / bar) compare populations across samples. No Tk here
beyond what the editor already owns; the window is ``ui_layout_editor``.
"""
from __future__ import annotations

import contextlib
import math

import numpy as np

from .errors import note
from .layout_page import (
    ALL_EVENTS,
    CHAN_STATS,
    PLOT_INSET,
    POP_STATS,
    axes_rect,
    child_gates,
    fill_text,
    item_populations,
    item_samples,
    split_box,
)

# Cluster-label columns a plot can be coloured by (as categories).
LABEL_COLUMNS = ('cluster', 'leiden', 'flowsom_meta', 'cell_cycle')
CAT_PREFIX = 'cat:'
OUTSIDE_WARN = 0.02           # offer Zoom to fit above this fraction
# The default colour list: layer 1 red, layer 2 blue, ... (a preference,
# ``layout_colors``; ``layout_color_override`` makes every layer use it
# instead of its sample's or gate's own colour).
DEFAULT_COLORS: tuple[str, ...] = ('#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd',
                  '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f')
_LAYER_COLORS = DEFAULT_COLORS
_CHART_INSET = (0.14, 0.22, 0.04, 0.11)


def color_prefs():
    """``(colours, override)`` from the preferences."""
    try:
        from .prefs import read_prefs
        prefs = read_prefs()
    except Exception:                                 # noqa: BLE001
        prefs = {}
    cols = [str(c) for c in (prefs.get('layout_colors') or ()) if c]
    return (cols or list(DEFAULT_COLORS),
            bool(prefs.get('layout_color_override', False)))


def layer_id(ctx, gid, pinned=True):
    """The key an item's ``layer_colors`` uses for one layer. An item that
    follows the page's sample keys by population only, so its colours hold
    for every sample."""
    pop = ctx.keys.get(gid, '') if gid else ''
    return f"{ctx.sample if pinned else ''}::{pop}"


class SampleContext:
    """One sample's gates, population keys and stats, computed once per
    render of a page (every item on it shares them)."""

    def __init__(self, ed, sample):
        from .gating import population_keys, population_path
        self.ed = ed
        self.sample = sample
        s = ed._samples.get(sample)
        self.loaded = s is not None
        self.gates = ed._sample_gates.get(sample, {}) or {}
        self.order = ed._sample_gate_order.get(sample, list(self.gates))
        labels = dict(getattr(ed, '_channel_labels', {}) or {})
        if s is not None:
            labels.update(getattr(s, 'channel_labels', {}) or {})
        self.labels = labels
        self.transforms = getattr(s, 'data_transforms', None) if s else None
        # A FlowJo workspace's axes, which its GeoMean is taken on.
        self.stat_axes = getattr(s, 'stat_axes', None) if s else None
        self.keys = population_keys(self.gates, labels)
        self.by_key = {k: gid for gid, k in self.keys.items()}
        self._path = lambda gid: population_path(
            self.gates, gid, transforms=self.transforms)
        self._rows: dict = {}

    def gid(self, key):
        """Gate id for population ``key`` (None = all events); KeyError when
        this sample has no such population."""
        if key is ALL_EVENTS or key == '':
            return None
        return self.by_key[key]

    def name(self, gid):
        if gid is None:
            return 'All events'
        return self._path(gid).split('/')[-1]

    def path(self, gid):
        return 'All events' if gid is None else self._path(gid)

    def rows(self, want, channels=()):
        """``{gid: stats row}`` over every population of the sample."""
        key = (tuple(sorted(want)), tuple(channels))
        if key not in self._rows:
            s = self.ed._samples.get(self.sample)
            if s is None:
                self._rows[key] = {}
            else:
                from .gating import population_stats
                rows = population_stats(
                    self.sample, s.data, self.gates, self.order, self.labels,
                    [c for c in channels if c in s.data.columns], set(want),
                    CHAN_STATS, transforms=self.transforms,
                    stat_axes=self.stat_axes)
                self._rows[key] = {r['__gid__']: r for r in rows}
        return self._rows[key]

    def unknown(self, gid):
        """Why population ``gid`` cannot be computed (an unconverted FlowJo
        ellipse, a broken boolean, ... or anything built on one), else
        None. Such a population is missing (NA), never a measured 0."""
        if gid is None:
            return None
        from .pipeline import unknown_population_reason
        s = self.ed._samples.get(self.sample)
        return unknown_population_reason(
            self.gates, gid, columns=None if s is None else s.data.columns)

    def n_events(self):
        s = self.ed._samples.get(self.sample)
        return 0 if s is None else len(s.data)

    def mask(self, gid):
        """Boolean mask of population ``gid`` over every event."""
        s = self.ed._samples[self.sample]
        if gid is None:
            return np.ones(len(s.data), dtype=bool)
        from .pipeline import cumulative_gate_mask
        return np.asarray(cumulative_gate_mask(self.gates, gid, s.data),
                          dtype=bool)


class RenderContext:
    """What one render of a page shares: the page's sample and a
    ``SampleContext`` per sample any item draws."""

    def __init__(self, ed, page_sample):
        self.ed = ed
        self.page_sample = page_sample
        self._ctx: dict = {}
        self.info: dict = {}           # item id -> {'outside', 'ax'}
        self.colors, self.override = color_prefs()

    def layer_colors(self, item, layers):
        """One colour per layer: the item's own choice for that layer, else
        the default list when overriding, else the sample's colour (layers
        that differ by sample) or the gate's colour (by population), else
        the default list."""
        own = item.get('layer_colors') or {}
        n_s = len({c.sample for c, _g in layers})
        n_p = len(set(item_populations(item)))
        out = []
        for i, (ctx, gid) in enumerate(layers):
            default = self.colors[i % len(self.colors)]
            c = own.get(layer_id(ctx, gid, bool(item.get('samples'))))
            if not c and not self.override:
                if n_s > 1 and n_p == 1:
                    c = self.ed._sample_colors.get(ctx.sample)
                elif n_p > 1 and gid is not None:
                    c = ctx.gates.get(gid, {}).get('color')
            out.append(c or default)
        return out

    def __call__(self, sample):
        if sample not in self._ctx:
            self._ctx[sample] = SampleContext(self.ed, sample)
        return self._ctx[sample]

    def layers(self, item):
        """``[(SampleContext, gid), ...]`` the item draws (sample x
        population) and how many pairs this data lacks."""
        out, missing = [], 0
        for sample in item_samples(item, self.page_sample):
            ctx = self(sample)
            if not ctx.loaded:
                missing += 1
                continue
            for key in item_populations(item):
                try:
                    out.append((ctx, ctx.gid(key)))
                except KeyError:
                    missing += 1
        return out, missing


def _fmt(v, stat):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return '—'
    if stat == 'Count':
        return f'{int(round(f)):,}'
    if stat.startswith('%') or stat in ('CV', 'rCV'):
        return f'{f:.2f}'
    return f'{f:,.1f}'


def _note(ax, text, color='#888888'):
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=8,
            color=color, transform=ax.transAxes, wrap=True)


def _layer_label(ctx, gid, n_samples, n_pops):
    parts = []
    if n_samples > 1:
        parts.append(str(ctx.sample))
    if n_pops > 1 or not parts:
        parts.append(ctx.name(gid))
    return ' · '.join(parts)


@contextlib.contextmanager
def _population(ed, gid):
    """Point the editor's ``_get_df`` at population ``gid`` for a while."""
    prev = getattr(ed, '_render_population', 'auto')
    ed._render_population = gid
    try:
        yield
    finally:
        ed._render_population = prev


@contextlib.contextmanager
def _view(ed, item):
    """Apply an item's pinned axis types and zoom to the editor's
    per-channel scale/range for the duration of a render. An item marked
    ``_fit`` (an exploded plot's overview) shows all its data whatever
    range the main window has."""
    saved = []
    for ch, scale, lim in ((item.get('x'), item.get('xscale'), item.get('xlim')),
                           (item.get('y'), item.get('yscale'), item.get('ylim'))):
        if not ch:
            continue
        saved.append((ch, ed._channel_scale.get(ch, _MISSING),
                      ed._channel_range.get(ch, _MISSING)))
        if scale:
            ed._channel_scale[ch] = scale
        if lim:
            ed._channel_range[ch] = (float(lim[0]), float(lim[1]))
        elif item.get('_fit'):
            ed._channel_range.pop(ch, None)
    try:
        yield
    finally:
        for ch, scale, rng in reversed(saved):
            for store, val in ((ed._channel_scale, scale),
                               (ed._channel_range, rng)):
                if val is _MISSING:
                    store.pop(ch, None)
                else:
                    store[ch] = val


_MISSING = object()


def _layer_frame(ed, ctx, gid, x, y, for_hist=False):
    with _population(ed, gid):
        return ed._get_df(ctx.sample, x, y, for_hist=for_hist)


def _category_colors(ctx, col, values):
    """``{label: colour}``: a cluster's own gate colour where an imported
    cluster/category gate has one, else a fixed palette."""
    own = {}
    for g in ctx.gates.values():
        if g.get('channel') == col and g.get('kind') in ('cluster', 'category'):
            v = g.get('cluster_id', g.get('value'))
            if v is not None and g.get('color'):
                own[str(v)] = g['color']
    import matplotlib
    cmap = matplotlib.colormaps['tab20']
    out = {}
    for i, v in enumerate(sorted({str(v) for v in values})):
        out[v] = own.get(v) or cmap(i % 20)
    return out


def _outside_fraction(frames, x, y, item):
    """Fraction of the plotted events outside a pinned zoom (0 when none)."""
    xl, yl = item.get('xlim'), item.get('ylim')
    if not (xl or yl):
        return 0.0
    n = out = 0
    for df in frames:
        if df is None or x not in df.columns:
            continue
        keep = np.ones(len(df), dtype=bool)
        xv = np.asarray(df[x].values, dtype=float)
        ok = np.isfinite(xv)
        if xl:
            lo, hi = sorted(xl)
            keep &= (xv >= lo) & (xv <= hi)
        if yl and y and y in df.columns:
            yv = np.asarray(df[y].values, dtype=float)
            ok &= np.isfinite(yv)
            lo, hi = sorted(yl)
            keep &= (yv >= lo) & (yv <= hi)
        n += int(ok.sum())
        out += int((ok & ~keep).sum())
    return out / n if n else 0.0


def _gate_anchor(g, x, y, ax):
    """Where a gate's percentage goes: ``(xy, transform)`` or None."""
    k = g.get('kind')
    xl, yl = sorted(ax.get_xlim()), sorted(ax.get_ylim())

    def clip(v, lim):
        return min(max(float(v), lim[0]), lim[1])
    if k in ('threshold', 'interval'):
        ch = g.get('channel')
        if k == 'threshold':
            v = float(g['value'])
        else:
            v = (float(g['lo']) + float(g['hi'])) / 2.0
        if ch == x:
            return (clip(v, xl), 0.93), ax.get_xaxis_transform()
        if ch == y:
            return (0.93, clip(v, yl)), ax.get_yaxis_transform()
        return None
    if g.get('x_channel') != x or g.get('y_channel') != y:
        return None
    if k == 'rect':
        cx = (clip(g['x0'], xl) + clip(g['x1'], xl)) / 2.0
        cy = (clip(g['y0'], yl) + clip(g['y1'], yl)) / 2.0
        return (cx, cy), ax.transData
    verts = g.get('vertices')
    if verts:
        v = np.asarray(verts, dtype=float)
        if v.ndim == 2 and v.shape[1] == 2 and len(v):
            cx = float(np.mean([clip(p, xl) for p in v[:, 0]]))
            cy = float(np.mean([clip(p, yl) for p in v[:, 1]]))
            return (cx, cy), ax.transData
    mean = g.get('mean')
    if mean is not None and len(mean) >= 2:
        return (clip(mean[0], xl), clip(mean[1], yl)), ax.transData
    return None


def _overlay_painter(ed, layers, item, x, y, mode, frames, colors):
    """Draw every layer into ``ed.ax`` (the target of ``_render_into``):
    step histograms, or scatter coloured per layer or by cluster."""
    color = item.get('color') or ''
    n_s = len({c.sample for c, _g in layers})
    n_p = len(set(item_populations(item)))

    def paint():
        ax = ed.ax
        for i, (ctx, gid) in enumerate(layers):
            df = _layer_frame(ed, ctx, gid, x, y, for_hist=mode == 'histogram')
            frames.append(df)
            if df is None or df.empty or x not in df.columns:
                continue
            lbl = _layer_label(ctx, gid, n_s, n_p)
            lc = colors[i]
            xv = np.asarray(df[x].values, dtype=float)
            if mode == 'histogram' or not y:
                xv = xv[np.isfinite(xv)]
                if not xv.size:
                    continue
                edges = ed._axis_bin_edges(xv, x, 128)
                h, e = np.histogram(xv, bins=edges)
                h = h / max(h.sum(), 1)
                ax.step(e[:-1], h, where='post', color=lc, lw=1.1, label=lbl)
                continue
            if y not in df.columns:
                continue
            yv = np.asarray(df[y].values, dtype=float)
            if color.startswith(CAT_PREFIX) and color[len(CAT_PREFIX):] in df:
                col = color[len(CAT_PREFIX):]
                cats = df[col].astype(str).values
                cmap = _category_colors(ctx, col, cats)
                ax.scatter(xv, yv, s=2, linewidths=0, alpha=0.6,
                           c=[cmap[c] for c in cats], rasterized=True)
            else:
                ax.scatter(xv, yv, s=2, linewidths=0, alpha=0.4, color=lc,
                           label=lbl, rasterized=True)
        if len(layers) > 1 and not color.startswith(CAT_PREFIX):
            ax.legend(fontsize=6, markerscale=4, loc='upper right',
                      framealpha=0.8)
        elif color.startswith(CAT_PREFIX) and layers:
            col = color[len(CAT_PREFIX):]
            ctx = layers[0][0]
            seen = set()
            for df in frames:
                if df is not None and col in df:
                    seen |= set(df[col].astype(str).values)
            cmap = _category_colors(ctx, col, seen)
            import matplotlib.lines as mlines
            hands = [mlines.Line2D([], [], ls='', marker='o', ms=4,
                                   color=cmap[v], label=v)
                     for v in sorted(cmap)[:20]]
            if hands:
                ax.legend(handles=hands, fontsize=5.5, loc='upper right',
                          ncol=2 if len(hands) > 8 else 1, framealpha=0.8,
                          title=col, title_fontsize=6)
    return paint


def _draw_one(rc, fig, box, item, iid=None):
    """One plot panel: ``(ax, meta)``, meta None when only a note was drawn."""
    ax = fig.add_axes(axes_rect(box, PLOT_INSET))
    ed = rc.ed
    layers, missing = rc.layers(item)
    if not layers:
        _note(ax, 'This sample has no such population' if missing
              else f'Sample not loaded:\n{rc.page_sample}')
        return ax, None
    x, y, mode = item.get('x'), item.get('y'), item.get('mode', 'dot')
    if mode == 'histogram':
        y = None
    known = [(c, g) for c, g in layers if not c.unknown(g)]
    if not known:
        _note(ax, f'NA: this population cannot be computed\n'
                  f'({layers[0][0].unknown(layers[0][1])})')
        return ax, None
    layers = known
    ctx0, gid0 = layers[0]
    if item.get('_outline'):        # an exploded gate's own outline
        drawn = {g: ctx0.gates[g] for g in item['_outline']
                 if g in ctx0.gates}
    else:
        drawn = (child_gates(ctx0.gates, gid0)
                 if item.get('draw_gates', True) else {})
    color = item.get('color') or 'By density'
    frames: list = []
    painter = None
    if len(layers) > 1 or color.startswith(CAT_PREFIX):
        painter = _overlay_painter(ed, layers, item, x, y, mode, frames,
                                   rc.layer_colors(item, layers))
    with _view(ed, item):
        ed._render_into(ax, [ctx0.sample], x, y, mode,
                        'By density' if painter else color,
                        draw_gates=bool(drawn), draw_overlays=False,
                        population=gid0, gates=drawn, painter=painter)
    # The plotting code's own notes ('n events cannot be shown on a log
    # axis') are sized for the live plot; in a small panel they would spill
    # over the next item.
    for t in ax.texts:
        t.set_clip_on(True)
    if item.get('title'):
        title = item['title']
    elif len(layers) == 1:
        title = ctx0.name(gid0)
    else:
        title = ', '.join(dict.fromkeys(
            c.name(g) for c, g in layers))[:60]
    ax.set_title(title, fontsize=8)
    if drawn and x and len(layers) == 1:
        rows = ctx0.rows({'Count', '%Parent'})
        for cg, g in drawn.items():
            r = rows.get(cg)
            pos = _gate_anchor(g, x, y, ax)
            if r is None or pos is None:
                continue
            (px, py), tr = pos
            pct = ('NA' if r.get('__unknown__')
                   else f"{_fmt(r.get('%Parent'), '%')}%")
            # No box behind the label (it read as a patch of another
            # background): a thin halo in the plot's own colour keeps it
            # legible over the events.
            import matplotlib.patheffects as pe
            t = ax.text(px, py, f"{ctx0.name(cg)}\n{pct}",
                        transform=tr, ha='center', va='center', fontsize=6.5,
                        color=g.get('color', 'black'), clip_on=True)
            t.set_path_effects([pe.withStroke(linewidth=2,
                                              foreground=ax.get_facecolor())])
    if iid is not None:
        rc.info.setdefault(iid, {})['ax'] = ax        # for zooming it
    frac = 0.0
    if item.get('xlim') or item.get('ylim'):
        if not frames:
            frames = [_layer_frame(ed, c, g, x, y, mode == 'histogram')
                      for c, g in layers]
        frac = _outside_fraction(frames, x, y, item)
        if iid is not None:
            rc.info.setdefault(iid, {})['outside'] = frac
    return ax, {'layers': layers, 'x': x, 'y': y, 'outside': frac}


def draw_plot(rc, fig, box, item, iid=None):
    ax, meta = _draw_one(rc, fig, box, item, iid)
    if meta is not None and not item.get('_part'):
        _explode(rc, fig, box, item, ax, meta, iid)
    return ax


# ── the exploded view ───────────────────────────────────────────────────
# A zoomed plot (or one gate on it) magnified beside the whole plot: the
# region outlined on the whole plot, its corners joined to the magnified
# copy. Both halves are ordinary plot panels (``_draw_one``).

EXPLODE_PAD = 0.06        # room around an exploded gate, fraction of its span
_REGION_COLOR = '#444444'
REGION_GID = 'openflo-exploded-region'   # the outline, for finding it again


def explode_target(rc, item, meta):
    """What an item's exploded view magnifies: ``('gate', gid)``,
    ``('zoom', None)`` or None (drawn as an ordinary plot)."""
    want = item.get('explode')
    if want is False:
        return None
    layers = meta['layers']
    hist = not meta['y']
    key = item.get('explode_gate')
    if key and len(layers) == 1:
        ctx, gid = layers[0]
        try:
            cg = ctx.gid(key)
        except KeyError:
            cg = None
        if (cg is not None and cg in child_gates(ctx.gates, gid)
                and not ctx.unknown(cg)):
            return ('gate', cg)
    if not (item.get('xlim') or (item.get('ylim') and not hist)):
        return None
    if want is None and meta['outside'] <= OUTSIDE_WARN:
        return None           # the zoom shows (nearly) everything already
    return ('zoom', None)


def _explode(rc, fig, box, item, ax, meta, iid):
    target = explode_target(rc, item, meta)
    if target is None:
        return
    x, y = meta['x'], meta['y']
    fw, fh = fig.get_size_inches()
    aspect = (box[2] * fw) / max(box[3] * fh, 1e-9)
    over_box, inset_box, side = split_box(
        box, item.get('inset_side') or 'auto', aspect)
    part = dict(item, explode=False, explode_gate=None, _part=True)
    kind, cg = target
    if kind == 'zoom':
        ax.set_position(axes_rect(inset_box, PLOT_INSET))
        ax.set_title(f'{ax.get_title()} (zoomed)', fontsize=8)
        inset = ax
        over, _m = _draw_one(rc, fig, over_box,
                             dict(part, xlim=None, ylim=None, _fit=True))
        region = (sorted(ax.get_xlim()),
                  sorted(ax.get_ylim()) if y else None)
    else:
        ctx = meta['layers'][0][0]
        region = _gate_region(rc.ed, ctx, cg, x, y, ax)
        if region is None:
            return
        ax.set_position(axes_rect(over_box, PLOT_INSET))
        over = ax
        inset, _m = _draw_one(rc, fig, inset_box, dict(
            part, populations=[ctx.keys[cg]], xlim=region[0],
            ylim=region[1], title=None, _outline=[cg]))
    _include(over, region)
    region = _clip(over, region)
    _outline_region(over, region)
    _connect(fig, over, inset, region, side)
    if iid is not None:
        info = rc.info.setdefault(iid, {})
        info.update(exploded=kind, overview=over, inset=inset)


def _finite(vals):
    from .editor_gating import _is_open
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and not _is_open(f):
            out.append(f)
    return out


def _gate_extent(g, x, y):
    """``(xs, ys)``: coordinates of gate ``g``'s outline on a plot of
    ``x`` against ``y`` (open sides and other axes left out)."""
    k = g.get('kind')
    xs: list = []
    ys: list = []
    if k in ('threshold', 'interval'):
        vals = ([g.get('value')] if k == 'threshold'
                else [g.get('lo'), g.get('hi')])
        if g.get('channel') == x:
            xs = _finite(vals)
        elif y and g.get('channel') == y:
            ys = _finite(vals)
        return xs, ys
    if g.get('x_channel') != x or g.get('y_channel') != y:
        return xs, ys
    if k == 'rect':
        return _finite([g.get('x0'), g.get('x1')]), _finite([g.get('y0'), g.get('y1')])
    if k == 'polygon':
        v = np.asarray(g.get('vertices') or [], dtype=float)
        if v.ndim == 2 and v.shape[1] == 2:
            return _finite(v[:, 0]), _finite(v[:, 1])
    if k == 'ellipsoid' and g.get('mean') is not None and g.get('cov') is not None:
        try:
            m = np.asarray(g['mean'], dtype=float)[:2]
            c = np.asarray(g['cov'], dtype=float)[:2, :2]
            d = float(g.get('distance_sq', 4.0))
            hx, hy = math.sqrt(d * c[0, 0]), math.sqrt(d * c[1, 1])
        except (TypeError, ValueError, IndexError):
            return xs, ys
        return _finite([m[0] - hx, m[0] + hx]), _finite([m[1] - hy, m[1] + hy])
    return xs, ys


def _gate_region(ed, ctx, cg, x, y, ax):
    """The limits ``(xlim, ylim)`` an exploded gate is shown at: its
    outline and its events, with a margin in each axis's own scale. None
    when the gate has neither on this plot."""
    df = _layer_frame(ed, ctx, cg, x, y, for_hist=not y)
    gx, gy = _gate_extent(ctx.gates[cg], x, y)
    out = []
    for ch, shape, axis, lim in ((x, gx, ax.xaxis, ax.get_xlim()),
                                 (y, gy, ax.yaxis, ax.get_ylim())):
        if not ch:
            out.append(None)
            continue
        vals = list(shape)
        if df is not None and ch in df.columns:
            v = np.asarray(df[ch].values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size:
                vals += [float(v.min()), float(v.max())]
        if not vals:
            return None
        tr = axis.get_transform()
        t = np.asarray(tr.transform(np.asarray([min(vals), max(vals)])),
                       dtype=float)
        full = np.asarray(tr.transform(np.asarray(sorted(lim), dtype=float)),
                          dtype=float)
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(full))):
            out.append(sorted(lim))
            continue
        pad = (t[1] - t[0]) * EXPLODE_PAD or abs(full[1] - full[0]) * 0.05
        back = np.asarray(tr.inverted().transform(
            np.asarray([t[0] - pad, t[1] + pad])), dtype=float)
        out.append(sorted(float(b) for b in back)
                   if np.all(np.isfinite(back)) else [min(vals), max(vals)])
    return out[0], out[1]


def _include(ax, region):
    """Widen ``ax``'s limits (in each axis's own scale) to take in the
    region, so its outline is never cut off."""
    for lim, axis, get, put in ((region[0], ax.xaxis, ax.get_xlim, ax.set_xlim),
                                (region[1], ax.yaxis, ax.get_ylim, ax.set_ylim)):
        if lim is None:
            continue
        cur = sorted(get())
        tr = axis.get_transform()
        try:
            c = tr.transform(np.asarray(cur, dtype=float))
            r = tr.transform(np.asarray(lim, dtype=float))
        except Exception as exc:
            note("transforming axis limits for the figure layout", exc)
            continue
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
            continue
        lo, hi = min(c[0], r[0]), max(c[1], r[1])
        if lo < c[0] or hi > c[1]:
            span = hi - lo
            back = tr.inverted().transform(
                np.asarray([lo - 0.02 * span if lo < c[0] else lo,
                            hi + 0.02 * span if hi > c[1] else hi]))
            if np.all(np.isfinite(back)):
                put(float(back[0]), float(back[1]))


def _clip(ax, region):
    """The region cut to ``ax``'s limits: a part no axis position can
    show (below zero on a log view) is outlined from the axis's edge."""
    out = []
    for lim, cur in ((region[0], ax.get_xlim()), (region[1], ax.get_ylim())):
        if lim is None:
            out.append(None)
            continue
        lo, hi = sorted(cur)
        out.append([min(max(float(lim[0]), lo), hi),
                    max(min(float(lim[1]), hi), lo)])
    return out[0], out[1]


def _corners(region):
    """``(transform-maker, x0, x1, y0, y1)``: a histogram's region spans
    the axes' height."""
    (x0, x1) = region[0]
    if region[1] is None:
        return (lambda a: a.get_xaxis_transform()), x0, x1, 0.0, 1.0
    (y0, y1) = region[1]
    return (lambda a: a.transData), x0, x1, y0, y1


def _outline_region(ax, region):
    import matplotlib.patches as mpatches
    tmk, x0, x1, y0, y1 = _corners(region)
    ax.add_artist(mpatches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=tmk(ax), fill=False,
        edgecolor=_REGION_COLOR, linewidth=1.0, zorder=7, gid=REGION_GID))


# Which corners of the region join which corners of the inset, per side.
_LINKS = {'right': (((1, 1), (0, 1)), ((1, 0), (0, 0))),
          'left': (((0, 1), (1, 1)), ((0, 0), (1, 0))),
          'below': (((0, 0), (0, 1)), ((1, 0), (1, 1))),
          'above': (((0, 1), (0, 0)), ((1, 1), (1, 0)))}


def _connect(fig, over, inset, region, side):
    from matplotlib.patches import ConnectionPatch
    tmk, x0, x1, y0, y1 = _corners(region)
    for (rx, ry), (ix, iy) in _LINKS[side]:
        con = ConnectionPatch(
            xyA=(x1 if rx else x0, y1 if ry else y0), coordsA=tmk(over),
            xyB=(ix, iy), coordsB=inset.transAxes,
            color=_REGION_COLOR, linewidth=0.8, linestyle='-', alpha=0.8,
            zorder=0)
        fig.add_artist(con)


def draw_stats(rc, fig, box, item, iid=None):
    # Keep the title's line inside the box (a preview is cut at the box).
    ax = fig.add_axes(axes_rect(box, (0.0, 0.0, 0.0, 0.12)))
    ax.set_axis_off()
    stats = [s for s in item.get('stats') or () if s in POP_STATS + CHAN_STATS]
    chan = item.get('channel')
    chan_stats = [s for s in stats if s in CHAN_STATS] if chan else []
    pop_stats = [s for s in stats if s in POP_STATS]
    ax.set_title(item.get('title') or 'Statistics', fontsize=8, loc='left')
    samples = item_samples(item, rc.page_sample)
    multi = len(samples) > 1
    lbl = ''
    body = []
    shown_rows = []             # whose GeoMean axis the heading names
    for sample in samples:
        ctx = rc(sample)
        if not ctx.loaded:
            body.append(([sample] if multi else [])
                        + ['(sample not loaded)'] + ['—'] * (
                            len(pop_stats) + len(chan_stats)))
            continue
        rows = ctx.rows(set(pop_stats) | set(chan_stats),
                        [chan] if chan_stats else [])
        lbl = ctx.labels.get(chan, chan) if chan else ''
        for key in item.get('populations') or ():
            lead = [sample] if multi else []
            try:
                gid = ctx.gid(key)
            except KeyError:
                body.append(lead + ['(not in this sample)']
                            + ['—'] * (len(pop_stats) + len(chan_stats)))
                continue
            if gid is None:
                n = ctx.n_events()
                body.append(lead + ['All events'] + [
                    _fmt(n, 'Count') if s == 'Count'
                    else ('100.00' if s == '%Total' else '—')
                    for s in pop_stats] + ['—'] * len(chan_stats))
                continue
            r = rows.get(gid, {})
            shown_rows.append(r)
            if r.get('__unknown__') or ctx.unknown(gid):
                # Missing, not measured: NA in every statistic.
                body.append(lead + [ctx.name(gid)]
                            + ['NA'] * (len(pop_stats) + len(chan_stats)))
                continue
            body.append(lead + [ctx.name(gid)]
                        + [_fmt(r.get(s), s) for s in pop_stats]
                        + [_fmt(r.get(f'{s} {lbl}'), s) for s in chan_stats])
    from .gating import geomean_axis_note
    geo_axis = geomean_axis_note(shown_rows, lbl) if lbl else None
    cols = ((['Sample'] if multi else []) + ['Population'] + pop_stats
            + [f'{s} {lbl}' + (f' ({geo_axis})' if s == 'GeoMean' and geo_axis
                               else '') for s in chan_stats])
    if not body:
        _note(ax, 'No populations chosen')
        return ax
    tbl = ax.table(cellText=body, colLabels=cols, loc='upper center',
                   cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_text_props(weight='bold')
    return ax


def draw_chart(rc, fig, box, item, iid=None):
    """Violin / box of a channel per population and sample, or bars of one
    statistic per sample for each population."""
    ax = fig.add_axes(axes_rect(box, _CHART_INSET))
    ed = rc.ed
    kind = item.get('chart', 'violin')
    chan = item.get('channel')
    layers, missing = rc.layers(item)
    if not layers:
        _note(ax, 'This sample has no such population' if missing
              else 'No populations chosen')
        return ax
    n_s = len({c.sample for c, _g in layers})
    n_p = len(set(item_populations(item)))
    labels = [_layer_label(c, g, n_s, n_p) for c, g in layers]
    colors = rc.layer_colors(item, layers)
    if kind == 'bar':
        value = item.get('value', '%Parent')
        chan_stat = value in CHAN_STATS
        if chan_stat and not chan:
            _note(ax, f'Choose a channel for {value}')
            return ax
        vals = []
        geo_rows = []               # whose GeoMean axis the label names
        for ctx, gid in layers:
            if gid is None:
                v = (ctx.n_events() if value == 'Count'
                     else 100.0 if value == '%Total' else float('nan'))
                if chan_stat:
                    s = ed._samples[ctx.sample]
                    from .pipeline import to_linear
                    lin = to_linear(s.data[chan].values,
                                    (ctx.transforms or {}).get(chan))
                    v = float(np.nanmedian(lin)) if value == 'Median' else (
                        float(np.nanmean(lin)))
            else:
                rows = ctx.rows({value}, [chan] if chan_stat else [])
                lbl = ctx.labels.get(chan, chan) if chan else ''
                r = rows.get(gid, {})
                # Each sample names the channel by its own label.
                ax_used = (r.get('__geomean_axis__') or {}).get(lbl)
                if ax_used:
                    geo_rows.append({'__geomean_axis__': {'': ax_used}})
                v = r.get(f'{value} {lbl}' if chan_stat else value,
                          float('nan'))
            if ctx.unknown(gid):
                v = float('nan')
            vals.append(float(v) if v is not None else float('nan'))
        ax.bar(range(len(vals)), vals, color=colors, edgecolor='none')
        for i, (ctx, gid) in enumerate(layers):
            if ctx.unknown(gid):
                ax.text(i, 0, 'NA', ha='center', va='bottom', fontsize=7,
                        color='#888888')
        from .gating import geomean_axis_note
        geo_axis = (geomean_axis_note(geo_rows, '')
                    if value == 'GeoMean' else None)
        ax.set_ylabel(f'{value} {ctx_label(layers, chan)}'.strip()
                      + (f' ({geo_axis})' if geo_axis else ''), fontsize=7)
    else:
        if not chan:
            _note(ax, 'Choose a channel to compare')
            return ax
        data = []
        for ctx, gid in layers:
            s = ed._samples[ctx.sample]
            if chan not in s.data.columns or ctx.unknown(gid):
                data.append(np.array([np.nan]))
                continue
            v = np.asarray(s.data[chan].values, dtype=float)[ctx.mask(gid)]
            v = v[np.isfinite(v)]
            if v.size > 20000:              # a KDE needs no more than this
                v = np.random.default_rng(0).choice(v, 20000, replace=False)
            data.append(v if v.size else np.array([np.nan]))
        ok = [i for i, d in enumerate(data) if np.isfinite(d).any()
              and d.size > 1]
        if kind == 'violin' and ok:
            parts = ax.violinplot([data[i] for i in ok], positions=ok,
                                  showmedians=True, widths=0.8)
            bodies = parts.get('bodies', []) if isinstance(parts, dict) \
                else []
            for j, b in enumerate(bodies):
                b.set_facecolor(colors[ok[j]])
                b.set_alpha(0.55)
        elif ok:
            ax.boxplot([data[i] for i in ok], positions=ok, widths=0.6,
                       showfliers=False)
        # Stored values on the channel's own display scale, ticked in
        # intensities as the live axes are.
        with _on_axes(ed, ax):
            try:
                ed._apply_axis_to_ax(chan, 'y', np.concatenate(
                    [d for d in data if d.size]))
            except Exception as exc:
                note("applying the axis range in the figure layout", exc)
        for i, (ctx, gid) in enumerate(layers):
            if ctx.unknown(gid):
                ax.text(i, 0.5, 'NA', ha='center', va='center', fontsize=7,
                        color='#888888', transform=ax.get_xaxis_transform())
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.set_ylabel(ctx_label(layers, chan), fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_title(item.get('title') or {
        'violin': 'Distribution', 'box': 'Distribution',
        'bar': item.get('value', '')}.get(kind, ''), fontsize=8)
    return ax


def ctx_label(layers, chan):
    if not chan or not layers:
        return ''
    return layers[0][0].labels.get(chan, chan)


@contextlib.contextmanager
def _on_axes(ed, ax):
    saved = (ed.ax, ed.fig)
    ed.ax, ed.fig = ax, ax.figure
    try:
        yield
    finally:
        ed.ax, ed.fig = saved


def draw_text(rc, fig, box, item, iid=None):
    ax = fig.add_axes(axes_rect(box))
    ax.set_axis_off()
    ax.text(0.0, 0.5, fill_text(item.get('text', ''), rc.page_sample),
            ha='left', va='center', fontsize=float(item.get('size', 12)),
            transform=ax.transAxes)
    return ax


_DRAW = {'plot': draw_plot, 'stats': draw_stats, 'text': draw_text,
         'chart': draw_chart}


def draw_item(rc, fig, box, item, iid=None):
    fn = _DRAW.get(item.get('kind'))
    if fn is None:
        return None
    try:
        return fn(rc, fig, box, item, iid)
    except Exception as exc:                          # noqa: BLE001
        ax = fig.add_axes(axes_rect(box))
        _note(ax, f'Could not draw:\n{type(exc).__name__}: {exc}', 'red')
        return ax


def render_page(ed, layout, sample, dpi=100, rc=None):
    """The whole page for ``sample`` as a matplotlib Figure of the page's
    size in inches."""
    from matplotlib.figure import Figure
    w, h = layout.page_size
    fig = Figure(figsize=(w, h), dpi=dpi, facecolor='white')
    rc = rc or RenderContext(ed, sample)
    for iid in layout.order:
        it = layout.items[iid]
        draw_item(rc, fig, it['box'], it, iid)
    return fig


def render_item(ed, layout, iid, sample, px_per_inch, rc=None):
    """One item alone, at its size on the page, for the editor's preview.
    Drawn into a figure the size of its box so it looks as on the page."""
    from matplotlib.figure import Figure
    it = layout.items[iid]
    pw, ph = layout.page_size
    _x, _y, bw, bh = it['box']
    fig = Figure(figsize=(max(bw * pw, 0.1), max(bh * ph, 0.1)),
                 dpi=px_per_inch, facecolor='white')
    draw_item(rc or RenderContext(ed, sample), fig, (0.0, 0.0, 1.0, 1.0),
              it, iid)
    return fig


def label_columns(ed):
    """Cluster-label columns present in any loaded sample."""
    out = []
    for s in getattr(ed, '_samples', {}).values():
        cols = getattr(getattr(s, 'data', None), 'columns', ())
        for c in LABEL_COLUMNS:
            if c in cols and c not in out:
                out.append(c)
    return out
