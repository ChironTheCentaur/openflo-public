"""The layout editor's page model (beta) -- Tk-free, headless-testable.

A layout is a page holding items: plots of a gated population, stats tables
and text boxes. Each item's box is in PAGE FRACTIONS (0..1, origin top-left,
as the editor's canvas draws it), so one layout renders the same on screen,
on any page size, and into a matplotlib Figure (``axes_rect`` turns a box
into the Figure's bottom-left ``add_axes`` rectangle).

Populations are named by their cross-sample key (``gating.population_keys``),
never by gate id: ids differ between samples, and the point of a layout is
that it re-applies to every sample (FlowJo's batch). ``resolve_population``
maps a key back to one sample's gate id.
"""
from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Mapping

FORMAT = 'openflo-layout'
VERSION = 2          # 2: several pages per file (1: one page)

# (label, width_in, height_in)
PAGE_SIZES = {
    'Letter landscape': (11.0, 8.5),
    'Letter portrait': (8.5, 11.0),
    'A4 landscape': (11.69, 8.27),
    'A4 portrait': (8.27, 11.69),
}
DEFAULT_PAGE = 'Letter landscape'

ITEM_KINDS = ('plot', 'stats', 'text', 'chart')
POP_STATS = ('Count', '%Parent', '%Total')
CHAN_STATS = ('Median', 'Mean', 'GeoMean', 'CV', 'rCV')

MIN_SIZE = 0.04          # smallest box side, page fraction
ALL_EVENTS = None        # the population key of the ungated sample

_ids = itertools.count(1)


def _new_id(existing=()):
    while True:
        iid = f'item{next(_ids)}'
        if iid not in existing:
            return iid


def clamp_box(x, y, w, h):
    """Keep a box on the page and no smaller than ``MIN_SIZE``."""
    w = min(max(float(w), MIN_SIZE), 1.0)
    h = min(max(float(h), MIN_SIZE), 1.0)
    x = min(max(float(x), 0.0), 1.0 - w)
    y = min(max(float(y), 0.0), 1.0 - h)
    return x, y, w, h


def snap(v, step):
    """Round ``v`` to the nearest multiple of ``step`` (no-op for step <= 0)."""
    if not step or step <= 0:
        return float(v)
    return round(float(v) / step) * step


def axes_rect(box, inset=(0.0, 0.0, 0.0, 0.0)):
    """matplotlib ``add_axes`` rect ``[left, bottom, width, height]`` (figure
    fractions, origin bottom-left) for a page box ``(x, y, w, h)`` (origin
    top-left). ``inset`` (left, bottom, right, top) is a fraction of the box
    kept free inside it, for tick labels and a title."""
    x, y, w, h = box
    il, ib, ir, it = inset
    left = x + il * w
    bottom = (1.0 - y - h) + ib * h
    return [left, bottom, max(w * (1.0 - il - ir), 1e-6),
            max(h * (1.0 - ib - it), 1e-6)]


# Room for tick labels, axis labels and a title inside a plot's box.
PLOT_INSET = (0.20, 0.17, 0.05, 0.11)


def plot_item(population=ALL_EVENTS, x=None, y=None, mode='dot',
              color='By density', draw_gates=True, title=None,
              box=(0.0, 0.0, 0.3, 0.3), populations=None, samples=(),
              xscale=None, yscale=None, xlim=None, ylim=None,
              explode=None, explode_gate=None, inset_side='auto'):
    """A plot. ``populations`` (default ``[population]``) and ``samples``
    (default: the page's sample) are overlaid, one layer per pair.
    ``xscale``/``yscale`` pin the axis types and ``xlim``/``ylim`` the zoom
    (stored coordinates) as they were in the plot it was taken from; None
    follows the editor's current setting / autoscale.

    The exploded view: the whole plot beside a magnified copy of a region,
    the region outlined and joined to the copy by connector lines.
    ``explode_gate`` (a child gate's population key) magnifies that gate's
    events; otherwise the region is the zoom. ``explode`` None explodes a
    zoom that hides part of the data (``layout_render.OUTSIDE_WARN``),
    True any zoom, False never. ``inset_side`` is one of ``INSET_SIDES``."""
    pops = list(populations) if populations else [population]
    return {'kind': 'plot', 'populations': pops, 'samples': list(samples),
            'x': x, 'y': y, 'mode': mode, 'color': color,
            'draw_gates': bool(draw_gates), 'title': title,
            'xscale': xscale, 'yscale': yscale,
            'xlim': list(xlim) if xlim else None,
            'ylim': list(ylim) if ylim else None,
            'explode': None if explode is None else bool(explode),
            'explode_gate': explode_gate,
            'inset_side': inset_side if inset_side in INSET_SIDES else 'auto',
            'box': list(clamp_box(*box))}


# Where an exploded plot's magnified copy goes ('auto': beside a wide box,
# below a tall one).
INSET_SIDES = ('auto', 'right', 'left', 'below', 'above')


def split_box(box, side, aspect=1.0, gap=0.04):
    """``(overview, inset, side)``: page box ``box`` cut in two halves for
    an exploded plot, the inset on ``side``. ``aspect`` is the box's width
    over its height in inches (for 'auto'); ``gap`` a fraction of the box
    left between the halves for the connector lines."""
    x, y, w, h = box
    if side not in INSET_SIDES[1:]:
        side = 'right' if aspect >= 1.15 else 'below'
    if side in ('right', 'left'):
        hw = w * (1.0 - gap) / 2.0
        a, b = (x, y, hw, h), (x + w - hw, y, hw, h)
        return (a, b, side) if side == 'right' else (b, a, side)
    hh = h * (1.0 - gap) / 2.0
    a, b = (x, y, w, hh), (x, y + h - hh, w, hh)
    return (a, b, side) if side == 'below' else (b, a, side)


def item_populations(item):
    """An item's population keys (old single-``population`` items too)."""
    pops = item.get('populations')
    if pops:
        return list(pops)
    return [item.get('population', ALL_EVENTS)]


def item_samples(item, page_sample):
    """The samples an item draws: its own list, else the page's sample."""
    return list(item.get('samples') or ()) or [page_sample]


def stats_item(populations=(), stats=('Count', '%Parent'), channel=None,
               title=None, box=(0.0, 0.0, 0.3, 0.2), samples=()):
    return {'kind': 'stats', 'populations': list(populations),
            'samples': list(samples),
            'stats': [s for s in stats if s in POP_STATS + CHAN_STATS],
            'channel': channel, 'title': title,
            'box': list(clamp_box(*box))}


CHART_TYPES = ('violin', 'box', 'bar')
CHART_VALUES = ('%Parent', '%Total', 'Count', 'Median', 'Mean', 'GeoMean')


def chart_item(chart='violin', populations=(), samples=(), channel=None,
               value='Median', title=None, box=(0.0, 0.0, 0.3, 0.3)):
    """A statistics chart across samples and populations.

    ``violin`` / ``box``: the per-event distribution of ``channel`` in each
    population (one per sample x population). ``bar``: one ``value`` per
    sample for each population (a frequency or MFI comparison)."""
    return {'kind': 'chart', 'chart': chart if chart in CHART_TYPES
            else 'violin', 'populations': list(populations),
            'samples': list(samples), 'channel': channel,
            'value': value if value in CHART_VALUES else 'Median',
            'title': title, 'box': list(clamp_box(*box))}


def text_item(text='{sample}', size: float = 12.0, box=(0.0, 0.0, 0.4, 0.06)):
    return {'kind': 'text', 'text': str(text), 'size': float(size),
            'box': list(clamp_box(*box))}


class Layout:
    """An ordered list of items (later items draw on top) on one page."""

    def __init__(self, page=DEFAULT_PAGE, items=None, name='Page 1'):
        self.page = page if page in PAGE_SIZES else DEFAULT_PAGE
        self.name = str(name or 'Page')
        self.items: dict[str, dict] = {}
        for it in items or ():
            self.add(it)

    # ── items ────────────────────────────────────────────────────────────
    @property
    def order(self):
        return list(self.items)

    def add(self, item, item_id=None):
        """Add ``item`` (a dict from ``plot_item`` / ``stats_item`` /
        ``text_item``); returns its id."""
        if item.get('kind') not in ITEM_KINDS:
            raise ValueError(f"unknown layout item kind {item.get('kind')!r}")
        iid = item_id if item_id and item_id not in self.items \
            else _new_id(self.items)
        it = copy.deepcopy(dict(item))
        it['box'] = list(clamp_box(*it.get('box', (0, 0, 0.3, 0.3))))
        self.items[iid] = it
        return iid

    def remove(self, iid):
        self.items.pop(iid, None)

    def duplicate(self, iid, offset=0.03):
        it = copy.deepcopy(self.items[iid])
        x, y, w, h = it['box']
        it['box'] = list(clamp_box(x + offset, y + offset, w, h))
        return self.add(it)

    def raise_to_top(self, iid):
        if iid in self.items:
            self.items[iid] = self.items.pop(iid)

    def move(self, iid, dx, dy, step=0.0):
        x, y, w, h = self.items[iid]['box']
        self.items[iid]['box'] = list(clamp_box(
            snap(x + dx, step), snap(y + dy, step), w, h))

    def resize(self, iid, dw, dh, step=0.0):
        """Grow/shrink from the bottom-right corner (top-left stays put)."""
        x, y, w, h = self.items[iid]['box']
        nw = snap(x + w + dw, step) - x
        nh = snap(y + h + dh, step) - y
        nw = min(max(nw, MIN_SIZE), 1.0 - x)
        nh = min(max(nh, MIN_SIZE), 1.0 - y)
        self.items[iid]['box'] = list(clamp_box(x, y, nw, nh))

    def hit(self, fx, fy):
        """Topmost item whose box contains page point ``(fx, fy)``, or None."""
        for iid in reversed(self.order):
            x, y, w, h = self.items[iid]['box']
            if x <= fx <= x + w and y <= fy <= y + h:
                return iid
        return None

    def next_slot(self, w=0.3, h=0.3, cols=3, margin=0.02):
        """A free-ish box for a new item: the first cell of a ``cols``-wide
        grid that no existing item's top-left falls in."""
        taken = {(round(it['box'][0], 3), round(it['box'][1], 3))
                 for it in self.items.values()}
        step_x = (1.0 - margin) / cols
        step_y = h + margin
        for row in range(int(1.0 / max(step_y, MIN_SIZE)) + 1):
            for col in range(cols):
                x = margin + col * step_x
                y = margin + row * step_y
                if (round(x, 3), round(y, 3)) in taken:
                    continue
                if y + h <= 1.0 + 1e-9:
                    return clamp_box(x, y, min(w, step_x - margin), h)
        return clamp_box(margin, margin, w, h)

    @property
    def page_size(self):
        return PAGE_SIZES[self.page]

    def edges(self, skip=None):
        """``(xs, ys)``: the page's and every other item's edges and
        centres, the targets an item being dragged snaps to."""
        xs, ys = [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]
        for iid, it in self.items.items():
            if iid == skip:
                continue
            x, y, w, h = it['box']
            xs += [x, x + w / 2.0, x + w]
            ys += [y, y + h / 2.0, y + h]
        return xs, ys

    # ── persistence ──────────────────────────────────────────────────────
    def to_dict(self):
        """One page (an entry of a layout file's ``pages``)."""
        return {'name': self.name, 'page': self.page,
                'items': [dict(copy.deepcopy(it), id=iid)
                          for iid, it in self.items.items()]}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, Mapping):
            raise ValueError('not an OpenFlo layout page')
        lay = cls(page=d.get('page', DEFAULT_PAGE),
                  name=d.get('name') or 'Page 1')
        for it in d.get('items') or ():
            if not isinstance(it, Mapping) or it.get('kind') not in ITEM_KINDS:
                continue
            it = dict(it)
            iid = it.pop('id', None)
            lay.add(it, item_id=iid)
        return lay


EXPORT_DPI = 300                 # publication default
EXPORT_DPI_CHOICES = (150, 300, 600, 1200)
ZOOM_CHOICES = ('Fit', '50%', '75%', '100%', '150%', '200%')
SCREEN_DPI = 96                  # what 100% zoom means


# Not part of what an item shows: where it sits, its id and the host's
# labels for it. Everything else -- zoom and axis types included -- is.
_NOT_SHOWN = ('box', 'uid', 'host_from', 'host_name', 'host_ref', 'title')
UID_PREFIX = {'plot': 'P', 'stats': 'S', 'chart': 'C', 'text': 'T'}


def item_signature(it):
    """What an item shows: two items with the same signature are true
    duplicates (same populations, axes, zoom, colours, ...). Changing the
    zoom of a copy makes it a different plot."""
    return json.dumps({k: v for k, v in it.items()
                       if k not in _NOT_SHOWN and v is not None},
                      sort_keys=True, default=str)


class LayoutDoc:
    """A layout file: several pages (as FlowJo's layout editor has), plus
    the export resolution."""

    def __init__(self, pages=None, export_dpi=EXPORT_DPI, shelf=None):
        self.pages: list[Layout] = list(pages) if pages else [Layout()]
        self.export_dpi = int(export_dpi)
        # The cross-page host: items kept beside the pages, to place on any
        # of them (each placement is a copy).
        self.shelf: list[dict] = [copy.deepcopy(dict(it)) for it in shelf or ()
                                  if isinstance(it, Mapping)
                                  and it.get('kind') in ITEM_KINDS]

    def shelve(self, item, origin=None):
        """Keep a copy of ``item`` in the cross-page host (``origin``: the
        page it came from, shown in its name there); returns its index."""
        it = copy.deepcopy(dict(item))
        if origin:
            it['host_from'] = str(origin)
        # One id ties the kept item to the item it came from and every copy
        # placed from it, so the host can point them out on the pages.
        self.shelf.append(it)
        return len(self.shelf) - 1

    def ensure_uids(self):
        """Give every item on the pages a unique, lasting id (P1, S2, C3,
        T4 by kind). A copy -- duplicated, pasted, placed from the host --
        that shares an id with an earlier item gets a new one."""
        seen, top = set(), {}
        for page in self.pages:
            for it in page.items.values():
                u = str(it.get('uid') or '')
                if u[1:].isdigit():
                    top[u[0]] = max(top.get(u[0], 0), int(u[1:]))
        for page in self.pages:
            for it in page.items.values():
                u = it.get('uid')
                if u and u not in seen:
                    seen.add(u)
                    continue
                pre = UID_PREFIX.get(str(it.get('kind')), 'I')
                top[pre] = top.get(pre, 0) + 1
                it['uid'] = f'{pre}{top[pre]}'
                seen.add(it['uid'])

    def matches(self, shelf_idx):
        """``[(page index, item id), ...]`` that are still true duplicates
        of shelf item ``shelf_idx`` (see ``item_signature``): a placed copy
        whose zoom or settings were changed since is not."""
        sig = item_signature(self.shelf[shelf_idx])
        return [(p, iid) for p, page in enumerate(self.pages)
                for iid, other in page.items.items()
                if item_signature(other) == sig]

    def place_from_shelf(self, idx, page_idx, box=None):
        """Put a copy of shelf item ``idx`` on page ``page_idx`` (at ``box``,
        else where it was); returns its new item id."""
        it = copy.deepcopy(self.shelf[idx])
        for k in ('host_from', 'host_name'):      # the host's own labels
            it.pop(k, None)
        if box is not None:
            it['box'] = list(clamp_box(*box))
        return self.pages[page_idx].add(it)

    def unique_name(self, base):
        names = {p.name for p in self.pages}
        if base not in names:
            return base
        for i in itertools.count(2):
            cand = f'{base} {i}'
            if cand not in names:
                return cand
        return base                                       # pragma: no cover

    def add_page(self, page=None):
        page = page or Layout(name='Page')
        page.name = self.unique_name(page.name)
        self.pages.append(page)
        return len(self.pages) - 1

    def remove_page(self, idx):
        """Drop page ``idx``; the last page is kept."""
        if len(self.pages) > 1 and 0 <= idx < len(self.pages):
            self.pages.pop(idx)

    def duplicate_page(self, idx):
        src = self.pages[idx]
        cp = Layout.from_dict(copy.deepcopy(src.to_dict()))
        cp.name = self.unique_name(f'{src.name} copy')
        self.pages.insert(idx + 1, cp)
        return idx + 1

    def copy_item(self, src_idx, iid, dst_idx):
        """Copy one item to another page, at the same place on the page."""
        it = copy.deepcopy(self.pages[src_idx].items[iid])
        return self.pages[dst_idx].add(it)

    def import_pages(self, other, prefix=''):
        """Append another layout's (another project's) pages, renamed so
        they stay distinguishable. Returns the new pages' indices."""
        out = []
        for p in other.pages:
            cp = Layout.from_dict(copy.deepcopy(p.to_dict()))
            cp.name = f'{prefix}{p.name}' if prefix else p.name
            out.append(self.add_page(cp))
        return out

    def to_dict(self):
        return {'format': FORMAT, 'version': VERSION,
                'export_dpi': self.export_dpi,
                'pages': [p.to_dict() for p in self.pages],
                'shelf': copy.deepcopy(self.shelf)}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, Mapping) or d.get('format') != FORMAT:
            raise ValueError('not an OpenFlo layout file')
        if int(d.get('version', 0)) > VERSION:
            raise ValueError(
                f"layout version {d.get('version')} is newer than this "
                f"OpenFlo reads ({VERSION})")
        if 'pages' in d:
            pages = [Layout.from_dict(p) for p in d.get('pages') or ()
                     if isinstance(p, Mapping)]
        else:                                  # version 1: a single page
            pages = [Layout.from_dict(d)]
        try:
            dpi = int(d.get('export_dpi', EXPORT_DPI))
        except (TypeError, ValueError):
            dpi = EXPORT_DPI
        return cls(pages or None, export_dpi=dpi, shelf=d.get('shelf'))

    def dumps(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def loads(cls, text):
        return cls.from_dict(json.loads(text))


# ── soft snapping ────────────────────────────────────────────────────────

def _nearest(values, targets, tol):
    """``(shift, target)`` moving the closest of ``values`` onto a target
    within ``tol``, or ``(0.0, None)``."""
    best = (tol + 1e-12, 0.0, None)
    for v in values:
        for t in targets:
            d = abs(t - v)
            if d < best[0]:
                best = (d, t - v, t)
    return best[1], best[2]


def grid_lines(step):
    if not step or step <= 0:
        return []
    n = int(round(1.0 / step))
    return [i * step for i in range(n + 1)]


def soft_snap(box, mode, targets, tol):
    """Pull a box being moved (``mode='move'``: either edge or its centre)
    or resized (``'resize'``: the right/bottom edge) onto the nearest target
    within ``tol`` (page fraction), separately in x and y. ``targets`` is
    ``(xs, ys)``. Returns ``(box, (gx, gy))``: the snapped box and the guide
    lines it snapped to (None when free), for the editor to draw."""
    x, y, w, h = box
    xs, ys = targets
    if mode == 'move':
        dx, gx = _nearest((x, x + w / 2.0, x + w), xs, tol)
        dy, gy = _nearest((y, y + h / 2.0, y + h), ys, tol)
        return clamp_box(x + dx, y + dy, w, h), (gx, gy)
    dx, gx = _nearest((x + w,), [v for v in xs if v > x + MIN_SIZE], tol)
    dy, gy = _nearest((y + h,), [v for v in ys if v > y + MIN_SIZE], tol)
    return clamp_box(x, y, w + dx, h + dy), (gx, gy)


# ── populations ──────────────────────────────────────────────────────────

def population_choices(gates, order, channel_labels, transforms=None):
    """``[(key, display_path), ...]`` for one sample's populations, the
    ungated sample first (key ``ALL_EVENTS``)."""
    from .gating import population_keys, population_path
    keys = population_keys(gates, channel_labels)
    out = [(ALL_EVENTS, 'All events')]
    for gid in order or list(gates):
        if gid in gates:
            out.append((keys[gid],
                        population_path(gates, gid, transforms=transforms)))
    return out


def resolve_population(gates, channel_labels, key):
    """The gate id carrying population ``key`` in this sample's ``gates``;
    ``None`` for ``ALL_EVENTS``. Raises ``KeyError`` when the sample has no
    such population (a layout made on another panel or tree)."""
    if key is ALL_EVENTS or key == '':
        return None
    from .gating import population_keys
    for gid, k in population_keys(gates, channel_labels).items():
        if k == key:
            return gid
    raise KeyError(key)


def child_gates(gates, gid):
    """Gates drawn on a plot of population ``gid`` (None = ungated): its
    direct children, as FlowJo shows the gates made on that plot."""
    return {k: g for k, g in gates.items() if g.get('parent_id') == gid
            or (gid is None and not g.get('parent_id'))}


# Gates a strategy plot can show: drawn on two axes or one.
_STRATEGY_KINDS = ('rect', 'polygon', 'ellipsoid', 'quadrant', 'threshold',
                   'interval', 'flowjo_ellipse')


def gating_strategy_steps(gates, order=None, gid=None):
    """``[(parent id, x, y, [gate ids])]``: one plotting step per parent
    population and pair of axes, in tree order. With ``gid``, only the
    steps from the root down to that gate (its own step included);
    otherwise every gate drawn on axes. Sibling gates on the same axes
    (quadrants, a positive and a negative cut) share one step."""
    ids = [g for g in (order or gates) if g in gates]
    ids += [g for g in gates if g not in ids]
    if gid is not None:
        chain, seen, cur = [], set(), gid
        while cur in gates and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = gates[cur].get('parent_id')
        wanted = set(chain)
        ids = [g for g in ids if g in wanted]
    steps: dict = {}
    for g_id in ids:
        g = gates[g_id]
        if g.get('kind') not in _STRATEGY_KINDS:
            continue
        if g.get('channel'):
            x, y = g['channel'], None
        elif g.get('x_channel'):
            x, y = g['x_channel'], g.get('y_channel')
        else:
            continue
        key = (g.get('parent_id') if g.get('parent_id') in gates else None,
               x, y)
        steps.setdefault(key, []).append(g_id)
    return [(pid, x, y, gids) for (pid, x, y), gids in steps.items()]


def gating_strategy_page(gates, keys, order=None, gid=None,
                         name='Gating strategy', page=DEFAULT_PAGE,
                         mode='pseudocolor', margin=0.02):
    """A page laying out a gating strategy: a title line, then one plot per
    step of ``gating_strategy_steps`` in a grid, each the parent population
    on the gate's axes with its gates drawn, titled with the gates' names.
    ``keys`` is ``gating.population_keys`` of the same ``gates``. Batch
    export then draws it for every sample."""
    steps = gating_strategy_steps(gates, order, gid)
    lay = Layout(page=page, name=name)
    lay.add(text_item('{sample}: gating strategy', 12.0,
                      box=(margin, margin, 1.0 - 2 * margin, 0.06)))
    if not steps:
        return lay
    n = len(steps)
    w_in, h_in = PAGE_SIZES[lay.page]
    cols = max(1, min(n, round((n * w_in / h_in) ** 0.5 + 0.49)))
    rows = -(-n // cols)
    top = margin + 0.06 + margin
    cw = (1.0 - margin) / cols
    rh = (1.0 - top) / rows
    for i, (pid, x, y, gids) in enumerate(steps):
        r, c = divmod(i, cols)
        title = ', '.join(str(gates[g].get('name') or keys.get(g, g)
                              ).rsplit('/', 1)[-1] for g in gids)
        lay.add(plot_item(
            population=keys.get(pid, ALL_EVENTS) if pid else ALL_EVENTS,
            x=x, y=y, mode='histogram' if y is None else mode,
            draw_gates=True, title=title,
            box=(margin + c * cw, top + r * rh, cw - margin, rh - margin)))
    return lay


def fill_text(template, sample):
    """Substitute ``{sample}`` in a text item; other braces stay literal."""
    return str(template).replace('{sample}', str(sample or ''))


def safe_filename(name, maxlen=80):
    keep = ''.join(c if (c.isalnum() or c in '-_.') else '_'
                   for c in str(name))
    return (keep.strip('._') or 'sample')[:maxlen]
