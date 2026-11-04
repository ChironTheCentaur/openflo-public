"""Characterization of the canvas gesture handlers: _hit_test, _on_press and
_on_motion (editor_gating.GatingMixin).

These three methods were split into per-stage / per-drag-kind helpers. This
file pins what they do, so the split is provably behaviour-preserving:

* first-principles checks -- a click on a handle grabs that handle, a drag
  puts the handle under the cursor, a translate moves by the cursor delta,
  the 2.5%-of-view tolerance is honoured on both sides, the priority order
  (1-D lines > quadrant cross > rect handles > polygon handles > ellipse
  handles > interiors) holds where handles overlap;
* a golden sweep -- synthetic gates of every kind (1-D threshold / interval,
  a 4-rect quadrant set, rects, polygons including points added along a bent
  edge, ellipsoids, plus gates that must NOT be hit) on a linear and a log
  view, a grid of synthetic clicks, and for a subset of them a press with each
  button / modifier followed by motions. What _hit_test returns, what the press
  sets (drag state, translate context, gate dicts, undo step, cursor) and what
  each motion does to the gates, together with the order of the editor / Tk /
  matplotlib calls made, is compared with a record taken from the code before
  the split (tests/golden/editor_gating_dispatch.json). Floats are
  compared to 10 significant digits so the record does not depend on the last
  bit of a platform's floating point.

Regenerate the record (only from code known to be right) with
OPENFLO_WRITE_GATING_GOLDEN=1.

All data is generated: channels FL1-A / FL2-A / FL3-A of a sample_a.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tests.conftest import editor_or_skip

X, Y, Z = 'FL1-A', 'FL2-A', 'FL3-A'
GOLDEN = Path(__file__).parent / 'golden' / 'editor_gating_dispatch.json'
SCALES = ('linear', 'log')
LIMITS = {'linear': ((-2.0e3, 1.02e5), (-2.0e3, 1.02e5)),
          'log': ((5.0, 2.0e5), (5.0, 2.0e5))}
TOL = 0.025            # _hit_test's tolerance, a fraction of the view


# ── Editor ──────────────────────────────────────────────────────────────────

def _editor_or_skip():
    return editor_or_skip()


def _setup(ed):
    rng = np.random.default_rng(7)
    n = 1500
    df = pd.DataFrame({X: 10 ** rng.uniform(1, 5, n),
                       Y: 10 ** rng.uniform(1, 5, n),
                       Z: 10 ** rng.uniform(1, 5, n)})
    cols = list(df.columns)
    ed._samples['sample_a'] = SimpleNamespace(
        name='sample_a', path='sample_a.fcs', data=df, fluor_channels=[Z],
        channel_labels={c: c for c in cols})
    ed._sample_order.append('sample_a')
    ed._sample_colors['sample_a'] = '#1f77b4'
    ed._sample_trial['sample_a'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['sample_a'] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()
    ed._set_active_sample('sample_a')
    ed.x_combo.set(X)
    ed.y_combo.set(Y)
    ed.mode_var.set('dot')


def _set_scale(ed, scale):
    """Replot on `scale`, then pin the figure, the axes box and the limits,
    so every transform is the same from run to run."""
    ed._gates.clear()
    ed._gate_id_order[:] = []
    ed._channel_scale[X] = scale
    ed._channel_scale[Y] = scale
    ed._replot()
    fig = ed.fig
    try:
        fig.set_layout_engine('none')
    except Exception:                           # noqa: BLE001 - best effort; if matplotlib refuses the swap, size and dpi are still pinned below
        pass
    fig.set_dpi(100)
    fig.set_size_inches(6.4, 5.0, forward=False)
    ed.ax.set_position([0.125, 0.11, 0.775, 0.77])
    (xl, xh), (yl, yh) = LIMITS[scale]
    ed.ax.set_xlim(xl, xh)
    ed.ax.set_ylim(yl, yh)
    assert ed.ax.get_xscale() == scale


def _d(ed, fx, fy):
    """Data point at axes fraction (fx, fy)."""
    tr = ed.ax.transAxes + ed.ax.transData.inverted()
    x, y = tr.transform((fx, fy))
    return float(x), float(y)


def _f(ed, x, y):
    """Axes fraction of data point (x, y)."""
    tr = ed.ax.transData + ed.ax.transAxes.inverted()
    fx, fy = tr.transform((x, y))
    return float(fx), float(fy)


# ── Scenes: synthetic gates of every kind, placed by axes fraction ─────────

def _rect(ed, f0, f1, **kw):
    (x0, y0), (x1, y1) = _d(ed, *f0), _d(ed, *f1)
    g = {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
         'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}
    g.update(kw)
    return g


def _poly(ed, fverts):
    return ed._drawn_polygon(X, Y, [_d(ed, *p) for p in fverts])


def _ellipse(ed, fc, fa, fb, rho=0.0):
    """Ellipsoid centred at fraction fc; semi-axes reach fractions fa / fb
    along x / y (in data, so lopsided on a log view)."""
    cx, cy = _d(ed, *fc)
    a = abs(_d(ed, fc[0] + fa, fc[1])[0] - cx)
    b = abs(_d(ed, fc[0], fc[1] + fb)[1] - cy)
    return {'kind': 'ellipsoid', 'x_channel': X, 'y_channel': Y,
            'mean': [cx, cy],
            'cov': [[a * a, rho * a * b], [rho * a * b, b * b]],
            'distance_sq': 1.0}


def _quad_set(ed, fo, qs='qs1'):
    from openflo.scales import OPEN
    xc, yc = _d(ed, *fo)
    out = []
    for label, x0, x1, y0, y1 in [
            ('Q++ (x>, y>)', xc, OPEN, yc, OPEN),
            ('Q+- (x>, y<)', xc, OPEN, -OPEN, yc),
            ('Q-+ (x<, y>)', -OPEN, xc, yc, OPEN),
            ('Q-- (x<, y<)', -OPEN, xc, -OPEN, yc)]:
        out.append({'kind': 'rect', 'x_channel': X, 'y_channel': Y,
                    'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'label': label,
                    'quad_set': qs, 'quad_origin_x': xc,
                    'quad_origin_y': yc})
    return out


def _scene_lines(ed):
    return [
        {'kind': 'interval', 'channel': X,
         'lo': _d(ed, 0.30, 0.5)[0], 'hi': _d(ed, 0.70, 0.5)[0]},
        {'kind': 'threshold', 'channel': Y, 'value': _d(ed, 0.5, 0.60)[1]},
        {'kind': 'threshold', 'channel': Z, 'value': 100.0},   # not drawn
        _rect(ed, (0.20, 0.20), (0.50, 0.80)),
        _ellipse(ed, (0.78, 0.30), 0.10, 0.08),
    ]


def _scene_quad(ed):
    return _quad_set(ed, (0.45, 0.55)) + [
        _rect(ed, (0.60, 0.10), (0.90, 0.35)),
        _poly(ed, [(0.47, 0.70), (0.80, 0.68), (0.62, 0.92)]),
        _ellipse(ed, (0.20, 0.25), 0.10, 0.12, rho=0.4),
    ]


def _scene_shapes(ed):
    return [
        _rect(ed, (0.08, 0.08), (0.40, 0.45)),
        _poly(ed, [(0.50, 0.50), (0.90, 0.52), (0.92, 0.85), (0.70, 0.95),
                   (0.52, 0.80)]),
        # Overlaps the pentagon; its first vertex is 0.012 off a corner.
        _poly(ed, [(0.512, 0.508), (0.60, 0.30), (0.95, 0.20)]),
        # A long hypotenuse: on the log view points are added along it.
        _poly(ed, [(0.05, 0.55), (0.40, 0.55), (0.05, 0.95)]),
        _ellipse(ed, (0.30, 0.30), 0.14, 0.09, rho=-0.5),
        {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,       # < 3 verts
         'vertices': [list(_d(ed, 0.6, 0.6)), list(_d(ed, 0.7, 0.7))]},
        _rect(ed, (0.10, 0.10), (0.90, 0.90), x_channel=Z),      # off view
        {'kind': 'ellipsoid', 'x_channel': X, 'y_channel': Y,     # singular
         'mean': list(_d(ed, 0.45, 0.65)), 'cov': [[0.0, 0.0], [0.0, 0.0]],
         'distance_sq': 1.0},
    ]


def _scene_upper(ed):
    """Clear of the low end, where on a log view a rect's data-unit
    tolerance reaches across the plot."""
    return [
        _rect(ed, (0.80, 0.80), (0.97, 0.97)),
        _ellipse(ed, (0.62, 0.88), 0.08, 0.06, rho=0.3),
        _poly(ed, [(0.82, 0.30), (0.97, 0.35), (0.90, 0.60)]),
    ]


SCENES = {'lines': _scene_lines, 'quad': _scene_quad, 'shapes': _scene_shapes,
          'upper': _scene_upper}


def _build_scene(ed, name):
    gates = {}
    for i, g in enumerate(SCENES[name](ed), start=1):
        g = dict(g)
        g.setdefault('parent_id', None)
        g.setdefault('color', '#d62728')
        g.setdefault('enabled', True)
        gates[f'g{i}'] = g
    return gates


def _install(ed, scene, scale):
    """Reset every piece of gesture state, then put `scene` back."""
    ed._drag_state = None
    ed._drag_translate_ctx = None
    ed._drag_poly_vertices = None
    ed._end_drag_blit()
    ed._drag_blit = False
    ed._drag_bg = None
    ed._zoom_start = None
    ed._pan_start = None
    ed._legend_drag = None
    ed._zoom_mode = False
    if ed._zoom_rect_artist is not None:
        try:
            ed._zoom_rect_artist.remove()
        except Exception:                       # noqa: BLE001 - the zoom rectangle may already be off the axes; it is dropped either way
            pass
        ed._zoom_rect_artist = None
    ed._undo_stack.clear()
    ed._redo_stack.clear()
    ed._undo_pending = False
    ed._quad_set_seq = 50
    ed._gate_id_seq = 100
    ed._sample_gate_seq[ed._active_sample] = 100
    ed._backgate_legend_anchor = (0.020, 0.965)
    ed._gates.clear()
    ed._gates.update(copy.deepcopy(scene))
    ed._gate_id_order[:] = list(scene)
    (xl, xh), (yl, yh) = LIMITS[scale]
    ed.ax.set_xlim(xl, xh)
    ed.ax.set_ylim(yl, yh)
    ed._redraw_only_gates()


# ── Events, normalisation, call log ─────────────────────────────────────────

def _ev(ed, fx, fy, button=1, key=None, dblclick=False, inaxes=True,
        drop=None):
    """A synthetic matplotlib mouse event at axes fraction (fx, fy).
    ``drop`` = 'x' / 'y' / 'xy' blanks that data coordinate."""
    x, y = _d(ed, fx, fy)
    px, py = ed.ax.transData.transform((x, y))
    drop = drop or ''
    return SimpleNamespace(
        button=button, key=key, dblclick=dblclick,
        inaxes=ed.ax if inaxes else None,
        xdata=None if ('x' in drop or not inaxes) else x,
        ydata=None if ('y' in drop or not inaxes) else y,
        x=float(px), y=float(py))


def _norm(v, digits):
    if isinstance(v, dict):
        return {str(k): _norm(v[k], digits) for k in sorted(v, key=str)}
    if isinstance(v, (list, tuple)):
        return [_norm(x, digits) for x in v]
    if isinstance(v, np.ndarray):
        return _norm(v.tolist(), digits)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not np.isfinite(v):
            return repr(v)
        return float(f'{v:.{digits}g}') if digits else v
    if v is None or isinstance(v, str):
        return v
    return repr(type(v).__name__)


SPIED = ('_hit_test', '_legend_press', '_checkpoint', '_redraw_only_gates',
         '_blit_drag_frame', '_update_quad_set', '_insert_polygon_vertex',
         '_delete_polygon_vertex', '_add_gate_multi', '_schedule_replot',
         '_refresh_gate_list', '_begin_poly_vertex_drag', '_move_poly_vertex',
         '_polygon_under_point', '_is_point_along_edge', '_gid_from_hit',
         '_reposition_backgate_legend', '_ellipse_rim_distance',
         '_ellipse_geom', '_selected_gate_id')


class _TkSpy:
    """Stands in for the canvas's Tk widget and logs what is asked of it."""

    def __init__(self, real, log):
        self._real, self._log = real, log

    def config(self, *a, **k):
        self._log.append(['tk.config', _norm(k, 0)])
        return self._real.config(*a, **k)

    def focus_set(self):
        self._log.append(['tk.focus_set'])
        return self._real.focus_set()

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_spies(ed):
    """Log, in order, each editor helper the handlers call and each call on
    the Tk widget / the axes / the canvas they make. Returns the log."""
    log: list = []

    def wrap(name, fn):
        def spy(*a, **k):
            log.append([name])
            return fn(*a, **k)
        return spy

    for name in SPIED:
        setattr(ed, name, wrap(name, getattr(ed, name)))
    tk_widget = ed.canvas.get_tk_widget()
    ed.canvas.get_tk_widget = lambda: _TkSpy(tk_widget, log)
    for obj, label, names in ((ed.ax, 'ax', ('set_xlim', 'set_ylim',
                                             'add_patch')),
                              (ed.canvas, 'canvas', ('draw_idle', 'draw',
                                                     'blit'))):
        for name in names:
            setattr(obj, name, wrap(f'{label}.{name}', getattr(obj, name)))
    return log


def _snapshot(ed, scene, ret, log, digits):
    gates = ed._gates
    delta = {gid: g for gid, g in gates.items() if scene.get(gid) != g}
    zr = ed._zoom_rect_artist
    return _norm({
        'ret': ret,
        'drag': ed._drag_state,
        'ctx': ed._drag_translate_ctx,
        'base': ed._drag_poly_vertices,
        'blit': bool(getattr(ed, '_drag_blit', False)),
        'changed': delta,
        'removed': sorted(set(scene) - set(gates)),
        'order': list(ed._gate_id_order),
        'undo': len(ed._undo_stack),
        'zoom_start': ed._zoom_start,
        'pan_start': ed._pan_start,
        'qs_seq': ed._quad_set_seq,
        'xlim': list(ed.ax.get_xlim()),
        'ylim': list(ed.ax.get_ylim()),
        'vlines': {k: list(ln.get_xdata()) for k, ln in ed._vlines.items()},
        'hlines': {k: list(ln.get_ydata()) for k, ln in ed._hlines.items()},
        'zoom_rect': (None if zr is None
                      else [zr.get_x(), zr.get_y(), zr.get_width(),
                            zr.get_height()]),
        'legend': ed._backgate_legend_anchor,
        'calls': list(log),
    }, digits)


# ── Sweep ───────────────────────────────────────────────────────────────────

def _evenly(seq, cap):
    """At most ``cap`` items of `seq`, evenly spread (all when None)."""
    if cap is None or len(seq) <= cap:
        return list(seq)
    return [seq[int(i)] for i in np.linspace(0, len(seq) - 1, cap)]


def _handles(ed, scene, cap=None):
    """Axes fractions of the handles in `scene` (of a polygon, ``cap``
    vertices and ``cap`` edge midpoints at most)."""
    pts = []
    for g in scene.values():
        k = g.get('kind')
        if k == 'rect' and not g.get('quad_set') and g['x_channel'] == X:
            fx0, fy0 = _f(ed, g['x0'], g['y0'])
            fx1, fy1 = _f(ed, g['x1'], g['y1'])
            pts += [(fx0, fy0), (fx1, fy0), (fx0, fy1), (fx1, fy1),
                    ((fx0 + fx1) / 2, fy0), ((fx0 + fx1) / 2, fy1),
                    (fx0, (fy0 + fy1) / 2), (fx1, (fy0 + fy1) / 2)]
        elif g.get('quad_set'):
            fo = _f(ed, g['quad_origin_x'], g['quad_origin_y'])
            pts += [fo, (fo[0], 0.2), (0.85, fo[1])]
        elif k == 'polygon' and len(g['vertices']) >= 3:
            fv = [_f(ed, *v) for v in g['vertices']]
            pts += _evenly(fv, cap)
            pts += _evenly([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                            for a, b in zip(fv, fv[1:] + fv[:1], strict=True)],
                           cap)
        elif k == 'ellipsoid':
            geom = ed._ellipse_geom(g)
            if geom is not None:
                (mx, my), _inv, _r0, (hx, hy) = geom
                pts += [_f(ed, mx, my), _f(ed, hx, hy)]
                ev, evec = np.linalg.eigh(np.asarray(g['cov'], dtype=float))
                for s in (1.0, -1.0):
                    for j in (0, 1):
                        r = s * np.sqrt(ev[j]) * evec[:, j]
                        pts.append(_f(ed, mx + r[0], my + r[1]))
        elif k == 'threshold' and g['channel'] in (X, Y):
            v = float(g['value'])
            pts.append((_f(ed, v, 1.0)[0], 0.4) if g['channel'] == X
                       else (0.4, _f(ed, 1.0, v)[1]))
        elif k == 'interval' and g['channel'] in (X, Y):
            for v in (float(g['lo']), float(g['hi'])):
                pts.append((_f(ed, v, 1.0)[0], 0.4) if g['channel'] == X
                           else (0.4, _f(ed, 1.0, v)[1]))
    return [p for p in pts if all(np.isfinite(p))]


OFFSETS = ((0.0, 0.0), (0.012, 0.0), (0.0, -0.012), (0.024, 0.024),
           (-0.026, 0.0), (0.0, 0.026), (0.018, -0.02))
PRESS_OFFSETS = ((0.0, 0.0), (0.012, -0.012), (-0.026, 0.0))
MOTIONS = ((0.06, -0.035, None), (-0.13, 0.09, None), (0.02, 0.03, 'x'),
           (0.0, 0.0, 'xy'))


def _press_motion_case(ed, scene, scale, log, digits, fx, fy, mode):
    _install(ed, scene, scale)
    kw = {'l': {}, 'ls': {'key': 'shift'}, 'la': {'key': 'alt'},
          'lca': {'key': 'ctrl+alt'}, 'r': {'button': 3},
          'ld': {'dblclick': True}, 'm': {'button': 2},
          'z': {}, 'out': {'inaxes': False}}[mode]
    if mode == 'z':
        ed._zoom_mode = True
    del log[:]
    ed._on_press(_ev(ed, fx, fy, **kw))
    steps = [_snapshot(ed, scene, None, log, digits)]
    for dx, dy, drop in MOTIONS:
        del log[:]
        ev = _ev(ed, fx + dx, fy + dy, inaxes=(drop != 'xy'), drop=drop)
        ed._on_motion(ev)
        steps.append(_snapshot(ed, scene, None, log, digits))
    return steps


def _sweep(ed, log, scale, scene_name, *, grid, press_grid, n_random=0,
           seed=0, digits=10, cap=None, digest=False):
    """Hit-test every point of a ``grid`` x ``grid`` lattice, each handle
    (with offsets around the 2.5% tolerance) and ``n_random`` random points;
    press + move from each handle and a ``press_grid`` lattice. With
    ``digest`` each press case is kept as a hash of its record."""
    _set_scale(ed, scale)
    scene = _build_scene(ed, scene_name)
    _install(ed, scene, scale)
    handles = _handles(ed, scene, cap)
    pts = [(i / (grid - 1), j / (grid - 1))
           for i in range(grid) for j in range(grid)]
    pts += [(hx + ox, hy + oy) for hx, hy in handles for ox, oy in OFFSETS]
    rng = np.random.default_rng(seed)
    pts += [tuple(p) for p in rng.uniform(-0.02, 1.02, (n_random, 2))]
    hits = []
    for fx, fy in pts:
        hits.append([_norm(fx, digits), _norm(fy, digits),
                     _norm(ed._hit_test(_ev(ed, fx, fy)), digits)])
    hits.append(['out', _norm(ed._hit_test(_ev(ed, 0.5, 0.5, inaxes=False)),
                              digits)])
    hits.append(['nox', _norm(ed._hit_test(_ev(ed, 0.5, 0.5, drop='x')),
                              digits)])

    cases = []
    for hx, hy in handles:
        for ox, oy in PRESS_OFFSETS:
            for mode in (('l', 'ls', 'r', 'la') if (ox, oy) == (0.0, 0.0)
                         else ('l',)):
                cases.append((hx + ox, hy + oy, mode))
    lattice = [(0.04 + 0.92 * i / (press_grid - 1),
                0.04 + 0.92 * j / (press_grid - 1))
               for i in range(press_grid) for j in range(press_grid)]
    for k, (fx, fy) in enumerate(lattice):
        cases.append((fx, fy, 'l'))
        cases.append((fx, fy, ('ld', 'r', 'la', 'm', 'z', 'lca', 'out')[k % 7]))
    for fx, fy in rng.uniform(0.0, 1.0, (n_random // 4, 2)):
        cases.append((float(fx), float(fy),
                      ('l', 'ls', 'r', 'la', 'ld', 'm', 'z')[rng.integers(7)]))
    presses = []
    for fx, fy, mode in cases:
        rec = _press_motion_case(ed, scene, scale, log, digits, fx, fy, mode)
        if digest:
            rec = _digest(rec)
        presses.append([_norm(fx, digits), _norm(fy, digits), mode, rec])
    legend = _legend_drag_case(ed, scene, scale, log, digits)
    if digest:
        legend = _digest(legend)
    _install(ed, scene, scale)
    return {'scene': _norm(scene, digits), 'hits': hits, 'presses': presses,
            'legend': legend}


def _legend_drag_case(ed, scene, scale, log, digits):
    _install(ed, scene, scale)
    px, py = ed.ax.transAxes.transform((0.05, 0.9))
    ed._legend_drag = {'sx': float(px), 'sy': float(py),
                       'anchor': (0.020, 0.965)}
    out = []
    for fx, fy in ((0.25, 0.7), (0.99, 0.01), (-0.5, 1.5)):
        del log[:]
        ed._on_motion(_ev(ed, fx, fy))
        out.append(_snapshot(ed, scene, None, log, digits))
    ed._legend_drag = None
    return out


def _digest(rec):
    import hashlib
    blob = json.dumps(rec, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def collect(ed, log, *, grid=21, press_grid=4, n_random=0, seed=0,
            digits=10, cap=6, digest=True):
    return {f'{scale}/{name}': _sweep(ed, log, scale, name, grid=grid,
                                      press_grid=press_grid,
                                      n_random=n_random, seed=seed,
                                      digits=digits, cap=cap, digest=digest)
            for scale in SCALES for name in SCENES}


def _first_diff(a, b, path='$'):
    if type(a) is not type(b):
        return f'{path}: {a!r} != {b!r}'
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f'{path}.{k}: present in only one'
            d = _first_diff(a[k], b[k], f'{path}.{k}')
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f'{path}: length {len(a)} != {len(b)}'
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = _first_diff(x, y, f'{path}[{i}]')
            if d:
                return d
        return None
    return None if a == b else f'{path}: {a!r} != {b!r}'


# ── Tests ───────────────────────────────────────────────────────────────────

def test_sweep_matches_the_record_taken_before_the_split():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        log = _install_spies(ed)
        got = collect(ed, log)
        if os.environ.get('OPENFLO_WRITE_GATING_GOLDEN'):
            GOLDEN.parent.mkdir(exist_ok=True)
            GOLDEN.write_text(json.dumps(got, separators=(',', ':'))
                              + '\n')
        want = json.loads(GOLDEN.read_text())
        # The sweep reached every drag kind, on both views.
        for scale in SCALES:
            kinds = {h[2][0] for k, v in got.items() if k.startswith(scale)
                     for h in v['hits'] if len(h) == 3 and h[2]}
            assert kinds == {'v', 'h', 'quad_origin', 'quad_x', 'quad_y',
                             'rect_corner', 'rect_edge', 'poly_vertex',
                             'poly_edge', 'ellipse_rotate', 'ellipse_rim',
                             'rect_translate', 'poly_translate',
                             'ellipse_translate'}, (scale, kinds)
        diff = _first_diff(json.loads(json.dumps(got)), want)
        assert diff is None, diff
    finally:
        root.destroy()


def _fresh(ed, scale, scene_name):
    _set_scale(ed, scale)
    scene = _build_scene(ed, scene_name)
    _install(ed, scene, scale)
    return scene


def test_hit_test_grabs_each_handle_within_tolerance_and_in_priority_order():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        hit = lambda fx, fy: ed._hit_test(_ev(ed, fx, fy))  # noqa: E731
        scene = _fresh(ed, 'linear', 'lines')
        # 1-D lines win over everything, within 2.5% of the view.
        assert hit(0.30, 0.40) == ('v', 'g1:lo')
        assert hit(0.70 + 0.024, 0.10) == ('v', 'g1:hi')
        assert hit(0.70 + 0.026, 0.10) is None
        assert hit(0.95, 0.60 - 0.024) == ('h', 'g2')
        # The rect's corner at (0.2, 0.2), its bottom edge, its inside.
        assert hit(0.205, 0.195) == ('rect_corner', 'g4', 'bl')
        assert hit(0.24, 0.20) == ('rect_edge', 'g4', 'bottom')
        assert hit(0.24, 0.40) == ('rect_translate', 'g4')
        # The ellipse: inside translates, the grip rotates, the rim resizes.
        (mx, my), _i, _r, (hx, hy) = ed._ellipse_geom(scene['g5'])
        assert hit(*_f(ed, mx, my)) == ('ellipse_translate', 'g5')
        assert hit(*_f(ed, hx, hy)) == ('ellipse_rotate', 'g5')
        assert hit(0.78, 0.30 + 0.08) == ('ellipse_rim', 'g5')
        # Off the axes, or without data coordinates: nothing.
        assert ed._hit_test(_ev(ed, 0.3, 0.4, inaxes=False)) is None
        assert ed._hit_test(_ev(ed, 0.3, 0.4, drop='y')) is None

        scene = _fresh(ed, 'linear', 'quad')
        assert hit(0.45, 0.55) == ('quad_origin', 'qs1')
        assert hit(0.45 + 0.02, 0.10) == ('quad_x', 'qs1')
        assert hit(0.95, 0.55 - 0.02) == ('quad_y', 'qs1')
        # A polygon corner 0.02 off the cross still loses to it.
        assert hit(0.47, 0.70) == ('quad_x', 'qs1')
        assert hit(0.62, 0.92) == ('poly_vertex', 'g6', 2)
        assert hit(0.75, 0.20) == ('rect_translate', 'g5')
        # Quadrant insides are not translatable.
        assert hit(0.30, 0.90) is None

        _fresh(ed, 'linear', 'shapes')
        # Overlapping polygons: the NEAREST vertex wins.
        assert hit(0.511, 0.507) == ('poly_vertex', 'g3', 0)
        assert hit(0.501, 0.501) == ('poly_vertex', 'g2', 0)
        # Undrawn gates (< 3 vertices, other channels, a singular
        # ellipsoid) are never hit.
        assert hit(0.65, 0.65) == ('poly_translate', 'g2')
        assert hit(0.45, 0.65) is None
        assert hit(0.85, 0.15) is None

        # On the log view rect handles keep their DATA-unit tolerance
        # (2.5% of the data span, ~5000 here), so a corner near the low
        # end reaches far across the plot; polygon and ellipse handles are
        # measured on screen.
        _fresh(ed, 'log', 'shapes')
        assert ed.ax.get_xlim()[1] * TOL > 4999
        assert hit(0.50, 0.50) == ('rect_corner', 'g1', 'bl')
        scene = _fresh(ed, 'log', 'upper')
        assert hit(0.80, 0.97) == ('rect_corner', 'g1', 'tl')
        assert hit(0.88, 0.97) == ('rect_edge', 'g1', 'top')
        assert hit(0.88, 0.88) == ('rect_translate', 'g1')
        (mx, my), _i, _r, (hx, hy) = ed._ellipse_geom(scene['g2'])
        assert hit(*_f(ed, hx, hy)) == ('ellipse_rotate', 'g2')
        assert hit(*_f(ed, mx, my)) == ('ellipse_translate', 'g2')
        assert hit(0.62, 0.88 + 0.06) == ('ellipse_rim', 'g2')
        assert hit(0.97, 0.35) == ('poly_vertex', 'g3', 1) or \
            hit(0.97, 0.35)[0] == 'poly_vertex'
    finally:
        root.destroy()


def test_long_edge_on_log_view_keeps_points_along_it_out_of_corner_hits():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        scene = _fresh(ed, 'log', 'shapes')
        verts = scene['g4']['vertices']
        assert len(verts) > 3          # points were added along the hypotenuse
        along = [i for i in range(len(verts))
                 if ed._is_point_along_edge('g4', i)]
        assert along
        i = along[len(along) // 2]
        fx, fy = _f(ed, *verts[i])
        assert ed._hit_test(_ev(ed, fx, fy)) == ('poly_vertex', 'g4', i)
        # A right-click there adds a vertex (it is on an edge), not a delete.
        n = len(verts)
        ed._on_press(_ev(ed, fx, fy, button=3))
        assert len(ed._gates['g4']['vertices']) >= n
        assert ed._drag_state is None
    finally:
        root.destroy()


def test_press_then_motion_puts_the_handle_under_the_cursor():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        for scale in ('linear',):
            scene = _fresh(ed, scale, 'lines')
            ed._on_press(_ev(ed, 0.20, 0.20))
            assert ed._drag_state == ('rect_corner', 'g4', 'bl')
            assert len(ed._undo_stack) == 1
            ed._on_motion(_ev(ed, 0.26, 0.16))
            g = ed._gates['g4']
            assert (g['x0'], g['y0']) == _d(ed, 0.26, 0.16)
            assert (g['x1'], g['y1']) == (scene['g4']['x1'], scene['g4']['y1'])
            # Past the opposite corner: the bounds stay sorted.
            ed._on_motion(_ev(ed, 0.60, 0.90))
            g = ed._gates['g4']
            assert g['x0'] < g['x1'] and g['y0'] < g['y1']
            assert (g['x1'], g['y1']) == _d(ed, 0.60, 0.90)

            # Interval: drag lo past hi, and they swap.
            _install(ed, scene, scale)
            ed._on_press(_ev(ed, 0.30, 0.95))
            assert ed._drag_state == ('v', 'g1:lo')
            ed._on_motion(_ev(ed, 0.85, 0.95))
            g = ed._gates['g1']
            assert g['lo'] == scene['g1']['hi']
            assert g['hi'] == _d(ed, 0.85, 0.95)[0]

            # Translate: by the cursor delta, from the geometry at press.
            _install(ed, scene, scale)
            ed._on_press(_ev(ed, 0.40, 0.40))
            assert ed._drag_state == ('rect_translate', 'g4')
            ax, ay = _d(ed, 0.40, 0.40)
            bx, by = _d(ed, 0.43, 0.45)
            ed._on_motion(_ev(ed, 0.43, 0.45))
            g = ed._gates['g4']
            assert g['x0'] == float(scene['g4']['x0']) + (bx - ax)
            assert g['y1'] == float(scene['g4']['y1']) + (by - ay)

            # Shift forces translate even on a corner.
            _install(ed, scene, scale)
            ed._on_press(_ev(ed, 0.20, 0.20, key='shift'))
            assert ed._drag_state == ('rect_translate', 'g4')

            # A motion with no data coordinates leaves the gate alone.
            _install(ed, scene, scale)
            ed._on_press(_ev(ed, 0.20, 0.20))
            ed._on_motion(_ev(ed, 0.3, 0.3, drop='xy'))
            assert ed._gates['g4'] == scene['g4']

            scene = _fresh(ed, scale, 'quad')
            ed._on_press(_ev(ed, 0.45, 0.10))
            assert ed._drag_state == ('quad_x', 'qs1')
            ed._on_motion(_ev(ed, 0.40, 0.30))
            nx = _d(ed, 0.40, 0.30)[0]
            ox, oy = scene['g1']['quad_origin_x'], scene['g1']['quad_origin_y']
            for gid in ('g1', 'g2', 'g3', 'g4'):
                g = ed._gates[gid]
                assert (g['quad_origin_x'], g['quad_origin_y']) == (nx, oy)
            assert ed._gates['g1']['x0'] == nx and ed._gates['g3']['x1'] == nx
            assert ox != nx

            # Ellipse rim: the rim passes through the cursor afterwards.
            scene = _fresh(ed, scale, 'lines')
            ed._on_press(_ev(ed, 0.78, 0.38))
            assert ed._drag_state == ('ellipse_rim', 'g5')
            tx, ty = _d(ed, 0.80, 0.42)
            ed._on_motion(_ev(ed, 0.80, 0.42))
            g = ed._gates['g5']
            d = np.array([tx, ty]) - np.asarray(g['mean'])
            assert abs(d @ np.linalg.inv(g['cov']) @ d - 1.0) < 1e-9

        # The log view, on handles clear of the data-unit rect tolerance.
        scene = _fresh(ed, 'log', 'upper')
        ed._on_press(_ev(ed, 0.97, 0.80))
        assert ed._drag_state == ('rect_corner', 'g1', 'br')
        ed._on_motion(_ev(ed, 0.93, 0.76))
        g = ed._gates['g1']
        assert (g['x1'], g['y0']) == _d(ed, 0.93, 0.76)
        assert (g['x0'], g['y1']) == (scene['g1']['x0'], scene['g1']['y1'])
        _install(ed, scene, 'log')
        ed._on_press(_ev(ed, 0.62, 0.94))
        assert ed._drag_state == ('ellipse_rim', 'g2')
        tx, ty = _d(ed, 0.62, 0.96)
        ed._on_motion(_ev(ed, 0.62, 0.96))
        g = ed._gates['g2']
        d = np.array([tx, ty]) - np.asarray(g['mean'])
        assert abs(d @ np.linalg.inv(g['cov']) @ d - 1.0) < 1e-9
        _install(ed, scene, 'log')
        (mx, my), _i, _r, _hxy = ed._ellipse_geom(scene['g2'])
        p0 = _ev(ed, *_f(ed, mx, my))
        ed._on_press(p0)
        assert ed._drag_state == ('ellipse_translate', 'g2')
        assert ed._drag_translate_ctx['orig'] == scene['g2']
        ed._on_motion(_ev(ed, 0.60, 0.85))
        tx, ty = _d(ed, 0.60, 0.85)
        assert ed._gates['g2']['mean'] == [mx + (tx - p0.xdata),
                                           my + (ty - p0.ydata)]
    finally:
        root.destroy()


def test_right_and_alt_clicks_edit_polygon_vertices():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        scene = _fresh(ed, 'linear', 'shapes')
        # Right-click a pentagon corner: deleted, one undo step, no drag.
        ed._on_press(_ev(ed, 0.92, 0.85, button=3))
        assert len(ed._gates['g2']['vertices']) == 4
        assert ed._drag_state is None and len(ed._undo_stack) == 1
        # Right-click an edge: a vertex is inserted after the edge's start.
        _install(ed, scene, 'linear')
        ed._on_press(_ev(ed, 0.70, 0.51, button=3))
        v = ed._gates['g2']['vertices']
        assert len(v) == 6 and v[1] == list(_d(ed, 0.70, 0.51))
        # Right-click on nothing: nothing.
        _install(ed, scene, 'linear')
        ed._on_press(_ev(ed, 0.97, 0.03, button=3))
        assert ed._gates == scene and len(ed._undo_stack) == 0
        # Left-click an edge: inserts and drags the new vertex.
        _install(ed, scene, 'linear')
        ed._on_press(_ev(ed, 0.70, 0.51))
        assert ed._drag_state == ('poly_vertex', 'g2', 1)
        ed._on_motion(_ev(ed, 0.72, 0.45))
        assert ed._gates['g2']['vertices'][1] == list(_d(ed, 0.72, 0.45))
        # Alt-click inside: a vertex appended to that polygon.
        _install(ed, scene, 'linear')
        ed._on_press(_ev(ed, 0.75, 0.75, key='alt'))
        assert ed._gates['g2']['vertices'][-1] == list(_d(ed, 0.75, 0.75))
        assert ed._drag_state is None
    finally:
        root.destroy()


def test_pan_zoom_and_quadrant_add_gestures():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        scene = _fresh(ed, 'linear', 'lines')
        (xl, xh), (yl, yh) = LIMITS['linear']
        # Middle-drag pans by the pixel delta at press-time scale.
        p0 = _ev(ed, 0.5, 0.5, button=2)
        ed._on_press(p0)
        assert ed._pan_start is not None
        p1 = _ev(ed, 0.6, 0.45)
        ed._on_motion(p1)
        bw, bh = ed._pan_start[4:]
        dx = (p1.x - p0.x) / bw * (xh - xl)
        dy = (p1.y - p0.y) / bh * (yh - yl)
        assert np.allclose(ed.ax.get_xlim(), (xl - dx, xh - dx))
        assert np.allclose(ed.ax.get_ylim(), (yl - dy, yh - dy))
        assert dx > 0 and dy < 0
        # Zoom mode: a press records the start, a motion draws the band.
        _install(ed, scene, 'linear')
        ed._zoom_mode = True
        ed._on_press(_ev(ed, 0.2, 0.3))
        assert ed._zoom_start == _d(ed, 0.2, 0.3)
        ed._on_motion(_ev(ed, 0.5, 0.1))
        r = ed._zoom_rect_artist
        (x0, y0), (x1, y1) = _d(ed, 0.2, 0.3), _d(ed, 0.5, 0.1)
        assert (r.get_x(), r.get_y()) == (min(x0, x1), min(y0, y1))
        assert np.isclose(r.get_width(), abs(x1 - x0))
        # Quadrant tool double-click on empty space: 4 open rects.
        _install(ed, scene, 'linear')
        ed._on_press(_ev(ed, 0.9, 0.9, dblclick=True))
        new = [g for gid, g in ed._gates.items() if gid not in scene]
        assert len(new) == 4 and {g['quad_set'] for g in new} == {'qs51'}
        assert {g['quad_origin_x'] for g in new} == {_d(ed, 0.9, 0.9)[0]}
    finally:
        root.destroy()
