"""What ``GatingMixin._draw_gates`` draws for each gate kind, and where it
keeps it (editor_gating).

One helper per kind draws a gate (threshold, interval, rect, polygon,
ellipsoid); these pin each kind's artists through ``_draw_gates`` itself:
the artist type, its data, colour, line width / style, alpha, z-order,
whether it was added as a patch (in ``dataLim``, so in the autoscale) or as
a plain artist (an open-sided shape, kept out of it), the
``_vlines`` / ``_hlines`` / ``_shape_artists`` keys the drag code reads,
the order of the artists, and the gates it skips (open thresholds, other
channels, too few vertices, degenerate ellipses, kinds it does not draw).

No Tk: a real matplotlib axes on a stand-in editor built from the mixin.
Generated gate coordinates and events; channels FL1-A / FL2-A / FSC-A.
"""
from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, Polygon, Rectangle  # noqa: E402

from openflo import plotmath  # noqa: E402
from openflo.editor_gating import GatingMixin  # noqa: E402
from openflo.scales import OPEN  # noqa: E402

X, Y, Z = 'FL1-A', 'FL2-A', 'FSC-A'
RNG = np.random.default_rng(20261110)


class _Ed(GatingMixin):
    """Just what _draw_gates reads: the gates, the axes and the ellipse
    geometry (the window takes both from plotmath)."""

    _ellipse_params = staticmethod(plotmath.ellipse_params)
    _ellipse_geom = staticmethod(plotmath.ellipse_geom)

    def __init__(self, gates):
        self._gates = dict(gates)
        self.ax = Figure().add_subplot()
        # Generated events, so dataLim has something besides the gates.
        ev = RNG.normal(500.0, 150.0, (200, 2))
        self.ax.scatter(ev[:, 0], ev[:, 1], s=1)
        self._vlines, self._hlines, self._shape_artists = (
            {'stale': None}, {'stale': None}, {'stale': None})


def _draw(gates, x=X, y=Y):
    ed = _Ed(gates)
    before = list(ed.ax.get_children())
    lim = ed.ax.dataLim.frozen()
    assert ed._draw_gates(x, y) is None
    added = [a for a in ed.ax.get_children() if a not in before]
    return ed, added, lim


def _rgba(c):
    return tuple(np.round(mcolors.to_rgba(c), 6))


def _vals():
    lo, hi = sorted(RNG.uniform(50.0, 900.0, 2))
    return float(lo), float(hi)


# ── 1-D gates ───────────────────────────────────────────────────────────────

def test_threshold_is_a_dashed_vertical_line_on_x_and_horizontal_on_y():
    vx, vy = _vals()
    ed, added, _ = _draw({
        'tx': {'kind': 'threshold', 'channel': X, 'value': vx, 'color': '#2ca02c'},
        'ty': {'kind': 'threshold', 'channel': Y, 'value': str(vy), 'color': '#8c564b'},
        'tz': {'kind': 'threshold', 'channel': Z, 'value': vx}})
    assert list(ed._vlines) == ['tx'] and list(ed._hlines) == ['ty']
    assert ed._shape_artists == {}
    v, h = ed._vlines['tx'], ed._hlines['ty']
    assert added == [v, h]
    assert list(v.get_xdata()) == [vx, vx] and list(h.get_ydata()) == [vy, vy]
    assert v.get_color() == '#2ca02c' and h.get_color() == '#8c564b'
    for ln in (v, h):
        assert (ln.get_linewidth(), ln.get_linestyle(), ln.get_alpha()) == (1.3, '--', 0.85)


def test_open_threshold_is_skipped_even_on_the_x_channel():
    ed, added, _ = _draw({
        'a': {'kind': 'threshold', 'channel': X, 'value': -OPEN},
        'b': {'kind': 'threshold', 'channel': Y, 'value': 2 * OPEN},
        'c': {'kind': 'threshold', 'channel': X, 'value': OPEN - 1.0}})
    assert added == [ed._vlines['c']] and list(ed._vlines) == ['c']
    assert ed._vlines['c'].get_color() == 'red'                  # default
    assert ed._hlines == {}


def test_interval_draws_each_closed_bound_lo_then_hi():
    lo, hi = _vals()
    ed, added, _ = _draw({
        'ix': {'kind': 'interval', 'channel': X, 'lo': lo, 'hi': hi, 'color': '#e377c2'},
        'iy': {'kind': 'interval', 'channel': Y, 'lo': -OPEN, 'hi': hi, 'color': '#bcbd22'},
        'iz': {'kind': 'interval', 'channel': Z, 'lo': lo, 'hi': hi},
        'io': {'kind': 'interval', 'channel': X, 'lo': lo, 'hi': OPEN}})
    assert list(ed._vlines) == ['ix:lo', 'ix:hi', 'io:lo']
    assert list(ed._hlines) == ['iy:hi']
    assert added == [ed._vlines['ix:lo'], ed._vlines['ix:hi'], ed._hlines['iy:hi'],
                     ed._vlines['io:lo']]
    assert ed._vlines['ix:lo'].get_xdata()[0] == lo
    assert ed._vlines['ix:hi'].get_xdata()[0] == hi
    assert ed._hlines['iy:hi'].get_ydata()[0] == hi
    assert ed._vlines['ix:lo'].get_color() == '#e377c2'
    assert ed._hlines['iy:hi'].get_color() == '#bcbd22'
    assert ed._vlines['io:lo'].get_color() == 'red'              # default
    for ln in added:
        assert (ln.get_linewidth(), ln.get_linestyle(), ln.get_alpha()) == (1.2, '--', 0.85)


def test_histogram_call_draws_only_the_x_channel_lines():
    vx, vy = _vals()
    ed, added, _ = _draw({
        't': {'kind': 'threshold', 'channel': X, 'value': vx},
        'u': {'kind': 'threshold', 'channel': Y, 'value': vy},
        'i': {'kind': 'interval', 'channel': Y, 'lo': vx, 'hi': vy},
        'r': {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
              'x0': vx, 'x1': vy, 'y0': vx, 'y1': vy}}, y=None)
    assert list(ed._vlines) == ['t'] and ed._hlines == {} and ed._shape_artists == {}
    assert added == [ed._vlines['t']]


# ── Rectangles ──────────────────────────────────────────────────────────────

def _style_ok(p, color):
    assert p.get_fill() is False
    assert _rgba(p.get_edgecolor()) == _rgba(mcolors.to_rgba(color, 0.95))
    assert (p.get_linewidth(), p.get_linestyle(), p.get_alpha()) == (1.4, '-', 0.95)


def test_rect_is_a_rectangle_patch_with_sorted_corners():
    x0, x1 = _vals()
    y0, y1 = _vals()
    ed, added, _ = _draw({
        'r': {'kind': 'rect', 'x_channel': X, 'y_channel': Y, 'color': '#9467bd',
              'x0': x1, 'x1': x0, 'y0': y1, 'y1': y0},
        'swapped': {'kind': 'rect', 'x_channel': Y, 'y_channel': X,
                    'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1},
        'other': {'kind': 'rect', 'x_channel': X, 'y_channel': Z,
                  'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1}})
    p = ed._shape_artists['r']
    assert list(ed._shape_artists) == ['r'] and added == [p]
    assert type(p) is Rectangle and p in ed.ax.patches
    assert p.get_xy() == (x0, y0)
    assert (p.get_width(), p.get_height()) == (x1 - x0, y1 - y0)
    _style_ok(p, '#9467bd')
    assert ed._vlines == {} and ed._hlines == {}


def test_open_sided_rect_is_kept_out_of_the_patches_and_the_autoscale():
    lo, hi = _vals()
    ed, added, lim = _draw({
        'q1': {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
               'x0': lo, 'x1': OPEN, 'y0': -OPEN, 'y1': hi},
        'q2': {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
               'x0': -OPEN, 'x1': lo, 'y0': -OPEN, 'y1': hi},
        'top': {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
                'x0': lo, 'x1': hi, 'y0': lo, 'y1': OPEN}})
    p1, p2 = ed._shape_artists['q1'], ed._shape_artists['q2']
    p3 = ed._shape_artists['top']                      # open in y alone
    assert added == [p1, p2, p3] and type(p3) is type(p1)
    assert type(p1) is type(p2) and p1 is not p2      # one patch per gate
    assert not isinstance(p1, (Rectangle, Polygon))
    assert np.allclose(ed.ax.dataLim.get_points(), lim.get_points())
    _style_ok(p1, 'red')
    assert np.array_equal(p1._verts, [[lo, -OPEN], [OPEN, -OPEN], [OPEN, hi], [lo, hi]])


def test_closed_rect_enters_the_autoscale():
    ed, _, _ = _draw({'r': {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
                              'x0': -5000.0, 'x1': 9000.0, 'y0': -4000.0, 'y1': 8000.0}})
    (x0, y0), (x1, y1) = ed.ax.dataLim.get_points()
    assert (x0, y0, x1, y1) == (-5000.0, -4000.0, 9000.0, 8000.0)


# ── Polygons ────────────────────────────────────────────────────────────────

def test_polygon_is_a_closed_polygon_patch():
    verts = RNG.uniform(0.0, 1000.0, (5, 2))
    ed, added, _ = _draw({
        'p': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
              'vertices': verts.tolist(), 'color': '#ff7f0e'},
        'two': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                'vertices': verts[:2].tolist()},
        'flat': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                 'vertices': verts[:, 0].tolist()},
        'three_d': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                    'vertices': [[1.0, 2.0, 3.0]] * 4},
        'other': {'kind': 'polygon', 'x_channel': Z, 'y_channel': Y,
                  'vertices': verts.tolist()},
        'other_y': {'kind': 'polygon', 'x_channel': X, 'y_channel': Z,
                    'vertices': verts.tolist()}})
    p = ed._shape_artists['p']
    assert list(ed._shape_artists) == ['p'] and added == [p]
    assert type(p) is Polygon and p in ed.ax.patches and p.get_closed()
    assert np.array_equal(p.get_xy()[:-1], verts)
    _style_ok(p, '#ff7f0e')


def test_closed_polygon_enters_the_autoscale():
    far = RNG.uniform(5000.0, 9000.0, (4, 2))
    ed, _, _ = _draw({'p': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                            'vertices': far.tolist()}})
    (_, _), (x1, y1) = ed.ax.dataLim.get_points()
    assert (x1, y1) == (far[:, 0].max(), far[:, 1].max())


def test_triangle_is_enough_and_open_polygon_stays_out_of_the_autoscale():
    tri = RNG.uniform(0.0, 1000.0, (3, 2))
    opened = tri.copy()
    opened[0, 0] = -OPEN
    ed, added, lim = _draw({
        'open': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                 'vertices': opened.tolist()}})
    p = ed._shape_artists['open']
    assert added == [p]
    assert not isinstance(p, Polygon)
    assert np.array_equal(p._verts, opened)
    assert np.allclose(ed.ax.dataLim.get_points(), lim.get_points())
    _style_ok(p, 'red')
    ed, added, _ = _draw({'tri': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                                  'vertices': tri.tolist()}})
    assert type(ed._shape_artists['tri']) is Polygon


# ── Ellipses ────────────────────────────────────────────────────────────────

def _ellipse(color=None, cov=((90_000.0, 20_000.0), (20_000.0, 40_000.0))):
    g = {'kind': 'ellipsoid', 'x_channel': X, 'y_channel': Y,
         'mean': RNG.uniform(300.0, 700.0, 2).tolist(),
         'cov': [list(r) for r in cov], 'distance_sq': 1.0}
    if color:
        g['color'] = color
    return g


def test_ellipsoid_is_a_patch_then_its_rotation_handle():
    g = _ellipse('#17becf')
    g['mean'] = (RNG.uniform(4000.0, 6000.0, 2)).tolist()   # beyond the events
    ed, added, _ = _draw({'e': g, 'flat': _ellipse(cov=((0.0, 0.0), (0.0, 0.0))),
                          'other': dict(_ellipse(), y_channel=Z)})
    p, h = ed._shape_artists['e'], ed._shape_artists['e:rot']
    assert list(ed._shape_artists) == ['e', 'e:rot'] and added == [p, h]
    assert type(p) is Ellipse and p in ed.ax.patches
    _style_ok(p, '#17becf')
    # Added as a patch: its extent is in the autoscale.
    assert ed.ax.dataLim.x1 > g['mean'][0] + 250.0 > 4000.0
    assert isinstance(h, Line2D)
    hx, hy = plotmath.ellipse_geom(g)[3]
    assert list(h.get_xydata()[0]) == [hx, hy]
    assert (h.get_marker(), h.get_markersize(), h.get_linestyle()) == ('o', 5, 'None')
    assert (h.get_markerfacecolor(), h.get_markeredgecolor()) == ('#17becf', 'white')
    assert (h.get_markeredgewidth(), h.get_alpha(), h.get_zorder()) == (0.6, 0.9, 6)


def test_ellipsoid_without_geometry_has_no_handle():
    class NoGeom(_Ed):
        _ellipse_geom = staticmethod(lambda g: None)

    ed = NoGeom({'e': _ellipse()})
    ed._draw_gates(X, Y)
    assert list(ed._shape_artists) == ['e']
    assert ed.ax.patches[-1] is ed._shape_artists['e']


# ── The loop ────────────────────────────────────────────────────────────────

def test_artists_follow_gate_order_and_the_registries_start_empty():
    lo, hi = _vals()
    gates = {
        'e': _ellipse(),
        'p': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
              'vertices': RNG.uniform(0, 900, (4, 2)).tolist()},
        'b': {'kind': 'boolean', 'op': 'and', 'operands': ['t', 'i']},
        'i': {'kind': 'interval', 'channel': Y, 'lo': lo, 'hi': hi},
        'q': {'kind': 'quadrant', 'x_channel': X, 'y_channel': Y,
              'x0': lo, 'x1': hi, 'y0': lo, 'y1': hi},
        'n': {'channel': X, 'value': lo},
        'r': {'kind': 'rect', 'x_channel': X, 'y_channel': Y,
              'x0': lo, 'x1': hi, 'y0': lo, 'y1': hi},
        't': {'kind': 'threshold', 'channel': X, 'value': hi},
        'a': {'kind': 'autoclean', 'channel': X, 'value': lo},
    }
    ed, added, _ = _draw(gates)
    s = ed._shape_artists
    assert added == [s['e'], s['e:rot'], s['p'], ed._hlines['i:lo'], ed._hlines['i:hi'],
                     s['r'], ed._vlines['t']]
    assert list(s) == ['e', 'e:rot', 'p', 'r']
    assert list(ed._vlines) == ['t'] and list(ed._hlines) == ['i:lo', 'i:hi']


def test_no_gates_leaves_empty_registries():
    ed, added, _ = _draw({})
    assert added == [] and (ed._vlines, ed._hlines, ed._shape_artists) == ({}, {}, {})


@pytest.mark.parametrize('bad', [
    {'kind': 'threshold', 'channel': X, 'value': 'abc'},
    {'kind': 'interval', 'channel': X, 'lo': 1.0},
    {'kind': 'rect', 'x_channel': X, 'y_channel': Y, 'x0': 1.0, 'x1': 2.0, 'y0': 1.0},
])
def test_a_broken_gate_is_left_out_and_the_others_drawn(bad):
    """A gate missing a bound or with a bound that is not a number has no
    region to draw (gate_record_problem); it used to raise on every
    redraw. Its population reads as unknown in Statistics instead."""
    vx, _ = _vals()
    ed = _Ed({'t': {'kind': 'threshold', 'channel': X, 'value': vx}, 'bad': bad,
              'u': {'kind': 'threshold', 'channel': Y, 'value': vx}})
    ed._draw_gates(X, Y)
    assert list(ed._vlines) == ['t'] and list(ed._hlines) == ['u']
    assert 'bad' not in ed._shape_artists


# ── Review follow-ups: self-plots, string values, one-channel matches ──────

def test_a_channel_plotted_against_itself_gets_only_vertical_lines():
    lo, hi = _vals()
    ed, added, _ = _draw({
        't': {'kind': 'threshold', 'channel': X, 'value': lo},
        'i': {'kind': 'interval', 'channel': X, 'lo': lo, 'hi': hi}}, x=X, y=X)
    assert list(ed._vlines) == ['t', 'i:lo', 'i:hi'] and ed._hlines == {}
    assert added == [ed._vlines['t'], ed._vlines['i:lo'], ed._vlines['i:hi']]


def test_string_values_are_drawn_as_numbers_on_x_and_y():
    lo, hi = _vals()
    ed, _, _ = _draw({
        'tx': {'kind': 'threshold', 'channel': X, 'value': str(lo)},
        'ix': {'kind': 'interval', 'channel': X, 'lo': str(lo), 'hi': str(hi)},
        'iy': {'kind': 'interval', 'channel': Y, 'lo': str(lo), 'hi': str(hi)}})
    got = [ed._vlines['tx'].get_xdata()[0], ed._vlines['ix:lo'].get_xdata()[0],
           ed._vlines['ix:hi'].get_xdata()[0], ed._hlines['iy:lo'].get_ydata()[0],
           ed._hlines['iy:hi'].get_ydata()[0]]
    assert got == [lo, lo, hi, lo, hi]
    assert all(type(v) is float for v in got)


def test_rect_on_the_wrong_x_channel_is_not_drawn():
    x0, x1 = _vals()
    ed, added, _ = _draw({
        'wrong_x': {'kind': 'rect', 'x_channel': Z, 'y_channel': Y,
                    'x0': x0, 'x1': x1, 'y0': x0, 'y1': x1}})
    assert added == [] and ed._shape_artists == {}


def test_polygon_open_only_in_y_stays_out_of_the_autoscale():
    verts = RNG.uniform(0.0, 1000.0, (4, 2))
    verts[2, 1] = OPEN
    ed, added, lim = _draw({
        'open_y': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
                   'vertices': verts.tolist()}})
    p = ed._shape_artists['open_y']
    assert added == [p] and not isinstance(p, Polygon)
    assert np.array_equal(p._verts, verts)
    assert np.allclose(ed.ax.dataLim.get_points(), lim.get_points())
