"""A shape drawn to the end of an axis is open there, as on screen.

The editor's default view of a logicle-stored fluorescence channel is 'log'.
A log view cannot show intensity <= 0, so it draws every such event at one
screen position, log10(1e-6), which is also the bottom of the axes: the
compensated negatives sit piled on the axis line. Their STORED values run
below logicle(0) = 0.111, but nothing on that axis maps back below 0.111, so
a rectangle drawn down to the axis (a drag past the plot is clamped to the
axes limit) stored a lower edge of 0.111 and left the pile out. Measured on
50,000 events (30% around zero): 22,347 drawn inside a box from the axis up
to intensity 100, 7,326 counted.

Every drawing tool now makes an edge at the bottom of the axis (within 2 px,
a marker's width) the open sentinel -1e12, and an edge at the top +1e12 --
the form a FlowJo quadrant has, written back to FlowJo as an open side, and
how FlowJo itself counts a gate edge at the bottom of its axis. Ground truth
in each test is the drawn shape on screen.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.path import Path

import openflo.pipeline as fp
from tests.conftest import gui_unavailable


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable("tkinter not available — headless environment")
    try:
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _linear(n=50_000, seed=0):
    """Compensated-like intensities, each channel independently 30% around
    zero and 70% bright."""
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(2):
        v = np.concatenate([rng.normal(0, 150, int(n * 0.3)),
                            rng.lognormal(np.log(5000), 0.6, n - int(n * 0.3))])
        cols.append(rng.permutation(v))
    return np.column_stack(cols)


def _setup(ed, scale=None):
    """FL1/FL2 stored logicle, as the loader stores fluorescence; `scale`
    None keeps the editor default ('log')."""
    lin = _linear()
    stored = fp.transform_values(lin.ravel(), 'logicle').reshape(lin.shape)
    df = pd.DataFrame(stored, columns=['FL1-A', 'FL2-A'])
    cols = list(df.columns)
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=r'C:\exp\s1.fcs', data=df, fluor_channels=cols,
        channel_labels={c: c for c in cols})
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._channel_transform = {c: 'logicle' for c in cols}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')
    ed.mode_var.set('dot')
    if scale:
        ed._channel_scale.update({c: scale for c in cols})
    ed._replot()
    ed.canvas.draw()
    return lin, df


def _to_data(ed, sx, sy):
    """Data coordinates of a point given in the axes' SCALE space (log10
    intensity on the log view)."""
    x = ed.ax.xaxis.get_transform().inverted().transform(np.array([sx]))[0]
    y = ed.ax.yaxis.get_transform().inverted().transform(np.array([sy]))[0]
    return float(x), float(y)


def _scale_xy(ed, df):
    return np.column_stack([
        ed.ax.xaxis.get_transform().transform(df['FL1-A'].to_numpy()),
        ed.ax.yaxis.get_transform().transform(df['FL2-A'].to_numpy())])


def _bottoms(ed):
    """(x, y) of the axes' lower-left corner in data, as a selector clamps
    a drag that leaves the plot there."""
    return min(ed.ax.get_xbound()), min(ed.ax.get_ybound())


def _new_gate(ed, before):
    (gid,) = [g for g in ed._gates if g not in before]
    return ed._gates[gid]


def _inside(ed, df, scale_pts):
    path = Path(np.asarray(scale_pts, dtype=float))
    return int(path.contains_points(_scale_xy(ed, df)).sum())


def test_default_view_piles_negatives_on_the_bottom_of_the_axis():
    """The premise: 30% of events (intensity <= 0) sit at log10(1e-6) = -6,
    which is the bottom of the axes, and their stored values reach below
    anything the axis maps back to."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        s = _scale_xy(ed, df)
        x_lo, y_lo = _bottoms(ed)
        tr = ed.ax.yaxis.get_transform()
        assert np.isclose(s[:, 1].min(), -6.0)
        assert np.isclose(tr.transform(np.array([y_lo]))[0], -6.0)
        assert int((lin[:, 1] <= 0).sum()) > 7000
        assert int((df['FL2-A'] < y_lo).sum()) > 7000
    finally:
        root.destroy()


def test_rect_dragged_past_the_bottom_counts_the_events_drawn_inside():
    """Dragged out of the plot, the selector clamps the corner to the axes
    bounds. The box then holds every event piled on the axis; it counted
    only those above intensity ~0."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        x_lo, y_lo = _bottoms(ed)
        x_hi, y_hi = _to_data(ed, 2.0, 2.0)             # intensity 100
        before = set(ed._gates)
        ed._on_rect_select(SimpleNamespace(xdata=x_lo, ydata=y_lo),
                           SimpleNamespace(xdata=x_hi, ydata=y_hi))
        g = _new_gate(ed, before)
        assert g['x0'] == -1e12 and g['y0'] == -1e12
        assert g['x1'] == x_hi and g['y1'] == y_hi
        s = _scale_xy(ed, df)
        drawn = int(((s[:, 0] < 2.0) & (s[:, 1] < 2.0)).sum())
        assert drawn == int(((lin[:, 0] < 100) & (lin[:, 1] < 100)).sum())
        assert int(fp.gate_to_mask(g, df).sum()) == drawn
        assert 'open' in ed.status_var.get()
    finally:
        root.destroy()


def test_rect_edge_within_a_marker_of_the_axis_is_open():
    """A press one pixel above the axis line is on the piled markers."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        ylo, yhi = ed.ax.yaxis.get_transform().transform(
            np.array(sorted(ed.ax.get_ybound())))
        one_px = (yhi - ylo) / ed.ax.bbox.height
        (x0, y0), (x1, y1) = (_to_data(ed, 1.0, ylo + one_px),
                              _to_data(ed, 3.0, 2.0))
        before = set(ed._gates)
        ed._on_rect_select(SimpleNamespace(xdata=x0, ydata=y0),
                           SimpleNamespace(xdata=x1, ydata=y1))
        g = _new_gate(ed, before)
        assert g['y0'] == -1e12 and g['x0'] == x0      # x0 was not at an end
        want = int(((lin[:, 0] >= 10) & (lin[:, 0] < 1000)
                    & (lin[:, 1] < 100)).sum())
        got = int(fp.gate_to_mask(g, df).sum())
        assert abs(got - want) <= 2, (got, want)
    finally:
        root.destroy()


def test_rect_inside_the_plot_stays_closed():
    """Edges away from the axis ends are stored as drawn; the pile stays
    out of a box drawn above it."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        before = set(ed._gates)
        lo, hi = _to_data(ed, -3.0, 1.0), _to_data(ed, 3.0, 3.0)
        ed._on_rect_select(SimpleNamespace(xdata=lo[0], ydata=lo[1]),
                           SimpleNamespace(xdata=hi[0], ydata=hi[1]))
        g = _new_gate(ed, before)
        assert (g['x0'], g['y0']) == lo and (g['x1'], g['y1']) == hi
    finally:
        root.destroy()


def test_rect_dragged_past_the_top_is_open_above():
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        x_hi, y_hi = max(ed.ax.get_xbound()), max(ed.ax.get_ybound())
        lo = _to_data(ed, 3.0, 3.0)
        before = set(ed._gates)
        ed._on_rect_select(SimpleNamespace(xdata=lo[0], ydata=lo[1]),
                           SimpleNamespace(xdata=x_hi, ydata=y_hi))
        g = _new_gate(ed, before)
        assert g['x1'] == 1e12 and g['y1'] == 1e12
        want = int(((lin[:, 0] >= 1000) & (lin[:, 1] >= 1000)).sum())
        assert abs(int(fp.gate_to_mask(g, df).sum()) - want) <= 2
    finally:
        root.destroy()


def test_polygon_with_its_base_on_the_axis_counts_the_pile_inside_it():
    """A triangle whose base is clamped to the bottom of the axes: the piled
    events between its two sides are drawn inside (on the axis line, under
    the base). Before, all of them were outside."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        y_lo = _bottoms(ed)[1]
        tri = [(_to_data(ed, 0.5, 0)[0], y_lo), (_to_data(ed, 4.5, 0)[0], y_lo),
               _to_data(ed, 2.5, 3.0)]
        before = set(ed._gates)
        ed._on_poly_select(tri)
        g = _new_gate(ed, before)
        # On screen the base is the axis line; the pile sits on it.
        drawn_tri = [(0.5, -6.05), (4.5, -6.05), (2.5, 3.0)]
        drawn = _inside(ed, df, drawn_tri)
        piled = _inside(ed, df, [(0.5, -6.05), (4.5, -6.05),
                                 (4.5, -5.95), (0.5, -5.95)])
        assert piled > 5000                       # the pile is under the base
        got = int(fp.gate_to_mask(g, df).sum())
        assert abs(got - drawn) <= 0.01 * drawn, (got, drawn)
        assert np.abs(np.asarray(g['vertices'])).max() == 1e12
    finally:
        root.destroy()


def test_lasso_leaving_the_plot_on_the_left_is_open_there():
    """A lasso dragged out of the plot on the left (its points clamped to
    the axes limit) holds every event piled on the y axis between them."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        x_lo = _bottoms(ed)[0]
        t = np.linspace(-np.pi / 2, np.pi / 2, 60)
        loop = [(-6.0 + 9.0 * np.cos(a), 3.5 + 1.0 * np.sin(a)) for a in t]
        loop += [(-6.0, 4.5), (-6.0, 2.5)]
        pts = [(x_lo if sx == -6.0 else _to_data(ed, sx, sy)[0],
                _to_data(ed, sx, sy)[1]) for sx, sy in loop]
        before = set(ed._gates)
        ed._on_lasso_select(pts)
        g = _new_gate(ed, before)
        drawn = _inside(ed, df, [(sx - 0.05 if sx == -6.0 else sx, sy)
                                 for sx, sy in loop])
        piled = int(((lin[:, 0] <= 0)
                     & (lin[:, 1] >= 10 ** 2.5) & (lin[:, 1] < 10 ** 4.5)).sum())
        assert piled > 5000
        got = int(fp.gate_to_mask(g, df).sum())
        assert abs(got - drawn) <= 0.01 * drawn, (got, drawn)
    finally:
        root.destroy()


def test_polygon_inside_the_plot_is_not_opened():
    """No vertex near an axis end: the drawn vertices are all kept (with
    points added along the edges the view bends) and nothing is opened."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale='linear')
        before = set(ed._gates)
        tri = [(0.4, 0.4), (0.8, 0.4), (0.6, 0.8)]
        ed._on_poly_select(tri)
        g = _new_gate(ed, before)
        kept = {tuple(v) for v in g['vertices']}
        assert all(v in kept for v in tri)
        assert np.abs(np.asarray(g['vertices'])).max() < 1.0
    finally:
        root.destroy()


def test_rect_edge_dragged_to_the_axis_opens_on_release():
    """Edit mode: a rectangle's bottom edge dragged down onto the axis
    stayed at 0.111; it opens when the drag ends."""
    root, ed = _editor_or_skip()
    try:
        lin, df = _setup(ed)
        lo, hi = _to_data(ed, 1.0, 1.0), _to_data(ed, 3.0, 3.0)
        gid = ed._add_gate({'kind': 'rect', 'x_channel': 'FL1-A',
                            'y_channel': 'FL2-A', 'x0': lo[0], 'x1': hi[0],
                            'y0': lo[1], 'y1': hi[1]})
        ed._drag_state = ('rect_edge', gid, 'bottom')
        ed._gates[gid]['y0'] = _bottoms(ed)[1]
        ed._on_release(SimpleNamespace(inaxes=ed.ax, xdata=0.5, ydata=0.1))
        assert ed._gates[gid]['y0'] == -1e12
        assert ed._gates[gid]['x0'] == lo[0]
    finally:
        root.destroy()
