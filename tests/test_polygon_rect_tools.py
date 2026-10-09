"""Polygon / lasso / rectangle drawing tools: the commit path and whether the
committed gate selects what is drawn.

Coverage before: _on_rect_select ran 1/11 statements, _on_poly_select 1/7,
_on_lasso_select 1/7.

Ground truth: the shape on screen. The polygon patch joins its vertices with
straight SCREEN lines; gate_to_mask joins them with straight DATA lines. On a
linear axis those coincide; on a log axis (the editor default for channels
stored linear, e.g. FSC/SSC) they do not: a 2-decade triangle selected 46,226
events where 24,898 are inside the drawn shape. A polygon committed on a
non-linear axis now gets points along each on-screen edge (DENSIFY), so the
stored data-space polygon is the drawn one.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


def _setup(ed, scale: str | None = 'linear'):
    rng = np.random.default_rng(1)
    n = 40_000
    df = pd.DataFrame({'FSC-A': 10 ** rng.uniform(2, 6, n),
                       'SSC-A': 10 ** rng.uniform(2, 6, n),
                       'CD3': rng.normal(0, 1, n)})
    cols = list(df.columns)
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=r'C:\exp\s1.fcs', data=df, fluor_channels=['CD3'],
        channel_labels={c: c for c in cols})
    ed._sample_order.append('s1')
    ed._sample_colors['s1'] = '#1f77b4'
    ed._sample_trial['s1'] = 'T'
    ed._trial_order.append('T')
    ed._sample_plot_enabled['s1'] = True
    ed._channels = cols
    ed._channel_labels = {c: c for c in cols}
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed.x_combo.set('FSC-A')
    ed.y_combo.set('SSC-A')
    ed.mode_var.set('dot')
    if scale:
        ed._channel_scale['FSC-A'] = scale
        ed._channel_scale['SSC-A'] = scale
    ed._replot()
    return df


def _new_gate(ed, before):
    (gid,) = [g for g in ed._gates if g not in before]
    return gid


def _drawn_and_selected(ed, gid, df):
    import openflo.pipeline as fp
    ed._redraw_only_gates()
    ed.canvas.draw()
    patch = ed._shape_artists[gid]
    tr = patch.get_transform()
    tpath = tr.get_affine().transform_path(tr.transform_path_non_affine(patch.get_path()))
    disp = ed.ax.transData.transform(df[['FSC-A', 'SSC-A']].values)
    return tpath.contains_points(disp), fp.gate_to_mask(ed._gates[gid], df)


_POLY = [(1e3, 2e3), (9e4, 1.5e3), (6e4, 7e4), (2e4, 9e4), (1.5e3, 3e4)]


def test_polygon_tool_commits_vertices_and_selects_what_is_drawn_on_linear_axes():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed)
        before = set(ed._gates)
        ed._on_poly_select(_POLY)
        gid = _new_gate(ed, before)
        g = ed._gates[gid]
        assert g['kind'] == 'polygon'
        assert (g['x_channel'], g['y_channel']) == ('FSC-A', 'SSC-A')
        assert g['vertices'] == [[float(x), float(y)] for x, y in _POLY]
        drawn, sel = _drawn_and_selected(ed, gid, df)
        assert sel.sum() > 1000
        assert (drawn != sel).sum() <= 2
        # Fewer than 3 vertices commits nothing.
        n = len(ed._gates)
        ed._on_poly_select(_POLY[:2])
        assert len(ed._gates) == n
    finally:
        root.destroy()


def test_lasso_tool_commits_its_path():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed)
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        path = list(zip(5e4 + 3e4 * np.cos(t), 5e4 + 2e4 * np.sin(t), strict=True))
        before = set(ed._gates)
        ed._on_lasso_select(path)
        gid = _new_gate(ed, before)
        assert ed._gates[gid]['vertices'] == [[float(x), float(y)] for x, y in path]
        drawn, sel = _drawn_and_selected(ed, gid, df)
        assert sel.sum() > 200 and (drawn != sel).sum() <= 2
    finally:
        root.destroy()


def test_rect_tool_selects_what_is_drawn_even_on_log_axes():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale=None)
        assert ed.ax.get_xscale() == 'log'
        before = set(ed._gates)
        ed._on_rect_select(SimpleNamespace(xdata=1e5, ydata=3e4),
                           SimpleNamespace(xdata=1e3, ydata=2e3))
        gid = _new_gate(ed, before)
        g = ed._gates[gid]
        assert (g['x0'], g['x1'], g['y0'], g['y1']) == (1e3, 1e5, 2e3, 3e4)
        drawn, sel = _drawn_and_selected(ed, gid, df)
        assert sel.sum() > 1000 and (drawn != sel).sum() == 0
    finally:
        root.destroy()


def test_polygon_drawn_on_log_axes_selects_what_is_drawn():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale=None)
        assert ed.ax.get_xscale() == 'log'
        before = set(ed._gates)
        ed._on_poly_select([(1e3, 1e3), (1e5, 1e3), (1e3, 1e5)])
        gid = _new_gate(ed, before)
        drawn, sel = _drawn_and_selected(ed, gid, df)
        assert (drawn != sel).sum() <= 0.01 * max(int(drawn.sum()), 1), (
            int(drawn.sum()), int(sel.sum()))
    finally:
        root.destroy()


def test_log_axis_commit_keeps_the_drawn_vertices_and_adds_only_to_bent_edges():
    """Densifying adds points along the on-screen edges; the user's own
    vertices are kept exactly, and an edge that is straight in data too (axis
    parallel, on log axes) gains nothing."""
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale=None)
        before = set(ed._gates)
        tri = [[1e3, 1e3], [1e5, 1e3], [1e3, 1e5]]
        ed._on_poly_select([tuple(v) for v in tri])
        verts = ed._gates[_new_gate(ed, before)]['vertices']
        assert verts[:2] == tri[:2] and verts[-1] == tri[2] and len(verts) > 3
        # A lasso on log axes: its path survives as a subsequence, and the
        # committed gate selects what is drawn.
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        path = list(zip(10 ** (4 + 0.8 * np.cos(t)), 10 ** (4 + 0.5 * np.sin(t)),
                        strict=True))
        before = set(ed._gates)
        ed._on_lasso_select(path)
        gid = _new_gate(ed, before)
        kept = {tuple(v) for v in ed._gates[gid]['vertices']}
        assert all((float(x), float(y)) in kept for x, y in path)
        drawn, sel = _drawn_and_selected(ed, gid, df)
        assert sel.sum() > 1000 and (drawn != sel).sum() <= 0.01 * drawn.sum()
    finally:
        root.destroy()


_TRI = [(1e3, 1e3), (1e5, 1e3), (1e3, 1e5)]


def _ev(ed, x, y, button=1):
    px, py = ed.ax.transData.transform((x, y))
    return SimpleNamespace(inaxes=ed.ax, xdata=x, ydata=y, x=px, y=py,
                           button=button, key=None, dblclick=False)


def _log_triangle(ed):
    """A triangle committed on the default log axes, as drawn: its corners
    plus points along the bent edge (DENSIFY). Returns its gate id."""
    before = set(ed._gates)
    ed._on_poly_select(_TRI)
    gid = _new_gate(ed, before)
    ed._replot()
    return gid


def test_a_click_on_a_corner_of_a_log_axis_polygon_grabs_that_corner():
    """The polygon hit-test measured its tolerance in data units and took the
    FIRST vertex inside it. On a log axis that spans decades at the low end:
    a click exactly on corner (1e3, 1e5) grabbed the point (1540, 64938)
    along the hypotenuse. It is measured on screen now, nearest first."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)
        gid = _log_triangle(ed)
        verts = ed._gates[gid]['vertices']
        for c in _TRI:
            hit = ed._hit_test(_ev(ed, *c))
            assert hit is not None and hit[:2] == ('poly_vertex', gid)
            assert verts[hit[2]] == list(c)
    finally:
        root.destroy()


def _drag(ed, src, dst):
    ed._on_press(_ev(ed, *src))
    ed._on_motion(_ev(ed, *dst))
    ed._on_release(_ev(ed, *dst))


def _as_drawn(ed, poly):
    """The vertices the polygon tool stores for ``poly`` drawn on this view
    (the gate is removed again)."""
    before = set(ed._gates)
    ed._on_poly_select(poly)
    gid = _new_gate(ed, before)
    verts = ed._gates[gid]['vertices']
    ed._purge_gate_subtree('s1', gid)
    return verts


def test_dragging_a_corner_on_log_axes_stores_what_drawing_the_new_shape_would():
    """A triangle drawn on log axes is stored as its corners plus points along
    the bent edge. Dragging a corner moved only that point: a notch whose
    edges, straight in data, bent on screen (862 of 4,957 events disagreed
    between drawn and selected). The vertex's edges are now re-routed from
    its neighbouring corners, straight on screen, as a new drawing is."""
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale=None)
        gid = _log_triangle(ed)
        orig = [list(v) for v in ed._gates[gid]['vertices']]
        ed.update_idletasks()                 # the drag is its own undo step
        m = (3e5, 3e3)
        _drag(ed, _TRI[1], m)
        drawn, sel = _drawn_and_selected(ed, gid, df)
        assert (drawn != sel).sum() <= 0.01 * drawn.sum(), (
            int(drawn.sum()), int(sel.sum()))
        assert ed._gates[gid]['vertices'] == _as_drawn(ed, [_TRI[0], m, _TRI[2]])
        ed._undo()
        assert ed._gates[gid]['vertices'] == orig
        # A point taken from along the hypotenuse becomes a fourth corner.
        j = len(orig) // 2
        m = (5e4, 5e4)
        _drag(ed, orig[j], m)
        assert ed._gates[gid]['vertices'] == _as_drawn(
            ed, [_TRI[0], _TRI[1], m, _TRI[2]])
    finally:
        root.destroy()


def test_an_edge_drag_on_a_log_view_reroutes_only_that_edge():
    """A polygon stored with straight DATA edges (drawn on a linear view, or
    imported) and edited on a log view: clicking an edge inserts a vertex,
    and its two new edges are straight on screen; every other stored point
    is kept."""
    from openflo.plotmath import polyline_on_screen
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        quad = [(2e4, 2e4), (8e4, 2e4), (8e4, 8e4), (2e4, 8e4)]
        before = set(ed._gates)
        ed._on_poly_select(quad)
        gid = _new_gate(ed, before)
        ed._channel_scale.pop('FSC-A')
        ed._channel_scale.pop('SSC-A')
        ed._replot()
        assert ed.ax.get_xscale() == 'log'
        old = ed._gates[gid]['vertices']
        tr = ed.ax.transData
        mid = tr.inverted().transform(tr.transform(quad[:2]).mean(axis=0))
        m = (5e4, 6e3)
        _drag(ed, tuple(mid), m)
        tr = ed.ax.transData
        want = (old[:1] + polyline_on_screen([old[0], m, old[1]], tr.transform,
                                             tr.inverted().transform).tolist()[1:-1]
                + old[1:])
        assert ed._gates[gid]['vertices'] == want and len(want) > 5
    finally:
        root.destroy()


def test_deleting_a_corner_on_log_axes_joins_its_neighbours_on_screen():
    """Right-click on a corner of a polygon drawn on log axes removed that one
    point, leaving the points along its two edges: a chopped corner. Now the
    corner goes with them; a triangle keeps its three corners."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)
        gid = _log_triangle(ed)
        tri = [list(v) for v in ed._gates[gid]['vertices']]
        ed._on_press(_ev(ed, *_TRI[2], button=3))
        assert ed._gates[gid]['vertices'] == tri
        quad = [(1e3, 1e3), (1e5, 1e3), (1e5, 1e5), (1e3, 3e4)]
        before = set(ed._gates)
        ed._on_poly_select(quad)
        qid = _new_gate(ed, before)
        ed._purge_gate_subtree('s1', gid)
        ed._replot()
        ed._on_press(_ev(ed, *quad[2], button=3))
        assert ed._gates[qid]['vertices'] == _as_drawn(
            ed, [quad[0], quad[1], quad[3]])
    finally:
        root.destroy()


def test_a_user_vertex_collinear_on_screen_survives_a_neighbours_drag():
    """(1e4, 1e3) is exactly on the screen line through its neighbours, like
    a point added along a drawn edge, and was dropped when (1e5, 1e3) was
    dragged. That edge is straight in data too, where no point is ever
    added, so it is the user's own vertex."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)
        quad = [(1e3, 1e3), (1e4, 1e3), (1e5, 1e3), (1e3, 1e5)]
        before = set(ed._gates)
        ed._on_poly_select(quad)
        gid = _new_gate(ed, before)
        ed._replot()
        _drag(ed, quad[2], (2e5, 2e3))
        assert [1e4, 1e3] in ed._gates[gid]['vertices']
    finally:
        root.destroy()


def test_dragging_a_bent_edge_between_two_points_reroutes_from_its_corners():
    """A click on a drawn edge between two of the points stored along it adds
    a vertex there. Added and then re-routed, it made those two points
    corners, and an 80 px drag raised a tent between them. The new vertex
    is re-routed from the edge's own corners, as drawing the new shape
    would store it."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)
        tri = [(200.0, 200.0), (8e5, 300.0), (300.0, 8e5)]
        before = set(ed._gates)
        ed._on_poly_select(tri)
        gid = _new_gate(ed, before)
        ed._replot()
        verts = ed._gates[gid]['vertices']
        tr = ed.ax.transData
        d = tr.transform(np.asarray(verts))
        gaps = np.hypot(*np.diff(np.vstack([d, d[:1]]), axis=0).T)
        j = int(np.argmax(gaps))
        a, b = d[j], d[(j + 1) % len(d)]
        normal = np.array([b[1] - a[1], a[0] - b[0]]) / np.hypot(*(b - a))
        if normal @ ((a + b) / 2 - d.mean(axis=0)) < 0:
            normal = -normal
        # A click 3 px off the edge, as a hand puts it.
        click = tr.inverted().transform((a + b) / 2 + 3 * normal)
        assert ed._hit_test(_ev(ed, *click))[:2] == ('poly_edge', gid)
        m = tuple(tr.inverted().transform((a + b) / 2 + 80 * normal))
        _drag(ed, tuple(click), m)
        k = [verts.index(list(c)) for c in tri]          # corners' positions
        after = sum(kk <= j for kk in k)                 # edge j follows corner after-1
        shape = tri[:after] + [m] + tri[after:]
        # The drag target lies just below the axes (measured: y 59.8 under a
        # bottom of 63.1), where the polygon tool would also open the drawing
        # at the axis end (it never gets such a point: its selector clamps
        # it); the re-route is compared with the tool's points along the
        # bent edges.
        got, want = ed._gates[gid]['vertices'], ed._vertices_as_drawn(shape)
        assert len(got) == len(want) and np.allclose(got, want, rtol=1e-12)
    finally:
        root.destroy()


def test_right_click_on_a_bent_edge_adds_a_vertex_and_one_undo_step():
    """Right-click on a drawn edge of a log-axis polygon lands nearest one of
    the points stored along it, so it 'deleted' that point. Re-routing then
    put the edge back as it was, and left an undo step that undid nothing.
    It now adds a vertex at the click, re-routed from the edge's corners.
    A refused delete (a triangle's corner) records no undo step."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)
        gid = _log_triangle(ed)
        verts = [list(v) for v in ed._gates[gid]['vertices']]
        tr = ed.ax.transData
        j = len(verts) // 2                            # along the hypotenuse
        click = tuple(tr.inverted().transform(
            tr.transform(verts[j]) + np.array([4.0, 4.0])))
        assert ed._hit_test(_ev(ed, *click)) == ('poly_vertex', gid, j)
        ed.update_idletasks()
        n = len(ed._undo_stack)
        ed._on_press(_ev(ed, *click, button=3))
        got = ed._gates[gid]['vertices']
        want = _as_drawn(ed, [_TRI[0], _TRI[1], click, _TRI[2]])
        assert len(got) == len(want) and np.allclose(got, want, rtol=1e-12)
        assert len(ed._undo_stack) == n + 1
        ed._undo()
        assert ed._gates[gid]['vertices'] == verts
        # A triangle's corner cannot go: no change, and no undo step.
        ed.update_idletasks()
        n = len(ed._undo_stack)
        ed._on_press(_ev(ed, *_TRI[2], button=3))
        assert ed._gates[gid]['vertices'] == verts
        assert len(ed._undo_stack) == n
    finally:
        root.destroy()
