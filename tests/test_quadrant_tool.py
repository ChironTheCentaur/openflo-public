"""The editor's quadrant tool (double-click with the Quadrant tool active).

Coverage before: no test drives _on_press's quadrant branch
(editor_gating.py:633-681); tests/test_ellipsoid_quadrant.py checks only the
WSP-imported quadrant, whose open sides are +-1e12.

Ground truth: a quadrant split PARTITIONS the parent -- every event in exactly
one of Q++ / Q+- / Q-+ / Q--. The tool used to build the outer sides from the
axis VIEW at the moment of the click (ax.get_xlim / get_ylim), so events
outside the view belonged to no quadrant. On log axes (the editor's old
default) that was every event <= 0, i.e. the compensated negatives (37% of events in
this fixture). The outer sides are now the same open +-1e12 the importer uses.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

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


def _setup(ed, scale: str | None = 'linear', transform=None):
    """CD3 / CD4: compensated negatives and a positive population. With
    ``transform`` they are stored in its units, as the loader stores fluor
    channels, and every view of them is a FuncScale."""
    rng = np.random.default_rng(3)
    n = 8000
    neg = rng.normal(0, 150, (n // 2, 2))               # compensated negatives
    pos = 10 ** rng.normal(3.3, 0.35, (n // 2, 2))
    xy = np.vstack([neg, pos])
    if transform:
        from openflo.pipeline import transform_values
        xy = np.column_stack([transform_values(xy[:, k], method=transform)
                              for k in (0, 1)])
        ed._channel_transform.update({'CD3': transform, 'CD4': transform})
    df = pd.DataFrame({'FSC-A': rng.uniform(1e4, 2e5, n),
                       'SSC-A': rng.uniform(1e4, 2e5, n),
                       'CD3': xy[:, 0], 'CD4': xy[:, 1]})
    cols = list(df.columns)
    ed._samples['s1'] = SimpleNamespace(
        name='s1', path=r'C:\exp\s1.fcs', data=df, fluor_channels=['CD3', 'CD4'],
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
    ed.x_combo.set('CD3')
    ed.y_combo.set('CD4')
    ed.mode_var.set('dot')
    ed.gate_tool_var.set('quadrant')
    if scale:
        ed._channel_scale['CD3'] = scale
        ed._channel_scale['CD4'] = scale
    ed._replot()
    return df


def _ev(ed, x, y, dblclick=False):
    px, py = ed.ax.transData.transform((x, y))
    return SimpleNamespace(button=1, key=None, dblclick=dblclick, inaxes=ed.ax,
                           xdata=x, ydata=y, x=px, y=py)


def _hits(ed, df):
    import openflo.pipeline as fp
    quads = [g for g in ed._gates.values() if g.get('quad_set')]
    assert len(quads) == 4
    assert len({g['quad_set'] for g in quads}) == 1
    assert sorted(g['label'][:3] for g in quads) == ['Q++', 'Q+-', 'Q-+', 'Q--']
    hits = np.zeros(len(df), dtype=int)
    for g in quads:
        hits += fp.gate_to_mask(g, df).astype(int)
    return quads, hits


def test_quadrant_tool_partitions_the_events_on_linear_axes():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed)
        ed._on_press(_ev(ed, 300.0, 300.0, dblclick=True))
        quads, hits = _hits(ed, df)
        assert (hits == 1).all(), (int((hits == 0).sum()), int((hits > 1).sum()))
        by = {g['label'][:3]: g for g in quads}
        x, y = df.CD3.values, df.CD4.values
        import openflo.pipeline as fp
        assert (fp.gate_to_mask(by['Q++'], df) == ((x >= 300) & (y >= 300))).all()
        assert (fp.gate_to_mask(by['Q--'], df) == ((x < 300) & (y < 300))).all()
    finally:
        root.destroy()


def test_quadrant_origin_drag_moves_the_split_and_keeps_the_partition():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed)
        ed._on_press(_ev(ed, 300.0, 300.0, dblclick=True))
        ed._redraw_only_gates()
        ed._on_press(_ev(ed, 300.0, 300.0))             # grab the centre
        assert ed._drag_state[0] == 'quad_origin'
        ed._on_motion(_ev(ed, 900.0, 1200.0))
        ed._on_release(_ev(ed, 900.0, 1200.0))
        quads, hits = _hits(ed, df)
        assert (hits == 1).all()
        by = {g['label'][:3]: g for g in quads}
        x, y = df.CD3.values, df.CD4.values
        import openflo.pipeline as fp
        assert (fp.gate_to_mask(by['Q+-'], df) == ((x >= 900) & (y < 1200))).all()
        assert (fp.gate_to_mask(by['Q-+'], df) == ((x < 900) & (y >= 1200))).all()
    finally:
        root.destroy()


def test_quadrant_tool_partitions_the_events_on_log_axes():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale='log')
        assert ed.ax.get_xscale() == 'log'
        ed._on_press(_ev(ed, 300.0, 300.0, dblclick=True))
        _quads, hits = _hits(ed, df)
        assert (hits == 1).all(), int((hits == 0).sum())
    finally:
        root.destroy()


def test_open_quadrant_sides_do_not_drag_the_view_out():
    # The open sides are +-1e12. Added with add_patch they entered dataLim,
    # and the replotted linear view spanned +-1.1e12 (measured), every event
    # in one pixel -- the case for an imported QuadrantGate too.
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        ed.canvas.draw()
        before = ed.ax.get_xlim(), ed.ax.get_ylim()
        ed._on_press(_ev(ed, 300.0, 300.0, dblclick=True))
        ed._replot()
        ed.canvas.draw()
        assert (ed.ax.get_xlim(), ed.ax.get_ylim()) == before
        assert len(ed._shape_artists) == 4             # still drawn
    finally:
        root.destroy()


def test_quadrant_tool_partitions_the_events_after_zooming_in():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed)
        ed.ax.set_xlim(-1000, 5000)
        ed.ax.set_ylim(-1000, 5000)
        ed._on_press(_ev(ed, 300.0, 300.0, dblclick=True))
        _quads, hits = _hits(ed, df)
        assert (hits == 1).all(), int((hits == 0).sum())
    finally:
        root.destroy()


def _on_screen(ed, artist):
    """``artist``'s outline in display pixels, as Agg is handed it."""
    path = artist.get_path().transformed(artist.get_patch_transform()
                                         + ed.ax.transData)
    return path.vertices


def _near_outline(points, outlines, px=1.0):
    """Whether each display point lies within ``px`` of a segment of one of
    the closed ``outlines``."""
    near = np.zeros(len(points), dtype=bool)
    for v in outlines:
        a, ab = v[:-1], v[1:] - v[:-1]
        for p_i, p in enumerate(points):
            t = np.clip(((p - a) * ab).sum(axis=1)
                        / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
            near[p_i] |= bool(np.hypot(*(p - (a + t[:, None] * ab)).T).min() <= px)
    return near


@pytest.mark.parametrize('view', [None, 'symlog', 'linear'])
def test_quadrant_outlines_are_drawn_on_a_logicle_channel_s_views(view):
    """Fluor data is stored logicle-baked, so every view of it is a FuncScale,
    and the open sides' ±1e12 went through it to +inf (log view), ±inf
    (symlog) or ~1e304 px (linear intensity). Agg drops such segments: on the
    default view Q++ had 2 of its 5 screen vertices finite and only the
    L below and left of the origin was drawn; on symlog, no outline at all.
    The partition was right throughout; the dividers the user sees and
    grabs were not there."""
    from openflo.pipeline import transform_values
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale=view, transform='logicle')
        assert not ed.ax.transData.is_affine
        c = float(transform_values(np.array([300.0]), method='logicle')[0])
        ed._on_press(_ev(ed, c, c, dblclick=True))
        quads, hits = _hits(ed, df)
        assert (hits == 1).all()
        ed._replot()
        ed.canvas.draw()
        outlines = [_on_screen(ed, ed._shape_artists[gid])
                    for gid, g in ed._gates.items() if g.get('quad_set')]
        assert len(outlines) == 4
        for v in outlines:
            assert np.isfinite(v).all() and np.abs(v).max() < 1e5, v
        # The four dividers, from the origin out to each side of the axes.
        o = ed.ax.transData.transform((c, c))
        bb = ed.ax.bbox
        for end in ((o[0], bb.y1 - 2), (o[0], bb.y0 + 2),
                    (bb.x0 + 2, o[1]), (bb.x1 - 2, o[1])):
            line = o + np.linspace(0.05, 1.0, 20)[:, None] * (np.asarray(end) - o)
            assert _near_outline(line, outlines).all(), end
    finally:
        root.destroy()


def test_an_open_bound_draws_no_line_and_leaves_the_view_alone():
    """An imported max-only range is an interval with lo = -1e12. Its line
    at -1e12 put that in the autoscale, and a linear view of the channel
    spanned -1.05e12..5e10 with every event in one pixel; on a logicle
    channel's log view it drew a stray line at the axes edge."""
    from openflo.pipeline import transform_values
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale='log', transform='logicle')
        ed.x_combo.set('FSC-A')
        ed.y_combo.set('SSC-A')
        ed._channel_scale['FSC-A'] = ed._channel_scale['SSC-A'] = 'linear'
        ed._replot()
        ed.canvas.draw()
        before = ed.ax.get_xlim()
        fsc = ed._add_gate({'kind': 'interval', 'channel': 'FSC-A',
                            'lo': -1e12, 'hi': 5e4, 'parent_id': None})
        hi = float(transform_values(np.array([1000.0]), method='logicle')[0])
        cd3 = ed._add_gate({'kind': 'interval', 'channel': 'CD3',
                            'lo': -1e12, 'hi': hi, 'parent_id': None})
        ed._replot()
        ed.canvas.draw()
        assert ed.ax.get_xlim() == before
        assert set(ed._vlines) == {f'{fsc}:hi'}
        ed.x_combo.set('CD3')
        ed.y_combo.set('CD4')
        ed._replot()
        ed.canvas.draw()
        assert set(ed._vlines) == {f'{cd3}:hi'}
    finally:
        root.destroy()


@pytest.mark.parametrize('chans', [('CD3', 'CD4'), ('FSC-A', 'SSC-A')])
def test_a_polygon_running_out_to_an_open_side_is_drawn_finite_in_view(chans):
    """A FlowJo ellipse cut at a biex axis end is imported as a polygon whose
    cut runs out to ±1e12: on a logicle channel's log view those vertices
    had no finite pixel, and on a linear view add_patch put 1e12 in the
    autoscale."""
    from openflo.pipeline import transform_values
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale='log', transform='logicle')
        x, y = chans
        ed.x_combo.set(x)
        ed.y_combo.set(y)
        if x == 'FSC-A':
            ed._channel_scale['FSC-A'] = ed._channel_scale['SSC-A'] = 'linear'
            lo, hi = 5e4, 1e5
        else:
            lo, hi = (float(v) for v in transform_values(
                np.array([100.0, 2000.0]), method='logicle'))
        ed._replot()
        ed.canvas.draw()
        before = ed.ax.get_xlim(), ed.ax.get_ylim()
        gid = ed._add_gate({'kind': 'polygon', 'x_channel': x, 'y_channel': y,
                            'vertices': [[lo, lo], [1e12, lo], [1e12, hi],
                                         [lo, hi]], 'parent_id': None})
        ed._replot()
        ed.canvas.draw()
        assert (ed.ax.get_xlim(), ed.ax.get_ylim()) == before
        v = _on_screen(ed, ed._shape_artists[gid])
        assert np.isfinite(v).all() and np.abs(v).max() < 1e5, v
        # Its three finite sides are drawn where they are.
        corners = ed.ax.transData.transform([[lo, lo], [lo, hi]])
        assert _near_outline(corners, [v]).all()
    finally:
        root.destroy()
