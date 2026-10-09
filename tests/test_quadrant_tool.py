"""The editor's quadrant tool (double-click with the Quadrant tool active).

Coverage before: no test drives _on_press's quadrant branch
(editor_gating.py:633-681); tests/test_ellipsoid_quadrant.py checks only the
WSP-imported quadrant, whose open sides are +-1e12.

Ground truth: a quadrant split PARTITIONS the parent -- every event in exactly
one of Q++ / Q+- / Q-+ / Q--. The tool used to build the outer sides from the
axis VIEW at the moment of the click (ax.get_xlim / get_ylim), so events
outside the view belonged to no quadrant. On the editor's default log axes
that was every event <= 0, i.e. the compensated negatives (37% of events in
this fixture). The outer sides are now the same open +-1e12 the importer uses.
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


def _setup(ed, scale='linear'):
    rng = np.random.default_rng(3)
    n = 8000
    neg = rng.normal(0, 150, (n // 2, 2))               # compensated negatives
    pos = 10 ** rng.normal(3.3, 0.35, (n // 2, 2))
    xy = np.vstack([neg, pos])
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


def test_quadrant_tool_partitions_the_events_on_default_log_axes():
    root, ed = _editor_or_skip()
    try:
        df = _setup(ed, scale=None)
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
