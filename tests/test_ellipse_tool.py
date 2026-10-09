"""The editor's ellipse tool: drawing, rendering, and the three edit drags.

Coverage before: _on_ellipse_select ran 1 of 16 statements, _ellipse_patch 1
of 7, and the ellipse translate / rim / rotate branches of _on_motion none.

Ground truth: the tool's own contract -- a drag from corner to corner commits
the ellipse inscribed in that box, ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1; a rim
drag puts the rim under the cursor; a rotate drag turns the covariance by the
swept angle; a translate drag moves the mean by the cursor delta.

Guarded by the same Tk-availability helper as tests/test_gui_ux.py.
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


def _setup(ed, scale: str | None = 'linear', seed=0, transform=None):
    """CD3 / CD4 span 10..1e4. With ``transform`` they are stored in that
    transform's units, as the loader stores fluor channels, and the default
    view (scale None) is the log of the intensity: a FuncScale."""
    rng = np.random.default_rng(seed)
    n = 4000
    df = pd.DataFrame({'FSC-A': rng.uniform(1e3, 1e5, n),
                       'SSC-A': rng.uniform(1e3, 1e5, n),
                       'CD3': _stored(10 ** rng.uniform(1, 4, n), transform),
                       'CD4': _stored(10 ** rng.uniform(1, 4, n), transform)})
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
    if transform:
        ed._channel_transform.update({'CD3': transform, 'CD4': transform})
    ed._populate_channel_combos()
    ed._set_active_sample('s1')
    ed.x_combo.set('CD3')
    ed.y_combo.set('CD4')
    ed.mode_var.set('dot')
    if scale:
        ed._channel_scale['CD3'] = scale
        ed._channel_scale['CD4'] = scale
    ed._replot()
    return df


def _stored(intensity, transform):
    if not transform:
        return np.asarray(intensity, dtype=float)
    from openflo.pipeline import transform_values
    return transform_values(np.asarray(intensity, dtype=float), method=transform)


def _ev(ed, x, y, **kw):
    px, py = ed.ax.transData.transform((x, y))
    return SimpleNamespace(button=1, key=None, dblclick=False, inaxes=ed.ax,
                           xdata=x, ydata=y, x=px, y=py, **kw)


def _grid(x0, x1, y0, y1, n=201):
    gx, gy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return pd.DataFrame({'CD3': gx.ravel(), 'CD4': gy.ravel()})


def _drawn_vs_selected(ed, gid, n=241):
    """Disagreement between the DRAWN patch (screen, what the renderer does)
    and the gate's selection, on a pixel grid around the patch."""
    import openflo.pipeline as fp
    ed._redraw_only_gates()
    ed.canvas.draw()
    patch = ed._shape_artists[gid]
    tr = patch.get_transform()
    tpath = tr.get_affine().transform_path(tr.transform_path_non_affine(patch.get_path()))
    ext = tpath.get_extents()
    pad = 0.15 * max(ext.width, ext.height)
    gx, gy = np.meshgrid(np.linspace(ext.x0 - pad, ext.x1 + pad, n),
                         np.linspace(ext.y0 - pad, ext.y1 + pad, n))
    disp = np.column_stack([gx.ravel(), gy.ravel()])
    drawn = tpath.contains_points(disp)
    data = ed.ax.transData.inverted().transform(disp)
    sel = fp.gate_to_mask(ed._gates[gid], pd.DataFrame({'CD3': data[:, 0],
                                                        'CD4': data[:, 1]}))
    return int((drawn != sel).sum()), int(sel.sum())


def _d2(gate, x, y):
    d = np.array([x, y]) - np.asarray(gate['mean'])
    return float(d @ np.linalg.inv(gate['cov']) @ d)


def test_ellipse_tool_commits_the_box_inscribed_ellipse():
    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        before = set(ed._gates)
        # Dragged from the upper-right corner to the lower-left one.
        ed._on_ellipse_select(SimpleNamespace(xdata=6000.0, ydata=4000.0),
                              SimpleNamespace(xdata=2000.0, ydata=3000.0))
        (gid,) = [g for g in ed._gates if g not in before]
        g = ed._gates[gid]
        assert g['kind'] == 'ellipsoid'
        assert (g['x_channel'], g['y_channel']) == ('CD3', 'CD4')
        grid = _grid(1500, 6500, 2500, 4500)
        truth = (((grid.CD3 - 4000) / 2000) ** 2 + ((grid.CD4 - 3500) / 500) ** 2) <= 1
        got = fp.gate_to_mask(g, grid)
        assert (got != truth.values).sum() <= 2          # boundary ulp ties only
        assert 0.3 * len(grid) < got.sum() < 0.9 * len(grid)
        # A zero-height drag commits nothing.
        n = len(ed._gates)
        ed._on_ellipse_select(SimpleNamespace(xdata=10.0, ydata=50.0),
                              SimpleNamespace(xdata=90.0, ydata=50.0))
        assert len(ed._gates) == n
    finally:
        root.destroy()


def test_ellipse_patch_matches_selection_on_linear_axes():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = ed._add_gate({'kind': 'ellipsoid', 'x_channel': 'CD3',
                            'y_channel': 'CD4', 'mean': [5000.0, 5000.0],
                            'cov': [[4e6, 3e6], [3e6, 4e6]], 'distance_sq': 1.0,
                            'parent_id': None})
        bad, n_sel = _drawn_vs_selected(ed, gid)
        # Only pixels on the outline disagree (measured: 0.15%).
        assert bad < 0.01 * n_sel, (bad, n_sel)
    finally:
        root.destroy()


def test_ellipse_rim_drag_puts_the_rim_under_the_cursor():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = ed._add_gate({'kind': 'ellipsoid', 'x_channel': 'CD3',
                            'y_channel': 'CD4', 'mean': [5000.0, 5000.0],
                            'cov': [[1e6, 0.0], [0.0, 1e6]], 'distance_sq': 1.0,
                            'parent_id': None})
        ed._redraw_only_gates()
        ed._on_press(_ev(ed, 6000.0, 5000.0))          # on the rim (md = r0)
        assert ed._drag_state == ('ellipse_rim', gid)
        ed._on_motion(_ev(ed, 7500.0, 5000.0))         # pull outward
        ed._on_release(_ev(ed, 7500.0, 5000.0))
        g = ed._gates[gid]
        assert _d2(g, 7500.0, 5000.0) == pytest.approx(1.0, rel=1e-9)
        assert g['mean'] == [5000.0, 5000.0]
    finally:
        root.destroy()


def test_ellipse_rotate_drag_turns_the_covariance_by_the_swept_angle():
    from openflo.plotmath import ellipse_geom
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        cov0 = np.array([[4e6, 0.0], [0.0, 1e6]])
        gid = ed._add_gate({'kind': 'ellipsoid', 'x_channel': 'CD3',
                            'y_channel': 'CD4', 'mean': [5000.0, 5000.0],
                            'cov': cov0.tolist(), 'distance_sq': 1.0,
                            'parent_id': None})
        ed._redraw_only_gates()
        _c, _inv, _r0, (hx, hy) = ellipse_geom(ed._gates[gid])
        ed._on_press(_ev(ed, hx, hy))
        assert ed._drag_state == ('ellipse_rotate', gid)
        th = np.radians(30.0)
        a0 = np.arctan2(hy - 5000.0, hx - 5000.0)
        r = np.hypot(hx - 5000.0, hy - 5000.0)
        tx, ty = 5000.0 + r * np.cos(a0 + th), 5000.0 + r * np.sin(a0 + th)
        ed._on_motion(_ev(ed, tx, ty))
        ed._on_release(_ev(ed, tx, ty))
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        assert np.allclose(ed._gates[gid]['cov'], rot @ cov0 @ rot.T, rtol=1e-9)
    finally:
        root.destroy()


def test_ellipse_translate_drag_moves_the_mean_by_the_cursor_delta():
    root, ed = _editor_or_skip()
    try:
        _setup(ed)
        gid = ed._add_gate({'kind': 'ellipsoid', 'x_channel': 'CD3',
                            'y_channel': 'CD4', 'mean': [5000.0, 5000.0],
                            'cov': [[1e6, 2e5], [2e5, 1e6]], 'distance_sq': 1.0,
                            'parent_id': None})
        ed._redraw_only_gates()
        ed._on_press(_ev(ed, 5100.0, 4950.0))           # interior
        assert ed._drag_state == ('ellipse_translate', gid)
        ed._on_motion(_ev(ed, 5600.0, 4150.0))          # +500, -800
        ed._on_release(_ev(ed, 5600.0, 4150.0))
        g = ed._gates[gid]
        assert g['mean'] == pytest.approx([5500.0, 4200.0])
        assert g['cov'] == [[1e6, 2e5], [2e5, 1e6]]
    finally:
        root.destroy()


def test_ellipse_drawn_on_log_axes_selects_what_is_drawn():
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)                           # default: log
        assert ed.ax.get_xscale() == 'log'
        before = set(ed._gates)
        ed._on_ellipse_select(SimpleNamespace(xdata=30.0, ydata=30.0),
                              SimpleNamespace(xdata=3000.0, ydata=3000.0))
        (gid,) = [g for g in ed._gates if g not in before]
        bad, n_sel = _drawn_vs_selected(ed, gid)
        assert bad < 0.01 * n_sel, (bad, n_sel)
    finally:
        root.destroy()


def _display_path(artist):
    tr = artist.get_transform()
    return tr.get_affine().transform_path(
        tr.transform_path_non_affine(artist.get_path()))


def _outline_vs_selection(ed, tpath, gate, n=241):
    """Sample a pixel grid around the display-space outline ``tpath``: the
    points inside it against the points ``gate`` selects. Returns the number
    that disagree farther than half a pixel from the outline (where nothing
    is ambiguous) and the number selected."""
    import openflo.pipeline as fp
    ext = tpath.get_extents()
    pad = 0.15 * max(ext.width, ext.height)
    gx, gy = np.meshgrid(np.linspace(ext.x0 - pad, ext.x1 + pad, n),
                         np.linspace(ext.y0 - pad, ext.y1 + pad, n))
    disp = np.column_stack([gx.ravel(), gy.ravel()])
    data = ed.ax.transData.inverted().transform(disp)
    sel = np.asarray(fp.gate_to_mask(gate, pd.DataFrame(
        {gate['x_channel']: data[:, 0], gate['y_channel']: data[:, 1]})))
    bad = disp[tpath.contains_points(disp) != sel]
    v = tpath.vertices
    a, ab = v[:-1], v[1:] - v[:-1]
    far = 0
    for p in bad:
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12),
                    0.0, 1.0)
        far += np.hypot(*(p - (a + t[:, None] * ab)).T).min() > 0.5
    return int(far), int(sel.sum())


@pytest.mark.parametrize('transform', [None, 'logicle'])
def test_ellipse_on_a_log_view_is_an_ellipse_drawn_and_previewed_as_selected(
        transform, tmp_path):
    """On a log view (a linear-stored channel's log axis, or a logicle
    channel's log-of-intensity view) the tool keeps an ellipsoid gate -- the
    ellipse inscribed in the dragged box in data, with its resize / rotate
    handles and its FlowJo export -- and both its outline and the drag
    preview trace that ellipse's true boundary on the view, so the events
    inside what is drawn are the events it selects."""
    import flowio

    import openflo.pipeline as fp
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None, transform=transform)
        assert not ed.ax.transData.is_affine
        lo, hi = (float(v) for v in _stored([30.0, 3000.0], transform))
        before = set(ed._gates)
        ed._on_ellipse_select(SimpleNamespace(xdata=lo, ydata=lo),
                              SimpleNamespace(xdata=hi, ydata=hi))
        (gid,) = [g for g in ed._gates if g not in before]
        g = ed._gates[gid]
        assert g['kind'] == 'ellipsoid'
        c, r = (lo + hi) / 2, (hi - lo) / 2
        assert g['mean'] == [c, c] and g['distance_sq'] == 1.0
        assert g['cov'] == [[r * r, 0.0], [0.0, r * r]]
        ed._redraw_only_gates()
        ed.canvas.draw()
        far, n_sel = _outline_vs_selection(
            ed, _display_path(ed._shape_artists[gid]), g)
        assert far == 0 and n_sel > 1000, (far, n_sel)

        # The preview while dragging that box is the same boundary.
        ed.gate_tool_var.set('ellipse')
        ed._activate_gate_tool()
        sel = ed._selector
        sel.extents = (lo, hi, lo, hi)
        assert sel.extents == pytest.approx((lo, hi, lo, hi), rel=1e-12)
        far, _ = _outline_vs_selection(ed, _display_path(sel._selection_artist), g)
        assert far == 0, far

        # FlowJo export: a 64-vertex polygon, as for any ellipsoid gate.
        fcs = tmp_path / 's1.fcs'
        ev = ed._samples['s1'].data[['CD3', 'CD4']].to_numpy(np.float32)
        with open(fcs, 'wb') as fh:
            flowio.create_fcs(fh, ev.ravel().tolist(), ['CD3', 'CD4'])
        w = fp.WspWriter()
        w.add_sample('s1', str(fcs), ['CD3', 'CD4'], [dict(g, id=gid)])
        w.write(str(tmp_path / 'out.wsp'))
        xml = (tmp_path / 'out.wsp').read_text(encoding='utf-8')
        assert 'EllipsoidGate' not in xml
        assert xml.count('<gating:vertex') == 64
        assert any('exported as a 64-vertex polygon' in m for m in w.warnings)
    finally:
        root.destroy()


def test_ellipsoid_gate_outline_on_log_axes_is_the_region_it_selects():
    """An existing ellipsoid gate (a GMM auto-gate, an imported one) keeps
    its (mean, cov); only its outline is traced. The Ellipse patch disagreed
    with the selection on 1.97% of this one's area (measured)."""
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=None)
        gate = {'kind': 'ellipsoid', 'x_channel': 'CD3', 'y_channel': 'CD4',
                'mean': [1500.0, 1500.0], 'cov': [[4e5, 2e5], [2e5, 3e5]],
                'distance_sq': 4.0, 'parent_id': None}
        gid = ed._add_gate(dict(gate))
        bad, n_sel = _drawn_vs_selected(ed, gid)
        assert bad < 0.005 * n_sel, (bad, n_sel)
        assert ed._gates[gid]['mean'] == gate['mean']
        assert ed._gates[gid]['cov'] == gate['cov']
    finally:
        root.destroy()


@pytest.mark.parametrize('transform,scale', [(None, None), ('logicle', None),
                                             ('logicle', 'linear'),
                                             ('logicle', 'symlog')])
def test_ellipse_handles_are_hit_where_they_are_drawn_on_a_nonlinear_view(
        transform, scale):
    """The rim band (0.6..1.5 of the rim's Mahalanobis radius) and the
    rotation grip's tolerance were measured in data units. On a non-linear
    view the band reaches decades past one side, so a click 80 px off the
    outline took the rim. The grip, 18% of the height beyond the rim in
    data, can sit a few pixels from it there, so a click on the rim took
    rotate. Measured on screen now: the outline is the rim, the drawn grip
    rotates, the nearer of the two wins, and inside is translate."""
    from openflo.plotmath import ellipse_geom, ellipse_outline, ellipse_params
    root, ed = _editor_or_skip()
    try:
        _setup(ed, scale=scale, transform=transform)
        assert not ed.ax.transData.is_affine
        lo, hi = (float(v) for v in _stored([30.0, 3000.0], transform))
        before = set(ed._gates)
        ed._on_ellipse_select(SimpleNamespace(xdata=lo, ydata=lo),
                              SimpleNamespace(xdata=hi, ydata=hi))
        (gid,) = [g for g in ed._gates if g not in before]
        ed._redraw_only_gates()
        ed.canvas.draw()
        g = ed._gates[gid]
        tr = ed.ax.transData
        d = tr.transform(ellipse_outline(ellipse_params(g), tr.transform))

        def hit(px):
            h = ed._hit_test(_ev(ed, *tr.inverted().transform(px)))
            return h[0] if h else None

        for k, out in ((np.argmin(d[:, 0]), (-80, 0)), (np.argmax(d[:, 0]), (80, 0)),
                       (np.argmax(d[:, 1]), (0, 80)), (np.argmin(d[:, 1]), (0, -80))):
            assert hit(d[k]) == 'ellipse_rim', (k, d[k])
            assert hit(d[k] + np.array(out, float)) != 'ellipse_rim', (k, out)
        _c, _inv, _r0, grip = ellipse_geom(g)
        assert hit(tr.transform(grip)) == 'ellipse_rotate'
        assert hit(tr.transform(g['mean'])) == 'ellipse_translate'
    finally:
        root.destroy()
