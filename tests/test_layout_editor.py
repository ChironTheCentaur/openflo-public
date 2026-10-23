"""The layout editor (beta): the page model, re-applying one layout to
samples whose gate ids differ, the rendered numbers, and the window's drag,
resize and batch export. Synthetic samples and generic channels only."""
from __future__ import annotations

import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openflo import layout_page as lp  # noqa: E402
from openflo.async_task import wait_for_job  # noqa: E402
from tests.conftest import gui_unavailable  # noqa: E402

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']


# ── page model (no Tk) ───────────────────────────────────────────────────

def test_clamp_keeps_boxes_on_the_page_and_above_the_minimum():
    assert lp.clamp_box(0.9, 0.9, 0.3, 0.3) == pytest.approx((0.7, 0.7, 0.3, 0.3))
    x, y, w, h = lp.clamp_box(-1, -1, 0, 0)
    assert (x, y) == (0.0, 0.0) and w == h == lp.MIN_SIZE


def test_move_and_resize_snap_and_stay_on_the_page():
    lay = lp.Layout()
    iid = lay.add(lp.plot_item(box=(0.1, 0.1, 0.3, 0.3)))
    lay.move(iid, 0.013, 0.0, step=0.05)
    assert lay.items[iid]['box'][0] == pytest.approx(0.1)   # snapped back
    lay.move(iid, 5.0, 5.0)
    x, y, w, h = lay.items[iid]['box']
    assert (x + w, y + h) == pytest.approx((1.0, 1.0))
    lay.resize(iid, -1.0, -1.0)
    assert lay.items[iid]['box'][2:] == pytest.approx([lp.MIN_SIZE] * 2)
    # The top-left corner stays put while resizing.
    assert lay.items[iid]['box'][:2] == pytest.approx([x, y])


def test_hit_returns_the_topmost_item():
    lay = lp.Layout()
    a = lay.add(lp.plot_item(box=(0.0, 0.0, 0.5, 0.5)))
    b = lay.add(lp.stats_item(box=(0.25, 0.25, 0.5, 0.5)))
    assert lay.hit(0.3, 0.3) == b
    lay.raise_to_top(a)
    assert lay.hit(0.3, 0.3) == a
    assert lay.hit(0.9, 0.05) is None


def test_next_slot_does_not_stack_new_items_on_old_ones():
    lay = lp.Layout()
    corners = set()
    for _ in range(6):
        box = lay.next_slot(0.3, 0.3)
        corners.add((round(box[0], 3), round(box[1], 3)))
        lay.add(lp.plot_item(box=box))
    assert len(corners) == 6


def test_axes_rect_flips_to_matplotlibs_bottom_left_origin():
    assert lp.axes_rect((0.1, 0.2, 0.3, 0.4)) == pytest.approx(
        [0.1, 0.4, 0.3, 0.4])


def test_layout_round_trips_and_refuses_foreign_or_newer_files():
    doc = lp.LayoutDoc(export_dpi=600)
    lay = doc.pages[0]
    lay.page = 'A4 portrait'
    lay.add(lp.plot_item(population='n:cells', x='FL1-A', mode='histogram',
                         xscale='log', xlim=(1.0, 2.0)))
    lay.add(lp.stats_item(populations=['n:cells/n:pos'], stats=['Count']))
    lay.add(lp.chart_item('bar', populations=['n:cells'], value='%Parent'))
    lay.add(lp.text_item('Run {sample}'))
    doc.add_page()
    back = lp.LayoutDoc.loads(doc.dumps())
    assert back.export_dpi == 600
    assert [p.name for p in back.pages] == [p.name for p in doc.pages]
    assert back.pages[0].page == 'A4 portrait'
    assert back.pages[0].order == lay.order
    assert back.pages[0].items == lay.items
    with pytest.raises(ValueError):
        lp.LayoutDoc.from_dict({'format': 'something-else'})
    d = doc.to_dict()
    d['version'] = lp.VERSION + 1
    with pytest.raises(ValueError):
        lp.LayoutDoc.from_dict(d)


def test_a_version_1_single_page_file_still_opens():
    v1 = {'format': lp.FORMAT, 'version': 1, 'page': 'A4 landscape',
          'items': [dict(lp.text_item('x'), id='item9')]}
    doc = lp.LayoutDoc.from_dict(v1)
    assert len(doc.pages) == 1 and doc.pages[0].page == 'A4 landscape'
    assert list(doc.pages[0].items) == ['item9']
    assert doc.export_dpi == lp.EXPORT_DPI == 300
    # One constant, not two copies.
    from openflo import theme
    assert theme.EXPORT_DPI is lp.EXPORT_DPI
    assert theme.EXPORT_DPI_CHOICES is lp.EXPORT_DPI_CHOICES


def test_pages_copy_import_and_keep_one():
    doc = lp.LayoutDoc()
    iid = doc.pages[0].add(lp.text_item('a', box=(0.1, 0.2, 0.3, 0.1)))
    j = doc.add_page()
    new = doc.copy_item(0, iid, j)
    assert doc.pages[j].items[new]['box'] == doc.pages[0].items[iid]['box']
    other = lp.LayoutDoc([lp.Layout(name='Summary')])
    idx = doc.import_pages(other, prefix='other: ')
    assert doc.pages[idx[0]].name == 'other: Summary'
    assert len({p.name for p in doc.pages}) == len(doc.pages)
    dup = doc.duplicate_page(0)
    assert doc.pages[dup].items == doc.pages[0].items
    while len(doc.pages) > 1:
        doc.remove_page(0)
    doc.remove_page(0)                          # the last page stays
    assert len(doc.pages) == 1


def test_soft_snap_pulls_to_near_edges_only():
    lay = lp.Layout()
    lay.add(lp.plot_item(box=(0.5, 0.5, 0.2, 0.2)))
    targets = lay.edges()
    # Left edge 0.006 from the other item's right edge (0.7): pulled on.
    box, (gx, gy) = lp.soft_snap((0.706, 0.1, 0.2, 0.2), 'move', targets, 0.01)
    assert box[0] == pytest.approx(0.7) and gx == pytest.approx(0.7)
    assert gy is None
    # Out of reach: left alone.
    box, (gx, _gy) = lp.soft_snap((0.73, 0.1, 0.2, 0.2), 'move', targets, 0.01)
    assert box[0] == pytest.approx(0.73) and gx is None
    # Resize pulls the right/bottom edge, the top-left stays.
    box, _g = lp.soft_snap((0.1, 0.1, 0.395, 0.3), 'resize', targets, 0.01)
    assert box[:2] == pytest.approx([0.1, 0.1])
    assert box[0] + box[2] == pytest.approx(0.5)
    assert lp.grid_lines(0.25) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])


def _tree(ids):
    """Cells (rect) -> Pos (threshold) under the given gate ids."""
    cells, pos = ids
    return {cells: {'kind': 'rect', 'name': 'Cells', 'x_channel': 'FSC-A',
                    'y_channel': 'SSC-A', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1,
                    'parent_id': None},
            pos: {'kind': 'threshold', 'name': 'Pos', 'channel': 'FL1-A',
                  'op': '>', 'value': 0.5, 'parent_id': cells}}


def test_population_resolves_by_name_whatever_the_gate_ids():
    a, b = _tree(('g1', 'g2')), _tree(('g7', 'g3'))
    key = lp.population_choices(a, ['g1', 'g2'], {})[2][0]
    assert lp.resolve_population(a, {}, key) == 'g2'
    assert lp.resolve_population(b, {}, key) == 'g3'
    assert lp.resolve_population(b, {}, lp.ALL_EVENTS) is None
    with pytest.raises(KeyError):
        lp.resolve_population({}, {}, key)


def test_child_gates_are_the_gates_drawn_on_a_populations_plot():
    t = _tree(('g1', 'g2'))
    assert set(lp.child_gates(t, None)) == {'g1'}
    assert set(lp.child_gates(t, 'g1')) == {'g2'}
    assert lp.child_gates(t, 'g2') == {}


def test_fill_text_and_safe_filename():
    assert lp.fill_text('Run {sample} {x}', 's1') == 'Run s1 {x}'
    assert lp.safe_filename('a/b:c d') == 'a_b_c_d'
    assert lp.safe_filename('...') == 'sample'


# ── rendering through a real editor ──────────────────────────────────────

def _editor_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available')
    try:
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001
        gui_unavailable(f'Tk cannot initialise without a display: {e}')
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _load(ed, name, seed, n, pos_cut, extra_gate=False):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0, 1000, (n, len(COLS))), columns=COLS)
    s = FlowSample.from_dataframe(df, name=name, path=f'{name}.fcs')
    ed._samples[name] = s
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._set_active_sample(name)
    if extra_gate:          # shifts this sample's gate ids off the other's
        ed._add_gate({'kind': 'threshold', 'channel': 'FL2-A', 'op': '>',
                      'value': 10.0, 'name': 'Other', 'parent_id': None},
                     audit=False)
    cells = ed._add_gate({'kind': 'rect', 'name': 'Cells',
                          'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                          'x0': 100.0, 'x1': 900.0, 'y0': 100.0, 'y1': 900.0,
                          'parent_id': None}, audit=False)
    pos = ed._add_gate({'kind': 'threshold', 'name': 'Pos',
                        'channel': 'FL1-A', 'op': '>', 'value': pos_cut,
                        'parent_id': cells}, audit=False)
    return cells, pos


def _expected(ed, name, gid):
    from openflo.gating import gate_counts
    return gate_counts(ed._sample_gates[name], gid, ed._samples[name].data)


def _table_text(fig):
    out = []
    for ax in fig.axes:
        for t in ax.tables:
            cells = t.get_celld()
            rows = sorted({r for r, _c in cells})
            cols = sorted({c for _r, c in cells})
            out.append([[cells[r, c].get_text().get_text() for c in cols]
                        for r in rows])
    return out


def test_one_layout_renders_each_samples_own_numbers():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext, render_page
        c1, p1 = _load(ed, 's1', 1, 4000, 300.0)
        c2, p2 = _load(ed, 's2', 2, 3000, 700.0, extra_gate=True)
        assert (c1, p1) != (c2, p2), 'the test needs differing gate ids'
        ed._set_active_sample('s1')
        keys = SampleContext(ed, 's1').keys
        lay = lp.Layout()
        lay.add(lp.plot_item(population=keys[c1], x='FL1-A', y='FL2-A',
                             mode='dot', color='By sample',
                             box=(0.0, 0.0, 0.5, 0.5)))
        lay.add(lp.stats_item(populations=[keys[c1], keys[p1]],
                              stats=['Count', '%Parent'],
                              box=(0.5, 0.0, 0.5, 0.3)))
        live = (ed.ax, ed.fig, ed._gates)
        for name, cells, pos in (('s1', c1, p1), ('s2', c2, p2)):
            fig = render_page(ed, lay, name)
            n_cells = _expected(ed, name, cells)[0]
            n_pos, n_par, _ = _expected(ed, name, pos)
            (table,) = _table_text(fig)
            assert table[1][:2] == ['Cells', f'{n_cells:,}']
            assert table[2][:3] == ['Pos', f'{n_pos:,}',
                                    f'{100.0 * n_pos / n_par:.2f}']
            # The plot shows exactly the Cells population of THIS sample.
            plot_ax = fig.axes[0]
            n_drawn = sum(len(c.get_offsets()) for c in plot_ax.collections)
            assert n_drawn == n_cells
            # Its child gate is drawn with its % of parent.
            labels = [t.get_text() for t in plot_ax.texts]
            assert any(t.startswith('Pos\n') for t in labels), labels
        # The live plot's state is untouched.
        assert (ed.ax, ed.fig, ed._gates) == live
        assert getattr(ed, '_render_population', 'auto') == 'auto'
    finally:
        root.destroy()


def test_a_population_missing_from_a_sample_is_said_not_drawn_wrong():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import render_page
        _load(ed, 's1', 1, 500, 300.0)
        lay = lp.Layout()
        lay.add(lp.plot_item(population='n:not-here', x='FL1-A',
                             mode='histogram'))
        fig = render_page(ed, lay, 's1')
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any('no such population' in t for t in texts), texts
    finally:
        root.destroy()


def test_a_new_layout_starts_at_the_preferences_dpi(monkeypatch):
    """A new layout exports at the user's Preferences dpi (theme.export_dpi),
    not a constant 300; a layout the editor already holds keeps its own."""
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        from openflo import theme
        # Not the full 1240x860 window in a test.
        monkeypatch.setattr(ule.LayoutEditorWindow, 'geometry',
                            lambda self, *a, **k: None)
        store = {'export_provenance': False}
        monkeypatch.setattr(theme, 'read_prefs', lambda: dict(store))
        wins = []
        for dpi in (600, None):
            store.pop('export_dpi', None)
            if dpi is not None:
                store['export_dpi'] = dpi
            ed._layout_doc = None
            wins.append(ule.LayoutEditorWindow(ed))
            wins[-1].withdraw()
            want = dpi or 300
            assert wins[-1].doc.export_dpi == want
            assert wins[-1].dpi_var.get() == str(want)
        store['export_dpi'] = 600
        ed._layout_doc = lp.LayoutDoc(export_dpi=150)
        wins.append(ule.LayoutEditorWindow(ed))
        wins[-1].withdraw()
        assert wins[-1].doc.export_dpi == 150
        for w in wins:
            w.destroy()
    finally:
        root.destroy()


def test_window_drag_resize_and_batch_export(tmp_path, monkeypatch):
    root, ed = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import openflo.ui_layout_editor as ule
        _load(ed, 's1', 1, 800, 300.0)
        _load(ed, 's2', 2, 600, 500.0, extra_gate=True)
        win = ule.LayoutEditorWindow(ed)
        win.update()
        win._add_text()
        iid = win._sel
        before = list(win.layout.items[iid]['box'])

        def ev(x, y):
            return SimpleNamespace(x=x, y=y)
        x0, y0, x1, y1 = win._box_px(before)
        win._on_press(ev(x0 + 5, y0 + 5))
        win._on_drag(ev(x0 + 105, y0 + 65))
        win._on_release(ev(x0 + 105, y0 + 65))
        moved = win.layout.items[iid]['box']
        assert moved[0] > before[0] and moved[1] > before[1]
        assert moved[2:] == pytest.approx(before[2:])

        x0, y0, x1, y1 = win._box_px(moved)
        win._on_press(ev(x1 - 2, y1 - 2))           # the corner handle
        win._on_drag(ev(x1 + 80, y1 + 40))
        win._on_release(ev(x1 + 80, y1 + 40))
        grown = win.layout.items[iid]['box']
        assert grown[:2] == pytest.approx(moved[:2])
        assert grown[2] > moved[2] and grown[3] > moved[3]

        out = tmp_path / 'batch.pdf'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        win._export_batch()
        assert wait_for_job(win._busy)
        data = out.read_bytes()
        assert data.startswith(b'%PDF')
        assert b'/Count 2' in data              # a page per sample

        png = tmp_path / 'each.png'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(png))
        win._export_batch()
        assert wait_for_job(win._busy)
        assert sorted(p.name for p in tmp_path.glob('each_*.png')) == [
            'each_s1.png', 'each_s2.png']

        lay_path = tmp_path / 'my.oflayout'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(lay_path))
        win._save_layout()
        win.layout.remove(iid)
        monkeypatch.setattr(ule.filedialog, 'askopenfilename',
                            lambda *a, **k: str(lay_path))
        win._open_layout()
        assert list(win.layout.items) == [iid]
        assert ed._layout_doc is win.doc
        win.destroy()
    finally:
        root.destroy()


# ── round 2: overlays, kept zoom, charts, clusters, drops, pages ─────────

def _no_downsample(ed):
    for v in ('ds_display_var', 'ds_propagate_var'):
        var = getattr(ed, v, None)
        if var is not None:
            var.set(False)


def _n_drawn(ax):
    return sum(len(c.get_offsets()) for c in ax.collections)


def test_one_plot_overlays_several_samples_and_populations():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext, render_page
        c1, p1 = _load(ed, 's1', 1, 900, 300.0)
        c2, p2 = _load(ed, 's2', 2, 700, 600.0, extra_gate=True)
        _no_downsample(ed)
        keys = SampleContext(ed, 's1').keys
        lay = lp.Layout()
        lay.add(lp.plot_item(populations=[keys[c1], keys[p1]],
                             samples=['s1', 's2'], x='FL1-A', y='FL2-A',
                             box=(0, 0, 0.5, 0.5)))
        fig = render_page(ed, lay, 's1')
        want = sum(_expected(ed, s, g)[0]
                   for s, g in (('s1', c1), ('s1', p1), ('s2', c2),
                                ('s2', p2)))
        assert _n_drawn(fig.axes[0]) == want
        legend = fig.axes[0].get_legend()
        assert legend is not None and len(legend.get_texts()) == 4
    finally:
        root.destroy()


def test_a_plot_keeps_its_zoom_and_axis_type_and_offers_fit():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import OUTSIDE_WARN, RenderContext, render_page
        c1, _p1 = _load(ed, 's1', 1, 900, 300.0)
        _no_downsample(ed)
        key = RenderContext(ed, 's1')('s1').keys[c1]
        ed._channel_scale['FL1-A'] = 'log'
        before = (dict(ed._channel_scale), dict(ed._channel_range))
        lay = lp.Layout()
        iid = lay.add(lp.plot_item(population=key, x='FL1-A', y='FL2-A',
                                   xscale='linear', xlim=(0.0, 300.0),
                                   ylim=(0.0, 1000.0)))
        rc = RenderContext(ed, 's1')
        fig = render_page(ed, lay, 's1', rc=rc)
        ax = fig.axes[0]
        assert ax.get_xscale() == 'linear'
        assert ax.get_xlim() == pytest.approx((0.0, 300.0))
        # Uniform 0..1000 on FL1-A: about 70% sits right of 300.
        assert rc.info[iid]['outside'] == pytest.approx(0.7, abs=0.06)
        assert rc.info[iid]['outside'] > OUTSIDE_WARN
        # The live per-channel settings are back as they were.
        assert (dict(ed._channel_scale), dict(ed._channel_range)) == before
        # Without a kept zoom there is nothing to offer.
        lay.items[iid]['xlim'] = lay.items[iid]['ylim'] = None
        rc = RenderContext(ed, 's1')
        render_page(ed, lay, 's1', rc=rc)
        assert rc.info.get(iid, {}).get('outside', 0.0) == 0.0
    finally:
        root.destroy()


def test_charts_compare_populations_across_samples():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext, render_page
        c1, p1 = _load(ed, 's1', 1, 900, 300.0)
        _c2, p2 = _load(ed, 's2', 2, 700, 600.0, extra_gate=True)
        key = SampleContext(ed, 's1').keys[p1]
        lay = lp.Layout()
        lay.add(lp.chart_item('bar', populations=[key], samples=['s1', 's2'],
                              value='%Parent', box=(0, 0, 0.5, 0.5)))
        lay.add(lp.chart_item('violin', populations=[key],
                              samples=['s1', 's2'], channel='FL1-A',
                              box=(0.5, 0, 0.5, 0.5)))
        fig = render_page(ed, lay, 's1')
        bars = [p.get_height() for p in fig.axes[0].patches]
        want = []
        for s, g in (('s1', p1), ('s2', p2)):
            n, par, _ = _expected(ed, s, g)
            want.append(100.0 * n / par)
        assert bars == pytest.approx(want)
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [
            's1', 's2']
        assert fig.axes[1].collections, 'no violins drawn'
    finally:
        root.destroy()


def test_cluster_results_colour_a_plot_and_are_populations():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import (
            CAT_PREFIX,
            SampleContext,
            label_columns,
            render_page,
        )
        _load(ed, 's1', 1, 600, 300.0)
        s = ed._samples['s1']
        s.data['cluster'] = np.arange(len(s.data)) % 3
        ed._set_active_sample('s1')
        cid = ed._add_gate({'kind': 'cluster', 'channel': 'cluster',
                            'cluster_id': 1, 'name': 'Cluster 1',
                            'color': '#123456', 'parent_id': None},
                           audit=False)
        assert label_columns(ed) == ['cluster']
        ctx = SampleContext(ed, 's1')
        paths = [p for _k, p in lp.population_choices(ctx.gates, ctx.order,
                                                      ctx.labels)]
        assert 'Cluster 1' in paths
        lay = lp.Layout()
        lay.add(lp.plot_item(x='FL1-A', y='FL2-A',
                             color=f'{CAT_PREFIX}cluster'))
        lay.add(lp.stats_item(populations=[ctx.keys[cid]], stats=['Count'],
                              box=(0.5, 0, 0.4, 0.2)))
        fig = render_page(ed, lay, 's1')
        leg = fig.axes[0].get_legend()
        assert leg is not None and leg.get_title().get_text() == 'cluster'
        assert sorted(t.get_text() for t in leg.get_texts()) == ['0', '1', '2']
        # The imported cluster's own colour is used for its events.
        import matplotlib.colors as mc
        cols = {mc.to_hex(c) for c in fig.axes[0].collections[0]
                .get_facecolors()}
        assert '#123456' in cols
        (table,) = _table_text(fig)
        assert table[1] == ['Cluster 1', f'{(s.data.cluster == 1).sum():,}']
    finally:
        root.destroy()


def test_drops_from_the_tree_land_as_plots_and_join_items(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        import openflo.prefs as prefs
        import openflo.ui_layout_editor as ule
        from openflo.tree_ids import gate_iid
        c1, p1 = _load(ed, 's1', 1, 600, 300.0)
        c2, _p2 = _load(ed, 's2', 2, 500, 600.0, extra_gate=True)
        monkeypatch.setattr(prefs, 'read_prefs', lambda: {})
        # The live plot while Cells of s1 was selected.
        ed._viewed_population = ('s1', c1)
        ed.x_combo.set('FL1-A')
        ed.y_combo.set('FL2-A')
        ed.mode_var.set('dot')
        ed._remember_population_view()
        ed._viewed_population = None
        ed.x_combo.set('FSC-A')                 # what shows now differs
        ed.y_combo.set('SSC-A')
        assert ed._view_for_population('s1', c1)['x'] == 'FL1-A'
        monkeypatch.setattr(prefs, 'read_prefs',
                            lambda: {'layout_drop_view': 'current'})
        assert ed._view_for_population('s1', c1)['x'] == 'FSC-A'
        monkeypatch.setattr(prefs, 'read_prefs', lambda: {})

        ed._open_layout_editor()
        win = ed._layout_window
        win.deiconify()
        win.update()
        win._fit()       # the page's place at the window's real size
        cv = win.canvas
        x0, y0, x1, y1 = win._box_px((0.0, 0.0, 1.0, 1.0))
        at = (cv.winfo_rootx() + int(x0 + 0.2 * (x1 - x0)),
              cv.winfo_rooty() + int(y0 + 0.3 * (y1 - y0)))
        assert ed._drop_to_layout([('s1', c1)], *at)
        (iid,) = win.layout.items
        it = win.layout.items[iid]
        assert it['kind'] == 'plot' and it['samples'] == []
        assert (it['x'], it['y']) == ('FL1-A', 'FL2-A')   # as last shown
        # A second sample's gate dropped on that plot overlays it there.
        win.update()
        bx0, by0, bx1, by1 = win._box_px(it['box'])
        at = (cv.winfo_rootx() + int((bx0 + bx1) / 2 - cv.canvasx(0)),
              cv.winfo_rooty() + int((by0 + by1) / 2 - cv.canvasy(0)))
        assert ed._drop_to_layout([('s2', c2)], *at)
        assert list(win.layout.items) == [iid]
        assert win.layout.items[iid]['samples'] == ['s1', 's2']
        # Outside the page: not consumed.
        assert not ed._drop_to_layout([('s1', p1)], cv.winfo_rootx() - 50,
                                      cv.winfo_rooty() - 50)
        # The tree drag reaches the same place.
        ev = type('E', (), {'x_root': at[0], 'y_root': at[1]})()
        ed._press_selection = ()
        assert ed._maybe_drop_to_layout(gate_iid('s1', p1), ev)
        assert len(win.layout.items[iid]['populations']) == 2
        assert isinstance(win, ule.LayoutEditorWindow)
    finally:
        root.destroy()


def test_export_uses_the_chosen_dpi_and_every_page(tmp_path, monkeypatch):
    root, ed = _editor_or_skip()
    try:
        import struct

        import openflo.ui_layout_editor as ule
        _load(ed, 's1', 1, 300, 300.0)
        win = ule.LayoutEditorWindow(ed)
        win.update()
        win._add_text()
        win._new_page()
        win._add_text()
        assert len(win.doc.pages) == 2
        win.dpi_var.set('150')
        win._on_dpi()
        assert win.doc.export_dpi == 150
        out = tmp_path / 'p.png'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        win._export_batch()
        assert wait_for_job(win._busy)
        files = sorted(tmp_path.glob('p_*.png'))
        assert [f.name for f in files] == ['p_s1_Page_1.png', 'p_s1_Page_2.png']
        w, h = struct.unpack('>II', files[0].read_bytes()[16:24])
        assert (w, h) == (int(11 * 150), int(8.5 * 150))
        win.destroy()
    finally:
        root.destroy()


# ── round 3: zoom inside plots, plots feeding tables, colours ────────────

def test_layer_colours_natural_override_and_per_item(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        import matplotlib.colors as mc

        import openflo.layout_render as lr
        c1, p1 = _load(ed, 's1', 1, 400, 300.0)
        _no_downsample(ed)
        ed._sample_gates['s1'][c1]['color'] = '#111111'
        ed._sample_gates['s1'][p1]['color'] = '#222222'
        keys = lr.SampleContext(ed, 's1').keys
        item = lp.plot_item(populations=[keys[c1], keys[p1]], x='FL1-A',
                            mode='histogram')
        lay = lp.Layout()
        iid = lay.add(item)

        def line_colours(rc):
            fig = lr.render_page(ed, lay, 's1', rc=rc)
            return [mc.to_hex(ln.get_color()) for ln in fig.axes[0].lines
                    if ln.get_label() and not ln.get_label().startswith('_')]
        monkeypatch.setattr(lr, 'color_prefs',
                            lambda: (['#aa0000', '#0000aa'], False))
        assert line_colours(lr.RenderContext(ed, 's1')) == ['#111111',
                                                            '#222222']
        monkeypatch.setattr(lr, 'color_prefs',
                            lambda: (['#aa0000', '#0000aa'], True))
        assert line_colours(lr.RenderContext(ed, 's1')) == ['#aa0000',
                                                            '#0000aa']
        ctx = lr.SampleContext(ed, 's1')
        lay.items[iid]['layer_colors'] = {
            lr.layer_id(ctx, p1, pinned=False): '#00aa00'}
        assert line_colours(lr.RenderContext(ed, 's1')) == ['#aa0000',
                                                            '#00aa00']
    finally:
        root.destroy()


def test_wheel_zoom_and_dropping_a_plot_on_a_table(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import openflo.ui_layout_editor as ule
        from openflo.layout_render import SampleContext
        c1, _p1 = _load(ed, 's1', 1, 600, 300.0)
        _load(ed, 's2', 2, 500, 600.0, extra_gate=True)
        win = ule.LayoutEditorWindow(ed)
        win.update()
        key = SampleContext(ed, 's1').keys[c1]
        plot = win.layout.add(lp.plot_item(population=key, x='FL1-A',
                                           y='FL2-A', samples=['s1', 's2'],
                                           box=(0.05, 0.05, 0.4, 0.5)))
        table = win.layout.add(lp.stats_item(stats=['Count'],
                                             box=(0.55, 0.05, 0.4, 0.3)))
        win._redraw()
        ax = win._axes[plot]
        x0, x1 = ax.get_xlim()
        win._zoom_item(plot, 0.5)
        zx = win.layout.items[plot]['xlim']
        assert zx[1] - zx[0] < (x1 - x0) * 0.95
        assert zx[0] > x0 and zx[1] < x1
        # The wheel over a plot that isn't selected scrolls the page and
        # leaves the plot alone; over the selected plot it zooms the plot.
        bx0, by0, bx1, by1 = win._box_px(win.layout.items[plot]['box'])
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2

        def wheel(delta, state=0):
            return SimpleNamespace(x=cx - win.canvas.canvasx(0),
                                   y=cy - win.canvas.canvasy(0),
                                   delta=delta, state=state)
        win._sel = None
        win._on_wheel(wheel(-120))
        assert win.layout.items[plot]['xlim'] == zx
        win._sel = plot
        win._on_wheel(wheel(-120))
        zx2 = win.layout.items[plot]['xlim']
        assert zx2[1] - zx2[0] > zx[1] - zx[0]
        # Ctrl+wheel zooms the page instead (any item, selected or not).
        win._on_wheel(wheel(120, state=0x0004))
        assert win._zoom != 'Fit' and win.zoom_var.get().endswith('%')
        assert win.layout.items[plot]['xlim'] == zx2
        # Drag the plot onto the table: the table takes its samples and
        # population; the plot goes back where it was.
        before = list(win.layout.items[plot]['box'])
        # The page was just zoomed (Ctrl+wheel): aim at it as it is now, and
        # at a point of the table that is on screen at this zoom.
        win._redraw()
        win.update_idletasks()
        bx0, by0, bx1, by1 = win._box_px(win.layout.items[plot]['box'])
        tx0, ty0, tx1, ty1 = win._box_px(win.layout.items[table]['box'])
        tx1, ty1 = min(tx1, tx0 + 40), min(ty1, ty0 + 40)

        def e(x, y):
            return SimpleNamespace(x=x - win.canvas.canvasx(0),
                                   y=y - win.canvas.canvasy(0))
        win._sel = None
        win._on_press(e(bx0 + 10, by0 + 10))
        win._on_drag(e((tx0 + tx1) / 2, (ty0 + ty1) / 2))
        win._on_release(e((tx0 + tx1) / 2, (ty0 + ty1) / 2))
        t = win.layout.items[table]
        assert t['samples'] == ['s1', 's2']
        assert t['populations'] == [key]
        assert win.layout.items[plot]['box'] == before
        win.destroy()
    finally:
        root.destroy()


def test_default_colours_dialog_saves_the_list_and_toggle(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        import openflo.prefs as prefs
        import openflo.ui_layout_editor as ule
        store = {}
        monkeypatch.setattr(prefs, 'read_prefs', lambda: dict(store))
        monkeypatch.setattr(prefs, 'write_pref',
                            lambda k, v: store.__setitem__(k, v))
        called = []
        dlg = ule._ColorsDialog(ed, on_ok=lambda: called.append(1))
        assert dlg.colors[:2] == ['#d62728', '#1f77b4']   # red, blue
        assert dlg.override_var.get() is False            # off by default
        dlg.colors.append('#123456')
        dlg.override_var.set(True)
        dlg._save()
        assert store['layout_colors'][-1] == '#123456'
        assert store['layout_color_override'] is True and called == [1]
    finally:
        root.destroy()


def test_a_population_that_cannot_be_computed_reads_na_not_zero():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext, render_page
        _c1, p1 = _load(ed, 's1', 1, 400, 300.0)
        ed._set_active_sample('s1')
        fj = ed._add_gate({'kind': 'flowjo_ellipse', 'name': 'FJ',
                           'parent_id': None}, audit=False)
        empty = ed._add_gate({'kind': 'threshold', 'name': 'None',
                              'channel': 'FL1-A', 'op': '>', 'value': 1e9,
                              'parent_id': None}, audit=False)
        keys = SampleContext(ed, 's1').keys
        lay = lp.Layout()
        lay.add(lp.stats_item(populations=[keys[fj], keys[empty], keys[p1]],
                              stats=['Count', '%Parent'],
                              box=(0, 0, 0.5, 0.3)))
        lay.add(lp.chart_item('bar', populations=[keys[fj], keys[empty]],
                              value='Count', box=(0.5, 0, 0.5, 0.5)))
        lay.add(lp.plot_item(population=keys[fj], x='FL1-A', y='FL2-A',
                             box=(0, 0.5, 0.5, 0.5)))
        fig = render_page(ed, lay, 's1')
        (table,) = _table_text(fig)
        assert table[1] == ['FJ', 'NA', 'NA']
        assert table[2][:2] == ['None', '0']           # measured, empty
        bar_ax, plot_ax = fig.axes[1], fig.axes[2]
        heights = [p.get_height() for p in bar_ax.patches]
        assert np.isnan(heights[0]) and heights[1] == 0
        assert 'NA' in [t.get_text() for t in bar_ax.texts]
        assert any(t.get_text().startswith('NA') for t in plot_ax.texts)
    finally:
        root.destroy()


# ── round 4: panning, centring, the cross-page host ──────────────────────

def test_the_cross_page_host_round_trips():
    doc = lp.LayoutDoc()
    doc.add_page()
    i = doc.shelve(lp.text_item('kept', box=(0.1, 0.1, 0.3, 0.1)))
    a = doc.place_from_shelf(i, 1)
    assert doc.pages[1].items[a]['text'] == 'kept'
    b = doc.place_from_shelf(i, 0, box=(0.5, 0.5, 0.3, 0.1))
    assert doc.pages[0].items[b]['box'][:2] == pytest.approx([0.5, 0.5])
    back = lp.LayoutDoc.loads(doc.dumps())
    assert back.shelf == doc.shelf


def test_pan_centre_and_host_in_the_window(monkeypatch):
    root, ed = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import openflo.ui_layout_editor as ule
        _load(ed, 's1', 1, 600, 300.0)
        win = ule.LayoutEditorWindow(ed)
        win.deiconify()
        win.update()
        win._fit()
        plot = win.layout.add(lp.plot_item(x='FL1-A', y='FL2-A',
                                           xscale='linear', yscale='linear',
                                           xlim=(0, 1000), ylim=(0, 1000),
                                           box=(0.3, 0.1, 0.4, 0.5)))
        win._redraw()
        ax = win._axes[plot]
        bx0, by0, _bx1, _by1 = win._box_px(win.layout.items[plot]['box'])
        fig_h = ax.figure.bbox.height

        def canvas_pt(xd, yd):
            px, py = ax.transData.transform((xd, yd))
            return bx0 + px, by0 + fig_h - py
        # Centre on (250, 750): the view becomes 250±500 x 750±500.
        win._center_item(plot, canvas_pt(250, 750))
        it = win.layout.items[plot]
        assert sum(it['xlim']) / 2 == pytest.approx(250, abs=5)
        assert sum(it['ylim']) / 2 == pytest.approx(750, abs=5)
        assert it['xlim'][1] - it['xlim'][0] == pytest.approx(1000, abs=5)
        # A middle-drag moves the view by the dragged distance.
        ax = win._axes[plot]
        bx0, by0, _bx1, _by1 = win._box_px(it['box'])
        a, b = canvas_pt(100, 500), canvas_pt(300, 500)
        out = win._view_shift(plot, a, b)
        assert out['xlim'][0] == pytest.approx(it['xlim'][0] - 200, abs=5)
        assert out['ylim'] == pytest.approx(it['ylim'], abs=5)
        # Drag the plot onto the host: a copy is kept, the plot stays.
        before = list(it['box'])
        hl = win.host_list
        win._sel = None
        x0, y0, _x1, _y1 = win._box_px(it['box'])
        win._on_press(SimpleNamespace(x=x0 + 8 - win.canvas.canvasx(0),
                                      y=y0 + 8 - win.canvas.canvasy(0)))
        rel = SimpleNamespace(x=0, y=0,
                              x_root=hl.winfo_rootx() + 10,
                              y_root=hl.winfo_rooty() + 10)
        win._on_release(SimpleNamespace(
            x=rel.x_root - win.canvas.winfo_rootx() - win.canvas.canvasx(0),
            y=rel.y_root - win.canvas.winfo_rooty() - win.canvas.canvasy(0),
            x_root=rel.x_root, y_root=rel.y_root))
        assert len(win.doc.shelf) == 1
        assert win.layout.items[plot]['box'] == before
        assert hl.size() == 1
        # Its name tells it apart: number, kind, population, axes, page.
        label = hl.get(0)
        assert label.startswith('1. P1 · Plot · All events · ')
        assert 'FL1-A / FL2-A (dot)' in label and 'from Page 1' in label
        import openflo.ui_layout_editor as ule2
        monkeypatch.setattr(ule2.simpledialog, 'askstring',
                            lambda *a, **k: 'Overview dots')
        win._rename_host(0)
        assert hl.get(0) == '1. P1 · Overview dots'
        # A copy placed on a page carries none of the host's labels.
        iid2 = win.doc.place_from_shelf(0, 0)
        assert 'host_name' not in win.layout.items[iid2]
        win.layout.remove(iid2)
        # Placed on every page from the host.
        win._new_page()
        win._place_host(0, range(len(win.doc.pages)))
        assert all(any(i.get('kind') == 'plot' for i in p.items.values())
                   for p in win.doc.pages)
        win.destroy()
    finally:
        root.destroy()


def test_middle_drag_moves_the_data_not_the_plot():
    root, ed = _editor_or_skip()
    try:
        from types import SimpleNamespace

        import openflo.ui_layout_editor as ule
        _load(ed, 's1', 1, 600, 300.0)
        win = ule.LayoutEditorWindow(ed)
        win.deiconify()
        win.update()
        win._fit()
        plot = win.layout.add(lp.plot_item(x='FL1-A', y='FL2-A',
                                           xscale='linear', yscale='linear',
                                           xlim=(0, 1000), ylim=(0, 1000),
                                           box=(0.3, 0.1, 0.4, 0.5)))
        win._redraw()
        box = list(win.layout.items[plot]['box'])
        ax = win._axes[plot]
        bx0, by0, _bx1, _by1 = win._box_px(box)
        fig_h = ax.figure.bbox.height
        # Data units per screen pixel, from the plot as drawn.
        (xlo, xhi), (ylo, yhi) = ax.get_xlim(), ax.get_ylim()
        ux = (xhi - xlo) / ax.bbox.width
        uy = (yhi - ylo) / ax.bbox.height

        def ev(xd, yd):
            # Whole pixels, as Tk delivers them (canvasx/canvasy round a
            # fraction anyway), so each point is off by up to half a pixel.
            px, py = ax.transData.transform((xd, yd))
            return SimpleNamespace(
                x=round(bx0 + px - win.canvas.canvasx(0)),
                y=round(by0 + fig_h - py - win.canvas.canvasy(0)))

        def moved(a, b):
            """Data shift (x, y) the drag from event a to event b asks for."""
            return (b.x - a.x) * ux, (a.y - b.y) * uy
        start = ev(500, 500)
        win._on_pan_start(start)
        mid = ev(600, 500)
        win._on_pan_move(mid)                  # live: the view moves already
        dx, _dy = moved(start, mid)
        assert dx == pytest.approx(100, abs=ux)
        assert win.layout.items[plot]['xlim'][0] == pytest.approx(xlo - dx)
        end = ev(700, 400)
        win._on_pan_move(end)
        win._on_pan_end(end)
        it = win.layout.items[plot]
        dx, dy = moved(start, end)
        assert (dx, dy) == pytest.approx((200, -100), abs=max(ux, uy))
        assert it['xlim'] == pytest.approx([xlo - dx, xhi - dx])
        assert it['ylim'] == pytest.approx([ylo - dy, yhi - dy])
        assert it['box'] == box                     # the plot itself stayed
        win.destroy()
    finally:
        root.destroy()


def test_host_hover_points_out_matches_and_the_host_widens():
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        _load(ed, 's1', 1, 300, 300.0)
        win = ule.LayoutEditorWindow(ed)
        win.deiconify()
        win.update()
        win._fit()
        a = win.layout.add(lp.plot_item(x='FL1-A', y='FL2-A',
                                        box=(0.1, 0.1, 0.3, 0.3)))
        # Same entity drawn independently (same populations and axes).
        twin = win.layout.add(lp.plot_item(x='FL1-A', y='FL2-A',
                                           box=(0.5, 0.1, 0.3, 0.3)))
        other = win.layout.add(lp.text_item('x', box=(0.1, 0.6, 0.3, 0.1)))
        win.doc.shelve(win.layout.items[a], origin=win.layout.name)
        win._new_page()                                  # page 2 is current
        win.doc.place_from_shelf(0, 1)
        win._go_page(0)
        win.tabs.select(0)
        win._refresh_host()
        matches = win.doc.matches(0)
        assert (0, a) in matches and (0, twin) in matches
        assert (0, other) not in matches and any(p == 1 for p, _ in matches)
        win._host_mark(0)
        assert len(win.canvas.find_withtag('hostmark')) == 2
        assert win.tabs.tab(1, 'text').startswith('◆')
        assert not win.tabs.tab(0, 'text').startswith('◆')
        win._host_unmark()
        assert not win.canvas.find_withtag('hostmark')
        assert not win.tabs.tab(1, 'text').startswith('◆')
        # The host sits in a paned window: its edge is a draggable sash.
        assert len(win._body.panes()) == 2
        win.destroy()
    finally:
        root.destroy()


def test_items_get_unique_ids_and_only_true_duplicates_match():
    doc = lp.LayoutDoc()
    p0 = doc.pages[0]
    a = p0.add(lp.plot_item(x='FL1-A', y='FL2-A'))
    s = p0.add(lp.stats_item(populations=['n:cells']))
    b = p0.duplicate(a)                     # carries a's id until fixed
    doc.ensure_uids()
    uids = [p0.items[i]['uid'] for i in (a, s, b)]
    assert uids[0].startswith('P') and uids[1].startswith('S')
    assert len(set(uids)) == 3              # the duplicate got its own
    assert p0.items[a]['uid'] == uids[0]    # and the original kept its
    doc.add_page()
    i = doc.shelve(p0.items[a], origin='Page 1')
    c = doc.place_from_shelf(i, 1)
    doc.ensure_uids()
    assert doc.pages[1].items[c]['uid'] not in uids
    found = set(doc.matches(i))
    assert {(0, a), (0, b), (1, c)} <= found
    # Zooming the placed copy makes it a different plot: no longer marked.
    doc.pages[1].items[c]['xlim'] = [0.0, 10.0]
    assert (1, c) not in set(doc.matches(i))
    back = lp.LayoutDoc.loads(doc.dumps())
    assert [it['uid'] for it in back.pages[0].items.values()] == [
        it['uid'] for it in p0.items.values()]
