"""The layout editor's exploded view: the whole plot beside a magnified copy
of its zoom (or of one gate), the region outlined and joined to the copy by
connector lines, in the page preview and in every page / batch export.

Every test runs on generated data (uniform or Gaussian events on generic
channels) and checks both sides: the case that must explode, and the cases
that must stay one plain plot."""
from __future__ import annotations

import io
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from openflo import layout_page as lp  # noqa: E402
from tests.test_layout_editor import (  # noqa: E402
    _editor_or_skip,
    _expected,
    _load,
    _n_drawn,
    _no_downsample,
)

LIN = dict(xscale='linear', yscale='linear')


# ── page model and helpers (no Tk) ───────────────────────────────────────

def test_plot_items_default_to_auto_explode_and_refuse_unknown_sides():
    it = lp.plot_item(x='FL1-A', y='FL2-A')
    assert (it['explode'], it['explode_gate'], it['inset_side']) == (
        None, None, 'auto')
    it = lp.plot_item(explode=0, inset_side='left', explode_gate='n:a')
    assert it['explode'] is False
    assert (it['explode_gate'], it['inset_side']) == ('n:a', 'left')
    assert lp.plot_item(explode=1)['explode'] is True
    assert lp.plot_item(inset_side='diagonal')['inset_side'] == 'auto'
    # Older items without the keys still load and compare as before.
    doc = lp.LayoutDoc()
    doc.pages[0].add(lp.plot_item(x='FL1-A', explode=True, inset_side='above'))
    back = lp.LayoutDoc.loads(doc.dumps())
    assert back.pages[0].items == doc.pages[0].items


@pytest.mark.parametrize('side', ['right', 'left', 'below', 'above'])
def test_split_box_puts_the_inset_on_the_asked_side(side):
    box = (0.1, 0.2, 0.6, 0.4)
    over, inset, got = lp.split_box(box, side)
    assert got == side
    for b in (over, inset):                       # both inside the box
        assert b[0] >= box[0] - 1e-9 and b[1] >= box[1] - 1e-9
        assert b[0] + b[2] <= box[0] + box[2] + 1e-9
        assert b[1] + b[3] <= box[1] + box[3] + 1e-9
    # Page boxes have their origin top-left: 'below' is a larger y.
    ox, oy = over[0] + over[2] / 2, over[1] + over[3] / 2
    ix, iy = inset[0] + inset[2] / 2, inset[1] + inset[3] / 2
    want = {'right': ix > ox and iy == pytest.approx(oy),
            'left': ix < ox and iy == pytest.approx(oy),
            'below': iy > oy and ix == pytest.approx(ox),
            'above': iy < oy and ix == pytest.approx(ox)}
    assert want[side] and sum(want.values()) == 1
    # The halves do not overlap: there is a gap for the connector lines.
    if side in ('right', 'left'):
        left, right = sorted((over, inset))
        assert left[0] + left[2] < right[0]
    else:
        top, bottom = sorted((over, inset), key=lambda b: b[1])
        assert top[1] + top[3] < bottom[1]


def test_split_box_auto_follows_the_box_shape():
    assert lp.split_box((0, 0, 0.5, 0.3), 'auto', aspect=2.0)[2] == 'right'
    assert lp.split_box((0, 0, 0.3, 0.5), 'auto', aspect=0.6)[2] == 'below'
    assert lp.split_box((0, 0, 0.3, 0.3), None, aspect=1.0)[2] == 'below'


def test_gate_extent_reads_each_gate_kind_and_skips_open_sides():
    from openflo.layout_render import _gate_extent
    x, y = 'FL1-A', 'FL2-A'
    rect = {'kind': 'rect', 'x_channel': x, 'y_channel': y,
            'x0': 30.0, 'x1': 10.0, 'y0': 5.0, 'y1': 50.0}
    assert [sorted(v) for v in _gate_extent(rect, x, y)] == [[10, 30], [5, 50]]
    poly = {'kind': 'polygon', 'x_channel': x, 'y_channel': y,
            'vertices': [[1, 2], [9, 3], [4, 8]]}
    xs, ys = _gate_extent(poly, x, y)
    assert (min(xs), max(xs), min(ys), max(ys)) == (1, 9, 2, 8)
    ell = {'kind': 'ellipsoid', 'x_channel': x, 'y_channel': y,
           'mean': [100.0, 200.0], 'cov': [[4.0, 0.0], [0.0, 9.0]],
           'distance_sq': 4.0}
    xs, ys = _gate_extent(ell, x, y)
    assert sorted(xs) == pytest.approx([96, 104])
    assert sorted(ys) == pytest.approx([194, 206])
    thr = {'kind': 'threshold', 'channel': y, 'op': '>', 'value': 7.0}
    assert _gate_extent(thr, x, y) == ([], [7.0])
    iv = {'kind': 'interval', 'channel': x, 'lo': 2.0, 'hi': 3.0}
    assert _gate_extent(iv, x, None) == ([2.0, 3.0], [])
    # An open side (an imported quadrant's +-1e12) is not a coordinate.
    quad = dict(rect, x1=1e12, y0=-1e12)
    assert _gate_extent(quad, x, y) == ([30.0], [50.0])
    # A gate on other channels (or the swapped pair) has no outline here.
    assert _gate_extent(rect, y, x) == ([], [])
    assert _gate_extent(rect, 'FSC-A', 'SSC-A') == ([], [])
    assert _gate_extent(thr, 'FSC-A', 'SSC-A') == ([], [])


@pytest.mark.parametrize('kind', ['logicle', 'asinh'])
def test_a_logicle_view_zoomed_between_decades_still_has_ticks(kind):
    from openflo.pipeline import transform_values
    from openflo.scales import intensity_ticks

    def stored(v):
        return [float(x) for x in transform_values(np.asarray(v, dtype=float),
                                                   method='logicle')]
    # A magnified inset between 10^2 and 10^3: round values, all inside.
    lo, hi = stored([560.0, 760.0])
    pos, vals = intensity_ticks('logicle', kind, lo, hi)
    assert 2 <= len(vals) <= 6
    assert all(560.0 <= v <= 760.0 for v in vals)
    assert all(lo - 1e-9 <= p <= hi + 1e-9 for p in pos)
    # A view holding several decades keeps 0 and the decades only.
    lo, hi = stored([0.0, 1e5])
    _pos, vals = intensity_ticks('logicle', kind, lo, hi)
    assert len(vals) >= 2
    assert all(v == 0 or float(np.log10(abs(v))).is_integer() for v in vals)


# ── rendering through a real editor ──────────────────────────────────────

def _render(ed, item, sample='s1'):
    from openflo.layout_render import RenderContext, render_page
    lay = lp.Layout()
    iid = lay.add(item)
    rc = RenderContext(ed, sample)
    fig = render_page(ed, lay, sample, rc=rc)
    return fig, rc.info.get(iid, {})


def _connectors(fig):
    from matplotlib.patches import ConnectionPatch
    return [a for a in fig.artists if isinstance(a, ConnectionPatch)]


def _outlines(ax):
    from openflo.layout_render import REGION_GID
    return [a for a in ax.get_children() if a.get_gid() == REGION_GID]


def _centre(ax):
    p = ax.get_position()
    return (p.x0 + p.x1) / 2, (p.y0 + p.y1) / 2


def _assert_plain(fig, info):
    assert len(fig.axes) == 1
    assert not _connectors(fig) and not _outlines(fig.axes[0])
    assert 'exploded' not in info


def test_a_zoom_that_hides_data_is_shown_exploded():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext
        cells, _pos = _load(ed, 's1', 1, 1200, 300.0)
        _no_downsample(ed)
        key = SampleContext(ed, 's1').keys[cells]
        n = _expected(ed, 's1', cells)[0]
        df = ed._samples['s1'].data
        fig, info = _render(ed, lp.plot_item(
            population=key, x='FL1-A', y='FL2-A', xlim=(0.0, 300.0),
            ylim=(0.0, 1000.0), inset_side='right', **LIN))
        assert info['exploded'] == 'zoom'
        over, inset = info['overview'], info['inset']
        assert len(fig.axes) == 2 and info['ax'] is inset
        # The magnified copy keeps the zoom; the whole plot shows all data.
        assert inset.get_xlim() == pytest.approx((0.0, 300.0))
        lo, hi = sorted(over.get_xlim())
        assert lo <= df['FL1-A'].min() and hi >= df['FL1-A'].max() * 0.99
        assert _n_drawn(over) == _n_drawn(inset) == n
        # The zoom is outlined on the whole plot ...
        (rect,) = _outlines(over)
        assert (rect.get_x(), rect.get_y()) == pytest.approx((0.0, 0.0))
        assert (rect.get_width(), rect.get_height()) == pytest.approx(
            (300.0, 1000.0))
        # ... and its right corners join the inset's left ones.
        cons = _connectors(fig)
        assert len(cons) == 2
        assert sorted(c.xy1 for c in cons) == [(300.0, 0.0), (300.0, 1000.0)]
        assert sorted(c.xy2 for c in cons) == [(0, 0), (0, 1)]
        assert _centre(inset)[0] > _centre(over)[0]
        # A log view zoomed inside a decade keeps the plot code's plain
        # round ticks in both halves (exploded or not).
        for explode in (None, False):
            fig2, _i = _render(ed, lp.plot_item(
                population=key, x='FL1-A', y='FL2-A', xscale='log',
                yscale='log', xlim=(560.0, 760.0), ylim=(150.0, 350.0),
                explode=explode))
            ax = fig2.axes[0]
            assert ax.get_xscale() == 'log'
            assert ax.xaxis.get_major_formatter()(600.0, 0) == '600'
        for fmt in ('png', 'pdf', 'svg'):          # every export format
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=150)
            assert buf.tell() > 1000
    finally:
        root.destroy()


def test_plots_that_must_not_explode_stay_one_plot():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext
        cells, _pos = _load(ed, 's1', 2, 800, 300.0)
        _no_downsample(ed)
        key = SampleContext(ed, 's1').keys[cells]
        base = dict(population=key, x='FL1-A', y='FL2-A', **LIN)
        hiding = dict(base, xlim=(0.0, 300.0), ylim=(0.0, 1000.0))
        _assert_plain(*_render(ed, lp.plot_item(**hiding, explode=False)))
        # A zoom that shows every event (a dropped plot keeps the limits
        # it had) is not exploded unless asked ...
        wide = dict(base, xlim=(-50.0, 1050.0), ylim=(-50.0, 1050.0))
        _assert_plain(*_render(ed, lp.plot_item(**wide)))
        fig, info = _render(ed, lp.plot_item(**wide, explode=True))
        assert info['exploded'] == 'zoom' and len(_connectors(fig)) == 2
        # A zoom reaching past the data widens the whole plot to take in
        # all of its outline.
        _fig, info = _render(ed, lp.plot_item(
            **dict(base, xlim=(-400.0, 1400.0), ylim=(0.0, 500.0)),
            explode=True))
        lo, hi = sorted(info['overview'].get_xlim())
        assert lo <= -400.0 and hi >= 1400.0
        # ... and a plot without a zoom has nothing to explode.
        _assert_plain(*_render(ed, lp.plot_item(**base, explode=True)))
    finally:
        root.destroy()


_SIDE_A = {'right': {(300.0, 0.0), (300.0, 1000.0)},
           'left': {(0.0, 0.0), (0.0, 1000.0)},
           'below': {(0.0, 0.0), (300.0, 0.0)},
           'above': {(0.0, 1000.0), (300.0, 1000.0)}}
_SIDE_B = {'right': {(0, 0), (0, 1)}, 'left': {(1, 0), (1, 1)},
           'below': {(0, 1), (1, 1)}, 'above': {(0, 0), (1, 0)}}


@pytest.mark.parametrize('side', ['right', 'left', 'below', 'above'])
def test_the_inset_goes_where_asked_and_the_lines_meet_the_near_corners(side):
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 3, 600, 300.0)
        _no_downsample(ed)
        fig, info = _render(ed, lp.plot_item(
            x='FL1-A', y='FL2-A', xlim=(0.0, 300.0), ylim=(0.0, 1000.0),
            inset_side=side, box=(0.05, 0.05, 0.6, 0.6), **LIN))
        (ox, oy), (ix, iy) = _centre(info['overview']), _centre(info['inset'])
        # Figure coordinates: 'below' is a smaller y.
        assert {'right': ix > ox, 'left': ix < ox, 'below': iy < oy,
                'above': iy > oy}[side]
        cons = _connectors(fig)
        assert {tuple(c.xy1) for c in cons} == _SIDE_A[side]
        assert {tuple(c.xy2) for c in cons} == _SIDE_B[side]
    finally:
        root.destroy()


def test_a_histogram_zoom_explodes_along_x():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 4, 900, 300.0)
        _no_downsample(ed)
        fig, info = _render(ed, lp.plot_item(
            x='FL1-A', mode='histogram', xlim=(600.0, 900.0),
            inset_side='below', xscale='linear'))
        assert info['exploded'] == 'zoom'
        assert info['inset'].get_xlim() == pytest.approx((600.0, 900.0))
        lo, hi = sorted(info['overview'].get_xlim())
        assert lo < 100.0 and hi > 900.0
        over = info['overview']
        (rect,) = _outlines(over)
        # The outline spans the axes' height, whatever the y units.
        fig.canvas.draw()
        box = rect.get_window_extent()
        assert (box.y0, box.y1) == pytest.approx(
            (over.bbox.y0, over.bbox.y1), abs=1.5)
        assert box.width < over.bbox.width / 2
        assert {c.xy1 for c in _connectors(fig)} == {(600.0, 0.0),
                                                     (900.0, 0.0)}
        _assert_plain(*_render(ed, lp.plot_item(
            x='FL1-A', mode='histogram', xscale='linear')))
    finally:
        root.destroy()


def test_a_chosen_gate_is_magnified_with_its_own_events():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext
        cells, pos = _load(ed, 's1', 5, 1500, 300.0)
        _no_downsample(ed)
        keys = SampleContext(ed, 's1').keys
        n_cells = _expected(ed, 's1', cells)[0]
        fig, info = _render(ed, lp.plot_item(
            x='FSC-A', y='SSC-A', explode_gate=keys[cells], **LIN))
        assert info['exploded'] == 'gate'
        over, inset = info['overview'], info['inset']
        assert info['ax'] is over and len(fig.axes) == 2
        # The inset draws exactly the gate's events, around the gate.
        assert _n_drawn(inset) == n_cells
        assert _n_drawn(over) == len(ed._samples['s1'].data)
        for lim in (inset.get_xlim(), inset.get_ylim()):
            lo, hi = sorted(lim)
            assert lo < 100.0 < 900.0 < hi           # the outline fits ...
            assert lo > 30.0 and hi < 970.0          # ... with a margin
        # The gate's own outline is drawn in the inset.
        from matplotlib.patches import Rectangle
        assert any(isinstance(p, Rectangle) and p.get_width() ==
                   pytest.approx(800.0) for p in inset.patches)
        (rect,) = _outlines(over)
        assert rect.get_x() == pytest.approx(sorted(inset.get_xlim())[0])
        assert len(_connectors(fig)) == 2

        # A threshold child on the Cells plot: from its cut to its events.
        df = ed._samples['s1'].data
        fig, info = _render(ed, lp.plot_item(
            population=keys[cells], x='FL1-A', y='FL2-A',
            explode_gate=keys[pos], **LIN))
        assert info['exploded'] == 'gate'
        assert _n_drawn(info['inset']) == _expected(ed, 's1', pos)[0]
        lo, hi = sorted(info['inset'].get_xlim())
        assert lo < 300.0 < 310.0 < hi and hi >= df['FL1-A'].max() * 0.99
    finally:
        root.destroy()


def test_a_gate_that_is_not_on_the_plot_is_not_exploded():
    root, ed = _editor_or_skip()
    try:
        from openflo.layout_render import SampleContext
        cells, pos = _load(ed, 's1', 6, 700, 300.0)
        _no_downsample(ed)
        keys = SampleContext(ed, 's1').keys
        base = dict(x='FSC-A', y='SSC-A', **LIN)
        # Pos is made on the Cells plot, not on the ungated one.
        _assert_plain(*_render(ed, lp.plot_item(**base,
                                                explode_gate=keys[pos])))
        # A key this sample lacks.
        _assert_plain(*_render(ed, lp.plot_item(**base,
                                                explode_gate='n:absent')))
        # Explode off wins over a chosen gate.
        _assert_plain(*_render(ed, lp.plot_item(
            **base, explode_gate=keys[cells], explode=False)))
        # Several layers: the gate's % would be ambiguous, so no inset.
        _load(ed, 's2', 7, 500, 300.0)
        _assert_plain(*_render(ed, lp.plot_item(
            **base, samples=['s1', 's2'], explode_gate=keys[cells])))
    finally:
        root.destroy()


def test_the_overview_ignores_the_main_windows_range_and_restores_it():
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 8, 900, 300.0)
        _no_downsample(ed)
        ed._channel_range['FL1-A'] = (10.0, 20.0)       # main window zoom
        ed._channel_range['FL2-A'] = (40.0, 60.0)
        before = (dict(ed._channel_scale), dict(ed._channel_range))
        df = ed._samples['s1'].data
        _fig, info = _render(ed, lp.plot_item(
            x='FL1-A', y='FL2-A', xlim=(0.0, 300.0), ylim=(0.0, 1000.0),
            **LIN))
        lo, hi = sorted(info['overview'].get_xlim())
        assert lo <= df['FL1-A'].min() and hi >= df['FL1-A'].max() * 0.99
        lo, hi = sorted(info['overview'].get_ylim())
        assert lo <= df['FL2-A'].min() and hi >= df['FL2-A'].max() * 0.99
        assert (dict(ed._channel_scale), dict(ed._channel_range)) == before
    finally:
        root.destroy()


def test_a_logicle_channel_explodes_and_exports():
    root, ed = _editor_or_skip()
    try:
        from openflo import pipeline as fp
        from openflo.pipeline import FlowSample
        rng = np.random.default_rng(9)
        lin = np.concatenate([rng.normal(0, 150, (2000, 2)),
                              rng.lognormal(np.log(5000), 0.6, (6000, 2))])
        df = pd.DataFrame(fp.transform_values(lin.ravel(), 'logicle')
                          .reshape(lin.shape), columns=['FL1-A', 'FL2-A'])
        s = FlowSample.from_dataframe(df, name='s1', path='s1.fcs')
        ed._samples['s1'] = s
        ed._sample_order.append('s1')
        ed._sample_colors['s1'] = '#1f77b4'
        ed._sample_trial['s1'] = 'T'
        ed._trial_order.append('T')
        ed._sample_plot_enabled['s1'] = True
        ed._channels = list(df.columns)
        ed._channel_labels = {c: c for c in df.columns}
        ed._channel_transform = {c: 'logicle' for c in df.columns}
        ed._set_active_sample('s1')
        _no_downsample(ed)
        hi_x = float(np.quantile(df['FL1-A'], 0.6))
        fig, info = _render(ed, lp.plot_item(
            # The log view: the zoom's negative events have no position there.
            x='FL1-A', y='FL2-A', xscale='log', yscale='log',
            xlim=(float(df['FL1-A'].min()), hi_x),
            ylim=(float(df['FL2-A'].min()), float(df['FL2-A'].max()))))
        assert info['exploded'] == 'zoom'
        over = info['overview']
        lo, hi = sorted(over.get_xlim())
        assert hi >= float(df['FL1-A'].max()) * 0.99
        (rect,) = _outlines(over)
        assert lo <= rect.get_x() and rect.get_x() + rect.get_width() <= hi
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        assert buf.tell() > 1000
    finally:
        root.destroy()


def test_the_window_drops_the_fit_badge_and_the_dialog_keeps_the_choices():
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        from openflo.layout_render import OUTSIDE_WARN, SampleContext
        cells, pos = _load(ed, 's1', 10, 900, 300.0)
        _no_downsample(ed)
        keys = SampleContext(ed, 's1').keys
        win = ule.LayoutEditorWindow(ed)
        win.update()
        zoom = dict(x='FL1-A', y='FL2-A', xlim=(0.0, 300.0),
                    ylim=(0.0, 1000.0), **LIN)
        a = win.layout.add(lp.plot_item(**zoom))
        b = win.layout.add(lp.plot_item(**zoom, explode=False,
                                        box=(0.5, 0.5, 0.3, 0.3)))
        win._redraw()
        assert win._outside[a] == 0.0          # all its data is shown
        assert win._outside[b] > OUTSIDE_WARN  # still offers Zoom to fit
        got = []
        dlg = ule._ItemDialog(win, lp.plot_item(population=keys[cells],
                                                **zoom),
                              win._choices(), on_ok=got.append)
        # The gates drawn on this population's plot, and only those.
        assert list(dlg._gate_keys.values()) == [keys[pos]]
        it = dlg._collect_plot()
        assert (it['explode'], it['explode_gate'], it['inset_side']) == (
            None, None, 'auto')
        dlg.explode_var.set(ule._EXPLODE_LABELS[2])
        dlg.side_var.set('Left')
        dlg.egate_var.set(next(iter(dlg._gate_keys)))
        it = dlg._collect_plot()
        assert (it['explode'], it['explode_gate'], it['inset_side']) == (
            False, keys[pos], 'left')
        dlg.explode_var.set(ule._EXPLODE_LABELS[1])
        assert dlg._collect_plot()['explode'] is True
        dlg.destroy()
        win.destroy()
    finally:
        root.destroy()
