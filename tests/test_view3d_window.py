"""Analyze > 3D plot: the window against the editor's own 2D plot.

Generated samples only. FL1-A / FL2-A are stored logicle (as the loader
stores fluorescence) and SSC-A is stored linear, so a logicle view of SSC-A
is a display transform of a linear channel: the case the mobile harness
cannot tell apart (#356). Each setting in the window is changed and its
effect checked.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

from openflo import view3d as v3  # noqa: E402
from openflo.async_task import wait_for_job  # noqa: E402

COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']


def _sample_frame(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'FSC-A': rng.uniform(2e4, 2.4e5, n),
                       'SSC-A': rng.lognormal(10.0, 0.6, n),
                       'FL1-A': rng.lognormal(6.0, 1.3, n),
                       'FL2-A': rng.lognormal(5.0, 1.0, n),
                       # Stored linear, either side of zero: a symlog view's
                       # threshold changes where its events land.
                       'FL3-A': rng.normal(50.0, 400.0, n)})
    df['cluster'] = (df['FL1-A'] > df['FL1-A'].median()).astype(int)
    return df


def _editor(n=3000, extra=None):
    from openflo.pipeline import FlowSample
    root, ed, _gui = _editor_or_skip()
    df = _sample_frame(n)
    if extra:
        for k, v in extra.items():
            df[k] = v
    s = FlowSample.from_dataframe(df, name='S1')
    s.apply_transform(['FL1-A', 'FL2-A'])
    ed._samples['S1'] = s
    ed._sample_order.append('S1')
    ed._sample_trial['S1'] = 'T'
    ed._sample_plot_enabled['S1'] = True
    ed._sample_gates['S1'] = {}
    ed._sample_gate_order['S1'] = []
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed._set_active_sample('S1')
    return root, ed


def _window(ed, root):
    w = ed._open_view3d()
    assert w is not None
    w.geometry('720x620')
    root.update()
    w.refresh()
    root.update()
    return w


def _set_axes(w, x, y, z):
    for var, ch in zip(w.axis_vars, (x, y, z), strict=True):
        var.set(w.editor._fmt_channel(ch))
    w.refresh()


def _front_fractions(w):
    c = w._cloud
    sx, sy, _ = v3.project(c.points, v3.FRONT_VIEW)
    return np.column_stack([(sx + 1) / 2, (sy + 1) / 2])


def _plot_fractions(ed, x, y, xs, ys):
    """Axes fractions the editor's 2D plot gives these stored values."""
    disp = ed.ax.transData.transform(np.column_stack([xs, ys]))
    return ed.ax.transAxes.inverted().transform(disp)


@pytest.mark.parametrize('x,y,scales,cap', [
    ('SSC-A', 'FL1-A', {'SSC-A': 'logicle'}, None),    # linear shown logicle
    ('FSC-A', 'FL2-A', {'FL2-A': 'log'}, None),         # logicle shown log
    ('FL1-A', 'SSC-A', {'SSC-A': 'log'}, None),         # native log
    ('FSC-A', 'SSC-A', {}, None),                       # both linear
    ('FL3-A', 'FL1-A', {'FL3-A': 'symlog'}, None),      # native symlog
    # A subset: the scale must still be anchored on the whole column, as
    # the 2D plot anchors it, not on the events drawn.
    ('SSC-A', 'FL1-A', {'SSC-A': 'logicle'}, 1500),
    ('FL3-A', 'FL1-A', {'FL3-A': 'symlog'}, 1500),
])
def test_front_view_matches_the_2d_plot(x, y, scales, cap):
    root, ed = _editor()
    try:
        ed._channel_scale.update(scales)
        s = ed._samples['S1']
        # The same window on both: an explicit range on each axis, which
        # the 2D plot applies as its limits.
        for ch in (x, y):
            vals = s.data[ch].to_numpy()
            lo, hi = np.percentile(vals, [2, 98])
            ed._channel_range[ch] = (float(lo), float(hi))
        ed.x_combo.set(ed._fmt_channel(x))
        ed.y_combo.set(ed._fmt_channel(y))
        ed.mode_var.set('dot')
        ed.max_points_var.set('All')
        ed._replot()
        root.update()
        w = _window(ed, root)
        if cap:
            w.cap_var.set(str(cap))
        _set_axes(w, x, y, 'FL2-A' if 'FL2-A' not in (x, y) else 'FL1-A')
        got = _front_fractions(w)
        df = w._frame
        want = _plot_fractions(ed, x, y, df[x].to_numpy(), df[y].to_numpy())
        n = len(s.data)
        loaded = cap or n
        assert len(got) > 0.8 * loaded * 0.96 * 0.5
        np.testing.assert_allclose(got, want, atol=1e-6)
        # Events outside either range are left off, as the 2D axes clip
        # them, and counted exactly; shown is every event loaded.
        pick = v3.seeded_subset(n, cap, v3.DEFAULT_SEED)
        rows = np.arange(n) if pick is None else pick
        inside = np.ones(len(rows), dtype=bool)
        for ch in (x, y):
            v = s.data[ch].to_numpy()[rows]
            lo, hi = ed._channel_range[ch]
            inside &= (v >= lo) & (v <= hi)
        assert w._cloud.shown == loaded
        assert w._cloud.off_axis == int((~inside).sum()) > 0
        assert len(df) == int(inside.sum())
        assert f"{w._cloud.off_axis:,} off the axes" in w.status.cget('text')
    finally:
        root.destroy()


@pytest.mark.parametrize('lo_end', [0.0, -5.0])
def test_a_log_range_from_zero_keeps_its_top_as_the_2d_plot_does(lo_end):
    """A range whose low end has no place on a log axis: the 2D plot drops
    that end and keeps the top, and so must the 3D plot's front view."""
    root, ed = _editor()
    try:
        ed._channel_scale['SSC-A'] = 'log'
        s = ed._samples['S1']
        hi = float(np.percentile(s.data['SSC-A'], 90))
        ed._channel_range['SSC-A'] = (lo_end, hi)
        ed.x_combo.set(ed._fmt_channel('SSC-A'))
        ed.y_combo.set(ed._fmt_channel('FL1-A'))
        ed.mode_var.set('dot')
        ed.max_points_var.set('All')
        ed._replot()
        root.update()
        assert ed.ax.get_xlim()[1] == pytest.approx(hi)
        w = _window(ed, root)
        _set_axes(w, 'SSC-A', 'FL1-A', 'FL2-A')
        df = w._frame
        want = _plot_fractions(ed, 'SSC-A', 'FL1-A', df['SSC-A'].to_numpy(),
                               df['FL1-A'].to_numpy())
        np.testing.assert_allclose(_front_fractions(w), want, atol=1e-6)
        above = int((s.data['SSC-A'].to_numpy() > hi).sum())
        assert w._cloud.off_axis == above > 0
    finally:
        root.destroy()


def test_front_view_matches_an_autoscaled_2d_plot_without_ranges():
    import matplotlib.pyplot as plt
    root, ed = _editor()
    try:
        ed._channel_scale['SSC-A'] = 'logicle'
        w = _window(ed, root)
        _set_axes(w, 'SSC-A', 'FL1-A', 'FL2-A')
        df = w._frame
        assert w._cloud.off_axis == 0
        fig, ax = plt.subplots()
        ax.scatter(df['SSC-A'], df['FL1-A'], s=1)
        for ch, setter in (('SSC-A', ax.set_xscale), ('FL1-A', ax.set_yscale)):
            data = ed._samples['S1'].data[ch].to_numpy()
            setter('function', functions=ed._axis_view_funcs(ch, data))
        ax.autoscale_view()
        disp = ax.transData.transform(df[['SSC-A', 'FL1-A']].to_numpy())
        want = ax.transAxes.inverted().transform(disp)
        plt.close(fig)
        np.testing.assert_allclose(_front_fractions(w), want, atol=1e-6)
    finally:
        root.destroy()


def test_axes_choice_reaches_the_points_and_the_labels():
    root, ed = _editor()
    try:
        w = _window(ed, root)
        _set_axes(w, 'FL1-A', 'FL2-A', 'FSC-A')
        a = w._cloud.points.copy()
        assert w.scene().labels == ('FL1-A', 'FL2-A', 'FSC-A')
        _set_axes(w, 'FL1-A', 'FL2-A', 'SSC-A')
        b = w._cloud.points
        np.testing.assert_allclose(a[:, :2], b[:, :2])
        assert not np.allclose(a[:, 2], b[:, 2])
        assert w.scene().labels[2] == 'SSC-A'
    finally:
        root.destroy()


def test_cap_and_seed_pick_a_reproducible_labelled_subset():
    root, ed = _editor(n=5000)
    try:
        w = _window(ed, root)
        assert w._cloud.shown == 5000 and w._cloud.total == 5000
        assert w.status.cget('text').startswith('Showing all 5,000 events')
        w.cap_var.set('2,000')
        w.seed_var.set('11')
        w.refresh()
        first = w._frame_rows.copy()
        assert w._cloud.shown == 2000 and w._cloud.total == 5000
        assert ("Showing 2,000 of 5,000 events (random subset, seed 11)"
                in w.status.cget('text'))
        w.refresh()
        np.testing.assert_array_equal(w._frame_rows, first)
        w.seed_var.set('12')
        w.refresh()
        assert not np.array_equal(w._frame_rows, first)
        w.cap_var.set('3k')
        w.refresh()
        assert w._cloud.shown == 3000
        # Not a cap: put back to the last good one.
        w.cap_var.set('12')
        w.seed_var.set('-1')
        w.refresh()
        assert w.cap_var.get() == '3,000' and w._cloud.shown == 3000
        assert w.seed_var.get() == '12'
    finally:
        root.destroy()


def test_population_choice_draws_the_gate_or_everything():
    root, ed = _editor(n=4000)
    try:
        gid = ed._add_gate({'kind': 'threshold', 'name': 'Bright',
                            'channel': 'FSC-A', 'op': '>', 'value': 1.2e5,
                            'parent_id': None}, audit=False)
        root.update()
        inside = int((ed._samples['S1'].data['FSC-A'] > 1.2e5).sum())
        w = _window(ed, root)
        w.pop_var.set('Bright')
        w.refresh()
        assert w._cloud.total == inside
        # As shown: the main plot filtering to the gated events.
        ed.gate_display_var.set('filter')
        w.pop_var.set('As shown in the plot')
        w.refresh()
        assert w._cloud.total == inside
        # All events: every event, whatever the main plot filters to.
        w.pop_var.set('All events')
        w.refresh()
        assert w._cloud.total == 4000
        assert w._gate_map == {'Bright': gid}
    finally:
        root.destroy()


def test_colour_modes_color_the_right_events():
    root, ed = _editor(n=3000)
    try:
        gid = ed._add_gate({'kind': 'threshold', 'name': 'Bright',
                            'channel': 'FSC-A', 'op': '>', 'value': 1.2e5,
                            'parent_id': None, 'color': '#0000ff'}, audit=False)
        ed._sample_gates['S1'][gid]['color'] = '#0000ff'
        w = _window(ed, root)
        # A subset, so the gate's mask must be read at the drawn rows.
        w.cap_var.set('1,000')
        w.refresh()
        assert w._cloud.shown == 1000
        dens = w._cloud.colors.copy()
        w.color_mode_var.set('Gate')
        w._on_color_mode()
        assert w.color_target_var.get() == 'Bright'
        fsc = w._frame['FSC-A'].to_numpy()
        cols = w._cloud.colors
        assert (cols[fsc > 1.2e5] == (0, 0, 255)).all()
        assert (cols[fsc <= 1.2e5] == v3.OUTSIDE_GREY).all()
        w.color_mode_var.set('Channel')
        w._on_color_mode()
        w.color_target_var.set('FL2-A')
        w._recolor_only()
        np.testing.assert_array_equal(
            w._cloud.colors, v3.marker_colors(w._frame['FL2-A'].to_numpy()))
        w.color_mode_var.set('Cluster')
        w._on_color_mode()
        assert w.color_target_var.get() == 'cluster'
        cl = w._frame['cluster'].to_numpy()
        cols = w._cloud.colors
        assert len({tuple(c) for c in cols[cl == 0]}) == 1
        assert len({tuple(c) for c in cols[cl == 1]}) == 1
        assert tuple(cols[cl == 0][0]) != tuple(cols[cl == 1][0])
        w.color_mode_var.set('Density')
        w._on_color_mode()
        np.testing.assert_array_equal(w._cloud.colors, dens)
        assert str(w.color_target_combo.state()).count('disabled')
    finally:
        root.destroy()


def test_point_size_drag_zoom_and_view_buttons():
    root, ed = _editor()
    try:
        w = _window(ed, root)
        assert w.scene().point_size == v3.DEFAULT_POINT_SIZE
        w.size_var.set('4')
        w._recolor_only()
        assert w.scene().point_size == 4
        start = w.view
        assert (start.azimuth, start.elevation) == (v3.DEFAULT_AZIMUTH,
                                                    v3.DEFAULT_ELEVATION)

        class E:
            def __init__(self, x, y, delta=0):
                self.x, self.y, self.delta = x, y, delta
        w._on_press(E(100, 100))
        w._on_drag(E(140, 90))
        assert w.view.azimuth == pytest.approx(start.azimuth + 20)
        assert w.view.elevation == pytest.approx(start.elevation - 5)
        w._on_wheel(E(0, 0, 120))
        assert w.view.zoom == pytest.approx(1.1)
        w._on_wheel(E(0, 0, -120))
        assert w.view.zoom == pytest.approx(1.0)
        w.front_view()
        assert w.view == v3.FRONT_VIEW
        w.reset_view()
        assert w.view == v3.View()
        root.update()
        assert w.last_frame_ms > 0
        assert 'frame' in w.status.cget('text')
    finally:
        root.destroy()


def test_save_png_gif_and_frames_choice(tmp_path):
    from PIL import Image
    root, ed = _editor(n=1500)
    try:
        w = _window(ed, root)
        png = w.save_png(str(tmp_path / 'v.png'))
        with Image.open(png) as im:
            assert im.size == w._export_size()
            assert im.size[0] % 2 == 0 and im.size[1] % 2 == 0
        for n in v3.TURNTABLE_FRAMES:
            w.frames_var.set(str(n))
            job = w.save_gif(str(tmp_path / f't{n}.gif'))
            assert job is not None
            assert wait_for_job(w.busy)
            with Image.open(tmp_path / f't{n}.gif') as im:
                assert im.n_frames == n
                assert im.info.get('loop') == 0
                assert im.size == w._export_size()
            assert w.status.cget('text').startswith('Saved')
    finally:
        root.destroy()


def test_mp4_button_follows_the_encoder(monkeypatch, tmp_path):
    root, ed = _editor(n=800)
    try:
        monkeypatch.setattr(v3, 'mp4_encoder', lambda: None)
        w = _window(ed, root)
        assert 'disabled' in w.mp4_btn.state()
        shown = []
        import openflo.ui_view3d as uv
        monkeypatch.setattr(uv.ui_errors, 'showinfo',
                            lambda *a, **k: shown.append(a))
        assert w.save_mp4(str(tmp_path / 'x.mp4')) is None
        assert shown
        w.destroy()
        written = []
        monkeypatch.setattr(v3, 'mp4_encoder', lambda: '/fake/ffmpeg')
        monkeypatch.setattr(v3, 'save_mp4',
                            lambda frames, p, exe, n: written.append(
                                (len(list(frames)), p, exe, n)))
        w = _window(ed, root)
        assert 'disabled' not in w.mp4_btn.state()
        w.frames_var.set('72')
        w.save_mp4(str(tmp_path / 'y.mp4'))
        assert wait_for_job(w.busy)
        assert written == [(72, str(tmp_path / 'y.mp4'), '/fake/ffmpeg', 72)]
    finally:
        root.destroy()


def test_selected_gate_is_the_starting_population(monkeypatch):
    root, ed = _editor(n=2000)
    try:
        gid = ed._add_gate({'kind': 'threshold', 'name': 'Bright',
                            'channel': 'FSC-A', 'op': '>', 'value': 1.2e5,
                            'parent_id': None}, audit=False)
        monkeypatch.setattr(ed, '_selected_gate_id', lambda: gid)
        w = _window(ed, root)
        assert w.pop_var.get() == 'Bright'
        assert w._cloud.total == int(
            (ed._samples['S1'].data['FSC-A'] > 1.2e5).sum())
    finally:
        root.destroy()


def test_plot_background_follows_the_editor():
    from openflo.theme import THEME_ALL_DARK, THEME_LIGHT, THEMES
    root, ed = _editor(n=800)
    try:
        w = _window(ed, root)
        light = v3.hex_rgb(THEMES[THEME_LIGHT]['plot_bg'])
        dark = v3.hex_rgb(THEMES[THEME_ALL_DARK]['plot_bg'])
        ed._plot_bg_var.set('light')
        assert w.scene().background == light
        ed._plot_bg_var.set('dark')
        sc = w.scene()
        assert sc.background == dark
        assert sc.ink == v3.hex_rgb(THEMES[THEME_ALL_DARK]['plot_fg'])
    finally:
        root.destroy()


def test_a_3d_embedding_opens_on_its_three_components():
    rng = np.random.default_rng(4)
    extra = {f'UMAP{i}': rng.normal(0, 1, 3000) for i in (1, 2, 3)}
    root, ed = _editor(extra=extra)
    try:
        w = _window(ed, root)
        assert w._axes() == ['UMAP1', 'UMAP2', 'UMAP3']
        assert w._cloud.shown == 3000
    finally:
        root.destroy()


def test_sample_choice_switches_the_data():
    from openflo.pipeline import FlowSample
    root, ed = _editor(n=1000)
    try:
        f2 = _sample_frame(2500, seed=9)
        f2['FL4-A'] = np.linspace(1.0, 100.0, 2500)
        s2 = FlowSample.from_dataframe(f2, name='S2')
        s2.apply_transform(['FL1-A', 'FL2-A'])
        ed._samples['S2'] = s2
        ed._sample_order.append('S2')
        ed._sample_trial['S2'] = 'T'
        ed._sample_plot_enabled['S2'] = True
        ed._sample_gates['S2'] = {}
        ed._sample_gate_order['S2'] = []
        ed._sample_gates['S2']['g9'] = {'kind': 'threshold', 'name': 'Only2',
                                        'channel': 'FSC-A', 'op': '>',
                                        'value': 1e5, 'parent_id': None}
        ed._sample_gate_order['S2'].append('g9')
        w = _window(ed, root)
        assert w._cloud.total == 1000
        assert 'Only2' not in w.pop_combo['values']
        w.sample_var.set('S2')
        w._on_sample()
        assert w._cloud.total == 2500
        # Its own gates and channels are offered.
        assert 'Only2' in w.pop_combo['values']
        assert 'FL4-A' in w.axis_combos[0]['values']
    finally:
        root.destroy()


def test_menu_opens_one_window_and_says_when_nothing_is_loaded():
    root, ed, _gui = _editor_or_skip()
    try:
        assert ed._open_view3d() is None
        assert 'Load a sample' in ed.status_var.get()
    finally:
        root.destroy()
    root, ed = _editor()
    try:
        a = ed._open_view3d()
        b = ed._open_view3d()
        assert a is b
    finally:
        root.destroy()


def _shown(w):
    """The window put on screen (test windows open withdrawn), so Tk
    delivers the events generated on it."""
    w.deiconify()
    w.update()
    return w


def _drag(c, x, y):
    c.event_generate('<Motion>', x=x, y=y, state=0x100)   # Button1 held


def _turned(w):
    """The view turned and zoomed away from where the window opened."""
    w.set_view(v3.View(azimuth=63.0, elevation=-21.0, zoom=1.7))
    return w.view


def test_exports_are_the_view_on_screen(tmp_path, monkeypatch):
    from PIL import Image
    root, ed = _editor(n=1200)
    try:
        w = _window(ed, root)
        view = _turned(w)
        png = w.save_png(str(tmp_path / 'v.png'))
        with Image.open(png) as im:
            got = np.asarray(im.convert('RGB'))
        size = w._export_size()
        want = np.asarray(v3.render(w.scene(), view, *size))
        np.testing.assert_array_equal(got, want)
        assert not os.path.exists(png + '.part')
        made = []
        monkeypatch.setattr(v3, 'save_gif', lambda frames, p, n: made.extend(frames))
        w.frames_var.set('36')
        w.save_gif(str(tmp_path / 't.gif'))
        assert wait_for_job(w.busy)
        assert len(made) == 36
        for k in (0, 1, 9):
            np.testing.assert_array_equal(
                np.asarray(made[k]),
                np.asarray(v3.render(w.scene(), view.turned(d_azimuth=10.0 * k), *size)))
    finally:
        root.destroy()


def test_linux_wheel_buttons_zoom_in_and_out():
    root, ed = _editor(n=500)
    try:
        w = _shown(_window(ed, root))
        w.canvas.event_generate('<Button-4>', x=20, y=20)
        root.update()
        assert w.view.zoom == pytest.approx(1.1)
        w.canvas.event_generate('<Button-5>', x=20, y=20)
        w.canvas.event_generate('<Button-5>', x=20, y=20)
        root.update()
        assert w.view.zoom == pytest.approx(1 / 1.1)
    finally:
        root.destroy()


def test_the_mouse_wheel_zooms_by_how_far_it_turned():
    """A notch (delta 120) zooms one step; a precision touchpad's small
    deltas zoom by their share of a notch, not a whole step each."""
    root, ed = _editor(n=500)
    try:
        w = _shown(_window(ed, root))
        # delta: how far the wheel turned (Tk units), not a tolerance.
        w.canvas.event_generate('<MouseWheel>', x=20, y=20, delta=120)
        root.update()
        assert w.view.zoom == pytest.approx(1.1)
        for _ in range(4):
            w.canvas.event_generate('<MouseWheel>', x=20, y=20, delta=-30)
        root.update()
        assert w.view.zoom == pytest.approx(1.0)
        # delta: how far the wheel turned (Tk units), not a tolerance.
        w.canvas.event_generate('<MouseWheel>', x=20, y=20, delta=24)
        root.update()
        assert w.view.zoom == pytest.approx(1.1 ** 0.2)
    finally:
        root.destroy()


def test_the_wheel_asks_tk_which_platform_it_is_on(monkeypatch):
    seen = []

    def units(system, version):
        seen.append((system, version))
        return 1                                   # as Tk 8.6 on macOS
    monkeypatch.setattr(v3, 'wheel_units', units)
    root, ed = _editor(n=500)
    try:
        w = _shown(_window(ed, root))
        # delta: how far the wheel turned (Tk units), not a tolerance.
        w.canvas.event_generate('<MouseWheel>', x=20, y=20, delta=2)
        root.update()
        assert w.view.zoom == pytest.approx(1.1 ** 2)
        assert seen == [(root.tk.call('tk', 'windowingsystem'),
                         str(root.tk.call('info', 'tclversion')))]
    finally:
        root.destroy()


def test_picking_a_point_size_redraws(monkeypatch):
    root, ed = _editor(n=500)
    try:
        w = _shown(_window(ed, root))
        drawn = []
        monkeypatch.setattr(w, 'redraw', lambda: drawn.append(w.scene().point_size))
        w.size_var.set('3')
        w.size_combo.event_generate('<<ComboboxSelected>>')
        root.update()
        assert drawn and drawn[-1] == 3
    finally:
        root.destroy()


def test_enter_in_max_events_reloads():
    root, ed = _editor(n=3000)
    try:
        w = _shown(_window(ed, root))
        assert w._cloud.shown == 3000
        w.cap_var.set('1,000')
        w.cap_entry.focus_force()
        w.cap_entry.event_generate('<Return>')
        root.update()
        assert w._cloud.shown == 1000
    finally:
        root.destroy()


def test_releasing_the_button_ends_the_drag():
    root, ed = _editor(n=500)
    try:
        w = _shown(_window(ed, root))
        root.update()
        start = w.view
        c = w.canvas
        c.event_generate('<ButtonPress-1>', x=100, y=100)
        _drag(c, 120, 100)
        c.event_generate('<ButtonRelease-1>', x=120, y=100)
        root.update()
        assert w.view.azimuth == pytest.approx(start.azimuth + 10)
        # A new press far away starts a new drag: no jump.
        c.event_generate('<ButtonPress-1>', x=400, y=300)
        _drag(c, 402, 300)
        root.update()
        assert w.view.azimuth == pytest.approx(start.azimuth + 11)
        assert w.view.elevation == pytest.approx(start.elevation)
    finally:
        root.destroy()


def test_a_failed_save_shows_why_and_a_canceled_one_saves_nothing(tmp_path, monkeypatch):
    import openflo.ui_view3d as uv
    root, ed = _editor(n=500)
    try:
        w = _window(ed, root)
        shown = []
        monkeypatch.setattr(uv.ui_errors, 'showerror', lambda *a, **k: shown.append(a))

        def broken(frames, path, n):
            next(iter(frames))
            raise RuntimeError('disk full')
        monkeypatch.setattr(v3, 'save_gif', broken)
        w.save_gif(str(tmp_path / 'x.gif'))
        wait_for_job(w.busy)
        root.update()
        assert shown and 'disk full' in shown[0][1]
        assert w.status.cget('text') == 'Could not save the turntable.'
        monkeypatch.undo()
        # Cancel part-way through the turn: the file that was there stays.
        old = tmp_path / 'keep.gif'
        old.write_bytes(b'older file')
        orig = v3.render
        held = {}

        def render_then_cancel(*a, **k):
            if 'job' in held:
                held['job'].cancel()
            return orig(*a, **k)
        monkeypatch.setattr(v3, 'render', render_then_cancel)
        held['job'] = w.save_gif(str(old))
        wait_for_job(w.busy)
        root.update()
        assert w.status.cget('text') == 'Canceled; nothing was saved.'
        assert old.read_bytes() == b'older file'
        assert not os.path.exists(str(old) + '.part')
    finally:
        root.destroy()


def test_cluster_colours_are_the_2d_plots():
    root, ed = _editor(n=1500)
    try:
        # Cluster 1 has a population in the tree, in its own colour: the
        # 2D plot (and so the cloud) takes that colour for its dots.
        ed._sample_gates['S1']['c1'] = {'kind': 'cluster', 'channel': 'cluster',
                                        'cluster_id': 1, 'name': 'Cluster 1',
                                        'color': '#123456', 'parent_id': None}
        w = _window(ed, root)
        w.color_mode_var.set('Cluster')
        w._on_color_mode()
        vals = w._frame['cluster'].to_numpy()
        want = {v: v3.hex_rgb(c) for v, c, _ in ed._label_colours('cluster', vals, ['S1'])}
        cols = w._cloud.colors
        assert want[1] == (0x12, 0x34, 0x56)
        for v, rgb in want.items():
            assert (cols[vals == v] == rgb).all()
    finally:
        root.destroy()


def test_a_turntable_is_saved_at_most_1280_wide(monkeypatch):
    root, ed = _editor(n=500)
    try:
        w = _window(ed, root)
        monkeypatch.setattr(w.canvas, 'winfo_width', lambda: 3000)
        monkeypatch.setattr(w.canvas, 'winfo_height', lambda: 1500)
        assert w._export_size() == (1280, 640)
    finally:
        root.destroy()


def test_each_frame_is_drawn_as_the_writer_takes_it(tmp_path, monkeypatch):
    root, ed = _editor(n=500)
    try:
        w = _window(ed, root)
        drawn = []
        orig = v3.render
        monkeypatch.setattr(v3, 'render', lambda *a, **k: drawn.append(1) or orig(*a, **k))
        seen = []

        def writer(frames, p, n):
            for _f in frames:
                seen.append(len(drawn))
        monkeypatch.setattr(v3, 'save_gif', writer)
        w.frames_var.set('36')
        w.save_gif(str(tmp_path / 't.gif'))
        assert wait_for_job(w.busy)
        assert seen == list(range(1, 37))
    finally:
        root.destroy()


def test_a_failed_png_leaves_the_old_file(tmp_path, monkeypatch):
    root, ed = _editor(n=500)
    try:
        w = _window(ed, root)
        old = tmp_path / 'keep.png'
        old.write_bytes(b'older picture')
        orig = v3.render

        class Half:
            def __init__(self, im):
                self.im = im

            def save(self, path, **kw):
                open(path, 'wb').write(b'half')
                raise OSError('disk full')
        monkeypatch.setattr(v3, 'render', lambda *a, **k: Half(orig(*a, **k)))
        with pytest.raises(OSError):
            w.save_png(str(old))
        assert old.read_bytes() == b'older picture'
        assert not os.path.exists(str(old) + '.part')
    finally:
        root.destroy()
