"""openflo.view3d: the 3D plot's maths and pictures, without Tk.

Every check runs on generated data. The front view (azimuth 0, elevation 0)
is checked against matplotlib drawing the same X/Y with the same scales, the
subset against its seed, and the saved GIF / MP4 by decoding what was written.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from openflo import view3d as v3  # noqa: E402


def _cells(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.uniform(5e3, 2.5e5, n),                 # scatter-like, linear
            rng.lognormal(7.0, 1.2, n),                 # fluorescence-like
            rng.normal(0.0, 1.0, n))


def _mpl_fractions(x, y, xscale, yscale):
    """Axes fractions matplotlib gives each (x, y) after autoscaling a
    scatter of them on these scales."""
    fig, ax = plt.subplots()
    try:
        ax.scatter(x, y, s=1)
        for setter, sc in ((ax.set_xscale, xscale), (ax.set_yscale, yscale)):
            if isinstance(sc, tuple):
                setter(sc[0], **sc[1])
            else:
                setter(sc)
        ax.autoscale_view()
        fig.canvas.draw()
        disp = ax.transData.transform(np.column_stack([x, y]))
        return ax.transAxes.inverted().transform(disp)
    finally:
        plt.close(fig)


def _front_fractions(x, y, tx, ty):
    sx_screen, sy_screen = tx(x), ty(y)
    pts = np.column_stack([
        v3.normalize(sx_screen, v3.axis_bounds(sx_screen)),
        v3.normalize(sy_screen, v3.axis_bounds(sy_screen)),
        np.zeros(len(x))])
    sx, sy, _ = v3.project(pts, v3.FRONT_VIEW)
    return np.column_stack([(sx + 1) / 2, (sy + 1) / 2])


@pytest.mark.parametrize('xscale,yscale,x_around_zero', [
    ('linear', 'log', False),
    ('log', 'linear', False),
    # Values either side of zero, many inside the linear part: the
    # threshold changes where they land, not only an offset.
    (('symlog', {'linthresh': 100.0}), 'log', True),
])
def test_front_view_is_the_2d_plot_on_native_scales(xscale, yscale,
                                                    x_around_zero):
    x, y, z = _cells()
    if x_around_zero:
        x = z * 300.0

    def tr(sc):
        if isinstance(sc, tuple):
            return v3.screen_transform(sc[0], linthresh=sc[1]['linthresh'])
        return v3.screen_transform(sc)
    want = _mpl_fractions(x, y, xscale, yscale)
    got = _front_fractions(x, y, tr(xscale), tr(yscale))
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_front_view_is_the_2d_plot_on_a_logicle_view_of_a_linear_channel():
    # SSC-A stored linear, shown on logicle: the 2D plot uses a FuncScale
    # built by scales.view_funcs; the 3D axis must use the very same.
    from openflo.scales import view_funcs
    x, y, _ = _cells()
    fx = view_funcs('linear', 'logicle', x)
    fy = view_funcs('linear', 'logicle', y)
    assert fx is not None and fy is not None
    want = _mpl_fractions(x, y, ('function', {'functions': fx}),
                          ('function', {'functions': fy}))
    got = _front_fractions(x, y, v3.screen_transform('logicle', funcs=fx),
                           v3.screen_transform('logicle', funcs=fy))
    np.testing.assert_allclose(got, want, atol=1e-9)
    # The view is not linear: the same points on linear axes land elsewhere.
    lin = _front_fractions(x, y, v3.screen_transform('linear'),
                           v3.screen_transform('linear'))
    assert np.abs(lin - got).max() > 0.1


def test_log_axis_leaves_out_what_it_cannot_place():
    out = v3.screen_transform('log')(np.array([-5.0, 0.0, 10.0, 1000.0]))
    assert np.isnan(out[:2]).all()
    np.testing.assert_allclose(out[2:], [1.0, 3.0])


def test_explicit_range_sets_the_bounds():
    s = np.array([0.0, 1.0, 2.0])
    assert v3.axis_bounds(s, (-1.0, 4.0)) == (-1.0, 4.0)
    assert v3.axis_bounds(s, (4.0, -1.0)) == (-1.0, 4.0)
    lo, hi = v3.axis_bounds(s)
    assert (lo, hi) == pytest.approx((-0.1, 2.1))
    assert v3.axis_bounds(np.array([3.0, 3.0])) == (2.5, 3.5)
    assert v3.axis_bounds(np.array([np.nan])) == (0.0, 1.0)


def test_a_range_end_with_no_place_on_screen_is_autoscaled_alone():
    """A range from 0 on a log axis: matplotlib ignores the 0 and keeps
    the other end, so the 3D axis does too."""
    s = np.array([0.0, 1.0, 2.0])
    assert v3.axis_bounds(s, (-np.inf, 1.5)) == pytest.approx((-0.1, 1.5))
    assert v3.axis_bounds(s, (0.5, np.nan)) == pytest.approx((0.5, 2.1))
    assert v3.axis_bounds(s, (np.nan, -np.inf)) == pytest.approx((-0.1, 2.1))
    empty = np.array([np.nan])
    assert v3.axis_bounds(empty, (-np.inf, 4.0)) == (3.5, 4.5)
    assert v3.axis_bounds(empty, (np.nan, np.nan)) == (0.0, 1.0)


@pytest.mark.parametrize('delta,units,want', [
    (120, 120, 1.1),                 # one notch away: closer
    (-120, 120, 1 / 1.1),            # one notch toward: farther
    (240, 120, 1.1 ** 2),            # two notches in one event
    (30, 120, 1.1 ** 0.25),          # a touchpad's quarter notch
    (-12, 120, 1.1 ** -0.1),
    (6000, 120, 1.1 ** 3),           # at most three notches an event
    (-6000, 120, 1.1 ** -3),
    (2, 1, 1.1 ** 2),                # Tk 8.6 on macOS: scroll lines
    (0, 120, 1.0),
])
def test_the_wheel_zooms_by_the_notches_it_turned(delta, units, want):
    assert v3.wheel_zoom(delta, units) == pytest.approx(want)


@pytest.mark.parametrize('system,version,want', [
    ('win32', '8.6', 120), ('x11', '8.6', 120), ('aqua', '8.6', 1),
    ('aqua', '8.7', 120), ('aqua', '9.0', 120), ('x11', '9.0', 120),
])
def test_a_wheel_notch_is_120_units_but_a_line_on_old_macos_tk(system, version, want):
    assert v3.wheel_units(system, version) == want


def test_a_touchpads_small_steps_add_up_to_the_notches_they_make():
    z = v3.View()
    for _ in range(8):
        z = z.zoomed(v3.wheel_zoom(15))          # 8 x 15 = one notch
    assert z.zoom == pytest.approx(1.1)


# ── the subset ────────────────────────────────────────────────────────────

def test_subset_is_reproducible_from_its_seed():
    a = v3.seeded_subset(500_000, 200_000, 7)
    b = v3.seeded_subset(500_000, 200_000, 7)
    c = v3.seeded_subset(500_000, 200_000, 8)
    assert a is not None and b is not None and c is not None
    assert a.size == 200_000 and np.unique(a).size == 200_000
    assert np.all(np.diff(a) > 0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert v3.seeded_subset(1000, 1000, 7) is None
    assert v3.seeded_subset(1000, None, 7) is None


def test_subset_note_says_what_is_drawn():
    assert v3.subset_note(5000, 5000, 3) == "Showing all 5,000 events"
    assert v3.subset_note(200_000, 950_000, 3) == (
        "Showing 200,000 of 950,000 events (random subset, seed 3)")


# ── the view ──────────────────────────────────────────────────────────────

def test_front_view_shows_x_and_y_and_turns_about_y():
    p = np.array([[0.3, -0.4, 0.9]])
    sx, sy, d = v3.project(p, v3.FRONT_VIEW)
    assert (sx[0], sy[0], d[0]) == pytest.approx((0.3, -0.4, 0.9))
    sx, sy, d = v3.project(p, v3.View(azimuth=90.0, elevation=0.0))
    assert (sx[0], sy[0], d[0]) == pytest.approx((0.9, -0.4, -0.3))
    sx, sy, d = v3.project(p, v3.View(azimuth=0.0, elevation=90.0))
    assert (sx[0], sy[0], d[0]) == pytest.approx((0.3, -0.9, -0.4))


def test_azimuth_turns_first_then_elevation_tilts():
    # Turned 90 degrees then tilted 90: the X axis points up the screen
    # (tilting first would leave it pointing away from the viewer).
    sx, sy, d = v3.project(np.array([[1.0, 0.0, 0.0]]),
                           v3.View(azimuth=90.0, elevation=90.0))
    assert (sx[0], sy[0], d[0]) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_drag_wraps_azimuth_and_stops_elevation_at_the_poles():
    v = v3.View(azimuth=170.0, elevation=80.0, zoom=2.0)
    t = v.turned(d_azimuth=20.0, d_elevation=30.0)
    assert t.azimuth == pytest.approx(-170.0)
    assert t.elevation == 90.0 and t.zoom == 2.0
    assert v.turned(d_elevation=-400).elevation == -90.0
    assert v.zoomed(100).zoom == v3.MAX_ZOOM
    assert v.zoomed(0.001).zoom == v3.MIN_ZOOM
    assert v.zoomed(1.5).zoom == pytest.approx(3.0)


def test_pixels_put_up_on_top_and_scale_with_zoom():
    px, py = v3.to_pixels(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                          200, 100, 1.0)
    assert (px[0], py[0]) == (100.0, 50.0)
    assert px[1] > 100 and py[1] < 50
    px2, _ = v3.to_pixels(np.array([1.0]), np.array([0.0]), 200, 100, 2.0)
    assert px2[0] - 100 == pytest.approx(2 * (px[1] - 100))


# ── drawing ───────────────────────────────────────────────────────────────

def test_nearest_point_is_drawn_on_top():
    red, blue = (255, 0, 0), (0, 0, 255)
    colors = np.array([red, blue], dtype=np.uint8)
    px = np.array([5.2, 5.7])
    py = np.array([5.1, 5.9])
    for depth, want in (([0.9, -0.3], red), ([-0.9, 0.3], blue)):
        img = v3.rasterize(px, py, np.array(depth), colors, 10, 10, 1, (0, 0, 0))
        assert tuple(img[5, 5]) == want
        assert (img.reshape(-1, 3) != 0).any(axis=1).sum() == 1


def test_point_size_is_a_square_of_pixels():
    one = np.array([[255, 255, 255]], dtype=np.uint8)
    for size in v3.POINT_SIZES:
        img = v3.rasterize(np.array([10.5]), np.array([10.5]), np.zeros(1),
                           one, 30, 30, size, (0, 0, 0))
        assert int((img[..., 0] > 0).sum()) == size * size
    edge = v3.rasterize(np.array([0.2]), np.array([0.2]), np.zeros(1), one,
                        30, 30, 4, (0, 0, 0))
    # Columns and rows -2 .. 1 around pixel 0: only 0 and 1 are inside.
    assert int((edge[..., 0] > 0).sum()) == 4


def test_render_draws_points_and_axis_names():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, (2000, 3))
    cols = np.full((2000, 3), (200, 30, 30), dtype=np.uint8)
    img = v3.render(v3.Scene(pts, cols, labels=('FL1-A', 'FL2-A', 'FL3-A')),
                    v3.View(), 320, 240)
    a = np.asarray(img)
    assert img.size == (320, 240)
    assert ((a[..., 0] == 200) & (a[..., 1] == 30)).sum() > 500
    empty = np.asarray(v3.render(v3.Scene(np.zeros((0, 3)),
                                          np.zeros((0, 3), np.uint8)),
                                 v3.View(), 320, 240))
    assert (empty != 255).any()          # the cube and the labels


# ── colours ───────────────────────────────────────────────────────────────

def test_density_colors_crowded_points_differently_from_lonely_ones():
    rng = np.random.default_rng(3)
    core = rng.normal(0.0, 0.05, (3000, 3))
    lonely = np.array([[0.9, 0.9, 0.9]])
    cols = v3.density_colors(np.vstack([core, lonely]), 'viridis')
    from openflo.theme import plt_get_cmap
    lo = (np.asarray(plt_get_cmap('viridis')(0.0))[:3] * 255 + .5).astype(int)
    near_lo_lonely = np.abs(cols[-1].astype(int) - lo).max()
    near_lo_core = np.abs(cols[:3000].astype(int) - lo).max(axis=1)
    assert near_lo_lonely < 40
    assert np.median(near_lo_core) > 120


def test_density_colors_use_the_2d_plots_gamma():
    # Two piles far apart, one half the other: the smaller sits at
    # 0.5 ** 0.4 of the color map (the 2D plot's PowerNorm), not at 0.5.
    pts = np.vstack([np.full((2000, 3), 0.5), np.full((1000, 3), -0.5)])
    cols = v3.density_colors(pts, 'viridis')
    from openflo.theme import plt_get_cmap
    cmap = plt_get_cmap('viridis')
    want = (np.asarray(cmap(0.5 ** 0.4))[:3] * 255 + .5).astype(int)
    assert np.abs(cols[-1].astype(int) - want).max() <= 1
    top = (np.asarray(cmap(1.0))[:3] * 255 + .5).astype(int)
    assert np.abs(cols[0].astype(int) - top).max() <= 1


def test_density_counts_the_neighborhood_not_only_the_bin():
    # One point in the bin next to a pile reads as crowded: the histogram is
    # smoothed by a bin, so a neighbor's pile counts.
    w = 2.0 / v3.DENSITY_BINS
    pile = np.full((1000, 3), 0.5 * w)                 # centre of a bin
    beside = np.array([[1.5 * w, 0.5 * w, 0.5 * w]])   # the next bin over
    far = np.array([[-0.9, -0.9, -0.9]])
    cols = v3.density_colors(np.vstack([pile, beside, far]), 'viridis')
    from openflo.theme import plt_get_cmap
    lo = (np.asarray(plt_get_cmap('viridis')(0.0))[:3] * 255 + .5).astype(int)
    assert np.abs(cols[1000].astype(int) - lo).max() > 3 * np.abs(
        cols[1001].astype(int) - lo).max()


def test_marker_colors_follow_the_value_and_grey_the_missing():
    vals = np.r_[np.linspace(0, 100, 999), np.nan]
    cols = v3.marker_colors(vals, 'viridis')
    from openflo.theme import plt_get_cmap
    top = (np.asarray(plt_get_cmap('viridis')(1.0))[:3] * 255 + .5).astype(int)
    assert np.abs(cols[998].astype(int) - top).max() <= 2
    assert tuple(cols[-1]) == v3.OUTSIDE_GREY
    assert not np.array_equal(cols[0], cols[998])


def test_one_outlier_does_not_squeeze_the_marker_colors():
    vals = np.r_[np.linspace(0, 1, 999), 1e6]
    cols = v3.marker_colors(vals, 'viridis')
    from openflo.theme import plt_get_cmap
    top = (np.asarray(plt_get_cmap('viridis')(1.0))[:3] * 255 + .5).astype(int)
    assert np.abs(cols[995].astype(int) - top).max() <= 30


def test_category_and_gate_colors():
    cats = v3.category_colors(np.array([0, 1, 2, 1]),
                              [(0, '#ff0000'), (1, '#00ff00')])
    assert [tuple(c) for c in cats] == [(255, 0, 0), (0, 255, 0),
                                        v3.OUTSIDE_GREY, (0, 255, 0)]
    g = v3.gate_colors(np.array([True, False]), '#0000ff')
    assert [tuple(c) for c in g] == [(0, 0, 255), v3.OUTSIDE_GREY]


# ── turntable ─────────────────────────────────────────────────────────────

def test_turntable_turns_once_keeping_tilt_and_zoom():
    start = v3.View(azimuth=10.0, elevation=20.0, zoom=1.5)
    for n in v3.TURNTABLE_FRAMES:
        views = v3.turntable_views(start, n)
        assert len(views) == n
        assert views[0] == start
        steps = np.diff([v.azimuth for v in views]) % 360
        np.testing.assert_allclose(steps, 360.0 / n)
        assert {v.elevation for v in views} == {20.0}
        assert {v.zoom for v in views} == {1.5}
    with pytest.raises(ValueError):
        v3.turntable_views(start, 0)
    # Each frame's time is rounded to a whole millisecond: 36 frames x 0.5 ms.
    assert v3.frame_ms(36) * 36 == pytest.approx(6000, abs=18)
    assert v3.frame_ms(72) < v3.frame_ms(36)


def _scene(n=500):
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1, 1, (n, 3))
    return v3.Scene(pts, v3.density_colors(pts, 'viridis'))


@pytest.mark.parametrize('n', v3.TURNTABLE_FRAMES)
def test_gif_decodes_with_its_frames_size_and_loop(tmp_path, n):
    seen = []
    frames = v3.turntable_frames(_scene(), v3.View(), n, 120, 90,
                                 progress=lambda i, t: seen.append((i, t)))
    assert seen == []                            # nothing rendered until asked
    path = tmp_path / 'turn.gif'
    v3.save_gif(frames, str(path), n)
    assert seen[-1] == (n, n) and len(seen) == n
    with Image.open(path) as im:
        assert im.format == 'GIF'
        assert im.n_frames == n
        assert im.size == (120, 90)
        assert im.info.get('loop') == 0          # loops forever
        assert im.info.get('duration') == pytest.approx(v3.frame_ms(n), abs=10)
        first = np.asarray(im.convert('RGB'))
        im.seek(1)
        assert im.disposal_method == 1           # each frame left in place
        im.seek(n // 4)
        quarter = np.asarray(im.convert('RGB'))
    want = np.asarray(v3.render(_scene(), v3.View(), 120, 90))
    # The first frame is the view given (palette rounding aside).
    assert np.mean(np.abs(first.astype(int) - want.astype(int))) < 3
    assert not np.array_equal(quarter, first)


def test_gif_frames_reach_the_writer_as_palette_images(tmp_path, monkeypatch):
    """Pillow's GIF writer keeps every frame it is given until the file is
    done, so each goes in as a 256-color palette image (a third of RGB)."""
    modes = []
    real = Image.Image.save

    def spy(im, fp, *a, **k):
        frames = list(k.pop('append_images', []))
        modes.extend([im.mode] + [f.mode for f in frames])
        return real(im, fp, *a, append_images=frames, **k)
    monkeypatch.setattr(Image.Image, 'save', spy)
    frames = v3.turntable_frames(_scene(), v3.View(), 36, 120, 90)
    v3.save_gif(frames, str(tmp_path / 'turn.gif'), 36)
    assert modes == ['P'] * 36


def test_saving_nothing_is_refused(tmp_path):
    with pytest.raises(ValueError):
        v3.save_gif([], str(tmp_path / 'x.gif'), 36)
    with pytest.raises(ValueError):
        v3.save_mp4([], str(tmp_path / 'x.mp4'), 'ffmpeg', 36)
    assert os.listdir(tmp_path) == []


def test_mp4_command_reads_rgb_frames_and_writes_h264():
    cmd = v3.mp4_command('/opt/ff', 320, 240, 6.0, 'out.mp4.part')
    assert cmd[0] == '/opt/ff' and cmd[-3:] == ['-f', 'mp4', 'out.mp4.part']
    joined = ' '.join(cmd)
    for part in ('-f rawvideo', '-pix_fmt rgb24', '-s 320x240', '-r 6',
                 '-i -', '-c:v libx264', '-pix_fmt yuv420p'):
        assert part in joined
    assert v3.even_size(321, 241) == (320, 240)
    assert v3.even_size(1, 1) == (2, 2)


def test_a_turntable_is_saved_no_larger_than_1280_on_its_long_side():
    assert v3.MAX_EXPORT_SIDE == 1280
    assert v3.export_size(800, 600) == (800, 600)
    assert v3.export_size(801, 601) == (800, 600)
    assert v3.export_size(1920, 1040) == (1280, 692)
    assert v3.export_size(1400, 2560) == (700, 1280)
    assert v3.export_size(3840, 2160) == (1280, 720)


def _fake_ffmpeg(tmp_path, code=0, read=True):
    """A stand-in encoder: writes the byte count it read and its argv to the
    output path, or fails with ``code`` (with ``read`` False, at once,
    without reading its input, as ffmpeg does with no libx264)."""
    script = tmp_path / 'fake_ffmpeg.py'
    script.write_text(textwrap.dedent(f"""
        import sys
        if not {read}:
            sys.stderr.write("Unknown encoder 'libx264'")
            sys.exit(1)
        data = sys.stdin.buffer.read()
        if {code}:
            sys.stderr.write('no encoder here')
            sys.exit({code})
        open(sys.argv[-1], 'w').write(str(len(data)) + ' ' + ' '.join(sys.argv[1:]))
    """))
    if os.name == 'nt':
        exe = tmp_path / 'fake_ffmpeg.bat'
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        exe = tmp_path / 'fake_ffmpeg'
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_mp4_pipes_every_frame_to_the_encoder(tmp_path, monkeypatch):
    procs = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(v3.subprocess, 'Popen',
                        lambda *a, **k: procs.append(real_popen(*a, **k)) or procs[-1])
    frames = v3.turntable_frames(_scene(), v3.View(), 36, 64, 48)
    out = tmp_path / 'turn.mp4'
    v3.save_mp4(frames, str(out), _fake_ffmpeg(tmp_path), 36)
    assert procs[0].stdin.closed and procs[0].stderr.closed
    count, *argv = out.read_text().split(' ')
    assert int(count) == 36 * 64 * 48 * 3
    assert '64x48' in argv and '6' in argv       # 36 frames in six seconds
    assert argv[-1] == str(out) + '.part'        # written aside, then moved
    assert not os.path.exists(str(out) + '.part')


def _one_alive_frames(n, w=64, h=48):
    """Turntable frames that check, as each is made, that the one before
    has already been let go: frames are streamed, never all held."""
    import gc
    import weakref
    live = []
    for im in v3.turntable_frames(_scene(), v3.View(), n, w, h):
        gc.collect()
        # The one just written may still be in the writer's hands; none
        # before it.
        assert sum(r() is not None for r in live[:-1]) == 0, 'earlier frames are still held'
        live.append(weakref.ref(im))
        yield im
        del im


def test_frames_are_written_one_at_a_time(tmp_path):
    v3.save_mp4(_one_alive_frames(36), str(tmp_path / 'a.mp4'),
                _fake_ffmpeg(tmp_path), 36)
    v3.save_gif(_one_alive_frames(36), str(tmp_path / 'a.gif'), 36)
    with Image.open(tmp_path / 'a.gif') as im:
        assert im.n_frames == 36


def test_mp4_says_why_the_encoder_failed(tmp_path):
    frames = v3.turntable_frames(_scene(), v3.View(), 36, 64, 48)
    with pytest.raises(RuntimeError, match='no encoder here'):
        v3.save_mp4(frames, str(tmp_path / 'x.mp4'),
                    _fake_ffmpeg(tmp_path, code=3), 36)
    odd = (f.resize((63, 48)) for f in v3.turntable_frames(_scene(), v3.View(), 36, 64, 48))
    with pytest.raises(ValueError, match='even'):
        v3.save_mp4(odd, str(tmp_path / 'x.mp4'), _fake_ffmpeg(tmp_path), 36)


def test_an_encoder_that_quits_without_reading_still_says_why(tmp_path):
    """Large frames fill the pipe after ffmpeg has gone: the write fails
    (BrokenPipeError) and ffmpeg's own message is what the user sees."""
    frames = v3.turntable_frames(_scene(), v3.View(), 36, 640, 480)
    with pytest.raises(RuntimeError, match="Unknown encoder 'libx264'"):
        v3.save_mp4(frames, str(tmp_path / 'x.mp4'),
                    _fake_ffmpeg(tmp_path, read=False), 36)
    assert os.listdir(tmp_path) == ['fake_ffmpeg.py', 'fake_ffmpeg']


class _WindowsPipe:
    """ffmpeg gone on Windows: a write raises OSError EINVAL, not
    BrokenPipeError. Keeps the Popen keyword arguments it was given."""
    seen: dict = {}

    def __init__(self, cmd, **kw):
        _WindowsPipe.seen = kw
        import io

        class In:
            def write(self, _b):
                raise OSError(22, 'Invalid argument')

            def close(self):
                raise OSError(22, 'Invalid argument')
        self.stdin = In()
        self.stderr = io.BytesIO(b"Unknown encoder 'libx264'")

    def wait(self):
        return 1

    def kill(self):
        pass


def test_on_windows_the_reason_comes_through_and_no_console_opens(tmp_path, monkeypatch):
    monkeypatch.setattr(v3.subprocess, 'Popen', _WindowsPipe)
    monkeypatch.setattr(v3.sys, 'platform', 'win32')
    frames = v3.turntable_frames(_scene(), v3.View(), 36, 64, 48)
    with pytest.raises(RuntimeError, match="Unknown encoder 'libx264'"):
        v3.save_mp4(frames, str(tmp_path / 'x.mp4'), 'ffmpeg', 36)
    assert _WindowsPipe.seen['creationflags'] == 0x08000000   # CREATE_NO_WINDOW
    monkeypatch.setattr(v3.sys, 'platform', 'linux')
    with pytest.raises(RuntimeError):
        v3.save_mp4(v3.turntable_frames(_scene(), v3.View(), 2, 64, 48),
                    str(tmp_path / 'x.mp4'), 'ffmpeg', 2)
    assert 'creationflags' not in _WindowsPipe.seen


def _stopped_at(k):
    """Turntable frames whose writer is stopped (as Cancel stops it) after
    ``k`` frames."""
    for i, im in enumerate(v3.turntable_frames(_scene(), v3.View(), 36, 64, 48)):
        if i == k:
            raise KeyboardInterrupt('canceled')
        yield im


@pytest.mark.parametrize('kind', ['gif', 'mp4', 'mp4-fails'])
def test_a_failed_or_stopped_save_leaves_the_old_file_alone(tmp_path, kind):
    old = tmp_path / f'keep.{kind[:3]}'
    old.write_bytes(b'the file that was there')
    exe = _fake_ffmpeg(tmp_path, code=3 if kind == 'mp4-fails' else 0)
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        if kind == 'gif':
            v3.save_gif(_stopped_at(5), str(old), 36)
        elif kind == 'mp4':
            v3.save_mp4(_stopped_at(5), str(old), exe, 36)
        else:
            v3.save_mp4(v3.turntable_frames(_scene(), v3.View(), 36, 64, 48),
                        str(old), exe, 36)
    assert old.read_bytes() == b'the file that was there'
    assert not os.path.exists(str(old) + '.part')


def test_a_finished_save_replaces_the_old_file(tmp_path):
    old = tmp_path / 'keep.gif'
    old.write_bytes(b'old')
    v3.save_gif(v3.turntable_frames(_scene(), v3.View(), 36, 64, 48), str(old), 36)
    with Image.open(old) as im:
        assert im.n_frames == 36


def test_colors_need_no_tk():
    """The core module colours and draws with tkinter missing (the phone,
    the CLI, a server)."""
    import subprocess
    code = textwrap.dedent("""
        import sys
        sys.modules['tkinter'] = None
        import numpy as np
        from openflo import view3d as v3
        p = np.random.default_rng(1).uniform(-1, 1, (200, 3))
        v3.density_colors(p, 'viridis')
        v3.marker_colors(p[:, 0])
        v3.render(v3.Scene(p, v3.density_colors(p, 'turbo')), v3.View(), 60, 40)
        assert 'tkinter' not in sys.modules or sys.modules['tkinter'] is None
        assert 'openflo.theme' not in sys.modules
        print('ok')
    """)
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    r = subprocess.run([sys.executable, '-c', code], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == 'ok', r.stderr


def test_no_encoder_found_means_no_mp4(monkeypatch):
    monkeypatch.setattr(v3.shutil, 'which', lambda _n: None)
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', None)
    assert v3.mp4_encoder() is None
    monkeypatch.setattr(v3.shutil, 'which', lambda _n: '/usr/local/bin/ffmpeg')
    assert v3.mp4_encoder() == '/usr/local/bin/ffmpeg'


@pytest.mark.skipif(os.name == 'nt', reason='checks the process table')
def test_a_stopped_mp4_save_ends_ffmpeg(tmp_path, monkeypatch):
    """Cancel part-way: ffmpeg is stopped and waited for, not left reading
    its input (or writing the .part file once its pipe closes), and both
    its pipes are closed."""
    procs = []
    real_popen = subprocess.Popen

    def popen(*a, **k):
        procs.append(real_popen(*a, **k))
        return procs[-1]
    monkeypatch.setattr(v3.subprocess, 'Popen', popen)
    script = tmp_path / 'fake_ffmpeg.py'
    pidfile = tmp_path / 'pid'
    script.write_text(textwrap.dedent(f"""
        import os, sys
        open({str(pidfile)!r}, 'w').write(str(os.getpid()))
        sys.stdin.buffer.read()
        open(sys.argv[-1], 'w').write('late')
    """))
    exe = tmp_path / 'fake_ffmpeg'
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / 'x.mp4'

    def stopped_once_running():
        import time
        frames = v3.turntable_frames(_scene(), v3.View(), 36, 64, 48)
        yield next(frames)
        for _ in range(200):                     # ffmpeg up and reading
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.05)
        raise KeyboardInterrupt('canceled')
    with pytest.raises(KeyboardInterrupt):
        v3.save_mp4(stopped_once_running(), str(out), str(exe), 36)
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):      # gone and reaped
        os.kill(pid, 0)
    assert not out.exists() and not os.path.exists(str(out) + '.part')
    assert procs[0].stdin.closed and procs[0].stderr.closed
