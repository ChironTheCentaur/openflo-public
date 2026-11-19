"""OpenFlo's mark (app_icon.py): one drawing for the desktop window, the
Windows installer and the phone apps, the same on every run, and the one
Sky approved (tests/reference/app_icon_approved_96.png is the approved
1024-pixel iOS icon scaled down)."""
import math
import os
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip('PIL')
np = pytest.importorskip('numpy')

from PIL import Image  # noqa: E402

from openflo.app_icon import (  # noqa: E402
    ADAPTIVE_SCALE,
    BG_OUTER,
    DOTS,
    GATE,
    TILE_SCALE,
    UNITS,
    icon_image,
    icon_svg,
)

SHAPES = ('tile', 'square', 'circle', 'foreground', 'background', 'monochrome')
APPROVED = os.path.join(os.path.dirname(__file__), 'reference', 'app_icon_approved_96.png')


def px(img):
    return np.asarray(img, dtype=int)


def test_the_mark_is_the_same_every_time():
    for shape in SHAPES:
        assert icon_image(64, shape).tobytes() == icon_image(64, shape).tobytes()


@pytest.mark.parametrize('size', [16, 48, 192])
def test_each_shape_is_its_size_and_rgba(size):
    for shape in SHAPES:
        img = icon_image(size, shape)
        assert img.size == (size, size) and img.mode == 'RGBA'


def test_an_unknown_shape_is_refused():
    with pytest.raises(ValueError):
        icon_image(32, 'hexagon')


def test_the_ios_icon_matches_the_approved_design():
    ref = px(Image.open(APPROVED).convert('RGB'))
    got = px(icon_image(192, 'square').convert('RGB').resize((96, 96), Image.Resampling.LANCZOS))
    diff = np.abs(got - ref)
    # Measured 1.16 mean; moving the gate 2 units or dropping the glow
    # takes it past 2.
    assert diff.mean() < 1.6, diff.mean()


def test_which_shapes_fill_their_corners():
    s = 128
    for shape in ('tile', 'circle', 'foreground', 'monochrome'):
        assert icon_image(s, shape).getpixel((0, 0))[3] == 0, shape
    for shape in ('square', 'background'):
        img = icon_image(s, shape)
        for xy in ((0, 0), (s - 1, s - 1), (0, s - 1)):
            r, g, b, a = img.getpixel(xy)
            # The gradient's dark outer colour, opaque (the App Store
            # refuses transparency).
            assert a == 255 and max(abs(r - BG_OUTER[0]), abs(g - BG_OUTER[1]),
                                    abs(b - BG_OUTER[2])) <= 12, (shape, xy)


def test_the_tile_is_the_square_inset_with_rounded_corners():
    s = 200
    tile = px(icon_image(s, 'tile'))
    sq = px(icon_image(s, 'square'))
    # Its middle is the square's middle, drawn a little smaller.
    assert tile[s // 2, s // 2, 3] == 255
    assert tile[s // 2, 3, 3] == 0 and tile[3, s // 2, 3] == 0
    # The hot cluster is in the same place relative to the tile.

    def hot_centre(img):
        ys, xs = np.nonzero((img[..., 0] > img[..., 2] + 100) & (img[..., 3] == 255))
        return xs.mean(), ys.mean()

    centre = s / 2
    for a, b in zip(hot_centre(tile), hot_centre(sq), strict=True):
        assert abs((a - centre) - (b - centre) * 0.91) <= 2


def test_the_hot_cluster_sits_in_the_white_gate_and_the_cool_cloud_below_left():
    s = 432
    fg = px(icon_image(s, 'foreground'))
    k = s / UNITS

    def at(u, v):
        c = UNITS / 2
        return int((c + (u - c) * ADAPTIVE_SCALE) * k), int((c + (v - c) * ADAPTIVE_SCALE) * k)

    x, y, _w, h, _rx, _sw = GATE
    # Middle of the gate's left edge: white and opaque.
    gx, gy = at(x, y + h / 2)
    assert (fg[gy, gx] >= 245).all()
    # The gated core is hot (red over blue), the far cloud cool.
    cx, cy = at(67.03, 43.85)
    assert fg[cy, cx, 0] > fg[cy, cx, 2] + 120
    bx, by = at(37.68, 77.44)
    assert fg[by, bx, 2] > fg[by, bx, 0] + 120


def test_the_adaptive_artwork_keeps_within_the_safe_zone():
    s = 432
    a = px(icon_image(s, 'foreground'))[..., 3]
    ys, xs = np.nonzero(a > 100)          # dots and gate, not the faint glow
    r = np.hypot(xs + 0.5 - s / 2, ys + 0.5 - s / 2)
    assert r.max() <= s * 33 / UNITS + 1
    # And it fills it: the gate's corner reaches near the edge.
    assert r.max() >= s * 31 / UNITS


def test_the_monochrome_layer_is_the_foreground_in_white():
    s = 108
    mono = px(icon_image(s, 'monochrome'))
    fg = px(icon_image(s, 'foreground'))
    assert (mono[..., 3] == fg[..., 3]).all() and (fg[..., 3] > 0).any()
    # White wherever it shows (resizing zeroes the colour of clear pixels).
    assert (mono[mono[..., 3] > 0][:, :3] >= 250).all()


def test_the_svg_carries_the_same_artwork():
    root = ET.fromstring(icon_svg())
    ns = '{http://www.w3.org/2000/svg}'
    assert root.get('viewBox') == '0 0 108 108'
    circles = root.findall(f'{ns}circle')
    assert len(circles) == len(DOTS) + 1               # the dots and the glow
    c = UNITS / 2
    x0, y0, r0, _ = DOTS[0]
    first = circles[1]
    assert math.isclose(float(first.get('cx')), c + (x0 - c) * TILE_SCALE, abs_tol=0.01)
    assert math.isclose(float(first.get('cy')), c + (y0 - c) * TILE_SCALE, abs_tol=0.01)
    assert math.isclose(float(first.get('r')), r0 * TILE_SCALE, abs_tol=0.01)
    (gate,) = [e for e in root.findall(f'{ns}rect') if e.get('stroke')]
    x, y, w, _h, _rx, sw = GATE
    assert math.isclose(float(gate.get('y')), c + (y - c) * TILE_SCALE, abs_tol=0.01)
    assert math.isclose(float(gate.get('x')), c + (x - c) * TILE_SCALE, abs_tol=0.01)
    assert math.isclose(float(gate.get('width')), w * TILE_SCALE, abs_tol=0.01)
    assert math.isclose(float(gate.get('stroke-width')), sw * TILE_SCALE, abs_tol=0.01)
