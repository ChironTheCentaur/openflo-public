"""OpenFlo's mark: two cell populations on a flow-cytometry dot plot (a cool
blue cloud, a green middle cluster and a hot orange one), the hot one inside
a white gate with a soft glow behind it, on a deep indigo tile.

One drawing for every place the mark appears: the desktop window icon
(gui.py), the Windows installer's .ico (scripts/build_windows_installer.py)
and the phone apps' launcher icons, themed icon and splash screens
(mobile/scripts/make_icons.py), so the app looks the same everywhere.
The artwork is fixed data (DOTS, GLOW, GATE in a 108-unit square, the
Android adaptive icon's grid), so it never changes between runs. Needs
Pillow and numpy, imported when called."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from PIL import Image as _Image

UNITS = 108.0                   # the artwork's grid (Android's adaptive layer)
CENTRE = UNITS / 2
# Background: a radial gradient lit up and to the right, behind the gate.
BG_INNER = (0x2c, 0x3a, 0x8a)
BG_OUTER = (0x0c, 0x0f, 0x26)
BG_FOCUS = (0.62, 0.40)         # gradient centre, fraction of the side
BG_RADIUS = 0.80                # gradient radius, fraction of the side
# The glow behind the gated cluster: centre, radius, colour, opacity, blur.
GLOW = (67.0, 43.0, 17.0, (0xff, 0x6a, 0x3a), 0.30, 5.5)
# The gate: x, y, width, height, corner radius, stroke width (stroke centred
# on the outline, as in SVG).
GATE = (54.0, 30.0, 26.0, 26.0, 6.5, 3.1)
# The events: (x, y, radius, 0xRRGGBB), drawn in this order.
DOTS = (
    (50.52, 60.09, 1.61, 0x2e68dd), (51.28, 61.35, 1.61, 0x2e68dd),
    (31.43, 58.84, 1.62, 0x2e6bdc), (49.86, 63.34, 1.65, 0x2e70da),
    (25.42, 67.48, 1.66, 0x2e74d9), (47.66, 63.40, 1.68, 0x2e78d7),
    (25.70, 72.66, 1.68, 0x2e78d7), (45.07, 61.14, 1.69, 0x2e7ad7),
    (37.68, 77.44, 1.71, 0x2e7cd6), (27.15, 71.95, 1.71, 0x2e7ed5),
    (36.79, 76.29, 1.74, 0x2e83d3), (37.36, 61.69, 1.74, 0x2e83d3),
    (45.31, 65.85, 1.74, 0x2e84d3), (35.94, 74.59, 1.78, 0x2e8cd0),
    (40.14, 63.72, 1.78, 0x2e8dd0), (32.23, 72.60, 1.79, 0x2e8fcf),
    (42.06, 66.63, 1.80, 0x2e91cf), (40.17, 71.88, 1.80, 0x2e91cf),
    (35.61, 72.92, 1.81, 0x2e94ce), (35.72, 65.69, 1.82, 0x2e94cd),
    (36.62, 65.64, 1.82, 0x2e96cd), (33.29, 68.63, 1.83, 0x2e96cd),
    (38.97, 65.97, 1.83, 0x2e97cc), (39.41, 69.67, 1.85, 0x2e9bcb),
    (35.38, 67.64, 1.85, 0x2e9ccb), (38.40, 68.01, 1.87, 0x2ea0c9),
    (48.36, 45.66, 1.94, 0x2eafc4), (53.33, 67.36, 1.95, 0x2eb1c4),
    (63.36, 56.48, 1.95, 0x2eb1c4), (45.11, 68.05, 1.96, 0x30b2c0),
    (39.25, 65.02, 1.96, 0x31b3be), (39.81, 54.23, 1.98, 0x36b5b5),
    (49.38, 47.51, 1.99, 0x37b5b3), (38.64, 60.74, 1.99, 0x37b5b2),
    (60.36, 51.37, 2.00, 0x39b6b0), (60.53, 58.00, 2.00, 0x39b6b0),
    (57.11, 62.62, 2.00, 0x39b6af), (57.77, 50.01, 2.02, 0x3eb8a6),
    (56.55, 61.56, 2.03, 0x41b9a2), (53.85, 49.19, 2.03, 0x43ba9f),
    (55.25, 49.70, 2.04, 0x43ba9e), (49.63, 50.21, 2.05, 0x48bc96),
    (46.13, 52.10, 2.06, 0x4abc93), (57.03, 54.31, 2.08, 0x4dbe8d),
    (51.39, 51.11, 2.08, 0x4fbe8a), (43.74, 56.95, 2.09, 0x51bf87),
    (44.36, 55.83, 2.09, 0x52bf85), (49.75, 52.09, 2.10, 0x53c082),
    (44.77, 60.36, 2.10, 0x54c082), (54.93, 56.83, 2.12, 0x58c27a),
    (49.99, 61.18, 2.12, 0x58c27a), (46.76, 57.20, 2.15, 0x61c56b),
    (67.86, 56.75, 2.10, 0x54c081), (73.32, 55.40, 2.12, 0x59c279),
    (55.65, 50.62, 2.12, 0x59c278), (61.03, 53.31, 2.17, 0x64c665),
    (58.36, 34.01, 2.19, 0x6ec75d), (56.04, 47.13, 2.21, 0x79c75a),
    (64.46, 32.49, 2.22, 0x81c757), (58.54, 37.23, 2.27, 0xa5c64c),
    (65.19, 51.76, 2.27, 0xa7c64b), (64.89, 34.27, 2.28, 0xadc549),
    (74.78, 48.30, 2.30, 0xb7c546), (73.53, 49.27, 2.30, 0xbbc544),
    (75.37, 45.91, 2.32, 0xc6c541), (59.42, 46.12, 2.32, 0xc9c540),
    (66.75, 35.78, 2.34, 0xd4c43d), (74.16, 46.27, 2.35, 0xdbc43a),
    (59.74, 44.05, 2.36, 0xe2c438), (73.29, 46.63, 2.37, 0xe7c436),
    (59.83, 42.30, 2.37, 0xe9c436), (73.10, 46.62, 2.37, 0xebc435),
    (70.82, 47.55, 2.40, 0xeeb633), (65.17, 47.08, 2.44, 0xeca232),
    (63.70, 39.97, 2.45, 0xeb9732), (67.46, 39.21, 2.46, 0xeb9332),
    (64.48, 39.82, 2.46, 0xeb9132), (70.66, 44.09, 2.48, 0xea8832),
    (64.57, 40.27, 2.48, 0xea8832), (70.43, 42.22, 2.48, 0xea8532),
    (66.33, 46.08, 2.48, 0xea8532), (65.61, 40.26, 2.49, 0xe97f31),
    (63.90, 43.42, 2.50, 0xe97d31), (68.28, 40.53, 2.50, 0xe97c31),
    (69.91, 44.21, 2.50, 0xe97c31), (69.83, 42.21, 2.50, 0xe97a31),
    (69.19, 42.11, 2.52, 0xe86f31), (67.03, 43.85, 2.57, 0xe55230),
)

# How big the artwork is drawn, about the centre, in each use. The Android
# adaptive layers keep it inside the safe zone, a circle 66 units across
# that no launcher's mask clips (its furthest point, the outer edge of the
# lowest-left dot, is 35.6 units out at full size); a filled tile (iOS,
# desktop, splash, the legacy Android icons) shows it larger.
ADAPTIVE_SCALE = 0.92
TILE_SCALE = 1.35

Shape = Literal['tile', 'square', 'circle', 'foreground', 'background', 'monochrome']


def _rgb(c: int) -> tuple[int, int, int]:
    return (c >> 16) & 255, (c >> 8) & 255, c & 255


def _background(D: int) -> _Image.Image:
    import numpy as np
    from PIL import Image
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float64) + 0.5
    d = np.hypot(xx - BG_FOCUS[0] * D, yy - BG_FOCUS[1] * D) / (BG_RADIUS * D)
    t = np.clip(d, 0.0, 1.0)[..., None]
    rgb = np.array(BG_INNER, float) * (1 - t) + np.array(BG_OUTER, float) * t
    a = np.full((D, D, 1), 255.0)
    return Image.fromarray(np.round(np.concatenate([rgb, a], axis=2)).astype(np.uint8), 'RGBA')


def _artwork(D: int, scale: float) -> _Image.Image:
    """Glow, dots and gate on transparent, `D` pixels for the 108 units,
    drawn `scale` times about the centre."""
    from PIL import Image, ImageDraw, ImageFilter
    k = D / UNITS

    def px(u: float) -> float:
        return (CENTRE + (u - CENTRE) * scale) * k

    def ln(u: float) -> float:
        return u * scale * k

    out = Image.new('RGBA', (D, D), (0, 0, 0, 0))
    gx, gy, gr, gc, gop, gblur = GLOW
    mask = Image.new('L', (D, D), 0)
    ImageDraw.Draw(mask).ellipse([px(gx) - ln(gr), px(gy) - ln(gr),
                                  px(gx) + ln(gr), px(gy) + ln(gr)], fill=round(255 * gop))
    mask = mask.filter(ImageFilter.GaussianBlur(ln(gblur)))
    glow = Image.new('RGBA', (D, D), (*gc, 255))
    glow.putalpha(mask)
    out.alpha_composite(glow)
    draw = ImageDraw.Draw(out)
    for x, y, r, c in DOTS:
        draw.ellipse([px(x) - ln(r), px(y) - ln(r), px(x) + ln(r), px(y) + ln(r)],
                     fill=(*_rgb(c), 255))
    x, y, w, h, rx, sw = GATE
    half = sw / 2               # Pillow strokes inside the box; SVG centres it
    draw.rounded_rectangle([px(x - half), px(y - half), px(x + w + half), px(y + h + half)],
                           radius=ln(rx + half), outline=(255, 255, 255, 255),
                           width=max(1, round(ln(sw))))
    return out


def _square(D: int) -> _Image.Image:
    canvas = _background(D)
    canvas.alpha_composite(_artwork(D, TILE_SCALE))
    return canvas


def icon_image(size: int, shape: Shape = 'tile') -> _Image.Image:
    """The mark as an RGBA image `size` pixels square.

    ``'square'``: the tile to the edges, artwork at TILE_SCALE (iOS rounds
    it itself). ``'tile'``: the same as a rounded tile with clear corners
    (desktop window and installer, splash, legacy Android square icon).
    ``'circle'``: the square cut to a circle (legacy Android round icon).
    Android's adaptive icon in three layers, artwork at ADAPTIVE_SCALE:
    ``'background'`` (the gradient), ``'foreground'`` (glow, dots and gate
    on transparent) and ``'monochrome'`` (the foreground in white, its
    alpha kept, for the system to tint as a themed icon)."""
    from PIL import Image, ImageChops, ImageDraw
    ss = 4                        # supersample, then LANCZOS down
    D = size * ss
    if shape == 'tile':
        pad = round(D * 0.045)
        inner = _square(D - 2 * pad)
        canvas = Image.new('RGBA', (D, D), (0, 0, 0, 0))
        canvas.paste(inner, (pad, pad))
        mask = Image.new('L', (D, D), 0)
        ImageDraw.Draw(mask).rounded_rectangle([pad, pad, D - pad - 1, D - pad - 1],
                                               radius=D * 0.2, fill=255)
        canvas.putalpha(ImageChops.multiply(canvas.getchannel('A'), mask))
    elif shape in ('square', 'circle'):
        canvas = _square(D)
        if shape == 'circle':
            mask = Image.new('L', (D, D), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, D - 1, D - 1], fill=255)
            canvas.putalpha(mask)
    elif shape == 'background':
        canvas = _background(D)
    elif shape == 'foreground':
        canvas = _artwork(D, ADAPTIVE_SCALE)
    elif shape == 'monochrome':
        canvas = Image.new('RGBA', (D, D), (255, 255, 255, 255))
        canvas.putalpha(_artwork(D, ADAPTIVE_SCALE).getchannel('A'))
    else:
        raise ValueError(f'unknown icon shape {shape!r}')
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def icon_svg() -> str:
    """The ``'square'`` mark as SVG (the website's favicon and touch icons
    are made from it), from the same data as icon_image."""
    s = TILE_SCALE

    def p(u: float) -> str:
        return f'{CENTRE + (u - CENTRE) * s:.2f}'

    def ln(u: float) -> str:
        return f'{u * s:.2f}'

    def hexc(c: tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*c)

    gx, gy, gr, gc, gop, gblur = GLOW
    x, y, w, h, rx, sw = GATE
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {UNITS:g} {UNITS:g}">',
        '<defs>'
        f'<radialGradient id="bg" cx="{BG_FOCUS[0]}" cy="{BG_FOCUS[1]}" r="{BG_RADIUS}">'
        f'<stop offset="0" stop-color="{hexc(BG_INNER)}"/>'
        f'<stop offset="1" stop-color="{hexc(BG_OUTER)}"/></radialGradient>'
        '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feGaussianBlur stdDeviation="{ln(gblur)}"/></filter></defs>',
        f'<rect width="{UNITS:g}" height="{UNITS:g}" fill="url(#bg)"/>',
        f'<circle cx="{p(gx)}" cy="{p(gy)}" r="{ln(gr)}" fill="{hexc(gc)}" '
        f'opacity="{gop}" filter="url(#glow)"/>',
    ]
    parts += [f'<circle cx="{p(cx)}" cy="{p(cy)}" r="{ln(r)}" fill="{hexc(_rgb(c))}"/>'
              for cx, cy, r, c in DOTS]
    parts.append(f'<rect x="{p(x)}" y="{p(y)}" width="{ln(w)}" height="{ln(h)}" rx="{ln(rx)}" '
                 f'fill="none" stroke="#fff" stroke-width="{ln(sw)}"/>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'
