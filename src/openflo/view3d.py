"""3D scatter view: the maths and the pictures, with no Tk.

The 3D window (``ui_view3d``) draws three channels, or the three components
of a 3D embedding, as a cloud you can turn. Everything it shows is computed
here so it can be tested without a display:

* each axis is put on screen with the SAME transform the 2D plot uses for
  that channel (matplotlib's own linear / log / symlog / function scale), so
  the cloud seen from the front (azimuth 0, elevation 0) is the 2D plot of X
  against Y;
* at most ``MAX_EVENTS`` events are drawn: a seeded random subset, so the
  same seed always picks the same events;
* the picture is drawn with numpy into an RGB array (no OpenGL: none is in
  the dependencies), nearest point on top, then handed to Pillow;
* a turntable is a list of such pictures, saved as an endlessly looping GIF
  or, when an ffmpeg program is found, an H.264 MP4.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .cloud3d import DENSITY_BINS, density_values
from .cloud3d import normalize as normalize  # re-exported: ui_view3d, tests
from .cloud3d import seeded_subset as seeded_subset  # re-exported

# The most events drawn at once. More than this turns slowly on an ordinary
# laptop and adds nothing a viewer can see in a cloud of points.
MAX_EVENTS = 200_000
DEFAULT_SEED = 42
TURNTABLE_FRAMES = (36, 72)
DEFAULT_FRAMES = 36
POINT_SIZES = (1, 2, 3, 4)
DEFAULT_POINT_SIZE = 2
# A full turn of a turntable takes this long whatever the frame count.
TURN_SECONDS = 6.0
# The view the window opens on: turned enough that all three axes show.
DEFAULT_AZIMUTH = -35.0
DEFAULT_ELEVATION = 25.0
MIN_ZOOM, MAX_ZOOM = 0.3, 8.0
# One wheel notch zooms by ZOOM_STEP. Tk reports a notch as a <MouseWheel>
# delta of 120 (Windows, and every platform from Tk 8.7); precision
# touchpads and free-spinning wheels send many smaller deltas.
ZOOM_STEP = 1.1
WHEEL_NOTCH = 120
MAX_WHEEL_NOTCHES = 3
# matplotlib's default autoscale margin, which the 2D plot's axes carry.
AXIS_MARGIN = 0.05
OUTSIDE_GREY = (150, 150, 150)
COLOR_MODES = ('density', 'marker', 'cluster', 'gate')


@dataclass(frozen=True)
class View:
    """Where the camera looks from. Azimuth turns about the vertical (Y)
    axis, elevation tilts toward the top; both in degrees. At (0, 0) X runs
    left to right, Y bottom to top and Z points at the viewer."""
    azimuth: float = DEFAULT_AZIMUTH
    elevation: float = DEFAULT_ELEVATION
    zoom: float = 1.0

    def turned(self, d_azimuth: float = 0.0, d_elevation: float = 0.0) -> View:
        """This view turned by a drag: azimuth wraps, elevation stops at the
        poles so the picture never flips upside down."""
        az = (self.azimuth + d_azimuth + 180.0) % 360.0 - 180.0
        el = min(90.0, max(-90.0, self.elevation + d_elevation))
        return replace(self, azimuth=az, elevation=el)

    def zoomed(self, factor: float) -> View:
        """This view zoomed by ``factor`` (>1 closer), kept in range."""
        z = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * float(factor)))
        return replace(self, zoom=z)


FRONT_VIEW = View(azimuth=0.0, elevation=0.0, zoom=1.0)


def wheel_units(windowing_system: str, tcl_version: str) -> float:
    """``<MouseWheel>`` delta units in one wheel notch: ``WHEEL_NOTCH``,
    except Tk 8.6 on macOS ('aqua'), which reports scroll lines."""
    major, _, minor = tcl_version.partition('.')
    old_tk = (int(major), int(minor or 0)) < (8, 7)
    return 1 if windowing_system == 'aqua' and old_tk else WHEEL_NOTCH


def wheel_zoom(delta: float, units_per_notch: float = WHEEL_NOTCH) -> float:
    """The zoom factor of one <MouseWheel> event: ``ZOOM_STEP`` for each
    notch's worth of ``delta`` (above 1 rolling away from the user), so a
    touchpad's many small deltas add up to the same zoom as the notches
    they stand for. One event counts as at most ``MAX_WHEEL_NOTCHES``."""
    notches = float(delta) / float(units_per_notch)
    notches = max(-MAX_WHEEL_NOTCHES, min(MAX_WHEEL_NOTCHES, notches))
    return float(ZOOM_STEP ** notches)


# ── axes ─────────────────────────────────────────────────────────────────

def screen_transform(scale: str, funcs: tuple[Callable, Callable] | None = None,
                     linthresh: float | None = None) -> Callable[[Any], np.ndarray]:
    """Stored value -> screen position along one axis, through matplotlib's
    own scale classes, exactly as the 2D plot sets the axis: ``funcs`` (a
    channel stored in a non-linear transform, shown as a view of its
    intensity) wins; otherwise ``scale`` is 'log', 'symlog' (with
    ``linthresh``) or linear. Values a scale cannot place (log of zero or
    less) come back NaN, where the 2D plot leaves them off the axes."""
    from matplotlib import scale as mscale
    if funcs is not None:
        tr = mscale.FuncScale(None, funcs).get_transform()
    elif scale == 'log':
        tr = mscale.LogScale(None, nonpositive='mask').get_transform()
    elif scale == 'symlog':
        tr = mscale.SymmetricalLogScale(
            None, linthresh=float(linthresh or 1.0)).get_transform()
    else:
        tr = mscale.LinearScale(None).get_transform()

    def forward(values: Any) -> np.ndarray:
        a = np.asarray(values, dtype=float).ravel()
        with np.errstate(all='ignore'):
            out = np.ma.filled(np.ma.asarray(tr.transform(a), dtype=float),
                               np.nan)
        out = np.array(out, dtype=float).ravel()
        out[~np.isfinite(out)] = np.nan
        return out
    return forward


def axis_bounds(screen: np.ndarray, explicit: tuple[float, float] | None = None,
                ) -> tuple[float, float]:
    """``(lo, hi)`` in screen units for one axis: ``explicit`` (an axis range
    already put on screen) when given, else the data's extent plus
    matplotlib's 5% margin each side, as the 2D plot autoscales. An end of
    ``explicit`` that is not finite (a range starting at 0 on a log axis)
    is autoscaled and the other end kept, as matplotlib ignores only that
    end of the 2D plot's limits. A flat or empty axis gets a unit span so
    nothing divides by zero."""
    ends = (math.nan, math.nan) if explicit is None else (
        float(explicit[0]), float(explicit[1]))
    lo, hi = ends
    if not (math.isfinite(lo) and math.isfinite(hi)):
        v = np.asarray(screen, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0 and not (math.isfinite(lo) or math.isfinite(hi)):
            return 0.0, 1.0
        if v.size:
            d_lo, d_hi = float(v.min()), float(v.max())
            pad = (d_hi - d_lo) * AXIS_MARGIN
            lo = lo if math.isfinite(lo) else d_lo - pad
            hi = hi if math.isfinite(hi) else d_hi + pad
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi == lo:
        mid = lo if math.isfinite(lo) else (hi if math.isfinite(hi) else 0.0)
        return mid - 0.5, mid + 0.5
    return (lo, hi) if lo < hi else (hi, lo)


def subset_note(shown: int, total: int, seed: int) -> str:
    """The line under the plot saying how many events it draws."""
    if shown >= total:
        return f"Showing all {total:,} events"
    return (f"Showing {shown:,} of {total:,} events "
            f"(random subset, seed {seed})")


# ── projection ───────────────────────────────────────────────────────────

def rotation(view: View) -> np.ndarray:
    """3x3 matrix turning data points into camera space: azimuth about the
    vertical axis first, then elevation about the horizontal one."""
    a = math.radians(view.azimuth)
    e = math.radians(view.elevation)
    ry = np.array([[math.cos(a), 0.0, math.sin(a)],
                   [0.0, 1.0, 0.0],
                   [-math.sin(a), 0.0, math.cos(a)]])
    rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(e), -math.sin(e)],
                   [0.0, math.sin(e), math.cos(e)]])
    return rx @ ry


def project(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(sx, sy, depth)`` of points in -1..1 cube space, seen orthographically
    from ``view``: sx right, sy up, depth toward the viewer. At the front view
    ``sx, sy`` are the points' own X and Y."""
    cam = np.asarray(points, dtype=float) @ rotation(view).T
    return cam[:, 0], cam[:, 1], cam[:, 2]


def _pixel_scale(width: int, height: int, zoom: float) -> float:
    # The cube's corners lie sqrt(3) from its centre: at zoom 1 the whole cube
    # stays inside the picture whichever way it is turned.
    return zoom * (min(width, height) / 2.0 - 4.0) / math.sqrt(3.0)


def to_pixels(sx: np.ndarray, sy: np.ndarray, width: int, height: int,
              zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera x/y -> pixel column/row (row 0 at the top)."""
    s = _pixel_scale(width, height, zoom)
    return width / 2.0 + s * np.asarray(sx), height / 2.0 - s * np.asarray(sy)


# ── colours ──────────────────────────────────────────────────────────────

def _cmap_rgb(cmap_name: str, t: np.ndarray) -> np.ndarray:
    # matplotlib's registry directly: theme.plt_get_cmap would pull in Tk.
    import matplotlib
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0))
    return (np.asarray(rgba)[:, :3] * 255.0 + 0.5).astype(np.uint8)


def density_colors(points: np.ndarray, cmap_name: str,
                   bins: int = DENSITY_BINS) -> np.ndarray:
    """Colour each point by how crowded its neighbourhood is in 3D: a
    ``bins``³ histogram of the cube, smoothed by one bin, read back per
    point, then spread over the colour map with the 2D plot's gamma-0.4
    norm (``plotmath.density_norm``) so a dense core does not wash out the
    rest."""
    from .plotmath import density_norm
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z = density_values(points, bins)
    norm = density_norm(z)
    return _cmap_rgb(cmap_name, np.asarray(norm(z), dtype=float))


def marker_colors(values: np.ndarray, cmap_name: str = 'viridis') -> np.ndarray:
    """Colour by a channel's stored value, as the 2D plot's colour-by-channel
    does, spread across its 0.5-99.5 percentile range so a few outliers do
    not squeeze everything else into one shade. Missing values are grey."""
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    out = np.empty((len(v), 3), dtype=np.uint8)
    out[:] = OUTSIDE_GREY
    if not fin.any():
        return out
    lo, hi = np.percentile(v[fin], [0.5, 99.5])
    span = (hi - lo) if hi > lo else 1.0
    out[fin] = _cmap_rgb(cmap_name, (v[fin] - lo) / span)
    return out


def hex_rgb(colour: str) -> tuple[int, int, int]:
    """'#rrggbb' (or any matplotlib colour) -> an (r, g, b) of 0..255."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(colour)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def category_colors(values: np.ndarray,
                    palette: Sequence[tuple[Any, str]]) -> np.ndarray:
    """One colour per population: ``palette`` is ``[(value, '#colour')]``;
    a value it does not list is grey."""
    v = np.asarray(values)
    out = np.empty((len(v), 3), dtype=np.uint8)
    out[:] = OUTSIDE_GREY
    for value, colour in palette:
        out[v == value] = hex_rgb(colour)
    return out


def gate_colors(inside: np.ndarray, colour: str) -> np.ndarray:
    """Events inside a gate in the gate's colour, the rest grey."""
    m = np.asarray(inside, dtype=bool)
    out = np.empty((len(m), 3), dtype=np.uint8)
    out[:] = OUTSIDE_GREY
    out[m] = hex_rgb(colour)
    return out


# ── drawing ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scene:
    """What one picture shows: the points (-1..1 cube space), their colours,
    the axis names and the look."""
    points: np.ndarray
    colors: np.ndarray
    labels: tuple[str, str, str] = ('X', 'Y', 'Z')
    point_size: int = DEFAULT_POINT_SIZE
    background: tuple[int, int, int] = (255, 255, 255)
    ink: tuple[int, int, int] = (40, 40, 40)


def rasterize(px: np.ndarray, py: np.ndarray, depth: np.ndarray,
              colors: np.ndarray, width: int, height: int, size: int,
              background: tuple[int, int, int]) -> np.ndarray:
    """An ``(height, width, 3)`` uint8 picture of the points, each a
    ``size``-pixel square, the one nearest the viewer on top wherever they
    overlap (a depth buffer of each pixel's nearest point)."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = background
    n = len(px)
    if n == 0:
        return img
    order = np.argsort(np.asarray(depth, dtype=float), kind='stable')
    rank = np.arange(n, dtype=np.int64)        # larger rank = nearer
    xi = np.floor(np.asarray(px, dtype=float)[order]).astype(np.int64)
    yi = np.floor(np.asarray(py, dtype=float)[order]).astype(np.int64)
    nearest = np.full(width * height, -1, dtype=np.int64)
    size = max(1, int(size))
    offs = range(-(size // 2), size - size // 2)
    for dy in offs:
        for dx in offs:
            x = xi + dx
            y = yi + dy
            ok = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            np.maximum.at(nearest, y[ok] * width + x[ok], rank[ok])
    hit = nearest >= 0
    img.reshape(-1, 3)[hit] = np.asarray(colors)[order][nearest[hit]]
    return img


_CUBE_EDGES = [((-1, -1, -1), (1, -1, -1)), ((-1, -1, -1), (-1, 1, -1)),
               ((-1, -1, -1), (-1, -1, 1)), ((1, 1, 1), (-1, 1, 1)),
               ((1, 1, 1), (1, -1, 1)), ((1, 1, 1), (1, 1, -1)),
               ((1, -1, -1), (1, 1, -1)), ((1, -1, -1), (1, -1, 1)),
               ((-1, 1, -1), (1, 1, -1)), ((-1, 1, -1), (-1, 1, 1)),
               ((-1, -1, 1), (1, -1, 1)), ((-1, -1, 1), (-1, 1, 1))]
# The three edges from the low corner are the axes; the label sits past
# each one's far end.
_AXIS_ENDS = ((1.12, -1.0, -1.0), (-1.0, 1.12, -1.0), (-1.0, -1.0, 1.12))


def _to_screen(pts: np.ndarray, view: View, width: int, height: int):
    sx, sy, _ = project(np.asarray(pts, dtype=float), view)
    return to_pixels(sx, sy, width, height, view.zoom)


def _blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))


def _draw_frame(image: Any, scene: Scene, view: View) -> None:
    """The cube's edges (the three axes darker) and the axis names, drawn
    over the points with Pillow."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    w, h = image.size
    faint = _blend(scene.background, scene.ink, 0.25)
    for i, (a, b) in enumerate(_CUBE_EDGES):
        px, py = _to_screen(np.array([a, b]), view, w, h)
        draw.line([(px[0], py[0]), (px[1], py[1])],
                  fill=scene.ink if i < 3 else faint, width=1)
    try:
        font: Any = ImageFont.load_default(size=max(11, min(w, h) // 45))
    except TypeError:  # Pillow without sized default fonts
        font = ImageFont.load_default()
    px, py = _to_screen(np.array(_AXIS_ENDS), view, w, h)
    for label, x, y in zip(scene.labels, px, py, strict=True):
        draw.text((float(x), float(y)), str(label), fill=scene.ink,
                  font=font, anchor='mm')


def render(scene: Scene, view: View, width: int, height: int):
    """One picture of ``scene`` from ``view``, as a Pillow RGB image."""
    from PIL import Image
    pts = np.asarray(scene.points, dtype=float)
    if len(pts):
        sx, sy, depth = project(pts, view)
        px, py = to_pixels(sx, sy, width, height, view.zoom)
    else:
        px = py = depth = np.zeros(0)
    arr = rasterize(px, py, depth, scene.colors, width, height,
                    scene.point_size, scene.background)
    image = Image.fromarray(arr, 'RGB')
    _draw_frame(image, scene, view)
    return image


# ── turntable ────────────────────────────────────────────────────────────

def turntable_views(view: View, n_frames: int) -> list[View]:
    """``n_frames`` views one full turn about the vertical axis, starting
    at ``view`` and keeping its elevation and zoom."""
    n = int(n_frames)
    if n < 1:
        raise ValueError("a turntable needs at least one frame")
    step = 360.0 / n
    return [view.turned(d_azimuth=i * step) for i in range(n)]


def frame_ms(n_frames: int) -> int:
    """Milliseconds each frame shows for, so a full turn takes about
    ``TURN_SECONDS`` whatever the frame count (a GIF keeps delays in
    10 ms steps: 72 frames play at 80 ms, 5.76 s a turn)."""
    return max(20, int(round(TURN_SECONDS * 1000.0 / int(n_frames))))


# The longest side a turntable is saved at. Each frame is rendered, written
# and dropped in turn, so only one is held at a time; the cap keeps that
# frame, ffmpeg's buffers and Pillow's GIF palette images (which it keeps
# until the file is written, one byte a pixel) small whatever the window.
MAX_EXPORT_SIDE = 1280


def export_size(width: int, height: int) -> tuple[int, int]:
    """The size a turntable is saved at: the window's, shrunk so its long
    side is at most ``MAX_EXPORT_SIDE`` (keeping its shape), then even."""
    w, h = max(2, int(width)), max(2, int(height))
    k = min(1.0, MAX_EXPORT_SIDE / max(w, h))
    return even_size(int(w * k), int(h * k))


def turntable_frames(scene: Scene, view: View, n_frames: int, width: int,
                     height: int, progress: Callable[[int, int], None] | None = None,
                     ) -> Iterator[Any]:
    """The turntable's pictures, in order, each rendered only when the
    writer asks for it; ``progress(i, n)`` after each (it may raise to
    stop the turn part-way)."""
    views = turntable_views(view, n_frames)
    for i, v in enumerate(views):
        yield render(scene, v, width, height)
        if progress is not None:
            progress(i + 1, len(views))


def _again(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    """``first`` then ``rest``, keeping no hold on ``first`` once it has
    been handed on (``itertools.chain`` keeps it for the whole run)."""
    box = [first]
    del first
    yield box.pop()
    yield from rest


@contextmanager
def written_whole(path: str) -> Iterator[str]:
    """A ``.part`` path beside ``path`` to write to, put in place of
    ``path`` only once the writing finished; on any failure or cancel it
    is removed and ``path`` (an older file there too) is left alone."""
    part = path + '.part'
    try:
        yield part
        os.replace(part, path)
    except BaseException:
        with suppress(OSError):
            os.remove(part)
        raise


def save_gif(frames: Iterable[Any], path: str, n_frames: int) -> None:
    """Write ``frames`` (Pillow images, taken one at a time) as a GIF that
    loops forever, each turned into a palette image as it comes."""
    it = iter(frames)
    first = next(it, None)
    if first is None:
        raise ValueError("no frames to save")

    def palette(im: Any) -> Any:
        return im.convert('RGB').quantize(colors=256, dither=0)
    head = palette(first)
    del first                     # only the palette copies are kept
    with written_whole(path) as part:
        head.save(part, format='GIF', save_all=True,
                            append_images=(palette(f) for f in it),
                            duration=frame_ms(n_frames), loop=0, disposal=1,
                            optimize=False)


def mp4_encoder() -> str | None:
    """Path of an ffmpeg program to encode MP4 with, or None. OpenFlo does
    not ship one: it uses ffmpeg from the PATH, or the one the optional
    imageio-ffmpeg package carries."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except (OSError, RuntimeError):
        return None


def even_size(width: int, height: int) -> tuple[int, int]:
    """H.264 in yuv420p needs even dimensions: round each down to even."""
    return max(2, int(width) // 2 * 2), max(2, int(height) // 2 * 2)


def mp4_command(exe: str, width: int, height: int, fps: float,
                path: str) -> list[str]:
    """The ffmpeg command reading raw RGB frames on stdin and writing an
    H.264 MP4 most players and slide software open (``-f mp4``: the path
    may be a ``.part`` file)."""
    return [exe, '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{int(width)}x{int(height)}', '-r', f'{fps:g}', '-i', '-',
            '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-f', 'mp4', path]


# Windows: start ffmpeg with no console window (a windowless app would
# otherwise flash one over itself on every export).
_CREATE_NO_WINDOW = 0x08000000


def _popen_flags(platform: str | None = None) -> dict[str, Any]:
    if (platform or sys.platform) == 'win32':
        return {'creationflags': _CREATE_NO_WINDOW}
    return {}


def save_mp4(frames: Iterable[Any], path: str, exe: str, n_frames: int) -> None:
    """Encode ``frames`` (same-size Pillow images, even dimensions, taken
    one at a time) to an MP4 at ``path`` with the ffmpeg at ``exe``;
    raises RuntimeError with ffmpeg's own message when it fails."""
    it = iter(frames)
    first = next(it, None)
    if first is None:
        raise ValueError("no frames to save")
    w, h = first.size
    if (w, h) != even_size(w, h):
        raise ValueError("MP4 frames need an even width and height")
    fps = n_frames / TURN_SECONDS
    it = _again(first, it)
    del first                     # held only by the writing loop below
    with written_whole(path) as part:
        proc = subprocess.Popen(mp4_command(exe, w, h, fps, part),
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                **_popen_flags())
        assert proc.stdin is not None and proc.stderr is not None
        try:
            try:
                for f in it:
                    try:
                        proc.stdin.write(np.asarray(f.convert('RGB'),
                                                    dtype=np.uint8).tobytes())
                    except OSError:
                        # ffmpeg stopped reading (BrokenPipeError; on
                        # Windows OSError EINVAL): its own message says
                        # why, below.
                        break
                with suppress(OSError):
                    proc.stdin.close()
            except BaseException:
                proc.kill()
                proc.wait()
                raise
            err = proc.stderr.read().decode('utf-8', 'replace')
            code = proc.wait()
        finally:
            # Both pipes are closed however the save ends, a stopped one too.
            for pipe in (proc.stdin, proc.stderr):
                with suppress(OSError):
                    pipe.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg could not write the MP4: {err.strip()}")


def cluster_palette(labels: np.ndarray) -> list[tuple[Any, str]]:
    """``[(label, '#color')]`` for a cluster column: a whole-number id takes
    the color the editor gives that cluster's population (GATE_PALETTE by
    id), the noise id -1 is grey, anything else is colored in order."""
    from .cluster_labels import GATE_PALETTE, NOISE_CLUSTER_COLOR
    vals = [v for v in dict.fromkeys(np.asarray(labels).tolist()) if v == v]
    out = []
    for i, v in enumerate(sorted(vals, key=str)):
        if isinstance(v, (int, float)) and float(v).is_integer():
            c = (NOISE_CLUSTER_COLOR if int(v) < 0
                 else GATE_PALETTE[int(v) % len(GATE_PALETTE)])
        else:
            c = GATE_PALETTE[i % len(GATE_PALETTE)]
        out.append((v, c))
    return out


def save_embedding_turntable(coords: np.ndarray, labels: np.ndarray | None,
                             path: str, axis_names=('UMAP1', 'UMAP2', 'UMAP3'),
                             n_frames: int = DEFAULT_FRAMES, size: int = 600,
                             seed: int = DEFAULT_SEED,
                             cmap_name: str = 'turbo') -> int:
    """Write a looping GIF turntable of a 3-D embedding (``coords`` n x 3, on
    linear axes as the editor shows embeddings), colored by ``labels`` (a
    cluster column) or by 3D density on ``cmap_name`` without them (no Tk,
    so the CLI can call it headless). At most ``MAX_EVENTS``
    events, a subset picked with ``seed``. Returns the number drawn."""
    xyz = np.asarray(coords, dtype=float)
    ok = np.all(np.isfinite(xyz), axis=1)
    pos = np.flatnonzero(ok)
    pick = seeded_subset(len(pos), MAX_EVENTS, seed)
    if pick is not None:
        pos = pos[pick]
    xyz = xyz[pos]
    pts = np.column_stack([normalize(xyz[:, i], axis_bounds(xyz[:, i]))
                           for i in range(3)]) if len(xyz) else np.zeros((0, 3))
    if labels is not None:
        lab = np.asarray(labels)[pos]
        colors = category_colors(lab, cluster_palette(lab))
    else:
        colors = density_colors(pts, cmap_name)
    scene = Scene(points=pts, colors=colors,
                  labels=(str(axis_names[0]), str(axis_names[1]),
                          str(axis_names[2])))
    w, h = even_size(size, size)
    save_gif(turntable_frames(scene, View(), n_frames, w, h), path, n_frames)
    return len(pts)
