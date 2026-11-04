"""Figure exports have a transparent background by default (PNG, SVG, PDF and
the clipboard's PNG), in every path that writes one: the windows'
savefig_background, the layout editor's batch PDF, and the exports that call
savefig directly (HTML report, workspace and CLI plots). A JPEG cannot be
transparent, so it is written on the figure's own solid colour. Generated
figures only."""
from __future__ import annotations

import io
import re
import zlib

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from openflo import export_stamp, theme
from openflo import layout_page as lp
from openflo.async_task import wait_for_job
from tests.conftest import editor_or_skip
from tests.test_tool_window import _root_or_skip


@pytest.fixture(autouse=True)
def _no_footer(monkeypatch):
    # The provenance footer is grey text in a corner; keep it off the pixels
    # the checks read.
    monkeypatch.setattr(theme, 'read_prefs',
                        lambda: {'export_provenance': False})


def _figure(face='white', axes_face='white'):
    fig = Figure(figsize=(3, 2), dpi=50)
    fig.set_facecolor(face)
    ax = fig.add_subplot()
    ax.set_facecolor(axes_face)
    ax.plot([0, 1], [0, 1], color='tab:blue')
    return fig, ax


def _rgba(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert('RGBA'))


def _axes_pixel(fig, ax, img):
    """An RGBA pixel inside the axes, away from the line and the ticks."""
    fig.canvas.draw()
    x0, y0, x1, y1 = ax.get_window_extent().extents
    h = img.shape[0]
    sx = img.shape[1] / fig.bbox.width
    px = int((x0 + 0.85 * (x1 - x0)) * sx)          # lower right: no line
    py = h - int((y0 + 0.15 * (y1 - y0)) * sx)
    return img[py, px]


def _pdf_streams(data):
    for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
        try:
            yield zlib.decompress(m.group(1))
        except zlib.error:
            yield m.group(1)


def _pdf_paints_white(data):
    """Whether a PDF page fills a shape white: matplotlib sets the white
    (``1 g`` or ``1 1 1 rg``), draws the outline, and fills it (``f``)."""
    white = re.compile(rb'(?<![\d.])(?:1 g|1 1 1 rg)\n'
                       rb'(?:[-\d. ]+ [ml]\n)+h\n\s*f\n')
    return any(white.search(s) for s in _pdf_streams(data)
               if b'/DeviceRGB cs' in s)


# ── savefig_background ──────────────────────────────────────────────────

def test_png_export_is_transparent_by_default_outside_and_inside_the_axes():
    fig, ax = _figure()
    buf = io.BytesIO()
    theme.savefig_background(fig, buf, fmt='png', dpi=50, tight=False)
    img = _rgba(buf.getvalue())
    assert img[0, 0, 3] == 0                     # the figure's own face
    assert _axes_pixel(fig, ax, img)[3] == 0     # the axes face
    assert img[..., 3].max() > 200               # the line is still drawn
    # The preview keeps its white faces.
    assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)


def test_white_is_still_opaque_white_when_picked():
    fig, ax = _figure()
    buf = io.BytesIO()
    theme.savefig_background(fig, buf, background='White', fmt='png',
                             dpi=50, tight=False)
    img = _rgba(buf.getvalue())
    assert tuple(img[0, 0]) == (255, 255, 255, 255)
    assert tuple(_axes_pixel(fig, ax, img)) == (255, 255, 255, 255)


def test_svg_and_pdf_exports_paint_no_white_background(tmp_path):
    fig, _ax = _figure()
    svg, pdf = tmp_path / 'f.svg', tmp_path / 'f.pdf'
    theme.savefig_background(fig, svg)
    theme.savefig_background(fig, pdf)
    text = svg.read_text()
    assert '#ffffff' not in text.lower()
    assert not _pdf_paints_white(pdf.read_bytes())
    # The check sees a white background when one is there.
    theme.savefig_background(fig, svg, background='White')
    theme.savefig_background(fig, pdf, background='White')
    assert '#ffffff' in svg.read_text().lower()
    assert _pdf_paints_white(pdf.read_bytes())


@pytest.mark.parametrize('background', ['Transparent', 'Translucent'])
def test_jpeg_cannot_be_transparent_so_it_gets_the_figures_solid_colour(
        tmp_path, background):
    light, _ = _figure()
    out = tmp_path / 'light.jpg'
    theme.savefig_background(light, out, background=background, dpi=50,
                             tight=False)
    corner = np.asarray(Image.open(out).convert('RGB'))[0, 0].astype(int)
    assert (corner > 245).all()                  # white, not black
    dark, _ = _figure(face='#121317', axes_face='#121317')
    out = tmp_path / 'dark.jpg'
    theme.savefig_background(dark, out, background=background, dpi=50,
                             tight=False)
    corner = np.asarray(Image.open(out).convert('RGB'))[0, 0].astype(int)
    assert (abs(corner - [0x12, 0x13, 0x17]) < 8).all()   # its own dark face


# ── the helper for direct savefig calls ─────────────────────────────────

def test_background_kwargs_by_format():
    fig, _ = _figure(face='#336699')
    for path in ('a.png', 'a.svg', 'a.pdf', 'a.tif', 'a.eps'):
        assert export_stamp.background_kwargs(fig, path) == {
            'transparent': True}
    assert export_stamp.background_kwargs(fig, fmt='png') == {
        'transparent': True}
    for path in ('a.jpg', 'A.JPEG'):
        face = export_stamp.background_kwargs(fig, path)['facecolor']
        assert face == pytest.approx((0x33 / 255, 0x66 / 255, 0x99 / 255))
    clear, _ = _figure(face='none')
    assert export_stamp.background_kwargs(clear, 'a.jpg') == {
        'facecolor': (1.0, 1.0, 1.0)}


def test_html_report_figure_is_transparent():
    import base64

    from openflo.report import figure_to_data_uri
    fig, _ax = _figure()
    uri = figure_to_data_uri(fig, dpi=50)
    img = _rgba(base64.b64decode(uri.split(',', 1)[1]))
    assert img[0, 0, 3] == 0
    assert img[..., 3].max() > 200


def test_workspace_embedding_plot_is_transparent(tmp_path):
    import pandas as pd

    from openflo.pipeline import FlowSample
    from openflo.workspace_run import _plot_embedding
    rng = np.random.default_rng(0)
    s = FlowSample.from_dataframe(pd.DataFrame(
        {'UMAP1': rng.normal(size=200), 'UMAP2': rng.normal(size=200),
         'cluster': rng.integers(0, 3, 200)}), name='synthetic_1')
    out = _plot_embedding(s, 'UMAP', str(tmp_path), 'synthetic_1')
    assert out
    img = np.asarray(Image.open(out).convert('RGBA'))
    assert img[0, 0, 3] == 0
    assert img[..., 3].max() > 200


# The cluster heatmap's colormap call warns on the newest matplotlib, and the
# test settings turn that warning into a failed (skipped) heatmap.
@pytest.mark.filterwarnings('ignore::PendingDeprecationWarning')
def test_cli_plots_are_transparent(tmp_path, synthetic_fcs):
    from openflo import cli
    from tests.test_cli_dpi import _sample
    samples = [_sample(synthetic_fcs, 'sample_a'),
               _sample(synthetic_fcs, 'sample_b')]
    cli.save_plots(samples[0], str(tmp_path), dpi=40)
    cli.save_group_pair_scatters(samples, 'Grp', str(tmp_path), dpi=40)
    pngs = sorted(tmp_path.glob('*.png'))
    kinds = {p.stem.rsplit('_', 1)[1] for p in pngs}
    assert {'scatter', 'ssc', 'heatmap', 'overlay', 'grid'} <= kinds
    for png in pngs:
        assert np.asarray(Image.open(png).convert('RGBA'))[0, 0, 3] == 0, \
            png.name


# ── clipboard ───────────────────────────────────────────────────────────

def test_clipboard_png_is_transparent_and_the_windows_bitmap_is_on_white():
    from openflo.clipboard_export import figure_png, png_to_dib
    fig, _ = _figure()
    png = figure_png(fig, dpi=50)
    assert _rgba(png)[0, 0, 3] == 0
    dib = png_to_dib(png)
    bmp = Image.open(io.BytesIO(b'BM' + (14 + len(dib)).to_bytes(4, 'little')
                                + b'\0\0\0\0' + (54).to_bytes(4, 'little')
                                + dib)).convert('RGB')
    # Each pixel is the PNG laid on white: a faint (part-transparent) edge
    # pixel stays faint instead of turning solid when the alpha is dropped.
    rgba = _rgba(png).astype(float)
    a = rgba[..., 3:] / 255
    on_white = rgba[..., :3] * a + 255 * (1 - a)
    assert ((0 < rgba[..., 3]) & (rgba[..., 3] < 255)).any()
    assert np.abs(np.asarray(bmp).astype(float) - on_white).max() <= 1.5


# ── windows: the default pick ───────────────────────────────────────────


def test_background_box_starts_on_transparent_light_or_dark(monkeypatch):
    import openflo.tool_window as tw
    root = _root_or_skip()
    try:
        for dark in (False, True):
            # The old selector started on Dark in a dark dialog; the new one
            # no longer asks, so this patch only matters to the old code.
            monkeypatch.setattr(tw, 'dialog_dark_on', lambda _w, d=dark: d)
            var, combo = tw.background_selector(root)
            assert var.get() == 'Transparent'
            assert combo['values'][0] == 'Transparent'
            assert set(combo['values']) == {
                'Transparent', 'White', 'Dark', 'Translucent'}
        assert 'JPEG' in tw.BACKGROUND_TIP
    finally:
        root.destroy()


def test_main_plot_export_is_transparent(tmp_path, monkeypatch):
    root, ed = editor_or_skip()
    try:
        import openflo.editor_export as ee
        out = tmp_path / 'plot.png'
        ed._samples = {'s': object()}
        monkeypatch.setattr(ee.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        monkeypatch.setattr(ed, '_audit', lambda *a, **k: None)
        ed._save_plot_image()
        assert out.exists(), ed.status_var.get()
        assert np.asarray(Image.open(out).convert('RGBA'))[0, 0, 3] == 0
    finally:
        root.destroy()


def test_layout_batch_pdf_is_transparent_including_the_plot_faces(
        tmp_path, monkeypatch):
    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        monkeypatch.setattr(ule.LayoutEditorWindow, 'geometry',
                            lambda self, *a, **k: None)
        _load(ed, 's1', 1, 400, 300.0)
        _load(ed, 's2', 2, 400, 500.0)
        win = ule.LayoutEditorWindow(ed)
        win.withdraw()
        assert win.bg_var.get() == 'Transparent'
        win._insert(lp.plot_item(x='FSC-A', y='SSC-A', mode='dot',
                                 box=(0.1, 0.1, 0.5, 0.5)))
        out = tmp_path / 'batch.pdf'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        win._export_batch()
        assert wait_for_job(win._busy)
        assert not _pdf_paints_white(out.read_bytes())
        win.bg_var.set('White')                  # the check can see white
        win._export_batch()
        assert wait_for_job(win._busy)
        assert _pdf_paints_white(out.read_bytes())
        win.destroy()
    finally:
        root.destroy()


def test_volcano_export_is_transparent(tmp_path, monkeypatch):
    from tests.test_dialog_tooltips import _compare_results
    root, ed = editor_or_skip()
    try:
        import openflo.ui_diff as ud
        out = tmp_path / 'volcano.png'
        monkeypatch.setattr(ud.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        win = ud.CompareAllWindow(ed, _compare_results(), ['A', 'B'],
                                  '%Parent')
        win.withdraw()
        win._export_figure()
        img = np.asarray(Image.open(out).convert('RGBA'))
        assert img[0, 0, 3] == 0
        assert img[..., 3].max() > 200
        win.destroy()
    finally:
        root.destroy()


def test_no_window_hard_codes_an_opaque_export_background():
    """Every export starts on the shared default: no window passes its own
    White / Dark start to the Background box or to savefig_background."""
    from pathlib import Path

    import openflo
    pattern = re.compile(
        r"""StringVar\(value=['"](White|Dark)['"]\)"""
        r"""|background=['"](White|Dark)['"]"""
        r"""|['"]Dark['"] if .* else ['"]White['"]""")
    bad = []
    for path in sorted(Path(openflo.__file__).parent.glob('*.py')):
        for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(),
                                 1):
            # savefig_background's own JPEG fallback picks a solid colour.
            if pattern.search(line) and 'solid_face(fig)' not in line:
                bad.append(f'{path.name}:{n}')
    assert not bad, bad


def test_layout_batch_pdf_keeps_translucent(tmp_path, monkeypatch):
    from tests.test_layout_editor import _editor_or_skip, _load
    root, ed = _editor_or_skip()
    try:
        import openflo.ui_layout_editor as ule
        monkeypatch.setattr(ule.LayoutEditorWindow, 'geometry',
                            lambda self, *a, **k: None)
        _load(ed, 's1', 1, 400, 300.0)
        win = ule.LayoutEditorWindow(ed)
        win.withdraw()
        win._insert(lp.plot_item(x='FSC-A', y='SSC-A', mode='dot',
                                 box=(0.1, 0.1, 0.5, 0.5)))
        out = tmp_path / 'batch.pdf'
        monkeypatch.setattr(ule.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        for bg, wash in (('Translucent', True), ('White', False)):
            win.bg_var.set(bg)
            win._export_batch()
            assert wait_for_job(win._busy)
            assert (b'/ca 0.5' in out.read_bytes()) is wash, bg
        win.destroy()
    finally:
        root.destroy()


def test_clipboard_bitmap_of_a_dark_figure_is_on_its_own_face():
    from openflo.clipboard_export import figure_png, png_to_dib
    dark, _ = _figure(face='#121317', axes_face='#121317')
    face = tuple(round(255 * c) for c in export_stamp.solid_face(dark))
    dib = png_to_dib(figure_png(dark, dpi=50), face)
    bmp = Image.open(io.BytesIO(b'BM' + (14 + len(dib)).to_bytes(4, 'little')
                                + b'\0\0\0\0' + (54).to_bytes(4, 'little')
                                + dib)).convert('RGB')
    assert np.asarray(bmp)[0, 0].tolist() == [0x12, 0x13, 0x17]
