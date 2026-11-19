"""Editable vector figures (complaint 18): PDF text is embedded as TrueType,
SVG text stays text, the export DPI and figure font follow Preferences,
and export settings do not leak into other plots.

Each check has a negative partner (a setting switched off draws text as
shapes). Generated figures only.
"""
from __future__ import annotations

import io
import re
import struct

import tests._inline_editor as shared
from openflo import layout_page, theme
from tests._inline_editor import _figure

prefs = shared.prefs


def _save(fig, fmt, **kw):
    buf = io.BytesIO()
    theme.savefig_background(fig, buf, fmt=fmt, **kw)
    return buf.getvalue()


def test_pdf_text_is_embedded_as_truetype(prefs):
    pdf = _save(_figure(), 'pdf')
    assert b'/FontFile2' in pdf and b'/Type3' not in pdf
    prefs['export_editable_text'] = False
    pdf = _save(_figure(), 'pdf')
    assert b'/Type3' in pdf and b'/FontFile2' not in pdf


def test_svg_text_stays_text(prefs):
    svg = _save(_figure(), 'svg').decode()
    assert re.search(r'<text[^>]*>GATELABEL</text>', svg)
    prefs['export_editable_text'] = False
    svg = _save(_figure(), 'svg').decode()
    assert not re.search(r'<text[ >]', svg)       # glyphs drawn as paths


def test_export_settings_do_not_leak_into_other_plots(prefs):
    import matplotlib
    before = matplotlib.rcParams['pdf.fonttype']
    _save(_figure(), 'pdf')
    assert matplotlib.rcParams['pdf.fonttype'] == before


def _png_size(png):
    return struct.unpack('>II', png[16:24])


def test_export_dpi_follows_the_preference(prefs):
    fig = _figure()                        # 3 x 2 inches
    assert _png_size(_save(fig, 'png', tight=False)) == (900, 600)  # 300
    prefs['export_dpi'] = 600
    assert _png_size(_save(fig, 'png', tight=False)) == (1800, 1200)
    prefs['export_dpi'] = 'junk'
    assert theme.export_dpi() == layout_page.EXPORT_DPI == 300
    # A caller's explicit DPI (the layout editor's own) still wins.
    assert _png_size(_save(fig, 'png', tight=False, dpi=100)) == (300, 200)


def test_figure_font_preference_sets_and_restores_the_plot_font(prefs):
    import matplotlib
    keys = ('font.family', 'font.sans-serif')
    original = {k: list(matplotlib.rcParams[k]) for k in keys}
    try:
        prefs['figure_font'] = 'DejaVu Serif'
        theme.apply_figure_font()
        assert matplotlib.rcParams['font.sans-serif'][0] == 'DejaVu Serif'
        fig = _figure()                    # drawn after the change
        pdf = _save(fig, 'pdf')
        assert b'DejaVuSerif' in pdf
        prefs['figure_font'] = ''
        theme.apply_figure_font()
        assert {k: list(matplotlib.rcParams[k]) for k in keys} == original
        assert b'DejaVuSerif' not in _save(_figure(), 'pdf')
    finally:
        for k, v in original.items():
            matplotlib.rcParams[k] = v


def test_installed_fonts_are_a_subset_of_the_offer():
    fonts = theme.installed_figure_fonts()
    assert fonts and set(fonts) <= set(theme.FIGURE_FONTS)
    assert 'DejaVu Sans' in fonts           # matplotlib's own, always there
