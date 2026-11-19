"""Preferences > Appearance accessibility settings: Text size, Color-blind
safe colors and High contrast.

- Window text never drops below theme.FONT_MIN (narrow windows used to
  step down to 7 pt), and Text size adds points to it, reaching the ttk
  style, the named fonts, Tk's own fonts (menus, Text, lists) and the tree
  row height, live from the Preferences window.
- Color-blind safe colors draws gates that still have their automatic
  colour, backgating and the density colouring in colour-blind safe colours,
  without changing the colour stored on the gate.
- High contrast strengthens secondary text and borders (7:1 / 3:1) and
  draws gates twice as thick and solid.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from openflo import plotmath  # noqa: E402
from openflo.cluster_labels import GATE_PALETTE  # noqa: E402
from openflo.editor_gating import GatingMixin  # noqa: E402
from openflo.group_palettes import SAFE_GATE_COLORS, safe_gate_color  # noqa: E402
from tests.test_theme_colours import contrast  # noqa: E402

X, Y = 'FL1-A', 'FL2-A'


@pytest.fixture
def prefs_file(monkeypatch, tmp_path):
    """A prefs.json of the test's own; returns a writer for it."""
    from openflo import prefs
    path = tmp_path / 'prefs.json'
    monkeypatch.setattr(prefs, '_prefs_path', lambda: str(path))

    def put(**values):
        cur = json.loads(path.read_text()) if path.exists() else {}
        cur.update(values)
        path.write_text(json.dumps(cur))
    return put


def _hue(c):
    r, g, b = mcolors.to_rgb(c)
    return ('red', 'green', 'blue')[int(np.argmax([r, g, b]))]


def _same(a, b):
    return np.allclose(mcolors.to_rgba(a)[:3], mcolors.to_rgba(b)[:3])


# ── Text size ──────────────────────────────────────────────────────────────

def test_window_text_never_goes_below_the_minimum_at_any_width():
    from openflo.theme import CHROME_WIDE, FONT_MIN, chrome_font_size
    for w in (320, 800, 980, 1040, 1100, 1199, 1200, 1379):
        assert chrome_font_size(w) == FONT_MIN, w
    assert chrome_font_size(CHROME_WIDE) == 10
    assert chrome_font_size(2560) == 10


def test_text_size_adds_its_points_at_every_width():
    from openflo.prefs import TEXT_SIZES, text_size_points
    from openflo.theme import FONT_MIN, chrome_font_size
    assert [pts for _k, _l, pts in TEXT_SIZES] == [0, 2, 4, 6]
    for key, _lbl, pts in TEXT_SIZES:
        assert text_size_points(key) == pts
        assert chrome_font_size(900, text_size_points(key)) == FONT_MIN + pts
        assert chrome_font_size(1500, text_size_points(key)) == 10 + pts


def test_text_size_preference_reads_back_and_unknown_is_default(prefs_file):
    from openflo.prefs import load_text_size, text_size_points
    assert load_text_size() == 'default' and text_size_points() == 0
    prefs_file(text_size='larger')
    assert load_text_size() == 'larger' and text_size_points() == 4
    prefs_file(text_size='huge')
    assert load_text_size() == 'default' and text_size_points() == 0


def _editor():
    from tests.conftest import editor_or_skip
    return editor_or_skip()


def _tk_sizes(root):
    import tkinter.font as tkfont
    return {n: int(tkfont.nametofont(n, root=root).cget('size'))
            for n in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont',
                      'TkHeadingFont', 'TkFixedFont')}


def test_preferences_text_size_resizes_every_font_and_tree_rows_live(
        prefs_file):
    import tkinter.font as tkfont
    from tkinter import ttk

    from openflo.ui_preferences import PreferencesDialog
    root, ed = _editor()
    try:
        ed.winfo_width = lambda: 1500
        ed._apply_chrome_scale(force=True)
        st = ttk.Style(ed)
        base_tk = _tk_sizes(ed)
        base_rows = int(st.lookup('Treeview', 'rowheight'))
        assert ed._chrome_font_size == 10
        dlg = PreferencesDialog(ed)
        dlg._text_size_var.set('Largest')
        dlg.text_size_combo.event_generate('<<ComboboxSelected>>')
        from openflo.prefs import load_text_size
        assert load_text_size() == 'largest'
        assert ed._chrome_font_size == 16
        assert '16' in str(st.lookup('.', 'font'))
        assert abs(int(tkfont.nametofont('OpenFloHeading', root=ed)
                       .actual('size'))) == 16
        grown = _tk_sizes(ed)
        for name, size in base_tk.items():
            assert abs(grown[name]) == abs(size) + 6, (name, size, grown)
        assert int(st.lookup('Treeview', 'rowheight')) > base_rows
        # Back to Default puts Tk's own fonts back where they were.
        dlg._text_size_var.set('Default')
        dlg.text_size_combo.event_generate('<<ComboboxSelected>>')
        assert ed._chrome_font_size == 10
        assert _tk_sizes(ed) == base_tk
        dlg.destroy()
    finally:
        root.destroy()


def test_a_narrow_main_window_keeps_readable_text(prefs_file):
    from tkinter import ttk

    from openflo.theme import FONT_MIN
    root, ed = _editor()
    try:
        ed.winfo_width = lambda: 980
        ed._apply_chrome_scale(force=True)
        assert ed._chrome_font_size == FONT_MIN
        assert str(FONT_MIN) in str(ttk.Style(ed).lookup('.', 'font'))
        prefs_file(text_size='large')
        ed._apply_chrome_scale(force=True)
        assert ed._chrome_font_size == FONT_MIN + 2
        # Tree rows never get shorter than the 26 px they always had.
        prefs_file(text_size='default')
        ed._apply_chrome_scale(force=True)
        assert int(ttk.Style(ed).lookup('Treeview', 'rowheight')) == 26
    finally:
        root.destroy()


def test_tk_font_sizes_in_pixels_grow_too(prefs_file):
    """Tk keeps a size in pixels as a negative number: larger text makes
    it more negative, and Default puts it back."""
    import tkinter.font as tkfont
    root, ed = _editor()
    try:
        font = tkfont.nametofont('TkFixedFont', root=ed)
        font.configure(size=-13)
        ed._tk_font_base = None              # first seen again, at -13
        ed._apply_text_size_to_tk_fonts(4)
        assert int(font.cget('size')) == -17
        ed._apply_text_size_to_tk_fonts(0)
        assert int(font.cget('size')) == -13
    finally:
        root.destroy()


# ── Colour-blind safe colours ───────────────────────────────────────────────

def test_safe_gate_colours_replace_the_red_and_green_automatic_colours():
    assert safe_gate_color(GATE_PALETTE[0]) == SAFE_GATE_COLORS[0]
    assert safe_gate_color(GATE_PALETTE[1]) == SAFE_GATE_COLORS[1]
    assert safe_gate_color(GATE_PALETTE[1].upper()) == SAFE_GATE_COLORS[1]
    assert safe_gate_color('red') == SAFE_GATE_COLORS[0]
    for i, c in enumerate(GATE_PALETTE):
        assert safe_gate_color(c) == SAFE_GATE_COLORS[i % len(SAFE_GATE_COLORS)]
    # A colour picked by hand or imported is kept; nothing is not a colour.
    assert safe_gate_color('#123456') == '#123456'
    assert safe_gate_color(None) is None
    # The safe set has no yellow (a thin line vanishes on white) and no
    # repeats.
    assert '#F0E442' not in SAFE_GATE_COLORS
    assert len(set(SAFE_GATE_COLORS)) == len(SAFE_GATE_COLORS)


class _Ed(GatingMixin):
    """Just what _draw_gates reads, on a real matplotlib axes."""

    _ellipse_params = staticmethod(plotmath.ellipse_params)
    _ellipse_geom = staticmethod(plotmath.ellipse_geom)

    def __init__(self, gates):
        self._gates = dict(gates)
        self.ax = Figure().add_subplot()


def _gates():
    return {
        'r': {'kind': 'rect', 'x_channel': X, 'y_channel': Y, 'x0': 1.0,
              'x1': 5.0, 'y0': 1.0, 'y1': 5.0, 'color': GATE_PALETTE[1]},
        'p': {'kind': 'polygon', 'x_channel': X, 'y_channel': Y,
              'vertices': [[1, 1], [4, 1], [2, 4]], 'color': '#123456'},
        'e': {'kind': 'ellipsoid', 'x_channel': X, 'y_channel': Y,
              'mean': [3.0, 3.0], 'cov': [[1.0, 0.0], [0.0, 1.0]],
              'distance_sq': 1.0, 'color': GATE_PALETTE[0]},
        't': {'kind': 'threshold', 'channel': X, 'value': 2.0},
        'i': {'kind': 'interval', 'channel': Y, 'lo': 1.0, 'hi': 3.0,
              'color': GATE_PALETTE[3]},
        # The same kinds on the other axis: horizontal / vertical lines.
        't2': {'kind': 'threshold', 'channel': Y, 'value': 2.5},
        'i2': {'kind': 'interval', 'channel': X, 'lo': 1.5, 'hi': 3.5},
    }


def test_safe_colours_draw_automatic_gate_colours_safe_and_keep_the_rest(
        prefs_file):
    gates = _gates()
    ed = _Ed(gates)
    ed._draw_gates(X, Y)
    assert _same(ed._shape_artists['r'].get_edgecolor(), GATE_PALETTE[1])
    assert _same(ed._vlines['t'].get_color(), 'red')
    prefs_file(safe_colors=True)
    ed = _Ed(gates)
    ed._draw_gates(X, Y)
    assert _same(ed._shape_artists['r'].get_edgecolor(), SAFE_GATE_COLORS[1])
    assert _same(ed._shape_artists['e'].get_edgecolor(), SAFE_GATE_COLORS[0])
    assert _same(ed._shape_artists['p'].get_edgecolor(), '#123456')
    assert _same(ed._vlines['t'].get_color(), SAFE_GATE_COLORS[0])
    assert _same(ed._hlines['i:lo'].get_color(), SAFE_GATE_COLORS[3])
    # The gate keeps its own colour: off again draws it as before.
    assert gates['r']['color'] == GATE_PALETTE[1]


def test_safe_colours_make_density_viridis_over_the_view_menu_choice(
        prefs_file):
    from openflo.theme import density_cmap
    prefs_file(density_cmap='jet')
    assert density_cmap() == 'jet'
    prefs_file(safe_colors=True)
    assert density_cmap() == 'viridis'
    prefs_file(safe_colors=False)
    assert density_cmap() == 'jet'


def test_safe_colours_reach_backgate_and_gate_labels_and_highlight(
        prefs_file):
    from tests.test_gui_ux import _load_sample
    root, ed = _editor()
    try:
        _load_sample(ed, cols=('FSC-A', 'SSC-A', X, Y))
        ed._set_active_sample('s1')
        ed.x_combo.set(X)
        ed.y_combo.set(Y)
        gid = ed._add_gate({'kind': 'rect', 'x_channel': X, 'y_channel': Y,
                            'x0': -1.0, 'x1': 0.5, 'y0': -1.0, 'y1': 2.0,
                            'name': 'low', 'parent_id': None,
                            'color': GATE_PALETTE[1]})
        own = ed._sample_gates['s1'][gid]

        def row_colour():
            ed._refresh_gate_list()
            return str(ed.gate_tv.tag_configure(f'col_s1_{gid}', 'foreground'))
        assert _same(row_colour(), GATE_PALETTE[1])
        ed._gate_labels_var.set(True)
        ed._replot()
        texts = [t for t in ed.ax.texts if 'low' in t.get_text()]
        # (a label may be darkened to read on white; its hue stays)
        assert texts and _hue(texts[0].get_color()) == 'green'
        # Backgate: a gate's automatic colour and the fallback set.
        assert ed._backgate_color('s1', gid, 0) == GATE_PALETTE[1]
        assert ed._backgate_color('s1', 'none', 1) == ed._BACKGATE_COLORS[1]
        prefs_file(safe_colors=True)
        # The Samples & Gates row matches the colour the plot draws.
        assert _same(row_colour(), SAFE_GATE_COLORS[1])
        assert ed._backgate_color('s1', gid, 0) == SAFE_GATE_COLORS[1]
        assert ed._backgate_color('s1', 'none', 1) == SAFE_GATE_COLORS[1]
        # Highlight mode paints the gate's events in the safe colour.
        ed.gate_display_var.set('highlight')
        ed._replot()
        hl = [c for c in ed.ax.collections
              if str(c.get_label()).startswith('gate ')]
        assert hl and _same(hl[0].get_facecolor()[0], SAFE_GATE_COLORS[1])
        # Labels on the plot follow the colour the gate is drawn in.
        texts = [t for t in ed.ax.texts if 'low' in t.get_text()]
        assert texts and _hue(texts[0].get_color()) == 'blue'
        drawn = ed._shape_artists[gid].get_edgecolor()
        assert _same(drawn, SAFE_GATE_COLORS[1])
        assert own['color'] == GATE_PALETTE[1]
    finally:
        root.destroy()


# ── High contrast ──────────────────────────────────────────────────────────

def test_high_contrast_palettes_read_at_aaa_on_every_theme():
    from openflo.theme import THEMES, high_contrast_palette
    for name, pal in THEMES.items():
        hc = high_contrast_palette(pal)
        for ground in ('bg', 'panel', 'active'):
            assert contrast(hc['muted'], hc[ground]) >= 7.0, (name, ground)
        for ground in ('bg', 'panel'):
            assert contrast(hc['border'], hc[ground]) >= 3.0, (name, ground)
            assert contrast(hc['border'], hc[ground]) > contrast(
                pal['border'], pal[ground])
        assert hc['fg'] == pal['fg'] and hc['bg'] == pal['bg']
        assert pal is not hc and THEMES[name] == pal


def test_high_contrast_draws_gates_thick_and_solid(prefs_file):
    ed = _Ed(_gates())
    ed._draw_gates(X, Y)
    thin = {k: (a.get_linewidth(), a.get_alpha())
            for k, a in {**ed._shape_artists, **ed._vlines,
                         **ed._hlines}.items() if ':rot' not in k}
    prefs_file(high_contrast=True)
    ed = _Ed(_gates())
    ed._draw_gates(X, Y)
    for k, (lw, _a) in thin.items():
        art = {**ed._shape_artists, **ed._vlines, **ed._hlines}[k]
        assert art.get_linewidth() == pytest.approx(lw * 2), k
        assert art.get_alpha() == 1.0, k


def test_preferences_high_contrast_retheme_the_windows(prefs_file):
    from openflo.theme import (
        THEMES,
        current_palette,
        high_contrast_palette,
        set_active_palette,
    )
    from openflo.ui_preferences import PreferencesDialog
    root, ed = _editor()
    mode = ed._theme_var.get()
    try:
        # View > Density colors names no reason while safe colours are off.
        ed._density_cmap_var.set('jet')
        ed._set_density_cmap()
        assert 'Viridis' not in ed.status_var.get()
        assert current_palette()['muted'] == THEMES[mode]['muted']
        dlg = PreferencesDialog(ed)
        dlg._high_contrast_var.set(True)
        dlg.high_contrast_cb.invoke()        # toggles it back off ...
        dlg.high_contrast_cb.invoke()        # ... and on, through its command
        assert dlg._high_contrast_var.get() is True
        assert current_palette()['muted'] == high_contrast_palette(
            THEMES[mode])['muted']
        dlg.high_contrast_cb.invoke()
        assert current_palette()['muted'] == THEMES[mode]['muted']
        replots, rows = [], []
        ed._schedule_replot = lambda *a: replots.append(a)
        ed._refresh_gate_list = lambda *a: rows.append(a)
        dlg.safe_colors_cb.invoke()
        from openflo.prefs import load_safe_colors
        assert load_safe_colors() is True and replots and rows
        dlg.destroy()
        # View > Density colors says why the plot stays Viridis.
        ed._density_cmap_var.set('jet')
        ed._set_density_cmap()
        assert 'Viridis' in ed.status_var.get()
    finally:
        set_active_palette(THEMES[mode])     # no high contrast left behind
        root.destroy()


def test_density_menu_shows_the_saved_choice_under_safe_colours(prefs_file):
    """The View > Density colors tick is the saved map, not the Viridis
    drawn over it, so turning safe colours off shows the map then drawn."""
    prefs_file(density_cmap='jet', safe_colors=True)
    root, ed = _editor()
    try:
        assert ed._density_cmap_var.get() == 'jet'
    finally:
        root.destroy()
