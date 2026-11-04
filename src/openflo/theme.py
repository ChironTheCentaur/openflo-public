"""Theme palette, dark-figure helpers, and publication figure export.

Shared appearance/rendering helpers, kept out of gui.py so the ~24 dialog
modules import this small module instead of the whole GUI. The heavy
``apply_theme`` / flat-indicator styling lives in ui_style.py (it's editor-only)
and pushes the active palette here via ``set_active_palette``.
"""
from __future__ import annotations

import tkinter as tk
import warnings
from typing import Literal

from .export_stamp import DEFAULT_EXPORT_BACKGROUND
from .group_palettes import (
    DEFAULT_GROUP_PALETTE,
    GROUP_PALETTES,
    n_group_colors,
    palette_colors,
)
from .group_palettes import (
    OKABE_ITO as OKABE_ITO,  # public: theme.OKABE_ITO
)
from .group_palettes import (
    TAB10 as TAB10,  # public: theme.TAB10
)
from .layout_page import EXPORT_DPI
from .prefs import read_prefs
from .ui_logic import (
    DARK_PLOT_THEMES,
    THEME_ALL_DARK,
    THEME_DARK_PLOT,
    THEME_DARK_WINDOWS,
    THEME_LIGHT,
    plot_is_dark,
)


def plt_get_cmap(name):
    """matplotlib colormap copy, so a window's colours never touch the global
    registry entry. Set extremes with ``.with_extremes(bad=...)``, which
    returns a new colormap; the in-place ``set_bad`` is deprecated. Uses the current ``matplotlib.colormaps`` API,
    falling back to the legacy ``cm.get_cmap`` on old matplotlib."""
    import matplotlib
    try:
        return matplotlib.colormaps[name].copy()
    except (AttributeError, KeyError):
        import matplotlib.cm as _cm
        return _cm.get_cmap(name).copy()


def dialog_dark_on(widget):
    """True if pop-up figures should render dark: they follow the plot
    background of the active theme (View, Plot background). Walks up from a
    dialog widget to the editor that owns ``_theme_var``."""
    w = widget
    for _ in range(8):
        tv = getattr(w, '_theme_var', None)
        if tv is not None:
            try:
                return plot_is_dark(tv.get())
            except Exception:  # noqa: BLE001 - theme variable may be unset or gone
                return False
        w = getattr(w, 'master', None)
        if w is None:
            break
    return False


def theme_figure_dark(fig):
    """Recolour a matplotlib Figure to the dark plot palette (figure, axes,
    ticks, labels, spines, grid, legend, suptitle) for dark-mode previews /
    exports. Best-effort."""
    pal = THEMES[THEME_ALL_DARK]
    bg, fg = pal['plot_bg'], pal['plot_fg']
    spine, grid = pal['plot_spine'], pal['plot_grid']
    try:
        fig.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(bg)
            ax.tick_params(colors=fg, which='both')
            for s in ax.spines.values():
                s.set_color(spine)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color(grid)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(bg)
                leg.get_frame().set_edgecolor(spine)
                for t in leg.get_texts():
                    t.set_color(fg)
        sup = getattr(fig, '_suptitle', None)
        if sup is not None:
            sup.set_color(fg)
    except Exception:  # noqa: BLE001 - figure colouring is cosmetic
        pass


def match_dialog(fig, widget) -> bool:
    """Recolour `fig` dark when the dialog `widget` follows a dark plot
    background; return whether it did."""
    dark = dialog_dark_on(widget)
    if dark:
        theme_figure_dark(fig)
    return dark


def _dark_for_save(fig):
    """Recolour a light ``fig`` dark (``theme_figure_dark``) and return a
    callable that puts every colour it changed back, so a Dark export of a
    light preview is dark without changing the preview. A figure that is
    already dark is left alone (the callable does nothing)."""
    from matplotlib.colors import to_hex
    if _is_dark(to_hex(fig.get_facecolor())):
        return lambda: None
    saved = []                         # (setter, its old value)

    def keep(setter, value):
        saved.append((setter, value))

    keep(fig.set_facecolor, fig.get_facecolor())
    sup = getattr(fig, '_suptitle', None)
    if sup is not None:
        keep(sup.set_color, sup.get_color())
    for ax in fig.axes:
        keep(ax.set_facecolor, ax.get_facecolor())
        for s in ax.spines.values():
            keep(s.set_edgecolor, s.get_edgecolor())
        for t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            keep(t.set_color, t.get_color())
        for gl in ax.get_xgridlines() + ax.get_ygridlines():
            keep(gl.set_color, gl.get_color())
        for name in ('x', 'y'):
            axis = getattr(ax, f'{name}axis')
            for which, ticks in (('major', axis.get_major_ticks()),
                                 ('minor', axis.get_minor_ticks())):
                if ticks:
                    t0 = ticks[0]
                    keep(lambda cols, a=ax, n=name, w=which: a.tick_params(
                        axis=n, which=w, color=cols[0], labelcolor=cols[1]),
                         (t0.tick1line.get_color(), t0.label1.get_color()))
        for t in ax.texts:
            keep(t.set_color, t.get_color())
        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            keep(frame.set_facecolor, frame.get_facecolor())
            keep(frame.set_edgecolor, frame.get_edgecolor())
            for t in leg.get_texts():
                keep(t.set_color, t.get_color())
    for t in fig.texts:
        keep(t.set_color, t.get_color())
    theme_figure_dark(fig)
    # Free text (annotations, figure captions) that theme_figure_dark leaves
    # alone: dark ink would vanish on the dark face.
    fg = THEMES[THEME_ALL_DARK]['plot_fg']
    for t in [*fig.texts, *(t for ax in fig.axes for t in ax.texts)]:
        if _is_dark(to_hex(t.get_color())):
            t.set_color(fg)

    def undo():
        for setter, value in reversed(saved):
            try:
                setter(value)
            except Exception:  # noqa: BLE001 - restoring colours is cosmetic
                pass
    return undo


def _provenance_footer(fig):
    """Stamp a subtle OpenFlo version (+ git SHA) footer onto ``fig`` for the
    duration of a save, and return a callable that removes it. Honours the
    ``export_provenance`` preference (default on). Reproducibility for
    paper-ready exports; the on-screen preview is never changed."""
    try:
        if not read_prefs().get('export_provenance', True):
            return lambda: None
        from .capabilities import build_provenance
        txt = fig.text(0.995, 0.005, build_provenance(), ha='right',
                       va='bottom', fontsize=6, color='#888888', alpha=0.7)
        return txt.remove
    except Exception:  # noqa: BLE001 - the footer is optional; it must not block the export
        return lambda: None


# Exported figures, unless the preferences say otherwise: EXPORT_DPI (imported
# above). It and EXPORT_DPI_CHOICES live once in the Tk-free layout model.

# Fonts offered for figures, those installed being shown. Journals usually ask
# for Arial or Helvetica; DejaVu Sans is matplotlib's own and always there.
FIGURE_FONTS = ('Arial', 'Helvetica', 'Liberation Sans', 'Calibri',
                'Segoe UI', 'DejaVu Sans', 'Times New Roman',
                'Liberation Serif')


def export_dpi():
    """The DPI raster exports use: the ``export_dpi`` preference, else
    EXPORT_DPI. Vector text and lines in PDF and SVG do not depend on it."""
    try:
        dpi = int(read_prefs().get('export_dpi', EXPORT_DPI))
    except (TypeError, ValueError):
        return EXPORT_DPI
    return dpi if 50 <= dpi <= 2400 else EXPORT_DPI


def installed_figure_fonts():
    """The FIGURE_FONTS matplotlib can find on this machine, in order."""
    try:
        from matplotlib import font_manager
        have = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # noqa: BLE001 - no usable font manager falls back to DejaVu Sans
        return ['DejaVu Sans']
    return [f for f in FIGURE_FONTS if f in have] or ['DejaVu Sans']


def figure_rc(prefs=None):
    """matplotlib settings every figure export runs under.

    Editable text (the ``export_editable_text`` preference, on by default):
    PDF and PostScript embed the font as TrueType (Type 42) instead of
    drawing each glyph as a Type 3 procedure, and SVG writes ``<text>``
    instead of paths, so Illustrator and Inkscape open the labels as text.
    The font (``figure_font``) leads the sans-serif list; empty keeps
    matplotlib's own."""
    import matplotlib
    prefs = read_prefs() if prefs is None else prefs
    rc = {}
    if prefs.get('export_editable_text', True):
        rc.update({'pdf.fonttype': 42, 'ps.fonttype': 42,
                   'svg.fonttype': 'none'})
    font = str(prefs.get('figure_font') or '').strip()
    if font:
        rest = [f for f in matplotlib.rcParams['font.sans-serif'] if f != font]
        rc.update({'font.family': ['sans-serif'],
                   'font.sans-serif': [font, *rest]})
    return rc


def apply_figure_font(prefs=None):
    """Make the ``figure_font`` preference the font of every plot drawn from
    now on (the GUI calls this at start and when the preference changes), so
    the screen shows what the export will. An empty preference puts back
    the fonts in use before the first change (matplotlibrc's, if any)."""
    import matplotlib
    keys = ('font.family', 'font.sans-serif')
    if not _FONT_BEFORE:
        _FONT_BEFORE.update({k: list(matplotlib.rcParams[k]) for k in keys})
    rc = figure_rc(prefs)
    for key in keys:
        matplotlib.rcParams[key] = rc.get(key, _FONT_BEFORE[key])


_FONT_BEFORE: dict = {}


def savefig_background(fig, path, background=DEFAULT_EXPORT_BACKGROUND,
                       dpi=None, tight=True, fmt=None, corrections=None):
    """Save ``fig`` with a publication-export background:

      • ``Transparent`` — full alpha (sits on a coloured page / poster /
                          slide); the default (DEFAULT_EXPORT_BACKGROUND).
                          The ink is the figure's own, so a dark-theme
                          figure keeps its light labels
      • ``White``       — opaque white
      • ``Dark``        — the dark plot palette; a light figure is
                          recoloured for the save only (_dark_for_save)
      • ``Translucent`` — 50% white wash (figure AND per-axes patches)

    The per-axes patch alpha is changed only for the duration of the save and
    restored afterwards, so an on-screen preview of ``fig`` is unaffected.
    PNG / PDF / SVG carry the alpha; TIFF may flatten it. JPEG cannot: a
    Transparent or Translucent JPEG is saved White, or Dark when the figure
    itself is dark, so its labels stay readable. ``tight=False``
    keeps the figure's own size (a layout page) instead of cropping it to
    its content. ``dpi`` None takes the export DPI preference; ``fmt``
    names the format when ``path`` is a file object. Text is saved
    editable and in the chosen font (figure_rc). The file's own metadata
    records the OpenFlo version and ``corrections`` (export_stamp).
    """
    import matplotlib
    from matplotlib.colors import to_hex

    from .export_stamp import keeps_transparency, savefig_metadata, solid_face
    if (background in ('Transparent', 'Translucent')
            and not keeps_transparency(path, fmt)):
        background = 'Dark' if _is_dark(to_hex(solid_face(fig))) else 'White'
    if dpi is None:
        dpi = export_dpi()
    kw = {'format': fmt} if fmt else {}
    meta = savefig_metadata(path, fmt, corrections)
    if meta:
        kw['metadata'] = meta
    bbox = 'tight' if tight else None
    axes_alpha = None
    if background == 'Transparent':
        kw['transparent'] = True
    elif background == 'Translucent':
        kw['facecolor'] = (1.0, 1.0, 1.0, 0.5)
        axes_alpha = 0.5
    elif background == 'Dark':
        # A figure already dark (View, Plot background: Dark) keeps its own
        # facecolor, so the export matches the preview. A light one is
        # recoloured dark for the save only, so Dark always writes a dark
        # file and the preview stays as it is.
        undark = _dark_for_save(fig)
        kw['facecolor'] = fig.get_facecolor()
        kw['edgecolor'] = 'none'
        _drop = _provenance_footer(fig)
        try:
            with matplotlib.rc_context(figure_rc()):
                fig.savefig(path, dpi=dpi, bbox_inches=bbox, **kw)
        finally:
            _drop()
            undark()
        return
    else:                              # White
        kw['facecolor'] = 'white'
    restore = []
    if axes_alpha is not None:
        for ax in fig.axes:
            restore.append((ax, ax.patch.get_facecolor()))
            ax.patch.set_facecolor((1.0, 1.0, 1.0, axes_alpha))
    _drop = _provenance_footer(fig)
    try:
        with matplotlib.rc_context(figure_rc()):
            fig.savefig(path, dpi=dpi, bbox_inches=bbox, edgecolor='none',
                        **kw)
    finally:
        _drop()
        for ax, fc in restore:
            ax.patch.set_facecolor(fc)


# The 'sleek' palette — a refined, near-black / off-white look with a cool accent
# and hairline borders (dependency-free; hand-styled onto the built-in clam theme
# by ui_style.apply_sleek). Plot canvases sit a touch off the chrome so the axes box
# reads as an inset. Same keys as before, so every current_palette() consumer
# works unchanged.
_PLOT_LIGHT = {'plot_bg': '#ffffff', 'plot_fg': '#1a1b1e',
               'plot_grid': '#ececf0', 'plot_spine': '#c9ccd3'}
_PLOT_MID = {'plot_bg': '#121317', 'plot_fg': '#e7e8ec',
             'plot_grid': '#23252c', 'plot_spine': '#3a3d47'}

# Near-black dark chrome, shared by 'dark' (white plots) and 'midnight' (dark
# plots) — only the plot canvas differs between them.
_SLEEK_DARK = {'bg': '#0f1013', 'panel': '#17181c', 'fg': '#e7e8ec',
               'accent': '#6ea8fe', 'accfg': '#0b1220', 'border': '#262830',
               'muted': '#8a8c96', 'active': '#20222a',
               'trough': '#0c0d10', 'thumb': '#2c2e37'}

_SLEEK_LIGHT = {'bg': '#f6f7f9', 'panel': '#ffffff', 'fg': '#1a1b1e',
                'accent': '#2f6fed', 'accfg': '#ffffff', 'border': '#e4e5ea',
                'muted': '#6b6e78', 'active': '#eceef2',
                'trough': '#e6e7eb', 'thumb': '#c6c9d1'}

# Message colours, a hover highlight and the log pane, per chrome, so none of
# them is hard-coded: every text colour reads at 4.5:1 or more on the
# window and panel colours (tests/test_theme_colours.py).
_STATUS_LIGHT = {'error': '#b42318', 'warning': '#8a5300', 'success': '#1d7a3a',
                 'highlight': '#fff2a8', 'highlight_fg': '#1a1b1e',
                 'console_bg': '#ffffff', 'console_fg': '#1a1b1e'}
_STATUS_DARK = {'error': '#ff8a80', 'warning': '#f0c060', 'success': '#7fd99a',
                'highlight': '#4a4322', 'highlight_fg': '#e7e8ec',
                'console_bg': '#0c0d10', 'console_fg': '#d4d4d4'}

# A theme is a window colour (chrome: light or dark) and a plot background
# (light or dark), chosen separately. The saved keys are named by the
# THEME_* constants in ui_logic, which say what each key holds.
THEMES = {
    THEME_LIGHT: dict(**_SLEEK_LIGHT, **_STATUS_LIGHT, **_PLOT_LIGHT),
    THEME_DARK_WINDOWS: dict(**_SLEEK_DARK, **_STATUS_DARK, **_PLOT_LIGHT),
    THEME_ALL_DARK: dict(**_SLEEK_DARK, **_STATUS_DARK, **_PLOT_MID),
    THEME_DARK_PLOT: dict(**_SLEEK_LIGHT, **_STATUS_LIGHT, **_PLOT_MID),
}

CHROME_NAMES = {'light': 'Light', 'dark': 'Dark'}
PLOT_BACKGROUND_NAMES = {'light': 'White', 'dark': 'Black'}
_THEME_KEYS = {('light', 'light'): THEME_LIGHT,
               ('dark', 'light'): THEME_DARK_WINDOWS,
               ('dark', 'dark'): THEME_ALL_DARK,
               ('light', 'dark'): THEME_DARK_PLOT}
# The public name PLOT_DARK_THEMES is deprecated and served by __getattr__
# below; use ui_logic.DARK_PLOT_THEMES.

# Old private names of the dialog-theme helpers, now public; served with a
# DeprecationWarning by __getattr__ below.
_RENAMED = {'_dialog_dark_on': 'dialog_dark_on',
            '_theme_figure_dark': 'theme_figure_dark'}


def __getattr__(name):
    if name == 'EXPORT_DPI_CHOICES':
        warnings.warn('theme.EXPORT_DPI_CHOICES is deprecated and will be '
                      'removed in a future release; use '
                      'layout_page.EXPORT_DPI_CHOICES', DeprecationWarning,
                      stacklevel=2)
        from .layout_page import EXPORT_DPI_CHOICES
        return EXPORT_DPI_CHOICES
    if name == 'PLOT_DARK_THEMES':
        warnings.warn('PLOT_DARK_THEMES is deprecated and will be removed in '
                      'a future release', DeprecationWarning, stacklevel=2)
        return set(DARK_PLOT_THEMES)
    if name in _RENAMED:
        new = _RENAMED[name]
        warnings.warn(f'{name} is deprecated and will be removed in a future '
                      f'release; use {new}', DeprecationWarning, stacklevel=2)
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def theme_key(chrome, plot):
    """The theme for a window colour and a plot background ('light' or
    'dark' each)."""
    return _THEME_KEYS.get((chrome, plot), 'light')


def theme_parts(key):
    """``(chrome, plot)`` of a theme key; an unknown key reads as light."""
    for parts, k in _THEME_KEYS.items():
        if k == key:
            return parts
    return ('light', 'light')


_ACTIVE_PALETTE = THEMES['light']
# Custom flat check/radio indicator elements (PIL-drawn, anti-aliased) keyed by
# theme; created once per theme and kept alive so Tk doesn't GC the images.


def current_palette():
    """The palette of the theme currently applied (chrome colours)."""
    return _ACTIVE_PALETTE


def _is_dark(hex_color):
    """True if ``hex_color`` (``#rrggbb``) is a dark surface (rel. luminance
    below the mid-point) — used to pick light-on-dark vs dark-on-light tints."""
    try:
        h = hex_color.lstrip('#')
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128
    except (ValueError, TypeError, AttributeError):
        return False


def plot_ink(widget):
    """Foreground colour for FREE plot annotations — significance brackets,
    empty-state text, reference spokes/labels — that ``theme_figure_dark``
    leaves untouched (it recolours spines/ticks/labels but not arbitrary
    ``ax.text`` / ``Line2D`` artists). Tracks the SAME dark/light decision as
    the figure (``dialog_dark_on``) so the ink stays legible on the canvas:
    the midnight plot foreground when the figure is dark, else a dark ink."""
    return THEMES[THEME_ALL_DARK if dialog_dark_on(widget)
                  else THEME_LIGHT]['plot_fg']


def flag_tints():
    """Treeview ``bad`` / ``warn`` row tints matched to the active chrome:
    light pink/amber rows (with dark ink) on light themes, deep tints (with the
    themed light ink) on dark themes, so flagged rows never render as an
    unreadable light band under Midnight. Returns ``{tag: (background,
    foreground)}``."""
    pal = current_palette()
    if _is_dark(pal['bg']):
        return {'bad': ('#4a1f1f', pal['fg']), 'warn': ('#463a1a', pal['fg'])}
    return {'bad': ('#ffe9e6', '#7a1f1f'), 'warn': ('#fff7d9', '#6a5310')}


# ── Crash reporting ──────────────────────────────────────────────────────────
# Flow-cytometry file paths and sample names can be identifying (subject IDs,
# study names). Rather than blunt redaction we TOKENISE: each sensitive value
# is replaced by a stable token (same value → same token everywhere, so the
# trace still correlates), and the token→value map is kept in a LOCAL sister
# file that is never meant to be submitted. The submittable log carries only
# tokens; the user (or a maintainer with the user's key file) can decode it.



def set_active_palette(pal):
    """Set the module-global active palette (called by ui_style.apply_theme)."""
    global _ACTIVE_PALETTE
    _ACTIVE_PALETTE = pal
    return pal


# ── Named fonts ─────────────────────────────────────────────────────────────
# Windows name these instead of a ('TkDefaultFont', 9, 'bold') tuple: a tuple
# is read as a family called "TkDefaultFont", which no system has, so Tk
# substituted another face, and its size did not follow the chrome's scaling.
# The named fonts take the default font's real family and are resized with
# the chrome (ensure_fonts is called from apply_theme and the window-width
# scaling), and none is below FONT_MIN points.
FONT_MIN = 9
NAMED_FONTS = ('OpenFloHeading', 'OpenFloSmall', 'OpenFloNote',
               'OpenFloLarge', 'OpenFloTitle', 'OpenFloMono')


def ensure_fonts(root, size=None):
    """Create, or resize to the chrome's ``size`` (points; the default
    font's size when None), the named fonts: Heading (bold), Small, Note
    (small italic), Large, Title (large bold) and Mono (fixed width)."""
    import tkinter.font as tkfont
    try:
        base = tkfont.nametofont('TkDefaultFont', root=root)
        fixed = tkfont.nametofont('TkFixedFont', root=root)
    except tk.TclError:
        return
    fam = base.actual('family')
    sz = int(size or abs(int(base.actual('size'))) or 10)
    small = max(FONT_MIN, sz - 1)
    mono = fixed.actual('family')
    specs: tuple[tuple[str, str, int, Literal['normal', 'bold'],
                       Literal['roman', 'italic']], ...] = (
        # name, family, size, weight, slant
        ('OpenFloHeading', fam, max(FONT_MIN, sz), 'bold', 'roman'),
        ('OpenFloSmall', fam, small, 'normal', 'roman'),
        ('OpenFloNote', fam, small, 'normal', 'italic'),
        ('OpenFloLarge', fam, max(FONT_MIN, sz + 1), 'normal', 'roman'),
        ('OpenFloTitle', fam, max(FONT_MIN, sz + 3), 'bold', 'roman'),
        ('OpenFloMono', mono, small, 'normal', 'roman'),
    )
    for name, family, pts, weight, slant in specs:
        try:
            tkfont.nametofont(name, root=root).configure(
                family=family, size=pts, weight=weight, slant=slant)
        except Exception:  # noqa: BLE001 - a named font Tk does not have yet is created below
            try:
                font = tkfont.Font(root=root, name=name, family=family,
                                   size=pts, weight=weight, slant=slant)
                # The Tk font outlives this Python object: Font deletes the
                # named font when collected, which would take it from every
                # widget using it.
                font.delete_font = False
            except tk.TclError:
                pass


# ── Density (pseudocolor) colour maps ───────────────────────────────────────
# Turbo keeps the familiar blue-to-red look of a flow pseudocolor plot but is
# close to perceptually uniform, so density steps read evenly; the classic
# 'jet' rainbow puts sharp false edges at cyan and yellow. Chosen in View →
# Density colors, persisted as the 'density_cmap' preference.
DENSITY_CMAPS = (('turbo', 'Turbo'), ('jet', 'Classic rainbow'),
                 ('viridis', 'Viridis'))
DEFAULT_DENSITY_CMAP = 'turbo'


def density_cmap() -> str:
    """The matplotlib colour map name pseudocolor plots use."""
    name = str(read_prefs().get('density_cmap', DEFAULT_DENSITY_CMAP))
    return name if name in dict(DENSITY_CMAPS) else DEFAULT_DENSITY_CMAP


# ── Group (categorical) colours ─────────────────────────────────────────────
# The palettes and the colour logic live in the Tk-free group_palettes module
# (the library colours groups without loading Tk); here the user's View →
# Group colors choice, the 'group_palette' preference, picks the palette.


def group_palette_name() -> str:
    """The group colour palette preference ('okabe_ito' or 'classic')."""
    name = str(read_prefs().get('group_palette', DEFAULT_GROUP_PALETTE))
    return name if name in dict(GROUP_PALETTES) else DEFAULT_GROUP_PALETTE


def group_colors(name: str | None = None) -> tuple[str, ...]:
    """The cycle of group colours, safest first; ``name`` None follows the
    preference (see group_palettes.palette_colors)."""
    return palette_colors(group_palette_name() if name is None else name)


def categorical_colors(n: int, name: str | None = None) -> list[str]:
    """``n`` distinct colours for ``n`` groups, in order; ``name`` None
    follows the preference (see group_palettes.n_group_colors)."""
    return n_group_colors(n, group_palette_name() if name is None else name)
