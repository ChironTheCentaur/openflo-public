"""Theme palette, dark-figure helpers, and publication figure export.

Shared appearance/rendering helpers, kept out of gui.py so the ~24 dialog
modules import this small module instead of the whole GUI. The heavy
``apply_theme`` / flat-indicator styling stays in gui.py (it's editor-only) and
pushes the active palette here via ``set_active_palette``.
"""
from __future__ import annotations

from typing import Literal

from .prefs import read_prefs
from .ui_logic import should_use_dark


def plt_get_cmap(name):
    """matplotlib colormap copy (so per-window set_bad doesn't mutate the
    global registry entry). Uses the current ``matplotlib.colormaps`` API,
    falling back to the legacy ``cm.get_cmap`` on old matplotlib."""
    import matplotlib
    try:
        return matplotlib.colormaps[name].copy()
    except (AttributeError, KeyError):
        import matplotlib.cm as _cm
        return _cm.get_cmap(name).copy()


def _dialog_dark_on(widget):
    """True if pop-up figures should render dark: they follow the plot
    background of the active theme (View, Plot background). Walks up from a
    dialog widget to the editor that owns ``_theme_var``."""
    w = widget
    for _ in range(8):
        tv = getattr(w, '_theme_var', None)
        if tv is not None:
            try:
                return should_use_dark(tv.get())
            except Exception:
                return False
        w = getattr(w, 'master', None)
        if w is None:
            break
    return False


def _theme_figure_dark(fig):
    """Recolour a matplotlib Figure to the dark plot palette (figure, axes,
    ticks, labels, spines, grid, legend, suptitle) for dark-mode previews /
    exports. Best-effort."""
    pal = THEMES['midnight']
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
    except Exception:
        pass


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
    except Exception:
        return lambda: None


# Exported figures, unless the preferences say otherwise.
EXPORT_DPI = 300
EXPORT_DPI_CHOICES = (150, 300, 600, 1200)

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
    except Exception:
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


def savefig_background(fig, path, background='White', dpi=None, tight=True,
                       fmt=None):
    """Save ``fig`` with a publication-export background:

      • ``White``       — opaque white (default)
      • ``Transparent`` — full alpha (sits on a coloured page / poster / slide)
      • ``Translucent`` — 50% white wash (figure AND per-axes patches)

    The per-axes patch alpha is changed only for the duration of the save and
    restored afterwards, so an on-screen preview of ``fig`` is unaffected.
    PNG / PDF / SVG carry the alpha; TIFF may flatten it. ``tight=False``
    keeps the figure's own size (a layout page) instead of cropping it to
    its content. ``dpi`` None takes the export DPI preference; ``fmt``
    names the format when ``path`` is a file object. Text is saved
    editable and in the chosen font (figure_rc).
    """
    import matplotlib
    if dpi is None:
        dpi = export_dpi()
    kw = {'format': fmt} if fmt else {}
    bbox = 'tight' if tight else None
    axes_alpha = None
    if background == 'Transparent':
        kw['transparent'] = True
    elif background == 'Translucent':
        kw['facecolor'] = (1.0, 1.0, 1.0, 0.5)
        axes_alpha = 0.5
    elif background == 'Dark':
        # Keep the figure's own (dark) facecolor — used when the preview is
        # already dark (View, Plot background: Dark), so the export matches it.
        kw['facecolor'] = fig.get_facecolor()
        kw['edgecolor'] = 'none'
        _drop = _provenance_footer(fig)
        try:
            with matplotlib.rc_context(figure_rc()):
                fig.savefig(path, dpi=dpi, bbox_inches=bbox, **kw)
        finally:
            _drop()
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
# by gui._apply_sleek). Plot canvases sit a touch off the chrome so the axes box
# reads as an inset. Same keys as before, so every current_palette() consumer
# works unchanged.
_PLOT_LIGHT = dict(plot_bg='#ffffff', plot_fg='#1a1b1e',
                   plot_grid='#ececf0', plot_spine='#c9ccd3')
_PLOT_MID = dict(plot_bg='#121317', plot_fg='#e7e8ec',
                 plot_grid='#23252c', plot_spine='#3a3d47')

# Near-black dark chrome, shared by 'dark' (white plots) and 'midnight' (dark
# plots) — only the plot canvas differs between them.
_SLEEK_DARK = dict(bg='#0f1013', panel='#17181c', fg='#e7e8ec',
                   accent='#6ea8fe', accfg='#0b1220', border='#262830',
                   muted='#8a8c96', active='#20222a',
                   trough='#0c0d10', thumb='#2c2e37')

_SLEEK_LIGHT = dict(bg='#f6f7f9', panel='#ffffff', fg='#1a1b1e',
                    accent='#2f6fed', accfg='#ffffff', border='#e4e5ea',
                    muted='#6b6e78', active='#eceef2',
                    trough='#e6e7eb', thumb='#c6c9d1')

# Message colours, a hover highlight and the log pane, per chrome, so none of
# them is hard-coded (UI 37): every text colour reads at 4.5:1 or more on the
# window and panel colours (tests/test_batch_i_menus_fonts_colours.py).
_STATUS_LIGHT = dict(error='#b42318', warning='#8a5300', success='#1d7a3a',
                     highlight='#fff2a8', highlight_fg='#1a1b1e',
                     console_bg='#ffffff', console_fg='#1a1b1e')
_STATUS_DARK = dict(error='#ff8a80', warning='#f0c060', success='#7fd99a',
                    highlight='#4a4322', highlight_fg='#e7e8ec',
                    console_bg='#0c0d10', console_fg='#d4d4d4')

# A theme is a window colour (chrome: light or dark) and a plot background
# (light or dark), chosen separately (UI 38). Keys kept from the three themes
# saved before: 'light', 'dark' (dark chrome, light plot) and 'midnight'
# (dark chrome, dark plot); 'light-dark' is light chrome with a dark plot.
THEMES = {
    'light': dict(**_SLEEK_LIGHT, **_STATUS_LIGHT, **_PLOT_LIGHT),
    'dark':  dict(**_SLEEK_DARK, **_STATUS_DARK, **_PLOT_LIGHT),
    'midnight': dict(**_SLEEK_DARK, **_STATUS_DARK, **_PLOT_MID),
    'light-dark': dict(**_SLEEK_LIGHT, **_STATUS_LIGHT, **_PLOT_MID),
}

CHROME_NAMES = {'light': 'Light', 'dark': 'Dark'}
PLOT_BACKGROUND_NAMES = {'light': 'White', 'dark': 'Black'}
_THEME_KEYS = {('light', 'light'): 'light', ('dark', 'light'): 'dark',
               ('dark', 'dark'): 'midnight', ('light', 'dark'): 'light-dark'}
# Themes whose plots (and pop-up figures) are dark.
PLOT_DARK_THEMES = {'midnight', 'light-dark'}


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


# Themes whose CHROME (window, panels, title bar) is dark.
_DARK_MODES = {'dark', 'midnight'}


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
    except Exception:
        return False


def plot_ink(widget):
    """Foreground colour for FREE plot annotations — significance brackets,
    empty-state text, reference spokes/labels — that ``_theme_figure_dark``
    leaves untouched (it recolours spines/ticks/labels but not arbitrary
    ``ax.text`` / ``Line2D`` artists). Tracks the SAME dark/light decision as
    the figure (``_dialog_dark_on``) so the ink stays legible on the canvas:
    the midnight plot foreground when the figure is dark, else a dark ink."""
    return THEMES['midnight' if _dialog_dark_on(widget) else 'light']['plot_fg']


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
    """Set the module-global active palette (called by gui.apply_theme)."""
    global _ACTIVE_PALETTE
    _ACTIVE_PALETTE = pal
    return pal


# ── Named fonts (UI 36) ─────────────────────────────────────────────────────
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
    except Exception:
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
        except Exception:
            try:
                font = tkfont.Font(root=root, name=name, family=family,
                                   size=pts, weight=weight, slant=slant)
                # The Tk font outlives this Python object: Font deletes the
                # named font when collected, which would take it from every
                # widget using it.
                font.delete_font = False
            except Exception:
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
# Samples, conditions and groups are told apart by colour. matplotlib's tab10
# puts a red and a green side by side, the pair the commonest colour-vision
# deficiency (red-green, about 1 man in 12) confuses. Okabe and Ito's
# colour-universal palette (2002) stays distinct under every common
# deficiency; its black is swapped for a dark grey so it shows on the dark
# theme as well as on white (a lighter grey would sit close to its pink for
# a deuteranope). Chosen in View → Group colors, persisted as the
# 'group_palette' preference.
OKABE_ITO = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9',
             '#D55E00', '#F0E442', '#666666')
TAB10 = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')
GROUP_PALETTES = (('okabe_ito', 'Color-blind safe (Okabe-Ito)'),
                  ('classic', 'Classic (tab10)'))
DEFAULT_GROUP_PALETTE = 'okabe_ito'


def group_palette_name() -> str:
    """The group colour palette preference ('okabe_ito' or 'classic')."""
    name = str(read_prefs().get('group_palette', DEFAULT_GROUP_PALETTE))
    return name if name in dict(GROUP_PALETTES) else DEFAULT_GROUP_PALETTE


def group_colors(name: str | None = None) -> tuple[str, ...]:
    """The cycle of group colours, safest first.

    Only eight colours stay distinct under colour-vision deficiency, so the
    Okabe-Ito cycle continues with tab10's colours that it lacks (brown,
    olive, cyan, then the rest) for a ninth group onwards rather than
    repeating a colour."""
    name = group_palette_name() if name is None else name
    if name == 'classic':
        return TAB10
    extra = tuple(c for c in (TAB10[5], TAB10[8], TAB10[9], TAB10[3],
                              TAB10[4], TAB10[6]) if c not in OKABE_ITO)
    return OKABE_ITO + extra


def categorical_colors(n: int, name: str | None = None) -> list[str]:
    """``n`` distinct colours for ``n`` groups, in order.

    Up to the palette's own length the colours are its first ``n`` (so two
    groups get its two most distinct colours, not its first and last);
    beyond that tab20, then evenly spaced hues."""
    n = max(0, int(n))
    base = group_colors(name)
    if n <= len(base):
        return list(base[:n])
    import matplotlib
    from matplotlib.colors import to_hex
    cmap = matplotlib.colormaps['tab20' if n <= 20 else 'gist_ncar']
    if n <= 20:
        return [to_hex(cmap(i)) for i in range(n)]
    return [to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
