"""Theme palette, dark-figure helpers, and publication figure export.

Shared appearance/rendering helpers, kept out of gui.py so the ~24 dialog
modules import this small module instead of the whole GUI. The heavy
``apply_theme`` / flat-indicator styling stays in gui.py (it's editor-only) and
pushes the active palette here via ``set_active_palette``.
"""
from __future__ import annotations

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
    """True if pop-up figures should render dark: either the 'Dark figures in
    pop-ups' toggle is on, OR the active theme is Midnight (a dark-plot theme,
    so its pop-ups should match). Walks up from a dialog widget to the editor
    that owns the ``_dark_figs`` / ``_theme_var`` vars."""
    w = widget
    for _ in range(8):
        v = getattr(w, '_dark_figs', None)
        tv = getattr(w, '_theme_var', None)
        if v is not None or tv is not None:
            try:
                dark = bool(v.get()) if v is not None else False
                theme = tv.get() if tv is not None else None
                return should_use_dark(dark, theme)
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


def savefig_background(fig, path, background='White', dpi=300):
    """Save ``fig`` with a publication-export background:

      • ``White``       — opaque white (default)
      • ``Transparent`` — full alpha (sits on a coloured page / poster / slide)
      • ``Translucent`` — 50% white wash (figure AND per-axes patches)

    The per-axes patch alpha is changed only for the duration of the save and
    restored afterwards, so an on-screen preview of ``fig`` is unaffected.
    PNG / PDF / SVG carry the alpha; TIFF may flatten it."""
    kw = {}
    axes_alpha = None
    if background == 'Transparent':
        kw['transparent'] = True
    elif background == 'Translucent':
        kw['facecolor'] = (1.0, 1.0, 1.0, 0.5)
        axes_alpha = 0.5
    elif background == 'Dark':
        # Keep the figure's own (dark) facecolor — used when the preview is
        # already dark (View → Dark figures), so the export matches it.
        kw['facecolor'] = fig.get_facecolor()
        kw['edgecolor'] = 'none'
        _drop = _provenance_footer(fig)
        try:
            fig.savefig(path, dpi=dpi, bbox_inches='tight', **kw)
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
        fig.savefig(path, dpi=dpi, bbox_inches='tight', edgecolor='none', **kw)
    finally:
        _drop()
        for ax, fc in restore:
            ax.patch.set_facecolor(fc)


_PLOT_LIGHT = dict(plot_bg='#ffffff', plot_fg='#20242b',
                   plot_grid='#e6e8ec', plot_spine='#b9bdc6')


THEMES = {
    'light': dict(bg='#f3f4f6', panel='#ffffff', fg='#20242b',
                  accent='#2563eb', accfg='#ffffff', border='#d4d7dd',
                  muted='#5b616b', active='#e8eaee',
                  trough='#cdd0d6', thumb='#aeb4bd',     # ~15-20% darker base
                  **_PLOT_LIGHT),
    'dark':  dict(bg='#1e2127', panel='#262a31', fg='#d7dbe0',
                  accent='#4f8cff', accfg='#0d1117', border='#363b44',
                  muted='#9aa0a8', active='#30353d',
                  trough='#171a1f', thumb='#3c424c',     # ~20% darker base
                  **_PLOT_LIGHT),
    # Dark chrome (identical to 'dark') + a dark plot canvas.
    'midnight': dict(bg='#1e2127', panel='#262a31', fg='#d7dbe0',
                     accent='#4f8cff', accfg='#0d1117', border='#363b44',
                     muted='#9aa0a8', active='#30353d',
                     trough='#171a1f', thumb='#3c424c',
                     plot_bg='#14171c', plot_fg='#d7dbe0',
                     plot_grid='#2c313a', plot_spine='#454b55'),
}

# Themes whose CHROME (window, panels, title bar) is dark.


_DARK_MODES = {'dark', 'midnight'}


_ACTIVE_PALETTE = THEMES['light']
# Custom flat check/radio indicator elements (PIL-drawn, anti-aliased) keyed by
# theme; created once per theme and kept alive so Tk doesn't GC the images.


def current_palette():
    """The palette of the theme currently applied (chrome colours)."""
    return _ACTIVE_PALETTE


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
