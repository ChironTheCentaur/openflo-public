"""ttk theming for the OpenFlo GUI: the 'sleek' chrome and ``apply_theme``.

``apply_theme`` styles ``clam`` to the active palette (from ``theme``) and
seeds the Tk option database so raw-tk widgets follow it. Moved here from
``gui.py``; the old ``gui`` names still resolve, with a DeprecationWarning.
"""
from __future__ import annotations

import tkinter as tk

from .theme import THEMES, ensure_fonts, set_active_palette

__all__ = ['apply_sleek', 'apply_theme', 'install_flat_indicators']


_FLAT_IND = {'made': set(), 'imgs': {}}


def install_flat_indicators(style, mode, pal):
    """Replace clam's jagged check/radio indicators with clean supersampled
    PIL shapes so the square checkbox and round radio share one crisp style:
    panel fill + muted outline, accent fill (radio dot / check tick) when on.
    Best-effort — on any failure the default indicators are left in place."""
    try:
        from PIL import Image, ImageDraw, ImageTk
    except ImportError:
        return

    def _rgb(h):
        h = h.lstrip('#')
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    panel, muted, accent, bg = (_rgb(pal['panel']) + (255,),
                                _rgb(pal['muted']) + (255,),
                                _rgb(pal['accent']) + (255,),
                                _rgb(pal['bg']) + (255,))
    scale, size = 4, 15
    D = size * scale
    pad = scale

    def _photo(img):
        return ImageTk.PhotoImage(img.resize((size, size),
                                             Image.Resampling.LANCZOS))

    def _radio(selected):
        img = Image.new('RGBA', (D, D), bg)
        d = ImageDraw.Draw(img)
        d.ellipse([pad, pad, D - pad, D - pad], fill=panel, outline=muted,
                  width=scale)
        if selected:
            m = D * 0.30
            d.ellipse([m, m, D - m, D - m], fill=accent)
        return _photo(img)

    def _check(selected):
        img = Image.new('RGBA', (D, D), bg)
        d = ImageDraw.Draw(img)
        r = D * 0.22
        d.rounded_rectangle([pad, pad, D - pad, D - pad], radius=r,
                            fill=panel, outline=muted, width=scale)
        if selected:
            d.line([(D * 0.28, D * 0.52), (D * 0.44, D * 0.68),
                    (D * 0.74, D * 0.32)], fill=accent,
                   width=int(scale * 1.6), joint='curve')
        return _photo(img)

    try:
        for kind, drawer, wclass, elabel in (
                ('flatradio', _radio, 'TRadiobutton', 'Radiobutton'),
                ('flatcheck', _check, 'TCheckbutton', 'Checkbutton')):
            name = f'{kind}_{mode}'
            if name not in _FLAT_IND['made']:
                off_i, on_i = drawer(False), drawer(True)
                _FLAT_IND['imgs'][name] = (off_i, on_i)   # keep refs alive
                style.element_create(name + '.indicator', 'image', off_i,
                                     ('selected', on_i), sticky='', border=0,
                                     padding=0)
                _FLAT_IND['made'].add(name)
            style.layout(wclass, [
                (f'{elabel}.padding', {'sticky': 'nswe', 'children': [
                    (name + '.indicator', {'side': 'left', 'sticky': ''}),
                    (f'{elabel}.label',
                     {'side': 'left', 'sticky': 'nswe'})]})])
    except Exception as exc:  # noqa: BLE001 - theming is best-effort; a failure is printed and the stock indicators stay
        print(f"[theme] flat indicators: {exc}", flush=True)


def apply_sleek(style, mode, pal, font, bold):
    """OpenFlo's 'sleek' chrome: the built-in ``clam`` theme hand-styled to a
    refined near-black / off-white palette — flat surfaces, hairline borders tied
    to the palette (clam's default near-white bevels are stark in dark mode),
    generous padding, muted flat headings/tabs, a cool accent on focus, and
    PIL-drawn flat check/radio indicators. Dependency-free (no third-party ttk
    theme)."""
    BG, PANEL, FG = pal['bg'], pal['panel'], pal['fg']
    ACC, ACCFG, BORDER = pal['accent'], pal['accfg'], pal['border']
    ACTIVE, MUTED = pal['active'], pal['muted']
    TROUGH, THUMB = pal['trough'], pal['thumb']
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass
    style.configure('.', background=BG, foreground=FG, font=font,
                    bordercolor=BORDER, lightcolor=BORDER,
                    darkcolor=BORDER, fieldbackground=PANEL)
    style.configure('TFrame', background=BG)
    style.configure('TPanedwindow', background=BG)
    style.configure('TLabel', background=BG, foreground=FG)
    style.configure('TLabelframe', background=BG, bordercolor=BORDER)
    style.configure('TLabelframe.Label', background=BG, foreground=MUTED,
                    font=bold)
    style.configure('TButton', padding=(14, 7),
                    background=PANEL, foreground=FG,
                    bordercolor=BORDER, lightcolor=BORDER,
                    darkcolor=BORDER, relief='flat', focuscolor=BORDER)
    # The side panel's buttons: tighter padding, and the widgets ask for their
    # text's natural width (width=0) instead of clam's 11-character minimum.
    # At the default 1500px window those minimums made the panel 747px wide,
    # half the window, and clipped the plot's control bar (893px needed, 741px
    # left). Inherits TButton's colours and hover map.
    # Labels stay short (beta tags live in the tips) so they don't widen the
    # three equal columns: a long one made the panel 507px instead of 372px.
    style.configure('Side.TButton', padding=(8, 5), justify='center')
    # A cool accent hairline appears on hover/focus (the sleek signature).
    style.map('TButton',
              background=[('active', ACTIVE), ('pressed', ACTIVE)],
              foreground=[('active', FG)],
              bordercolor=[('active', ACC), ('focus', ACC)],
              lightcolor=[('active', ACTIVE)],
              darkcolor=[('active', ACTIVE)])
    # Check/radio indicators: flat (no 3-D bevel ring), panel fill + accent when
    # selected — then swapped for the PIL-drawn shapes below.
    for _ind in ('TCheckbutton', 'TRadiobutton'):
        style.configure(_ind, background=BG, foreground=FG,
                        indicatorcolor=PANEL, indicatorrelief='flat',
                        indicatormargin=2, bordercolor=PANEL,
                        lightcolor=PANEL, darkcolor=PANEL)
        style.map(_ind,
                  background=[('active', BG)],
                  indicatorcolor=[('selected', ACC), ('pressed', ACC)],
                  bordercolor=[('', PANEL)],
                  lightcolor=[('', PANEL)], darkcolor=[('', PANEL)])
    install_flat_indicators(style, mode, pal)
    style.configure('TMenubutton', padding=(10, 6),
                    background=PANEL, foreground=FG)
    style.configure('TEntry', padding=5, fieldbackground=PANEL,
                    foreground=FG, insertcolor=FG, bordercolor=BORDER)
    style.configure('TCombobox', padding=5, fieldbackground=PANEL,
                    foreground=FG, bordercolor=BORDER, arrowcolor=MUTED)
    style.map('TCombobox', fieldbackground=[('readonly', PANEL)],
              foreground=[('readonly', FG)])
    style.configure('TNotebook', background=BG, bordercolor=BORDER)
    style.configure('TNotebook.Tab', padding=(16, 8),
                    background=BG, foreground=MUTED)
    style.map('TNotebook.Tab', background=[('selected', PANEL)],
              foreground=[('selected', FG)])
    style.configure('Treeview', background=PANEL, fieldbackground=PANEL,
                    foreground=FG, rowheight=26, bordercolor=BORDER)
    # Flat, muted headings that sit on the window bg (not a raised panel).
    style.configure('Treeview.Heading', font=bold, padding=(8, 6),
                    background=BG, foreground=MUTED, relief='flat')
    style.map('Treeview.Heading',
              background=[('active', ACTIVE), ('pressed', ACTIVE)],
              foreground=[('active', FG), ('pressed', FG)],
              relief=[('active', 'flat'), ('pressed', 'flat')])
    style.map('Treeview', background=[('selected', ACC)],
              foreground=[('selected', ACCFG)])
    style.configure('TSeparator', background=BORDER)
    # Scrollbars: a recessed trough + a slightly lighter, grabbable thumb.
    for _sb in ('Vertical.TScrollbar', 'Horizontal.TScrollbar'):
        style.configure(_sb, background=THUMB, troughcolor=TROUGH,
                        bordercolor=TROUGH, lightcolor=TROUGH,
                        darkcolor=TROUGH, arrowcolor=MUTED, relief='flat')
        style.map(_sb, background=[('active', BORDER)])
    style.configure('TProgressbar', background=ACC, troughcolor=TROUGH,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)


def apply_theme(root, mode='light'):
    """Apply the chosen chrome theme ('light'/'dark'/'midnight') across the whole
    app via the 'sleek' styling (``clam`` hand-styled to the palette;
    dependency-free). The Tk option database is also seeded so raw-tk widgets
    (menus, Text, Listbox, Canvas, Toplevels, and the Combobox drop-down list) in
    every window follow the palette. Matplotlib plots and explicit gate colours
    are untouched; every step is best-effort. Returns the active palette dict;
    re-callable to switch themes live."""
    pal = set_active_palette(THEMES.get(mode, THEMES['light']))
    BG, PANEL, FG = pal['bg'], pal['panel'], pal['fg']
    ACC, ACCFG = pal['accent'], pal['accfg']
    try:
        import tkinter.font as tkfont
        from tkinter import ttk
        style = ttk.Style(root)
        fam = 'Segoe UI' if 'Segoe UI' in set(tkfont.families(root)) else None
        if fam:                              # retune the named fonts in place
            for nm in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont',
                       'TkHeadingFont'):
                try:
                    tkfont.nametofont(nm).configure(family=fam, size=10)
                except tk.TclError:
                    pass
        font = (fam or 'TkDefaultFont', 10)
        bold = (fam or 'TkDefaultFont', 10, 'bold')
        apply_sleek(style, mode, pal, font, bold)
        # A single shared "muted / secondary text" label style for hints, status
        # lines and captions across every dialog. Defined once here (not a
        # hand-set foreground per widget) so it tracks the palette and recolours
        # live on a theme switch instead of freezing a hardcoded grey. Foreground
        # only; the background inherits the themed TLabel.
        try:
            style.configure('Muted.TLabel', foreground=pal['muted'])
            # Messages in the palette's own colours, never a hex per widget:
            # problems, cautions and success.
            style.configure('Error.TLabel', foreground=pal['error'])
            style.configure('Warning.TLabel', foreground=pal['warning'])
            style.configure('Success.TLabel', foreground=pal['success'])
        except tk.TclError:
            pass
        ensure_fonts(root)                   # the named fonts
        # tk (non-ttk) widgets read these from the option DB at creation time —
        # the ttk styling above only reaches ttk widgets, so raw tk (menus, Text,
        # Listbox, Canvas, Toplevels) is themed here. Applies to every dialog
        # opened afterwards.
        try:
            root.option_add('*tearOff', False)
            root.option_add('*Menu.background', PANEL)
            root.option_add('*Menu.foreground', FG)
            root.option_add('*Menu.activeBackground', ACC)
            root.option_add('*Menu.activeForeground', ACCFG)
            # Without this, disabled items (e.g. a greyed-out "Paste") fall back
            # to the OS engraved-grey style, which renders garbled on a dark menu.
            root.option_add('*Menu.disabledForeground', pal['muted'])
            # Flat dropdowns with no Tk-drawn border / raised highlight (any
            # remaining hairline is the OS popup frame, not Tk-controllable).
            root.option_add('*Menu.relief', 'flat')
            root.option_add('*Menu.borderWidth', 0)
            root.option_add('*Menu.activeBorderWidth', 0)
            # Dialog Toplevels grid ttk widgets directly on themselves, so the
            # Toplevel's own (otherwise default-light) bg shows through; tie it
            # to the theme. Applies to Toplevels created after this call.
            root.option_add('*Toplevel.background', BG)
            # tk (non-ttk) widgets default to a white background that ignores
            # the ttk theme — the source of the glaring white space in dialogs
            # under the dark themes. Theme them here so Canvas areas, Text
            # panes and Listboxes follow the palette. (Applies to widgets
            # created after this runs — i.e. every dialog opened afterwards.)
            root.option_add('*Canvas.background', BG)
            root.option_add('*Text.background', PANEL)
            root.option_add('*Text.foreground', FG)
            root.option_add('*Text.insertBackground', FG)
            root.option_add('*Listbox.background', PANEL)
            root.option_add('*Listbox.foreground', FG)
            root.option_add('*Listbox.selectBackground', ACC)
            root.option_add('*Listbox.selectForeground', ACCFG)
            # The ttk.Combobox drop-down list is a SEPARATE popdown Listbox that
            # neither ttk.Style nor the generic *Listbox reliably reaches; target
            # it explicitly so the dropdown matches the palette in dark themes.
            root.option_add('*TCombobox*Listbox.background', PANEL)
            root.option_add('*TCombobox*Listbox.foreground', FG)
            root.option_add('*TCombobox*Listbox.selectBackground', ACC)
            root.option_add('*TCombobox*Listbox.selectForeground', ACCFG)
        except tk.TclError:
            pass
    except Exception as exc:  # noqa: BLE001 - theming is best-effort; a failure is printed and the rest of the styling is skipped
        print(f"[theme] {exc}", flush=True)
    return pal
