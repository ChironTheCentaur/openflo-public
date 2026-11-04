"""Colours come from the theme: error, warning, highlight and log colours
per theme, each readable (4.5:1) on the window and panel colours. Theme
is Light or Dark windows plus a White or Black plot background; pop-up
figures follow the plot background.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import ast
import tkinter as tk
from tkinter import ttk

import tests._sample_editor as shared
from tests._editor_ui import SRC, _entries, _menus

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def _luminance(hex_color):
    h = hex_color.lstrip('#')
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
           for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_contrast_is_computed_as_wcag_does():
    assert round(contrast('#000000', '#ffffff'), 1) == 21.0
    assert round(contrast('#777777', '#ffffff'), 2) == 4.48


def test_every_text_colour_reads_on_its_background():
    from openflo.theme import THEMES
    low = []
    for name, pal in THEMES.items():
        for text in ('fg', 'muted', 'error', 'warning', 'success'):
            for ground in ('bg', 'panel'):
                c = contrast(pal[text], pal[ground])
                if c < 4.5:
                    low.append((name, text, ground, round(c, 2)))
        for text, ground in (('highlight_fg', 'highlight'),
                             ('console_fg', 'console_bg'),
                             ('accfg', 'accent'),
                             ('plot_fg', 'plot_bg')):
            c = contrast(pal[text], pal[ground])
            if c < 4.5:
                low.append((name, text, ground, round(c, 2)))
    assert not low, low


_COLOUR_KW = {'foreground', 'background', 'fg', 'bg', 'insertbackground',
              'selectbackground', 'selectforeground', 'activebackground',
              'activeforeground', 'disabledforeground', 'highlightcolor',
              'highlightbackground', 'troughcolor'}


# The layout editor draws the printed page, which is white in every theme
# (its desk is set from the palette once drawn); its colours dialog numbers
# each colour swatch in white.
_PAGE_DRAWING = {'ui_layout_editor.py', 'ui_layout_dialogs.py'}


def _hard_coded_colours(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            v = kw.value
            if (kw.arg in _COLOUR_KW and isinstance(v, ast.Constant)
                    and isinstance(v.value, str) and v.value
                    and v.value != 'White'):     # savefig_background's name
                yield node.lineno, kw.arg, v.value


def test_no_widget_colour_is_hard_coded():
    found = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'theme.py' or path.name in _PAGE_DRAWING:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found += [f"{path.name}:{ln} {k}={v}"
                  for ln, k, v in _hard_coded_colours(tree)]
    assert not found, found


def test_the_check_finds_a_hard_coded_colour():
    tree = ast.parse("ttk.Label(f, foreground='red')\n"
                     "tk.Text(f, background='#1e1e1e')")
    assert len(list(_hard_coded_colours(tree))) == 2


def test_log_highlight_and_messages_follow_the_theme(editor):
    from openflo.theme import THEMES
    root, ed = editor
    log = ed._log_text
    assert str(log.cget('background')) == THEMES['light']['console_bg']
    assert str(ed.gate_tv.tag_configure('drop_target', 'background')) == \
        THEMES['light']['highlight']
    style = ttk.Style(root)
    assert str(style.lookup('Error.TLabel', 'foreground')) == \
        THEMES['light']['error']
    ed._theme_var.set('midnight')
    ed._set_theme()
    root.update()
    assert str(log.cget('background')) == THEMES['midnight']['console_bg']
    assert str(log.cget('foreground')) == THEMES['midnight']['console_fg']
    assert str(ed.gate_tv.tag_configure('drop_target', 'background')) == \
        THEMES['midnight']['highlight']
    assert str(style.lookup('Error.TLabel', 'foreground')) == \
        THEMES['midnight']['error']
    ed._theme_var.set('light')
    ed._set_theme()


def test_theme_is_window_colour_plus_plot_background(editor):
    from openflo.theme import (
        THEMES,
        current_palette,
        dialog_dark_on,
        theme_key,
        theme_parts,
    )
    root, ed = editor
    for key in ('light', 'dark', 'midnight', 'light-dark'):
        assert theme_key(*theme_parts(key)) == key
    assert theme_parts('no-such-theme') == ('light', 'light')
    menus = _menus(ed)
    view = [label for _t, label, _a in _entries(menus['View'])]
    assert "Theme" in view and "Plot background" in view
    assert not [label for label in view if 'Dark figures' in label]
    ed._chrome_var.set('light')
    ed._plot_bg_var.set('dark')
    ed._set_theme_parts()
    root.update()
    assert ed._theme_var.get() == 'light-dark'
    pal = current_palette()
    assert pal['bg'] == THEMES['light']['bg']
    assert pal['plot_bg'] == THEMES['midnight']['plot_bg']
    assert dialog_dark_on(tk.Toplevel(ed))       # pop-ups follow the plot
    assert ed.status_var.get() == \
        "Light windows with a black plot background."
    ed._theme_var.set('dark')                     # e.g. from Preferences
    ed._set_theme()
    root.update()
    assert (ed._chrome_var.get(), ed._plot_bg_var.get()) == ('dark', 'light')
    assert not dialog_dark_on(tk.Toplevel(ed))
    ed._theme_var.set('light')
    ed._set_theme()
