"""Helpers other modules use have public names; the old private names still
work, with a DeprecationWarning (best-practices finding N4)."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'

# defining module -> {old private name: new public name}
RENAMED = {
    'theme': {'_dialog_dark_on': 'dialog_dark_on',
              '_theme_figure_dark': 'theme_figure_dark'},
    'ui_figure_window': {'_FigureWindow': 'FigureWindow'},
    'editor_session': {'_frame_digest': 'frame_digest'},
    'pipeline': {'_WSP_BOOLEAN_OPS': 'WSP_BOOLEAN_OPS',
                 '_scale_time_coords': 'scale_time_coords',
                 '_wsp_axis_floor_lo': 'wsp_axis_floor_lo',
                 '_wsp_axis_floors': 'wsp_axis_floors',
                 '_wsp_comp_channels': 'wsp_comp_channels',
                 '_wsp_ellipse_gate': 'wsp_ellipse_gate',
                 '_wsp_polygon_edge_axes': 'wsp_polygon_edge_axes'},
}
# These modules import tkinter at import time.
NEEDS_TK = {'theme', 'ui_figure_window', 'editor_session'}


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Dialog:
    """A stand-in dialog whose owner carries the theme variable."""

    def __init__(self, theme):
        self.master = type('Owner', (), {'_theme_var': _Var(theme),
                                         'master': None})()


def _figure():
    from matplotlib.figure import Figure
    fig = Figure()
    fig.add_subplot(1, 1, 1)
    return fig


@pytest.mark.parametrize('theme_name', ['midnight', 'light-dark'])
def test_match_dialog_recolours_for_a_dark_plot_background(theme_name):
    pytest.importorskip('tkinter')
    from matplotlib.colors import to_hex

    from openflo.theme import THEMES, match_dialog
    fig = _figure()
    assert match_dialog(fig, _Dialog(theme_name)) is True
    dark_bg = THEMES['midnight']['plot_bg']
    assert to_hex(fig.get_facecolor()) == to_hex(dark_bg)
    assert to_hex(fig.axes[0].get_facecolor()) == to_hex(dark_bg)


@pytest.mark.parametrize('theme_name', ['light', 'dark'])
def test_match_dialog_leaves_a_light_plot_background(theme_name):
    pytest.importorskip('tkinter')
    from openflo.theme import match_dialog
    fig = _figure()
    before = tuple(fig.get_facecolor())
    assert match_dialog(fig, _Dialog(theme_name)) is False
    assert tuple(fig.get_facecolor()) == before


@pytest.mark.parametrize('module, old, new', [
    (mod, old, new) for mod, names in RENAMED.items()
    for old, new in names.items()])
def test_old_private_name_is_a_deprecated_alias(module, old, new):
    if module in NEEDS_TK:
        pytest.importorskip('tkinter')
    mod = importlib.import_module(f'openflo.{module}')
    with pytest.warns(DeprecationWarning, match=new):
        alias = getattr(mod, old)
    # A new name that has since moved on is served by its new home.
    home, attr = getattr(mod, '_MOVED', {}).get(new, (module, new))
    assert alias is getattr(importlib.import_module(f'openflo.{home}'), attr)


def test_no_other_module_uses_the_old_private_names():
    owner = {old: mod for mod, names in RENAMED.items() for old in names}
    found = []
    for path in sorted(SRC.rglob('*.py')):
        here = path.stem
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in owner and owner[alias.name] != here:
                        found.append(f'{path.name}:{node.lineno} imports '
                                     f'{alias.name}')
            elif (isinstance(node, ast.Attribute) and node.attr in owner
                  and owner[node.attr] != here):
                found.append(f'{path.name}:{node.lineno} uses .{node.attr}')
    assert found == []
