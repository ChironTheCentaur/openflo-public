"""Theme keys are named by what they hold (best-practices finding N5).

The saved keys predate the window/plot split, so 'dark' has a white plot and
'light-dark' has a dark plot. Code reads them through named constants and two
helpers, one per kind of darkness; the saved strings themselves are unchanged.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from openflo import ui_logic
from openflo.ui_logic import (
    DARK_PLOT_THEMES,
    THEME_ALL_DARK,
    THEME_DARK_PLOT,
    THEME_DARK_WINDOWS,
    THEME_LIGHT,
    plot_is_dark,
    windows_are_dark,
)

SRC = pathlib.Path(ui_logic.__file__).parent


def test_saved_keys_are_unchanged():
    # Preferences and sessions hold these strings, so they must not move.
    assert (THEME_LIGHT, THEME_DARK_WINDOWS, THEME_ALL_DARK, THEME_DARK_PLOT) \
        == ('light', 'dark', 'midnight', 'light-dark')


@pytest.mark.parametrize('key, windows, plot', [
    ('light', False, False),
    ('dark', True, False),          # dark windows, white plot
    ('midnight', True, True),
    ('light-dark', False, True),    # light windows, dark plot
    (None, False, False),
    ('no-such-theme', False, False),
])
def test_each_kind_of_darkness(key, windows, plot):
    assert windows_are_dark(key) is windows
    assert plot_is_dark(key) is plot


def test_palettes_match_the_names():
    theme = pytest.importorskip('openflo.theme')
    light_bg = theme.THEMES[THEME_LIGHT]['bg']
    white_plot = theme.THEMES[THEME_LIGHT]['plot_bg']
    for key, pal in theme.THEMES.items():
        assert (pal['bg'] != light_bg) is windows_are_dark(key), key
        assert (pal['plot_bg'] != white_plot) is plot_is_dark(key), key
    # theme_key builds each saved key from its (window, plot) parts.
    assert theme.theme_key('dark', 'light') == THEME_DARK_WINDOWS
    assert theme.theme_key('light', 'dark') == THEME_DARK_PLOT


def test_should_use_dark_is_a_deprecated_alias():
    for key in ('light', 'dark', 'midnight', 'light-dark', None):
        with pytest.warns(DeprecationWarning, match='plot_is_dark'):
            assert ui_logic.should_use_dark(key) is plot_is_dark(key)


def test_plot_dark_themes_still_served():
    theme = pytest.importorskip('openflo.theme')
    with pytest.warns(DeprecationWarning, match='PLOT_DARK_THEMES'):
        assert theme.PLOT_DARK_THEMES == set(DARK_PLOT_THEMES)


def _dark_key_sets(tree):
    """Set, tuple or list literals that hold two or more theme key strings."""
    keys = {'dark', 'midnight', 'light-dark'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
            vals = {e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if len(vals & keys) >= 2:
                yield node.lineno


def test_dark_key_sets_are_defined_once():
    # Only ui_logic spells out which keys are dark; everyone else asks it.
    found = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'ui_logic.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found += [f'{path.name}:{ln}' for ln in _dark_key_sets(tree)]
    assert found == []
