"""Pure UI decision logic — no Tkinter, no matplotlib.

The first step of decomposing the ``gui.py`` monolith: lift small, pure pieces
out of widget callbacks so they can be unit-tested without a display (and so CI
exercises them headlessly). Behaviour here must match what the GUI did inline.
Grow this module as more logic is extracted; keep it import-light and Tk-free.
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping


def format_channel(det: str, labels: Mapping[str, str] | None = None) -> str:
    """``'Label (DET)'`` when a distinct antibody label exists for the
    detector, else just ``'DET'`` — how channels are shown in the axis pickers
    and tree."""
    lbl = (labels or {}).get(det, det)
    return f'{lbl} ({det})' if lbl and lbl != det else det


def resolve_channel(display: str):
    """Recover the raw detector from a ``'Label (DET)'`` display string (or
    return it unchanged when there's no parenthesised detector). Inverse of
    :func:`format_channel`."""
    if not display:
        return None
    m = re.match(r'.*\(([^)]+)\)\s*$', display)
    return m.group(1) if m else display


def short_label(name, width: int = 24) -> str:
    """Trim a name for a panel title: ``'…'``-elided when longer than
    ``width``."""
    name = str(name)
    return name if len(name) <= width else name[:width - 1] + '…'


def filter_choices(typed: str, choices: Iterable[str]) -> list[str]:
    """Choices containing ``typed`` (case-insensitive). Empty query → all;
    no match → all (so the dropdown never goes empty). Backs the type-to-filter
    channel pickers."""
    items = list(choices)
    if not typed:
        return items
    low = typed.lower()
    return [v for v in items if low in v.lower()] or items


def resolve_choice(typed: str, choices: Iterable[str], fallback: str = '') -> str:
    """Snap ``typed`` to an exact (case-insensitive) match, else the first
    substring match, else ``fallback`` — so an invalid channel can never be
    committed to an axis."""
    items = list(choices)
    if not typed:
        return fallback
    low = typed.lower()
    exact = next((v for v in items if v.lower() == low), None)
    if exact is not None:
        return exact
    return next((v for v in items if low in v.lower()), fallback)


def has_real_gates(gates: Mapping[str, Mapping]) -> bool:
    """True if any gate is a positive population. Auto-clean gates are negative
    selections (events to drop), not populations to highlight or filter to, so
    they don't count — drives whether Highlight / Filter display modes apply."""
    return any(g.get('kind') != 'autoclean' for g in gates.values())


# Theme keys, as saved in preferences and sessions. A theme is a window
# colour (chrome) and a plot background, each light or dark. The saved strings
# predate that split, so two of them read wrong: 'dark' has dark windows but a
# WHITE plot, and 'light-dark' has light windows with a dark plot. Code names
# them by these constants; the strings themselves must stay as saved.
THEME_LIGHT = 'light'                 # light windows, white plot
THEME_DARK_WINDOWS = 'dark'           # dark windows, white plot
THEME_ALL_DARK = 'midnight'           # dark windows, dark plot
THEME_DARK_PLOT = 'light-dark'        # light windows, dark plot

# Themes whose plot background (and pop-up figures) is dark.
DARK_PLOT_THEMES = frozenset({THEME_ALL_DARK, THEME_DARK_PLOT})
# Themes whose window chrome (windows, panels, title bar) is dark.
DARK_WINDOW_THEMES = frozenset({THEME_DARK_WINDOWS, THEME_ALL_DARK})


def plot_is_dark(theme: str | None) -> bool:
    """Whether the theme's plot background is dark, so pop-up figures render
    dark to match the main plot."""
    return str(theme) in DARK_PLOT_THEMES


def windows_are_dark(theme: str | None) -> bool:
    """Whether the theme's window chrome is dark (title bars follow it)."""
    return str(theme) in DARK_WINDOW_THEMES


def should_use_dark(theme: str | None) -> bool:
    """Deprecated name of :func:`plot_is_dark`; will be removed in a future
    release."""
    warnings.warn('should_use_dark is deprecated and will be removed in a '
                  'future release; use plot_is_dark', DeprecationWarning,
                  stacklevel=2)
    return plot_is_dark(theme)


def accel(keys: str, platform: str | None = None) -> str:
    """A shortcut as this system writes it: Ctrl is Command on macOS, where
    the same keys are bound to Command too (LifecycleMixin._bind_shortcuts)."""
    import sys
    plat = sys.platform if platform is None else platform
    return keys.replace("Ctrl+", "Cmd+") if plat == 'darwin' else keys
