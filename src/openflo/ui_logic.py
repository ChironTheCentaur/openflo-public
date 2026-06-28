"""Pure UI decision logic — no Tkinter, no matplotlib.

The first step of decomposing the ``gui.py`` monolith: lift small, pure pieces
out of widget callbacks so they can be unit-tested without a display (and so CI
exercises them headlessly). Behaviour here must match what the GUI did inline.
Grow this module as more logic is extracted; keep it import-light and Tk-free.
"""
from __future__ import annotations

import re
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


def should_use_dark(dark_toggle: bool, theme: str | None) -> bool:
    """Whether pop-up figures should render dark: the 'Dark figures in pop-ups'
    toggle is on, OR the active theme is the dark-plot 'midnight' theme (so its
    pop-ups match the main canvas)."""
    return bool(dark_toggle) or (str(theme) == 'midnight')
