"""Hover tooltips for the secondary windows.

The editor gained tooltips early and the dialogs never did: an audit of the 30
``ui_*`` modules found **zero** tooltip calls against ~165 interactive widgets,
so Preferences, Compensation, Frequencies and the rest offered no hover help at
all. That is the wrong way round -- the main window's controls are the ones a
user sees constantly and learns by repetition, while a dialog's options are the
ones met once a month and forgotten in between.

The tooltip machinery itself already exists and is fine (``gui._ToolTip``,
attached via the editor's gated ``_tip``). This module is just the shared entry
point so a dialog can reach it without importing ``gui`` -- which would be a
circular import, since ``gui`` builds the dialogs.

Usage inside any dialog that keeps ``self._editor``::

    from .ui_tips import tip
    tip(self._editor, widget, "What this control does, and when it matters.")

Gating comes free: ``editor._tip`` respects **View -> Show hover tips**, so one
toggle silences every window.

Writing a good tooltip: say what the control DOES and what changes if you use
it. Do not restate the label -- "Seed: the seed" helps nobody. Mention the unit,
the default, or the consequence where there is one.
"""
from __future__ import annotations


def tip(editor, widget, text) -> None:
    """Attach a hover tooltip to ``widget``, gated by the editor's View toggle.

    Best-effort and silent: a dialog opened without an editor (tests, a
    stand-alone harness) simply gets no tooltip rather than an error. ``text``
    may be a callable, resolved at hover time, for a message that depends on
    current state.
    """
    if editor is None:
        return
    attach = getattr(editor, '_tip', None)
    if attach is None:
        return
    try:
        attach(widget, text)
    except Exception:                                            # noqa: BLE001
        pass


def tips(editor, pairs) -> None:
    """Attach several at once: ``tips(ed, [(widget, "text"), ...])``."""
    for widget, text in pairs:
        tip(editor, widget, text)


def find_editor(widget):
    """Walk up from ``widget`` to the editor that owns it, or None.

    Shared helpers in ``tool_window`` build widgets for a dozen dialogs and
    only ever receive ``parent``. Rather than thread ``editor`` through every
    call site, resolve it: dialogs keep the editor as ``.editor`` or
    ``._editor``, and the editor itself is the object carrying ``_tip``.
    """
    seen = 0
    while widget is not None and seen < 40:
        if hasattr(widget, '_tip'):
            return widget
        for attr in ('editor', '_editor'):
            cand = getattr(widget, attr, None)
            if cand is not None and hasattr(cand, '_tip'):
                return cand
        widget = getattr(widget, 'master', None)
        seen += 1
    return None


def auto_tip(widget, text) -> None:
    """``tip`` without an explicit editor -- resolves it from the hierarchy.

    Use in shared widget factories. Silent when no editor is found, which is
    the right behaviour for a dialog opened stand-alone in a test.
    """
    tip(find_editor(widget), widget, text)
