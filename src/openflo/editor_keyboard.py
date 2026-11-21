"""Keyboard use of the plot: move a gate with the arrow keys.

Gating is otherwise mouse-only, which shuts out anyone who cannot drag
precisely (a trackpad, a tremor, a screen reader user working by keys). The
plot takes keyboard focus (click it, or F6 from the gate tree), and the arrow
keys then move the gate selected in the tree: a small step, or a large one
with Shift. One run of key presses on one gate is one undo step.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import time
import tkinter as tk

from .editor_base import EditorMixin
from .gate_move import axis_move_fns, display_delta, move_gate
from .gate_store import ONE_D_KINDS
from .ui_logic import resolve_channel

# Step per arrow press, as a fraction of the visible axis span.
NUDGE_STEP = 0.005
NUDGE_STEP_LARGE = 0.05
# Presses on the same gate closer together than this share one undo step.
_NUDGE_GROUP_S = 1.0
# The tree, the counts and a gated view update once the keys pause.
_NUDGE_SETTLE_MS = 350

_ARROWS = {'Left': (-1, 0), 'Right': (1, 0), 'Up': (0, 1), 'Down': (0, -1)}


def nudge_gate(g, dx, dy, x_channel, y_channel, x_fns=None, y_fns=None):
    """Move gate ``g`` (a gate dict, changed in place) by ``dx`` along the
    plotted x channel and ``dy`` along y, in the axes' display units
    (``x_fns`` / ``y_fns`` the axes, gate_move; None for a linear one), so
    a gate on a log or logicle axis keeps its look, as a drag moves it.
    Returns True if the gate moved. Only the parts of the gate drawn on
    these axes move: a threshold on the y channel ignores ``dx``."""
    kind = g.get('kind')
    if kind in ('rect', 'polygon', 'ellipsoid'):
        if (g.get('x_channel'), g.get('y_channel')) != (x_channel, y_channel):
            return False
        moved = move_gate(g, dx, dy, x_fns, y_fns) if (dx or dy) else None
    elif kind in ONE_D_KINDS:
        ch = g.get('channel')
        step, fns = ((dx, x_fns) if ch == x_channel
                     else (dy, y_fns) if ch == y_channel else (0.0, None))
        moved = move_gate(g, step, 0.0, fns) if step else None
    else:
        return False
    if moved is None or all(g.get(k) == v for k, v in moved.items()):
        # Nothing to move (a gate wholly below a log view's floor stays):
        # no undo step for it.
        return False
    g.update(moved)
    return True


class KeyboardMixin(EditorMixin):
    """Arrow-key moves for the selected gate, and focus between tree and
    plot."""

    def _bind_plot_keys(self):
        """Let the plot take keyboard focus and answer the arrow keys. A
        focus ring shows when it has focus, so a keyboard user can see
        where the keys go."""
        from .theme import current_palette
        w = self.canvas.get_tk_widget()
        pal = current_palette()
        try:
            w.configure(takefocus=1, highlightthickness=2,
                        highlightbackground=pal['bg'],
                        highlightcolor=pal.get('accent', pal['fg']))
        except tk.TclError:
            pass
        for key in _ARROWS:
            w.bind(f'<{key}>', lambda e, k=key: self._on_arrow_key(k, False))
            w.bind(f'<Shift-{key}>',
                   lambda e, k=key: self._on_arrow_key(k, True))
        w.bind('<F6>', lambda _e: self._focus_gate_tree())
        if hasattr(self, 'gate_tv'):
            self.gate_tv.bind('<F6>', lambda _e: self._focus_plot(), add='+')

    def _focus_plot(self):
        try:
            self.canvas.get_tk_widget().focus_set()
        except tk.TclError:
            return None
        self.status_var.set(
            "Plot has keyboard focus: arrow keys move the selected gate "
            "(Shift for larger steps); F6 goes back to the gate list.")
        return 'break'

    def _focus_gate_tree(self):
        if hasattr(self, 'gate_tv'):
            self.gate_tv.focus_set()
        return 'break'

    def _on_arrow_key(self, key, large):
        sx, sy = _ARROWS[key]
        step = NUDGE_STEP_LARGE if large else NUDGE_STEP
        self._nudge_selected_gate(sx * step, sy * step)
        return 'break'

    def _nudge_selected_gate(self, fx, fy):
        """Move the gate selected in the tree by ``fx`` / ``fy`` of the
        visible x / y span. Returns True if it moved."""
        gid = self._selected_gate_id()
        g = self._gates.get(gid) if gid else None
        if g is None:
            self.status_var.set(
                "Select a gate of this sample in the gate list to move it "
                "with the arrow keys.")
            return False
        x = resolve_channel(self.x_combo.get())
        y = resolve_channel(self.y_combo.get())
        # A step is a share of the axis as SHOWN: on a log axis the same
        # distance on screen at any height, as a drag moves the gate.
        xf = axis_move_fns(self.ax.xaxis)
        yf = axis_move_fns(self.ax.yaxis)
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        dx = fx * display_delta(xf, xl[0], xl[1])
        dy = fy * display_delta(yf, yl[0], yl[1])
        trial = dict(g)
        if not nudge_gate(trial, dx, dy, x, y, xf, yf):
            self.status_var.set(
                f"'{g.get('name', gid)}' is not drawn on these axes, so the "
                "arrow keys cannot move it here.")
            return False
        now = time.monotonic()
        last = getattr(self, '_nudge_last', None)
        if not last or last[0] != gid or now - last[1] > _NUDGE_GROUP_S:
            self._checkpoint()
        self._nudge_last = (gid, now)
        g.update(trial)
        self._redraw_only_gates()
        pending = getattr(self, '_nudge_after', None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
        self._nudge_after = self.after(_NUDGE_SETTLE_MS, self._nudge_settled)
        return True

    def _nudge_settled(self):
        """The keys paused: refresh the tree text and count again, as the
        end of a mouse drag does."""
        self._nudge_after = None
        self._refresh_gate_list()
        self._recount_after_drag()
