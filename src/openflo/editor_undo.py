"""Undo / redo via gate-state checkpoints.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin


class UndoMixin(EditorMixin):
    """Checkpoint / undo / redo of editor gate state."""

    def _checkpoint(self):
        """Record a pre-mutation undo checkpoint. Call BEFORE mutating gate
        state. No-op while suspended (bulk loads) or when one was already
        taken this Tk event (coalesces a multi-step gesture into one undo)."""
        if self._suspend_undo or self._undo_pending:
            return
        self._undo_pending = True
        self._undo_stack.append(self._gate_state_snapshot())
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        try:
            self.after_idle(self._clear_undo_pending)
        except Exception:
            self._undo_pending = False

    def _undo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._undo_stack:
            self.status_var.set("Nothing to undo.")
            return
        self._redo_stack.append(self._gate_state_snapshot())
        self._restore_gate_state(self._undo_stack.pop())
        self.status_var.set(
            f"Undo. ({len(self._undo_stack)} more, {len(self._redo_stack)} redo)")

    def _redo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._redo_stack:
            self.status_var.set("Nothing to redo.")
            return
        self._undo_stack.append(self._gate_state_snapshot())
        self._restore_gate_state(self._redo_stack.pop())
        self.status_var.set(
            f"Redo. ({len(self._redo_stack)} more)")

    def _gate_state_snapshot(self):
        import copy
        seq = dict(self._sample_gate_seq)
        active = self._active_sample
        if active is not None:
            seq[active] = max(seq.get(active, 0), self._gate_id_seq)
        ws = getattr(self, '_workspace_panel', None)
        return {
            'gates':          copy.deepcopy(self._sample_gates),
            'order':          copy.deepcopy(self._sample_gate_order),
            'seq':            seq,
            'cluster_labels': copy.deepcopy(self._cluster_labels),
            'quad_seq':       getattr(self, '_quad_set_seq', 0),
            'workspace':      ws.model.to_dict() if ws is not None else None,
        }

    def _restore_gate_state(self, snap):
        import copy
        self._sample_gates      = copy.deepcopy(snap['gates'])
        self._sample_gate_order = copy.deepcopy(snap['order'])
        self._sample_gate_seq   = dict(snap['seq'])
        self._cluster_labels    = copy.deepcopy(snap['cluster_labels'])
        self._quad_set_seq      = snap.get('quad_seq', getattr(
            self, '_quad_set_seq', 0))
        # Rebind the active-sample shortcuts to the restored containers.
        active = self._active_sample
        if active in self._sample_gates:
            self._gates         = self._sample_gates[active]
            self._gate_id_order = self._sample_gate_order.setdefault(active, [])
            self._gate_id_seq   = self._sample_gate_seq.get(active, 0)
        else:
            self._gates = {}
            self._gate_id_order = []
            self._gate_id_seq = 0
        self._refresh_gate_list()
        self._schedule_replot(0)
        # Widen undo to the Pipeline Workspace: restore its model too (same
        # Undo button reverts workspace add/remove/group/comp/fmo/clear).
        ws = getattr(self, '_workspace_panel', None)
        ws_snap = snap.get('workspace')
        if ws is not None and ws_snap is not None:
            try:
                ws.restore_model(ws_snap)
            except Exception:
                pass

    def _clear_undo_pending(self):
        self._undo_pending = False
