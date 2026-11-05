"""Undo / redo via gate-state checkpoints.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin
from .errors import note


class UndoMixin(EditorMixin):
    """Checkpoint / undo / redo of editor gate state."""

    def _checkpoint(self):
        """Record a pre-mutation undo checkpoint. Call BEFORE mutating gate
        state. No-op while suspended (bulk loads) or when one was already
        taken this Tk event (coalesces a multi-step gesture into one undo)."""
        if self._suspend_undo or self._undo_pending:
            return
        self._undo_pending = True
        self._note_unsaved()
        self._undo_stack.append(self._gate_state_snapshot())
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        try:
            self.after_idle(self._clear_undo_pending)
        except Exception:  # noqa: BLE001 - undo coalescing flag only
            self._undo_pending = False

    def _undo_barrier(self, what):
        """Mark a change Undo cannot reverse (`what`: "compensation of 'x'",
        "the transform of FL1-A"). The first Undo after it says it cannot,
        and the next one goes on to the gate edits before. Stepping silently
        past it would reverse the gate edit made BEFORE it, which reads as
        undoing the change itself."""
        if self._suspend_undo:
            return
        self._note_unsaved()
        self._undo_stack.append({'barrier': str(what)})
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _forget_undo_samples(self, names):
        """Drop removed samples from every undo / redo checkpoint, so Undo
        cannot bring back the gates of a sample that is gone."""
        names = set(names)
        for snap in list(self._undo_stack) + list(self._redo_stack):
            if 'barrier' in snap:
                continue
            for key in ('gates', 'order', 'seq', 'cluster_labels'):
                for n in names:
                    (snap.get(key) or {}).pop(n, None)

    def _to_checkpoint_tab(self, stack, verb):
        """Show the tray tab the checkpoint on top of ``stack`` was taken in
        (its gates are that tab's own). False, with the checkpoint dropped
        and said, when that tab has been deleted."""
        tab = stack[-1].get('tab')
        ts = getattr(self, '_view_tabs', None)
        if tab is None or ts is None or tab is ts.active:
            return True
        idx = next((i for i, t in enumerate(ts.tabs) if t is tab), None)
        if idx is None:
            stack.pop()
            self.status_var.set(f"Cannot {verb}: that change was made in a "
                                "tab that has been deleted.")
            return False
        self._switch_tab(idx)
        return True

    def _undo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._undo_stack:
            self.status_var.set("Nothing to undo.")
            return
        if 'barrier' in self._undo_stack[-1]:
            what = self._undo_stack.pop()['barrier']
            more = ("Undo again to undo the gate change before it."
                    if self._undo_stack else "Nothing before it to undo.")
            self.status_var.set(f"Undo cannot reverse {what}. {more}")
            return
        if not self._to_checkpoint_tab(self._undo_stack, "undo"):
            return
        self._redo_stack.append(self._gate_state_snapshot())
        self._restore_gate_state(self._undo_stack.pop())
        self._note_unsaved()
        self.status_var.set(
            f"Undo. ({len(self._undo_stack)} more, {len(self._redo_stack)} redo)")

    def _redo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._redo_stack:
            self.status_var.set("Nothing to redo.")
            return
        if not self._to_checkpoint_tab(self._redo_stack, "redo"):
            return
        self._undo_stack.append(self._gate_state_snapshot())
        self._restore_gate_state(self._redo_stack.pop())
        self._note_unsaved()
        self.status_var.set(
            f"Redo. ({len(self._redo_stack)} more)")

    def _gate_state_snapshot(self):
        import copy
        seq = dict(self._sample_gate_seq)
        active = self._active_sample
        if active is not None:
            seq[active] = max(seq.get(active, 0), self._gate_id_seq)
        ws = getattr(self, '_workspace_panel', None)
        ts = getattr(self, '_view_tabs', None)
        return {
            # The tray tab whose gates these are: each tab keeps its own,
            # so Undo goes back to that tab before restoring them.
            'tab':            ts.active if ts is not None else None,
            'gates':          copy.deepcopy(self._sample_gates),
            'order':          copy.deepcopy(self._sample_gate_order),
            'seq':            seq,
            'cluster_labels': copy.deepcopy(self._cluster_labels),
            'quad_seq':       self._quad_set_seq,
            'workspace':      ws.model.to_dict() if ws is not None else None,
            # Undoing a fix's Apply offers the fix again (editor_fixes).
            'fixes':          (list(getattr(self, '_applied_fixes', None) or []),
                               list(getattr(self, '_declined_fixes', None)
                                    or [])),
        }

    def _restore_gate_state(self, snap):
        import copy
        # A sample's id counter never winds back: the next gate must not get
        # an undone gate's id, or a clip of the undone gate, pasted back
        # under its id as a Cut is (_paste_gate_tree), becomes the operand of
        # a boolean that had used the newer gate.
        seq = dict(self._sample_gate_seq)
        if self._active_sample is not None:
            seq[self._active_sample] = max(seq.get(self._active_sample, 0),
                                           self._gate_id_seq)
        for name, n in snap['seq'].items():
            seq[name] = max(seq.get(name, 0), n)
        self._sample_gates      = copy.deepcopy(snap['gates'])
        self._sample_gate_order = copy.deepcopy(snap['order'])
        self._sample_gate_seq   = seq
        self._cluster_labels    = copy.deepcopy(snap['cluster_labels'])
        self._quad_set_seq      = snap.get('quad_seq', self._quad_set_seq)
        if 'fixes' in snap:
            self._applied_fixes = list(snap['fixes'][0])
            self._declined_fixes = list(snap['fixes'][1])
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
            except Exception as exc:  # noqa: BLE001 - noted; a workspace restore failure must not undo the gate restore
                note("restoring the workspace model on undo", exc)

    def _clear_undo_pending(self):
        self._undo_pending = False
