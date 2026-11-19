"""Undo / redo via gate-state checkpoints (and the channel labels a
checkpoint carries, editor_channels._set_channel_labels).

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from .editor_base import EditorMixin
from .errors import note


def _mirror_snapshot(ed, snap):
    """The state of editor ``ed`` now, to step back to after restoring
    ``snap``: a gate snapshot, plus the current labels of the channels
    ``snap`` relabels when it carries a channel-label change
    (editor_channels._set_channel_labels)."""
    now = ed._gate_state_snapshot()
    if 'labels' in snap:
        now['labels'] = ed._label_layer(snap['labels']['channel'])
    return now


def _gate_changes(before, after):
    """What going from gates ``before`` to ``after`` (``{sample: {gid:
    gate}}``) does, as ``[(sample, verb, gate name)]`` with verb 'removed',
    'restored', 'switched on', 'switched off' or 'changed', in sample then
    gate order."""
    out = []
    for n in list(dict.fromkeys(list(before) + list(after))):
        b, a = before.get(n) or {}, after.get(n) or {}
        for gid in list(dict.fromkeys(list(b) + list(a))):
            if gid not in a:
                verb = 'removed'
            elif gid not in b:
                verb = 'restored'
            elif a[gid] == b[gid]:
                continue
            elif ({k: v for k, v in a[gid].items() if k != 'enabled'}
                  == {k: v for k, v in b[gid].items() if k != 'enabled'}):
                verb = ('switched on' if a[gid].get('enabled', True)
                        else 'switched off')
            else:
                verb = 'changed'
            g = a.get(gid) or b.get(gid) or {}
            out.append((n, verb, str(g.get('name') or gid)))
    return out


def _describe_changes(changes, shown, limit=2):
    """Status text naming the first ``limit`` gate ``changes`` (see
    ``_gate_changes``), marking samples the tab does not show (``shown``:
    sample -> bool), so an Undo never reverts a gate the user cannot see
    without saying so (T3)."""
    if not changes:
        return ''
    parts = []
    for n, verb, name in changes[:limit]:
        where = n if shown(n) else f"{n}, not shown in this tab"
        parts.append(f"{verb} '{name}' ({where})")
    more = len(changes) - limit
    text = '; '.join(parts)
    if more > 0:
        text += f"; {more} more gate change(s)"
    hidden = sum(1 for n, _v, _g in changes[limit:] if not shown(n))
    if hidden:
        text += f", {hidden} on samples not shown in this tab"
    return text + '.'


class UndoMixin(EditorMixin):
    """Checkpoint / undo / redo of editor gate state."""

    def _checkpoint(self):
        """Record a pre-mutation undo checkpoint. Call BEFORE mutating gate
        state. No-op while suspended (bulk loads) or when one was already
        taken this Tk event (coalesces a multi-step gesture into one undo)."""
        if self._suspend_undo or self._undo_pending:
            return
        self._undo_pending = True
        self._push_undo(self._gate_state_snapshot())
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
        self._push_undo({'barrier': str(what)})

    def _push_undo(self, entry, keep_redo=False):
        """Put ``entry`` (a checkpoint or a barrier) on the undo stack, as
        unsaved work, capped at _UNDO_MAX. A new change makes the redo
        stack stale, so it is cleared unless ``keep_redo`` (Redo itself)."""
        self._note_unsaved()
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self._UNDO_MAX:
            self._undo_stack.pop(0)
        if not keep_redo:
            self._redo_stack.clear()

    def _checkpoint_tab_deletion(self, idx, keep_redo=False):
        """Record the undo checkpoint for deleting the active tab, which sat
        at ``idx``: the live gates are that tab's own, so a plain checkpoint
        carries them; ``tab_at`` tells Undo to put the tab back there first.
        Call BEFORE removing the tab. ``keep_redo`` is for Redo itself,
        which must not drop what is left to redo."""
        if self._suspend_undo:
            return
        snap = self._gate_state_snapshot()
        snap['tab_at'] = int(idx)
        self._push_undo(snap, keep_redo=keep_redo)

    def _forget_undo_samples(self, names):
        """Drop removed samples from every undo / redo checkpoint, so Undo
        cannot bring back the gates of a sample that is gone (nor a deleted
        tab's membership of it)."""
        names = set(names)
        for snap in list(self._undo_stack) + list(self._redo_stack):
            if 'barrier' in snap:
                continue
            for key in ('gates', 'order', 'seq', 'cluster_labels'):
                for n in names:
                    (snap.get(key) or {}).pop(n, None)
            if 'tab_at' in snap and snap.get('tab') is not None:
                for n in names:
                    snap['tab'].forget_sample(n)

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
            at = stack[-1].get('tab_at')
            if at is None:
                stack.pop()
                self.status_var.set(f"Cannot {verb}: that change was made "
                                    "in a tab that has been deleted.")
                return False
            # The checkpoint is the tab's deletion: put it back first, so
            # its gates restore into it.
            idx = self._reinsert_tab(tab, at)
        self._switch_tab(idx)
        return True

    def _reverted_text(self, snap, tab_before):
        """Undo/Redo status detail for restoring ``snap``: the tab it went
        to when that is not ``tab_before`` (the checkpoint was taken
        there), and the gates it changes."""
        ts = getattr(self, '_view_tabs', None)
        try:
            changes = _gate_changes(self._sample_gates,
                                    self._restored_gates(snap)[0])
            what = _describe_changes(changes, self._tab_has_sample)
        except Exception as exc:  # noqa: BLE001 - reported via note(); the status detail must never stop Undo
            note("describe what Undo reverts", exc)
            what = ''
        if ts is not None and ts.active is not tab_before:
            what = f"back in tab '{ts.active.name}'. {what}".strip()
        return what

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
        tab_before = getattr(getattr(self, '_view_tabs', None), 'active', None)
        if not self._to_checkpoint_tab(self._undo_stack, "undo"):
            return
        snap = self._undo_stack.pop()
        detail = ('' if 'tab_at' in snap
                  else self._reverted_text(snap, tab_before))
        redo = _mirror_snapshot(self, snap)
        put_back = 'tab_at' in snap
        if put_back:
            # Redo deletes the tab again (it is the active one by now).
            redo['delete_tab'] = True
        self._redo_stack.append(redo)
        self._restore_gate_state(snap)
        self._note_unsaved()
        what = (f"Undo: tab '{snap['tab'].name}' is back." if put_back
                else f"Undo: {detail}" if detail else "Undo.")
        self.status_var.set(
            f"{what} ({len(self._undo_stack)} more, "
            f"{len(self._redo_stack)} redo)")

    def _redo(self, event=None):
        if event is not None and self._focus_in_text():
            return
        if not self._redo_stack:
            self.status_var.set("Nothing to redo.")
            return
        tab_before = getattr(getattr(self, '_view_tabs', None), 'active', None)
        if not self._to_checkpoint_tab(self._redo_stack, "redo"):
            return
        snap = self._redo_stack.pop()
        if snap.get('delete_tab'):
            # Redo of a tab's deletion: the tab is active again, with its
            # gates live, so the undo checkpoint is the deletion's own.
            ts = self._tabs()
            self._tab_snapshot()
            self._checkpoint_tab_deletion(ts.current, keep_redo=True)
            self._drop_active_tab()
            self._note_unsaved()
            self.status_var.set(
                f"Redo: tab deleted. ({len(self._redo_stack)} more)")
            return
        detail = self._reverted_text(snap, tab_before)
        self._undo_stack.append(_mirror_snapshot(self, snap))
        self._restore_gate_state(snap)
        self._note_unsaved()
        what = f"Redo: {detail}" if detail else "Redo."
        self.status_var.set(f"{what} ({len(self._redo_stack)} more)")

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
            # Undo past a workspace Load puts its file back too.
            'workspace_file': (ws.file_state() if ws is not None
                               and hasattr(ws, 'file_state') else None),
            # Undoing a fix's Apply offers its results again (editor_fixes):
            # the corrected ones through their restored stamps, the declined
            # ones through this.
            'fixes_declined': copy.deepcopy(
                list(getattr(self, '_declined_fixes', None) or [])),
        }

    def _restored_gates(self, snap):
        """The gates, gate order and cluster labels restoring ``snap`` puts
        live: copies of the snapshot's, plus the current ones of each loaded
        sample it does not hold. Loads are not on the undo stack (they run
        with undo suspended), so a sample loaded after the checkpoint keeps
        what it has now (U1): dropping it would leave the sample with no
        gates, and the active sample's gates detached from the ones the
        session saves."""
        import copy
        gates = copy.deepcopy(snap['gates'])
        order = copy.deepcopy(snap['order'])
        labels = copy.deepcopy(snap['cluster_labels'])
        for n in self._samples:
            if n in snap['gates']:
                continue
            gates[n] = self._sample_gates.get(n) or {}
            order[n] = self._sample_gate_order.get(n) or list(gates[n])
            if n in self._cluster_labels:
                labels[n] = self._cluster_labels[n]
        return gates, order, labels

    def _restore_gate_state(self, snap):
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
        gates, order, labels = self._restored_gates(snap)
        self._sample_gates      = gates
        self._sample_gate_order = order
        self._sample_gate_seq   = seq
        self._cluster_labels    = labels
        self._quad_set_seq      = snap.get('quad_seq', self._quad_set_seq)
        if 'fixes_declined' in snap:
            import copy
            self._declined_fixes = copy.deepcopy(snap['fixes_declined'])
        # Rebind the active-sample shortcuts to the restored containers
        # (always the entries the session saves, never a detached dict).
        self._rebind_active_gates()
        self._gate_id_seq = self._sample_gate_seq.get(self._active_sample, 0)
        if 'labels' in snap:            # a channel-label change
            self._restore_label_layer(snap['labels'])
        self._refresh_gate_list()
        self._schedule_replot(0)
        # Widen undo to the Pipeline Workspace: restore its model too (same
        # Undo button reverts workspace add/remove/group/comp/fmo/clear).
        ws = getattr(self, '_workspace_panel', None)
        ws_snap = snap.get('workspace')
        if ws is not None and ws_snap is not None:
            try:
                ws.restore_model(ws_snap,
                                 file_state=snap.get('workspace_file'))
            except Exception as exc:  # noqa: BLE001 - noted; a workspace restore failure must not undo the gate restore
                note("restoring the workspace model on undo", exc)

    def _clear_undo_pending(self):
        self._undo_pending = False
