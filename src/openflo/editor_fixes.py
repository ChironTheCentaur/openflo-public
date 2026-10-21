"""Fixes that reach a reopened session -- editor mixin.

The registry and the scan are in fixes (Tk-free). This offers what a scan
finds when a session opens (and from Tools > Review fixes for this
session…), applies the ticked rows as one undo step, records the decision in
the session, and offers what was built on a corrected result as a follow-up
re-run, with the files exported from it before the fix. Nothing changes
unless the user applies.
"""
from __future__ import annotations

from . import fixes as _fixes
from . import result_stamp as _stamps
from .editor_base import EditorMixin


class FixesMixin(EditorMixin):
    """Review and apply registered fixes to this session's saved results."""

    def _settle_gate_stamps(self):
        """Take the stamp off every gate moved by hand since a step placed it
        (result_stamp.settle_gate_stamps): the gate is the user's now, and a
        fix must never overwrite it. Called on every gate-list refresh, which
        every gate edit ends with, and before a save."""
        for gates in list(getattr(self, '_sample_gates', {}).values()):
            _stamps.settle_gate_stamps(gates)

    def _check_fixes_on_open(self):
        """After a session's samples have loaded: show the review window when
        a fix applies to one of its results the session has not yet applied
        or declined. Returns the window, or None."""
        try:
            self._settle_gate_stamps()
            targets = _fixes.scan(self)
        except Exception as exc:                          # noqa: BLE001
            print(f"[fixes] scan failed: {exc}", flush=True)
            return None
        if not targets:
            return None
        return self._review_fixes(targets)

    def _review_fixes(self, targets=None):
        """Show the fixes for this session's results. Without ``targets``
        (the Tools menu) the ones applied or declined earlier are listed too,
        marked so, and can be applied now."""
        from .ui_fixes import FixReviewWindow
        self._settle_gate_stamps()
        if targets is None:
            targets = _fixes.scan(self, include_settled=True)
        rows = _fixes.plan(self, targets)
        return FixReviewWindow(self, rows, on_apply=self._apply_fix_rows,
                               on_rerun=self._rerun_follow_ups,
                               on_open=self._open_follow_up,
                               stale_exports=self._stale_exports)

    def _stale_exports(self):
        """Files exported from the samples the last Apply corrected, before
        it (fixes.stale_exports): listed, never re-exported."""
        return _fixes.stale_exports(self._audit_log.entries())

    def _apply_fix_rows(self, chosen, shown=()):
        """Apply the ``chosen`` review rows (one undo step) and record, for
        the session, which fixes were applied and which of the ``shown`` ones
        were declined (an informational row counts as seen, so it is not
        shown again). Returns the follow-up rows (fixes.downstream)."""
        chosen = [r for r in chosen if not r.get('informational')
                  and r.get('new') is not None]
        applied = []
        for r in chosen:
            for f in r['fixes']:
                if f not in applied:
                    applied.append(f)
        declined = []
        for r in shown:
            for f in r.get('fixes') or ():
                if (f not in applied and f not in declined
                        and r.get('status') == 'open'):
                    declined.append(f)
        done = []
        if chosen:
            self._checkpoint()
            done = _fixes.apply_rows(self, chosen)
            cols = sorted({s for s, kind, _k in done if kind == 'column'})
            if cols:
                # Undo restores gates, not data columns: say so rather than
                # step past the change.
                self._undo_barrier("the corrected columns")
                self._data_changed(cols)
                self._refresh_channel_choices()
        _fixes.record(self, applied, declined)
        self._note_unsaved()
        self._audit('fix.apply', fixes=applied, declined=declined,
                    results=[{'sample': r['sample'], 'result': r['result'],
                              'key': r['key'], 'fixes': list(r['fixes']),
                              'made_by': r['stamp'].get('version')}
                             for r in chosen])
        self._refresh_gate_list()
        self._schedule_replot(0)
        if done:
            self.status_var.set(
                f"Applied {', '.join(applied)} to {len(done)} result(s); "
                "the originals are kept in the session.")
        elif declined:
            self.status_var.set("Fixes left as they were; they will not be "
                                "offered again (Tools > Review fixes for "
                                "this session…).")
        return _fixes.downstream(self, done)

    def _rerun_follow_ups(self, follows):
        """Re-run the chosen follow-ups (results built on a corrected one)
        with the current code: one undo step. Only re-fitted gates re-run
        from here; the window opens the dialog for the rest
        (_open_follow_up). Returns how many changed."""
        rows = [_fixes.rerun_row(self, f) for f in follows if f.get('rerun')]
        rows = [r for r in rows if r.get('new') is not None]
        if not rows:
            return 0
        self._checkpoint()
        done = _fixes.apply_rows(self, rows)
        self._audit('fix.rerun', results=[
            {'sample': r['sample'], 'result': r['result'], 'key': r['key'],
             'made_by': r['stamp'].get('version')} for r in rows])
        self._refresh_gate_list()
        self._schedule_replot(0)
        self.status_var.set(f"Re-ran {len(done)} result(s) built on the "
                            "corrected ones.")
        return len(done)

    def _open_follow_up(self, follow):
        """Open the dialog that re-runs a follow-up's step (its ``opens``),
        its sample selected first -- and for a gate, the population it was
        fitted in, which Auto-gate fits in. It only opens: the step runs when
        the user starts it there. Returns whether a dialog was opened."""
        opener = getattr(self, follow.get('opens') or '', None)
        if opener is None:
            return False
        sample = follow['sample']
        if sample in self._samples:
            gid = None
            if follow['result'] == 'gate':
                gid = (self._sample_gates.get(sample, {}).get(follow['key'])
                       or {}).get('parent_id')
            try:
                iid = (self._gate_iid(sample, gid) if gid
                       else self._sample_iid(sample))
                if not self.gate_tv.exists(iid):
                    iid = self._sample_iid(sample)
                self.gate_tv.selection_set(iid)
                self.gate_tv.see(iid)
                # Now, not on the queued select event: the dialog reads the
                # active sample as it opens.
                self._on_tree_select()
            except Exception as exc:                      # noqa: BLE001
                print(f"[fixes] select {sample}: {exc}", flush=True)
        opener()
        return True
