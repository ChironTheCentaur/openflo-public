"""Live refresh of open analysis windows when the editor's analysis changes.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).

A window that shows numbers computed from the editor (Statistics, Plate
view, Kinetics...) registers once with ``_watch_changes(win, on_change,
parts)``. The editor calls ``_notify_changed()`` wherever its analysis state
may have changed (every gate edit ends in ``_refresh_gate_list``, a data
rewrite in ``_data_changed``, an audited step in ``_audit``, a session change
in ``_note_unsaved``). That call is cheap: with no window watching it does
nothing, else it (re)starts one timer, so a burst of edits (a drag, typing)
makes one refresh after it settles. When the timer fires, each watched part
is fingerprinted once, and a window is called back only when a part it
watches differs from what it last showed, so a zoom or an axis change (which
also rebuilds the tree) recomputes nothing. A window that is open but not
shown (minimised) is left until it is shown again, and a closed one is
unhooked by its ``<Destroy>``.
"""
from __future__ import annotations

import hashlib
import json
import tkinter as tk

from .editor_base import EditorMixin
from .errors import note

# How long the editor must be quiet before open windows recompute (ms).
WATCH_DELAY_MS = 500

# What a window can watch:
#   gates       every sample's gates (shape, parent, name) and gate order
#   samples     the loaded samples, their order, ticks, trials, set-asides
#   data        each sample's data version (compensation, transforms, trims)
#   stat_forms  the GeoMean / rCV forms chosen in Preferences
#   audit       the analysis steps recorded in the audit trail
WATCH_PARTS = ('gates', 'samples', 'data', 'stat_forms', 'audit')


def _gates_digest(sample_gates, gate_order):
    """One digest of every sample's gates and gate order."""
    from .plotcache import gate_content_sig
    h = hashlib.blake2b(digest_size=12)
    for name in sorted(sample_gates, key=str):
        gates = sample_gates.get(name) or {}
        h.update(repr((str(name), list(gate_order.get(name) or ()))).encode())
        for gid in sorted(gates, key=str):
            g = gates[gid]
            h.update(repr((str(gid), g.get('name'), g.get('label'))).encode())
            h.update(gate_content_sig(g).encode('utf-8', 'surrogatepass'))
    return h.hexdigest()


class WatchMixin(EditorMixin):
    """Debounced, fingerprinted refresh of the analysis windows watching
    this editor."""

    def _watch_changes(self, win, on_change, parts=WATCH_PARTS):
        """Call ``on_change()`` after the editor's ``parts`` change, while
        ``win`` is open; see the module docstring. ``win`` is the Toplevel
        whose closing unhooks it and whose showing catches it up."""
        parts = tuple(p for p in parts if p in WATCH_PARTS)
        watchers = self.__dict__.setdefault('_watchers', {})
        key = str(win)
        rec = {'win': win, 'on_change': on_change, 'parts': parts,
               'seen': self._watch_state(parts)}
        watchers[key] = rec

        def _gone(e):
            if e.widget is win and watchers.get(key) is rec:
                watchers.pop(key, None)

        def _shown(e):
            if e.widget is win and watchers.get(key) is rec:
                self._notify_changed()
        try:
            win.bind('<Destroy>', _gone, add='+')
            win.bind('<Map>', _shown, add='+')
        except tk.TclError:
            watchers.pop(key, None)

    def _notify_changed(self):
        """Something the analysis windows show may have changed: refresh the
        watching windows once the editor has been quiet WATCH_DELAY_MS."""
        if not self.__dict__.get('_watchers'):
            return
        # Not ``_watch_after``: that is the folder watch's poll timer
        # (editor_load), on this same editor object.
        pending = self.__dict__.get('_live_refresh_after')
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
        try:
            self._live_refresh_after = self.after(WATCH_DELAY_MS, self._flush_watchers)
        except tk.TclError:                  # the editor is closing
            self._live_refresh_after = None

    def _flush_watchers(self):
        """Call back each shown window whose watched parts changed."""
        self._live_refresh_after = None
        if getattr(self, '_clustering_busy', False):
            # A clustering worker is rewriting sample frames off the Tk
            # thread; fingerprinting them now would race. Its finish
            # rewrites the data (_data_changed), which notifies again.
            return
        cache: dict = {}
        for rec in list(self.__dict__.get('_watchers', {}).values()):
            self._watch_run(rec, cache)

    def _watch_run(self, rec, cache):
        win = rec['win']
        try:
            if not win.winfo_exists():
                self.__dict__.get('_watchers', {}).pop(str(win), None)
                return
            if not win.winfo_viewable():     # minimised: caught up on <Map>
                return
        except tk.TclError:
            return
        state = self._watch_state(rec['parts'], cache)
        if state == rec['seen']:
            return
        rec['seen'] = state
        try:
            rec['on_change']()
        except Exception as exc:  # noqa: BLE001 - noted; one window's failed refresh must not stop the others or the edit that caused it
            note(f"live refresh of {type(win).__name__}", exc)

    def _watch_state(self, parts, cache=None):
        """``{part: fingerprint}`` for ``parts``; ``cache`` shares each
        fingerprint between the windows of one flush."""
        cache = {} if cache is None else cache
        out = {}
        for p in parts:
            if p not in cache:
                cache[p] = self._watch_fingerprint(p)
            out[p] = cache[p]
        return out

    def _watch_fingerprint(self, part):
        loaded = [n for n in self._sample_order if n in self._samples]
        if part == 'gates':
            return _gates_digest(self._sample_gates, self._sample_gate_order)
        if part == 'samples':
            return json.dumps(
                [[str(n), bool(self._sample_plot_enabled.get(n, True)),
                  str(self._sample_trial.get(n, '')),
                  str(self._sample_set_aside.get(n, ''))]
                 for n in loaded])
        if part == 'data':
            return tuple(self._plotcache_version(n) for n in loaded)
        if part == 'stat_forms':
            from .prefs import load_stat_forms
            return tuple(sorted(load_stat_forms().items()))
        if part == 'audit':
            # The trail only grows, and a session load replaces the list:
            # a trim or undo of it in place would need a content digest.
            log = self.__dict__.get('_audit_log')
            return (id(log), len(log) if log is not None else 0)
        return None
