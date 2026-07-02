"""Copy / cut / paste of samples, gate trees, and FCS from the clipboard.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import json
import os
import re
import time

from .editor_base import EditorMixin

# Marker key identifying an OpenFlo cross-instance sample-transfer bundle on the
# OS clipboard (vs. a plain file-path copy or unrelated text).
_TRANSFER_MARKER = '__openflo_transfer__'


class ClipboardMixin(EditorMixin):
    """Clipboard handlers for the tree: copy/cut/paste gate subtrees, paste FCS, copy a sample path, and cross-instance sample transfer."""

    # ── cross-instance transfer (Copy here → Pull in another OpenFlo) ────────
    def _sample_transfer_bundle(self, names):
        """JSON-able bundle (raw FCS path + gates + grouping) for ``names`` — the
        payload copied to the OS clipboard so another running OpenFlo instance
        can pull the sample(s) in WITH their gates. Only raw-FCS samples are
        included (the other instance re-loads from the .fcs)."""
        samples = []
        for name in names:
            s = self._samples.get(name)
            if s is None:
                continue
            path = getattr(s, 'path', '') or ''
            if not path or not path.lower().endswith('.fcs'):
                continue
            gates = self._sample_gates.get(name, {})
            order = self._sample_gate_order.get(name, list(gates))
            glist = []
            for gid in order:
                g = gates.get(gid)
                if g is None:
                    continue
                gd = dict(g)
                gd['id'] = gid
                glist.append(gd)
            entry = {'name': name, 'path': path, 'gates': glist,
                     'trial': self._sample_trial.get(name, '')}
            if name in self._sample_is_comp:
                entry['is_comp'] = bool(self._sample_is_comp[name])
            samples.append(entry)
        return {_TRANSFER_MARKER: 1, 'samples': samples}

    def _copy_samples_transfer(self, names):
        """Copy sample(s) (with their gates) to the OS clipboard for transfer to
        another OpenFlo instance."""
        names = [n for n in (names or []) if n in self._samples]
        if not names:
            self.status_var.set("No sample selected to copy.")
            return
        bundle = self._sample_transfer_bundle(names)
        if not bundle['samples']:
            self.status_var.set(
                "Selected sample(s) have no raw FCS to transfer "
                "(processed/CSV samples can't be re-loaded elsewhere).")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(json.dumps(bundle))
        except Exception as exc:
            self.status_var.set(f"Copy for transfer failed: {exc}")
            return
        ngates = sum(len(s['gates']) for s in bundle['samples'])
        self.status_var.set(
            f"Copied {len(bundle['samples'])} sample(s) + {ngates} gate(s) — "
            f"Pull them in another OpenFlo window (right-click → Pull).")

    def _transfer_dir(self):
        d = os.path.join(os.path.expanduser('~'), '.openflo', 'transfer')
        os.makedirs(d, exist_ok=True)
        return d

    def _send_samples_transfer(self, names):
        """MARK sample(s) to move to another instance. Unlike Copy, this stages
        a MOVE: the bundle goes on the clipboard with a move id, and the rows are
        flagged (✄) — but they're NOT removed yet. They're removed only once the
        destination actually pastes them (it drops a <move_id>.done marker we
        poll for in _watch_pending_move). Copying anything else cancels the
        pending move (the samples stay)."""
        names = [n for n in (names or []) if n in self._samples]
        if not names:
            self.status_var.set("No sample selected to send.")
            return
        bundle = self._sample_transfer_bundle(names)
        sent = [s['name'] for s in bundle['samples']]
        if not sent:
            self.status_var.set(
                "Selected sample(s) have no raw FCS to send "
                "(processed/CSV samples can't be re-loaded elsewhere).")
            return
        move_id = f"{os.getpid()}-{int(time.time() * 1000)}"
        bundle['move_id'] = move_id
        bundle['move_pid'] = os.getpid()
        payload = json.dumps(bundle)
        try:
            self.clipboard_clear()
            self.clipboard_append(payload)
        except Exception as exc:
            self.status_var.set(f"Send failed: {exc}")
            return
        # Stage (don't delete). Remember the exact clipboard text so the watcher
        # can tell "still ours" from "user copied something else".
        self._pending_move = {'id': move_id, 'names': sent, 'clip': payload}
        self._mark_pending_move(sent, True)
        ngates = sum(len(s['gates']) for s in bundle['samples'])
        self.status_var.set(
            f"Marked {len(sent)} sample(s) + {ngates} gate(s) to move (✄) — "
            f"paste in another OpenFlo window to complete; copy anything else "
            f"to cancel.")
        self._watch_pending_move()

    def _mark_pending_move(self, names, on):
        """Flag/unflag sample rows as pending-move (drives the ✄ indicator in
        _insert_sample_subtree) and refresh the tree."""
        cur = getattr(self, '_pending_move_names', None)
        if cur is None:
            cur = self._pending_move_names = set()
        if on:
            cur.update(names)
        else:
            cur.difference_update(names)
        try:
            self._refresh_gate_list()
        except Exception:
            pass

    def _watch_pending_move(self):
        """Poll a staged move: complete it (remove the samples here) once the
        destination drops the <move_id>.done marker; cancel it if the clipboard
        was overwritten. Reschedules itself while still pending."""
        pm = getattr(self, '_pending_move', None)
        if not pm:
            return
        done = os.path.join(self._transfer_dir(), pm['id'] + '.done')
        if os.path.isfile(done):
            try:
                os.remove(done)
            except Exception:
                pass
            names = pm['names']
            self._pending_move = None
            getattr(self, '_pending_move_names', set()).difference_update(names)
            self._remove_samples([n for n in names if n in self._samples])
            self.status_var.set(
                f"Moved {len(names)} sample(s) to another OpenFlo window.")
            return
        try:
            still_ours = (self.clipboard_get() == pm['clip'])
        except Exception:
            still_ours = False
        if not still_ours:
            names = pm['names']
            self._pending_move = None
            self._mark_pending_move(names, False)
            self.status_var.set(
                "Move cancelled (clipboard changed) — sample(s) kept.")
            return
        self.after(700, self._watch_pending_move)

    def _read_transfer_bundle(self):
        """Parse a transfer bundle off the OS clipboard, or None if the
        clipboard doesn't hold one."""
        try:
            text = self.clipboard_get()
        except Exception:
            return None
        if not isinstance(text, str) or _TRANSFER_MARKER not in text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
        if (not isinstance(data, dict) or not data.get(_TRANSFER_MARKER)
                or not isinstance(data.get('samples'), list)):
            return None
        return data

    def _paste_samples_transfer(self):
        """Pull sample(s) (FCS + gates + grouping) from a transfer bundle copied
        in another OpenFlo instance. Stages each sample's gates/trial keyed by
        file path (reusing the session-restore drain in ``_on_loaded``) and
        queues the FCS load on the background pool."""
        bundle = self._read_transfer_bundle()
        if bundle is None:
            self.status_var.set(
                "Clipboard has no OpenFlo sample transfer — Copy one in "
                "another OpenFlo window first.")
            return
        paths, missing = [], []
        for entry in bundle.get('samples', []):
            path = entry.get('path') or ''
            if not path or not os.path.isfile(path):
                missing.append(entry.get('name')
                               or os.path.basename(path) or '?')
                continue
            pkey = os.path.normcase(os.path.abspath(path))
            meta: dict = {'gates': list(entry.get('gates') or [])}
            if entry.get('trial'):
                meta['trial'] = entry['trial']
            if 'is_comp' in entry:
                meta['is_comp'] = bool(entry['is_comp'])
            self._pending_sample_meta[pkey] = meta
            paths.append(path)
        if paths:
            self._queue_fcs_loads(paths)
        # If this bundle was SENT (a staged move from another instance), drop a
        # marker so the source instance completes the move (removes its copies).
        mid = bundle.get('move_id')
        if mid and paths and bundle.get('move_pid') != os.getpid():
            try:
                open(os.path.join(self._transfer_dir(), str(mid) + '.done'),
                     'w').close()
            except Exception:
                pass
        msg = f"Pulling {len(paths)} sample(s) from another instance."
        if missing:
            msg += f"  Missing FCS for: {', '.join(missing[:4])}"
            if len(missing) > 4:
                msg += f" (+{len(missing) - 4})"
        self.status_var.set(msg)

    def _paste_fcs_from_clipboard(self):
        """Read the OS clipboard and return any .fcs paths found.
        Tolerates Explorer's 'Copy as path' quoted form and multi-line
        / whitespace-separated entries. Defensive: any clipboard format
        we can't interpret as text is silently ignored."""
        try:
            text = self.clipboard_get()
        except Exception:
            return []
        if not isinstance(text, str):
            return []
        candidates = []
        for tok in re.split(r'[\r\n]+', text):
            for piece in re.split(r'(?<=\.fcs)\s+', tok, flags=re.I):
                p = piece.strip().strip('"').strip("'").strip()
                if p:
                    candidates.append(p)
        out = []
        for p in candidates:
            try:
                if p.lower().endswith('.fcs') and os.path.isfile(p):
                    out.append(p)
            except Exception:
                continue
        return out

    def _paste_gate_tree(self):
        """Paste the clipboard subtree into the active sample. The
        subtree's root attaches under the currently-selected gate (or
        as a root in the active sample if a sample row / nothing is
        selected). Multiple pastes don't consume the clipboard."""
        import copy as _copy
        if self._active_sample is None or not self._clip_payload:
            return
        subtree = _copy.deepcopy(self._clip_payload)
        if not subtree:
            return
        self._checkpoint()
        root_clip_id = subtree[0].get('_clip_id')

        # Resolve paste parent from current selection.
        paste_parent = None
        sel = self.gate_tv.selection()
        if sel:
            parsed = self._parse_iid(sel[0])
            if parsed:
                if parsed[0] == 'sample' and parsed[1] != self._active_sample:
                    self._set_active_sample(parsed[1])
                if parsed[0] == 'gate' and parsed[1] == self._active_sample:
                    paste_parent = parsed[2]

        # Assign fresh ids in the active sample.
        old_to_new = {}
        for g in subtree:
            self._gate_id_seq += 1
            if self._active_sample is not None:
                self._sample_gate_seq[self._active_sample] = self._gate_id_seq
            old_to_new[g['_clip_id']] = f'g{self._gate_id_seq}'

        for g in subtree:
            clip_id = g.pop('_clip_id', None)
            new_id  = old_to_new[clip_id]
            if clip_id == root_clip_id:
                g['parent_id'] = paste_parent
            else:
                g['parent_id'] = old_to_new.get(
                    g.get('parent_id'), paste_parent)
            self._gates[new_id] = g
            self._gate_id_order.append(new_id)

        self.status_var.set(
            f"Pasted {len(subtree)} gate(s) into '{self._active_sample}'.")
        self._refresh_gate_list()
        if self.gate_display_var.get() in ('filter', 'highlight'):
            self._schedule_replot(0)

    def _on_copy(self, event=None):
        sel = self.gate_tv.selection()
        if not sel:
            return 'break'
        parsed = self._parse_iid(sel[0])
        if parsed is None:
            return 'break'
        if parsed[0] == 'gate':
            subtree = self._collect_gate_subtree(parsed[1], parsed[2])
            if subtree:
                self._clip_kind    = 'gate_tree'
                self._clip_payload = subtree
                self.status_var.set(
                    f"Copied {len(subtree)} gate(s) "
                    f"(paste under a gate to nest, or a sample row for root).")
        elif parsed[0] == 'sample':
            name = parsed[1]
            sample = self._samples.get(name)
            path = getattr(sample, 'path', None) if sample else None
            if path:
                try:
                    self.clipboard_clear()
                    self.clipboard_append(path)
                except Exception:
                    pass
                self._clip_kind    = 'sample_paths'
                self._clip_payload = [path]
                self.status_var.set(f"Copied path of '{name}' to clipboard.")
        return 'break'

    def _on_cut(self, event=None):
        sel = self.gate_tv.selection()
        if not sel:
            return 'break'
        parsed = self._parse_iid(sel[0])
        if parsed is None:
            return 'break'
        if parsed[0] != 'gate':
            # Cutting a sample is destructive; we just Copy and don't
            # auto-remove the sample. User uses Remove for that.
            return self._on_copy(event)
        # Copy first, then delete the subtree from its source sample.
        self._on_copy(event)
        sample_name, gid = parsed[1], parsed[2]
        self._remove_gate_cascade_in(sample_name, gid)
        self._refresh_gate_list()
        if self.gate_display_var.get() in ('filter', 'highlight'):
            self._schedule_replot(0)
        return 'break'

    def _on_paste(self, event=None):
        """Order of precedence:
          1. Internal gate-tree clipboard → paste into active sample.
          2. OS clipboard FCS paths       → queue them as new samples.
        Reports a brief no-op message if neither applies."""
        if self._clip_kind == 'gate_tree' and self._clip_payload:
            self._paste_gate_tree()
            return 'break'
        fcs = self._paste_fcs_from_clipboard()
        if fcs:
            self._queue_fcs_loads(fcs)
            return 'break'
        self.status_var.set(
            "Nothing to paste (no copied gates, no .fcs paths on clipboard).")
        return 'break'

    def _copy_sample_path(self, name):
        sample = self._samples.get(name)
        path = getattr(sample, 'path', None) if sample else None
        if not path:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(path)
        except Exception:
            pass
        self.status_var.set(f"Copied path of '{name}' to clipboard.")
