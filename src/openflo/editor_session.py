"""Session save / load / autosave + recent-sessions menu — editor mixin.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import glob
import json
import os
import time
from tkinter import filedialog, messagebox

from .editor_base import EditorMixin
from .prefs import read_prefs, write_pref


class SessionMixin(EditorMixin):
    """Full-state session persistence (.flowsession), periodic autosave, and
    the Open-Recent submenu."""

    SESSION_EXT = '.flowsession'

    _AUTOSAVE_MS = 300_000          # periodic autosave cadence (5 min)

    def _instance_session_id(self):
        """A per-process id (PID + start token) so two OpenFlo instances run
        from the same repo at once don't write the SAME autosave file and
        clobber each other. Stable for the life of this process."""
        sid = getattr(self, '_session_id', None)
        if sid is None:
            sid = f"{os.getpid()}-{int(time.time())}"
            self._session_id = sid
        return sid

    def _autosave_dir(self):
        d = os.path.join(os.path.expanduser('~'), '.openflo', 'autosave')
        os.makedirs(d, exist_ok=True)
        return d

    def _session_autosave_path(self):
        """Per-instance autosave file (``session-<pid>-<token>.flowsession``).
        Two concurrent instances get distinct files, so neither clobbers the
        other's last session; the PID in the name lets resume tell a live
        instance's file from an orphan (see _find_resumable_session)."""
        return os.path.join(
            self._autosave_dir(),
            'session-' + self._instance_session_id() + self.SESSION_EXT)

    @staticmethod
    def _pid_alive(pid):
        """Best-effort: is a process with this PID currently running?"""
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            if os.name == 'nt':
                import ctypes
                # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) — granted even for
                # processes we can't fully open; handle truthy ⇒ alive.
                h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if h:
                    ctypes.windll.kernel32.CloseHandle(h)
                    return True
                return False
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _pid_from_autosave(self, path):
        """Owning PID encoded in a 'session-<pid>-<token>.flowsession' name."""
        try:
            return int(os.path.basename(path).split('-', 2)[1])
        except (IndexError, ValueError):
            return None

    def _find_resumable_session(self):
        """Newest autosave that's an ORPHAN — its owning instance is no longer
        running — which is what we offer to resume on open. Files owned by a
        live process (another running instance, or this one) are skipped so
        resume never steals or clobbers a concurrent session. The legacy
        single-file autosave from older builds is included as a candidate."""
        cands = []
        for p in glob.glob(os.path.join(self._autosave_dir(),
                                        'session-*' + self.SESSION_EXT)):
            pid = self._pid_from_autosave(p)
            # Skip any file owned by a still-running process — another live
            # instance, or this one (we never resume our own running session).
            if pid is not None and self._pid_alive(pid):
                continue
            cands.append(p)
        legacy = os.path.join(os.path.expanduser('~'), '.openflo',
                              'last_session' + self.SESSION_EXT)
        if os.path.isfile(legacy):
            cands.append(legacy)
        if not cands:
            return None
        try:
            cands.sort(key=os.path.getmtime, reverse=True)
        except Exception:
            pass
        return cands[0]

    def _prune_autosaves(self, keep_days=14):
        """Delete orphaned autosaves older than keep_days so the dir doesn't
        grow without bound across launches. Never removes a file owned by a
        currently-running process."""
        cutoff = time.time() - keep_days * 86400
        for p in glob.glob(os.path.join(self._autosave_dir(),
                                        'session-*' + self.SESSION_EXT)):
            try:
                if os.path.getmtime(p) >= cutoff:
                    continue
                pid = self._pid_from_autosave(p)
                if pid is not None and self._pid_alive(pid):
                    continue
                os.remove(p)
            except Exception:
                pass

    def _session_state(self):
        """Serialise the full editor state to a JSON-able dict."""
        from datetime import datetime
        samples = []
        for name in self._sample_order:
            s = self._samples.get(name)
            if s is None:
                continue
            entry = {
                'name': name,
                'path': getattr(s, 'path', '') or '',
                'color': self._sample_colors.get(name, '#1f77b4'),
                'plot_enabled': bool(self._sample_plot_enabled.get(name, False)),
                'trial': self._sample_trial.get(name, 'Trial'),
            }
            # Persist a manual Comps/Samples override only when set (so
            # name-based detection still applies to untouched samples on load).
            if name in self._sample_is_comp:
                entry['is_comp'] = bool(self._sample_is_comp[name])
            samples.append(entry)
        # Per-sample gates as ordered lists carrying their editor id +
        # parent_id (so the hierarchy restores).
        sample_gates = {}
        for name, gates in self._sample_gates.items():
            order = self._sample_gate_order.get(name, list(gates))
            out = []
            for gid in order:
                g = gates.get(gid)
                if g is None:
                    continue
                gd = dict(g)
                gd['id'] = gid
                out.append(gd)
            sample_gates[name] = out
        # _channel_range values are tuples → JSON lists. Skip any None
        # (auto-range) entries — the type allows None even though we
        # pop rather than store it.
        ranges = {ch: [float(rng[0]), float(rng[1])]
                  for ch, rng in self._channel_range.items()
                  if rng is not None}
        from .session_format import SESSION_FORMAT, SESSION_VERSION
        return {
            'format': SESSION_FORMAT,
            'version': SESSION_VERSION,
            'created': datetime.now().isoformat(timespec='seconds'),
            'active_sample': self._active_sample,
            'samples': samples,
            'sample_gates': sample_gates,
            'channel_scale': dict(self._channel_scale),
            'channel_range': ranges,
            'channel_labels': dict(self._channel_labels),
            'plot_mode': self.mode_var.get(),
            'x_channel': self.x_combo.get(),
            'y_channel': self.y_combo.get(),
            'color_channel': self.color_combo.get(),
            'downsample_display': bool(self.ds_display_var.get()),
            'downsample_propagate': bool(self.ds_propagate_var.get()),
            'max_points': self.max_points_var.get(),
            'show_removed': bool(self.show_removed_var.get()),
            'contour_scatter': bool(self.contour_scatter_var.get()),
            'contour_outliers': bool(self.contour_outliers_var.get()),
            'hist_y_mode': self.hist_y_mode.get(),
            'cluster_labels': dict(self._cluster_labels),   # reserved slot
            'audit': self._audit_log.to_list(),
        }

    @staticmethod
    def _has_computed_columns(s):
        """True if a sample's data carries columns produced by analysis
        (clustering / embeddings / FMO gates / calibration) that aren't in the
        raw FCS — i.e. worth persisting so a reopened session keeps them."""
        data = getattr(s, 'data', None)
        cols = set(data.columns) if data is not None else set()
        if cols & {'cluster', 'leiden', 'flowsom', 'flowsom_meta',
                   'pseudotime', 'cell_cycle'}:
            return True
        for c in cols:
            cu = str(c)
            if cu.endswith('_pos') or cu.startswith('MESF:'):
                return True
            up = cu.upper()
            for p in ('UMAP', 'TSNE', 'TRIMAP', 'PACMAP', 'PHATE'):
                if up.startswith(p) and up[len(p):] in ('1', '2'):
                    return True
        return False

    def _resolve_processed_csv(self, s, nm):
        """Absolute path to sample ``nm``'s processed-data sidecar, or '' if
        none exists. Prefers the session's recorded ``processed_csv`` (resolved
        against the session dir); when that's missing or stale, falls back to
        the conventional ``<stem>_data/<safe>.csv`` location so the computed
        columns (clusters / UMAP) are still recovered even if the pointer was
        dropped — e.g. by a racing exit-autosave whose in-memory sample had
        already reloaded the raw FCS."""
        sess_dir = getattr(self, '_session_dir', '') or ''
        pcsv = s.get('processed_csv') or ''
        if pcsv and not os.path.isabs(pcsv):
            pcsv = os.path.join(sess_dir, pcsv)
        if pcsv and os.path.isfile(pcsv):
            return pcsv
        data_dir = getattr(self, '_session_data_dir', '') or ''
        if data_dir:
            guess = os.path.join(data_dir, self._sidecar_safe_name(nm) + '.csv')
            if os.path.isfile(guess):
                return guess
        return ''

    def _write_session(self, path):
        """Core writer — shared by Save Session… and autosave. Also writes a
        processed-data sidecar (``<stem>_data/<name>.csv``) for any sample
        carrying computed columns (clusters / UMAP / FMO gates), and records its
        relative path on the sample entry — so reopening the session restores
        those results instead of re-reading the bare raw FCS."""
        data = self._session_state()
        stem = os.path.splitext(path)[0]
        data_dir = stem + '_data'
        made_dir = False
        for entry in data.get('samples', []):
            s = self._samples.get(entry.get('name'))
            if s is None or not self._has_computed_columns(s):
                continue
            safe = self._sidecar_safe_name(entry['name'])
            try:
                if not made_dir:
                    os.makedirs(data_dir, exist_ok=True)
                    made_dir = True
                s.data.to_csv(os.path.join(data_dir, safe + '.csv'), index=False)
                entry['processed_csv'] = os.path.join(
                    os.path.basename(data_dir), safe + '.csv')
            except Exception as exc:
                print(f"[session] processed sidecar for {entry['name']} "
                      f"failed: {exc}", flush=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return data

    def _save_session(self):
        if not self._samples:
            self.status_var.set("Load at least one sample before saving a session.")
            return
        path = filedialog.asksaveasfilename(
            title="Save editor session",
            defaultextension=self.SESSION_EXT,
            initialfile='session' + self.SESSION_EXT,
            filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            data = self._write_session(path)
            self._push_recent_session(path)
            self.status_var.set(
                f"Saved session: {len(data['samples'])} sample(s) → "
                f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Save session failed: {exc}")
            messagebox.showerror(
                "Save session failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load editor session",
            filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                       ('All files', '*.*')])
        if not path:
            return
        self._load_session_path(path)

    def _load_session_path(self, path):
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            self.status_var.set(f"Load session failed: {exc}")
            messagebox.showerror(
                "Load session failed",
                f"{type(exc).__name__}: {exc}\n\nPath: {path}",
                parent=self)
            return
        if data.get('format') != 'openflo-session':
            messagebox.showerror(
                "Not a session file",
                f"{os.path.basename(path)} isn't an OpenFlo session "
                "(missing format marker).", parent=self)
            return
        # Auto-upgrade an older saved session to the current schema so files
        # from a previous OpenFlo keep opening. Refuse one written by a NEWER
        # build (we can't know its schema) rather than mis-read it.
        from .session_format import SessionVersionError, migrate_session
        try:
            data, _mig_notes = migrate_session(data)
        except SessionVersionError as exc:
            messagebox.showerror("Can't open session", str(exc), parent=self)
            return
        # A fresh session starts a fresh history — undo shouldn't cross
        # back into the previous session's gates.
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Dir of the session file — used to resolve relative processed-data
        # sidecars during restore. _session_data_dir is the conventional
        # sidecar folder (<stem>_data), used as a fallback when a sample's
        # processed_csv pointer is missing.
        self._session_dir = os.path.dirname(os.path.abspath(path))
        self._session_data_dir = os.path.splitext(os.path.abspath(path))[0] + '_data'
        self._apply_session(data)
        self._push_recent_session(path)
        if _mig_notes:
            self._audit('session.migrate', steps=_mig_notes)
            self.status_var.set(
                "Opened a session from an older OpenFlo and upgraded its "
                "format — re-save (Ctrl+S) to keep it current.")

    def _apply_session(self, data):
        """Restore editor state from a parsed session dict.

        Display state is restored immediately. Samples are loaded
        asynchronously (same threaded path as Add-FCS); their gates are
        staged in `_pending_sample_gates` and applied by `_on_loaded`
        as each FCS finishes parsing — reusing the WSP-ingest mechanism.
        """
        # Restore global display config up front (independent of samples).
        self._channel_scale.update(
            {k: str(v) for k, v in (data.get('channel_scale') or {}).items()})
        self._channel_range.update(
            {k: (float(v[0]), float(v[1]))
             for k, v in (data.get('channel_range') or {}).items()
             if isinstance(v, (list, tuple)) and len(v) == 2})
        self._channel_labels.update(
            {k: str(v) for k, v in (data.get('channel_labels') or {}).items()})
        # Restore the provenance trail, then log the load itself so the
        # reopened session records that it was reopened (and from where).
        from .audit import AuditLog
        self._audit_log = AuditLog.from_list(data.get('audit') or [])
        self._audit('session.load',
                    created=data.get('created', ''),
                    n_samples=len(data.get('samples', [])))
        win = getattr(self, '_audit_window', None)
        if win is not None and win.winfo_exists():
            win.refresh()
        # cluster_labels round-trips through JSON, which stringifies the
        # inner int cluster-id keys. Coerce them back to int so lookups by
        # the numeric id (from the data column) hit.
        for sname, lbls in (data.get('cluster_labels') or {}).items():
            if not isinstance(lbls, dict):
                continue
            coerced = {}
            for cid, nm in lbls.items():
                try:
                    coerced[int(cid)] = nm
                except (TypeError, ValueError):
                    coerced[cid] = nm
            self._cluster_labels[sname] = coerced
        try:
            self.ds_display_var.set(bool(data.get('downsample_display', True)))
            self.ds_propagate_var.set(bool(data.get('downsample_propagate', False)))
            self._sync_ds_mode_var()
            self._update_ds_visibility()   # hide Max points if restored Off
            if data.get('max_points'):
                self.max_points_var.set(str(data['max_points']))
            self.show_removed_var.set(bool(data.get('show_removed', False)))
            self.contour_scatter_var.set(
                bool(data.get('contour_scatter', True)))
            self.contour_outliers_var.set(
                bool(data.get('contour_outliers', True)))
            if data.get('hist_y_mode') in ('Fraction', 'Count', '% of Max'):
                self.hist_y_mode.set(data['hist_y_mode'])
            if data.get('plot_mode') in self.PLOT_MODES:
                self.mode_var.set(data['plot_mode'])
            self._sync_hist_y_combo()
        except Exception:
            pass

        # Stage each sample's restore bundle — grouping (trial + Comps/Samples
        # override) AND its gates — keyed by FILE PATH so it survives name
        # disambiguation across reloads. `_on_loaded` drains it by the loaded
        # sample's path. Cleared first so a prior session's missing-file entries
        # can't leak onto a later load.
        self._pending_sample_meta.clear()
        sample_gates = data.get('sample_gates') or {}
        processed_loads = []          # [(name, csv_path)] — restored with cols
        for s in data.get('samples', []):
            nm = s.get('name')
            if not nm:
                continue
            # Prefer the processed-data sidecar (carries clusters/UMAP) when it
            # exists; fall back to the raw FCS path.
            pcsv = self._resolve_processed_csv(s, nm)
            if pcsv:
                load_path = pcsv
                processed_loads.append((nm, pcsv))
            else:
                load_path = s.get('path') or ''
            if not load_path:
                continue
            pkey = os.path.normcase(os.path.abspath(load_path))
            m: dict[str, object] = {'gates': list(sample_gates.get(nm, []))}
            if s.get('trial'):
                m['trial'] = s['trial']
            if 'is_comp' in s:
                m['is_comp'] = bool(s['is_comp'])
            self._pending_sample_meta[pkey] = m

        # Remember the combo selections + active sample to restore once
        # at least one sample has loaded (combos populate from sample 1).
        self._session_restore = {
            'x': data.get('x_channel'),
            'y': data.get('y_channel'),
            'color': data.get('color_channel'),
            'active': data.get('active_sample'),
            'plot_enabled': {s['name']: s.get('plot_enabled', False)
                             for s in data.get('samples', [])},
        }

        # Restore samples. Processed samples (a CSV carrying computed columns —
        # clusters, UMAP, …) load on the Tk thread; the rest queue as raw FCS on
        # the background pool. EITHER way we paint a ⏳ placeholder row for every
        # sample FIRST, so resuming a session fills the tree immediately and
        # each row then swaps to its real entry as that sample finishes — a big
        # session no longer looks frozen while it loads.
        proc_names = {nm for nm, _ in processed_loads}
        if processed_loads:
            # openflo.pipeline → flowio; if the data deps aren't installed this
            # raises. Degrade to loading those samples as raw FCS rather than
            # crashing the restore (and the window) on startup.
            try:
                import pandas  # noqa: F401

                from . import pipeline  # noqa: F401
            except Exception as exc:
                print(f"[session] data deps unavailable, loading "
                      f"{len(processed_loads)} sample(s) as raw FCS instead: "
                      f"{exc}", flush=True)
                processed_loads = []
                proc_names = set()

        # Raw FCS paths (everything not loaded from a processed CSV). Missing
        # files are reported, not queued.
        paths, missing = [], []
        for s in data.get('samples', []):
            if s.get('name') in proc_names:
                continue
            p = s.get('path') or ''
            if p and os.path.isfile(p):
                paths.append(p)
            else:
                missing.append(s.get('name') or os.path.basename(p) or '?')

        # Record each processed sample's trial up front (keyed by its CSV path)
        # so its ⏳ row groups correctly the instant it's queued.
        for nm, csvp in processed_loads:
            ap = os.path.normcase(os.path.abspath(csvp))
            meta = self._pending_sample_meta.get(ap)
            if meta and meta.get('trial'):
                self._sample_trial[nm] = str(meta['trial'])
        # Queue BOTH raw FCS and processed sidecars on the bounded background
        # pool — every sample shows as a ⏳ row immediately and loads off the Tk
        # thread, so resuming a big session (even one with large processed
        # CSVs) never freezes the window. One grouped rebuild + repaint paints
        # the whole sample list at once.
        # The saved active sample loads first (priority 0) so its plot is what
        # appears soonest on resume.
        front = {data.get('active_sample')} if data.get('active_sample') else set()
        if paths:
            self._queue_fcs_loads(paths, front_names=front)
        if processed_loads:
            self._queue_processed_loads(processed_loads, front_names=front)
        self._refresh_gate_list()
        try:
            self.update_idletasks()
        except Exception:
            pass

        msg = f"Loading session: {len(paths) + len(processed_loads)} sample(s)"
        if missing:
            msg += f" — missing FCS for: {', '.join(missing[:4])}"
            if len(missing) > 4:
                msg += f" (+{len(missing) - 4})"
        self.status_var.set(msg)
        # Apply the deferred combo/active restore after the load queue
        # has had a chance to populate channels.
        self.after(600, self._apply_session_restore)

    def _apply_session_restore(self):
        """Second half of session restore: combo selections, plot-enabled
        toggles, active sample. Deferred so the first sample's channels
        have populated the combos."""
        info = getattr(self, '_session_restore', None)
        if not info:
            return
        # Make sure every loaded sample's columns (incl. restored UMAP/cluster)
        # are in the combo lists before re-selecting the saved axes — otherwise
        # the saved 'UMAP1'/'cluster' view silently fails to reopen.
        self._refresh_channel_choices()
        for name, on in info.get('plot_enabled', {}).items():
            if name in self._samples:
                self._sample_plot_enabled[name] = bool(on)
        for combo, key in ((self.x_combo, 'x'), (self.y_combo, 'y'),
                           (self.color_combo, 'color')):
            val = info.get(key)
            if val and val in combo['values']:
                combo.set(val)
        active = info.get('active')
        if active and active in self._samples:
            self._set_active_sample(active)
        self._session_restore = None
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _periodic_autosave(self):
        """Autosave the session every few minutes (primary window only), so a
        hard crash loses less than the close-time autosave alone. Reschedules
        itself; best-effort and silent."""
        try:
            if self._primary and self._samples:
                self._write_session(self._session_autosave_path())
        except Exception as exc:
            print(f"[session] periodic autosave failed: {exc}", flush=True)
        try:
            self.after(self._AUTOSAVE_MS, self._periodic_autosave)
        except Exception:
            pass

    def _push_recent_session(self, path):
        """Record a just-opened/saved session at the top of the recent list
        (deduped, most-recent-first, capped)."""
        try:
            ap = os.path.abspath(path)
            seen = os.path.normcase(ap)
            recent = [p for p in read_prefs().get('recent_sessions', [])
                      if isinstance(p, str) and os.path.normcase(p) != seen]
            recent.insert(0, ap)
            write_pref('recent_sessions', recent[:8])
        except Exception:
            pass

    @staticmethod
    def _recent_sessions():
        """Recent session paths that still exist on disk."""
        return [p for p in read_prefs().get('recent_sessions', [])
                if isinstance(p, str) and os.path.isfile(p)]

    def _fill_recent_menu(self, menu):
        """(Re)build the Open Recent submenu — called each time File opens."""
        menu.delete(0, 'end')
        recent = self._recent_sessions()
        if not recent:
            menu.add_command(label="(no recent sessions)", state='disabled')
            return
        for p in recent:
            menu.add_command(label=os.path.basename(p),
                             command=lambda q=p: self._load_session_path(q))
        menu.add_separator()
        menu.add_command(label="Clear recent",
                         command=lambda: write_pref('recent_sessions', []))
