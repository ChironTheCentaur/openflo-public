"""Session save / load / autosave + recent-sessions menu — editor mixin.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
import threading
import time
from tkinter import filedialog, messagebox
from typing import Any, cast

from . import paths as _paths
from . import session_lock as _lock
from .editor_base import EditorMixin
from .prefs import read_prefs, write_pref
from .session_format import comp_entry, comp_from_entry
from .session_sidecar import SIDECAR_EXT
from .ui_errors import show_error


def _report_nonfinite(where, value, replacement):
    print(f"[session] {where} was {value}, which JSON cannot represent — "
          f"stored as {replacement}", flush=True)


def _json_safe(obj, report=None, _where='state'):
    """Replace values RFC 8259 cannot carry, reporting each one.

    ``Infinity`` / ``-Infinity`` become the ±1e12 sentinel this codebase
    already uses for an open gate bound; ``NaN`` becomes ``null``, since a
    missing value is what it means. A numpy scalar becomes its Python value:
    an int64 a Leiden / FlowSOM population took from its column made every
    save of the session fail ("not JSON serializable"). Everything else is
    returned unchanged.
    """
    import numpy as np
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj):
            if report:
                report(_where, 'NaN', 'null')
            return None
        if math.isinf(obj):
            sentinel = -1e12 if obj < 0 else 1e12
            if report:
                report(_where, obj, sentinel)
            return sentinel
        return obj
    if isinstance(obj, dict):
        return {(k.item() if isinstance(k, np.generic) else k):
                _json_safe(v, report, f'{_where}.{k}')
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, report, f'{_where}[{i}]')
                for i, v in enumerate(obj)]
    return obj


def _frame_digest(df):
    """A digest of everything ``df.to_csv(index=False)`` writes — column
    names, dtypes and every value — or None when the frame can't be hashed
    (then the caller just writes it)."""
    import hashlib

    import pandas as pd
    # The stubs miss pandas.util's re-export; it is public API.
    hash_pandas_object = cast(Any, pd).util.hash_pandas_object
    try:
        h = hashlib.blake2b(digest_size=20)
        h.update(repr((list(map(str, df.columns)),
                       list(map(str, df.dtypes)), df.shape)).encode())
        h.update(hash_pandas_object(df, index=False).to_numpy()
                 .tobytes())
        return h.hexdigest()
    except Exception:                       # noqa: BLE001  unhashable cells
        return None


def _file_stamp(path):
    """(size, mtime) of `path`, or None when it is missing, so a sidecar
    deleted or replaced behind the editor's back is written again."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def _same_file(a, b):
    """Do paths `a` and `b` name the same file (by spelling)?"""
    return (os.path.normcase(os.path.abspath(a))
            == os.path.normcase(os.path.abspath(b)))


def _discard_quietly(path):
    """Remove `path` if it is there (nothing for '')."""
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _offline_parts(ed):
    """``(entries, gates_by_name)`` of editor `ed`'s offline samples (files
    missing when the session opened; SessionMixin._keep_offline) that no
    loaded sample has replaced."""
    off = getattr(ed, '_offline_samples', None) or {}
    loaded = getattr(ed, '_samples', None) or {}
    entries, gates = [], {}
    for name, rec in off.items():
        if name in loaded:
            continue
        entries.append(dict(rec['entry']))
        gates[name] = [dict(g) for g in rec['gates']]
    return entries, gates


def _copy_atomic(src, dst):
    """Copy `src` over `dst` through a temporary file, so `dst` is whole or
    unchanged, and is a new file (never the source written through)."""
    tmp = dst + '.tmp'
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def session_backup_path(path):
    """Where the previous save of the session at `path` is kept:
    ``name.backup.flowsession`` beside it, so it opens like any session."""
    stem, ext = os.path.splitext(path)
    return f"{stem}.backup{ext or '.flowsession'}"


def _session_file_readable(path):
    """True when `path` parses as an OpenFlo session (a JSON object with the
    format marker). A file cut short, empty, or not a session is False."""
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:                               # noqa: BLE001
        return False
    return isinstance(data, dict) and data.get('format') == 'openflo-session'


def _keep_session_backup(path):
    """Keep the previous save of a session the user named as
    `session_backup_path(path)`, so one bad save (or a file damaged later)
    never costs the only copy. Autosaves are skipped: there is one per run
    already. A previous file that does not read as a session is not kept, so
    it never replaces a good backup."""
    try:
        auto = os.path.normcase(os.path.abspath(os.path.join(
            os.path.expanduser('~'), '.openflo', 'autosave')))
        here = os.path.normcase(os.path.dirname(os.path.abspath(path)))
        if here == auto or not _session_file_readable(path):
            return
        backup = session_backup_path(path)
        tmp = backup + '.tmp'
        shutil.copyfile(path, tmp)
        os.replace(tmp, backup)
    except Exception as exc:                        # noqa: BLE001
        print(f"[session] backup of {path} not kept: {exc}", flush=True)


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
        # A crash or power cut can leave the newest autosave damaged (cut
        # short, or zero bytes). It used to be the only one offered, so a
        # damaged newest file hid every older one that was fine.
        for p in cands:
            if _session_file_readable(p):
                return p
        return None

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
                # Also drop the processed-data sidecar written alongside the
                # session (<stem>_data/); the old pruner removed only the
                # .flowsession file, leaking these (often large) event-table
                # CSV folders — one per app run — forever.
                data_dir = os.path.splitext(p)[0] + '_data'
                if os.path.isdir(data_dir):
                    shutil.rmtree(data_dir, ignore_errors=True)
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
                # A sample restored from a processed CSV keeps the raw FCS it
                # came from in `source_fcs`; record THAT, so the link to the
                # FCS survives a second save instead of pointing at the CSV.
                'path': (getattr(s, 'source_fcs', '')
                         or getattr(s, 'path', '') or ''),
                'color': self._sample_colors.get(name, '#1f77b4'),
                'plot_enabled': bool(self._sample_plot_enabled.get(name, False)),
                'trial': self._sample_trial.get(name, 'Trial'),
            }
            # Persist a manual Comps/Samples override only when set (so
            # name-based detection still applies to untouched samples on load).
            if name in self._sample_is_comp:
                entry['is_comp'] = bool(self._sample_is_comp[name])
            # Set aside from comparisons (with the reason) and the sample's
            # note: additive keys, which an older OpenFlo ignores.
            aside = getattr(self, '_sample_set_aside', {})
            notes = getattr(self, '_sample_notes', {})
            if name in aside:
                entry['set_aside'] = str(aside[name])
            if name in notes:
                entry['note'] = str(notes[name])
            # The transform each column carries. A sample reloaded from its
            # raw FCS comes back on the loader's logicle; without this, one
            # that landed first put the whole editor on logicle and gates
            # drawn in asinh coordinates selected nothing. Additive, inside
            # the sample entry (like basename / rel_path / processed_data), so
            # SESSION_KEYS and SESSION_VERSION are unchanged and an older
            # OpenFlo ignores it.
            rec = getattr(s, 'data_transforms', None)
            if rec is not None:
                entry['transforms'] = {str(c): dict(spec)
                                       for c, spec in rec.items()
                                       if c in s.data.columns}
            comp = comp_entry(s)
            if comp is not None:
                entry['comp'] = comp
            # Whether acquisition QC ran on it, so a reload from the raw FCS
            # gives the same events whatever Preferences says by then.
            entry['qc_applied'] = bool(getattr(s, 'qc_applied', False))
            # The channels this sample is the FMO control for (FMO gating),
            # so histograms can draw it again after a reopen. Additive, inside
            # the sample entry, as 'transforms' is.
            fmo_of = sorted(ch for ch, fmo in (getattr(
                self, '_fmo_controls', None) or {}).items() if fmo == name)
            if fmo_of:
                entry['fmo_of'] = fmo_of
            # Derived channels' formulas (derived.py), so a reload from the
            # raw FCS recomputes them. Additive, like 'transforms'.
            der = getattr(s, 'derived', None)
            if der:
                entry['derived'] = {str(c): str(f) for c, f in der.items()
                                    if c in s.data.columns}
            samples.append(entry)
        # Samples whose files were missing when the session opened: written
        # back as they were read, gates and all (_keep_offline).
        off_entries, off_gates = _offline_parts(self)
        samples.extend(off_entries)
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
        for name, gl in off_gates.items():
            if not sample_gates.get(name):
                sample_gates[name] = gl
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
            # v2: the tray's tabs (saved views) and the layout editor's pages.
            'view_tabs': (self._tabs_state()
                          if hasattr(self, '_tabs_state') else None),
            'layout': (self._layout_doc.to_dict()
                       if getattr(self, '_layout_doc', None) is not None
                       else None),
        }

    @staticmethod
    def _has_computed_columns(s):
        """True if a sample's data carries columns produced by analysis
        (clustering / embeddings / FMO gates / calibration) that aren't in the
        raw FCS — i.e. worth persisting so a reopened session keeps them.
        Also True for a sample that has no raw FCS to rebuild it from."""
        # A sample built FROM a CSV (a restored sidecar, 'Load in editor',
        # File > Load CSV) has no FCS behind it: its entry's path is that
        # CSV, which the restore reads as an FCS and fails. Its sidecar was
        # dropped on the next save whenever nothing else called for one --
        # a compensated sample restored from its sidecar has no
        # comp_channels -- so Save As or the autosave lost it (2.6.1 too).
        path = str(getattr(s, 'path', '') or '')
        if not path.lower().endswith('.fcs'):
            return True
        data = getattr(s, 'data', None)
        cols = set(data.columns) if data is not None else set()
        if cols & {'cluster', 'leiden', 'flowsom', 'flowsom_meta',
                   'pseudotime', 'cell_cycle'}:
            return True
        # A COMPENSATED sample carries no new column, but its values have been
        # transformed in place — so it differs from the raw FCS and must be
        # persisted too. Otherwise a reopened session reloads the RAW (un-
        # compensated) data and any gate/population drawn in compensated space
        # selects the wrong events (populations "don't reattach").
        if getattr(s, 'comp_channels', None):
            return True
        for c in cols:
            cu = str(c)
            # U: = spectral-unmixing abundances, which an uncompensated
            # sample (no other reason for a sidecar) lost on reload.
            if (cu.endswith('_pos') or cu.startswith('MESF:')
                    or cu.startswith(('U:', 'D:'))):
                return True
            up = cu.upper()
            for p in ('UMAP', 'TSNE', 'TRIMAP', 'PACMAP', 'PHATE'):
                if up.startswith(p) and up[len(p):] in ('1', '2'):
                    return True
        return False

    def _resolve_processed_csv(self, s, nm):
        """Absolute path to sample ``nm``'s processed-data sidecar (binary
        ``.npz``, or the CSV older sessions wrote), or '' if none exists.
        Prefers the session's recorded pointer (``processed_data``, or an
        older session's ``processed_csv``) resolved against the session dir;
        when that's missing or stale, falls back to the conventional
        ``<stem>_data/<safe>.npz`` / ``.csv`` location so the computed
        columns (clusters / UMAP) are still recovered even if the pointer was
        dropped — e.g. by a racing exit-autosave whose in-memory sample had
        already reloaded the raw FCS."""
        sess_dir = getattr(self, '_session_dir', '') or ''
        for key in ('processed_data', 'processed_csv'):
            # Normalise legacy Windows backslashes so a Win-authored pointer
            # relinks on Linux/macOS/web (matches the rel_path handling).
            pcsv = (s.get(key) or '').replace('\\', '/')
            if pcsv and sess_dir:
                # The pointer is ALWAYS app-generated relative to the session
                # dir. Reject an absolute or ..-escaping value (a crafted
                # session trying to read an arbitrary local file) — it must
                # resolve INSIDE sess_dir.
                cand = os.path.normpath(os.path.join(sess_dir, pcsv))
                root = os.path.normpath(sess_dir)
                if os.path.isabs(pcsv) or not (
                        cand == root or cand.startswith(root + os.sep)):
                    print(f"[session] ignoring out-of-tree {key}: {pcsv!r}",
                          flush=True)
                    pcsv = ''
                else:
                    pcsv = cand
            elif pcsv and not os.path.isabs(pcsv):
                pcsv = os.path.join(sess_dir, pcsv)
            if pcsv and os.path.isfile(pcsv):
                return pcsv
        data_dir = getattr(self, '_session_data_dir', '') or ''
        if data_dir:
            # Current stem first, then the pre-digest one. Sidecar names used
            # to be a plain character substitution, which was not injective
            # (two samples could be handed one file); they now carry a digest
            # of the original name. Sessions written before that change still
            # have the old files on disk, and this fallback is exactly the
            # path that recovers clusters/UMAP when the recorded pointer was
            # dropped — so it has to look for both.
            for stem in (self._sidecar_safe_name(nm),
                         _paths.legacy_sidecar_name(nm)):
                for ext in (SIDECAR_EXT, '.csv'):
                    guess = os.path.join(data_dir, stem + ext)
                    if os.path.isfile(guess):
                        return guess
        return ''

    def _resolve_sample_path(self, s):
        """Existing raw-FCS path for a session sample, relinking a moved/copied
        project. Tries the stored absolute ``path``, then ``rel_path`` resolved
        against the session dir, then ``basename`` in the session dir — so a
        project opened on another machine (or saved by the web app) finds its
        data once it sits next to the session file. ''/None if nothing exists."""
        p = s.get('path') or ''
        if p and os.path.isfile(p):
            return p
        sess_dir = getattr(self, '_session_dir', '') or ''
        if sess_dir:
            # Accept both new (forward-slash) and legacy (Windows backslash)
            # rel_path values so an old session still relinks on Linux/macOS.
            rel = (s.get('rel_path') or '').replace('\\', '/')
            if rel:
                cand = os.path.normpath(os.path.join(sess_dir, rel))
                root = os.path.normpath(sess_dir)
                if not (cand == root or cand.startswith(root + os.sep)):
                    # Legit projects may keep raw FCS in a sibling dir, so this is
                    # ALLOWED — but a session relinking OUTSIDE its own tree is
                    # worth flagging (a crafted session could point at any local
                    # file). Warn; don't block valid cross-dir projects.
                    print(f"[session] note: rel_path resolves outside the session "
                          f"dir: {cand}", flush=True)
                if os.path.isfile(cand):
                    return cand
            base = s.get('basename') or (os.path.basename(p) if p else '')
            if base:
                cand = os.path.join(sess_dir, base)
                if os.path.isfile(cand):
                    return cand
        return p if p else ''

    def _write_session(self, path):
        """Core writer — shared by Save Session… and autosave. Also writes a
        processed-data sidecar (``<stem>_data/<name>.csv``) for any sample
        carrying computed columns (clusters / UMAP / FMO gates), and records its
        relative path on the sample entry — so reopening the session restores
        those results instead of re-reading the bare raw FCS."""
        data, failures = self._run_session_write(
            self._plan_session_write(path))
        self._sidecar_failures = failures
        return data

    def _plan_session_write(self, path):
        """The Tk-thread half of a session write: snapshot the session state
        (it reads Tk variables) and pick the samples that need a sidecar. The
        returned plan is all `_run_session_write` needs, so that half can run
        on a worker thread while the window keeps answering."""
        data = self._session_state()
        sess_dir = os.path.dirname(os.path.abspath(path))
        jobs = []
        for entry in data.get('samples', []):
            # Portable relink hints: basename + a path relative to the session
            # file, so a moved/copied project (or one opened on another machine,
            # e.g. via the web app) can re-find its raw FCS even when the stored
            # absolute path no longer exists. Additive — inside the sample entry,
            # so the locked top-level SESSION_KEYS are unchanged.
            ep = entry.get('path') or ''
            entry['basename'] = os.path.basename(ep) if ep else ''
            if ep:
                try:
                    # Store with FORWARD slashes so a Windows-authored session
                    # relinks on Linux/macOS/web (a stored 'raw\\x.fcs' would be
                    # a literal filename char there). Normalised back on read.
                    entry['rel_path'] = os.path.relpath(
                        ep, sess_dir).replace(os.sep, '/')
                except ValueError:        # different drive on Windows → no rel
                    pass
            s = self._samples.get(entry.get('name'))
            if s is None or not self._has_computed_columns(s):
                continue
            # Snapshot what the writer reads, so it can run on a worker while
            # the window stays live: a shallow copy is a new frame over the
            # same column arrays, so an analysis adding or replacing columns
            # on the sample meanwhile doesn't change what this save writes.
            rec = getattr(s, 'data_transforms', None)
            jobs.append({'entry': entry, 'name': entry['name'],
                         'data': s.data.copy(deep=False),
                         'transforms': None if rec is None else dict(rec),
                         'source': str(getattr(s, 'path', '') or ''),
                         'safe': self._sidecar_safe_name(entry['name'])})
        return {'path': path, 'data': data, 'jobs': jobs}

    def _run_session_write(self, plan, progress=None):
        """Write the sidecars and the session file a `_plan_session_write`
        plan describes. Touches no Tk state, so the close runs it on a worker
        thread. ``progress(done, total, name)`` is called before each step.
        Returns ``(data, sidecar_failures)``."""
        # One writer at a time: the periodic autosave runs on a worker, and
        # a close (or Save) can start while it is still writing. They may
        # write the same files, so the later one waits for the earlier.
        lock = self.__dict__.setdefault('_session_write_lock',
                                        threading.Lock())
        with lock:
            return self._run_session_write_locked(plan, progress)

    def _run_session_write_locked(self, plan, progress):
        path, data, jobs = plan['path'], plan['data'], plan['jobs']
        data_dir = os.path.splitext(path)[0] + '_data'
        total = len(jobs) + 1               # + the session file itself
        made_dir = False
        # Track samples whose processed-data sidecar failed to write: those
        # samples reload from the bare raw FCS with their computed columns
        # (clusters / UMAP / FMO gates) GONE. Surfaced by _save_session so the
        # save doesn't silently report full success while dropping analysis.
        sidecar_failures = []
        used_stems: dict[str, str] = {}     # sidecar stem -> sample that took it
        for i, job in enumerate(jobs):
            entry, safe, df = job['entry'], job['safe'], job['data']
            if progress is not None:
                progress(i, total, entry['name'])
            # Two samples must never share a sidecar. `safe_sidecar_name` is
            # injective now, but this loop is where the damage happened —
            # the file was overwritten and the SAME path recorded on both
            # entries, and restore prefers the sidecar over the raw FCS, so
            # one sample silently came back holding the other's events. Belt
            # and braces: if a stem ever repeats again, skip rather than
            # overwrite, and report it like any other sidecar failure.
            if safe in used_stems:
                print(f"[session] processed sidecar for {entry['name']} "
                      f"skipped: name collides with "
                      f"{used_stems[safe]!r} on disk", flush=True)
                sidecar_failures.append(entry['name'])
                continue
            used_stems[safe] = entry['name']
            try:
                if not made_dir:
                    os.makedirs(data_dir, exist_ok=True)
                    made_dir = True
                stem = os.path.join(data_dir, safe)
                # The transform each column carries, beside the sidecar rather
                # than in the session JSON: the restore reads it with the
                # sidecar (as File > Load CSV does with a CSV), the locked
                # SESSION_KEYS and SESSION_VERSION are untouched, and an older
                # OpenFlo simply ignores the extra file. An earlier save's
                # record goes BEFORE the sidecar is rewritten, so a failed
                # write leaves none. Both formats share it (it is named from
                # the stem).
                from .pipeline import (
                    discard_transforms_sidecar,
                    write_transforms_sidecar,
                )
                discard_transforms_sidecar(stem + SIDECAR_EXT)
                side_path = self._write_sidecar(df, stem, keep=job['source'])
                rec = job['transforms']
                if rec is not None:
                    write_transforms_sidecar(side_path, rec, df.columns,
                                             len(df))
                entry['processed_data'] = os.path.join(
                    os.path.basename(data_dir),
                    os.path.basename(side_path)).replace(os.sep, '/')
            except Exception as exc:
                print(f"[session] processed sidecar for {entry['name']} "
                      f"failed: {exc}", flush=True)
                sidecar_failures.append(entry['name'])
        if progress is not None:
            progress(len(jobs), total, '')
        # Atomic write: dump to a temp file then replace, so a mid-write failure
        # (disk full, or an un-serialisable object in the state) can't truncate
        # and corrupt the existing good session — or the resumable autosave that
        # resume depends on.
        # `Infinity` and `NaN` are Python extensions that RFC 8259 does not
        # allow. json.dump writes them happily and json.load reads them back,
        # so a session containing one is valid HERE and unreadable to
        # JSON.parse, jq, or any other consumer — a failure that could only
        # ever surface somewhere else. Repair-and-report rather than raise:
        # refusing to save would cost the user their work over a value we can
        # represent perfectly well.
        data = cast('dict[str, Any]', _json_safe(data, _report_nonfinite))
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        _keep_session_backup(path)
        os.replace(tmp, path)
        return data, sidecar_failures

    def _write_sidecar(self, df, stem, keep=''):
        """Write `df` as the processed-data sidecar ``<stem>.npz`` (or
        ``<stem>.csv`` when it holds a column the binary format can't carry;
        see session_sidecar) and return the path written. Removes the other
        format's file of the same stem, so a restore can never pick up a
        stale table beside the current one -- unless that file is `keep`,
        the one the sample was loaded from (it may still be read).

        Skips the work when it can:

        * this editor already wrote exactly these values to that path and
          the file is untouched since (every autosave after the first);
        * a sidecar this editor loaded or wrote ELSEWHERE holds exactly
          these values and is untouched since -- the session it was opened
          from, which the autosave and the close write beside a different
          file. That file is copied rather than re-encoded.

        Hashing the frame costs a small fraction of encoding it."""
        from .session_sidecar import LEGACY_EXT, frame_arrays, write_frame
        digest = _frame_digest(df)
        binary = frame_arrays(df) is not None
        path = stem + (SIDECAR_EXT if binary else LEGACY_EXT)
        other = stem + (LEGACY_EXT if binary else SIDECAR_EXT)
        if keep and _same_file(other, keep):
            other = ''
        key = os.path.normcase(os.path.abspath(path))
        written = self.__dict__.setdefault('_sidecar_written', {})
        stamp = _file_stamp(path)
        if (digest is not None and stamp is not None
                and written.get(key) == (digest, stamp)):
            _discard_quietly(other)
            return path
        written.pop(key, None)              # stale until the write completes
        src = self._known_sidecar(digest) if binary else None
        if src is not None and os.path.normcase(src) != key:
            _copy_atomic(src, path)
        elif binary:
            write_frame(df, path)
        else:
            # Whole or not at all, like the binary format: a crash mid-write
            # left a CSV cut short that the next restore loaded as the
            # sample (401 of 1,000 events).
            tmp = path + '.tmp'
            try:
                df.to_csv(tmp, index=False)
                os.replace(tmp, path)
            except BaseException:
                _discard_quietly(tmp)
                raise
        _discard_quietly(other)
        if digest is not None:
            self._remember_sidecar(path, digest)
        return path

    def _remember_sidecar(self, path, digest):
        """Record that the sidecar at `path` holds the frame `digest`
        describes, as it is on disk now. Safe from a load worker: two plain
        dict stores."""
        key = os.path.normcase(os.path.abspath(path))
        stamp = _file_stamp(path)
        if stamp is None:
            return
        self.__dict__.setdefault('_sidecar_written', {})[key] = (digest,
                                                                 stamp)
        if key.endswith(SIDECAR_EXT):
            self.__dict__.setdefault('_sidecar_by_digest', {})[digest] = (
                key, stamp)

    def _known_sidecar(self, digest):
        """A binary sidecar on disk known to hold the frame `digest`
        describes, unchanged since, or None."""
        if digest is None:
            return None
        hit = self.__dict__.get('_sidecar_by_digest', {}).get(digest)
        if hit is None or _file_stamp(hit[0]) != hit[1]:
            return None
        return hit[0]

    def _open_session_file(self):
        """The session file this editor was opened from or last saved to,
        while it still exists; None otherwise (a new or resumed session)."""
        path = self.__dict__.get('_session_file')
        if path and os.path.isfile(path):
            return path
        return None

    def _remember_session_file(self, path, read_only=None):
        """Record `path` as the file Save writes to, and name it in the
        window title. None forgets it. The lock on the file this window held
        before is let go, and the file's stamp is kept, so a save can tell
        when someone else saved it in between (session_lock).

        `read_only` names a session opened read-only: Save asks for a new
        name, and the title says so."""
        old = self.__dict__.get('_session_file')
        if old and not (path and _same_file(old, path)):
            _lock.release(old, self._session_lock_token())
        self._session_file = path
        self._session_stamp = _lock.file_stamp(path) if path else None
        self._session_read_only = read_only
        self._refresh_title()

    def _window_title(self):
        """"session name — OpenFlo", with a * before the name while there
        are changes since the last save or open (UI 30)."""
        path = self.__dict__.get('_session_file')
        read_only = self.__dict__.get('_session_read_only')
        if path:
            name = os.path.basename(path)
        elif read_only:
            name = f"{os.path.basename(read_only)} (read-only)"
        else:
            name = "Untitled session"
        mark = '*' if self.__dict__.get('_unsaved') else ''
        return f"{mark}{name} — OpenFlo"

    def _refresh_title(self):
        try:
            self.title(self._window_title())
        except Exception:                           # noqa: BLE001
            pass

    def _note_unsaved(self):
        """Something the session saves changed: mark the title."""
        if not self.__dict__.get('_unsaved'):
            self._unsaved = True
            self._refresh_title()

    def _note_saved(self):
        """The session was just saved or opened: clear the mark."""
        self._unsaved = False
        self._refresh_title()

    def _session_lock_token(self):
        """This window's token in session lock files (session_lock)."""
        tok = self.__dict__.get('_lock_token')
        if tok is None:
            tok = self._lock_token = _lock.new_token()
        return tok

    def _release_session_lock(self):
        """Let go of the lock on the session this window has open (on
        close)."""
        own = self.__dict__.get('_session_file')
        if own:
            _lock.release(own, self._session_lock_token())

    def _ask_choice(self, title, text, choices, default=0):
        """A small modal question with one button per ``(label, value)``
        in `choices`; returns the value picked. Escape or closing the window
        returns the last choice's value (Cancel, by convention)."""
        import tkinter as tk
        from tkinter import ttk
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.resizable(False, False)
        try:
            dlg.transient(self.winfo_toplevel())
        except Exception:                           # noqa: BLE001
            pass
        cancel = choices[-1][1]
        picked = {'v': cancel}
        body = ttk.Frame(dlg, padding=16)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text=text, justify='left',
                  wraplength=440).pack(anchor='w')
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(14, 0))

        def pick(v):
            picked['v'] = v
            dlg.destroy()
        buttons = []
        for label, value in choices:
            b = ttk.Button(bb, text=label,
                           command=lambda v=value: pick(v))
            b.pack(side='left', padx=(6, 0))
            buttons.append(b)
        dlg.bind('<Escape>', lambda _e: pick(cancel))
        dlg.protocol('WM_DELETE_WINDOW', lambda: pick(cancel))
        try:
            buttons[default].focus_set()
            dlg.grab_set()
        except Exception:                           # noqa: BLE001
            pass
        dlg.wait_window()
        return picked['v']

    def _confirm_session_target(self, path, closing=False):
        """Before writing the session to `path`, make sure that doesn't
        overwrite someone else's work. Returns the path to write (`path`,
        or a copy's name the user picked) or None to write nothing.

        The file this window opened or last saved is checked against its
        stamp: a change means another window or person saved it since, and
        the user picks Save as a copy, Overwrite or Cancel. Any other file is
        checked for a lock held by another window."""
        own = self.__dict__.get('_session_file')
        name = os.path.basename(path)
        if own and _same_file(own, path):
            if not _lock.changed_on_disk(path, self.__dict__.get(
                    '_session_stamp')):
                return path
            gone = not os.path.exists(path)
            what = ("was deleted or moved" if gone else
                    "was changed by another window or person")
            choice = self._ask_choice(
                "Session changed on disk",
                f"{name} {what} after this window opened or last saved "
                "it. Saving here would replace that version.\n\n"
                "Save as a copy keeps both. Overwrite replaces the file "
                "with this window's session."
                + ("\n\nCancel keeps OpenFlo open." if closing else ""),
                [("Save as a copy", 'copy'), ("Overwrite", 'overwrite'),
                 ("Cancel", 'cancel')])
            self._audit('session.disk_conflict', path=path, choice=choice)
            if choice == 'overwrite':
                return path
            if choice != 'copy':
                return None
            stem, ext = os.path.splitext(name)
            copy = filedialog.asksaveasfilename(
                title="Save a copy of the session",
                defaultextension=self.SESSION_EXT,
                initialdir=os.path.dirname(os.path.abspath(path)),
                initialfile=f"{stem} (copy){ext or self.SESSION_EXT}",
                filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                           ('All files', '*.*')])
            if not copy or _same_file(copy, path):
                return None if not copy else path
            return self._confirm_session_target(copy, closing)
        held = _lock.read_lock(path)
        if (held is None or held.get('token') == self._session_lock_token()
                or _lock.is_stale(held) or _lock.held_here(held)):
            return path
        choice = self._ask_choice(
            "Session open elsewhere",
            f"{name} is open in {_lock.describe_holder(held)}. Saving "
            "here would replace the version that window saves.",
            [("Overwrite", 'overwrite'), ("Cancel", 'cancel')], default=1)
        return path if choice == 'overwrite' else None

    def _save_session_as(self):
        """Save under a new name, which later saves then write to."""
        self._save_session(save_as=True)

    def _save_session(self, save_as=False):
        if not self._samples:
            self.status_var.set("Load at least one sample before saving a session.")
            return
        path = None if save_as else self._open_session_file()
        if path is None:
            own = (self.__dict__.get('_session_file')
                   or self.__dict__.get('_session_read_only'))
            path = filedialog.asksaveasfilename(
                title="Save editor session",
                defaultextension=self.SESSION_EXT,
                initialfile=(os.path.basename(own) if own
                             else 'session' + self.SESSION_EXT),
                filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                           ('All files', '*.*')])
        if not path:
            return
        path = self._confirm_session_target(path)
        if not path:
            self.status_var.set("Session not saved.")
            return
        try:
            data = self._write_session(path)
            _lock.acquire(path, self._session_lock_token(), force=True)
            self._remember_session_file(path)
            self._note_saved()
            self._push_recent_session(path)
            failed = getattr(self, '_sidecar_failures', []) or []
            if failed:
                # The session JSON saved, but analysis columns for these samples
                # did NOT — warn instead of reporting clean success.
                self.status_var.set(
                    f"Saved session with WARNINGS → {os.path.basename(path)} "
                    f"({len(failed)} sample(s) lost computed results)")
                bullets = '\n'.join(f"  • {nm}" for nm in failed)
                messagebox.showwarning(
                    "Session saved — analysis not fully written",
                    "The session file was saved, but the processed-data sidecar "
                    "failed for these sample(s), so their computed columns "
                    "(clusters / UMAP / FMO gates) will NOT be restored on "
                    "reload — they'll fall back to the raw FCS:\n\n"
                    f"{bullets}\n\n"
                    "Check disk space / folder permissions and re-save.",
                    parent=self)
            else:
                self.status_var.set(
                    f"Saved session: {len(data['samples'])} sample(s) → "
                    f"{os.path.basename(path)}")
        except Exception as exc:
            self.status_var.set(f"Save session failed: {exc}")
            show_error(self, "Save session failed",
                       "The session could not be saved.", exc, path)

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load editor session",
            filetypes=[('OpenFlo session or project',
                        '*' + self.SESSION_EXT + ' *.zip'),
                       ('OpenFlo session', '*' + self.SESSION_EXT),
                       ('OpenFlo project archive', '*.zip'),
                       ('All files', '*.*')])
        if not path:
            return
        self._load_session_path(path)

    def _export_project_archive(self):
        """File > Export project archive: one zip with the session, its
        saved data and every sample's FCS file, to send or move as one."""
        if not self._samples:
            self.status_var.set(
                "Load at least one sample before exporting a project.")
            return
        from .project_archive import ARCHIVE_EXT, write_project_archive
        own = self._open_session_file()
        stem = (os.path.splitext(os.path.basename(own))[0] if own
                else 'project')
        out = filedialog.asksaveasfilename(
            title="Export project archive",
            defaultextension=ARCHIVE_EXT, initialfile=stem + ARCHIVE_EXT,
            filetypes=[('OpenFlo project archive', '*' + ARCHIVE_EXT),
                       ('All files', '*.*')])
        if not out:
            return
        import tempfile
        self.status_var.set("Exporting project archive…")
        try:
            self.update_idletasks()
        except Exception:                           # noqa: BLE001
            pass
        tmp = tempfile.mkdtemp(prefix='openflo-project-')
        try:
            sess = os.path.join(tmp, 'project' + self.SESSION_EXT)
            self._write_session(sess)
            manifest = write_project_archive(sess, out)
        except Exception as exc:                    # noqa: BLE001
            self.status_var.set(f"Project export failed: {exc}")
            show_error(self, "Project export failed",
                       "The project could not be exported.", exc, out)
            return
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._audit('project.export', path=out, samples=manifest['samples'],
                    missing=manifest['missing'])
        n = len(manifest['samples'])
        self.status_var.set(
            f"Exported project: {n} sample(s) → {os.path.basename(out)}")
        if manifest['missing']:
            messagebox.showwarning(
                "Project exported without some data files",
                "These samples' FCS files were not found, so the archive "
                "holds their gates but not their events:\n\n  "
                + "\n  ".join(manifest['missing'][:10])
                + ("\n  …" if len(manifest['missing']) > 10 else ""),
                parent=self)

    def _open_project_archive_path(self, path):
        """Unpack a project archive beside itself and open its session."""
        from .project_archive import open_project_archive
        try:
            sess, manifest = open_project_archive(path)
        except Exception as exc:                    # noqa: BLE001
            self.status_var.set(f"Open project failed: {exc}")
            show_error(self, "Can't open project",
                       "The project could not be opened.", exc, path)
            return
        self._audit('project.open', path=path, folder=os.path.dirname(sess))
        self._load_session_path(sess)
        self.status_var.set(
            f"Opened project archive {os.path.basename(path)}; its files "
            f"are in {os.path.dirname(sess)}")

    def _compare_saved_session(self):
        """File → Compare with saved session: every population's count with
        this session's gates beside the saved session's gates, both on the
        events loaded now (session_compare)."""
        if not any(n in self._samples for n in self._sample_order):
            self.status_var.set("Load samples (or open a session) first, "
                                "then compare it with a saved one.")
            return
        path = filedialog.askopenfilename(
            title="Compare with saved session",
            filetypes=[('OpenFlo session', '*' + self.SESSION_EXT),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            if data.get('format') != 'openflo-session':
                raise ValueError("not an OpenFlo session (no format marker)")
            from .session_format import migrate_session
            data, _notes = migrate_session(data)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror("Compare with saved session",
                                 f"Could not read {os.path.basename(path)}:"
                                 f"\n{exc}", parent=self)
            return
        current = {}
        for n in self._sample_order:
            s = self._samples.get(n)
            if s is None:
                continue
            current[n] = (dict(self._sample_gates.get(n) or {}), s.data,
                          getattr(s, 'data_transforms', None))
        from .async_task import run_async
        from .session_compare import compare_sessions
        self.status_var.set(f"Comparing with {os.path.basename(path)}…")
        run_async(self, lambda: compare_sessions(current, data),
                  on_done=lambda rows: self._show_session_compare(rows, path),
                  on_error=lambda exc: self.status_var.set(
                      f"Compare with saved session failed: "
                      f"{type(exc).__name__}: {exc}"))

    def _show_session_compare(self, rows, path):
        from .session_compare import COLUMNS
        from .ui_table import TableWindow
        base = os.path.basename(path)
        if not rows:
            self.status_var.set(f"{base} has none of the samples loaded "
                                "now, so there is nothing to compare.")
            return
        moved = [r for r in rows if r['moved']]
        samples = len({r['Sample'] for r in rows})
        status = (f"{len(rows)} population(s) in {samples} sample(s); "
                  + (f"{len(moved)} differ (highlighted)." if moved
                     else "every count matches."))
        self.status_var.set(f"Compared with {base}: " + status)
        self._audit('session.compare', path=path, n_rows=len(rows),
                    n_moved=len(moved))
        TableWindow(
            self, f"Compare with {base}", COLUMNS, rows,
            note=("Each population counted with this session's gates and "
                  f"with the gates saved in {base}, both on the events "
                  "loaded now, so a difference comes from the gates. "
                  "Samples are matched by name and populations by their "
                  "place in the tree and name."),
            status=status, csv_name='session_comparison.csv',
            flag_key='moved')

    def _load_session_path(self, path):
        from .project_archive import is_project_archive
        if is_project_archive(path):
            self._open_project_archive_path(path)
            return
        backup = path.endswith('.backup' + self.SESSION_EXT)
        read_only = None
        if not backup:
            held = _lock.acquire(path, self._session_lock_token())
            if held is not None:
                choice = self._ask_choice(
                    "Session open elsewhere",
                    f"{os.path.basename(path)} is already open in "
                    f"{_lock.describe_holder(held)}.\n\nOpen read-only "
                    "to look without changing the file (Save then asks for "
                    "a new name). Open anyway if that window is no longer "
                    "in use; OpenFlo still asks before overwriting a save "
                    "made there.",
                    [("Open read-only", 'readonly'),
                     ("Open anyway", 'anyway'), ("Cancel", 'cancel')])
                self._audit('session.open_locked', path=path, choice=choice)
                if choice == 'cancel':
                    return
                if choice == 'anyway':
                    _lock.acquire(path, self._session_lock_token(),
                                  force=True)
                else:
                    read_only = path
        stamp = _lock.file_stamp(path)
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            if not read_only and not backup:
                _lock.release(path, self._session_lock_token())
            self.status_var.set(f"Load session failed: {exc}")
            self._report_unreadable_session(path, exc)
            return
        if data.get('format') != 'openflo-session':
            if not read_only and not backup:
                _lock.release(path, self._session_lock_token())
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
            if not read_only and not backup:
                _lock.release(path, self._session_lock_token())
            messagebox.showerror("Can't open session", str(exc), parent=self)
            return
        # A fresh session starts a fresh history — undo shouldn't cross
        # back into the previous session's gates.
        self._undo_stack.clear()
        self._redo_stack.clear()
        # Dir of the session file — used to resolve relative processed-data
        # sidecars during restore. _session_data_dir is the conventional
        # sidecar folder (<stem>_data), used as a fallback when a sample's
        # processed_data pointer is missing.
        self._session_dir = os.path.dirname(os.path.abspath(path))
        self._session_data_dir = os.path.splitext(os.path.abspath(path))[0] + '_data'
        self._apply_session(data)
        self._push_recent_session(path)
        # Save writes back to the file opened -- but never over the backup
        # copy, which keeps the earlier save until the user picks a name,
        # nor over a file opened read-only.
        own = None if (backup or read_only) else path
        self._remember_session_file(own, read_only=read_only)
        if own:
            # As read, not as it is now: a save made elsewhere while this
            # session was being applied is still caught.
            self._session_stamp = stamp
        self._note_saved()
        if _mig_notes:
            self._audit('session.migrate', steps=_mig_notes)
            self.status_var.set(
                "Opened a session from an older OpenFlo and upgraded its "
                "format — re-save (Ctrl+S) to keep it current.")

    def _report_unreadable_session(self, path, exc):
        """Say plainly that the session file can't be read, and offer the
        backup of its previous save when one reads."""
        name = os.path.basename(path)
        if isinstance(exc, OSError):
            why = f"OpenFlo couldn't open {name}: {exc.strerror or exc}."
        else:
            why = (f"{name} is damaged or incomplete, so OpenFlo can't read "
                   "it. This happens when a save is interrupted, or the file "
                   "was cut short when it was copied.")
        backup = session_backup_path(path)
        if (os.path.abspath(path) != os.path.abspath(backup)
                and _session_file_readable(backup)):
            when = time.strftime('%Y-%m-%d %H:%M',
                                 time.localtime(os.path.getmtime(backup)))
            if messagebox.askyesno(
                    "Session can't be read",
                    f"{why}\n\nOpenFlo kept the previous save of this "
                    f"session ({when}). Open that instead?\n\n"
                    f"Backup: {os.path.basename(backup)}", parent=self):
                self._audit('session.open_backup', path=path, backup=backup)
                self._load_session_path(backup)
            return
        show_error(
            self, "Session can't be read",
            f"{why}\n\nNo backup of an earlier save was found.", exc, path)

    def _apply_session(self, data):
        """Restore editor state from a parsed session dict.

        Display state is restored immediately. Samples are loaded
        asynchronously (same threaded path as Add-FCS); their gates are
        staged in `_pending_sample_gates` and applied by `_on_loaded`
        as each FCS finishes parsing — reusing the WSP-ingest mechanism.
        """
        # Restore global display config up front (independent of samples).
        self._fmo_controls = {}         # this session's own, from its samples
        self._channel_scale.update(
            {k: str(v) for k, v in (data.get('channel_scale') or {}).items()})
        for k, v in (data.get('channel_range') or {}).items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:                            # skip a corrupt range, don't
                    self._channel_range[k] = (float(v[0]), float(v[1]))
                except (TypeError, ValueError):  # abort the whole session load
                    pass
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
        # v2: the tray's tabs and the layout editor's pages.
        try:
            self._restore_tabs_state(data.get('view_tabs'))
        except Exception as exc:                      # noqa: BLE001
            print(f"[session] tabs not restored: {exc}", flush=True)
        if data.get('layout'):
            try:
                from .layout_page import LayoutDoc
                self._layout_doc = LayoutDoc.from_dict(data['layout'])
            except Exception as exc:                  # noqa: BLE001
                print(f"[session] layout not restored: {exc}", flush=True)
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
        getattr(self, '_pending_sample_prep', {}).clear()
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

        missing = self._restore_samples(data.get('samples', []),
                                        data.get('sample_gates') or {},
                                        data.get('active_sample'))
        if missing:
            # The status line is overwritten by the first sample that loads;
            # say it where it stays, and offer to find the files.
            self.after(800, self._offer_locate_missing)
        # Apply the deferred combo/active restore after the load queue
        # has had a chance to populate channels.
        self.after(600, self._apply_session_restore)

    # ── Samples whose files are missing ──────────────────────────────────────

    def _keep_offline(self, entry, gates):
        """Keep a session sample whose file was not found, with its gates, so
        the next save (or autosave) writes it back unchanged. It used to be
        skipped, and the save after it dropped the sample and every gate on
        it from the session for good."""
        e = dict(entry)
        # A sidecar pointer is relative to the session it came from and was
        # not found either; the raw-FCS pointers are what relinking needs.
        for key in ('processed_data', 'processed_csv'):
            e.pop(key, None)
        p = e.get('path') or ''
        if p and not os.path.isabs(p):
            e['path'] = os.path.abspath(os.path.join(
                getattr(self, '_session_dir', '') or '', p))
        self.__dict__.setdefault('_offline_samples', {})[e['name']] = {
            'entry': e, 'gates': [dict(g) for g in gates or []]}

    def _offline_session_parts(self):
        """``(entries, gates_by_name)`` of the offline samples no loaded
        sample has replaced, for _session_state."""
        return _offline_parts(self)

    def _relink_missing(self, folder):
        """Look for the offline samples' files under `folder` (and its
        subfolders) by file name, and load the ones found with their saved
        gates. Returns ``(found, still_missing)`` sample names."""
        off = getattr(self, '_offline_samples', None) or {}
        want = {}
        for name, rec in off.items():
            if name in self._samples:
                continue
            e = rec['entry']
            base = e.get('basename') or os.path.basename(e.get('path') or '')
            if base:
                want.setdefault(base.lower(), []).append(name)
        hits = {}
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                for name in want.get(f.lower(), ()):
                    hits.setdefault(name, os.path.join(root, f))
            if len(hits) == sum(len(v) for v in want.values()):
                break
        samples, gates = [], {}
        for name, path in hits.items():
            rec = off.pop(name)
            e = dict(rec['entry'], path=path)
            e.pop('rel_path', None)
            samples.append(e)
            gates[name] = rec['gates']
        if samples:
            self._restore_samples(samples, gates)
            self._audit('session.relink', folder=folder,
                        samples=sorted(hits))
        still = sorted(n for n in off if n not in self._samples)
        return sorted(hits), still

    def _offer_locate_missing(self):
        """Name the samples whose files were not found and offer to look for
        them in a folder the user picks."""
        _, gates = self._offline_session_parts()
        names = sorted(gates)
        if not names:
            return
        from tkinter import filedialog, messagebox
        shown = '\n'.join(f"  • {n}" for n in names[:12])
        more = f"\n  … and {len(names) - 12} more" if len(names) > 12 else ''
        if not messagebox.askyesno(
                "Sample files not found",
                f"The files for {len(names)} sample(s) were not found:\n\n"
                f"{shown}{more}\n\nTheir gates are kept and saved with the "
                "session. Look for the files in a folder now?", parent=self):
            return
        folder = filedialog.askdirectory(
            parent=self, title="Folder holding the missing sample files")
        if not folder:
            return
        found, still = self._relink_missing(folder)
        if still:
            messagebox.showwarning(
                "Some files still missing",
                f"Found {len(found)} of {len(found) + len(still)}. Not found "
                f"in that folder: {', '.join(still[:8])}"
                + (" …" if len(still) > 8 else '')
                + "\n\nTheir gates stay in the session.", parent=self)
        else:
            self.status_var.set(f"Found all {len(found)} missing sample "
                                "file(s); loading them")

    def _restore_samples(self, samples, sample_gates, active=None):
        """Stage and queue a session's samples (each one's grouping, gates,
        transforms and matrix, keyed by its file path) and return the names
        whose files could not be found. Those are kept as offline samples
        (_keep_offline), so a save does not drop them."""
        processed_loads = []          # [(name, csv_path)] — restored with cols
        for s in samples:
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
                load_path = self._resolve_sample_path(s)
            if not load_path:
                continue
            pkey = os.path.normcase(os.path.abspath(load_path))
            m: dict[str, object] = {'gates': list(sample_gates.get(nm, []))}
            if s.get('trial'):
                m['trial'] = s['trial']
            if 'is_comp' in s:
                m['is_comp'] = bool(s['is_comp'])
            if isinstance(s.get('fmo_of'), list):
                self._set_fmo_controls({str(ch): nm for ch in s['fmo_of']})
            if 'transforms' in s:
                from .pipeline import transform_record_from_json
                try:
                    m['transforms'] = transform_record_from_json(
                        s['transforms'])
                except ValueError as exc:
                    print(f"[session] {nm}: transform record ignored "
                          f"({exc})", flush=True)
            if s.get('derived'):
                m['derived'] = dict(s['derived'])
            self._pending_sample_meta[pkey] = m
            # The matrix this sample was compensated with (a raw-FCS reload
            # re-applies it instead of $SPILL; a processed CSV, already
            # compensated, gets it re-attached) and, for a CSV, its raw FCS.
            fcs = self._resolve_sample_path(s) if pcsv else ''
            # A session written before 'qc_applied' existed was loaded with
            # QC (it then always ran), so that is what a raw reload repeats.
            self.stage_sample_prep(
                load_path, comp=comp_from_entry(s.get('comp')),
                source_fcs=(fcs if fcs and fcs.lower().endswith('.fcs')
                            else None),
                qc=bool(s.get('qc_applied', True)))

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
        # files are reported and kept offline, not queued.
        paths, missing = [], []
        for s in samples:
            if s.get('name') in proc_names:
                continue
            p = self._resolve_sample_path(s)        # relink moved/copied projects
            if p and os.path.isfile(p):
                paths.append(p)
            else:
                missing.append(s.get('name')
                               or s.get('basename')
                               or os.path.basename(s.get('path') or '') or '?')
                self._keep_offline(s, sample_gates.get(s.get('name'), []))

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
        front = {active} if active else set()
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
        return missing

    def _apply_session_restore(self, _tries=0):
        """Second half of session restore: combo selections, plot-enabled
        toggles, active sample. Deferred so channels have populated the combos —
        and RE-APPLIED while samples are still loading, since a slow resume can
        finish long after the initial 600 ms one-shot (a late-loading sample
        would otherwise keep its default unchecked plot state / a saved axis on
        its late column wouldn't be re-selected)."""
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
        if active and active in self._samples and self._active_sample != active:
            self._set_active_sample(active)
        # Still loading? Re-apply shortly so late samples pick up their state.
        # Bounded (~9s past the 600 ms) so a stuck load can't loop forever.
        if getattr(self, '_loading', None) and _tries < 30:
            self.after(300, lambda: self._apply_session_restore(_tries + 1))
            return
        self._session_restore = None
        # plot_enabled is only correct now, and the propagate floor is taken
        # over ENABLED samples — so re-apply the trim here, after the last
        # sample has landed and its saved toggle has been restored. Convergent,
        # so overlapping with the load-settle pass is harmless.
        if (getattr(self, 'ds_propagate_var', None) is not None
                and self.ds_propagate_var.get()):
            self._apply_propagate_downsample()
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _periodic_autosave(self):
        """Autosave the session every few minutes (primary window only), so a
        hard crash loses less than the close-time autosave alone. Reschedules
        itself; best-effort and silent.

        The state is snapshotted here, on the Tk thread, and the files are
        written on a worker (as the close does), so the windows don't freeze
        while large samples are written. One autosave at a time: a tick that
        finds the last one still writing skips its turn."""
        if getattr(self, '_closing', False):
            return          # the close is writing the same file on a worker
        try:
            if (self._primary and self._samples
                    and not self._autosave_running()):
                plan = self._plan_session_write(self._session_autosave_path())
                self._start_autosave(plan)
        except Exception as exc:
            print(f"[session] periodic autosave failed: {exc}", flush=True)
        try:
            self.after(self._AUTOSAVE_MS, self._periodic_autosave)
        except Exception:
            pass

    def _autosave_running(self):
        t = getattr(self, '_autosave_thread', None)
        return t is not None and t.is_alive()

    def _start_autosave(self, plan):
        """Write an autosave `plan` on a worker thread. Not a daemon: a save
        under way when the program exits is finished, not cut off (and the
        close waits for it anyway; see _run_session_write)."""
        import threading

        def work():
            try:
                _data, failed = self._run_session_write(plan)
                if failed:
                    print(f"[session] periodic autosave: sidecar failed for "
                          f"{', '.join(failed)}", flush=True)
            except Exception as exc:               # noqa: BLE001
                print(f"[session] periodic autosave failed: {exc}",
                      flush=True)

        self._autosave_thread = threading.Thread(
            target=work, name='openflo-autosave', daemon=False)
        self._autosave_thread.start()

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
