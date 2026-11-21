"""Session save / load / autosave + recent-sessions menu — editor mixin.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import copy
import glob
import json
import logging
import os
import shutil
import threading
import time
import warnings
from tkinter import TclError, filedialog, messagebox
from typing import Any, cast

from . import result_stamp as _stamps
from . import session_lock as _lock
from . import session_models as _models
from . import tk_threads, ui_errors
from .channel_roles import embedding_prefix
from .editor_base import EditorMixin
from .errors import note
from .paths import is_fcs_file, relink_hints, saved_file_name
from .prefs import read_prefs, write_pref
from .session_format import column_records as _column_records
from .session_format import json_safe as _json_safe
from .session_format import qc_records as _qc_records
from .session_format import report_nonfinite as _report_nonfinite
from .session_open import offline_entry, plan_samples, raw_loads
from .session_sidecar import SIDECAR_EXT
from .ui_errors import show_error
from .view_tabs import parse_ranges

log = logging.getLogger(__name__)

# Hover tips for the session-conflict choices (_confirm_session_target).
_COPY_TIP = ("Save this window's session under a new name; you pick it "
             "next. Both versions are kept and nothing is lost.")
_OVERWRITE_TIP = ("Replace the file with this window's session. The other "
                  "version is lost; it cannot be got back from OpenFlo.")


def frame_digest(df):
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
    except Exception:                       # noqa: BLE001 - unhashable cells mean no fingerprint
        return None


def __getattr__(name):
    # The old private name of frame_digest, served with a DeprecationWarning.
    if name == '_frame_digest':
        warnings.warn('_frame_digest is deprecated and will be removed in a '
                      'future release; use frame_digest', DeprecationWarning,
                      stacklevel=2)
        return frame_digest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


def _short(name, most=28):
    """`name` cut to `most` characters with an ellipsis, for a button."""
    return name if len(name) <= most else name[:most - 1] + '…'


def _coerce_cluster_ids(lbls):
    """A copy of one sample's saved cluster labels with each cluster-id key
    turned back into an int (JSON stringified them); a key that is not a
    number is kept as it is."""
    coerced = {}
    for cid, nm in lbls.items():
        try:
            coerced[int(cid)] = nm
        except (TypeError, ValueError):
            coerced[cid] = nm
    return coerced


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


def _note_kept_links(ed, plan):
    """Remember, per sample of a session being opened whose raw FCS was not
    found (kept offline, or loaded from its sidecar), the ``rel_path`` it
    was saved with, the folder that is relative to and the raw path it
    names. Its saved path is from the computer that last had the file, so
    a save keeps this link (carried over to the folder saved in) rather
    than build one from that path (`_kept_link`). A sample whose file was
    found forgets it."""
    links = ed.__dict__.setdefault('_kept_links', {})
    sess_dir = getattr(ed, '_session_dir', '') or ''
    for p in plan:
        if not p.name:
            continue
        rel = p.entry.get('rel_path') or ''
        if p.raw_found or not rel or not sess_dir:
            links.pop(p.name, None)
        else:
            links[p.name] = (rel, sess_dir, p.raw_path)


def _kept_link(ed, entry):
    """``(rel_path, folder it is relative to)`` to keep for session `entry`
    (`_note_kept_links`), or ``('', '')``: for an offline sample, and for a
    loaded one still naming the raw path it was opened with."""
    name = entry.get('name')
    rec = (getattr(ed, '_kept_links', None) or {}).get(name)
    if rec is None:
        return '', ''
    rel, folder, raw = rec
    if name in ed._samples and (entry.get('path') or '') != raw:
        return '', ''
    return rel, folder


def _loaded_sample_entries(ed):
    """The session entry of each loaded sample of editor `ed`, in sample
    order (_session_state's 'samples', before the offline ones)."""
    # The last spectral unmix's spectra and QC (session_models), held by
    # one unmixed sample's entry.
    unmix = _models.unmix_record(ed)
    samples = []
    for name in ed._sample_order:
        s = ed._samples.get(name)
        if s is not None:
            samples.append(_sample_entry(ed, name, s, unmix))
    return samples


def _sample_entry(ed, name, s, unmix):
    """Sample `name`'s (data `s`) entry in the session's 'samples' list.
    Key order is the order written to the file."""
    entry = _sample_basics(ed, name, s)
    entry.update(_column_records(s))
    entry.update(_qc_records(s))
    entry.update(_editor_records(ed, name))
    entry.update(_analysis_records(name, s, unmix))
    # v3: which OpenFlo saved this sample, and the stamp each computed
    # column got when it was made (result_stamp). The stamps go back
    # as they were read, so a save never claims an older result.
    entry['app_version'] = _stamps.app_version()
    entry['result_stamps'] = _stamps.column_stamps(s)
    # A corrected column's original (fixes): its stamp, and the
    # column its old values are kept under in the sidecar.
    entry['result_originals'] = _stamps.column_originals(s)
    # v3: its gate counts and statistics (saved_stats), filled in
    # by the write once the data's digest is taken.
    entry['saved_stats'] = None
    return entry


def _sample_basics(ed, name, s):
    """Where sample `name` came from and how the editor shows it."""
    entry = {
        'name': name,
        # A sample restored from a processed CSV keeps the raw FCS it
        # came from in `source_fcs`; record THAT, so the link to the
        # FCS survives a second save instead of pointing at the CSV.
        'path': (getattr(s, 'source_fcs', '')
                 or getattr(s, 'path', '') or ''),
        'color': ed._sample_colors.get(name, '#1f77b4'),
        'plot_enabled': bool(ed._sample_plot_enabled.get(name, False)),
        'trial': ed._sample_trial.get(name, 'Trial'),
    }
    # Persist a manual Comps/Samples override only when set (so
    # name-based detection still applies to untouched samples on load).
    if name in ed._sample_is_comp:
        entry['is_comp'] = bool(ed._sample_is_comp[name])
    # Set aside from comparisons (with the reason) and the sample's
    # note: additive keys, which an older OpenFlo ignores.
    aside = ed._sample_set_aside
    notes = ed._sample_notes
    if name in aside:
        entry['set_aside'] = str(aside[name])
    if name in notes:
        entry['note'] = str(notes[name])
    return entry


def _editor_records(ed, name):
    """What editor `ed` holds about sample `name` outside its data: its
    downsample trim and the channels it is the FMO control for."""
    entry = {}
    # A 'Display + data' trim: the length it was cut from and each
    # floor, so a reload from the raw FCS replays it and gets the
    # same events. The floor was taken over the samples present and
    # ticked when it ran; taken again on reopen it would give a
    # sample all its events back once its smallest partner was
    # removed or unticked. Additive, inside the sample entry, like 'qc_applied'.
    from .editor_downsample import trim_entry
    trim = trim_entry(ed, name)
    if trim is not None:
        entry['downsample_trim'] = trim
    # The channels this sample is the FMO control for (FMO gating),
    # so histograms can draw it again after a reopen. Additive, inside
    # the sample entry, as 'transforms' is.
    fmo_of = sorted(ch for ch, fmo in (getattr(
        ed, '_fmo_controls', None) or {}).items() if fmo == name)
    if fmo_of:
        entry['fmo_of'] = fmo_of
    return entry


def _analysis_records(name, s, unmix):
    """Records of the analyses run on sample `name` (data `s`) that a
    reload from the raw FCS needs to redo or explain them."""
    entry = {}
    # Derived channels' formulas (derived.py), so a reload from the
    # raw FCS recomputes them. Additive, like 'transforms'.
    der = getattr(s, 'derived', None)
    if der:
        entry['derived'] = {str(c): str(f) for c, f in der.items()
                            if c in s.data.columns}
    # Bead normalization (ui_mass_cytometry): the factor record, so a
    # reload from the raw FCS applies it again and Undo still works.
    # Additive, like 'derived'.
    bn = getattr(s, 'bead_norm', None)
    if bn:
        entry['bead_norm'] = dict(bn)
    entry.update(_model_records(name, s))
    if unmix is not None and unmix[0] == name:
        entry[_models.UNMIX_KEY] = unmix[1]
    # Batch normalization (CytoNorm) replaced its values in place:
    # the record, which also makes the save keep a sidecar of them
    # (_has_computed_columns). Additive, like 'bead_norm'.
    cn = getattr(s, 'batch_norm', None)
    if cn:
        entry['batch_norm'] = dict(cn)
    return entry


def _model_records(name, s):
    """The cell-cycle and FlowSOM models (session_models) of sample `name`
    (data `s`): with only the columns kept, a reopened sample would have
    its phases and metaclusters but the Cell cycle window and the SOM tree
    would say to run them again. Additive, like 'bead_norm'."""
    entry = {}
    for key, make in ((_models.CELL_CYCLE_KEY,
                       _models.cell_cycle_record),
                      (_models.FLOWSOM_KEY, _models.flowsom_record)):
        try:
            rec = make(s)
        except TypeError as exc:
            note(f"session: {name}: {key} not saved", exc)
            continue
        if rec is not None:
            entry[key] = rec
    return entry


def _sample_gate_lists(ed):
    """Editor `ed`'s per-sample gates as ordered lists carrying their
    editor id + parent_id (so the hierarchy restores)."""
    from .view_tabs import gate_lists
    return gate_lists(ed._sample_gates, ed._sample_gate_order)


def _fixes_applied_ids(ed):
    """The ids of the fixes applied to ``ed``'s results (fixes.applied_ids),
    for the session's ``fixes_applied`` (informational: a reopen reads the
    results' kept originals, not this list)."""
    from . import fixes
    return fixes.applied_ids(ed)


def _view_settings(ed):
    """Editor `ed`'s channel, plot and view settings, its audit trail and
    fixes, in the order _session_state writes them after the gates."""
    # _channel_range values are tuples → JSON lists. Skip any None
    # (auto-range) entries — the type allows None even though we
    # pop rather than store it.
    ranges = {ch: [float(rng[0]), float(rng[1])]
              for ch, rng in ed._channel_range.items()
              if rng is not None}
    return {
        'channel_scale': dict(ed._channel_scale),
        'channel_range': ranges,
        'channel_labels': dict(ed._channel_labels),
        'plot_mode': ed.mode_var.get(),
        'x_channel': ed.x_combo.get(),
        'y_channel': ed.y_combo.get(),
        'color_channel': ed.color_combo.get(),
        'downsample_display': bool(ed.ds_display_var.get()),
        'downsample_propagate': bool(ed.ds_propagate_var.get()),
        'max_points': ed.max_points_var.get(),
        'show_removed': bool(ed.show_removed_var.get()),
        'contour_scatter': bool(ed.contour_scatter_var.get()),
        'contour_outliers': bool(ed.contour_outliers_var.get()),
        'hist_y_mode': ed.hist_y_mode.get(),
        'cluster_labels': dict(ed._cluster_labels),   # reserved slot
        'audit': ed._audit_log.to_list(),
        # v3: the fixes (fixes.py) applied to this session's results (read
        # from the results' kept originals) and declined, per result.
        'fixes_applied': _fixes_applied_ids(ed),
        'fixes_declined': copy.deepcopy(
            list(getattr(ed, '_declined_fixes', None) or [])),
        # v2: the tray's tabs (saved views) and the layout editor's pages.
        'view_tabs': (ed._tabs_state()
                      if hasattr(ed, '_tabs_state') else None),
        'layout': (ed._layout_doc.to_dict()
                   if getattr(ed, '_layout_doc', None) is not None
                   else None),
    }


def _folders_matching(saved, found):
    """How many folders directly above the file agree between a sample's
    `saved` path and a `found` one, from the innermost out ('Day 9/tube.fcs'
    against '.../moved/Day 9/tube.fcs' is 1). Either separator, any case: a
    session saved on another system keeps its own."""
    import re

    def parents(p):
        return [d.casefold() for d in re.split(r'[\\/]+', p)[:-1] if d][::-1]
    n = 0
    for a, b in zip(parents(saved), parents(found), strict=False):
        if a != b:
            break
        n += 1
    return n


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
    except (OSError, ValueError):
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
    except Exception as exc:                        # noqa: BLE001 - the backup is a courtesy copy; failing to keep it is noted and never blocks the save
        note(f"session: backup of {path} not kept", exc)


class SessionMixin(EditorMixin):
    """Full-state session persistence (.flowsession), periodic autosave, and
    the Open-Recent submenu."""

    SESSION_EXT = '.flowsession'

    NEW_WINDOW_PAUSE_S = 2.0        # File > New window: one start per pause
    _AUTOSAVE_MS = 300_000          # periodic autosave cadence (5 min)
    _AUTOSAVE_WARN_AFTER = 2        # failed autosaves in a row before a warning

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

    # One answer to "is this PID running?" for autosaves and session locks.
    # A process we may not signal (PermissionError) is alive, so resume and
    # pruning never take a file a running instance still owns.
    _pid_alive = staticmethod(_lock.pid_alive)

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
        except OSError:
            pass
        # A crash or power cut can leave the newest autosave damaged (cut
        # short, or zero bytes), so the newest READABLE one is offered: a
        # damaged newest file must not hide every older one that is fine.
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
                # session (<stem>_data/); removing only the .flowsession
                # file would leak these (often large) event-table CSV
                # folders — one per app run — forever.
                data_dir = os.path.splitext(p)[0] + '_data'
                if os.path.isdir(data_dir):
                    shutil.rmtree(data_dir, ignore_errors=True)
            except OSError:
                pass

    def _session_state(self):
        """Serialise the full editor state to a JSON-able dict."""
        from datetime import datetime
        for gates in self._sample_gates.values():
            _stamps.settle_gate_stamps(gates)   # a hand-edited gate is the user's
        samples = _loaded_sample_entries(self)
        # Samples whose files were missing when the session opened: written
        # back as they were read, gates and all (_keep_offline).
        off_entries, off_gates = _offline_parts(self)
        samples.extend(off_entries)
        sample_gates = _sample_gate_lists(self)
        for name, gl in off_gates.items():
            if not sample_gates.get(name):
                sample_gates[name] = gl
        from .session_format import SESSION_FORMAT, SESSION_VERSION
        return {
            'format': SESSION_FORMAT,
            'version': SESSION_VERSION,
            'app_version': _stamps.app_version(),   # v3: the OpenFlo that wrote it
            'created': datetime.now().isoformat(timespec='seconds'),
            'active_sample': self._active_sample,
            'samples': samples,
            'sample_gates': sample_gates,
            **_view_settings(self),
        }

    @staticmethod
    def _has_computed_columns(s):
        """True if a sample's data carries columns produced by analysis
        (clustering / embeddings / FMO gates / calibration) that aren't in the
        raw FCS — i.e. worth persisting so a reopened session keeps them.
        Also True for a sample that has no raw FCS to rebuild it from."""
        # A sample built FROM a CSV (a restored sidecar, 'Load in editor',
        # File > Load CSV) has no FCS behind it: its entry's path is that
        # CSV, which the restore reads as an FCS and fails. Its sidecar must
        # be kept even when nothing else calls for one -- a compensated
        # sample restored from its sidecar has no comp_channels -- or Save As
        # or the autosave would lose it.
        path = str(getattr(s, 'path', '') or '')
        if not is_fcs_file(path):
            return True
        data = getattr(s, 'data', None)
        cols = set(data.columns) if data is not None else set()
        if cols & {'cluster', 'leiden', 'flowsom', 'flowsom_meta',
                   'pseudotime', 'cell_cycle', 'debarcode'}:
            return True
        if any(_stamps.is_original_column(c) for c in cols):
            return True                 # a corrected column's kept original
        # A COMPENSATED sample carries no new column, but its values have been
        # transformed in place — so it differs from the raw FCS and must be
        # persisted too. Otherwise a reopened session reloads the RAW (un-
        # compensated) data and any gate/population drawn in compensated space
        # selects the wrong events (populations "don't reattach").
        if getattr(s, 'comp_channels', None):
            return True
        # Batch normalization (CytoNorm) likewise changes the values in place
        # and cannot be redone from the raw FCS: without a sidecar an
        # uncompensated sample would reopen on its raw values, the
        # normalization silently gone.
        if getattr(s, 'batch_norm', None):
            return True
        for c in cols:
            cu = str(c)
            # U: = spectral-unmixing abundances, which an uncompensated
            # sample (no other reason for a sidecar) would lose on reload.
            if (cu.endswith('_pos') or cu.startswith(('MESF:', 'U:', 'D:'))):
                return True
            if embedding_prefix(cu) is not None:
                return True
        return False

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
        numbers = []                # (entry, _saved_stats_plan)
        for entry in data.get('samples', []):
            # Portable relink hints: basename + a path relative to the session
            # file, so a moved/copied project (or one opened on another machine,
            # e.g. via the web app) can re-find its raw FCS even when the stored
            # absolute path no longer exists. Kept inside the sample entry, so
            # the locked top-level SESSION_KEYS are unchanged. Forward slashes,
            # whatever OS saves (paths.relink_hints); an offline sample keeps
            # the link it was opened with.
            # A link that cannot be worked out here is dropped, not kept
            # relative to the folder the session was opened from.
            kept_rel, kept_from = _kept_link(self, entry)
            entry.pop('rel_path', None)
            entry.update(relink_hints(entry.get('path') or '', sess_dir,
                                      kept_rel=kept_rel, kept_from=kept_from))
            s = self._samples.get(entry.get('name'))
            plan_nums = getattr(self, '_saved_stats_plan', None)
            nums = (plan_nums(entry.get('name'))
                    if s is not None and plan_nums else None)
            if nums is not None:
                numbers.append((entry, nums))
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
                         'safe': self._sidecar_safe_name(entry['name']),
                         'digest': nums['digest'] if nums else None})
        return {'path': path, 'data': data, 'jobs': jobs,
                'saved_stats': numbers}

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
            # injective, but this loop is where a collision would do damage —
            # the file overwritten and the SAME path recorded on both
            # entries, and restore prefers the sidecar over the raw FCS, so
            # one sample would silently come back holding the other's events.
            # Belt and braces: if a stem ever repeats, skip rather than
            # overwrite, and report it like any other sidecar failure.
            if safe in used_stems:
                note(f"session: processed sidecar for {entry['name']} skipped: "
                     f"name collides with {used_stems[safe]!r} on disk", None)
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
                from .transform_sidecar import (
                    discard_transforms_sidecar,
                    write_transforms_sidecar,
                )
                discard_transforms_sidecar(stem + SIDECAR_EXT)
                # One digest serves the sidecar and the saved numbers.
                job['digest'] = job.get('digest') or frame_digest(df)
                side_path = self._write_sidecar(df, stem, keep=job['source'],
                                                digest=job['digest'])
                rec = job['transforms']
                if rec is not None:
                    write_transforms_sidecar(side_path, rec, df.columns,
                                             len(df))
                entry['processed_data'] = os.path.join(
                    os.path.basename(data_dir),
                    os.path.basename(side_path)).replace(os.sep, '/')
            except Exception as exc:  # noqa: BLE001 - noted and listed; the user is warned once the save finishes
                note(f"session: processed sidecar for {entry['name']} failed", exc)
                sidecar_failures.append(entry['name'])
        if progress is not None:
            progress(len(jobs), total, '')
        digests = {j['name']: j.get('digest') for j in jobs}
        for entry, nums in plan.get('saved_stats') or ():
            entry['saved_stats'] = self._saved_stats_finish(
                nums, digests.get(entry['name']))
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

    def _write_sidecar(self, df, stem, keep='', digest=None):
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

        Hashing the frame costs a small fraction of encoding it. `digest`
        is the frame's ``frame_digest`` when the caller already took it."""
        from .session_sidecar import LEGACY_EXT, frame_arrays, write_frame
        if digest is None:
            digest = frame_digest(df)
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
        are changes since the last save or open."""
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
        except Exception:                           # noqa: BLE001 - title-bar text is cosmetic
            pass

    def _note_unsaved(self):
        """Something the session saves changed: mark the title, and let the
        open analysis windows catch up (editor_watch)."""
        self._notify_changed()
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

    def _ask_choice(self, title, text, choices, default=0, tips=None):
        """A small modal question with one button per ``(label, value)``
        in `choices`; returns the value picked. Escape or closing the window
        returns the last choice's value (Cancel, by convention).

        ``tips``, optional, holds one hover tip per choice, in order (None or
        a missing entry for a button that needs none, such as Cancel): what
        picking that button does, so a choice with consequences says them."""
        import tkinter as tk
        from tkinter import ttk
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.resizable(False, False)
        try:
            dlg.transient(self.winfo_toplevel())
        except TclError:
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
        tips = list(tips or ())
        for i, (label, value) in enumerate(choices):
            b = ttk.Button(bb, text=label,
                           command=lambda v=value: pick(v))
            b.pack(side='left', padx=(6, 0))
            buttons.append(b)
            if i < len(tips) and tips[i]:
                self._tip(b, tips[i])
        dlg.bind('<Escape>', lambda _e: pick(cancel))
        dlg.protocol('WM_DELETE_WINDOW', lambda: pick(cancel))
        try:
            buttons[default].focus_set()
            dlg.grab_set()
        except TclError:
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
                 ("Cancel", 'cancel')],
                tips=[_COPY_TIP, _OVERWRITE_TIP])
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
            [("Overwrite", 'overwrite'), ("Cancel", 'cancel')], default=1,
            tips=[_OVERWRITE_TIP])
        return path if choice == 'overwrite' else None

    def _save_session_as(self):
        """Save under a new name, which later saves then write to."""
        self._save_session(save_as=True)

    def _save_session(self, save_as=False):
        """Save the session (Ctrl+S; Save as asks for a name). Returns True
        when the session file was written, False when nothing was saved."""
        if not self._samples:
            self.status_var.set("Load at least one sample before saving a session.")
            return False
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
            return False
        path = self._confirm_session_target(path)
        if not path:
            self.status_var.set("Session not saved.")
            return False
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
                ui_errors.showwarning(
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
        except Exception as exc:  # noqa: BLE001 - shown in the status line and an error dialog; the editor carries on
            self.status_var.set(f"Save session failed: {exc}")
            show_error(self, "Save session failed",
                       "The session could not be saved.", exc, path)
            return False
        return True

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load editor session",
            filetypes=[('OpenFlo session or project',
                        '*' + self.SESSION_EXT + ' *.flowproject *.zip'),
                       ('OpenFlo session', '*' + self.SESSION_EXT),
                       ('OpenFlo project archive', '*.flowproject *.zip'),
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
        if self._archive_export_running():
            self.status_var.set("A project archive is already being written; "
                                "wait for it to finish.")
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
        tmp = tempfile.mkdtemp(prefix='openflo-project-')
        try:
            # The Tk-thread half: snapshot the session (it reads Tk state).
            # The files are written on a worker, as the autosave does, so a
            # large project does not freeze the windows while it is zipped.
            plan = self._plan_session_write(
                os.path.join(tmp, 'project' + self.SESSION_EXT))
        except Exception as exc:                    # noqa: BLE001 - any failure to snapshot the session is shown in the status line and an error dialog
            shutil.rmtree(tmp, ignore_errors=True)
            self._archive_export_failed(exc, out)
            return

        def work():
            try:
                self._run_session_write(plan)
                return write_project_archive(plan['path'], out)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        def done(manifest):
            self._archive_exporting = False
            self._archive_exported(manifest, out)

        def failed(exc):
            self._archive_exporting = False
            self._archive_export_failed(exc, out)

        # A second export, or a close, waits until this one has finished
        # (see _on_close): the zip goes to a .tmp file renamed at the end.
        self._archive_exporting = True
        self.run_async(work, on_done=done, on_error=failed,
                       busy_msg="Exporting project archive…")

    def _archive_export_running(self):
        """Whether File > Export project archive is still writing."""
        return bool(getattr(self, '_archive_exporting', False))

    def _archive_export_failed(self, exc, out):
        self.status_var.set(f"Project export failed: {exc}")
        show_error(self, "Project export failed",
                   "The project could not be exported.", exc, out)

    def _archive_exported(self, manifest, out):
        self._audit('project.export', path=out, samples=manifest['samples'],
                    missing=manifest['missing'])
        n = len(manifest['samples'])
        self.status_var.set(
            f"Exported project: {n} sample(s) → {os.path.basename(out)}")
        if manifest['missing']:
            ui_errors.showwarning(
                "Project exported without some data files",
                "These samples' FCS files were not found, so the archive "
                "holds their gates but not their events:\n\n  "
                + "\n  ".join(manifest['missing'][:10])
                + ("\n  …" if len(manifest['missing']) > 10 else ""),
                parent=self)

    def _open_nearby_send(self):
        """File > Nearby send…: send files to, and receive them from,
        OpenFlo on phones and computers on this network (ui_nearby)."""
        from .ui_nearby import NearbyWindow
        self._show_or_raise('nearby', lambda: NearbyWindow(self, editor=self))

    def _expand_project_archive(self):
        """File > Expand project archive: a .flowproject back to ordinary
        files (``project_archive.expand_project``) in a new folder, named
        after it, inside a folder the user picks. Nothing is opened."""
        from .project_archive import ARCHIVE_EXTS, expand_project, unpack_folder_for
        path = filedialog.askopenfilename(
            title="Expand project archive",
            filetypes=[('OpenFlo project archive',
                        ' '.join('*' + e for e in ARCHIVE_EXTS)),
                       ('All files', '*.*')])
        if not path:
            return
        into = filedialog.askdirectory(
            title="Put its files in a new folder inside…",
            initialdir=os.path.dirname(path))
        if not into:
            return
        dest = unpack_folder_for(os.path.join(into, os.path.basename(path)))

        def done(r):
            self._audit('project.expand', path=path, folder=r['folder'],
                        fcs=len(r['fcs']), missing=r['missing'])
            note_out = (f"; {r['left_out']} gate(s) FlowJo cannot hold left "
                        "out of the .wsp" if r['left_out'] else '')
            note_missing = (f"; saved without the FCS file of "
                            f"{len(r['missing'])} sample(s), left out of "
                            "the .wsp" if r['missing'] else '')
            self.status_var.set(
                f"Expanded {os.path.basename(path)}: {len(r['fcs'])} FCS "
                f"file(s) and {os.path.basename(r['wsp'])} in "
                f"{r['folder']}{note_out}{note_missing}")

        def failed(exc):
            self.status_var.set(f"Expand project failed: {exc}")
            show_error(self, "Can't expand project",
                       "The project could not be expanded.", exc, path)

        self.run_async(lambda: expand_project(path, dest), on_done=done,
                       on_error=failed, busy_msg="Expanding project…")

    def _open_project_archive_path(self, path, then=None, confirmed=False):
        """Open a project archive: a copy already unpacked from it when
        there is one (``project_copies.find_copy``: beside it or in the
        internal project folder), else unpack it, off the Tk thread (a big
        archive on a slow USB stick takes minutes; the window keeps
        answering and the status line counts the megabytes), beside it or,
        on a removable drive, in the internal project folder
        (``project_copies.unpack_destination``); then open its session. A
        copy whose session was saved since it was unpacked is asked about:
        open it, or unpack a fresh copy. The offer to save the current
        session comes once the files are there, so work done meanwhile is
        asked about too (unless `confirmed`: a caller already asked). One
        project opens at a time; another is refused until it has.
        ``then(opened)`` is called on the Tk thread once it has opened
        (True) or not (False; already reported)."""
        from .project_archive import open_project_archive
        from .project_copies import find_copy, unpack_destination
        name = os.path.basename(path)
        busy = self.__dict__.get('_project_opening')
        if busy:
            self.status_var.set(
                f"A project is still opening ({busy}): wait for it to "
                f"open, then open {name}.")
            return self._session_open_done(False, then)
        self._project_opening = name
        busy_msg = f"Opening project {name}…"
        unpacking = f"Unpacking project {name}"

        def not_opened():
            self._project_opening = None
            status = self.status_var.get()
            if status == busy_msg or status.startswith(unpacking):
                # refused or cancelled with a dialog only: nothing is
                # opening any more
                self.status_var.set(f"Project {name} was not opened.")
            self._session_open_done(False, then)

        def opened(sess, reused):
            self._project_opening = None
            folder = os.path.dirname(sess)
            self._audit('project.open', path=path, folder=folder,
                        reused=reused)
            if not self._load_session_path(sess, confirmed=confirmed):
                not_opened()
                return
            self.status_var.set(
                f"Opened project {name} from the copy already unpacked in "
                f"{folder}" if reused else
                f"Opened project archive {name}; its files are in {folder}")
            self._session_open_done(True, then)

        def failed(exc):
            self._project_opening = None
            self.status_var.set(f"Open project failed: {exc}")
            show_error(self, "Can't open project",
                       "The project could not be opened.", exc, path)
            self._session_open_done(False, then)

        def unpack():
            report = self._unpack_progress(unpacking)
            self.run_async(
                lambda: open_project_archive(
                    path, unpack_destination(path), progress=report),
                on_done=lambda r: opened(r[0], False), on_error=failed,
                busy_msg=busy_msg)

        def found(copy):
            if copy is None:
                unpack()
            elif not copy.changed:
                opened(copy.session, True)
            else:
                choice = self._ask_choice(
                    "Project already unpacked",
                    f"{name} was unpacked before, into {copy.folder}, and "
                    "its session there has been saved since.\n\nOpen that "
                    "copy with the changes saved in it, or unpack a fresh "
                    f"copy of the project as {name} holds it?",
                    [("Open saved copy", 'copy'),
                     ("Unpack fresh copy", 'fresh'), ("Cancel", 'cancel')],
                    tips=["Open the folder unpacked before, with the work "
                          "saved there. Nothing is unpacked again.",
                          "Unpack the archive again into a new folder, as "
                          "it was saved. The earlier copy is kept as it "
                          "is."])
                self._audit('project.copy_choice', path=path, choice=choice)
                if choice == 'copy':
                    opened(copy.session, True)
                elif choice == 'fresh':
                    unpack()
                else:
                    not_opened()

        self.run_async(lambda: find_copy(path), on_done=found,
                       on_error=failed, busy_msg=busy_msg)
        return None

    def _unpack_progress(self, text):
        """A ``progress(done, total)`` for ``open_project_archive``, run on
        a worker: it puts ``<text>: <done> of <total> MB`` on the status
        line, each time the whole megabytes written change."""
        last = [-1]

        def report(done, total):
            mb = done // 1_000_000
            if mb != last[0]:
                last[0] = mb
                self._busy(f"{text}: {mb:,} of "
                           f"{(total + 999_999) // 1_000_000:,} MB")
        return report

    @staticmethod
    def _session_open_done(opened, then):
        """Tell `then` (when given) whether a session opened; returns it."""
        if then is not None:
            then(opened)
        return opened

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
        except Exception as exc:                      # noqa: BLE001 - any file that will not read or migrate as a session is reported in a dialog; nothing is changed
            ui_errors.showerror("Compare with saved session",
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

    def _load_session_path(self, path, confirmed=False, then=None):
        """Open the session (or project archive) at `path` in this window,
        in place of what it holds. Unless `confirmed` (a caller already
        asked), the user is first offered to save the current session
        (_ok_to_replace_session); it is closed (_reset_session_state) only
        once the new file has read, so a file that will not open leaves it
        as it was. Returns whether the session opened: False when the user
        cancelled or the file could not be read (already reported); None
        for a project archive, which unpacks in the background and asks
        about saving once it has (False at once when another project is
        still opening). ``then``
        (when given) is called with True/False once it is known, in every
        case."""
        from .project_archive import is_project_archive
        if is_project_archive(path):
            # asked once it has unpacked, about the window as it is then
            return self._open_project_archive_path(path, then, confirmed)
        if not confirmed and not self._ok_to_replace_session(path):
            return self._session_open_done(False, then)
        return self._session_open_done(self._read_session_path(path), then)

    def _read_session_path(self, path):
        """_load_session_path for a session file (not an archive), once
        the user has agreed to replace the current one: read, check and
        apply it. Returns whether it opened."""
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
                     ("Open anyway", 'anyway'), ("Cancel", 'cancel')],
                    tips=["Open the session to look at it without changing "
                          "the file. Saving then asks for a new name, so the "
                          "other window's work is safe.",
                          "Open it normally, as if no other window had it. "
                          "Choose this only if that window is closed or no "
                          "longer in use; OpenFlo still asks before saving "
                          "over its work."])
                self._audit('session.open_locked', path=path, choice=choice)
                if choice == 'cancel':
                    return False
                if choice == 'anyway':
                    _lock.acquire(path, self._session_lock_token(),
                                  force=True)
                else:
                    read_only = path
        stamp = _lock.file_stamp(path)
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError, RecursionError) as exc:  # an unreadable, malformed or absurdly nested session is reported to the user; lock released if this window took it
            if not read_only and not backup:
                _lock.release(path, self._session_lock_token())
            self.status_var.set(f"Load session failed: {exc}")
            self._report_unreadable_session(path, exc)
            return False
        if not isinstance(data, dict) or data.get('format') != 'openflo-session':
            if not read_only and not backup:
                _lock.release(path, self._session_lock_token())
            ui_errors.showerror(
                "Not a session file",
                f"{os.path.basename(path)} isn't an OpenFlo session "
                "(missing format marker).", parent=self)
            return False
        # Auto-upgrade an older saved session to the current schema so files
        # from a previous OpenFlo keep opening. Refuse one written by a NEWER
        # build (we can't know its schema) rather than mis-read it.
        from .session_format import SessionError, migrate_session
        try:
            data, _mig_notes = migrate_session(data)
        except SessionError as exc:
            if not read_only and not backup:
                _lock.release(path, self._session_lock_token())
            ui_errors.showerror("Can't open session", str(exc), parent=self)
            return False
        # The file reads: close what this window holds, so the session
        # opens alone and Save writes only it back (W1).
        self._reset_session_state()
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
        return True

    # ── Opening a session in place of the current one (W1) ────────────────

    def _holds_session(self):
        """Whether this window holds a session to lose: samples loaded or
        loading, or samples kept from a session whose files were missing."""
        return bool(self._samples or self._loading
                    or self.__dict__.get('_offline_samples'))

    def _ok_to_replace_session(self, path):
        """Before the session at `path` opens in place of this window's:
        offer to save the current one. Asked whenever it holds samples,
        unless every change is already saved in its own file (a resumed
        autosave or an untitled session has no file of its own, so it is
        asked about too). Returns whether to go on opening."""
        return self._offer_save_current(
            "open the session", "Open another session",
            lambda cur: (
                f"Save changes to {cur} before opening "
                f"{os.path.basename(path)}?\n\nThe session opens in this "
                "window in place of the current one: its samples, gates, "
                "tabs and channel settings are closed first."),
            ["Save the current session first (a new session asks "
             "for a name), then open the other one.",
             "Close the current session without saving its changes, "
             "then open the other one. Changes since the last save "
             "are lost."],
            'session.replace', path=path)

    def _offer_save_current(self, then, title, question, tips, action,
                            **details):
        """Before this window's session is closed (to open another, or by
        File > Close session): offer to save it. Asked whenever it holds
        samples, unless every change is already saved in its own file (a
        resumed autosave or an untitled session has no file of its own, so
        it is asked about too). A sample move between windows still under
        way stops it, naming what to do `then`. `question(cur)` words the
        prompt for the session's name; the answer is audited as `action`.
        Returns whether to go on.

        A job still working on the session's samples (clustering, a job in
        the status bar, an embedding comparison, a project archive export,
        any other background job of the window's) stops it too: its results would land in the next session's
        same-named samples."""
        busy = self._session_busy_reason()
        if busy is not None:
            self.status_var.set(
                f"Wait: {busy}. Let it finish (or press Cancel in the "
                f"status bar), then {then}.")
            return False
        if not self._holds_session():
            return True
        if (self.__dict__.get('_pending_move')
                or self.__dict__.get('_inbound_move')):
            self.status_var.set(
                "A sample move between windows is still under way: finish "
                f"or cancel it, then {then}.")
            return False
        own = self._open_session_file()
        if own and not self.__dict__.get('_unsaved'):
            return True
        cur = os.path.basename(own) if own else "the current session"
        choice = self._ask_choice(
            title, question(cur),
            [("Save", 'save'), ("Don't save", 'discard'),
             ("Cancel", 'cancel')],
            tips=tips)
        self._audit(action, choice=choice, **details)
        if choice == 'save':
            return self._save_session()
        return choice == 'discard'

    def _session_busy_reason(self):
        """What is still working on this window's session in the
        background, in words, or None when nothing is."""
        if getattr(self, '_clustering_busy', False):
            return "clustering is still running"
        strip = self.__dict__.get('_job_strip')
        if strip is not None and strip.running:
            what = (strip.text or '').rstrip('…').strip() or "a job"
            return f"{what} is still running"
        if getattr(self, '_dr_running', False):
            return "the embedding comparison is still running"
        if self._archive_export_running():
            return "the project archive is still being written"
        # Any other background job of this window's (run_async): a transform
        # change or spectral unmixing writes its result into samples looked
        # up by name when it lands.
        for msg in (self.__dict__.get('_session_jobs') or {}).values():
            what = (msg or '').rstrip('…').strip()
            return (f"a background job ({what}) is still running" if what
                    else "a background job is still running")
        return None

    # ── File > Close session and File > New window ───────────────────────────

    def _close_session(self):
        """File > Close session: close the session this window holds, after
        offering to save it, and leave the window as a new one opens, ready
        for the next set of samples. The session file's lock is let go, so
        another window can open it. Returns whether it closed."""
        if not (self._holds_session()
                or self.__dict__.get('_session_file')
                or self.__dict__.get('_session_read_only')):
            self.status_var.set("No session is open, so there is nothing "
                                "to close.")
            return False
        if not self._offer_save_current(
                "close the session", "Close session",
                lambda cur: (
                    f"Save changes to {cur} before closing it?\n\nIts "
                    "samples, gates, tabs and channel settings are closed, "
                    "and this window stays open, empty, for the next "
                    "samples."),
                ["Save the session first (a new session asks for a name), "
                 "then close it.",
                 "Close the session without saving its changes. Changes "
                 "since the last save are lost."],
                'session.close'):
            return False
        self._reset_session_state()
        self._start_empty_session()
        self.status_var.set("Session closed. Add FCS files or open a "
                            "session to start the next one.")
        return True

    def _start_empty_session(self):
        """After _reset_session_state with no session to open in its place:
        clear what opening one would have set (_apply_session), so the
        window holds what a new window does. Plot display settings stay."""
        from .audit import AuditLog
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._fmo_controls = {}
        self._session_channel_labels = {}
        self._explicit_channel_labels = {}
        self._declined_fixes = []
        self._legacy_decline_samples = None
        self._pending_sample_meta.clear()
        self._pending_sample_prep.clear()
        self._session_restore = None
        for k in ('_session_dir', '_session_data_dir'):
            self.__dict__.pop(k, None)
        try:
            self._restore_tabs_state(None)
        except Exception as exc:                      # noqa: BLE001 - a tab strip that will not rebuild is noted; the session is closed anyway
            note("close session: tabs not reset", exc)
        # A new trail: the closed session's records are not this one's.
        self._audit_log = AuditLog()
        win = self.__dict__.get('_audit_window')
        try:
            if win is not None and win.winfo_exists():
                win.refresh(self._audit_log)
        except TclError:
            pass
        self._remember_session_file(None)       # lets go of its lock
        self._note_saved()

    def _open_new_window(self):
        """File > New window: start another OpenFlo, with its own empty
        window, so two sessions can be worked on side by side. It is a
        separate program, as a second launch is (own autosave, own session
        locks), and it does not offer to resume a session. Returns whether
        it started. A second request within ``NEW_WINDOW_PAUSE_S`` (a held
        Ctrl+N repeats) starts nothing more."""
        now = time.monotonic()
        last = self.__dict__.get('_new_window_at')
        if last is not None and now - last < self.NEW_WINDOW_PAUSE_S:
            self.status_var.set("A new window is already starting…")
            return False
        import subprocess
        import sys
        import tempfile

        from .new_window import new_window_command, new_window_env
        out: Any
        try:
            # Real output handles: a console-less Python started with none
            # can stop before its window opens (openflo-gui.bat).
            out = open(os.path.join(tempfile.gettempdir(),
                                    'openflo-gui.log'), 'a', encoding='utf-8')
        except OSError:
            out = subprocess.DEVNULL
        try:
            subprocess.Popen(
                new_window_command(), env=new_window_env(),
                stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                close_fds=True,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                start_new_session=(sys.platform != 'win32'))
        except (OSError, ValueError) as exc:
            note("start a new OpenFlo window", exc)
            self.status_var.set(f"Could not start a new window: {exc}")
            return False
        finally:
            if out is not subprocess.DEVNULL:
                out.close()
        self._new_window_at = now
        self.status_var.set("Starting a new OpenFlo window…")
        return True

    def _reset_session_state(self):
        """Close the session this window holds, before another opens in
        its place or by File > Close session: what a new window starts with, so nothing of the old
        session (samples, gates, tabs, layout, channel settings, files kept
        offline) is shown with the new one or saved into its file. The
        plot display settings stay; the new session sets its own, as it
        does its tabs and the session's own channel labels."""
        self._drop_queued_loads()
        if self.__dict__.get('_watch_dir'):
            self._stop_watch_folder()
        self._close_session_windows()
        self._remove_samples(list(self._samples))
        self._clear_session_records()
        self._clear_session_view()

    def _drop_queued_loads(self):
        """Drop the loads still queued, and make those in flight land
        nowhere (editor_loadpool._post_load_result): they are the old
        session's samples."""
        import queue
        keep = []
        while True:
            try:
                item = self._load_queue.get_nowait()
            except queue.Empty:
                break
            if item[2] is None:
                keep.append(item)               # a shutdown sentinel
            else:
                (self.__dict__.get('_load_job_epochs') or {}).pop(item[1],
                                                                   None)
        for item in keep:
            self._load_queue.put(item)
        self._load_epoch = self.__dict__.get('_load_epoch', 0) + 1
        self._loading.clear()
        self._load_total = 0
        self._load_done = 0
        self._update_progress_bar()

    def _close_session_windows(self):
        """Close the tool windows that show the old session's samples."""
        wins = list((self.__dict__.get('_tool_windows') or {}).values())
        wins += [self.__dict__.get(k) for k in
                 ('_layout_window', '_batch_norm_win', '_bead_norm_win')]
        for win in wins:
            try:
                if win is not None and win.winfo_exists():
                    win.destroy()
            except TclError:
                pass
        (self.__dict__.get('_tool_windows') or {}).clear()
        for k in ('_layout_window', '_batch_norm_win', '_bead_norm_win'):
            if k in self.__dict__:
                setattr(self, k, None)

    def _clear_session_records(self):
        """Forget the old session's per-sample and per-session records
        that removing its samples leaves behind."""
        # The restore itself resets the rest: the staged per-file bundles,
        # the FMO controls, the tabs, the session's own labels and the
        # deferred restore (_stage_session_restore, _restore_channel_display,
        # _restore_view_pages).
        for d in (self._pending_sample_gates, self._pending_wsp_trial,
                  self._pending_wsp_axes, self._sample_colors,
                  self._sample_trial, self._sample_is_comp,
                  self._sample_set_aside, self._sample_notes,
                  self._sample_plot_enabled, self._cluster_labels,
                  self._sample_gates, self._sample_gate_seq,
                  self._sample_gate_order, self._path_to_name,
                  self._name_to_path):
            d.clear()
        self._trial_order = []
        self._restore_lost_operands = []
        self._restore_built_on = []
        after = self.__dict__.get('_load_settle_after')
        if after is not None:
            try:
                self.after_cancel(after)
            except TclError:
                pass
            self._load_settle_after = None
        for k in ('_offline_samples', '_kept_links', '_load_notes',
                  '_qc_removed_batch', '_uncompensated_imports',
                  '_unconverted_imports', '_unconverted_booleans',
                  '_transform_notes', '_last_loaded', '_sidecar_fallbacks',
                  '_sidecar_written', '_sidecar_by_digest',
                  '_sidecar_failures', '_plotcache_digests',
                  '_batch_norm_originals', '_batch_norm_lost',
                  '_last_unmix_pairs',
                  # Set by a pending change (the FCS export channel names)
                  # when it lands; harmless before.
                  '_restored_channel_labels'):
            self.__dict__.pop(k, None)
        self._last_unmix_qc = None
        self._last_unmix_spectra = None
        self._last_unmix_detectors = None
        self._layout_doc = None

    def _clear_session_view(self):
        """Put the channels, axes and population views back as a new
        window has them (the first sample of the new session sets the
        channels again; one whose files are all missing keeps none)."""
        self._channels = []
        self._channel_transform = {}
        self._channel_scale = {}
        self._channel_range = {}
        self._channel_labels = {}
        self._xy_choices = []
        self._color_choices = []
        for combo in (self.x_combo, self.y_combo, self.color_combo):
            combo['values'] = ()
            combo.set('')
        self._backgate = []
        self._backgate_hidden = set()
        self._gate_density_full = set()
        self._quad_set_seq = 0
        self._viewed_population = None
        self._population_views = {}
        self._kept_zoom = None
        self._home_view = None
        self._legend_pos_cache = {}
        self._refresh_gate_list()
        self._schedule_replot(0)

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
                self._load_session_path(backup, confirmed=True)
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
        self._restore_channel_display(data)
        self._restore_provenance(data)
        self._restore_cluster_labels(data)
        self._restore_view_pages(data)
        self._restore_display_settings(data)
        self._stage_session_restore(data)

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

    def _restore_channel_display(self, data):
        """Restore the session's channel scales, ranges and labels, and
        clear the FMO controls (this session's own come from its samples)."""
        self._fmo_controls = {}         # this session's own, from its samples
        self._channel_scale.update(
            {k: str(v) for k, v in (data.get('channel_scale') or {}).items()})
        # A corrupt range is skipped, not allowed to abort the session load.
        self._channel_range.update(
            parse_ranges(data.get('channel_range') or {}))
        # Kept apart too: the first sample to land replaces the editor's
        # labels with its own (_on_loaded), and these must win over those.
        self._session_channel_labels = {
            k: str(v) for k, v in (data.get('channel_labels') or {}).items()}
        self._channel_labels.update(self._session_channel_labels)

    def _restore_provenance(self, data):
        """Restore the provenance trail and the fixes applied or declined,
        then log the load itself so the reopened session records that it
        was reopened (and from where)."""
        from .audit import AuditLog
        from .channel_label_trail import explicit_labels_from_trail
        self._audit_log = AuditLog.from_list(data.get('audit') or [])
        # The labels set explicitly (relabels, templates), which an FCS
        # export writes in every file, rebuilt from the trail's
        # channel_labels.set records (_export_channel_labels).
        self._explicit_channel_labels = explicit_labels_from_trail(
            data.get('audit'), data.get('channel_labels'))
        # The fixes this session declined, per result (fixes.scan skips
        # them). Applied ones are read from the results' kept originals, so
        # ``fixes_applied`` is not read back. A bare id (a session saved
        # before declines were per result) is expanded once every sample
        # the session lists is in (_settle_legacy_declines).
        self._declined_fixes = [
            copy.deepcopy(x) if isinstance(x, dict) else str(x)
            for x in data.get('fixes_declined') or []
            if isinstance(x, dict) or (x is not None and str(x))]
        self._legacy_decline_samples = (
            [str(e.get('name')) for e in data.get('samples') or []
             if isinstance(e, dict) and e.get('name') is not None]
            if any(isinstance(x, str) for x in self._declined_fixes)
            else None)
        self._audit('session.load',
                    created=data.get('created', ''),
                    n_samples=len(data.get('samples', [])))
        win = self._audit_window
        if win is not None and win.winfo_exists():
            win.refresh(self._audit_log)    # the new trail, not the old one

    def _restore_cluster_labels(self, data):
        """Restore the per-sample cluster labels.

        cluster_labels round-trips through JSON, which stringifies the
        inner int cluster-id keys. Coerce them back to int so lookups by
        the numeric id (from the data column) hit."""
        for sname, lbls in (data.get('cluster_labels') or {}).items():
            if isinstance(lbls, dict):
                self._cluster_labels[sname] = _coerce_cluster_ids(lbls)

    def _restore_view_pages(self, data):
        """v2: restore the tray's tabs and the layout editor's pages, each
        best-effort (a failure is printed; the rest of the session loads)."""
        try:
            self._restore_tabs_state(data.get('view_tabs'))
        except Exception as exc:                      # noqa: BLE001 - the tabs are restored best-effort; the failure is noted and the rest of the session still loads
            note("session: tabs not restored", exc)
        if data.get('layout'):
            try:
                from .layout_page import LayoutDoc
                self._layout_doc = LayoutDoc.from_dict(data['layout'])
            except Exception as exc:                  # noqa: BLE001 - a layout that will not rebuild is noted; the rest of the session still loads
                note("session: layout not restored", exc)

    def _restore_max_points(self, text):
        """Put a saved 'Max points' value back in the box. Older versions
        read 0 as no cap, so a saved 0 comes back as All; any other value
        that is not a valid cap is left out with a logged note, and the box
        keeps the cap it shows."""
        from .editor_plot import parse_point_cap
        if text == '0':
            text = 'All'
        if parse_point_cap(text) is None:
            note(f"session: Max points {text!r} is not a valid cap; "
                 "kept the current one", None)
            return
        self.max_points_var.set(text)
        self._mp_last_good = text

    def _restore_display_settings(self, data):
        """Restore the plot display settings (downsampling, removed-event
        and contour toggles, histogram y mode, plot mode); a failure is
        reported and the rest of the session still loads."""
        try:
            self.ds_display_var.set(bool(data.get('downsample_display', True)))
            self.ds_propagate_var.set(bool(data.get('downsample_propagate', False)))
            self._sync_ds_mode_var()
            self._update_ds_visibility()   # hide Max points if restored Off
            if data.get('max_points'):
                self._restore_max_points(str(data['max_points']).strip())
            self.show_removed_var.set(bool(data.get('show_removed', False)))
            self.contour_scatter_var.set(
                bool(data.get('contour_scatter', True)))
            self.contour_outliers_var.set(
                bool(data.get('contour_outliers', True)))
            if data.get('hist_y_mode') in ('Fraction', 'Count', '% of Max'):
                self.hist_y_mode.set(data['hist_y_mode'])
            if data.get('plot_mode') in self.PLOT_MODES:
                self.mode_var.set(data['plot_mode'])
            self._update_mode_options()
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("restore display settings from the session file", exc)

    def _stage_session_restore(self, data):
        """Clear the per-sample restore bundles and remember the combo
        selections, plot toggles and active sample to restore once at
        least one sample has loaded (combos populate from sample 1)."""
        # Stage each sample's restore bundle — grouping (trial + Comps/Samples
        # override) AND its gates — keyed by FILE PATH so it survives name
        # disambiguation across reloads. `_on_loaded` drains it by the loaded
        # sample's path. Cleared first so a prior session's missing-file entries
        # can't leak onto a later load.
        self._pending_sample_meta.clear()
        self._pending_sample_prep.clear()
        self._session_restore = {
            'x': data.get('x_channel'),
            'y': data.get('y_channel'),
            'color': data.get('color_channel'),
            'active': data.get('active_sample'),
            'plot_enabled': {s['name']: s.get('plot_enabled', False)
                             for s in data.get('samples', [])},
        }

    # ── Samples whose files are missing ──────────────────────────────────────

    def _keep_offline(self, entry, gates):
        """Keep a session sample whose file was not found, with its gates, so
        the next save (or autosave) writes it back unchanged. Skipped, the
        save after it would drop the sample and every gate on it from the
        session for good."""
        e = offline_entry(entry, getattr(self, '_session_dir', '') or '')
        self.__dict__.setdefault('_offline_samples', {})[e['name']] = {
            'entry': e, 'gates': [dict(g) for g in gates or []]}

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
            base = saved_file_name(e)
            if base:
                want.setdefault(base.lower(), []).append(name)
        # Every file of a wanted name, per sample. Samples that share a file
        # name ('tube.fcs' from Day 6 and from Day 9) must not both take the
        # first one found: one sample's events would load under the other's
        # name and gates. Each takes the file whose folders match its saved
        # path's, and no file goes to two samples.
        cands: dict[str, list[str]] = {}
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                for name in want.get(f.lower(), ()):
                    cands.setdefault(name, []).append(os.path.join(root, f))
            if all(len(names) == 1 and names[0] in cands
                   for names in want.values()):
                break           # each wanted file name is one sample's: found
        hits = {}
        for names in want.values():
            ranked = sorted(
                ((-_folders_matching(off[nm]['entry'].get('path') or '', p),
                  i, nm, p)
                 for i, nm in enumerate(names) for p in cands.get(nm, ())),
                key=lambda r: r[:2])
            taken = set()
            for _score, _i, nm, p in ranked:
                if nm not in hits and p not in taken:
                    hits[nm] = p
                    taken.add(p)
        return self._load_relinked(hits, folder)

    def _load_relinked(self, hits, folder):
        """Load the offline samples named in `hits` (``{name: file}``)
        from those files, with their saved gates, and return
        ``(found, still_missing)`` sample names; `folder` is where they
        were looked for (for the audit trail)."""
        off = getattr(self, '_offline_samples', None) or {}
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

    def _relink_from_project(self, folder):
        """Load the offline samples' files from the unpacked project in
        `folder`, which the user chose: every match there
        (``project_copies.project_matches``, by sample name, file name and
        data set; verified or not). Returns ``(found, still_missing)``
        sample names."""
        from .project_copies import folder_files, project_matches
        off = getattr(self, '_offline_samples', None) or {}
        entries = [rec['entry'] for name, rec in off.items()
                   if name not in self._samples]
        hits = folder_files(project_matches(entries, [folder]), folder)
        return self._load_relinked(hits, folder)

    def _load_missing_from_project(self, source):
        """Load the offline samples from `source`, which the user chose: an
        unpacked project folder, or a project archive, reused when a copy
        of it was unpacked before, else unpacked now off the Tk thread with
        the megabytes counted on the status line (``project_copies``). The
        samples are those of the session open when it was chosen: one
        opened meanwhile is left alone."""
        from .project_archive import open_project_archive
        from .project_copies import find_copy, unpack_destination
        name = os.path.basename(source)
        report = self._unpack_progress(f"Unpacking project {name}")
        held = self.__dict__.get('_offline_samples')

        def work():
            if os.path.isdir(source):
                return source
            c = find_copy(source)
            if c is not None:
                return c.folder
            sess, _ = open_project_archive(
                source, unpack_destination(source), progress=report)
            return os.path.dirname(sess)

        def done(folder):
            if self.__dict__.get('_offline_samples') is not held:
                self.status_var.set(
                    f"Project {name} is ready, but another session was "
                    "opened meanwhile: nothing was loaded from it.")
                return
            found, still = self._relink_from_project(folder)
            if still:
                ui_errors.showwarning(
                    "Some files still missing",
                    f"Found {len(found)} of {len(found) + len(still)} in "
                    f"project {name}. Not found there: "
                    f"{', '.join(still[:8])}"
                    + (" …" if len(still) > 8 else '')
                    + "\n\nTheir gates stay in the session.", parent=self)
            else:
                self.status_var.set(
                    f"Found all {len(found)} missing sample file(s) in "
                    f"project {name}; loading them")

        def failed(exc):
            self.status_var.set(f"Open project failed: {exc}")
            show_error(self, "Can't open project",
                       "The project could not be opened.", exc, source)

        self.run_async(work, on_done=done, on_error=failed,
                       busy_msg=f"Opening project {name}…")

    def _missing_sources(self, names):
        """``[(project folder or archive, samples it may hold), ...]``
        beside the session for the offline samples `names`, most first:
        unpacked projects holding files of their names (not verified as
        the same recordings, or they would have loaded), then archives
        listing them whose copy is not among those. At most 3."""
        sess_dir = getattr(self, '_session_dir', '') or ''
        if not sess_dir:
            return []
        from .project_copies import (
            candidate_folders,
            find_copy,
            project_archives,
            project_folders,
            project_matches,
        )
        off = getattr(self, '_offline_samples', None) or {}
        entries = [off[n]['entry'] for n in names if n in off]
        out = candidate_folders(project_matches(entries,
                                                project_folders(sess_dir)))
        have = {f for f, _n in out}
        for arc in project_archives(sess_dir, names):
            c = find_copy(arc)
            if c is None or c.folder not in have:
                out.append((arc, 0))
        return out[:3]

    def _ask_missing_source(self, names, sources, shown):
        """Ask which project of `sources` (``_missing_sources``) to load
        the missing samples from, or to choose a folder; returns a source,
        'folder' or 'cancel'. When two projects hold as many samples,
        neither is the default button: the user picks."""
        lines = '\n'.join(
            f"  • {os.path.basename(src)}"
            + (f" ({n} of them)" if n else " (a project archive)")
            for src, n in sources)
        choices = [(f"Load from {_short(os.path.basename(src))}", src)
                   for src, _n in sources]
        tie = len(sources) > 1 and sources[0][1] == sources[1][1]
        choices += [("Choose folder…", 'folder'), ("Not now", 'cancel')]
        return self._ask_choice(
            "Sample files not found",
            f"The files for {len(names)} sample(s) were not found:\n\n"
            f"{shown}\n\nFiles of the same names are in:\n{lines}\n\n"
            "OpenFlo cannot tell whether they are the same recordings "
            "(instruments use the same file names every day), so check "
            "that this is the right project before loading from it.",
            choices, default=len(choices) - 1 if tie else 0,
            tips=[f"Load the missing samples from {os.path.basename(src)}, "
                  "with their saved gates (an archive is unpacked once, "
                  "then reused)." for src, _n in sources]
            + ["Pick a folder to look for the files in by name.",
               "Keep the samples offline: their gates stay in the "
               "session."])

    def _offer_locate_missing(self):
        """Name the samples whose files were not found and offer to load
        them from a project beside the session that holds files of their
        names (``_missing_sources``), or to look for them in a folder the
        user picks."""
        _, gates = _offline_parts(self)
        names = sorted(gates)
        if not names:
            return
        from tkinter import filedialog, messagebox
        shown = '\n'.join(f"  • {n}" for n in names[:12])
        if len(names) > 12:
            shown += f"\n  … and {len(names) - 12} more"
        sources = self._missing_sources(names)
        if sources:
            choice = self._ask_missing_source(names, sources, shown)
            self._audit('session.missing_choice', choice=choice)
            if choice == 'cancel':
                return
            if choice != 'folder':
                self._load_missing_from_project(choice)
                return
        elif not messagebox.askyesno(
                "Sample files not found",
                f"The files for {len(names)} sample(s) were not found:\n\n"
                f"{shown}\n\nTheir gates are kept and saved with the "
                "session. Look for the files in a folder now?", parent=self):
            return
        folder = filedialog.askdirectory(
            parent=self, title="Folder holding the missing sample files")
        if not folder:
            return
        found, still = self._relink_missing(folder)
        if still:
            ui_errors.showwarning(
                "Some files still missing",
                f"Found {len(found)} of {len(found) + len(still)}. Not found "
                f"in that folder: {', '.join(still[:8])}"
                + (" …" if len(still) > 8 else '')
                + "\n\nTheir gates stay in the session.", parent=self)
        else:
            self.status_var.set(f"Found all {len(found)} missing sample "
                                "file(s); loading them")

    def _seed_sample_name(self, name, pkey):
        """Register the session's saved `name` for the file at `pkey` (a
        normcased absolute path) before its load is queued. The loader names
        a raw FCS by `_sample_name_for`, which picks a fresh name: a sample
        saved disambiguated ('tube [Day 9]') would come back as 'tube', and
        its FMO mapping, colour, cluster labels and active-sample slot, all
        keyed by the saved name, would miss it. Left alone when the file or the name
        already belongs to a loaded sample, which then keeps its own."""
        held = self._path_to_name.get(pkey)
        if held is not None and held in self._samples:
            return
        other = self._name_to_path.get(name)
        if name in self._samples or (other is not None and other != pkey):
            return
        if held is not None:
            self._name_to_path.pop(held, None)
        self._path_to_name[pkey] = name
        self._name_to_path[name] = pkey

    def _restore_samples(self, samples, sample_gates, active=None):
        """Stage and queue a session's samples (each one's grouping, gates,
        transforms and matrix, keyed by its file path) and return the names
        whose files could not be found. Those are kept as offline samples
        (_keep_offline), so a save does not drop them. What to load and how
        is session_open's plan; this puts it into the editor."""
        plan = plan_samples(
            samples, sample_gates, getattr(self, '_session_dir', '') or '',
            data_dir=getattr(self, '_session_data_dir', '') or '')
        _note_kept_links(self, plan)
        for p in plan:
            self._stage_planned_sample(p)

        # Restore samples. Processed samples (a CSV carrying computed columns —
        # clusters, UMAP, …) load on the Tk thread; the rest queue as raw FCS on
        # the background pool. EITHER way we paint a ⏳ placeholder row for every
        # sample FIRST, so resuming a session fills the tree immediately and
        # each row then swaps to its real entry as that sample finishes — a big
        # session does not look frozen while it loads.
        processed_loads = self._processed_loads_if_deps(
            [(p.name, p.processed_path) for p in plan
             if p.staged and p.processed_path])
        proc_names = {nm for nm, _ in processed_loads}

        # Raw FCS paths (everything not loaded from a processed CSV). Missing
        # files are reported and kept offline, not queued.
        paths, gone = raw_loads(plan, proc_names)
        missing = [p.label for p in gone]
        for p in gone:
            self._keep_offline(p.entry, p.gates)

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
            # A path alone is the file as one sample; (path, n) its data
            # set n, for a file holding several.
            self._queue_fcs_loads([p if n is None else (p, n)
                                   for p, n in paths], front_names=front)
        if processed_loads:
            self._queue_processed_loads(processed_loads, front_names=front)
        self._refresh_gate_list()
        try:
            self.update_idletasks()
        except TclError:
            pass

        msg = f"Loading session: {len(paths) + len(processed_loads)} sample(s)"
        moved = sorted({os.path.basename(p.found_in) for p in plan
                        if p.found_in})
        if moved:
            n = sum(1 for p in plan if p.found_in)
            msg += (f" — {n} found in project {', '.join(moved)} (the same "
                    "recordings)")
            self._audit('session.found_in_project', samples=sorted(
                p.name for p in plan if p.found_in), folders=moved)
        if missing:
            msg += f" — missing FCS for: {', '.join(missing[:4])}"
            if len(missing) > 4:
                msg += f" (+{len(missing) - 4})"
        self.status_var.set(msg)
        return missing

    def _stage_planned_sample(self, p):
        """Put planned sample `p` (session_open.PlannedSample) into the
        editor before its load is queued: its saved name for its file, its
        FMO role and colour, its metadata (_pending_sample_meta) and how its
        file is prepared (stage_sample_prep), keyed by the path it loads
        from."""
        if not p.staged:
            return
        if p.load_found:            # a missing one is relinked later
            self._seed_sample_name(p.name, p.key)
        if p.fmo_of is not None:
            self._set_fmo_controls(dict.fromkeys(p.fmo_of, p.name))
        if p.color is not None and p.name not in self._samples:
            self._sample_colors[p.name] = p.color
        self._pending_sample_meta[p.key] = p.meta
        self.stage_sample_prep(p.load_path, **p.prep)

    @staticmethod
    def _processed_loads_if_deps(processed_loads):
        """`processed_loads`, or [] when the data deps are missing.

        openflo.pipeline → flowio; if the data deps aren't installed this
        raises. Degrade to loading those samples as raw FCS rather than
        crashing the restore (and the window) on startup."""
        if not processed_loads:
            return processed_loads
        try:
            import pandas  # noqa: F401  # pyright: ignore[reportUnusedImport]

            from . import pipeline  # noqa: F401  # pyright: ignore[reportUnusedImport]
        except Exception as exc:  # noqa: BLE001 - a broken data install loads the samples as raw FCS instead
            note(f"session: data deps unavailable, loading "
                 f"{len(processed_loads)} sample(s) as raw FCS instead", exc)
            return []
        return processed_loads

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
        # Once the saved sample is active and highlighted, later passes leave
        # the highlight to the user: re-asserting it on every pass replaced a
        # row they picked on another sample while the rest loaded.
        if active is None or (active not in self._samples
                              and active not in self._loading):
            info['active_shown'] = True     # nothing saved to put back
        if not info.get('active_shown'):
            if (active and active in self._samples
                    and self._active_sample != active):
                self._set_active_sample(active)
            # Highlight the saved sample's row before anything rebuilds the
            # tree. The load settle may have selected the sample active while
            # loading (the first to land); left there, the rebuild below
            # re-selects that row, _on_tree_select makes it active again and
            # _plot_sample_row ticks it, undoing both the saved highlight and
            # the saved ticks.
            shown = self._select_restored_active()
            info['active_shown'] = bool(
                active and shown and self._active_sample == active)
        # Still loading? Re-apply shortly so late samples pick up their state.
        # Bounded (~9s past the 600 ms) so a stuck load can't loop forever.
        if self._loading and _tries < 30:
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
        if not info.get('active_shown'):
            self._select_restored_active()
        self._schedule_replot(0)
        # Every sample is in: offer the fixes for results made before them.
        self._check_fixes_on_open()

    def _select_restored_active(self):
        """Select the active sample's row unless the selection already
        belongs to it, marked as an echo so _on_tree_select does not redo
        (or re-tick) anything for it. A missing row (still loading) waits for
        the next restore pass. True when the active sample's row (or a row
        of its) is selected."""
        active = self._active_sample
        if active is None:
            return False
        sel = self.gate_tv.selection()
        p = self._parse_iid(sel[0]) if sel else None
        if p and p[0] not in ('trial', 'subgroup') and p[1] == active:
            return True
        iid = self._sample_iid(active)
        if self.gate_tv.exists(iid):
            self._reselect_tree_row(iid)
            return True
        return False

    def _periodic_autosave(self):
        """Autosave the session every few minutes (primary window only), so a
        hard crash loses less than the close-time autosave alone. Reschedules
        itself. A failure is logged and shown in the status line; once
        `_AUTOSAVE_WARN_AFTER` in a row have failed, the user is told once
        (see `_autosave_finished`).

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
        except Exception as exc:  # noqa: BLE001 - counted and reported by _autosave_finished; the autosave reschedules itself
            self._autosave_finished(str(exc) or type(exc).__name__)
        try:
            self.after(self._AUTOSAVE_MS, self._periodic_autosave)
        except TclError:
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
            error = None
            try:
                _data, failed = self._run_session_write(plan)
                if failed:
                    error = ("computed results could not be written for "
                             + ', '.join(failed))
            except Exception as exc:               # noqa: BLE001 - worker thread: any autosave failure is reported on the Tk thread rather than killing the thread
                error = str(exc) or type(exc).__name__
            # Tk may only be touched from the main thread: hand the outcome
            # over to it, where the failure streak is counted and shown.
            tk_threads.call_on_main_thread(
                lambda: self._autosave_finished(error))

        self._autosave_thread = threading.Thread(
            target=work, name='openflo-autosave', daemon=False)
        self._autosave_thread.start()

    def _autosave_finished(self, error):
        """Count the outcome of one periodic autosave (Tk thread). `error`
        is None when it saved, else what went wrong.

        Each failure is logged and shown in the status line. A single
        failure can be passing (a full disk freed a moment later, a file
        briefly locked), so the user is only interrupted once
        `_AUTOSAVE_WARN_AFTER` autosaves IN A ROW have failed: by then a
        crash would lose real work. That warning is shown once per streak,
        not on every 5-minute tick; a successful autosave ends the streak,
        and a new streak warns again."""
        if error is None:
            if getattr(self, '_autosave_warned', False):
                log.warning("session: periodic autosave is working again")
                self._set_autosave_status("Autosave is working again.")
            self._autosave_failures = 0
            self._autosave_warned = False
            return
        n = getattr(self, '_autosave_failures', 0) + 1
        self._autosave_failures = n
        note(f"session: periodic autosave failed ({n} in a row): {error}", None)
        self._set_autosave_status(f"Autosave failed ({n} in a row): {error}")
        if (n < self._AUTOSAVE_WARN_AFTER
                or getattr(self, '_autosave_warned', False)
                or getattr(self, '_closing', False)):   # the close reports its own
            return
        self._autosave_warned = True
        minutes = max(1, round(self._AUTOSAVE_MS * n / 60_000))
        try:
            ui_errors.showwarning(
                "Autosave is failing",
                f"OpenFlo has not been able to autosave this session for "
                f"the last {n} tries (about {minutes} minutes):\n\n{error}\n\n"
                f"If OpenFlo or the computer stops unexpectedly, that work "
                f"cannot be restored. Use File → Save session to keep a "
                f"copy. OpenFlo keeps trying and will not ask again until "
                f"an autosave has worked.",
                parent=self)
        except TclError:
            pass

    def _set_autosave_status(self, text):
        var = getattr(self, 'status_var', None)
        if var is None:
            return
        try:
            var.set(text)
        except TclError:
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
        except Exception as exc:  # noqa: BLE001 - reported via note()
            note("update the recent-sessions list", exc)

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
