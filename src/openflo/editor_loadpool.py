"""Threaded FCS/CSV loading, progress bar, and sample removal.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox

from .editor_base import EditorMixin
from .errors import note
from .errors import note as _note_error
from .paths import sample_key
from .sample_prep import mark_processed_transforms

BASE = os.path.dirname(os.path.abspath(__file__))
# Per pool thread: the session epoch of the job it is running
# (LoadPoolMixin._post_load_result).
_JOB = threading.local()
_LOAD_POOL_SIZE = 2       # conservative fallback when auto-detection fails
_POOL_MIN, _POOL_MAX = 1, 8


def _default_pool_size():
    """A sensible concurrent-loader count for this machine.

    FCS loading is mostly numpy/pandas (QC, compensation, logicle). The numpy
    ops release the GIL, but the Python glue between them does NOT — so with
    many workers the AGGREGATE glue holds the GIL almost continuously and the Tk
    event loop is starved, freezing the window during a big resume. Past ~2
    workers there's little load-time win (the GIL serialises the glue) but a
    large responsiveness cost, so the default is capped at 2 — still off the UI
    thread and still parallel for the GIL-releasing parts, while leaving the Tk
    thread enough GIL time to stay movable. Also bounded by cores / RAM for
    low-end machines. Power users can override via the ``load_workers`` pref
    (range [1, 8]) if they prefer raw throughput over UI smoothness."""
    cores = os.cpu_count() or 2
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        ram_cap = max(1, int((ram_gb - 3.0) / 2.0))
    except Exception:  # noqa: BLE001 - noted; falls back to the default pool size
        note("could not read system RAM to size the load pool")
        ram_cap = _LOAD_POOL_SIZE
    return max(_POOL_MIN, min(2, cores - 1, ram_cap))


class LoadPoolMixin(EditorMixin):
    """The background load pool (worker threads + progress bar) plus CSV import, name disambiguation, and sample/trial removal."""

    _TRANSFORM_NOTES_SHOWN = 12     # notes listed in one dialog before "+N"

    def _note_transform_assumption(self, note):
        """Queue a mark_processed_transforms note for the next dialog."""
        if not note:
            return
        notes = getattr(self, '_transform_notes', None)
        if notes is None:
            notes = self._transform_notes = []
        notes.append(note)

    def _flush_transform_notes(self):
        """Show every queued note in ONE warning dialog, once per load batch.

        A status line is not enough: _on_load_settled / _show_sample_count
        rewrite it on the next event-loop turn, and a session restore's notes
        never reached the window at all (only the console)."""
        notes = getattr(self, '_transform_notes', None) or []
        if not notes:
            return
        self._transform_notes = []
        cap = self._TRANSFORM_NOTES_SHOWN
        shown = [f"• {n}" for n in notes[:cap]]
        if len(notes) > cap:
            shown.append(f"… and {len(notes) - cap} more (see the console).")
        for n in notes:
            note(f"load: {n}", None)
        messagebox.showwarning(
            "Scale assumed for CSV data",
            "These CSVs have no usable transform record, so OpenFlo assumed "
            "the scale of their columns:\n\n" + "\n\n".join(shown) +
            "\n\nA column of UNKNOWN scale is kept exactly as stored (its "
            "gates select what they did) but gets no linear statistics. "
            "Saving again records these assumptions with the data.",
            parent=self)

    def _import_processed_csv(self, path):
        """Load one processed events table (cluster / embedding columns) as
        a sample — used by the workspace Results 'Load in editor' action:
        a run's binary ``<label>_events.npz`` or the ``<label>_events.csv``
        of an earlier run (or of the 'Also write events as CSV' setting).
        Auto-disambiguates the name so re-importing a run doesn't collide.
        Returns the name or None."""
        import json

        from .flow_sample import FlowSample
        from .session_sidecar import read_sidecar
        base = os.path.basename(path).rsplit('.', 1)[0]
        for suf in ('_processed', '_events'):
            if base.endswith(suf):
                base = base[:-len(suf)]
        name, i = base, 2
        while name in self._samples:
            name = f"{base} ({i})"
            i += 1
        try:
            df = read_sidecar(path)
        except Exception as exc:  # noqa: BLE001 - a bad file is reported in the status line
            self.status_var.set(f"Load failed: {exc}")
            return None
        labels = None
        sidecar = os.path.join(os.path.dirname(path), f'{base}_labels.json')
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, encoding='utf-8') as fh:
                    labels = json.load(fh)
            except (OSError, ValueError):  # noted; an unreadable or malformed labels file just loads the sample without labels
                _note_error("could not read the labels file of a loaded sample")
                labels = None
        s = FlowSample.from_dataframe(df, name=name, labels=labels, path=path)
        # A workspace run's events table: OpenFlo's own data.
        note = mark_processed_transforms(s, path, provenance='events')
        self._on_loaded(name, s)
        self.status_var.set(
            f"Loaded '{name}' — use Edit → Populations to import its cluster "
            "column, or plot UMAP1/UMAP2." + (f"  [{note}]" if note else ''))
        self._note_transform_assumption(note)
        self._flush_transform_notes()
        return name

    def _sample_name_for(self, path, set_number=None, base=None):
        """Stable, collision-free key/display name for an FCS ``path``.

        Samples are keyed app-wide by name (``self._samples``, the per-sample
        gate stores, tree iids, the workspace, statistics…). Day-organised
        drops reuse filenames across days — e.g.
        ``Compensation Controls_…_008.fcs`` appears under Day 6 *and* Day 9 —
        so a bare basename would make the second file collide with the first
        and be silently skipped as 'already loaded'. We disambiguate a
        colliding basename with its day/trial (then a numeric counter as a last
        resort) and remember the path→name mapping, so repeat calls — and
        different callers like the ``.wsp`` ingest and the FCS queue — always
        resolve the same file to the same name.

        One data set of a file holding several (`set_number`) is a sample of
        its own, keyed apart from the file (``paths.sample_key``) and named
        from `base` (``fcs_data_sets.data_set_names``) by the same rule."""
        # normcase so the registry key is case-insensitive on Windows — a .wsp
        # whose stored path case differs from the on-disk folder (day6\ vs
        # Day6\) must map to the SAME file, not load it twice.
        ap = sample_key(path, set_number)
        cached = self._path_to_name.get(ap)
        if cached is not None:
            return cached
        from .sample_prep import sample_name_for
        if set_number and not base:
            base = (f"{os.path.basename(path).rsplit('.', 1)[0]} "
                    f"#{int(set_number)}")
        name = sample_name_for(
            path, lambda n: n in self._name_to_path or n in self._samples,
            base=base if set_number else None)
        self._path_to_name[ap] = name
        self._name_to_path[name] = ap
        return name

    def _pool_size(self):
        """Concurrent loader-thread count: a Preferences override (``load_workers``,
        clamped to [1, 8]) when set, else a hardware-derived default."""
        try:
            from .prefs import read_prefs
            pref = read_prefs().get('load_workers')
            if pref is not None:
                return max(_POOL_MIN, min(_POOL_MAX, int(pref)))
        except Exception as exc:  # noqa: BLE001 - noted; falls back to the hardware-derived pool size
            note("reading the load-workers preference", exc)
        return _default_pool_size()

    def _enqueue_load(self, payload, priority=1):
        """Put one load job on the priority queue. Lower ``priority`` runs first;
        a monotonic sequence counter breaks ties as FIFO. ``payload`` is the job
        tuple (raw FCS / 'csv' sidecar) or ``None`` (shutdown sentinel). Always
        called on the Tk thread, so the counter is single-writer.

        A job is stamped here, on the Tk thread, with the session it
        belongs to (``_load_epoch``, keyed by its sequence number): read by
        the worker after it takes the job, another session opened in
        between would pass for the job's own (_job_epoch)."""
        self._load_seq += 1
        if payload is not None:
            self.__dict__.setdefault('_load_job_epochs', {})[
                self._load_seq] = self.__dict__.get('_load_epoch', 0)
        self._load_queue.put((priority, self._load_seq, payload))

    def _job_epoch(self, seq):
        """The session epoch job `seq` was queued under (_enqueue_load),
        taken off the record; a job queued some other way takes the
        current one. Safe off the Tk thread (dict.pop is atomic)."""
        stamped = self.__dict__.get('_load_job_epochs')
        epoch = None if stamped is None else stamped.pop(seq, None)
        return self.__dict__.get('_load_epoch', 0) if epoch is None else epoch

    def _lower_loader_priority(self):
        """Best-effort: drop THIS worker thread's OS scheduling priority so a
        heavy batch of loads doesn't starve the UI thread. Windows uses
        SetThreadPriority(BELOW_NORMAL); Linux lowers the thread's nice. No-op
        elsewhere or on any failure."""
        try:
            if sys.platform == 'win32':
                import ctypes
                THREAD_PRIORITY_BELOW_NORMAL = -1
                h = ctypes.windll.kernel32.GetCurrentThread()
                ctypes.windll.kernel32.SetThreadPriority(
                    h, THREAD_PRIORITY_BELOW_NORMAL)
            elif hasattr(os, 'setpriority') and hasattr(os, 'gettid'):
                # setpriority / PRIO_PROCESS / gettid are Linux-only; the hasattr
                # guard makes this runtime-safe, but a non-Linux `os` stub omits
                # them, so the type checker flags the line (CI pyright did).
                os.setpriority(os.PRIO_PROCESS, os.gettid(), 10)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - thread priority is a nicety
            pass

    def _ensure_load_pool(self):
        """Spawn the loader-thread pool once, lazily on the first enqueue. The
        size comes from `_pool_size()` (hardware default or a Preferences
        override). Daemon threads so they never block process exit; touched only
        on the main thread, so the one-shot guard is race-free."""
        if self._load_pool_started:
            return
        for i in range(self._pool_size()):
            t = threading.Thread(target=self._load_pool_worker,
                                 name=f'fcs-load-{i}', daemon=True)
            t.start()
            self._load_pool.append(t)
        self._load_pool_started = True

    def _load_pool_worker(self):
        """Pool worker: block on the priority queue, load one sample at a time.
        A ``None`` payload is the shutdown sentinel. ``_load_worker`` /
        ``_load_csv_worker`` run the whole pipeline off-thread and post
        ``_on_loaded`` / ``_on_load_error`` to the Tk thread via ``self.after``;
        the ``finally`` posts a completion tick so progress advances even when a
        load raises."""
        self._lower_loader_priority()
        gen = self._load_gen
        while True:
            try:
                _prio, _seq, job = self._load_queue.get()
            except Exception:  # noqa: BLE001 - the worker just exits if its queue is broken
                break
            if job is None:
                break
            if self._load_stop.is_set() or self._load_gen != gen:
                # Closing, or this pool was replaced after a cancelled close:
                # leave the job queued, so a close the user cancels (its save
                # failed) still loads this sample.
                self._load_queue.put((_prio, _seq, job))
                break
            # Jobs are (name, path) for raw FCS read as one sample, (name,
            # path, 'fcs', set number) for one data set of a file holding
            # several, or (name, path, 'csv', labels) for a processed-data
            # sidecar.
            name, path, *rest = job
            kind = rest[0] if rest else 'fcs'
            # The session this job belongs to: a result posted after another
            # session opened in this window lands nowhere (_post_load_result).
            epoch = _JOB.epoch = self._job_epoch(_seq)
            try:
                if kind == 'csv':
                    labels = rest[1] if len(rest) > 1 else {}
                    self._load_csv_worker(name, path, labels)
                elif len(rest) > 1:
                    self._load_worker(name, path, rest[1])
                else:
                    self._load_worker(name, path)
            except Exception as exc:  # noqa: BLE001 - printed to the log; backstop so one bad load cannot kill a pool thread
                # _load_worker catches its own pipeline errors; this is a
                # backstop so an unexpected throw can't permanently kill a pool
                # thread (which would shrink the pool and stall the queue).
                note(f"load: pool worker error for {name}", exc)
            finally:
                # Tally on the Tk thread, not here: two pool workers writing
                # `self._load_done += 1` concurrently would race (load-add-store
                # spans several bytecodes; the GIL can interleave them and drop
                # an increment, so the bar would never reach N/N). Posting the
                # tick keeps `_load_total`/`_load_done` single-writer.
                try:
                    self._post_load_result(self._mark_one_done, epoch)
                except Exception:  # noqa: BLE001 - the window is gone, so the worker just exits
                    # Window/interpreter gone — nothing left to update.
                    break

    def _mark_one_done(self):
        """Tk-thread: record that one load finished (success or error) and
        refresh the bar. Counter writes live only here + on enqueue, both on the
        main thread, so no lock is needed."""
        self._load_done += 1
        self._update_progress_bar()

    def _update_progress_bar(self):
        """Reflect the load counters in the bar. Runs on the Tk thread (always
        reached via ``self.after``). Hidden when nothing is queued; shown and
        sized to ``_load_total`` otherwise; schedules a brief auto-hide once the
        run drains."""
        try:
            total, done = self._load_total, self._load_done
            if total <= 0:
                self._load_progress_frame.grid_remove()
                return
            self._load_progress_frame.grid()
            self.progress_bar.configure(maximum=total)
            self._load_progress_var.set(done)
            self._load_progress_lbl_var.set(f'{done}/{total} loaded')
            if done >= total:
                # Linger briefly at N/N, then hide+reset (re-checked in
                # _finish_progress so a mid-delay drop keeps the bar alive).
                self.after(800, self._finish_progress)
        except Exception:  # noqa: BLE001 - progress bar may already be gone
            pass

    def _finish_progress(self):
        """Hide + reset the progress bar, but only if the run is still complete
        — files dropped during the 800 ms linger extend ``_load_total``, in
        which case we leave the bar running."""
        try:
            if self._load_total > 0 and self._load_done >= self._load_total:
                self._load_total = 0
                self._load_done = 0
                self._load_progress_var.set(0)
                self._load_progress_lbl_var.set('')
                self._load_progress_frame.grid_remove()
        except Exception:  # noqa: BLE001 - progress bar may already be gone
            pass

    def _sample_lb_insert_loading(self, name):
        # Insert a placeholder sample row into the merged tree; it'll
        # be replaced with the proper '■ <name>' row once _on_loaded
        # fires (or removed by _on_load_error on failure). The tree no longer
        # shows what `_refresh_gate_list` last built, so its next call rebuilds.
        self._tree_shown_ops = None
        try:
            self.gate_tv.insert(
                '', 'end', iid=self._sample_iid(name),
                text=f'⏳ {name}', values=('',),
                tags=('loading',))
        except tk.TclError:
            pass

    def _take_pending_prep(self, path, set_number=None):
        """Pop what a .wsp import, a session restore or a sample transfer
        staged for the file at `path` (data set `set_number` of a file
        holding several; see `stage_sample_prep`), or None. dict.pop is
        atomic, so a pool thread may take it."""
        pend = self._pending_sample_prep
        if not pend or not path:
            return None
        return pend.pop(sample_key(path, set_number), None)

    def stage_sample_prep(self, path, comp=None, source_fcs=None, qc=None,
                          qc_removed=None, lmd_data_set=None,
                          instrument_compensated=False, set_number=None):
        """Stage, on the Tk thread and BEFORE the load is queued, how the file
        at `path` is to be prepared: `comp` is {'channels', 'matrix',
        'source'} -- the matrix this sample was compensated with where it came
        from -- `source_fcs` the raw FCS behind a processed CSV, `qc`
        whether acquisition QC ran on it there (None: the Preferences
        setting) and `qc_removed` how many events that QC removed (put back
        on a processed CSV, which holds only the events kept);
        `lmd_data_set` which copy of an .lmd `source_fcs` the values are and
        `instrument_compensated` whether the instrument compensated them
        (sample_prep.sidecar_sample); `set_number` which data set of a file
        holding several the sample is (the raw file read, or `source_fcs`).
        A record for one data set is kept apart from the file's
        (``paths.sample_key``)."""
        if not path or (comp is None and not source_fcs and qc is None
                        and not qc_removed and not instrument_compensated
                        and not set_number):
            return
        pend = self._pending_sample_prep
        if pend is None:
            pend = self._pending_sample_prep = {}
        pend[sample_key(path, set_number)] = {
            'comp': comp, 'source_fcs': source_fcs, 'qc': qc,
            'qc_removed': qc_removed, 'lmd_data_set': lmd_data_set,
            'instrument_compensated': instrument_compensated,
            'set_number': set_number}

    def _load_worker(self, name, path, set_number=None):
        """Pool worker for raw FCS: read and prepare the file at `path`
        (data set `set_number` of a file holding several, None for the file
        as one sample) with what was staged for it, and post the result."""
        try:
            prep = self._take_pending_prep(path, set_number)
            if set_number:
                prep = {**(prep or {}), 'set_number': set_number}
            s = self._prepared_fcs_sample(path, prep)
            # A session sample that saved its numbers: hashed here, off the
            # Tk thread, for the restore to check them against.
            meta = self._pending_sample_meta.get(
                sample_key(path, set_number)) or {}
            if meta.get('saved_stats'):
                self._plotcache_note_loaded_digest(s)
            self._post_load_result(lambda: self._on_loaded(name, s))
        except Exception as exc:  # noqa: BLE001 - handed to the Tk thread's load-error report
            # Bind exc as a default arg — `except … as exc` deletes the name
            # at block exit, so the bare lambda would NameError when fired.
            self._post_load_result(
                lambda e=exc: self._on_load_error(name, e))

    def _post_load_result(self, fn, epoch=None):
        """Run `fn` (a load's result: landing, error or progress tick) on
        the Tk thread, unless another session opened in this window since
        the load started (editor_session._drop_queued_loads bumps
        ``_load_epoch``): the old session's sample must not land in the new
        one. `epoch` defaults to the one the pool worker running this load
        took its job under; a load run outside the pool takes the current
        one."""
        if epoch is None:
            epoch = getattr(_JOB, 'epoch', None)
        if epoch is None:
            epoch = self.__dict__.get('_load_epoch', 0)

        def land():
            if self.__dict__.get('_load_epoch', 0) == epoch:
                fn()
        self.after(0, land)

    def _prepared_fcs_sample(self, path, prep=None):
        """The FlowSample for the raw FCS at `path`, prepared as the loader
        prepares every one (sample_prep.prepare_fcs_sample), following what
        `prep` (stage_sample_prep) says, with this window's labels and the
        Preferences QC setting. Safe off the Tk thread."""
        from .sample_prep import prepare_fcs_sample
        qc_default = False
        if (prep or {}).get('qc') is None:
            # Read only when nothing was staged.
            from .prefs import load_qc_enabled
            qc_default = load_qc_enabled()
        return prepare_fcs_sample(path, prep, labels=self.labels_str,
                                  qc_default=qc_default)

    def _load_csv_worker(self, name, path, labels):
        """Pool worker for a processed-data sidecar CSV (workspace results
        carrying cluster / UMAP / … columns). Reads + builds the FlowSample
        off the Tk thread — like ``_load_worker`` but skipping the FCS QC /
        compensation / transform, which are already baked into the saved data —
        then posts ``_on_loaded`` back to the Tk thread. ``labels`` is a snapshot
        taken at enqueue time so this never touches shared editor state."""
        try:
            from .sample_prep import read_sidecar_checked, sidecar_sample
            prep = self._take_pending_prep(path) or {}
            # A sidecar cut short (a crash mid-write) must not load as the
            # whole sample (401 of 1,000 events, silently). When it can't be
            # read, or holds fewer rows than its record says, the sample is
            # rebuilt from its raw FCS instead, and the user is told.
            df, problem = read_sidecar_checked(path)
            if problem:
                fcs = prep.get('source_fcs')
                if not (fcs and os.path.isfile(fcs)):
                    raise ValueError(
                        f"{os.path.basename(path)}: {problem}, and the "
                        "sample's FCS file was not found")
                raw = self._prepared_fcs_sample(fcs, prep)

                def _land_raw():
                    # The restore staged this sample's gates and grouping
                    # under the sidecar's path; it now comes from the FCS.
                    meta = self._pending_sample_meta
                    old = os.path.normcase(os.path.abspath(path))
                    if old in meta:
                        meta[sample_key(fcs, prep.get('set_number'))] = (
                            meta.pop(old))
                    self._on_loaded(name, raw)
                    self._note_sidecar_fallback(name, problem)
                self._post_load_result(_land_raw)
                return
            # The matrix, raw FCS (and its keywords) and QC record go back
            # on it; the note names any scale assumed, shown at the end of
            # the batch (_on_load_settled), so a restore shows one dialog.
            s, note = sidecar_sample(df, path, prep, name=name,
                                     labels=labels)
            # Remember that this file holds exactly this sample's data, so
            # the next save copies it, or leaves it, instead of encoding the
            # whole table again. Hashed here, off the Tk thread.
            remember = getattr(self, '_remember_sidecar', None)
            if remember is not None:
                from .editor_session import frame_digest
                digest = frame_digest(s.data)
                if digest is not None:
                    remember(path, digest)
                    self._plotcache_note_loaded_digest(s, digest)

            def _land():
                self._on_loaded(name, s)
                self._note_transform_assumption(note)
            self._post_load_result(_land)
        except Exception as exc:  # noqa: BLE001 - handed to the Tk thread's load-error report
            self._post_load_result(
                lambda e=exc: self._on_load_error(name, e))

    def _note_sidecar_fallback(self, name, problem):
        """Tell the user, once per batch, which samples were rebuilt from
        their FCS because their saved data file was incomplete: anything
        computed on them (clusters, embeddings) has to be run again."""
        pending = self.__dict__.setdefault('_sidecar_fallbacks', [])
        pending.append((name, problem))
        self._audit('sample.sidecar_fallback', sample=name, problem=problem)
        if len(pending) > 1:
            return

        def show():
            items = list(self._sidecar_fallbacks)
            self._sidecar_fallbacks.clear()
            from tkinter import messagebox
            messagebox.showwarning(
                "Saved data incomplete",
                "The saved data for these samples was incomplete, so they "
                "were reloaded from their FCS files:\n\n"
                + '\n'.join(f"  • {n}: {p}" for n, p in items[:12])
                + ("\n  …" if len(items) > 12 else '')
                + "\n\nGates are kept. Clusters, embeddings and other "
                "computed results for them need to be run again.",
                parent=self)
        self.after(500, show)

    def _queue_processed_loads(self, items, front_names=()):
        """Queue processed-data sidecar CSVs on the same bounded pool as raw FCS,
        so resuming a session with big workspace sidecars stays responsive
        instead of blocking the window. ``items`` is ``[(name, csv_path)]``;
        names in ``front_names`` load first (priority 0). Inserts each as a ⏳
        placeholder and counts it in the progress bar. Labels are snapshotted
        here (on the Tk thread) and handed to the worker so the background read
        never iterates shared editor state."""
        if not items:
            return
        self._ensure_load_pool()
        front = set(front_names)
        labels = dict(self._channel_labels)         # Tk-thread snapshot
        n = 0
        for name, path in items:
            if name in self._samples or name in self._loading:
                continue
            self._loading.add(name)
            ap = os.path.normcase(os.path.abspath(path))
            self._name_to_path.setdefault(name, ap)
            self._path_to_name.setdefault(ap, name)
            self._sample_lb_insert_loading(name)
            self._enqueue_load((name, path, 'csv', labels),
                               priority=0 if name in front else 1)
            self._load_total += 1
            n += 1
        if n:
            self._update_progress_bar()

    def _on_load_error(self, name, exc):
        self._loading.discard(name)
        self._tree_shown_ops = None
        try:
            self.gate_tv.delete(self._sample_iid(name))
        except tk.TclError:
            pass
        # A failed load must resolve its slot in a staged cross-instance move
        # as NOT taken, so the source keeps its copy instead of deleting a
        # sample that never arrived here.
        note = getattr(self, '_note_inbound_landed', None)
        if note is not None:
            p = (self._name_to_path or {}).get(name)
            if p:
                note(os.path.normcase(os.path.abspath(p)), ok=False)
        self.status_var.set(f"Failed to load {name}: {exc}")
        if not self._loading:          # the batch ended on a failure
            self._flush_transform_notes()

    def _remove_selected(self):
        """Remove the selected SAMPLE(s), or — if any TRIAL row is selected —
        every sample (and its gates) in those trials. Gate-row selections are
        ignored (use Clear gate for those)."""
        sel = self.gate_tv.selection()
        if not sel:
            self.status_var.set("Select a sample or trial row to remove "
                                "from the session.")
            return
        parsed = [self._parse_iid(s) for s in sel]
        trials  = [p[1] for p in parsed if p and p[0] == 'trial']
        samples = [p[1] for p in parsed if p and p[0] == 'sample']
        if trials:
            self._remove_trials(trials)
            return
        if not samples:
            self.status_var.set("Select a sample or trial row to remove "
                                "(use Clear gate for gates).")
            return
        if not self._confirm_remove_samples(samples):
            return
        n = self._remove_samples(samples)
        self.status_var.set(f"Removed {n} sample(s) from the session "
                            f"(all tabs).")

    @staticmethod
    def _samples_label(samples):
        """``sample 's1'`` or ``3 samples ('s1', 's2', 's3')``; past five
        names, ``and N more``."""
        if len(samples) == 1:
            return f"sample '{samples[0]}'"
        shown = ', '.join(f"'{n}'" for n in samples[:5])
        more = f" and {len(samples) - 5} more" if len(samples) > 5 else ""
        return f"{len(samples)} samples ({shown}{more})"

    def _confirm_remove_samples(self, samples):
        """Ask before unloading ``samples`` from the session: it drops their
        whole gate tree and is not on the undo stack (samples hold large
        frames, and every tab keeps its own copy of their state). Names the
        samples and their gate count, and points to Remove from this tab,
        which keeps them loaded. Same wording as Remove trial."""
        samples = [n for n in samples if n in self._samples]
        if not samples:
            self.status_var.set("The selected sample(s) are still loading; "
                                "remove them once they have loaded.")
            return False
        n_gates = sum(len(self._sample_gates.get(n) or {}) for n in samples)
        label = self._samples_label(samples)
        it = 'it' if len(samples) == 1 else 'them'
        hint = ("" if self._tabs().active.is_all else
                f"\nTo keep {it} loaded and only take {it} off this tab, "
                f"use Remove ▸ From this tab.")
        return messagebox.askyesno(
            "Remove sample",
            f"Remove {label} and {n_gates} gate(s) from the session "
            f"(all tabs)?\nThis can't be undone.{hint}",
            default=messagebox.NO, parent=self)

    def _trial_members_in_tab(self, trial):
        """``trial``'s loaded samples that the active tab shows (every member
        on the All tab) — what a trial row in the tree stands for — and the
        members the tab hides, as ``(shown, hidden)``."""
        shown, hidden = [], []
        for n in self._trial_members(trial):
            (shown if self._tab_has_sample(n) else hidden).append(n)
        return shown, hidden

    def _remove_trials(self, trials):
        """Remove the samples (+ gates) a trial row stands for from the
        session: on the All tab every member of ``trials``, on another tab
        only the members that tab shows — the hidden ones stay loaded.
        Confirmed, naming the samples, because — like single-sample Remove —
        it isn't on the undo stack."""
        members, hidden = [], []
        for t in trials:
            shown, hid = self._trial_members_in_tab(t)
            members.extend(n for n in shown if n not in members)
            hidden.extend(n for n in hid if n not in hidden)
        if not members and not hidden:
            # Empty trial header(s) — just forget them.
            for t in trials:
                if t in self._trial_order:
                    self._trial_order.remove(t)
            self._refresh_gate_list()
            return
        label = (f"trial '{trials[0]}'" if len(trials) == 1
                 else f"{len(trials)} trials")
        if not members:
            self.status_var.set(
                f"This tab shows no sample of {label}; its "
                f"{len(hidden)} sample(s) stay loaded. To remove the whole "
                f"trial, use the All tab.")
            return
        n_gates = sum(len(self._sample_gates.get(n) or {}) for n in members)
        them = 'it' if len(members) == 1 else 'them'
        hint = ""
        if not self._tabs().active.is_all:
            if hidden:
                hint += (f"\nThe {self._samples_label(hidden)} of {label} "
                         f"this tab doesn't show "
                         f"{'stays' if len(hidden) == 1 else 'stay'} "
                         f"loaded; to remove the whole trial, use the All "
                         f"tab.")
            hint += (f"\nTo keep {them} loaded and only take {them} off "
                     f"this tab, use Remove trial from this tab.")
        if not messagebox.askyesno(
                "Remove trial",
                f"Remove {label} — {self._samples_label(members)} and "
                f"{n_gates} gate(s) — from the session (all tabs)?"
                f"\nThis can't be undone.{hint}",
                default=messagebox.NO, parent=self):
            return
        self._remove_samples(members)
        kept = (f"; {len(hidden)} sample(s) this tab doesn't show stay loaded"
                if hidden else "")
        self.status_var.set(f"Removed {label} ({len(members)} sample(s)) "
                            f"from the session{kept}.")

    def _remove_samples(self, names):
        """Tear down a list of samples completely: FlowSample, gate tree,
        colours, plot/display state, cluster labels, trial membership. Rebinds
        the active sample if it was removed, drops now-empty trials, and
        refreshes. Returns the count removed. (Not undoable — samples hold
        large frames; matches the historic single-sample Remove.)"""
        removed = 0
        for name in list(names):
            if name not in self._samples:
                continue
            del self._samples[name]
            self._note_unsaved()
            self._sample_colors.pop(name, None)
            if name in self._sample_order:
                self._sample_order.remove(name)
            self._sample_gates.pop(name, None)
            self._forget_undo_samples([name])
            self._sample_gate_seq.pop(name, None)
            self._sample_gate_order.pop(name, None)
            self._sample_plot_enabled.pop(name, None)
            self._cluster_labels.pop(name, None)
            self._sample_trial.pop(name, None)
            self._sample_is_comp.pop(name, None)
            self._sample_set_aside.pop(name, None)
            self._sample_notes.pop(name, None)
            (getattr(self, '_batch_norm_originals', None) or {}).pop(name,
                                                                     None)
            ap = self._name_to_path.pop(name, None)
            if ap is not None:
                self._path_to_name.pop(ap, None)
            for ckey in [k for k in self._autoclean_cache if k[0] == name]:
                self._autoclean_cache.pop(ckey, None)
            for ckey in [k for k in self._autoclean_count_cache if k[0] == name]:
                self._autoclean_count_cache.pop(ckey, None)
            for ckey in [k for k in self._autoclean_method_cache if k[0] == name]:
                self._autoclean_method_cache.pop(ckey, None)
            # Its display caches and any queued background work go with it.
            self._plotcache_forget(name)
            # And every tab forgets it, so a sample loaded later under the
            # same name doesn't reappear in the tabs this one was in.
            self._tabs().forget_sample(name)
            removed += 1
        if not removed:
            return 0
        # A gate clip copied from a removed sample names ids of a tree that is
        # gone, and the sample loaded again under that name restarts them at
        # g1: paste it as one from another sample (fresh ids, no outside
        # operands kept), not back under ids that now name other gates.
        if self._clip_kind == 'gate_tree':
            for g in self._clip_payload or ():
                if g.get('_clip_sample') not in self._samples:
                    g['_clip_sample'] = None
        # Keep only trials that still have a loaded sample, preserving order.
        self._trial_order = [t for t in self._trial_order
                             if any(self._sample_trial.get(n) == t
                                    for n in self._samples)]
        if self._active_sample not in self._samples:
            self._set_active_sample(next(iter(self._samples), None))
        self._refresh_gate_list()
        self._schedule_replot(0)
        return removed
