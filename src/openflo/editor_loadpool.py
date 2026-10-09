"""Threaded FCS/CSV loading, progress bar, and sample removal.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import sys
import threading
from tkinter import messagebox

from .editor_base import EditorMixin

BASE = os.path.dirname(os.path.abspath(__file__))
_LOAD_POOL_SIZE = 2       # conservative fallback when auto-detection fails
_POOL_MIN, _POOL_MAX = 1, 8


def mark_processed_transforms(sample, csv_path=None, provenance='foreign'):
    """Record the transform each column of a sample built from a CSV carries.
    Returns a note naming what was assumed ('' when nothing was).

    Every CSV OpenFlo writes has a record beside it (pipeline.
    write_transforms_sidecar); when it is there and matches the CSV, it is
    the answer. Without one, `provenance` decides:

    * ``'session'`` (a session's processed-data sidecar) or ``'events'`` (a
      workspace _events.csv) -- a CSV an older OpenFlo wrote from the
      editor's own data: its fluor columns are display-TRANSFORMED by
      definition, never linear. One whose finite values lie in the loader's
      logicle range is logicle (the default), however few or alike they are
      (one event, an empty channel); one outside it carries a transform that
      cannot be known (asinh, ...) and is recorded UNKNOWN: kept exactly as
      stored, never re-transformed, no linear statistics. Taking it as
      linear and re-transforming it emptied asinh gates (2.6.1 session:
      215 -> 0 events). The UNKNOWN entry names its origin -- the session's
      data folder, or this one file -- so a pooled run can tell columns that
      share one old editor's transform from ones that may not.
    * ``'foreign'`` -- File > Load CSV: a column is GUESSED logicle only if
      all its finite values lie in that range and are not all equal (an
      empty channel would invert to -110.87), else taken as linear (a linear
      export guessed logicle read back as 5.2e44).

    Either way 'U:' and 'MESF:' columns are linear (that is how OpenFlo
    wrote them; 2.6.1's U: values guessed logicle went 0.436 -> 841), and
    non-float columns (cluster ids, '_pos' flags) are labels."""
    import numpy as np
    import pandas as pd

    from .pipeline import (
        logicle_display_bounds,
        read_transforms_sidecar,
        transform_spec,
        unknown_spec,
    )
    ours = provenance in ('session', 'events')       # an older OpenFlo's CSV
    label = os.path.basename(csv_path) if csv_path else sample.name
    problem = ''
    if csv_path:
        try:
            rec = read_transforms_sidecar(csv_path, sample.data.columns,
                                          len(sample.data))
        except ValueError as exc:
            rec, problem = None, str(exc)
        if rec is not None:
            sample.data_transforms = rec
            return ''
    lo, hi = logicle_display_bounds()
    logicle, unknown, linear = [], [], []
    for c in sample.fluor_channels:
        if not pd.api.types.is_float_dtype(sample.data[c]):
            continue
        if str(c).startswith(('U:', 'MESF:')):
            linear.append(c)
            continue
        v = np.asarray(sample.data[c].values, dtype=float)
        v = v[np.isfinite(v)]
        if ours:
            # Display-transformed by definition: one event, or a constant
            # (empty) channel, in range is logicle like any other.
            in_range = not v.size or (lo <= v.min() and v.max() <= hi)
        else:
            in_range = bool(v.size) and lo <= v.min() < v.max() <= hi
        if in_range:
            logicle.append(c)
        elif ours:
            unknown.append(c)
        else:
            linear.append(c)
    where = os.path.normcase(os.path.abspath(csv_path or sample.name))
    origin = (f"session:{os.path.dirname(where)}" if provenance == 'session'
              else f"file:{where}")
    rec = {c: transform_spec('logicle') for c in logicle}
    rec.update({c: unknown_spec(origin) for c in unknown})
    sample.data_transforms = rec
    if not (logicle or unknown or linear or problem):
        return ''

    def _names(cols):
        return ', '.join(map(str, cols[:6])) + (
            f' (+{len(cols) - 6})' if len(cols) > 6 else '')
    why = (f"its transform record was ignored ({problem})" if problem
           else "no transform record"
           + (" (written by an older OpenFlo)" if ours else ""))
    parts = []
    if logicle:
        parts.append(f"GUESSED logicle from their range: {_names(logicle)}")
    if unknown:
        parts.append(f"scale UNKNOWN, kept as stored, no linear statistics: "
                     f"{_names(unknown)}")
    if linear:
        parts.append(f"taken as linear: {_names(linear)}")
    return f"{label}: {why} — {'; '.join(parts) or 'no fluor columns'}."


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
    except Exception:
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
            print(f"[load] {n}", flush=True)
        messagebox.showwarning(
            "Scale assumed for CSV data",
            "These CSVs have no usable transform record, so OpenFlo assumed "
            "the scale of their columns:\n\n" + "\n\n".join(shown) +
            "\n\nA column of UNKNOWN scale is kept exactly as stored (its "
            "gates select what they did) but gets no linear statistics. "
            "Saving again records these assumptions with the data.",
            parent=self)

    def _import_processed_csv(self, path):
        """Load one processed CSV (cluster / embedding columns) as a sample —
        used by the workspace Results 'Load in editor' action. Auto-disambiguates
        the name so re-importing a run doesn't collide. Returns the name or None."""
        import json

        import pandas as pd

        from .pipeline import FlowSample
        base = os.path.basename(path).rsplit('.', 1)[0]
        for suf in ('_processed', '_events'):
            if base.endswith(suf):
                base = base[:-len(suf)]
        name, i = base, 2
        while name in self._samples:
            name = f"{base} ({i})"
            i += 1
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self.status_var.set(f"Load failed: {exc}")
            return None
        labels = None
        sidecar = os.path.join(os.path.dirname(path), f'{base}_labels.json')
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, encoding='utf-8') as fh:
                    labels = json.load(fh)
            except Exception:
                labels = None
        s = FlowSample.from_dataframe(df, name=name, labels=labels, path=path)
        # A workspace run's _events.csv: OpenFlo's own data.
        note = mark_processed_transforms(s, path, provenance='events')
        self._on_loaded(name, s)
        self.status_var.set(
            f"Loaded '{name}' — use Edit → Populations to import its cluster "
            "column, or plot UMAP1/UMAP2." + (f"  [{note}]" if note else ''))
        self._note_transform_assumption(note)
        self._flush_transform_notes()
        return name

    def _sample_name_for(self, path):
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
        resolve the same file to the same name."""
        # normcase so the registry key is case-insensitive on Windows — a .wsp
        # whose stored path case differs from the on-disk folder (day6\ vs
        # Day6\) must map to the SAME file, not load it twice.
        ap = os.path.normcase(os.path.abspath(path))
        cached = self._path_to_name.get(ap)
        if cached is not None:
            return cached
        base = os.path.basename(path).rsplit('.', 1)[0]
        name = base
        if name in self._name_to_path or name in self._samples:
            from .workspace import derive_trial_name
            trial = derive_trial_name(path)
            name = f'{base} [{trial}]'
            n = 2
            while name in self._name_to_path or name in self._samples:
                name = f'{base} [{trial}] ({n})'
                n += 1
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
        except Exception:
            pass
        return _default_pool_size()

    def _enqueue_load(self, payload, priority=1):
        """Put one load job on the priority queue. Lower ``priority`` runs first;
        a monotonic sequence counter breaks ties as FIFO. ``payload`` is the job
        tuple (raw FCS / 'csv' sidecar) or ``None`` (shutdown sentinel). Always
        called on the Tk thread, so the counter is single-writer."""
        self._load_seq += 1
        self._load_queue.put((priority, self._load_seq, payload))

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
        except Exception:
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
        while True:
            try:
                _prio, _seq, job = self._load_queue.get()
            except Exception:
                break
            if job is None or self._load_stop.is_set():
                break
            # Jobs are (name, path) for raw FCS or (name, path, 'csv', labels)
            # for a processed-data sidecar. Unpack tolerantly so a bare 2-tuple
            # (the historic shape) still means a raw-FCS load.
            name, path, *rest = job
            kind = rest[0] if rest else 'fcs'
            try:
                if kind == 'csv':
                    labels = rest[1] if len(rest) > 1 else {}
                    self._load_csv_worker(name, path, labels)
                else:
                    self._load_worker(name, path)
            except Exception as exc:
                # _load_worker catches its own pipeline errors; this is a
                # backstop so an unexpected throw can't permanently kill a pool
                # thread (which would shrink the pool and stall the queue).
                print(f"[load] pool worker error for {name}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            finally:
                # Tally on the Tk thread, not here: two pool workers writing
                # `self._load_done += 1` concurrently would race (load-add-store
                # spans several bytecodes; the GIL can interleave them and drop
                # an increment, so the bar would never reach N/N). Posting the
                # tick keeps `_load_total`/`_load_done` single-writer.
                try:
                    self.after(0, self._mark_one_done)
                except Exception:
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
        except Exception:
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
        except Exception:
            pass

    def _sample_lb_insert_loading(self, name):
        # Insert a placeholder sample row into the merged tree; it'll
        # be replaced with the proper '■ <name>' row once _on_loaded
        # fires (or removed by _on_load_error on failure).
        try:
            self.gate_tv.insert(
                '', 'end', iid=self._sample_iid(name),
                text=f'⏳ {name}', values=('',),
                tags=('loading',))
        except Exception:
            pass

    def _take_pending_prep(self, path):
        """Pop what a .wsp import, a session restore or a sample transfer
        staged for the file at `path` (see `stage_sample_prep`), or None.
        dict.pop is atomic, so a pool thread may take it."""
        pend = getattr(self, '_pending_sample_prep', None)
        if not pend or not path:
            return None
        return pend.pop(os.path.normcase(os.path.abspath(path)), None)

    def stage_sample_prep(self, path, comp=None, source_fcs=None, qc=None):
        """Stage, on the Tk thread and BEFORE the load is queued, how the file
        at `path` is to be prepared: `comp` is {'channels', 'matrix',
        'source'} -- the matrix this sample was compensated with where it came
        from -- `source_fcs` the raw FCS behind a processed CSV, and `qc`
        whether acquisition QC ran on it there (None: the Preferences
        setting)."""
        if not path or (comp is None and not source_fcs and qc is None):
            return
        pend = getattr(self, '_pending_sample_prep', None)
        if pend is None:
            pend = self._pending_sample_prep = {}
        pend[os.path.normcase(os.path.abspath(path))] = {
            'comp': comp, 'source_fcs': source_fcs, 'qc': qc}

    def _load_worker(self, name, path):
        try:
            from .cli import parse_labels
            from .pipeline import FlowSample
            s = FlowSample(path)
            # A sample arriving from a .wsp, a saved session or another window
            # brings the matrix it was compensated with there. Using the FCS's
            # own $SPILL instead silently swapped a custom (e.g. bead-derived)
            # matrix for the acquisition one, under gates drawn for the other.
            prep = self._take_pending_prep(path) or {}
            # Acquisition QC removes the events it flags, before any gate.
            # FlowJo has no such step and the .wsp has no QC population, so
            # with QC the same gates count differently there (the load status
            # and the .wsp export say so from s.qc_removed). Off unless turned
            # on in Preferences; a sample that comes back (session, transfer)
            # keeps the choice it was loaded with.
            qc = prep.get('qc')
            if qc is None:
                from .prefs import load_qc_enabled
                qc = load_qc_enabled()
            if qc:
                s.run_qc()
            comp = prep.get('comp')
            if comp:
                s._apply_comp(comp['matrix'], comp['channels'],
                              source=comp.get('source') or 'manual')
            else:
                s.auto_compensate()
            s.apply_transform()
            if self.labels_str:
                lbl = parse_labels(self.labels_str)
                if lbl:
                    s.set_labels(lbl)
            self.after(0, lambda: self._on_loaded(name, s))
        except Exception as exc:
            # Bind exc as a default arg — `except … as exc` deletes the name
            # at block exit, so the bare lambda would NameError when fired.
            self.after(0, lambda e=exc: self._on_load_error(name, e))

    def _load_csv_worker(self, name, path, labels):
        """Pool worker for a processed-data sidecar CSV (workspace results
        carrying cluster / UMAP / … columns). Reads + builds the FlowSample
        off the Tk thread — like ``_load_worker`` but skipping the FCS QC /
        compensation / transform, which are already baked into the saved data —
        then posts ``_on_loaded`` back to the Tk thread. ``labels`` is a snapshot
        taken at enqueue time so this never touches shared editor state."""
        try:
            import pandas as pd

            from .pipeline import FlowSample
            df = pd.read_csv(path)
            s = FlowSample.from_dataframe(df, name=name, labels=labels,
                                          path=path)
            # A session's processed-data sidecar: OpenFlo's own data. Its
            # note waits for the end of the batch (_on_load_settled), so a
            # restore shows one dialog, not one per sample.
            note = mark_processed_transforms(s, path, provenance='session')
            # The CSV holds values compensated with the session's matrix; put
            # the matrix (and the raw FCS) back on the sample so the .wsp
            # export, Compensation QC and the matrix editor still see them.
            prep = self._take_pending_prep(path) or {}
            comp = prep.get('comp')
            if comp:
                import numpy as np
                s.comp_matrix = np.asarray(comp['matrix'], dtype=float)
                s.comp_channels = [str(c) for c in comp['channels']]
                s.compensation_source = comp.get('source') or 'manual'
            if prep.get('source_fcs'):
                s.source_fcs = prep['source_fcs']
            if prep.get('qc') is not None:
                # The CSV holds the events left after the QC it was saved
                # with; a rebuild from its FCS repeats that.
                s.qc_applied = bool(prep['qc'])

            def _land():
                self._on_loaded(name, s)
                self._note_transform_assumption(note)
            self.after(0, _land)
        except Exception as exc:
            self.after(0, lambda e=exc: self._on_load_error(name, e))

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
        try:
            self.gate_tv.delete(self._sample_iid(name))
        except Exception:
            pass
        # A failed load must resolve its slot in a staged cross-instance move
        # as NOT taken, so the source keeps its copy instead of deleting a
        # sample that never arrived here.
        note = getattr(self, '_note_inbound_landed', None)
        if note is not None:
            p = (getattr(self, '_name_to_path', {}) or {}).get(name)
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
        n = self._remove_samples(samples)
        self.status_var.set(f"Removed {n} sample(s).")

    def _remove_trials(self, trials):
        """Remove every sample (+ gates) belonging to ``trials``. Confirmed,
        because — like single-sample Remove — it isn't on the undo stack."""
        members = []
        for t in trials:
            members.extend(n for n in self._trial_members(t) if n not in members)
        if not members:
            # Empty trial header(s) — just forget them.
            for t in trials:
                if t in self._trial_order:
                    self._trial_order.remove(t)
            self._refresh_gate_list()
            return
        label = (f"trial '{trials[0]}'" if len(trials) == 1
                 else f"{len(trials)} trials")
        if not messagebox.askyesno(
                "Remove trial",
                f"Remove {label} — {len(members)} sample(s) and all their "
                f"gates?\nThis can't be undone.",
                parent=self):
            return
        self._remove_samples(members)
        self.status_var.set(f"Removed {label} ({len(members)} sample(s)).")

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
            self._sample_colors.pop(name, None)
            if name in self._sample_order:
                self._sample_order.remove(name)
            self._sample_gates.pop(name, None)
            self._sample_gate_seq.pop(name, None)
            self._sample_gate_order.pop(name, None)
            self._sample_plot_enabled.pop(name, None)
            self._cluster_labels.pop(name, None)
            self._sample_trial.pop(name, None)
            self._sample_is_comp.pop(name, None)
            ap = self._name_to_path.pop(name, None)
            if ap is not None:
                self._path_to_name.pop(ap, None)
            for ckey in [k for k in self._ac_cache if k[0] == name]:
                self._ac_cache.pop(ckey, None)
            for ckey in [k for k in self._ac_count_cache if k[0] == name]:
                self._ac_count_cache.pop(ckey, None)
            for ckey in [k for k in self._ac_method_cache if k[0] == name]:
                self._ac_method_cache.pop(ckey, None)
            # Its display caches and any queued background work go with it.
            self._pc_forget(name)
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
