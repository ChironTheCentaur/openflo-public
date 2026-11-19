"""FCS / CSV / WSP loading, drag-drop, watch-folder.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from . import ui_errors
from .editor_base import EditorMixin
from .errors import note as _note_error
from .paths import (
    CONTAINER_FILE_PATTERNS,
    FCS_FILE_PATTERNS,
    dialog_start_dir,
    is_fcs_container,
    is_fcs_file,
    is_single_fcs_file,
    sample_key,
)
from .ui_errors import show_error
from .vendor_formats import (
    VENDOR_FILE_PATTERNS,
    vendor_format,
    vendor_status,
    vendor_summary,
)
from .wsp_notes import (
    sample_load_notes,
    uncompensated_gate_note,
    uncompensated_summary,
)
from .wsp_notes import wsp_load_notes as _wsp_load_notes
from .wsp_notes import wsp_summary as _wsp_summary

log = logging.getLogger(__name__)

# Package directory (same value as gui.BASE): never a dialog's start folder.
BASE = os.path.dirname(os.path.abspath(__file__))

# The Tools > Files & checks entry that starts and stops the folder watch,
# keyed by "watching now".
WATCH_FOLDER_LABELS = {False: "Watch folder…", True: "Stop watching folder"}


def _samples_count(n):
    """'1 sample' / '3 samples'."""
    return '1 sample' if n == 1 else f'{n} samples'


class LoadMixin(EditorMixin):
    """Adding samples, drag-and-drop import, async load queue, watch-folder."""

    def _open_dir(self):
        """The folder the last file was picked from in this window, else
        the data folder OpenFlo was started with (None when neither)."""
        return getattr(self, '_last_open_dir', None) or self.fcs_dir

    def _add_samples(self):
        """Add-FCS button: dialog picker, then queue.

        Accepts ``.fcs`` (and Beckman Coulter ``.lmd``), a BD Accuri file
        (``.c6``, ``.ci``, ``.cfl``: one sample per well), ``.wsp`` and a BD
        FACSDiva experiment exported as ``.xml``. A workspace (or experiment) is "exploded": every
        sample it references is queued for load, and the gate trees
        attached to each are merged into that sample's per-sample gate
        storage as it finishes loading.
        """
        from .paths import is_workspace_file
        init = dialog_start_dir(self._open_dir(), BASE)
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Select FCS file(s) or a workspace",
            filetypes=[
                # Instrument files are shown too: picking one says how to
                # export it as FCS (vendor_formats).
                ('FCS & workspaces',
                 f'{FCS_FILE_PATTERNS} {CONTAINER_FILE_PATTERNS} *.wsp *.xml '
                 f'{VENDOR_FILE_PATTERNS}'),
                ('FCS files',              FCS_FILE_PATTERNS),
                ('BD Accuri files',        CONTAINER_FILE_PATTERNS),
                ('FlowJo workspace',       '*.wsp'),
                ('BD FACSDiva experiment (XML)', '*.xml'),
                ('Instrument files (export as FCS first)',
                 VENDOR_FILE_PATTERNS),
                ('All files',              '*.*')])
        if not paths:
            return
        self._last_open_dir = os.path.dirname(paths[0])
        fcs_paths, wsp_paths, other_xml = [], [], []
        for p in paths:
            if is_workspace_file(p):
                wsp_paths.append(p)
            elif p.lower().endswith('.xml'):
                other_xml.append(os.path.basename(p))
            else:
                fcs_paths.append(p)
        if other_xml:
            self.status_var.set(
                f"Not opened (not a FACSDiva experiment): {', '.join(other_xml)}."
                " A Gating-ML file applies from Tools > Templates > From file….")
        # Process workspaces first so their pending-gates map is populated
        # before any FCS load completes.
        for wsp in wsp_paths:
            self._ingest_wsp(wsp)
        if fcs_paths:
            self._queue_fcs_loads(fcs_paths)

    def _load_processed_data(self):
        """Load a pipeline-processed CSV (already compensated + transformed,
        carrying cluster / UMAP / flowsom columns) as a sample, via
        FlowSample.from_dataframe. No QC / compensation / transform is
        re-applied. A sibling ``<name>_labels.json`` ({detector: label}) is
        picked up if present. Also reads a workspace run's binary events
        table (``<label>_events.npz``)."""
        init = dialog_start_dir(self._open_dir(), BASE)
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Load processed data",
            filetypes=[('Processed data', '*.csv *.npz'),
                       ('Processed CSV', '*.csv'),
                       ('Workspace run events', '*.npz'),
                       ('All files', '*.*')])
        if not paths:
            return
        self._last_open_dir = os.path.dirname(paths[0])
        import json

        from .events_csv import read_events_csv
        from .flow_sample import FlowSample
        from .sample_prep import mark_processed_transforms
        from .session_sidecar import SIDECAR_EXT, read_frame
        added = 0
        notes = ''
        for p in paths:
            binary = p.lower().endswith(SIDECAR_EXT)
            name = os.path.basename(p).rsplit('.', 1)[0]
            for suf in ('_processed',) + (('_events',) if binary else ()):
                if name.endswith(suf):
                    name = name[:-len(suf)]
            if name in self._samples:
                self.status_var.set(f"{name} already loaded — skipped.")
                continue
            try:
                # Only OpenFlo writes the binary table: a workspace run's
                # events, from the editor's own (display-transformed) data.
                df, how = (read_frame(p), []) if binary else read_events_csv(p)
            except Exception as exc:  # noqa: BLE001 - a bad file is reported in the status line and the batch goes on
                self.status_var.set(f"Load CSV failed: {exc}")
                continue
            for h in how:
                self._add_load_note(f"{os.path.basename(p)}: {h}")
            labels = None
            sidecar = os.path.join(os.path.dirname(p), f'{name}_labels.json')
            if os.path.isfile(sidecar):
                try:
                    with open(sidecar, encoding='utf-8') as fh:
                        labels = json.load(fh)
                except (OSError, ValueError):  # noted; an unreadable or malformed labels file just loads the sample without labels
                    _note_error("could not read the labels file of a loaded sample")
                    labels = None
            s = FlowSample.from_dataframe(df, name=name, labels=labels, path=p)
            note = mark_processed_transforms(
                s, p, provenance='events' if binary else 'foreign')
            notes += f"  [{note}]" if note else ''
            self._note_transform_assumption(note)
            self._on_loaded(name, s)
            added += 1
        if added:
            self.status_var.set(
                f"Loaded {added} processed sample(s). Use Populations… to "
                "import cluster/FlowSOM columns; plot UMAP1/UMAP2 to view."
                + notes)
        self._flush_transform_notes()      # one dialog for the whole batch
        self._flush_load_notes()

    def _ingest_wsp(self, wsp_path):
        """Parse a FlowJo workspace, discover its referenced FCS files,
        and stage each sample's gate tree for application once the FCS
        finishes loading.

        We DON'T apply gates immediately — they go into
        ``self._pending_sample_gates`` keyed by sample name. The
        ``_on_loaded`` hook drains that map per-sample.

        Sample-name resolution:
          - ``<SampleNode name="...">`` provides the display name
            (matches what FlowJo shows).
          - ``<DataSet uri="...">`` gives the FCS path. We try the path
            as-is, then the user-set ``self.fcs_dir``, then the WSP's own
            directory, then the subfolders of those two (compare.FcsLocator).

        A sample in one of the workspace's own groups goes into an OpenFlo
        group of that name (the first, when it is in several).
        """
        from .compare import FcsLocator
        from .wsp_open import WspUnreadable, plan_wsp

        wsp = os.path.basename(wsp_path)
        locator = FcsLocator(wsp_path, self.fcs_dir)
        try:
            plan = plan_wsp(wsp_path, locator)
        except WspUnreadable as exc:
            self.status_var.set(f"[WSP] {wsp}: {exc}")
            return
        names = [self._stage_wsp_sample(rec) for rec in plan.samples]
        if not plan.samples:
            self.status_var.set(
                f"[WSP] {wsp}: no samples resolved "
                f"({len(plan.unresolved)} unresolved)")
            return
        self.status_var.set(_wsp_summary(wsp, plan))
        # The status line is rewritten by the first sample that loads; the
        # same facts also go in the notes shown when the loading is done.
        in_several = [(rec.node_name or nm, rec.groups)
                      for nm, rec in zip(names, plan.samples, strict=True)
                      if len(rec.groups) > 1]
        for note in _wsp_load_notes(wsp, plan, in_several):
            self._add_load_note(note)
        # One sample per workspace sample, under the name its gates were
        # staged for: the data set its <Keywords> matched in a file holding
        # several, else the file as one sample (_fcs_load_jobs).
        self._queue_fcs_loads([
            rec.fcs_path if rec.set_number is None
            else (rec.fcs_path, rec.set_number) for rec in plan.samples])

    def _stage_wsp_sample(self, rec):
        """Stage workspace sample `rec` (wsp_open.WspSample) under the name
        the FCS queue will give its file, or its data set: its gates, axes
        and group go into the pending maps `_on_loaded` drains, its own
        matrix into the prep of its load. Returns that name."""
        # Same name the FCS queue will assign (collision-safe per day), so
        # `_on_loaded` drains these gates onto the right sample. A data set
        # of a file holding several is a sample of its own (rec.set_number).
        name = (self._sample_name_for(rec.fcs_path) if rec.set_number is None
                else self._sample_name_for(rec.fcs_path, rec.set_number,
                                           base=rec.set_name))
        self._pending_sample_gates[name] = rec.gates
        # The axes the workspace declares for this sample: FlowJo's
        # GeoMean is a mean on them (gating.population_stats).
        if not hasattr(self, '_pending_wsp_axes'):
            self._pending_wsp_axes = {}
        self._pending_wsp_axes[name] = rec.stat_axes
        if rec.groups:
            self._pending_wsp_trial[name] = rec.groups[0]
        if rec.comp is not None and name not in self._samples:
            more = ({} if rec.set_number is None
                    else {'set_number': rec.set_number})
            self.stage_sample_prep(rec.fcs_path, comp=rec.comp, **more)
        return name

    def _import_dropped_paths(self, paths):
        """Import a drop of files and/or folders. Folders are expanded to
        the ``.fcs`` / ``.wsp`` files within (see ``_expand_dropped_paths``).

        Workspaces are ingested first so their gate trees are staged before
        any FCS load completes; ``_ingest_wsp`` also queues the FCS each
        workspace references. Remaining loose FCS are then queued — the
        ``_loading`` guard means a sample referenced by both a dropped .wsp
        and a dropped .fcs is loaded only once (the workspace wins, so its
        gates ride along).

        A session or project file (``.flowsession`` / ``.flowproject``)
        opens in place of the current session, as File > Open session does
        (offering to save first). It is a whole session, so the first one
        wins and nothing else in the drop is loaded into it; the status line
        says what was left out."""
        vendor: list[str] = []
        sessions: list[str] = []
        fcs_paths, wsp_paths = self._expand_dropped_paths(
            paths, vendor, sessions)
        if sessions:
            skipped = len(sessions) - 1 + len(fcs_paths) + len(wsp_paths) \
                + len(vendor)
            # Opened from the event loop once the drop has returned: on
            # Windows tkdnd answers the OS's OLE Drop only when this
            # handler returns, so a save question (or a long open) inside
            # it would hang the drag source and the window.
            first = sessions[0]
            self.after_idle(
                lambda: self._open_dropped_session(first, skipped))
            return
        if not fcs_paths and not wsp_paths and not vendor:
            self.status_var.set(
                "Drop contained no .fcs files, workspaces or sessions.")
            return
        for wsp in wsp_paths:
            self._ingest_wsp(wsp)
        if fcs_paths or vendor:
            # The instrument files ride along so the queue says, in one
            # summary, how to export each kind as FCS.
            self._queue_fcs_loads(fcs_paths + vendor)

    def _open_dropped_session(self, path, skipped):
        """Open the session or project `path` dropped with `skipped` other
        files (left out: a session opens on its own). Once it has opened,
        which for a project archive is after its unpack, the status line
        adds what was left out after the open's own message; a failed open
        keeps its error and a cancelled one leaves the line as it was."""
        before = self.status_var.get()
        if skipped:
            # Cleared so the open's own message (an upgrade note, the
            # archive's folder) can be told from what was there before.
            self.status_var.set('')

        def then(opened):
            said = self.status_var.get().strip()
            if not opened:
                if not said:
                    self.status_var.set(before)
            elif skipped:
                said = said or f"Opened {os.path.basename(path)}"
                if not said.endswith(('.', '!', '?', '…')):
                    said += '.'
                self.status_var.set(
                    f"{said} A session opens on its own, so {skipped} "
                    "other dropped file(s) were not loaded.")
        self._load_session_path(path, then=then)

    def _queue_fcs_loads(self, paths, front_names=()):
        """Queue a list of FCS paths for background loading. Shared by
        the Add-FCS button, the OS-clipboard paste, and the file-drop
        target. Skips non-existent / non-.fcs / already-loaded entries
        with a brief status note. Loads run through a bounded worker pool so a
        big folder drop can't exhaust memory; a progress bar tracks completion.

        A file holding several data sets ($NEXTDATA) is queued as one sample
        per data set, unless something was staged for it as one sample
        (``_fcs_load_jobs``). An entry may instead be ``(path, set
        number)``: that one data set, as a session or a transfer saved it
        (None: the file as one sample).

        ``front_names`` (e.g. a restored session's active sample) load at
        priority 0 so their plot appears first; when the editor is empty and no
        front is given, the FIRST queued file takes that slot — it's the one
        that auto-renders, so the user sees a plot fastest."""
        self._ensure_load_pool()
        front = set(front_names)
        auto_front = (not self._samples and not front)   # first file when empty
        added = 0
        skipped = []
        split = []
        vendor = []
        for item in paths:
            p, number = item if isinstance(item, tuple) else (item, False)
            p = (p or '').strip().strip('"').strip("'")
            if not p:
                continue
            if not os.path.isfile(p):
                skipped.append(f'{os.path.basename(p)}(missing)')
                continue
            if vendor_format(p) is not None:
                vendor.append(p)
                continue
            if not is_fcs_file(p):
                skipped.append(f'{os.path.basename(p)}(not .fcs or .lmd)')
                continue
            jobs = self._fcs_load_jobs(p, number)
            if number is False and is_fcs_container(p) and jobs[0][1]:
                split.append(f'{os.path.basename(p)} holds '
                             f'{_samples_count(len(jobs))}, one per well')
            elif len(jobs) > 1:
                split.append(f'{os.path.basename(p)} holds {len(jobs)} data '
                             'sets, one sample each')
            for name, n in jobs:
                if name in self._samples:
                    skipped.append(f'{name}(already loaded)')
                    continue
                if name in self._loading:
                    # Already queued (e.g. a .wsp ingest queued it just now,
                    # or the same file appears twice in a folder drop). Don't
                    # queue a second job for the same sample.
                    continue
                self._loading.add(name)
                self._sample_lb_insert_loading(name)
                # Hand off to the bounded pool instead of spawning a thread
                # per file. _load_total only ever grows here (never reset
                # mid-run), so files dropped while a run is in flight extend
                # the bar (e.g. 2/5 → 2/8) rather than restarting it. The
                # first-when-empty / named front sample loads first so its
                # plot renders soonest.
                prio = 0 if (name in front or (auto_front and added == 0)) else 1
                self._enqueue_load((name, p) if n is None
                                   else (name, p, 'fcs', n), priority=prio)
                self._load_total += 1
                added += 1
        if added:
            self._update_progress_bar()
        if added or skipped:
            note = f"Queued {added}."
            if split:
                note += f"  ({'; '.join(split[:2])}.)"
            if skipped:
                note += f"  Skipped: {', '.join(skipped[:4])}"
                if len(skipped) > 4:
                    note += f" (+{len(skipped) - 4} more)"
            self.status_var.set(note)
        if vendor:
            self._report_vendor_files(vendor, alone=not (added or skipped))

    def _report_vendor_files(self, paths, alone=True):
        """Say, in one dialog, that the instrument files `paths` were not
        opened and how to export each kind as FCS (vendor_formats). When
        nothing else was queued or skipped (`alone`), the status line says
        it alone; otherwise it follows the queue's own note."""
        said = vendor_status(paths)
        self.status_var.set(
            said if alone else f"{self.status_var.get()}  {said}")
        ui_errors.showinfo("Export as FCS first", vendor_summary(paths),
                            parent=self)

    def _fcs_load_jobs(self, path, number=False):
        """``[(sample name, set number)]`` to queue for the FCS at `path`.
        `number` False (a path alone): one per data set when it holds
        several (``fcs_data_sets.sample_data_sets``, named by
        ``data_set_names`` and made unique), with a load note when its
        $NEXTDATA chain is broken; else the file as one sample (None). A
        path something was staged for as one sample -- the name a workspace
        import or a session gave it, a session's or a transfer's record --
        also loads as one sample: what was staged is for the file, and an
        older session's sample of such a file was its first data set.
        Otherwise `number` is the data set to load (None: the file as one
        sample)."""
        key = sample_key(path)
        if number is False and (key in self._path_to_name
                                or key in self._pending_sample_meta):
            number = None
        if number is not False:
            return [(self._sample_name_for(path, number), number)]
        from .fcs_data_sets import data_set_names, sample_data_sets
        sets, chain_note = sample_data_sets(path)
        if not sets:
            return [(self._sample_name_for(path), None)]
        if chain_note:
            self._add_load_note(chain_note)
        names = data_set_names(sets, path)
        return [(self._sample_name_for(path, ds.number, base=nm), ds.number)
                for ds, nm in zip(sets, names, strict=True)]

    def _add_load_note(self, text):
        """Queue `text` for the notes shown once the current loading is
        done (_flush_load_notes), and record it in the history."""
        self.__dict__.setdefault('_load_notes', []).append(str(text))
        self._audit('load.note', note=str(text))

    def _flush_load_notes(self):
        """Show the queued load notes in one dialog. Warnings from reading
        files and workspaces (a matrix that could not be used, a gate type
        not imported, a channel renamed) must not reach only the log or a
        status line the next event overwrites."""
        notes = list(dict.fromkeys(getattr(self, '_load_notes', None) or []))
        self._load_notes = []
        if not notes:
            return
        for n in notes:
            _note_error(f"load: {n}", None)
        cap = 15
        shown = [f"• {n}" for n in notes[:cap]]
        if len(notes) > cap:
            shown.append(f"… and {len(notes) - cap} more (see Analyze → "
                         "History).")
        ui_errors.showwarning(
            "Notes from loading",
            "\n".join(shown) + "\n\nThese are also kept in the history.",
            parent=self)

    def _on_loaded(self, name, sample):
        self._loading.discard(name)
        for note in sample_load_notes(name, sample):
            self._add_load_note(note)
        # A sample arriving under a name (first load, reload, session restore)
        # starts with no cached display work from whatever held it before.
        self._plotcache_forget(name)
        # The propagate downsample must NOT happen here, per arrival, comparing
        # this sample against the floor of whatever had loaded so far: the
        # outcome would depend on the order the pool happens to finish
        # samples in — measured, only a descending arrival order trims
        # anything at all. It is applied once to the whole set, after the
        # load burst settles; see `_apply_propagate_downsample`.
        self._samples[name] = sample
        if name not in self._sample_order:
            self._sample_order.append(name)
        pkey, meta = self._take_session_meta(sample)
        drawn_on = self._loaded_onto_transforms(sample, meta)
        self._restore_session_meta(name, sample, meta)
        # Its restored gates were drawn on the transforms it was saved with;
        # where adopting put the data on other parameters (a cofactor-5
        # asinh channel onto the editor's cofactor 150), they move with it.
        from .wsp_gate_coords import gate_rescale_fns
        rescale = gate_rescale_fns(
            drawn_on, getattr(sample, 'data_transforms', None))
        self._register_loaded_path(name, sample)
        # Color is assigned lazily — only when a sample is actually displayed
        # (see `_color_for`). Until then the tree row stays neutral so loading
        # many trials doesn't paint a rainbow of undisplayed samples.

        self._assign_loaded_trial(name, sample, meta)

        # Initialise per-sample gate state.
        self._sample_gates.setdefault(name, {})
        self._sample_gate_seq.setdefault(name, 0)
        self._sample_gate_order.setdefault(name, [])

        # Drain pending gates: a restored session bundles them in `meta` (keyed
        # by path, above); a .wsp ingest stages them in `_pending_sample_gates`
        # by name. We rebind `_gates` / `_gate_id_order` to this sample's
        # storage via `_set_active_sample`, populate, then leave it active iff
        # it was the first sample loaded.
        pending = self._loaded_pending_gates(name, sample, meta, rescale)
        id_map = (self._install_pending_gates(name, sample, pending)
                  if pending else {})
        # Every other tray tab keeps its own gates (editor_tabs).
        arrived = getattr(self, '_tabs_sample_arrived', None)
        every_tab = None
        if arrived is not None:
            every_tab = arrived(name, rescale if meta is not None else None,
                                id_map if meta is not None else None)
        if meta is not None:
            # A restore gives the gates fresh ids, in every tab: the
            # session's per-result fix declines follow them
            # (fixes.remap_declines), a gate only another tab holds included.
            from .fixes import remap_declines
            remap_declines(self, name,
                           id_map if every_tab is None else every_tab)
        # Plot inclusion: enable ONLY the very first sample loaded — the
        # user gets an immediate render to confirm the load worked.
        # Subsequent loads start unchecked so opening a session with many
        # samples doesn't cascade-render N overlays on every Add.
        was_first = (len(self._samples) == 1)
        self._sample_plot_enabled.setdefault(name, was_first)

        # First sample populates the channel choices and becomes active.
        if len(self._samples) == 1:
            self._seed_first_sample_channels(sample, meta)
        # NOTE: later samples may carry columns the first didn't — those get
        # unioned into the combos in `_on_load_settled`. Doing it (plus the
        # tree rebuild, panel-mismatch scan and replot) PER sample is O(N) work
        # that, on a bulk resume of N samples, thrashes the Tk thread into a
        # multi-second freeze. It's deferred + coalesced below instead.
        if self._active_sample is None:
            self._set_active_sample(name)
        self._record_loaded(name, sample, meta, pkey)

    def _take_session_meta(self, sample):
        """(path key, restored-session record) of a sample `_on_loaded` is
        taking; the record is None for a sample added now, which marks the
        window unsaved."""
        # A restored session may pin this sample to a manually-regrouped day /
        # Comps-Samples side (and carries its gates and transforms). It's
        # keyed by FILE PATH, not name, so collision-disambiguated names can't
        # mismatch on reload.
        spath = getattr(sample, 'path', '') or ''
        pkey = (sample_key(spath, getattr(sample, 'set_number', None))
                if spath else None)
        meta = self._pending_sample_meta.pop(pkey, None) if pkey else None
        if meta is None:            # added now, not restored from a session
            self._note_unsaved()
        return pkey, meta

    def _loaded_onto_transforms(self, sample, meta):
        """Put an arriving sample on the editor's transforms; returns the
        transforms its data (and its restored gates) were on before."""
        # Before anything reads its values: a restored sample goes back on the
        # transforms it was saved with (a raw-FCS reload arrives on the
        # loader's logicle), so whichever sample lands first seeds the editor
        # with the saved transforms; then the first sample seeds the
        # per-channel transforms, a later one is put on those in effect.
        if meta and meta.get('transforms') is not None:
            self._restore_sample_transforms(sample, meta['transforms'])
        drawn_on = dict(getattr(sample, 'data_transforms', None) or {})
        self._adopt_sample_transforms(sample, first=(len(self._samples) == 1))
        return drawn_on

    # A restored session's per-sample records, in the order they are put back
    # (_restore_session_meta): (key in the record, method taking
    # (name, sample, value)).
    _SESSION_META_RESTORERS = (
        ('bead_norm', '_restore_meta_bead_norm'),
        ('batch_norm', '_restore_meta_batch_norm'),
        ('stat_axes', '_restore_meta_stat_axes'),
        ('derived', '_restore_meta_derived'),
        ('result_stamps', '_restore_meta_result_stamps'),
        ('result_originals', '_restore_meta_result_originals'),
        ('downsample_trim', '_restore_meta_downsample_trim'),
    )

    def _restore_session_meta(self, name, sample, meta):
        """Put back what a restored session saved for `sample` (each
        non-empty record, then its models)."""
        for key, attr in self._SESSION_META_RESTORERS:
            if meta and meta.get(key):
                getattr(self, attr)(name, sample, meta[key])
        if meta:
            self._restore_session_models(name, sample, meta)

    def _restore_meta_bead_norm(self, name, sample, rec):
        # Bead normalization: a sample reopened from its raw FCS gets
        # the saved factors again; one from its processed sidecar
        # already holds the normalized values (ui_mass_cytometry).
        from .ui_mass_cytometry import restore_on_load
        why = restore_on_load(sample, rec)
        if why:
            self._add_load_note(f"{name}: bead normalization not "
                                f"restored ({why})")

    def _restore_meta_batch_norm(self, name, sample, rec):
        # Batch normalization (CytoNorm): its values live only in the
        # processed sidecar. Reopened from the raw FCS instead (the
        # sidecar missing), the sample is on its raw values; say so.
        if is_fcs_file(getattr(sample, 'path', '') or ''):
            self._add_load_note(
                f"{name}: batch normalization not restored (its saved "
                "values were not found, so the raw FCS values are "
                "used; run Normalize batches again)")
        else:
            sample.batch_norm = dict(rec)

    def _restore_meta_stat_axes(self, name, sample, rec):
        # The FlowJo workspace axes its GeoMean is taken on (see
        # wsp_gate_coords.wsp_stat_axes), as saved.
        sample.stat_axes = {c: a for c, a in rec.items()
                            if c in sample.data.columns}

    def _restore_meta_derived(self, name, sample, rec):
        # Derived channels: formulas back on the sample, and any column
        # a raw-FCS reload lacks computed again (derived.restore).
        from .derived import restore as _restore_derived
        for col, why in _restore_derived(sample, rec):
            self._add_load_note(f"{name}: derived channel {col} not "
                                f"restored ({why})")

    def _restore_meta_result_stamps(self, name, sample, rec):
        # The stamps its computed columns were saved with (result_stamp),
        # kept on the sample so the next save writes them back unchanged.
        sample.result_stamps = dict(rec)

    def _restore_meta_result_originals(self, name, sample, rec):
        # Corrected columns' originals (fixes): kept, with the stamp they
        # had, in a column of their own the next save writes back.
        sample.result_originals = dict(rec)

    def _restore_meta_downsample_trim(self, name, sample, rec):
        # Its 'Display + data' trim as saved: replayed on a raw-FCS
        # reload, so it holds the events it was saved with whatever the
        # restore's own re-trim (over the samples present and ticked
        # NOW) would pick; that one only ever trims further.
        why = self._ds_restore_trim(name, sample, rec)
        if why:
            self._add_load_note(why)

    def _restore_session_models(self, name, sample, meta):
        # The cell-cycle / FlowSOM models its label columns came with
        # (session_models). A record that does not fit the reopened
        # sample is left out with a log line: the columns still load,
        # and the window asks for a re-run as it does without one.
        from . import session_models as _models
        for key, put in ((_models.CELL_CYCLE_KEY,
                          _models.restore_cell_cycle),
                         (_models.FLOWSOM_KEY, _models.restore_flowsom)):
            if meta.get(key) is None:
                continue
            why = put(sample, meta[key])
            if why:
                _note_error(f"session: {name}: {key} not restored ({why})", None)
        # The last spectral unmix (its spectra, detectors and QC report,
        # which the editor holds), checked against this sample's U:
        # columns; Panel check and Spectral QC read it again.
        if meta.get(_models.UNMIX_KEY) is not None:
            why = _models.restore_unmix(self, name, sample,
                                        meta[_models.UNMIX_KEY])
            if why:
                _note_error(f"session: {name}: {_models.UNMIX_KEY} not restored ({why})", None)

    def _register_loaded_path(self, name, sample):
        # Keep the path⇄name registry in sync for entry points that bypass
        # `_sample_name_for` (e.g. processed-CSV load), so later loads still
        # see this name as taken and disambiguate around it.
        ap = sample_key(getattr(sample, 'path', '') or '',
                        getattr(sample, 'set_number', None))
        if ap:
            self._path_to_name.setdefault(ap, name)
        self._name_to_path.setdefault(name, ap or name)

    def _assign_loaded_trial(self, name, sample, meta):
        # Record the sample's trial (its FCS folder, or a Day N above it) so the
        # tree can group it and the workspace can label its origin.
        from .workspace_run import derive_trial_name
        trial = derive_trial_name(getattr(sample, 'path', None))
        wsp_trial = self._pending_wsp_trial.pop(name, None)
        if wsp_trial:
            trial = wsp_trial          # its group in the imported .wsp
        if meta:
            if meta.get('trial'):
                trial = meta['trial']
            if 'is_comp' in meta:
                self._sample_is_comp[name] = bool(meta['is_comp'])
            if meta.get('set_aside'):
                self._sample_set_aside[name] = str(meta['set_aside'])
            if meta.get('note'):
                self._sample_notes[name] = str(meta['note'])
        self._sample_trial[name] = trial
        if trial not in self._trial_order:
            self._trial_order.append(trial)

    def _loaded_pending_gates(self, name, sample, meta, rescale):
        """The gates to put on an arriving sample: a restored session's
        (`meta` not None), else those a .wsp ingest staged for `name`."""
        if meta is not None:
            from .session_open import rescaled_gates
            return rescaled_gates(meta.get('gates'), rescale)
        return self._wsp_pending_gates(name, sample)

    def _wsp_pending_gates(self, name, sample):
        pending = self._pending_sample_gates.pop(name, None)
        axes = self._pending_wsp_axes.pop(name, None)
        if axes:
            sample.stat_axes = {c: dict(a) for c, a in axes.items()
                                if c in sample.data.columns}
        if pending:
            from .wsp_open import gates_on_sample
            pending, uncompensated = gates_on_sample(
                pending, self._stored_transforms(name, sample), sample)
            # Said when the load settles.
            self._uncompensated_imports = (
                getattr(self, '_uncompensated_imports', []) + uncompensated)
        return pending

    def _install_pending_gates(self, name, sample, pending):
        """Add `pending` gates to sample `name`'s storage (made active for
        it), then hand the active sample back. Returns the saved id -> new
        id map of the gates added."""
        saved_active = self._active_sample
        self._set_active_sample(name)
        prev_suspend = self._suspend_undo
        self._suspend_undo = True       # bulk load isn't an undo step
        try:
            id_map = self._add_pending_gates(pending)
        finally:
            self._suspend_undo = prev_suspend
        self._note_unconverted_gates(name, sample, pending)
        if saved_active is not None and saved_active != name:
            # Restore the previously-active sample; this sample's
            # gates are now persisted in `_sample_gates[name]`.
            self._set_active_sample(saved_active)
        return id_map

    def _add_pending_gates(self, pending):
        """Install gate list `pending` on the active sample
        (gate_store.install_gates), the editor filling in each gate's colour
        and parent as for a gate drawn here. Returns saved id -> new id."""
        from .gate_store import install_gates
        n0 = len(self._gate_id_order)
        old_to_new = install_gates(self._gates, self._gate_id_order, pending,
                                   self._next_gate_id,
                                   fill=self._fill_new_gate)
        self._note_broken_booleans(self._gate_id_order[n0:])
        return old_to_new

    def _note_broken_booleans(self, added_ids):
        # A boolean whose operand could not be resolved (it was deleted
        # before the save) selects nothing, and so does a boolean built on
        # it: said once the load settles (_on_load_settled), as Clear / Cut
        # say it.
        from .gate_eval import broken_booleans
        from .gate_text import listed_name
        restored = {gid: self._gates[gid] for gid in added_ids}
        lost_ids, built_ids = broken_booleans(restored)
        self._restore_lost_operands.extend(
            listed_name(restored[i]) for i in lost_ids)
        self._restore_built_on.extend(
            listed_name(restored[i]) for i in built_ids)

    def _note_unconverted_gates(self, name, sample, pending):
        # Named again when the load settles: that status replaces the
        # import's, and an empty placeholder must not pass unnoticed.
        self._unconverted_imports = (
            getattr(self, '_unconverted_imports', [])
            + [g for g in pending if g.get('kind') == 'flowjo_ellipse'])
        # A boolean built on one selects nothing as well (gate_to_mask).
        from .flowjo_placeholders import placeholder_booleans
        self._unconverted_booleans = (
            getattr(self, '_unconverted_booleans', [])
            + placeholder_booleans(pending))
        # Populations of a clustering (or other label column) this
        # sample came back without -- a session whose processed data was
        # not found reloads the bare FCS -- read NA; say why once.
        cols = set(sample.data.columns)
        gone = sorted({str(g.get('channel') or 'cluster') for g in pending
                       if g.get('kind') in ('cluster', 'category')
                       and (g.get('channel') or 'cluster') not in cols})
        if gone:
            self._add_load_note(
                f"{name}: the {', '.join(gone)} results saved with its "
                "populations were not found, so those populations read NA")

    def _seed_first_sample_channels(self, sample, meta):
        from .result_stamp import is_original_column
        self._channels       = [c for c in sample.data.columns
                                if not is_original_column(c)]
        labels = dict(sample.channel_labels)
        if meta is not None:
            # Restored from a session: the labels it saved (a template's,
            # an edit) win over the file's own $PnS; the sample only fills
            # in channels the session had no label for. Replacing them
            # would lose every saved label whenever a raw-FCS sample
            # landed first.
            labels.update(getattr(self, '_session_channel_labels', {}))
        self._channel_labels = labels
        # (_channel_transform was seeded from this sample above.)
        self._populate_channel_combos()

    def _record_loaded(self, name, sample, meta, pkey):
        # Provenance: record the load (QC + auto-compensation + transform ran
        # in the loader). Capture the data identity for reproducibility.
        # FlowSample records where its matrix came from ('$SPILL', 'wsp:…',
        # 'manual'); None means nothing was applied, so a sample with no
        # spillover is not recorded as compensated.
        comp = getattr(sample, 'compensation_source', None) or 'none'
        self._audit('sample.load', sample=name,
                    path=getattr(sample, 'path', '') or '',
                    n_events=int(len(sample.data)),
                    channels=int(sample.data.shape[1]),
                    trial=self._sample_trial.get(name, ''),
                    compensation=comp,
                    qc=bool(getattr(sample, 'qc_applied', False)),
                    decoded=dict(getattr(sample, 'fcs_decoding', None) or {}))
        # Cheap per-sample status. The expensive UI sync is coalesced into one
        # debounced pass so a burst of loads doesn't lock the window.
        self._last_loaded = name
        # If this sample arrived as part of a staged cross-instance MOVE, this
        # is the point at which we have genuinely taken it — the source may
        # only delete its copy once every accepted path reaches here.
        if pkey:
            note = getattr(self, '_note_inbound_landed', None)
            if note is not None:
                note(pkey, ok=True)
        qc = int(getattr(sample, 'qc_removed', 0) or 0)
        if qc:
            self._qc_removed_batch = (getattr(self, '_qc_removed_batch', [])
                                      + [qc])
        # Its gate counts and statistics as the session saved them, used
        # only if they were computed by this OpenFlo on exactly this data.
        if meta and meta.get('saved_stats'):
            self._saved_stats_restore(name, meta['saved_stats'])
        self.status_var.set(
            f"{len(self._samples)} sample(s) loaded "
            f"(latest: {name}, {len(sample.data):,} events"
            + (f" after QC removed {qc:,}" if qc else '') + ")…")
        self._schedule_load_settle()

    def _schedule_load_settle(self, ms=120):
        """Debounce the heavy post-load UI sync. Each `_on_loaded` (re)arms a
        single timer; a burst of N loads collapses to ~one settle pass instead
        of N full tree/combo rebuilds on the Tk thread."""
        prev = getattr(self, '_load_settle_after', None)
        if prev:
            try:
                self.after_cancel(prev)
            except tk.TclError:
                pass
        self._load_settle_after = self.after(ms, self._on_load_settled)

    @staticmethod
    def _uncompensated_import_note(pairs):
        """Status note for imported .wsp gates FlowJo evaluates on a
        channel's UNcompensated parameter (`pairs` of (gate, channel)), or
        '' (wsp_notes.uncompensated_summary), logged as well."""
        if not pairs:
            return ''
        _note_error("workspace: gate(s) on FlowJo's uncompensated parameter, evaluated "
                    "here on compensated values: "
                    + ', '.join(f"'{g}' ({ch})" for g, ch in pairs), None)
        return uncompensated_summary(pairs)

    def _on_load_settled(self):
        """Run once after a load burst quiesces: union channel choices, rebuild
        the tree (swapping ⏳ placeholders for real rows), highlight the
        active sample unless the user picked rows meanwhile
        (_user_picked_in), flag a mismatched fluor panel, and replot."""
        self._load_settle_after = None
        # Samples in name order, not the order the loader happened to finish
        # them in (ctrl_d1, ctrl_d2, treat_d1, ctrl_d3, ...).
        from .workspace_run import natural_sort_key
        self._sample_order.sort(key=natural_sort_key)
        # Trim to the smallest enabled sample now that the set is stable. Once,
        # over the whole set, so the result does not depend on arrival order.
        if (getattr(self, 'ds_propagate_var', None) is not None
                and self.ds_propagate_var.get()):
            self._apply_propagate_downsample()
        if len(self._samples) > 1:
            # Union columns the first sample lacked into the axis/colour combos.
            self._refresh_channel_choices()
        self._refresh_gate_list()
        # Highlight the active sample (the first loaded, the one plotted),
        # not the last one to arrive: that one is loaded unticked, and the
        # highlight is what gates and compensation act on. Still scroll to
        # the newest arrival so the user sees it landed. Rows the user
        # picked while the files landed stay picked: replacing them with
        # the sample row made "New tab from selection" (and Copy to tab,
        # Apply to) act on the active sample alone.
        last = getattr(self, '_last_loaded', None)
        active = self._active_sample
        see = (last, self.gate_tv.see)
        steps = ([see] if self._user_picked_in(active)
                 else [(active, self.gate_tv.selection_set), see])
        for nm, act in steps:
            if nm is not None:
                try:
                    act(self._sample_iid(nm))
                except Exception as exc:  # noqa: BLE001 - noted; a missing tree row must not stop the load from finishing
                    _note_error("running an action on a sample in the tree", exc)
        note = self._lost_operand_note(self._restore_lost_operands,
                                       self._restore_built_on)
        self._restore_lost_operands.clear()
        self._restore_built_on.clear()
        msg = (f"{len(self._samples)} sample(s) loaded{note}. "
               "Double-click the plot to add a gate.")
        # Cross-sample stats tie by antibody label, so a mismatched panel just
        # means some labels won't be shared — surface it, don't block.
        if self._fluor_panel_warning():
            msg += "  [!] samples differ in fluor panel — see Statistics."
        from .flowjo_placeholders import flowjo_placeholder_note
        msg += flowjo_placeholder_note(
            getattr(self, '_unconverted_imports', []),
            getattr(self, '_unconverted_booleans', []))
        msg += self._uncompensated_import_note(
            getattr(self, '_uncompensated_imports', []))
        qc = getattr(self, '_qc_removed_batch', [])
        if qc:
            # Gates count only what QC kept; FlowJo counts every event.
            msg += (f"  Acquisition QC removed {sum(qc):,} event(s) in "
                    f"{len(qc)} sample(s) before gating (FlowJo counts them).")
        self._qc_removed_batch = []
        for g, ch in getattr(self, '_uncompensated_imports', []):
            self._add_load_note(uncompensated_gate_note(g, ch))
        self._unconverted_imports = []
        self._unconverted_booleans = []
        self._uncompensated_imports = []
        self.status_var.set(msg)
        self._schedule_replot(0)
        # A session restore's CSVs queue their transform notes as they land;
        # show them once the whole batch is in.
        if not self._loading:
            self._flush_transform_notes()
            self._flush_load_notes()

    def _user_picked_in(self, active):
        """True when the tree's selection is a pick to leave alone: the load
        itself only ever selects the active sample's row, so anything else
        selected is the user's. Not kept: an empty selection, or one that
        no longer matches ``active`` (the active sample was switched without
        the tree, as a session restore does): a gate row of another sample,
        or a first row (the one _on_tree_select follows) that is another
        sample's or a still-loading sample's row. Headers, Presort rows and
        several samples' rows stay picked."""
        sel = self.gate_tv.selection()
        if not sel:
            return False
        rows = [self._parse_iid(iid) for iid in sel]
        if any(p is not None and p[0] == 'gate' and p[1] != active
               for p in rows):
            return False
        first = rows[0]
        return not (first is not None and first[0] in ('sample', 'gate')
                    and first[1] != active)

    def _on_dnd_drop(self, event):
        """OS file-drop onto the gate tree. Parses tkdnd's spaces-and-
        braces filelist format using Tk's own splitlist, then imports any
        .fcs / .wsp entries. Dropped folders are walked recursively, so a
        whole trial folder (or a parent of several trial folders) imports
        in one gesture, each sample grouped under its own trial. Everything
        is wrapped in defensive try/except so a malformed drop can't crash
        the GUI."""
        try:
            raw = getattr(event, 'data', '') or ''
            log.debug("drop event raw=%r", raw)
            try:
                paths = list(self.tk.splitlist(raw))
            except Exception:  # noqa: BLE001 - noted; falls back to splitting the drop on spaces
                _note_error("could not split the dropped file list; split on spaces")
                paths = raw.split()
            if paths:
                self._import_dropped_paths(paths)
        except Exception as exc:  # noqa: BLE001 - noted; a malformed drop must not crash the GUI
            _note_error("drag-and-drop: drop not handled", exc)
        try:
            return event.action
        except AttributeError:
            return None

    def _load_example_data(self):
        """Generate a small synthetic PBMC dataset and load it — lets a new
        user try OpenFlo with no FCS files of their own. Files are written once
        to ~/.openflo/example_data and reused on later calls."""
        try:
            from .synthetic import make_immunophenotyping_dataset
        except Exception as exc:  # noqa: BLE001 - a broken generator import is shown in a warning dialog
            ui_errors.showwarning(
                "Example data unavailable",
                f"Couldn't load the synthetic-data generator:\n{exc}",
                parent=self)
            return
        if self._samples and not messagebox.askyesno(
                "Load example data",
                "Add a synthetic example dataset (2 groups × 2 donors, "
                "PBMC-like) alongside your current samples?", parent=self):
            return
        out = os.path.join(os.path.expanduser('~'), '.openflo', 'example_data')
        self.status_var.set("Generating example dataset…")
        self.update_idletasks()
        try:
            paths = make_immunophenotyping_dataset(
                out, groups=('ctrl', 'treat'), donors=2, n=5000, seed=0)
        except Exception as exc:  # noqa: BLE001 - shown in the status line and an error dialog; the editor carries on
            self.status_var.set(f"Example data failed: {exc}")
            show_error(self, "Example data failed",
                       "The example data could not be made.", exc)
            return
        self._queue_fcs_loads(paths)
        self.status_var.set(
            f"Loading {len(paths)} example sample(s) — synthetic PBMC, "
            "ctrl vs treat (CD3/4/8/19/56/14).")

    def _stop_watch_folder(self):
        """Stop watching the folder File > Watch folder chose."""
        self._watch_dir = None
        wa = getattr(self, '_watch_after', None)
        if wa:
            try:
                self.after_cancel(wa)
            except tk.TclError:
                pass
            self._watch_after = None
        self.status_var.set("Stopped watching folder.")
        self._sync_watch_menu()

    def _toggle_watch_folder(self):
        """Watch a folder and auto-load new .fcs files as they appear (e.g. an
        instrument's export folder). Run again to stop."""
        if getattr(self, '_watch_dir', None):
            self._stop_watch_folder()
            return
        d = filedialog.askdirectory(title="Watch folder for new FCS files")
        if not d:
            return
        self._watch_dir = d
        # Seed with what's already there so only NEW files load.
        # Key the seen-set on the EXACT filename (only the extension test is
        # case-insensitive). Lower-casing the key collapses case-distinct files
        # ('Sample.fcs' vs 'sample.fcs') on a case-sensitive (Linux) FS, so the
        # second would never auto-load.
        self._watch_seen = {f for f in os.listdir(d) if is_single_fcs_file(f)}
        self.status_var.set(
            f"Watching {os.path.basename(d)} — new .fcs and .lmd files load "
            "automatically (Tools → Files & checks → Watch folder again to "
            "stop).")
        self._sync_watch_menu()
        self._poll_watch()

    def _sync_watch_menu(self):
        """Label the watch entry for what picking it does now: start a
        watch, or stop the running one."""
        menu = getattr(self, '_watch_menu', None)
        if menu is None:
            return
        want = WATCH_FOLDER_LABELS[bool(getattr(self, '_watch_dir', None))]
        try:
            for i in range((menu.index('end') or 0) + 1):
                if (menu.type(i) == 'command' and menu.entrycget(i, 'label')
                        in WATCH_FOLDER_LABELS.values()):
                    menu.entryconfigure(i, label=want)
        except tk.TclError:             # the menu is gone (closing)
            pass

    def _poll_watch(self):
        d = getattr(self, '_watch_dir', None)
        if not d:
            return
        if not os.path.isdir(d):
            # Moved, renamed or deleted (or a drive unplugged): say so, as
            # the status still read "Watching ...".
            self._watch_dir = None
            self._watch_after = None
            self.status_var.set(
                f"Stopped watching {os.path.basename(d)}: the folder is no "
                "longer there.")
            self._sync_watch_menu()
            return
        try:
            cur = {f for f in os.listdir(d) if is_single_fcs_file(f)}
            new = [os.path.join(d, f) for f in cur
                   if f not in self._watch_seen]
            if new:
                self._watch_seen.update(cur)
                self._queue_fcs_loads(new)
                self.status_var.set(
                    f"Watch: loading {len(new)} new file(s) from "
                    f"{os.path.basename(d)}…")
        except Exception as exc:  # noqa: BLE001 - noted; one bad poll must not stop the folder watch
            _note_error("folder watch: poll failed", exc)
        self._watch_after = self.after(4000, self._poll_watch)
