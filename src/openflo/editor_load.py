"""FCS / CSV / WSP loading, drag-drop, watch-folder.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox

from .editor_base import EditorMixin
from .errors import note as _note_error
from .paths import dialog_start_dir
from .ui_errors import show_error

# Package directory (same value as gui.BASE): never a dialog's start folder.
BASE = os.path.dirname(os.path.abspath(__file__))


class LoadMixin(EditorMixin):
    """Adding samples, drag-and-drop import, async load queue, watch-folder."""

    def _open_dir(self):
        """The folder the last file was picked from in this window, else
        the data folder OpenFlo was started with (None when neither)."""
        return getattr(self, '_last_open_dir', None) or self.fcs_dir

    def _add_samples(self):
        """Add-FCS button: dialog picker, then queue.

        Accepts both ``.fcs`` and ``.wsp``. A workspace is "exploded":
        every ``<Sample>`` it references is queued for load, and the
        gate trees attached to each ``<SampleNode>`` are merged into
        that sample's per-sample gate storage as it finishes loading.
        """
        init = dialog_start_dir(self._open_dir(), BASE)
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Select FCS file(s) or FlowJo workspace",
            filetypes=[
                ('FCS & FlowJo workspace', '*.fcs *.wsp'),
                ('FCS files',              '*.fcs'),
                ('FlowJo workspace',       '*.wsp'),
                ('All files',              '*.*')])
        if not paths:
            return
        self._last_open_dir = os.path.dirname(paths[0])
        fcs_paths, wsp_paths = [], []
        for p in paths:
            (wsp_paths if p.lower().endswith('.wsp') else fcs_paths).append(p)
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
        picked up if present."""
        init = dialog_start_dir(self._open_dir(), BASE)
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Load processed data (CSV)",
            filetypes=[('Processed CSV', '*.csv'), ('All files', '*.*')])
        if not paths:
            return
        self._last_open_dir = os.path.dirname(paths[0])
        import json

        from .editor_loadpool import mark_processed_transforms
        from .pipeline import FlowSample, read_events_csv
        added = 0
        notes = ''
        for p in paths:
            name = os.path.basename(p).rsplit('.', 1)[0]
            if name.endswith('_processed'):
                name = name[:-len('_processed')]
            if name in self._samples:
                self.status_var.set(f"{name} already loaded — skipped.")
                continue
            try:
                df, how = read_events_csv(p)
            except Exception as exc:
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
                except Exception:
                    labels = None
            s = FlowSample.from_dataframe(df, name=name, labels=labels, path=p)
            note = mark_processed_transforms(s, p, provenance='foreign')
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
        from xml.etree import ElementTree as ET

        from .compare import FcsLocator
        from .pipeline import (
            WspReader,
            _flowjo_placeholder_note,
            fcs_event_total,
            wsp_event_mismatch,
            wsp_sample_events,
            wsp_stat_axes,
        )

        try:
            reader = WspReader(wsp_path)
        except Exception as exc:
            self.status_var.set(f"[WSP] {os.path.basename(wsp_path)}: {exc}")
            return

        # Re-parse the file ourselves to walk per-sample. WspReader's
        # extract_gates() flattens; we need samples + their own gate
        # subtrees so this editor can attach the right tree to the right
        # FCS.
        try:
            ns_re = re.compile(r'\{.*?\}')
            tree = ET.parse(wsp_path)
            root = tree.getroot()
            for elem in root.iter():
                elem.tag = ns_re.sub('', elem.tag)
                if elem.attrib:
                    elem.attrib = {ns_re.sub('', k): v
                                   for k, v in elem.attrib.items()}
        except Exception as exc:
            self.status_var.set(f"[WSP] {os.path.basename(wsp_path)}: {exc}")
            return

        locator = FcsLocator(wsp_path, self.fcs_dir)
        groups = reader.sample_groups()
        in_several = []

        resolved, unresolved, imported, unread = [], [], [], []
        mismatched = []         # (sample, why): file's events != workspace's
        for sample_elem in root.iter('Sample'):
            ds = sample_elem.find('DataSet')
            uri = ds.get('uri') if ds is not None else None
            sn  = sample_elem.find('SampleNode')
            if sn is None:
                continue
            fcs_path = locator.find(uri)
            if fcs_path is None:
                unresolved.append(sn.get('name') or '(unnamed)')
                continue
            # A workspace can hold several files of one name; the one found
            # may not be the one FlowJo gated. Imported anyway, and said.
            why = wsp_event_mismatch(wsp_sample_events(sample_elem),
                                     fcs_event_total(fcs_path))
            if why:
                mismatched.append((sn.get('name') or '(unnamed)', why))

            # Per-sample gate extraction — scope the reader's walker to
            # just this <SampleNode>. Returns the same gate-dict format
            # as a full-document extract_gates() call.
            # FlowJo writes Time gates in seconds, scaled with the FCS's own
            # timestep; hand over the one from the FCS this sample resolved to.
            gates = reader.extract_gates(
                sample_node=sn,
                timestep=WspReader.sample_timestep(sample_elem, fcs_path))
            unread.extend(reader.unread_populations)
            # Same name the FCS queue will assign (collision-safe per day), so
            # `_on_loaded` drains these gates onto the right sample.
            sample_name = self._sample_name_for(fcs_path)
            self._pending_sample_gates[sample_name] = gates
            # The axes the workspace declares for this sample: FlowJo's
            # GeoMean is a mean on them (gating.population_stats).
            if not hasattr(self, '_pending_wsp_axes'):
                self._pending_wsp_axes = {}
            self._pending_wsp_axes[sample_name] = wsp_stat_axes(sample_elem)
            sid = sn.get('sampleID') or (ds.get('sampleID')
                                         if ds is not None else None)
            mine = groups.get(sid) or []
            if mine:
                self._pending_wsp_trial[sample_name] = mine[0]
                if len(mine) > 1:
                    in_several.append((sn.get('name') or sample_name, mine))
            # Compensate as the workspace does. These gates were drawn on
            # values compensated with the sample's OWN matrix in the .wsp;
            # the loader used to apply the FCS's $SPILL regardless, so a
            # workspace with a bead-derived (or edited) matrix had its gates
            # evaluated on differently compensated data. A sample with no
            # matrix of its own gets FlowJo's Acquisition-defined one -- the
            # file's $SPILL -- which is the loader's default anyway.
            own = reader.sample_matrix(sample_elem)
            if own is not None and sample_name not in self._samples:
                self.stage_sample_prep(fcs_path, comp={
                    'channels': list(own['channels']),
                    'matrix': own['matrix'],
                    'source': f"wsp:{own.get('name') or 'sample'}"})
            resolved.append((sample_name, fcs_path, len(gates)))
            imported.extend(gates)

        if not resolved:
            self.status_var.set(
                f"[WSP] {os.path.basename(wsp_path)}: no samples resolved "
                f"({len(unresolved)} unresolved)")
            return

        summary = (f"[WSP] {os.path.basename(wsp_path)}: queued "
                   f"{len(resolved)} sample(s), "
                   f"{sum(n for _, _, n in resolved)} gate(s)")
        if unresolved:
            summary += f" — couldn't locate FCS for: {', '.join(unresolved[:3])}"
            if len(unresolved) > 3:
                summary += f" (+{len(unresolved) - 3})"
        # An ellipse the import could not convert counts nothing; say so where
        # the user looks, not only in the log.
        summary += _flowjo_placeholder_note(imported)
        if mismatched:
            summary += (f" — {len(mismatched)} sample(s) matched to an FCS "
                        "file with a different number of events than the "
                        "workspace records (see the load notes)")
        if unread:
            # Left out with everything under them, not lifted to the
            # grandparent (see WspReader.extract_gates).
            summary += (f" — {len(unread)} population(s) with a gate type "
                        "OpenFlo cannot read were not imported, nor were the "
                        f"{sum(n for _, _, n in unread)} under them: "
                        + ', '.join(f"'{nm}'" for nm, _, _ in unread[:3])
                        + (f" (+{len(unread) - 3})" if len(unread) > 3 else ''))
        self.status_var.set(summary)
        # The status line is rewritten by the first sample that loads; the
        # same facts also go in the notes shown when the loading is done.
        wsp = os.path.basename(wsp_path)
        if unresolved:
            self._add_load_note(f"{wsp}: FCS file not found for "
                                + ', '.join(unresolved))
        for fname, n in locator.ambiguous:
            self._add_load_note(
                f"{wsp}: {n} files named {fname} in its folders; none was "
                "picked. Point the FCS folder at the right one.")
        for nm, why in mismatched:
            self._add_load_note(f"{wsp}: '{nm}': {why}")
        for nm, mine in in_several:
            self._add_load_note(
                f"{wsp}: '{nm}' is in the groups {', '.join(mine)}; it was "
                f"put in '{mine[0]}'")
        for nm, tags, n_under in unread:
            self._add_load_note(
                f"{wsp}: population '{nm}' ({', '.join(tags)}) not imported"
                + (f", nor the {n_under} under it" if n_under else ''))
        for g in imported:
            if g.get('kind') == 'flowjo_ellipse':
                self._add_load_note(
                    f"{wsp}: ellipse '{g.get('label') or '?'}' could not be "
                    "converted and reads NA")
        self._queue_fcs_loads([p for _, p, _ in resolved])

    def _import_dropped_paths(self, paths):
        """Import a drop of files and/or folders. Folders are expanded to
        the ``.fcs`` / ``.wsp`` files within (see ``_expand_dropped_paths``).

        Workspaces are ingested first so their gate trees are staged before
        any FCS load completes; ``_ingest_wsp`` also queues the FCS each
        workspace references. Remaining loose FCS are then queued — the
        ``_loading`` guard means a sample referenced by both a dropped .wsp
        and a dropped .fcs is loaded only once (the workspace wins, so its
        gates ride along)."""
        fcs_paths, wsp_paths = self._expand_dropped_paths(paths)
        if not fcs_paths and not wsp_paths:
            self.status_var.set(
                "Drop contained no .fcs or .wsp files.")
            return
        for wsp in wsp_paths:
            self._ingest_wsp(wsp)
        if fcs_paths:
            self._queue_fcs_loads(fcs_paths)

    def _queue_fcs_loads(self, paths, front_names=()):
        """Queue a list of FCS paths for background loading. Shared by
        the Add-FCS button, the OS-clipboard paste, and the file-drop
        target. Skips non-existent / non-.fcs / already-loaded entries
        with a brief status note. Loads run through a bounded worker pool so a
        big folder drop can't exhaust memory; a progress bar tracks completion.

        ``front_names`` (e.g. a restored session's active sample) load at
        priority 0 so their plot appears first; when the editor is empty and no
        front is given, the FIRST queued file takes that slot — it's the one
        that auto-renders, so the user sees a plot fastest."""
        self._ensure_load_pool()
        front = set(front_names)
        auto_front = (not self._samples and not front)   # first file when empty
        added = 0
        skipped = []
        for p in paths:
            p = (p or '').strip().strip('"').strip("'")
            if not p:
                continue
            if not os.path.isfile(p):
                skipped.append(f'{os.path.basename(p)}(missing)')
                continue
            if not p.lower().endswith('.fcs'):
                skipped.append(f'{os.path.basename(p)}(not .fcs)')
                continue
            name = self._sample_name_for(p)
            if name in self._samples:
                skipped.append(f'{name}(already loaded)')
                continue
            if name in self._loading:
                # Already queued (e.g. a .wsp ingest queued it just now, or
                # the same file appears twice in a folder drop). Don't queue
                # a second job for the same sample.
                continue
            self._loading.add(name)
            self._sample_lb_insert_loading(name)
            # Hand off to the bounded pool instead of spawning a thread per
            # file. _load_total only ever grows here (never reset mid-run), so
            # files dropped while a run is in flight extend the bar (e.g.
            # 2/5 → 2/8) rather than restarting it. The first-when-empty / named
            # front sample loads first so its plot renders soonest.
            prio = 0 if (name in front or (auto_front and added == 0)) else 1
            self._enqueue_load((name, p), priority=prio)
            self._load_total += 1
            added += 1
        if added:
            self._update_progress_bar()
        if added or skipped:
            note = f"Queued {added}."
            if skipped:
                note += f"  Skipped: {', '.join(skipped[:4])}"
                if len(skipped) > 4:
                    note += f" (+{len(skipped) - 4} more)"
            self.status_var.set(note)

    def _add_load_note(self, text):
        """Queue `text` for the notes shown once the current loading is
        done (_flush_load_notes), and record it in the history."""
        self.__dict__.setdefault('_load_notes', []).append(str(text))
        self._audit('load.note', note=str(text))

    def _flush_load_notes(self):
        """Show the queued load notes in one dialog. Warnings from reading
        files and workspaces (a matrix that could not be used, a gate type
        not imported, a channel renamed) used to reach only the log or a
        status line the next event overwrote."""
        notes = list(dict.fromkeys(getattr(self, '_load_notes', None) or []))
        self._load_notes = []
        if not notes:
            return
        for n in notes:
            print(f"[load] {n}", flush=True)
        cap = 15
        shown = [f"• {n}" for n in notes[:cap]]
        if len(notes) > cap:
            shown.append(f"… and {len(notes) - cap} more (see Analyze → "
                         "History).")
        messagebox.showwarning(
            "Notes from loading",
            "\n".join(shown) + "\n\nThese are also kept in the history.",
            parent=self)

    def _on_loaded(self, name, sample):
        self._loading.discard(name)
        for w in getattr(sample, 'load_warnings', None) or ():
            self._add_load_note(f"{name}: {w}")
        # A sample arriving under a name (first load, reload, session restore)
        # starts with no cached display work from whatever held it before.
        self._pc_forget(name)
        # The propagate downsample used to happen HERE, per arrival, comparing
        # this sample against the floor of whatever had loaded so far. That
        # made the outcome depend on the order the pool happened to finish
        # samples in — measured, only a descending arrival order trimmed
        # anything at all. It is now applied once to the whole set, after the
        # load burst settles; see `_apply_propagate_downsample`.
        self._samples[name] = sample
        if name not in self._sample_order:
            self._sample_order.append(name)
        # A restored session may pin this sample to a manually-regrouped day /
        # Comps-Samples side (and carries its gates and transforms). It's
        # keyed by FILE PATH, not name, so collision-disambiguated names can't
        # mismatch on reload.
        spath = getattr(sample, 'path', '') or ''
        pkey = os.path.normcase(os.path.abspath(spath)) if spath else None
        meta = self._pending_sample_meta.pop(pkey, None) if pkey else None
        if meta is None:            # added now, not restored from a session
            self._note_unsaved()
        # Before anything reads its values: a restored sample goes back on the
        # transforms it was saved with (a raw-FCS reload arrives on the
        # loader's logicle), so whichever sample lands first seeds the editor
        # with the saved transforms; then the first sample seeds the
        # per-channel transforms, a later one is put on those in effect.
        if meta and meta.get('transforms') is not None:
            self._restore_sample_transforms(sample, meta['transforms'])
        drawn_on = dict(getattr(sample, 'data_transforms', None) or {})
        self._adopt_sample_transforms(sample, first=(len(self._samples) == 1))
        if meta and meta.get('bead_norm'):
            # Bead normalization: a sample reopened from its raw FCS gets
            # the saved factors again; one from its processed sidecar
            # already holds the normalized values (ui_mass_cytometry).
            from .ui_mass_cytometry import restore_on_load
            why = restore_on_load(sample, meta['bead_norm'])
            if why:
                self._add_load_note(f"{name}: bead normalization not "
                                    f"restored ({why})")
        if meta and meta.get('batch_norm'):
            # Batch normalization (CytoNorm): its values live only in the
            # processed sidecar. Reopened from the raw FCS instead (the
            # sidecar missing), the sample is on its raw values; say so.
            if str(getattr(sample, 'path', '') or '').lower().endswith(
                    '.fcs'):
                self._add_load_note(
                    f"{name}: batch normalization not restored (its saved "
                    "values were not found, so the raw FCS values are "
                    "used; run Normalize batches again)")
            else:
                sample.batch_norm = dict(meta['batch_norm'])
        if meta and meta.get('stat_axes'):
            # The FlowJo workspace axes its GeoMean is taken on (see
            # pipeline.wsp_stat_axes), as saved.
            sample.stat_axes = {c: a for c, a in meta['stat_axes'].items()
                                if c in sample.data.columns}
        if meta and meta.get('derived'):
            # Derived channels: formulas back on the sample, and any column
            # a raw-FCS reload lacks computed again (derived.restore).
            from .derived import restore as _restore_derived
            for col, why in _restore_derived(sample, meta['derived']):
                self._add_load_note(f"{name}: derived channel {col} not "
                                    f"restored ({why})")
        if meta and meta.get('result_stamps'):
            # The stamps its computed columns were saved with (result_stamp),
            # kept on the sample so the next save writes them back unchanged.
            sample.result_stamps = dict(meta['result_stamps'])
        if meta and meta.get('result_originals'):
            # Corrected columns' originals (fixes): kept, with the stamp they
            # had, in a column of their own the next save writes back.
            sample.result_originals = dict(meta['result_originals'])
        if meta and meta.get('downsample_trim'):
            # Its 'Display + data' trim as saved: replayed on a raw-FCS
            # reload, so it holds the events it was saved with whatever the
            # restore's own re-trim (over the samples present and ticked
            # NOW) would pick; that one only ever trims further.
            why = self._ds_restore_trim(name, sample, meta['downsample_trim'])
            if why:
                self._add_load_note(why)
        if meta:
            # The cell-cycle / FlowSOM models its label columns came with
            # (session_models). A record that does not fit the reopened
            # sample is left out with a log line: the columns still load,
            # and the window asks for a re-run as it did before.
            from . import session_models as _models
            for key, put in ((_models.CELL_CYCLE_KEY,
                              _models.restore_cell_cycle),
                             (_models.FLOWSOM_KEY, _models.restore_flowsom)):
                if meta.get(key) is None:
                    continue
                why = put(sample, meta[key])
                if why:
                    print(f"[session] {name}: {key} not restored ({why})",
                          flush=True)
            # The last spectral unmix (its spectra, detectors and QC report,
            # which the editor holds), checked against this sample's U:
            # columns; Panel check and Spectral QC read it again.
            if meta.get(_models.UNMIX_KEY) is not None:
                why = _models.restore_unmix(self, name, sample,
                                            meta[_models.UNMIX_KEY])
                if why:
                    print(f"[session] {name}: {_models.UNMIX_KEY} not "
                          f"restored ({why})", flush=True)
        # Its restored gates were drawn on the transforms it was saved with;
        # where adopting put the data on other parameters (a cofactor-5
        # asinh channel onto the editor's cofactor 150), they move with it.
        from .pipeline import gate_rescale_fns
        rescale = gate_rescale_fns(
            drawn_on, getattr(sample, 'data_transforms', None))
        # Keep the path⇄name registry in sync for entry points that bypass
        # `_sample_name_for` (e.g. processed-CSV load), so later loads still
        # see this name as taken and disambiguate around it.
        ap = os.path.normcase(os.path.abspath(getattr(sample, 'path', '') or ''))
        if ap:
            self._path_to_name.setdefault(ap, name)
        self._name_to_path.setdefault(name, ap or name)
        # Color is assigned lazily — only when a sample is actually displayed
        # (see `_color_for`). Until then the tree row stays neutral so loading
        # many trials doesn't paint a rainbow of undisplayed samples.

        # Record the sample's trial (its FCS folder, or a Day N above it) so the
        # tree can group it and the workspace can label its origin.
        from .workspace import derive_trial_name
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

        # Initialise per-sample gate state.
        self._sample_gates.setdefault(name, {})
        self._sample_gate_seq.setdefault(name, 0)
        self._sample_gate_order.setdefault(name, [])

        # Drain pending gates: a restored session bundles them in `meta` (keyed
        # by path, above); a .wsp ingest stages them in `_pending_sample_gates`
        # by name. We rebind `_gates` / `_gate_id_order` to this sample's
        # storage via `_set_active_sample`, populate, then leave it active iff
        # it was the first sample loaded.
        if meta is not None:
            pending = meta.get('gates') or None
            if pending and rescale:
                from .pipeline import map_gate_coords
                from .result_stamp import carry_gate_stamp

                # A change of units, not a hand edit: a stamped gate keeps
                # its stamp (resealed on the new coordinates).
                pending = [carry_gate_stamp(g, map_gate_coords(g, rescale))
                           for g in pending]
        else:
            pending = self._pending_sample_gates.pop(name, None)
            axes = getattr(self, '_pending_wsp_axes', {}).pop(name, None)
            if axes:
                sample.stat_axes = {c: dict(a) for c, a in axes.items()
                                    if c in sample.data.columns}
            if pending:
                # .wsp gates are in FlowJo's linear scale units; this sample
                # stores its fluorescence transformed. Unconverted, an
                # imported "FL1-A > 1000" compared logicle values (< ~1.2)
                # against 1000 and held no events.
                from .pipeline import (
                    gate_scale_fns,
                    map_gate_coords,
                    mark_unknown_scale,
                    wsp_uncompensated_channels,
                )
                stored = self._stored_transforms(name, sample)
                fns = gate_scale_fns(stored, 'from_linear')
                # A channel of unknown scale cannot take FlowJo's units: the
                # gate is kept but reads NA (mark_unknown_scale).
                pending = [mark_unknown_scale(map_gate_coords(g, fns), stored)
                           for g in pending]
                # A gate FlowJo drew on a channel's UNcompensated parameter
                # (its bare name, no 'Comp-') counts measured values there;
                # this sample holds only compensated ones (a gate at 2,500:
                # 8,413 events in FlowJo, 5,216 here). Said at load settle.
                comp = list(getattr(sample, 'comp_channels', None) or [])
                self._uncompensated_imports = (
                    getattr(self, '_uncompensated_imports', [])
                    + [(g.get('label') or g.get('name') or '?', ch)
                       for g in pending
                       for ch in wsp_uncompensated_channels(g, comp)])
        if pending:
            saved_active = self._active_sample
            self._set_active_sample(name)
            old_to_new = {}
            added, added_ids = [], []
            prev_suspend = self._suspend_undo
            self._suspend_undo = True       # bulk load isn't an undo step
            try:
                for raw in pending:
                    g = dict(raw)
                    src_id = g.pop('_import_id', None) or g.pop('id', None)
                    parent = g.get('parent_id')
                    if parent is not None:
                        g['parent_id'] = old_to_new.get(parent)
                    # A restored session's gates carry their saved flag, kept.
                    # A .wsp's carry none and come in ticked, as gates drawn
                    # here do: they used to come in unticked, so a workspace
                    # opened with every gate looking switched off.
                    g.setdefault('enabled', True)
                    # Siblings on one channel are distinct populations.
                    gid = self._add_gate(g, replace_1d=False)
                    added.append(g)
                    added_ids.append(gid)
                    if src_id is not None:
                        old_to_new[src_id] = gid
                # A session saves each gate's editor id, and the restore hands
                # out fresh sequential ones -- so once a gate has been deleted
                # the ids shift, and a boolean's saved operands named other
                # gates (measured: AND(CD3+, CD4+) came back as AND(CD4+,
                # itself), 0 events where 4,983 is right). Remapped like
                # parent_id, after the loop so every id is known.
                from .gating import remap_operands
                from .pipeline import broken_booleans, describe_gate
                for g in added:
                    remap_operands(g, old_to_new)
                # One that could not be resolved (its operand was deleted
                # before the save) now selects nothing, and so does a boolean
                # built on it: said once the load settles (_on_load_settled),
                # as Clear / Cut say it.
                restored = {gid: self._gates[gid] for gid in added_ids}
                lost_ids, built_ids = broken_booleans(restored)
                self._restore_lost_operands.extend(
                    restored[i].get('name') or describe_gate(restored[i])
                    for i in lost_ids)
                self._restore_built_on.extend(
                    restored[i].get('name') or describe_gate(restored[i])
                    for i in built_ids)
            finally:
                self._suspend_undo = prev_suspend
            # Named again when the load settles: that status replaces the
            # import's, and an empty placeholder must not pass unnoticed.
            self._unconverted_imports = (
                getattr(self, '_unconverted_imports', [])
                + [g for g in pending if g.get('kind') == 'flowjo_ellipse'])
            # A boolean built on one selects nothing as well (gate_to_mask).
            from .pipeline import placeholder_booleans
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
            if saved_active is not None and saved_active != name:
                # Restore the previously-active sample; this sample's
                # gates are now persisted in `_sample_gates[name]`.
                self._set_active_sample(saved_active)
        # Plot inclusion: enable ONLY the very first sample loaded — the
        # user gets an immediate render to confirm the load worked.
        # Subsequent loads start unchecked so opening a session with many
        # samples doesn't cascade-render N overlays on every Add.
        was_first = (len(self._samples) == 1)
        self._sample_plot_enabled.setdefault(name, was_first)

        # First sample populates the channel choices and becomes active.
        if len(self._samples) == 1:
            from .result_stamp import is_original_column
            self._channels       = [c for c in sample.data.columns
                                    if not is_original_column(c)]
            labels = dict(sample.channel_labels)
            if meta is not None:
                # Restored from a session: the labels it saved (a template's,
                # an edit) win over the file's own $PnS; the sample only fills
                # in channels the session had no label for. Replacing them
                # lost every saved label whenever a raw-FCS sample landed
                # first.
                labels.update(getattr(self, '_session_channel_labels', {}))
            self._channel_labels = labels
            # (_channel_transform was seeded from this sample above.)
            self._populate_channel_combos()
        # NOTE: later samples may carry columns the first didn't — those get
        # unioned into the combos in `_on_load_settled`. Doing it (plus the
        # tree rebuild, panel-mismatch scan and replot) PER sample is O(N) work
        # that, on a bulk resume of N samples, thrashes the Tk thread into a
        # multi-second freeze. It's deferred + coalesced below instead.
        if self._active_sample is None:
            self._set_active_sample(name)

        # Provenance: record the load (QC + auto-compensation + transform ran
        # in the loader). Capture the data identity for reproducibility.
        # FlowSample records where its matrix came from ('$SPILL', 'wsp:…',
        # 'manual'); None means nothing was applied. This used to read
        # 'auto/$SPILL' for every sample, including ones with no spillover.
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
        ''. The editor holds that channel compensated only, so these gates
        cannot count what FlowJo counts; openflo-compare can."""
        if not pairs:
            return ''
        print("[WSP] gate(s) on FlowJo's uncompensated parameter, evaluated "
              "here on compensated values: "
              + ', '.join(f"'{g}' ({ch})" for g, ch in pairs), flush=True)
        gates = list(dict.fromkeys(g for g, _ in pairs))
        chans = sorted({ch for _, ch in pairs})
        shown = ', '.join(f"'{g}'" for g in gates[:3]) + (
            f' (+{len(gates) - 3})' if len(gates) > 3 else '')
        return (f" — {len(gates)} imported gate(s) are on FlowJo's "
                f"UNcompensated {', '.join(chans)} (no 'Comp-' in the "
                f"workspace) but this sample holds it compensated, so their "
                f"counts will differ from FlowJo's: {shown}")

    def _on_load_settled(self):
        """Run once after a load burst quiesces: union channel choices, rebuild
        the tree (swapping ⏳ placeholders for real rows), re-select the most
        recently loaded sample, flag a mismatched fluor panel, and replot."""
        self._load_settle_after = None
        # Samples in name order, not the order the loader happened to finish
        # them in (ctrl_d1, ctrl_d2, treat_d1, ctrl_d3, ...).
        from .workspace import natural_sort_key
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
        # the newest arrival so the user sees it landed.
        last = getattr(self, '_last_loaded', None)
        active = self._active_sample
        for nm, act in ((active, self.gate_tv.selection_set),
                        (last, self.gate_tv.see)):
            if nm is not None:
                try:
                    act(self._sample_iid(nm))
                except Exception as exc:
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
        from .pipeline import _flowjo_placeholder_note
        msg += _flowjo_placeholder_note(
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
            self._add_load_note(
                f"imported gate '{g}' is on FlowJo's uncompensated {ch}, but "
                "this sample holds it compensated: its count will differ "
                "from FlowJo's")
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
            print(f"[DnD] drop event raw={raw!r}", flush=True)
            try:
                paths = list(self.tk.splitlist(raw))
            except Exception:
                paths = raw.split()
            if paths:
                self._import_dropped_paths(paths)
        except Exception as exc:
            print(f"[DnD] drop handler failed: {type(exc).__name__}: {exc}",
                  flush=True)
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
        except Exception as exc:
            messagebox.showwarning(
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
        except Exception as exc:
            self.status_var.set(f"Example data failed: {exc}")
            show_error(self, "Example data failed",
                       "The example data could not be made.", exc)
            return
        self._queue_fcs_loads(paths)
        self.status_var.set(
            f"Loading {len(paths)} example sample(s) — synthetic PBMC, "
            "ctrl vs treat (CD3/4/8/19/56/14).")

    def _toggle_watch_folder(self):
        """Watch a folder and auto-load new .fcs files as they appear (e.g. an
        instrument's export folder). Run again to stop."""
        if getattr(self, '_watch_dir', None):
            self._watch_dir = None
            wa = getattr(self, '_watch_after', None)
            if wa:
                try:
                    self.after_cancel(wa)
                except tk.TclError:
                    pass
                self._watch_after = None
            self.status_var.set("Stopped watching folder.")
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
        self._watch_seen = {f for f in os.listdir(d)
                            if f.lower().endswith('.fcs')}
        self.status_var.set(
            f"Watching {os.path.basename(d)} — new .fcs files load "
            "automatically (Tools → Watch folder again to stop).")
        self._poll_watch()

    def _poll_watch(self):
        d = getattr(self, '_watch_dir', None)
        if not d or not os.path.isdir(d):
            self._watch_dir = None
            return
        try:
            cur = {f for f in os.listdir(d) if f.lower().endswith('.fcs')}
            new = [os.path.join(d, f) for f in cur
                   if f not in self._watch_seen]
            if new:
                self._watch_seen.update(cur)
                self._queue_fcs_loads(new)
                self.status_var.set(
                    f"Watch: loading {len(new)} new file(s) from "
                    f"{os.path.basename(d)}…")
        except Exception as exc:
            print(f"[watch] {exc}", flush=True)
        self._watch_after = self.after(4000, self._poll_watch)
