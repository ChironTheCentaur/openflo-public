"""FCS / CSV / WSP loading, drag-drop, watch-folder.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import re
from tkinter import filedialog, messagebox

from .editor_base import EditorMixin

# Package directory (same value as gui.BASE) — used to locate bundled assets.
BASE = os.path.dirname(os.path.abspath(__file__))


class LoadMixin(EditorMixin):
    """Adding samples, drag-and-drop import, async load queue, watch-folder."""

    def _add_samples(self):
        """Add-FCS button: dialog picker, then queue.

        Accepts both ``.fcs`` and ``.wsp``. A workspace is "exploded":
        every ``<Sample>`` it references is queued for load, and the
        gate trees attached to each ``<SampleNode>`` are merged into
        that sample's per-sample gate storage as it finishes loading.
        """
        init = self.fcs_dir if self.fcs_dir and os.path.isdir(self.fcs_dir) else BASE
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Select FCS file(s) or FlowJo workspace",
            filetypes=[
                ('FCS & FlowJo workspace', '*.fcs *.wsp'),
                ('FCS files',              '*.fcs'),
                ('FlowJo workspace',       '*.wsp'),
                ('All files',              '*.*')])
        if not paths:
            return
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
        init = self.fcs_dir if self.fcs_dir and os.path.isdir(self.fcs_dir) else BASE
        paths = filedialog.askopenfilenames(
            initialdir=init, title="Load processed data (CSV)",
            filetypes=[('Processed CSV', '*.csv'), ('All files', '*.*')])
        if not paths:
            return
        import json

        import pandas as pd

        from .pipeline import FlowSample
        added = 0
        for p in paths:
            name = os.path.basename(p).rsplit('.', 1)[0]
            if name.endswith('_processed'):
                name = name[:-len('_processed')]
            if name in self._samples:
                self.status_var.set(f"{name} already loaded — skipped.")
                continue
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                self.status_var.set(f"Load CSV failed: {exc}")
                continue
            labels = None
            sidecar = os.path.join(os.path.dirname(p), f'{name}_labels.json')
            if os.path.isfile(sidecar):
                try:
                    with open(sidecar, encoding='utf-8') as fh:
                        labels = json.load(fh)
                except Exception:
                    labels = None
            s = FlowSample.from_dataframe(df, name=name, labels=labels, path=p)
            self._on_loaded(name, s)
            added += 1
        if added:
            self.status_var.set(
                f"Loaded {added} processed sample(s). Use Populations… to "
                "import cluster/FlowSOM columns; plot UMAP1/UMAP2 to view.")

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
            as-is, then the WSP's own directory, then the user-set
            ``self.fcs_dir``.
        """
        from xml.etree import ElementTree as ET

        from .compare import _resolve_fcs_uri
        from .pipeline import WspReader

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

        wsp_dir = os.path.dirname(os.path.abspath(wsp_path))
        fcs_dir_hint = (self.fcs_dir
                       if self.fcs_dir and os.path.isdir(self.fcs_dir)
                       else None)

        resolved, unresolved = [], []
        for sample_elem in root.iter('Sample'):
            ds = sample_elem.find('DataSet')
            uri = ds.get('uri') if ds is not None else None
            sn  = sample_elem.find('SampleNode')
            if sn is None:
                continue
            # Try the uri as-is first, then the WSP's own folder, then
            # the editor's fcs_dir hint. _resolve_fcs_uri already covers
            # the first + fcs_dir; we add the WSP-folder fallback here.
            fcs_path = _resolve_fcs_uri(uri, fcs_dir_hint)
            if fcs_path is None and uri:
                from urllib.parse import unquote, urlparse
                raw = unquote(urlparse(uri).path) if uri.startswith('file:') else uri
                cand = os.path.join(wsp_dir, os.path.basename(raw))
                if os.path.isfile(cand):
                    fcs_path = cand
            if fcs_path is None:
                unresolved.append(sn.get('name') or '(unnamed)')
                continue

            # Per-sample gate extraction — scope the reader's walker to
            # just this <SampleNode>. Returns the same gate-dict format
            # as a full-document extract_gates() call.
            gates = reader.extract_gates(sample_node=sn)
            # Same name the FCS queue will assign (collision-safe per day), so
            # `_on_loaded` drains these gates onto the right sample.
            sample_name = self._sample_name_for(fcs_path)
            self._pending_sample_gates[sample_name] = gates
            resolved.append((sample_name, fcs_path, len(gates)))

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
        self.status_var.set(summary)
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

    def _on_loaded(self, name, sample):
        self._loading.discard(name)
        # Propagate downsample to this freshly-loaded sample BEFORE we
        # publish it to self._samples so the first replot already sees
        # the trimmed size.
        if (getattr(self, 'ds_propagate_var', None) is not None
                and self.ds_propagate_var.get()
                and self._samples):
            floor = self._smallest_loaded_sample_size()
            if floor is not None and floor > 0 and len(sample.data) > floor:
                sample.data = sample.data.sample(
                    floor, random_state=42).reset_index(drop=True)
        self._samples[name] = sample
        if name not in self._sample_order:
            self._sample_order.append(name)
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

        # Record the sample's trial (grandparent folder of its FCS path) so the
        # tree can group it and the workspace can label its origin.
        from .workspace import derive_trial_name
        trial = derive_trial_name(getattr(sample, 'path', None))
        # A restored session may pin this sample to a manually-regrouped day /
        # Comps-Samples side (and carries its gates). It's keyed by FILE PATH,
        # not name, so collision-disambiguated names can't mismatch on reload.
        spath = getattr(sample, 'path', '') or ''
        pkey = os.path.normcase(os.path.abspath(spath)) if spath else None
        meta = self._pending_sample_meta.pop(pkey, None) if pkey else None
        if meta:
            if meta.get('trial'):
                trial = meta['trial']
            if 'is_comp' in meta:
                self._sample_is_comp[name] = bool(meta['is_comp'])
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
        else:
            pending = self._pending_sample_gates.pop(name, None)
        if pending:
            saved_active = self._active_sample
            self._set_active_sample(name)
            old_to_new = {}
            prev_suspend = self._suspend_undo
            self._suspend_undo = True       # bulk load isn't an undo step
            try:
                for raw in pending:
                    g = dict(raw)
                    src_id = g.pop('_import_id', None) or g.pop('id', None)
                    parent = g.get('parent_id')
                    if parent is not None:
                        g['parent_id'] = old_to_new.get(parent)
                    # Imported gates start DISABLED so a freshly-loaded sample
                    # (only the first is displayed) isn't a wall of active
                    # toggles. WSP gates carry no 'enabled' → default off; a
                    # restored session's gates carry their saved flag → kept.
                    g.setdefault('enabled', False)
                    gid = self._add_gate(g)
                    if src_id is not None:
                        old_to_new[src_id] = gid
            finally:
                self._suspend_undo = prev_suspend
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
            self._channels       = list(sample.data.columns)
            self._channel_labels = dict(sample.channel_labels)
            # Loader applies logicle to fluor channels; everything else is
            # left linear. Seeds the per-channel transform editor.
            fluor = set(getattr(sample, 'fluor_channels', []) or [])
            self._channel_transform = {
                c: ('logicle' if c in fluor else 'linear')
                for c in self._channels}
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
        comp = getattr(sample, 'compensation_source', None) or 'auto/$SPILL'
        self._audit('sample.load', sample=name,
                    path=getattr(sample, 'path', '') or '',
                    n_events=int(len(sample.data)),
                    channels=int(sample.data.shape[1]),
                    trial=self._sample_trial.get(name, ''),
                    compensation=comp)
        # Cheap per-sample status. The expensive UI sync is coalesced into one
        # debounced pass so a burst of loads doesn't lock the window.
        self._last_loaded = name
        self.status_var.set(
            f"{len(self._samples)} sample(s) loaded "
            f"(latest: {name}, {len(sample.data):,} events)…")
        self._schedule_load_settle()

    def _schedule_load_settle(self, ms=120):
        """Debounce the heavy post-load UI sync. Each `_on_loaded` (re)arms a
        single timer; a burst of N loads collapses to ~one settle pass instead
        of N full tree/combo rebuilds on the Tk thread."""
        prev = getattr(self, '_load_settle_after', None)
        if prev:
            try:
                self.after_cancel(prev)
            except Exception:
                pass
        self._load_settle_after = self.after(ms, self._on_load_settled)

    def _on_load_settled(self):
        """Run once after a load burst quiesces: union channel choices, rebuild
        the tree (swapping ⏳ placeholders for real rows), re-select the most
        recently loaded sample, flag a mismatched fluor panel, and replot."""
        self._load_settle_after = None
        if len(self._samples) > 1:
            # Union columns the first sample lacked into the axis/colour combos.
            self._refresh_channel_choices()
        self._refresh_gate_list()
        last = getattr(self, '_last_loaded', None)
        if last is not None:
            try:
                self.gate_tv.selection_set(self._sample_iid(last))
                self.gate_tv.see(self._sample_iid(last))
            except Exception:
                pass
        msg = f"{len(self._samples)} sample(s) loaded. Double-click the plot to add a gate."
        # Cross-sample stats tie by antibody label, so a mismatched panel just
        # means some labels won't be shared — surface it, don't block.
        if self._fluor_panel_warning():
            msg += "  [!] samples differ in fluor panel — see Statistics."
        self.status_var.set(msg)
        self._schedule_replot(0)

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
        except Exception:
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
            messagebox.showerror("Example data failed",
                                 f"{type(exc).__name__}: {exc}", parent=self)
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
                except Exception:
                    pass
                self._watch_after = None
            self.status_var.set("Stopped watching folder.")
            return
        d = filedialog.askdirectory(title="Watch folder for new FCS files")
        if not d:
            return
        self._watch_dir = d
        # Seed with what's already there so only NEW files load.
        self._watch_seen = {f.lower() for f in os.listdir(d)
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
            cur = {f.lower(): f for f in os.listdir(d)
                   if f.lower().endswith('.fcs')}
            new = [os.path.join(d, cur[k]) for k in cur
                   if k not in self._watch_seen]
            if new:
                self._watch_seen.update(cur.keys())
                self._queue_fcs_loads(new)
                self.status_var.set(
                    f"Watch: loading {len(new)} new file(s) from "
                    f"{os.path.basename(d)}…")
        except Exception as exc:
            print(f"[watch] {exc}", flush=True)
        self._watch_after = self.after(4000, self._poll_watch)
