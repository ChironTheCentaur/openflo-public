"""Tools-menu launchers + compensation / unmix / cytonorm backend.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .editor_base import EditorMixin


class ToolsMixin(EditorMixin):
    """Compensation, transforms, calibration, CytoNorm, spectral unmix, figure layout, gate-tree, and the standalone tool dialogs."""

    def _batch_correct_cytonorm(self):
        """Batch-correct loaded samples with CytoNorm 2.0 (control-free 'goal'
        mode). Samples are batched by their trial/day; per-marker intensities
        are quantile-normalized within FlowSOM metaclusters onto the pooled
        goal. Modifies each sample's data in place (re-add samples to revert).
        The classic controls-anchored mode is CLI-only."""
        import pandas as pd

        from .pipeline import CytoNorm
        from .workspace import proper_run_channels

        by_batch: dict[str, list] = {}
        for name in self._sample_order:
            s = self._samples.get(name)
            if s is not None:
                by_batch.setdefault(
                    self._sample_trial.get(name, 'Trial'), []).append((name, s))
        if len(by_batch) < 2:
            self.status_var.set("Batch-norm needs ≥2 batches — load samples "
                                "from multiple trials/days first.")
            return
        # Shared marker channels present in every sample.
        all_samples = [s for lst in by_batch.values() for _, s in lst]
        shared = None
        for s in all_samples:
            cs = set(proper_run_channels(s))
            shared = cs if shared is None else (shared & cs)
        channels = [c for c in proper_run_channels(all_samples[0])
                    if c in (shared or set()) and c in all_samples[0].data.columns]
        if len(channels) < 2:
            self.status_var.set("Batch-norm: <2 shared marker channels across "
                                "the loaded samples.")
            return

        nsamp = len(all_samples)
        if not messagebox.askyesno(
                "Batch-normalize (CytoNorm 2.0)",
                f"Normalize {nsamp} sample(s) across {len(by_batch)} batches "
                f"on {len(channels)} markers?\n\n"
                f"This modifies the loaded sample data in place (gating, "
                f"clustering and stats will use the corrected values). "
                f"Re-add the samples to revert — it is not undoable.",
                parent=self):
            return

        # Fit + apply off the Tk thread; the corrected frames are computed there
        # (reading sample data, no mutation) and only ASSIGNED back to
        # ``s.data`` on the Tk thread in _done — so the window stays responsive
        # and there's no cross-thread write race.
        def _work():
            events_by_batch = {}
            for batch, lst in by_batch.items():
                frames = [s.data[channels] for _, s in lst
                          if all(c in s.data.columns for c in channels)]
                if frames:
                    events_by_batch[batch] = pd.concat(frames,
                                                       ignore_index=True)
            if len(events_by_batch) < 2:
                return None
            cn = CytoNorm(channels, n_metaclusters=10, mode='goal').fit(
                events_by_batch)
            qc = cn.qc(events_by_batch)
            corrected = {name: cn.apply(s.data, batch)
                         for batch, lst in by_batch.items()
                         for name, s in lst}
            return {'corrected': corrected, 'qc': qc,
                    'batches': sorted(events_by_batch)}

        def _done(res):
            if res is None:
                self.status_var.set("Batch-norm: <2 usable batches.")
                return
            for name, newdf in res['corrected'].items():
                s = self._samples.get(name)
                if s is None:
                    continue
                s.data = newdf
                # data object changed → drop its cached masks.
                self._data_changed(name)
            qc = res['qc']
            # NaN marks a channel that could not be evaluated; averaging it
            # in would hide every real score behind one unmeasurable marker.
            before = float(np.nanmean([d['before'] for d in qc.values()]))
            after = float(np.nanmean([d['after'] for d in qc.values()]))
            if np.isfinite(before) and before > 0:
                msg = (f"CytoNorm applied: {len(channels)} markers · "
                       f"{len(res['batches'])} batches · mean batch→goal "
                       f"distance {before:.3f} → {after:.3f} "
                       f"({100 * (1 - after / before):.0f}% lower).")
            else:
                msg = "CytoNorm applied."
            self._refresh_gate_list()
            self._schedule_replot(0)
            self._audit('cytonorm', mode='goal', n_metaclusters=10,
                        n_samples=nsamp, n_batches=len(res['batches']),
                        batches=res['batches'], n_markers=len(channels),
                        dist_before=round(before, 4),
                        dist_after=round(after, 4))
            self.status_var.set(msg)

        def _err(exc):
            import traceback
            traceback.print_exc()
            self.status_var.set(f"CytoNorm failed: {type(exc).__name__}: {exc}")

        self.run_async(_work, on_done=_done, on_error=_err,
                       busy_msg=(f"CytoNorm: fitting across {len(by_batch)} "
                                 f"batches on {len(channels)} markers…"))

    def _open_spectral_unmix(self):
        """Open the spectral-unmixing dialog. Designate loaded single-stain
        controls (→ fluorophore) and an unstained control; every other loaded
        sample gets unmixed into per-fluor ``U:`` abundance channels."""
        if len(self._samples) < 2:
            self.status_var.set(
                "Load the single-stain controls (+ unstained) and your "
                "sample(s) first, then Unmix.")
            return
        names = [n for n in self._sample_order if n in self._samples]
        ref = self._samples.get(self._active_sample) or self._samples[names[0]]
        detectors = list(getattr(ref, 'fluor_channels', None) or [])
        from .ui_spectral_unmix import SpectralUnmixDialog
        SpectralUnmixDialog(self, names, detectors, self._apply_spectral_unmix)

    def _apply_spectral_unmix(self, singles, unstained, detectors, nonneg):
        """Build reference spectra from the assigned controls and unmix every
        non-control loaded sample, adding ``U:<fluor>`` abundance channels. The
        spectra build + per-sample unmix + QC run off the Tk thread on detector
        arrays snapshotted here; only the resulting ``U:`` columns are written to
        ``s.data`` on the Tk thread (in on_done) — responsive, no write race."""
        from .pipeline import UnknownScaleError
        from .spectral import (
            build_reference_spectra,
            measured_signal,
            unmix,
            unmixing_qc,
        )
        # Declared at method scope so the on_done closure may assign them.
        self._last_unmix_qc = getattr(self, '_last_unmix_qc', None)
        self._last_unmix_spectra = getattr(self, '_last_unmix_spectra', None)

        def _linear(s, cols):
            # Unmixing is LINEAR least squares (Y = A @ S) on what the
            # detectors MEASURED: not the loader's logicle (abundances off by
            # up to 4,999 on a 0-5,000 scale, Spearman 0.72-0.79 against the
            # truth; on linear input, 6e-4) and not the $SPILL compensation
            # the loader applied either (see measured_signal). The CLI unmixes
            # the same measured values.
            return measured_signal(s, cols)

        try:
            stains = {}
            for nm, fluor in singles.items():
                s = self._samples[nm]
                cols = [d for d in detectors if d in s.data.columns]
                stains[fluor] = _linear(s, cols)
            un = None
            if unstained and unstained in self._samples:
                s = self._samples[unstained]
                un = _linear(s, [d for d in detectors if d in s.data.columns])
        except Exception as exc:
            self.status_var.set(
                f"Spectral build failed: {type(exc).__name__}: {exc}")
            return
        control_names = set(singles) | ({unstained} if unstained else set())
        # Per-group: this reference set was built from THESE controls, so only
        # unmix samples in the same group/trial as the controls. Samples in
        # other groups need their own controls — never unmix several groups
        # together with one shared spectra set.
        control_trials = {self._trial_for(nm) for nm in control_names
                          if nm in self._samples}
        # Snapshot each target's detector array on the Tk thread.
        targets = []           # (name, ndarray)
        skipped_groups = 0
        unknown_scale = []     # samples whose detectors' scale is unknown
        for nm in list(self._samples):
            if nm in control_names:
                continue
            if control_trials and self._trial_for(nm) not in control_trials:
                skipped_groups += 1
                continue
            s = self._samples[nm]
            cols = [d for d in detectors if d in s.data.columns]
            try:
                targets.append((nm, _linear(s, cols) if cols else None))
            except UnknownScaleError:
                unknown_scale.append(nm)     # no linear detectors: left out

        def _work():
            diag = []
            spectra, fluors = build_reference_spectra(
                stains, unstained=un, detectors=detectors, diagnostics=diag)
            results = []       # (name, abundance matrix A)
            for nm, Y in targets:
                if Y is None or Y.shape[1] != spectra.shape[1]:
                    continue
                results.append((nm, unmix(Y, spectra, nonneg=nonneg)))
            qc_stains = dict(stains)
            if un is not None and 'Autofluorescence' in fluors:
                qc_stains['Autofluorescence'] = un
            try:
                qc = unmixing_qc(qc_stains, spectra, fluors, unstained=un,
                                 reference_spectra=diag)
            except Exception as exc:
                print(f"[spectral-qc] {type(exc).__name__}: {exc}", flush=True)
                qc = None
            return {'spectra': spectra, 'fluors': fluors, 'results': results,
                    'qc': qc, 'spectra_diag': diag}

        def _done(res):
            from .pipeline import transform_spec, transform_values
            spectra, fluors = res['spectra'], res['fluors']
            qc = res['qc']
            diag = res.get('spectra_diag') or []
            applied = 0
            for nm, A in res['results']:
                s = self._samples.get(nm)
                if s is None:
                    continue
                rec = getattr(s, 'data_transforms', None)
                for j, f in enumerate(fluors):
                    col = f'U:{f}'
                    if rec is None:
                        # Nowhere to record a transform: linear, as given.
                        s.data[col] = A[:, j]
                        continue
                    # Abundances are on the detectors' linear intensity scale,
                    # so they take the detectors' default display (logicle),
                    # or whatever the Transform editor has since put this
                    # column on -- stored that way and RECORDED, so
                    # linear_values and statistics get the abundance back.
                    method = self._channel_transform.setdefault(col, 'logicle')
                    s.data[col] = transform_values(A[:, j], method=method)
                    rec = {k: v for k, v in rec.items() if k != col}
                    if method != 'linear':
                        rec[col] = transform_spec(method)
                if rec is not None:
                    s.data_transforms = rec
                self._data_changed(nm)
                applied += 1
            self._refresh_channel_choices()
            self._plot_reference_spectra(spectra, fluors)
            self._last_unmix_qc = qc
            self._last_unmix_spectra = (spectra, fluors)
            self._audit('unmix', n_samples=applied, n_fluors=len(fluors),
                        n_detectors=int(spectra.shape[1]),
                        fluors=list(fluors), nonneg=bool(nonneg),
                        unstained=unstained or None,
                        condition_number=(round(qc['condition_number'], 1)
                                          if qc else None),
                        similar_pairs=(len(qc['similar_pairs']) if qc else None),
                        spectra_methods={d['fluor']: d['method'] for d in diag},
                        dim_controls=[d['fluor'] for d in diag
                                      if d['warning']])
            sim_note = ""
            if qc and qc['similar_pairs']:
                sim_note = (f"  [!] {len(qc['similar_pairs'])} spectrally-"
                            f"similar pair(s) — see Spectral QC.")
            # A spectrum estimated differently from before, or one too dim to
            # trust, is otherwise invisible: it plots like any other.
            matched = [d['fluor'] for d in diag if d['method'] == 'matched']
            if matched:
                sim_note += (f"  {', '.join(matched)}: spectrum estimated "
                             "against matched autofluorescence.")
            dim = [d['fluor'] for d in diag if d['warning']]
            if dim:
                sim_note += (f"  [!] Too dim for a reliable spectrum: "
                             f"{', '.join(dim)} (use beads or a brighter "
                             "fluor) — see Spectral QC.")
            grp_note = ""
            if skipped_groups:
                grp_note = (f"  ({skipped_groups} sample(s) in other groups "
                            "left unmixed — run Unmix per group with its own "
                            "controls.)")
            if unknown_scale:
                grp_note += (f"  [!] Left unmixed, detector scale unknown "
                             f"(older OpenFlo CSV): {', '.join(unknown_scale)}.")
            self.status_var.set(
                f"Unmixed {applied} sample(s) in this group → {len(fluors)} U: "
                f"channels ({len(fluors)} fluors × {spectra.shape[1]} "
                f"detectors). Select a 'U:' channel to plot.{sim_note}{grp_note}")
            self._refresh_gate_list()
            if qc is not None:
                self._show_spectral_qc(qc)

        def _err(exc):
            self.status_var.set(
                f"Spectral unmix failed: {type(exc).__name__}: {exc}")

        self.run_async(_work, on_done=_done, on_error=_err,
                       busy_msg="Spectral unmixing…")

    def _show_spectral_qc(self, qc=None):
        """Open the Spectral-QC window for the given (or last) unmixing QC
        report: similarity + spillover-spread heatmaps, condition number, and
        the flagged similar/spread pairs, with export."""
        from .ui_spectral_qc import SpectralQCWindow
        qc = qc or self._last_unmix_qc
        if qc is None:
            self.status_var.set("Run Unmix first — no spectral QC yet.")
            return
        SpectralQCWindow(self, qc, audit=self._audit)

    def _open_transform_editor(self):
        """Per-channel display-transform editor. Re-maps each channel's
        transform across every loaded sample. Gates already drawn on a
        re-transformed channel keep their old coordinates, so the user is
        warned to re-check them."""
        from .pipeline import TRANSFORM_METHODS
        if not self._samples:
            self.status_var.set("Load a sample first.")
            return
        dlg = tk.Toplevel(self)
        dlg.title("Channel transforms")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.grab_set()
        dlg.geometry("380x460")

        ttk.Label(dlg, text="Transform per channel:",
                  font=('TkDefaultFont', 9, 'bold')).pack(
            side='top', fill='x', padx=10, pady=(10, 4))

        holder = ttk.Frame(dlg)
        holder.pack(side='top', fill='both', expand=True, padx=10, pady=6)
        cv = tk.Canvas(holder, highlightthickness=0)
        sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        cv.pack(side='left', fill='both', expand=True)
        inner = ttk.Frame(cv)
        _win = cv.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: cv.configure(scrollregion=cv.bbox('all')))
        # Stretch the inner frame to the canvas width so there's no dead
        # column of (now-dark, but still empty) space on the right.
        cv.bind('<Configure>', lambda e: cv.itemconfigure(_win, width=e.width))

        combos = {}
        for ch in self._channels:
            row = ttk.Frame(inner)
            row.pack(side='top', fill='x', pady=1)
            ttk.Label(row, text=self._fmt_channel(ch), width=22).pack(
                side='left')
            var = tk.StringVar(
                value=self._channel_transform.get(ch, 'linear'))
            ttk.Combobox(row, textvariable=var, state='readonly',
                         values=list(TRANSFORM_METHODS)).pack(
                side='left', fill='x', expand=True, padx=(0, 8))
            combos[ch] = var

        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)

        def do_apply():
            new = {ch: var.get() for ch, var in combos.items()}
            # 'log' has no value for an event at or below zero, and switching
            # back later inverts NaN to NaN, so the loss is permanent for the
            # loaded data and the session. Asked here, before anything moves;
            # 'No' leaves the dialog open to pick another method.
            drop = self._log_would_drop(new)
            if drop and not messagebox.askyesno(
                    "Log transform",
                    "These events are at or below zero (compensated) and "
                    "have no logarithm:\n\n"
                    + '\n'.join(f"  {self._fmt_channel(ch)}: {n:,} of "
                                f"{tot:,} events ({100 * n / tot:.1f}%)"
                                for ch, (n, tot) in drop.items())
                    + "\n\nOn 'log' they get no value: they leave every gate "
                    "and statistic on that channel, and switching back later "
                    "will not restore them (reloading the sample after "
                    "switching back does). "
                    "logicle, hyperlog and asinh keep them; an axis can also "
                    "be shown on a log scale without changing the data.\n\n"
                    "Switch to log anyway?",
                    icon='warning', default='no', parent=dlg):
                return
            dlg.destroy()
            # Backgrounded: _apply_channel_transforms now sets its own status
            # and replots on completion.
            self._apply_channel_transforms(new)

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply", command=do_apply).pack(
            side='right', padx=(0, 6))

    def _open_figure_layout(self):
        """Open the figure-layout dialog (small-multiple publication figure
        builder) seeded from the current plot selection."""
        samples = self._selected_samples()
        if not samples:
            messagebox.showinfo(
                "Figure layout",
                "Enable one or more samples (☑ in the tree) first.",
                parent=self)
            return
        cur_x = self.x_combo.get()
        cur_y = self.y_combo.get()
        xc = self._resolve_channel(cur_x)
        yc = self._resolve_channel(cur_y)
        default_pairs = ''
        if xc and yc:
            default_pairs = f"{self._fmt_channel(xc)} / {self._fmt_channel(yc)}"
        from .ui_figure_layout import FigureLayoutDialog
        FigureLayoutDialog(self, len(samples), self.mode_var.get(),
                           default_pairs, self._build_and_preview_figure)

    def _open_layout_editor(self):
        """The side panel's "Layout (to be added)" button. A FlowJo-style
        layout editor (a page to arrange plots and stats on, batched across
        samples) is not built yet; say so, and offer the preset multi-panel
        figure builder that does exist."""
        if messagebox.askyesno(
                "Layout (to be added)",
                "A FlowJo-style layout editor is to be added.\n\n"
                "Until then, Figure layout… builds a multi-panel figure "
                "from preset arrangements (per sample, per channel pair, "
                "or a grid). Open it now?",
                parent=self):
            self._open_figure_layout()

    def _open_voltage_dialog(self):
        """PMT voltage / stain-index titration optimizer."""
        from .ui_voltage import VoltageDialog
        VoltageDialog(self)

    def _open_compare_wsp(self):
        """Compare a FlowJo .wsp's gate counts against OpenFlo's."""
        from .ui_compare import CompareWspDialog
        CompareWspDialog(self)

    def _open_synthetic_dialog(self):
        """Generate a synthetic dataset and load it."""
        from .ui_synth import SyntheticDialog
        SyntheticDialog(self)

    def _open_quick_preview(self):
        """Quick single-sample raw-data QC viewer."""
        from .ui_preview import QuickPreviewDialog
        QuickPreviewDialog(self)

    def _open_fcs_inspector(self):
        """Raw FCS metadata viewer (channels, keywords, spillover)."""
        from .ui_inspect import FcsInspectorDialog
        FcsInspectorDialog(self)

    def _open_comp_qc(self):
        """Show a spillover heatmap + metrics for the active sample's
        compensation matrix."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        s = self._samples[name]
        mat = getattr(s, 'comp_matrix', None)
        chans = getattr(s, 'comp_channels', None)
        if mat is None or not chans:
            messagebox.showinfo(
                "Compensation QC",
                "This sample has no compensation matrix yet. Apply or import "
                "one first (Tools → Compensation…).", parent=self)
            return
        try:
            from .comp_qc import comp_qc_figure
            fig = comp_qc_figure(mat, list(chans), title=name)
        except Exception as exc:
            self.status_var.set(f"Compensation QC failed: {exc}")
            return
        from .ui_figure_window import _FigureWindow
        _FigureWindow(self, fig, f"Compensation QC — {name}")

    def _open_gate_tree(self):
        """Show the active sample's gating hierarchy as a diagram."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        if not self._gates:
            messagebox.showinfo("Gating tree", "This sample has no gates yet.",
                                parent=self)
            return
        try:
            from .gatetree import gate_tree_figure
            from .theme import _dialog_dark_on
            fig = gate_tree_figure(dict(self._gates), sample_name=name,
                                   dark=_dialog_dark_on(self))
        except Exception as exc:
            self.status_var.set(f"Gating tree failed: {exc}")
            return
        from .ui_figure_window import _FigureWindow
        _FigureWindow(self, fig, f"Gating tree — {name}")

    def _open_calibration_dialog(self):
        from .ui_calibration import CalibrationDialog
        if not self._samples:
            self.status_var.set("Load a bead sample to calibrate.")
            return
        CalibrationDialog(self)

    def _open_comp_editor(self):
        """Pop the compensation matrix editor against the active sample
        (so the editor can auto-import from $SPILL / a sibling .wsp / a
        sibling compensation.csv). When the user clicks Apply, the
        active sample's data is re-compensated in place; subsequent
        gate evaluations and plots use the corrected values."""
        from .ui_comp import CompensationEditorWindow
        if self._active_sample is None or self._active_sample not in self._samples:
            self.status_var.set(
                "Pick a sample first — the editor uses it to find a "
                "matrix and to apply the result.")
            return
        sample = self._samples[self._active_sample]

        def _on_apply(channels, matrix):
            try:
                # Re-load the raw FCS so a second Apply doesn't compound
                # compensation on top of an already-compensated copy, then
                # run the SAME pipeline the loader does — QC, compensate,
                # and (critically) the logicle transform. Without the
                # transform the data would be left on the raw linear scale
                # while the plots / gates / axis scales all expect logicle
                # space, which looks like gross over-compensation (every
                # channel slammed negative).
                from .pipeline import FlowSample
                # A sample restored from a processed CSV records its raw FCS
                # in `source_fcs`; its `path` is the CSV, which can't be
                # re-read as an FCS.
                fresh = FlowSample(getattr(sample, 'source_fcs', None)
                                   or sample.path)
                # The same event set as the sample being replaced: QC only if
                # it ran when the sample was loaded (whatever Preferences
                # says now), so the gates keep counting the same events.
                if getattr(sample, 'qc_applied', False):
                    fresh.run_qc()
                fresh.manual_compensate(matrix, list(channels))
                if not fresh.is_compensated():
                    # No channel matched, or the matrix is singular: the
                    # sample keeps its current compensation, and must not be
                    # reported as recompensated.
                    self.status_var.set(
                        "Compensation NOT applied — the matrix is singular "
                        "or names none of this sample's channels (see log).")
                    return
                fresh.apply_transform()
                # The loader's logicle is only the default: a channel the
                # Transform editor has since put on another method must come
                # back on THAT method, or the data and the axes (and the next
                # re-transform, which inverts the recorded method) disagree.
                self._conform_sample_transforms(fresh)
                # Preserve antibody labels set on the original sample.
                if getattr(sample, 'channel_labels', None):
                    fresh.set_labels(dict(sample.channel_labels))
                # Drop the new data into the existing FlowSample so every
                # downstream reference (self._samples[name].data, etc.)
                # sees the recompensated values.
                sample.data = fresh.data
                sample.data_transforms = dict(fresh.data_transforms)
                # Persist the applied matrix so it survives a reopen of the
                # editor and rides along in the .wsp export.
                sample.comp_matrix   = getattr(fresh, 'comp_matrix', None)
                sample.comp_channels = list(getattr(fresh, 'comp_channels', []))
                sample.compensation_source = fresh.compensation_source
                self._data_changed([n for n, smp in self._samples.items()
                                    if smp is sample])
                self.status_var.set(
                    f"Applied {len(channels)}×{len(channels)} matrix to "
                    f"'{self._active_sample}'.")
                self._schedule_replot(0)
            except Exception as exc:
                self.status_var.set(f"Compensation failed: {exc}")

        CompensationEditorWindow(self, sample=sample, on_apply=_on_apply)
