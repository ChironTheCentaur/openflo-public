"""Tools-menu launchers + compensation / unmix / cytonorm backend.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .editor_base import EditorMixin


class ToolsMixin(EditorMixin):
    """Compensation, transforms, calibration, CytoNorm, spectral unmix, figure layout, gate-tree, and the standalone tool dialogs."""

    def _batch_correct_cytonorm(self):
        """Tools → Normalize batches (CytoNorm)…: batches by trial or an FCS
        keyword, a reference sample per batch (or CytoNorm 2.0's pooled
        goal), applied to the loaded samples and/or written as FCS copies,
        with a before/after view (ui_batch_normalize)."""
        if len(self._samples) < 2:
            self.status_var.set("Normalizing needs samples from at least 2 "
                                "batches — load them first.")
            return
        from .ui_batch_normalize import BatchNormalizeWindow
        self._batch_norm_win = BatchNormalizeWindow(self)

    def _restore_batch_norm(self):
        """Tools → Undo batch normalization: put every normalized sample back
        on the values it had before the first normalization (kept in
        memory when it was applied), without reloading. Returns how many
        samples were restored."""
        keep = getattr(self, '_batch_norm_originals', None) or {}
        done = []
        for name, df in keep.items():
            s = self._samples.get(name)
            if s is None:
                continue
            s.data = df
            s.batch_norm = None
            self._data_changed(name)
            done.append(name)
        self._batch_norm_originals = {}
        if done:
            self._audit('cytonorm.restored', samples=done)
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set(f"Batch normalization undone: {len(done)} "
                                "sample(s) back on their original values.")
        elif any(getattr(s, 'batch_norm', None)
                 for s in self._samples.values()):
            # Normalized in an earlier sitting: the session kept the
            # normalized values, not the ones before, so there is nothing
            # here to go back to.
            self.status_var.set("Nothing to undo: the values from before "
                                "batch normalization are not kept once the "
                                "session is closed. Reload the FCS files "
                                "to start from the raw values.")
        else:
            self.status_var.set("No batch normalization to undo.")
        return len(done)

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

    def _apply_spectral_unmix(self, singles, unstained, detectors, nonneg,
                              af_signatures=1):
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
        self._last_unmix_pairs = getattr(self, '_last_unmix_pairs', None)
        self._last_unmix_detectors = getattr(self, '_last_unmix_detectors', None)

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
            results = []       # (name, abundance matrix A, AF index)
            multi = (af_signatures > 1 and un is not None
                     and 'Autofluorescence' in fluors)
            if multi:
                # Several AF spectra from the unstained control; each cell
                # keeps the one that explains it best (Complaints 24).
                from .spectral_af import af_signatures as find_af
                from .spectral_af import unmix_multi_af
                j_af = fluors.index('Autofluorescence')
                S_f = np.delete(spectra, j_af, axis=0)
                sigs = find_af(un, k=int(af_signatures))
            for nm, Y in targets:
                if Y is None or Y.shape[1] != spectra.shape[1]:
                    continue
                if multi:
                    A_f, ab, idx = unmix_multi_af(Y, S_f, sigs, nonneg=nonneg)
                    results.append((nm, np.insert(A_f, j_af, ab, axis=1),
                                    np.where(idx >= 0, idx + 1.0, np.nan)))
                else:
                    results.append((nm, unmix(Y, spectra, nonneg=nonneg),
                                    None))
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
                    'qc': qc, 'spectra_diag': diag,
                    'n_af': len(sigs) if multi else 1}

        def _done(res):
            from .pipeline import transform_spec, transform_values
            spectra, fluors = res['spectra'], res['fluors']
            qc = res['qc']
            diag = res.get('spectra_diag') or []
            applied = 0
            self._last_unmix_pairs = None
            for nm, A, af_idx in res['results']:
                s = self._samples.get(nm)
                if s is None:
                    continue
                if self._last_unmix_pairs is None:
                    self._last_unmix_pairs = (nm, A, list(fluors))
                if af_idx is not None:
                    # Which AF signature each cell took: 1, 2, ... (linear).
                    s.data['U:AF index'] = af_idx
                    self._channel_transform['U:AF index'] = 'linear'
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
            self._last_unmix_detectors = list(detectors)
            self._audit('unmix', n_samples=applied, n_fluors=len(fluors),
                        n_detectors=int(spectra.shape[1]),
                        fluors=list(fluors), nonneg=bool(nonneg),
                        af_signatures=res.get('n_af', 1),
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
            self.status_var.set(
                "No spectral QC yet: run Tools, Spectral unmix first.")
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
                  font='OpenFloHeading').pack(
            side='top', fill='x', padx=10, pady=(10, 4))

        # Buttons before the list: the packer hides what it packs last.
        btns = ttk.Frame(dlg)
        btns.pack(side='bottom', fill='x', padx=10, pady=10)
        from .tool_window import scrolled_body
        inner = scrolled_body(dlg, padding=(10, 6, 10, 6))
        inner.columnconfigure(1, weight=1)

        combos = {}
        for i, ch in enumerate(self._channels):
            # Grid, not a fixed-width label: the label column is as wide as
            # the longest channel name and every dropdown keeps its width.
            ttk.Label(inner, text=self._fmt_channel(ch)).grid(
                row=i, column=0, sticky='w', padx=(0, 8), pady=1)
            var = tk.StringVar(
                value=self._channel_transform.get(ch, 'linear'))
            ttk.Combobox(inner, textvariable=var, state='readonly', width=10,
                         values=list(TRANSFORM_METHODS)).grid(
                row=i, column=1, sticky='ew', padx=(0, 8), pady=1)
            combos[ch] = var


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
        ttk.Button(btns, text="Apply", default='active', command=do_apply).pack(
            side='right', padx=(0, 6))

    def _open_figure_layout(self):
        """Open the figure-layout dialog (small-multiple publication figure
        builder) seeded from the current plot selection."""
        samples = self._selected_samples()
        if not samples:
            messagebox.showinfo(
                "Quick figure grid",
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
        """The side panel's "Layout (beta)" button: the FlowJo-style layout
        editor (a page of plots, stats and text, re-applied per sample). One
        window per editor; a second click raises it."""
        win = getattr(self, '_layout_window', None)
        try:
            if win is not None and win.winfo_exists():
                win.deiconify()
                win.lift()
                return
        except tk.TclError:
            pass
        from .ui_layout_editor import LayoutEditorWindow
        self._layout_window = LayoutEditorWindow(self)

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
        name = self._active_sample
        sample = self._samples[name]

        def _on_apply(channels, matrix, targets=None):
            names = [n for n in (targets or [name]) if n in self._samples]
            if len(names) <= 1:
                self._comp_apply_sample(names[0] if names else name,
                                        channels, matrix)
                return
            done, failed = [], []
            for n in names:
                ok, msg = self._comp_apply_sample(n, channels, matrix,
                                                  report=False)
                (done if ok else failed).append((n, msg))
            msg = (f"Applied {len(channels)}×{len(channels)} matrix to "
                   f"{len(done)} sample(s) in this batch")
            if done:
                msg += f" ({', '.join(n for n, _ in done)})"
            msg += '.'
            if failed:
                msg += (f" NOT applied to {len(failed)}: "
                        + '; '.join(f"{n}: {m}" for n, m in failed))
            notes = [m if not m.startswith('(') else f"'{n}' {m[1:-1]}."
                     for n, m in done if m]
            if notes:
                msg += ' ' + ' '.join(dict.fromkeys(notes))
            self.status_var.set(msg)
            self._schedule_replot(0)

        def _batch(channels):
            from .acquisition import batch_of
            meta = {n: getattr(smp, 'metadata', None) or {}
                    for n, smp in self._samples.items()}
            return batch_of(name, meta, channels)

        CompensationEditorWindow(self, sample=sample, on_apply=_on_apply,
                                 batch=_batch)

    def _comp_apply_sample(self, name, channels, matrix, report=True):
        """Compensate one sample with ``matrix`` on ``channels``. Returns
        ``(applied, note)``: why not when it wasn't, else anything the user
        should know (events left unknown, results computed before). With
        ``report`` the status bar says so and the plot redraws, as for one
        Apply."""
        from .pipeline import CompensationError, UnknownScaleError
        sample = self._samples[name]

        def said(ok, note):
            if report:
                # A note in brackets says how ("(rebuilt from x.fcs)").
                how = note if note.startswith('(') else ''
                rest = '' if how else note
                self.status_var.set(
                    (f"Applied {len(channels)}×{len(channels)} matrix to "
                     f"'{name}'{' ' + how if how else ''}. {rest}"
                     if ok else note).strip())
                if ok:
                    self._schedule_replot(0)
            return ok, note

        path = str(getattr(sample, 'path', '') or '')
        if not sample.comp_channels and not path.lower().endswith('.fcs'):
            # Built from a CSV (a restored session sidecar, Load CSV)
            # with no compensation record: its values may already be
            # compensated, with no record of the matrix, so un-mixing
            # them again would compound it. A session sidecar that names
            # its raw FCS is rebuilt from that file, as before.
            if getattr(sample, 'source_fcs', None):
                return said(*self._comp_apply_by_reload(
                    name, sample, channels, matrix))
            return said(False,
                f"Cannot re-compensate '{name}': it was built from "
                f"{os.path.basename(path) or 'a data table'}, which does "
                "not record the compensation already in its values. Load "
                "it from its FCS file to compensate it.")
        try:
            # Re-compensate the sample's own events in place. Apply used
            # to swap in a fresh load of the FCS, which dropped every
            # computed column (clusters, embeddings, U: abundances, _pos
            # flags), so their populations selected nothing, gates on
            # them everything, and a downsampled sample came back whole.
            n_unknown = sample.recompensate(matrix, list(channels))
        except UnknownScaleError as exc:
            return said(False, f"Cannot re-compensate '{name}': {exc}")
        except (ValueError, np.linalg.LinAlgError) as exc:
            # No channel matched, or the matrix is singular: the sample
            # keeps its current compensation, and must not be reported
            # as recompensated.
            return said(False,
                f"Compensation NOT applied to '{name}' — {exc}. The sample "
                "keeps its current compensation.")
        except CompensationError as exc:
            return said(False, f"Compensation failed for '{name}': {exc}")
        # The data object changed: drop everything cached against it.
        self._data_changed([n for n, smp in self._samples.items()
                            if smp is sample])
        self._undo_barrier(f"the compensation of '{name}'")
        note = ''
        if n_unknown:
            note += (f"{n_unknown} event(s) in '{name}' have a log-scale "
                     "value <= 0, which has no linear value, so their "
                     "re-compensated values that depend on it are unknown "
                     "(empty). ")
        stale = self._computed_results(sample)
        if stale:
            note += ("Kept as they were, but computed before this "
                     f"compensation change: {', '.join(stale)}. Re-run "
                     "them to update.")
        return said(True, note.strip())

    def _comp_apply_by_reload(self, name, sample, channels, matrix):
        """Compensate ``sample`` by rebuilding it from its raw FCS
        (``source_fcs``): for a sample restored from a session CSV that
        carries no record of the compensation already in its values, so it
        cannot be re-compensated in place. Computed columns are not in the
        FCS and do not come back. Returns ``(applied, note)``."""
        try:
            from .pipeline import FlowSample
            fresh = FlowSample(sample.source_fcs)
            # The same event set as the sample being replaced: QC only if
            # it ran when the sample was loaded (whatever Preferences says
            # now), so the gates keep counting the same events.
            if getattr(sample, 'qc_applied', False):
                fresh.run_qc()
            fresh.manual_compensate(matrix, list(channels))
            if not fresh.is_compensated():
                return False, (
                    f"Compensation NOT applied to '{name}' — the matrix is "
                    "singular or names none of this sample's channels (see "
                    "log).")
            fresh.apply_transform()
            # A channel the Transform editor put on another method must
            # come back on THAT method.
            self._conform_sample_transforms(fresh)
            if getattr(sample, 'channel_labels', None):
                fresh.set_labels(dict(sample.channel_labels))
            sample.data = fresh.data
            sample.data_transforms = dict(fresh.data_transforms)
            sample.comp_matrix   = getattr(fresh, 'comp_matrix', None)
            sample.comp_channels = list(getattr(fresh, 'comp_channels', []))
            sample.compensation_source = fresh.compensation_source
            # The fresh data has no cell_cycle / flowsom columns, so the
            # models that labelled them go too: left on, the Cell cycle
            # window and SOM tree would open on columns that are not there.
            sample.cell_cycle_result = None
            sample.flowsom_result = None
            self._data_changed([n for n, smp in self._samples.items()
                                if smp is sample])
            self._undo_barrier(f"the compensation of '{name}'")
            return True, (f"(rebuilt from "
                          f"{os.path.basename(str(sample.source_fcs))})")
        except Exception as exc:
            return False, f"Compensation failed for '{name}': {exc}"

    # Label columns the clustering methods write, as the user knows them.
    _CLUSTER_COLUMN_NAMES = {'cluster': 'PhenoGraph clusters',
                             'leiden': 'Leiden clusters',
                             'flowsom': 'FlowSOM nodes',
                             'flowsom_meta': 'FlowSOM metaclusters'}

    def _computed_results(self, sample):
        """What analysis added to ``sample``'s data, as the user knows it:
        each clustering, each embedding, then the other added columns by name
        (U: abundances, _pos flags, ...). An FCS-loaded sample's own channels
        are not in it."""
        own = set(getattr(sample, 'channel_names', None) or [])
        out = []
        for c in sample.data.columns:
            if c in own:
                continue
            s = str(c)
            emb = next((p for p in ('UMAP', 'TSNE', 'TRIMAP', 'PACMAP', 'PHATE')
                        if s.upper() == p + '1' or s.upper() == p + '2'), None)
            if emb is not None:
                item = f'{emb} embedding ({emb}1/{emb}2)'
            else:
                item = self._CLUSTER_COLUMN_NAMES.get(s)
                item = f'{item} ({s})' if item else s
            if item not in out:
                out.append(item)
        return out
