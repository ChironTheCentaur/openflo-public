"""Automatic gating: singlet, GMM, and threshold strategies.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import numpy as np

from .editor_base import EditorMixin


class AutoGateMixin(EditorMixin):
    """One-click automatic gate placement (singlet / GMM / threshold) across the loaded samples."""

    def _find_area_height_channels(self):
        """Best-guess (area, height) scatter channel pair for a singlet gate.
        Prefers FSC-A/FSC-H, then SSC-A/SSC-H. Returns (area, height) or
        (None, None)."""
        chans = list(self._channels)
        up = {c: c.upper() for c in chans}
        for stem in ('FSC', 'SSC'):
            area = next((c for c in chans
                         if stem in up[c] and '-A' in up[c]), None)
            height = next((c for c in chans
                           if stem in up[c] and '-H' in up[c]), None)
            if area and height:
                return area, height
        return None, None

    def _auto_gate(self):
        """Open the auto-gate dialog: well-posed, reviewable gate proposals
        (singlet ratio-band, BIC-selected GMM ellipses, or 1-D valley/Otsu
        threshold), each reported with a quality score. Proposals are added
        as ordinary undoable gates for the user to accept / tweak / delete."""
        from .ui_autogate import AutoGateDialog
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        area, height = self._find_area_height_channels()
        AutoGateDialog(self, has_singlet=bool(area and height),
                       area=area, height=height,
                       cur_x=self._fmt_channel(x) if x else '',
                       cur_y=self._fmt_channel(y) if y else '',
                       on_apply=self._run_auto_gate)

    def _run_auto_gate(self, opts):
        """Execute the chosen auto-gate method against the active sample and
        add the proposal(s). ``opts`` comes from AutoGateDialog."""
        name = self._active_sample
        if name is None or name not in self._samples:
            self.status_var.set("Select a sample first.")
            return
        method = opts.get('method')
        if method == 'singlet':
            self._auto_gate_singlet(name, opts)
        elif method == 'gmm':
            self._auto_gate_gmm(name, opts)
        elif method == 'threshold':
            self._auto_gate_threshold(name)
        self._schedule_replot(0)

    def _auto_gate_singlet(self, name, opts):
        from .pipeline import auto_singlet_gate
        area, height = opts.get('area'), opts.get('height')
        if not (area and height):
            self.status_var.set("No FSC-A/FSC-H pair found for a singlet gate.")
            return
        df = self._get_df(name, area, height)
        if area not in df.columns or height not in df.columns:
            self.status_var.set("Active sample lacks the FSC-A/FSC-H channels.")
            return
        verts, q = auto_singlet_gate(
            np.asarray(df[area].values, dtype=float),
            np.asarray(df[height].values, dtype=float),
            k=float(opts.get('k', 3.0)))
        if not verts or q is None:
            self.status_var.set(
                "Singlet gate: ratio band undefined (too little spread/data).")
            return
        self._add_gate_multi({'kind': 'polygon', 'x_channel': area,
                              'y_channel': height, 'vertices': verts,
                              'name': 'Singlets'}, audit=False)
        # Switch the view so the user sees what was proposed.
        self.x_combo.set(self._fmt_channel(area))
        self.y_combo.set(self._fmt_channel(height))
        if self.mode_var.get() == 'histogram':
            self.mode_var.set('pseudocolor')
        trust = ('clean' if q['frac_kept'] > 0.8 and q['ratio_cv'] < 0.12
                 else 'REVIEW')
        self._audit('autogate.singlet', sample=name, area=area, height=height,
                    k=float(opts.get('k', 3.0)),
                    frac_kept=round(q['frac_kept'], 4),
                    ratio_cv=round(q['ratio_cv'], 4), trust=trust)
        self.status_var.set(
            f"Singlet gate [{trust}]: keeps {q['frac_kept'] * 100:.1f}% "
            f"(ratio CV {q['ratio_cv']:.3f}). Drag vertices to adjust.")

    def _auto_gate_gmm(self, name, opts):
        from .pipeline import gmm_ellipse_gates
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        if not x or not y:
            self.status_var.set("Pick X and Y channels for a 2-D auto-gate.")
            return
        df = self._get_df(name, x, y)
        if x not in df.columns or y not in df.columns:
            self.status_var.set("Active sample lacks those channels.")
            return
        proposals = gmm_ellipse_gates(
            np.asarray(df[x].values, dtype=float),
            np.asarray(df[y].values, dtype=float),
            max_components=int(opts.get('max_components', 6)),
            coverage=float(opts.get('coverage', 0.90)),
            min_weight=float(opts.get('min_weight', 0.02)))
        if not proposals:
            self.status_var.set("Auto-gate: no populations found (too little "
                                "data or no structure).")
            return
        weak = 0
        for i, (gate, info) in enumerate(proposals, 1):
            gate = dict(gate, x_channel=x, y_channel=y, name=f'Pop {i}')
            self._add_gate_multi(gate, audit=False)
            if info.get('separation') is not None and info['separation'] < 2.0:
                weak += 1
        k = proposals[0][1]['n_components']
        note = (f" — {weak} overlap heavily (separation < 2); review those"
                if weak else "")
        self._audit('autogate.gmm', sample=name, x=x, y=y,
                    n_populations=len(proposals), k_bic=k,
                    coverage=float(opts.get('coverage', 0.90)),
                    weak_overlap=weak)
        self.status_var.set(
            f"GMM found {len(proposals)} population(s) of k={k} (BIC). "
            f"Added as ellipse gates{note}.")

    def _auto_gate_threshold(self, name):
        from .pipeline import auto_threshold
        x = self._resolve_channel(self.x_combo.get())
        if not x:
            self.status_var.set("Pick an X channel first.")
            return
        df = self._get_df(name, x)
        if x not in df.columns:
            self.status_var.set("Active sample lacks that channel.")
            return
        thr = auto_threshold(np.asarray(df[x].values, dtype=float))
        if thr is None:
            self.status_var.set("Auto-gate: not enough data to split.")
            return
        self._add_gate_multi({'kind': 'threshold', 'channel': x,
                              'value': float(thr)}, audit=False)
        self._audit('autogate.threshold', sample=name, channel=x,
                    value=float(thr))
        self.status_var.set(
            f"Auto threshold on {self._fmt_channel(x)} = {thr:.3g}.")
