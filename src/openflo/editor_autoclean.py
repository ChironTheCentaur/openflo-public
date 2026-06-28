"""Auto-clean gate creation, parameter editing, and per-method masks.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from .editor_base import EditorMixin


class AutoCleanMixin(EditorMixin):
    """Debris/viability/bead auto-clean gates: build, edit params, compute masks and removed-event counts."""

    def _create_autoclean_gate(self):
        """Auto-clean button: add an 'autocleaned sample' gate to EVERY selected
        sample (or the active sample if the selection covers none). It's a recipe
        gate (a group of toggleable cleaning methods — debris / doublets /
        margin / flow-rate / drift), not fixed geometry: each method recomputes
        from the sample's own data. Rendered as a collapsed group; build
        downstream gates under it to gate on cleaned events. One group per
        sample; samples that already have one are skipped."""
        targets = self._selected_sample_names()
        if not targets:
            self.status_var.set("Load and select a sample first.")
            return
        from .pipeline import default_autoclean_methods
        saved_active = self._active_sample
        added, skipped, bead_name = [], [], None
        for name in targets:
            if name not in self._samples:
                continue
            gates = self._sample_gates.get(name, {})
            if any(g.get('kind') == 'autoclean' for g in gates.values()):
                skipped.append(name)
                continue
            # _add_gate writes into the ACTIVE sample's gate store, so point it
            # at each target in turn (cheap — just rebinds references). The
            # repeated _add_gate checkpoints coalesce into one undo step within
            # this event, so a single Ctrl+Z removes them all.
            self._set_active_sample(name)
            gate = {
                'kind':      'autoclean',
                'name':      'autocleaned sample',
                'parent_id': None,
                'color':     '#808080',     # never drawn (no geometry)
                'open':      False,         # collapsed by default
                'methods':   default_autoclean_methods(),
            }
            bead_name = bead_name or self._autoclean_stamp_refs(name, gate)
            self._add_gate(gate, parent_id=None)
            added.append(name)
        # Leave the originally-active sample active so the view doesn't jump.
        if saved_active is not None and saved_active != self._active_sample:
            self._set_active_sample(saved_active)
        self._refresh_gate_list()
        self._schedule_replot(0)

        if not added:
            self.status_var.set(
                f"Auto-clean is already on {'that sample' if len(skipped) == 1 else f'those {len(skipped)} samples'} "
                "(toggle its methods under it, or Copy it to other samples).")
            return
        beadmsg = (f"Debris cut calibrated to beads ‘{bead_name}’." if bead_name
                   else "No bead file found — debris uses the auto-valley cut.")
        where = added[0] if len(added) == 1 else f"{len(added)} samples"
        extra = f" ({len(skipped)} already had it)" if skipped else ""
        self.status_var.set(
            f"Added 'autocleaned sample' to {where}{extra}. " + beadmsg +
            " Set display to 'filter' to apply it; right-click its "
            "Debris/Dead-cells methods to switch mode or set bead size.")

    def _resolve_bead_anchor(self):
        """Median FSC-A of a size-calibration bead sample among the loaded
        samples — the absolute-size anchor for the debris cut. Scans sample
        names for a bead / rainbow / calibration token; returns
        ``(median_fsc, sample_name)`` or ``(None, None)``. FSC-A is linear in
        the editor's data (only fluorescence channels are transformed), so the
        median is a valid linear size ruler."""
        from .pipeline import _autoclean_find_scatter
        tokens = ('bead', 'rainbow', 'calib')
        for nm in self._sample_order:
            if not any(tok in nm.lower() for tok in tokens):
                continue
            sd = getattr(self._samples.get(nm), 'data', None)
            if sd is None or len(sd) == 0:
                continue
            fsc = _autoclean_find_scatter(sd, 'FSC', '-A')
            if fsc is None:
                continue
            vals = np.asarray(sd[fsc].values, dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 100:
                continue
            return float(np.median(vals)), nm
        return None, None

    def _autoclean_stamp_refs(self, name, gate):
        """Stamp environment-derived references into an auto-clean recipe in
        place: the debris method's bead anchor (``bead_fsc``) and the viability
        method's dye ``channel``. Missing references are left unset, so the
        pure masks degrade gracefully (debris → auto-valley, viability →
        token auto-detect or no-op). Returns the bead sample's name, or None."""
        from .pipeline import find_viability_channel
        sd = getattr(self._samples.get(name), 'data', None)
        labels = getattr(self._samples.get(name), 'channel_labels', {}) or {}
        bead_fsc, bead_name = self._resolve_bead_anchor()
        for m in gate.get('methods') or []:
            key = m.get('key')
            mp = m.setdefault('params', {})
            if key == 'debris':
                if bead_fsc:
                    mp['bead_fsc'] = bead_fsc
                else:
                    mp.pop('bead_fsc', None)
            elif key == 'viability' and sd is not None and not mp.get('channel'):
                ch = find_viability_channel(list(sd.columns), labels)
                if ch:
                    mp['channel'] = ch
        return bead_name

    def _autoclean_method(self, name, gid, key):
        """The (gate, method-dict) pair for method ``key`` under auto-clean
        gate ``gid`` on ``name``; ``(None, None)`` if absent."""
        g = self._sample_gates.get(name, {}).get(gid)
        if g is None or g.get('kind') != 'autoclean':
            return None, None
        for m in g.get('methods') or []:
            if m.get('key') == key:
                return g, m
        return g, None

    def _autoclean_invalidate(self, name, gid):
        """Drop the cached masks/counts for one auto-clean gate and replot."""
        self._ac_cache.pop((name, gid), None)
        self._ac_count_cache.pop((name, gid), None)
        self._ac_method_cache.pop((name, gid), None)
        self._refresh_gate_list()
        self._schedule_replot(0)

    def _autoclean_set_param(self, name, gid, key, **params):
        """Set (or, when a value is None, clear) params on one auto-clean
        method, with an undo checkpoint + cache invalidation."""
        _g, m = self._autoclean_method(name, gid, key)
        if m is None:
            return
        self._checkpoint()
        mp = m.setdefault('params', {})
        for k, v in params.items():
            if v is None:
                mp.pop(k, None)
            else:
                mp[k] = v
        self._autoclean_invalidate(name, gid)

    def _autoclean_set_debris_mode(self, name, gid, mode):
        """Switch the debris method between 'bead' (absolute size) and
        'valley'. Selecting 'bead' re-resolves the bead anchor from the
        loaded samples; if none is found it stays in bead mode but the mask
        falls back to the valley cut until a bead file is added."""
        _g, m = self._autoclean_method(name, gid, 'debris')
        if m is None:
            return
        self._checkpoint()
        mp = m.setdefault('params', {})
        mp['mode'] = mode
        if mode == 'bead':
            bead_fsc, bead_name = self._resolve_bead_anchor()
            if bead_fsc:
                mp['bead_fsc'] = bead_fsc
                self.status_var.set(f"Debris → beads ‘{bead_name}’.")
            else:
                mp.pop('bead_fsc', None)
                self.status_var.set(
                    "Debris → beads, but no bead file is loaded — falls back "
                    "to the auto-valley cut until one is added.")
        else:
            self.status_var.set("Debris → auto-valley (density) cut.")
        self._autoclean_invalidate(name, gid)

    def _autoclean_prompt_float(self, name, gid, key, param, title, prompt,
                                default, minval=0.0):
        from tkinter import simpledialog
        _g, m = self._autoclean_method(name, gid, key)
        if m is None:
            return
        cur = float((m.get('params') or {}).get(param, default))
        val = simpledialog.askfloat(title, prompt, initialvalue=cur,
                                    minvalue=minval, parent=self)
        if val is not None:
            self._autoclean_set_param(name, gid, key, **{param: float(val)})

    def _autoclean_set_viability_channel(self, name, gid, channel):
        """Pin the viability dye channel (``channel=None`` ⇒ auto-detect)."""
        self._autoclean_set_param(name, gid, 'viability',
                                  channel=(channel or None))

    def _edit_autoclean_params(self, name=None, gid=None):
        """Modal dialog to tune an auto-clean gate's per-method parameters and
        enabled flags. Defaults to the active sample's first auto-clean gate.
        Applies write back the recipe, invalidate the mask cache, checkpoint,
        and replot. Parameters are sample-agnostic — they recompute per sample.
        """
        if name is None:
            name = self._active_sample
        gates = self._sample_gates.get(name, {})
        if gid is None:
            gid = next((k for k, g in gates.items()
                        if g.get('kind') == 'autoclean'), None)
        g = gates.get(gid) if gid else None
        if g is None or g.get('kind') != 'autoclean':
            self.status_var.set("No auto-clean gate to edit.")
            return
        methods = g.get('methods') or []

        dlg = tk.Toplevel(self)
        dlg.title("Auto-clean parameters")
        dlg.transient(self)  # type: ignore[arg-type]
        dlg.resizable(False, False)
        ttk.Label(
            dlg, padding=(12, 10, 12, 4), justify='left',
            text=("Tune the cleaning recipe. Values recompute per sample "
                  "(no fixed coordinates).")).pack(anchor='w')
        body = ttk.Frame(dlg, padding=(12, 0))
        body.pack(fill='both', expand=True)

        int_keys = {'n_bins'}
        str_keys = {'mode', 'channel'}     # parsed as text, not numbers
        auto_keys = {'min_fsc', 'channel', 'max_signal'}  # blank ⇒ auto (pop)
        rows = []   # (method, enabled_var, {param_key: (StringVar, is_int)})
        for m in methods:
            key = m.get('key', '')
            sec = ttk.LabelFrame(body, text=m.get('label', key), padding=6)
            sec.pack(fill='x', pady=4)
            en = tk.BooleanVar(value=bool(m.get('enabled', True)))
            ttk.Checkbutton(sec, text="enabled", variable=en).grid(
                row=0, column=0, columnspan=2, sticky='w')
            params = dict(m.get('params') or {})
            if key == 'debris':            # surface the optional manual override
                params.setdefault('min_fsc', None)
            if key == 'viability':         # surface the optional dye channel
                params.setdefault('channel', None)
            pentries = {}
            r = 1
            for pk, pv in params.items():
                ttk.Label(sec, text=f'{pk}:').grid(
                    row=r, column=0, sticky='e', padx=(0, 6), pady=1)
                sv = tk.StringVar(value=('' if pv is None else str(pv)))
                ttk.Entry(sec, textvariable=sv, width=14).grid(
                    row=r, column=1, sticky='w', pady=1)
                pentries[pk] = (sv, pk in int_keys)
                r += 1
            hint = {'debris':    "(mode bead→valley · blank min_fsc/min_um/bead_um = auto)",
                    'viability': "(blank channel = auto-detect viability dye)"}.get(key)
            if hint:
                ttk.Label(sec, text=hint, foreground='grey',
                          font=('TkDefaultFont', 8)).grid(
                    row=r, column=0, columnspan=2, sticky='w')
            rows.append((m, en, pentries))

        err = tk.StringVar(value='')
        ttk.Label(dlg, textvariable=err, foreground='#b00',
                  padding=(12, 0)).pack(anchor='w')

        def _apply():
            staged = []
            for m, en, pentries in rows:
                params = {}
                for pk, (sv, is_int) in pentries.items():
                    raw = sv.get().strip()
                    if raw == '':
                        continue        # blank: leave unchanged (auto)
                    if pk in str_keys:
                        params[pk] = raw
                        continue
                    try:
                        params[pk] = int(float(raw)) if is_int else float(raw)
                    except ValueError:
                        err.set(f"{m.get('label')}: '{pk}' must be a number.")
                        return
                staged.append((m, bool(en.get()), params, pentries))
            self._checkpoint()
            for m, enabled, params, pentries in staged:
                m['enabled'] = enabled
                mp = m.setdefault('params', {})
                # A cleared auto field (min_fsc / channel / max_signal) reverts
                # that method to its automatic detection.
                for ak in auto_keys:
                    if (ak in pentries
                            and pentries[ak][0].get().strip() == ''):
                        mp.pop(ak, None)
                mp.update(params)
            self._ac_cache.pop((name, gid), None)   # recipe changed → recompute
            self._ac_count_cache.pop((name, gid), None)
            self._ac_method_cache.pop((name, gid), None)
            dlg.destroy()
            self._refresh_gate_list()
            self._schedule_replot(0)
            self.status_var.set("Auto-clean parameters updated.")

        btns = ttk.Frame(dlg, padding=12)
        btns.pack(anchor='e')
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side='right')
        ttk.Button(btns, text="Apply", command=_apply).pack(
            side='right', padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: dlg.destroy())
        try:
            dlg.grab_set()
            self.wait_window(dlg)
        except Exception:
            pass

    def _autoclean_overrides(self, name, df):
        """``{gid: df-aligned keep-mask}`` for each auto-clean gate on ``name``,
        or ``None`` when there are none. Each mask is computed once on the FULL
        sample data and cached by (data identity, recipe signature), then
        reindexed to ``df``'s rows — so a chain that nests populations under an
        auto-clean root reuses the cached cleaning instead of recomputing it per
        node and per replot."""
        gates = self._sample_gates.get(name, {})
        ac_gids = [gid for gid, g in gates.items()
                   if g.get('kind') == 'autoclean']
        if not ac_gids:
            return None
        import pandas as pd

        from .pipeline import autoclean_keep_mask, autoclean_methods_signature
        sd = getattr(self._samples.get(name), 'data', None)
        if sd is None:
            return None
        data_id = id(sd)
        out = {}
        for gid in ac_gids:
            g = gates[gid]
            sig = autoclean_methods_signature(g)
            ent = self._ac_cache.get((name, gid))
            if ent is not None and ent[0] == data_id and ent[1] == sig:
                full = ent[2]
            else:
                full = pd.Series(autoclean_keep_mask(g, sd), index=sd.index)
                self._ac_cache[(name, gid)] = (data_id, sig, full)
            # df ⊆ sd (assign/dropna preserve the index); align by label.
            out[gid] = full.reindex(df.index, fill_value=True).to_numpy()
        return out

    def _autoclean_method_masks(self, name):
        """``{method_key: full-data boolean removed-mask}`` for every ENABLED
        cleaning method across the sample's auto-clean gate(s), cached by
        (data identity, recipe signature). Each mask marks the events that
        method removes on its own. ``{}`` when there's no auto-clean gate."""
        gates = self._sample_gates.get(name, {})
        ac_gids = [gid for gid, g in gates.items()
                   if g.get('kind') == 'autoclean']
        if not ac_gids:
            return {}
        import pandas as pd

        from .pipeline import autoclean_keep_mask, autoclean_methods_signature
        sd = getattr(self._samples.get(name), 'data', None)
        if sd is None or len(sd) == 0:
            return {}
        data_id = id(sd)
        out = {}
        for gid in ac_gids:
            g = gates[gid]
            sig = autoclean_methods_signature(g)
            ent = self._ac_method_cache.get((name, gid))
            if ent is not None and ent[0] == data_id and ent[1] == sig:
                masks = ent[2]
            else:
                masks = {}
                for m in g.get('methods', []):
                    if not m.get('enabled', True):
                        continue
                    solo = {'kind': 'autoclean',
                            'methods': [{**m, 'enabled': True}]}
                    rm = ~np.asarray(autoclean_keep_mask(solo, sd), dtype=bool)
                    masks[m.get('key', '')] = pd.Series(rm, index=sd.index)
                self._ac_method_cache[(name, gid)] = (data_id, sig, masks)
            # First enabled method (recipe order) wins an event it removes.
            for key, ser in masks.items():
                out.setdefault(key, ser)
        return out

    def _autoclean_counts(self, name, gid):
        """``(total, total_drop, {key: drop}, {key: reason|None})`` for the
        auto-clean gate ``gid`` on sample ``name`` — computed on the FULL sample
        data and cached by (data identity, recipe signature). ``total_drop`` is
        how many events the enabled recipe removes (union); each ``method_drop``
        is how many that single method removes on its own; ``reasons`` explains
        any method that removed nothing (e.g. "no viability dye detected") so a
        0-drop isn't a silent mystery. ``None`` when the sample isn't loaded or
        the gate isn't auto-clean."""
        g = self._sample_gates.get(name, {}).get(gid)
        if g is None or g.get('kind') != 'autoclean':
            return None
        sd = getattr(self._samples.get(name), 'data', None)
        if sd is None or len(sd) == 0:
            return None
        from .pipeline import (
            autoclean_keep_mask,
            autoclean_method_diagnostic,
            autoclean_methods_signature,
        )
        labels = getattr(self._samples.get(name), 'channel_labels', {}) or {}
        data_id = id(sd)
        sig = autoclean_methods_signature(g)
        ent = self._ac_count_cache.get((name, gid))
        if ent is not None and ent[0] == data_id and ent[1] == sig:
            return (ent[2], ent[3], ent[4], ent[5])
        total = len(sd)
        total_drop = int((~np.asarray(autoclean_keep_mask(g, sd))).sum())
        per_method = {}
        reasons = {}
        for m in g.get('methods', []):
            mkey = m.get('key', '')
            solo = {'kind': 'autoclean',
                    'methods': [{**m, 'enabled': True}]}
            drop = int((~np.asarray(autoclean_keep_mask(solo, sd))).sum())
            per_method[mkey] = drop
            reasons[mkey] = (autoclean_method_diagnostic(
                mkey, sd, m.get('params') or {}, labels) if drop == 0 else None)
        self._ac_count_cache[(name, gid)] = (
            data_id, sig, total, total_drop, per_method, reasons)
        return (total, total_drop, per_method, reasons)
