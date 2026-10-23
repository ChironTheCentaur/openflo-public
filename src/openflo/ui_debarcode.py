"""Debarcode window: split a pooled, barcoded sample back into its samples
(openflo.debarcode).

Pick the pooled sample and the population to debarcode (live single cells,
say), load the barcode key, map its columns onto the data's channels, and
set the separation cutoff. Preview shows each sample's yield and how the
yields fall as the cutoff rises; Apply adds each barcoded sample as a
population under the chosen one, and Save each as FCS writes them out.

The per-event assignment is kept in a ``debarcode`` text column of the
sample's data and each population is a 'category' gate on it -- the same
machinery cell-cycle phases use -- so they show in Statistics and survive a
session save and reload.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from . import debarcode as dbc
from .errors import note
from .tool_window import figure_panel
from .ui_errors import show_error
from .ui_tips import tip

COLUMN = 'debarcode'          # the sample's per-event assignment column
NOT_DEBARCODED = ''           # events outside the debarcoded population
STEP = 'debarcode'            # its result stamp and audit action
ALL_EVENTS = 'All events'
COLS = (('sample', 'Sample'), ('code', 'Code'), ('events', 'Events'),
        ('pct', '% of parent'))


def population_name(sample):
    """The tree name of one barcoded sample's population."""
    return f"{sample} (barcode)"


def _is_ours(g):
    return g.get('kind') == 'category' and g.get('channel') == COLUMN


def apply_result(ed, name, parent_gid, mask, result, add_unassigned=True,
                 inputs=()):
    """Write ``result`` (for the events in ``mask``) into sample ``name``'s
    ``debarcode`` column and add one population per key sample under
    ``parent_gid`` (None: at the top), plus 'Unassigned' when asked.

    A re-run updates the populations in place (same value, new parent) and
    removes the ones that no longer apply, unless another gate uses them.
    Returns ``(gate ids in key order, populations kept because in use)``."""
    from .pipeline import GATE_PALETTE, NOISE_CLUSTER_COLOR
    from .result_stamp import stamp_columns
    s = ed._samples[name]
    col = np.full(len(s.data), NOT_DEBARCODED, dtype=object)
    col[np.asarray(mask, dtype=bool)] = result.labels()
    ed._checkpoint()
    s.data[COLUMN] = col
    stamp_columns(s, [COLUMN], STEP, list(inputs))
    ed._data_changed(name)
    gates = ed._sample_gates.setdefault(name, {})
    order = ed._sample_gate_order.setdefault(name, [])
    wanted = list(result.key.samples) + ([dbc.UNASSIGNED] if add_unassigned
                                         else [])
    have = {g.get('value'): gid for gid, g in gates.items() if _is_ours(g)}
    out = []
    for i, smp in enumerate(wanted):
        gid = have.get(smp)
        if gid is None:
            gid = ed._next_gate_id_for(name)
            gates[gid] = {
                'kind': 'category', 'channel': COLUMN, 'value': smp,
                'name': population_name(smp),
                'color': (NOISE_CLUSTER_COLOR if smp == dbc.UNASSIGNED
                          else GATE_PALETTE[i % len(GATE_PALETTE)]),
                'enabled': False}
            order.append(gid)
        gates[gid]['parent_id'] = parent_gid
        out.append(gid)
    # A sample dropped from the key, or Unassigned no longer wanted.
    stale = [gid for gid, g in gates.items()
             if _is_ours(g) and g.get('value') not in wanted]
    kept = []
    for gid in stale:
        users = [o for o, g in gates.items() if g.get('parent_id') == gid
                 or gid in (g.get('operands') or [])]
        if users:
            kept.append(gates[gid].get('name', gid))
        else:
            ed._purge_gate_subtree(name, gid)
    ed._refresh_gate_list()
    return out, kept


class DebarcodeWindow(tk.Toplevel):
    """Sample, parent population, key and channel mapping, cutoff; a yield
    table and a yield-vs-cutoff plot; Apply and Save each as FCS."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Debarcode")
        self.geometry("900x700")
        self.minsize(640, 480)
        self._key: dbc.BarcodeKey | None = None
        self._key_path = ''
        self._result: dbc.DebarcodeResult | None = None
        self._mask = None
        self._parent = None
        self._map_vars: list[tk.StringVar] = []

        top = ttk.Frame(self, padding=6)
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Sample:").pack(side='left')
        names = [n for n in editor._sample_order if n in editor._samples]
        self.sample_var = tk.StringVar(
            master=self, value=(editor._active_sample
                                if editor._active_sample in names
                                else (names[0] if names else '')))
        _c = ttk.Combobox(top, textvariable=self.sample_var, width=22,
                          state='readonly', values=names)
        _c.pack(side='left', padx=(2, 8))
        _c.bind('<<ComboboxSelected>>', lambda e: self._sample_changed())
        tip(editor, _c, "The pooled, barcoded sample to split.")
        ttk.Label(top, text="Population:").pack(side='left')
        self.pop_var = tk.StringVar(master=self, value=ALL_EVENTS)
        self.pop_combo = ttk.Combobox(top, textvariable=self.pop_var,
                                      width=28, state='readonly')
        self.pop_combo.pack(side='left', padx=(2, 8))
        tip(editor, self.pop_combo,
            "Debarcode only these events, usually live single cells; the "
            "barcoded samples are added under this population.")
        _b = ttk.Button(top, text="Load key…", command=self._load_key)
        _b.pack(side='left')
        tip(editor, _b, "A CSV: a header row, then one row per sample: its "
                        "name, then 0 or 1 for each barcode channel (1 = "
                        "positive). Each sample needs its own code.")

        self.key_label = ttk.Label(self, text="No key loaded.",
                                   padding=(10, 0, 10, 2))
        self.key_label.pack(side='top', fill='x')
        self.map_frame = ttk.Frame(self, padding=(10, 0, 10, 4))
        self.map_frame.pack(side='top', fill='x')

        opts = ttk.Frame(self, padding=(6, 0, 6, 6))
        opts.pack(side='top', fill='x')
        ttk.Label(opts, text="Separation cutoff:").pack(side='left')
        self.cutoff_var = tk.StringVar(master=self,
                                       value=f"{dbc.DEFAULT_CUTOFF:g}")
        _s = ttk.Spinbox(opts, textvariable=self.cutoff_var, from_=0.0,
                         to=1.0, increment=0.05, width=6,
                         command=self._cutoff_changed)
        _s.pack(side='left', padx=(2, 8))
        _s.bind('<Return>', lambda e: self._cutoff_changed())
        _s.bind('<FocusOut>', lambda e: self._cutoff_changed())
        tip(editor, _s,
            "How cleanly an event's barcode channels must split into "
            "positive and negative (0 to 1, on each channel's rescaled "
            "range). Raising it keeps only clearly barcoded cells: purer "
            "samples, fewer events. Lowering it keeps more events but lets "
            "in doublets of two barcoded cells and misassigned cells. "
            f"{dbc.DEFAULT_CUTOFF:g} is a common starting point; check the "
            "plot for where the yields start to fall steeply.")
        self.second_var = tk.BooleanVar(master=self, value=True)
        _k = ttk.Checkbutton(opts, text="Rescale per sample",
                             variable=self.second_var)
        _k.pack(side='left', padx=(0, 8))
        tip(editor, _k,
            "A second pass (Zunder et al., 2015): rescale each sample's "
            "barcode channels by its own positive cells before assigning "
            "again, so a sample that stained dimmer separates as well as "
            "the others. Leave on unless the samples stained evenly.")
        self.unassigned_var = tk.BooleanVar(master=self, value=True)
        _k = ttk.Checkbutton(opts, text="Add unassigned events as a population",
                             variable=self.unassigned_var)
        _k.pack(side='left', padx=(0, 8))
        tip(editor, _k, "Also add the events no sample took (below the "
                        "cutoff, or a code no sample has).")
        _b = ttk.Button(opts, text="Preview", command=self._preview)
        _b.pack(side='left')
        tip(editor, _b, "Assign the events and show each sample's yield "
                        "without changing anything.")

        self.status_var = tk.StringVar(master=self, value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=860, justify='left',
                  padding=(10, 0, 10, 4)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(side='bottom', fill='x')
        _b = ttk.Button(btns, text="Apply", command=self._apply)
        _b.pack(side='left')
        tip(editor, _b, "Add each barcoded sample as a population under the "
                        "chosen one (and keep them in the session).")
        _b = ttk.Button(btns, text="Save each as FCS…",
                        command=self._save_fcs)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Apply, then write each barcoded sample to its own "
                        "FCS file in a folder (compensated linear values).")
        ttk.Button(btns, text="Close", command=self.destroy).pack(
            side='right')

        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(side='top', fill='both', expand=True)
        holder = ttk.Frame(body)
        holder.pack(side='left', fill='y')
        self.tv = ttk.Treeview(holder, show='headings', height=14,
                               columns=[c for c, _ in COLS])
        for c, t in COLS:
            self.tv.heading(c, text=t)
            self.tv.column(c, width=140 if c == 'sample' else 80,
                           anchor='w' if c in ('sample', 'code') else 'e')
        sb = ttk.Scrollbar(holder, orient='vertical', command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side='left', fill='y')
        sb.pack(side='left', fill='y')
        plot = ttk.Frame(body)
        plot.pack(side='left', fill='both', expand=True, padx=(8, 0))
        _f, self.fig, self.canvas = figure_panel(plot, figsize=(4.5, 4))
        self._sample_changed()

    # ── choices ────────────────────────────────────────────────────────────
    def _population_gids(self, name):
        ed = self.editor
        gates = ed._sample_gates.get(name, {})
        transforms = ed._display_transforms(name)
        out = {}
        for gid in ed._sample_gate_order.get(name, list(gates)):
            g = gates.get(gid)
            if g is None or g.get('kind') == 'group' or _is_ours(g):
                continue
            out.setdefault(ed._population_path(gates, gid,
                                               transforms=transforms), gid)
        return out

    def _sample_changed(self):
        name = self.sample_var.get()
        pops = self._population_gids(name) if name else {}
        self.pop_combo['values'] = [ALL_EVENTS] + list(pops)
        # Start on the population selected in the tree, when it is one.
        sel = None
        if name == self.editor._active_sample:
            try:
                sel = self.editor._selected_gate_id()
            except Exception as exc:              # noqa: BLE001
                note("reading the selected population", exc)
        path = next((p for p, g in pops.items() if g == sel), None)
        self.pop_var.set(path or ALL_EVENTS)
        self._result = None
        self._remap()

    def _sample(self):
        return self.editor._samples.get(self.sample_var.get())

    def _channel_choices(self):
        s = self._sample()
        if s is None:
            return []
        import pandas.api.types as pdt

        from .pipeline import _is_time
        return [c for c in s.data.columns if c != COLUMN
                and not _is_time(str(c))
                and pdt.is_numeric_dtype(s.data[c])
                and not pdt.is_bool_dtype(s.data[c])]

    def _guess_channel(self, key_col, chans):
        """The data channel a key column names: by detector, then by marker
        label (case-insensitive)."""
        if key_col in chans:
            return key_col
        s = self._sample()
        labels = dict(self.editor._channel_labels)
        labels.update(getattr(s, 'channel_labels', {}) or {})
        low = key_col.strip().lower()
        for c in chans:
            if c.lower() == low or str(labels.get(c, '')).lower() == low:
                return c
        return ''

    # ── key ────────────────────────────────────────────────────────────────
    def _load_key(self):
        path = filedialog.askopenfilename(
            parent=self, title="Load barcode key",
            filetypes=[('CSV', '*.csv'), ('Text', '*.txt *.tsv'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            key = dbc.read_key(path)
        except dbc.BarcodeKeyError as exc:
            self.status_var.set(f"Key not loaded: {exc}")
            return
        self._key, self._key_path = key, path
        self._result = None
        self._remap()
        self.status_var.set(
            f"Key loaded: {len(key.samples)} samples over "
            f"{len(key.channels)} barcode channels. Check the channels, "
            "then Preview.")

    def _remap(self):
        """One row per key column: the data channel it is read from."""
        for w in self.map_frame.winfo_children():
            w.destroy()
        self._map_vars = []
        key = self._key
        if key is None:
            self.key_label.configure(text="No key loaded.")
            return
        k = key.uniform_k
        scheme = (f"{k}-of-{len(key.channels)}" if k is not None
                  else "mixed numbers of positives")
        self.key_label.configure(
            text=f"Key {os.path.basename(self._key_path)}: "
                 f"{len(key.samples)} samples, {scheme}. "
                 "Barcode channels (key column → data channel):")
        chans = self._channel_choices()
        disp = [self.editor._fmt_channel(c) for c in chans]
        per_row = 3
        for i, kc in enumerate(key.channels):
            r, c = divmod(i, per_row)
            ttk.Label(self.map_frame, text=f"{kc} →").grid(
                row=r, column=2 * c, sticky='e', padx=(0 if c == 0 else 10, 2))
            guess = self._guess_channel(kc, chans)
            var = tk.StringVar(master=self, value=(
                self.editor._fmt_channel(guess) if guess else ''))
            cb = ttk.Combobox(self.map_frame, textvariable=var, width=18,
                              state='readonly', values=disp)
            cb.grid(row=r, column=2 * c + 1, sticky='w', pady=1)
            tip(self.editor, cb, f"The data channel that carries key column "
                                 f"'{kc}'.")
            self._map_vars.append(var)

    def _mapped_channels(self):
        """Data channel per key column, or raise BarcodeKeyError."""
        chans = self._channel_choices()
        by_disp = {self.editor._fmt_channel(c): c for c in chans}
        assert self._key is not None
        out = []
        for kc, var in zip(self._key.channels, self._map_vars, strict=True):
            ch = by_disp.get(var.get())
            if ch is None:
                raise dbc.BarcodeKeyError(
                    f"Pick the data channel for key column '{kc}'.")
            out.append(ch)
        if len(set(out)) != len(out):
            raise dbc.BarcodeKeyError(
                "Two key columns are mapped to the same data channel.")
        return out

    def _cutoff(self):
        try:
            v = float(str(self.cutoff_var.get()).strip())
        except ValueError:
            return None
        return v if np.isfinite(v) and 0.0 <= v <= 1.0 else None

    # ── compute ────────────────────────────────────────────────────────────
    def _compute(self):
        """Debarcode the chosen events; the result, or None (status says
        why)."""
        from .pipeline import cumulative_gate_mask
        ed = self.editor
        s = self._sample()
        if s is None:
            self.status_var.set("Load the pooled sample first.")
            return None
        if self._key is None:
            self.status_var.set("Load a barcode key first.")
            return None
        cutoff = self._cutoff()
        if cutoff is None:
            self.status_var.set("The separation cutoff must be a number "
                                "from 0 to 1.")
            return None
        try:
            chans = self._mapped_channels()
        except dbc.BarcodeKeyError as exc:
            self.status_var.set(str(exc))
            return None
        name = self.sample_var.get()
        pop = self.pop_var.get() or ALL_EVENTS
        if pop == ALL_EVENTS:
            mask = np.ones(len(s.data), dtype=bool)
        else:
            gid = self._population_gids(name).get(pop)
            if gid is None:
                self.status_var.set(f"'{pop}' is no longer in {name}.")
                return None
            mask = np.asarray(cumulative_gate_mask(
                ed._sample_gates.get(name, {}), gid, s.data), dtype=bool)
        if not mask.any():
            self.status_var.set("The chosen population has no events.")
            return None
        key = self._key.with_channels(chans)
        x = s.data.loc[mask, chans]
        try:
            res = dbc.debarcode(x, key, cutoff=cutoff,
                                second_pass=bool(self.second_var.get()))
        except ValueError as exc:
            self.status_var.set(str(exc))
            return None
        self._mask = mask
        self._result = res
        self._parent = None if pop == ALL_EVENTS else \
            self._population_gids(name).get(pop)
        return res

    def _preview(self):
        if self._compute() is not None:
            self._show()

    def _cutoff_changed(self):
        cutoff = self._cutoff()
        if self._result is None or cutoff is None:
            return
        self._result = self._result.with_cutoff(cutoff)
        self._show()

    def _show(self):
        res = self._result
        if res is None:
            return
        y = res.yields()
        total = max(len(res.match), 1)
        self.tv.delete(*self.tv.get_children())
        for i, smp in enumerate(res.key.samples):
            self.tv.insert('', 'end', values=(
                smp, res.key.code_text(i), f"{y[smp]:,}",
                f"{100.0 * y[smp] / total:.1f}"))
        un = y[dbc.UNASSIGNED]
        self.tv.insert('', 'end', values=(
            dbc.UNASSIGNED, '', f"{un:,}", f"{100.0 * un / total:.1f}"))
        self._draw()
        low = [s for s in res.key.samples if y[s] == 0]
        msg = (f"{total - un:,} of {total:,} events assigned "
               f"({100.0 * (total - un) / total:.1f}%) at cutoff "
               f"{res.cutoff:g}.")
        if low:
            msg += "  No events for: " + ", ".join(low) + "."
        self.status_var.set(msg)

    def _draw(self):
        res = self._result
        fig = self.fig
        fig.clear()
        ax = fig.add_subplot(111)
        if res is not None:
            curve = res.yield_curve()
            cut = curve.index.to_numpy(dtype=float)
            total = max(len(res.match), 1)
            for smp in res.key.samples:
                ax.plot(cut, curve[smp].to_numpy() / total * 100.0,
                        lw=0.8, color='0.6')
            assigned = total - curve[dbc.UNASSIGNED].to_numpy()
            ax.plot(cut, assigned / total * 100.0, lw=2.0, color='#1f77b4',
                    label='All samples')
            ax.axvline(res.cutoff, color='#d62728', ls='--', lw=1)
            ax.set_xlabel("Separation cutoff")
            ax.set_ylabel("% of events assigned")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        try:
            from .theme import _dialog_dark_on, _theme_figure_dark
            if _dialog_dark_on(self):
                _theme_figure_dark(fig)
        except Exception:  # noqa: BLE001 - dialog theming is cosmetic
            pass
        self.canvas.draw_idle()

    # ── apply / export ─────────────────────────────────────────────────────
    def _apply(self):
        """Add the populations; their gate ids, or None."""
        res = self._compute()
        if res is None:
            return None
        self._show()
        ed = self.editor
        name = self.sample_var.get()
        assert self._key is not None and self._mask is not None
        chans = list(res.key.channels)
        try:
            gids, kept = apply_result(
                ed, name, self._parent, self._mask, res,
                add_unassigned=bool(self.unassigned_var.get()),
                inputs=[self._parent or 'all'] + chans
                + [f'cutoff={res.cutoff:g}'])
        except Exception as exc:                       # noqa: BLE001
            show_error(self, "Debarcode",
                       "The barcoded samples could not be added.", exc)
            return None
        y = res.yields()
        try:
            ed._audit(STEP, sample=name, channels=chans,
                      population=self.pop_var.get(), cutoff=res.cutoff,
                      second_pass=res.second_pass,
                      n_samples=len(res.key.samples),
                      n_assigned=int(len(res.match) - y[dbc.UNASSIGNED]),
                      n_events=int(len(res.match)))
        except Exception as exc:                       # noqa: BLE001
            note("recording debarcoding in the audit trail", exc)
        msg = (f"Added {len(res.key.samples)} barcoded samples to {name}"
               f" under {self.pop_var.get()}; "
               f"{len(res.match) - y[dbc.UNASSIGNED]:,} of "
               f"{len(res.match):,} events assigned.")
        if kept:
            msg += ("  Kept (other gates use them): " + ", ".join(kept)
                    + ".")
        self.status_var.set(msg)
        ed.status_var.set(msg)
        self._sample_changed_keep_result()
        return gids

    def _sample_changed_keep_result(self):
        """Refresh the population list after Apply, keeping the result."""
        res, mask, pop = self._result, self._mask, self.pop_var.get()
        name = self.sample_var.get()
        self.pop_combo['values'] = ([ALL_EVENTS]
                                    + list(self._population_gids(name)))
        self.pop_var.set(pop)
        self._result, self._mask = res, mask

    def _save_fcs(self):
        """Apply, then write each barcoded sample's events to its own FCS in
        a folder, with the existing population export."""
        out = filedialog.askdirectory(
            parent=self, title="Save each barcoded sample (FCS) to…")
        if not out:
            return
        gids = self._apply()
        if not gids:
            return
        import copy

        from .export_stamp import corrections_for
        from .fcs_export import (
            export_populations,
            population_export_frame,
            source_keywords,
        )
        from .pipeline import cumulative_gate_mask
        ed = self.editor
        name = self.sample_var.get()
        s = ed._samples[name]
        gates = copy.deepcopy(ed._sample_gates.get(name, {}))
        fixes = corrections_for(ed)
        labels = dict(ed._channel_labels)
        labels.update(getattr(s, 'channel_labels', {}) or {})

        def work(job):
            pops, ranges, dropped = {}, {}, []
            for i, gid in enumerate(gids):
                smp = gates[gid].get('value')
                job.progress(i, len(gids), f"Saving {smp}…")
                m = np.asarray(cumulative_gate_mask(gates, gid, s.data),
                               dtype=bool)
                if not m.any():
                    continue
                pops[str(smp)], ranges[str(smp)], lost = (
                    population_export_frame(s, m))
                dropped += [c for c in lost
                            if c not in dropped and c != COLUMN]
            if not pops:
                return [], dropped
            keywords, voltages = source_keywords(s)
            return export_populations(
                pops, out, channel_labels=labels, population_ranges=ranges,
                keywords=keywords, voltages=voltages,
                corrections=fixes), dropped

        def done(res):
            paths, dropped = res
            if not paths:
                self.status_var.set("No barcoded sample has events to save.")
                return
            ed._audit('population.export_fcs', path=out, files=list(paths),
                      samples=[name])
            msg = (f"Saved {len(paths)} barcoded sample(s) as FCS to {out} "
                   "(compensated linear values).")
            if dropped:
                msg += ("  Not exported (no FCS values): "
                        + ", ".join(dropped) + ".")
            self.status_var.set(msg)
            ed.status_var.set(msg)

        def failed(exc):
            show_error(self, "Save each as FCS",
                       "The FCS files could not be written.", exc)

        ed.run_job(work, on_done=done, on_error=failed,
                   text="Saving barcoded samples…")
