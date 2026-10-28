"""Tools > Mass cytometry: bead normalization and Gaussian-parameter clean-up.

**Bead normalization** (Finck et al. 2013; the computation is
:mod:`openflo.bead_normalize`). The user picks the bead channels and how
bead events are found (high on every bead channel, or a population drawn
on them); the dialog follows the beads over Time in every chosen sample,
scales each sample's mass channels by the factor that holds the beads level
(on their own, a reference file's or all files' shared baseline), and shows
a chosen bead channel over Time before and after, with the factor. The
change is made as CytoNorm's is (ui_batch_normalize): the loaded values are
replaced, Undo cannot reverse it, and **Undo bead normalization** puts them
back. Unlike CytoNorm it needs no copy of the data: each sample keeps a
small record (``sample.bead_norm``: the factor knots over Time) that undoes
it exactly, travels with a saved session, and is applied again when a
session reopens a sample from its raw FCS file. The bead events are removed,
when asked, by a 'beads' method in a clean-up gate, not by deleting them.

**Gaussian-parameter clean-up** (:mod:`openflo.gaussian_cleanup`) adds the
'gaussian' method to the same clean-up gate: an auto-clean recipe gate, so
its bounds are recomputed on every sample and saved with the session.

The functions above the windows work without Tk and are what the tests and
the session restore call.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np
import pandas as pd

from .ui_errors import show_error
from .ui_tips import tip

# The clean-up gate the two features share: one auto-clean recipe gate per
# sample, marked so a second run updates it instead of adding another.
CLEANUP_NAME = 'mass cytometry clean-up'
CLEANUP_MARK = 'mass_cleanup'
BEADS_LABEL = 'Bead events'
ALL_EVENTS = 'Above the threshold on every bead channel'
BASE_EACH = "Each file's own beads"
BASE_ALL = "All files together"


# ── Editor-side helpers (no Tk) ─────────────────────────────────────────────

def candidate_samples(editor, selected_only=False):
    """The loaded samples to work on, in tree order: every one, or the
    selected ones; compensation controls left out."""
    names = (list(editor._selected_sample_names()) if selected_only
             else list(editor._sample_order))
    return [n for n in names if n in editor._samples and not editor._is_comp(n)]


def sample_mass_channels(sample):
    """The intensity channels of `sample` a normalization may scale: its
    marker (fluor) channels without Time and the Gaussian parameters."""
    from .bead_normalize import mass_channels
    chans = list(getattr(sample, 'fluor_channels', None)
                 or sample.data.columns)
    return [c for c in mass_channels(chans) if c in sample.data.columns]


def linear_frame(sample, columns):
    """``sample``'s LINEAR values of `columns` as a new frame (the display
    transform each one carries undone)."""
    from .pipeline import linear_values
    return pd.DataFrame({c: np.asarray(linear_values(sample, c), dtype=float)
                         for c in dict.fromkeys(columns)},
                        index=sample.data.index)


def _write_linear(sample, frame, columns):
    """Put `frame`'s LINEAR `columns` back into a new copy of
    ``sample.data``, each on the display transform it carries."""
    from .pipeline import transform_params, transform_values
    rec = getattr(sample, 'data_transforms', None) or {}
    data = sample.data.copy()
    for c in columns:
        if c not in data.columns:
            continue
        lin = frame[c].to_numpy(dtype=float)
        spec = rec.get(c)
        if spec and spec.get('method', 'linear') != 'linear':
            lin = transform_values(lin, **transform_params(spec))
        data[c] = np.asarray(lin, dtype=float)
    sample.data = data


def population_ids(editor, name):
    """``{population path: gate id}`` of `name`'s gates (groups left out)."""
    gates = editor._sample_gates.get(name, {})
    transforms = editor._display_transforms(name)
    out = {}
    for gid in editor._sample_gate_order.get(name, list(gates)):
        g = gates.get(gid)
        if g is None or g.get('kind') == 'group':
            continue
        out.setdefault(editor._population_path(gates, gid,
                                               transforms=transforms), gid)
    return out


def run_bead_normalization(editor, names, bead_channels, *, population=None,
                           thresholds=None, window=500, baseline='each'):
    """Fit (no Tk; safe off the window thread). Nothing loaded changes.

    Returns ``{'results': {name: normalize_runs entry}, 'skipped':
    [(name, why)], 'methods': sentence}``. A sample already normalized, or
    lacking a bead channel or Time, is skipped and said why."""
    from .bead_normalize import (
        find_time_channel,
        fit_run,
        methods_text,
        normalize_runs,
    )
    from .pipeline import UnknownScaleError, cumulative_gate_mask
    frames, masks, skipped = {}, {}, []
    for n in names:
        s = editor._samples.get(n)
        if s is None:
            continue
        if getattr(s, 'bead_norm', None):
            skipped.append((n, "already bead-normalized (undo it first)"))
            continue
        tcol = find_time_channel(s.data.columns)
        missing = [c for c in bead_channels if c not in s.data.columns]
        if tcol is None or missing:
            skipped.append((n, "no Time channel" if tcol is None
                            else f"no {', '.join(missing)}"))
            continue
        chans = sample_mass_channels(s)
        try:
            frames[n] = linear_frame(s, [tcol, *bead_channels, *chans])
        except UnknownScaleError:
            skipped.append((n, "a channel's scale is unknown"))
            continue
        if population:
            gid = population_ids(editor, n).get(population)
            if gid is None:
                frames.pop(n)
                skipped.append((n, "the bead population is not drawn"))
                continue
            masks[n] = np.asarray(cumulative_gate_mask(
                editor._sample_gates.get(n, {}), gid, s.data), dtype=bool)
    if not frames:
        return {'results': {}, 'skipped': skipped, 'methods': ''}
    if baseline not in ('each', 'all') and baseline not in frames:
        raise ValueError(f"the reference file {baseline!r} is not among the "
                         "samples normalized")
    good: dict[str, Any] = {}
    for n in list(frames):
        # Fit each run alone first, so one sample without beads is named
        # and skipped instead of stopping the others.
        try:
            fit_run(frames[n], bead_channels, bead_mask=masks.get(n),
                    thresholds=thresholds)
            good[n] = frames[n]
        except ValueError as exc:
            skipped.append((n, str(exc)))
    if not good:
        return {'results': {}, 'skipped': skipped, 'methods': ''}
    if baseline not in ('each', 'all') and baseline not in good:
        raise ValueError(f"the reference file {baseline!r} could not be "
                         "fitted")
    results = normalize_runs(good, bead_channels, bead_masks=masks or None,
                             thresholds=thresholds, window=window,
                             baseline=baseline)
    for n, r in results.items():
        r['record']['channels'] = [c for c in r['record']['channels']
                                   if c in sample_mass_channels(
                                       editor._samples[n])]
    return {'results': results, 'skipped': skipped,
            'methods': methods_text({n: r['record']
                                     for n, r in results.items()})}


def apply_bead_results(editor, res, add_gate=True):
    """Write a :func:`run_bead_normalization` result into the loaded
    samples (window thread): values replaced, the record kept on each
    sample, and with `add_gate` a clean-up gate removing the bead events.
    Returns the names changed."""
    done = []
    for n, r in res['results'].items():
        s = editor._samples.get(n)
        if s is None:
            continue
        rec = r['record']
        _write_linear(s, r['frame'], rec['channels'])
        s.bead_norm = rec
        editor._data_changed(n)
        done.append(n)
    if not done:
        return done
    editor._undo_barrier("bead normalization (use Undo bead normalization)")
    if add_gate:
        first = res['results'][done[0]]['record']
        set_cleanup_method(editor, done, beads_method(first['bead_channels']))
    editor._audit('bead_normalize', samples=done,
                  bead_channels=list(res['results'][done[0]]['record']
                                     ['bead_channels']),
                  window=res['results'][done[0]]['record']['window'],
                  baseline=res['results'][done[0]]['record']['baseline_from'],
                  n_beads={n: res['results'][n]['record']['n_beads']
                           for n in done},
                  beads_removed=bool(add_gate), methods=res['methods'])
    editor._refresh_gate_list()
    editor._schedule_replot(0)
    return done


def reapply_bead_norm(sample, record, inverse=False):
    """Apply (or, `inverse`, undo) a kept record to ``sample.data`` in place
    of its values. Used by Undo and when a session reopens a sample from its
    raw file."""
    from .bead_normalize import apply_record
    chans = [c for c in record['channels'] if c in sample.data.columns]
    tcol = record['time_channel']
    lin = linear_frame(sample, [tcol, *chans])
    out = apply_record(lin, {**record, 'channels': chans}, inverse=inverse)
    _write_linear(sample, out, chans)


def restore_on_load(sample, record):
    """A session reopened `sample` with a bead normalization `record`: from
    its raw FCS (path ending .fcs) the factors are applied again, from its
    processed sidecar the values already carry them. The record goes back on
    the sample either way. Returns '' or why it could not be applied."""
    from .pipeline import UnknownScaleError
    rec = dict(record)
    need = ('time_channel', 'channels', 'knots_t', 'knots_f')
    if any(k not in rec for k in need):
        return "the saved record is incomplete"
    if str(getattr(sample, 'path', '') or '').lower().endswith('.fcs'):
        if rec['time_channel'] not in sample.data.columns:
            return "no Time channel"
        try:
            reapply_bead_norm(sample, rec)
        except (UnknownScaleError, ValueError) as exc:
            return str(exc)
    sample.bead_norm = rec
    return ''


def restore_bead_normalization(editor, names=None):
    """Put the normalized samples (all, or `names`) back on the values they
    had before. Returns the names restored."""
    done = []
    for n in list(names if names is not None else editor._sample_order):
        s = editor._samples.get(n)
        rec = getattr(s, 'bead_norm', None) if s is not None else None
        if s is None or not rec:
            continue
        reapply_bead_norm(s, rec, inverse=True)
        s.bead_norm = None
        editor._data_changed(n)
        done.append(n)
    if done:
        editor._audit('bead_normalize.restored', samples=done)
        editor._refresh_gate_list()
        editor._schedule_replot(0)
    return done


def beads_method(bead_channels):
    """The auto-clean method dict that removes bead events."""
    return {'key': 'beads', 'label': 'Bead events (mass cytometry)',
            'enabled': True,
            'params': {'channels': ','.join(str(c) for c in bead_channels)}}


def cleanup_gate_id(editor, name):
    """The id of `name`'s mass cytometry clean-up gate, or None."""
    for gid, g in editor._sample_gates.get(name, {}).items():
        if g.get('kind') == 'autoclean' and g.get(CLEANUP_MARK):
            return gid
    return None


def set_cleanup_method(editor, names, method):
    """Add `method` to each sample's clean-up gate (made when missing), or
    replace the method with the same key. One undo step. Returns the names."""
    import copy
    editor._checkpoint()
    out = []
    for n in names:
        if n not in editor._samples:
            continue
        editor._sample_gates.setdefault(n, {})
        gid = cleanup_gate_id(editor, n)
        m = copy.deepcopy(method)
        if gid is None:
            gate = {'kind': 'autoclean', 'name': CLEANUP_NAME,
                    'parent_id': None, 'color': '#808080', 'open': False,
                    CLEANUP_MARK: True, 'methods': [m]}
            editor._add_gate_to_sample(n, gate)
        else:
            g = editor._sample_gates[n][gid]
            methods = [x for x in g.get('methods') or []
                       if x.get('key') != m['key']]
            g['methods'] = methods + [m]
            editor._autoclean_invalidate(n, gid)
        out.append(n)
    return out


# ── Bead normalization window ───────────────────────────────────────────────

class BeadNormalizeWindow(tk.Toplevel):
    """Tools > Mass cytometry > Bead normalization…"""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Bead normalization")
        self.geometry("940x700")
        self.minsize(640, 480)
        self._res: dict[str, Any] | None = None
        ed = editor
        first = ed._samples.get(ed._active_sample) or next(
            (ed._samples[n] for n in ed._sample_order if n in ed._samples),
            None)
        self._chans = sample_mass_channels(first) if first is not None else []
        labels = getattr(first, 'channel_labels', None) or {}

        top = ttk.Frame(self, padding=10)
        top.pack(side='top', fill='x')
        left = ttk.Frame(top)
        left.pack(side='left', fill='y')
        ttk.Label(left, text="Bead channels:").pack(anchor='w')
        lbf = ttk.Frame(left)
        lbf.pack(fill='y', expand=True)
        self.chan_list = tk.Listbox(lbf, selectmode='multiple', height=8,
                                    width=24, exportselection=False)
        sb = ttk.Scrollbar(lbf, orient='vertical',
                           command=self.chan_list.yview)
        self.chan_list.configure(yscrollcommand=sb.set)
        self.chan_list.pack(side='left', fill='y')
        sb.pack(side='left', fill='y')
        from .bead_normalize import suggest_bead_channels
        pre = set(suggest_bead_channels(self._chans, labels))
        for i, c in enumerate(self._chans):
            lab = labels.get(c)
            self.chan_list.insert('end', f"{c} ({lab})" if lab and lab != c
                                  else c)
            if c in pre:
                self.chan_list.selection_set(i)
        tip(editor, self.chan_list,
            "The channels of the metals the normalization beads carry. "
            "Pick every one (usually 4 or 5): a bead event is high on all of "
            "them, and the factor is fitted across them. Channels whose "
            "marker label says 'bead' are picked for you.")

        opts = ttk.Frame(top, padding=(14, 0, 0, 0))
        opts.pack(side='left', fill='both', expand=True)
        ttk.Label(opts, text="Bead events:").grid(row=0, column=0,
                                                  sticky='w')
        self.how_var = tk.StringVar(master=self, value='threshold')
        _r = ttk.Radiobutton(opts, text="high on every bead channel",
                             value='threshold', variable=self.how_var)
        _r.grid(row=0, column=1, sticky='w')
        tip(editor, _r, "Find the beads as the events above a threshold on "
                        "every bead channel (Otsu's split of "
                        "arcsinh(x/5), worked out per sample).")
        _r = ttk.Radiobutton(opts, text="the population",
                             value='population', variable=self.how_var)
        _r.grid(row=1, column=1, sticky='w')
        tip(editor, _r, "Use a population you gated on the beads instead.")
        self.pop_var = tk.StringVar(master=self, value='')
        pops: dict[str, None] = {}
        for n in ed._sample_order:
            if n in ed._samples:
                for p in population_ids(ed, n):
                    pops.setdefault(p, None)
        self.pop_combo = ttk.Combobox(opts, textvariable=self.pop_var,
                                      values=list(pops), width=28,
                                      state='readonly')
        self.pop_combo.grid(row=1, column=2, sticky='w', padx=4)
        tip(editor, self.pop_combo, "The population holding the bead events "
                                    "(by its path, on every sample).")
        ttk.Label(opts, text="Thresholds:").grid(row=2, column=0,
                                                 sticky='w', pady=(4, 0))
        self.thr_var = tk.StringVar(master=self, value='')
        _e = ttk.Entry(opts, textvariable=self.thr_var, width=40)
        _e.grid(row=2, column=1, columnspan=2, sticky='w', pady=(4, 0))
        tip(editor, _e,
            "Blank: automatic per sample (Otsu's split). Or set some as "
            "channel=value on arcsinh(x/5), e.g. Mass140=4, comma-separated; "
            "the others stay automatic. The thresholds each sample used are "
            "listed under the plots.")
        ttk.Label(opts, text="Window (beads):").grid(row=3, column=0,
                                                     sticky='w', pady=(4, 0))
        self.win_var = tk.StringVar(master=self, value='500')
        _e = ttk.Entry(opts, textvariable=self.win_var, width=8)
        _e.grid(row=3, column=1, sticky='w', pady=(4, 0))
        tip(editor, _e,
            "How many consecutive beads each running median covers (500 by "
            "default, as Finck et al.). Fewer follow faster drift but carry "
            "more noise; a run with fewer beads uses them all.")
        ttk.Label(opts, text="Baseline:").grid(row=4, column=0, sticky='w',
                                               pady=(4, 0))
        self.base_var = tk.StringVar(master=self, value=BASE_ALL)
        names = candidate_samples(ed)
        self.base_combo = ttk.Combobox(
            opts, textvariable=self.base_var, state='readonly', width=28,
            values=[BASE_ALL, BASE_EACH, *names])
        self.base_combo.grid(row=4, column=1, columnspan=2, sticky='w',
                             pady=(4, 0))
        tip(editor, self.base_combo,
            "The level the beads are held at. All files together: one "
            "shared level, so runs become comparable with each other (the "
            "usual choice). Each file's own: only the drift within each run "
            "is removed. A file name: that run's beads are the reference.")
        ttk.Label(opts, text="Samples:").grid(row=5, column=0, sticky='w',
                                              pady=(4, 0))
        self.scope_var = tk.StringVar(master=self, value='all')
        sc = ttk.Frame(opts)
        sc.grid(row=5, column=1, columnspan=2, sticky='w', pady=(4, 0))
        for val, text, help_ in (
                ('all', "all loaded", "Every loaded sample except "
                                      "compensation controls."),
                ('selected', "selected", "Only the samples selected in "
                                         "the tree.")):
            _r = ttk.Radiobutton(sc, text=text, value=val,
                                 variable=self.scope_var)
            _r.pack(side='left', padx=(0, 6))
            tip(editor, _r, help_)
        self.gate_var = tk.BooleanVar(master=self, value=True)
        _c = ttk.Checkbutton(opts, text="Remove bead events with a "
                             "clean-up gate", variable=self.gate_var)
        _c.grid(row=6, column=0, columnspan=3, sticky='w', pady=(6, 0))
        tip(editor, _c,
            "Add a 'Bead events' method to the sample's mass cytometry "
            "clean-up gate, so gates under it leave the beads out. Nothing "
            "is deleted; untick the method to see them again.")

        self.status_var = tk.StringVar(master=self, value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=900, justify='left',
                  padding=(10, 0, 10, 2)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        self.run_btn = ttk.Button(btns, text="Normalize", default='active',
                                  command=self._run)
        self.run_btn.pack(side='left')
        tip(editor, self.run_btn,
            "Find the beads, fit the factor over Time and scale the mass "
            "channels of every chosen sample. Time and the Gaussian "
            "parameters are not changed.")
        self.undo_btn = ttk.Button(btns, text="Undo bead normalization",
                                   command=self._undo)
        self.undo_btn.pack(side='left', padx=(6, 0))
        tip(editor, self.undo_btn,
            "Put the normalized samples back on their values from before. "
            "Also under Tools > Mass cytometry.")
        _b = ttk.Button(btns, text="Copy methods", command=self._copy_methods)
        _b.pack(side='left', padx=6)
        tip(editor, _b, "Copy the methods sentence (method, citation, beads, "
                        "window, baseline and factor range).")
        ttk.Button(btns, text="Close", command=self.destroy).pack(
            side='right')

        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(side='top', fill='both', expand=True)
        row = ttk.Frame(body)
        row.pack(anchor='w', fill='x')
        ttk.Label(row, text="Show:").pack(side='left')
        self.show_sample = tk.StringVar(master=self, value='')
        self.show_combo = ttk.Combobox(row, textvariable=self.show_sample,
                                       state='readonly', width=22)
        self.show_combo.pack(side='left', padx=4)
        tip(editor, self.show_combo, "The sample the plots show.")
        self.show_chan = tk.StringVar(master=self, value='')
        self.show_chan_combo = ttk.Combobox(
            row, textvariable=self.show_chan, state='readonly', width=16)
        self.show_chan_combo.pack(side='left', padx=4)
        tip(editor, self.show_chan_combo,
            "The bead channel drawn over Time, before and after.")
        for w in (self.show_combo, self.show_chan_combo):
            w.bind('<<ComboboxSelected>>', lambda _e: self._draw())
        from .tool_window import figure_panel
        _f, self.fig, self.canvas = figure_panel(body, figsize=(8.6, 3.2))
        self.text = tk.Text(body, height=6, wrap='word')
        self.text.pack(fill='x')
        self._sync_buttons()

    # -- settings ---------------------------------------------------------
    def bead_channels(self):
        return [self._chans[int(i)] for i in self.chan_list.curselection()]

    def select_bead_channels(self, chans):
        self.chan_list.selection_clear(0, 'end')
        for i, c in enumerate(self._chans):
            if c in chans:
                self.chan_list.selection_set(i)

    def _thresholds(self):
        """``{channel: value}`` typed, or None after saying what is wrong."""
        out: dict[str, float] = {}
        for part in self.thr_var.get().split(','):
            part = part.strip()
            if not part:
                continue
            ch, sep, val = part.rpartition('=')
            try:
                v = float(val)
            except ValueError:
                v = float('nan')
            if not sep or not ch.strip() or not np.isfinite(v):
                self.status_var.set(f"Threshold '{part}' is not "
                                    "channel=number.")
                return None
            out[ch.strip()] = v
        return out

    def _baseline(self):
        b = self.base_var.get()
        return {BASE_ALL: 'all', BASE_EACH: 'each'}.get(b, b)

    def _sync_buttons(self):
        has = any(getattr(self.editor._samples.get(n), 'bead_norm', None)
                  for n in self.editor._sample_order)
        self.undo_btn.configure(state='normal' if has else 'disabled')

    # -- run --------------------------------------------------------------
    def _run(self, sync=False):
        from .dialog_inputs import read_int
        beads = self.bead_channels()
        if not beads:
            self.status_var.set("Pick the bead channels first.")
            return
        pop = None
        if self.how_var.get() == 'population':
            pop = self.pop_var.get()
            if not pop:
                self.status_var.set("Choose the bead population.")
                return
        thr = self._thresholds()
        if thr is None:
            return
        unknown = [c for c in thr if c not in beads]
        if unknown:
            self.status_var.set(f"{', '.join(unknown)} is not a chosen bead "
                                "channel.")
            return
        window = read_int(self.win_var, "Window (beads)", lo=3, hi=100000,
                          parent=self)
        if window is None:
            return
        names = candidate_samples(self.editor,
                                  self.scope_var.get() == 'selected')
        if not names:
            self.status_var.set("No sample to normalize.")
            return
        base = self._baseline()
        add_gate = bool(self.gate_var.get())

        def work():
            return run_bead_normalization(
                self.editor, names, beads, population=pop, thresholds=thr,
                window=window, baseline=base)

        self.status_var.set(f"Following the beads in {len(names)} "
                            "sample(s)…")
        self.run_btn.configure(state='disabled')
        if sync:
            try:
                self._done(work(), add_gate)
            except Exception as exc:                     # noqa: BLE001
                self._failed(exc)
            return
        from .async_task import run_async
        run_async(self, work, on_done=lambda r: self._done(r, add_gate),
                  on_error=self._failed)

    def _failed(self, exc):
        self.run_btn.configure(state='normal')
        self.status_var.set(f"Bead normalization failed: {exc}")
        show_error(self, "Bead normalization",
                   "The samples could not be normalized.", exc)

    def _done(self, res, add_gate):
        self.run_btn.configure(state='normal')
        done = apply_bead_results(self.editor, res, add_gate=add_gate)
        if done:
            # A run that changed nothing keeps the last result on show.
            self._res = res
        if done:
            self.show_combo.configure(values=done)
            if self.show_sample.get() not in done:
                self.show_sample.set(done[0])
            beads = res['results'][done[0]]['record']['bead_channels']
            self.show_chan_combo.configure(values=beads)
            if self.show_chan.get() not in beads:
                self.show_chan.set(beads[0])
        msg = (f"Normalized {len(done)} sample(s)." if done
               else "Nothing was normalized.")
        if res['skipped']:
            msg += "  Skipped: " + '; '.join(f"{n} ({why})"
                                             for n, why in res['skipped'])
        self.status_var.set(msg)
        self._sync_buttons()
        self._draw()

    def _undo(self):
        done = restore_bead_normalization(self.editor)
        self.status_var.set(f"Restored the values of {len(done)} sample(s)."
                            if done else "Nothing to undo.")
        self._sync_buttons()

    def _copy_methods(self):
        if not self._res or not self._res.get('methods'):
            self.status_var.set("Normalize first.")
            return
        self.clipboard_clear()
        self.clipboard_append(self._res['methods'])
        self.status_var.set("Copied the methods sentence.")

    # -- plots ------------------------------------------------------------
    def traces(self, name, channel):
        """``(times, before, after, knots_t, knots_f)`` for the plot: the
        running median of `channel` over the sample's beads before and after
        the factor, and the factor's knots."""
        from .bead_normalize import bead_trace, factors_at
        r = self._res['results'].get(name) if self._res else None
        if r is None or channel not in r['fit']['bead_channels']:
            return None
        fit, rec = r['fit'], r['record']
        j = fit['bead_channels'].index(channel)
        t = fit['times']
        raw = fit['raw'][:, j]
        f = factors_at(t, rec['knots_t'], rec['knots_f'])
        tt, before = bead_trace(t, raw, rec['window'])
        _t, after = bead_trace(t, raw * f, rec['window'])
        return tt, before, after, rec['knots_t'], rec['knots_f']

    def _draw(self):
        from .theme import categorical_colors
        self.fig.clear()
        name, ch = self.show_sample.get(), self.show_chan.get()
        tr = self.traces(name, ch) if (self._res and name and ch) else None
        if tr is None:
            self.canvas.draw_idle()
            return
        t, before, after, kt, kf = tr
        rec = self._res['results'][name]['record'] if self._res else {}
        c_before, c_after = categorical_colors(2)
        ax = self.fig.add_subplot(1, 2, 1)
        ax.plot(t, before, color=c_before, lw=1.2, label="before")
        ax.plot(t, after, color=c_after, lw=1.2, label="after")
        base = rec.get('baseline', {}).get(ch)
        if base is not None:
            ax.axhline(base, color='0.5', lw=0.8, ls='--', label="baseline")
        ax.set_title(f"{ch}: bead running median", fontsize=9)
        ax.set_xlabel(rec.get('time_channel', 'Time'), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc='best')
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.plot(kt, kf, color=c_after, lw=1.2)
        ax2.axhline(1.0, color='0.5', lw=0.8, ls=':')
        ax2.set_title("Factor applied", fontsize=9)
        ax2.set_xlabel(rec.get('time_channel', 'Time'), fontsize=8)
        ax2.tick_params(labelsize=7)
        try:
            self.fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
            pass
        self.canvas.draw_idle()
        from .bead_normalize import drift_percent
        lines = []
        for n, r in (self._res['results'].items() if self._res else ()):
            fit, rc = r['fit'], r['record']
            f = np.asarray(rc['knots_f'])
            from .bead_normalize import factors_at, smooth_beads
            g = factors_at(fit['times'], rc['knots_t'], rc['knots_f'])
            _t, after_s = smooth_beads(fit['times'], fit['raw'] * g[:, None],
                                       rc['window'])
            lines.append(
                f"{n}: {rc['n_beads']:,} beads; factor {f.min():.3g} to "
                f"{f.max():.3g}; bead drift {drift_percent(fit['smoothed']):.1f}"
                f" % before, {drift_percent(after_s):.1f} % after")
            if rc.get('thresholds'):
                lines.append("    thresholds (arcsinh(x/5)): " + ', '.join(
                    f"{c} {v:.3g}" for c, v in rc['thresholds'].items()))
        lines += ['', self._res['methods'] if self._res else '']
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', '\n'.join(lines))
        self.text.configure(state='disabled')


# ── Gaussian clean-up window ────────────────────────────────────────────────

class GaussianCleanupWindow(tk.Toplevel):
    """Tools > Mass cytometry > Gaussian parameter clean-up…"""

    def __init__(self, editor):
        super().__init__(editor)
        from .gaussian_cleanup import DEFAULT_N_MAD, find_gaussian_channels
        self.editor = editor
        self.title("Gaussian parameter clean-up")
        self.geometry("820x640")
        self.minsize(560, 440)
        ed = editor
        names = candidate_samples(ed)
        found: dict[str, None] = {}
        for n in names:
            for c in find_gaussian_channels(ed._samples[n].data.columns):
                found.setdefault(c, None)
        self._params = list(found)

        top = ttk.Frame(self, padding=10)
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Parameters:").grid(row=0, column=0, sticky='w')
        pf = ttk.Frame(top)
        pf.grid(row=0, column=1, columnspan=3, sticky='w')
        self.param_vars: dict[str, tk.BooleanVar] = {}
        for c in self._params:
            v = tk.BooleanVar(master=self, value=True)
            _c = ttk.Checkbutton(pf, text=c, variable=v,
                                 command=self._preview)
            _c.pack(side='left', padx=(0, 8))
            tip(editor, _c, f"Clean on {c}: events far from the sample's "
                            "median on it are removed.")
            self.param_vars[c] = v
        if not self._params:
            ttk.Label(pf, text="No Event_length, Center, Offset, Width or "
                      "Residual channel in the loaded samples.",
                      style='Muted.TLabel').pack(side='left')
        ttk.Label(top, text="Keep within:").grid(row=1, column=0, sticky='w',
                                                 pady=(6, 0))
        self.n_var = tk.StringVar(master=self, value=f"{DEFAULT_N_MAD:g}")
        _e = ttk.Entry(top, textvariable=self.n_var, width=6)
        _e.grid(row=1, column=1, sticky='w', pady=(6, 0))
        _e.bind('<Return>', lambda _e: self._preview())
        tip(editor, _e,
            "How many robust SDs (1.4826 x MAD) either side of each sample's "
            "median are kept, per parameter. 3 is the usual outlier rule: "
            "about 0.3 % of a clean, normal spread lies beyond it. Larger "
            "keeps more.")
        ttk.Label(top, text="robust SDs of the median, on").grid(
            row=1, column=2, sticky='w', pady=(6, 0))
        self.scale_var = tk.StringVar(master=self, value='asinh')
        _s = ttk.Combobox(top, textvariable=self.scale_var, width=8,
                          state='readonly', values=('asinh', 'linear'))
        _s.grid(row=1, column=3, sticky='w', pady=(6, 0))
        _s.bind('<<ComboboxSelected>>', lambda _e: self._preview())
        tip(editor, _s,
            "asinh: bounds on arcsinh(x/5), which suits these right-skewed "
            "parameters (the default). linear: on the values as they are.")
        ttk.Label(top, text="Samples:").grid(row=2, column=0, sticky='w',
                                             pady=(6, 0))
        self.scope_var = tk.StringVar(master=self, value='all')
        sc = ttk.Frame(top)
        sc.grid(row=2, column=1, columnspan=3, sticky='w', pady=(6, 0))
        for val, text, help_ in (
                ('all', "all loaded", "Every loaded sample except "
                                      "compensation controls."),
                ('selected', "selected", "Only the samples selected in "
                                         "the tree.")):
            _r = ttk.Radiobutton(sc, text=text, value=val,
                                 variable=self.scope_var,
                                 command=self._preview)
            _r.pack(side='left', padx=(0, 6))
            tip(editor, _r, help_)

        self.status_var = tk.StringVar(master=self, value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  wraplength=780, justify='left',
                  padding=(10, 0, 10, 2)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        _b = ttk.Button(btns, text="Preview", command=self._preview)
        _b.pack(side='left')
        tip(editor, _b, "Count what these settings remove in each sample, "
                        "without changing anything.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(
            side='right')
        self.add_btn = ttk.Button(btns, text="Add clean-up gate",
                                  default='active', command=self._add)
        self.add_btn.pack(side='right', padx=(0, 6))
        tip(editor, self.add_btn,
            "Add these settings as the 'Gaussian parameters' method of each "
            "sample's mass cytometry clean-up gate (an auto-clean gate: "
            "recomputed per sample, saved with the session). Gate under it "
            "to work on the clean events.")

        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(side='top', fill='both', expand=True)
        self.tv = ttk.Treeview(body, show='headings', height=5,
                               columns=('sample', 'events', 'removed', 'pct'))
        for c, t, w in (('sample', 'Sample', 220), ('events', 'Events', 90),
                        ('removed', 'Removed', 90), ('pct', '% removed', 90)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor='w' if c == 'sample' else 'e')
        self.tv.pack(fill='x')
        from .tool_window import figure_panel
        _f, self.fig, self.canvas = figure_panel(body, figsize=(7.8, 2.8))
        self._rows: list[dict] = []
        self._preview()

    def params(self):
        """The method params these settings make, or None (status says
        why)."""
        from .dialog_inputs import parse_number
        n = parse_number(self.n_var.get(), lo=0.5, hi=50)
        if n is None:
            self.status_var.set("Keep within must be a number from 0.5 to "
                                "50.")
            return None
        chans = [c for c, v in self.param_vars.items() if v.get()]
        if not chans:
            self.status_var.set("Tick at least one parameter.")
            return None
        return {'channels': ','.join(chans), 'n_mad': float(n),
                'scale': self.scale_var.get()}

    def _names(self):
        return candidate_samples(self.editor,
                                 self.scope_var.get() == 'selected')

    def _preview(self):
        from .gaussian_cleanup import cleanup_summary, gaussian_keep_mask
        p = self.params()
        self.tv.delete(*self.tv.get_children())
        self._rows = []
        if p is None:
            return
        for n in self._names():
            df = self.editor._samples[n].data
            keep = gaussian_keep_mask(df, p)
            removed = int((~keep).sum())
            self._rows.append({'sample': n, 'events': len(df),
                               'removed': removed,
                               'summary': cleanup_summary(df, p)})
            self.tv.insert('', 'end', values=(
                n, f"{len(df):,}", f"{removed:,}",
                f"{100.0 * removed / max(len(df), 1):.2f}"))
        self.status_var.set(f"{len(self._rows)} sample(s) previewed." if
                            self._rows else "No sample with these "
                            "parameters.")
        self._draw()

    def _draw(self):
        self.fig.clear()
        if not self._rows:
            self.canvas.draw_idle()
            return
        row = next((r for r in self._rows
                    if r['sample'] == self.editor._active_sample),
                   self._rows[0])
        summ = row['summary']
        df = self.editor._samples[row['sample']].data
        scale = self.scale_var.get()
        k = max(len(summ), 1)
        for i, s in enumerate(summ):
            ax = self.fig.add_subplot(1, k, i + 1)
            v = np.asarray(df[s['channel']].values, dtype=float)
            v = v[np.isfinite(v)]
            tf = (lambda x: np.arcsinh(np.asarray(x) / 5.0)) \
                if scale == 'asinh' else (lambda x: np.asarray(x))
            if v.size:
                ax.hist(tf(v), bins=80, color='0.45')
            for b in (s['low'], s['high']):
                if np.isfinite(b):
                    ax.axvline(float(tf(b)), color='#d62728', lw=1)
            ax.set_title(s['channel'], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_yticks([])
        self.fig.suptitle(f"{row['sample']} ({scale})", fontsize=8)
        try:
            self.fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
            pass
        self.canvas.draw_idle()

    def _add(self):
        from .gaussian_cleanup import default_method
        p = self.params()
        if p is None:
            return
        names = self._names()
        if not names:
            self.status_var.set("No sample to clean.")
            return
        m = default_method(p['channels'].split(','), p['n_mad'], p['scale'])
        done = set_cleanup_method(self.editor, names, m)
        ed = self.editor
        ed._audit('gaussian_cleanup', samples=done,
                  parameters=p['channels'].split(','), n_mad=p['n_mad'],
                  scale=p['scale'])
        ed._refresh_gate_list()
        ed._schedule_replot(0)
        self.status_var.set(f"Clean-up gate set on {len(done)} sample(s). "
                            "Gate under it to work on the clean events.")
