"""Normalize batches (CytoNorm) dialog.

Data acquired on different days drift, and pooled multi-day data then split
by day instead of by biology. CytoNorm (Van Gassen et al. 2020) maps each
batch's intensities, per channel and FlowSOM metacluster, onto a shared goal
distribution. Its standard form learns the mapping from a reference sample
run in every batch (the same material each day); the control-free form
(CytoNorm 2.0) pools every sample of a batch.

The dialog chooses how samples fall into batches (their trial / day, or any
FCS keyword such as $DATE), which sample is each batch's reference (one name
token, e.g. "ref", that matches one sample per batch), and where the result
goes: the loaded samples, normalized FCS copies in a folder, or both. It
then shows each batch before and after on one channel, the per-batch
medians, and a methods sentence (also kept in the audit trail).

The fitting is batch_normalize (around pipeline.CytoNorm); this module is
the Tk around it.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, ttk

from .ui_errors import show_error
from .ui_tips import tip

BY_TRIAL = 'Trial / day'
BY_KEYWORD = 'FCS keyword'


def plan_batches(editor, by, keyword='', mode='goal', ref_token=''):
    """Group the editor's loaded samples into batches and, in reference
    mode, find each batch's reference. Compensation controls and samples
    set aside are left out.

    Returns ``{'batch_of', 'references', 'channels', 'left_out', 'error'}``;
    ``error`` is a sentence saying what stops the fit, or ''."""
    from .batch_normalize import MISSING_BATCH, keyword_value
    from .workspace import proper_run_channels
    names, left_out = [], []
    for n in editor._sample_order:
        if n not in editor._samples:
            continue
        if (editor._is_comp(n)
                or n in getattr(editor, '_sample_set_aside', {})):
            left_out.append(n)
            continue
        names.append(n)
    batch_of = {}
    for n in names:
        if by == BY_KEYWORD:
            meta = getattr(editor._samples[n], 'metadata', None) or {}
            batch_of[n] = keyword_value(meta, keyword) or MISSING_BATCH
        else:
            batch_of[n] = editor._sample_trial.get(n, 'Trial')
    out = {'batch_of': batch_of, 'references': {}, 'channels': [],
           'left_out': left_out, 'error': ''}
    batches = list(dict.fromkeys(batch_of.values()))
    if len(batches) < 2:
        out['error'] = (f"Found {len(batches)} batch(es); normalizing needs "
                        "samples from at least 2.")
        return out
    shared = None
    for n in names:
        cs = set(proper_run_channels(editor._samples[n]))
        shared = cs if shared is None else shared & cs
    first = editor._samples[names[0]]
    out['channels'] = [c for c in proper_run_channels(first)
                       if c in (shared or set()) and c in first.data.columns]
    if not out['channels']:
        out['error'] = "The samples share no marker channel to normalize."
        return out
    if mode == 'controls':
        tok = ref_token.strip().lower()
        if not tok:
            out['error'] = ("Type the name token of the reference sample, "
                            "e.g. ref.")
            return out
        problems = []
        for b in batches:
            hits = [n for n in names if batch_of[n] == b and tok in n.lower()]
            if len(hits) != 1:
                problems.append(f"{b}: {len(hits)} matches")
            else:
                out['references'][b] = hits[0]
        if problems:
            out['error'] = ("Each batch needs exactly one reference sample "
                            f"containing '{ref_token.strip()}' ("
                            + '; '.join(problems) + ").")
    return out


def run_normalization(editor, plan, mode, n_metaclusters, out_dir=None):
    """Fit and apply (no Tk; safe off the window thread). Returns the
    model, the corrected frames, the before/after summary, the methods
    sentence and any written copies. Nothing loaded is changed."""
    from .batch_normalize import (
        apply_batch_model,
        fit_batch_model,
        methods_text,
        normalization_summary,
        write_normalized_copies,
    )
    batch_of, refs = plan['batch_of'], plan['references'] or None
    frames = {n: editor._samples[n].data for n in batch_of}
    model = fit_batch_model(frames, batch_of, plan['channels'], mode=mode,
                            references=refs, n_metaclusters=n_metaclusters)
    corrected = apply_batch_model(model, frames, batch_of)
    summary = normalization_summary(frames, corrected, batch_of,
                                    plan['channels'], references=refs)
    written = []
    if out_dir:
        written = write_normalized_copies(
            {n: editor._samples[n] for n in batch_of}, corrected, batch_of,
            model, out_dir, references=refs)
    return {'model': model, 'corrected': corrected, 'summary': summary,
            'methods': methods_text(model, summary, references=refs),
            'written': written, 'before': frames}


class BatchNormalizeWindow(tk.Toplevel):
    """Analyze → Normalize batches (CytoNorm)…"""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Normalize batches (CytoNorm)")
        self.geometry("820x700")
        self.minsize(600, 480)
        self._result = None
        self._plan = None

        frm = ttk.Frame(self, padding=10)
        frm.pack(side='top', fill='x')

        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x')
        ttk.Label(row, text="Batches by:").pack(side='left')
        self.by_var = tk.StringVar(value=BY_TRIAL)
        by_help = {
            BY_TRIAL: "One batch per trial / day, as the sample tree groups "
                      "them.",
            BY_KEYWORD: "One batch per value of the FCS keyword typed "
                        "beside it, read from each file."}
        for val in (BY_TRIAL, BY_KEYWORD):
            _rb = ttk.Radiobutton(row, text=val, value=val,
                                  variable=self.by_var, command=self._preview)
            _rb.pack(side='left', padx=4)
            tip(editor, _rb, by_help[val])
        self.keyword_var = tk.StringVar(value='$DATE')
        kw = ttk.Entry(row, textvariable=self.keyword_var, width=14)
        kw.pack(side='left', padx=4)
        kw.bind('<KeyRelease>', lambda _e: self._preview())
        tip(editor, kw,
            "The FCS keyword whose value names a sample's batch, e.g. $DATE "
            "(the acquisition date) or a keyword your cytometer writes for "
            "the plate or run.")

        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x', pady=(6, 0))
        ttk.Label(row, text="Learn the mapping from:").pack(side='left')
        self.mode_var = tk.StringVar(value='controls')
        rb = ttk.Radiobutton(row, text="a reference sample in every batch,"
                             " named with", value='controls',
                             variable=self.mode_var, command=self._preview)
        rb.pack(side='left', padx=4)
        tip(editor, rb,
            "Learn each batch's mapping from its reference sample only: the "
            "same material run in every batch (CytoNorm's standard form).")
        self.ref_var = tk.StringVar(value='ref')
        ref = ttk.Entry(row, textvariable=self.ref_var, width=10)
        ref.pack(side='left')
        ref.bind('<KeyRelease>', lambda _e: self._preview())
        tip(editor, ref,
            "A piece of the reference sample's name, e.g. ref: the same "
            "control material run in every batch. CytoNorm's standard form, "
            "and the one to prefer: the mapping is learnt where biology is "
            "known to be equal.")
        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x')
        ttk.Label(row, text="", width=22).pack(side='left')
        _goal = ttk.Radiobutton(row, text="all samples of each batch (no "
                                "reference; CytoNorm 2.0)", value='goal',
                                variable=self.mode_var, command=self._preview)
        _goal.pack(side='left', padx=4)
        tip(editor, _goal,
            "No reference run: pool every sample of a batch. Assumes the "
            "batches hold comparable samples overall, so a real difference "
            "between batches would be normalized away.")

        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x', pady=(6, 0))
        ttk.Label(row, text="Metaclusters:").pack(side='left')
        self.meta_var = tk.IntVar(value=10)
        sp = ttk.Spinbox(row, from_=1, to=40, width=5,
                         textvariable=self.meta_var)
        sp.pack(side='left', padx=4)
        tip(editor, sp,
            "FlowSOM metaclusters, each normalized on its own so a shift "
            "that differs between populations is followed. 1 = one mapping "
            "per channel and batch (plain quantile normalization).")

        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x', pady=(6, 0))
        self.apply_var = tk.BooleanVar(value=True)
        _ap = ttk.Checkbutton(row, text="Apply to the loaded samples (not "
                              "undoable; reload the files to revert)",
                              variable=self.apply_var)
        _ap.pack(side='left')
        tip(editor, _ap,
            "Replace the loaded samples' values with the normalized ones, so "
            "gates, statistics and clustering use them.")
        row = ttk.Frame(frm)
        row.pack(anchor='w', fill='x')
        self.copies_var = tk.BooleanVar(value=False)
        _cp = ttk.Checkbutton(row, text="Write normalized FCS copies to:",
                              variable=self.copies_var)
        _cp.pack(side='left')
        tip(editor, _cp,
            "Write one <name>_norm.fcs per sample into the folder, with "
            "keywords recording the normalization. The original files are "
            "never changed.")
        self.dir_var = tk.StringVar(value='')
        _de = ttk.Entry(row, textvariable=self.dir_var, width=36)
        _de.pack(side='left', padx=4)
        tip(editor, _de, "The folder the normalized copies go to.")
        _br = ttk.Button(row, text="Browse…", command=self._pick_dir)
        _br.pack(side='left')
        tip(editor, _br, "Choose the folder for the normalized copies.")

        self.plan_var = tk.StringVar(value='')
        ttk.Label(frm, textvariable=self.plan_var, style='Muted.TLabel',
                  wraplength=780, justify='left').pack(anchor='w',
                                                       pady=(8, 0))

        self.status_var = tk.StringVar(value='')
        ttk.Label(self, textvariable=self.status_var, style='Muted.TLabel',
                  padding=(10, 0, 10, 2)).pack(side='bottom', fill='x')
        btns = ttk.Frame(self, padding=(10, 0, 10, 10))
        btns.pack(side='bottom', fill='x')
        self.run_btn = ttk.Button(btns, text="Normalize", command=self._run)
        self.run_btn.pack(side='left')
        tip(editor, self.run_btn,
            "Fit CytoNorm with these settings and send the result where "
            "ticked above.")
        self.restore_btn = ttk.Button(btns, text="Restore original values",
                                      command=self._restore)
        self.restore_btn.pack(side='left', padx=(6, 0))
        tip(editor, self.restore_btn,
            "Put the loaded samples back on the values they had before "
            "normalization, without reloading the files. Also under Tools.")
        _cm = ttk.Button(btns, text="Copy methods",
                         command=self._copy_methods)
        _cm.pack(side='left', padx=6)
        tip(editor, _cm,
            "Copy the methods sentence (method, citation, batches, "
            "reference, before and after spread) for a paper.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(
            side='right')

        body = ttk.Frame(self, padding=(10, 0, 10, 6))
        body.pack(side='top', fill='both', expand=True)
        crow = ttk.Frame(body)
        crow.pack(anchor='w', fill='x')
        ttk.Label(crow, text="Show channel:").pack(side='left')
        self.chan_var = tk.StringVar(value='')
        self.chan_combo = ttk.Combobox(crow, textvariable=self.chan_var,
                                       state='readonly', width=18)
        self.chan_combo.pack(side='left', padx=4)
        tip(editor, self.chan_combo,
            "The channel the before / after histograms show.")
        self.chan_combo.bind('<<ComboboxSelected>>',
                             lambda _e: self._draw())
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(7.5, 3.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.text = tk.Text(body, height=8, wrap='word')
        self.text.pack(fill='x')
        self._preview()

    def _pick_dir(self):
        d = filedialog.askdirectory(parent=self,
                                    title="Folder for normalized copies")
        if d:
            self.dir_var.set(d)
            self.copies_var.set(True)

    def _preview(self):
        """Show how the samples fall into batches before anything runs."""
        self._plan = plan_batches(
            self.editor, self.by_var.get(), self.keyword_var.get(),
            self.mode_var.get(), self.ref_var.get())
        p = self._plan
        groups = {}
        for n, b in p['batch_of'].items():
            groups.setdefault(b, []).append(n)
        parts = []
        for b, ns in groups.items():
            ref = p['references'].get(b)
            parts.append(f"{b}: {len(ns)} sample(s)"
                         + (f", reference {ref}" if ref else ''))
        lines = ['; '.join(parts) if parts else "No samples loaded."]
        if p['left_out']:
            lines.append("Left out (compensation controls or set aside): "
                         + ', '.join(p['left_out']))
        if p['error']:
            lines.append(p['error'])
        self.plan_var.set('\n'.join(lines))
        self.run_btn.configure(state='disabled' if p['error'] else 'normal')
        self.restore_btn.configure(
            state='normal' if getattr(self.editor, '_batch_norm_originals',
                                      None) else 'disabled')

    def _run(self, sync=False):
        self._preview()
        p = self._plan
        if not p or p['error']:
            return
        out_dir = self.dir_var.get().strip() if self.copies_var.get() else ''
        if self.copies_var.get() and not out_dir:
            self.status_var.set("Choose a folder for the copies first.")
            return
        if not self.apply_var.get() and not out_dir:
            self.status_var.set("Choose where the result goes: the loaded "
                                "samples, copies, or both.")
            return
        mode = self.mode_var.get()
        try:
            nmeta = max(1, int(self.meta_var.get()))
        except (tk.TclError, ValueError):
            nmeta = 10

        def work():
            return run_normalization(self.editor, p, mode, nmeta,
                                     out_dir or None)

        self.status_var.set(
            f"Fitting CytoNorm across {len(set(p['batch_of'].values()))} "
            f"batches on {len(p['channels'])} channels…")
        self.run_btn.configure(state='disabled')
        if sync:
            try:
                self._done(work(), mode, nmeta)
            except Exception as exc:                     # noqa: BLE001
                self._failed(exc)
            return
        from .async_task import run_async
        run_async(self, work, on_done=lambda r: self._done(r, mode, nmeta),
                  on_error=self._failed)

    def _failed(self, exc):
        self.run_btn.configure(state='normal')
        self.status_var.set(f"Normalization failed: {exc}")
        show_error(self, "Normalize batches",
                   "The batches could not be normalized.", exc)

    def _done(self, res, mode, nmeta):
        self.run_btn.configure(state='normal')
        self._result = res
        ed = self.editor
        applied = bool(self.apply_var.get())
        if applied:
            # Keep each sample's values from before the FIRST normalization,
            # so Restore original values goes back to them without reloading
            # (a second normalization stacks on the first; restore undoes
            # both).
            keep = ed.__dict__.setdefault('_batch_norm_originals', {})
            for n, df in res['corrected'].items():
                s = ed._samples.get(n)
                if s is None:
                    continue
                keep.setdefault(n, s.data)
                s.data = df
                ed._data_changed(n)
            ed._undo_barrier("batch normalization (use Restore original "
                             "values)")
            ed._refresh_gate_list()
            ed._schedule_replot(0)
        refs = self._plan['references'] if self._plan else {}
        ed._audit('cytonorm', mode=mode, n_metaclusters=nmeta,
                  n_samples=len(res['corrected']),
                  n_batches=len(res['model'].batches),
                  batches=list(res['model'].batches),
                  references=dict(refs), n_markers=len(
                      res['summary']['channel'].unique()),
                  applied=applied, copies=len(res['written']),
                  methods=res['methods'])
        chans = list(dict.fromkeys(res['summary']['channel']))
        self.chan_combo.configure(values=chans)
        if chans and self.chan_var.get() not in chans:
            self.chan_var.set(chans[0])
        self._draw()
        self._preview()
        bits = []
        if applied:
            bits.append(f"applied to {len(res['corrected'])} sample(s)")
        if res['written']:
            bits.append(f"{len(res['written'])} cop(ies) written to "
                        f"{os.path.basename(os.path.dirname(res['written'][0])) or res['written'][0]}")
        self.status_var.set("Normalized: " + ', '.join(bits) + '.')

    def _draw(self):
        """Each batch's distribution of the chosen channel, before and
        after, and the per-batch medians."""
        res = self._result
        if res is None:
            return
        import numpy as np

        from .theme import categorical_colors
        ch = self.chan_var.get()
        summ = res['summary']
        sub = summ[summ['channel'] == ch]
        batches = list(sub['batch'])
        colors = categorical_colors(len(batches))
        batch_of = self._plan['batch_of'] if self._plan else {}
        refs = self._plan['references'] if self._plan else {}
        self.fig.clear()
        axes = [self.fig.add_subplot(1, 2, k) for k in (1, 2)]
        for ax, key, title in ((axes[0], 'before', 'Before'),
                               (axes[1], 'corrected', 'After')):
            frames = res[key]
            for b, c in zip(batches, colors, strict=True):
                use = ([refs[b]] if refs.get(b) else
                       [n for n, bb in batch_of.items() if bb == b])
                vals = np.concatenate([frames[n][ch].to_numpy(dtype=float)
                                       for n in use if n in frames])
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    ax.hist(vals, bins=120, histtype='step', color=c,
                            density=True, label=str(b))
            ax.set_title(f"{title} ({'reference' if refs else 'pooled'})",
                         fontsize=9)
            ax.set_xlabel(ch, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[1].legend(fontsize=7, loc='best')
        try:
            self.fig.tight_layout()
        except Exception:                                # noqa: BLE001
            pass
        self.canvas.draw_idle()
        lines = [f"{ch}: median before → after"]
        for _i, r in sub.iterrows():
            lines.append(f"  {r['batch']}: {r['median_before']:.4g} → "
                         f"{r['median_after']:.4g}  (spread "
                         f"{r['spread_before']:.3g} → "
                         f"{r['spread_after']:.3g})")
        lines += ['', res['methods']]
        if res['written']:
            lines.append('')
            lines.append("Copies: " + ', '.join(
                os.path.basename(w) for w in res['written']))
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', '\n'.join(lines))
        self.text.configure(state='disabled')

    def _restore(self):
        n = self.editor._restore_batch_norm()
        self.status_var.set(f"Restored the original values of {n} "
                            "sample(s)." if n else "Nothing to restore.")
        self._preview()

    def _copy_methods(self):
        if self._result is None:
            return
        self.clipboard_clear()
        self.clipboard_append(self._result['methods'])
        self.status_var.set("Copied the methods sentence.")
