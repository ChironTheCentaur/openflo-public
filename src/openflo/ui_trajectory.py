"""Pseudotime / trajectory analysis window.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .async_task import BusyStrip
from .dialog_inputs import read_int
from .errors import note
from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    current_palette,
    savefig_background,
)
from .tool_window import background_selector, figure_panel
from .ui_notice import report_saved
from .ui_tips import auto_tip


class TrajectoryWindow(tk.Toplevel):
    """Pseudotime / trajectory inference.

    Builds a geodesic pseudotime over the enabled samples' shared fluor
    channels (concatenated, so a day-series becomes one continuous trajectory),
    rooted at the extreme of a chosen marker (e.g. CD902-high = most primitive),
    writes a ``pseudotime`` column back to every sample (selectable as a plot
    colour), and draws each marker's mean expression along pseudotime — the
    CD902-down / CD901b-up maturation curve. Exports the trends as a CSV / Prism XY
    table and the figure."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Trajectory / pseudotime")
        self.geometry("960x680")
        self._centers = None
        self._means = None              # display scale: the drawn curves
        self._means_linear = None       # linear: the exported table
        self._channels = []
        self._ran_on = []               # the samples the trajectory ran on

        chans = self._shared_channels()
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Root marker:").pack(side='left')
        self.root_var = tk.StringVar()
        disp = [editor._fmt_channel(c) for c in chans]
        self.root_combo = ttk.Combobox(ctl, textvariable=self.root_var,
                                       width=22, state='readonly', values=disp)
        self.root_combo.pack(side='left', padx=(2, 8))
        auto_tip(self.root_combo,
                 "Marker defining where the trajectory STARTS. "
                 "Pseudotime is measured outward from the cells at "
                 "this marker's extreme, so a different root "
                 "reorders the whole trajectory.")
        # Default to a stemness-ish marker if present (CD902), else first.
        default = next((editor._fmt_channel(c) for c in chans
                        if 'cd902' in (editor._channel_labels.get(c, c)).lower()),
                       disp[0] if disp else '')
        self.root_var.set(default)
        ttk.Label(ctl, text="Root end:").pack(side='left')
        self.dir_var = tk.StringVar(value='High')
        _dc = ttk.Combobox(ctl, textvariable=self.dir_var, width=6,
                           state='readonly', values=['High', 'Low'])
        _dc.pack(side='left', padx=(2, 8))
        auto_tip(_dc,
                 "Which end of the root marker is the start: 'High' begins at "
                 "the brightest cells, 'Low' at the dimmest. Pick the end "
                 "matching the biology -- the progenitor, not the endpoint.")
        ttk.Label(ctl, text="Neighbors:").pack(side='left')
        self.k_var = tk.StringVar(value='15')
        _sp = ttk.Spinbox(ctl, from_=5, to=50, width=5,
                          textvariable=self.k_var)
        _sp.pack(side='left', padx=(2, 8))
        auto_tip(_sp,
                 "Neighbours used to build the cell-to-cell graph. Higher "
                 "gives a smoother, more connected trajectory; lower keeps "
                 "fine structure but can fragment it into disconnected "
                 "pieces.")
        _cb = self._compute_btn = ttk.Button(ctl, text="Compute",
                                             command=self._compute)
        _cb.pack(side='left')
        auto_tip(_cb,
                 "Build the graph and order cells along it. Cost grows with "
                 "event count, so this can take a while on a large sample.")
        self.status = ttk.Label(ctl, text="", foreground=current_palette()['muted'])
        self.status.pack(side='left', padx=(8, 0))

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        _x1 = ttk.Button(exp, text="Trends CSV…",
                         command=lambda: self._export('tidy'))
        _x1.pack(side='left')
        auto_tip(_x1,
                 "Long-form table of marker intensity against pseudotime, one "
                 "row per bin per marker -- for R, pandas or any stats tool.")
        _x2 = ttk.Button(exp, text="Prism XY…",
                         command=lambda: self._export('prism'))
        _x2.pack(side='left', padx=(4, 0))
        auto_tip(_x2,
                 "GraphPad Prism XY layout: pseudotime as X, one Y column per "
                 "marker. Paste into an XY table to plot the trends directly.")
        _x3 = ttk.Button(exp, text="Export figure…",
                         command=lambda: self._export('figure'))
        _x3.pack(side='left', padx=(4, 0))
        auto_tip(_x3, "Save this plot as a publication figure (PNG/PDF/SVG) "
                      "at export resolution.")
        self.bg_var, _bg_combo = background_selector(exp)
        _bg_combo.pack(side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

        # The computation runs off the window thread, shown here with Cancel.
        self._busy = BusyStrip(self, padding=(6, 2))
        self._busy.pack(fill='x')

        # dark=False: themed in _draw after rendering (preserves prior behaviour).
        _frame, self._fig, self._canvas = figure_panel(
            self, figsize=(9, 4.8), dpi=100, dark=False)
        if not chans:
            self.status.configure(
                text="No shared fluor channels across the enabled samples.")

    def _samples(self):
        names = self.editor._selected_samples() or (
            [self.editor._active_sample] if self.editor._active_sample else [])
        return [n for n in names if n in self.editor._samples]

    def _shared_channels(self):
        shared = None
        for n in self._samples():
            s = self.editor._samples[n]
            cols = set(getattr(s, 'fluor_channels', []) or []) & set(s.data.columns)
            shared = cols if shared is None else (shared & cols)
        # Preserve the first sample's channel order.
        if not shared:
            return []
        first = self.editor._samples[self._samples()[0]]
        return [c for c in first.fluor_channels if c in shared]

    def _linear_matrix(self, name, chans):
        """``(n_events, len(chans))`` compensated linear values of sample
        `name`; NaN for a channel whose display scale is unknown."""
        import numpy as _np

        from .pipeline import UnknownScaleError, linear_values
        s = self.editor._samples[name]
        cols = []
        for c in chans:
            try:
                cols.append(linear_values(s, c))
            except UnknownScaleError:
                cols.append(_np.full(len(s.data), _np.nan))
        return _np.column_stack(cols)

    def _compute(self):
        """Pseudotime over the enabled samples, computed off the window
        thread (it builds a neighbour graph over every event); the result
        is written back on the window thread when it arrives."""
        from .trajectory import compute_pseudotime, pseudotime_trends
        names = self._samples()
        chans = self._shared_channels()
        root = self.editor._resolve_channel(self.root_var.get())
        if not names or not chans or root not in chans:
            self.status.configure(text="Need enabled samples + a root marker.")
            return
        k = read_int(self.k_var, "Neighbors", lo=5, hi=50,
                        parent=self)
        if k is None:
            return
        high = self.dir_var.get() == 'High'
        frames = {n: self.editor._samples[n].data for n in names}
        self.status.configure(text="")

        def work(job):
            import numpy as _np
            mats, bounds, pos = [], [], 0
            for n in names:
                m = frames[n][chans].to_numpy(dtype=float)
                mats.append(m)
                bounds.append((n, pos, pos + len(m)))
                pos += len(m)
            X = _np.vstack(mats) if mats else _np.empty((0, len(chans)))
            score = X[:, chans.index(root)]
            job.check()
            pt, _ = compute_pseudotime(X, score, high=high, n_neighbors=k)
            job.check()
            # The graph and the drawn curves use the display values; the
            # exported per-bin means are of the compensated LINEAR values,
            # the intensity the trends table claims to hold. The display
            # means were exported as that intensity: a CD902 trend falling
            # 9,860 -> -160 was written as 0.682 -> -0.037 (logicle).
            lin = _np.vstack([self._linear_matrix(n, chans) for n in names])
            centers, means = pseudotime_trends(pt, X, n_bins=20)
            _c, means_linear = pseudotime_trends(pt, lin, n_bins=20)
            return pt, bounds, centers, means, means_linear, len(X)

        def done(res):
            pt, bounds, centers, means, means_linear, n_cells = res
            if any(self.editor._samples.get(n) is None
                   or self.editor._samples[n].data is not frames[n]
                   for n in names):
                self.status.configure(
                    text="The samples changed while computing; nothing was "
                         "written. Compute again.")
                return
            # Write the pseudotime column back to each sample.
            from .result_stamp import stamp_columns
            for n, a, b in bounds:
                self.editor._samples[n].data['pseudotime'] = pt[a:b]
                stamp_columns(self.editor._samples[n], ['pseudotime'],
                              'trajectory', [f'root={root}'] + list(chans))
            self.editor._data_changed([n for n, _a, _b in bounds])
            self._centers, self._means = centers, means
            self._means_linear = means_linear
            self._channels = chans
            self._ran_on = list(names)
            self.status.configure(
                text=f"Done — {n_cells:,} cells across {len(names)} "
                     "sample(s). 'pseudotime' is now a plot color.")
            self.editor._refresh_channel_choices()
            self.editor._audit('trajectory', samples=names,
                               root=root, root_end=self.dir_var.get(),
                               n_neighbors=k, n_cells=int(n_cells))
            self._draw()

        def failed(exc):
            self.status.configure(text=f"Failed: {type(exc).__name__}: {exc}")

        def cancelled():
            self.status.configure(text="Cancelled; nothing was written.")

        self._busy.run(work, on_done=done, on_error=failed,
                       on_cancel=cancelled, text="Computing pseudotime…",
                       disable=[self._compute_btn])

    def _draw(self):
        import numpy as _np
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        means, centers = self._means, self._centers
        if means is not None and centers is not None:
            for j, ch in enumerate(self._channels):
                col = means[:, j]
                finite = _np.isfinite(col)
                if not finite.any():
                    continue
                lo, hi = _np.nanmin(col), _np.nanmax(col)
                norm = (col - lo) / (hi - lo) if hi > lo else col * 0
                ax.plot(centers[finite], norm[finite], marker='o', ms=3,
                        lw=1.4, label=self.editor._fmt_channel(ch))
            ax.set_xlabel('pseudotime')
            ax.set_ylabel('expression, display scale (per-marker min–max '
                          'normalized)')
            ax.set_title('Marker trends along pseudotime')
            if len(self._channels) <= 12:
                ax.legend(fontsize=7, framealpha=0.85, loc='best')
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _trends_frame(self):
        """Per-bin mean of each marker's compensated linear values, the
        column named for its scale as the Statistics window names it."""
        import pandas as pd
        means = self._means_linear
        if means is None or self._centers is None:
            return pd.DataFrame()
        data = {'pseudotime': self._centers}
        for j, ch in enumerate(self._channels):
            data[f'{self.editor._fmt_channel(ch)} mean (linear)'] = means[:, j]
        return pd.DataFrame(data)

    def _export(self, kind):
        if self._means is None:
            messagebox.showinfo("Trajectory", "Compute a trajectory first.",
                                parent=self)
            return
        if kind == 'figure':
            path = filedialog.asksaveasfilename(
                parent=self, defaultextension='.png', initialfile='trajectory.png',
                filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
            if path:
                savefig_background(self._fig, path, background=self.bg_var.get())
                self._done(path, 'figure')
            return
        # 'tidy' and 'prism' are the same shape here (a Prism XY table: the bin
        # centre column + one mean column per marker) — both paste into Prism XY.
        name = ('prism_xy.csv' if kind == 'prism'
                else 'trajectory_trends.csv')
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv', initialfile=name,
            filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if path:
            self._trends_frame().to_csv(path, index=False)
            self._done(path, kind)

    def _done(self, path, kind):
        try:
            self.editor._audit('trajectory.export', path=path, kind=kind,
                               samples=self._ran_on)
        except Exception as exc:
            note("recording the export in the audit trail", exc)
        report_saved(self, path, "Exported")
