"""Pseudotime / trajectory analysis window.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    savefig_background,
)
from .tool_window import background_selector, figure_panel


class TrajectoryWindow(tk.Toplevel):
    """Pseudotime / trajectory inference.

    Builds a geodesic pseudotime over the enabled samples' shared fluor
    channels (concatenated, so a day-series becomes one continuous trajectory),
    rooted at the extreme of a chosen marker (e.g. CD34-high = most primitive),
    writes a ``pseudotime`` column back to every sample (selectable as a plot
    colour), and draws each marker's mean expression along pseudotime — the
    CD34-down / CD11b-up maturation curve. Exports the trends as a CSV / Prism XY
    table and the figure."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Trajectory / pseudotime")
        self.geometry("960x680")
        self._centers = None
        self._means = None
        self._channels = []

        chans = self._shared_channels()
        ctl = ttk.Frame(self, padding=6)
        ctl.pack(fill='x', side='top')
        ttk.Label(ctl, text="Root marker:").pack(side='left')
        self.root_var = tk.StringVar()
        disp = [editor._fmt_channel(c) for c in chans]
        self.root_combo = ttk.Combobox(ctl, textvariable=self.root_var,
                                       width=22, state='readonly', values=disp)
        self.root_combo.pack(side='left', padx=(2, 8))
        # Default to a stemness-ish marker if present (CD34), else first.
        default = next((editor._fmt_channel(c) for c in chans
                        if 'cd34' in (editor._channel_labels.get(c, c)).lower()),
                       disp[0] if disp else '')
        self.root_var.set(default)
        ttk.Label(ctl, text="Root end:").pack(side='left')
        self.dir_var = tk.StringVar(value='High')
        ttk.Combobox(ctl, textvariable=self.dir_var, width=6, state='readonly',
                     values=['High', 'Low']).pack(side='left', padx=(2, 8))
        ttk.Label(ctl, text="Neighbors:").pack(side='left')
        self.k_var = tk.StringVar(value='15')
        ttk.Spinbox(ctl, from_=5, to=50, width=5,
                    textvariable=self.k_var).pack(side='left', padx=(2, 8))
        ttk.Button(ctl, text="Compute", command=self._compute).pack(side='left')
        self.status = ttk.Label(ctl, text="", foreground='#555')
        self.status.pack(side='left', padx=(8, 0))

        exp = ttk.Frame(self, padding=(6, 0))
        exp.pack(fill='x')
        ttk.Button(exp, text="Trends CSV…",
                   command=lambda: self._export('tidy')).pack(side='left')
        ttk.Button(exp, text="Prism XY…",
                   command=lambda: self._export('prism')).pack(
            side='left', padx=(4, 0))
        ttk.Button(exp, text="Figure…",
                   command=lambda: self._export('figure')).pack(
            side='left', padx=(4, 0))
        self.bg_var, _bg_combo = background_selector(exp)
        _bg_combo.pack(side='right')
        ttk.Label(exp, text="Fig background:").pack(side='right', padx=(0, 2))

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

    def _compute(self):
        from .trajectory import compute_pseudotime, pseudotime_trends
        names = self._samples()
        chans = self._shared_channels()
        root = self.editor._resolve_channel(self.root_var.get())
        if not names or not chans or root not in chans:
            self.status.configure(text="Need enabled samples + a root marker.")
            return
        self.status.configure(text="Computing pseudotime…")
        self.update_idletasks()
        self.configure(cursor='watch')
        try:
            import numpy as _np
            mats, bounds, pos = [], [], 0
            for n in names:
                df = self.editor._samples[n].data
                m = df[chans].to_numpy(dtype=float)
                mats.append(m)
                bounds.append((n, pos, pos + len(m)))
                pos += len(m)
            X = _np.vstack(mats) if mats else _np.empty((0, len(chans)))
            score = X[:, chans.index(root)]
            try:
                k = int(self.k_var.get())
            except ValueError:
                k = 15
            pt, _ = compute_pseudotime(X, score, high=(self.dir_var.get() ==
                                       'High'), n_neighbors=k)
            # Write the pseudotime column back to each sample.
            for n, a, b in bounds:
                self.editor._samples[n].data['pseudotime'] = pt[a:b]
            self._centers, self._means = pseudotime_trends(pt, X, n_bins=20)
            self._channels = chans
        except Exception as exc:
            self.status.configure(text=f"Failed: {type(exc).__name__}: {exc}")
            self.configure(cursor='')
            return
        self.configure(cursor='')
        self.status.configure(
            text=f"Done — {len(X):,} cells across {len(names)} sample(s). "
                 "'pseudotime' is now a plot colour.")
        self.editor._refresh_channel_choices()
        self.editor._audit('trajectory', samples=names,
                           root=root, root_end=self.dir_var.get(),
                           n_neighbors=k, n_cells=int(len(X)))
        self._draw()

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
            ax.set_ylabel('expression (per-marker min–max normalized)')
            ax.set_title('Marker trends along pseudotime')
            if len(self._channels) <= 12:
                ax.legend(fontsize=7, framealpha=0.85, loc='best')
        try:
            fig.tight_layout()
        except Exception:
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _trends_frame(self):
        import pandas as pd
        means = self._means
        if means is None or self._centers is None:
            return pd.DataFrame()
        data = {'pseudotime': self._centers}
        for j, ch in enumerate(self._channels):
            data[self.editor._fmt_channel(ch)] = means[:, j]
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
            self.editor._audit('trajectory.export', path=path, kind=kind)
        except Exception:
            pass
        messagebox.showinfo("Trajectory", f"Exported:\n{path}", parent=self)
