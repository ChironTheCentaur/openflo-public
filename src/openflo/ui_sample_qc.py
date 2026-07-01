"""Sample QC: EMD + MDS sample-similarity diagnostics.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import _dialog_dark_on, _theme_figure_dark, plt_get_cmap
from .tool_window import ask_csv_path, background_selector, export_figure, figure_panel


class SampleQCWindow(tk.Toplevel):
    """Cross-sample QC: an Earth-Mover's-distance similarity matrix between the
    enabled samples + an MDS embedding (batch effects / outlier samples show up
    as separated points). Exports the distance matrix, the figure, and an
    AnnData ``.h5ad`` for the scanpy ecosystem."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Sample QC — similarity & MDS")
        self.geometry("1000x640")
        self._names = []
        self._D = None

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        ttk.Label(bar, padding=6, text="EMD sample distance + MDS",
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left')
        ttk.Button(bar, text="Export AnnData (.h5ad)…",
                   command=self._export_h5ad).pack(side='right', padx=(0, 6),
                                                   pady=4)
        ttk.Button(bar, text="Distance CSV…",
                   command=self._export_csv).pack(side='right', padx=(0, 4),
                                                  pady=4)
        ttk.Button(bar, text="Figure…",
                   command=self._export_fig).pack(side='right', padx=(0, 4),
                                                  pady=4)
        self.bg_var, _bg_combo = background_selector(bar)
        _bg_combo.pack(side='right', padx=(0, 4))

        # dark=False: the empty figure isn't themed at creation; _draw() applies
        # the dark theme after rendering (matching the prior behaviour).
        _frame, self._fig, self._canvas = figure_panel(
            self, figsize=(10, 4.8), dpi=100, dark=False)
        self.after(50, self._compute)

    def _samples(self):
        return self.editor._selected_samples()

    def _markers(self):
        names = self._samples()
        shared = None
        for n in names:
            s = self.editor._samples[n]
            cols = set(getattr(s, 'fluor_channels', []) or []) & set(
                s.data.columns)
            shared = cols if shared is None else (shared & cols)
        first = self.editor._samples[names[0]]
        return [c for c in first.fluor_channels if c in (shared or set())]

    def _compute(self):
        from .interop import mds_embed, sample_distance_matrix
        names = self._samples()
        markers = self._markers()
        if len(names) < 2 or not markers:
            return
        data = {n: self.editor._samples[n].data for n in names}
        self._names, self._D = sample_distance_matrix(data, markers)
        self._xy = mds_embed(self._D)
        self.editor._audit('sample_qc', n_samples=len(names),
                           n_markers=len(markers))
        self._draw()

    def _draw(self):
        import numpy as _np
        fig = self._fig
        fig.clear()
        if self._D is None:
            return
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(self._D, cmap='magma')
        ax1.set_xticks(range(len(self._names)))
        ax1.set_yticks(range(len(self._names)))
        short = [self.editor._short_sample(n, 14) for n in self._names]
        ax1.set_xticklabels(short, rotation=90, fontsize=6)
        ax1.set_yticklabels(short, fontsize=6)
        ax1.set_title('EMD sample distance', fontsize=9)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        xy = self._xy
        trials = [self.editor._sample_trial.get(n, '') for n in self._names]
        uniq = list(dict.fromkeys(trials))
        cmap = plt_get_cmap('tab10')
        for k, t in enumerate(uniq):
            m = _np.array([tr == t for tr in trials])
            ax2.scatter(xy[m, 0], xy[m, 1], s=40, label=str(t) or '—',
                        color=cmap(k % 10), zorder=3)
        for i, n in enumerate(self._names):
            ax2.annotate(self.editor._short_sample(n, 12),
                         (xy[i, 0], xy[i, 1]), fontsize=6,
                         xytext=(3, 3), textcoords='offset points')
        ax2.set_title('MDS of samples', fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])
        if len(uniq) > 1:
            ax2.legend(fontsize=7, framealpha=0.85, title='trial')
        try:
            fig.tight_layout()
        except Exception:
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _export_csv(self):
        if self._D is None:
            return
        import pandas as pd
        path = ask_csv_path(self, initialfile='sample_distance.csv')
        if path:
            pd.DataFrame(self._D, index=pd.Index(self._names),
                         columns=pd.Index(self._names)).to_csv(path)
            messagebox.showinfo("Sample QC", f"Exported:\n{path}", parent=self)

    def _export_fig(self):
        if self._D is None:
            return
        export_figure(self, self._fig, background=self.bg_var.get(),
                      initialfile='sample_qc.png')

    def _export_h5ad(self):
        from .interop import write_h5ad
        names = self._samples()
        markers = self._markers()
        if len(names) < 1 or not markers:
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.h5ad',
            initialfile='openflo_export.h5ad',
            filetypes=[('AnnData', '*.h5ad'), ('All files', '*.*')])
        if not path:
            return
        data = {n: self.editor._samples[n].data for n in names}
        obs_cols = ['leiden', 'cluster', 'flowsom_meta', 'pseudotime']
        try:
            n_obs = write_h5ad(path, data, markers, obs_cols=obs_cols)
        except ImportError as exc:
            messagebox.showwarning("AnnData export", str(exc), parent=self)
            return
        except Exception as exc:
            messagebox.showerror("AnnData export", f"Failed: {exc}",
                                 parent=self)
            return
        try:
            self.editor._audit('anndata.export', path=path, n_events=n_obs,
                               n_markers=len(markers))
        except Exception:
            pass
        messagebox.showinfo("AnnData export",
                            f"Wrote {n_obs:,} events → {path}", parent=self)
