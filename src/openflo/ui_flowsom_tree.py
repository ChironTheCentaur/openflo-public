"""FlowSOM minimum-spanning-tree viewer.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    plt_get_cmap,
    savefig_background,
)
from .tool_window import background_selector, figure_panel


class FlowSOMTreeWindow(tk.Toplevel):
    """The classic FlowSOM star-tree: SOM nodes laid out on their minimal
    spanning tree, each drawn as a star glyph of its marker profile and
    coloured by metacluster, node size ∝ event count."""

    def __init__(self, editor, sample):
        super().__init__(editor)
        self.editor = editor
        self.sample = sample
        self.title(f"FlowSOM star tree — {sample}")
        self.geometry("900x780")
        s = editor._samples[sample]
        res = s.flowsom_result
        self._W = np.asarray(res['weights'], dtype=float)
        self._channels = list(res['channels'])
        self._n_meta = int(res.get('n_metaclusters', 1))

        df = s.data
        node = df['flowsom'].to_numpy()
        meta = df['flowsom_meta'].to_numpy()
        nn = len(self._W)
        self._counts = np.bincount(node[node >= 0], minlength=nn)
        node_meta = np.full(nn, -1, dtype=int)
        for nd in range(nn):
            mm = meta[node == nd]
            mm = mm[mm >= 0]
            if mm.size:
                node_meta[nd] = int(np.bincount(mm).argmax())
        self._node_meta = node_meta

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        ttk.Label(bar, padding=6,
                  text=f"{nn} SOM nodes · {self._n_meta} metaclusters · "
                       f"{len(self._channels)} markers",
                  font=('TkDefaultFont', 9, 'bold')).pack(side='left')
        ttk.Button(bar, text="Export figure…", command=self._export).pack(
            side='right', padx=6, pady=4)
        self.bg_var, _bg_combo = background_selector(bar)
        _bg_combo.pack(side='right')

        # dark=False: themed in _draw after rendering (preserves prior behaviour).
        _frame, self._fig, self._canvas = figure_panel(
            self, figsize=(8.5, 7.0), dpi=100, dark=False)
        self._draw()

    def _draw(self):
        import matplotlib.patches as mpatches

        from .pipeline import flowsom_layout, flowsom_mst
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        W = self._W
        edges, _ = flowsom_mst(W)
        pos = flowsom_layout(len(W), edges)
        if len(pos) == 0:
            if _dialog_dark_on(self):
                _theme_figure_dark(fig)
            self._canvas.draw()
            return
        # Per-channel scale of the prototypes to [0, 1] for the star spokes.
        lo = W.min(0)
        rng = W.max(0) - lo
        rng[rng == 0] = 1.0
        Ws = (W - lo) / rng
        extent = float(np.max(pos.max(0) - pos.min(0))) or 1.0
        base_r = extent * 0.045
        cmap = plt_get_cmap('tab20')

        for i, j in edges:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color='#cccccc', lw=0.8, zorder=1)

        M = len(self._channels)
        angs = np.linspace(0, 2 * np.pi, M, endpoint=False)
        cmax = float(self._counts.max()) or 1.0
        for nd in range(len(W)):
            x, y = pos[nd]
            scale = base_r * (0.45 + 1.4 * np.sqrt(self._counts[nd] / cmax))
            r = scale * (0.25 + 0.75 * Ws[nd])
            xs = x + r * np.cos(angs)
            ys = y + r * np.sin(angs)
            color = cmap((self._node_meta[nd] % 20) / 20.0) \
                if self._node_meta[nd] >= 0 else '#999999'
            ax.fill(xs, ys, color=color, alpha=0.85, zorder=3,
                    edgecolor='black', lw=0.3)

        # Reference star (marker → spoke) in the corner.
        rx, ry = pos[:, 0].min(), pos[:, 1].max()
        for k, ch in enumerate(self._channels):
            ax.plot([rx, rx + base_r * 1.5 * np.cos(angs[k])],
                    [ry, ry + base_r * 1.5 * np.sin(angs[k])],
                    color='#666', lw=0.6, zorder=2)
            ax.text(rx + base_r * 1.9 * np.cos(angs[k]),
                    ry + base_r * 1.9 * np.sin(angs[k]),
                    self.editor._fmt_channel(ch).split(' (')[0],
                    fontsize=6, ha='center', va='center', color='#444')
        # Metacluster legend.
        handles = [mpatches.Patch(color=cmap((m % 20) / 20.0), label=f"mc {m}")
                   for m in sorted(set(self._node_meta[self._node_meta >= 0]))]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc='lower right',
                      framealpha=0.85, ncol=2, title='metacluster')
        ax.set_title(f"FlowSOM star tree — {self.sample}", fontsize=10)
        ax.autoscale_view()
        try:
            fig.tight_layout()
        except Exception:
            pass
        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        self._canvas.draw()

    def _export(self):
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.png', initialfile='flowsom_tree.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')])
        if not path:
            return
        savefig_background(self._fig, path, background=self.bg_var.get())
        try:
            self.editor._audit('flowsom_tree.export', path=path,
                               sample=self.sample)
        except Exception:
            pass
        messagebox.showinfo("FlowSOM tree", f"Exported:\n{path}", parent=self)
