"""Spectral-unmixing QC: similarity + spillover-spread heatmaps.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    plt_get_cmap,
    savefig_background,
)
from .tool_window import background_selector, figure_panel


class SpectralQCWindow(tk.Toplevel):
    """Spectral-unmixing quality view: the spectral SIMILARITY matrix and the
    Spillover Spread Matrix (SSM) as heatmaps, the condition number, and the
    flagged similar / high-spread fluor pairs — with Markdown / PNG export."""

    def __init__(self, parent, qc, audit=None):
        super().__init__(parent)
        self.title("Spectral QC (unmixing diagnostics)")
        self.geometry("960x680")
        self._qc = qc
        self._audit = audit

        bar = ttk.Frame(self)
        bar.pack(fill='x', side='top')
        cond = qc.get('condition_number', float('nan'))
        cond_txt = "∞" if cond == float('inf') else f"{cond:.1f}"
        warn = "  [ill-conditioned]" if (cond == float('inf') or cond > 100) \
            else ""
        ttk.Label(
            bar,
            text=(f"{len(qc['fluors'])} fluors · condition number {cond_txt}"
                  f"{warn} · {len(qc['similar_pairs'])} similar pair(s)"),
            font=('TkDefaultFont', 9, 'bold')).pack(side='left', padx=6, pady=4)
        self._bg_var, _bg_combo = background_selector(bar)
        ttk.Button(bar, text="Export PNG…",
                   command=self._export_png).pack(side='right', padx=(0, 6),
                                                  pady=4)
        _bg_combo.pack(side='right', padx=(0, 4), pady=4)
        ttk.Label(bar, text="PNG background:").pack(side='right', padx=(0, 2))
        ttk.Button(bar, text="Export Markdown…",
                   command=self._export_md).pack(side='right', padx=(0, 4),
                                                 pady=4)

        # Build the empty canvas, draw the heatmaps into the figure, theme, then
        # render — dark=False so figure_panel doesn't theme the empty figure
        # before _draw_heatmaps adds the axes.
        _frame, self._fig, canvas = figure_panel(
            self, figsize=(9, 4.2), dpi=100, dark=False)
        self._draw_heatmaps(self._fig)
        if _dialog_dark_on(self):
            _theme_figure_dark(self._fig)
        canvas.draw()

        # Flagged pairs as plain text underneath.
        txt = tk.Text(self, height=8, wrap='word')
        txt.pack(fill='x', side='bottom')
        txt.insert('1.0', self._summary_text())
        txt.configure(state='disabled')

    def _draw_heatmaps(self, fig):
        import numpy as _np
        fluors = self._qc['fluors']
        n = len(fluors)
        sim = _np.asarray(self._qc['similarity'], dtype=float)
        ssm = _np.asarray(self._qc['ssm'], dtype=float)
        ssm_masked = _np.ma.masked_invalid(ssm)

        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(sim, vmin=0.0, vmax=1.0, cmap='magma',
                         aspect='auto')
        ax1.set_title('Spectral similarity', fontsize=9)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        cmap = plt_get_cmap('viridis')
        cmap.set_bad('lightgrey')
        im2 = ax2.imshow(ssm_masked, cmap=cmap, aspect='auto')
        ax2.set_title('Spillover spread (SSM)', fontsize=9)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for ax in (ax1, ax2):
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(fluors, rotation=90, fontsize=7)
            ax.set_yticklabels(fluors, fontsize=7)
        try:
            fig.tight_layout()
        except Exception:
            pass

    def _summary_text(self):
        lines = []
        sp = self._qc['similar_pairs']
        if sp:
            lines.append("Spectrally-similar pairs (hard to resolve):")
            for d in sp[:10]:
                lines.append(f"  • {d['fluor_a']} ~ {d['fluor_b']}  "
                             f"(cosine {d['similarity']:.3f})")
        else:
            lines.append("No fluor pair exceeds the similarity threshold — "
                         "spectra are well separated.")
        ws = self._qc['worst_spread']
        if ws:
            lines.append("")
            lines.append("Largest spillover spread (into ← from):")
            for d in ws:
                lines.append(f"  • {d['into']} ← {d['from']}  "
                             f"({d['spread']:.3g})")
        return "\n".join(lines)

    def _markdown(self):
        from datetime import datetime

        from . import __version__
        q = self._qc
        cond = q['condition_number']
        cond_txt = "inf" if cond == float('inf') else f"{cond:.2f}"
        out = ["# Spectral unmixing QC", ""]
        out.append(f"- **openflo_version**: {__version__}")
        out.append(f"- **exported**: "
                   f"{datetime.now().isoformat(timespec='seconds')}")
        out.append(f"- **fluors**: {len(q['fluors'])}")
        out.append(f"- **condition_number**: {cond_txt}")
        out.append("")
        out.append("## Spectrally-similar pairs")
        if q['similar_pairs']:
            out.append("| Fluor A | Fluor B | Cosine similarity |")
            out.append("|---|---|---|")
            for d in q['similar_pairs']:
                out.append(f"| {d['fluor_a']} | {d['fluor_b']} | "
                           f"{d['similarity']:.4f} |")
        else:
            out.append("None above threshold.")
        out.append("")
        out.append("## Largest spillover spread")
        if q['worst_spread']:
            out.append("| Into | From | Spread |")
            out.append("|---|---|---|")
            for d in q['worst_spread']:
                out.append(f"| {d['into']} | {d['from']} | "
                           f"{d['spread']:.4g} |")
        else:
            out.append("No measured spread (single-stain controls missing).")
        out.append("")
        return "\n".join(out)

    def _export_md(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export spectral QC", defaultextension='.md',
            initialfile='spectral_qc.md',
            filetypes=[('Markdown', '*.md'), ('All files', '*.*')])
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self._markdown())
        except Exception as exc:
            messagebox.showerror("Spectral QC",
                                 f"Export failed:\n{exc}", parent=self)
            return
        if self._audit:
            self._audit('spectral.qc.export', path=path)
        messagebox.showinfo("Spectral QC", f"Exported:\n{path}", parent=self)

    def _export_png(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Export spectral QC figure",
            defaultextension='.png', initialfile='spectral_qc.png',
            filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'),
                       ('SVG', '*.svg')])
        if not path:
            return
        bg = self._bg_var.get()
        try:
            savefig_background(self._fig, path, background=bg, dpi=300)
        except Exception as exc:
            messagebox.showerror("Spectral QC",
                                 f"Export failed:\n{exc}", parent=self)
            return
        if self._audit:
            self._audit('spectral.qc.export', path=path, background=bg)
        messagebox.showinfo("Spectral QC", f"Exported:\n{path}", parent=self)
