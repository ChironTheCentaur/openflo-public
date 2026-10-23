"""Spectral-unmixing QC: similarity + spillover-spread heatmaps.

Self-contained Tk window extracted from gui.py (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk

from .theme import (
    _dialog_dark_on,
    _theme_figure_dark,
    plt_get_cmap,
    savefig_background,
)
from .tool_window import background_selector, figure_panel
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import auto_tip


def fluor_ticks(ax, fluors):
    """Label both axes of a fluor-by-fluor heatmap with the fluor names."""
    ax.set_xticks(range(len(fluors)))
    ax.set_yticks(range(len(fluors)))
    ax.set_xticklabels(fluors, rotation=90, fontsize=7)
    ax.set_yticklabels(fluors, fontsize=7)


def draw_similarity(fig, ax, sim, fluors, title='Spectral similarity'):
    """The cosine-similarity heatmap (0-1, magma) with its colour bar, as
    Spectral QC and the panel check (ui_panel_check) draw it."""
    im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap='magma', aspect='auto')
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('fluor', fontsize=8)
    ax.set_ylabel('fluor', fontsize=8)
    fluor_ticks(ax, fluors)
    return im


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
        n_dim = sum(1 for d in (qc.get('reference_spectra') or [])
                    if d.get('warning'))
        dim_txt = f" · [!] {n_dim} control(s) too dim" if n_dim else ""
        ttk.Label(
            bar,
            text=(f"{len(qc['fluors'])} fluors · condition number {cond_txt}"
                  f"{warn} · {len(qc['similar_pairs'])} similar pair(s)"
                  f"{dim_txt}"),
            font='OpenFloHeading').pack(side='left', padx=6, pady=4)
        self._bg_var, _bg_combo = background_selector(bar)
        _sp = ttk.Button(bar, text="Export PNG…",
                         command=self._export_png)
        _sp.pack(side='right', padx=(0, 6), pady=4)
        auto_tip(_sp, "Save the similarity heatmap as an image.")
        _bg_combo.pack(side='right', padx=(0, 4), pady=4)
        ttk.Label(bar, text="PNG background:").pack(side='right', padx=(0, 2))
        _sm = ttk.Button(bar, text="Export Markdown…",
                         command=self._export_md)
        _sm.pack(side='right', padx=(0, 4), pady=4)
        if hasattr(parent, '_open_unmixed_pairs'):
            _pp = ttk.Button(bar, text="Unmixed pairs…",
                             command=parent._open_unmixed_pairs)
            _pp.pack(side='right', padx=(0, 4), pady=4)
            auto_tip(_pp, "Every pair of unmixed fluors of the first "
                          "unmixed sample, to spot a wrong spectrum.")
        auto_tip(_sm,
                 "Write the QC report: spectral similarity between "
                 "fluorophores and which pairs are too alike to separate "
                 "reliably -- the panel-design record.")

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
        sim = _np.asarray(self._qc['similarity'], dtype=float)
        ssm = _np.asarray(self._qc['ssm'], dtype=float)
        ssm_masked = _np.ma.masked_invalid(ssm)

        ax1 = fig.add_subplot(1, 2, 1)
        draw_similarity(fig, ax1, sim, fluors)

        ax2 = fig.add_subplot(1, 2, 2)
        cmap = plt_get_cmap('viridis')
        cmap.set_bad('lightgrey')
        # qc['ssm'][i, j] is the spread INTO fluor i FROM stain j. The usual
        # SSM layout (Nguyen 2013, SpectroFlo, FlowJo) puts the single stain
        # on the rows, so it is drawn transposed, and both axes say which is
        # which: unlabelled, the matrix read the wrong way round.
        im2 = ax2.imshow(ssm_masked.T, cmap=cmap, aspect='auto')
        ax2.set_title('Spillover spread (SSM)', fontsize=9)
        ax2.set_ylabel('single stain (spread from)', fontsize=8)
        ax2.set_xlabel('unmixed fluor (spread into)', fontsize=8)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        fluor_ticks(ax2, fluors)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
            pass

    def _summary_text(self):
        lines = []
        # A fluor with no usable reference spectrum has no similarity to
        # anything, so it can never appear among the similar pairs: without
        # this the window called such a panel "well separated".
        deg = list(self._qc.get('degenerate_fluors') or [])
        if deg:
            lines.append("No usable reference spectrum (cannot be unmixed): "
                         + ', '.join(deg))
        sp = self._qc['similar_pairs']
        if sp:
            lines.append("Spectrally-similar pairs (hard to resolve):")
            for d in sp[:10]:
                lines.append(f"  • {d['fluor_a']} ~ {d['fluor_b']}  "
                             f"(cosine {d['similarity']:.3f})")
        elif deg:
            lines.append("No other fluor pair exceeds the similarity "
                         "threshold.")
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
        # How each spectrum was estimated, and which controls are too dim
        # to define one: a contaminated spectrum looks like any other.
        ref = list(self._qc.get('reference_spectra') or [])
        if ref:
            from .spectral import reference_spectra_lines
            dim = [d['fluor'] for d in ref if d['warning']]
            lines.append("")
            lines.append("Reference spectra"
                         + (f" ([!] too dim: {', '.join(dim)})" if dim else "")
                         + ":")
            lines += reference_spectra_lines(ref)
        return "\n".join(lines)

    def _markdown(self):
        from datetime import datetime

        from .export_stamp import version
        q = self._qc
        cond = q['condition_number']
        cond_txt = "inf" if cond == float('inf') else f"{cond:.2f}"
        out = ["# Spectral unmixing QC", ""]
        out.append(f"- **openflo_version**: {version()}")
        out.append(f"- **exported**: "
                   f"{datetime.now().isoformat(timespec='seconds')}")
        out.append(f"- **fluors**: {len(q['fluors'])}")
        out.append(f"- **condition_number**: {cond_txt}")
        out.append("")
        deg = list(q.get('degenerate_fluors') or [])
        if deg:
            out.append("## Unusable reference spectra")
            out.append("No usable spectrum (all zero, e.g. a stain no "
                       "brighter than the unstained control), so these "
                       "cannot be unmixed: " + ', '.join(deg))
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
        ref = list(q.get('reference_spectra') or [])
        if ref:
            from .spectral import reference_spectra_markdown
            out += reference_spectra_markdown(ref)
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
            show_error(self, "Spectral QC",
                       "The report could not be exported.", exc, path)
            return
        if self._audit:
            self._audit('spectral.qc.export', path=path)
        report_saved(self, path, "Exported")

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
            savefig_background(self._fig, path, background=bg)
        except Exception as exc:
            show_error(self, "Spectral QC",
                       "The figure could not be exported.", exc, path)
            return
        if self._audit:
            self._audit('spectral.qc.export', path=path, background=bg)
        report_saved(self, path, "Exported")
