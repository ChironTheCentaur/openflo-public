"""Panel check window: the predicted spectral similarity and complexity of a
panel before it is stained (openflo.panel_check).

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import csv
import os
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np

from . import panel_check as pc
from .errors import note
from .tool_window import (
    ask_csv_path,
    background_selector,
    export_figure,
    figure_panel,
)
from .ui_errors import show_error
from .ui_spectral_qc import draw_similarity, fluor_ticks
from .ui_tips import tip

MODEL_NOTE = ("Approximate: modelled from each fluor's peak wavelengths on a "
              "generic instrument. Spectra measured from single-stain "
              "controls are better (Load spectra CSV…).")

# A spectra CSV with at least this many detectors is scored as spectral.
SPECTRAL_DETECTORS = 20


def _num(text):
    try:
        v = float(str(text).strip())
    except ValueError:
        return None
    return v if np.isfinite(v) else None


class PanelCheckWindow(tk.Toplevel):
    """Detector set (or loaded spectra), the fluors of the panel, the
    similarity threshold; the predicted similarity heatmap (and spillover,
    on a conventional instrument), the complexity index, the flagged pairs
    and which fluor to leave out."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Panel check")
        self.geometry("1000x700")
        self.minsize(760, 560)
        self.report: dict | None = None
        # The loaded spectra: (signatures, fluors, detectors), with the
        # name it shows under in the source list.
        self._loaded: tuple[np.ndarray, list, list] | None = None
        self._loaded_label = ''
        self._panel: list[str] = []

        top = ttk.Frame(self, padding=(6, 6, 6, 2))
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Signatures from:").pack(side='left')
        self.source_var = tk.StringVar(master=self, value=pc.DEFAULT_SET)
        self.source_box = ttk.Combobox(
            top, textvariable=self.source_var, state='readonly', width=56,
            values=list(pc.DETECTOR_SETS))
        self.source_box.pack(side='left', padx=(2, 8))
        self.source_box.bind('<<ComboboxSelected>>',
                             lambda _e: self._source_changed())
        tip(editor, self.source_box,
            "A generic instrument, by its lasers and filters, on which each "
            "fluor's spectrum is modelled; or spectra you loaded. Choose the "
            "one closest to your cytometer.")
        _b = ttk.Button(top, text="Load spectra CSV…",
                        command=self._load_csv)
        _b.pack(side='left')
        tip(editor, _b,
            "Read reference signatures from a CSV: a header row (Fluor, then "
            "one detector per column) and one row per fluor. Measured "
            "spectra beat the model.")
        self.unmix_btn = ttk.Button(top, text="Use last unmix",
                                    command=self._use_last_unmix)
        self.unmix_btn.pack(side='left', padx=(6, 0))
        tip(editor, self.unmix_btn,
            "Check the reference spectra of this session's last spectral "
            "unmix, measured from your single-stain controls.")
        if self._last_unmix() is None:
            self.unmix_btn.state(['disabled'])

        opt = ttk.Frame(self, padding=(6, 0, 6, 4))
        opt.pack(side='top', fill='x')
        ttk.Label(opt, text="Flag pairs above similarity:").pack(side='left')
        self.thr_var = tk.StringVar(master=self, value='')
        _e = ttk.Entry(opt, textvariable=self.thr_var, width=6)
        _e.pack(side='left', padx=(2, 4))
        _e.bind('<Return>', lambda _e: self._refresh())
        tip(editor, _e,
            "Cosine similarity (0-1) above which two fluors are flagged as "
            "hard to tell apart. 0.98 is the usual rule on a spectral "
            "instrument; on a conventional one 0.90, as it sees each fluor "
            "in only a few filters.")
        _b = ttk.Button(opt, text="Check", default='active',
                        command=self._refresh)
        _b.pack(side='left', padx=(4, 12))
        tip(editor, _b, "Recompute with this threshold.")
        self.note_var = tk.StringVar(master=self, value='')
        ttk.Label(opt, textvariable=self.note_var, style='Muted.TLabel',
                  wraplength=640, justify='left').pack(side='left',
                                                       fill='x')

        btns = ttk.Frame(self, padding=(10, 0, 10, 8))
        btns.pack(side='bottom', fill='x')
        _b = ttk.Button(btns, text="Export similarity…",
                        command=lambda: self._export_matrix('similarity'))
        _b.pack(side='left')
        tip(editor, _b, "Save the similarity matrix as a CSV, fluors on "
                        "both axes.")
        self.spill_btn = ttk.Button(
            btns, text="Export spillover…",
            command=lambda: self._export_matrix('spillover'))
        self.spill_btn.pack(side='left', padx=(6, 0))
        tip(editor, self.spill_btn,
            "Save the predicted spillover as a CSV: each row a fluor, each "
            "column the share of it seen in that column's fluor's detector.")
        _b = ttk.Button(btns, text="Save signatures…",
                        command=self._save_signatures)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Save the panel's signatures as a spectra CSV that "
                        "Load spectra CSV reads back.")
        self.bg_var, _bg = background_selector(btns)
        _bg.pack(side='left', padx=(12, 4))
        _b = ttk.Button(btns, text="Export figure…",
                        command=self._export_figure)
        _b.pack(side='left')
        tip(editor, _b, "Save the heatmap as PNG, PDF or SVG.")
        ttk.Button(btns, text="Close", command=self.destroy).pack(
            side='right')

        body = ttk.Frame(self, padding=(6, 0, 6, 4))
        body.pack(side='top', fill='both', expand=True)
        left = ttk.Frame(body)
        left.pack(side='left', fill='y')
        ttk.Label(left, text="Search:").pack(anchor='w')
        self.search_var = tk.StringVar(master=self, value='')
        _s = ttk.Entry(left, textvariable=self.search_var, width=26)
        _s.pack(fill='x')
        self.search_var.trace_add('write', lambda *_a: self._fill_lists())
        tip(editor, _s, "Show only the fluors whose name contains this.")
        ttk.Label(left, text="Available (double-click to add):").pack(
            anchor='w', pady=(4, 0))
        self.avail = tk.Listbox(left, height=11, width=26,
                                selectmode='extended', exportselection=False)
        self.avail.pack(fill='both', expand=True)
        self.avail.bind('<Double-Button-1>', lambda _e: self._add())
        mid = ttk.Frame(left)
        mid.pack(fill='x', pady=4)
        _b = ttk.Button(mid, text="Add ↓", command=self._add)
        _b.pack(side='left')
        tip(editor, _b, "Put the selected fluors in the panel.")
        _b = ttk.Button(mid, text="Remove ↑", command=self._remove)
        _b.pack(side='left', padx=(6, 0))
        tip(editor, _b, "Take the selected fluors out of the panel.")
        ttk.Label(left, text="Panel:").pack(anchor='w')
        self.chosen = tk.Listbox(left, height=9, width=26,
                                 selectmode='extended', exportselection=False)
        self.chosen.pack(fill='both', expand=True)
        self.chosen.bind('<Double-Button-1>', lambda _e: self._remove())

        right = ttk.Frame(body)
        right.pack(side='left', fill='both', expand=True, padx=(8, 0))
        self.header_var = tk.StringVar(master=self, value='')
        ttk.Label(right, textvariable=self.header_var,
                  font='OpenFloHeading').pack(side='top', anchor='w')
        self.summary = tk.Text(right, height=8, wrap='word')
        self.summary.pack(side='bottom', fill='x')
        _f, self.fig, self.canvas = figure_panel(right, figsize=(7, 3.8),
                                                 dark=False)
        self._source_changed()

    # ── sources ────────────────────────────────────────────────────────────
    def _last_unmix(self):
        """``(S, fluors, detectors)`` of the editor's last unmix, or None."""
        last = getattr(self.editor, '_last_unmix_spectra', None)
        if not last:
            return None
        S, fluors = last
        S = np.asarray(S, dtype=float)
        dets = list(getattr(self.editor, '_last_unmix_detectors', None)
                    or [])
        if len(dets) != S.shape[1]:
            dets = [f"D{k + 1}" for k in range(S.shape[1])]
        return S, list(fluors), dets

    def _is_model(self):
        return self.source_var.get() in pc.DETECTOR_SETS

    def _kind(self):
        name = self.source_var.get()
        if name in pc.DETECTOR_SETS:
            return pc.DETECTOR_SETS[name]['kind']
        if self._loaded is None:
            return 'spectral'
        return ('spectral' if len(self._loaded[2]) >= SPECTRAL_DETECTORS
                else 'conventional')

    def _detectors(self):
        if self._is_model():
            return list(pc.DETECTOR_SETS[self.source_var.get()]['detectors'])
        return []

    def _names(self):
        """Every fluor the current source offers."""
        if self._is_model():
            return pc.catalog_fluors()
        return list(self._loaded[1]) if self._loaded else []

    def _set_loaded(self, S, fluors, dets, label):
        self._loaded = (np.asarray(S, dtype=float), list(fluors), list(dets))
        self._loaded_label = label
        self.source_box.configure(values=[*pc.DETECTOR_SETS, label])
        self.source_var.set(label)
        self._panel = list(fluors)
        self._source_changed(keep_panel=True)

    def _source_changed(self, keep_panel=False):
        names = self._names()
        if not keep_panel:
            self._panel = [f for f in self._panel if f in names]
        self.thr_var.set(f"{pc.DEFAULT_THRESHOLD[self._kind()]:g}")
        self.note_var.set(MODEL_NOTE if self._is_model() else
                          f"Measured signatures: {len(names)} fluor(s) on "
                          f"{len(self._loaded[2]) if self._loaded else 0} "
                          "detector(s).")
        self._fill_lists()
        self._refresh()

    def _load_csv(self):
        path = filedialog.askopenfilename(
            parent=self, title="Load reference spectra",
            filetypes=[('CSV', '*.csv'), ('Text', '*.txt *.tsv'),
                       ('All files', '*.*')])
        if not path:
            return
        try:
            S, fluors, dets = pc.load_spectra_csv(path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            show_error(self, "Panel check",
                       "The spectra could not be read.", exc, path)
            return
        name = os.path.basename(path)
        self._set_loaded(S, fluors, dets,
                         "Spectra CSV: " + (name if len(name) <= 36
                                            else name[:33] + '…'))

    def _use_last_unmix(self):
        last = self._last_unmix()
        if last is None:
            self.header_var.set("No unmix in this session yet.")
            return
        self._set_loaded(*last, label="Last unmix (this session)")

    # ── the panel ──────────────────────────────────────────────────────────
    def _fill_lists(self):
        q = self.search_var.get().strip().casefold()
        self._shown = [f for f in self._names()
                       if f not in self._panel and q in f.casefold()]
        self.avail.delete(0, 'end')
        for f in self._shown:
            self.avail.insert('end', f)
        self.chosen.delete(0, 'end')
        for f in self._panel:
            self.chosen.insert('end', f)

    def add_fluors(self, names):
        """Put `names` in the panel (those the source offers) and recheck."""
        offered = set(self._names())
        for f in names:
            if f in offered and f not in self._panel:
                self._panel.append(f)
        self._fill_lists()
        self._refresh()

    def _add(self):
        self.add_fluors([self._shown[i] for i in self.avail.curselection()])

    def _remove(self):
        drop = {self._panel[i] for i in self.chosen.curselection()}
        self._panel = [f for f in self._panel if f not in drop]
        self._fill_lists()
        self._refresh()

    def _signatures(self):
        """``(S, detectors)`` of the panel on the current source."""
        if self._is_model():
            dets = self._detectors()
            return (pc.signature_matrix(self._panel, dets),
                    [d[0] for d in dets])
        S, fluors, dets = self._loaded  # type: ignore[misc]
        idx = [fluors.index(f) for f in self._panel]
        return S[idx], list(dets)

    def _threshold(self):
        t = _num(self.thr_var.get())
        if t is None or not 0 < t <= 1:
            t = pc.DEFAULT_THRESHOLD[self._kind()]
            self.thr_var.set(f"{t:g}")
        return t

    def _refresh(self):
        if not self._panel:
            self.report = None
            self.header_var.set("Add fluors to the panel.")
            self._draw()
            self._set_summary("")
            return
        S, dets = self._signatures()
        self.report = pc.check_panel(S, self._panel, dets,
                                     threshold=self._threshold())
        r = self.report
        self.header_var.set(
            f"{len(r['fluors'])} fluors · {len(dets)} detectors · "
            f"complexity index {self._cond_text(r['condition_number'])}"
            f" · {len(r['similar_pairs'])} similar pair(s)")
        self._draw()
        self._set_summary(self.summary_text())

    @staticmethod
    def _cond_text(cond):
        return "∞" if not np.isfinite(cond) else f"{cond:.2f}"

    def _show_spillover(self):
        return (self.report is not None and self._kind() == 'conventional')

    def _draw(self):
        fig = self.fig
        fig.clear()
        r = self.report
        if r is not None:
            fl = r['fluors']
            spill = self._show_spillover()
            ax = fig.add_subplot(1, 2 if spill else 1, 1)
            sim = np.asarray(r['similarity'], dtype=float)
            draw_similarity(fig, ax, sim, fl, title='Predicted similarity')
            self._annotate(ax, sim, r['threshold'])
            if spill:
                ax2 = fig.add_subplot(1, 2, 2)
                sp = np.ma.masked_invalid(np.asarray(r['spillover'],
                                                     dtype=float))
                im = ax2.imshow(sp, vmin=0.0, vmax=1.0, cmap='viridis',
                                aspect='auto')
                fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
                ax2.set_title('Predicted spillover', fontsize=9)
                ax2.set_ylabel('fluor', fontsize=8)
                ax2.set_xlabel("fluor's primary detector", fontsize=8)
                fluor_ticks(ax2, fl)
                self._annotate(ax2, np.asarray(r['spillover'], dtype=float),
                               None)
            try:
                fig.tight_layout()
            except Exception:  # noqa: BLE001 - crowded figures
                pass
        self.spill_btn.state(['!disabled'] if self._show_spillover()
                             else ['disabled'])
        try:
            from .theme import _dialog_dark_on, _theme_figure_dark
            if _dialog_dark_on(self):
                _theme_figure_dark(fig)
        except Exception:  # noqa: BLE001 - dialog theming is cosmetic
            pass
        self.canvas.draw_idle()

    @staticmethod
    def _annotate(ax, M, threshold):
        """Values in the cells of a small matrix; with `threshold`, the
        flagged pairs outlined."""
        n = M.shape[0]
        if n > 14:
            return
        for i in range(n):
            for j in range(n):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                if n <= 10:
                    ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                            fontsize=6, color='white' if v < 0.6 else 'black')
                if threshold is not None and i != j and v >= threshold:
                    ax.add_patch(_cell_outline(i, j))

    def summary_text(self):
        """The plain-text findings shown under the heatmap."""
        r = self.report
        if r is None:
            return ""
        lines = []
        if self._is_model():
            lines.append(MODEL_NOTE)
        cond = r['condition_number']
        n_fl, n_det = len(r['fluors']), len(r['detectors'])
        lines.append(f"Complexity index (condition number): "
                     f"{self._cond_text(cond)}"
                     + ("  [ill-conditioned]" if not cond <= 100 else ""))
        if n_fl > n_det:
            lines.append(f"{n_fl} fluors on {n_det} detectors: more fluors "
                         "than detectors cannot be told apart.")
        if r['degenerate_fluors']:
            lines.append("Not seen by these detectors (no laser excites "
                         "it, or no filter collects it): "
                         + ', '.join(r['degenerate_fluors']))
        sp = r['similar_pairs']
        if sp:
            lines.append(f"Similar pairs (above {r['threshold']:g}):")
            lines += [f"  • {d['fluor_a']} ~ {d['fluor_b']}  (cosine "
                      f"{d['similarity']:.3f})" for d in sp]
        else:
            lines.append(f"No pair above {r['threshold']:g}.")
        if self._kind() == 'conventional':
            if r['shared_primary']:
                lines.append("Same primary detector (cannot share a "
                             "conventional instrument):")
                lines += [f"  • {a} and {b} in {d}"
                          for a, b, d in r['shared_primary']]
            lines.append("Primary detectors: " + ', '.join(
                f"{f} {d}" for f, d in zip(r['fluors'], r['primary_detector'],
                                           strict=True) if d is not None))
        # Only a swap that helps: a lower, finite complexity index.
        better = [d for d in r['swap'] if np.isfinite(d['condition_number'])
                  and not d['condition_number'] >= cond]
        if better:
            lines.append("Leaving one out would cut the complexity index "
                         "most for:")
            lines += [f"  • {d['fluor']}: "
                      f"{self._cond_text(d['condition_number'])} without it"
                      for d in better[:3]]
        return "\n".join(lines)

    def _set_summary(self, text):
        self.summary.configure(state='normal')
        self.summary.delete('1.0', 'end')
        self.summary.insert('1.0', text)
        self.summary.configure(state='disabled')

    # ── exports ────────────────────────────────────────────────────────────
    def _write_rows(self, rows, title, initial, kind):
        path = ask_csv_path(self, title=title, initialfile=initial)
        if not path:
            return None
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)
        except Exception as exc:
            show_error(self, "Export failed",
                       "The panel check could not be exported.", exc, path)
            return None
        self.header_var.set(f"Exported → {os.path.basename(path)}")
        self._audit_export(path, kind)
        return path

    def _audit_export(self, path, kind):
        try:
            audit = getattr(self.editor, '_audit', None)
            if audit is not None:
                audit('panel_check.export', path=path, kind=kind,
                      fluors=list(self.report['fluors'])
                      if self.report else [])
        except Exception as exc:
            note("recording the export in the audit trail", exc)

    def _export_matrix(self, which):
        if self.report is None:
            self.header_var.set("Nothing to export: add fluors first.")
            return None
        return self._write_rows(pc.matrix_csv_rows(self.report, which),
                                f"Export predicted {which}",
                                f'panel_{which}.csv', which)

    def _save_signatures(self):
        if self.report is None:
            self.header_var.set("Nothing to save: add fluors first.")
            return None
        r = self.report
        return self._write_rows(
            pc.spectra_csv_rows(r['signatures'], r['fluors'],
                                r['detectors']),
            "Save signatures", 'panel_spectra.csv', 'signatures')

    def _export_figure(self):
        path = export_figure(self, self.fig, background=self.bg_var.get(),
                             initialfile='panel_check.png')
        if path:
            self._audit_export(path, 'figure')
        return path


def _cell_outline(i, j):
    from matplotlib.patches import Rectangle
    return Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.5,
                     edgecolor='#00c8ff')
