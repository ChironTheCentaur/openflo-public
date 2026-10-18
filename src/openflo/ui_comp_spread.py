"""Tools, Spreading matrix: the spillover spread of conventional
compensation (Complaints 24).

Each compensation channel takes a loaded single-stain control (matched
automatically by the channel it is brightest in, changeable), and an
optional unstained control; the matrix is computed on their compensated
linear values (comp_spread.compensation_spread_matrix).

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np

from .tool_window import (
    ask_csv_path,
    background_selector,
    export_figure,
    figure_panel,
    scrolled_body,
)
from .ui_errors import show_error
from .ui_notice import report_saved
from .ui_tips import auto_tip

NONE = "—"


def compensated_values(sample, channels):
    """``(events, channels)`` compensated linear values, or None when the
    sample lacks a channel or a recoverable linear scale."""
    from .pipeline import linear_values
    try:
        return np.column_stack([np.asarray(linear_values(sample, c),
                                           dtype=float) for c in channels])
    except Exception:                                  # noqa: BLE001
        return None


class SpreadMatrixWindow(tk.Toplevel):
    """Controls per channel (left) / matrix heatmap (right) / buttons."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Spreading matrix")
        self.minsize(820, 520)
        self.ssm = None
        self.fig: Any = None
        self.status_var = tk.StringVar(value='')
        ref = editor._samples.get(editor._active_sample)
        if ref is None:
            from .ui_proliferation import no_sample
            self.channels = []
            no_sample(self, "Load and select a compensated sample first.")
            return
        self.channels = list(getattr(ref, 'comp_channels', None) or [])
        names = [n for n in editor._sample_order if n in editor._samples]
        self.control_vars = {}
        self.unstained_var = tk.StringVar(value=NONE)

        bar = ttk.Frame(self, padding=(12, 4, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", command=self.destroy).pack(side='right')
        self.bg_var, bg = background_selector(bar)
        b = ttk.Button(bar, text="Export figure…", command=self._export_png)
        b.pack(side='left')
        auto_tip(b, "Save the matrix as PNG, SVG or PDF.")
        bg.pack(side='left', padx=(6, 12))
        c = ttk.Button(bar, text="Export CSV…", command=self._export_csv)
        c.pack(side='left')
        auto_tip(c, "The matrix, stains on the rows, as SpectroFlo shows "
                    "it; NA where a channel has no control.")
        ttk.Label(self, textvariable=self.status_var, wraplength=780,
                  justify='left', padding=(12, 2)).pack(side='bottom',
                                                        fill='x')

        side = ttk.Frame(self, padding=(12, 10, 6, 4))
        side.pack(side='left', fill='y')
        ttk.Label(side, text="Single-stain control for each channel",
                  font='OpenFloHeading').pack(anchor='w')
        body = scrolled_body(side)
        values = [NONE] + names
        for i, ch in enumerate(self.channels):
            ttk.Label(body, text=editor._fmt_channel(ch)).grid(
                row=i, column=0, sticky='e', padx=(0, 4), pady=1)
            var = tk.StringVar(value=NONE)
            cb = ttk.Combobox(body, textvariable=var, values=values,
                              state='readonly', width=22)
            cb.grid(row=i, column=1, sticky='w', pady=1)
            auto_tip(cb, "The control stained only for this channel's dye.")
            self.control_vars[ch] = var
        un = ttk.Frame(side)
        un.pack(anchor='w', pady=(8, 0))
        ttk.Label(un, text="Unstained:").pack(side='left')
        ucb = ttk.Combobox(un, textvariable=self.unstained_var, values=values,
                           state='readonly', width=22)
        ucb.pack(side='left', padx=4)
        auto_tip(ucb, "Used as the negative of a control that has no "
                      "negative events of its own.")
        btns = ttk.Frame(side)
        btns.pack(anchor='w', pady=(8, 0))
        a = ttk.Button(btns, text="Match controls", command=self.match)
        a.pack(side='left')
        auto_tip(a, "Give each channel the loaded sample brightest in it.")
        r = ttk.Button(btns, text="Compute", command=self.compute,
                       default='active')
        r.pack(side='left', padx=6)
        auto_tip(r, "Compute the spreading matrix from these controls.")

        right = ttk.Frame(self, padding=(6, 10, 12, 4))
        right.pack(side='left', fill='both', expand=True)
        _frame, self.fig, self.canvas = figure_panel(right, figsize=(6, 5))
        if not self.channels:
            self.status_var.set(
                f"{editor._active_sample} has no compensation. Apply a "
                "matrix first (Tools > Compensation…): the spreading matrix "
                "is measured on compensated controls.")
            return
        self.match()

    def match(self):
        """Give each channel the sample whose detected primary channel it is
        (comp_spread.primary_channel); a channel two samples claim takes the
        brighter. The unstained pick, if any, is left out."""
        from .comp_spread import primary_channel
        e = self.editor
        un = self.unstained_var.get()
        claims = {}
        for n in e._sample_order:
            if n not in e._samples or n == un:
                continue
            v = compensated_values(e._samples[n], self.channels)
            if v is None:
                continue
            p = primary_channel(v, self.channels)
            if p is None:
                continue
            j = self.channels.index(p)
            z = np.arcsinh(v[:, j] / 150.0)
            score = float(np.percentile(z, 99.5) - np.percentile(z, 25))
            if score > 1.0 and score > claims.get(p, (None, -1))[1]:
                claims[p] = (n, score)
        for ch, var in self.control_vars.items():
            var.set(claims[ch][0] if ch in claims else NONE)
        self.status_var.set(f"Matched {len(claims)} of {len(self.channels)} "
                            "channels; change any that are wrong, then "
                            "Compute.")

    def controls(self):
        """``{channel: sample}`` of the picked controls."""
        return {ch: v.get() for ch, v in self.control_vars.items()
                if v.get() != NONE}

    def compute(self):
        from .comp_spread import compensation_spread_matrix, worst_pairs
        e = self.editor
        picked = self.controls()
        if not picked:
            self.status_var.set("Pick at least one control.")
            return
        arrays, skipped = {}, []
        for ch, n in picked.items():
            v = compensated_values(e._samples[n], self.channels)
            if v is None or getattr(e._samples[n], 'comp_matrix', None) is None:
                skipped.append(n)
            else:
                arrays[ch] = v
        un = None
        if self.unstained_var.get() != NONE:
            un = compensated_values(e._samples[self.unstained_var.get()],
                                    self.channels)
        try:
            self.ssm, _ = compensation_spread_matrix(arrays, self.channels,
                                                     unstained=un)
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Spreading matrix",
                       "The matrix could not be computed.", exc)
            return
        self.draw()
        worst = worst_pairs(self.ssm, self.channels, top=3)
        note = "; ".join(f"{e._fmt_channel(a)} into {e._fmt_channel(b)} "
                         f"{s:.2f}" for a, b, s in worst)
        self.status_var.set(
            (f"Largest spread: {note}." if note else "No spread measured.")
            + (f"  Skipped (not compensated): {', '.join(skipped)}."
               if skipped else ""))
        e._audit('comp.spread_matrix', controls=picked,
                 unstained=self.unstained_var.get()
                 if self.unstained_var.get() != NONE else None)

    def display(self):
        """The matrix with stains on the rows (the usual display)."""
        return None if self.ssm is None else np.asarray(self.ssm).T

    def draw(self):
        from .theme import _dialog_dark_on, _theme_figure_dark
        self.fig.clear()
        mat = self.display()
        if mat is None:
            return
        ax = self.fig.add_subplot(111)
        labels = [self.editor._fmt_channel(c) for c in self.channels]
        finite = mat[np.isfinite(mat)]
        vmax = float(finite.max()) if finite.size and finite.max() > 0 else 1
        ax.imshow(np.ma.masked_invalid(mat), cmap='magma_r', vmin=0,
                  vmax=vmax, aspect='equal')
        n = len(labels)
        ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)
        ax.set_xlabel("spread into")
        ax.set_ylabel("stain")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, 'NA' if not np.isfinite(v) else f"{v:.2f}",
                    ha='center', va='center', fontsize=6,
                    color='white' if np.isfinite(v) and v > vmax / 2
                    else 'black')
        self.fig.tight_layout()
        if _dialog_dark_on(self):
            _theme_figure_dark(self.fig)
        self.canvas.draw_idle()

    def rows(self):
        mat = self.display()
        out = [['stain'] + list(self.channels)]
        if mat is None:
            return out
        for i, ch in enumerate(self.channels):
            out.append([ch] + ['NA' if not np.isfinite(v) else f"{v:.6g}"
                               for v in mat[i]])
        return out

    def _export_csv(self):
        if self.ssm is None:
            self.status_var.set("Compute the matrix first.")
            return
        path = ask_csv_path(self, initialfile='spreading_matrix.csv')
        if not path:
            return
        import csv
        try:
            with open(path, 'w', newline='', encoding='utf-8') as fh:
                csv.writer(fh).writerows(self.rows())
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Spreading matrix",
                       "The table could not be saved.", exc, path)
            return
        report_saved(self, path)

    def _export_png(self):
        if self.fig is None:
            return
        export_figure(self, self.fig, background=self.bg_var.get(),
                      initialfile='spreading_matrix.png')
