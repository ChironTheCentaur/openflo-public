"""Multi-panel figure-layout builder and preview.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .editor_base import EditorMixin
from .export_stamp import DEFAULT_EXPORT_BACKGROUND, EXPORT_BACKGROUNDS
from .theme import dialog_dark_on, match_dialog, savefig_background
from .tool_window import BACKGROUND_TIP
from .ui_errors import show_error
from .ui_logic import resolve_channel
from .ui_notice import report_saved

# A pair separator that no channel name holds: '/', '×' or ' vs '. An 'x'
# also separates a pair; see FigureMixin._pair_splits.
_PAIR_SEP = re.compile(r'\s*[/×]\s*|\s+vs\.?\s+')


class FigureMixin(EditorMixin):
    """Build and preview a multi-panel publication figure from a panel/pairs specification."""

    def _build_layout_figure(self, panels, ncols, draw_gates=True,
                             panel_size=(3.2, 2.6), dpi=120, suptitle=None):
        """Assemble a multi-panel matplotlib ``Figure`` from a list of panel
        specs (each: ``{samples, x, y, mode, color, title}``). Returns the
        Figure, or ``None`` when there are no panels."""
        from matplotlib.figure import Figure
        n = len(panels)
        if n == 0:
            return None
        ncols = max(1, min(int(ncols), n))
        nrows = int(np.ceil(n / ncols))
        fig = Figure(figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                     dpi=dpi)
        for i, spec in enumerate(panels):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            self._render_into(ax, spec.get('samples') or [], spec.get('x'),
                              spec.get('y'), spec.get('mode', 'dot'),
                              spec.get('color', 'By density'),
                              draw_gates=draw_gates)
            title = spec.get('title')
            if title:
                ax.set_title(title, fontsize=8)
        if suptitle:
            fig.suptitle(suptitle, fontsize=11)
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 - tight_layout can fail on crowded figures
            pass
        return fig

    def _resolve_token_to_channel(self, tok, fuzzy=True):
        """Map a user token (channel name, ``Label (DET)`` form, or a marker
        label like ``CD902``) to a real channel name, or ``None``. With
        ``fuzzy`` a token that is part of one label also matches it."""
        tok = (tok or '').strip()
        if not tok:
            return None
        if tok in self._channels:
            return tok
        ch = resolve_channel(tok)
        if ch in self._channels:
            return ch
        low = tok.lower()
        for det, lbl in self._channel_labels.items():
            if lbl and lbl.lower() == low and det in self._channels:
                return det
        if not fuzzy:
            return None
        for det, lbl in self._channel_labels.items():
            if lbl and low in lbl.lower() and det in self._channels:
                return det
        return None

    def _parse_pairs_report(self, text):
        """``(pairs, bad)``: the resolved ``(x, y)`` channel tuples of
        ``"FL1-A/FL2-A, FSC-AxSSC-A"``, and one line per entry that is not a
        pair or names no channel, for the warning."""
        pairs, bad = [], []
        for chunk in re.split(r'[,;\n]', text or ''):
            chunk = chunk.strip()
            if not chunk:
                continue
            splits = self._pair_splits(chunk)
            if (self._resolve_token_to_channel(chunk, fuzzy=False)
                    and not _PAIR_SEP.search(chunk)):
                splits = []               # one channel whose name has an x
            if not splits:
                bad.append(f"'{chunk}': not a pair (write X/Y)")
                continue
            for left, right, fuzzy in splits:
                xc = self._resolve_token_to_channel(left, fuzzy)
                yc = self._resolve_token_to_channel(right, fuzzy)
                if xc and yc:
                    pairs.append((xc, yc))
                    break
            else:
                left, right, fuzzy = splits[0]
                missing = [t.strip() or '(blank)' for t in (left, right)
                           if not self._resolve_token_to_channel(t, fuzzy)]
                bad.append(f"'{chunk}': no channel "
                           + ' or '.join(f"'{m}'" for m in missing))
        return pairs, bad

    @staticmethod
    def _pair_splits(chunk):
        """``[(x, y, fuzzy), ...]``: the ways ``chunk`` can be read as X/Y,
        best first; ``fuzzy`` lets part of a label name a channel. At its
        first '/', '×' or 'vs'; else at each 'x', those with spaces on both
        sides first. Channel names and labels can hold an x (FLX-A, "A x
        B"), so the caller keeps the first split whose two sides both name
        a channel, and an x with no spaces counts only for exact names (part
        of a label would match "FL" in FLX-A)."""
        parts = _PAIR_SEP.split(chunk, maxsplit=1)
        if len(parts) == 2:
            return [(parts[0], parts[1], True)]
        spaced, bare = [], []
        for m in re.finditer(r'\s*[xX]\s*', chunk):
            left, right = chunk[:m.start()], chunk[m.end():]
            if not (left.strip() and right.strip()):
                continue
            sep = m.group()
            if sep[0].isspace() and sep[-1].isspace():
                spaced.append((left, right, True))
            else:
                bare.append((left, right, False))
        return spaced + bare

    def _build_and_preview_figure(self, opts):
        """Build the multi-panel figure from the dialog's options and show it
        in a preview window with a Save control."""
        samples = self._selected_samples()
        if not samples:
            return
        mode = self.mode_var.get()
        color = self.color_combo.get()
        cur_x = resolve_channel(self.x_combo.get())
        cur_y = resolve_channel(self.y_combo.get())
        layout = opts.get('layout', 'per_sample')
        pairs, bad = self._parse_pairs_report(opts.get('pairs', ''))
        if bad and layout in ('per_pair', 'grid'):
            # Name the entries that match no channel: the figure leaves
            # them out (and never stands the current X/Y in for them).
            more = (f"The figure shows the other {len(pairs)} pair(s)."
                    if pairs else "Nothing to plot: fix the pairs and Build "
                    "again.")
            messagebox.showwarning(
                "Quick figure grid",
                "These channel pairs were not found:\n  "
                + '\n  '.join(bad) + f"\n\n{more}", parent=self)
            if not pairs:
                return
        ncols = opts.get('ncols', 3)
        draw_gates = opts.get('gates', True)

        panels = []
        if layout == 'single':
            panels.append({'samples': samples, 'x': cur_x, 'y': cur_y,
                           'mode': mode, 'color': color, 'title': None})
        elif layout == 'per_sample':
            for nm in samples:
                panels.append({'samples': [nm], 'x': cur_x, 'y': cur_y,
                               'mode': mode, 'color': color,
                               'title': self._short_sample(nm)})
        elif layout == 'per_pair':
            if not pairs and cur_x and cur_y:
                pairs = [(cur_x, cur_y)]
            for (px, py) in pairs:
                ttl = f"{self._fmt_channel(px)} / {self._fmt_channel(py)}"
                panels.append({'samples': samples, 'x': px, 'y': py, 'mode': mode,
                               'color': color, 'title': ttl})
        elif layout == 'grid':
            if not pairs and cur_x and cur_y:
                pairs = [(cur_x, cur_y)]
            ncols = max(1, len(pairs))
            for nm in samples:
                for (px, py) in pairs:
                    ttl = (f"{self._short_sample(nm, 16)} · "
                           f"{self._fmt_channel(px)}/{self._fmt_channel(py)}")
                    panels.append({'samples': [nm], 'x': px, 'y': py, 'mode': mode,
                                   'color': color, 'title': ttl})

        if not panels:
            messagebox.showwarning(
                "Quick figure grid",
                "Nothing to plot — check the layout and channel pairs.",
                parent=self)
            return

        fig = self._build_layout_figure(panels, ncols, draw_gates=draw_gates)
        if fig is None:
            return
        self._show_figure_preview(fig, samples)

    def _show_figure_preview(self, fig, samples=None):
        """Pop a Toplevel embedding ``fig`` with Save (PNG/PDF/SVG) / Close.
        ``samples`` (the samples drawn) go in the export's audit entry."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self)
        win.title("Figure preview")
        win.geometry("1000x720")

        bar = ttk.Frame(win)
        bar.pack(fill='x', side='top')

        # Export background: Transparent (the default, for any page, poster
        # or slide), White, Dark or Translucent (50% white). PNG / PDF / SVG
        # carry alpha; TIFF may flatten it depending on viewer.
        bg_var = tk.StringVar(value=DEFAULT_EXPORT_BACKGROUND)

        def _save():
            path = filedialog.asksaveasfilename(
                parent=win, title="Save figure",
                defaultextension='.png',
                filetypes=[('PNG image', '*.png'),
                           ('PDF document', '*.pdf'),
                           ('SVG vector', '*.svg'),
                           ('TIFF image', '*.tif *.tiff')])
            if not path:
                return
            bg = bg_var.get()
            try:
                savefig_background(fig, path, background=bg)
            except Exception as exc:  # noqa: BLE001 - a failed save is reported in an error dialog
                show_error(win, "Quick figure grid",
                           "The figure could not be saved.", exc, path)
                return
            self._audit('figure.export', path=path,
                        n_panels=len(fig.axes), background=bg,
                        **({'samples': list(samples)} if samples is not None
                           else {}))
            report_saved(win, path, "Saved")

        b_save = ttk.Button(bar, text="Export figure…", command=_save)
        b_save.pack(side='left', padx=4, pady=4)
        self._tip(b_save, "Save this figure as a file: PNG or TIFF (an "
                  "image), PDF or SVG (sharp at any size, for posters and "
                  "papers). You pick the name and the type next.")
        ttk.Button(bar, text="Close", command=win.destroy).pack(
            side='left', padx=(0, 4), pady=4)
        ttk.Label(bar, text="Background:").pack(side='left', padx=(12, 2))
        bg_cb = ttk.Combobox(bar, textvariable=bg_var, width=12,
                             state='readonly',
                             values=list(EXPORT_BACKGROUNDS))
        bg_cb.pack(side='left', pady=4)
        self._tip(bg_cb, BACKGROUND_TIP)

        match_dialog(fig, self)
        cf = ttk.Frame(win)
        cf.pack(fill='both', expand=True)
        canvas = FigureCanvasTkAgg(fig, master=cf)
        _w = canvas.get_tk_widget()
        if dialog_dark_on(self):
            from .theme import current_palette
            _w.configure(bg=current_palette()['bg'], highlightthickness=0)
        _w.pack(fill='both', expand=True)
        # A pan/zoom toolbar would be nice, but NavigationToolbar2Tk isn't in
        # matplotlib's type stubs (pyright flags the import); the Save control
        # plus matplotlib's own keymap is enough for a preview.
        canvas.draw()
