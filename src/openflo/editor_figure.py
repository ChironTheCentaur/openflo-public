"""Multi-panel figure-layout builder and preview.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .editor_base import EditorMixin
from .theme import _dialog_dark_on, _theme_figure_dark, savefig_background


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
        except Exception:
            pass
        return fig

    def _resolve_token_to_channel(self, tok):
        """Map a user token (channel name, ``Label (DET)`` form, or a marker
        label like ``CD34``) to a real channel name, or ``None``."""
        tok = (tok or '').strip()
        if not tok:
            return None
        if tok in self._channels:
            return tok
        ch = self._resolve_channel(tok)
        if ch in self._channels:
            return ch
        low = tok.lower()
        for det, lbl in self._channel_labels.items():
            if lbl and lbl.lower() == low and det in self._channels:
                return det
        for det, lbl in self._channel_labels.items():
            if lbl and low in lbl.lower() and det in self._channels:
                return det
        return None

    def _parse_pairs_str(self, text):
        """Parse ``"CD34/CD11b, CD11b/CD45"`` into resolved ``(x, y)`` channel
        tuples. Unresolvable tokens are skipped."""
        pairs = []
        for chunk in re.split(r'[,;\n]', text or ''):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = re.split(r'\s*[/xX×]\s*|\s+vs\.?\s+', chunk, maxsplit=1)
            if len(parts) != 2:
                continue
            xc = self._resolve_token_to_channel(parts[0])
            yc = self._resolve_token_to_channel(parts[1])
            if xc and yc:
                pairs.append((xc, yc))
        return pairs

    def _build_and_preview_figure(self, opts):
        """Build the multi-panel figure from the dialog's options and show it
        in a preview window with a Save control."""
        samples = self._selected_samples()
        if not samples:
            return
        mode = self.mode_var.get()
        color = self.color_combo.get()
        cur_x = self._resolve_channel(self.x_combo.get())
        cur_y = self._resolve_channel(self.y_combo.get())
        layout = opts.get('layout', 'per_sample')
        pairs = self._parse_pairs_str(opts.get('pairs', ''))
        ncols = opts.get('ncols', 3)
        draw_gates = opts.get('gates', True)

        panels = []
        if layout == 'single':
            panels.append(dict(samples=samples, x=cur_x, y=cur_y,
                               mode=mode, color=color, title=None))
        elif layout == 'per_sample':
            for nm in samples:
                panels.append(dict(samples=[nm], x=cur_x, y=cur_y,
                                   mode=mode, color=color,
                                   title=self._short_sample(nm)))
        elif layout == 'per_pair':
            if not pairs and cur_x and cur_y:
                pairs = [(cur_x, cur_y)]
            for (px, py) in pairs:
                ttl = f"{self._fmt_channel(px)} / {self._fmt_channel(py)}"
                panels.append(dict(samples=samples, x=px, y=py, mode=mode,
                                   color=color, title=ttl))
        elif layout == 'grid':
            if not pairs and cur_x and cur_y:
                pairs = [(cur_x, cur_y)]
            ncols = max(1, len(pairs))
            for nm in samples:
                for (px, py) in pairs:
                    ttl = (f"{self._short_sample(nm, 16)} · "
                           f"{self._fmt_channel(px)}/{self._fmt_channel(py)}")
                    panels.append(dict(samples=[nm], x=px, y=py, mode=mode,
                                       color=color, title=ttl))

        if not panels:
            messagebox.showwarning(
                "Figure layout",
                "Nothing to plot — check the layout and channel pairs.",
                parent=self)
            return

        fig = self._build_layout_figure(panels, ncols, draw_gates=draw_gates)
        if fig is None:
            return
        self._show_figure_preview(fig)

    def _show_figure_preview(self, fig):
        """Pop a Toplevel embedding ``fig`` with Save (PNG/PDF/SVG) / Close."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = tk.Toplevel(self)
        win.title("Figure preview")
        win.geometry("1000x720")

        bar = ttk.Frame(win)
        bar.pack(fill='x', side='top')

        # Export background: White (opaque, default), Transparent (fully — for
        # placing on a coloured page / poster), or Translucent (50% white).
        # PNG / PDF / SVG carry alpha; TIFF may flatten it depending on viewer.
        bg_var = tk.StringVar(value='Dark' if _dialog_dark_on(self) else 'White')

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
                savefig_background(fig, path, background=bg, dpi=300)
            except Exception as exc:
                messagebox.showerror(
                    "Figure layout", f"Could not save figure:\n{exc}",
                    parent=win)
                return
            self._audit('figure.export', path=path,
                        n_panels=len(fig.axes), background=bg)
            messagebox.showinfo("Figure layout", f"Saved:\n{path}",
                                parent=win)

        ttk.Button(bar, text="Save…", command=_save).pack(
            side='left', padx=4, pady=4)
        ttk.Button(bar, text="Close", command=win.destroy).pack(
            side='left', padx=(0, 4), pady=4)
        ttk.Label(bar, text="Background:").pack(side='left', padx=(12, 2))
        ttk.Combobox(bar, textvariable=bg_var, width=12, state='readonly',
                     values=['White', 'Dark', 'Transparent', 'Translucent']).pack(
            side='left', pady=4)

        if _dialog_dark_on(self):
            _theme_figure_dark(fig)
        cf = ttk.Frame(win)
        cf.pack(fill='both', expand=True)
        canvas = FigureCanvasTkAgg(fig, master=cf)
        _w = canvas.get_tk_widget()
        if _dialog_dark_on(self):
            from .theme import current_palette
            _w.configure(bg=current_palette()['bg'], highlightthickness=0)
        _w.pack(fill='both', expand=True)
        # A pan/zoom toolbar would be nice, but NavigationToolbar2Tk isn't in
        # matplotlib's type stubs (pyright flags the import); the Save control
        # plus matplotlib's own keymap is enough for a preview.
        canvas.draw()
