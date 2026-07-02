"""Shared scaffolding for the editor's tool / analysis pop-up windows.

Collapses the boilerplate that nearly every ``ui_*.py`` window repeated: a
themed ``Toplevel`` (title / geometry / optional modal / Escape-to-close /
titlebar tint), the matplotlib canvas block, the dark/white export-background
selector, and the export-path dialogs.

Deliberately composition-first, not a deep hierarchy: a window can subclass
``ToolWindow`` for the chrome and/or call the free helpers
(``figure_panel`` / ``background_selector`` / ``export_figure`` /
``ask_csv_path``) for the parts it needs. The windows vary too much (modal vs
not, figure vs not, takes-editor vs not) for one rigid base to fit all.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .theme import _dialog_dark_on, _theme_figure_dark, savefig_background


class ToolWindow(tk.Toplevel):
    """Thin themed ``Toplevel`` base for pop-up tools.

    Handles only the universal chrome: title, optional geometry, optional modal
    (``transient`` + ``grab_set``), Escape-to-close, and best-effort titlebar
    theming through the editor. Content/layout stays the subclass's job."""

    def __init__(self, parent, title, geometry=None, modal=False, editor=None):
        super().__init__(parent)
        self.title(title)
        if geometry:
            self.geometry(geometry)
        # The editor owns titlebar theming; fall back to the parent when it is
        # the editor (the common case) so callers needn't pass it explicitly.
        self.editor = (editor if editor is not None
                       else (parent if hasattr(parent, '_apply_titlebar_to')
                             else None))
        if modal:
            try:
                self.transient(parent)
                self.grab_set()
            except Exception:
                pass
        self.bind('<Escape>', lambda _e: self.destroy())
        ed = self.editor
        if ed is not None and hasattr(ed, '_apply_titlebar_to'):
            try:
                self.after(60, lambda: ed._apply_titlebar_to(self))
            except Exception:
                pass


def figure_panel(parent, figsize=(8, 5), dpi=100, dark=None, pack_kw=None):
    """Build the standard matplotlib canvas: returns ``(frame, fig, canvas)``.
    A ttk.Frame is packed to fill ``parent`` and the canvas is packed inside it
    (``pack_kw`` overrides the canvas's pack options, default fill+expand).
    ``dark`` (default: auto from the dialog theme) themes the figure dark for
    pop-up previews; pass ``dark=False`` for windows that theme in their own
    draw step. Replaces the identical Frame + FigureCanvasTkAgg block repeated
    across windows. For a window that draws content then shows it, call this for
    the empty fig, draw into ``fig``, then ``canvas.draw()``."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    if dark is None:
        dark = _dialog_dark_on(parent)
    dialog_dark = dark or _dialog_dark_on(parent)
    fig = Figure(figsize=figsize, dpi=dpi)
    if dark:
        _theme_figure_dark(fig)
    elif dialog_dark:
        # The caller themes its figure later (in its own _draw, after data is
        # collected) — but until then the EMPTY figure is matplotlib-white and
        # fills the canvas, so the window opens as a white slab in a dark UI.
        # Paint at least the figure facecolor dark now; _draw re-themes fully.
        from .theme import THEMES
        try:
            fig.set_facecolor(THEMES['midnight']['plot_bg'])
        except Exception:
            pass
    frame = ttk.Frame(parent)
    frame.pack(fill='both', expand=True)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    w = canvas.get_tk_widget()
    # The Tk canvas widget ALSO defaults to white — it shows around a fixed-size
    # figure and before the first draw. Paint it the dialog's own bg too.
    from .theme import current_palette
    try:
        if dialog_dark:
            w.configure(bg=current_palette()['bg'], highlightthickness=0)
    except Exception:
        pass
    w.pack(**(pack_kw or {'fill': 'both', 'expand': True}))
    if dialog_dark:
        # Paint the dark (empty) figure SYNCHRONOUSLY now, so the Agg buffer is
        # already dark before the window is shown — draw_idle would defer it
        # past the first expose and let a white frame through.
        try:
            canvas.draw()
        except Exception:
            canvas.draw_idle()
    else:
        canvas.draw_idle()
    return frame, fig, canvas


def background_selector(parent, var=None, dark=None):
    """The standard White / Dark / Transparent / Translucent export-background
    combobox. Returns ``(var, combobox)``; the combobox is NOT packed (the
    caller places it). ``var`` defaults to Dark when the dialog is dark."""
    if dark is None:
        dark = _dialog_dark_on(parent)
    if var is None:
        var = tk.StringVar(value='Dark' if dark else 'White')
    combo = ttk.Combobox(parent, textvariable=var, width=12, state='readonly',
                         values=['White', 'Dark', 'Transparent', 'Translucent'])
    return var, combo


def ask_figure_path(parent, title="Export figure", initialfile='figure.png'):
    """Standard Save-As dialog for a figure (PNG / PDF / SVG). Returns '' if
    cancelled."""
    return filedialog.asksaveasfilename(
        parent=parent, title=title, defaultextension='.png',
        initialfile=initialfile,
        filetypes=[('PNG image', '*.png'), ('PDF', '*.pdf'),
                   ('SVG vector', '*.svg')])


def ask_csv_path(parent, title="Export CSV", initialfile='export.csv'):
    """Standard Save-As dialog for a CSV. Returns '' if cancelled."""
    return filedialog.asksaveasfilename(
        parent=parent, title=title, defaultextension='.csv',
        initialfile=initialfile,
        filetypes=[('CSV', '*.csv'), ('All files', '*.*')])


def export_figure(parent, fig, background='White', initialfile='figure.png'):
    """Prompt for a path and save ``fig`` with the chosen background, showing a
    confirmation / error dialog. Returns the path written, or None."""
    path = ask_figure_path(parent, initialfile=initialfile)
    if not path:
        return None
    try:
        savefig_background(fig, path, background=background)
        messagebox.showinfo("Export", f"Saved:\n{path}", parent=parent)
        return path
    except Exception as exc:
        messagebox.showerror("Export failed",
                             f"{type(exc).__name__}: {exc}", parent=parent)
        return None
