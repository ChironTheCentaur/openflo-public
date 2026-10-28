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
from .ui_errors import show_error


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
            except tk.TclError:
                pass
        self.bind('<Escape>', lambda _e: self.destroy())
        ed = self.editor
        if ed is not None and hasattr(ed, '_apply_titlebar_to'):
            try:
                self.after(60, lambda: ed._apply_titlebar_to(self))
            except (tk.TclError, AttributeError):
                pass


# The buttons that dismiss a dialog. They sit at the right of its button
# row, after the main action (``standard_button_order``).
DISMISS_LABELS = ('Cancel', 'Close')

# Widgets where Return belongs to the widget (a new line, a row), not to the
# window's main action.
_RETURN_OWNERS = ('Text', 'Listbox', 'Treeview')


def close_window(win):
    """Close ``win`` the way its close box does: its WM_DELETE_WINDOW
    handler when it has one (a window that asks before discarding work),
    else destroy it."""
    try:
        cmd = win.protocol('WM_DELETE_WINDOW')
    except Exception:
        cmd = ''
    if cmd:
        try:
            win.tk.eval(cmd)
            return
        except tk.TclError:
            pass
    try:
        win.destroy()
    except tk.TclError:
        pass


def _widgets(w):
    yield w
    for c in w.winfo_children():
        yield from _widgets(c)


def _own_widgets(win):
    """``win``'s widgets, not those of windows opened from it."""
    yield win
    for c in win.winfo_children():
        if isinstance(c, tk.Toplevel):
            continue
        yield from _own_widgets(c)


def default_button(win):
    """The window's main action: the one button marked ``default='active'``,
    else None."""
    for w in _own_widgets(win):
        if w.winfo_class() not in ('TButton', 'Button'):
            continue
        try:
            if str(w.cget('default')) == 'active':
                return w
        except tk.TclError:
            continue
    return None


def _press_default(win, event=None):
    """Return in a dialog: run its main action, unless the focus is a widget
    that uses Return itself."""
    focus = getattr(event, 'widget', None)
    if focus is not None:
        try:
            if focus.winfo_class() in _RETURN_OWNERS or focus.bind('<Return>'):
                return None
        except tk.TclError:
            pass
    btn = default_button(win)
    if btn is None:
        return None
    try:
        btn.invoke()                         # a disabled button does nothing
    except tk.TclError:
        return None
    return 'break'


def standardise_dialog(win):
    """What every dialog does, applied once as it opens (the editor calls this
    for each child window, so no window has to remember it):

    * it belongs to the window that opened it (``transient``): it stays in
      front of it, has no taskbar entry of its own and minimises with it;
    * Escape closes it, as its close box would (``close_window``), unless it
      binds Escape itself;
    * Return runs its main action, the button marked ``default='active'``,
      unless it binds Return itself.

    A work window rather than a dialog (an editor, the layout editor, the
    popped-out workspace) sets ``independent = True`` and is left alone: a
    separate window, which Escape and Return don't close."""
    if getattr(win, '_openflo_standard', False):
        return
    try:
        win._openflo_standard = True
    except Exception:  # noqa: BLE001 - marker attribute is best-effort
        pass
    if getattr(win, 'independent', False):
        return
    try:
        if not str(win.wm_transient()):
            # Its opener's window. A window made on the hidden root is not
            # anyone's dialog (another editor window, say).
            master = win.master
            top = master.winfo_toplevel() if master is not None else None
            if top is not None and str(top) != '.':
                win.wm_transient(top)
    except tk.TclError:
        pass
    try:
        if not win.bind('<Escape>'):
            win.bind('<Escape>', lambda _e: close_window(win))
        if not win.bind('<Return>'):
            win.bind('<Return>', lambda e: _press_default(win, e))
            win.bind('<KP_Enter>', lambda e: _press_default(win, e))
    except tk.TclError:
        pass


def button_rows(win):
    """Each frame of ``win`` holding a Cancel or Close button, with its
    buttons left to right as shown: ``[(frame, [button, ...]), ...]``."""
    rows = []
    for f in _own_widgets(win):
        kids = [c for c in f.winfo_children()
                if c.winfo_class() in ('TButton', 'Button')
                and c.winfo_ismapped()]
        if not any(str(c.cget('text')) in DISMISS_LABELS for c in kids):
            continue
        rows.append((f, sorted(kids, key=lambda c: (c.winfo_y(),
                                                     c.winfo_x()))))
    return rows


# Clustering methods by their published names.
CLUSTER_METHOD_NAMES = {'phenograph': 'PhenoGraph', 'leiden': 'Leiden',
                        'flowsom': 'FlowSOM'}

# The hover tip on every clustering resolution box (Cluster dialog, workspace).
CLUSTER_RESOLUTION_TIP = (
    "How finely the cells are split into clusters. 1.0 is the standard "
    "result. If one population comes out as several clusters (over-split, "
    "common with only a few markers), lower it, e.g. to 0.3-0.5. Higher "
    "gives more, smaller clusters.")


def named_choice(parent, var, names, **kw):
    """A read-only dropdown showing ``names`` (``{value: display name}``)
    while ``var`` keeps the value (a settings key such as 'phenograph'). A
    value not in ``names`` shows as itself. Returns the combobox, unpacked."""
    shown = tk.StringVar(master=parent, value=names.get(var.get(),
                                                        var.get()))
    back = {v: k for k, v in names.items()}
    combo = ttk.Combobox(parent, textvariable=shown, state='readonly',
                         values=list(names.values()), **kw)
    busy = []

    def _to_var(*_a):
        if not busy:
            busy.append(1)
            try:
                var.set(back.get(shown.get(), shown.get()))
            finally:
                busy.clear()

    def _to_shown(*_a):
        if not busy:
            busy.append(1)
            try:
                shown.set(names.get(var.get(), var.get()))
            finally:
                busy.clear()

    shown.trace_add('write', _to_var)
    var.trace_add('write', _to_shown)
    combo.shown_var = shown  # type: ignore[attr-defined]  # keep it alive
    return combo


def ui_scale(widget) -> float:
    """How much larger than 100 % the display draws text: Tk's point-to-pixel
    ``scaling`` over its 96-DPI value, to the nearest 25 % step (the steps
    Windows offers). 1.0 on an unscaled display, 1.5 on a Windows display at
    150 % once the process is DPI aware. Never below 1, and a display a few
    DPI off 96 (common on Linux) counts as unscaled."""
    try:
        s = float(widget.tk.call('tk', 'scaling')) / (96 / 72)
    except (tk.TclError, ValueError):
        return 1.0
    return max(1.0, round(s * 4) / 4)


# Room left for the taskbar / dock and the window frame when a window is kept
# on the screen.
_SCREEN_MARGIN_W, _SCREEN_MARGIN_H = 16, 72
_TASKBAR_H = 48


def fit_window(win, anchor=None, center=True):
    """Size ``win`` so every control it holds is shown, and keep it on screen.

    The windows set fixed pixel sizes (``geometry('1000x560')``) that were
    chosen on one machine; with another font, or text scaled up on a high-DPI
    display, their content needs more room than that and Tk's packer simply
    stops showing whatever was packed last, usually the button row. So:

    * a fixed size is scaled by the display scaling (``ui_scale``), and grown
      to at least what the content asks for (``winfo_reqwidth/height``);
    * the result is capped at the screen, and the window centred on
      ``anchor`` (the main window), or on the screen when ``anchor`` is not
      shown, and moved fully onto the screen.

    Returns the ``(width, height, x, y)`` applied, or None if it failed."""
    try:
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        max_w = max(200, sw - _SCREEN_MARGIN_W)
        max_h = max(200, sh - _SCREEN_MARGIN_H)
        req_w, req_h = win.winfo_reqwidth(), win.winfo_reqheight()
        cur_w, cur_h = win.winfo_width(), win.winfo_height()
        if cur_w <= 1 or cur_h <= 1:             # not laid out yet
            cur_w, cur_h = req_w, req_h
        s = ui_scale(win)
        if (cur_w, cur_h) != (req_w, req_h):     # a size the code chose
            cur_w, cur_h = int(cur_w * s), int(cur_h * s)
        w = min(max(cur_w, req_w), max_w)
        h = min(max(cur_h, req_h), max_h)
        x, y = win.winfo_x(), win.winfo_y()
        if center:
            box = (0, 0, sw, sh)                 # the screen, by default
            try:
                if anchor is not None and anchor.winfo_viewable():
                    box = (anchor.winfo_rootx(), anchor.winfo_rooty(),
                           anchor.winfo_width(), anchor.winfo_height())
            except tk.TclError:
                pass
            ax, ay, aw, ah = box
            x, y = ax + (aw - w) // 2, ay + (ah - h) // 2
        x = min(max(0, x), max(0, sw - w))
        y = min(max(0, y), max(0, sh - _TASKBAR_H - h))
        win.geometry(f'{w}x{h}+{x}+{y}')
        return w, h, x, y
    except tk.TclError:
        return None


def grow_to_content(win):
    """Grow an open window that its content has outgrown (text filled in
    after it opened), within the screen; never shrink it or move it more
    than keeping it on screen needs."""
    try:
        win.update_idletasks()
        w = max(win.winfo_width(), win.winfo_reqwidth())
        h = max(win.winfo_height(), win.winfo_reqheight())
        if (w, h) == (win.winfo_width(), win.winfo_height()):
            return
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w = min(w, max(200, sw - _SCREEN_MARGIN_W))
        h = min(h, max(200, sh - _SCREEN_MARGIN_H))
        x = min(max(0, win.winfo_x()), max(0, sw - w))
        y = min(max(0, win.winfo_y()), max(0, sh - _TASKBAR_H - h))
        win.geometry(f'{w}x{h}+{x}+{y}')
    except tk.TclError:
        pass


def keep_on_screen(win):
    """Move ``win`` back fully onto the screen if it is not, keeping its size.
    The editor fits each new window when it opens, but on a busy Windows
    machine the system can still apply the place it first chose afterwards
    (a cascade that ran a tall window past the bottom of a laptop screen)."""
    try:
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        if w <= 1 or h <= 1:
            return
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        x0, y0 = win.winfo_x(), win.winfo_y()
        x = min(max(0, x0), max(0, sw - w))
        y = min(max(0, y0), max(0, sh - _TASKBAR_H - h))
        if (x, y) != (x0, y0):
            win.geometry(f'+{x}+{y}')
    except tk.TclError:
        pass


def scrolled_body(parent, padding: int | tuple[int, int, int, int] = 0):
    """A frame that scrolls vertically when its window is too short for it.

    Packed to fill ``parent``; returns the inner frame to build into. The
    canvas asks for the inner frame's full size, so ``fit_window`` opens the
    window large enough to show everything when the screen allows, and the
    scrollbar only matters on a screen too small (or a display scaled too
    far) for that."""
    holder = ttk.Frame(parent)
    holder.pack(side='top', fill='both', expand=True)
    cv = tk.Canvas(holder, highlightthickness=0, bd=0)
    sb = ttk.Scrollbar(holder, orient='vertical', command=cv.yview)
    cv.configure(yscrollcommand=sb.set)
    sb.pack(side='right', fill='y')
    cv.pack(side='left', fill='both', expand=True)
    inner = ttk.Frame(cv, padding=padding)
    item = cv.create_window((0, 0), window=inner, anchor='nw')

    def _on_inner(_e=None):
        # Tall content asks for the screen less room for the title bar, a
        # button row and the taskbar; beyond that it scrolls.
        cv.configure(scrollregion=cv.bbox('all'),
                     width=inner.winfo_reqwidth(),
                     height=min(inner.winfo_reqheight(),
                                max(200, cv.winfo_screenheight() - 220)))
    inner.bind('<Configure>', _on_inner)
    cv.bind('<Configure>', lambda e: cv.itemconfigure(item, width=e.width))

    def _wheel(e):
        if cv.yview() == (0.0, 1.0):              # everything already shows
            return
        step = -1 if (getattr(e, 'delta', 0) > 0 or e.num == 4) else 1
        cv.yview_scroll(step, 'units')
    # The wheel scrolls the body while the pointer is over it. Bound
    # application-wide only for that time: the wheel event goes to the
    # widget under the pointer, which is any of the body's many children.
    wheel = ('<MouseWheel>', '<Button-4>', '<Button-5>')
    holder.bind('<Enter>', lambda _e: [cv.bind_all(q, _wheel) for q in wheel])
    holder.bind('<Leave>', lambda _e: [cv.unbind_all(q) for q in wheel])
    return inner


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
        except Exception:  # noqa: BLE001 - dark figure colour is cosmetic
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
    except Exception:  # noqa: BLE001 - dialog background colour is cosmetic
        pass
    w.pack(**(pack_kw or {'fill': 'both', 'expand': True}))
    add_copy_menu(w, fig)
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


def is_aqua(widget):
    """True when ``widget`` is drawn by Tk's macOS (aqua) windowing system."""
    try:
        return widget.tk.call('tk', 'windowingsystem') == 'aqua'
    except tk.TclError:
        return False


def right_click_sequences(widget):
    """The event sequences that mean "right-click" on ``widget``.

    ``<Button-3>`` everywhere. On aqua the right button of Tk 8.6 is
    ``<Button-2>`` and Ctrl+click is the one-button alternative, so those
    are added there only: on X11 and Windows ``<Button-2>`` is the middle
    button."""
    seqs = ['<Button-3>']
    if is_aqua(widget):
        seqs += ['<Button-2>', '<Control-Button-1>']
    return seqs


def bind_right_click(widget, func, add=None):
    """Bind ``func`` (a context-menu opener) to every right-click sequence
    of ``widget`` (see ``right_click_sequences``). Context menus go through
    this rather than binding ``<Button-3>`` directly, which never fires for
    the right button on a Mac."""
    for seq in right_click_sequences(widget):
        widget.bind(seq, func, add)


def middle_button(widget):
    """The middle mouse button's number on ``widget``: 2, except on aqua
    with Tk 8.6, where the right button is 2 and the middle one 3 (Tk 8.7
    and later number them as X11 does). A
    middle-button binding made after ``bind_right_click`` on the same
    widget replaces that button's right-click binding, so on aqua the menu
    keeps the right button and Ctrl+click."""
    return 3 if is_aqua(widget) and tk.TkVersion < 8.7 else 2


def add_copy_menu(widget, fig):
    """Right-click on ``widget`` offers "Copy figure": ``fig`` goes to the
    clipboard as an image, rendered as its export would be."""

    def _copy():
        from .clipboard_export import copy_figure
        ok, msg = copy_figure(fig)
        if not ok:
            messagebox.showerror("Copy figure", msg,
                                 parent=widget.winfo_toplevel())

    def _popup(event):
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Copy figure", command=_copy)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    bind_right_click(widget, _popup, add='+')
    widget.copy_figure = _copy  # type: ignore[attr-defined]  # for tests
    return _copy


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
    # Attached here rather than at each call site: this selector appears in ten
    # dialogs, and 'Transparent' vs 'Translucent' is the kind of distinction
    # nobody guesses right without being told.
    from .ui_tips import auto_tip
    auto_tip(combo,
             "Background of the EXPORTED figure, not the one on screen. "
             "Transparent leaves the page showing through (good over a "
             "colored slide); Translucent keeps a faint wash so light points "
             "stay visible on a dark slide.")
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
    """Prompt for a path and save ``fig`` with the chosen background, saying
    so in the status line (an error in a dialog). Returns the path written,
    or None."""
    path = ask_figure_path(parent, initialfile=initialfile)
    if not path:
        return None
    try:
        savefig_background(fig, path, background=background)
        from .ui_notice import report_saved
        report_saved(parent, path)
        return path
    except Exception as exc:
        show_error(parent, "Export failed",
                   "The figure could not be saved.", exc, path)
        return None
