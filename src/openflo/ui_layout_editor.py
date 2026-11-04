"""FlowJo-style layout editor (beta).

Pages you place plots of gated populations, stats tables, charts and text
on; drag an item to move it (it softly snaps to the grid and to other
items' edges and centres), drag its corner handle to resize it, double-click
it to edit it. Samples and gates dragged from the Samples & Gates tree or the
Pipeline (docked or popped out) land as plots, or join the plot or table
they are dropped on. The same layout re-applies to every sample: pick a
sample to preview it, or Batch export… writes every page once per sample.

The page model is ``layout_page``; drawing is ``layout_render``; the item,
colours and layer-colours dialogs are ``ui_layout_dialogs``. This module is
only the window (see the ui_*.py convention).
"""
from __future__ import annotations

import base64
import io
import os
import sys
import threading
import tkinter as tk
import warnings
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Literal

from .async_task import BusyStrip, run_async
from .errors import note as _note_failure
from .export_audit import record_export
from .export_stamp import DEFAULT_EXPORT_BACKGROUND, EXPORT_BACKGROUNDS
from .layout_page import (
    ALL_EVENTS,
    EXPORT_DPI_CHOICES,
    PAGE_SIZES,
    SCREEN_DPI,
    ZOOM_CHOICES,
    Layout,
    LayoutDoc,
    chart_item,
    clamp_box,
    grid_lines,
    item_populations,
    layout_file_part,
    plot_item,
    population_choices,
    soft_snap,
    stats_item,
    text_item,
)
from .tool_window import (
    TEXT_INPUT_CLASSES,
    bind_right_click,
    delete_key_sequences,
    middle_button,
)
from .ui_errors import show_error
from .ui_layout_dialogs import ColorsDialog, ItemDialog
from .ui_logic import resolve_channel
from .ui_tips import auto_tip

_HANDLE = 9          # resize-handle side, px
_MARGIN = 16         # canvas space around the page, px
_SNAP_PX = 8         # soft-snap reach, px
_GRID_CHOICES = {'1/8 in': 0.125, '1/4 in': 0.25, '1/2 in': 0.5,
                 '1 in': 1.0}
_Side = Literal['left', 'right', 'top', 'bottom']
_LAYOUT_TYPES = [('OpenFlo layout', '*.oflayout'), ('JSON', '*.json')]


def _photo(fig, master):
    """A Tk image of ``fig`` (PNG through Tk's own decoder; no PIL). The
    fallback for a Tk that cannot read PPM data (see ``_photo_from``)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, facecolor='white')
    return tk.PhotoImage(master=master,
                         data=base64.b64encode(buf.getvalue()).decode('ascii'))


def _raster(fig):
    """The pixels of ``fig`` as binary PPM data, made off the Tk thread.

    The page preview only: exports still go through ``savefig``. These are
    the pixels the PNG route showed (the same Agg draw, white page; a PNG
    is only a compressed copy of them), without the PNG compression that
    cost as much as the drawing itself. Touches no Tk object."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    h, w = rgba.shape[:2]
    return b'P6\n%d %d\n255\n' % (w, h) + rgba[..., :3].tobytes()


_PPM_OK = [True]     # cleared once a Tk refuses PPM data; PNG from then on

# matplotlib's mathtext parser is one shared object holding per-parse state,
# so two threads parsing at once (the '$10^{k}$' tick labels) corrupt each
# other's results, and a wrong result can stay in its cache for exports.
# Drawing pixels off the window thread makes that possible; this lock makes
# every parse one at a time.
_MATHTEXT_LOCK = threading.RLock()


def _serialise_mathtext():
    """Make ``MathTextParser.parse`` take ``_MATHTEXT_LOCK``. Installed
    once, whatever the number of calls."""
    import functools

    from matplotlib import mathtext
    parse = mathtext.MathTextParser.parse
    if getattr(parse, '_openflo_locked', False):
        return

    @functools.wraps(parse)
    def locked(self, *args, **kwargs):
        with _MATHTEXT_LOCK:
            return parse(self, *args, **kwargs)
    locked.__dict__['_openflo_locked'] = True
    mathtext.MathTextParser.parse = locked


def _photo_from(ppm, fig, master):
    """A Tk image from ``_raster``'s PPM data (on the Tk thread), or from a
    PNG of ``fig`` where this Tk cannot read PPM data."""
    if _PPM_OK[0]:
        try:
            return tk.PhotoImage(master=master, data=ppm, format='ppm')
        except tk.TclError:
            _PPM_OK[0] = False
    return _photo(fig, master)


class LayoutEditorWindow(tk.Toplevel):
    """The layout editor. One per editor; the layout document lives on the
    editor (``_layout_doc``) so it survives closing the window."""

    # A work window, not a dialog (tool_window.standardise_dialog).
    independent = True

    def __init__(self, editor):
        super().__init__(editor)
        self.ed = editor
        self.title("Layout editor (beta)")
        self.geometry("1240x860")
        doc = getattr(editor, '_layout_doc', None)
        if not isinstance(doc, LayoutDoc):
            # A new layout exports at the Preferences dpi; a saved one keeps
            # its own.
            from .theme import export_dpi
            doc = LayoutDoc(export_dpi=export_dpi())
            editor._layout_doc = doc
        self.doc = doc
        self.page_idx = 0
        self._sel = None             # selected item id
        self._drag = None            # (mode, iid, start_px, start_box)
        self._images = {}            # cache key -> (PhotoImage, outside, ax)
        # Items are drawn into figures on the window thread (drawing borrows
        # the editor's plot settings for a moment) and turned into pixels on
        # a background thread, one at a time, so the window keeps answering
        # the mouse. A newer redraw makes older pending pixels moot.
        self._gen = 0                # redraw generation
        self._pending = 0            # items of this generation still drawing
        self._raster_busy = False    # a background thread is drawing
        self._raster_queue = None    # (gen, jobs) waiting for that thread
        _serialise_mathtext()
        self._shown = {}             # iid -> PhotoImage on the canvas
        self._outside = {}           # iid -> fraction outside its zoom
        self._geom = (1.0, 0, 0)     # px per inch, page origin x, y
        self._redraw_after = None
        self._zoom = 'Fit'
        self._axes = {}              # iid -> its plot's axes, for zooming
        # Undo and redo: the layout as it was before each change.
        self._undo_stack: list = []  # (layout JSON, page index)
        self._redo_stack: list = []
        self._last_state = None

        self._build_toolbar()
        self._build_tabs()
        # The status line is packed before the page so it keeps its height,
        # and wraps instead of running off both ends.
        self.status = tk.StringVar(value=(
            "Drag samples or gates in. Drag to move, a corner to resize, "
            "double-click to edit; Ctrl+wheel zooms the page; right-click "
            "for more. Ctrl+Z undoes, Ctrl+S saves the session."))
        self.status_label = ttk.Label(self, textvariable=self.status,
                                      anchor='w', justify='left',
                                      padding=(8, 3), wraplength=1200)
        self.status_label.pack(fill='x', side='bottom')
        self.status_label.bind('<Configure>', lambda e: self.status_label
                               .configure(wraplength=max(200, e.width - 16)))
        # A batch export's progress and Cancel, while it runs.
        self._busy = BusyStrip(self, padding=(8, 0))
        self._busy.pack(fill='x', side='bottom')
        body = ttk.PanedWindow(self, orient='horizontal')
        body.pack(fill='both', expand=True)
        self._body = body
        self._build_host(body)
        cf = ttk.Frame(body)
        body.add(cf, weight=1)
        self.canvas = tk.Canvas(cf, highlightthickness=0, bg='#8a8d94')
        xs = ttk.Scrollbar(cf, orient='horizontal', command=self.canvas.xview)
        ys = ttk.Scrollbar(cf, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        ys.grid(row=0, column=1, sticky='ns')
        xs.grid(row=1, column=0, sticky='ew')
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)
        try:
            from .theme import current_palette
            self.canvas.configure(bg=current_palette()['bg'])
        except Exception:                             # noqa: BLE001 - canvas colour is cosmetic
            pass

        c = self.canvas
        c.bind('<Configure>', lambda _e: self._schedule_redraw())
        c.bind('<ButtonPress-1>', self._on_press)
        c.bind('<B1-Motion>', self._on_drag)
        c.bind('<ButtonRelease-1>', self._on_release)
        c.bind('<Double-Button-1>', self._on_double)
        bind_right_click(c, self._on_context)
        c.bind('<MouseWheel>', self._on_wheel)
        # After the right-click binding: on aqua with Tk 8.6 the middle
        # button is 3, and this replaces the menu's <Button-3> there.
        mid = middle_button(c)
        c.bind(f'<ButtonPress-{mid}>', self._on_pan_start)
        c.bind(f'<B{mid}-Motion>', self._on_pan_move)
        c.bind(f'<ButtonRelease-{mid}>', self._on_pan_end)
        c.bind('<Button-4>', lambda e: self._on_wheel(e, 120))
        c.bind('<Button-5>', lambda e: self._on_wheel(e, -120))
        for seq in delete_key_sequences(self):
            self.bind(seq, self._on_delete_key)
        self.bind('<F5>', lambda _e: self._refresh())
        for keys, fn in (('z', self._undo), ('y', self._redo),
                         ('Z', self._redo), ('s', self._save_session)):
            for mod in (('Control', 'Command') if sys.platform == 'darwin'
                        else ('Control',)):
                self.bind(f'<{mod}-{"Shift-" if keys == "Z" else ""}'
                          f'Key-{keys}>',
                          lambda _e, f=fn: (f(), 'break')[1])
        self._redraw()

    @property
    def layout(self) -> Layout:
        """The page being edited; clamps ``page_idx`` into range first."""
        self.page_idx = min(max(self.page_idx, 0), len(self.doc.pages) - 1)
        return self.doc.pages[self.page_idx]

    # ── toolbar ──────────────────────────────────────────────────────────
    #
    # Two rows, each read left to right as the work goes:
    #   1. what you do to the page -- insert items, edit the selected one --
    #      then, on the right, which sample it shows and how it leaves
    #      (Export, with its resolution on the button; the layout File menu);
    #   2. the pages themselves -- their tabs, then the page's own settings
    #      (size, grid, snap) and the on-screen zoom.
    # Groups are split by one separator width and one gap size throughout.

    _GAP = 6                      # between groups, px
    _TIGHT = 2                    # inside a group, px

    def _toolbar_button(self, parent, text, cmd, tip, **kw):
        # Natural width: the theme's default minimum would make every button the
        # same wide slab, and the row would run off a 1240px window.
        kw.setdefault('width', 0)
        b = ttk.Button(parent, text=text, command=cmd, **kw)
        b.pack(side='left', padx=(0, self._TIGHT))
        auto_tip(b, tip)
        return b

    def _toolbar_sep(self, parent):
        ttk.Separator(parent, orient='vertical').pack(
            side='left', fill='y', padx=self._GAP, pady=2)

    def _toolbar_combo(self, parent, var, values, tip, width, cmd):
        cb = ttk.Combobox(parent, textvariable=var, width=width,
                          state='readonly', values=list(values))
        cb.pack(side='left', padx=(0, self._TIGHT))
        cb.bind('<<ComboboxSelected>>', lambda _e: cmd())
        auto_tip(cb, tip)
        return cb

    def _toolbar_menu(self, parent, text, tip):
        mb = ttk.Menubutton(parent, text=text, width=0)
        menu = tk.Menu(mb, tearoff=0)
        try:
            self.ed._theme_menu(menu)
        except Exception:                             # noqa: BLE001 - menu colours are cosmetic
            pass
        mb.configure(menu=menu)
        mb.pack(side='left', padx=(0, self._TIGHT))
        auto_tip(mb, tip)
        return mb, menu

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(8, 6, 8, 4))
        bar.pack(fill='x', side='top')
        # Two groups that share a row while the window is wide enough and
        # wrap onto two rows when it is not (_reflow_toolbar): on one fixed
        # row the packer drops Export and Layout off the end.
        left = ttk.Frame(bar)
        left.grid(row=0, column=0, sticky='w')

        # Insert
        b = self._toolbar_button
        b(left, "＋ Plot", self._add_plot,
          "Add a plot of a population. With a gate selected in the main "
          "window it starts as that gate's plot: its parent population on "
          "the gate's axes. Dragging a sample or gate onto the page does "
          "the same in one step.")
        b(left, "＋ Stats", self._add_stats,
          "Add a table of counts, frequencies or channel statistics for "
          "chosen populations and samples.")
        b(left, "＋ Chart", self._add_chart,
          "Add a violin, box or bar chart comparing populations across "
          "samples.")
        b(left, "＋ Text", self._add_text,
          "Add a text box. {sample} is replaced by the sample name.")
        b(left, "＋ Strategy", self._add_strategy_page,
          "Add a page laying out the gating strategy: one plot per gating "
          "step, each the parent population on the gate's axes. With a gate "
          "selected in the main window, only the steps down to it. Batch "
          "export draws it for every sample.")
        self._toolbar_sep(left)
        # The selected item
        # Greyed out while nothing is selected (_sync_item_buttons), when a
        # press would do nothing.
        self._item_buttons = (
            b(left, "Edit…", self._edit_selected,
              "Change the selected item's settings (or double-click it). "
              "Click an item on the page first."),
            b(left, "Duplicate", self._duplicate_selected,
              "Copy the selected item, offset slightly. Click an item on "
              "the page first."),
            b(left, "Delete", self._delete_selected,
              "Remove the selected item from the page (Delete key). Click "
              "an item on the page first."))
        self._sync_item_buttons()

        # Right side, packed right to left: File, Export, then the sample.
        right = ttk.Frame(bar)
        # Follows the rest instead of pinning to the window's right edge:
        # on a wide window an edge-pinned group drifted away from its row.
        right.grid(row=0, column=1, sticky='w', padx=(self._GAP, 0))
        self._toolbar_groups = (bar, left, right)
        self._toolbar_wrapped = False
        bar.bind('<Configure>', self._reflow_toolbar, add='+')
        self.sample_var = tk.StringVar()
        names = list(getattr(self.ed, '_sample_order', []) or [])
        first = (self.ed._active_sample
                 if getattr(self.ed, '_active_sample', None) in names
                 else (names[0] if names else ''))
        self.sample_var.set(first)
        ttk.Label(right, text="Sample").pack(side='left',
                                             padx=(0, self._TIGHT + 2))
        b(right, "◀", lambda: self._step_sample(-1),
          "Show the previous sample.", width=0)
        self.sample_combo = self._toolbar_combo(
            right, self.sample_var, names,
            "The sample the page shows. The layout is the same for every "
            "sample; populations are matched by name.", 26, self._redraw)
        b(right, "▶", lambda: self._step_sample(1),
          "Show the next sample.", width=0)
        b(right, "⟳", self._refresh,
          "Refresh: redraw from the current samples and gates, after a "
          "change in the main window (F5).", width=0)
        self._toolbar_sep(right)

        # Export: its resolution is on the button, so it is never hidden.
        self.dpi_var = tk.StringVar(value=str(self.doc.export_dpi))
        self.bg_var = tk.StringVar(value=DEFAULT_EXPORT_BACKGROUND)
        self._export_btn, em = self._toolbar_menu(
            right, '', "Export the page or every sample, and set the export "
            "resolution (dots per inch) and background.")
        em.add_command(label="Export page…", command=self._export_page)
        em.add_command(label="Batch export (every sample)…",
                       command=self._export_batch)
        em.add_separator()
        res = tk.Menu(em, tearoff=0)
        for d in EXPORT_DPI_CHOICES:
            res.add_radiobutton(
                label=f"{d} dpi" + ("  (publication)" if d == 300 else ""),
                value=str(d), variable=self.dpi_var, command=self._on_dpi)
        res.add_separator()
        res.add_command(label="Custom…", command=self._ask_dpi)
        em.add_cascade(label="Resolution", menu=res)
        bgm = tk.Menu(em, tearoff=0)
        # Dark is left out here: its recolouring is written for plot
        # windows, not for the free text and colours of a layout page.
        for v in (b for b in EXPORT_BACKGROUNDS if b != 'Dark'):
            bgm.add_radiobutton(label=v, value=v, variable=self.bg_var)
        em.add_cascade(label="Background", menu=bgm)
        for m in (res, bgm):
            try:
                self.ed._theme_menu(m)
            except Exception:                         # noqa: BLE001 - menu colours are cosmetic
                pass
        self.dpi_var.trace_add('write', lambda *_: self._label_export())
        self._label_export()

        _fb, fm = self._toolbar_menu(right, "Layout",
                                "Open or save the layout file, and the "
                                "default colors.")
        fm.add_command(label="Open layout…", command=self._open_layout)
        fm.add_command(label="Save layout…", command=self._save_layout)
        fm.add_separator()
        fm.add_command(label="Colors…", command=self._edit_default_colors)

    def _reflow_toolbar(self, _e=None):
        """Put the sample/Export/Layout group on its own row when the bar is
        narrower than both groups side by side, and back when it is not."""
        bar, left, right = self._toolbar_groups
        need = (left.winfo_reqwidth() + right.winfo_reqwidth() + self._GAP
                + 16)                          # the bar's own padding
        wrap = bar.winfo_width() < need
        if wrap == self._toolbar_wrapped:
            return
        self._toolbar_wrapped = wrap
        if wrap:
            right.grid_configure(row=1, column=0, padx=0, pady=(4, 0))
        else:
            right.grid_configure(row=0, column=1, padx=(self._GAP, 0),
                                 pady=0)

    def _label_export(self):
        btn = getattr(self, '_export_btn', None)
        if btn is not None:
            btn.configure(text=f"Export · {self.dpi_var.get()} dpi")

    def _ask_dpi(self):
        dpi = simpledialog.askinteger(
            "Export resolution", "Dots per inch:", minvalue=50,
            maxvalue=2400, initialvalue=self.doc.export_dpi, parent=self)
        if dpi:
            self.dpi_var.set(str(dpi))
            self._on_dpi()

    # ── pages ────────────────────────────────────────────────────────────
    def _build_tabs(self):
        """Second row: the page tabs, then this page's settings and the
        on-screen zoom."""
        row = ttk.Frame(self, padding=(8, 0, 8, 4))
        row.pack(fill='x', side='top')
        self.tabs = ttk.Notebook(row, height=1)
        self.tabs.pack(side='left')
        self.tabs.bind('<<NotebookTabChanged>>', lambda _e: self._on_tab())
        bind_right_click(self.tabs, self._on_tab_menu)
        b = self._toolbar_button
        b(row, "＋", self._new_page, "Add an empty page.", width=0)
        self._pages_btn = b(
            row, "▾", self._pages_menu_at_button,
            "Pages: rename, duplicate or delete this page; bring in pages "
            "from another project's layout file, or send this page to one.",
            width=0)
        self._toolbar_sep(row)
        self.page_var = tk.StringVar(value=self.layout.page)
        self._toolbar_combo(row, self.page_var, PAGE_SIZES,
                       "Paper size and orientation of this page. Items keep "
                       "their place as a fraction of the page.", 15,
                       self._on_page_size)
        self.grid_var = tk.BooleanVar(value=False)
        gr = ttk.Checkbutton(row, text="Grid", variable=self.grid_var,
                             command=self._redraw)
        gr.pack(side='left', padx=(self._GAP, self._TIGHT))
        auto_tip(gr, "Show a grid on the page. Items softly snap to its "
                     "lines while Snap is on.")
        self.grid_step_var = tk.StringVar(value='1/4 in')
        self._toolbar_combo(row, self.grid_step_var, _GRID_CHOICES,
                       "Grid spacing.", 6, self._redraw)
        self.snap_var = tk.BooleanVar(value=True)
        sn = ttk.Checkbutton(row, text="Snap", variable=self.snap_var)
        sn.pack(side='left', padx=(self._GAP, 0))
        auto_tip(sn, "Soft snapping: an item being moved or resized is "
                     "pulled onto nearby grid lines and other items' edges, "
                     "corners and centres, and guides show what it lines up "
                     "with. Off: free placement.")
        self._toolbar_sep(row)
        ttk.Label(row, text="Zoom").pack(side='left',
                                         padx=(0, self._TIGHT + 2))
        self.zoom_var = tk.StringVar(value='Fit')
        self._toolbar_combo(row, self.zoom_var, list(ZOOM_CHOICES) + ['Custom…'],
                       "On-screen resolution of the page: Fit fills the "
                       "window, 100% is 96 pixels per inch (Ctrl+wheel "
                       "changes it too). Exports use the Export resolution.",
                       8, self._on_zoom)
        self._rebuild_tabs()

    def _rebuild_tabs(self):
        self._tabs_busy = True
        try:
            for t in self.tabs.tabs():
                self.tabs.forget(t)
            for p in self.doc.pages:
                self.tabs.add(ttk.Frame(self.tabs, height=1), text=p.name)
            self.tabs.select(self.page_idx)
        finally:
            self._tabs_busy = False

    def _on_tab(self):
        if getattr(self, '_tabs_busy', False):
            return
        try:
            idx = self.tabs.index('current')
        except tk.TclError:
            return
        if idx != self.page_idx:
            self._go_page(idx)

    def _go_page(self, idx):
        self.page_idx = idx
        self._sel = None
        self.page_var.set(self.layout.page)
        self._redraw()

    def _new_page(self):
        idx = self.doc.add_page(Layout(page=self.layout.page,
                                       name=f'Page {len(self.doc.pages) + 1}'))
        self.page_idx = idx
        self._rebuild_tabs()
        self._go_page(idx)

    def _pages_menu(self):
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Rename page…", command=self._rename_page)
        m.add_command(label="Duplicate page", command=self._duplicate_page)
        m.add_command(label="Delete page", command=self._delete_page,
                      state='normal' if len(self.doc.pages) > 1
                      else 'disabled')
        m.add_command(label="New gating strategy page",
                      command=self._add_strategy_page)
        m.add_separator()
        m.add_command(label="Import pages from another layout…",
                      command=self._import_pages)
        m.add_command(label="Send this page to another layout…",
                      command=self._send_page)
        return m

    def _pages_menu_at_button(self):
        b = self._pages_btn
        m = self._pages_menu()
        try:
            m.tk_popup(b.winfo_rootx(), b.winfo_rooty() + b.winfo_height())
        finally:
            m.grab_release()

    def _on_tab_menu(self, event):
        try:
            idx = self.tabs.index(f'@{event.x},{event.y}')
        except tk.TclError:
            return
        if idx != self.page_idx:
            self.tabs.select(idx)
            self._go_page(idx)
        m = self._pages_menu()
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    def _rename_page(self):
        name = simpledialog.askstring("Rename page", "Page name:",
                                      initialvalue=self.layout.name,
                                      parent=self)
        if name and name.strip():
            self.layout.name = self.doc.unique_name(name.strip()) \
                if name.strip() != self.layout.name else name.strip()
            self._rebuild_tabs()
            self._record_change()      # its own undo step (no redraw follows)

    def _duplicate_page(self):
        self.page_idx = self.doc.duplicate_page(self.page_idx)
        self._rebuild_tabs()
        self._go_page(self.page_idx)

    def _delete_page(self):
        if len(self.doc.pages) <= 1:
            return
        if not messagebox.askyesno(
                "Delete page", f"Delete page '{self.layout.name}' and its "
                f"{len(self.layout.items)} item(s)?", parent=self):
            return
        self.doc.remove_page(self.page_idx)
        self.page_idx = max(0, self.page_idx - 1)
        self._rebuild_tabs()
        self._go_page(self.page_idx)

    def _import_pages(self):
        path = filedialog.askopenfilename(
            parent=self, title="Import pages from a layout",
            filetypes=_LAYOUT_TYPES)
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                other = LayoutDoc.loads(fh.read())
        except Exception as exc:                      # noqa: BLE001 - a file that does not parse as a layout is shown in an error dialog; nothing is imported
            show_error(self, "Layout editor",
                       "The layout could not be read.", exc, path)
            return
        stem = os.path.splitext(os.path.basename(path))[0]
        new = self.doc.import_pages(other, prefix=f'{stem}: ')
        self._rebuild_tabs()
        if new:
            self._go_page(new[0])
            self.tabs.select(new[0])
        self.status.set(f"Imported {len(new)} page(s). Their populations "
                        "are matched by name in this project's samples.")

    def _send_page(self):
        """Add this page to another layout file (another project's), or to
        a new one."""
        path = filedialog.asksaveasfilename(
            parent=self, title="Send this page to a layout file",
            defaultextension='.oflayout', confirmoverwrite=False,
            initialfile='layout.oflayout', filetypes=_LAYOUT_TYPES)
        if not path:
            return
        try:
            if os.path.exists(path):
                with open(path, encoding='utf-8') as fh:
                    other = LayoutDoc.loads(fh.read())
                other.import_pages(LayoutDoc([self.layout]))
            else:
                other = LayoutDoc([Layout.from_dict(self.layout.to_dict())],
                                  export_dpi=self.doc.export_dpi)
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(other.dumps())
        except Exception as exc:                      # noqa: BLE001 - any read, merge or write failure is shown in an error dialog
            show_error(self, "Layout editor",
                       "The layout could not be written.", exc, path)
            return
        self.status.set(f"Sent '{self.layout.name}' to "
                        f"{os.path.basename(path)}")

    # ── geometry ─────────────────────────────────────────────────────────
    def _fit(self):
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)
        pw, ph = self.layout.page_size
        if self._zoom == 'Fit':
            ppi = max(min((cw - 2 * _MARGIN) / pw,
                          (ch - 2 * _MARGIN) / ph), 10.0)
        else:
            ppi = max(float(self._zoom) / 100.0 * SCREEN_DPI, 10.0)
        tw, th = pw * ppi + 2 * _MARGIN, ph * ppi + 2 * _MARGIN
        ox = max((cw - pw * ppi) / 2.0, _MARGIN)
        oy = max((ch - ph * ppi) / 2.0, _MARGIN)
        self.canvas.configure(scrollregion=(0, 0, max(tw, cw), max(th, ch)))
        self._geom = (ppi, ox, oy)

    def _box_px(self, box):
        ppi, ox, oy = self._geom
        pw, ph = self.layout.page_size
        x, y, w, h = box
        return (ox + x * pw * ppi, oy + y * ph * ppi,
                ox + (x + w) * pw * ppi, oy + (y + h) * ph * ppi)

    def _cxy(self, event):
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _to_page(self, px, py):
        ppi, ox, oy = self._geom
        pw, ph = self.layout.page_size
        return (px - ox) / (pw * ppi), (py - oy) / (ph * ppi)

    def _grid_step(self):
        """``(sx, sy)`` grid spacing in page fractions, or (0, 0) off."""
        if not self.grid_var.get():
            return 0.0, 0.0
        inch = _GRID_CHOICES.get(self.grid_step_var.get(), 0.25)
        pw, ph = self.layout.page_size
        return inch / pw, inch / ph

    def _snap_targets(self, skip):
        xs, ys = self.layout.edges(skip)
        sx, sy = self._grid_step()
        return xs + grid_lines(sx), ys + grid_lines(sy)

    # ── drawing ──────────────────────────────────────────────────────────
    def _schedule_redraw(self):
        if self._redraw_after is not None:
            try:
                self.after_cancel(self._redraw_after)
            except tk.TclError:
                pass
        self._redraw_after = self.after(120, self._redraw)

    def _sample(self):
        nm = self.sample_var.get()
        return nm if nm in getattr(self.ed, '_samples', {}) else None

    def _gate_sig(self, samples):
        out = []
        for s in samples:
            try:
                out.append(tuple(sorted(self.ed._plotcache_sigs(s).items())))
            except Exception:                         # noqa: BLE001 - a sample whose gate signature cannot be read gets an empty one, so its items just redraw
                out.append(())
        return tuple(out)

    def _redraw(self):
        """Redraw the page; items whose look has not changed reuse their
        cached image. The others are drawn into figures here and turned
        into pixels in the background (``_raster_next``); until theirs
        arrive they show their last image, or a placeholder."""
        import json

        from .layout_page import item_samples
        from .layout_render import RenderContext, render_item
        self._redraw_after = None
        self._gen += 1
        gen = self._gen
        self.doc.ensure_uids()          # new or copied items get their ids
        c = self.canvas
        c.delete('all')
        self._fit()
        ppi = self._geom[0]
        x0, y0, x1, y1 = self._box_px((0, 0, 1, 1))
        c.create_rectangle(x0 + 3, y0 + 3, x1 + 3, y1 + 3, fill='#55575c',
                           outline='')
        c.create_rectangle(x0, y0, x1, y1, fill='white', outline='#999999')
        self._draw_grid()
        sample = self._sample()
        if not self.layout.items:
            c.create_text((x0 + x1) / 2, (y0 + y1) / 2, fill='#888888',
                          justify='center', font='OpenFloLarge',
                          text="Empty page.\nDrag samples or gates here, or "
                               "use ＋ Plot, ＋ Stats or ＋ Chart.")
        rc = None
        from .layout_render import color_prefs
        colsig = color_prefs()
        colsig = (tuple(colsig[0]), colsig[1])
        last = self._shown
        self._shown, self._outside, self._axes = {}, {}, {}
        jobs = []                       # (iid, key, fig, outside, ax)
        for iid in self.layout.order:
            it = self.layout.items[iid]
            bx0, by0, bx1, by1 = self._box_px(it['box'])
            wpx, hpx = int(bx1 - bx0), int(by1 - by0)
            sig = self._gate_sig(item_samples(it, sample)) if sample else ()
            key = (self.page_idx, iid, sample,
                   json.dumps(it, sort_keys=True, default=str), wpx, hpx, sig,
                   colsig)
            cached = self._images.get(key)
            if cached is None and sample is not None:
                if rc is None:
                    rc = RenderContext(self.ed, sample)
                try:
                    fig = render_item(self.ed, self.layout, iid, sample, ppi,
                                      rc=rc)
                    info = rc.info.get(iid, {})
                    # An exploded plot shows all its data: no Zoom to fit.
                    outside = (0.0 if info.get('exploded')
                               else info.get('outside', 0.0))
                    jobs.append((iid, key, fig, outside, info.get('ax')))
                    self._outside[iid] = outside
                    if info.get('ax') is not None:
                        self._axes[iid] = info['ax']
                except Exception as exc:              # noqa: BLE001 - one item failing to draw shows in the status line; the rest of the page still draws
                    self.status.set(f"Could not draw an item: {exc}")
            if cached is not None:
                img, outside, ax = cached
                self._shown[iid] = img
                self._outside[iid] = outside
                if ax is not None:
                    self._axes[iid] = ax
                c.create_image(bx0, by0, image=img, anchor='nw',
                               tags=('item', f'img:{iid}'))
            elif jobs and jobs[-1][0] == iid and iid in last:
                # Its last image stands in until the new one is drawn.
                self._shown[iid] = last[iid]
                c.create_image(bx0, by0, image=last[iid], anchor='nw',
                               tags=('item', f'img:{iid}'))
            else:
                c.create_rectangle(bx0, by0, bx1, by1, outline='#bbbbbb',
                                   dash=(3, 3), tags=('item', f'img:{iid}'))
                c.create_text((bx0 + bx1) / 2, (by0 + by1) / 2,
                              text=(f"{it['kind']}\ndrawing…"
                                    if jobs and jobs[-1][0] == iid
                                    else it['kind']),
                              justify='center', fill='#999999',
                              tags=('item', f'img:{iid}'))
        self._draw_fit_badges()
        self._draw_uids()
        self._draw_selection()
        self._record_change()
        self._pending = len(jobs)
        self._raster_next(gen, jobs)

    def _raster_next(self, gen, jobs):
        """Turn the next figure of redraw ``gen`` into pixels on a
        background thread, then the one after. Stops once a newer redraw
        has started (its own chain draws what it still needs). One thread
        at a time: while one draws, the newest chain waits for it."""
        if not jobs:
            return
        if self._raster_busy:
            self._raster_queue = (gen, jobs)
            return
        self._raster_busy = True
        iid, key, fig, outside, ax = jobs[0]
        run_async(self, lambda: _raster(fig),
                  on_done=lambda ppm: self._place(gen, iid, key, fig, ppm,
                                                  outside, ax),
                  on_error=lambda exc: self._raster_failed(gen, iid, exc),
                  on_finally=lambda: self._raster_done(gen, jobs[1:]))

    def _raster_done(self, gen, rest):
        self._raster_busy = False
        queued, self._raster_queue = self._raster_queue, None
        if gen == self._gen:
            self._pending = len(rest)
            self._raster_next(gen, rest)
        elif queued is not None:
            self._raster_next(*queued)

    def _raster_failed(self, gen, iid, exc):
        if gen != self._gen:
            return
        self.status.set(f"Could not draw an item: {exc}")
        for tid in self.canvas.find_withtag(f'img:{iid}'):
            if self.canvas.type(tid) == 'text':
                kind = self.layout.items.get(iid, {}).get('kind', 'item')
                self.canvas.itemconfigure(tid, text=f"{kind}\ncould not draw")

    def _place(self, gen, iid, key, fig, ppm, outside, ax):
        """Show an item's new pixels (Tk thread), unless a newer redraw has
        made them moot, and cache them for the redraws to come."""
        if gen != self._gen or iid not in self.layout.items:
            return
        img = _photo_from(ppm, fig, self)
        self._images = {k: v for k, v in self._images.items()
                        if k[:3] != key[:3]}
        self._images[key] = (img, outside, ax)
        self._shown[iid] = img
        c = self.canvas
        c.delete(f'img:{iid}')
        bx0, by0, _bx1, _by1 = self._box_px(self.layout.items[iid]['box'])
        c.create_image(bx0, by0, image=img, anchor='nw',
                       tags=('item', f'img:{iid}'))
        # Back into the page's stacking order, under the ids, the Zoom to
        # fit badges and the selection.
        for other in self.layout.order:
            c.tag_raise(f'img:{other}')
        for tag in ('fit', 'uid', 'sel'):
            c.tag_raise(tag)

    # ── Undo, redo and saving ──────────────────────────────────────────────
    _UNDO_LIMIT = 60

    def _doc_state(self):
        import json
        return json.dumps(self.doc.to_dict(), sort_keys=True, default=str)

    def _record_change(self):
        """Every redraw follows a change, a resize or a scroll: when the
        layout differs from the one last seen, that one goes on the undo
        stack and the session is marked unsaved."""
        if self._drag is not None:
            return                        # one drag is one change, at release
        try:
            state = self._doc_state()
        except Exception as exc:  # noqa: BLE001 - reported via note()
            _note_failure("record the layout undo state", exc)
            return
        if state == self._last_state:
            return
        if self._last_state is not None:
            self._undo_stack.append((self._last_state, self.page_idx))
            del self._undo_stack[:-self._UNDO_LIMIT]
            self._redo_stack.clear()
            self._note_unsaved()
        self._last_state = state

    def _note_unsaved(self):
        note = getattr(self.ed, '_note_unsaved', None)
        if note is not None:
            try:
                note()
            except Exception:  # noqa: BLE001 - post-change callback is best-effort
                pass

    def _restore(self, state, page):
        import json
        doc = LayoutDoc.from_dict(json.loads(state))
        self.doc = doc
        self.ed._layout_doc = doc
        self._last_state = state
        self._sel = None
        self._images.clear()
        self.dpi_var.set(str(doc.export_dpi))
        self._refresh_host()
        self._rebuild_tabs()
        self._go_page(min(max(page, 0), len(doc.pages) - 1))
        self._note_unsaved()

    def _undo(self):
        if not self._undo_stack:
            self.status.set("Nothing to undo.")
            return
        state, page = self._undo_stack.pop()
        self._redo_stack.append((self._last_state, self.page_idx))
        self._restore(state, page)
        self.status.set("Undone. Ctrl+Y redoes it.")

    def _redo(self):
        if not self._redo_stack:
            self.status.set("Nothing to redo.")
            return
        state, page = self._redo_stack.pop()
        self._undo_stack.append((self._last_state, self.page_idx))
        self._restore(state, page)
        self.status.set("Redone.")

    def _save_session(self):
        """Ctrl+S: save the session, which carries the layout."""
        save = getattr(self.ed, '_save_session', None)
        if save is None:
            return
        save()
        self.status.set("Saved the session with its layout.")

    def _draw_uids(self):
        """Each item's id in its top-left corner (screen only, not in
        exports), so look-alike plots can be told apart."""
        c = self.canvas
        c.delete('uid')
        for it in self.layout.items.values():
            u = it.get('uid')
            if not u:
                continue
            x0, y0, _x1, _y1 = self._box_px(it['box'])
            t = c.create_text(x0 + 4, y0 + 3, anchor='nw', text=u,
                              fill='#5b6270', font='OpenFloHeading',
                              tags=('uid',))
            bb = c.bbox(t)
            if bb:
                r = c.create_rectangle(bb[0] - 2, bb[1] - 1, bb[2] + 2,
                                       bb[3] + 1, fill='#eef0f4',
                                       outline='#c9ccd3', tags=('uid',))
                c.tag_lower(r, t)

    def _draw_grid(self):
        sx, sy = self._grid_step()
        if not sx:
            return
        c = self.canvas
        x0, y0, x1, y1 = self._box_px((0, 0, 1, 1))
        for gx in grid_lines(sx):
            px = x0 + gx * (x1 - x0)
            c.create_line(px, y0, px, y1, fill='#e3e6ec', tags=('grid',))
        for gy in grid_lines(sy):
            py = y0 + gy * (y1 - y0)
            c.create_line(x0, py, x1, py, fill='#e3e6ec', tags=('grid',))

    def _draw_fit_badges(self):
        """A plot whose kept zoom hides a substantial part of its data gets
        a 'Zoom to fit' badge; clicking it (only then) refits."""
        from .layout_render import OUTSIDE_WARN
        c = self.canvas
        c.delete('fit')
        for iid, frac in self._outside.items():
            if frac <= OUTSIDE_WARN or iid not in self.layout.items:
                continue
            bx0, _by0, _bx1, by1 = self._box_px(self.layout.items[iid]['box'])
            t = c.create_text(bx0 + 8, by1 - 6, anchor='sw',
                              text=f"⤢ Zoom to fit ({frac:.0%} outside)",
                              fill='white', font='OpenFloHeading',
                              tags=('fit', f'fit:{iid}'))
            bb = c.bbox(t)
            if bb:
                r = c.create_rectangle(bb[0] - 4, bb[1] - 2, bb[2] + 4,
                                       bb[3] + 2, fill='#d9822b',
                                       outline='', tags=('fit', f'fit:{iid}'))
                c.tag_lower(r, t)

    def _draw_selection(self, guides=(None, None)):
        c = self.canvas
        c.delete('sel')
        if self._sel not in self.layout.items:
            self._sel = None
            self._sync_item_buttons()
            return
        self._sync_item_buttons()
        bx0, by0, bx1, by1 = self._box_px(self.layout.items[self._sel]['box'])
        c.create_rectangle(bx0, by0, bx1, by1, outline='#2f6fde', width=2,
                           tags=('sel',))
        c.create_rectangle(bx1 - _HANDLE, by1 - _HANDLE, bx1, by1,
                           fill='#2f6fde', outline='white', tags=('sel',))
        gx, gy = guides
        px0, py0, px1, py1 = self._box_px((0, 0, 1, 1))
        if gx is not None:
            px = px0 + gx * (px1 - px0)
            c.create_line(px, py0, px, py1, fill='#e0457b', dash=(4, 3),
                          tags=('sel',))
        if gy is not None:
            py = py0 + gy * (py1 - py0)
            c.create_line(px0, py, px1, py, fill='#e0457b', dash=(4, 3),
                          tags=('sel',))

    def _sync_item_buttons(self):
        """Edit…, Duplicate and Delete act on the selected item: grey them
        out while there is none."""
        state = (['!disabled'] if self._sel in self.layout.items
                 else ['disabled'])
        for btn in getattr(self, '_item_buttons', ()):
            try:
                btn.state(state)
            except tk.TclError:
                pass

    def _nothing_selected(self):
        """True (with a status hint) when no item is selected, for the
        keyboard and menu paths to the item commands."""
        if self._sel in self.layout.items:
            return False
        self.status.set("Select an item first: click it on the page.")
        return True

    # ── mouse ────────────────────────────────────────────────────────────
    def _fit_badge_at(self, cx, cy):
        for tid in self.canvas.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1):
            for tag in self.canvas.gettags(tid):
                if tag.startswith('fit:'):
                    return tag[4:]
        return None

    def _on_press(self, event):
        self.canvas.focus_set()
        cx, cy = self._cxy(event)
        fit = self._fit_badge_at(cx, cy)
        if fit and fit in self.layout.items:
            self._zoom_to_fit(fit)
            self._drag = None
            return
        if self._sel in self.layout.items:
            _, _, bx1, by1 = self._box_px(
                self.layout.items[self._sel]['box'])
            if (bx1 - _HANDLE - 3 <= cx <= bx1 + 3
                    and by1 - _HANDLE - 3 <= cy <= by1 + 3):
                self._drag = ('resize', self._sel, (cx, cy),
                              list(self.layout.items[self._sel]['box']))
                return
        fx, fy = self._to_page(cx, cy)
        iid = self.layout.hit(fx, fy)
        self._sel = iid
        self._drag = (('move', iid, (cx, cy),
                       list(self.layout.items[iid]['box']))
                      if iid else None)
        self._draw_selection()

    def _drag_box(self, drag, event):
        """The box a drag has reached, soft-snapped, and its guides."""
        mode, iid, (sx, sy), box = drag
        cx, cy = self._cxy(event)
        ppi = self._geom[0]
        pw, ph = self.layout.page_size
        dx, dy = (cx - sx) / (pw * ppi), (cy - sy) / (ph * ppi)
        x, y, w, h = box
        if mode == 'move':
            nb = clamp_box(x + dx, y + dy, w, h)
        else:
            nb = clamp_box(x, y, w + dx, h + dy)
        guides = (None, None)
        if self.snap_var.get():
            tol = _SNAP_PX / (min(pw, ph) * ppi)
            nb, guides = soft_snap(nb, mode, self._snap_targets(iid), tol)
        return mode, iid, box, list(nb), guides, (dx, dy)

    def _on_drag(self, event):
        if not self._drag:
            return
        mode, iid, _box, nb, guides, _d = self._drag_box(self._drag, event)
        self.layout.items[iid]['box'] = nb
        if mode == 'move':
            bx0, by0, _bx1, _by1 = self._box_px(nb)
            for tag in self.canvas.find_withtag(f'img:{iid}'):
                if self.canvas.type(tag) == 'image':
                    self.canvas.coords(tag, bx0, by0)
            self.canvas.delete('fit')
        self._draw_selection(guides)

    def _on_release(self, event):
        if not self._drag:
            return
        mode, iid, box, nb, _g, (dx, dy) = self._drag_box(self._drag, event)
        self._drag = None
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            self.layout.items[iid]['box'] = box
            return
        if mode == 'move' and self._over_host(event):
            self.layout.items[iid]['box'] = box     # the item stays put
            self.doc.shelve(self.layout.items[iid], origin=self.layout.name)
            self._refresh_host()
            self.status.set("Kept a copy in the cross-page host: drag it onto "
                            "any page.")
            self._redraw()
            return
        if mode == 'move' and self.layout.items[iid]['kind'] == 'plot':
            fx, fy = self._to_page(*self._cxy(event))
            target = self._hit_other(fx, fy, iid)
            if target is not None and self.layout.items[target]['kind'] in (
                    'stats', 'chart'):
                self.layout.items[iid]['box'] = box     # the plot stays put
                self._feed_from_plot(iid, target)
                self._sel = target
                self._redraw()
                return
        self.layout.items[iid]['box'] = nb
        self._redraw()

    def _hit_other(self, fx, fy, skip):
        for other in reversed(self.layout.order):
            if other == skip:
                continue
            x, y, w, h = self.layout.items[other]['box']
            if x <= fx <= x + w and y <= fy <= y + h:
                return other
        return None

    def _feed_from_plot(self, plot_iid, target):
        """A plot dropped on a stats table or chart supplies its samples
        (and its populations, when the table has none yet)."""
        from .layout_page import item_samples
        plot = self.layout.items[plot_iid]
        it = self.layout.items[target]
        page = self._sample()
        add = [s for s in item_samples(plot, page) if s]
        if it.get('samples'):
            merged = list(it['samples']) + [s for s in add
                                            if s not in it['samples']]
            it['samples'] = merged
        else:
            # A table following the page takes the plot's samples; a plot
            # that follows the page too leaves it following.
            merged = add
            if plot.get('samples'):
                it['samples'] = list(add)
        if not it.get('populations'):
            it['populations'] = item_populations(plot)
        self.status.set(f"The {it['kind']} now covers "
                        f"{len([s for s in merged if s])} sample(s).")

    # ── zoom inside a plot ───────────────────────────────────────────────
    def _on_wheel(self, event, delta=None):
        """The wheel scrolls the page; Ctrl+wheel zooms the page, Shift+
        wheel scrolls sideways. Over the SELECTED plot it zooms that plot
        (click a plot first), so the page still scrolls over the others."""
        d = delta if delta is not None else getattr(event, 'delta', 0)
        if not d:
            return None
        state = int(getattr(event, 'state', 0) or 0)
        if state & 0x0004:                                    # Ctrl
            self._zoom_page(1.25 if d > 0 else 0.8)
            return 'break'
        if state & 0x0001:                                    # Shift
            self.canvas.xview_scroll(-1 if d > 0 else 1, 'units')
            return 'break'
        cx, cy = self._cxy(event)
        iid = self.layout.hit(*self._to_page(cx, cy))
        if (iid is not None and iid == self._sel
                and self.layout.items[iid]['kind'] == 'plot'):
            self._zoom_item(iid, 0.8 if d > 0 else 1.25, (cx, cy))
            return 'break'
        self.canvas.yview_scroll(-1 if d > 0 else 1, 'units')
        return 'break'

    def _zoom_page(self, factor):
        """Ctrl+wheel: the page's on-screen resolution, as the View
        dropdown sets it."""
        from .layout_page import SCREEN_DPI
        pct = int(round(self._geom[0] / SCREEN_DPI * 100 * factor))
        pct = min(max(pct, 10), 800)
        self._zoom = str(pct)
        self.zoom_var.set(f'{pct}%')
        self._redraw()

    def _view_shift(self, iid, frm, to, ax=None):
        """Limits of plot ``iid`` moved so the data under canvas point
        ``frm`` sits under ``to`` (each axis in its own scale), or None."""
        import numpy as np
        it = self.layout.items.get(iid)
        ax = ax if ax is not None else self._axes.get(iid)
        if it is None or ax is None:
            return None
        bx0, by0, _bx1, _by1 = self._box_px(it['box'])
        fig_h = ax.figure.bbox.height
        try:
            inv = ax.transData.inverted()
            a = inv.transform((frm[0] - bx0, fig_h - (frm[1] - by0)))
            b = inv.transform((to[0] - bx0, fig_h - (to[1] - by0)))
        except Exception:  # noqa: BLE001 - degenerate axes transform; ignore this drag
            return None
        hist = it.get('mode') == 'histogram' or not it.get('y')
        out = {}
        for name, lim, axis, i in (('xlim', ax.get_xlim(), ax.xaxis, 0),
                                   ('ylim', ax.get_ylim(), ax.yaxis, 1)):
            if name == 'ylim' and hist:
                continue
            tr = axis.get_transform()
            lo, hi = tr.transform(np.asarray(lim, dtype=float))
            da = float(tr.transform(np.asarray([a[i]], dtype=float))[0])
            db = float(tr.transform(np.asarray([b[i]], dtype=float))[0])
            d = da - db
            if not np.isfinite(d):
                return None
            back = tr.inverted().transform(np.asarray([lo + d, hi + d]))
            if not np.all(np.isfinite(back)):
                return None
            out[name] = sorted(float(v) for v in back)
        return out

    def _center_item(self, iid, at):
        """Right-click "Center zoom here": the clicked point becomes the
        middle of the plot's view, at the same zoom."""
        ax = self._axes.get(iid)
        it = self.layout.items.get(iid)
        if ax is None or it is None:
            return
        bx0, by0, _bx1, _by1 = self._box_px(it['box'])
        bb = ax.bbox
        mid = (bx0 + (bb.x0 + bb.x1) / 2.0,
               by0 + ax.figure.bbox.height - (bb.y0 + bb.y1) / 2.0)
        out = self._view_shift(iid, at, mid)
        if out:
            it.update(out)
            self._redraw()

    def _on_pan_start(self, event):
        """Middle button held on a plot: drag the data inside it. The plot
        stays where it is on the page; its view follows the pointer."""
        cx, cy = self._cxy(event)
        iid = self.layout.hit(*self._to_page(cx, cy))
        if (iid is None or self.layout.items[iid]['kind'] != 'plot'
                or iid not in self._axes):
            self._pan = None
            return
        # The axes as drawn at the press: every move is measured from them,
        # so redraws during the drag don't compound.
        self._pan = (iid, (cx, cy), self._axes[iid])
        self._pan_after = None
        self.canvas.configure(cursor='fleur')

    def _pan_to(self, event):
        pan = getattr(self, '_pan', None)
        if not pan:
            return None
        iid, start, ax0 = pan
        out = self._view_shift(iid, start, self._cxy(event), ax=ax0)
        if out:
            self.layout.items[iid].update(out)
        return out

    def _on_pan_move(self, event):
        if not getattr(self, '_pan', None):
            return
        self._pan_to(event)
        # Redraw the plot a few times a second while dragging, not on
        # every motion event.
        if getattr(self, '_pan_after', None) is None:
            def _tick():
                self._pan_after = None
                self._redraw()
            self._pan_after = self.after(80, _tick)

    def _on_pan_end(self, event):
        if not getattr(self, '_pan', None):
            return
        out = self._pan_to(event)
        self._pan = None
        pending = getattr(self, '_pan_after', None)
        if pending is not None:
            try:
                self.after_cancel(pending)
            except tk.TclError:
                pass
            self._pan_after = None
        self.canvas.configure(cursor='')
        if out:
            self.status.set("Moved the view. Zoom to fit (right-click) shows "
                            "all the data again.")
        self._redraw()

    def _zoom_item(self, iid, factor, at=None):
        """Zoom plot ``iid`` by ``factor`` (<1 in) about the canvas point
        ``at`` (its axes' centre when None), in each axis's own scale, and
        keep the result as the plot's zoom."""
        import numpy as np
        it = self.layout.items.get(iid)
        ax = self._axes.get(iid)
        if it is None or ax is None:
            return
        bx0, by0, _bx1, _by1 = self._box_px(it['box'])
        fig_h = ax.figure.bbox.height
        if at is not None:
            px, py = at[0] - bx0, fig_h - (at[1] - by0)
        else:
            bb = ax.bbox
            px, py = (bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0
        try:
            cxd, cyd = ax.transData.inverted().transform((px, py))
        except Exception:  # noqa: BLE001 - degenerate axes transform; ignore this move
            return
        hist = it.get('mode') == 'histogram' or not it.get('y')
        out = {}
        for name, lim, axis, c in (('xlim', ax.get_xlim(), ax.xaxis, cxd),
                                   ('ylim', ax.get_ylim(), ax.yaxis, cyd)):
            if name == 'ylim' and hist:
                continue
            tr = axis.get_transform()
            lo, hi = tr.transform(np.asarray(lim, dtype=float))
            ct = float(tr.transform(np.asarray([c], dtype=float))[0])
            if not np.isfinite(ct):
                ct = (lo + hi) / 2.0
            nlo = ct + (lo - ct) * factor
            nhi = ct + (hi - ct) * factor
            back = tr.inverted().transform(np.asarray([nlo, nhi]))
            if not np.all(np.isfinite(back)):
                return
            out[name] = sorted(float(v) for v in back)
        it.update(out)
        self.status.set("Zoomed. Zoom to fit (right-click) shows all the "
                        "data again.")
        self._redraw()

    def _on_double(self, event):
        fx, fy = self._to_page(*self._cxy(event))
        iid = self.layout.hit(fx, fy)
        if iid:
            self._sel = iid
            self._edit_selected()

    def _on_context(self, event):
        fx, fy = self._to_page(*self._cxy(event))
        iid = self.layout.hit(fx, fy)
        if not iid:
            return
        self._sel = iid
        self._draw_selection()
        it = self.layout.items[iid]
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Edit…", command=self._edit_selected)
        m.add_command(label="Duplicate", command=self._duplicate_selected)
        m.add_command(label="Bring to front", command=self._raise_selected)
        if it['kind'] == 'plot':
            cx, cy = self._cxy(event)
            m.add_command(label="Center zoom here",
                          command=lambda: self._center_item(iid, (cx, cy)))
            m.add_command(label="Zoom in",
                          command=lambda: self._zoom_item(iid, 0.8))
            m.add_command(label="Zoom out",
                          command=lambda: self._zoom_item(iid, 1.25))
            m.add_command(label="Zoom to fit",
                          command=lambda: self._zoom_to_fit(iid),
                          state='normal' if (it.get('xlim') or it.get('ylim'))
                          else 'disabled')
        others = [(i, p.name) for i, p in enumerate(self.doc.pages)
                  if i != self.page_idx]
        if others:
            cp, mv = tk.Menu(m, tearoff=0), tk.Menu(m, tearoff=0)
            for i, name in others:
                cp.add_command(label=name,
                               command=lambda i=i: self._copy_to_page(iid, i))
                mv.add_command(label=name, command=lambda i=i: (
                    self._copy_to_page(iid, i, move=True)))
            m.add_cascade(label="Copy to page", menu=cp)
            m.add_cascade(label="Move to page", menu=mv)
        m.add_separator()
        m.add_command(label="Delete", command=self._delete_selected)
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    # ── the cross-page host ──────────────────────────────────────────────
    def _build_host(self, parent):
        """The left margin: items kept to use on any page."""
        host = ttk.Frame(parent, padding=(6, 0, 4, 0))
        parent.add(host, weight=0)          # drag its right edge to widen
        ttk.Label(host, text="Cross-page host",
                  font='OpenFloHeading').pack(anchor='w')
        ttk.Label(host, text="Drag an item here to keep it;\ndrag it from "
                  "here onto any page.", style='Muted.TLabel',
                  justify='left').pack(anchor='w', pady=(0, 4))
        self.host_list = tk.Listbox(host, width=36, height=12,
                                    exportselection=False,
                                    activestyle='none')
        self.host_list.pack(fill='both', expand=True)
        auto_tip(self.host_list,
                 "Items kept beside the pages. Drag one onto the page to place "
                 "a copy; right-click to place it on every page or remove it. "
                 "Pointing at one marks where it is on the pages. Drag the "
                 "panel's edge to widen it.")
        hl = self.host_list
        hl.bind('<ButtonPress-1>', self._host_press)
        hl.bind('<B1-Motion>', self._host_motion)
        hl.bind('<ButtonRelease-1>', self._host_release)
        bind_right_click(hl, self._host_menu)
        hl.bind('<Motion>', self._host_hover)
        hl.bind('<Leave>', lambda _e: self._host_unmark())
        self._host_drag = None
        self._host_marked = None
        self._refresh_host()

    @staticmethod
    def _pop_name(key):
        """A readable population name from its key ('n:cells/n:cd3+' ->
        'cd3+'; unnamed 'threshold:fl1-a' -> 'threshold fl1-a')."""
        if not key:
            return 'All events'
        last = str(key).split('/')[-1]
        return last[2:] if last.startswith('n:') else last.replace(':', ' ')

    def _host_hover(self, event):
        i = self.host_list.nearest(event.y)
        bb = self.host_list.bbox(i) if 0 <= i < len(self.doc.shelf) else None
        if not bb or not (bb[1] <= event.y <= bb[1] + bb[3]):
            self._host_unmark()
            return
        if i != self._host_marked:
            self._host_mark(i)

    def _host_mark(self, i):
        """Point out where host item ``i`` is: outline its matches on this
        page; mark the tabs of other pages that hold one."""
        self._host_unmark()
        self._host_marked = i
        here, other_pages = [], set()
        for p, iid in self.doc.matches(i):
            if p == self.page_idx:
                here.append(iid)
            else:
                other_pages.add(p)
        c = self.canvas
        for iid in here:
            x0, y0, x1, y1 = self._box_px(self.layout.items[iid]['box'])
            c.create_rectangle(x0 - 3, y0 - 3, x1 + 3, y1 + 3,
                               outline='#f2a541', width=3, dash=(6, 3),
                               tags=('hostmark',))
        for p in other_pages:
            try:
                self.tabs.tab(p, text=f'◆ {self.doc.pages[p].name}')
            except tk.TclError:
                pass
        n_here, n_else = len(here), len(other_pages)
        if n_here or n_else:
            self.status.set(
                ', '.join(s for s in (
                    f"{n_here} on this page" if n_here else '',
                    f"on {n_else} other page(s), marked ◆" if n_else else '')
                    if s))

    def _host_unmark(self):
        if getattr(self, '_host_marked', None) is None:
            return
        self._host_marked = None
        self.canvas.delete('hostmark')
        for p, page in enumerate(self.doc.pages):
            try:
                self.tabs.tab(p, text=page.name)
            except tk.TclError:
                pass

    def _host_label(self, it, n=None):
        """What the host lists: its number, the kind, what it shows (name
        or populations, axes / statistic) and the page it came from."""
        from .layout_page import item_populations
        kind = it.get('kind', '?')
        num = f'{n}. ' if n is not None else ''
        if it.get('uid'):
            num += f"{it['uid']} · "
        if it.get('host_name'):
            return f"{num}{it['host_name']}"
        parts = [kind.capitalize()]
        if it.get('title'):
            parts.append(str(it['title']))
        elif kind == 'text':
            parts.append(repr(str(it.get('text', ''))[:20]))
        else:
            pops = (item_populations(it) if kind == 'plot'
                    else list(it.get('populations') or []))
            names = [self._pop_name(k) for k in pops[:2]]
            if len(pops) > 2:
                names.append(f'+{len(pops) - 2}')
            parts.append(', '.join(names) if names else 'no population')
        if kind == 'plot':
            ax = self.ed._fmt_channel(it['x']) if it.get('x') else '?'
            if it.get('y') and it.get('mode') != 'histogram':
                ax += f" / {self.ed._fmt_channel(it['y'])}"
            parts.append(f"{ax} ({it.get('mode', 'dot')})")
        elif kind == 'chart':
            parts.append(it.get('chart', 'violin')
                         + (f" {it['channel']}" if it.get('channel') else ''))
        elif kind == 'stats':
            parts.append(', '.join(it.get('stats') or []))
        if it.get('samples'):
            parts.append(f"{len(it['samples'])} samples")
        if it.get('host_from'):
            parts.append(f"from {it['host_from']}")
        return num + ' · '.join(parts)

    def _refresh_host(self):
        hl = getattr(self, 'host_list', None)
        if hl is None:
            return
        hl.delete(0, 'end')
        for n, it in enumerate(self.doc.shelf, 1):
            hl.insert('end', self._host_label(it, n))

    def _over_host(self, event):
        hl = getattr(self, 'host_list', None)
        if hl is None:
            return False
        try:
            xr = event.x_root if hasattr(event, 'x_root') else None
            yr = event.y_root if hasattr(event, 'y_root') else None
            if xr is None:
                xr = self.canvas.winfo_rootx() + event.x
                yr = self.canvas.winfo_rooty() + event.y
            x, y = hl.winfo_rootx(), hl.winfo_rooty()
            return (x <= xr < x + hl.winfo_width()
                    and y <= yr < y + hl.winfo_height())
        except tk.TclError:
            return False

    def _host_press(self, event):
        i = self.host_list.nearest(event.y)
        self._host_drag = i if 0 <= i < len(self.doc.shelf) else None

    def _host_motion(self, event):
        if self._host_drag is not None:
            self.host_list.configure(cursor='fleur')

    def _host_release(self, event):
        idx, self._host_drag = self._host_drag, None
        self.host_list.configure(cursor='')
        if idx is None or not self.point_in_canvas(event.x_root,
                                                   event.y_root):
            return
        cx = self.canvas.canvasx(event.x_root - self.canvas.winfo_rootx())
        cy = self.canvas.canvasy(event.y_root - self.canvas.winfo_rooty())
        fx, fy = self._to_page(cx, cy)
        _x, _y, w, h = self.doc.shelf[idx]['box']
        self._sel = self.doc.place_from_shelf(
            idx, self.page_idx, (fx - w / 2, fy - h / 2, w, h))
        self.status.set("Placed a copy from the cross-page host.")
        self._redraw()

    def _host_menu(self, event):
        i = self.host_list.nearest(event.y)
        if not 0 <= i < len(self.doc.shelf):
            return
        self.host_list.selection_clear(0, 'end')
        self.host_list.selection_set(i)
        m = tk.Menu(self, tearoff=0)
        m.add_command(label="Place on this page",
                      command=lambda: self._place_host(i, [self.page_idx]))
        m.add_command(label="Place on every page",
                      command=lambda: self._place_host(
                          i, range(len(self.doc.pages))))
        m.add_command(label="Rename…", command=lambda: self._rename_host(i))
        m.add_separator()
        m.add_command(label="Remove from the host",
                      command=lambda: (self.doc.shelf.pop(i),
                                       self._refresh_host()))
        try:
            m.tk_popup(event.x_root, event.y_root)
        finally:
            m.grab_release()

    def _rename_host(self, i):
        it = self.doc.shelf[i]
        name = simpledialog.askstring(
            "Rename", "Name in the cross-page host:", parent=self,
            initialvalue=it.get('host_name') or self._host_label(it))
        if name is None:
            return
        if name.strip():
            it['host_name'] = name.strip()
        else:
            it.pop('host_name', None)            # blank: back to automatic
        self._refresh_host()

    def _place_host(self, idx, pages):
        pages = list(pages)
        for p in pages:
            iid = self.doc.place_from_shelf(idx, p)
            if p == self.page_idx:
                self._sel = iid
        self.status.set(f"Placed on {len(pages)} page(s).")
        self._redraw()

    # ── drops from the editor ────────────────────────────────────────────
    def point_in_canvas(self, x_root, y_root):
        """Is the screen point over this window's page area? Geometry, not
        winfo_containing, which returns None during a drag's pointer grab."""
        c = self.canvas
        try:
            if not (self.winfo_exists() and self.winfo_ismapped()):
                return False
            x, y = c.winfo_rootx(), c.winfo_rooty()
            return (x <= x_root < x + c.winfo_width()
                    and y <= y_root < y + c.winfo_height())
        except tk.TclError:
            return False

    def drop_populations(self, targets, x_root, y_root):
        """``[(sample, gid|None), ...]`` dropped at a screen point: onto a
        plot, table or chart they join it; onto empty page they become a new
        plot, shown as the first population was last displayed (see the
        ``layout_drop_view`` preference)."""
        from .layout_render import SampleContext
        targets = [t for t in targets if t[0] in self.ed._samples]
        if not targets:
            return False
        keys, samples = [], []
        for sample, gid in targets:
            ctx = SampleContext(self.ed, sample)
            key = ALL_EVENTS if gid is None else ctx.keys.get(gid)
            if gid is not None and key is None:
                continue
            if key not in keys:
                keys.append(key)
            if sample not in samples:
                samples.append(sample)
        if not keys:
            return False
        # One sample: follow the page's sample, so the plot re-applies per
        # sample. Several: pin them, overlaid.
        pinned = samples if len(samples) > 1 else []
        self._fit()                  # the page's place on screen now
        cx = x_root - self.canvas.winfo_rootx()
        cy = y_root - self.canvas.winfo_rooty()
        fx, fy = self._to_page(self.canvas.canvasx(cx), self.canvas.canvasy(cy))
        hit = self.layout.hit(fx, fy)
        if hit and self.layout.items[hit]['kind'] in ('plot', 'stats', 'chart'):
            it = self.layout.items[hit]
            pops = item_populations(it) if it['kind'] == 'plot' \
                else list(it.get('populations') or ())
            it['populations'] = pops + [k for k in keys if k not in pops]
            it.pop('population', None)
            # Another sample joins as an overlay (the item then pins its
            # samples); the page's own sample changes nothing.
            have = list(it.get('samples') or ([self._sample()]
                                              if self._sample() else []))
            merged = have + [s for s in samples if s not in have]
            if len(merged) > 1:
                it['samples'] = merged
            self._sel = hit
            self.status.set(f"Added {len(targets)} to the "
                            f"{it['kind']}.")
        else:
            sample, gid = targets[0]
            view = self.ed._view_for_population(sample, gid)
            w, h = 0.3, 0.42
            box = clamp_box(fx - w / 2, fy - h / 2, w, h)
            self._sel = self.layout.add(plot_item(
                populations=keys, samples=pinned, box=box,
                x=view.get('x'), y=view.get('y'),
                mode=view.get('mode', 'dot'),
                color=view.get('color') or 'By density',
                xscale=view.get('xscale'), yscale=view.get('yscale'),
                xlim=view.get('xlim'), ylim=view.get('ylim')))
            self.status.set(f"Added a plot of {len(targets)} population(s).")
        if not pinned and samples and self.sample_var.get() != samples[0]:
            self.sample_var.set(samples[0])
        self._redraw()
        try:
            self.lift()
        except tk.TclError:
            pass
        return True

    # ── item actions ─────────────────────────────────────────────────────
    def _choices(self):
        """``[(key, path)]`` populations of the previewed sample."""
        sample = self._sample()
        if sample is None:
            return [(ALL_EVENTS, 'All events')]
        from .layout_render import SampleContext
        ctx = SampleContext(self.ed, sample)
        return population_choices(ctx.gates, ctx.order, ctx.labels,
                                  ctx.transforms)

    def _selected_gate_key(self):
        """``(key of the main window's selected gate, its gate)`` or
        ``(None, None)``."""
        try:
            gid = self.ed._selected_gate_id()
        except Exception:  # noqa: BLE001 - no gate selected; nothing preset
            gid = None
        sample = getattr(self.ed, '_active_sample', None)
        if not gid or sample is None:
            return None, None
        from .layout_render import SampleContext
        ctx = SampleContext(self.ed, sample)
        return ctx.keys.get(gid), ctx.gates.get(gid)

    def _add_plot(self):
        ed = self.ed
        try:
            view = ed._current_plot_view()
        except Exception:  # noqa: BLE001 - no current view; dialog defaults
            view = {}
        x, y = view.get('x'), view.get('y')
        mode = view.get('mode', 'dot')
        population = ALL_EVENTS
        _, gate = self._selected_gate_key()
        keep_view = True
        if gate is not None:
            # FlowJo's drag of a gate: the gate's own plot, i.e. its parent
            # population on the gate's axes, with the gate drawn.
            from .layout_render import SampleContext
            ctx = SampleContext(ed, ed._active_sample)
            pid = gate.get('parent_id')
            population = ctx.keys.get(pid, ALL_EVENTS) if pid else ALL_EVENTS
            if gate.get('channel'):
                keep_view = x == gate['channel']
                x = gate['channel']
                if mode != 'histogram' and y == x:
                    y = None
            elif gate.get('x_channel'):
                keep_view = (x, y) == (gate.get('x_channel'),
                                       gate.get('y_channel'))
                x, y = gate.get('x_channel'), gate.get('y_channel')
                if mode == 'histogram':
                    mode = 'dot'
        item = plot_item(
            population=population, x=x, y=y, mode=mode,
            color=view.get('color') or 'By density',
            xscale=view.get('xscale') if keep_view else None,
            yscale=view.get('yscale') if keep_view else None,
            xlim=view.get('xlim') if keep_view else None,
            ylim=view.get('ylim') if keep_view else None,
            box=self.layout.next_slot(0.3, 0.42))
        ItemDialog(self, item, self._choices(), on_ok=self._insert)

    def _add_stats(self):
        key, _gate = self._selected_gate_key()
        item = stats_item(populations=[key] if key else [],
                          box=self.layout.next_slot(0.3, 0.25))
        ItemDialog(self, item, self._choices(), on_ok=self._insert)

    def _add_chart(self):
        key, _gate = self._selected_gate_key()
        try:
            ch = resolve_channel(self.ed.x_combo.get())
        except Exception:  # noqa: BLE001 - no channel; user picks in dialog
            ch = None
        item = chart_item(populations=[key] if key else [ALL_EVENTS],
                          samples=self._batch_samples(), channel=ch,
                          box=self.layout.next_slot(0.3, 0.42))
        ItemDialog(self, item, self._choices(), on_ok=self._insert)

    def _add_text(self):
        self._insert(text_item(box=self.layout.next_slot(0.4, 0.06)))

    def _insert(self, item):
        self._sel = self.layout.add(item)
        self._redraw()

    def _edit_selected(self):
        iid, page = self._sel, self.layout
        if self._nothing_selected() or iid is None:
            return

        def ok(new):
            new['box'] = page.items[iid]['box']
            page.items[iid] = new
            self._redraw()
        ItemDialog(self, dict(page.items[iid]), self._choices(), on_ok=ok)

    def _zoom_to_fit(self, iid):
        it = self.layout.items.get(iid)
        if it is None:
            return
        it['xlim'] = it['ylim'] = None
        self.status.set("Zoomed to fit the data.")
        self._redraw()

    def _copy_to_page(self, iid, dst, move=False):
        self.doc.copy_item(self.page_idx, iid, dst)
        if move:
            self.layout.remove(iid)
            self._sel = None
        self.status.set(f"{'Moved' if move else 'Copied'} to "
                        f"'{self.doc.pages[dst].name}'.")
        self._redraw()

    def _duplicate_selected(self):
        if not self._nothing_selected():
            self._sel = self.layout.duplicate(self._sel)
            self._redraw()

    def _raise_selected(self):
        if self._sel in self.layout.items:
            self.layout.raise_to_top(self._sel)
            self._redraw()

    def _on_delete_key(self, event):
        """Delete (or, on a Mac, its delete key's BackSpace) removes the
        selected item, unless it was pressed in a box that takes text, where
        it edits the text."""
        try:
            if event.widget.winfo_class() in TEXT_INPUT_CLASSES:
                return None
        except (AttributeError, tk.TclError):
            pass
        self._delete_selected()
        return 'break'

    def _delete_selected(self):
        if not self._nothing_selected():
            self.layout.remove(self._sel)
            self._sel = None
            self._redraw()

    def _on_page_size(self):
        self.layout.page = self.page_var.get()
        self._images.clear()
        self._redraw()

    def _on_zoom(self):
        v = self.zoom_var.get()
        if v == 'Custom…':
            pct = simpledialog.askinteger(
                "View resolution", "Zoom (% of 96 pixels per inch):",
                minvalue=10, maxvalue=800, parent=self,
                initialvalue=100 if self._zoom == 'Fit' else int(self._zoom))
            if not pct:
                self.zoom_var.set('Fit' if self._zoom == 'Fit'
                                  else f'{self._zoom}%')
                return
            self._zoom = str(pct)
            self.zoom_var.set(f'{pct}%')
        else:
            self._zoom = 'Fit' if v == 'Fit' else v.rstrip('%')
        self._redraw()

    def _on_dpi(self):
        try:
            self.doc.export_dpi = int(self.dpi_var.get())
        except (TypeError, ValueError):
            self.dpi_var.set(str(self.doc.export_dpi))
        # Its own undo step, marked unsaved now: no redraw follows to record
        # it, and the next change must not carry it along when undone.
        self._record_change()

    def _step_sample(self, d):
        names = list(self.sample_combo['values'])
        if not names:
            return
        cur = self.sample_var.get()
        i = names.index(cur) if cur in names else -1
        self.sample_var.set(names[(i + d) % len(names)])
        self._redraw()

    def _refresh(self):
        """Re-read samples and gates (after an edit in the main window)."""
        names = list(getattr(self.ed, '_sample_order', []) or [])
        self.sample_combo['values'] = names
        if self.sample_var.get() not in names:
            self.sample_var.set(names[0] if names else '')
        self._images.clear()
        self._redraw()

    def _edit_default_colors(self):
        ColorsDialog(self, on_ok=lambda: self._redraw())

    # ── export ───────────────────────────────────────────────────────────
    def _batch_samples(self):
        try:
            names = list(self.ed._selected_samples())
        except Exception:                             # noqa: BLE001 - the selection read is recorded by _note_failure; the batch falls back to every loaded sample
            _note_failure("reading selected samples for the layout batch")
            names = []
        return names or [n for n in getattr(self.ed, '_sample_order', [])
                         if n in self.ed._samples]

    def _save_fig(self, fig, path, fmt=None):
        """Save one page: ``path`` is a file or, for the batch PDF, its open
        ``PdfPages`` (``fmt='pdf'``)."""
        from .theme import savefig_background
        savefig_background(fig, path, background=self.bg_var.get(),
                           dpi=self.doc.export_dpi, tight=False, fmt=fmt,
                           corrections=self._corrections())

    def _corrections(self):
        """The fixes the editor's analysis has applied, for the stamp in
        each exported file's metadata (export_stamp)."""
        from .export_stamp import corrections_for
        return corrections_for(self.ed)

    def _export_page(self):
        from .layout_render import render_page
        sample = self._sample()
        if sample is None or not self.layout.items:
            messagebox.showinfo("Layout editor",
                                "Add items and load a sample first.",
                                parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Export layout page", defaultextension='.pdf',
            initialfile=(f'{layout_file_part(sample)}_'
                         f'{layout_file_part(self.layout.name)}.pdf'),
            filetypes=[('PDF document', '*.pdf'), ('PNG image', '*.png'),
                       ('SVG vector', '*.svg'), ('TIFF image', '*.tif *.tiff'),
                       ('PowerPoint, one slide', '*.pptx')])
        if not path:
            return
        try:
            fig = render_page(self.ed, self.layout, sample)
            if path.lower().endswith('.pptx'):
                self._write_pptx(path, [(fig, sample, self.layout)])
            else:
                self._save_fig(fig, path)
        except Exception as exc:                      # noqa: BLE001 - any render or save failure is shown in an error dialog
            show_error(self, "Layout editor",
                       "The page could not be exported.", exc, path)
            return
        record_export(self.ed, 'layout.export', path,
                      n_items=len(self.layout.items), n_samples=1,
                      dpi=self.doc.export_dpi, samples=[sample])
        self.status.set(f"Saved {os.path.basename(path)} at "
                        f"{self.doc.export_dpi} dpi")

    def _export_batch(self):
        from .layout_render import render_page
        samples = self._batch_samples()
        pages = [p for p in self.doc.pages if p.items]
        if not samples or not pages:
            messagebox.showinfo("Layout editor",
                                "Add items and load samples first.",
                                parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title=f"Batch export ({len(samples)} samples)",
            defaultextension='.pdf', initialfile='layout.pdf',
            filetypes=[('PDF, every page per sample', '*.pdf'),
                       ('PNG, an image per sample and page', '*.png'),
                       ('SVG, a file per sample and page', '*.svg'),
                       ('PowerPoint, a slide per sample and page',
                        '*.pptx')])
        if not path:
            return
        stem, ext = os.path.splitext(path)
        dpi = self.doc.export_dpi
        written = []
        n = len(samples)

        # One sample per turn of the event loop, on the window thread (pages
        # draw through the editor): the window answers clicks in between, the
        # strip shows how far it got, and Cancel stops after that sample.
        slides = []                    # PowerPoint: (PNG, title, page size)
        outs: list = []                # PNG / SVG: a file per sample and page

        def steps():
            if ext.lower() == '.pptx':
                for i, nm in enumerate(samples):
                    yield (i, n, f"Exporting {nm} ({i + 1} of {n})…")
                    for page in pages:
                        slides.append(self._slide(
                            render_page(self.ed, page, nm), nm, page))
                    written.append(nm)
                yield (n, n, "Writing the PowerPoint file…")
                self._write_slides(path, slides)
            elif ext.lower() == '.pdf':
                import matplotlib
                from matplotlib.backends.backend_pdf import PdfPages

                from .export_stamp import savefig_metadata
                from .theme import figure_rc
                # Editable text, as every other export (figure_rc). The PDF
                # reads it while drawing a page (savefig_background runs each
                # page under it) and again when it writes the fonts at close,
                # which runs under it here; it is not left set while the
                # window waits between samples.
                rc = figure_rc()
                pdf = PdfPages(path, metadata=savefig_metadata(
                    fmt='pdf', corrections=self._corrections()))
                try:
                    for i, nm in enumerate(samples):
                        yield (i, n, f"Exporting {nm} ({i + 1} of {n})…")
                        for page in pages:
                            # One page of the PDF, saved as every other
                            # export is: the chosen background, per format.
                            # Figure.savefig appends a page when handed the
                            # open PdfPages (the file's metadata was set
                            # when it was opened).
                            self._save_fig(render_page(self.ed, page, nm),
                                           pdf, fmt='pdf')
                        written.append(nm)
                finally:
                    with matplotlib.rc_context(rc):
                        pdf.close()
            else:
                for i, nm in enumerate(samples):
                    yield (i, n, f"Exporting {nm} ({i + 1} of {n})…")
                    for page in pages:
                        suffix = (f'_{layout_file_part(page.name)}'
                                  if len(pages) > 1 else '')
                        out = f'{stem}_{layout_file_part(nm)}{suffix}{ext}'
                        self._save_fig(render_page(self.ed, page, nm), out)
                        outs.append(out)
                    written.append(nm)

        def done(_result):
            record_export(self.ed, 'layout.export', path,
                          n_items=sum(len(p.items) for p in pages),
                          n_samples=n, dpi=dpi, samples=list(samples),
                          **({'files': outs} if outs else {}))
            files = (1 if ext.lower() in ('.pdf', '.pptx')
                     else n * len(pages))
            self.status.set(f"Exported {n} sample(s) to {files} file(s) at "
                            f"{dpi} dpi.")

        def failed(exc):
            show_error(self, "Layout editor", "The batch export stopped.",
                       exc)

        def cancelled():
            self.status.set(f"Batch export cancelled after {len(written)} of "
                            f"{n} sample(s); "
                            + ("the PDF holds those." if ext.lower() == '.pdf'
                               else "no PowerPoint file was written."
                               if ext.lower() == '.pptx'
                               else "their files are written."))

        self._busy.run_steps(steps(), on_done=done, on_error=failed,
                             on_cancel=cancelled, text="Batch export…")

    def _slide(self, fig, sample, page):
        """``(PNG, title, page size)`` of one rendered page, at the export
        resolution and background, for a PowerPoint slide."""
        import io

        from .pptx_export import slide_title
        buf = io.BytesIO()
        from .theme import savefig_background
        savefig_background(fig, buf, background=self.bg_var.get(),
                           dpi=self.doc.export_dpi, tight=False, fmt='png',
                           corrections=self._corrections())
        return (buf.getvalue(), slide_title(sample, page.name),
                page.page_size)

    def _write_slides(self, path, slides):
        """One deck of ``slides``; the slide size is the first page's (a
        page of another size is drawn to fill it)."""
        from .pptx_export import write_pptx
        write_pptx(path, [(png, title) for png, title, _sz in slides],
                   slides[0][2], corrections=self._corrections())

    def _write_pptx(self, path, figs):
        """``figs`` (``(figure, sample, page)`` each) as a PowerPoint deck."""
        self._write_slides(path, [self._slide(f, nm, pg)
                                  for f, nm, pg in figs])

    def _add_strategy_page(self):
        """A new page laying out the gating strategy (layout_page): every
        gating step of the previewed sample, or, with a gate selected in
        the main window, the steps down to it. Batch export then draws it
        for every sample."""
        from .layout_page import gating_strategy_page, gating_strategy_steps
        from .layout_render import SampleContext
        sample = self._sample()
        if sample is None:
            messagebox.showinfo("Layout editor", "Load a sample first.",
                                parent=self)
            return
        ctx = SampleContext(self.ed, sample)
        gid = None
        if sample == getattr(self.ed, '_active_sample', None):
            try:
                gid = self.ed._selected_gate_id()
            except Exception:                         # noqa: BLE001 - no gate selected; lay out all
                gid = None
        if not gating_strategy_steps(ctx.gates, ctx.order, gid):
            messagebox.showinfo(
                "Layout editor",
                f"{sample} has no gates drawn on axes to lay out.",
                parent=self)
            return
        try:
            view = self.ed._current_plot_view()
        except Exception:                             # noqa: BLE001 - no current view; default mode
            view = {}
        mode = str(view.get('mode') or 'pseudocolor')
        if mode == 'histogram':
            mode = 'pseudocolor'
        page = gating_strategy_page(
            ctx.gates, ctx.keys, ctx.order, gid, page=self.layout.page,
            name=self.doc.unique_name('Gating strategy'), mode=mode)
        idx = self.doc.add_page(page)
        self.page_idx = idx
        self._rebuild_tabs()
        self._go_page(idx)
        steps = sum(1 for it in page.items.values() if it['kind'] == 'plot')
        self.status.set(f"Added a gating strategy page: {steps} step(s). "
                        "Export → Batch export draws it for every sample.")

    # ── layout files ─────────────────────────────────────────────────────
    def _save_layout(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Save layout", defaultextension='.oflayout',
            initialfile='layout.oflayout', filetypes=_LAYOUT_TYPES)
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(self.doc.dumps())
        self.status.set(f"Saved {os.path.basename(path)}")

    def _open_layout(self):
        path = filedialog.askopenfilename(parent=self, title="Open layout",
                                          filetypes=_LAYOUT_TYPES)
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as fh:
                doc = LayoutDoc.loads(fh.read())
        except Exception as exc:                      # noqa: BLE001 - a file that does not parse as a layout is shown in an error dialog; the open one is kept
            show_error(self, "Layout editor",
                       "The layout could not be opened.", exc, path)
            return
        self.doc = doc
        self.ed._layout_doc = doc
        self.page_idx = 0
        self.dpi_var.set(str(doc.export_dpi))
        self._refresh_host()
        self._sel = None
        self._images.clear()
        self._rebuild_tabs()
        self._go_page(0)


# Old names of what moved to ui_layout_dialogs, served with a
# DeprecationWarning by __getattr__ below.
_MOVED = {'_ItemDialog': 'ItemDialog', '_ColorsDialog': 'ColorsDialog',
          '_LayerColorsDialog': 'LayerColorsDialog',
          '_SCALES': '_SCALES', '_EXPLODE_LABELS': '_EXPLODE_LABELS',
          '_SIDE_LABELS': '_SIDE_LABELS', '_NO_GATE': '_NO_GATE'}


def __getattr__(name):
    if name in _MOVED:
        new = _MOVED[name]
        warnings.warn(f'ui_layout_editor.{name} is deprecated and will be '
                      f'removed in a future release; use '
                      f'ui_layout_dialogs.{new}', DeprecationWarning,
                      stacklevel=2)
        from . import ui_layout_dialogs
        return getattr(ui_layout_dialogs, new)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
