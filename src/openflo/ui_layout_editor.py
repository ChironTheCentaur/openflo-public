"""FlowJo-style layout editor (beta).

Pages you place plots of gated populations, stats tables, charts and text
on; drag an item to move it (it softly snaps to the grid and to other
items' edges and centres), drag its corner handle to resize it, double-click
it to edit it. Samples and gates dragged from the Samples & Gates tree or the
Pipeline (docked or popped out) land as plots, or join the plot or table
they are dropped on. The same layout re-applies to every sample: pick a
sample to preview it, or Batch export… writes every page once per sample.

The page model is ``layout_page``; drawing is ``layout_render``. This module
is only the window (see the ui_*.py convention).
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Literal

from .async_task import BusyStrip
from .dialog_inputs import read_number
from .errors import note as _note_failure
from .layout_page import (
    ALL_EVENTS,
    CHAN_STATS,
    CHART_TYPES,
    CHART_VALUES,
    EXPORT_DPI_CHOICES,
    PAGE_SIZES,
    POP_STATS,
    SCREEN_DPI,
    ZOOM_CHOICES,
    Layout,
    LayoutDoc,
    chart_item,
    clamp_box,
    grid_lines,
    item_populations,
    plot_item,
    population_choices,
    safe_filename,
    soft_snap,
    stats_item,
    text_item,
)
from .tool_window import bind_right_click, middle_button
from .ui_errors import show_error
from .ui_tips import auto_tip

_HANDLE = 9          # resize-handle side, px
_MARGIN = 16         # canvas space around the page, px
_SNAP_PX = 8         # soft-snap reach, px
_GRID_CHOICES = {'1/8 in': 0.125, '1/4 in': 0.25, '1/2 in': 0.5,
                 '1 in': 1.0}
_SCALES = ('(current)', 'logicle', 'asinh', 'log', 'linear', 'symlog')
_EXPLODE_LABELS = ('Auto (when the zoom hides data)', 'Always when zoomed',
                   'Off')
_SIDE_LABELS = {'auto': 'Auto', 'right': 'Right', 'left': 'Left',
                'below': 'Below', 'above': 'Above'}
_NO_GATE = '(the zoom)'
_Side = Literal['left', 'right', 'top', 'bottom']
_LAYOUT_TYPES = [('OpenFlo layout', '*.oflayout'), ('JSON', '*.json')]


def _photo(fig, master):
    """A Tk image of ``fig`` (PNG through Tk's own decoder; no PIL)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, facecolor='white')
    return tk.PhotoImage(master=master,
                         data=base64.b64encode(buf.getvalue()).decode('ascii'))


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
        self._images = {}            # cache key -> (PhotoImage, outside)
        self._shown = {}             # iid -> PhotoImage on the canvas
        self._outside = {}           # iid -> fraction outside its zoom
        self._geom = (1.0, 0, 0)     # px per inch, page origin x, y
        self._redraw_after = None
        self._zoom = 'Fit'
        self._axes = {}              # iid -> its plot's axes, for zooming
        # Undo and redo: the layout as it was before each change (UI 39).
        self._undo_stack: list = []  # (layout JSON, page index)
        self._redo_stack: list = []
        self._last_state = None

        self._build_toolbar()
        self._build_tabs()
        # The status line is packed before the page so it keeps its height,
        # and wraps instead of running off both ends (UI 39).
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
        self.bind('<Delete>', lambda _e: self._delete_selected())
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

    def _tb_button(self, parent, text, cmd, tip, **kw):
        # Natural width: the theme's default minimum made every button the
        # same wide slab, and the row ran off a 1240px window.
        kw.setdefault('width', 0)
        b = ttk.Button(parent, text=text, command=cmd, **kw)
        b.pack(side='left', padx=(0, self._TIGHT))
        auto_tip(b, tip)
        return b

    def _tb_sep(self, parent):
        ttk.Separator(parent, orient='vertical').pack(
            side='left', fill='y', padx=self._GAP, pady=2)

    def _tb_combo(self, parent, var, values, tip, width, cmd):
        cb = ttk.Combobox(parent, textvariable=var, width=width,
                          state='readonly', values=list(values))
        cb.pack(side='left', padx=(0, self._TIGHT))
        cb.bind('<<ComboboxSelected>>', lambda _e: cmd())
        auto_tip(cb, tip)
        return cb

    def _tb_menu(self, parent, text, tip):
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
        # row the packer dropped Export and Layout off the end.
        left = ttk.Frame(bar)
        left.grid(row=0, column=0, sticky='w')

        # Insert
        b = self._tb_button
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
        self._tb_sep(left)
        # The selected item
        b(left, "Edit…", self._edit_selected,
          "Change the selected item's settings (or double-click it).")
        b(left, "Duplicate", self._duplicate_selected,
          "Copy the selected item, offset slightly.")
        b(left, "Delete", self._delete_selected,
          "Remove the selected item from the page (Delete key).")

        # Right side, packed right to left: File, Export, then the sample.
        right = ttk.Frame(bar)
        # Follows the rest instead of pinning to the window's right edge:
        # on a wide window an edge-pinned group drifted away from its row.
        right.grid(row=0, column=1, sticky='w', padx=(self._GAP, 0))
        self._tb_groups = (bar, left, right)
        self._tb_wrapped = False
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
        self.sample_combo = self._tb_combo(
            right, self.sample_var, names,
            "The sample the page shows. The layout is the same for every "
            "sample; populations are matched by name.", 26, self._redraw)
        b(right, "▶", lambda: self._step_sample(1),
          "Show the next sample.", width=0)
        b(right, "⟳", self._refresh,
          "Refresh: redraw from the current samples and gates, after a "
          "change in the main window (F5).", width=0)
        self._tb_sep(right)

        # Export: its resolution is on the button, so it is never hidden.
        self.dpi_var = tk.StringVar(value=str(self.doc.export_dpi))
        self.bg_var = tk.StringVar(value='White')
        self._export_btn, em = self._tb_menu(
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
        for v in ('White', 'Transparent', 'Translucent'):
            bgm.add_radiobutton(label=v, value=v, variable=self.bg_var)
        em.add_cascade(label="Background", menu=bgm)
        for m in (res, bgm):
            try:
                self.ed._theme_menu(m)
            except Exception:                         # noqa: BLE001 - menu colours are cosmetic
                pass
        self.dpi_var.trace_add('write', lambda *_: self._label_export())
        self._label_export()

        _fb, fm = self._tb_menu(right, "Layout",
                                "Open or save the layout file, and the "
                                "default colors.")
        fm.add_command(label="Open layout…", command=self._open_layout)
        fm.add_command(label="Save layout…", command=self._save_layout)
        fm.add_separator()
        fm.add_command(label="Colors…", command=self._edit_default_colors)

    def _reflow_toolbar(self, _e=None):
        """Put the sample/Export/Layout group on its own row when the bar is
        narrower than both groups side by side, and back when it is not."""
        bar, left, right = self._tb_groups
        need = (left.winfo_reqwidth() + right.winfo_reqwidth() + self._GAP
                + 16)                          # the bar's own padding
        wrap = bar.winfo_width() < need
        if wrap == self._tb_wrapped:
            return
        self._tb_wrapped = wrap
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
        b = self._tb_button
        b(row, "＋", self._new_page, "Add an empty page.", width=0)
        self._pages_btn = b(
            row, "▾", self._pages_menu_at_button,
            "Pages: rename, duplicate or delete this page; bring in pages "
            "from another project's layout file, or send this page to one.",
            width=0)
        self._tb_sep(row)
        self.page_var = tk.StringVar(value=self.layout.page)
        self._tb_combo(row, self.page_var, PAGE_SIZES,
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
        self._tb_combo(row, self.grid_step_var, _GRID_CHOICES,
                       "Grid spacing.", 6, self._redraw)
        self.snap_var = tk.BooleanVar(value=True)
        sn = ttk.Checkbutton(row, text="Snap", variable=self.snap_var)
        sn.pack(side='left', padx=(self._GAP, 0))
        auto_tip(sn, "Soft snapping: an item being moved or resized is "
                     "pulled onto nearby grid lines and other items' edges, "
                     "corners and centres, and guides show what it lines up "
                     "with. Off: free placement.")
        self._tb_sep(row)
        ttk.Label(row, text="Zoom").pack(side='left',
                                         padx=(0, self._TIGHT + 2))
        self.zoom_var = tk.StringVar(value='Fit')
        self._tb_combo(row, self.zoom_var, list(ZOOM_CHOICES) + ['Custom…'],
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
        except Exception as exc:                      # noqa: BLE001
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
        except Exception as exc:                      # noqa: BLE001
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
            except tk.TclError:                         # noqa: BLE001
                pass
        self._redraw_after = self.after(120, self._redraw)

    def _sample(self):
        nm = self.sample_var.get()
        return nm if nm in getattr(self.ed, '_samples', {}) else None

    def _gate_sig(self, samples):
        out = []
        for s in samples:
            try:
                out.append(tuple(sorted(self.ed._pc_sigs(s).items())))
            except Exception:                         # noqa: BLE001
                out.append(())
        return tuple(out)

    def _redraw(self):
        """Redraw the page; items whose look has not changed reuse their
        cached image."""
        import json

        from .layout_page import item_samples
        from .layout_render import RenderContext, render_item
        self._redraw_after = None
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
        self._shown, self._outside, self._axes = {}, {}, {}
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
                    cached = (_photo(fig, self),
                              0.0 if info.get('exploded')
                              else info.get('outside', 0.0),
                              info.get('ax'))
                except Exception as exc:              # noqa: BLE001
                    cached = None
                    self.status.set(f"Could not draw an item: {exc}")
                if cached is not None:
                    self._images = {k: v for k, v in self._images.items()
                                    if k[:3] != key[:3]}
                    self._images[key] = cached
            if cached is not None:
                img, outside, ax = cached
                self._shown[iid] = img
                self._outside[iid] = outside
                if ax is not None:
                    self._axes[iid] = ax
                c.create_image(bx0, by0, image=img, anchor='nw',
                               tags=('item', f'img:{iid}'))
            else:
                c.create_rectangle(bx0, by0, bx1, by1, outline='#bbbbbb',
                                   dash=(3, 3), tags=('item', f'img:{iid}'))
                c.create_text((bx0 + bx1) / 2, (by0 + by1) / 2,
                              text=it['kind'], fill='#999999',
                              tags=('item', f'img:{iid}'))
        self._draw_fit_badges()
        self._draw_uids()
        self._draw_selection()
        self._record_change()

    # ── Undo, redo and saving (UI 39) ──────────────────────────────────────
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
            return
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
            bx0, by0, bx1, by1 = self._box_px(
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
            return
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
            except tk.TclError:                         # noqa: BLE001
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
        except Exception:                             # noqa: BLE001
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
        except Exception:                             # noqa: BLE001
            view = {}
        x, y = view.get('x'), view.get('y')
        mode = view.get('mode', 'dot')
        population = ALL_EVENTS
        key, gate = self._selected_gate_key()
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
        _ItemDialog(self, item, self._choices(), on_ok=self._insert)

    def _add_stats(self):
        key, _gate = self._selected_gate_key()
        item = stats_item(populations=[key] if key else [],
                          box=self.layout.next_slot(0.3, 0.25))
        _ItemDialog(self, item, self._choices(), on_ok=self._insert)

    def _add_chart(self):
        key, _gate = self._selected_gate_key()
        try:
            ch = self.ed._resolve_channel(self.ed.x_combo.get())
        except Exception:                             # noqa: BLE001
            ch = None
        item = chart_item(populations=[key] if key else [ALL_EVENTS],
                          samples=self._batch_samples(), channel=ch,
                          box=self.layout.next_slot(0.3, 0.42))
        _ItemDialog(self, item, self._choices(), on_ok=self._insert)

    def _add_text(self):
        self._insert(text_item(box=self.layout.next_slot(0.4, 0.06)))

    def _insert(self, item):
        self._sel = self.layout.add(item)
        self._redraw()

    def _edit_selected(self):
        if self._sel not in self.layout.items:
            return
        iid, page = self._sel, self.layout

        def ok(new):
            new['box'] = page.items[iid]['box']
            page.items[iid] = new
            self._redraw()
        _ItemDialog(self, dict(page.items[iid]), self._choices(), on_ok=ok)

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
        if self._sel in self.layout.items:
            self._sel = self.layout.duplicate(self._sel)
            self._redraw()

    def _raise_selected(self):
        if self._sel in self.layout.items:
            self.layout.raise_to_top(self._sel)
            self._redraw()

    def _delete_selected(self):
        if self._sel in self.layout.items:
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
        _ColorsDialog(self, on_ok=lambda: self._redraw())

    # ── export ───────────────────────────────────────────────────────────
    def _batch_samples(self):
        try:
            names = list(self.ed._selected_samples())
        except Exception:                             # noqa: BLE001
            names = []
        return names or [n for n in getattr(self.ed, '_sample_order', [])
                         if n in self.ed._samples]

    def _save_fig(self, fig, path):
        from .theme import savefig_background
        savefig_background(fig, path, background=self.bg_var.get(),
                           dpi=self.doc.export_dpi, tight=False,
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
            initialfile=(f'{safe_filename(sample)}_'
                         f'{safe_filename(self.layout.name)}.pdf'),
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
        except Exception as exc:                      # noqa: BLE001
            show_error(self, "Layout editor",
                       "The page could not be exported.", exc, path)
            return
        self._audit('layout.export', path=path, n_items=len(self.layout.items),
                    n_samples=1, dpi=self.doc.export_dpi, samples=[sample])
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
                transparent = self.bg_var.get() == 'Transparent'
                # Editable text, as every other export (figure_rc). The PDF
                # reads it while drawing a page and again when it writes the
                # fonts at close, so both run under it; it is not left set
                # while the window waits between samples.
                rc = figure_rc()
                pdf = PdfPages(path, metadata=savefig_metadata(
                    fmt='pdf', corrections=self._corrections()))
                try:
                    for i, nm in enumerate(samples):
                        yield (i, n, f"Exporting {nm} ({i + 1} of {n})…")
                        for page in pages:
                            fig = render_page(self.ed, page, nm)
                            with matplotlib.rc_context(rc):
                                pdf.savefig(fig, dpi=dpi,
                                            facecolor='none' if transparent
                                            else 'white')
                        written.append(nm)
                finally:
                    with matplotlib.rc_context(rc):
                        pdf.close()
            else:
                for i, nm in enumerate(samples):
                    yield (i, n, f"Exporting {nm} ({i + 1} of {n})…")
                    for page in pages:
                        suffix = (f'_{safe_filename(page.name)}'
                                  if len(pages) > 1 else '')
                        out = f'{stem}_{safe_filename(nm)}{suffix}{ext}'
                        self._save_fig(render_page(self.ed, page, nm), out)
                        outs.append(out)
                    written.append(nm)

        def done(_result):
            self._audit('layout.export', path=path,
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
            except Exception:                         # noqa: BLE001
                gid = None
        if not gating_strategy_steps(ctx.gates, ctx.order, gid):
            messagebox.showinfo(
                "Layout editor",
                f"{sample} has no gates drawn on axes to lay out.",
                parent=self)
            return
        try:
            view = self.ed._current_plot_view()
        except Exception:                             # noqa: BLE001
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

    def _audit(self, action, **kw):
        try:
            self.ed._audit(action, **kw)
        except Exception as exc:  # noqa: BLE001 - reported via note()
            _note_failure("write the layout audit entry", exc)
            pass

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
        except Exception as exc:                      # noqa: BLE001
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


class _ItemDialog(tk.Toplevel):
    """Edit one item's settings. Calls ``on_ok(item)`` with a new dict."""

    def __init__(self, parent, item, choices, on_ok):
        super().__init__(parent)
        self.transient(parent)
        self.resizable(False, False)
        self.item = dict(item)
        self.on_ok = on_ok
        self.ed = parent.ed
        self.choices = list(choices)
        self.samples = list(getattr(self.ed, '_sample_order', []) or [])
        self.page_sample = parent._sample()
        self.layer_colors = dict(self.item.get('layer_colors') or {})
        kind = self.item['kind']
        self.title({'plot': "Plot", 'stats': "Statistics", 'chart': "Chart",
                    'text': "Text"}[kind])
        self.body = ttk.Frame(self, padding=12)
        self.body.pack(fill='both', expand=True)
        getattr(self, f'_build_{kind}')()
        if kind in ('plot', 'chart'):
            lc = ttk.Button(self.body, text="Layer colors…",
                            command=self._open_layer_colors)
            lc.grid(row=98, column=1, sticky='w', pady=(6, 0))
            auto_tip(lc, "Set the color of each sample / population layer "
                         "of this item. Unset layers follow the default "
                         "colors (toolbar Colors…).")
        bb = ttk.Frame(self.body)
        bb.grid(row=99, column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="OK", default='active', command=self._ok).pack(side='right',
                                                         padx=(0, 6))
        self.bind('<Escape>', lambda _e: self.destroy())
        try:
            self.grab_set()
        except tk.TclError:                             # noqa: BLE001
            pass

    # helpers
    def _row(self, r, label, widget, tip=None):
        ttk.Label(self.body, text=label).grid(row=r, column=0, sticky='nw',
                                              pady=3, padx=(0, 8))
        widget.grid(row=r, column=1, sticky='we', pady=3)
        if tip:
            auto_tip(widget, tip)

    def _channel_values(self):
        return [self.ed._fmt_channel(c)
                for c in getattr(self.ed, '_channels', [])]

    def _fmt(self, ch):
        return self.ed._fmt_channel(ch) if ch else ''

    def _listbox(self, r, label, entries, chosen, tip, height=7):
        f = ttk.Frame(self.body)
        lb = tk.Listbox(f, selectmode='extended', height=height, width=56,
                        exportselection=False)
        sb = ttk.Scrollbar(f, orient='vertical', command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        for i, e in enumerate(entries):
            lb.insert('end', e)
            if i in chosen:
                lb.selection_set(i)
        self._row(r, label, f)
        auto_tip(lb, tip)
        return lb

    def _pop_list(self, r, keys):
        chosen = {i for i, (k, _p) in enumerate(self.choices) if k in keys}
        return self._listbox(
            r, "Populations:", [p for _k, p in self.choices], chosen,
            "One or more populations (Ctrl/Shift-click). Several are "
            "overlaid in one plot, or listed in one table or chart.")

    def _sample_list(self, r, chosen_names, tip):
        chosen = {i for i, n in enumerate(self.samples) if n in chosen_names}
        return self._listbox(r, "Samples:", self.samples, chosen, tip,
                             height=5)

    def _picked_pops(self, lb, fallback):
        keys = [self.choices[i][0] for i in lb.curselection()]
        if keys:
            return keys
        # Keys from another project's layout that this sample lacks stay.
        return [k for k in fallback
                if k not in {c for c, _p in self.choices}] or [ALL_EVENTS]

    def _picked_samples(self, lb):
        return [self.samples[i] for i in lb.curselection()]

    def _res(self, text):
        text = (text or '').strip()
        return self.ed._resolve_channel(text) if text else None

    # plot
    def _build_plot(self):
        from .layout_render import CAT_PREFIX, label_columns
        it = self.item
        self.pop_list = self._pop_list(0, item_populations(it))
        self.sample_list = self._sample_list(
            1, it.get('samples') or (),
            "Leave empty to follow the page's sample (the layout re-applies "
            "per sample). Pick several to overlay them in this plot.")
        chans = self._channel_values()
        self.x_var = tk.StringVar(value=self._fmt(it.get('x')))
        self.y_var = tk.StringVar(value=self._fmt(it.get('y')))
        self._row(2, "X:", ttk.Combobox(self.body, textvariable=self.x_var,
                                        values=chans, width=48))
        self._row(3, "Y:", ttk.Combobox(self.body, textvariable=self.y_var,
                                        values=chans, width=48))
        self.mode_var = tk.StringVar(value=it.get('mode', 'dot'))
        modes = [str(m) for m in getattr(
            self.ed, 'PLOT_MODES', ('dot', 'pseudocolor', 'contour',
                                    'histogram'))]
        self._row(4, "Plot type:", ttk.Combobox(
            self.body, textvariable=self.mode_var, values=modes,
            state='readonly', width=20))
        self._cats = {f'By {c} (clusters)': f'{CAT_PREFIX}{c}'
                      for c in label_columns(self.ed)}
        rev = {v: k for k, v in self._cats.items()}
        self.color_var = tk.StringVar(
            value=rev.get(str(it.get('color') or ''),
                          it.get('color') or 'By density'))
        self._row(5, "Color:", ttk.Combobox(
            self.body, textvariable=self.color_var, width=48,
            values=['By density'] + list(self._cats) + chans),
            "Dot plots: density, a cluster result (PhenoGraph, Leiden, "
            "FlowSOM: each cluster its own color), or a channel. Overlays "
            "color each layer.")
        sf = ttk.Frame(self.body)
        self.xscale_var = tk.StringVar(value=it.get('xscale') or '(current)')
        self.yscale_var = tk.StringVar(value=it.get('yscale') or '(current)')
        ttk.Label(sf, text="X").pack(side='left')
        ttk.Combobox(sf, textvariable=self.xscale_var, values=list(_SCALES),
                     state='readonly', width=10).pack(side='left', padx=4)
        ttk.Label(sf, text="Y").pack(side='left', padx=(8, 0))
        ttk.Combobox(sf, textvariable=self.yscale_var, values=list(_SCALES),
                     state='readonly', width=10).pack(side='left', padx=4)
        self._row(6, "Axis types:", sf,
                  "Kept from the plot this was taken from. (current) follows "
                  "the main window's setting for the channel.")
        self.zoom_keep = tk.BooleanVar(
            value=bool(it.get('xlim') or it.get('ylim')))
        zk = ttk.Checkbutton(self.body, text="Keep the zoom it was taken with "
                             "(untick to fit the data)",
                             variable=self.zoom_keep,
                             state='normal' if (it.get('xlim')
                                                or it.get('ylim'))
                             else 'disabled')
        zk.grid(row=7, column=1, sticky='w', pady=3)
        auto_tip(zk, "The plot keeps the zoomed view it had in the main "
                     "window. When a lot of its data falls outside, the page "
                     "offers Zoom to fit; it is never applied on its own.")
        self.gates_var = tk.BooleanVar(value=bool(it.get('draw_gates', True)))
        ttk.Checkbutton(self.body, text="Draw gates and their %",
                        variable=self.gates_var).grid(
            row=8, column=1, sticky='w', pady=3)
        self.title_var = tk.StringVar(value=it.get('title') or '')
        self._row(9, "Title:", ttk.Entry(self.body, textvariable=self.title_var,
                                         width=50),
                  "Blank: the population's name.")
        ef = ttk.Frame(self.body)
        cur = it.get('explode')
        self.explode_var = tk.StringVar(value=_EXPLODE_LABELS[
            {None: 0, True: 1, False: 2}.get(cur, 0)])
        ttk.Combobox(ef, textvariable=self.explode_var,
                     values=list(_EXPLODE_LABELS), state='readonly',
                     width=30).pack(side='left')
        ttk.Label(ef, text="Inset:").pack(side='left', padx=(10, 4))
        self.side_var = tk.StringVar(value=_SIDE_LABELS.get(
            it.get('inset_side') or 'auto', 'Auto'))
        ttk.Combobox(ef, textvariable=self.side_var,
                     values=list(_SIDE_LABELS.values()), state='readonly',
                     width=8).pack(side='left')
        self._row(10, "Exploded view:", ef,
                  "The whole plot, with the zoomed region (or the gate "
                  "below) outlined and shown magnified beside it, joined by "
                  "lines. Auto: when the zoom hides part of the data. Inset: "
                  "which side the magnified copy goes.")
        self._gate_keys = self._child_gate_choices(it)
        rev_g = {v: k for k, v in self._gate_keys.items()}
        self.egate_var = tk.StringVar(value=rev_g.get(
            it.get('explode_gate'), _NO_GATE))
        self._row(11, "Magnify gate:", ttk.Combobox(
            self.body, textvariable=self.egate_var, state='readonly',
            values=[_NO_GATE] + list(self._gate_keys), width=48),
            "Show one of this plot's gates exploded out: its events "
            "magnified beside the whole plot. (the zoom) magnifies the "
            "zoomed region instead.")

    def _child_gate_choices(self, it):
        """``{label: key}`` for the gates drawn on this plot (the page's
        sample), to pick one to explode."""
        from .layout_page import child_gates
        from .layout_render import SampleContext
        out = {}
        if not self.page_sample:
            return out
        try:
            ctx = SampleContext(self.ed, self.page_sample)
            gid = ctx.gid(item_populations(it)[0])
        except Exception:                             # noqa: BLE001
            return out
        for cg in child_gates(ctx.gates, gid):
            if cg in ctx.keys:
                out[ctx.path(cg)] = ctx.keys[cg]
        return out

    def _collect_plot(self):
        it = self.item
        color = self.color_var.get().strip() or 'By density'
        color = self._cats.get(color, color)
        keep = self.zoom_keep.get()
        xs, ys = self.xscale_var.get(), self.yscale_var.get()
        return plot_item(
            populations=self._picked_pops(self.pop_list,
                                          item_populations(it)),
            samples=self._picked_samples(self.sample_list),
            x=self._res(self.x_var.get()), y=self._res(self.y_var.get()),
            mode=self.mode_var.get(), color=color,
            draw_gates=self.gates_var.get(),
            title=self.title_var.get().strip() or None,
            xscale=None if xs == '(current)' else xs,
            yscale=None if ys == '(current)' else ys,
            xlim=it.get('xlim') if keep else None,
            ylim=it.get('ylim') if keep else None, box=it['box'],
            explode=(None, True, False)[_EXPLODE_LABELS.index(
                self.explode_var.get())],
            explode_gate=self._gate_keys.get(self.egate_var.get()),
            inset_side={v: k for k, v in _SIDE_LABELS.items()}.get(
                self.side_var.get(), 'auto'))

    # stats
    def _build_stats(self):
        it = self.item
        self.pop_list = self._pop_list(0, it.get('populations') or ())
        self.sample_list = self._sample_list(
            1, it.get('samples') or (),
            "Leave empty for the page's sample. Pick several for one table "
            "with a row per sample and population.")
        sf = ttk.Frame(self.body)
        self._row(2, "Statistics:", sf)
        have = set(it.get('stats') or ())
        self.stat_vars = {}
        for s in POP_STATS:
            v = tk.BooleanVar(value=s in have)
            self.stat_vars[s] = v
            ttk.Checkbutton(sf, text=s, variable=v).pack(side='left',
                                                         padx=(0, 8))
        self.chan_var = tk.StringVar(value=self._fmt(it.get('channel')))
        self._row(3, "Channel:", ttk.Combobox(
            self.body, textvariable=self.chan_var, width=48,
            values=[''] + self._channel_values()))
        cf = ttk.Frame(self.body)
        self._row(4, "Channel stats:", cf)
        for s in CHAN_STATS:
            v = tk.BooleanVar(value=s in have)
            self.stat_vars[s] = v
            ttk.Checkbutton(cf, text=s, variable=v).pack(side='left',
                                                         padx=(0, 8))
        self.title_var = tk.StringVar(value=it.get('title') or '')
        self._row(5, "Title:", ttk.Entry(self.body, textvariable=self.title_var,
                                         width=50))

    def _collect_stats(self):
        return stats_item(
            populations=[self.choices[i][0]
                         for i in self.pop_list.curselection()],
            samples=self._picked_samples(self.sample_list),
            stats=[s for s, v in self.stat_vars.items() if v.get()],
            channel=self._res(self.chan_var.get()),
            title=self.title_var.get().strip() or None, box=self.item['box'])

    # chart
    def _build_chart(self):
        it = self.item
        self.chart_var = tk.StringVar(value=it.get('chart', 'violin'))
        self._row(0, "Chart:", ttk.Combobox(
            self.body, textvariable=self.chart_var, values=list(CHART_TYPES),
            state='readonly', width=12),
            "violin / box: each population's distribution of the channel, "
            "per sample. bar: one statistic per sample and population.")
        self.pop_list = self._pop_list(1, it.get('populations') or ())
        self.sample_list = self._sample_list(
            2, it.get('samples') or (),
            "The samples compared. Leave empty for the page's sample only.")
        self.chan_var = tk.StringVar(value=self._fmt(it.get('channel')))
        self._row(3, "Channel:", ttk.Combobox(
            self.body, textvariable=self.chan_var, width=48,
            values=self._channel_values()))
        self.value_var = tk.StringVar(value=it.get('value', 'Median'))
        self._row(4, "Bar value:", ttk.Combobox(
            self.body, textvariable=self.value_var, values=list(CHART_VALUES),
            state='readonly', width=12),
            "What a bar chart shows. Median / Mean / GeoMean use the "
            "channel.")
        self.title_var = tk.StringVar(value=it.get('title') or '')
        self._row(5, "Title:", ttk.Entry(self.body, textvariable=self.title_var,
                                         width=50))

    def _collect_chart(self):
        it = self.item
        return chart_item(
            chart=self.chart_var.get(),
            populations=self._picked_pops(self.pop_list,
                                          it.get('populations') or ()),
            samples=self._picked_samples(self.sample_list),
            channel=self._res(self.chan_var.get()),
            value=self.value_var.get(),
            title=self.title_var.get().strip() or None, box=it['box'])

    # text
    def _build_text(self):
        it = self.item
        self.text_var = tk.StringVar(value=it.get('text', ''))
        self._row(0, "Text:", ttk.Entry(self.body, textvariable=self.text_var,
                                        width=50),
                  "{sample} is replaced by the sample name.")
        self.size_var = tk.StringVar(value=str(it.get('size', 12)))
        self._row(1, "Size (pt):", ttk.Spinbox(
            self.body, from_=5, to=48, width=6, textvariable=self.size_var))

    def _collect_text(self):
        size = read_number(self.size_var, "Size (pt)", lo=5, hi=48,
                           parent=self)
        if size is None:
            return None
        return text_item(text=self.text_var.get(), size=size,
                         box=self.item['box'])

    def _open_layer_colors(self):
        from .layout_render import RenderContext, layer_id
        cur = getattr(self, f"_collect_{self.item['kind']}")()
        if cur is None:
            return                      # a setting is not valid; already said
        rc = RenderContext(self.ed, self.page_sample)
        layers, _missing = rc.layers(cur)
        if not layers:
            return
        cur['layer_colors'] = {}
        natural = rc.layer_colors(cur, layers)
        pinned = bool(cur.get('samples'))
        rows = [(layer_id(c, g, pinned), f"{c.sample} · {c.name(g)}",
                 natural[i])
                for i, (c, g) in enumerate(layers)]

        def done(colors):
            self.layer_colors = colors
        _LayerColorsDialog(self, rows, self.layer_colors, done)

    def _ok(self):
        new = getattr(self, f"_collect_{self.item['kind']}")()
        if new is None:
            return                      # a setting is not valid; already said
        if self.layer_colors and new['kind'] in ('plot', 'chart'):
            new['layer_colors'] = dict(self.layer_colors)
        self.destroy()
        self.on_ok(new)


class _ColorsDialog(tk.Toplevel):
    """The default colour list (layer 1, layer 2, ...) and the override
    toggle, saved as preferences (``layout_colors``,
    ``layout_color_override``)."""

    def __init__(self, parent, on_ok=None):
        from .layout_render import color_prefs
        super().__init__(parent)
        self.title("Layout colors")
        self.transient(parent)
        self.resizable(False, False)
        self.on_ok = on_ok
        cols, override = color_prefs()
        self.colors: list[str] = [str(c) for c in cols]
        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text="Default colors, in layer order (sample or "
                  "gate 1, 2, ...):").pack(anchor='w')
        self.swatches = tk.Frame(body)
        self.swatches.pack(anchor='w', pady=(6, 6))
        row = ttk.Frame(body)
        row.pack(anchor='w')
        for text, cmd, tip in (
                ("Add", self._add, "Add a color at the end."),
                ("Reset", self._reset, "Back to the built-in list.")):
            b = ttk.Button(row, text=text, command=cmd)
            b.pack(side='left', padx=(0, 4))
            auto_tip(b, tip)
        self.override_var = tk.BooleanVar(value=override)
        ov = ttk.Checkbutton(
            body, text="Override: every layer uses this list instead of its "
            "sample's or gate's own color", variable=self.override_var)
        ov.pack(anchor='w', pady=(8, 0))
        auto_tip(ov, "Off (default): a layer takes its sample's color (or its "
                     "gate's, when layers are populations), and this list "
                     "fills in. A color set on an item's layer always wins.")
        ttk.Label(body, text="Click a color to change it; right-click "
                  "removes it.", style='Muted.TLabel').pack(anchor='w',
                                                         pady=(6, 0))
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(10, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="Save", default='active', command=self._save).pack(side='right',
                                                             padx=(0, 6))
        self._draw()

    def _draw(self):
        for w in self.swatches.winfo_children():
            w.destroy()
        for i, c in enumerate(self.colors):
            sw = tk.Label(self.swatches, text=str(i + 1), width=3, bg=c,
                          fg='white', relief='raised', cursor='hand2')
            sw.grid(row=i // 10, column=i % 10, padx=1, pady=1)
            sw.bind('<Button-1>', lambda _e, i=i: self._change(i))
            bind_right_click(sw, lambda _e, i=i: self._remove(i))

    def _change(self, i):
        from tkinter import colorchooser
        rgb, hexc = colorchooser.askcolor(color=self.colors[i], parent=self)
        if hexc:
            self.colors[i] = hexc
            self._draw()

    def _add(self):
        from tkinter import colorchooser
        rgb, hexc = colorchooser.askcolor(parent=self)
        if hexc:
            self.colors.append(hexc)
            self._draw()

    def _remove(self, i):
        if len(self.colors) > 1:
            self.colors.pop(i)
            self._draw()

    def _reset(self):
        from .layout_render import DEFAULT_COLORS
        self.colors = [str(c) for c in DEFAULT_COLORS]
        self._draw()

    def _save(self):
        from .prefs import write_pref
        write_pref('layout_colors', list(self.colors))
        write_pref('layout_color_override', bool(self.override_var.get()))
        self.destroy()
        if self.on_ok:
            self.on_ok()


class _LayerColorsDialog(tk.Toplevel):
    """Colour each layer of one plot or chart. Calls ``on_ok(colors)`` with
    ``{layer id: colour}`` (a layer left unset follows the defaults)."""

    def __init__(self, parent, layers, current, on_ok):
        super().__init__(parent)
        self.title("Layer colors")
        self.transient(parent)
        self.resizable(False, False)
        self.on_ok = on_ok
        self.colors = dict(current or {})
        body = ttk.Frame(self, padding=12)
        body.pack(fill='both', expand=True)
        self.rows = ttk.Frame(body)
        self.rows.pack(anchor='w')
        self.layers = layers                  # [(layer id, label, colour)]
        self._draw()
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(10, 0))
        ttk.Button(bb, text="Cancel", command=self.destroy).pack(side='right')
        ttk.Button(bb, text="OK", default='active', command=self._ok).pack(side='right',
                                                         padx=(0, 6))

    def _draw(self):
        for w in self.rows.winfo_children():
            w.destroy()
        for r, (lid, label, natural) in enumerate(self.layers):
            c = self.colors.get(lid) or natural
            sw = tk.Label(self.rows, text='   ', bg=c, relief='raised',
                          cursor='hand2')
            sw.grid(row=r, column=0, padx=(0, 6), pady=1)
            sw.bind('<Button-1>', lambda _e, lid=lid, c=c: self._pick(lid, c))
            ttk.Label(self.rows, text=label).grid(row=r, column=1, sticky='w')
            if lid in self.colors:
                b = ttk.Button(self.rows, text="Default", width=8,
                               command=lambda lid=lid: self._unset(lid))
                b.grid(row=r, column=2, padx=(6, 0))
                auto_tip(b, "Back to the default color for this layer.")

    def _pick(self, lid, cur):
        from tkinter import colorchooser
        _rgb, hexc = colorchooser.askcolor(color=cur, parent=self)
        if hexc:
            self.colors[lid] = hexc
            self._draw()

    def _unset(self, lid):
        self.colors.pop(lid, None)
        self._draw()

    def _ok(self):
        self.destroy()
        self.on_ok(self.colors)
