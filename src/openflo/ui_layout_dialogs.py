"""The layout editor's dialogs: an item's settings, the default layer colours
and one item's layer colours.

Opened by ``ui_layout_editor.LayoutEditorWindow``; the page model is
``layout_page`` and drawing is ``layout_render`` (see the ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from .dialog_inputs import read_number
from .layout_page import (
    ALL_EVENTS,
    CHAN_STATS,
    CHART_TYPES,
    CHART_VALUES,
    POP_STATS,
    chart_item,
    item_populations,
    plot_item,
    stats_item,
    text_item,
)
from .tool_window import bind_right_click
from .ui_logic import resolve_channel
from .ui_tips import auto_tip

_SCALES = ('(current)', 'logicle', 'asinh', 'log', 'linear', 'symlog')
_EXPLODE_LABELS = ('Auto (when the zoom hides data)', 'Always when zoomed',
                   'Off')
_SIDE_LABELS = {'auto': 'Auto', 'right': 'Right', 'left': 'Left',
                'below': 'Below', 'above': 'Above'}
_NO_GATE = '(the zoom)'
_POP_STAT_TIPS = {
    'Count': "Add a column with the number of events in the population.",
    '%Parent': "Add a column with the population as a percent of the "
               "population it was gated from.",
    '%Total': "Add a column with the population as a percent of all the "
              "sample's events.",
}
_CHAN_STAT_TIPS = {
    'Median': "Add the middle value of the channel: half the events are "
              "brighter. Robust to a few extreme events.",
    'Mean': "Add the average value of the channel. Pulled by very bright "
            "or very dim events.",
    'GeoMean': "Add the geometric mean of the channel, the usual summary of "
               "a log-scaled signal (its form is set in Preferences).",
    'CV': "Add the coefficient of variation: the spread as a percent of the "
          "mean. Lower is a tighter population.",
    'rCV': "Add the robust CV: a spread measured on the middle of the "
           "data, as a percent of the median, so outliers barely move it "
           "(its form is set in Preferences).",
}


class ItemDialog(tk.Toplevel):
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
        except tk.TclError:
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
        return resolve_channel(text) if text else None

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
                                        values=chans, width=48),
                  "The channel on the horizontal axis. Type to search, or "
                  "pick from the list.")
        self._row(3, "Y:", ttk.Combobox(self.body, textvariable=self.y_var,
                                        values=chans, width=48),
                  "The channel on the vertical axis. Ignored by a "
                  "histogram, which plots X alone.")
        self.mode_var = tk.StringVar(value=it.get('mode', 'dot'))
        modes = [str(m) for m in getattr(
            self.ed, 'PLOT_MODES', ('dot', 'pseudocolor', 'contour',
                                    'histogram'))]
        self._row(4, "Plot type:", ttk.Combobox(
            self.body, textvariable=self.mode_var, values=modes,
            state='readonly', width=20),
            "How the events are drawn: dots, dots colored by how crowded "
            "they are (pseudocolor), density outlines (contour), or a "
            "histogram of X.")
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
        for var, axis in ((self.xscale_var, "horizontal"),
                          (self.yscale_var, "vertical")):
            if axis == "vertical":
                ttk.Label(sf, text="Y").pack(side='left', padx=(8, 0))
            cb = ttk.Combobox(sf, textvariable=var, values=list(_SCALES),
                              state='readonly', width=10)
            cb.pack(side='left', padx=4)
            auto_tip(cb, f"How the {axis} axis is stretched. logicle / "
                         "asinh spread out dim events near zero, log suits "
                         "bright ones, linear keeps raw values. (current) "
                         "follows the main window.")
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
        dg = ttk.Checkbutton(self.body, text="Draw gates and their %",
                             variable=self.gates_var)
        dg.grid(row=8, column=1, sticky='w', pady=3)
        auto_tip(dg, "Outline this population's child gates on the plot, "
                     "each with the percent of events inside it. Untick "
                     "for a clean plot.")
        self.title_var = tk.StringVar(value=it.get('title') or '')
        self._row(9, "Title:", ttk.Entry(self.body, textvariable=self.title_var,
                                         width=50),
                  "Blank: the population's name.")
        ef = ttk.Frame(self.body)
        cur = it.get('explode')
        self.explode_var = tk.StringVar(value=_EXPLODE_LABELS[
            {None: 0, True: 1, False: 2}.get(cur, 0)])
        ex = ttk.Combobox(ef, textvariable=self.explode_var,
                          values=list(_EXPLODE_LABELS), state='readonly',
                          width=30)
        ex.pack(side='left')
        auto_tip(ex, "When to draw the magnified copy beside the whole "
                     "plot. Auto (default): only when the zoom hides part of "
                     "the data. Off: never.")
        ttk.Label(ef, text="Inset:").pack(side='left', padx=(10, 4))
        self.side_var = tk.StringVar(value=_SIDE_LABELS.get(
            it.get('inset_side') or 'auto', 'Auto'))
        side = ttk.Combobox(ef, textvariable=self.side_var,
                            values=list(_SIDE_LABELS.values()),
                            state='readonly', width=8)
        side.pack(side='left')
        auto_tip(side, "Which side of the plot the magnified copy goes. "
                       "Auto (default): beside a wide box, below a tall "
                       "one.")
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
        except Exception:                             # noqa: BLE001 - a population that cannot be resolved on the page sample simply offers no gates to explode
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
            cb = ttk.Checkbutton(sf, text=s, variable=v)
            cb.pack(side='left', padx=(0, 8))
            auto_tip(cb, _POP_STAT_TIPS.get(s, "Add this column to the "
                                                "table."))
        self.chan_var = tk.StringVar(value=self._fmt(it.get('channel')))
        self._row(3, "Channel:", ttk.Combobox(
            self.body, textvariable=self.chan_var, width=48,
            values=[''] + self._channel_values()),
            "The channel the channel statistics below are measured on. "
            "Leave blank for counts and percentages only.")
        cf = ttk.Frame(self.body)
        self._row(4, "Channel stats:", cf)
        for s in CHAN_STATS:
            v = tk.BooleanVar(value=s in have)
            self.stat_vars[s] = v
            cb = ttk.Checkbutton(cf, text=s, variable=v)
            cb.pack(side='left', padx=(0, 8))
            auto_tip(cb, _CHAN_STAT_TIPS.get(s, "Add this column, measured "
                                                 "on the channel above."))
        self.title_var = tk.StringVar(value=it.get('title') or '')
        self._row(5, "Title:", ttk.Entry(self.body, textvariable=self.title_var,
                                         width=50),
                  "Printed above the table. Blank: 'Statistics'.")

    def _collect_stats(self):
        stats = [s for s, v in self.stat_vars.items() if v.get()]
        channel = self._res(self.chan_var.get())
        chan_stats = [s for s in stats if s in CHAN_STATS]
        if chan_stats and not channel:
            # The table leaves channel statistics out without a channel:
            # say so, and keep the dialog open to pick one.
            messagebox.showwarning(
                "Statistics",
                f"{', '.join(chan_stats)} "
                f"{'needs' if len(chan_stats) == 1 else 'need'} a channel. "
                "Pick one in Channel, or untick "
                f"{'it' if len(chan_stats) == 1 else 'them'}.",
                parent=self)
            return None
        return stats_item(
            populations=[self.choices[i][0]
                         for i in self.pop_list.curselection()],
            samples=self._picked_samples(self.sample_list),
            stats=stats,
            channel=channel,
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
            values=self._channel_values()),
            "The channel whose values violin and box charts show, and "
            "that a bar of Median / Mean / GeoMean is measured on.")
        self.value_var = tk.StringVar(value=it.get('value', 'Median'))
        self._row(4, "Bar value:", ttk.Combobox(
            self.body, textvariable=self.value_var, values=list(CHART_VALUES),
            state='readonly', width=12),
            "What a bar chart shows. Median / Mean / GeoMean use the "
            "channel.")
        self.title_var = tk.StringVar(value=it.get('title') or '')
        self._row(5, "Title:", ttk.Entry(self.body, textvariable=self.title_var,
                                         width=50),
                  "Printed above the chart. Blank: 'Distribution', or the "
                  "bar value.")

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
            self.body, from_=5, to=48, width=6, textvariable=self.size_var),
            "Font size in points, 5 to 48. 12 (default) is ordinary body "
            "text on a printed page.")

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
        LayerColorsDialog(self, rows, self.layer_colors, done)

    def _ok(self):
        new = getattr(self, f"_collect_{self.item['kind']}")()
        if new is None:
            return                      # a setting is not valid; already said
        if self.layer_colors and new['kind'] in ('plot', 'chart'):
            new['layer_colors'] = dict(self.layer_colors)
        self.destroy()
        self.on_ok(new)


class ColorsDialog(tk.Toplevel):
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
        sv = ttk.Button(bb, text="Save", default='active', command=self._save)
        sv.pack(side='right', padx=(0, 6))
        auto_tip(sv, "Keep this list and the Override setting as your "
                     "defaults for every layout, now and in later sessions.")
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
            auto_tip(sw, f"Layer {i + 1}'s color. Click to pick another; "
                         "right-click to remove it from the list.")

    def _change(self, i):
        from tkinter import colorchooser
        _, hexc = colorchooser.askcolor(color=self.colors[i], parent=self)
        if hexc:
            self.colors[i] = hexc
            self._draw()

    def _add(self):
        from tkinter import colorchooser
        _, hexc = colorchooser.askcolor(parent=self)
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


class LayerColorsDialog(tk.Toplevel):
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
