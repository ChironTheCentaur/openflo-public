"""Analyze > 3D plot: three channels (or a 3D embedding) as a cloud you turn.

The maths and the pictures live in :mod:`openflo.view3d`; this window picks
the data from the editor (the sample, the population, the channels), keeps
the view, and saves PNG / GIF / MP4. Each axis uses the transform the 2D plot
gives that channel, so the front view is the 2D plot of X against Y.
"""
from __future__ import annotations

import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, ttk
from typing import Any

import numpy as np

from . import plotcache as _pcm
from . import ui_errors
from . import view3d as v3
from .async_task import BusyStrip
from .channel_roles import EMBED_PREFIXES
from .theme import THEME_ALL_DARK, THEME_LIGHT, THEMES, current_palette, density_cmap
from .tool_window import ToolWindow
from .ui_tips import auto_tip

AS_SHOWN = 'As shown in the plot'
ALL_EVENTS = 'All events'
COLOR_LABELS = {'density': 'Density', 'marker': 'Channel',
                'cluster': 'Cluster', 'gate': 'Gate'}


@dataclass
class Cloud:
    """The events the window draws, already in -1..1 cube space."""
    points: np.ndarray
    colors: np.ndarray
    total: int          # events in the population with all three values
    shown: int          # events drawn after the cap
    off_axis: int       # drawn-set events a scale cannot place or outside a set range
    seed: int


def embedding_triplet(columns) -> list[str] | None:
    """``[P1, P2, P3]`` of the first embedding with a third component among
    ``columns`` (UMAP1/UMAP2/UMAP3 ...), else None."""
    cols = set(columns)
    for p in EMBED_PREFIXES:
        trio = [f'{p}1', f'{p}2', f'{p}3']
        if all(c in cols for c in trio):
            return trio
    return None


def parse_cap(text) -> int | None:
    """The 'Max events' box: a whole number from 1,000 up to
    ``view3d.MAX_EVENTS`` ('150k' allowed); None for text that is not one."""
    t = str(text).strip().lower().replace(',', '').replace('_', '')
    mult = 1
    if t.endswith('k'):
        mult, t = 1000, t[:-1]
    try:
        n = int(float(t) * mult)
    except ValueError:
        return None
    if n < 1000 or n > v3.MAX_EVENTS:
        return None
    return n


def parse_seed(text) -> int | None:
    """The 'Seed' box: a whole number of 0 or more, else None."""
    try:
        n = int(str(text).strip())
    except ValueError:
        return None
    return n if n >= 0 else None


class View3DWindow(ToolWindow):
    """Turnable 3D scatter of one sample's population."""

    def __init__(self, editor):
        super().__init__(editor, "3D plot", geometry="980x760", editor=editor)
        self.editor = editor
        self._view = v3.View()
        self._cloud: Cloud | None = None
        # Filled per sample by _fill_sample_dependents; empty until a sample
        # is picked (a window opened with nothing loaded).
        self._chan_map: dict[str, str] = {}
        self._gate_map: dict[str, str] = {}
        self._photo = None
        self._drag: tuple[int, int] | None = None
        self._redraw_id = None
        self.last_frame_ms = 0.0
        self._build()
        self._fill_sample_choices()
        self.after_idle(self.refresh)

    # ── layout ───────────────────────────────────────────────────────────

    def _build(self):
        self._build_data_row()
        self._build_color_row()
        self._build_export_row()
        self.canvas = tk.Canvas(self, highlightthickness=0, width=640, height=560)
        self.canvas.pack(fill='both', expand=True, padx=6, pady=4)
        self.status = ttk.Label(self, text="", anchor='w',
                                foreground=current_palette()['muted'])
        self.status.pack(fill='x', padx=6, pady=(0, 6))
        self.busy = BusyStrip(self, padding=(6, 0))
        self.busy.pack(fill='x')
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', lambda _e: setattr(self, '_drag', None))
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Button-4>', lambda _e: self.zoom_by(v3.ZOOM_STEP))
        self.canvas.bind('<Button-5>', lambda _e: self.zoom_by(1 / v3.ZOOM_STEP))
        self.canvas.bind('<Configure>', lambda _e: self.schedule_redraw())

    def _combo(self, parent, label, var, width, tip, values=()):
        ttk.Label(parent, text=label).pack(side='left')
        cb = ttk.Combobox(parent, textvariable=var, width=width,
                          state='readonly', values=list(values))
        cb.pack(side='left', padx=(2, 8))
        cb.bind('<<ComboboxSelected>>', lambda _e: self.refresh())
        auto_tip(cb, tip)
        return cb

    def _build_data_row(self):
        row = ttk.Frame(self, padding=(6, 6, 6, 0))
        row.pack(fill='x')
        self.sample_var = tk.StringVar()
        self.sample_combo = self._combo(
            row, "Sample:", self.sample_var, 18, "The sample to draw.")
        self.sample_combo.bind('<<ComboboxSelected>>', self._on_sample)
        self.pop_var = tk.StringVar(value=AS_SHOWN)
        self.pop_combo = self._combo(
            row, "Population:", self.pop_var, 20,
            "Which events to draw: what the main plot shows, every event, "
            "or one gate's population.")
        self.axis_vars = [tk.StringVar() for _ in range(3)]
        self.axis_combos = [
            self._combo(row, f"{a}:", var, 14,
                        f"Channel on the {a} axis. Pick UMAP1/2/3 (or another "
                        "3D embedding) to see an embedding in 3D. The axis "
                        "uses the same scale as the 2D plot.")
            for a, var in zip('XYZ', self.axis_vars, strict=True)]

    def _build_color_row(self):
        row = ttk.Frame(self, padding=(6, 4, 6, 0))
        row.pack(fill='x')
        self.color_mode_var = tk.StringVar(value=COLOR_LABELS['density'])
        self.color_mode_combo = self._combo(
            row, "Color by:", self.color_mode_var, 9,
            "Density: how crowded each point's neighborhood is in 3D. "
            "Channel: a marker's value. Cluster: a clustering result. "
            "Gate: inside one gate, in its color, the rest gray.",
            values=list(COLOR_LABELS.values()))
        self.color_mode_combo.bind('<<ComboboxSelected>>', self._on_color_mode)
        self.color_target_var = tk.StringVar()
        self.color_target_combo = self._combo(
            row, "", self.color_target_var, 16,
            "The channel, cluster column or gate to color by.")
        self.size_var = tk.StringVar(value=str(v3.DEFAULT_POINT_SIZE))
        self.size_combo = self._combo(
            row, "Point size:", self.size_var, 3,
            "Pixels per point. Bigger points read better in a small GIF.",
            values=[str(s) for s in v3.POINT_SIZES])
        self.size_combo.bind('<<ComboboxSelected>>', lambda _e: self._recolor_only())
        ttk.Label(row, text="Max events:").pack(side='left')
        self.cap_var = tk.StringVar(value=f"{v3.MAX_EVENTS:,}")
        cap = self.cap_entry = ttk.Entry(row, textvariable=self.cap_var, width=9)
        cap.pack(side='left', padx=(2, 8))
        auto_tip(cap, f"Draw at most this many events, a random subset "
                      f"(1,000 to {v3.MAX_EVENTS:,}). Fewer turns faster.")
        ttk.Label(row, text="Seed:").pack(side='left')
        self.seed_var = tk.StringVar(value=str(v3.DEFAULT_SEED))
        seed = ttk.Entry(row, textvariable=self.seed_var, width=6)
        seed.pack(side='left', padx=(2, 8))
        auto_tip(seed, "Seed of the random subset: the same seed always "
                       "draws the same events.")
        for w in (cap, seed):
            w.bind('<Return>', lambda _e: self.refresh())
            w.bind('<FocusOut>', lambda _e: self.refresh())

    def _build_export_row(self):
        row = ttk.Frame(self, padding=(6, 4, 6, 0))
        row.pack(fill='x')
        for text, cmd, tip in (
                ("Reset view", self.reset_view,
                 "Back to the angled view the window opened with."),
                ("Front (X/Y)", self.front_view,
                 "Look straight down Z: the 2D plot of X against Y."),
                ("Save PNG…", self.save_png, "Save this view as a PNG, at the window's size up to 1,280 pixels on the long side.")):
            b = ttk.Button(row, text=text, command=cmd)
            b.pack(side='left', padx=(0, 4))
            auto_tip(b, tip)
        ttk.Separator(row, orient='vertical').pack(side='left', fill='y', padx=6)
        self.frames_var = tk.StringVar(value=str(v3.DEFAULT_FRAMES))
        self._combo(row, "Turntable frames:", self.frames_var, 4,
                    "Frames in one full turn. 72 turns more smoothly and "
                    "makes a file about twice as big.",
                    values=[str(n) for n in v3.TURNTABLE_FRAMES])
        self.gif_btn = ttk.Button(row, text="Save GIF…", command=self.save_gif)
        self.gif_btn.pack(side='left', padx=(0, 4))
        auto_tip(self.gif_btn, "Save one full turn as a looping GIF, from "
                               "the current tilt and zoom, at the window's "
                               f"size up to {v3.MAX_EXPORT_SIDE} pixels on "
                               "the long side.")
        self.mp4_btn = ttk.Button(row, text="Save MP4…", command=self.save_mp4)
        self.mp4_btn.pack(side='left')
        self._encoder = v3.mp4_encoder()
        if self._encoder is None:
            self.mp4_btn.state(['disabled'])
            auto_tip(self.mp4_btn, "MP4 needs ffmpeg, which was not found. "
                                   "Install ffmpeg (on the PATH) or the "
                                   "imageio-ffmpeg package, then reopen.")
        else:
            auto_tip(self.mp4_btn, "Save one full turn as an MP4 video, at "
                               f"the window's size up to {v3.MAX_EXPORT_SIDE} "
                               "pixels on the long side.")

    # ── choices ──────────────────────────────────────────────────────────

    def _sample(self):
        return self.editor._samples.get(self.sample_var.get())

    def _fill_sample_choices(self):
        ed = self.editor
        names = [n for n in ed._target_samples('all') if n in ed._samples]
        self.sample_combo['values'] = names
        active = ed._active_sample
        self.sample_var.set(active if active in names else (names[0] if names else ''))
        self._fill_sample_dependents()

    def _on_sample(self, _event=None):
        self._fill_sample_dependents()
        self.refresh()

    def _channel_options(self, s):
        """``{display name: column}`` of the sample's numeric, non-label
        columns, in the editor's channel order first."""
        ed = self.editor
        cols = [c for c in s.data.columns if s.data[c].dtype.kind in 'fiu'
                and not ed._is_label_column(c, s.data[c].to_numpy()[:0])]
        order = [c for c in ed._channels if c in cols] + [
            c for c in cols if c not in ed._channels]
        return {ed._fmt_channel(c): c for c in order}

    def _fill_sample_dependents(self):
        s = self._sample()
        if s is None:
            return
        ed = self.editor
        self._chan_map = self._channel_options(s)
        disp = list(self._chan_map)
        for cb in self.axis_combos:
            cb['values'] = disp
        for var, ch in zip(self.axis_vars, self._default_axes(s), strict=True):
            var.set(ed._fmt_channel(ch) if ch else '')
        gates = ed._sample_gates.get(self.sample_var.get(), {}) or {}
        self._gate_map = {f"{g.get('name') or gid}": gid for gid, g in gates.items()}
        self.pop_combo['values'] = [AS_SHOWN, ALL_EVENTS] + list(self._gate_map)
        sel = ed._selected_gate_id()
        self.pop_var.set(next((k for k, g in self._gate_map.items() if g == sel),
                              AS_SHOWN))
        self._fill_color_targets()

    def _default_axes(self, s):
        """An embedding's three components when the sample has one, else the
        main plot's X and Y and the next channel."""
        trio = embedding_triplet(s.data.columns)
        if trio:
            return trio
        chans = list(self._chan_map.values())
        ed = self.editor
        from .ui_logic import resolve_channel
        picked = [c for c in (resolve_channel(ed.x_combo.get()),
                              resolve_channel(ed.y_combo.get())) if c in chans]
        for c in chans:
            if len(picked) >= 3:
                break
            if c not in picked:
                picked.append(c)
        return (picked + ['', '', ''])[:3]

    def _label_columns(self, s):
        ed = self.editor
        return [c for c in s.data.columns
                if ed._is_label_column(c, s.data[c].to_numpy()[:0])]

    def _color_mode(self):
        lab = self.color_mode_var.get()
        return next((k for k, v in COLOR_LABELS.items() if v == lab), 'density')

    def _on_color_mode(self, _event=None):
        self._fill_color_targets()
        self._recolor_only()

    def _fill_color_targets(self):
        s = self._sample()
        mode = self._color_mode()
        if s is None or mode == 'density':
            values = []
        elif mode == 'marker':
            values = list(self._chan_map)
        elif mode == 'cluster':
            values = self._label_columns(s)
        else:
            values = list(self._gate_map)
        self.color_target_combo['values'] = values
        if self.color_target_var.get() not in values:
            self.color_target_var.set(values[0] if values else '')
        self.color_target_combo.state(['!disabled'] if values else ['disabled'])

    def show(self, name, axes, color_column=None):
        """Show sample ``name`` on ``axes`` (three column names), colored
        by cluster column ``color_column`` when given; refreshes."""
        if name in self.editor._samples:
            self.sample_var.set(name)
            self._fill_sample_dependents()
        ed = self.editor
        for var, ch in zip(self.axis_vars, axes, strict=True):
            var.set(ed._fmt_channel(ch))
        if color_column and color_column in self._label_columns(self._sample()):
            self.color_mode_var.set(COLOR_LABELS['cluster'])
            self._fill_color_targets()
            self.color_target_var.set(color_column)
        self.refresh()

    def columns_changed(self):
        """The samples' columns changed (a run added or removed some, such
        as UMAP3): keep the sample, axes and population that still exist,
        pick the rest again, and reload, so no stale picture stays up."""
        if not self.winfo_exists():
            return
        ed = self.editor
        s = self._sample()
        if s is None:
            self._fill_sample_choices()
            self.refresh()
            return
        axes, pop = self._axes(), self.pop_var.get()
        self._fill_sample_dependents()
        if all(c and c in s.data.columns for c in axes):
            for var, ch in zip(self.axis_vars, axes, strict=True):
                var.set(ed._fmt_channel(ch))
        if pop in tuple(self.pop_combo['values']):
            self.pop_var.set(pop)
        self.refresh()

    # ── data ─────────────────────────────────────────────────────────────

    def _axes(self):
        return [self._chan_map.get(v.get(), '') for v in self.axis_vars]

    def _gate_parent(self):
        p = self.pop_var.get()
        if p == ALL_EVENTS:
            return None
        return self._gate_map.get(p, 'auto')

    def _settings(self):
        """``(cap, seed)`` from the boxes; an invalid entry is put back to
        the last good value."""
        cap = parse_cap(self.cap_var.get())
        seed = parse_seed(self.seed_var.get())
        if cap is None:
            cap = getattr(self, '_good_cap', v3.MAX_EVENTS)
            self.cap_var.set(f"{cap:,}")
        if seed is None:
            seed = getattr(self, '_good_seed', v3.DEFAULT_SEED)
            self.seed_var.set(str(seed))
        self._good_cap, self._good_seed = cap, seed
        return cap, seed

    def _screen_fn(self, channel, data_sample):
        """Stored -> screen for ``channel``: the scale ``_apply_axis_to_ax``
        gives that channel on the 2D plot."""
        ed = self.editor
        funcs = ed._axis_view_funcs(channel, data_sample)
        scale = ed._scale_of(channel)
        lt = (ed._symlog_linthresh(data_sample)
              if funcs is None and scale == 'symlog' else None)
        return v3.screen_transform(scale, funcs=funcs, linthresh=lt)

    def axis_screen(self, channel, values):
        """``(screen values, (lo, hi) bounds)`` of ``values`` on ``channel``'s
        axis. The scale is anchored on the sample's whole column, as the 2D
        plot anchors it on the first plotted sample's."""
        s = self._sample()
        anchor = s.data[channel].to_numpy() if channel in s.data.columns else None
        fn = self._screen_fn(channel, anchor)
        screen = fn(values)
        rng = self.editor._channel_range.get(channel)
        explicit = None
        if rng is not None:
            # A non-finite end (0 on a log axis) is autoscaled there, as
            # the 2D plot ignores it and keeps the other.
            ends = fn(np.asarray(rng, dtype=float))
            explicit = (float(ends[0]), float(ends[1]))
        return screen, v3.axis_bounds(screen, explicit)

    def _rows(self, name, cols, cap, seed):
        """``(positions, total)``: the population's rows with a value on every
        axis, then the seeded subset of at most ``cap`` of them."""
        ed = self.editor
        n = len(ed._samples[name].data)
        spec = ed._plotcache_popspec(name, self._gate_parent())
        packed, _count = ed._plotcache_eligible(name, cols, spec)
        pos = (np.arange(n) if packed is None
               else np.flatnonzero(_pcm.unpack(packed, n)))
        pick = v3.seeded_subset(len(pos), cap, seed)
        return (pos if pick is None else pos[pick]), len(pos)

    def load_cloud(self) -> Cloud | None:
        """Gather the events and put them in cube space; None (with the
        reason on the status line) when there is nothing to draw."""
        name = self.sample_var.get()
        s = self._sample()
        axes = self._axes()
        if s is not None and any(c not in s.data.columns for c in axes if c):
            # A column the boxes name is gone: choose the axes again.
            self._fill_sample_dependents()
            axes = self._axes()
        if s is None or not all(axes):
            self._say("Pick a sample and three channels.")
            return None
        cap, seed = self._settings()
        ed = self.editor
        alias = ed._axis_alias_for_sample(s, axes)
        cols = ed._plotcache_cols(s, axes, alias)
        rows, total = self._rows(name, cols, cap, seed)
        df = ed._plotcache_frame(s, rows, alias)
        norm = []
        for ch in axes:
            screen, bounds = self.axis_screen(ch, df[ch].to_numpy())
            norm.append(v3.normalize(screen, bounds))
        pts = np.column_stack(norm) if len(df) else np.zeros((0, 3))
        keep = np.all(np.isfinite(pts) & (np.abs(pts) <= 1.0 + 1e-9), axis=1)
        self._frame, self._frame_rows = df.iloc[keep], rows[keep]
        pts = pts[keep]
        return Cloud(points=pts, colors=self._colors(pts), total=total,
                     shown=len(df), off_axis=int((~keep).sum()), seed=seed)

    def _colors(self, pts):
        mode = self._color_mode()
        target = self.color_target_var.get()
        df = self._frame
        ed = self.editor
        if mode == 'marker' and self._chan_map.get(target) in df.columns:
            return v3.marker_colors(df[self._chan_map[target]].to_numpy())
        if mode == 'cluster' and target in df.columns:
            vals = df[target].to_numpy()
            pal = [(v, c) for v, c, _ in ed._label_colours(
                target, vals, [self.sample_var.get()])]
            return v3.category_colors(vals, pal)
        if mode == 'gate' and target in self._gate_map:
            gid = self._gate_map[target]
            name = self.sample_var.get()
            inside = ed._plotcache_mask(name, gid)[self._frame_rows]
            g = (ed._sample_gates.get(name) or {}).get(gid) or {}
            return v3.gate_colors(inside, g.get('color') or current_palette()['accent'])
        return v3.density_colors(pts, density_cmap())

    # ── drawing ──────────────────────────────────────────────────────────

    def refresh(self, _event=None):
        """Reload the events (channels, population, cap or seed changed)."""
        if not self.winfo_exists():
            return
        self._cloud = self.load_cloud()
        self.redraw()

    def _recolor_only(self):
        if self._cloud is None:
            self.refresh()
            return
        self._cloud.colors = self._colors(self._cloud.points)
        self.redraw()

    def _look(self):
        """``(background, ink)`` of the plot background the editor uses."""
        var = getattr(self.editor, '_plot_bg_var', None)
        dark = var is not None and var.get() == 'dark'
        pal = THEMES[THEME_ALL_DARK if dark else THEME_LIGHT]
        return v3.hex_rgb(pal['plot_bg']), v3.hex_rgb(pal['plot_fg'])

    def scene(self) -> v3.Scene | None:
        """What the next picture shows (points, colors, axis names, point
        size, plot colors), or None when nothing is loaded."""
        c = self._cloud
        if c is None:
            return None
        bg, ink = self._look()
        labels = tuple(self.editor._fmt_channel(ch) for ch in self._axes())
        try:
            size = int(self.size_var.get())
        except ValueError:
            size = v3.DEFAULT_POINT_SIZE
        return v3.Scene(points=c.points, colors=c.colors, labels=labels,
                        point_size=size, background=bg, ink=ink)

    def schedule_redraw(self):
        """Redraw once the event loop is idle; repeated calls draw once."""
        if self._redraw_id is None:
            self._redraw_id = self.after_idle(self.redraw)

    def redraw(self):
        """Draw the scene from the current view into the canvas and time it."""
        self._redraw_id = None
        sc = self.scene()
        if sc is None:
            self.canvas.delete('all')
            return
        from PIL import ImageTk
        w = max(50, self.canvas.winfo_width())
        h = max(50, self.canvas.winfo_height())
        t0 = time.perf_counter()
        img = v3.render(sc, self._view, w, h)
        self.last_frame_ms = (time.perf_counter() - t0) * 1000.0
        self._photo = ImageTk.PhotoImage(img, master=self)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self._photo)
        self._say(self._status_text())

    def _status_text(self):
        c = self._cloud
        assert c is not None
        bits = [v3.subset_note(c.shown, c.total, c.seed)]
        if c.off_axis:
            bits.append(f"{c.off_axis:,} off the axes (outside a set range "
                        f"or not placeable on a log axis)")
        bits.append(f"azimuth {self._view.azimuth:.0f}°, elevation "
                    f"{self._view.elevation:.0f}°, frame {self.last_frame_ms:.0f} ms")
        return " · ".join(bits)

    def _say(self, text):
        self.status.configure(text=text)

    # ── view ─────────────────────────────────────────────────────────────

    @property
    def view(self) -> v3.View:
        """The current camera: azimuth, elevation and zoom."""
        return self._view

    def set_view(self, view: v3.View):
        """Look from ``view`` and redraw."""
        self._view = view
        self.schedule_redraw()

    def reset_view(self):
        """Back to the angled view the window opened with."""
        self.set_view(v3.View())

    def front_view(self):
        """Look straight down Z: the 2D plot of X against Y."""
        self.set_view(v3.FRONT_VIEW)

    def zoom_by(self, factor):
        """Zoom by ``factor`` (above 1 is closer), within the zoom limits."""
        self.set_view(self._view.zoomed(factor))

    def _on_press(self, e):
        self._drag = (e.x, e.y)

    def _on_drag(self, e):
        if self._drag is None:
            self._drag = (e.x, e.y)
            return
        dx, dy = e.x - self._drag[0], e.y - self._drag[1]
        self._drag = (e.x, e.y)
        # Half a degree per pixel; dragging up tilts the top away.
        self.set_view(self._view.turned(d_azimuth=dx * 0.5, d_elevation=dy * 0.5))

    def _on_wheel(self, e):
        units = v3.wheel_units(str(self.tk.call('tk', 'windowingsystem')),
                               str(self.tk.call('info', 'tclversion')))
        self.zoom_by(v3.wheel_zoom(getattr(e, 'delta', 0), units))

    # ── export ───────────────────────────────────────────────────────────

    def _ask_path(self, ext, kind):
        return filedialog.asksaveasfilename(
            parent=self, defaultextension=ext,
            filetypes=[(kind, f'*{ext}'), ('All files', '*.*')])

    def _export_size(self):
        """The window's drawing size, long side capped at
        ``view3d.MAX_EXPORT_SIDE`` (``view3d.export_size``)."""
        w = max(50, self.canvas.winfo_width())
        h = max(50, self.canvas.winfo_height())
        return v3.export_size(w, h)

    def save_png(self, path=None):
        """Save the current view as a PNG at the canvas size, at most 1,280
        pixels on the long side (asking where
        when ``path`` is None); returns the path, or None."""
        sc = self.scene()
        if sc is None:
            return None
        path = path or self._ask_path('.png', 'PNG image')
        if not path:
            return None
        w, h = self._export_size()
        image = v3.render(sc, self._view, w, h)
        with v3.written_whole(path) as part:
            image.save(part, format='PNG')
        self._say(f"Saved {path}")
        return path

    def _frames_choice(self):
        try:
            n = int(self.frames_var.get())
        except ValueError:
            n = v3.DEFAULT_FRAMES
        return n if n in v3.TURNTABLE_FRAMES else v3.DEFAULT_FRAMES

    def save_gif(self, path=None):
        """Save one full turn as a looping GIF; returns the job or None."""
        return self._save_turntable(path, '.gif', 'GIF animation', v3.save_gif)

    def save_mp4(self, path=None):
        """Save one full turn as an MP4 with ffmpeg; returns the job or None."""
        exe = self._encoder
        if exe is None:
            ui_errors.showinfo("3D plot", "MP4 needs ffmpeg, which was not "
                               "found.", parent=self)
            return None
        return self._save_turntable(
            path, '.mp4', 'MP4 video',
            lambda frames, p, n: v3.save_mp4(frames, p, exe, n))

    def _save_turntable(self, path, ext, kind, writer):
        """Render one full turn off the Tk thread (with progress and Cancel)
        and write it with ``writer(frames, path, n_frames)``; returns the
        job, or None when nothing was started."""
        sc = self.scene()
        if sc is None or self.busy.running:
            return None
        path = path or self._ask_path(ext, kind)
        if not path:
            return None
        n = self._frames_choice()
        w, h = self._export_size()
        view = self._view

        def work(job):
            def progress(i, total):
                # Also a cancel point: Cancel stops the turn here.
                job.progress(i, total, f"Frame {i} of {total}")
            # Each frame is rendered as the writer takes it and then
            # dropped; Cancel stops the turn and leaves no file.
            writer(v3.turntable_frames(sc, view, n, w, h, progress=progress),
                   path, n)
            return path

        buttons = [self.gif_btn] + ([self.mp4_btn] if self._encoder else [])
        return self.busy.run(work, on_done=self._saved,
                             on_error=self._save_failed,
                             on_cancel=lambda: self._say("Canceled; nothing was saved."),
                             text=f"Rendering {n} frames…", disable=buttons)

    def _saved(self, path):
        self._say(f"Saved {path}")

    def _save_failed(self, exc):
        self._say("Could not save the turntable.")
        ui_errors.showerror("3D plot", f"Could not save the turntable:\n{exc}",
                            parent=self)


def open_view3d(editor: Any):
    """Analyze > 3D plot…: the window, once per editor."""
    if not editor._samples:
        editor.status_var.set("Load a sample first to draw it in 3D.")
        return None
    return editor._show_or_raise('view3d', lambda: View3DWindow(editor))
