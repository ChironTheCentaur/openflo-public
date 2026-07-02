"""Plot rendering, axis scaling, density scatter, backgate overlays, and the
view controls (reset/zoom/swap) — editor mixin.

The matplotlib-canvas rendering half of ViewGateEditorWindow. Methods call
gating/model helpers on ``self``; see editor_base.EditorMixin.
"""
from __future__ import annotations

from tkinter import ttk

import numpy as np

from .editor_base import EditorMixin
from .theme import current_palette


class PlotMixin(EditorMixin):
    """Cloud / density / histogram rendering, axis display scales, backgate +
    highlight overlays, the legend, and reset/zoom/swap view controls."""

    def _on_axis_channel_change(self):
        """An axis channel was committed via the type-to-filter picker."""
        self._schedule_replot(0)

    def _selected_samples(self):
        """Samples currently checked for plot inclusion (☑ in the tree),
        in load order."""
        return self._target_samples('enabled')

    def _axis_alias_for_sample(self, s, dets):
        """Label-first axis resolution (#48).

        A chosen axis is a detector from the global panel (the first
        sample's columns). When THIS sample lacks that exact detector but
        carries the same antibody label on a different fluor, expose its
        own detector under the chosen name so the plot overlays on a
        common label axis instead of dropping the sample. Detectors the
        sample already has, and non-fluor axes (FSC-A/SSC-A), are left
        untouched.

        Returns {chosen_detector: own_detector} — an *alias* map. We alias
        (copy) rather than rename so the sample's own detector column stays
        present for gate masks, which read each sample's retargeted
        detectors (see #47).
        """
        from .pipeline import _sample_fluor_labels
        cols = set(s.data.columns)
        l2d = _sample_fluor_labels(s)        # {label: this sample's detector}
        alias = {}
        for det in dets:
            if not det or det in cols:
                continue                     # sample already carries it
            label = self._channel_labels.get(det, det)
            own = l2d.get(label)
            if own and own in cols:
                alias[det] = own
        return alias

    def _get_df(self, name, x, y=None, for_hist=False, downsample=True):
        s  = self._samples[name]
        df = s.data
        alias = self._axis_alias_for_sample(s, [x, y])
        if alias:
            # Add chosen-name columns from this sample's own detectors;
            # copy leaves s.data and the original detector columns intact.
            df = df.assign(**{chosen: df[own] for chosen, own in alias.items()})
        cols = [c for c in (x, y) if c]
        df = df.dropna(subset=[c for c in cols if c in df.columns])

        # Each sample applies its OWN gate tree.
        # Filter mode keeps events that are inside the cumulative chain of
        # ANY enabled gate (union of populations). So with `P enabled` and
        # `C enabled` you see events in P OR events in C, not the empty
        # intersection of two disjoint forks. With just `C enabled` you see
        # events in (root...P AND C) — ancestors always filter, regardless
        # of their toggle (the toggle is visibility, not chain membership).
        sample_gates = self._sample_gates.get(name, {})
        if self.apply_gates_var.get() and sample_gates:
            from .pipeline import cumulative_gate_mask
            overrides = self._autoclean_overrides(name, df)
            mask = np.zeros(len(df), dtype=bool)
            any_enabled = False
            for gid, g in sample_gates.items():
                if g.get('enabled', True):
                    mask |= cumulative_gate_mask(sample_gates, gid, df,
                                                 overrides=overrides)
                    any_enabled = True
            if any_enabled:
                df = df[mask]

        # Display-only auto-downsample: when enabled, every plotted
        # sample renders the same number of events as the smallest
        # loaded sample. Underlying FlowSample.data is untouched.
        #
        # The 60k ceiling is a *scatter-rendering* guard (drawing 200k points
        # is slow); a histogram bins cheaply, so for_hist skips that ceiling
        # and keeps raw counts truthful — while STILL honouring the
        # downsample-to-smallest toggle so overlaid counts stay comparable.
        # Max points only caps while downsampling is enabled (Display or
        # Display+data). With downsampling Off, draw every event (uncapped).
        _dv = getattr(self, 'ds_display_var', None)
        _pv = getattr(self, 'ds_propagate_var', None)
        ds_on = ((_dv is not None and _dv.get())
                 or (_pv is not None and _pv.get()))
        cap = (self._display_point_cap()
               if (downsample and not for_hist and ds_on) else None)
        if (downsample and _dv is not None and _dv.get()):
            floor = self._smallest_loaded_sample_size()
            if floor is not None and floor > 0:
                cap = floor if cap is None else min(cap, floor)
        if cap is not None and len(df) > cap:
            # Seed by name + cap so the same subsample is picked across
            # replots — keeps the plot stable while the user pans gates.
            seed = (hash((name, x, y, cap)) & 0xFFFF_FFFF)
            df = df.sample(cap, random_state=seed)
        return df

    def _display_point_cap(self):
        """Max events drawn per sample in scatter / pseudocolor / contour
        modes, from the 'Max points' control. 'All' (or blank / 0) removes the
        cap. Accepts plain integers or '250k'-style shorthand. Defaults to
        60 000 so large samples stay responsive; histograms ignore this."""
        v = getattr(self, 'max_points_var', None)
        if v is None:
            return 60_000
        raw = str(v.get()).strip().lower().replace(',', '')
        if raw in ('', 'all', '0', 'none'):
            return 1 << 62                      # effectively uncapped
        try:
            if raw.endswith('k'):
                return max(1000, int(float(raw[:-1]) * 1000))
            if raw.endswith('m'):
                return max(1000, int(float(raw[:-1]) * 1_000_000))
            return max(1000, int(float(raw)))
        except ValueError:
            return 60_000

    def _sample_display_count(self, name):
        """``(shown, total)`` events for ``name``: the full FlowSample size and
        how many are actually drawn after the display caps (the 60k scatter
        guard and the auto-downsample-to-smallest toggle). ``shown == total``
        when nothing is scaled down."""
        s = self._samples.get(name)
        data = getattr(s, 'data', None) if s is not None else None
        if data is None:
            return (0, 0)
        total = len(data)
        # Max points only caps while downsampling is on; Off → shown == total.
        _dv = getattr(self, 'ds_display_var', None)
        _pv = getattr(self, 'ds_propagate_var', None)
        ds_on = ((_dv is not None and _dv.get())
                 or (_pv is not None and _pv.get()))
        if not ds_on:
            return (total, total)
        cap = self._display_point_cap()
        if _dv is not None and _dv.get():
            floor = self._smallest_loaded_sample_size()
            if floor is not None and floor > 0:
                cap = min(cap, floor)
        return (min(total, cap), total)

    def _schedule_replot(self, delay_ms=100):
        if self._replot_after_id:
            try:
                self.after_cancel(self._replot_after_id)
            except Exception:
                pass
        self._replot_after_id = self.after(delay_ms, self._replot)

    def _render_placeholder(self):
        self.ax.clear()
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self._apply_plot_theme()
        self.canvas.draw_idle()
        self._show_empty_overlay()

    def _show_empty_overlay(self):
        """First-run / no-samples state: a few clickable starting points over
        the empty canvas, instead of a bare grey label."""
        host = getattr(self, '_plot_host', None)
        # Only for the genuinely-empty state. If samples exist but none are
        # checked, the bare placeholder is enough — don't pop start buttons.
        if host is None or self._samples:
            self._hide_empty_overlay()
            return
        ov = getattr(self, '_empty_overlay', None)
        if ov is None:
            ov = ttk.Frame(host, padding=24)
            ttk.Label(ov, text="No samples loaded",
                      font=('TkDefaultFont', 13, 'bold')).pack(pady=(0, 2))
            ttk.Label(ov, text="Get started:",
                      foreground='grey').pack(pady=(0, 12))
            for text, cmd in (("➕  Add FCS files…", self._add_samples),
                              ("🧪  Load example dataset", self._load_example_data),
                              ("📂  Open session…", self._load_session)):
                ttk.Button(ov, text=text, width=26,
                           command=cmd).pack(pady=3)
            ttk.Label(ov, text="…or drag & drop FCS files anywhere.",
                      foreground='grey').pack(pady=(12, 0))
            ttk.Label(ov, text="Resuming a larger session may take a moment "
                      "to load.", foreground='grey').pack(pady=(2, 0))
            self._empty_overlay = ov
        ov.place(relx=0.5, rely=0.5, anchor='center')
        ov.lift()

    def _hide_empty_overlay(self):
        ov = getattr(self, '_empty_overlay', None)
        if ov is not None:
            try:
                ov.place_forget()
            except Exception:
                pass

    def _apply_plot_theme(self):
        """Colour the interactive matplotlib canvas to the active theme.
        Light & dark chrome keep a white plot; 'midnight' darkens figure,
        axes, ticks, labels, spines, grid, the main legend and the backgate
        legend box. Call AFTER drawing (ax.clear resets the facecolor).
        Exports build their own figures and stay white — untouched here."""
        pal = current_palette()
        bg = pal.get('plot_bg', '#ffffff')
        fg = pal.get('plot_fg', '#20242b')
        spine = pal.get('plot_spine', '#b9bdc6')
        grid = pal.get('plot_grid', '#e6e8ec')
        try:
            self.fig.set_facecolor(bg)
            for ax in self.fig.axes:          # includes a colorbar axis if any
                ax.set_facecolor(bg)
                ax.tick_params(colors=fg, which='both')
                for s in ax.spines.values():
                    s.set_color(spine)
                ax.xaxis.label.set_color(fg)
                ax.yaxis.label.set_color(fg)
                ax.title.set_color(fg)
                for gl in ax.get_xgridlines() + ax.get_ygridlines():
                    gl.set_color(grid)
                leg = ax.get_legend()
                if leg is not None:
                    fr = leg.get_frame()
                    fr.set_facecolor(bg)
                    fr.set_edgecolor(spine)
                    for t in leg.get_texts():
                        t.set_color(fg)
                    if leg.get_title() is not None:
                        leg.get_title().set_color(fg)
        except Exception:
            pass

    def _replot(self):
        self._replot_after_id = None
        self._hide_empty_overlay()
        self._sync_display_mode_availability()
        # Remove any prior colorbar
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self.ax.clear()
        # Forget previously drawn gate Line2D objects (they were on the
        # old axes that we just cleared).
        self._vlines = {}
        self._hlines = {}

        samples = self._selected_samples()
        if not self._samples:
            self._render_placeholder()
            return
        if not samples:
            self.ax.text(0.5, 0.5, 'Select one or more samples on the left',
                         ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=11, color='grey')
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self._apply_plot_theme()
            self.canvas.draw_idle()
            return

        mode  = self.mode_var.get()
        x     = self._resolve_channel(self.x_combo.get())
        y     = self._resolve_channel(self.y_combo.get())
        color = self.color_combo.get()

        if not x:
            self._apply_plot_theme()
            self.canvas.draw_idle()
            return

        try:
            if mode == 'histogram':
                self._plot_histogram(samples, x)
            elif mode == 'dot':
                self._plot_dot(samples, x, y, color)
            elif mode == 'pseudocolor':
                self._plot_pseudocolor(samples, x, y)
            elif mode == 'contour':
                self._plot_contour(samples, x, y)
        except Exception as exc:
            self.ax.text(0.5, 0.5, f'Plot error:\n{exc}',
                         ha='center', va='center',
                         transform=self.ax.transAxes, fontsize=10, color='red')

        # Overlay auto-clean-removed events (red, on top) when requested.
        try:
            self._overlay_removed_events(samples, x, y, mode)
        except Exception as exc:
            print(f"[cleaned-out overlay] {type(exc).__name__}: {exc}",
                  flush=True)
        # Backgating: project selected populations onto the current plot.
        try:
            self._overlay_backgate(samples, x, y)
        except Exception as exc:
            print(f"[backgate overlay] {type(exc).__name__}: {exc}",
                  flush=True)

        self.ax.set_xlabel(self._fmt_channel(x), fontsize=9)
        if mode != 'histogram' and y:
            self.ax.set_ylabel(self._fmt_channel(y), fontsize=9)

        # Apply per-channel scale + range. Sample data from the FIRST
        # plotted sample (when one exists) gives the symlog linthresh a
        # data-driven anchor. Done AFTER plotting so the underlying
        # density / scatter has been drawn into linear coords; the
        # scale change is purely a display transform.
        first = samples[0] if samples else None
        sample_data = None
        if mode == 'histogram' and getattr(self, '_hist_x_anchor', None) is not None:
            # Anchor the symlog cofactor on the SAME array the histogram binned
            # on, so the screen-uniform bins and the axis transform agree.
            sample_data = self._hist_x_anchor
        elif first and x and first in self._samples:
            sdf = self._samples[first].data
            if x in sdf.columns:
                sample_data = sdf[x].values
        self._apply_axis_to_ax(x, 'x', sample_data)
        if mode != 'histogram' and y:
            ydata = None
            if first and y and first in self._samples:
                sdf = self._samples[first].data
                if y in sdf.columns:
                    ydata = sdf[y].values
            self._apply_axis_to_ax(y, 'y', ydata)

        # Highlight overlays sit on top of the base population. No-op
        # unless the user has switched to 'Highlight gated'.
        self._draw_highlight_overlays(
            samples, x, y if mode != 'histogram' else None)

        # Draw gates (shapes + threshold/interval lines) on top of the
        # overlays so they remain visible.
        self._draw_gates(x, y if mode != 'histogram' else None)
        self._refresh_gate_list()

        self.fig.tight_layout()
        self._apply_plot_theme()
        self.canvas.draw_idle()

        # ax.clear() blew away any matplotlib Selector — reattach.
        self._activate_gate_tool()
        # Show/hide the histogram slider panel and resync ranges.
        self._sync_slider_panel()

    def _axis_view_funcs(self, channel, data_sample=None):
        """``(forward, inverse)`` callables mapping a channel's STORED data
        coordinate to screen position for its chosen display scale — or
        ``None`` when the channel's data is already linear (the caller then
        uses matplotlib's native linear/symlog/log scale, which has nicer
        tick locators).

        Fluor data is baked into a nonlinear transform space
        (``_channel_transform``, e.g. logicle). The underlying *linear
        intensity* is the canonical master; it is recovered with
        ``inverse_transform_values``. The chosen scale (linear / symlog /
        log) is then a pure VIEW of that intensity, composed as::

            forward(d) = view_forward(inverse_baked(d))
            inverse(p) = forward_baked(view_inverse(p))

        So every scale is an independent, equation-derived view of the same
        compensated intensity — no double-transform — and gates (kept in
        stored data coords) auto-follow the axis transform for free. symlog
        uses an arcsinh view whose cofactor is anchored on the data.
        """
        from .scales import view_funcs
        tm = self._channel_transform.get(channel, 'linear')
        scale = self._channel_scale.get(channel, self._default_channel_scale)
        return view_funcs(tm, scale, data_sample)

    def _symlog_edges(self, lo, hi, n_bins, data_sample):
        """Bin edges uniform in matplotlib's symlog SCREEN transform (linear
        within ``linthresh``, log beyond), matching the native symlog display
        axis. Without this the density uses linear bins, which are far too
        coarse in the log decade (boxy artefacts ~10^3–10^4)."""
        from matplotlib.scale import SymmetricalLogTransform
        lt = self._symlog_linthresh(data_sample)
        t = SymmetricalLogTransform(10, lt, 1)
        slo = float(np.asarray(t.transform(np.array([float(lo)]))).ravel()[0])
        shi = float(np.asarray(t.transform(np.array([float(hi)]))).ravel()[0])
        if not (np.isfinite(slo) and np.isfinite(shi)) or shi <= slo:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        screen = np.linspace(slo, shi, int(n_bins) + 1)
        edges = np.unique(
            np.asarray(t.inverted().transform(screen), dtype=float).ravel())
        edges = edges[np.isfinite(edges)]
        if edges.size < 2:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        return edges.tolist()

    def _screen_uniform_edges(self, channel, lo, hi, n_bins, data_sample=None):
        """``n_bins + 1`` bin edges between data-coords ``lo`` and ``hi``,
        spaced uniformly in SCREEN space for the channel's display scale, so
        density bins aren't banded on a log / symlog / composite axis."""
        funcs = self._axis_view_funcs(channel, data_sample) if channel else None
        if funcs is None:
            scale = (self._channel_scale.get(channel, self._default_scale_for(channel))
                     if channel else 'linear')
            if scale == 'symlog':
                return self._symlog_edges(lo, hi, n_bins, data_sample)
            return self._hist_bin_edges(lo, hi, scale, n_bins)
        fwd, inv = funcs
        slo = float(np.asarray(fwd(np.array([lo], dtype=float)))[0])
        shi = float(np.asarray(fwd(np.array([hi], dtype=float)))[0])
        if not (np.isfinite(slo) and np.isfinite(shi)) or shi <= slo:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        screen = np.linspace(slo, shi, int(n_bins) + 1)
        edges = np.unique(np.asarray(inv(screen), dtype=float))
        edges = edges[np.isfinite(edges)]
        if edges.size < 2:
            return self._hist_bin_edges(lo, hi, 'linear', n_bins)
        return edges.tolist()

    def _axis_bin_edges(self, vals, channel, n_bins):
        """Bin edges for `vals` in the channel's *display* space, over the
        effective view range (explicit per-channel range if set, else a
        robust 0.5–99.5 percentile). So density bins are visually uniform
        on log/symlog and track the zoom instead of the full data extent."""
        rng = self._channel_range.get(channel) if channel else None
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
        else:
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo, hi = (float(v) for v in np.percentile(finite, [0.5, 99.5]))
            else:
                lo, hi = 0.0, 1.0
        if hi <= lo:
            lo = float(np.min(vals)) if vals.size else 0.0
            hi = float(np.max(vals)) if vals.size else 1.0
            if hi <= lo:
                hi = lo + 1.0
        return np.asarray(
            self._screen_uniform_edges(channel, lo, hi, n_bins, data_sample=vals),
            dtype=float)

    def _density_scatter(self, xs, ys, xch=None, ych=None):
        """Density-coloured scatter.

        Two modes, controlled by the 'True Gaussian KDE' checkbox:
          • Off (default, FlowJo-style): O(n) 2D histogram + smoothing
            + per-event lookup. Handles tens of millions of events in
            sub-second on CPU.
          • On: scipy.stats.gaussian_kde — mathematically smoother but
            O(n^2). Subsamples aggressively and warns the user.

        `xch`/`ych` (channel names) let the histogram bin in the axis's own
        space (log/symlog/linear) so density isn't banded on a log view.
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        finite = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[finite]; ys = ys[finite]
        if xs.size == 0:
            return

        true_kde = (hasattr(self, 'true_kde_var')
                    and self.true_kde_var.get())

        if true_kde:
            self._density_scatter_truekde(xs, ys)
        else:
            self._density_scatter_histogram(xs, ys, xch, ych)

    def _density_scatter_histogram(self, xs, ys, xch=None, ych=None):
        from .density import event_density
        BINS         = 256
        MAX_DISPLAY  = self._display_point_cap()
        try:
            # Bin in each axis's display space over the effective view
            # range — uniform linear bins would band on a log/symlog axis.
            x_edges = self._axis_bin_edges(xs, xch, BINS)
            y_edges = self._axis_bin_edges(ys, ych, BINS)
            # Smoothed per-event density (histogram2d → pad → gaussian → cubic
            # interpolation); see openflo.density.event_density.
            z = event_density(xs, ys, x_edges, y_edges)
            if xs.size > MAX_DISPLAY:
                rng = np.random.default_rng(42)
                sel = rng.choice(xs.size, MAX_DISPLAY, replace=False)
                xs_d, ys_d, z_d = xs[sel], ys[sel], z[sel]
            else:
                xs_d, ys_d, z_d = xs, ys, z
            order = z_d.argsort()
            self.ax.scatter(xs_d[order], ys_d[order], c=z_d[order],
                            cmap='jet', s=2, alpha=0.85, linewidths=0,
                            norm=self._density_norm(z_d), rasterized=True)
        except Exception as exc:
            print(f"[pseudocolor] density failed "
                  f"({type(exc).__name__}: {exc}); flat scatter fallback",
                  flush=True)
            self.ax.scatter(xs, ys, s=2, alpha=0.4,
                            color='steelblue', linewidths=0,
                            rasterized=True)

    def _density_scatter_truekde(self, xs, ys):
        """True scipy gaussian_kde path. The subsample + KDE maths lives in
        openflo.density.kde_density; here we post the trade-off status and draw
        (with a flat-scatter fallback)."""
        from .density import kde_density
        try:
            xs_d, ys_d, z, n_src = kde_density(xs, ys)
            if n_src < xs.size:
                try:
                    self.status_var.set(
                        f"True KDE on {n_src:,} source / {xs_d.size:,} "
                        f"display events (subsampled from {xs.size:,}).")
                except Exception:
                    pass
            order = z.argsort()
            self.ax.scatter(xs_d[order], ys_d[order], c=z[order],
                            cmap='jet', s=2, alpha=0.7, linewidths=0,
                            norm=self._density_norm(z), rasterized=True)
        except Exception as exc:
            print(f"[pseudocolor/KDE] failed "
                  f"({type(exc).__name__}: {exc}); flat scatter fallback",
                  flush=True)
            self.ax.scatter(xs, ys, s=2, alpha=0.4,
                            color='steelblue', linewidths=0,
                            rasterized=True)

    def _plot_contour(self, samples, x, y):
        """Smoothed density contours with outlier scatter underneath.

        Same O(n) histogram-based density as the pseudocolor path —
        gaussian_kde is overkill for flow data and doesn't scale.

        Each sample contributes:
          • A 128×128 2D histogram density, smoothed with gaussian_filter,
            rendered as 8 contour levels from 5% → 95% of peak so the
            outer line traces the population edge, not just the dense core.
          • A faint per-event scatter beneath the lines (master 'Contour
            scatter' toggle). Events below the lowest contour level are
            'outliers'; the 'Outliers' sub-toggle gates just those, so the
            scatter can show the within-population points only.
        """
        if not y:
            return
        from scipy.ndimage import gaussian_filter

        GRID            = 128
        SMOOTH_SIGMA    = 1.5
        LEVELS_FROM     = 0.05
        LEVELS_TO       = 0.95
        N_LEVELS        = 8
        MAX_OUTLIER_PTS = 30_000

        rng = np.random.default_rng(42)

        for name in samples:
            df = self._get_df(name, x, y)
            if df.empty or x not in df.columns or y not in df.columns:
                continue
            xv = np.asarray(df[x].values, dtype=float)
            yv = np.asarray(df[y].values, dtype=float)
            finite = np.isfinite(xv) & np.isfinite(yv)
            xv = xv[finite]; yv = yv[finite]
            if xv.size < 10:
                print(f"[contour] {name}: only {xv.size} finite points — skipped",
                      flush=True)
                continue
            try:
                xmin, xmax = float(xv.min()), float(xv.max())
                ymin, ymax = float(yv.min()), float(yv.max())
                if xmin == xmax or ymin == ymax:
                    print(f"[contour] {name}: degenerate range — skipped",
                          flush=True)
                    continue

                # 2% padding so outliers don't sit on the axis edge.
                xpad = (xmax - xmin) * 0.02
                ypad = (ymax - ymin) * 0.02
                xmin -= xpad; xmax += xpad
                ymin -= ypad; ymax += ypad

                color = self._color_for(name)

                # 1) Histogram-based density on the FULL population
                #    (O(n) — no subsampling needed). Bin in each axis's
                #    display space so the grid (and the contour lines) are
                #    even on a log/symlog axis.
                x_edges = self._axis_bin_edges(xv, x, GRID)
                y_edges = self._axis_bin_edges(yv, y, GRID)
                hist, x_edges, y_edges = np.histogram2d(
                    xv, yv, bins=[x_edges, y_edges])
                hist = gaussian_filter(hist, sigma=SMOOTH_SIGMA)
                fmax = float(hist.max())
                if fmax <= 0:
                    print(f"[contour] {name}: zero-density grid — skipped",
                          flush=True)
                    continue

                # 2) Scatter beneath the contours (master 'Contour scatter'
                #    toggle). Each event's density classifies it as inside the
                #    contoured population (>= the lowest contour level) or an
                #    outlier below it; the 'Outliers' sub-toggle gates only the
                #    latter, so it can show just the within-population points.
                show_scatter = (getattr(self, 'contour_scatter_var', None)
                                is None or self.contour_scatter_var.get())
                show_outliers = (getattr(self, 'contour_outliers_var', None)
                                 is None or self.contour_outliers_var.get())
                if show_scatter:
                    nbx, nby = len(x_edges) - 1, len(y_edges) - 1
                    ex = np.clip(np.searchsorted(x_edges, xv, side='right') - 1,
                                 0, nbx - 1)
                    ey = np.clip(np.searchsorted(y_edges, yv, side='right') - 1,
                                 0, nby - 1)
                    zev = hist[ex, ey]
                    keep = (np.ones(xv.size, dtype=bool) if show_outliers
                            else (zev >= fmax * LEVELS_FROM))
                    sx, sy = xv[keep], yv[keep]
                    if sx.size > MAX_OUTLIER_PTS:
                        out_idx = rng.choice(sx.size, MAX_OUTLIER_PTS,
                                             replace=False)
                        sx, sy = sx[out_idx], sy[out_idx]
                    if sx.size:
                        self.ax.scatter(sx, sy, s=1.5, alpha=0.18,
                                        color=color, linewidths=0,
                                        rasterized=True)

                # 3) Convert edges to centres for matplotlib.contour, then
                #    surround the density with a ring of zeros so every level
                #    forms a CLOSED loop (a population running to the binning
                #    edge would otherwise produce open contours).
                xc = 0.5 * (x_edges[:-1] + x_edges[1:])
                yc = 0.5 * (y_edges[:-1] + y_edges[1:])
                hist = np.pad(hist, 1, mode='constant', constant_values=0.0)
                xc = np.concatenate([[xc[0] - (xc[1] - xc[0])], xc,
                                     [xc[-1] + (xc[-1] - xc[-2])]])
                yc = np.concatenate([[yc[0] - (yc[1] - yc[0])], yc,
                                     [yc[-1] + (yc[-1] - yc[-2])]])
                xx, yy = np.meshgrid(xc, yc, indexing='ij')
                levels = np.linspace(fmax * LEVELS_FROM,
                                     fmax * LEVELS_TO,
                                     N_LEVELS)
                self.ax.contour(xx, yy, hist, levels=levels,
                                colors=[color], linewidths=1.1, alpha=0.9)

                # Legend stub.
                self.ax.plot([], [], color=color, label=name)
            except Exception as exc:
                import traceback
                print(f"[contour] {name}: {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc()
                raise
        if len(samples) > 1:
            self.ax.legend(fontsize=8, loc='best')

    def _backgate_color(self, sname, gid, idx):
        """Backgate overlay colour for a population: its gate colour if one is
        set (so 'Set colour…' from the tree or the legend swatch applies here
        too), else a stable fallback from the backgate palette."""
        g = self._sample_gates.get(sname, {}).get(gid) or {}
        return g.get('color') or self._BACKGATE_COLORS[idx % len(self._BACKGATE_COLORS)]

    def _overlay_backgate(self, samples, x, y):
        """Project each backgate target population onto the current plot in its
        own colour, on top. The population's cumulative gate mask is computed
        on its sample's full data, then those events are drawn at the current
        x/y — so you can see where a downstream population sits on any axes.
        Each target gets a clickable legend row (on/off · density · colour)."""
        targets = getattr(self, '_backgate', None)
        if not targets:
            return
        from .pipeline import cumulative_gate_mask
        rng = np.random.default_rng(42)
        CAP = 60_000
        full = getattr(self, '_gate_density_full', None) or set()
        hidden = getattr(self, '_backgate_hidden', None) or set()
        rows = []                 # legend rows: one per resolvable target
        for i, (sname, gid) in enumerate(targets):
            match = (sname, gid) not in full     # default: scaled to the cloud
            on = (sname, gid) not in hidden
            s = self._samples.get(sname)
            if s is None:
                continue
            sample_gates = self._sample_gates.get(sname, {})
            if gid not in sample_gates:
                continue
            df = s.data
            alias = self._axis_alias_for_sample(s, [x, y])
            if alias:
                df = df.assign(**{ch: df[own] for ch, own in alias.items()})
            cols = [c for c in (x, y) if c and c in df.columns]
            if not cols:
                continue
            try:
                overrides = self._autoclean_overrides(sname, df)
                mask = np.asarray(cumulative_gate_mask(
                    sample_gates, gid, df, overrides=overrides), dtype=bool)
            except Exception as exc:
                print(f"[backgate] {sname}/{gid}: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            sub = df[mask].dropna(subset=cols)
            ntot = len(sub)
            if ntot == 0:
                continue
            n_full = int(mask.sum())
            color = self._backgate_color(sname, gid, i)
            label = self._population_path(sample_gates, gid)
            if len(samples) > 1:
                label = f'{sname} › {label}'
            n_shown = n_full
            if on:
                # Match the cloud's display fraction so the overlay's dot
                # density is comparable to the background, not always full/60k.
                if match:
                    shown, total = self._sample_display_count(sname)
                    frac = (shown / total) if total else 1.0
                    draw_cap = min(CAP, max(1, int(round(ntot * frac))))
                else:
                    draw_cap = CAP
                if ntot > draw_cap:
                    sub = sub.sample(draw_cap, random_state=42)
                n_shown = len(sub)
                if y:
                    self.ax.scatter(sub[x].to_numpy(dtype=float),
                                    sub[y].to_numpy(dtype=float),
                                    s=8, c=color, alpha=0.9, linewidths=0,
                                    marker='o', zorder=6, rasterized=True)
                else:
                    # Histogram mode: rug ticks at the population's x-values.
                    xv = sub[x].to_numpy(dtype=float)
                    xv = xv[np.isfinite(xv)]
                    if xv.size > draw_cap:
                        xv = rng.choice(xv, draw_cap, replace=False)
                    ybot, ytop = self.ax.get_ylim()
                    self.ax.vlines(xv, ybot, ybot + (ytop - ybot) * 0.04,
                                   color=color, alpha=0.5, linewidth=0.5,
                                   zorder=6)
            if on and match and n_shown < n_full:
                cnt = f'{n_shown:,} of {n_full:,}'
            else:
                cnt = f'n={n_full:,}'
            rows.append({'target': (sname, gid), 'color': color, 'on': on,
                         'match': match, 'label': f'{label} ({cnt})'})
        self._draw_backgate_legend(rows)

    def _draw_backgate_legend(self, rows):
        """Custom clickable legend: per backgate a colour swatch (→ Set
        colour), an on/off dot, and a ☑/☐ density box. Draggable by its header
        and collapsible (▾/▸). Picks route through _on_canvas_pick via the
        artist→(action, target) map; the box's bbox is recorded so plot clicks
        over it don't fall through to gate creation."""
        self._backgate_legend_pick = {}
        self._backgate_legend_artists = []
        self._backgate_legend_rows = rows
        self._backgate_legend_bbox = None
        self._backgate_legend_header = None
        if not rows:
            return
        import matplotlib.patches as mpatches
        ax = self.ax
        ax0, top = self._backgate_legend_anchor
        collapsed = self._backgate_legend_collapsed
        ROW_H, GREY = 0.046, '#9a9a9a'
        # Theme-aware neutrals so the legend reads on a dark plot too.
        _pp = current_palette()
        INK = _pp.get('plot_fg', '#222222')        # header / glyphs / on-text
        BOX_FACE = _pp.get('plot_bg', 'white')
        BOX_EDGE = _pp.get('plot_spine', '#cfcfcf')
        HEAD_H, W = 0.034, 0.46
        n = 0 if collapsed else len(rows)
        x_sw = ax0
        x_on, x_den, x_lbl = ax0 + 0.032, ax0 + 0.064, ax0 + 0.096
        body_h = n * ROW_H
        box_top = top + 0.004
        box_bot = top - HEAD_H - body_h - 0.006

        def _keep(artist):
            self._backgate_legend_artists.append(artist)
            return artist

        bg = mpatches.FancyBboxPatch(
            (ax0 - 0.012, box_bot), W, box_top - box_bot,
            transform=ax.transAxes, boxstyle='round,pad=0.004',
            facecolor=BOX_FACE, edgecolor=BOX_EDGE, alpha=0.88, zorder=9)
        _keep(ax.add_patch(bg))
        # Header (drag handle): collapse glyph + title + count.
        hy = top - HEAD_H * 0.4
        col_glyph = _keep(ax.text(
            x_sw, hy, '▸' if collapsed else '▾', transform=ax.transAxes,
            fontsize=9, va='center', ha='center', color=INK,
            picker=True, zorder=10))
        self._backgate_legend_pick[col_glyph] = ('collapse', None)
        _keep(ax.text(x_sw + 0.022, hy, f'backgate ({len(rows)})',
                      transform=ax.transAxes, fontsize=8, fontweight='bold',
                      color=INK, va='center', ha='left', zorder=10))
        self._backgate_legend_header = (ax0 - 0.012, top - HEAD_H,
                                        ax0 - 0.012 + W, box_top)
        for r, row in enumerate(rows if not collapsed else []):
            y = top - HEAD_H - (r + 0.5) * ROW_H
            on, match, color, tgt = (row['on'], row['match'],
                                     row['color'], row['target'])
            sw = _keep(ax.text(x_sw, y, '■', color=(color if on else GREY),
                       transform=ax.transAxes, fontsize=11, va='center',
                       ha='center', picker=True, zorder=10))
            ong = _keep(ax.text(x_on, y, '◉' if on else '◯', color=INK,
                        transform=ax.transAxes, fontsize=10, va='center',
                        ha='center', picker=True, zorder=10))
            deng = _keep(ax.text(x_den, y, '☑' if match else '☐',
                         color=(INK if on else GREY),
                         transform=ax.transAxes, fontsize=10, va='center',
                         ha='center', picker=True, zorder=10))
            lt = _keep(ax.text(x_lbl, y, row['label'],
                       color=(INK if on else GREY),
                       transform=ax.transAxes, fontsize=8, va='center',
                       ha='left', picker=True, zorder=10))
            self._backgate_legend_pick[sw] = ('color', tgt)
            self._backgate_legend_pick[ong] = ('toggle', tgt)
            self._backgate_legend_pick[deng] = ('density', tgt)
            self._backgate_legend_pick[lt] = ('toggle', tgt)
        self._backgate_legend_bbox = (ax0 - 0.012, box_bot,
                                      ax0 - 0.012 + W, box_top)

    def _draw_highlight_overlays(self, samples, x, y):
        """For EVERY enabled gate (root, intermediate, or leaf) in each
        sample's own gate tree, overlay its cumulative-chain events on
        the base plot in the gate's colour. Parents drawn first so
        children's smaller, more-specific populations sit on top —
        FlowJo-style nested-population rendering. No-op outside
        'highlight' mode."""
        if (not hasattr(self, 'gate_display_var')
                or self.gate_display_var.get() != 'highlight'):
            return
        from .pipeline import cumulative_gate_mask

        is_hist = (y is None)

        for name in samples:
            sample_gates = self._sample_gates.get(name, {})
            if not sample_gates:
                continue
            order = [gid for gid in self._gates_topological_for(sample_gates)
                     if sample_gates[gid].get('enabled', True)]
            if not order:
                continue
            df = self._get_df(name, x, y)
            if df.empty or x not in df.columns or (y and y not in df.columns):
                continue
            # Use the same full-data auto-clean masks as filter mode so a
            # cleaning gate flags the SAME events here (otherwise the time-
            # binned methods would recompute on this downsampled / dropna'd
            # subset and disagree with the filtered view).
            overrides = self._autoclean_overrides(name, df)
            df_full = overrides_full = None        # lazy full-density df
            for gid in order:
                # Population marked full-density (☐ in the density column) draws
                # from the un-downsampled data; the default scaled view uses the
                # same downsampled cloud as the background.
                if (name, gid) in self._gate_density_full:
                    if df_full is None:
                        df_full = self._get_df(name, x, y, downsample=False)
                        overrides_full = self._autoclean_overrides(name, df_full)
                    gdf, gov = df_full, overrides_full
                else:
                    gdf, gov = df, overrides
                if gdf.empty or x not in gdf.columns or (y and y not in gdf.columns):
                    continue
                mask = cumulative_gate_mask(sample_gates, gid, gdf,
                                            overrides=gov)
                if not mask.any():
                    continue
                color = sample_gates[gid].get('color', '#e6194b')
                lbl = f'{name}:{gid}' if len(samples) > 1 else f'gate {gid}'
                if is_hist:
                    # Strip non-finite values + reuse the base axes' x-range
                    # so the highlight overlays line up with the underlying
                    # histogram bins. Skip if no finite values remain
                    # (rare but possible after a tight gate).
                    arr = np.asarray(gdf[x].values[mask], dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue
                    xlo, xhi = self.ax.get_xlim()
                    # Reuse the same scale-aware spacing + kernel smoothing as
                    # the base histogram so the highlight overlay lines up and
                    # reads as a smooth curve, not chunky step bars.
                    NBINS = 256
                    bin_edges = np.asarray(self._screen_uniform_edges(
                        x, xlo, xhi, NBINS, data_sample=arr), dtype=float)
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    from scipy.ndimage import gaussian_filter1d
                    counts, _ = np.histogram(arr, bins=bin_edges)
                    counts = counts.astype(float)
                    per_bin = arr.size / float(NBINS)
                    sigma = float(np.clip(
                        np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.5, 1.0, 4.0))
                    sm = gaussian_filter1d(counts, sigma=sigma, mode='constant')
                    # Fraction per bin (bins are screen-uniform; density would
                    # crush the bright tail on a log/symlog/composite axis).
                    # Distinct local — do NOT reassign the `y` parameter (None in
                    # histogram mode); shadowing it crashed the next gate/sample
                    # iteration (bool(ndarray) / channel-lookup on an array).
                    yprof = sm / arr.size if arr.size else sm
                    self.ax.fill_between(centers, yprof, color=color, alpha=0.40,
                                         linewidth=0)
                    self.ax.plot(centers, yprof, color=color, linewidth=1.3,
                                 label=lbl)
                else:
                    xv = np.asarray(gdf[x].values[mask])
                    yv = np.asarray(gdf[y].values[mask])
                    self.ax.scatter(xv, yv, s=4, alpha=0.85,
                                    color=color, linewidths=0,
                                    rasterized=True, label=lbl)
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(fontsize=8, loc='best', framealpha=0.85)

    def _reset_plot_view(self):
        """Restore the data-fit view (recomputes scales/limits from scratch)."""
        self._schedule_replot(0)
        self.status_var.set("Plot view reset.")

    def _toggle_zoom_tool(self):
        """Enter/leave zoom-to mode: grey out the gating tools while active so
        a drag zooms (rectangle) instead of creating a gate."""
        self._zoom_mode = bool(self._zoom_mode_var.get())
        for w in getattr(self, '_gate_tool_widgets', []):
            try:
                w.state(['disabled'] if self._zoom_mode else ['!disabled'])
            except Exception:
                pass
        self.status_var.set(
            "Zoom tool ON — drag a rectangle on the plot to zoom in "
            "(gating is paused)." if self._zoom_mode
            else "Zoom tool off — gating tools re-enabled.")

    def _zoom_step(self, factor):
        """Zoom in/out around the current plot centre (for the +/- buttons,
        since the cursor isn't over the plot)."""
        try:
            xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
            cx, cy = (xl[0] + xl[1]) / 2, (yl[0] + yl[1]) / 2
            self.ax.set_xlim(cx + (xl[0] - cx) * factor,
                             cx + (xl[1] - cx) * factor)
            self.ax.set_ylim(cy + (yl[0] - cy) * factor,
                             cy + (yl[1] - cy) * factor)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _swap_axes(self):
        """Swap the X and Y axis channels and replot. Scale/range follow
        because they're keyed by channel name, not axis slot."""
        try:
            x, y = self.x_combo.get(), self.y_combo.get()
        except Exception:
            return
        if not x or not y or x == y:
            return
        self.x_combo.set(y)
        self.y_combo.set(x)
        self._schedule_replot(0)
        self.status_var.set(f"Axes swapped — X: {y}   Y: {x}")

    def _apply_axis_to_ax(self, channel, axis_letter, data_sample=None):
        """Apply this channel's display scale + range to the matplotlib axes.

        Called at the end of ``_replot`` for both X and Y (when present).

        For a channel whose data is baked into a nonlinear transform
        (logicle/hyperlog/asinh/log), linear/symlog/log are rendered as
        composite FuncScale VIEWS of the underlying linear intensity (see
        ``_axis_view_funcs``) — proper, independent views with no double-
        transform, and gates auto-follow. Linear-data channels (scatter)
        use matplotlib's native linear/symlog/log scale; for symlog we pick
        a ``linthresh`` from the data (5th percentile of |data|), else 1.0.
        """
        scale = self._channel_scale.get(channel, self._default_scale_for(channel))
        set_scale = (self.ax.set_xscale if axis_letter == 'x'
                     else self.ax.set_yscale)
        set_lim   = (self.ax.set_xlim if axis_letter == 'x'
                     else self.ax.set_ylim)
        funcs = self._axis_view_funcs(channel, data_sample)
        try:
            if funcs is not None:
                set_scale('function', functions=funcs)
            elif scale == 'log':
                set_scale('log')
            elif scale == 'symlog':
                # Same linthresh the density binning uses → bins align with
                # the axis (no boxy artefacts in the log decade).
                set_scale('symlog',
                          linthresh=self._symlog_linthresh(data_sample))
            else:
                set_scale('linear')
        except Exception:
            # E.g. log scale with non-positive data — fall back silently
            # to linear rather than crashing the plot.
            try:
                set_scale('linear')
            except Exception:
                pass
        rng = self._channel_range.get(channel)
        if rng is not None:
            try:
                set_lim(rng[0], rng[1])
            except Exception:
                pass

    def _overlay_removed_events(self, samples, x, y, mode):
        """Draw the auto-clean-removed events on TOP of the current plot in
        red, so cleaning artefacts stay visible against the full sample even
        at a tiny error rate. Bypasses the display cap (surfacing the few
        dropped events is the whole point). Toggled by ``show_removed_var``.

        Scatter modes overlay the removed events as red dots; histogram mode
        overlays their channel distribution as a red curve scaled to the axis
        height (location, not magnitude — labelled as such)."""
        if not (getattr(self, 'show_removed_var', None)
                and self.show_removed_var.get()):
            return
        RED = '#e8000b'

        if mode == 'histogram':
            xs = []
            for name in samples:
                rem = self._removed_events(name, x, None)
                if rem is not None and x in rem.columns:
                    xs.append(np.asarray(rem[x].values, dtype=float))
            xs = np.concatenate(xs) if xs else np.array([])
            xs = xs[np.isfinite(xs)]
            if xs.size == 0:
                return
            from scipy.ndimage import gaussian_filter1d
            xlo, xhi = self.ax.get_xlim()
            _, ytop = self.ax.get_ylim()
            NBINS = 256
            edges = np.asarray(self._screen_uniform_edges(
                x, min(xlo, xhi), max(xlo, xhi), NBINS, data_sample=xs),
                dtype=float)
            centers = 0.5 * (edges[:-1] + edges[1:])
            counts = np.histogram(xs, bins=edges)[0].astype(float)
            sigma = float(np.clip(np.sqrt(NBINS / max(xs.size, 1e-6)) * 1.5,
                                  1.0, 4.0))
            sm = gaussian_filter1d(counts, sigma=sigma, mode='constant')
            peak = float(sm.max())
            if peak <= 0:
                return
            # Scale so the removed-event profile peaks at ~85% of the axis —
            # visible no matter how few were removed (shows WHERE, not height).
            y_ov = sm * (0.85 * ytop / peak)
            self.ax.fill_between(centers, y_ov, color=RED, alpha=0.22,
                                 linewidth=0, zorder=5)
            self.ax.plot(centers, y_ov, color=RED, linewidth=1.5, zorder=6,
                         label=f'cleaned-out (n={xs.size:,}, location)')
            self.ax.legend(fontsize=8, loc='best')
            return

        # Scatter modes (dot / pseudocolor / contour).
        # Each removed event is coloured by the cleaning method that dropped
        # it (each method = a distinct "section" / pullable population), and
        # the overlay is SUBSAMPLED to the same fraction the main plot shows
        # (shown/total) so the red layer's density stays proportionate to the
        # visible sample instead of over-dominating it.
        import matplotlib.patches as mpatches
        rng = np.random.default_rng(42)
        order = list(self._METHOD_COLORS.keys())
        groups: dict = {}          # method_key -> [xs_arrays], [ys_arrays]
        full_counts: dict = {}     # method_key -> total removed (full sample)
        for name in samples:
            rem = self._removed_events(name, x, y)
            if rem is None or x not in rem.columns or not y \
                    or y not in rem.columns:
                continue
            # Per-event method attribution (first enabled method, recipe order).
            method_masks = self._autoclean_method_masks(name)
            label = np.full(len(rem), '', dtype=object)
            for key in order + [k for k in method_masks if k not in order]:
                ser = method_masks.get(key)
                if ser is None:
                    continue
                m = ser.reindex(rem.index, fill_value=False).to_numpy()
                take = m & (label == '')
                label[take] = key
            rx = np.asarray(rem[x].values, dtype=float)
            ry = np.asarray(rem[y].values, dtype=float)
            fin = np.isfinite(rx) & np.isfinite(ry)
            rx, ry, lab = rx[fin], ry[fin], label[fin]
            for key in np.unique(lab):
                full_counts[key] = full_counts.get(key, 0) + int((lab == key).sum())
            # Proportional subsample to the displayed fraction of this sample.
            shown, total = self._sample_display_count(name)
            frac = (shown / total) if total else 1.0
            nrem = rx.size
            k = int(round(frac * nrem))
            if nrem and k == 0:
                k = min(nrem, 25)          # keep a real error rate visible
            if 0 < k < nrem:
                sel = rng.choice(nrem, k, replace=False)
                rx, ry, lab = rx[sel], ry[sel], lab[sel]
            for key in np.unique(lab):
                gx, gy = groups.setdefault(key, ([], []))
                mk = lab == key
                gx.append(rx[mk]); gy.append(ry[mk])
        if not groups:
            return
        handles = []
        for key in order + [k for k in groups if k not in order]:
            if key not in groups:
                continue
            gx = np.concatenate(groups[key][0])
            gy = np.concatenate(groups[key][1])
            if gx.size == 0:
                continue
            color = self._METHOD_COLORS.get(key, RED)
            self.ax.scatter(gx, gy, s=7, c=color, alpha=0.85, linewidths=0,
                            marker='o', zorder=5, rasterized=True)
            lbl = key or 'removed'
            handles.append(mpatches.Patch(
                color=color, label=f'{lbl} (n={full_counts.get(key, gx.size):,})'))
        if handles:
            self.ax.legend(handles=handles, fontsize=8, loc='best',
                           framealpha=0.85, title='cleaned-out')

    def _plot_histogram(self, samples, x):
        """Overlay per-sample density histograms of channel ``x``.

        Two failure modes the naïve ``ax.hist(df[x].values, bins=200,
        density=True)`` hits and that this implementation works around:

        1. **Non-finite values.** If any sample's column contains NaN /
           ±inf, matplotlib's histogram silently skips the offending
           bin or renders empty. Filter them up-front.
        2. **Vastly different ranges across samples / channels.** When
           sample A has data on logicle scale (~0–1) and sample B has
           raw scale (0–262144), matplotlib auto-ranges to the union →
           sample A collapses into a single bin at zero, sample B's
           bars become invisibly short. Use a robust per-sample percentile
           clip (0.1–99.9) unioned across samples, then pin every
           sample's bins to the same edges so the overlay is comparable.
        """
        clean_series = []
        for name in samples:
            df = self._get_df(name, x, None, for_hist=True)
            if x not in df.columns or df.empty:
                continue
            arr = np.asarray(df[x].values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            clean_series.append((name, arr))

        if not clean_series:
            self._hist_x_anchor = None
            self.ax.text(0.5, 0.5,
                         f'No finite data for "{x}" — nothing to plot.',
                         ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=10, color='#888')
            return

        # Union of robust per-sample [p0.1, p99.9] ranges, then a small
        # symmetric pad so the tails are visible. Falls back to (min,max)
        # for very small samples.
        lo, hi = np.inf, -np.inf
        for _name, arr in clean_series:
            if arr.size >= 20:
                a, b = np.percentile(arr, (0.1, 99.9))
            else:
                a, b = float(arr.min()), float(arr.max())
            if a < lo: lo = float(a)
            if b > hi: hi = float(b)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # Degenerate (constant) — fall back to ±1 around the value.
            lo, hi = lo - 1.0, lo + 1.0
        else:
            pad = (hi - lo) * 0.02
            lo, hi = lo - pad, hi + pad

        # Bin edges spaced uniformly in SCREEN space for the channel's
        # display scale (composite FuncScale view for nonlinear-baked
        # channels, else log/linear), so bins look even on the axis.
        NBINS = 256
        # Stash the exact array the bins are built on so _replot anchors the
        # symlog axis cofactor on the SAME data — otherwise the axis uses the
        # raw column and the symlog bins/axis disagree (banding).
        self._hist_x_anchor = np.asarray(clean_series[0][1], dtype=float)
        bin_edges = np.asarray(self._screen_uniform_edges(
            x, lo, hi, NBINS, data_sample=self._hist_x_anchor), dtype=float)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Y-axis mode (user-selectable):
        #   Fraction (default) — events per bin ÷ sample total (sums to 1).
        #   Count              — raw events per bin.
        #   % of Max           — each curve scaled so its tallest bin = 100.
        # Counts (not density=True): density divides by the bar's DATA-space
        # width, but bins are uniform in SCREEN space, so on a log/symlog/
        # composite axis their data widths vary enormously and density would
        # crush the bright tail while spiking the dim peak. Raw counts keep
        # the shape true and overlaid samples comparable.
        #
        # Each profile is rendered as a kernel-SMOOTHED filled curve rather
        # than raw step bars — the bars read as chunky/jagged, the smoothed
        # curve matches the FlowJo look (and the now-smooth pseudocolor).
        # Smoothing is adaptive: sparse populations get a wider kernel.
        from scipy.ndimage import gaussian_filter1d
        ymode = (self.hist_y_mode.get()
                 if getattr(self, 'hist_y_mode', None) is not None
                 else 'Fraction')
        for name, arr in clean_series:
            counts, _ = np.histogram(arr, bins=bin_edges)
            counts = counts.astype(float)
            per_bin = arr.size / float(NBINS)
            sigma = float(np.clip(np.sqrt(1.0 / max(per_bin, 1e-6)) * 1.5,
                                  1.0, 4.0))
            sm = gaussian_filter1d(counts, sigma=sigma, mode='constant')
            if ymode == 'Count':
                y = sm
            elif ymode == '% of Max':
                peak = float(sm.max()) if sm.size else 0.0
                y = sm * (100.0 / peak) if peak > 0 else sm
            else:   # Fraction
                y = sm / arr.size if arr.size else sm
            color = self._color_for(name)
            self.ax.fill_between(centers, y, color=color, alpha=0.30,
                                 linewidth=0)
            self.ax.plot(centers, y, color=color, linewidth=1.4, label=name)

        self.ax.set_ylabel(
            {'Count': 'count', '% of Max': '% of max'}.get(ymode, 'fraction'))
        self.ax.set_xlim(lo, hi)
        if len(clean_series) > 1:
            self.ax.legend(fontsize=8, loc='best')

    def _render_into(self, ax, samples, x, y, mode, color,
                     draw_gates=True, draw_overlays=True):
        """Render one plot panel into an arbitrary matplotlib ``Axes``,
        reusing the live plotting pipeline.

        The ``_plot_*`` / ``_overlay_*`` / ``_draw_gates`` / ``_apply_axis``
        helpers all draw into ``self.ax`` on ``self.fig``. Rather than
        duplicate their logic, this temporarily points those attributes
        (plus the gate-artist registries and the colorbar handle) at the
        supplied ``ax`` and its parent figure, renders, then restores live
        state in a ``finally`` so the on-screen plot is never disturbed.
        Colorbars are suppressed in panels (see ``_suppress_panel_cbar``).
        """
        saved = (self.ax, self.fig, self._cbar,
                 self._vlines, self._hlines,
                 getattr(self, '_shape_artists', {}))
        suppress_prev = getattr(self, '_suppress_panel_cbar', False)
        self.ax = ax
        self.fig = ax.figure
        self._cbar = None
        self._vlines, self._hlines, self._shape_artists = {}, {}, {}
        self._suppress_panel_cbar = True
        try:
            if not samples or not x:
                ax.text(0.5, 0.5, '(nothing to plot)', ha='center',
                        va='center', transform=ax.transAxes,
                        fontsize=9, color='grey')
                return
            try:
                if mode == 'histogram':
                    self._plot_histogram(samples, x)
                elif mode == 'dot':
                    self._plot_dot(samples, x, y, color)
                elif mode == 'pseudocolor':
                    self._plot_pseudocolor(samples, x, y)
                elif mode == 'contour':
                    self._plot_contour(samples, x, y)
            except Exception as exc:
                ax.text(0.5, 0.5, f'Plot error:\n{exc}', ha='center',
                        va='center', transform=ax.transAxes,
                        fontsize=8, color='red')
            if draw_overlays:
                try:
                    self._overlay_removed_events(samples, x, y, mode)
                except Exception:
                    pass
                try:
                    self._overlay_backgate(samples, x, y)
                except Exception:
                    pass
            ax.set_xlabel(self._fmt_channel(x), fontsize=8)
            if mode != 'histogram' and y:
                ax.set_ylabel(self._fmt_channel(y), fontsize=8)
            first = samples[0] if samples else None
            sdata = None
            if first and x and first in self._samples:
                sdf = self._samples[first].data
                if x in sdf.columns:
                    sdata = sdf[x].values
            self._apply_axis_to_ax(x, 'x', sdata)
            if mode != 'histogram' and y:
                ydata = None
                if first and y and first in self._samples:
                    sdf = self._samples[first].data
                    if y in sdf.columns:
                        ydata = sdf[y].values
                self._apply_axis_to_ax(y, 'y', ydata)
            if draw_overlays:
                try:
                    self._draw_highlight_overlays(
                        samples, x, y if mode != 'histogram' else None)
                except Exception:
                    pass
            if draw_gates:
                try:
                    self._draw_gates(x, y if mode != 'histogram' else None)
                except Exception:
                    pass
            ax.tick_params(labelsize=7)
        finally:
            (self.ax, self.fig, self._cbar,
             self._vlines, self._hlines, self._shape_artists) = saved
            self._suppress_panel_cbar = suppress_prev

    def _plot_dot(self, samples, x, y, color):
        if not y:
            return
        if color == 'By sample':
            for name in samples:
                df = self._get_df(name, x, y)
                if df.empty or x not in df.columns or y not in df.columns:
                    continue
                self.ax.scatter(df[x].values, df[y].values,
                                c=self._color_for(name),
                                s=2, alpha=0.35, linewidths=0, label=name)
            if len(samples) > 1:
                self.ax.legend(fontsize=8, markerscale=4, framealpha=0.85,
                               loc='best')
        elif color == 'By density':
            xs, ys = [], []
            for name in samples:
                df = self._get_df(name, x, y)
                if x not in df.columns or y not in df.columns:
                    continue
                xs.append(df[x].values); ys.append(df[y].values)
            if not xs:
                return
            xs = np.concatenate(xs); ys = np.concatenate(ys)
            self._density_scatter(xs, ys, x, y)
        else:
            cch = self._resolve_channel(color)
            xs, ys, cs = [], [], []
            for name in samples:
                df = self._get_df(name, x, y)
                if (cch not in df.columns or x not in df.columns
                        or y not in df.columns):
                    continue
                xs.append(df[x].values); ys.append(df[y].values)
                cs.append(df[cch].values)
            if xs:
                xs = np.concatenate(xs); ys = np.concatenate(ys)
                cs = np.concatenate(cs)
                sc = self.ax.scatter(xs, ys, c=cs, cmap='viridis',
                                     s=2, alpha=0.55, linewidths=0)
                # Suppress the colorbar when rendering small-multiple panels
                # for the figure exporter — it steals axes space and clutters
                # a grid. The live plot (flag unset) keeps it.
                if not getattr(self, '_suppress_panel_cbar', False):
                    self._cbar = self.fig.colorbar(
                        sc, ax=self.ax, label=self._fmt_channel(cch))

    def _plot_pseudocolor(self, samples, x, y):
        if not y:
            return
        xs, ys = [], []
        for name in samples:
            df = self._get_df(name, x, y)
            if x not in df.columns or y not in df.columns:
                continue
            xs.append(df[x].values); ys.append(df[y].values)
        if not xs:
            return
        xs = np.concatenate(xs); ys = np.concatenate(ys)
        self._density_scatter(xs, ys, x, y)

    def _default_scale_for(self, channel):
        """Default display scale for a channel with no explicit choice:
        'linear' for embedding axes (UMAP1/2, TSNE1/2, …), else the global
        default ('log', tuned for fluorescence)."""
        if channel:
            cu = str(channel).upper()
            for p in self._EMBED_AXIS_PREFIXES:
                if cu.startswith(p) and cu[len(p):] in ('1', '2'):
                    return 'linear'
        return self._default_channel_scale

    def _removed_events(self, name, x, y):
        """The events the auto-clean recipe REMOVES for ``name``, as a
        DataFrame carrying the (aliased) plot columns. Computed on the FULL
        sample — uncapped and ungated — so a small error rate isn't
        subsampled away before it can be shown. ``None`` when the sample has
        no auto-clean gate or nothing is removed."""
        s = self._samples.get(name)
        if s is None:
            return None
        df = s.data
        alias = self._axis_alias_for_sample(s, [x, y])
        if alias:
            df = df.assign(**{chosen: df[own] for chosen, own in alias.items()})
        cols = [c for c in (x, y) if c and c in df.columns]
        if not cols:
            return None
        df = df.dropna(subset=cols)
        overrides = self._autoclean_overrides(name, df)
        if not overrides:
            return None
        keep = np.ones(len(df), dtype=bool)
        for m in overrides.values():
            keep &= np.asarray(m, dtype=bool)
        removed = df[~keep]
        return removed if not removed.empty else None

    def _backgate_selected(self):
        """Set the backgate targets from the selected gate row(s): their
        populations get projected onto the current plot. Multi-select adds
        several, each its own colour."""
        targets = []
        for iid in self.gate_tv.selection():
            p = self._parse_iid(iid)
            if p and p[0] == 'gate':
                targets.append((p[1], p[2]))
        if not targets:
            self.status_var.set("Select a gate/population to backgate.")
            return
        self._backgate = targets
        self.status_var.set(
            f"Backgating {len(targets)} population(s) — shown in colour on the "
            f"plot. Right-click → Clear backgating to remove.")
        self._schedule_replot(0)

    def _clear_backgate(self):
        # Density preferences are a per-population property, not backgate state,
        # so they persist across clearing the backgate overlay.
        self._backgate = []
        self._backgate_hidden.clear()
        self._backgate_legend_pick = {}
        self.status_var.set("Backgating cleared.")
        self._schedule_replot(0)

    def _reposition_backgate_legend(self):
        """Cheap legend-only redraw (no full replot): remove the current
        legend artists and re-draw at the current anchor / collapsed state.
        Used for dragging and collapse so big scatters aren't re-rendered."""
        for a in list(self._backgate_legend_artists):
            try:
                a.remove()
            except Exception:
                pass
        self._draw_backgate_legend(self._backgate_legend_rows)
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _event_axes_frac(self, event):
        """Pixel event → (fx, fy) in axes-fraction coords, or None."""
        if getattr(event, 'x', None) is None or event.y is None:
            return None
        try:
            fx, fy = self.ax.transAxes.inverted().transform((event.x, event.y))
            return (float(fx), float(fy))
        except Exception:
            return None
