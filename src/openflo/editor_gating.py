"""Gate drawing, mouse interaction (create / edit / pan / zoom-rect / legend),
hit-testing, and gate-model CRUD — editor mixin.

The interactive-gating half of ViewGateEditorWindow. Methods call plot/model
helpers on ``self``; see editor_base.EditorMixin.
"""
from __future__ import annotations

import numpy as np

from .editor_base import EditorMixin


class GatingMixin(EditorMixin):
    """Gate overlays, canvas mouse handlers (gating + pan + zoom-rectangle +
    backgate-legend), hit-testing, vertex editing, and gate add/clear/purge."""

    def _add_gate(self, gate, parent_id=None, audit=True):
        """Register `gate` and return its id.

        Fills in defaults for `parent_id` (the currently-selected gate, or
        None for a root) and `color` (next palette entry) if the caller
        didn't supply them.

        Threshold/interval gates are *unique per (channel, parent)*: if a
        matching one already exists, this call replaces it in place — that's
        how a histogram slider live-updates a single gate, and how repeated
        double-clicks on the plot move the same threshold instead of
        stacking new ones.

        ``audit`` records a ``gate.add`` provenance entry for genuinely-new
        gates (suppressed during bulk import / session restore via
        ``_suspend_undo``, and by callers that log their own summary — e.g.
        auto-gate). Threshold/interval *replacements* are never logged (they'd
        flood from slider drags).
        """
        self._checkpoint()
        if 'parent_id' not in gate:
            gate['parent_id'] = (parent_id
                                 if parent_id is not None
                                 else self._selected_gate_id())
        if 'color' not in gate:
            gate['color'] = self._next_color()
        if 'enabled' not in gate:
            gate['enabled'] = True

        if gate.get('kind') in ('threshold', 'interval'):
            ch = gate['channel']
            pid = gate['parent_id']
            for gid in list(self._gates):
                g = self._gates[gid]
                if (g.get('kind') in ('threshold', 'interval')
                        and g.get('channel') == ch
                        and g.get('parent_id') == pid):
                    # Keep the existing colour so the user's visual cue
                    # doesn't change when they nudge a threshold.
                    gate['color'] = g.get('color', gate['color'])
                    self._gates[gid] = gate
                    return gid

        gid = self._next_gate_id()
        self._gates[gid] = gate
        self._gate_id_order.append(gid)
        if audit and not self._suspend_undo:
            from .pipeline import describe_gate
            self._audit('gate.add', sample=self._active_sample, id=gid,
                        kind=gate.get('kind'),
                        gate=describe_gate(gate))
        return gid

    def _gate_targets_for_new(self):
        """Samples a newly-created gate should land on: every displayed sample
        when the 'apply to all displayed' toggle is on; otherwise the tree
        multi-selection when more than one sample is selected; otherwise just
        the active sample. Active is always first."""
        if (getattr(self, '_gate_all_var', None) is not None
                and self._gate_all_var.get()):
            extra = self._target_samples('enabled')
        else:
            sel = self._target_samples('selected')
            extra = sel if len(sel) > 1 else []
        out = [n for n in extra if n in self._samples]
        active = self._active_sample
        if active in self._samples and active not in out:
            out.insert(0, active)
        return out

    def _add_gate_multi(self, gate, parent_id=None, audit=True):
        """Add ``gate`` to the active sample and — when 'apply to all displayed'
        is on or several samples are selected — replicate the SAME gate (fresh
        per-sample id, rooted) onto each other target. Used by the gate tools
        and auto-gate so a drawn/proposed gate lands on every chosen sample.
        Returns the active sample's new gid."""
        import copy as _copy
        gid = self._add_gate(gate, parent_id=parent_id, audit=audit)
        targets = [n for n in self._gate_targets_for_new()
                   if n != self._active_sample]
        if not targets:
            return gid
        template = self._gates.get(gid)
        if template is None:
            return gid
        for name in targets:
            self._sample_gates.setdefault(name, {})
            self._sample_gate_order.setdefault(name, [])
            self._sample_gate_seq.setdefault(name, 0)
            self._sample_gate_seq[name] += 1
            new_id = f'g{self._sample_gate_seq[name]}'
            g = _copy.deepcopy(template)
            g['id'] = new_id
            # Root the replica — the active sample's parent gid may not exist on
            # the target (different gate trees); the user can re-parent.
            g['parent_id'] = None
            self._sample_gates[name][new_id] = g
            self._sample_gate_order[name].append(new_id)
        # Show the replicated gates on the other samples in the tree.
        self._refresh_gate_list()
        self.status_var.set(
            f"Gate applied to {len(targets) + 1} samples.")
        return gid

    def _ordered_gate_ids(self):
        """All gate ids in insertion order, with any dangling ids
        (added via _add_gate's 1D replacement path) sorted at the end."""
        seen = set(self._gate_id_order)
        return list(self._gate_id_order) + [g for g in self._gates if g not in seen]

    def _legend_press(self, event):
        """If a press lands on the backgate legend, swallow it (return True so
        the plot doesn't create/select a gate underneath) and start a drag when
        it's on the header strip. Glyph clicks are handled separately by the
        pick event."""
        if not self._backgate_legend_bbox:
            return False
        fr = self._event_axes_frac(event)
        if not self._in_box(fr, self._backgate_legend_bbox):
            return False
        if getattr(event, 'button', 1) == 1 and self._in_box(
                fr, self._backgate_legend_header):
            self._legend_drag = {'sx': event.x, 'sy': event.y,
                                 'anchor': tuple(self._backgate_legend_anchor)}
        return True

    def _on_canvas_pick(self, event):
        """A clickable backgate legend glyph was clicked. Dispatch by action:
        'color' opens the colour chooser, 'toggle' shows/hides the backgate,
        'density' flips scaled↔full (and mirrors into the tree)."""
        entry = getattr(self, '_backgate_legend_pick', {}).get(event.artist)
        if entry is None:
            return
        action, tgt = entry
        if action == 'collapse':
            self._backgate_legend_collapsed = not self._backgate_legend_collapsed
            self._reposition_backgate_legend()
            return
        if action == 'color':
            self._pick_gate_color(*tgt)          # updates colour + replots
            return
        if action == 'toggle':
            if tgt in self._backgate_hidden:
                self._backgate_hidden.discard(tgt)
            else:
                self._backgate_hidden.add(tgt)
            self._schedule_replot(0)
            return
        if action == 'density':
            tgt_full = self._gate_density_full
            if tgt in tgt_full:
                tgt_full.discard(tgt)
            else:
                tgt_full.add(tgt)
            self._schedule_replot(0)

    def _draw_gates(self, x, y):
        """Render every gate that intersects the current axes, in the gate's
        own colour. Lines for 1D gates (vertical when channel == x;
        horizontal when == y); a matplotlib Patch for rect/polygon."""
        import matplotlib.patches as mpatches

        self._vlines, self._hlines, self._shape_artists = {}, {}, {}

        for gid, g in self._gates.items():
            color = g.get('color', 'red')
            k = g.get('kind')
            if k == 'threshold':
                ch = g['channel']
                if ch == x:
                    self._vlines[gid] = self.ax.axvline(
                        float(g['value']), color=color, lw=1.3,
                        ls='--', alpha=0.85)
                elif ch == y:
                    self._hlines[gid] = self.ax.axhline(
                        float(g['value']), color=color, lw=1.3,
                        ls='--', alpha=0.85)
            elif k == 'interval':
                ch = g['channel']
                lo, hi = float(g['lo']), float(g['hi'])
                if ch == x:
                    self._vlines[f'{gid}:lo'] = self.ax.axvline(
                        lo, color=color, lw=1.2, ls='--', alpha=0.85)
                    self._vlines[f'{gid}:hi'] = self.ax.axvline(
                        hi, color=color, lw=1.2, ls='--', alpha=0.85)
                elif ch == y:
                    self._hlines[f'{gid}:lo'] = self.ax.axhline(
                        lo, color=color, lw=1.2, ls='--', alpha=0.85)
                    self._hlines[f'{gid}:hi'] = self.ax.axhline(
                        hi, color=color, lw=1.2, ls='--', alpha=0.85)
            elif k == 'rect':
                if g.get('x_channel') == x and g.get('y_channel') == y:
                    x0, x1 = sorted([float(g['x0']), float(g['x1'])])
                    y0, y1 = sorted([float(g['y0']), float(g['y1'])])
                    patch = mpatches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        fill=False, edgecolor=color, linewidth=1.3,
                        linestyle='--', alpha=0.9)
                    self.ax.add_patch(patch)
                    self._shape_artists[gid] = patch
            elif k == 'polygon':
                if g.get('x_channel') == x and g.get('y_channel') == y:
                    verts = np.asarray(g['vertices'], dtype=float)
                    if verts.ndim == 2 and verts.shape[1] == 2 and len(verts) >= 3:
                        patch = mpatches.Polygon(
                            verts, closed=True, fill=False,
                            edgecolor=color, linewidth=1.3,
                            linestyle='--', alpha=0.9)
                        self.ax.add_patch(patch)
                        self._shape_artists[gid] = patch
            elif k == 'ellipsoid':
                if g.get('x_channel') == x and g.get('y_channel') == y:
                    patch = self._ellipse_patch(g, color)
                    if patch is not None:
                        self.ax.add_patch(patch)
                        self._shape_artists[gid] = patch
                        # Rotation-grip marker just beyond the rim, so the
                        # Edit tool's rotate handle is discoverable.
                        geom = self._ellipse_geom(g)
                        if geom is not None:
                            _c, _inv, _r0, (hx, hy) = geom
                            (handle,) = self.ax.plot(
                                [hx], [hy], marker='o', markersize=5,
                                markerfacecolor=color, markeredgecolor='white',
                                markeredgewidth=0.6, linestyle='None',
                                alpha=0.9, zorder=6)
                            self._shape_artists[f'{gid}:rot'] = handle

    def _ellipse_patch(self, gate, color):
        """Build a dashed matplotlib Ellipse patch for an ellipsoid gate,
        or None if the geometry is degenerate."""
        import matplotlib.patches as mpatches
        params = self._ellipse_params(gate)
        if params is None:
            return None
        cx, cy, width, height, angle = params
        return mpatches.Ellipse(
            (cx, cy), width, height, angle=angle,
            fill=False, edgecolor=color, linewidth=1.3,
            linestyle='--', alpha=0.9)

    def _hit_test(self, event):
        """Find a draggable handle near the cursor. Priority order:
          1. 1D axis lines (threshold / interval)
          2. Quadrant origin / x-axis / y-axis (so the central cross of a
             4-rect quadrant set wins over each rect's individual corner)
          3. Rect corners, then edges (non-quadrant rects)
          4. Polygon vertices, then edges (edge hits will INSERT a new
             vertex at the click point in _on_press).
        Returns a drag-state tuple or None. Tolerance is axis-fraction
        (2.5% of the current view span on each axis)."""
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return None
        xl, xh = self.ax.get_xlim()
        yl, yh = self.ax.get_ylim()
        tol = 0.025
        span_x = max(xh - xl, 1e-9)
        span_y = max(yh - yl, 1e-9)
        tol_x = span_x * tol
        tol_y = span_y * tol
        cx, cy = float(event.xdata), float(event.ydata)

        # 1) 1D axis lines (existing behaviour)
        for key, line in self._vlines.items():
            val = line.get_xdata()[0]
            if abs(cx - val) < tol_x:
                return ('v', key)
        for key, line in self._hlines.items():
            val = line.get_ydata()[0]
            if abs(cy - val) < tol_y:
                return ('h', key)

        x_ch = self._resolve_channel(self.x_combo.get())
        y_ch = self._resolve_channel(self.y_combo.get())
        # Only 2-D shapes actually DRAWN are hit-testable — keyed by gid in
        # _shape_artists by _draw_gates. In histogram mode (or for a degenerate
        # <3-vertex polygon) nothing 2-D is drawn, so its invisible handles must
        # not be grabbable (else a drag silently corrupts the gate's bounds).
        drawn = getattr(self, '_shape_artists', {})

        # 2) Quadrant-set handles. Drag the centre to move both axes;
        #    drag a divider line to move only that axis.
        seen_qs = set()
        for _gid, g in self._gates.items():
            qs = g.get('quad_set')
            if not qs or qs in seen_qs:
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            seen_qs.add(qs)
            x_o = g.get('quad_origin_x')
            y_o = g.get('quad_origin_y')
            if x_o is None or y_o is None:
                continue
            x_o, y_o = float(x_o), float(y_o)
            if abs(cx - x_o) < tol_x and abs(cy - y_o) < tol_y:
                return ('quad_origin', qs)
            if abs(cx - x_o) < tol_x:
                return ('quad_x', qs)
            if abs(cy - y_o) < tol_y:
                return ('quad_y', qs)

        # 3) Non-quadrant rect corners / edges.
        for gid, g in self._gates.items():
            if g.get('kind') != 'rect' or g.get('quad_set'):
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            if gid not in drawn:
                continue
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            for cn, hx, hy in (('bl', x0, y0), ('br', x1, y0),
                                ('tl', x0, y1), ('tr', x1, y1)):
                if abs(cx - hx) < tol_x and abs(cy - hy) < tol_y:
                    return ('rect_corner', gid, cn)
            if abs(cy - y0) < tol_y and x0 <= cx <= x1:
                return ('rect_edge', gid, 'bottom')
            if abs(cy - y1) < tol_y and x0 <= cx <= x1:
                return ('rect_edge', gid, 'top')
            if abs(cx - x0) < tol_x and y0 <= cy <= y1:
                return ('rect_edge', gid, 'left')
            if abs(cx - x1) < tol_x and y0 <= cy <= y1:
                return ('rect_edge', gid, 'right')

        # 4) Polygon vertices, then edges.
        for gid, g in self._gates.items():
            if g.get('kind') != 'polygon':
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            if gid not in drawn:
                continue
            verts = g.get('vertices') or []
            for i, (vx, vy) in enumerate(verts):
                if abs(cx - float(vx)) < tol_x and abs(cy - float(vy)) < tol_y:
                    return ('poly_vertex', gid, i)
            n = len(verts)
            for i in range(n):
                ax, ay = float(verts[i][0]),       float(verts[i][1])
                bx, by = float(verts[(i+1) % n][0]), float(verts[(i+1) % n][1])
                if self._point_segment_dist(cx, cy, ax, ay, bx, by,
                                            span_x, span_y) < tol:
                    return ('poly_edge', gid, i)

        # 4b) Ellipsoid: rotation handle, then rim (resize), then interior
        #     (translate). Mahalanobis radius md = sqrt((p-µ)ᵀ Σ⁻¹ (p-µ));
        #     the rim is md == sqrt(distance_sq).
        for gid, g in self._gates.items():
            if g.get('kind') != 'ellipsoid':
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            if gid not in drawn:
                continue
            geom = self._ellipse_geom(g)
            if geom is None:
                continue
            (mx, my), inv, r0, (hx, hy) = geom
            # Rotation handle (small marker offset beyond the rim).
            if abs(cx - hx) < tol_x and abs(cy - hy) < tol_y:
                return ('ellipse_rotate', gid)
            d = np.array([cx - mx, cy - my])
            md = float(np.sqrt(max(d @ inv @ d, 0.0)))
            if 0.6 * r0 <= md <= 1.5 * r0:
                return ('ellipse_rim', gid)
            # interior handled in section 5

        # 5) Interior translate. Lowest priority so corners/edges/vertices
        #    above still win when the cursor is near a handle. Quadrant
        #    rects are excluded — the quad-origin / quad-x / quad-y
        #    handles already cover translation for those.
        for gid, g in self._gates.items():
            if g.get('quad_set'):
                continue
            if g.get('x_channel') != x_ch or g.get('y_channel') != y_ch:
                continue
            if gid not in drawn:
                continue
            kind = g.get('kind')
            if kind == 'rect':
                x0, x1 = sorted([float(g['x0']), float(g['x1'])])
                y0, y1 = sorted([float(g['y0']), float(g['y1'])])
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    return ('rect_translate', gid)
            elif kind == 'polygon':
                verts = g.get('vertices') or []
                if len(verts) >= 3:
                    try:
                        from matplotlib.path import Path as _MplPath
                        if _MplPath(np.asarray(verts, dtype=float)
                                    ).contains_point((cx, cy)):
                            return ('poly_translate', gid)
                    except Exception:
                        pass
            elif kind == 'ellipsoid':
                geom = self._ellipse_geom(g)
                if geom is not None:
                    (mx, my), inv, r0, _h = geom
                    d = np.array([cx - mx, cy - my])
                    md = float(np.sqrt(max(d @ inv @ d, 0.0)))
                    if md < 0.6 * r0:
                        return ('ellipse_translate', gid)
        return None

    def _delete_polygon_vertex(self, gid, vi):
        """Delete vertex ``vi`` from polygon ``gid``. Refuses to drop
        the polygon below 3 vertices (degenerate)."""
        g = self._gates.get(gid)
        if g is None or g.get('kind') != 'polygon':
            return
        verts = list(g.get('vertices') or [])
        if len(verts) <= 3:
            return  # would degenerate
        if not (0 <= vi < len(verts)):
            return
        del verts[vi]
        g['vertices'] = verts
        self._redraw_only_gates()
        self._refresh_gate_list()

    def _on_press(self, event):
        btn = getattr(event, 'button', 1)
        # The backgate legend swallows clicks over it (so a stray click can't
        # create/select a gate underneath) and starts a drag on its header.
        if self._legend_press(event):
            return
        key = (getattr(event, 'key', None) or '').lower()
        is_shift = 'shift' in key
        is_alt   = 'alt'   in key

        # ── Zoom-to tool: left-drag draws a zoom rectangle (gating is off). ─
        if (self._zoom_mode and btn == 1 and event.inaxes is self.ax
                and event.xdata is not None):
            self._zoom_start = (event.xdata, event.ydata)
            return

        # ── Middle-button drag = pan (records a fixed press-time reference
        #    so the pan can't feed back on itself). ─────────────────────────
        if btn == 2 and event.inaxes is self.ax:
            try:
                bbox = self.ax.get_window_extent()
                self._pan_start = (event.x, event.y, self.ax.get_xlim(),
                                   self.ax.get_ylim(), bbox.width, bbox.height)
                self.canvas.get_tk_widget().config(cursor='fleur')
            except Exception:
                self._pan_start = None
            return

        # ── Right-click gestures (work in any tool) ───────────────────────
        # Smart "±vertex" — on a polygon vertex it deletes, on a polygon
        # edge it inserts. Anywhere else the right-click is a no-op so it
        # doesn't accidentally pan / open a menu / etc.
        if btn == 3:
            hit = self._hit_test(event)
            if hit:
                kind = hit[0]
                if kind == 'poly_vertex':
                    _, gid, vi = hit                       # type: ignore[misc]
                    self._checkpoint()
                    self._delete_polygon_vertex(gid, int(vi))
                    return
                if kind == 'poly_edge':
                    _, gid, edge_idx = hit                 # type: ignore[misc]
                    self._checkpoint()
                    self._insert_polygon_vertex(
                        gid, int(edge_idx) + 1,
                        event.xdata, event.ydata)
                    return
            return

        # ── Alt+left-click: drop a new vertex in the polygon under the
        #    cursor (or nearest polygon when no polygon contains it). ─────
        if (btn == 1 and is_alt and event.inaxes is self.ax
                and event.xdata is not None and event.ydata is not None):
            gid = self._polygon_under_point(event.xdata, event.ydata)
            if gid is not None:
                self._checkpoint()
                g = self._gates[gid]
                verts = list(g.get('vertices') or [])
                verts.append([float(event.xdata), float(event.ydata)])
                g['vertices'] = verts
                self._redraw_only_gates()
                self._refresh_gate_list()
                return

        # 1) Hit-test for any draggable gate handle (line, corner, edge,
        #    vertex, quadrant centre/axis). If something hits, start a
        #    drag and pick a cursor that hints what the gesture does.
        hit = self._hit_test(event)

        # ── Shift+left-drag: force translate-the-whole-gate even if the
        #    click landed on a vertex / edge / corner. Only meaningful
        #    for gates that have a translate mode (rect, polygon). ──────
        if btn == 1 and is_shift and hit:
            gid = self._gid_from_hit(hit)
            if gid and gid in self._gates:
                g = self._gates[gid]
                k = g.get('kind')
                if k == 'rect':
                    hit = ('rect_translate', gid)
                elif k == 'polygon':
                    hit = ('poly_translate', gid)
                elif k == 'ellipsoid':
                    hit = ('ellipse_translate', gid)
                # threshold/interval/quadrant have no translate kind —
                # fall through to existing behaviour.

        if hit:
            kind = hit[0]
            # poly_edge is special: insert a new vertex at the click
            # position and become a poly_vertex drag immediately. That
            # matches FlowJo's "click an edge to add a point and shape it".
            if kind == 'poly_edge':
                _, gid, edge_idx = hit   # type: ignore[misc]
                g = self._gates.get(gid)
                if g is not None and event.xdata is not None and event.ydata is not None:
                    self._checkpoint()
                    verts = list(g.get('vertices') or [])
                    new_i = int(edge_idx) + 1
                    verts.insert(new_i, [float(event.xdata),
                                         float(event.ydata)])
                    g['vertices'] = verts
                    self._drag_state = ('poly_vertex', gid, new_i)
                    self._redraw_only_gates()
                    self.canvas.get_tk_widget().config(cursor='fleur')
                return
            # A handle was grabbed — snapshot the pre-drag geometry so the
            # whole drag is one undo step (motion mutates in place).
            self._checkpoint()
            self._drag_state = hit
            # Translate / resize / rotate drags cache the press anchor +
            # a deep copy of the gate at start, so motion applies the
            # delta to the *original* geometry — no rounding drift over
            # the drag.
            if kind in ('rect_translate', 'poly_translate',
                        'ellipse_translate', 'ellipse_rim', 'ellipse_rotate'):
                import copy as _copy
                _, gid_t = hit                      # type: ignore[misc]
                g_t = self._gates.get(gid_t)
                if g_t is None or event.xdata is None or event.ydata is None:
                    self._drag_state = None
                    return
                ctx = {
                    'anchor_x': float(event.xdata),
                    'anchor_y': float(event.ydata),
                    'orig':     _copy.deepcopy(g_t),
                }
                # Rotation needs the press direction from the centre so
                # motion can measure the swept angle.
                if kind == 'ellipse_rotate':
                    om = g_t.get('mean', [0.0, 0.0])
                    ctx['rot_anchor_angle'] = float(np.arctan2(
                        float(event.ydata) - float(om[1]),
                        float(event.xdata) - float(om[0])))
                self._drag_translate_ctx = ctx
            # Cursor hint per drag kind.
            if kind in ('v', 'quad_x'):
                cur = 'sb_h_double_arrow'
            elif kind in ('h', 'quad_y'):
                cur = 'sb_v_double_arrow'
            elif kind == 'ellipse_rotate':
                cur = 'exchange'
            elif kind == 'rect_edge':
                cur = ('sb_v_double_arrow' if hit[2] in ('top', 'bottom')  # type: ignore[misc]
                       else 'sb_h_double_arrow')
            elif kind in ('rect_translate', 'poly_translate'):
                cur = 'fleur'
            else:                         # rect_corner / poly_vertex / quad_origin
                cur = 'fleur'
            self.canvas.get_tk_widget().config(cursor=cur)
            return

        if event.inaxes is not self.ax:
            return

        # 2) Region-shape tools (rect/polygon/lasso) are handled by their
        #    matplotlib Selector — we don't intercept those clicks here.
        tool = (self.gate_tool_var.get() if hasattr(self, 'gate_tool_var')
                else 'quadrant')
        if tool != 'quadrant':
            return

        # 3) Quadrant tool: double-click or shift-click → add gate(s).
        is_add = (event.dblclick
                  or 'shift' in (getattr(event, 'key', None) or '').lower())
        if not is_add:
            return
        x = self._resolve_channel(self.x_combo.get())
        y = self._resolve_channel(self.y_combo.get())
        mode = self.mode_var.get()
        if mode == 'histogram':
            # 1D: single threshold gate on the click x position.
            if x and event.xdata is not None:
                self._add_gate_multi({'kind': 'threshold', 'channel': x,
                                      'value': float(event.xdata)})
        else:
            # 2D: emit 4 quadrant rect gates centred on the click. Each
            # quadrant is a separate rect so the user can toggle / colour
            # them independently. Bounds clip to the current axis viewport
            # at the moment of the click.
            if (not x or not y
                    or event.xdata is None or event.ydata is None):
                return
            xc, yc = float(event.xdata), float(event.ydata)
            try:
                xl, xh = self.ax.get_xlim()
                yl, yh = self.ax.get_ylim()
            except Exception:
                xl, xh = xc - 1.0, xc + 1.0
                yl, yh = yc - 1.0, yc + 1.0
            parent = self._selected_gate_id()
            # Shared identifier so the 4 quadrant rects can be moved
            # together (drag the origin or one of the dividing axes) at
            # hit-test / motion time.
            self._quad_set_seq += 1
            qs_id = f'qs{self._quad_set_seq}'
            for label, x0, x1, y0, y1 in [
                    ('Q++ (x>, y>)', xc, xh, yc, yh),
                    ('Q+- (x>, y<)', xc, xh, yl, yc),
                    ('Q-+ (x<, y>)', xl, xc, yc, yh),
                    ('Q-- (x<, y<)', xl, xc, yl, yc)]:
                self._add_gate_multi({'kind': 'rect',
                                      'x_channel': x, 'y_channel': y,
                                      'x0': float(x0), 'x1': float(x1),
                                      'y0': float(y0), 'y1': float(y1),
                                      'label': label,
                                      'quad_set':       qs_id,
                                      'quad_origin_x':  float(xc),
                                      'quad_origin_y':  float(yc)},
                                     parent_id=parent)
        self._schedule_replot(0)

    def _on_scroll(self, event):
        """Mouse-wheel zoom around the cursor (no effect off the axes)."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        f = 1 / 1.2 if event.button == 'up' else 1.2
        cx, cy = event.xdata, event.ydata
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(cx + (xl[0] - cx) * f, cx + (xl[1] - cx) * f)
        self.ax.set_ylim(cy + (yl[0] - cy) * f, cy + (yl[1] - cy) * f)
        self.canvas.draw_idle()

    def _on_release(self, event):
        # Finish a zoom-to drag: set the limits to the rectangle (ignore a
        # zero-area click).
        if self._zoom_mode and self._zoom_start is not None:
            x0, y0 = self._zoom_start
            self._zoom_start = None
            if self._zoom_rect_artist is not None:
                try:
                    self._zoom_rect_artist.remove()
                except Exception:
                    pass
                self._zoom_rect_artist = None
            if (getattr(event, 'xdata', None) is not None
                    and abs(event.xdata - x0) > 1e-9
                    and abs(event.ydata - y0) > 1e-9):
                self.ax.set_xlim(min(x0, event.xdata), max(x0, event.xdata))
                self.ax.set_ylim(min(y0, event.ydata), max(y0, event.ydata))
            self.canvas.draw_idle()
            return
        if self._pan_start is not None:
            self._pan_start = None
            try:
                self.canvas.get_tk_widget().config(cursor='')
            except Exception:
                pass
            return
        if self._legend_drag is not None:
            self._legend_drag = None
            return
        was_dragging = self._drag_state is not None
        self._drag_state = None
        self._drag_translate_ctx = None    # always clear; cheap if unset
        self.canvas.get_tk_widget().config(cursor='')
        # Refresh the gate tree once at end-of-drag so the description
        # text (rect bounds, polygon vert count, etc.) reflects the new
        # geometry. We skipped this during motion for performance.
        if was_dragging or event is None or event.inaxes is None:
            self._refresh_gate_list()

    def _on_motion(self, event):
        # Zoom-to rubber-band rectangle while dragging in zoom mode.
        if self._zoom_mode and self._zoom_start is not None:
            if getattr(event, 'xdata', None) is None or event.inaxes is not self.ax:
                return
            x0, y0 = self._zoom_start
            ex, ey = float(event.xdata), float(event.ydata)
            if self._zoom_rect_artist is not None:
                try:
                    self._zoom_rect_artist.remove()
                except Exception:
                    pass
            from matplotlib.patches import Rectangle
            self._zoom_rect_artist = Rectangle(
                (min(x0, ex), min(y0, ey)),
                abs(ex - x0), abs(ey - y0), fill=False,
                edgecolor='#4f8cff', lw=1.2, ls='--', zorder=20)
            self.ax.add_patch(self._zoom_rect_artist)
            self.canvas.draw_idle()
            return
        # Middle-button pan in progress: shift the limits by the pixel delta
        # mapped through the press-time scale (stable, no feedback).
        if self._pan_start is not None:
            if getattr(event, 'x', None) is None or event.y is None:
                return
            ex, ey, xl, yl, bw, bh = self._pan_start
            if bw and bh:
                dx = (event.x - ex) / bw * (xl[1] - xl[0])
                dy = (event.y - ey) / bh * (yl[1] - yl[0])
                self.ax.set_xlim(xl[0] - dx, xl[1] - dx)
                self.ax.set_ylim(yl[0] - dy, yl[1] - dy)
                self.canvas.draw_idle()
            return
        # Dragging the backgate legend by its header → move its anchor and do a
        # cheap legend-only redraw (no full scatter re-render).
        if self._legend_drag is not None:
            d = self._legend_drag
            if getattr(event, 'x', None) is not None and event.y is not None:
                try:
                    inv = self.ax.transAxes.inverted()
                    f0 = inv.transform((d['sx'], d['sy']))
                    f1 = inv.transform((event.x, event.y))
                    ax0, ay0 = d['anchor']
                    nx = min(0.92, max(0.0, ax0 + (f1[0] - f0[0])))
                    ny = min(1.0, max(0.10, ay0 + (f1[1] - f0[1])))
                    self._backgate_legend_anchor = (nx, ny)
                    self._reposition_backgate_legend()
                except Exception:
                    pass
            return
        if self._drag_state is None or event.inaxes is not self.ax:
            return
        kind = self._drag_state[0]
        cx = event.xdata
        cy = event.ydata
        if cx is None and cy is None:
            return

        # ── 1D axis lines (existing) ─────────────────────────────────────
        if kind in ('v', 'h'):
            _, key = self._drag_state   # type: ignore[misc]
            if ':' in key:
                gid, side = key.split(':', 1)
            else:
                gid, side = key, None
            g = self._gates.get(gid)
            if g is None:
                return
            if kind == 'v' and cx is not None:
                new_val = float(cx)
                if g['kind'] == 'threshold':
                    g['value'] = new_val
                elif g['kind'] == 'interval':
                    g['lo' if side == 'lo' else 'hi'] = new_val
                    if g['lo'] > g['hi']:
                        g['lo'], g['hi'] = g['hi'], g['lo']
                line = self._vlines.get(key)
                if line is not None:
                    line.set_xdata([new_val, new_val])
            elif kind == 'h' and cy is not None:
                new_val = float(cy)
                if g['kind'] == 'threshold':
                    g['value'] = new_val
                elif g['kind'] == 'interval':
                    g['lo' if side == 'lo' else 'hi'] = new_val
                    if g['lo'] > g['hi']:
                        g['lo'], g['hi'] = g['hi'], g['lo']
                line = self._hlines.get(key)
                if line is not None:
                    line.set_ydata([new_val, new_val])
            self._refresh_gate_list()
            self.canvas.draw_idle()
            if self.apply_gates_var.get():
                self._schedule_replot(120)
            return

        # ── Rect corner / edge ───────────────────────────────────────────
        if kind in ('rect_corner', 'rect_edge'):
            _, gid, which = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            if g is None or g.get('kind') != 'rect':
                return
            x0, x1 = sorted([float(g['x0']), float(g['x1'])])
            y0, y1 = sorted([float(g['y0']), float(g['y1'])])
            if kind == 'rect_corner' and cx is not None and cy is not None:
                if which == 'bl':   x0, y0 = float(cx), float(cy)
                elif which == 'br': x1, y0 = float(cx), float(cy)
                elif which == 'tl': x0, y1 = float(cx), float(cy)
                elif which == 'tr': x1, y1 = float(cx), float(cy)
            elif kind == 'rect_edge':
                if which == 'top'    and cy is not None: y1 = float(cy)
                elif which == 'bottom' and cy is not None: y0 = float(cy)
                elif which == 'left'   and cx is not None: x0 = float(cx)
                elif which == 'right'  and cx is not None: x1 = float(cx)
            if x0 > x1: x0, x1 = x1, x0
            if y0 > y1: y0, y1 = y1, y0
            g['x0'], g['x1'], g['y0'], g['y1'] = x0, x1, y0, y1

        # ── Polygon vertex ───────────────────────────────────────────────
        elif kind == 'poly_vertex' and cx is not None and cy is not None:
            _, gid, vi = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            if g is None or g.get('kind') != 'polygon':
                return
            verts = list(g.get('vertices') or [])
            vi = int(vi)
            if 0 <= vi < len(verts):
                verts[vi] = [float(cx), float(cy)]
                g['vertices'] = verts

        # ── Translate the whole shape (rect / polygon / ellipsoid) ───────
        elif kind in ('rect_translate', 'poly_translate', 'ellipse_translate'):
            if cx is None or cy is None:
                return
            _, gid = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            ctx = self._drag_translate_ctx or {}
            orig = ctx.get('orig')
            if g is None or orig is None:
                return
            dx = float(cx) - float(ctx['anchor_x'])
            dy = float(cy) - float(ctx['anchor_y'])
            if kind == 'rect_translate' and g.get('kind') == 'rect':
                g['x0'] = float(orig['x0']) + dx
                g['x1'] = float(orig['x1']) + dx
                g['y0'] = float(orig['y0']) + dy
                g['y1'] = float(orig['y1']) + dy
            elif kind == 'poly_translate' and g.get('kind') == 'polygon':
                g['vertices'] = [[float(v[0]) + dx, float(v[1]) + dy]
                                 for v in orig.get('vertices', [])]
            elif kind == 'ellipse_translate' and g.get('kind') == 'ellipsoid':
                om = orig.get('mean', [0.0, 0.0])
                g['mean'] = [float(om[0]) + dx, float(om[1]) + dy]

        # ── Ellipsoid resize (drag the rim) ──────────────────────────────
        # Uniformly scale the covariance so the rim passes through the
        # cursor: if the cursor sits at Mahalanobis radius md (under the
        # ORIGINAL Σ), scaling Σ by (md/r0)² moves the rim onto it.
        elif kind == 'ellipse_rim' and cx is not None and cy is not None:
            _, gid = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            ctx = self._drag_translate_ctx or {}
            orig = ctx.get('orig')
            if g is None or orig is None or g.get('kind') != 'ellipsoid':
                return
            mean = np.asarray(orig['mean'], dtype=float)
            cov0 = np.asarray(orig['cov'], dtype=float)
            dist_sq = float(orig.get('distance_sq', 4.0))
            try:
                inv0 = np.linalg.inv(cov0)
            except np.linalg.LinAlgError:
                return
            d = np.array([float(cx) - mean[0], float(cy) - mean[1]])
            md_sq = float(d @ inv0 @ d)
            r0_sq = max(dist_sq, 1e-12)
            scale = md_sq / r0_sq                 # = (md/r0)²
            scale = max(scale, 1e-6)              # never collapse to zero
            g['cov'] = (cov0 * scale).tolist()

        # ── Ellipsoid rotate (drag the top handle) ───────────────────────
        # Rotate Σ by the angle swept between the handle's original
        # direction and the cursor direction, both measured from centre.
        elif kind == 'ellipse_rotate' and cx is not None and cy is not None:
            _, gid = self._drag_state   # type: ignore[misc]
            g = self._gates.get(gid)
            ctx = self._drag_translate_ctx or {}
            orig = ctx.get('orig')
            if g is None or orig is None or g.get('kind') != 'ellipsoid':
                return
            mean = np.asarray(orig['mean'], dtype=float)
            cov0 = np.asarray(orig['cov'], dtype=float)
            a0 = float(ctx.get('rot_anchor_angle', 0.0))
            a1 = float(np.arctan2(float(cy) - mean[1], float(cx) - mean[0]))
            dtheta = a1 - a0
            c, s = np.cos(dtheta), np.sin(dtheta)
            rot = np.array([[c, -s], [s, c]])
            g['cov'] = (rot @ cov0 @ rot.T).tolist()

        # ── Quadrant centre / dividers ───────────────────────────────────
        elif kind == 'quad_origin' and cx is not None and cy is not None:
            self._update_quad_set(self._drag_state[1],
                                  new_x=float(cx), new_y=float(cy))
        elif kind == 'quad_x' and cx is not None:
            self._update_quad_set(self._drag_state[1],
                                  new_x=float(cx), new_y=None)
        elif kind == 'quad_y' and cy is not None:
            self._update_quad_set(self._drag_state[1],
                                  new_x=None, new_y=float(cy))
        else:
            return

        # All non-line drags need a full overlay rebuild (the Patch
        # geometry has to be redrawn from the mutated gate dict). Tree
        # text changes for rect/polygon are unlikely to be interesting
        # mid-drag, so skip the heavier _refresh_gate_list — release
        # handler does that.
        self._redraw_only_gates()
        if self.apply_gates_var.get():
            self._schedule_replot(120)

    def _clear_selected_gate(self):
        """Clear button. Removes gates based on the current selection — the
        samples themselves are never removed (use Remove for that):
          • gate row   → that gate AND all its descendants (cascade)
          • sample row → every gate on that sample (the sample stays)
          • trial row  → every gate on all samples in that trial
        Handles a multi-row selection as a single undo step."""
        sel = self.gate_tv.selection()
        if not sel:
            self.status_var.set("Select a gate, sample, or trial to clear.")
            return
        gate_targets = []        # (sample_name, gid) → cascade
        sample_targets = []      # sample_name → all gates
        for iid in sel:
            p = self._parse_iid(iid)
            if p is None:
                continue
            if p[0] == 'gate':
                gate_targets.append((p[1], p[2]))
            elif p[0] == 'sample':
                sample_targets.append(p[1])
            elif p[0] == 'trial':
                sample_targets.extend(self._trial_members(p[1]))
        if not gate_targets and not sample_targets:
            self.status_var.set("Select a gate, sample, or trial to clear.")
            return
        self._checkpoint()
        removed = 0
        # Whole-sample clears first; a gate caught by both (its sample is also
        # selected) is then a no-op rather than a double-count.
        for name in dict.fromkeys(sample_targets):
            removed += self._purge_all_gates(name)
        for name, gid in gate_targets:
            removed += self._purge_gate_subtree(name, gid)
        self._refresh_gate_list()
        self._schedule_replot(0)
        if removed:
            self._audit('gate.remove', n_gates=removed,
                        samples=sorted({n for n in sample_targets}
                                       | {n for n, _ in gate_targets}))
        self.status_var.set(
            f"Cleared {removed} gate(s) — Ctrl+Z to undo." if removed
            else "No gates to clear.")

    def _remove_gate_cascade_in(self, sample_name, gid):
        """Cascade-delete a gate + its descendants from any sample's tree
        (not just the active one). Undoable."""
        self._checkpoint()
        self._purge_gate_subtree(sample_name, gid)

    def _purge_gate_subtree(self, sample_name, gid):
        """Remove ``gid`` and every descendant from ``sample_name``'s gate tree,
        in place (the caller owns the undo checkpoint). Mutating the existing
        dict/list — not rebinding them — keeps the active sample's ``self._gates``
        / ``_gate_id_order`` references valid. Returns the number removed."""
        target_gates = self._sample_gates.get(sample_name, {})
        order = self._sample_gate_order.get(sample_name, [])
        to_remove = [gid]
        seen = set()
        while to_remove:
            cur = to_remove.pop()
            if cur in seen or cur not in target_gates:
                continue
            seen.add(cur)
            for other_gid, og in target_gates.items():
                if og.get('parent_id') == cur:
                    to_remove.append(other_gid)
        for victim in seen:
            target_gates.pop(victim, None)
            if victim in order:
                order.remove(victim)
        return len(seen)

    def _purge_all_gates(self, sample_name):
        """Remove every gate from ``sample_name`` in place (caller owns the undo
        checkpoint), leaving the sample itself loaded. Clears the dict/list in
        place so the active sample's ``self._gates`` / ``_gate_id_order``
        references stay valid. Returns the number removed."""
        gates = self._sample_gates.get(sample_name)
        if not gates:
            return 0
        n = len(gates)
        gates.clear()
        order = self._sample_gate_order.get(sample_name)
        if order is not None:
            order.clear()
        return n

    def _clear_all(self):
        """Clear gates from EVERY loaded sample, leaving the samples themselves
        intact. Auto-clean gates are PRESERVED by default (they're the cleaning
        foundation) — a checkbox in the confirm dialog opts into clearing them
        too. Confirmed (bulk wipe) but undoable. To remove samples, use Remove.
        """
        total = sum(len(g) for g in self._sample_gates.values())
        if total == 0:
            self.status_var.set("No gates to clear.")
            return
        n = sum(1 for g in self._sample_gates.values() if g)
        n_ac = sum(1 for gates in self._sample_gates.values()
                   for g in gates.values() if g.get('kind') == 'autoclean')
        include_ac = self._ask_clear_all(total, n, n_ac)
        if include_ac is None:                 # cancelled
            return
        self._checkpoint()
        removed = 0
        for name in list(self._samples):
            gates = self._sample_gates.get(name, {})
            order = self._sample_gate_order.get(name, [])
            victims = [gid for gid, g in gates.items()
                       if include_ac or g.get('kind') != 'autoclean']
            for gid in victims:
                gates.pop(gid, None)
                if gid in order:
                    order.remove(gid)
                removed += 1
        self._refresh_gate_list()
        self._schedule_replot(0)
        kept = (" (auto-clean kept)" if n_ac and not include_ac else "")
        self.status_var.set(
            f"Cleared {removed} gate(s) from all samples{kept}; samples kept.")
