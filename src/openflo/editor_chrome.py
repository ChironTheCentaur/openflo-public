"""Window chrome: docking, pop-out, titlebar theming, and DPI scaling.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import sys

from .editor_base import EditorMixin
from .prefs import write_pref
from .theme import _DARK_MODES, current_palette


class ChromeMixin(EditorMixin):
    """Dock / pop-out the side and workspace panels, theme the title bar, and rescale chrome with the OS DPI / theme."""

    def _freeze_plot_redraw(self, _event=None):
        """While a pane sash is being dragged, suppress the matplotlib canvas's
        per-pixel re-raster (the source of the resize lag) by no-op'ing
        draw_idle. One real redraw happens on release (_thaw_plot_redraw)."""
        if getattr(self, '_plot_frozen', False):
            return
        try:
            self._plot_frozen = True
            self._saved_draw_idle = self.canvas.draw_idle
            self.canvas.draw_idle = lambda *a, **k: None
        except Exception:
            self._plot_frozen = False

    def _thaw_plot_redraw(self, _event=None):
        """Sash released — restore the canvas and do a single clean replot at
        the final size (correct margins/layout)."""
        if not getattr(self, '_plot_frozen', False):
            return
        self._plot_frozen = False
        try:
            self.canvas.draw_idle = self._saved_draw_idle
        except Exception:
            pass
        self._schedule_replot(0)

    def _on_chrome_configure(self, event):
        """Debounced reaction to the main window resizing — rescale the ttk
        control font so the control rows fit smaller screens."""
        if event.widget is not self:
            return
        try:
            if self._chrome_resize_after:
                self.after_cancel(self._chrome_resize_after)
            self._chrome_resize_after = self.after(150, self._apply_chrome_scale)
        except Exception:
            pass

    def _apply_chrome_scale(self, force=False):
        """Pick a ttk control-font size from the window width (stepped, so the
        layout doesn't churn) and apply it to the base ttk style. The plot,
        data and pop-up figures are unaffected — this is chrome only."""
        self._chrome_resize_after = None
        try:
            w = self.winfo_width()
            if w <= 1:
                return
            size = (10 if w >= 1380 else 9 if w >= 1200
                    else 8 if w >= 1040 else 7)
            if not force and size == self._chrome_font_size:
                return
            self._chrome_font_size = size
            import tkinter.font as tkfont
            from tkinter import ttk
            fams = set(tkfont.families(self))
            fam = 'Segoe UI' if 'Segoe UI' in fams else 'TkDefaultFont'
            st = ttk.Style(self)
            st.configure('.', font=(fam, size))
            st.configure('Treeview.Heading', font=(fam, size, 'bold'))
        except Exception:
            pass

    def _ensure_workspace_panel(self):
        """Build the Pipeline Workspace panel into its host on first reveal.
        Deferred from __init__ because it costs ~100 ms to construct; a session
        that never opens the workspace shouldn't pay that at startup. Idempotent.
        """
        if getattr(self, '_workspace_panel', None) is not None:
            return self._workspace_panel
        from .workspace import WorkspacePanel
        self._workspace_panel = WorkspacePanel(self._ws_host, editor=self,
                                               on_before_change=self._checkpoint)
        self._workspace_panel.pack(fill='both', expand=True)
        return self._workspace_panel

    def _workspace_open(self):
        """True if there's a live drop target: the docked pane is shown, or a
        workspace tab has been popped out into its own window."""
        panel = getattr(self, '_workspace_panel', None)
        if panel is None:
            return False
        try:
            return bool(getattr(self, '_workspace_shown', False)) or panel.popped_count() > 0
        except Exception:
            return False

    def _theme_tree_tags(self, pal=None):
        """Recolour the theme-dependent gate-tree tags from the palette."""
        pal = pal or current_palette()
        for tag in ('off', 'loading', 'subgroup_row'):
            try:
                self.gate_tv.tag_configure(tag, foreground=pal['muted'])
            except Exception:
                pass

    def _set_theme(self):
        """View → Theme: switch the chrome palette live, persist the choice,
        and recolour the bits Tk doesn't restyle on its own. The plot stays
        white under Light/Dark; 'Midnight' darkens the plot too (_replot →
        _apply_plot_theme)."""
        from .gui import apply_theme  # lazy: avoid import cycle with gui
        mode = self._theme_var.get()
        pal = apply_theme(self, mode)
        write_pref('theme', mode)
        try:
            self.configure(bg=pal['bg'])  # type: ignore[call-arg]
            self._left_host.configure(bg=pal['bg'])
            self._ws_host.configure(bg=pal['bg'])
            self.canvas.get_tk_widget().configure(bg=pal['bg'])
        except Exception:
            pass
        self._theme_tree_tags(pal)
        self._theme_menubar(pal)
        self._apply_titlebar_theme()
        self._refresh_gate_list()       # re-tags sample/subgroup fg from palette
        self._apply_chrome_scale(force=True)   # apply_theme reset the base font
        self._schedule_replot(0)
        self.status_var.set(f"{mode.capitalize()} theme applied.")

    def _toggle_left_popout(self):
        """Pop the Samples & Gates panel out into its own window (or re-dock
        it). Uses Tk 'wm manage' on the panel's tk.Frame host — the live tree
        and all its bindings move with the window, no rebuild needed."""
        host = self._left_host
        if self._left_popped:
            self._redock_left()
            return
        try:
            w = max(host.winfo_width(), 320)
            h = max(host.winfo_height(), 520)
            self._main_paned.forget(host)
            self.tk.call('wm', 'manage', host)
            self.tk.call('wm', 'title', host, 'OpenFlo — Samples & Gates')
            self.tk.call('wm', 'geometry', host, f'{w}x{h}')
            self.tk.call('wm', 'protocol', host, 'WM_DELETE_WINDOW',
                         self.register(self._redock_left))
            self._left_popped = True
            self._left_popbtn.configure(text="Dock")
            # Reveal the "On top" toggle (only meaningful while floating) and
            # apply its current state to the new window.
            try:
                self._left_ontop_cb.pack(side='left', padx=(4, 0),
                                         after=self._left_popbtn)
            except Exception:
                pass
            self._set_ontop(host, self._left_ontop_var.get())
            self.after(60, lambda: self._apply_titlebar_to(host))
        except Exception as exc:
            print(f"[popout] {exc}", flush=True)

    def _redock_left(self):
        """Re-embed the popped-out Samples & Gates panel as the first pane."""
        if not self._left_popped:
            return
        host = self._left_host
        try:
            self._left_ontop_cb.pack_forget()   # hide while docked
        except Exception:
            pass
        try:
            self.tk.call('wm', 'forget', host)
        except Exception:
            pass
        try:
            self._main_paned.insert(0, host, weight=0)
        except Exception:
            self._main_paned.add(host, weight=0)
        self._left_popped = False
        try:
            self._left_popbtn.configure(text="Pop out")
        except Exception:
            pass

    def _set_ontop(self, host, on):
        """Set/clear always-on-top on a wm-managed panel host. Best-effort."""
        try:
            self.tk.call('wm', 'attributes', host, '-topmost',
                         1 if on else 0)
        except Exception:
            pass

    def _apply_left_ontop(self):
        if getattr(self, '_left_popped', False):
            self._set_ontop(self._left_host, self._left_ontop_var.get())

    def _show_or_raise(self, key, factory):
        """Open a tool window as a singleton: if one is already open, raise it
        to the front (deiconify + lift + focus) instead of spawning a duplicate.
        ``factory()`` builds and returns the Toplevel. A window that's been
        closed (destroyed) fails the winfo_exists check, so a fresh one opens."""
        reg = self.__dict__.setdefault('_tool_windows', {})
        win = reg.get(key)
        if win is not None:
            try:
                if win.winfo_exists():
                    win.deiconify()
                    win.lift()
                    win.focus_force()
                    return win
            except Exception:
                pass
        win = factory()
        reg[key] = win
        return win

    def _toggle_workspace_popout(self):
        """Pop the WHOLE Pipeline Workspace (its bar + view) out into its own
        window, or re-dock it. Driven by the workspace's Pop out / Dock button,
        which floats with the window so it can re-dock from there."""
        host = getattr(self, '_ws_host', None)
        if host is None:
            return
        if getattr(self, '_ws_popped', False):
            self._redock_workspace()
            return
        if not getattr(self, '_workspace_shown', False):
            self._open_pipeline_workspace()      # must be a pane before popping
        try:
            self.update_idletasks()
            w = max(host.winfo_width(), 380)
            h = max(host.winfo_height(), 600)
            self._editor_paned.forget(host)
            self._workspace_shown = False
            self.tk.call('wm', 'manage', host)
            self.tk.call('wm', 'title', host, 'OpenFlo — Pipeline Workspace')
            self.tk.call('wm', 'geometry', host, f'{w}x{h}')
            self.tk.call('wm', 'protocol', host, 'WM_DELETE_WINDOW',
                         self.register(self._redock_workspace))
            self._ws_popped = True
            self._set_ws_popbtn("Dock")
            self._show_ws_ontop_cb(True)
            panel = getattr(self, '_workspace_panel', None)
            if panel is not None and hasattr(panel, '_ontop_var'):
                self._set_ontop(host, panel._ontop_var.get())
            self.after(60, lambda: self._apply_titlebar_to(host))
        except Exception as exc:
            print(f"[workspace popout] {exc}", flush=True)

    def _redock_workspace(self):
        if not getattr(self, '_ws_popped', False):
            return
        host = self._ws_host
        self._show_ws_ontop_cb(False)
        try:
            self.tk.call('wm', 'forget', host)
        except Exception:
            pass
        try:
            self._editor_paned.add(host, weight=3)
        except Exception:
            pass
        self._ws_popped = False
        self._workspace_shown = True
        self._set_ws_popbtn("Pop out")

    def _apply_ws_ontop(self, on):
        """Called by the workspace panel's On-top toggle while it's floating."""
        if getattr(self, '_ws_popped', False):
            self._set_ontop(self._ws_host, on)

    def _show_ws_ontop_cb(self, show):
        """Show/hide the workspace's On-top toggle (only while popped out)."""
        panel = getattr(self, '_workspace_panel', None)
        cb = getattr(panel, '_ontop_cb', None)
        btn = getattr(panel, '_popbtn', None)
        if cb is None:
            return
        try:
            if show:
                if btn is not None:
                    cb.pack(side='left', padx=(4, 0), after=btn)
                else:
                    cb.pack(side='left', padx=(4, 0))
            else:
                cb.pack_forget()
        except Exception:
            pass

    def _close_workspace(self):
        """Hide the Pipeline Workspace (re-docking it first if it's floated).
        Driven by the workspace's own ✕ Close button as well as the View menu."""
        if getattr(self, '_ws_popped', False):
            self._redock_workspace()
        if getattr(self, '_workspace_shown', False):
            self._open_pipeline_workspace()   # shown → toggles to hidden

    def _dock_all_panels(self):
        """Re-dock any floated panels (Samples & Gates, Pipeline Workspace)
        back into the main window — recovers a pop-out window that got buried
        behind other apps or moved off-screen. Also raises the main window."""
        n = 0
        if getattr(self, '_left_popped', False):
            self._redock_left()
            n += 1
        if getattr(self, '_ws_popped', False):
            self._redock_workspace()
            n += 1
        try:
            self.deiconify()
            self.lift()
            self.focus_force()
        except Exception:
            pass
        self.status_var.set(f"Docked {n} panel(s) back into the main window."
                            if n else "No floated panels to dock.")

    def _set_ws_popbtn(self, text):
        btn = getattr(getattr(self, '_workspace_panel', None), '_popbtn', None)
        if btn is not None:
            try:
                btn.configure(text=text)
            except Exception:
                pass

    def _apply_titlebar_to(self, win, nudge=False):
        """Match a window's native title bar (the OS caption) to the theme.
        Windows-only via the DWM immersive-dark-mode attribute; best-effort and
        a no-op elsewhere. Used for the editor AND every child dialog."""
        if sys.platform != 'win32':
            return
        try:
            import ctypes
            is_dark = self._theme_var.get() in _DARK_MODES
            dark = ctypes.c_int(1 if is_dark else 0)
            hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
            dwm = ctypes.windll.dwmapi
            for attr in (20, 19):        # 20 = Win10 1903+, 19 = older builds
                try:
                    dwm.DwmSetWindowAttribute(
                        hwnd, attr, ctypes.byref(dark), ctypes.sizeof(dark))
                except Exception:
                    pass
            # Force a NEUTRAL caption colour so the active title bar isn't
            # tinted with the system accent (blue). DWMWA_CAPTION_COLOR (35) is
            # Windows 11 22000+; on Windows 10 this silently no-ops (there the
            # active-caption accent is a global personalisation setting).
            pal = current_palette()
            if is_dark:
                hx = pal['panel'].lstrip('#')
                r, g, b = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
                cap = ctypes.c_int((b << 16) | (g << 8) | r)   # COLORREF 0x00BBGGRR
            else:
                cap = ctypes.c_int(-1)        # DWMWA_COLOR_DEFAULT (system)
            try:
                dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(cap),
                                          ctypes.sizeof(cap))
            except Exception:
                pass
            if nudge:
                # Nudge a 1px resize so the caption repaints immediately.
                w, h = win.winfo_width(), win.winfo_height()
                if w > 1 and h > 1:
                    win.geometry(f'{w + 1}x{h}')
                    win.update_idletasks()
                    win.geometry(f'{w}x{h}')
        except Exception:
            pass

    def _apply_titlebar_theme(self):
        self._apply_titlebar_to(self, nudge=True)

    def _on_toplevel_mapped(self, event):
        """Dark-theme the title bar of any child dialog as it opens (bound on
        the Toplevel class). Deferred slightly so the OS window frame exists
        when the DWM attribute is set. The editor itself is handled at startup."""
        w = getattr(event, 'widget', None)
        if w is None or w is self:
            return
        try:
            if str(w.winfo_class()) != 'Toplevel':   # ignore child-widget maps
                return
        except Exception:
            return
        self.after(30, lambda win=w: self._apply_titlebar_to(win, nudge=True))
        self.after(40, lambda win=w: self._place_child(win))

    def _place_child(self, win):
        """If View → New windows open at is set, move a freshly-opened child
        dialog to the chosen corner of the main window."""
        corner = self._spawn_corner.get()
        if corner == 'off':
            return
        try:
            win.update_idletasks()
            ox, oy = self.winfo_rootx(), self.winfo_rooty()
            ow = self.winfo_width()
            ww = win.winfo_width()
            margin = 24
            y = oy + margin
            if corner == 'top-right':
                x = ox + ow - ww - margin
            else:                                # top-left
                x = ox + margin
            win.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass
