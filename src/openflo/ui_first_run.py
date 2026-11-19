"""The first-run welcome window and the getting-started tour panel.

Both are small, non-modal windows over the editor: neither grabs input, so
the editor stays usable while they are open. The flag and the steps are in
first_run; the editor opens these through its Help mixin.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .first_run import PREF_KEY, TOUR_STEPS, tour_step_text
from .prefs import write_pref
from .ui_tips import auto_tip

# Width of the outline drawn round the widget a tour step points at.
OUTLINE = 3
# How often (ms) the tour card checks that its widget has not moved.
WATCH_MS = 250


def _screen_box(widget):
    """Where `widget` is on screen: (x, y, width, height)."""
    return (widget.winfo_rootx(), widget.winfo_rooty(),
            widget.winfo_width(), widget.winfo_height())


class WelcomeWindow(tk.Toplevel):
    """Welcome to OpenFlo / [Open the sample] [Open my own files]
    [Don't show again]. Shown once: opening it records that (first_run)."""

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.title("Welcome to OpenFlo")
        self.resizable(False, False)
        try:
            self.transient(editor)
        except tk.TclError:
            pass
        write_pref(PREF_KEY, True)          # shown once: not again
        frm = ttk.Frame(self, padding=18)
        frm.pack(fill='both', expand=True)
        ttk.Label(frm, text="Welcome to OpenFlo",
                  font='OpenFloTitle').pack(anchor='w')
        ttk.Label(frm, wraplength=380, justify='left', text=(
            "Open the sample to see a gated plot right away, or open your "
            "own files. A short tour then shows where things are.")).pack(
            anchor='w', pady=(4, 14))
        self.sample_btn = ttk.Button(frm, text="Open the sample",
                                     command=self._open_sample)
        self.files_btn = ttk.Button(frm, text="Open my own files",
                                    command=self._open_files)
        self.dismiss_btn = ttk.Button(frm, text="Don't show again",
                                      command=self.destroy)
        for b in (self.sample_btn, self.files_btn, self.dismiss_btn):
            b.pack(fill='x', pady=2)
        auto_tip(self.sample_btn, "Open two generated samples, a control "
                                  "and a treated one, with their gates "
                                  "drawn. A copy is made in your OpenFlo "
                                  "folder, so you can change it freely.")
        auto_tip(self.files_btn, "Choose .fcs files from your computer to "
                                 "load and plot.")
        auto_tip(self.dismiss_btn, "Close this window for good. Help, "
                                   "Getting started tour and Help, Open "
                                   "the sample data still offer the tour "
                                   "and the sample.")
        ttk.Label(frm, style='Muted.TLabel', wraplength=380, justify='left',
                  text="Help > Getting started tour shows the tour again."
                  ).pack(anchor='w', pady=(12, 0))
        self.bind('<Escape>', lambda _e: self.destroy())
        self.sample_btn.focus_set()

    def _open_sample(self):
        self.destroy()
        self.editor._open_sample_data(tour=True)

    def _open_files(self):
        self.destroy()
        self.editor._add_samples()
        self.editor._start_tour()


def tour_target(editor, key):
    """The editor widget tour step `key` points at, or None."""
    try:
        if key == 'files':
            return editor._menubar_buttons[0]
        if key == 'tree':
            return editor.gate_tv
        if key == 'axes':
            return editor.x_combo.master.master
        if key == 'gates':
            return editor._gate_tool_widgets[0].master
    except (AttributeError, IndexError):
        return None
    return None


class TourPanel(tk.Toplevel):
    """Step i of N / title / text / [Back] [Next] [Close], placed beside
    the widget the step is about, which gets an outline."""

    def __init__(self, editor, start=0):
        super().__init__(editor)
        self.editor = editor
        self.title("Getting started")
        self.resizable(False, False)
        try:
            self.transient(editor)
        except tk.TclError:
            pass
        self.index = 0
        self._outline: list[tk.Frame] = []
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill='both', expand=True)
        self.step_label = ttk.Label(frm, style='Muted.TLabel')
        self.step_label.pack(anchor='w')
        self.title_label = ttk.Label(frm, font='OpenFloHeading')
        self.title_label.pack(anchor='w', pady=(2, 4))
        self.text_label = ttk.Label(frm, wraplength=300, justify='left')
        self.text_label.pack(anchor='w')
        bar = ttk.Frame(frm)
        bar.pack(fill='x', pady=(12, 0))
        # Back, Next, then Close at the end, as every dialog orders them.
        self.close_btn = ttk.Button(bar, text="Close", command=self.destroy)
        self.close_btn.pack(side='right')
        self.next_btn = ttk.Button(bar, text="Next", command=self.next)
        self.next_btn.pack(side='right', padx=(0, 6))
        self.back_btn = ttk.Button(bar, text="Back", command=self.back)
        self.back_btn.pack(side='right', padx=(0, 6))
        auto_tip(self.next_btn, "Go to the next step. On the last step, "
                                "Done ends the tour.")
        auto_tip(self.back_btn, "Go back to the step before.")
        self.bind('<Escape>', lambda _e: self.destroy())
        self.bind('<Destroy>', self._on_destroy, add='+')
        # Follow the editor: when it moves (the user drags it, or the window
        # manager settles it after it opens, as macOS does late), place the
        # panel beside its widget again so it never covers it.
        self._follow_job = None
        self._follow_id = editor.bind('<Configure>', self._on_editor_moved,
                                      add='+')
        # macOS does not always send <Configure> when the window server moves
        # the editor (mirror CI: the card stayed put after a move), so also
        # look a few times a second and re-place when the widget has moved.
        self._placed_for = None
        self._watch_job = None
        self._watch()
        self.show(start)
        # Keys go to the tour, as to the welcome: Enter/Space for Next,
        # Escape to close. With no widget of its own to focus, macOS left
        # the keys with the editor and Escape did nothing.
        self.next_btn.focus_set()

    def show(self, index):
        """Show step `index` (clamped to the steps there are)."""
        self.index = max(0, min(index, len(TOUR_STEPS) - 1))
        key, title, text = TOUR_STEPS[self.index]
        self.step_label.configure(text=tour_step_text(self.index))
        self.title_label.configure(text=title)
        self.text_label.configure(text=text)
        self.back_btn.state(['disabled' if self.index == 0 else '!disabled'])
        last = self.index == len(TOUR_STEPS) - 1
        self.next_btn.configure(text="Done" if last else "Next")
        self._point_at(tour_target(self.editor, key))

    def next(self):
        """Go to the next step; on the last one (Done), close the tour."""
        if self.index >= len(TOUR_STEPS) - 1:
            self.destroy()
        else:
            self.show(self.index + 1)

    def back(self):
        """Go back one step."""
        self.show(self.index - 1)

    def target(self):
        """The widget the current step points at, or None (for tests)."""
        return tour_target(self.editor, TOUR_STEPS[self.index][0])

    def _clear_outline(self):
        for f in self._outline:
            try:
                f.destroy()
            except tk.TclError:
                pass
        self._outline = []

    def _on_destroy(self, event):
        if event.widget is self:
            self._clear_outline()
            self._stop_following()

    def _on_editor_moved(self, event):
        """The editor moved or resized: re-place the panel once Tk is idle
        (one re-place for a burst of events)."""
        if event.widget is not self.editor or self._follow_job is not None:
            return
        try:
            self._follow_job = self.after_idle(self._refollow)
        except tk.TclError:
            pass

    def _refollow(self):
        self._follow_job = None
        try:
            if self.winfo_exists():
                self._point_at(self.target())
        except tk.TclError:
            pass

    def _watch(self):
        """Re-place the panel when its widget has moved or changed size on
        screen since it was placed (a dragged sash sends the editor no
        <Configure>), then look again shortly, until the panel closes."""
        self._watch_job = None
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            widget = self.target()
            if widget is not None and self._placed_for is not None \
                    and _screen_box(widget) != self._placed_for:
                self._point_at(widget)
        except tk.TclError:
            pass        # e.g. the widget is being rebuilt: look next time
        try:
            self._watch_job = self.after(WATCH_MS, self._watch)
        except tk.TclError:
            pass

    def _stop_following(self):
        """Remove only this panel's <Configure> binding from the editor;
        the editor keeps its own (tkinter's unbind drops all of them before
        Python 3.13)."""
        for name in ('_follow_job', '_watch_job'):
            job = getattr(self, name, None)
            setattr(self, name, None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        fid, self._follow_id = self._follow_id, None
        if fid is None:
            return
        try:
            script = self.editor.bind('<Configure>')
            self.editor.bind('<Configure>', '\n'.join(
                line for line in script.split('\n') if fid not in line))
            self.editor.deletecommand(fid)
        except tk.TclError:
            pass

    def _point_at(self, widget):
        """Outline `widget` in the editor and move this panel beside it."""
        self._clear_outline()
        if widget is None:
            return
        ed = self.editor
        try:
            ed.update_idletasks()
            if not widget.winfo_ismapped():
                return
            x = widget.winfo_rootx() - ed.winfo_rootx()
            y = widget.winfo_rooty() - ed.winfo_rooty()
            w, h = widget.winfo_width(), widget.winfo_height()
        except tk.TclError:
            return
        from .theme import current_palette
        color = current_palette().get('accent', '#2f6fdf')
        t = OUTLINE
        for fx, fy, fw, fh in ((x - t, y - t, w + 2 * t, t),
                               (x - t, y + h, w + 2 * t, t),
                               (x - t, y, t, h),
                               (x + w, y, t, h)):
            f = tk.Frame(ed, background=color, highlightthickness=0, bd=0)
            f.place(x=fx, y=fy, width=fw, height=fh)
            f.lift()
            self._outline.append(f)
        # Beside the widget: below it, or above it when there is no room.
        try:
            self.update_idletasks()
            px = widget.winfo_rootx()
            py = widget.winfo_rooty() + h + 8
            if py + self.winfo_reqheight() > self.winfo_screenheight():
                py = max(0, widget.winfo_rooty() - self.winfo_reqheight() - 8)
            px = max(0, min(px, self.winfo_screenwidth()
                            - self.winfo_reqwidth()))
            self.geometry(f"+{px}+{py}")
            self._placed_for = _screen_box(widget)
        except tk.TclError:
            pass
