"""Help-menu dialogs (About / Environment / docs / shortcuts) — editor mixin.

Cohesive, low-coupling slice of ViewGateEditorWindow. The TYPE_CHECKING block
declares the few editor attributes/methods these dialogs use.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from .theme import current_palette
from .ui_logic import accel

if TYPE_CHECKING:
    # Type the mixin as a Tk widget (always mixed into a Toplevel) so passing
    # ``self`` as a dialog parent/master type-checks. tk.Misc (not Toplevel)
    # keeps the MRO consistent with the editor's own tk.Toplevel base; runtime
    # base is plain object.
    _MixinBase = tk.Misc
else:
    _MixinBase = object


class HelpMixin(_MixinBase):
    """About / Environment / Documentation / Keyboard-shortcuts dialogs."""

    if TYPE_CHECKING:                       # provided by the composed editor
        status_var: tk.StringVar

        def _apply_titlebar_to(self, win, nudge: bool = False,
                               size: tuple[int, int] | None = None): ...
        def _report_a_problem(self): ...

    def _show_about(self):
        """Help, About OpenFlo: version, licence, how to cite, and links."""
        AboutWindow(self)

    def _show_environment(self):
        """Help → Environment: which optional/core engines can run, so a
        greyed-out method or skipped run is explained, with install hints for
        whatever is missing. "Can run" is an import (the same check the
        Pipeline and Compare embeddings act on), done off the Tk thread: the
        window opens at once and fills in what the imports find."""
        from .async_task import run_async
        from .capabilities import (
            check_engines,
            install_hint,
            openflo_version,
            probe_capabilities,
        )
        caps = probe_capabilities(load=False)
        dlg = tk.Toplevel(self)
        dlg.title("Environment")
        dlg.transient(self)  # type: ignore[arg-type]  # self is the editor Toplevel
        self.after(60, lambda: self._apply_titlebar_to(dlg))
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill='both', expand=True)
        # Close first, at the bottom: what is packed last is what a short
        # window hides.
        ttk.Button(frm, text="Close", command=dlg.destroy).pack(
            side='bottom', anchor='e', pady=(12, 0))
        ttk.Label(frm, text=f"OpenFlo {openflo_version()}",
                  font='OpenFloTitle').pack(anchor='w')
        ttk.Label(frm, text="Optional and core analysis engines:",
                  foreground=current_palette().get('muted', 'grey')).pack(
            anchor='w', pady=(0, 8))
        cols = ('status', 'engine', 'powers', 'version')
        tv = ttk.Treeview(frm, columns=cols, show='headings', height=len(caps))
        for c, w, t in (('status', 44, ''), ('engine', 110, 'Engine'),
                        ('powers', 250, 'Powers'), ('version', 150, 'Version')):
            tv.heading(c, text=t)
            tv.column(c, width=w, anchor='w', stretch=(c == 'powers'))
        notes = ttk.Frame(frm)
        notes.pack(side='bottom', fill='x', pady=(10, 0))
        tv.pack(fill='both', expand=True)

        def show(caps):
            tv.delete(*tv.get_children())
            for cap in caps:
                if not cap['checked'] and cap['available']:
                    mark, ver = '…', 'checking…'
                elif cap['available']:
                    mark, ver = '✓', cap['version']
                elif cap['problem'] == 'not installed':
                    mark, ver = '✗', 'not installed'
                else:                       # the reason is listed below
                    mark, ver = '✗', 'fails to load'
                tv.insert('', 'end', values=(
                    mark, cap['label'], cap['powers'], ver))
            for w in notes.winfo_children():
                w.destroy()
            missing = [c for c in caps if not c['available']]
            broken = [c for c in missing if c['problem'] != 'not installed']
            if missing:
                ttk.Label(notes, text="To enable what's missing:",
                          font='OpenFloHeading').pack(anchor='w')
                for h in sorted({install_hint(c['extra']) for c in missing}):
                    ttk.Label(notes, text="    " + h,
                              font='OpenFloMono').pack(anchor='w')
                if broken:
                    ttk.Label(
                        notes, wraplength=480, justify='left',
                        foreground=current_palette().get('muted', 'grey'),
                        text="Installed but failing to load, each missing "
                             "a package of its own; reinstalling with the "
                             "command above usually brings it in:"
                    ).pack(anchor='w', pady=(4, 0))
                    for c in broken:
                        why = c['problem']
                        if why.startswith('fails to load (') and \
                                why.endswith(')'):
                            why = why[len('fails to load ('):-1]
                        ttk.Label(notes, wraplength=480, justify='left',
                                  text=f"    {c['label']}: {why}").pack(
                            anchor='w')
            elif all(c['checked'] for c in caps):
                ttk.Label(notes, text="All engines available.",
                          foreground=current_palette().get('muted', 'grey')
                          ).pack(anchor='w')

        def refresh(_r=None):
            show(probe_capabilities(load=False))
            from .tool_window import grow_to_content
            grow_to_content(dlg)            # the notes may need more room

        show(caps)
        if not all(c['checked'] for c in caps):
            run_async(dlg, check_engines, on_done=refresh)
        dlg.bind('<Escape>', lambda _e: dlg.destroy())

    def _run_diagnostics(self):
        """Help → Run diagnostics: launch the install health check
        (``openflo.diagnostics``) in a SEPARATE process so a genuinely broken
        install or a native-library crash is reported rather than taking the
        editor down with it. Output is shown in a scrollable pane."""
        import subprocess
        import sys
        import threading

        try:
            self.status_var.set("Running diagnostics… (a few seconds)")
        except Exception:
            pass

        def work():
            try:
                import os
                env = {**os.environ, 'PYTHONUTF8': '1'}
                proc = subprocess.run(
                    [sys.executable, '-m', 'openflo.diagnostics'],
                    capture_output=True, text=True, encoding='utf-8',
                    errors='replace', env=env, timeout=180)
                out = (proc.stdout or '') + (
                    f"\n[stderr]\n{proc.stderr}" if proc.stderr.strip() else '')
                rc = proc.returncode
            except Exception as exc:                       # noqa: BLE001
                # The check process itself couldn't run — that's a strong
                # "something is wrong with this install" signal in its own right.
                out = (f"Could not run the diagnostics process:\n\n{exc}\n\n"
                       f"Interpreter: {sys.executable}")
                rc = 2
            try:
                self.after(0, lambda: self._show_diagnostics_result(out, rc))
            except Exception:
                pass
        threading.Thread(target=work, daemon=True).start()

    def _show_diagnostics_result(self, text, returncode):
        """Render the diagnostics report in a scrollable, copyable dialog."""
        healthy = returncode == 0
        try:
            self.status_var.set(
                "Diagnostics: install healthy." if healthy
                else "Diagnostics: issues found — see the report.")
        except Exception:
            pass
        dlg = tk.Toplevel(self)
        dlg.title("OpenFlo diagnostics")
        dlg.transient(self)  # type: ignore[arg-type]  # self is the editor Toplevel
        dlg.geometry("680x560")
        self.after(60, lambda: self._apply_titlebar_to(dlg))
        pal = current_palette()
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill='both', expand=True)
        ttk.Label(
            frm,
            text=("✓  Install healthy" if healthy
                  else "✗  Issues found — see below"),
            font='OpenFloTitle',
            style='Success.TLabel' if healthy else 'Error.TLabel').pack(
                anchor='w')
        ttk.Label(
            frm,
            text=("Seeded synthetic data reproduced the reference baseline and "
                  "every pinned dependency matched."
                  if healthy else
                  "One or more checks fell outside norms. The [FAIL] rows below "
                  "say what; a reinstall hint is at the bottom."),
            foreground=pal.get('muted', 'grey'), wraplength=640,
            justify='left').pack(anchor='w', pady=(2, 8))

        body = ttk.Frame(frm)
        body.pack(fill='both', expand=True)
        sb = ttk.Scrollbar(body, orient='vertical')
        txt = tk.Text(body, wrap='none', font='OpenFloMono',
                      yscrollcommand=sb.set, bg=pal['bg'], fg=pal['fg'],
                      relief='flat', borderwidth=0,
                      insertbackground=pal['fg'])
        sb.configure(command=txt.yview)
        sb.pack(side='right', fill='y')
        txt.pack(side='left', fill='both', expand=True)
        txt.insert('1.0', text)
        txt.configure(state='disabled')

        btns = ttk.Frame(frm)
        btns.pack(fill='x', pady=(10, 0))

        def _copy():
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                self.status_var.set("Diagnostics report copied to the clipboard.")
            except Exception:
                pass

        ttk.Button(btns, text="Copy report", command=_copy).pack(side='left')
        if not healthy:
            ttk.Button(btns, text="Report a problem…",
                       command=self._report_a_problem).pack(
                side='left', padx=(6, 0))
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side='right')
        dlg.bind('<Escape>', lambda _e: dlg.destroy())

    def _open_documentation(self):
        """Open the project README / docs in the default browser."""
        import webbrowser

        from .update import DOCS_URL
        webbrowser.open(DOCS_URL)
        self.status_var.set("Opened the OpenFlo documentation in your browser.")

    def _open_guide(self, topic=None):
        """Help, User guide (F1): the guide bundled with OpenFlo, which
        needs no connection (Complaints 27)."""
        from .ui_guide import GuideWindow
        GuideWindow(self, topic=topic)

    def _show_shortcuts(self):
        """Help, Keyboard shortcuts: a two-column reference."""
        ShortcutsWindow(self)


# (section, [(keys, action), ...]) as bound in LifecycleMixin._bind_shortcuts
# (and undo, redo and the clipboard where those are bound). A test checks
# each one is bound, so the list can't drift from the keys.
SHORTCUTS = (
    ("File", (
        ("Ctrl+Shift+A", "Add FCS files"),
        ("Ctrl+O", "Open session"),
        ("Ctrl+S", "Save session"),
        ("Ctrl+E", "Export FlowJo workspace"),
        ("Ctrl+Shift+S", "Export plot image"),
        ("Ctrl+W", "Close the window"),
        ("Ctrl+Q", "Quit OpenFlo"),
    )),
    ("Edit", (
        ("Ctrl+Z", "Undo"),
        ("Ctrl+Y", "Redo"),
        ("Ctrl+C", "Copy gates"),
        ("Ctrl+X", "Cut gates"),
        ("Ctrl+V", "Paste gates"),
        ("Delete", "Delete the selected gate"),
        ("Ctrl+,", "Preferences"),
    )),
    ("View", (
        ("Ctrl+F", "Find a sample or gate"),
        ("Ctrl+0", "Fit the plot to the data"),
        ("Ctrl++", "Zoom in"),
        ("Ctrl+-", "Zoom out"),
        ("Ctrl+1", "Show all events"),
        ("Ctrl+2", "Highlight the gated events"),
        ("Ctrl+3", "Show only the gated events"),
        ("F5", "Redraw the plot"),
        ("Esc", "Leave the zoom tool"),
        ("F6", "Move focus between the gate list and the plot (then "
               "the arrow keys move the selected gate; Shift for "
               "larger steps)"),
        ("F9", "Show or hide the pipeline workspace"),
        ("Ctrl+`", "Show or hide the log and console"),
    )),
    ("Analyze", (
        ("Ctrl+T", "Statistics"),
    )),
    ("Help", (
        ("F1", "User guide"),
    )),
)


class ShortcutsWindow(tk.Toplevel):
    """Keyboard shortcuts in two columns, grouped by menu, with a Close
    button; Escape closes it."""

    def __init__(self, editor):
        super().__init__(editor)
        self.title("Keyboard shortcuts")
        self.minsize(380, 300)
        bar = ttk.Frame(self, padding=(12, 0, 12, 10))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", default='active',
                   command=self.destroy).pack(side='right')
        holder = ttk.Frame(self, padding=(12, 12, 12, 8))
        holder.pack(side='top', fill='both', expand=True)
        rows = sum(len(items) + 1 for _sec, items in SHORTCUTS)
        self.tree = ttk.Treeview(holder, columns=('keys', 'action'),
                                 show='headings', height=min(rows, 22),
                                 selectmode='none')
        self.tree.heading('keys', text="Keys")
        self.tree.heading('action', text="Does")
        self.tree.column('keys', width=130, anchor='w', stretch=False)
        self.tree.column('action', width=280, anchor='w')
        sb = ttk.Scrollbar(holder, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.tree.tag_configure('section', font='OpenFloHeading')
        for section, items in SHORTCUTS:
            self.tree.insert('', 'end', values=(section, ''),
                             tags=('section',))
            for keys, action in items:
                self.tree.insert('', 'end',
                                 values=(f"   {accel(keys)}", action))
        self.bind('<Escape>', lambda _e: self.destroy())


CITATION = ("Skyler Niedzielski, OpenFlo. ORCID 0009-0004-4727-4639. "
            "GitHub's \u201cCite this repository\u201d button gives the "
            "full entry.")


class AboutWindow(tk.Toplevel):
    """Version, licence, how to cite (with a Copy button) and links."""

    def __init__(self, editor):
        super().__init__(editor)
        from . import __version__
        from .update import DOCS_URL, ISSUES_URL
        self.title("About OpenFlo")
        self.resizable(False, False)
        body = ttk.Frame(self, padding=(18, 16, 18, 8))
        body.pack(side='top', fill='both', expand=True)
        ttk.Label(body, text="OpenFlo", font='OpenFloTitle').pack(anchor='w')
        self.version_label = ttk.Label(body, text=f"Version {__version__}")
        self.version_label.pack(anchor='w')
        ttk.Label(body, wraplength=420, justify='left', padding=(0, 8, 0, 0),
                  text="Flow cytometry analysis: gating, statistics, "
                       "clustering and figures, with FlowJo workspace "
                       "import and export.").pack(anchor='w')
        ttk.Label(body, wraplength=420, justify='left', padding=(0, 8, 0, 0),
                  text="Licence: MIT, free to use.").pack(anchor='w')
        ttk.Label(body, wraplength=420, justify='left', padding=(0, 8, 0, 0),
                  text="If OpenFlo contributes to research you publish or "
                       "present, please cite it:").pack(anchor='w')
        ttk.Label(body, wraplength=420, justify='left', padding=(12, 2, 0, 0),
                  text=CITATION).pack(anchor='w')
        ttk.Label(body, wraplength=420, justify='left', padding=(0, 8, 0, 0),
                  text="Developed with assistance from Anthropic's Claude "
                       "(Claude Code).").pack(anchor='w')
        bar = ttk.Frame(self, padding=(18, 4, 18, 12))
        bar.pack(side='bottom', fill='x')
        ttk.Button(bar, text="Close", default='active',
                   command=self.destroy).pack(side='right')
        ttk.Button(bar, text="Copy citation",
                   command=self.copy_citation).pack(side='left')
        ttk.Button(bar, text="Documentation",
                   command=lambda: self._open(DOCS_URL)).pack(
            side='left', padx=(6, 0))
        ttk.Button(bar, text="Report a problem",
                   command=lambda: self._open(ISSUES_URL)).pack(
            side='left', padx=(6, 0))
        self.bind('<Escape>', lambda _e: self.destroy())

    def copy_citation(self):
        self.clipboard_clear()
        self.clipboard_append(CITATION)

    @staticmethod
    def _open(url):
        import webbrowser
        webbrowser.open(url)
