"""Help-menu dialogs (About / Environment / docs / shortcuts) — editor mixin.

Cohesive, low-coupling slice of ViewGateEditorWindow. The TYPE_CHECKING block
declares the few editor attributes/methods these dialogs use.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

from .theme import current_palette

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

        def _apply_titlebar_to(self, win, nudge: bool = False): ...
        def _report_a_problem(self): ...

    def _show_about(self):
        from . import __version__
        messagebox.showinfo(
            "About OpenFlo",
            f"OpenFlo {__version__}\n\n"
            "Flow cytometry analysis pipeline + gate editor with FlowJo "
            ".wsp round-trip.\n\n"
            "Free to use (MIT). The one ask: if OpenFlo contributes to research "
            "you publish or present, please cite it —\n"
            "    Skyler Niedzielski, OpenFlo.\n"
            "    ORCID 0009-0004-4727-4639\n"
            "(GitHub's “Cite this repository” button generates the full entry.)"
            "\n\n"
            "Developed with assistance from Anthropic's Claude (Claude Code).",
            parent=self)

    def _show_environment(self):
        """Help → Environment: which optional/core engines are installed, so a
        greyed-out method or skipped run is explained, with install hints for
        whatever is missing."""
        from .capabilities import install_hint, openflo_version, probe_capabilities
        caps = probe_capabilities()
        dlg = tk.Toplevel(self)
        dlg.title("Environment")
        dlg.transient(self)  # type: ignore[arg-type]  # self is the editor Toplevel
        self.after(60, lambda: self._apply_titlebar_to(dlg))
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill='both', expand=True)
        ttk.Label(frm, text=f"OpenFlo {openflo_version()}",
                  font=('TkDefaultFont', 11, 'bold')).pack(anchor='w')
        ttk.Label(frm, text="Optional and core analysis engines:",
                  foreground=current_palette().get('muted', 'grey')).pack(
            anchor='w', pady=(0, 8))
        cols = ('status', 'engine', 'powers', 'version')
        tv = ttk.Treeview(frm, columns=cols, show='headings', height=len(caps))
        for c, w, t in (('status', 44, ''), ('engine', 110, 'Engine'),
                        ('powers', 250, 'Powers'), ('version', 90, 'Version')):
            tv.heading(c, text=t)
            tv.column(c, width=w, anchor='w', stretch=(c == 'powers'))
        missing = []
        for cap in caps:
            mark = '✓' if cap['available'] else '✗'
            tv.insert('', 'end', values=(
                mark, cap['label'], cap['powers'],
                cap['version'] if cap['available'] else 'not installed'))
            if not cap['available']:
                missing.append(cap)
        tv.pack(fill='both', expand=True)
        if missing:
            hints = sorted({install_hint(c['extra']) for c in missing})
            box = ttk.Frame(frm)
            box.pack(fill='x', pady=(10, 0))
            ttk.Label(box, text="To enable what's missing:",
                      font=('TkDefaultFont', 9, 'bold')).pack(anchor='w')
            for h in hints:
                ttk.Label(box, text="    " + h,
                          font=('TkFixedFont', 9)).pack(anchor='w')
        else:
            ttk.Label(frm, text="All engines available.",
                      foreground=current_palette().get('muted', 'grey')).pack(
                anchor='w', pady=(10, 0))
        ttk.Button(frm, text="Close", command=dlg.destroy).pack(
            anchor='e', pady=(12, 0))
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
            font=('TkDefaultFont', 12, 'bold'),
            foreground=(pal.get('ok', 'green') if healthy
                        else pal.get('err', '#c0392b'))).pack(anchor='w')
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
        txt = tk.Text(body, wrap='none', font=('TkFixedFont', 9),
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
        webbrowser.open("https://github.com/ChironTheCentaur/openflo#readme")
        self.status_var.set("Opened the OpenFlo documentation in your browser.")

    def _show_shortcuts(self):
        """A quick reference of the keyboard shortcuts."""
        messagebox.showinfo(
            "Keyboard shortcuts",
            "File\n"
            "    Ctrl+Shift+A    Add FCS\n"
            "    Ctrl+O          Open session\n"
            "    Ctrl+S          Save session\n"
            "    Ctrl+E          Export → FlowJo .wsp\n"
            "    Ctrl+Shift+S    Save plot as image\n"
            "    Ctrl+W          Close\n\n"
            "Edit\n"
            "    Ctrl+Z          Undo\n"
            "    Ctrl+Y          Redo\n"
            "    Ctrl+C/X/V      Copy / cut / paste\n"
            "    Delete          Delete selected gate\n"
            "    Ctrl+,          Preferences\n\n"
            "View / navigation\n"
            "    Ctrl+F          Find sample / gate\n"
            "    Ctrl+0          Reset plot view (fit)\n"
            "    Ctrl++ / Ctrl+-  Zoom in / out\n"
            "    Ctrl+1/2/3      Display: all / highlight / filter\n"
            "    F5              Replot\n"
            "    Esc             Cancel zoom tool\n"
            "    F9              Toggle Pipeline Workspace\n"
            "    Ctrl+`          Toggle log / console\n\n"
            "Analyze\n"
            "    Ctrl+T          Statistics\n\n"
            "Help\n"
            "    F1              About OpenFlo",
            parent=self)
