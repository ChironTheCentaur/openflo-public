"""In-app update check, offer, and run.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from tkinter import TclError, messagebox

from .editor_base import EditorMixin


def ask_update_or_cancel(parent, title, text):
    """A modal question with two buttons, Update and Cancel. True for
    Update; Cancel, Escape or closing the window is False."""
    import tkinter as tk
    from tkinter import ttk
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.resizable(False, False)
    try:
        dlg.transient(parent.winfo_toplevel())
    except TclError:
        pass
    answer = {'update': False}
    body = ttk.Frame(dlg, padding=16)
    body.pack(fill='both', expand=True)
    ttk.Label(body, text=text, justify='left', wraplength=420).pack(
        anchor='w')
    bb = ttk.Frame(body)
    bb.pack(anchor='e', pady=(14, 0))

    def choose(value):
        answer['update'] = value
        dlg.destroy()
    ttk.Button(bb, text="Cancel", command=lambda: choose(False)).pack(
        side='right')
    up = ttk.Button(bb, text="Update", command=lambda: choose(True))
    up.pack(side='right', padx=(0, 6))
    dlg.bind('<Escape>', lambda _e: choose(False))
    dlg.protocol('WM_DELETE_WINDOW', lambda: choose(False))
    try:
        up.focus_set()
        dlg.grab_set()
    except TclError:
        pass
    dlg.wait_window()
    return answer['update']


class UpdateMixin(EditorMixin):
    """Check for updates, offer an upgrade, run it, and report the result."""

    def _check_for_updates(self, silent=False):
        """Check GitHub for a newer release on a daemon thread (network off the
        Tk thread). ``silent=True`` (startup check) only speaks up when an
        update exists and never shows an error dialog."""
        import threading

        from . import update as _upd
        if not silent:
            try:
                self.status_var.set("Checking for updates…")
            except TclError:
                pass

        def work():
            res = _upd.check_for_update()
            try:
                self.after(0, lambda: self._on_update_checked(res, silent))
            except Exception:  # noqa: BLE001 - window may be closed before the update check returns
                pass
        threading.Thread(target=work, daemon=True).start()

    def _on_update_checked(self, res, silent):
        if res is None:
            if not silent:
                messagebox.showinfo(
                    "Check for updates",
                    "Couldn't reach GitHub to check for updates "
                    "(offline or rate-limited).", parent=self)
            return
        if not res.get('available'):
            try:
                self.status_var.set(f"OpenFlo {res['current']} is up to date.")
            except TclError:
                pass
            if not silent:
                messagebox.showinfo(
                    "Check for updates",
                    f"OpenFlo {res['current']} is up to date.", parent=self)
            return
        try:
            self.status_var.set(
                f"Update available: OpenFlo {res['latest']} — "
                "Help ▸ Check for updates")
        except TclError:
            pass
        self._offer_update(res)

    def _offer_update(self, res):
        from . import update as _upd
        kind = _upd.detect_install_kind()
        how = ("a 'git pull' in your source checkout" if kind == 'git'
               else "'pip install --upgrade' from GitHub")
        # Two plain choices. Yes / No / Cancel, with No opening a web page,
        # read as three ways to say yes.
        if ask_update_or_cancel(
                self, "Update available",
                f"OpenFlo {res['latest']} is available "
                f"(you have {res['current']}).\n\n"
                f"Update now via {how}? OpenFlo must be restarted "
                f"afterward.\n\nRelease notes: {res.get('url', '')}"):
            self._run_update(kind)

    def _run_update(self, kind):
        import threading

        from . import update as _upd
        try:
            self.status_var.set("Updating OpenFlo… (this may take a minute)")
        except TclError:
            pass

        def work():
            ok, log = _upd.run_update(kind=kind)
            try:
                self.after(0, lambda: self._on_update_done(ok, log))
            except Exception:  # noqa: BLE001 - window may be closed before the update finishes
                pass
        threading.Thread(target=work, daemon=True).start()

    def _on_update_done(self, ok, log):
        tail = '\n'.join((log or '').splitlines()[-12:])
        if ok:
            self.status_var.set("Update installed — restart OpenFlo to use it.")
            messagebox.showinfo(
                "Update complete",
                "Update installed. Restart OpenFlo to use the new version.\n\n"
                + tail, parent=self)
        else:
            self.status_var.set("Update failed — see the message.")
            messagebox.showerror(
                "Update failed", "The update did not complete:\n\n" + tail,
                parent=self)
