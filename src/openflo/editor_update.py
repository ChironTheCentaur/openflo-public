"""In-app update check, offer, and run.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from tkinter import messagebox

from .editor_base import EditorMixin


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
            except Exception:
                pass

        def work():
            res = _upd.check_for_update()
            try:
                self.after(0, lambda: self._on_update_checked(res, silent))
            except Exception:
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
            except Exception:
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
        except Exception:
            pass
        self._offer_update(res)

    def _offer_update(self, res):
        import webbrowser

        from . import update as _upd
        kind = _upd.detect_install_kind()
        how = ("a 'git pull' in your source checkout" if kind == 'git'
               else "'pip install --upgrade' from GitHub")
        ans = messagebox.askyesnocancel(
            "Update available",
            f"OpenFlo {res['latest']} is available "
            f"(you have {res['current']}).\n\n"
            f"Update now via {how}? OpenFlo must be restarted afterward.\n\n"
            "  • Yes — update now\n"
            "  • No — open the release page in your browser\n"
            "  • Cancel — not now", parent=self)
        if ans is None:
            return
        if ans is False:
            try:
                webbrowser.open(res['url'])
            except Exception:
                pass
            return
        self._run_update(kind)

    def _run_update(self, kind):
        import threading

        from . import update as _upd
        try:
            self.status_var.set("Updating OpenFlo… (this may take a minute)")
        except Exception:
            pass

        def work():
            ok, log = _upd.run_update(kind=kind)
            try:
                self.after(0, lambda: self._on_update_done(ok, log))
            except Exception:
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
