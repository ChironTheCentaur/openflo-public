"""In-app update check, offer, and run.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

from tkinter import TclError, messagebox

from .editor_base import EditorMixin
from .errors import note as note_handled
from .ui_tips import auto_tip


def _notes_box(parent, text, height):
    """A read-only, word-wrapped, scrolling text box showing ``text``."""
    import tkinter as tk
    from tkinter import ttk

    from .tool_window import set_readonly_text
    frame = ttk.Frame(parent)
    frame.pack(fill='both', expand=True, pady=(10, 0))
    box = tk.Text(frame, wrap='word', height=height, width=64,
                  relief='flat', borderwidth=1, padx=6, pady=4)
    sb = ttk.Scrollbar(frame, orient='vertical', command=box.yview)
    box.configure(yscrollcommand=sb.set)
    sb.pack(side='right', fill='y')
    box.pack(side='left', fill='both', expand=True)
    set_readonly_text(box, text)
    return box


def _open_url(url):
    import webbrowser
    try:
        webbrowser.open(url)
    except webbrowser.Error as exc:
        note_handled("opening a web page", exc)


def ask_update_or_cancel(parent, title, text, notes=None, link=None,
                         action="Update", action_tip=None):
    """A modal question with two buttons, ``action`` (Update) and Cancel.
    True for the action; Cancel, Escape or closing the window is False.
    ``notes`` (text) shows in a scrolling box under the question; ``link``
    is a ``(label, url)`` button that opens the page and leaves the question
    open."""
    import tkinter as tk
    from tkinter import ttk
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.resizable(bool(notes), bool(notes))
    try:
        dlg.transient(parent.winfo_toplevel())
    except TclError:
        pass
    answer = {'update': False}
    body = ttk.Frame(dlg, padding=16)
    body.pack(fill='both', expand=True)
    # Buttons first, at the bottom, so long notes never push them off screen.
    bb = ttk.Frame(body)
    bb.pack(side='bottom', fill='x', pady=(14, 0))
    ttk.Label(body, text=text, justify='left', wraplength=440).pack(
        anchor='w')
    if notes:
        _notes_box(body, notes, height=12)

    def choose(value):
        answer['update'] = value
        dlg.destroy()
    if link:
        label, url = link
        ttk.Button(bb, text=label, command=lambda: _open_url(url)).pack(
            side='left')
    ttk.Button(bb, text="Cancel", command=lambda: choose(False)).pack(
        side='right')
    up = ttk.Button(bb, text=action, command=lambda: choose(True))
    up.pack(side='right', padx=(0, 6))
    auto_tip(up, action_tip or (
        "Download and install the new version now. OpenFlo "
        "keeps running while it installs; restart it afterwards to use "
        "the new version. Cancel keeps the version you have."))
    dlg.bind('<Escape>', lambda _e: choose(False))
    dlg.protocol('WM_DELETE_WINDOW', lambda: choose(False))
    try:
        up.focus_set()
        dlg.grab_set()
    except TclError:
        pass
    dlg.wait_window()
    return answer['update']


def offer_parts(res, kind):
    """What the update offer says for check result ``res`` and install
    ``kind``: ``{text, notes, link, action, action_tip}`` (see
    :func:`ask_update_or_cancel`)."""
    from .update import CHANGELOG_URL
    head = (f"OpenFlo {res['latest']} is available "
            f"(you have {res['current']}).")
    url = res.get('url', '')
    if not res.get('has_release', True):
        notes = ("This version was tagged without a GitHub release, so "
                 "there are no release notes for it. The changelog lists "
                 "what changed.")
        link = ("Changelog", CHANGELOG_URL)
    else:
        notes = res.get('notes') or "This release has no notes."
        link = ("Full release notes", url)
    if kind == 'installer':
        return {'text': head + "\n\nTo update, open the release page, "
                "download the new installer and run it. It upgrades OpenFlo "
                "in place; close OpenFlo first.",
                'notes': notes, 'link': link,
                'action': "Open release page",
                'action_tip': "Open the release page in your web browser to "
                "download the new installer. Nothing is installed until you "
                "run it. Cancel keeps the version you have."}
    how = ("a 'git pull' in your source checkout" if kind == 'git'
           else "'pip install --upgrade' from GitHub")
    return {'text': head + f"\n\nUpdate now via {how}? OpenFlo must be "
            "restarted afterward.",
            'notes': notes, 'link': link, 'action': "Update",
            'action_tip': None}


def show_whats_new(parent, version, text):
    """The What's new window: ``version``'s changelog notes (``text``) in a
    scrolling box, a link to the full changelog, and Close."""
    import tkinter as tk
    from tkinter import ttk

    from .update import CHANGELOG_URL
    dlg = tk.Toplevel(parent)
    dlg.title(f"What's new in OpenFlo {version}")
    try:
        dlg.transient(parent.winfo_toplevel())
    except TclError:
        pass
    body = ttk.Frame(dlg, padding=16)
    body.pack(fill='both', expand=True)
    bb = ttk.Frame(body)
    bb.pack(side='bottom', fill='x', pady=(12, 0))
    ttk.Label(body, text=f"What's new in OpenFlo {version}",
              font='OpenFloHeading').pack(anchor='w')
    _notes_box(body, text, height=20)
    full = ttk.Button(bb, text="Full changelog",
                      command=lambda: _open_url(CHANGELOG_URL))
    full.pack(side='left')
    auto_tip(full, "Open the full changelog, every version, in your web "
             "browser. Help > What's new opens this window again.")
    ttk.Button(bb, text="Close", command=dlg.destroy).pack(side='right')
    dlg.bind('<Escape>', lambda _e: dlg.destroy())
    return dlg


def startup_popups_allowed(environ=None):
    """False under pytest: a test never gets a window it did not ask for."""
    import os
    env = os.environ if environ is None else environ
    return 'PYTEST_CURRENT_TEST' not in env


class UpdateMixin(EditorMixin):
    """Check for updates, offer an upgrade, run it, and report the result;
    show What's new after an upgrade."""

    # Whether What's new opens on this start (set by
    # _schedule_startup_checks); the welcome does not open with it.
    _whats_new_at_start = False

    def _schedule_startup_checks(self):
        """The main window's start: What's new on the first start of a newer
        version, and (unless turned off in Preferences) a quiet update check
        off the Tk thread. Neither runs under pytest. Whether What's new is
        due is decided here, before the welcome or anything else this start
        saves to preferences; the window opens once the editor is up and
        "Resume last session?" is answered (_after_resume_step), in place
        of the welcome."""
        from .prefs import load_check_updates_on_start
        if not startup_popups_allowed():
            return
        show = self._whats_new_due()
        # The welcome (gui.main) stays shut on this start: an upgrade is not
        # a first launch, and one window at a time.
        self._whats_new_at_start = show
        try:
            if show:
                # Never over "Resume last session?": its question runs a
                # nested event loop that keeps firing timers.
                self.after(800, lambda: self._after_resume_step(
                    self._show_whats_new))
            if load_check_updates_on_start():
                self.after(2500,
                           lambda: self._check_for_updates(silent=True))
        except TclError:
            pass

    def _whats_new_due(self):
        """Whether What's new opens on this start: the first start of a newer
        version, an upgrade from 3.0.0 included (whats_new.
        first_start_of_new_version; a fresh install records its version and
        shows nothing)."""
        from .prefs import read_prefs, write_pref
        from .update import current_version
        from .whats_new import first_start_of_new_version
        try:
            return first_start_of_new_version(current_version(), read_prefs,
                                              write_pref)
        except Exception as exc:  # noqa: BLE001 - a prefs problem must never stop the app starting
            note_handled("checking for a new version's notes", exc)
            return False

    def _show_whats_new(self):
        """Help > What's new: this version's changelog notes."""
        from .update import current_version
        from .whats_new import whats_new_text
        version = current_version()
        return show_whats_new(self, version, whats_new_text(version))

    def _check_for_updates(self, silent=False):
        """Check GitHub for a newer release on a daemon thread (network off the
        Tk thread). ``silent=True`` (startup check) only speaks up when an
        update exists and never shows an error dialog."""
        from . import update as _upd
        from .async_task import run_async
        if not silent:
            try:
                self.status_var.set("Checking for updates…")
            except TclError:
                pass

        # The startup check never shows an error dialog; asked for, an error
        # goes to the window's error handler.
        quiet = ((lambda e: note_handled("check for updates at startup", e))
                 if silent else self._update_check_failed)
        run_async(self, _upd.check_for_update,
                  on_done=lambda res: self._on_update_checked(res, silent),
                  on_error=quiet)

    def _update_check_failed(self, exc):
        """An asked-for check raised: the status line stops saying
        "Checking…", and the error goes to the window's error handler."""
        from .errors import report
        self._set_update_status("Couldn't check for updates.")
        report(self, exc)

    def _set_update_status(self, text):
        try:
            self.status_var.set(text)
        except TclError:
            pass

    def _on_update_checked(self, res, silent):
        if res is None:
            if not silent:
                self._set_update_status(
                    "Couldn't check for updates (offline or rate-limited).")
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
        # Two plain choices. Yes / No / Cancel, with No opening a web page,
        # reads as three ways to say yes.
        parts = offer_parts(res, kind)
        if not ask_update_or_cancel(self, "Update available", **parts):
            return
        if kind == 'installer':
            # Never pip into the installer's own Python: the next installer
            # upgrades it in place.
            _open_url(res.get('url') or _upd.PUBLIC_URL + "/releases/latest")
            self._set_update_status(
                "Opened the release page: run the new installer to update.")
            return
        self._run_update(kind)

    def _run_update(self, kind):
        from . import update as _upd
        from .async_task import run_async
        try:
            self.status_var.set("Updating OpenFlo… (this may take a minute)")
        except TclError:
            pass

        run_async(self, lambda: _upd.run_update(kind=kind),
                  on_done=lambda r: self._on_update_done(*r))

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
