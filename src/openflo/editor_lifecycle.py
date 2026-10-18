"""Window lifecycle: geometry, shortcuts, close, resume, and glue dialogs.

Self-contained slice of ViewGateEditorWindow (see editor_base.EditorMixin).
"""
from __future__ import annotations

import json
import os
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .editor_base import EditorMixin
from .prefs import read_prefs, write_pref
from .ui_errors import show_error
from .update import ISSUES_URL


def _error_report_path():
    d = os.path.join(os.path.expanduser('~'), '.openflo')
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, 'error_report.log')        # submittable (tokenised)


def _error_keymap_path():
    return os.path.splitext(_error_report_path())[0] + '.keys.json'  # LOCAL only


def _tokenise_for_report(text, extra_values=()):
    """Replace sensitive substrings in ``text`` with stable tokens, persisting
    the token→value map to the local sister key file. ``extra_values`` are
    exact strings the caller knows are sensitive (e.g. loaded sample names and
    their file paths). Returns the tokenised text safe to submit."""
    import getpass
    import socket

    s = str(text)
    # Load / build the persistent value→token map (so a path tokenised last
    # week keeps the same token today).
    try:
        with open(_error_keymap_path(), encoding='utf-8') as f:
            tok_to_val = json.load(f)
    except Exception:
        tok_to_val = {}
    val_to_tok = {v: k for k, v in tok_to_val.items()}
    counters = {}
    for tok in tok_to_val:
        kind = tok.strip('<>').split(':', 1)[0]
        n = int(tok.strip('<>').rsplit(':', 1)[-1])
        counters[kind] = max(counters.get(kind, 0), n)

    def _tok(kind, value):
        if not value or value in val_to_tok:
            return val_to_tok.get(value, value)
        counters[kind] = counters.get(kind, 0) + 1
        token = f"<{kind}:{counters[kind]}>"
        val_to_tok[value] = token
        tok_to_val[token] = value
        return token

    # Gather sensitive values, longest first so a path is replaced before the
    # bare username/home nested inside it.
    values = []
    for v in extra_values:
        if v and isinstance(v, str):
            values.append(('id', v))
    for m in re.findall(r'[A-Za-z]:\\[^\s\'"|<>]+', s):       # Windows paths
        values.append(('path', m))
    for m in re.findall(r'(?<![\w.])(?:/[^/\s\'"|<>]+){2,}', s):  # POSIX paths
        values.append(('path', m))
    for m in re.findall(r'[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}', s):   # emails
        values.append(('email', m))
    try:
        host = socket.gethostname()
        if host and host in s:
            values.append(('host', host))
    except Exception:
        pass
    try:
        user = getpass.getuser()
        if user and user in s:
            values.append(('user', user))
    except Exception:
        pass
    home = os.path.expanduser('~')
    if home and home in s:
        values.append(('path', home))

    for kind, value in sorted(set(values), key=lambda kv: -len(kv[1])):
        s = s.replace(value, _tok(kind, value))

    try:
        with open(_error_keymap_path(), 'w', encoding='utf-8') as f:
            json.dump(tok_to_val, f, indent=2)
    except Exception:
        pass
    return s


class LifecycleMixin(EditorMixin):
    """Persist/restore geometry, bind shortcuts, handle close and session resume, the menu status bar, and small glue dialogs."""

    def _focus_in_text(self):
        """True when a text-entry widget has focus, so Ctrl+Z/Y should edit
        the text rather than the gate history."""
        try:
            w = self.focus_get()
            return bool(w) and w.winfo_class() in ('TEntry', 'Entry',
                                                   'TCombobox', 'Text')
        except Exception:
            return False

    def _on_menu_select(self, event):
        """``<<MenuSelect>>`` handler — show the highlighted entry's help in
        the status bar (only while hover tips are enabled)."""
        try:
            if not self._tooltips_enabled.get():
                return
            menu = event.widget
            idx = menu.index('active')
            if idx in (None, 'none'):
                return
            if menu.type(idx) not in ('command', 'cascade', 'checkbutton',
                                      'radiobutton'):
                return
            label = menu.entrycget(idx, 'label')
        except Exception:
            return
        help_text = self._MENU_ITEM_HELP.get(label)
        if not help_text:
            return
        # Remember what the bar said before the menu opened, so closing it
        # (without picking anything) leaves the status as we found it.
        if self._status_before_menu is None:
            self._status_before_menu = self.status_var.get()
        self.status_var.set(help_text)

    def _on_menu_unmap(self, _event=None):
        """A dropdown closed — restore the pre-menu status text."""
        if self._status_before_menu is not None:
            try:
                self.status_var.set(self._status_before_menu)
            except Exception:
                pass
            self._status_before_menu = None

    def _upgrade_session_file(self):
        """File → Upgrade saved session…: migrate an older .flowsession to the
        current schema and write it back out, without opening it. For batch /
        downstream users who want to refresh saved files; the same upgrade also
        happens automatically when a session is opened."""
        from .session_format import SessionVersionError, migrate_session
        path = filedialog.askopenfilename(
            title="Select a session to upgrade",
            filetypes=[("OpenFlo session", "*" + self.SESSION_EXT),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            data, notes = migrate_session(data)
        except SessionVersionError as exc:
            messagebox.showerror("Upgrade failed", str(exc), parent=self)
            return
        except Exception as exc:                       # noqa: BLE001
            show_error(self, "Upgrade failed",
                       "The session could not be upgraded.", exc, path)
            return
        if not notes:
            messagebox.showinfo(
                "Already current",
                f"{os.path.basename(path)} is already the current session "
                "format — nothing to upgrade.", parent=self)
            return
        out = filedialog.asksaveasfilename(
            title="Save upgraded session as",
            defaultextension=self.SESSION_EXT,
            initialfile=os.path.splitext(os.path.basename(path))[0]
            + '_upgraded' + self.SESSION_EXT,
            filetypes=[("OpenFlo session", "*" + self.SESSION_EXT)])
        if not out:
            return
        try:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:                       # noqa: BLE001
            show_error(self, "Write failed",
                       "The upgraded session could not be written.", exc,
                       out)
            return
        messagebox.showinfo(
            "Session upgraded",
            "Upgraded to the current format:\n  " + "\n  ".join(notes)
            + f"\n\nSaved → {os.path.basename(out)}", parent=self)
        self.status_var.set(f"Upgraded session → {os.path.basename(out)}")

    def _report_callback_exception(self, exc, val, tb):
        """Tk routes every unhandled callback error here (installed on the
        root). The traceback prints to stderr, which the log tee mirrors into
        the log pane; the status bar is flagged; a SCRUBBED copy goes to a
        submittable report file (Help, Report a problem…); and the first one
        opens an error window in plain words, with the traceback under Show
        details. (It used to pop the log and console open, a developer's
        view, which people took for a crash.)"""
        import traceback
        text = ''.join(traceback.format_exception(exc, val, tb))
        # The console is local to the user, so it shows the REAL traceback
        # (paths intact = more useful). Unchanged path: stderr → tee → pane.
        print(text, file=sys.stderr, flush=True)
        # Submittable copy: tokenise sensitive values (paths/emails/host/user
        # + the loaded sample names and their files), keymap kept locally.
        try:
            extra = list(self._samples.keys())
            for s in self._samples.values():
                p = getattr(s, 'path', None)
                if p:
                    extra.append(str(p))
            tokenised = _tokenise_for_report(text, extra)
            with open(_error_report_path(), 'a', encoding='utf-8') as f:
                f.write(f"\n----- {val.__class__.__name__} -----\n{tokenised}")
        except Exception:
            pass
        self._error_count = getattr(self, '_error_count', 0) + 1
        # Flag it, and say so once in a window that doesn't wait (it may
        # repeat on every redraw; the status bar counts the rest).
        try:
            self.status_var.set(
                "⚠ Something went wrong. Help, Report a problem… sends the "
                "details to the developers.")
            if self._error_count == 1:
                from .ui_errors import ErrorDialog, error_details, plain_error
                ErrorDialog(
                    self, "Something went wrong",
                    "OpenFlo hit a problem it did not expect, and the last "
                    "action may not have finished. Your session is still "
                    "open; saving it now is a good idea.\n\n"
                    + plain_error(val), error_details(val))
        except Exception:
            pass

    def _open_preferences(self):
        """Consolidated settings dialog (theme, hover tips)."""
        from .ui_preferences import PreferencesDialog
        PreferencesDialog(self)

    def _report_a_problem(self):
        """Open the tokenised error report and the issue tracker so the user
        can submit a bug. Sensitive values (paths, sample names, emails) are
        replaced by tokens; the token→value key stays in a LOCAL sister file
        that is NOT meant to be submitted (see _tokenise_for_report)."""
        path = _error_report_path()
        keys = _error_keymap_path()
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        msg = (
            "OpenFlo keeps a tokenised error report you can attach to a bug "
            "report. File paths, sample names, usernames and emails are "
            "replaced with tokens (e.g. <path:1>, <id:2>).\n\n"
            f"Submit this file (safe — tokenised):\n    {path}\n\n"
            f"Keep this one PRIVATE (maps tokens → real values, for your own "
            f"decoding — do NOT submit):\n    {keys}\n\n"
            + ("Opening the report now. " if exists else
               "No errors have been recorded yet. ")
            + f"File issues at:\n    {ISSUES_URL}")
        messagebox.showinfo("Report a problem", msg, parent=self)
        try:
            import subprocess
            import webbrowser
            if exists:
                if sys.platform == 'win32':
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', path])
                else:
                    subprocess.Popen(['xdg-open', path])
            webbrowser.open(ISSUES_URL)
        except Exception:
            pass

    def _restore_geometry(self, default):
        """Return the saved 'WxH+X+Y' geometry if it's sane and on-screen,
        else the default (so an unplugged monitor can't strand the window)."""
        g = read_prefs().get('geometry')
        if not isinstance(g, str):
            return default
        m = re.match(r'(\d+)x(\d+)([+-]\d+)([+-]\d+)$', g)
        if not m:
            return default
        w, h, x, y = (int(m.group(1)), int(m.group(2)),
                      int(m.group(3)), int(m.group(4)))
        try:
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        except Exception:
            return default
        if w < 800 or h < 500 or w > sw + 80 or h > sh + 80:
            return default                      # absurd size → default
        if x < -80 or y < -40 or x > sw - 100 or y > sh - 100:
            return f"{w}x{h}"                   # offscreen position → size only
        return g

    def _save_geometry(self):
        """Persist the current size/position (primary window only)."""
        if not self._primary:
            return
        try:
            # state()=='zoomed' (maximised) reports the restored geometry on
            # some platforms; saving it is fine — reopens maximised-ish.
            write_pref('geometry', self.geometry())
        except Exception:
            pass

    def _bind_shortcuts(self):
        """Keyboard accelerators (undo, redo and clipboard are bound elsewhere).

        Grouped: File/Help (existing), then the gating-loop verbs that were
        previously mouse-only — Find, reset/zoom the plot view, replot — plus
        Preferences and a couple of panel/figure conveniences."""
        for seq, fn in (
                # File / Help
                ('<Control-o>', self._load_session),
                ('<Control-s>', self._save_session),
                ('<Control-e>', self._export_flowjo_wsp),
                ('<Control-w>', self._on_close),
                ('<Control-q>', self._on_close),
                ('<F1>', self._open_guide),
                ('<Control-Shift-A>', self._add_samples),
                ('<Control-Shift-S>', self._save_plot_image),
                # Edit
                ('<Control-comma>', self._open_preferences),
                # Navigation / view (the gating loop)
                ('<Control-f>', self._focus_find),
                ('<Control-Key-0>', self._reset_plot_view),
                ('<Control-equal>', lambda: self._zoom_step(1 / 1.25)),
                ('<Control-plus>', lambda: self._zoom_step(1 / 1.25)),
                ('<Control-KP_Add>', lambda: self._zoom_step(1 / 1.25)),
                ('<Control-minus>', lambda: self._zoom_step(1.25)),
                ('<Control-KP_Subtract>', lambda: self._zoom_step(1.25)),
                ('<F5>', lambda: self._schedule_replot(0)),
                ('<Escape>', self._cancel_active_tool),
                # Display mode (gating view)
                ('<Control-Key-1>', lambda: self._set_display_mode('all')),
                ('<Control-Key-2>', lambda: self._set_display_mode('highlight')),
                ('<Control-Key-3>', lambda: self._set_display_mode('filter')),
                # Analyze
                ('<Control-t>', self._open_stats_window),
                # Panels
                ('<F9>', self._open_pipeline_workspace),
                ('<Control-grave>', self._toggle_log_shortcut)):
            seqs = [seq]
            # On macOS the same shortcuts answer to Command, which the menus
            # show (editor_menu.accel).
            if sys.platform == 'darwin' and 'Control-' in seq:
                seqs.append(seq.replace('Control-', 'Command-'))
            for s in seqs:
                try:
                    self.bind(s, lambda _e, f=fn: (f(), 'break')[1])
                except Exception:
                    pass

    def _focus_find(self):
        """Ctrl+F — focus and select the Find box above the sample/gate tree."""
        ent = getattr(self, '_find_entry', None)
        if ent is None:
            return
        try:
            ent.focus_set()
            ent.selection_range(0, 'end')
        except Exception:
            pass

    def _cancel_active_tool(self):
        """Escape — back out of the zoom-to tool if it's armed (a no-op
        otherwise, so it won't swallow Escape from anything else)."""
        try:
            if getattr(self, '_zoom_mode', False) or self._zoom_mode_var.get():
                self._zoom_mode_var.set(False)
                self._toggle_zoom_tool()
        except Exception:
            pass

    def _ask_close_choice(self):
        """Save and close / Close without saving / Cancel. Saving a big
        session can take a while (every sample's data is written), so the
        choice is the user's. Returns 'save', 'discard' or 'cancel'."""
        dlg = tk.Toplevel(self)
        dlg.title("Close OpenFlo")
        dlg.resizable(False, False)
        try:
            dlg.transient(self.winfo_toplevel())
        except Exception:                             # noqa: BLE001
            pass
        choice = {'v': 'cancel'}
        body = ttk.Frame(dlg, padding=16)
        body.pack(fill='both', expand=True)
        ttk.Label(body, justify='left', wraplength=420, text=(
            "Save this session before closing?\n\nSaving writes the data "
            "of every sample that changed since the last save, which can "
            "take a minute for a large session; a progress window shows how "
            "far it has got. The saved session reopens on the next "
            "launch.")).pack(anchor='w')
        bb = ttk.Frame(body)
        bb.pack(anchor='e', pady=(14, 0))

        def pick(v):
            choice['v'] = v
            dlg.destroy()
        ttk.Button(bb, text="Cancel", command=lambda: pick('cancel')).pack(
            side='right')
        ttk.Button(bb, text="Close without saving",
                   command=lambda: pick('discard')).pack(side='right',
                                                         padx=(0, 6))
        save = ttk.Button(bb, text="Save and close",
                          command=lambda: pick('save'))
        save.pack(side='right', padx=(0, 6))
        dlg.bind('<Escape>', lambda _e: pick('cancel'))
        dlg.protocol('WM_DELETE_WINDOW', lambda: pick('cancel'))
        try:
            save.focus_set()
            dlg.grab_set()
        except Exception:                             # noqa: BLE001
            pass
        dlg.wait_window()
        return choice['v']

    def _on_close(self, ask=True):
        """Close the editor, first asking whether to save the session (when
        there is one): Save and close autosaves it, Close without saving
        doesn't, Cancel keeps the window open. When this editor is the app's
        primary window, closing it tears the whole app down (kills any
        running pipeline subprocess via App.shutdown, which destroys the
        root + this editor); otherwise it just closes this Toplevel.

        The save runs on a worker thread behind a small progress window, so
        a big session (a minute of sidecar CSVs) doesn't freeze every window
        into "Not Responding". A second close while it runs only raises that
        window: each extra click used to queue another full save."""
        if getattr(self, '_closing', False):
            prog = getattr(self, '_close_progress', None)
            if prog is not None:
                try:
                    prog.lift()
                except Exception:
                    pass
            return
        save = True
        if ask and self._samples:
            choice = self._ask_close_choice()
            if choice == 'cancel':
                return
            save = choice == 'save'
        # The session file gets this save too (below), but never over a
        # save made in another window since this one opened or saved it:
        # the user picks a copy, Overwrite, or Cancel (keep the window).
        own = None
        if self._samples and save:
            own = self._open_session_file()
            if own:
                own = self._confirm_session_target(own, closing=True)
                if own is None:
                    return
        self._closing = True
        # Wake any blocked load workers so they exit instead of holding the
        # queue forever (daemon=True is the backstop). Best-effort, non-blocking
        # — we don't join, so a slow in-flight FlowSample can't hang the close.
        self._save_geometry()
        # Stop the background plot warm-up and drop its caches (also bound to
        # <Destroy>, for closes that don't come through here).
        try:
            self._pc_shutdown()
        except Exception:
            pass
        try:
            self._load_stop.set()
            # One sentinel per live worker (the pool size is dynamic now).
            # Priority -1 so blocked workers pick it up immediately.
            for _ in range(max(1, len(self._load_pool))):
                self._enqueue_load(None, priority=-1)
        except Exception:
            pass
        if not (self._samples and save):
            self._finish_close(None)
            return
        try:
            # The session file the user opened or saved gets this save too,
            # not only the autosave: closing used to leave that file as it
            # was at the last Ctrl+S. The autosave still follows, so the
            # session reopens on the next launch as the close prompt says.
            plans = []
            if own:
                plans.append(self._plan_session_write(own))
            plans.append(self._plan_session_write(
                self._session_autosave_path()))
        except Exception as exc:
            self._finish_close(exc)
            return
        self._start_close_save(plans)

    def _start_close_save(self, plans):
        """Run the session-write `plans` (one plan, or a list written in
        order) on a worker thread and poll it from the Tk loop, showing
        "Saving session… n of N" until it ends."""
        import threading
        if isinstance(plans, dict):
            plans = [plans]
        state = {'step': (0, len(plans[0]['jobs']) + 1, ''), 'done': False,
                 'error': None}

        def progress(done, total, name):
            state['step'] = (done, total, name)

        def work():
            failed: list = []
            for plan in plans:
                try:
                    _data, bad = self._run_session_write(plan, progress)
                    failed.extend(n for n in bad if n not in failed)
                except Exception as exc:           # noqa: BLE001
                    if state['error'] is None:
                        state['error'] = exc
            state['failed'] = failed
            state['done'] = True

        self._close_progress = self._make_close_progress()
        self._close_thread = threading.Thread(
            target=work, name='openflo-close-save', daemon=False)
        self._close_thread.start()

        def poll():
            if not state['done']:
                self._show_close_progress(*state['step'])
                self.after(100, poll)
                return
            self._close_thread.join()
            prog, self._close_progress = self._close_progress, None
            self._close_progress_ui = None
            if prog is not None:
                try:
                    prog.grab_release()
                    prog.destroy()
                except Exception:
                    pass
            self._finish_close(state['error'], state.get('failed') or [])
        self.after(0, poll)

    def _make_close_progress(self):
        """A small modal "Saving session…" window over this editor. Its grab
        keeps the session from being edited while the worker writes it."""
        try:
            top = tk.Toplevel(self)
            top.title("Closing OpenFlo")
            top.transient(self)  # type: ignore[arg-type]
            top.resizable(False, False)
            top.protocol('WM_DELETE_WINDOW', lambda: None)
            frm = ttk.Frame(top, padding=14)
            frm.pack(fill='both', expand=True)
            msg = tk.StringVar(master=top, value="Saving session…")
            name = tk.StringVar(master=top, value='')
            ttk.Label(frm, textvariable=msg).pack(anchor='w')
            ttk.Label(frm, textvariable=name,
                      style='Muted.TLabel').pack(anchor='w', pady=(2, 6))
            bar = ttk.Progressbar(frm, length=320, mode='determinate')
            bar.pack(fill='x')
            self._close_progress_ui = (msg, name, bar)
            top.update_idletasks()
            try:
                x = self.winfo_rootx() + (self.winfo_width()
                                          - top.winfo_reqwidth()) // 2
                y = self.winfo_rooty() + (self.winfo_height()
                                          - top.winfo_reqheight()) // 3
                top.geometry(f"+{max(0, x)}+{max(0, y)}")
            except Exception:
                pass
            try:
                top.grab_set()
            except Exception:
                pass
            return top
        except Exception:
            return None

    def _show_close_progress(self, done, total, name):
        ui = getattr(self, '_close_progress_ui', None)
        if ui is None:
            return
        msg, name_var, bar = ui
        try:
            if done < total - 1:
                msg.set(f"Saving session… {done + 1} of {total - 1} samples")
                name_var.set(name)
            else:
                msg.set("Saving session… writing the session file")
                name_var.set('')
            bar.configure(maximum=max(1, total), value=done)
        except Exception:
            pass

    def _finish_close(self, error, failed=()):
        """Report how the close-time autosave went, then close. `error` is
        the exception the save raised, or None; `failed` names the samples
        whose sidecar could not be written."""
        # Report BEFORE detaching the log sinks: a print() emitted after the
        # tees are removed reaches nothing the user can see, and this is the
        # last moment the session state still exists. A failure here means the
        # work is gone and the next launch has nothing to resume, so it is
        # raised as a dialog rather than a message into a closing pane.
        if error is not None:
            print(f"[session] autosave failed: {error}", flush=True)
            try:
                messagebox.showerror(
                    "Autosave failed — session not saved",
                    f"OpenFlo could not autosave this session:\n\n{error}\n\n"
                    f"Nothing will be restored on the next launch. Cancel the "
                    f"close and use File → Save session if the work matters.",
                    parent=self)
            except Exception:                          # noqa: BLE001
                pass
        elif failed:
            # Same condition _save_session warns about: the sample
            # reloads from raw FCS with its clusters / UMAP / FMO gates
            # gone. Silently degrading on exit hid that entirely.
            try:
                messagebox.showwarning(
                    "Session autosaved WITH WARNINGS",
                    "Computed results could not be written for:\n\n  "
                    + "\n  ".join(list(failed)[:8])
                    + ("\n  …" if len(failed) > 8 else "")
                    + "\n\nThese samples will reload from raw FCS on "
                      "resume, without their clusters / embeddings / FMO "
                      "gates.", parent=self)
            except Exception:                      # noqa: BLE001
                pass
        try:
            self._release_session_lock()
        except Exception:                          # noqa: BLE001
            pass
        # Stop mirroring stdout/stderr into this (closing) editor's pane.
        for tee in getattr(self, '_log_tees', []):
            try:
                tee.remove_sink(self._log_queue)
            except Exception:
                pass
        if self._primary and self._app is not None:
            try:
                self._app.destroy()      # destroy the Tk root → exits mainloop
                return
            except Exception:
                pass
        self.destroy()

    def _open_pipeline_workspace(self):
        """Toggle the docked Pipeline Workspace pane. Showing it splits the
        plot area via the sash; hiding it gives the plot the full width."""
        host = getattr(self, '_ws_host', None)
        if host is None:
            return
        # If it's floating, the menu toggle re-docks it rather than erroring.
        if getattr(self, '_ws_popped', False):
            self._redock_workspace()
            return
        if getattr(self, '_workspace_shown', False):
            try:
                self._editor_paned.forget(host)
            except Exception:
                pass
            self._workspace_shown = False
            self.status_var.set("Pipeline workspace hidden.")
            return
        try:
            self._ensure_workspace_panel()   # build on first reveal
            self._editor_paned.add(host, weight=3)
            self._workspace_shown = True
            self.update_idletasks()
            try:
                total = self._editor_paned.winfo_width()
                if total > 100:
                    # Open the workspace wide enough for its controls, but
                    # never more than 60% (keep the plot usable).
                    #
                    # The floor is MEASURED, not guessed: ask the panel what it
                    # actually needs. The previous flat 320 px was a guess that
                    # happened to be 17 px short of the real toolbar width, so
                    # the 'Group v' menubutton opened clipped on a normal
                    # monitor. A measured floor also stays correct when someone
                    # adds a control, instead of silently going stale.
                    need = 0
                    try:
                        host.update_idletasks()
                        need = host.winfo_reqwidth()
                    except Exception:
                        pass
                    floor = max(320, min(need + 8, 520))
                    panel_w = min(max(floor, int(total * 0.38)),
                                  int(total * 0.6))

                    # GROW rather than steal, when the screen allows. Docking
                    # the workspace used to take its width out of the plot
                    # column: at the default 1500px window that dropped the
                    # plot side to ~512px while its control rows need ~754, so
                    # the axis combos, 'Auto-gate' and 'Show cleaned-out
                    # events' were all clipped -- controls the user had not
                    # asked to give up. Widening the window instead keeps them
                    # intact. Skipped when maximised or when the screen has no
                    # room, where sharing is the only option.
                    try:
                        if self.state() == 'normal':
                            cur_w = self.winfo_width()
                            cur_h = self.winfo_height()
                            room = self.winfo_screenwidth() - 40
                            grow = min(panel_w, max(0, room - cur_w))
                            if grow > 40:
                                self.geometry(f'{cur_w + grow}x{cur_h}')
                                self.update_idletasks()
                                total = self._editor_paned.winfo_width()
                    except Exception:
                        pass
                    self._editor_paned.sashpos(0, max(60, total - panel_w))

                    # Second pass: the plot side carries control rows whose
                    # required width is independent of the panel, so growing by
                    # the panel width alone can still leave them a few dozen
                    # pixels short. Ask what it actually needs and close the
                    # gap, once, still bounded by the screen.
                    try:
                        bar = getattr(self, '_ctrl_bar', None)
                        short = 0 if bar is None else (
                            bar.winfo_reqwidth() - bar.winfo_width())
                        if short > 0 and self.state() == 'normal':
                            cur_w = self.winfo_width()
                            room = self.winfo_screenwidth() - 40 - cur_w
                            # +8: the sash lands on whole pixels, so an
                            # exact fit can still round a few px short.
                            extra = min(short + 8, max(0, room))
                            if extra > 0:
                                self.geometry(f'{cur_w + extra}x{self.winfo_height()}')
                                self.update_idletasks()
                                total = self._editor_paned.winfo_width()
                                self._editor_paned.sashpos(
                                    0, max(60, total - panel_w))
                    except Exception:
                        pass
            except Exception:
                pass
            self.status_var.set(
                "Pipeline workspace shown. Drag samples / gate leaves in; each tab is a separate query.")
        except Exception as exc:
            self.status_var.set(f"Couldn't show pipeline workspace: {exc}")

    def _maybe_resume_session(self):
        """On open, if a non-empty autosaved session exists, offer to
        resume it. Only prompts when the editor opened empty (don't
        clobber samples the caller passed in)."""
        if self._samples:
            return
        self._prune_autosaves()
        # Offer the newest ORPHANED autosave (one whose owning instance has
        # exited); a file still owned by another running instance is skipped so
        # two concurrent instances don't fight over the same session.
        path = self._find_resumable_session()
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return
        # Valid JSON is not necessarily a session. A file whose top level is an
        # array, string or number parses fine and then blows up on `.get`
        # below — an uncaught AttributeError DURING STARTUP, outside the
        # handler above. `_find_resumable_session` appends the legacy
        # `last_session.flowsession` unconditionally, which is the plausible
        # way a foreign-format file reaches here.
        if not isinstance(data, dict):
            return
        n = len(data.get('samples', []))
        if n == 0:
            return
        # The session loads FCS/CSV through openflo.pipeline → flowio. If the
        # data dependencies aren't installed (e.g. a bare clone with no
        # `pip install`), say so plainly rather than prompting and then failing
        # every sample load.
        try:
            import flowio  # noqa: F401
        except Exception:
            messagebox.showwarning(
                "Data dependencies not installed",
                "An auto-saved session was found, but OpenFlo's data "
                "libraries (FlowIO, etc.) aren't installed in this "
                "environment, so it can't be opened.\n\n"
                "Install them, then reopen:\n"
                "    pip install -e .\n\n"
                "(or run the bundled openflo-gui launcher, which installs "
                "them for you). Starting with an empty session.",
                parent=self)
            return
        when = data.get('created', 'unknown time')
        if messagebox.askyesno(
                "Resume last session?",
                f"Found an auto-saved session from {when} with "
                f"{n} sample(s).\n\nResume it?",
                parent=self):
            # Run the same schema migrate/version check as a manual Load, so an
            # autosave from an older OpenFlo is upgraded (and one from a NEWER
            # build is refused) rather than mis-read.
            from .session_format import SessionVersionError, migrate_session
            try:
                data, _ = migrate_session(data)
            except SessionVersionError:
                messagebox.showwarning(
                    "Can't resume session",
                    "The auto-saved session was written by a newer OpenFlo and "
                    "can't be read by this build. Starting with an empty "
                    "session.", parent=self)
                return
            # Only AFTER the session is accepted: set the dir so relative
            # processed-data sidecars resolve against the autosave location
            # (not the CWD). Assigning before the migrate left both paths
            # pointing at a REJECTED file when the version check refused it.
            self._session_dir = os.path.dirname(os.path.abspath(path))
            self._session_data_dir = (
                os.path.splitext(os.path.abspath(path))[0] + '_data')
            self._apply_session(data)
