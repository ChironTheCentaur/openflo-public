"""Help > Report a problem…: write a bug report or an idea and send it to
the OpenFlo developers through the website (openflo.feedback).

Nothing is sent until the person presses Send. What is sent is what they
typed, the app's version and, only when they tick the box, the tokenised
error report; "Show what is sent" shows exactly that text first.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from . import feedback, ui_errors
from .async_task import run_async
from .errors import note
from .ui_tips import tips

TITLE = "Report a problem or an idea"


# The end of the report is what is sent (at most 5000 characters), and the
# file only grows: read its last part, not all of it, on the Tk thread.
TAIL_BYTES = 64 * 1024


def _read_report(path):
    """The end of the tokenised error report (its last ``TAIL_BYTES``,
    starting at a whole line), or '' when there is none."""
    if not path:
        return ''
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            raw = f.read()
    except OSError:
        return ''
    if size > TAIL_BYTES:
        nl = raw.find(b'\n')
        raw = raw[nl + 1:] if nl >= 0 else b''
    return raw.decode('utf-8', errors='replace')


class FeedbackDialog(tk.Toplevel):
    """The report form. ``report_path`` is the tokenised error report (its
    token key file is never read; None: no report to attach); ``version``
    the running OpenFlo's (None: read it); ``sender(fields)`` posts a report
    and answers None or an error code (feedback.send; tests pass a
    stand-in)."""

    def __init__(self, editor, report_path=None, version=None, sender=None,
                 scrub=None):
        if version is None:
            from .result_stamp import app_version
            version = app_version()
        super().__init__(editor)
        self.title(TITLE)
        self._editor = editor
        self._report_path = report_path
        self._version = str(version or '')[:feedback.LIMITS['version']]
        self._sender = sender or (lambda f: feedback.send(
            f, user_agent=f"OpenFlo/{self._version}"))
        # The report was tokenised when each error was written, but by the
        # rules of the version that wrote it: it is tokenised again, by
        # today's rules, before it is shown or sent (scrub; None: as is,
        # for a dialog opened without a report).
        self._scrub = scrub or (lambda text: text)
        self.sending = False
        self.sent = False

        body = ttk.Frame(self, padding=(16, 14, 16, 6))
        body.pack(side='top', fill='both', expand=True)
        body.columnconfigure(1, weight=1)

        self.kind = tk.StringVar(value='bug')
        kinds = ttk.Frame(body)
        kinds.grid(row=0, column=0, columnspan=2, sticky='w')
        radios = []
        for key, label in feedback.KINDS.items():
            r = ttk.Radiobutton(kinds, text=label, value=key, variable=self.kind)
            r.pack(side='left', padx=(0, 12))
            radios.append(r)

        ttk.Label(body, text="Title").grid(row=1, column=0, sticky='w', pady=(10, 2))
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(body, textvariable=self.title_var, width=60)
        self.title_entry.grid(row=1, column=1, sticky='ew', pady=(10, 2))

        ttk.Label(body, text="What happened, or what would help").grid(
            row=2, column=0, columnspan=2, sticky='w', pady=(8, 2))
        self.details = tk.Text(body, width=70, height=9, wrap='word', undo=True)
        self.details.grid(row=3, column=0, columnspan=2, sticky='nsew')
        body.rowconfigure(3, weight=1)

        ttk.Label(body, text="Reply to (optional)").grid(row=4, column=0, sticky='w', pady=(8, 2))
        self.contact_var = tk.StringVar()
        contact = ttk.Entry(body, textvariable=self.contact_var, width=40)
        contact.grid(row=4, column=1, sticky='w', pady=(8, 2))

        has_report = bool(feedback.block_text(_read_report(report_path)))
        self.attach = tk.BooleanVar(value=False)
        self.attach_box = ttk.Checkbutton(
            body, variable=self.attach,
            text="Attach the error report (file paths and names, sample "
                 "names and emails replaced by tokens)" if has_report else
                 "Attach the error report (none recorded yet)")
        self.attach_box.grid(row=5, column=0, columnspan=2, sticky='w', pady=(8, 0))
        if not has_report:
            self.attach_box.state(['disabled'])

        ttk.Label(body, wraplength=520, justify='left', padding=(0, 8, 0, 0),
                  text=f"Sent through getopenflo.com to the OpenFlo developers "
                       f"as a private report, with the app's version "
                       f"({self._version or 'unknown'}). Nothing else from "
                       f"this computer is sent.").grid(
            row=6, column=0, columnspan=2, sticky='w')
        self.message = ttk.Label(body, wraplength=520, justify='left',
                                 style='Error.TLabel')
        self.message.grid(row=7, column=0, columnspan=2, sticky='w', pady=(6, 0))

        bar = ttk.Frame(self, padding=(16, 4, 16, 12))
        bar.pack(side='bottom', fill='x')
        self.send_button = ttk.Button(bar, text="Send", default='active',
                                      command=self.send_report)
        self.send_button.pack(side='right')
        cancel = ttk.Button(bar, text="Cancel", command=self.destroy)
        cancel.pack(side='right', padx=(0, 6))
        show = ttk.Button(bar, text="Show what is sent", command=self.show_sent)
        show.pack(side='left')
        open_b = ttk.Button(bar, text="Open the error report file",
                            command=self.open_report)
        open_b.pack(side='left', padx=(6, 0))
        if not has_report:
            open_b.state(['disabled'])

        tips(editor, [
            (radios[0], "A report of something that does not work as it "
                        "should."),
            (radios[1], "A suggestion: something OpenFlo could do, or do "
                        "better."),
            (self.title_entry, "A few words the report is known by (3 to "
                               "120 characters)."),
            (self.details, "What you did, what happened and what you "
                           "expected (10 to 5000 characters). Leave out "
                           "names of samples or markers you would rather "
                           "keep private."),
            (self.send_button, "Send the report now. If it cannot be sent, "
                               "what you typed stays here."),
            (cancel, "Close without sending anything."),
            (contact, "Optional. Only so the developers can ask a question "
                      "or tell you it is fixed; it stays in the private "
                      "report."),
            (self.attach_box, "Adds the end of OpenFlo's error report, "
                              "where file paths, sample names, user names "
                              "and emails are already replaced by tokens. "
                              "Channel names and other words in an error "
                              "message are not: check with Show what is "
                              "sent. Off unless you tick it."),
            (show, "Show the exact text that Send will send."),
            (open_b, "Open the tokenized error report in a text editor. Its "
                     "key file (the real values behind the tokens) stays "
                     "on this computer and is never sent."),
        ])
        self.bind('<Escape>', lambda _e: self.destroy())
        self.title_entry.focus_set()

    # -- what is sent -------------------------------------------------------
    def _report_text(self):
        """The report's end, tokenised again by today's rules."""
        text = _read_report(self._report_path)
        return self._scrub(text) if text.strip() else ''

    def fields(self):
        """The report as it will be posted (feedback.clean applied)."""
        details = self.details.get('1.0', 'end-1c')
        if self.attach.get():
            details, _how = feedback.with_report(
                details, self._report_text())
        return feedback.clean({
            'kind': self.kind.get(), 'app': 'desktop',
            'version': self._version, 'title': self.title_var.get(),
            'details': details, 'contact': self.contact_var.get(),
        })

    def show_sent(self):
        """A read-only window with the exact text Send posts."""
        f = self.fields()
        win = tk.Toplevel(self)
        win.title("What is sent")
        text = tk.Text(win, width=80, height=24, wrap='word')
        text.pack(fill='both', expand=True, padx=8, pady=8)
        text.insert('1.0', "\n".join(
            [f"{feedback.KINDS.get(f['kind'], f['kind'])}",
             f"App: OpenFlo desktop, version {f['version'] or 'unknown'}",
             f"Reply to: {f['contact'] or 'not given'}",
             f"Title: {f['title']}", "", f['details']]))
        text.configure(state='disabled')
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))
        win.bind('<Escape>', lambda _e: win.destroy())
        return text

    def open_report(self):
        """Open the tokenised report in the system's text viewer."""
        try:
            import subprocess
            import sys
            p = self._report_path
            if not p:
                return
            if sys.platform == 'win32':
                os.startfile(p)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', p])
            else:
                subprocess.Popen(['xdg-open', p])
        except Exception as exc:  # noqa: BLE001 - noted; opening a viewer is a convenience
            note("open the error report", exc)

    # -- sending ------------------------------------------------------------
    def _say(self, text):
        self.message.configure(text=text)

    def send_report(self):
        """Check the report, then post it off the Tk thread. What was typed
        stays in the form on any failure."""
        if self.sending:
            return
        f = self.fields()
        if self.attach.get():
            _text, how = feedback.with_report(
                self.details.get('1.0', 'end-1c'), self._report_text())
            if how == 'none':
                self._say("There is no room left for the error report: "
                          "shorten the details or untick the box.")
                return
        bad = feedback.check(f)
        if bad:
            self._say(feedback.ERRORS[bad])
            return
        self.sending = True
        self.send_button.state(['disabled'])
        self._say("Sending…")
        run_async(self, lambda: self._sender(f), on_done=self._sent,
                  on_error=lambda _e: self._sent('unavailable'))

    def _sent(self, error):
        self.sending = False
        try:
            self.send_button.state(['!disabled'])
        except tk.TclError:
            return
        if error:
            more = ("" if error in ('title', 'details', 'contact', 'kind')
                    else f" You can also use the form at {feedback.FORM_URL}.")
            self._say(feedback.ERRORS.get(error, feedback.ERRORS['unavailable']) + more)
            return
        self.sent = True
        self._say("")
        ui_errors.showinfo(TITLE, "Thank you. The report reached the "
                            "OpenFlo developers.", parent=self)
        self.destroy()
