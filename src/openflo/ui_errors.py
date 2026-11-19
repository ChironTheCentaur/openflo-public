"""Error messages people can read: what went wrong in plain words, with the
technical description one click away under "Show details".

Every error, warning and notice window in the app is ``ErrorDialog``: its
text can be selected and copied, a Copy button copies all of it, and web
links and file paths in it open when clicked. ``showerror``, ``showwarning``
and ``showinfo`` here take the same arguments as ``tkinter.messagebox``'s.

Every failure a window reports goes through ``show_error``. The message box
does not lead with the exception itself ("KeyError: 'x'"): the first line says
what OpenFlo was doing, the next says why in plain words (``plain_error``),
and the exception and its traceback wait behind the Details toggle, with a
button to copy them into a bug report.

Self-contained Tk window (see ui_*.py convention).
"""
from __future__ import annotations

import errno
import os
import re
import tkinter as tk
import traceback
from tkinter import ttk

from .clipboard_export import clipboard_put


def _where(exc):
    name = getattr(exc, 'filename', None)
    return f" ({os.path.basename(str(name))})" if name else ""


def plain_error(exc) -> str:
    """One or two sentences saying, without programming terms, why ``exc``
    happened."""
    where = _where(exc)
    if isinstance(exc, FileNotFoundError):
        return f"The file or folder could not be found{where}."
    if isinstance(exc, PermissionError):
        return ("OpenFlo was not allowed to open or write the file" + where
                + ". It may be open in another program, or the folder may "
                  "be read-only.")
    if isinstance(exc, IsADirectoryError):
        return f"A folder was given where a file was expected{where}."
    if isinstance(exc, NotADirectoryError):
        return f"A file was given where a folder was expected{where}."
    if isinstance(exc, MemoryError):
        return ("There is not enough memory for this. Close other programs, "
                "or load fewer samples or events, and try again.")
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "The disk is full. Free some space and try again."
    if isinstance(exc, UnicodeError):
        return ("The file has characters OpenFlo could not read as text; it "
                "may not be the kind of file expected.")
    if isinstance(exc, KeyError):
        key = exc.args[0] if exc.args else ''
        return f"Something this needs was not found: {key!s}."
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        mod = getattr(exc, 'name', None) or ''
        return ("A component this needs is not installed"
                + (f" ({mod})" if mod else "") + ". Help, Environment "
                "lists what is installed.")
    text = str(exc).strip()
    if not text:
        return "Something went wrong that OpenFlo did not expect."
    text = text[0].upper() + text[1:]
    return text if text.endswith(('.', '!', '?', ')')) else text + '.'


def error_details(exc) -> str:
    """The technical description: the exception and, when it was raised,
    its traceback."""
    if exc is None:
        return ''
    if getattr(exc, '__traceback__', None) is not None:
        return ''.join(traceback.format_exception(type(exc), exc,
                                                  exc.__traceback__)).strip()
    return f"{type(exc).__name__}: {exc}"


_URL = re.compile(r'https?://[^\s<>"\'\]]+')
_PATH_START = re.compile(r'(?<![\w/.:])(?:[A-Za-z]:[\\/]|/(?=[^\s/]))')
_TRAIL = '.,;:!?)\'"'


def _trim(text):
    return text.rstrip().rstrip(_TRAIL)


def _path_at(line, start):
    """The longest path that starts at ``start`` in ``line`` and exists,
    else the rest of the line when its folder exists (a file that was not
    found), else ''. Paths may hold spaces, so the end is found by trying
    each word boundary from the right."""
    rest = line[start:]
    ends = [len(rest)] + [m.start() for m in re.finditer(r'\s', rest)][::-1]
    for end in ends:
        cand = _trim(rest[:end])
        if len(cand) > 3 and os.path.exists(cand):
            return cand
    whole = _trim(rest)
    if len(whole) > 3 and os.path.isdir(os.path.dirname(whole)):
        return whole
    return ''


def find_links(text):
    """(start, end, kind, target) for each web link ('url') and each file or
    folder path ('path') in ``text``, as character offsets."""
    found = []
    for m in _URL.finditer(text):
        url = _trim(m.group())
        found.append((m.start(), m.start() + len(url), 'url', url))
    taken = [(a, b) for a, b, _k, _t in found]
    offset = 0
    for line in text.split('\n'):
        for m in _PATH_START.finditer(line):
            a = offset + m.start()
            if any(lo <= a < hi for lo, hi in taken):
                continue
            path = _path_at(line, m.start())
            if path:
                found.append((a, a + len(path), 'path', path))
                taken.append((a, a + len(path)))
        offset += len(line) + 1
    return sorted(found)


def open_link(kind, target):
    """Open a web link in the browser, or show a path's folder."""
    if kind == 'url':
        import webbrowser
        return webbrowser.open(target)
    from .ui_notice import open_folder
    return open_folder(target)


_ICONS = {'error': 'error', 'warning': 'warning', 'info': 'information'}


class ErrorDialog(tk.Toplevel):
    """One window for every error, warning and notice: an icon, the message
    (text that can be selected, with web links and file paths that open when
    clicked), then [Copy] [Show details] ... [Close]. Details start hidden;
    Return and Escape close it. Sky, 11-18: a message box's text cannot be
    selected or copied."""

    def __init__(self, parent, title="Error", message='', details='',
                 kind='error'):
        super().__init__(parent)
        self.title(title)
        # Vertical resize too: a long message scrolls, and a taller window
        # shows more of it.
        self.resizable(True, True)
        self.message = message
        self.details = details
        self.kind = kind
        self.links = []
        try:
            top = parent.winfo_toplevel() if parent is not None else None
            if top is not None and str(top) != '.':
                self.transient(top)
        except tk.TclError:
            pass
        body = ttk.Frame(self, padding=(16, 14, 16, 8))
        body.pack(side='top', fill='both', expand=True)
        icon = _ICONS.get(kind, 'error')
        try:
            # Tk's own message-box icons, made when its script is loaded.
            self.tk.eval('catch {auto_load ::tk::MessageBox}')
            self.icon_label = ttk.Label(body, image=f'::tk::icons::{icon}')
        except tk.TclError:
            self.icon_label = tk.Label(body, bitmap=icon[:4] if icon == 'information' else icon)
        self.icon_label.pack(side='left', anchor='n', padx=(0, 12))
        self.message_frame = ttk.Frame(body)
        self.message_frame.pack(side='top', fill='both', expand=True)
        self.message_box = self._message_text(self.message_frame, message)
        self.message_box.grid(row=0, column=0, sticky='nsew')
        self.message_frame.columnconfigure(0, weight=1)
        self.message_frame.rowconfigure(0, weight=1)

        bar = ttk.Frame(self, padding=(16, 0, 16, 12))
        bar.pack(side='bottom', fill='x')
        self.close_button = ttk.Button(bar, text="Close", default='active',
                                       command=self.destroy)
        self.close_button.pack(side='right')
        self.copy_button = ttk.Button(bar, text="Copy", command=self.copy)
        self.copy_button.pack(side='left')
        from .ui_tips import auto_tip
        auto_tip(self.copy_button, "Copy this message"
                 + (" and its technical description" if details else "")
                 + ", to paste into an email or a bug report "
                 "(Help, Report a problem…).")
        self.details_button = None
        self._box = tk.Text(self, height=10, width=72, wrap='none',
                            font='OpenFloMono')
        if details:
            self._box.insert('1.0', details)
            self._box.configure(state='disabled')
            self.details_button = ttk.Button(bar, text="Show details",
                                             command=self.toggle_details)
            self.details_button.pack(side='left', padx=(6, 0))
            auto_tip(self.details_button, "The technical description of the "
                     "error, for a bug report.")
        self.bind('<Escape>', lambda _e: self.destroy())
        self.bind('<Return>', lambda _e: self.destroy())
        self.close_button.focus_set()

    def _message_text(self, body, message):
        """A read-only Text that looks like a label: the words can be
        selected and copied (Ctrl+C), and links open on a click."""
        lines = message.split('\n') or ['']
        height = min(18, sum(max(1, -(-len(ln) // 60)) for ln in lines))
        style = ttk.Style(self)
        bg = style.lookup('TFrame', 'background') or self.cget('background')
        box = tk.Text(body, width=60, height=height, wrap='word',
                      relief='flat', borderwidth=0, highlightthickness=0,
                      background=bg, font='TkDefaultFont', cursor='xterm',
                      takefocus=True)
        box.insert('1.0', message)
        # A message taller than the box gets a scrollbar, from the first
        # time some of it is out of view. (It stays: hiding it again would
        # widen the box, rewrap the words and could flip back and forth.)
        self.message_scrollbar = ttk.Scrollbar(body, orient='vertical',
                                               command=box.yview)

        def scrolled(first, last):
            self.message_scrollbar.set(first, last)
            whole = float(first) <= 0.0 and float(last) >= 1.0
            # (Before the box is on screen its size, and so its view, is
            # not yet real.)
            if (not whole and box.winfo_ismapped()
                    and not self.message_scrollbar.winfo_manager()):
                self.message_scrollbar.grid(row=0, column=1, sticky='ns')
        box.configure(yscrollcommand=scrolled)
        from .theme import current_palette
        box.tag_configure('link', underline=True,
                          foreground=current_palette().get('link', 'blue'))
        for i, (a, b, kind, target) in enumerate(find_links(message)):
            tag = f'link{i}'
            box.tag_add('link', f'1.0+{a}c', f'1.0+{b}c')
            box.tag_add(tag, f'1.0+{a}c', f'1.0+{b}c')
            box.tag_bind(tag, '<Button-1>',
                         lambda _e, k=kind, t=target: open_link(k, t))
            box.tag_bind(tag, '<Enter>',
                         lambda _e: box.configure(cursor='hand2'))
            box.tag_bind(tag, '<Leave>',
                         lambda _e: box.configure(cursor='xterm'))
            self.links.append((kind, target))
        box.configure(state='disabled')
        # A disabled Text does not take focus on a click, so Ctrl+C would
        # go elsewhere; give it focus, and let Ctrl+A select everything.
        box.bind('<Button-1>', lambda _e: box.focus_set(), add='+')
        box.bind('<Control-a>', lambda _e: (box.tag_add('sel', '1.0', 'end-1c'),
                                            'break')[1])
        return box

    @property
    def details_shown(self):
        """True while the details box is visible."""
        return bool(self._box.winfo_manager())

    def toggle_details(self):
        """Show or hide the details box and resize the dialog to fit (button callback)."""
        if self.details_shown:
            self._box.pack_forget()
            if self.details_button is not None:
                self.details_button.configure(text="Show details")
        else:
            self._box.pack(side='top', fill='both', expand=True,
                           padx=16, pady=(0, 10))
            if self.details_button is not None:
                self.details_button.configure(text="Hide details")
        self.update_idletasks()
        self.geometry('')                      # fit the new content

    def copy_text(self):
        """What Copy puts on the clipboard: the message, then the details."""
        return '\n\n'.join(p for p in (self.message, self.details) if p)

    def copy(self):
        """Copy the message and details to the clipboard (button callback)."""
        try:
            clipboard_put(self, self.copy_text())
        except tk.TclError:
            pass


def _present(parent, title, message, details, kind='error'):
    """Show the dialog and wait until it is closed, as a message box would.
    (Tests replace this: see tests/conftest.py.)"""
    dlg = ErrorDialog(parent, title, message, details, kind)
    try:
        dlg.grab_set()
    except tk.TclError:
        pass
    dlg.wait_window()


def show_message(parent, title, message='', kind='error', details=''):
    """Show ``message`` in the shared window and wait until it is closed.
    ``kind`` is 'error', 'warning' or 'info' (the icon)."""
    _present(parent, title, message, details, kind)
    return 'ok'


def showerror(title=None, message=None, parent=None, **_options):
    """Drop-in for ``tkinter.messagebox.showerror``: the shared window."""
    return show_message(parent, title or "Error", message or '', 'error')


def showwarning(title=None, message=None, parent=None, **_options):
    """Drop-in for ``tkinter.messagebox.showwarning``: the shared window."""
    return show_message(parent, title or "Warning", message or '', 'warning')


def showinfo(title=None, message=None, parent=None, **_options):
    """Drop-in for ``tkinter.messagebox.showinfo``: the shared window."""
    return show_message(parent, title or "OpenFlo", message or '', 'info')


def error_message(message='', exc=None, path=None) -> str:
    """The text shown: what failed, why in plain words, and the file."""
    parts = [p for p in (message.strip() if message else '',
                         plain_error(exc) if exc is not None else '') if p]
    text = "\n\n".join(parts)
    if path:
        text += f"\n\nFile: {path}"
    return text


def show_error(parent, title, message='', exc=None, path=None):
    """Report a failure: ``message`` says what OpenFlo was doing ("The
    session could not be saved."), ``exc`` why (in plain words, with the
    technical description under Show details), ``path`` the file."""
    _present(parent, title, error_message(message, exc, path),
             error_details(exc), 'error')
