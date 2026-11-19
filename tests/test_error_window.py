"""Errors say what failed and why in plain words, with the exception and
its traceback under Show details (ui_errors); no message box shows a raw
exception, and an uncaught error opens that window instead of the
developer console.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import ast
import re

import tests._editor_ui as ui
import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from tests._editor_ui import SRC

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

root = ui.root


def _raised(exc):
    try:
        raise exc
    except BaseException as e:               # noqa: BLE001 - returns whatever was raised, with its traceback, for the error window to show
        return e


def test_plain_error_names_no_exception_types():
    import errno

    from openflo.ui_errors import plain_error
    cases = {
        FileNotFoundError(2, 'No such file', 'synthetic_1.fcs'):
            'could not be found (synthetic_1.fcs)',
        PermissionError(13, 'Denied', 'out.csv'): 'open in another program',
        MemoryError(): 'not enough memory',
        OSError(errno.ENOSPC, 'No space left on device'): 'disk is full',
        KeyError('FL9-A'): 'was not found: FL9-A',
        ValueError('the percentile must be between 0 and 100'):
            'The percentile must be between 0 and 100.',
        RuntimeError(''): 'did not expect',
        UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad'): 'not be the kind',
    }
    for exc, want in cases.items():
        text = plain_error(exc)
        assert want in text, (exc, text)
        assert type(exc).__name__ not in text, text
        assert 'Errno' not in text, text


def test_error_dialog_hides_details_until_asked(root):
    from openflo.ui_errors import ErrorDialog, error_details, error_message
    exc = _raised(KeyError('FL9-A'))
    msg = error_message("The statistics could not be exported.", exc,
                        'synthetic.csv')
    assert msg.startswith("The statistics could not be exported.")
    assert 'File: synthetic.csv' in msg and 'KeyError' not in msg
    details = error_details(exc)
    assert details.startswith('Traceback') and "KeyError: 'FL9-A'" in details
    dlg = ErrorDialog(root, "Export failed", msg, details)
    root.update()
    assert dlg.message_box.get('1.0', 'end-1c') == msg
    assert not dlg.details_shown
    dlg.details_button.invoke()
    root.update()
    assert dlg.details_shown
    assert str(dlg.details_button.cget('text')) == "Hide details"
    assert "KeyError: 'FL9-A'" in dlg._box.get('1.0', 'end')
    dlg.copy_button.invoke()
    assert root.clipboard_get() == msg + '\n\n' + details
    dlg.details_button.invoke()
    root.update()
    assert not dlg.details_shown
    dlg.focus_force()
    root.update()
    dlg.event_generate('<Escape>')
    root.update()
    assert not dlg.winfo_exists()


def test_show_error_reports_through_the_dialog(monkeypatch):
    from openflo import ui_errors
    seen = []
    monkeypatch.setattr(ui_errors, '_present',
                        lambda parent, title, message, details, kind:
                        seen.append((title, message, details, kind)))
    exc = _raised(PermissionError(13, 'Denied', 'synthetic.csv'))
    ui_errors.show_error(None, "Export failed",
                         "The table could not be exported.", exc,
                         'synthetic.csv')
    (title, message, details, kind), = seen
    assert title == "Export failed" and kind == 'error'
    assert 'PermissionError' not in message
    assert 'open in another program' in message
    assert 'PermissionError' in details


def _error_calls():
    """(file:line, source) of every message box call in the package."""
    out = []
    for f in sorted(SRC.rglob('*.py')):
        src = f.read_text(encoding='utf-8')
        for n in ast.walk(ast.parse(src)):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ('showerror', 'showwarning',
                                        'showinfo')):
                out.append((f"{f.name}:{n.lineno}",
                            ast.get_source_segment(src, n) or ''))
    return out


def raw_exception_boxes(calls):
    return [where for where, seg in calls
            if re.search(r'type\(\w+\)\.__name__', seg)]


def test_no_message_box_shows_a_raw_exception():
    calls = _error_calls()
    assert len(calls) > 50
    assert not raw_exception_boxes(calls)
    # and the check finds the old form
    assert raw_exception_boxes([('x.py:1', 'messagebox.showerror("Export '
                                 'failed", f"{type(exc).__name__}: {exc}")')])


def test_an_uncaught_error_opens_the_error_window_once(editor):
    root, ed = editor
    from openflo.ui_errors import ErrorDialog
    shown_log = ed._show_log_var.get()
    for _ in range(2):
        exc = _raised(ZeroDivisionError('division by zero'))
        ed._report_callback_exception(type(exc), exc, exc.__traceback__)
    root.update()
    dialogs = [w for w in sweep._walk(root) if isinstance(w, ErrorDialog)]
    assert len(dialogs) == 1
    assert 'did not expect' in dialogs[0].message
    assert 'ZeroDivisionError' in dialogs[0].details
    assert ed._show_log_var.get() == shown_log    # the console stays shut
    assert 'Something went wrong' in ed.status_var.get()


# ── One window for every error, warning and notice (Sky, 11-18) ──────────────

def test_the_message_text_can_be_selected_and_copied(root):
    from openflo.ui_errors import ErrorDialog
    msg = "The table could not be exported.\n\nFile: synthetic.csv"
    dlg = ErrorDialog(root, "Export failed", msg, '', 'warning')
    root.update()
    box = dlg.message_box
    assert box.winfo_class() == 'Text'                   # not a Label
    assert str(box.cget('state')) == 'disabled'          # read-only
    assert dlg.details_button is None
    box.tag_add('sel', '1.4', '1.9')
    assert box.get('sel.first', 'sel.last') == 'table'
    box.focus_force()
    root.update()
    box.event_generate('<Control-a>')
    root.update()
    assert box.get('sel.first', 'sel.last') == msg
    dlg.copy_button.invoke()
    assert root.clipboard_get() == msg                   # no details: message only
    assert str(dlg.copy_button.cget('text')) == 'Copy'
    dlg.destroy()


def test_each_kind_shows_its_own_icon(root):
    from openflo.ui_errors import ErrorDialog
    for kind, icon in (('error', 'error'), ('warning', 'warning'),
                       ('info', 'information')):
        dlg = ErrorDialog(root, "T", "x", '', kind)
        assert [str(i) for i in dlg.icon_label.cget('image')] == [
            f'::tk::icons::{icon}']
        dlg.destroy()


def test_links_and_paths_are_found(tmp_path):
    from openflo.ui_errors import find_links
    folder = tmp_path / 'run 1'
    folder.mkdir()
    f = folder / 'synthetic_1.fcs'
    f.write_text('x')
    missing = folder / 'not here.wsp'
    text = (f"See https://www.getopenflo.com/feedback.\nFile: {f}\n"
            f"Workspace not found:\n{missing}\nratio 1/2, and (/x).\n"
            f"Saved {f} (3 rows).\nhttps://www.getopenflo.com/?from={tmp_path}")
    got = [(k, t) for _a, _b, k, t in find_links(text)]
    assert got == [('url', 'https://www.getopenflo.com/feedback'),
                   ('path', str(f)), ('path', str(missing)), ('path', str(f)),
                   ('url', f'https://www.getopenflo.com/?from={tmp_path}')]
    for a, b, _k, t in find_links(text):
        assert text[a:b] == t


def test_a_link_opens_on_a_click(root, tmp_path, monkeypatch):
    from openflo import ui_errors, ui_notice
    opened = []
    monkeypatch.setattr('webbrowser.open', lambda url: opened.append(('web', url)))
    monkeypatch.setattr(ui_notice, 'open_folder',
                        lambda p: opened.append(('folder', p)))
    f = tmp_path / 'synthetic.csv'
    f.write_text('x')
    msg = f"Could not write:\n{f}\nHelp: https://www.getopenflo.com/feedback"
    dlg = ui_errors.ErrorDialog(root, "Export failed", msg)
    dlg.deiconify()
    dlg.lift()
    root.update()
    assert dlg.links == [('path', str(f)),
                         ('url', 'https://www.getopenflo.com/feedback')]
    box = dlg.message_box
    for i, (_kind, target) in enumerate(dlg.links):
        ranges = box.tag_ranges(f'link{i}')
        assert box.get(ranges[0], ranges[1]) == target
        assert f'link{i}' in box.tag_names(ranges[0])
        assert 'link' in box.tag_names(ranges[0])
        x, y, *_ = box.bbox(ranges[0])
        box.event_generate('<Motion>', x=x + 2, y=y + 2)   # sets 'current'
        root.update()
        assert str(box.cget('cursor')) == 'hand2'          # a link under the mouse
        box.event_generate('<Button-1>', x=x + 2, y=y + 2)
        root.update()
    assert opened == [('folder', str(f)),
                      ('web', 'https://www.getopenflo.com/feedback')]
    assert str(box.tag_cget('link', 'underline')) in ('1', 'true')
    from openflo.theme import current_palette
    assert str(box.tag_cget('link', 'foreground')) == current_palette()['link']
    dlg.destroy()


def test_the_messagebox_stand_ins_use_the_shared_window(monkeypatch):
    from openflo import ui_errors
    seen = []
    monkeypatch.setattr(ui_errors, '_present',
                        lambda *a: seen.append(a))
    assert ui_errors.showerror("A", "a", parent=None) == 'ok'
    ui_errors.showwarning("B", "b", parent=None, icon='warning')
    ui_errors.showinfo("C", "c")
    assert seen == [(None, "A", "a", '', 'error'),
                    (None, "B", "b", '', 'warning'),
                    (None, "C", "c", '', 'info')]


def test_no_window_uses_a_tk_message_box_to_report():
    """Every showerror/showwarning/showinfo in the app goes through
    ui_errors (the shared window), not tkinter.messagebox."""
    bad = [where for where, seg in _error_calls()
           if seg.startswith('messagebox.')]
    assert bad == []
    assert len(_error_calls()) > 50



_SHOW = ('showerror', 'showwarning', 'showinfo')


def _messagebox_show_refs(source, name='x.py'):
    """file:line of every reference to a message box show function in
    *source* that does not go through ``ui_errors``: any ``<x>.showerror``
    (an alias such as ``mb``, ``tkinter.messagebox``, ``gui.messagebox``, or a
    conditional pick of the function) and any ``from tkinter.messagebox
    import showerror``. Out of scope: ``getattr(messagebox, 'show' + kind)``
    and ``tk.call('tk_messageBox', ...)``, which no module uses."""
    out = []
    for n in ast.walk(ast.parse(source)):
        if (isinstance(n, ast.Attribute) and n.attr in _SHOW
                and not (isinstance(n.value, ast.Name)
                         and n.value.id == 'ui_errors')):
            out.append(f"{name}:{n.lineno}")
        elif (isinstance(n, ast.ImportFrom)
              and (n.module or '').endswith('messagebox')
              and any(a.name in _SHOW or a.name == '*' for a in n.names)):
            out.append(f"{name}:{n.lineno}")
    return out


def test_no_module_picks_a_tk_message_box_function():
    """No module outside ui_errors, in any subpackage, refers to tkinter's
    message box show functions, in any of the forms below."""
    files = sorted(SRC.rglob('*.py'))
    assert any(f.parent != SRC for f in files)        # subpackages are read
    bad = [ref for f in files if f != SRC / 'ui_errors.py'
           for ref in _messagebox_show_refs(f.read_text(encoding='utf-8'),
                                            str(f.relative_to(SRC)))]
    assert bad == []
    # and the check finds each form
    for form in (
            "(messagebox.showinfo if ok else messagebox.showwarning)('T')",
            "from tkinter import messagebox as mb\nmb.showerror('T', 'm')",
            "from tkinter.messagebox import showerror\nshowerror('T', 'm')",
            "import tkinter.messagebox\ntkinter.messagebox.showerror('T')",
            "gui.messagebox.showwarning('T', 'm')"):
        assert _messagebox_show_refs(form), form
    assert _messagebox_show_refs("ui_errors.showerror('T', 'm')") == []



def test_a_click_on_the_message_gives_it_focus(root):
    """Older Tk gives a disabled Text no focus on a click (outside Windows),
    so mouse-select then Ctrl+C would copy nothing: the box's own click
    binding focuses it. Tk's class binding is left out here, so the test
    reads that binding on any Tk."""
    from openflo.ui_errors import ErrorDialog
    dlg = ErrorDialog(root, "Export failed", "The table could not be exported.")
    dlg.deiconify()
    dlg.lift()
    dlg.close_button.focus_force()
    root.update()
    assert dlg.focus_get() == dlg.close_button
    box = dlg.message_box
    box.bindtags((str(box), str(dlg), 'all'))            # no 'Text' class
    box.event_generate('<Button-1>', x=4, y=4)
    root.update()
    assert dlg.focus_get() == box
    dlg.destroy()


def test_without_tk_message_box_icons_a_bitmap_shows(root, monkeypatch):
    """Where Tk's message-box images are missing, the icon is Tk's built-in
    bitmap of the same kind ('info' for information)."""
    from openflo import ui_errors

    class NoImages(ui_errors.ttk.Label):
        def __init__(self, *a, **k):
            if 'image' in k:
                raise ui_errors.tk.TclError('image doesn\'t exist')
            super().__init__(*a, **k)
    monkeypatch.setattr(ui_errors.ttk, 'Label', NoImages)
    for kind, bitmap in (('error', 'error'), ('warning', 'warning'),
                         ('info', 'info')):
        dlg = ui_errors.ErrorDialog(root, "T", "x", '', kind)
        assert str(dlg.icon_label.cget('bitmap')) == bitmap
        dlg.destroy()


def test_a_path_glued_to_a_word_is_not_a_link(tmp_path):
    """A path start needs a non-word character before it, so 'x/home/...'
    (or 'and/or', '1/2') is never linked, even when the path exists."""
    from openflo.ui_errors import find_links
    assert tmp_path.exists()
    assert find_links(f"x{tmp_path}") == []
    assert [k for _a, _b, k, _t in find_links(f"x {tmp_path}")] == ['path']


def test_plain_http_addresses_are_links_too():
    from openflo.ui_errors import find_links
    assert [(k, t) for _a, _b, k, t in find_links("see http://example.org/a.")] \
        == [('url', 'http://example.org/a')]


def test_the_stand_ins_give_a_title_when_none_is_passed(monkeypatch):
    from openflo import ui_errors
    seen = []
    monkeypatch.setattr(ui_errors, '_present', lambda *a: seen.append(a[1]))
    ui_errors.showerror(message='m')
    ui_errors.showwarning(message='m')
    ui_errors.showinfo(message='m')
    assert seen == ["Error", "Warning", "OpenFlo"]


def test_a_long_message_scrolls_to_its_last_line(root):
    """A message taller than the box (the 18-line cap) gets a scrollbar, and
    its last line can be brought into view; a short one gets none."""
    from openflo.ui_errors import ErrorDialog
    lines = [f"synthetic_{i}.fcs: events missing after row {i}"
             for i in range(1, 41)] + ["Gates are kept; run the rest again."]
    dlg = ErrorDialog(root, "Saved data incomplete", '\n'.join(lines), '',
                      'warning')
    dlg.deiconify()
    root.update()
    box = dlg.message_box
    assert box.dlineinfo('end-1c linestart') is None      # out of view at first
    assert dlg.message_scrollbar.winfo_manager() == 'grid'
    assert dlg.message_scrollbar.winfo_ismapped()
    box.yview_moveto(1.0)
    root.update()
    assert box.dlineinfo('end-1c linestart') is not None
    assert box.get('end-1c linestart', 'end-1c') == lines[-1]
    assert dlg.resizable() == (True, True)
    before = box.winfo_height()
    dlg.geometry(f"{dlg.winfo_width()}x{dlg.winfo_height() + 200}")
    root.update()
    assert box.winfo_height() > before                    # a taller window shows more
    dlg.destroy()
    short = ErrorDialog(root, "T", "One line.", '', 'info')
    short.deiconify()
    root.update()
    assert short.message_scrollbar.winfo_manager() == ''
    short.destroy()
