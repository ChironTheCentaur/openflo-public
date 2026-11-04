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
    assert str(dlg.message_label.cget('text')) == msg
    assert not dlg.details_shown
    dlg.details_button.invoke()
    root.update()
    assert dlg.details_shown
    assert str(dlg.details_button.cget('text')) == "Hide details"
    assert "KeyError: 'FL9-A'" in dlg._box.get('1.0', 'end')
    dlg.copy_button.invoke()
    assert root.clipboard_get() == details
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
                        lambda parent, title, message, details:
                        seen.append((title, message, details)))
    exc = _raised(PermissionError(13, 'Denied', 'synthetic.csv'))
    ui_errors.show_error(None, "Export failed",
                         "The table could not be exported.", exc,
                         'synthetic.csv')
    (title, message, details), = seen
    assert title == "Export failed"
    assert 'PermissionError' not in message
    assert 'open in another program' in message
    assert 'PermissionError' in details


def _error_calls():
    """(file:line, source) of every message box call in the package."""
    out = []
    for f in sorted(SRC.glob('*.py')):
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
    assert 'did not expect' in str(dialogs[0].message_label.cget('text'))
    assert 'ZeroDivisionError' in dialogs[0].details
    assert ed._show_log_var.get() == shown_log    # the console stays shut
    assert 'Something went wrong' in ed.status_var.get()
