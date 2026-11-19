"""Help > Report a problem…: the form that sends a report to the developers
(openflo.ui_feedback). A stand-in sender records what would be posted; no
network. Generated text and made-up file names only.
"""
from __future__ import annotations

import pytest

import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep
from tests._tk_wait import pump_until

virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

TOKENISED = ("----- KeyError -----\nTraceback: <path:1> line 3\n"
             "KeyError: 'FL2-A'\n")
SECRET = "real-value-behind-a-token"


@pytest.fixture
def editor(virtual_screen, no_dialogs, engines_known, monkeypatch):
    # The send runs later on the Tk thread instead of a worker thread: a
    # worker cannot post back to Tk while no mainloop runs (as in tests).
    import tests.test_windows_nothing_hidden as sw
    from openflo import ui_feedback

    def later(widget, work, on_done=None, on_error=None, on_finally=None):
        from tests._inline_editor import _inline
        widget.after(1, lambda: _inline(widget, work, on_done, on_error, on_finally))
    monkeypatch.setattr(ui_feedback, 'run_async', later)
    virtual_screen(1920, 1080)
    root = sw._root(1.0)
    try:
        yield root, sw._editor(root)
    finally:
        root.destroy()


class Sender:
    def __init__(self, answer=None):
        self.answer = answer
        self.sent = []

    def __call__(self, fields):
        self.sent.append(dict(fields))
        return self.answer


def _report(tmp_path, text=TOKENISED):
    p = tmp_path / 'error_report.log'
    p.write_text(text, encoding='utf-8')
    # The key file beside it (tokens -> real values) must never be sent.
    (tmp_path / 'error_report.keys.json').write_text(
        '{"<path:1>": "' + SECRET + '"}', encoding='utf-8')
    return str(p)


def _wait(root, until, seconds=5.0):
    return pump_until(root, until, timeout=seconds)


def _fill(dlg, title='Plot freezes', details='It stops after I zoom in twice.', contact=''):
    dlg.title_var.set(title)
    dlg.details.insert('1.0', details)
    dlg.contact_var.set(contact)


def _dialog(ed, path, sender):
    from openflo.ui_feedback import FeedbackDialog
    return FeedbackDialog(ed, path, '3.1.0', sender=sender)


def test_help_report_a_problem_opens_the_form(editor, monkeypatch, tmp_path):
    from openflo import editor_lifecycle
    from openflo.result_stamp import app_version
    from openflo.ui_feedback import FeedbackDialog
    root, ed = editor
    path = _report(tmp_path)
    monkeypatch.setattr(editor_lifecycle, '_error_report_path', lambda: path)
    ed._report_a_problem()
    root.update()
    dlg = ed._feedback_dialog
    assert isinstance(dlg, FeedbackDialog)
    assert dlg._report_path == path
    assert dlg._version == app_version()[:40]
    dlg.destroy()


def test_about_report_a_problem_opens_the_form(editor, monkeypatch):
    from openflo.editor_help import AboutWindow
    root, ed = editor
    opened = []
    monkeypatch.setattr(type(ed), '_report_a_problem', lambda self: opened.append(self))
    monkeypatch.setattr('webbrowser.open', lambda url: opened.append(url))
    win = AboutWindow(ed)
    root.update()
    buttons = {str(w.cget('text')): w for w in sweep._walk(win) if w.winfo_class() == 'TButton'}
    buttons['Report a problem'].invoke()
    assert opened == [ed]


def test_the_report_is_sent_only_when_ticked(editor, tmp_path):
    root, ed = editor
    path = _report(tmp_path)
    sender = Sender()
    dlg = _dialog(ed, path, sender)
    assert dlg.attach.get() is False
    assert 'disabled' not in dlg.attach_box.state()
    _fill(dlg, contact='someone@example.com')
    dlg.send_report()
    assert _wait(root, lambda: sender.sent and not dlg.winfo_exists())
    (f,) = sender.sent
    assert f == {'kind': 'bug', 'app': 'desktop', 'version': '3.1.0', 'title': 'Plot freezes',
                 'details': 'It stops after I zoom in twice.', 'contact': 'someone@example.com'}

    sender = Sender()
    dlg = _dialog(ed, path, sender)
    _fill(dlg)
    dlg.kind.set('feature')
    dlg.attach.set(True)
    dlg.send_report()
    assert _wait(root, lambda: sender.sent and not dlg.winfo_exists())
    (f,) = sender.sent
    assert f['kind'] == 'feature'
    assert f['details'].startswith('It stops after I zoom in twice.')
    assert "KeyError: 'FL2-A'" in f['details'] and '<path:1>' in f['details']
    assert all(SECRET not in v for v in f.values())


def test_show_what_is_sent_shows_the_exact_details(editor, tmp_path):
    _root, ed = editor
    dlg = _dialog(ed, _report(tmp_path), Sender())
    _fill(dlg)
    dlg.attach.set(True)
    shown = dlg.show_sent().get('1.0', 'end-1c')
    assert dlg.fields()['details'] in shown
    assert '<path:1>' in shown
    assert SECRET not in shown
    dlg.attach.set(False)
    assert '<path:1>' not in dlg.show_sent().get('1.0', 'end-1c')
    dlg.destroy()


def test_no_report_recorded_cannot_be_ticked(editor, tmp_path):
    _root, ed = editor
    dlg = _dialog(ed, str(tmp_path / 'none.log'), Sender())
    assert 'disabled' in dlg.attach_box.state()
    assert 'none recorded' in str(dlg.attach_box.cget('text'))
    dlg.destroy()


def test_a_bad_form_is_explained_and_nothing_sent(editor, tmp_path):
    from openflo import feedback
    root, ed = editor
    sender = Sender()
    dlg = _dialog(ed, _report(tmp_path), sender)
    _fill(dlg, title='ab')
    dlg.send_report()
    root.update()
    assert sender.sent == []
    assert str(dlg.message.cget('text')) == feedback.ERRORS['title']
    dlg.title_var.set('Plot freezes')
    dlg.contact_var.set('not an address')
    dlg.send_report()
    root.update()
    assert sender.sent == []
    assert str(dlg.message.cget('text')) == feedback.ERRORS['contact']
    dlg.destroy()


def test_no_room_for_the_report_says_so(editor, tmp_path):
    root, ed = editor
    sender = Sender()
    dlg = _dialog(ed, _report(tmp_path), sender)
    _fill(dlg, details='w' * 4990)
    dlg.attach.set(True)
    dlg.send_report()
    root.update()
    assert sender.sent == []
    assert 'no room' in str(dlg.message.cget('text'))
    dlg.destroy()


def test_a_failed_send_keeps_the_text_and_points_to_the_website(editor, tmp_path):
    from openflo import feedback
    root, ed = editor
    sender = Sender('offline')
    dlg = _dialog(ed, _report(tmp_path), sender)
    _fill(dlg)
    dlg.send_report()
    assert 'disabled' in dlg.send_button.state()
    assert _wait(root, lambda: not dlg.sending)
    assert dlg.winfo_exists() and not dlg.sent
    assert 'disabled' not in dlg.send_button.state()
    msg = str(dlg.message.cget('text'))
    assert msg.startswith(feedback.ERRORS['offline']) and feedback.FORM_URL in msg
    assert dlg.title_var.get() == 'Plot freezes'
    assert dlg.details.get('1.0', 'end-1c') == 'It stops after I zoom in twice.'
    dlg.destroy()


def test_a_send_that_raises_is_a_failure_not_a_crash(editor, tmp_path):
    from openflo import feedback
    root, ed = editor

    def boom(_fields):
        raise RuntimeError('planted')
    dlg = _dialog(ed, _report(tmp_path), boom)
    _fill(dlg)
    dlg.send_report()
    assert _wait(root, lambda: not dlg.sending)
    assert str(dlg.message.cget('text')).startswith(feedback.ERRORS['unavailable'])
    dlg.destroy()



def test_the_report_is_tokenised_again_before_it_is_shown_or_sent(editor, monkeypatch, tmp_path):
    # A report written by an older tokeniser can still hold a folder name
    # with a space; Help > Report a problem tokenises it again.
    from openflo import editor_lifecycle
    _root, ed = editor
    old = tmp_path / 'error_report.log'
    old.write_text("----- OSError -----\nOSError: 'D:\\\\Lab Data\\\\run 4\\\\tubeQ.fcs'\n", encoding='utf-8')
    monkeypatch.setattr(editor_lifecycle, '_error_report_path', lambda: str(old))
    ed._report_a_problem()
    dlg = ed._feedback_dialog
    _fill(dlg)
    dlg.attach.set(True)
    shown = dlg.show_sent().get('1.0', 'end-1c')
    sent = dlg.fields()['details']
    for text in (shown, sent):
        assert 'Lab Data' not in text and 'tubeQ' not in text and '<path:' in text
    dlg.destroy()


def test_only_the_end_of_a_long_report_is_read(tmp_path):
    from openflo import ui_feedback
    p = tmp_path / 'error_report.log'
    # 75-byte lines: 64 KB from the end falls inside a line.
    lines = [f'entry {i:06d} ' + 'z' * 61 for i in range(3000)]
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    text = ui_feedback._read_report(str(p))
    assert len(text.encode('utf-8')) <= ui_feedback.TAIL_BYTES
    assert text.endswith(lines[-1] + '\n')
    assert text.split('\n')[0] in lines          # starts at a whole line
    assert 'entry 000000' not in text
    small = tmp_path / 'small.log'
    small.write_text('one\ntwo\n', encoding='utf-8')
    assert ui_feedback._read_report(str(small)) == 'one\ntwo\n'
