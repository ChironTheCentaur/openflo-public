"""The layout editor names its real buttons in the empty-page hint, keeps
its status line whole inside the window, and has undo, redo and Ctrl+S.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tests._editor_ui as ui
import tests._sample_editor as shared
import tests.test_windows_nothing_hidden as sweep

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

root = ui.root


def _layout_editor(root, ed):
    from openflo.ui_layout_editor import LayoutEditorWindow
    win = LayoutEditorWindow(ed)
    root.update()
    win._redraw()
    return win


def test_layout_editor_hint_names_its_real_buttons(editor):
    root, ed = editor
    win = _layout_editor(root, ed)
    texts = [win.canvas.itemcget(i, 'text')
             for i in win.canvas.find_all()
             if win.canvas.type(i) == 'text']
    hint = next(t for t in texts if t.startswith("Empty page."))
    buttons = {w.cget('text') for w in sweep._walk(win)
               if w.winfo_class() == 'TButton'}
    named = [b for b in ("＋ Plot", "＋ Stats", "＋ Chart") if b in hint]
    assert named == ["＋ Plot", "＋ Stats", "＋ Chart"]
    assert set(named) <= buttons
    assert "Add plot" not in hint and "…." not in hint
    win.destroy()


def test_layout_editor_status_line_wraps_inside_the_window(editor):
    root, ed = editor
    win = _layout_editor(root, ed)
    win.geometry('900x700')
    win.status.set("Status: " + "a long message " * 30)
    root.update()
    lbl = win.status_label
    assert lbl.winfo_ismapped()
    assert 0 < int(lbl.cget('wraplength')) <= win.winfo_width()
    assert lbl.winfo_y() + lbl.winfo_height() <= win.winfo_height()
    assert lbl.winfo_reqwidth() <= win.winfo_width()
    win.destroy()


def test_layout_editor_undo_redo_and_save(editor, monkeypatch):
    from openflo.layout_page import text_item
    root, ed = editor
    ed._note_saved()
    win = _layout_editor(root, ed)
    assert not ed.title().startswith('*')
    win.layout.add(text_item('synthetic note'))
    win._redraw()
    assert len(win.layout.items) == 1
    assert ed.title().startswith('*')                # session now unsaved
    win.focus_force()
    root.update()
    win.event_generate('<Control-Key-z>')
    root.update()
    assert len(win.layout.items) == 0
    assert ed._layout_doc is win.doc
    win.event_generate('<Control-Key-y>')
    root.update()
    assert len(win.layout.items) == 1
    saved = []
    monkeypatch.setattr(type(ed), '_save_session',
                        lambda self: saved.append(1))
    win.event_generate('<Control-Key-s>')
    root.update()
    assert saved == [1]
    win.destroy()
