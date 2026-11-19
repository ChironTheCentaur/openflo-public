"""A saved file says so in the status bar, with Open folder, instead of an
OK-only pop-up; Open folder uses each system's file browser.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import ast
import tkinter as tk

import tests._sample_editor as shared
from tests._editor_ui import SRC

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


_SAVED_WORDS = ('saved', 'exported', 'wrote', 'written', 'signed')


def _first_text(node):
    """The leading text of a call argument (a literal, or the literal start
    of an f-string or concatenation)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        for v in node.values:
            if isinstance(v, ast.Constant):
                return v.value
            return ''
    if isinstance(node, ast.BinOp):
        return _first_text(node.left)
    return ''


def test_no_info_box_says_a_file_was_saved():
    found = []
    for path in sorted(SRC.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'showinfo' and len(node.args) > 1):
                text = _first_text(node.args[1]).strip().lower()
                if text.startswith(_SAVED_WORDS):
                    found.append(f"{path.name}:{node.lineno}")
    assert not found, found


def test_the_check_finds_a_saved_box():
    code = 'messagebox.showinfo("Export", f"Saved:\\n{path}", parent=p)'
    call = ast.parse(code).body[0].value
    assert _first_text(call.args[1]).lower().startswith(_SAVED_WORDS)


def test_an_export_says_so_in_the_status_bar_with_open_folder(
        editor, tmp_path, monkeypatch):
    from tkinter import messagebox

    from matplotlib.figure import Figure

    from openflo import tool_window, ui_notice
    root, ed = editor
    infos = []
    monkeypatch.setattr(messagebox, 'showinfo',
                        lambda *a, **k: infos.append(a))
    out = tmp_path / 'synthetic_plot.png'
    monkeypatch.setattr(tool_window, 'ask_figure_path',
                        lambda *a, **k: str(out))
    win = tk.Toplevel(ed)
    win.status_var = tk.StringVar(master=win, value='')
    fig = Figure(figsize=(2, 2))
    fig.add_subplot().plot([0, 1], [0, 1])
    assert tool_window.export_figure(win, fig) == str(out)
    root.update()
    assert out.exists() and not infos
    assert ed.status_var.get() == "Saved synthetic_plot.png"
    assert win.status_var.get() == "Saved synthetic_plot.png"
    btn = ed._saved_folder_btn
    assert btn.winfo_manager() == 'pack'
    opened = []
    monkeypatch.setattr(ui_notice, 'open_folder',
                        lambda p: opened.append(p))
    btn.invoke()
    assert opened == [str(out)]
    ed.status_var.set("Something else happened.")
    root.update()
    assert btn.winfo_manager() == ''
    win.destroy()


def test_open_folder_command_per_system(tmp_path, monkeypatch):
    import sys

    from openflo.ui_notice import folder_command
    f = tmp_path / 'synthetic.csv'
    f.write_text('x\n')
    monkeypatch.setattr(sys, 'platform', 'win32')
    assert folder_command(f) == ['explorer', f'/select,{f}']
    monkeypatch.setattr(sys, 'platform', 'darwin')
    assert folder_command(f) == ['open', '-R', str(f)]
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert folder_command(f) == ['xdg-open', str(tmp_path)]
