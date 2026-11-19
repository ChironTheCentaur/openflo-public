"""Every window opened from the menus behaves the same way: it belongs to
the editor (``transient``), closes on Escape, and has its main action,
then Cancel or Close, at the right; Return runs the main action
(tool_window.standardise_dialog). Work windows keep their own place, and
dropdowns show the methods' names while keeping the settings keys.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import tests._editor_ui as ui
import tests._sample_editor as shared
from tests._editor_ui import _open_windows

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known

root = ui.root


def test_every_dialog_is_attached_closes_on_escape_and_orders_its_buttons(
        editor):
    from openflo.tool_window import DISMISS_LABELS, button_rows
    root, ed = editor
    opened, failures = [], []
    for label, win in _open_windows(root, ed):
        opened.append(label)
        name = f"{label} [{win.title()}]"
        if getattr(win, 'independent', False):
            if str(win.wm_transient()):
                failures.append(f"{name}: a work window made transient")
            continue
        if str(win.wm_transient()) != str(ed):
            failures.append(f"{name}: not attached to the editor "
                            f"({win.wm_transient()!r})")
        for _frame, buttons in button_rows(win):
            texts = [str(b.cget('text')) for b in buttons]
            if texts[-1] not in DISMISS_LABELS:
                failures.append(f"{name}: {texts} does not end with "
                                "Cancel or Close")
            if 'Cancel' in texts and len(texts) > 1:
                main = buttons[-2]
                if str(main.cget('default')) != 'active':
                    failures.append(f"{name}: {texts[-2]!r} beside Cancel "
                                    "is not the default (Return) action")
        win.focus_force()
        win.event_generate('<Escape>')
        root.update()
        if win.winfo_exists():
            failures.append(f"{name}: Escape did not close it")
    assert len(opened) >= 25, opened
    assert not failures, "\n".join(failures)


def _dialog(root, owner_cls=tk.Toplevel):
    owner = owner_cls(root)
    win = tk.Toplevel(owner)
    ran = []
    bar = ttk.Frame(win)
    bar.pack(side='bottom')
    ttk.Button(bar, text="Cancel", command=win.destroy).pack(side='right')
    ok = ttk.Button(bar, text="Apply", default='active',
                    command=lambda: ran.append('apply'))
    ok.pack(side='right')
    entry = ttk.Entry(win)
    entry.pack()
    text = tk.Text(win, height=2)
    text.pack()
    return owner, win, ran, ok, entry, text


def test_standardise_attaches_and_binds_the_keys(root):
    from openflo.tool_window import standardise_dialog
    owner, win, ran, ok, entry, text = _dialog(root)
    standardise_dialog(win)
    root.update()
    assert str(win.wm_transient()) == str(owner)
    entry.focus_force()
    root.update()
    entry.event_generate('<Return>')
    root.update()
    assert ran == ['apply']
    text.focus_force()
    root.update()
    text.event_generate('<Return>')          # a new line, not the action
    root.update()
    assert ran == ['apply']
    ok.state(['disabled'])
    entry.focus_force()
    entry.event_generate('<Return>')
    root.update()
    assert ran == ['apply']                  # a disabled action doesn't run
    entry.event_generate('<Escape>')
    root.update()
    assert not win.winfo_exists()


def test_named_choice_shows_names_and_keeps_keys(root):
    from openflo.tool_window import CLUSTER_METHOD_NAMES, named_choice
    var = tk.StringVar(master=root, value='phenograph')
    combo = named_choice(root, var, CLUSTER_METHOD_NAMES)
    assert list(combo.cget('values')) == ['PhenoGraph', 'Leiden', 'FlowSOM']
    assert combo.get() == 'PhenoGraph'
    combo.set('FlowSOM')                     # picked from the list
    assert var.get() == 'flowsom'
    var.set('leiden')                        # set by code (a loaded recipe)
    assert combo.get() == 'Leiden'
    var.set('custom')                        # unknown key shows as itself
    assert combo.get() == 'custom' and var.get() == 'custom'


def test_escape_asks_a_window_that_asks_before_closing(root):
    from openflo.tool_window import standardise_dialog
    _owner, win, *_ = _dialog(root)
    asked = []
    win.protocol('WM_DELETE_WINDOW', lambda: asked.append(1))
    standardise_dialog(win)
    win.focus_force()
    root.update()
    win.event_generate('<Escape>')
    root.update()
    assert asked == [1] and win.winfo_exists()


def test_work_windows_and_root_windows_are_left_alone(root):
    from openflo.tool_window import standardise_dialog
    work = tk.Toplevel(root)
    work.independent = True                  # type: ignore[attr-defined]
    child = tk.Toplevel(work)
    child.independent = True                 # type: ignore[attr-defined]
    standardise_dialog(child)
    assert not str(child.wm_transient())
    assert not child.bind('<Escape>')
    other_editor = tk.Toplevel(root)          # made on the hidden root
    standardise_dialog(other_editor)
    assert not str(other_editor.wm_transient())


def test_a_window_keeps_its_own_escape(root):
    from openflo.tool_window import standardise_dialog
    _owner, win, *_ = _dialog(root)
    own = []
    win.bind('<Escape>', lambda _e: own.append(1))
    standardise_dialog(win)
    win.focus_force()
    root.update()
    win.event_generate('<Escape>')
    root.update()
    assert own == [1] and win.winfo_exists()
