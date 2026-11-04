"""Keyboard shortcuts: every shortcut listed in Help > Keyboard shortcuts
and shown in a menu is really bound; the shortcuts window lists them in
two columns; macOS shows Cmd and binds Command as well as Control.

Every check runs on generated data with synthetic sample names and generic
channels (FL1-A ...), and is proved to fail on the old behaviour by a
mutation check (see docs/test-assurance/).
"""
from __future__ import annotations

import pytest

import tests._sample_editor as shared
from openflo.ui_logic import accel
from tests._editor_ui import _all_entries, _sequence

editor = shared.editor
virtual_screen = shared.virtual_screen
no_dialogs = shared.no_dialogs
engines_known = shared.engines_known


def test_every_listed_shortcut_is_bound(editor):
    from openflo.editor_help import SHORTCUTS
    _root, ed = editor
    bound = set(ed.bind()) | set(ed.bind_all()) | set(ed.gate_tv.bind())
    listed = [keys for _s, items in SHORTCUTS for keys, _a in items]
    assert len(listed) >= 25
    missing = [k for k in listed if _sequence(k) not in bound]
    assert not missing, (missing, sorted(bound))


@pytest.mark.parametrize('platform, save', [
    (None, None), ('linux', 'Ctrl+S'), ('win32', 'Ctrl+S'),
    ('darwin', 'Cmd+S')])
def test_the_shortcut_window_lists_them_in_two_columns(editor, monkeypatch,
                                                       platform, save):
    import sys

    import openflo.editor_help as eh
    from openflo.editor_help import SHORTCUTS, ShortcutsWindow
    from openflo.ui_logic import accel
    root, ed = editor
    if platform is None:
        # This machine's own label: Cmd on a Mac, Ctrl elsewhere.
        save = 'Cmd+S' if sys.platform == 'darwin' else 'Ctrl+S'
    else:
        monkeypatch.setattr(eh, 'accel',
                            lambda keys: accel(keys, platform=platform))
    win = ShortcutsWindow(ed)
    root.update()
    rows = [win.tree.item(i, 'values') for i in win.tree.get_children()]
    assert all(len(r) == 2 for r in rows)
    assert len(rows) == sum(len(items) + 1 for _s, items in SHORTCUTS)
    assert (f'   {save}', 'Save session') in [tuple(r) for r in rows]
    assert str(win.wm_transient()) == str(ed)
    win.focus_force()
    root.update()
    win.event_generate('<Escape>')
    root.update()
    assert not win.winfo_exists()


def test_menus_show_their_shortcuts_and_each_is_bound(editor):
    _root, ed = editor
    shown = {label: acc for _t, label, acc in _all_entries(ed) if acc}
    for label, acc in (("Undo", "Ctrl+Z"), ("Redo", "Ctrl+Y"),
                       ("Find sample or gate", "Ctrl+F"),
                       ("Zoom in", "Ctrl++"), ("Zoom out", "Ctrl+-"),
                       ("Redraw plot", "F5"), ("User guide", "F1"),
                       ("Delete gate", "Delete")):
        assert shown.get(label) == accel(acc), (label, shown.get(label))
    bound = set(ed.bind()) | set(ed.bind_all()) | set(ed.gate_tv.bind())
    # A Mac shows Cmd, and binds Control as well as Command to the same keys.
    unbound = [(label, acc) for label, acc in shown.items()
               if acc != 'Delete'
               and _sequence(acc.replace('Cmd+', 'Ctrl+')) not in bound]
    assert not unbound, unbound


def test_macos_shows_cmd_and_binds_command(monkeypatch):
    import sys

    from openflo.editor_lifecycle import LifecycleMixin
    from openflo.ui_logic import accel
    assert accel("Ctrl+Shift+S", 'darwin') == "Cmd+Shift+S"
    assert accel("Ctrl+Z", 'win32') == "Ctrl+Z"
    assert accel("F5", 'darwin') == "F5"

    class Stub:
        def __init__(self):
            self.seqs = []

        def bind(self, seq, _fn):
            self.seqs.append(seq)

        def __getattr__(self, _name):
            return lambda *a, **k: None

    monkeypatch.setattr(sys, 'platform', 'darwin')
    stub = Stub()
    LifecycleMixin._bind_shortcuts(stub)
    assert '<Command-q>' in stub.seqs and '<Control-q>' in stub.seqs
    assert '<Command-o>' in stub.seqs and '<F5>' in stub.seqs
    monkeypatch.setattr(sys, 'platform', 'linux')
    stub = Stub()
    LifecycleMixin._bind_shortcuts(stub)
    assert not [s for s in stub.seqs if 'Command' in s]
