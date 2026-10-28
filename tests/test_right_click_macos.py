"""Context menus open on a Mac as well as on X11 and Windows.

Under Tk 8.6 on macOS (aqua) the right mouse button is ``<Button-2>`` and
Ctrl+click is the one-button alternative; elsewhere the right button is
``<Button-3>`` and ``<Button-2>`` is the middle one. A context menu bound
only to ``<Button-3>`` never opens from a Mac's right button, so every
right-click binding in the package goes through
``tool_window.bind_right_click``.

- A source scan fails on any direct ``<Button-3>`` binding.
- With the aqua branch forced, ``<Button-2>`` and Ctrl+click on the real
  tab strip open its menu; off aqua ``<Button-2>`` stays the middle button.
- The layout editor's middle-drag pan keeps the middle button on both.

Synthetic sample names and generic channels only.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from tests.conftest import gui_unavailable  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / 'src' / 'openflo'
COLS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
_BUTTON3 = re.compile(r'<(?:[\w-]+-)?(?:Button|ButtonPress|ButtonRelease)-3>'
                      r'|<(?:[\w-]+-)?3>')
_BINDERS = {'bind', 'bind_all', 'bind_class', 'tag_bind'}


def _direct_button3_binds():
    """``file:line`` of every bind / bind_all / bind_class / tag_bind call
    in the package whose event sequence is a literal naming button 3."""
    found = []
    for path in sorted(SRC.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _BINDERS):
                continue
            for arg in node.args[:2]:
                parts = ([arg] if isinstance(arg, ast.Constant)
                         else arg.values if isinstance(arg, ast.JoinedStr)
                         else [])
                text = ''.join(p.value for p in parts
                               if isinstance(p, ast.Constant)
                               and isinstance(p.value, str))
                if _BUTTON3.search(text):
                    found.append(f'{path.name}:{node.lineno}')
    return found


def test_no_right_click_is_bound_to_button_3_directly():
    assert _direct_button3_binds() == [], (
        "bind right-click menus with tool_window.bind_right_click, which "
        "adds the macOS right button and Ctrl+click")


def test_the_scan_sees_a_direct_button_3_binding(tmp_path, monkeypatch):
    (tmp_path / 'sample_widget.py').write_text(
        "def build(w, cb):\n"
        "    w.bind('<Button-3>', cb)\n"
        "    w.tag_bind('row', '<ButtonRelease-3>', cb)\n"
        "    w.bind('<Button-1>', cb)\n", encoding='utf-8')
    monkeypatch.setattr(sys.modules[__name__], 'SRC', tmp_path)
    assert _direct_button3_binds() == ['sample_widget.py:2',
                                       'sample_widget.py:3']


# ── the real widgets ─────────────────────────────────────────────────────

def _editor_or_skip():
    try:
        import tkinter as tk
    except ImportError:
        gui_unavailable('tkinter not available')
    try:
        root = tk.Tk()
    except Exception as e:                            # noqa: BLE001
        gui_unavailable(f'Tk cannot initialise without a display: {e}')
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def _load(ed, name, seed):
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(1, 1000, (300, len(COLS))), columns=COLS)
    ed._samples[name] = FlowSample.from_dataframe(df, name=name,
                                                  path=f'{name}.fcs')
    ed._sample_order.append(name)
    ed._sample_colors[name] = '#1f77b4'
    ed._sample_trial[name] = 'T'
    if 'T' not in ed._trial_order:
        ed._trial_order.append('T')
    ed._sample_plot_enabled[name] = True
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    ed._populate_channel_combos()
    ed._set_active_sample(name)


def _tab_xy(ed, idx):
    """A point inside tab ``idx``, relative to the strip."""
    nb = ed._tab_nb
    for y in range(0, nb.winfo_height(), 2):
        for x in range(0, nb.winfo_width(), 2):
            try:
                if int(nb.index(f'@{x},{y}')) == idx:
                    return x, y
            except Exception:                         # noqa: BLE001
                continue
    pytest.fail(f'tab {idx} not found on the strip')


@pytest.fixture
def popups(monkeypatch):
    import tkinter as tk
    opened = []
    monkeypatch.setattr(tk.Menu, 'tk_popup',
                        lambda self, *a, **k: opened.append(self))
    monkeypatch.setattr(tk.Menu, 'post',
                        lambda self, *a, **k: opened.append(self))
    return opened


def _labels(menu):
    end = menu.index('end')
    return [menu.entrycget(i, 'label') for i in range(end + 1)
            if menu.type(i) != 'separator']


def _tab_strip_clicks(sequences, opened):
    """Open an editor with a second tab, send each event to that tab on
    the strip, and return the labels of the menus (recorded in ``opened``)
    that each event opened."""
    root, ed = _editor_or_skip()
    try:
        _load(ed, 's1', 1)
        ed._refresh_gate_list()
        b = ed._tabs().new_tab('B', members={'s1': None})
        ed._rebuild_tab_strip()
        ed.deiconify()
        ed.update()
        x, y = _tab_xy(ed, b)
        out = []
        for seq in sequences:
            before = len(opened)
            ed._tab_nb.event_generate(seq, x=x, y=y)
            ed.update()
            out.append([_labels(m) for m in opened[before:]])
        return out
    finally:
        root.destroy()


def test_the_mac_right_button_and_ctrl_click_open_the_tab_menu(
        monkeypatch, popups):
    from openflo import tool_window
    monkeypatch.setattr(tool_window, 'is_aqua', lambda _w: True)
    right, ctrl, other = _tab_strip_clicks(
        ['<Button-2>', '<Control-Button-1>', '<Button-3>'], popups)
    for menus in (right, ctrl, other):
        assert len(menus) == 1
        assert menus[0][:3] == ['Rename…', 'Duplicate', 'Delete']


def test_off_the_mac_the_middle_button_opens_no_tab_menu(
        monkeypatch, popups):
    from openflo import tool_window
    monkeypatch.setattr(tool_window, 'is_aqua', lambda _w: False)
    middle, right = _tab_strip_clicks(['<Button-2>', '<Button-3>'], popups)
    assert middle == []
    assert len(right) == 1 and right[0][0] == 'Rename…'


@pytest.mark.parametrize('aqua, menu_buttons, pan_button', [
    (False, ('<Button-3>',), 2),
    (True, ('<Button-2>', '<Control-Button-1>'), 3),
])
def test_layout_canvas_keeps_middle_drag_pan_and_the_menu(
        monkeypatch, aqua, menu_buttons, pan_button):
    """On X11 / Windows the middle button (2) pans a plot and button 3
    opens the item menu; on aqua with Tk 8.6 the right button (2) and
    Ctrl+click open the menu and the middle button (3) pans."""
    import tkinter as tk

    from openflo import tool_window
    from openflo.ui_layout_editor import LayoutEditorWindow
    if aqua and tk.TkVersion >= 8.7:
        pytest.skip('Tk 8.7+ numbers aqua buttons as X11 does')
    monkeypatch.setattr(tool_window, 'is_aqua', lambda _w: aqua)
    calls = []
    monkeypatch.setattr(LayoutEditorWindow, '_on_context',
                        lambda self, e: calls.append(('menu', e.num)))
    monkeypatch.setattr(LayoutEditorWindow, '_on_pan_start',
                        lambda self, e: calls.append(('pan', e.num)))
    monkeypatch.setattr(LayoutEditorWindow, '_on_pan_end',
                        lambda self, e: None)
    root, ed = _editor_or_skip()
    try:
        win = LayoutEditorWindow(ed)
        root.update()
        c = win.canvas
        for seq in menu_buttons:
            calls.clear()
            c.event_generate(seq, x=20, y=20)
            assert calls and calls[0][0] == 'menu', seq
        calls.clear()
        c.event_generate(f'<ButtonPress-{pan_button}>', x=20, y=20)
        c.event_generate(f'<ButtonRelease-{pan_button}>', x=20, y=20)
        assert calls == [('pan', pan_button)]
        win.destroy()
    finally:
        root.destroy()
