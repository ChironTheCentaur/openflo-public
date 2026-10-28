"""Copy gates to samples: the dialog's Copy button is on screen, and copying
again doesn't stack duplicate gates.

The dialog was fixed at 420 px wide with four themed buttons in one row
(about 560 px), so pack gave the last one packed -- Copy -- no room: the
user saw Select all, Deselect all and Cancel and had to widen the window to
find Copy.

Copying appended every gate. A target that already held the gate (copied
once before, or drawn on it with "all shown") got a second, identical-looking
row. Now the same gate is reused; the same population drawn differently
(same kind, axes and name under the same parent) takes the copied geometry
in place; anything less certain is still appended.
"""
from tkinter import ttk

import pytest

from tests.test_boolean_gate_tools import (
    _editor_or_skip,
    _frame,
    _load,
    _toplevel,
    _walk,
)


@pytest.fixture
def editor():
    root, ed = _editor_or_skip()
    yield ed
    root.destroy()


def _rect(x0, x1, parent=None, **extra):
    return dict({'kind': 'rect', 'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
                 'x0': x0, 'x1': x1, 'y0': 2e4, 'y1': 8e4,
                 'parent_id': parent}, **extra)


def _two_samples(ed):
    _load(ed, 's1', _frame(1, 500))
    _load(ed, 's2', _frame(2, 500))
    ed._set_active_sample('s1')


def test_copy_button_fits_in_the_dialog(editor):
    import openflo.gui as gui
    gui.apply_theme(editor.master, 'light')      # the app's padded buttons
    _two_samples(editor)
    editor._add_gate(_rect(2e4, 9e4))
    editor.deiconify()
    editor.update()
    editor._open_copy_gates_dialog()
    dlg = _toplevel(editor, 'Copy gates')
    dlg.update()
    buttons = {w.cget('text'): w for w in _walk(dlg)
               if isinstance(w, ttk.Button)}
    assert set(buttons) == {'Select all', 'Deselect all', 'Copy', 'Cancel'}
    for text, b in buttons.items():
        right = b.winfo_rootx() + b.winfo_width()
        assert b.winfo_ismapped() and b.winfo_width() > 10, text
        assert right <= dlg.winfo_rootx() + dlg.winfo_width(), text
    dlg.destroy()


def test_copying_twice_adds_nothing_the_second_time(editor):
    _two_samples(editor)
    root = editor._add_gate(_rect(2e4, 9e4))
    editor._add_gate({'kind': 'threshold', 'channel': 'CD3', 'value': 0.0,
                      'parent_id': root})
    editor._copy_gates_to(['s2'])
    once = {k: dict(v) for k, v in editor._sample_gates['s2'].items()}
    editor._copy_gates_to(['s2'])
    assert editor._sample_gates['s2'].keys() == once.keys()
    assert ': 0 added · 2 already there' in editor.status_var.get()


def test_the_same_population_redrawn_is_updated_in_place(editor):
    _two_samples(editor)
    root = editor._add_gate(_rect(2e4, 9e4))
    editor._add_gate({'kind': 'threshold', 'channel': 'CD3', 'value': 0.0,
                      'parent_id': root})
    editor._copy_gates_to(['s2'])
    s2 = editor._sample_gates['s2']
    (s2_root,) = [k for k, g in s2.items() if g['kind'] == 'rect']
    (s2_child,) = [k for k, g in s2.items() if g['kind'] == 'threshold']
    s2[s2_root]['enabled'] = False                   # the target's own state
    editor._gates[root]['x0'] = 1.5e4                # edit the source gate
    editor._copy_gates_to(['s2'])
    assert sorted(s2) == sorted([s2_root, s2_child])
    assert s2[s2_root]['x0'] == pytest.approx(1.5e4)
    assert s2[s2_root]['enabled'] is False
    assert s2[s2_child]['parent_id'] == s2_root
    assert 'updated in place' in editor.status_var.get()


def test_a_different_gate_on_the_same_axes_is_still_appended(editor):
    _two_samples(editor)
    editor._add_gate(_rect(2e4, 9e4))
    editor._copy_gates_to(['s2'])
    editor._set_active_sample('s2')
    editor._add_gate(_rect(1e4, 3e4, name='Small'))
    editor._set_active_sample('s1')
    editor._add_gate(_rect(5e4, 9e4, name='Large'))
    editor._copy_gates_to(['s2'])
    names = sorted(g.get('name') or '' for g in
                   editor._sample_gates['s2'].values())
    assert names == ['', 'Large', 'Small']


def test_two_unnamed_lookalikes_are_not_guessed_between(editor):
    """With two unnamed gates on the same axes in the target, which one the
    copy 'is' can't be told: append rather than overwrite either."""
    _two_samples(editor)
    editor._set_active_sample('s2')
    editor._add_gate(_rect(1e4, 3e4))
    editor._add_gate(_rect(5e4, 9e4))
    editor._set_active_sample('s1')
    editor._add_gate(_rect(2e4, 7e4))
    editor._copy_gates_to(['s2'])
    x0s = sorted(g['x0'] for g in editor._sample_gates['s2'].values())
    assert x0s == pytest.approx([1e4, 2e4, 5e4])
