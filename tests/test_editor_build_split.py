"""The editor window is built one section at a time.

``ViewGateEditorWindow._build`` used to make the whole window in one method
of about 820 lines. It is now a short list of calls, one builder method per
panel (menubar strip and main split, sample I/O bar, samples-and-gates tree,
side buttons, log console, right column, the plot control rows, plot area,
histogram sliders, load progress, status bar, window hooks, tree bindings),
in the order the widgets were always made: Tk names widgets by creation
order, so that order is part of the window's behaviour.

These tests pin the split (each builder exists and ``_build`` stays short)
and that the built window still has its key widgets.
"""
from __future__ import annotations

import inspect

import pytest

from tests.conftest import gui_unavailable

BUILDERS = (
    '_build_main_split', '_build_left_panel', '_build_sample_io_bar',
    '_build_gate_tree', '_build_side_buttons', '_build_log_console',
    '_build_right_column', '_build_plot_controls', '_build_axes_row',
    '_build_mode_row', '_build_gate_tool_row', '_build_display_row',
    '_build_plot_area', '_build_histogram_sliders', '_build_load_progress',
    '_build_status_bar', '_build_window_hooks', '_build_tree_bindings',
)


def _editor_class():
    try:
        from openflo.gui import ViewGateEditorWindow
    except ImportError as exc:
        gui_unavailable(f"the GUI cannot be imported: {exc}")
    return ViewGateEditorWindow


def test_build_is_a_short_list_of_section_builders():
    cls = _editor_class()
    missing = [n for n in BUILDERS if not callable(cls.__dict__.get(n))]
    assert missing == []
    src = inspect.getsource(cls._build)
    assert len(src.splitlines()) < 60
    # Every builder is called from _build, directly or (control rows) from
    # _build_plot_controls.
    called = src + inspect.getsource(cls._build_plot_controls)
    assert [n for n in BUILDERS if f'self.{n}(' not in called] == []


def test_no_builder_name_is_defined_twice_in_the_class_tree():
    """A builder must not shadow (or be shadowed by) a mixin method."""
    cls = _editor_class()
    for name in BUILDERS:
        owners = [k.__name__ for k in cls.__mro__ if name in vars(k)]
        assert owners == ['ViewGateEditorWindow'], (name, owners)


@pytest.fixture
def editor(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    cls = _editor_class()
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        gui_unavailable(f"Tk cannot start: {exc}")
    root.withdraw()
    ed = cls(root, fcs_dir=None, labels_str='', on_save=None, primary=False,
             defer_plot_canvas=True)
    ed.withdraw()
    ed._finish_plot_canvas()
    yield ed
    root.destroy()


def test_the_built_window_has_its_key_widgets(editor):
    from tkinter import ttk
    ed = editor
    # The samples-and-gates tree, with its heading and tags.
    assert isinstance(ed.gate_tv, ttk.Treeview)
    assert ed.gate_tv.heading('#0', 'text') == '▾ Samples & Gates'
    assert 'sample' in ed.gate_tv.tk.splitlist(
        ed.gate_tv.tk.call(str(ed.gate_tv), 'tag', 'names'))
    # The plot control combos, all inside the control bar.
    for combo in (ed.x_combo, ed.y_combo, ed.color_combo, ed.hist_y_combo,
                  ed._mp_combo):
        assert isinstance(combo, ttk.Combobox)
        assert str(combo).startswith(str(ed._ctrl_bar))
    assert ed.mode_var.get() == 'pseudocolor'
    assert ed.gate_tool_var.get() == 'quadrant'
    assert set(ed._display_radios) == {'all', 'highlight', 'filter'}
    # The log + console pane, in the left panel.
    assert ed._log_text.winfo_class() == 'Text'
    assert str(ed._console_entry).startswith(str(ed._log_frame))
    assert str(ed._log_frame).startswith(str(ed._left_host))
    # The status bar, on the window's bottom row, showing status_var.
    bars = ed.grid_slaves(row=2, column=0)
    assert len(bars) == 1
    labels = [w for w in bars[0].winfo_children()[0].winfo_children()
              if w.winfo_class() == 'TLabel']
    assert labels and str(labels[0].cget('textvariable')) == str(ed.status_var)
    assert ed.status_var.get() == "Add one or more FCS files to begin."
    # The plot sits in the right column, beside the workspace host.
    assert ed.canvas.get_tk_widget().winfo_manager() == 'grid'
    assert str(ed._plot_host).startswith(str(ed._editor_paned))
    # The tree keeps the keys bound after it is built.
    for seq in ('<Control-c>', '<Delete>', '<Button-1>', '<<TreeviewSelect>>'):
        assert ed.gate_tv.bind(seq)
