"""Tests for the shared tool-window scaffolding (openflo.tool_window)."""
import os

import pytest


def _root_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip("tkinter not available")
    try:
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        pytest.skip(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    return root


def test_tool_window_chrome():
    from openflo.tool_window import ToolWindow
    root = _root_or_skip()
    try:
        w = ToolWindow(root, "T", geometry="320x200")
        assert w.title() == "T"
        assert w.bind('<Escape>')                   # escape-to-close bound
        w.destroy()
    finally:
        root.destroy()


def test_figure_panel_and_background_selector():
    from openflo.tool_window import background_selector, figure_panel
    root = _root_or_skip()
    try:
        _frame, fig, _canvas = figure_panel(root, figsize=(4, 3), dpi=80,
                                            dark=False)
        assert tuple(fig.get_size_inches()) == (4.0, 3.0)
        var, combo = background_selector(root, dark=False)
        assert var.get() == 'White'
        assert tuple(combo['values']) == (
            'White', 'Dark', 'Transparent', 'Translucent')
    finally:
        root.destroy()


def test_export_figure_writes(tmp_path):
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from matplotlib.figure import Figure

    from openflo.tool_window import export_figure
    root = _root_or_skip()
    saved = (fd.asksaveasfilename, mb.showinfo)
    try:
        out = tmp_path / 'f.png'
        fd.asksaveasfilename = lambda *a, **k: str(out)
        mb.showinfo = lambda *a, **k: None
        fig = Figure()
        fig.add_subplot(111).plot([0, 1], [0, 1])
        p = export_figure(root, fig, background='White')
        assert p == str(out) and out.is_file()
    finally:
        fd.asksaveasfilename, mb.showinfo = saved
        root.destroy()


def test_figure_window_constructs_and_exports(tmp_path):
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb

    from matplotlib.figure import Figure

    from openflo.ui_figure_window import _FigureWindow
    root = _root_or_skip()
    saved = (fd.asksaveasfilename, mb.showinfo)
    try:
        out = tmp_path / 'fw.png'
        fd.asksaveasfilename = lambda *a, **k: str(out)
        mb.showinfo = lambda *a, **k: None
        fig = Figure()
        fig.add_subplot(111).plot([0, 1], [1, 0])
        w = _FigureWindow(root, fig, "FW")
        w._export()
        assert out.is_file()
        w.destroy()
    finally:
        fd.asksaveasfilename, mb.showinfo = saved
        root.destroy()
