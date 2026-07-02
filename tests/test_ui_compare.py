"""Headless smoke test for the FlowJo-workspace comparison dialog.

Constructs the gate-editor window and the CompareWspDialog without any file,
then destroys them. Skips cleanly when no display / Tk is available.
"""
import importlib
import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

import tkinter as tk  # noqa: E402


def test_compare_dialog_constructs_and_destroys():
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:  # pragma: no cover — no display in CI
        pytest.skip(str(e))

    gui = importlib.import_module("openflo.gui")
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str="", on_save=None, primary=False)
    ed.withdraw()

    from openflo.ui_compare import CompareWspDialog
    d = CompareWspDialog(ed)
    # No file picked: the table is empty and export is disabled.
    assert d._tree.get_children() == ()
    assert str(d._export_btn["state"]) == "disabled"
    d.destroy()
    ed.destroy()
    root.destroy()
