"""Headless smoke test for the voltage-titration dialog.

Construction must not require picking a folder/channel — the dialog opens
empty and the user selects later. We only assert it builds, then tear it down.
"""
import importlib
import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import tkinter as tk  # noqa: E402

import pytest  # noqa: E402


def test_voltage_dialog_constructs():
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:  # no display / tkinter unavailable
        pytest.skip(str(e))

    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str='', on_save=None, primary=False)
    ed.withdraw()

    try:
        from openflo.ui_voltage import VoltageDialog
        d = VoltageDialog(ed)
        # Opens empty: no folder/channel chosen, Run/Export disabled.
        assert d._paths == []
        assert str(d._run_btn['state']) == 'disabled'
        assert str(d._export_btn['state']) == 'disabled'
        d.destroy()
    finally:
        ed.destroy()
        root.destroy()
