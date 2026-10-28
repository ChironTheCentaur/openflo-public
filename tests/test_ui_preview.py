"""Headless smoke test for the Quick preview dialog.

Constructs the editor + the dialog WITHOUT requiring an FCS file, then tears
both down. Skips cleanly on machines without a usable Tk display.
"""

import importlib
import os

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

import tkinter as tk  # noqa: E402


def test_quick_preview_dialog_constructs_and_destroys():
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:  # no display available
        pytest.skip(str(e))

    try:
        gui = importlib.import_module('openflo.gui')
        gui.messagebox.askyesno = lambda *a, **k: True
        ed = gui.ViewGateEditorWindow(
            root, fcs_dir=None, labels_str='', on_save=None, primary=False)
        ed.withdraw()

        from openflo.ui_preview import QuickPreviewDialog
        d = QuickPreviewDialog(ed)
        # Threshold parsing helper is pure — exercise it without a file.
        assert QuickPreviewDialog._parse_thr('') is None
        assert QuickPreviewDialog._parse_thr('abc') is None
        assert QuickPreviewDialog._parse_thr('1.5') == 1.5
        d.destroy()
    finally:
        root.destroy()
