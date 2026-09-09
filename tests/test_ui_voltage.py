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


def test_voltage_initial_plot_is_dark_under_midnight():
    """The empty plot shown on open (before any run) must follow the Midnight
    theme — regression for it rendering white until an analysis was run."""
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                          # noqa: BLE001
        pytest.skip(str(e))
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str='', on_save=None, primary=False)
    ed.withdraw()
    try:
        gui.apply_theme(ed, 'midnight')
        ed._theme_var.set('midnight')
        from openflo.ui_voltage import VoltageDialog
        d = VoltageDialog(ed)                        # no run — just the build
        d.withdraw()
        # figure recoloured to the midnight plot bg, and the Tk canvas widget
        # bg matches (no white margin)
        r, g, b, _a = d._fig.get_facecolor()
        assert (r + g + b) / 3 < 0.3                 # clearly dark
        assert str(d._canvas.get_tk_widget().cget('bg')) == \
            gui.THEMES['midnight']['plot_bg']
        d.destroy()
    finally:
        gui.apply_theme(ed, 'light')
        ed.destroy()
        root.destroy()


def test_chan_lookup_initialised_before_any_folder_pick():
    """``_chan_lookup`` is only assigned inside ``_populate_channels``, which
    returns EARLY when the first FCS is unreadable. ``_selected_channel()``
    reads it, so it must exist from construction — otherwise the dialog is one
    reordering (or one unreadable file) away from an AttributeError."""
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                          # noqa: BLE001
        pytest.skip(str(e))
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str='', on_save=None, primary=False)
    ed.withdraw()
    try:
        from openflo.ui_voltage import VoltageDialog
        d = VoltageDialog(ed)
        d.withdraw()
        # Exists immediately, before any folder has been chosen.
        assert d._chan_lookup == {}
        # And the reader is safe both with an empty and a non-empty selection.
        assert d._selected_channel() is None
        d._channel_var.set('PE-A  (CD3)')
        assert d._selected_channel() == 'PE-A  (CD3)'   # falls back to `shown`
        d.destroy()
    finally:
        ed.destroy()
        root.destroy()
