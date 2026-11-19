"""Headless smoke test for the voltage-titration dialog.

Construction must not require picking a folder/channel — the dialog opens
empty and the user selects later. We assert it builds, then tear it down,
and that closing it gives the main status bar back.
"""
import importlib

from tests.conftest import editor_or_skip


def test_voltage_dialog_constructs():
    root, ed = editor_or_skip()

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
    root, ed = editor_or_skip()
    ui_style = importlib.import_module('openflo.ui_style')
    gui = importlib.import_module('openflo.gui')
    try:
        ui_style.apply_theme(ed, 'midnight')
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
        ui_style.apply_theme(ed, 'light')
        ed.destroy()
        root.destroy()


def test_chan_lookup_initialised_before_any_folder_pick():
    """``_chan_lookup`` is only assigned inside ``_populate_channels``, which
    returns EARLY when the first FCS is unreadable. ``_selected_channel()``
    reads it, so it must exist from construction — otherwise the dialog is one
    reordering (or one unreadable file) away from an AttributeError."""
    root, ed = editor_or_skip()
    try:
        from openflo.ui_voltage import VoltageDialog
        d = VoltageDialog(ed)
        d.withdraw()
        # Exists immediately, before any folder has been chosen.
        assert d._chan_lookup == {}
        # And the reader is safe both with an empty and a non-empty selection.
        assert d._selected_channel() is None
        d._channel_var.set('FL5-A  (CD3)')
        assert d._selected_channel() == 'FL5-A  (CD3)'   # falls back to `shown`
        d.destroy()
    finally:
        ed.destroy()
        root.destroy()


def test_closing_gives_the_main_status_bar_back():
    """The window's opening instruction stayed in the main status bar after
    it closed, by Close or by the close box."""
    root, ed = editor_or_skip()
    try:
        from openflo.ui_voltage import VoltageDialog
        ed.status_var.set("Loaded 3 samples.")
        d = VoltageDialog(ed)
        assert ed.status_var.get().startswith('Pick a folder')
        d.destroy()                                   # its Close button
        assert ed.status_var.get() == "Loaded 3 samples."
        # The close box (Tk destroys the window without Python's destroy).
        d = VoltageDialog(ed)
        d.update()
        d.tk.call('destroy', str(d))
        assert ed.status_var.get() == "Loaded 3 samples."
        # Something written there since is left alone.
        d = VoltageDialog(ed)
        ed.status_var.set("Gate added.")
        d.destroy()
        assert ed.status_var.get() == "Gate added."
    finally:
        root.destroy()
