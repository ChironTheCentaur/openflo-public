"""Headless smoke test for the synthetic-dataset dialog (``ui_synth``)."""
import importlib
import os
import tempfile

import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')

import tkinter as tk  # noqa: E402


def _make_editor():
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:  # no display available
        pytest.skip(str(e))
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(
        root, fcs_dir=None, labels_str='', on_save=None, primary=False)
    ed.withdraw()
    return root, ed


def test_dialog_constructs_and_destroys():
    root, ed = _make_editor()
    try:
        from openflo.ui_synth import SyntheticDialog
        d = SyntheticDialog(ed)
        # Switching dataset type should rebuild fields without error.
        for name in ('Cell cycle', 'Spectral controls', 'Everything (full)'):
            d._type_var.set(name)
            d._rebuild_fields()
        d.destroy()
    finally:
        root.destroy()


def test_generator_returns_paths_into_tmpdir():
    """Prove the wiring target works: one generator writes .fcs and returns
    their paths."""
    from openflo import synthetic
    with tempfile.TemporaryDirectory() as tmp:
        paths = synthetic.make_cell_cycle_dataset(tmp, samples=2, n=400, seed=1)
        assert paths, 'generator returned no paths'
        for p in paths:
            assert p.lower().endswith('.fcs')
            assert os.path.isfile(p)
