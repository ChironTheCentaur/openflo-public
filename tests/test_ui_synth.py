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


def test_event_counts_start_from_each_generators_own_default(monkeypatch):
    """Each event-count field shows its generator's default (the dialog
    showed 5,000 / 8,000 instead), so "Everything (full)" writes what
    openflo-synth writes."""
    import inspect

    from openflo import synthetic, ui_synth
    want = {'PBMC immunophenotyping': 20_000,
            'Differentiation time-course': 20_000,
            'Cell cycle': 20_000,
            'Spectral controls': 6000,
            'Calibration beads': 12_000,
            'Everything (full)': 20_000}
    got = {name: dict((k, d) for k, _l, d in spec['fields'])['n']
           for name, spec in ui_synth.DATASETS.items()}
    assert got == want
    assert want['Everything (full)'] == inspect.signature(
        synthetic.make_dataset).parameters['n'].default
    # openflo-synth's --events default is the same 20,000.
    seen = {}
    files = ('pbmc_files', 'pbmc_batch_files', 'fmo_files', 'cellcycle_files',
             'compensation_files', 'differentiation_files', 'spectral_files')
    monkeypatch.setattr(synthetic, 'make_dataset', lambda **kw: (
        seen.update(kw), {'out_dir': '', **dict.fromkeys(files, 0)})[1])
    synthetic.main([])
    assert seen['n'] == want['Everything (full)']


def test_dialog_fields_show_the_generator_defaults():
    root, ed = _make_editor()
    try:
        from openflo.ui_synth import SyntheticDialog
        d = SyntheticDialog(ed)
        for name, n in (('Spectral controls', 6000),
                        ('Calibration beads', 12_000),
                        ('Everything (full)', 20_000)):
            d._type_var.set(name)
            d._rebuild_fields()
            assert d._field_vars['n'].get() == str(n)
            assert d._field_vars['seed'].get() == '0'      # seeds unchanged
        d.destroy()
    finally:
        root.destroy()
