"""Headless smoke test for the synthetic-dataset dialog (``ui_synth``)."""
import os
import tempfile

from tests.conftest import editor_or_skip


def test_dialog_constructs_and_destroys():
    root, ed = editor_or_skip()
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
    got = {name: {k: d for k, _l, d in spec['fields']}['n']
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
    root, ed = editor_or_skip()
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


def test_dialog_checks_each_field_by_its_label_and_range():
    """FILE-5/6/7: messages said "'n' must be an integer.", 10^9 events
    passed (and would exhaust memory), and a negative seed passed the form
    only to fail in the generator with no field named. Each field is now
    checked against its range and named by its label."""
    import pytest
    root, ed = editor_or_skip()
    try:
        from openflo.ui_synth import SyntheticDialog
        d = SyntheticDialog(ed)
        d._type_var.set('PBMC immunophenotyping')
        d._rebuild_fields()
        f = d._field_vars
        assert d._collect_params() == {'n': 20_000, 'seed': 0, 'donors': 3}
        cases = [
            ('n', 'abc', "Events per sample must be a whole number from 1 "
                         "to 5000000 (it says “abc”)."),
            ('n', '1000000000', "Events per sample must be a whole number "
                                "from 1 to 5000000 (it says “1000000000”)."),
            ('seed', '-1', "Random seed must be a whole number from 0 to "
                           "4294967295 (it says “-1”)."),
            ('donors', '2.5', "Donors per group must be a whole number from "
                              "1 to 50 (it says “2.5”)."),
        ]
        for key, text, msg in cases:
            good = f[key].get()
            f[key].set(text)
            with pytest.raises(ValueError) as exc:
                d._collect_params()
            assert str(exc.value) == msg
            f[key].set(good)
        f['n'].set('5,000,000')
        f['seed'].set('4294967295')
        assert d._collect_params() == {'n': 5_000_000, 'seed': 4294967295,
                                       'donors': 3}
        d.destroy()
    finally:
        root.destroy()
