"""openflo-run's programmatic entry point lives in openflo.cli_run, its
batch correction in openflo.cli_cytonorm and its example panel in
openflo.cli_defaults. openflo.cli.run and the example names are still those
objects on cli; the old private names resolve to the same objects, with a
DeprecationWarning that names the new home."""
from __future__ import annotations

import pytest

import openflo.cli as cli
from openflo import cli_cytonorm, cli_defaults, cli_run

OLD_TO_NEW = {
    '_fit_cytonorm': (cli_cytonorm, 'fit_cytonorm'),
    '_gpu_present': (cli_run, 'gpu_present'),
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    mod, new = OLD_TO_NEW[old]
    short = mod.__name__.rsplit('.', 1)[1]
    with pytest.warns(DeprecationWarning, match=f'use {short}.{new}$'):
        got = getattr(cli, old)
    assert got is getattr(mod, new)
    assert old not in vars(cli)


def test_run_stays_the_public_cli_entry_point():
    assert vars(cli)['run'] is cli_run.run


@pytest.mark.parametrize('name', ['CD901b', 'CD902', 'CD903', 'FMOS_SET_A',
                                  'FMOS_SET_B', 'DEFAULT_FMO_SETS',
                                  'DEFAULT_GROUPS'])
def test_example_panel_names_stay_on_cli(name):
    assert vars(cli)[name] is getattr(cli_defaults, name)


def test_run_falls_back_to_the_example_fmo_sets(tmp_path, monkeypatch):
    # fmo_sets=None means the example sets, now read from cli_defaults.
    seen = []
    monkeypatch.setattr(cli_run, 'run_independent',
                        lambda trial_dirs, out_dir, groups, fmo_sets, *a, **k:
                        seen.append(fmo_sets) or True)
    monkeypatch.setattr(cli_run, 'gpu_present', lambda: False)
    trial = tmp_path / 'trial_1'
    trial.mkdir()
    (trial / 'sample_1.fcs').write_bytes(b'')
    assert cli.run([str(trial)], str(tmp_path / 'out'),
                   [{'name': 'G1', 'fmo_set': '', 'samples': ['sample_1']}]
                   ) == 0
    assert seen == [cli_defaults.DEFAULT_FMO_SETS]


def test_batch_correct_calls_the_moved_fit(tmp_path, monkeypatch):
    # run() fits CytoNorm through cli_run's import of fit_cytonorm and hands
    # the model to the mode.
    calls, models = [], []

    def fake_fit(groups, default_trial, **kw):
        calls.append((default_trial, kw))
        return {'model': 1}, ['FL1-A', 'FL2-A']
    monkeypatch.setattr(cli_run, 'fit_cytonorm', fake_fit)
    monkeypatch.setattr(cli_run, 'run_independent',
                        lambda *a, cytonorm=None, **k:
                        models.append(cytonorm) or True)
    monkeypatch.setattr(cli_run, 'gpu_present', lambda: False)
    trial = tmp_path / 'trial_1'
    trial.mkdir()
    (trial / 'sample_1.fcs').write_bytes(b'')
    cli.run([str(trial)], str(tmp_path / 'out'),
            [{'name': 'G1', 'fmo_set': '', 'samples': ['sample_1']}],
            fmo_sets={}, batch_correct=True, cytonorm_mode='controls',
            cytonorm_metaclusters=7, cytonorm_control='ref')
    assert calls == [(str(trial), {'mode': 'controls', 'n_metaclusters': 7,
                                   'control_token': 'ref', 'qc': False})]
    assert models == [{'model': 1}]
