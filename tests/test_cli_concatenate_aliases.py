"""openflo-run's concatenate mode lives in openflo.cli_concatenate; its old
private name on openflo.cli still resolves to the same object, with a
DeprecationWarning that names the new one."""
from __future__ import annotations

import pytest

import openflo.cli as cli
from openflo import cli_concatenate, cli_run


def test_old_cli_name_is_the_new_object_with_a_warning():
    with pytest.warns(DeprecationWarning,
                      match='use cli_concatenate.run_concatenated$'):
        got = cli._run_concatenated
    assert got is cli_concatenate.run_concatenated
    assert '_run_concatenated' not in vars(cli)


def test_cli_runs_the_moved_concatenate_mode(tmp_path, monkeypatch):
    # run() calls the object cli_run imported, so patching
    # cli_run.run_concatenated (not a copy left behind anywhere) is what a
    # --batch-mode concatenate run reaches.
    assert vars(cli_run)['run_concatenated'] is cli_concatenate.run_concatenated
    seen = []
    monkeypatch.setattr(cli_run, 'run_concatenated',
                        lambda trial_dirs, out_dir, groups, *a, **k:
                        seen.append([g['name'] for g in groups]) or True)
    monkeypatch.setattr(cli_run, 'run_independent',
                        lambda *a, **k: pytest.fail('independent mode ran'))
    monkeypatch.setattr(cli_run, 'gpu_present', lambda: False)
    trial = tmp_path / 'trial_1'
    trial.mkdir()
    (trial / 'sample_1.fcs').write_bytes(b'')
    cli.run([str(trial)], str(tmp_path / 'out'),
            [{'name': 'G1', 'fmo_set': '', 'samples': ['sample_1']}],
            fmo_sets={}, batch_mode='concatenate')
    assert seen == [['G1']]
