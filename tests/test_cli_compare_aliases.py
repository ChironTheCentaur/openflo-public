"""openflo-run's per-trial setup, group comparisons and independent mode
live in openflo.cli_trials, openflo.cli_compare and openflo.cli_independent;
their old private names on openflo.cli still resolve to the same objects,
with a DeprecationWarning that names the new one."""
from __future__ import annotations

import pytest

import openflo.cli as cli
from openflo import (
    cli_compare,
    cli_concatenate,
    cli_independent,
    cli_parallel,
    cli_plots,
    cli_run,
    cli_trials,
)

OLD_TO_NEW = {
    '_trial_label': (cli_trials, 'trial_label'),
    '_calibrate_bead_micron': (cli_trials, 'calibrate_bead_micron'),
    '_already_done': (cli_trials, 'already_done'),
    '_build_fmo_thresholds_for_groups': (cli_trials,
                                         'build_fmo_thresholds_for_groups'),
    '_SHARED_IDS_NOTE': (cli_compare, 'SHARED_IDS_NOTE'),
    '_COMPARE_OWN': (cli_compare, 'COMPARE_OWN'),
    '_compare_files': (cli_compare, 'compare_files'),
    '_compare_group_pair': (cli_compare, 'compare_group_pair'),
    '_add_pvalues': (cli_compare, 'add_pvalues'),
    '_shared_n_jobs': (cli_compare, 'shared_n_jobs'),
    '_STAGING_MARK': (cli_compare, 'STAGING_MARK'),
    '_is_staging': (cli_compare, 'is_staging'),
    '_clear_stale_staging': (cli_compare, 'clear_stale_staging'),
    '_remove_earlier_set': (cli_compare, 'remove_earlier_set'),
    '_publish_compare_set': (cli_compare, 'publish_compare_set'),
    '_not_in_this_run': (cli_compare, 'not_in_this_run'),
    '_samples_to_compare': (cli_compare, 'samples_to_compare'),
    '_group_plots_job': (cli_compare, 'group_plots_job'),
    '_draw_group_plots': (cli_compare, 'draw_group_plots'),
    '_write_group_outputs': (cli_compare, 'write_group_outputs'),
    '_run_independent': (cli_independent, 'run_independent'),
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    module, new = OLD_TO_NEW[old]
    short = module.__name__.rsplit('.', 1)[1]
    with pytest.warns(DeprecationWarning, match=f'use {short}.{new}$'):
        got = getattr(cli, old)
    assert got is getattr(module, new)
    assert old not in vars(cli)


def test_public_cli_names_stay():
    # Scripts call these on cli; cli no longer uses the last two itself.
    assert vars(cli)['build_fmo_thresholds'] is cli_trials.build_fmo_thresholds
    assert vars(cli)['run_group_umap'] is cli_plots.run_group_umap
    assert (vars(cli)['save_group_pair_scatters']
            is cli_plots.save_group_pair_scatters)


def test_cli_does_not_hold_its_own_copy_of_the_group_outputs():
    # Both modes call write_group_outputs through cli_compare, so patching
    # it (or what it calls) there reaches independent and concatenate runs.
    for module, new in OLD_TO_NEW.values():
        if module is cli_compare:
            assert new not in vars(cli)
    assert vars(cli_concatenate)['cli_compare'] is cli_compare
    assert vars(cli_independent)['cli_compare'] is cli_compare
    assert vars(cli_compare)['cli_parallel'] is cli_parallel


def test_cli_uses_the_moved_trial_setup():
    # Both modes call these objects, not copies.
    for name in ('trial_label', 'calibrate_bead_micron', 'already_done',
                 'build_fmo_thresholds_for_groups'):
        assert vars(cli_concatenate)[name] is getattr(cli_trials, name)
        assert vars(cli_independent)[name] is getattr(cli_trials, name)
    assert vars(cli_run)['run_independent'] is cli_independent.run_independent
    assert vars(cli_run)['run_concatenated'] is cli_concatenate.run_concatenated


def test_group_plots_job_pickles_by_its_new_home():
    # A spawned worker (Windows) imports the group-plots function by this path.
    import pickle
    fn = pickle.loads(pickle.dumps(cli_compare.group_plots_job))
    assert fn is cli_compare.group_plots_job
    assert fn.__module__ == 'openflo.cli_compare'


@pytest.mark.parametrize('mode', ['independent', 'concatenate'])
def test_both_modes_reach_a_patched_write_group_outputs(tmp_path, monkeypatch,
                                                        mode):
    seen = []
    monkeypatch.setattr(cli_parallel, 'run_sample_tasks',
                        lambda tasks, *a, **k: {})
    monkeypatch.setattr(cli_compare, 'write_group_outputs',
                        lambda groups, results, reused, out, *a, **k:
                        seen.append((mode, [g['name'] for g in groups], out))
                        or True)
    trial = tmp_path / 'trial_1'
    trial.mkdir()
    groups = [{'name': 'G1', 'fmo_set': '', 'samples': []}]
    run = (cli_independent.run_independent if mode == 'independent'
           else cli_concatenate.run_concatenated)
    assert run([str(trial)], str(tmp_path / 'out'), groups, {}, 30, 99.0, {},
               {}, None, False, 1, 1, None, 0.0, 0.0, None, 8.0, None,
               lambda *a, **k: None) is True
    assert seen == [(mode, ['G1'], str(tmp_path / 'out'))]
