"""openflo-run's per-sample work lives in openflo.cli_parallel; its old
private names on openflo.cli still resolve to the same objects, with a
DeprecationWarning that names the new one."""
from __future__ import annotations

import pytest

import openflo.cli as cli
from openflo import (
    cli_concatenate,
    cli_groups,
    cli_independent,
    cli_parallel,
    cli_plots,
)

OLD_TO_NEW = {
    '_load_task_sample': 'load_task_sample',
    '_process_sample_task': 'process_sample_task',
    '_write_sample_outputs': 'write_sample_outputs',
    '_run_sample_tasks': 'run_sample_tasks',
}


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    new = OLD_TO_NEW[old]
    with pytest.warns(DeprecationWarning, match=f'use cli_parallel.{new}'):
        got = getattr(cli, old)
    assert got is getattr(cli_parallel, new)
    assert old not in vars(cli)


def test_save_plots_stays_a_public_cli_name():
    # cli no longer calls it, but it is still the cli_plots object on cli.
    assert vars(cli)['save_plots'] is cli_plots.save_plots


def test_samples_are_named_and_drawn_as_the_run_does():
    assert vars(cli_parallel)['output_name'] is cli_groups.output_name
    assert vars(cli_parallel)['save_plots'] is cli_plots.save_plots


def test_cli_does_not_hold_its_own_copy_of_the_sample_work():
    # Both modes call these through cli_parallel, so patching them there
    # reaches both the per-sample processing and the --skip-existing reload.
    for new in OLD_TO_NEW.values():
        assert new not in vars(cli)
    assert vars(cli_concatenate)['cli_parallel'] is cli_parallel
    assert vars(cli_independent)['cli_parallel'] is cli_parallel


def test_worker_function_pickles_by_its_new_home():
    # A spawned worker (Windows) imports the task function by this path.
    import pickle
    fn = pickle.loads(pickle.dumps(cli_parallel.process_sample_task))
    assert fn is cli_parallel.process_sample_task
    assert fn.__module__ == 'openflo.cli_parallel'


def test_the_stats_line_keeps_the_cli_logger_name(tmp_path, monkeypatch,
                                                  caplog):
    class _Stats:
        def to_csv(self, path, index):
            open(path, 'w').close()

    class _Sample:
        def cluster_stats(self):
            return _Stats()
    monkeypatch.setattr(cli_parallel, 'save_plots',
                        lambda s, out_dir, **k: (tmp_path / 'G1').mkdir())
    with caplog.at_level('INFO', logger='openflo.cli'):
        cli_parallel.write_sample_outputs(
            _Sample(), {'group': 'G1', 'name': 'synthetic_1',
                        'out': str(tmp_path)})
    assert [r.name for r in caplog.records] == ['openflo.cli']
    assert 'Stats →' in caplog.records[0].getMessage()
    assert (tmp_path / 'G1' / 'synthetic_1_stats.csv').is_file()
