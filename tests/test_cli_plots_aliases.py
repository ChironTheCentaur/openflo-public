"""openflo-run's plots live in openflo.cli_plots; their old names on
openflo.cli still resolve to the same objects: the public ones as they were,
the private ones with a DeprecationWarning that names the new one."""
from __future__ import annotations

import os

import pytest

import openflo.cli as cli
from openflo import cli_compare, cli_groups, cli_plots, cli_run

OLD_TO_NEW = {
    '_resolve_pair': 'resolve_pair',
    '_intensity_axes': 'intensity_axes',
    '_show_saved_group_plots': 'show_saved_group_plots',
    '_KEEP_FINAL_PLOTS': 'KEEP_FINAL_PLOTS',
    '_STAGING_PREFIX': 'STAGING_PREFIX',
}

PUBLIC = ('DEFAULT_SCATTER_PAIRS', 'parse_pairs', 'run_group_umap',
          'save_group_pair_scatters', 'save_plots')


@pytest.mark.parametrize('old', sorted(OLD_TO_NEW))
def test_old_cli_name_is_the_new_object_with_a_warning(old):
    new = OLD_TO_NEW[old]
    with pytest.warns(DeprecationWarning, match=f'use cli_plots.{new}'):
        got = getattr(cli, old)
    assert got is getattr(cli_plots, new)
    assert old not in vars(cli)


@pytest.mark.parametrize('name', PUBLIC)
def test_public_cli_name_is_the_moved_object(name):
    # Plain attributes, no warning: scripts call them on cli. The sample
    # and group writers look them up on cli_parallel and cli_compare.
    assert vars(cli)[name] is getattr(cli_plots, name)


def test_plots_name_files_as_the_run_does():
    assert vars(cli_plots)['output_name'] is cli_groups.output_name



class _FakeSample:
    """Just enough of a FlowSample for run_group_umap to draw and save."""
    name = 'fake'
    scatter_channels = []

    def run_umap(self, **kw):
        pass

    def plot_umap(self, **kw):
        pass


@pytest.mark.parametrize('keep', [False, True])
def test_group_umap_keeps_its_figure_only_with_show_plots(keep, tmp_path,
                                                         monkeypatch):
    # The UMAP keeps its figure open for plt.show() only when --show-plots'
    # flag is on.
    import matplotlib.pyplot as plt
    monkeypatch.setattr(cli_plots, 'concatenate', lambda samples: None)
    monkeypatch.setattr(cli_plots, 'KEEP_FINAL_PLOTS', keep)
    plt.close('all')
    cli.run_group_umap([_FakeSample()], 'G1', str(tmp_path))
    assert (tmp_path / 'G1_umap.png').is_file()
    assert bool(plt.get_fignums()) is keep
    plt.close('all')


def test_reopening_skips_staging_folders(tmp_path, monkeypatch):
    # Only folders with the staging prefix are skipped: the comparison
    # publishing in cli_compare makes them with the same constant.
    import matplotlib.pyplot as plt
    import numpy as np
    monkeypatch.setattr(cli_plots, 'STAGING_PREFIX', '.synthetic-staging-')
    for folder in ('kept', '.synthetic-staging-1', '.compare-staging-1'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'G1_umap.png').write_bytes(b'')
    shown = []
    monkeypatch.setattr(plt, 'imread',
                        lambda p: shown.append(p) or np.zeros((2, 2, 3)))
    assert cli_plots.show_saved_group_plots(str(tmp_path)) == 2
    assert sorted(os.path.basename(os.path.dirname(p)) for p in shown) == [
        '.compare-staging-1', 'kept']
    plt.close('all')


def test_cli_compare_uses_the_one_staging_prefix():
    assert vars(cli_compare)['STAGING_PREFIX'] is cli_plots.STAGING_PREFIX


@pytest.mark.parametrize('mode', ['independent', 'concatenate'])
@pytest.mark.parametrize('show', [False, True])
def test_run_passes_show_plots_down_and_leaves_the_default(show, mode,
                                                           tmp_path,
                                                           monkeypatch):
    # run() hands --show-plots to the group plots as keep_open and never
    # sets the module default, so one run cannot change what the next run
    # (or a script drawing group plots itself) keeps open. It stops at the
    # batch step, which records what it was given.
    class Stop(Exception):
        pass
    seen = {}

    def stop(*a, **k):
        seen.update(k)
        raise Stop
    monkeypatch.setattr(cli_plots, 'KEEP_FINAL_PLOTS', 'untouched')
    monkeypatch.setattr(cli_run, 'normalise_groups', lambda g: g)
    monkeypatch.setattr(cli_run, 'group_spec_problems', lambda g: [])
    monkeypatch.setattr(cli_run, 'sample_file_problems', lambda g, d: [])
    monkeypatch.setattr(cli_run, 'run_independent', stop)
    monkeypatch.setattr(cli_run, 'run_concatenated', stop)
    groups = [{'name': 'G1', 'samples': ['synthetic_1']}]
    with pytest.raises(Stop):
        cli.run([str(tmp_path)], str(tmp_path / 'out'), groups,
                batch_mode=mode, show_plots=show)
    assert seen['keep_open'] is show
    assert cli_plots.KEEP_FINAL_PLOTS == 'untouched'


@pytest.mark.parametrize('default', [False, True])
@pytest.mark.parametrize('keep', [False, True])
def test_keep_open_wins_over_the_default(keep, default, tmp_path,
                                         monkeypatch):
    # A run's keep_open decides, whatever the module default says; the
    # default applies only when keep_open is left out.
    import matplotlib.pyplot as plt
    monkeypatch.setattr(cli_plots, 'concatenate', lambda samples: None)
    monkeypatch.setattr(cli_plots, 'KEEP_FINAL_PLOTS', default)
    plt.close('all')
    cli.run_group_umap([_FakeSample()], 'G1', str(tmp_path), keep_open=keep)
    assert bool(plt.get_fignums()) is keep
    plt.close('all')
    assert cli_plots.keeps_open(None) is default
    assert cli_plots.keeps_open(keep) is keep
